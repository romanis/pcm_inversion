#ifndef __PCM_AGGREGATE_HPP__
#define __PCM_AGGREGATE_HPP__

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>
#include <omp.h>
#include <Eigen/Dense>

/**
 * Layer 1 of the PCM kernel: integration of per-draw functionals over the draws of one market
 * (docs/estimation_harness/DESIGN.md 3.0). The model:
 *
 * Consumer i (a draw with weight w_i) gets utility from product j
 *     u_ij = delta_j + x_j . t_i - c_i * a_i * p_j,     u_i0 = 0,
 * with a_i ~ LogNormal(0, sigma_p) integrated in closed form. With Params::lo > 0 the price sensitivity
 * is bounded below, in one of two forms (Params::bound_kind):
 *   ALPHA  a_i is truncated below at a_lo,i = lo / c_i, so alpha_i = c_i a_i >= lo (LEARNINGS 29);
 *   A      a_i is truncated below at a_lo = lo for every draw, so alpha_i >= lo c_i (LEARNINGS 32): the
 *          bound on alpha moves with the draw's price scale, and log c_i no longer acts through it.
 * With Q = 0 (c_i = 1) the two coincide. The median normalisation stays on the untruncated parent.
 * lo = 0 is the untruncated model, computed by the same code as before the bound existed.
 *   t_i (taste vector, size K) = sum_m theta_m * taste_basis(i, m) on column taste_col[m]
 *   log c_i (price-sensitivity scale) = sum_q phi_q * scale_basis(i, q)
 * Dividing utility by c_i gives a pure vertical model in "dollar" qualities
 *     delta_cond_ij = (delta_j + x_j . t_i) / c_i
 * which is what pcm_geometry::cond_share solves for one draw (layer 0).
 *
 * The caller builds t_i and log c_i from its own parameters; this module only needs the
 * bases to return derivatives with respect to every parameter. It knows draws, weights and the
 * block reduction; it does not know data shares, inversion or moments.
 */
namespace pcm_aggregate {

struct Market {
    Eigen::MatrixXd x;               //!< N x K product characteristics
    Eigen::ArrayXd  p;               //!< N prices, strictly increasing
    Eigen::ArrayXd  weights;         //!< R draw weights
    Eigen::MatrixXd taste_basis;     //!< R x M, d t_i[taste_col[m]] / d theta_m
    std::vector<int> taste_col;      //!< M, characteristic column each taste parameter loads on
    Eigen::MatrixXd scale_basis;     //!< R x Q, d log c_i / d phi_q
};

/** Which factor of alpha_i = c_i a_i the lower bound Params::lo applies to. */
enum class BoundKind { ALPHA, A };

struct Params {
    Eigen::ArrayXd theta;            //!< M taste parameters
    Eigen::ArrayXd phi;              //!< Q price-scale parameters
    double sigma_p;                  //!< sd of log price sensitivity
    double lo = 0.0;                 //!< lower bound (0: none) on alpha_i (ALPHA) or a_i (A); fixed, not estimated
    BoundKind bound_kind = BoundKind::ALPHA;
};

/** The truncation point of a_i for a draw with price scale c: lo / c (ALPHA) or lo (A); 0 without a bound. */
inline double draw_a_lo(const Params & prm, double c) {
    return prm.bound_kind == BoundKind::A ? prm.lo : prm.lo / c;
}

/** True when the truncation point depends on c_i, so log c acts on shares through it (ALPHA with lo > 0). */
inline bool bound_moves_with_scale(const Params & prm) {
    return prm.lo > 0.0 && prm.bound_kind == BoundKind::ALPHA;
}

/** "alpha" or "a". */
inline const char * bound_kind_name(BoundKind k) { return k == BoundKind::A ? "a" : "alpha"; }

/**
 * A team size another thread may change while a call runs: the calling thread installs it with
 * ThreadBudgetScope, and for_each_draw_block (so shares(), frontier_lift() and evaluate_market) sets the
 * calling thread's OpenMP team size from it at its start. A market that finishes can so hand its threads to
 * the markets still running, between two share evaluations of theirs. Results do not depend on it:
 * every draw loop sums fixed 64-draw blocks in block order.
 */
using ThreadBudget = std::atomic<int>;

/** Installs `budget` (nullptr: none) for the calling thread and restores the previous one on scope exit. */
class ThreadBudgetScope {
public:
    explicit ThreadBudgetScope(ThreadBudget * budget);
    ~ThreadBudgetScope();
    ThreadBudgetScope(const ThreadBudgetScope &) = delete;
    ThreadBudgetScope & operator=(const ThreadBudgetScope &) = delete;
private:
    ThreadBudget * saved_;
};

/** Sets the calling thread's OpenMP team size from its installed budget, if it has one with a value >= 1. */
void apply_thread_budget();

struct Shares {
    Eigen::ArrayXd  s;               //!< N predicted shares
    Eigen::MatrixXd d_delta;         //!< N x N, ds/d delta
    Eigen::MatrixXd d_theta;         //!< N x (M + Q + 1), ds/d[theta, phi, sigma_p]
    Eigen::MatrixXd d_price;         //!< N x N, ds/dp
};

/**
 * Shares and (optionally) all derivatives for one market.
 * @param with_derivatives if false only s is filled
 */
Shares shares(const Eigen::ArrayXd & delta, const Market & m, const Params & prm, bool with_derivatives);

/**
 * For every product, the smallest increase in delta_j that puts it on at least one consumer's
 * frontier with a window of positive mass (zero for products already there, +inf if none is
 * reachable). With lo > 0 the hull is that of (p, dc - a_lo,i p), a_lo,i = draw_a_lo(prm, c_i), whose
 * increasing part holds exactly the vertices with entry slope above a_lo,i (LEARNINGS 29).
 */
Eigen::ArrayXd frontier_lift(const Eigen::ArrayXd & delta, const Market & mk, const Params & prm);

/** Draws per block of every draw loop. Part of the numerical contract: it fixes the summation order. */
constexpr int DRAW_BLOCK = 64;

/** The fixed partition of R draws into n = ceil(R / 64) blocks (at least one): block b holds draws
 *  [begin(b), end(b)), begin(b) = floor(R b / n). It does not depend on the thread count. */
struct DrawBlocks {
    int R, n;
    explicit DrawBlocks(int R_) : R(R_), n(std::max(1, (R_ + DRAW_BLOCK - 1) / DRAW_BLOCK)) {}
    int begin(int b) const { return static_cast<int>((static_cast<long>(R) * b) / n); }
    int end(int b) const { return begin(b + 1); }
};

/** How a draw loop runs. */
struct DrawLoop {
    int first = 0;       //!< the loop covers draws [first, first + R)
    /** > 0: OpenMP team of that size; 0: the calling thread's team after its ThreadBudget (apply_thread_budget);
     *  < 0: serial, for a loop that already runs inside a parallel region. */
    int n_threads = 0;
    int wave = 0;        //!< > 0: at most this many block accumulators held at once (memory); 0: all blocks
};

/**
 * The one draw loop of the kernel. Splits draws [first, first + R) into DrawBlocks(R), runs
 * body(acc, i0, i1) for every block in parallel (dynamic schedule, one accumulator per block, which body
 * initialises), then calls reduce(acc) serially in block order. Results are therefore bit-identical for every
 * thread count. Per-draw outputs written by index need no reduce: pass a no-op.
 */
template <class Acc, class Body, class Reduce>
void for_each_draw_block(int R, Body && body, Reduce && reduce, const DrawLoop & loop = DrawLoop())
{
    const DrawBlocks blocks(R);
    if (loop.n_threads == 0) apply_thread_budget();
    const int team = loop.n_threads > 0 ? loop.n_threads : omp_get_max_threads();
    const int held = loop.wave > 0 ? std::min(blocks.n, loop.wave) : blocks.n;
    std::vector<Acc> part(held);
    for (int w0 = 0; w0 < blocks.n; w0 += held) {
        const int w1 = std::min(blocks.n, w0 + held);
#pragma omp parallel for schedule(dynamic) num_threads(team) if(loop.n_threads >= 0 && w1 - w0 > 1)
        for (int b = w0; b < w1; ++b)
            body(part[b - w0], loop.first + blocks.begin(b), loop.first + blocks.end(b));
        for (int b = w0; b < w1; ++b) reduce(part[b - w0]);
    }
}

/** No accumulator: for draw loops that write per-draw outputs by index. */
struct NoAcc {};

/**
 * Draw i's conditional qualities dc = (delta + x t_i) / c_i and price scale c_i = exp(phi . scale_basis_i), with
 * t_i = sum_m theta_m taste_basis(i, m) on column taste_col[m] (``t`` is scratch of size K). Returns c_i.
 */
inline double conditional_quality(const Eigen::ArrayXd & delta, const Market & mk, const Params & prm, int i,
                                  Eigen::VectorXd & t, Eigen::ArrayXd & dc)
{
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    t.setZero();
    for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
    const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
    const double c = std::exp(log_c);
    dc = (delta + (mk.x * t).array()) / c;
    return c;
}

}  // namespace pcm_aggregate

#endif
