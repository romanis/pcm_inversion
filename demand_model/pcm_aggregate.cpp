#include "pcm_aggregate.hpp"
#include "pcm_geometry.hpp"

#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
#include <limits>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pcm_aggregate {

namespace {
thread_local ThreadBudget * current_budget = nullptr;
}  // namespace

ThreadBudgetScope::ThreadBudgetScope(ThreadBudget * budget) : saved_(current_budget) { current_budget = budget; }
ThreadBudgetScope::~ThreadBudgetScope() { current_budget = saved_; }

void apply_thread_budget()
{
    if (!current_budget) return;
    const int n = current_budget->load(std::memory_order_relaxed);
    if (n >= 1 && n != omp_get_max_threads()) omp_set_num_threads(n);
}

// For every product, the smallest increase in delta_j that puts it on at least one consumer's
// frontier. For draw i the frontier is the increasing part of the upper convex hull of the points
// (p_k, dc_ik) together with the origin; a product below the hull by g (in conditional-quality units)
// needs delta_j raised by c_i * g for that consumer. The lift is the minimum over draws. Products
// already on some frontier get zero. Cost: one hull per draw, about one share evaluation.
ArrayXd frontier_lift(const ArrayXd & delta, const Market & mk, const Params & prm)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const double inf = std::numeric_limits<double>::infinity();
    ArrayXd lift = ArrayXd::Constant(N, inf);
    for_each_draw_block<ArrayXd>(R, [&](ArrayXd & acc, int i0, int i1) {
        acc = ArrayXd::Constant(N, inf);
        VectorXd t(K);
        ArrayXd dc(N), gap(N);
        std::vector<int> hull;
        hull.reserve(N + 1);
        for (int i = i0; i < i1; ++i) {
            const double c = conditional_quality(delta, mk, prm, i, t, dc);
            // With a bound, the hull of the sheared qualities dc - a_lo,i p: its increasing part holds exactly
            // the vertices with entry slope above a_lo,i, the products with a window of positive mass. The
            // vertical gap, and so the lift, is the same in both coordinates.
            if (prm.lo > 0.0) dc -= draw_a_lo(prm, c) * mk.p;
            pcm_geometry::frontier_gap(dc, mk.p, hull, gap);
            for (int k = 0; k < N; ++k) acc[k] = std::min(acc[k], c * gap[k]);
        }
    }, [&](const ArrayXd & acc) { lift = lift.min(acc); });
    return lift;
}

Shares shares(const ArrayXd & delta, const Market & mk, const Params & prm, bool with_derivatives)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    if (mk.x.rows() != N || mk.p.size() != N)
        throw std::runtime_error("pcm_aggregate::shares: x/p/delta sizes differ");
    if (mk.taste_basis.rows() != R || mk.taste_basis.cols() != M || (int)mk.taste_col.size() != M)
        throw std::runtime_error("pcm_aggregate::shares: taste_basis must be R x M and taste_col size M");
    if (Q > 0 && (mk.scale_basis.rows() != R || mk.scale_basis.cols() != Q))
        throw std::runtime_error("pcm_aggregate::shares: scale_basis must be R x Q");
    for (int c : mk.taste_col)
        if (c < 0 || c >= K) throw std::runtime_error("pcm_aggregate::shares: taste_col out of range");
    for (int j = 1; j < N; ++j)
        if (!(mk.p[j] > mk.p[j - 1])) throw std::runtime_error("pcm_aggregate::shares: prices must be strictly increasing");
    if (N > 0 && !(mk.p[0] > 0)) throw std::runtime_error("pcm_aggregate::shares: prices must be positive");
    if (!(prm.sigma_p > 0)) throw std::runtime_error("pcm_aggregate::shares: sigma_p must be positive");
    if (!(prm.lo >= 0) || !std::isfinite(prm.lo))
        throw std::runtime_error("pcm_aggregate::shares: the bound lo must be finite and non-negative");

    Shares out;
    out.s = ArrayXd::Zero(N);
    if (with_derivatives) {
        out.d_delta = MatrixXd::Zero(N, N);
        out.d_theta = MatrixXd::Zero(N, M + Q + 1);
        out.d_price = MatrixXd::Zero(N, N);
    }

    const bool scale_bound = bound_moves_with_scale(prm);

    // Draws are independent: fixed blocks of 64 draws each accumulate partial sums in parallel,
    // and the blocks are added in index order, so results are bit-identical for any thread count.
    for_each_draw_block<Shares>(R, [&](Shares & acc, int i0, int i1) {
        acc.s = ArrayXd::Zero(N);
        if (with_derivatives) {
            acc.d_delta = MatrixXd::Zero(N, N);
            acc.d_theta = MatrixXd::Zero(N, M + Q + 1);
            acc.d_price = MatrixXd::Zero(N, N);
        }
        ArrayXd s_c(N), dsig(N), dlogc(N), dc(N);
        MatrixXd Jd(N, N), Jp(N, N);
        VectorXd t(K);

        for (int i = i0; i < i1; ++i) {
            const double c = conditional_quality(delta, mk, prm, i, t, dc);

            s_c.setZero();
            if (with_derivatives) { Jd.setZero(); Jp.setZero(); dsig.setZero(); dlogc.setZero(); }
            pcm_geometry::cond_share(dc, mk.p, prm.sigma_p, draw_a_lo(prm, c), s_c,
                                     with_derivatives ? &Jd : nullptr, &dsig, &Jp, scale_bound ? &dlogc : nullptr);

            const double w = mk.weights[i];
            acc.s += w * s_c;
            if (!with_derivatives) continue;

            acc.d_delta.noalias() += (w / c) * Jd;
            acc.d_price.noalias() += w * Jp;
            for (int m = 0; m < M; ++m)
                acc.d_theta.col(m).noalias() += (w * mk.taste_basis(i, m) / c) * (Jd * mk.x.col(mk.taste_col[m]));
            if (Q > 0) {
                // log c moves s through dc = y / c (-Jd dc) and, with an ALPHA bound, through a_lo = lo / c
                VectorXd Jdc = Jd * dc.matrix();
                if (scale_bound) Jdc -= dlogc.matrix();
                for (int q = 0; q < Q; ++q)
                    acc.d_theta.col(M + q).noalias() += (-w * mk.scale_basis(i, q)) * Jdc;
            }
            acc.d_theta.col(M + Q) += w * dsig.matrix();
        }
    }, [&](const Shares & acc) {
        out.s += acc.s;
        if (with_derivatives) {
            out.d_delta += acc.d_delta;
            out.d_theta += acc.d_theta;
            out.d_price += acc.d_price;
        }
    });
    return out;
}

}  // namespace pcm_aggregate
