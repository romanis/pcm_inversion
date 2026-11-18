#ifndef __PCM_DEMAND_HPP__
#define __PCM_DEMAND_HPP__

#include <vector>
#include <Eigen/Dense>

/**
 * Pure characteristics demand with observed demographics, for one market.
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
 * which is what pcm_share::cond_share already solves.
 *
 * The caller builds t_i and log c_i from its own parameters; this module only needs the
 * bases to return derivatives with respect to every parameter.
 */
namespace pcm_demand {

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

struct InversionResult {
    Eigen::ArrayXd delta;
    double max_abs_residual;
    unsigned iterations;
    unsigned share_evaluations;
    bool converged;
};

/**
 * Finds delta with shares(delta) == s_data by Newton with an Armijo line search on
 * 0.5*||s - s_data||^2, kept inside the region where every share is positive and the
 * inside shares sum below one; Levenberg-Marquardt damping when Newton gives no descent.
 * @param delta0 starting point; ignored (replaced by the vertical-model guess) if empty or infeasible
 */
InversionResult invert(const Eigen::ArrayXd & s_data, const Market & m, const Params & prm,
                       const Eigen::ArrayXd & delta0, double tol = 1e-12,
                       unsigned max_share_evaluations = 5000);

/**
 * Per-draw building blocks, exposed for the estimation harness kernel (pcm_kernel.cpp).
 * Not part of the stable interface of this module.
 */
namespace detail {

/**
 * Vertical model for one draw in conditional qualities d (N), prices p (N, increasing) and
 * LogNormal(0, sigma) price sensitivity truncated below at a_lo (a_lo <= 0: not truncated). Writes
 * into global-index outputs, which must be zeroed: s (N) shares, and when Jd is not null also Jd
 * (N x N) ds/dd, dsig (N) ds/dsigma, Jp (N x N) ds/dp and, when dlogc is not null, dlogc (N) the
 * derivative of s through a_lo = lo / c with respect to log c at fixed d (the d = y / c part is
 * the caller's -Jd d term; only meaningful for BoundKind::ALPHA, where a_lo = lo / c). Products off the
 * frontier, and frontier products whose window lies entirely below a_lo, get zero share and zero derivatives.
 */
void cond_vertical(const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma, double a_lo,
                   Eigen::ArrayXd & s, Eigen::MatrixXd * Jd, Eigen::ArrayXd * dsig, Eigen::MatrixXd * Jp,
                   Eigen::ArrayXd * dlogc);

/**
 * For every product, the smallest increase in delta_j that puts it on at least one consumer's
 * frontier with a window of positive mass (zero for products already there, +inf if none is
 * reachable). With lo > 0 the hull is that of (p, dc - a_lo,i p), a_lo,i = draw_a_lo(prm, c_i), whose
 * increasing part holds exactly the vertices with entry slope above a_lo,i (LEARNINGS 29).
 */
Eigen::ArrayXd frontier_lift(const Eigen::ArrayXd & delta, const Market & mk, const Params & prm);

/**
 * The vertical-model starting point at prm.sigma_p and prm.lo (c = 1, no taste heterogeneity, where both
 * bound kinds coincide), which inverts the shares exactly in that model: F_T(e_k) = sum_{j >= k} s_j.
 * lo = 0 is pcm_share::initial_guess.
 */
Eigen::ArrayXd initial_guess(const Eigen::ArrayXd & s_data, const Eigen::ArrayXd & p, const Params & prm);

/** Window anatomy of one draw's frontier (LEARNINGS 32). */
struct DrawAnatomy {
    int frontier_size = 0;           //!< products with a window of positive mass
    double mass = 0.0;               //!< sum of those windows' masses (the draw's inside share)
    double log_range = 0.0;          //!< log(e_0 / a_lo), e_0 the cutoff of the cheapest frontier product; 0 if mass = 0
    double mass_log_width = 0.0;     //!< sum_k m_k log(e_k / max(e_{k+1}, a_lo)) over positive-mass windows
};

/**
 * Anatomy of the vertical model of one draw in conditional qualities d, prices p, LogNormal(0, sigma)
 * truncated below at a_lo. Without a bound (a_lo <= 0) the range and the cheapest window are infinite.
 */
DrawAnatomy draw_anatomy(const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma, double a_lo);

}  // namespace detail

}  // namespace pcm_demand

#endif
