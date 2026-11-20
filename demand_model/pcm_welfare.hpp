#ifndef __PCM_WELFARE_HPP__
#define __PCM_WELFARE_HPP__

#include <Eigen/Dense>
#include "pcm_aggregate.hpp"

/**
 * Consumer surplus of the exact pure characteristics model by the frontier formula
 * (docs/estimation_harness/DESIGN.md 3.4).
 *
 * Draw i has conditional qualities q_k = (delta_k + x_k . t_i) / c_i and price sensitivity
 * c_i a_i with a_i ~ LogNormal(0, sigma_p). Its frontier is the increasing part of the upper
 * convex hull of {(0, 0)} U {(p_k, q_k)}; frontier product k is bought for a in (e_{k+1}, e_k),
 * where e are the hull slopes (e_0 = infinity). Surplus in price units is
 *     CS_i = sum_k [ q_k E(1/a; e_{k+1} < a < e_k) - p_k P(e_{k+1} < a < e_k) ],
 *     E(1/a; lo < a < hi) = exp(s^2/2) [Phi((ln hi + s^2)/s) - Phi((ln lo + s^2)/s)].
 * The utility scale cancels (LEARNINGS 11), so CS is invariant to (delta, theta, c) -> kappa (...).
 * With a bound lo > 0 (pcm_aggregate::Params), a_i is truncated below at a_lo,i = draw_a_lo(prm, c_i) (lo / c_i
 * for BoundKind::ALPHA, lo for A): P and E become the truncated F_T and E_T(1/a) of the window ends clipped at
 * a_lo,i, so the last window ends at a_lo,i and CS_i <= max_k q_k / a_lo,i - p_k (LEARNINGS 29). Under ALPHA
 * the bound scales with the utility, so the invariance holds with lo -> kappa lo; under A it holds with
 * c -> kappa c and lo fixed.
 */
namespace pcm_welfare {

struct Surplus {
    double total = 0.0;                  //!< sum_i w_i CS_i, in price units per household
    Eigen::ArrayXd per_draw;             //!< R, CS_i of each draw (not multiplied by its weight)
};

/**
 * Consumer surplus of one market at delta.
 * Prices must be positive and strictly increasing, sigma_p > 0; dimension mismatches throw
 * std::invalid_argument. The result is bit-identical for every n_threads.
 */
Surplus consumer_surplus(const Eigen::ArrayXd & delta, const pcm_aggregate::Market & mk,
                         const pcm_aggregate::Params & prm, int n_threads);

}  // namespace pcm_welfare

#endif
