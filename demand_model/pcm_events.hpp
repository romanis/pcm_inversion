#ifndef __PCM_EVENTS_HPP__
#define __PCM_EVENTS_HPP__

#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include "pcm_aggregate.hpp"

/**
 * First frontier events along a direction, and frontier signatures
 * (docs/estimation_harness/DESIGN.md 3.3).
 *
 * Draw i's frontier F_i is the set of vertices of the increasing part of the upper convex hull of
 * {(0, 0)} U {(p_j, y_ij)}, y_ij = delta_j + x_j . t_i, excluding the origin. Collinear points are
 * not vertices, which matches pcm_geometry::index_positive_shares: a product is on F_i
 * exactly when it has positive conditional share. Membership does not depend on c_i or sigma_p
 * (DESIGN F5), so the scale and sigma_p components of a direction never cause an event. With a bound
 * lo > 0 (pcm_aggregate::Params) the hull is that of (p_j, y_ij - lo_y,i p_j) with lo_y,i the bound in y units:
 * a vertex has a window of positive mass exactly when its entry slope in y units exceeds lo_y,i, so F_i is the
 * set of products with positive conditional share under the bound. Under BoundKind::ALPHA lo_y,i = lo, so F5
 * still holds; under BoundKind::A lo_y,i = lo c_i, so membership depends on c_i (never on sigma_p) and the scale
 * component of a direction moves the shear, linearised in max_step as d lo_y,i = lo_y,i (scale_basis_i . dphi).
 *
 * Every function processes draws in fixed blocks of 64 and reduces in draw order, so results are
 * bit-identical for every thread count. n_threads <= 0 uses the OpenMP default team size. The
 * caller's OpenMP thread-count setting is not changed.
 */
namespace pcm_events {

struct StepEvents {
    double tau_first = 0.0;              //!< min over draws of tau_i
    double tau_vanish_lb = 0.0;          //!< min_j max_{i: j in F_i} tau_i (products with no buyer skipped)
    Eigen::ArrayXd tau_draw;             //!< R, first event per draw (tau_max if none)
    Eigen::ArrayXi kind_draw;            //!< R, 0 none, 1 product leaves, 2 product enters
    Eigen::ArrayXi product_draw;         //!< R, product involved in the first event (-1 if none)
    Eigen::MatrixXd mass_profile;        //!< n_grid x 2: (tau, max_j affected mass of j)
    Eigen::ArrayXd buyer_mass;           //!< N, B_j = sum_{i: j in F_i} w_i at tau = 0
};

/** Number of rows of StepEvents::mass_profile: tau_g = tau_max g / (n_grid - 1), g = 0..n_grid-1. */
constexpr int MASS_GRID = 64;

/**
 * Event times along delta(tau) = delta + tau delta_dot, theta(tau) = theta + tau dtheta.
 *
 * tau_i is the first tau in [0, tau_max) at which a hull certificate of draw i fails (C1-C4 of
 * DESIGN 3.3); tau_draw = tau_max and kind_draw = 0 when none fails. When two certificates of one
 * draw fail at the same tau, a leaving vertex (C1, C2) is reported before an entering product
 * (C3, C4), and within each group the cheaper product first.
 *
 * mass_profile(g, 1) = max_j A_j(tau_g), A_j(tau) = sum_{i: j in F_i, kind_i != 0, tau_i <= tau} w_i / B_j,
 * over products with B_j > 0. tau_vanish_lb is tau_max when no product has a buyer.
 *
 * @param delta_dot D dtheta from the current evaluation (linearisation of delta(theta)), size N
 * @param dtheta    direction in [taste (M), scale (Q), sigma_p (1)] order; only the taste part moves y
 * @param tau_max   positive and finite
 * @throws std::invalid_argument on inconsistent sizes, unsorted or non-positive prices, non-finite input
 */
StepEvents max_step(const Eigen::ArrayXd & delta, const Eigen::ArrayXd & delta_dot,
                    const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm,
                    const Eigen::ArrayXd & dtheta, double tau_max, int n_threads);

/** Hash of each draw's frontier F_i (the ordered product indices), one entry per draw. */
std::vector<uint64_t> frontier_signature(const Eigen::ArrayXd & delta, const pcm_aggregate::Market & mk,
                                         const pcm_aggregate::Params & prm, int n_threads);

/**
 * Realised affected mass for the guard of DESIGN 4.4: for each product j,
 * A_j = sum_{i: j in F_i, changed[i]} w_i / B_j with F_i and B_j taken at (delta, prm), 0 if B_j = 0.
 * changed[i] is typically frontier_signature at the trial point != frontier_signature here.
 */
Eigen::ArrayXd affected_mass(const Eigen::ArrayXd & delta, const pcm_aggregate::Market & mk,
                             const pcm_aggregate::Params & prm, const std::vector<uint8_t> & changed,
                             int n_threads);

}  // namespace pcm_events

#endif
