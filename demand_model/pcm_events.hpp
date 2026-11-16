#ifndef __PCM_EVENTS_HPP__
#define __PCM_EVENTS_HPP__

#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include "pcm_demand.hpp"

/**
 * First frontier events along a direction, and frontier signatures
 * (docs/estimation_harness/DESIGN.md 3.3). WP0 declares the interface; WP2 implements it.
 */
namespace pcm_events {

struct StepEvents {
    double tau_first = 0.0;              //!< min over draws of tau_i
    double tau_vanish_lb = 0.0;          //!< min_j max_{i: j in F_i} tau_i
    Eigen::ArrayXd tau_draw;             //!< R, first event per draw (tau_max if none)
    Eigen::ArrayXi kind_draw;            //!< R, 0 none, 1 product leaves, 2 product enters
    Eigen::ArrayXi product_draw;         //!< R, product involved in the first event
    Eigen::MatrixXd mass_profile;        //!< n_grid x 2: (tau, max_j affected mass of j)
};

/**
 * Event times along delta(tau) = delta + tau delta_dot, theta(tau) = theta + tau dtheta.
 * @param delta_dot D dtheta from the current evaluation (linearisation of delta(theta))
 * @param dtheta    direction in [taste, scale, sigma_p] order; only the taste part moves frontiers
 */
StepEvents max_step(const Eigen::ArrayXd & delta, const Eigen::ArrayXd & delta_dot,
                    const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                    const Eigen::ArrayXd & dtheta, double tau_max, int n_threads);

/** Hash of each draw's frontier F_i, one entry per draw. */
std::vector<uint64_t> frontier_signature(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                                         const pcm_demand::Params & prm, int n_threads);

}  // namespace pcm_events

#endif
