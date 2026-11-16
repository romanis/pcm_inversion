#ifndef __PCM_WELFARE_HPP__
#define __PCM_WELFARE_HPP__

#include <Eigen/Dense>
#include "pcm_demand.hpp"

/**
 * Consumer surplus by the frontier formula (docs/estimation_harness/DESIGN.md 3.4).
 * WP0 declares the interface; WP3 implements it.
 */
namespace pcm_welfare {

struct Surplus {
    double total = 0.0;                  //!< in price units per household
    Eigen::ArrayXd per_draw;             //!< R
};

Surplus consumer_surplus(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, int n_threads);

}  // namespace pcm_welfare

#endif
