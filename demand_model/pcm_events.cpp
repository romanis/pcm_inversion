#include "pcm_events.hpp"

#include <stdexcept>

namespace pcm_events {

StepEvents max_step(const Eigen::ArrayXd & delta, const Eigen::ArrayXd & delta_dot,
                    const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                    const Eigen::ArrayXd & dtheta, double tau_max, int n_threads) {
    throw std::logic_error("pcm_events::max_step is not implemented yet (WP2, DESIGN 3.3)");
}

std::vector<uint64_t> frontier_signature(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                                         const pcm_demand::Params & prm, int n_threads) {
    throw std::logic_error("pcm_events::frontier_signature is not implemented yet (WP2, DESIGN 3.3)");
}

}  // namespace pcm_events
