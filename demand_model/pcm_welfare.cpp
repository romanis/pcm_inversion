#include "pcm_welfare.hpp"

#include <stdexcept>

namespace pcm_welfare {

Surplus consumer_surplus(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, int n_threads) {
    throw std::logic_error("pcm_welfare::consumer_surplus is not implemented yet (WP3, DESIGN 3.4)");
}

}  // namespace pcm_welfare
