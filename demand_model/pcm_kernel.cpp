#include "pcm_kernel.hpp"

#include <stdexcept>

namespace pcm_kernel {

SolveResult solve_market(const Eigen::ArrayXd & s_data, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, const Eigen::ArrayXd & delta_warm,
                         const pcm_demand::Params * prm_warm, const MicroRequest * micro,
                         const SolveOptions & opt) {
    throw std::logic_error("pcm_kernel::solve_market is not implemented yet (WP1, DESIGN 3.2)");
}

}  // namespace pcm_kernel
