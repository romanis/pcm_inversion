#ifndef __PCM_GROUPS_HPP__
#define __PCM_GROUPS_HPP__

#include <vector>
#include <Eigen/Dense>
#include "pcm_demand.hpp"

/**
 * Per-household purchase-weighted sums for GMY's micro moments (GMY spec WP-G5).
 *
 * For each group g (a household: its draws are consecutive), with s_ij the exact PCM choice probabilities and
 * U' = [1, U]:  S(g, c) = sum_{i in g} w_i sum_j U'_jc s_ij, so column 0 is the purchase probability P_g and the
 * others the purchase-weighted characteristic sums. Exactly what evaluate_market returns for a micro part with
 * h = 1 and product weights U'_c on a market made of group g's draws alone, accumulated per group instead of per
 * part, so memory is G x (C+1) x N rather than (G (C+1)) x N per draw block.
 */
namespace pcm_kernel {

struct GroupRequest {
    Eigen::ArrayXi group_of_draw;    //!< R, 0..n_groups-1, non-decreasing (a household's draws are consecutive)
    int n_groups = 0;                //!< G
    Eigen::MatrixXd U;               //!< N x C product columns
};

struct GroupSums {
    Eigen::MatrixXd S;                       //!< G x (C+1): column 0 = P_g, columns 1..C = S_g[c]
    Eigen::ArrayXd w;                        //!< G, sum of the draw weights of each group
    std::vector<Eigen::MatrixXd> d_delta;    //!< G entries, (C+1) x N partials in delta (empty without derivatives)
    std::vector<Eigen::MatrixXd> d_theta;    //!< G entries, (C+1) x P partials in [theta, phi, sigma_p]
};

/**
 * The group sums at a fixed delta. Within a group the draws are summed in the block partition and order of
 * evaluate_market (blocks of 64 draws, partial sums added in block order), so each group's rows equal
 * evaluate_market on that group's draws bit for bit; groups run in parallel (one thread owns a group), so results
 * do not depend on the thread count. Uses the calling thread's OpenMP team size. Throws std::invalid_argument on
 * bad input (sizes, a decreasing or out-of-range group_of_draw).
 */
GroupSums group_sums(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                     const GroupRequest & gr, bool derivatives);

}  // namespace pcm_kernel

#endif
