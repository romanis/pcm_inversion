#ifndef __PCM_SECOND_CHOICE_HPP__
#define __PCM_SECOND_CHOICE_HPP__

#include <vector>
#include <Eigen/Dense>
#include "pcm_aggregate.hpp"

/**
 * Exact second choices of the pure characteristics model (GMY spec WP-G6).
 *
 * For draw i (conditional qualities d = (delta + x t_i) / c_i, price sensitivity a ~ LogNormal(0, sigma_p) truncated at
 * a_lo,i) the first choice is the frontier product whose window holds a (pcm_geometry::cond_share). Removing the first
 * choice j = f_r changes the frontier only between its neighbours L = f_{r-1} (the origin for the cheapest) and
 * R = f_{r+1} (none for the dearest): the reduced chain L, g_1..g_q, R is the upper hull of L, the products priced
 * between them, and R (without R, the increasing hull of the products dearer than L). For a in j's window the second
 * choice is the reduced-chain vertex whose window holds a; with L the origin, a above the first reduced cutoff buys
 * nothing and that mass is dropped (GMY conditions on an inside second choice, per household). With m_ijk the masses,
 * M_ij = sum_k m_ijk and q_ijk = m_ijk / M_ij (0 when M_ij = 0):
 *     N[j, k] = sum_i w_i s_ij q_ijk,   s_j = sum_i w_i s_ij,   C[j, k] = N[j, k] / s_j.
 */
namespace pcm_kernel {

/** Weights of the adjoint contraction: A'_c[j, k] (K entries, each N x N) and B_c[j] (K x N). */
struct SecondChoiceAdjoint {
    std::vector<Eigen::MatrixXd> A;
    Eigen::MatrixXd B;
};

struct SecondChoice {
    Eigen::MatrixXd C;             //!< N x N, rows of products nobody buys are zero
    Eigen::ArrayXd s;              //!< N model shares (the denominators)
    Eigen::MatrixXd N;             //!< N x N joint masses sum_i w_i s_ij q_ijk
    Eigen::MatrixXd out_delta;     //!< K x N: sum_jk A'_c[j,k] dN_jk/d delta - sum_j B_c[j] ds_j/d delta (empty without adjoint)
    Eigen::MatrixXd out_theta;     //!< K x P, columns [theta, phi, sigma_p]
    double seconds = 0.0;
    unsigned draws_with_no_inside_second = 0;   //!< (draw, j) pairs with s_ij > 0 and M_ij = 0
};

/**
 * Second choices of one market at a fixed delta, with the adjoint contraction when ``adj`` is not null. Draws are
 * processed in blocks of 64 whose partial sums are added in block order, so results do not depend on the thread
 * count. Uses the calling thread's OpenMP team size. Throws std::invalid_argument on bad input.
 */
SecondChoice second_choice(const Eigen::ArrayXd & delta, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm,
                           const SecondChoiceAdjoint * adj);

}  // namespace pcm_kernel

#endif
