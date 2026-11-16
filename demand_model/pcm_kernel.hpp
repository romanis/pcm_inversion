#ifndef __PCM_KERNEL_HPP__
#define __PCM_KERNEL_HPP__

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "pcm_demand.hpp"

/**
 * Estimation harness kernel: frontier lift, inversion and derivatives of one market in one call
 * (docs/estimation_harness/DESIGN.md 3.1-3.2). WP0 declares the interface; WP1 implements it.
 */
namespace pcm_kernel {

/** Agent-level micro weights for pooled micro moments (DESIGN 2.3, 3.1). */
struct MicroRequest {
    Eigen::MatrixXd H;               //!< R x L agent weights h_l(i) (rows aligned with draws)
    Eigen::MatrixXd U;               //!< N x Pm product weights u_p = v_p - v0_p
    std::vector<int> h_of_part;      //!< Pm, column of H used by part p
};

enum class Status { OK, NOT_CONVERGED, INFEASIBLE_START, SINGULAR_DS_DDELTA, INVALID_INPUT };

struct SolveOptions {
    double tol = 1e-9;                                    //!< max |s - s_data|
    unsigned warm_evals = 2500, cold_evals = 40000, path_evals = 1500;
    double path_seconds = 90.0;                           //!< budget for the theta-path continuation
    int n_threads = 1;                                    //!< OpenMP threads for this call
    bool derivatives = true;                              //!< ds/ddelta, ds/dtheta, ddelta/dtheta
    bool price_derivatives = false;                       //!< ds/dp (postestimation)
    double rcond_min = 1e-13;                             //!< below this ds/ddelta is reported singular
};

struct SolveResult {
    Status status = Status::INVALID_INPUT;
    Eigen::ArrayXd delta, s;
    Eigen::MatrixXd ds_ddelta, ds_dtheta, ddelta_dtheta, ds_dp;   //!< N x N, N x P, N x P, N x N
    Eigen::VectorXd micro;                                        //!< Pm part sums sum_i w_i h(i) sum_j u_j s_ij
    Eigen::MatrixXd micro_d_delta;                                //!< Pm x N
    Eigen::MatrixXd micro_d_theta;                                //!< Pm x P
    double max_abs_residual = 0.0, rcond = 0.0, seconds = 0.0;
    unsigned share_evaluations = 0;
    std::string stages;                                           //!< "warm 3s; path tau=1 12s" etc.
};

/**
 * Inverts one market and returns shares, derivatives and micro parts at the converged delta.
 * P = M + Q + 1 in the column order of pcm_demand::Shares::d_theta: taste, scale, sigma_p.
 * @param delta_warm empty: no warm start
 * @param prm_warm   theta at which delta_warm was solved, or nullptr
 * @param micro      nullptr: no micro parts
 */
SolveResult solve_market(const Eigen::ArrayXd & s_data, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, const Eigen::ArrayXd & delta_warm,
                         const pcm_demand::Params * prm_warm, const MicroRequest * micro,
                         const SolveOptions & opt);

}  // namespace pcm_kernel

#endif
