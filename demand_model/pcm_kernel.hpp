#ifndef __PCM_KERNEL_HPP__
#define __PCM_KERNEL_HPP__

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "pcm_demand.hpp"

/**
 * Estimation harness kernel: frontier lift, inversion and derivatives of one market in one call
 * (docs/estimation_harness/DESIGN.md 3.1-3.2).
 */
namespace pcm_kernel {

/** Agent-level micro weights for pooled micro moments (DESIGN 2.3, 3.1). */
struct MicroRequest {
    Eigen::MatrixXd H;               //!< R x L agent weights h_l(i) (rows aligned with draws)
    Eigen::MatrixXd U;               //!< N x Pm product weights u_p = v_p - v0_p
    std::vector<int> h_of_part;      //!< Pm, column of H used by part p
};

enum class Status { OK, NOT_CONVERGED, INFEASIBLE_START, SINGULAR_DS_DDELTA, INVALID_INPUT };

/** Lower-case name of a status ("ok", "not_converged", "infeasible_start", "singular_ds_ddelta", "invalid_input"). */
const char * status_name(Status s);

struct SolveOptions {
    double tol = 1e-9;                                    //!< max |s - s_data|
    unsigned warm_evals = 2500, cold_evals = 40000, path_evals = 1500;
    double path_seconds = 90.0;                           //!< budget for the theta-path continuation
    int n_threads = 1;                                    //!< OpenMP threads for this call
    bool derivatives = true;                              //!< ds/ddelta, ds/dtheta, ddelta/dtheta
    bool price_derivatives = false;                       //!< ds/dp (postestimation)
    double rcond_min = 1e-13;                             //!< below this ds/ddelta is reported singular
    /**
     * Draw homotopy of the cold stage (0: off, the cold stage as before). With cold_draws > 0 the cold stage first
     * inverts on every k-th draw, k = cold_draw_factor^L with L the largest level leaving at least cold_draws draws,
     * then warm-starts each finer level (k / cold_draw_factor, ..., 1) from the coarser solution with warm_evals. A
     * share evaluation on R / k draws costs 1 / k of a full one, and the cold path needs thousands of evaluations
     * where a warm start from a coarser solution needs tens. If a level fails, the plain cold stage runs.
     */
    unsigned cold_draws = 0;
    int cold_draw_factor = 8;                             //!< thinning factor between the levels (at least 2)
    /**
     * Optional team size that another thread may change during the call (pcm_demand::ThreadBudget): read before
     * every share evaluation, so it overrides n_threads from then on. Results are the same for every value.
     */
    pcm_demand::ThreadBudget * thread_budget = nullptr;
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
    std::string stages;                                           //!< "warm ok 0.01s 4ev; derivatives 0.02s" etc.
};

/**
 * Inverts one market and returns shares, derivatives and micro parts at the converged delta.
 * P = M + Q + 1 in the column order of pcm_demand::Shares::d_theta: taste, scale, sigma_p.
 *
 * Stages: warm (from delta_warm at prm), theta path (from prm_warm to prm, only if the warm stage
 * failed), draws (the draw homotopy, only with opt.cold_draws > 0 and no converged stage yet), cold
 * (vertical-model guess). s and micro are evaluated at the returned delta in every
 * status except INVALID_INPUT. The derivative fields (ds_ddelta, ds_dtheta, ddelta_dtheta, ds_dp,
 * micro_d_delta, micro_d_theta) are filled only when the inversion converged and they were
 * requested; otherwise they are empty. With status SINGULAR_DS_DDELTA, ddelta_dtheta is NaN and the
 * other derivatives are filled. Never throws for bad input: that is status INVALID_INPUT with the
 * reason in `stages`. Sets the calling thread's OpenMP team size to opt.n_threads for the call and
 * restores it on return.
 * @param delta_warm empty: no warm start
 * @param prm_warm   theta at which delta_warm was solved, or nullptr; its bound lo may differ from prm's (same
 *                   kind): the theta path then moves the bound too, a continuation in a_lo from a solved a_lo
 * @param micro      nullptr: no micro parts
 */
SolveResult solve_market(const Eigen::ArrayXd & s_data, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, const Eigen::ArrayXd & delta_warm,
                         const pcm_demand::Params * prm_warm, const MicroRequest * micro,
                         const SolveOptions & opt);

/** Shares, derivatives and micro parts at a given delta (no inversion). */
struct MarketEvaluation {
    Eigen::ArrayXd s;                                   //!< N
    Eigen::MatrixXd ds_ddelta, ds_dtheta, ds_dp;        //!< N x N, N x P, N x N (empty unless requested)
    Eigen::VectorXd micro;                              //!< Pm (empty without a micro request)
    Eigen::MatrixXd micro_d_delta, micro_d_theta;       //!< Pm x N, Pm x P (empty unless derivatives)
};

/**
 * One pass over the draws at a fixed delta: the draw loop solve_market uses after the inversion.
 * s, ds_ddelta, ds_dtheta and ds_dp equal pcm_demand::shares up to summation order within a draw.
 * Fixed blocks of 64 draws summed in block order, so results are bit-identical for every thread
 * count. Uses the calling thread's OpenMP team size. Throws std::invalid_argument on bad input.
 */
MarketEvaluation evaluate_market(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                                 const pcm_demand::Params & prm, const MicroRequest * micro,
                                 bool derivatives, bool price_derivatives);

/** Per-draw window anatomy (pcm_demand::detail::DrawAnatomy), (R,) arrays. */
struct FrontierAnatomy {
    Eigen::ArrayXi frontier_size;      //!< products with a window of positive mass
    Eigen::ArrayXd mass;               //!< the draw's inside share
    Eigen::ArrayXd log_range;          //!< log(e_0 / a_lo,i), 0 for a draw that buys nothing, inf without a bound
    Eigen::ArrayXd mass_log_width;     //!< sum over positive-mass windows of mass x log width
};

/**
 * Window anatomy of every draw of one market at delta: the inputs of the bounded elasticity law (LEARNINGS 32),
 * own elasticity about (p / hull gap) / (window width in log alpha). Uses the calling thread's OpenMP team
 * size. Throws std::invalid_argument on bad input.
 */
FrontierAnatomy frontier_anatomy(const Eigen::ArrayXd & delta, const pcm_demand::Market & mk,
                                 const pcm_demand::Params & prm);

}  // namespace pcm_kernel

#endif
