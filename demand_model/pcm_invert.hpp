#ifndef __PCM_INVERT_HPP__
#define __PCM_INVERT_HPP__

#include <limits>
#include <string>
#include <Eigen/Dense>
#include "pcm_aggregate.hpp"

/**
 * Layer 2 of the PCM kernel: solve delta with pcm_aggregate::shares(delta) == s_data for one market
 * (docs/estimation_harness/DESIGN.md 3.0). It knows layer 1 and the solver options; it does not know moments,
 * GMM or papers.
 */
namespace pcm_invert {

struct InversionResult {
    Eigen::ArrayXd delta;
    double max_abs_residual;
    unsigned iterations;
    unsigned share_evaluations;
    bool converged;
};

/**
 * Finds delta with shares(delta) == s_data by Newton with an Armijo line search on
 * 0.5*||s - s_data||^2, kept inside the region where every share is positive and the
 * inside shares sum below one; Levenberg-Marquardt damping when Newton gives no descent.
 * @param delta0 starting point; ignored (replaced by the vertical-model guess) if empty or infeasible
 */
InversionResult invert(const Eigen::ArrayXd & s_data, const pcm_aggregate::Market & m, const pcm_aggregate::Params & prm,
                       const Eigen::ArrayXd & delta0, double tol = 1e-12,
                       unsigned max_share_evaluations = 5000);

/** Budgets of the stage machine (pcm_kernel::SolveOptions carries the same fields). */
struct Options {
    double tol = 1e-9;                                    //!< max |s - s_data|
    unsigned warm_evals = 2500, cold_evals = 40000, path_evals = 1500;
    double path_seconds = 90.0;                           //!< budget for the theta-path continuation
    unsigned cold_draws = 0;                              //!< draw homotopy of the cold stage (0: off)
    int cold_draw_factor = 8;                             //!< thinning factor between its levels (at least 2)
};

/** What the stages reached: the delta of the smallest finite residual (a converged one wins). */
struct Solution {
    Eigen::ArrayXd delta;                                 //!< empty if no stage returned a delta
    double residual = std::numeric_limits<double>::infinity();   //!< finite once a stage reached a feasible delta
    bool converged = false;
    unsigned evals = 0;                                   //!< share evaluations of all stages
    std::string stages;                                   //!< "warm ok 0.01s 4ev; cold ..." etc.
};

/**
 * The stage machine of pcm_kernel::solve_market: warm (from delta_warm at prm), theta path (from *prm_warm to prm,
 * only if the warm stage failed; the bound moves along when prm_warm->lo differs), draws (the draw homotopy, only
 * with opt.cold_draws > 0 and no converged stage yet), cold (vertical-model guess). Inputs are validated by the
 * caller. Writes into ``out`` as it goes, so a stage that throws leaves the earlier stages' record there.
 * @param delta_warm empty: no warm start
 * @param prm_warm   theta at which delta_warm was solved, or nullptr
 */
void solve(const Eigen::ArrayXd & s_data, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm,
           const Eigen::ArrayXd & delta_warm, const pcm_aggregate::Params * prm_warm, const Options & opt,
           Solution & out);

}  // namespace pcm_invert

#endif
