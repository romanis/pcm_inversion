#ifndef __MARKET_INVERSION__
#define __MARKET_INVERSION__

#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "pcm_market_share.hpp"
#include "nlopt.hpp"

namespace share_inversion{
    extern double MIN_ADMISSIBLE_SHARE; //!< minimal share of any product (or outside option) that is supported by the inversion mechanism. If any share is lower than this, inversion is deemed unstable

    /**
     * @brief Selects which inversion algorithm to use when calling invert_shares.
     *
     * NLOPT_SLSQP uses the existing NLopt LD_SLSQP constrained optimizer.
     * CONE_AWARE_NEWTON uses the cone-aware Newton solver with feasibility backtracking.
     */
    enum class InversionMethod {
        NLOPT_SLSQP,
        CONE_AWARE_NEWTON
    };

    /**
     * @brief struct that is holding information relevant to computation of PCM share
     * 
     */
    struct pcm_parameters{
        Eigen::MatrixXd x; //!< Matrix of features of the products. Size [n_products x n_features]
        Eigen::ArrayXd p; //!< Array of prices of products. Size n_products
        Eigen::ArrayXd sigma_x; //!< Array of standard deviations of random taste for every feature. Size n_features
        double sigma_p; //!< Standard deviation of the distribution of (log of) price sensitivity
        Eigen::ArrayXXd grid; //!< Grid that integrates out the distribution for the random taste for features. Size [n_draws x n_features]. The more draws the more exact is the integration
        Eigen::ArrayXd weights; //!< Array of weights of every point in a grid. Size n_draws
        Eigen::ArrayXd data_shares; //!< Observed shares of products. Size n_products. Should sum up to (strictly) less than 1
        int func_evals = 0, jacobian_evals = 0; //!< counters of the time we have called the function with and without jacobian
        nlopt::algorithm nlopt_algo; //!< one of the algorithms from NLOPT library. default is nlopt::LD_SLSQP
        double share_equality_tolerances; //!< tolerance with withich we expect the inversion to come to observed shares. default is 1e-8
        double delta_step_tolerance; //!< tolerance to step in delta space. default is 1e-5
        std::vector<double> delta_initial; //!< initial step for the inversion. 
                                            //!< Highly recommended NOT to provide it because the algorithm has a very good initial guess of its own. Only provide it if you really know what you are doing.
                                            //!< If not all shares are positive at this initial guess, it will be ignored.
        unsigned max_number_of_function_calls; //!< maximum number of times share predicting function will be called. default 1000
        unsigned number_of_times_function_called = 0;
        


        pcm_parameters(Eigen::MatrixXd& x, Eigen::ArrayXd &p, double sigma_p, Eigen::ArrayXd &sigma_x, Eigen::ArrayXXd& grid, 
                       Eigen::ArrayXd& weights, Eigen::ArrayXd& data_shares, std::vector<double> delta_initial = std::vector<double>(), 
                       nlopt::algorithm nlopt_algo = nlopt::LD_SLSQP, 
                       double share_equality_tolerances = 1e-8, double delta_step_tolerance = 1e-5, unsigned max_number_of_function_calls = 1000) :
                            x(x), p(p), sigma_x(sigma_x), sigma_p(sigma_p), grid(grid), weights(weights), data_shares(data_shares), 
                            nlopt_algo(nlopt_algo), share_equality_tolerances(share_equality_tolerances), delta_step_tolerance(delta_step_tolerance), delta_initial(delta_initial),
                            max_number_of_function_calls(max_number_of_function_calls) {}; //!< constructor of the struct
    } ;

    /**
     * @brief Inverts PCM market shares using the NLopt LD_SLSQP constrained optimizer.
     *
     * @param params all parameters of the market; observed shares are in params.data_shares
     * @return Eigen::ArrayXd delta of size n_products such that unc_share(delta) == data_shares
     */
    Eigen::ArrayXd invert_shares_nlopt(pcm_parameters & params);

    /**
     * @brief Inverts PCM market shares to recover delta, dispatching to the algorithm
     *        selected by method. Default method is NLOPT_SLSQP, preserving all existing
     *        call sites without modification.
     *
     * @param params all parameters of the market; observed shares are in params.data_shares
     * @param method inversion algorithm to use; defaults to InversionMethod::NLOPT_SLSQP
     * @return Eigen::ArrayXd delta of size n_products such that unc_share(delta) == data_shares
     */
    Eigen::ArrayXd invert_shares(pcm_parameters & params,
                                  InversionMethod method);
}

namespace cone_newton {

/**
 * @brief Selects the starting point for the cone-aware Newton iteration.
 *
 * Returns pcm_share::initial_guess unless params.delta_initial is non-empty,
 * matches the product count, and its predicted shares are strictly positive
 * with sum below 1 - share_inversion::MIN_ADMISSIBLE_SHARE.
 *
 * @param params fully populated share_inversion::pcm_parameters struct
 * @return Eigen::ArrayXd initial delta of size n_products inside the feasibility cone P
 */
Eigen::ArrayXd select_initial_delta(share_inversion::pcm_parameters & params);

/**
 * @brief Computes a Newton step from the current Jacobian and residual.
 *
 * Attempts J.lu().solve(-r). If the result contains NaN or its Euclidean norm
 * exceeds 1e6, falls back to the diagonal-Jacobian steepest-descent step
 * -J.diagonal()^{-1} * r.
 *
 * @param J dense Jacobian matrix of size n x n returned by pcm_share::unc_share
 * @param r residual array of size n equal to s - data_shares
 * @return Eigen::VectorXd Newton step of size n
 */
Eigen::VectorXd compute_newton_step(const Eigen::MatrixXd & J,
                                    const Eigen::ArrayXd  & r);

/**
 * @brief Backtracks the Newton step to restore feasibility.
 *
 * Starting from lambda = 1, halves lambda up to max_halvings = 50 times until
 * all shares of delta + lambda * step are strictly positive and their sum is
 * below 1 - share_inversion::MIN_ADMISSIBLE_SHARE. Increments calls for each
 * pcm_share::unc_share evaluation.
 *
 * @param delta current iterate of size n
 * @param step Newton step of size n
 * @param params fully populated share_inversion::pcm_parameters struct
 * @param calls reference to the function-evaluation counter; incremented in place
 * @return std::pair containing the best feasible delta_candidate and its share vector
 */
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
    backtrack_to_feasibility(const Eigen::ArrayXd             & delta,
                             const Eigen::VectorXd            & step,
                             share_inversion::pcm_parameters  & params,
                             unsigned                         & calls);

/**
 * @brief Attempts to repair an infeasible delta_candidate via adjust heuristic.
 *
 * Calls adjust_deltas_towards_positive_shares on delta_candidate; if the
 * resulting shares are all strictly positive, returns the repaired delta.
 * Otherwise returns delta_candidate unchanged.
 *
 * @param delta_candidate infeasible iterate of size n
 * @param s_candidate predicted shares for delta_candidate; some may be non-positive
 * @param params fully populated share_inversion::pcm_parameters struct
 * @param calls reference to the function-evaluation counter; incremented in place
 * @return Eigen::ArrayXd repaired delta of size n, or delta_candidate if repair fails
 */
Eigen::ArrayXd recover_feasibility(const Eigen::ArrayXd             & delta_candidate,
                                   const Eigen::ArrayXd             & s_candidate,
                                   share_inversion::pcm_parameters  & params,
                                   unsigned                         & calls);

/**
 * @brief Cone-aware Newton solver: finds delta such that unc_share(delta) == data_shares.
 *
 * Orchestrates select_initial_delta, then iterates: compute_newton_step,
 * backtrack_to_feasibility, recover_feasibility until the L-inf share residual
 * drops below params.share_equality_tolerances or the call budget is exhausted.
 *
 * @param params fully populated share_inversion::pcm_parameters struct
 * @return Eigen::ArrayXd delta solution of size n_products
 */
Eigen::ArrayXd invert(share_inversion::pcm_parameters & params);

} // namespace cone_newton

#endif //__MARKET_INVERSION__
