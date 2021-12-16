#ifndef __MARKET_INVERSION__
#define __MARKET_INVERSION__

#include <vector>
#include <Eigen/Dense>
#include "pcm_market_share.hpp"
#include "nlopt.hpp"

namespace share_inversion{
    double MIN_ADMISSIBLE_SHARE = 1e-5; //!< minimal share of any product (or outside option) that is supported by the inversion mechanism. If any share is lower than this, inversion is deemed unstable

    /**
     * @brief struct that is holding information relevant to computation of PCM share
     * 
     */
    struct pcm_parameters{
        Eigen::MatrixXd x; //!< Matrix of features of the products. Size [n_products x n_features]
        Eigen::ArrayXd p; //!< Array of prices of products. Size n_products
        double sigma_p; //!< Standard deviation of the distribution of (log of) price sensitivity
        Eigen::ArrayXd sigma_x; //!< Array of standard deviations of random taste for every feature. Size n_features
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
     * @brief inverts PCM market shares conditional on all other parameters of the model. Observed shares are in params.data_shares
     * 
     * @param params all the parameters of the market, see the meanings of the individual members in class description
     * @return Eigen::ArrayXd 
     */
    Eigen::ArrayXd invert_shares(pcm_parameters & params);
}

#endif //__MARKET_INVERSION__