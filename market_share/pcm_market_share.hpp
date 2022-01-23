#ifndef __PCM_MARKET_SHARE_H__
#define __PCM_MARKET_SHARE_H__

#include <vector>
#include <Eigen/Dense>


using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pcm_share{
    /**
     * @brief Computes unconditional marker share of each product
     * 
     * @param delta_bar average quality of each good size Nx1 N - number of products
     * @param x values of heterogeneity size NxK K - number of characteristics of horizontal differentiation
     * @param p vector of prices size Nx1
     * @param sigma_p standard deviation of log of price sensitivity
     * @param sigma_x vector of standard deviation of each horizontal differentiation size K.
     * @param jacobian jacobian of the system with respect to deltas size NxN
     * @param grid the grid to integrate with respect to draws of heterogeneity. size DxK where D is the size of the number of draws 
     * @param weights weights of each point in  grid size K 
     * @return std::vector<double> 
     */
    Eigen::ArrayXd unc_share(const Eigen::ArrayXd& delta_bar, const Eigen::MatrixXd& x, const Eigen::ArrayXd& p, 
                                    double sigma_p, const Eigen::ArrayXd& sigma_x, const Eigen::ArrayXXd& grid, 
                                    const Eigen::ArrayXd & weights, Eigen::MatrixXd & jacobian);

    /**
     * @brief Computes unconditional marker share of each product without computing jacobian
     * 
     * @param delta_bar average quality of each good size Nx1 N - number of products
     * @param x values of heterogeneity size NxK K - number of characteristics of horizontal differentiation
     * @param p vector of prices size Nx1
     * @param sigma_p standard deviation of log of price sensitivity
     * @param sigma_x vector of standard deviation of each horizontal differentiation size K.
     * @param grid the grid to integrate with respect to draws of heterogeneity. size DxK where D is the size of the number of draws 
     * @param weights weights of each point in  grid size K 
     * @return std::vector<double> 
     */
    Eigen::ArrayXd unc_share(const Eigen::ArrayXd& delta_bar, const Eigen::MatrixXd& x, const Eigen::ArrayXd& p, 
                                    double sigma_p, const Eigen::ArrayXd& sigma_x, const Eigen::ArrayXXd& grid, 
                                    const Eigen::ArrayXd & weights);

    /**
     * @brief computes market share of vertical model conditional on draw of heterogeneity. 
     * 
     * @param delta vertical qualities of the products
     * @param p prices of products
     * @param sigma_p standard deviation of the price coefficient
     * @param jacobian matrix of jacobian with respect to deltas
     * @param check_positive_shares flag to check if all the shares are positive conditional on deltas and prices
     * @return std::vector<double> 
     */
    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, Eigen::MatrixXd & jacobian, bool check_positive_shares = true); 

    /**
     * @brief computes market share of vertical model conditional on draw of heterogeneity without jacobian
     * 
     * @param delta vertical qualities of the products
     * @param p prices of products
     * @param sigma_p standard deviation of the price coefficient
     * @param check_positive_shares flag to check if all the shares are positive conditional on deltas and prices
     * @return std::vector<double> 
     */
    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, bool check_positive_shares = true); 


    Eigen::ArrayXd initial_guess(const Eigen::ArrayXd& shares_data, const Eigen::ArrayXd& p, double sigma_p);


}
#endif // __PCM_MARKET_SHARE_H__