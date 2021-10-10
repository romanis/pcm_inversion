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
     * @param sigma_x vector of standard deviation of each horizontal differentiation.
     * @param jacobian jacibian of the system with respect to deltas
     * @param grid the grid to integrate with respect to draws of heterogeneity. size KxD
     * @param weights weights of each point in  grid size K 
     * @return std::vector<double> 
     */
    std::vector<double> unc_share(const VectorXd& delta_bar, const MatrixXd& x, const VectorXd& p, double sigma_p, const VectorXd& sigma_x, const MatrixXd& grid, const VectorXd & weights, MatrixXd & jacobian);
}

#endif // __PCM_MARKET_SHARE_H__