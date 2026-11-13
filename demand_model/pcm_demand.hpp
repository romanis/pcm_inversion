#ifndef __PCM_DEMAND_HPP__
#define __PCM_DEMAND_HPP__

#include <vector>
#include <Eigen/Dense>

/**
 * Pure characteristics demand with observed demographics, for one market.
 *
 * Consumer i (a draw with weight w_i) gets utility from product j
 *     u_ij = delta_j + x_j . t_i - c_i * a_i * p_j,     u_i0 = 0,
 * with a_i ~ LogNormal(0, sigma_p) integrated in closed form.
 *   t_i (taste vector, size K) = sum_m theta_m * taste_basis(i, m) on column taste_col[m]
 *   log c_i (price-sensitivity scale) = sum_q phi_q * scale_basis(i, q)
 * Dividing utility by c_i gives a pure vertical model in "dollar" qualities
 *     delta_cond_ij = (delta_j + x_j . t_i) / c_i
 * which is what pcm_share::cond_share already solves.
 *
 * The caller builds t_i and log c_i from its own parameters; this module only needs the
 * bases to return derivatives with respect to every parameter.
 */
namespace pcm_demand {

struct Market {
    Eigen::MatrixXd x;               //!< N x K product characteristics
    Eigen::ArrayXd  p;               //!< N prices, strictly increasing
    Eigen::ArrayXd  weights;         //!< R draw weights
    Eigen::MatrixXd taste_basis;     //!< R x M, d t_i[taste_col[m]] / d theta_m
    std::vector<int> taste_col;      //!< M, characteristic column each taste parameter loads on
    Eigen::MatrixXd scale_basis;     //!< R x Q, d log c_i / d phi_q
};

struct Params {
    Eigen::ArrayXd theta;            //!< M taste parameters
    Eigen::ArrayXd phi;              //!< Q price-scale parameters
    double sigma_p;                  //!< sd of log price sensitivity
};

struct Shares {
    Eigen::ArrayXd  s;               //!< N predicted shares
    Eigen::MatrixXd d_delta;         //!< N x N, ds/d delta
    Eigen::MatrixXd d_theta;         //!< N x (M + Q + 1), ds/d[theta, phi, sigma_p]
    Eigen::MatrixXd d_price;         //!< N x N, ds/dp
};

/**
 * Shares and (optionally) all derivatives for one market.
 * @param with_derivatives if false only s is filled
 */
Shares shares(const Eigen::ArrayXd & delta, const Market & m, const Params & prm, bool with_derivatives);

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
InversionResult invert(const Eigen::ArrayXd & s_data, const Market & m, const Params & prm,
                       const Eigen::ArrayXd & delta0, double tol = 1e-12,
                       unsigned max_share_evaluations = 5000);

}  // namespace pcm_demand

#endif
