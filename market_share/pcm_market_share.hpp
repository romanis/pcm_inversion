#ifndef __PCM_MARKET_SHARE_H__
#define __PCM_MARKET_SHARE_H__

#include <vector>
#include <cmath>
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


    /**
     * Truncated lognormal building blocks of the vertical model, shared by every caller (pcm_demand, the second-choice
     * kernel). Not part of the stable interface.
     */
    namespace detail {

    constexpr double INV_SQRT_2PI = 0.39894228040143267794;

    /** LogNormal(0, sigma) cdf F, pdf f and dF/dsigma g at e > 0 (all 0 at e <= 0); h = dF/dlog c, zero without
     *  truncation. */
    struct LnTerms { double F, f, g, h; };

    /** Lower truncation of a ~ LogNormal(0, sigma) at a_lo for one draw: z_lo = log(a_lo) / sigma, the mass kept
     *  Z = 1 - Phi(z_lo) and phi(z_lo). Off when a_lo <= 0 (the untruncated model). */
    struct Trunc {
        bool on = false;
        double a_lo = 0.0, z_lo = 0.0, Phi_lo = 0.0, Z = 1.0, phi_lo = 0.0;
    };

    inline Trunc make_trunc(double a_lo, double sigma) {
        Trunc tr;
        if (!(a_lo > 0.0)) return tr;
        tr.on = true;
        tr.a_lo = a_lo;
        tr.z_lo = std::log(a_lo) / sigma;
        tr.Phi_lo = 0.5 * std::erfc(-tr.z_lo / std::sqrt(2.0));
        tr.Z = 0.5 * std::erfc(tr.z_lo / std::sqrt(2.0));
        tr.phi_lo = INV_SQRT_2PI * std::exp(-0.5 * tr.z_lo * tr.z_lo);
        return tr;
    }

    /** F, f, g, h of the (truncated) lognormal at e. With truncation F_T(e) = (Phi(z) - Phi(z_lo)) / Z for e > a_lo,
     *  else 0; Z underflowing to 0 means no mass above the bound (every window empty). */
    inline LnTerms ln_terms(double e, double sigma, const Trunc & tr) {
        if (!tr.on) {
            if (e <= 0.0) return {0.0, 0.0, 0.0, 0.0};
            const double z = std::log(e) / sigma;
            const double phi = INV_SQRT_2PI * std::exp(-0.5 * z * z);
            return {0.5 * std::erfc(-z / std::sqrt(2.0)), phi / (e * sigma), -phi * z / sigma, 0.0};
        }
        if (e <= tr.a_lo || !(tr.Z > 0.0)) return {0.0, 0.0, 0.0, 0.0};
        const double z = std::log(e) / sigma;
        const double phi = INV_SQRT_2PI * std::exp(-0.5 * z * z);
        const double Qz = 0.5 * std::erfc(z / std::sqrt(2.0));          // 1 - Phi(z)
        // the difference of the two cdfs taken in the tail where it is accurate
        const double num = tr.z_lo < 0.0 ? 0.5 * std::erfc(-z / std::sqrt(2.0)) - tr.Phi_lo : tr.Z - Qz;
        const double one_minus_F = Qz / tr.Z;
        const double sZ = sigma * tr.Z;
        return {num / tr.Z, phi / (e * sZ), (-phi * z + tr.phi_lo * tr.z_lo * one_minus_F) / sZ,
                tr.phi_lo * one_minus_F / sZ};
    }

    /** The conditional vertical share on the given frontier ``ind`` (increasing prices; see cond_share below for the
     *  outputs). ``check_order`` throws std::runtime_error when the cutoffs are not decreasing. */
    void vertical_share(const std::vector<int> & ind, const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma,
                        double a_lo, Eigen::ArrayXd & s, Eigen::MatrixXd * Jd, Eigen::ArrayXd * dsig,
                        Eigen::MatrixXd * Jp, Eigen::ArrayXd * dlogc, bool check_order);

    }  // namespace detail

    /**
     * @brief The conditional vertical share of one draw with a lower bound and its derivatives, written into
     * global-index outputs (which must be zeroed).
     *
     * Qualities d (N) and prices p (N, increasing), price sensitivity LogNormal(0, sigma) truncated below at a_lo
     * (a_lo <= 0: not truncated). Products off the frontier (index_positive_shares), and frontier products whose
     * window lies below a_lo, get zero share and zero derivatives. When Jd is not null it receives ds/dd (N x N);
     * dsig (ds/dsigma, N), Jp (ds/dp, N x N) and dlogc (the derivative through a_lo = lo / c in log c at fixed d)
     * are filled when not null. The untruncated overloads above are this function with a_lo = 0.
     */
    void cond_share(const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma, double a_lo, Eigen::ArrayXd & s,
                    Eigen::MatrixXd * Jd, Eigen::ArrayXd * dsig, Eigen::MatrixXd * Jp, Eigen::ArrayXd * dlogc);

}

/** Indices of the products on the vertical frontier (increasing upper hull with the origin) of qualities delta at
 *  increasing prices p: the products with an open interval of buyers. */
std::vector<int> index_positive_shares(const Eigen::ArrayXd & delta, const Eigen::ArrayXd & p);

#endif // __PCM_MARKET_SHARE_H__