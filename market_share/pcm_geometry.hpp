#ifndef __PCM_GEOMETRY_HPP__
#define __PCM_GEOMETRY_HPP__

#include <cmath>
#include <vector>
#include <Eigen/Dense>

/**
 * Layer 0 of the PCM kernel: the vertical choice problem of ONE consumer (docs/estimation_harness/DESIGN.md 3.0).
 *
 * A draw sees qualities d (N, in its own "dollar" units) at increasing prices p (N) and has price sensitivity
 * a ~ LogNormal(0, sigma), optionally truncated below at a_lo (a_lo <= 0: not truncated). Its frontier is the
 * increasing part of the upper convex hull of {(0, 0)} U {(p_j, d_j)}; frontier product k is bought for a in
 * (e_{k+1}, e_k), e the hull slopes. Everything here takes one draw: no draw weights, draw indices, markets,
 * parameters theta / phi, inversion or moments.
 */
namespace pcm_geometry {

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

/** Indices of the products on the vertical frontier (increasing upper hull with the origin) of qualities delta at
 *  increasing prices p: the products with an open interval of buyers. */
std::vector<int> index_positive_shares(const Eigen::ArrayXd & delta, const Eigen::ArrayXd & p);

/** The conditional vertical share on the given frontier ``ind`` (increasing prices; see cond_share below for the
 *  outputs). ``check_order`` throws std::runtime_error when the cutoffs are not decreasing. */
void vertical_share(const std::vector<int> & ind, const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma,
                    double a_lo, Eigen::ArrayXd & s, Eigen::MatrixXd * Jd, Eigen::ArrayXd * dsig,
                    Eigen::MatrixXd * Jp, Eigen::ArrayXd * dlogc, bool check_order);

/**
 * The conditional vertical share of one draw with a lower bound and its derivatives, written into
 * global-index outputs (which must be zeroed).
 *
 * Qualities d (N) and prices p (N, increasing), price sensitivity LogNormal(0, sigma) truncated below at a_lo
 * (a_lo <= 0: not truncated). Products off the frontier (index_positive_shares), and frontier products whose
 * window lies below a_lo, get zero share and zero derivatives. When Jd is not null it receives ds/dd (N x N);
 * dsig (ds/dsigma, N), Jp (ds/dp, N x N) and dlogc (the derivative through a_lo = lo / c in log c at fixed d)
 * are filled when not null.
 */
void cond_share(const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma, double a_lo, Eigen::ArrayXd & s,
                Eigen::MatrixXd * Jd, Eigen::ArrayXd * dsig, Eigen::MatrixXd * Jp, Eigen::ArrayXd * dlogc);

/** The qualities that give shares_data exactly in the untruncated vertical model at prices p and sigma. */
Eigen::ArrayXd initial_guess(const Eigen::ArrayXd & shares_data, const Eigen::ArrayXd & p, double sigma);

/**
 * The same with a truncated below at lo: F_T(e_k) = sum_{j >= k} s_j. lo <= 0 is the untruncated initial_guess.
 */
Eigen::ArrayXd initial_guess(const Eigen::ArrayXd & s_data, const Eigen::ArrayXd & p, double sigma, double lo);

/** Window anatomy of one draw's frontier (ROUND5 5.1 item 5, 5.3 item 10). */
struct DrawAnatomy {
    int frontier_size = 0;           //!< products with a window of positive mass
    double mass = 0.0;               //!< sum of those windows' masses (the draw's inside share)
    double log_range = 0.0;          //!< log(e_0 / a_lo), e_0 the cutoff of the cheapest frontier product; 0 if mass = 0
    double mass_log_width = 0.0;     //!< sum_k m_k log(e_k / max(e_{k+1}, a_lo)) over positive-mass windows
};

/**
 * Anatomy of the vertical model of one draw in conditional qualities d, prices p, LogNormal(0, sigma)
 * truncated below at a_lo. Without a bound (a_lo <= 0) the range and the cheapest window are infinite.
 */
DrawAnatomy draw_anatomy(const Eigen::ArrayXd & d, const Eigen::ArrayXd & p, double sigma, double a_lo);

/**
 * For every product, its vertical gap below one draw's frontier: max(0, h(p_k) - q_k) with h the increasing part
 * of the upper hull of {(0, 0)} U {(p_k, q_k)} (constant at its top beyond the top vertex). Zero for products on
 * the frontier. ``hull`` is scratch. Under a bound the caller passes the sheared qualities q - a_lo p, whose
 * increasing hull holds exactly the vertices with a window of positive mass; the gap is the same in both.
 */
void frontier_gap(const Eigen::ArrayXd & q, const Eigen::ArrayXd & p, std::vector<int> & hull, Eigen::ArrayXd & gap);

}  // namespace pcm_geometry

#endif
