#include "pcm_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <boost/math/distributions/lognormal.hpp>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::seq;
using Eigen::indexing::last;

namespace pcm_geometry {

std::vector<int> index_positive_shares(const ArrayXd& delta, const ArrayXd & p){
    std::vector<int> ind;
    for(int j=0; j<delta.size(); ++j){
//            calculate upper bound on price elasticity of people who will choose product j
        double min_upper, max_lower;
        if(j>0){
            ArrayXd u_alpha = (delta[j] - delta(seq(0, j-1))) / (p[j] - p(seq(0, j-1)));
            min_upper = std::min(u_alpha.minCoeff(), delta[j]/p[j]);
        }else{
            min_upper = delta[j]/p[j];
        }

        if(j < delta.size()-1){
            ArrayXd l_alpha = (delta[j] - delta(seq(j+1, last))) / (p[j] - p(seq(j+1, last)));
            max_lower = std::max(l_alpha.maxCoeff(), 0.0);
        }else{
            max_lower = 0;
        }

//            if there is slack between up and lower, it has positive market share
        if(min_upper > max_lower){
            ind.push_back(j);
        }
    }
    return ind;
}

void vertical_share(const std::vector<int> & ind, const ArrayXd & d, const ArrayXd & p, double sigma,
                    double a_lo, ArrayXd & s, MatrixXd * Jd, ArrayXd * dsig, MatrixXd * Jp,
                    ArrayXd * dlogc, bool check_order)
{
    const int m = static_cast<int>(ind.size());
    if (m == 0) return;

    // endpoints e_0..e_m (e_m = 0) and the price gaps they divide by
    std::vector<double> e(m + 1, 0.0), gap(m, 0.0);
    gap[0] = p[ind[0]];
    e[0] = d[ind[0]] / gap[0];
    for (int k = 1; k < m; ++k) {
        gap[k] = p[ind[k]] - p[ind[k - 1]];
        e[k] = (d[ind[k]] - d[ind[k - 1]]) / gap[k];
    }
    if (check_order) {
        for (int k = 0; k < m; ++k)
            if (e[k + 1] > e[k])
                throw std::runtime_error("endpoints are not sorted properly. check which goods go to conditional market share");
    }
    const Trunc tr = make_trunc(a_lo, sigma);
    std::vector<LnTerms> t(m + 1);
    for (int k = 0; k <= m; ++k) t[k] = ln_terms(e[k], sigma, tr);

    for (int k = 0; k < m; ++k) s[ind[k]] = t[k].F - t[k + 1].F;
    if (!Jd) return;

    // s_k = F(e_k) - F(e_{k+1}); e_k depends on d_k (+1/gap_k) and d_{k-1} (-1/gap_k)
    for (int k = 0; k < m; ++k) {
        const int gk = ind[k];
        if (dsig) (*dsig)[gk] = t[k].g - t[k + 1].g;
        if (dlogc) (*dlogc)[gk] = t[k].h - t[k + 1].h;
        // through e_k
        (*Jd)(gk, gk) += t[k].f / gap[k];
        if (Jp) (*Jp)(gk, gk) += t[k].f * (-e[k] / gap[k]);
        if (k > 0) {
            (*Jd)(gk, ind[k - 1]) += -t[k].f / gap[k];
            if (Jp) (*Jp)(gk, ind[k - 1]) += t[k].f * (e[k] / gap[k]);
        }
        // through e_{k+1} (absent for the last product: e_m = 0 is fixed)
        if (k + 1 < m) {
            const double fk1 = t[k + 1].f, g1 = gap[k + 1];
            (*Jd)(gk, ind[k + 1]) -= fk1 / g1;
            (*Jd)(gk, gk) -= -fk1 / g1;
            if (Jp) {
                (*Jp)(gk, ind[k + 1]) -= fk1 * (-e[k + 1] / g1);
                (*Jp)(gk, gk) -= fk1 * (e[k + 1] / g1);
            }
        }
    }
}

void cond_share(const ArrayXd & d, const ArrayXd & p, double sigma, double a_lo, ArrayXd & s, MatrixXd * Jd,
                ArrayXd * dsig, MatrixXd * Jp, ArrayXd * dlogc)
{
    vertical_share(index_positive_shares(d, p), d, p, sigma, a_lo, s, Jd, dsig, Jp, dlogc, false);
}

ArrayXd initial_guess(const ArrayXd& shares_data, const ArrayXd& p, double sigma_p){
    ArrayXd guess(p.size());
//    calculate initial guess based on undisturbed model
//    delta_0(1) = p_sim(1) * logninv(sum(share),0,sigma_p);
    double sum_share=shares_data.sum();

    //    create lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);
//    compute the first delta guess
    guess[0] = (p[0]*boost::math::quantile(lognormDistr, sum_share));
//    compute the rest guesses
    for(int i=1; i<p.size(); ++i){
        sum_share -= shares_data[i-1];
        guess[i] = (guess[i-1] + (p[i] - p[i-1])*boost::math::quantile(lognormDistr, sum_share));
    }
    return guess;
}

ArrayXd initial_guess(const ArrayXd & s_data, const ArrayXd & p, double sigma, double lo)
{
    if (!(lo > 0.0)) return initial_guess(s_data, p, sigma);
    // F_T(e_k) = cum_k means Phi(z(e_k)) = Phi_lo + Z cum_k: the untruncated quantile at Phi_lo + Z cum_k.
    // The untruncated initial_guess takes quantiles at the shares cumulated from each product up, so scale the
    // shares by Z and append one dearer dummy product of share Phi_lo; the real products' guesses do not
    // depend on the dummy's price.
    const Trunc tr = make_trunc(lo, sigma);
    const int N = static_cast<int>(s_data.size());
    if (N == 0) return ArrayXd();
    ArrayXd s2(N + 1), p2(N + 1);
    s2.head(N) = tr.Z * s_data;
    s2[N] = tr.Phi_lo;
    p2.head(N) = p;
    p2[N] = p[N - 1] + std::max(1.0, std::abs(p[N - 1]));
    return initial_guess(s2, p2, sigma).head(N);
}

DrawAnatomy draw_anatomy(const ArrayXd & d, const ArrayXd & p, double sigma, double a_lo)
{
    DrawAnatomy out;
    const std::vector<int> ind = index_positive_shares(d, p);
    const int m = static_cast<int>(ind.size());
    if (m == 0) return out;
    std::vector<double> e(m + 1, 0.0);
    e[0] = d[ind[0]] / p[ind[0]];
    for (int k = 1; k < m; ++k) e[k] = (d[ind[k]] - d[ind[k - 1]]) / (p[ind[k]] - p[ind[k - 1]]);
    const Trunc tr = make_trunc(a_lo, sigma);
    const double inf = std::numeric_limits<double>::infinity();
    double F_hi = ln_terms(e[0], sigma, tr).F;
    for (int k = 0; k < m; ++k) {
        const double F_lo = ln_terms(e[k + 1], sigma, tr).F;
        const double mk = F_hi - F_lo;
        F_hi = F_lo;
        if (!(mk > 0.0)) continue;
        ++out.frontier_size;
        out.mass += mk;
        const double lo = std::max(e[k + 1], tr.on ? a_lo : 0.0);
        out.mass_log_width += lo > 0.0 ? mk * std::log(e[k] / lo) : inf;
    }
    if (out.mass > 0.0) out.log_range = tr.on ? std::log(e[0] / a_lo) : inf;
    return out;
}

void frontier_gap(const ArrayXd & q, const ArrayXd & p, std::vector<int> & hull, ArrayXd & gap)
{
    const int N = static_cast<int>(q.size());
    gap.resize(N);
    // upper hull by price (prices are sorted); index -1 stands for the origin (0, 0)
    auto px = [&](int k) { return k < 0 ? 0.0 : p[k]; };
    auto qy = [&](int k) { return k < 0 ? 0.0 : q[k]; };
    hull.clear();
    hull.push_back(-1);
    for (int k = 0; k < N; ++k) {
        while (hull.size() >= 2) {
            const int a = hull[hull.size() - 2], bq = hull.back();
            const double cross = (px(bq) - px(a)) * (qy(k) - qy(a)) - (qy(bq) - qy(a)) * (px(k) - px(a));
            if (cross >= 0) hull.pop_back(); else break;
        }
        hull.push_back(k);
    }
    // keep the increasing part: drop vertices after the maximum quality
    double top = 0.0;
    size_t end = 1;
    for (size_t h = 1; h < hull.size(); ++h) {
        if (qy(hull[h]) > top) { top = qy(hull[h]); end = h + 1; }
    }
    // evaluate the hull at each product's price
    size_t seg = 0;
    for (int k = 0; k < N; ++k) {
        while (seg + 1 < end && px(hull[seg + 1]) < p[k]) ++seg;
        double h;
        if (seg + 1 >= end) h = top;
        else {
            const int a = hull[seg], bq = hull[seg + 1];
            const double w = (p[k] - px(a)) / (px(bq) - px(a));
            h = qy(a) + w * (qy(bq) - qy(a));
        }
        gap[k] = std::max(0.0, h - q[k]);
    }
}

}  // namespace pcm_geometry
