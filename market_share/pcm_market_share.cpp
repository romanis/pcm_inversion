#include "pcm_market_share.hpp"
#include <iostream>
#include <map>
#include <vector>
#include <utility>
#include <cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <boost/math/distributions/lognormal.hpp>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Core>



using namespace std;
using namespace Eigen;
using Eigen::indexing::last;
using pcm_share::detail::LnTerms;
using pcm_share::detail::Trunc;
using pcm_share::detail::ln_terms;
using pcm_share::detail::make_trunc;

std::vector<int> index_positive_shares(const ArrayXd& delta, const ArrayXd & p){
    vector<int> ind;
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

namespace pcm_share{
    Eigen::ArrayXd unc_share(const ArrayXd& delta_bar, const MatrixXd& x, const ArrayXd& p, double sigma_p, const ArrayXd& sigma_x, const ArrayXXd& grid, const ArrayXd & weights, MatrixXd & jacobian)
    {
        Eigen::ArrayXd un_share = ArrayXd::Zero(delta_bar.size());
        // set jacobian to zero matrix
        jacobian = MatrixXd::Zero(delta_bar.size(), delta_bar.size());
    
//    check that vector x[0] and sigma_x are the same dimension
        if(x.cols() != sigma_x.size()){
            throw runtime_error("number of columns in x is different from size of sigma_x");
        }
    //    check that all x vectors have the same length
        
        if (x.cols() != grid.cols()){
            throw runtime_error("number of columns in x is different from number of columns of grid");
        }
        if (x.rows() != delta_bar.size()){
            throw runtime_error("number of rows in x is different from the number of elements in delta_bar");
        }
        if(p.minCoeff()<=0){
            throw runtime_error("one of elements of p vector is negative or zero");
        }
    
    
    //    loop over all points in the grid
#pragma omp parallel for schedule(dynamic, 10)
        for(int draw=0; draw<weights.size(); ++draw){
    //        calculate the conditional quality
    //        delta_hat = delta + sigma_x*x*nu(i,:);
            ArrayXd delta_cond = delta_bar + (x*(sigma_x.transpose()*grid.row(draw)).matrix().transpose()).array();
    //        determine indexes of goods that have positive market shares
            vector<int> ind = index_positive_shares(delta_cond, p);

    //        if there is a product with positive market share
            if(ind.size() > 0){
    //            pick those deltas and prices that correspond to products with positive market shares
                MatrixXd jacobian_tmp;
                ArrayXd delta_positive = delta_cond(ind);
                ArrayXd p_positive = p(ind);
                
                
                ArrayXd shares_tmp = cond_share(delta_positive,p_positive, sigma_p, jacobian_tmp, false);

    //            add shares to corresponding dimensions of un_share
#pragma omp critical
{
                un_share(ind) += shares_tmp*weights[draw];
                jacobian(ind, ind) += jacobian_tmp*weights[draw];
}
                
            }
        }
        return un_share;
    }


    Eigen::ArrayXd unc_share(const ArrayXd& delta_bar, const MatrixXd& x, const ArrayXd& p, double sigma_p, const ArrayXd& sigma_x, const ArrayXXd& grid, const ArrayXd & weights)
    {
        Eigen::ArrayXd un_share = ArrayXd::Zero(delta_bar.size());
    
//    check that vector x[0] and sigma_x are the same dimension
        if(x.cols() != sigma_x.size()){
            throw runtime_error("number of columns in x is different from size of sigma_x");
        }
    //    check that all x vectors have the same length
        
        if (x.cols() != grid.cols()){
            throw runtime_error("number of columns in x is different from number of columns of grid");
        }
        if (x.rows() != delta_bar.size()){
            throw runtime_error("number of rows in x is different from the number of elements in delta_bar");
        }
        if(p.minCoeff()<=0){
            throw runtime_error("one of elements of p vector is negative or zero");
        }
    
    
    //    loop over all points in the grid
    #pragma omp parallel for schedule(dynamic, 10)
        for(int draw=0; draw<weights.size(); ++draw){
    //        calculate the conditional quality
    //        delta_hat = delta + sigma_x*x*nu(i,:);
            ArrayXd delta_cond = delta_bar + (x*(sigma_x.transpose()*grid.row(draw)).matrix().transpose()).array();
    //        determine indexes of goods that have positive market shares
            vector<int> ind = index_positive_shares(delta_cond, p);


    //        if there is a product with positive market share
            if(ind.size() > 0){
    //            pick those deltas and prices that correspond to products with positive market shares
                ArrayXd delta_positive = delta_cond(ind);
                ArrayXd p_positive = p(ind);
                
                
                ArrayXd shares_tmp = cond_share(delta_positive,p_positive, sigma_p, false);

    //            add shares to corresponding dimensions of un_share
#pragma omp critical
                un_share(ind) += shares_tmp*weights[draw];
            }
        }
        // cout<<"share"<<endl;
        //    for(auto it: un_share){
        //        cout<< it<<endl;
        //    }
        return un_share;
    }


    namespace {
    // the frontier of the untruncated overloads: index_positive_shares, or every product
    vector<int> frontier_of(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, bool check_positive_shares){
        if(!std::is_sorted(p.begin(), p.end())){
            throw std::runtime_error("price vector not sorted");
        }
        if(delta.size() != p.size()){
            throw runtime_error("price vector and delta sizes differ");
        }
        if(check_positive_shares) return index_positive_shares(delta, p);
        vector<int> ind(delta.size());
        for(int i = 0; i < delta.size(); ++i) ind[i] = i;
        return ind;
    }
    }  // namespace

    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, Eigen::MatrixXd & jacobian,  bool check_positive_shares){
    //    conditional on heterogeneity market shares of the pure vertical model and their jacobian in delta
        const vector<int> ind = frontier_of(delta, p, check_positive_shares);
        Eigen::ArrayXd con_share = ArrayXd::Zero(delta.size());
        jacobian = MatrixXd::Zero(delta.size(), delta.size());
        detail::vertical_share(ind, delta, p, sigma_p, 0.0, con_share, &jacobian, nullptr, nullptr, nullptr, true);
        return con_share;
    }


    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p,  bool check_positive_shares){
    //    conditional on heterogeneity market shares of the pure vertical model
        const vector<int> ind = frontier_of(delta, p, check_positive_shares);
        Eigen::ArrayXd con_share = ArrayXd::Zero(delta.size());
        detail::vertical_share(ind, delta, p, sigma_p, 0.0, con_share, nullptr, nullptr, nullptr, nullptr, true);
        return con_share;
    }


    void detail::vertical_share(const vector<int> & ind, const ArrayXd & d, const ArrayXd & p, double sigma,
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
                    throw runtime_error("endpoints are not sorted properly. check which goods go to conditional market share");
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
        detail::vertical_share(index_positive_shares(d, p), d, p, sigma, a_lo, s, Jd, dsig, Jp, dlogc, false);
    }



    Eigen::ArrayXd initial_guess(const Eigen::ArrayXd& shares_data, const Eigen::ArrayXd& p, double sigma_p){
        ArrayXd guess(p.size());
    //    calculate initial guess based on undisturbed model
    //    delta_0(1) = p_sim(1) * logninv(sum(share),0,sigma_p);
        double sum_share=shares_data.sum();
        
        //    create lognormal distr
        boost::math::lognormal lognormDistr(0, sigma_p);
    //    cout<<boost::math::quantile(lognormDistr, sum_share)<<endl;
    //    compute the first delta guess
        guess[0] = (p[0]*boost::math::quantile(lognormDistr, sum_share));
    //    compute the rest guesses
        for(int i=1; i<p.size(); ++i){
            sum_share -= shares_data[i-1];
            guess[i] = (guess[i-1] + (p[i] - p[i-1])*boost::math::quantile(lognormDistr, sum_share));
        }
        return guess;
    }

}
