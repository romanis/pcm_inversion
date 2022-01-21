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
                int num_share=0, num_i=0, num_j=0;

                un_share(ind) += shares_tmp*weights[draw];
                
                jacobian(ind, ind) += jacobian_tmp*weights[draw];
                
            }
        }
        // cout<<"share"<<endl;
        //    for(auto it: un_share){
        //        cout<< it<<endl;
        //    }
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
                int num_share=0, num_i=0, num_j=0;

                un_share(ind) += shares_tmp*weights[draw];
                                
            }
        }
        // cout<<"share"<<endl;
        //    for(auto it: un_share){
        //        cout<< it<<endl;
        //    }
        return un_share;
    }


    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, Eigen::MatrixXd & jacobian,  bool check_positive_shares){
    //    this function calculates conditional on heterogeneity market shares of pure vertical model
        Eigen::ArrayXd con_share = ArrayXd::Zero(delta.size());
    //    make sure prices are sorted
        if(!std::is_sorted(p.begin(), p.end())){
            throw std::runtime_error("price vector not sorted");
    //        return con_share;
        }
        if(delta.size() != p.size()){
            throw runtime_error("price vector and delta sizes differ");
        }

        vector<int> ind;
        if(check_positive_shares){
            ind = index_positive_shares(delta, p);
        }else{
            for(int i = 1; i<delta.size(); i++){
                ind.push_back(i);
            }
        }
        Eigen::ArrayXd positive_shares = ArrayXd::Zero(ind.size());
        Eigen::ArrayXd delta_positive = delta(ind);
        Eigen::ArrayXd p_positive = p(ind);
        
    //    compute points of being indifferent
        Eigen::ArrayXd endpoints = ArrayXd::Zero(positive_shares.size()+1);
    //    first endpoint is delta(0)/p(0)
        endpoints[0] = (delta_positive[0]/p_positive[0]);
    //    the other endpoints can be calculated diffrently

        endpoints(seq(1, last-1)) = (delta_positive(seq(1, last)) - delta_positive(seq(0, last-1)) ) / ( p_positive(seq(1, last)) - p_positive(seq(0, last-1)) ) ; // note that the last endpoint stays zero

        
    //    need to check that endpoints are sorted in reverse order
        ArrayXd ep_tmp(endpoints);
        reverse(ep_tmp.begin(), ep_tmp.end());
        if(!is_sorted(ep_tmp.begin(), ep_tmp.end())){
            throw runtime_error("endpoints are not sorted poperly. check which goods go to conditional market share");
        }
        // cout<<" endpoints \n" << endpoints<<endl;
        //    creadte lognormal distr
        boost::math::lognormal lognormDistr(0, sigma_p);
        //    calculate market shares
        double previous_cdf = boost::math::cdf(lognormDistr,endpoints[0]);
        for(int i=0; i< ind.size(); ++i){
            double current_cdf = boost::math::cdf(lognormDistr,endpoints[i+1]);
            positive_shares[i] = previous_cdf - current_cdf;
            previous_cdf = current_cdf;
        }
        con_share(ind) = positive_shares;
        
        //    calculate jacobian

        {
            jacobian = MatrixXd::Zero(delta.size(), delta.size());
            Eigen::MatrixXd jacobian_positive = MatrixXd::Zero(ind.size(), ind.size());
        //        if only one product, simple
            if(ind.size() == 1){
                jacobian_positive(0,0) = (boost::math::pdf(lognormDistr, endpoints[0]))/p_positive[0];
        //            cout<<"jacobian "<<jacobian[0][0]<<endl;
                
            }
            else{
                
        //            hardcode derivatives of 1,1 1,2 and end,end-1 and end,end
                jacobian_positive(0,0) = (boost::math::pdf(lognormDistr, endpoints[0]))/p_positive[0];
        //            cout<< "size of jacobian "<< jacobian.size()<<endl;
        //            all the rest calculate algirithmically
                for(int i=1; i<ind.size(); ++i){
                    double val = (boost::math::pdf(lognormDistr, endpoints[i]))/(p_positive[i]  -p_positive[i-1]);
                    jacobian_positive(i,i-1) -= val;
                    jacobian_positive(i-1,i) -= val;
                    jacobian_positive(i,i)   += val;
                    jacobian_positive(i-1, i-1) += val;
                }
            }
            jacobian(ind,ind) = jacobian_positive;
        }
        return con_share;
    }


    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p,  bool check_positive_shares){
    //    this function calculates conditional on heterogeneity market shares of pure vertical model
        Eigen::ArrayXd con_share = ArrayXd::Zero(delta.size());
    //    make sure prices are sorted
        if(!std::is_sorted(p.begin(), p.end())){
            throw std::runtime_error("price vector not sorted");
    //        return con_share;
        }
        if(delta.size() != p.size()){
            throw runtime_error("price vector and delta sizes differ");
        }

        vector<int> ind;
        if(check_positive_shares){
            ind = index_positive_shares(delta, p);
        }else{
            for(int i = 1; i<delta.size(); i++){
                ind.push_back(i);
            }
        }
        Eigen::ArrayXd positive_shares = ArrayXd::Zero(ind.size());
        Eigen::ArrayXd delta_positive = delta(ind);
        Eigen::ArrayXd p_positive = p(ind);
        
    //    compute points of being indifferent
        Eigen::ArrayXd endpoints = ArrayXd::Zero(positive_shares.size()+1);
    //    first endpoint is delta(0)/p(0)
        endpoints[0] = (delta_positive[0]/p_positive[0]);
    //    the other endpoints can be calculated diffrently

        endpoints(seq(1, last-1)) = (delta_positive(seq(1, last)) - delta_positive(seq(0, last-1)) ) / ( p_positive(seq(1, last)) - p_positive(seq(0, last-1)) ) ; // note that the last endpoint stays zero

        
    //    need to check that endpoints are sorted in reverse order
        ArrayXd ep_tmp(endpoints);
        reverse(ep_tmp.begin(), ep_tmp.end());
        if(!is_sorted(ep_tmp.begin(), ep_tmp.end())){
            throw runtime_error("endpoints are not sorted poperly. check which goods go to conditional market share");
        }
        // cout<<" endpoints \n" << endpoints<<endl;
    //    creadte lognormal distr
        boost::math::lognormal lognormDistr(0, sigma_p);
    //    calculate market shares
        double previous_cdf = boost::math::cdf(lognormDistr,endpoints[0]);
        for(int i=0; i< ind.size(); ++i){
            double current_cdf = boost::math::cdf(lognormDistr,endpoints[i+1]);
            positive_shares[i] = previous_cdf - current_cdf;
            previous_cdf = current_cdf;
        }
        con_share(ind) = positive_shares;
        
        return con_share;
    }

    

    Eigen::ArrayXd initial_guess(const Eigen::ArrayXd& shares_data, const Eigen::ArrayXd& p, double sigma_p){
        ArrayXd guess(p.size());
    //    calculate initial guess based on undisturbed model
    //    delta_0(1) = p_sim(1) * logninv(sum(share),0,sigma_p);
        double sum_share=shares_data.sum();
        
        //    creadte lognormal distr
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