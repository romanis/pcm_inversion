#include "pcm_market_share.hpp"
#include<iostream>
#include<map>
#include<vector>
// #include "TasmanianSparseGrid.hpp"
#include <utility>
#include <cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <boost/math/distributions/lognormal.hpp>
#include <omp.h>
#include <Dense>
#include <Core>

using namespace std;
using namespace Eigen;
using Eigen::indexing::last;

namespace pcm_share{
    Eigen::ArrayXd unc_share(const ArrayXd& delta_bar, const ArrayXXd& x, const ArrayXd& p, double sigma_p, const ArrayXd& sigma_x, const ArrayXXd& grid, const ArrayXd & weights, MatrixXd & jacobian)
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
    
    //    create vector of locks 
    vector<omp_lock_t> locks;

    for(int draw=0; draw< delta_bar.size(); ++draw){
        omp_lock_t writelock;
        omp_init_lock(&writelock);
        locks.push_back(writelock);
    }
    
//    loop over all points in the grid
#pragma omp parallel for num_threads(4) schedule(dynamic,1)
    for(int draw=0; draw<weights.size(); ++draw){
//        calculate the conditional quality
//        delta_hat = delta + sigma_x*x*nu(i,:);
        ArrayXd delta_cond = delta_bar + (x.matrix()*(sigma_x.transpose()*grid.row(draw)).matrix()).array();
        
//        determine indexes of goods that have positive market shares
        vector<int> ind;
        for(int j=0; j<delta_bar.size(); ++j){
//            calculate upper bound on price elasticity of people who will choose product j
            ArrayXd u_alpha = (delta_cond[j] - delta_cond(seq(0, j-1))) / (p[j] - p(seq(0, j-1)));
            double min_upper = std::min(u_alpha.minCoeff(), delta_cond[j]/p[j]);
            
//            create lower bound on price elasticity of those who buy this product
            vector<double> lower_alpha;
            for(int k=delta_cond.size()-1; k>j; --k){
                if(p[j] != p[k]){
                    lower_alpha.push_back((delta_cond[j]-delta_cond[k])/(p[j]-p[k]));
                }
//                else, push back a large negative number
                else{
                    lower_alpha.push_back((delta_cond[j]-delta_cond[k])*(-1e10));
                }
            }
            lower_alpha.push_back(0);
            ArrayXd l_alpha = (delta_cond[j] - delta_cond(seq(j+1, last))) / (p[j] - p(seq(j+1, last)));
            double max_lower = std::max(l_alpha.maxCoeff(), 0.0);
            
//            if there is slack between up and lower, it has positive market share
            if(min_upper > max_lower){
                ind.push_back(j);
            }
            
        }

//        if there is a product with positive market share
        if(ind.size() > 0){
//            pick those deltas and prices that correspond to products with positive market shares
            MatrixXd jacobian_tmp;
            ArrayXd delta_positive = delta_cond(ind);
            ArrayXd p_positive = p(ind);
            
            
            ArrayXd shares_tmp = cond_share(delta_positive,p_positive, sigma_p, jacobian_tmp);

//            add shares to corresponding dimensions of un_share
            int num_share=0, num_i=0, num_j=0;

            un_share(ind) += shares_tmp*weights[draw];
            
            jacobian(ind, ind) += jacobian_tmp;
            
        }
    }
    cout<<"share"<<endl;
       for(auto it: un_share){
           cout<< it<<endl;
       }
    return un_share;
    }


    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, Eigen::MatrixXd & jacobian){
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
        
    //    compute points of being indifferent
        Eigen::ArrayXd endpoints = ArrayXd::Zero(delta.size()+1);
    //    first endpoint is delta(0)/p(0)
        endpoints[0] = (delta[0]/p[0]);
    //    the other endpoints can be calculated diffrently

        endpoints(seq(1, last-1)) = (delta(seq(1, last)) - delta(seq(0, last-1)) ) / ( p(seq(1, last)) - p(seq(0, last-1)) ) ; // note that the last endpoint stays zero
        
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
        for(int i=0; i< delta.size(); ++i){
            double current_cdf = boost::math::cdf(lognormDistr,endpoints[i+1]);
            con_share[i] = previous_cdf - current_cdf;
            previous_cdf = current_cdf;
        }
        
    //    calculate jacobian

        {
            jacobian = MatrixXd::Zero(delta.size(), delta.size());
    //        if only one product, simple
            if(delta.size() == 1){
                jacobian(0,0) = (boost::math::pdf(lognormDistr, endpoints[0]))/p[0];
    //            cout<<"jacobian "<<jacobian[0][0]<<endl;
                
            }
            else{
                
    //            hardcode derivatives of 1,1 1,2 and end,end-1 and end,end
                jacobian(0,0) = (boost::math::pdf(lognormDistr, endpoints[0]))/p[0];
    //            cout<< "size of jacobian "<< jacobian.size()<<endl;
    //            all the rest calculate algirithmically
                for(int i=1; i<delta.size(); ++i){
                    double val = (boost::math::pdf(lognormDistr, endpoints[i]))/(p[i]  -p[i-1]);
                    jacobian(i,i-1) -= val;
                    jacobian(i-1,i) -= val;
                    jacobian(i,i)   += val;
                    jacobian(i-1, i-1) += val;
                }
            }
        }
        return con_share;
    }

}