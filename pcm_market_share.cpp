#include "pcm_market_share.hpp"
#include<iostream>
#include<map>
#include<vector>
#include "TasmanianSparseGrid.hpp"
#include <utility>
#include <cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <boost/math/distributions/lognormal.hpp>
#include <omp.h>
using namespace std;
using namespace Eigen;

namespace pcm_share
{
    std::vector<double> unc_share(const VectorXd& delta_bar, const MatrixXd& x, const VectorXd& p, double sigma_p, const VectorXd& sigma_x, const MatrixXd& grid, const VectorXd & weights, MatrixXd & jacobian)
    {
        vector<double> un_share(delta_bar.size(), 0.0);
        // set jacobian to zero matrix
        jacobian = MatrixXd::Zero(delta_bar.size(), delta_bar.size());
    
//    check that vector x[0] and sigma_x are the same dimension
    if(x.ncol() != sigma_x.size()){
        throw runtime_error("number of columns in x is different from size of sigma_x");
    }
//    check that all x vectors have the same length
    
    if (x[0].size() != grid[0].size()){
        throw runtime_error("size of x[0] is different from size of grid");
    }
    if (x.size() != delta_bar.size()){
        throw runtime_error("size of x is different from size of grid");
    }
    
    //    create vector of locks 
    vector<omp_lock_t> locks;

    for(int i=0; i< delta_bar.size(); ++i){
        omp_lock_t writelock;
        omp_init_lock(&writelock);
        locks.push_back(writelock);
    }
    
//    loop over all points in the grid
#pragma omp parallel for num_threads(4) schedule(dynamic,1)
    for(int i=0; i<weights.size(); ++i){
//        calculate the conditional quality
//        delta_hat = delta + sigma_x*x*nu(i,:);
        vector<double> delta_cond(delta_bar);
        for(int k=0; k<delta_cond.size(); ++k){
            for(int j=0; j<x[0].size(); ++j){
                delta_cond[k] += sigma_x[j]*x[k][j]*grid[i][j];
            }
        }

        
//        determine indexes of goods that have positive market shares
        vector<int> ind;
        for(int j=0; j<delta_bar.size(); ++j){
//            calculate upper bound on price elasticity of people who will choose product j
            vector<double> up_alpha;
            for(int k=0; k<j; ++k){
//                if p[j] != p[k], use simple formula
                if(p[j] != p[k]){
                    up_alpha.push_back((delta_cond[j]-delta_cond[k])/(p[j]-p[k]));
                }
//                else push back the difference times a big number
                else{
                    up_alpha.push_back((delta_cond[j]-delta_cond[k]) * 1e10);
                }
            }
            up_alpha.push_back(delta_cond[j]/p[j]);
            
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
            

            
//            if there is slack between up and lower, it has positive market share
            if(*min_element(up_alpha.begin(), up_alpha.end()) > *max_element(lower_alpha.begin(), lower_alpha.end())){
                ind.push_back(j);
            }
            
        }

//        if there is a product with positive market share
        if(ind.size() > 0){
//            pick those deltas and prices that correspond to products with positive market shares
            vector<double> delta_positive, p_positive;
            vector<vector<double>> jacobian_tmp;
            
            for(auto i_positive : ind){
                delta_positive.push_back(delta_cond[i_positive]);
                p_positive.push_back(p[i_positive]);
            }
            vector<double> shares_tmp = cond_share(delta_positive,p_positive, sigma_p, jacobian_tmp);

//            add shares to corresponding dimensions of un_share
            int num_share=0, num_i=0, num_j=0;

            
            for(auto i_dim : ind){
                omp_set_lock(&locks[i_dim]);
                un_share[i_dim]+=weights[i]*shares_tmp[num_share++];
                omp_unset_lock(&locks[i_dim]);
            }
            
            
//            add jacobian elements to their positions
            for(auto i_dim :ind){
                omp_set_lock(&locks[i_dim]);
                for (auto j_dim : ind){
                    jacobian[i_dim][j_dim] += weights[i]*jacobian_tmp[num_i][num_j];
                    ++num_j;
                }
                omp_unset_lock(&locks[i_dim]);
                num_j = 0;
                num_i++;
            }
            
        }
    }
//    cout<<"share"<<endl;
//        for(auto it: un_share){
//            cout<< it<<endl;
//        }
    return un_share;
    }
}