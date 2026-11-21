#include "pcm_market_share.hpp"
#include <iostream>
#include <map>
#include <vector>
#include <utility>
#include <cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Core>



using namespace std;
using namespace Eigen;

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
        pcm_geometry::vertical_share(ind, delta, p, sigma_p, 0.0, con_share, &jacobian, nullptr, nullptr, nullptr, true);
        return con_share;
    }


    Eigen::ArrayXd cond_share(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p,  bool check_positive_shares){
    //    conditional on heterogeneity market shares of the pure vertical model
        const vector<int> ind = frontier_of(delta, p, check_positive_shares);
        Eigen::ArrayXd con_share = ArrayXd::Zero(delta.size());
        pcm_geometry::vertical_share(ind, delta, p, sigma_p, 0.0, con_share, nullptr, nullptr, nullptr, nullptr, true);
        return con_share;
    }


    Eigen::ArrayXd initial_guess(const Eigen::ArrayXd& shares_data, const Eigen::ArrayXd& p, double sigma_p){
        return pcm_geometry::initial_guess(shares_data, p, sigma_p);
    }

}
