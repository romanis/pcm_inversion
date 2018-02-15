/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pcm_market_share.cpp
 * Author: Roman
 * 
 * Created on February 13, 2018, 5:07 PM
 */

#include <pcm_market_share.h>
#include<iostream>
#include<map>
#include<vector>
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"
#include <utility>
#include <c++/5.3.1/cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <boost/math/distributions/lognormal.hpp>

using namespace std;
using namespace TasGrid;

bool pcm_market_share::set_grid(std::vector<std::vector<double> > grid1, std::vector<double> weights1){
//    check consistency of grid and weight
    if(grid1.size() != weights1.size()) {
        cout << "dimension of grid is inconsistent with dimension of weights ";
        return false;
    }
    dimension = grid1[0].size();
    
    int i=0;
    for(int point = 0; point != grid1.size(); ++point){
        i++;
        if(dimension != grid1[point].size()){
            cout << "dimension of one of the points in grid is inconsistent with dimension of other points " << grid1[point].size()<<" " <<i<<endl;
            for(auto it : grid1[point]){
                cout<<it<<" ";
            }
            return false;
        }
    }
    
    grid = grid1;
    weights = weights1;
    return true;
}

bool pcm_market_share::set_grid(int dim, int n){
//    use tasmanian to calculate points and weights
    TasmanianSparseGrid grid;
    
    grid.makeGlobalGrid(dim,1,n,type_tensor,rule_gausshermite);
    double * weights1 = grid.getQuadratureWeights();
    double* coordinates = grid.getPoints();
    std::vector<std::vector<double> > grid1;
    std::vector<double> weights2;
    int i = 0;
    cout<<"num points " <<pow(n+1,dim)<<endl;
//    there are (n+1)^dim points it is stacked in points vector
    for(int point_number = 0; point_number < pow((n+1),dim); ++point_number){
        std::vector<double> point;
//        extract n+1 points from array of coordinates
        for(int j = 0; j< dim; j++){
            point.push_back((coordinates[i]*M_SQRT2));
            i++;
        }
        weights2.push_back((weights1[point_number]/pow(std::sqrt(M_PI),dim)));
        grid1.push_back(point);
    }
//    print points and weights
    for(i=0; i<pow(n+1,dim); ++i){
        cout<<weights2[i]<<" ";
        for(auto it : grid1[i]){
            cout<<it<<" ";
        }
        cout<<endl;
    }
    
    this->set_grid(grid1, weights2);
    return true;
}

std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p){
//    this function calculates conditional on heterogeneity market shares of pure vertical model
    vector<double> con_share;
//    make sure prices are sorted
    if(!std::is_sorted(p.begin(), p.end())){
        throw std::runtime_error("price vector not sorted");
//        return con_share;
    }
    if(delta.size() != p.size()){
        throw runtime_error("price vector and delta sizes differ");
    }
    
//    compute points of being indifferent
    vector<double> endpoints;
//    first endpoint is delta(0)/p(0)
    endpoints.push_back(delta[0]/p[0]);
//    the other endpoints can be calculated diffrently
    for(int i=1; i<delta.size(); ++i){
        endpoints.push_back((delta[i]-delta[i-1])/(p[i]-p[i-1]));
    }
//    need to check that endpoints are sorted in reverse order
    vector <double> ep_tmp(endpoints);
    reverse(ep_tmp.begin(), ep_tmp.end());
    if(!is_sorted(ep_tmp.begin(), ep_tmp.end())){
        throw runtime_error("endpoints are not sorted poperly. check which goods go to conditional market share");
    }
    
//    creadte lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);
//    put down market shares
    con_share.push_back(boost::math::cdf(lognormDistr,endpoints[0]));
    for(int i=1; i< delta.size(); ++i){
//        push back cdf at new point
        con_share.push_back(boost::math::cdf(lognormDistr,endpoints[i]));
//        subtract from previous point the pushed value
        con_share[i-1] -= con_share[i];
    }
//    cout<<"lognorm cdf " << boost::math::cdf(lognormDistr,0.5)<<endl;
//    for(auto it : con_share){
//        cout<<it<<endl;
//    }
    
    
    
    return con_share;
}

std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p, vector<vector<double> > & jacobian){
//    this function calculates conditional on heterogeneity market shares of pure vertical model
    vector<double> con_share;
//    make sure prices are sorted
    if(!std::is_sorted(p.begin(), p.end())){
        throw std::runtime_error("price vector not sorted");
//        return con_share;
    }
    if(delta.size() != p.size()){
        throw runtime_error("price vector and delta sizes differ");
    }
    
//    compute points of being indifferent
    vector<double> endpoints;
//    first endpoint is delta(0)/p(0)
    endpoints.push_back(delta[0]/p[0]);
//    the other endpoints can be calculated diffrently
    for(int i=1; i<delta.size(); ++i){
        endpoints.push_back((delta[i]-delta[i-1])/(p[i]-p[i-1]));
    }
//    need to check that endpoints are sorted in reverse order
    vector <double> ep_tmp(endpoints);
    reverse(ep_tmp.begin(), ep_tmp.end());
    if(!is_sorted(ep_tmp.begin(), ep_tmp.end())){
        throw runtime_error("endpoints are not sorted poperly. check which goods go to conditional market share");
    }
    
//    creadte lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);
//    put down market shares
    con_share.push_back(boost::math::cdf(lognormDistr,endpoints[0]));
    for(int i=1; i< delta.size(); ++i){
//        push back cdf at new point
        con_share.push_back(boost::math::cdf(lognormDistr,endpoints[i]));
//        subtract from previous point the pushed value
        con_share[i-1] -= con_share[i];
    }
    
    
//    cout<<"lognorm cpdf " << endpoints[0]<<" "<< boost::math::pdf(lognormDistr, 1)<<endl;
//    for(auto it : con_share){
//        cout<<it<<endl;
//    }
    
//    calculate jacobian

    {  
//        if only one product, simple
        if(delta.size() == 1){
            vector<double> dp_i(delta.size(),0.0);
            dp_i[0] = (boost::math::pdf(lognormDistr, endpoints[0]))/p[0];
            jacobian.push_back(dp_i);
//            cout<<"jacobian "<<jacobian[0][0]<<endl;
            
        }
        else{
            
//            hardcode derivatives of 1,1 1,2 and end,end-1 and end,end
            vector<double> dp_0(delta.size(),0.0);
            dp_0[0] = (boost::math::pdf(lognormDistr, endpoints[0]))/p[0] + (boost::math::pdf(lognormDistr, endpoints[1]))/(p[1]-p[0]);
            dp_0[1] = -(boost::math::pdf(lognormDistr, endpoints[1]))/(p[1]-p[0]);
            jacobian.push_back(dp_0);
            cout<< "size of jacobian "<< jacobian.size()<<endl;
//            all the rest calculate algirithmically
            for(int i=1; i<delta.size()-1; ++i){
                vector<double> dp_i(delta.size(),0.0);
                dp_i[i-1]   = -(boost::math::pdf(lognormDistr, endpoints[i]))/(p[i]  -p[i-1]);
                dp_i[i]     =  (boost::math::pdf(lognormDistr, endpoints[i]))/(p[i]  -p[i-1]) + (boost::math::pdf(lognormDistr, endpoints[i+1]))/(p[i+1]-p[i]);
                dp_i[i+1]   =                                                                 - (boost::math::pdf(lognormDistr, endpoints[i+1]))/(p[i+1]-p[i]);
                jacobian.push_back(dp_i);
            }
            vector<double> dp_e(delta.size(),0.0);
            dp_e[delta.size()-2] = -(boost::math::pdf(lognormDistr, endpoints[delta.size()-1]))/(p[delta.size()-1]-p[delta.size()-2]);
            dp_e[delta.size()-1] = (boost::math::pdf(lognormDistr, endpoints[delta.size()-1]))/(p[delta.size()-1]-p[delta.size()-2]);
            jacobian.push_back(dp_e);
        }
    }
    return con_share;
}