/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pcm_market_share.h
 * Author: Roman
 *
 * Created on February 13, 2018, 5:07 PM
 */

#ifndef PCM_MARKET_SHARE_H
#define PCM_MARKET_SHARE_H
#include <vector>
#include <map>
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"

class pcm_market_share{
private:
    int dimension;
    std::vector<std::vector<double> > grid;
    std::vector<double> weights;
    
public:
    pcm_market_share ();
    bool set_grid(std::vector<std::vector<double> > grid1, std::vector<double> weights1);
    bool set_grid(int dim, int n); // sets up a grid to integrate with respect to standard normal measure gauss hermite quadrature with dimension and number of points in each dimension
    
    
};


std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p, std::vector<std::vector<double> > & jacobian ); // computes market share of vertical model (conditional on draw of heterogeneity. 
std::vector<double> cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p ); //does the same but does not calculate jacobian

inline void print_jacobian(std::vector<std::vector<double> > jac){
    for(auto p : jac){
        for(auto b : p){
            std::cout<<b<<"\t";
        }
        std::cout<<std::endl;
    }
}
inline pcm_market_share::pcm_market_share() : dimension(1)  { 
//    set up grid and weights
    std::vector<double> point;
    point.push_back(0);
    grid.push_back(point);
    weights.push_back(1);
}

#endif /* PCM_MARKET_SHARE_H */