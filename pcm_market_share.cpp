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

std::vector<double> pcm_market_share::cond_share(std::vector<double> delta, std::vector<double> p, double sigma_p){
    vector<double> con_share;
    
//    compute points of being indifferent
    
    return con_share;
}