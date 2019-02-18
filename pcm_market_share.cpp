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
#include <cmath>
#include <math.h>
#include <algorithm>    // std::is_sorted, std::prev_permutation
#include <array>  
#include <boost/math/distributions/lognormal.hpp>
#include <omp.h>

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
//    cout<<"num points " <<pow(n+1,dim)<<endl;
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
//    for(i=0; i<pow(n+1,dim); ++i){
//        cout<<weights2[i]<<" ";
//        for(auto it : grid1[i]){
//            cout<<it<<" ";
//        }
//        cout<<endl;
//    }
    
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
        cout<<"endpoints, deltas and prices\n";
        
        for(auto t : ep_tmp){
            cout<< t << "\t";
        }
        cout<<"\ndeltas\n";
        for(auto t: delta){
            cout<<t<<"\t";
        }
        cout<<"\nprices\n";
        for(auto t: p){
            cout<<t<<"\t";
        }
        cout<<endl;
        throw runtime_error("endpoints are not sorted properly. check which goods go to conditional market share");
    }
    
//    creadte lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);

//    if(endpoints[0] > 1e5){
//        cout<<std::scientific;
//        cout<< "endpoint if too large " << endpoints[0]<<" " << delta[0] << " " <<p[0]<<endl;
//    }
    
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
        cout<<"endpoints, deltas and prices\n";
        
        for(auto t : ep_tmp){
            cout<< t << "\t";
        }
        cout<<"\ndeltas\n";
        for(auto t: delta){
            cout<<t<<"\t";
        }
        cout<<"\nprices\n";
        for(auto t: p){
            cout<<t<<"\t";
        }
        cout<<endl;
        throw runtime_error("endpoints are not sorted poperly. check which goods go to conditional market share");
    }
    
//    creadte lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);
//    put down market shares
//    cout<<endl<<endpoints[0]<<" " <<delta[0]<< " " <<p[0]<<endl;
    con_share.push_back(boost::math::cdf(lognormDistr,endpoints[0]));
    for(int i=1; i< delta.size(); ++i){
//        push back cdf at new point
//        cout<<endpoints[i]<<endl;
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
//            cout<< "size of jacobian "<< jacobian.size()<<endl;
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

std::vector<double> pcm_market_share::unc_share(std::vector<double> delta_bar, std::vector<std::vector<double>> x, std::vector<double> p, double sigma_p, std::vector<double> sigma_x ){
/*
 * delta_bar is average quality of each good size Nx1 N- number of products
 * x - values of heterogeneity size Kx1 K - number of characteristics of horizontal differentiation 
 * p - vector of prices size Nx1
 * sigma_p - standard deviation of log of price sensitivity
 * sigma_x - vector of standard deviation of each horizontal differentiation.
 */
    vector<double> un_share(delta_bar.size(), 0.0);
//    check that vector x[0] and sigma_x are the same dimension
    if(x[0].size() != sigma_x.size()){
        throw runtime_error("size of x[0] is different from size of sigma_x");
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
            if(delta_cond[k] > 1e5){
                cout<<std::scientific;
                cout<<"too large delta " << delta_cond[k]<< " delta bar "<< delta_bar[k]<< " sigma x "<<sigma_x[j]<< " x " <<x[k][j] << " grid " <<grid[i][j]<<endl;
            }
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
                    up_alpha.push_back((delta_cond[j]-delta_cond[k]) * 1e20);
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
                    lower_alpha.push_back((delta_cond[j]-delta_cond[k])*(-1e20));
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
            for(auto i_positive : ind){
                delta_positive.push_back(delta_cond[i_positive]);
                p_positive.push_back(p[i_positive]);
                if(delta_cond[i_positive] > 1e5){
                    cout<<std::scientific;
                    cout<<"too large delta " << delta_cond[i_positive]<< " "<<i_positive<< " " <<delta_bar[i_positive]<<endl;
                }
            }
            
            vector<double> shares_tmp = cond_share(delta_positive,p_positive, sigma_p);

//            add shares to corresponding dimensions of un_share
            int num_share=0;

            
            for(auto i_dim : ind){
                omp_set_lock(&locks[i_dim]);
                un_share[i_dim]+=weights[i]*shares_tmp[num_share++];
                omp_unset_lock(&locks[i_dim]);
            }
            
        }
    }
//    cout<<"share"<<endl;
//        for(auto it: un_share){
//            cout<< it<<endl;
//        }
    return un_share;
}


std::vector<double> pcm_market_share::unc_share(std::vector<double> delta_bar, std::vector<std::vector<double>> x, std::vector<double> p, double sigma_p, std::vector<double> sigma_x, std::vector<std::vector<double> > & jacobian){
/*
 * delta_bar is average quality of each good size Nx1 N- number of products
 * x - values of heterogeneity size NxK K - number of characteristics of horizontal differentiation 
 * p - vector of prices size Nx1
 * sigma_p - standard deviation of log of price sensitivity
 * sigma_x - vector of standard deviation of each horizontal differentiation.
 * jacobian - jacibian of the system with respect to deltas
 */
    vector<double> un_share(delta_bar.size(), 0.0);
//    clean supplied jacobian
    jacobian.clear();
//    fill jacobian with zeros
    for(int i=0; i< delta_bar.size(); ++ i){
        vector<double> tmp(delta_bar.size(), 0.0);
        jacobian.push_back(tmp);
    }
    
//    check that vector x[0] and sigma_x are the same dimension
    if(x[0].size() != sigma_x.size()){
        throw runtime_error("size of x[0] is different from size of sigma_x");
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
                    up_alpha.push_back((delta_cond[j]-delta_cond[k]) * 1e20);
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
                    lower_alpha.push_back((delta_cond[j]-delta_cond[k])*(-1e20));
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

std::vector<double> pcm_market_share::initial_guess(){
    vector<double> guess;
//    calculate initial guess based on undisturbed model
//    delta_0(1) = p_sim(1) * logninv(sum(share),0,sigma_p);
    double sum_share=0;
    for(auto share: shares_data){
        sum_share += share;
    }
    //    creadte lognormal distr
    boost::math::lognormal lognormDistr(0, sigma_p);
//    cout<<boost::math::quantile(lognormDistr, sum_share)<<endl;
//    compute the first delta guess
    guess.push_back(p[0]*boost::math::quantile(lognormDistr, sum_share));
//    compute the rest guesses
    for(int i=1; i<shares_data.size(); ++i){
        sum_share -=shares_data[i-1];
        guess.push_back(guess[i-1] + (p[i] - p[i-1])*boost::math::quantile(lognormDistr, sum_share));
    }
    return guess;
}

void pcm_market_share::get_traction(){
    vector<double> sch_start = this->unc_share(this->getXInitial(), x, p, sigma_p, sigmax);
//    if all of market shares are greater than 1e-4, return
    if(all_of(sch_start.begin(), sch_start.end(),[](double it){return it > 0;})){
        return;
    }
    
    while(!(all_of(sch_start.begin(), sch_start.end(),[](double it){return it > 0;}))){
//        increase all coordinates that have zero market share
        vector<double> X = this->getXInitial();
        double max = *(max_element(X.begin(), X.end()));
        double min = *(min_element(X.begin(), X.end()));
        for(int i=0; i<shares_data.size(); ++i){
            if(!(sch_start[i]>0)){
                X[i] += 1e-2*(max-min);
            }
        }
//        update market shares
        this->setXInitial(X);
        sch_start = this->unc_share(X, x, p, sigma_p, sigmax);
//        cout<< "_____________________________"<<endl;
//        std::cout.precision(4);
//        std::cout << std::fixed;
//        for(auto it : sch_start){
//            cout<<it << " ";
//        }
//        cout<<endl;
//        for(auto it : X){
//            cout<<it << " ";
//        }
//        cout<<endl;
    }
    return;
}

double pcm_market_share::relax_til_solved(std::vector<double> & solution, std::vector<double> starting_point){
    double factor=1;
    this->decrease_sigma_x(1.1);
    factor /=1.1;
    this->setXInitial(this->initial_guess());
    this->get_traction();
    this->setXInitial(starting_point);
    knitro::KTRSolver solver1(this, KTR_GRADOPT_EXACT, KTR_HESSOPT_BFGS);
    solver1.setParam(KTR_PARAM_ALG, 2); // 2 = CG algorithm 3 = active set
    solver1.setParam(KTR_PARAM_MAXIT, 300);
    solver1.setParam(KTR_PARAM_FTOL, 1e-12);
    solver1.setParam(KTR_PARAM_XTOL, 1e-8);
    solver1.setParam(KTR_PARAM_OPTTOL, 1e-7);
    solver1.setParam(KTR_PARAM_DERIVCHECK, 0);
    
    int result = solver1.solve();
    solution = solver1.getXValues();
    while(result != 0){
        this->decrease_sigma_x(1.1);
        factor /=1.1;
        this->setXInitial(solution);
        this->get_traction();
//        knitro::KTRSolver solver(this, KTR_GRADOPT_EXACT, KTR_HESSOPT_BFGS);
        solver1.setParam(KTR_PARAM_ALG, 2); // 2 = CG algorithm 3 = active set
        solver1.setParam(KTR_PARAM_MAXIT, 300);
        solver1.setParam(KTR_PARAM_FTOL, 1e-12);
        solver1.setParam(KTR_PARAM_XTOL, 1e-8);
        solver1.setParam(KTR_PARAM_OPTTOL, 1e-7);
        solver1.setParam(KTR_PARAM_DERIVCHECK, 0);
        result = solver1.solve();
        solution = solver1.getXValues();
    }
    
    return factor;
}

bool pcm_market_share::solve_for_delta(){
    bool solved = false;
    double factor = 1.5; // contraction of the grid after each unsuccessful iteration
    int count = 0; // count number of contractions
//    set initial guess for 
    this->setXInitial(this->initial_guess());
    this->get_traction();
//    try to solve 
    knitro::KTRSolver solver1(this, KTR_GRADOPT_EXACT, KTR_HESSOPT_BFGS);
    solver1.setParam(KTR_PARAM_ALG, 2); // 2 = CG algorithm 3 = active set
    int num_iter = x.size()*5;
    solver1.setParam(KTR_PARAM_MAXIT, num_iter); // number of iterations 3 times bigger than the number of products.
    solver1.setParam(KTR_PARAM_FTOL, 1e-10);
    solver1.setParam(KTR_PARAM_XTOL, 1e-12);
    solver1.setParam(KTR_PARAM_OPTTOL, 1e-7);
    solver1.setParam(KTR_PARAM_DERIVCHECK, 0);
    solver1.setParam(KTR_PARAM_BAR_MURULE, 2); // set murule to adaptive
    solver1.setParam(KTR_PARAM_PAR_NUMTHREADS, 20);
    solver1.setParam(KTR_PARAM_LINSOLVER, 6); // intel lin solver
//    solver1.setParam(KTR_PARAM_PAR_LSNUMTHREADS, 5);
//    solver1.setParam(KTR_PARAM_PAR_BLASNUMTHREADS,5);
    solver1.setParam(KTR_PARAM_FEASTOL, 1e-10);
    solver1.setParam(KTR_PARAM_MULTISTART, 0);
    solver1.setParam(KTR_PARAM_PAR_CONCURRENT_EVALS,1); //concurrent evaluations. wired, but without concurrent evaluations it takes less time
    solver1.setParam(KTR_PARAM_OUTLEV, 0); // suppress output
    
    int result = solver1.solve();
    std::vector<double> solution = solver1.getXValues();
//    decrease the sigma while solution is not found
    while(result != 0){
//        cout<<"decreasing sigma to solve for easier problem number of times "<<count+1;
        this->decrease_sigma_x(1/factor);
//        cout<<"sigma [0] is now\t" << sigmax[0]<<"\t tolerance violation is " <<endl;
        this->get_traction();
        result = solver1.solve();
        count++;
    }
//    cout<<"increasing sigma back\n";
//    increase the sigma back and keep searching for the solution
    for(count; count>0; --count){
        this->increase_sigma_x(1/factor);
        this->get_traction();
        result = solver1.solve();
    }
    this->setXInitial(solver1.getXValues());
    
    return result==0;
}