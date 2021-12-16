#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>
#include "pcm_market_share.hpp"
#include "market_inversion.hpp"
#include "grid_generator.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"

// struct that is going to take cara of the pcm computations
using namespace Eigen;

int main(){

    int num_prod = 50, num_x_dim = 4; 
    double min_admissible_share = 0.001, sigma_p = 1;
    
    Eigen::ArrayXd delta_bar = Eigen::ArrayXd::Zero(num_prod);
    Eigen::ArrayXd p(num_prod);
    Eigen::MatrixXd jacobian;

    for(int i = 0; i< num_prod; ++i){
        delta_bar[i] = i+1;
        p[i]=pow(i+1, 1.2);
    }
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_x_dim);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_x_dim);
    Eigen::ArrayXd weights;
    Eigen::ArrayXXd grid;
    generate_tasmanian_global_grid(num_x_dim, 6, grid, weights);
    auto un_sh = pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    while((un_sh < min_admissible_share).any()){
        for(int i=0; i< num_prod; ++i){
            if(un_sh[i] < min_admissible_share){
                delta_bar[i] += 0.01*std::abs(delta_bar[0] - delta_bar[num_prod-1]);
            }
        }
        un_sh = pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    }
    std::cout<<"unc share\n"<<un_sh<<std::endl<<"sum of all shares " <<un_sh.sum()<<std::endl;

    share_inversion::pcm_parameters param(x, p, 2.0, sigma_x, grid, weights, un_sh);
    
    
    auto eigen_solution = share_inversion::invert_shares(param);

    Eigen::ArrayX4d sh(num_prod, 4);
    
    auto shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    sh <<shares_at_optimum, un_sh, shares_at_optimum-un_sh, eigen_solution;
    std::cout<<"shares at optimum\n" << sh << std::endl;
    return 0;
}