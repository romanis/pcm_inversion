#include "pcm_market_share.hpp"
#include <iostream>
#include <Dense>
#include "Core"
#include <boost/math/distributions/lognormal.hpp>
#include <chrono>

using namespace std;
using namespace pcm_share;
int main(){
    int num_prod = 30, num_x_dim = 5;
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
    pcm_share::generate_tasmanian_global_grid(num_x_dim, 6, grid, weights);
    auto un_sh = pcm_share::unc_share(delta_bar, x, p, 1, sigma_x, grid, weights, jacobian);
    delta_bar[0] += 1e-5;
    cout<<"number of points\n" <<weights.size()<<endl;
    auto start = chrono::high_resolution_clock::now();
    auto un_sh1= pcm_share::unc_share(delta_bar, x, p, 1, sigma_x, grid, weights, jacobian);
    cout<<"time to compute share and jacobian " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count()<<endl;

    return 0;

}