#include "pcm_market_share.hpp"
#include <iostream>
#include <Dense>
#include "Core"
#include <boost/math/distributions/lognormal.hpp>
using namespace std;
using namespace pcm_share;
int main(){
    int num_prod = 5, num_x_dim = 3;
    Eigen::ArrayXd delta_bar = Eigen::ArrayXd::Zero(num_prod);
    Eigen::MatrixXd jacobian;
    auto p = delta_bar*delta_bar;
    for(int i = 0; i< num_prod; ++i){
        delta_bar[i] = i;
    }
    
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_x_dim);
    cout<<sigma_x<<endl;

    Eigen::ArrayXXd x = Eigen::ArrayXXd::Random(num_prod, num_x_dim);
    cout<<x<<endl;

    Eigen::ArrayXd weights;
    Eigen::ArrayXXd grid;
    pcm_share::generate_tasmanian_global_grid(num_x_dim, 5, grid, weights);
    
    auto un_sh = pcm_share::unc_share(delta_bar, x, p, 1, sigma_x, grid, weights, jacobian);
    cout<<"shares\n"<<un_sh<<"\njacobian\n"<< jacobian<<endl;

    return 0;

}