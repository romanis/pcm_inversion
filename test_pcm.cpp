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
    Eigen::ArrayXd p(num_prod);
    Eigen::MatrixXd jacobian;
    
    for(int i = 0; i< num_prod; ++i){
        delta_bar[i] = i+1;
        p[i]=(i+1)*(i+1);
    }
    
    Eigen::ArrayXd sigma_x = Eigen::ArrayXd::Ones(num_x_dim);
    cout<<sigma_x<<endl;

    Eigen::MatrixXd x = Eigen::MatrixXd::Random(num_prod, num_x_dim);
    cout<<x<<endl;

    Eigen::ArrayXd weights;
    Eigen::ArrayXXd grid;
    pcm_share::generate_tasmanian_global_grid(num_x_dim, 6, grid, weights);
    auto un_sh = pcm_share::unc_share(delta_bar, x, p, 1, sigma_x, grid, weights, jacobian);
    delta_bar[0] += 1e-5;
    cout<<delta_bar<<endl<<endl<<p<<endl;
    auto un_sh1= pcm_share::unc_share(delta_bar, x, p, 1, sigma_x, grid, weights, jacobian);

    cout<<"shares\n"<<un_sh<<"\njacobian\n"<< jacobian<<endl<< (un_sh1 - un_sh)*1e5;

    return 0;

}