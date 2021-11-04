#include "pcm_market_share.hpp"
#include <iostream>
#include <Dense>
#include <boost/math/distributions/lognormal.hpp>
using namespace std;
using namespace pcm_share;
int main(){
    Eigen::ArrayXd delta(3);
    Eigen::ArrayXd p(3);
    double sigma_p;
    Eigen::MatrixXd jacobian(3,3);
    delta << 1, 2, 3;
    p << 1, 4, 9;
    sigma_p = 1;
    boost::math::lognormal lognormDistr(0, sigma_p);
    auto shares_initial = cond_share(delta, p, sigma_p, jacobian);
    cout<<"shares \n" << shares_initial <<endl;
    cout<<"jacobian\n" << jacobian<<endl;
    delta[2] += 1e-5;
    cout<<"derivative\n" << (cond_share(delta, p, sigma_p, jacobian) - shares_initial).transpose()*1e5<<endl;

}