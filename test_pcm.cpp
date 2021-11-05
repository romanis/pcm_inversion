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
    for(int i = 0; i< num_prod; ++i){
        delta_bar[i] = i;
    }
    cout<<delta_bar<<endl;

    Eigen::ArrayXXd x = Eigen::ArrayXXd::Random(num_prod, num_x_dim);
    cout<<x<<endl;
    return 0;

}