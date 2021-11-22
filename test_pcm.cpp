#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>
#include "pcm_market_share.hpp"
#include "Eigen/Dense"

// struct that is going to take cara of the pcm computations
using namespace Eigen;
struct pcm_parameters{
    Eigen::MatrixXd x;
    Eigen::ArrayXd p;
    double sigma_p;
    Eigen::ArrayXd sigma_x;
    Eigen::ArrayXXd grid;
    Eigen::ArrayXd weights;
    Eigen::ArrayXd data_shares;

    pcm_parameters(MatrixXd& x, ArrayXd &p, double sigma_p, ArrayXd &sigma_x, ArrayXXd& grid, ArrayXd& weights, ArrayXd& data_shares) :
    x(x), p(p), sigma_x(sigma_x), sigma_p(sigma_p), grid(grid), weights(weights), data_shares(data_shares) {};
} ;

/**
 * @brief Computes the positive and negative constraints that all predicted shares are equal to data shares
 * 
 * @param m number of constraints (2*number of variables)
 * @param result array of deviations of constraints from zero
 * @param n number of variables (same as number of products)
 * @param delta_bars values of parameters (vertical qualities of the products)
 * @param grad vectorized row major Jacobian 
 * @param pcm_data pointer to the struct that is holding the pcm parameters
 */
void c(unsigned m, double *result, unsigned n, const double* deltas, double* grad, void* pcm_data){
    pcm_parameters * params = (pcm_parameters *) pcm_data;
    Eigen::MatrixXd jacobian;
    // map deltas to eigen array 
    Eigen::Map<const Eigen::ArrayXd> eigen_deltas(deltas, n); 
    Eigen::ArrayXd u_share = pcm_share::unc_share(eigen_deltas, params->x, params->p, params->sigma_p, params->sigma_x, params->grid, params->weights, jacobian);
    u_share -= params->data_shares;
    
    // copy D into result
    for(int i=0; i<=n; ++i){
        result[i] = u_share[i];
    }

    if(!grad) return;

    // reshape jacobian
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> M2(jacobian);
    Map<RowVectorXd> v2(M2.data(), M2.size());
    /// copy jacobian
    for(int i=0; i<v2.size(); ++i){
        grad[i] = v2[i];
    }
}


double myfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    if (!grad.empty()) {
        for(int i=0; i < x.size() ; ++i){
          grad[i] = 0;
        }
    }
    return 0;
}



int main(){

    int num_prod = 10, num_x_dim = 4;
    nlopt::opt opt(nlopt::LD_SLSQP, num_prod);
    opt.set_min_objective(myfunc, NULL);
    
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
    pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, un_sh);
    pcm_parameters params[1] = {param};
    std::vector<double> tols(num_prod, 1e-8);

    opt.add_equality_mconstraint(c, &params[0], tols);
    opt.set_xtol_rel(1e-4);
    std::vector<double> x_initial(num_prod, 0);
    for(int i = 0; i<num_prod; ++i){
        x_initial[i] = std::pow(1.0*i, 1.5) + std::rand()/INT_MAX;
    }
    double minf;

    try{
        nlopt::result result = opt.optimize(x_initial, minf);
        std::cout <<"result " << result<< " found minimum at f(" ;
        for(int i = 0; i<x_initial.size(); ++i){
            std::cout<<x_initial[i]<<"\t";
        }
        std::cout << ") = "
            << std::setprecision(10) << minf << std::endl;
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }
    return 0;
}