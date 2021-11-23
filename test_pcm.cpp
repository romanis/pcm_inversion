#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>
#include "pcm_market_share.hpp"
#include "Eigen/Dense"
#include "Eigen/Core"

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
    int func_evals = 0, jacobian_evals = 0;

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
    if(grad){
        Eigen::ArrayXd u_share = pcm_share::unc_share(eigen_deltas, params->x, params->p, params->sigma_p, params->sigma_x, params->grid, params->weights, jacobian);
        u_share -= params->data_shares;
        params->jacobian_evals++;
        // copy D into result
        for(int i=0; i<=n; ++i){
            result[i] = u_share[i];
        }

        // reshape jacobian
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> M2(jacobian);
        Map<RowVectorXd> v2(M2.data(), M2.size());
        /// copy jacobian
        for(int i=0; i<v2.size(); ++i){
            grad[i] = v2[i];
        }
    }else{
        Eigen::ArrayXd u_share = pcm_share::unc_share(eigen_deltas, params->x, params->p, params->sigma_p, params->sigma_x, params->grid, params->weights);
        u_share -= params->data_shares;
        params->func_evals++;
        // copy D into result
        for(int i=0; i<=n; ++i){
            result[i] = u_share[i];
        }
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

    int num_prod = 50, num_x_dim = 4; 
    double min_asmissible_share = 0.001, sigma_p = 1;
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
    auto un_sh = pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    while((un_sh < min_asmissible_share).any()){
        for(int i=0; i< num_prod; ++i){
            if(un_sh[i] < min_asmissible_share){
                delta_bar[i] += 0.01*std::abs(delta_bar[0] - delta_bar[num_prod-1]);
            }
        }
        un_sh = pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    }
    std::cout<<"unc share\n"<<un_sh<<std::endl<<"sum of all shares " <<un_sh.sum()<<std::endl;

    
    pcm_parameters param(x, p, 1.0, sigma_x, grid, weights, un_sh);
    pcm_parameters params[1] = {param};
    std::vector<double> tols(num_prod, 1e-8);

    opt.add_equality_mconstraint(c, &params[0], tols);
    opt.set_xtol_rel(1e-4);
    std::vector<double> x_initial(num_prod, 0);
    auto initial_guess = pcm_share::initial_guess(un_sh, p, sigma_p);
    for(int i = 0; i<num_prod; ++i){
        x_initial[i] = initial_guess[i];
    }
    double minf;
    bool success = false;
    try{
        nlopt::result result = opt.optimize(x_initial, minf);
        std::cout<<"success of the first try, resulting code " << result<<std::endl;
        Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);
        Eigen::ArrayX3d sh(num_prod, 3);
        
        auto shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);
        if((shares_at_optimum > 0).all() && shares_at_optimum.sum() < (1.0 - min_asmissible_share)){
            success = true;
        }
    }
    catch(std::exception &e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    while(!success){
        Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);
        Eigen::ArrayX3d sh(num_prod, 3);
        
        auto shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);

        int count_increments = 0;
        while((shares_at_optimum == 0).any() || (shares_at_optimum.sum() > 1.0 - min_asmissible_share)){
            if(shares_at_optimum.sum() > 1.0 - min_asmissible_share){
                std::cout<<"subtracting from all shares\n";
                eigen_solution -= std::abs(eigen_solution[0] - eigen_solution[num_prod-1]);
            }
            std::cout<<"number of zero shares " << (shares_at_optimum == 0).count()<<std::endl;
            if(shares_at_optimum[0] == 0){
                // find positive share index
                int j = 1;
                while(shares_at_optimum[j++] == 0){}
                eigen_solution[0] += 0.1*std::abs(eigen_solution[j] - eigen_solution[0]);
            }
            if(shares_at_optimum[num_prod-1] == 0){
                // find positive share index
                int j = num_prod - 2;
                while(shares_at_optimum[j--] == 0){}
                eigen_solution[num_prod-1] += 0.1*std::abs(eigen_solution[j] - eigen_solution[num_prod-1]);
            }
            // else, increase the deltas of the shares with zero predicted share
            for(int i=1; i<num_prod-1; ++i){
                if(shares_at_optimum[i] == 0){
                    eigen_solution[i] += 0.1*std::abs(eigen_solution[i-1] - eigen_solution[i+1]);
                }
            }
            shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);
            count_increments++;
        }
        std::cout<<"it took "<<count_increments<< " increments\n";
        try{
            nlopt::result result = opt.optimize(x_initial, minf);
            success = true;
            std::cout <<"number of func and jacobian evaluations is " << params[0].func_evals << "\t" << params[0].jacobian_evals<<"\nresult " << result<< " found minimum at f(" ;
            for(int i = 0; i<x_initial.size(); ++i){
                std::cout<<x_initial[i]<<"\t";
            }
            std::cout << ") = "
                << std::setprecision(10) << minf << std::endl;
        }catch(std::exception &e) {
            std::cout << "nlopt failed: " << e.what() << std::endl;
            Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);
            Eigen::ArrayX3d sh(num_prod, 3);
            
            auto shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);
            if((shares_at_optimum > 0).all()){
                std::cout<<"still, all shares are positive\n";
                success = true;
            }
        }
    }

    Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);
    Eigen::ArrayX4d sh(num_prod, 4);
    
    auto shares_at_optimum = pcm_share::unc_share(eigen_solution, x, p, sigma_p, sigma_x, grid, weights, jacobian);
    sh <<shares_at_optimum, un_sh, shares_at_optimum-un_sh, eigen_solution;
    std::cout<<"shares at optimum\n" << sh << std::endl;
    return 0;
}