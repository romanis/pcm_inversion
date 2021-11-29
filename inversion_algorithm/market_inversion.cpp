#include "market_inversion.hpp"
#include "nlopt.hpp"


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
    share_inversion::pcm_parameters * params = (share_inversion::pcm_parameters *) pcm_data;
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
        Eigen::Map<Eigen::RowVectorXd> v2(M2.data(), M2.size());
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
    params->number_of_times_function_called++;
}

/**
 * @brief function that computes objective and its gradient. zero in this case as we are solving pure feasibility problem
 * 
 * @param x 
 * @param grad 
 * @param my_func_data 
 * @return double 
 */
double myfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    if (!grad.empty()) {
        for(int i=0; i < x.size() ; ++i){
          grad[i] = 0;
        }
    }
    return 0;
}

void adjust_deltas_towards_positive_shares(Eigen::ArrayXd & shares_at_optimum, std::vector<double> & eigen_solution){
    unsigned num_prod = shares_at_optimum.size();
    if(shares_at_optimum.sum() > 1.0 - share_inversion::MIN_ADMISSIBLE_SHARE){
        for(auto & d: eigen_solution){
            d -= std::abs(eigen_solution[0] - eigen_solution[num_prod-1]);
        }        
    }
    // std::cout<<"number of zero shares " << (shares_at_optimum == 0).count()<<std::endl;
    if(shares_at_optimum[0] == 0){
        // find positive share index
        int j = 1;
        while(shares_at_optimum[j++] == 0){}
        eigen_solution[0] += 0.1*std::abs(eigen_solution[j] - eigen_solution[0]);
    }
    if(shares_at_optimum[num_prod-1] == 0){
        // find positive share index
        unsigned j = num_prod - 2;
        while(shares_at_optimum[j--] == 0){}
        eigen_solution[num_prod-1] += 0.1*std::abs(eigen_solution[j] - eigen_solution[num_prod-1]);
    }
    // else, increase the deltas of the shares with zero predicted share
    for(int i=1; i<num_prod-1; ++i){
        if(shares_at_optimum[i] == 0){
            eigen_solution[i] += 0.1*std::abs(eigen_solution[i-1] - eigen_solution[i+1]);
        }
    }
    return;
}

namespace share_inversion{
    Eigen::ArrayXd invert_shares(share_inversion::pcm_parameters & params){
        params.number_of_times_function_called = 0;
        // create instance of optimization. dimensionality of the problem is the number of shares to invert
        nlopt::opt opt(params.nlopt_algo, params.data_shares.size());
        opt.set_min_objective(myfunc, NULL);
        unsigned num_prod = params.data_shares.size();
        std::vector<double> tols(num_prod, params.share_equality_tolerances);

        opt.add_equality_mconstraint(c, &params, tols);
        opt.set_xtol_rel(params.delta_step_tolerance);

        std::vector<double> x_initial(num_prod, 0);
        if (!params.delta_initial.empty() && params.delta_initial.size() == params.data_shares.size()){
            x_initial = params.delta_initial;
        }else{
            auto initial_guess = pcm_share::initial_guess(params.data_shares, params.p, params.sigma_p);
            for(unsigned i = 0; i<num_prod; ++i){
                x_initial[i] = initial_guess[i];
            }
        }
        /// map X to eigen array
        Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);

        // start solution
        double minf;
        bool success = false;
        /// try first solution from this starting point
        try{
            nlopt::result result = opt.optimize(x_initial, minf);
                        
            auto shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            if((shares_at_optimum > 0).all() && shares_at_optimum.sum() < 1.0 - MIN_ADMISSIBLE_SHARE){
                success = true;
            }
        }
        catch(std::exception &e) {
            // std::cout << "nlopt failed: " << e.what() << std::endl;
        }

        /// likely it will not be the success from the first time, in this case enter the loop of adjusting deltas to the point all products have positive predicted shares
        while(!success && params.number_of_times_function_called < params.max_number_of_function_calls){    
            /// compute current shares at the point of current optimum candidate        
            auto shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            /// adjust current optimum candidate until all predicted shares are positive and are totally lower than 
            while((shares_at_optimum == 0).any() || (shares_at_optimum.sum() > 1.0 - MIN_ADMISSIBLE_SHARE)){
                adjust_deltas_towards_positive_shares(shares_at_optimum, x_initial);
                /// compute new shares at optimum candidate
                shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            }
            /// start new optimization routine from the new point
            try{
                nlopt::result result = opt.optimize(x_initial, minf);
                success = true;
            }catch(std::exception &e) {
                /// sometimes the solution has numeric problems, but for all other purposes, it had converged
                shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
                if((shares_at_optimum > 0).all()){
                    success = true;
                }
            }
        }

        
        return eigen_solution;
    }
}