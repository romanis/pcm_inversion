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
        for(unsigned i=0; i<=n; ++i){
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
        for(unsigned i=0; i<=n; ++i){
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
        for(unsigned i=0; i < x.size() ; ++i){
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
    for(unsigned i=1; i<num_prod-1; ++i){
        if(shares_at_optimum[i] == 0){
            eigen_solution[i] += 0.1*std::abs(eigen_solution[i-1] - eigen_solution[i+1]);
        }
    }
    return;
}

/**
 * @brief Selects the starting point for the cone-aware Newton iteration.
 *
 * Returns pcm_share::initial_guess unless params.delta_initial is non-empty,
 * matches the product count, and its predicted shares are strictly positive
 * with sum below 1 - share_inversion::MIN_ADMISSIBLE_SHARE.
 *
 * @param params fully populated share_inversion::pcm_parameters struct
 * @return Eigen::ArrayXd initial delta of size n_products inside the feasibility cone P
 */
Eigen::ArrayXd cone_newton::select_initial_delta(share_inversion::pcm_parameters & params)
{
    const unsigned n = static_cast<unsigned>(params.data_shares.size());
    Eigen::ArrayXd delta = pcm_share::initial_guess(
        params.data_shares, params.p, params.sigma_p);

    if (!params.delta_initial.empty() &&
        static_cast<long int>(params.delta_initial.size()) == params.data_shares.size())
    {
        Eigen::Map<const Eigen::ArrayXd> delta_user(params.delta_initial.data(), n);
        Eigen::ArrayXd s_user = pcm_share::unc_share(
            delta_user, params.x, params.p, params.sigma_p,
            params.sigma_x, params.grid, params.weights);
        if ((s_user > 0).all() &&
            s_user.sum() < 1.0 - share_inversion::MIN_ADMISSIBLE_SHARE)
        {
            delta = delta_user;
        }
    }
    return delta;
}

/**
 * @brief Computes a Newton step from the current Jacobian and residual.
 *
 * Attempts J.lu().solve(-r). If the result contains NaN or its Euclidean norm
 * exceeds 1e6, falls back to the diagonal-Jacobian steepest-descent step
 * -J.diagonal()^{-1} * r.
 *
 * @param J dense Jacobian matrix of size n x n returned by pcm_share::unc_share
 * @param r residual array of size n equal to s - data_shares
 * @return Eigen::VectorXd Newton step of size n
 */
Eigen::VectorXd cone_newton::compute_newton_step(const Eigen::MatrixXd & J,
                                                  const Eigen::ArrayXd  & r)
{
    Eigen::VectorXd step = J.lu().solve((-r).matrix());
    if (step.hasNaN() || step.norm() > 1e6)
    {
        step = -(J.diagonal().array().inverse() * r).matrix();
    }
    return step;
}

/**
 * @brief Backtracks the Newton step to restore feasibility.
 *
 * Starting from lambda = 1, halves lambda up to max_halvings = 50 times until
 * all shares of delta + lambda * step are strictly positive and their sum is
 * below 1 - share_inversion::MIN_ADMISSIBLE_SHARE. Increments calls for each
 * pcm_share::unc_share evaluation.
 *
 * @param delta current iterate of size n
 * @param step Newton step of size n
 * @param params fully populated share_inversion::pcm_parameters struct
 * @param calls reference to the function-evaluation counter; incremented in place
 * @return std::pair containing the best feasible delta_candidate and its share vector
 */
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
cone_newton::backtrack_to_feasibility(const Eigen::ArrayXd            & delta,
                                      const Eigen::VectorXd           & step,
                                      share_inversion::pcm_parameters & params,
                                      unsigned                        & calls)
{
    const unsigned max_halvings = 50;
    double lambda = 1.0;
    Eigen::ArrayXd delta_candidate = delta + lambda * step.array();
    Eigen::ArrayXd s_candidate = pcm_share::unc_share(
        delta_candidate, params.x, params.p, params.sigma_p,
        params.sigma_x, params.grid, params.weights);
    ++calls;

    for (unsigned k = 0;
         k < max_halvings &&
         (!((s_candidate > 0).all()) ||
          s_candidate.sum() >= 1.0 - share_inversion::MIN_ADMISSIBLE_SHARE);
         ++k)
    {
        lambda *= 0.5;
        delta_candidate = delta + lambda * step.array();
        s_candidate = pcm_share::unc_share(
            delta_candidate, params.x, params.p, params.sigma_p,
            params.sigma_x, params.grid, params.weights);
        ++calls;
    }
    return {delta_candidate, s_candidate};
}

/**
 * @brief Attempts to repair an infeasible delta_candidate via adjust heuristic.
 *
 * Calls adjust_deltas_towards_positive_shares on delta_candidate; if the
 * resulting shares are all strictly positive, returns the repaired delta.
 * Otherwise returns delta_candidate unchanged.
 *
 * @param delta_candidate infeasible iterate of size n
 * @param s_candidate predicted shares for delta_candidate; some may be non-positive
 * @param params fully populated share_inversion::pcm_parameters struct
 * @param calls reference to the function-evaluation counter; incremented in place
 * @return Eigen::ArrayXd repaired delta of size n, or delta_candidate if repair fails
 */
Eigen::ArrayXd cone_newton::recover_feasibility(
    const Eigen::ArrayXd            & delta_candidate,
    const Eigen::ArrayXd            & s_candidate,
    share_inversion::pcm_parameters & params,
    unsigned                        & calls)
{
    const unsigned n = static_cast<unsigned>(delta_candidate.size());
    std::vector<double> x_recover(
        delta_candidate.data(), delta_candidate.data() + n);
    Eigen::ArrayXd s_mut = s_candidate;
    adjust_deltas_towards_positive_shares(s_mut, x_recover);
    Eigen::Map<Eigen::ArrayXd> recovered(x_recover.data(), n);
    Eigen::ArrayXd s_after = pcm_share::unc_share(
        recovered, params.x, params.p, params.sigma_p,
        params.sigma_x, params.grid, params.weights);
    ++calls;
    if ((s_after > 0).all())
        return recovered;
    return delta_candidate;
}

/**
 * @brief Cone-aware Newton solver: finds delta such that unc_share(delta) == data_shares.
 *
 * Orchestrates select_initial_delta, then iterates: compute_newton_step,
 * backtrack_to_feasibility, recover_feasibility until the L-inf share residual
 * drops below params.share_equality_tolerances or the call budget is exhausted.
 *
 * @param params fully populated share_inversion::pcm_parameters struct
 * @return Eigen::ArrayXd delta solution of size n_products
 */
Eigen::ArrayXd cone_newton::invert(share_inversion::pcm_parameters & params)
{
    Eigen::ArrayXd delta = cone_newton::select_initial_delta(params);

    unsigned calls = 0;
    const unsigned max_calls = params.max_number_of_function_calls;
    const unsigned max_iter  = 100;

    for (unsigned iter = 0; iter < max_iter; ++iter)
    {
        if (calls >= max_calls)
            break;

        Eigen::MatrixXd J;
        Eigen::ArrayXd s = pcm_share::unc_share(
            delta, params.x, params.p, params.sigma_p,
            params.sigma_x, params.grid, params.weights, J);
        ++calls;

        Eigen::ArrayXd r = s - params.data_shares;

        if (r.abs().maxCoeff() < params.share_equality_tolerances)
            break;

        Eigen::VectorXd step = cone_newton::compute_newton_step(J, r);

        auto [delta_candidate, s_candidate] =
            cone_newton::backtrack_to_feasibility(delta, step, params, calls);

        if (!((s_candidate > 0).all()))
        {
            delta = cone_newton::recover_feasibility(
                delta_candidate, s_candidate, params, calls);
        }
        else
        {
            delta = delta_candidate;
        }
    }

    return delta;
}

namespace share_inversion{
    double MIN_ADMISSIBLE_SHARE = 1e-5; // extern definition to avoind multiple definition error on linker stage

    /**
     * @brief Inverts PCM market shares using the NLopt LD_SLSQP constrained optimizer.
     *
     * Creates an nlopt::opt instance, adds an equality multi-constraint via c(), and
     * enters a retry loop using adjust_deltas_towards_positive_shares() when predicted
     * shares are not all strictly positive.
     *
     * @param params all parameters of the market; observed shares are in params.data_shares
     * @return Eigen::ArrayXd delta of size n_products such that unc_share(delta) == data_shares
     */
    Eigen::ArrayXd invert_shares_nlopt(share_inversion::pcm_parameters & params){
        params.number_of_times_function_called = 0;
        nlopt::opt opt(params.nlopt_algo, params.data_shares.size());
        opt.set_min_objective(myfunc, NULL);
        unsigned num_prod = params.data_shares.size();
        std::vector<double> tols(num_prod, params.share_equality_tolerances);

        opt.add_equality_mconstraint(c, &params, tols);
        opt.set_xtol_rel(params.delta_step_tolerance);

        std::vector<double> x_initial(num_prod, 0);
        if (!params.delta_initial.empty() && (long int) params.delta_initial.size() == params.data_shares.size()){
            x_initial = params.delta_initial;
        }else{
            auto initial_guess = pcm_share::initial_guess(params.data_shares, params.p, params.sigma_p);
            for(unsigned i = 0; i<num_prod; ++i){
                x_initial[i] = initial_guess[i];
            }
        }
        Eigen::Map<Eigen::ArrayXd> eigen_solution(x_initial.data(), num_prod);

        double minf;
        bool success = false;
        try{
            opt.optimize(x_initial, minf);
                        
            auto shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            if((shares_at_optimum > 0).all() && shares_at_optimum.sum() < 1.0 - MIN_ADMISSIBLE_SHARE){
                success = true;
            }
        }
        catch(std::exception &e) {
        }

        while(!success && params.number_of_times_function_called < params.max_number_of_function_calls){    
            auto shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            while((shares_at_optimum == 0).any() || (shares_at_optimum.sum() > 1.0 - MIN_ADMISSIBLE_SHARE)){
                adjust_deltas_towards_positive_shares(shares_at_optimum, x_initial);
                shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
            }
            try{
                opt.optimize(x_initial, minf);
                success = true;
            }catch(std::exception &e) {
                shares_at_optimum = pcm_share::unc_share(eigen_solution, params.x, params.p, params.sigma_p, params.sigma_x, params.grid, params.weights);
                if((shares_at_optimum > 0).all()){
                    success = true;
                }
            }
        }

        
        return eigen_solution;
    }

    /**
     * @brief Inverts PCM market shares to recover delta, dispatching to the algorithm
     *        selected by method.
     *
     * For NLOPT_SLSQP, delegates to invert_shares_nlopt().
     * For CONE_AWARE_NEWTON, delegates to cone_newton::invert().
     *
     * @param params all parameters of the market; observed shares are in params.data_shares
     * @param method inversion algorithm to use
     * @return Eigen::ArrayXd delta of size n_products such that unc_share(delta) == data_shares
     */
    Eigen::ArrayXd invert_shares(share_inversion::pcm_parameters & params,
                                  share_inversion::InversionMethod method)
    {
        if (method == share_inversion::InversionMethod::CONE_AWARE_NEWTON)
            return cone_newton::invert(params);

        return share_inversion::invert_shares_nlopt(params);
    }
}
