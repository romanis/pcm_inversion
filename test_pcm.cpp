#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>
#include "pcm_market_share.hpp"
#include "Eigen/Dense"

// struct that is going to take cara of the pcm computations

typedef struct {
    Eigen::MatrixXd x;
    Eigen::ArrayXd p;
    double sigma_p;
    Eigen::ArrayXd sigma_x;
    Eigen::ArrayXXd grid;
    Eigen::ArrayXd weights;
    Eigen::ArrayXd data_shares;
} pcm_parameters;

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
}


double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    if (grad) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

typedef struct {
    double a, b;
} my_constraint_data;

double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
    my_constraint_data *d = (my_constraint_data *) data;
    double a = d->a, b = d->b;
    if (grad) {
        grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
        grad[1] = -1.0;
    }
    return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
 }


int main(){
  nlopt::opt opt(nlopt::LD_MMA, 2);
  std::vector<double> lb(2);
  lb[0] = -HUGE_VAL; lb[1] = 0;
  opt.set_lower_bounds(lb);
  opt.set_min_objective(myfunc, NULL);
  my_constraint_data data[2] = { {2,0}, {-1,1} };
  opt.add_inequality_constraint(myconstraint, &data[0], 1e-8);
  opt.add_inequality_constraint(myconstraint, &data[1], 1e-8);
  opt.set_xtol_rel(1e-4);
  std::vector<double> x(2);
  x[0] = 1.234; x[1] = 5.678;
  double minf;

  try{
      nlopt::result result = opt.optimize(x, minf);
      std::cout <<"result " << result<< " found minimum at f(" << x[0] << "," << x[1] << ") = "
          << std::setprecision(10) << minf << std::endl;
  }
  catch(std::exception &e) {
      std::cout << "nlopt failed: " << e.what() << std::endl;
  }
  return 0;
}