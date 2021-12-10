#include "TasmanianSparseGrid.hpp"

#include <Eigen/Dense>
#include <vector>



void generate_tasmanian_global_grid(int dim, int n, Eigen::ArrayXXd& grid, Eigen::ArrayXd & weights){
//    use tasmanian to calculate points and weights
        TasGrid::TasmanianSparseGrid tgr;

        
        tgr.makeGlobalGrid(dim,1,n,TasGrid::type_tensor,TasGrid::rule_gausshermite);
        std::vector<double> weights1 = tgr.getQuadratureWeights();
        std::vector<double> coordinates = tgr.getPoints();
        grid = Eigen::ArrayXXd::Zero(tgr.getNumPoints(), dim);
        weights = Eigen::ArrayXd::Zero(tgr.getNumPoints());
        for(int point_number = 0; point_number < tgr.getNumPoints(); ++point_number){
            weights[point_number] = weights1[point_number]/pow(std::sqrt(M_PI),dim);
            for(int j = 0; j < dim; j++){
                grid(point_number, j) = coordinates[j + dim*point_number]*M_SQRT2;
            }
        }

}