#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRSolver.h"
#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRProblem.h"
#include <iostream>
#include "pcm_market_share.h"
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"
#include <bits/stdc++.h>
using namespace TasGrid;
using namespace std;

class ProblemExample : public knitro::KTRProblem {
private:
    // objective properties
    void setObjectiveProperties() {
        setObjType(knitro::KTREnums::ObjectiveType::ObjGeneral);
        setObjGoal(knitro::KTREnums::ObjectiveGoal::Minimize);
    }

    // variable bounds. All variables 0 <= x.
    void setVariableProperties() {
        setVarLoBnds(0.0);
    }

    // constraint properties
    void setConstraintProperties() {
        // set constraint types
        setConTypes(0, knitro::KTREnums::ConstraintType::ConGeneral);
        setConTypes(1, knitro::KTREnums::ConstraintType::ConGeneral);

        // set constraint lower bounds to zero for all variables
        setConLoBnds(0.0);

        // set constraint upper bounds
        setConUpBnds(0, 0.0);
        setConUpBnds(1, KTR_INFBOUND);
    }

  public:
      // constructor: pass number of variables and constraints to base class.
      // 3 variables, 2 constraints.
      ProblemExample() : KTRProblem(3, 2) {
          // set problem properties in constructor
          setObjectiveProperties();
          setVariableProperties();
          setConstraintProperties();
      }

      // Objective and constraint evaluation function
      // overrides KTRIProblem class
      double evaluateFC(
          const std::vector<double>& x,
          std::vector<double>& c,
          std::vector<double>& objGrad,
          std::vector<double>& jac) {

          // constraints
          c[0] = 8.0e0*x[0] + 14.0e0*x[1] + 7.0e0*x[2] - 56.0e0;
          c[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 25.0e0;

          // return objective function value
          return 1000 - x[0] * x[0] - 2.0e0*x[1] * x[1] - x[2] * x[2]
              - x[0] * x[1] - x[0] * x[2];
      }
  };

  int main(int argc, char *argv[]) {
//      // Create a problem instance.
//      ProblemExample* problem = new ProblemExample();
//
//      // Create a solver - optional arguments: use numerical derivative evaluation.
//      knitro::KTRSolver solver(problem, KTR_GRADOPT_FORWARD, KTR_HESSOPT_BFGS);
//
//      int solveStatus = solver.solve();
//
//      if (solveStatus != 0) {
//          std::cout << std::endl;
//    std::cout << "Knitro failed to solve the problem, final status = ";
//          std::cout << solveStatus << std::endl;
//      }
//      else {
//          std::cout << std::endl << "Knitro successful, objective is = ";
//          std::cout << solver.getObjValue() << std::endl;
//      }
      
      pcm_market_share share1;
//      share1.set_grid(std::stoi(argv[1]),std::stoi(argv[2]));
      
      std::vector<double> delta, delta_p;
      std::vector<double> p;
      std::vector<vector<double> > jacobian;
      double sigma_p=1;
      delta.push_back(1);
      delta_p.push_back(1);
      
      delta.push_back(2);
      delta_p.push_back(2-1e-4);
      
      delta.push_back(4);
      delta_p.push_back(4);
      
      p.push_back(stod(argv[1]));
      p.push_back(stod(argv[2]));
      p.push_back(stod(argv[3]));
      cond_share(delta,p,sigma_p, jacobian);
      vector<double> val1 = cond_share(delta,p,sigma_p);
      vector<double> val2 = cond_share(delta_p,p,sigma_p);
      cout<<"numerical jac " <<(val1[2]-val2[2])*1e4<<endl;
      print_jacobian(jacobian);
      return 0;
  }