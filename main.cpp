#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRSolver.h"
#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRProblem.h"
#include <iostream>
#include "pcm_market_share.h"
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"
#include <bits/stdc++.h>
#include <random>
#include <numeric>
#include <string>
#include <functional>
#include <algorithm>    // std::all_of
#include <array>

using namespace TasGrid;
using namespace std;

inline void printSolutionResults(knitro::KTRISolver & solver, int solveStatus) {
   if (solveStatus != 0) {
     std::cout << "Failed to solve problem, final status = " << solveStatus << std::endl;
//     return;
   }
   else{
    std::cout << "---------- Solution found ----------" << std::endl << std::endl;
   }

   std::cout.precision(4);
   std::cout << std::fixed;

   // Objective value
   std::cout << std::right << std::setw(28) << "Objective value = " << solver.getObjValue() << std::endl;

   // Solution point
   std::cout << std::right << std::setw(29) << "Final point = (";
   const std::vector<double>& point = solver.getXValues();
   std::vector<double>::const_iterator it = point.begin();
   while ( it != point.end()) {
       std::cout << *it;
       if (++it != point.end())
           std::cout << ", ";
   }
   std::cout << ")" << std::endl;

   if (!((solver.getProblem())->isMipProblem()))
   {
       std::cout << std::right << std::setw(28) << "Feasibility violation = " << solver.getAbsFeasError() << std::      endl;
       std::cout << std::right << std::setw(28) << "KKT optimality violation = " << solver.getAbsOptError() <<          std::endl;
   }
   else {
       std::cout << std::right << std::setw(28) << "Absolute integrality gap = " << solver.getMipAbsGap() << std::      endl;
   }
   std::cout << std::endl;
 }



  int main(int argc, char *argv[]) {
    vector<double> sch ;//= {0.1, 0.2, 0.1};
      
      
    std::vector<double> c;  
    std::vector<double> objGrad; 
    std::vector<double> jac;
      
    std::vector<double> delta, delta_p;
    std::vector<double> p;
    int dim = 3;
    int num_prod = 100;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
      
    std::uniform_real_distribution<double> unif(-1,1);
    std::default_random_engine re;
    
      
    vector<vector<double>> x;
    for(int i = 0; i< num_prod; i++){
        vector<double> x_tmp;
        for (int j=0; j< dim; ++j){
            x_tmp.push_back(unif(generator));
        }
        p.push_back(pow((i+1),1));
        delta.push_back(i);
        x.push_back(x_tmp);
        sch.push_back(unif(generator)+2);
    }
    double sum_sch=1;
    for(int i = 0; i<sch.size(); ++i){
        sum_sch += sch[i];
//          sch[i] = sch[i]/sum_sch;
//          cout<<"share " <<sch[i]<<endl;
    }
    for(int i = 0; i<sch.size(); ++i){
//          sum_sch += sch[i];
        sch[i] = sch[i]/(1.3*sum_sch);
        cout<<"share " <<sch[i]<<endl;
    }
    vector<double> sigmax(x[0].size(),1.5);
    cout<<"size of sigmax "<<sigmax.size()<<endl;
    cout<<"size of x[0] "<<x[0].size()<<endl;
    std::vector<vector<double> > jacobian;
    double sigma_p=1;

    pcm_market_share share1(sch, x, sigmax, p, sigma_p);
    share1.set_grid(dim,15);
      
//      cond_share(delta,p,sigma_p, jacobian);
//      vector<double> val1 = share1.unc_share(delta, x, p, sigma_p, sigmax,jacobian);
//      vector<double> val2 = share1.unc_share(delta_p, x, p, sigma_p, sigmax,jacobian);
//      cout<<"numerical jac " <<(val1[1]-val2[1])*1e4<<endl;
//    print_jacobian(jacobian);
    double start = omp_get_wtime();
    
    cout<< "shares at start \n";
    vector<double> sch_start = share1.unc_share(share1.initial_guess(), x, p, sigma_p, sigmax,jacobian);
//    share1.setXInitial(share1.initial_guess());
    for(auto it: share1.unc_share(share1.initial_guess(), x, p, sigma_p, sigmax,jacobian)){
        cout<< it<<endl;
    }
//    if not all of initial shares greater than zero 
    if(!all_of(sch_start.begin(), sch_start.end(), [](double it){return it>1e-4;})){
//        increase deltas so that all have positive market shares
        cout<<"not all values positive \n";
    }
    cout<<" value " <<share1.evaluateFC(share1.initial_guess(),c,objGrad,jac)<<endl;
//    vector<double> val1 = share1.unc_share(share1.initial_guess(), x, p, sigma_p, sigmax,jacobian);
//    delta= share1.initial_guess();
//    delta[1] += 1e-6;
//    cout<<" value prime " <<share1.evaluateFC(delta,c,objGrad,jac)<<endl;
//    vector<double> val2 = share1.unc_share(delta, x, p, sigma_p, sigmax,jacobian);
//    cout<<"numerical derivative "<< (val2[0] - val1[0])*1e6<<endl;
//    print_jacobian(jacobian);
//    share1.evaluateGA(delta,objGrad, jac);
//    for(auto it : objGrad){
//        cout<< "derivative " <<it<<endl;
//    }
    
//    return 0;
    share1.evaluateGA(delta, objGrad, jac);
//    for(auto it: jac){
//        cout<<it<<" ";
//    }
    cout<<endl<< "time to calculate "<< omp_get_wtime() - start<<endl;
    share1.setXInitial(share1.initial_guess());
    share1.get_traction();
    cout<<"initial guess \n";
    for(auto it :share1.initial_guess()){
        cout<< it<<endl;
    }
    knitro::KTRSolver solver(&share1, KTR_GRADOPT_EXACT, KTR_HESSOPT_BFGS);
    solver.setParam(KTR_PARAM_ALG, 2); // 2 = CG algorithm 3 = active set
    solver.setParam(KTR_PARAM_MAXIT, 200);
    solver.setParam(KTR_PARAM_FTOL, 1e-12);
    solver.setParam(KTR_PARAM_XTOL, 1e-8);
    solver.setParam(KTR_PARAM_OPTTOL, 1e-7);
    solver.setParam(KTR_PARAM_DERIVCHECK, 0);
    int result = solver.solve();
    vector<double> solution;
    double factor;
    if(result != 0){
        
        factor = share1.relax_til_solved(solution, solver.getXValues());
        cout<<"needed to contract " <<factor<<endl;
    }
    share1.setXInitial(solution);
    solver.solve();
    printSolutionResults(solver, result);
    cout<< " shares at optimum "<<endl;
    std::cout.precision(4);
    std::cout << std::fixed;
    int i=0;
    cout<< "final sigma x "<< endl;
//    share1.decrease_sigma_x(1/factor);
    for(auto it : share1.get_sigma_x()){
        cout<< it<<" ";
    }
    cout<< endl;
    
    for(auto it : share1.unc_share(solution, x, p, sigma_p, sigmax)){
        cout<<it<<" "<<sch[i++]<<endl;
    }
    share1.evaluateGA(solver.getXValues(),objGrad, jac);
    cout<< "gradient at solution "<<endl;
//    for(auto it : objGrad){
//        cout<<it<<" "<<endl;
//    }
//      print_jacobian(jacobian);
      
      
//      ProblemExample* problem = new ProblemExample();
//
//      // Create a solver - optional arguments: use numerical derivative evaluation.
//      knitro::KTRSolver solver1(problem, KTR_GRADOPT_FORWARD, KTR_HESSOPT_BFGS);
//
//      int solveStatus = solver1.solve();
      return 0;
  }
