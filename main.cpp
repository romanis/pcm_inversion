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
#include <omp.h>
#include <math.h>
//#include <mkl.h>
#include "matrix_inverse.h"

using namespace TasGrid;
using namespace std;

std::vector<double> operator * (std::vector<std::vector<double>> A, std::vector<double> x){
    std::vector<double> result = std::vector<double>(A.size(), 0);
    for(int i=0; i<A.size(); ++i){
        for(int j=0; j< A[0].size(); ++j){
            result[i] += A[i][j]*x[j];
        }
    }
    return result;
}

std::vector<double> operator - (std::vector<double> x1, std::vector<double> x2){
    std::vector<double> result = std::vector<double>(x1.size(), 0);
   
    for(int j=0; j< x1.size(); ++j){
        result[j] += x1[j]-x2[j];
    }

    return result;
}



int main(int argc, char *argv[]) {
    vector<double> sch ;//= {0.1, 0.2, 0.1};
      
      
    std::vector<double> c;  
    std::vector<double> objGrad; 
    std::vector<double> jac;
      
    std::vector<double> delta, delta_p;
    std::vector<double> p;
    int dim = 4;
    int num_prod = 10;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed = 1000000;
    std::default_random_engine generator (seed);
      
    std::uniform_real_distribution<double> unif(-1,1);
    std::default_random_engine re;
    
//    vector<vector<double>> A_inv, A = vector<vector<double > > (4, vector<double>(4,0));
//    vector<double> b = vector<double> (4, 1);
//    A[0][0] = 1;
//    A[0][1] = 0;
//    A[0][2] = 3;
//    A[1][0] = M_PI;
//    A[1][1] = 4;
//    A[1][2] = 0;
//    A[2][2] = 1;
//    A[2][0] = 1;
//    A[2][1] = 0;
//    A[3][3] = 2;
//    A[3][0] = 4;
//    cout<< "determinant of A "<<det_permutations(A) << " " << det(A)<<endl;
//    A_inv = inv_det(A);
////    vector<double> x_solution = A_inv_b_iter(A,b);
////    for(int i=0; i<3; ++i){
////        for(int j=0; j<3; ++j)
////            cout<<A[i][j]<<"\t";
////        cout<<endl;
////    }
////    cout<< endl;
//    cout<<"A inverted\n";
//    for(int i=0; i<4; ++i){
//        for(int j=0; j<4; ++j)
//            cout<<A_inv[i][j]<<"\t";
//        cout<<endl;
//    }
//    
//    A_inv = inv_det_permute(A);
//    cout<<"A inverted with permutations \n";
//    for(int i=0; i<4; ++i){
//        for(int j=0; j<4; ++j)
//            cout<<A_inv[i][j]<<"\t";
//        cout<<endl;
//    }
////    cout<<det(A)<<endl;
//    cout<<" solution Ainv b\n";
////    vector<double> x1 = A_inv*b;
////    for(auto it: x_solution){
////        cout<<it<<endl;
////    }
////    cout<<endl;
//    return 0;
    
      
    vector<vector<double>> x;
//    generate uniformly distributed maket shares and X and P
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
    
//    equate two prices
//    p[1] = p[0];
//    p[2] = p[0];
    
//    normalize shares to sum to less than 1
    {
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
    }
//    generate vector of std of heterogeneity
    vector<double> sigmax(x[0].size(),1.5);
    cout<<"size of sigmax "<<sigmax.size()<<endl;
    cout<<"size of x[0] "<<x[0].size()<<endl;
    std::vector<vector<double> > jacobian;
    double sigma_p=1;

    pcm_market_share share1(sch, x, sigmax, p, sigma_p);
    
//    set grid for integration, 15 points per dimension is usually enough
    share1.set_grid(dim, 10);
      
//      cond_share(delta,p,sigma_p, jacobian);
//      vector<double> val1 = share1.unc_share(delta, x, p, sigma_p, sigmax,jacobian);
//      vector<double> val2 = share1.unc_share(delta_p, x, p, sigma_p, sigmax,jacobian);
//      cout<<"numerical jac " <<(val1[1]-val2[1])*1e4<<endl;
//    print_jacobian(jacobian);
    double start = omp_get_wtime();
    
    cout<< "shares at start \n";
    vector<double> sch_start = share1.unc_share(share1.initial_guess(), x, p, sigma_p, sigmax, jacobian);
    cout<<endl<< "time to calculate "<< omp_get_wtime() - start<<endl;
    
    cout<<"sh at start;\t sh in data\n";
    double distrepancy = 0;
    for(int sh = 0; sh< sch_start.size(); ++sh){
        cout<< sch_start[sh]<<" \t" <<sch[sh]  <<"\n";
        distrepancy += abs(sch_start[sh] - sch[sh]);
    }
    cout<< "discrepancy " << distrepancy<<endl;
    
//    solve system of equations with newton method
//    stat variables
    double discrepancy_direct, discrepancy_iter;
    int num_iter_direct=0, num_iter_iter=0;
//    evaluate constraints
    vector<double> delta_start = share1.initial_guess();
//    vector<double> delta_backup = delta_start;
    double start_iterative = omp_get_wtime();
    return 0;
    while(distrepancy > 1e-10){
        share1.evaluateFC(delta_start,c,objGrad,jac);
        share1.evaluateGA(delta_start, objGrad, jac);
    //    repack jacobian
        vector<vector<double>> jacobian_square;
        {
    //        create a head of a new vector
        int head = 0;
        for(int i=0; i<c.size(); ++i){
            vector<double> j_i(&jac[head], &jac[head+c.size()]);
            jacobian_square.push_back(j_i);
            head+=c.size();
        }
        }
    //    compute difference with next newton iteration
//        delta_start = delta_start - inv_det(jacobian_square)*c;

//        solve system jacobian*x = c
        delta_start = delta_start - A_inv_b_iter(jacobian_square, c);

        sch_start = share1.unc_share(delta_start, x, p, sigma_p, sigmax, jacobian);
        distrepancy = 0;
        for(int sh = 0; sh< sch_start.size(); ++sh){
            cout<< sch_start[sh]<<" \t" <<sch[sh]  <<"\n";
            distrepancy += abs(sch_start[sh] - sch[sh]);
        }
        cout<< "discrepancy " << distrepancy<<endl;
        num_iter_iter++;
        discrepancy_iter = distrepancy;
    }
    double time_iterative = omp_get_wtime() - start_iterative;
    
    
//    do the same thing with direct method
    delta_start = share1.initial_guess();
    double start_direct = omp_get_wtime();
    distrepancy = 1;
    while(distrepancy > 1e-10){
        share1.evaluateFC(delta_start,c,objGrad,jac);
        share1.evaluateGA(delta_start, objGrad, jac);
    //    repack jacobian
        vector<vector<double>> jacobian_square;
        {
    //        create a head of a new vector
        int head = 0;
        for(int i=0; i<c.size(); ++i){
            vector<double> j_i(&jac[head], &jac[head+c.size()]);
            jacobian_square.push_back(j_i);
            head+=c.size();
        }
        }
    //    compute difference with next newton iteration
        delta_start = delta_start - inv_det_permute(jacobian_square)*c;

//        solve system jacobian*x = c
//        delta_start = delta_start - A_inv_b_iter(jacobian_square, c, delta_start);

        sch_start = share1.unc_share(delta_start, x, p, sigma_p, sigmax, jacobian);
        distrepancy = 0;
        for(int sh = 0; sh< sch_start.size(); ++sh){
            cout<< sch_start[sh]<<" \t" <<sch[sh]  <<"\n";
            distrepancy += abs(sch_start[sh] - sch[sh]);
        }
        cout<< "discrepancy " << distrepancy<<endl;
        num_iter_direct++;
        discrepancy_direct = distrepancy;
    }
    double time_direct = omp_get_wtime() - start_direct;
    
    cout<< "\t Direct \t iterative\n";
    cout<<"time\t" << time_direct<<"\t" <<time_iterative<<endl;
    cout<<"# iter\t" << num_iter_direct<<"\t" << num_iter_iter<<endl;
    cout<<"final\t"<< discrepancy_direct<< "\t" << discrepancy_iter<<endl;
    return 0;
    
//    return 0;
//    share1.setXInitial(share1.initial_guess());
//    for(auto it: share1.unc_share(share1.initial_guess(), x, p, sigma_p, sigmax,jacobian)){
//        cout<< it<<endl;
//    }
//    if not all of initial shares greater than zero 
    if(!all_of(sch_start.begin(), sch_start.end(), [](double it){return it>1e-4;})){
//        increase deltas so that all have positive market shares
        cout<<"not all values are above 1e-4 \n";
//        share1.get_traction();
    }
//    cout<<" value " <<share1.evaluateFC(share1.initial_guess(),c,objGrad,jac)<<endl;
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
//    share1.setXInitial(share1.initial_guess());
//    share1.get_traction();
//    cout<<"initial guess \n";
//    for(auto it :share1.initial_guess()){
//        cout<< it<<endl;
//    }
    share1.solve_for_delta();
    
    cout<< " shares at optimum "<<endl;
    std::cout.precision(4);
    std::cout << std::fixed;
    int i=0;
    for(auto it : share1.unc_share(share1.getXInitial(), x, p, sigma_p, sigmax)){
        cout<<share1.getXInitial()[i]<< " " <<it<<" "<<sch[i++]<<endl;
    }
    cout<< "final sigma x "<< endl;
//    share1.decrease_sigma_x(1/factor);
    for(auto it : share1.get_sigma_x()){
        cout<< it<<" ";
    }
    cout<< endl;
    return 0;
    
    
    
    
    
    
    
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
//    int i=0;
    
    
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
