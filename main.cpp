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
    int dim = 7;
    int num_prod = 10;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed = 1000001;
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
    share1.set_grid(dim, 3);
      
//      cond_share(delta,p,sigma_p, jacobian);
//      vector<double> val1 = share1.unc_share(delta, x, p, sigma_p, sigmax,jacobian);
//      vector<double> val2 = share1.unc_share(delta_p, x, p, sigma_p, sigmax,jacobian);
//      cout<<"numerical jac " <<(val1[1]-val2[1])*1e4<<endl;
//    print_jacobian(jacobian);
    double start = omp_get_wtime();
    
    cout<< "shares at start \n";
    vector<double> sch_start = share1.unc_share(share1.initial_guess(), jacobian);
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
//    return 0;
    
//    ################################
//    parameter controling direct of interative solver
    bool direct = true;
    
    while(distrepancy > 1e-10 & !direct){

        vector<vector<double>> jacobian_square;
        
        vector<double> sh_pred = share1.unc_share(delta_start, jacobian_square);
//        sh_pred = share1.unc_share(delta_start);
        c.clear();
        for(int i=0; i< sh_pred.size(); ++i){
            c.push_back(sh_pred[i] - sch[i]);
            
        }
    //    compute difference with next newton iteration
//        delta_start = delta_start - inv_det(jacobian_square)*c;

//        solve system jacobian*x = c
        delta_start = delta_start - A_inv_b_iter(jacobian_square, c);

        sch_start = sh_pred;
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
    cout<< "time " << time_iterative<<endl;
//    return 0;
//    do the same thing with direct method
    delta_start = share1.initial_guess();
    double start_direct = omp_get_wtime();
    distrepancy = 1;
    while(distrepancy > 1e-10 & direct){

        vector<vector<double>> jacobian_square;
        
        vector<double> sh_pred = share1.unc_share(delta_start, jacobian_square);
        c.clear();
        for(int i=0; i< sh_pred.size(); ++i){
            c.push_back(sh_pred[i] - sch[i]);
            
        }
    //    compute difference with next newton iteration
//        delta_start = delta_start - inv_det(jacobian_square)*c;

//        solve system jacobian*x = c
        delta_start = delta_start - inv_det_permute(jacobian_square)*c;

        sch_start = share1.unc_share(delta_start);
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
    
}
