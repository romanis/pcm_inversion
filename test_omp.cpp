#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRSolver.h"
#include "/opt/aci/sw/knitro/10.2.1/examples/C++/include/KTRProblem.h"
#include <iostream>
#include "pcm_market_share.h"
#include "/storage/home/rji5040/work/Tasmanian_run/include/TasmanianSparseGrid.hpp"
//#include <bits/stdc++.h>
#include <random>
#include <numeric>
#include <string>
#include <functional>
#include <algorithm>    // std::all_of
#include <array>
#include <omp.h>

using namespace std;

  int main(int argc, char *argv[]) {
    
    vector<double> tmp;
    for(int i=0; i<10000000; ++i){
        tmp.push_back(i);
    }
    double start = omp_get_wtime();
#pragma omp parallel for num_threads(20)
    for(int i=0; i<10000000; ++i){
        tmp[i] = exp(i);
    }
    cout<<"time " << omp_get_wtime() - start<<endl;
    return 0;
  }
