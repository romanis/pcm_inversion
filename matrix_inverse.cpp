/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   matrix_inverse.cpp
 * Author: roman
 * 
 * Created on September 16, 2018, 8:14 PM
 */

#include "matrix_inverse.h"
#include <vector>
#include<iostream>
#include<math.h>
#include<stdio.h>
#include <algorithm>
#include <omp.h>

using namespace std;

double det(vector<vector<double> > A)
{
    int n = A.size();
    double d = 0;
//    this function finds determinant recursively 
    
    int c, subi, i, j, subj;
//    if n is 2, trivial case
    if (n == 2) 
    {
        return( (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]));
    }
//    also write manually for 3x3 minor
    else if(n ==3){
        return(A[0][0]*A[1][1]*A[2][2]+A[0][2]*A[1][0]*A[2][1] + A[0][1]*A[1][2]*A[2][0] - A[0][2]*A[1][1]*A[2][0] - A[0][1]*A[1][0]*A[2][2] - A[0][0]*A[1][2]*A[2][1]);
    }
//    else, create a sub-matrix and use determinant on it
    else
    {  
//        loop over all elements of the first row and exclude that colunm from the matrix, also exclude the first row
        for(c = 0; c < n; c++)
        { 
            vector<vector<double> > sub_A = A_minus_ij(A, 0, c); 
//            recursive formula for the determinant
            d = d + (pow(-1 ,c) * A[0][c] * det(sub_A));
        }
    }
    return d;
}

double det_permutations(std::vector<std::vector<double> > A, std::vector<std::vector<int>> & permutations){
//    loop over all permutations and multiply the elements of A
    double d = 0;
    vector<int> row_index;
//    create row and column indexes, then permute column index with std::next_permutation(.,.)
    for(int i=0; i<A.size(); ++i){
        row_index.push_back(i);
    }
    
//    cout<< "time to pre-compute permutations " << omp_get_wtime() - start<<endl;
//    create +-1 factor, next_permutation changes it every two permutations
    
    int count_permutations = 0;
//    until there exists next permutation 
#pragma omp parallel for num_threads(8) schedule(dynamic,10) reduction(+:d)
    for(int i = 0; i<permutations.size(); ++i){
        vector<int> col_index = permutations[i];
//        determine the signature of permutation
//        int num_permutations =0;
//        for(int i = 0; i< col_index.size()-1; ++i){
//            for(int j=i+1; j<col_index.size(); ++j){
//                if(col_index[j]<col_index[i]){
//                    count_permutations++;
//                }
//            }
//        }
//        int factor = ((count_permutations % 2 == 0) ? 1 : -1);
        int factor = sign_permutation(col_index);
//        walk over all col and row indexes and multiply elements of A, factor changes every two iterations
        double element = 1;
//        compute this element of product
        for(int i=0; i< row_index.size(); i++){
            element *= A[row_index[i]][col_index[i]];
        }
//        add element to determinant
        d += element*factor;
        
    }
    return d;
}


int sign_permutation(std::vector<int> b){
    vector<bool> visited(b.size(), false); 
    int sign =1;
    for(int i=0; i<b.size(); ++i){
        if(visited[i] == false){
            int next = i;
            int L=0;
            while(!visited[next]){
                L++;
                visited[next] =  true;
                next = b[next];
            }
            if(L % 2 == 0){
                sign *= -1;
            }
        }
    }
    
    return sign;
}

vector<vector<double>>  A_minus_ij(vector<vector<double>> A, int i_m, int j_m){
    int n = A.size();
//    create a submatrix 
    vector<vector<double> > sub_A = vector<vector<double> > (n-1, vector<double>(n-1,0));
//    loop over all elements of A to copy all row but i, all columns but j
    int subi=0;
    int subj=0;
    for(int i = 0; i < n; i++){  
//                index of the sub-matrix columns 
        if(i == i_m){
            continue;
        }
        subj = 0;
//                loop over matrix A's columns 
        for(int j = 0; j < n; j++)
        {    
//                    skip the c's column
            if (j == j_m)
            {
                continue;
            }
//                    copy elements of matrix A to the submatrix
            sub_A[subi][subj] = A[i][j];
            subj++;
        }
        subi++;
    }
    return sub_A;
}

vector<vector<double>> inv_det(vector<vector<double>> A){
    int n = A.size();
//    create an empty matrix 
    vector<vector<double>> A_inv =  vector<vector<double> > (n, vector<double>(n, 0)); 
//    compute determinant of A
    double det_A = det(A);
//    loop over all coefficients of inverse matrix and compute determinants of matrix without that row and column
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){           
//            create submatrix without ith row and jth column
            vector<vector<double>> A_ij = A_minus_ij(A,i,j);
            A_inv[j][i] = pow(-1, i+j+2)*det(A_ij)/det_A;
        }
    }
    
    return A_inv;
}

vector<vector<double>> inv_det_permute(vector<vector<double>> A){
    int n = A.size();
//    create an empty matrix 
    vector<vector<double>> A_inv =  vector<vector<double> > (n, vector<double>(n, 0)); 
    
//    pre-compute permutations once here
    vector<int> col_index;
//    create row and column indexes, then permute column index with std::next_permutation(.,.)
    for(int i=0; i<A.size(); ++i){
        col_index.push_back(i);
    }
//    pre-compute permutations and store them
    vector<vector<int>> indexes;
    indexes.push_back(col_index);
//    double start = omp_get_wtime();
    while(next_permutation(col_index.begin(), col_index.end())){
        indexes.push_back(col_index);
    }
//    pre-compute permutations for submatrixes
    vector<int> col_index_sub;
//    create row and column indexes, then permute column index with std::next_permutation(.,.)
    for(int i=0; i<A.size()-1; ++i){
        col_index_sub.push_back(i);
    }
//    pre-compute permutations and store them
    vector<vector<int>> indexes_sub;
    indexes_sub.push_back(col_index);
//    double start = omp_get_wtime();
    while(next_permutation(col_index_sub.begin(), col_index_sub.end())){
        indexes_sub.push_back(col_index_sub);
    }
    
    
//    compute determinant of A
    double det_A = det_permutations(A, indexes);
//    loop over all coefficients of inverse matrix and compute determinants of matrix without that row and column
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){           
//            create submatrix without ith row and jth column
            vector<vector<double>> A_ij = A_minus_ij(A,i,j);
            A_inv[j][i] = pow(-1, i+j+2)*det_permutations(A_ij, indexes_sub)/det_A;
        }
    }
    
    return A_inv;
}

std::vector<double> A_inv_b_iter(std::vector<std::vector<double>> A, std::vector<double> b, vector<double> x){
//    run intil convergence solving each linear equality at a time. 
    double discrepancy = 1;
    int iter_num = 0;
//    if x was not supplied, make it a vector of zeros
//    if(x.size() == 0){
//        x = vector<double> (A.size(), 1);
//    }
//    random filling
    if(x.size() == 0){
        for(int i=0; i<A.size(); ++i){
            x.push_back((double)rand() / RAND_MAX);
        }
    }
//    for gauss jacobi use x_old
    vector<double> x_old = x;
    while (discrepancy > 1e-12 & iter_num++ < 1000){
//        create a local measure of max discreapancy 
        double max_discrepancy = 0;
//        at each iteration loop over rows of A and at each subiteration solve for x_i conitional on all other x_{-i}
#pragma omp parallel for reduction(max : discrepancy) num_threads(1)
        for(int i=0; i<A.size(); ++i){
//            assume that A[i][i] != 0
//            compute b[i] - sum(a[ij]*x[j] j!=i
            double rhs = b[i];
            for(int j=0; j<A.size(); ++j){
                if(j==i){
                    continue;
                }
                else{
                    rhs -= A[i][j] * x_old[j];
                }
            }
//            if discrepancy between A[i][i]*x[i] and rhs is greater than max discrepancy, replace the max discrepancy 
            if(abs(A[i][i]*x[i] - rhs) > max_discrepancy){
                max_discrepancy = abs(A[i][i]*x[i] - rhs);
            }
//            update x[i]
            x[i] = rhs / A[i][i];
        }
//        update discrepancy 
        discrepancy = max_discrepancy;
        x_old = x;
//        cout<< "max discrepancy is " << discrepancy<<endl;
    }
    
    return x;
}