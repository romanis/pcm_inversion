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

std::vector<double> A_inv_b_iter(std::vector<std::vector<double>> A, std::vector<double> b, vector<double> x){
//    run intil convergence solving each linear equality at a time. 
    double discrepancy = 1;
    int iter_num = 0;
//    if x was not supplied, make it a vector of zeros
    if(x.size() == 0){
        x = vector<double> (A.size(), 1);
    }
//    for gauss jacobi use x_old
    vector<double> x_old = x;
    while (discrepancy > 1e-12 & iter_num++ < 1000){
//        create a local measure of max discreapancy 
        double max_discrepancy = 0;
//        at each iteration loop over rows of A and at each subiteration solve for x_i conitional on all other x_{-i}
        for(int i=0; i<A.size(); ++i){
//            assume that A[i][i] != 0
//            compute b[i] - sum(a[ij]*x[j] j!=i
            double rhs = b[i];
            for(int j=0; j<A.size(); ++j){
                if(j==i){
                    continue;
                }
                else{
                    rhs -= A[i][j] * x[j];
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