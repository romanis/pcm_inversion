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
            A_inv[i][j] = det(A_ij)/det_A;
        }
    }
    
    return A_inv;
}
