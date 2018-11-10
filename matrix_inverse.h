/* 
 * File:   matrix_inverse.h
 * Author: roman
 *
 * Created on September 16, 2018, 8:14 PM
 * These routines are related to matrix inversions by different means
 */

#ifndef MATRIX_INVERSE_H
#define MATRIX_INVERSE_H
#include <vector>

// routine to compute determinant of matrix
double det(std::vector<std::vector<double>>  A);
double det_permutations(std::vector<std::vector<double>> A, std::vector<std::vector<int>> & permutations);

// routine returns matrix without i row and j column
std::vector<std::vector<double>> A_minus_ij(std::vector<std::vector<double>> A, int i, int j);

// routine to compute the inverse of a matrix with det(A_r)/det(A)
std::vector<std::vector<double>> inv_det(std::vector<std::vector<double>> A);

// routine to compute the inverse of a matrix with det(A_r)/det(A) computing determinants with permutations
std::vector<std::vector<double>> inv_det_permute(std::vector<std::vector<double>> A);

//routine computes the sign of permutation
int sign_permutation(std::vector<int>  b);

// routine to solve Ax=b iteratively with gauss jacobi
std::vector<double> A_inv_b_iter(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double> x = std::vector<double>(0, 0));



#endif /* MATRIX_INVERSE_H */

