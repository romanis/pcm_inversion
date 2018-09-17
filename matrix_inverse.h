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

// routine returns matrix without i row and j column
std::vector<std::vector<double>> A_minus_ij(std::vector<std::vector<double>> A, int i, int j);

// routine to compute the inverse of a matrix with det(A_r)/det(A)
std::vector<std::vector<double>> inv_det(std::vector<std::vector<double>> A);



#endif /* MATRIX_INVERSE_H */

