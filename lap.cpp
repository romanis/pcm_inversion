#include <stdlib.h>
#include <stdio.h>
#include "mkl.h"

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda );
extern void print_int_vector( char* desc, MKL_INT n, MKL_INT* a );

/* Parameters */
#define N 5
#define NRHS 3
#define LDA N
#define LDB N

/* Main program */
int main() {
	/* Locals */
	MKL_INT n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
	/* Local arrays */
	MKL_INT ipiv[N];
	double a[LDA*N] = {
	    2, 0,  0,  0,  0,
	    0, 2,  0,  0,  0,
	    0, 0,  2,  0,  0,
	    0, 0,  0,  2,  0,
	    0, 0,  0,  0,  2
	};
	double b[LDB*NRHS] = {
	    4.02,  6.19, -8.22, -7.57, -3.03,
	   -1.56,  4.00, -8.67,  1.75,  2.86,
	    9.81, -4.09, -4.57, -8.61,  8.99
	};
	/* Executable statements */
	//printf( "LAPACKE_dgesv (column-major, high-level) Example Program Results\n" );
	/* Solve the equations A*X = B */
	//info = LAPACKE_dgesv( LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv, b, ldb );
    int info1 = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, a, LDA, ipiv);
    info = LAPACKE_dgetri(LAPACK_COL_MAJOR, N, a, LDA, ipiv);
	/* Check for the exact singularity */
	if( info > 0 ) {
		printf( "The diagonal element of the triangular factor of A,\n" );
		printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
		printf( "the solution could not be computed.\n" );
		//exit( 1 );
	}
	/* Print solution */
	//print_matrix( "Solution", n, nrhs, b, ldb );
	/* Print details of LU factorization */
	print_matrix( "Details of LU factorization", n, n, a, lda );
	/* Print pivot indices */
	print_int_vector( "Pivot indices", n, ipiv );
	exit( 0 );
} /* End of LAPACKE_dgesv Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
	MKL_INT i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
		printf( "\n" );
	}
}

/* Auxiliary routine: printing a vector of integers */
void print_int_vector( char* desc, MKL_INT n, MKL_INT* a ) {
	MKL_INT j;
	printf( "\n %s\n", desc );
	for( j = 0; j < n; j++ ) printf( " %6i", a[j] );
	printf( "\n" );
}
