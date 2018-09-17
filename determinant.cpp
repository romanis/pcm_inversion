/*
 * C++ Program to Find the Determinant of a Given Matrix
 */
#include<iostream>
#include<math.h>
#include<stdio.h>

using namespace std;

double det(int n, double * A)
{
    double d = 0;
//    this function finds determinant recursively 
    
    int c, subi, i, j, subj;
//    create an empty sub-matrix that will be filled and used to find determinant
    double sub_A[(n-1)*(n-1)]; 
//    if n is 2, trivial case
    if (n == 2) 
    {
        return( (A[0] * A[3]) - (A[1] * A[2]));
    }
//    else, create a sub-matrix and use determinant on it
    else
    {  
//        loop over all elements of the first row and exclude that colunm from the matrix, also exclude the first row
        for(c = 0; c < n; c++)
        {  
//            index of submatrix's first row
            subi = 0;  
//            loop over all rows of original matrix except the frist one
            for(i = 1; i < n; i++)
            {  
//                index of the sub-matrix columns 
                subj = 0;
//                loop over matrix A's columns 
                for(j = 0; j < n; j++)
                {    
//                    skip the c's column
                    if (j == c)
                    {
                        continue;
                    }
//                    copy elements of matrix A to the submatrix
                    sub_A[subi*(n-1) + subj] = A[i*n+j];
                    subj++;
                }
                subi++;
            }
//            recursive formula for the determinant
        d = d + (pow(-1 ,c) * A[c] * det(n - 1 ,sub_A));
        }
    }
    return d;
}
int main()
{
    int n;
    cout<<"enter the order of matrix\n" ;
    cin>>n;
    double mat[100];
    int i, j;
    cout<<"enter the elements"<<endl;
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            cin>>mat[i*n + j];
        }
    }
    cout<<"\ndeterminant\n"<<det(n,mat)<<endl;
}