#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

#include "include/lapacke.h"

/* This file contains all the auxiliary functions required in the MLGP library */

// TODO: error handling

void chol(mlgpMatrix_t A)
{

  /* computes the Cholesky factorization of A */

  int info;
  int N = (int)A.nrows;
  char uplo = 'L';

  POTRF(&uplo,&N,A.m,&N,&info);

  // fill the upper triangular
  for(int i=0;i<N;i++){
  for(int j=i+1;j<N;j++){
    A.m[i+j*N] = 0.0;
  }}

}

void chol_packed(mlgpMatrix_t A)
{

  /* computes the Cholesky factorization of A */

  int info;
  int N = (int)A.nrows;
  char uplo = 'U';

  PPTRF(&uplo,&N,A.m,&info);

}

void inv_chol(mlgpMatrix_t L)
{
  
  /* computes the inverse of a symmetric square NxN 
   * matrix using its Cholesky factorization (in place) */

  int info;
  int N = (int)L.nrows;
  char uplo = 'L';

  POTRI(&uplo,&N,L.m,&N,&info);

  for(int i=0;i<N;i++){
  for(int j=i+1;j<N;j++){
    L.m[i+N*j] = L.m[j+N*i];
  }}

}

void inv_chol_packed(mlgpMatrix_t L)
{
  
  /* computes the inverse of a symmetric square NxN 
   * matrix using its Cholesky factorization (in place) */

  int info;
  int N = (int)L.nrows;
  char uplo = 'U';

  PPTRI(&uplo,&N,L.m,&info);

}

void solve_chol_multiple(mlgpMatrix_t L, mlgpMatrix_t B)
{

  /* solves Ax = B where A = LL^T */

  int info;
  int N = (int)L.nrows;
  int Nrhs = (int)B.ncols;
  char uplo = 'L';

  POTRS(&uplo,&N,&Nrhs,L.m,&N,B.m,&N,&info);

}

void solve_chol_one(mlgpMatrix_t L, mlgpVector_t B)
{

  /* solves Ax = B where A = LL^T */

  int info;
  int N = (int)L.nrows;
  int Nrhs = (int)1;
  char uplo = 'L';

  POTRS(&uplo,&N,&Nrhs,L.m,&N,B.v,&N,&info);

}

void solve_chol_packed_multiple(mlgpMatrix_t L, mlgpMatrix_t B)
{

  /* solves Ax = B where A = LL^T */

  int info;
  int N = (int)L.nrows;
  int Nrhs = (int)B.ncols;
  char uplo = 'U';

  PPTRS(&uplo,&N,&Nrhs,L.m,B.m,&N,&info);

}

void solve_chol_packed_one(mlgpMatrix_t L, mlgpVector_t B)
{

  /* solves Ax = B where A = LL^T */

  int info;
  int N = (int)L.nrows;
  int Nrhs = 1;
  char uplo = 'U';

  PPTRS(&uplo,&N,&Nrhs,L.m,B.v,&N,&info);

}

mlgpFloat_t log_det_tr(mlgpMatrix_t A)
{

  /* computes the log determinant of a triangular matrix */

  int N = A.nrows;
  mlgpFloat_t logdet = 0;
  for(int i=0;i<N;i++){ logdet+=log(A.m[i*(N+1)]); }
  return logdet;
}

mlgpFloat_t log_det_tr_packed(mlgpMatrix_t A)
{

  /* computes the log determinant of a packed upper triangular matrix */

  int N = A.nrows;
  mlgpFloat_t logdet = 0;
  int skip = 1;
  int count = 0;
  for(int i=0;i<N;i++){
    logdet+=log(A.m[count+i]);
    count+=skip;
    skip++;
  }
  return logdet;
}
