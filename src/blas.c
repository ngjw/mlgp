#include "../include/mlgp.h"
#include "include/mlgp_internal.h"


FLOAT MLGP_DOT(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY)
{
  #ifdef DOUBLE
  #define DOT(...) ddot_(__VA_ARGS__)
  #else
  #define DOT(...) sdot_(__VA_ARGS__)
  #endif

  return DOT(&N, X, &incX, Y, &incY);
}

void MLGP_COPY(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY)
{
  #ifdef DOUBLE
  #define COPY(...) dcopy_(__VA_ARGS__)
  #else
  #define COPY(...) scopy_(__VA_ARGS__)
  #endif

  return COPY(&N, X, &incX, Y, &incY);
}

void MLGP_SCAL(unsigned N, FLOAT a, FLOAT* X, unsigned incX)
{
  #ifdef DOUBLE
  #define SCAL(...) dscal_(__VA_ARGS__)
  #else
  #define SCAL(...) sscal_(__VA_ARGS__)
  #endif
  return SCAL(&N, &a, X, &incX);
}

void MLGP_AXPY(unsigned N, FLOAT a, FLOAT* X, 
               unsigned incX, FLOAT* Y, unsigned incY)
{
  #ifdef DOUBLE
  #define AXPY(...) daxpy_(__VA_ARGS__)
  #else
  #define AXPY(...) saxpy_(__VA_ARGS__)
  #endif
  return AXPY(&N, &a, X, &incX, Y, &incY);
  
}

void MLGP_GEMV(char transA, unsigned M, unsigned N, FLOAT a,
               FLOAT* A, unsigned LDA, FLOAT* X, unsigned incX, FLOAT b,
               FLOAT* Y, unsigned incY)
{
  #ifdef DOUBLE
  #define GEMV(...) dgemv_(__VA_ARGS__)
  #else
  #define GEMV(...) sgemv_(__VA_ARGS__)
  #endif
  return GEMV(&transA, &M, &N, &a, A, &LDA, X, &incX, &b, Y, &incY);
}

void MLGP_GEMM(char transA, char transB, unsigned M, unsigned N, unsigned K,
               FLOAT a, FLOAT* A, unsigned LDA, FLOAT* B, unsigned LDB,
               FLOAT b, FLOAT* C, unsigned LDC)
{
  #ifdef DOUBLE
  #define GEMM(...) dgemm_(__VA_ARGS__)
  #else
  #define GEMM(...) sgemm_(__VA_ARGS__)
  #endif
  return GEMM(&transA, &transB, &M, &N, &K, &a, A, &LDA, B, &LDB, &b, C, &LDC);
  
}

void MLGP_SPR(char UPLO, unsigned N, FLOAT a,
               FLOAT* X, unsigned incX, FLOAT* AP)
{
  #ifdef DOUBLE
  #define SPR(...) dspr_(__VA_ARGS__)
  #else
  #define SPR(...) sspr_(__VA_ARGS__)
  #endif
  return SPR(&UPLO, &N, &a, X, &incX, AP);
}
