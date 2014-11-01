#ifndef MLGP_INTERNAL_BLAS_H
#define MLGP_INTERNAL_BLAS_H

#ifndef FLOAT
#ifdef DOUBLE
#define FLOAT double
#else
#define FLOAT float
#endif
#endif

#ifdef DOUBLE
#define MLGP_DOT(...)  mlgp_ddot(__VA_ARGS__)
#define MLGP_COPY(...) mlgp_dcopy(__VA_ARGS__)
#define MLGP_SCAL(...) mlgp_dscal(__VA_ARGS__)
#define MLGP_AXPY(...) mlgp_daxpy(__VA_ARGS__)
#define MLGP_GEMV(...) mlgp_dgemv(__VA_ARGS__)
#define MLGP_GEMM(...) mlgp_dgemm(__VA_ARGS__)
#define MLGP_SPR(...)  mlgp_dspr(__VA_ARGS__)
#else
#define MLGP_DOT(...)  mlgp_sdot(__VA_ARGS__)
#define MLGP_COPY(...) mlgp_scopy(__VA_ARGS__)
#define MLGP_SCAL(...) mlgp_sscal(__VA_ARGS__)
#define MLGP_AXPY(...) mlgp_saxpy(__VA_ARGS__)
#define MLGP_GEMV(...) mlgp_sgemv(__VA_ARGS__)
#define MLGP_GEMM(...) mlgp_sgemm(__VA_ARGS__)
#define MLGP_SPR(...)  mlgp_sspr(__VA_ARGS__)
#endif

// BLAS function prototypes

#ifdef DOUBLE
FLOAT ddot_(unsigned*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void dcopy_(unsigned*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void dscal_(unsigned*,FLOAT*,FLOAT*,unsigned*);
void daxpy_(unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void dgemv_(char*,unsigned*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*,FLOAT*,FLOAT*,unsigned*);
void dgemm_(char*,char*,unsigned*,unsigned*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*,FLOAT*,FLOAT*,unsigned*);
void dspr_(char*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*);
#else
FLOAT sdot_(unsigned*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void scopy_(unsigned*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void sscal_(unsigned*,FLOAT*,FLOAT*,unsigned*);
void saxpy_(unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*);
void sgemv_(char*,unsigned*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*,FLOAT*,FLOAT*,unsigned*);
void sgemm_(char*,char*,unsigned*,unsigned*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*,unsigned*,FLOAT*,FLOAT*,unsigned*);
void sspr_(char*,unsigned*,FLOAT*,FLOAT*,unsigned*,FLOAT*);
#endif


FLOAT mlgp_sdot(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_scopy(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_sscal(unsigned N, FLOAT a, FLOAT* X, unsigned incX);
void mlgp_saxpy(unsigned N, FLOAT a, FLOAT* X, 
               unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_sgemv(char transA, unsigned M, unsigned N, FLOAT a,
               FLOAT* A, unsigned LDA, FLOAT* X, unsigned incX, FLOAT b,
               FLOAT* Y, unsigned incY);
void mlgp_sgemm(char transA, char transB, unsigned M, unsigned N, unsigned K,
               FLOAT a, FLOAT* A, unsigned LDA, FLOAT* B, unsigned LDB,
               FLOAT b, FLOAT* C, unsigned LDC);
void mlgp_sspr(char UPLO, unsigned N, FLOAT a,
               FLOAT* X, unsigned incX, FLOAT* AP);


FLOAT mlgp_ddot(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_dcopy(unsigned N, FLOAT* X, unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_dscal(unsigned N, FLOAT a, FLOAT* X, unsigned incX);
void mlgp_daxpy(unsigned N, FLOAT a, FLOAT* X, 
               unsigned incX, FLOAT* Y, unsigned incY);
void mlgp_dgemv(char transA, unsigned M, unsigned N, FLOAT a,
               FLOAT* A, unsigned LDA, FLOAT* X, unsigned incX, FLOAT b,
               FLOAT* Y, unsigned incY);
void mlgp_dgemm(char transA, char transB, unsigned M, unsigned N, unsigned K,
               FLOAT a, FLOAT* A, unsigned LDA, FLOAT* B, unsigned LDB,
               FLOAT b, FLOAT* C, unsigned LDC);
void mlgp_dspr(char UPLO, unsigned N, FLOAT a,
               FLOAT* X, unsigned incX, FLOAT* AP);

#endif /* MLGP_INTERNAL_BLAS_H */
