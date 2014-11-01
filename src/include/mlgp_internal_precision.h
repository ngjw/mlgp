#ifndef MLGP_INTERNAL_PRECISION_H
#define MLGP_INTERNAL_PRECISION_H

#if defined BOTHPRECISION
  #if defined DOUBLE
  #define COMPILEONCE
  #endif
#else
#define COMPILEONCE
#endif

#ifdef DOUBLE

#ifndef FLOAT
#define FLOAT double
#endif

#define GESV  dgesv_
#define GETRF dgetrf_
#define GETRI dgetri_
#define POTRF dpotrf_
#define POTRI dpotri_
#define POTRS dpotrs_
#define PPSV  dppsv_
#define PPTRF dpptrf_
#define PPTRI dpptri_
#define PPTRS dpptrs_
#define PF %lf

#define MATRIX_T mlgpDMatrix_t
#define VECTOR_T mlgpDVector_t
#define MEAN_T   mlgpDMean_t
#define COV_T    mlgpDCov_t
#define LIK_T    mlgpDLik_t
#define INF_T    mlgpDInf_t
#define LIKPARAMS_T mlgpDLikParams_t

#define MLGP_CREATEVECTOR(...) mlgp_createVector_dp(__VA_ARGS__)
#define MLGP_CREATEMATRIX(...) mlgp_createMatrix_dp(__VA_ARGS__)
#define MLGP_CREATEMATRIXNOMALLOC(...) mlgp_createMatrixNoMalloc_dp(__VA_ARGS__)
#define MLGP_CREATEVECTORNOMALLOC(...) mlgp_createVectorNoMalloc_dp(__VA_ARGS__)
#define MLGP_READVECTOR(...) mlgp_readVector_dp(__VA_ARGS__)
#define MLGP_READMATRIX(...) mlgp_readMatrix_dp(__VA_ARGS__)
#define MLGP_FREEVECTOR(...) mlgp_freeVector_dp(__VA_ARGS__)
#define MLGP_FREEMATRIX(...) mlgp_freeMatrix_dp(__VA_ARGS__)
#define MLGP_CREATEMEAN(...) mlgp_createMean_dp(__VA_ARGS__)
#define MLGP_CREATECOV(...) mlgp_createCov_dp (__VA_ARGS__)
#define MLGP_CREATEINF(...) mlgp_createInf_dp (__VA_ARGS__)
#define MLGP_CREATELIK(...) mlgp_createLik_dp (__VA_ARGS__)
#define MLGP_FREEMEAN(...) mlgp_freeMean_dp(__VA_ARGS__)
#define MLGP_FREECOV(...) mlgp_freeCov_dp (__VA_ARGS__)
#define MLGP_FREEINF(...) mlgp_freeInf_dp (__VA_ARGS__)
#define MLGP_FREELIK(...) mlgp_freeLik_dp (__VA_ARGS__)

#else

#ifndef FLOAT
#define FLOAT float
#endif

#define GESV  sgesv_
#define GETRF sgetrf_
#define GETRI sgetri_
#define POTRF spotrf_
#define POTRI spotri_
#define POTRS spotrs_
#define PPSV  sppsv_
#define PPTRF spptrf_
#define PPTRI spptri_
#define PPTRS spptrs_
#define PF %f

#define MATRIX_T mlgpSMatrix_t
#define VECTOR_T mlgpSVector_t
#define MATRIX_T mlgpSMatrix_t
#define VECTOR_T mlgpSVector_t
#define MEAN_T   mlgpSMean_t
#define COV_T    mlgpSCov_t
#define LIK_T    mlgpSLik_t
#define INF_T    mlgpSInf_t
#define LIKPARAMS_T mlgpSLikParams_t

#define MLGP_CREATEVECTOR(...) mlgp_createVector_sp(__VA_ARGS__)
#define MLGP_CREATEMATRIX(...) mlgp_createMatrix_sp(__VA_ARGS__)
#define MLGP_CREATEMATRIXNOMALLOC(...) mlgp_createMatrixNoMalloc_sp(__VA_ARGS__)
#define MLGP_CREATEVECTORNOMALLOC(...) mlgp_createVectorNoMalloc_sp(__VA_ARGS__)
#define MLGP_READVECTOR(...) mlgp_readVector_sp(__VA_ARGS__)
#define MLGP_READMATRIX(...) mlgp_readMatrix_sp(__VA_ARGS__)
#define MLGP_FREEVECTOR(...) mlgp_freeVector_sp(__VA_ARGS__)
#define MLGP_FREEMATRIX(...) mlgp_freeMatrix_sp(__VA_ARGS__)
#define MLGP_CREATEMEAN(...) mlgp_createMean_sp(__VA_ARGS__)
#define MLGP_CREATECOV(...) mlgp_createCov_sp (__VA_ARGS__)
#define MLGP_CREATEINF(...) mlgp_createInf_sp (__VA_ARGS__)
#define MLGP_CREATELIK(...) mlgp_createLik_sp (__VA_ARGS__)
#define MLGP_FREEMEAN(...) mlgp_freeMean_sp(__VA_ARGS__)
#define MLGP_FREECOV(...) mlgp_freeCov_sp (__VA_ARGS__)
#define MLGP_FREEINF(...) mlgp_freeInf_sp (__VA_ARGS__)
#define MLGP_FREELIK(...) mlgp_freeLik_sp (__VA_ARGS__)

#endif

#endif /* MLGP_INTERNAL_PRECISION_H */
