#ifndef MLGP_INTERNAL_PRECISION_H
#define MLGP_INTERNAL_PRECISION_H

#ifdef DOUBLE
#define CBLAS_COPY cblas_dcopy
#define CBLAS_AXPY cblas_daxpy
#define CBLAS_GEMM cblas_dgemm
#define CBLAS_GEMV cblas_dgemv
#define CBLAS_DOT  cblas_ddot
#define CBLAS_SCAL cblas_dscal
#define CBLAS_SPR  cblas_dspr
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

#else

#define CBLAS_COPY cblas_scopy
#define CBLAS_AXPY cblas_saxpy
#define CBLAS_GEMM cblas_sgemm
#define CBLAS_GEMV cblas_sgemv
#define CBLAS_DOT  cblas_sdot
#define CBLAS_SCAL cblas_sscal
#define CBLAS_SPR  cblas_sspr
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

#endif

#endif /* MLGP_INTERNAL_PRECISION_H */
