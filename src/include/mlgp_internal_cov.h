#ifndef MLGP_INTERNAL_COV_H
#define MLGP_INTERNAL_COV_H

#include "../../include/mlgp.h"

#define NCP_SEiso 2
#define NCP_SEard (dim+1)

/* cov related functions */

// macro for all composite covariance functions
#define covCOMPOSITE     (covSum|covProd)

#ifdef DOUBLE
#define COVFUNC_FT covDFunc_ft
#define COVFUNCDERIV_FT covDFuncDeriv_ft
#define COVFUNCDERIVNODE_T covDFuncDerivNode_t
#define COVFUNCNODE_T covDFuncNode_t
#define COVFUNCCOMP_FT covDFuncComp_ft
#else
#define COVFUNC_FT covSFunc_ft
#define COVFUNCDERIV_FT covSFuncDeriv_ft
#define COVFUNCDERIVNODE_T covSFuncDerivNode_t
#define COVFUNCNODE_T covSFuncNode_t
#define COVFUNCCOMP_FT covSFuncComp_ft
#endif

/* function pointer for cov functions */
typedef FLOAT (*COVFUNC_FT)(
  FLOAT*,
  unsigned,
  FLOAT*,
  unsigned,
  unsigned,
  FLOAT*,
  FLOAT*
);

typedef struct COVFUNCNODE_T COVFUNCNODE_T;
struct COVFUNCNODE_T
{
  COVFUNC_FT func;
  FLOAT *params;
  COVFUNCNODE_T *next;
};

typedef FLOAT (*COVFUNCDERIV_FT)(
  FLOAT*,
  unsigned,
  FLOAT*,
  unsigned,
  unsigned,
  FLOAT*,
  unsigned,
  FLOAT*
);

typedef struct COVFUNCDERIVNODE_T COVFUNCDERIVNODE_T;
struct COVFUNCDERIVNODE_T
{
  COVFUNCDERIV_FT func;
  FLOAT *params;
  COVFUNCDERIVNODE_T *next;
};

#ifdef DOUBLE
#define MLGP_NPARAMS_COV mlgp_nparams_cov_dp
#define MLGP_COV_PARAM_TRANS mlgp_cov_param_trans_dp
#define MLGP_COV mlgp_cov_dp
#define MLGP_COV_SINGLE mlgp_cov_single_dp
#define MLGP_COV_COMPOSITE mlgp_cov_composite_dp
#define MLGP_CREATECOVFUNCLIST mlgp_CreateCovFuncList_dp
#define MLGP_FREECOVFUNCLIST mlgp_freeCovFuncList_dp
#define MLGP_COVSEISO mlgp_covSSiso_dp
#define MLGP_COVSEARD mlgp_covSSard_dp
#define MLGP_COVDUMMYZERO mlgp_covDummyZero_dp
#define MLGP_COVDUMMYONE mlgp_covDummyDne_dp
#define MLGP_COVSUM mlgp_covSum_dp
#define MLGP_COVPROD mlgp_covProd_dp
#define MLGP_COVSEARD_DERIVATIVES mlgp_covSEard_derivatives_dp
#define MLGP_COVSEISO_DERIVATIVES mlgp_covSEiso_derivatives_dp
#define MLGP_COVSUM_DERIVATIVES mlgp_covSum_derivatives_dp
#define MLGP_COVPROD_DERIVATIVES mlgp_covProd_derivatives_dp
#else
#define MLGP_NPARAMS_COV mlgp_nparams_cov_sp
#define MLGP_COV_PARAM_TRANS mlgp_cov_param_trans_sp
#define MLGP_COV mlgp_cov_sp
#define MLGP_COV_SINGLE mlgp_cov_single_sp
#define MLGP_COV_COMPOSITE mlgp_cov_composite_sp
#define MLGP_CREATECOVFUNCLIST mlgp_CreateCovFuncList_sp
#define MLGP_FREECOVFUNCLIST mlgp_freeCovFuncList_sp
#define MLGP_COVSEISO mlgp_covSSiso_sp
#define MLGP_COVSEARD mlgp_covSSard_sp
#define MLGP_COVDUMMYZERO mlgp_covDummyZero_sp
#define MLGP_COVDUMMYONE mlgp_covDummyDne_sp
#define MLGP_COVSUM mlgp_covSum_sp
#define MLGP_COVPROD mlgp_covProd_sp
#define MLGP_COVSEARD_DERIVATIVES mlgp_covSEard_derivatives_sp
#define MLGP_COVSEISO_DERIVATIVES mlgp_covSEiso_derivatives_sp
#define MLGP_COVSUM_DERIVATIVES mlgp_covSum_derivatives_sp
#define MLGP_COVPROD_DERIVATIVES mlgp_covProd_derivatives_sp

#endif

typedef FLOAT (*COVFUNCCOMP_FT)(
  FLOAT*,
  unsigned,
  FLOAT*,
  unsigned,
  unsigned,
  FLOAT*,
  FLOAT*,
  COVFUNCNODE_T*
);

void MLGP_COV_PARAM_TRANS(
  COV_T cov,
  unsigned dim
);

/* main cov function wrapper */
mlgpStatus_t MLGP_COV(
  MATRIX_T K,
  MATRIX_T X1,
  MATRIX_T X2,
  COV_T cov,
  MATRIX_T dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* cov function for single cov functions */
mlgpStatus_t MLGP_COV_SINGLE(
  MATRIX_T K,
  MATRIX_T X1,
  MATRIX_T X2,
  COV_T cov,
  MATRIX_T dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* cov function for composite cov functions */
mlgpStatus_t MLGP_COV_COMPOSITE(
  MATRIX_T K,
  MATRIX_T X1,
  MATRIX_T X2,
  COV_T cov,
  MATRIX_T dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* creates the linked list of cov functions and their parameters */
COVFUNCNODE_T *MLGP_CREATECOVFUNCLIST(
  COV_T cov,
  unsigned dim,
  unsigned covToSkip,
  unsigned zeroOrOne
);

void MLGP_FREECOVFUNCLIST(
  COVFUNCNODE_T *list
);

/* cov functions */
FLOAT MLGP_COVSEISO(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
);

FLOAT MLGP_COVSEARD(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
);

FLOAT MLGP_COVDUMMYZERO(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
);

FLOAT MLGP_COVDUMMYONE(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
);

FLOAT MLGP_COVSUM(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff,
  COVFUNCNODE_T *funcs
);

FLOAT MLGP_COVPROD(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff,
  COVFUNCNODE_T *funcs
);

/* derivative functions */
FLOAT MLGP_COVSEARD_DERIVATIVES(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  unsigned param_i,
  FLOAT *diff
);

FLOAT MLGP_COVSEISO_DERIVATIVES(
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  unsigned param_i,
  FLOAT *diff
);

#endif /* MLGP_INTERNAL_COV_H */
