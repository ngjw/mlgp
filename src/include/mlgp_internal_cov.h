#ifndef MLGP_INTERNAL_COV_H
#define MLGP_INTERNAL_COV_H

#include "../../include/mlgp.h"

#define NCP_SEiso 2
#define NCP_SEard (dim+1)

/* cov related functions */

// macro for all composite covariance functions
#define covCOMPOSITE     (covSum|covProd)

/* function pointer for cov functions */
typedef mlgpFloat_t (*covFunc_ft)(
  mlgpFloat_t*,
  unsigned,
  mlgpFloat_t*,
  unsigned,
  unsigned,
  mlgpFloat_t*,
  mlgpFloat_t*
);

typedef struct covFuncNode_t covFuncNode_t;
struct covFuncNode_t
{
  covFunc_ft func;
  mlgpFloat_t *params;
  covFuncNode_t *next;
};

typedef mlgpFloat_t (*covFuncDeriv_ft)(
  mlgpFloat_t*,
  unsigned,
  mlgpFloat_t*,
  unsigned,
  unsigned,
  mlgpFloat_t*,
  unsigned,
  mlgpFloat_t*
);

typedef struct covFuncDerivNode_t covFuncDerivNode_t;
struct covFuncDerivNode_t
{
  covFuncDeriv_ft func;
  mlgpFloat_t *params;
  covFuncDerivNode_t *next;
};

typedef mlgpFloat_t (*covFuncComp_ft)(
  mlgpFloat_t*,
  unsigned,
  mlgpFloat_t*,
  unsigned,
  unsigned,
  mlgpFloat_t*,
  mlgpFloat_t*,
  covFuncNode_t*
);

void mlgp_cov_param_trans(
  mlgpCov_t cov,
  unsigned dim
);

/* main cov function wrapper */
mlgpStatus_t mlgp_cov(
  mlgpMatrix_t K,
  mlgpMatrix_t X1,
  mlgpMatrix_t X2,
  mlgpCov_t cov,
  mlgpMatrix_t dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* cov function for single cov functions */
mlgpStatus_t mlgp_cov_single(
  mlgpMatrix_t K,
  mlgpMatrix_t X1,
  mlgpMatrix_t X2,
  mlgpCov_t cov,
  mlgpMatrix_t dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* cov function for composite cov functions */
mlgpStatus_t mlgp_cov_composite(
  mlgpMatrix_t K,
  mlgpMatrix_t X1,
  mlgpMatrix_t X2,
  mlgpCov_t cov,
  mlgpMatrix_t dK,
  unsigned param_i,
  mlgpOptions_t options
);

/* creates the linked list of cov functions and their parameters */
covFuncNode_t *mlgp_createCovFuncList(
  mlgpCov_t cov,
  unsigned dim,
  unsigned covToSkip,
  unsigned zeroOrOne
);

void mlgp_freeCovFuncList(
  covFuncNode_t *list
);

/* cov functions */
mlgpFloat_t mlgp_covSEiso(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
);

mlgpFloat_t mlgp_covSEard(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
);

mlgpFloat_t mlgp_covDummyZero(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
);

mlgpFloat_t mlgp_covDummyOne(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
);

mlgpFloat_t mlgp_covSum(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff,
  covFuncNode_t *funcs
);

mlgpFloat_t mlgp_covProd(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff,
  covFuncNode_t *funcs
);

/* derivative functions */
mlgpFloat_t mlgp_covSEard_derivatives(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  unsigned param_i,
  mlgpFloat_t *diff
);

mlgpFloat_t mlgp_covSEiso_derivatives(
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  unsigned param_i,
  mlgpFloat_t *diff
);

#endif /* MLGP_INTERNAL_COV_H */
