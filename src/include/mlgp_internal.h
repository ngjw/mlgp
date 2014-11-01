#ifndef MLGP_INTERNAL_H
#define MLGP_INTERNAL_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "../../include/mlgp.h"

#include "mlgp_internal_precision.h"
#include "mlgp_internal_cov.h"
#include "mlgp_internal_mean.h"
#include "mlgp_internal_consts.h"
#include "mlgp_internal_linalg.h"
#include "mlgp_internal_options.h"
#include "mlgp_internal_blas.h"
#include "mlgp_internal_misc.h"

#ifdef DOUBLE
#define MLGP_LIKELIHOOD(...) mlgp_likelihood_dp(__VA_ARGS__)
#define MLGP_LIKELIHOOD_CPU(...) mlgp_likelihood_cpu_dp(__VA_ARGS__)

#define MLGP_PREDICT(...) mlgp_predict_dp(__VA_ARGS__)
#define MLGP_PREDICT_CPU(...) mlgp_predict_cpu_dp(__VA_ARGS__)
#else
#define MLGP_LIKELIHOOD(...) mlgp_likelihood_sp(__VA_ARGS__)
#define MLGP_LIKELIHOOD_CPU(...) mlgp_likelihood_cpu_sp(__VA_ARGS__)

#define MLGP_PREDICT(...) mlgp_predict_sp(__VA_ARGS__)
#define MLGP_PREDICT_CPU(...) mlgp_predict_cpu_sp(__VA_ARGS__)
#endif

mlgpStatus_t MLGP_PREDICT_CPU(
  MATRIX_T X,
  VECTOR_T y,
  MATRIX_T Xs,
  VECTOR_T ymu,
  VECTOR_T ys2,
  VECTOR_T fmu,
  VECTOR_T fs2,
  INF_T inf,
  MEAN_T mean,
  COV_T cov,
  LIK_T lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

mlgpStatus_t MLGP_LIKELIHOOD_CPU(
  FLOAT* nll,
  MATRIX_T X,
  VECTOR_T y,
  INF_T inf,
  MEAN_T mean,
  COV_T cov,
  LIK_T lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);


#endif /* MLGP_INTERNAL_H */
