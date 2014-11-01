#ifndef MLGP_INTERNAL_MEAN_H
#define MLGP_INTERNAL_MEAN_H

#include "../../include/mlgp.h"

/* mean functions */

#ifdef DOUBLE
#define MLGP_NPARAMS_MEAN(...) mlgp_nparams_mean_dp(__VA_ARGS__)
#define MLGP_MEAN(...) mlgp_mean_dp(__VA_ARGS__)
#define MLGP_MEANONE(...) mlgp_meanOne_dp(__VA_ARGS__)
#define MLGP_MEANZERO(...) mlgp_meanZero_dp(__VA_ARGS__)
#define MLGP_MEANCONST(...) mlgp_meanConst_dp(__VA_ARGS__)
#define MLGP_MEANLINEAR(...) mlgp_meanLinear_dp(__VA_ARGS__)
#else
#define MLGP_NPARAMS_MEAN(...) mlgp_nparams_mean_sp(__VA_ARGS__)
#define MLGP_MEAN(...) mlgp_mean_sp(__VA_ARGS__)
#define MLGP_MEANONE(...) mlgp_meanOne_sp(__VA_ARGS__)
#define MLGP_MEANZERO(...) mlgp_meanZero_sp(__VA_ARGS__)
#define MLGP_MEANCONST(...) mlgp_meanConst_sp(__VA_ARGS__)
#define MLGP_MEANLINEAR(...) mlgp_meanLinear_sp(__VA_ARGS__)
#endif

mlgpStatus_t MLGP_MEAN(
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t MLGP_MEANONE(
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t MLGP_MEANCONST(
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t MLGP_MEANLINEAR(
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
);

#endif /* MLGP_INTERNAL_MEAN_H */
