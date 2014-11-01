#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t MLGP_PREDICT (
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
)
{
  return MLGP_PREDICT_CPU(
                   X,
                   y,
                   Xs,
                   ymu,
                   ys2,
                   fmu,
                   fs2,
                   inf,
                   mean,
                   cov,
                   lik,
                   workspace,
                   options
                 );
}
