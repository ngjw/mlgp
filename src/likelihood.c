#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t MLGP_LIKELIHOOD (
  FLOAT* nll,
  MATRIX_T X,
  VECTOR_T y,
  INF_T inf,
  MEAN_T mean,
  COV_T cov,
  LIK_T lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
)
{

  return MLGP_LIKELIHOOD_CPU(
                           nll,
                           X,
                           y,
                           inf,
                           mean,
                           cov,
                           lik,
                           workspace,
                           options
                         );
  
}
