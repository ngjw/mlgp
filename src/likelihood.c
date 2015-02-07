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

  /* returns the negative log marginal likelihood of a GP p(y|X) if the
   * NODERIVATIVES bit is not set in options.opts, derivatives of the log
   * marginal likelihood with respect to the mean/covariance/noise will be
   * computed and stored in mean.dparams/cov.dparams/lik.daparms
   * */

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
