#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t mlgp_likelihood (
  mlgpFloat_t* nll,
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
)
{

  return mlgp_likelihood_cpu(
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
