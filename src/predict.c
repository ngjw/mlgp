#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t mlgp_predict (
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpMatrix_t Xs,
  mlgpVector_t ymu,
  mlgpVector_t ys2,
  mlgpVector_t fmu,
  mlgpVector_t fs2,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
)
{

  return mlgp_predict_cpu(
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
