#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t mlgp_mean (
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

  if(mean.mean_funcs&meanOne){ mlgp_meanOne(y,X,mean,dy,param_i,options); }
  if(mean.mean_funcs&meanConst){ mlgp_meanConst(y,X,mean,dy,param_i,options); }
  if(mean.mean_funcs&meanLinear){ mlgp_meanLinear(y,X,mean,dy,param_i,options); }

  return mlgpSuccess;

}
