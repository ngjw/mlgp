#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t MLGP_MEAN (
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T   mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

  if(mean.mean_funcs&meanOne){ MLGP_MEANONE(y,X,mean,dy,param_i,options); }
  if(mean.mean_funcs&meanConst){ MLGP_MEANCONST(y,X,mean,dy,param_i,options); }
  if(mean.mean_funcs&meanLinear){ MLGP_MEANLINEAR(y,X,mean,dy,param_i,options); }

  return mlgpSuccess;

}
