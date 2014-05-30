#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

unsigned mlgp_nparams_mean (
  mlgpMean_t mean,
  unsigned dim
)
{

  /* returns the number of parameters each mean function has */

  unsigned nparams = 0;

  if(mean.mean_funcs&meanConst ){ nparams+=1; }
  if(mean.mean_funcs&meanLinear ){ nparams+=dim; }

  return nparams;
}
