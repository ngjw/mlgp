#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

/* covariance function: squared exponential with 
 * automatic relevance determination distance measure */

FLOAT MLGP_COVSEARD (
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
)
{

  MLGP_COPY(dim,y,incy,diff,1);
  MLGP_AXPY(dim,-1.0,x,incx,diff,1);
  for(int i=0;i<dim;i++){ diff[i]*=params[i]; }
  return params[dim]*exp(-MLGP_DOT(dim,diff,1,diff,1));

}

FLOAT MLGP_COVSEARD_DERIVATIVES (
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  unsigned param_i,
  FLOAT *diff
)
{

  MLGP_COPY(dim,y,incy,diff,1);
  MLGP_AXPY(dim,-1.0,x,incx,diff,1);
  FLOAT dist = diff[param_i];
  for(int i=0;i<dim;i++){ diff[i]*=params[i]; }
  FLOAT k = params[dim]*exp(-MLGP_DOT(dim,diff,1,diff,1));

  if(param_i<dim){
    return dist*dist*k*params[param_i]*params[param_i]*2;
  }else if(param_i == dim){
    return 2.0*k;
  }

}
