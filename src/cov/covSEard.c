#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

/* covariance function: squared exponential with 
 * automatic relevance determination distance measure */

mlgpFloat_t mlgp_covSEard (
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
)
{

  CBLAS_COPY(dim,y,incy,diff,1);
  CBLAS_AXPY(dim,-1.0,x,incx,diff,1);
  for(int i=0;i<dim;i++){ diff[i]*=params[i]; }
  return params[dim]*exp(-CBLAS_DOT(dim,diff,1,diff,1));

}

mlgpFloat_t mlgp_covSEard_derivatives (
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  unsigned param_i,
  mlgpFloat_t *diff
)
{

  CBLAS_COPY(dim,y,incy,diff,1);
  CBLAS_AXPY(dim,-1.0,x,incx,diff,1);
  mlgpFloat_t dist = diff[param_i];
  for(int i=0;i<dim;i++){ diff[i]*=params[i]; }
  mlgpFloat_t k = params[dim]*exp(-CBLAS_DOT(dim,diff,1,diff,1));

  if(param_i<dim){
    return dist*dist*k*params[param_i]*params[param_i]*2;
  }else if(param_i == dim){
    return 2.0*k;
  }

}
