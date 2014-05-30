#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

/* covariance function: squared exponential with isotropic distance measure */

mlgpFloat_t mlgp_covSEiso (
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
  CBLAS_AXPY(dim,-1.,x,incx,diff,1);
  return params[1]*exp(-params[0]*CBLAS_DOT(dim,diff,1,diff,1));
}

mlgpFloat_t mlgp_covSEiso_derivatives (
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
  mlgpFloat_t sqdist;

  CBLAS_COPY(dim,y,incy,diff,1);
  CBLAS_AXPY(dim,-1.,x,incx,diff,1); // diff
  sqdist = CBLAS_DOT(dim,diff,1,diff,1); // dist

  if(param_i == 0){
    return 2.0*params[0]*sqdist*params[1]*exp(-params[0]*sqdist);
  }else if(param_i == 1){
    return 2.0*params[1]*exp(-params[0]*sqdist);
  }

}
