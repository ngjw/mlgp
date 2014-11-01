#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

/* covariance function: squared exponential with isotropic distance measure */

FLOAT MLGP_COVSEISO (
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
  MLGP_AXPY(dim,-1.,x,incx,diff,1);
  return params[1]*exp(-params[0]*MLGP_DOT(dim,diff,1,diff,1));
}

FLOAT MLGP_COVSEISO_DERIVATIVES (
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
  FLOAT sqdist;

  MLGP_COPY(dim,y,incy,diff,1);
  MLGP_AXPY(dim,-1.,x,incx,diff,1); // diff
  sqdist = MLGP_DOT(dim,diff,1,diff,1); // dist

  if(param_i == 0){
    return 2.0*params[0]*sqdist*params[1]*exp(-params[0]*sqdist);
  }else if(param_i == 1){
    return 2.0*params[1]*exp(-params[0]*sqdist);
  }

}
