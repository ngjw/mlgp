#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

FLOAT MLGP_COVSUM (
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff,
  COVFUNCNODE_T *funcs
)
{

  FLOAT k = 0;
  while(funcs!=NULL){
    k += (*(funcs->func))(x,incx,y,incy,dim,funcs->params,diff);
    funcs = funcs->next;
  }
  return k;

}

FLOAT MLGP_COVSUM_DERIVATIVES (
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

  return 0;

}
