#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpFloat_t mlgp_covProd (
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff,
  covFuncNode_t *funcs
)
{

  mlgpFloat_t k = 1;
  while(funcs!=NULL){
    k *= (*(funcs->func))(x,incx,y,incy,dim,funcs->params,diff);
    funcs = funcs->next;
  }
  return k;

}

mlgpFloat_t mlgp_covProd_derivatives (
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

  return 0;

}
