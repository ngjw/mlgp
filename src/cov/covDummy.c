#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpFloat_t mlgp_covDummyZero (
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
)
{

  return 0.0;

}

mlgpFloat_t mlgp_covDummyOne (
  mlgpFloat_t *x,
  unsigned incx,
  mlgpFloat_t *y,
  unsigned incy,
  unsigned dim,
  mlgpFloat_t *params,
  mlgpFloat_t *diff
)
{

  return 1.0;

}
