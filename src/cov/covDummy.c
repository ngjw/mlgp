#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

FLOAT MLGP_COVDUMMYZERO (
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
)
{

  return 0.0;

}

FLOAT MLGP_COVDUMMYONE (
  FLOAT *x,
  unsigned incx,
  FLOAT *y,
  unsigned incy,
  unsigned dim,
  FLOAT *params,
  FLOAT *diff
)
{

  return 1.0;

}
