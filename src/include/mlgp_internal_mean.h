#ifndef MLGP_INTERNAL_MEAN_H
#define MLGP_INTERNAL_MEAN_H

#include "../../include/mlgp.h"

/* mean related functions */

mlgpStatus_t mlgp_mean(
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_meanOne(
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_meanConst(
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_meanLinear(
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
);

#endif /* MLGP_INTERNAL_MEAN_H */
