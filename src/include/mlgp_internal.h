#ifndef MLGP_INTERNAL_H
#define MLGP_INTERNAL_H

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/mlgp.h"

#include "mlgp_internal_precision.h"
#include "mlgp_internal_aux.h"
#include "mlgp_internal_cov.h"
#include "mlgp_internal_mean.h"
#include "mlgp_internal_consts.h"
#include "mlgp_internal_options.h"

mlgpStatus_t mlgp_predict_cpu(
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpMatrix_t Xs,
  mlgpVector_t ymu,
  mlgpVector_t ys2,
  mlgpVector_t fmu,
  mlgpVector_t fs2,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_likelihood_cpu(
  mlgpFloat_t* nll,
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);


#endif /* MLGP_INTERNAL_H */
