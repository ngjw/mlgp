#ifndef MLGP_H
#define MLGP_H

#ifdef HAVELBFGS
#include <lbfgs.h>
#endif

#include "types.h"
#include "options.h"
#include "mean.h"
#include "cov.h"
#include "lik.h"
#include "inf.h"

mlgpStatus_t mlgp_predict(
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

mlgpStatus_t mlgp_likelihood(
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

#ifdef HAVELBFGS
mlgpStatus_t mlgp_train(
  mlgpFloat_t* final_nll,
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpTrainOpts_t trainopts,
  mlgpOptions_t options
);
#endif /* HAVELBFGS */


mlgpMatrix_t mlgp_createMatrix(unsigned nrows, unsigned ncols);
mlgpVector_t mlgp_createVector(unsigned length);

mlgpMatrix_t mlgp_createMatrixNoMalloc(unsigned nrows, unsigned ncols);
mlgpVector_t mlgp_createVectorNoMalloc(unsigned length);

mlgpMatrix_t mlgp_readMatrix(unsigned nrows, unsigned ncols, const char *filename);
mlgpVector_t mlgp_readVector(unsigned length, const char *filename);

mlgpStatus_t mlgp_freeMatrix(mlgpMatrix_t m);
mlgpStatus_t mlgp_freeVector(mlgpVector_t v);
mlgpStatus_t mlgp_freeWorkspace(mlgpWorkspace_t ws);

mlgpMean_t mlgp_createMean(unsigned mean_funcs, unsigned dim);
mlgpCov_t  mlgp_createCov (unsigned cov_funcs,  unsigned dim);
mlgpInf_t  mlgp_createInf (unsigned inf_func);
mlgpLik_t  mlgp_createLik (unsigned lik_func);

mlgpStatus_t mlgp_freeMean(mlgpMean_t mean);
mlgpStatus_t mlgp_freeCov (mlgpCov_t  cov );
mlgpStatus_t mlgp_freeInf (mlgpInf_t  inf );
mlgpStatus_t mlgp_freeLik (mlgpLik_t  lik );

#endif /* MLGP_H */
