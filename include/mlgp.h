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

mlgpStatus_t mlgp_predict_sp (
  mlgpSMatrix_t X,
  mlgpSVector_t y,
  mlgpSMatrix_t Xs,
  mlgpSVector_t ymu,
  mlgpSVector_t ys2,
  mlgpSVector_t fmu,
  mlgpSVector_t fs2,
  mlgpSInf_t inf,
  mlgpSMean_t mean,
  mlgpSCov_t cov,
  mlgpSLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_predict_dp (
  mlgpDMatrix_t X,
  mlgpDVector_t y,
  mlgpDMatrix_t Xs,
  mlgpDVector_t ymu,
  mlgpDVector_t ys2,
  mlgpDVector_t fmu,
  mlgpDVector_t fs2,
  mlgpDInf_t inf,
  mlgpDMean_t mean,
  mlgpDCov_t cov,
  mlgpDLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_likelihood_sp (
  float* nll,
  mlgpSMatrix_t X,
  mlgpSVector_t y,
  mlgpSInf_t inf,
  mlgpSMean_t mean,
  mlgpSCov_t cov,
  mlgpSLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_likelihood_dp (
  double* nll,
  mlgpDMatrix_t X,
  mlgpDVector_t y,
  mlgpDInf_t inf,
  mlgpDMean_t mean,
  mlgpDCov_t cov,
  mlgpDLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
);

#ifdef HAVELBFGS
mlgpStatus_t mlgp_train_sp(
  float* final_nll,
  mlgpSMatrix_t X,
  mlgpSVector_t y,
  mlgpSInf_t inf,
  mlgpSMean_t mean,
  mlgpSCov_t cov,
  mlgpSLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpTrainOpts_t trainopts,
  mlgpOptions_t options
);

mlgpStatus_t mlgp_train_dp(
  double* final_nll,
  mlgpDMatrix_t X,
  mlgpDVector_t y,
  mlgpDInf_t inf,
  mlgpDMean_t mean,
  mlgpDCov_t cov,
  mlgpDLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpTrainOpts_t trainopts,
  mlgpOptions_t options
);
#endif /* HAVELBFGS */


mlgpSVector_t mlgp_createVector_sp(unsigned length);
mlgpDVector_t mlgp_createVector_dp(unsigned length);

mlgpSMatrix_t mlgp_createMatrix_sp(unsigned nrows, unsigned ncols);
mlgpDMatrix_t mlgp_createMatrix_dp(unsigned nrows, unsigned ncols);

mlgpSMatrix_t mlgp_createMatrixNoMalloc_sp(unsigned nrows, unsigned ncols);
mlgpDMatrix_t mlgp_createMatrixNoMalloc_dp(unsigned nrows, unsigned ncols);

mlgpSVector_t mlgp_createVectorNoMalloc_sp(unsigned length);
mlgpDVector_t mlgp_createVectorNoMalloc_dp(unsigned length);

mlgpSMatrix_t mlgp_readMatrix_sp(unsigned nrows, unsigned ncols, const char *filename);
mlgpDMatrix_t mlgp_readMatrix_dp(unsigned nrows, unsigned ncols, const char *filename);

mlgpSVector_t mlgp_readVector_sp(unsigned length, const char *filename);
mlgpDVector_t mlgp_readVector_dp(unsigned length, const char *filename);

mlgpStatus_t mlgp_freeVector_sp(mlgpSVector_t v);
mlgpStatus_t mlgp_freeVector_dp(mlgpDVector_t v);

mlgpStatus_t mlgp_freeMatrix_sp(mlgpSMatrix_t m);
mlgpStatus_t mlgp_freeMatrix_dp(mlgpDMatrix_t m);

mlgpStatus_t mlgp_freeWorkspace(mlgpWorkspace_t ws);

mlgpSMean_t mlgp_createMean_sp(unsigned mean_funcs, unsigned dim);
mlgpDMean_t mlgp_createMean_dp(unsigned mean_funcs, unsigned dim);

mlgpSCov_t  mlgp_createCov_sp(unsigned cov_funcs,  unsigned dim);
mlgpDCov_t  mlgp_createCov_dp(unsigned cov_funcs,  unsigned dim);

mlgpSInf_t  mlgp_createInf_sp(unsigned inf_func);
mlgpDInf_t  mlgp_createInf_dp(unsigned inf_func);

mlgpSLik_t  mlgp_createLik_sp(unsigned lik_func);
mlgpDLik_t  mlgp_createLik_dp(unsigned lik_func);

mlgpStatus_t mlgp_freeMean_sp(mlgpSMean_t mean);
mlgpStatus_t mlgp_freeMean_dp(mlgpDMean_t mean);

mlgpStatus_t mlgp_freeCov_sp (mlgpSCov_t  cov );
mlgpStatus_t mlgp_freeCov_dp (mlgpDCov_t  cov );

mlgpStatus_t mlgp_freeInf_sp(mlgpSInf_t  inf );
mlgpStatus_t mlgp_freeInf_dp(mlgpDInf_t  inf );

mlgpStatus_t mlgp_freeLik_sp(mlgpSLik_t  lik );
mlgpStatus_t mlgp_freeLik_dp(mlgpDLik_t  lik );

#endif /* MLGP_H */
