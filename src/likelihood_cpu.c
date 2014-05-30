#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t mlgp_likelihood_cpu (
  mlgpFloat_t* nll,
  mlgpMatrix_t X,
  mlgpVector_t y,
  mlgpInf_t inf,
  mlgpMean_t mean,
  mlgpCov_t cov,
  mlgpLik_t lik,
  mlgpWorkspace_t* workspace,
  mlgpOptions_t options
)
{

  /* Exact inference with Gaussian Likelihood */

  /* This function computes the negative log marginal likelihood (nll)
   * and its derivatives with respect to the mean, covariance and likelihood
   * function hyperparameters.
   * 
   * Arguments:
   * - nll          : Pointer to a mlgpFloat_t to store the negative log
   *                  marginal likelihood
   *
   * - X            : mlgpMatrix_t containing the matrix of training inputs.
   *                  Each row represents one input vector. Column major format.
   *
   * - y            : mlgpVector_t containing the vector of training targets.
   *
   * - inf          : mlgpInf_t specifying the inference method to be used
   *                  (currently only the exact inference method available).
   *
   * - mean         : mlgpMean_t specifying the mean function and its parameters
   *
   * - cov          : mlgpCov_t specifying the covaraince function and its
   *                  parameters
   *
   * - lik          : mlgpLik_t specifying the likelihood function and its
   *                  parameters (currently only the Gaussian likelihood
   *                  avaialble).
   *
   * - workspace    : mlgpWorkspace_t containing the pointer to pointer for
   *                  the working memory. If the NOWORKSPACE flag is set in
   *                  options.opts, then this function takes care of memory
   *                  allocation entirely. If the CREATEWORKSPACE flag is 
   *                  set, this function only allocates the required working
   *                  memory in workspace->ws[0] (based on the parameters passed
   *                  in) and returns without performing any computation.
   *
   *                  If the SAVE flags is set, then workspace->ws[1] will
   *                  contain a pointer to a N*N + N block of memory containing
   *                  the vector (K^-1)*(y-m) and the Cholesky factor of the 
   *                  covariance matrix K (to use for prediction).
   *                  Note for this workspace->ws will need to have at least two
   *                  pointers worth of memory allocated to it.
   *
   * - options      : mlgpOptions_t specifying the options for the computation.
   *                  Possible options are
   *                  - CREATEWORKSPACE (as described above)
   *                  - NOWORKSPACE (as described above)
   *                  - PACKED uses packed storage for symmetric matrices
   *                  - SAVE (as described above)
   */

  unsigned N, dim;
  unsigned sizeK, memoryNeeded;
  unsigned np_cov, np_mean;

  ptrdiff_t shift;

  mlgpMatrix_t K, dKdTheta1, dKdTheta2;
  mlgpVector_t ymm, Kinvy;

  mlgpFloat_t sig_n2;

  mlgpOptions_t int_opts;

  N = X.nrows;
  dim = X.ncols;

  sizeK = (options.opts&PACKED) ? (N*(N+1))/2 : N*N;

  memoryNeeded = 3*sizeK + 2*N;

  /* Compute and assign working memory needed */
  if(options.opts&CREATEWORKSPACE || options.opts&NOWORKSPACE){
    
    if(options.opts&SAVE){
      workspace->ws = (mlgpFloat_t**)malloc(2*sizeof(mlgpFloat_t*));
      workspace->size = 2;
      workspace->allocated = 0x3u;
    }else{
      workspace->ws = (mlgpFloat_t**)malloc(sizeof(mlgpFloat_t*));
      workspace->size = 1;
      workspace->allocated = 0x1u;
    }
  
    workspace->ws[0] = (mlgpFloat_t*)malloc(memoryNeeded*sizeof(mlgpFloat_t));

    if(options.opts&SAVE){
      workspace->ws[1] = (mlgpFloat_t*)malloc((N*N+N)*sizeof(mlgpFloat_t));
    }

    if(options.opts&CREATEWORKSPACE){
      return mlgpSuccess;
    }
  }

  K         = mlgp_createMatrixNoMalloc(N,N);
  dKdTheta1 = mlgp_createMatrixNoMalloc(N,N);
  dKdTheta2 = mlgp_createMatrixNoMalloc(N,N);

  ymm       = mlgp_createVectorNoMalloc(N);
  Kinvy     = mlgp_createVectorNoMalloc(N);


  shift = 0;
  K.m         = workspace->ws[0]+shift; shift+=sizeK; // N*N - covariance matrix
  dKdTheta1.m = workspace->ws[0]+shift; shift+=sizeK; // N*N - covariance derivatives
  dKdTheta2.m = workspace->ws[0]+shift; shift+=sizeK; // N*N - covariance derivatives
  ymm.v       = workspace->ws[0]+shift; shift+=N;     // N   - y-m
  Kinvy.v     = workspace->ws[0]+shift; shift+=N;     // N   - (K^-1)*y
  


  /* ymm contains y */
  CBLAS_COPY(N,y.v,1,ymm.v,1);

  /* apply mean function, ymm contains y-m */
  int_opts.opts = _SUBMEAN;
  mlgp_mean(ymm,X,mean,ymm,0,int_opts);

  /* copy y-m into Kinvy */
  CBLAS_COPY(N,ymm.v,1,Kinvy.v,1);

  /* generate the covariance matrix in K */
  int_opts.opts = _SYMM;
  if(options.opts&PACKED){ int_opts.opts|=_PACKED; }
  mlgp_cov(K,X,X,cov,K,0,int_opts);

  /* cache K in dKdTheta2 for derivatives computation */
  if(options.opts&PACKED){
    CBLAS_COPY((N*(N+1))/2,K.m,1,dKdTheta2.m,1);
  }else{
    CBLAS_COPY(N*N,K.m,1,dKdTheta2.m,1);
  }

  /* add noise term to the diagonal */
  sig_n2 = exp(2.*lik.params[0]);
  if(options.opts&PACKED){
    for(unsigned i=0;i<N;i++){
      K.m[i+(i*(i+1))/2] += sig_n2;
    }
  }else{
    CBLAS_AXPY(N,1.,&sig_n2,0,K.m,N+1);
  }

  /* Cholesky decomposition K = LL^T */
  /* solve K(Kinvy) = y for Kinvy using L */
  if(options.opts&PACKED){
    chol_packed(K);
    solve_chol_packed_one(K,Kinvy);
  }else{
    chol(K);
    solve_chol_one(K,Kinvy);
  }

  /* first term in nll = 0.5*y'*(K^-1)*y */
  *nll = 0.5*CBLAS_DOT(N,ymm.v,1,Kinvy.v,1);


  /* second term in nll = 0.5*log(det(K)) 
   * = 0.5*log(det(LL')) = 0.5*log(det(L)^2) = log(det(L))*/
  if(options.opts&PACKED){
    *nll += log_det_tr_packed(K);
  }else{
    *nll += log_det_tr(K);
  }

  /* third term in nll = (N/2)*log(2*pi)  */ 
  *nll += N*HALFLOG2PI;

  if(options.opts&SAVE){
    CBLAS_COPY(N,Kinvy.v,1,workspace->ws[1],1);
    CBLAS_COPY(N*N,K.m,1,workspace->ws[1]+N,1);
  }

  /* derivatives */
  if(!(options.opts&NODERIVATIVES)){

    /* precompute Q = (K^-1) - Kinvy'*Kinvy (stored in K) */
    if(options.opts&PACKED){
      inv_chol_packed(K);
      CBLAS_SPR(CblasColMajor,CblasUpper,N,-1.,Kinvy.v,1,K.m);
    }else{
      inv_chol(K);
      CBLAS_GEMM(CblasColMajor,CblasNoTrans,
      CblasNoTrans,N,N,1,-1.,Kinvy.v,N,Kinvy.v,1,1.,K.m,N);
    }

    /* get number of covariance function parameters */
    np_cov = mlgp_nparams_cov(cov,dim);

    /* covariance function options flag for derivatives computation */
    int_opts.opts = _SYMM|_DERIVATIVES|_PRECOMPUTE;
    if(options.opts&PACKED){ int_opts.opts|=_PACKED; }

    /* iterate through covariance parameters and compute derivatives for each */
    for(unsigned p_i=0;p_i<np_cov;p_i++){

      mlgp_cov(dKdTheta2,X,X,cov,dKdTheta1,p_i,int_opts);

      if(options.opts&PACKED){
        // elementwise product of Q and dKdTheta1 = trace(Q*dKdTheta1)
        cov.dparams[p_i] = CBLAS_DOT((N*(N+1))/2,dKdTheta1.m,1,K.m,1); 

        // subtract duplicates from using the diagonal twice
        for(unsigned i=0;i<N;i++){
          cov.dparams[p_i] -= K.m[i+(i*(i+1))/2]*dKdTheta1.m[i+(i*(i+1))/2]/2;
        }
      }else{
        // elementwise product of Q and dKdTheta1 = trace(Q*dKdTheta1)
        cov.dparams[p_i] = CBLAS_DOT(N*N,dKdTheta1.m,1,K.m,1)/2;
      }
    }

    /* get number of mean parameters */
    np_mean = mlgp_nparams_mean(mean,dim);

    /* iterate through mean parameters and compute derivatives for each */
    /* using ymm as memory to hold mean derivatives */
    int_opts.opts = _DERIVATIVES;
    for(unsigned p_i=0;p_i<np_mean;p_i++){
      mlgp_mean(ymm,X,mean,ymm,p_i,int_opts);
      mean.dparams[p_i] = -CBLAS_DOT(N,Kinvy.v,1,ymm.v,1);
    }

    /* derivative wrt gaussian noise parameter = sig_n*tr(Q) */
    if(options.opts&PACKED){
      lik.dparams[0] = 0;
      for(unsigned i=0;i<N;i++){
        lik.dparams[0] += K.m[i+(i*(i+1))/2]*sig_n2;
      }
    }else{
      lik.dparams[0] = CBLAS_DOT(N,K.m,N+1,&sig_n2,0);
    }

    if(options.opts&NOWORKSPACE){
      free(workspace->ws[0]);
      workspace->allocated&=(~0x1u);
    }
  }

  return mlgpSuccess;
  
}
