#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t mlgp_predict_cpu (
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
)
{
  /* This function computes the predictive mean and variances of prediction 
   * inputs Xs given training inputs X and outputs y.
   *
   * Arguments:
   * - X            : mlgpMatrix_t containing the matrix of training inputs.
   *                  Each row represents one input vector. Column major format.
   *
   * - y            : mlgpVector_t containing the vector of training targets.
   *
   * - Xs           : mlgpMatrix_t containing the matrix of prediction inputs.
   *                  Each row represents one input vector. Column major format.
   *
   * - ymu          : mlgpVector_t containing the computed predictive means
   * 
   * - ys2          : mlgpVector_t containing the computed predictive variances
   *
   * - fmu          : mlgpVector_t containing the computed predictive latent
   *                  means (currently not implemented)
   *
   * - fs2          : mlgpVector_t containing the computed predictive latent
   *                  variances (currently not implemented)
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
   *                  covariance matrix K (precomputed from the likelihood
   *                  function).
   *
   * - options      : mlgpOptions_t specifying the options for the computation.
   *                  Possible options are
   *                  - CREATEWORKSPACE (as described above)
   *                  - NOWORKSPACE (as described above)
   *                  - PACKED uses packed storage for symmetric matrices
   *                  - SAVE (as described above)
   */


  unsigned N, Ns;
  unsigned sizeK, memoryNeeded;
  ptrdiff_t shift;

  mlgpMatrix_t K, Ks, Ks_temp, temp_mat;
  mlgpVector_t Kinvy;
  mlgpOptions_t int_opts;

  mlgpFloat_t sig_n2;

  N = X.nrows;
  Ns = Xs.nrows;

  sizeK = (options.opts&PACKED) ? (N*(N+1))/2 : N*N;

  memoryNeeded = sizeK + 2*N*Ns + N;

  K       = mlgp_createMatrixNoMalloc(N,N);
  Ks      = mlgp_createMatrixNoMalloc(Ns,N);
  Ks_temp = mlgp_createMatrixNoMalloc(N,Ns);
  Kinvy   = mlgp_createVectorNoMalloc(N);


  /* Compute and assign working memory needed */
  if(options.opts&CREATEWORKSPACE || options.opts&NOWORKSPACE){
    
    if(options.opts&SAVE){
      if(workspace->allocated&0x1u){
        free(workspace->ws[0]);
      }
      memoryNeeded = 2*N*Ns;
    }

    workspace->ws[0] = (mlgpFloat_t*)malloc(memoryNeeded*sizeof(mlgpFloat_t));

    if(options.opts&CREATEWORKSPACE){
      return mlgpSuccess;
    }
  }
  
  shift = 0;
  Ks.m       = workspace->ws[0]+shift; shift+=Ns*N;
  Ks_temp.m  = workspace->ws[0]+shift; shift+=Ns*N;

  if(options.opts&SAVE){
    Kinvy.v    = workspace->ws[1];
    K.m        = workspace->ws[1]+N;
  }else{
    Kinvy.v    = workspace->ws[0]+shift; shift+=N;
    K.m        = workspace->ws[0]+shift; shift+=sizeK;
  }

  if(!(options.opts&SAVE)){
    /* copy training inputs y into Kinvy */
    CBLAS_COPY(N,y.v,1,Kinvy.v,1);

    /* subtract prior mean (in Ks) from y (in Kinvy) 
     * Kinvy contains (y-mu) */
    int_opts.opts = _SUBMEAN;
    mlgp_mean(Kinvy,X,mean,Kinvy,0,int_opts);
  }

  sig_n2 = exp(2.*lik.params[0]);

  /* generate training data covariance matrix  in K */
  if(!(options.opts&SAVE)){
    int_opts.opts = _SYMM;
    if(options.opts&PACKED){ int_opts.opts|=_PACKED; }
    mlgp_cov(K,X,X,cov,K,0,int_opts);

  /* add Gaussian noise term to the diagonal 
     * K contains K + diag(sig_n^2) */
    if(options.opts&PACKED){
      for(unsigned i=0;i<N;i++){
        K.m[i+(i*(i+1))/2] += sig_n2;
      }
    }else{
      CBLAS_AXPY(N,1.,&sig_n2,0,K.m,N+1);
    }

    /* solve K(Kinvy) = (y-mu) 
     * Kinvy contains (K^-1)*(y-mu) 
     * K contains L (its cholesky factor) */
    if(options.opts&PACKED){
      chol_packed(K);
      solve_chol_packed_one(K,Kinvy);
    }else{
      chol(K);
      solve_chol_one(K,Kinvy);
    }

  }

  /* generate cross covariance matrix in Ks */
  int_opts.opts = _NONE;
  mlgp_cov(Ks,Xs,X,cov,Ks,0,int_opts);

  /* ymu = (Ks*Kinv*(y-mu)) + mus */
  CBLAS_GEMV(CblasColMajor,CblasNoTrans,Ns,N,1.,Ks.m,Ns,Kinvy.v,1,0.,ymu.v,1);
  int_opts.opts = _ADDMEAN;
  mlgp_mean(ymu,Xs,mean,ymu,0,int_opts);

  /* prediction variance */
  for(unsigned i=0;i<N;i++){
  for(unsigned j=0;j<Ns;j++){
    /* copy transpose of Ks to Ks_temp */
    Ks_temp.m[i+j*N] = Ks.m[j+i*Ns];
  }}


  if(options.opts&PACKED){
    solve_chol_packed_multiple(K,Ks_temp);
  }else{
    solve_chol_multiple(K,Ks_temp);
  }

  int_opts.opts = _SELF;
  temp_mat.m = ys2.v;

  mlgp_cov(temp_mat,Xs,Xs,cov,temp_mat,0,int_opts);

  for(unsigned i=0;i<Ns;i++){
    ys2.v[i] -= CBLAS_DOT(N,Ks.m+i,Ns,Ks_temp.m+i*N,1);
    ys2.v[i] += sig_n2;
  }


  if(options.opts&NOWORKSPACE){
    free(workspace->ws[0]);
    workspace->allocated&=(~0x1u);
  }

  return mlgpSuccess;
}
