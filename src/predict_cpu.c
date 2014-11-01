#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

mlgpStatus_t MLGP_PREDICT_CPU (
  MATRIX_T X,
  VECTOR_T y,
  MATRIX_T Xs,
  VECTOR_T ymu,
  VECTOR_T ys2,
  VECTOR_T fmu,
  VECTOR_T fs2,
  INF_T inf,
  MEAN_T mean,
  COV_T cov,
  LIK_T lik,
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

  MATRIX_T K, Ks, Ks_temp, temp_mat;
  VECTOR_T Kinvy;
  mlgpOptions_t int_opts;

  FLOAT sig_n2;

  N = X.nrows;
  Ns = Xs.nrows;

  sizeK = (options.opts&PACKED) ? (N*(N+1))/2 : N*N;

  memoryNeeded = sizeK + 2*N*Ns + N;

  K       = MLGP_CREATEMATRIXNOMALLOC(N,N);
  Ks      = MLGP_CREATEMATRIXNOMALLOC(Ns,N);
  Ks_temp = MLGP_CREATEMATRIXNOMALLOC(N,Ns);
  Kinvy   = MLGP_CREATEVECTORNOMALLOC(N);


  /* Compute and assign working memory needed */
  if(options.opts&CREATEWORKSPACE || options.opts&NOWORKSPACE){
    
    if(options.opts&SAVE){
      if(workspace->allocated&0x1u){
        free(workspace->ws[0]);
      }
      memoryNeeded = 2*N*Ns;
    }

    workspace->ws[0] = malloc(memoryNeeded*sizeof(FLOAT));

    if(options.opts&CREATEWORKSPACE){
      return mlgpSuccess;
    }
  }
  
  shift = 0;
  Ks.m       = (FLOAT*)workspace->ws[0]+shift; shift+=Ns*N;
  Ks_temp.m  = (FLOAT*)workspace->ws[0]+shift; shift+=Ns*N;

  if(options.opts&SAVE){
    Kinvy.v    = (FLOAT*)workspace->ws[1];
    K.m        = (FLOAT*)workspace->ws[1]+N;
  }else{
    Kinvy.v    = (FLOAT*)workspace->ws[0]+shift; shift+=N;
    K.m        = (FLOAT*)workspace->ws[0]+shift; shift+=sizeK;
  }

  if(!(options.opts&SAVE)){
    /* copy training inputs y into Kinvy */
    MLGP_COPY(N,y.v,1,Kinvy.v,1);

    /* subtract prior mean (in Ks) from y (in Kinvy) 
     * Kinvy contains (y-mu) */
    int_opts.opts = _SUBMEAN;
    MLGP_MEAN(Kinvy,X,mean,Kinvy,0,int_opts);
  }

  sig_n2 = exp(2.*lik.params[0]);

  /* generate training data covariance matrix in K */
  if(!(options.opts&SAVE)){
    int_opts.opts = _SYMM;
    if(options.opts&PACKED){ int_opts.opts|=_PACKED; }
    MLGP_COV(K,X,X,cov,K,0,int_opts);

  /* add noise term to the diagonal 
     * K contains K + diag(sig_n^2) */
    if(options.opts&PACKED){
      for(unsigned i=0;i<N;i++){
        K.m[i+(i*(i+1))/2] += sig_n2;
      }
    }else{
      MLGP_AXPY(N,1.,&sig_n2,0,K.m,N+1);
    }

    /* solve K(Kinvy) = (y-mu) 
     * Kinvy contains (K^-1)*(y-mu) 
     * K contains L (its cholesky factor) */
    if(options.opts&PACKED){
      CHOL_PACKED(K);
      SOLVE_CHOL_PACKED_ONE(K,Kinvy);
    }else{
      CHOL(K);
      SOLVE_CHOL_ONE(K,Kinvy);
    }

  }

  /* generate cross covariance matrix in Ks */
  int_opts.opts = _NONE;
  MLGP_COV(Ks,Xs,X,cov,Ks,0,int_opts);

  /* ymu = (Ks*Kinv*(y-mu)) + mus */
  MLGP_GEMV('N',Ns,N,1.,Ks.m,Ns,Kinvy.v,1,0.,ymu.v,1);
  int_opts.opts = _ADDMEAN;
  MLGP_MEAN(ymu,Xs,mean,ymu,0,int_opts);

  /* prediction variance */
  for(unsigned i=0;i<N;i++){
  for(unsigned j=0;j<Ns;j++){
    /* copy transpose of Ks to Ks_temp */
    Ks_temp.m[i+j*N] = Ks.m[j+i*Ns];
  }}


  if(options.opts&PACKED){
    SOLVE_CHOL_PACKED_MULTIPLE(K,Ks_temp);
  }else{
    SOLVE_CHOL_MULTIPLE(K,Ks_temp);
  }

  int_opts.opts = _SELF;
  temp_mat.m = ys2.v;

  MLGP_COV(temp_mat,Xs,Xs,cov,temp_mat,0,int_opts);

  for(unsigned i=0;i<Ns;i++){
    ys2.v[i] -= MLGP_DOT(N,Ks.m+i,Ns,Ks_temp.m+i*N,1);
    ys2.v[i] += sig_n2;
  }


  if(options.opts&NOWORKSPACE){
    free(workspace->ws[0]);
    workspace->allocated&=(~0x1u);
  }

  return mlgpSuccess;
}
