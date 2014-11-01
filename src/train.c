#include "include/mlgp_internal.h"
#include "../include/mlgp.h"
#include <lbfgs.h>

#ifdef DOUBLE
#define PROGRESS progress_dp
#define EVALUATE evaluate_dp
#define MLGP_TRAIN mlgp_train_dp
#else
#define PROGRESS progress_sp
#define EVALUATE evaluate_sp
#define MLGP_TRAIN mlgp_train_sp
#endif

typedef struct
{
  FLOAT* nll;
  MATRIX_T X;
  VECTOR_T y;
  INF_T inf;
  MEAN_T mean;
  COV_T cov;
  LIK_T lik;
  mlgpWorkspace_t* workspace;
  mlgpOptions_t options;
  unsigned np_cov;
  unsigned np_mean;
  unsigned np_lik;
  int verbose;
}
LIKPARAMS_T;

static int PROGRESS (
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
)
{

  LIKPARAMS_T* ep = (LIKPARAMS_T*)instance;

  if(ep->verbose){
    printf("Iteration %d:\n", k);
    printf("  f(x) = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm, step);
    printf("\n");
  }
  return 0;
}

lbfgsfloatval_t EVALUATE (
  void* instance,
  const lbfgsfloatval_t *x,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
)
{
  
  LIKPARAMS_T* ep = (LIKPARAMS_T*)instance;

  ep->mean.params = (FLOAT*)x;
  ep->cov.params  = (FLOAT*)x+ep->np_mean;
  ep->lik.params  = (FLOAT*)x+ep->np_mean+ep->np_cov;

  ep->mean.dparams = (FLOAT*)g;
  ep->cov.dparams  = (FLOAT*)g+ep->np_mean;
  ep->lik.dparams  = (FLOAT*)g+ep->np_mean+ep->np_cov;

  MLGP_LIKELIHOOD(
   ep->nll,
   ep->X,
   ep->y,
   ep->inf,
   ep->mean,
   ep->cov,
   ep->lik,
   ep->workspace,
   ep->options
  );

  return (lbfgsfloatval_t)(*(ep->nll));
}

mlgpStatus_t MLGP_TRAIN (
  FLOAT* final_nll,
  MATRIX_T X,
  VECTOR_T y,
  INF_T inf,
  MEAN_T mean,
  COV_T cov,
  LIK_T lik,
  mlgpWorkspace_t* workspace,
  mlgpTrainOpts_t trainopts,
  mlgpOptions_t options
)
{

  int status;
  unsigned nparams; 
  unsigned N, dim;
  unsigned sizeK, memoryNeeded;
  LIKPARAMS_T ep;

  N = X.nrows;
  dim = X.ncols;

  sizeK = (options.opts&PACKED) ? (N*(N+1))/2 : N*N;
  memoryNeeded = 3*sizeK + 2*N;

  /* Compute and assign working memory needed */
  if(options.opts&CREATEWORKSPACE || options.opts&NOWORKSPACE){
    
    if(options.opts&SAVE){
      workspace->ws = malloc(2*sizeof(FLOAT*));
      workspace->size = 2;
      workspace->allocated = 0x3u;
    }else{
      workspace->ws = malloc(sizeof(FLOAT*));
      workspace->size = 1;
      workspace->allocated = 0x1u;
    }
  
    workspace->ws[0] = malloc(memoryNeeded*sizeof(FLOAT));

    if(options.opts&SAVE){
      workspace->ws[1] = malloc((N*N+N)*sizeof(FLOAT));
    }

    if(options.opts&CREATEWORKSPACE){
      return mlgpSuccess;
    }
  }

  ep.verbose = (options.opts&VERBOSE) ? 1:0;
  ep.nll = final_nll;
  ep.X = X;
  ep.y = y;
  ep.inf = inf;
  ep.mean = mean;
  ep.cov = cov;
  ep.lik = lik;
  ep.workspace = workspace;
  ep.options = options;
  ep.np_cov = MLGP_NPARAMS_COV(cov,dim);
  ep.np_mean = MLGP_NPARAMS_MEAN(mean,dim);
  ep.np_lik = 1;

  nparams = ep.np_cov + ep.np_mean + ep.np_lik;

  lbfgsfloatval_t fx, *x;

  x = lbfgs_malloc(nparams);

  if(trainopts.use_defaults){
    lbfgs_parameter_init(&(trainopts.lbfgsparams));
  }

  ep.mean.params = (FLOAT*)x;
  ep.cov.params =  (FLOAT*)x+ep.np_mean;
  ep.lik.params =  (FLOAT*)x+ep.np_mean+ep.np_cov;

  ep.options.opts&=(~SAVE);
  ep.options.opts&=(~NOWORKSPACE|CREATEWORKSPACE);

  status = lbfgs(nparams, x, &fx, EVALUATE, PROGRESS, &ep, &(trainopts.lbfgsparams));

  MLGP_COPY(ep.np_mean,(FLOAT*)x,1,mean.params,1);
  MLGP_COPY(ep.np_cov,(FLOAT*)x+ep.np_mean,1,cov.params,1);
  MLGP_COPY(ep.np_lik,(FLOAT*)x+ep.np_mean+ep.np_cov,1,lik.params,1);

  *final_nll = fx;

  if(options.opts&SAVE){
    options.opts|=(NODERIVATIVES|SAVE);
    MLGP_LIKELIHOOD(final_nll,X,y,inf,mean,cov,lik,workspace,options);
  }


  /* Report the result. */
  if(options.opts&VERBOSE){
    printf("\nL-BFGS optimization terminated with status code = %d\n", status);
    printf("  negative log-likelihood = %f\n", fx);
  }

  lbfgs_free(x);

  if(options.opts&NOWORKSPACE){
    free(workspace->ws[0]);
    workspace->allocated&=(~0x1u);
  }
  
  return mlgpSuccess;

}
