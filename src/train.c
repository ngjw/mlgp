#include "include/mlgp_internal.h"
#include "../include/mlgp.h"
#include <lbfgs.h>

typedef struct
{
  mlgpFloat_t* nll;
  mlgpMatrix_t X;
  mlgpVector_t y;
  mlgpInf_t inf;
  mlgpMean_t mean;
  mlgpCov_t cov;
  mlgpLik_t lik;
  mlgpWorkspace_t* workspace;
  mlgpOptions_t options;
  unsigned np_cov;
  unsigned np_mean;
  unsigned np_lik;
  int verbose;
}
mlgp_likParams_t;

static int progress (
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

  mlgp_likParams_t* ep = (mlgp_likParams_t*)instance;

  if(ep->verbose){
    printf("Iteration %d:\n", k);
    printf("  f(x) = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm, step);
    printf("\n");
  }
  return 0;
}

lbfgsfloatval_t evaluate (
  void* instance,
  const lbfgsfloatval_t *x,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
)
{
  
  mlgp_likParams_t* ep = (mlgp_likParams_t*)instance;

  ep->mean.params = (mlgpFloat_t*)x;
  ep->cov.params  = (mlgpFloat_t*)x+ep->np_mean;
  ep->lik.params  = (mlgpFloat_t*)x+ep->np_mean+ep->np_cov;

  ep->mean.dparams = (mlgpFloat_t*)g;
  ep->cov.dparams  = (mlgpFloat_t*)g+ep->np_mean;
  ep->lik.dparams  = (mlgpFloat_t*)g+ep->np_mean+ep->np_cov;

  mlgp_likelihood(
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

mlgpStatus_t mlgp_train (
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
)
{

  int status;
  unsigned nparams; 
  unsigned N, dim;
  unsigned sizeK, memoryNeeded;
  mlgp_likParams_t ep;

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
  ep.np_cov = mlgp_nparams_cov(cov,dim);
  ep.np_mean = mlgp_nparams_mean(mean,dim);
  ep.np_lik = 1;

  nparams = ep.np_cov + ep.np_mean + ep.np_lik;

  lbfgsfloatval_t fx, *x;

  x = lbfgs_malloc(nparams);

  if(trainopts.use_defaults){
    lbfgs_parameter_init(&(trainopts.lbfgsparams));
  }

  ep.mean.params = x;
  ep.cov.params =  x+ep.np_mean;
  ep.lik.params =  x+ep.np_mean+ep.np_cov;

  ep.options.opts&=(~SAVE);
  ep.options.opts&=(~NOWORKSPACE|CREATEWORKSPACE);

  status = lbfgs(nparams, x, &fx, evaluate, progress, &ep, &(trainopts.lbfgsparams));

  CBLAS_COPY(ep.np_mean,x,1,mean.params,1);
  CBLAS_COPY(ep.np_cov,x+ep.np_mean,1,cov.params,1);
  CBLAS_COPY(ep.np_lik,x+ep.np_mean+ep.np_cov,1,lik.params,1);

  *final_nll = fx;

  if(options.opts&SAVE){
    options.opts|=(NODERIVATIVES|SAVE);
    mlgp_likelihood(final_nll,X,y,inf,mean,cov,lik,workspace,options);
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
