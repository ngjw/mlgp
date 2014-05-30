#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

unsigned mlgp_nparams_cov (
  mlgpCov_t cov,
  unsigned dim
)
{
  /* returns the number of parameters each covariance function has */

  unsigned nparams = 0;

  if(cov.cov_funcs&covSEiso){ nparams+=NCP_SEiso; }
  if(cov.cov_funcs&covSEard){ nparams+=NCP_SEard; }

  return nparams;
}


void mlgp_cov_param_trans$ (
  mlgpCov_t cov, 
  unsigned dim
)
{
  /* transforms the cov parameters into form used by the cov functions */

  ptrdiff_t param_offset = 0;
  
  if(cov.cov_funcs&covSEiso){
    cov.params[param_offset+0] = 0.5*exp(-2.0*cov.params[param_offset+0]);
    cov.params[param_offset+1] = exp(2.0*cov.params[param_offset+1]);
    param_offset+=NCP_SEiso;
  }

  if(cov.cov_funcs&covSEard){
    for(int i=0;i<dim;i++){
      cov.params[param_offset+i] = SQRT_HALF*exp(-cov.params[param_offset+i]);
    }
    cov.params[param_offset+dim] = exp(2.0*cov.params[param_offset+dim]);
    param_offset+=NCP_SEard;
  }

}


/* creates the linked list of cov functions and their parameters */
covFuncNode_t *mlgp_createCovFuncList (
  mlgpCov_t cov,
  unsigned dim,
  unsigned covToSkip,
  unsigned zeroOrOne
)
{
  unsigned param_offset;
  covFuncNode_t *head = (covFuncNode_t*)malloc(sizeof(covFuncNode_t));
  covFuncNode_t *cur = head;

  if(cov.cov_funcs&covSum){
    param_offset = 0;
  }else if(cov.cov_funcs&covProd){
    param_offset = 0;
  }

  if(cov.cov_funcs&covSEiso){
    if(covToSkip&covSEiso){
      cur->func = (zeroOrOne==0) ? &mlgp_covDummyZero : &mlgp_covDummyOne;
    }else{
      cur->func = &mlgp_covSEiso;
    }
    cur->params = cov.params + param_offset;
    cur->next = NULL;
    param_offset += 2;
  }

  if(cov.cov_funcs&covSEard){
    cur->next = (covFuncNode_t*)malloc(sizeof(covFuncNode_t));
    cur = cur->next;

    if(covToSkip&covSEard){
      cur->func = (zeroOrOne==0) ? &mlgp_covDummyZero : &mlgp_covDummyOne;
    }else{
      cur->func = &mlgp_covSEard;
    }

    cur->params = cov.params + param_offset;
    cur->next = NULL;
    param_offset += dim+1;
  }

  return head;

}

void mlgp_freeCovFuncList (
  covFuncNode_t *list
)
{
  covFuncNode_t *temp = list;
  while(list!=NULL){
    temp = list->next; 
    free(list);
    list = temp;
  }

}
