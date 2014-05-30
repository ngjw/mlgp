#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t mlgp_cov_single (
  mlgpMatrix_t K,
  mlgpMatrix_t X1,
  mlgpMatrix_t X2,
  mlgpCov_t cov,
  mlgpMatrix_t dK,
  unsigned param_i,
  mlgpOptions_t options
)
{

  unsigned N1, N2, dim;
  unsigned sizeK;
  unsigned nparams;
  mlgpFloat_t *diff, *tempParams;

  N1 = X1.nrows;
  N2 = X2.nrows;
  dim = X1.ncols;

  if(options.opts&_PACKED){
    sizeK = (N1*(N1+1))/2;
  }else if(options.opts&_SELF){
    sizeK = N1;
  }else if(options.opts&_SYMM){
    sizeK = N1*N1;
  }else{
    sizeK = N1*N2;
  }

  // ptr to covariance function
  covFunc_ft covFunc;
  covFuncDeriv_ft covFuncDeriv;
  
  // number of parameters
  nparams = mlgp_nparams_cov(cov,dim);

  // cache log(params)
  tempParams = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));
  CBLAS_COPY(nparams,cov.params,1,tempParams,1);

  // parameter transformation
  mlgp_cov_param_trans(cov,dim);

  diff = (mlgpFloat_t*)malloc(dim*sizeof(mlgpFloat_t));

  if(!(options.opts&_DERIVATIVES)){

    switch(cov.cov_funcs){
      case covSEiso:
        covFunc = &mlgp_covSEiso; break;
      case covSEard:
        covFunc = &mlgp_covSEard; break;
      default:
        return mlgpError;
    }
  
    // fill in values of covariance matrix   
    if(options.opts&_SYMM){
      if(options.opts&_PACKED){
        // packed storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          K.m[i+(j*(j+1))/2]  = (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params,diff);
        }}
      }else{
        // full storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          K.m[i+N1*j]  = (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params,diff);
          if(i!=j){ K.m[j+N1*i] = K.m[i+N1*j]; }
        }}
      }
    }else if(options.opts&_SELF){
      // compute covariance of each vector in X with itself only
      for(int i=0;i<N1;i++){
        K.m[i]  = (*covFunc)(X1.m+i,N1,X1.m+i,N1,dim,cov.params,diff);
      }
    }else{
      // non symmetric covariance matrix (cross covariance for prediction)
      for(int i=0;i<N1;i++){
      for(int j=0;j<N2;j++){
        K.m[i+N1*j]  = (*covFunc)(X1.m+i,N1,X2.m+j,N2,dim,cov.params,diff);
      }}
    }

  }else{

    switch(cov.cov_funcs){
      case covSEiso:
        covFuncDeriv = &mlgp_covSEiso_derivatives;
        break;
      case covSEard:
        covFuncDeriv = &mlgp_covSEard_derivatives;
        break;
    }
    
    if(options.opts&_PACKED){
      // packed storage
      for(int i=0;i<N1;i++){
      for(int j=i;j<N1;j++){
        dK.m[i+(j*(j+1))/2]  = (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params,param_i,diff);
      }}
    }else{
      // full storage
      for(int i=0;i<N1;i++){
      for(int j=i;j<N1;j++){
        dK.m[i+N1*j]  = (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params,param_i,diff);
        if(i!=j){ dK.m[j+N1*i] = dK.m[i+N1*j]; }
      }}
    }

  }

  free(diff);

  // restore log(parameters)
  CBLAS_COPY(nparams,tempParams,1,cov.params,1);

  return mlgpSuccess;

}
