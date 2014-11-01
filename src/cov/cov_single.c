#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t MLGP_COV_SINGLE (
  MATRIX_T K,
  MATRIX_T X1,
  MATRIX_T X2,
  COV_T cov,
  MATRIX_T dK,
  unsigned param_i,
  mlgpOptions_t options
)
{

  unsigned N1, N2, dim;
  unsigned sizeK;
  unsigned nparams;
  FLOAT *diff, *tempParams;

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
  COVFUNC_FT covFunc;
  COVFUNCDERIV_FT covFuncDeriv;
  
  // number of parameters
  nparams = MLGP_NPARAMS_COV(cov,dim);

  // cache log(params)
  tempParams = (FLOAT*)malloc(nparams*sizeof(FLOAT));
  MLGP_COPY(nparams,cov.params,1,tempParams,1);

  // parameter transformation
  MLGP_COV_PARAM_TRANS(cov,dim);

  diff = (FLOAT*)malloc(dim*sizeof(FLOAT));

  if(!(options.opts&_DERIVATIVES)){

    switch(cov.cov_funcs){
      case covSEiso:
        covFunc = &MLGP_COVSEISO; break;
      case covSEard:
        covFunc = &MLGP_COVSEARD; break;
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
        covFuncDeriv = &MLGP_COVSEISO_DERIVATIVES;
        break;
      case covSEard:
        covFuncDeriv = &MLGP_COVSEARD_DERIVATIVES;
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
  MLGP_COPY(nparams,tempParams,1,cov.params,1);

  return mlgpSuccess;

}
