#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t mlgp_cov_composite (
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
  ptrdiff_t param_offset;
  mlgpFloat_t *diff; 
  covFuncComp_ft covFunc;
  mlgpFloat_t *tempParams;

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

  nparams = mlgp_nparams_cov(cov,dim);

  tempParams = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));

  diff = (mlgpFloat_t*)malloc(dim*sizeof(mlgpFloat_t));

  // parameter transformation
  CBLAS_COPY(nparams,cov.params,1,tempParams,1);
  mlgp_cov_param_trans(cov,dim);

  covFuncNode_t *covFuncs = mlgp_createCovFuncList(cov,dim,0,0);

  if(cov.cov_funcs&covSum){
    covFunc = &mlgp_covSum;
  }else if(cov.cov_funcs&covProd){
    covFunc = &mlgp_covProd;
  }

  if(!(options.opts&_DERIVATIVES)){
  
   if(options.opts&_SYMM){
     if(options.opts&_PACKED){
       // packed storage
       for(int i=0;i<N1;i++){
       for(int j=i;j<N1;j++){
         K.m[i+(j*(j+1))/2] = (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params+param_offset,diff,covFuncs);
       }}
     }else{
       // full storage
       for(int i=0;i<N1;i++){
       for(int j=i;j<N1;j++){
         K.m[i+N1*j] = (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params+param_offset,diff,covFuncs);
         if(i!=j){ K.m[j+N1*i] = K.m[i+N1*j]; }
       }}
     }
   }else if(options.opts&_SELF){
     // compute covariance of each vector in X with itself only
     for(int i=0;i<N1;i++){
       K.m[i] = (*covFunc)(X1.m+i,N1,X1.m+i,N1,dim,cov.params+param_offset,diff,covFuncs);
     }
   }else{
     // non symmetric covariance matrix (cross covariance for prediction)
     for(int i=0;i<N1;i++){
     for(int j=0;j<N2;j++){
       K.m[i+N1*j] = (*covFunc)(X1.m+i,N1,X2.m+j,N2,dim,cov.params+param_offset,diff,covFuncs);
     }}
   }

  }else{

    // DERIVATIVES

    // identify which cov function the chosen parameter belongs to
    // TODO: THIS PART ONLY WORKS FOR covSum NOW

    covFuncDeriv_ft covFuncDeriv;

    unsigned count = 0;
    unsigned curCovFunc;
    param_offset = 0;

    if(cov.cov_funcs&covSEiso && param_i >= count){
      count+=NCP_SEiso;
      curCovFunc = covSEiso;
    }

    if(cov.cov_funcs&covSEard && param_i >= count){
      count+=NCP_SEard;
      curCovFunc = covSEard;
    }

    switch(curCovFunc){
      case covSEiso:
        param_offset = count - NCP_SEiso;
        covFuncDeriv = &mlgp_covSEiso_derivatives;
        break;
      case covSEard:
        param_offset = count - NCP_SEard;
        covFuncDeriv = &mlgp_covSEard_derivatives;
        break;
    }

    if(cov.cov_funcs&covSum){
      if(options.opts&_PACKED){
        // packed storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          dK.m[i+(j*(j+1))/2]  = (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params+param_offset,param_i-(unsigned)param_offset,diff);
        }}
      }else{
        // full storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          dK.m[i+N1*j]  = (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params+param_offset,param_i-(unsigned)param_offset,diff);
          if(i!=j){ dK.m[j+N1*i] = dK.m[i+N1*j]; }
        }}
      }
    }else if(cov.cov_funcs&covProd){

       mlgp_freeCovFuncList(covFuncs);
       covFuncs = mlgp_createCovFuncList(cov,dim,curCovFunc,1);

      if(options.opts&_PACKED){
        // packed storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          dK.m[i+(j*(j+1))/2]  = 
          (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params+param_offset,diff,covFuncs)
          *
          (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params+param_offset,param_i-(unsigned)param_offset,diff);
        }}
      }else{
        // full storage
        for(int i=0;i<N1;i++){
        for(int j=i;j<N1;j++){
          dK.m[i+N1*j]  = 
          (*covFunc)(X1.m+i,N1,X1.m+j,N1,dim,cov.params+param_offset,diff,covFuncs)
          * 
          (*covFuncDeriv)(X1.m+i,N1,X2.m+j,N2,dim,cov.params+param_offset,param_i-(unsigned)param_offset,diff);
          if(i!=j){ dK.m[j+N1*i] = dK.m[i+N1*j]; }
        }}
      }
    }
    mlgp_freeCovFuncList(covFuncs);

  }

  // restore log(parameters)
  CBLAS_COPY(nparams,tempParams,1,cov.params,1);

  free(diff);

  printf("cov_composite terminate %d\n",param_i);
  return mlgpSuccess;

}
