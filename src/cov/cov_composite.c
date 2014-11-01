#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t MLGP_COV_COMPOSITE (
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
  ptrdiff_t param_offset;
  FLOAT *diff; 
  COVFUNCCOMP_FT covFunc;
  FLOAT *tempParams;

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

  nparams = MLGP_NPARAMS_COV(cov,dim);

  tempParams = (FLOAT*)malloc(nparams*sizeof(FLOAT));

  diff = (FLOAT*)malloc(dim*sizeof(FLOAT));

  // parameter transformation
  MLGP_COPY(nparams,cov.params,1,tempParams,1);
  MLGP_COV_PARAM_TRANS(cov,dim);

  COVFUNCNODE_T *covFuncs = MLGP_CREATECOVFUNCLIST(cov,dim,0,0);

  if(cov.cov_funcs&covSum){
    covFunc = &MLGP_COVSUM;
  }else if(cov.cov_funcs&covProd){
    covFunc = &MLGP_COVPROD;
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

    COVFUNCDERIV_FT covFuncDeriv;

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
        covFuncDeriv = &MLGP_COVSEISO_DERIVATIVES;
        break;
      case covSEard:
        param_offset = count - NCP_SEard;
        covFuncDeriv = &MLGP_COVSEARD_DERIVATIVES;
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

       MLGP_FREECOVFUNCLIST(covFuncs);
       covFuncs = MLGP_CREATECOVFUNCLIST(cov,dim,curCovFunc,1);

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
    MLGP_FREECOVFUNCLIST(covFuncs);

  }

  // restore log(parameters)
  MLGP_COPY(nparams,tempParams,1,cov.params,1);

  free(diff);

  printf("cov_composite terminate %d\n",param_i);
  return mlgpSuccess;

}
