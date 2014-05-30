#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t mlgp_meanLinear (
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m[i] = sum_i^d a_i*x_i */

    if(!(options.opts&_DERIVATIVES)){
      unsigned dim = X.ncols;

      if(options.opts&_ADDMEAN){
        for(int i=0;i<y.length;i++){
          y.v[i]+= CBLAS_DOT(dim,mean.params,1,X.m+i,X.nrows);
        }
      }else if(options.opts&_SUBMEAN){
        for(int i=0;i<y.length;i++){
          y.v[i]-= CBLAS_DOT(dim,mean.params,1,X.m+i,X.nrows);
        }
      }
    }else{
      CBLAS_COPY(dy.length,X.m+param_i*X.nrows,1,dy.v,1);
    }

    return mlgpSuccess;

}
