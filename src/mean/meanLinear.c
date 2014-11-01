#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t MLGP_MEANLINEAR (
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m[i] = sum_i^d a_i*x_i */

    if(!(options.opts&_DERIVATIVES)){
      unsigned dim = X.ncols;

      if(options.opts&_ADDMEAN){
        for(int i=0;i<y.length;i++){
          y.v[i]+= MLGP_DOT(dim,mean.params,1,X.m+i,X.nrows);
        }
      }else if(options.opts&_SUBMEAN){
        for(int i=0;i<y.length;i++){
          y.v[i]-= MLGP_DOT(dim,mean.params,1,X.m+i,X.nrows);
        }
      }
    }else{
      MLGP_COPY(dy.length,X.m+param_i*X.nrows,1,dy.v,1);
    }

    return mlgpSuccess;

}
