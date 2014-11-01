#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t MLGP_MEANCONST (
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m = (c,..,c) */

    if(!(options.opts&_DERIVATIVES)){
      if(options.opts&_ADDMEAN){
        MLGP_AXPY(y.length,1,mean.params,0,y.v,1);
      }else if(options.opts&_SUBMEAN){
        MLGP_AXPY(y.length,-1,mean.params,0,y.v,1);
      }
    }else{
      FLOAT one = 1;
      MLGP_COPY(dy.length,&one,0,dy.v,1);
    }

    return mlgpSuccess;

}
