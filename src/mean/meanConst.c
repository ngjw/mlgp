#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t mlgp_meanConst (
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m = (c,..,c) */

    if(!(options.opts&_DERIVATIVES)){
      if(options.opts&_ADDMEAN){
        CBLAS_AXPY(y.length,1,mean.params,0,y.v,1);
      }else if(options.opts&_SUBMEAN){
        CBLAS_AXPY(y.length,-1,mean.params,0,y.v,1);
      }
    }else{
      mlgpFloat_t one = 1;
      CBLAS_COPY(dy.length,&one,0,dy.v,1);
    }

    return mlgpSuccess;

}
