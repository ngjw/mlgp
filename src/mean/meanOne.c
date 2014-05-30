#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t mlgp_meanOne (
  mlgpVector_t y,
  mlgpMatrix_t X,
  mlgpMean_t mean,
  mlgpVector_t dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m = (1,..,1) */

    mlgpFloat_t one = 1.;

    if(options.opts&_ADDMEAN){
      CBLAS_AXPY(y.length,1,&one,0,y.v,1);
    }else if(options.opts&_SUBMEAN){
      CBLAS_AXPY(y.length,-1,&one,0,y.v,1);
    }

    return mlgpSuccess;

}
