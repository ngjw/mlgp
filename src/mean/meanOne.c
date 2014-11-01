#include "../include/mlgp_internal.h"
#include "../../include/mlgp.h"

mlgpStatus_t MLGP_MEANONE (
  VECTOR_T y,
  MATRIX_T X,
  MEAN_T mean,
  VECTOR_T dy,
  unsigned param_i,
  mlgpOptions_t options
)
{

    /* mean function m = (1,..,1) */

    FLOAT one = 1.;

    if(options.opts&_ADDMEAN){
      MLGP_AXPY(y.length,1,&one,0,y.v,1);
    }else if(options.opts&_SUBMEAN){
      MLGP_AXPY(y.length,-1,&one,0,y.v,1);
    }

    return mlgpSuccess;

}
