#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t MLGP_COV(
  MATRIX_T K,
  MATRIX_T X1,
  MATRIX_T X2,
  COV_T cov,
  MATRIX_T dK,
  unsigned param_i,
  mlgpOptions_t options
){

  if(cov.cov_funcs&covCOMPOSITE){
    return MLGP_COV_COMPOSITE(K,X1,X2,cov,dK,param_i,options);
  }else{
    return MLGP_COV_SINGLE(K,X1,X2,cov,dK,param_i,options); 
  }

}
