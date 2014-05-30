#include "../../include/mlgp.h"
#include "../include/mlgp_internal.h"

mlgpStatus_t mlgp_cov(
  mlgpMatrix_t K,
  mlgpMatrix_t X1,
  mlgpMatrix_t X2,
  mlgpCov_t cov,
  mlgpMatrix_t dK,
  unsigned param_i,
  mlgpOptions_t options
){

  if(cov.cov_funcs&covCOMPOSITE){
    return mlgp_cov_composite(K,X1,X2,cov,dK,param_i,options);
  }else{
    return mlgp_cov_single(K,X1,X2,cov,dK,param_i,options); 
  }

}
