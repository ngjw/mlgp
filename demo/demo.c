#include <stdio.h>

#include "../include/mlgp.h"

int main(){

  // load data
  unsigned N_tr, N_tt, dim;

  N_tr = 500; N_tt = 30; dim = 1;

  mlgpDMatrix_t x_tr = mlgp_readMatrix_dp(N_tr,dim,"../demo/data/training_inputs");
  mlgpDMatrix_t x_tt = mlgp_readMatrix_dp(N_tt,dim,"../demo/data/test_inputs");
  mlgpDVector_t y_tr = mlgp_readVector_dp(N_tr,"../demo/data/training_outputs");


  // specify mean/covariance/inference/likelihood functions/methods
  mlgpDMean_t mean = mlgp_createMean_dp(meanLinear,dim);
  mlgpDCov_t  cov  = mlgp_createCov_dp(covSEiso,dim);
  mlgpDInf_t  inf  = mlgp_createInf_dp(infExact);
  mlgpDLik_t  lik  = mlgp_createLik_dp(likGauss);

  unsigned np_mean = mlgp_nparams_mean_dp(mean,dim); 
  unsigned np_cov = mlgp_nparams_cov_dp(cov,dim); 

  // initialise parameters, set all to 1 for demo
  for(int i=0; i<np_mean;i++){ mean.params[i] = 1; }
  for(int i=0; i<np_cov;i++){ cov.params[i] = 1; }
  lik.params[0] = 1;

  // output variable (negative log likelihood)
  double nll;
  
  // workspace and options
  mlgpWorkspace_t workspace;
  mlgpOptions_t options; 

  options.opts = NOWORKSPACE|VERBOSE;

  // training function
  #ifdef HAVELBFGS
  mlgpTrainOpts_t trainopts;
  trainopts.use_defaults = 1;
  mlgp_train_dp(&nll,x_tr,y_tr,inf,mean,cov,lik,&workspace,trainopts,options);
  #endif

  // likelihood function
  mlgp_likelihood_dp(&nll,x_tr,y_tr,inf,mean,cov,lik,&workspace,options);

  // allocate space for prediction outputs
  mlgpDVector_t ymu  = mlgp_createVector_dp((unsigned)N_tt);
  mlgpDVector_t ys2  = mlgp_createVector_dp((unsigned)N_tt);
  mlgpDVector_t fmu  = mlgp_createVector_dp((unsigned)N_tt);
  mlgpDVector_t fs2  = mlgp_createVector_dp((unsigned)N_tt);

  mlgp_predict_dp(x_tr,y_tr,x_tt,ymu,ys2,fmu,fs2,inf,mean,cov,lik,&workspace,options);

  // print outputs
  printf("\n\nnegative log likelihood = %16.4lf",nll);

  printf("\n\nmean parameters and derivative w.r.t. mean parameters:\n");
  for(int i=0;i<np_mean;i++){
    printf("[%2d] %16.4lf  %16.4lf\n",i,mean.params[i],mean.dparams[i]);
  }

  printf("\n\ncov parameters and derivative w.r.t. cov parameters:\n");
  for(int i=0;i<np_cov;i++){
    printf("[%2d] %16.4lf  %16.4lf\n",i,cov.params[i],cov.dparams[i]);
  }

  printf("\n\nlik parameters and derivative w.r.t. lik parameters: \n[%d] %16.4lf  %16.4lf",0,lik.params[0],lik.dparams[0]);

  printf("\n\nprediction inputs, means, and variances: \n");
  for(int i=0; i<N_tt; i++){
    printf("[%2d] %16.4lf  %16.4lf  %16.4lf\n",i,x_tt.m[i],ymu.v[i], ys2.v[i]); 
  }

  printf("\n\n");

  mlgp_freeMatrix_dp(x_tt);
  mlgp_freeMatrix_dp(x_tr);
  mlgp_freeVector_dp(y_tr);

  mlgp_freeVector_dp(ymu);
  mlgp_freeVector_dp(ys2);
  mlgp_freeVector_dp(fmu);
  mlgp_freeVector_dp(fs2);

  mlgp_freeMean_dp(mean);
  mlgp_freeCov_dp(cov);
  mlgp_freeInf_dp(inf);
  mlgp_freeLik_dp(lik);

  return 0;
}
