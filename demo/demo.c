#include <stdio.h>

#include "../include/mlgp.h"

int main(){

  // load data
  unsigned N_tr, N_tt, dim;

  N_tr = 500; N_tt = 30; dim = 1;

  mlgpMatrix_t x_tr = mlgp_readMatrix(N_tr,dim,"../demo/data/training_inputs");
  mlgpMatrix_t x_tt = mlgp_readMatrix(N_tt,dim,"../demo/data/test_inputs");
  mlgpVector_t y_tr = mlgp_readVector(N_tr,"../demo/data/training_outputs");


  // specify mean/covariance/inference/likelihood functions/methods
  mlgpMean_t mean = mlgp_createMean(meanLinear,dim);
  mlgpCov_t  cov  = mlgp_createCov(covSEiso,dim);
  mlgpInf_t  inf  = mlgp_createInf(infExact);
  mlgpLik_t  lik  = mlgp_createLik(likGauss);

  unsigned np_mean = mlgp_nparams_mean(mean,dim); 
  unsigned np_cov = mlgp_nparams_cov(cov,dim); 

  // initialise parameters, set all to 1 for demo
  for(int i=0; i<np_mean;i++){ mean.params[i] = 1; }
  for(int i=0; i<np_cov;i++){ cov.params[i] = 1; }
  lik.params[0] = 1;

  // output variable (negative log likelihood)
  mlgpFloat_t nll;
  
  // workspace and options
  mlgpWorkspace_t workspace;
  mlgpOptions_t options; 

  options.opts = NOWORKSPACE|VERBOSE;

  // training function
  #ifdef HAVELBFGS
  mlgpTrainOpts_t trainopts;
  trainopts.use_defaults = 1;
  mlgp_train(&nll,x_tr,y_tr,inf,mean,cov,lik,&workspace,trainopts,options);
  #endif

  // likelihood function
  mlgp_likelihood(&nll,x_tr,y_tr,inf,mean,cov,lik,&workspace,options);

  // allocate space for prediction outputs
  mlgpVector_t ymu  = mlgp_createVector((unsigned)N_tt);
  mlgpVector_t ys2  = mlgp_createVector((unsigned)N_tt);
  mlgpVector_t fmu  = mlgp_createVector((unsigned)N_tt);
  mlgpVector_t fs2  = mlgp_createVector((unsigned)N_tt);

  mlgp_predict(x_tr,y_tr,x_tt,ymu,ys2,fmu,fs2,inf,mean,cov,lik,&workspace,options);

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

  mlgp_freeMatrix(x_tt);
  mlgp_freeMatrix(x_tr);
  mlgp_freeVector(y_tr);

  mlgp_freeVector(ymu);
  mlgp_freeVector(ys2);
  mlgp_freeVector(fmu);
  mlgp_freeVector(fs2);

  mlgp_freeMean(mean);
  mlgp_freeCov(cov);
  mlgp_freeInf(inf);
  mlgp_freeLik(lik);

  return 0;
}
