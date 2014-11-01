#ifndef TYPES_H
#define TYPES_H

/* types defined in this file
 *
 * ----------------------------------------------------------------------------
 * | mlgpStatus_t    | All functions in this library return this (enumerated) |
 * |                 | type indicating if the function call was successful    |
 * |                 | (mlgpSuccess) or one of the error codes.               |
 * |--------------------------------------------------------------------------|
 * | mlgpSMean_t     | Contains information about the mean function to be     |
 * | mlgpDMean_t     | used in the GP.                                        |
 * |                 | mlgpSMean_t - Single precision.                        |
 * |                 | mlgpDMean_t - Double precision.                        |
 * |--------------------------------------------------------------------------|
 * | mlgpSCov_t      | Contains information about the covariance function to  |
 * | mlgpDCov_t      | be used in the GP.                                     |
 * |                 | mlgpSCov_t - Single Precision.                          |
 * |                 | mlgpDCov_t - Double Precision.                          |
 * |--------------------------------------------------------------------------|
 * | mlgpSLik_t      | Contains information about the likelihood function to  |
 * | mlgpDLik_t      | be used in the GP.                                     |
 * |                 | mlgpSLik_t - Single precision.                          |
 * |                 | mlgpDLik_t - Single precision.                          |
 * |--------------------------------------------------------------------------|
 * | mlgpSInf_t      | Contains information about the inference method to be  |
 * | mlgpDInf_t      | used in the GP.                                        |
 * |                 | mlgpSInf_t - Single precision.                         |
 * |                 | mlgpSInf_t - Double precision.                         |
 * |--------------------------------------------------------------------------|
 * | mlgpOptions_t   | Contains the options to be passed into the functions   |
 * |                 | in this library.                                       |
 * |--------------------------------------------------------------------------|
 * | mlgpSMatrix_t   | Struct with a pointer to a contiguous block of         |
 * | mlgpDMatrix_t   | memory containing a matrix stored in column major      |
 * |                 | format, number of rows and columns of the matrix.      |
 * |                 | mlgpSMatrix_t - Single precision.                      |
 * |                 | mlgpDMatrix_t - Double precision.                      |
 * |--------------------------------------------------------------------------|
 * | mlgpSVector_t   | Struct with a pointer to a contiguous block of memory  |
 * | mlgpDVector_t   | containing a vector, and the length of the vector.     |
 * |                 | mlgpSVector_t - Single precision.                      |
 * |                 | mlgpDVector_t - Double precision.                      |
 * |--------------------------------------------------------------------------|
 * | mlgpTrainOpts_t | Contains the options for the l-BFGS optimisation used  |
 * |                 | in the train function.                                 |
 * |--------------------------------------------------------------------------|
 * | mlgpWorkspace_t | Struct with a pointer to pointer to block(s) of memory |
 * |                 | required as working memory/or for caching results.     |
 * ----------------------------------------------------------------------------
 *
 */

typedef enum 
{ 
  mlgpSuccess = 0, 
  mlgpError = 1 
}
mlgpStatus_t;

typedef struct mlgpSMean_t mlgpSMean_t;
struct mlgpSMean_t
{ 
  unsigned mean_funcs;
  unsigned nfuncs;
  float *params;
  float *dparams;
  mlgpSMean_t *comps; // for composite mean functions
};

typedef struct mlgpDMean_t mlgpDMean_t;
struct mlgpDMean_t
{ 
  unsigned mean_funcs;
  unsigned nfuncs;
  double *params;
  double *dparams;
  mlgpDMean_t *comps; // for composite mean functions
};

typedef struct mlgpSCov_t mlgpSCov_t;
struct mlgpSCov_t
{ 
  unsigned cov_funcs;
  unsigned nfuncs;
  float *params;
  float *dparams;
  mlgpSCov_t *comps; // for composite covariance functions
};

typedef struct mlgpDCov_t mlgpDCov_t;
struct mlgpDCov_t
{ 
  unsigned cov_funcs;
  unsigned nfuncs;
  double *params;
  double *dparams;
  mlgpDCov_t *comps; // for composite covariance functions
};

typedef struct
{ 
  unsigned lik_func;
  float *params;
  float *dparams;
}
mlgpSLik_t;

typedef struct
{ 
  unsigned lik_func;
  double *params;
  double *dparams;
}
mlgpDLik_t;

typedef struct
{ 
  unsigned inf_func;
}
mlgpSInf_t;

typedef struct
{ 
  unsigned inf_func;
}
mlgpDInf_t;

typedef struct
{ 
  unsigned opts;
}
mlgpOptions_t;

typedef struct
{
  float *m;
  unsigned nrows;
  unsigned ncols;
}
mlgpSMatrix_t;

typedef struct
{
  double *m;
  unsigned nrows;
  unsigned ncols;
}
mlgpDMatrix_t;

typedef struct
{
  float *v;
  unsigned length;
}
mlgpSVector_t;

typedef struct
{
  double *v;
  unsigned length;
}
mlgpDVector_t;

#ifdef HAVELBFGS
#include <lbfgs.h>

typedef struct
{
  lbfgs_parameter_t lbfgsparams;
  unsigned use_defaults;
}
mlgpTrainOpts_t;
#endif

typedef struct
{
  void **ws; // pointer to memory (cast to float/double)
  unsigned size;
  unsigned allocated;
}
mlgpWorkspace_t;

#endif /* TYPES_H */
