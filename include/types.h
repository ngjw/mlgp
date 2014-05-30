#ifndef TYPES_H
#define TYPES_H

#include <lbfgs.h>

/* types defined in this file
 *
 * ----------------------------------------------------------------------------
 * | mlgpFloat_t     | Floating point number type. If -DDOUBLE flag set at    |
 * |                 | at compilation then it is of type double, otherwise    |
 * |                 | it is of type float.                                   |
 * |--------------------------------------------------------------------------|
 * | mlgpStatus_t    | All functions in this library return this (enumerated) |
 * |                 | type indicating if the function call was successful    |
 * |                 | (mlgpSuccess) or one of the error codes.               |
 * |--------------------------------------------------------------------------|
 * | mlgpMean_t      | Contains information about the mean function to be     |
 * |                 | used in the GP.                                        |
 * |--------------------------------------------------------------------------|
 * | mlgpCov_t       | Contains information about the covariance function to  |
 * |                 | be used in the GP.                                     |
 * |--------------------------------------------------------------------------|
 * | mlgpLik_t       | Contains information about the likelihood function to  |
 * |                 | be used in the GP.                                     |
 * |--------------------------------------------------------------------------|
 * | mlgpInf_t       | Contains information about the inference method to be  |
 * |                 | used in the GP.                                        |
 * |--------------------------------------------------------------------------|
 * | mlgpOptions_t   | Contains the options to be passed into the functions   |
 * |                 | in this library.                                       |
 * |--------------------------------------------------------------------------|
 * | mlgpMatrix_t    | Struct with a pointer to a contiguous block of         |
 * |                 | memory containing a matrix stored in column major      |
 * |                 | format, number of rows and columns of the matrix.      |
 * |--------------------------------------------------------------------------|
 * | mlgpVector_t    | Struct with a pointer to a contiguous block of memory  |
 * |                 | containing a vector, and the length of the vector.     |
 * |--------------------------------------------------------------------------|
 * | mlgpTrainOpts_t | Contains the options for the l-BFGS optimisation used  |
 * |                 | in the train function.                                 |
 * |--------------------------------------------------------------------------|
 * | mlgpWorkspace_t | Struct with a pointer to pointer to block(s) of memory |
 * |                 | required as working memory/or for caching results.     |
 * ----------------------------------------------------------------------------
 *
 */

#ifdef DOUBLE
#define mlgpFloat_t double
#else
#define mlgpFloat_t float
#endif

typedef enum 
{ 
  mlgpSuccess = 0, 
  mlgpError = 1 
}
mlgpStatus_t;

typedef struct mlgpMean_t mlgpMean_t;
struct mlgpMean_t
{ 
  unsigned mean_funcs;
  unsigned nfuncs;
  mlgpFloat_t *params;
  mlgpFloat_t *dparams;
  mlgpMean_t *comps;
};

typedef struct mlgpCov_t mlgpCov_t;
struct mlgpCov_t
{ 
  unsigned cov_funcs;
  unsigned nfuncs;
  mlgpFloat_t *params;
  mlgpFloat_t *dparams;
  mlgpCov_t *comps;
};

typedef struct
{ 
  unsigned lik_func;
  mlgpFloat_t *params;
  mlgpFloat_t *dparams;
}
mlgpLik_t;

typedef struct
{ 
  unsigned inf_func;
}
mlgpInf_t;

typedef struct
{ 
  unsigned opts;
}
mlgpOptions_t;

typedef struct
{
  mlgpFloat_t *m;
  unsigned nrows;
  unsigned ncols;
}
mlgpMatrix_t;

typedef struct
{
  mlgpFloat_t *v;
  unsigned length;
}
mlgpVector_t;

#ifdef HAVELBFGS
typedef struct
{
  lbfgs_parameter_t lbfgsparams;
  unsigned use_defaults;
}
mlgpTrainOpts_t;
#endif

typedef struct
{
  mlgpFloat_t **ws;  
  unsigned size;
  unsigned allocated;
}
mlgpWorkspace_t;

#endif /* TYPES_H */
