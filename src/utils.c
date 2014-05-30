#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

/* functions defined in this file
 *
 * unsigned countBits(unsigned b);
 *
 * mlgpMean_t mlgp_createMean(unsigned mean_funcs, unsigned dim);
 * mlgpCov_t  mlgp_createCov (unsigned cov_funcs,  unsigned dim);
 * mlgpInf_t  mlgp_createInf (unsigned inf_func);
 * mlgpLik_t  mlgp_createLik (unsigned lik_func);
 * 
 * mlgpMatrix_t mlgp_createMatrix(unsigned nrows, unsigned ncols);
 * mlgpVector_t mlgp_createVector(unsigned length);
 * 
 * mlgpMatrix_t mlgp_createMatrixNoMalloc(unsigned nrows, unsigned ncols);
 * mlgpVector_t mlgp_createVectorNoMalloc(unsigned length);
 * 
 * mlgpMatrix_t mlgp_readMatrix(unsigned nrows, unsigned ncols, const char *filename);
 * mlgpVector_t mlgp_readVector(unsigned length, const char *filename);
 * 
 * mlgpStatus_t mlgp_freeMean(mlgpMean_t mean);
 * mlgpStatus_t mlgp_freeCov (mlgpCov_t  cov );
 * mlgpStatus_t mlgp_freeInf (mlgpInf_t  inf );
 * mlgpStatus_t mlgp_freeLik (mlgpLik_t  lik );
 *
 * mlgpStatus_t mlgp_freeMatrix(mlgpMatrix_t m);
 * mlgpStatus_t mlgp_freeVector(mlgpVector_t v);
 * mlgpStatus_t mlgp_freeWorkspace(mlgpWorkspace_t ws);
 * 
 */

unsigned countBits (
  unsigned b
)
{
  // counts the number of 1 bits in an unsigned integer
  unsigned count = 0;
  while(b){
    count++;
    b&=(b-1);
  }
  return count;
}

mlgpMean_t mlgp_createMean (
  unsigned mean_funcs,
  unsigned dim
)
{
  /* creates a mlgpMean_t struct and allocates the required memory for it */
  mlgpMean_t mean = {.mean_funcs=mean_funcs, .nfuncs=1, .params=NULL, .dparams=NULL  };
  unsigned nparams = mlgp_nparams_mean(mean,dim);
  mean.params = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));
  mean.dparams = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));
  return mean;
}

mlgpCov_t  mlgp_createCov (
unsigned cov_funcs,  unsigned dim
)
{
  /* creates a mlgpCov_t struct and allocates the required memory for it */
  mlgpCov_t cov = {.cov_funcs=cov_funcs, .nfuncs=1, .params=NULL, .dparams=NULL  };
  unsigned nparams = mlgp_nparams_cov(cov,dim);
  cov.params = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));
  cov.dparams = (mlgpFloat_t*)malloc(nparams*sizeof(mlgpFloat_t));
  cov.nfuncs = countBits(cov_funcs&(~covSum)&(~covProd));
  return cov;
}

mlgpInf_t  mlgp_createInf (
  unsigned inf_func
)
{
  /* creates a mlgpInf_t struct */
  mlgpInf_t inf = {.inf_func = inf_func };
  return inf;
}

mlgpLik_t  mlgp_createLik (
  unsigned lik_func
)
{
  /* creates a mlgpLik_t struct and allocates the required memory for it */
  mlgpLik_t lik = {.lik_func = lik_func };
  lik.params = (mlgpFloat_t*)malloc(sizeof(mlgpFloat_t));
  lik.dparams = (mlgpFloat_t*)malloc(sizeof(mlgpFloat_t));
  return lik;
}

mlgpMatrix_t mlgp_createMatrix (
  unsigned nrows, 
  unsigned ncols
)
{

  /* creates a mlgpMatrix_t struct and allocates required memory */

  mlgpMatrix_t matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.m = (mlgpFloat_t*)malloc(nrows*ncols*sizeof(mlgpFloat_t));
  return matrix;
}

mlgpMatrix_t mlgp_createMatrixNoMalloc (
  unsigned nrows, 
  unsigned ncols
)
{

  /* creates a mlgpMatrix_t struct without allocating memory */

  mlgpMatrix_t matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.m = NULL;
  return matrix;
}

mlgpVector_t mlgp_createVector (
  unsigned length
)
{

  /* creates a mlgpVector_t struct and allocates required memory */

  mlgpVector_t vector;
  vector.length = length;
  vector.v = (mlgpFloat_t*)malloc(length*sizeof(mlgpFloat_t));
  return vector;
}

mlgpVector_t mlgp_createVectorNoMalloc (
  unsigned length
)
{

  /* creates a mlgpVector_t struct without allocating memory */

  mlgpVector_t vector;
  vector.length = length;
  vector.v = NULL;
  return vector;
}

mlgpMatrix_t mlgp_readMatrix (
  unsigned nrows, 
  unsigned ncols,
  const char *filename
)
{

  /* creates a mlgpMatrix_t struct and reads the data in a given file into it */

  mlgpMatrix_t matrix = mlgp_createMatrix(nrows,ncols);
  FILE *fp;
  unsigned row = 0, col = 0;
  unsigned i = 0;

  fp = fopen(filename,"r");

	#ifdef DOUBLE
	while(fscanf(fp,"%lf",matrix.m+row+col*nrows) != EOF && i<nrows*ncols){
    col++;
    if(col==ncols){
      col = 0;
      row+=1;
    }
    i++;
  }
	#else
	while(fscanf(fp,"%f",matrix.m+row+col*nrows) != EOF && i<nrows*ncols){
    col++;
    if(col==ncols){
      col = 0;
      row+=1;
    }
    i++;
  }
	#endif
	fclose(fp);

  return matrix;
}

mlgpVector_t mlgp_readVector (
  unsigned length,
  const char *filename
)
{

  /* creates a mlgpVector_t struct and reads the data in a given file into it */

  mlgpVector_t vector = mlgp_createVector(length);

  FILE *fp;
	unsigned i=0;

  fp = fopen(filename,"r");

	#ifdef DOUBLE
	while(fscanf(fp,"%lf",vector.v+i) != EOF && i<vector.length) i++; 
	#else
	while(fscanf(fp,"%f",vector.v+i) != EOF && i<vector.length) i++; 
	#endif
	fclose(fp);

  return vector;
}

mlgpStatus_t mlgp_freeMean (
  mlgpMean_t mean
)
{
  /* frees the memory in a mlgpMean_t */
  free(mean.params);
  free(mean.dparams);
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeCov (
  mlgpCov_t cov
)
{
  /* frees the memory in a mlgpCov_t */
  free(cov.params);
  free(cov.dparams);
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeLik (
  mlgpLik_t lik
)
{
  /* frees the memory in a mlgpLik_t */
  free(lik.params);
  free(lik.dparams);
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeInf (
  mlgpInf_t inf
)
{
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeMatrix (
  mlgpMatrix_t m
)
{
  /* frees the memory in a mlgpMatrix_t */
  free(m.m);
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeVector (
  mlgpVector_t v
)
{
  /* frees the memory in a mlgpVector_t */
  free(v.v);
  return mlgpSuccess;
}

mlgpStatus_t mlgp_freeWorkspace (
  mlgpWorkspace_t workspace
)
{
  /* frees the memory in a mlgpWorkspace_t */
  unsigned i = 0;
  while(workspace.allocated){
    if(workspace.allocated%2){
      free(workspace.ws[i]);
    }
    workspace.allocated>>=1;
    i++;
  }
  free(workspace.ws);
  return mlgpSuccess;
}
