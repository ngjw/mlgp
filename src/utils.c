#include "../include/mlgp.h"
#include "include/mlgp_internal.h"

/* functions defined in this file
 *
 * unsigned countBits(unsigned b);
 *
 * MEAN_T mlgp_createMean(unsigned mean_funcs, unsigned dim);
 * COV_T  mlgp_createCov (unsigned cov_funcs,  unsigned dim);
 * INF_T  mlgp_createInf (unsigned inf_func);
 * LIK_T  mlgp_createLik (unsigned lik_func);
 * 
 * MATRIX_T mlgp_createMatrix(unsigned nrows, unsigned ncols);
 * VECTOR_T mlgp_createVector(unsigned length);
 * 
 * MATRIX_T mlgp_createMatrixNoMalloc(unsigned nrows, unsigned ncols);
 * VECTOR_T mlgp_createVectorNoMalloc(unsigned length);
 * 
 * MATRIX_T mlgp_readMatrix(unsigned nrows, unsigned ncols, const char *filename);
 * VECTOR_T mlgp_readVector(unsigned length, const char *filename);
 * 
 * mlgpStatus_t mlgp_freeMean(MEAN_T mean);
 * mlgpStatus_t mlgp_freeCov (COV_T  cov );
 * mlgpStatus_t mlgp_freeInf (INF_T  inf );
 * mlgpStatus_t mlgp_freeLik (LIK_T  lik );
 *
 * mlgpStatus_t mlgp_freeMatrix(MATRIX_T m);
 * mlgpStatus_t mlgp_freeVector(VECTOR_T v);
 * mlgpStatus_t mlgp_freeWorkspace(mlgpWorkspace_t ws);
 * 
 */

MEAN_T MLGP_CREATEMEAN (
  unsigned mean_funcs,
  unsigned dim
)
{
  /* creates a MEAN_T struct and allocates the required memory for it */
  MEAN_T mean = {.mean_funcs=mean_funcs, .nfuncs=1, .params=NULL, .dparams=NULL  };
  unsigned nparams = MLGP_NPARAMS_MEAN(mean,dim);
  mean.params = (FLOAT*)malloc(nparams*sizeof(FLOAT));
  mean.dparams = (FLOAT*)malloc(nparams*sizeof(FLOAT));
  return mean;
}

COV_T  MLGP_CREATECOV (
unsigned cov_funcs,  unsigned dim
)
{
  /* creates a COV_T struct and allocates the required memory for it */
  COV_T cov = {.cov_funcs=cov_funcs, .nfuncs=1, .params=NULL, .dparams=NULL  };
  unsigned nparams = MLGP_NPARAMS_COV(cov,dim);
  cov.params = (FLOAT*)malloc(nparams*sizeof(FLOAT));
  cov.dparams = (FLOAT*)malloc(nparams*sizeof(FLOAT));
  cov.nfuncs = countBits(cov_funcs&(~covSum)&(~covProd));
  return cov;
}

INF_T  MLGP_CREATEINF (
  unsigned inf_func
)
{
  /* creates a INF_T struct */
  INF_T inf = {.inf_func = inf_func };
  return inf;
}

LIK_T  MLGP_CREATELIK (
  unsigned lik_func
)
{
  /* creates a LIK_T struct and allocates the required memory for it */
  LIK_T lik = {.lik_func = lik_func };
  lik.params = (FLOAT*)malloc(sizeof(FLOAT));
  lik.dparams = (FLOAT*)malloc(sizeof(FLOAT));
  return lik;
}

MATRIX_T MLGP_CREATEMATRIX (
  unsigned nrows, 
  unsigned ncols
)
{

  /* creates a MATRIX_T struct and allocates required memory */

  MATRIX_T matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.m = (FLOAT*)malloc(nrows*ncols*sizeof(FLOAT));
  return matrix;
}

MATRIX_T MLGP_CREATEMATRIXNOMALLOC (
  unsigned nrows, 
  unsigned ncols
)
{

  /* creates a MATRIX_T struct without allocating memory */

  MATRIX_T matrix;
  matrix.nrows = nrows;
  matrix.ncols = ncols;
  matrix.m = NULL;
  return matrix;
}

VECTOR_T MLGP_CREATEVECTOR (
  unsigned length
)
{

  /* creates a VECTOR_T struct and allocates required memory */

  VECTOR_T vector;
  vector.length = length;
  vector.v = (FLOAT*)malloc(length*sizeof(FLOAT));
  return vector;
}

VECTOR_T MLGP_CREATEVECTORNOMALLOC (
  unsigned length
)
{

  /* creates a VECTOR_T struct without allocating memory */

  VECTOR_T vector;
  vector.length = length;
  vector.v = NULL;
  return vector;
}

MATRIX_T MLGP_READMATRIX (
  unsigned nrows, 
  unsigned ncols,
  const char *filename
)
{

  /* creates a MATRIX_T struct and reads the data in a given file into it */

  MATRIX_T matrix = MLGP_CREATEMATRIX(nrows,ncols);
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

VECTOR_T MLGP_READVECTOR (
  unsigned length,
  const char *filename
)
{

  /* creates a VECTOR_T struct and reads the data in a given file into it */

  VECTOR_T vector = MLGP_CREATEVECTOR(length);

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

mlgpStatus_t MLGP_FREEMEAN (
  MEAN_T mean
)
{
  /* frees the memory in a MEAN_T */
  free(mean.params);
  free(mean.dparams);
  return mlgpSuccess;
}

mlgpStatus_t MLGP_FREECOV (
  COV_T cov
)
{
  /* frees the memory in a COV_T */
  free(cov.params);
  free(cov.dparams);
  return mlgpSuccess;
}

mlgpStatus_t MLGP_FREELIK (
  LIK_T lik
)
{
  /* frees the memory in a LIK_T */
  free(lik.params);
  free(lik.dparams);
  return mlgpSuccess;
}

mlgpStatus_t MLGP_FREEINF (
  INF_T inf
)
{
  return mlgpSuccess;
}

mlgpStatus_t MLGP_FREEMATRIX (
  MATRIX_T m
)
{
  /* frees the memory in a MATRIX_T */
  free(m.m);
  return mlgpSuccess;
}

mlgpStatus_t MLGP_FREEVECTOR (
  VECTOR_T v
)
{
  /* frees the memory in a VECTOR_T */
  free(v.v);
  return mlgpSuccess;
}

#ifdef COMPILEONCE

/* these functions are only compiled once (in the case where both
 * single precision and double precision code is to be compiled */

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
#endif /* COMPILEONCE */
