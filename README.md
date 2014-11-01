Machine Learning with Gaussian Processes
====

This is a C implementation of Gaussian Process (GPs).

The motivation for this library was to create an efficient and portable library
that implements GPs.

The current library implements GP regression, includes just a few mean
functions and two covariance functions, but more will be added (work is
ongoing).

This library will also include a GPU (CUDA) implementation of GPs based on a
prototype developed as part of a MSc project at the Department of Computing,
Imperial College London.

Other features to be added:
- GPs for classification
- Mex compilation scripts (for Matlab)
- Python module

Library dependencies:
- BLAS
- LAPACK
- libLBFGS (this is optional, required for the training function. Can be found
  at http://www.chokkan.org/software/liblbfgs/)

Update 02/11/2014:
- Compilation of both single and double precision code simultaneously into the
  same library. Many macros added to specify types and functions for each
  precision (as of now, this is done by having the same code compiled twice).
- Removed CBLAS dependency.
