Machine Learning with Gaussian Processes
====

This is a C implementation of Gaussian Process (GPs).

The motivation for this library was to create an efficient and portable library
that implements GPs.

The current library implements GP regression, includes just a few mean functions
and two covariance functions, but more will be added (work is ongoing).

This library will also include a GPU (CUDA) implementation of GPs based on a
prototype developed as part of a MSc project at the Department of Computing,
Imperial College London.

Other features to be added:
- GPs for classification
- Mex compilation scripts (for Matlab)
- Python module
