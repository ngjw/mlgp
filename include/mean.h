#ifndef MEAN_H
#define MEAN_H

unsigned mlgp_nparams_mean(mlgpMean_t mean, unsigned dim);

#define meanZero    0x1u
#define meanOne     0x2u
#define meanConst   0x4u
#define meanLinear  0x8u

#endif /* MEAN_H */
