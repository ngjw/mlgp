#ifndef COV_H
#define COV_H

unsigned mlgp_nparams_cov(mlgpCov_t cov, unsigned dim);

#define covSEiso    0x01u
#define covSEard    0x02u

// composite functions
#define covSum      0x04u
#define covProd     0x08u

#endif /* COV_H */
