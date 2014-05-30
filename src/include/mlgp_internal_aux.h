#ifndef MLGP_INTERNAL_AUX_H
#define MLGP_INTERNAL_AUX_H

void chol(mlgpMatrix_t A);
void chol_packed(mlgpMatrix_t A);
void inv_chol(mlgpMatrix_t L);
void inv_chol_packed(mlgpMatrix_t L);
void solve_chol_multiple(mlgpMatrix_t L, mlgpMatrix_t B);
void solve_chol_one(mlgpMatrix_t L, mlgpVector_t B);
void solve_chol_packed_multiple(mlgpMatrix_t L, mlgpMatrix_t B);
void solve_chol_packed_one(mlgpMatrix_t L, mlgpVector_t B);
mlgpFloat_t log_det_tr(mlgpMatrix_t A);
mlgpFloat_t log_det_tr_packed(mlgpMatrix_t A);

#endif /* MLGP_INTERNAL_AUX_H */
