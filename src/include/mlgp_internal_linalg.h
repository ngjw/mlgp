#ifndef MLGP_INTERNAL_AUX_H
#define MLGP_INTERNAL_AUX_H

#ifdef DOUBLE
#define CHOL(...) chol_dp(__VA_ARGS__)
#define CHOL_PACKED(...) chol_packed_dp(__VA_ARGS__)
#define INV_CHOL(...) inv_chol_dp(__VA_ARGS__)
#define INV_CHOL_PACKED(...) inv_chol_packed_dp(__VA_ARGS__)
#define SOLVE_CHOL_MULTIPLE(...) solve_chol_multiple_dp(__VA_ARGS__)
#define SOLVE_CHOL_ONE(...) solve_chol_one_dp(__VA_ARGS__)
#define SOLVE_CHOL_PACKED_MULTIPLE(...) solve_chol_packed_multiple_dp(__VA_ARGS__)
#define SOLVE_CHOL_PACKED_ONE(...) solve_chol_packed_one_dp(__VA_ARGS__)
#define LOG_DET_TR(...) log_det_tr_dp(__VA_ARGS__)
#define LOG_DET_TR_PACKED(...) log_det_tr_packed_dp(__VA_ARGS__)
#else
#define CHOL(...) chol_sp(__VA_ARGS__)
#define CHOL_PACKED(...) chol_packed_sp(__VA_ARGS__)
#define INV_CHOL(...) inv_chol_sp(__VA_ARGS__)
#define INV_CHOL_PACKED(...) inv_chol_packed_sp(__VA_ARGS__)
#define SOLVE_CHOL_MULTIPLE(...) solve_chol_multiple_sp(__VA_ARGS__)
#define SOLVE_CHOL_ONE(...) solve_chol_one_sp(__VA_ARGS__)
#define SOLVE_CHOL_PACKED_MULTIPLE(...) solve_chol_packed_multiple_sp(__VA_ARGS__)
#define SOLVE_CHOL_PACKED_ONE(...) solve_chol_packed_one_sp(__VA_ARGS__)
#define LOG_DET_TR(...) log_det_tr_sp(__VA_ARGS__)
#define LOG_DET_TR_PACKED(...) log_det_tr_packed_sp(__VA_ARGS__)
#endif

int CHOL(MATRIX_T A);
int CHOL_PACKED(MATRIX_T A);
int INV_CHOL(MATRIX_T L);
int INV_CHOL_PACKED(MATRIX_T L);
int SOLVE_CHOL_MULTIPLE(MATRIX_T L, MATRIX_T B);
int SOLVE_CHOL_ONE(MATRIX_T L, VECTOR_T B);
int SOLVE_CHOL_PACKED_MULTIPLE(MATRIX_T L, MATRIX_T B);
int SOLVE_CHOL_PACKED_ONE(MATRIX_T L, VECTOR_T B);
FLOAT LOG_DET_TR(MATRIX_T A);
FLOAT LOG_DET_TR_PACKED(MATRIX_T A);

#endif /* MLGP_INTERNAL_AUX_H */
