/* r4d_gemm_bf16_nt_pf.h -- drafter_bf: the entry of the skinny bf16 GEMM's prefetching decode form
 * (r4d_gemm_bf16_nt_pf.hip), called by r4d_rad_gemm_bf16_m64 (r4d_gemm_bf16_nt_m64.hip) ahead of its stock launch
 * when MB is 1 and the launch is not chained. Off unless switched on: the runtime switch drafter_bf (libr4d's
 * rad_fn4_* hooks, RADIANCE_DS41_DRAFTER_BF at load) or the bench pin r4d_bf_pf_pin. */
#ifndef R4D_GEMM_BF16_NT_PF_H
#define R4D_GEMM_BF16_NT_PF_H

#include <cstdint>

/* C[M, N] = A[M, K] . W[N, K]^T, bf16, row pitches lda / ldc in elements, SK the stock launch's split. */
struct R4dBfPfCall {
    const void* a;
    const void* w;
    void* y;
    int M, K, N, SK;
    long long lda, ldc;
};

constexpr int R4D_BF_PF_DECLINE = 1;

int r4d_bf_pf_on();
int r4d_bf_pf_gemm(RadStream s, const R4dBfPfCall& c);
void r4d_bf_pf_set(int on);
int r4d_bf_pf_get();

#endif
