/* r4d_ds41_f32pf.h -- lever GPF-b: the entry of gemm_nt_f32's tiled prefill form (r4d_ds41_f32pf.hip), called by
 * gemm_f32_body (r4d_glm.hip) ahead of the stock launch at the prefill rows. Off unless switched on: the runtime switch
 * f32_pf (libr4d's rad_fn4_* hooks, RADIANCE_DS41_F32_PF at load) or the bench pin r4d_f32pf_pin. Every form keeps the
 * stock kernel's per-output arithmetic (32 lane-strided fmaf chains, then the xor butterfly), so the bytes are the same,
 * for f32 and for bf16 weight storage alike. */
#ifndef R4D_DS41_F32PF_H
#define R4D_DS41_F32PF_H

#include <cstdint>

/* y[M, N] (f32, pitch yp) = x[M, K] (bf16, pitch xp) . w[N, K]^T (f32, or bf16 when wbf16; pitch wp), pitches in
 * elements, as gemm_f32_body validated them. */
struct R4dF32PfCall {
    const uint16_t* x;
    const void* w;
    float* y;
    long xp, wp, yp;
    int M, N, K;
    bool wbf16;
};

constexpr int R4D_F32PF_DECLINE = 1;

int r4d_f32pf_on();
/* RAD_OK, a launch failure, or R4D_F32PF_DECLINE (switched off, M < 64 without a pin, K not whole chunks). */
int r4d_f32pf_gemm(RadStream s, const R4dF32PfCall& c);
/* The runtime switch f32_pf: 0 off, 1 the rule (form 20 above 63 rows), 20.. a form code above 63 rows. 1 applied,
 * -1 an unknown value. */
int r4d_f32pf_set(int v);
int r4d_f32pf_get();
int r4d_f32pf_known(int v);

#endif
