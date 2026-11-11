/* r4d_ds41_ggml_dual.h -- lever BI-e: gemm_nt_ggml over two row groups of one weight in ONE launch
 * (r4d_ds41_ggml_dual.hip), called by gemm_nt_ggml_dual's entry (r4d_ds41_ggml.hip) ahead of its two gemm_nt_ggml
 * launches.
 *
 * A mixed step under lever BI (deepseek41 ds41_bi) issues each dense GGUF weight once a row group: its decode rows
 * (M0 <= 48: the decode band, lever D2's multi-row form / the stock GEMV, one bit pattern) and its prefill rows (each
 * prefill sequence above the band: the WMMA kernel / lever GPF's forms, one bit pattern). This launch runs both
 * groups: blocks [0, nA) are the multi-row form's blocks over rows a, the others the WMMA form's blocks over rows a1
 * (or the reverse order), each block running that form's kernel body unchanged, so every output is the bytes the
 * group's own launch gives. Fewer launches, and the decode group's weight read beside the prefill group's (the
 * weight in L2 / MALL for one of them) instead of a launch of its own. */
#ifndef R4D_DS41_GGML_DUAL_H
#define R4D_DS41_GGML_DUAL_H

#include <cstdint>

struct R4dGgDualCall {
    int ty;                       /* GG_* (q5_K, q6_K, q8_0 here) */
    const uint8_t* w;
    long long w_ld;               /* bytes a weight row */
    int N, K;
    const uint16_t* x0;           /* the decode group: M0 rows (1..48), bf16, x0_ld elements a row */
    long long x0_ld;
    void* y0;
    long long y0_ld;
    int M0;
    const uint16_t* x1;           /* the prefill group: M1 rows (> 48), bf16, 16-byte aligned rows */
    long long x1_ld;
    void* y1;
    long long y1_ld;
    int M1;
};

/* r4d_gg_dual_gemm's answer for a pair it does not take (a block type, a shape or a prefill form it has no kernel
 * for, M0 past the band or M1 inside it): the caller issues the two launches. */
constexpr int R4D_GG_DUAL_DECLINE = 1;

/* One launch on `s`: RAD_OK, a RAD_E_* launch failure, or R4D_GG_DUAL_DECLINE. */
int r4d_gg_dual_gemm(RadStream s, const R4dGgDualCall& c);

#endif
