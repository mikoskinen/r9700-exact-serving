/* r4d_ds41_ggml_pf.h -- lever GPF: the entry of gemm_nt_ggml / logits_gemm_ggml's prefill WMMA forms
 * (r4d_ds41_ggml_pf.hip), called by gg_gemm (r4d_ds41_ggml.hip) ahead of the stock WMMA launch (gg_wm_kernel) for
 * the q5_K / q6_K / q8_0 rows above the decode band.
 *
 * Off unless switched on: the runtime switch gg_pf (libr4d's rad_fn4_* hooks, RADIANCE_DS41_GG_PF at load) or the
 * bench pin r4d_gg_pf_pin. Off, gg_gemm runs today's kernel untouched. Every form has the stock kernel's per-output
 * arithmetic (the same WMMA fragments, chains and scale fmas in the same order), so the bytes are the same. */
#ifndef R4D_DS41_GGML_PF_H
#define R4D_DS41_GGML_PF_H

#include <cstdint>

/* One prefill GEMM as gg_gemm resolved it for the stock WMMA kernel: y[M, N] = x[M, K] . w[N, K]^T, the weight in
 * the type's native blocks (w_ld bytes a row), x bf16 (x_ld elements a row, 16-byte aligned rows), y bf16 or, for
 * logits_gemm_ggml, f32 (y_ld elements a row). `ty` is gg_gemm's GG_* code. */
struct R4dGgPfCall {
    int ty;
    bool f32out;
    const uint16_t* x;
    long long x_ld;
    const uint8_t* w;
    long long w_ld;
    void* y;
    long long y_ld;
    int M, N, K;
};

/* r4d_gg_pf_gemm's answer for a call no form takes (a block type other than q5_K / q6_K / q8_0, a shape the forms
 * do not cover): the caller runs the stock kernel. */
constexpr int R4D_GG_PF_DECLINE = 1;

/* Nonzero while a form is switched on (pinned, or gg_pf != 0). */
int r4d_gg_pf_on();
/* Launches the chosen form on `s`: RAD_OK, a RAD_E_* launch failure, or R4D_GG_PF_DECLINE. */
int r4d_gg_pf_gemm(RadStream s, const R4dGgPfCall& c);
/* The runtime switch gg_pf: 0 off, 1 the rule (the per-shape table, RADIANCE_DS41_GG_PF_TABLE or the built-in one),
 * 2 the rule with the table's row-ranged entries (lever A11, "type:NxK@LO-HI=form": a launch of LO..HI rows takes that
 * form, 0 = the stock kernel; other launches as 1), or a form code (r4d_gg_pf_known) pinned for every shape. Returns 1
 * when applied, -1 when the value is unknown. */
int r4d_gg_pf_set(int v);
int r4d_gg_pf_get();
/* Whether `v` is a value gg_pf takes. */
int r4d_gg_pf_known(int v);

#endif
