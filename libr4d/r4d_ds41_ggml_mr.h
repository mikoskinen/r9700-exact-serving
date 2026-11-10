/* r4d_ds41_ggml_mr.h -- lever D2: the entry of the decode band's multi-row form (r4d_ds41_ggml_mr.hip),
 * called by gg_gemm (r4d_ds41_ggml.hip) for gemm_nt_ggml / logits_gemm_ggml at M <= kGgDecodeRows.
 *
 * The form is off unless switched on: r4d_gg_mr_pin (bench / test hook, exported) or, once merged with the
 * runtime switch table, gg_gemv = 2. Off, gg_gemm runs today's kernel untouched. */
#ifndef R4D_DS41_GGML_MR_H
#define R4D_DS41_GGML_MR_H

#include <cstdint>

/* One decode-band GEMM as gg_gemm resolved it: y[M, N] = x[M, K] . w[N, K]^T, the weight in the type's
 * native blocks (w_ld bytes a row), x bf16 (x_ld elements a row, x_al: 16-byte aligned rows), y bf16 or,
 * for logits_gemm_ggml, f32 (y_ld elements a row). `ty` is gg_gemm's GG_* code. */
struct R4dGgMrCall {
    int ty;
    bool f32out;
    const uint16_t* x;
    long long x_ld;
    int x_al;
    const uint8_t* w;
    long long w_ld;
    void* y;
    long long y_ld;
    int M, N, K;
};

/* r4d_gg_mr_gemm's answer for a call the form does not take (a block type other than q5_K / q6_K / q8_0,
 * M past 48, a shape whose LDS would not fit): the caller runs the stock kernel. */
constexpr int R4D_GG_MR_DECLINE = 1;

/* Nonzero while the form is switched on (pinned or the rule). */
int r4d_gg_mr_on();
/* Launches the form on `s`: RAD_OK, a RAD_E_* launch failure, or R4D_GG_MR_DECLINE. */
int r4d_gg_mr_gemm(RadStream s, const R4dGgMrCall& c);
/* The on/off state the runtime switch sets (0 off, 1 the form's rule); a pin overrides it. */
void r4d_gg_mr_set(int on);

#endif
