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

/* Lever K1 folds a / e (rmsnorm_gemm_nt_ggml, swiglu_clamped_gemm_nt_ggml): the activation's producer run in the
 * form's staging. The call's x is the producer's input: pro 1 rmsnorm's x [M, K] (gain g [K] f32, eps, wadd, mult,
 * nw = the waves of rmsnorm's workgroup a row at this M, r4d_act_threads(M, K) / 32); pro 2 swiglu_clamped's gate_up
 * [M, 2K] (limit; u_al: the up half's rows 16-byte aligned). h [M, K] bf16 (h_ld elements a row, h_al: 16-byte
 * aligned rows) receives the producer's output, the GEMM reads its values: y and h are the bytes of the producer's
 * launch then gemm_nt_ggml's. */
struct R4dGgMrFold {
    int pro;
    const float* g;
    float eps, wadd, mult;
    int nw;
    float limit;
    uint16_t* h;
    long long h_ld;
    int h_al, u_al;
};
/* RAD_OK, a launch failure, or R4D_GG_MR_DECLINE (the caller issues the producer and gemm_nt_ggml). Does not read
 * the runtime switch: the caller does (r4d_gg_mr_on). */
int r4d_gg_mr_fold(RadStream s, const R4dGgMrCall& c, const R4dGgMrFold& f);

/* Lever 1U-GQ (ds41-int11, gemm_nt_ggml_pair): calls a and b (the same block type, K and M, bf16 out) in ONE launch of
 * the form, each call's rows in its own blocks of the grid, each output the form's operations: the two launches'
 * bytes. RAD_OK, a launch failure, or R4D_GG_MR_DECLINE (the caller issues the two launches). Does not read the
 * runtime switch: the caller does (r4d_gg_mr_on). */
int r4d_gg_mr_pair(RadStream s, const R4dGgMrCall& a, const R4dGgMrCall& b);

/* Lever A4 (ds41-a4): the fn4 switch gg_gemv_m16 (0 / 1): under D2, a decode-band call of M <= 16 rows takes the A4
 * table's form (r4d_ds41_ggml_mr.hip, mr_choose); 0 = D2's pick everywhere. */
void r4d_gg_mr_m16_set(int v);
int r4d_gg_mr_m16_get();

#endif
