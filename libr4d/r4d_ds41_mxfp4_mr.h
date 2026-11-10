/* r4d_ds41_mxfp4_mr.h -- drafter_mx: the entry of moe_gemm_ggml type=mxfp4's multi-row forms
 * (r4d_ds41_mxfp4_mr.hip), called by mx_moe (r4d_ds41_mxfp4.hip) ahead of the stock kernels. The DeepSeek-V4.1
 * DSpark drafter's routed experts are the only MXFP4 grouped GEMM the 2.25 bpw line runs.
 *
 * Off unless switched on: the runtime switch drafter_mx (libr4d's rad_fn4_* hooks, RADIANCE_DS41_DRAFTER_MX at
 * load) or the bench pin r4d_mx_mr_pin. Off, mx_moe runs today's kernels untouched. */
#ifndef R4D_DS41_MXFP4_MR_H
#define R4D_DS41_MXFP4_MR_H

#include <cstdint>

/* One grouped GEMM as mx_moe resolved it: y[i] = a[row(i)] . w[e(i)]^T over the T expert-sorted rows, expert e's
 * rows [off[e], off[e + 1]) (the first from 0, the last to T), row(i) the token (a_order token: sorted[i] / top_k)
 * or the slot (a_order sorted), a dropped slot (sorted < 0) or a token past AR a row of zeros. The experts are
 * a table of device pointers (wtab) or stacked w_es bytes apart (w), MXFP4 blocks, w_ld bytes a row.
 * `gemv`: the stock path would run the gemv band's kernel (else the tiled kernel); the form keeps that path's
 * summation order. */
struct R4dMxMrCall {
    const uint16_t* x;
    long long x_ld;
    int x_al;
    int AR;
    const uint8_t* w;
    const uint8_t* const* wtab;
    long long w_ld, w_es;
    uint16_t* y;
    long long y_ld;
    const int32_t* sorted;
    const int32_t* off;
    int T, N, K;
    int n_expert, top_k, a_sorted;
    int tokens;
    bool gemv;
};

/* r4d_mx_mr_moe's answer for a call the forms do not take: the caller runs the stock kernels. */
constexpr int R4D_MX_MR_DECLINE = 1;

/* Nonzero while the forms are switched on (the switch or a pin). */
int r4d_mx_mr_on();
/* Launches a form on `s`: RAD_OK, a RAD_E_* launch failure, or R4D_MX_MR_DECLINE. */
int r4d_mx_mr_moe(RadStream s, const R4dMxMrCall& c);
/* The runtime switch (drafter_mx): 0 off, 1 on. */
void r4d_mx_mr_set(int on);
int r4d_mx_mr_get();

#endif
