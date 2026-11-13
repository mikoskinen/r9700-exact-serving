/* r4d_ds41_wmk.h -- lever WMK (ds41-wmk2): the part-order WMMA form (gg_wk_kernel, r4d_ds41_wmk.hip), called by gg_gemm_run,
 * gemm_nt_ggml_pair and the K1 folds a / e (r4d_ds41_ggml.hip) under fn4 ds41_wmk 1 (<= 48 rows) or the bench pin
 * r4d_gg_wk_pin (every M). Its own unit, so its code object loads only when a part-order launch is first issued:
 * with the switch off nothing of it reaches the card and r4d_ds41_ggml.hip's code object is int16's. */
#ifndef R4D_DS41_WMK_H
#define R4D_DS41_WMK_H

#include <cstdint>
#include "rad_types.h"

/* r4d_gg_wk_run's answer when the form does not take the call: the caller runs its other path. */
constexpr int R4D_WK_DECLINE = -77001;

/* The fold operands r4d_gg_wk_run takes for PRO 1 / 2 (lever WMK's folds). */
struct GgWkFold {
    int pro;
    const float* g;
    float eps, wadd, mult;
    int nw;
    float limit;
    uint16_t* h;
    long long h_ld;
    int h_al;
};

/* 1 or 2 weights (nm) of one GGUF type (gg_gemm's GG_* code) at one K over the same M rows, y = x . w^T, under launch code
 * `code` (the bench pin's value or lever WMK's kWmkCode; the codes are documented in r4d_ds41_wmk.hip). fo (nm 1 only): a
 * fold's producer in the staging. R4D_WK_DECLINE when the form does not take the call. */
int r4d_gg_wk_run(RadStream s, int code, int ty, bool f32out, int nm, const uint16_t* const* xd, const long long* x_ld,
                  const int* x_al, const uint8_t* const* wd, long long w_ld, void* const* yd, const long long* y_ld,
                  const long long* Nv, long long M, long long K, const GgWkFold* fo);

/* Lever WMK's prefill rows (G3): above the decode band under ds41_wmk 1, the part order on lever GPF's tile form `form` (the one
 * gg_pf would pick for the call; 0 or a form not built with the part fold = 322, the stock geometry). R4D_WK_DECLINE: not a
 * dense GGUF call the forms take (the caller runs today's prefill kernel). */
int r4d_gg_wkpf_run(RadStream s, int ty, bool f32out, const uint16_t* x, long long x_ld, int x_al, const uint8_t* w,
                    long long w_ld, void* y, long long y_ld, long long M, long long N, long long K, int form);

#endif  /* R4D_DS41_WMK_H */
