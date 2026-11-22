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

/* Lever SPD-F (ds41-etp-spd, 15.11.2026; SPEED-D, the AQL stamps of fu2-s1: at 17-48 rows the decode band's f32 projections
 * run the 8x1 tile -- every wave re-reads its 8 rows of x for ONE column: the router (router_gemm_f32_scored_topk_scatter,
 * N 384) 58 / 85 us a call at 32 / 47 rows, the compressor's kv | gate (gemm_nt_f32, N 1024) 268 / 375 us, both on card 0's
 * chain). The runtime switch "ds41_f32dec" (libr4d's rad_fn4_*, RADIANCE_DS41_F32DEC at load): 0 the stock rule (today),
 * 1 the decode band (rows <= 63) of both in the forms r4d_f32dec_*_form picks, 2 the router's only, 3 gemm_nt_f32's only.
 * tp8 loop D28 (19.11.2026): 4 .. 7 = 1 with the router's 17..48 rows in the 2-D wave map form 15 .. 18 (r4d_ds41_elem.hip).
 * Lever RGP (DRAFTER round 17): 8 = 1 with the router's 17..64 rows in form 19; lever RGK (DRAFTER round 39): 9 = form 21 at 1..64 rows;
 * lever RGM (DRAFTER round 40, RULE B: not the stock arithmetic): 10 = form 22 at 1..64 rows + gemm_nt_f32 code 40 at the router's shape;
 * 11 (lever RGC) = 10 + the compressors' kv | gate (N 1024) in code 40; 12 (lever RGD, DRAFTER round 41, RULE B for the drafter's
 * proposals only) = 11 + the DSpark drafter's router (128 experts: form 22 in its fused launch, code 40 for its gemm_nt_f32 at N 128
 * and its TP4 column split N 32).
 * Every form keeps the stock per-output arithmetic (lane l folds k = l, l + 32, .. ascending with fmaf, then the xor
 * butterfly: r4d_gemm_f32_tile.h / the GPF-b note above), so the bytes are the stock rule's at every form. */
int r4d_f32dec_get();
int r4d_f32dec_set(int v);    /* 1 applied, -1 an unknown value */
int r4d_f32dec_known(int v);
/* Lever DTKG (DRAFTER track round 110; rule B for the drafter's PROPOSALS only, as lever DTK): the DSpark drafter's router
 * (128 experts) selects its top KE of its declared top-k in process, in passes of at most ROWS rows (0 = every pass) -- carried
 * by fn4 drafter_mx's hundreds (rad_fn4_switch: drafter_mx = form + 100 KE + 1000 (ROWS / 8)). The slots KE .. k - 1 of each
 * row are written as dropped (id -1, weight 0: moe_scatter drops an id outside the slice, the gather skips a -1 inverse), the
 * kept slots' weights renormalised over the KE as a declared top-KE router's: the same ids, weights and live rows as lever
 * DTK's start-time top_k KE (r4d_selftest SPD_PART=dtk). code = 100 KE + 1000 G; 0 = off. */
int r4d_dtk_known(int code);
void r4d_dtk_set(int code);
int r4d_dtk_get();
/* the router kernels' k argument for a call: k, or k | (KE << 16) where the lever applies (n_expert 128, KE < k, m <= ROWS) */
int r4d_dtk_pack(int ne, int k, int m);
/* the form for a call: 0 = the stock rule; gemm forms 1..15 tile_wave (R, C, U) forms, 20.. the tiled forms above
 * (code = 20 + index into kF32Forms, any M); router forms 1.. (R, C, U, NW) of the fused launch */
int r4d_f32dec_gemm_form(int rows, int n);
int r4d_f32dec_rgs_form(int rows);
/* a tiled form (code 20..) at any M (the decode band's entry): RAD_OK, a launch failure or R4D_F32PF_DECLINE */
int r4d_f32pf_gemm_code(RadStream s, const R4dF32PfCall& c, int code);
/* Lever RGM (DRAFTER round 40, RULE B; r4d_ds41_elem.hip): ds41_f32dec 10 = the router's logits on bf16 WMMA in every pass --
 * the fused decode launch in form 22 (1..64 rows) and gemm_nt_f32 at the router's shape (N 384 or its 96-column split, K 5120,
 * bf16 weights) at any M in code 40; every other gemm_nt_f32 takes value 1's rule. The bench / test pin r4d_f32dec_pin(-1, 40)
 * puts every gemm_nt_f32 call code 40 can take on it at any M. */
int r4d_rgm_gemm(RadStream s, const R4dF32PfCall& c);
/* gemm_nt_f32's entry ahead of every other rule (r4d_f32pf_gemm calls it): code 40 when value 10 holds and the call is the
 * router's shape, or when the pin is 40; else R4D_F32PF_DECLINE */
int r4d_rgm_route(RadStream s, const R4dF32PfCall& c);

#endif
