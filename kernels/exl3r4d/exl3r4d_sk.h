/* exl3r4d_sk.h -- the decode GEMM's split-K choice (launch.c fill: Exl3r4dGemm::sk), its compile-time knobs and the
 * rule as ONE pure function, exl3r4d_sk_choose, so it can be read and tested on the host (tests/gpu/sk_rule_test.c,
 * label host). Plain C: no device code, no radiance types, no environment (REQUIREMENTS R6). The knobs below moved
 * here from launch.c unchanged (ds41 exl3-splitk, 2026-11-06); E1 and E2 at the end are new.
 *
 * Split-K sk: a workgroup's waves split the K loop sk ways (wave = col_local * sk + ksplit, exl3r4d.hip moe_pass) and
 * the partial sums meet through LDS in wave order (exl3r4d_common.hip.h splitk_reduce: acc_0 + acc_1 + ... + acc_{sk-1},
 * deterministic). sk only reorders the fp32 sums of one output; the weights, the activation format and the rotations
 * are untouched. Every K is a multiple of 128 (exl3_geom_parse), so K / 16 k-steps divide by 1, 2, 4 and 8. */
#ifndef EXL3R4D_SK_H
#define EXL3R4D_SK_H

#include <stdint.h>

/* Decode defaults, from Dylan's measured choice (exl3-moe hip.py:594-601 and FACTS-exl3 §2.4): two column tiles
 * a wave, LDS split-K 2 ("measured best at M 1..5, 384 and 256 wide"). Re-taken on this card (runs/shape2, prep
 * path): within 4% of the best of NPW {2,4} x SK {1,2,4,8} at M=1 for both ops; NPW 4 / SK 1 is 5.7% faster at
 * gate_up M=16 and NPW 2 / SK 1 5.2% at down M=16 -- tunables for rad-tune (PLAN P7), not a hand-picked change. */
#ifndef EXL3R4D_NPW
#define EXL3R4D_NPW 2
#endif
#ifndef EXL3R4D_SK
#define EXL3R4D_SK 1  /* adopted (round 2, OCC-gu): gate_up split-K 1 -- tp2 M=4 -6.8%, M=1 +3.9% */
#endif
/* Per op (round 2, D.8 retune): down's own column tiles a wave and split-K; default the shared ones. */
#ifndef EXL3R4D_NPW_DN
#define EXL3R4D_NPW_DN EXL3R4D_NPW
#endif
/* gate_up's LDS split-K at FEW tokens (D16, deterministic: the two halves combined in a fixed order through LDS;
 * approved 2026-10-12): a launch of at most EXL3R4D_SK_SMALL_M tokens takes split-K 2 (round 2's OCC-gu: split-K 1 is
 * +3.9% at tp2 M=1). Tokens = sorted rows / top_k, so the engine's band-top M parameter does not matter. 0: off; 2
 * by default (engine v14). The
 * bits of those launches move (within the tolerance suites); the fused gate_up_down launch is split-K 1 only. */
#ifndef EXL3R4D_SK_SMALL_M
#define EXL3R4D_SK_SMALL_M 2
#endif
#ifndef EXL3R4D_SK_SMALL_V   /* the split-K those launches take */
#define EXL3R4D_SK_SMALL_V 2
#endif
/* L7 (EXL3R4D_SK_FIT): ...but only when every unit the grid can hold -- its column blocks times min(T, n_expert) live
 * experts -- is resident at once at split-K 2 (exl3r4d_gu_resident: the device's WGPs times the workgroups a WGP holds).
 * Split-K 2 doubles a workgroup's waves, so a WGP holds half as many (174 VGPRs: 2 sixteen-wave workgroups a WGP, 64
 * on a 32-WGP card), and a grid past that runs a second round of latency-bound workgroups. Reading (P9-L7 sB-1,
 * 0000:03:00.0, gate_up GEMM alone): tp1 M=2 = 5 blocks x 20 rows -> 100 > 64 -> split-K 2 66.3 us, split-K 1 59.8
 * (libr4d 62.6); tp2 M=2 (3 x 20 = 60) and M=1 at tp1 / tp2 (50 / 30) fit and keep split-K 2 (tp2 M=1 is +7 % at
 * split-K 1). Deterministic in the device, the geometry and the token count, never the routing. Measured (sD-1, ABAB
 * x3): chain tp1 M=2 104.8 -> 98.6 us (libr4d 103.2), MTP-like 2-token runs 93.0 -> 86.5 (libr4d 90.9); every other
 * shape within 0.07 us and its y byte-identical (tp1 M=1/3/4, tp2 M=1..4). NUMERICS (accepted by the orchestrator,
 * 2026-10-14, as a change of summation order only): a launch that leaves split-K 2 -- tp1 M=2 here -- sums each output
 * over all of K in one chain (split-K 1's fp32 order, the M >= 3 form) instead of two halves combined through LDS; the
 * quantised weights, the activation format and the rotation are untouched; the suites pass at their tolerances
 * (sG-1: device 946/0, real 32/0, tp 360/0, P2 cases 457/0, band tops 80/0). tp2 bits never move. 0: the token rule. */
#ifndef EXL3R4D_SK_FIT
#define EXL3R4D_SK_FIT 1
#endif
/* P10 L4 (EXL3R4D_SK4_M, reading; the four-class launches only, so tp4 alone): gate_up launches of at most this many
 * tokens take split-K 4 -- 32 waves a workgroup, one workgroup a WGP -- when every unit is resident at once. At tp4's
 * widths (128 / 256 a class) M=1 is 11-22 units on 32 WGPs, each a latency-bound 16-wave stream. 0: off.
 * REFUTED (P10 R8, 0000:13:00.0, ABAB x3, tp4 rank 0 split chain): M=1 36.00 -> 36.64 us (gate_up traced 18.20 ->
 * 18.68 at 1024 threads); the rule kept every other shape at its split-K. Twice the waves do not shorten a unit. */
#ifndef EXL3R4D_SK4_M
#define EXL3R4D_SK4_M 0
#endif
#ifndef EXL3R4D_SK_ONE_V     /* reading arm: another split-K for 1-token launches only (0: as above) */
#define EXL3R4D_SK_ONE_V 0
#endif
/* ...and down's, likewise (D16; round 2's D.8 chose down split-K 1 from M = 1 / 4 / 16 -- M = 2 never measured).
 * P9-L7 sM-1 (split-K 2 at every decode M; tp2's 24-k-step units are latency chains, two rounds): SLOWER everywhere --
 * down tp2 M=1..6 +1.2/+3.6/+3.9/+3.5/+3.4/+3.9 us, MTP-verify M=4 +2.4, tp1 M=1..4 +1.2..+3.9; chain tp2 M=1/2/3
 * +1.3/+4.0/+4.9. Halving each wave's chain does not pay for half the resident units. Off. */
#ifndef EXL3R4D_SK_DN_SMALL_M
#define EXL3R4D_SK_DN_SMALL_M 0
#endif
#ifndef EXL3R4D_SK_DN   /* adopted (round 2, D.8): down at split-K 1 -- tp2 M=1 -4.9%, M=4 -8.6%, M=16 -11.2% */
#define EXL3R4D_SK_DN 1
#endif

/* ---- E1 / E2 (ds41 exl3-splitk, 2026-11-06): the EP split-K rules. They apply only to a launch whose op carries
 * `ep_ranks` > 1 (EXL3 interface v9: the experts are whole and spread over that many ranks); without it every launch
 * takes the rules above, to the bit. MODEL NUMBERS, NOT MEASURED ON A CARD (~/gpu/ds41/exl3/PERF-REPORT.md §3):
 * DeepSeek V4.1 Flash at TP4 / EP4 (96 of 384 experts a rank, top-6), gate_up [4608, 5120] (18 column blocks, 320
 * k-steps), down [5120, 2304] (40 blocks, 144 k-steps), bits 2 and 2.5, R9700 32 WGPs. Under EP a launch's sorted rows
 * T = tokens * top_k, but only this rank's experts' rows are live: about T / ep_ranks on average, 2.8 of 6 on the
 * slowest rank at one token. The rules above count every row as live (min(T, n_expert) experts).
 *
 * E2 (EXL3R4D_SK_FIT_EP, gate_up): SK_FIT counts units = column blocks x min(T, n_expert, ceil(SK_FIT_EP * T /
 * ep_ranks)) -- this rank's share of the rows with a factor-2 margin for the routing's imbalance. One token: T = 6,
 * ep 4 -> 3 live -> 54 units <= 64 resident at split-K 2 (174-180 VGPRs, 2 sixteen-wave workgroups a WGP) -> split-K 2
 * kept, where the rule above counts 6 x 18 = 108 > 64 and drops to 1. Two tokens: T = 12 -> 6 -> 108 -> split-K 1 as
 * before (the slowest rank's load there is usually 4-5 experts). Model: gate_up ~74 -> ~40 us a layer when it fits;
 * P(the slowest rank holds >= 4 of 6 at one token) ~0.15, and then 72 units run 8 in a second round. 0: off.
 *
 * E1 (EXL3R4D_SK_DN_EP_M, down, h_rot only -- the input arrives rotated, so the kernel is the plain K loop whose
 * occupancy exl3r4d_dn_resident models): a launch of at most SK_DN_EP_M tokens starts at split-K SK_DN_EP_V1 (one
 * token) or SK_DN_EP_V (2..SK_DN_EP_M tokens) and halves until the split-K divides every class's k-steps and, with
 * EXL3R4D_SK_DN_EP_FIT, until units = column blocks x min(T, n_expert, ceil(SK_DN_EP_LOAD * T / ep_ranks)) are
 * resident at once (exl3r4d_dn_resident; unknown keeps it). DeepSeek's down K loop is 144 k-steps of ~0.30 us on the
 * latency chain (round 2's D.8 chose split-K 1 on Qwen's 20-24-step down, where halving the chain did not pay for half
 * the resident units; P9-L7 sM-1). Model: one token at split-K 4: 40 x 3 = 120 units <= 128 resident (92-96 VGPRs, 4
 * sixteen-wave workgroups a WGP), ~5 + 36 x 0.30 + 1 ~ 17 us against ~48 a layer; two tokens: 40 x 6 = 240 <= 256 at
 * split-K 2; three and four tokens: 360 / 480 > 256 -> split-K 1. 0: off. */
#ifndef EXL3R4D_SK_FIT_EP
#define EXL3R4D_SK_FIT_EP 2
#endif
#ifndef EXL3R4D_SK_DN_EP_M
#define EXL3R4D_SK_DN_EP_M 4
#endif
#ifndef EXL3R4D_SK_DN_EP_V1   /* E1: one token's starting split-K */
#define EXL3R4D_SK_DN_EP_V1 4
#endif
#ifndef EXL3R4D_SK_DN_EP_V    /* E1: 2..SK_DN_EP_M tokens' starting split-K */
#define EXL3R4D_SK_DN_EP_V 2
#endif
#ifndef EXL3R4D_SK_DN_EP_LOAD /* E1: the live-expert margin (as SK_FIT_EP); 0: count min(T, n_expert) */
#define EXL3R4D_SK_DN_EP_LOAD 2
#endif
#ifndef EXL3R4D_SK_DN_EP_FIT  /* E1: 1 halves the split-K until every unit is resident; 0 keeps the starting one
                               * (a GPU A/B arm: under the latency chain a second round may still beat split-K 1) */
#define EXL3R4D_SK_DN_EP_FIT 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Workgroups of the decode GEMM (gate_up 1 / down 0) at split-K sk resident at once on this device, 0 when unknown.
 * launch.c wraps exl3r4d_gu_resident / exl3r4d_dn_resident; a test passes a table. */
typedef int (*Exl3r4dResidentFn)(void* ctx, int gate_up, int sk);

typedef struct Exl3r4dSkIn {
    int     gate_up;      /* 1 moe_exl3_gate_up, 0 moe_exl3_down */
    int     n_class;      /* 1, 2, or 4 (the _mod4 forms) */
    int     h_rot;        /* the geometry's h_rot (down: its input arrives rotated) */
    int     ep_ranks;     /* v9 `ep_ranks`, 1 when absent */
    int64_t T;            /* the launch's sorted rows (rows of y) */
    int64_t top_k, n_expert;
    int64_t y_cols;       /* the output's width (gate_up: per matrix) */
    int64_t ksteps_gcd;   /* the gcd of every class's K / 16 (a split-K must divide every class's K loop) */
} Exl3r4dSkIn;

/* The live experts a rule counts: min(T, n_expert), and under EP with a margin `load` > 0 at most
 * ceil(load * T / ep_ranks). */
static inline int64_t exl3r4d_sk_live(const Exl3r4dSkIn* in, int load) {
    const int64_t all = in->T < in->n_expert ? in->T : in->n_expert;
    if (in->ep_ranks <= 1 || load <= 0) return all;
    const int64_t share = ((int64_t)load * in->T + in->ep_ranks - 1) / in->ep_ranks;
    return share < all ? share : all;
}

/* The split-K of one decode launch. Deterministic in the geometry, the row count and the device (through `res`),
 * never the routing. Without ep_ranks (<= 1) it is launch.c's inline rule before E1 / E2, line for line. */
static inline int exl3r4d_sk_choose(const Exl3r4dSkIn* in, Exl3r4dResidentFn res, void* ctx) {
    const int gate_up = in->gate_up;
    const int64_t tok = in->top_k > 0 ? (in->T + in->top_k - 1) / in->top_k : in->T;
    const int64_t ncb = (in->y_cols + 127) / 128;
    int sk = gate_up ? EXL3R4D_SK : EXL3R4D_SK_DN;
    if (gate_up && EXL3R4D_SK_SMALL_M > 0 && tok <= EXL3R4D_SK_SMALL_M) sk = EXL3R4D_SK_SMALL_V;
    if (gate_up && EXL3R4D_SK_ONE_V > 0 && tok <= 1) sk = EXL3R4D_SK_ONE_V;
    if (gate_up && EXL3R4D_SK_FIT && sk > 1) {
        const int64_t units = ncb * exl3r4d_sk_live(in, EXL3R4D_SK_FIT_EP);   /* E2: this rank's share under EP */
        const int fit = res ? res(ctx, 1, sk) : 0;
        if (fit > 0 && units > fit) sk = 1;
    }
#if EXL3R4D_SK4_M > 0   /* (a preprocessor branch: off, the rule compiles as it did) */
    if (gate_up && in->n_class == 4 && tok <= EXL3R4D_SK4_M) {
        const int64_t units = ncb * (in->T < in->n_expert ? in->T : in->n_expert);
        const int fit = res ? res(ctx, 1, 4) : 0;
        if (fit > 0 && units <= fit) sk = 4;
    }
#endif
    if (!gate_up && EXL3R4D_SK_DN_SMALL_M > 0 && tok <= EXL3R4D_SK_DN_SMALL_M) sk = 2;
    /* E1: down's split-K at few tokens under EP (h_rot: the plain K loop exl3r4d_dn_resident models) */
    if (!gate_up && in->ep_ranks > 1 && in->h_rot && EXL3R4D_SK_DN_EP_M > 0 && tok <= EXL3R4D_SK_DN_EP_M) {
        int s = tok <= 1 ? EXL3R4D_SK_DN_EP_V1 : EXL3R4D_SK_DN_EP_V;
        const int64_t units = ncb * exl3r4d_sk_live(in, EXL3R4D_SK_DN_EP_LOAD);
        while (s > 1) {
            const int fits_wg = (8 / EXL3R4D_NPW_DN) * s * 32 <= 1024;   /* the block exl3r4d_gemm_lds allows */
            if (fits_wg && in->ksteps_gcd > 0 && in->ksteps_gcd % s == 0) {
                if (!EXL3R4D_SK_DN_EP_FIT) break;
                const int fit = res ? res(ctx, 0, s) : 0;
                if (fit <= 0 || units <= fit) break;
            }
            s /= 2;
        }
        sk = s > 1 ? s : 1;
    }
    return sk;
}

#ifdef __cplusplus
}
#endif

#endif /* EXL3R4D_SK_H */
