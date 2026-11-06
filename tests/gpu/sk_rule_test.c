/* sk_rule_test.c -- kernels/exl3r4d/exl3r4d_sk.h on the host (label host; ds41 exl3-splitk, 2026-11-06).
 *
 *   1. ORACLE: without ep_ranks (absent = 1, or 0) exl3r4d_sk_choose is launch.c's inline rule before E1 / E2
 *      (radiance-exl3 f2f3c3f7 kernels/exl3r4d/launch.c:237-253, copied below verbatim but for the operand names) for
 *      every T 1..96, top_k {1,2,6,8,10}, n_expert {2..512}, y_cols {128..5120}, n_class {1,2,4}, h_rot {0,1}, gate_up
 *      and down, and a resident table of {0, 16, 64, 128} -- the same sk and the same resident queries.
 *   2. DEEPSEEK: ep_ranks 4, the R9700 resident model (wq_capacity's rule over the shipped code objects' VGPR counts:
 *      gate_up 180 / 178 VGPRs, down 92 / 96 at K 2 / 2.5; 32 WGPs) -- the E1 / E2 table at 1..8 tokens, checked.
 *   3. NUMERICS (a model of the summation order only, not the device's WMMA): one output of a K-long dot product of
 *      f16-valued inputs, fp32 k-steps of 16 products, split-K s partial chains joined in wave order as
 *      splitk_reduce joins them (acc_0 += acc_1, += acc_2, ...), against the fp64 sum: every split's error inside the
 *      recursive-summation bound of its own chain, and split-K 4's of the same order as the shipped split-K 2's.
 *
 *   cc -std=c11 -O2 -Wall -Wextra -ffp-contract=off -I kernels/exl3r4d tests/gpu/sk_rule_test.c -lm && ./a.out [-v]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "exl3r4d_sk.h"

static int fails = 0, checks = 0;
#define CHECK(c, ...)                                     \
    do {                                                  \
        ++checks;                                         \
        if (!(c)) {                                       \
            ++fails;                                      \
            if (fails <= 20) {                            \
                fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                fprintf(stderr, __VA_ARGS__);             \
                fputc('\n', stderr);                      \
            }                                             \
        }                                                 \
    } while (0)

/* ------------------------------------------------------------------ 1. the oracle */
typedef struct Tab {
    int gu[9], dn[9];   /* resident workgroups at split-K 1..8 (0: unknown) */
    int calls_gu, calls_dn;
    char log[256];
    int nlog;
} Tab;
static int tab_res(void* ctx, int gate_up, int sk) {
    Tab* t = (Tab*)ctx;
    if (sk < 1 || sk > 8) return 0;
    if (t->nlog < (int)sizeof t->log - 2) { t->log[t->nlog++] = (char)(gate_up ? 'g' : 'd'); t->log[t->nlog++] = (char)('0' + sk); }
    if (gate_up) { ++t->calls_gu; return t->gu[sk]; }
    ++t->calls_dn;
    return t->dn[sk];
}

typedef struct G { int64_t y_cols, n_expert, top_k; int n_class, h_rot, bits; } G;

/* launch.c's rule as it stood at f2f3c3f7 (fill(): d->sk), exl3r4d_gu_resident(...) read through the table. */
static int old_rule(int gate_up, int64_t T, const G* g, Tab* t) {
    int sk;
    sk = gate_up ? EXL3R4D_SK : EXL3R4D_SK_DN;
    if (gate_up && EXL3R4D_SK_SMALL_M > 0 && (T + g->top_k - 1) / g->top_k <= EXL3R4D_SK_SMALL_M) sk = EXL3R4D_SK_SMALL_V;
    if (gate_up && EXL3R4D_SK_ONE_V > 0 && (T + g->top_k - 1) / g->top_k <= 1) sk = EXL3R4D_SK_ONE_V;
    if (gate_up && EXL3R4D_SK_FIT && sk > 1) {
        const int64_t units = (g->y_cols + 127) / 128 * (T < g->n_expert ? T : g->n_expert);
        const int fit = tab_res(t, 1, sk);   /* exl3r4d_gu_resident(g->bits, EXL3R4D_NPW, d->sk, g->h_rot) */
        if (fit > 0 && units > fit) sk = 1;
    }
#if EXL3R4D_SK4_M > 0
    if (gate_up && g->n_class == 4 && (T + g->top_k - 1) / g->top_k <= EXL3R4D_SK4_M) {
        const int64_t units = (g->y_cols + 127) / 128 * (T < g->n_expert ? T : g->n_expert);
        const int fit = tab_res(t, 1, 4);
        if (fit > 0 && units <= fit) sk = 4;
    }
#endif
    if (!gate_up && EXL3R4D_SK_DN_SMALL_M > 0 && (T + g->top_k - 1) / g->top_k <= EXL3R4D_SK_DN_SMALL_M) sk = 2;
    return sk;
}

static int new_rule(int gate_up, int64_t T, const G* g, int ep, int64_t kgcd, Tab* t) {
    Exl3r4dSkIn si;
    memset(&si, 0, sizeof si);
    si.gate_up = gate_up;
    si.n_class = g->n_class;
    si.h_rot = g->h_rot;
    si.ep_ranks = ep;
    si.T = T;
    si.top_k = g->top_k;
    si.n_expert = g->n_expert;
    si.y_cols = g->y_cols;
    si.ksteps_gcd = kgcd;
    return exl3r4d_sk_choose(&si, tab_res, t);
}

static void oracle(void) {
    static const int64_t topks[] = { 1, 2, 6, 8, 10 };
    static const int64_t nexps[] = { 2, 3, 8, 16, 64, 96, 128, 256, 384, 512 };
    static const int64_t ycols[] = { 128, 256, 384, 1024, 2304, 2560, 4096, 5120 };
    static const int ress[] = { 0, 16, 64, 128 };
    static const int ncls[] = { 1, 2, 4 };
    long n = 0;
    for (int gate_up = 0; gate_up < 2; ++gate_up)
        for (size_t a = 0; a < sizeof topks / sizeof *topks; ++a)
            for (size_t b = 0; b < sizeof nexps / sizeof *nexps; ++b)
                for (size_t c = 0; c < sizeof ycols / sizeof *ycols; ++c)
                    for (size_t r = 0; r < sizeof ress / sizeof *ress; ++r)
                        for (size_t q = 0; q < sizeof ncls / sizeof *ncls; ++q)
                            for (int h = 0; h < 2; ++h)
                                for (int64_t T = 1; T <= 96; ++T)
                                    for (int ep = 0; ep <= 1; ++ep) {
                                        const G g = { ycols[c], nexps[b], topks[a], ncls[q], h, 2 };
                                        Tab t1, t2;
                                        memset(&t1, 0, sizeof t1);
                                        for (int s = 1; s <= 8; ++s) t1.gu[s] = t1.dn[s] = ress[r] * (s == 1 ? 2 : 1);
                                        t2 = t1;
                                        const int o = old_rule(gate_up, T, &g, &t1);
                                        const int m = new_rule(gate_up, T, &g, ep, 8 * 40, &t2);
                                        CHECK(o == m, "oracle: gate_up %d T %lld top_k %lld n_expert %lld y_cols %lld res %d "
                                              "n_class %d h_rot %d ep %d: old sk %d, new %d", gate_up, (long long)T,
                                              (long long)g.top_k, (long long)g.n_expert, (long long)g.y_cols, ress[r],
                                              g.n_class, h, ep, o, m);
                                        CHECK(t1.nlog == t2.nlog && !memcmp(t1.log, t2.log, (size_t)t1.nlog),
                                              "oracle: resident queries differ (gate_up %d T %lld)", gate_up, (long long)T);
                                        ++n;
                                    }
    printf("1. oracle: %ld launches without ep_ranks, new rule == f2f3c3f7's inline rule (sk and resident queries)\n", n);
}

/* ------------------------------------------------------------------ 2. the DeepSeek table */
/* wq_capacity (exl3r4d.hip) over the shipped gfx1201 code objects: VGPRs rounded to 8, waves a SIMD
 * min(16, 1536 / vgpr), 4 SIMDs a WGP, LDS 128 KiB a WGP; x 32 WGPs. */
static int wq_model(int vgpr, int threads, int smem) {
    const int regs = (vgpr + 7) / 8 * 8, wps = 1536 / regs < 16 ? 1536 / regs : 16;
    const int by_vgpr = wps * 4 / (threads / 32);
    const int by_lds = (128 * 1024) / (smem + 140);   /* + the kernels' 140 B static __shared__ */
    return (by_vgpr < by_lds ? by_vgpr : by_lds) * 32;
}
/* exl3r4d_gemm_lds: max(MATS * 8 KiB epilogue tiles, wv * npw * (sk - 1) * 1 KiB), gate_up h_rot >= 20 KiB */
static int lds_model(int gate_up, int sk) {
    const int mats = gate_up ? 2 : 1, wv = 8 * mats / 2;
    int need = mats * 8 * 256 * 4, red = wv * 2 * (sk - 1) * 256 * 4;
    need = red > need ? red : need;
    if (gate_up && need < 16384 + 4096) need = 16384 + 4096;
    return need;
}

static void deepseek(int verbose) {
    /* the shipped exl3r4d.so (f2f3c3f7): gemm_kernel<2,2,2> 180, <21,2,2> 178, <2,2,1> 92, <21,2,1> 96 VGPRs */
    static const struct { const char* k; int vgu, vdn; } bitsv[] = { { "K 2", 180, 92 }, { "K 2.5", 178, 96 } };
    printf("2. DeepSeek V4.1 Flash, TP4 / EP4 (96 experts a rank, top-6), R9700 resident model (32 WGPs)\n");
    for (size_t bi = 0; bi < 2; ++bi) {
        Tab t;
        memset(&t, 0, sizeof t);
        for (int s = 1; s <= 8; s *= 2) {
            t.gu[s] = (16 / 2) * s * 32 <= 1024 ? wq_model(bitsv[bi].vgu, 8 * s * 32, lds_model(1, s)) : 0;
            t.dn[s] = 4 * s * 32 <= 1024 ? wq_model(bitsv[bi].vdn, 4 * s * 32, lds_model(0, s)) : 0;
        }
        printf("   %s resident: gate_up sk1 %d sk2 %d | down sk1 %d sk2 %d sk4 %d\n", bitsv[bi].k, t.gu[1], t.gu[2],
               t.dn[1], t.dn[2], t.dn[4]);
        CHECK(t.gu[2] == 64, "gate_up split-K 2 resident %d (launch.c's comment: 64 on a 32-WGP card)", t.gu[2]);
        /*                      tokens: 1  2  3  4  5  6  7  8 */
        static const int want_gu[8] = { 2, 1, 1, 1, 1, 1, 1, 1 };
        static const int want_dn[8] = { 4, 2, 1, 1, 1, 1, 1, 1 };
        static const int base_gu[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        static const int base_dn[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        const G gu = { 2304, 96, 6, 2, 1, 2 }, dn = { 5120, 96, 6, 2, 1, 2 };
        printf("   %s tokens:            1  2  3  4  5  6  7  8\n", bitsv[bi].k);
        int row[4][8];
        for (int tok = 1; tok <= 8; ++tok) {
            const int64_t T = 6 * tok;
            row[0][tok - 1] = new_rule(1, T, &gu, 1, 5120 / 16, &t);
            row[1][tok - 1] = new_rule(1, T, &gu, 4, 5120 / 16, &t);
            row[2][tok - 1] = new_rule(0, T, &dn, 1, 2304 / 16, &t);
            row[3][tok - 1] = new_rule(0, T, &dn, 4, 2304 / 16, &t);
            CHECK(row[0][tok - 1] == base_gu[tok - 1], "gate_up base tok %d sk %d", tok, row[0][tok - 1]);
            CHECK(row[1][tok - 1] == want_gu[tok - 1], "gate_up E2 tok %d sk %d want %d", tok, row[1][tok - 1], want_gu[tok - 1]);
            CHECK(row[2][tok - 1] == base_dn[tok - 1], "down base tok %d sk %d", tok, row[2][tok - 1]);
            CHECK(row[3][tok - 1] == want_dn[tok - 1], "down E1 tok %d sk %d want %d", tok, row[3][tok - 1], want_dn[tok - 1]);
        }
        static const char* names[4] = { "gate_up, no ep_ranks", "gate_up, ep_ranks 4 ", "down, no ep_ranks   ",
                                         "down, ep_ranks 4    " };
        for (int r = 0; r < 4; ++r) {
            printf("     %s ", names[r]);
            for (int i = 0; i < 8; ++i) printf(" %d", row[r][i]);
            printf("\n");
        }
        if (verbose)
            for (int tok = 1; tok <= 4; ++tok) {
                const int64_t T = 6 * tok, l2 = (2 * T + 3) / 4 < T ? (2 * T + 3) / 4 : T;
                printf("     tokens %d: live counted %lld (rule above: %lld); gate_up units %lld, down units %lld\n", tok,
                       (long long)l2, (long long)T, (long long)(18 * l2), (long long)(40 * l2));
            }
    }
    /* unknown residency (a tool without a device): E2 keeps split-K 2 wherever SK_SMALL_M gives it, E1 its start */
    {
        Tab t;
        memset(&t, 0, sizeof t);
        const G gu = { 2304, 96, 6, 2, 1, 2 }, dn = { 5120, 96, 6, 2, 1, 2 };
        CHECK(new_rule(1, 6, &gu, 4, 320, &t) == 2 && new_rule(1, 12, &gu, 4, 320, &t) == 2, "unknown residency, gate_up");
        CHECK(new_rule(0, 6, &dn, 4, 144, &t) == 4 && new_rule(0, 24, &dn, 4, 144, &t) == 2 &&
              new_rule(0, 30, &dn, 4, 144, &t) == 1, "unknown residency, down");
        /* a k-step count split-K 4 does not divide falls to 2 (and to 1 below that) */
        CHECK(new_rule(0, 6, &dn, 4, 6, &t) == 2 && new_rule(0, 6, &dn, 4, 3, &t) == 1, "k-step divisibility");
        /* h_rot 0: E1 does not apply (down's P1c form, another occupancy) */
        const G dn0 = { 5120, 96, 6, 2, 0, 2 };
        CHECK(new_rule(0, 6, &dn0, 4, 144, &t) == 1, "E1 needs h_rot");
        /* ep_ranks 2 (the margin then counts every row: 2 T / 2 = T) leaves gate_up as without it */
        Tab r;
        memset(&r, 0, sizeof r);
        r.gu[2] = 64;
        CHECK(new_rule(1, 6, &gu, 2, 320, &r) == 1, "ep 2 counts T rows: 108 > 64 -> 1");
    }
}

/* ------------------------------------------------------------------ 3. numerics (summation order model) */
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double urand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (double)(rng >> 11) * (1.0 / 9007199254740992.0);
}
static double nrand(void) {
    double u = urand(), v = urand();
    if (u < 1e-300) u = 1e-300;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}
/* round to an f16 value (11 significant bits; the inputs here stay in f16's normal range) */
static float f16r(double x) {
    if (x == 0.0) return 0.0f;
    int e;
    const double m = frexp(x, &e);
    return (float)ldexp(nearbyint(m * 2048.0) / 2048.0, e);
}

/* One output: fp32 chains of 16-product k-steps over [k0, k1), each k-step's 16 products summed in fp32 and added to
 * the chain (a model: the device's WMMA sums inside an instruction in its own order). */
static float chain(const float* a, const float* w, int k0, int k1) {
    float acc = 0.0f;
    for (int k = k0; k < k1; k += 16) {
        float st = 0.0f;
        for (int i = 0; i < 16; ++i) st = st + a[k + i] * w[k + i];   /* f16 x f16 is exact in f32 */
        acc = acc + st;
    }
    return acc;
}
static float splitk(const float* a, const float* w, int K, int sk) {
    const int kps = K / 16 / sk;
    float part[8];
    for (int s = 0; s < sk; ++s) part[s] = chain(a, w, s * kps * 16, (s + 1) * kps * 16);
    float acc = part[0];
    for (int s = 1; s < sk; ++s) acc += part[s];   /* splitk_reduce: ksplit 0 adds ksplit 1, 2, ... in order */
    return acc;
}

/* bf16 (down's y) round to nearest even, as f32_to_bf16 */
static float bf16r(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

static void numerics(void) {
    static const int Ks[2] = { 5120, 2304 };
    const double u = ldexp(1.0, -24);
    printf("3. numerics model (summation order only): one output of K f16 products, fp32 k-steps of 16, split-K s\n"
           "   chains joined in wave order, against fp64; 4000 outputs a K, a ~ N(0,1), w ~ N(0,0.02), both f16\n");
    for (int ki = 0; ki < 2; ++ki) {
        const int K = Ks[ki], n = 4000;
        float* a = (float*)malloc(sizeof(float) * (size_t)K);
        float* w = (float*)malloc(sizeof(float) * (size_t)K);
        double maxrel[4] = { 0 }, sumsq[4] = { 0 }, maxbnd[4] = { 0 }, maxd12 = 0, maxd14 = 0;
        int cnt_diff2 = 0, cnt_diff4 = 0, bf_diff2 = 0, bf_diff4 = 0;
        for (int o = 0; o < n; ++o) {
            for (int k = 0; k < K; ++k) { a[k] = f16r(nrand()); w[k] = f16r(0.02 * nrand()); }
            double exact = 0.0, absum = 0.0;
            for (int k = 0; k < K; ++k) { exact += (double)a[k] * (double)w[k]; absum += fabs((double)a[k] * (double)w[k]); }
            float r[3];
            for (int si = 0; si < 3; ++si) {
                const int sk = 1 << si;
                r[si] = splitk(a, w, K, sk);
                const double err = fabs((double)r[si] - exact);
                /* recursive summation: 16 - 1 adds a k-step, K / 16 / sk - 1 a chain, sk - 1 joins, + 1 */
                const double bound = (double)(15 + K / 16 / sk + sk) * u * absum;
                CHECK(err <= bound, "K %d sk %d: error %.3g past its chain bound %.3g", K, sk, err, bound);
                const double rel = err / absum;   /* relative to sum |a w|, the bound's scale */
                if (rel > maxrel[si]) maxrel[si] = rel;
                if (bound / absum > maxbnd[si]) maxbnd[si] = bound / absum;
                sumsq[si] += rel * rel;
            }
            const double d12 = fabs((double)r[1] - (double)r[0]) / absum, d14 = fabs((double)r[2] - (double)r[0]) / absum;
            if (d12 > maxd12) maxd12 = d12;
            if (d14 > maxd14) maxd14 = d14;
            cnt_diff2 += r[1] != r[0];
            cnt_diff4 += r[2] != r[0];
            bf_diff2 += bf16r(r[1]) != bf16r(r[0]);
            bf_diff4 += bf16r(r[2]) != bf16r(r[0]);
        }
        printf("   K %4d: |err| / sum|a w|  max  sk1 %.2e  sk2 %.2e  sk4 %.2e | rms  sk1 %.2e  sk2 %.2e  sk4 %.2e\n", K,
               maxrel[0], maxrel[1], maxrel[2], sqrt(sumsq[0] / n), sqrt(sumsq[1] / n), sqrt(sumsq[2] / n));
        printf("           bound (n-1)u: sk1 %.2e  sk2 %.2e  sk4 %.2e | outputs != sk1: sk2 %d/%d (max %.2e)  sk4 %d/%d (max %.2e)\n",
               maxbnd[0], maxbnd[1], maxbnd[2], cnt_diff2, n, maxd12, cnt_diff4, n, maxd14);
        printf("           rounded to bf16 (one ulp at the most): outputs != sk1's: sk2 %d/%d  sk4 %d/%d\n", bf_diff2, n,
               bf_diff4, n);
        /* split-K 4 is the same class of change as the shipped split-K 2: within 1.5x of its worst case */
        CHECK(maxrel[2] <= 1.5 * maxrel[1] + 1e-12, "K %d: split-K 4's worst error %.3g vs split-K 2's %.3g", K, maxrel[2], maxrel[1]);
        free(a);
        free(w);
    }
}

int main(int argc, char** argv) {
    const int verbose = argc > 1 && !strcmp(argv[1], "-v");
    oracle();
    deepseek(verbose);
    numerics();
    printf("%s: %d checks, %d failed\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
