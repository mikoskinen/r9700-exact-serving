/* tools/dbench/smp.cpp (DRAFTER track 21.11.2026 round 79, STATE Next 3 -> lever SNW) -- one card, through libr4d's own
 * exported entries (dlopen'd): the trunk's sampler chain under TP4 greedy (core/sample/sampler.cpp run(): every row
 * staged top_k 1, no flag, vocabulary split 4 ways: sample_topk -> all_gather -> sample_merge_topk -> sample_pick) at
 * the served candidate width (max_candidates 1024 a rank, merge of 4 x 1024 = 4096 pairs) vs a narrow width W (lever
 * SNW: an all-top-1 step needs one live candidate a rank; the gather moves whole 16-byte words, so W >= 2).
 *
 *  A  bytes: the narrow chain's token of every row == the served chain's == the host argmax over the four ranks'
 *     shards (the packed (key, ~id) rule: the largest logit, the lower global id on a tie); rows carry planted ties
 *     across and inside shards.
 *  B  time (warm: the head writes the logits right before the chain, 32320 x 4 B a row): event-timed p50 / p10 / p90 of
 *     ITERS reps, each kernel alone (topk on rank 1's shard with vocab_off 32320; merge over the gathered pairs the four
 *     ranks' topk wrote, copied into the row-interleaved layout untimed; pick), the three back to back (chain, no
 *     gather), and 20 back-to-back launches of each / 20 (card time without the event pair).
 *  The all_gather is not timed here (no P2P on this host): the bytes a rank moves are printed (served M x 1024 x 8 B to
 *  each of 3 peers vs M x W x 8 B).
 * Usage: smp LIBR4D [ITERS] [W]   (env SMP_REV: each M's narrow arm before the served one) */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_abi.h"
#include "rad_sample.h"

using Entry = int (*)(const RadArgs*, RadStream);

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }

static RadTensor T2(void* d, uint32_t dt, int64_t r, int64_t c, int64_t ld) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor T1(void* d, uint32_t dt, int64_t n) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }

struct Call {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
    void fix() { a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; }
};
static void pct(std::vector<double> v, double* p10, double* p50, double* p90) {
    std::sort(v.begin(), v.end());
    *p10 = v[v.size() / 10]; *p50 = v[v.size() / 2]; *p90 = v[9 * v.size() / 10];
}
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: smp LIBR4D [ITERS] [W]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const int WN = argc > 3 ? std::atoi(argv[3]) : 2;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto topk = (Entry)dlsym(lib, "r4d_rad_sample_topk");
    auto merge = (Entry)dlsym(lib, "r4d_rad_sample_merge_topk");
    auto pick = (Entry)dlsym(lib, "r4d_rad_sample_pick");
    if (!topk || !merge || !pick) { std::fprintf(stderr, "symbols missing\n"); return 2; }
    const int NV = 32320, WS = 4, NC = 1024, MAXM = 64;
    std::printf("smp: %s, n_vocab %d a rank x %d ranks, served width %d vs narrow %d, %d reps\n", argv[1], NV, WS, NC, WN, iters);
    /* the four ranks' logit shards [WS][MAXM][NV] */
    float* dlg; hc(hipMalloc(&dlg, (size_t)WS * MAXM * NV * 4), "lg");
    std::vector<float> hl((size_t)WS * MAXM * NV);
    for (auto& v : hl) v = uni(-12.0f, 18.0f);
    std::vector<int> want(MAXM);
    for (int m = 0; m < MAXM; ++m) {
        /* the row's winner: a planted peak; every fourth row a tie of two ids (across shards on odd rows, inside one
         * shard on even rows) at the peak, the lower global id must win */
        const float peak = 30.0f + (float)(m % 7);
        const int r0 = (int)(r32() % WS), c0 = (int)(r32() % NV);
        hl[((size_t)r0 * MAXM + m) * NV + c0] = peak;
        int g0 = r0 * NV + c0, win = g0;
        if (m % 4 == 0) {
            int r1 = (m % 8 == 0) ? r0 : (r0 + 1 + (int)(r32() % (WS - 1))) % WS;
            int c1 = (int)(r32() % NV);
            if (r1 == r0 && c1 == c0) c1 = (c0 + 1) % NV;
            hl[((size_t)r1 * MAXM + m) * NV + c1] = peak;
            win = std::min(g0, r1 * NV + c1);
        }
        want[m] = win;
    }
    hc(hipMemcpy(dlg, hl.data(), hl.size() * 4, hipMemcpyHostToDevice), "cp");
    RadSampleParams* dsp; hc(hipMalloc(&dsp, (size_t)MAXM * sizeof(RadSampleParams)), "sp");
    {
        std::vector<RadSampleParams> hp(MAXM);
        for (int m = 0; m < MAXM; ++m) {
            std::memset(&hp[m], 0, sizeof(hp[m]));
            hp[m].seed = 0x1234567ull + (uint64_t)m; hp[m].pos = 100 + (uint64_t)m;
            hp[m].top_k = 1; hp[m].mask_row = -1; hp[m].flags = 0;   /* sampler.cpp build_greedy / build_step under a split vocabulary */
        }
        hc(hipMemcpy(dsp, hp.data(), hp.size() * sizeof(RadSampleParams), hipMemcpyHostToDevice), "cp");
    }
    /* planes, per width: ci / cv [MAXM, NC] (the narrow arm uses their first W columns at pitch NC, as the engine would),
     * pairs [WS][MAXM][2W] tight, gathered [MAXM][WS * 2W] tight, tokens [MAXM] */
    int32_t* dci; float* dcv; hc(hipMalloc(&dci, (size_t)MAXM * NC * 4), "ci"); hc(hipMalloc(&dcv, (size_t)MAXM * NC * 4), "cv");
    uint32_t *dpr[2], *dgt[2]; int32_t* dtok[2];
    const int Wd[2] = { NC, WN };
    for (int w = 0; w < 2; ++w) {
        hc(hipMalloc(&dpr[w], (size_t)WS * MAXM * 2 * Wd[w] * 4), "pr");
        hc(hipMalloc(&dgt[w], (size_t)MAXM * WS * 2 * Wd[w] * 4), "gt");
        hc(hipMalloc(&dtok[w], (size_t)MAXM * 4), "tok");
    }
    hipStream_t st; hc(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1; hc(hipEventCreate(&e0), "ev"); hc(hipEventCreate(&e1), "ev");
    int bad = 0;
    std::printf("%4s %5s | %22s %22s %22s %22s | %8s %8s %8s | %10s %8s\n", "M", "width", "topk p50 (p10/p90) us", "merge p50 (p10/p90)",
                "pick p50 (p10/p90)", "chain3 p50 (p10/p90)", "topk x20", "merge x20", "pick x20", "gather B/peer", "tokens");
    const std::vector<int> Ms = { 8, 16, 24, 32, 48, 64 };
    for (int M : Ms) {
        std::vector<int32_t> tok[3];
        double res[3][5][3];
        const bool rev = std::getenv("SMP_REV") != nullptr;   /* round 79c: the narrow arms first (order check) */
        for (int wi = 0; wi < 3; ++wi) {
            /* arm 0 served, 1 narrow (topk at W too), 2 hybrid = the served topk + the narrow gather of its pair plane's first
             * W pairs a row (the gather's wider source pitch) + the narrow merge and pick (round 79e: r79b-d's narrow topk
             * costs +5.6 us at 48 rows, a launch effect of the tiny LDS, not the width) */
            const int w = rev ? 2 - wi : wi;
            const int W = Wd[w == 1 ? 1 : 0];    /* the topk's planes */
            const int Wm = Wd[w == 0 ? 0 : 1];   /* the merge's / pick's */
            uint32_t* pr = dpr[w == 1 ? 1 : 0];
            uint32_t* gt = dgt[w == 0 ? 0 : 1];
            Call ct[WS], cm, cp;
            for (int r = 0; r < WS; ++r) {
                ct[r].t = { T2(dlg + (size_t)r * MAXM * NV, RAD_F32, M, NV, NV), T1(dsp, RAD_U32, (int64_t)M * 24),
                            T2(dci, RAD_I32, M, W, NC), T2(dcv, RAD_F32, M, W, NC),
                            T2(pr + (size_t)r * MAXM * 2 * W, RAD_I32, M, 2 * W, 2 * W) };
                ct[r].p = { PI("n_vocab", NV), PI("vocab_off", (long long)r * NV) };
                ct[r].fix();
            }
            cm.t = { T2(gt, RAD_I32, M, (int64_t)WS * 2 * Wm, (int64_t)WS * 2 * Wm), T1(dsp, RAD_U32, (int64_t)M * 24),
                     T2(dci, RAD_I32, M, Wm, NC), T2(dcv, RAD_F32, M, Wm, NC) };
            cm.p = { PI("n_cand", Wm) };
            cm.fix();
            cp.t = { T2(dci, RAD_I32, M, Wm, NC), T2(dcv, RAD_F32, M, Wm, NC), T1(dsp, RAD_U32, (int64_t)M * 24), T1(dtok[w], RAD_I32, M) };
            cp.p = { PI("n_cand", Wm) };
            cp.fix();
            auto run = [&](Call& c, Entry f, const char* nm) {
                const int rc = f(&c.a, (RadStream)st);
                if (rc != 0) { std::fprintf(stderr, "%s rc %d at M %d W %d\n", nm, rc, M, W); std::exit(3); }
            };
            /* the four ranks' topk, then the gather's placement (rank r's row m at gathered row m, slot r) */
            auto gather = [&] {
                for (int r = 0; r < WS; ++r) run(ct[r], topk, "topk");
                for (int r = 0; r < WS; ++r)
                    hc(hipMemcpy2DAsync(gt + (size_t)r * 2 * Wm, (size_t)WS * 2 * Wm * 4, pr + (size_t)r * MAXM * 2 * W,
                                        (size_t)2 * W * 4, (size_t)2 * Wm * 4, (size_t)M, hipMemcpyDeviceToDevice, st), "g2d");
            };
            gather();
            run(cm, merge, "merge");
            run(cp, pick, "pick");
            hc(hipStreamSynchronize(st), "sync");
            tok[w].resize(M);
            hc(hipMemcpy(tok[w].data(), dtok[w], (size_t)M * 4, hipMemcpyDeviceToHost), "cp");
            auto timed = [&](int arm) {
                hc(hipEventRecord(e0, st), "rec");
                if (arm == 0) run(ct[1], topk, "topk");
                else if (arm == 1) run(cm, merge, "merge");
                else if (arm == 2) run(cp, pick, "pick");
                else { run(ct[1], topk, "topk"); run(cm, merge, "merge"); run(cp, pick, "pick"); }
                hc(hipEventRecord(e1, st), "rec");
                hc(hipEventSynchronize(e1), "sync");
                float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
                return 1000.0 * ms;
            };
            auto batch = [&](int arm) {
                hc(hipEventRecord(e0, st), "rec");
                for (int i = 0; i < 20; ++i) {
                    if (arm == 0) run(ct[1], topk, "topk");
                    else if (arm == 1) run(cm, merge, "merge");
                    else run(cp, pick, "pick");
                }
                hc(hipEventRecord(e1, st), "rec");
                hc(hipEventSynchronize(e1), "sync");
                float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
                return 1000.0 * ms / 20.0;
            };
            for (int i = 0; i < 10; ++i) for (int a = 0; a < 4; ++a) timed(a);
            std::vector<double> t[4], b[3];
            for (int i = 0; i < iters; ++i) {
                for (int k = 0; k < 4; ++k) { const int a = (i + k) % 4; t[a].push_back(timed(a)); }
                if (i % 4 == 0) for (int a = 0; a < 3; ++a) b[a].push_back(batch(a));
            }
            for (int a = 0; a < 4; ++a) pct(t[a], &res[w][a][0], &res[w][a][1], &res[w][a][2]);
            double d0, d2;
            for (int a = 0; a < 3; ++a) { double x; pct(b[a], &d0, &x, &d2); res[w][4][a] = x; }
        }
        long nd = 0, nh = 0;
        long ny = 0;
        for (int m = 0; m < M; ++m) { nd += tok[0][m] != tok[1][m]; nh += tok[0][m] != want[m]; ny += tok[0][m] != tok[2][m]; }
        bad += (nd || nh || ny) ? 1 : 0;
        for (int w = 0; w < 3; ++w)
            std::printf("%4d %5s | %8.2f (%5.2f/%6.2f) %8.2f (%5.2f/%6.2f) %8.2f (%5.2f/%6.2f) %8.2f (%5.2f/%6.2f) | %8.2f %8.2f %8.2f | %10lld %8s\n",
                        M, w == 0 ? "1024" : w == 1 ? "nar" : "hyb", res[w][0][1], res[w][0][0], res[w][0][2], res[w][1][1], res[w][1][0], res[w][1][2], res[w][2][1],
                        res[w][2][0], res[w][2][2], res[w][3][1], res[w][3][0], res[w][3][2], res[w][4][0], res[w][4][1], res[w][4][2],
                        (long long)M * 2 * Wd[w == 0 ? 0 : 1] * 4,
                        w == 1 ? (nd ? "DIFFER" : "equal") : w == 2 ? (ny ? "DIFFER" : "equal") : (nh ? "!= host" : "= host"));
    }
    std::printf("smp: %s\n", bad ? "TOKENS DIFFER" : "every M: the narrow and hybrid chains' tokens == the served chain's == the host argmax (ties: lower id)");
    return bad ? 1 : 0;
}
