/* tools/dbench/dph.cpp (DRAFTER track 21.11.2026 round 68, lever DPH) -- one card, through libr4d's own exported entries
 * (dlopen'd): the DSpark query pass's tail before the head (deepseek41.cpp: the last stage's post_f, d.contract, d.fnorm)
 * as today's three ops vs ONE mhc_post_collapse_rmsnorm (the op every stage boundary already runs as pcn_f / pcn_a):
 *
 *  A  bytes: r4d_rad_mhc_post (x, h, post, comb -> out), r4d_rad_mhc_collapse (out, pre -> xc), r4d_rad_rmsnorm (xc, w -> y)
 *     vs r4d_rad_mhc_post_collapse_rmsnorm (x, h, post, comb, out, pre, xc, w, y): out, xc and y byte-compared at every row
 *     count the engine runs (and more), on random rows, rows of large / tiny magnitudes, and rows holding -0.0 / inf / NaN.
 *     Each compare twice: mhc_post_collapse_rmsnorm as served at the stage boundaries (norm folded over r4d_act_threads(M, n)
 *     threads) and the new mhc_post_collapse_rmsnorm_l256 (rmsnorm's 256-lane partition; libr4d of round 68 on).
 *  C  batch invariance: each form's rows 0 .. k-1 at M = k against the same rows inside an M = 96 launch.
 *  B  time: each of the three back to back, the three in sequence, the fused one in both folds (the HIP launch path,
 *     event-timed, p50 of 7 x ITERS): an upper bound of the kernels, not of the served gaps (~4.2 us a dependent gap).
 * Usage: dph LIBR4D [ITERS] */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

using Entry = int (*)(const RadArgs*, RadStream);

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }
static uint16_t h_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

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
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadParam PF(const char* k, double v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

struct Call {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
    void fix() { a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; }
};
static double p50(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dph LIBR4D [ITERS]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto post = (Entry)dlsym(lib, "r4d_rad_mhc_post");
    auto collapse = (Entry)dlsym(lib, "r4d_rad_mhc_collapse");
    auto norm = (Entry)dlsym(lib, "r4d_rad_rmsnorm");
    auto pcn = (Entry)dlsym(lib, "r4d_rad_mhc_post_collapse_rmsnorm");
    auto pcnl = (Entry)dlsym(lib, "r4d_rad_mhc_post_collapse_rmsnorm_l256");   /* lever DPH: libr4d round 68 on */
    if (!post || !collapse || !norm || !pcn || !pcnl) {
        std::fprintf(stderr, "symbols missing (%p %p %p %p %p)\n", (void*)post, (void*)collapse, (void*)norm, (void*)pcn, (void*)pcnl);
        return 2;
    }
    std::printf("dph: %s\n", argv[1]);
    const int D = 5120, HC = 4, MAXM = 96;
    const double eps = 1e-6;   /* m.c.eps is read from the model; the kernels take any: the compare holds per eps */
    hipStream_t st; hc(hipStreamCreate(&st), "stream");
    hipEvent_t e0, e1; hc(hipEventCreate(&e0), "ev"); hc(hipEventCreate(&e1), "ev");
    uint16_t *dx, *dh, *doA, *doB, *dxcA, *dxcB, *dyA, *dyB, *doL, *dxcL, *dyL;
    float *dpo, *dcb, *dpre, *dw;
    const size_t nx = (size_t)MAXM * D, nh = nx * HC;
    hc(hipMalloc(&dx, nx * 2), "m"); hc(hipMalloc(&dh, nh * 2), "m");
    hc(hipMalloc(&doA, nh * 2), "m"); hc(hipMalloc(&doB, nh * 2), "m");
    hc(hipMalloc(&dxcA, nx * 2), "m"); hc(hipMalloc(&dxcB, nx * 2), "m");
    hc(hipMalloc(&dyA, nx * 2), "m"); hc(hipMalloc(&dyB, nx * 2), "m");
    hc(hipMalloc(&doL, nh * 2), "m"); hc(hipMalloc(&dxcL, nx * 2), "m"); hc(hipMalloc(&dyL, nx * 2), "m");
    hc(hipMalloc(&dpo, (size_t)MAXM * HC * 4), "m"); hc(hipMalloc(&dcb, (size_t)MAXM * HC * HC * 4), "m");
    hc(hipMalloc(&dpre, (size_t)MAXM * HC * 4), "m"); hc(hipMalloc(&dw, (size_t)D * 4), "m");

    /* inputs: kind 0 = typical (the served value ranges), 1 = per-row magnitudes from 2^-60 to 2^60, 2 = specials sprinkled */
    auto fill_inputs = [&](int kind) {
        std::vector<uint16_t> hx(nx), hh(nh);
        std::vector<float> hpo((size_t)MAXM * HC), hcb((size_t)MAXM * HC * HC), hpre((size_t)MAXM * HC), hw(D);
        for (int m = 0; m < MAXM; ++m) {
            const float sc = kind == 1 ? std::ldexp(1.0f, (int)(r32() % 121) - 60) : 1.0f;
            for (int i = 0; i < D; ++i) hx[(size_t)m * D + i] = h_bf16(uni(-3.0f, 3.0f) * sc);
            for (int i = 0; i < HC * D; ++i) hh[(size_t)m * HC * D + i] = h_bf16(uni(-8.0f, 8.0f) * sc);
            for (int k = 0; k < HC; ++k) { hpo[(size_t)m * HC + k] = uni(0.0f, 2.0f); hpre[(size_t)m * HC + k] = uni(0.0f, 1.0f); }
            for (int k = 0; k < HC * HC; ++k) hcb[(size_t)m * HC * HC + k] = uni(0.0f, 1.0f);
        }
        if (kind == 2) {
            const uint16_t sp[] = { 0x8000, 0x0000, 0x7f80, 0xff80, 0x7fc1, 0x0001, 0x8001, 0x7f7f };
            for (size_t i = 0; i < nx; i += 97) hx[i] = sp[r32() % 8];
            for (size_t i = 0; i < nh; i += 389) hh[i] = sp[r32() % 8];
            for (int m = 0; m < MAXM; m += 7) hx[(size_t)m * D + 5] = 0x8000;   /* -0.0 into rows */
            for (int m = 3; m < MAXM; m += 11)                                   /* a whole zero row */
                for (int i = 0; i < D; ++i) hx[(size_t)m * D + i] = 0;
        }
        for (int i = 0; i < D; ++i) hw[i] = uni(0.5f, 1.5f);
        hc(hipMemcpy(dx, hx.data(), nx * 2, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dh, hh.data(), nh * 2, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dpo, hpo.data(), hpo.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dcb, hcb.data(), hcb.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dpre, hpre.data(), hpre.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice), "cp");
    };
    auto calls = [&](int M, Call& cp, Call& cc, Call& cn, Call& cf, Call& cl) {
        cp.t = { T2(dx, RAD_BF16, M, D, D), T2(dh, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D), T2(dpo, RAD_F32, M, HC, HC),
                 T2(dcb, RAD_F32, M, HC * HC, HC * HC), T2(doA, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D) };
        cp.p = { PI("M", M), PI("n", D), PI("hc", HC), PS("dtype", "bf16") }; cp.fix();
        cc.t = { T2(doA, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D), T2(dpre, RAD_F32, M, HC, HC), T2(dxcA, RAD_BF16, M, D, D) };
        cc.p = { PI("M", M), PI("n", D), PI("hc", HC), PS("dtype", "bf16") }; cc.fix();
        cn.t = { T2(dxcA, RAD_BF16, M, D, D), T1(dw, RAD_F32, D), T2(dyA, RAD_BF16, M, D, D) };
        cn.p = { PI("M", M), PI("n", D), PF("eps", eps), PS("dtype", "bf16") }; cn.fix();
        cf.t = { T2(dx, RAD_BF16, M, D, D), T2(dh, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D), T2(dpo, RAD_F32, M, HC, HC),
                 T2(dcb, RAD_F32, M, HC * HC, HC * HC), T2(doB, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D),
                 T2(dpre, RAD_F32, M, HC, HC), T2(dxcB, RAD_BF16, M, D, D), T1(dw, RAD_F32, D), T2(dyB, RAD_BF16, M, D, D) };
        cf.p = { PI("M", M), PI("n", D), PI("hc", HC), PF("eps", eps), PS("dtype", "bf16") }; cf.fix();
        cl.t = cf.t; cl.t[4].data = doL; cl.t[6].data = dxcL; cl.t[8].data = dyL;
        cl.p = cf.p; cl.fix();
    };
    const int rows[] = { 1, 2, 3, 5, 8, 12, 16, 24, 30, 32, 33, 40, 48, 64, 80, 96 };
    int failsF = 0, failsL = 0;
    long tot_cmp = 0;
    auto get = [&](const uint16_t* d, size_t n) { std::vector<uint16_t> v(n); hc(hipMemcpy(v.data(), d, n * 2, hipMemcpyDeviceToHost), "cp"); return v; };
    auto ndiff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long k = 0; for (size_t i = 0; i < a.size(); ++i) k += a[i] != b[i]; return k; };
    for (int kind = 0; kind < 3; ++kind) {
        fill_inputs(kind);
        int fF = 0, fL = 0;
        for (int M : rows) {
            Call cp, cc, cn, cf, cl;
            calls(M, cp, cc, cn, cf, cl);
            for (auto* d : { doA, doB, doL }) hc(hipMemset(d, 0x5a, nh * 2), "ms");
            for (auto* d : { dxcA, dxcB, dxcL, dyA, dyB, dyL }) hc(hipMemset(d, 0xa5, nx * 2), "ms");
            const int r1 = post(&cp.a, (RadStream)st), r2 = collapse(&cc.a, (RadStream)st), r3 = norm(&cn.a, (RadStream)st);
            const int r4 = pcn(&cf.a, (RadStream)st), r5 = pcnl(&cl.a, (RadStream)st);
            hc(hipStreamSynchronize(st), "sync");
            const size_t no = (size_t)M * HC * D, nr = (size_t)M * D;
            const auto oA = get(doA, no), oB = get(doB, no), oL = get(doL, no);
            const auto xA = get(dxcA, nr), xB = get(dxcB, nr), xL = get(dxcL, nr);
            const auto yA = get(dyA, nr), yB = get(dyB, nr), yL = get(dyL, nr);
            const long bO = ndiff(oA, oB), bX = ndiff(xA, xB), bY = ndiff(yA, yB);
            const long lO = ndiff(oA, oL), lX = ndiff(xA, xL), lY = ndiff(yA, yL);
            tot_cmp += (long)(no + 2 * nr);
            const bool okF = !bO && !bX && !bY && !(r1 | r2 | r3 | r4), okL = !lO && !lX && !lY && !(r1 | r2 | r3 | r5);
            fF += !okF; fL += !okL;
            if (kind > 0 && okL) continue;   /* every row count for typical inputs; otherwise where `lanes` 256 differs */
            std::printf("A  kind %d M %2d: rc chain %d %d %d / fused %d / lanes %d; vs the chain: fused (served fold) out %ld xc %ld y %ld, "
                        "l256 out %ld xc %ld y %ld of %zu / %zu / %zu -> served fold %s, l256 %s\n", kind, M, r1, r2, r3, r4, r5,
                        bO, bX, bY, lO, lX, lY, no, nr, nr, okF ? "IDENTICAL" : "DIFFER", okL ? "IDENTICAL" : "DIFFER");
        }
        std::printf("A  kind %d (%s): (M) cells differing from the chain: served fold %d of %zu, l256 %d of %zu\n", kind,
                    kind == 0 ? "typical" : kind == 1 ? "magnitudes 2^-60 .. 2^60 a row" : "specials", fF, sizeof rows / sizeof rows[0], fL,
                    sizeof rows / sizeof rows[0]);
        failsF += fF; failsL += fL;
    }
    std::printf("A  %ld values a form compared; l256 vs the chain: %d cells differ -> %s (served fold: %d cells differ)\n", tot_cmp,
                failsL, failsL ? "FAIL" : "PASS", failsF);
    int fails = failsL;

    /* C: batch invariance, rows 0 .. k-1 at M = k vs the same rows of an M = 96 launch (kind 1 inputs: the most flips) */
    fill_inputs(1);
    {
        Call cp, cc, cn, cf, cl;
        calls(96, cp, cc, cn, cf, cl);
        post(&cp.a, (RadStream)st); collapse(&cc.a, (RadStream)st); norm(&cn.a, (RadStream)st);
        pcn(&cf.a, (RadStream)st); pcnl(&cl.a, (RadStream)st);
        hc(hipStreamSynchronize(st), "sync");
        const auto yA96 = get(dyA, (size_t)96 * D), yB96 = get(dyB, (size_t)96 * D), yL96 = get(dyL, (size_t)96 * D);
        int vA = 0, vB = 0, vL = 0;
        std::string lA, lB, lL;
        for (int k : { 1, 5, 8, 16, 24, 30, 32, 33, 40, 48, 64, 80, 95 }) {
            Call p2, c2, n2, f2, l2;
            calls(k, p2, c2, n2, f2, l2);
            post(&p2.a, (RadStream)st); collapse(&c2.a, (RadStream)st); norm(&n2.a, (RadStream)st);
            pcn(&f2.a, (RadStream)st); pcnl(&l2.a, (RadStream)st);
            hc(hipStreamSynchronize(st), "sync");
            const size_t nr = (size_t)k * D;
            const auto yA = get(dyA, nr), yB = get(dyB, nr), yL = get(dyL, nr);
            auto pre = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long q = 0; for (size_t i = 0; i < nr; ++i) q += a[i] != b[i]; return q; };
            const long dA = pre(yA, yA96), dB = pre(yB, yB96), dL = pre(yL, yL96);
            char buf[64];
            if (dA) { ++vA; std::snprintf(buf, sizeof buf, " %d(%ld)", k, dA); lA += buf; }
            if (dB) { ++vB; std::snprintf(buf, sizeof buf, " %d(%ld)", k, dB); lB += buf; }
            if (dL) { ++vL; std::snprintf(buf, sizeof buf, " %d(%ld)", k, dL); lL += buf; }
        }
        std::printf("C  y rows of an M = k launch vs the same rows at M = 96, k in 13 values: chain %d differ%s; fused served fold %d%s; "
                    "l256 %d%s -> l256 %s\n", vA, lA.c_str(), vB, lB.c_str(), vL, lL.c_str(), vL ? "VARIES" : "INVARIANT");
        fails += vL + vA;
    }

    fill_inputs(0);
    for (int M : { 8, 16, 24, 32, 40, 48, 64, 80 }) {
        Call cp, cc, cn, cf, cl;
        calls(M, cp, cc, cn, cf, cl);
        auto timed = [&](auto&& body) {
            std::vector<double> v;
            for (int r = 0; r < 7; ++r) {
                body(); hc(hipStreamSynchronize(st), "s");
                hc(hipEventRecord(e0, st), "e");
                for (int i = 0; i < iters; ++i) body();
                hc(hipEventRecord(e1, st), "e"); hc(hipEventSynchronize(e1), "e");
                float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
                v.push_back(1000.0 * ms / iters);
            }
            return p50(v);
        };
        const double tp = timed([&] { post(&cp.a, (RadStream)st); });
        const double tc = timed([&] { collapse(&cc.a, (RadStream)st); });
        const double tn = timed([&] { norm(&cn.a, (RadStream)st); });
        const double t3 = timed([&] { post(&cp.a, (RadStream)st); collapse(&cc.a, (RadStream)st); norm(&cn.a, (RadStream)st); });
        const double tf = timed([&] { pcn(&cf.a, (RadStream)st); });
        const double tl = timed([&] { pcnl(&cl.a, (RadStream)st); });
        std::printf("B  M %2d: us a launch back to back (p50 of 7 x %d): post %.2f, collapse %.2f, rmsnorm %.2f, the three in sequence %.2f; "
                    "fused served fold %.2f, fused l256 %.2f (%+.2f us vs the three, %+.1f %%)\n", M, iters, tp, tc, tn, t3, tf, tl, tl - t3,
                    100.0 * (tl - t3) / t3);
    }
    std::printf("dph: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
