/* tools/dbench/dkp.cpp (DRAFTER track 21.11.2026 round 72, lever DKP) -- one card, through libr4d's own exported entry
 * r4d_rad_gemm_nt_ggml_tiled (dlopen'd, fn4 ds41_wmk set to the served 14): the DSpark CONTEXT pass's three window-key
 * projections (each stage's wkv, q8_0 512 x 5120, over the same normed rows cn [M, 5120]) as today's chain -- one after the
 * other on one stream (the engine plays them ordered: each stage's d_kvfs is behind its d_kv and the next d_kv writes the
 * same d.ckv) -- vs lever DKP's form, the three projections into their own buffers started together (the engine's
 * unordered marks; here three streams forked from and joined to the first by events, which adds the event packets the
 * engine's form does not have: an upper bound of DKP's three).
 *
 *  A  bytes: each stage's output of the concurrent form == the chain's, byte for byte, at every M.
 *  B  time: cold (a 256 MiB memset before each rep evicts L2 + MALL; the served weights are cold behind the trunk pass),
 *     event-timed on stream 0, p50 / p10 / p90 of ITERS reps, arms interleaved in each rep: chain3 (three back to back),
 *     conc3 (forked), one (a single projection), wide (ONE launch over the three weights as one 1536 x 5120 q8_0: the
 *     proxy of three unordered launches sharing the card, since the HIP fork / join's event packets cost more than the
 *     launches; its bytes vs the chain's are printed too).
 * Usage: dkp LIBR4D [ITERS] */
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

using Entry = int (*)(const RadArgs*, RadStream);
using Fn4 = int (*)(const char*, int);
using Fn4v = int (*)(const char*, int*);

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }
static uint16_t h_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static uint16_t h_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; std::memcpy(&u, &h, 2); return u; }

static RadTensor T2(void* d, uint32_t dt, int64_t r, int64_t c, int64_t ld) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

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
    if (argc < 2) { std::fprintf(stderr, "usage: dkp LIBR4D [ITERS]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 60;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto gemm = (Entry)dlsym(lib, "r4d_rad_gemm_nt_ggml_tiled");
    auto fn4 = (Fn4)dlsym(lib, "rad_fn4_switch");
    auto fn4v = (Fn4v)dlsym(lib, "rad_fn4_value");
    if (!gemm || !fn4 || !fn4v) { std::fprintf(stderr, "symbols missing\n"); return 2; }
    int wv = -1;
    const int sw = fn4("ds41_wmk", 14);
    fn4v("ds41_wmk", &wv);
    std::printf("dkp: %s, fn4 ds41_wmk 14 -> %d (applied %d), %d reps\n", argv[1], sw, wv, iters);
    const int K = 5120, N = 512, MAXM = 64, S = 3;
    const long long rb = (long long)K / 32 * 34;
    std::vector<int> Ms = { 16, 24, 32, 43, 48, 64 };
    uint16_t* dx; hc(hipMalloc(&dx, (size_t)MAXM * K * 2), "x");
    uint8_t* dw[S]; uint16_t *dyA[S], *dyB[S], *dyW;
    uint8_t* dwall; hc(hipMalloc(&dwall, (size_t)S * N * rb), "w");   /* the three weights contiguous (wide arm) */
    hc(hipMalloc(&dyW, (size_t)MAXM * S * N * 2), "y");
    for (int s = 0; s < S; ++s) {
        dw[s] = dwall + (size_t)s * N * rb;
        hc(hipMalloc(&dyA[s], (size_t)MAXM * N * 2), "y");
        hc(hipMalloc(&dyB[s], (size_t)MAXM * N * 2), "y");
    }
    {   /* rows ~ the normed main_x (unit rms), weights q8_0 blocks with small f16 scales */
        std::vector<uint16_t> hx((size_t)MAXM * K);
        for (auto& v : hx) v = h_bf16(uni(-1.7f, 1.7f));
        hc(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice), "cp");
        std::vector<uint8_t> hw((size_t)N * rb);
        for (int s = 0; s < S; ++s) {
            for (long long r = 0; r < N; ++r)
                for (long long b = 0; b < K / 32; ++b) {
                    uint8_t* p = &hw[(size_t)(r * rb + b * 34)];
                    const uint16_t sc = h_f16(uni(0.0005f, 0.004f));
                    std::memcpy(p, &sc, 2);
                    for (int j = 0; j < 32; ++j) p[2 + j] = (uint8_t)(int8_t)((int)(r32() % 255) - 127);
                }
            hc(hipMemcpy(dw[s], hw.data(), hw.size(), hipMemcpyHostToDevice), "cp");
        }
    }
    void* dev; const size_t evb = 256ull << 20; hc(hipMalloc(&dev, evb), "evict");
    hipStream_t st[S];
    for (int s = 0; s < S; ++s) hc(hipStreamCreateWithFlags(&st[s], hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1, ef, ej[S];
    hc(hipEventCreate(&e0), "ev"); hc(hipEventCreate(&e1), "ev");
    hc(hipEventCreateWithFlags(&ef, hipEventDisableTiming), "ev");
    for (int s = 0; s < S; ++s) hc(hipEventCreateWithFlags(&ej[s], hipEventDisableTiming), "ev");
    int bad = 0;
    std::printf("%5s %28s %28s %28s %28s %10s %10s %10s\n", "M", "chain3 p50 (p10 / p90) us", "conc3 p50 (p10 / p90) us",
                "one p50 (p10 / p90) us", "wide p50 (p10 / p90) us", "chain-wide", "bytes", "wide bytes");
    for (int M : Ms) {
        Call cA[S], cB[S];
        for (int s = 0; s < S; ++s) {
            for (Call* c : { &cA[s], &cB[s] }) {
                c->t = { T2(dx, RAD_BF16, M, K, K), T2(dw[s], RAD_U8, N, rb, rb), T2(c == &cA[s] ? dyA[s] : dyB[s], RAD_BF16, M, N, N) };
                c->p = { PI("M", MAXM), PI("N", N), PI("K", K), PS("type", "q8_0"), PS("dtype", "bf16") };
                c->fix();
            }
        }
        Call cW;
        cW.t = { T2(dx, RAD_BF16, M, K, K), T2(dwall, RAD_U8, (int64_t)S * N, rb, rb), T2(dyW, RAD_BF16, M, (int64_t)S * N, (int64_t)S * N) };
        cW.p = { PI("M", MAXM), PI("N", (long long)S * N), PI("K", K), PS("type", "q8_0"), PS("dtype", "bf16") };
        cW.fix();
        auto run = [&](Call& c, hipStream_t s) {
            const int rc = gemm(&c.a, (RadStream)s);
            if (rc != 0) { std::fprintf(stderr, "gemm rc %d at M %d\n", rc, M); std::exit(3); }
        };
        auto evict = [&] { hc(hipMemsetAsync(dev, (int)(r32() & 0xff), evb, st[0]), "evict"); };
        auto timed = [&](int arm) {
            evict();
            hc(hipEventRecord(e0, st[0]), "rec");
            if (arm == 0) {
                for (int s = 0; s < S; ++s) run(cA[s], st[0]);
            } else if (arm == 1) {
                hc(hipEventRecord(ef, st[0]), "rec");
                for (int s = 1; s < S; ++s) hc(hipStreamWaitEvent(st[s], ef, 0), "wait");
                for (int s = 0; s < S; ++s) run(cB[s], st[s]);
                for (int s = 1; s < S; ++s) { hc(hipEventRecord(ej[s], st[s]), "rec"); hc(hipStreamWaitEvent(st[0], ej[s], 0), "wait"); }
            } else if (arm == 2) {
                run(cA[0], st[0]);
            } else {
                run(cW, st[0]);
            }
            hc(hipEventRecord(e1, st[0]), "rec");
            hc(hipEventSynchronize(e1), "sync");
            float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
            return 1000.0 * ms;
        };
        for (int w = 0; w < 5; ++w) { timed(0); timed(1); timed(2); timed(3); }
        std::vector<double> t[4];
        for (int i = 0; i < iters; ++i) {
            const int ord[4] = { i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4 };
            for (int a : ord) t[a].push_back(timed(a));
        }
        hc(hipDeviceSynchronize(), "sync");
        long nd = 0;
        for (int s = 0; s < S; ++s) {
            std::vector<uint16_t> a((size_t)M * N), b((size_t)M * N);
            hc(hipMemcpy(a.data(), dyA[s], a.size() * 2, hipMemcpyDeviceToHost), "cp");
            hc(hipMemcpy(b.data(), dyB[s], b.size() * 2, hipMemcpyDeviceToHost), "cp");
            for (size_t i = 0; i < a.size(); ++i) nd += a[i] != b[i];
        }
        bad += nd != 0;
        long nw = 0;
        {
            std::vector<uint16_t> w((size_t)M * S * N);
            hc(hipMemcpy(w.data(), dyW, w.size() * 2, hipMemcpyDeviceToHost), "cp");
            for (int s = 0; s < S; ++s) {
                std::vector<uint16_t> a((size_t)M * N);
                hc(hipMemcpy(a.data(), dyA[s], a.size() * 2, hipMemcpyDeviceToHost), "cp");
                for (int r = 0; r < M; ++r)
                    for (int j = 0; j < N; ++j) nw += a[(size_t)r * N + j] != w[(size_t)r * S * N + (size_t)s * N + j];
            }
        }
        double q[4][3];
        for (int a = 0; a < 4; ++a) pct(t[a], &q[a][0], &q[a][1], &q[a][2]);
        std::printf("%5d %10.1f (%6.1f / %6.1f) %10.1f (%6.1f / %6.1f) %10.1f (%6.1f / %6.1f) %10.1f (%6.1f / %6.1f) %10.1f %10s %10ld\n", M,
                    q[0][1], q[0][0], q[0][2], q[1][1], q[1][0], q[1][2], q[2][1], q[2][0], q[2][2], q[3][1], q[3][0], q[3][2],
                    q[0][1] - q[3][1], nd ? "DIFFER" : "equal", nw);
    }
    std::printf("dkp: %s\n", bad ? "BYTES DIFFER" : "every M: the concurrent form's outputs == the chain's, byte for byte");
    return bad ? 1 : 0;
}
