/* wbench (DRAFTER track 18.11.2026) -- one card, the DSpark walk (dspark_walk_step, libr4d r4d_ds41_walk.hip) as a drafter
 * pass runs it at TP4: a rank's vocabulary shard N = 32320 of V = 129280, rank R = 256, steps 5 (+ the last merge), S
 * sequences; succ / pred bf16 random, logits f32 random. Per (form, sequences a pass) the us of each step (median over
 * iters) and the tokens, byte-compared with the first variant's. world 1 (each step's keys are the next step's prev).
 * usage: wbench [--seqs 8,16] [--iters 50] */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;

extern "C" int r4d_rad_dspark_walk_step(const RadArgs* a, RadStream s);
extern "C" void r4d_walk_seqs(int n);
extern "C" void r4d_walk_wm(int on);
extern "C" void r4d_walk_geo(int le, int pf, int tpw);   /* lever WKG (round 60) */

/* round 60: --cold = a 256 MiB read before each pass's step 0 (the served step 0 follows the head's 136 MB stream: succ is
 * not in the 64 MiB LLC there; steps 1.. find it), and the empty-launch event cost of this stream as the offset */
__global__ void wb_flush_kernel(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc ^= p[i].x ^ p[i].w;
    if (acc == 0x12345678u) *out = acc;
}
__global__ void wb_empty_kernel() {}

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

static int g_bi = 0;
int main(int argc, char** argv) {
    std::vector<int> Ss = { 8, 16 };
    int iters = 50, cold = 0;
    int64_t steps = 5;
    std::vector<std::string> geos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--cold") cold = 1;
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--geo" && i + 1 < argc) {   /* <le><pf><tpw>,... : the WMMA form's WKG geometries (004 = as before) */
            std::string v = argv[++i]; size_t p = 0;
            while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); geos.push_back(v.substr(p, q - p)); p = q + 1; }
        }
        else if (a == "--bi") g_bi = 1;
        else if (a == "--seqs" && i + 1 < argc) {
            Ss.clear(); std::string v = argv[++i]; size_t p = 0;
            while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); Ss.push_back(std::atoi(v.substr(p, q - p).c_str())); p = q + 1; }
        }
    }
    const int64_t V = 129280, N = 32320, R = 256;
    std::vector<uint16_t> hp((size_t)(V * R)), hs((size_t)(V * R));
    uint64_t rs = 0x5eedull;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    for (auto& v : hp) v = bf((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    for (auto& v : hs) v = bf((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    void *dp, *dsu;
    CK(hipMalloc(&dp, hp.size() * 2)); CK(hipMemcpy(dp, hp.data(), hp.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dsu, hs.size() * 2)); CK(hipMemcpy(dsu, hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    int fails = 0;
    const size_t nfl = (256u << 20) / 16;
    void *dfl = nullptr, *dfo = nullptr;
    CK(hipMalloc(&dfl, nfl * 16)); CK(hipMemset(dfl, 1, nfl * 16)); CK(hipMalloc(&dfo, 16));
    {   /* the empty launch between two events on this stream: the offset every step row carries */
        std::vector<float> w;
        for (int it = 0; it < 60; ++it) {
            hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
            CK(hipEventRecord(e0, s)); hipLaunchKernelGGL(wb_empty_kernel, dim3(1), dim3(32), 0, s); CK(hipEventRecord(e1, s));
            CK(hipStreamSynchronize(s)); float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 10) w.push_back(ms * 1000.f);
            CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
        }
        std::sort(w.begin(), w.end());
        std::printf("empty launch between events: p50 %.1f us (p10 %.1f, p90 %.1f)\n", w[w.size() / 2], w[w.size() / 10], w[w.size() * 9 / 10]);
    }
    std::printf("steps %lld, %s, iters %d\n", (long long)steps, cold ? "COLD (256 MiB read before each step 0)" : "warm", iters);
    if (g_bi) {
        const int S0 = 16;
        std::vector<float> hl((size_t)(S0 * steps * N));
        for (auto& v : hl) v = (float)((int)(rnd() % 20001) - 10000) / 1000.0f;
        std::vector<int32_t> ha((size_t)S0);
        for (auto& v : ha) v = (int32_t)(rnd() % V);
        auto runw = [&](const std::vector<int>& order, std::vector<int32_t>& tok) {
            const int S = (int)order.size();
            std::vector<float> l2((size_t)(S * steps * N)); std::vector<int32_t> a2((size_t)S);
            for (int q = 0; q < S; ++q) {
                a2[(size_t)q] = ha[(size_t)order[(size_t)q]];
                std::memcpy(&l2[(size_t)q * steps * N], &hl[(size_t)order[(size_t)q] * steps * N], (size_t)(steps * N) * 4);
            }
            void *dl, *da, *dprev, *dkeys, *dtok;
            CK(hipMalloc(&dl, l2.size() * 4)); CK(hipMemcpy(dl, l2.data(), l2.size() * 4, hipMemcpyHostToDevice));
            CK(hipMalloc(&da, (size_t)S * 4)); CK(hipMemcpy(da, a2.data(), (size_t)S * 4, hipMemcpyHostToDevice));
            CK(hipMalloc(&dprev, (size_t)S * 16)); CK(hipMalloc(&dkeys, (size_t)S * 16)); CK(hipMalloc(&dtok, (size_t)S * steps * 4));
            for (int64_t st = 0; st <= steps; ++st) {
                std::vector<RadTensor> ts = { T(dl, RAD_F32, { S * steps, N }), T(da, RAD_I32, { S }), T(dprev, RAD_F32, { S, 4 }),
                                              T(dp, RAD_BF16, { V, R }), T(dsu, RAD_BF16, { V, R }), T(dkeys, RAD_F32, { S, 4 }),
                                              T(dtok, RAD_I32, { S, steps }) };
                std::vector<RadParam> ps = { PI("step", st), PI("steps", steps), PI("rank", R), PI("n_vocab", V), PI("N", N),
                                             PI("vocab_off", 0), PI("world", 1), PI("anchor_stride", 1), PI("form", 0) };
                RadArgs a{}; a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
                if (st < steps) CK(hipMemsetAsync(dkeys, 0, (size_t)S * 16, s));
                r4d_rad_dspark_walk_step(&a, (RadStream)s);
                if (st < steps) CK(hipMemcpyAsync(dprev, dkeys, (size_t)S * 16, hipMemcpyDeviceToDevice, s));
            }
            CK(hipStreamSynchronize(s));
            tok.resize((size_t)(S * steps)); CK(hipMemcpy(tok.data(), dtok, tok.size() * 4, hipMemcpyDeviceToHost));
            hipFree(dl); hipFree(da); hipFree(dprev); hipFree(dkeys); hipFree(dtok);
        };
        /* round 60: with --geo every geometry's WMMA form, each against its own full-batch run AND the first geometry's */
        std::vector<std::string> bg = geos.empty() ? std::vector<std::string>{ "f0", "004" } : geos;
        if (!geos.empty()) bg.insert(bg.begin(), "f0");
        std::vector<int32_t> tg0;
        for (size_t gi = 0; gi < bg.size(); ++gi) {
            const int wm = bg[gi] == "f0" ? 0 : 1;
            r4d_walk_wm(wm);
            if (wm) r4d_walk_geo(bg[gi][0] - '0', bg[gi][1] - '0', bg[gi][2] - '0');
            std::vector<int> full; for (int q = 0; q < S0; ++q) full.push_back(q);
            std::vector<int32_t> t0; runw(full, t0);
            if (wm && tg0.empty()) tg0 = t0;
            else if (wm && t0 != tg0) { std::printf("geometry %s: full-batch tokens DIFFER from %s's\n", bg[gi].c_str(), bg[1].c_str()); ++fails; }
            int bad = 0, cmpn = 0;
            for (int pre : { 1, 3, 4, 8, 12 }) {
                std::vector<int> o(full.begin(), full.begin() + pre); std::vector<int32_t> t; runw(o, t);
                for (int q = 0; q < pre * steps; ++q) { ++cmpn; bad += t[(size_t)q] != t0[(size_t)q]; }
            }
            { std::vector<int> o(full.rbegin(), full.rend()); std::vector<int32_t> t; runw(o, t);
              for (int q = 0; q < S0; ++q) for (int l = 0; l < steps; ++l) { ++cmpn; bad += t[(size_t)(q * steps + l)] != t0[(size_t)((S0 - 1 - q) * steps + l)]; } }
            std::printf("batch invariance (%s): prefixes 1 / 3 / 4 / 8 / 12 of 16 + the reversed order: %d of %d tokens differ\n",
                        wm ? ("wmma geo " + bg[gi]).c_str() : "form 0", bad, cmpn);
            fails += bad != 0;
        }
        r4d_walk_wm(0); r4d_walk_geo(0, 0, 4);
    }
    for (int S : Ss) {
        std::vector<float> hl((size_t)(S * steps * N));
        for (auto& v : hl) v = (float)((int)(rnd() % 20001) - 10000) / 1000.0f;
        std::vector<int32_t> ha((size_t)S);
        for (auto& v : ha) v = (int32_t)(rnd() % V);
        void *dl, *da, *dprev, *dkeys, *dtok;
        CK(hipMalloc(&dl, hl.size() * 4)); CK(hipMemcpy(dl, hl.data(), hl.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&da, (size_t)S * 4)); CK(hipMemcpy(da, ha.data(), (size_t)S * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&dprev, (size_t)S * 16)); CK(hipMalloc(&dkeys, (size_t)S * 16)); CK(hipMalloc(&dtok, (size_t)S * steps * 4));
        std::vector<int32_t> ref, wmref;
        struct V_ { std::string nm; int form, seqs, wm, le, pf, tpw; };
        std::vector<V_> vs = { { "form0 seqs8", 0, 8, 0, 0, 0, 4 }, { "form0 seqs16", 0, 16, 0, 0, 0, 4 }, { "wmma", 0, 0, 1, 0, 0, 4 }, { "wmma again", 0, 0, 1, 0, 0, 4 } };
        if (!geos.empty()) {
            vs.clear();
            for (int rep = 0; rep < 2; ++rep)   /* A B .. then the same list again */
                for (const std::string& g : geos)
                    vs.push_back({ "geo " + g + (rep ? "'" : ""), 0, 0, 1, g[0] - '0', g[1] - '0', g[2] - '0' });
        }
        for (const V_& v : vs) {
            r4d_walk_seqs(v.seqs);
            r4d_walk_wm(v.wm);
            r4d_walk_geo(v.le, v.pf, v.tpw);
            std::vector<std::vector<float>> us((size_t)steps + 1);
            std::vector<int32_t> got((size_t)(S * steps));
            for (int it = 0; it < iters + 3; ++it) {
                for (int64_t st = 0; st <= steps; ++st) {
                    std::vector<RadTensor> ts = { T(dl, RAD_F32, { S * steps, N }), T(da, RAD_I32, { S }), T(dprev, RAD_F32, { S, 4 }),
                                                  T(dp, RAD_BF16, { V, R }), T(dsu, RAD_BF16, { V, R }), T(dkeys, RAD_F32, { S, 4 }),
                                                  T(dtok, RAD_I32, { S, steps }) };
                    std::vector<RadParam> ps = { PI("step", st), PI("steps", steps), PI("rank", R), PI("n_vocab", V), PI("N", N),
                                                 PI("vocab_off", 0), PI("world", 1), PI("anchor_stride", 1), PI("form", v.form) };
                    RadArgs a{}; a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
                    if (st < steps) CK(hipMemsetAsync(dkeys, 0, (size_t)S * 16, s));
                    if (cold && st == 0) hipLaunchKernelGGL(wb_flush_kernel, dim3(1024), dim3(256), 0, s, (const uint4*)dfl, nfl, (unsigned*)dfo);
                    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                    CK(hipEventRecord(e0, s));
                    const int rc = r4d_rad_dspark_walk_step(&a, (RadStream)s);
                    CK(hipEventRecord(e1, s));
                    CK(hipStreamSynchronize(s));
                    if (rc != 0) { std::printf("  %s step %lld rc %d\n", v.nm.c_str(), (long long)st, rc); ++fails; break; }
                    float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
                    if (it >= 3) us[(size_t)st].push_back(ms * 1000.f);
                    CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
                    if (st < steps) CK(hipMemcpyAsync(dprev, dkeys, (size_t)S * 16, hipMemcpyDeviceToDevice, s));
                }
            }
            CK(hipMemcpy(got.data(), dtok, got.size() * 4, hipMemcpyDeviceToHost));
            bool same = true;
            int ndiff = 0;
            if (ref.empty()) ref = got; else { same = got == ref; for (size_t q = 0; q < got.size(); ++q) ndiff += got[q] != ref[q]; }
            if (!same && (!v.wm || !geos.empty())) ++fails;
            if (v.wm && !wmref.empty() && got != wmref) { std::printf("  wmma NOT DETERMINISTIC\n"); ++fails; }
            if (v.wm && wmref.empty()) wmref = got;
            double tot = 0;
            std::printf("S %2d %-12s tokens %s (%d of %zu differ) |", S, v.nm.c_str(), same ? "IDENTICAL" : "DIFFER", ndiff, got.size());
            for (int64_t st = 0; st <= steps; ++st) {
                auto w = us[(size_t)st]; std::sort(w.begin(), w.end());
                const double m = w.empty() ? 0 : w[w.size() / 2];
                tot += m;
                std::printf(" %.1f", m);
            }
            std::printf(" | sum %.1f us\n", tot);
            std::fflush(stdout);
        }
        r4d_walk_seqs(0); r4d_walk_wm(0);
        hipFree(dl); hipFree(da); hipFree(dprev); hipFree(dkeys); hipFree(dtok);
    }
    std::printf("WVERDICT %s\n", fails ? "SOME VARIANT DIFFERS / FAILED" : "all variants' tokens identical");
    return fails ? 1 : 0;
}
