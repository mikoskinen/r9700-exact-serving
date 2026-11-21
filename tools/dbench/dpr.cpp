/* tools/dbench/dpr.cpp (DRAFTER track 21.11.2026 round 67, levers DPR + CNF) -- one card, through libr4d's own exported
 * entries (dlopen'd):
 *
 *  A  lever DPR's identity: r4d_rad_mhc_collapse over h0 = hc copies of x (h0's column strips, as the hc expand casts write
 *     them) with pre_f one-hot on stream 0 (pre0 fill 0, pre1 fill 1) == x? x = every one of the 65,536 bf16 patterns; the
 *     differing patterns are printed (the reasoning in deepseek41.cpp g_sw_dmr's comment says: -0.0 and NaN payloads only).
 *  B  the work DPR removes, at T rows: r4d_rad_fill [T, hc] 0, r4d_rad_fill [T, 1] 1, r4d_rad_mhc_collapse [T, hc x D] ->
 *     [T, D]; one stream, back to back (the HIP launch path, not the engine's AQL queues: an upper bound of their kernel
 *     time, not of the served gaps), event-timed: mean a launch over 200 back to back, and the three as a sequence.
 *  C  lever CNF: r4d_rad_dspark_conf (libr4d's conf_kernel) vs conf_cnf_kernel below (the same per-thread sums in the same
 *     order and the same pairwise tree, its last five levels in wave 0 by __shfl_down, the predecessor id read by every
 *     thread, the x loads unrolled): outputs byte-compared over random inputs at every (M, steps) the engine runs, times as B.
 * Usage: dpr LIBR4D [ITERS] */
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

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
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

/* ---- lever CNF: libr4d r4d_dspark_extra.hip conf_kernel's arithmetic in fewer barriers */
namespace cnf {
typedef unsigned short u16;
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float((unsigned)v << 16); }
struct ConfArgs {   /* = r4d_dspx::ConfArgs */
    const u16* x; long long x_ld;
    const int* anchor; long long astr;
    const int* tok; long long t_ld;
    const u16* pred;
    const float* w;
    float* out; long long o_ld;
    int steps, D, R, V;
    float bias;
};
constexpr int kT = 256;
__global__ __launch_bounds__(256) void conf_cnf_kernel(ConfArgs A) {
    __shared__ float part[kT];
    const int s = (int)blockIdx.x, j = (int)blockIdx.y, t = (int)threadIdx.x;
    const int pid = j == 0 ? A.anchor[(long long)s * A.astr] : A.tok[(long long)s * A.t_ld + j - 1];
    const u16* xr = A.x + ((long long)s * A.steps + j) * A.x_ld;
    float acc = 0.0f;
    int k = t;
    for (; k + 3 * kT < A.D; k += 4 * kT) {   /* the loads of four terms first, then the same sums in k order */
        const float w0 = A.w[k], w1 = A.w[k + kT], w2 = A.w[k + 2 * kT], w3 = A.w[k + 3 * kT];
        const u16 x0 = xr[k], x1 = xr[k + kT], x2 = xr[k + 2 * kT], x3 = xr[k + 3 * kT];
        acc += w0 * bf2f(x0);
        acc += w1 * bf2f(x1);
        acc += w2 * bf2f(x2);
        acc += w3 * bf2f(x3);
    }
    for (; k < A.D; k += kT) acc += A.w[k] * bf2f(xr[k]);
    if (pid >= 0 && pid < A.V) {
        const u16* pr = A.pred + (long long)pid * A.R;
        for (int r = t; r < A.R; r += kT) acc += A.w[A.D + r] * bf2f(pr[r]);
    }
    part[t] = acc;
    __syncthreads();
    for (int o = kT / 2; o >= 32; o >>= 1) {   /* levels 128, 64, 32 across waves, as the reference's tree */
        if (t < o) part[t] += part[t + o];
        __syncthreads();
    }
    if (t < 32) {   /* levels 16 .. 1 inside wave 0 (wave32): lane t adds lane t + o, the reference's pairs */
        float v = part[t];
        for (int o = 16; o > 0; o >>= 1) {
            const float u = __shfl_down(v, o, 32);
            if (t < o) v += u;
        }
        if (t == 0) A.out[(long long)s * A.o_ld + j] = v + A.bias;
    }
}
}  // namespace cnf

static double p50(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dpr LIBR4D [ITERS]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto collapse = (Entry)dlsym(h, "r4d_rad_mhc_collapse");
    auto fill = (Entry)dlsym(h, "r4d_rad_fill");
    auto conf = (Entry)dlsym(h, "r4d_rad_dspark_conf");
    auto cnfhook = (void (*)(int))dlsym(h, "r4d_conf_cnf");   /* lever CNF's hook: libr4d of round 67 on */
    std::printf("dpr: %s; lever CNF hook %s\n", argv[1], cnfhook ? "present (part D runs)" : "absent");
    if (!collapse || !fill || !conf) { std::fprintf(stderr, "symbols missing (%p %p %p)\n", (void*)collapse, (void*)fill, (void*)conf); return 2; }
    const int D = 5120, HC = 4;
    hipStream_t st; hc(hipStreamCreate(&st), "stream");
    hipEvent_t e0, e1; hc(hipEventCreate(&e0), "ev"); hc(hipEventCreate(&e1), "ev");
    int fails = 0;

    /* ---------------- A: the identity over every bf16 pattern */
    {
        const int R = (65536 + D - 1) / D;   /* 13 rows */
        std::vector<uint16_t> hx((size_t)R * D);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (uint16_t)(i & 0xffff);
        uint16_t *dx, *dh, *dy; float* dp;
        hc(hipMalloc(&dx, hx.size() * 2), "m"); hc(hipMalloc(&dy, hx.size() * 2), "m");
        hc(hipMalloc(&dh, hx.size() * 2 * HC), "m"); hc(hipMalloc(&dp, (size_t)R * HC * 4), "m");
        hc(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice), "cp");
        for (int i = 0; i < HC; ++i)   /* the hc expand casts: x into h0's strips */
            hc(hipMemcpy2D(dh + (size_t)i * D, (size_t)HC * D * 2, dx, (size_t)D * 2, (size_t)D * 2, R, hipMemcpyDeviceToDevice), "cp2d");
        hc(hipMemset(dy, 0x5a, hx.size() * 2), "ms");
        Call f0, f1, cc;
        f0.t = { T2(dp, RAD_F32, R, HC, HC) }; f0.p = { PI("M", R), PI("n", HC), PS("dtype", "f32"), PF("value", 0.0) }; f0.fix();
        f1.t = { T2(dp, RAD_F32, R, 1, HC) }; f1.p = { PI("M", R), PI("n", 1), PS("dtype", "f32"), PF("value", 1.0) }; f1.fix();
        cc.t = { T2(dh, RAD_BF16, R, (int64_t)HC * D, (int64_t)HC * D), T2(dp, RAD_F32, R, HC, HC), T2(dy, RAD_BF16, R, D, D) };
        cc.p = { PI("M", R), PI("n", D), PI("hc", HC), PS("dtype", "bf16") }; cc.fix();
        int rc = fill(&f0.a, (RadStream)st) | fill(&f1.a, (RadStream)st) | collapse(&cc.a, (RadStream)st);
        hc(hipStreamSynchronize(st), "sync");
        std::vector<uint16_t> hy(hx.size());
        std::vector<float> hp((size_t)R * HC);
        hc(hipMemcpy(hy.data(), dy, hy.size() * 2, hipMemcpyDeviceToHost), "cp");
        hc(hipMemcpy(hp.data(), dp, hp.size() * 4, hipMemcpyDeviceToHost), "cp");
        int pre_ok = 1;
        for (int r = 0; r < R; ++r)
            for (int i = 0; i < HC; ++i) pre_ok &= hp[(size_t)r * HC + i] == (i == 0 ? 1.0f : 0.0f);
        long nd = 0, nd_neg0 = 0, nd_nan = 0, nd_inf = 0, nd_other = 0, nnan = 0;
        for (size_t i = 0; i < 65536; ++i) {
            const uint16_t v = hx[i];
            const bool nan = (v & 0x7f80) == 0x7f80 && (v & 0x7f);
            nnan += nan;
            if (hy[i] == v) continue;
            ++nd;
            if (v == 0x8000) ++nd_neg0;
            else if (nan) ++nd_nan;
            else if ((v & 0x7fff) == 0x7f80) ++nd_inf;   /* 0 x inf = NaN in the zero-gated streams */
            else { if (nd_other < 8) std::printf("A  differs: x 0x%04x -> 0x%04x\n", v, hy[i]); ++nd_other; }
        }
        std::printf("A  identity (rc %d, pre_f one-hot %s): of 65,536 bf16 patterns %ld differ: -0.0 %ld, NaN %ld (of %ld NaN patterns), "
                    "+-inf %ld, other %ld -> %s\n", rc, pre_ok ? "yes" : "NO", nd, nd_neg0, nd_nan, nnan, nd_inf, nd_other,
                    nd_other == 0 && pre_ok && rc == 0 ? "PASS (every finite x but -0.0: identical)" : "FAIL");
        fails += !(nd_other == 0 && pre_ok && rc == 0);
        hipFree(dx); hipFree(dy); hipFree(dh); hipFree(dp);
    }

    /* ---------------- B: the removed work at the engine's row counts */
    {
        const int rows[] = { 16, 32, 40, 48, 80 };
        const int MAXT = 80;
        uint16_t *dx, *dh; float* dp;
        hc(hipMalloc(&dx, (size_t)MAXT * D * 2), "m"); hc(hipMalloc(&dh, (size_t)MAXT * D * 2 * HC), "m");
        hc(hipMalloc(&dp, (size_t)MAXT * HC * 4), "m");
        hc(hipMemset(dh, 0x3c, (size_t)MAXT * D * 2 * HC), "ms");
        for (int T : rows) {
            Call f0, f1, cc;
            f0.t = { T2(dp, RAD_F32, T, HC, HC) }; f0.p = { PI("M", T), PI("n", HC), PS("dtype", "f32"), PF("value", 0.0) }; f0.fix();
            f1.t = { T2(dp, RAD_F32, T, 1, HC) }; f1.p = { PI("M", T), PI("n", 1), PS("dtype", "f32"), PF("value", 1.0) }; f1.fix();
            cc.t = { T2(dh, RAD_BF16, T, (int64_t)HC * D, (int64_t)HC * D), T2(dp, RAD_F32, T, HC, HC), T2(dx, RAD_BF16, T, D, D) };
            cc.p = { PI("M", T), PI("n", D), PI("hc", HC), PS("dtype", "bf16") }; cc.fix();
            auto timed = [&](auto&& body, int n) {
                std::vector<double> v;
                for (int r = 0; r < 7; ++r) {
                    body(); hc(hipStreamSynchronize(st), "s");
                    hc(hipEventRecord(e0, st), "e");
                    for (int i = 0; i < n; ++i) body();
                    hc(hipEventRecord(e1, st), "e"); hc(hipEventSynchronize(e1), "e");
                    float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
                    v.push_back(1000.0 * ms / n);
                }
                return p50(v);
            };
            const double tf0 = timed([&] { fill(&f0.a, (RadStream)st); }, iters);
            const double tf1 = timed([&] { fill(&f1.a, (RadStream)st); }, iters);
            const double tcc = timed([&] { collapse(&cc.a, (RadStream)st); }, iters);
            const double tseq = timed([&] { fill(&f0.a, (RadStream)st); fill(&f1.a, (RadStream)st); collapse(&cc.a, (RadStream)st); }, iters);
            std::printf("B  T %2d: us a launch back to back (p50 of 7 x %d): pre0 fill %.2f, pre1 fill %.2f, stage-0 collapse %.2f; the three "
                        "in sequence %.2f a pass\n", T, iters, tf0, tf1, tcc, tseq);
        }
        hipFree(dx); hipFree(dh); hipFree(dp);
    }

    /* ---------------- C: lever CNF vs libr4d's conf_kernel */
    {
        const int V = 129280, Rs[] = { 256, 512 };
        struct Sh { int M, steps; };
        const Sh shapes[] = { { 16, 2 }, { 8, 5 }, { 16, 5 }, { 1, 5 }, { 3, 2 }, { 32, 2 }, { 24, 3 } };
        const int MAXR = 512, MAXM = 32, MAXS = 5;
        uint16_t *dx, *dpr; int *da, *dt; float *dw, *do1, *do2;
        hc(hipMalloc(&dx, (size_t)MAXM * MAXS * D * 2), "m");
        hc(hipMalloc(&dpr, (size_t)V * MAXR * 2), "m");
        hc(hipMalloc(&da, (size_t)MAXM * MAXS * 4), "m"); hc(hipMalloc(&dt, (size_t)MAXM * MAXS * 4), "m");
        hc(hipMalloc(&dw, (size_t)(D + MAXR) * 4), "m");
        hc(hipMalloc(&do1, (size_t)MAXM * MAXS * 4), "m"); hc(hipMalloc(&do2, (size_t)MAXM * MAXS * 4), "m");
        {
            std::vector<uint16_t> hx((size_t)MAXM * MAXS * D), hp((size_t)V * MAXR);
            for (auto& v : hx) v = h_bf16(((int)(r32() % 20001) - 10000) * 1e-3f);
            for (auto& v : hp) v = h_bf16(((int)(r32() % 20001) - 10000) * 1e-4f);
            std::vector<float> hw(D + MAXR);
            for (auto& v : hw) v = ((int)(r32() % 20001) - 10000) * 1e-5f;
            std::vector<int> ha(MAXM * MAXS), ht(MAXM * MAXS);
            for (int i = 0; i < MAXM * MAXS; ++i) {
                ha[i] = (int)(r32() % V);
                ht[i] = i % 17 == 5 ? -1 : i % 23 == 7 ? V + 3 : (int)(r32() % V);   /* out-of-range predecessors too */
            }
            hc(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice), "cp");
            hc(hipMemcpy(dpr, hp.data(), hp.size() * 2, hipMemcpyHostToDevice), "cp");
            hc(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice), "cp");
            hc(hipMemcpy(da, ha.data(), ha.size() * 4, hipMemcpyHostToDevice), "cp");
            hc(hipMemcpy(dt, ht.data(), ht.size() * 4, hipMemcpyHostToDevice), "cp");
        }
        for (int R : Rs)
            for (const Sh& s : shapes) {
                const int M = s.M, steps = s.steps;
                Call c;
                c.t = { T2(dx, RAD_BF16, (int64_t)M * steps, D, D), T1(da, RAD_I32, (int64_t)M * steps),
                        steps > 1 ? T2(dt, RAD_I32, M, steps - 1, steps) : T2(dt, RAD_I32, M, 1, 1),
                        T2(dpr, RAD_BF16, V, R, R), T1(dw, RAD_F32, D + R), T2(do1, RAD_F32, M, steps, steps) };
                c.p = { PI("M", M), PI("steps", steps), PI("n_embd", D), PI("rank", R), PI("n_vocab", V),
                        PI("anchor_stride", steps), PF("bias", -0.375), PS("dtype", "bf16") };
                c.fix();
                cnf::ConfArgs A{};
                A.x = dx; A.x_ld = D; A.anchor = da; A.astr = steps; A.tok = dt; A.t_ld = steps > 1 ? steps : 1;
                A.pred = dpr; A.w = dw; A.out = do2; A.o_ld = steps; A.steps = steps; A.D = D; A.R = R; A.V = V; A.bias = -0.375f;
                hc(hipMemset(do1, 0x7f, (size_t)M * steps * 4), "ms"); hc(hipMemset(do2, 0x7e, (size_t)M * steps * 4), "ms");
                const int rc = conf(&c.a, (RadStream)st);
                hipLaunchKernelGGL(cnf::conf_cnf_kernel, dim3((unsigned)M, (unsigned)steps), dim3(256), 0, st, A);
                hc(hipStreamSynchronize(st), "sync");
                std::vector<uint32_t> o1((size_t)M * steps), o2((size_t)M * steps);
                hc(hipMemcpy(o1.data(), do1, o1.size() * 4, hipMemcpyDeviceToHost), "cp");
                hc(hipMemcpy(o2.data(), do2, o2.size() * 4, hipMemcpyDeviceToHost), "cp");
                int nd = 0;
                for (size_t i = 0; i < o1.size(); ++i) nd += o1[i] != o2[i];
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
                const double t1 = timed([&] { conf(&c.a, (RadStream)st); });
                const double t2 = timed([&] { hipLaunchKernelGGL(cnf::conf_cnf_kernel, dim3((unsigned)M, (unsigned)steps), dim3(256), 0, st, A); });
                std::printf("C  R %d M %2d steps %d: rc %d, %d of %d outputs differ -> %s; us a launch back to back: libr4d conf_kernel %.2f, "
                            "CNF %.2f (%+.1f %%)\n", R, M, steps, rc, nd, M * steps, nd == 0 && rc == 0 ? "IDENTICAL" : "DIFFER", t1, t2,
                            100.0 * (t2 - t1) / t1);
                fails += !(nd == 0 && rc == 0);
                if (cnfhook) {   /* D: the library's own lever CNF entry (r4d_conf_cnf(1)) vs its conf_kernel */
                    c.t[5].data = do2; c.fix();
                    hc(hipMemset(do2, 0x7d, (size_t)M * steps * 4), "ms");
                    cnfhook(1);
                    const int rc2 = conf(&c.a, (RadStream)st);
                    hc(hipStreamSynchronize(st), "sync");
                    hc(hipMemcpy(o2.data(), do2, o2.size() * 4, hipMemcpyDeviceToHost), "cp");
                    int nd2 = 0;
                    for (size_t i = 0; i < o1.size(); ++i) nd2 += o1[i] != o2[i];
                    const double t3 = timed([&] { conf(&c.a, (RadStream)st); });
                    cnfhook(0);
                    c.t[5].data = do1; c.fix();
                    std::printf("D  R %d M %2d steps %d: libr4d CNF entry rc %d, %d of %d outputs differ from its conf_kernel -> %s; us a launch "
                                "back to back %.2f vs %.2f (%+.1f %%)\n", R, M, steps, rc2, nd2, M * steps, nd2 == 0 && rc2 == 0 ? "IDENTICAL" : "DIFFER",
                                t3, t1, 100.0 * (t3 - t1) / t1);
                    fails += !(nd2 == 0 && rc2 == 0);
                }
            }
        hipFree(dx); hipFree(dpr); hipFree(da); hipFree(dt); hipFree(dw); hipFree(do1); hipFree(do2);
    }
    std::printf("dpr: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
