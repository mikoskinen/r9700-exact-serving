// kb2_m1.hip (12.11.2026, ds-main-port kbench builder) -- lever M1 (audit 4): DeepSeek-V4.1's one-pass mHC in prefill on
// one card. The int7 source (ds41-int7 0391a6e libr4d/r4d_ds41_elem.hip, copied verbatim to kb/src) is included whole:
// STK = post_collapse_norm_body / mhc_mix_body (the stock launches); LIB = the image's libr4d.so through
// r4d_rad_mhc_post_collapse_rmsnorm / r4d_rad_mhc_mix (+ its scratch).
//
// The pair (arch/deepseek41/deepseek41.cpp 1871-1880, 4600-4625): pcn_a = mhc_post_collapse_rmsnorm(act.x, h1, post,
// comb, pre_f -> h0, xcol, xn) and then mix_a = mhc_mix(h0 -> pre_a, post, comb), n 5120, hc 4 (h rows 20480 bf16),
// 24 dots + the sum of squares, 20 Sinkhorn passes, eps 1e-20 (rms_norm_eps), hc_eps 1e-6. Inputs: post / comb / pre_f
// from LIB mhc_mix over a random h1 with blk.9's hc_ffn weights; mix_a's weights and pcn's gain are blk.10's
// hc_attn_* / attn_norm (real tensors read from the 2.25 .rad into kb/data; random only if --random-weights).
//
// F1 (the fused prototype, three launches): K1 = mhc_post + mhc_collapse (pcn's column arithmetic) for R rows x a
// 1024-column block of n, the post'd streams kept in LDS a 128-column slab at a time and mhc_mix's projection + sum of
// squares formed from them in mix_split_kernel's exact order (wave w the dots w, w+8, w+16 and wave 0 the squares, lane
// L columns L, L+32, .. of each 1024-column chunk in order from zero, r4d_wave_sum, the chunk partials into mix's
// scratch layout); then the stock mix_finish_kernel over those partials; K2 = the norm over the stored xcol in pcn's
// order (virtual thread v of r4d_act_threads(M, n) adds columns v, v + THR, .. from zero; r4d_wave_sum; the wave sums
// from zero in order: r4d_ds41_pcn.h's pcn_arrive). Exact by construction; the byte check holds it to LIB.
#include "r4d_ds41_elem.hip"
#include "kb2_common.h"

#include <cmath>
#include <functional>

const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;
unsigned int* r4d_8a8_split_ctr(RadStream, const void*) { return nullptr; }
extern "C" int r4d_ds41_entry_store_mx4(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

namespace {

constexpr int kHC = 4, kMIX = kHC * (kHC + 2), kS = kMIX + 1, kCB = 1024, kSlab = 128;

struct M1Args {
    const uint16_t* xin; long xp;    /* [M, n] act.x */
    const uint16_t* h;   long hp;    /* [M, 4n] streams before */
    const float* post;   long ptp;   /* [M, 4] */
    const float* comb;   long cbp;   /* [M, 16] */
    const float* pre;    long pp;    /* [M, 4] the collapse weights (pre_f) */
    uint16_t* out;       long op;    /* [M, 4n] streams after (h0) */
    uint16_t* xc;        long xcp;   /* [M, n] */
    const float* fn;     long fp;    /* [24, 4n] mix_a's projection */
    float* part;                     /* [M][25][chunks] mix's scratch layout */
    int n, chunks;
    long M;
};

/* K1: grid (n / 1024 column blocks, ceil(M / R) row groups), 256 threads. Phase B's split: wave w forms the partials
 * of stream k = w / 2 (its chunk k * (n / 1024) + b): dots (w & 1) + 2 q, q < 12, and the squares on (w & 1) == 0.
 * Each partial is one lane's own sequence over the chunk's 32 steps in step order, as mix_split_kernel forms it, so
 * which wave holds it does not change its bits. */
template <int R>
__global__ __launch_bounds__(256) void m1f_k1_kernel(M1Args A, int* flag, int vid) {
    constexpr int WV = 8, NCG = WV / R, CPW = kSlab / NCG, UA = CPW / 32, JD = kMIX / 2;
    static_assert(kHC == 4 && WV == 2 * kHC, "two waves a stream");
    static_assert(WV % R == 0 && CPW % 32 == 0, "R divides the 8 waves");
    __shared__ uint16_t ob[R][kHC][kSlab];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *flag = vid;
    const int b = blockIdx.x, n = A.n;
    const long g0 = (long)blockIdx.y * R;
    const int nr = (int)(A.M - g0 < R ? A.M - g0 : R);
    /* phase A's thread: one row (wave % R), a column group of the slab */
    const int ra = wave % R, cg = wave / R;
    const long ma = g0 + (ra < nr ? ra : 0);
    float pj[kHC], pk[kHC], cb[kHC][kHC];
#pragma unroll
    for (int k = 0; k < kHC; ++k) {
        pj[k] = A.pre[ma * A.pp + k];
        pk[k] = A.post[ma * A.ptp + k];
#pragma unroll
        for (int j = 0; j < kHC; ++j) cb[j][k] = A.comb[ma * A.cbp + j * kHC + k];
    }
    /* phase B's wave: stream kw, dots par + 2 q, the squares when par == 0 */
    const int kw = wave >> 1, par = wave & 1;
    const bool sq = par == 0;
    const float* wr = A.fn + (long)par * A.fp + (long)kw * n;
    float acc[JD + 1][R];
#pragma unroll
    for (int q = 0; q <= JD; ++q)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[q][r] = 0.0f;
#pragma unroll 1
    for (int sb = 0; sb < kCB / kSlab; ++sb) {
        const int c0 = b * kCB + sb * kSlab;
        /* A: mhc_post + mhc_collapse at (ra, columns c0 + cg CPW + lane + 32 u) -- post_collapse_norm_kernel's
         * column, the same expressions */
        if (ra < nr) {
            const uint16_t* hr = A.h + ma * A.hp;
#pragma unroll
            for (int u = 0; u < UA; ++u) {
                const int cl = cg * CPW + lane + 32 * u, i = c0 + cl;
                uint32_t hv[kHC];
#pragma unroll
                for (int j = 0; j < kHC; ++j) hv[j] = hr[(long)j * n + i];
                const uint32_t xi = A.xin[ma * A.xp + i];
                float v = 0;
#pragma unroll
                for (int k = 0; k < kHC; ++k) {
                    float uu = 0;
#pragma unroll
                    for (int j = 0; j < kHC; ++j) uu += cb[j][k] * bf16_to_f32((uint16_t)hv[j]);
                    uu += pk[k] * bf16_to_f32((uint16_t)xi);
                    const uint16_t o = f32_to_bf16(uu);
                    A.out[ma * A.op + (long)k * n + i] = o;
                    ob[ra][k][cl] = o;
                    v += pj[k] * ds_bf(o);
                }
                A.xc[ma * A.xcp + i] = ds_bf16(v);
            }
        }
        __syncthreads();
        /* B: mix's projection of stream kw over the slab's 4 steps of 32 columns */
#pragma unroll
        for (int u = 0; u < kSlab / 32; ++u) {
            const int cl = 32 * u + lane, i = c0 + cl;
            float v[R];
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = r < nr ? ds_bf(ob[r][kw][cl]) : 0.0f;
            float w[JD];
#pragma unroll
            for (int q = 0; q < JD; ++q) w[q] = wr[(long)(2 * q) * A.fp + i];
#pragma unroll
            for (int q = 0; q < JD; ++q)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[q][r] += v[r] * w[q];
            if (sq)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[JD][r] += v[r] * v[r];
        }
        __syncthreads();
    }
    /* the chunk partials: chunk kw * (n / 1024) + b of the flattened 4n row */
    const int chunk = kw * (n / kCB) + b;
#pragma unroll
    for (int q = 0; q <= JD; ++q) {
        if (q == JD && !sq) continue;
        const int j = q < JD ? par + 2 * q : kMIX;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = r4d_wave_sum(acc[q][r]);
            if (!lane && r < nr) A.part[((g0 + r) * kS + j) * A.chunks + chunk] = s;
        }
    }
}

/* K2: the norm over the stored xcol, a workgroup of 256 a row, in pcn's order (pcn_arrive's virtual threads) */
__global__ __launch_bounds__(256) void m1f_k2_kernel(const uint16_t* xc, long xcp, const float* w, uint16_t* y, long yp,
                                                     int n, int tv, float eps, long M, int* flag, int vid) {
    __shared__ float vss[1024];
    __shared__ float wsum[32];
    if (blockIdx.x == 0 && threadIdx.x == 0) *flag = vid;
    const long m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* xr = xc + m * xcp;
    for (int v = (int)threadIdx.x; v < tv; v += 256) {
        float ss = 0.0f;
        for (int i = v; i < n; i += tv) {
            const float xv = bf16_to_f32(xr[i]);
            ss += xv * xv;
        }
        vss[v] = ss;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nwv = tv / 32;
    for (int vw = wave; vw < nwv; vw += 8) {
        const float t = r4d_wave_sum(vss[vw * 32 + lane]);
        if (lane == 0) wsum[vw] = t;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int q = 0; q < nwv; ++q) sum += wsum[q];
    const float sc = rsqrtf(sum / (float)n + eps);
    uint16_t* yr = y + m * yp;
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * sc * (w[i] + 0.0f) * 1.0f);
}

}  // namespace

/* ======================================================================== host */
namespace {
using namespace kb;

constexpr int kN = 5120, kW = kHC * kN, kIters = 20;
constexpr float kEps = 1e-20f, kHcEps = 1e-6f;

bool read_f32(const std::string& path, std::vector<float>& v, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    v.resize(n);
    const size_t got = std::fread(v.data(), 4, n, f);
    std::fclose(f);
    return got == n;
}

struct Weights {
    std::vector<float> fn9, base9, scale9, fn10, base10, scale10, norm10;
    bool real = false;
};
bool load_weights(const std::string& dir, bool random, Weights& w) {
    if (!random) {
        bool ok = read_f32(dir + "/blk9_hc_ffn_fn.f32", w.fn9, (size_t)kMIX * kW) &&
                  read_f32(dir + "/blk9_hc_ffn_base.f32", w.base9, kMIX) &&
                  read_f32(dir + "/blk9_hc_ffn_scale.f32", w.scale9, 3) &&
                  read_f32(dir + "/blk10_hc_attn_fn.f32", w.fn10, (size_t)kMIX * kW) &&
                  read_f32(dir + "/blk10_hc_attn_base.f32", w.base10, kMIX) &&
                  read_f32(dir + "/blk10_hc_attn_scale.f32", w.scale10, 3) &&
                  read_f32(dir + "/blk10_attn_norm.f32", w.norm10, kN);
        w.real = ok;
        return ok;
    }
    Rng g(77);
    auto fill = [&](std::vector<float>& v, size_t n, double sc, double off) {
        v.resize(n);
        for (auto& x : v) x = (float)(g.gauss() * sc + off);
    };
    fill(w.fn9, (size_t)kMIX * kW, 0.01, 0);
    fill(w.base9, kMIX, 0.5, 0);
    fill(w.scale9, 3, 0.1, 0.5);
    fill(w.fn10, (size_t)kMIX * kW, 0.01, 0);
    fill(w.base10, kMIX, 0.5, 0);
    fill(w.scale10, 3, 0.1, 0.5);
    fill(w.norm10, kN, 0.2, 1.0);
    w.real = false;
    return true;
}

struct Bufs {
    long M = 0;
    uint16_t *h1, *xin, *h0, *xc, *y, *h0f, *xcf, *yf;
    float *pre_f, *post, *comb;          /* the previous mix's (pcn's inputs) */
    float *pre_a, *post_a, *comb_a;      /* LIB mix_a's */
    float *pre_s, *post_s, *comb_s;      /* the form under check */
    float *part, *scratch;
    int64_t scratch_bytes = 0;
};

}  // namespace

int main(int argc, char** argv) {
    const char* libp = "/opt/radiance/share/radiance/kernels/libr4d.so";
    std::string wdir = "/kbdata";
    bool random = false;
    int reps = 30, warm = 3;
    std::vector<long> rows_list = { 4096, 8, 1 };
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--lib") && i + 1 < argc) libp = argv[++i];
        else if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) wdir = argv[++i];
        else if (!std::strcmp(argv[i], "--random-weights")) random = true;
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf("kb2_m1 [--lib PATH] [--weights DIR] [--random-weights] [--reps 30]\n");
            return 0;
        }
    }
    hipDeviceProp_t prop;
    KB_CK(hipGetDeviceProperties(&prop, 0));
    std::printf("KB2 DEV %s %s cus=%d clock_khz=%d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.clockRate);
    Weights W;
    if (!load_weights(wdir, random, W)) {
        std::printf("KB2 VIRHE M1 weights not readable in %s (blk9_hc_ffn_*, blk10_hc_attn_*, blk10_attn_norm .f32)\n",
                    wdir.c_str());
        return 2;
    }
    std::printf("KB2 M1W weights=%s scale10=%.6g,%.6g,%.6g base10[0]=%.6g norm10[0]=%.6g\n", W.real ? "real" : "random",
                W.scale10[0], W.scale10[1], W.scale10[2], W.base10[0], W.norm10[0]);
    Lib lib;
    LaunchFn lib_mix = nullptr, lib_pcn = nullptr;
    ScratchFn lib_mix_scr = nullptr;
    if (lib.open(libp)) {
        lib_mix = lib.sym<LaunchFn>("r4d_rad_mhc_mix");
        lib_pcn = lib.sym<LaunchFn>("r4d_rad_mhc_post_collapse_rmsnorm");
        lib_mix_scr = lib.sym<ScratchFn>("r4d_rad_mhc_mix_scratch");
    }
    if (!lib_mix || !lib_pcn || !lib_mix_scr) {
        std::printf("KB2 VIRHE LIB mhc entry points not loaded from %s\n", libp);
        return 2;
    }
    hipStream_t s;
    KB_CK(hipStreamCreate(&s));
    Cmper cmp;
    int* flag = dalloc<int>(4);
    int nvirhe = 0;
    float *fn9 = dup(W.fn9), *base9 = dup(W.base9), *scale9 = dup(W.scale9);
    float *fn10 = dup(W.fn10), *base10 = dup(W.base10), *scale10 = dup(W.scale10), *norm10 = dup(W.norm10);
    attr("m1_pcn_stock<f32,256,4>", post_collapse_norm_kernel<true, 256, 4>, 256, 0);
    attr("m1_mix_split<4,8,8,4>", mix_split_kernel<4, 8, 8, 4>, 256, 0);
    attr("m1_mix_finish<4>", mix_finish_kernel<4>, 128, 0);
    attr("m1f_k1<8>", m1f_k1_kernel<8>, 256, 0);
    attr("m1f_k1<4>", m1f_k1_kernel<4>, 256, 0);
    attr("m1f_k2", m1f_k2_kernel, 256, 0);

    for (long M : rows_list) {
        Rng g(1000 + (uint64_t)M);
        Bufs B;
        B.M = M;
        std::vector<uint16_t> h1((size_t)M * kW), xin((size_t)M * kN);
        for (long r = 0; r < M; ++r) {
            const double rs = std::ldexp(1.0, (int)(g.next() % 3) - 1);
            for (long i = 0; i < kW; ++i) h1[(size_t)r * kW + i] = h_bf16((float)(g.gauss() * rs));
            for (long i = 0; i < kN; ++i) xin[(size_t)r * kN + i] = h_bf16((float)(g.gauss() * 0.3));
        }
        B.h1 = dup(h1);
        B.xin = dup(xin);
        B.h0 = dalloc<uint16_t>((size_t)M * kW);
        B.h0f = dalloc<uint16_t>((size_t)M * kW);
        B.xc = dalloc<uint16_t>((size_t)M * kN);
        B.xcf = dalloc<uint16_t>((size_t)M * kN);
        B.y = dalloc<uint16_t>((size_t)M * kN);
        B.yf = dalloc<uint16_t>((size_t)M * kN);
        B.pre_f = dalloc<float>((size_t)M * kHC);
        B.post = dalloc<float>((size_t)M * kHC);
        B.comb = dalloc<float>((size_t)M * kHC * kHC);
        B.pre_a = dalloc<float>((size_t)M * kHC);
        B.post_a = dalloc<float>((size_t)M * kHC);
        B.comb_a = dalloc<float>((size_t)M * kHC * kHC);
        B.pre_s = dalloc<float>((size_t)M * kHC);
        B.post_s = dalloc<float>((size_t)M * kHC);
        B.comb_s = dalloc<float>((size_t)M * kHC * kHC);
        const int chunks = (int)mix_chunks(kW);
        B.part = dalloc<float>((size_t)M * kS * chunks);

        auto mix_args = [&](uint16_t* h, float* fn, float* sc, float* bs, float* pre, float* post, float* comb,
                            std::vector<RadTensor>& t, std::vector<RadParam>& p) {
            t = { tens(h, RAD_BF16, { M, kW }), tens(fn, RAD_F32, { kMIX, kW }), tens(sc, RAD_F32, { 3 }),
                  tens(bs, RAD_F32, { kMIX }), tens(pre, RAD_F32, { M, kHC }), tens(post, RAD_F32, { M, kHC }),
                  tens(comb, RAD_F32, { M, kHC * kHC }) };
            p = { pint("M", M), pint("n", kN), pint("hc", kHC), pint("iterations", kIters), pf64("eps", (double)kEps),
                  pf64("hc_eps", (double)kHcEps), pstr("dtype", "bf16") };
        };
        std::vector<RadTensor> tm9, tma, tms, tpc, tps;
        std::vector<RadParam> pm9, pma, pms, ppc, pps;
        mix_args(B.h1, fn9, scale9, base9, B.pre_f, B.post, B.comb, tm9, pm9);
        RadArgs am9 = args(tm9, pm9);
        B.scratch_bytes = lib_mix_scr(&am9);
        B.scratch = dalloc<float>((size_t)(B.scratch_bytes / 4 + 64));
        am9.scratch = B.scratch;
        am9.scratch_bytes = B.scratch_bytes;
        /* pcn's inputs: the previous connection's gates, from LIB mix over h1 */
        int rc = lib_mix(&am9, (RadStream)s);
        KB_CK(hipStreamSynchronize(s));
        if (rc != RAD_OK) {
            std::printf("KB2 VIRHE M1 LIB mix(h1) rc=%d rows=%ld\n", rc, M);
            ++nvirhe;
            continue;
        }
        /* LIB pcn: h1 -> h0, xc, y; LIB mix_a: h0 -> pre_a, post_a, comb_a */
        tpc = { tens(B.xin, RAD_BF16, { M, kN }), tens(B.h1, RAD_BF16, { M, kW }), tens(B.post, RAD_F32, { M, kHC }),
                tens(B.comb, RAD_F32, { M, kHC * kHC }), tens(B.h0, RAD_BF16, { M, kW }), tens(B.pre_f, RAD_F32, { M, kHC }),
                tens(B.xc, RAD_BF16, { M, kN }), tens(norm10, RAD_F32, { kN }), tens(B.y, RAD_BF16, { M, kN }) };
        ppc = { pint("M", M), pint("n", kN), pint("hc", kHC), pf64("eps", (double)kEps), pstr("dtype", "bf16") };
        RadArgs apc = args(tpc, ppc);
        mix_args(B.h0, fn10, scale10, base10, B.pre_a, B.post_a, B.comb_a, tma, pma);
        RadArgs ama = args(tma, pma, B.scratch, B.scratch_bytes);
        rc = lib_pcn(&apc, (RadStream)s);
        const int rc2 = lib_mix(&ama, (RadStream)s);
        KB_CK(hipStreamSynchronize(s));
        if (rc != RAD_OK || rc2 != RAD_OK) {
            std::printf("KB2 VIRHE M1 LIB pcn rc=%d mix rc=%d rows=%ld\n", rc, rc2, M);
            ++nvirhe;
            continue;
        }
        /* the forms under check write h0f / xcf / yf and pre_s / post_s / comb_s */
        tps = tpc;
        tps[4].data = B.h0f;
        tps[6].data = B.xcf;
        tps[8].data = B.yf;
        RadArgs aps = args(tps, ppc);
        mix_args(B.h0f, fn10, scale10, base10, B.pre_s, B.post_s, B.comb_s, tms, pms);
        RadArgs ams = args(tms, pms, B.scratch, B.scratch_bytes);
        MixOut mo;
        mo.scale = scale10;
        mo.sst = 1;
        mo.base = base10;
        mo.bst = 1;
        mo.pre = B.pre_s;
        mo.prp = kHC;
        mo.prs = 1;
        mo.post = B.post_s;
        mo.pop = kHC;
        mo.pos = 1;
        mo.comb = B.comb_s;
        mo.cp = kHC * kHC;
        mo.cs = 1;
        mo.width = kW;
        mo.iters = kIters;
        mo.eps = kEps;
        mo.heps = kHcEps;
        M1Args A;
        A.xin = B.xin; A.xp = kN; A.h = B.h1; A.hp = kW; A.post = B.post; A.ptp = kHC; A.comb = B.comb;
        A.cbp = kHC * kHC; A.pre = B.pre_f; A.pp = kHC; A.out = B.h0f; A.op = kW; A.xc = B.xcf; A.xcp = kN;
        A.fn = fn10; A.fp = kW; A.part = B.part; A.n = kN; A.chunks = chunks; A.M = M;
        const int tv = r4d_act_threads((int)M, kN);
        auto k1 = [&](int R) {
            if (R == 8) m1f_k1_kernel<8><<<dim3(kN / kCB, (unsigned)((M + 7) / 8)), 256, 0, s>>>(A, flag, 61);
            else m1f_k1_kernel<4><<<dim3(kN / kCB, (unsigned)((M + 3) / 4)), 256, 0, s>>>(A, flag, 62);
        };
        auto fin = [&] { mix_finish_kernel<4><<<dim3((unsigned)((M + 3) / 4)), 128, 0, s>>>(B.part, chunks, mo, M); };
        auto k2 = [&] { m1f_k2_kernel<<<dim3((unsigned)M), 256, 0, s>>>(B.xcf, kN, norm10, B.yf, kN, kN, tv, kEps, M, flag + 1, 63); };

        struct Form {
            std::string name;
            int vid0, vid1;   /* run flags: flag[0], flag[1] (0 = none) */
            bool pcn, mix;    /* which outputs it forms */
            std::function<void()> launch;
        };
        std::vector<Form> forms = {
            { "LIB_PCN", 0, 0, true, false, [&] { lib_pcn(&aps, (RadStream)s); } },
            { "LIB_MIX", 0, 0, false, true, [&] { lib_mix(&ams, (RadStream)s); } },
            { "STK_PCN", 0, 0, true, false, [&] { post_collapse_norm_body(&aps, (RadStream)s); } },
            { "STK_MIX", 0, 0, false, true, [&] { mhc_mix_body(&ams, (RadStream)s); } },
            { "LIB_PAIR", 0, 0, true, true, [&] { lib_pcn(&aps, (RadStream)s); lib_mix(&ams, (RadStream)s); } },
            { "F1R8", 61, 63, true, true, [&] { k1(8); fin(); k2(); } },
            { "F1R4", 62, 63, true, true, [&] { k1(4); fin(); k2(); } },
            { "F1R8_K1", 61, 0, false, false, [&] { k1(8); } },
            { "F1R4_K1", 62, 0, false, false, [&] { k1(4); } },
            { "F1_FIN", 0, 0, false, false, [&] { fin(); } },
            { "F1_K2", 0, 63, false, false, [&] { k2(); } },
        };
        std::printf("KB2 M1CASE rows=%ld n=%d hc=%d width=%d chunks=%d pcn_thr=%d scratch_bytes=%lld weights=%s\n", M, kN, kHC,
                    kW, chunks, tv, (long long)B.scratch_bytes, W.real ? "real" : "random");
        for (auto& f : forms) {
            const bool check = f.pcn || f.mix;
            if (!check) continue;
            /* the mix forms read h0f: LIB's h0 unless the form makes its own */
            if (f.mix && !f.pcn) KB_CK(hipMemcpyAsync(B.h0f, B.h0, (size_t)M * kW * 2, hipMemcpyDeviceToDevice, s));
            KB_CK(hipMemsetAsync(B.xcf, 0xa5, (size_t)M * kN * 2, s));
            KB_CK(hipMemsetAsync(B.yf, 0xa5, (size_t)M * kN * 2, s));
            if (f.pcn) KB_CK(hipMemsetAsync(B.h0f, 0xa5, (size_t)M * kW * 2, s));
            KB_CK(hipMemsetAsync(B.pre_s, 0xa5, (size_t)M * kHC * 4, s));
            KB_CK(hipMemsetAsync(B.post_s, 0xa5, (size_t)M * kHC * 4, s));
            KB_CK(hipMemsetAsync(B.comb_s, 0xa5, (size_t)M * kHC * kHC * 4, s));
            KB_CK(hipMemsetAsync(flag, 0, 8, s));
            f.launch();
            KB_CK(hipGetLastError());
            KB_CK(hipStreamSynchronize(s));
            int fl[2] = { 0, 0 };
            KB_CK(hipMemcpy(fl, flag, 8, hipMemcpyDeviceToHost));
            const bool ran = (f.vid0 == 0 || fl[0] == f.vid0) && (f.vid1 == 0 || fl[1] == f.vid1);
            if (!ran) {
                std::printf("KB2 VIRHE M1CHK %s rows=%ld run flags %d,%d (want %d,%d): the form did not run\n", f.name.c_str(), M,
                            fl[0], fl[1], f.vid0, f.vid1);
                ++nvirhe;
                continue;
            }
            std::string res;
            unsigned long long tot = 0, compared = 0;
            auto add = [&](const char* nm, Cmp c) {
                char b[160];
                std::snprintf(b, sizeof b, " %s=%llu/%llu", nm, c.diff, c.compared);
                res += b;
                tot += c.diff;
                compared += c.compared;
            };
            if (f.pcn) {
                add("h0", cmp.halfs(B.h0, B.h0f, M * kW, s));
                add("xcol", cmp.halfs(B.xc, B.xcf, M * kN, s));
                add("xn", cmp.halfs(B.y, B.yf, M * kN, s));
            }
            if (f.mix) {
                add("pre", cmp.words(B.pre_a, B.pre_s, kHC, kHC, M, kHC, s));
                add("post", cmp.words(B.post_a, B.post_s, kHC, kHC, M, kHC, s));
                add("comb", cmp.words(B.comb_a, B.comb_s, kHC * kHC, kHC * kHC, M, kHC * kHC, s));
            }
            std::printf("KB2 M1CHK %s rows=%ld compared_bytes=%llu diff_elems=%llu verdict=%s flags=%s (diff/bytes:%s)\n",
                        f.name.c_str(), M, compared, tot, tot == 0 ? "SAME" : "DIFFERS", f.vid0 || f.vid1 ? "ran" : "n/a",
                        res.c_str());
            if (tot) ++nvirhe;
            std::fflush(stdout);
        }
        /* timing (the mix forms read LIB's h0 copy in h0f) */
        KB_CK(hipMemcpyAsync(B.h0f, B.h0, (size_t)M * kW * 2, hipMemcpyDeviceToDevice, s));
        const int rp = M >= 1024 ? reps : reps * 3;
        for (auto& f : forms) {
            const Tm t = time_it(s, warm, rp, f.launch);
            const double bytes = (double)M * (f.pcn && !f.mix ? (kN * 2.0 * 3 + kW * 2.0 * 2)
                                              : (!f.pcn && f.mix) ? kW * 2.0 : (kN * 2.0 * 3 + kW * 2.0 * 2));
            std::printf("KB2 M1T %s rows=%ld med_us=%.1f p10_us=%.1f p90_us=%.1f n=%d gbs=%.1f\n", f.name.c_str(), M, t.med,
                        t.p10, t.p90, t.n, bytes / (t.med * 1e-6) / 1e9);
            std::fflush(stdout);
        }
        void* ps[] = { B.h1, B.xin, B.h0, B.xc, B.y, B.h0f, B.xcf, B.yf, B.pre_f, B.post, B.comb, B.pre_a, B.post_a,
                       B.comb_a, B.pre_s, B.post_s, B.comb_s, B.part, B.scratch };
        for (void* p : ps) KB_CK(hipFree(p));
    }
    std::printf("KB2 M1DONE virhe=%d\n", nvirhe);
    return nvirhe ? 1 : 0;
}
