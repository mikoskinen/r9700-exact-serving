/* gsk (DRAFTER track, tp8 loop 18.11.2026) -- one card: the DSpark drafter's dense q8_0 linears at a drafting pass's rows (M 16 /
 * 40 / 80) in a split-K WMMA form, to measure the prize before any serving change. The served forms at these rows (WMK's W prefill
 * form 414 / 214 above 48 rows, W decode below) run one 16-column tile a wave over the whole K: at N 512-1536 that is 32-96 tiles
 * x a few row tiles for 64 CUs, so they stream the weight at 90-190 GB/s (r4d_selftest --perfwkpf / --perfwmg, LOG 20:10 / 20:24).
 *
 * gsk_kernel<RT, S>: a block = one 16-column tile (16 weight rows) x RT 16-row tiles of x, S waves, wave s over the K blocks
 * [s nbw, (s + 1) nbw); a wave reads its 16 rows' K range in chunks of 4 q8_0 blocks (136 contiguous bytes a row: two lanes a row,
 * 17 aligned dwords each, the next chunk in registers while the current one is computed), unpacks them into LDS as [block][row]
 * 32 int8 + an f32 scale, and per block runs 2 RT v_wmma_f32_16x16x16_bf16 (A = x rows as bf16 from global / L2, B = the 16 rows'
 * int8 as exact bf16), the block's partial from 0 and one fma of its scale into the f32 accumulator. The S partials are summed in
 * LDS in wave order 0 .. S - 1 by one thread an output, rounded once to bf16.
 * Not the served order: rule B (deterministic, a row's bytes independent of the other rows) -- checked here: repeat, rows reversed,
 * the first 16 rows alone vs inside M; and vs an f64 CPU dot on sampled outputs (bf16 rounding only).
 *
 * usage: gsk [--M 16,40,80] [--iters 200] [--shapes all|NxK,...] [--S s]   (S 0: per shape, the default)
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v8f wmma(u4 a, u4 b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), c);
}
/* four int8 (a dword) -> four exact bf16 (two dwords) */
__device__ __forceinline__ void i8x4_bf(uint32_t w, uint32_t* lo, uint32_t* hi) {
    const float f0 = (float)(int)(int8_t)(w & 0xffu), f1 = (float)(int)(int8_t)((w >> 8) & 0xffu);
    const float f2 = (float)(int)(int8_t)((w >> 16) & 0xffu), f3 = (float)(int)(int8_t)(w >> 24);
    *lo = __builtin_amdgcn_perm(__float_as_uint(f1), __float_as_uint(f0), 0x07060302u);
    *hi = __builtin_amdgcn_perm(__float_as_uint(f3), __float_as_uint(f2), 0x07060302u);
}
__device__ __forceinline__ float h2f(uint32_t h) { return (float)__builtin_bit_cast(_Float16, (uint16_t)h); }

struct Args {
    const uint16_t* x; const uint8_t* w; uint16_t* y;
    int M, N, K, row_bytes;
};

template <int RT, int S>
__global__ __launch_bounds__(32 * S) void gsk_kernel(Args A) {
    constexpr int CH = 4;                            /* blocks a chunk */
    __shared__ __align__(16) uint32_t qs[S][CH][16][8];   /* [wave][block][row][32 int8] */
    __shared__ float ds[S][CH][16];
    __shared__ float red[S][256];                    /* one 16 x 16 tile of partials a wave */
    const int lane = (int)threadIdx.x & 31, wv = (int)threadIdx.x >> 5;
    const int n0 = (int)blockIdx.x * 16, m0 = (int)blockIdx.y * 16 * RT;
    const int nb = A.K / 32, nbw = nb / S, b0 = wv * nbw;
    /* loader role: row lr = lane / 2, half lh = lane & 1 -> 17 dwords of the row's 136-byte chunk */
    const int lr = lane >> 1, lh = lane & 1;
    const int wrow = min(n0 + lr, A.N - 1);
    const uint32_t* wsrc = (const uint32_t*)(A.w + (long long)wrow * A.row_bytes);
    /* compute role: column n = lane % 16 (weight row), K half h */
    const int nl = lane & 15, h = lane >> 4;
    uint32_t pre[17];
    auto load = [&](int c) {
        const uint32_t* p = wsrc + ((b0 + c * CH) * 34) / 4 + lh * 17;
#pragma unroll
        for (int i = 0; i < 17; ++i) pre[i] = __builtin_nontemporal_load(p + i);
    };
    auto unpack = [&]() {   /* my two blocks (2 lh, 2 lh + 1) of row lr into LDS */
        const int ja = 2 * lh, jb = 2 * lh + 1;
        ds[wv][ja][lr] = h2f(pre[0] & 0xffffu);
#pragma unroll
        for (int i = 0; i < 8; ++i) qs[wv][ja][lr][i] = __builtin_amdgcn_alignbyte(pre[i + 1], pre[i], 2);
        ds[wv][jb][lr] = h2f(pre[8] >> 16);
#pragma unroll
        for (int i = 0; i < 8; ++i) qs[wv][jb][lr][i] = pre[9 + i];
    };
    v8f acc[RT];
#pragma unroll
    for (int t = 0; t < RT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;
    const int nch = nbw / CH;
    load(0);
    for (int c = 0; c < nch; ++c) {
        unpack();
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        if (c + 1 < nch) load(c + 1);
#pragma unroll 1
        for (int j = 0; j < CH; ++j) {
            const int kb = (b0 + c * CH + j) * 32;   /* the block's first K */
            /* B: the 16 rows' values K 8h .. 8h + 7 (WMMA 0) and 16 + 8h .. (WMMA 1) */
            const uint32_t q0 = qs[wv][j][nl][2 * h], q1 = qs[wv][j][nl][2 * h + 1];
            const uint32_t q2 = qs[wv][j][nl][4 + 2 * h], q3 = qs[wv][j][nl][4 + 2 * h + 1];
            const float d = ds[wv][j][nl];
            uint32_t c[8];
            i8x4_bf(q0, &c[0], &c[1]); i8x4_bf(q1, &c[2], &c[3]);
            i8x4_bf(q2, &c[4], &c[5]); i8x4_bf(q3, &c[6], &c[7]);
            const u4 bw0 = { c[0], c[1], c[2], c[3] }, bw1 = { c[4], c[5], c[6], c[7] };
            const v8f z = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int t = 0; t < RT; ++t) {
                const int m = m0 + 16 * t + nl;
                u4 a0 = { 0u, 0u, 0u, 0u }, a1 = { 0u, 0u, 0u, 0u };
                if (m < A.M) {
                    const u4* xp = (const u4*)(A.x + (long long)m * A.K + kb + 8 * h);
                    a0 = xp[0];
                    a1 = xp[2];   /* + 16 values */
                }
                v8f P = wmma(a0, bw0, z);
                P = wmma(a1, bw1, P);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[t][i] = __builtin_fmaf(d, P[i], acc[t][i]);
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
    /* the S partials, tile by tile, summed in wave order; D lane L: column L % 16, rows 8 (L / 16) + i */
#pragma unroll
    for (int t = 0; t < RT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) red[wv][(8 * h + i) * 16 + nl] = acc[t][i];
        __syncthreads();
        for (int o = (int)threadIdx.x; o < 256; o += 32 * S) {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < S; ++w) s += red[w][o];
            const int m = m0 + 16 * t + o / 16, n = n0 + o % 16;
            if (m < A.M && n < A.N) {
                const uint32_t u = __float_as_uint(s);
                A.y[(long long)m * A.N + n] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            }
        }
        __syncthreads();
    }
}

static int launch(int rt, int S, const Args& A, hipStream_t s) {
    const dim3 grid((unsigned)((A.N + 15) / 16), (unsigned)((A.M + 16 * rt - 1) / (16 * rt)));
#define L(R, SS) if (rt == R && S == SS) { hipLaunchKernelGGL((gsk_kernel<R, SS>), grid, dim3(32 * SS), 0, s, A); return 0; }
#define LS(R) L(R, 1) L(R, 2) L(R, 4) L(R, 5) L(R, 8) L(R, 10) L(R, 16)
    LS(1) LS(2) LS(3) LS(4) LS(5)
#undef LS
#undef L
    return -1;
}

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static float h2f_host(uint16_t h) { _Float16 v; std::memcpy(&v, &h, 2); return (float)v; }

struct Shape { const char* name; int N, K; };
static const Shape kShapes[] = {
    { "q_a", 1280, 5120 }, { "kv", 512, 5120 }, { "o_a", 1024, 4096 }, { "shexp_gu", 1536, 5120 }, { "q_b", 8192, 1280 },
    { "o_b", 5120, 2048 }, { "shexp_dn", 5120, 768 },
};
static int pick_S(int N, int K) {   /* waves a block: enough K splits for ~4 waves a SIMD at the drafter's N */
    const int nb = K / 32;
    int want = N <= 768 ? 16 : N <= 2048 ? 8 : 2;
    for (int s = want; s >= 1; --s)
        if ((s == 1 || s == 2 || s == 4 || s == 5 || s == 8 || s == 10 || s == 16) && nb % (4 * s) == 0) return s;
    return 1;
}

int main(int argc, char** argv) {
    std::string ms = "16,40,80", shapes = "all";
    int iters = 200, Sfix = 0, RTfix = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--M") ms = nx();
        else if (a == "--iters") iters = std::atoi(nx().c_str());
        else if (a == "--shapes") shapes = nx();
        else if (a == "--S") Sfix = std::atoi(nx().c_str());
        else if (a == "--RT") RTfix = std::atoi(nx().c_str());
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    std::vector<int> Ml;
    for (size_t p = 0; p < ms.size();) { size_t q = ms.find(',', p); if (q == std::string::npos) q = ms.size(); Ml.push_back(std::atoi(ms.substr(p, q - p).c_str())); p = q + 1; }
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    for (const Shape& sh : kShapes) {
        if (shapes != "all" && shapes.find(sh.name) == std::string::npos) continue;
        const int N = sh.N, K = sh.K, nb = K / 32, rb = nb * 34;
        const size_t wbytes = (size_t)N * rb;
        std::vector<uint8_t> hw(wbytes);
        uint64_t rs = 0x9E3779B97F4A7C15ull ^ (uint64_t)(N * 131 + K);
        auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
        for (int64_t b = 0; b < (int64_t)N * nb; ++b) {
            uint8_t* p = hw.data() + b * 34;
            const _Float16 d = (_Float16)((float)(1 + rnd() % 1000) * 1e-5f);
            std::memcpy(p, &d, 2);
            for (int i = 0; i < 32; ++i) p[2 + i] = (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
        }
        const int copies = (int)std::max<size_t>(2, (256ull << 20) / wbytes + 1);
        uint8_t* dw = nullptr; CK(hipMalloc((void**)&dw, wbytes * copies));
        for (int c = 0; c < copies; ++c) CK(hipMemcpy(dw + c * wbytes, hw.data(), wbytes, hipMemcpyHostToDevice));
        const int S = Sfix > 0 ? Sfix : pick_S(N, K);
        if (nb % (4 * S)) { std::printf("%s: S %d does not divide %d blocks in chunks of 4\n", sh.name, S, nb); continue; }
        for (int M : Ml) {
            std::vector<uint16_t> hx((size_t)M * K);
            for (auto& v : hx) v = f2bf((float)((int)(rnd() % 2001) - 1000) / 500.0f);
            uint16_t *dx, *dy, *dy2; CK(hipMalloc((void**)&dx, hx.size() * 2)); CK(hipMalloc((void**)&dy, (size_t)M * N * 2)); CK(hipMalloc((void**)&dy2, (size_t)M * N * 2));
            CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            const int rt = RTfix > 0 ? std::min(RTfix, (M + 15) / 16) : std::min(5, (M + 15) / 16);
            Args A{ dx, dw, dy, M, N, K, rb };
            if (launch(rt, S, A, s)) { std::printf("%s M %d: no form RT %d S %d\n", sh.name, M, rt, S); continue; }
            CK(hipStreamSynchronize(s));
            std::vector<uint16_t> y0((size_t)M * N), y1((size_t)M * N);
            CK(hipMemcpy(y0.data(), dy, y0.size() * 2, hipMemcpyDeviceToHost));
            /* rule B: (1) repeat (2) rows reversed (3) the first 16 rows alone (M 16) -- bytes of each row equal */
            int rb_fail = 0;
            { Args B = A; B.y = dy2; launch(rt, S, B, s); CK(hipStreamSynchronize(s)); CK(hipMemcpy(y1.data(), dy2, y1.size() * 2, hipMemcpyDeviceToHost)); rb_fail += y1 != y0; }
            {
                std::vector<uint16_t> hr(hx.size());
                for (int m = 0; m < M; ++m) std::memcpy(&hr[(size_t)m * K], &hx[(size_t)(M - 1 - m) * K], (size_t)K * 2);
                uint16_t* dr; CK(hipMalloc((void**)&dr, hr.size() * 2)); CK(hipMemcpy(dr, hr.data(), hr.size() * 2, hipMemcpyHostToDevice));
                Args B = A; B.x = dr; B.y = dy2; launch(rt, S, B, s); CK(hipStreamSynchronize(s));
                CK(hipMemcpy(y1.data(), dy2, y1.size() * 2, hipMemcpyDeviceToHost));
                for (int m = 0; m < M; ++m) if (std::memcmp(&y1[(size_t)m * N], &y0[(size_t)(M - 1 - m) * N], (size_t)N * 2)) { ++rb_fail; break; }
                hipFree(dr);
            }
            if (M > 16) {
                Args B = A; B.M = 16; B.y = dy2; launch(1, S, B, s); CK(hipStreamSynchronize(s));
                CK(hipMemcpy(y1.data(), dy2, (size_t)16 * N * 2, hipMemcpyDeviceToHost));
                if (std::memcmp(y1.data(), y0.data(), (size_t)16 * N * 2)) ++rb_fail;
            }
            /* values: sampled outputs vs an f64 dot of the same q8_0 values */
            double maxrel = 0, sd = 0, sr = 0;
            for (int k = 0; k < 2000; ++k) {
                const int m = (int)(rnd() % M), n = (int)(rnd() % N);
                double ref = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* p = hw.data() + (size_t)n * rb + b * 34;
                    uint16_t dh; std::memcpy(&dh, p, 2);
                    double part = 0;
                    for (int i = 0; i < 32; ++i) part += (double)(int8_t)p[2 + i] * bf2f(hx[(size_t)m * K + b * 32 + i]);
                    ref += part * h2f_host(dh);
                }
                const double got = bf2f(y0[(size_t)m * N + n]);
                sd += (got - ref) * (got - ref); sr += ref * ref;
                maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 1e-30));
            }
            /* timing: weights rotated over `copies` (cold past 256 MiB), events around `iters` launches */
            for (int w = 0; w < 20; ++w) { Args B = A; B.w = dw + (size_t)(w % copies) * wbytes; launch(rt, S, B, s); }
            CK(hipEventRecord(e0, s));
            for (int it = 0; it < iters; ++it) { Args B = A; B.w = dw + (size_t)(it % copies) * wbytes; launch(rt, S, B, s); }
            CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
            float ms_ = 0; CK(hipEventElapsedTime(&ms_, e0, e1));
            const double us = ms_ * 1000.0 / iters;
            std::printf("GSK %-9s %5dx%-5d M %3d RT %d S %2d: %7.1f us  %5.0f GB/s  %5.1f TF/s | rule B %s | vs f64 rms rel %.2e max rel %.2e\n",
                        sh.name, N, K, M, rt, S, us, (double)wbytes / (us * 1e3), 2.0 * M * N * K / (us * 1e6), rb_fail ? "FAIL" : "ok",
                        sr > 0 ? std::sqrt(sd / sr) : 0.0, maxrel);
            std::fflush(stdout);
            if (rb_fail) ++fails;
            hipFree(dx); hipFree(dy); hipFree(dy2);
        }
        hipFree(dw);
    }
    std::printf("GSKVERDICT %s\n", fails ? "RULE B FAIL" : "rule B ok on every (shape, M)");
    return fails ? 1 : 0;
}
