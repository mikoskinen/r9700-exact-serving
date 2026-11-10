/* r4d_ds41_f32pf.hip -- lever GPF-b: gemm_nt_f32's prefill rows (the router and the compressor's fused kv | gate
 * projection: f32 logits from bf16 activations and f32 weights, or bf16 weights widened in the kernel under
 * RADIANCE_DS41_ROUTER_BF16) in a tiled form with the stock kernel's per-output arithmetic. Off unless switched on
 * (r4d_ds41_f32pf.h); gemm_f32_body (r4d_glm.hip) calls it ahead of the stock launch.
 *
 * ---- the stock kernel (r4d_glm.hip gemm_nt_f32_tile_kernel<8, 4, 1> above 63 rows)
 *
 *  A wave owns 8 rows x 4 columns; lane l folds k = l, l + 32, ... in ascending order (sum = fmaf(x, w, sum), x the bf16
 *  widened, w the f32 or the widened bf16), then the xor butterfly over 16, 8, 4, 2, 1 and lane 0 stores. Each k step a
 *  lane issues 8 activation and 4 weight loads (2 / 4-byte, one step in flight) for 32 fmas: profiled at the 128k
 *  prefill (dsm-225, rank 0) the router (N 96) runs 508 us a call and the compressor's kv | gate (N 1024) 4861 us, 7-8
 *  TFLOP/s, 31 ms of the 1206 ms step.
 *
 * ---- the form (f32pf_kernel<TM, TN, WM, WN, S, WT>)
 *
 *  Still a lane per k class (the butterfly needs the 32 classes of an output in one wave, lane = class). A wave owns
 *  TM rows x TN columns, WM x WN waves a block; a chunk of 32 S values of k is staged in LDS by the block in the
 *  lanes' order -- xs[row][class][S] (bf16) and ws[col][class][S] (f32, the bf16 storage widened exactly) -- so a lane
 *  reads its S values of a row with one 16-byte load and of a column with two, and every value it reads serves TN
 *  (TM) fmas. The next chunk's global loads are issued before the current chunk's fmas.
 *
 * ---- exactness: each output (row r, column n) is the same operations as the stock kernel's
 *
 *  Class l of (r, n) is folded by lane l of the wave owning (r, n): sum = fmaf(x[r][k], w[n][k], sum) from 0.0f over
 *  k = l, l + 32, ... ascending (chunks ascending, within a chunk the S steps ascending), x widened by the same shift
 *  (bf16w_lo / _hi = bf16_to_f32), w the same f32 (a bf16 weight widened by the same shift); no other operation touches
 *  the sum (-ffp-contract=off). Then sum = sum + partner(sum) over 16, 8, 4, 2, 1 (r4d_xor_lane: the shuffle's partner and
 *  operand order), and the result -- the same in every lane -- is stored once. Rows and columns past M / N read the last
 *  valid one and are not stored. K must be a multiple of 32 S (else the stock kernel runs). r4d_selftest --gpcheck
 *  byte-compares every form with the stock row at the DS shapes (both storages); tests/r4d_emu ds41pf runs both. */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_ds41_f32pf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

typedef __attribute__((address_space(1))) const float* f32pf_gf;

template <typename WT> __device__ __forceinline__ float f32pf_w(WT v);
template <> __device__ __forceinline__ float f32pf_w<float>(float v) { return v; }
template <> __device__ __forceinline__ float f32pf_w<uint16_t>(uint16_t v) { return bf16_to_f32(v); }
template <typename WT> __device__ __forceinline__ WT f32pf_ld(const WT* p);
template <> __device__ __forceinline__ float f32pf_ld<float>(const float* p) { return *(f32pf_gf)p; }
template <> __device__ __forceinline__ uint16_t f32pf_ld<uint16_t>(const uint16_t* p) { return (uint16_t)ldg_u16(p); }

template <int TM, int TN, int WM, int WN, int S, typename WT>
__global__ __launch_bounds__(32 * WM * WN) void f32pf_kernel(const uint16_t* x, const WT* w, float* y, long xp, long wp,
                                                             long yp, int m, int n, int k) {
    static_assert(S == 4 || S == 8, "a chunk is 4 or 8 steps of 32");
    constexpr int NTH = 32 * WM * WN, BM = TM * WM, BN = TN * WN;
    constexpr int XP = BM * 32 / NTH, WPN = BN * 32 / NTH;   /* (row, class) / (column, class) pairs a thread stages */
    static_assert(XP * NTH == BM * 32 && WPN * NTH == BN * 32, "the chunk's pairs split evenly over the block");
#ifdef R4D_EMU_HOST
    uint8_t* const smem = (uint8_t*)emu_dyn_lds();
#else
    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
#endif
    uint16_t* xs = (uint16_t*)(void*)smem;                       /* [BM][32][S] */
    float* ws = (float*)(void*)(smem + BM * 32 * S * 2);         /* [BN][32][S] */
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WM, wn = wave / WM;
    const int r0 = (int)blockIdx.y * BM, n0 = (int)blockIdx.x * BN;
    /* staging sources: pair p = tid + NTH j is (row or column p >> 5, class p & 31 = lane) */
    const uint16_t* xsrc[XP];
    const WT* wsrc[WPN];
#pragma unroll
    for (int j = 0; j < XP; ++j) {
        const int r = r0 + ((tid + NTH * j) >> 5);
        xsrc[j] = x + (long)(r < m ? r : m - 1) * xp + lane;
    }
#pragma unroll
    for (int j = 0; j < WPN; ++j) {
        const int c = n0 + ((tid + NTH * j) >> 5);
        wsrc[j] = w + (long)(c < n ? c : n - 1) * wp + lane;
    }
    uint16_t xr[XP][S];
    WT wr[WPN][S];
    auto gload = [&](int kc) {
#pragma unroll
        for (int j = 0; j < XP; ++j)
#pragma unroll
            for (int i = 0; i < S; ++i) xr[j][i] = (uint16_t)ldg_u16(xsrc[j] + kc + 32 * i);
#pragma unroll
        for (int j = 0; j < WPN; ++j)
#pragma unroll
            for (int i = 0; i < S; ++i) wr[j][i] = f32pf_ld<WT>(wsrc[j] + kc + 32 * i);
    };
    auto lstore = [&]() {
#pragma unroll
        for (int j = 0; j < XP; ++j) {
            const int p = tid + NTH * j;
            uint32_t u[S / 2];
#pragma unroll
            for (int i = 0; i < S / 2; ++i) u[i] = (uint32_t)xr[j][2 * i] | ((uint32_t)xr[j][2 * i + 1] << 16);
            uint32_t* d = (uint32_t*)(void*)(xs + (p >> 5) * 32 * S + lane * S);
            if constexpr (S == 8) {
                r4d_u32x4 v;
                v[0] = u[0]; v[1] = u[1]; v[2] = u[2]; v[3] = u[3];
                *(r4d_u32x4*)(void*)d = v;
            } else {
                d[0] = u[0];
                d[1] = u[1];
            }
        }
#pragma unroll
        for (int j = 0; j < WPN; ++j) {
            const int p = tid + NTH * j;
            float* d = ws + (p >> 5) * 32 * S + lane * S;
#pragma unroll
            for (int i = 0; i < S; i += 4) {
                r4d_u32x4 v;
#pragma unroll
                for (int e = 0; e < 4; ++e) v[e] = __float_as_uint(f32pf_w<WT>(wr[j][i + e]));
                *(r4d_u32x4*)(void*)(d + i) = v;
            }
        }
    };
    float sum[TM][TN];
#pragma unroll
    for (int q = 0; q < TM; ++q)
#pragma unroll
        for (int c = 0; c < TN; ++c) sum[q][c] = 0;
    const int KC = 32 * S, NC = k / KC;
    gload(0);
    lstore();
    __syncthreads();
    for (int ch = 0; ch < NC; ++ch) {
        if (ch + 1 < NC) gload((ch + 1) * KC);   /* the next chunk in flight behind this chunk's fmas */
        const uint16_t* xl = xs + (wm * TM) * 32 * S + lane * S;
        const float* wl = ws + (wn * TN) * 32 * S + lane * S;
        uint32_t xv[TM][S / 2];
#pragma unroll
        for (int q = 0; q < TM; ++q) {
            const uint32_t* s4 = (const uint32_t*)(const void*)(xl + q * 32 * S);
#pragma unroll
            for (int i = 0; i < S / 2; ++i) xv[q][i] = s4[i];
        }
#pragma unroll
        for (int i = 0; i < S; ++i) {
            float wv[TN];
#pragma unroll
            for (int c = 0; c < TN; ++c) wv[c] = wl[c * 32 * S + i];
#pragma unroll
            for (int q = 0; q < TM; ++q) {
                const float v = (i & 1) ? bf16w_hi(xv[q][i >> 1]) : bf16w_lo(xv[q][i >> 1]);
#pragma unroll
                for (int c = 0; c < TN; ++c) sum[q][c] = fmaf(v, wv[c], sum[q][c]);
            }
        }
        if (ch + 1 < NC) {
            __syncthreads();   /* every wave has read the chunk */
            lstore();
            __syncthreads();
        }
    }
    /* the butterfly: sum + partner over 16, 8, 4, 2, 1 (the shuffle loop's partner and operand order) */
#pragma unroll
    for (int q = 0; q < TM; ++q)
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            float s = sum[q][c];
            s = s + r4d_xor_lane<16>(s);
            s = s + r4d_xor_lane<8>(s);
            s = s + r4d_xor_lane<4>(s);
            s = s + r4d_xor_lane<2>(s);
            s = s + r4d_xor_lane<1>(s);
            sum[q][c] = s;
        }
    /* every lane holds every result now; lane e % 32 stores result e */
#pragma unroll
    for (int q = 0; q < TM; ++q)
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            const int e = q * TN + c, r = r0 + wm * TM + q, col = n0 + wn * TN + c;
            if ((e & 31) == lane && r < m && col < n) y[(long)r * yp + col] = sum[q][c];
        }
}

/* ---- the forms (code = 20 + index): (TM, TN, WM, WN, S). */
struct F32Form { int tm, tn, wm, wn, s; };
constexpr F32Form kF32Forms[] = { { 8, 8, 4, 2, 8 }, { 8, 8, 2, 4, 8 }, { 8, 4, 4, 2, 8 }, { 4, 8, 8, 1, 8 }, { 16, 4, 2, 4, 8 },
                                  { 8, 8, 4, 2, 4 } };
constexpr int kF32NForms = (int)(sizeof(kF32Forms) / sizeof(kF32Forms[0]));

template <int TM, int TN, int WM, int WN, int S>
int f32pf_launch_f(RadStream s, const R4dF32PfCall& c) {
    constexpr int BM = TM * WM, BN = TN * WN;
    if (c.K % (32 * S)) return R4D_F32PF_DECLINE;
    const dim3 grid((unsigned)((c.N + BN - 1) / BN), (unsigned)((c.M + BM - 1) / BM));
    if (grid.y > 65535u) return R4D_F32PF_DECLINE;
    const unsigned lds = (unsigned)(BM * 32 * S * 2 + BN * 32 * S * 4);
    if (c.wbf16)
        return R4D_EMIT(s, (f32pf_kernel<TM, TN, WM, WN, S, uint16_t>), grid, dim3(32 * WM * WN), lds, c.x,
                        (const uint16_t*)c.w, c.y, c.xp, c.wp, c.yp, c.M, c.N, c.K);
    return R4D_EMIT(s, (f32pf_kernel<TM, TN, WM, WN, S, float>), grid, dim3(32 * WM * WN), lds, c.x, (const float*)c.w,
                    c.y, c.xp, c.wp, c.yp, c.M, c.N, c.K);
}
int f32pf_launch(RadStream s, int form, const R4dF32PfCall& c) {
    switch (form) {
    case 0: return f32pf_launch_f<8, 8, 4, 2, 8>(s, c);
    case 1: return f32pf_launch_f<8, 8, 2, 4, 8>(s, c);
    case 2: return f32pf_launch_f<8, 4, 4, 2, 8>(s, c);
    case 3: return f32pf_launch_f<4, 8, 8, 1, 8>(s, c);
    case 4: return f32pf_launch_f<16, 4, 2, 4, 8>(s, c);
    case 5: return f32pf_launch_f<8, 8, 4, 2, 4>(s, c);
    default: return R4D_F32PF_DECLINE;
    }
}

bool f32pf_value_known(int v) { return v == 0 || v == 1 || (v >= 20 && v < 20 + kF32NForms); }

int f32pf_env() {
    const char* v = std::getenv("RADIANCE_DS41_F32_PF");
    if (!v || !*v) return 0;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (*end || !f32pf_value_known((int)x)) {
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_F32_PF=%s is not 0 / 1 / a form code 20..%d: the stock gemm_nt_f32\n",
                     v, 19 + kF32NForms);
        return 0;
    }
    if (x)
        std::fprintf(stderr, "radiance: libr4d: gemm_nt_f32's prefill rows take lever GPF-b's tiled form (RADIANCE_DS41_F32_PF=%ld, the fn4 switch f32_pf)\n", x);
    return (int)x;
}
std::atomic<int> g_f32pf_on{ f32pf_env() };
int g_f32pf_pin = 0;
char g_f32pf_last[96];

}  // namespace

int r4d_f32pf_on() { return g_f32pf_pin != 0 ? 1 : (g_f32pf_on.load(std::memory_order_relaxed) != 0); }
int r4d_f32pf_known(int v) { return f32pf_value_known(v) ? 1 : 0; }
int r4d_f32pf_set(int v) {
    if (!f32pf_value_known(v)) return -1;
    g_f32pf_on.store(v, std::memory_order_relaxed);
    return 1;
}
int r4d_f32pf_get() { return g_f32pf_on.load(std::memory_order_relaxed); }

/* The rule (gg 1): form 20 (8 x 8 a wave, 4 x 2 waves, 8 steps a chunk) above 63 rows; a pin takes any M. */
int r4d_f32pf_gemm(RadStream s, const R4dF32PfCall& c) {
    if (c.M < 1 || c.N < 1 || c.K < 1) return R4D_F32PF_DECLINE;
    int v = g_f32pf_pin != 0 ? g_f32pf_pin : g_f32pf_on.load(std::memory_order_relaxed);
    if (v == 0) return R4D_F32PF_DECLINE;
    if (g_f32pf_pin == 0 && c.M < 64) return R4D_F32PF_DECLINE;
    if (v == 1 || v == -1) v = 20;
    std::snprintf(g_f32pf_last, sizeof g_f32pf_last, "f32 form %d", v);
    return f32pf_launch(s, v - 20, c);
}

/* r4d_f32pf_pin (bench / test hook): 0 clears, a form code 20.. pins it at every M, -1 the rule's form at every M. */
extern "C" void r4d_f32pf_pin(int code) { g_f32pf_pin = (code == -1 || (code >= 20 && code < 20 + kF32NForms)) ? code : 0; }
extern "C" const char* r4d_f32pf_last() { return g_f32pf_last; }
extern "C" int r4d_f32pf_forms(int* out, int cap) {
    for (int i = 0; i < kF32NForms && i < cap; ++i) out[i] = 20 + i;
    return kF32NForms;
}
