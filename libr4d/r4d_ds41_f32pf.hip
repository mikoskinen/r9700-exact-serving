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

/* ---- the forms (code = 20 + index): (TM, TN, WM, WN, S). Codes 26.. (lever SPD-F, ds41-etp-spd) are row blocks of 8 / 16
 * for the decode band (a pinned or ds41_f32dec call at any M); the prefill rule never takes them. */
struct F32Form { int tm, tn, wm, wn, s; };
constexpr F32Form kF32Forms[] = { { 8, 8, 4, 2, 8 }, { 8, 8, 2, 4, 8 }, { 8, 4, 4, 2, 8 }, { 4, 8, 8, 1, 8 }, { 16, 4, 2, 4, 8 },
                                  { 8, 8, 4, 2, 4 },
                                  { 8, 4, 1, 4, 8 }, { 8, 8, 1, 4, 8 }, { 4, 8, 2, 4, 8 }, { 16, 4, 1, 4, 8 }, { 8, 4, 2, 4, 8 },
                                  { 4, 4, 2, 4, 8 } };
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
    /* lever SPD-F's decode-band row blocks (26..31) */
    case 6: return f32pf_launch_f<8, 4, 1, 4, 8>(s, c);
    case 7: return f32pf_launch_f<8, 8, 1, 4, 8>(s, c);
    case 8: return f32pf_launch_f<4, 8, 2, 4, 8>(s, c);
    case 9: return f32pf_launch_f<16, 4, 1, 4, 8>(s, c);
    case 10: return f32pf_launch_f<8, 4, 2, 4, 8>(s, c);
    case 11: return f32pf_launch_f<4, 4, 2, 4, 8>(s, c);
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

bool r4d_rgm_wants();   /* lever RGM (below): value 10 or the pin 40 needs gemm_nt_f32's calls through r4d_f32pf_gemm */
int r4d_f32pf_on() {
    return g_f32pf_pin != 0 ? 1 : (g_f32pf_on.load(std::memory_order_relaxed) != 0 || r4d_rgm_wants());
}
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
    /* lever RGM (rule B): the router's logits on WMMA at every M, ahead of every other rule */
    if (const int rc = r4d_rgm_route(s, c); rc != R4D_F32PF_DECLINE) return rc;
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

/* ==================================================================== lever SPD-F (ds41-etp-spd, 15.11.2026)
 * The decode band's (rows <= 63) f32 projections in wider forms (r4d_ds41_f32pf.h). The switch: 0 the stock rule,
 * 1 both, 2 the router's fused launch only, 3 gemm_nt_f32 only; RADIANCE_DS41_F32DEC at load. The bench / test pin
 * (r4d_f32dec_pin) names a form for every decode-band call of each kind (0 = the stock form there, -1 = not pinned: the
 * switch's rule), whatever the switch. Every form keeps each output's stock arithmetic: the same bytes. */
namespace {
int f32dec_env() {
    const char* v = std::getenv("RADIANCE_DS41_F32DEC");
    if (!v || !*v) return 0;
    const int x = std::atoi(v);
    return x >= 0 && x <= 11 ? x : 0;   /* 4 .. 7: tp8 loop D28; 8: lever RGP (DRAFTER round 17); 9: RGK; 10: RGM (round 40); 11: RGM + RGC */
}
std::atomic<int> g_f32dec{ f32dec_env() };
std::atomic<int> g_f32dec_pin_rgs{ -1 }, g_f32dec_pin_gemm{ -1 };
std::atomic<long long> g_f32dec_n[2] = { { 0 }, { 0 } };   /* calls a form other than the stock took: router, gemm */
}  // namespace

/* one stderr line the first time each (kind, height band) takes a non-stock form after a switch change (the engine log shows
 * the forms ran); band 0 <= 8 rows, 1 <= 16, 2 above */
static std::atomic<unsigned> g_f32dec_noted{ 0 };
static void f32dec_note(int kind, int rows, int form) {
    const unsigned bit = 1u << (kind * 3 + (rows <= 8 ? 0 : rows <= 16 ? 1 : 2));
    if (g_f32dec_noted.fetch_or(bit, std::memory_order_relaxed) & bit) return;
    std::fprintf(stderr, "radiance: libr4d: lever SPD-F (ds41_f32dec %d): %s takes form %d at %d rows (first such call)\n",
                 g_f32dec.load(std::memory_order_relaxed), kind ? "gemm_nt_f32" : "router_gemm_f32_scored_topk_scatter", form,
                 rows);
}

int r4d_f32dec_known(int v) { return v >= 0 && v <= 11 ? 1 : 0; }   /* 4 .. 7: tp8 loop D28; 8: lever RGP; 9: lever RGK; 10: lever RGM; 11: RGM + RGC */
int r4d_f32dec_set(int v) {
    if (!r4d_f32dec_known(v)) return -1;
    g_f32dec.store(v, std::memory_order_relaxed);
    g_f32dec_noted.store(0, std::memory_order_relaxed);
    return 1;
}
int r4d_f32dec_get() { return g_f32dec.load(std::memory_order_relaxed); }

/* The rule (switch 1 / 2 / 3): a form per height band -- rows <= 8, 9 .. 16, 17 .. 48 (the router) / 63 (gemm_nt_f32) -- by the
 * one-card bench (SPEED-D b1, 15.11.2026, r4d_selftest --case spdf32 on gpu02 03:00, us a call warm / cold):
 *   router (K 5120, 384 experts): <= 8 the stock 2x2 (no form beats it warm and cold); 9-16 form 3, 4x4 U8 16 waves (16 rows
 *     26.0 / 30.1 -> 22.4 / 28.8); 17-48 form 8, 4x8 U8 16 waves (32 rows 47.8 / 51.6 -> 33.8 / 41.0, 47 rows 96.2 / 100.0 ->
 *     51.8 / 54.6);
 *   gemm_nt_f32 (the compressor, N 1024): <= 8 the stock; 9-16 form 30, tiled 8x4 a wave, 2x4 waves (16 rows 40.4 / 50.0 ->
 *     25.9 / 32.2); 17-63 form 4, 4x8 U8 (32 rows 204.6 / 292.7 -> 41.0 / 51.5, 47 rows 304.8 / 420.2 -> 59.5 / 60.3).
 * RADIANCE_DS41_F32DEC_FORMS=r1,r2,r3,g1,g2,g3 at load replaces the table (an A/B knob; 0 = the stock form in that band; a code the
 * band's kind does not have = the stock form there). */
static int g_f32dec_rule[6] = { 0, 3, 8, 0, 30, 4 };
static bool f32dec_rule_load() {
    const char* e = std::getenv("RADIANCE_DS41_F32DEC_FORMS");
    if (!e || !*e) return true;
    int v[6], n = 0;
    const char* p = e;
    while (n < 6 && *p) {
        char* end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p || x < 0 || x > 99) break;
        v[n++] = (int)x;
        p = end;
        if (*p == ',') ++p;
    }
    if (n != 6 || *p) {
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_F32DEC_FORMS=%s is not six form codes: the built-in table\n", e);
        return true;
    }
    for (int i = 0; i < 6; ++i) g_f32dec_rule[i] = v[i];
    std::fprintf(stderr, "radiance: libr4d: lever SPD-F's table from RADIANCE_DS41_F32DEC_FORMS: router %d / %d / %d, gemm_nt_f32 %d / %d / %d\n",
                 v[0], v[1], v[2], v[3], v[4], v[5]);
    return true;
}
static const bool g_f32dec_rule_loaded = f32dec_rule_load();
static int f32dec_band(int rows) { return rows <= 8 ? 0 : rows <= 16 ? 1 : 2; }
static bool f32dec_rgs_code(int f) { return f >= 0 && f <= 20; }   /* 15 .. 18: tp8 loop D28's 2-D maps; 19 / 20: lever RGP */
static bool f32dec_gemm_code(int f) { return (f >= 0 && f <= 12) || (f >= 20 && f < 20 + kF32NForms); }
static int r4d_f32dec_rule_gemm(int rows, int n) {
    (void)g_f32dec_rule_loaded;
    /* Lever WQ-R (DRAFTER track, 18.11.2026 round 5): the table's forms were measured at the compressor's N 1024; a narrow
     * projection (N <= 128: the DSpark drafter's router logits, 128 experts) gets 8 / 4-40 workgroups from them. There the
     * 2x2 tile with 16 steps (form 9, the stock rule's own form up to 16 rows) is the fastest at every height to 48
     * (tools/dbench/fbench, xeon f1, busy card, N 128 K 5120 f32 weight, us): M 8 / 16 / 24 / 40 / 48 form 9 5.9 / 7.9 /
     * 9.5 / 13.8 / 16.9 vs the table's 26 / 30 / 4 / 4 / 4: 16.5-20.7 (form 30 at 16 rows 20.5 = lq145's served f32pf
     * 20.6-21.0 a stage). Every form the stock bytes (fbench: all 25 forms identical at every M). */
    if (n <= 128 && rows <= 48) return 9;
    const int f = g_f32dec_rule[3 + f32dec_band(rows)];
    return f32dec_gemm_code(f) ? f : 0;
}
static int r4d_f32dec_rule_rgs(int rows) {
    const int f = g_f32dec_rule[f32dec_band(rows)];
    return f32dec_rgs_code(f) ? f : 0;
}
int r4d_f32dec_gemm_form(int rows, int n) {
    if (rows > 63) return 0;
    const int p = g_f32dec_pin_gemm.load(std::memory_order_relaxed);
    int f = 0;
    if (p >= 0) f = p;
    else {
        const int v = g_f32dec.load(std::memory_order_relaxed);
        /* 4 .. 7: 1's gemm rule (D28); 8 (lever RGP) too -- F74 (DRAFTER round 20): without it RGP's value sent the compressor's
         * 17..63-row calls to the stock form (lq190: gemm_nt_f32_tile<8,1,8> 363-500 us vs form 4's 60-87, L +0.31 ms at c16);
         * 9 (lever RGK) too; 10 (lever RGM) for every shape but the router's (r4d_rgm_route took that one already) */
        if (v == 1 || v == 3 || (v >= 4 && v <= 11)) f = r4d_f32dec_rule_gemm(rows, n);
    }
    if (f > 0) {
        g_f32dec_n[1].fetch_add(1, std::memory_order_relaxed);
        if (p < 0) f32dec_note(1, rows, f);
    }
    return f;
}
int r4d_f32dec_rgs_form(int rows) {
    if (rows > 64) return 0;
    const int p = g_f32dec_pin_rgs.load(std::memory_order_relaxed);
    const int v = g_f32dec.load(std::memory_order_relaxed);
    /* above 48 rows (only under the core's RADIANCE_RGS_ROWS=64): a pin or lever RGP's value, else the stock form as before */
    if (rows > 48 && p < 0 && v != 8 && v != 9 && v != 10 && v != 11) return 0;
    int f = 0;
    if (p >= 0) f = p;
    else {
        /* lever RGP (DRAFTER track round 17): 8 = 1 with the router's 17..64 rows in form 19 (the selection a row group at a
         * time; the stock bytes) */
        if (v == 8) f = rows > 16 ? 19 : r4d_f32dec_rule_rgs(rows);
        /* lever RGK (DRAFTER track round 39): 9 = form 21 at 1..64 rows (form 19's map + the scores / keys in the GEMM epilogue; the
         * stock bytes; xeon card 1 r39t read-cold us a call vs 1's rule: 1 / 8 rows 24.0 / 24.9 vs 25.7 / 26.0, 16 24.9 vs 30.0,
         * 32 28.6 vs 41.2, 48 38.5 vs 56.0, 64 43.9 vs 91.4) */
        else if (v == 9) f = 21;
        /* lever RGM (DRAFTER track round 40, RULE B): 10 = form 22 at 1..64 rows (the WMMA logits; with gemm_nt_f32's code 40 at the
         * router's shape, every pass forms a router logit by the same function) */
        else if (v == 10 || v == 11) f = 22;
        else if (v == 1 || v == 2) f = r4d_f32dec_rule_rgs(rows);
        /* tp8 loop D28 (DECODE round 22): 4 .. 7 = 1 with the router's 17..48 rows in the 2-D wave map form 15 .. 18 (a workgroup
         * = row groups x expert groups, so each weight row is read by fewer workgroups; the stock bytes) */
        else if (v >= 4 && v <= 7) f = rows > 16 ? 11 + v : r4d_f32dec_rule_rgs(rows);
    }
    if (f > 0) {
        g_f32dec_n[0].fetch_add(1, std::memory_order_relaxed);
        if (p < 0) f32dec_note(0, rows, f);
    }
    return f;
}
int r4d_f32pf_gemm_code(RadStream s, const R4dF32PfCall& c, int code) {
    if (code == 40) return r4d_rgm_gemm(s, c);
    if (c.M < 1 || c.N < 1 || c.K < 1 || code < 20 || code >= 20 + kF32NForms) return R4D_F32PF_DECLINE;
    return f32pf_launch(s, code - 20, c);
}
/* bench / test hooks: pin the router's and gemm_nt_f32's decode-band forms (-1 = the switch's rule, 0 = the stock form);
 * the counts of calls that took a non-stock form (router, gemm), for GET /admin/fn4 and the tests */
extern "C" void r4d_f32dec_pin(int rgs_form, int gemm_form) {
    g_f32dec_pin_rgs.store(rgs_form, std::memory_order_relaxed);
    g_f32dec_pin_gemm.store(gemm_form, std::memory_order_relaxed);
}
extern "C" int rad_f32dec_stats(long long* out, int n) {
    if (!out || n < 1) return 2;
    out[0] = g_f32dec_n[0].load(std::memory_order_relaxed);
    if (n > 1) out[1] = g_f32dec_n[1].load(std::memory_order_relaxed);
    return 2;
}

/* lever RGM (DRAFTER track round 40, RULE B; the kernels and r4d_rgm_gemm in r4d_ds41_elem.hip): gemm_nt_f32 at the router's shape --
 * N 384 (every expert's logit) or 96 (glm_router's TP4 column split), K 5120, bf16 weights -- under ds41_f32dec 10 takes code 40 at
 * every M, so a decode row of a mixed pass, a prefill row and a decode-only pass's row (form 22) all get their logits from the same
 * function. The pin 40 (bench / test) takes every call code 40 can at any M. */
bool r4d_rgm_wants() {
    const int v = g_f32dec.load(std::memory_order_relaxed);
    return g_f32dec_pin_gemm.load(std::memory_order_relaxed) == 40 || v == 10 || v == 11;
}
int r4d_rgm_route(RadStream s, const R4dF32PfCall& c) {
    const int p = g_f32dec_pin_gemm.load(std::memory_order_relaxed);
    if (p == 40) return r4d_rgm_gemm(s, c);
    const int v = g_f32dec.load(std::memory_order_relaxed);
    /* 10: the router (N 384 / its 96-column split); 11 (lever RGC, round 40, RULE B too): + the ratio > 1 compressors' fused kv | gate
     * projection (N 1024, K 5120, bf16 under ROUTER_BF16) at every M -- its prefill rows and its decode-band calls alike */
    if (p >= 0 || (v != 10 && v != 11) || (c.N != 384 && c.N != 96 && !(v == 11 && c.N == 1024))) return R4D_F32PF_DECLINE;
    const int rc = r4d_rgm_gemm(s, c);
    if (rc != R4D_F32PF_DECLINE) {
        g_f32dec_n[1].fetch_add(1, std::memory_order_relaxed);
        f32dec_note(1, c.M, 40);
    }
    return rc;
}
