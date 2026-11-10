// R4D -- DeepSeek-V4.1's sparse ops. Each kernel rounds where its libref function does
// (libref/ref_sparse_kv.cpp) and sums in libref's order, so the outputs are libref's bits:
//
//   index_score       one thread an entry (or HS waves an entry group, each a slice of the heads,
//                     the head sum then closed in head order from LDS): the dot in d order,
//                     rounded to bf16, relu x the rounded head weight, rounded, the head sum in head
//                     order, rounded. With `keep` (index_candidates' bits, the candidate restriction)
//                     an entry outside the kept blocks scores -inf without a key read, and every
//                     kept entry is computed by the same instructions as without keep (same bits).
//   index_candidates  one 1024-thread block a row: block maxima, the newest block pinned, an
//   index_topk        exact radix select of the k largest (value desc, position asc: libref's
//                     partial_sort order), then an ordered compaction. index_topk at decode
//                     heights over long rows splits a row into chunks (each keeps its exact top-k,
//                     which holds every global winner) and merges them in a second launch.
//   attn_sparse_sink  one workgroup a (d half, head group, row): every score q.k in d order (keys
//                     transposed through a wave-private LDS tile so the loads stay coalesced), the
//                     max, p = exp(s - max) by glibc's expf algorithm (the same double table and
//                     polynomial, so p is std::exp's float), sum p and sum p k in key order, then
//                     exp(sink - max), the division and one bf16 rounding.
//
// q.k and q_h.k_j multiply two bf16 values, exact in f32, so a fused multiply-add rounds exactly
// where libref's multiply-then-add does; p k is not exact and stays an unfused multiply and add.
//
// Prefill heights (more than kWmRows = 48 rows a pass) take WMMA forms instead, which are NOT
// libref's bits (kbench tolerance): attn_sparse_sink with an online softmax, v_exp_f32 and P as
// bf16 hi + lo; index_score with only the dot's summation order changed (every rounding after it
// is libref's). The decode band keeps the exact kernels, so speculative verify reproduces decode.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_ds41_qrt.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"

#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

/* The ops' dynamic LDS. tests/r4d_emu runs these sources on the host, which has no extern
 * __shared__: there it is the launch's dynamic LDS, read at each use. */
#ifdef R4D_EMU_HOST
#define r4d_ds41_sparse_dyn ((unsigned char*)emu_dyn_lds())
#else
extern __shared__ __align__(16) unsigned char r4d_ds41_sparse_dyn[];
#endif

namespace {

constexpr float kDsInf = __builtin_huge_valf();

/* ================================================================== scalar helpers */

/* rad_f32_to_bf16 (abi/rad_plugin.h): NaN quieted, everything else to nearest even. */
__device__ __forceinline__ uint32_t ds_bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x0040u) & 0xffffu;
    return ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) & 0xffffu;
}
__device__ __forceinline__ float ds_round_bf16(float f) { return __uint_as_float(ds_bf16_bits(f) << 16); }
__device__ __forceinline__ float ds_bf(uint32_t h) { return __uint_as_float((h & 0xffffu) << 16); }

/* libref's seq_of_row: the s with cu[s] <= row < cu[s+1], by bisection. */
__device__ __forceinline__ int ds_seq_of_row(const int* cu, long cs, int S, long row) {
    int lo = 0, hi = S;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if ((long)cu[(long)mid * cs] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* glibc's expf (sysdeps/ieee754/flt-32/e_expf.c, EXP2F_TABLE_BITS 5): 2^(k/32) from a table and a
 * cubic for 2^(r/32), in double, rounded once to float. A host replica of exactly these steps
 * matched libm's expf on 685k arguments (subnormal and overflow edges included). */
__constant__ unsigned long long kDsExp2fTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
__device__ __forceinline__ float ds_expf(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {                          /* |x| >= 88, or inf/nan */
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;            /* x * 32 / ln 2 */
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = kDsExp2fTab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An order-preserving image of a float with -0 folded onto +0 (libref's comparator calls them
 * equal). A candidate is > -inf, so its key is > 0x007fffff; 0 marks "not a candidate". */
__device__ __forceinline__ uint32_t ds_key(float v) {
    const uint32_t u = __float_as_uint(v + 0.0f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ void ds_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/* ================================================================== index_score */
/* q, w and pos start at the run's first row (row_part); row0 is that row's index in cu_seqlens.
 * M is the score's rows (the pass height, which picks the form), Mv how many of them the step
 * holds: a row from Mv on is past the step and scores -inf without a read. */
struct DsScoreArgs {
    const uint16_t* q;
    const uint16_t* w;
    const uint16_t* kc;
    const int* bt;
    const int* pos;
    const int* cu;
    float* sc;
    /* keep (optional): index_candidates' bits, row m for score row m, a bit a block of `block`
     * positions (bshift = log2(block) for a power of two, else -1); null scores every entry. */
    const int* keep;
    long qr, wr, wc, btr, btc, nblk, ps, cs, sr, scs, row0, kr, kcs;
    int nbt, S, M, Mv, C, H, ratio, cbs, slot, block, bshift;
    int kmx;                                         /* key_format mxfp4: the cache holds packed keys */
    float wscale;
};

/* ---- MXFP4 index keys (index_score key_format mxfp4, entry_store format mxfp4): an n-value key as
 * n/2 bytes of e2m1 codes (value 2i in byte i's low nibble, 2i+1 in its high nibble; bit 3 the
 * sign), then n/32 ue8m0 scale bytes (a block of 32 values each, scale 2^(e-127)), then zeros to a
 * multiple of 16 bytes: 80 bytes (40 bf16 slots) for the 128-value key. The scale is quant_roundtrip's
 * (fp4_e2m1, ue8m0, block 32): the power of two at or above max(amax, 6 x 2^-126) x f32(1/6). For
 * values that rule already produced (every index key deepseek41 stores) the codes are exact and the
 * unpacked bf16 is the stored bf16, bit for bit: such a block is c_i 2^k with c_i e2m1 codes, and
 * the rule on it picks 2^k / 2^j with c_max 2^j in (3, 6], where every |c_i| <= c_max times 2^j is
 * again a code (docs/OPS.md). */
__host__ __device__ constexpr int ds_mx4_bytes(int n) { return (n / 2 + n / 32 + 15) / 16 * 16; }
constexpr int kDsMx4Bytes = ds_mx4_bytes(128);       /* 80 */

/* bf16 bits of e2m1 code c (bit 3 the sign) times 2^(e-127), e >= 1 (the scale floor): exponent
 * e + (m >> 1) - 1 for magnitude code m = 1..7 (0.5, 1, 1.5, 2, 3, 4, 6), mantissa 0x40 at m = 3, 5,
 * 7; 0.5 x 2^-126 is bf16's denormal 0x0040. */
__host__ __device__ __forceinline__ uint32_t ds_mx4_bf16(uint32_t c, uint32_t e) {
    const uint32_t m = c & 7u, s = (c & 8u) << 12;
    if (m == 0u) return s;
    const uint32_t E = e + (m >> 1) - 1u;
    if (E == 0u) return s | 0x40u;
    return s | (E << 7) | ((m & 1u) && m > 1u ? 0x40u : 0u);
}
/* Eight codes (a word, value i at bits 4i) at scale byte e -> four bf16 pairs (value 2t low). */
__device__ __forceinline__ uint4 ds_mx4_unpack8(uint32_t w, uint32_t e) {
    uint32_t o[4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
        o[t] = ds_mx4_bf16((w >> (8 * t)) & 15u, e) | (ds_mx4_bf16((w >> (8 * t + 4)) & 15u, e) << 16);
    return make_uint4(o[0], o[1], o[2], o[3]);
}
/* ---- the packed compressed latent (attn_sparse_sink latent_format nvfp4, entry_store format nvfp4):
 * an n-value entry as n/2 bytes of e2m1 codes (value 2i in byte i's low nibble, 2i+1 in its high
 * nibble, bit 3 the sign), then n/16 OCP e4m3 scale bytes (a block of 16 values each), then zeros to
 * a multiple of 16 bytes: 288 bytes (144 bf16 slots) for the 512-value latent. A value is
 * e2m1(c) x e4m3(s) in f32, exact, and its bf16 is the high half: the latent DeepSeek-V4 stores is
 * quant_roundtrip(fp4_e2m1, e4m3, 16), every value c_i s exactly, so unpacked it is the stored bf16
 * bit for bit. The packer (ds41_entry_store_nv4_kernel) searches the block's scale among amax / c,
 * c = 6, 4, 3, 2, 1.5, 1, 0.5, for one that is an e4m3 value and gives every value back exactly (the
 * block's own scale is always among them; re-deriving it by the rule is not exact where e4m3 is
 * subnormal), and falls back to the rule (lossy) for values the rule did not produce. */
__host__ __device__ constexpr int ds_nv4_bytes(int n) { return (n / 2 + n / 16 + 15) / 16 * 16; }
constexpr int kDsNv4Bytes = ds_nv4_bytes(512);          /* 288: the 512-value latent */

/* OCP e4m3 (fn) byte -> f32, exact (rad_fp8e4m3_to_f32's rule; 0x7f / 0xff NaN). */
__host__ __device__ __forceinline__ float ds_e4m3_f32(uint32_t b) {
    const uint32_t sg = (b & 0x80u) << 24, e = (b >> 3) & 15u;
    uint32_t m = b & 7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = sg;
        } else {
            int sh = 0;
            while (!(m & 8u)) { m <<= 1; ++sh; }
            u = sg | ((uint32_t)(127 - 6 - sh) << 23) | ((m & 7u) << 20);
        }
    } else if (e == 15u && m == 7u) {
        u = sg | 0x7fc00000u;
    } else {
        u = sg | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __builtin_bit_cast(float, u);
}
/* f32 -> OCP e4m3, round to nearest even, saturating at 448 (rad_f32_to_fp8e4m3's rule). */
__host__ __device__ __forceinline__ uint32_t ds_f32_e4m3(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f), sg = (u >> 24) & 0x80u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return sg | 0x7fu;
    if (a >= 0x43e00000u) return sg | 0x7eu;
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return sg;
    if (e < -6) {
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        uint32_t q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return sg | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    if (q == 8u) { q = 0u; ++ee; }
    if (ee > 15 || (ee == 15 && q == 7u)) return sg | 0x7eu;
    return sg | ((uint32_t)ee << 3) | q;
}
/* bf16 bits of e2m1 code c (bit 3 the sign) times scale s (an e4m3 value): the f32 product's high
 * half, exact for every value the roundtrip stored. */
__host__ __device__ __forceinline__ uint32_t ds_nv4_bf16(uint32_t c, float s) {
    const uint32_t m = c & 7u;
    const float mag = m < 4u ? 0.5f * (float)m : (float)(2u << ((m - 4u) >> 1)) * ((m & 1u) ? 1.5f : 1.0f);
    return (__builtin_bit_cast(uint32_t, mag * s) >> 16) | ((c & 8u) << 12);
}
/* Eight codes (a word, value i at bits 4i), one 16-block's scale s -> four bf16 pairs. */
__device__ __forceinline__ uint4 ds_nv4_unpack8(uint32_t w, float s) {
    uint32_t o[4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
        o[t] = ds_nv4_bf16((w >> (8 * t)) & 15u, s) | (ds_nv4_bf16((w >> (8 * t + 4)) & 15u, s) << 16);
    return make_uint4(o[0], o[1], o[2], o[3]);
}

/* ---- the decode kernels' unpack (8.11.): branch-free, about six instructions a value. A value is e2m1(c) x s,
 * exact in f32; it is computed as (2 |e2m1(c)|) x (s / 2): 2 |e2m1| is an integer 0..12 from a nibble table (0xC8643210:
 * 0 1 2 3 4 6 8 12), s / 2 is exact (ds_e4m3_half), so the product is the f32 the reference's mag x s is -- and its bf16,
 * the high half, the stored bit -- with the code's sign bit or-ed in as ds_nv4_bf16 does. All 256 scale bytes x 16 codes
 * give ds_nv4_bf16's bits (NaN scales: NaN). ds_e4m3_f32's subnormal loop and ds_nv4_bf16's selects compiled to some 180
 * branches and 27 instructions a value in the decode kernels' unrolled loops: 240 us a call against bf16's 15 at one row. */
/* 8.11. (v3): the magnitude bits of b at bit 20 are an f32 whose value is e4m3's times 2^-120 -- normal and subnormal
 * alike, since the f32 subnormal field lines up with e4m3's (exponent 0: m 2^-3 2^-126 = m 2^-9 2^-120) -- so one exact
 * multiply by 2^119 gives s / 2 (f32 subnormals are kept: denorm mode 3). The same bits as the select form it replaces
 * for all 256 bytes (NaN 0x7f / 0xff -> 0x7fc00000 | sign), in about seven instructions instead of fifteen; the decode
 * kernels run it once a scale byte (16 values in the scores, 2 in the direct PV). */
__device__ __forceinline__ float ds_e4m3_half(uint32_t b) {
    const uint32_t mg = (b << 20) & 0x07f00000u;
    uint32_t u = __builtin_bit_cast(uint32_t, __builtin_bit_cast(float, mg) * 0x1p119f);
    u = mg == 0x07f00000u ? 0x7fc00000u : u;
    return __builtin_bit_cast(float, u | ((b & 0x80u) << 24));
}
/* The f32 of code c (its low four bits; bit 3 the sign) at half scale sh. */
__device__ __forceinline__ float ds_nv4_val(uint32_t c, float sh) {
    const float v = (float)((0xC8643210u >> ((c & 7u) << 2)) & 15u) * sh;
    return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, v) | ((c & 8u) << 28));
}
/* The same for the code at bits [T, T + 4) of w (T a constant): a bit-field extract, the table's field, a convert, a
 * multiply and the sign -- eight instructions. */
template <int T>
__device__ __forceinline__ float ds_nv4_at(uint32_t w, float sh) {
    const uint32_t m4 = ((w >> T) & 7u) << 2;
    const float v = (float)((0xC8643210u >> m4) & 15u) * sh;
    const uint32_t sg = T + 3 <= 31 ? (w >> (T + 3)) << 31 : 0u;
    return __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, v) | sg);
}
/* ds_nv4_unpack8 by ds_nv4_val: eight codes (a word) at half scale sh -> four bf16 pairs, the same bits. */
__device__ __forceinline__ uint4 ds_nv4_unpack8_fast(uint32_t w, float sh) {
    uint32_t o[4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
        o[t] = (__builtin_bit_cast(uint32_t, ds_nv4_val(w >> (8 * t), sh)) >> 16) |
               (__builtin_bit_cast(uint32_t, ds_nv4_val(w >> (8 * t + 4), sh)) & 0xffff0000u);
    return make_uint4(o[0], o[1], o[2], o[3]);
}

/* e2m1's code of v (quant_roundtrip's e2m1_code): nearest magnitude, ties to the even code, past 6
 * the largest; NaN 0. */
__host__ __device__ __forceinline__ uint32_t ds_e2m1_code(float v) {
    if (v != v) return 0u;
    const float a = v < 0.0f ? -v : v;
    const uint32_t m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return (__builtin_bit_cast(uint32_t, v) >> 28 & 8u) | m;
}

/* keep's block of position j. */
__device__ __forceinline__ long ds_keep_block(const DsScoreArgs& a, long j) {
    return a.bshift >= 0 ? (j >> a.bshift) : j / a.block;
}
/* Is position j kept: bit b % 32 of word b / 32 of the row, b = j's block. */
__device__ __forceinline__ bool ds_kept(const DsScoreArgs& a, const int* krow, long j) {
    const long b = ds_keep_block(a, j);
    return (((unsigned)krow[(b >> 5) * a.kcs] >> (b & 31)) & 1u) != 0u;
}
/* Position j's index key in row m's sequence s, or null (past the row's reach or C, an unmapped or
 * out-of-range block). */
__device__ __forceinline__ const uint16_t* ds_entry_key(const DsScoreArgs& a, int s, long j, long reach) {
    if (j >= a.C || j >= reach) return nullptr;
    const long col = (j * a.ratio) / a.cbs;
    if (col >= a.nbt) return nullptr;
    const long blk = a.bt[(long)s * a.btr + col * a.btc];
    if (blk < 0 || blk >= a.nblk) return nullptr;
    return a.kc + (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
}

/* One entry's score from its key kp (null: none): the dot in d order rounded to bf16, relu x the
 * rounded head weight, rounded, summed in head order -- this wave's heads [h0, h1); with HS > 1 the
 * waves' slices meet in `part` ([H][EPB]) and wave sub 0 closes the sum in head order. Called
 * block-uniformly (HS > 1 holds two barriers). Sub 0's result is the entry's f32 sum (before the
 * final rounding). */
template <int HS>
__device__ __forceinline__ float ds_score_entry(const DsScoreArgs& a, long m, const uint16_t* kp, float* part,
                                                int e, int sub, int h0, int h1) {
    constexpr int DM = 128, EPB = 256 / HS;
    float acc = 0.0f;
    if (kp) {
        float k[DM];
        /* mxfp4: 64 code bytes then 4 scale bytes, unpacked to the bf16 the rule stored (same floats) */
        [[maybe_unused]] const uint32_t sc4 = a.kmx ? *(const uint32_t*)((const unsigned char*)kp + DM / 2) : 0u;
#pragma unroll
        for (int c = 0; c < DM / 8; ++c) {
            const uint4 v = a.kmx ? ds_mx4_unpack8(((const uint32_t*)kp)[c], (sc4 >> (8 * (c / 4))) & 0xffu)
                                  : *(const uint4*)(kp + c * 8);
            k[c * 8 + 0] = bf16w_lo(v.x); k[c * 8 + 1] = bf16w_hi(v.x);
            k[c * 8 + 2] = bf16w_lo(v.y); k[c * 8 + 3] = bf16w_hi(v.y);
            k[c * 8 + 4] = bf16w_lo(v.z); k[c * 8 + 5] = bf16w_hi(v.z);
            k[c * 8 + 6] = bf16w_lo(v.w); k[c * 8 + 7] = bf16w_hi(v.w);
        }
        const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr);
        const uint16_t* wrow = a.w + m * a.wr;
        for (int h = h0; h < h1; h += 4) {
            const int nh = min(4, h1 - h);
            const uint32_t* q0 = qrow + (long)h * (DM / 2);
            const uint32_t* q1 = qrow + (long)(nh > 1 ? h + 1 : h) * (DM / 2);
            const uint32_t* q2 = qrow + (long)(nh > 2 ? h + 2 : h) * (DM / 2);
            const uint32_t* q3 = qrow + (long)(nh > 3 ? h + 3 : h) * (DM / 2);
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
            for (int i = 0; i < DM / 2; ++i) {
                const uint32_t w0 = q0[i], w1 = q1[i], w2 = q2[i], w3 = q3[i];
                d0 = __builtin_fmaf(bf16w_lo(w0), k[2 * i], d0);
                d1 = __builtin_fmaf(bf16w_lo(w1), k[2 * i], d1);
                d2 = __builtin_fmaf(bf16w_lo(w2), k[2 * i], d2);
                d3 = __builtin_fmaf(bf16w_lo(w3), k[2 * i], d3);
                d0 = __builtin_fmaf(bf16w_hi(w0), k[2 * i + 1], d0);
                d1 = __builtin_fmaf(bf16w_hi(w1), k[2 * i + 1], d1);
                d2 = __builtin_fmaf(bf16w_hi(w2), k[2 * i + 1], d2);
                d3 = __builtin_fmaf(bf16w_hi(w3), k[2 * i + 1], d3);
            }
            const float dd[4] = { d0, d1, d2, d3 };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u >= nh) break;
                const int hh = h + u;
                const float dot = ds_round_bf16(dd[u]);
                const float wh = ds_round_bf16(ds_bf(wrow[(long)hh * a.wc]) * a.wscale);
                const float c = ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh);
                if (HS == 1) acc = acc + c;
                else part[hh * EPB + e] = c;
            }
        }
    }
    if (HS > 1) {
        __syncthreads();
        if (sub == 0 && kp) {
            acc = 0.0f;
            for (int hh = 0; hh < a.H; ++hh) acc = acc + part[hh * EPB + e];
        }
        __syncthreads();
    }
    return acc;
}

/* One thread an entry; HS waves of a block share 32 entries, each wave a slice of the heads (its
 * q then uniform, read through scalar loads). A thread holds its key as 128 floats. Without keep
 * (the candidate restriction takes ds41_index_score_kept_kernel below). */
template <int HS>
__global__ __launch_bounds__(256) void ds41_index_score_kernel(DsScoreArgs a) {
    constexpr int EPB = 256 / HS;
    float* part = (float*)r4d_ds41_sparse_dyn;       /* [H][EPB], HS > 1 */
    /* The wave index through readfirstlane, so a wave's head slice (and its q) is provably uniform
     * and q stays in scalar loads. */
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int grp = wave / HS, sub = wave % HS;
    const int e = grp * 32 + lane;
    const long jb = (long)blockIdx.x * EPB, j = jb + e;
    const int hp = (a.H + HS - 1) / HS, h0 = sub * hp, h1 = min(a.H, h0 + hp);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        float* srow = a.sc + m * a.sr;
        const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        if (jb >= reach) {                           /* block-uniform: no key in this block */
            if (sub == 0 && j < a.C) srow[j * a.scs] = -kDsInf;
            continue;
        }
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
        const uint16_t* kp = ds_entry_key(a, s, j, reach);
        const float acc = ds_score_entry<HS>(a, m, kp, part, e, sub, h0, h1);
        if (sub == 0 && j < a.C) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ---- the candidate restriction over the kept blocks only (index_score's decode band and
 * index_topk): a row's keep words below its reach, their kept blocks ranked in block order. */

/* Word w of a row's keep, masked to its blocks below nb. */
__device__ __forceinline__ uint32_t ds_keep_word(const int* krow, long kcs, long w, long nb) {
    uint32_t bits = (uint32_t)krow[w * kcs];
    const long lo = w << 5;
    if (lo + 32 > nb) bits &= nb - lo >= 32 ? ~0u : ((1u << (uint32_t)(nb - lo)) - 1u);
    return bits;
}

/* A block-wide exclusive scan of NT threads (NT a multiple of 32, at most 1024); *total is the
 * sum. ws holds 33 words. Ends on a barrier. */
template <int NT>
__device__ __forceinline__ uint32_t ds_excl_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
    constexpr int NW = NT / 32;
    static_assert(NT % 32 == 0 && NW <= 32, "a block of 32 waves at most");
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint32_t inc = r4d_wave_prefix_u32(v, lane);
    if (lane == 31) ws[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const uint32_t w = lane < NW ? ws[lane] : 0u;
        const uint32_t wi = r4d_wave_prefix_u32(w, lane);
        if (lane < NW) ws[lane] = wi - w;
        if (lane == 31) ws[32] = wi;
    }
    __syncthreads();
    const uint32_t r = ws[wave] + inc - v;
    *total = ws[32];
    __syncthreads();
    return r;
}

/* Thread t's share of a row's keep words below nb blocks: words [t W, t W + W), W = ceil(nw / NT). */
template <int NT>
struct DsKeptScan {
    long w0, w1;                                     /* this thread's words */
    uint32_t off;                                    /* kept blocks before them (block order) */
    uint32_t total;                                  /* the row's kept blocks below nb */
};

template <int NT>
__device__ __forceinline__ DsKeptScan<NT> ds_kept_scan(const int* krow, long kcs, long nb, uint32_t* ws) {
    DsKeptScan<NT> k;
    const long nw = (nb + 31) >> 5, per = (nw + NT - 1) / NT;
    k.w0 = (long)threadIdx.x * per;
    k.w1 = k.w0 + per < nw ? k.w0 + per : nw;
    uint32_t c = 0;
    for (long w = k.w0; w < k.w1; ++w) c += (uint32_t)__builtin_popcount(ds_keep_word(krow, kcs, w, nb));
    k.off = ds_excl_scan<NT>(c, ws, &k.total);
    return k;
}

/* The kept blocks of rank [r0, r0 + n) to list[rank - r0], in block order. The caller's barrier
 * publishes the list. */
template <int NT>
__device__ __forceinline__ void ds_kept_list(const DsKeptScan<NT>& k, const int* krow, long kcs, long nb,
                                             uint32_t r0, uint32_t n, uint32_t* list) {
    uint32_t r = k.off;
    for (long w = k.w0; w < k.w1 && r < r0 + n; ++w) {
        uint32_t bits = ds_keep_word(krow, kcs, w, nb);
        const uint32_t c = (uint32_t)__builtin_popcount(bits);
        if (r + c <= r0) { r += c; continue; }       /* the whole word is before the range */
        while (bits) {
            const uint32_t b = (uint32_t)__builtin_ctz(bits);
            bits &= bits - 1u;
            if (r >= r0 && r < r0 + n) list[r - r0] = (uint32_t)(w << 5) + b;
            ++r;
        }
    }
}

/* kDsKeptEst: kept positions a row is expected to hold (DeepSeek-V4.1: 2048 blocks of 8), which
 * sizes the kept kernel's grid and head split -- any grid gives the same bits. kDsKeptSpan: the
 * kept blocks a workgroup's entries can touch, EPB / block + 2 at block >= 1. */
constexpr long kDsKeptEst = 16384;
constexpr int kDsKeptSpan = 258;

/* index_score with keep on the scalar kernel's arithmetic (the decode band, and any pass the WMMA
 * form does not take): a row's kept positions below its reach are compacted (block order, then
 * position order) and dealt EPB to a workgroup over a grid-stride loop, so the work is the kept
 * count's, not the context's. A kept entry runs ds_score_entry exactly as the unrestricted kernel
 * does (same bits); every other position of the row, to C, is written -inf by a sweep the grid
 * shares. */
template <int HS>
__global__ __launch_bounds__(256) void ds41_index_score_kept_kernel(DsScoreArgs a) {
    constexpr int EPB = 256 / HS;
    float* part = (float*)r4d_ds41_sparse_dyn;       /* [H][EPB], HS > 1 */
    __shared__ uint32_t ws[33];
    __shared__ uint32_t list[kDsKeptSpan];
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int grp = wave / HS, sub = wave % HS;
    const int e = grp * 32 + lane;
    const int hp = (a.H + HS - 1) / HS, h0 = sub * hp, h1 = min(a.H, h0 + hp);
    const long block = a.block;
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        float* srow = a.sc + m * a.sr;
        const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        const int* krow = reach > 0 ? a.keep + m * a.kr : nullptr;
        /* the sweep: positions [x C / G, (x + 1) C / G) of this workgroup, -inf unless kept below reach */
        {
            const long g = gridDim.x, x = blockIdx.x;
            const long p0 = (long)a.C * x / g, p1 = (long)a.C * (x + 1) / g;
            for (long j = p0 + threadIdx.x; j < p1; j += 256)
                if (j >= reach || !ds_kept(a, krow, j)) srow[j * a.scs] = -kDsInf;
        }
        if (reach <= 0) continue;                    /* block-uniform */
        const long nbr = (reach < a.C ? reach : (long)a.C) + block - 1;
        const long nb = nbr / block;                 /* blocks holding a passed position */
        const DsKeptScan<256> ks = ds_kept_scan<256>(krow, a.kcs, nb, ws);
        const long nk = (long)ks.total * block;      /* compacted entries (some past reach) */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
        for (long x0 = (long)blockIdx.x * EPB; x0 < nk; x0 += (long)gridDim.x * EPB) {
            const long r0 = x0 / block, r1 = (x0 + EPB - 1) / block + 1;   /* ranks [r0, r1) */
            const long rn = (r1 < (long)ks.total ? r1 : (long)ks.total) - r0;
            ds_kept_list<256>(ks, krow, a.kcs, nb, (uint32_t)r0, (uint32_t)rn, list);
            __syncthreads();
            const long x = x0 + e;
            long j = -1;
            if (x < nk) {
                const long r = x / block;
                j = (long)list[r - r0] * block + (x - r * block);
            }
            const uint16_t* kp = j >= 0 ? ds_entry_key(a, s, j, reach) : nullptr;
            const float acc = ds_score_entry<HS>(a, m, kp, part, e, sub, h0, h1);
            if (sub == 0 && j >= 0 && j < reach && j < a.C) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
            __syncthreads();                         /* the list is rewritten next round */
        }
    }
}

/* ================================================================== exact top-k by radix select
 *
 * One block of 1024 threads selects, among n keys (0 = not a candidate), the K largest under (key
 * desc, index asc). Three histogram passes (11, 11, 10 bits) find the K-th key T and how many keys
 * equal to T are taken (`need`, the lowest-indexed ones); an emit pass then walks the keys in index
 * order with a block scan, so each taken key knows its rank. A thread folds runs of one digit into
 * a single LDS add, which keeps the first pass (most keys share their top bits) off the atomic
 * unit. Up to kDsCache keys stay in LDS after the first pass. */
constexpr int kDsSelThreads = 1024;
constexpr int kDsCache = 8192;
constexpr int kDsCacheUse = kDsCache;               /* rows up to this many keys run from LDS */

struct DsSelShared {
    uint32_t hist[2048];
    uint32_t ws[33];
    uint32_t prefix, need, total, all, digit;
};

__device__ __forceinline__ uint32_t ds_block_excl_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint32_t inc = r4d_wave_prefix_u32(v, lane);
    if (lane == 31) ws[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const uint32_t w = ws[lane];
        const uint32_t wi = r4d_wave_prefix_u32(w, lane);
        ws[lane] = wi - w;
        if (lane == 31) ws[32] = wi;
    }
    __syncthreads();
    const uint32_t r = ws[wave] + inc - v;
    *total = ws[32];
    __syncthreads();
    return r;
}

/* The digit (from the top) whose cumulative count reaches `need`; updates need and total. */
__device__ __forceinline__ void ds_find_digit(DsSelShared* s, int nbins) {
    const int t = threadIdx.x;
    const uint32_t need = s->need;
    const int r0 = 2 * t, r1 = 2 * t + 1;
    const uint32_t c0 = r0 < nbins ? s->hist[nbins - 1 - r0] : 0u;
    const uint32_t c1 = r1 < nbins ? s->hist[nbins - 1 - r1] : 0u;
    uint32_t total;
    const uint32_t above = ds_block_excl_scan(c0 + c1, s->ws, &total);
    if (c0 && above < need && need <= above + c0) {
        s->digit = (uint32_t)(nbins - 1 - r0);
        s->need = need - above;
    } else if (c1 && above + c0 < need && need <= above + c0 + c1) {
        s->digit = (uint32_t)(nbins - 1 - r1);
        s->need = need - above - c0;
    }
    if (t == 0) s->total = total;
    __syncthreads();
}

template <class Get>
__device__ void ds_select(const Get& get, uint32_t n, uint32_t K, uint32_t* cache, DsSelShared* s) {
    const int t = threadIdx.x;
    for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
    if (t == 0) { s->need = K; s->prefix = 0u; s->all = 0u; }
    __syncthreads();
    {
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = get(i);
            if (cache) cache[i] = key;
            if (key) {
                const uint32_t d = key >> 21;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
    }
    __syncthreads();
    ds_find_digit(s, 2048);
    if (s->total <= K) {                             /* every candidate is taken */
        __syncthreads();
        if (t == 0) s->all = 1u;
        __syncthreads();
        return;
    }
    uint32_t prefix = s->digit;
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0, nbits = pass == 1 ? 11 : 10, hi = shift + nbits;
        const uint32_t mask = (1u << nbits) - 1u;
        __syncthreads();
        for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
        __syncthreads();
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = cache ? cache[i] : get(i);
            if (key && (key >> hi) == prefix) {
                const uint32_t d = (key >> shift) & mask;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
        __syncthreads();
        ds_find_digit(s, 1 << nbits);
        prefix = (prefix << nbits) | s->digit;
    }
    __syncthreads();
    if (t == 0) s->prefix = prefix;
    __syncthreads();
}

/* Walks the keys in index order: emit(i, take, out) for every i < n, `out` the taken key's rank in
 * index order; then finish(count taken). */
template <class Get, class Emit, class Finish>
__device__ void ds_emit(const Get& get, uint32_t n, const uint32_t* cache, DsSelShared* s,
                        const Emit& emit, const Finish& finish) {
    const bool all = s->all != 0u;
    const uint32_t T = all ? 0u : s->prefix, need = all ? 0u : s->need;
    uint32_t base_gt = 0u, base_eq = 0u;
    for (uint32_t i0 = 0; i0 < n; i0 += kDsSelThreads) {
        const uint32_t i = i0 + threadIdx.x;
        const uint32_t key = i < n ? (cache ? cache[i] : get(i)) : 0u;
        const uint32_t gt = key > T ? 1u : 0u;
        const uint32_t eq = (T != 0u && key == T) ? 1u : 0u;
        uint32_t tot;
        const uint32_t ex = ds_block_excl_scan(gt | (eq << 16), s->ws, &tot);
        const uint32_t eq_rank = base_eq + (ex >> 16);
        const bool take = gt || (eq && eq_rank < need);
        const uint32_t out = base_gt + (ex & 0xffffu) + min(eq_rank, need);
        if (i < n) emit(i, take, out);
        base_gt += tot & 0xffffu;
        base_eq += tot >> 16;
    }
    finish(base_gt + min(base_eq, need));
}

/* ------------------------------------------------------------------ index_topk */
/* keep: index_candidates' bits, a bit a block of `block` positions (bshift = log2(block) when
 * block is a power of two, else -1). kept: walk only the kept blocks (keep given, block <= 2^16). */
struct DsTopkArgs {
    const float* sc;
    const int* keep;
    const int* pos;
    int* sel;
    long sr, scs, kr, kcs, ps, selr, selc;
    int M, C, K, ratio, block, bshift, kept;
};

__device__ __forceinline__ long ds_topk_reach(const DsTopkArgs& a, long m) {
    long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
    if (reach > a.C) reach = a.C;
    return reach < 0 ? 0 : reach;
}

/* The key of position j0 + i of a row: not a candidate outside keep's blocks or at -inf (or NaN). */
struct DsTopkKey {
    const float* row;
    const int* krow;
    long scs, kcs, j0;
    int block, bshift;
    __device__ __forceinline__ long pos(uint32_t i) const { return j0 + (long)i; }
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        const long j = j0 + (long)i;
        if (krow) {
            const uint32_t b = bshift >= 0 ? (uint32_t)j >> bshift : (uint32_t)j / (uint32_t)block;
            if (!(((uint32_t)krow[(long)(b >> 5) * kcs] >> (b & 31u)) & 1u)) return 0u;
        }
        const float v = row[j * scs];
        return v > -kDsInf ? ds_key(v) : 0u;
    }
};

/* The same keys over the kept blocks only: entry i is offset i % block of the (i / block)-th
 * listed block (list: kept blocks in block order). Positions keep their order, so (key desc, index
 * asc) over these entries is (key desc, position asc) over the row's candidates -- the selection
 * DsTopkKey makes over [0, reach), whose other positions are not candidates. Past reach (in the
 * last kept block) an entry is not a candidate either. */
struct DsKeptKey {
    const float* row;
    const uint32_t* list;
    long scs, reach;
    int block, bshift;
    __device__ __forceinline__ long pos(uint32_t i) const {
        const uint32_t r = bshift >= 0 ? i >> bshift : i / (uint32_t)block;
        return (long)list[r] * block + (long)(i - r * (uint32_t)block);
    }
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        const long j = pos(i);
        if (j >= reach) return 0u;
        const float v = row[j * scs];
        return v > -kDsInf ? ds_key(v) : 0u;
    }
};

/* kDsKeptMax: kept blocks of a row index_topk lists in LDS (a row with more walks its positions). */
constexpr int kDsKeptMax = 4096;

/* One row's top-K into srow: the selection over n keys of `get`, positions in order, -1 after. */
template <class Get>
__device__ __forceinline__ void ds_topk_row(const Get& get, uint32_t n, uint32_t* cache, DsSelShared* s,
                                            int* srow, long selc, int K) {
    uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
    ds_select(get, n, (uint32_t)K, c, s);
    ds_emit(get, n, c, s,
            [&](uint32_t i, bool take, uint32_t out) { if (take) srow[(long)out * selc] = (int)get.pos(i); },
            [&](uint32_t tot) {
                for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
            });
    __syncthreads();
}

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_kernel(DsTopkArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    __shared__ uint32_t list[kDsKeptMax];
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const uint32_t n = (uint32_t)ds_topk_reach(a, m);
        const int* krow = a.keep ? a.keep + m * a.kr : nullptr;
        int* srow = a.sel + m * a.selr;
        if (krow && a.kept) {                        /* the kept blocks only, when they fit the list */
            const long nb = ((long)n + a.block - 1) / a.block;
            const DsKeptScan<kDsSelThreads> ks = ds_kept_scan<kDsSelThreads>(krow, a.kcs, nb, s.ws);
            if (ks.total <= (uint32_t)kDsKeptMax) {  /* block-uniform */
                ds_kept_list<kDsSelThreads>(ks, krow, a.kcs, nb, 0u, ks.total, list);
                __syncthreads();
                const DsKeptKey get{ a.sc + m * a.sr, list, a.scs, (long)n, a.block, a.bshift };
                ds_topk_row(get, ks.total * (uint32_t)a.block, cache, &s, srow, a.selc, a.K);
                continue;
            }
        }
        const DsTopkKey get{ a.sc + m * a.sr, krow, a.scs, a.kcs, 0, a.block, a.bshift };
        ds_topk_row(get, n, cache, &s, srow, a.selc, a.K);
    }
}

/* The decode form over long rows. Stage 1: each (chunk, row) keeps its chunk's exact top-K as
 * (key << 32 | position), in position order, and its count. A row's winner is among its chunk's
 * K best, so stage 2's top-K over the chunks in order (position order again) is the row's.
 * With kept (keep, every chunk's list fitting, see index_topk_body): the chunks deal the row's kept
 * blocks instead of its positions -- `used[m]` chunks of ceil(kept / used) blocks each, used the
 * fewest that keeps a chunk within kDsKeptChunk entries (at most nch) -- and stage 2 merges only
 * the used chunks. Without kept, used[m] = nch. */
constexpr int kDsKeptChunk = 2048;

struct DsSplitArgs {
    DsTopkArgs t;
    unsigned long long* cand;
    int* cnt;
    int* used;
    int nch, chunk;
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split1_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    __shared__ uint32_t list[kDsKeptMax];
    const int c = blockIdx.x;
    const long m = blockIdx.y;
    const long reach = ds_topk_reach(a.t, m);
    unsigned long long* out = a.cand + (m * a.nch + c) * (long)a.t.K;
    int* cnt = a.cnt + m * a.nch + c;
    const int* krow = a.t.keep ? a.t.keep + m * a.t.kr : nullptr;
    if (krow && a.t.kept) {
        const long block = a.t.block, nb = (reach + block - 1) / block;
        const DsKeptScan<kDsSelThreads> ks = ds_kept_scan<kDsSelThreads>(krow, a.t.kcs, nb, s.ws);
        const long nk = ks.total;
        long used = (nk * block + kDsKeptChunk - 1) / kDsKeptChunk;
        used = used < a.nch ? used : a.nch;
        const long per = used > 0 ? (nk + used - 1) / used : 0;   /* blocks a chunk */
        const long r0 = (long)c * per, rn = c < used ? (nk - r0 < per ? nk - r0 : per) : 0;
        if (c == 0 && threadIdx.x == 0) a.used[m] = (int)used;
        if (rn <= 0) {                               /* block-uniform */
            if (threadIdx.x == 0) *cnt = 0;
            return;
        }
        ds_kept_list<kDsSelThreads>(ks, krow, a.t.kcs, nb, (uint32_t)r0, (uint32_t)rn, list);
        __syncthreads();
        const DsKeptKey get{ a.t.sc + m * a.t.sr, list, a.t.scs, reach, a.t.block, a.t.bshift };
        const uint32_t n = (uint32_t)(rn * block);
        uint32_t* cc = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.t.K, cc, &s);
        ds_emit(get, n, cc, &s,
                [&](uint32_t i, bool take, uint32_t o) {
                    if (take) out[o] = ((unsigned long long)get(i) << 32) | (unsigned long long)(uint32_t)get.pos(i);
                },
                [&](uint32_t tot) { if (threadIdx.x == 0) *cnt = (int)tot; });
        return;
    }
    if (c == 0 && threadIdx.x == 0) a.used[m] = a.nch;
    const long j0 = (long)c * a.chunk;
    long n = reach - j0;
    n = n < 0 ? 0 : (n > a.chunk ? a.chunk : n);
    const DsTopkKey get{ a.t.sc + m * a.t.sr, krow, a.t.scs, a.t.kcs, j0, a.t.block, a.t.bshift };
    ds_select(get, (uint32_t)n, (uint32_t)a.t.K, cache, &s);
    ds_emit(get, (uint32_t)n, cache, &s,
            [&](uint32_t i, bool take, uint32_t o) {
                if (take)
                    out[o] = ((unsigned long long)cache[i] << 32) |
                             (unsigned long long)(uint32_t)(j0 + (long)i);
            },
            [&](uint32_t tot) { if (threadIdx.x == 0) *cnt = (int)tot; });
}

struct DsCandKey {
    const unsigned long long* cr;
    const int* cn;
    uint32_t K;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        return (int)(i % K) < cn[i / K] ? (uint32_t)(cr[i] >> 32) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split2_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.t.M; m += gridDim.x) {
        const unsigned long long* cr = a.cand + m * a.nch * (long)a.t.K;
        int used = a.used[m];
        used = used < 0 ? 0 : (used > a.nch ? a.nch : used);
        const uint32_t n = (uint32_t)used * (uint32_t)a.t.K;
        const DsCandKey get{ cr, a.cnt + m * a.nch, (uint32_t)a.t.K };
        uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.t.K, c, &s);
        int* srow = a.t.sel + m * a.t.selr;
        const long selc = a.t.selc;
        const int K = a.t.K;
        ds_emit(get, n, c, &s,
                [&](uint32_t i, bool take, uint32_t out) {
                    if (take) srow[(long)out * selc] = (int)(uint32_t)(cr[i] & 0xffffffffull);
                },
                [&](uint32_t tot) {
                    for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
                });
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ index_candidates */
/* keep: a bit a block, word b / 32 of a row holding blocks [32 (b / 32), 32 (b / 32) + 32). */
struct DsCandArgs {
    const float* sc;
    const int* pos;
    int* keep;
    long sr, scs, ps, kr, kcs;
    int M, C, block, KB, ratio;
};

/* Block b's score: the max of its positions (std::max(cur, v): a NaN never replaces), +inf when
 * pinned. A -inf block is never kept, so it is not a candidate. */
struct DsBlockKey {
    const float* row;
    long scs, C, block, pin;
    __device__ __forceinline__ uint32_t operator()(uint32_t b) const {
        float bs = -kDsInf;
        const long j0 = (long)b * block;
        const long j1 = j0 + block < C ? j0 + block : C;
        for (long j = j0; j < j1; ++j) {
            const float v = row[j * scs];
            bs = bs < v ? v : bs;
        }
        if ((long)b == pin) bs = kDsInf;
        return bs > -kDsInf ? ds_key(bs) : 0u;
    }
};

static_assert(kDsSelThreads % 32 == 0, "index_candidates writes a wave's ballot as a keep word");

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_candidates_kernel(DsCandArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const long nb = ((long)a.C + a.block - 1) / a.block;
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
        const long pin = (reach >= 1 && (reach - 1) / a.block < nb) ? (reach - 1) / a.block : -1;
        const DsBlockKey get{ a.sc + m * a.sr, a.scs, a.C, a.block, pin };
        uint32_t* c = nb <= kDsCacheUse ? cache : nullptr;
        ds_select(get, (uint32_t)nb, (uint32_t)a.KB, c, &s);
        int* krow = a.keep + m * a.kr;
        const long kcs = a.kcs;
        /* ds_emit walks the blocks kDsSelThreads at a time, thread t on block i0 + t, so a wave's
         * 32 lanes hold one word's blocks and its lane 0 the word's first: the wave's ballot of
         * `take` is the word. Past the last block the lanes do not run and their bits are 0; a
         * wave whose lane 0 does not run holds no block. */
        ds_emit(get, (uint32_t)nb, c, &s,
                [&](uint32_t b, bool take, uint32_t) {
                    const uint32_t word = __builtin_amdgcn_ballot_w32(take);
                    if ((threadIdx.x & 31u) == 0u) krow[(long)(b >> 5) * kcs] = (int)word;
                },
                [&](uint32_t) {});
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink */
struct DsAttnArgs {
    const uint16_t* q;
    const uint16_t* wc;
    const int* wbt;
    const int* wused;
    const uint16_t* cc;
    const int* cbt;
    const int* sel;
    const int* pos;
    const int* cu;
    const float* sink;
    uint16_t* out;
    long qr, outr, wbtr, wbtc, nwblk, cbtr, cbtc, ncblk, selr, selc, wus, ps, cs, sks;
    int nwbt, bs, ncbt, cbs, ratio, S, M, H, win, nk, nkp;
    int cnv;                                         /* latent_format nvfp4: compressed entries packed */
    int cslot;                                       /* the compressed cache's slots a position (D / ratio plain) */
    float scale;
};

constexpr int kDsAttnD = 512;

/* attn_sparse_sink_rope_roundtrip (lever K1 fold d): the decode band's kernel with the inverse rope and the
 * round trip of its output (rope_rows_roundtrip's work) in its epilogue. The rope's angle table and geometry
 * (rope_rows_body's cos_sin plane, columns [off, off + 2 half) of each 512-wide head, gptj pairs, sign -1 for
 * inverse) and the round trip's format and scale (quant_roundtrip's, blocks of 32). */
struct DsOrt {
    const float* cs;
    long cp, cst, tab_rows;
    int off, half, fmt, sc;
    float sign;
};
struct DsAttnArgsOrt : DsAttnArgs {
    DsOrt o;
};
template <bool ORT>
using DsAttnArgsT = std::conditional_t<ORT, DsAttnArgsOrt, DsAttnArgs>;

/* The bf16 at column d of a packed entry (byte d/2's nibble, block d/16's scale), and the pair at an
 * even d as a word (value d low). */
__device__ __forceinline__ uint32_t ds_nv4_col(const uint16_t* kk, int d) {
    const unsigned char* b = (const unsigned char*)kk;
    const uint32_t code = ((uint32_t)b[d / 2] >> (4 * (d & 1))) & 15u;
    return ds_nv4_bf16(code, ds_e4m3_f32(b[kDsAttnD / 2 + d / 16]));
}
__device__ __forceinline__ uint32_t ds_nv4_pair(uint32_t byte2, uint32_t sbyte) {
    const float sc = ds_e4m3_f32(sbyte);
    return ds_nv4_bf16(byte2 & 15u, sc) | (ds_nv4_bf16((byte2 >> 4) & 15u, sc) << 16);
}
constexpr int kDsTileRow = 80;                       /* 32 bf16 of a key + 16 bytes of padding */
constexpr int kDsTileBytes = 8 * 32 * kDsTileRow;    /* one 32-key tile a wave */

__host__ __device__ constexpr long ds_attn_tile_off(int nkp, int hg) {
    return ((long)nkp * 8 + (long)hg * nkp * 4 + (long)hg * 8 + 15) / 16 * 16;
}

/* Workgroup (d half, head group, row). LDS: the row's key pointers (null = no key), scores then p
 * [HG][nkp], max and den [HG], and the waves' transpose tiles. */
template <int HG, int DS>
__global__ __launch_bounds__(256) void ds41_attn_sparse_sink_kernel(DsAttnArgs a) {
    constexpr int D = kDsAttnD, NW = 8, DPT = D / DS / 256;
    static_assert(DPT == 1 || DPT == 2, "one or two columns a thread");
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    float* sp = (float*)(r4d_ds41_sparse_dyn + (long)a.nkp * 8);
    float* smx = sp + (long)HG * a.nkp;
    float* sden = smx + HG;
    unsigned char* tile = r4d_ds41_sparse_dyn + ds_attn_tile_off(a.nkp, HG);
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.y * HG;
    const int hmax = min(HG, a.H - hg0);
    const int dcol = blockIdx.x * (D / DS) + tid * DPT;
    for (long m = blockIdx.z; m < a.M; m += gridDim.z) {
        /* ---- the row's keys, in libref's order: the window, then the selections */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < a.nkp; k += 256) {
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)a.cslot;
                }
            }
            kp[k] = ptr;
        }
        __syncthreads();

        /* ---- scores: a lane a key, each dot in d order. A wave loads its 32 keys 32 columns at a
         * time (four lanes a key, 16 bytes each), transposes them through its tile and reads its own
         * key's row back; q is uniform, so it comes through scalar loads. */
        {
            const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
            unsigned char* tb = tile + wave * (32 * kDsTileRow);
            for (int kb = wave * 32; kb < a.nk; kb += NW * 32) {
                const uint16_t* lp[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int kk = kb + i * 8 + (lane >> 2);
                    lp[i] = kk < a.nk ? kp[kk] : nullptr;
                }
                float acc[HG];
#pragma unroll
                for (int h = 0; h < HG; ++h) acc[h] = 0.0f;
                for (int c = 0; c < D / 32; ++c) {
                    uint4 v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int col = c * 32 + (lane & 3) * 8;
                        if (a.cnv && kb + i * 8 + (lane >> 2) >= a.win && lp[i]) {   /* a packed entry */
                            const unsigned char* b = (const unsigned char*)lp[i];
                            v[i] = ds_nv4_unpack8(*(const uint32_t*)(b + col / 2), ds_e4m3_f32(b[D / 2 + col / 16]));
                        } else {
                            v[i] = lp[i] ? *(const uint4*)(lp[i] + col) : make_uint4(0, 0, 0, 0);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        *(uint4*)(tb + (i * 8 + (lane >> 2)) * kDsTileRow + (lane & 3) * 16) = v[i];
                    ds_wave_sync();
                    /* Eight columns at a time keeps HG x 4 q words live, within the scalar file. */
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 ri = *(const uint4*)(tb + lane * kDsTileRow + i * 16);
                        const uint32_t wk[4] = { ri.x, ri.y, ri.z, ri.w };
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float k0 = bf16w_lo(wk[e]), k1 = bf16w_hi(wk[e]);
                            const int qi = c * 16 + i * 4 + e;
#pragma unroll
                            for (int h = 0; h < HG; ++h) {
                                const uint32_t qw = qrow[(long)(h < hmax ? h : 0) * (D / 2) + qi];
                                acc[h] = __builtin_fmaf(bf16w_lo(qw), k0, acc[h]);
                                acc[h] = __builtin_fmaf(bf16w_hi(qw), k1, acc[h]);
                            }
                        }
                    }
                    ds_wave_sync();                  /* the tile's reads before the next chunk's writes */
                }
                const int k = kb + lane;
                if (k < a.nk) {
                    const bool have = kp[k] != nullptr;
#pragma unroll
                    for (int h = 0; h < HG; ++h) sp[(long)h * a.nkp + k] = have ? acc[h] * a.scale : -kDsInf;
                }
            }
        }
        __syncthreads();

        /* ---- per head (one wave each): max floored at -1e30, then p in place (0 for no key) */
        if (wave < HG) {
            float* sh = sp + (long)wave * a.nkp;
            float mx = -1e30f;
            for (int k = lane; k < a.nk; k += 32) {
                const float v = sh[k];
                mx = mx < v ? v : mx;
            }
            mx = r4d_wave_max(mx);
            for (int k = lane; k < a.nkp; k += 32)
                sh[k] = (k < a.nk && kp[k]) ? ds_expf(sh[k] - mx) : 0.0f;
            if (lane == 0) smx[wave] = mx;
        }
        __syncthreads();

        /* ---- sum p k and sum p, key by key in libref's order. A key with no entry adds p = 0 times
         * k = 0, which leaves both sums as they were (they start at +0 and never reach -0). */
        float acc[HG][DPT], den[HG];
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            den[h] = 0.0f;
#pragma unroll
            for (int d = 0; d < DPT; ++d) acc[h][d] = 0.0f;
        }
        for (int k = 0; k < a.nkp; k += 4) {
            float kv[4][DPT];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint16_t* ptr = kp[k + u];
                const bool pk = a.cnv && k + u >= a.win && ptr;   /* a packed entry */
                if constexpr (DPT == 2) {
                    const uint32_t w = pk ? ds_nv4_col(ptr, dcol) | (ds_nv4_col(ptr, dcol + 1) << 16)
                                          : (ptr ? *(const uint32_t*)(ptr + dcol) : 0u);
                    kv[u][0] = bf16w_lo(w);
                    kv[u][DPT - 1] = bf16w_hi(w);
                } else {
                    kv[u][0] = pk ? bf16w_lo(ds_nv4_col(ptr, dcol))
                                  : (ptr ? bf16_to_f32((uint16_t)ldg_u16(ptr + dcol)) : 0.0f);
                }
            }
#pragma unroll
            for (int h = 0; h < HG; ++h) {
                const float4 pv = *(const float4*)(sp + (long)h * a.nkp + k);
                const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
#pragma unroll
                    for (int d = 0; d < DPT; ++d) {
                        const float prod = pk[u] * kv[u][d];
                        acc[h][d] = acc[h][d] + prod;
                    }
                }
                if (wave == 0) {
#pragma unroll
                    for (int u = 0; u < 4; ++u) den[h] = den[h] + pk[u];
                }
            }
        }
        if (wave == 0 && lane < hmax) {
            float dn = den[0];
#pragma unroll
            for (int h = 1; h < HG; ++h)
                if (lane == h) dn = den[h];
            sden[lane] = dn + ds_expf(a.sink[(long)(hg0 + lane) * a.sks] - smx[lane]);
        }
        __syncthreads();
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                const uint32_t lo = ds_bf16_bits(acc[h][0] / dn), hi = ds_bf16_bits(acc[h][DPT - 1] / dn);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[h][0] / dn);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink, decode band
 *
 * Passes of at most kWmRows rows (decode, speculative verify) run the chains of the kernel above,
 * each in its order and with its roundings, so its bits; only which thread runs a chain changes.
 * (Where two NaNs of different sign meet in one add, which one propagates follows the operand order
 * the compiler picked, in the kernel above as here: such outputs are NaN in both, their sign bit
 * may differ.) A workgroup is a (head group of HG, row) of NT threads:
 *   keys     the row's key pointers in LDS (a zero row for no key), q as f32 in LDS, the expf
 *            table, the sinks.
 *   scores   a lane a key, HS heads at a time: each dot in d order, the key read straight from
 *            global 16 bytes at a time, two 128-byte lines in flight, q broadcast from LDS (four
 *            heads: from its bf16 row by scalar loads). Each
 *            wave keeps its 32 keys' maximum per head (floored at -1e30; a NaN score never wins,
 *            as in `mx < v ? v : mx`); the max is order-free, a -0/+0 choice included (s - max
 *            and sink - max give the same exp either way).
 *   softmax  p = glibc expf(s - max), a thread a key.
 *   PV       a thread owns DPT latent columns x HPT heads and sums p k over the keys in key order;
 *            sum p, key by key, in the first wave of each head part. The keys come either straight
 *            from global (PVD: a wave walks 32-key blocks, one block's loads in flight, no barrier)
 *            or through LDS tiles of NT/64 keys (a thread one 16-byte piece, P tiles in flight in
 *            registers, a barrier a tile). A no-key slot adds p = 0 times a zero row: +0, which
 *            leaves a sum that starts at +0 unchanged.
 * No global load may sit under a branch: the wait-count pass then waits for every load in flight
 * (s_wait_loadcnt 0) and the pipelining is gone; loads past the end read the zero row instead. */
__device__ __align__(16) uint16_t kDsZeroKey[kDsAttnD];

/* 16 bytes at p[i] by a global (not flat) load. */
typedef unsigned int ds_v4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ds_ldg16(const uint16_t* p, int i) {
    const ds_v4u v = ((__attribute__((address_space(1))) const ds_v4u*)p)[i];
    return make_uint4(v.x, v.y, v.z, v.w);
}

/* ---- the packed latent's loads (latent_format nvfp4, layout above ds_nv4_bytes): 64 columns of a
 * packed entry from column chunk c (8 columns a chunk, c a multiple of 8) are 32 code bytes at byte
 * 4c and 4 scale bytes at byte 256 + c/2; unpacked, they are the 8 x 16 bytes of bf16 the plain
 * entry holds there. The zero row (all zero bytes) unpacks to zeros. */
struct DsNv4Raw {
    uint4 c0, c1;
    uint32_t sc;
};
__device__ __forceinline__ DsNv4Raw ds_nv4_ld64(const uint16_t* kk, int c) {
    DsNv4Raw r;
    r.c0 = ds_ldg16(kk, c / 4);
    r.c1 = ds_ldg16(kk, c / 4 + 1);
    r.sc = ((__attribute__((address_space(1))) const uint32_t*)kk)[64 + c / 8];
    return r;
}
__device__ __forceinline__ void ds_nv4_unpack64(const DsNv4Raw& r, uint4 (&o)[8]) {
    const uint32_t w[8] = { r.c0.x, r.c0.y, r.c0.z, r.c0.w, r.c1.x, r.c1.y, r.c1.z, r.c1.w };
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = ds_nv4_unpack8(w[i], ds_e4m3_f32((r.sc >> (8 * (i / 2))) & 0xffu));
}

struct DsDecLds {
    long kp, sp, smx, sden, tab, u, sq, bmax, nvt, total;
};
/* The packed direct PV's decode tables (8.11., v3), past everything else: 256 code-byte entries of
 * two words -- the bits of (float) 2|e2m1| of the low and the high nibble, each with its code's sign
 * bit -- then 256 floats ds_e4m3_half(b). */
constexpr long kDsNvTabBytes = 256 * 8 + 256 * 4;
/* nkq: keys rounded up to 32. kp [nkq] pointers, sp [HG][nkq] scores then p, smx / sden [HG] (sden
 * holds the sinks until the den), tab the 32 doubles of the expf table; then, in one region,
 * sq [HG][512] q and bmax [HG][nkq/32] block maxima (scores) or the two key tiles of a tiled PV
 * (2 x nt x 16 bytes; nt = 0 for the direct PV); nvt the packed direct PV's tables (nvt true). */
__host__ __device__ constexpr DsDecLds ds_dec_lds(int nkq, int hg, int nt, bool nvt = false) {
    DsDecLds l{};
    l.kp = 0;
    l.sp = (long)nkq * 8;
    l.smx = l.sp + (long)hg * nkq * 4;
    l.sden = l.smx + (long)hg * 4;
    l.tab = (l.sden + (long)hg * 4 + 15) / 16 * 16;
    l.u = l.tab + 32 * 8;
    l.sq = l.u;
    l.bmax = l.sq + (long)hg * kDsAttnD * 4;
    const long sc = l.bmax + (long)hg * (nkq / 32) * 4, pv = l.u + 2L * nt * 16;
    l.total = sc > pv ? sc : pv;
    l.nvt = (l.total + 15) / 16 * 16;
    if (nvt) l.total = l.nvt + kDsNvTabBytes;
    return l;
}

/* ds_expf with the table read from `tab` (an LDS copy of kDsExp2fTab): the same operations. */
__device__ __forceinline__ float ds_expf_tab(float x, const unsigned long long* tab) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = tab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An offset of 0 the compiler cannot see through, made after the running sums and before the
 * next 16 key bytes are unpacked: the q reads and key conversions of 8 columns cannot be hoisted
 * above the previous columns' fmas (HS x 64 live floats spilled otherwise). */
template <int HS>
__device__ __forceinline__ int ds_dec_step(const float (&acc)[HS], uint32_t (&w)[4]) {
    int z;
#ifdef R4D_EMU_HOST   /* tests/r4d_emu: no ISA on the host; the offset is 0 there too */
    (void)acc;
    (void)w;
    z = 0;
#else
    if constexpr (HS == 1)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3]) : "v"(acc[0]));
    else if constexpr (HS == 2)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]));
    else
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]), "v"(acc[2]), "v"(acc[3]));
#endif
    return z;
}

/* 64 latent columns of one key (8 x 16 bytes) into the HS dots, in d order. With more than one
 * head, step i+1's q reads go out (behind the fence of step i-1's sums) before step i's fmas. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot(float (&acc)[HS], const uint4 (&r)[8], const float* qp) {
    static_assert(HS == 1 || HS == 2 || HS == 4, "head subgroup");
    if constexpr (HS == 1) {
        /* one head: 64 hoisted q floats fit (no spill), and the hoist hides the LDS latency */
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
            const float4 qa = *(const float4*)(qp + i * 8);
            const float4 qb = *(const float4*)(qp + i * 8 + 4);
            const float qv[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[0] = __builtin_fmaf(qv[2 * e], bf16w_lo(wk[e]), acc[0]);
                acc[0] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wk[e]), acc[0]);
            }
        }
    } else {
        float4 qc[HS][2], qn[HS][2];
        uint32_t wc[4] = { r[0].x, r[0].y, r[0].z, r[0].w }, wn[4];
        {
            const float* qi = qp + ds_dec_step<HS>(acc, wc);
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                qc[h][0] = *(const float4*)(qi + h * kDsAttnD);
                qc[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i + 1 < 8) {
                wn[0] = r[i + 1].x; wn[1] = r[i + 1].y; wn[2] = r[i + 1].z; wn[3] = r[i + 1].w;
                const float* qi = qp + ds_dec_step<HS>(acc, wn) + (i + 1) * 8;
#pragma unroll
                for (int h = 0; h < HS; ++h) {
                    qn[h][0] = *(const float4*)(qi + h * kDsAttnD);
                    qn[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
                }
            }
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                const float qv[8] = { qc[h][0].x, qc[h][0].y, qc[h][0].z, qc[h][0].w,
                                      qc[h][1].x, qc[h][1].y, qc[h][1].z, qc[h][1].w };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[h] = __builtin_fmaf(qv[2 * e], bf16w_lo(wc[e]), acc[h]);
                    acc[h] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wc[e]), acc[h]);
                }
            }
            if (i + 1 < 8) {
#pragma unroll
                for (int h = 0; h < HS; ++h) { qc[h][0] = qn[h][0]; qc[h][1] = qn[h][1]; }
#pragma unroll
                for (int e = 0; e < 4; ++e) wc[e] = wn[e];
            }
        }
    }
}

/* The HS-head dot of ds_dec_dot with q read straight from its bf16 row by scalar loads (wave-uniform
 * addresses: qw the row's words at this chunk, ho[h] head h's word offset), so the q values take
 * neither LDS bandwidth nor VGPRs. The same fmas in the same order. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot_s(float (&acc)[HS], const uint4 (&r)[8], const uint32_t* qw,
                                             const int (&ho)[HS]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
        const int z = __builtin_amdgcn_readfirstlane(ds_dec_step<HS>(acc, wk));
        const uint32_t* qi = qw + z + i * 4;
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const uint32_t* qh = qi + ho[h];
            const uint32_t qq[4] = { qh[0], qh[1], qh[2], qh[3] };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[h] = __builtin_fmaf(bf16w_lo(qq[e]), bf16w_lo(wk[e]), acc[h]);
                acc[h] = __builtin_fmaf(bf16w_hi(qq[e]), bf16w_hi(wk[e]), acc[h]);
            }
        }
    }
}

/* ds_dec_dot (QS false) or ds_dec_dot_s (QS) over a packed key's 64 columns, the raw code words and
 * scale bytes of ds_nv4_ld64: each 8-column word is unpacked right before its fmas, behind the fence
 * of the previous word's sums (ds_dec_step), so one word's bf16 is live at a time. Unpacking all 64
 * columns ahead (ds_nv4_unpack64) held 32 more VGPRs beside the raw words and the hoisted q, and the
 * 1024-thread forms spilled (7.11.). The same fmas in the same order: the same bits. */
template <int HS, bool QS>
__device__ __forceinline__ void ds_dec_dot_nv(float (&acc)[HS], const DsNv4Raw& r, const float* qp,
                                              const uint32_t* qw, const int (&ho)[HS]) {
    const uint32_t w8[8] = { r.c0.x, r.c0.y, r.c0.z, r.c0.w, r.c1.x, r.c1.y, r.c1.z, r.c1.w };
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t f[4] = { w8[i], (r.sc >> (8 * (i / 2))) & 0xffu, 0u, 0u };
        const int z = ds_dec_step<HS>(acc, f);
        const float sh = ds_e4m3_half(f[1]);
        const float kf[8] = {                            /* the 8 values as f32: the bf16's own value */
            ds_nv4_at<0>(f[0], sh), ds_nv4_at<4>(f[0], sh), ds_nv4_at<8>(f[0], sh), ds_nv4_at<12>(f[0], sh),
            ds_nv4_at<16>(f[0], sh), ds_nv4_at<20>(f[0], sh), ds_nv4_at<24>(f[0], sh), ds_nv4_at<28>(f[0], sh) };
        if constexpr (QS) {
            const uint32_t* qi = qw + __builtin_amdgcn_readfirstlane(z) + i * 4;
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                const uint32_t* qh = qi + ho[h];
                const uint32_t qq[4] = { qh[0], qh[1], qh[2], qh[3] };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[h] = __builtin_fmaf(bf16w_lo(qq[e]), kf[2 * e], acc[h]);
                    acc[h] = __builtin_fmaf(bf16w_hi(qq[e]), kf[2 * e + 1], acc[h]);
                }
            }
        } else {
            /* one head: q not behind the fence (64 hoisted floats, as ds_dec_dot); more: behind it */
            const float* qi = qp + (HS == 1 ? 0 : z) + i * 8;
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                const float4 qa = *(const float4*)(qi + h * kDsAttnD);
                const float4 qb = *(const float4*)(qi + h * kDsAttnD + 4);
                const float qv[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[h] = __builtin_fmaf(qv[2 * e], kf[2 * e], acc[h]);
                    acc[h] = __builtin_fmaf(qv[2 * e + 1], kf[2 * e + 1], acc[h]);
                }
            }
        }
    }
}

/* Four keys (their DPT columns in kw) into the HPT heads' sums p k, and sum p in the den wave,
 * each chain in key order (libref's mul, then add). */
template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pv4(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kw)[4],
                                           const float* ph, int nkq, int k0, bool dw) {
    float kf[4][DPT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        kf[u][0] = bf16w_lo(kw[u]);
        if constexpr (DPT == 2) kf[u][DPT - 1] = bf16w_hi(kw[u]);
    }
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        const float4 pv = *(const float4*)(ph + (long)j * nkq + k0);
        const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                const float prod = pk[u] * kf[u][d];
                acc[j][d] = acc[j][d] + prod;
            }
        }
        if (dw) {
#pragma unroll
            for (int u = 0; u < 4; ++u) den[j] = den[j] + pk[u];
        }
    }
}

/* The DPT columns at dcol of the 32 keys whose pointers lanes 0..31 hold (a wave-uniform pointer a
 * key, through readlane), straight from global. */
template <int DPT>
__device__ __forceinline__ void ds_dec_pvld(uint32_t (&kv)[32], const uint16_t* lp, int dcol) {
    const unsigned long long v = (unsigned long long)lp;
    const int lo = (int)(uint32_t)v, hi = (int)(uint32_t)(v >> 32);
#pragma unroll
    for (int u = 0; u < 32; ++u) {
        const unsigned long long pu = ((unsigned long long)(uint32_t)__builtin_amdgcn_readlane(hi, u) << 32) |
                                      (uint32_t)__builtin_amdgcn_readlane(lo, u);
        const uint16_t* p = (const uint16_t*)pu + dcol;
        if constexpr (DPT == 2) kv[u] = *(__attribute__((address_space(1))) const uint32_t*)p;
        else kv[u] = ldg_u16(p);
    }
}

/* ds_dec_pvld for packed keys (DPT 2: dcol even, the pair in one code byte): the code byte and the
 * scale byte of N of the 32 keys, those whose pointers lanes b0 .. b0 + N - 1 hold (b0 wave-uniform). The two bytes of a
 * key sit in two registers until they land, so a packed ring holds half the plain ring's keys (N 16)
 * for the same 32 loads in flight: a 32-key packed ring (64 registers in flight + 32 unpacking)
 * spilled 280-400 VGPRs to scratch in every 1024-thread form (7.11.: 1 row 30.3 ms/step vs 20.6). */
/* 8.11. (v3): both bytes of a key land in one register -- the code byte in the low half, the scale byte in the high
 * (kv = code | scale << 16) -- as two 16-bit halves of a vector (d16 loads). A 32-key slot then holds what a 16-key
 * slot held, so the direct PV walks the packed blocks a block a step, two blocks of loads in flight, as the plain walk
 * does (the 16-key ring took twice the steps, each behind the previous slot's sums). (Buffer loads from the same base,
 * tried for their SGPR addressing, land the bytes in two registers and spilled.) */
typedef unsigned short ds_u16x2 __attribute__((ext_vector_type(2)));
template <int N>
__device__ __forceinline__ void ds_dec_pvld_nv(uint32_t (&kv)[N], const uint16_t* lp, uint32_t cb, uint32_t sb, int b0) {
    const unsigned long long v = (unsigned long long)lp;
    const int lo = (int)(uint32_t)v, hi = (int)(uint32_t)(v >> 32);
#pragma unroll
    for (int u = 0; u < N; ++u) {
        const unsigned long long pu = ((unsigned long long)(uint32_t)__builtin_amdgcn_readlane(hi, b0 + u) << 32) |
                                      (uint32_t)__builtin_amdgcn_readlane(lo, b0 + u);
        const __attribute__((address_space(1))) const unsigned char* b =
            (const __attribute__((address_space(1))) unsigned char*)pu;
        ds_u16x2 h;
        h.x = (unsigned short)b[cb];
        h.y = (unsigned short)b[sb];
        kv[u] = __builtin_bit_cast(uint32_t, h);
    }
}
/* p through an empty asm that reads the PV sums: the packed PV ring's next loads cannot be hoisted
 * above the previous slot's fmas (the scheduler otherwise issued a whole block's loads at once, two
 * registers a key, and spilled). */
template <int HPT, int DPT>
__device__ __forceinline__ const uint16_t* ds_dec_pfence(const uint16_t* p, const float (&acc)[HPT][DPT]) {
#ifdef R4D_EMU_HOST
    (void)acc;
    return p;
#else
    static_assert(HPT * DPT <= 4, "the fence reads at most four sums");
    unsigned long long v = (unsigned long long)p;
    uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
    const float* f = &acc[0][0];
    if constexpr (HPT * DPT == 1) asm volatile("" : "+v"(lo), "+v"(hi) : "v"(f[0]));
    else if constexpr (HPT * DPT == 2) asm volatile("" : "+v"(lo), "+v"(hi) : "v"(f[0]), "v"(f[1]));
    else asm volatile("" : "+v"(lo), "+v"(hi) : "v"(f[0]), "v"(f[1]), "v"(f[2]), "v"(f[3]));
    return (const uint16_t*)(((unsigned long long)hi << 32) | lo);
#endif
}

/* ds_dec_pv4 over four packed keys (kv: code byte | scale byte << 16, DPT 2: the column pair of one code byte), the
 * values as f32 by ds_nv4_at -- the same products and sums in the same order as over their bf16. The forms with two
 * heads a PV thread (HPT 2) decode in registers, shared by the heads (the LDS tables below spilled them). */
template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pv4_nv(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kw)[4],
                                              const float* ph, int nkq, int k0, bool dw) {
    static_assert(DPT == 2, "a code byte is a column pair");
    float kf[4][DPT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float sh = ds_e4m3_half(kw[u] >> 16);
        kf[u][0] = ds_nv4_at<0>(kw[u], sh);
        kf[u][1] = ds_nv4_at<4>(kw[u], sh);
    }
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        const float4 pv = *(const float4*)(ph + (long)j * nkq + k0);
        const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                const float prod = pk[u] * kf[u][d];
                acc[j][d] = acc[j][d] + prod;
            }
        }
        if (dw) {
#pragma unroll
            for (int u = 0; u < 4; ++u) den[j] = den[j] + pk[u];
        }
    }
}
template <int HPT, int DPT, int N>
__device__ __forceinline__ void ds_dec_pvsum_nv(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kv)[N],
                                                const float* ph, int nkq, int k0, bool dw) {
#pragma unroll
    for (int u0 = 0; u0 < N; u0 += 4) {
        const uint32_t kw[4] = { kv[u0], kv[u0 + 1], kv[u0 + 2], kv[u0 + 3] };
        ds_dec_pv4_nv<HPT, DPT>(acc, den, kw, ph, nkq, k0 + u0, dw);
    }
}
/* Does a decode form decode the packed PV by the LDS tables: the direct PV with one head a PV thread (the kernel's
 * NHP = min(NT / (512 / DPT), HG), HPT = HG / NHP). The host sizes the LDS by it. */
__host__ __device__ constexpr bool ds_dec_nvt(int hg, int nt, int dpt, bool pvd, bool nv) {
    const int ncg = kDsAttnD / dpt, nhp = (nt / ncg) < hg ? (nt / ncg) : hg;
    return nv && pvd && hg / nhp == 1;
}
/* ds_dec_pv4 over four packed keys (kv: code byte | scale byte << 16, DPT 2: the column pair of one code byte) by
 * the LDS tables (tc: a code byte's two nibbles as the bits of (float) 2|e2m1| with the sign bit,
 * ts: ds_e4m3_half of a scale byte): the value is |t| x sh with t's sign bit or-ed in -- ds_nv4_at's float, its product
 * and its sign, the same bits -- in two loads, two multiplies and two ors a key instead of some twenty instructions. */
template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pv4_nvt(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kw)[4],
                                               const float* ph, int nkq, int k0, bool dw, const uint2* tc,
                                               const float* ts) {
    static_assert(DPT == 2, "a code byte is a column pair");
    float kf[4][DPT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float sh = ts[kw[u] >> 16];
        const uint2 t = tc[kw[u] & 0xffu];
        kf[u][0] = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, __builtin_fabsf(__builtin_bit_cast(float, t.x)) * sh) |
                                                 (t.x & 0x80000000u));
        kf[u][1] = __builtin_bit_cast(float, __builtin_bit_cast(uint32_t, __builtin_fabsf(__builtin_bit_cast(float, t.y)) * sh) |
                                                 (t.y & 0x80000000u));
    }
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        const float4 pv = *(const float4*)(ph + (long)j * nkq + k0);
        const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                const float prod = pk[u] * kf[u][d];
                acc[j][d] = acc[j][d] + prod;
            }
        }
        if (dw) {
#pragma unroll
            for (int u = 0; u < 4; ++u) den[j] = den[j] + pk[u];
        }
    }
}
template <int HPT, int DPT, int N>
__device__ __forceinline__ void ds_dec_pvsum_nvt(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kv)[N],
                                                 const float* ph, int nkq, int k0, bool dw, const uint2* tc,
                                                 const float* ts) {
#pragma unroll
    for (int u0 = 0; u0 < N; u0 += 4) {
        const uint32_t kw[4] = { kv[u0], kv[u0 + 1], kv[u0 + 2], kv[u0 + 3] };
        ds_dec_pv4_nvt<HPT, DPT>(acc, den, kw, ph, nkq, k0 + u0, dw, tc, ts);
    }
}
/* The table entries: code byte c's two values' (float) 2|e2m1| bits with the sign bit (ds_nv4_at<0>, <4>), and
 * ds_e4m3_half(c). */
__device__ __forceinline__ uint2 ds_nv4_tab(uint32_t c) {
    const uint32_t lo = __builtin_bit_cast(uint32_t, (float)((0xC8643210u >> ((c & 7u) << 2)) & 15u)) | ((c & 8u) << 28);
    const uint32_t hi = __builtin_bit_cast(uint32_t, (float)((0xC8643210u >> (((c >> 4) & 7u) << 2)) & 15u)) |
                        ((c & 0x80u) << 24);
    return make_uint2(lo, hi);
}

template <int HPT, int DPT, int N = 32>
__device__ __forceinline__ void ds_dec_pvsum(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kv)[N],
                                             const float* ph, int nkq, int k0, bool dw) {
#pragma unroll
    for (int u0 = 0; u0 < N; u0 += 4) {
        const uint32_t kw[4] = { kv[u0], kv[u0 + 1], kv[u0 + 2], kv[u0 + 3] };
        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, k0 + u0, dw);
    }
}


/* attn_sparse_sink_rope_roundtrip's epilogue on one thread's column pair (c0 = dcol, c0 + 1) of a head:
 * rope_rows_roundtrip's work (rope_rt_kernel, the separate launch) on the pair of bf16 (lo, hi) the plain
 * kernel stores there, with r4d_ds41_qrt.h's rules:
 *   - the rope: when the pair is rotated (off <= c0 < off + 2 half, c0 - off even, a gptj pair inside the
 *     thread) and the row's position is in the table, rope_rows' expressions on the stored values,
 *     a' = bf16(a c - b s), b' = bf16(a s + b c), c = cs[p][i], s = sign cs[p][half + i], i = (c0 - off) / 2;
 *   - the round trip: the block's amax over its 32 columns, which are this lane's two and those of the 15
 *     other lanes of its half-wave (columns 2 cg, cg = tid % 256, so lanes 0-15 and 16-31 each hold one
 *     32-aligned block), every |v| a NaN-free non-negative float, so fmaxf of the lane's pair then the
 *     16-lane butterfly is the same maximum as quant_group_kernel's 32-lane butterfly; then block_scale and
 *     roundtrip of each value, rounded to bf16.
 * Every lane of the wave calls it (the callers' branches are wave-uniform). */
__device__ __forceinline__ void ds_ort_pair(uint32_t& lo, uint32_t& hi, bool rot, float c, float sn, int fmt,
                                            int sc) {
    namespace q = r4d_ds41q;
    float v0 = q::ds_bf((uint16_t)lo), v1 = q::ds_bf((uint16_t)hi);
    if (rot) {
        const float va = v0, vb = v1;
        v0 = q::ds_bf(q::ds_bf16(va * c - vb * sn));
        v1 = q::ds_bf(q::ds_bf16(va * sn + vb * c));
    }
    const float amax = r4d_xor_max<16>(fmaxf(q::ds_amax_term(v0), q::ds_amax_term(v1)));
    const float bs = q::block_scale(amax, fmt, sc);
    lo = q::ds_bf16(q::roundtrip(v0, bs, fmt));
    hi = q::ds_bf16(q::roundtrip(v1, bs, fmt));
}

/* ORT (attn_sparse_sink_rope_roundtrip): the epilogue also runs ds_ort_pair on every stored pair, so the
 * output is the bytes attn_sparse_sink + rope_rows_roundtrip leave; everything before it is the same code. */
template <int HG, int NT, int DPT, int P, bool PVD, bool NV, bool ORT = false>
__global__ __launch_bounds__(NT) void ds41_attn_dec_kernel(DsAttnArgsT<ORT> a) {
    static_assert(!NV || DPT == 2, "the packed latent's PV reads a column pair from one code byte");
    static_assert(!ORT || DPT == 2, "the fused round trip takes a column pair a thread");
    constexpr int D = kDsAttnD, NW = NT / 32, NCG = D / DPT, TK = NT / 64;
    constexpr int NHP = (NT / NCG) < HG ? (NT / NCG) : HG;   /* head parts in PV */
    constexpr int HPT = HG / NHP;                           /* heads a PV thread */
    constexpr int HS = HG < 4 ? HG : 4, NSG = HG / HS;      /* heads a score task, tasks a key block */
    constexpr bool QS = HS == 4;                            /* q by scalar loads in the scores */
    static_assert(HG % HS == 0, "head subgroups");
    static_assert(NT % 64 == 0 && 32 % TK == 0 && TK % 4 == 0 && NT >= NCG && HG % NHP == 0 &&
                  (DPT == 1 || DPT == 2) && P >= 1, "layout");
    const int nkq = (a.nk + 31) / 32 * 32, nkb = nkq / 32, ntl = nkq / TK;
    constexpr bool NVT = ds_dec_nvt(HG, NT, DPT, PVD, NV);   /* the packed PV by LDS tables */
    const DsDecLds L = ds_dec_lds(nkq, HG, PVD ? 0 : NT, NVT);
    const uint16_t** kp = (const uint16_t**)(r4d_ds41_sparse_dyn + L.kp);
    float* sp = (float*)(r4d_ds41_sparse_dyn + L.sp);
    float* sq = (float*)(r4d_ds41_sparse_dyn + L.sq);
    float* bmax = (float*)(r4d_ds41_sparse_dyn + L.bmax);
    float* smx = (float*)(r4d_ds41_sparse_dyn + L.smx);
    float* sden = (float*)(r4d_ds41_sparse_dyn + L.sden);
    unsigned long long* tab = (unsigned long long*)(r4d_ds41_sparse_dyn + L.tab);
    unsigned char* tiles = r4d_ds41_sparse_dyn + L.u;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.x * HG;
    const int hmax = min(HG, a.H - hg0);
    const long m = blockIdx.y;
    const uint16_t* zero = kDsZeroKey;

    /* ---- the row's keys, in libref's order: the window, then the selections */
    {
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < nkq; k += NT) {
            const uint16_t* ptr = zero;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)a.cslot;
                }
            }
            kp[k] = ptr;
        }
        const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        for (int i = tid; i < (QS ? 0 : HG * (D / 2)); i += NT) {
            const int h = i / (D / 2), c = i % (D / 2);
            const uint32_t w = qrow[(long)(h < hmax ? h : 0) * (D / 2) + c];
            *(float2*)(sq + h * D + 2 * c) = make_float2(bf16w_lo(w), bf16w_hi(w));
        }
        if (tid < 32) tab[tid] = kDsExp2fTab[tid];
        if (tid < hmax) sden[tid] = a.sink[(long)(hg0 + tid) * a.sks];   /* the sinks, until the den */
        if constexpr (NVT) {   /* the packed direct PV's decode tables */
            for (int c = tid; c < 256; c += NT) {
                ((uint2*)(r4d_ds41_sparse_dyn + L.nvt))[c] = ds_nv4_tab((uint32_t)c);
                ((float*)(r4d_ds41_sparse_dyn + L.nvt + 256 * 8))[c] = ds_e4m3_half((uint32_t)c);
            }
        }
    }
    lds_barrier();

    /* ---- scores: a lane a key, every dot in d order. A wave task is (32 keys, HS heads); the
     * NSG tasks of a key block go to neighbouring waves, so the block's second read hits L0.
     * Four heads read q by scalar loads (QS; LDS-bound otherwise), fewer from LDS. */
    /* Key blocks [0, nwb) are plain bf16; with a packed latent (a.cnv, the window a multiple of 32)
     * blocks [nwb, nkb) are the packed compressed entries, scored by the same task with packed
     * loads unpacked to the same 16-byte bf16 words right before the dot: two loops, no branch
     * around a load in either. */
    const int nwb = NV ? a.win / 32 : nkb;
    auto score_task = [&](int t, auto packed) {
        constexpr bool PK = decltype(packed)::value;
        const int kb = t / NSG, g = t % NSG;
        const int k = kb * 32 + lane;
        const uint16_t* kk = kp[k];
        const bool have = kk != zero;
        const float* qg = sq + g * HS * D;
        float acc[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) acc[h] = 0.0f;
        const uint32_t* qw = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        int ho[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) ho[h] = (g * HS + h < hmax ? g * HS + h : 0) * (D / 2);
        [[maybe_unused]] uint4 ra[8], rb[8];
        [[maybe_unused]] DsNv4Raw pa, pb;
        if constexpr (PK) pa = ds_nv4_ld64(kk, 0);
        else {
#pragma unroll
            for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16(kk, i);
        }
        /* No branch around a load (the wait counts would fall back to 0): the last pass reloads
         * its own first half, unused. */
#pragma unroll 1
        for (int c = 0; c < D / 8; c += 16) {
            if constexpr (PK) {
                /* packed: each word unpacked right before its fmas (ds_dec_dot_nv) */
                const int cn = c + 16 < D / 8 ? c + 16 : c;
                pb = ds_nv4_ld64(kk, c + 8);
                ds_dec_dot_nv<HS, QS>(acc, pa, qg + c * 8, qw + c * 4, ho);
                pa = ds_nv4_ld64(kk, cn);
                ds_dec_dot_nv<HS, QS>(acc, pb, qg + (c + 8) * 8, qw + (c + 8) * 4, ho);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) rb[i] = ds_ldg16(kk, c + 8 + i);
                if constexpr (QS) ds_dec_dot_s<HS>(acc, ra, qw + c * 4, ho);
                else ds_dec_dot<HS>(acc, ra, qg + c * 8);
                const int cn = c + 16 < D / 8 ? c + 16 : c;
#pragma unroll
                for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16(kk, cn + i);
                if constexpr (QS) ds_dec_dot_s<HS>(acc, rb, qw + (c + 8) * 4, ho);
                else ds_dec_dot<HS>(acc, rb, qg + (c + 8) * 8);
            }
        }
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const int hh = g * HS + h;
            const float v = have ? acc[h] * a.scale : -kDsInf;
            sp[(long)hh * nkq + k] = v;
            const float t2 = r4d_wave_max(-1e30f < v ? v : -1e30f);
            if (lane == 0) bmax[hh * nkb + kb] = t2;
        }
    };
    {
        int t = wave;
        for (; t < nwb * NSG; t += NW) score_task(t, std::false_type{});
        if constexpr (NV)
            for (; t < nkb * NSG; t += NW) score_task(t, std::true_type{});
    }

    /* PV through LDS tiles: the first tiles' loads go out now, under the softmax. A thread carries
     * one 16-byte piece (key lk of the tile, piece lc) of each of P tiles. No load sits under a
     * branch (the wait counts would fall back to 0): past the last tile they read the zero row. */
    const int lk = tid >> 6, lc = tid & 63;
    uint4 rg[PVD ? 1 : P];
    /* NV: a tile of packed keys (tile t * TK >= win) carries its piece as the code word (byte 4 lc)
     * and the scale byte (256 + lc / 2), unpacked when it goes to LDS; every load reads either the
     * key or the zero row, so none sits under a branch. */
    [[maybe_unused]] uint32_t rw[NV && !PVD ? P : 1], rs[NV && !PVD ? P : 1];
    [[maybe_unused]] const int ptl = NV ? a.win / TK : ntl;   /* tiles [0, ptl) plain */
    auto ldp = [&](int s, int tt, const uint16_t* pp) {      /* tile tt's piece into ring slot s */
        if constexpr (NV) {
            const bool pk = tt >= ptl;
            rg[s] = ds_ldg16(pk ? zero : pp, lc);
            const __attribute__((address_space(1))) const unsigned char* b =
                (const __attribute__((address_space(1))) unsigned char*)(pk ? pp : zero);
            rw[s] = *(const __attribute__((address_space(1))) uint32_t*)(b + 4 * lc);
            rs[s] = b[kDsAttnD / 2 + lc / 2];
        } else {
            (void)tt;
            rg[s] = ds_ldg16(pp, lc);
        }
    };
    auto piece = [&](int s, int tt) -> uint4 {              /* ring slot s (tile tt) as 16 bf16 bytes */
        if constexpr (NV) return tt >= ptl ? ds_nv4_unpack8_fast(rw[s], ds_e4m3_half(rs[s])) : rg[s];
        else { (void)tt; return rg[s]; }
    };
    if constexpr (!PVD) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
            const uint16_t* pp = kp[min(s, ntl - 1) * TK + lk];
            ldp(s, s, s < ntl ? pp : zero);
        }
    }
    lds_barrier();

    /* ---- per head: the max, then p in place (0 for no key) */
#pragma unroll 1
    for (int h = 0; h < HG; ++h) {
        float t = -1e30f;
        for (int j = lane; j < nkb; j += 32) t = fmaxf(t, bmax[h * nkb + j]);
        const float mx = r4d_wave_max(t);
        if (tid == 0) smx[h] = mx;
        float* sh = sp + (long)h * nkq;
        for (int k = tid; k < nkq; k += NT) sh[k] = kp[k] != zero ? ds_expf_tab(sh[k] - mx, tab) : 0.0f;
    }
    lds_barrier();                                       /* sq and bmax are dead: tiles may land */

    /* ---- sum p k and sum p, key by key in libref's order */
    const int cg = tid % NCG, hp = tid / NCG;
    const int dcol = cg * DPT;
    float acc[HPT][DPT], den[HPT];
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        den[j] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d) acc[j][d] = 0.0f;
    }
    const bool pvt = hp < NHP;                           /* wave-uniform: NCG is a multiple of 32 */
    const bool dw = cg < 32;
    const float* ph = sp + (long)(pvt ? hp * HPT : 0) * nkq;
    if constexpr (PVD) {
        /* Straight from global: a wave walks 32-key blocks, lane L holding block key L's pointer
         * (handed to the whole wave by readlane), one block's loads in flight while the previous
         * block is summed. Past the last block the loads read the zero row. The plain blocks
         * [0, nwb) walk so; the packed ones (a.cnv) after them in 16-key slots (below). */
        if (pvt) {
            auto walk = [&](int kb0, int kb1) {
                if (kb0 >= kb1) return;
                uint32_t kv[2][32];
                const uint16_t* lp = kp[kb0 * 32 + lane];
                ds_dec_pvld<DPT>(kv[0], lp, dcol);
#pragma unroll 1
                for (int kb = kb0; kb < kb1; kb += 2) {
                    lp = kb + 1 < kb1 ? kp[(kb + 1) * 32 + lane] : zero;
                    ds_dec_pvld<DPT>(kv[1], lp, dcol);
                    ds_dec_pvsum<HPT, DPT>(acc, den, kv[0], ph, nkq, kb * 32, dw);
                    lp = kb + 2 < kb1 ? kp[(kb + 2) * 32 + lane] : zero;
                    ds_dec_pvld<DPT>(kv[0], lp, dcol);
                    if (kb + 1 < kb1) ds_dec_pvsum<HPT, DPT>(acc, den, kv[1], ph, nkq, (kb + 1) * 32, dw);
                }
            };
            walk(0, nwb);
            /* The packed blocks [nwb, nkb), a block (32 keys, one register a key: ds_dec_pvld_nv) a
             * step: the next block's loads go out while this one is summed, in key order, as the
             * plain walk above; past the last block the loads read the zero row (no load under a
             * branch). The next loads sit behind the fence of the previous block's sums. */
            if constexpr (NV) {
                if (nwb < nkb) {
                    const uint32_t cb = (uint32_t)dcol / 2u, sb = (uint32_t)(kDsAttnD / 2 + dcol / 16);
                    [[maybe_unused]] const uint2* tc = (const uint2*)(r4d_ds41_sparse_dyn + L.nvt);
                    [[maybe_unused]] const float* ts = (const float*)(r4d_ds41_sparse_dyn + L.nvt + 256 * 8);
                    uint32_t kv[2][32];
                    const uint16_t* lp = kp[nwb * 32 + lane];
                    ds_dec_pvld_nv<32>(kv[0], lp, cb, sb, 0);
#pragma unroll 1
                    for (int kb = nwb; kb < nkb; kb += 2) {
                        lp = kb + 1 < nkb ? kp[(kb + 1) * 32 + lane] : zero;
                        lp = ds_dec_pfence<HPT, DPT>(lp, acc);
                        ds_dec_pvld_nv<32>(kv[1], lp, cb, sb, 0);
                        if constexpr (NVT) ds_dec_pvsum_nvt<HPT, DPT, 32>(acc, den, kv[0], ph, nkq, kb * 32, dw, tc, ts);
                        else ds_dec_pvsum_nv<HPT, DPT, 32>(acc, den, kv[0], ph, nkq, kb * 32, dw);
                        lp = kb + 2 < nkb ? kp[(kb + 2) * 32 + lane] : zero;
                        lp = ds_dec_pfence<HPT, DPT>(lp, acc);
                        ds_dec_pvld_nv<32>(kv[0], lp, cb, sb, 0);
                        if (kb + 1 < nkb) {
                            if constexpr (NVT) ds_dec_pvsum_nvt<HPT, DPT, 32>(acc, den, kv[1], ph, nkq, (kb + 1) * 32, dw, tc, ts);
                            else ds_dec_pvsum_nv<HPT, DPT, 32>(acc, den, kv[1], ph, nkq, (kb + 1) * 32, dw);
                        }
                    }
                }
            }
        }
    } else {
        *(uint4*)(tiles + lk * (D * 2) + lc * 16) = piece(0, 0);
        {
            const uint16_t* pp = kp[min(P, ntl - 1) * TK + lk];
            ldp(0, P, P < ntl ? pp : zero);
        }
        lds_barrier();
#pragma unroll 1
        for (int t0 = 0; t0 < ntl; t0 += P) {
#pragma unroll
            for (int s = 0; s < P; ++s) {
                const int t = t0 + s;
                /* the pointer of the ring's next load first, off the tail of the step */
                const uint16_t* pp = kp[min(t + 1 + P, ntl - 1) * TK + lk];
                if (pvt && t < ntl) {
                    const unsigned char* tb = tiles + (t & 1) * (TK * D * 2) + dcol * 2;
#pragma unroll
                    for (int u0 = 0; u0 < TK; u0 += 4) {
                        uint32_t kw[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            if constexpr (DPT == 2) kw[u] = *(const uint32_t*)(tb + (u0 + u) * (D * 2));
                            else kw[u] = *(const uint16_t*)(tb + (u0 + u) * (D * 2));
                        }
                        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, t * TK + u0, dw);
                    }
                }
                /* tile t+1 into the other buffer (read last before the previous barrier) */
                const int sn = (s + 1) % P;
                *(uint4*)(tiles + ((t + 1) & 1) * (TK * D * 2) + lk * (D * 2) + lc * 16) = piece(sn, t + 1);
                ldp(sn, t + 1 + P, t + 1 + P < ntl ? pp : zero);
                lds_barrier();
            }
        }
    }
    if (pvt && cg == 0) {
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h < hmax) sden[h] = den[j] + ds_expf_tab(sden[h] - smx[h], tab);
        }
    }
    lds_barrier();
    if (pvt) {
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
        [[maybe_unused]] bool orot = false;
        [[maybe_unused]] float oc = 0.0f, osn = 0.0f;
        if constexpr (ORT) {   /* rope_rt_kernel's pair test and angles, for this thread's pair */
            const long p = a.pos[m * a.ps];
            const int d0 = dcol - a.o.off;
            orot = d0 >= 0 && d0 < 2 * a.o.half && p >= 0 && p < a.o.tab_rows;
            if (orot) {
                const int i = d0 >> 1;
                oc = a.o.cs[p * a.o.cp + (long)i * a.o.cst];
                osn = a.o.sign * a.o.cs[p * a.o.cp + (long)(a.o.half + i) * a.o.cst];
            }
        }
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                [[maybe_unused]] uint32_t lo = ds_bf16_bits(acc[j][0] / dn), hi = ds_bf16_bits(acc[j][DPT - 1] / dn);
                if constexpr (ORT) ds_ort_pair(lo, hi, orot, oc, osn, a.o.fmt, a.o.sc);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[j][0] / dn);
            }
        }
    }
}

/* ================================================================== attn_sparse_sink on WMMA
 *
 * Passes of more than kWmRows rows (prefill) on v_wmma_f32_16x16x16_bf16. A workgroup is a
 * (16-head tile, row); its four waves own 128 latent columns each and walk the row's keys 16 at
 * a time:
 *   S^T = K Q^T     over the wave's 128 columns: A = 16 keys as stored (8 columns a lane), B = Q^T
 *                   held in registers. The four partials meet in LDS and every wave sums them in
 *                   wave order, so all four hold the same scores: lane L head L%16, keys 8(L/16)+e.
 *   softmax         online: a running max per head (floored at -1e30), p = exp2(s log2e - max) by
 *                   v_exp_f32, the accumulators rescaled when the max moves, den summing f32 p.
 *   O^T += K^T P^T  A = K^T by global_load_tr_b128 (lane L column L%16, keys 8(L/16)+e), B = P^T
 *                   straight from the score layout, P = hi + lo in bf16 (two WMMAs; hi truncated,
 *                   lo = p - hi rounded, so |p - hi - lo| <= 2^-17 p).
 * Not libref's bits: q.k products are exact but sum in WMMA order, the exponential is v_exp_f32
 * and the softmax online; the old kernel stays for the decode band. A key
 * with no entry reads a zero row and scores -inf (0 * NaN from a recycled slot would be NaN). */
typedef __bf16 ds_v8bf __attribute__((ext_vector_type(8)));
typedef float ds_v8f __attribute__((ext_vector_type(8)));
typedef short ds_v8s __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(1))) ds_v8s ds_gv8s;

constexpr int kWmRows = 48;                          /* passes up to this many rows stay scalar */
constexpr int kWmWaves = 4;
constexpr int kWmCols = kDsAttnD / kWmWaves;         /* latent columns a wave */
constexpr int kWmSbufBytes = 2 * kWmWaves * 32 * 8 * 4;

__host__ __device__ constexpr long ds_wm_mask_off(int nkt) { return (long)nkt * 16 * 8; }
__host__ __device__ constexpr long ds_wm_sbuf_off(int nkt) {
    return (ds_wm_mask_off(nkt) + (long)nkt * 4 + 15) / 16 * 16;
}

__device__ __forceinline__ ds_v8f ds_wmma(uint4 a, uint4 b, ds_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ds_v8bf, a),
                                                             __builtin_bit_cast(ds_v8bf, b), c);
}
/* global_load_tr_b128: lane 8g+e gives the address of 8 consecutive bf16 and lane 8g+i receives
 * value i of lanes 8g..8g+7, in lane order (an 8x8 transpose within each group of eight lanes). */
__device__ __forceinline__ uint4 ds_load_tr(const uint16_t* p) {
    return __builtin_bit_cast(uint4, __builtin_amdgcn_global_load_tr_b128_v8i16((ds_gv8s*)p));
}
__device__ __forceinline__ float ds_xor16(float v) {
    const int x = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
}
__device__ __forceinline__ float ds_ex2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ uint32_t ds_rne_hi(float f) {        /* bf16 of a finite f, high half */
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}

/* NV (latent_format nvfp4): a 16-key tile at or past the window (win a multiple of 16) holds packed
 * entries. Its K rows (kn) are the 32 code bytes and 4 scale bytes of the lane's 64 columns, unpacked
 * to the same bf16 words; its transposed rows (kt) come through this wave's 4 KiB of LDS (past the
 * score buffers): every lane writes its key's 64 values column-major, [128 columns][16 keys], and
 * reads kt[j] as the 8 keys global_load_tr would hand it, one 16-byte read each. */
constexpr int kWmNvBytes = kWmWaves * kWmCols * 16 * 2;

template <bool LO, bool NV>
__global__ __launch_bounds__(128) void ds41_attn_sparse_sink_wm_kernel(DsAttnArgs a, float sl2) {
    constexpr int D = kDsAttnD;
    constexpr float kL2e = 1.4426950408889634f;
    const int nkt = (a.nk + 15) >> 4, nkk = nkt * 16;
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    uint32_t* vmask = (uint32_t*)(r4d_ds41_sparse_dyn + ds_wm_mask_off(nkt));
    float4* sbuf = (float4*)(r4d_ds41_sparse_dyn + ds_wm_sbuf_off(nkt));
    const int tid = threadIdx.x, lane = tid & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hl = blockIdx.x * 16 + (lane & 15);
    const bool hok = hl < a.H;
    const int c0 = wave * kWmCols;
    const int kq = lane & 15;                        /* S^T's A row: key kq of the tile */
    const int ktr = 8 * hf + (lane & 7);             /* K^T's transposing row: key ktr */
    const int dtr = c0 + 8 * ((lane >> 3) & 1);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        /* ---- the row's keys in libref's order (the window, then the selections), a null one
         * pointing at a zero row; a 16-bit mask a tile of which are keys */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k0 = 0; k0 < nkk; k0 += 128) {
            const int k = k0 + tid;
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = a.cc + (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)a.cslot;
                }
            }
            if (k < nkk) kp[k] = ptr ? ptr : kDsZeroKey;
            const uint32_t b = (uint32_t)__builtin_amdgcn_ballot_w32(ptr != nullptr);
            const int t0 = (k0 + wave * 32) >> 4;
            if (lane == 0) {
                if (t0 < nkt) vmask[t0] = b & 0xffffu;
                if (t0 + 1 < nkt) vmask[t0 + 1] = b >> 16;
            }
        }
        __syncthreads();

        /* ---- Q^T for this lane's head and the wave's columns: step i, half hf = columns
         * c0 + 64 hf + 8 i .. + 7 (the same split as the key rows below) */
        uint4 qf[8];
        {
            const uint16_t* qp = a.q + m * a.qr + (long)(hok ? hl : 0) * D + c0 + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qf[i] = *(const uint4*)(qp + 8 * i);
                if (!hok) qf[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        ds_v8f o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (ds_v8f)(0.0f);
        float mrun = -1e30f, den = 0.0f;
        uint4 kn[8];
        [[maybe_unused]] DsNv4Raw kr;                /* NV: the next packed tile's raw K row */
        [[maybe_unused]] unsigned char* tw = (unsigned char*)sbuf + kWmSbufBytes + wave * (kWmCols * 16 * 2);
        [[maybe_unused]] const int ptk = NV ? a.win / 16 : nkt;   /* tiles [0, ptk) plain */
        auto ldk = [&](int tt) {                     /* tile tt's K row: kn, or (packed) kr */
            const uint16_t* pk = kp[tt * 16 + kq];
            if (NV && tt >= ptk) {
                const unsigned char* b = (const unsigned char*)pk;
                kr.c0 = *(const uint4*)(b + (c0 + 64 * hf) / 2);
                kr.c1 = *(const uint4*)(b + (c0 + 64 * hf) / 2 + 16);
                kr.sc = *(const uint32_t*)(b + D / 2 + (c0 + 64 * hf) / 16);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) kn[i] = *(const uint4*)(pk + c0 + 64 * hf + 8 * i);
            }
        };
        ldk(0);
        for (int t = 0; t < nkt; ++t) {
            uint4 kt[8];
            if (NV && t >= ptk) {                    /* tile-uniform */
                ds_nv4_unpack64(kr, kn);
                /* key kq's columns c0 + 64 hf + 8 i + e -> T[64 hf + 8 i + e][kq] */
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t w[4] = { kn[i].x, kn[i].y, kn[i].z, kn[i].w };
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int col = 64 * hf + 8 * i + e;
                        *(uint16_t*)(tw + (col * 16 + kq) * 2) = (uint16_t)((e & 1) ? (w[e / 2] >> 16) : (w[e / 2] & 0xffffu));
                    }
                }
                ds_wave_sync();
                /* kt[j] of lane L: keys 8 (L / 16) .. + 7 at column 8 ((L / 8) & 1) + 16 j + L % 8 */
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int col = 8 * ((lane >> 3) & 1) + 16 * j + (lane & 7);
                    kt[j] = *(const uint4*)(tw + (col * 16 + 8 * hf) * 2);
                }
                ds_wave_sync();                      /* the reads before the next tile's writes */
            } else {
                const uint16_t* pt = kp[t * 16 + ktr] + dtr;
#pragma unroll
                for (int j = 0; j < 8; ++j) kt[j] = ds_load_tr(pt + 16 * j);
            }
            ds_v8f sp = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) sp = ds_wmma(kn[i], qf[i], sp);
            if (t + 1 < nkt) ldk(t + 1);
            float4* sb = sbuf + (t & 1) * (kWmWaves * 64);
            sb[wave * 64 + lane] = make_float4(sp[0], sp[1], sp[2], sp[3]);
            sb[wave * 64 + 32 + lane] = make_float4(sp[4], sp[5], sp[6], sp[7]);
            lds_barrier();
            float x[8];
            {
                const float4 l0 = sb[lane], h0 = sb[32 + lane];
                x[0] = l0.x; x[1] = l0.y; x[2] = l0.z; x[3] = l0.w;
                x[4] = h0.x; x[5] = h0.y; x[6] = h0.z; x[7] = h0.w;
#pragma unroll
                for (int w = 1; w < kWmWaves; ++w) {
                    const float4 l = sb[w * 64 + lane], h = sb[w * 64 + 32 + lane];
                    x[0] += l.x; x[1] += l.y; x[2] += l.z; x[3] += l.w;
                    x[4] += h.x; x[5] += h.y; x[6] += h.z; x[7] += h.w;
                }
            }
            const uint32_t vm = vmask[t] >> (8 * hf);
            float tmax = -kDsInf;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = ((vm >> e) & 1u) ? x[e] * sl2 : -kDsInf;
                tmax = fmaxf(tmax, x[e]);
            }
            tmax = fmaxf(tmax, ds_xor16(tmax));
            const float mnew = fmaxf(mrun, tmax);
            const float alpha = ds_ex2(mrun - mnew);
            float pe[8], ps = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pe[e] = ds_ex2(x[e] - mnew);
                ps += pe[e];
            }
            den = den * alpha + ps;
            if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] *= alpha;
            }
            mrun = mnew;
            uint32_t hw[4], lw[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float p0 = pe[2 * q], p1 = pe[2 * q + 1];
                if constexpr (LO) {
                    const uint32_t u0 = __float_as_uint(p0) & 0xffff0000u, u1 = __float_as_uint(p1) & 0xffff0000u;
                    hw[q] = __builtin_amdgcn_perm(u1, u0, 0x07060302u);
                    lw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1 - __uint_as_float(u1)),
                                                  ds_rne_hi(p0 - __uint_as_float(u0)), 0x07060302u);
                } else {
                    hw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1), ds_rne_hi(p0), 0x07060302u);
                }
            }
            const uint4 ph = make_uint4(hw[0], hw[1], hw[2], hw[3]);
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], ph, o[j]);
            if constexpr (LO) {
                const uint4 pl = make_uint4(lw[0], lw[1], lw[2], lw[3]);
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], pl, o[j]);
            }
        }
        /* ---- den over both key halves, the sink, one rounding: lane L holds head L%16 at
         * columns c0 + 16 j + 8 hf + e */
        float dt = den + ds_xor16(den);
        if (hok) {
            dt += ds_ex2(a.sink[(long)hl * a.sks] * kL2e - mrun);
            uint16_t* op = a.out + m * a.outr + (long)hl * D + c0 + 8 * hf;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    w[q] = ds_bf16_bits(o[j][2 * q] / dt) | (ds_bf16_bits(o[j][2 * q + 1] / dt) << 16);
                *(uint4*)(op + 16 * j) = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== index_score on WMMA
 *
 * Passes of more than kWmRows rows. A wave is a row and walks 16-entry tiles of its workgroup's
 * kIwEB entries: S = Q K^T (A = the row's 16-head tiles of q, B = 16 entries as stored; 8 steps of
 * 16 dims, q held in registers), D lane L = entry L%16, heads 8(L/16)+e. Then libref's roundings:
 * the dot to bf16, relu x the rounded head weight, rounded, summed in head order (the running sum
 * hops lanes L <-> L^16 between 8-head runs), rounded. Only the dot's summation order differs
 * from libref, so a score moves only where the f32 dot sat at a bf16 rounding boundary.
 *
 * KEEP (the candidate restriction): a workgroup's 8 waves walk kIwKeepEB entries each instead, and
 * a 16-entry tile none of whose entries is kept is skipped (-inf written, no key read, no WMMA).
 * A kept tile runs exactly the instructions of the KEEP = false form on the same operands, an
 * unkept entry in it reading the zero key (a WMMA column is its own entry's), and an unkept entry
 * scores -inf: so every kept entry has the KEEP = false form's bits. The wave's keep words over
 * its span sit one a lane: a 1024-entry span aligned to 1024 touches at most 32 words at any
 * block (exactly 32 at block 1, at most 18 at block >= 2). */
constexpr int kIwEB = 128;                           /* entries a workgroup */
constexpr int kIwKeepEB = 1024;                      /* ... with keep */
static_assert(kIwKeepEB == 1024, "the keep words of a span, one a lane, assume 1024-entry spans");

template <int HT, bool KEEP>
__global__ __launch_bounds__(256) void ds41_index_score_wm_kernel(DsScoreArgs a) {
    constexpr int DM = 128, EB = KEEP ? kIwKeepEB : kIwEB;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const long m = (long)blockIdx.y * 8 + wave;
    if (m >= a.M) return;
    const long jb = (long)blockIdx.x * EB;
    const long jend = jb + EB < (long)a.C ? jb + EB : (long)a.C;
    float* srow = a.sc + m * a.sr;
    const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
    if (jb >= reach) {
        for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
        return;
    }
    /* KEEP: lane i holds the row's keep word kw0 + i of the words over [jb, min(jend, reach)), 0
     * past them; none set is a span with no kept entry, -inf without a key read (wave-uniform). */
    [[maybe_unused]] unsigned kword = 0u;
    [[maybe_unused]] long kw0 = 0;
    if constexpr (KEEP) {
        const int* krow = a.keep + m * a.kr;
        const long jl = (jend < reach ? jend : reach) - 1;
        kw0 = ds_keep_block(a, jb) >> 5;
        const long nw = (ds_keep_block(a, jl) >> 5) - kw0 + 1;
        kword = lane < nw ? (unsigned)krow[(kw0 + lane) * a.kcs] : 0u;
        if (__builtin_amdgcn_ballot_w32(kword != 0u) == 0u) {
            for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            return;
        }
    }
    const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
    uint4 qf[HT][8];
    float wh[HT][8];
#pragma unroll
    for (int t = 0; t < HT; ++t) {
        const int h = t * 16 + (lane & 15);
        const uint16_t* qp = a.q + m * a.qr + (long)(h < a.H ? h : 0) * DM + 64 * hf;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qf[t][i] = *(const uint4*)(qp + 8 * i);
            if (h >= a.H) qf[t][i] = make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int hh = t * 16 + 8 * hf + e;
            wh[t][e] = hh < a.H ? ds_round_bf16(ds_bf(a.w[m * a.wr + (long)hh * a.wc]) * a.wscale) : 0.0f;
        }
    }
    for (long j0 = jb; j0 < jend; j0 += 16) {
        if (j0 >= reach) {                           /* wave-uniform: the rest is past the row */
            for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            break;
        }
        const long j = j0 + (lane & 15);
        bool live = j < jend && j < reach;
        if constexpr (KEEP) {
            /* This entry's bit from the lane holding its word; a tile with none kept is skipped
             * (wave-uniform: lanes L and L^16 hold the same entry). */
            long wi = live ? (ds_keep_block(a, j) >> 5) - kw0 : 0;
            wi = wi < 0 ? 0 : (wi > 31 ? 31 : wi);
            const unsigned wd = (unsigned)__builtin_amdgcn_ds_bpermute((int)wi << 2, (int)kword);
            live = live && ((wd >> (ds_keep_block(a, j) & 31)) & 1u) != 0u;
            if (__builtin_amdgcn_ballot_w32(live) == 0u) {
                if (hf == 1 && j < jend) srow[j * a.scs] = -kDsInf;
                continue;
            }
        }
        const uint16_t* kp = nullptr;
        if (live) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long blk = a.bt[(long)s * a.btr + col * a.btc];
                if (blk >= 0 && blk < a.nblk) kp = a.kc + (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
            }
        }
        uint4 kb[8];
        if (a.kmx) {   /* this lane's half: code bytes [32 hf, 32 hf + 32), scale bytes 2 hf, 2 hf + 1 */
            const unsigned char* pb = (const unsigned char*)(kp ? kp : kDsZeroKey);
            const uint4 c0 = *(const uint4*)(pb + 32 * hf), c1 = *(const uint4*)(pb + 32 * hf + 16);
            const uint32_t e0 = pb[64 + 2 * hf], e1 = pb[64 + 2 * hf + 1];
            const uint32_t cw[8] = { c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w };
#pragma unroll
            for (int i = 0; i < 8; ++i) kb[i] = ds_mx4_unpack8(cw[i], i < 4 ? e0 : e1);
        } else {
            const uint16_t* pk = (kp ? kp : kDsZeroKey) + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) kb[i] = *(const uint4*)(pk + 8 * i);
        }
        ds_v8f d[HT];
#pragma unroll
        for (int t = 0; t < HT; ++t) {
            d[t] = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) d[t] = ds_wmma(qf[t][i], kb[i], d[t]);
        }
        /* heads 0-7 in lane L, 8-15 in L^16, then 16-23 back in L and 24-31 in L^16 */
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < HT; ++t) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (t + half > 0) acc = ds_xor16(acc);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float dot = ds_round_bf16(d[t][e]);
                    acc = acc + ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh[t][e]);
                }
            }
        }
        if (hf == 1 && j < jend) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ================================================================== entry_store, mxfp4 */
/* entry_store's format mxfp4: one thread a value, a wave a 32-value block, a workgroup a row (n / 32
 * waves): the block's ue8m0 scale by quant_roundtrip's rule from its amax, the e2m1 codes of v /
 * scale (exact for a power of two), two a byte, then the scale bytes and the zero pad (layout above
 * ds_mx4_bytes). */
__global__ __launch_bounds__(1024) void ds41_entry_store_mx4_kernel(const uint16_t* x, long xp, const int32_t* page,
                                                                    long pgs, unsigned char* cache, long blocks,
                                                                    int n, long W) {
    const int lane = threadIdx.x & 31, b = threadIdx.x >> 5;
    const int bytes = ds_mx4_bytes(n);
    for (long w = blockIdx.x; w < W; w += gridDim.x) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;        /* block-uniform */
        const float v = ds_bf(x[w * xp + (long)b * 32 + lane]);
        float am = v < 0.0f ? -v : v;
        am = am == am ? am : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float t = __shfl_xor(am, o);
            am = t > am ? t : am;
        }
        am = am > 6.0f * 0x1p-126f ? am : 6.0f * 0x1p-126f;
        const uint32_t u = __builtin_bit_cast(uint32_t, am * (1.0f / 6.0f));
        const int k = (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
        const float scale = __builtin_bit_cast(float, (uint32_t)(k + 127) << 23);
        const uint32_t c = ds_e2m1_code(v / scale);
        const uint32_t co = (uint32_t)__shfl_xor((int)c, 1);
        unsigned char* e = cache + pg * (long)bytes;
        if (!(lane & 1)) e[b * 16 + lane / 2] = (unsigned char)(c | (co << 4));
        if (lane == 0) e[n / 2 + b] = (unsigned char)(k + 127);
        if (b == 0 && n / 2 + n / 32 + lane < bytes) e[n / 2 + n / 32 + lane] = 0;
    }
}

/* ================================================================== entry_store, nvfp4 */
/* entry_store's format nvfp4: one thread a value, sixteen lanes a block (n / 16 blocks, n threads a
 * workgroup, a workgroup a row). The block's scale: among amax / c (c = 6, 4, 3, 2, 1.5, 1, 0.5) the
 * first that is an e4m3 value and gives every value of the block back bit for bit (ds_nv4_bf16 of
 * its code); none (a block quant_roundtrip did not produce) -> the rule's fp8(max(amax, 6 x 2^-9) /
 * 6), lossy. Then the codes (e2m1 of v / scale), two a byte, the scale bytes, the zero pad. */
__global__ __launch_bounds__(1024) void ds41_entry_store_nv4_kernel(const uint16_t* x, long xp, const int32_t* page,
                                                                    long pgs, unsigned char* cache, long blocks,
                                                                    int n, long W) {
    const int lane = threadIdx.x & 31, i = threadIdx.x, blk = i >> 4;
    const int bytes = ds_nv4_bytes(n);
    const unsigned half = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
    const float kCand[7] = { 6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f };
    for (long w = blockIdx.x; w < W; w += gridDim.x) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;        /* block-uniform */
        const uint32_t bits = x[w * xp + i];
        const float v = ds_bf((uint16_t)bits);
        float am = v < 0.0f ? -v : v;
        am = am == am ? am : 0.0f;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            const float t = __shfl_xor(am, o);
            am = t > am ? t : am;
        }
        uint32_t sb = ds_f32_e4m3((am > 6.0f * 0x1p-9f ? am : 6.0f * 0x1p-9f) / 6.0f);   /* the rule */
        bool found = am == 0.0f;
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            const float sc = am / kCand[c];
            const uint32_t b8 = ds_f32_e4m3(sc);
            const bool exact = am > 0.0f && ds_e4m3_f32(b8) == sc;
            const bool ok = exact && ds_nv4_bf16(ds_e2m1_code(v / sc), sc) == bits;
            const unsigned all = (unsigned)__builtin_amdgcn_ballot_w32(ok) & half;
            if (!found && all == half) { found = true; sb = b8; }
        }
        const float scale = ds_e4m3_f32(sb);
        const float q = v / scale;
        const uint32_t c = ds_e2m1_code(q != q ? q : (q > 6.0f ? 6.0f : (q < -6.0f ? -6.0f : q)));
        const uint32_t co = (uint32_t)__shfl_xor((int)c, 1);
        unsigned char* e = cache + pg * (long)bytes;
        if (!(lane & 1)) e[i / 2] = (unsigned char)(c | (co << 4));
        if ((lane & 15) == 0) e[n / 2 + blk] = (unsigned char)sb;
        if (i < bytes - n / 2 - n / 16) e[n / 2 + n / 16 + i] = 0;
    }
}

/* ================================================================== launch shims */
/* Linear addressing of an index operand: rank 1 by its stride, anything else only if dense. */
long lin_stride(const RadTensor& t) {
    if (t.rank == 1) return (long)t.stride[0];
    return r4d_contig(&t) ? 1 : -1;
}

bool aligned(const void* p, unsigned bytes) { return ((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0; }

constexpr int kDsChunk = 8192;                       /* index_topk's split: positions a chunk */
constexpr int kDsSplitRows = 16;                     /* ... at up to this many rows */
constexpr int kDsSplitMinChunks = 4;                 /* ... over at least this many chunks */
constexpr int kDsSplitMaxK = 2048;

/* index_candidates' keep words a row: a bit a block of `block` positions over `width`. */
long long keep_words(long long width, long long block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

/* The split's scratch: candidates [rows][nch][K] u64, counts [rows][nch] i32, used chunks [rows] i32. */
int64_t topk_split_bytes(long rows, long C, long K) {
    const long nch = (C + kDsChunk - 1) / kDsChunk;
    if (rows < 1 || nch < kDsSplitMinChunks || K < 1 || K > kDsSplitMaxK) return 0;
    return (int64_t)rows * nch * K * 8 + (int64_t)rows * nch * 4 + (int64_t)rows * 4 + 16;
}

int index_score_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), Dm = r4d_geti_or(a, "dim", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long part = r4d_geti_or(a, "row_part", -1);
    const long long block = r4d_geti_or(a, "block", 0);
    const float wscale = (float)r4d_getf(a, "wscale", 0.0);
    for (int i = 0; i < 7; ++i)
        if (!a->t[i].data) return R4D_NO(RAD_E_INVAL);
    const RadTensor* keep = a->n_t > 7 && a->t[7].data ? &a->t[7] : nullptr;
    /* key_format: bf16 (Dm values an entry) or mxfp4 (ds_mx4_bytes(Dm) bytes an entry, unpacked to the
     * same bf16); an entry is `ratio` slots of the cache's last extent either way. */
    const char* kf = r4d_gets(a, "key_format");
    const bool kmx = kf && !std::strcmp(kf, "mxfp4");
    if (kf && !kmx && std::strcmp(kf, "bf16")) return R4D_NO(RAD_E_UNSUPPORTED);
    const long kslots = kmx ? (long)ds_mx4_bytes((int)Dm) / 2 : (long)Dm;
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || !std::isfinite(wscale) || q.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] ||
        w.shape[1] != H || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != kslots ||
        kc.shape[2] % ratio || bt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] < 1 || (part < 0 && sc.shape[0] != q.shape[0]) || sc.shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || w.dtype != RAD_BF16 || kc.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        pos.dtype != RAD_I32 || cu.dtype != RAD_I32 || sc.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* This row: 128-wide keys read 16 bytes at a time, q read as bf16 pairs. */
    if (Dm != 128 || ratio > 16) return R4D_NO(RAD_E_UNSUPPORTED);
    const long ps = lin_stride(pos), cs = lin_stride(cu);
    if (ps < 0 || cs < 0 || q.stride[1] != 1 || (q.stride[0] & 1) || !aligned(q.data, 4) ||
        !aligned(kc.data, 16) || q.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 ||
        r4d_numel(&cu) - 1 > INT_MAX || bt.shape[1] > INT_MAX || H > INT_MAX / 128)
        return R4D_NO(RAD_E_SHAPE);
    if (sc.shape[0] > INT_MAX || (part > 0 && part > (long long)INT_MAX)) return R4D_NO(RAD_E_SHAPE);
    /* A run (row_part p) scores step rows [p R, p R + R), R the score's rows: q, w and pos are read
     * from row p R on; the rows the step does not hold score -inf. */
    const long row0 = part > 0 ? (long)part * (long)sc.shape[0] : 0;
    const long Mv = row0 < (long)q.shape[0]
                        ? ((long)q.shape[0] - row0 < (long)sc.shape[0] ? (long)q.shape[0] - row0
                                                                       : (long)sc.shape[0])
                        : 0;
    const long r0 = Mv > 0 ? row0 : 0;               /* no pointer past the operands */
    /* keep: a row for each row the run holds (at most the score's), keep_words over its width. */
    if (keep && (block < 1 || block > INT_MAX || keep->rank != 2 || keep->dtype != RAD_I32 ||
                 keep->shape[0] < Mv || keep->shape[0] > sc.shape[0] ||
                 keep->shape[1] != keep_words(sc.shape[1], block) || keep->stride[1] < 1))
        return R4D_NO(RAD_E_SHAPE);
    DsScoreArgs s{};
    s.q = (const uint16_t*)q.data + r0 * (long)q.stride[0];
    s.w = (const uint16_t*)w.data + r0 * (long)w.stride[0];
    s.kc = (const uint16_t*)kc.data;
    s.bt = (const int*)bt.data;
    s.pos = (const int*)pos.data + r0 * ps;
    s.cu = (const int*)cu.data;
    s.sc = (float*)sc.data;
    s.row0 = row0;
    s.qr = (long)q.stride[0];
    s.wr = (long)w.stride[0];
    s.wc = (long)w.stride[1];
    s.btr = (long)bt.stride[0];
    s.btc = (long)bt.stride[1];
    s.nblk = (long)kc.shape[0];
    s.ps = ps;
    s.cs = cs;
    s.sr = (long)sc.stride[0];
    s.scs = (long)sc.stride[1];
    s.nbt = (int)bt.shape[1];
    s.S = (int)(r4d_numel(&cu) - 1);
    s.M = (int)sc.shape[0];
    s.Mv = (int)Mv;
    s.C = (int)sc.shape[1];
    s.H = (int)H;
    s.ratio = (int)ratio;
    s.cbs = (int)kc.shape[2];
    s.slot = (int)kc.shape[3];
    s.kmx = kmx ? 1 : 0;
    s.wscale = wscale;
    s.keep = keep ? (const int*)keep->data : nullptr;
    s.kr = keep ? (long)keep->stride[0] : 0;
    s.kcs = keep ? (long)keep->stride[1] : 0;
    s.block = keep ? (int)block : 1;
    s.bshift = -1;
    for (int sh = 0; sh < 31; ++sh)
        if ((1LL << sh) == (long long)s.block) s.bshift = sh;
    /* Prefill heights on WMMA (up to 32 heads, 16-byte q rows); the decode band on the scalar
     * kernel below (libref's bits). The form is picked by the pass height alone, with or without
     * keep, so a kept entry's score is the unrestricted op's. */
    if (s.M > kWmRows && H <= 32 && s.qr % 8 == 0 && aligned(s.q, 16) &&
        (s.M + 7) / 8 <= 65535) {
        if (keep) {
            const dim3 grid((unsigned)((s.C + kIwKeepEB - 1) / kIwKeepEB), (unsigned)((s.M + 7) / 8));
            if (H > 16) return R4D_EMIT(stream, (ds41_index_score_wm_kernel<2, true>), grid, dim3(256), 0u, s);
            return R4D_EMIT(stream, (ds41_index_score_wm_kernel<1, true>), grid, dim3(256), 0u, s);
        }
        const dim3 grid((unsigned)((s.C + kIwEB - 1) / kIwEB), (unsigned)((s.M + 7) / 8));
        if (H > 16) return R4D_EMIT(stream, (ds41_index_score_wm_kernel<2, false>), grid, dim3(256), 0u, s);
        return R4D_EMIT(stream, (ds41_index_score_wm_kernel<1, false>), grid, dim3(256), 0u, s);
    }
    /* Head split: more workgroups while the grid is short of the card. The head sum is closed in
     * head order either way, so every split gives the same bits. With keep the work is a row's kept
     * positions (kDsKeptEst expected), not its width. */
    const long cw = keep ? (s.C < kDsKeptEst ? (long)s.C : kDsKeptEst) : (long)s.C;
    int HS = 1;
    while (HS < (keep ? 4 : 8) && HS * 2 <= H && (cw * HS * 2 + 255) / 256 * s.M <= 1024 &&
           (long)H * (256 / (HS * 2)) * 4 <= 32768)
        HS *= 2;
    const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
    const int epb = 256 / HS;
    const dim3 grid((unsigned)((cw + epb - 1) / epb), gy);
    const uint32_t lds = HS > 1 ? (uint32_t)(H * epb * 4) : 0u;
    if (keep) switch (HS) {
        case 1: return R4D_EMIT(stream, ds41_index_score_kept_kernel<1>, grid, dim3(256), lds, s);
        case 2: return R4D_EMIT(stream, ds41_index_score_kept_kernel<2>, grid, dim3(256), lds, s);
        default: return R4D_EMIT(stream, ds41_index_score_kept_kernel<4>, grid, dim3(256), lds, s);
    }
    switch (HS) {
        case 1: return R4D_EMIT(stream, ds41_index_score_kernel<1>, grid, dim3(256), lds, s);
        case 2: return R4D_EMIT(stream, ds41_index_score_kernel<2>, grid, dim3(256), lds, s);
        case 4: return R4D_EMIT(stream, ds41_index_score_kernel<4>, grid, dim3(256), lds, s);
        default: return R4D_EMIT(stream, ds41_index_score_kernel<8>, grid, dim3(256), lds, s);
    }
}

int index_candidates_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const long long block = r4d_geti_or(a, "block", 0), kb = r4d_geti_or(a, "topk_blocks", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    if (!sc.data || !pos.data || !keep.data) return R4D_NO(RAD_E_INVAL);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || keep.rank != 2 || keep.shape[0] != sc.shape[0] ||
        keep.shape[1] != keep_words(sc.shape[1], block))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || keep.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 || block > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsCandArgs c{};
    c.sc = (const float*)sc.data;
    c.pos = (const int*)pos.data;
    c.keep = (int*)keep.data;
    c.sr = (long)sc.stride[0];
    c.scs = (long)sc.stride[1];
    c.ps = ps;
    c.kr = (long)keep.stride[0];
    c.kcs = (long)keep.stride[1];
    c.M = (int)sc.shape[0];
    c.C = (int)sc.shape[1];
    c.block = (int)block;
    c.KB = (int)(kb < INT_MAX ? kb : INT_MAX);
    c.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    const unsigned g = (unsigned)(c.M < 65535 ? c.M : 65535);
    return R4D_EMIT(stream, ds41_index_candidates_kernel, dim3(g), dim3(kDsSelThreads), 0, c);
}

int index_topk_params(const RadArgs* a, long long* topk, long long* ratio) {
    *topk = r4d_geti_or(a, "topk", 0);
    *ratio = r4d_geti_or(a, "ratio", 0);
    return (*topk < 1 || *ratio < 1) ? RAD_E_SHAPE : RAD_OK;
}

int index_topk_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    if (!sc.data || !pos.data || !sel.data || sc.rank != 2 || sc.shape[0] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || sel.rank != 2 || sel.shape[0] != sc.shape[0] ||
        sel.shape[1] != topk ||
        (keep && (block < 1 || block > INT_MAX || keep->rank != 2 || keep->shape[0] != sc.shape[0] ||
                  keep->shape[1] != keep_words(sc.shape[1], block))))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || sel.dtype != RAD_I32 ||
        (keep && keep->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] < 0 || sc.shape[1] > INT_MAX / 2 ||
        topk > INT_MAX / 2)
        return R4D_NO(RAD_E_SHAPE);
    DsTopkArgs t{};
    t.sc = (const float*)sc.data;
    t.keep = keep ? (const int*)keep->data : nullptr;
    t.pos = (const int*)pos.data;
    t.sel = (int*)sel.data;
    t.sr = (long)sc.stride[0];
    t.scs = (long)sc.stride[1];
    t.kr = keep ? (long)keep->stride[0] : 0;
    t.kcs = keep ? (long)keep->stride[1] : 0;
    t.ps = ps;
    t.selr = (long)sel.stride[0];
    t.selc = (long)sel.stride[1];
    t.M = (int)sc.shape[0];
    t.C = (int)sc.shape[1];
    t.K = (int)topk;
    t.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    t.block = keep ? (int)block : 1;
    t.bshift = -1;
    for (int sh = 0; sh < 31; ++sh)
        if ((1LL << sh) == (long long)t.block) t.bshift = sh;
    /* With keep, walk the kept blocks only (the same selection: DsKeptKey). */
    t.kept = keep && block <= 65536 ? 1 : 0;
    /* Few rows over a long row: chunks in parallel, then one merge a row. The same selection, so
     * the same bits, as the one-block-a-row kernel. */
    const int64_t need = t.M <= kDsSplitRows ? topk_split_bytes(t.M, t.C, t.K) : 0;
    if (need > 0 && a->scratch && a->scratch_bytes >= need && aligned(a->scratch, 8)) {
        DsSplitArgs sp{};
        sp.t = t;
        sp.chunk = kDsChunk;
        sp.nch = (t.C + kDsChunk - 1) / kDsChunk;
        /* the kept chunks' lists: at most ceil(blocks / nch) blocks a chunk (every block kept) */
        const long nbw = ((long)t.C + t.block - 1) / t.block;
        if ((nbw + sp.nch - 1) / sp.nch > kDsKeptMax) sp.t.kept = 0;
        sp.cand = (unsigned long long*)a->scratch;
        sp.cnt = (int*)((unsigned char*)a->scratch + (int64_t)t.M * sp.nch * t.K * 8);
        sp.used = sp.cnt + (int64_t)t.M * sp.nch;
        const int rc = R4D_EMIT(stream, ds41_index_topk_split1_kernel,
                                dim3((unsigned)sp.nch, (unsigned)t.M), dim3(kDsSelThreads), 0, sp);
        if (rc != RAD_OK) return rc;
        return R4D_EMIT(stream, ds41_index_topk_split2_kernel, dim3((unsigned)t.M),
                        dim3(kDsSelThreads), 0, sp);
    }
    const unsigned g = (unsigned)(t.M < 65535 ? t.M : 65535);
    return R4D_EMIT(stream, ds41_index_topk_kernel, dim3(g), dim3(kDsSelThreads), 0, t);
}

template <int HG, int DS>
int attn_emit(RadStream stream, const DsAttnArgs& s, unsigned gz) {
    const uint32_t lds = (uint32_t)(ds_attn_tile_off(s.nkp, HG) + kDsTileBytes);
    const dim3 grid((unsigned)DS, (unsigned)((s.H + HG - 1) / HG), gz);
    return R4D_EMIT(stream, (ds41_attn_sparse_sink_kernel<HG, DS>), grid, dim3(256), lds, s);
}

template <int HG, int NT, int DPT, bool PVD, bool ORT = false>
int attn_dec_go(RadStream stream, const DsAttnArgsT<ORT>& s) {
    constexpr int P = 4;                                /* key tiles in flight in PV */
    const int nkq = (s.nk + 31) / 32 * 32;
    const long lds = ds_dec_lds(nkq, HG, PVD ? 0 : NT, s.cnv && ds_dec_nvt(HG, NT, DPT, PVD, true)).total;
    if (lds > 65536) return RAD_E_UNSUPPORTED;
    const dim3 grid((unsigned)((s.H + HG - 1) / HG), (unsigned)s.M);
    if (s.cnv) return R4D_EMIT(stream, (ds41_attn_dec_kernel<HG, NT, DPT, P, PVD, true, ORT>), grid, dim3(NT), (uint32_t)lds, s);
    return R4D_EMIT(stream, (ds41_attn_dec_kernel<HG, NT, DPT, P, PVD, false, ORT>), grid, dim3(NT), (uint32_t)lds, s);
}

/* The decode band's kernel. The form by how many (row, head) pairs the pass has, any choice giving
 * the same bits (measured on R9700, 32 heads x 640 keys: one row 17 us, 2 rows 27, 3-4 rows 44,
 * 5-8 rows 73, 48 rows 367; the scalar kernel 129-582). On an LDS refusal (long rows), the next
 * smaller form; the one-head form takes every row this op accepts. */
/* Packed-latent forms only the A/B knob reaches (no plain twin is instantiated). */
template <int HG, int NT, int DPT, bool PVD>
int attn_dec_go_nv(RadStream stream, const DsAttnArgs& s) {
    constexpr int P = 4;
    const int nkq = (s.nk + 31) / 32 * 32;
    const long lds = ds_dec_lds(nkq, HG, PVD ? 0 : NT, ds_dec_nvt(HG, NT, DPT, PVD, true)).total;
    if (lds > 65536 || !s.cnv) return RAD_E_UNSUPPORTED;
    const dim3 grid((unsigned)((s.H + HG - 1) / HG), (unsigned)s.M);
    return R4D_EMIT(stream, (ds41_attn_dec_kernel<HG, NT, DPT, P, PVD, true>), grid, dim3(NT), (uint32_t)lds, s);
}

/* RADIANCE_DS41_NVFORM=N (8.11., an A/B knob for the latency bench; unset: the rule below): the packed latent's decode
 * band in form N -- 0-6 as attn_dec_emit's switch, 7 one head tiled (512 threads), 8 four heads direct at 512 threads
 * (two heads a PV thread), 9 two heads direct at 512 threads. Every form sums each output in the same order: the same
 * bits. A form the LDS refuses falls back to the rule. */
int ds_nvform() {
    static const int f = [] {
        const char* e = std::getenv("RADIANCE_DS41_NVFORM");
        return e && *e ? std::atoi(e) : -1;
    }();
    return f;
}

/* ORT (attn_sparse_sink_rope_roundtrip): the same forms with the fused epilogue; the knob's packed-only forms
 * 7-9 are not instantiated with it (the rule serves those requests). */
template <bool ORT = false>
int attn_dec_emit(RadStream stream, const DsAttnArgsT<ORT>& s) {
    if (s.M > 65535 || (s.H + 0L) > 65535L * 8) return RAD_E_UNSUPPORTED;
    const long hr = (long)s.M * s.H;
    if (s.cnv && ds_nvform() >= 0) {
        int rc = RAD_E_UNSUPPORTED;
        switch (ds_nvform()) {
            case 0: rc = attn_dec_go<1, 1024, 2, true, ORT>(stream, s); break;
            case 1: rc = attn_dec_go<2, 1024, 2, true, ORT>(stream, s); break;
            case 2: rc = attn_dec_go<4, 1024, 2, true, ORT>(stream, s); break;
            case 3: rc = attn_dec_go<8, 1024, 2, true, ORT>(stream, s); break;
            case 4: rc = attn_dec_go<8, 512, 2, false, ORT>(stream, s); break;
            case 5: rc = attn_dec_go<4, 512, 2, false, ORT>(stream, s); break;
            case 6: rc = attn_dec_go<2, 512, 2, false, ORT>(stream, s); break;
            case 7: if constexpr (!ORT) rc = attn_dec_go_nv<1, 512, 2, false>(stream, s); break;
            case 8: if constexpr (!ORT) rc = attn_dec_go_nv<4, 512, 2, true>(stream, s); break;
            case 9: if constexpr (!ORT) rc = attn_dec_go_nv<2, 512, 2, true>(stream, s); break;
            default: break;
        }
        if (rc != RAD_E_UNSUPPORTED) return rc;
    }
    int c = hr <= 32 ? 0 : (hr <= 64 ? 1 : (hr <= 128 ? 2 : (hr <= 256 ? 3 : 4)));
    static const int next[7] = { -1, 0, 1, 2, 5, 6, 1 };
    while (c >= 0) {
        int rc = RAD_E_UNSUPPORTED;
        switch (c) {
            case 0: rc = attn_dec_go<1, 1024, 2, true, ORT>(stream, s); break;
            case 1: rc = attn_dec_go<2, 1024, 2, true, ORT>(stream, s); break;
            case 2: rc = attn_dec_go<4, 1024, 2, true, ORT>(stream, s); break;
            case 3: rc = attn_dec_go<8, 1024, 2, true, ORT>(stream, s); break;
            case 4: rc = attn_dec_go<8, 512, 2, false, ORT>(stream, s); break;
            case 5: rc = attn_dec_go<4, 512, 2, false, ORT>(stream, s); break;
            case 6: rc = attn_dec_go<2, 512, 2, false, ORT>(stream, s); break;
            default: break;
        }
        if (rc != RAD_E_UNSUPPORTED) return rc;
        c = next[c];
    }
    return RAD_E_UNSUPPORTED;
}

/* attn_sparse_sink's checks and arguments (the launch below; attn_sparse_sink_rope_roundtrip's too). */
int attn_setup(const RadArgs* a, DsAttnArgs* res) {
    /* Operands 4-6 are optional (null without a compressed source), as in libref. */
    if (!a || a->n_t < 11 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2], &wused = a->t[3];
    const RadTensor* cc = a->t[4].data ? &a->t[4] : nullptr;
    const RadTensor* cbt = a->t[5].data ? &a->t[5] : nullptr;
    const RadTensor* sel = a->t[6].data ? &a->t[6] : nullptr;
    const RadTensor &pos = a->t[7], &cu = a->t[8], &sink = a->t[9], &out = a->t[10];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0);
    const long long win = r4d_geti_or(a, "window", 0), ratio = r4d_geti_or(a, "ratio", 0);
    const long long topk = r4d_geti_or(a, "topk", 0);
    const float scale = (float)r4d_getf(a, "scale", 0.0);
    const bool comp = ratio > 0;
    /* latent_format nvfp4: the compressed cache holds each entry packed (ds_nv4_bytes(D) bytes); the
     * window cache stays bf16, the window a multiple of 32 (the kernels' key blocks start packed at it). */
    const char* lf = r4d_gets(a, "latent_format");
    const bool cnv = comp && lf && !std::strcmp(lf, "nvfp4");
    if (lf && !cnv && std::strcmp(lf, "bf16") && !(std::strcmp(lf, "nvfp4") == 0 && !comp))
        return R4D_NO(RAD_E_UNSUPPORTED);
    const long long cslots = cnv ? ds_nv4_bytes((int)D) / 2 : D;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio)))
        return R4D_NO(RAD_E_SHAPE);
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&sink) != H ||
        sink.dtype != RAD_F32 || r4d_numel(&cu) < 2 || r4d_numel(&wused) < r4d_numel(&cu) - 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cbs = comp ? cc->shape[2] : 0;
    if (cnv && win % 32) return R4D_NO(RAD_E_UNSUPPORTED);
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != cslots || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || wc.dtype != RAD_BF16 || out.dtype != RAD_BF16 || wbt.dtype != RAD_I32 ||
        wused.dtype != RAD_I32 || pos.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (comp && (cc->dtype != RAD_BF16 || cbt->dtype != RAD_I32 || sel->dtype != RAD_I32)))
        return R4D_NO(RAD_E_DTYPE);
    /* This row: the 512-wide latent, read 16 bytes at a time; at most 1024 + 2048 keys a row. */
    if (D != kDsAttnD || win > 1024 || topk > 2048 || ratio > 64) return R4D_NO(RAD_E_UNSUPPORTED);
    const long S = (long)(r4d_numel(&cu) - 1);
    const long ps = lin_stride(pos), cs = lin_stride(cu), wus = lin_stride(wused), sks = lin_stride(sink);
    if (ps < 0 || cs < 0 || wus < 0 || sks < 0 || q.stride[1] != 1 || out.stride[1] != 1 ||
        (q.stride[0] & 1) || (out.stride[0] & 1) || !aligned(q.data, 4) || !aligned(out.data, 4) ||
        !aligned(wc.data, 16) || (comp && !aligned(cc->data, 16)) || wbt.shape[0] < S ||
        q.shape[0] > INT_MAX || S > INT_MAX || wbt.shape[1] > INT_MAX ||
        (comp && cbt->shape[1] > INT_MAX) || wc.shape[2] < 1 || wc.shape[2] > INT_MAX || cbs > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsAttnArgs s{};
    s.q = (const uint16_t*)q.data;
    s.wc = (const uint16_t*)wc.data;
    s.wbt = (const int*)wbt.data;
    s.wused = (const int*)wused.data;
    s.cc = comp ? (const uint16_t*)cc->data : nullptr;
    s.cbt = comp ? (const int*)cbt->data : nullptr;
    s.sel = comp ? (const int*)sel->data : nullptr;
    s.pos = (const int*)pos.data;
    s.cu = (const int*)cu.data;
    s.sink = (const float*)sink.data;
    s.out = (uint16_t*)out.data;
    s.qr = (long)q.stride[0];
    s.outr = (long)out.stride[0];
    s.wbtr = (long)wbt.stride[0];
    s.wbtc = (long)wbt.stride[1];
    s.nwblk = (long)wc.shape[0];
    s.cbtr = comp ? (long)cbt->stride[0] : 0;
    s.cbtc = comp ? (long)cbt->stride[1] : 0;
    s.ncblk = comp ? (long)cc->shape[0] : 0;
    s.selr = comp ? (long)sel->stride[0] : 0;
    s.selc = comp ? (long)sel->stride[1] : 0;
    s.wus = wus;
    s.ps = ps;
    s.cs = cs;
    s.sks = sks;
    s.nwbt = (int)wbt.shape[1];
    s.bs = (int)wc.shape[2];
    s.ncbt = comp ? (int)cbt->shape[1] : 0;
    s.cbs = comp ? (int)cbs : 1;
    s.ratio = comp ? (int)ratio : 1;
    s.cnv = cnv ? 1 : 0;
    s.cslot = comp ? (int)cc->shape[3] : 1;
    s.S = (int)S;
    s.M = (int)q.shape[0];
    s.H = (int)H;
    s.win = (int)win;
    s.nk = (int)(win + (comp ? topk : 0));
    s.nkp = (s.nk + 3) / 4 * 4;
    s.scale = scale;
    *res = s;
    return RAD_OK;
}

int attn_body(const RadArgs* a, RadStream stream) {
    DsAttnArgs s{};
    {
        const int rc = attn_setup(a, &s);
        if (rc != RAD_OK) return rc;
    }
    const long long H = s.H;
    const float scale = s.scale;
    const RadTensor &q = a->t[0], &out = a->t[10];
    /* Prefill heights on WMMA (16-byte q and out rows); the decode band on the scalar kernels
     * below (libref's bits). */
    {
        const int nkt = (s.nk + 15) / 16;
        const long lds = ds_wm_sbuf_off(nkt) + kWmSbufBytes + (s.cnv ? kWmNvBytes : 0);
        if (s.M > kWmRows && s.qr % 8 == 0 && s.outr % 8 == 0 && aligned(q.data, 16) &&
            aligned(out.data, 16) && lds <= 65536 && (H + 15) / 16 <= 65535) {
            const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
            const dim3 grid((unsigned)((H + 15) / 16), gy);
            const float sl2 = (float)((double)scale * 1.4426950408889634);
            if (s.cnv) return R4D_EMIT(stream, (ds41_attn_sparse_sink_wm_kernel<true, true>), grid, dim3(128), (uint32_t)lds, s, sl2);
            return R4D_EMIT(stream, (ds41_attn_sparse_sink_wm_kernel<true, false>), grid, dim3(128), (uint32_t)lds, s, sl2);
        }
    }
    /* Head group and d split by how many workgroups the rows give. Every choice sums each output
     * in the same order, so the bits do not depend on it. */
    const long rows = s.M;
    if (rows <= kWmRows) {
        const int rc = attn_dec_emit<false>(stream, s);
        if (rc != RAD_E_UNSUPPORTED) return rc;
    }
    int HG = 8, DS = 1;
    if (rows * H < 2048) HG = rows * H >= 512 ? 4 : 2;
    if (HG == 2) DS = 2;
    while (HG > 1 && ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) HG /= 2;
    if (ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) return R4D_NO(RAD_E_UNSUPPORTED);
    const unsigned gz = (unsigned)(s.M < 65535 ? s.M : 65535);
    if (DS == 2) {
        if (HG == 2) return attn_emit<2, 2>(stream, s, gz);
        return attn_emit<1, 2>(stream, s, gz);
    }
    switch (HG) {
        case 8: return attn_emit<8, 1>(stream, s, gz);
        case 4: return attn_emit<4, 1>(stream, s, gz);
        case 2: return attn_emit<2, 1>(stream, s, gz);
        default: return attn_emit<1, 1>(stream, s, gz);
    }
}

/* attn_sparse_sink_rope_roundtrip (lever K1 fold d): attn_sparse_sink over operands 0-10, then
 * rope_rows_roundtrip over its `out` with cos_sin (operand 11) at the attention's positions -- the
 * attention output's inverse rope and fp8 round trip (DeepSeek-V4.1's o, before wo_a). A pass of at most
 * kWmRows rows whose geometry the fused epilogue takes (gptj, `off` even so a rotated pair is one thread's
 * column pair, blocks of 32 = a half-wave of column pairs) runs the decode band's kernel with ds_ort_pair in
 * its epilogue: one launch, the bytes of the two. Any other pass (prefill heights, another geometry) issues
 * attn_sparse_sink's launch and then rope_rows_roundtrip's, the two ops themselves. Parameters: the
 * attention's, then rope_rows' rot / off / mode / inverse (n_head = heads, head_dim = dim) and
 * quant_roundtrip's block / format, its scale as `block_scale` (`scale` is the attention's). */
int attn_ort_body(const RadArgs* a, RadStream stream) {
    namespace q = r4d_ds41q;
    if (!a || a->n_t < 12 || !a->t) return RAD_E_INVAL;
    const RadTensor &pos = a->t[7], &out = a->t[10], &cs = a->t[11];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0);
    const long long rot = r4d_geti_or(a, "rot", 0), off = r4d_geti_or(a, "off", -1);
    const long long inv = r4d_geti_or(a, "inverse", -1), block = r4d_geti_or(a, "block", 0);
    const char* mode = r4d_gets(a, "mode");
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "block_scale");
    if (!mode || !fs || !ss) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    int fmt, sc;   /* quant_rt_body's names */
    if (!std::strcmp(fs, "fp8_e4m3")) fmt = q::kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) fmt = q::kFp4;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (!std::strcmp(ss, "ue8m0")) sc = q::kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && fmt == q::kFp4) sc = q::kE4m3;
    else if (!std::strcmp(ss, "f32") && fmt == q::kFp8) sc = q::kScaleF32;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (H < 1 || D < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > D || (inv != 0 && inv != 1) ||
        block < 1 || (H * D) % block)
        return R4D_NO(RAD_E_SHAPE);
    /* rope_rows_body's angle plane */
    if (!cs.data || !pos.data || !out.data || cs.dtype != RAD_F32 || cs.rank != 2 || cs.shape[1] != rot ||
        out.rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    if (out.shape[0] <= kWmRows && gptj && !(off & 1) && block == 32) {
        DsAttnArgs s{};
        const int rc = attn_setup(a, &s);
        if (rc != RAD_OK) return rc;
        DsAttnArgsOrt so{};
        static_cast<DsAttnArgs&>(so) = s;
        so.o.cs = (const float*)cs.data;
        so.o.cp = (long)cs.stride[0];
        so.o.cst = (long)cs.stride[1];
        so.o.tab_rows = (long)cs.shape[0];
        so.o.off = (int)off;
        so.o.half = (int)(rot / 2);
        so.o.fmt = fmt;
        so.o.sc = sc;
        so.o.sign = inv ? -1.0f : 1.0f;
        const int rc2 = attn_dec_emit<true>(stream, so);
        if (rc2 != RAD_E_UNSUPPORTED) return rc2;
    }
    /* the two launches: attn_sparse_sink, then rope_rows_roundtrip over out at its own parameters */
    {
        const int rc = attn_body(a, stream);
        if (rc != RAD_OK) return rc;
    }
    auto pint = [](const char* k, long long v) { RadParam r{}; r.key = k; r.kind = RAD_P_INT; r.ival = v; return r; };
    auto pstr = [](const char* k, const char* v) { RadParam r{}; r.key = k; r.kind = RAD_P_STR; r.sval = v; return r; };
    RadParam pp[] = { pint("M", out.shape[0]), pint("n_head", H), pint("head_dim", D), pint("rot", rot),
                      pint("off", off), pstr("mode", mode), pint("inverse", inv), pint("block", block),
                      pstr("format", fs), pstr("scale", ss), pstr("dtype", "bf16") };
    RadTensor tt[3] = { out, cs, pos };
    RadArgs ra = *a;
    ra.t = tt;
    ra.n_t = 3;
    ra.p = pp;
    ra.n_p = (int)(sizeof(pp) / sizeof(pp[0]));
    return r4d_rad_rope_rows_roundtrip(&ra, stream);
}

}  // namespace

/* entry_store's format mxfp4 (r4d_ds41_elem.hip's entry_store_body hands it over): x [W, n] bf16, n a
 * multiple of 32 up to 1024, into block page[w] of a cache [blocks, heads, block_size, slot] whose
 * heads x block_size x slot bf16 slots hold ds_mx4_bytes(n) bytes. */
extern "C" int r4d_ds41_entry_store_mx4(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    if (n < 32 || n > 1024 || n % 32 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        r4d_numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] * 2 != ds_mx4_bytes((int)n) || x.stride[1] != 1 ||
        !aligned(cache.data, 16))
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || page.dtype != RAD_I32 || cache.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long W = x.shape[0], pgs = lin_stride(page);
    if (pgs < 0) return R4D_NO(RAD_E_SHAPE);
    const unsigned g = (unsigned)(W < 65535 ? W : 65535);
    return R4D_EMIT(stream, ds41_entry_store_mx4_kernel, dim3(g), dim3((unsigned)n), 0u, (const uint16_t*)x.data,
                    (long)x.stride[0], (const int32_t*)page.data, pgs, (unsigned char*)cache.data,
                    (long)cache.shape[0], (int)n, W);
}

/* entry_store's format nvfp4 (entry_store_body hands it over): x [W, n] bf16, n a multiple of 16 up to
 * 1024, into block page[w] of a cache [blocks, heads, block_size, slot] whose bf16 slots hold
 * ds_nv4_bytes(n) bytes. */
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    if (n < 32 || n > 1024 || n % 32 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        r4d_numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] * 2 != ds_nv4_bytes((int)n) || x.stride[1] != 1 ||
        !aligned(cache.data, 16))
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || page.dtype != RAD_I32 || cache.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long W = x.shape[0], pgs = lin_stride(page);
    if (pgs < 0) return R4D_NO(RAD_E_SHAPE);
    const unsigned g = (unsigned)(W < 65535 ? W : 65535);
    return R4D_EMIT(stream, ds41_entry_store_nv4_kernel, dim3(g), dim3((unsigned)n), 0u, (const uint16_t*)x.data,
                    (long)x.stride[0], (const int32_t*)page.data, pgs, (unsigned char*)cache.data,
                    (long)cache.shape[0], (int)n, W);
}

extern "C" int r4d_rad_ds41_index_score(const RadArgs* a, RadStream stream) {
    return index_score_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_candidates(const RadArgs* a, RadStream stream) {
    return index_candidates_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_topk(const RadArgs* a, RadStream stream) {
    return index_topk_body(a, stream);
}
/* Called at declare with the band's bound and no tensors: the split's partials for up to
 * kDsSplitRows rows of `width` positions; 0 (no split) when a row is short. */
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs* a) {
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return 0;
    long long M = r4d_geti_or(a, "M", kDsSplitRows);
    if (M < 1 || M > kDsSplitRows) M = kDsSplitRows;
    return topk_split_bytes((long)M, (long)r4d_geti_or(a, "width", 0), (long)topk);
}
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs* a, RadStream stream) {
    return attn_body(a, stream);
}
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs* a, RadStream stream) {
    return attn_ort_body(a, stream);
}
