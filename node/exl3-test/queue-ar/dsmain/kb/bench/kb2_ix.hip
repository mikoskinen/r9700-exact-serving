// kb2_ix.hip (12.11.2026, ds-main-port kbench builder) -- lever IX (audit 7): DeepSeek-V4.1's prefill index_score on
// one card. The int7 source (ds41-int7 0391a6e libr4d/r4d_ds41_sparse.hip, copied verbatim to kb/src) is included
// whole, so STK is the stock WMMA kernel (ds41_index_score_wm_kernel) through the stock launch (index_score_body);
// LIB is the image's own libr4d.so through r4d_rad_ds41_index_score. Variants are new kernels in this file.
//
// Shapes (arch/deepseek41/deepseek41.cpp 2053-2077, 1393-1399; r4d_ds41_sparse.hip 2801-2899): a 4096-row prefill
// step scored in runs of irows = max(256, 256 MiB / (4 cmax)) = 256 rows (cmax = 524288 / 1 + 1 at MML 524288), the
// score plane [256, width] f32 at row pitch cmax, width = 524288 / ratio + 1; 32 heads x 128 (WMMA form, HT 2), q bf16
// [T, 4096] (quant_roundtrip fp4_e2m1 / ue8m0 / 32 values), w bf16 [T, 32], wscale 1/64; the keys MXFP4 packed (80 B
// an entry: production RADIANCE_DS41_IDXKEY=mxfp4), one entry a cache block (block size = ratio); keep (layers 24-36:
// candidate blocks of 8, 2048 blocks + the newest pinned, random here).
#include "r4d_ds41_sparse.hip"
#include "kb2_common.h"

#include <cmath>
#include <functional>
#include <map>

const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

namespace {

constexpr int kVPF = 1;                                       /* variant flag: prefetch the next tile's key words */
constexpr int kA_NOWMMA = 1, kA_NOEPI = 2, kA_NOUNPACK = 4, kA_NOSTORE = 8;   /* ablations: bytes differ by design */

/* ---- index_score_body's argument setup, copied (r4d_ds41_sparse.hip 2801-2885), for the variants' own launches */
bool mk_score_args(const RadArgs* a, DsScoreArgs* out) {
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), Dm = r4d_geti_or(a, "dim", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long part = r4d_geti_or(a, "row_part", -1);
    const long long block = r4d_geti_or(a, "block", 0);
    const float wscale = (float)r4d_getf(a, "wscale", 0.0);
    const RadTensor* keep = a->n_t > 7 && a->t[7].data ? &a->t[7] : nullptr;
    const char* kf = r4d_gets(a, "key_format");
    const bool kmx = kf && !std::strcmp(kf, "mxfp4");
    if (H != 32 || Dm != 128) return false;
    const long ps = lin_stride(pos), cs = lin_stride(cu);
    const long row0 = part > 0 ? (long)part * (long)sc.shape[0] : 0;
    const long Mv = row0 < (long)q.shape[0]
                        ? ((long)q.shape[0] - row0 < (long)sc.shape[0] ? (long)q.shape[0] - row0 : (long)sc.shape[0])
                        : 0;
    const long r0 = Mv > 0 ? row0 : 0;
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
    *out = s;
    return true;
}

/* A lane's key pointer for entry j of sequence s (the stock kernel's lookup), null when unmapped. */
__device__ __forceinline__ const uint16_t* ix_kp(const DsScoreArgs& a, int s, long j) {
    const long col = (j * a.ratio) / a.cbs;
    if (col < a.nbt) {
        const long blk = a.bt[(long)s * a.btr + col * a.btc];
        if (blk >= 0 && blk < a.nblk) return a.kc + (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
    }
    return nullptr;
}
/* A lane's raw key words (its half) and the unpack, exactly the stock kernel's. */
/* A lane's raw MXFP4 key words (its half: 32 code bytes, 2 scale bytes) and the stock unpack. The variants take the
 * production key format (RADIANCE_DS41_IDXKEY=mxfp4) at compile time; a bf16-key call is refused (*flag = -vid). */
struct KRaw {
    uint4 c0, c1;
    uint32_t e0, e1;
};
__device__ __forceinline__ void ix_kload(const uint16_t* kp, int hf, KRaw& r) {
    const unsigned char* pb = (const unsigned char*)(kp ? kp : kDsZeroKey);
    r.c0 = *(const uint4*)(pb + 32 * hf);
    r.c1 = *(const uint4*)(pb + 32 * hf + 16);
    r.e0 = pb[64 + 2 * hf];
    r.e1 = pb[64 + 2 * hf + 1];
}
template <int ABL>
__device__ __forceinline__ void ix_unpack(const KRaw& r, uint4 (&kb)[8]) {
    const uint32_t cw[8] = { r.c0.x, r.c0.y, r.c0.z, r.c0.w, r.c1.x, r.c1.y, r.c1.z, r.c1.w };
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr ((ABL & kA_NOUNPACK) != 0)
            kb[i] = make_uint4(cw[i], cw[i] ^ r.e0, cw[i] ^ r.e1, cw[i]);
        else
            kb[i] = ds_mx4_unpack8(cw[i], i < 4 ? r.e0 : r.e1);
    }
}

/* ---- IXV: the stock kernel's per-output arithmetic (same q / key operands, the same 8-step WMMA chain a head tile,
 * the same epilogue in head order), with EB entries a workgroup, RPW rows a wave sharing each unpacked key tile, and
 * (PF) the next tile's key words loaded before this tile's math. A non-live entry of a row scores -inf as in the stock
 * kernel (the WMMA column of an entry depends on its own key only, so a key another row needs does not move a row's
 * live scores). Rows of a wave must be of one sequence (else *flag = -vid: refused). */
template <int HT, bool KEEP, int EB, int RPW, int FL, int ABL>
__global__ __launch_bounds__(256) void ixv_kernel(DsScoreArgs a, int* flag, int vid) {
    constexpr int DM = 128;
    constexpr bool PF = (FL & kVPF) != 0;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *flag = a.kmx ? vid : -vid;
    if (!a.kmx) return;
    const long mb = ((long)blockIdx.y * 8 + wave) * RPW;
    if (mb >= a.M) return;
    const long jb = (long)blockIdx.x * EB;
    const long jend = jb + EB < (long)a.C ? jb + EB : (long)a.C;
    long reach[RPW];
    bool done[RPW];
    float* srow[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const long m = mb + r;
        const bool rv = m < a.M;
        reach[r] = rv && m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        srow[r] = a.sc + (rv ? m : mb) * a.sr;
        done[r] = !rv;
        if (rv && jb >= reach[r]) {
            for (long j = jb + lane; j < jend; j += 32) srow[r][j * a.scs] = -kDsInf;
            done[r] = true;
        }
    }
    auto all_done = [&]() {
        bool al = true;
#pragma unroll
        for (int r = 0; r < RPW; ++r) al = al && done[r];
        return al;
    };
    if (all_done()) return;
    [[maybe_unused]] unsigned kword[RPW];
    [[maybe_unused]] long kw0[RPW];
    if constexpr (KEEP) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            kword[r] = 0u;
            kw0[r] = 0;
            if (done[r]) continue;
            const int* krow = a.keep + (mb + r) * a.kr;
            const long jl = (jend < reach[r] ? jend : reach[r]) - 1;
            kw0[r] = ds_keep_block(a, jb) >> 5;
            const long nw = (ds_keep_block(a, jl) >> 5) - kw0[r] + 1;
            kword[r] = lane < nw ? (unsigned)krow[(kw0[r] + lane) * a.kcs] : 0u;
            if (__builtin_amdgcn_ballot_w32(kword[r] != 0u) == 0u) {
                for (long j = jb + lane; j < jend; j += 32) srow[r][j * a.scs] = -kDsInf;
                done[r] = true;
            }
        }
        if (all_done()) return;
    }
    int s = 0;
    {
        bool first = true;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (mb + r >= a.M) continue;
            const int sr = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + mb + r);
            if (first) s = sr;
            else if (sr != s && lane == 0) *flag = -vid;   /* a row pair across sequences: refused */
            first = false;
        }
    }
    uint4 qf[RPW][HT][8];
    float wh[RPW][HT][8];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const long m = mb + r < a.M ? mb + r : mb;
#pragma unroll
        for (int t = 0; t < HT; ++t) {
            const int h = t * 16 + (lane & 15);
            const uint16_t* qp = a.q + m * a.qr + (long)(h < a.H ? h : 0) * DM + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qf[r][t][i] = *(const uint4*)(qp + 8 * i);
                if (h >= a.H) qf[r][t][i] = make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int hh = t * 16 + 8 * hf + e;
                wh[r][t][e] = hh < a.H ? ds_round_bf16(ds_bf(a.w[m * a.wr + (long)hh * a.wc]) * a.wscale) : 0.0f;
            }
        }
    }
    /* liveness of entry j for row r (stock: j < jend, j < reach, and with keep its bit) */
    auto live_of = [&](int r, long j) -> bool {
        bool lv = !done[r] && j < jend && j < reach[r];
        if constexpr (KEEP) {
            long wi = lv ? (ds_keep_block(a, j) >> 5) - kw0[r] : 0;
            wi = wi < 0 ? 0 : (wi > 31 ? 31 : wi);
            const unsigned wd = (unsigned)__builtin_amdgcn_ds_bpermute((int)wi << 2, (int)kword[r]);
            lv = lv && ((wd >> (ds_keep_block(a, j) & 31)) & 1u) != 0u;
        }
        return lv;
    };
    /* the key pointer of tile j0 for this lane: looked up when any row of the wave needs the entry */
    auto tile_kp = [&](long j0) -> const uint16_t* {
        const long j = j0 + (lane & 15);
        bool any = false;
#pragma unroll
        for (int r = 0; r < RPW; ++r) any = any || live_of(r, j);
        return any ? ix_kp(a, s, j) : nullptr;
    };
    auto compute = [&](const KRaw& rr, const uint16_t* kp, long j, const bool (&lv)[RPW]) {
        uint4 kb[8];
        ix_unpack<ABL>(rr, kb);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (done[r]) continue;                   /* wave-uniform */
            ds_v8f d[HT];
#pragma unroll
            for (int t = 0; t < HT; ++t) {
                d[t] = (ds_v8f)(0.0f);
                if constexpr ((ABL & kA_NOWMMA) != 0) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) d[t][i] = __uint_as_float(kb[i].x ^ qf[r][t][i].y) + d[t][i];
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) d[t] = ds_wmma(qf[r][t][i], kb[i], d[t]);
                }
            }
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < HT; ++t) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    if (t + half > 0) acc = ds_xor16(acc);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        if constexpr ((ABL & kA_NOEPI) != 0) {
                            acc = acc + d[t][e] * wh[r][t][e];
                        } else {
                            const float dot = ds_round_bf16(d[t][e]);
                            acc = acc + ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh[r][t][e]);
                        }
                    }
                }
            }
            const float v = (lv[r] && kp) ? ds_round_bf16(acc) : -kDsInf;
            if constexpr ((ABL & kA_NOSTORE) != 0) {
                if (hf == 1 && j < jend && v == 1234.5f) srow[r][j * a.scs] = v;
            } else {
                if (hf == 1 && j < jend) srow[r][j * a.scs] = v;
            }
        }
    };
    [[maybe_unused]] KRaw rawc;
    [[maybe_unused]] const uint16_t* kpc = nullptr;
    if constexpr (PF) {
        kpc = tile_kp(jb);
        ix_kload(kpc, hf, rawc);
    }
    for (long j0 = jb; j0 < jend; j0 += 16) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (!done[r] && j0 >= reach[r]) {        /* wave-uniform: the rest is past the row */
                for (long j = j0 + lane; j < jend; j += 32) srow[r][j * a.scs] = -kDsInf;
                done[r] = true;
            }
        }
        if (all_done()) break;
        const long j = j0 + (lane & 15);
        bool lv[RPW];
        bool any = false;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            lv[r] = live_of(r, j);
            any = any || lv[r];
        }
        if constexpr (KEEP) {
            if (__builtin_amdgcn_ballot_w32(any) == 0u) {
#pragma unroll
                for (int r = 0; r < RPW; ++r)
                    if (!done[r] && hf == 1 && j < jend) srow[r][j * a.scs] = -kDsInf;
                if constexpr (PF) {
                    if (j0 + 16 < jend) {
                        kpc = tile_kp(j0 + 16);
                        ix_kload(kpc, hf, rawc);
                    }
                }
                continue;
            }
        }
        if constexpr (PF) {
            KRaw rn;
            const uint16_t* kpn = nullptr;
            if (j0 + 16 < jend) {
                kpn = tile_kp(j0 + 16);
                ix_kload(kpn, hf, rn);
            }
            compute(rawc, kpc, j, lv);
            rawc = rn;
            kpc = kpn;
        } else {
            const uint16_t* kp = any ? ix_kp(a, s, j) : nullptr;
            KRaw rr;
            ix_kload(kp, hf, rr);
            compute(rr, kp, j, lv);
        }
    }
}

/* ---- IXL: the workgroup's EB entries unpacked once into LDS (a padded row of 17 uint4 an entry: lanes on 16
 * consecutive entries hit distinct banks), then each wave walks them for its RPW rows with the stock arithmetic.
 * KEEP = false. All rows of the workgroup must be of one sequence (else *flag = -vid). */
template <int HT, int EB, int RPW>
__global__ __launch_bounds__(256) void ixl_kernel(DsScoreArgs a, int* flag, int vid) {
    constexpr int DM = 128, KP = 17;
    __shared__ uint4 kl[EB * KP];
    __shared__ unsigned char kmap[EB];
    __shared__ long s_reach[8 * RPW];
    __shared__ int s_seq[8 * RPW];
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *flag = a.kmx ? vid : -vid;
    if (!a.kmx) return;                               /* uniform: the whole launch */
    const long g0 = (long)blockIdx.y * 8 * RPW;
    if (threadIdx.x < 8 * RPW) {
        const long m = g0 + threadIdx.x;
        s_reach[threadIdx.x] = m < a.M && m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        s_seq[threadIdx.x] = m < a.M ? ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m) : -1;
    }
    __syncthreads();
    long rwg = 0;
    int s = -1;
    for (int i = 0; i < 8 * RPW; ++i) {
        rwg = s_reach[i] > rwg ? s_reach[i] : rwg;
        if (s_seq[i] >= 0) {
            if (s < 0) s = s_seq[i];
            else if (s_seq[i] != s && threadIdx.x == 0) *flag = -vid;
        }
    }
    if (s < 0) s = 0;
    const long jb = (long)blockIdx.x * EB;
    const long jend = jb + EB < (long)a.C ? jb + EB : (long)a.C;
    const long mb = g0 + (long)wave * RPW;
    if (jb >= rwg) {                                  /* workgroup-uniform: every row past this span */
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const long m = mb + r;
            if (m >= a.M) continue;
            float* srow = a.sc + m * a.sr;
            for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
        }
        return;
    }
    /* stage: thread pairs a key (half hf2 each), the stock unpack */
    for (int e = (int)threadIdx.x >> 1; e < EB; e += 128) {
        const int h2 = threadIdx.x & 1;
        const long j = jb + e;
        const bool lv = j < jend && j < rwg;
        const uint16_t* kp = lv ? ix_kp(a, s, j) : nullptr;
        KRaw r;
        ix_kload(kp, h2, r);
        uint4 kb[8];
        ix_unpack<0>(r, kb);
#pragma unroll
        for (int i = 0; i < 8; ++i) kl[e * KP + 8 * h2 + i] = kb[i];
        if (h2 == 0) kmap[e] = kp ? 1 : 0;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const long m = mb + r;
        if (m >= a.M) continue;
        float* srow = a.sc + m * a.sr;
        const long reach = s_reach[wave * RPW + r];
        if (jb >= reach) {
            for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            continue;
        }
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
            if (j0 >= reach) {
                for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
                break;
            }
            const int el = (int)(j0 - jb) + (lane & 15);
            const long j = j0 + (lane & 15);
            const bool live = j < jend && j < reach && kmap[el];
            uint4 kb[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) kb[i] = kl[el * KP + 8 * hf + i];
            ds_v8f d[HT];
#pragma unroll
            for (int t = 0; t < HT; ++t) {
                d[t] = (ds_v8f)(0.0f);
#pragma unroll
                for (int i = 0; i < 8; ++i) d[t] = ds_wmma(qf[t][i], kb[i], d[t]);
            }
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
            if (hf == 1 && j < jend) srow[j * a.scs] = live ? ds_round_bf16(acc) : -kDsInf;
        }
    }
}

/* ---- the fp8 forms. Inputs prepared by ix8_prep_kernel from the bf16 values (q per call; keys as if the cache
 * stored them so): REBASED (F8R): a 128-vector's values times 2^-E (E = its largest binade - 8) in e4m3 (exact when
 * the vector spans <= 16 binades; else counted `bad`), the dot on v_wmma_f32_16x16x16_fp8_fp8 with the stock K
 * slicing, then ldexp(D, Eq + Ek) -- the stock bits iff the fp8 WMMA adds its exact products as the bf16 one does
 * (the byte check on stress data decides). BLOCK (F8B, the audit's literal block dot): each 32-block's codes
 * (v / 2^(its binade - 2), e2m1, exact in e4m3) dotted on fp8 WMMA, K slices within a block, the four block
 * partials scaled by 2^(eq_b + ek_b) and added in block order: another summation order, NOT exact by construction. */
typedef int ds_v2i __attribute__((ext_vector_type(2)));
__device__ __forceinline__ ds_v8f ds_wmma8(uint2 a, uint2 b, ds_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(ds_v2i, a),
                                                                 __builtin_bit_cast(ds_v2i, b), c);
}
__device__ __forceinline__ int ix_binade(float v) {           /* floor(log2 |v|) of a normal f32, -1000 at 0 */
    const uint32_t u = __float_as_uint(v) & 0x7fffffffu;
    return u == 0u ? -1000 : (int)(u >> 23) - 127;
}
__device__ __forceinline__ uint32_t ix_e4m3(float v, unsigned* bad) {
    const uint32_t b = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false) & 0xffu;
    if (ds_e4m3_f32(b) != v) atomicAdd(bad, 1u);
    return b;
}
/* one thread a 128-vector: bf16 src (row pitch sp, 128 a vector), out8 [vec][128], rebased exponent ER[vec] and the
 * block exponents EB[vec][4] (the form's own: F8R uses ER and out8 rebased, F8B uses EB and out8b codes) */
__global__ void ix8_prep_kernel(const uint16_t* src, long nvec, unsigned char* out8, int* er,
                                unsigned char* out8b, int* eb, unsigned* bad) {
    const long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    const uint16_t* p = src + v * 128;
    int emax = -1000, bm[4] = { -1000, -1000, -1000, -1000 };
    for (int i = 0; i < 128; ++i) {
        const int e = ix_binade(ds_bf(p[i]));
        emax = e > emax ? e : emax;
        bm[i / 32] = e > bm[i / 32] ? e : bm[i / 32];
    }
    const int E = emax > -1000 ? emax - 8 : 0;
    er[v] = E;
    for (int b = 0; b < 4; ++b) eb[v * 4 + b] = bm[b] > -1000 ? bm[b] - 2 : 0;
    for (int i = 0; i < 128; i += 4) {
        uint32_t w = 0, wb = 0;
        for (int k = 0; k < 4; ++k) {
            const float x = ds_bf(p[i + k]);
            w |= ix_e4m3(ldexpf(x, -E), bad) << (8 * k);
            wb |= ix_e4m3(ldexpf(x, -eb[v * 4 + (i + k) / 32]), bad + 1) << (8 * k);
        }
        *(uint32_t*)(out8 + v * 128 + i) = w;
        *(uint32_t*)(out8b + v * 128 + i) = wb;
    }
}
/* the key cache unpacked to bf16 (one thread an entry), for the prep and the bf16-key reference */
__global__ void ix_unpack_cache_kernel(const unsigned char* mx, long n, uint16_t* out) {
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const unsigned char* p = mx + e * kDsMx4Bytes;
    for (int i = 0; i < 128; ++i) {
        const uint32_t c = (p[i / 2] >> (4 * (i & 1))) & 15u;
        out[e * 128 + i] = (uint16_t)ds_mx4_bf16(c, p[64 + i / 32]);
    }
}

struct Ix8Args {
    const unsigned char* q8;   /* [T][32][128] for the step's rows (rebased or block codes) */
    const int* qe;             /* F8R: [T][32]; F8B: [T][32][4] */
    const unsigned char* k8;   /* [nblk][128] */
    const int* ke;             /* F8R: [nblk]; F8B: [nblk][4] */
};

template <int HT, bool BLOCK>
__global__ __launch_bounds__(256) void ix8_kernel(DsScoreArgs a, Ix8Args x, int* flag, int vid) {
    constexpr int EB = kIwEB;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *flag = vid;
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
    const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
    const long qrow = a.row0 + m;                     /* the step row: q8 / qe are the step's */
    uint2 qa[HT][8];
    float wh[HT][8];
    int qx[HT][8][BLOCK ? 4 : 1];
#pragma unroll
    for (int t = 0; t < HT; ++t) {
        const int h = t * 16 + (lane & 15);
        const unsigned char* qp = x.q8 + (qrow * 32 + h) * 128;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d0 = BLOCK ? 16 * i + 8 * hf : 64 * hf + 8 * i;
            qa[t][i] = *(const uint2*)(qp + d0);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int hh = t * 16 + 8 * hf + e;
            wh[t][e] = ds_round_bf16(ds_bf(a.w[m * a.wr + (long)hh * a.wc]) * a.wscale);
            if constexpr (BLOCK) {
#pragma unroll
                for (int b = 0; b < 4; ++b) qx[t][e][b] = x.qe[(qrow * 32 + hh) * 4 + b];
            } else {
                qx[t][e][0] = x.qe[qrow * 32 + hh];
            }
        }
    }
    for (long j0 = jb; j0 < jend; j0 += 16) {
        if (j0 >= reach) {
            for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            break;
        }
        const long j = j0 + (lane & 15);
        const bool live = j < jend && j < reach;
        long blk = -1;
        if (live) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long b = a.bt[(long)s * a.btr + col * a.btc];
                if (b >= 0 && b < a.nblk) blk = b;
            }
        }
        const unsigned char* kp = x.k8 + (blk >= 0 ? blk : 0) * 128;
        uint2 kb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d0 = BLOCK ? 16 * i + 8 * hf : 64 * hf + 8 * i;
            kb[i] = blk >= 0 ? *(const uint2*)(kp + d0) : make_uint2(0u, 0u);
        }
        int kx[BLOCK ? 4 : 1];
#pragma unroll
        for (int b = 0; b < (BLOCK ? 4 : 1); ++b) kx[b] = blk >= 0 ? x.ke[blk * (BLOCK ? 4 : 1) + b] : 0;
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < HT; ++t) {
            float dv[8];
            if constexpr (BLOCK) {
                ds_v8f pb[4];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    pb[b] = (ds_v8f)(0.0f);
                    pb[b] = ds_wmma8(qa[t][2 * b], kb[2 * b], pb[b]);
                    pb[b] = ds_wmma8(qa[t][2 * b + 1], kb[2 * b + 1], pb[b]);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float sum = 0.0f;
#pragma unroll
                    for (int b = 0; b < 4; ++b) sum = sum + ldexpf(pb[b][e], qx[t][e][b] + kx[b]);
                    dv[e] = sum;
                }
            } else {
                ds_v8f d = (ds_v8f)(0.0f);
#pragma unroll
                for (int i = 0; i < 8; ++i) d = ds_wmma8(qa[t][i], kb[i], d);
#pragma unroll
                for (int e = 0; e < 8; ++e) dv[e] = ldexpf(d[e], qx[t][e][0] + kx[0]);
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (t + half > 0) acc = ds_xor16(acc);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float dot = ds_round_bf16(dv[e]);
                    acc = acc + ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh[t][e]);
                }
            }
        }
        if (hf == 1 && j < jend) srow[j * a.scs] = (live && blk >= 0) ? ds_round_bf16(acc) : -kDsInf;
    }
}

}  // namespace

/* ======================================================================== host */
namespace {

using namespace kb;

constexpr long kT = 4096, kRun = 256, kH = 32, kD = 128;
constexpr long kMaxCtx = 524288, kCmax = kMaxCtx + 1;
constexpr int kKeepBlock = 8, kKeepTop = 2048;

/* quant_roundtrip(fp4_e2m1, ue8m0, 32) of a 32-value block: codes + the scale byte (entry_store_mx4's rule) */
static const float kE2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
void qrt_block(const float* v, uint8_t* codes, uint8_t* sbyte) {
    float am = 0.0f;
    for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(v[i]));
    am = am > 6.0f * 0x1p-126f ? am : 6.0f * 0x1p-126f;
    const float t = am * (1.0f / 6.0f);
    uint32_t u;
    std::memcpy(&u, &t, 4);
    const int k = (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
    const float scale = std::ldexp(1.0f, k);
    for (int i = 0; i < 32; ++i) codes[i] = (uint8_t)ds_e2m1_code(v[i] / scale);
    *sbyte = (uint8_t)(k + 127);
}
float code_val(uint8_t c, int k) { return (c & 8 ? -1.0f : 1.0f) * kE2m1[c & 7] * std::ldexp(1.0f, k); }

/* a 128-vector's values: N(0,1) x a vector scale; stress: each 32-block times 2^U{-5..5} more */
void gen_vec(Rng& g, bool stress, float vscale, float* v) {
    for (int b = 0; b < 4; ++b) {
        const float bs = stress ? std::ldexp(1.0f, (int)(g.next() % 11) - 5) : 1.0f;
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = (float)g.gauss() * vscale * bs;
    }
}

struct Data {
    long ctx = 0, ratio = 0, C = 0, nent = 0, nblk = 0;
    bool stress = false, keep = false;
    uint16_t* q = nullptr;     /* [T, 4096] */
    uint16_t* w = nullptr;     /* [T, 32] */
    uint8_t* kmx = nullptr;    /* [nblk][80] */
    uint16_t* kbf = nullptr;   /* [nblk][128] (bf16 reference form) */
    int* bt = nullptr;         /* [1, C + 1] */
    int* pos = nullptr;        /* [T] */
    int* cu = nullptr;         /* [2] */
    int* keepw = nullptr;      /* [kRun, kwords] for the timed run */
    long kwords = 0;
    /* fp8 prep */
    unsigned char *q8r = nullptr, *q8b = nullptr, *k8r = nullptr, *k8b = nullptr;
    int *qer = nullptr, *qeb = nullptr, *ker = nullptr, *keb = nullptr;
    unsigned* bad = nullptr;
    void release() {
        void* ps[] = { q, w, kmx, kbf, bt, pos, cu, keepw, q8r, q8b, k8r, k8b, qer, qeb, ker, keb, bad };
        for (void* p : ps)
            if (p) KB_CK(hipFree(p));
        *this = Data();
    }
};

Data make_data(long ctx, long ratio, bool stress, bool keep, int run, uint64_t seed) {
    Data d;
    d.ctx = ctx;
    d.ratio = ratio;
    d.C = kMaxCtx / ratio + 1;
    d.nent = ctx / ratio + 1;
    d.nblk = d.nent + 16;
    d.stress = stress;
    d.keep = keep;
    Rng g(seed);
    std::vector<uint16_t> q((size_t)kT * kH * kD);
    std::vector<float> v(128);
    uint8_t codes[32], sb;
    for (long r = 0; r < kT; ++r)
        for (long h = 0; h < kH; ++h) {
            gen_vec(g, stress, (float)std::ldexp(1.0, (int)(g.next() % 5) - 2), v.data());
            for (int b = 0; b < 4; ++b) {
                qrt_block(&v[b * 32], codes, &sb);
                for (int i = 0; i < 32; ++i)
                    q[(size_t)(r * kH + h) * kD + b * 32 + i] = h_bf16(code_val(codes[i], (int)sb - 127));
            }
        }
    std::vector<uint16_t> w((size_t)kT * kH);
    for (auto& x : w) x = h_bf16((float)g.gauss());
    std::vector<uint8_t> km((size_t)d.nblk * kDsMx4Bytes, 0);
    for (long e = 0; e < d.nblk; ++e) {
        gen_vec(g, stress, (float)std::ldexp(1.0, (int)(g.next() % 3) - 1), v.data());
        uint8_t* p = &km[(size_t)e * kDsMx4Bytes];
        for (int b = 0; b < 4; ++b) {
            qrt_block(&v[b * 32], codes, &sb);
            for (int i = 0; i < 32; ++i) {
                const int idx = b * 32 + i;
                p[idx / 2] |= (uint8_t)(codes[i] << (4 * (idx & 1)));
            }
            p[64 + b] = sb;
        }
    }
    std::vector<int> bt((size_t)d.C + 1, -1);
    for (long j = 0; j < d.nblk && j < d.C + 1; ++j) bt[j] = (int)j;
    std::vector<int> pos(kT);
    for (long m = 0; m < kT; ++m) pos[m] = (int)(ctx - kT + m);
    std::vector<int> cu = { 0, (int)kT };
    d.q = dup(q);
    d.w = dup(w);
    d.kmx = dup(km);
    d.bt = dup(bt);
    d.pos = dup(pos);
    d.cu = dup(cu);
    d.kbf = dalloc<uint16_t>((size_t)d.nblk * 128);
    ix_unpack_cache_kernel<<<(unsigned)((d.nblk + 255) / 256), 256>>>(d.kmx, d.nblk, d.kbf);
    KB_CK(hipGetLastError());
    if (keep) {
        /* the timed run's rows: 2048 random blocks of 8 below the row's reach + the newest pinned (all when fewer) */
        d.kwords = ((d.C + kKeepBlock - 1) / kKeepBlock + 31) / 32;
        std::vector<int> kw((size_t)kRun * d.kwords, 0);
        for (long r = 0; r < kRun; ++r) {
            const long m = (long)run * kRun + r;
            const long reach = ((long)pos[m] + 1) / ratio;
            const long nb = (reach + kKeepBlock - 1) / kKeepBlock;
            auto set = [&](long b) { kw[(size_t)r * d.kwords + b / 32] |= (int)(1u << (b % 32)); };
            if (nb <= kKeepTop) {
                for (long b = 0; b < nb; ++b) set(b);
            } else {
                set(nb - 1);
                long got = 1;
                std::vector<uint8_t> used((size_t)nb, 0);
                used[nb - 1] = 1;
                while (got < kKeepTop) {
                    const long b = (long)(g.next() % (uint64_t)nb);
                    if (used[b]) continue;
                    used[b] = 1;
                    set(b);
                    ++got;
                }
            }
        }
        d.keepw = dup(kw);
    }
    /* fp8 prep of the keys (as if stored so) and of the step's q (the per-call part, timed separately) */
    d.bad = dalloc<unsigned>(4);
    d.k8r = dalloc<unsigned char>((size_t)d.nblk * 128);
    d.k8b = dalloc<unsigned char>((size_t)d.nblk * 128);
    d.ker = dalloc<int>((size_t)d.nblk);
    d.keb = dalloc<int>((size_t)d.nblk * 4);
    ix8_prep_kernel<<<(unsigned)((d.nblk + 127) / 128), 128>>>(d.kbf, d.nblk, d.k8r, d.ker, d.k8b, d.keb, d.bad);
    KB_CK(hipGetLastError());
    d.q8r = dalloc<unsigned char>((size_t)kT * kH * 128);
    d.q8b = dalloc<unsigned char>((size_t)kT * kH * 128);
    d.qer = dalloc<int>((size_t)kT * kH);
    d.qeb = dalloc<int>((size_t)kT * kH * 4);
    KB_CK(hipDeviceSynchronize());
    return d;
}

struct Plane {
    float* p = nullptr;
    void alloc() { p = dalloc<float>((size_t)kRun * kCmax); }
};

struct Ctx {
    hipStream_t s;
    Cmper cmp;
    Plane ref, out;
    int* flag = nullptr;
    Lib lib;
    LaunchFn lib_score = nullptr;
    int warm = 2, reps = 10;
    bool quick = false;
    int nvirhe = 0;
};

constexpr uint32_t kSentinel = 0x7fa5a5a5u;   /* a NaN no kernel writes */

/* The RadArgs of run `run` (row_part) of the step: q, w, pos the step's; sc the plane at pitch cmax */
struct RunArgs {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a;
    RadArgs* A() {   /* the operand / parameter arrays may have moved with the struct */
        a.t = t.data();
        a.p = p.data();
        return &a;
    }
};
RunArgs run_args(const Data& d, int run, float* plane, bool bf16keys, bool keep) {
    RunArgs r;
    const long slot = bf16keys ? 128 / d.ratio : 40 / d.ratio;
    r.t.push_back(tens(d.q, RAD_BF16, { kT, kH * kD }));
    r.t.push_back(tens(d.w, RAD_BF16, { kT, kH }));
    r.t.push_back(tens(bf16keys ? (void*)d.kbf : (void*)d.kmx, RAD_BF16, { d.nblk, 1, d.ratio, slot }));
    r.t.push_back(tens(d.bt, RAD_I32, { 1, d.C + 1 }));
    r.t.push_back(tens(d.pos, RAD_I32, { kT }));
    r.t.push_back(tens(d.cu, RAD_I32, { 2 }));
    r.t.push_back(tens(plane, RAD_F32, { kRun, d.C }, { kCmax, 1 }));
    if (keep) r.t.push_back(tens(d.keepw, RAD_I32, { kRun, d.kwords }));
    r.p.push_back(pint("M", kRun));
    r.p.push_back(pint("heads", kH));
    r.p.push_back(pint("dim", kD));
    r.p.push_back(pint("ratio", d.ratio));
    r.p.push_back(pint("width", d.C));
    r.p.push_back(pf64("wscale", 1.0 / std::sqrt((double)kD * (double)kH)));
    r.p.push_back(pstr("dtype", "bf16"));
    r.p.push_back(pint("row_part", run));
    if (keep) r.p.push_back(pint("block", kKeepBlock));
    if (!bf16keys) r.p.push_back(pstr("key_format", "mxfp4"));
    r.a = args(r.t, r.p);
    return r;
}

/* the run's work: live entries (sum of reach over its rows) and the plane's words */
struct Work {
    double live = 0, kept = 0;
    long rlo = 0, rhi = 0;
};
Work run_work(const Data& d, int run) {
    Work w;
    w.rlo = (d.ctx - kT + (long)run * kRun + 1) / d.ratio;
    w.rhi = (d.ctx - kT + (long)run * kRun + kRun) / d.ratio;
    for (long r = 0; r < kRun; ++r) w.live += (double)((d.ctx - kT + (long)run * kRun + r + 1) / d.ratio);
    return w;
}

void fill(Ctx& c, float* p) {
    kb_fill32<<<4096, 256, 0, c.s>>>((uint32_t*)p, kCmax, kRun, kCmax, kSentinel);
    KB_CK(hipGetLastError());
}

const char* kind_name(long ratio, bool keep) { return keep ? "r1k" : (ratio == 2 ? "r2" : "r1"); }

}  // namespace

int main(int argc, char** argv) {
    using namespace kb;
    const char* libp = "/opt/radiance/share/radiance/kernels/libr4d.so";
    std::vector<long> ctxs = { 136 * 1024, 64 * 1024, 16 * 1024 };
    int run = 7;
    bool do_main = true, do_stress = true;
    Ctx c;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--lib") && i + 1 < argc) libp = argv[++i];
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) c.reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warm") && i + 1 < argc) c.warm = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--run") && i + 1 < argc) run = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--quick")) { c.quick = true; ctxs = { 64 * 1024 }; }
        else if (!std::strcmp(argv[i], "--no-stress")) do_stress = false;
        else if (!std::strcmp(argv[i], "--stress-only")) do_main = false;
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctxs.clear();
            std::string s = argv[++i];
            size_t p0 = 0;
            while (p0 < s.size()) {
                size_t p1 = s.find(',', p0);
                if (p1 == std::string::npos) p1 = s.size();
                ctxs.push_back(std::atol(s.substr(p0, p1 - p0).c_str()));
                p0 = p1 + 1;
            }
        } else if (!std::strcmp(argv[i], "--help")) {
            std::printf("kb2_ix [--lib PATH] [--ctx 139264,65536,16384] [--run 7] [--reps 10] [--warm 2] [--quick] [--no-stress | --stress-only]\n");
            return 0;
        }
    }
    hipDeviceProp_t prop;
    KB_CK(hipGetDeviceProperties(&prop, 0));
    std::printf("KB2 DEV %s %s cus=%d clock_khz=%d mem_gib=%.1f l2_mib=%.1f\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount, prop.clockRate, prop.totalGlobalMem / 1073741824.0, prop.l2CacheSize / 1048576.0);
    KB_CK(hipStreamCreate(&c.s));
    if (c.lib.open(libp)) c.lib_score = c.lib.sym<LaunchFn>("r4d_rad_ds41_index_score");
    if (!c.lib_score) {
        std::printf("KB2 VIRHE LIB r4d_rad_ds41_index_score not loaded from %s\n", libp);
        return 2;
    }
    c.ref.alloc();
    c.out.alloc();
    c.flag = dalloc<int>(4);
    attr("ix_stock_wm<2,false>", ds41_index_score_wm_kernel<2, false>, 256, 0);
    attr("ix_stock_wm<2,true>", ds41_index_score_wm_kernel<2, true>, 256, 0);
    attr("ixv_v128", ixv_kernel<2, false, 128, 1, 0, 0>, 256, 0);
    attr("ixv_eb256", ixv_kernel<2, false, 256, 1, 0, 0>, 256, 0);
    attr("ixv_pf", ixv_kernel<2, false, 128, 1, kVPF, 0>, 256, 0);
    attr("ixv_r2", ixv_kernel<2, false, 128, 2, 0, 0>, 256, 0);
    attr("ixv_kpf", ixv_kernel<2, true, 1024, 1, kVPF, 0>, 256, 0);
    attr("ixv_v1024k", ixv_kernel<2, true, 1024, 1, 0, 0>, 256, 0);
    attr("ixl_lds", ixl_kernel<2, 128, 1>, 256, 0);
    attr("ixl_lds_r2", ixl_kernel<2, 128, 2>, 256, 0);
    attr("ix8_f8r", ix8_kernel<2, false>, 256, 0);
    attr("ix8_f8b", ix8_kernel<2, true>, 256, 0);
    std::fflush(stdout);

    /* One case: (ctx, ratio, keep, data). Every form launched once into a sentinel-filled plane, its run flag checked
     * and its bytes compared with LIB's over the whole [256, width] (NI: [256, its launched width]); then timed. */
    auto one_case = [&](long ctx, long ratio, bool keep, bool stress, bool timing) {
        const char* kn = kind_name(ratio, keep);
        const char* dn = stress ? "stress" : "real";
        Data d = make_data(ctx, ratio, stress, keep, run, 0x5eed0000ull + (uint64_t)ctx * 7 + ratio * 3 + keep + (stress ? 101 : 0));
        unsigned hbad[2] = { 0, 0 };
        /* the step's q in fp8 (per call; timed) */
        auto qprep = [&] {
            ix8_prep_kernel<<<(unsigned)((kT * kH + 127) / 128), 128, 0, c.s>>>(d.q, kT * kH, d.q8r, d.qer, d.q8b, d.qeb, d.bad);
        };
        qprep();
        KB_CK(hipStreamSynchronize(c.s));
        KB_CK(hipMemcpy(hbad, d.bad, 8, hipMemcpyDeviceToHost));
        const Work wk = run_work(d, run);
        std::printf("KB2 IXCASE ctx=%ld kind=%s data=%s run=%d M=%ld C=%ld pitch=%ld reach=%ld..%ld live=%.0f "
                    "plane_mib=%.1f keys=%ld f8_unrepresentable=%u/%u\n",
                    ctx, kn, dn, run, kRun, d.C, kCmax, wk.rlo, wk.rhi, wk.live, kRun * d.C * 4.0 / 1048576.0, d.nent,
                    hbad[0], hbad[1]);
        RunArgs ra_ref = run_args(d, run, c.ref.p, false, keep);
        RunArgs ra_out = run_args(d, run, c.out.p, false, keep);
        DsScoreArgs sa{};
        if (!mk_score_args(ra_out.A(), &sa)) {
            std::printf("KB2 VIRHE IX args\n");
            ++c.nvirhe;
            return;
        }
        fill(c, c.ref.p);
        int rc = c.lib_score(ra_ref.A(), (RadStream)c.s);
        KB_CK(hipStreamSynchronize(c.s));
        if (rc != RAD_OK) {
            std::printf("KB2 VIRHE IXCHK LIB rc=%d ctx=%ld kind=%s\n", rc, ctx, kn);
            ++c.nvirhe;
            return;
        }
        const unsigned long long unw = c.cmp.count_ne(c.ref.p, kCmax, kRun, 0, d.C, kSentinel, c.s);
        const unsigned long long sent_left = (unsigned long long)kRun * d.C - unw;
        std::printf("KB2 IXREF ctx=%ld kind=%s data=%s LIB wrote %llu of %ld words (sentinel left %llu)\n", ctx, kn, dn,
                    unw, kRun * d.C, sent_left);
        /* dead region past the run's furthest reach: LIB's -inf words (what NI does not write) */
        const unsigned long long dead_ninf =
            c.cmp.count_ne(c.ref.p, kCmax, kRun, std::min(wk.rhi, d.C), d.C, 0xff800000u, c.s);
        std::printf("KB2 IXDEAD ctx=%ld kind=%s data=%s cols=%ld..%ld non_minus_inf=%llu\n", ctx, kn, dn,
                    std::min(wk.rhi, d.C), d.C, dead_ninf);

        struct Form {
            std::string name;
            int vid;
            bool exact;       /* exact by construction: a difference is a VIRHE */
            bool ablation;    /* bytes differ by design: no check */
            long cols;        /* compared columns (0 = the whole width) */
            std::function<int()> launch;
            bool ok = true;   /* launched, ran, (exact) the same bytes: timed only then */
        };
        std::vector<Form> forms;
        const dim3 b256(256);
        const unsigned gy8 = (unsigned)((sa.M + 7) / 8), gy16 = (unsigned)((sa.M + 15) / 16);
        auto gx = [&](int eb) { return (unsigned)((sa.C + eb - 1) / eb); };
        const int EBK = keep ? kIwKeepEB : kIwEB;
        forms.push_back({ "LIB", 0, true, false, 0, [&] { return c.lib_score(ra_out.A(), (RadStream)c.s); } });
        forms.push_back({ "STK", 0, true, false, 0, [&] { return index_score_body(ra_out.A(), (RadStream)c.s); } });
        /* NI: the stock kernel over [0, the run's furthest reach) only */
        forms.push_back({ "NI", 0, true, false, std::min(wk.rhi, d.C), [&] {
                              const dim3 g((unsigned)((std::min(wk.rhi, d.C) + EBK - 1) / EBK), gy8);
                              if (keep) ds41_index_score_wm_kernel<2, true><<<g, b256, 0, c.s>>>(sa);
                              else ds41_index_score_wm_kernel<2, false><<<g, b256, 0, c.s>>>(sa);
                              return (int)hipGetLastError();
                          } });
#define IXV(NAME, VID, KEEPV, EBV, RPWV, FLV, ABLV, EXACT, ABL)                                                       \
    forms.push_back({ NAME, VID, EXACT, ABL, 0, [&] {                                                                  \
                          const dim3 g(gx(EBV), RPWV == 2 ? gy16 : gy8);                                                \
                          ixv_kernel<2, KEEPV, EBV, RPWV, FLV, ABLV><<<g, b256, 0, c.s>>>(sa, c.flag, VID);              \
                          return (int)hipGetLastError();                                                                \
                      } })
        if (!keep) {
            IXV("V128", 11, false, 128, 1, 0, 0, true, false);
            IXV("EB256", 12, false, 256, 1, 0, 0, true, false);
            IXV("EB512", 13, false, 512, 1, 0, 0, true, false);
            IXV("PF", 14, false, 128, 1, kVPF, 0, true, false);
            IXV("PF512", 15, false, 512, 1, kVPF, 0, true, false);
            IXV("R2", 16, false, 128, 2, 0, 0, true, false);
            forms.push_back({ "LDS", 21, true, false, 0, [&] {
                                  ixl_kernel<2, 128, 1><<<dim3(gx(128), gy8), b256, 0, c.s>>>(sa, c.flag, 21);
                                  return (int)hipGetLastError();
                              } });
            forms.push_back({ "LDSR2", 22, true, false, 0, [&] {
                                  ixl_kernel<2, 128, 2><<<dim3(gx(128), gy16), b256, 0, c.s>>>(sa, c.flag, 22);
                                  return (int)hipGetLastError();
                              } });
            forms.push_back({ "F8R", 31, false, false, 0, [&] {
                                  Ix8Args x{ d.q8r, d.qer, d.k8r, d.ker };
                                  ix8_kernel<2, false><<<dim3(gx(128), gy8), b256, 0, c.s>>>(sa, x, c.flag, 31);
                                  return (int)hipGetLastError();
                              } });
            forms.push_back({ "F8B", 32, false, false, 0, [&] {
                                  Ix8Args x{ d.q8b, d.qeb, d.k8b, d.keb };
                                  ix8_kernel<2, true><<<dim3(gx(128), gy8), b256, 0, c.s>>>(sa, x, c.flag, 32);
                                  return (int)hipGetLastError();
                              } });
            if (!stress) {
                IXV("A_NOWMMA", 41, false, 128, 1, 0, kA_NOWMMA, false, true);
                IXV("A_NOEPI", 42, false, 128, 1, 0, kA_NOEPI, false, true);
                IXV("A_NOUNPACK", 43, false, 128, 1, 0, kA_NOUNPACK, false, true);
                IXV("A_NOSTORE", 44, false, 128, 1, 0, kA_NOSTORE, false, true);
                IXV("A_NOWMMA_NOEPI", 45, false, 128, 1, 0, kA_NOWMMA | kA_NOEPI, false, true);
            }
        } else {
            IXV("V1024", 51, true, 1024, 1, 0, 0, true, false);
            IXV("KPF", 52, true, 1024, 1, kVPF, 0, true, false);
            IXV("KR2", 53, true, 1024, 2, 0, 0, true, false);
            IXV("K256", 55, true, 256, 1, 0, 0, true, false);
            IXV("K256PF", 56, true, 256, 1, kVPF, 0, true, false);
        }
#undef IXV
        /* the bf16-key form (RADIANCE_DS41_IDXKEY unset): LIB and STK on the unpacked cache, checked against LIB mx4 */
        RunArgs ra_bf = run_args(d, run, c.out.p, true, keep);
        if (!keep && !stress) {
            forms.push_back({ "LIB_BF16KEY", 0, true, false, 0, [&] { return c.lib_score(ra_bf.A(), (RadStream)c.s); } });
        }
        for (auto& f : forms) {
            fill(c, c.out.p);
            int z = 0;
            KB_CK(hipMemcpyAsync(c.flag, &z, 4, hipMemcpyHostToDevice, c.s));
            const int lrc = f.launch();
            KB_CK(hipStreamSynchronize(c.s));
            int fl = 0;
            KB_CK(hipMemcpy(&fl, c.flag, 4, hipMemcpyDeviceToHost));
            const char* fstate = f.vid == 0 ? "n/a" : (fl == f.vid ? "ran" : (fl == -f.vid ? "REFUSED" : "MISSING"));
            if (lrc != 0 || (f.vid != 0 && fl != f.vid)) {
                std::printf("KB2 VIRHE IXCHK %s ctx=%ld kind=%s data=%s launch rc=%d flag=%d (%s)\n", f.name.c_str(), ctx,
                            kn, dn, lrc, fl, fstate);
                ++c.nvirhe;
                f.ok = false;
                continue;
            }
            if (f.ablation) {
                std::printf("KB2 IXCHK %s ctx=%ld kind=%s data=%s ablation (bytes differ by design) flag=%s\n",
                            f.name.c_str(), ctx, kn, dn, fstate);
            } else {
                const long cols = f.cols > 0 ? std::min(f.cols, d.C) : d.C;
                const Cmp r = c.cmp.words(c.ref.p, c.out.p, kCmax, kCmax, kRun, cols, c.s);
                const char* verdict = r.diff == 0 ? "SAME" : (f.exact ? "DIFFERS" : "NONEXACT_DIFFERS");
                long fr = -1, fc = -1;
                if (r.diff) {
                    fr = (long)(r.first / (unsigned long long)cols);
                    fc = (long)(r.first % (unsigned long long)cols);
                }
                std::printf("KB2 IXCHK %s ctx=%ld kind=%s data=%s compared_bytes=%llu cols=%ld diff_words=%llu first=%ld,%ld "
                            "flag=%s verdict=%s\n",
                            f.name.c_str(), ctx, kn, dn, r.compared, cols, r.diff, fr, fc, fstate, verdict);
                if (r.diff && f.exact) {
                    ++c.nvirhe;
                    f.ok = false;
                }
            }
            std::fflush(stdout);
        }
        if (timing) {
            const double flop = wk.live * 8192.0;
            for (auto& f : forms) {
                if (!f.ok) {
                    std::printf("KB2 IXT %s ctx=%ld kind=%s skipped (its check failed)\n", f.name.c_str(), ctx, kn);
                    continue;
                }
                const Tm t = time_it(c.s, c.warm, c.reps, [&] { f.launch(); });
                double bytes = (double)kRun * (double)(f.name == "NI" ? std::min(wk.rhi, d.C) : d.C) * 4.0 +
                               (double)d.nent * (f.name == "LIB_BF16KEY" ? 256.0 : (f.name[0] == 'F' ? 132.0 : 80.0)) +
                               (double)kRun * kH * kD * 2.0;
                std::printf("KB2 IXT %s ctx=%ld kind=%s med_us=%.1f p10_us=%.1f p90_us=%.1f n=%d live_tflops=%.2f "
                            "compulsory_gbs=%.1f\n",
                            f.name.c_str(), ctx, kn, t.med, t.p10, t.p90, t.n, flop / (t.med * 1e-6) / 1e12,
                            bytes / (t.med * 1e-6) / 1e9);
                std::fflush(stdout);
            }
            if (!keep) {
                const Tm t = time_it(c.s, c.warm, c.reps, qprep);
                std::printf("KB2 IXT F8QPREP ctx=%ld kind=%s med_us=%.1f p10_us=%.1f p90_us=%.1f n=%d (a whole 4096-row step's "
                            "q, once a step; a run's share = /16)\n",
                            ctx, kn, t.med, t.p10, t.p90, t.n);
            }
        }
        d.release();
    };

    if (do_main)
        for (long ctx : ctxs) {
            one_case(ctx, 2, false, false, true);
            one_case(ctx, 1, false, false, true);
            if (ctx / 1 >= 2L * kKeepTop * kKeepBlock) one_case(ctx, 1, true, false, true);
        }
    /* stress data (block binades spread over +-5: the 128-term dot rounds), checks only: every exact form must still
     * be SAME; F8R / F8B show whether the fp8 WMMA forms hold the stock bits when the sums round */
    if (do_stress) {
        one_case(c.quick ? 16 * 1024 : 64 * 1024, 1, false, true, false);
        one_case(c.quick ? 16 * 1024 : 64 * 1024, 2, false, true, false);
    }
    std::printf("KB2 IXDONE virhe=%d\n", c.nvirhe);
    return c.nvirhe ? 1 : 0;
}
