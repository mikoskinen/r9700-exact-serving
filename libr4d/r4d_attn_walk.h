// Lever P-2 (Qwen loop round 11, QUEUE Q8): the TP4 sparse prefill attention as a q-tiled UNION WALK, bench form.
//
// Today (attn_decode_gq_h256_gqa6_bs4_fp8kv at a prefill piece) every query row is its own "sequence": a one-wave
// workgroup walks the row's own QSA selection (513 four-token pages at 116k, 129 tiles of 16 keys) and reads every
// selected page from L2 / MALL. Consecutive rows of a prompt select mostly the same pages (union factor 2.4 / 3.7 / 5.9
// at 4 / 8 / 16 rows, the 12.11. dumps), so the same KV bytes cross the memory system up to ~6 times for 16 rows.
//
// The walk: a workgroup takes R consecutive rows (one compute wave each) and NLW = 4 loader waves. The loaders stream the
// UNION of the R rows' pages (ascending page id, built outside: `ulist`, `uoff`) into an LDS ring of NS = 2M page slots,
// M pages a round; each compute wave runs ITS OWN row exactly as today's kernel does (the same q fragments, the same
// tiles = its own list entries 4t .. 4t + 3 in list order, the same WMMA sequence, the lazy max, the f16 partial), only
// the bytes of a tile come from the ring when all four of its pages are resident, else from the cache as today. A tile is
// taken in the round its last page arrives, in the row's tile order, so the arithmetic per row is today's operation for
// operation -> the same bytes (the bench compares o / codes / scales with today's launch). Rows whose tile reaches past
// the ring (a page that left it, a list entry out of ascending order, the row's last partial tile) read that tile from
// the cache: still today's bytes.
//
// Only splits == 1 (the law's count for a prefill launch of >= 129 rows); the merge is today's (gate + int8 / E4M3 codes).
// Ring slot layout (2,112 bytes): K as [token][256 codes] at a 272-byte token pitch (the 16 keys of a tile's fragment
// read fall on distinct banks within a page), then V as [dim][4 tokens] (one dword a dim = a PV fragment's half word).
#pragma once

namespace r4dwalk {
constexpr int HD = 256, GQA = 6, BS = 4, TILE = 16, NKS = HD / 16, PF = 2, ROWB = 2 * HD;
constexpr int KST = 272, VOFFL = 4 * KST, SLOT = VOFFL + 4 * HD;
constexpr int NLW = 4, NLT = NLW * 32;   // loader waves / threads
// a wave-uniform four-entry array at a lane-dependent index, as selects (no private-memory indexing)
__device__ __forceinline__ int pick4(const int* v, int i) {
    return i == 0 ? v[0] : i == 1 ? v[1] : i == 2 ? v[2] : v[3];
}
__device__ __forceinline__ uint32_t u4get(r4d_u32x4 v, int q) { return v[q]; }
}

template<int R, int M>
__global__ __launch_bounds__((R + r4dwalk::NLW) * 32)
void r4d_attn_walk_kernel(const R4DArgs a, const int* __restrict__ ulist, const int* __restrict__ uoff,
                          const int* __restrict__ upos, int* __restrict__ wstat)
{
    using namespace r4dwalk;
    constexpr int NS = 2 * M;
    typedef DT16<1> D;
    typedef D::frag frag16;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;
    __shared__ __attribute__((aligned(16))) uint8_t win[NS * SLOT];

    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __builtin_amdgcn_readfirstlane(tid >> 5);   // wave-uniform: the role and the row
    const int grp = blockIdx.x, kvh = blockIdx.y, kvc = kvh;
    const int u0 = uoff[grp], nu = uoff[grp + 1] - u0;
    const int nr = (nu + M - 1) / M;
    const uint8_t* __restrict__ kvb = (const uint8_t*)a.kv + (size_t)kvc * a.kv_head_stride;

    if (warp >= R) {
        // ---- the loader waves: round j's pages into ring slots p mod NS (K rows, V transposed to [dim][token])
        const int lt = tid - R * 32;                     // 0 .. NLT - 1
        constexpr int KJ = (64 * M + NLT - 1) / NLT;           // 16-byte K chunks a thread a round (64 a page)
        constexpr int VJ = (16 * M + NLT - 1) / NLT;           // 16-dim V chunks a thread a round (16 a page), 4 tokens each
        r4d_u32x4 kr[KJ];                                // native vectors: an array of uint4 stays in scratch
        r4d_u32x4 vr[VJ * 4];
        auto issue = [&](int j) __attribute__((always_inline)) {
            #pragma unroll
            for (int r = 0; r < KJ; ++r) {
                const int id = lt + NLT * r, ch = id & 63, p = min(j * M + min(id >> 6, M - 1), nu - 1);
                const int pg = ulist[u0 + p];       // clamped: a valid page, stored only where (b < M, p < nu)
                kr[r] = *(const r4d_u32x4*)(kvb + (size_t)pg * a.kv_block_stride + (size_t)(ch >> 4) * ROWB + (ch & 15) * 16);
            }
            #pragma unroll
            for (int r = 0; r < VJ; ++r) {
                const int id = lt + NLT * r, dc = id & 15, p = min(j * M + min(id >> 4, M - 1), nu - 1);
                const int pg = ulist[u0 + p];
                const uint8_t* src = kvb + (size_t)pg * a.kv_block_stride + HD + dc * 16;
                #pragma unroll
                for (int t = 0; t < 4; ++t) vr[r * 4 + t] = *(const r4d_u32x4*)(src + (size_t)t * ROWB);
            }
        };
        auto store = [&](int j) __attribute__((always_inline)) {
            #pragma unroll
            for (int r = 0; r < KJ; ++r) {
                const int id = lt + NLT * r, b = id >> 6, ch = id & 63, p = j * M + b;
                if (b < M && p < nu)
                    *(r4d_u32x4*)(win + (p % NS) * SLOT + (ch >> 4) * KST + (ch & 15) * 16) = kr[r];
            }
            #pragma unroll
            for (int r = 0; r < VJ; ++r) {
                const int id = lt + NLT * r, b = id >> 4, dc = id & 15, p = j * M + b;
                if (b < M && p < nu) {
                    uint8_t* dst = win + (p % NS) * SLOT + VOFFL + dc * 16 * 4;
                    const r4d_u32x4 t0 = vr[r * 4 + 0], t1 = vr[r * 4 + 1], t2 = vr[r * 4 + 2], t3 = vr[r * 4 + 3];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {      // dims dc * 16 + 4 q .. + 3
                        const uint32_t w0 = u4get(t0, q), w1 = u4get(t1, q), w2 = u4get(t2, q), w3 = u4get(t3, q);
                        // dword for dim k = byte k of w0 | byte k of w1 << 8 | ... (token i = byte i)
                        const uint32_t lo01 = __builtin_amdgcn_perm(w1, w0, 0x05010400u);   // w0.b0 w1.b0 w0.b1 w1.b1
                        const uint32_t lo23 = __builtin_amdgcn_perm(w3, w2, 0x05010400u);
                        const uint32_t hi01 = __builtin_amdgcn_perm(w1, w0, 0x07030602u);   // w0.b2 w1.b2 w0.b3 w1.b3
                        const uint32_t hi23 = __builtin_amdgcn_perm(w3, w2, 0x07030602u);
                        uint4 o;
                        o.x = __builtin_amdgcn_perm(lo23, lo01, 0x05040100u);   // dim 0: w0.b0 w1.b0 w2.b0 w3.b0
                        o.y = __builtin_amdgcn_perm(lo23, lo01, 0x07060302u);   // dim 1
                        o.z = __builtin_amdgcn_perm(hi23, hi01, 0x05040100u);   // dim 2
                        o.w = __builtin_amdgcn_perm(hi23, hi01, 0x07060302u);   // dim 3
                        *(uint4*)(dst + q * 16) = o;
                    }
                }
            }
        };
        if (nr > 0) issue(0);
        for (int j = 0; j < nr; ++j) {
            store(j);
            lds_barrier();                       // B_j: round j's pages visible
            if (j + 1 < nr) issue(j + 1);        // in flight during round j's compute
            lds_barrier();                       // A_j: round j computed; round j + 1 may overwrite round j - 1's slots
        }
        return;
    }

    // ---- a compute wave: row `seq`, today's per-row arithmetic (r4d_attn_decode_body at NWARPS 1, TILE 16, h256, gqa 6,
    // page 4, fp8 cache, G8_D_OPT | O_V8: F16, MSKIP, FOLDQ, KLDS0, PF16, CONTIG, PF 2, DOT2; one split)
    const int seq = grp * R + warp;
    const int c = lane & 15, h = lane >> 4;
    int ctx = 0, qoff = 0, qlen = 0;
    if (seq < a.num_seqs) {
        ctx  = a.seqused_k[seq];
        qoff = r4d_q_off(a.cu_q, seq, a.q_len);
        qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    }
    const bool act = seq < a.num_seqs && ctx > 0 && qlen > 0;
    if (!act) {                                               // no row: only the rounds' barriers
        for (int j = 0; j < nr; ++j) { lds_barrier(); lds_barrier(); }
        return;
    }
    const int ntl = act ? r4d_attn_tiles(ctx, TILE) : 0;
    const int qi = c / GQA, hi = c - qi * GQA;
    const bool live = act && (qi < qlen);
    const int qrow = qi < qlen ? qi : (qlen - 1);
    const int qhead = kvh * GQA + hi;
    const int* __restrict__ bt = a.block_table + (size_t)(act ? seq : 0) * a.max_blocks;
    const int* __restrict__ up = upos + (size_t)(act ? seq : 0) * a.max_blocks;
    const float kdesc = (act && a.k_descale) ? a.k_descale[seq * a.kv_heads + kvc] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    {
        const uint16_t* qp = (const uint16_t*)a.q + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HD;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + 8 * h);
            const uint2 h0 = *(const uint2*)(qp + 16 * t + 8 * h + 4);
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, 1, 1>(l0.x, mul), r4d_qcvt<D, 1, 1>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, 1, 1>(h0.x, mul), r4d_qcvt<D, 1, 1>(h0.y, mul)));
        }
    }
    const int pos = ctx - qlen + qrow;
    const int cz  = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[act ? seq : 0] : a.causal;
    const int klimit = cz ? pos : ctx - 1;
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};
    const int lastb = (ctx - 1) / BS;
    int n_lds = 0, n_glb = 0;

    // one tile: QK, the lazy max, P, PV -- today's sequence; `lds` = every page of the tile is in the ring (slots sl[4])
    auto tile = [&](bool lds, int ti, const int* e, const int* sl) __attribute__((always_inline)) {
        const int k0 = ti * TILE;
        const int kk = min(k0 + c, ctx - 1);
        const uint8_t* growg = kvb + (size_t)pick4(e, (kk - k0) / BS) * a.kv_block_stride + (size_t)(kk % BS) * ROWB;
        const uint8_t* growl = win + pick4(sl, c >> 2) * SLOT + (c & 3) * KST;
        v8f s = (v8f){0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int g = 0; g < NKS; g += PF) {
            frag16 kb[PF];
            uint2 rawk[PF];
            if (lds) {
                #pragma unroll
                for (int j = 0; j < PF; ++j) rawk[j] = *(const uint2*)(growl + 16 * (g + j) + 8 * h);
            } else {
                #pragma unroll
                for (int j = 0; j < PF; ++j) rawk[j] = *(const uint2*)(growg + 16 * (g + j) + 8 * h);
            }
            #pragma unroll
            for (int j = 0; j < PF; ++j) {
                uint32_t w4[4];
                fp8x8_to_16x4w<D>(rawk[j], w4);
                kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
            }
            sched_barrier();
            #pragma unroll
            for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
        }
        float sc[8];
        #pragma unroll
        for (int e2 = 0; e2 < 8; ++e2) sc[e2] = s[e2];
        const int kbase = k0 + 8 * h;
        if (wave_any(kbase + 7 > klimit)) {
            #pragma unroll
            for (int e2 = 0; e2 < 8; ++e2) sc[e2] = (kbase + e2 <= klimit) ? sc[e2] : -INFINITY;
        }
        float smax = sc[0];
        #pragma unroll
        for (int e2 = 1; e2 < 8; ++e2) smax = fmaxf(smax, sc[e2]);
        if (wave_any(smax > m_ref + PGROW)) {
            const float rmax  = fmaxf(smax, swap16(smax));
            const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
            const float alpha = ex2(m_ref - mnew);
            #pragma unroll
            for (int dt = 0; dt < NKS; ++dt)
                #pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) acc[dt][e2] *= alpha;
            l_i *= alpha;
            m_ref = mnew;
            off   = PSHIFT - mnew;
        }
        float pf[8];
        #pragma unroll
        for (int e2 = 0; e2 < 8; ++e2) pf[e2] = ex2(sc[e2] + off);
        uint32_t pw[4];
        #pragma unroll
        for (int e2 = 0; e2 < 8; e2 += 2) pw[e2 >> 1] = D::pk(pf[e2], pf[e2 + 1]);
        {
            const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
            #pragma unroll
            for (int e2 = 0; e2 < 4; ++e2)
                l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e2]), one, l_i, false);
        }
        const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));
        // V: lane (c, h) wants dim c + 16 t, keys 8 h .. 8 h + 7 = pages 2 h and 2 h + 1 of the tile
        const uint8_t* vl0 = win + (h ? sl[2] : sl[0]) * SLOT + VOFFL;
        const uint8_t* vl1 = win + (h ? sl[3] : sl[1]) * SLOT + VOFFL;
        #pragma unroll
        for (int g = 0; g < NKS; g += PF) {
            frag16 vb[PF];
            uint2 rawv[PF];
            if (lds) {
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int d = c + (g + j) * 16;
                    rawv[j].x = *(const uint32_t*)(vl0 + d * 4);
                    rawv[j].y = *(const uint32_t*)(vl1 + d * 4);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int d = c + (g + j) * 16;
                    uint32_t wv[2] = { 0u, 0u };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int kv8 = min(k0 + 8 * h + i, ctx - 1);
                        const uint32_t byt = kvb[(size_t)pick4(e, (kv8 - k0) / BS) * a.kv_block_stride + (size_t)(kv8 % BS) * ROWB
                                                 + HD + d];
                        wv[i >> 2] |= byt << (8 * (i & 3));
                    }
                    rawv[j] = make_uint2(wv[0], wv[1]);
                }
            }
            #pragma unroll
            for (int j = 0; j < PF; ++j) {
                uint32_t w4[4];
                fp8x8_to_16x4w<D>(rawv[j], w4);
                vb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
            }
            sched_barrier();
            #pragma unroll
            for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
        }
    };

    // the row's tiles in order, one loop: a full tile waits (barriers) for the round its last page arrives in; it reads the
    // ring if all four pages are still resident there, else the cache; the partial last tile reads the cache. Round j holds
    // ring positions [(j - 1) M, (j + 1) M) between its barriers B_j (barrier 2 j + 1) and A_j (2 j + 2).
    int nb = 0;                                               // barriers passed
    for (int ti = 0; ti < ntl; ++ti) {
        const int t0 = ti * (TILE / BS);
        const bool full = (ti + 1) * TILE <= ctx;
        int e[4], sl[4] = { 0, 0, 0, 0 };
        bool res = false;
        if (full) {
            int mx = -1, mn = 1 << 30;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                e[i] = bt[t0 + i];
                const int p = up[t0 + i];
                mx = max(mx, p); mn = min(mn, p);
                sl[i] = p % NS;
            }
            const int need = min(mx / M, nr - 1);
            while (nb < 2 * need + 1) { lds_barrier(); ++nb; }
            const int jc = (nb - 1) >> 1;
            res = nb > 0 && (nb & 1) && mn >= (jc - 1) * M && mx < (jc + 1) * M;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) e[i] = bt[min(t0 + i, lastb)];
        }
        tile(res, ti, e, sl);
        if (res) ++n_lds; else ++n_glb;
    }
    while (nb < 2 * nr) { lds_barrier(); ++nb; }
    if (wstat && lane == 0 && act) { atomicAdd(&wstat[0], n_lds); atomicAdd(&wstat[1], n_glb); }

    if (!live) return;
    const float lrow = l_i + swap16_fi(l_i);
    const int tok = qoff + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead);    // splits 1, sp 0
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads;
    uint16_t* so = (uint16_t*)a.scratch;
    float* sm = (float*)(so + nrow * HD);
    float* slp = sm + nrow;
    uint16_t* op = so + idx * HD;
    const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt) {
        uint32_t tw[4];
        #pragma unroll
        for (int e2 = 0; e2 < 8; e2 += 2) tw[e2 >> 1] = DT16<1>::pk(acc[dt][e2] * inv, acc[dt][e2 + 1] * inv);
        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
    if (h == 0) { sm[idx] = m_ref; slp[idx] = lrow; }
}

#ifndef R4D_WALK_KERNEL_ONLY   /* tests/r4d_emu/test_qwalk.cpp compiles the kernel alone, beside today's */
/* The bench entry (r4d_selftest --perfatwalk): attn_decode_gq_h256_gqa6_bs4_fp8kv's RadArgs, plus the union lists
 * (`ulist` page ids of group g at [uoff[g], uoff[g + 1]), groups of R consecutive rows; `upos` [rows][max_blocks] each
 * entry's place in its group's union), R in {4, 8, 16}, M in {7, 15}; `wstat` (2 ints, may be null) counts tiles from
 * the ring / from the cache. The walk writes today's f16 partials at splits 1; the merge is today's gated one.
 * Returns RAD_OK, a refusal (splits != 1, a geometry it does not serve) or -100 - hipError_t. */
extern "C" int r4d_attn_walk_gq6_bench(const RadArgs* a, RadStream stream, const int* ulist, const int* uoff,
                                       const int* upos, int R, int M, int* wstat) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    /* the split count today's launch takes (g8_decode_launch: the law, or a tuned `splits`); g8_decode_common leaves r.splits
     * at the tuned value (0 = the law), so the law is asked here -- the walk serves only the one-split form (Qwen loop F33) */
    const int law = r4d_decode_splits(&r, G8_D_TILE);
    if (law != 1 || r.q_len != 1 || r.head_dim != 256 || r.block_size != 4 || r.q_heads != 6 * r.kv_heads)
        return R4D_NO(RAD_E_SHAPE);
    const int groups = (r.num_seqs + R - 1) / R;
    int emit_rc = RAD_OK;
#define WALK_LAUNCH(RR, MM) emit_rc = R4D_EMIT(stream, (r4d_attn_walk_kernel<RR, MM>), dim3(groups, r.kv_heads, 1), \
                                               dim3((RR + r4dwalk::NLW) * 32), 0, r, ulist, uoff, upos, wstat)
    if      (R == 4  && M == 15) WALK_LAUNCH(4, 15);
    else if (R == 8  && M == 15) WALK_LAUNCH(8, 15);
    else if (R == 16 && M == 15) WALK_LAUNCH(16, 15);
    else if (R == 8  && M == 7)  WALK_LAUNCH(8, 7);
    else if (R == 16 && M == 7)  WALK_LAUNCH(16, 7);
    else return R4D_NO(RAD_E_UNSUPPORTED);
#undef WALK_LAUNCH
    if (emit_rc != RAD_OK) return -5;
    CHK8();
    const int jobs3 = r4d_q_total(r) * r.q_heads;
    const int splits = 1;
    if (jobs3 >= G8_D_VECMIN) {
        emit_rc = gq.i8 ? R4D_EMIT(stream, (r4d_attn_splitkv_combine_vec_gq_kernel<256, 4, 1, G8_D_JPB, 0, 2>),
                                   dim3((jobs3 + G8_D_JPB - 1) / G8_D_JPB), dim3(32 * G8_D_JPB), 0, r, splits, G8_D_TILE, gq)
                        : R4D_EMIT(stream, (r4d_attn_splitkv_combine_vec_gq_kernel<256, 4, 1, G8_D_JPB, 0, 1>),
                                   dim3((jobs3 + G8_D_JPB - 1) / G8_D_JPB), dim3(32 * G8_D_JPB), 0, r, splits, G8_D_TILE, gq);
    } else {
        emit_rc = gq.i8 ? R4D_EMIT(stream, (r4d_attn_splitkv_combine_gq_kernel<256, 4, 1, 0, 2>), dim3(jobs3), dim3(256),
                                   splits * sizeof(float), r, splits, G8_D_TILE, gq)
                        : R4D_EMIT(stream, (r4d_attn_splitkv_combine_gq_kernel<256, 4, 1, 0, 1>), dim3(jobs3), dim3(256),
                                   splits * sizeof(float), r, splits, G8_D_TILE, gq);
    }
    if (emit_rc != RAD_OK) return -5;
    CHK8();
    return RAD_OK;
}
#endif
