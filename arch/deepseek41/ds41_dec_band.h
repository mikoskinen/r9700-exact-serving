/* ds41_dec_band.h -- tp8 loop FINDINGS F58: a decode segment of more than the decode band's rows (kBiBand = 48) in
 * deepseek41's step_bi, cut into pieces that each stay in the band. Header-only so tests/ds41_dband_test.cpp checks the
 * arithmetic the plugin issues by.
 *
 * Why: libr4d's attn_sparse_sink and index_score pick their form by the launch's height -- the scalar decode kernels
 * (libref's bits) up to kWmRows = 48 rows, WMMA forms above it, which are NOT those bits (r4d_ds41_sparse.hip's head
 * note: an online softmax, v_exp_f32 and P as bf16 hi + lo; index_score's dot in another order). step_bi issued the
 * attention once a segment and the indexer's runs at irows (>= 96) rows, so a decode segment of 49 .. rows (depth-3
 * drafting at lanes of 16: 16 x 4 = 64 rows, fn4 sched_row_cap 64) took the WMMA forms, and a sequence's answer
 * depended on how many rows shared its step (F58: lq150 loadid c32 9 / 20 DIFFERENT at sched_row_cap 64, a1 IDENTICAL).
 *
 * The attention reads a row's sequence from the launch's cu_seqlens (ds_seq_of_row over cu relative to the launch's
 * first row), its window / compressed block tables a sequence, and its positions, selections and output a row, so a
 * piece is a run of WHOLE consecutive decode sequences [s0, s1) at rows [r0, r0 + n), its cu = cu[s0 .. s1] - cu[s0].
 * A decode row's attention then runs in the decode band in some piece of at most `cap` rows -- the same scalar
 * kernels, libref's bits, whatever the piece's other rows (every head group / d split sums each output in the same
 * order: r4d_ds41_sparse.hip attn_body) -- i.e. its alone bytes. The pieces are as few as greedy packing gives and as
 * even as that count allows (a 64-row segment of 16 x 4: 32 + 32, not 48 + 16).
 *
 * The decode sequences' first rows come from the host mirror of positions (each sequence's rows are consecutive
 * positions; a jump starts the next one). Where that does not give exactly n_seq_decode starts (two sequences whose
 * positions continue), the caller reads cu_seqlens back (bi_segments' rule for prefill sequences). */
#ifndef DS41_DEC_BAND_H
#define DS41_DEC_BAND_H

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

struct Ds41DecPiece {
    int64_t r0 = 0, n = 0;    /* rows [r0, r0 + n) of the step (the decode segment starts at row 0) */
    int64_t s0 = 0, s1 = 0;   /* its decode sequences [s0, s1) */
};

/* The first row of each of the step's `dn` decode sequences over its `dt` decode rows from the host positions: true
 * when the position jumps give exactly dn starts (st[0] = 0), false otherwise (st cleared). */
inline bool ds41_dec_starts(const int32_t* pos_h, int64_t dt, int64_t dn, std::vector<int64_t>& st) {
    st.clear();
    if (!pos_h || dt < 1 || dn < 1 || dn > dt) return false;
    st.push_back(0);
    for (int64_t r = 1; r < dt; ++r)
        if (pos_h[r] != pos_h[r - 1] + 1) st.push_back(r);
    if ((int64_t)st.size() != dn) {
        st.clear();
        return false;
    }
    return true;
}

/* Greedy pieces of whole sequences, each at most cap rows (a sequence taller than cap is a piece alone: returned as is,
 * the caller counts it). st: the sequences' first rows (ascending, st[0] = 0), dt: the decode rows. */
inline void ds41_dec_greedy(const std::vector<int64_t>& st, int64_t dt, int64_t cap, std::vector<Ds41DecPiece>& out) {
    out.clear();
    const int64_t dn = (int64_t)st.size();
    int64_t s = 0;
    while (s < dn) {
        const int64_t r0 = st[(size_t)s];
        int64_t e = s + 1;
        while (e < dn && (e + 1 < dn ? st[(size_t)e + 1] : dt) - r0 <= cap) ++e;
        const int64_t r1 = e < dn ? st[(size_t)e] : dt;
        out.push_back({ r0, r1 - r0, s, e });
        s = e;
    }
}

/* The pieces for a decode segment of dt rows over the sequences st: none when dt <= cap (one launch, as today); else
 * the greedy count k at cap, then the smallest cap' in [ceil(dt / k), cap] that still packs into k pieces (the even
 * split). Returns the number of pieces above cap (a single sequence taller than the band; 0 for any decode step). */
inline int64_t ds41_dec_pieces(const std::vector<int64_t>& st, int64_t dt, int64_t cap, std::vector<Ds41DecPiece>& out) {
    out.clear();
    if (dt <= cap || st.empty() || cap < 1) return 0;
    ds41_dec_greedy(st, dt, cap, out);
    const int64_t k = (int64_t)out.size();
    std::vector<Ds41DecPiece> t;
    for (int64_t c = (dt + k - 1) / k; c < cap; ++c) {
        ds41_dec_greedy(st, dt, c, t);
        if ((int64_t)t.size() <= k) {
            out.swap(t);
            break;
        }
    }
    int64_t over = 0;
    for (const Ds41DecPiece& p : out) over += p.n > cap ? 1 : 0;
    return over;
}

/* tp8 loop D22: a run of n rows in launches of at most `band` rows, in order ([0, band), [band, 2 band), ..., the last
 * shorter) -- step_bi's dense() and logits head issue a decode-only run by it (band kBiBand = 48, or
 * RADIANCE_DS41_DENSE_BAND under lever WMK). Returns the number of launches. */
inline int64_t ds41_band_cuts(int64_t n, int64_t band, std::vector<std::pair<int64_t, int64_t>>& out) {
    out.clear();
    if (n < 1 || band < 1) return 0;
    for (int64_t a = 0; a < n; a += band) out.push_back({ a, std::min(band, n - a) });
    return (int64_t)out.size();
}

/* tp8 loop D22 (DECODE round 18): the band step_bi's dense() and the GGUF logits head issue a decode-only run by -- `bi_band`
 * (kBiBand) off lever WMK (libr4d's fn4 ds41_wmk < 1), 64 under ds41_wmk 5 (libr4d serves its decode band's part-order launch
 * up to 64 rows there), else `env_band` (RADIANCE_DS41_DENSE_BAND: 49 .. 64, or kBiBand). Under WMK every dense GGUF GEMM is
 * one part order at every row count, so the band moves launches, never a row's bytes. */
inline int64_t ds41_dense_band(int wmk, int64_t env_band, int64_t bi_band) {
    if (wmk < 1) return bi_band;
    return wmk >= 5 ? 64 : env_band;   /* tp8 loop D25 step 2: ds41_wmk 6 / 7 = 5 + another 49..64-row launch code, the band 64 too */
}

/* RADIANCE_DS41_BI_DBAND=2: a recorded pass's key for its pieces (rad_arch_pass_key) -- a function of each piece's first
 * row and first sequence (with the core's key, which holds the step's rows and sequences, that is the whole issue the
 * pieces shape); never 0 (0 = not recorded). */
inline uint64_t ds41_dec_pieces_key(const std::vector<Ds41DecPiece>& pc) {
    uint64_t h = 0xf58d0a7c3b1e2d4full;
    for (const Ds41DecPiece& p : pc) {
        h ^= (uint64_t)p.r0 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= (uint64_t)p.s0 * 0xbf58476d1ce4e5b9ull + (h << 6) + (h >> 2);
    }
    return h | 2;
}

#endif
