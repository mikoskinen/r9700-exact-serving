/* ds41_engram_dedup.h -- lever E1 (ds41-e1, 12.11.2026): an Engram layer's kv projection over a step's UNIQUE n-gram id
 * rows, then each row's kv row copied from its unique row. Pure host arithmetic (the plan, and the pass emulated for the
 * tests); deepseek41.cpp issues it (engdd_issue), tests/ds41_engdd_test.cpp checks it without a card.
 *
 * WHY EXACT. inference/model.py:353 `kv = wkv(embed(hash_ids))`: an Engram kv row is a function of the row's hashed
 * n-gram ids alone (engram.py: the ids from the compressed token ids p .. p - (ngram - 1); no activation reaches it, only
 * the gate reads h). deepseek41 follows it: ngram_ids from the token history only (step(), before the blocks: the batch's
 * tokens through token_map, each layer's kv_engram_tokens conv state), then the table rows (embed_lookup[_planes], one
 * row an id, dequantised to bf16), their copy to the card (cast bf16 -> bf16, a byte copy) and the GEMM (gemm_nt_ggml,
 * q6_K N 25600 K 6144 on DeepSeek-V4.1-Flash). So two rows of a step with equal id tuples read byte-identical table rows,
 * and get byte-identical kv rows from any GEMM form whose per-row arithmetic does not depend on the launch's other rows:
 * gemm_nt_ggml above its decode band (r4d_ds41_ggml.hip gg_gemm_run: M > kGgDecodeRows = 48 runs the WMMA kernel or lever
 * GPF's forms, whose rule is a function of (type, N, K) only, r4d_ds41_ggml_pf.hip pf_rule; "each output (row r, weight
 * row n) is the same sequence of operations ... the tile geometry, the block's thread count and the grid only change
 * which lane / block holds it"; a row's activation group sums are its own chunk's lane pair / quad; no split-K). At
 * <= 48 rows the decode kernels run (gg_gemv / lever D2's multi-row forms: another summation order).
 *
 * THE PLAN (one Engram layer of one step). The launches the undeduplicated issue makes (`Launch` {r0, n}: step() the
 * step's rows in one; step_bi a prefill segment in one, the decode segment in pieces of <= 48 rows) are the reference:
 *   - a launch of <= kFormRows rows is KEPT: its rows go to the compact rows as they are and its GEMM is issued at its own
 *     n rows (the same form as undeduplicated);
 *   - the rows of every wider launch form the POOL; the pool's distinct id rows (hashed, equal only when the whole row of
 *     H ids is equal) are its U unique rows, in the order of their first occurrence; each pooled row maps to its unique
 *     row (`map`);
 *   - the lookup and the copy run over the compact rows (the kept launches' rows, then the U unique rows: `src` names the
 *     step row each compact row's ids are taken from), the GEMM over the U unique rows into a scratch plane (one launch of
 *     U rows), and gather_rows writes every pooled row's kv row from its unique row (`runs`: the pooled rows as runs of
 *     consecutive step rows, a gather each; a byte copy);
 *   - GUARD: U <= kFormRows would take rows of launches wider than 48 into the decode kernels: not deduplicated
 *     (kGuard). Fewer than pooled / min_share_div repeated rows: not deduplicated either (kLowShare: the gather's bytes
 *     outweigh the GEMM rows saved).
 * Every pooled row's kv bytes are then its unique row's, which are the bytes the undeduplicated launch gives that row
 * (same table rows, same per-row WMMA arithmetic); every kept row is computed as undeduplicated. */
#ifndef DS41_ENGRAM_DEDUP_H
#define DS41_ENGRAM_DEDUP_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace ds41_engdd {

constexpr int64_t kFormRows = 48;     /* gemm_nt_ggml's decode band (r4d_ds41_ggml.hip kGgDecodeRows) */
constexpr int64_t kMinShareDiv = 16;  /* deduplicate when at least 1 / 16 of the pooled rows repeat */

struct Launch { int64_t r0, n; };
struct Kept { int64_t r0, n, c0; };   /* a kept launch: step rows [r0, r0 + n) at compact rows [c0, c0 + n) */
enum Outcome { kNone = 0, kGuard = 1, kLowShare = 2, kDedup = 3 };

struct Plan {
    Outcome out = kNone;
    int64_t T = 0, H = 0;
    int64_t pooled = 0;                       /* rows of launches wider than kFormRows */
    int64_t uniq = 0;                         /* the pool's distinct id rows */
    int64_t rows = 0;                         /* compact rows: the kept launches' rows, then the unique rows */
    int64_t u0 = 0;                           /* the unique rows' first compact row */
    std::vector<Kept> kept;
    std::vector<int32_t> src;                 /* [rows] the step row a compact row's ids come from */
    std::vector<int32_t> map;                 /* [T] a pooled row's unique row (0 .. uniq - 1), -1 for a kept row */
    std::vector<std::pair<int64_t, int64_t>> runs;   /* the pooled rows, runs {r0, n} of consecutive step rows */
    std::vector<int32_t> slot, first;         /* the hash table and each unique row's first step row (scratch) */
};

/* FNV-1a over the row's bytes; equality is decided by the whole row, so a collision only costs a probe. */
inline uint64_t row_hash(const int32_t* r, int64_t H) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(r);
    for (int64_t i = 0; i < H * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h ^ (h >> 29);
}

/* The plan for one Engram layer of a step: `ids` the layer's [T, H] row ids (host), `ls` the undeduplicated launches,
 * ascending and covering the rows they name once. min_share_div <= 0: deduplicate at any repeat. */
inline Outcome plan(const int32_t* ids, int64_t T, int64_t H, const Launch* ls, size_t nl, Plan& p,
                    int64_t min_share_div = kMinShareDiv) {
    p.out = kNone;
    p.T = T;
    p.H = H;
    p.pooled = p.uniq = p.rows = p.u0 = 0;
    p.kept.clear();
    p.src.clear();
    p.runs.clear();
    p.first.clear();
    p.map.assign((size_t)(T > 0 ? T : 0), -1);
    if (!ids || T <= 0 || H <= 0) return p.out;
    for (size_t i = 0; i < nl; ++i) {
        const Launch& l = ls[i];
        if (l.n <= kFormRows || l.r0 < 0 || l.r0 + l.n > T) continue;
        p.pooled += l.n;
        if (!p.runs.empty() && p.runs.back().first + p.runs.back().second == l.r0) p.runs.back().second += l.n;
        else p.runs.push_back({ l.r0, l.n });
    }
    if (p.pooled == 0) return p.out;
    size_t cap = 64;
    while (cap < 2 * (size_t)p.pooled) cap <<= 1;
    p.slot.assign(cap, -1);
    const size_t bytes = (size_t)H * sizeof(int32_t);
    for (const auto& u : p.runs)
        for (int64_t r = u.first; r < u.first + u.second; ++r) {
            const int32_t* row = ids + r * H;
            size_t i = (size_t)row_hash(row, H) & (cap - 1);
            while (p.slot[i] >= 0 && std::memcmp(ids + (int64_t)p.first[(size_t)p.slot[i]] * H, row, bytes) != 0)
                i = (i + 1) & (cap - 1);
            if (p.slot[i] < 0) {
                p.slot[i] = (int32_t)p.first.size();
                p.first.push_back((int32_t)r);
            }
            p.map[(size_t)r] = p.slot[i];
        }
    p.uniq = (int64_t)p.first.size();
    if (p.uniq <= kFormRows) return p.out = kGuard;
    const int64_t rep = p.pooled - p.uniq;
    if (rep <= 0 || (min_share_div > 0 && rep * min_share_div < p.pooled)) return p.out = kLowShare;
    int64_t c = 0;
    for (size_t i = 0; i < nl; ++i) {
        const Launch& l = ls[i];
        if (l.n > kFormRows || l.n <= 0 || l.r0 < 0 || l.r0 + l.n > T) continue;
        p.kept.push_back({ l.r0, l.n, c });
        for (int64_t r = l.r0; r < l.r0 + l.n; ++r) p.src.push_back((int32_t)r);
        c += l.n;
    }
    p.u0 = c;
    for (int32_t r : p.first) p.src.push_back(r);
    p.rows = p.u0 + p.uniq;
    return p.out = kDedup;
}

/* The compact ids: row c of `out` [rows, H] is step row src[c]'s ids. */
inline void compact_ids(const Plan& p, const int32_t* ids, int32_t* out) {
    const size_t bytes = (size_t)p.H * sizeof(int32_t);
    for (int64_t c = 0; c < p.rows; ++c) std::memcpy(out + c * p.H, ids + (int64_t)p.src[(size_t)c] * p.H, bytes);
}

/* The pass as the card runs it, on the host (the tests): `rowfn(ids_row, out_row)` is one row's lookup + copy + GEMM
 * (a function of its ids row alone, as the card's above the decode band); `kv` [T, W] the step's kv rows. Kept launches'
 * rows are computed at their compact rows and written at their step rows (their GEMM's y is the step rows), the unique
 * rows into `ku` [uniq, W], and each pooled run gathered from ku through map (gather_rows: y[i] = x[idx[i]]). */
template <typename RowFn>
inline void emulate(const Plan& p, const int32_t* ids, int64_t W, RowFn rowfn, uint16_t* kv, std::vector<int32_t>& cids,
                    std::vector<uint16_t>& ku) {
    cids.assign((size_t)(p.rows * p.H), 0);
    compact_ids(p, ids, cids.data());
    for (const Kept& k : p.kept)
        for (int64_t i = 0; i < k.n; ++i) rowfn(cids.data() + (k.c0 + i) * p.H, kv + (k.r0 + i) * W);
    ku.assign((size_t)(p.uniq * W), 0);
    for (int64_t u = 0; u < p.uniq; ++u) rowfn(cids.data() + (p.u0 + u) * p.H, ku.data() + u * W);
    for (const auto& run : p.runs)
        for (int64_t i = 0; i < run.second; ++i) {
            const int32_t s = p.map[(size_t)(run.first + i)];
            if (s >= 0 && s < p.uniq) std::memcpy(kv + (run.first + i) * W, ku.data() + (int64_t)s * W, (size_t)W * 2);
            else std::memset(kv + (run.first + i) * W, 0, (size_t)W * 2);   /* gather_rows' out-of-range row */
        }
}

}  // namespace ds41_engdd

#endif
