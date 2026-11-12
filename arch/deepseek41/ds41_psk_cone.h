/* ds41_psk_cone.h -- lever PSK's row plans in deepseek41, pure host arithmetic (no RadCtx, no model), so
 * tests/ds41_cone_test.cpp checks them without a card:
 *
 *   cone / rows   the cone of a cut layer and the rows a prefill segment runs there (step_bi, rad_psk_plan);
 *   plan_end      the first layer of a planned pass at which no prefill sequence runs a row (rad_psk_plan): the
 *                 stager stages no routed layer from it on. Lever PSK-L20 (ds41-pskl20): the scan starts at the last
 *                 kv source (layer 20 of the 1017) when that layer is split -- its cone part (the query, the indexer,
 *                 the attention, the MoE, the window store) runs over C_20 = kBase + (n_layer - 1 - 20) x window rows,
 *                 its all-row part (the collapse + norm, the compressor, the latent / index-key stores) over every row,
 *                 and the all-row part touches no routed expert, so a pass with no row in layer 20's cone stages no
 *                 layer from 20 on;
 *   pfctx_runs    lever PSK-DS (ds41-pskl20): the rows of a DSpark context pass (draft_pass -1, one row a trunk row)
 *                 a drafter can still read -- each entry's last `keep` rows (the drafter window, c.window + 1 = 129:
 *                 a query reads the last `window` context keys before its anchor, the anchor at or past the entry's
 *                 end) -- as runs of consecutive rows, each run at least `min_run` rows (a run shorter is widened over
 *                 rows it skipped: a computed row is today's row), so every launch keeps the form band today's
 *                 one launch over the pass had.
 *
 * Exactness of the plans is argued at their callers (deepseek41.cpp: step_bi's PSK-L20 note, dsp_context). */
#ifndef RAD_ARCH_DS41_PSK_CONE_H
#define RAD_ARCH_DS41_PSK_CONE_H

#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

namespace ds41cone {

/* Each cut prefill segment runs at least this many rows (> 120: above every row-count form switch of the trunk). */
constexpr int64_t kBase = 160;
/* A prefill sequence the scheduler did not plan. */
constexpr int32_t kNoHint = INT32_MIN;

/* Layer l's cone: the rows its chunk's end (or a target) reads back through layers l .. n_layer - 1. */
inline int64_t cone(int64_t n_layer, int64_t window, int64_t l) { return kBase + (n_layer - 1 - l) * window; }

/* The rows a prefill segment of `n` rows runs at a layer of cone C, its next target `ahead` rows past its end: none
 * when the target's cone does not reach the chunk, else its last max(C - ahead, kBase) rows, at most n. */
inline int64_t rows(int64_t C, int32_t ahead, int64_t n) {
    if (ahead == kNoHint) return n;
    if ((int64_t)ahead >= C) return 0;
    return std::min(n, std::max(C - (int64_t)ahead, kBase));
}

/* The first layer in [from, n_layer) at which none of the n planned prefill sequences runs a row (`ahead[i]` already
 * resolved: a plan's value, or the unplanned default), n_layer when every layer runs one. The cones shrink with l, so
 * every layer after it runs no row either. */
inline int plan_end(int n_layer, int window, int from, int n, const int32_t* ahead, const int32_t* rws) {
    for (int l = from; l < n_layer; ++l) {
        bool any = false;
        for (int i = 0; i < n && !any; ++i) any = rows(cone(n_layer, window, l), ahead[i], rws[i]) > 0;
        if (!any) return l;
    }
    return n_layer;
}

/* Lever PSK-DS: the runs [r0, r0 + n) of a context pass of T rows and S entries (one entry a trunk sequence, in the
 * trunk's order, each at consecutive positions) that a drafter keeping `keep` positions can still read. The entries are
 * found from the host positions: an entry starts where a row's position is not its predecessor's + 1, and the count
 * must be S (two entries whose positions happen to continue each other, or an entry whose rows are not consecutive --
 * a tree's -- cannot be told apart: no plan). Each entry keeps its last min(rows, keep) rows; a run shorter than
 * `min_run` is widened forward over skipped rows (merging with the next run it reaches), at the pass's end backward.
 * Returns false (run every row, the pass as it is) when nothing would be skipped, there is no host copy of the
 * positions, or the entries cannot be told apart; else true, `runs` sorted and disjoint, every run >= min_run rows (or
 * the whole pass), and `*run_rows` their total (< T). */
inline bool pfctx_runs(const int32_t* pos_h, int64_t T, int64_t S, int64_t keep, int64_t min_run,
                       std::vector<std::pair<int64_t, int64_t>>* runs, int64_t* run_rows) {
    runs->clear();
    if (run_rows) *run_rows = T;
    if (!pos_h || T <= 0 || S <= 0 || keep <= 0 || T <= keep || T <= min_run) return false;
    /* the entries' first rows */
    std::vector<int64_t> st;
    st.reserve((size_t)S + 1);
    st.push_back(0);
    for (int64_t r = 1; r < T; ++r)
        if (pos_h[r] != pos_h[r - 1] + 1) {
            if ((int64_t)st.size() >= S) return false;   /* more breaks than entries: not consecutive within one */
            st.push_back(r);
        }
    if ((int64_t)st.size() != S) return false;
    st.push_back(T);
    /* each entry's kept tail, merged where they touch */
    std::vector<std::pair<int64_t, int64_t>> kp;   /* [a, b) */
    bool skips = false;
    for (int64_t i = 0; i < S; ++i) {
        const int64_t a = st[(size_t)i], b = st[(size_t)i + 1];
        const int64_t a2 = std::max(a, b - keep);
        skips = skips || a2 > a;
        if (!kp.empty() && kp.back().second == a2) kp.back().second = b;
        else kp.push_back({ a2, b });
    }
    if (!skips) return false;
    /* widen the short runs: forward over skipped rows (taking in every run reached), at the pass's end backward */
    std::vector<std::pair<int64_t, int64_t>> out;
    for (size_t i = 0; i < kp.size(); ++i) {
        int64_t a = kp[i].first, b = kp[i].second;
        if (!out.empty() && out.back().second >= a) {   /* reached by the run before */
            a = out.back().first;
            b = std::max(b, out.back().second);
            out.pop_back();
        }
        if (b - a < min_run) {
            b = std::min(T, a + min_run);
            while (i + 1 < kp.size() && kp[i + 1].first <= b) b = std::max(b, kp[++i].second);
            if (b - a < min_run) {
                a = std::max<int64_t>(0, b - min_run);
                while (!out.empty() && out.back().second >= a) {
                    a = std::min(a, out.back().first);
                    out.pop_back();
                }
            }
        }
        out.push_back({ a, b });
    }
    int64_t tot = 0;
    for (const auto& u : out) tot += u.second - u.first;
    if (tot >= T) return false;
    for (const auto& u : out) runs->push_back({ u.first, u.second - u.first });
    if (run_rows) *run_rows = tot;
    return true;
}

}  // namespace ds41cone

#endif
