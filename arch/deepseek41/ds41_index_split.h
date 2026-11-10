/* ds41_index_split.h -- how deepseek41 scores a step's indexer rows in runs (index_runs() in
 * deepseek41.cpp) and, under tensor parallelism, deals those runs over the attention ranks (the
 * indexer's row split). Header-only so tests/r4d_emu/test_ds41idx.cpp checks the same arithmetic
 * the plugin issues by.
 *
 * A step of T rows is n = ceil(T / irows) runs of R = ceil(T / n) rows (the last may be shorter),
 * run p scoring step rows [p R, p R + R) into one R-row plane: exactly the unsplit op's rows, so
 * every run is the op the unsplit step issues (same parameters, same pass height, same kernel
 * form). The split hands rank r the runs [p0, p1) = [r q, min(n, r q + q)), q = ceil(n / world):
 * the rank scores, selects and writes their rows into a q R-row part, and an all_gather (plain
 * concatenation, rank r's part at r q R rows) lays every rank's part where the unsplit selection
 * puts it -- row p R + i of run p lands at row r q R + (p - p0) R + i = p R + i. A rank with no
 * run, or a short last run, contributes rows that land at or past T, which nothing reads.
 *
 * Every rank computes the same deal from the same T, so they issue the same collective. The
 * candidate layer and every layer that reads its blocks share a ratio (deepseek41 refuses
 * otherwise), so they share `live` and make the same split decision: a rank's keep rows are
 * always the ones it computed. */
#ifndef DS41_INDEX_SPLIT_H
#define DS41_INDEX_SPLIT_H

#include <cstdint>

struct Ds41IdxRuns {
    int64_t n = 0, R = 0;            /* n runs of R rows */
    bool    split = false;           /* dealt over the ranks */
    int64_t q = 0, p0 = 0, p1 = 0;   /* runs a rank (the gather's q R rows), this rank's [p0, p1) */
};

/* The fewest positions a row may pass at a layer for the split to pay for its gather: below it a
 * layer's whole indexer is a few microseconds a run and the collective's latency (tens of
 * microseconds) would cost more than the three quarters it saves. */
static constexpr int64_t kDs41SplitMinLive = 4096;

/* T rows, planes of irows rows, `live` positions at this layer, the attention world and this
 * rank's index in it; world 1 (or a split not asked for) is the unsplit deal. min_live: the split's
 * threshold (deepseek41 passes RADIANCE_DS41_SPLIT_MIN_LIVE when set). Every rank reads the same
 * value, so every rank makes the same decision. */
inline Ds41IdxRuns ds41_idx_runs(int64_t T, int64_t irows, int64_t live, int world, int rank,
                                 bool want_split, int64_t min_live = kDs41SplitMinLive) {
    Ds41IdxRuns r;
    if (T < 1 || irows < 1) return r;
    r.n = (T + irows - 1) / irows;
    r.R = (T + r.n - 1) / r.n;
    r.split = want_split && world > 1 && r.n >= 2 && live >= min_live;
    if (r.split) {
        r.q = (r.n + world - 1) / world;
        r.p0 = (int64_t)rank * r.q < r.n ? (int64_t)rank * r.q : r.n;
        r.p1 = r.p0 + r.q < r.n ? r.p0 + r.q : r.n;
    } else {
        r.q = r.n;
        r.p0 = 0;
        r.p1 = r.n;
    }
    return r;
}

#endif
