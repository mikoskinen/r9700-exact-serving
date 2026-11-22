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
 * The balanced deal (`balance`, deepseek41's idx_split switch at 1). Where world does not divide n,
 * the deal above leaves ranks short: a mixed step of 2048 prefill rows beside decode rows is 9 runs,
 * dealt 3 / 3 / 3 / 0 at four ranks, so every rank waits at the gather for 3 R = 3/4 of the step's
 * rows on three of them (the max rank 33 % over the mean). Balanced, every rank takes an equal block
 * of the step's rows: B = ceil(T / world) rows a rank in q = ceil(B / irows) runs of R = ceil(B / q),
 * n = ceil(T / R) runs in all (world q but for a last rank the step ends inside), and the same
 * [p0, p1) = [r q, min(n, r q + q)) and the same gather (rank r's q R rows at r q R). Only the run
 * height and the rank that scores a row change; a row's arithmetic does not:
 *   - index_score computes each row alone (libr4d's WMMA form: a wave a row, the dot over its own q
 *     and the keys, the same instructions at any pass height; libref / libavx: a loop a row), so a
 *     row's bits depend on the kernel form only, which the kernels pick by the pass height: the
 *     balanced R is kept on the same side of kDs41IdxFormRows as the unbalanced one (else the step
 *     takes the unbalanced deal), and at deepseek41's irows >= 256 both are over 64 rows anyway;
 *   - index_candidates and index_topk are exact selections a row (ties to the lower block / position,
 *     the -1 fill), the same set whatever rows share a launch;
 *   - the gather lays rank r's part at r q R and rank r scored rows [r q R, r q R + q R): row t of the
 *     step lands at row t of sel, as before; rows past T are written by nothing that is read.
 * Where world divides n the balanced deal is the deal above exactly (q = n / world, R = ceil(T / n)),
 * so it is taken only where it differs. It needs n <= max_runs, the run ops declared (row_part 0 ..
 * max_runs - 1; 0 = no limit): a step past it (17 runs at 4160 rows, balanced 20) takes the deal
 * above. q <= ceil(max_tok / (world irows)) = qmax and R <= irows, so the part, sel and the score
 * plane hold it as declared.
 *
 * Every rank computes the same deal from the same T, so they issue the same collective. The
 * candidate layer and every layer that reads its blocks share a ratio (deepseek41 refuses
 * otherwise), so they share `live` and make the same split decision: a rank's keep rows are
 * always the ones it computed (`balance` is one value for a whole step: the switch moves only
 * between steps). */
#ifndef DS41_INDEX_SPLIT_H
#define DS41_INDEX_SPLIT_H

#include <cstdint>

struct Ds41IdxRuns {
    int64_t n = 0, R = 0;            /* n runs of R rows */
    bool    split = false;           /* dealt over the ranks */
    int64_t q = 0, p0 = 0, p1 = 0;   /* runs a rank (the gather's q R rows), this rank's [p0, p1) */
    bool    balanced = false;        /* the balanced deal, where it differs from the plain one */
};

/* The fewest positions a row may pass at a layer for the split to pay for its gather: below it a
 * layer's whole indexer is a few microseconds a run and the collective's latency (tens of
 * microseconds) would cost more than the three quarters it saves. */
static constexpr int64_t kDs41SplitMinLive = 4096;

/* The tallest pass an index_score kernel runs in its decode form (libr4d's scalar kernel, libref's
 * bits; taller passes take the WMMA form, another dot order: libr4d kWmRows). The balanced deal
 * never moves a run's height across it. */
static constexpr int64_t kDs41IdxFormRows = 48;

/* T rows, planes of irows rows, `live` positions at this layer, the attention world and this
 * rank's index in it; world 1 (or a split not asked for) is the unsplit deal. min_live: the split's
 * threshold (deepseek41 passes RADIANCE_DS41_SPLIT_MIN_LIVE when set). balance: the balanced deal
 * (above) within max_runs runs (0: no limit). Every rank reads the same values, so every rank makes
 * the same decision. */
inline Ds41IdxRuns ds41_idx_runs(int64_t T, int64_t irows, int64_t live, int world, int rank,
                                 bool want_split, int64_t min_live = kDs41SplitMinLive,
                                 bool balance = false, int64_t max_runs = 0) {
    Ds41IdxRuns r;
    if (T < 1 || irows < 1) return r;
    r.n = (T + irows - 1) / irows;
    r.R = (T + r.n - 1) / r.n;
    r.split = want_split && world > 1 && r.n >= 2 && live >= min_live;
    if (r.split) {
        r.q = (r.n + world - 1) / world;
        if (balance && r.n % world != 0) {
            const int64_t B = (T + world - 1) / world;
            const int64_t q = (B + irows - 1) / irows;
            const int64_t R = (B + q - 1) / q;
            const int64_t n = (T + R - 1) / R;
            if ((max_runs < 1 || n <= max_runs) && (R > kDs41IdxFormRows) == (r.R > kDs41IdxFormRows)) {
                r.n = n;
                r.R = R;
                r.q = q;
                r.balanced = true;
            }
        }
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
