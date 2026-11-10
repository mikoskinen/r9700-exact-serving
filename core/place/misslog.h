/* misslog.h -- RADIANCE_DEBUG_MISS_LOG=FILE: for every pass, rank and routed layer, how many of the
 * rank's routed experts were live and which tier each was read from, so the routed-expert kernels'
 * time (--profile-ops) can be regressed on host-tier misses (tools/missreg.py). A measurement aid:
 * unset, nothing here runs.
 *
 * ---- where the numbers come from --------------------------------------------------------------
 *
 * The routing histograms the core already copies to a host bank once a pass (Ctx::route_flush) and
 * drains a step later for the heat engine (Engine::tiering_dispatch) -- the rows --expert-stats
 * counts. This adds no device work, no copy and no synchronisation. A layer's `live` count is the
 * rank's experts with a nonzero count in its row (rank-local experts under expert parallelism).
 *
 * Residency is the host's: each expert unit's tier as the heat engine holds it (HeatEngine::tier;
 * the plan's where the plan is static). The mover publishes tier changes only between passes and on
 * the rank thread (Ctx::run_step syncs the residency table at its top), so the tiers are snapshotted
 * on the rank thread right before each pass is issued, into a ring keyed by step, and the drain
 * classes every histogram by its own pass's snapshot: the tier the kernels of that pass read.
 *
 * The heat engine prices a histogram (the routed / stream_bytes counters behind /stats) with the
 * tiers at the drain instead, which already include the moves published since the pass; `mv`
 * counts the live experts it priced on another tier than the kernels read. A histogram whose pass
 * left no snapshot (the ring lapped it, or a pass outside the step loop) is classed by the drain's
 * tiers and tagged with how many steps those are newer (`lag`).
 *
 * Within a pass the stagers still change where a kernel reads. The prefill stager copies the
 * host-pool experts of an armed (large) trunk pass into a VRAM ring (`st` 1), and the file stager
 * reads file-tier experts into staging buffers on every pass (they count as `other`). So `zc`, the
 * live host-pool experts the kernels read zero-copy from pinned memory, is given only for passes the
 * prefill stager did not arm; a draft head's layer is never staged.
 *
 * ---- cost ------------------------------------------------------------------------------------
 *
 * Unset: one null test per pass and per drain. Set, on the rank thread: a byte per expert unit of
 * the pass's layers at each issue (43 layers x 96 experts: 4 KiB, a few microseconds) and a walk of
 * each drained row. Lines go to a per-rank buffer and a thread of the log's own writes them once a
 * second, so no rank thread does file I/O.
 *
 * ---- the file (format radiance-miss-log/1) ----------------------------------------------------
 *
 * Text. Lines starting with '#' are the header, the profile markers and the closing summary; every
 * other line is one (step, rank) and one group of layers: the trunk's layers, or the draft head's
 * (layers past the trunk). Space separated:
 *
 *   step kind rank tok seq passes st lag live vram host other zc hkib mv first:layers
 *
 *   step     RadBatch::step of the pass the histograms belong to
 *   kind     prefill | decode | verify | mixed for the trunk pass (verify: decode rows carrying
 *            drafts); draft for the draft head's layers; unknown where the ring has no record
 *   tok seq  the pass's rows and sequences; on a draft line the step's last draft pass's
 *   passes   the passes the bank holds (Ctx::last_histogram_reports). A draft line's counts are its
 *            last draft pass's, since each draft pass overwrites the head's row
 *   st       1 when the prefill stager armed the trunk pass
 *   lag      0: classed by the pass's own snapshot; k > 0: no snapshot, classed by the tiers of the
 *            drain k steps later; -1: the same, at the shutdown drain
 *   live vram host other   summed over the line's layers; live = vram + host + other. host is the
 *            pinned host pool (Tier::HostPinned); other is pageable, file and mapped tiers
 *   zc       the live host-pool experts the kernels read zero-copy: host where st is 0, else -1
 *   hkib     KiB of the live host-pool units (this rank's shard of each), the bytes zc streams
 *   mv       live experts the drain's tiers place differently from the snapshot; -1 when lag != 0
 *   first:layers   per layer from `first`, comma separated: "v", "v.h" or "v.h.o" (vram, host,
 *            other; missing trailing fields are 0), "-" where the layer has no histogram here
 *
 * A step's rows normally land together and make one trunk line and one draft line per rank; a
 * layer drained later makes another line for the same step with only that layer filled.
 *
 *   # profile K step N [final]   --profile-ops printed its K-th table set (one per rank, counters
 *                                cumulative) behind step N; `final` is the one at shutdown
 */
#pragma once
#include "../rad_core.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {

/* The environment variable, and its value or null when unset or empty. */
constexpr const char* kMissLogEnv = "RADIANCE_DEBUG_MISS_LOG";
const char* miss_log_path();

enum MlKind : uint8_t { kMlPrefill = 0, kMlDecode = 1, kMlVerify = 2, kMlMixed = 3, kMlDraft = 4,
                        kMlUnknown = 5 };
constexpr int kMlKinds = 6;
const char* ml_kind_name(int kind);
/* A trunk pass's kind from its batch shape. */
int ml_trunk_kind(int64_t n_seq, int64_t n_seq_decode, int64_t n_tok);

/* What the log counts a tier as. kMlHole is a histogram index with no unit (counted as other). */
enum MlClass : uint8_t { kMlVram = 0, kMlHost = 1, kMlOther = 2, kMlHole = 3 };
inline uint8_t ml_class(Tier t) {
    return t == Tier::VRAM ? kMlVram : t == Tier::HostPinned ? kMlHost : kMlOther;
}

/* One rank's side. Every call but take() and the summary is made on the rank's thread (or, at
 * shutdown, by the engine thread after the rank threads have joined). */
class MissRank {
public:
    /* `unit_of[l * n_expert + e]` is the unit of the rank's expert e of routed layer l, -1 where
     * none; `unit_bytes[u]` its bytes; `unit_class[u]` its tier class now, for the header. Layers
     * [n_trunk, n_layer) are a draft head's. */
    void init(int rank, int n_layer, int n_trunk, int64_t n_expert, std::vector<int32_t> unit_of,
              std::vector<int64_t> unit_bytes, const std::vector<uint8_t>& unit_class);

    int rank() const { return rank_; }
    int64_t n_expert() const { return n_expert_; }

    /* At issue, before Ctx::run_step: files the pass and snapshots the tier class of every expert
     * unit of the layers it runs (the trunk's for draft_pass 0, the head's otherwise).
     * `tier_of(unit)` returns the unit's Tier now. */
    template <class TierOf>
    void issued(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode, int64_t n_tok,
                TierOf&& tier_of) {
        uint8_t* snap = file_pass(step, draft_pass, n_seq, n_seq_decode, n_tok);
        if (!snap) return;
        for (const PosUnit& p : units_[draft_pass == 0 ? 0 : 1]) snap[p.pos] = ml_class(tier_of(p.unit));
    }
    /* After the pass is issued: whether the prefill stager armed it (trunk passes only). */
    void ran(int step, int draft_pass, bool staged);

    /* One drained histogram of `layer` from step `step`: `n` counts standing for `reports` passes.
     * `now_step` is the step whose dispatch drains it (-1 at shutdown); `tier_of` as for issued(),
     * read for the live experts only. */
    template <class TierOf>
    void credit(int layer, int step, const int32_t* counts, int64_t n, int reports, int now_step,
                TierOf&& tier_of) {
        Cell* c = nullptr;
        const uint8_t* snap = nullptr;
        if (!begin_credit(layer, step, reports, now_step, &c, &snap)) return;
        if (n > n_expert_) n = n_expert_;
        const int32_t* uo = unit_of_.data() + (int64_t)layer * n_expert_;
        for (int64_t e = 0; e < n; ++e) {
            if (counts[e] <= 0) continue;
            const int32_t u = uo[e];
            const uint8_t now = u >= 0 ? ml_class(tier_of(u)) : (uint8_t)kMlHole;
            count(c, snap ? snap[e] : now, now, u, snap != nullptr);
        }
    }
    /* After a drain: renders the lines of every step it touched into the buffer. */
    void end_drain();

    /* The writer's side: moves the rendered lines out. */
    void take(std::string* out);
    /* The closing summary: '#' lines for the file and one line for the log. */
    void summary(std::string* file_lines, std::string* log_line);

    int last_step() const { return last_step_; }

private:
    friend class MissLog;
    struct PosUnit { int32_t pos; int32_t unit; };
    struct Cell {
        int32_t v = 0, h = 0, o = 0, mv = 0;
        int64_t hbytes = 0;
        int32_t reps = 0;
        bool    seen = false;
    };
    struct Rec {
        int32_t step = -1;
        bool    trunk = false, staged = false;
        uint8_t kind = kMlUnknown;
        int64_t n_tok = 0, n_seq = 0;
        int32_t draft_passes = 0;
        int64_t draft_tok = 0, draft_seq = 0;
        bool    snap_ok[2] = { false, false };   /* trunk, head */
        int32_t lag[2] = { 0, 0 };
        bool    touched = false;
        std::vector<uint8_t> snap;               /* [n_layer * n_expert] */
        std::vector<Cell>    cell;               /* [n_layer] */
    };
    /* Deeper than the route ring's 32 banks, so a landed histogram's pass is still here. */
    static constexpr int kRing = 64;

    uint8_t* file_pass(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode, int64_t n_tok);
    /* The ring's record of `step`, reset when its slot held another step. `credit` true: a slot
     * holding a newer step is not taken from it; a record of its own (stale_) stands in. */
    Rec&     rec_for(int step, bool credit = false);
    static void reset(Rec& r, int step);
    bool     begin_credit(int layer, int step, int reports, int now_step, Cell** c,
                          const uint8_t** snap);
    void     count(Cell* c, uint8_t was, uint8_t now, int32_t unit, bool snapped) {
        const uint8_t k = was == kMlHole ? (uint8_t)kMlOther : was;
        if (k == kMlVram) ++c->v;
        else if (k == kMlHost) {
            ++c->h;
            if (unit >= 0) c->hbytes += bytes_[(size_t)unit];
        } else ++c->o;
        if (snapped && was != now) ++c->mv;
    }
    void     render(Rec& r, int group, std::string* out);

    int     rank_ = 0, n_layer_ = 0, n_trunk_ = 0;
    int64_t n_expert_ = 0;
    std::vector<int32_t> unit_of_;
    std::vector<int64_t> bytes_;
    std::vector<PosUnit> units_[2];   /* trunk, head: snapshot position and unit */
    std::vector<Rec>     ring_;
    std::vector<Rec*>    touched_;    /* records touched since the last end_drain */
    std::vector<std::unique_ptr<Rec>> stale_;   /* histograms older than their ring slot's step */
    int last_step_ = -1;

    /* Header facts: units, and their classes and bytes at init. */
    int64_t init_units_ = 0, init_cls_[3] = {}, init_bytes_ = 0;

    /* Totals for the summary, per kind. */
    struct Tot {
        uint64_t lines = 0, live = 0, v = 0, h = 0, o = 0, zc_lines = 0, zc = 0, hbytes = 0,
                 mv = 0, lag_lines = 0;
    };
    Tot      tot_[kMlKinds];
    uint64_t filed_trunk_ = 0, filed_draft_ = 0, stray_ = 0;

    std::mutex  out_mu_;
    std::string out_;
    std::string line_;                /* render's scratch */
};

/* The facts the header names. */
struct MlInfo {
    std::string model, arch;
    int tp = 1;
    int n_layer = 0, n_trunk = 0;
    int64_t n_expert = 0;
};

/* The file, one MissRank a rank, and the writer thread. */
class MissLog {
public:
    ~MissLog();
    /* Sizes the ranks; the caller init()s each before start(). */
    void      init(const std::string& path, const MlInfo& info);
    MissRank* rank(int r) { return r >= 0 && (size_t)r < ranks_.size() ? ranks_[(size_t)r].get() : nullptr; }
    int       n_rank() const { return (int)ranks_.size(); }
    const std::string& path() const { return path_; }

    /* Opens the file, writes the header and starts the writer; RAD_E_IO when the file cannot be
     * opened. `flush_ms` is the writer's interval. */
    int  start(int flush_ms = 1000);
    /* --profile-ops printed a table set behind `step`. Any thread. */
    void profile_mark(int step, bool final_);
    /* The largest step any rank filed. */
    int  last_step() const;
    /* Stops the writer, writes what is left, then `tail` and the ranks' summaries, and closes.
     * Logs one line a rank. */
    void close(const std::string& tail = std::string());

private:
    void flush_all();

    std::string path_;
    MlInfo      info_;
    std::vector<std::unique_ptr<MissRank>> ranks_;
    FILE*       f_ = nullptr;
    int         flush_ms_ = 1000;
    int         marks_ = 0;
    std::string marks_buf_;
    std::mutex  marks_mu_;
    std::mutex  file_mu_;
    std::string scratch_;

    std::thread             writer_;
    std::mutex              wmu_;
    std::condition_variable wcv_;
    bool                    wstop_ = false;
};

}  /* namespace rad */
