/* misslog.cpp -- RADIANCE_DEBUG_MISS_LOG: the per-rank tally, its lines and the file. See
 * misslog.h. */
#include "misslog.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

namespace rad {

const char* miss_log_path() {
    const char* v = std::getenv(kMissLogEnv);
    return v && *v ? v : nullptr;
}

const char* ml_kind_name(int kind) {
    switch (kind) {
        case kMlPrefill: return "prefill";
        case kMlDecode:  return "decode";
        case kMlVerify:  return "verify";
        case kMlMixed:   return "mixed";
        case kMlDraft:   return "draft";
        default:         return "unknown";
    }
}

int ml_trunk_kind(int64_t n_seq, int64_t n_seq_decode, int64_t n_tok) {
    if (n_seq_decode <= 0) return kMlPrefill;
    if (n_seq_decode < n_seq) return kMlMixed;
    /* Every row a decode row: more rows than sequences means drafts to verify. */
    return n_tok > n_seq ? kMlVerify : kMlDecode;
}

/* ------------------------------------------------------------------ one rank */

void MissRank::init(int rank, int n_layer, int n_trunk, int64_t n_expert,
                    std::vector<int32_t> unit_of, std::vector<int64_t> unit_bytes,
                    const std::vector<uint8_t>& unit_class) {
    rank_ = rank;
    n_layer_ = std::max(n_layer, 0);
    n_trunk_ = std::max(0, std::min(n_trunk, n_layer_));
    n_expert_ = std::max<int64_t>(n_expert, 0);
    const size_t plane = (size_t)n_layer_ * (size_t)n_expert_;
    unit_of_ = std::move(unit_of);
    unit_of_.resize(plane, -1);
    bytes_ = std::move(unit_bytes);
    for (int g = 0; g < 2; ++g) units_[g].clear();
    init_units_ = init_bytes_ = 0;
    for (int64_t& c : init_cls_) c = 0;
    for (size_t pos = 0; pos < plane; ++pos) {
        const int32_t u = unit_of_[pos];
        if (u < 0 || (size_t)u >= bytes_.size()) {
            unit_of_[pos] = -1;
            continue;
        }
        const int layer = (int)(pos / (size_t)n_expert_);
        units_[layer < n_trunk_ ? 0 : 1].push_back({ (int32_t)pos, u });
        ++init_units_;
        init_bytes_ += bytes_[(size_t)u];
        const uint8_t k = (size_t)u < unit_class.size() ? unit_class[(size_t)u] : (uint8_t)kMlOther;
        ++init_cls_[k <= kMlOther ? k : kMlOther];
    }
    ring_.assign((size_t)kRing, Rec{});
    for (Rec& r : ring_) {
        r.snap.assign(plane, (uint8_t)kMlHole);
        r.cell.assign((size_t)n_layer_, Cell{});
    }
    touched_.clear();
    stale_.clear();
    last_step_ = -1;
    for (Tot& t : tot_) t = Tot{};
    filed_trunk_ = filed_draft_ = stray_ = 0;
}

void MissRank::reset(Rec& r, int step) {
    r.step = step;
    r.trunk = r.staged = false;
    r.kind = kMlUnknown;
    r.n_tok = r.n_seq = 0;
    r.draft_passes = 0;
    r.draft_tok = r.draft_seq = 0;
    r.snap_ok[0] = r.snap_ok[1] = false;
    r.lag[0] = r.lag[1] = 0;
    r.touched = false;
    std::fill(r.cell.begin(), r.cell.end(), Cell{});
}

MissRank::Rec& MissRank::rec_for(int step, bool credit) {
    Rec& r = ring_[(size_t)(step % kRing)];
    if (r.step == step) return r;
    if (credit && r.step > step) {
        /* A histogram older than the pass now in its slot (one the ring lapped, or a pass from
         * before the step loop with a low step number): never evict the newer pass's record. */
        for (const auto& s : stale_)
            if (s->step == step) return *s;
        stale_.push_back(std::make_unique<Rec>());
        Rec& s = *stale_.back();
        s.cell.assign((size_t)n_layer_, Cell{});
        reset(s, step);
        return s;
    }
    /* The slot's last step was rendered at its drain; whatever never drained is lost here (its
     * bank was recycled too, which Ctx::route_dropped counts). */
    reset(r, step);
    return r;
}

uint8_t* MissRank::file_pass(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode,
                             int64_t n_tok) {
    if (step < 0 || ring_.empty()) return nullptr;
    Rec& r = rec_for(step);
    last_step_ = std::max(last_step_, step);
    if (draft_pass == 0) {
        r.trunk = true;
        r.kind = (uint8_t)ml_trunk_kind(n_seq, n_seq_decode, n_tok);
        r.n_tok = n_tok;
        r.n_seq = n_seq;
        r.staged = false;
        r.snap_ok[0] = true;
        ++filed_trunk_;
    } else {
        /* Each draft pass overwrites the head's row in the step's bank, and its snapshot here. */
        ++r.draft_passes;
        r.draft_tok = n_tok;
        r.draft_seq = n_seq;
        r.snap_ok[1] = true;
        ++filed_draft_;
    }
    return r.snap.data();
}

void MissRank::ran(int step, int draft_pass, bool staged) {
    if (step < 0 || draft_pass != 0 || ring_.empty()) return;
    Rec& r = ring_[(size_t)(step % kRing)];
    if (r.step == step) r.staged = staged;
}

bool MissRank::begin_credit(int layer, int step, int reports, int now_step, Cell** c,
                            const uint8_t** snap) {
    if (layer < 0 || layer >= n_layer_ || step < 0 || ring_.empty()) return false;
    Rec& r = rec_for(step, true);
    const int g = layer < n_trunk_ ? 0 : 1;
    if (g == 0 && !r.trunk) ++stray_;
    Cell& cell = r.cell[(size_t)layer];
    /* A layer drained twice for one step (a second drain after a render) starts afresh. */
    cell = Cell{};
    cell.seen = true;
    cell.reps = std::max(reports, 1);
    if (r.snap_ok[g]) {
        *snap = r.snap.data() + (int64_t)layer * n_expert_;
    } else {
        *snap = nullptr;
        const int32_t lag = now_step < 0 ? -1 : std::max(1, now_step - step);
        if (r.lag[g] == 0 || (lag > r.lag[g] && r.lag[g] > 0)) r.lag[g] = lag;
    }
    if (!r.touched) {
        r.touched = true;
        touched_.push_back(&r);
    }
    *c = &cell;
    return true;
}

void MissRank::render(Rec& r, int group, std::string* out) {
    const int lo = group == 0 ? 0 : n_trunk_, hi = group == 0 ? n_trunk_ : n_layer_;
    int first = -1, last = -1, reps = 0;
    int64_t v = 0, h = 0, o = 0, mv = 0, hb = 0;
    for (int l = lo; l < hi; ++l) {
        const Cell& c = r.cell[(size_t)l];
        if (!c.seen) continue;
        if (first < 0) first = l;
        last = l;
        v += c.v;
        h += c.h;
        o += c.o;
        mv += c.mv;
        hb += c.hbytes;
        reps = std::max(reps, c.reps);
    }
    if (first < 0) return;

    const int kind = group == 1 ? (int)kMlDraft : r.trunk ? (int)r.kind : (int)kMlUnknown;
    const int64_t tok = group == 1 ? r.draft_tok : r.n_tok;
    const int64_t seq = group == 1 ? r.draft_seq : r.n_seq;
    const bool staged = group == 0 && r.staged;
    const bool snapped = r.snap_ok[group];
    const int32_t lag = snapped ? 0 : r.lag[group];
    const int64_t zc = staged ? -1 : h;
    const int64_t hkib = (hb + 1023) / 1024;
    const int64_t mvo = snapped ? mv : -1;

    char head[256];
    const int n = std::snprintf(head, sizeof head,
                                "%d %s %d %lld %lld %d %d %d %lld %lld %lld %lld %lld %lld %lld %d:",
                                r.step, ml_kind_name(kind), rank_, (long long)tok, (long long)seq,
                                reps, staged ? 1 : 0, (int)lag, (long long)(v + h + o), (long long)v,
                                (long long)h, (long long)o, (long long)zc, (long long)hkib,
                                (long long)mvo, first);
    out->append(head, (size_t)std::max(n, 0));
    char cb[48];
    for (int l = first; l <= last; ++l) {
        const Cell& c = r.cell[(size_t)l];
        if (l > first) out->push_back(',');
        if (!c.seen) {
            out->push_back('-');
            continue;
        }
        const int m = c.o ? std::snprintf(cb, sizeof cb, "%d.%d.%d", c.v, c.h, c.o)
                    : c.h ? std::snprintf(cb, sizeof cb, "%d.%d", c.v, c.h)
                          : std::snprintf(cb, sizeof cb, "%d", c.v);
        out->append(cb, (size_t)std::max(m, 0));
    }
    out->push_back('\n');

    Tot& t = tot_[kind];
    ++t.lines;
    t.live += (uint64_t)(v + h + o);
    t.v += (uint64_t)v;
    t.h += (uint64_t)h;
    t.o += (uint64_t)o;
    if (!staged) {
        ++t.zc_lines;
        t.zc += (uint64_t)h;
        t.hbytes += (uint64_t)hb;
    }
    if (snapped) t.mv += (uint64_t)mv;
    else ++t.lag_lines;

    /* Rendered: a later drain of one of these layers makes a line of its own. */
    for (int l = first; l <= last; ++l) r.cell[(size_t)l] = Cell{};
}

void MissRank::end_drain() {
    if (touched_.empty()) return;
    std::sort(touched_.begin(), touched_.end(),
              [](const Rec* a, const Rec* b) { return a->step < b->step; });
    line_.clear();
    for (Rec* r : touched_) {
        if (!r->touched) continue;
        render(*r, 0, &line_);
        render(*r, 1, &line_);
        r->touched = false;
    }
    touched_.clear();
    stale_.clear();
    if (line_.empty()) return;
    std::lock_guard<std::mutex> lk(out_mu_);
    out_ += line_;
}

void MissRank::take(std::string* out) {
    std::lock_guard<std::mutex> lk(out_mu_);
    out->append(out_);
    out_.clear();
}

void MissRank::summary(std::string* file_lines, std::string* log_line) {
    char b[512];
    for (int k = 0; k < kMlKinds; ++k) {
        const Tot& t = tot_[k];
        if (!t.lines) continue;
        std::snprintf(b, sizeof b,
                      "# summary rank %d kind %s lines %llu live %llu vram %llu host %llu other %llu "
                      "zc_lines %llu zc %llu hkib %llu mv %llu lag_lines %llu\n",
                      rank_, ml_kind_name(k), (unsigned long long)t.lines,
                      (unsigned long long)t.live, (unsigned long long)t.v,
                      (unsigned long long)t.h, (unsigned long long)t.o,
                      (unsigned long long)t.zc_lines, (unsigned long long)t.zc,
                      (unsigned long long)((t.hbytes + 1023) / 1024), (unsigned long long)t.mv,
                      (unsigned long long)t.lag_lines);
        *file_lines += b;
    }
    std::snprintf(b, sizeof b, "# filed rank %d trunk_passes %llu draft_passes %llu "
                               "histograms_without_trunk_pass %llu\n",
                  rank_, (unsigned long long)filed_trunk_, (unsigned long long)filed_draft_,
                  (unsigned long long)stray_);
    *file_lines += b;

    /* The log line: decode and verify steps together, the case the log exists for. */
    Tot d;
    for (int k : { (int)kMlDecode, (int)kMlVerify }) {
        const Tot& t = tot_[k];
        d.lines += t.lines;
        d.live += t.live;
        d.v += t.v;
        d.h += t.h;
        d.o += t.o;
        d.zc_lines += t.zc_lines;
        d.zc += t.zc;
        d.hbytes += t.hbytes;
        d.mv += t.mv;
        d.lag_lines += t.lag_lines;
    }
    if (!d.lines) {
        std::snprintf(b, sizeof b, "rank %d: no decode or verify step drained", rank_);
    } else {
        const double n = (double)d.lines, nz = d.zc_lines ? (double)d.zc_lines : 1.0;
        std::snprintf(b, sizeof b,
                      "rank %d: %llu decode/verify steps, a step %.1f live experts: %.1f vram, "
                      "%.2f host, %.2f other (hit %.2f %%); %.2f read zero-copy from the pinned "
                      "pool, %.2f MiB; %.2f a step priced on a moved tier by the drain; %llu "
                      "line(s) without a snapshot",
                      rank_, (unsigned long long)d.lines, (double)d.live / n, (double)d.v / n,
                      (double)d.h / n, (double)d.o / n,
                      d.live ? 100.0 * (double)d.v / (double)d.live : 100.0,
                      (double)d.zc / nz, (double)d.hbytes / nz / 1048576.0, (double)d.mv / n,
                      (unsigned long long)d.lag_lines);
    }
    *log_line = b;
}

/* ------------------------------------------------------------------ the file */

void MissLog::init(const std::string& path, const MlInfo& info) {
    path_ = path;
    info_ = info;
    ranks_.clear();
    for (int r = 0; r < std::max(info.tp, 1); ++r) ranks_.push_back(std::make_unique<MissRank>());
}

int MissLog::start(int flush_ms) {
    if (f_ || path_.empty()) return RAD_E_STATE;
    f_ = std::fopen(path_.c_str(), "w");
    if (!f_) {
        RAD_WARN("%s: cannot write %s: %s", kMissLogEnv, path_.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    flush_ms_ = flush_ms > 0 ? flush_ms : 1000;
    std::string h;
    char b[512];
    /* One word each, so the header splits on spaces. */
    const auto word = [](std::string s) {
        for (char& c : s) if (c == ' ' || c == '\t' || c == '\n') c = '_';
        return s.empty() ? std::string("-") : s;
    };
    const std::string model = word(info_.model), arch = word(info_.arch);
    h += "# radiance-miss-log/1\n";
    std::snprintf(b, sizeof b,
                  "# model %s arch %s tp %d routed_layers %d trunk_layers %d experts %lld "
                  "started_unix %lld pid %d\n",
                  model.c_str(), arch.c_str(), info_.tp, info_.n_layer,
                  info_.n_trunk, (long long)info_.n_expert, (long long)std::time(nullptr),
                  (int)getpid());
    h += b;
    h += "# fields: step kind rank tok seq passes st lag live vram host other zc hkib mv "
         "first:layers (v[.h[.o]] per layer, - none; see core/place/misslog.h)\n";
    for (const auto& r : ranks_) {
        std::snprintf(b, sizeof b, "# rank %d expert_units %lld at_start vram %lld host %lld "
                                   "other %lld unit_bytes_mean %lld\n",
                      r->rank_, (long long)r->init_units_, (long long)r->init_cls_[0],
                      (long long)r->init_cls_[1], (long long)r->init_cls_[2],
                      (long long)(r->init_units_ ? r->init_bytes_ / r->init_units_ : 0));
        h += b;
    }
    if (std::fwrite(h.data(), 1, h.size(), f_) != h.size() || std::fflush(f_) != 0) {
        RAD_WARN("%s: writing %s failed: %s", kMissLogEnv, path_.c_str(), std::strerror(errno));
        std::fclose(f_);
        f_ = nullptr;
        return RAD_E_IO;
    }
    {
        std::lock_guard<std::mutex> lk(wmu_);
        wstop_ = false;
    }
    writer_ = std::thread([this] {
        const auto every = std::chrono::milliseconds(flush_ms_);
        for (;;) {
            std::unique_lock<std::mutex> lk(wmu_);
            if (wcv_.wait_for(lk, every, [this] { return wstop_; })) break;
            lk.unlock();
            flush_all();
        }
    });
    return RAD_OK;
}

void MissLog::flush_all() {
    std::lock_guard<std::mutex> lk(file_mu_);
    if (!f_) return;
    scratch_.clear();
    {
        std::lock_guard<std::mutex> mk(marks_mu_);
        scratch_.swap(marks_buf_);
    }
    for (const auto& r : ranks_) r->take(&scratch_);
    if (scratch_.empty()) return;
    if (std::fwrite(scratch_.data(), 1, scratch_.size(), f_) != scratch_.size() ||
        std::fflush(f_) != 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            RAD_WARN("%s: writing %s failed: %s", kMissLogEnv, path_.c_str(), std::strerror(errno));
        }
    }
    /* Keep the buffer's capacity small after a burst (a long prefill's lines). */
    if (scratch_.capacity() > (1u << 22)) std::string().swap(scratch_);
}

void MissLog::profile_mark(int step, bool final_) {
    char b[96];
    std::lock_guard<std::mutex> lk(marks_mu_);
    ++marks_;
    std::snprintf(b, sizeof b, "# profile %d step %d%s\n", marks_, step, final_ ? " final" : "");
    marks_buf_ += b;
}

int MissLog::last_step() const {
    int s = -1;
    for (const auto& r : ranks_) s = std::max(s, r->last_step());
    return s;
}

void MissLog::close(const std::string& tail) {
    {
        std::lock_guard<std::mutex> lk(wmu_);
        wstop_ = true;
    }
    wcv_.notify_all();
    if (writer_.joinable()) writer_.join();
    if (!f_) return;
    flush_all();
    std::string s = tail, log_line;
    for (const auto& r : ranks_) {
        r->summary(&s, &log_line);
        RAD_INFO("miss log: %s", log_line.c_str());
    }
    std::lock_guard<std::mutex> lk(file_mu_);
    if (!s.empty()) (void)std::fwrite(s.data(), 1, s.size(), f_);
    if (std::fclose(f_) != 0)
        RAD_WARN("%s: closing %s failed: %s", kMissLogEnv, path_.c_str(), std::strerror(errno));
    else
        RAD_INFO("miss log: written to %s", path_.c_str());
    f_ = nullptr;
}

MissLog::~MissLog() { close(); }

}  /* namespace rad */
