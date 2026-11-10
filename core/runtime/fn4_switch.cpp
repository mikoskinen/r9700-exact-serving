/* fn4_switch.cpp -- see fn4_switch.h. */
#include "fn4_switch.h"
#include "../rad_internal.h"

#include <dlfcn.h>
#include <link.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace fn4 {

namespace {

const char* const kNames[kSwitches] = { "draft_adaptive", "gg_gemv", "aql_acq", "ds41_argather",
                                        "ds41_shexp_lane", "ds41_kvfold", "ds41_ofold", "ds41_onefill",
                                        "draft_depth0", "draft_probe", "draft_hybrid", "draft_hybrid_match",
                                        "draft_probe_backoff", "draft_lend", "draft_lend_borrow",
                                        "idx_split", "ds41_qnfold", "ds41_swfold", "draft_ctxskip" };
/* Switches a plugin serves (the rest are the core's own). */
bool plugin_switch(int i) {
    return i == kGgGemv || i == kDs41Argather || i == kDs41ShexpLane || i == kDs41KvFold || i == kDs41OFold ||
           i == kDs41OneFill ||
           i == kIdxSplit || i == kDs41QnFold || i == kDs41SwFold;
}

/* One loaded object's hooks; accepts and value may be absent (a plugin built before them). */
struct Hooks {
    int (*sw)(const char*, int) = nullptr;
    int (*accepts)(const char*, int) = nullptr;
    int (*value)(const char*, int*) = nullptr;
};

struct State {
    /* draft_adaptive: the scheduler's window (-1 until one registers); gg_gemv: the plugins' value as last read
     * (state_json reads it live); aql_acq: the switch once set, -1 = the environment's rule. */
    std::atomic<int> val[kSwitches];
    std::atomic<int> draft_capable{-1};      /* -1 no scheduler registered, 0 / 1 */
    std::atomic<int> draft_configured{0};    /* --draft-window adaptive at start */
    std::atomic<int> hybrid_capable{-1};     /* -1 no scheduler registered, 0 / 1 */
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> applied_step{-1};
    std::atomic<int64_t> cnt[6];   /* note_draft_counts */
    std::atomic<int> lend_capable{-1};   /* -1 no scheduler registered, 0 / 1 */
    std::atomic<int64_t> lend[17];       /* note_lend_sched [0, 9) and 16, note_lend_engine [9, 16) */
    std::mutex mu;
    int  want[kSwitches];
    bool pend[kSwitches];
    bool any_pending = false;
    /* any_pending as seen without the lock: the scheduler thread tests it every step release and
     * takes the lock only when a request queued something. */
    std::atomic<bool> queued{false};
    bool bump = false;   /* "epoch_bump": a new epoch with no switch changed (re-records like a flip) */
    /* The plugins' hooks. Looked for again while none has been found: a request may come before the
     * engine loaded its plugins. */
    bool scanned = false;
    std::vector<Hooks> hooks;
    State() {
        for (int i = 0; i < kSwitches; ++i) {
            val[i].store(-1, std::memory_order_relaxed);
            want[i] = 0;
            pend[i] = false;
        }
        for (auto& c : cnt) c.store(0, std::memory_order_relaxed);
        for (auto& c : lend) c.store(0, std::memory_order_relaxed);
    }
};

State& st() {
    static State* s = new State();
    return *s;
}

int collect(struct dl_phdr_info* info, size_t, void* data) {
    auto* paths = static_cast<std::vector<std::string>*>(data);
    if (info->dlpi_name && *info->dlpi_name) paths->emplace_back(info->dlpi_name);
    return 0;
}

template <typename F>
F sym(void* h, const char* name) {
    dlerror();
    void* f = dlsym(h, name);
    F fn = nullptr;
    if (f && !dlerror()) std::memcpy(&fn, &f, sizeof f);
    return fn;
}

/* Every loaded object that exports rad_fn4_switch: the kernel and architecture plugins are
 * dlopen'ed RTLD_LOCAL, so their symbols are reached through their own handles. Under s.mu. */
void find_hooks(State& s) {
    if (s.scanned && !s.hooks.empty()) return;
    s.scanned = true;
    std::vector<std::string> paths;
    dl_iterate_phdr(collect, &paths);
    for (const std::string& p : paths) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!h) continue;
        Hooks k;
        k.sw = sym<int (*)(const char*, int)>(h, "rad_fn4_switch");
        if (k.sw) {
            k.accepts = sym<int (*)(const char*, int)>(h, "rad_fn4_accepts");
            k.value = sym<int (*)(const char*, int*)>(h, "rad_fn4_value");
            bool dup = false;
            for (const Hooks& g : s.hooks) dup = dup || g.sw == k.sw;
            if (!dup) s.hooks.push_back(k);
        }
        dlclose(h);   /* the NOLOAD reference only; the object stays loaded */
    }
}

/* The value the plugins apply for switch i: the first that reports one, or -1 when none serves it. Under s.mu. */
int plugin_value(State& s, int i) {
    find_hooks(s);
    for (const Hooks& k : s.hooks) {
        int v = 0;
        if (k.value && k.value(kNames[i], &v) > 0) return v;
    }
    return -1;
}

int env_acq() {
    static const int env = [] {
        const char* e = std::getenv("RADIANCE_AQL_ACQ");
        return !e ? 0 : !std::strcmp(e, "rule") ? 1 : !std::strcmp(e, "rule-w") ? 2 : 0;
    }();
    return env;
}

/* The reported value of switch i. Under s.mu. */
int reported(State& s, int i) {
    if (plugin_switch(i)) return plugin_value(s, i);
    if (i == kAqlAcq) {
        const int v = s.val[i].load(std::memory_order_acquire);
        return v >= 0 ? v : env_acq();
    }
    return s.val[i].load(std::memory_order_acquire);
}

}  // namespace

bool sched_switch(int i) {
    return i == kDraftAdaptive || i == kDraftDepth0 || i == kDraftProbe || i == kDraftHybrid ||
           i == kDraftHybridMatch || i == kDraftProbeBackoff || i == kDraftLend || i == kDraftLendBorrow;
}

const char* switch_name(int i) { return i >= 0 && i < kSwitches ? kNames[i] : "?"; }

int switch_index(const std::string& name) {
    for (int i = 0; i < kSwitches; ++i)
        if (name == kNames[i]) return i;
    return -1;
}

int value(int i) {
    if (i < 0 || i >= kSwitches) return 0;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    return reported(s, i);
}
uint64_t epoch() { return st().epoch.load(std::memory_order_acquire); }

int aql_acq_rule() {
    const int v = st().val[kAqlAcq].load(std::memory_order_acquire);
    return v >= 0 ? v : env_acq();
}
bool aql_acq_none() {
    static const bool none = [] { const char* e = std::getenv("RADIANCE_AQL_ACQ"); return e && !std::strcmp(e, "none"); }();
    return none;
}

void draft_register(bool capable, bool adaptive) {
    State& s = st();
    s.draft_capable.store(capable ? 1 : 0, std::memory_order_release);
    s.draft_configured.store(adaptive ? 1 : 0, std::memory_order_release);
    s.val[kDraftAdaptive].store(adaptive ? 1 : 0, std::memory_order_release);
}

void note_draft_adaptive(bool adaptive) {
    st().val[kDraftAdaptive].store(adaptive ? 1 : 0, std::memory_order_release);
}

void hybrid_register(bool capable) {
    st().hybrid_capable.store(capable ? 1 : 0, std::memory_order_release);
}

void note_sched(int i, int value) {
    if (sched_switch(i)) st().val[i].store(value, std::memory_order_release);
}

void note_draft_counts(int64_t zero, int64_t probe, int64_t moves, int64_t hyb_lookup, int64_t hyb_head,
                       int64_t hyb_unahead) {
    State& s = st();
    const int64_t v[6] = { zero, probe, moves, hyb_lookup, hyb_head, hyb_unahead };
    for (int k = 0; k < 6; ++k) s.cnt[k].store(v[k], std::memory_order_relaxed);
}

void lend_register(bool capable) { st().lend_capable.store(capable ? 1 : 0, std::memory_order_release); }

void note_lend_sched(int64_t lent_cycles, int64_t plain_steps, int64_t plain_ms, int64_t probe_steps,
                     int64_t probe_ms, int64_t transit_steps, int64_t transit_ms, int64_t backoff_k,
                     int64_t probe_interval, int64_t borrow_cycles) {
    State& s = st();
    const int64_t v[9] = { lent_cycles, plain_steps, plain_ms, probe_steps, probe_ms, transit_steps, transit_ms,
                           backoff_k, probe_interval };
    for (int k = 0; k < 9; ++k) s.lend[k].store(v[k], std::memory_order_relaxed);
    s.lend[16].store(borrow_cycles, std::memory_order_relaxed);
}

void note_lend_engine(int64_t state, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t draft_resident,
                      int64_t draft_units, int64_t slab_mib) {
    State& s = st();
    const int64_t v[7] = { state, lends, reclaims, lent_units, draft_resident, draft_units, slab_mib };
    for (int k = 0; k < 7; ++k) s.lend[9 + k].store(v[k], std::memory_order_relaxed);
}

bool check(const std::string& name, int v, std::string* err) {
    if (name == "epoch_bump") return true;
    const int i = switch_index(name);
    if (i < 0) {
        if (err) *err = "unknown switch '" + name + "' (draft_adaptive, gg_gemv, aql_acq, ds41_argather, "
                        "ds41_shexp_lane, ds41_kvfold, ds41_ofold, ds41_onefill, draft_depth0, draft_probe, "
                        "draft_hybrid, draft_hybrid_match, draft_probe_backoff, draft_lend, draft_lend_borrow, "
                        "idx_split, ds41_qnfold, ds41_swfold, draft_ctxskip, epoch_bump)";
        return false;
    }
    State& s = st();
    if (i == kDraftProbeBackoff) {
        if (v < 0 || v > 8) {
            if (err) *err = "draft_probe_backoff takes 0 (off) .. 8 (doublings of the probe interval)";
            return false;
        }
        const int cap = s.draft_capable.load(std::memory_order_acquire);
        if (v > 0 && cap != 1) {
            if (err)
                *err = std::string("draft_probe_backoff=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no scheduler has registered a drafter"
                                : "this deployment's drafter cannot run the adaptive window");
            return false;
        }
        return true;
    }
    if (i == kDraftLendBorrow) {
        if (v < 0 || v > 1) {
            if (err) *err = "draft_lend_borrow takes 1 (on) or 0 (off)";
            return false;
        }
        const int cap = s.lend_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string("draft_lend_borrow=1 refused: ") +
                       (cap < 0 ? "the drafter's lend is not set up: RADIANCE_DRAFT_LEND was not set at start"
                                : "the drafter's lend is not set up here: no drafter expert slab to lend");
            return false;
        }
        return true;
    }
    if (i == kDraftLend) {
        if (v < -1 || v > 4096) {
            if (err) *err = "draft_lend takes 0 (off), 1 (on), 2 .. 4096 (on, after that many cycles at depth 0) or -1 "
                            "(lent at every step, an identity check)";
            return false;
        }
        const int cap = s.lend_capable.load(std::memory_order_acquire);
        if (v != 0 && cap != 1) {
            if (err)
                *err = std::string("draft_lend=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "the drafter's lend is not set up: RADIANCE_DRAFT_LEND was not set at start"
                                : "the drafter's lend is not set up here: no drafter expert slab to lend (a "
                                  "static plan, a drafter without routed experts of its own, or one that "
                                  "cannot run the adaptive window)");
            return false;
        }
        return true;
    }
    if (i == kDraftDepth0 || i == kDraftHybrid) {
        if (v < 0 || v > 1) {
            if (err) *err = std::string(kNames[i]) + " takes 1 (on) or 0 (off)";
            return false;
        }
        const int cap = (i == kDraftHybrid ? s.hybrid_capable : s.draft_capable).load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string(kNames[i]) + "=1 refused: " +
                       (cap < 0 ? std::string("no scheduler has registered a drafter")
                        : i == kDraftHybrid
                            ? std::string("this deployment has no draft head that drafts chains (a serial or block "
                                          "head, not a tree, with --num-speculative-tokens N)")
                            : std::string("this deployment's drafter cannot run the adaptive window"));
            return false;
        }
        return true;
    }
    if (i == kDraftProbe) {
        if (v < 2 || v > 4096) {
            if (err) *err = "draft_probe takes 2 .. 4096 (cycles at depth 0 between probes)";
            return false;
        }
        return true;
    }
    if (i == kDraftHybridMatch) {
        if (v < 1 || v > 256) {
            if (err) *err = "draft_hybrid_match takes 1 .. 256 (tokens the lookup's match runs back)";
            return false;
        }
        return true;
    }
    if (i == kDraftAdaptive) {
        if (v < -1 || v > 1) {
            if (err) *err = "draft_adaptive takes 1 (adaptive), 0 (fixed) or -1 (the configured window)";
            return false;
        }
        const int cap = s.draft_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = cap < 0 ? "draft_adaptive=1 refused: no scheduler has registered a drafter"
                               : "draft_adaptive=1 refused: this deployment's drafter cannot run the adaptive window "
                                 "(it needs a serial or block head drafting, --num-speculative-tokens N)";
            return false;
        }
        return true;
    }
    if (i == kAqlAcq) {
        if (v < 0 || v > 2) {
            if (err) *err = "aql_acq takes 0 (off), 1 (rule) or 2 (rule-w)";
            return false;
        }
        return true;
    }
    /* A plugin's switch: the plugin that serves it says which values it knows. */
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    int owners = 0, refused = 0, unchecked = 0;
    for (const Hooks& k : s.hooks) {
        if (!k.accepts) { ++unchecked; continue; }
        const int r = k.accepts(kNames[i], v);
        if (r > 0) ++owners;
        if (r < 0) ++refused;
    }
    if (refused > 0) {
        if (err) *err = std::string(kNames[i]) + "=" + std::to_string(v) + " refused: the plugin that serves it does not "
                        "know that value";
        return false;
    }
    if (owners == 0 && unchecked == 0) {
        if (err) *err = std::string(kNames[i]) + " refused: no loaded plugin serves it";
        return false;
    }
    return true;
}

bool request(const std::string& name, int v, std::string* err) {
    if (!check(name, v, err)) return false;
    if (name == "epoch_bump") {
        State& s = st();
        std::lock_guard<std::mutex> lk(s.mu);
        if (v) { s.bump = true; s.any_pending = true; s.queued.store(true, std::memory_order_release); }
        return true;
    }
    const int i = switch_index(name);
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    s.want[i] = v;
    s.pend[i] = true;
    s.any_pending = true;
    s.queued.store(true, std::memory_order_release);
    return true;
}

bool apply_pending(int step, int* sched) {
    if (sched)
        for (int i = 0; i < kSwitches; ++i) sched[i] = kNoChange;
    State& s = st();
    if (!s.queued.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lk(s.mu);
    s.queued.store(false, std::memory_order_relaxed);
    if (!s.any_pending) return false;
    s.any_pending = false;
    bool changed = false;
    std::string what;
    for (int i = 0; i < kSwitches; ++i) {
        if (!s.pend[i]) continue;
        s.pend[i] = false;
        int v = s.want[i];
        if (i == kDraftAdaptive) {
            if (v < 0) v = s.draft_configured.load(std::memory_order_acquire);
            if (v == 1 && s.draft_capable.load(std::memory_order_acquire) != 1) {
                RAD_WARN("fn4: draft_adaptive -> 1 at step %d refused: the drafter cannot run the adaptive window", step);
                continue;
            }
        }
        if (reported(s, i) == v) {
            if (i == kAqlAcq) s.val[i].store(v, std::memory_order_release);   /* pinned, the same rule */
            continue;
        }
        int handled = 1;
        if (plugin_switch(i)) {
            /* Every loaded object exporting the hook is asked; none taking it, or one refusing, leaves the
             * value as it was. What is reported afterwards is what the plugins say they apply. */
            find_hooks(s);
            handled = 0;
            bool refused = false;
            for (const Hooks& k : s.hooks) {
                const int r = k.sw(kNames[i], v);
                if (r > 0) ++handled;
                if (r < 0) refused = true;
            }
            if (refused || handled == 0) {
                RAD_WARN("fn4: %s -> %d at step %d refused (%d plugin(s) took it); unchanged", kNames[i], v, step,
                         handled);
                continue;
            }
            if (const int got = plugin_value(s, i); got >= 0) v = got;
        }
        if (sched_switch(i) && sched) sched[i] = v;
        s.val[i].store(v, std::memory_order_release);
        changed = true;
        what += std::string(what.empty() ? "" : ", ") + kNames[i] + "=" + std::to_string(v);
        if (handled > 1) what += " (" + std::to_string(handled) + " plugins)";
    }
    if (s.bump) {
        s.bump = false;
        changed = true;
        what += std::string(what.empty() ? "" : ", ") + "epoch_bump";
    }
    if (changed) {
        s.epoch.fetch_add(1, std::memory_order_acq_rel);
        s.applied_step.store(step, std::memory_order_release);
        RAD_INFO("fn4: switches %s applied at step %d (epoch %llu)", what.c_str(), step,
                 (unsigned long long)s.epoch.load(std::memory_order_relaxed));
    }
    return changed;
}

std::string state_json() {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    std::string o = "{\"switches\": {";
    for (int i = 0; i < kSwitches; ++i)
        o += std::string(i ? ", " : "") + "\"" + kNames[i] + "\": " + std::to_string(reported(s, i));
    o += "}, \"pending\": {";
    bool first = true;
    for (int i = 0; i < kSwitches; ++i) {
        if (!s.pend[i]) continue;
        o += std::string(first ? "" : ", ") + "\"" + kNames[i] + "\": " + std::to_string(s.want[i]);
        first = false;
    }
    if (s.bump) o += std::string(first ? "" : ", ") + "\"epoch_bump\": 1";
    o += "}, \"epoch\": " + std::to_string(s.epoch.load(std::memory_order_acquire)) +
         ", \"applied_step\": " + std::to_string(s.applied_step.load(std::memory_order_acquire)) +
         ", \"plugins\": " + std::to_string((int)s.hooks.size()) +
         ", \"draft_capable\": " + std::to_string(s.draft_capable.load(std::memory_order_acquire));
    static const char* const kCnt[6] = { "zero_cycles", "probe_cycles", "zero_moves", "hybrid_lookup", "hybrid_head",
                                         "hybrid_unahead" };
    o += ", \"draft\": {";
    for (int k = 0; k < 6; ++k)
        o += std::string(k ? ", " : "") + "\"" + kCnt[k] + "\": " +
             std::to_string((long long)s.cnt[k].load(std::memory_order_relaxed));
    o += "}";
    /* Lever DL, once a scheduler registered its lend (set up or not). */
    if (const int cap = s.lend_capable.load(std::memory_order_acquire); cap >= 0) {
        static const char* const kLend[17] = { "lent_cycles", "lent_plain_steps", "lent_plain_ms", "lent_probe_steps",
                                               "lent_probe_ms", "transit_steps", "transit_ms", "backoff_k",
                                               "probe_interval", "state", "lends", "reclaims", "lent_units",
                                               "draft_resident", "draft_units", "slab_mib", "borrow_cycles" };
        o += ", \"lend\": {\"capable\": " + std::to_string(cap);
        for (int k = 0; k < 17; ++k)
            o += std::string(", \"") + kLend[k] + "\": " +
                 std::to_string((long long)s.lend[k].load(std::memory_order_relaxed));
        o += "}";
    }
    o += "}";
    return o;
}

}  // namespace fn4
}  // namespace rad
