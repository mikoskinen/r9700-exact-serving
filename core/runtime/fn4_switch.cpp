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

const char* const kNames[kSwitches] = { "no_route", "gather_inv", "hc_fusedown", "hc_normwave",
                                         "draft_adaptive", "hc_writevec", "hc_readv", "ar_lanethreads",
                                         "hc_upmix", "ar_laneblocks", "aql_acq", "exl3_pfprep", "exl3_mapshare", "exl3_gd",
                                         "moe_pair", "exl3_dz", "exl3_bi", "exl3_dual", "hc_bi", "bi_seg", "att_bi", "arw_rows", "sched_bi_chunk", "hc_spprobe", "hc_sp", "mtp_flat" };
const char* const kEnv[kSwitches]   = { "RADIANCE_FN4_NO_ROUTE", "RADIANCE_FN4_GATHER_INV",
                                         "RADIANCE_FN4_HC_FUSEDOWN", "RADIANCE_FN4_HC_NORMWAVE",
                                         nullptr, "RADIANCE_FN4_HC_WRITEVEC", "RADIANCE_FN4_HC_READV",
                                         "RADIANCE_FN4_AR_LANE_THREADS", "RADIANCE_FN4_HC_UPMIX",
                                         "RADIANCE_FN4_AR_LANE_BLOCKS", nullptr, nullptr,
                                         "RADIANCE_FN4_X3MAP", "RADIANCE_FN4_X3GD",
                                         "RADIANCE_FN4_MOE_PAIR", "RADIANCE_FN4_X3DZ",
                                         "RADIANCE_FN4_X3BI", "RADIANCE_FN4_X3DUAL", "RADIANCE_FN4_HC_BI", "RADIANCE_FN4_BI_SEG", "RADIANCE_FN4_ATT_BI", "RADIANCE_FN4_ARW_ROWS", "RADIANCE_SCHED_BI_CHUNK", "RADIANCE_FN4_SPPROBE", "RADIANCE_FN4_HC_SP", "RADIANCE_FN4_MTP_FLAT" };   /* the arch reads these itself */

/* the value a switch starts with: unset / empty / 0 -> 0, a number > 1 as given (hc_fusedown's form,
 * hc_normwave's row floor), anything else 1 -- as the plugins read the same variables */
int env_val(const char* k) {
    const char* v = k ? std::getenv(k) : nullptr;
    if (!(v && *v && *v != '0')) return 0;
    const int n = std::atoi(v);
    return n > 1 ? n : 1;
}

struct State {
    std::atomic<int> val[kSwitches];
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> applied_step{-1};
    std::mutex mu;
    int  want[kSwitches];
    bool pend[kSwitches];
    bool any_pending = false;
    bool bump = false;   /* "epoch_bump": a new epoch with no switch changed (re-records like a flip) */
    /* The plugins' hooks, found once at the first change. */
    bool hooks_found = false;
    std::vector<int (*)(const char*, int)> hooks;
    State() {
        for (int i = 0; i < kSwitches; ++i) {
            val[i].store(i == kDraftAdaptive || i == kAqlAcq ? -1 : env_val(kEnv[i]), std::memory_order_relaxed);
            want[i] = 0;
            pend[i] = false;
        }
        if (val[kNoRoute].load(std::memory_order_relaxed))
            RAD_INFO("RADIANCE_FN4_NO_ROUTE: no routing reports (the heat engine, mover and file "
                     "stager get no histograms; for --placement all_vram)");
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

/* Every loaded object that exports rad_fn4_switch: the kernel and architecture plugins are
 * dlopen'ed RTLD_LOCAL, so their symbols are reached through their own handles. */
void find_hooks(State& s) {
    if (s.hooks_found) return;
    s.hooks_found = true;
    std::vector<std::string> paths;
    dl_iterate_phdr(collect, &paths);
    for (const std::string& p : paths) {
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!h) continue;
        dlerror();
        void* f = dlsym(h, "rad_fn4_switch");
        if (f && !dlerror()) {
            int (*fn)(const char*, int) = nullptr;
            std::memcpy(&fn, &f, sizeof f);
            bool dup = false;
            for (auto* g : s.hooks) dup = dup || g == fn;
            if (!dup) s.hooks.push_back(fn);
        }
        dlclose(h);   /* the NOLOAD reference only; the object stays loaded */
    }
}

}  // namespace

const char* switch_name(int i) { return i >= 0 && i < kSwitches ? kNames[i] : "?"; }

int switch_index(const std::string& name) {
    for (int i = 0; i < kSwitches; ++i)
        if (name == kNames[i]) return i;
    return -1;
}

int value(int i) { return i >= 0 && i < kSwitches ? st().val[i].load(std::memory_order_acquire) : 0; }
bool no_route() { return st().val[kNoRoute].load(std::memory_order_relaxed) == 1; }
uint64_t epoch() { return st().epoch.load(std::memory_order_acquire); }

int aql_acq_rule() {
    static const int env = [] {
        const char* e = std::getenv("RADIANCE_AQL_ACQ");
        return !e ? 0 : !std::strcmp(e, "rule") ? 1 : !std::strcmp(e, "rule-w") ? 2 : 0;
    }();
    const int v = st().val[kAqlAcq].load(std::memory_order_acquire);
    return v >= 0 ? v : env;
}
int sched_bi_chunk() { return st().val[kSchedBiChunk].load(std::memory_order_acquire); }
bool aql_acq_none() {
    static const bool none = [] { const char* e = std::getenv("RADIANCE_AQL_ACQ"); return e && !std::strcmp(e, "none"); }();
    return none;
}

bool request(const std::string& name, int v, std::string* err) {
    if (name == "epoch_bump") {
        State& s = st();
        std::lock_guard<std::mutex> lk(s.mu);
        if (v) { s.bump = true; s.any_pending = true; }
        return true;
    }
    const int i = switch_index(name);
    if (i < 0) {
        if (err) *err = "unknown switch '" + name + "'";
        return false;
    }
    /* hc_fusedown 0 / 1 / 2 (the tail form), hc_normwave and hc_writevec 0 / 1 / N >= 2 (from N rows),
     * the rest 0 / 1 */
    const int lo = i == kDraftAdaptive ? -1 : 0;
    const int hi = i == kHcFuseDown ? 2 : (i == kHcNormWave || i == kHcWriteVec) ? 4096
                 : i == kHcReadV ? 63 : i == kArLaneThreads ? 1024 : i == kHcUpMix ? 15 : i == kArLaneBlocks ? 24
                 : i == kAqlAcq ? 2 : i == kExl3PfPrep ? 2 : i == kMoePair ? 2 : i == kHcBi ? 3 : i == kBiSeg ? 2 : i == kArwRows ? 4096 : i == kSchedBiChunk ? 4 : 1;
    if (v < lo || v > hi) {
        if (err) *err = std::string(kNames[i]) + " takes " + std::to_string(lo) + ".." + std::to_string(hi);
        return false;
    }
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    s.want[i] = v;
    s.pend[i] = true;
    s.any_pending = true;
    return true;
}

bool apply_pending(int step, int* draft_adaptive) {
    if (draft_adaptive) *draft_adaptive = -1;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.any_pending) return false;
    s.any_pending = false;
    bool changed = false;
    std::string what;
    for (int i = 0; i < kSwitches; ++i) {
        if (!s.pend[i]) continue;
        s.pend[i] = false;
        const int v = s.want[i];
        if (s.val[i].load(std::memory_order_relaxed) == v) continue;
        int handled = 1;
        if (i == kGatherInv || i == kHcFuseDown || i == kHcNormWave || i == kHcWriteVec ||
            i == kHcReadV || i == kArLaneThreads || i == kHcUpMix || i == kArLaneBlocks || i == kExl3PfPrep ||
            i == kExl3MapShare || i == kExl3Gd || i == kMoePair || i == kExl3Dz || i == kExl3Bi || i == kExl3Dual || i == kHcBi || i == kBiSeg || i == kAttBi || i == kArwRows || i == kHcSpProbe || i == kHcSp || i == kMtpFlat) {
            find_hooks(s);
            handled = 0;
            bool refused = false;
            for (auto* fn : s.hooks) {
                const int r = fn(kNames[i], v);
                if (r > 0) ++handled;
                if (r < 0) refused = true;
            }
            if (refused || handled == 0) {
                RAD_WARN("fn4: %s -> %d at step %d refused (%d plugin(s) took it%s); unchanged",
                         kNames[i], v, step, handled,
                         i == kGatherInv ? "; gather_inv needs RADIANCE_FN4_SWITCHABLE=1 at start" : "");
                continue;
            }
        }
        if (i == kDraftAdaptive && draft_adaptive) *draft_adaptive = v;
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
        o += std::string(i ? ", " : "") + "\"" + kNames[i] + "\": " +
             std::to_string(s.val[i].load(std::memory_order_acquire));
    o += "}, \"pending\": {";
    bool first = true;
    for (int i = 0; i < kSwitches; ++i) {
        if (!s.pend[i]) continue;
        o += std::string(first ? "" : ", ") + "\"" + kNames[i] + "\": " + std::to_string(s.want[i]);
        first = false;
    }
    o += "}, \"epoch\": " + std::to_string(s.epoch.load(std::memory_order_acquire)) +
         ", \"applied_step\": " + std::to_string(s.applied_step.load(std::memory_order_acquire)) +
         ", \"plugins\": " + std::to_string(s.hooks_found ? (int)s.hooks.size() : -1) + "}";
    return o;
}

}  // namespace fn4
}  // namespace rad
