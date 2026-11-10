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

const char* const kNames[kSwitches] = { "draft_adaptive", "gg_gemv", "aql_acq" };
/* The variable a switch starts from; libr4d reads RADIANCE_DS41_GG_GEMV itself the same way (r4d_ds41_ggml.hip), so
 * the two agree before the first request. draft_adaptive and aql_acq start at -1 (the configuration / the env). */
const char* const kEnv[kSwitches]   = { nullptr, "RADIANCE_DS41_GG_GEMV", nullptr };
/* The values a request may set. */
const int kLo[kSwitches] = { -1, 0, 0 };
const int kHi[kSwitches] = { 1, 1, 2 };

/* the value a switch starts with: unset / empty / 0 -> 0, a number > 1 as given, anything else 1 */
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
    /* any_pending as seen without the lock: the scheduler thread tests it every step release and
     * takes the lock only when a request queued something. */
    std::atomic<bool> queued{false};
    bool bump = false;   /* "epoch_bump": a new epoch with no switch changed (re-records like a flip) */
    /* The plugins' hooks, found once at the first change a plugin handles. */
    bool hooks_found = false;
    std::vector<int (*)(const char*, int)> hooks;
    State() {
        for (int i = 0; i < kSwitches; ++i) {
            val[i].store(i == kDraftAdaptive || i == kAqlAcq ? -1 : env_val(kEnv[i]), std::memory_order_relaxed);
            want[i] = 0;
            pend[i] = false;
        }
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
uint64_t epoch() { return st().epoch.load(std::memory_order_acquire); }

int aql_acq_rule() {
    static const int env = [] {
        const char* e = std::getenv("RADIANCE_AQL_ACQ");
        return !e ? 0 : !std::strcmp(e, "rule") ? 1 : !std::strcmp(e, "rule-w") ? 2 : 0;
    }();
    const int v = st().val[kAqlAcq].load(std::memory_order_acquire);
    return v >= 0 ? v : env;
}
bool aql_acq_none() {
    static const bool none = [] { const char* e = std::getenv("RADIANCE_AQL_ACQ"); return e && !std::strcmp(e, "none"); }();
    return none;
}

bool check(const std::string& name, int v, std::string* err) {
    if (name == "epoch_bump") return true;
    const int i = switch_index(name);
    if (i < 0) {
        if (err) *err = "unknown switch '" + name + "' (draft_adaptive, gg_gemv, aql_acq, epoch_bump)";
        return false;
    }
    if (v < kLo[i] || v > kHi[i]) {
        if (err) *err = std::string(kNames[i]) + " takes " + std::to_string(kLo[i]) + ".." + std::to_string(kHi[i]);
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

bool apply_pending(int step, int* draft_adaptive) {
    if (draft_adaptive) *draft_adaptive = kNoChange;
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
        const int v = s.want[i];
        if (s.val[i].load(std::memory_order_relaxed) == v) continue;
        int handled = 1;
        if (i == kGgGemv) {
            /* A plugin's switch: every loaded object exporting the hook is asked; none taking it, or one
             * refusing, leaves the value as it was. */
            find_hooks(s);
            handled = 0;
            bool refused = false;
            for (auto* fn : s.hooks) {
                const int r = fn(kNames[i], v);
                if (r > 0) ++handled;
                if (r < 0) refused = true;
            }
            if (refused || handled == 0) {
                RAD_WARN("fn4: %s -> %d at step %d refused (%d plugin(s) took it); unchanged", kNames[i], v, step,
                         handled);
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
    if (s.bump) o += std::string(first ? "" : ", ") + "\"epoch_bump\": 1";
    o += "}, \"epoch\": " + std::to_string(s.epoch.load(std::memory_order_acquire)) +
         ", \"applied_step\": " + std::to_string(s.applied_step.load(std::memory_order_acquire)) +
         ", \"plugins\": " + std::to_string(s.hooks_found ? (int)s.hooks.size() : -1) + "}";
    return o;
}

}  // namespace fn4
}  // namespace rad
