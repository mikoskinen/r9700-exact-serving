/* fn4_switch.cpp -- see fn4_switch.h. */
#include "fn4_switch.h"
#include "../rad_internal.h"

#include <dlfcn.h>
#include <link.h>

#include <atomic>
#include <cctype>
#include <cstdio>
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
                                        "drafter_mx", "drafter_bf",
                                        "idx_split", "ds41_qnfold", "ds41_swfold", "draft_ctxskip",
                                        "dspark_conf", "dspark_alt", "dspark_conf_rows", "gg_pf", "f32_pf",
                                        "dspark_walk", "ds41_bi", "draft_sample", "draft_lend_hyst", "arena_d0",
                                        "draft_lend_w", "draft_occ", "draft_occ_k", "place_lf", "sched_bi_chunk",
                                        "sched_interleave", "sched_mixed_tok", "sched_short_bypass",
                                        "sched_bypass_total", "ds41_psk", "ds41_psk_k", "ds41_psk_l20",
                                        "dspark_pfctx", "ds41_psk_cr", "ds41_engdd", "sched_noa", "stage_short",
                                        "stage_pick", "place_hb", "stage_full_min", "stage_full_pick", "mhc_fuse",
                                        "ix_lds", "ix_ni", "ds41_arcoded", "sched_row_cap",
                                        "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "sched_cb_hold_k",
                                        "exl3_x3d", "gg_gemv_m16" };
/* Switches a plugin serves (the rest are the core's own). */
bool plugin_switch(int i) {
    return i == kGgGemv || i == kDs41Argather || i == kDs41ShexpLane || i == kDs41KvFold || i == kDs41OFold ||
           i == kDs41OneFill || i == kDrafterMx || i == kDrafterBf ||
           i == kIdxSplit || i == kDs41QnFold || i == kDs41SwFold || i == kGgPf || i == kF32Pf || i == kDsparkWalk ||
           i == kDs41Bi || i == kDs41Psk || i == kDs41PskL20 || i == kDsparkPfctx || i == kDs41PskCr || i == kDs41Engdd ||
           i == kMhcFuse || i == kIxLds || i == kIxNi || i == kDs41Arcoded || i == kDs41MixWv || i == kDs41Gq ||
           i == kDs41E2 ||
           i == kExl3X3d ||   /* exl3_x3d (62 here; 57 on ds41-x3d-eng10): the ds41-x3d exl3r4d (X3D-F) */
           i == kGgGemvM16;   /* gg_gemv_m16 (63, ds41-a4): libr4d (lever A4) */
}

/* One loaded object's hooks; accepts and value may be absent (a plugin built before them). */
struct Hooks {
    int (*sw)(const char*, int) = nullptr;
    int (*accepts)(const char*, int) = nullptr;
    int (*value)(const char*, int*) = nullptr;
    /* lever PSK design 2 (deepseek41's rad_psk_plan), absent elsewhere */
    int (*psk_plan)(int, long long, int, const int32_t*, const int32_t*) = nullptr;
    int (*psk_stats)(long long*, int) = nullptr;   /* its live counters (ds41-int7-pskglue) */
    int (*pfctx_stats)(long long*, int) = nullptr; /* lever PSK-DS's (ds41-pskl20) */
    /* lever PSK-CR (ds41-pskcr): the plan with each prefill sequence's replay flag (rad_psk_plan2), and the rows a replay
     * reaches back from its end (rad_psk_cr_reach) */
    int (*psk_plan2)(int, long long, int, const int32_t*, const int32_t*, const int32_t*) = nullptr;
    int (*psk_cr_reach)(void) = nullptr;
    int (*engdd_stats)(long long*, int) = nullptr; /* lever E1's live counters (deepseek41's rad_engdd_stats, ds41-e1) */
    int (*i9b_stats)(long long*, int) = nullptr;   /* levers M1 / IX-NI (deepseek41's rad_i9b_stats, ds41-int9b) */
    int (*ix_stats)(long long*, int) = nullptr;    /* levers IX-LDSR2 / IX-NI (libr4d's rad_ix_stats, ds41-int9b) */
    int (*arc_stats)(long long*, int) = nullptr;   /* lever ARC (libr4d's rad_arc_stats, ds41-int10) */
    int (*i11_stats)(long long*, int) = nullptr;   /* levers 1U-GQ / 1U-E2 (deepseek41's rad_i11_stats, ds41-int11) */
    int (*mx_stats)(long long*, int) = nullptr;    /* lever 1U-MX (libr4d's rad_mx_stats, ds41-int11) */
};

struct State {
    /* draft_adaptive: the scheduler's window (-1 until one registers); gg_gemv: the plugins' value as last read
     * (state_json reads it live); aql_acq: the switch once set, -1 = the environment's rule. */
    std::atomic<int> val[kSwitches];
    std::atomic<int> draft_capable{-1};      /* -1 no scheduler registered, 0 / 1 */
    std::atomic<int> draft_configured{0};    /* --draft-window adaptive at start */
    std::atomic<int> hybrid_capable{-1};     /* -1 no scheduler registered, 0 / 1 */
    std::atomic<int> conf_capable{-1}, alt_capable{-1};   /* dspark_conf / dspark_alt: -1 no engine registered */
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> applied_step{-1};
    std::atomic<int64_t> cnt[8];   /* note_draft_counts */
    std::atomic<int> lend_capable{-1};   /* -1 no scheduler registered, 0 / 1 */
    std::atomic<int64_t> lend[18];       /* note_lend_sched [0, 9), 16 and 17, note_lend_engine [9, 16) */
    std::atomic<int64_t> acc[4];         /* note_draft_accept */
    std::atomic<int> ctxskip_capable{-1};   /* -1 no engine registered, 0 / 1 */
    std::atomic<int> arena_d0_capable{-1};  /* lever Z0-A: -1 no engine registered, 0 / 1 */
    std::atomic<int> draft_w_capable{-1};   /* lever Z0-W: -1 no engine registered, 0 / 1 */
    std::atomic<int> place_lf_capable{-1};  /* lever LF: -1 no engine registered, 0 / 1 */
    std::atomic<int64_t> lendw[5];          /* note_lend_w */
    std::atomic<int64_t> ctxs[8];        /* note_ctxskip */
    std::atomic<int> sample_capable{-1};   /* draft_sample: -1 no engine registered, 0 / 1 */
    std::atomic<int64_t> smp[6];         /* note_sample */
    /* lever DST (sched_register): --prefill-interleave as configured (-1 auto), the shared budget, whether a shared
     * step exists (-1 no scheduler registered), the chunk quantum */
    std::atomic<int> il_flag{0};
    std::atomic<int> mixed_start{-1}, mixed_capable{-1}, quantum{16};
    std::atomic<int> max_tok{0};                            /* sched_short_bypass's ceiling */
    std::atomic<int64_t> dst[2];                            /* note_sched_dst: bypassed, turned away */
    std::atomic<int64_t> noa[kNoaCounts];                   /* note_sched_noa (lever NOA) */
    std::atomic<int64_t> cap[kCapCounts];                   /* note_sched_cap (lever CAP48) */
    std::atomic<int64_t> cbh[kCbhCounts];                   /* note_sched_cbh (lever CBH) */
    std::atomic<int> stage_capable{-1}, hb_capable{-1};     /* lever SA / HB: -1 no engine registered, 0 / 1 */
    std::atomic<int64_t> sahb[kSahbRanks][kSahbN];          /* note_sahb */
    std::atomic<int> sahb_ranks{0};
    std::atomic<int> sp_capable{-1};                        /* lever SP: -1 no engine registered, 0 / 1 */
    std::atomic<int64_t> sp[kSahbRanks][kSpN];              /* note_sp */
    std::atomic<int> sp_ranks{0};
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
        for (auto& c : acc) c.store(0, std::memory_order_relaxed);
        for (auto& c : ctxs) c.store(0, std::memory_order_relaxed);
        for (auto& c : smp) c.store(0, std::memory_order_relaxed);
        for (auto& c : dst) c.store(0, std::memory_order_relaxed);
        for (auto& c : noa) c.store(0, std::memory_order_relaxed);
        for (auto& c : cap) c.store(0, std::memory_order_relaxed);
        for (auto& c : cbh) c.store(0, std::memory_order_relaxed);
        for (auto& r : sahb)
            for (auto& c : r) c.store(0, std::memory_order_relaxed);
        for (auto& r : sp)
            for (auto& c : r) c.store(0, std::memory_order_relaxed);
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
            k.psk_plan = sym<int (*)(int, long long, int, const int32_t*, const int32_t*)>(h, "rad_psk_plan");
            k.psk_stats = sym<int (*)(long long*, int)>(h, "rad_psk_stats");
            k.pfctx_stats = sym<int (*)(long long*, int)>(h, "rad_dspark_pfctx_stats");
            k.psk_plan2 = sym<int (*)(int, long long, int, const int32_t*, const int32_t*, const int32_t*)>(h, "rad_psk_plan2");
            k.psk_cr_reach = sym<int (*)(void)>(h, "rad_psk_cr_reach");
            k.engdd_stats = sym<int (*)(long long*, int)>(h, "rad_engdd_stats");
            k.i9b_stats = sym<int (*)(long long*, int)>(h, "rad_i9b_stats");
            k.ix_stats = sym<int (*)(long long*, int)>(h, "rad_ix_stats");
            k.arc_stats = sym<int (*)(long long*, int)>(h, "rad_arc_stats");
            k.i11_stats = sym<int (*)(long long*, int)>(h, "rad_i11_stats");
            k.mx_stats = sym<int (*)(long long*, int)>(h, "rad_mx_stats");
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

int env_sched_bi() {
    static const int env = [] {
        const char* e = std::getenv("RADIANCE_SCHED_BI_CHUNK");
        return e && e[0] >= '0' && e[0] <= '4' && !e[1] ? e[0] - '0' : 0;
    }();
    return env;
}

/* Lever CAP48: RADIANCE_DECODE_ROW_CAP at start, 0 .. 4096 (0 / unset: off); anything else is said once and means off. */
int env_row_cap() {
    static const int env = [] {
        const char* e = std::getenv("RADIANCE_DECODE_ROW_CAP");
        if (!e || !*e) return 0;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (!std::isdigit((unsigned char)e[0]) || (end && *end) || v < 0 || v > 4096) {
            std::fprintf(stderr, "radiance: scheduler: RADIANCE_DECODE_ROW_CAP=%s is not 0 .. 4096: no decode row cap\n", e);
            return 0;
        }
        return (int)v;
    }();
    return env;
}

/* Lever CBH: RADIANCE_SCHED_CB_HOLD_MS at start, 0 .. 60000 (0 / unset: off), and RADIANCE_SCHED_CB_HOLD_K, 2 .. 64 (unset: 2);
 * anything else is said once and means off / 2. */
int env_int(const char* name, int lo, int hi, int dflt, const char* what) {
    const char* e = std::getenv(name);
    if (!e || !*e) return dflt;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (!std::isdigit((unsigned char)e[0]) || (end && *end) || v < lo || v > hi) {
        std::fprintf(stderr, "radiance: scheduler: %s=%s is not %d .. %d: %s\n", name, e, lo, hi, what);
        return dflt;
    }
    return (int)v;
}
int env_cb_hold_ms() {
    static const int env = env_int("RADIANCE_SCHED_CB_HOLD_MS", 0, 60000, 0, "no co-batch hold");
    return env;
}
int env_cb_hold_k() {
    static const int env = env_int("RADIANCE_SCHED_CB_HOLD_K", 2, 64, 2, "a hold ends at 2 waiting");
    return env;
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
    if (i == kSchedBiChunk) {
        const int v = s.val[i].load(std::memory_order_acquire);
        return v >= 0 ? v : env_sched_bi();
    }
    if (i == kSchedInterleave) {   /* lever DST: set, else the flag (-1 = auto) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v >= 0 ? v : s.il_flag.load(std::memory_order_acquire);
    }
    if (i == kSchedMixedTok) {     /* lever DST: set, else the shared budget (-1 before a scheduler registered) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v > 0 ? v : s.mixed_start.load(std::memory_order_acquire);
    }
    if (i == kStageShort || i == kStagePick || i == kPlaceHb || i == kStageFullMin || i == kStageFullPick) {
        /* lever SA / HB / SP: set, else 0 (off) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v > 0 ? v : 0;
    }
    if (i == kSchedShortBypass || i == kSchedBypassTotal || i == kSchedNoa) {   /* lever DST / NOA: set, else 0 (off) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v > 0 ? v : 0;
    }
    if (i == kSchedRowCap) {       /* lever CAP48: set, else RADIANCE_DECODE_ROW_CAP (0 = off) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v >= 0 ? v : env_row_cap();
    }
    if (i == kSchedCbHoldMs) {     /* lever CBH: set, else RADIANCE_SCHED_CB_HOLD_MS (0 = off) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v >= 0 ? v : env_cb_hold_ms();
    }
    if (i == kSchedCbHoldK) {      /* lever CBH: set, else RADIANCE_SCHED_CB_HOLD_K (2) */
        const int v = s.val[i].load(std::memory_order_acquire);
        return v > 0 ? v : env_cb_hold_k();
    }
    return s.val[i].load(std::memory_order_acquire);
}

}  // namespace

bool sched_switch(int i) {
    return i == kDraftAdaptive || i == kDraftDepth0 || i == kDraftProbe || i == kDraftHybrid ||
           i == kDraftHybridMatch || i == kDraftProbeBackoff || i == kDraftLend || i == kDraftLendBorrow ||
           i == kDraftCtxSkip || i == kDraftLendHyst || i == kArenaD0 || i == kDraftLendW ||
           i == kDraftOcc || i == kDraftOccK || i == kPlaceLf || i == kDs41PskK || i == kStageShort ||
           i == kStagePick || i == kPlaceHb || i == kStageFullMin || i == kStageFullPick;
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
int sched_bi_chunk() {
    const int v = st().val[kSchedBiChunk].load(std::memory_order_acquire);
    return v >= 0 ? v : env_sched_bi();
}
void sched_register(int interleave_flag, int64_t shared_tok, bool shared_capable, int64_t quantum, int64_t max_tok) {
    State& s = st();
    s.max_tok.store((int)max_tok, std::memory_order_release);
    s.il_flag.store(interleave_flag < 0 ? -1 : interleave_flag, std::memory_order_release);
    s.mixed_start.store((int)shared_tok, std::memory_order_release);
    s.mixed_capable.store(shared_capable ? 1 : 0, std::memory_order_release);
    s.quantum.store(quantum > 0 ? (int)quantum : 1, std::memory_order_release);
}
int sched_interleave() {
    const int v = st().val[kSchedInterleave].load(std::memory_order_acquire);
    return v >= 0 ? v : -1;
}
int sched_mixed_tok() {
    const int v = st().val[kSchedMixedTok].load(std::memory_order_acquire);
    return v > 0 ? v : 0;
}
int sched_short_bypass() {
    const int v = st().val[kSchedShortBypass].load(std::memory_order_acquire);
    return v > 0 ? v : 0;
}
int sched_bypass_total() {
    const int v = st().val[kSchedBypassTotal].load(std::memory_order_acquire);
    return v > 0 ? v : 0;
}
void note_sched_dst(int64_t bypassed, int64_t bypass_refused) {
    State& s = st();
    s.dst[0].store(bypassed, std::memory_order_relaxed);
    s.dst[1].store(bypass_refused, std::memory_order_relaxed);
}
int sched_noa() {
    const int v = st().val[kSchedNoa].load(std::memory_order_acquire);
    return v > 0 ? v : 0;
}
void note_sched_noa(const int64_t* v) {
    State& s = st();
    for (int k = 0; k < kNoaCounts; ++k) s.noa[k].store(v[k], std::memory_order_relaxed);
}
int sched_row_cap() {
    const int v = st().val[kSchedRowCap].load(std::memory_order_acquire);
    return v >= 0 ? v : env_row_cap();
}
void note_sched_cap(const int64_t* v) {
    State& s = st();
    for (int k = 0; k < kCapCounts; ++k) s.cap[k].store(v[k], std::memory_order_relaxed);
}
int sched_cb_hold_ms() {
    const int v = st().val[kSchedCbHoldMs].load(std::memory_order_acquire);
    return v >= 0 ? v : env_cb_hold_ms();
}
int sched_cb_hold_k() {
    const int v = st().val[kSchedCbHoldK].load(std::memory_order_acquire);
    return v > 0 ? v : env_cb_hold_k();
}
void note_sched_cbh(const int64_t* v) {
    State& s = st();
    for (int k = 0; k < kCbhCounts; ++k) s.cbh[k].store(v[k], std::memory_order_relaxed);
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
                       int64_t hyb_unahead, int64_t occ_cycles, int64_t occ_moves) {
    State& s = st();
    const int64_t v[8] = { zero, probe, moves, hyb_lookup, hyb_head, hyb_unahead, occ_cycles, occ_moves };
    for (int k = 0; k < 8; ++k) s.cnt[k].store(v[k], std::memory_order_relaxed);
}

void lend_register(bool capable) { st().lend_capable.store(capable ? 1 : 0, std::memory_order_release); }

void note_draft_accept(int64_t head_drafted, int64_t head_accepted, int64_t lookup_drafted, int64_t lookup_accepted) {
    State& s = st();
    const int64_t v[4] = { head_drafted, head_accepted, lookup_drafted, lookup_accepted };
    for (int k = 0; k < 4; ++k) s.acc[k].store(v[k], std::memory_order_relaxed);
}

void ctxskip_register(bool capable) { st().ctxskip_capable.store(capable ? 1 : 0, std::memory_order_release); }
void arena_d0_register(bool capable) { st().arena_d0_capable.store(capable ? 1 : 0, std::memory_order_release); }
void draft_w_register(bool capable) { st().draft_w_capable.store(capable ? 1 : 0, std::memory_order_release); }
void place_lf_register(bool capable) { st().place_lf_capable.store(capable ? 1 : 0, std::memory_order_release); }
void sahb_register(bool stage_capable, bool hb_capable) {
    State& s = st();
    s.stage_capable.store(stage_capable ? 1 : 0, std::memory_order_release);
    s.hb_capable.store(hb_capable ? 1 : 0, std::memory_order_release);
}
void note_sahb(int rank, const int64_t* v) {
    if (rank < 0 || rank >= kSahbRanks || !v) return;
    State& s = st();
    for (int k = 0; k < kSahbN; ++k) s.sahb[rank][k].store(v[k], std::memory_order_relaxed);
    int n = s.sahb_ranks.load(std::memory_order_relaxed);
    while (n < rank + 1 && !s.sahb_ranks.compare_exchange_weak(n, rank + 1, std::memory_order_relaxed)) {}
}
void sp_register(bool capable) { st().sp_capable.store(capable ? 1 : 0, std::memory_order_release); }
void note_sp(int rank, const int64_t* v) {
    if (rank < 0 || rank >= kSahbRanks || !v) return;
    State& s = st();
    for (int k = 0; k < kSpN; ++k) s.sp[rank][k].store(v[k], std::memory_order_relaxed);
    int n = s.sp_ranks.load(std::memory_order_relaxed);
    while (n < rank + 1 && !s.sp_ranks.compare_exchange_weak(n, rank + 1, std::memory_order_relaxed)) {}
}
void note_lend_w(int64_t lent, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t mib) {
    State& s = st();
    const int64_t v[5] = { lent, lends, reclaims, lent_units, mib };
    for (int k = 0; k < 5; ++k) s.lendw[k].store(v[k], std::memory_order_relaxed);
}

void note_ctxskip(int64_t skipped, int64_t refill_passes, int64_t refill_rows, int64_t dropped_rows,
                  int64_t ring_full, int64_t pending_rows, int64_t ring_slots, int64_t ring_mib) {
    State& s = st();
    const int64_t v[8] = { skipped, refill_passes, refill_rows, dropped_rows, ring_full, pending_rows, ring_slots,
                           ring_mib };
    for (int k = 0; k < 8; ++k) s.ctxs[k].store(v[k], std::memory_order_relaxed);
}

void note_lend_sched(int64_t lent_cycles, int64_t plain_steps, int64_t plain_ms, int64_t probe_steps,
                     int64_t probe_ms, int64_t transit_steps, int64_t transit_ms, int64_t backoff_k,
                     int64_t probe_interval, int64_t borrow_cycles, int64_t hyst_holds) {
    State& s = st();
    const int64_t v[9] = { lent_cycles, plain_steps, plain_ms, probe_steps, probe_ms, transit_steps, transit_ms,
                           backoff_k, probe_interval };
    for (int k = 0; k < 9; ++k) s.lend[k].store(v[k], std::memory_order_relaxed);
    s.lend[16].store(borrow_cycles, std::memory_order_relaxed);
    s.lend[17].store(hyst_holds, std::memory_order_relaxed);
}

void note_lend_engine(int64_t state, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t draft_resident,
                      int64_t draft_units, int64_t slab_mib) {
    State& s = st();
    const int64_t v[7] = { state, lends, reclaims, lent_units, draft_resident, draft_units, slab_mib };
    for (int k = 0; k < 7; ++k) s.lend[9 + k].store(v[k], std::memory_order_relaxed);
}

bool dspark_conf_value_ok(int v) {
    return v == 0 || v == 1 || v == 2 || v == 101 || (v >= 5 && v <= 95) || (v >= 105 && v <= 195);
}

void dspark_register(bool conf_capable, bool alt_capable, int conf_start, int alt_start) {
    State& s = st();
    s.conf_capable.store(conf_capable ? 1 : 0, std::memory_order_release);
    s.alt_capable.store(alt_capable ? 1 : 0, std::memory_order_release);
    s.val[kDsparkConf].store(conf_capable && dspark_conf_value_ok(conf_start) ? conf_start : 0,
                             std::memory_order_release);
    s.val[kDsparkAlt].store(alt_capable && alt_start == 1 ? 1 : 0, std::memory_order_release);
    s.val[kDsparkConfRows].store(0, std::memory_order_release);   /* ds41-small: no load bound at start */
}

void sample_register(bool capable, int start) {
    State& s = st();
    s.sample_capable.store(capable ? 1 : 0, std::memory_order_release);
    s.val[kDraftSample].store(capable && start >= 1 && start <= 200 ? start : 0, std::memory_order_release);
}

void note_sample(int64_t rows_sampled, int64_t rows_greedy, int64_t rounds_bound, int64_t rounds_malformed,
                 int64_t bind_fallback, int64_t lookup_cleared) {
    State& s = st();
    const int64_t v[6] = { rows_sampled, rows_greedy, rounds_bound, rounds_malformed, bind_fallback, lookup_cleared };
    for (int k = 0; k < 6; ++k) s.smp[k].fetch_add(v[k], std::memory_order_relaxed);
}

int get(int i) {
    if (i < 0 || i >= kSwitches) return -1;
    return st().val[i].load(std::memory_order_acquire);
}

namespace {
std::atomic<int64_t> g_psk_eng[3];   /* note_psk_engine */
}
void note_psk_engine(int64_t planned, int64_t ended_early, int64_t stager_skipped) {
    g_psk_eng[0].store(planned, std::memory_order_relaxed);
    g_psk_eng[1].store(ended_early, std::memory_order_relaxed);
    g_psk_eng[2].store(stager_skipped, std::memory_order_relaxed);
}
void psk_engine(int64_t out[3]) {
    for (int i = 0; i < 3; ++i) out[i] = g_psk_eng[i].load(std::memory_order_relaxed);
}
int psk_stats(int64_t out[kPskStats]) {
    for (int i = 0; i < kPskStats; ++i) out[i] = 0;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    for (const Hooks& k : s.hooks)
        if (k.psk_stats) {
            long long v[kPskStats] = {};
            const int n = k.psk_stats(v, kPskStats);
            for (int i = 0; i < n && i < kPskStats; ++i) out[i] = (int64_t)v[i];
            return n;
        }
    return 0;
}
int dspark_pfctx_stats(int64_t out[4]) {
    out[0] = -1;
    for (int i = 1; i < 4; ++i) out[i] = 0;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    for (const Hooks& k : s.hooks)
        if (k.pfctx_stats) {
            long long v[4] = { -1, 0, 0, 0 };
            const int n = k.pfctx_stats(v, 4);
            for (int i = 0; i < n && i < 4; ++i) out[i] = (int64_t)v[i];
            return n;
        }
    return 0;
}

/* Under s.mu. */
static void i9b_collect(State& s, int64_t* out) {
    for (int i = 0; i < kI9bStats; ++i) out[i] = 0;
    find_hooks(s);
    bool got_a = false, got_b = false;
    for (const Hooks& k : s.hooks) {
        if (k.i9b_stats && !got_a) {
            long long v[6] = { 0, 0, 0, 0, 0, 0 };
            const int n = k.i9b_stats(v, 6);
            for (int i = 0; i < n && i < 6; ++i) out[i] = (int64_t)v[i];
            got_a = n > 0;
        }
        if (k.ix_stats && !got_b) {
            long long v[4] = { 0, 0, 0, 0 };
            const int n = k.ix_stats(v, 4);
            for (int i = 0; i < n && i < 4; ++i) out[6 + i] = (int64_t)v[i];
            got_b = n > 0;
        }
    }
}
void i9b_stats(int64_t out[kI9bStats]) {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    i9b_collect(s, out);
}

/* Under s.mu. */
static void arc_collect(State& s, int64_t* out) {
    for (int i = 0; i < kArcStats; ++i) out[i] = 0;
    find_hooks(s);
    for (const Hooks& k : s.hooks) {
        if (!k.arc_stats) continue;
        long long v[kArcStats] = {};
        const int n = k.arc_stats(v, kArcStats);
        for (int i = 0; i < n && i < kArcStats; ++i) out[i] = (int64_t)v[i];
        if (n > 0) break;
    }
}
void arc_stats(int64_t out[kArcStats]) {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    arc_collect(s, out);
}

/* Under s.mu. */
static void i11_collect(State& s, int64_t* out) {
    for (int i = 0; i < kI11Stats; ++i) out[i] = 0;
    find_hooks(s);
    bool got_a = false, got_b = false;
    for (const Hooks& k : s.hooks) {
        if (k.i11_stats && !got_a) {
            long long v[6] = { 0, 0, 0, 0, 0, 0 };
            const int n = k.i11_stats(v, 6);
            for (int i = 0; i < n && i < 6; ++i) out[i] = (int64_t)v[i];
            got_a = n > 0;
        }
        if (k.mx_stats && !got_b) {
            long long v[2] = { 0, 0 };
            const int n = k.mx_stats(v, 2);
            for (int i = 0; i < n && i < 2; ++i) out[6 + i] = (int64_t)v[i];
            got_b = n > 0;
        }
    }
}
void i11_stats(int64_t out[kI11Stats]) {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    i11_collect(s, out);
}

int engdd_stats(int64_t out[7]) {
    for (int i = 0; i < 7; ++i) out[i] = 0;
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    for (const Hooks& k : s.hooks)
        if (k.engdd_stats) {
            long long v[7] = { 0, 0, 0, 0, 0, 0, 0 };
            const int n = k.engdd_stats(v, 7);
            for (int i = 0; i < n && i < 7; ++i) out[i] = (int64_t)v[i];
            return n;
        }
    return 0;
}

int psk_plan(int step, int64_t dec_tok, int n, const int32_t* ahead, const int32_t* rows, const int32_t* flags) {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    /* lever PSK-CR: a plan with a replay entry goes to rad_psk_plan2 where the plugin has it; one without (a plugin built
     * before it) takes the plain plan -- the replay's rows then run uncut, its global-KV stores still masked by the batch */
    bool any = false;
    for (int i = 0; flags && i < n; ++i) any = any || flags[i] != 0;
    for (const Hooks& k : s.hooks) {
        if (any && k.psk_plan2) return k.psk_plan2(step, (long long)dec_tok, n, ahead, rows, flags);
        if (k.psk_plan) return k.psk_plan(step, (long long)dec_tok, n, ahead, rows);
    }
    return -1;
}

int psk_cr_reach() {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    find_hooks(s);
    for (const Hooks& k : s.hooks)
        if (k.psk_cr_reach) return k.psk_cr_reach();
    return -1;
}

bool check(const std::string& name, int v, std::string* err) {
    if (name == "epoch_bump") return true;
    const int i = switch_index(name);
    if (i < 0) {
        if (err) *err = "unknown switch '" + name + "' (draft_adaptive, gg_gemv, aql_acq, ds41_argather, "
                        "ds41_shexp_lane, ds41_kvfold, ds41_ofold, ds41_onefill, draft_depth0, draft_probe, "
                        "draft_hybrid, draft_hybrid_match, draft_probe_backoff, draft_lend, draft_lend_borrow, "
                        "drafter_mx, drafter_bf, idx_split, ds41_qnfold, ds41_swfold, draft_ctxskip, dspark_conf, dspark_alt, "
                        "dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, "
                        "draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok, "
                        "sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx, "
                        "ds41_psk_cr, ds41_engdd, sched_noa, stage_short, stage_pick, place_hb, stage_full_min, "
                        "stage_full_pick, mhc_fuse, ix_lds, ix_ni, ds41_arcoded, sched_row_cap, ds41_mix_wv, ds41_gq, ds41_e2, "
                        "sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, gg_gemv_m16, epoch_bump)";
        return false;
    }
    State& s = st();
    if (i == kDraftSample) {
        if (v < 0 || v > 200) {
            if (err) *err = "draft_sample takes 0 (greedy drafts, the agreement rule), 1 (sampled drafts x ~ Q at the "
                            "request's temperature, the maximal coupling at the verify) or 2 .. 200 (Q at the "
                            "temperature x v / 100)";
            return false;
        }
        const int cap = s.sample_capable.load(std::memory_order_acquire);
        if (v != 0 && cap != 1) {
            if (err)
                *err = std::string("draft_sample=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no engine has registered a drafter"
                                : "this server has no coupled drafter (--draft-coupling on, and a drafter whose walk "
                                  "declares the coupling planes: deepseek41's DSpark over 2 .. 4 ranks)");
            return false;
        }
        return true;
    }
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
    if (i == kDraftOcc || i == kDraftOccK) {   /* lever Z0-O */
        if (i == kDraftOcc && (v < 0 || v > 64)) {
            if (err) *err = "draft_occ takes 0 (off) .. 64 (decode rows that hold depth 0 once sustained draft_occ_k cycles)";
            return false;
        }
        if (i == kDraftOccK && (v < 1 || v > 4096)) {
            if (err) *err = "draft_occ_k takes 1 .. 4096 (consecutive cycles draft_occ counts each way)";
            return false;
        }
        const int cap = s.draft_capable.load(std::memory_order_acquire);
        if (i == kDraftOcc && v > 0 && cap != 1) {
            if (err)
                *err = std::string("draft_occ=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no scheduler has registered a drafter"
                                : "this deployment's drafter cannot run the adaptive window");
            return false;
        }
        return true;
    }
    if (i == kPlaceLf) {   /* lever LF */
        if (v < 0 || v > 1) {
            if (err) *err = "place_lf takes 1 (lent slots filled by the long-memory decode heat) or 0 (off)";
            return false;
        }
        const int cap = s.place_lf_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string("place_lf=1 refused: ") +
                       (cap < 0 ? "no engine has registered its placement"
                                : "placement is static here (no heat engine on every rank): nothing fills lent slots");
            return false;
        }
        return true;
    }
    if (i == kStageShort || i == kStagePick || i == kPlaceHb) {   /* lever SA / HB (ds41-sahb) */
        const int hi = i == kStageShort ? 65536 : i == kStagePick ? 64 : 4096;
        if (v < 0 || v > hi) {
            if (err)
                *err = i == kStageShort ? "stage_short takes 0 (off) or 1 .. 65536 (the fewest tokens a step at the span's "
                                          "shared level arms the prefill stager at)"
                     : i == kStagePick  ? "stage_pick takes 0 (every pooled unit) or 1 .. 64 (the units one of the last K "
                                          "stager-size trunk steps routed)"
                                        : "place_hb takes 0 (off) or 1 .. 4096 (units a named rank keeps resident by the "
                                          "long-memory prefill heat)";
            return false;
        }
        const int cap = (i == kPlaceHb ? s.hb_capable : s.stage_capable).load(std::memory_order_acquire);
        if (v > 0 && i != kStagePick && cap != 1) {
            if (err)
                *err = std::string(kNames[i]) + "=" + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no engine has registered it"
                        : i == kPlaceHb ? "no dual heat on every rank (RADIANCE_HEAT_PREFILL_DECAY at start, a dynamic plan)"
                                        : "no prefill stager with a span's shared level on every rank (--prefill-span > 1, "
                                          "a dynamic plan with a staging region)");
            return false;
        }
        return true;
    }
    if (i == kStageFullMin || i == kStageFullPick) {   /* lever SP (ds41-int9) */
        const int hi = i == kStageFullMin ? 1048576 : 64;
        if (v < 0 || v > hi) {
            if (err)
                *err = i == kStageFullMin ? "stage_full_min takes 0 (off) or 1 .. 1048576 (the fewest tokens a full-plan "
                                            "step arms the prefill stager at; a smaller one streams its pooled experts "
                                            "in place)"
                                          : "stage_full_pick takes 0 (every pooled unit) or 1 .. 64 (the units one of the "
                                            "last K stager-size trunk steps routed)";
            return false;
        }
        const int cap = s.sp_capable.load(std::memory_order_acquire);
        if (v > 0 && cap != 1) {
            if (err)
                *err = std::string(kNames[i]) + "=" + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no engine has registered it"
                                : "no prefill stager on every rank (a dynamic plan with a staging region)");
            return false;
        }
        return true;
    }
    if (i == kDs41PskK) {   /* lever PSK design 2: the scheduler's anchor stride */
        if (v < 1 || v > 4096) {
            if (err) *err = "ds41_psk_k takes 1 .. 4096 (every K-th periodic checkpoint position is an anchor under ds41_psk 2 / 3)";
            return false;
        }
        return true;
    }
    if (i == kDraftLendW) {
        if (v < 0 || v > 1) {
            if (err) *err = "draft_lend_w takes 1 (on) or 0 (off)";
            return false;
        }
        const int cap = s.draft_w_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string("draft_lend_w=1 refused: ") +
                       (cap < 0 ? "no engine has registered the drafter's weight lend"
                                : "the drafter's weight lend is not set up: RADIANCE_DRAFT_LEND_WEIGHTS (with "
                                  "RADIANCE_DRAFT_LEND) at start, and a drafter with query-pass weights on every rank");
            return false;
        }
        return true;
    }
    if (i == kArenaD0) {
        if (v < 0 || v > 1) {
            if (err) *err = "arena_d0 takes 1 (on) or 0 (off)";
            return false;
        }
        const int cap = s.arena_d0_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string("arena_d0=1 refused: ") +
                       (cap < 0 ? "no engine has registered its activation arena"
                                : "no decode rung below the largest of RADIANCE_DS41_ARENA_DECODE_ROWS (e.g. 8,48)");
            return false;
        }
        return true;
    }
    if (i == kDraftLendHyst) {
        if (v < 0 || v > 100) {
            if (err) *err = "draft_lend_hyst takes 0 (off) .. 100 (percent a move off a lent depth 0 must gain)";
            return false;
        }
        const int cap = s.lend_capable.load(std::memory_order_acquire);
        if (v > 0 && cap != 1) {
            if (err)
                *err = std::string("draft_lend_hyst=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "the drafter's lend is not set up: RADIANCE_DRAFT_LEND was not set at start"
                                : "the drafter's lend is not set up here: no drafter expert slab to lend");
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
    if (i == kDraftCtxSkip) {
        if (v < 0 || v > 1) {
            if (err) *err = "draft_ctxskip takes 1 (on) or 0 (off)";
            return false;
        }
        const int cap = s.ctxskip_capable.load(std::memory_order_acquire);
        if (v == 1 && cap != 1) {
            if (err)
                *err = std::string("draft_ctxskip=1 refused: ") +
                       (cap < 0 ? "the context skip is not set up: RADIANCE_DRAFT_CTXSKIP was not set at start"
                                : "the context skip is not set up here: no windowed block drafter drafting, or no "
                                  "tapped rows the engine can keep");
            return false;
        }
        return true;
    }
    if (i == kDraftDepth0 || i == kDraftHybrid) {
        /* draft_depth0 2: depth 0 forced (a test hook), 3: depth 0 by occupancy only (ds41-int10); accepted wherever 1 is. */
        if (v < 0 || v > (i == kDraftDepth0 ? 3 : 1)) {
            if (err)
                *err = std::string(kNames[i]) +
                       (i == kDraftDepth0 ? " takes 1 (on), 2 (forced, a test hook), 3 (occupancy only) or 0 (off)"
                                          : " takes 1 (on) or 0 (off)");
            return false;
        }
        const int cap = (i == kDraftHybrid ? s.hybrid_capable : s.draft_capable).load(std::memory_order_acquire);
        if (v >= 1 && cap != 1) {
            if (err)
                *err = std::string(kNames[i]) + "=" + std::to_string(v) + " refused: " +
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
    if (i == kDsparkConfRows) {
        if (v < 0 || v > 64) {
            if (err) *err = "dspark_conf_rows takes 0 (the confidence plane at every cycle) or 1 .. 64 (only where the "
                            "step decoded at most that many sequences)";
            return false;
        }
        if (v != 0 && s.conf_capable.load(std::memory_order_acquire) != 1) {
            if (err)
                *err = std::string("dspark_conf_rows=") + std::to_string(v) + " refused: " +
                       (s.conf_capable.load(std::memory_order_acquire) < 0
                            ? "no engine has registered a drafter"
                            : "this deployment's drafter declares no confidence plane (deepseek41: "
                              "RADIANCE_DS41_DSPARK_CONF at start)");
            return false;
        }
        return true;
    }
    if (i == kDsparkConf || i == kDsparkAlt) {
        const bool conf = i == kDsparkConf;
        if (conf ? !dspark_conf_value_ok(v) : (v < 0 || v > 1)) {
            if (err)
                *err = conf ? "dspark_conf takes 0 (off), 1 (the reference rule: verify the longest prefix with "
                              "sigmoid >= 0.5), 5..95 (that rule at v/100), 101 or 105..195 (the rule capped by the "
                              "adaptive window, at (v-100)/100, 101 = 0.5) or 2 (observe: count, cut nothing)"
                            : "dspark_alt takes 1 (the shadow tree's branches, measured) or 0 (off)";
            return false;
        }
        const int cap = (conf ? s.conf_capable : s.alt_capable).load(std::memory_order_acquire);
        if (v != 0 && cap != 1) {
            if (err)
                *err = std::string(kNames[i]) + "=" + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no engine has registered a drafter"
                                : conf ? "this deployment's drafter declares no confidence plane (deepseek41: "
                                         "RADIANCE_DS41_DSPARK_CONF at start)"
                                       : "this deployment's drafter declares no alternative branches (deepseek41: "
                                         "RADIANCE_DS41_DSPARK_ALT at start)");
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
    if (i == kSchedBiChunk) {
        if (v < 0 || v > 4) {
            if (err) *err = "sched_bi_chunk takes 0 (today), 1 (alone's boundaries), 2 (the interval grid), 3 / 4 (alone's "
                            "pieces cut on the interval / RADIANCE_SCHED_BI_GRID grid)";
            return false;
        }
        return true;
    }
    if (i == kSchedInterleave) {   /* lever DST */
        if (v < -1 || v > 64) {
            if (err) *err = "sched_interleave takes 0 .. 64 (decode-only steps after a chunk beside decoding sequences; 0 "
                            "= none, a chunk may ride every step) or -1 (back to --prefill-interleave)";
            return false;
        }
        return true;
    }
    if (i == kSchedShortBypass) {  /* lever DST */
        const int q = s.quantum.load(std::memory_order_acquire), top = s.max_tok.load(std::memory_order_acquire);
        if (v != 0 && (v < 0 || (q > 0 && v % q != 0) || (top > 0 && v > top) || top <= 0)) {
            if (err)
                *err = "sched_short_bypass takes 0 (off) or a multiple of the quantum (" + std::to_string(q) + ") up to " +
                       (top > 0 ? std::to_string(top) : std::string("--max-num-batched-tokens (no scheduler registered)")) +
                       ": the longest uncached rest a waiting request may run as one whole piece on a held step";
            return false;
        }
        return true;
    }
    if (i == kSchedNoa) {          /* lever NOA */
        const int cap = s.mixed_capable.load(std::memory_order_acquire);
        if (v != 0 && (v < 1 || v > 2 || cap != 1)) {
            if (err)
                *err = (v < 1 || v > 2) ? std::string("sched_noa takes 0 (off), 1 (one prefill a step D turns away beside another "
                                                      "prefill takes D's piece at the step's --max-num-batched-tokens room) "
                                                      "or 2 (as many as that room holds)")
                              : std::string("sched_noa=") + std::to_string(v) + " refused: " +
                                    (cap < 0 ? "no scheduler has registered"
                                             : "no shared step here (--prefill-span 1: every step is planned at "
                                               "--max-num-batched-tokens)");
            return false;
        }
        return true;
    }
    if (i == kSchedCbHoldMs) {     /* lever CBH */
        if (v < 0 || v > 60000) {
            if (err) *err = "sched_cb_hold_ms takes 0 (off) or 1 .. 60000: how long a waiting newcomer's admission may be held "
                            "(while >= 2 sequences decode) so that sched_cb_hold_k newcomers start in one step";
            return false;
        }
        return true;
    }
    if (i == kSchedCbHoldK) {      /* lever CBH */
        if (v < 2 || v > 64) {
            if (err) *err = "sched_cb_hold_k takes 2 .. 64: the waiting newcomers that end a co-batch hold";
            return false;
        }
        return true;
    }
    if (i == kSchedRowCap) {       /* lever CAP48 */
        if (v < 0 || v > 4096) {
            if (err) *err = "sched_row_cap takes 0 (off) or 1 .. 4096: the most decode rows a step carries (drafts trimmed, "
                            "the deepest first; a step whose decoding sequences alone exceed it is left as planned)";
            return false;
        }
        return true;
    }
    if (i == kSchedBypassTotal) {  /* lever DST */
        const int q = s.quantum.load(std::memory_order_acquire);
        if (v != 0 && (v < 0 || v > 65536 || (q > 0 && v % q != 0))) {
            if (err)
                *err = "sched_bypass_total takes 0 (one-piece requests only) or a multiple of the quantum (" + std::to_string(q) +
                       ") up to 65536: the longest uncached rest the bypass takes and runs piece by piece";
            return false;
        }
        return true;
    }
    if (i == kSchedMixedTok) {     /* lever DST */
        const int cap = s.mixed_capable.load(std::memory_order_acquire);
        const int top = s.mixed_start.load(std::memory_order_acquire);
        const int q = s.quantum.load(std::memory_order_acquire);
        if (v != 0 && cap != 1) {
            if (err)
                *err = std::string("sched_mixed_tok=") + std::to_string(v) + " refused: " +
                       (cap < 0 ? "no scheduler has registered" : "no shared step here (--prefill-span 1: every step is "
                                                                  "planned at --max-num-batched-tokens)");
            return false;
        }
        if (v != 0 && (v < 128 || v > top || (q > 0 && v % q != 0))) {
            if (err)
                *err = "sched_mixed_tok takes 0 (back to the shared budget, " + std::to_string(top) + ") or 128 .. " +
                       std::to_string(top) + ", a multiple of the quantum (" + std::to_string(q) +
                       "): the budget of a step that is not a lone prefill's span step, never above the one the "
                       "plugin's arenas were sized for";
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
        /* lever DST: back to the flag / the shared budget is stored as unset (-1 / 0), compared as the value it means */
        if (i == kSchedInterleave && v < 0) {
            const int fl = s.il_flag.load(std::memory_order_acquire);
            const int was = s.val[i].load(std::memory_order_acquire);
            s.val[i].store(-1, std::memory_order_release);
            if (was < 0 || was == fl) continue;   /* the flag's count already ran */
            changed = true;
            what += std::string(what.empty() ? "" : ", ") + kNames[i] + "=flag(" + std::to_string(fl) + ")";
            continue;
        }
        if (i == kSchedMixedTok && v <= 0) {
            const int was = s.val[i].load(std::memory_order_acquire);
            s.val[i].store(0, std::memory_order_release);
            if (was <= 0 || was == s.mixed_start.load(std::memory_order_acquire)) continue;   /* the budget already ran */
            changed = true;
            what += std::string(what.empty() ? "" : ", ") + kNames[i] + "=" +
                    std::to_string(s.mixed_start.load(std::memory_order_acquire)) + " (start)";
            continue;
        }
        if (reported(s, i) == v) {
            if (i == kAqlAcq || i == kSchedBiChunk || i == kSchedInterleave || i == kSchedMixedTok || i == kSchedShortBypass ||
                i == kSchedBypassTotal || i == kSchedNoa || i == kStageShort || i == kStagePick || i == kPlaceHb ||
                i == kStageFullMin || i == kStageFullPick || i == kSchedRowCap || i == kSchedCbHoldMs || i == kSchedCbHoldK)
                s.val[i].store(v, std::memory_order_release);   /* pinned, the same */
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
    static const char* const kCnt[8] = { "zero_cycles", "probe_cycles", "zero_moves", "hybrid_lookup", "hybrid_head",
                                         "hybrid_unahead", "occ_cycles", "occ_moves" };
    o += ", \"draft\": {";
    for (int k = 0; k < 8; ++k)
        o += std::string(k ? ", " : "") + "\"" + kCnt[k] + "\": " +
             std::to_string((long long)s.cnt[k].load(std::memory_order_relaxed));
    o += "}";
    static const char* const kAcc[4] = { "head_drafted", "head_accepted", "lookup_drafted", "lookup_accepted" };
    o += ", \"accept\": {";
    for (int k = 0; k < 4; ++k)
        o += std::string(k ? ", " : "") + "\"" + kAcc[k] + "\": " +
             std::to_string((long long)s.acc[k].load(std::memory_order_relaxed));
    o += "}";
    /* Lever SA / HB (ds41-sahb), once an engine registered (capable or not): per rank, cumulative unless noted. */
    if (const int sc = s.stage_capable.load(std::memory_order_acquire); sc >= 0) {
        static const char* const kSa[kSahbN] = {
            "short_steps", "short_layers", "short_units", "short_bytes", "pick_kept", "pick_dropped", "big_steps",
            "expect_units", "expect_bytes", "expect_sum_units", "expect_n", "hb_reserve", "hb_reserved", "hb_resident",
            "hb_promotions", "sa_recall_bytes" };
        o += ", \"sahb\": {\"stage_capable\": " + std::to_string(sc) +
             ", \"hb_capable\": " + std::to_string(s.hb_capable.load(std::memory_order_acquire)) + ", \"ranks\": [";
        const int nr = s.sahb_ranks.load(std::memory_order_relaxed);
        for (int r = 0; r < nr && r < kSahbRanks; ++r) {
            o += std::string(r ? ", " : "") + "{\"rank\": " + std::to_string(r);
            for (int k = 0; k < kSahbN; ++k)
                o += std::string(", \"") + kSa[k] + "\": " +
                     std::to_string((long long)s.sahb[r][k].load(std::memory_order_relaxed));
            o += "}";
        }
        o += "]}";
    }
    /* Lever SP (ds41-int9), once an engine registered (capable or not): per rank, cumulative. */
    if (const int pc = s.sp_capable.load(std::memory_order_acquire); pc >= 0) {
        static const char* const kSp[kSpN] = { "full_steps", "full_rows", "full_layers", "full_units", "full_bytes",
                                               "full_pick_kept", "full_pick_dropped", "stream_steps", "stream_rows" };
        o += ", \"sp\": {\"capable\": " + std::to_string(pc) + ", \"ranks\": [";
        const int nr = s.sp_ranks.load(std::memory_order_relaxed);
        for (int r = 0; r < nr && r < kSahbRanks; ++r) {
            o += std::string(r ? ", " : "") + "{\"rank\": " + std::to_string(r);
            for (int k = 0; k < kSpN; ++k)
                o += std::string(", \"") + kSp[k] + "\": " +
                     std::to_string((long long)s.sp[r][k].load(std::memory_order_relaxed));
            o += "}";
        }
        o += "]}";
    }
    /* Levers M1 / IX-LDSR2 / IX-NI (ds41-int9b): always present, 0s where no plugin serves them (i9b_stats). */
    {
        int64_t v[kI9bStats];
        i9b_collect(s, v);
        o += ", \"i9b\": {\"mhc\": {\"declared\": " + std::to_string((long long)v[0]) +
             ", \"pairs\": " + std::to_string((long long)v[1]) + ", \"rows\": " + std::to_string((long long)v[2]) +
             ", \"small\": " + std::to_string((long long)v[3]) + "}, \"ix\": {\"lds_launches\": " +
             std::to_string((long long)v[6]) + ", \"lds_rows\": " + std::to_string((long long)v[7]) +
             ", \"ni_launches\": " + std::to_string((long long)v[8]) + ", \"ni_cand_launches\": " +
             std::to_string((long long)v[9]) + "}, \"ni\": {\"runs\": " + std::to_string((long long)v[4]) +
             ", \"dead_bytes\": " + std::to_string((long long)v[5]) + "}}";
    }
    /* Lever ARC (ds41-int10): always present, 0s where no plugin serves it (arc_stats). */
    {
        int64_t v[kArcStats];
        arc_collect(s, v);
        o += ", \"arc\": {\"coded_launches\": " + std::to_string((long long)v[0]) +
             ", \"coded_rows\": " + std::to_string((long long)v[1]) + ", \"plain_launches\": " +
             std::to_string((long long)v[2]) + ", \"plain_rows\": " + std::to_string((long long)v[3]) +
             ", \"below\": " + std::to_string((long long)v[4]) + ", \"declined\": " + std::to_string((long long)v[5]) +
             ", \"coded_all_ranks\": " + std::to_string((long long)v[6]) + "}";
    }
    /* Levers 1U-MX / 1U-GQ / 1U-E2 (ds41-int11): always present, 0s where no plugin serves them (i11_stats). */
    {
        int64_t v[kI11Stats];
        i11_collect(s, v);
        o += ", \"i11\": {\"gq\": {\"declared\": " + std::to_string((long long)v[0]) +
             ", \"qkv_pairs\": " + std::to_string((long long)v[1]) + ", \"out_pairs\": " + std::to_string((long long)v[2]) +
             "}, \"e2\": {\"declared\": " + std::to_string((long long)v[3]) + ", \"splits\": " +
             std::to_string((long long)v[4]) + ", \"rows\": " + std::to_string((long long)v[5]) +
             "}, \"mx\": {\"wide_launches\": " + std::to_string((long long)v[6]) + ", \"wide_rows\": " +
             std::to_string((long long)v[7]) + "}}";
    }
    /* Lever DL, once a scheduler registered its lend (set up or not). */
    if (const int cap = s.lend_capable.load(std::memory_order_acquire); cap >= 0) {
        static const char* const kLend[18] = { "lent_cycles", "lent_plain_steps", "lent_plain_ms", "lent_probe_steps",
                                               "lent_probe_ms", "transit_steps", "transit_ms", "backoff_k",
                                               "probe_interval", "state", "lends", "reclaims", "lent_units",
                                               "draft_resident", "draft_units", "slab_mib", "borrow_cycles",
                                               "hyst_holds" };
        o += ", \"lend\": {\"capable\": " + std::to_string(cap);
        for (int k = 0; k < 18; ++k)
            o += std::string(", \"") + kLend[k] + "\": " +
                 std::to_string((long long)s.lend[k].load(std::memory_order_relaxed));
        /* Lever Z0-W's, once an engine registered it set up. */
        if (s.draft_w_capable.load(std::memory_order_acquire) == 1) {
            static const char* const kW[5] = { "w_lent", "w_lends", "w_reclaims", "w_lent_units", "w_mib" };
            for (int k = 0; k < 5; ++k)
                o += std::string(", \"") + kW[k] + "\": " + std::to_string((long long)s.lendw[k].load(std::memory_order_relaxed));
        }
        o += "}";
    }
    /* Lever SD, once the engine registered it (capable or not). */
    if (const int cap = s.sample_capable.load(std::memory_order_acquire); cap >= 0) {
        static const char* const kSm[6] = { "rows_sampled", "rows_greedy", "rounds_bound", "rounds_malformed",
                                            "bind_fallback", "lookup_cleared" };
        o += ", \"sample\": {\"capable\": " + std::to_string(cap);
        for (int k = 0; k < 6; ++k)
            o += std::string(", \"") + kSm[k] + "\": " + std::to_string((long long)s.smp[k].load(std::memory_order_relaxed));
        o += "}";
    }
    /* Lever DST's bypass counts, once a scheduler registered (sched_register). */
    if (s.max_tok.load(std::memory_order_acquire) > 0)
        o += ", \"dst\": {\"bypassed\": " + std::to_string((long long)s.dst[0].load(std::memory_order_relaxed)) +
             ", \"bypass_refused\": " + std::to_string((long long)s.dst[1].load(std::memory_order_relaxed)) + "}";
    /* Lever NOA's counts, once a scheduler registered (sched_register). */
    if (s.max_tok.load(std::memory_order_acquire) > 0) {
        static const char* const kNoa[kNoaCounts] = { "admitted", "pieces", "rows", "displaced", "waited", "steps",
                                                      "pf_steps_1", "pf_steps_2", "pf_steps_3", "pf_steps_4plus" };
        o += ", \"noa\": {";
        for (int k = 0; k < kNoaCounts; ++k)
            o += std::string(k ? ", " : "") + "\"" + kNoa[k] + "\": " +
                 std::to_string((long long)s.noa[k].load(std::memory_order_relaxed));
        o += "}";
    }
    /* Lever CAP48's counts, once a scheduler registered (sched_register). */
    if (s.max_tok.load(std::memory_order_acquire) > 0) {
        static const char* const kCap[kCapCounts] = { "steps_capped", "drafts_trimmed", "ahead_capped", "steps_over",
                                                      "steps_gt48", "max_rows", "decode_steps" };
        o += ", \"cap\": {\"in_force\": " + std::to_string(reported(s, kSchedRowCap));
        for (int k = 0; k < kCapCounts; ++k)
            o += std::string(", \"") + kCap[k] + "\": " + std::to_string((long long)s.cap[k].load(std::memory_order_relaxed));
        o += "}";
    }
    /* Lever CBH's counts, once a scheduler registered (sched_register). */
    if (s.max_tok.load(std::memory_order_acquire) > 0) {
        static const char* const kCbh[kCbhCounts] = { "holds", "by_k", "by_timeout", "by_idle", "by_other", "hold_ms",
                                                      "held_steps", "cob_steps", "cob_newcomers", "single_steps" };
        o += ", \"cbh\": {\"in_force_ms\": " + std::to_string(reported(s, kSchedCbHoldMs)) +
             ", \"k\": " + std::to_string(reported(s, kSchedCbHoldK));
        for (int k = 0; k < kCbhCounts; ++k)
            o += std::string(", \"") + kCbh[k] + "\": " + std::to_string((long long)s.cbh[k].load(std::memory_order_relaxed));
        o += "}";
    }
    /* The depth-0 context skip, once the engine registered it (set up or not). */
    if (const int cap = s.ctxskip_capable.load(std::memory_order_acquire); cap >= 0) {
        static const char* const kCs[8] = { "skipped_steps", "refill_passes", "refill_rows", "dropped_rows",
                                            "ring_full_refills", "pending_rows", "ring_slots", "ring_mib" };
        o += ", \"ctxskip\": {\"capable\": " + std::to_string(cap);
        for (int k = 0; k < 8; ++k)
            o += std::string(", \"") + kCs[k] + "\": " +
                 std::to_string((long long)s.ctxs[k].load(std::memory_order_relaxed));
        o += "}";
    }
    o += "}";
    return o;
}

}  // namespace fn4
}  // namespace rad

/* rad_runtime.h: a plugin reads a switch's applied value at issue. */
extern "C" int rad_fn4_get(const char* name, int* value) {
    if (!name) return 0;
    const int i = rad::fn4::switch_index(name);
    if (i < 0) return 0;
    if (value) *value = rad::fn4::value(i);
    return 1;
}
