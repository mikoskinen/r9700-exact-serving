/* aql.h -- declarations shared by the AQL backend's translation units.
 *
 * aql_code.cpp reads code objects (offload bundle, ELF note, msgpack metadata), aql.cpp owns the
 * queues and all submission, and aql_blit.hip holds the backend's own copy and fill kernels.
 * aql_blit.hip is compiled as HIP, so this header must not pull in HSA.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace rad {
namespace aql {

/* ------------------------------------------------------------------ code objects */

/* One kernel argument as laid out in the code object's metadata. `hidden` names the implicit
 * arguments the compiler appends; explicit parameters map in order onto hipLaunchKernel's
 * `kernelParams`. */
enum class Hidden : uint8_t {
    None,           /* an explicit parameter */
    Pad,            /* hidden_none: bytes the compiler reserved and reads nothing from */
    BlockCountX, BlockCountY, BlockCountZ,
    GroupSizeX, GroupSizeY, GroupSizeZ,
    RemainderX, RemainderY, RemainderZ,
    GlobalOffsetX, GlobalOffsetY, GlobalOffsetZ,
    GridDims,
    DynamicLds,
    Unsupported,    /* anything else: a kernel that wants it is refused by name */
};

struct KernelArg {
    uint32_t offset = 0;
    uint32_t size   = 0;
    Hidden   hidden = Hidden::None;
    std::string kind;              /* the metadata's value_kind, for the refusal message */
};

struct KernelMeta {
    std::string symbol;            /* "<name>.kd" */
    uint32_t kernarg_size = 0;
    uint32_t group_size   = 0;     /* static LDS */
    uint32_t private_size = 0;     /* static scratch */
    bool     dynamic_stack = false;
    std::vector<KernelArg> args;
};

/* Selects the card's code object from a clang offload bundle, or fails with `*why` naming what
 * the bundle holds and what the card accepts. `isas` lists the card's accepted targets in the
 * runtime's order of preference ("amdgcn-amd-amdhsa--gfx1036", then the family's
 * "amdgcn-amd-amdhsa--gfx10-3-generic"), each with its modes ("gfx942:sramecc+:xnack-"). The
 * bundle has no total length, so it is walked entry by entry. Compressed bundles are refused;
 * nothing in this tree builds one. */
bool bundle_code_object(const void* bundle, const std::vector<std::string>& isas, const void** out,
                        size_t* out_size, std::string* why);

/* Every kernel described in an AMDGPU code object's metadata note. */
bool code_object_kernels(const void* elf, size_t size, std::vector<KernelMeta>* out,
                         std::string* why);

/* ------------------------------------------------------------------ fence policy
 *
 * Extra fence scope for the next dispatch on this thread, beyond the agent scope every kernel
 * gets. Set by the backend around its own blits; cleared by the dispatch that consumes it. */
/* kFenceAside (tp8 loop W1, lever WAITK's wait kernel): a packet that only schedules. It takes the agent-scope acquire
 * every live dispatch keeps, but not the acquire bookkeeping (the system acquire owed after a host wait or a lap stays
 * with the next real packet, which then gets exactly the scope it would have got without the aside packet), and it
 * never completes a pending rad_ride (the next real dispatch does, as without it). */
enum : uint8_t { kFenceAcquireSystem = 1, kFenceReleaseSystem = 2, kFenceAside = 4 };
void set_next_fence(uint8_t f);

/* ------------------------------------------------------------------ lever WAITK
 *
 * RADIANCE_AQL_WAITK (DRAFTER track round 49; reviewed + made exact-by-construction in tp8 loop W1): a wait whose
 * signal is still pending when it is enqueued gets a one-wave wait kernel (aql_blit.hip) on the waiting stream ahead of
 * its unchanged barrier-AND, so the packet processor never holds a pending barrier (one costs every other queue of its
 * CP pipe ~+40 us a packet, FINDINGS DRAFTER note 00:18 21.11.). The start value: unset / empty / 0 / negative / not a
 * number = off, 1 = event waits (rad_event_wait), 2 or more = event + ride waits (rad_ride_wait). In process: fn4 aql_waitk
 * (switch 106, tp8 loop W1s) 0 / 1 / 2 replaces the start value from the step it is applied at. */
inline int waitk_mode_of(const char* env) {
    if (!env || !*env) return 0;
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end == env || v <= 0) return 0;
    return v >= 2 ? 2 : 1;
}
/* What one wait does: kWaitkOff = only the barrier, as without the lever (the mode does not cover this kind of wait,
 * `need` 1 an event wait / 2 a ride wait, or this thread is recording a tape: a recorded wait kernel would poll a
 * retired signal at play, and a played tape's waits go through rad_event_wait again, which adds the kernel then);
 * kWaitkMet = only the barrier (the signal `value` read at enqueue is 0 already); kWaitkKernel = the wait kernel, then
 * the barrier. */
enum : int { kWaitkOff = 0, kWaitkMet = 1, kWaitkKernel = 2 };
inline int waitk_plan(int mode, int need, bool capturing, int64_t value) {
    if (mode < need || capturing) return kWaitkOff;
    return value == 0 ? kWaitkMet : kWaitkKernel;
}
/* Since start: wait kernels issued, waits met at enqueue, waits the mode covers but a capture kept plain. */
struct WaitkCounts { int64_t kernels = 0, met = 0, captured = 0; };
WaitkCounts waitk_counts();

/* ------------------------------------------------------------------ test hooks */

/* Whether the signal behind a rad_ride token could be handed out again now (tests/device_test.cpp). */
bool ride_reusable(void* token);
/* tests/device_test.cpp: replaces RADIANCE_AQL_WAITK's start value for this process; -1 = the start value again. */
void waitk_set_for_test(int mode);
/* The mode the next wait runs at: the test value, else fn4 aql_waitk once set (tp8 loop W1s), else RADIANCE_AQL_WAITK's start value
 * (0 in a build without the AQL backend). */
int waitk_mode_now();
/* Whether lever IKA plays its marks now = PlayRun's `ika_live` (RADIANCE_AQL_IKA=1 at start and fn4 aql_ika >= 1); false in a build
 * without the AQL backend. */
bool ika_live_now();
/* Lever IKB (DRAFTER round 53): fn4 aql_ika 2 (with the env) -- core's copy / fill launchers (aql_blit.hip) pick their self-acquiring
 * forms only then, so aql_ika 0 and 1 launch the plain kernels (0 = the packets without either lever, 1 = lever IKA alone; a change
 * of the switch re-records). false in a build without the AQL backend. */
bool ikb_live_now();
/* Lever CBL (DRAFTER round 64): fn4 aql_ika 3 with lever CIK on (RADIANCE_AQL_CIK=1 at start) -- core's copy / fill launchers pick their
 * chain-capable forms (aql_blit.hip *_cik), which the backend chains as it chains CIK kernels. false in a build without the AQL backend. */
bool cbl_live_now();
/* Lever CRD (DRAFTER round 64): replaces RADIANCE_AQL_CRD's start value (the chained run cap, 0 = none) for this process; -1 = the start
 * value again. A bench / test hook. */
void crd_set_for_test(int cap);

/* ------------------------------------------------------------------ lever IKA: which kernels a module names
 *
 * RADIANCE_AQL_IKA=1 (DRAFTER track round 50; core reviewed in tp8 loop W3): a kernel is self-acquiring when the shared
 * object holding its code lists its identifier in `r4d_ika_kernels` (aql.cpp ika_kernel). The list holds bare
 * identifiers; the symbol is the Itanium-mangled name, so the match is on the kernel's OWN identifier: the source name
 * right after `_Z` (`_Z19exl3r4d_gemm_kernelILi..`), or the last source name of a nested name before its template
 * arguments or its end (`_ZN12_GLOBAL__N_113binary_kernelEPKt..` -> binary_kernel: libr4d's kernels sit in anonymous
 * namespaces). Namespace components and template arguments never match (`_Z6kernelI13binary_kernelEvv` is `kernel`).
 * A name that does not start with `_Z` is an extern "C" kernel: its symbol is its identifier. Anything this does not
 * parse gives "" -- such a kernel is never treated as self-acquiring (both packet fences stay). */
inline std::string ika_identifier(const std::string& sym) {
    if (sym.compare(0, 2, "_Z") != 0) return sym;
    size_t p = 2;
    auto source_name = [&](std::string* out) {   /* <number><identifier> at p */
        size_t q = p, n = 0;
        while (q < sym.size() && sym[q] >= '0' && sym[q] <= '9') n = n * 10 + (size_t)(sym[q++] - '0');
        if (q == p || n == 0 || q + n > sym.size()) return false;
        *out = sym.substr(q, n);
        p = q + n;
        return true;
    };
    auto abi_tags = [&] {                         /* B<source-name>... after a name */
        std::string tag;
        while (p < sym.size() && sym[p] == 'B') { ++p; if (!source_name(&tag)) return false; }
        return true;
    };
    if (p < sym.size() && sym[p] == 'L') ++p;     /* internal linkage (a static kernel) */
    std::string id;
    if (p < sym.size() && sym[p] == 'N') {        /* N [CV] [ref] <prefix>* <unqualified-name> E */
        ++p;
        while (p < sym.size() && (sym[p] == 'r' || sym[p] == 'V' || sym[p] == 'K')) ++p;
        if (p < sym.size() && (sym[p] == 'R' || sym[p] == 'O')) ++p;
        if (sym.compare(p, 2, "St") == 0) p += 2; /* std:: */
        bool any = false;
        while (p < sym.size()) {
            if (sym[p] == 'E' || sym[p] == 'I') return any ? id : std::string();
            if (sym[p] == 'L') { ++p; continue; } /* internal linkage inside a nested name */
            if (!source_name(&id) || !abi_tags()) return std::string();
            any = true;
        }
        return std::string();
    }
    if (sym.compare(p, 2, "St") == 0) p += 2;
    if (!source_name(&id) || !abi_tags()) return std::string();
    return id;
}
/* Lever IKA at play (tp8 loop W3s: fn4 aql_ika = switch 107). A recording makes the marks of both forms (Ctx::haz_mark): `noacq`
 * the acquire rule's own (its windows as without the lever), `noacq_ika` the rule's under IKA's windows (a dropped IKA acquire
 * does not start a new window), `ikaok` IKA's own drop of a self-acquiring (`ika`) record. `live` = the switch at 1 (which needs
 * RADIANCE_AQL_IKA=1 at start). Whether the record's agent-scope acquire may be dropped (the caller adds the rule's other
 * conditions: an unchained record, not behind foreign packets, the rule on): live, IKA's marks; else the rule's own, exactly as
 * without the lever. */
inline bool ika_play_mark(bool live, bool noacq, bool noacq_ika, bool ika, bool ikaok) {
    return live ? (noacq_ika || (ika && ikaok)) : noacq;
}
/* Whether a record's AGENT-scope release may be dropped: live only, a self-acquiring kernel's only (it released its writes before
 * its waves ended), and only on a packet no record or ride completes (that signal orders other queues / the host after it). */
inline bool ika_play_release_none(bool live, bool ika, bool completes) { return live && ika && !completes; }

/* tp8 loop W37 (DECODE round 68, review of the DRAFTER's lever CKO, round 105): how ONE played record goes out on its queue --
 * PlayRun::put's ordering decision as a pure function, so the host can drive tapes through it (tests/device_test). `q` = the queue's
 * state before the record (Stream's fields of the same names), `r` the record's facts, `k` the switches as put() reads them.
 *   chain    the record plays chained (its chain record on, from the queue's argument slot)
 *   ordered  its packet keeps the barrier bit (starts behind everything before it on the queue)
 *   target   its chained wait's target: 0 = none (an ordered record under CIK), else the counter value it polls for
 * Lever CIK plays a recorded-ORDERED chained record behind a chained one unordered (it waits in-kernel for the chained lanes before
 * it); lever CKO keeps the recording's proof behind such a record (`ahead`): an overlap record (proved independent of the packets
 * from the last ordered record on, NOT of those before it) chains to the counter before that record (`ahead_base`) or, where it
 * cannot wait in-kernel, keeps its barrier bit; play_keep_acquire (b) keeps the first barrier-bit packet's acquire behind it. */
struct PlayQ {
    bool     prev_chained = false, plain_inflight = false, ahead = false;
    uint64_t ahead_base = 0, chain_issued = 0;
    int      chain_run = 0;
    uint64_t chain_run_blocks = 0;
};
struct PlayRec {
    bool     chain_ok = false;  /* a chain-capable record with a tape, the queue's counter, room for the record in its arguments */
    bool     lanes = false;     /* a lever CIK record (adds its lanes; aql_cik 0 plays it unchained) */
    bool     overlap = false;   /* Ctx::haz_mark: touches nothing in flight since the last ordered record */
    bool     foreign = false;   /* the first packet behind packets this thread did not play */
    bool     done = false;      /* carries a completion signal (a record or a ride) */
    uint64_t adds = 0;          /* what it adds to the counter when chained: lanes (CIK) or blocks */
    uint64_t blocks = 0;        /* its blocks (lever CRB's measure) */
};
struct PlayKnobs {
    bool     cik_live = false;  /* lever CIK at play (RADIANCE_AQL_CIK=1 and fn4 aql_cik >= 1) */
    bool     cik_on = false;    /* RADIANCE_AQL_CIK=1 (an ordered chained record names target 0) */
    int      crd = 0;           /* lever CRD's cap (0 = none) */
    uint64_t crb = 0;           /* lever CRB's cap (0 = none) */
    uint64_t cik_max = UINT64_MAX;  /* lever CKW (DRAFTER round 108): fn4 aql_cik N >= 2 -- a CIK record of more than N lanes plays
                                     * unchained (as the IKA kernel it also is); RADIANCE_AQL_CIK_LANES's bound in process */
};
struct PlayOut {
    bool     chain = false, ordered = false;
    bool     ahead_chain = false;  /* lever CKO (a): chained to ahead_base */
    bool     cko_order = false;    /* lever CKO (a): an overlap record that kept its bit behind a record that went ahead */
    uint64_t target = 0;
};
inline PlayOut play_order(const PlayQ& q, const PlayRec& r, const PlayKnobs& k) {
    PlayOut o;
    if (r.chain_ok) {
        o.chain = true;
        if (r.lanes && !k.cik_live) o.chain = false;   /* tp8 loop W9: aql_cik 0 plays a CIK record as the IKA kernel it also is */
        if (r.lanes && r.adds > k.cik_max) o.chain = false;   /* lever CKW: wider than aql_cik N lanes -> as an IKA kernel */
        if (o.chain && r.overlap && q.ahead && !r.foreign && !r.done) o.ahead_chain = true;
        else if (r.lanes && r.overlap && !r.foreign && !r.done) o.chain = false;   /* lever CIK: an overlap record plays unchained */
    }
    o.ordered = r.foreign || !r.overlap || r.done;
    if (!o.ordered && q.ahead && !o.ahead_chain) { o.ordered = true; o.cko_order = true; }
    if (o.chain && q.prev_chained && !q.plain_inflight && !r.foreign && !r.done) o.ordered = false;
    if (o.chain && !o.ordered && k.crd && q.chain_run >= k.crd) o.ordered = true;
    if (o.chain && !o.ordered && k.crb && q.chain_run_blocks + r.blocks > k.crb) o.ordered = true;
    o.target = (o.ordered && k.cik_on) ? 0 : (o.ahead_chain && !o.ordered) ? q.ahead_base : q.chain_issued;
    return o;
}
/* The queue's state after the record went out as `o` (PlayRun::put's tail). */
inline void play_advance(PlayQ& q, const PlayRec& r, const PlayOut& o, const PlayKnobs& k) {
    if (o.ordered) q.ahead = false;
    else if (o.chain && !r.overlap) { q.ahead = true; q.ahead_base = q.chain_issued; }
    if (o.chain) q.chain_issued += r.adds;
    q.prev_chained = o.chain;
    q.plain_inflight = o.ordered ? !o.chain : q.plain_inflight || !o.chain;
    q.chain_run = o.ordered ? 0 : q.chain_run + 1;
    q.chain_run_blocks = o.ordered ? 0 : q.chain_run_blocks + (k.crb ? r.blocks : 0);
}
/* Lever CKO (b): a barrier-bit packet behind a record that went ahead keeps an agent-scope acquire the IKA / rule marks dropped
 * (unless RADIANCE_AQL_ACQ=none, the unsafe experiment): it runs after everything before it ended, the invalidation the recording's
 * acquire-rule window took the ahead record's acquire for. Returns whether the dropped acquire comes back. */
inline bool play_keep_acquire(bool ordered, bool ahead, bool acq_dropped, bool acq_none_mode) {
    return ordered && ahead && acq_dropped && !acq_none_mode;
}

/* tp8 loop R48 (PREFILL round 48) -- lever CIK's width bound. A chain-capable IKA kernel (libr4d R4D_IKA_CHAIN, DRAFTER round
 * 62) adds to its queue's ONE chain counter at every lane's exit: one atomic add a wave, all on one address. A decode dispatch
 * is a few thousand waves; a prefill chunk's per-row kernels are hundreds of thousands (quant_group_kernel over 4,096 rows x
 * 5,120 columns: 21 M work-items = 655 k wave adds, its QG8 form 82 k), which the L2 serialises at the counter's line.
 * RADIANCE_AQL_CIK_LANES=N (start env, read only while CIK is on): a CIK kernel's dispatch of more than N work-items goes out
 * with its record OFF -- it plays as lever IKA's self-acquiring kernel (ordered, no wait, no add), the same kernel and bytes.
 * Unset / empty / 0 / negative / not a number = no bound (every CIK dispatch chains: the DRAFTER's form). */
inline uint64_t cik_lanes_of(const char* env) {
    if (!env || !*env) return UINT64_MAX;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    if (end == env || v <= 0) return UINT64_MAX;
    return (uint64_t)v;
}
/* Whether a dispatch of `lanes` work-items of a kernel that chains under CIK (`cik`) is kept unchained by the bound `max`. A named
 * chained form (r4dchain*, its own block-count done: `cik` false) is never bounded. */
inline bool cik_too_wide(bool cik, uint64_t lanes, uint64_t max) { return cik && lanes > max; }

/* Lever CIK-X (tp8 loop R49c, PREFILL round 49): whether a kernel is chain-capable under lever CIK. A named chained form (r4dchain*:
 * its own block-count done) never is; an IKA-listed kernel of a CHAIN module (r4d_ika_chain() = 1) is unless the module also lists
 * it in r4d_ika_nochain_kernels() (libr4d r4d_ika_x.h / r4d_ika_nochain.inc: built with IKA's plain entry in a CIK-X build, so it
 * neither waits for a record nor adds to the counter -- it must be issued / played as a plain, ordered IKA kernel); core's listed
 * blits are (lever CBL). A module without the nochain export lists nothing: today's decision. */
inline bool cik_capable(bool named_chained, bool chain_module, bool ika_listed_here, bool nochain_listed, bool core_cik) {
    return !named_chained && ((chain_module && ika_listed_here && !nochain_listed) || core_cik);
}

/* Whether `sym`'s own identifier is one of the `n` names of `list` (null / empty entries skipped). */
inline bool ika_listed(const std::string& sym, const char* const* list, int n) {
    const std::string id = ika_identifier(sym);
    if (id.empty() || !list) return false;
    for (int i = 0; i < n; ++i)
        if (list[i] && *list[i] && id == list[i]) return true;
    return false;
}

/* Lever CTA (DRAFTER track round 63; a pure function since tp8 loop W8, so the host checks the layout): the places, from the tape
 * copy's CTA base, of the blocks of `n` chained records with argument lengths `len[i]` (each <= `rec_off`): a group's blocks start
 * 64 bytes apart, every block's arguments end by `rec_off` past its group's start and its record (`rec_bytes`) sits at its own
 * place + `rec_off`, so no block's arguments meet a record; a full group's successor starts past its last record. Returns the end
 * of the last record (0 for none). */
inline size_t cta_pack(const uint32_t* len, size_t n, size_t rec_off, size_t rec_bytes, size_t* off) {
    size_t g = 0, o = 0, end = 0;   /* the group's start, the next block's place in it */
    for (size_t i = 0; i < n; ++i) {
        const size_t st = ((size_t)len[i] + 63) & ~size_t(63);
        if (o && o + st > rec_off) {   /* the group is full: the next starts past its last record */
            g = (g + o + rec_off + 63) & ~size_t(63);
            o = 0;
        }
        off[i] = g + o;
        if (g + o + rec_off + rec_bytes > end) end = g + o + rec_off + rec_bytes;
        o += st ? st : 64;
    }
    return end;
}

}  // namespace aql
}  // namespace rad

struct RadCopyRange;

/* Blit kernel launchers, defined in aql_blit.hip. Each issues one kernel onto `stream` through
 * hipLaunchKernel, which the backend intercepts. The caller guarantees alignment: the 16-byte
 * forms need both addresses and the length 16-byte aligned. */
extern "C" {
/* `link`: either side is pinned host memory, so use one narrow workgroup instead of a full grid
 * (see aql_blit.hip). */
int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream, bool link = false);
int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                        uint64_t width, uint64_t height, void* stream);
int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream);
/* Lever WAITK: one wave polls the int64 at `value` (an amd_signal_t's value) until it reads 0. */
int rad_aql_blit_wait(const void* value, void* stream);
/* rad_memcpy_ranges_async's kernel: `prefix` a multiple of 16, both bases 16-byte aligned. */
int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix, const RadCopyRange* ranges,
                        int n, void* stream);
/* Lever IKA (DRAFTER round 51): the self-acquiring copy / fill forms of aql_blit.hip by identifier, the list core/device/aql.cpp
 * ika_kernel reads for a fat binary of core's own object (libr4d's is r4d_ika_kernels). */
const char* const* rad_aql_blit_ika_kernels(int* n);
/* Lever CBL (DRAFTER round 64): the chain-capable copy / fill forms of aql_blit.hip by identifier (each also in the list above). */
const char* const* rad_aql_blit_cik_kernels(int* n);
}
