/* heat.h -- the heat engine: which conditional weights should be in VRAM, decided from the
 * router's observed choices.
 *
 * Layer offload gets an exact prefetch schedule, because declare enumerates the access sequence.
 * Routing (RAD_ACCESS_CONDITIONAL) is data-dependent, so expert offload needs a policy instead.
 * This file is that policy: from a tier and a heat per unit and a dispatch count, it proposes
 * swaps. It is pure host code with no device calls and no knowledge of slabs, streams or events
 * (that is mover.cpp), so the decision half, hysteresis included, is testable without a card.
 *
 * ---- one step behind ---------------------------------------------------------------------------
 *
 * Routing runs on device and never blocks the compute stream. The histogram is copied back
 * asynchronously and consumed on the next step (spec §5.5), so every decision is one step stale.
 * Ranking is unaffected, since a hot expert stays hot across many steps. But an expert that is not
 * resident this step cannot be fetched in time: it is served from where it lies, streamed from the
 * pinned pool or staged off the file tier. A last-moment fetch would have to fit between the
 * router's output and the expert GEMM's first block, which is microseconds. Hit rate is therefore
 * a property of the placement policy, and the run phase has no device round-trip.
 *
 * ---- why LFU with decay ------------------------------------------------------------------------
 *
 * In routing-trace replays, runtime placement beats static profile placement by a wide margin, and
 * the move rate bounds it, not the ranking: at the same swaps a step, an oracle that knows every
 * future routing scores no better (78.0% vs 78.4% at 64 swaps a step, Qwen3.8-Flash-Next TP1, 40%
 * of experts resident). Unrestricted Belady reaches 92% only by admitting every miss at once, which
 * here would copy again bytes the GEMM just streamed. The levers are the move budget
 * (RAD_PLACE_MOVES_PER_STEP), the slots behind it, and keeping a prompt's routing from swamping the
 * answer's (HeatParams::credit_cap), not the decay constant or the guards.
 */
#pragma once
#include "planner.h"

#include <string>
#include <atomic>
#include <memory>
#include <vector>

namespace rad {

struct HeatParams {
    /* Applied to a layer's whole plane each time its histogram is credited. Every layer is
     * credited once a step, so planes decay in lockstep and heats stay comparable across layers,
     * which the global ranking needs. 0.95 is a half-life of about 14 steps. In replay at 64 swaps
     * a step, hit rate is flat from 0.8 to 0.95 and falls above it (77.1% at 0.98, 74.2% at 0.99):
     * a longer memory ranks experts the conversation has moved past above the current ones. */
    float decay = 0.95f;

    /* Promote only when the candidate beats the victim by this ratio, so the two ends of the
     * ranking do not trade places on noise. */
    float hysteresis = 1.25f;

    /* The additive half of the guard, in activations. It stops a swap against a zero-heat victim,
     * where any ratio passes; right after warmup that is most of the resident set.
     *
     * Do not raise it to save link traffic: traffic is U-shaped in this threshold. A declined
     * promotion comes back as a miss, streamed over the same link, so past the minimum a tighter
     * bar loses hit rate and saves no bytes. A swap costs its promotion: the link is near full
     * duplex (R9700: 28.5 GB/s up and 25.0 down at once, 28.8 up alone), so a demotion rides the
     * idle direction while a promotion shares the upstream with every miss. At the default budget,
     * lower values replay within 0.2 points of hit rate and promote a fifth more. */
    float min_gain = 1.5f;

    /* Swaps step() may propose per slab class each step, after every layer is credited. The value
     * and its reason are at RAD_PLACE_MOVES_PER_STEP, which also sizes the free slots a swap lands
     * in (shadow and turnover slots); a budget raised here without them is spent on
     * `starved_slab`. Promotions share the link with streamed misses, but each swap saves many
     * times its own bytes in misses.
     *
     * Per class because a rank holding an uneven slice of every expert has two classes of about
     * half an expert each; a shared budget would make residency converge half as fast. */
    int moves_per_step = RAD_PLACE_MOVES_PER_STEP;

    /* The most one histogram credits, in selections; a larger histogram is scaled to this total.
     * Decode routes a few rows a sequence and stays under it. A prefill chunk routes thousands
     * and would otherwise make the resident set the prompt's for dozens of steps, while prefill
     * gains little from residency (a chunk reads nearly every expert; the stager copies the rest).
     * Mixed-workload replay, Qwen3.8-Flash-Next TP1: decode hit rate 77.3% uncapped, 78.9% at
     * 1024, 79.0% at 512 or 256. */
    float credit_cap = 512.0f;

    /* How much of a step a histogram from a step with prefill rows counts as, 0..1: its decay and
     * its credit are both scaled to that share (observe's `share`), so 0 leaves the heat exactly
     * as the decode steps around the prompt left it and 1 is the step every histogram counted as
     * before (bit for bit). The engine gives a step of `n` rows, `d` of them decode rows, the share
     * (d + prefill_share * (n - d)) / n; a draft head's layers keep 1.
     *
     * Why: the cap bounds one chunk, not a prompt. Under expert parallelism a rank sees about
     * 8192 x top-k / world selections of a chunk, always over the cap, and a decode step about
     * top-k / world, so each chunk credits several hundred decode steps' worth and decays the
     * decode heat once more. DeepSeek-V4.1-Flash EXL3 at TP4 (6.11.): after a 243k-token cold
     * prefill (30 chunks) the routed experts ran 8.3 ms a step instead of 5.0 and the MoE
     * all-reduce waited 10.7 ms instead of 5.0, recovering over ~200 decode steps as the moves
     * put the decode set back. Placement never changes an output (the same kernel reads the same
     * bytes from whichever tier holds them), so the share is exact. RADIANCE_HEAT_PREFILL. */
    float prefill_share = 1.0f;

    /* Dual heat (on when 0 < prefill_decay < 1): a decode heat and a prefill heat per unit, ranked
     * by their sum. A step's decode rows credit the decode heat, which decays with them as one heat
     * does (decay^decode share, so a prefill step leaves it as it was); its prefill rows credit the
     * prefill heat (times prefill_gain), and the prefill heat decays by prefill_decay at every
     * credited step, prefill or decode. During a prompt its chunks keep the prefill heat high and
     * the ranking follows the prompt (its experts come in, as with one heat: 6.11. EXL3 TP4 at 47
     * flags, prefill share 0 cost 31 % of a 266k prompt's prefill rate); after it, the prefill heat
     * is gone within tens of steps (0.7^12 = 1.4 %) and the ranking is the decode heat the prompt
     * left untouched, not one the prompt's chunks decayed under their own credit (one heat at
     * share 1: 28.6 ms/step decoding right after that prompt against 23.5 at share 0).
     * prefill_share is not read in this mode. RADIANCE_HEAT_PREFILL_DECAY, _PREFILL_GAIN. */
    float prefill_decay = 0.0f;
    float prefill_gain = 1.0f;

    /* Dual heat only: the fraction of a slab class's resident units, the hottest by DECODE heat,
     * that no promotion may demote (0 = none). A prompt then fills the rest of the resident set
     * and the decode core stays on the card through it, where unprotected it is evicted wholesale
     * and comes back only as fast as its units' decode heat clears min_gain (at TP4 one token
     * credits ~1.5 selections a rank, so most of a decode set sits below 1.5: 6.11., 26.8 ms/step
     * decoding after a 266k prompt against 23.5 with no prompt credit). Thresholds are taken once
     * a step (step()), so a unit's protection is one step stale. A unit with no decode heat is never
     * protected. RADIANCE_HEAT_DECODE_PIN. */
    float decode_pin = 0.0f;

    /* Dispatches to watch before moving anything. 0, because min_gain already requires
     * `hot >= cold*hysteresis + 1.5`, which against a zero-heat victim is two tokens agreeing in
     * one dispatch; a warmup only delays convergence. */
    int warmup_dispatches = 0;

    /* The hot VRAM cache's admission bar: the heat at which the file stager
     * may read a routed expert into a slab slot permanently instead of its per-pass buffer.
     * Absolute, not relative to a victim: admission happens when the unit is read anyway, and the
     * shadow slots are the budget. A file-tier miss is a pread plus a relayout every token until
     * the expert stays, so anything routed in the last few steps is worth a slot: 0.5 survives
     * one decay of a single credit. Eviction is free (the container is ground truth), so a
     * mistake costs one future re-read, and the slot count — not this bar — bounds the cache. */
    float admit_heat = 0.5f;

    /* Lever LF (ds41-lf): the long-memory decode heat's decay a credited step. A second decode heat kept
     * beside the one placement ranks, credited and decayed on the same pass (a decode share decays it by
     * long_decay^share), only slower: 0.998 is a half-life of ~350 steps, so it ranks an expert by how often
     * decode routed it over the last few hundred steps, not the last ~14. It never moves anything by itself;
     * the mover's fill of lent slots ranks by it under fn4 place_lf 1 (Mover::fill). Outside (0, 1): the
     * default. RADIANCE_PLACE_LF_DECAY. */
    float long_decay = 0.998f;

    /* The trunk's layer count: units of layers at or past it (a draft head's) are counted apart in the
     * slot-class counters (kSlotDraftOwn), whatever slot holds them. -1: every layer is the trunk's. */
    int32_t trunk_layers = -1;

    /* Lever HB (ds41-sahb, fn4 place_hb): the long-memory prefill heat's decay, applied once a credited layer
     * histogram of a step with prefill rows (dual heat only; a decode step leaves it). 0.8 weighs the last
     * prompts 1, 0.8, 0.64, ...: the experts the recent prompts keep routing. Outside (0, 1): the default.
     * RADIANCE_PLACE_HB_DECAY. */
    float hb_decay = 0.8f;
};

/* Slot classes (ds41-lf, /stats experts.ranks[].slots): the slot a unit sits in while in VRAM, kept after it
 * leaves (a miss is counted under the class of the slot its expert last sat in). kSlotNone: never in VRAM since
 * start; kSlotTrunk: a budgeted slab slot or the static region (the plan's resident set and every swap's
 * promotion); kSlotKV .. kSlotDraftW: a slot lent by that lender (Mover::kLendKV + 2 ..); kSlotDraftOwn: a draft
 * head's own units, wherever they sit (HeatParams::trunk_layers). */
enum SlotClass : uint8_t { kSlotNone = 0, kSlotTrunk = 1, kSlotKV = 2, kSlotArena = 3, kSlotDraft = 4,
                           kSlotDraftW = 5, kSlotDraftOwn = 6, kSlotClasses = 7 };
inline const char* slot_class_name(int c) {
    static const char* const k[kSlotClasses] = { "cold", "trunk", "kv", "arena", "dslab", "dw", "draft" };
    return c >= 0 && c < kSlotClasses ? k[c] : "?";
}
/* Routed experts by slot class, counted on observe's pricing pass with the same weight as HeatStats::routed:
 * `hit[c]` resident in a class-c slot, `miss[c]` off the device with class c its last slot. */
struct SlotStats { uint64_t hit[kSlotClasses] = {}, miss[kSlotClasses] = {}; };

/* The share of a step (HeatParams::prefill_share) a histogram of a step with `n_tok` rows,
 * `n_tok_decode` of them decode rows, is credited at: the decode rows in full, the prefill rows at
 * `prefill_share`, over the rows. 1 for a pure decode step, `prefill_share` for a pure prefill one,
 * and 1 whenever the share is 1 or the rows are unknown. */
inline float heat_step_share(float prefill_share, int64_t n_tok, int64_t n_tok_decode) {
    if (!(prefill_share < 1.0f) || n_tok <= 0 || n_tok_decode >= n_tok) return 1.0f;
    const double d = n_tok_decode > 0 ? (double)n_tok_decode : 0.0, n = (double)n_tok;
    const double p = prefill_share > 0.0f ? (double)prefill_share : 0.0;
    return (float)((d + p * (n - d)) / n);
}

/* A promotion and its matching demotion. The mover issues two transfers, but they are decided
 * together: a promotion without its demotion would grow the resident set past the slab. */
struct Swap {
    int32_t promote = -1;   /* unit index */
    int32_t demote = -1;
};

/* One movable unit as step() ranks it: its heat, its layer's resident count, its index. */
struct HeatRanked { float h; int32_t rc; int32_t unit; };

struct HeatStats {
    uint64_t dispatches = 0;
    uint64_t proposed = 0, promotions = 0, demotions = 0;
    /* Why a dispatch moved nothing. `refused` is the guard declining, the healthy steady state.
     * `no_candidate` is an empty end of the ranking (everything movable is resident, or nothing
     * is), a budget question rather than a policy one. */
    uint64_t refused = 0, no_candidate = 0;
    /* Progress or thrash. `distinct_promoted` counts first-time promotions (still discovering the
     * working set); `readmits` counts promotions of units demoted earlier, link traffic spent to
     * return to where it started. Readmits say whether the guards are set right. */
    uint64_t distinct_promoted = 0, readmits = 0;
    /* Routed experts and how many were resident, counted on the histogram pass observe() already
     * makes. A miss does not stall: the GEMM reads the expert from its tier, which for a
     * host-pinned unit crosses the link on the step's critical path. `stream_bytes` is that
     * traffic, the cost promotions exist to avoid. Weighted by the histogram's `weight`, since a
     * draft head's histogram stands for several routing passes; the hit rate is unaffected. */
    uint64_t routed = 0, routed_resident = 0, stream_bytes = 0;
};

/* A counter one thread writes while another reads it (the HTTP thread's /stats and LiveView):
 * a relaxed atomic, so the reader sees a whole value from one side of a write or the other.
 * Copyable, a copy taking the current value, so a struct of them copies and resets by
 * assignment. */
template <class T>
class Relaxed {
public:
    Relaxed(T v = 0) : v_(v) {}
    Relaxed(const Relaxed& o) : v_(o.load()) {}
    Relaxed& operator=(const Relaxed& o) {
        v_.store(o.load(), std::memory_order_relaxed);
        return *this;
    }
    operator T() const { return load(); }
    T load() const { return v_.load(std::memory_order_relaxed); }
    Relaxed& operator++() { v_.fetch_add(1, std::memory_order_relaxed); return *this; }
    Relaxed& operator+=(T d) { v_.fetch_add(d, std::memory_order_relaxed); return *this; }

private:
    std::atomic<T> v_;
};

/* HeatStats as the engine counts it, readable from the HTTP thread; snapshot() is the plain copy
 * the readers take. */
struct HeatCounters {
    Relaxed<uint64_t> dispatches, proposed, promotions, demotions, refused, no_candidate,
                      distinct_promoted, readmits, routed, routed_resident, stream_bytes;

    HeatStats snapshot() const {
        HeatStats s;
        s.dispatches        = dispatches;
        s.proposed          = proposed;
        s.promotions        = promotions;
        s.demotions         = demotions;
        s.refused           = refused;
        s.no_candidate      = no_candidate;
        s.distinct_promoted = distinct_promoted;
        s.readmits          = readmits;
        s.routed            = routed;
        s.routed_resident   = routed_resident;
        s.stream_bytes      = stream_bytes;
        return s;
    }
};

/* The per-unit tier mirror HeatEngine keeps for the HTTP thread: one relaxed atomic byte per
 * unit, written by set_tier and read by tier_published. A type of its own because it must be
 * copyable -- HeatEngine is copyable, and a copy takes the tier values into an array of its
 * own -- which a bare unique_ptr of atomics is not. */
struct TierMirror {
    std::unique_ptr<std::atomic<uint8_t>[]> a;
    size_t n = 0;

    void assign(size_t m) {
        n = m;
        a.reset(m ? new std::atomic<uint8_t>[m] : nullptr);
    }
    void set(size_t i, uint8_t v) const { a[i].store(v, std::memory_order_relaxed); }
    uint8_t get(size_t i) const { return a[i].load(std::memory_order_relaxed); }

    TierMirror() = default;
    TierMirror(const TierMirror& o) {
        assign(o.n);
        for (size_t i = 0; i < n; ++i) set(i, o.get(i));
    }
    TierMirror& operator=(const TierMirror& o) {
        if (this == &o) return *this;
        assign(o.n);
        for (size_t i = 0; i < n; ++i) set(i, o.get(i));
        return *this;
    }
};

class HeatEngine {
public:
    /* Binds to the plan. A static plan is a mode, not a failure: init returns RAD_OK and
     * enabled() is false. Under Config::deterministic the plan is static and step() proposes
     * nothing for the whole run (spec §17). */
    int init(const Plan& plan, const HeatParams& params);
    bool enabled() const { return enabled_; }

    /* ---- input: last step's routing histogram ------------------------------------------------
     *
     * `counts[e]` is how many of the step's tokens routed to expert e of this layer. The credit
     * is the count, not 1: a dispatch reads an expert once however many tokens chose it, but an
     * expert several tokens chose is far likelier to be chosen again soon.
     *
     * `weight` is how many of the layer's routing passes this histogram stands for (see
     * Ctx::last_histogram_reports): 1 for a non-speculative step, n_spec for a draft head. It
     * scales the credit, not the decay, because every layer still decays exactly once a step.
     *
     * `share` is how much of a step the histogram's step counts as (HeatParams::prefill_share):
     * the decay is decay^share and the credit is scaled by share. 1 is the plain step, exactly as
     * without the parameter; 0 or less leaves every heat as it was and only prices the routing
     * (hit rate, streamed bytes), so the counters still see every step. `pshare` is read only in
     * dual-heat mode (HeatParams::prefill_decay): the step's prefill share. */
    void observe(int32_t layer, const int32_t* counts, int32_t n_expert, float weight = 1.0f,
                 float share = 1.0f, float pshare = 0.0f);
    /* Dual heat (HeatParams::prefill_decay): `share` is the step's decode share and `pshare` its
     * prefill share; see observe_dual. */
    bool dual() const { return params_.prefill_decay > 0.0f && params_.prefill_decay < 1.0f; }
    /* The same, sparse: distinct expert ids and their counts. `counts` may be null for one each. */
    void observe_ids(int32_t layer, const int32_t* experts, const int32_t* counts, int32_t n);
    /* Starts fetching a histogram observe() will read. A row the card copied into host memory
     * misses on every line, so a caller that knows its next rows asks early. */
    static void prefetch(const int32_t* counts, int32_t n_expert) {
        for (int32_t j = 0; j < n_expert; j += 64 / (int32_t)sizeof(int32_t))
            __builtin_prefetch(counts + j);
    }

    /* ---- tier map, written by the mover when a move publishes, not when it is issued ---------
     *
     * Until publish both copies are valid and dispatch reads the old one, so the tier here must
     * match dispatch's view. `set_busy` is what keeps an in-flight unit from being picked again. */
    void set_tier(int32_t unit, Tier t);
    void set_busy(int32_t unit, bool busy);
    Tier tier(int32_t unit) const { return tier_[(size_t)unit]; }
    /* The tier as the HTTP thread may safely read it: set_tier mirrors every change into a
     * relaxed atomic, and /stats and LiveView read the mirror. tier_ itself belongs to the
     * dispatch thread; reading it from another thread mid-step is a data race. */
    Tier tier_published(int32_t unit) const {
        return (Tier)tier_pub_.get((size_t)unit);
    }
    bool busy(int32_t unit) const { return busy_[(size_t)unit] != 0; }

    /* ---- the hot VRAM cache ----------------------------------------------
     *
     * A unit the file stager admitted keeps heat's history and, while resident, joins the
     * demotion ranking as an evictable victim — but never the promotion ranking: its admission
     * is the stager's own decision (it is read anyway; the slot is the only extra cost), and
     * once evicted it has no address to promote from (the container, read by the stager).
     *
     * set_evictable marks a unit the mover has admitted or may admit. admit_hot is the
     * admission test: heat past the bar and not already in flight. */
    void set_evictable(int32_t unit, bool evictable);
    /* A deposit's plan says Tier::SSD -- non-movable --
     * but its pool slot makes it a PCIe-speed promotion candidate all the same. The deposit
     * flips movability here, keeping the plan's word for everything else. */
    void set_movable(int32_t unit, bool movable);
    bool evictable(int32_t unit) const { return evictable_[(size_t)unit] != 0; }
    bool admit_hot(int32_t unit) const {
        return enabled_ && unit >= 0 && (size_t)unit < tier_.size() &&
               tier_[(size_t)unit] == Tier::SSD && !busy_[(size_t)unit] &&
               pos_[(size_t)unit] >= 0 &&
               heat_[(size_t)pos_[(size_t)unit]] >= params_.admit_heat;
    }

    /* The best swap available, or false when the guards refuse. */
    bool pick(Swap* out) const;

    /* One step's proposals, at most `moves_per_step` per class.
     *
     * Every unit in the result is marked busy. The caller must either issue the transfer
     * (set_tier() on publish, set_busy(false) after the quarantine) or abandon() the pair. A
     * dropped proposal leaves both units busy forever, with no counter to show it. */
    int step(std::vector<Swap>* out);
    void abandon(const Swap& s);

    float heat(int32_t unit) const { return heat_[(size_t)pos_[(size_t)unit]]; }
    /* Resident units in this unit's layer: the tie-break every residency decision needs (see
     * `res_count_`). `pick` uses it on an exact heat tie, and so must anything else that chooses. */
    int32_t layer_residents(int32_t unit) const {
        const int32_t l = (size_t)unit < layer_of_.size() ? layer_of_[(size_t)unit] : -1;
        return l >= 0 && (size_t)l < res_count_.size() ? res_count_[(size_t)l] : 0;
    }
    /* A snapshot by value: the counters are atomics read while the dispatch thread writes them.
     * Engine-internal decisions must use the individual counters (`warm`), not a snapshot. */
    HeatStats stats() const { return stats_.snapshot(); }

    /* ds41-lf. The slot class the mover published `unit` into (SlotClass; kept after it leaves VRAM). Called by
     * the mover on the dispatch thread when a unit lands in a slot; a draft head's unit keeps kSlotDraftOwn. */
    void set_slot_class(int32_t unit, uint8_t cls) {
        if (unit < 0 || (size_t)unit >= pos_.size() || cls >= kSlotClasses) return;
        uint8_t& c = scls_[(size_t)pos_[(size_t)unit]];
        if (c != kSlotDraftOwn) c = cls;
    }
    uint8_t slot_class(int32_t unit) const { return scls_[(size_t)pos_[(size_t)unit]]; }
    /* The long-memory decode heat (HeatParams::long_decay); lever LF's fill ranking. */
    float heat_long(int32_t unit) const { return heat_l_[(size_t)pos_[(size_t)unit]]; }
    /* ---- lever HB (ds41-sahb, fn4 place_hb): a reserve by the long-memory prefill heat -------------------------
     *
     * set_hb_reserve(R), R > 0, dual heat only: from the next step() on, the R units of each slab class with the
     * highest long-memory prefill heat (heat_pl_, HeatParams::hb_decay; never one at zero) rank kResBonus above
     * their heat -- above every unprotected resident and below every decode-pinned one (kPinBonus) -- so ordinary
     * swaps promote them (against the coldest unprotected residents) and none of them is a victim while reserved.
     * 0: no bonus (today's ranking; the heat itself is still kept, which changes no other heat). Placement only. */
    void set_hb_reserve(int32_t r) { hb_r_ = r > 0 && dual() ? r : 0; }
    int32_t hb_reserve() const { return hb_r_; }
    float heat_prefill_long(int32_t unit) const {
        return heat_pl_.empty() ? 0.0f : heat_pl_[(size_t)pos_[(size_t)unit]];
    }
    /* The reserve as the last step() set it: units holding it, of them resident; promotions it proposed (cumulative). */
    struct HbStats { int64_t reserved = 0, resident = 0, promotions = 0; };
    HbStats hb_stats() const { return HbStats{ hb_reserved_.load(), hb_resident_.load(), (int64_t)hb_promotions_.load() }; }
    /* By value, readable from the HTTP thread (relaxed counters). */
    SlotStats slot_stats() const {
        SlotStats s;
        for (int c = 0; c < kSlotClasses; ++c) { s.hit[c] = slot_hit_[c]; s.miss[c] = slot_miss_[c]; }
        return s;
    }

    /* Exit banner, ending with the per-layer resident counts, which show whether every layer kept
     * its share of slots. */
    std::string report() const;

private:
    HeatParams params_;
    bool       enabled_ = false;
    int32_t    n_layer_ = 0;
    HeatCounters stats_;
    /* ds41-lf: routed experts by slot class (slot_stats), and the class by position (set_slot_class). */
    Relaxed<uint64_t> slot_hit_[kSlotClasses], slot_miss_[kSlotClasses];
    std::vector<uint8_t> scls_;
    /* Lever LF: the long-memory decode heat by position (HeatParams::long_decay). */
    std::vector<float>   heat_l_;
    /* observe's tail: the slot-class tallies of one pass into the counters, `passes` each. */
    void add_slots(const uint32_t* hit, const uint32_t* miss, uint64_t passes) {
        for (int c = 0; c < kSlotClasses; ++c) {
            if (hit[c]) slot_hit_[c] += (uint64_t)hit[c] * passes;
            if (miss[c]) slot_miss_[c] += (uint64_t)miss[c] * passes;
        }
    }

    /* Heat by position, not by unit. Each layer owns a run of positions whose front is indexed by
     * expert: expert e of layer l is at slot_lo_[l] + e, whether or not the plan has a unit for it
     * or that unit may move. Then come the layer's other movable units, then padding to a multiple
     * of kLanes. After the layers: the movable units with no layer, padded to a screening block,
     * then everything else.
     *
     * This lines a histogram up with the positions it credits, so observe() walks counts[e] and
     * the heat side by side in fixed-width vector blocks, with no unit lookup or per-expert branch.
     * A position with no unit is a hole: never a candidate, never priced. `pos_` and `unit_at_`
     * translate (-1 for a hole); everything outside the position arrays is indexed by MoveUnit
     * index, as in the plan. */
    std::vector<float>   heat_;               /* by position */
    /* Dual heat only (HeatParams::prefill_decay): the decode and prefill heats by position; heat_
     * is their sum, which every ranking reads. Empty otherwise. */
    std::vector<float>   heat_d_, heat_p_;
    /* HeatParams::decode_pin: per slab class, the decode heat from which a resident unit is no
     * victim (+inf: none), and step()'s scratch for finding it. */
    std::vector<float>   pin_thr_;
    std::vector<std::vector<float>> pin_tmp_;
    void update_pin();
    /* Lever HB: the long-memory prefill heat by position (dual heat only), the reserve's size, and per class the
     * long prefill heat from which a unit holds it (+inf: none). */
    std::vector<float>   heat_pl_;
    int32_t              hb_r_ = 0;
    std::vector<float>   hb_thr_;
    std::vector<std::vector<float>> hb_tmp_;
    Relaxed<int64_t>     hb_reserved_, hb_resident_;
    Relaxed<uint64_t>    hb_promotions_;
    void update_reserve();
    void observe_dual(int32_t layer, const int32_t* counts, int32_t n, float w, uint64_t passes,
                      float share, float pshare);
    std::vector<int32_t> pos_;                /* unit -> position */
    std::vector<int32_t> unit_at_;            /* position -> unit, -1 for a hole */
    /* Per position, what step() ranks it as: 0 not a candidate (fixed, in flight, or a hole), 1 a
     * candidate off the device, 2 a candidate on it. Kept by set_tier and set_busy. */
    std::vector<uint8_t> rank_as_;
    /* Per position, what a routed read of it costs: `routed_as_` is 0 for a hole, 1 off the device
     * and 2 in VRAM, and `miss_bytes_` is the unit's bytes when it is off the device and 0 when it
     * is not. Kept by set_tier, so pricing a histogram is arithmetic on the pass that credits it. */
    std::vector<uint8_t> routed_as_;
    std::vector<int64_t> miss_bytes_;
    /* Slot s's positions are [slot_lo_[s], slot_lo_[s + 1]): layer s, and slot n_layer_ the
     * movable units with no layer. Every movable unit is below slot_lo_[n_layer_ + 1], which is a
     * whole number of screening blocks. The first n_ex_[l] positions of layer l are its experts. */
    std::vector<int32_t> slot_lo_;
    std::vector<int32_t> n_ex_;
    void set_place(int32_t unit);
    /* Unit bytes from the plan, so a miss is priced where it is detected. */
    std::vector<int64_t> bytes_;
    std::vector<Tier>    tier_;
    /* Per unit, the tier as a relaxed atomic mirror, for the HTTP thread's reads (tier_published
     * reads, set_tier and init write). Sized with tier_; element type uint8_t to keep the read
     * lock-free and the mirror one byte per unit. */
    TierMirror tier_pub_;
    std::vector<uint8_t> busy_;
    /* Per unit, whether heat may rank it as a demotion victim though the plan called it
     * unmovable: the mover admitted it into the hot VRAM cache. Never a
     * promotion candidate: with no pool slot, only the stager can bring the unit back. */
    std::vector<uint8_t> evictable_;
    std::vector<uint8_t> movable_;
    /* --expert-pin-top: a decode-hot resident the planner pinned is never a demotion candidate,
     * even where a later pass made its unit movable again. */
    std::vector<uint8_t> pinned_;
    std::vector<int32_t> layer_of_;
    /* Bit 0: promoted before. Bit 1: demoted before. Feeds the readmit counters. */
    std::vector<uint8_t> ever_;

    /* Resident units per layer. A layer starved of slots misses on nearly every token however
     * good the ranking inside it is. Most resident units are cold, so exact `heat == 0.0f` ties
     * are common at both ends and the tie-break is effectively the policy; breaking it by index
     * would strip low-numbered layers first. Ties take the victim from the layer holding the most
     * and promote into the layer holding the fewest. */
    std::vector<int32_t> res_count_;
    /* A swap stays inside one slab class: each half uses a slot of its own class, so a pair across
     * classes would drain one class's spare slots into the other's until promotions starve. A rank
     * has one expert class when every expert is the same size, two when it holds an uneven slice
     * of each expert's width (three 128-column blocks of one parity, two of the other). `cls_` is
     * the dense class index, -1 for a unit that cannot move. */
    std::vector<int32_t> cls_;
    int32_t              n_cls_ = 1;
    /* step()'s two ends of the ranking, `moves_per_step` entries a class, kept between calls so a
     * step allocates nothing. */
    std::vector<HeatRanked> rank_hot_, rank_cold_;
    std::vector<int32_t>    n_hot_, n_cold_, m_of_;
    /* Per slot, its lowest movable unit index: the bound step()'s layer screen needs on a tie. */
    std::vector<int32_t> min_unit_;

    /* Each slot's two ends, over movable units not in flight: the hottest off the device and the
     * coldest on it, and how many of each. A dispatch that proposes nothing (nearly all of them
     * once residency settles) is then decided from one entry a layer, not a scan of every unit.
     * Slot n_layer_ is the movable units with no layer.
     *
     * observe() updates a layer's ends on the pass that credits it. A tier or busy change
     * invalidates the unit's slot, which step() rebuilds from its positions on its next read. */
    std::vector<float>   end_hot_, end_cold_;
    std::vector<int32_t> end_n_off_, end_n_on_;
    std::vector<int32_t> end_n_cold_;   /* units on the device at exactly the cold end's heat */
    std::vector<uint8_t> end_ok_;
    void end_rebuild(size_t slot);
    void end_invalidate(int32_t unit) {
        if (end_ok_.empty()) return;
        const int32_t l = (size_t)unit < layer_of_.size() ? layer_of_[(size_t)unit] : -1;
        end_ok_[l >= 0 ? (size_t)l : (size_t)n_layer_] = 0;
    }

    bool warm() const { return (int64_t)stats_.dispatches >= params_.warmup_dispatches; }
};

}  /* namespace rad */
