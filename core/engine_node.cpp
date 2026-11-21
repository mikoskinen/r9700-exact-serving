/* engine_node.cpp -- the two-node pipeline in the engine (tp8 S4; abi/rad_pp.h, core/net/).
 *
 * One engine on two machines, each a --tp group of its own. The leader (--pp-rank 0) runs what a
 * one-machine engine runs -- HTTP, the scheduler, the sampler, the drafter -- with the follower's
 * layers taken out of its program; the follower (--pp-rank 1) runs only the passes of its layers.
 * Per step:
 *
 *   leader: build the step (BatchBuilder) -> ship its staging span to the follower (kStep) ->
 *           release its ranks: embedding -> stage_send X1 ... stage_recv X2 -> its layers -> head,
 *           sampler; every drafter build is shipped too, upload-only, so both nodes' slabs see every
 *           build in order (BatchBuilder::Wire)
 *   follower: replay the span on its own ranks' slabs -> release them: stage_recv X1 -> its layers
 *           -> stage_send X2
 *
 * The plugin places the stage ops (it knows what is live at its cut); the core owns the rings and
 * the link and keeps every host-side decision on the leader. In this build a step is issued after
 * the previous one has committed (no issue-ahead: the next step's rows are built on the host, not
 * advanced on the leader's cards), one step is in flight (one lane), and the features whose state
 * lives on one node's host are off or refused: the prefix cache and its tiers, the KV-to-expert
 * loan, runtime LoRA, encoders, streamed KV (each named at start).
 *
 * Failure: any fault on either node takes the link down (node_link.h), the waiting stage_recv on
 * the other node fails its pass, the leader fails every request (fail_loop) and both processes
 * exit non-zero; a clean stop of the leader stops the follower with status 0. Either node can be
 * started again; they rendezvous afresh.
 */
#include "engine_priv.h"

#include "net/node_link.h"
#include "net/node_rendezvous.h"
#include "runtime/fn4_switch.h"
#include "util/stallwatch.h"
#include "device/device.h"
#include "pp_nodeidle.h"   /* tp8 dev D8 */
#include "pp_pftl_fol.h"  /* tp8 loop P22 */

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <cstring>

/* ================================================================== the plugin's view (rad_pp.h) */
namespace {
struct PpGlobal {
    std::atomic<int>     nodes{ 1 }, rank{ 0 }, cut_h{ 0 }, cut_t{ 0 }, link{ 0 };
    std::atomic<int64_t> row_bytes[RAD_PP_N_RINGS] = {};
    std::atomic<int>     parts[RAD_PP_N_RINGS] = {};
    /* tp8 S5: every lane's rings (ring index lane * RAD_PP_N_RINGS + crossing), the lanes, and the
     * lane and part of the pass the ranks issue now (Engine::pp_set_pass, with every rank parked). */
    std::atomic<RadPpRing*> ring[RAD_PP_MAX_RING_IX] = {};
    std::atomic<int>     lanes{ 1 }, lane{ 0 }, part{ RAD_PP_PART_WHOLE };
    /* tp8 dev M4: RAD_PP_PART_SEG's layers, and the segment boundaries the plugin declared on this node (rad_pp_segments) */
    std::atomic<int>     seg_l0{ 0 }, seg_l1{ 0 };
    std::mutex seg_mu;
    std::vector<int32_t> segs;
    /* tp8 dev D10 round 2: X1 carries the step's token ids and the receiver takes them before any read (rad_pp_x1_ids) */
    std::atomic<int>     x1_ids{ 0 };
    /* tp8 loop P28: the band the plugin declared (rad_pp_band) and the pass being issued's band word (0: not a band pass) */
    std::atomic<int>     band_k{ 0 }, band{ 0 };
    /* tp8 loop D39: the tail band the plugin declared (rad_pp_tail_band) and the pass being issued's tail band word */
    std::atomic<int>     tband_k{ 0 }, tband{ 0 };
    /* tp8 loop D46: the plugin keeps the tail band layers' state on the leader for every sequence (rad_pp_tail_copy) */
    std::atomic<int>     tcopy{ 0 };
    /* tp8 loop P41: the reverse (head) band layer the plugin declared (rad_pp_rband) and the pass being issued's word (0: none) */
    std::atomic<int>     rband_l{ 0 }, rband{ 0 };
    /* tp8 loop P41 step 5: the pass being issued crosses X1 in the keys form (the leader's rule asked for it: a lone prompt chunk
     * of a clean sequence while fn4 pp_rband > 0); 0 every other pass */
    std::atomic<int>     rkeys{ 0 };
    /* tp8 loop D40: the leader head / tail segment being issued's RAD_PP_LSEG_* bits (0 every other pass) */
    std::atomic<int>     lflags{ 0 };
    /* tp8 S6: the KV layers both nodes hold of one source (rad_pp_kv_copy): group, slot here, model layer */
    std::mutex kv_mu;
    std::vector<std::array<int32_t, 3>> kv_copies;
};
PpGlobal& pp_g() { static PpGlobal g; return g; }
}  /* namespace */

extern "C" void rad_pp_kv_copy(int32_t group, int32_t slot, int32_t layer) {
    PpGlobal& g = pp_g();
    std::lock_guard<std::mutex> lk(g.kv_mu);
    for (const auto& c : g.kv_copies)
        if (c[0] == group && c[2] == layer) return;
    g.kv_copies.push_back({ group, slot, layer });
}

extern "C" int rad_pp_info(RadPpInfo* out) {
    if (!out) return RAD_E_INVAL;
    std::memset(out, 0, sizeof *out);
    PpGlobal& g = pp_g();
    out->nodes = g.nodes.load();
    out->rank = g.rank.load();
    out->cut_h = g.cut_h.load();
    out->cut_t = g.cut_t.load();
    out->link = g.link.load();
    out->lanes = g.lanes.load();
    const int lane = std::min(std::max(g.lane.load(), 0), RAD_PP_MAX_LANES - 1);
    out->lane = lane;
    out->part = g.part.load();
    out->seg_l0 = g.seg_l0.load();
    out->seg_l1 = g.seg_l1.load();
    out->reserved[0] = g.band.load();   /* tp8 loop P28: the pass's band k (abi/rad_pp.h rad_pp_band) */
    out->reserved[2] = g.tband.load();  /* tp8 loop D39: the pass's tail band j (abi/rad_pp.h rad_pp_tail_band) */
    out->reserved[1] = g.lflags.load();  /* tp8 loop D40: a leader segment's RAD_PP_LSEG_* bits */
    out->reserved[3] = g.rband.load();  /* tp8 loop P41: the pass's reverse band layer l (abi/rad_pp.h rad_pp_rband) */
    out->reserved[4] = g.rkeys.load();  /* tp8 loop P41 step 5: 1 = X1 in the keys form (layer l's window keys for the follower) */
    for (int i = 0; i < RAD_PP_N_RINGS; ++i) out->ring[i] = g.ring[lane * RAD_PP_N_RINGS + i].load();
    return 0;
}

extern "C" void rad_pp_segments(const int32_t* bounds, int n) {
    PpGlobal& g = pp_g();
    std::lock_guard<std::mutex> lk(g.seg_mu);
    g.segs.clear();
    for (int i = 0; bounds && i < n && i < RAD_PP_MAX_SEGS; ++i) g.segs.push_back(bounds[i]);
}

extern "C" void rad_pp_x1_ids(int on) { pp_g().x1_ids.store(on ? 1 : 0, std::memory_order_relaxed); }
extern "C" void rad_pp_band(int k) { pp_g().band_k.store(k > 0 ? k : 0, std::memory_order_relaxed); }
extern "C" void rad_pp_tail_band(int j) { pp_g().tband_k.store(j > 0 ? j : 0, std::memory_order_relaxed); }
extern "C" void rad_pp_tail_copy(int form) { pp_g().tcopy.store(form > 0 ? form : 0, std::memory_order_relaxed); }
extern "C" void rad_pp_rband(int l) { pp_g().rband_l.store(l > 0 ? l : 0, std::memory_order_relaxed); }

namespace rad {
/* The part of the leader's pass being issued (Ctx::pass_key keys recorded passes by it: a lane step's
 * head and tail share their batch). RAD_PP_PART_WHOLE on one machine and on one lane. */
int pp_pass_part() { return pp_g().part.load(std::memory_order_relaxed); }
/* tp8 loop D46: the pass being issued's band words (P28's k, D39's j x 16; 0 on every other pass), for Ctx::pass_key -- a
 * DECODE pass may now be a tail band pass (fn4 pp_tband 3), and decode passes are recorded: the band form is its own tape */
int pp_pass_bands() {
    const PpGlobal& g = pp_g();
    return g.band.load(std::memory_order_relaxed) + 16 * g.tband.load(std::memory_order_relaxed);
}
/* tp8 loop P10: passes of at least this many rows are issued live, never recorded (RADIANCE_PP_SEND_ASYNC on this node's
 * send ring: a payload of that many rows crosses from a side stream no tape would hold); 0 = no such rule */
std::atomic<int64_t> g_pp_live_rows{ 0 };
int64_t pp_live_rows() { return g_pp_live_rows.load(std::memory_order_relaxed); }
/* tp8 dev M4: a segment part's layers, for its key (each segment is its own recording) */
int pp_pass_seg() {
    const PpGlobal& g = pp_g();
    return g.seg_l0.load(std::memory_order_relaxed) * 256 + g.seg_l1.load(std::memory_order_relaxed);
}
/* ... and the boundaries the plugin declared on this node */
std::vector<int32_t> pp_segments() {
    PpGlobal& g = pp_g();
    std::lock_guard<std::mutex> lk(g.seg_mu);
    return g.segs;
}
/* tp8 dev D10 round 2: the plugin declared X1's token-ids part (rad_pp_x1_ids) */
bool pp_x1_ids() { return pp_g().x1_ids.load(std::memory_order_relaxed) != 0; }
/* tp8 loop P28: the band the plugin declared on this node (rad_pp_band), 0 none */
int pp_band_k() { return pp_g().band_k.load(std::memory_order_relaxed); }
/* tp8 loop D39: the tail band the plugin declared on this node (rad_pp_tail_band), 0 none */
int pp_tail_band_k() { return pp_g().tband_k.load(std::memory_order_relaxed); }
/* tp8 loop D46: the plugin's tail band copy form (rad_pp_tail_copy) on this node, 0 none (core/sched/tcopy.h) */
int pp_tail_copy() { return pp_g().tcopy.load(std::memory_order_relaxed); }
/* tp8 loop P41: the reverse band layer the plugin declared on this node (rad_pp_rband), 0 none */
int pp_rband_l() { return pp_g().rband_l.load(std::memory_order_relaxed); }
}  /* namespace rad */

extern "C" void rad_pp_crossing(int ring, int64_t row_bytes, int parts) {
    if (ring < 0 || ring >= RAD_PP_N_RINGS || row_bytes < 0) return;
    PpGlobal& g = pp_g();
    int64_t cur = g.row_bytes[ring].load();
    while (row_bytes > cur && !g.row_bytes[ring].compare_exchange_weak(cur, row_bytes)) {}
    int pc = g.parts[ring].load();
    while (parts > pc && !g.parts[ring].compare_exchange_weak(pc, parts)) {}
}

namespace rad {

namespace {

/* The step's extras beside its batch: what prepare_release decided on the leader that the
 * follower's plugin reads. */
constexpr uint32_t kStepMagic = 0x34535050u;   /* "PPS4" (review F2 / F4: requests + clears; tp8 S5: lane; tp8 S6: checkpoints) */
constexpr int      kMaxFn4 = 256;
static_assert(fn4::kSwitches <= kMaxFn4, "every fn4 switch fits the step header");
constexpr int      kMaxClears = 1 << 16;
struct StepHdr {
    uint32_t magic = kStepMagic;
    uint32_t run = 0;           /* 1: the follower runs the pass; 0: applies the uploads only */
    int32_t  step = 0;
    int32_t  n_fn4 = 0;
    int32_t  psk_valid = 0;
    int32_t  psk_n = 0;
    int64_t  psk_dec = 0;
    int32_t  n_clears = 0;      /* (group, slot) pairs after the PSK plan: state slots to zero first */
    int32_t  lane = 0;          /* tp8 S5: the lane whose builder and rings the step is (0 on one lane) */
    int32_t  n_ck = 0;          /* tp8 S6: int32 words of checkpoint restores / saves after the clears (pp_ck_list) */
    /* tp8 dev M4: bit 0 -- the step keeps its place on the follower: it has prompt rows (a chunk may read the KV of an
     * earlier chunk of its prompt still in flight in another lane: a lent chunk, or a home chunk planned behind one) or a
     * piece of another lane's sequence (StepEntry::home >= 0); 0 -- a decode step of this lane's own sequences, which reads
     * only committed KV and may run between another step's segments; tp8 dev D10 round 2: bit 1 -- the leader issued the
     * step ahead of its lane's collect (RADIANCE_PP_ISSUE_AHEAD=1): its X1 lands only after the leader's tail of the step
     * before it */
    int32_t  flags = 0;
    /* Review F2 / F3: the leader's switch REQUESTS (fn4::requests), not its applied values: a lever
     * the leader does not declare reads 0 there and must not turn the follower's off. */
    uint32_t fn4_n[kMaxFn4] = {};
    int32_t  fn4_v[kMaxFn4] = {};
};

const char* role(int rank) { return rank == 0 ? "leader" : "follower"; }

/* Options that are each node's own: they may differ between the two. */
const std::vector<std::string>& node_own_options() {
    static const std::vector<std::string> v = {
        "--pp-rank", "--pp-peer", "--host", "--port", "--model", "--radiance-home", "--download-dir",
        "--model-revision", "--gpu-memory-mib", "--gpu-headroom-mib", "--host-pool-mib",
        /* not --tp or --p2p (review F5): a TP2 node behind a TP4 one folds its sums over another rank
         * count, so its answers are neither build's; RADIANCE_PP_ALLOW_DIFF="option --tp" for a test */
        "-v", "-vv", "--live", "--http-threads", "--api-key", "--api-key-env", "--served-model-name",
        "--log-requests", "--kld-record", "--kld-ref", "--kld-out", "--kld-corpus", "--expert-stats",
        "--profile-ops",
        /* tp8 loop D20 (loop FINDINGS F57): the block drafter's short block runs on the leader only (the follower declares
         * no drafter: deepseek41 lr_settle), so its pick is the leader's own */
        "--draft-short-from",
    };
    return v;
}

/* RADIANCE_* settings that are each node's own. */
bool node_own_env(const std::string& k) {
    /* tp8 loop F66: RADIANCE_KV_PLACED(_TEST_MIB) sizes this node's own KV pool; the agreement after it is both nodes' */
    static const char* own[] = { "RADIANCE_PP_", "RADIANCE_HOST_DEVICES", "RADIANCE_HOST_VRAM_MIB",
                                 "RADIANCE_HOME", "RADIANCE_IMAGE_ID", "RADIANCE_DUMP_", "RADIANCE_LOG",
                                 "RADIANCE_PROFILE", "RADIANCE_KV_PLACED" };
    for (const char* o : own)
        if (k.rfind(o, 0) == 0) return true;
    /* tp8 loop D20 (loop FINDINGS F57): the leader's drafter knobs, whole keys (the follower declares no drafter: deepseek41
     * lr_settle; its engine runs no query pass) -- D18's short block, D19's tiers */
    static const char* own_key[] = { "RADIANCE_DS41_DSPARK_SHORT", "RADIANCE_DRAFT_SHORT_TIERS" };
    for (const char* o : own_key)
        if (k == o) return true;
    return false;
}

std::vector<std::string> allowed_differences() {
    std::vector<std::string> out;
    const char* e = std::getenv("RADIANCE_PP_ALLOW_DIFF");
    if (!e) return out;
    std::string s = e;
    size_t at = 0;
    while (at <= s.size()) {
        size_t c = s.find(',', at);
        if (c == std::string::npos) c = s.size();
        if (c > at) out.push_back(s.substr(at, c - at));
        at = c + 1;
    }
    return out;
}

/* An absolute path to a regular file (a setting that names one). */
bool is_regular_file(const std::string& v) {
    if (v.empty() || v[0] != '/') return false;
    struct stat st {};
    return ::stat(v.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

int env_ms(const char* k, int dflt) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    const long n = std::strtol(v, nullptr, 10);
    return n > 0 ? (int)std::min<long>(n, 86400000L) : dflt;
}

}  /* namespace */

/* ================================================================== start-up */
int Engine::pp_check() {
    if (!pp_on()) return RAD_OK;
    if (cfg_.pp_cut_t <= cfg_.pp_cut_h) {
        RAD_ERR("--pp-nodes 2 needs --pp-cut H,T: the follower's layers [H, T)");
        return RAD_E_INVAL;
    }
    if (cfg_.pp_peer.empty() && !cfg_.debug_graph) {
        RAD_ERR("--pp-nodes 2 needs --pp-peer HOST[:PORT]: the other node");
        return RAD_E_INVAL;
    }
    struct Refusal { bool on; const char* what; };
    const Refusal no[] = {
        { cfg_.calib(), "--calib-corpus (the calibration tap runs on one machine)" },
        { cfg_.lora.enabled(), "runtime LoRA (--lora): the adapters' slots live on one node's cards" },
        { cfg_.kv_resident > 0, "--kv-resident (streamed KV): the stage runs are planned from the leader's tables" },
    };
    for (const Refusal& r : no)
        if (r.on) {
            RAD_ERR("two-node pipeline (tp8 S4): %s is not carried to the other node in this build; "
                    "start without it", r.what);
            return RAD_E_UNSUPPORTED;
        }
    /* tp8 S6: the prefix cache adopts blocks and restores snapshots on the leader's say. A hit's
     * blocks are the same ids on both nodes (one block table, the leader's), written there by the
     * steps that wrote them, the copied sources' entries included; each step frame carries the
     * step's checkpoint restores and saves, which the follower runs on its own state groups and
     * window pages (pp_ck_list / pp_ck_apply). The host KV tier's copies are not carried: the tier
     * is off on a pipeline node, so a cached prefix lives in VRAM only. With the cache on, the
     * prefill chunks are the one-machine start's plan (the --prefix-resume exact grid); S4 hw's
     * pp_plan_prefix (defect 3: a cache turned off moved the plan and a near tie's bits) still
     * records the start's value and plans the same. */
    cfg_.pp_plan_prefix = cfg_.prefix_cache;
    /* (review of S6) RADIANCE_PP_PREFIX_CACHE=0: S4's rule for a like-for-like card run -- the cache
     * off on both nodes (no hit, no snapshot, nothing mirrored), the chunk plan still the cache's
     * (pp_plan_prefix, defect 3). The rendezvous compares RADIANCE_* values, so both nodes take it. */
    if (const char* pc = std::getenv("RADIANCE_PP_PREFIX_CACHE"); pc && !std::strcmp(pc, "0") &&
        (cfg_.prefix_cache || cfg_.prefix_cache_mib > 0)) {
        cfg_.prefix_cache = false;
        cfg_.prefix_cache_mib = 0;
        decide("--no-prefix-cache", "prefix caching off",
               "a two-node pipeline node with RADIANCE_PP_PREFIX_CACHE=0 (tp8 S4's rule): no prefix hit "
               "or snapshot on either node; its prefill chunks are planned as the cache's (--prefix-resume "
               "exact grid), so a cold prompt is cut as on one machine");
        RAD_INFO("two-node pipeline: RADIANCE_PP_PREFIX_CACHE=0: prefix caching off on this node (tp8 S4's rule), the "
                 "prefill chunks planned as the cache's");
    }
    if (cfg_.prefix_cache_mib > 0) {
        cfg_.prefix_cache_mib = 0;
        decide("--prefix-cache-mib 0", "the prefix cache's host KV tier off",
               "a two-node pipeline node (tp8 S6): the tier's block copies are not carried to the "
               "follower; prefix hits on VRAM blocks and checkpoint restores are");
    }
    PpGlobal& g = pp_g();
    g.nodes = 2;
    g.rank = cfg_.pp_rank;
    g.cut_h = cfg_.pp_cut_h;
    g.cut_t = cfg_.pp_cut_t;
    g.link = cfg_.node_link;
    /* tp8 S5: lanes (core/sched/lanes.h) */
    n_lanes_ = cfg_.pp_lanes_now();
    if (n_lanes_ > 1) {
        if (cfg_.max_seqs < n_lanes_) {
            RAD_ERR("--pp-lanes %d needs at least as many sequences (--max-num-seqs %lld): every lane holds its own",
                    n_lanes_, (long long)cfg_.max_seqs);
            return RAD_E_INVAL;
        }
        /* tp8 dev F35: RADIANCE_PP_XSTATS=1 takes it anyway, for the per-layer histograms (selections, distinct experts a
         * layer-pass): each pass reports only its own layers, so those are right; the file's issued step / token counts are not
         * (a lane step's head and tail count twice) */
        const char* xv = std::getenv("RADIANCE_PP_XSTATS");
        if (!cfg_.expert_stats.empty() && xv && !std::strcmp(xv, "1")) {
            RAD_WARN("two-node pipeline: --expert-stats with --pp-lanes %d (RADIANCE_PP_XSTATS=1, tp8 dev F35): the per-layer histograms, "
                     "selections and distinct experts a layer-pass are each pass's; the issued steps / tokens count a lane step's "
                     "two leader passes twice", n_lanes_);
        } else if (!cfg_.expert_stats.empty()) {
            RAD_ERR("two-node pipeline (tp8 S5): --expert-stats counts a lane step's two leader passes as two steps; "
                    "start it with --pp-lanes 1 (or RADIANCE_PP_XSTATS=1 for the per-layer histograms alone)");
            return RAD_E_UNSUPPORTED;
        }
    }
    g.lanes = n_lanes_;
    decide("two-node pipeline", fmt("%s of W(%d,%d), link %s, %d lane%s", role(cfg_.pp_rank), cfg_.pp_cut_h,
                                    cfg_.pp_cut_t, cfg_.node_link ? "verbs" : "tcp", n_lanes_, n_lanes_ == 1 ? "" : "s"),
           fmt("the follower holds layers [%d, %d), the leader every other layer, the embedding and "
               "the head; a step crosses twice%s", cfg_.pp_cut_h, cfg_.pp_cut_t,
               n_lanes_ > 1 ? fmt("; %d lanes, each a scheduler with one step in flight, so both nodes work at "
                                  "once on different lanes (tp8 S5)", n_lanes_).c_str()
                            : "; one step in flight (--pp-lanes 1)"));
    return RAD_OK;
}

int Engine::pp_connect() {
    if (!pp_on() || cfg_.debug_graph) return RAD_OK;
    net::LinkOpts o = net::LinkOpts::from_env();
    /* tp8 dev 1a: RADIANCE_PP_FOLLOWER_EMBED=1 at W(0, t) -- the follower looks the embedding up itself, the leader sends
     * no X1 (the plugins; at h > 0 X1 is a layer's residual and stays) */
    const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
    const bool fe_on = fe && !std::strcmp(fe, "1") && cfg_.pp_cut_h == 0;
    /* tp8 dev D10 round 2: with RADIANCE_PP_ISSUE_AHEAD=1 at W(0, t) the plugins' X1 carries the step's token ids, so the
     * leader sends X1 beside the follower's own lookup (read before declare: the plugins decide it from the same switch) */
    const char* iav = std::getenv("RADIANCE_PP_ISSUE_AHEAD");
    o.no_x1 = fe_on && !(iav && !std::strcmp(iav, "1"));
    if (fe && !std::strcmp(fe, "1"))
        RAD_INFO("two-node pipeline: RADIANCE_PP_FOLLOWER_EMBED=1 %s", fe_on
                     ? "-- the follower looks the step's embedding up itself; the leader sends no X1 (tp8 dev 1a)"
                     : fmt("is not taken at W(%d,%d): X1 is the residual after layer %d (it applies at W(0, t) only)",
                           cfg_.pp_cut_h, cfg_.pp_cut_t, cfg_.pp_cut_h - 1).c_str());
    std::string err;
    if (pp_follower()) {
        /* Robustness 4: listen where the leader reaches this node (the address the route to
         * --pp-peer leaves from) and take a connection only from --pp-peer's address.
         * RADIANCE_PP_BIND names the address instead; RADIANCE_PP_ANY_PEER=1 takes any peer. */
        const char* bind = std::getenv("RADIANCE_PP_BIND");
        std::string at = bind && *bind ? bind : net::local_address_for(cfg_.pp_peer, cfg_.pp_port);
        if (at.empty()) at = "0.0.0.0";
        const char* any = std::getenv("RADIANCE_PP_ANY_PEER");
        const bool any_peer = any && !std::strcmp(any, "1");
        pp_link_ = net::NodeLink::listen(at, cfg_.pp_port, o, &err, any_peer ? std::string() : cfg_.pp_peer);
    } else {
        pp_link_ = net::NodeLink::connect(cfg_.pp_peer, cfg_.pp_port, o, &err);
    }
    if (!pp_link_) {
        RAD_ERR("two-node pipeline: %s", err.c_str());
        return RAD_E_IO;
    }
    /* Phase 1: everything both nodes must hold the same before either declares a weight. */
    net::Facts f;
    f.add("protocol", "tp8-s4 wire 1");
    f.add("role", role(cfg_.pp_rank));
    f.add("cut", fmt("%d,%d", cfg_.pp_cut_h, cfg_.pp_cut_t));
    f.add("link", cfg_.node_link ? "verbs" : "tcp");
    if (fe_on) f.add("follower embed", "on");   /* tp8 dev 1a: only when on, so the flags-off facts are as before */
    f.add("binary", net::file_identity("/proc/self/exe"));
    for (const std::string& p : net::loaded_plugin_identities()) {
        const size_t eq = p.find('=');
        f.add("plugin " + p.substr(0, eq), p.substr(eq + 1));
    }
    const char* img = std::getenv("RADIANCE_IMAGE_ID");
    f.add("image", img && *img ? img : "unset");
    /* The model by its bytes, not its path: the two machines hold their own copies (review F5: 17
     * pieces over the whole file, not its first 64 MiB). */
    f.add("model", net::sampled_identity(cfg_.model));
    f.add("model arch", plugins_.arch_id + " " + plugins_.arch_quant);
    {
        std::string sw;
        for (int i = 0; i < fn4::kSwitches; ++i)
            sw += fmt("%s%s=%d", i ? "," : "", fn4::switch_name(i), fn4::value(i));
        f.add("switches", sw);
    }
    {
        std::vector<std::string> env;
        for (char** e = environ; *e; ++e) {
            const std::string kv = *e;
            /* RADIANCE_* and the kernel library's R4D_* (review F5: the published env sets both) */
            if (kv.rfind("RADIANCE_", 0) != 0 && kv.rfind("R4D_", 0) != 0) continue;
            const std::string k = kv.substr(0, kv.find('='));
            if (node_own_env(k)) continue;
            env.push_back(kv);
        }
        std::sort(env.begin(), env.end());
        for (const std::string& kv : env) {
            const size_t eq = kv.find('=');
            const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
            f.add("env " + k, v);
            /* A setting that names a file: the file's bytes too, not only its path (review F5). */
            if (is_regular_file(v)) f.add("env file " + k, net::sampled_identity(v));
        }
    }
    {
        const std::vector<std::string>& own = node_own_options();
        for (const diag::ConfigRow& r : diag::config_rows(cfg_)) {
            f.add("option " + r.flag, r.value);
            if (std::find(own.begin(), own.end(), r.flag) == own.end() && is_regular_file(r.value))
                f.add("option file " + r.flag, net::sampled_identity(r.value));
        }
    }
    net::Facts theirs;
    if (net::exchange(*pp_link_, net::kHello, f, &theirs, o.connect_ms, &err) < 0) {
        RAD_ERR("two-node pipeline: the rendezvous failed: %s", err.c_str());
        return RAD_E_IO;
    }
    std::vector<std::string> own = { "role" };
    for (const std::string& k : node_own_options()) own.push_back("option " + k);
    for (const std::string& k : allowed_differences()) own.push_back(k);
    const std::vector<std::string> diff = net::differences(f, theirs, own);
    const std::string* peer_role = theirs.find("role");
    std::vector<std::string> bad = diff;
    if (!peer_role || *peer_role == role(cfg_.pp_rank))
        bad.push_back(fmt("role: both nodes are the %s (--pp-rank)", role(cfg_.pp_rank)));
    if (!bad.empty()) {
        for (const std::string& d : bad) RAD_ERR("two-node pipeline: the nodes differ: %s", d.c_str());
        RAD_ERR("two-node pipeline: %zu difference(s) between the nodes before declare: both refuse to "
                "start (RADIANCE_PP_ALLOW_DIFF=key,... accepts named ones)", bad.size());
        pp_link_->fail("the rendezvous found differences between the nodes");
        return RAD_E_STATE;
    }
    RAD_INFO("two-node pipeline: rendezvous 1 passed: %zu facts equal on both nodes (binary %s, model %s)",
             f.kv.size(), f.find("binary")->c_str(), f.find("model")->c_str());
    return RAD_OK;
}

/* KV capacity (review F1). The leader's scheduler hands out every block id both nodes use, and
 * each node carves its own groups -- the ones it holds for real and, zeroed, the other node's. A
 * cap set after the carve (a loan) is not a cap: the first allocation past it recalls the loan,
 * and a zeroed group is never lent, so it keeps the leader's own count. Capping before the carve
 * is the one-machine engine's rule for unequal ranks: each node pages at most the least of the two
 * counts, so every group -- zeroed or not -- holds the same blocks on both (rendezvous 2 checks). */
int Engine::pp_kv_agree(int64_t* least) {
    if (!pp_link_) return RAD_OK;
    const int64_t mine = *least;   /* -1: every rank of this node holds no paged cache */
    net::Facts f, theirs;
    f.add("kv tokens", std::to_string(mine < 0 ? -1 : mine));
    std::string err;
    if (net::exchange(*pp_link_, net::kKvCap, f, &theirs, pp_link_->opts().connect_ms, &err) < 0) {
        RAD_ERR("two-node pipeline: the KV capacities were not exchanged: %s", err.c_str());
        return RAD_E_IO;
    }
    const std::string* t = theirs.find("kv tokens");
    char* end = nullptr;
    const long long peer = t ? std::strtoll(t->c_str(), &end, 10) : -2;
    if (!t || !end || *end || peer < -1) {
        pp_link_->fail("the peer's KV capacity is not a token count");
        RAD_ERR("two-node pipeline: the peer's KV capacity is not a token count");
        return RAD_E_STATE;
    }
    int64_t agreed = mine;
    if (peer >= 0 && (agreed < 0 || peer < agreed)) agreed = peer;
    *least = agreed;
    RAD_INFO("two-node pipeline: KV capped on both nodes before the carve at %lld tokens (this node "
             "pages %lld, the %s %lld): every group, a zeroed one too, holds the same blocks on both",
             (long long)agreed, (long long)mine, pp_leader() ? "follower" : "leader", peer);
    return RAD_OK;
}

int Engine::pp_serve_setup() {
    if (!pp_on()) return RAD_OK;
    const Program& p0 = ranks_[0]->program;
    if (p0.encoder.modalities != 0 || p0.audio_encoder.modalities != 0) {
        RAD_ERR("two-node pipeline (tp8 S4): the program declares an encoder, whose rows are not "
                "shipped to the follower; serve text only (--mm-max-patches 0 or a text container)");
        return RAD_E_UNSUPPORTED;
    }
    PpGlobal& g = pp_g();
    /* ---- the rings: this node's ring of each crossing, sized for the plugin's rows; one pair a lane
     * (tp8 S5), so each ring has at most one payload in flight as with one lane */
    for (int id = 0; id < RAD_PP_N_RINGS; ++id) {
        const int64_t rb = g.row_bytes[id].load();
        const int parts = std::max(1, g.parts[id].load());
        if (rb <= 0) {
            RAD_ERR("two-node pipeline: the '%s' plugin names no stage crossing X%d (rad_pp_crossing): it "
                    "cannot run as a node of a two-node pipeline", plugins_.arch_plugin.c_str(), id + 1);
            return RAD_E_UNSUPPORTED;
        }
        /* + each part's alignment + the payload's trailer (review F6) */
        const int64_t cap = rb * cfg_.max_tok + (int64_t)parts * RAD_PP_PART_ALIGN + RAD_PP_TRAILER;
        for (int lane = 0; lane < n_lanes_; ++lane) {
            const int ix = lane * RAD_PP_N_RINGS + id;
            pp_ring_cap_[ix] = cap;
            void* mem = rad_dev_alloc(RAD_PP_RING_HDR + cap, RAD_MEM_HOST_PINNED);
            if (!mem) {
                RAD_ERR("two-node pipeline: no pinned host memory for crossing X%d's ring of lane %d (%s)", id + 1,
                        lane, humanb(RAD_PP_RING_HDR + cap).c_str());
                return RAD_E_NOMEM;
            }
            RadPpRing* r = static_cast<RadPpRing*>(mem);
            /* X1 goes leader -> follower, X2 follower -> leader. */
            const bool sends = (id == RAD_PP_X1) == pp_leader();
            net::ring_init(r, id, (uint64_t)cap, sends ? 0 : cfg_.tp,
                           (uint64_t)env_ms("RADIANCE_PP_RECV_DEADLINE_MS", 600000));
            /* tp8 loop P8: RADIANCE_PP_SEND_SPLIT -- who copies this node's payloads into its send ring */
            if (sends && cfg_.tp >= 2) {
                r->send_split = (uint64_t)pp_send_split();
                r->send_ranks = (uint64_t)cfg_.tp;
            }
            (sends ? pp_send_ : pp_recv_)[ix] = r;
        }
    }
    /* ---- tp8 loop P10: RADIANCE_PP_SEND_ASYNC -- one card staging buffer on the sending rank (rank 0, `sender` of the
     * plugin's stage_send), kept by the budget beside the step staging; every send ring names it */
    if (const int64_t arows = pp_send_async_rows(); arows > 0) {
        const int64_t sb = pp_send_async_bytes();
        if (sb <= 0) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_SEND_ASYNC=%lld but this node's crossing has no row bytes",
                    (long long)arows);
            return RAD_E_STATE;
        }
        RAD_TRY(rad_dev_set(ranks_[0]->device));
        pp_stage_dev_ = rad_dev_alloc(sb, RAD_MEM_DEVICE);
        if (!pp_stage_dev_) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_SEND_ASYNC: no device memory for the %s staging buffer on card %d "
                    "(raise --gpu-headroom-mib by %lld, or unset the switch)", humanb(sb).c_str(), ranks_[0]->device,
                    (long long)((sb + (1 << 20) - 1) >> 20));
            return RAD_E_NOMEM;
        }
        int rings = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            RadPpRing* r = pp_send_[ix];
            if (!r) continue;
            r->stage_dev = (uint64_t)(uintptr_t)pp_stage_dev_;
            r->stage_cap = (uint64_t)sb;
            r->async_rows = (uint64_t)arows;
            ++rings;
        }
        g_pp_live_rows.store(arows, std::memory_order_relaxed);
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_ASYNC=%lld -- a %s payload of >= %lld rows is packed into a %s "
                 "staging buffer on card %d on the pass's stream and crosses into the send ring from a side stream of that "
                 "card, so the next pass starts at once (%d send ring(s)); passes of >= %lld rows are issued live, never "
                 "recorded (tp8 loop P10)", (long long)arows, pp_leader() ? "X1" : "X2", (long long)arows,
                 humanb(sb).c_str(), ranks_[0]->device, rings, (long long)arows);
    }
    /* ---- tp8 loop P10b: RADIANCE_PP_RECV_ASYNC -- a card staging buffer on every rank of this node, kept by the budget
     * beside the step staging; every receive ring names them */
    if (const int64_t rrows = pp_recv_async_rows(); rrows > 0) {
        const int64_t rb = pp_recv_async_bytes();
        if (rb <= 0) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_RECV_ASYNC=%lld but this node's crossing has no row bytes", (long long)rrows);
            return RAD_E_STATE;
        }
        if ((int)ranks_.size() > RAD_PP_MAX_RANKS) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_RECV_ASYNC: more ranks than a ring names (%d)", RAD_PP_MAX_RANKS);
            return RAD_E_UNSUPPORTED;
        }
        std::vector<void*> st(ranks_.size(), nullptr);
        for (size_t q = 0; q < ranks_.size(); ++q) {
            RAD_TRY(rad_dev_set(ranks_[q]->device));
            st[q] = rad_dev_alloc(rb, RAD_MEM_DEVICE);
            if (!st[q]) {
                RAD_ERR("two-node pipeline: RADIANCE_PP_RECV_ASYNC: no device memory for the %s staging buffer on card %d "
                        "(raise --gpu-headroom-mib by %lld, or unset the switch)", humanb(rb).c_str(), ranks_[q]->device,
                        (long long)((rb + (1 << 20) - 1) >> 20));
                (void)rad_dev_set(ranks_[0]->device);
                return RAD_E_NOMEM;
            }
        }
        (void)rad_dev_set(ranks_[0]->device);
        int rings = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            RadPpRing* r = pp_recv_[ix];
            if (!r) continue;
            for (size_t q = 0; q < st.size(); ++q) r->recv_stage[q] = (uint64_t)(uintptr_t)st[q];
            r->recv_cap = (uint64_t)rb;
            r->recv_rows = (uint64_t)rrows;
            ++rings;
        }
        const int64_t prev = g_pp_live_rows.load(std::memory_order_relaxed);
        g_pp_live_rows.store(prev > 0 ? std::min(prev, rrows) : rrows, std::memory_order_relaxed);
        RAD_INFO("two-node pipeline: RADIANCE_PP_RECV_ASYNC=%lld -- a %s payload of >= %lld rows is copied from the ring into "
                 "a %s staging buffer on each of the %zu cards by a side stream issued with the pass (it waits on the card "
                 "for the payload to land, beside the previous pass) and unpacked on the pass's stream (%d receive ring(s)); "
                 "passes of >= %lld rows are issued live, never recorded (tp8 loop P10b)", (long long)rrows,
                 pp_leader() ? "X2" : "X1", (long long)rrows, humanb(rb).c_str(), st.size(), rings,
                 (long long)g_pp_live_rows.load(std::memory_order_relaxed));
    }
    /* ---- tp8 loop P46: RADIANCE_PP_SEND_ASH -- a card staging buffer on every rank of this node for its share of a row-sharded
     * payload, kept by the budget beside the step staging; every send ring names them, fn4 pp_send_ash turns the path on / off */
    if (const int64_t hrows = pp_send_ash_rows(); hrows > 0) {
        const int64_t hb = pp_send_ash_bytes();
        if (hb <= 0) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_SEND_ASH=%lld but this node's crossing has no row bytes", (long long)hrows);
            return RAD_E_STATE;
        }
        if ((int)ranks_.size() > RAD_PP_MAX_RANKS) {
            RAD_ERR("two-node pipeline: RADIANCE_PP_SEND_ASH: more ranks than a ring names (%d)", RAD_PP_MAX_RANKS);
            return RAD_E_UNSUPPORTED;
        }
        std::vector<void*> st(ranks_.size(), nullptr);
        for (size_t q = 0; q < ranks_.size(); ++q) {
            RAD_TRY(rad_dev_set(ranks_[q]->device));
            st[q] = rad_dev_alloc(hb, RAD_MEM_DEVICE);
            if (!st[q]) {
                RAD_ERR("two-node pipeline: RADIANCE_PP_SEND_ASH: no device memory for the %s staging buffer on card %d "
                        "(raise --gpu-headroom-mib by %lld, or unset the switch)", humanb(hb).c_str(), ranks_[q]->device,
                        (long long)((hb + (1 << 20) - 1) >> 20));
                (void)rad_dev_set(ranks_[0]->device);
                return RAD_E_NOMEM;
            }
        }
        (void)rad_dev_set(ranks_[0]->device);
        int rings = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            RadPpRing* r = pp_send_[ix];
            if (!r) continue;
            for (size_t q = 0; q < st.size(); ++q) r->send_stage[q] = (uint64_t)(uintptr_t)st[q];
            r->send_stage_cap = (uint64_t)hb;
            r->ash_rows = (uint64_t)hrows;
            ++rings;
        }
        pp_ash_apply();
        const int64_t prev = g_pp_live_rows.load(std::memory_order_relaxed);
        g_pp_live_rows.store(prev > 0 ? std::min(prev, hrows) : hrows, std::memory_order_relaxed);
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_ASH=%lld -- a row-sharded %s payload of >= %lld rows: every one of the %zu ranks "
                 "packs its share into its own %s staging buffer on the pass's stream and a side stream of its card copies it into "
                 "the send ring (the split send's handshake unchanged), so the pass's stream goes on with the next pass (%d send "
                 "ring(s); fn4 pp_send_ash %d); passes of >= %lld rows are issued live, never recorded (tp8 loop P46)",
                 (long long)hrows, pp_leader() ? "X1" : "X2", (long long)hrows, st.size(), humanb(hb).c_str(), rings,
                 fn4::pp_send_ash(), (long long)g_pp_live_rows.load(std::memory_order_relaxed));
    }
    /* ---- rendezvous 2: what both carved and laid out */
    net::Facts f;
    {
        const KVManager& kv = ranks_[0]->kv;
        int64_t tokens = 0;
        for (const KVGroupPlan& p : kv.plans()) {
            f.add(fmt("kv group %d", p.index),
                  fmt("%s kind %d block %lld window_blocks %lld states %lld", p.name.c_str(), p.kind,
                      (long long)p.block_size, (long long)p.window_blocks, (long long)p.n_states));
            /* Review F1: each group's carve -- block (or state) count and whether this node holds it
             * -- checked below: an id the leader hands out must exist wherever the follower holds the
             * group (pp_kv_agree capped both before the carve, so the counts come out equal). */
            f.add(fmt("kv blocks %d", p.index),
                  fmt("%lld %d", (long long)(p.paged() ? p.n_blocks : p.n_states), p.zero ? 1 : 0));
            if (p.kind == RAD_KV_FULL && p.block_size > 0 && !p.zero) {
                const int64_t t = p.n_blocks * p.block_size;
                if (tokens == 0 || t < tokens) tokens = t;
            }
        }
        f.add("kv tokens", std::to_string(tokens));
        /* tp8 S6: the leader's checkpoint slot ids must exist on the follower (its snapshots of its own
         * state groups land in the same slots) */
        f.add("checkpoint slots", std::to_string((long long)kv.checkpoint_slots()));
    }
    f.add("step batch layout", fmt("%016llx", (unsigned long long)sched_.wire_layout()));
    f.add("lanes", std::to_string(n_lanes_));
    for (int id = 0; id < RAD_PP_N_RINGS; ++id)
        f.add(fmt("crossing X%d", id + 1), fmt("%lld bytes a row in %d part(s)", (long long)g.row_bytes[id].load(),
                                               g.parts[id].load()));
    net::Facts theirs;
    std::string err;
    if (net::exchange(*pp_link_, net::kHello2, f, &theirs, pp_link_->opts().connect_ms, &err) < 0) {
        RAD_ERR("two-node pipeline: the second rendezvous failed: %s", err.c_str());
        return RAD_E_IO;
    }
    std::vector<std::string> own2 = { "kv tokens", "checkpoint slots" };
    for (const KVGroupPlan& p : ranks_[0]->kv.plans()) own2.push_back(fmt("kv blocks %d", p.index));
    std::vector<std::string> diff = net::differences(f, theirs, own2);
    /* The carve rule (review F1), the same verdict on both nodes: wherever the follower holds a
     * group, the leader's count of it may not exceed the follower's. A group the follower zeroes
     * (the leader's own sources, the drafter's) it never reads or writes, so its count there is free. */
    for (const KVGroupPlan& p : ranks_[0]->kv.plans()) {
        const std::string k = fmt("kv blocks %d", p.index);
        const std::string* a = f.find(k);
        const std::string* t = theirs.find(k);
        long long n_me = -1, n_peer = -1;
        int z_me = 0, z_peer = 0;
        if (!a || !t || std::sscanf(a->c_str(), "%lld %d", &n_me, &z_me) != 2 ||
            std::sscanf(t->c_str(), "%lld %d", &n_peer, &z_peer) != 2) {
            diff.push_back(fmt("%s: this node '%s', the peer '%s'", k.c_str(), a ? a->c_str() : "none",
                               t ? t->c_str() : "none"));
            continue;
        }
        const long long lead_n = pp_leader() ? n_me : n_peer, fol_n = pp_leader() ? n_peer : n_me;
        const bool fol_holds = !(pp_leader() ? z_peer : z_me);
        if (fol_holds && lead_n > fol_n)
            diff.push_back(fmt("KV group '%s': the leader can hand out %lld blocks, the follower holds %lld: ids past "
                               "the follower's carve", p.name.c_str(), lead_n, fol_n));
        else if (lead_n != fol_n)
            RAD_INFO("two-node pipeline: KV group '%s' holds %lld blocks on the leader, %lld on the follower, which "
                     "%s", p.name.c_str(), lead_n, fol_n, fol_holds ? "holds it (more than enough)" : "zeroes it");
    }
    {
        const std::string* a = f.find("checkpoint slots");
        const std::string* t = theirs.find("checkpoint slots");
        const long long me = a ? std::atoll(a->c_str()) : 0, them = t ? std::atoll(t->c_str()) : 0;
        const long long lead_ck = pp_leader() ? me : them, fol_ck = pp_leader() ? them : me;
        if (cfg_.prefix_cache && fol_ck < lead_ck)
            diff.push_back(fmt("checkpoint slots: the leader hands out %lld, the follower holds %lld (give both nodes "
                               "the same --checkpoint-slots)", lead_ck, fol_ck));
    }
    if (!diff.empty()) {
        for (const std::string& d : diff) RAD_ERR("two-node pipeline: the nodes differ: %s", d.c_str());
        RAD_ERR("two-node pipeline: %zu difference(s) after the KV carve: both refuse to serve", diff.size());
        pp_link_->fail("the second rendezvous found differences between the nodes");
        return RAD_E_STATE;
    }
    /* KV: block ids are the leader's; neither node may hand out one the other has not carved. Both
     * capped before the carve (pp_kv_agree) and every group's block count is equal (above). "kv
     * tokens" is the least over this node's own groups, for the log. */
    const int64_t mine = std::atoll(f.find("kv tokens")->c_str());
    const int64_t peer = std::atoll(theirs.find("kv tokens") ? theirs.find("kv tokens")->c_str() : "0");
    /* The follower refuses a replayed block id or slot past its own carve instead of writing it
     * (BatchBuilder::set_wire_limits): loud where a cap broke, never a write outside the pool. */
    if (pp_follower()) {
        std::vector<int64_t> ids, slots;
        for (const KVGroupPlan& p : ranks_[0]->kv.plans()) {
            if (p.index < 0) continue;
            if ((size_t)p.index >= ids.size()) { ids.resize((size_t)p.index + 1, -1); slots.resize((size_t)p.index + 1, -1); }
            if (p.zero) continue;   /* nothing carved here, nothing written: no limit */
            if (p.paged() && p.n_blocks > 0 && p.block_size > 0) {
                ids[(size_t)p.index] = p.n_blocks;
                slots[(size_t)p.index] = p.n_blocks * p.block_size;
            } else if (p.stateful() && p.n_states > 0) {
                ids[(size_t)p.index] = slots[(size_t)p.index] = p.n_states;
            }
        }
        for (int lane = 1; lane < n_lanes_; ++lane) sched_of(lane).set_wire_limits(ids, slots);
        sched_.set_wire_limits(std::move(ids), std::move(slots));
    }
    /* ---- the verbs data path, then the link's threads */
    if (cfg_.node_link == 1) {
        sockaddr_storage a{};
        socklen_t al = sizeof a;
        char ip[INET6_ADDRSTRLEN] = "";
        if (getsockname(pp_link_->fd(), (sockaddr*)&a, &al) == 0 && a.ss_family == AF_INET)
            inet_ntop(AF_INET, &((sockaddr_in*)&a)->sin_addr, ip, sizeof ip);
        const char* dev = std::getenv("RADIANCE_PP_DEV");
        const char* gid = std::getenv("RADIANCE_PP_GID");
        std::unique_ptr<net::DataPath> dp =
            net::verbs_open_n(*pp_link_, dev ? dev : "auto", ip, gid ? std::atoi(gid) : -1, pp_send_,
                              pp_recv_, n_lanes_ * RAD_PP_N_RINGS, &err);
        if (!dp) {
            RAD_ERR("two-node pipeline: --node-link verbs: %s", err.c_str());
            return RAD_E_UNSUPPORTED;
        }
        RAD_INFO("two-node pipeline: %s", dp->describe().c_str());
        pp_link_->set_data_path(std::move(dp));
    }
    pp_link_->attach_n(pp_send_, pp_recv_, n_lanes_ * RAD_PP_N_RINGS);
    /* Review F2 / F3: the follower's answers to the leader's switch asks and its applied values are
     * taken off the leader's receive thread (nothing else reads the leader's control frames while it
     * serves). */
    if (pp_leader())
        pp_link_->on_control = [this](net::Msg& m) -> bool {
            if (m.h.kind == net::kFn4Ans) {
                std::lock_guard<std::mutex> lk(pp_ans_mu_);
                pp_ans_seq_ = m.h.seq;
                pp_ans_verdict_ = (int)(int64_t)m.h.aux;
                pp_ans_why_.assign(m.body.begin(), m.body.end());
                pp_ans_cv_.notify_all();
                return true;
            }
            if (m.h.kind == net::kFn4State) {
                std::vector<int32_t> v(m.body.size() / sizeof(int32_t));
                if (!v.empty()) std::memcpy(v.data(), m.body.data(), v.size() * sizeof(int32_t));
                fn4::note_peer(v.data(), (int)v.size());
                return true;
            }
            return false;
        };
    pp_link_->start();
    /* tp8 S5: RADIANCE_PP_STAMPS=PATH -- every stage pass's busy interval on this node into
     * PATH.leader / PATH.follower (Engine::pp_stamp), for the host backend's throughput check */
    if (const char* sp = std::getenv("RADIANCE_PP_STAMPS"); sp && *sp) {
        const std::string path = fmt("%s.%s", sp, pp_leader() ? "leader" : "follower");
        pp_stamps_ = std::fopen(path.c_str(), "w");
        if (pp_stamps_) RAD_INFO("two-node pipeline: stage busy stamps into %s", path.c_str());
    }
    for (int ix = 0; ix < n_lanes_ * RAD_PP_N_RINGS; ++ix)
        g.ring[ix] = pp_send_[ix] ? pp_send_[ix] : pp_recv_[ix];
    /* Ready on both sides before the leader serves. */
    if (pp_link_->send(net::kReady, 0, 0, role(cfg_.pp_rank), (int64_t)std::strlen(role(cfg_.pp_rank))) < 0) {
        RAD_ERR("two-node pipeline: %s", pp_link_->why().c_str());
        return RAD_E_IO;
    }
    net::Msg m;
    if (pp_link_->recv(&m, pp_link_->opts().connect_ms) < 0 || m.h.kind != net::kReady) {
        RAD_ERR("two-node pipeline: the peer did not report ready: %s", pp_link_->why().c_str());
        return RAD_E_IO;
    }
    if (pp_leader()) {
        for (int lane = 0; lane < n_lanes_; ++lane)
            sched_of(lane).set_build_hook([this, lane](const RadBatch* b) { pp_on_build(b, lane); });
        /* A switch request names a lever of the whole pipeline: one the leader does not declare (its
         * layers are the follower's) is taken when the follower takes it (fn4::check). */
        fn4::set_peer_check([this](const std::string& n, int v, std::string* why) { return pp_peer_check(n, v, why); });
    }
    /* tp8 S6: the KV-copy check (debug). Each copied layer's pool starts zeroed on both nodes, so a
     * block neither node has written compares equal and the CRC is over the blocks the leader has
     * handed out. */
    pp_kv_check_every_ = env_ms("RADIANCE_PP_KV_CHECK", 0);
    if (pp_kv_check_every_ > 0) {
        std::vector<std::array<int32_t, 3>> regs;
        {
            std::lock_guard<std::mutex> lk(g.kv_mu);
            regs = g.kv_copies;
        }
        std::string names;
        for (const auto& c : regs)
            for (auto& rk : ranks_) {
                const KVGroupPlan* p = rk->kv.plan(c[0]);
                if (!p || !p->base || p->zero || !p->paged() || c[1] < 0 || c[1] >= p->n_layers) continue;
                if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                (void)rad_memset_async((char*)p->base + (int64_t)c[1] * p->layer_stride(), 0, p->layer_stride(),
                                       rk->ctx.stream());
                (void)rad_stream_sync(rk->ctx.stream());
                if (rk.get() == ranks_[0].get())
                    names += fmt("%s%s layer %d", names.empty() ? "" : ", ", p->name.c_str(), (int)c[2]);
            }
        if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
        RAD_INFO("two-node pipeline: RADIANCE_PP_KV_CHECK=%d: the copied KV layers (%s) zeroed on every rank; the "
                 "leader sends their CRC-32C every %d trunk steps and the follower compares its own",
                 pp_kv_check_every_, names.empty() ? "none on this cut" : names.c_str(), pp_kv_check_every_);
        /* tp8 loop P37: the check's timing under lanes (the leader plans; the follower compares whenever it arrives) */
        if (const char* v = std::getenv("RADIANCE_PP_KV_CHECK_DEFER"); v && std::atoi(v) > 0 && cfg_.pp_rank == 0) {
            pp_kv_check_defer_ = true;
            RAD_INFO("two-node pipeline: RADIANCE_PP_KV_CHECK_DEFER=1 (tp8 loop P37): under lanes a due KV-copy check waits for a "
                     "round with no prompt rows in flight or waiting, at most %d more trunk steps, instead of draining the "
                     "pipeline inside a prompt (the copies compared and the tokens are unchanged)", pp_kv_check_every_);
        }
    }
    RAD_INFO("two-node pipeline: the %s of W(%d,%d) is ready; link %s:%d (%s), crossings X1 %lld / X2 %lld "
             "bytes a row (rings %s / %s, %d lane%s), KV %lld tokens here, %lld on the peer",
             role(cfg_.pp_rank), cfg_.pp_cut_h, cfg_.pp_cut_t, cfg_.pp_peer.c_str(), cfg_.pp_port,
             cfg_.node_link ? "verbs" : "tcp", (long long)g.row_bytes[0].load(), (long long)g.row_bytes[1].load(),
             humanb(pp_ring_cap_[0]).c_str(), humanb(pp_ring_cap_[1]).c_str(), n_lanes_, n_lanes_ == 1 ? "" : "s",
             (long long)mine, (long long)peer);
    return RAD_OK;
}

/* ================================================================== the leader */
void Engine::pp_on_build(const RadBatch* b, int lane) {
    if (!pp_link_ || !b) return;
    if (pp_trunk_pending_[lane]) {
        pp_link_->fail(fmt("lane %d: the leader built a pass before its trunk step was released (a schedule this "
                           "build does not ship)", lane));
        return;
    }
    if (b->draft_pass == 0 && b->enc == 0) { pp_trunk_pending_[lane] = true; return; }
    /* A drafter's pass: its uploads, not its pass, on the follower. Its frame is made here -- the
     * span is the builder's until the next build -- and sent by the step loop before the next trunk
     * step (pp_ship), so no socket write runs inside the scheduler's lock, which every drafter build
     * holds (review robustness 1(e)). The follower needs it only before that trunk step. */
    std::vector<uint8_t> fr;
    const char* span = nullptr;
    int64_t n = 0;
    if (pp_frame(b, false, &fr, &span, &n, lane) < 0) return;
    const size_t at = fr.size();
    fr.resize(at + (size_t)n);
    if (n > 0) std::memcpy(fr.data() + at, span, (size_t)n);
    pp_draft_q_.emplace_back(b->step, lane, std::move(fr));
}

void Engine::pp_set_pass(int lane, int part, int seg_l0, int seg_l1, int band, int lflags, int tband, int rband, int rkeys) {
    PpGlobal& g = pp_g();
    g.lane.store(lane, std::memory_order_relaxed);
    g.part.store(part, std::memory_order_relaxed);
    g.band.store(band, std::memory_order_relaxed);   /* tp8 loop P28: every other pass resets it */
    g.tband.store(tband, std::memory_order_relaxed); /* tp8 loop D39: the same */
    g.rband.store(rband, std::memory_order_relaxed); /* tp8 loop P41: the same */
    g.rkeys.store(rkeys, std::memory_order_relaxed); /* tp8 loop P41 step 5: the same */
    /* tp8 dev M9: a leader's head / tail part names its segment too (seg_l1 > 0); 0 = the whole part, as before */
    const bool seg = part == RAD_PP_PART_SEG || ((part == RAD_PP_PART_HEAD || part == RAD_PP_PART_TAIL) && seg_l1 > 0);
    g.seg_l0.store(seg ? seg_l0 : 0, std::memory_order_relaxed);
    g.seg_l1.store(seg ? seg_l1 : 0, std::memory_order_relaxed);
    /* tp8 loop D40: only a leader head / tail segment carries RAD_PP_LSEG_* bits; every other pass resets them */
    g.lflags.store(seg && part != RAD_PP_PART_SEG ? lflags : 0, std::memory_order_relaxed);
}

int Engine::pp_frame(const RadBatch* b, bool run, std::vector<uint8_t>* out, const char** span, int64_t* span_n,
                     int lane) {
    BatchBuilder::Wire w;
    const char* sp = nullptr;
    if (!sched_of(lane).wire_last(&w, &sp) || w.step != b->step || w.draft_pass != b->draft_pass) {
        pp_link_->fail(fmt("step %d: the build to ship is not the batch's", b->step));
        return RAD_E_STATE;
    }
    StepHdr h;
    h.run = run ? 1 : 0;
    h.step = b->step;
    h.lane = lane;
    if (sched_of(lane).plan().phase != RAD_PHASE_DECODE) h.flags |= 1;
    for (const StepEntry& e : sched_of(lane).plan().e)
        if (e.home >= 0) h.flags |= 1;
    if (run && pp_ship_ahead_) h.flags |= 2;   /* tp8 dev D10 round 2 */
    if (run && pp_ship_band_) h.flags |= 4;    /* tp8 loop P28: a band pass (the follower's parts start at h + k) */
    if (run && pp_ship_tband_) h.flags |= 8;   /* tp8 loop D39: a tail band pass (the follower's parts end at t - j) */
    if (run && pp_ship_rband_) h.flags |= 32;  /* tp8 loop P41: a reverse pass (the follower's parts start at the band layer h - 1) */
    if (run && pp_ship_rkeys_) h.flags |= 64;  /* tp8 loop P41 step 5: X1 in the keys form (the follower stores layer h - 1's keys) */
    if (run && lane >= 0 && lane < 32 && ((pp_relay_mask_ >> lane) & 1u)) h.flags |= 16;   /* tp8 MERGE: a relay lane's BG step (bit 8 = D39's tail band on the dev line) */
    if (run) {
        h.n_fn4 = fn4::kSwitches;
        fn4::requests(h.fn4_n, h.fn4_v);
        h.psk_valid = psk_last_valid_ ? 1 : 0;
        h.psk_dec = psk_last_dec_;
        h.psk_n = psk_last_valid_ ? (int32_t)psk_ahead_.size() : 0;
        if (pp_clears_.size() > (size_t)kMaxClears) {
            pp_link_->fail(fmt("step %d: %zu state slots to zero, past the step frame's %d", b->step,
                               pp_clears_.size(), kMaxClears));
            return RAD_E_STATE;
        }
        h.n_clears = (int32_t)pp_clears_.size();
    }
    static thread_local std::vector<int32_t> ck;
    ck.clear();
    if (run) {
        if (const int cs = pp_ck_list(&ck, lane); cs < 0) {
            pp_link_->fail(fmt("step %d: its checkpoint restores / saves could not be listed", b->step));
            return cs;
        }
        h.n_ck = (int32_t)ck.size();
    }
    out->assign(sizeof h + (size_t)h.psk_n * 3 * sizeof(int32_t) + (size_t)h.n_clears * 2 * sizeof(int32_t) +
                    (size_t)h.n_ck * sizeof(int32_t) + sizeof w, 0);
    uint8_t* q = out->data();
    std::memcpy(q, &h, sizeof h);
    q += sizeof h;
    if (h.psk_n > 0) {
        std::memcpy(q, psk_ahead_.data(), (size_t)h.psk_n * 4); q += (size_t)h.psk_n * 4;
        std::memcpy(q, psk_rows_.data(), (size_t)h.psk_n * 4);  q += (size_t)h.psk_n * 4;
        for (int i = 0; i < h.psk_n; ++i) {
            const int32_t fl = (size_t)i < psk_flags_.size() ? psk_flags_[(size_t)i] : 0;
            std::memcpy(q, &fl, 4); q += 4;
        }
    }
    for (int i = 0; i < h.n_clears; ++i) {
        const int32_t gs[2] = { pp_clears_[(size_t)i].first, pp_clears_[(size_t)i].second };
        std::memcpy(q, gs, sizeof gs);
        q += sizeof gs;
    }
    if (h.n_ck > 0) {
        std::memcpy(q, ck.data(), (size_t)h.n_ck * sizeof(int32_t));
        q += (size_t)h.n_ck * sizeof(int32_t);
    }
    std::memcpy(q, &w, sizeof w);
    if (run) pp_clears_.clear();
    *span = sp;
    *span_n = w.used;
    return RAD_OK;
}

int Engine::pp_ship(const RadBatch* b, bool run, int lane) {
    if (!pp_link_) return RAD_OK;
    /* The drafter builds since the last trunk step, in their order, before this one (each names its
     * lane in its header: the follower replays it into that lane's builder). */
    for (auto& [step, dlane, fr] : pp_draft_q_) {
        (void)dlane;
        if (const int st = pp_link_->send(net::kStep, (uint64_t)step, 0, fr.data(), (int64_t)fr.size()); st < 0) {
            pp_draft_q_.clear();
            return st;
        }
        ++pp_builds_shipped_;
    }
    pp_draft_q_.clear();
    /* tp8 S6 (RADIANCE_PP_KV_CHECK=N): every N trunk steps, the copied KV layers' CRCs after the last
     * one, before this step's frame (the follower compares them after its own last step) */
    if (run && pp_kv_check_every_ > 0 && pp_steps_shipped_ > 0 && pp_steps_shipped_ % pp_kv_check_every_ == 0 &&
        n_lanes_ <= 1)
        pp_kv_check_send();
    std::vector<uint8_t> head;
    const char* span = nullptr;
    int64_t n = 0;
    RAD_TRY(pp_frame(b, run, &head, &span, &n, lane));
    const int st = pp_link_->send2(net::kStep, (uint64_t)b->step, run ? 1 : 0, head.data(), (int64_t)head.size(),
                                   span, n);
    if (st < 0) return st;
    ++pp_builds_shipped_;
    if (run) {
        ++pp_steps_shipped_;
        /* Robustness 2: the leader's own stage of this step has begun; its X1 is due. */
        pp_link_->note_released((uint64_t)pp_steps_shipped_);
    }
    return RAD_OK;
}

/* tp8 dev M9: pp_ship(b, true, lane) in two halves -- the drafter builds queued and the step's frame made now (at the step's
 * first head segment: its release's PSK plan, clears and checkpoint list are this step's), the frame sent at its last
 * head segment, so the follower does not issue the pass (and spin on its card for X1) while the leader's head segments and
 * the other lanes' parts between them run. */
int Engine::pp_ship_hold(const RadBatch* b, int lane) {
    if (!pp_link_) return RAD_OK;
    if (lane < 0 || lane >= RAD_PP_MAX_LANES) return RAD_E_INVAL;
    for (auto& [step, dlane, fr] : pp_draft_q_) {
        (void)dlane;
        if (const int st = pp_link_->send(net::kStep, (uint64_t)step, 0, fr.data(), (int64_t)fr.size()); st < 0) {
            pp_draft_q_.clear();
            return st;
        }
        ++pp_builds_shipped_;
    }
    pp_draft_q_.clear();
    PpHeld& h = pp_held_[lane];
    const char* span = nullptr;
    int64_t n = 0;
    RAD_TRY(pp_frame(b, true, &h.head, &span, &n, lane));
    h.span.assign(span, span + n);
    h.step = b->step;
    h.on = true;
    return RAD_OK;
}

int Engine::pp_ship_release(int lane) {
    if (!pp_link_) return RAD_OK;
    if (lane < 0 || lane >= RAD_PP_MAX_LANES || !pp_held_[lane].on) return RAD_E_STATE;
    PpHeld& h = pp_held_[lane];
    h.on = false;
    const int st = pp_link_->send2(net::kStep, (uint64_t)h.step, 1, h.head.data(), (int64_t)h.head.size(), h.span.data(),
                                   (int64_t)h.span.size());
    if (st < 0) return st;
    ++pp_builds_shipped_;
    ++pp_steps_shipped_;
    pp_link_->note_released((uint64_t)pp_steps_shipped_);
    return RAD_OK;
}

/* The leader asks the follower whether it takes a switch value (fn4::check, on an HTTP thread). The
 * follower answers between its steps (follower_loop). */
int Engine::pp_peer_check(const std::string& name, int v, std::string* why) {
    if (!pp_link_ || !pp_link_->alive()) {
        *why = "the link to the follower is down";
        return 0;
    }
    std::lock_guard<std::mutex> one(pp_ask_mu_);
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lk(pp_ans_mu_);
        id = ++pp_ask_seq_;
    }
    const std::string body = name + "\t" + std::to_string(v);
    if (pp_link_->send(net::kFn4Ask, id, 0, body.data(), (int64_t)body.size()) < 0) {
        *why = "the ask did not go out: " + pp_link_->why();
        return 0;
    }
    const int wait_ms = env_ms("RADIANCE_PP_ASK_MS", 10000);
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
    std::unique_lock<std::mutex> lk(pp_ans_mu_);
    while (pp_ans_seq_ != id) {
        if (!pp_link_->alive()) { *why = "the link to the follower went down"; return 0; }
        if (std::chrono::steady_clock::now() >= end) {
            *why = fmt("the follower did not answer within %d ms", wait_ms);
            return 0;
        }
        pp_ans_cv_.wait_for(lk, std::chrono::milliseconds(50));
    }
    *why = pp_ans_why_;
    return pp_ans_verdict_ > 0 ? 1 : -1;
}

/* tp8 loop P8: RADIANCE_PP_SEND_SPLIT=0 (default: rank 0 copies a payload into the send ring), 1 (every rank its share), 2
 * (rank 0 copies it, every other rank compares its share of its own parts: a check of 1's premise, the parts replicated
 * over the ranks); read once, said once. */
int Engine::pp_send_split() {
    static const int mode = [] {
        const char* v = std::getenv("RADIANCE_PP_SEND_SPLIT");
        const int m = v && (v[0] == '1' || v[0] == '2') && !v[1] ? v[0] - '0' : 0;
        if (v && *v && !m && std::strcmp(v, "0"))
            RAD_WARN("two-node pipeline: RADIANCE_PP_SEND_SPLIT=%s is not 0 / 1 / 2: rank 0 copies each payload", v);
        return m;
    }();
    static bool said = false;
    if (mode && !said) {
        said = true;
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_SPLIT=%d -- %s (tp8 loop P8)", mode,
                 mode == 1 ? "every rank copies its share of a crossing's payload into this node's send ring, rank 0 "
                             "publishes it (X1 on the leader, X2 on the follower)"
                           : "rank 0 copies each payload, every other rank compares its share of its own parts with "
                             "the ring's bytes (the check of RADIANCE_PP_SEND_SPLIT=1); the sums at stop");
    }
    return mode;
}

/* tp8 loop P10: RADIANCE_PP_SEND_ASYNC=N (rows; unset / 0 = off): a payload of >= N rows is packed into a
 * card staging buffer on the pass's stream and crosses PCIe from a side stream (abi/rad_pp.h RadPpRing::async_rows);
 * every pass of >= N rows is issued live. Read once. */
int64_t Engine::pp_send_async_rows() {
    static const int64_t rows = [] {
        const char* v = std::getenv("RADIANCE_PP_SEND_ASYNC");
        if (!v || !*v) return (int64_t)0;
        char* end = nullptr;
        const long long n = std::strtoll(v, &end, 10);
        if (!end || *end || n < 0) {
            RAD_WARN("two-node pipeline: RADIANCE_PP_SEND_ASYNC=%s is not a row count: every payload on the pass's stream", v);
            return (int64_t)0;
        }
        return (int64_t)n;
    }();
    return rows;
}

/* The staging buffer RADIANCE_PP_SEND_ASYNC needs on this node's sending card: the largest send ring's capacity (one
 * payload a node in flight on the side stream), 0 when off or before the crossings are declared. The budget keeps it
 * beside the step staging on rank 0 (engine_bringup.cpp), pp_serve_setup allocates it. */
int64_t Engine::pp_send_async_bytes() const {
    if (!pp_on() || pp_send_async_rows() <= 0) return 0;
    const PpGlobal& g = pp_g();
    const int id = pp_leader() ? RAD_PP_X1 : RAD_PP_X2;
    const int64_t rb = g.row_bytes[id].load();
    const int parts = std::max(1, g.parts[id].load());
    if (rb <= 0) return 0;
    return rb * cfg_.max_tok + (int64_t)parts * RAD_PP_PART_ALIGN + RAD_PP_TRAILER;
}

/* tp8 loop P10b: RADIANCE_PP_RECV_ASYNC=N (rows; unset / 0 = off): a payload of >= N rows is copied from the receive ring
 * into the rank's card staging buffer from a side stream issued with the pass, and unpacked on the pass's stream; every
 * pass of >= N rows is issued live. Read once. */
int64_t Engine::pp_recv_async_rows() {
    static const int64_t rows = [] {
        const char* v = std::getenv("RADIANCE_PP_RECV_ASYNC");
        if (!v || !*v) return (int64_t)0;
        char* end = nullptr;
        const long long n = std::strtoll(v, &end, 10);
        if (!end || *end || n < 0) {
            RAD_WARN("two-node pipeline: RADIANCE_PP_RECV_ASYNC=%s is not a row count: every payload on the pass's stream", v);
            return (int64_t)0;
        }
        return (int64_t)n;
    }();
    return rows;
}

/* The staging buffer RADIANCE_PP_RECV_ASYNC needs on every rank of this node: the receive ring's capacity. */
int64_t Engine::pp_recv_async_bytes() const {
    if (!pp_on() || pp_recv_async_rows() <= 0) return 0;
    const PpGlobal& g = pp_g();
    const int id = pp_leader() ? RAD_PP_X2 : RAD_PP_X1;
    const int64_t rb = g.row_bytes[id].load();
    const int parts = std::max(1, g.parts[id].load());
    if (rb <= 0) return 0;
    return rb * cfg_.max_tok + (int64_t)parts * RAD_PP_PART_ALIGN + RAD_PP_TRAILER;
}

/* tp8 loop P46: RADIANCE_PP_SEND_ASH=N (rows; unset / 0 = off): a row-sharded share of a payload of >= N rows crosses from its
 * card's side stream through the rank's staging buffer (abi/rad_pp.h RadPpRing::ash_rows); every pass of >= N rows is issued
 * live. Read once. */
int64_t Engine::pp_send_ash_rows() {
    static const int64_t rows = [] {
        const char* v = std::getenv("RADIANCE_PP_SEND_ASH");
        if (!v || !*v) return (int64_t)0;
        char* end = nullptr;
        const long long n = std::strtoll(v, &end, 10);
        if (!end || *end || n < 0) {
            RAD_WARN("two-node pipeline: RADIANCE_PP_SEND_ASH=%s is not a row count: every share on the pass's stream", v);
            return (int64_t)0;
        }
        return (int64_t)n;
    }();
    return rows;
}

/* The staging buffer RADIANCE_PP_SEND_ASH needs on every rank of this node: a rank's rows of the masked parts are at most
 * ceil(--max-num-batched-tokens / ranks) rows of the crossing's row bytes, and rank `sender`'s whole other parts (a
 * crossing's kv-source / keep words, a small share of its row bytes) fit the sixteenth on top; a share over it takes the pass's
 * stream (counted: RadPpRing::ash_over). 0 when off or before the crossings are declared. */
int64_t Engine::pp_send_ash_bytes() const {
    if (!pp_on() || pp_send_ash_rows() <= 0) return 0;
    const PpGlobal& g = pp_g();
    const int id = pp_leader() ? RAD_PP_X1 : RAD_PP_X2;
    const int64_t rb = g.row_bytes[id].load();
    const int parts = std::max(1, g.parts[id].load());
    if (rb <= 0) return 0;
    const int64_t tp = std::max(1, cfg_.tp);
    const int64_t q = (cfg_.max_tok + tp - 1) / tp;
    return rb * q + rb * cfg_.max_tok / 16 + (int64_t)parts * RAD_PP_PART_ALIGN;
}

/* fn4 pp_send_ash's value into this node's send rings (every rank thread parked: setup, or a switch change between steps) */
void Engine::pp_ash_apply() {
    if (pp_send_ash_rows() <= 0) return;
    const int v = fn4::pp_send_ash() == 1 ? 1 : 0;
    for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix)
        if (RadPpRing* r = pp_send_[ix]) r->ash_on = (uint64_t)v;
    if (pp_ash_on_ >= 0 && pp_ash_on_ != v) {
        uint64_t sends = 0, over = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix)
            if (const RadPpRing* r = pp_send_[ix])
                for (int q = 0; q < RAD_PP_MAX_RANKS; ++q) {
                    sends += __atomic_load_n(&r->ash_sends[q], __ATOMIC_ACQUIRE);
                    over += __atomic_load_n(&r->ash_over[q], __ATOMIC_ACQUIRE);
                }
        RAD_INFO("two-node pipeline: fn4 pp_send_ash %d -> %d on this node's send rings (%llu shares from the side streams so far, "
                 "%llu over the buffer; tp8 loop P46)", pp_ash_on_, v, (unsigned long long)sends, (unsigned long long)over);
    }
    pp_ash_on_ = v;
}

/* tp8 loop P8: what RADIANCE_PP_SEND_SPLIT=2's other ranks compared, summed over the lanes' send rings, a line a crossing. */
void Engine::pp_split_note() {
    /* tp8 loop P46: how many row-sharded shares crossed from the side streams, by rank */
    if (pp_send_ash_rows() > 0) {
        uint64_t all = 0, sends[RAD_PP_MAX_RANKS] = {}, over = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            const RadPpRing* r = pp_send_[ix];
            if (!r) continue;
            all += __atomic_load_n(&r->ready, __ATOMIC_ACQUIRE);
            for (int q = 0; q < RAD_PP_MAX_RANKS; ++q) {
                sends[q] += __atomic_load_n(&r->ash_sends[q], __ATOMIC_ACQUIRE);
                over += __atomic_load_n(&r->ash_over[q], __ATOMIC_ACQUIRE);
            }
        }
        std::string by;
        for (int q = 0; q < RAD_PP_MAX_RANKS && q < (int)ranks_.size(); ++q)
            by += fmt("%s%llu", q ? " / " : "", (unsigned long long)sends[q]);
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_ASH=%lld -- of this node's %llu %s payloads, shares from the side streams by "
                 "rank %s, %llu share(s) over the staging buffer on the pass's stream (fn4 pp_send_ash %d; tp8 loop P46)",
                 (long long)pp_send_ash_rows(), (unsigned long long)all, pp_leader() ? "X1" : "X2", by.c_str(),
                 (unsigned long long)over, fn4::pp_send_ash());
    }
    /* tp8 loop P10: how many payloads crossed from the side stream */
    if (pp_send_async_rows() > 0) {
        uint64_t all = 0, async = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            const RadPpRing* r = pp_send_[ix];
            if (!r) continue;
            all += __atomic_load_n(&r->ready, __ATOMIC_ACQUIRE);
            async += __atomic_load_n(&r->async_sends, __ATOMIC_ACQUIRE);
        }
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_ASYNC=%lld -- %llu of this node's %llu %s payloads crossed from the "
                 "side stream (tp8 loop P10)", (long long)pp_send_async_rows(), (unsigned long long)async,
                 (unsigned long long)all, pp_leader() ? "X1" : "X2");
    }
    /* tp8 loop P10b: how many payloads every rank took through its staging buffer */
    if (pp_recv_async_rows() > 0) {
        uint64_t all = 0, async = 0;
        for (int ix = 0; ix < RAD_PP_MAX_RING_IX; ++ix) {
            const RadPpRing* r = pp_recv_[ix];
            if (!r) continue;
            for (int q = 0; q < RAD_PP_MAX_RANKS && (uint64_t)q < r->n_ranks; ++q) {
                all += __atomic_load_n(&r->taken[q], __ATOMIC_ACQUIRE);
                async += __atomic_load_n(&r->recv_async[q], __ATOMIC_ACQUIRE);
            }
        }
        RAD_INFO("two-node pipeline: RADIANCE_PP_RECV_ASYNC=%lld -- %llu of this node's %llu %s receives (all ranks) came "
                 "through the staging buffers from the side stream (tp8 loop P10b)", (long long)pp_recv_async_rows(),
                 (unsigned long long)async, (unsigned long long)all, pp_leader() ? "X2" : "X1");
    }
    for (int id = 0; id < RAD_PP_N_RINGS; ++id) {
        uint64_t payloads = 0, cmp = 0, diff = 0, mode = 0, by[RAD_PP_MAX_RANKS] = {}, part[RAD_PP_MAX_PARTS] = {};
        uint64_t pbytes[RAD_PP_MAX_PARTS] = {}, nparts = 0;
        for (int lane = 0; lane < n_lanes_; ++lane) {
            const RadPpRing* r = pp_send_[lane * RAD_PP_N_RINGS + id];
            if (!r || !r->send_split) continue;
            mode = r->send_split;
            payloads += __atomic_load_n(&r->ready, __ATOMIC_ACQUIRE);
            nparts = std::max<uint64_t>(nparts, std::min<uint64_t>(r->parts, RAD_PP_MAX_PARTS));
            for (int k = 0; k < RAD_PP_MAX_PARTS; ++k) pbytes[k] = std::max<uint64_t>(pbytes[k], r->part_bytes[k]);
            for (int q = 0; q < RAD_PP_MAX_RANKS && (uint64_t)q < r->send_ranks; ++q) {
                cmp += __atomic_load_n(&r->cmp_words[q], __ATOMIC_ACQUIRE);
                const uint64_t d = __atomic_load_n(&r->diff_words[q], __ATOMIC_ACQUIRE);
                by[q] += d;
                diff += d;
                for (int k = 0; k < RAD_PP_MAX_PARTS; ++k) part[k] += __atomic_load_n(&r->diff_part[q][k], __ATOMIC_ACQUIRE);
            }
        }
        if (mode != 2) continue;
        std::string per, parts;
        for (int q = 1; q < cfg_.tp && q < RAD_PP_MAX_RANKS; ++q) per += fmt("%srank %d %llu", q > 1 ? ", " : "", q,
                                                                          (unsigned long long)by[q]);
        for (uint64_t k = 0; k < nparts; ++k)
            parts += fmt("%spart %llu (%llu B in the last payload) %llu", k ? ", " : "", (unsigned long long)k,
                         (unsigned long long)pbytes[k], (unsigned long long)part[k]);
        RAD_INFO("two-node pipeline: RADIANCE_PP_SEND_SPLIT=2 check of X%d's %llu payloads: %llu 4-byte words compared, "
                 "%llu differ (%s; by part, in this node's crossing line's order: %s)%s", id + 1, (unsigned long long)payloads,
                 (unsigned long long)cmp, (unsigned long long)diff, per.c_str(), parts.c_str(),
                 diff ? " -- the parts are NOT replicated: do not use RADIANCE_PP_SEND_SPLIT=1" : " -- every share = rank 0's bytes");
    }
}

void Engine::pp_close(const char* why) {
    if (pp_stamps_) { std::fclose(pp_stamps_); pp_stamps_ = nullptr; }
    if (!pp_link_ || pp_closed_) return;
    pp_closed_ = true;
    pp_split_note();
    if (pp_leader()) fn4::set_peer_check(nullptr);
    pp_link_->close(why);
    RAD_INFO("two-node pipeline: link closed (%s); %lld steps and %lld builds shipped", why,
             (long long)pp_steps_shipped_, (long long)pp_builds_shipped_);
}

/* ================================================================== tp8 S6: checkpoints, the KV-copy check */
/* The step's checkpoint restores (kind 0: before the pass) and saves (kind 1: behind it), as the leader's step
 * loop and save_checkpoints run them: each [kind, checkpoint slot, n state slots, n window rows, lever TI's first
 * row (-1: none), the sequence's state slots (per stateful group, plans() order), the window block ids], from the
 * leader's tables. The follower's
 * groups are the leader's in the same order, so the ids address its own pools and checkpoint store. */
int Engine::pp_ck_list(std::vector<int32_t>* out, int lane) {
    out->clear();
    /* tp8 S5 + S6: the lane's plan (its scheduler's lock is lane 0's: one lock for every lane) */
    Scheduler& ls = sched_of(lane);
    const StepPlan& plan = ls.plan();
    bool any = false;
    for (const StepEntry& e : plan.e) any = any || e.ckpt_restore >= 0 || (e.ckpt_slot >= 0 && e.ckpt_tok >= 0);
    if (!any) return RAD_OK;
    return ls.locked([&]() -> int {
        std::vector<int32_t> slots, win;
        for (const StepEntry& e : plan.e)
            for (int kind = 0; kind < 2; ++kind) {
                int32_t ck = -1;
                int64_t pos = 0;
                if (kind == 0) {
                    if (e.ckpt_restore < 0) continue;
                    ck = e.ckpt_restore;
                    pos = e.ctx_len;
                } else {
                    if (e.ckpt_slot < 0 || e.ckpt_tok < 0) continue;
                    ck = e.ckpt_slot;
                    pos = e.ctx_len + e.ckpt_tok + 1;
                }
                if (!state_slots_of(e.seq, &slots)) continue;
                win.clear();
                if (ranks_[0]->kv.has_window_snapshot() && !ranks_[0]->kv.window_rows(e.seq, pos, &win)) return RAD_E_STATE;
                /* lever TI (save_checkpoints): a snapshot inside the entry's chunk patches the CONV window with the
                 * plugin's mapped ids of the rows ending at it -- the step row after the snapshot, or -1 */
                int32_t first = -1;
                if (kind == 1 && e.ckpt_tok < e.n_tokens - 1) {
                    int64_t row0 = 0;
                    for (const StepEntry& o : plan.e) {
                        if (&o == &e) break;
                        row0 += o.n_tokens;
                    }
                    first = (int32_t)(row0 + e.ckpt_tok + 1);
                }
                out->push_back(kind);
                out->push_back(ck);
                out->push_back((int32_t)slots.size());
                out->push_back((int32_t)win.size());
                out->push_back(first);
                out->insert(out->end(), slots.begin(), slots.end());
                out->insert(out->end(), win.begin(), win.end());
            }
        return RAD_OK;
    });
}

int Engine::pp_ck_apply(const int32_t* q, int32_t n, bool save) {
    int32_t i = 0;
    std::vector<int32_t> win;
    constexpr int32_t H = 5;   /* kind, slot, n state slots, n window rows, the TI patch's first row */
    while (i + H <= n) {
        const int32_t kind = q[i], ck = q[i + 1], ns = q[i + 2], nw = q[i + 3], first = q[i + 4];
        if (ns < 0 || nw < 0 || (int64_t)i + H + ns + nw > n) return RAD_E_STATE;
        if ((kind == 1) == save) {
            /* tp8 F2: the slot ids are the leader's (its store hands them out and gives them back); this node's store
             * mirrors them, so a save takes the leader's id here (it is never handed out on this node) and a restore
             * finds what the save behind an earlier pass wrote. A give-back is not mirrored: a slot the leader reuses
             * is simply written again. */
            if (save && ck_store_.enabled() && !ck_store_.in_use(ck) && !ck_store_.take(ck)) {
                RAD_ERR("follower: the leader's checkpoint slot %d is outside this node's store of %lld slots (start both "
                        "nodes with the same --checkpoint-slots)", (int)ck, (long long)ck_store_.capacity());
                return RAD_E_STATE;
            }
            win.assign(q + i + H + ns, q + i + H + ns + nw);
            for (auto& rk : ranks_) {
                if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                int st = save ? rk->kv.checkpoint_save(q + i + H, ns, ck, rk->ctx.stream(), &win)
                              : rk->kv.checkpoint_restore(q + i + H, ns, ck, rk->ctx.stream(), &win);
                /* lever TI: the CONV window of a snapshot inside the chunk, from this node's mapped ids -- only on a node that
                 * holds an Engram layer (ti_capable_: the plugin's rad_ckpt_conv_src probe). A node with none keeps the
                 * CONV group (every group of the published declare) but no layer of it reads the group, so its rows in a
                 * snapshot are never read here and stay the chunk end's (loop FINDINGS F27: W(15,35)'s follower, behind a
                 * TI-capable leader, failed the pair at the first such snapshot) */
                if (st >= 0 && save && first >= 0 && rk->kv.has_conv_state() && ti_capable_) {
                    const void* ids = nullptr;
                    int ctx = 0;
                    if (fn4::ckpt_conv_src(rk->index, &ids, &ctx) <= 0 || !ids || ctx <= 0 || first < ctx) {
                        RAD_ERR("follower: rank %d has no source of the CONV window at row %d for a snapshot inside "
                                "the chunk", rk->index, (int)first);
                        st = RAD_E_STATE;
                    } else {
                        st = rk->kv.checkpoint_patch_conv(ck, (const char*)ids + (int64_t)(first - ctx) * 4, ctx,
                                                          rk->ctx.stream());
                    }
                }
                if (st < 0) {
                    RAD_ERR("follower: rank %d could not %s checkpoint slot %d: %s", rk->index,
                            save ? "save into" : "restore from", (int)ck, rad_strerror(st));
                    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                    return st;
                }
            }
            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
        }
        i += H + ns + nw;
    }
    return i == n ? RAD_OK : RAD_E_STATE;
}

/* CRC-32C of a copied layer's first `top` blocks on rank 0 (its stream drained first). */
int Engine::pp_kv_crc(int32_t group, int32_t slot, int64_t top, uint32_t* crc, int64_t* bytes) {
    const KVGroupPlan* p = ranks_[0]->kv.plan(group);
    if (!p || !p->base || p->zero || !p->paged() || slot < 0 || slot >= p->n_layers) return RAD_E_STATE;
    const int64_t per = p->layer_bytes_per_block(), nb = std::max<int64_t>(0, std::min<int64_t>(top, p->pool_blocks()));
    const char* base = (const char*)p->base + (int64_t)slot * p->layer_stride();
    *bytes = nb * per;
    RadStream s = ranks_[0]->ctx.stream();
    if (rad_stream_sync(s) < 0) return RAD_E_DEVICE;
    std::vector<char> buf((size_t)std::min<int64_t>(std::max<int64_t>(*bytes, 1), 64ll << 20));
    uint32_t c = 0;
    /* RADIANCE_PP_KV_CHECK_DUMP=dir: each node also writes the bytes it hashed (debug of the debug) */
    const char* dd = std::getenv("RADIANCE_PP_KV_CHECK_DUMP");
    FILE* df = nullptr;
    if (dd && *dd) df = std::fopen(fmt("%s/kvcheck-%s-g%d-s%d-%lld.bin", dd, role(cfg_.pp_rank), (int)group, (int)slot,
                                       (long long)pp_kv_checks_).c_str(), "wb");
    for (int64_t off = 0; off < *bytes; off += (int64_t)buf.size()) {
        const int64_t k = std::min<int64_t>((int64_t)buf.size(), *bytes - off);
        if (rad_memcpy_async(buf.data(), base + off, k, s) < 0 || rad_stream_sync(s) < 0) {
            if (df) std::fclose(df);
            return RAD_E_DEVICE;
        }
        c = net::crc32c(buf.data(), (size_t)k, c);
        if (df) std::fwrite(buf.data(), 1, (size_t)k, df);
    }
    if (df) std::fclose(df);
    *crc = c;
    return RAD_OK;
}

/* The leader: [n, then per copied layer: group, layer, blocks, crc, bytes] after trunk step pp_steps_shipped_. */
void Engine::pp_kv_check_send() {
    std::vector<std::array<int32_t, 3>> regs;
    {
        PpGlobal& g = pp_g();
        std::lock_guard<std::mutex> lk(g.kv_mu);
        regs = g.kv_copies;
    }
    std::vector<int64_t> body(1, 0);
    for (const auto& c : regs) {
        const int64_t top = ranks_[0]->kv.held_top_blocks(c[0]);
        uint32_t crc = 0;
        int64_t bytes = 0;
        if (pp_kv_crc(c[0], c[1], top, &crc, &bytes) < 0) continue;
        body.insert(body.end(), { (int64_t)c[0], (int64_t)c[2], top, (int64_t)crc, bytes });
        ++body[0];
    }
    if (pp_link_->send(net::kKvCheck, (uint64_t)pp_steps_shipped_, 0, body.data(), (int64_t)(body.size() * 8)) < 0)
        RAD_WARN("two-node pipeline: the KV-copy check after step %lld did not go out", (long long)pp_steps_shipped_);
    ++pp_kv_checks_;
}

/* The follower: its own CRC of each layer the leader named, over the same blocks, after its own last step. */
int Engine::pp_kv_check_recv(const std::vector<uint8_t>& body, uint64_t step) {
    if (body.size() < 8 || body.size() % 8) return RAD_OK;
    std::vector<int64_t> v(body.size() / 8);
    std::memcpy(v.data(), body.data(), body.size());
    std::vector<std::array<int32_t, 3>> regs;
    {
        PpGlobal& g = pp_g();
        std::lock_guard<std::mutex> lk(g.kv_mu);
        regs = g.kv_copies;
    }
    const int64_t n = v[0];
    for (int64_t k = 0; k < n && (size_t)(1 + 5 * k + 4) < v.size(); ++k) {
        const int64_t* e = &v[(size_t)(1 + 5 * k)];
        const int32_t grp = (int32_t)e[0], layer = (int32_t)e[1];
        int32_t slot = -1;
        for (const auto& c : regs)
            if (c[0] == grp && c[2] == layer) slot = c[1];
        uint32_t crc = 0;
        int64_t bytes = 0;
        const KVGroupPlan* p = ranks_[0]->kv.plan(grp);
        if (slot < 0 || pp_kv_crc(grp, slot, e[2], &crc, &bytes) < 0) {
            ++pp_kv_bad_;
            RAD_ERR("follower: KV-copy check after trunk step %llu: %s layer %d is not held here",
                    (unsigned long long)step, p ? p->name.c_str() : "?", (int)layer);
            continue;
        }
        ++pp_kv_checks_;
        if ((uint32_t)e[3] == crc && e[4] == bytes) {
            RAD_INFO("follower: KV-copy check after trunk step %llu: %s layer %d, %lld blocks (%s): CRC-32C %08x on "
                     "both nodes (%lld checks, %lld differ)", (unsigned long long)step, p->name.c_str(), (int)layer,
                     (long long)e[2], humanb(bytes).c_str(), crc, (long long)pp_kv_checks_, (long long)pp_kv_bad_);
        } else {
            ++pp_kv_bad_;
            RAD_ERR("follower: KV-copy check after trunk step %llu: %s layer %d, %lld blocks: the leader's CRC-32C %08x "
                    "(%lld bytes), this node's %08x (%lld bytes) -- the copy is NOT the owner's bytes (%lld checks, %lld "
                    "differ)", (unsigned long long)step, p->name.c_str(), (int)layer, (long long)e[2], (unsigned)e[3],
                    (long long)e[4], crc, (long long)bytes, (long long)pp_kv_checks_, (long long)pp_kv_bad_);
        }
    }
    return RAD_OK;
}

/* ================================================================== the follower */
/* Its applied switch values, to the leader (GET /admin/fn4's "peer"). */
void Engine::pp_send_fn4_state() {
    if (!pp_link_ || !pp_link_->alive()) return;
    int32_t v[fn4::kSwitches];
    for (int i = 0; i < fn4::kSwitches; ++i) v[i] = fn4::value(i);
    (void)pp_link_->send(net::kFn4State, 0, 0, v, (int64_t)sizeof v);
}

/* tp8 dev D8 (RADIANCE_PP_NODE_IDLE=1, either node): rank 0's card time in parts, between them and in its receives' waits
 * (core/pp_nodeidle.h); 1 when on (`*own` holds it, pp_ni_ points at it), 0 when off. */
int Engine::pp_node_idle_open(std::unique_ptr<PpNodeIdle>* own) {
    pp_ni_ = nullptr;
    const char* v = std::getenv("RADIANCE_PP_NODE_IDLE");
    if (!v || (std::strcmp(v, "1") && std::strcmp(v, "2")) || ranks_.empty()) return 0;
    *own = std::make_unique<PpNodeIdle>();
    PpNodeIdle& ni = **own;
    ni.split = !std::strcmp(v, "2");   /* tp8 loop D45: + the line by stretch (BG step in flight or not) */
    if (const char* ps = std::getenv("RADIANCE_PP_NODE_IDLE_S"); ps && *ps) {   /* tp8 loop D45: the lines' period */
        const int n = std::atoi(ps);
        if (n >= 1 && n <= 600) ni.period_s = n;
        else RAD_WARN("RADIANCE_PP_NODE_IDLE_S=%s is not taken (1..600 s): the node-idle lines stay a minute apart", ps);
    }
    const bool lead = !pp_follower();
    ni.node = lead ? "leader" : "follower";
    ni.crossing = lead ? "X2" : "X1";
    ni.s = ranks_[0]->ctx.stream();
    for (int l = 0; l < n_lanes_; ++l)
        if (RadPpRing* r = pp_recv_[l * RAD_PP_N_RINGS + (lead ? RAD_PP_X2 : RAD_PP_X1)]) ni.rings.push_back(r);
    /* tp8 loop DECODE round 20: the node's cards for the VRAM line (rad_dev_props also opens the driver's counter) */
    for (const auto& r : ranks_) {
        RadDeviceProps pr{};
        ni.devs.push_back(r->device);
        ni.totals.push_back(rad_dev_props(r->device, &pr) >= 0 && pr.vram_bytes > 0 ? pr.vram_bytes : -1);
    }
    ni.arm();
    pp_ni_ = own->get();
    RAD_INFO("two-node pipeline: RADIANCE_PP_NODE_IDLE=1 -- rank 0's card time in parts, between parts and waiting on the "
             "card for the %s's %s (%zu ring(s)), a line a minute and the run's at stop (tp8 dev D8)",
             lead ? "follower" : "leader", ni.crossing, ni.rings.size());
    if (ni.split || ni.period_s != 60)
        RAD_INFO("two-node pipeline: RADIANCE_PP_NODE_IDLE=%s -- %sthe lines every %d s (tp8 loop D45)", v,
                 ni.split ? (lead ? "+ the same time by stretch, with a BG step in flight (a BG lane busy or holding a newcomer at "
                                    "the round's top) or without; "
                                  : "+ the same time by stretch, with a BG lane's frame in pass on this node or without; ")
                          : "",
                 ni.period_s);
    return 1;
}

int Engine::follower_loop() {
    if (rad_dev_set(ranks_[0]->device) < 0) RAD_ERR("follower loop: could not bind rank 0's device");
    rad::StallWatch watch("follower", 20000);
    int64_t steps = 0, builds = 0;
    std::unique_ptr<PpNodeIdle> ni_own;   /* tp8 dev D8 */
    struct NiReset { Engine* e; ~NiReset() { e->pp_ni_ = nullptr; } } ni_reset{ this };
    pp_node_idle_open(&ni_own);
    /* tp8 loop D51 (RADIANCE_PP_LOAD_REPORT_MS=N, 1 .. 1000, with RADIANCE_PP_NODE_IDLE): this follower's recent card working share
     * over RADIANCE_PP_TBAND_BOUND_MS (default 200) goes to the leader every N ms in a heartbeat's aux (fn4 pp_tband 5 reads it) */
    pp_load_ms_ = 0;
    if (const char* v = std::getenv("RADIANCE_PP_LOAD_REPORT_MS"); v && *v && pp_link_) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= 1000 && pp_ni_) {
            pp_load_ms_ = n;
            pp_bound_ms_ = 200;
            if (const char* w = std::getenv("RADIANCE_PP_TBAND_BOUND_MS"); w && *w) pp_bound_ms_ = std::min(5000, std::max(20, std::atoi(w)));
            pp_ni_->recent_keep_ns = 4ull * (uint64_t)pp_bound_ms_ * 1000000ull;
            pp_link_->set_load_period(n);
            RAD_INFO("follower: RADIANCE_PP_LOAD_REPORT_MS=%d -- this node's card working share over the last %d ms goes to the leader "
                     "every %d ms (a heartbeat's aux; fn4 pp_tband 5 there, tp8 loop D51)", n, pp_bound_ms_, n);
        } else {
            RAD_WARN("follower: RADIANCE_PP_LOAD_REPORT_MS=%s is not taken: %s (tp8 loop D51)", v,
                     pp_ni_ ? "1 .. 1000 ms" : "it needs RADIANCE_PP_NODE_IDLE=1 / 2 on this node");
        }
    }
    /* tp8 loop P22 (RADIANCE_PP_PF_TIMELINE=1, the leader's switch, on this node too): the follower's side of a lone cold
     * prompt's timeline (core/pp_pftl_fol.h) */
    std::unique_ptr<PfFolTimeline> pff;
    if (const char* v = std::getenv("RADIANCE_PP_PF_TIMELINE"); v && !std::strcmp(v, "1") && !ranks_.empty()) {
        pff = std::make_unique<PfFolTimeline>();
        pff->s = ranks_[0]->ctx.stream();
        for (int l = 0; l < n_lanes_; ++l)
            if (RadPpRing* r = pp_recv_[l * RAD_PP_N_RINGS + RAD_PP_X1]) pff->rings.push_back(r);
        pff->arm();
        RAD_INFO("follower: RADIANCE_PP_PF_TIMELINE=1 -- a lone prefill's passes are timed on rank 0's card (with their X1 waits, "
                 "%zu ring(s)), a line a prompt once its first decode step or the next prompt arrives (tp8 loop P22)",
                 pff->rings.size());
    }
    std::vector<int32_t> ahead, rows, flags, clears, ckw;
    /* tp8 dev D10 (RADIANCE_PP_ISSUE_AHEAD=1): the leader ships an undrafted lane's next step before it has that step's token
     * ids (its card writes them behind the sampler), so this node must read none: refused where its pass computes n-gram ids
     * from them on the host (deepseek41's Engram layers) or embeds them (RADIANCE_PP_FOLLOWER_EMBED=1). The pass itself waits on
     * its card for X1 as always. */
    if (const char* ia = std::getenv("RADIANCE_PP_ISSUE_AHEAD"); ia && !std::strcmp(ia, "1")) {
        std::string why;
        for (const OpInfo& o : ranks_[0]->program.ops)
            if (o.op == "ngram_ids") {
                why = "its pass computes n-gram ids from the step's token ids on the host (an Engram layer on this node)";
                break;
            }
        const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
        const bool fe_on = fe && !std::strcmp(fe, "1") && cfg_.pp_cut_h == 0;
        /* tp8 dev D10 round 2: the plugin's X1 carries the step's token ids (the leader's card's) and this node's pass takes
         * them first, into its own batch: then it may embed / hash them whatever the frame's host ids were */
        if (pp_x1_ids()) {
            RAD_INFO("follower: RADIANCE_PP_ISSUE_AHEAD=1 -- this node's pass takes the step's token ids from X1 (the leader's "
                     "card's) before it reads any%s%s; the leader may ship a lane's next step before its collect (tp8 dev D10 "
                     "round 2)", fe_on ? " (it embeds them: RADIANCE_PP_FOLLOWER_EMBED=1)" : "",
                     why.empty() ? "" : " (an Engram layer here hashes them)");
        } else if (fe && !std::strcmp(fe, "1")) {
            /* the leader does not take it beside the follower's embedding (engine_lanes.cpp): nothing is shipped early */
            RAD_INFO("follower: RADIANCE_PP_ISSUE_AHEAD=1 is not taken with RADIANCE_PP_FOLLOWER_EMBED=1 (this node embeds the "
                     "step's token ids; the leader issues nothing ahead) (tp8 dev D10)");
        } else if (!why.empty()) {
            RAD_ERR("follower: RADIANCE_PP_ISSUE_AHEAD=1 refused: %s, and a step the leader issues ahead of its collect ships "
                    "without them -- a cut whose follower holds no such layer (deepseek41: --pp-cut 15,35 keeps both Engram "
                    "layers on the leader), or unset it (tp8 dev D10)", why.c_str());
            pp_link_->fail("the follower refused RADIANCE_PP_ISSUE_AHEAD=1");
            return fail_loop(RAD_E_UNSUPPORTED);
        } else {
            RAD_INFO("follower: RADIANCE_PP_ISSUE_AHEAD=1 -- this node's pass reads no token id; the leader may ship a lane's "
                     "next step before its collect (tp8 dev D10)");
        }
    }
    pp_send_fn4_state();
    /* tp8 dev M4 (RADIANCE_PP_SEG_LAYERS=k, read by the plugin, which declares the boundaries; RADIANCE_PP_SEG_ROWS=n, default
     * 512): a pass of >= n rows is issued in layer segments at the declared boundaries, and between two segments every step
     * frame of another lane that has arrived and carries no lent piece runs its whole pass (the frames behind a held one wait
     * with it, in order). The segment boundary saves and restores the pass's live set on each rank (the stage ops' loop
     * mode), so a pass between segments may use the arena freely. RADIANCE_PP_SEG_AHEAD (default 1): segments queued on the
     * card ahead of the running one; RADIANCE_PP_SEG_TEST_WAIT_MS (host tests): wait that long for a frame at each gap. */
    const std::vector<int32_t> segs = pp_segments();
    int64_t seg_rows = 512, seg_ahead = 1, seg_wait_ms = 0;
    /* only passes with decode rows beside their prompt rows (a newcomer's MIXED step: another lane decodes) unless
     * RADIANCE_PP_SEG_PREFILL=1 (a lone prefill chunk too: its segments cost a boundary each and nothing decodes beside it) */
    bool seg_prefill = false;
    if (const char* v = std::getenv("RADIANCE_PP_SEG_PREFILL"); v && !std::strcmp(v, "1")) seg_prefill = true;
    /* tp8 dev M8 (RADIANCE_PP_BG_LANE=k on this node too): the BG lanes' passes -- newcomers lent whole, no decode rows -- run in
     * segments with the decode lanes' passes between them; other lone prefill passes (a cold prompt's) stay whole */
    const int seg_bg = LaneSet::bg_env() > 0 && LaneSet::bg_env() < n_lanes_ ? n_lanes_ - LaneSet::bg_env() : -1;
    if (const char* v = std::getenv("RADIANCE_PP_SEG_ROWS"); v && *v) seg_rows = std::max<int64_t>(1, std::atoll(v));
    if (const char* v = std::getenv("RADIANCE_PP_SEG_AHEAD"); v && *v) seg_ahead = std::min<int64_t>(1, std::max<int64_t>(0, std::atoll(v)));
    if (const char* v = std::getenv("RADIANCE_PP_SEG_TEST_WAIT_MS"); v && *v) seg_wait_ms = std::max<int64_t>(0, std::atoll(v));
    RadEvent seg_ev[2] = {};
    if (seg_bg >= 0 && segs.empty())
        RAD_WARN("follower: RADIANCE_PP_BG_LANE is set but no layer segments are declared (RADIANCE_PP_SEG_LAYERS on this node): the "
                 "BG lanes' passes run whole and every decode pass queues behind them, as a MIXED step's did (tp8 dev M8)");
    if (!segs.empty()) {
        std::string b;
        for (int32_t x : segs) b += fmt("%s%d", b.empty() ? "" : ", ", (int)x);
        for (RadEvent& e : seg_ev)
            if (rad_event_create(&e) < 0) {
                RAD_ERR("follower: could not create the segment events");
                return fail_loop(RAD_E_DEVICE);
            }
        RAD_INFO("follower: passes of >= %lld rows%s run in layer segments at the boundaries %s of W(%d,%d), other lanes' "
                 "passes between them (tp8 dev M4; %lld segment(s) queued ahead)%s", (long long)seg_rows,
                 seg_prefill ? "" : " with decode rows (MIXED; RADIANCE_PP_SEG_PREFILL=1: lone prefill chunks too)", b.c_str(),
                 cfg_.pp_cut_h, cfg_.pp_cut_t, (long long)seg_ahead,
                 seg_bg >= 0 ? fmt("; and every pass of the BG lane(s) %d-%d (RADIANCE_PP_BG_LANE, tp8 dev M8)", seg_bg, n_lanes_ - 1).c_str() : "");
    }
    int64_t seg_passes = 0, seg_parts = 0, seg_between = 0, seg_held = 0;
    /* tp8 dev D10 round 2: a frame the leader issued ahead of its lane's collect (flags bit 1) arrives about a leader tail
     * before its X1; between a split pass's segments it waits here, in arrival order with the decode frames behind it, until
     * its X1 has landed -- so the card runs the split pass's next segments instead of spinning in its X1 receive. `x1_runs`:
     * the passes issued a lane (each takes one X1 payload), so a frame's payload is number x1_runs + 1 of its lane's ring. */
    std::deque<net::Msg> ia_pend;
    int64_t x1_runs[RAD_PP_MAX_LANES] = {};
    int64_t ia_deferred = 0;
    /* tp8 loop D15 (RADIANCE_PP_FOL_ORDER=bits, this node; unset / 0 = the order above, every frame behind a held one held):
     * the follower's card spins in a pass's X1 receive whenever the leader's stream has not sent that X1 yet (a BG / MIXED
     * head or tail ahead of it there), while a split pass's next segments wait behind the spin (loop FINDINGS F44 14:05 /
     * PROFILE 07:00: X1 wait 13-17 % of this card at c32, the follower's largest idle item). Between a split pass's segments:
     * bit 1 -- every decode frame of another lane waits here until its X1 payload has landed (not only one the leader issued
     * ahead), and the waiting frames go in the order their X1s land, each lane's own in its order (a frame with a switch
     * request is a barrier) -- the card runs the split pass's segments meanwhile; bit 2 -- a frame that keeps its place
     * (prompt rows, a lent piece, the split pass's lane: StepHdr bit 0) holds only the frames of ITS lane behind it, not every
     * frame: a decode frame of a lane with nothing held (and no switch request, clears or checkpoint words, none held before it
     * with a switch request) runs between the segments as if nothing were held. Only the order of different lanes' passes on
     * this node changes, never a pass: a decode step reads only committed KV (bit 0's rule), its own lane's frames keep their
     * order. Bit 1 is not taken where this node's pass takes no X1 (RADIANCE_PP_FOLLOWER_EMBED=1 at W(0, t) without the D10
     * ids). Bit 4 (with bit 1, which it sets): the next segment waits up to RADIANCE_PP_FOL_X1_GRACE_MS (default 10) after a
     * waiting frame arrived for its X1 -- one whose leader head runs now lands within ~a head's time and goes first, as today;
     * one whose head is queued behind the leader's BG / MIXED work lets the segments run. */
    int fol_order = 0;
    if (const char* v = std::getenv("RADIANCE_PP_FOL_ORDER"); v && *v) fol_order = std::atoi(v) & 7;
    if (fol_order & 4) fol_order |= 1;
    uint64_t fo_grace_ns = 10ull * 1000000ull;
    if (const char* v = std::getenv("RADIANCE_PP_FOL_X1_GRACE_MS"); v && *v)
        fo_grace_ns = (uint64_t)std::max(0LL, std::atoll(v)) * 1000000ull;
    {
        const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
        const bool fe_on = fe && !std::strcmp(fe, "1") && cfg_.pp_cut_h == 0;
        if ((fol_order & 1) && fe_on && !pp_x1_ids()) {
            RAD_WARN("follower: RADIANCE_PP_FOL_ORDER bit 1 is not taken here: this node's pass takes no X1 "
                     "(RADIANCE_PP_FOLLOWER_EMBED=1 at W(0, t)); bit 2 stays (tp8 loop D15)");
            fol_order &= ~5;
        }
        if (fol_order)
            RAD_INFO("follower: RADIANCE_PP_FOL_ORDER=%d -- between a split pass's segments %s%s%s (tp8 loop D15)", fol_order,
                     (fol_order & 1) ? "another lane's decode frame goes once its X1 has landed, in the order the X1s land"
                                     : "",
                     (fol_order & 3) == 3 ? "; " : "",
                     (fol_order & 2) ? "a held frame holds only its own lane's frames behind it" : "");
        if (fol_order & 4)
            RAD_INFO("follower: RADIANCE_PP_FOL_ORDER bit 4 -- a segment waits up to %llu ms after a waiting frame arrived for its "
                     "X1 (RADIANCE_PP_FOL_X1_GRACE_MS, tp8 loop D15)", (unsigned long long)(fo_grace_ns / 1000000ull));
    }
    /* tp8 loop IDEAS #9 (RADIANCE_PP_FOL_RESERVE=L, this node; unset / 0 = off): RELAY by a body's reserve. A BG lane's split
     * pass (a newcomer's body: prompt rows only, RADIANCE_PP_BG_LANE) issues its segments up to layer L as before and then holds
     * the rest (its parts from L on) until a step frame of ANOTHER BG lane with prompt rows has arrived -- the leader has issued
     * the next newcomer's head, whose X1 is about a head's time away -- or RADIANCE_PP_FOL_RESERVE_MS (default 300) have passed
     * since the hold began; meanwhile every frame runs between as at any segment gap. So the held layers fill this card while
     * the leader runs the next head (at c32 the follower's whole waiting is that head's shadow: loop FINDINGS DECODE note
     * 13:51), and the held newcomer's leader tail then runs beside the next body. Only WHEN a pass's later segments are issued
     * changes, never a pass or its bytes (the M4 segment gaps' rule); its first token comes later by the hold. L must be a
     * declared segment boundary inside W(h, t); with RADIANCE_PP_FOL_ORDER bit 2 the decode frames behind the next newcomer's
     * held frame keep running too. */
    int fol_reserve = 0;
    uint64_t fol_reserve_ns = 300ull * 1000000ull;
    if (const char* v = std::getenv("RADIANCE_PP_FOL_RESERVE"); v && *v) fol_reserve = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("RADIANCE_PP_FOL_RESERVE_MS"); v && *v)
        fol_reserve_ns = (uint64_t)std::max(1LL, std::atoll(v)) * 1000000ull;
    if (fol_reserve > 0) {
        const bool at_boundary = std::find(segs.begin(), segs.end(), (int32_t)fol_reserve) != segs.end() &&
                                 fol_reserve > cfg_.pp_cut_h && fol_reserve < cfg_.pp_cut_t;
        if (seg_bg < 0 || !at_boundary) {
            RAD_WARN("follower: RADIANCE_PP_FOL_RESERVE=%d is not taken here: %s (tp8 loop IDEAS #9)", fol_reserve,
                     seg_bg < 0 ? "no BG lane on this node (RADIANCE_PP_BG_LANE)"
                                : "it is not one of the declared segment boundaries inside W(h, t) (RADIANCE_PP_SEG_LAYERS)");
            fol_reserve = 0;
        } else {
            RAD_INFO("follower: RADIANCE_PP_FOL_RESERVE=%d -- a BG newcomer's body holds its layers %d..%d until another BG lane's "
                     "frame arrives or %llu ms pass (RADIANCE_PP_FOL_RESERVE_MS), every other frame running meanwhile (tp8 loop "
                     "IDEAS #9)", fol_reserve, fol_reserve, cfg_.pp_cut_t - 1, (unsigned long long)(fol_reserve_ns / 1000000ull));
        }
    }
    /* fn4 pp_fol_reserve (93, the follower's; its check takes these boundaries): the start value is the env's, the leader's
     * request moves it in process (0 = off) */
    uint64_t res_mask = 0;
    if (seg_bg >= 0)
        for (int32_t x : segs)
            if (x > cfg_.pp_cut_h && x < cfg_.pp_cut_t && x < 63) res_mask |= 1ull << x;
    fn4::reserve_register(res_mask);
    auto reserve_now = [&]() -> int {
        int v = fn4::get(fn4::kPpFolReserve);
        if (v < 0) v = fol_reserve;   /* never requested: the start's */
        return v > 0 && v < 63 && ((res_mask >> v) & 1) ? v : 0;
    };
    /* tp8 loop D33 (fn4 pp_seg_set, the follower's; RADIANCE_PP_SEG_SET=v its start value, unset 0 = every declared boundary):
     * which declared boundaries inside W(h, t) split a pass -- every s-th (s = v / 64; 0 = 1), then the first n = v % 64 of those
     * (0: all), and a body reserve's boundary whatever the set. With RADIANCE_PP_SEG_MAX=32 the plugin declares past the old 15
     * (at W(10,32) SEG_LAYERS=1: 11..31), so one start compares 15 (= 11..25, the end lump [25,32) of the program before D33), 0
     * (11..31) and 128 (12, 14 .. 30 = SEG_LAYERS=2's set). Only where a pass's segments end changes, never its ops (M4's rule). */
    std::vector<int32_t> seg_in;
    for (int32_t x : segs)
        if (x > cfg_.pp_cut_h && x < cfg_.pp_cut_t) seg_in.push_back(x);
    fn4::segset_register((int)seg_in.size());
    auto segset_ok = [](int v) { return v >= 0 && v <= 511 && (v & 63) <= 31; };
    int segset_start = 0;
    const char* segset_env = std::getenv("RADIANCE_PP_SEG_SET");
    if (segset_env && *segset_env) {
        const int v = std::atoi(segset_env);
        if (!segset_ok(v) || (v != 0 && seg_in.empty()))
            RAD_WARN("follower: RADIANCE_PP_SEG_SET=%s is not taken here: %s (tp8 loop D33)", segset_env,
                     segset_ok(v) ? "no segment boundary declared on this node (RADIANCE_PP_SEG_LAYERS)"
                                  : "it takes n + 64 x s, n 0 .. 31, s 0 .. 7");
        else
            segset_start = v;
    }
    auto segset_now = [&]() -> int {
        int v = fn4::get(fn4::kPpSegSet);
        if (v < 0) v = segset_start;   /* never requested: the start's */
        return segset_ok(v) ? v : 0;
    };
    /* the boundaries a pass of this set splits at (ascending), with the reserve's boundary res_l (0: none) */
    auto segset_pick = [&](int sv, int res_l) -> std::vector<int32_t> {
        if (sv == 0) return seg_in;
        const int every = std::max(1, sv >> 6), first = sv & 63;
        std::vector<int32_t> out;
        int taken = 0;
        for (size_t j = 0; j < seg_in.size(); ++j) {
            bool pick = (int)(j + 1) % every == 0 && (first == 0 || taken < first);
            if (pick) ++taken;
            if (pick || seg_in[j] == res_l) out.push_back(seg_in[j]);
        }
        return out;
    };
    /* the D33 line: split passes and their segments by set, and the other passes run at gaps past the old cap's last boundary
     * (seg_in[14]: the gaps inside the former end lump) -- its target quantity on the follower */
    const int32_t lump_at = seg_in.size() > 15 ? seg_in[14] : INT32_MAX;
    const bool d33_line = (std::getenv("RADIANCE_PP_SEG_MAX") && *std::getenv("RADIANCE_PP_SEG_MAX")) || (segset_env && *segset_env);
    int64_t d33_passes = 0, d33_parts = 0, d33_gaps_past = 0, d33_between_past = 0;
    int d33_last = -1;
    if (d33_line && !seg_in.empty()) {
        std::string b;
        for (int32_t x : seg_in) b += fmt("%s%d", b.empty() ? "" : ", ", (int)x);
        RAD_INFO("follower: segment set (tp8 loop D33): %zu declared boundaries inside W(%d,%d) (%s); fn4 pp_seg_set starts at %d "
                 "(RADIANCE_PP_SEG_SET; 0 = all, n + 64 x s = every s-th then the first n)%s", seg_in.size(), cfg_.pp_cut_h,
                 cfg_.pp_cut_t, b.c_str(), segset_start,
                 lump_at != INT32_MAX ? fmt("; gaps past layer %d (the old 15-boundary cap's last) counted apart", (int)lump_at).c_str()
                                      : "");
    }
    auto d33_print = [&](const char* when) {
        if (!d33_line || seg_in.empty()) return;
        RAD_INFO("follower: segment set %s (pp_seg_set now %d, tp8 loop D33): %lld split passes in %lld segments (%.2f a pass); "
                 "%lld gaps past layer %d with %lld other passes run in them", when, segset_now(), (long long)d33_passes,
                 (long long)d33_parts, d33_passes > 0 ? (double)d33_parts / (double)d33_passes : 0.0, (long long)d33_gaps_past,
                 lump_at != INT32_MAX ? (int)lump_at : -1, (long long)d33_between_past);
    };
    int64_t res_n = 0, res_frame = 0, res_time = 0;
    uint64_t res_ns = 0;
    auto reserve_line = [&](const char* when) {
        if (fol_reserve > 0 || res_n > 0)
            RAD_INFO("follower: body reserve %s (RADIANCE_PP_FOL_RESERVE=%d, now %d, tp8 loop IDEAS #9): %lld BG bodies held their "
                     "layers L..%d, %lld released by another BG lane's frame, %lld by the %llu-ms limit; held %.1f ms in all (%.1f "
                     "a body)", when, fol_reserve, reserve_now(), (long long)res_n, cfg_.pp_cut_t - 1, (long long)res_frame,
                     (long long)res_time, (unsigned long long)(fol_reserve_ns / 1000000ull), (double)res_ns / 1e6,
                     res_n > 0 ? (double)res_ns / 1e6 / (double)res_n : 0.0);
    };
    int64_t fo_waited = 0, fo_jumped = 0, fo_passed = 0;
    /* D15: every run frame carries the leader's whole switch-request table; a frame whose table differs from the one that
     * arrived before it changes a switch here (fn4::follow) and so keeps its place, and nothing passes it (its lane, step) */
    std::vector<uint8_t> fo_req;
    std::set<int64_t> fo_chg;
    std::map<int64_t, uint64_t> fo_arr;   /* bit 4: a run frame's arrival time (its lane, step) */
    int64_t fo_graced = 0;
    auto fo_key = [](const StepHdr& hh) -> int64_t { return ((int64_t)hh.lane << 32) | (uint32_t)hh.step; };
    auto note_arrival = [&](const net::Msg& mm) {
        if (!fol_order || mm.h.kind != net::kStep) return;
        StepHdr hh;
        if (mm.body.size() < sizeof hh) return;
        std::memcpy(&hh, mm.body.data(), sizeof hh);
        if (!hh.run) return;
        const size_t nb = sizeof hh.fn4_n + sizeof hh.fn4_v;
        std::vector<uint8_t> t(nb);
        std::memcpy(t.data(), hh.fn4_n, sizeof hh.fn4_n);
        std::memcpy(t.data() + sizeof hh.fn4_n, hh.fn4_v, sizeof hh.fn4_v);
        if (fo_req.empty() || fo_req != t) fo_chg.insert(fo_key(hh));
        fo_req.swap(t);
        if (fol_order & 4) fo_arr[fo_key(hh)] = rad_mono_ns();
    };
    auto is_chg = [&](const StepHdr& hh) -> bool { return hh.run && fo_chg.count(fo_key(hh)) != 0; };
    auto x1_landed = [&](const net::Msg& mm) -> bool {
        StepHdr hh;
        if (mm.body.size() < sizeof hh) return true;
        std::memcpy(&hh, mm.body.data(), sizeof hh);
        if (!((hh.flags & 2) || (fol_order & 1)) || !hh.run || hh.lane < 0 || hh.lane >= n_lanes_ || hh.lane >= RAD_PP_MAX_LANES)
            return true;
        const RadPpRing* x1 = pp_recv_[hh.lane * RAD_PP_N_RINGS + RAD_PP_X1];
        return !x1 || (int64_t)__atomic_load_n(&x1->landed, __ATOMIC_ACQUIRE) > x1_runs[hh.lane];
    };
    std::deque<net::Msg> held;   /* frames that wait for the split pass to end (and every frame behind them) */
    /* A frame other than a step's: the leader's switch ask, its KV-copy check. */
    auto other_frame = [&](net::Msg& m) -> int {
        if (m.h.kind == net::kFn4Ask) {
            /* The leader's ask (fn4::set_peer_check): this node's own verdict, with its reason. */
            const std::string body(m.body.begin(), m.body.end());
            const size_t tab = body.find('\t');
            std::string err;
            int verdict = -1;
            if (tab == std::string::npos) err = "a malformed ask";
            else verdict = fn4::check_node(body.substr(0, tab), std::atoi(body.c_str() + tab + 1), &err);
            if (pp_link_->send(net::kFn4Ans, m.h.seq, (uint64_t)(int64_t)verdict, err.data(), (int64_t)err.size()) < 0)
                return fail_loop(RAD_E_IO);
            return RAD_OK;
        }
        if (m.h.kind == net::kKvCheck) {   /* tp8 S6 (debug): the leader's CRCs after its last trunk step */
            if (pp_kv_check_recv(m.body, m.h.seq) < 0) return fail_loop(RAD_E_STATE);
            return RAD_OK;
        }
        pp_link_->fail(fmt("the follower got a %s frame while serving", net::frame_name(m.h.kind)));
        return fail_loop(RAD_E_IO);
    };
    /* A step frame: replay its build into its lane's builder and, with `run`, issue its pass -- whole, or (`may_split`, a
     * pass of >= seg_rows rows, boundaries declared) in segments with other lanes' passes between them. */
    std::function<int(net::Msg&, bool)> step_frame = [&](net::Msg& m, bool may_split) -> int {
        int st = RAD_OK;
        StepHdr h;
        BatchBuilder::Wire w;
        if (m.body.size() < sizeof h) { pp_link_->fail("a step frame shorter than its header"); return fail_loop(RAD_E_IO); }
        std::memcpy(&h, m.body.data(), sizeof h);
        const size_t psk_bytes = (size_t)std::max(0, h.psk_n) * 3 * sizeof(int32_t);
        const size_t clr_bytes = (size_t)std::max(0, h.n_clears) * 2 * sizeof(int32_t);
        const size_t ck_bytes = (size_t)std::max(0, h.n_ck) * sizeof(int32_t);
        if (h.magic != kStepMagic || h.psk_n < 0 || h.n_fn4 < 0 || h.n_fn4 > kMaxFn4 || h.n_clears < 0 ||
            h.n_clears > kMaxClears || h.n_ck < 0 ||
            m.body.size() < sizeof h + psk_bytes + clr_bytes + ck_bytes + sizeof w) {
            pp_link_->fail(fmt("step frame %llu: a damaged header", (unsigned long long)m.h.seq));
            return fail_loop(RAD_E_IO);
        }
        const uint8_t* q = m.body.data() + sizeof h;
        ahead.resize((size_t)h.psk_n);
        rows.resize((size_t)h.psk_n);
        flags.resize((size_t)h.psk_n);
        if (h.psk_n > 0) {
            std::memcpy(ahead.data(), q, (size_t)h.psk_n * 4); q += (size_t)h.psk_n * 4;
            std::memcpy(rows.data(), q, (size_t)h.psk_n * 4);  q += (size_t)h.psk_n * 4;
            std::memcpy(flags.data(), q, (size_t)h.psk_n * 4); q += (size_t)h.psk_n * 4;
        }
        clears.resize((size_t)h.n_clears * 2);
        if (h.n_clears > 0) { std::memcpy(clears.data(), q, clr_bytes); q += clr_bytes; }
        ckw.resize((size_t)h.n_ck);
        if (h.n_ck > 0) { std::memcpy(ckw.data(), q, ck_bytes); q += ck_bytes; }
        std::memcpy(&w, q, sizeof w);
        q += sizeof w;
        const int64_t span_bytes = (int64_t)(m.body.size() - (size_t)(q - m.body.data()));
        if (h.lane < 0 || h.lane >= n_lanes_) {
            pp_link_->fail(fmt("step frame %llu names lane %d of %d", (unsigned long long)m.h.seq, h.lane, n_lanes_));
            return fail_loop(RAD_E_IO);
        }
        /* tp8 S5: the lane's own builder and slab; its rings for the pass */
        Scheduler& ls = sched_of(h.lane);
        const RadBatch* b = ls.wire_replay(w, (const char*)q, span_bytes, h.run != 0);
        if (!b) {
            pp_link_->fail(fmt("the follower could not replay step %d", h.step));
            return fail_loop(RAD_E_STATE);
        }
        ++builds;
        if (!h.run) return RAD_OK;
        if (!fo_chg.empty()) fo_chg.erase(fo_key(h));   /* tp8 loop D15: a switch change is applied below */
        if (!fo_arr.empty()) fo_arr.erase(fo_key(h));
        if (h.lane < RAD_PP_MAX_LANES) ++x1_runs[h.lane];   /* tp8 dev D10 round 2: this pass takes its lane's next X1 */
        /* Review F4: the state slots the leader zeroed before this step (a fresh sequence's), zeroed
         * here on this node's own groups, as Engine::clear_fresh_states does on the leader. */
        for (int k = 0; k < h.n_clears; ++k) {
            const int32_t grp = clears[(size_t)k * 2], slot = clears[(size_t)k * 2 + 1];
            for (auto& rk : ranks_) {
                const KVGroupPlan* gp = rk->kv.plan(grp);
                if (!gp || !gp->stateful()) continue;
                const int64_t bytes = gp->layer_bytes_per_state();
                for (int64_t l = 0; l < gp->n_layers; ++l)
                    if (void* p = rk->kv.state_ptr(grp, (int32_t)l, slot))
                        rad_memset_async(p, 0, bytes, rk->ctx.stream());
            }
        }
        /* Review F2 / F3: the switch requests the leader applied since the last step -- only those
         * that act on this node's passes or placement; a value this node refuses (a lever it does not
         * declare: the drafter's, an Engram lever on a node without Engram layers) is skipped and named,
         * never a failure. */
        if (h.n_fn4 > 0) {
            std::vector<std::string> skipped;
            const int asked = fn4::follow(h.fn4_n, h.fn4_v, std::min(h.n_fn4, (int)fn4::kSwitches), pp_fn4_taken_,
                                          &skipped);
            for (const std::string& sk : skipped)
                RAD_WARN("follower: the leader's switch request %s -- not taken on this node, which keeps its "
                         "own value; the pipeline goes on", sk.c_str());
            if (asked > 0) {
                int sw[fn4::kSwitches];
                if (fn4::apply_pending(h.step, sw)) fn4_apply_changes(sw);
            }
            if (asked > 0 || !skipped.empty()) pp_send_fn4_state();
        }
        /* The leader's PSK plan for this pass, handed to the plugin as psk_release does. */
        int32_t end = INT32_MAX;
        if (h.psk_valid) {
            const int l = fn4::psk_plan(h.step, h.psk_dec, h.psk_n, ahead.data(), rows.data(), flags.data());
            if (l >= 0) end = l;
        }
        for (auto& r : ranks_) r->stager.set_layer_end(end);
        /* tp8 S6: the prefix hits' restores into this node's state groups and window pages, before the
         * pass, as the leader's step loop does on its own */
        if (h.n_ck > 0 && (st = pp_ck_apply(ckw.data(), h.n_ck, false)) < 0) {
            pp_link_->fail(fmt("the follower could not restore step %d's checkpoints: %s", h.step, rad_strerror(st)));
            return fail_loop(st);
        }
        /* the saves behind the pass need the frame's words after a pass between segments reused the scratch */
        const std::vector<int32_t> ck_keep = h.n_ck > 0 ? ckw : std::vector<int32_t>{};
        /* tp8 loop P28: a band pass (frame flags bit 2) starts at h + k on this node */
        const int band = (h.flags & 4) ? pp_band_k() : 0;
        if ((h.flags & 4) && band <= 0) {
            pp_link_->fail(fmt("step %d: the leader ran a band pass, but this node's plugin declared no band "
                               "(RADIANCE_DS41_PP_BAND on both nodes)", h.step));
            return fail_loop(RAD_E_STATE);
        }
        const int pass_h = cfg_.pp_cut_h + band;
        /* tp8 loop D39: a tail band pass (frame flags bit 3) ends at t - j on this node (the plugin then stores the band
         * layers' own KV / window rows from what it holds) */
        const int tband = (h.flags & 8) ? pp_tail_band_k() : 0;
        if ((h.flags & 8) && (tband <= 0 || band > 0)) {
            pp_link_->fail(fmt("step %d: the leader ran a tail band pass, but %s", h.step,
                               band > 0 ? "the frame is a (head) band pass's too" : "this node's plugin declared no tail band"));
            return fail_loop(RAD_E_STATE);
        }
        const int pass_t = cfg_.pp_cut_t - tband;
        /* tp8 loop P41: a reverse pass (frame flags bit 5) starts at the band layer l = h - 1 on this node (never beside a band
         * / tail band word: the plugin declares the reverse band only without them) */
        const int rband = (h.flags & 32) ? pp_rband_l() : 0;
        if ((h.flags & 32) && (rband <= 0 || rband != cfg_.pp_cut_h - 1 || band > 0 || tband > 0)) {
            pp_link_->fail(fmt("step %d: the leader ran a reverse (head band) pass, but %s", h.step,
                               rband <= 0 ? "this node's plugin declared no reverse band (RADIANCE_DS41_PP_RBAND=1 on both nodes)"
                               : rband != cfg_.pp_cut_h - 1 ? "its declared layer is not h - 1"
                                                           : "the frame is a band / tail band pass's too"));
            return fail_loop(RAD_E_STATE);
        }
        const int pass_h0 = rband > 0 ? rband : pass_h;
        /* tp8 loop P41 step 5: X1 in the keys form (frame flags bit 6; only beside a declared reverse band, never on a reverse pass) */
        const int rkeys = (h.flags & 64) ? 1 : 0;
        if ((h.flags & 64) && (pp_rband_l() <= 0 || rband > 0)) {
            pp_link_->fail(fmt("step %d: the leader asked for X1's keys form, but %s", h.step,
                               rband > 0 ? "the frame is a reverse pass's too" : "this node's plugin declared no reverse band"));
            return fail_loop(RAD_E_STATE);
        }
        /* tp8 dev M4: the pass's parts -- [h, t) whole, or its segments at the declared boundaries */
        std::vector<std::pair<int, int>> parts;
        const bool split = may_split && !segs.empty() && b->n_tok >= seg_rows &&
                           (seg_prefill || b->phase == RAD_PHASE_MIXED || b->n_tok_decode > 0 || (seg_bg >= 0 && h.lane >= seg_bg) ||
                            (h.flags & 16));   /* tp8 MERGE bit 32: a relay lane's BG step, segmented as a BG lane's (header bit 16 here) */
        if (split) {
            /* tp8 loop D33: the set's boundaries (0 = every declared one: the program before D33) */
            const int sv = segset_now();
            const std::vector<int32_t> pick =
                segset_pick(sv, seg_bg >= 0 && h.lane >= seg_bg && b->n_tok_decode == 0 ? reserve_now() : 0);
            int a = pass_h0;
            for (int32_t x : pick) {
                if (x <= a || x >= pass_t) continue;
                parts.emplace_back(a, x);
                a = x;
            }
            parts.emplace_back(a, pass_t);
            ++d33_passes;
            d33_parts += (int64_t)parts.size();
            if ((d33_line || sv != 0) && sv != d33_last) {
                std::string bs;
                for (size_t k = 1; k < parts.size(); ++k) bs += fmt("%s%d", bs.empty() ? "" : ", ", parts[k].first);
                RAD_INFO("follower: pp_seg_set %d (tp8 loop D33): a split pass of W(%d,%d) now ends its segments at %s (%zu segments)",
                         sv, pass_h, cfg_.pp_cut_t, bs.empty() ? "none" : bs.c_str(), parts.size());
                d33_last = sv;
            }
        } else {
            parts.emplace_back(pass_h0, pass_t);
        }
        const int lane = h.lane, sstep = h.step;
        /* between two segments: the frames that have arrived (no wait on a card past the segment's run; a host test's
         * RADIANCE_PP_SEG_TEST_WAIT_MS) */
        auto between = [&](RadEvent done_ev) -> int {
            const uint64_t t0 = rad_mono_ns();
            /* tp8 loop D15 bit 2: the lanes with a frame held (its lane's later frames stay behind it) and whether a held
             * frame carries a switch request (then every later frame stays behind it); a frame held before this pass's
             * segments that may pass now (a decode frame of a lane with nothing held before it) waits for its X1 below */
            uint32_t held_lanes = 0;
            bool held_fn4 = false;
            bool fo_in_grace = false;   /* bit 4: this gap is held back for a frame's X1 (counted once) */
            auto lane_bit = [](const StepHdr& hh) -> uint32_t {
                return hh.lane >= 0 && hh.lane < 32 ? 1u << hh.lane : ~0u;
            };
            auto note_held = [&](const net::Msg& mm) {
                StepHdr hh;
                if (mm.body.size() < sizeof hh) { held_fn4 = true; return; }
                std::memcpy(&hh, mm.body.data(), sizeof hh);
                held_lanes |= lane_bit(hh);
                if (is_chg(hh)) held_fn4 = true;
            };
            /* bit 2: may a frame that is not a keeper pass the held ones */
            auto may_pass = [&](const StepHdr& hh) -> bool {
                return !(held_lanes & lane_bit(hh)) && !held_fn4 && !is_chg(hh) && hh.n_clears == 0 && hh.n_ck == 0;
            };
            /* D15: with the switch on, a frame that changes a switch keeps its place too (the frames waiting for their
             * X1 that came before it run before it, every later one after it) */
            auto keeps_place = [&](const net::Msg& mm, const StepHdr& hh) -> bool {
                return mm.body.size() < sizeof hh || ((hh.flags & 1) && hh.run) || hh.lane == lane ||
                       (fol_order && is_chg(hh));
            };
            if (fol_order & 2) {
                for (size_t i = 0; i < held.size();) {
                    StepHdr hh;
                    const bool hdr = held[i].body.size() >= sizeof hh;
                    if (hdr) std::memcpy(&hh, held[i].body.data(), sizeof hh);
                    if (hdr && !keeps_place(held[i], hh) && may_pass(hh)) {
                        if (hh.run) ++fo_passed;
                        ia_pend.push_back(std::move(held[i]));
                        held.erase(held.begin() + (std::ptrdiff_t)i);
                        continue;
                    }
                    note_held(held[i]);
                    ++i;
                }
            }
            /* the frames waiting for their X1: D10's in their order; bit 1 -- each once its X1 has landed, a lane's own in
             * their order, none past a waiting switch request */
            auto issue_pend = [&]() -> int {
                uint32_t blocked = 0;
                for (size_t i = 0; i < ia_pend.size();) {
                    StepHdr hp;
                    if (ia_pend[i].body.size() < sizeof hp) break;
                    std::memcpy(&hp, ia_pend[i].body.data(), sizeof hp);
                    const uint32_t lb = lane_bit(hp);
                    if (i > 0 && (!(fol_order & 1) || is_chg(hp))) break;
                    if (!(blocked & lb) && x1_landed(ia_pend[i])) {
                        if (i > 0) ++fo_jumped;
                        net::Msg mp = std::move(ia_pend[i]);
                        ia_pend.erase(ia_pend.begin() + (std::ptrdiff_t)i);
                        if (const int fs = step_frame(mp, false); fs < 0) return fs;
                        if (hp.run) { ++seg_between; ++steps; }
                        continue;
                    }
                    blocked |= lb;
                    if (is_chg(hp) || !(fol_order & 1)) break;
                    ++i;
                }
                return RAD_OK;
            };
            for (;;) {
                /* tp8 dev D10 round 2: the frames issued ahead whose X1 has landed (and the decode frames behind them);
                 * tp8 loop D15 bit 1: every waiting frame whose X1 has landed */
                if (const int ps = issue_pend(); ps < 0) return ps;
                const bool dev_done = !done_ev || rad_event_query(done_ev) != 0;
                const bool waited = (rad_mono_ns() - t0) / 1000000ull >= (uint64_t)seg_wait_ms;
                net::Msg m2;
                const int rs = pp_link_->recv(&m2, 0);
                if (rs == RAD_E_NOTFOUND) {
                    /* tp8 loop D15 bit 4: a waiting frame that arrived less than the grace ago keeps the next segment back */
                    bool grace = false;
                    if ((fol_order & 4) && dev_done && waited && !ia_pend.empty()) {
                        const uint64_t now = rad_mono_ns();
                        for (const net::Msg& mp : ia_pend) {
                            StepHdr hp;
                            if (mp.body.size() < sizeof hp) continue;
                            std::memcpy(&hp, mp.body.data(), sizeof hp);
                            const auto it = fo_arr.find(fo_key(hp));
                            if (hp.run && it != fo_arr.end() && now - it->second < fo_grace_ns) { grace = true; break; }
                        }
                        if (grace && !fo_in_grace) { fo_in_grace = true; ++fo_graced; }
                    }
                    if (dev_done && waited && !grace) { fo_in_grace = false; return RAD_OK; }
                    fn4::nap_us(50);   /* tp8 loop D61: a spin of 50 us under host_spin site 8 */
                    continue;
                }
                if (rs < 0) {
                    if (pp_link_->peer_stopped()) return 1;   /* the leader stopped: the main loop's clean stop */
                    RAD_ERR("follower: the link to the leader is down: %s", pp_link_->why().c_str());
                    return fail_loop(RAD_E_IO);
                }
                if (m2.h.kind != net::kStep) {
                    if (const int os = other_frame(m2); os < 0) return os;
                    continue;
                }
                note_arrival(m2);
                StepHdr h2;
                if (m2.body.size() >= sizeof h2) std::memcpy(&h2, m2.body.data(), sizeof h2);
                /* a step that keeps its place (prompt rows, a lent piece) and every frame behind it wait for the split pass;
                 * a decode step runs its whole pass now; a build without a pass is replayed here (tp8 loop D15 bit 2: a
                 * decode frame passes the held ones when nothing of its lane is held) */
                const bool keeps = keeps_place(m2, h2);
                bool hold = keeps || !held.empty();
                if (hold && !keeps && (fol_order & 2) && may_pass(h2)) {
                    hold = false;
                    if (h2.run) ++fo_passed;
                }
                if (hold) {
                    if (h2.run) ++seg_held;
                    if (fol_order & 2) note_held(m2);
                    held.push_back(std::move(m2));
                    continue;
                }
                bool wait = !x1_landed(m2);   /* tp8 dev D10 round 2: its X1 is not here yet */
                if (fol_order & 1) {
                    /* tp8 loop D15 bit 1: behind a waiting frame of its lane or a waiting switch request too */
                    for (const net::Msg& mp : ia_pend) {
                        StepHdr hp;
                        if (mp.body.size() < sizeof hp) { wait = true; break; }
                        std::memcpy(&hp, mp.body.data(), sizeof hp);
                        if (hp.lane == h2.lane || is_chg(hp)) { wait = true; break; }
                    }
                    if (is_chg(h2) && !ia_pend.empty()) wait = true;
                } else if (!ia_pend.empty()) {
                    wait = true;
                }
                if (wait) {
                    if (h2.flags & 2) { if (ia_pend.empty()) ++ia_deferred; }
                    else if ((fol_order & 1) && h2.run) ++fo_waited;
                    ia_pend.push_back(std::move(m2));
                    continue;
                }
                if (const int fs = step_frame(m2, false); fs < 0) return fs;
                if (h2.run) { ++seg_between; ++steps; }
            }
        };
        /* tp8 loop P22: a lone prefill pass is timed (its parts together); any other run pass ends the open prompt.
         * tp8 loop P22b: a split one (a BG newcomer's body in segments, other lanes' passes between them) by its segments,
         * on its own record -- the passes between neither end nor join the open prompt */
        const bool pfl = pff && b->phase == RAD_PHASE_PREFILL && b->n_seq == 1;
        /* (a split prefill pass of several prompts too: a batched BG step, RADIANCE_PP_BG_ROOM / pp_bg_room) */
        const bool pfs = pff && split && b->phase == RAD_PHASE_PREFILL && b->n_seq >= 1 && !pff->seg_open;
        const int64_t pfs_between0 = seg_between;
        if (pff && !pfl) pff->close();
        if (pfs)
            pff->seg_begin(b->n_tok, b->max_ctx_len == 0, lane, h.psk_valid && h.psk_n > 0 ? ahead[0] : INT32_MIN,
                           pp_recv_[lane * RAD_PP_N_RINGS + RAD_PP_X1], b->n_seq);
        else if (pfl)
            pff->begin(b->n_tok, b->max_ctx_len == 0, lane, h.psk_valid && h.psk_n > 0 ? ahead[0] : INT32_MIN);
        /* tp8 loop D45 (RADIANCE_PP_NODE_IDLE=2): a BG lane's frame in pass on this node -- its parts and every pass run between
         * them count to the stretch with a BG step in flight, until this frame's handling returns */
        struct NiBg {
            PpNodeIdle* n;
            NiBg(PpNodeIdle* p, bool on) : n(p && on && p->split ? p : nullptr) { if (n) ++n->bg_depth; }
            ~NiBg() { if (n) --n->bg_depth; }
        } ni_bg(pp_ni_, seg_bg >= 0 && lane >= seg_bg);
        for (size_t k = 0; k < parts.size(); ++k) {
            if (k > 0) {
                /* the segment k - 1 - ahead must have run before segment k goes out (ahead 1: one queued behind the
                 * running one), the other lanes' frames issued meanwhile */
                const int64_t back = (int64_t)k - 1 - seg_ahead;
                const int64_t between0 = seg_between;   /* tp8 loop D33: the passes run at this gap */
                if (const int bs = between(back >= 0 ? seg_ev[back & 1] : nullptr); bs != 0) return bs;   /* < 0 failed, 1 stop */
                /* tp8 loop IDEAS #9: a BG body's reserve -- its parts from RADIANCE_PP_FOL_RESERVE on wait for another BG lane's
                 * frame (held behind this pass: it keeps its place) or the limit; every other frame runs between meanwhile */
                if (const int res_l = split && seg_bg >= 0 && lane >= seg_bg && b->n_tok_decode == 0 ? reserve_now() : 0;
                    res_l > 0 && parts[k].first == res_l) {
                    auto other_bg = [&]() -> bool {
                        for (const net::Msg& mm : held) {
                            StepHdr hh;
                            if (mm.body.size() < sizeof hh) continue;
                            std::memcpy(&hh, mm.body.data(), sizeof hh);
                            if (hh.run && (hh.flags & 1) && hh.lane >= seg_bg && hh.lane < n_lanes_ && hh.lane != lane) return true;
                        }
                        return false;
                    };
                    const uint64_t r0 = rad_mono_ns();
                    int how = 0;
                    for (;;) {
                        if (other_bg()) { how = 1; break; }
                        if (rad_mono_ns() - r0 >= fol_reserve_ns) { how = 2; break; }
                        if (const int bs = between(nullptr); bs != 0) return bs;   /* < 0 failed, 1 stop */
                        if (other_bg()) { how = 1; break; }
                        fn4::nap_us(100);   /* tp8 loop D61: a spin of 100 us under host_spin site 8 */
                    }
                    ++res_n;
                    if (how == 1) ++res_frame;
                    else ++res_time;
                    res_ns += rad_mono_ns() - r0;
                }
                if (parts[k].first > lump_at) {   /* tp8 loop D33: a gap inside the former end lump (a reserve's hold too) */
                    ++d33_gaps_past;
                    d33_between_past += seg_between - between0;
                }
                /* the pass between may have changed what this pass's issue reads */
                for (auto& r : ranks_) r->stager.set_layer_end(end);
            }
            issue_sched_.store(&ls, std::memory_order_release);
            if (split) pp_set_pass(lane, RAD_PP_PART_SEG, parts[k].first, parts[k].second, band, 0, tband, rband, rkeys);
            else pp_set_pass(lane, RAD_PP_PART_WHOLE, 0, 0, band, 0, tband, rband, rkeys);
            issue_what_.store(kIssueFollower, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8: the ranks are parked, rank 0's stream is this thread's */
            if (pfs) pff->seg_a();            /* tp8 loop P22b */
            barrier_->arrive_and_wait();      /* release the ranks */
            barrier_->arrive_and_wait();      /* and wait for them */
            if (pfs) pff->seg_b();
            if (pp_ni_) pp_ni_->end();
            if (split) {
                ++seg_parts;
                if (rad_event_record(seg_ev[k & 1], ranks_[0]->ctx.stream()) < 0) {
                    RAD_ERR("follower: could not record step %d's segment event", sstep);
                    return fail_loop(RAD_E_DEVICE);
                }
            }
            st = step_status_.load(std::memory_order_acquire);
            if (st < 0) break;
        }
        pp_set_pass(lane, RAD_PP_PART_WHOLE);
        if (split) ++seg_passes;
        if (pfs) pff->seg_end(seg_between - pfs_between0);
        if (pfl && !pfs) pff->end();
        else if (pff) pff->flush(false);
        /* tp8 dev D10 round 2: frames still waiting for their X1 go first after the split pass (they came before any held) */
        if (split && !ia_pend.empty()) {
            held.insert(held.begin(), std::make_move_iterator(ia_pend.begin()), std::make_move_iterator(ia_pend.end()));
            ia_pend.clear();
        }
        /* tp8 S5: busy from X1's arrival (rank 0 took it) to X2's publish (the host backend notes both) */
        if (pp_stamps_) {
            const RadPpRing* x1 = pp_recv_[lane * RAD_PP_N_RINGS + RAD_PP_X1];
            const RadPpRing* x2 = pp_send_[lane * RAD_PP_N_RINGS + RAD_PP_X2];
            if (x1 && x2 && x1->taken_ns && x2->ready_ns) pp_stamp("mid", lane, sstep, x1->taken_ns, x2->ready_ns);
        }
        st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("follower step %d failed: %s -- the pipeline cannot continue", sstep, rad_strerror(st));
            pp_link_->fail(fmt("the follower's step %d failed: %s", sstep, rad_strerror(st)));
            return fail_loop(st);
        }
        /* tp8 S6: the step's checkpoint saves, behind its pass on every rank's stream (save_checkpoints) */
        if (!ck_keep.empty() && (st = pp_ck_apply(ck_keep.data(), (int32_t)ck_keep.size(), true)) < 0) {
            pp_link_->fail(fmt("the follower could not save step %d's checkpoints: %s", sstep, rad_strerror(st)));
            return fail_loop(st);
        }
        /* tp8 dev D9 (loop FINDINGS F15 answer 02:24): --profile-ops prints here during the run too, at the leader's step numbers
         * (every RADIANCE_PROFILE_EVERY, default 32: the same steps as the leader's lane loop), so the follower's table can be cut
         * to decode-only intervals as the leader's; it printed only at exit. The ranks are parked: the counters are host-side. */
        static const int prof_every = [] {
            const char* v = std::getenv("RADIANCE_PROFILE_EVERY");
            const int k = (v && *v) ? (int)std::strtol(v, nullptr, 10) : 32;
            return k > 0 ? k : 32;
        }();
        if (cfg_.profile_ops && sstep > 0 && (sstep % prof_every) == 0) {
            RAD_INFO("follower: --profile-ops tables after step %d (lane %d; RADIANCE_PROFILE_EVERY=%d, tp8 dev D9)", sstep, lane, prof_every);
            for (auto& rk : ranks_) rk->ctx.dump_profile();
        }
        return RAD_OK;
    };
    while (!stopping_.load(std::memory_order_acquire)) {
        watch.progress();
        if (pp_stop_requested()) {
            RAD_INFO("follower: stopping on a signal after %lld steps: the leader is told (kBye) and goes down "
                     "with it", (long long)steps);
            if (pp_ni_) pp_ni_->stop();
            if (pff) pff->stop();   /* tp8 loop P22 / P22b */
            loop_status_.store(RAD_OK, std::memory_order_release);
            stopping_.store(true, std::memory_order_release);
            if (barrier_) barrier_->abort();
            pp_close("the follower was stopped (SIGTERM / SIGINT)");
            return RAD_OK;
        }
        net::Msg m;
        if (!held.empty()) {   /* tp8 dev M4: the frames that waited for a split pass, in their order */
            m = std::move(held.front());
            held.pop_front();
        } else {
            int st = pp_link_->recv(&m, 200);
            if (st == RAD_E_NOTFOUND) continue;
            if (st < 0) {
                if (pp_link_->peer_stopped()) {
                    RAD_INFO("follower: the leader stopped (%s) after %lld steps; this node stops too",
                             pp_link_->why().c_str(), (long long)steps);
                    if (pp_ni_) pp_ni_->stop();
                    if (pff) pff->stop();   /* tp8 loop P22 / P22b */
                    loop_status_.store(RAD_OK, std::memory_order_release);
                    stopping_.store(true, std::memory_order_release);
                    if (barrier_) barrier_->abort();
                    if (!segs.empty())
                        RAD_INFO("follower: %lld passes ran in %lld segments, %lld other passes between their segments, "
                                 "%lld held in their place (tp8 dev M4)", (long long)seg_passes, (long long)seg_parts,
                                 (long long)seg_between, (long long)seg_held);
                    if (ia_deferred > 0)
                        RAD_INFO("follower: %lld frames issued ahead waited between a split pass's segments for their X1 "
                                 "(tp8 dev D10 round 2)", (long long)ia_deferred);
                    reserve_line("over the run");
                    d33_print("over the run");
                    if (fol_order && !segs.empty())
                        RAD_INFO("follower: frame order over the run (RADIANCE_PP_FOL_ORDER=%d, tp8 loop D15): %lld decode "
                                 "frames waited between segments for their X1, %lld went before an older waiting frame, %lld "
                                 "passed a held frame, %lld segment gaps held back for an X1 within the grace", fol_order,
                                 (long long)fo_waited, (long long)fo_jumped, (long long)fo_passed, (long long)fo_graced);
                    return RAD_OK;
                }
                RAD_ERR("follower: the link to the leader is down: %s", pp_link_->why().c_str());
                return fail_loop(RAD_E_IO);
            }
            note_arrival(m);   /* tp8 loop D15: a frame from the link (not one that waited) */
        }
        if (m.h.kind != net::kStep) {
            if (const int os = other_frame(m); os < 0) return os;
            continue;
        }
        StepHdr h0;
        const bool run = m.body.size() >= sizeof h0 && (std::memcpy(&h0, m.body.data(), sizeof h0), h0.run != 0);
        const int fs = step_frame(m, true);
        if (fs < 0) return fs;
        if (fs == 1) {   /* tp8 dev M4: the leader stopped while a split pass waited between its segments */
            if (pp_ni_) pp_ni_->stop();
            if (pff) pff->stop();   /* tp8 loop P22 / P22b */
            RAD_INFO("follower: the leader stopped (%s) after %lld steps, during a pass's segments; this node stops too",
                     pp_link_->why().c_str(), (long long)steps);
            reserve_line("over the run");
            loop_status_.store(RAD_OK, std::memory_order_release);
            stopping_.store(true, std::memory_order_release);
            if (barrier_) barrier_->abort();
            return RAD_OK;
        }
        if (!run) continue;
        /* The host backend runs every launch on this thread's rank; a card's stream is drained by the
         * next step's waits. Nothing else is read back here: the follower samples nothing. */
        ++steps;
        if (steps == 1 || steps % 1024 == 0) {
            RAD_INFO("follower: %lld steps run, %lld builds replayed", (long long)steps, (long long)builds);
            if (const std::string sl = fn4::spin_line(); !sl.empty())   /* tp8 loop D61 */
                RAD_INFO("follower: %s, since its last change (tp8 loop D61)", sl.c_str());
            if (const std::string sl = fn4::slack_line(); !sl.empty())   /* tp8 loop D63 */
                RAD_INFO("follower: %s, since its last change (tp8 loop D63)", sl.c_str());
        }
        if (pp_ni_) pp_ni_->minute();
        if (pp_load_ms_ > 0 && pp_ni_) pp_link_->set_load(pp_ni_->recent_pm((uint64_t)pp_bound_ms_ * 1000000ull));   /* tp8 loop D51 */
        if (!segs.empty() && seg_passes > 0 && (seg_passes == 1 || seg_passes % 256 == 0) && steps % 64 == 0)
            RAD_INFO("follower: %lld passes ran in %lld segments, %lld other passes between their segments, %lld held "
                     "in their place (tp8 dev M4)", (long long)seg_passes, (long long)seg_parts, (long long)seg_between,
                     (long long)seg_held);
        /* tp8 loop D15: the frame-order counts a minute (a cell's window = two lines' difference) */
        if (fol_order && !segs.empty()) {
            static uint64_t fo_line_ns = 0;
            const uint64_t now = rad_mono_ns();
            if (fo_line_ns == 0) fo_line_ns = now;
            if (now - fo_line_ns >= 60ull * 1000000000ull) {
                fo_line_ns = now;
                RAD_INFO("follower: frame order so far (RADIANCE_PP_FOL_ORDER=%d, tp8 loop D15): %lld decode frames waited "
                         "between segments for their X1, %lld went before an older waiting frame, %lld passed a held frame, "
                         "%lld segment gaps held back for an X1 within the grace; %lld passes in %lld segments, %lld other "
                         "passes between them, %lld held", fol_order, (long long)fo_waited, (long long)fo_jumped,
                         (long long)fo_passed, (long long)fo_graced, (long long)seg_passes,
                         (long long)seg_parts, (long long)seg_between, (long long)seg_held);
            }
        }
        /* tp8 loop D33: the segment set's counts a minute (RADIANCE_PP_BGRATE_LINE_S, default 60 s; a cell = two lines' difference) */
        if (d33_line && !seg_in.empty()) {
            static uint64_t d33_line_ns = 0;
            static const uint64_t d33_period_ns = [] {
                const char* v = std::getenv("RADIANCE_PP_BGRATE_LINE_S");
                const long long s = v && *v ? std::atoll(v) : 60;
                return (uint64_t)std::max(1LL, s) * 1000000000ull;
            }();
            const uint64_t now = rad_mono_ns();
            if (d33_line_ns == 0) d33_line_ns = now;
            if (now - d33_line_ns >= d33_period_ns) {
                d33_line_ns = now;
                d33_print("so far");
            }
        }
        /* tp8 loop IDEAS #9: the reserve's counts a minute */
        if (fol_reserve > 0 || res_n > 0) {
            static uint64_t res_line_ns = 0;
            const uint64_t now = rad_mono_ns();
            if (res_line_ns == 0) res_line_ns = now;
            if (now - res_line_ns >= 60ull * 1000000000ull) {
                res_line_ns = now;
                reserve_line("so far");
            }
        }
    }
    for (RadEvent& e : seg_ev) if (e) rad_event_destroy(e);
    return RAD_OK;
}

}  /* namespace rad */
