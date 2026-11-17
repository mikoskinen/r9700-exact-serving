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

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
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
    for (int i = 0; i < RAD_PP_N_RINGS; ++i) out->ring[i] = g.ring[lane * RAD_PP_N_RINGS + i].load();
    return 0;
}

namespace rad {
/* The part of the leader's pass being issued (Ctx::pass_key keys recorded passes by it: a lane step's
 * head and tail share their batch). RAD_PP_PART_WHOLE on one machine and on one lane. */
int pp_pass_part() { return pp_g().part.load(std::memory_order_relaxed); }
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
    };
    return v;
}

/* RADIANCE_* settings that are each node's own. */
bool node_own_env(const std::string& k) {
    static const char* own[] = { "RADIANCE_PP_", "RADIANCE_HOST_DEVICES", "RADIANCE_HOST_VRAM_MIB",
                                 "RADIANCE_HOME", "RADIANCE_IMAGE_ID", "RADIANCE_DUMP_", "RADIANCE_LOG",
                                 "RADIANCE_PROFILE" };
    for (const char* o : own)
        if (k.rfind(o, 0) == 0) return true;
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
        if (!cfg_.expert_stats.empty()) {
            RAD_ERR("two-node pipeline (tp8 S5): --expert-stats counts a lane step's two leader passes as two steps; "
                    "start it with --pp-lanes 1");
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
    o.no_x1 = fe_on;
    if (fe && !std::strcmp(fe, "1"))
        RAD_INFO("two-node pipeline: RADIANCE_PP_FOLLOWER_EMBED=1 %s", fe_on
                     ? "-- the follower looks the step's embedding up itself; the leader's head part issues no lookup "
                       "and no X1 (tp8 dev 1a)"
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
            (sends ? pp_send_ : pp_recv_)[ix] = r;
        }
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

void Engine::pp_set_pass(int lane, int part) {
    PpGlobal& g = pp_g();
    g.lane.store(lane, std::memory_order_relaxed);
    g.part.store(part, std::memory_order_relaxed);
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

void Engine::pp_close(const char* why) {
    if (pp_stamps_) { std::fclose(pp_stamps_); pp_stamps_ = nullptr; }
    if (!pp_link_ || pp_closed_) return;
    pp_closed_ = true;
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
                /* lever TI: the CONV window of a snapshot inside the chunk, from this node's mapped ids */
                if (st >= 0 && save && first >= 0 && rk->kv.has_conv_state()) {
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

int Engine::follower_loop() {
    if (rad_dev_set(ranks_[0]->device) < 0) RAD_ERR("follower loop: could not bind rank 0's device");
    rad::StallWatch watch("follower", 20000);
    int64_t steps = 0, builds = 0;
    std::vector<int32_t> ahead, rows, flags, clears, ckw;
    pp_send_fn4_state();
    while (!stopping_.load(std::memory_order_acquire)) {
        watch.progress();
        if (pp_stop_requested()) {
            RAD_INFO("follower: stopping on a signal after %lld steps: the leader is told (kBye) and goes down "
                     "with it", (long long)steps);
            loop_status_.store(RAD_OK, std::memory_order_release);
            stopping_.store(true, std::memory_order_release);
            if (barrier_) barrier_->abort();
            pp_close("the follower was stopped (SIGTERM / SIGINT)");
            return RAD_OK;
        }
        net::Msg m;
        int st = pp_link_->recv(&m, 200);
        if (st == RAD_E_NOTFOUND) continue;
        if (st < 0) {
            if (pp_link_->peer_stopped()) {
                RAD_INFO("follower: the leader stopped (%s) after %lld steps; this node stops too",
                         pp_link_->why().c_str(), (long long)steps);
                loop_status_.store(RAD_OK, std::memory_order_release);
                stopping_.store(true, std::memory_order_release);
                if (barrier_) barrier_->abort();
                return RAD_OK;
            }
            RAD_ERR("follower: the link to the leader is down: %s", pp_link_->why().c_str());
            return fail_loop(RAD_E_IO);
        }
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
            continue;
        }
        if (m.h.kind == net::kKvCheck) {   /* tp8 S6 (debug): the leader's CRCs after its last trunk step */
            if (pp_kv_check_recv(m.body, m.h.seq) < 0) return fail_loop(RAD_E_STATE);
            continue;
        }
        if (m.h.kind != net::kStep) {
            pp_link_->fail(fmt("the follower got a %s frame while serving", net::frame_name(m.h.kind)));
            return fail_loop(RAD_E_IO);
        }
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
        if (!h.run) continue;
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
            for (const std::string& s : skipped)
                RAD_WARN("follower: the leader's switch request %s -- not taken on this node, which keeps its "
                         "own value; the pipeline goes on", s.c_str());
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
        issue_sched_.store(&ls, std::memory_order_release);
        pp_set_pass(h.lane, RAD_PP_PART_WHOLE);
        issue_what_.store(kIssueFollower, std::memory_order_release);
        step_batch_.store(b, std::memory_order_release);
        barrier_->arrive_and_wait();      /* release the ranks */
        barrier_->arrive_and_wait();      /* and wait for them */
        /* tp8 S5: busy from X1's arrival (rank 0 took it) to X2's publish (the host backend notes both) */
        if (pp_stamps_) {
            const RadPpRing* x1 = pp_recv_[h.lane * RAD_PP_N_RINGS + RAD_PP_X1];
            const RadPpRing* x2 = pp_send_[h.lane * RAD_PP_N_RINGS + RAD_PP_X2];
            if (x1 && x2 && x1->taken_ns && x2->ready_ns) pp_stamp("mid", h.lane, h.step, x1->taken_ns, x2->ready_ns);
        }
        st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("follower step %d failed: %s -- the pipeline cannot continue", h.step, rad_strerror(st));
            pp_link_->fail(fmt("the follower's step %d failed: %s", h.step, rad_strerror(st)));
            return fail_loop(st);
        }
        /* tp8 S6: the step's checkpoint saves, behind its pass on every rank's stream (save_checkpoints) */
        if (h.n_ck > 0 && (st = pp_ck_apply(ckw.data(), h.n_ck, true)) < 0) {
            pp_link_->fail(fmt("the follower could not save step %d's checkpoints: %s", h.step, rad_strerror(st)));
            return fail_loop(st);
        }
        /* The host backend runs every launch on this thread's rank; a card's stream is drained by the
         * next step's waits. Nothing else is read back here: the follower samples nothing. */
        ++steps;
        if (steps == 1 || steps % 1024 == 0)
            RAD_INFO("follower: %lld steps run, %lld builds replayed", (long long)steps, (long long)builds);
    }
    return RAD_OK;
}

}  /* namespace rad */
