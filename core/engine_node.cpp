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
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

/* ================================================================== the plugin's view (rad_pp.h) */
namespace {
struct PpGlobal {
    std::atomic<int>     nodes{ 1 }, rank{ 0 }, cut_h{ 0 }, cut_t{ 0 }, link{ 0 };
    std::atomic<int64_t> row_bytes[RAD_PP_N_RINGS] = {};
    std::atomic<int>     parts[RAD_PP_N_RINGS] = {};
    std::atomic<RadPpRing*> ring[RAD_PP_N_RINGS] = {};
};
PpGlobal& pp_g() { static PpGlobal g; return g; }
}  /* namespace */

extern "C" int rad_pp_info(RadPpInfo* out) {
    if (!out) return RAD_E_INVAL;
    std::memset(out, 0, sizeof *out);
    PpGlobal& g = pp_g();
    out->nodes = g.nodes.load();
    out->rank = g.rank.load();
    out->cut_h = g.cut_h.load();
    out->cut_t = g.cut_t.load();
    out->link = g.link.load();
    for (int i = 0; i < RAD_PP_N_RINGS; ++i) out->ring[i] = g.ring[i].load();
    return 0;
}

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
constexpr uint32_t kStepMagic = 0x31535050u;   /* "PPS1" */
constexpr int      kMaxFn4 = 256;
struct StepHdr {
    uint32_t magic = kStepMagic;
    uint32_t run = 0;           /* 1: the follower runs the pass; 0: applies the uploads only */
    int32_t  step = 0;
    int32_t  n_fn4 = 0;
    int32_t  psk_valid = 0;
    int32_t  psk_n = 0;
    int64_t  psk_dec = 0;
    int32_t  fn4[kMaxFn4] = {};
};

const char* role(int rank) { return rank == 0 ? "leader" : "follower"; }

/* Options that are each node's own: they may differ between the two. */
const std::vector<std::string>& node_own_options() {
    static const std::vector<std::string> v = {
        "--pp-rank", "--pp-peer", "--host", "--port", "--model", "--radiance-home", "--download-dir",
        "--model-revision", "--gpu-memory-mib", "--gpu-headroom-mib", "--host-pool-mib", "--tp", "--p2p",
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
    /* The prefix cache adopts blocks and restores snapshots on the leader's say; the follower's
     * half of that (its KV tier, its snapshot slots) is S8's. */
    if (cfg_.prefix_cache || cfg_.prefix_cache_mib > 0) {
        cfg_.prefix_cache = false;
        cfg_.prefix_cache_mib = 0;
        decide("--no-prefix-cache", "prefix caching off",
               "a two-node pipeline node (tp8 S4): a prefix hit, its tier copies and snapshot restores "
               "are not yet carried to the follower");
    }
    PpGlobal& g = pp_g();
    g.nodes = 2;
    g.rank = cfg_.pp_rank;
    g.cut_h = cfg_.pp_cut_h;
    g.cut_t = cfg_.pp_cut_t;
    g.link = cfg_.node_link;
    decide("two-node pipeline", fmt("%s of W(%d,%d), link %s", role(cfg_.pp_rank), cfg_.pp_cut_h,
                                    cfg_.pp_cut_t, cfg_.node_link ? "verbs" : "tcp"),
           fmt("the follower holds layers [%d, %d), the leader every other layer, the embedding and "
               "the head; a step crosses twice", cfg_.pp_cut_h, cfg_.pp_cut_t));
    return RAD_OK;
}

int Engine::pp_connect() {
    if (!pp_on() || cfg_.debug_graph) return RAD_OK;
    net::LinkOpts o = net::LinkOpts::from_env();
    std::string err;
    pp_link_ = pp_follower() ? net::NodeLink::listen(std::getenv("RADIANCE_PP_BIND") ? std::getenv("RADIANCE_PP_BIND") : "0.0.0.0",
                                                    cfg_.pp_port, o, &err)
                             : net::NodeLink::connect(cfg_.pp_peer, cfg_.pp_port, o, &err);
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
    f.add("binary", net::file_identity("/proc/self/exe"));
    for (const std::string& p : net::loaded_plugin_identities()) {
        const size_t eq = p.find('=');
        f.add("plugin " + p.substr(0, eq), p.substr(eq + 1));
    }
    const char* img = std::getenv("RADIANCE_IMAGE_ID");
    f.add("image", img && *img ? img : "unset");
    /* The model by its bytes, not its path: the two machines hold their own copies. */
    f.add("model", net::file_identity(cfg_.model, int64_t(64) << 20));
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
            if (kv.rfind("RADIANCE_", 0) != 0) continue;
            const std::string k = kv.substr(0, kv.find('='));
            if (node_own_env(k)) continue;
            env.push_back(kv);
        }
        std::sort(env.begin(), env.end());
        for (const std::string& kv : env) {
            const size_t eq = kv.find('=');
            f.add("env " + kv.substr(0, eq), kv.substr(eq + 1));
        }
    }
    for (const diag::ConfigRow& r : diag::config_rows(cfg_)) f.add("option " + r.flag, r.value);
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

int Engine::pp_serve_setup() {
    if (!pp_on()) return RAD_OK;
    const Program& p0 = ranks_[0]->program;
    if (p0.encoder.modalities != 0 || p0.audio_encoder.modalities != 0) {
        RAD_ERR("two-node pipeline (tp8 S4): the program declares an encoder, whose rows are not "
                "shipped to the follower; serve text only (--mm-max-patches 0 or a text container)");
        return RAD_E_UNSUPPORTED;
    }
    PpGlobal& g = pp_g();
    /* ---- the rings: this node's ring of each crossing, sized for the plugin's rows */
    for (int id = 0; id < RAD_PP_N_RINGS; ++id) {
        const int64_t rb = g.row_bytes[id].load();
        const int parts = std::max(1, g.parts[id].load());
        if (rb <= 0) {
            RAD_ERR("two-node pipeline: the '%s' plugin names no stage crossing X%d (rad_pp_crossing): it "
                    "cannot run as a node of a two-node pipeline", plugins_.arch_plugin.c_str(), id + 1);
            return RAD_E_UNSUPPORTED;
        }
        const int64_t cap = rb * cfg_.max_tok + (int64_t)parts * RAD_PP_PART_ALIGN;
        pp_ring_cap_[id] = cap;
        void* mem = rad_dev_alloc(RAD_PP_RING_HDR + cap, RAD_MEM_HOST_PINNED);
        if (!mem) {
            RAD_ERR("two-node pipeline: no pinned host memory for crossing X%d's ring (%s)", id + 1,
                    humanb(RAD_PP_RING_HDR + cap).c_str());
            return RAD_E_NOMEM;
        }
        RadPpRing* r = static_cast<RadPpRing*>(mem);
        /* X1 goes leader -> follower, X2 follower -> leader. */
        const bool sends = (id == RAD_PP_X1) == pp_leader();
        net::ring_init(r, id, (uint64_t)cap, sends ? 0 : cfg_.tp,
                       (uint64_t)env_ms("RADIANCE_PP_RECV_DEADLINE_MS", 600000));
        (sends ? pp_send_ : pp_recv_)[id] = r;
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
            if (p.kind == RAD_KV_FULL && p.block_size > 0 && !p.zero) {
                const int64_t t = p.n_blocks * p.block_size;
                if (tokens == 0 || t < tokens) tokens = t;
            }
        }
        f.add("kv tokens", std::to_string(tokens));
    }
    f.add("step batch layout", fmt("%016llx", (unsigned long long)sched_.wire_layout()));
    for (int id = 0; id < RAD_PP_N_RINGS; ++id)
        f.add(fmt("crossing X%d", id + 1), fmt("%lld bytes a row in %d part(s)", (long long)g.row_bytes[id].load(),
                                               g.parts[id].load()));
    net::Facts theirs;
    std::string err;
    if (net::exchange(*pp_link_, net::kHello2, f, &theirs, pp_link_->opts().connect_ms, &err) < 0) {
        RAD_ERR("two-node pipeline: the second rendezvous failed: %s", err.c_str());
        return RAD_E_IO;
    }
    std::vector<std::string> diff = net::differences(f, theirs, { "kv tokens" });
    if (!diff.empty()) {
        for (const std::string& d : diff) RAD_ERR("two-node pipeline: the nodes differ: %s", d.c_str());
        RAD_ERR("two-node pipeline: %zu difference(s) after the KV carve: both refuse to serve", diff.size());
        pp_link_->fail("the second rendezvous found differences between the nodes");
        return RAD_E_STATE;
    }
    /* KV: block ids are the leader's; neither node may hand out one the other has not carved. The
     * leader caps its full groups at the smaller capacity with a standing loan (KVManager::set_loan;
     * nothing reclaims it, since the loan to the experts is off on a pipeline node). */
    const int64_t mine = std::atoll(f.find("kv tokens")->c_str());
    const int64_t peer = std::atoll(theirs.find("kv tokens") ? theirs.find("kv tokens")->c_str() : "0");
    if (pp_leader() && peer < mine) {
        const int ls = ranks_[0]->kv.set_loan(mine - peer);
        if (ls < 0) {
            RAD_ERR("two-node pipeline: cannot cap the leader's KV at the follower's %lld tokens", (long long)peer);
            return ls;
        }
        for (size_t i = 1; i < ranks_.size(); ++i) (void)ranks_[i]->kv.follow(ranks_[0]->kv);
        RAD_INFO("two-node pipeline: the leader's KV capped at the follower's %lld tokens (it carved %lld)",
                 (long long)peer, (long long)mine);
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
            net::verbs_open(*pp_link_, dev ? dev : "auto", ip, gid ? std::atoi(gid) : -1, pp_send_,
                            pp_recv_, &err);
        if (!dp) {
            RAD_ERR("two-node pipeline: --node-link verbs: %s", err.c_str());
            return RAD_E_UNSUPPORTED;
        }
        RAD_INFO("two-node pipeline: %s", dp->describe().c_str());
        pp_link_->set_data_path(std::move(dp));
    }
    pp_link_->attach(pp_send_, pp_recv_);
    pp_link_->start();
    for (int id = 0; id < RAD_PP_N_RINGS; ++id)
        g.ring[id] = pp_send_[id] ? pp_send_[id] : pp_recv_[id];
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
    if (pp_leader())
        sched_.set_build_hook([this](const RadBatch* b) { pp_on_build(b); });
    RAD_INFO("two-node pipeline: the %s of W(%d,%d) is ready; link %s:%d (%s), crossings X1 %lld / X2 %lld "
             "bytes a row (rings %s / %s), KV %lld tokens here, %lld on the peer",
             role(cfg_.pp_rank), cfg_.pp_cut_h, cfg_.pp_cut_t, cfg_.pp_peer.c_str(), cfg_.pp_port,
             cfg_.node_link ? "verbs" : "tcp", (long long)g.row_bytes[0].load(), (long long)g.row_bytes[1].load(),
             humanb(pp_ring_cap_[0]).c_str(), humanb(pp_ring_cap_[1]).c_str(), (long long)mine, (long long)peer);
    return RAD_OK;
}

/* ================================================================== the leader */
void Engine::pp_on_build(const RadBatch* b) {
    if (!pp_link_ || !b) return;
    if (pp_trunk_pending_) {
        pp_link_->fail("the leader built a pass before its trunk step was released (a schedule this "
                       "build does not ship)");
        return;
    }
    if (b->draft_pass == 0 && b->enc == 0) { pp_trunk_pending_ = true; return; }
    /* A drafter's pass: its uploads, not its pass, on the follower. */
    (void)pp_ship(b, false);
}

int Engine::pp_ship(const RadBatch* b, bool run) {
    if (!pp_link_) return RAD_OK;
    BatchBuilder::Wire w;
    const char* span = nullptr;
    if (!sched_.wire_last(&w, &span) || w.step != b->step || w.draft_pass != b->draft_pass) {
        pp_link_->fail(fmt("step %d: the build to ship is not the batch's", b->step));
        return RAD_E_STATE;
    }
    StepHdr h;
    h.run = run ? 1 : 0;
    h.step = b->step;
    if (run) {
        h.n_fn4 = std::min<int>(fn4::kSwitches, kMaxFn4);
        for (int i = 0; i < h.n_fn4; ++i) h.fn4[i] = fn4::value(i);
        h.psk_valid = psk_last_valid_ ? 1 : 0;
        h.psk_dec = psk_last_dec_;
        h.psk_n = psk_last_valid_ ? (int32_t)psk_ahead_.size() : 0;
    }
    std::vector<uint8_t> head(sizeof h + (size_t)h.psk_n * 3 * sizeof(int32_t) + sizeof w);
    uint8_t* q = head.data();
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
    std::memcpy(q, &w, sizeof w);
    const int st = pp_link_->send2(net::kStep, (uint64_t)b->step, h.run, head.data(), (int64_t)head.size(),
                                   span, w.used);
    if (st < 0) return st;
    ++pp_builds_shipped_;
    if (run) ++pp_steps_shipped_;
    return RAD_OK;
}

void Engine::pp_close(const char* why) {
    if (!pp_link_) return;
    pp_link_->close(why);
    RAD_INFO("two-node pipeline: link closed (%s); %lld steps and %lld builds shipped", why,
             (long long)pp_steps_shipped_, (long long)pp_builds_shipped_);
}

/* ================================================================== the follower */
int Engine::follower_loop() {
    if (rad_dev_set(ranks_[0]->device) < 0) RAD_ERR("follower loop: could not bind rank 0's device");
    rad::StallWatch watch("follower", 20000);
    int64_t steps = 0, builds = 0;
    std::vector<int32_t> ahead, rows, flags;
    while (!stopping_.load(std::memory_order_acquire)) {
        watch.progress();
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
        if (m.h.kind != net::kStep) {
            pp_link_->fail(fmt("the follower got a %s frame while serving", net::frame_name(m.h.kind)));
            return fail_loop(RAD_E_IO);
        }
        StepHdr h;
        BatchBuilder::Wire w;
        if (m.body.size() < sizeof h) { pp_link_->fail("a step frame shorter than its header"); return fail_loop(RAD_E_IO); }
        std::memcpy(&h, m.body.data(), sizeof h);
        const size_t psk_bytes = (size_t)std::max(0, h.psk_n) * 3 * sizeof(int32_t);
        if (h.magic != kStepMagic || h.psk_n < 0 || h.n_fn4 < 0 || h.n_fn4 > kMaxFn4 ||
            m.body.size() < sizeof h + psk_bytes + sizeof w) {
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
        std::memcpy(&w, q, sizeof w);
        q += sizeof w;
        const int64_t span_bytes = (int64_t)(m.body.size() - (size_t)(q - m.body.data()));
        const RadBatch* b = sched_.wire_replay(w, (const char*)q, span_bytes, h.run != 0);
        if (!b) {
            pp_link_->fail(fmt("the follower could not replay step %d", h.step));
            return fail_loop(RAD_E_STATE);
        }
        ++builds;
        if (!h.run) continue;
        /* The leader's switches, as its release applied them. */
        if (h.n_fn4 > 0) {
            bool changed = false;
            for (int i = 0; i < h.n_fn4 && i < fn4::kSwitches; ++i) {
                if (fn4::value(i) == h.fn4[i]) continue;
                std::string err;
                if (!fn4::request(fn4::switch_name(i), h.fn4[i], &err)) {
                    pp_link_->fail(fmt("the follower cannot take the leader's switch %s=%d: %s",
                                       fn4::switch_name(i), h.fn4[i], err.c_str()));
                    return fail_loop(RAD_E_STATE);
                }
                changed = true;
            }
            if (changed) {
                int sw[fn4::kSwitches];
                (void)fn4::apply_pending(h.step, sw);
            }
        }
        /* The leader's PSK plan for this pass, handed to the plugin as psk_release does. */
        int32_t end = INT32_MAX;
        if (h.psk_valid) {
            const int l = fn4::psk_plan(h.step, h.psk_dec, h.psk_n, ahead.data(), rows.data(), flags.data());
            if (l >= 0) end = l;
        }
        for (auto& r : ranks_) r->stager.set_layer_end(end);
        issue_what_.store(kIssueFollower, std::memory_order_release);
        step_batch_.store(b, std::memory_order_release);
        barrier_->arrive_and_wait();      /* release the ranks */
        barrier_->arrive_and_wait();      /* and wait for them */
        st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("follower step %d failed: %s -- the pipeline cannot continue", h.step, rad_strerror(st));
            pp_link_->fail(fmt("the follower's step %d failed: %s", h.step, rad_strerror(st)));
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
