/* node_link.h -- the link between the two nodes of a pipelined engine (tp8 S4; abi/rad_pp.h).
 *
 * One connection carries everything: the start-up rendezvous, the leader's step messages (the
 * step batch's staging span, which the follower replays on its own cards), the crossings' payloads
 * (X1, X2), heartbeats and the abort that takes both nodes down. Every frame is a 64-byte header
 * with its own CRC-32C, then the payload; a control payload carries a CRC-32C too, a crossing's
 * only on request (RADIANCE_PP_CRC_DATA=1), since at prefill one is 170 MB.
 *
 * Two transports. Kernel TCP (node_link_tcp.cpp) carries every frame on the socket; it is the
 * bring-up link and the fallback. RDMA verbs (node_link_verbs.cpp) keep the socket for control and
 * move the crossings' payloads by RDMA WRITE from the sender's pinned send ring into the
 * receiver's registered receive ring, followed by a WRITE_WITH_IMM the receiver's completion
 * thread turns into `landed` (host-staged at both ends: the NIC touches only pinned host memory).
 *
 * Threads, each started by start(): a receive thread (frames off the socket: control into a queue
 * recv() drains, crossings straight into their receive ring), a proxy thread (watches every send
 * ring's `ready` and posts the payload), a heartbeat thread (a frame every heartbeat_ms when
 * nothing else went out; the link is declared down after dead_ms without a frame from the peer).
 * Any failure -- a short read, a bad CRC, a frame out of order, a missed heartbeat, the peer's
 * abort -- runs fail(): the reason is kept, every ring's `abort` word is set so a stage_recv
 * waiting on a card or a host thread gives up, and every recv() returns RAD_E_IO.
 */
#pragma once
#include "rad_internal.h"
#include "rad_pp.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad::net {

/* ------------------------------------------------------------------ frames */
constexpr uint32_t kWireMagic   = 0x31505052u;   /* "RPP1" */
constexpr uint16_t kWireVersion = 1;

enum FrameKind : uint16_t {
    kHello     = 1,   /* rendezvous, phase 1 (before declare): the node's facts */
    kHello2    = 2,   /* rendezvous, phase 2 (after the KV carve): geometry, layout, crossings */
    kStep      = 3,   /* leader -> follower: one step (BatchWire header + the staging span) */
    kData      = 4,   /* a crossing's payload; lane = the ring id, aux = rows, aux2 = parts */
    kHeartbeat = 5,
    kAbort     = 6,   /* the sender is going down; payload = the reason */
    kBye       = 7,   /* a clean stop */
    kVerbs     = 8,   /* the verbs data path's endpoint (QP, GID, ring addresses and keys) */
    kReady     = 9,   /* rendezvous done on this side; the payload names the side */
    kKvCap     = 10,  /* before the KV carve: the tokens this node can page (both cap at the least) */
    kFn4Ask    = 11,  /* leader -> follower: would this node take switch name=value? (seq = ask id) */
    kFn4Ans    = 12,  /* follower -> leader: the answer to kFn4Ask (aux: 1 takes, 0 serves none, -1 refuses) */
    kFn4State  = 13,  /* follower -> leader: the switch values the follower applies (after a change) */
    kUser      = 32,  /* first kind a test may use */
};

enum FrameFlags : uint32_t {
    kFlagCrc = 1u,    /* crc_payload covers the payload */
};

struct WireHdr {
    uint32_t magic;
    uint16_t version;
    uint16_t kind;
    uint32_t lane;
    uint32_t flags;
    uint64_t seq;
    uint64_t bytes;
    uint64_t aux;
    uint64_t aux2;
    uint64_t reserved;
    uint32_t crc_payload;
    uint32_t crc_hdr;      /* CRC-32C of the 60 bytes before it */
};
static_assert(sizeof(WireHdr) == 64, "the frame header is 64 bytes on both nodes");

/* CRC-32C (Castagnoli), SSE4.2 where the CPU has it. `crc` chains calls. */
uint32_t crc32c(const void* p, size_t n, uint32_t crc = 0);

/* Fills magic, version and crc_hdr; the caller sets the rest. */
void     seal(WireHdr* h);
/* Why a header read off the wire is not one, or empty when it is. */
std::string check_header(const WireHdr& h, int64_t max_bytes);

const char* frame_name(uint16_t kind);

/* The largest payload any frame may carry: a prefill crossing's rows at the widest live set the
 * plugins declare stay far below it; a corrupt length above it is refused before any read. */
constexpr int64_t kMaxFrameBytes = int64_t(4) << 30;

/* ------------------------------------------------------------------ the link */
struct Msg {
    WireHdr              h{};
    std::vector<uint8_t> body;
};

struct LinkOpts {
    int  heartbeat_ms  = 100;
    int  dead_ms       = 2000;    /* RADIANCE_PP_DEAD_MS */
    int  connect_ms    = 600000;  /* how long the leader retries the follower, the follower waits */
    bool crc_data      = false;   /* RADIANCE_PP_CRC_DATA */
    int  poll_us       = 20;      /* the proxy's sleep between scans of idle send rings; 0 spins */
    /* Robustness 2: the longest this node's own stage may run once its input is there (the
     * follower: X1 landed -> its X2 published; the leader: a step released -> its X1 published)
     * before the link goes down naming the stuck stage. 0 = no limit (the rings' device deadline,
     * RADIANCE_PP_RECV_DEADLINE_MS, alone). RADIANCE_PP_STAGE_MS. */
    int  stage_ms      = 180000;
    /* The environment's settings over the defaults (RADIANCE_PP_DEAD_MS, _HEARTBEAT_MS,
     * _CRC_DATA, _POLL_US, _STAGE_MS). */
    static LinkOpts from_env();
};

class DataPath;   /* node_link_verbs.cpp */

class NodeLink {
public:
    /* The follower listens, the leader connects; both return a link whose threads are not yet
     * running (start()). Null with the reason in `err`. */
    /* `expect` (robustness 4): the addresses the leader may connect from (--pp-peer resolved); a
     * connection from any other is closed and the wait goes on. Empty: the first connection. */
    static std::unique_ptr<NodeLink> listen(const std::string& bind_host, int port,
                                            const LinkOpts& o, std::string* err,
                                            const std::string& expect = std::string());
    static std::unique_ptr<NodeLink> connect(const std::string& host, int port,
                                             const LinkOpts& o, std::string* err);
    /* For tests: a link over an already connected stream socket. */
    static std::unique_ptr<NodeLink> adopt(int fd, const LinkOpts& o);

    ~NodeLink();
    NodeLink(const NodeLink&) = delete;
    NodeLink& operator=(const NodeLink&) = delete;

    /* The crossings this node takes part in: rings it sends from and rings the peer's payloads
     * land in, indexed by ring id (RAD_PP_X1 / RAD_PP_X2). Before start(). A null entry is a ring
     * this node does not hold in that role. */
    void attach(RadPpRing* const send[RAD_PP_N_RINGS], RadPpRing* const recv[RAD_PP_N_RINGS]);
    /* tp8 S5: the rings of `n` ring indices (lane * RAD_PP_N_RINGS + crossing, abi/rad_pp.h), at most
     * RAD_PP_MAX_RING_IX. attach() is attach_n(.., RAD_PP_N_RINGS): one lane. */
    void attach_n(RadPpRing* const* send, RadPpRing* const* recv, int n);

    /* The verbs data path replaces the socket for kData frames (node_link_verbs.cpp). Before
     * start(), after attach(). */
    void set_data_path(std::unique_ptr<DataPath> dp);
    DataPath* data_path() const { return dp_.get(); }

    void start();

    /* One control frame; the payload is CRC'd. Thread-safe. RAD_E_IO once the link is down. */
    int send(uint16_t kind, uint64_t seq, uint64_t aux, const void* p, int64_t n);
    /* The same with the payload in two pieces, sent without joining them. */
    int send2(uint16_t kind, uint64_t seq, uint64_t aux, const void* a, int64_t na,
              const void* b, int64_t nb);
    /* The next control frame (not a heartbeat or a crossing). `timeout_ms` < 0 waits for ever.
     * RAD_E_IO when the link is down (why() says why), RAD_E_NOTFOUND on a timeout. Before
     * start() it reads the socket itself (the rendezvous runs this way). */
    int recv(Msg* m, int timeout_ms);

    bool alive() const { return !down_.load(std::memory_order_acquire); }
    /* The link went down because the peer stopped cleanly (kBye), not on a fault. */
    bool peer_stopped() const { return peer_bye_.load(std::memory_order_acquire); }
    std::string why() const;

    /* Takes the link down: keeps the first reason, sets every ring's abort word, wakes recv(),
     * tells the peer (best effort) unless `from_peer`. Idempotent. */
    void fail(const std::string& why, bool from_peer = false);
    /* A clean stop: tells the peer, stops the threads, closes the socket. */
    void close(const char* reason = "the node is stopping");

    /* Counters for the log and the tests. */
    struct Stats {
        uint64_t frames_out = 0, frames_in = 0, bytes_out = 0, bytes_in = 0;
        uint64_t data_out = 0, data_in = 0, heartbeats_in = 0;
    };
    Stats stats() const;

    /* Called on the receive thread for every kData frame landed (ring id, seq); tests hook it. */
    std::function<void(int, uint64_t)> on_landed;
    /* Called on the receive thread for every control frame once started; true consumes it (it is
     * not queued for recv()). Set before start(). The leader takes the follower's switch answers
     * and state here (engine_node.cpp). */
    std::function<bool(Msg&)> on_control;
    /* The leader: trunk steps released so far. Its X1 for step k is due from then (the stage
     * watchdog, LinkOpts::stage_ms). */
    void note_released(uint64_t k) { released_.store(k, std::memory_order_release); }

    /* The data path's hand-off: a payload of ring `id` written in full by the NIC (verbs) or the
     * receive thread (TCP); publishes it to the local consumers. Checks order and overrun. */
    int land(int id, uint64_t seq, uint64_t rows, uint64_t bytes, uint64_t parts);
    /* By ring index (lane * RAD_PP_N_RINGS + crossing). */
    RadPpRing* recv_ring(int id) const { return id >= 0 && id < RAD_PP_MAX_RING_IX ? recv_[id] : nullptr; }
    RadPpRing* send_ring(int id) const { return id >= 0 && id < RAD_PP_MAX_RING_IX ? send_[id] : nullptr; }
    int fd() const { return fd_; }
    const LinkOpts& opts() const { return o_; }

private:
    explicit NodeLink(int fd, const LinkOpts& o);
    /* `try_only`: give up (RAD_E_FULL) when another writer holds the socket (the heartbeat). */
    int  write_frame(WireHdr& h, const void* a, int64_t na, const void* b, int64_t nb, bool try_only = false);
    /* A frame of at most a few hundred bytes written with a deadline, the socket taken only if it is
     * free within `lock_ms` (close()'s kBye, fail()'s kAbort): never blocks behind a stuck writer. */
    void write_bounded(uint16_t kind, const std::string& body, int lock_ms, int send_ms);
    /* The stage watchdog, from the proxy thread. */
    void stage_watch();
    int  read_exact(void* p, int64_t n, int idle_ms);
    int  read_frame(Msg* m, bool* is_data, int idle_ms);
    void rx_loop();
    void proxy_loop();
    void hb_loop();
    int  send_data(int id, RadPpRing* r);

    int      fd_ = -1;
    LinkOpts o_;
    RadPpRing* send_[RAD_PP_MAX_RING_IX] = {};
    RadPpRing* recv_[RAD_PP_MAX_RING_IX] = {};
    uint64_t   sent_[RAD_PP_MAX_RING_IX] = {};
    int        n_ix_ = RAD_PP_N_RINGS;   /* ring indices attached: RAD_PP_N_RINGS a lane */
    std::unique_ptr<DataPath> dp_;

    std::mutex              tx_mu_;
    std::atomic<bool>       down_{ false };
    std::atomic<bool>       stopping_{ false };
    std::atomic<bool>       started_{ false };
    std::atomic<bool>       peer_bye_{ false };   /* the peer said kBye: a clean stop */
    mutable std::mutex      why_mu_;
    std::string             why_;

    std::mutex              q_mu_;
    std::condition_variable q_cv_;
    std::deque<Msg>         q_;

    std::atomic<uint64_t>   last_rx_ns_{ 0 }, last_tx_ns_{ 0 };
    /* Robustness 1: a writer in a frame (holding tx_mu_) and when it last moved a byte. */
    std::atomic<bool>       tx_busy_{ false };
    std::atomic<bool>       tx_torn_{ false };   /* a writer gave up mid-frame: no frame may follow */
    std::atomic<uint64_t>   tx_progress_ns_{ 0 };
    std::atomic<uint64_t>   released_{ 0 };
    uint64_t                due_k_ = 0, due_ns_ = 0, watch_ns_ = 0;   /* the proxy thread's */
    std::thread             rx_, proxy_, hb_;
    mutable std::mutex      st_mu_;
    Stats                   st_;
};

/* The verbs data path: RC QP between the two nodes' NICs, each crossing's receive ring registered
 * on the receiver and written by the sender's NIC. Built only where rdma-core's headers were found
 * at configure time (RAD_HAVE_VERBS); libibverbs is opened at run time, so a host without it runs
 * every other part of the engine. */
class DataPath {
public:
    virtual ~DataPath() = default;
    /* Posts ring `id`'s payload of `bytes` as sequence `seq`; returns when the NIC has taken it
     * (the send ring may then be rewritten). */
    virtual int post(int id, const RadPpRing* r, uint64_t seq, uint64_t rows, uint64_t bytes,
                     uint64_t parts) = 0;
    virtual void stop() = 0;
    virtual std::string describe() const = 0;
};

/* Sets the verbs path up over `link` (which must be connected and not started): opens the RDMA
 * device that holds `local_ip` (or `dev` by name), registers the rings, exchanges the endpoint
 * over the link (kVerbs) and brings the QP to RTS. `gid_index` < 0 picks the RoCE v2 IPv4 GID of
 * `local_ip`. Null with the reason when anything is missing. */
std::unique_ptr<DataPath> verbs_open(NodeLink& link, const std::string& dev,
                                     const std::string& local_ip, int gid_index,
                                     RadPpRing* const send[RAD_PP_N_RINGS],
                                     RadPpRing* const recv[RAD_PP_N_RINGS], std::string* err);
/* tp8 S5: the same over `n` ring indices (lane * RAD_PP_N_RINGS + crossing), at most RAD_PP_MAX_RING_IX. */
std::unique_ptr<DataPath> verbs_open_n(NodeLink& link, const std::string& dev,
                                       const std::string& local_ip, int gid_index,
                                       RadPpRing* const* send, RadPpRing* const* recv, int n,
                                       std::string* err);
/* Whether this build carries the verbs path, and whether libibverbs loads here. */
bool verbs_built();
bool verbs_loadable(std::string* why);

/* Robustness 4: the local address the kernel would send from to reach `host` (no packet sent), ""
 * when it cannot tell; the follower binds it so it listens only where the leader can reach it. */
std::string local_address_for(const std::string& host, int port);

/* ------------------------------------------------------------------ rings */
/* A ring set up over `mem` (at least RAD_PP_RING_HDR + cap bytes, 64-byte aligned). */
void ring_init(RadPpRing* r, int id, uint64_t cap, int n_ranks, uint64_t deadline_ms);
/* A host thread's stage_send / stage_recv, for the tests and the host backend's kernels (libref
 * has its own copy of the same steps). */
int  ring_put(RadPpRing* r, const void* p, uint64_t bytes, uint64_t rows);
int  ring_take(RadPpRing* r, int rank, void* p, uint64_t bytes, uint64_t rows);

}  /* namespace rad::net */
