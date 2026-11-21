/* node_link_verbs.cpp -- the RDMA data path of the two-node link (node_link.h, DataPath).
 *
 * Host-staged at both ends (tp8 PLAN section 3): the sender's NIC reads the pinned send ring, the
 * receiver's NIC writes the registered receive ring; no card memory is registered, so neither the
 * far-quadrant NIC rate nor dma-buf eviction enters (DRIVER.md 2.5). Per payload, on one RC QP:
 *
 *   1. RDMA WRITE, inline, 32 bytes {seq, rows, bytes, parts} into the receive ring's mailbox
 *      (RAD_PP_RING_HDR - 64, a part of the header no counter uses);
 *   2. RDMA WRITE_WITH_IMM of the payload into the receive ring's payload, immediate =
 *      ring id << 24 | (seq & 0xffffff), signalled; post() returns when its completion is in, so
 *      the sender's stage_send can rewrite the ring.
 *
 * RC executes the two in order, so when the receiver's completion thread sees the immediate the
 * mailbox and the payload are both in memory; it then calls NodeLink::land, which publishes the
 * payload with `landed` (mode I of the S2 bench, the spec-safe one: no reliance on PCIe ordering of
 * a NIC write against a flag word). Control frames, the rendezvous and the heartbeat stay on the
 * TCP socket.
 *
 * libibverbs is opened with dlopen, so the engine binary does not depend on it: a host without
 * rdma-core runs everything else, and `--node-link verbs` there stops the start with the reason.
 * The static inline verbs (ibv_post_send, ibv_post_recv, ibv_poll_cq) call through the context's
 * ops table and need no symbol.
 *
 * Status (17.11.2026): compiled against rdma-core 50 (Ubuntu 24.04), never run -- the workstation
 * has no RDMA device and no soft-RoCE (rdma_rxe needs root). The QP set-up follows the S2 bench
 * (~/gpu/bench-rdma/src/ppx.cpp rdma_setup), which ran on gpu02 / gpu03 on 14.11.
 */
#include "net/node_link.h"
#include "runtime/fn4_switch.h"   /* tp8 loop D63: host_slack */

#if defined(RAD_HAVE_VERBS)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion"
#include <infiniband/verbs.h>
#pragma GCC diagnostic pop
#endif

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <fstream>

namespace rad::net {

#if !defined(RAD_HAVE_VERBS)

bool verbs_built() { return false; }
bool verbs_loadable(std::string* why) {
    if (why) *why = "this build has no RDMA verbs path (rdma-core's headers were not found at configure time)";
    return false;
}
std::unique_ptr<DataPath> verbs_open(NodeLink&, const std::string&, const std::string&, int,
                                     RadPpRing* const[RAD_PP_N_RINGS], RadPpRing* const[RAD_PP_N_RINGS],
                                     std::string* err) {
    *err = "this build has no RDMA verbs path (rdma-core's headers were not found at configure time); "
           "use --node-link tcp";
    return nullptr;
}
std::unique_ptr<DataPath> verbs_open_n(NodeLink&, const std::string&, const std::string&, int,
                                       RadPpRing* const*, RadPpRing* const*, int, std::string* err) {
    *err = "this build has no RDMA verbs path (rdma-core's headers were not found at configure time); "
           "use --node-link tcp";
    return nullptr;
}

#else

namespace {

/* ------------------------------------------------------------------ libibverbs, opened at run time */
struct Ibv {
    void* so = nullptr;
    ibv_device** (*get_device_list)(int*) = nullptr;
    void (*free_device_list)(ibv_device**) = nullptr;
    const char* (*get_device_name)(ibv_device*) = nullptr;
    ibv_context* (*open_device)(ibv_device*) = nullptr;
    int (*close_device)(ibv_context*) = nullptr;
    ibv_pd* (*alloc_pd)(ibv_context*) = nullptr;
    int (*dealloc_pd)(ibv_pd*) = nullptr;
    ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int) = nullptr;
    int (*dereg_mr)(ibv_mr*) = nullptr;
    ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int) = nullptr;
    int (*destroy_cq)(ibv_cq*) = nullptr;
    ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*) = nullptr;
    int (*destroy_qp)(ibv_qp*) = nullptr;
    int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int) = nullptr;
    int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*) = nullptr;
    int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*) = nullptr;
    const char* (*wc_status_str)(ibv_wc_status) = nullptr;
    std::string err;

    bool load() {
        if (so) return true;
        so = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!so) { err = fmt("libibverbs.so.1 does not load: %s", dlerror()); return false; }
        bool ok = true;
        auto sym = [&](auto& f, const char* n) {
            f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(dlsym(so, n));
            if (!f) { ok = false; err += fmt("%s%s", err.empty() ? "libibverbs lacks " : ", ", n); }
        };
        sym(get_device_list, "ibv_get_device_list");
        sym(free_device_list, "ibv_free_device_list");
        sym(get_device_name, "ibv_get_device_name");
        sym(open_device, "ibv_open_device");
        sym(close_device, "ibv_close_device");
        sym(alloc_pd, "ibv_alloc_pd");
        sym(dealloc_pd, "ibv_dealloc_pd");
        sym(reg_mr, "ibv_reg_mr");
        sym(dereg_mr, "ibv_dereg_mr");
        sym(create_cq, "ibv_create_cq");
        sym(destroy_cq, "ibv_destroy_cq");
        sym(create_qp, "ibv_create_qp");
        sym(destroy_qp, "ibv_destroy_qp");
        sym(modify_qp, "ibv_modify_qp");
        sym(query_gid, "ibv_query_gid");
        /* The exported ibv_query_port fills the compat prefix of ibv_port_attr (verbs.h's macro
         * wraps it); state and active_mtu are in that prefix. */
        sym(query_port, "ibv_query_port");
        sym(wc_status_str, "ibv_wc_status_str");
        if (!ok) { dlclose(so); so = nullptr; }
        return ok;
    }
};
Ibv& ibv() { static Ibv v; return v; }

std::string sysrd(const std::string& f) {
    std::ifstream in(f);
    std::string s;
    std::getline(in, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

/* "10.200.0.3" -> "ffff:0ac8:0003", the tail of its IPv4-mapped GID. */
std::string ipv4_gid_tail(const std::string& ip) {
    in_addr a{};
    if (inet_pton(AF_INET, ip.c_str(), &a) != 1) return "";
    const uint8_t* b = (const uint8_t*)&a;
    return fmt("ffff:%02x%02x:%02x%02x", b[0], b[1], b[2], b[3]);
}

/* The RoCE v2 IPv4 GID of `ip` on port 1 of `d`, or -1. */
int find_gid(ibv_device* d, const std::string& ip, std::string* gid) {
    const std::string base = std::string(d->ibdev_path) + "/ports/1/";
    const std::string want = ipv4_gid_tail(ip);
    for (int i = 0; i < 256; ++i) {
        const std::string g = sysrd(base + "gids/" + std::to_string(i));
        const std::string t = sysrd(base + "gid_attrs/types/" + std::to_string(i));
        if (g.empty() || g == "0000:0000:0000:0000:0000:0000:0000:0000") continue;
        if (t.find("v2") == std::string::npos) continue;
        if (g.find("0000:0000:0000:0000:0000:ffff:") != 0) continue;
        if (!want.empty() && (g.size() < want.size() ||
                              g.compare(g.size() - want.size(), want.size(), want) != 0))
            continue;
        if (gid) *gid = g;
        return i;
    }
    return -1;
}

/* What each side tells the other, as one kVerbs frame. */
struct Endpoint {
    uint32_t qpn = 0, psn = 0;
    uint8_t  gid[16] = {};
    uint32_t mtu = 0;
    uint32_t pad = 0;
    /* this side's receive rings, by ring index (tp8 S5: lane * RAD_PP_N_RINGS + crossing) */
    uint64_t ring_addr[RAD_PP_MAX_RING_IX] = {};
    uint32_t ring_rkey[RAD_PP_MAX_RING_IX] = {};
    uint64_t ring_cap[RAD_PP_MAX_RING_IX] = {};
};

constexpr uint64_t kMailboxOff = RAD_PP_RING_HDR - 64;
constexpr int      kRecvDepth = 64;

class Verbs final : public DataPath {
public:
    explicit Verbs(NodeLink& l) : link_(l) {}
    ~Verbs() override { stop(); teardown(); }

    int open(const std::string& devname, const std::string& local_ip, int gid_index,
             RadPpRing* const* send, RadPpRing* const* recv, int n_ix, std::string* err) {
        n_ix_ = std::max<int>(RAD_PP_N_RINGS, std::min<int>(n_ix, RAD_PP_MAX_RING_IX));
        Ibv& F = ibv();
        if (!F.load()) { *err = F.err; return RAD_E_UNSUPPORTED; }
        int nd = 0;
        dl_ = F.get_device_list(&nd);
        if (!dl_ || nd == 0) {
            *err = "no RDMA device (a container without --device /dev/infiniband?)";
            return RAD_E_NOTFOUND;
        }
        for (int i = 0; i < nd && !dev_; ++i) {
            if (devname.empty() || devname == "auto") {
                std::string g;
                if (find_gid(dl_[i], local_ip, &g) >= 0) dev_ = dl_[i];
            } else if (devname == F.get_device_name(dl_[i])) {
                dev_ = dl_[i];
            }
        }
        if (!dev_) {
            *err = devname.empty() || devname == "auto"
                ? fmt("no RDMA device holds a RoCE v2 GID for %s (%d device(s) present)", local_ip.c_str(), nd)
                : fmt("RDMA device %s not found (%d present)", devname.c_str(), nd);
            return RAD_E_NOTFOUND;
        }
        gidx_ = gid_index >= 0 ? gid_index : find_gid(dev_, local_ip, &gid_str_);
        if (gidx_ < 0) {
            *err = fmt("no RoCE v2 IPv4 GID for %s on %s: pass --node-gid N", local_ip.c_str(),
                       F.get_device_name(dev_));
            return RAD_E_NOTFOUND;
        }
        ctx_ = F.open_device(dev_);
        if (!ctx_) { *err = fmt("ibv_open_device: %s", strerror(errno)); return RAD_E_DEVICE; }
        ibv_port_attr pa{};
        if (F.query_port(ctx_, 1, &pa) != 0) { *err = fmt("ibv_query_port: %s", strerror(errno)); return RAD_E_DEVICE; }
        if (pa.state != IBV_PORT_ACTIVE) {
            *err = fmt("%s port 1 is not active (state %d): the link is down", F.get_device_name(dev_), (int)pa.state);
            return RAD_E_DEVICE;
        }
        mtu_ = pa.active_mtu;
        pd_ = F.alloc_pd(ctx_);
        if (!pd_) { *err = fmt("ibv_alloc_pd: %s", strerror(errno)); return RAD_E_DEVICE; }
        /* Every ring, send and receive, whole (header + payload). The send side's lkey covers the
         * payload the NIC reads; the receive side's rkey the mailbox and payload it writes. */
        for (int i = 0; i < n_ix_; ++i) {
            for (int side = 0; side < 2; ++side) {
                RadPpRing* r = side == 0 ? send[i] : recv[i];
                if (!r) continue;
                const int acc = IBV_ACCESS_LOCAL_WRITE | (side == 1 ? IBV_ACCESS_REMOTE_WRITE : 0);
                ibv_mr* mr = F.reg_mr(pd_, (void*)r, (size_t)(RAD_PP_RING_HDR + r->cap), acc);
                if (!mr) {
                    *err = fmt("ibv_reg_mr of crossing %d's %s ring (%s): %s -- run the container with "
                               "--ulimit memlock=-1:-1", i, side == 0 ? "send" : "receive",
                               humanb((int64_t)(RAD_PP_RING_HDR + r->cap)).c_str(), strerror(errno));
                    return RAD_E_NOMEM;
                }
                (side == 0 ? smr_ : rmr_)[i] = mr;
                (side == 0 ? send_ : recv_)[i] = r;
            }
        }
        scq_ = F.create_cq(ctx_, 256, nullptr, nullptr, 0);
        rcq_ = F.create_cq(ctx_, 4 * kRecvDepth, nullptr, nullptr, 0);
        if (!scq_ || !rcq_) { *err = fmt("ibv_create_cq: %s", strerror(errno)); return RAD_E_DEVICE; }
        ibv_qp_init_attr qia{};
        qia.send_cq = scq_;
        qia.recv_cq = rcq_;
        qia.qp_type = IBV_QPT_RC;
        qia.sq_sig_all = 0;
        qia.cap.max_send_wr = 128;
        qia.cap.max_recv_wr = 2 * kRecvDepth;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        qia.cap.max_inline_data = 64;
        qp_ = F.create_qp(pd_, &qia);
        if (!qp_) { *err = fmt("ibv_create_qp: %s", strerror(errno)); return RAD_E_DEVICE; }
        ibv_gid g{};
        if (F.query_gid(ctx_, 1, gidx_, &g) != 0) { *err = fmt("ibv_query_gid %d: %s", gidx_, strerror(errno)); return RAD_E_DEVICE; }

        Endpoint me;
        me.qpn = qp_->qp_num;
        me.psn = 0x3000u + (uint32_t)(getpid() & 0xfff);
        std::memcpy(me.gid, g.raw, 16);
        me.mtu = (uint32_t)mtu_;
        for (int i = 0; i < n_ix_; ++i)
            if (rmr_[i]) {
                me.ring_addr[i] = (uint64_t)(uintptr_t)recv_[i];
                me.ring_rkey[i] = rmr_[i]->rkey;
                me.ring_cap[i] = recv_[i]->cap;
            }
        /* The exchange: both sides send, then read, over the not yet started link. */
        if (link_.send(kVerbs, 0, 0, &me, sizeof me) < 0) { *err = "sending the verbs endpoint: " + link_.why(); return RAD_E_IO; }
        Msg m;
        const int rs = link_.recv(&m, link_.opts().connect_ms);
        if (rs < 0 || m.h.kind != kVerbs || m.body.size() != sizeof(Endpoint)) {
            *err = rs < 0 ? "the peer's verbs endpoint did not arrive: " + link_.why()
                          : fmt("expected the peer's verbs endpoint, got a %s frame of %zu bytes",
                                frame_name(m.h.kind), m.body.size());
            return RAD_E_IO;
        }
        std::memcpy(&peer_, m.body.data(), sizeof peer_);
        for (int i = 0; i < n_ix_; ++i)
            if (send_[i] && (!peer_.ring_addr[i] || peer_.ring_cap[i] < send_[i]->cap)) {
                *err = fmt("crossing %d: the peer's receive ring holds %llu bytes, this node sends up to %llu",
                           i, (unsigned long long)peer_.ring_cap[i], (unsigned long long)send_[i]->cap);
                return RAD_E_STATE;
            }

        const int ACC = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        ibv_qp_attr at{};
        at.qp_state = IBV_QPS_INIT;
        at.pkey_index = 0;
        at.port_num = 1;
        at.qp_access_flags = ACC;
        if (F.modify_qp(qp_, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            *err = fmt("QP to INIT: %s", strerror(errno));
            return RAD_E_DEVICE;
        }
        if (post_recvs(kRecvDepth) < 0) { *err = fmt("ibv_post_recv: %s", strerror(errno)); return RAD_E_DEVICE; }
        at = ibv_qp_attr{};
        at.qp_state = IBV_QPS_RTR;
        at.path_mtu = (ibv_mtu)std::min<uint32_t>(me.mtu, peer_.mtu);
        at.dest_qp_num = peer_.qpn;
        at.rq_psn = peer_.psn;
        at.max_dest_rd_atomic = 16;
        at.min_rnr_timer = 12;
        at.ah_attr.is_global = 1;
        std::memcpy(at.ah_attr.grh.dgid.raw, peer_.gid, 16);
        at.ah_attr.grh.sgid_index = (uint8_t)gidx_;
        at.ah_attr.grh.hop_limit = 64;
        at.ah_attr.port_num = 1;
        if (F.modify_qp(qp_, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
            *err = fmt("QP to RTR: %s", strerror(errno));
            return RAD_E_DEVICE;
        }
        at = ibv_qp_attr{};
        at.qp_state = IBV_QPS_RTS;
        /* 4.096 us x 2^14 = 67 ms a retry, 7 retries: a flap under ~0.5 s recovers without an
         * error (TF 2.7). */
        at.timeout = 14;
        at.retry_cnt = 7;
        at.rnr_retry = 7;
        at.sq_psn = me.psn;
        at.max_rd_atomic = 16;
        if (F.modify_qp(qp_, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                      IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
            *err = fmt("QP to RTS: %s", strerror(errno));
            return RAD_E_DEVICE;
        }
        char gs[64] = "?";
        inet_ntop(AF_INET6, peer_.gid, gs, sizeof gs);
        desc_ = fmt("verbs %s gid[%d] %s, QP 0x%x -> 0x%x (%s), mtu %d, RC timeout 14 retry 7",
                    F.get_device_name(dev_), gidx_, gid_str_.c_str(), me.qpn, peer_.qpn, gs,
                    128 << (int)std::min<uint32_t>(me.mtu, peer_.mtu));   /* the RTR's path_mtu (`at` holds the RTS attributes now) */
        cq_ = std::thread([this] { cq_loop(); });
        return RAD_OK;
    }

    int post(int id, const RadPpRing* r, uint64_t seq, uint64_t rows, uint64_t bytes,
             uint64_t parts) override {
        if (stop_.load(std::memory_order_acquire)) return RAD_E_IO;
        if (id < 0 || id >= n_ix_ || !smr_[id] || r != send_[id]) return RAD_E_STATE;
        std::lock_guard<std::mutex> lk(post_mu_);
        uint64_t mb[4] = { seq, rows, bytes, parts };
        ibv_sge ms{ (uint64_t)(uintptr_t)mb, (uint32_t)sizeof mb, 0 };
        ibv_sge ps{ (uint64_t)(uintptr_t)((const char*)r + RAD_PP_RING_HDR), (uint32_t)bytes, smr_[id]->lkey };
        ibv_send_wr w[2]{};
        w[0].wr_id = 1;
        w[0].sg_list = &ms;
        w[0].num_sge = 1;
        w[0].opcode = IBV_WR_RDMA_WRITE;
        w[0].send_flags = IBV_SEND_INLINE;
        w[0].wr.rdma.remote_addr = peer_.ring_addr[id] + kMailboxOff;
        w[0].wr.rdma.rkey = peer_.ring_rkey[id];
        w[0].next = &w[1];
        w[1].wr_id = 2;
        w[1].sg_list = bytes ? &ps : nullptr;
        w[1].num_sge = bytes ? 1 : 0;
        w[1].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        w[1].send_flags = IBV_SEND_SIGNALED;
        w[1].imm_data = htonl(((uint32_t)id << 24) | (uint32_t)(seq & 0xffffffu));
        w[1].wr.rdma.remote_addr = peer_.ring_addr[id] + RAD_PP_RING_HDR;
        w[1].wr.rdma.rkey = peer_.ring_rkey[id];
        ibv_send_wr* bad = nullptr;
        if (ibv_post_send(qp_, w, &bad) != 0) {
            link_.fail(fmt("verbs: ibv_post_send of crossing %d payload %llu: %s", id,
                           (unsigned long long)seq, strerror(errno)));
            return RAD_E_IO;
        }
        /* Wait for the payload's completion: the send ring is then free for the next stage_send. A
         * completion that never comes (a local HCA that stopped; RC retries give an error completion
         * for a silent remote, ~0.54 s) takes the link down after dead_ms (robustness 2). */
        const uint64_t t0 = rad_mono_ns();
        const uint64_t limit = (uint64_t)link_.opts().dead_ms * 1000000ull;
        uint32_t spins = 0;
        for (;;) {
            if (stop_.load(std::memory_order_acquire)) return RAD_E_IO;
            ibv_wc wc{};
            const int n = ibv_poll_cq(scq_, 1, &wc);
            if (n < 0) { link_.fail("verbs: polling the send completions failed"); return RAD_E_IO; }
            if (n == 0) {
                if ((++spins & 1023u) == 0 && rad_mono_ns() - t0 > limit) {
                    link_.fail(fmt("verbs: crossing %d payload %llu has no send completion after %d ms", id,
                                   (unsigned long long)seq, link_.opts().dead_ms));
                    return RAD_E_IO;
                }
                __builtin_ia32_pause();
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                link_.fail(fmt("verbs: crossing %d payload %llu completed with %s", id,
                               (unsigned long long)seq, ibv().wc_status_str(wc.status)));
                return RAD_E_IO;
            }
            if (wc.wr_id == 2) return RAD_OK;
        }
    }

    void stop() override {
        stop_.store(true, std::memory_order_release);
        if (cq_.joinable() && std::this_thread::get_id() != cq_.get_id()) cq_.join();
    }

    std::string describe() const override { return desc_; }

private:
    int post_recvs(int n) {
        for (int i = 0; i < n; ++i) {
            ibv_recv_wr w{};
            w.wr_id = 7;
            w.num_sge = 0;
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(qp_, &w, &bad) != 0) return RAD_E_DEVICE;
        }
        return RAD_OK;
    }

    void cq_loop() {
        uint32_t idle = 0;
        while (!stop_.load(std::memory_order_acquire)) {
            ibv_wc wc[16];
            const int n = ibv_poll_cq(rcq_, 16, wc);
            if (n < 0) { link_.fail("verbs: polling the receive completions failed"); return; }
            if (n == 0) {
                if (++idle > 1024 && link_.opts().poll_us > 0)
                    fn4::slack_sleep_us(link_.opts().poll_us, fn4::kSlackLinkRecv);   /* tp8 loop D63: under host_slack's timer slack */
                else
                    __builtin_ia32_pause();
                continue;
            }
            idle = 0;
            for (int k = 0; k < n; ++k) {
                if (wc[k].status != IBV_WC_SUCCESS) {
                    link_.fail(fmt("verbs: a receive completed with %s", ibv().wc_status_str(wc[k].status)));
                    return;
                }
                if (wc[k].opcode != IBV_WC_RECV_RDMA_WITH_IMM) continue;
                const uint32_t imm = ntohl(wc[k].imm_data);
                const int id = (int)(imm >> 24);
                RadPpRing* r = link_.recv_ring(id);
                if (!r) { link_.fail(fmt("verbs: a payload for crossing %d, which this node does not receive", id)); return; }
                uint64_t mb[4];
                std::memcpy(mb, (const char*)r + kMailboxOff, sizeof mb);
                const uint64_t landed = __atomic_load_n(&r->landed, __ATOMIC_ACQUIRE);
                if (mb[0] != landed + 1 || (mb[0] & 0xffffffu) != (imm & 0xffffffu)) {
                    link_.fail(fmt("verbs: crossing %d: payload %llu arrived after %llu", id,
                                   (unsigned long long)mb[0], (unsigned long long)landed));
                    return;
                }
                /* Robustness 7 / review F6: the NIC wrote this payload over the last one; every local
                 * reader must have taken that one already (the schedule guarantees it: tp8 S5 gives
                 * every lane its own rings and at most one step in flight, so the one-lane argument
                 * holds ring by ring). A reader still behind means a torn payload: loud, never silent. */
                for (uint64_t k = 0; k < r->n_ranks && k < RAD_PP_MAX_RANKS; ++k)
                    if (__atomic_load_n(&r->taken[k], __ATOMIC_ACQUIRE) < landed) {
                        link_.fail(fmt("verbs: crossing %d: payload %llu landed while local rank %llu had not taken "
                                       "payload %llu (a ring overrun)", id, (unsigned long long)mb[0],
                                       (unsigned long long)k, (unsigned long long)landed));
                        return;
                    }
                if (link_.land(id, mb[0], mb[1], mb[2], mb[3]) < 0) return;
                if (post_recvs(1) < 0) { link_.fail("verbs: re-posting a receive failed"); return; }
            }
        }
    }

    void teardown() {
        Ibv& F = ibv();
        if (!F.so) return;
        if (qp_) F.destroy_qp(qp_);
        if (scq_) F.destroy_cq(scq_);
        if (rcq_) F.destroy_cq(rcq_);
        for (int i = 0; i < RAD_PP_MAX_RING_IX; ++i) {
            if (smr_[i]) F.dereg_mr(smr_[i]);
            if (rmr_[i]) F.dereg_mr(rmr_[i]);
        }
        if (pd_) F.dealloc_pd(pd_);
        if (ctx_) F.close_device(ctx_);
        if (dl_) F.free_device_list(dl_);
        qp_ = nullptr; scq_ = rcq_ = nullptr; pd_ = nullptr; ctx_ = nullptr; dl_ = nullptr;
    }

    NodeLink&     link_;
    ibv_device**  dl_ = nullptr;
    ibv_device*   dev_ = nullptr;
    ibv_context*  ctx_ = nullptr;
    ibv_pd*       pd_ = nullptr;
    ibv_cq*       scq_ = nullptr;
    ibv_cq*       rcq_ = nullptr;
    ibv_qp*       qp_ = nullptr;
    ibv_mr*       smr_[RAD_PP_MAX_RING_IX] = {};
    ibv_mr*       rmr_[RAD_PP_MAX_RING_IX] = {};
    RadPpRing*    send_[RAD_PP_MAX_RING_IX] = {};
    RadPpRing*    recv_[RAD_PP_MAX_RING_IX] = {};
    int           n_ix_ = RAD_PP_N_RINGS;
    int           gidx_ = -1;
    int           mtu_ = 0;
    std::string   gid_str_, desc_;
    Endpoint      peer_;
    std::mutex    post_mu_;
    std::atomic<bool> stop_{ false };
    std::thread   cq_;
};

}  /* namespace */

bool verbs_built() { return true; }

bool verbs_loadable(std::string* why) {
    const bool ok = ibv().load();
    if (!ok && why) *why = ibv().err;
    return ok;
}

std::unique_ptr<DataPath> verbs_open(NodeLink& link, const std::string& dev, const std::string& local_ip,
                                     int gid_index, RadPpRing* const send[RAD_PP_N_RINGS],
                                     RadPpRing* const recv[RAD_PP_N_RINGS], std::string* err) {
    return verbs_open_n(link, dev, local_ip, gid_index, send, recv, RAD_PP_N_RINGS, err);
}

std::unique_ptr<DataPath> verbs_open_n(NodeLink& link, const std::string& dev, const std::string& local_ip,
                                       int gid_index, RadPpRing* const* send, RadPpRing* const* recv, int n,
                                       std::string* err) {
    auto v = std::make_unique<Verbs>(link);
    if (v->open(dev, local_ip, gid_index, send, recv, n, err) < 0) return nullptr;
    return v;
}

#endif  /* RAD_HAVE_VERBS */

}  /* namespace rad::net */
