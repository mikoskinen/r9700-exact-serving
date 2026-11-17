/* node_link.cpp -- the frames, the TCP transport and the rings of the two-node link (node_link.h). */
#include "net/node_link.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__)
#include <nmmintrin.h>
#endif

namespace rad::net {

/* ================================================================== CRC-32C */
namespace {
struct CrcTable {
    uint32_t t[256];
    CrcTable() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
            t[i] = c;
        }
    }
};
const CrcTable& crc_table() { static const CrcTable t; return t; }

uint32_t crc_sw(const uint8_t* p, size_t n, uint32_t c) {
    const CrcTable& T = crc_table();
    c = ~c;
    while (n--) c = T.t[(c ^ *p++) & 0xffu] ^ (c >> 8);
    return ~c;
}

#if defined(__x86_64__)
__attribute__((target("sse4.2"))) uint32_t crc_hw(const uint8_t* p, size_t n, uint32_t c) {
    uint64_t v = ~(uint64_t)c & 0xffffffffull;
    while (n >= 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        v = _mm_crc32_u64(v, w);
        p += 8; n -= 8;
    }
    uint32_t c32 = (uint32_t)v;
    while (n--) c32 = _mm_crc32_u8(c32, *p++);
    return ~c32;
}
#endif
}  /* namespace */

uint32_t crc32c(const void* p, size_t n, uint32_t crc) {
#if defined(__x86_64__)
    static const bool hw = __builtin_cpu_supports("sse4.2");
    if (hw) return crc_hw(static_cast<const uint8_t*>(p), n, crc);
#endif
    return crc_sw(static_cast<const uint8_t*>(p), n, crc);
}

void seal(WireHdr* h) {
    h->magic = kWireMagic;
    h->version = kWireVersion;
    h->crc_hdr = crc32c(h, offsetof(WireHdr, crc_hdr));
}

std::string check_header(const WireHdr& h, int64_t max_bytes) {
    if (h.magic != kWireMagic)
        return fmt("a frame without the link's magic (0x%08x): the stream is out of step or the "
                   "peer is not a radiance node", h.magic);
    if (h.crc_hdr != crc32c(&h, offsetof(WireHdr, crc_hdr)))
        return "a frame header whose CRC does not match: bytes were damaged or lost on the way";
    if (h.version != kWireVersion)
        return fmt("a frame of link version %u; this node speaks %u: the two nodes run different "
                   "builds", (unsigned)h.version, (unsigned)kWireVersion);
    if ((int64_t)h.bytes < 0 || (int64_t)h.bytes > max_bytes)
        return fmt("a %s frame announcing %llu payload bytes, past the %lld this link accepts",
                   frame_name(h.kind), (unsigned long long)h.bytes, (long long)max_bytes);
    return std::string();
}

const char* frame_name(uint16_t k) {
    switch (k) {
        case kHello: return "hello";
        case kHello2: return "hello2";
        case kStep: return "step";
        case kData: return "data";
        case kHeartbeat: return "heartbeat";
        case kAbort: return "abort";
        case kBye: return "bye";
        case kVerbs: return "verbs";
        case kReady: return "ready";
        case kKvCap: return "kv-cap";
        case kFn4Ask: return "fn4-ask";
        case kFn4Ans: return "fn4-answer";
        case kFn4State: return "fn4-state";
        default: return k >= kUser ? "user" : "unknown";
    }
}

/* ================================================================== options */
static int env_int(const char* k, int dflt, int lo, int hi) {
    const char* v = std::getenv(k);
    if (!v || !*v) return dflt;
    char* e = nullptr;
    const long n = std::strtol(v, &e, 10);
    if (e == v || *e || n < lo || n > hi) {
        RAD_WARN("%s=%s is not a whole number in [%d, %d]; %d is used", k, v, lo, hi, dflt);
        return dflt;
    }
    return (int)n;
}

LinkOpts LinkOpts::from_env() {
    LinkOpts o;
    o.dead_ms = env_int("RADIANCE_PP_DEAD_MS", o.dead_ms, 50, 3600000);
    o.heartbeat_ms = env_int("RADIANCE_PP_HEARTBEAT_MS", o.heartbeat_ms, 5, 60000);
    if (o.heartbeat_ms * 3 > o.dead_ms) o.heartbeat_ms = std::max(5, o.dead_ms / 3);
    o.crc_data = env_int("RADIANCE_PP_CRC_DATA", 0, 0, 1) == 1;
    o.poll_us = env_int("RADIANCE_PP_POLL_US", o.poll_us, 0, 100000);
    o.connect_ms = env_int("RADIANCE_PP_CONNECT_MS", o.connect_ms, 100, 86400000);
    o.stage_ms = env_int("RADIANCE_PP_STAGE_MS", o.stage_ms, 0, 86400000);
    return o;
}

/* ================================================================== sockets */
static uint64_t now_ns() { return rad_mono_ns(); }

static void tune_socket(int fd) {
    const int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    const int buf = 8 << 20;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
    const int fl = fcntl(fd, F_GETFD);
    if (fl >= 0) (void)fcntl(fd, F_SETFD, fl | FD_CLOEXEC);
}

static bool resolve(const std::string& host, int port, sockaddr_storage* out, socklen_t* len,
                    std::string* err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string ps = std::to_string(port);
    const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), ps.c_str(), &hints, &res);
    if (rc != 0 || !res) {
        *err = fmt("cannot resolve %s: %s", host.c_str(), gai_strerror(rc));
        return false;
    }
    std::memcpy(out, res->ai_addr, res->ai_addrlen);
    *len = (socklen_t)res->ai_addrlen;
    freeaddrinfo(res);
    return true;
}

static std::string addr_str(const sockaddr_storage& a) {
    char h[INET6_ADDRSTRLEN] = "?";
    int port = 0;
    if (a.ss_family == AF_INET) {
        const sockaddr_in* s = (const sockaddr_in*)&a;
        inet_ntop(AF_INET, &s->sin_addr, h, sizeof h);
        port = ntohs(s->sin_port);
    } else if (a.ss_family == AF_INET6) {
        const sockaddr_in6* s = (const sockaddr_in6*)&a;
        inet_ntop(AF_INET6, &s->sin6_addr, h, sizeof h);
        port = ntohs(s->sin6_port);
    }
    return fmt("%s:%d", h, port);
}

static std::string ip_str(const sockaddr_storage& a) {
    char h[INET6_ADDRSTRLEN] = "";
    if (a.ss_family == AF_INET) inet_ntop(AF_INET, &((const sockaddr_in*)&a)->sin_addr, h, sizeof h);
    else if (a.ss_family == AF_INET6) inet_ntop(AF_INET6, &((const sockaddr_in6*)&a)->sin6_addr, h, sizeof h);
    return h;
}

/* Every address `host` resolves to, as text. */
static std::vector<std::string> ips_of(const std::string& host) {
    std::vector<std::string> out;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return out;
    for (addrinfo* r = res; r; r = r->ai_next) {
        sockaddr_storage a{};
        std::memcpy(&a, r->ai_addr, std::min<size_t>(r->ai_addrlen, sizeof a));
        const std::string ip = ip_str(a);
        if (!ip.empty() && std::find(out.begin(), out.end(), ip) == out.end()) out.push_back(ip);
    }
    freeaddrinfo(res);
    return out;
}

std::string local_address_for(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    const std::string ps = std::to_string(port > 0 ? port : 9);
    if (host.empty() || getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || !res) return "";
    std::string out;
    const int fd = socket(res->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd >= 0) {
        /* A datagram socket's connect sends nothing: it only picks the route and the source address. */
        if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            sockaddr_storage a{};
            socklen_t l = sizeof a;
            if (getsockname(fd, (sockaddr*)&a, &l) == 0) out = ip_str(a);
        }
        ::close(fd);
    }
    freeaddrinfo(res);
    return out;
}

std::unique_ptr<NodeLink> NodeLink::listen(const std::string& bind_host, int port,
                                           const LinkOpts& o, std::string* err, const std::string& expect) {
    sockaddr_storage a{};
    socklen_t al = 0;
    if (!resolve(bind_host.empty() ? "0.0.0.0" : bind_host, port, &a, &al, err)) return nullptr;
    const int ls = socket(a.ss_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (ls < 0) { *err = fmt("socket: %s", strerror(errno)); return nullptr; }
    const int one = 1;
    (void)setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (sockaddr*)&a, al) != 0) {
        *err = fmt("cannot listen on %s: %s", addr_str(a).c_str(), strerror(errno));
        ::close(ls);
        return nullptr;
    }
    if (::listen(ls, 1) != 0) {
        *err = fmt("listen on %s: %s", addr_str(a).c_str(), strerror(errno));
        ::close(ls);
        return nullptr;
    }
    const std::vector<std::string> allowed = expect.empty() ? std::vector<std::string>() : ips_of(expect);
    if (!expect.empty() && allowed.empty()) {
        *err = fmt("cannot resolve the leader's address %s (--pp-peer)", expect.c_str());
        ::close(ls);
        return nullptr;
    }
    RAD_INFO("node link: waiting on %s for the leader%s%s (up to %d s)", addr_str(a).c_str(),
             expect.empty() ? "" : " from ", expect.c_str(), o.connect_ms / 1000);
    const uint64_t end = now_ns() + (uint64_t)o.connect_ms * 1000000ull;
    int fd = -1;
    while (fd < 0) {
        const uint64_t t = now_ns();
        if (t >= end) {
            *err = fmt("no leader connected to %s within %d s", addr_str(a).c_str(), o.connect_ms / 1000);
            ::close(ls);
            return nullptr;
        }
        pollfd p{ ls, POLLIN, 0 };
        const int left = (int)std::min<uint64_t>((end - t) / 1000000ull + 1, 1000);
        const int pr = poll(&p, 1, left);
        if (pr < 0 && errno != EINTR) { *err = fmt("poll: %s", strerror(errno)); ::close(ls); return nullptr; }
        if (pr <= 0) continue;
        sockaddr_storage pa{};
        socklen_t pl = sizeof pa;
        fd = accept4(ls, (sockaddr*)&pa, &pl, SOCK_CLOEXEC);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            *err = fmt("accept: %s", strerror(errno));
            ::close(ls);
            return nullptr;
        }
        /* Robustness 4: only the leader named by --pp-peer; a probe, a scan or a stale container of
         * another run is turned away and the wait goes on. */
        if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), ip_str(pa)) == allowed.end()) {
            RAD_WARN("node link: a connection from %s is not the leader (--pp-peer %s): closed, still waiting",
                     addr_str(pa).c_str(), expect.c_str());
            ::close(fd);
            fd = -1;
            continue;
        }
        RAD_INFO("node link: the leader connected from %s", addr_str(pa).c_str());
    }
    ::close(ls);
    tune_socket(fd);
    return adopt(fd, o);
}

std::unique_ptr<NodeLink> NodeLink::connect(const std::string& host, int port, const LinkOpts& o,
                                            std::string* err) {
    sockaddr_storage a{};
    socklen_t al = 0;
    if (!resolve(host, port, &a, &al, err)) return nullptr;
    const uint64_t end = now_ns() + (uint64_t)o.connect_ms * 1000000ull;
    int tries = 0;
    std::string last;
    for (;;) {
        const int fd = socket(a.ss_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) { *err = fmt("socket: %s", strerror(errno)); return nullptr; }
        if (::connect(fd, (sockaddr*)&a, al) == 0) {
            tune_socket(fd);
            RAD_INFO("node link: connected to the follower at %s after %d attempt(s)",
                     addr_str(a).c_str(), tries + 1);
            return adopt(fd, o);
        }
        last = strerror(errno);
        ::close(fd);
        ++tries;
        if (now_ns() >= end) {
            *err = fmt("the follower at %s did not answer within %d s (%d attempts, last: %s)",
                       addr_str(a).c_str(), o.connect_ms / 1000, tries, last.c_str());
            return nullptr;
        }
        if (tries == 1 || tries % 50 == 0)
            RAD_INFO("node link: the follower at %s is not listening yet (%s); retrying",
                     addr_str(a).c_str(), last.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

std::unique_ptr<NodeLink> NodeLink::adopt(int fd, const LinkOpts& o) {
    return std::unique_ptr<NodeLink>(new NodeLink(fd, o));
}

NodeLink::NodeLink(int fd, const LinkOpts& o) : fd_(fd), o_(o) {
    last_rx_ns_ = last_tx_ns_ = now_ns();
    /* Robustness 1(c): unacknowledged data to a host that is gone, or a zero window that never
     * opens, closes the connection a second after dead_ms (the writer's own check comes first). Not a
     * TCP socket (a test's socket pair): refused, harmless. */
    const unsigned int ut = (unsigned int)o_.dead_ms + 1000u;
    (void)setsockopt(fd_, IPPROTO_TCP, TCP_USER_TIMEOUT, &ut, sizeof ut);
}

NodeLink::~NodeLink() {
    stopping_.store(true, std::memory_order_release);
    if (dp_) dp_->stop();
    if (fd_ >= 0) (void)shutdown(fd_, SHUT_RDWR);
    q_cv_.notify_all();
    if (rx_.joinable()) rx_.join();
    if (proxy_.joinable()) proxy_.join();
    if (hb_.joinable()) hb_.join();
    if (fd_ >= 0) ::close(fd_);
}

void NodeLink::attach(RadPpRing* const send[RAD_PP_N_RINGS], RadPpRing* const recv[RAD_PP_N_RINGS]) {
    attach_n(send, recv, RAD_PP_N_RINGS);
}

void NodeLink::attach_n(RadPpRing* const* send, RadPpRing* const* recv, int n) {
    n_ix_ = std::max((int)RAD_PP_N_RINGS, std::min(n, (int)RAD_PP_MAX_RING_IX));
    for (int i = 0; i < RAD_PP_MAX_RING_IX; ++i) {
        send_[i] = send && i < n ? send[i] : nullptr;
        recv_[i] = recv && i < n ? recv[i] : nullptr;
        sent_[i] = send_[i] ? send_[i]->ready : 0;
    }
}

void NodeLink::set_data_path(std::unique_ptr<DataPath> dp) { dp_ = std::move(dp); }

void NodeLink::start() {
    if (started_.exchange(true)) return;
    last_rx_ns_ = last_tx_ns_ = now_ns();
    rx_ = std::thread([this] { rx_loop(); });
    hb_ = std::thread([this] { hb_loop(); });
    bool any = false;
    for (int i = 0; i < RAD_PP_MAX_RING_IX; ++i) any = any || send_[i];
    if (any) proxy_ = std::thread([this] { proxy_loop(); });
}

std::string NodeLink::why() const {
    std::lock_guard<std::mutex> lk(why_mu_);
    return why_;
}

NodeLink::Stats NodeLink::stats() const {
    std::lock_guard<std::mutex> lk(st_mu_);
    return st_;
}

void NodeLink::fail(const std::string& why, bool from_peer) {
    {
        std::lock_guard<std::mutex> lk(why_mu_);
        if (!why_.empty()) return;
        why_ = why;
    }
    down_.store(true, std::memory_order_release);
    for (int i = 0; i < RAD_PP_MAX_RING_IX; ++i) {
        if (send_[i]) __atomic_store_n(&send_[i]->abort, 1ull, __ATOMIC_RELEASE);
        if (recv_[i]) __atomic_store_n(&recv_[i]->abort, 1ull, __ATOMIC_RELEASE);
    }
    /* Tell the peer, unless it told us or a writer holds the socket (it then fails on the
     * shutdown below, and the peer sees the connection close). The frame goes out whole or not at
     * all within 100 ms (robustness 6: a partial one loses the reason). */
    if (!from_peer && !stopping_.load(std::memory_order_acquire) && fd_ >= 0)
        write_bounded(kAbort, why, 0, 100);
    if (fd_ >= 0) (void)shutdown(fd_, SHUT_RDWR);
    if (dp_) dp_->stop();
    q_cv_.notify_all();
    if (stopping_.load(std::memory_order_acquire)) return;
    if (peer_bye_.load(std::memory_order_acquire)) RAD_INFO("node link: closed: %s", why.c_str());
    else                                           RAD_ERR("node link: down: %s", why.c_str());
}

void NodeLink::close(const char* reason) {
    if (!stopping_.exchange(true)) {
        /* Robustness 1(d): the kBye never waits behind a stuck writer: the socket is taken if free
         * within 200 ms and the frame written within 500 ms, else the peer sees the close. */
        if (alive() && fd_ >= 0) write_bounded(kBye, reason, 200, 500);
        {
            std::lock_guard<std::mutex> lk(why_mu_);
            if (why_.empty()) why_ = reason;
        }
        down_.store(true, std::memory_order_release);
        for (int i = 0; i < RAD_PP_MAX_RING_IX; ++i) {
            if (send_[i]) __atomic_store_n(&send_[i]->abort, 1ull, __ATOMIC_RELEASE);
            if (recv_[i]) __atomic_store_n(&recv_[i]->abort, 1ull, __ATOMIC_RELEASE);
        }
        if (dp_) dp_->stop();
        if (fd_ >= 0) (void)shutdown(fd_, SHUT_RDWR);
        q_cv_.notify_all();
    }
    if (rx_.joinable()) rx_.join();
    if (proxy_.joinable()) proxy_.join();
    if (hb_.joinable()) hb_.join();
}

/* ------------------------------------------------------------------ writing */
int NodeLink::write_frame(WireHdr& h, const void* a, int64_t na, const void* b, int64_t nb, bool try_only) {
    if (na < 0 || nb < 0) return RAD_E_INVAL;
    h.bytes = (uint64_t)(na + nb);
    if (h.flags & kFlagCrc) {
        uint32_t c = crc32c(a, (size_t)na);
        if (nb > 0) c = crc32c(b, (size_t)nb, c);
        h.crc_payload = c;
    }
    seal(&h);
    std::unique_lock<std::mutex> lk(tx_mu_, std::defer_lock);
    if (try_only) {
        if (!lk.try_lock()) return RAD_E_FULL;
    } else {
        lk.lock();
    }
    if (down_.load(std::memory_order_acquire) && h.kind != kBye) return RAD_E_IO;
    if (tx_torn_.load(std::memory_order_acquire)) return RAD_E_IO;
    iovec v[3] = { { &h, sizeof h }, { const_cast<void*>(a), (size_t)na },
                   { const_cast<void*>(b), (size_t)nb } };
    int iv = 0, niv = 1 + (na > 0 ? 1 : 0) + (nb > 0 ? 1 : 0);
    if (na <= 0 && nb > 0) v[1] = v[2];
    const int64_t total = (int64_t)sizeof h + na + nb;
    int64_t left = total;
    /* Robustness 1(a): never block in the socket. A peer that stopped reading but keeps its
     * connection (a frozen process, a paused container, a host that hangs) fills both socket
     * buffers (~16 MiB on gpu02 / gpu03) and then accepts nothing: after dead_ms without a byte
     * accepted the link goes down here, whatever the frame's size. */
    const uint64_t stall_ns = (uint64_t)o_.dead_ms * 1000000ull;
    uint64_t moved = now_ns();
    tx_progress_ns_.store(moved, std::memory_order_relaxed);
    tx_busy_.store(true, std::memory_order_release);
    auto give_up = [&](const std::string& why) {
        if (left < total) tx_torn_.store(true, std::memory_order_release);
        tx_busy_.store(false, std::memory_order_release);
        lk.unlock();
        if (h.kind != kBye) fail(why);
        return RAD_E_IO;
    };
    while (left > 0) {
        msghdr mh{};
        mh.msg_iov = v + iv;
        mh.msg_iovlen = (size_t)(niv - iv);
        const ssize_t n = sendmsg(fd_, &mh, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                const uint64_t t = now_ns();
                if (t - moved > stall_ns)
                    return give_up(fmt("sending a %s frame: the peer accepted no byte for %d ms with %lld of %lld "
                                       "bytes left (dead after %d ms): it stopped reading", frame_name(h.kind),
                                       (int)((t - moved) / 1000000ull), (long long)left, (long long)total,
                                       o_.dead_ms));
                if (stopping_.load(std::memory_order_acquire) || (down_.load(std::memory_order_acquire) && h.kind != kBye))
                    return give_up("the link went down during a send");
                pollfd pf{ fd_, POLLOUT, 0 };
                (void)poll(&pf, 1, 20);
                continue;
            }
            return give_up(fmt("sending a %s frame: %s", frame_name(h.kind), strerror(errno)));
        }
        left -= n;
        size_t k = (size_t)n;
        while (iv < niv && k >= v[iv].iov_len) { k -= v[iv].iov_len; ++iv; }
        if (iv < niv) { v[iv].iov_base = (char*)v[iv].iov_base + k; v[iv].iov_len -= k; }
        moved = now_ns();
        last_tx_ns_.store(moved, std::memory_order_relaxed);
        tx_progress_ns_.store(moved, std::memory_order_relaxed);
    }
    tx_busy_.store(false, std::memory_order_release);
    lk.unlock();
    std::lock_guard<std::mutex> sl(st_mu_);
    ++st_.frames_out;
    st_.bytes_out += sizeof h + (uint64_t)(na + nb);
    if (h.kind == kData) ++st_.data_out;
    return RAD_OK;
}

void NodeLink::write_bounded(uint16_t kind, const std::string& body, int lock_ms, int send_ms) {
    std::unique_lock<std::mutex> lk(tx_mu_, std::defer_lock);
    const uint64_t t0 = now_ns();
    while (!lk.try_lock()) {
        if (now_ns() - t0 >= (uint64_t)lock_ms * 1000000ull) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (tx_torn_.load(std::memory_order_acquire)) return;   /* mid-frame: a frame now would be garbage */
    WireHdr h{};
    h.kind = kind;
    h.bytes = body.size();
    h.flags = kFlagCrc;
    h.crc_payload = crc32c(body.data(), body.size());
    seal(&h);
    std::string fr((const char*)&h, sizeof h);
    fr += body;
    size_t off = 0;
    const uint64_t t1 = now_ns();
    while (off < fr.size()) {
        const ssize_t n = ::send(fd_, fr.data() + off, fr.size() - off, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) { off += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (now_ns() - t1 >= (uint64_t)send_ms * 1000000ull) break;
            pollfd pf{ fd_, POLLOUT, 0 };
            (void)poll(&pf, 1, 10);
            continue;
        }
        break;
    }
    if (off > 0 && off < fr.size()) tx_torn_.store(true, std::memory_order_release);
    if (off > 0) last_tx_ns_.store(now_ns(), std::memory_order_relaxed);
}

int NodeLink::send(uint16_t kind, uint64_t seq, uint64_t aux, const void* p, int64_t n) {
    return send2(kind, seq, aux, p, n, nullptr, 0);
}

int NodeLink::send2(uint16_t kind, uint64_t seq, uint64_t aux, const void* a, int64_t na,
                    const void* b, int64_t nb) {
    if (!alive()) return RAD_E_IO;
    WireHdr h{};
    h.kind = kind;
    h.seq = seq;
    h.aux = aux;
    h.flags = kFlagCrc;
    return write_frame(h, a, na, b, nb);
}

int NodeLink::send_data(int id, RadPpRing* r) {
    const uint64_t seq = __atomic_load_n(&r->ready, __ATOMIC_ACQUIRE);
    const uint64_t bytes = r->bytes, rows = r->rows, parts = r->parts;
    if (bytes > r->cap) {
        fail(fmt("crossing %d: stage_send published %llu bytes in a ring of %llu", id,
                 (unsigned long long)bytes, (unsigned long long)r->cap));
        return RAD_E_FULL;
    }
    if (dp_) {
        const int st = dp_->post(id, r, seq, rows, bytes, parts);
        if (st < 0) { fail(fmt("crossing %d: the verbs path refused payload %llu", id, (unsigned long long)seq)); return st; }
        std::lock_guard<std::mutex> sl(st_mu_);
        ++st_.data_out;
        st_.bytes_out += bytes;
        return RAD_OK;
    }
    WireHdr h{};
    h.kind = kData;
    h.lane = (uint32_t)id;
    h.seq = seq;
    h.aux = rows;
    h.aux2 = parts;
    h.flags = o_.crc_data ? (uint32_t)kFlagCrc : 0u;
    return write_frame(h, (const char*)r + RAD_PP_RING_HDR, (int64_t)bytes, nullptr, 0);
}

/* ------------------------------------------------------------------ reading */
/* `idle_ms`: how long the read may go without a byte before it fails (< 0: until the link
 * stops). */
int NodeLink::read_exact(void* p, int64_t n, int idle_ms) {
    char* c = static_cast<char*>(p);
    uint64_t idle_from = now_ns();
    while (n > 0) {
        if (stopping_.load(std::memory_order_acquire)) return RAD_E_CANCELLED;
        pollfd pf{ fd_, POLLIN, 0 };
        const int pr = poll(&pf, 1, 50);
        if (pr < 0) {
            if (errno == EINTR) continue;
            return RAD_E_IO;
        }
        if (pr == 0) {
            if (idle_ms >= 0 && now_ns() - idle_from > (uint64_t)idle_ms * 1000000ull)
                return RAD_E_NOTFOUND;
            continue;
        }
        const ssize_t k = ::read(fd_, c, (size_t)std::min<int64_t>(n, 1 << 24));
        if (k < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return RAD_E_IO;
        }
        if (k == 0) return RAD_E_IO;   /* the peer closed */
        c += k;
        n -= k;
        idle_from = now_ns();
        last_rx_ns_.store(idle_from, std::memory_order_relaxed);
        std::lock_guard<std::mutex> sl(st_mu_);
        st_.bytes_in += (uint64_t)k;
    }
    return RAD_OK;
}

int NodeLink::land(int id, uint64_t seq, uint64_t rows, uint64_t bytes, uint64_t parts) {
    RadPpRing* r = recv_ring(id);
    if (!r) return RAD_E_STATE;
    r->rows = rows;
    r->bytes = bytes;
    r->parts = parts;
    __atomic_store_n(&r->landed, seq, __ATOMIC_RELEASE);
    {
        std::lock_guard<std::mutex> sl(st_mu_);
        ++st_.data_in;
    }
    if (on_landed) on_landed(id, seq);
    return RAD_OK;
}

void NodeLink::rx_loop() {
    std::vector<uint8_t> drop;
    while (!stopping_.load(std::memory_order_acquire) && alive()) {
        WireHdr h{};
        /* Between frames nothing times out here: the heartbeat thread watches the silence. */
        int st = read_exact(&h, sizeof h, -1);
        if (st == RAD_E_CANCELLED) return;
        if (st < 0) {
            fail(st == RAD_E_IO ? "the peer closed the connection or it broke (no frame header)"
                                : "reading a frame header failed");
            return;
        }
        const std::string bad = check_header(h, kMaxFrameBytes);
        if (!bad.empty()) { fail(bad); return; }
        {
            std::lock_guard<std::mutex> sl(st_mu_);
            ++st_.frames_in;
        }
        if (h.kind == kHeartbeat) {
            std::lock_guard<std::mutex> sl(st_mu_);
            ++st_.heartbeats_in;
            continue;
        }
        if (h.kind == kData) {
            RadPpRing* r = recv_ring((int)h.lane);
            if (!r) { fail(fmt("a payload for crossing %u, which this node does not receive", h.lane)); return; }
            if (h.bytes > r->cap) {
                fail(fmt("crossing %u: a payload of %llu bytes for a ring of %llu", h.lane,
                         (unsigned long long)h.bytes, (unsigned long long)r->cap));
                return;
            }
            const uint64_t landed = __atomic_load_n(&r->landed, __ATOMIC_ACQUIRE);
            if (h.seq != landed + 1) {
                fail(fmt("crossing %u: payload %llu arrived after %llu: the two nodes are out of step",
                         h.lane, (unsigned long long)h.seq, (unsigned long long)landed));
                return;
            }
            /* One slot: every local reader must have taken the last payload before this one is
             * written over it. The schedule guarantees it (one step in flight a lane); a wait
             * here past dead_ms is a schedule bug and fails loudly instead of tearing a payload. */
            const uint64_t t0 = now_ns();
            for (;;) {
                bool all = true;
                for (uint64_t k = 0; k < r->n_ranks && k < RAD_PP_MAX_RANKS; ++k)
                    all = all && __atomic_load_n(&r->taken[k], __ATOMIC_ACQUIRE) >= landed;
                if (all) break;
                if (now_ns() - t0 > (uint64_t)o_.dead_ms * 1000000ull) {
                    fail(fmt("crossing %u: payload %llu arrived while a local rank had not taken "
                             "payload %llu (a ring overrun)", h.lane, (unsigned long long)h.seq,
                             (unsigned long long)landed));
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            char* dst = (char*)r + RAD_PP_RING_HDR;
            st = read_exact(dst, (int64_t)h.bytes, o_.dead_ms);
            if (st == RAD_E_CANCELLED) return;
            if (st < 0) { fail(fmt("crossing %u: the payload stopped mid-frame", h.lane)); return; }
            if ((h.flags & kFlagCrc) && crc32c(dst, (size_t)h.bytes) != h.crc_payload) {
                fail(fmt("crossing %u: payload %llu fails its CRC", h.lane, (unsigned long long)h.seq));
                return;
            }
            (void)land((int)h.lane, h.seq, h.aux, h.bytes, h.aux2);
            continue;
        }
        Msg m;
        m.h = h;
        m.body.resize((size_t)h.bytes);
        st = read_exact(m.body.data(), (int64_t)h.bytes, o_.dead_ms);
        if (st == RAD_E_CANCELLED) return;
        if (st < 0) { fail(fmt("a %s frame stopped mid-payload", frame_name(h.kind))); return; }
        if ((h.flags & kFlagCrc) && crc32c(m.body.data(), m.body.size()) != h.crc_payload) {
            fail(fmt("a %s frame fails its CRC", frame_name(h.kind)));
            return;
        }
        if (h.kind == kAbort) {
            fail("the peer went down: " + std::string(m.body.begin(), m.body.end()), true);
            return;
        }
        if (h.kind == kBye) {
            peer_bye_.store(true, std::memory_order_release);
            fail("the peer stopped: " + std::string(m.body.begin(), m.body.end()), true);
            return;
        }
        if (on_control && on_control(m)) continue;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            q_.push_back(std::move(m));
        }
        q_cv_.notify_all();
    }
}

int NodeLink::recv(Msg* m, int timeout_ms) {
    if (!started_.load(std::memory_order_acquire)) {
        /* Before the threads run (the rendezvous): read the socket here, skipping heartbeats. */
        for (;;) {
            if (!alive()) return RAD_E_IO;
            WireHdr h{};
            int st = read_exact(&h, sizeof h, timeout_ms);
            if (st == RAD_E_NOTFOUND) return RAD_E_NOTFOUND;
            if (st < 0) { fail("the peer closed the connection or it broke"); return RAD_E_IO; }
            const std::string bad = check_header(h, kMaxFrameBytes);
            if (!bad.empty()) { fail(bad); return RAD_E_IO; }
            m->h = h;
            m->body.resize((size_t)h.bytes);
            st = read_exact(m->body.data(), (int64_t)h.bytes, o_.dead_ms);
            if (st < 0) { fail(fmt("a %s frame stopped mid-payload", frame_name(h.kind))); return RAD_E_IO; }
            if ((h.flags & kFlagCrc) && crc32c(m->body.data(), m->body.size()) != h.crc_payload) {
                fail(fmt("a %s frame fails its CRC", frame_name(h.kind)));
                return RAD_E_IO;
            }
            if (h.kind == kHeartbeat) continue;
            if (h.kind == kAbort || h.kind == kBye) {
                if (h.kind == kBye) peer_bye_.store(true, std::memory_order_release);
                fail(std::string(h.kind == kAbort ? "the peer went down: " : "the peer stopped: ") +
                         std::string(m->body.begin(), m->body.end()), true);
                return RAD_E_IO;
            }
            if (h.kind == kData) { fail("a crossing's payload before the link was started"); return RAD_E_IO; }
            std::lock_guard<std::mutex> sl(st_mu_);
            ++st_.frames_in;
            return RAD_OK;
        }
    }
    std::unique_lock<std::mutex> lk(q_mu_);
    auto ready = [&] { return !q_.empty() || !alive(); };
    if (timeout_ms < 0) q_cv_.wait(lk, ready);
    else if (!q_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready)) return RAD_E_NOTFOUND;
    if (q_.empty()) return RAD_E_IO;
    *m = std::move(q_.front());
    q_.pop_front();
    return RAD_OK;
}

/* ------------------------------------------------------------------ the proxy and the heartbeat */
void NodeLink::proxy_loop() {
    uint32_t idle = 0;
    while (!stopping_.load(std::memory_order_acquire) && alive()) {
        stage_watch();
        bool moved = false;
        for (int id = 0; id < n_ix_; ++id) {
            RadPpRing* r = send_[id];
            if (!r) continue;
            const uint64_t rd = __atomic_load_n(&r->ready, __ATOMIC_ACQUIRE);
            if (rd == sent_[id]) continue;
            if (rd != sent_[id] + 1) {
                fail(fmt("crossing %d: stage_send published payload %llu after %llu without the "
                         "link sending the one between (one slot a ring)", id,
                         (unsigned long long)rd, (unsigned long long)sent_[id]));
                return;
            }
            /* A stage op on this node that gave up (its deadline, a payload not of its step) set
             * its ring's abort word before this payload was published, in stream order: the pass
             * computed nothing to send, so the link goes down instead. */
            for (int k = 0; k < n_ix_; ++k)
                for (RadPpRing* q : { send_[k], recv_[k] })
                    if (q && __atomic_load_n(&q->abort, __ATOMIC_ACQUIRE)) {
                        fail(fmt("crossing %d's stage op on this node gave up (abort word %llu: 3 a "
                                 "deadline, 4 a payload of another step, 5 a stale payload): payload %llu "
                                 "of crossing %d is not sent", k, (unsigned long long)q->abort,
                                 (unsigned long long)rd, id));
                        return;
                    }
            if (send_data(id, r) < 0) return;
            sent_[id] = rd;
            /* The ring is free again: stage_send waits for this before it writes the next one. */
            __atomic_store_n(&r->sent, rd, __ATOMIC_RELEASE);
            moved = true;
        }
        if (moved) { idle = 0; continue; }
        if (o_.poll_us <= 0 || ++idle < 64) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            continue;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(o_.poll_us));
    }
}

void NodeLink::hb_loop() {
    const uint64_t hb = (uint64_t)o_.heartbeat_ms * 1000000ull;
    const uint64_t dead = (uint64_t)o_.dead_ms * 1000000ull;
    while (!stopping_.load(std::memory_order_acquire) && alive()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, o_.heartbeat_ms / 2)));
        const uint64_t t = now_ns();
        /* Robustness 1(b): the silence check first, never behind the socket's lock. */
        const uint64_t rx = last_rx_ns_.load(std::memory_order_relaxed);
        if (t > rx && t - rx > dead) {
            fail(fmt("no frame from the peer for %llu ms (heartbeat every %d ms, dead after %d ms): "
                     "the peer process or the link is gone", (unsigned long long)((t - rx) / 1000000ull),
                     o_.heartbeat_ms, o_.dead_ms));
            return;
        }
        /* A writer that moved no byte for dead_ms (+ 1 s slack: it checks itself first). */
        const uint64_t tp = tx_progress_ns_.load(std::memory_order_relaxed);
        if (tx_busy_.load(std::memory_order_acquire) && t > tp && t - tp > dead + 1000000000ull) {
            fail(fmt("a frame write moved no byte for %llu ms: the peer stopped reading",
                     (unsigned long long)((t - tp) / 1000000ull)));
            return;
        }
        if (t - last_tx_ns_.load(std::memory_order_relaxed) >= hb) {
            WireHdr h{};
            h.kind = kHeartbeat;
            /* Only when the socket is free: a writer holding it sends bytes, which are this node's
             * heartbeat to the peer's receive thread. */
            if (write_frame(h, nullptr, 0, nullptr, 0, true) == RAD_E_IO) return;
        }
    }
}

/* Robustness 2: this node's own stage owes a crossing payload -- the follower an X2 for an X1 that
 * landed, the leader an X1 for a step it released -- for longer than stage_ms: its cards (a kernel
 * that never ends) or its host are stuck while the heartbeats still flow. The link goes down
 * naming it, instead of the peer's cards waiting out the rings' device deadline (600 s). */
void NodeLink::stage_watch() {
    if (o_.stage_ms <= 0) return;
    const uint64_t t = now_ns();
    if (t - watch_ns_ < 10000000ull) return;   /* every 10 ms */
    watch_ns_ = t;
    uint64_t due = 0, done = 0;
    const char* what = nullptr;
    /* tp8 S5: summed over the lanes' rings -- each lane step owes one payload of each crossing */
    if (send_[RAD_PP_X2] && recv_[RAD_PP_X1]) {
        for (int ix = 0; ix + 1 < n_ix_; ix += RAD_PP_N_RINGS) {
            if (recv_[ix + RAD_PP_X1]) due += __atomic_load_n(&recv_[ix + RAD_PP_X1]->landed, __ATOMIC_ACQUIRE);
            if (send_[ix + RAD_PP_X2]) done += __atomic_load_n(&send_[ix + RAD_PP_X2]->ready, __ATOMIC_ACQUIRE);
        }
        what = "X2 for an X1 that landed";
    } else if (send_[RAD_PP_X1]) {
        due = released_.load(std::memory_order_acquire);
        for (int ix = 0; ix + 1 < n_ix_; ix += RAD_PP_N_RINGS)
            if (send_[ix + RAD_PP_X1]) done += __atomic_load_n(&send_[ix + RAD_PP_X1]->ready, __ATOMIC_ACQUIRE);
        what = "X1 for a step it released";
    } else {
        return;
    }
    const uint64_t want = done + 1;
    if (due < want) { due_k_ = 0; return; }
    if (due_k_ != want) { due_k_ = want; due_ns_ = t; return; }
    if (t - due_ns_ > (uint64_t)o_.stage_ms * 1000000ull)
        fail(fmt("this node's stage has owed its %s (payload %llu) for %llu ms, past RADIANCE_PP_STAGE_MS=%d: "
                 "its cards or its host are stuck", what, (unsigned long long)want,
                 (unsigned long long)((t - due_ns_) / 1000000ull), o_.stage_ms));
}

/* ================================================================== rings */
void ring_init(RadPpRing* r, int id, uint64_t cap, int n_ranks, uint64_t deadline_ms) {
    std::memset((void*)r, 0, RAD_PP_RING_HDR);
    r->cap = cap;
    r->n_ranks = (uint64_t)std::max(0, std::min(n_ranks, RAD_PP_MAX_RANKS));
    r->deadline_ms = deadline_ms;
    r->id = (uint64_t)id;
    __atomic_store_n(&r->magic, RAD_PP_RING_MAGIC, __ATOMIC_RELEASE);
}

int ring_put(RadPpRing* r, const void* p, uint64_t bytes, uint64_t rows) {
    if (!r || r->magic != RAD_PP_RING_MAGIC) return RAD_E_STATE;
    if (rad_pp_align(bytes) + RAD_PP_TRAILER > r->cap) return RAD_E_FULL;
    /* One slot: the link must have sent the last payload before this one overwrites it. */
    const uint64_t t0 = now_ns();
    while (__atomic_load_n(&r->sent, __ATOMIC_ACQUIRE) < r->ready) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        if (r->deadline_ms && now_ns() - t0 > r->deadline_ms * 1000000ull) return RAD_E_NOTFOUND;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
    char* const dst = (char*)r + RAD_PP_RING_HDR;
    std::memcpy(dst, p, (size_t)bytes);
    /* review F6: the payload's number in the trailer at its aligned end, as the stage ops write it */
    const uint64_t at = rad_pp_align(bytes), seq = r->ready + 1;
    std::memset(dst + at, 0, RAD_PP_TRAILER);
    __atomic_store_n(reinterpret_cast<uint64_t*>(dst + at), seq, __ATOMIC_RELEASE);
    r->rows = rows;
    r->bytes = at + RAD_PP_TRAILER;
    r->parts = 1;
    r->part_bytes[0] = bytes;
    __atomic_store_n(&r->ready, seq, __ATOMIC_RELEASE);
    return RAD_OK;
}

int ring_take(RadPpRing* r, int rank, void* p, uint64_t bytes, uint64_t rows) {
    if (!r || r->magic != RAD_PP_RING_MAGIC || rank < 0 || rank >= RAD_PP_MAX_RANKS) return RAD_E_STATE;
    const uint64_t want = __atomic_load_n(&r->taken[rank], __ATOMIC_ACQUIRE) + 1;
    const uint64_t t0 = now_ns();
    const uint64_t dl = r->deadline_ms ? r->deadline_ms * 1000000ull : ~0ull;
    uint32_t spins = 0;
    while (__atomic_load_n(&r->landed, __ATOMIC_ACQUIRE) < want) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        if (now_ns() - t0 > dl) return RAD_E_NOTFOUND;
        if (++spins > 256) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    const uint64_t at = rad_pp_align(bytes);
    if (r->rows != rows || r->bytes != at + RAD_PP_TRAILER) return RAD_E_SHAPE;
    const char* const src = (const char*)r + RAD_PP_RING_HDR;
    if (__atomic_load_n(reinterpret_cast<const uint64_t*>(src + at), __ATOMIC_ACQUIRE) != want) return RAD_E_STATE;
    std::memcpy(p, src, (size_t)bytes);
    __atomic_store_n(&r->taken[rank], want, __ATOMIC_RELEASE);
    return RAD_OK;
}

}  /* namespace rad::net */
