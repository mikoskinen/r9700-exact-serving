/* tests/node_link_test.cpp -- the two-node link (core/net/node_link.h, tp8 S4) on the host: frames,
 * CRCs, the TCP transport over socket pairs and localhost, the crossings' rings, and every way the
 * link must go down cleanly.
 *
 *   crc        CRC-32C's check value, the SSE4.2 and table forms equal, chaining
 *   frames     control frames round-trip; a damaged header or payload, a wrong magic or version, a
 *              length past the bound, a truncated frame: each refused with a reason, never accepted
 *   fuzz       5,000 frames with one random byte flipped (header or CRC'd payload) and 2,000 random
 *              byte strings: the receiver refuses every one and never crashes or hangs
 *   crossings  X1 / X2 over started links, 2 readers a receive ring, 1,000 / 500 rounds of random sizes up to
 *              256 KiB and one of 64 MiB, every byte checked; TCP over localhost (listen / connect)
 *   failure    the peer killed (SIGKILL) or frozen (SIGSTOP: heartbeat loss) mid-run: the waiting reader
 *              returns RAD_E_IO and why() names it, within dead_ms; a clean close says so; an overrun,
 *              a skipped sequence and a payload for a ring this node does not hold take the link down */
#include "rad_test.h"

#include "net/node_link.h"

#include <signal.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace rad::net;

struct Pair {
    std::unique_ptr<NodeLink> a, b;
};

Pair socket_pair(const LinkOpts& o = LinkOpts{}) {
    int sv[2] = { -1, -1 };
    Pair p;
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) return p;
    p.a = NodeLink::adopt(sv[0], o);
    p.b = NodeLink::adopt(sv[1], o);
    return p;
}

/* A ring in plain host memory (the host backend's pinned memory is plain memory too), sized as the
 * engine sizes one: `cap` payload bytes + a part's alignment + the payload's trailer. */
struct Ring {
    std::vector<uint64_t> mem;
    RadPpRing* r = nullptr;
    Ring(int id, uint64_t cap, int readers, uint64_t deadline_ms = 10000) {
        const uint64_t room = cap + RAD_PP_PART_ALIGN + RAD_PP_TRAILER;
        mem.assign((RAD_PP_RING_HDR + room) / 8 + 1, 0);
        r = reinterpret_cast<RadPpRing*>(mem.data());
        ring_init(r, id, room, readers, deadline_ms);
    }
};

LinkOpts fast_opts(int dead_ms = 400) {
    LinkOpts o;
    o.dead_ms = dead_ms;
    o.heartbeat_ms = std::max(5, dead_ms / 8);
    o.poll_us = 10;
    o.connect_ms = 10000;
    return o;
}

/* Seeded bytes, eight at a time (splitmix64), so the large rounds stay cheap. */
void fill(std::vector<uint8_t>& v, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
    size_t i = 0;
    for (; i + 8 <= v.size(); i += 8) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        std::memcpy(v.data() + i, &z, 8);
    }
    for (; i < v.size(); ++i) v[i] = (uint8_t)(x >> (8 * (i & 7)));
}

/* ------------------------------------------------------------------ crc */
TEST(crc32c_has_its_check_value_and_both_forms_agree) {
    CHECK_EQ(crc32c("123456789", 9), 0xE3069283u);
    CHECK_EQ(crc32c("", 0), 0u);
    std::vector<uint8_t> v(70001);
    fill(v, 7);
    /* Chained over any split equals one pass. */
    const uint32_t whole = crc32c(v.data(), v.size());
    for (size_t cut : { (size_t)0, (size_t)1, (size_t)7, (size_t)8, (size_t)4097, v.size() - 3 })
        CHECK_EQ(crc32c(v.data() + cut, v.size() - cut, crc32c(v.data(), cut)), whole);
}

/* ------------------------------------------------------------------ frames */
TEST(a_control_frame_round_trips_before_and_after_start) {
    Pair p = socket_pair(fast_opts());
    REQUIRE(p.a && p.b);
    const std::string body = "the follower's facts";
    CHECK_OK(p.a->send(kHello, 7, 9, body.data(), (int64_t)body.size()));
    Msg m;
    CHECK_OK(p.b->recv(&m, 2000));
    CHECK_EQ(m.h.kind, kHello);
    CHECK_EQ(m.h.seq, 7u);
    CHECK_EQ(m.h.aux, 9u);
    CHECK(std::string(m.body.begin(), m.body.end()) == body);
    /* Two pieces arrive as one payload. */
    CHECK_OK(p.b->send2(kStep, 1, 2, "abc", 3, "defg", 4));
    CHECK_OK(p.a->recv(&m, 2000));
    CHECK(std::string(m.body.begin(), m.body.end()) == "abcdefg");
    /* Started: the same through the receive thread's queue, heartbeats in between. */
    p.a->start();
    p.b->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    CHECK_OK(p.a->send(kUser, 3, 0, "x", 1));
    CHECK_OK(p.b->recv(&m, 2000));
    CHECK_EQ(m.h.kind, kUser);
    CHECK(p.b->stats().heartbeats_in > 0);
    /* Nothing more: a timeout, not a failure. */
    CHECK_EQ(p.b->recv(&m, 50), RAD_E_NOTFOUND);
    CHECK(p.a->alive() && p.b->alive());
}

/* A raw frame onto `fd`: a sealed header, then the payload. */
void raw_frame(int fd, WireHdr h, const std::string& body, bool crc = true) {
    h.bytes = body.size();
    if (crc) { h.flags |= kFlagCrc; h.crc_payload = crc32c(body.data(), body.size()); }
    seal(&h);
    std::string w((const char*)&h, sizeof h);
    w += body;
    (void)!write(fd, w.data(), w.size());
}

/* Starts a receiver on one end and returns the other end's raw fd. */
struct Raw {
    int fd = -1;
    std::unique_ptr<NodeLink> rx;
    explicit Raw(const LinkOpts& o = fast_opts()) {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) return;
        fd = sv[0];
        rx = NodeLink::adopt(sv[1], o);
    }
    ~Raw() { if (fd >= 0) close(fd); }
};

TEST(a_damaged_or_foreign_frame_is_refused_with_a_reason) {
    struct Case { const char* what; std::function<void(int)> write; const char* in_why; };
    const Case cases[] = {
        { "wrong magic", [](int fd) { WireHdr h{}; h.kind = kHello; seal(&h); h.magic = 0x12345678;
                                     (void)!write(fd, &h, sizeof h); }, "magic" },
        { "header bit flip", [](int fd) { WireHdr h{}; h.kind = kHello; seal(&h); h.seq ^= 4;
                                         (void)!write(fd, &h, sizeof h); }, "CRC" },
        { "version", [](int fd) { WireHdr h{}; h.kind = kHello; h.magic = kWireMagic; h.version = 9;
                                 h.crc_hdr = crc32c(&h, 60); (void)!write(fd, &h, sizeof h); }, "version" },
        { "length past the bound", [](int fd) { WireHdr h{}; h.kind = kHello; h.bytes = (uint64_t)kMaxFrameBytes + 1;
                                               seal(&h); (void)!write(fd, &h, sizeof h); }, "past" },
        { "payload CRC", [](int fd) { WireHdr h{}; h.kind = kHello; h.bytes = 4; h.flags = kFlagCrc;
                                     h.crc_payload = crc32c("abcd", 4); seal(&h);
                                     (void)!write(fd, &h, sizeof h); (void)!write(fd, "abce", 4); }, "CRC" },
        { "truncated payload", [](int fd) { WireHdr h{}; h.kind = kHello; h.bytes = 100; seal(&h);
                                           (void)!write(fd, &h, sizeof h); (void)!write(fd, "abc", 3);
                                           shutdown(fd, SHUT_WR); }, "payload" },
        { "the peer closes", [](int fd) { shutdown(fd, SHUT_WR); }, "closed" },
    };
    for (const Case& c : cases) {
        Raw r;
        REQUIRE(r.rx);
        c.write(r.fd);
        Msg m;
        const int st = r.rx->recv(&m, 2000);
        CHECK_EQ(st, RAD_E_IO);
        CHECK(!r.rx->alive());
        const bool named = r.rx->why().find(c.in_why) != std::string::npos;
        if (!named) fprintf(stderr, "    %s: why() = '%s'\n", c.what, r.rx->why().c_str());
        CHECK(named);
    }
}

TEST(fuzz_no_damaged_frame_is_accepted_and_none_crashes) {
    std::mt19937_64 g(17);
    int refused = 0, accepted_damaged = 0;
    for (int it = 0; it < 5000; ++it) {
        Raw r(fast_opts(200));
        REQUIRE(r.rx);
        std::string body((size_t)(g() % 300), '\0');
        for (char& ch : body) ch = (char)(g() >> 56);
        WireHdr h{};
        h.kind = (uint16_t)(1 + g() % 40);
        if (h.kind == kHeartbeat || h.kind == kData) h.kind = kHello;
        h.seq = g();
        h.aux = g();
        h.bytes = body.size();
        h.flags = kFlagCrc;
        h.crc_payload = crc32c(body.data(), body.size());
        seal(&h);
        std::string w((const char*)&h, sizeof h);
        w += body;
        /* One byte of the header or payload flipped (never a no-op flip). */
        const size_t at = (size_t)(g() % w.size());
        w[at] = (char)(w[at] ^ (uint8_t)(1u << (g() % 8)));
        (void)!write(r.fd, w.data(), w.size());
        shutdown(r.fd, SHUT_WR);
        Msg m;
        const int st = r.rx->recv(&m, 1000);
        if (st == RAD_OK) ++accepted_damaged; else ++refused;
    }
    CHECK_EQ(accepted_damaged, 0);
    CHECK_EQ(refused, 5000);
    /* Random byte strings: refused (or a timeout on a string shorter than a header), never accepted. */
    int junk_accepted = 0;
    for (int it = 0; it < 2000; ++it) {
        Raw r(fast_opts(200));
        REQUIRE(r.rx);
        std::string w((size_t)(1 + g() % 200), '\0');
        for (char& ch : w) ch = (char)(g() >> 56);
        (void)!write(r.fd, w.data(), w.size());
        shutdown(r.fd, SHUT_WR);
        Msg m;
        if (r.rx->recv(&m, 500) == RAD_OK) ++junk_accepted;
    }
    CHECK_EQ(junk_accepted, 0);
}

/* ------------------------------------------------------------------ crossings */
/* A runs the leader's side (sends X1, receives X2), B the follower's (receives X1, sends X2); two
 * reader threads a receive ring stand for two ranks. Each round: A puts X1, both B readers take it
 * and B puts X2 derived from it, both A readers take X2 and check it. */
void run_crossings(NodeLink& A, NodeLink& B, int rounds, uint64_t cap, bool big) {
    Ring a_x1(RAD_PP_X1, cap, 0), a_x2(RAD_PP_X2, cap, 2), b_x1(RAD_PP_X1, cap, 2), b_x2(RAD_PP_X2, cap, 0);
    RadPpRing* as[RAD_PP_N_RINGS] = { a_x1.r, nullptr };
    RadPpRing* ar[RAD_PP_N_RINGS] = { nullptr, a_x2.r };
    RadPpRing* bs[RAD_PP_N_RINGS] = { nullptr, b_x2.r };
    RadPpRing* br[RAD_PP_N_RINGS] = { b_x1.r, nullptr };
    A.attach(as, ar);
    B.attach(bs, br);
    A.start();
    B.start();
    std::mt19937_64 g(5);
    std::atomic<int> bad{ 0 };
    std::vector<uint64_t> sizes;
    for (int i = 0; i < rounds; ++i) sizes.push_back(1 + g() % (1u << 18));
    if (big) sizes[(size_t)rounds / 2] = cap;
    /* The follower: two reader ranks; rank 0 also answers. */
    std::thread fol([&] {
        std::vector<uint8_t> in0, in1;
        for (int i = 0; i < rounds; ++i) {
            const uint64_t n = sizes[(size_t)i];
            in0.resize(n);
            in1.resize(n);
            std::thread r1([&] { if (ring_take(b_x1.r, 1, in1.data(), n, (uint64_t)i + 1) != RAD_OK) ++bad; });
            if (ring_take(b_x1.r, 0, in0.data(), n, (uint64_t)i + 1) != RAD_OK) ++bad;
            r1.join();
            if (in0 != in1) ++bad;
            for (uint8_t& c : in0) c = (uint8_t)(c ^ 0x5a);
            if (ring_put(b_x2.r, in0.data(), n, (uint64_t)i + 1) != RAD_OK) ++bad;
        }
    });
    std::vector<uint8_t> out, back0, back1;
    for (int i = 0; i < rounds; ++i) {
        const uint64_t n = sizes[(size_t)i];
        out.resize(n);
        fill(out, (uint64_t)i);
        CHECK_OK(ring_put(a_x1.r, out.data(), n, (uint64_t)i + 1));
        back0.resize(n);
        back1.resize(n);
        std::thread r1([&] { if (ring_take(a_x2.r, 1, back1.data(), n, (uint64_t)i + 1) != RAD_OK) ++bad; });
        if (ring_take(a_x2.r, 0, back0.data(), n, (uint64_t)i + 1) != RAD_OK) ++bad;
        r1.join();
        for (uint64_t k = 0; k < n; ++k)
            if (back0[k] != (uint8_t)(out[k] ^ 0x5a) || back1[k] != back0[k]) { ++bad; break; }
        if (bad) break;
    }
    fol.join();
    CHECK_EQ(bad.load(), 0);
    CHECK_EQ(a_x2.r->landed, (uint64_t)rounds);
    CHECK_EQ(b_x1.r->landed, (uint64_t)rounds);
    CHECK(A.alive() && B.alive());
    A.close("test done");
    B.close("test done");
}

TEST(crossings_carry_every_byte_over_a_socket_pair) {
    Pair p = socket_pair(fast_opts(5000));
    REQUIRE(p.a && p.b);
    run_crossings(*p.a, *p.b, 1000, 64ull << 20, true);
}

TEST(crossings_carry_every_byte_over_localhost_tcp) {
    const int port = 20000 + (int)(getpid() % 20000);
    std::unique_ptr<NodeLink> fol;
    std::string ferr;
    std::thread t([&] { fol = NodeLink::listen("127.0.0.1", port, fast_opts(5000), &ferr); });
    std::string lerr;
    std::unique_ptr<NodeLink> lead = NodeLink::connect("127.0.0.1", port, fast_opts(5000), &lerr);
    t.join();
    if (!lead || !fol) fprintf(stderr, "    connect: %s / listen: %s\n", lerr.c_str(), ferr.c_str());
    REQUIRE(lead && fol);
    run_crossings(*lead, *fol, 500, 8ull << 20, false);
}

/* ------------------------------------------------------------------ failure */
/* A child process holding one end: it answers heartbeats until killed or stopped. */
struct ChildPeer {
    pid_t pid = -1;
    std::unique_ptr<NodeLink> link;
    Ring x2{ RAD_PP_X2, 1 << 16, 1, 0 };
    explicit ChildPeer(const LinkOpts& o) {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) return;
        pid = fork();
        if (pid == 0) {
            close(sv[0]);
            std::unique_ptr<NodeLink> c = NodeLink::adopt(sv[1], o);
            c->start();
            for (;;) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        close(sv[1]);
        link = NodeLink::adopt(sv[0], o);
        RadPpRing* s[RAD_PP_N_RINGS] = { nullptr, nullptr };
        RadPpRing* r[RAD_PP_N_RINGS] = { nullptr, x2.r };
        link->attach(s, r);
        link->start();
    }
    ~ChildPeer() {
        link.reset();
        if (pid > 0) { kill(pid, SIGKILL); int st; waitpid(pid, &st, 0); }
    }
};

/* A reader waiting for X2 when the peer dies returns RAD_E_IO, and within dead_ms + slack. */
void check_reader_released(ChildPeer& c, int sig, int dead_ms, const char* in_why) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(c.link->alive());
    std::atomic<int> st{ 1 };
    const auto t0 = std::chrono::steady_clock::now();
    std::thread reader([&] {
        uint8_t buf[16];
        st = ring_take(c.x2.r, 0, buf, sizeof buf, 1);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    kill(c.pid, sig);
    reader.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK_EQ(st.load(), RAD_E_IO);
    CHECK(!c.link->alive());
    const bool named = c.link->why().find(in_why) != std::string::npos;
    if (!named) fprintf(stderr, "    why() = '%s'\n", c.link->why().c_str());
    CHECK(named);
    fprintf(stderr, "    signal %d: the reader was released after %.0f ms (dead_ms %d)\n", sig, ms, dead_ms);
    CHECK(ms < dead_ms + 1500);
}

TEST(a_killed_peer_releases_the_waiting_reader_at_once) {
    ChildPeer c(fast_opts(2000));
    REQUIRE(c.pid > 0);
    check_reader_released(c, SIGKILL, 2000, "closed");
}

TEST(a_frozen_peer_is_declared_down_after_dead_ms) {
    ChildPeer c(fast_opts(400));
    REQUIRE(c.pid > 0);
    check_reader_released(c, SIGSTOP, 400, "no frame from the peer");
}

/* Robustness review finding 1: the peer freezes (SIGSTOP: it stops reading, its kernel keeps the
 * connection open) while this node writes a crossing larger than both socket buffers (~16 MiB with
 * gpu02 / gpu03's 4 MiB wmem_max / rmem_max; X1 at 4,160 rows is 42.6 MB). Before the fix the writer
 * blocked in sendmsg holding the socket, the heartbeat queued behind it and the reader waited out the
 * ring's deadline (6,003 ms here), a control send and close() blocked. Now: the reader is released
 * with RAD_E_IO within dead_ms + 1.5 s, a control send fails at once, close() returns. */
TEST(a_peer_frozen_under_a_crossing_larger_than_the_socket_buffers_is_declared_down) {
    const int dead_ms = 400;
    const uint64_t bytes = 42598400ull, cap = 64ull << 20;
    const int port = 30000 + (int)(getpid() % 15000);
    pid_t pid = fork();
    if (pid == 0) {
        std::string err;
        std::unique_ptr<NodeLink> f = NodeLink::listen("127.0.0.1", port, fast_opts(dead_ms), &err, "127.0.0.1");
        if (!f) _exit(3);
        Ring x1(RAD_PP_X1, cap, 1, 0), x2(RAD_PP_X2, 4096, 0, 0);
        RadPpRing* fs[RAD_PP_N_RINGS] = { nullptr, x2.r };
        RadPpRing* fr[RAD_PP_N_RINGS] = { x1.r, nullptr };
        f->attach(fs, fr);
        f->start();
        std::vector<uint8_t> in(cap), out(64, 0x5a);
        for (uint64_t k = 1;; ++k) {
            while (__atomic_load_n(&x1.r->landed, __ATOMIC_ACQUIRE) < k) {
                if (!f->alive()) _exit(0);
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            const uint64_t n = x1.r->bytes - RAD_PP_TRAILER, rows = x1.r->rows;   /* aligned sizes here */
            if (ring_take(x1.r, 0, in.data(), n, rows) != RAD_OK) _exit(4);
            if (ring_put(x2.r, out.data(), 64, 1) != RAD_OK) _exit(5);
        }
    }
    REQUIRE(pid > 0);
    std::string err;
    std::unique_ptr<NodeLink> L = NodeLink::connect("127.0.0.1", port, fast_opts(dead_ms), &err);
    if (!L) { kill(pid, SIGKILL); waitpid(pid, nullptr, 0); }
    REQUIRE(L);
    Ring x1(RAD_PP_X1, cap, 0, 0), x2(RAD_PP_X2, 4096, 1, 6000);
    RadPpRing* ls[RAD_PP_N_RINGS] = { x1.r, nullptr };
    RadPpRing* lr[RAD_PP_N_RINGS] = { nullptr, x2.r };
    L->attach(ls, lr);
    L->start();
    std::vector<uint8_t> big(bytes, 0xab), back(64);
    CHECK_OK(ring_put(x1.r, big.data(), 1024, 1));   /* one good round */
    CHECK_OK(ring_take(x2.r, 0, back.data(), 64, 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    kill(pid, SIGSTOP);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_OK(ring_put(x1.r, big.data(), bytes, 1));
    const int take = ring_take(x2.r, 0, back.data(), 64, 1);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "    42.6 MB crossing, peer frozen: the reader released after %.0f ms -> %d (%s)\n", ms, take,
            L->why().c_str());
    CHECK_EQ(take, RAD_E_IO);
    CHECK(ms < dead_ms + 1500);
    CHECK(!L->alive());
    auto ctl = std::async(std::launch::async, [&] { return L->send(kStep, 9, 0, "x", 1); });
    CHECK(ctl.wait_for(std::chrono::milliseconds(1500)) == std::future_status::ready);
    auto cl = std::async(std::launch::async, [&] { L->close("the leader is stopping"); return 0; });
    CHECK(cl.wait_for(std::chrono::milliseconds(1500)) == std::future_status::ready);
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    if (ctl.valid()) CHECK_EQ(ctl.get(), RAD_E_IO);
    if (cl.valid()) (void)cl.get();
    /* The writer's own limit, without the heartbeat (a link not started): a peer that never reads
     * fails a 64 MiB frame dead_ms after the last byte it accepted, instead of blocking for ever. */
    Pair p = socket_pair(fast_opts(300));
    REQUIRE(p.a && p.b);
    std::vector<uint8_t> blob(64ull << 20, 1);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK_EQ(p.a->send(kUser, 1, 0, blob.data(), (int64_t)blob.size()), RAD_E_IO);
    const double ms1 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    fprintf(stderr, "    a 64 MiB frame to a peer that never reads: failed after %.0f ms: %s\n", ms1, p.a->why().c_str());
    CHECK(ms1 < 300 + 1000);
    CHECK(p.a->why().find("accepted no byte") != std::string::npos);
}

/* Robustness review finding 2: a node whose own stage never finishes while its host still sends
 * heartbeats (a hung kernel) takes the link down after RADIANCE_PP_STAGE_MS, naming it, instead of the
 * peer's cards waiting out the rings' deadline. The follower owes X2 once X1 landed; the leader owes X1
 * once it released a step. */
TEST(a_stage_that_never_finishes_takes_the_link_down_after_stage_ms) {
    LinkOpts o = fast_opts(2000);
    o.stage_ms = 300;
    {
        Pair p = socket_pair(o);
        REQUIRE(p.a && p.b);
        Ring a_x1(RAD_PP_X1, 4096, 0, 0), b_x1(RAD_PP_X1, 4096, 1, 0), b_x2(RAD_PP_X2, 4096, 0, 0),
            a_x2(RAD_PP_X2, 4096, 1, 0);
        RadPpRing* as[RAD_PP_N_RINGS] = { a_x1.r, nullptr };
        RadPpRing* ar[RAD_PP_N_RINGS] = { nullptr, a_x2.r };
        RadPpRing* bs[RAD_PP_N_RINGS] = { nullptr, b_x2.r };
        RadPpRing* br[RAD_PP_N_RINGS] = { b_x1.r, nullptr };
        p.a->attach(as, ar);
        p.b->attach(bs, br);
        p.a->start();
        p.b->start();
        uint8_t buf[64] = {};
        p.a->note_released(1);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK_OK(ring_put(a_x1.r, buf, 64, 1));   /* the leader's X1: it lands on the follower */
        /* The follower's stage never publishes its X2. */
        Msg m;
        CHECK_EQ(p.b->recv(&m, 3000), RAD_E_IO);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "    follower stage stuck: down after %.0f ms: %s\n", ms, p.b->why().c_str());
        CHECK(p.b->why().find("RADIANCE_PP_STAGE_MS") != std::string::npos);
        CHECK(p.b->why().find("X2") != std::string::npos);
        CHECK(ms >= 290 && ms < 1500);
        CHECK_EQ(p.a->recv(&m, 3000), RAD_E_IO);   /* the leader is told */
        CHECK(p.a->why().find("RADIANCE_PP_STAGE_MS") != std::string::npos);
    }
    {
        Pair p = socket_pair(o);
        REQUIRE(p.a && p.b);
        Ring a_x1(RAD_PP_X1, 4096, 0, 0), a_x2(RAD_PP_X2, 4096, 1, 0);
        RadPpRing* as[RAD_PP_N_RINGS] = { a_x1.r, nullptr };
        RadPpRing* ar[RAD_PP_N_RINGS] = { nullptr, a_x2.r };
        p.a->attach(as, ar);
        p.a->start();
        p.b->start();
        p.a->note_released(1);   /* a released step whose X1 never comes */
        Msg m;
        CHECK_EQ(p.a->recv(&m, 3000), RAD_E_IO);
        CHECK(p.a->why().find("X1 for a step it released") != std::string::npos);
    }
    {
        /* A stage that finishes within the limit: nothing happens. */
        Pair p = socket_pair(o);
        REQUIRE(p.a && p.b);
        Ring a_x1(RAD_PP_X1, 4096, 0, 0), b_x1(RAD_PP_X1, 4096, 0, 0);   /* no local reader on b */
        RadPpRing* as[RAD_PP_N_RINGS] = { a_x1.r, nullptr };
        RadPpRing* br[RAD_PP_N_RINGS] = { b_x1.r, nullptr };
        RadPpRing* none[RAD_PP_N_RINGS] = { nullptr, nullptr };
        p.a->attach(as, none);
        p.b->attach(none, br);
        p.a->start();
        p.b->start();
        uint8_t buf[64] = {};
        for (uint64_t k = 1; k <= 5; ++k) {
            p.a->note_released(k);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            CHECK_OK(ring_put(a_x1.r, buf, 64, 1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        CHECK(p.a->alive() && p.b->alive());
        CHECK_EQ(b_x1.r->landed, 5ull);
    }
}

/* Robustness review finding 4: the follower takes the leader's connection only from --pp-peer's
 * address; any other is closed and the wait goes on. */
TEST(the_follower_takes_a_connection_only_from_its_peer_address) {
    const int port = 20000 + (int)((getpid() * 7) % 20000);
    std::unique_ptr<NodeLink> fol;
    std::string ferr;
    std::thread t([&] { fol = NodeLink::listen("127.0.0.1", port, fast_opts(2000), &ferr, "127.0.0.2"); });
    auto dial = [&](const char* from) {
        const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_in src{};
        src.sin_family = AF_INET;
        inet_pton(AF_INET, from, &src.sin_addr);
        (void)bind(fd, (sockaddr*)&src, sizeof src);
        sockaddr_in dst{};
        dst.sin_family = AF_INET;
        dst.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, "127.0.0.1", &dst.sin_addr);
        for (int i = 0; i < 100 && ::connect(fd, (sockaddr*)&dst, sizeof dst) != 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return fd;
    };
    const int stranger = dial("127.0.0.1");
    char c = 0;
    pollfd pf{ stranger, POLLIN, 0 };
    CHECK(poll(&pf, 1, 3000) == 1);
    CHECK_EQ(read(stranger, &c, 1), (ssize_t)0);   /* closed by the follower */
    close(stranger);
    const int leader = dial("127.0.0.2");
    t.join();
    REQUIRE(fol);
    WireHdr h{};
    h.kind = kHello;
    raw_frame(leader, h, "the leader");
    Msg m;
    CHECK_OK(fol->recv(&m, 2000));
    CHECK(std::string(m.body.begin(), m.body.end()) == "the leader");
    close(leader);
}

TEST(a_clean_close_says_so_and_an_abort_carries_its_reason) {
    {
        Pair p = socket_pair(fast_opts());
        p.a->start();
        p.b->start();
        p.a->close("the leader is stopping");
        Msg m;
        CHECK_EQ(p.b->recv(&m, 2000), RAD_E_IO);
        CHECK(p.b->why().find("the peer stopped: the leader is stopping") != std::string::npos);
    }
    {
        Pair p = socket_pair(fast_opts());
        p.a->start();
        p.b->start();
        p.a->fail("rank 2 step 41: a device fault");
        Msg m;
        CHECK_EQ(p.b->recv(&m, 2000), RAD_E_IO);
        CHECK(p.b->why().find("rank 2 step 41: a device fault") != std::string::npos);
    }
}

TEST(an_overrun_a_skipped_payload_and_a_foreign_ring_take_the_link_down) {
    /* Overrun: two payloads land while the one reader takes neither. */
    {
        Pair p = socket_pair(fast_opts(300));
        Ring s(RAD_PP_X1, 4096, 0), r(RAD_PP_X1, 4096, 1);
        RadPpRing* as[RAD_PP_N_RINGS] = { s.r, nullptr };
        RadPpRing* none[RAD_PP_N_RINGS] = { nullptr, nullptr };
        RadPpRing* br[RAD_PP_N_RINGS] = { r.r, nullptr };
        p.a->attach(as, none);
        p.b->attach(none, br);
        p.a->start();
        p.b->start();
        uint8_t buf[64] = {};
        CHECK_OK(ring_put(s.r, buf, 64, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK_OK(ring_put(s.r, buf, 64, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        CHECK(!p.b->alive());
        CHECK(p.b->why().find("overrun") != std::string::npos);
    }
    /* A payload numbered past the next one. */
    {
        Raw r;
        Ring x(RAD_PP_X1, 4096, 1);
        RadPpRing* none[RAD_PP_N_RINGS] = { nullptr, nullptr };
        RadPpRing* br[RAD_PP_N_RINGS] = { x.r, nullptr };
        r.rx->attach(none, br);
        r.rx->start();
        WireHdr h{};
        h.kind = kData;
        h.lane = RAD_PP_X1;
        h.seq = 2;
        raw_frame(r.fd, h, std::string(16, 'a'), false);
        Msg m;
        CHECK_EQ(r.rx->recv(&m, 2000), RAD_E_IO);
        CHECK(r.rx->why().find("out of step") != std::string::npos);
        CHECK_EQ(x.r->abort, 1ull);
    }
    /* A payload for a ring this node does not receive. */
    {
        Raw r;
        r.rx->start();
        WireHdr h{};
        h.kind = kData;
        h.lane = RAD_PP_X2;
        h.seq = 1;
        raw_frame(r.fd, h, std::string(16, 'a'), false);
        Msg m;
        CHECK_EQ(r.rx->recv(&m, 2000), RAD_E_IO);
        CHECK(r.rx->why().find("does not receive") != std::string::npos);
    }
}

TEST(a_ring_refuses_a_payload_past_its_capacity_and_a_reader_of_the_wrong_shape) {
    Ring s(RAD_PP_X1, 128, 1);   /* 128 + 256 + 64 bytes: one aligned part of up to 384 B and its trailer */
    uint8_t buf[1024] = {};
    CHECK_EQ(ring_put(s.r, buf, 1024, 1), RAD_E_FULL);
    CHECK_OK(ring_put(s.r, buf, 64, 3));
    CHECK_EQ(s.r->bytes, (uint64_t)(RAD_PP_PART_ALIGN + RAD_PP_TRAILER));   /* the trailer travels with it */
    /* A host-only loop-back: the reader of the same ring sees it as landed once copied over. */
    __atomic_store_n(&s.r->landed, 1ull, __ATOMIC_RELEASE);
    CHECK_EQ(ring_take(s.r, 0, buf, 64, 4), RAD_E_SHAPE);   /* rows 3, not 4 */
    CHECK_OK(ring_take(s.r, 0, buf, 64, 3));
    /* The next take waits for payload 2 and runs into the deadline. */
    s.r->deadline_ms = 30;
    CHECK_EQ(ring_take(s.r, 0, buf, 64, 3), RAD_E_NOTFOUND);
}

/* Review F6: a payload whose header is this step's (rows, bytes) but whose trailer is not the number
 * waited for -- stale bytes, or a ring written over -- is refused, never copied. */
TEST(a_stale_payload_of_the_same_shape_is_refused_by_its_trailer) {
    Ring s(RAD_PP_X1, 4096, 1);
    std::vector<uint8_t> a(1000, 0x11), b(1000, 0);
    CHECK_OK(ring_put(s.r, a.data(), a.size(), 7));
    __atomic_store_n(&s.r->sent, 1ull, __ATOMIC_RELEASE);
    __atomic_store_n(&s.r->landed, 1ull, __ATOMIC_RELEASE);
    CHECK_OK(ring_take(s.r, 0, b.data(), b.size(), 7));
    CHECK(a == b);
    /* Payload 2 "lands" (the counter) while the ring still holds payload 1's bytes and trailer. */
    __atomic_store_n(&s.r->landed, 2ull, __ATOMIC_RELEASE);
    CHECK_EQ(ring_take(s.r, 0, b.data(), b.size(), 7), RAD_E_STATE);
    CHECK_EQ(s.r->taken[0], 1ull);   /* not counted as taken */
    /* The real payload 2 is taken. */
    CHECK_OK(ring_put(s.r, a.data(), a.size(), 7));
    CHECK_OK(ring_take(s.r, 0, b.data(), b.size(), 7));
}

TEST(the_verbs_path_is_absent_or_loads_by_name) {
    std::string why;
    const bool built = verbs_built();
    const bool loads = verbs_loadable(&why);
    fprintf(stderr, "    verbs: built %d, libibverbs loads %d%s%s\n", (int)built, (int)loads,
            why.empty() ? "" : ": ", why.c_str());
    if (!built) CHECK(!loads && !why.empty());
}

}  /* namespace */

RAD_TEST_MAIN()
