// bwcc.cpp (9.11.2026, queue-ar, item 111) -- host <-> card and card <-> card copy bandwidth on every visible card, one at
// a time and CONCURRENTLY (rocm-bandwidth-test 2.6.0 measures only one transfer at a time). hipMemcpyAsync from / to
// pinned host memory (hipHostMalloc, each card's own buffer, allocated by the thread that drives that card) and
// hipMemcpyPeerAsync between cards (peer access enabled where hipDeviceCanAccessPeer says so). Read-only for the cards'
// state: allocates, copies, frees.
//
//   bwcc [MiB=256] [REPS=20] [ROUNDS=3]
//
// Every test runs ROUNDS times; a line per test gives each stream's GB/s (10^9 bytes / s) of its best round and the
// aggregate (all bytes / the wall from the first start to the last end) of that round. Tests:
//   h2d / d2h / bidir alone        each card by itself (bidir: one H2D and one D2H stream at once)
//   h2d / d2h / bidir all          every card at once
//   pair h2d                        every pair of cards at once (which pairs share an uplink)
//   p2p i->j alone                  all directed pairs, one at a time
//   p2p ring all                    i -> i+1 (mod n) on every card at once; p2p all-pairs all: every directed pair at once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "bwcc: %s failed: %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

using clk = std::chrono::steady_clock;

struct Xfer {           // one stream's job: REPS copies of `bytes`
    int kind;           // 0 h2d, 1 d2h, 2 p2p
    int dev;            // the card that owns the stream (the source for p2p)
    int peer;           // p2p destination
};

static size_t g_bytes;
static int g_reps, g_rounds, g_n;
static std::vector<void*> g_host, g_host2, g_dev, g_dev2;
static std::vector<std::string> g_bdf;

struct Res { double gbps; clk::time_point s, e; };

// Run the jobs at once (one thread each); returns per-job results of the best round (by aggregate).
static std::vector<Res> run(const std::vector<Xfer>& jobs, double* agg_out) {
    std::vector<Res> best;
    double best_agg = -1;
    for (int round = 0; round < g_rounds; ++round) {
        std::vector<Res> r(jobs.size());
        std::barrier sync((std::ptrdiff_t)jobs.size());
        std::vector<std::thread> th;
        for (size_t k = 0; k < jobs.size(); ++k) {
            th.emplace_back([&, k] {
                const Xfer& x = jobs[k];
                CK(hipSetDevice(x.dev));
                hipStream_t s;
                CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
                // distinct buffers when one card runs two streams (bidir) or is the source of several p2p jobs
                void* hsrc = g_host[x.dev];
                void* hdst = g_host2[x.dev];
                void* dbuf = x.kind == 1 ? g_dev2[x.dev] : g_dev[x.dev];
                auto issue = [&] {
                    if (x.kind == 0) CK(hipMemcpyAsync(dbuf, hsrc, g_bytes, hipMemcpyHostToDevice, s));
                    else if (x.kind == 1) CK(hipMemcpyAsync(hdst, dbuf, g_bytes, hipMemcpyDeviceToHost, s));
                    else CK(hipMemcpyPeerAsync(g_dev2[x.peer], x.peer, g_dev[x.dev], x.dev, g_bytes, s));
                };
                issue(); issue();
                CK(hipStreamSynchronize(s));
                sync.arrive_and_wait();
                const auto t0 = clk::now();
                for (int i = 0; i < g_reps; ++i) issue();
                CK(hipStreamSynchronize(s));
                const auto t1 = clk::now();
                r[k] = Res{(double)g_bytes * g_reps / std::chrono::duration<double>(t1 - t0).count() / 1e9, t0, t1};
                CK(hipStreamDestroy(s));
            });
        }
        for (auto& t : th) t.join();
        auto s0 = r[0].s, e1 = r[0].e;
        for (auto& x : r) { s0 = std::min(s0, x.s); e1 = std::max(e1, x.e); }
        const double agg = (double)g_bytes * g_reps * jobs.size() / std::chrono::duration<double>(e1 - s0).count() / 1e9;
        if (agg > best_agg) { best_agg = agg; best = r; }
    }
    *agg_out = best_agg;
    return best;
}

static const char* kname(int k) { return k == 0 ? "h2d" : k == 1 ? "d2h" : "p2p"; }

static void report(const char* test, const std::vector<Xfer>& jobs) {
    double agg = 0;
    auto r = run(jobs, &agg);
    std::string line = std::string("BW ") + test + " |";
    char b[256];
    for (size_t k = 0; k < jobs.size(); ++k) {
        const Xfer& x = jobs[k];
        if (x.kind == 2) std::snprintf(b, sizeof b, " %s->%s %.1f", g_bdf[x.dev].c_str(), g_bdf[x.peer].c_str(), r[k].gbps);
        else std::snprintf(b, sizeof b, " %s %s %.1f", kname(x.kind), g_bdf[x.dev].c_str(), r[k].gbps);
        line += b;
    }
    std::snprintf(b, sizeof b, " | aggregate %.1f GB/s", agg);
    line += b;
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const long mib = argc > 1 ? std::atol(argv[1]) : 256;
    g_reps = argc > 2 ? std::atoi(argv[2]) : 20;
    g_rounds = argc > 3 ? std::atoi(argv[3]) : 3;
    g_bytes = (size_t)mib << 20;
    CK(hipGetDeviceCount(&g_n));
    if (g_n < 1) { std::fprintf(stderr, "bwcc: no card\n"); return 2; }
    g_host.resize(g_n); g_host2.resize(g_n); g_dev.resize(g_n); g_dev2.resize(g_n); g_bdf.resize(g_n);
    for (int d = 0; d < g_n; ++d) {
        char bus[64] = {};
        CK(hipDeviceGetPCIBusId(bus, sizeof bus, d));
        g_bdf[d] = std::string(bus).substr(5, 7);   // "0000:83:00.0" -> "83:00.0"
        std::thread([d] {
            CK(hipSetDevice(d));
            CK(hipHostMalloc(&g_host[d], g_bytes, hipHostMallocDefault));
            CK(hipHostMalloc(&g_host2[d], g_bytes, hipHostMallocDefault));
            std::memset(g_host[d], 1, g_bytes);
            std::memset(g_host2[d], 2, g_bytes);
            CK(hipMalloc(&g_dev[d], g_bytes));
            CK(hipMalloc(&g_dev2[d], g_bytes));
        }).join();
    }
    std::string peers;
    for (int i = 0; i < g_n; ++i)
        for (int j = 0; j < g_n; ++j) {
            if (i == j) continue;
            int can = 0;
            CK(hipDeviceCanAccessPeer(&can, i, j));
            if (can) { CK(hipSetDevice(i)); hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                       if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) can = 0; (void)hipGetLastError(); }
            peers += " " + g_bdf[i] + "->" + g_bdf[j] + (can ? ":peer" : ":nopeer");
        }
    std::printf("BWCC cards %d, %ld MiB x %d copies, best of %d rounds; card order:", g_n, mib, g_reps, g_rounds);
    for (int d = 0; d < g_n; ++d) std::printf(" %d=%s", d, g_bdf[d].c_str());
    std::printf(" | peer access:%s\n", peers.c_str());
    std::fflush(stdout);

    for (int k = 0; k < 2; ++k)
        for (int d = 0; d < g_n; ++d) report((std::string(kname(k)) + " alone").c_str(), {{k, d, -1}});
    for (int d = 0; d < g_n; ++d) report("bidir alone", {{0, d, -1}, {1, d, -1}});
    for (int k = 0; k < 2; ++k) {
        std::vector<Xfer> j;
        for (int d = 0; d < g_n; ++d) j.push_back({k, d, -1});
        report((std::string(kname(k)) + " all").c_str(), j);
    }
    {
        std::vector<Xfer> j;
        for (int d = 0; d < g_n; ++d) { j.push_back({0, d, -1}); j.push_back({1, d, -1}); }
        report("bidir all", j);
    }
    for (int a = 0; a < g_n; ++a)
        for (int b = a + 1; b < g_n; ++b) report("pair h2d", {{0, a, -1}, {0, b, -1}});
    for (int a = 0; a < g_n; ++a)
        for (int b = a + 1; b < g_n; ++b) report("pair d2h", {{1, a, -1}, {1, b, -1}});
    if (g_n > 1) {
        for (int i = 0; i < g_n; ++i)
            for (int j = 0; j < g_n; ++j)
                if (i != j) report("p2p alone", {{2, i, j}});
        std::vector<Xfer> ring, all;
        for (int i = 0; i < g_n; ++i) ring.push_back({2, i, (i + 1) % g_n});
        report("p2p ring all", ring);
        for (int i = 0; i < g_n; ++i)
            for (int j = 0; j < g_n; ++j)
                if (i != j) all.push_back({2, i, j});
        report("p2p all-pairs all", all);
    }
    for (int d = 0; d < g_n; ++d) {
        CK(hipSetDevice(d));
        CK(hipFree(g_dev[d])); CK(hipFree(g_dev2[d]));
        CK(hipHostFree(g_host[d])); CK(hipHostFree(g_host2[d]));
    }
    std::printf("BWCC done\n");
    return 0;
}
