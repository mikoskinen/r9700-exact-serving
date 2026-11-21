// q2bench -- TWO QUEUES IN ONE PROCESS on one node's four cards, TP4 with the all-reduces (tp8 loop DECODE R1; COORDINATOR
// 21:58 = audit DECISION-2 route R1, re-aimed 22:20 by DECISION-2-skeptic: "the bench must be TP4 with the all-reduces in it
// (not one card), newcomer-like pass on queue A + 16-row decode passes on queue B vs serial").
//
//   q2bench --exl3 <exl3r4d.so> --r4d <libr4d.so> [--fn4x NAME=V ...] [--fn4r NAME=V ...] [--MA 1100] [--LA 22]
//           [--MB 16,32] [--LB 22] [--experts 384] [--topk 6] [--hidden 5120] [--widths 640,512] [--bits 2] [--h-rot 1]
//           [--pools 2] [--reps 3] [--warm-ms 600] [--dense 0|1] [--arms n16,h16,n32,h32,l16,x16] [--plan] [--check]
//           [--dense-shapes N:K,...] [--dense-ar N] [--rows-only]   (Qwen loop round 9: another model's dense shapes, q6_K stand-ins, the
//           first all-reduce after the first N of them; e.g. Qwen Flash-Next tp4 rank shapes 4096:2560,2560:1536,512:2560 / 2)
//
// WHAT RUNS. Four host threads, one a card (HIP devices 0..3), each its rank of a TP4 node. Every card has four streams:
// qa (normal priority), qal (the lowest), qb (normal), qbh (the highest). A "layer" is the launches a deepseek41 layer issues
// at the rank-0 TP4 shapes, minus the attention core / indexer / small element-wise ops:
//   dense (libr4d gemm_nt_ggml, q6_K bytes, the served forms via the engine's env): attn_q_a 1280x5120, attn_kv 512x5120,
//     attn_q_b 8192x1280, attn_output_a 1024x4096, attn_output_b 5120x2048; then the attention all-reduce;
//     shexp_gate_up 1536x5120, shexp_down 5120x768;
//   the routed experts (exl3r4d gate_up -> down, two parity classes at --widths, K --bits, h_rot; _m16 up to 16 rows, lever
//     P's _dk at 17..48, _pf above), a lognormal routing draw a layer (perfmoe's), then the MoE all-reduce (libr4d's
//     ar_oneshot_4rank_exact below 524,289 elements, ar_twoshot_4rank_exact from it: the rows' own bounds).
// Work A = one newcomer's BG body: --LA layers at --MA rows (the follower's 22-layer half at ~1.1k rows). Work B = one decode
// pass: --LB layers at MB rows (16 = c32's undrafted lane, 32 = a drafted lane of 16 + 16 or c16's 8 x 4). A and B read
// weights of their own (A pools 0..P-1, B pools P..2P-1; a pool is ~0.9 GB a card at the default shapes, so every layer
// streams from DRAM), and every all-reduce instance of A and of B is its own buffer set (libr4d keys an instance on numel).
//
// THE MEASUREMENT (every rank, in lockstep; rank 0's device clock reported, the others' spread printed):
//   T_A   = an A body alone on qa;  T_B(m) = a B pass alone on qb (median of 6 back to back).
//   overlap arm X (n16 = B16 on qb beside A on qa; h16 = B16 on qbh (high); l16 = A on qal (low) beside B16 on qb; x16 = A low +
//   B16 high; n32 / h32 the same at 32 rows):
//     A's body is enqueued on its queue, then ceil(3 T_A / T_B) + 2 B passes on the other, at most three in flight (the same
//     count on every rank, so every all-reduce has its peers). A' = A's body time under overlap; nB = the B passes inside A's
//     window (fractional at its ends, device timestamps); B' = the p50 B pass inside the window.
//     S = T_A / A' + nB x T_B / A'  (= serial time of the work done / overlapped time; F13's D(3)/D(1) + P(3)/P(2)).
//   Rule-1 bars (DECISION-2-skeptic, COORDINATOR 22:20): S >= ~1.07 for a follower queue at c32, >= ~1.12-1.14 for the
//   leader at c16. F13 (two PROCESSES, 5-layer proxy, no dense): S 1.10 / 1.08 with the decode step x3.2.
//   OVCHECK: A's last-layer outputs (dense y, expert y) after an overlapped body are compared byte for byte with an A body
//   alone (a pass's arithmetic must not see what runs beside it: M6's exactness premise, here at kernel level).
//   Q2K: every KFD process's queues by gpu id (/sys/class/kfd/kfd/proc/*/queues; in a container getpid() is not KFD's pid, so
//   all are listed: ours is the one with >= 4 a card when the node runs nothing else), read during the first overlap arm's run.
// --plan: host only (no HIP call): the configuration, launches a layer, bytes a card. --check: host only, the window / S
// arithmetic on made-up timelines (serial, perfect overlap, time-sliced, passes straddling the window's ends).
//
// Limits: no attention core / indexer / router / norms (the dense GEMMs + experts + all-reduces are ~70-75 % of a pass's
// op time on the stamps, F44 / F51 / X4); HIP streams stand in for the engine's AQL queues (the same KFD user queues; the
// engine sets no priority today); rank 0's widths on every rank; synthetic routing (lognormal, perfmoe's draw).
// Exit 0, 1 on a failed launch / init, 3 on a hang (a phase without progress for 120 s).
#include "rad_abi.h"
#include "rad_plugin.h"
#include "exl3_abi.h"

#include <hip/hip_runtime_api.h>

#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kDtTrellis = RAD_DT_PLUGIN_BASE + 0;   // exl3r4d's trellis record (kernels/exl3r4d/exl3r4d.h)
constexpr int kWorld = 4;

[[noreturn]] void die(const std::string& m) {
    std::fflush(stdout);
    std::fprintf(stderr, "q2bench: %s\n", m.c_str());
    std::_Exit(2);
}
void ck(hipError_t e, const char* what, int dev) {
    if (e != hipSuccess) die(std::string(what) + " (device " + std::to_string(dev) + "): " + hipGetErrorString(e));
}

RadParam PI(const char* k, long long v) { RadParam p = RAD_INT(k, v); return p; }
RadParam PS(const char* k, const char* v) { RadParam p = RAD_STR(k, v); return p; }
RadTensor T(void* d, uint32_t dt, std::vector<int64_t> s) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = d;
    t.dtype = dt;
    t.rank = (uint32_t)s.size();
    for (size_t i = 0; i < s.size(); ++i) t.shape[i] = s[i];
    rad_tensor_pack(&t);
    return t;
}
RadTensor TAB(void* tab, uint32_t dt, std::vector<int64_t> s) {   // a pointer table: stride[0] = 0
    RadTensor t = T(tab, dt, s);
    t.stride[0] = 0;
    return t;
}

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
double unif() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

void* g_lib_h[2] = { nullptr, nullptr };   // [0] exl3r4d, [1] libr4d
const RadKernelInfo* row_of(int lib, const std::string& path, const char* name) {
    if (!g_lib_h[lib]) {
        g_lib_h[lib] = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!g_lib_h[lib]) die("dlopen " + path + ": " + dlerror());
    }
    auto count = (int (*)())dlsym(g_lib_h[lib], "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(g_lib_h[lib], "rad_kernel_at");
    if (!count || !at) die(path + " is not a radiance kernel library");
    for (int i = 0; i < count(); ++i)
        if (at(i) && at(i)->name && std::strcmp(at(i)->name, name) == 0) return at(i);
    die(path + " has no row " + name);
}

// perfmoe's routing draw: expert popularity lognormal (sigma 0.8) a draw, each token's top_k by a Gumbel draw, stable sort.
struct Route {
    std::vector<int32_t> sorted, off;
    std::vector<int64_t> rows;
    int64_t live = 0;
};
Route draw(int64_t M, int64_t E, int64_t topk) {
    std::vector<double> logp((size_t)E);
    for (auto& v : logp) v = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < M; ++t) {
        for (int64_t e = 0; e < E; ++e) sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + topk, sc.end(),
                          [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
        for (int64_t j = 0; j < topk; ++j) by[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * topk + j));
    }
    Route r;
    r.off.assign((size_t)E + 1, 0);
    r.rows.assign((size_t)E, 0);
    for (int64_t e = 0; e < E; ++e) {
        r.rows[(size_t)e] = (int64_t)by[(size_t)e].size();
        r.live += r.rows[(size_t)e] > 0;
        r.off[(size_t)e + 1] = r.off[(size_t)e] + (int32_t)r.rows[(size_t)e];
        r.sorted.insert(r.sorted.end(), by[(size_t)e].begin(), by[(size_t)e].end());
    }
    return r;
}

struct Opt {
    std::string exl3, r4d;
    std::vector<std::pair<std::string, int>> fn4x, fn4r;   // exl3r4d's / libr4d's fn4 switches
    int64_t MA = 1100, E = 384, topk = 6, hidden = 5120;
    int LA = 22, LB = 22, bits = 2, h_rot = 1, pools = 2, reps = 3;
    std::vector<int64_t> MB = { 16, 32 };
    std::vector<int64_t> widths = { 640, 512 };
    float warm_ms = 600;
    bool dense = true, plan = false, rows_only = false;   // rows_only (Qwen loop round 9): the host row check, then exit
    std::vector<std::string> arms = { "n16", "h16", "n32", "h32", "l16", "x16" };
};

// The dense shapes a layer (rank 0 of TP4; tools: r4d_selftest ds_gg_perf's list), q6_K, in issue order; the attention
// all-reduce goes after index 4 (attn_output_b).
struct Dense { const char* name; int64_t N, K; };
/* --dense-shapes N:K,... (Qwen loop round 9, QUEUE Q5b): another model's dense shapes as q6_K stand-ins (the same launch form,
 * ~0.82 bytes a weight), the attention / mixer all-reduce after the first --dense-ar of them; default the DeepSeek list. */
std::vector<Dense> kDense = { { "attn_q_a", 1280, 5120 }, { "attn_kv", 512, 5120 }, { "attn_q_b", 8192, 1280 },
                              { "attn_output_a", 1024, 4096 }, { "attn_output_b", 5120, 2048 },
                              { "shexp_gate_up", 1536, 5120 }, { "shexp_down", 5120, 768 } };
int kNDense = 7, g_dense_ar = 5;
constexpr int kQ6Blk = 210;   // q6_K: 210 bytes a 256-value block, its fp16 d at byte 208
int64_t dense_bytes(const Dense& d) { return d.N * (d.K / 256) * kQ6Blk; }

// One launch of a layer: the row, its parameters and operands, its instance and scratch.
struct Call {
    const RadKernelInfo* row = nullptr;
    const std::vector<RadParam>* p = nullptr;
    std::vector<RadTensor> t;
    void* inst = nullptr;
    void* scratch = nullptr;
    int64_t sb = 0;
    int rank = 0, world = 1;
};
struct Work {   // A, or one B row count
    std::string name;
    int64_t M = 0;
    int L = 0;
    std::vector<std::vector<Call>> layer;
    void* dense_y = nullptr; size_t dense_yb = 0;   // OVCHECK: the last dense GEMM's output
    void* chain_y = nullptr; size_t chain_yb = 0;   // OVCHECK: the routed down's output
    double live = 0;                                // live experts a layer (mean)
};

// Shared between the rank threads.
class Barrier {
public:
    explicit Barrier(int n) : n_(n) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        const int g = gen_;
        if (++in_ == n_) { in_ = 0; ++gen_; cv_.notify_all(); return; }
        cv_.wait(lk, [&] { return gen_ != g; });
    }
private:
    std::mutex m_;
    std::condition_variable cv_;
    int n_, in_ = 0, gen_ = 0;
};
Barrier g_bar(kWorld);
std::atomic<long long> g_beat{ 0 };      // the watchdog's heartbeat (steady ms)
std::atomic<const char*> g_phase{ "setup" };
std::atomic<int> g_fail{ 0 };
long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void beat(const char* ph) { g_phase.store(ph); g_beat.store(now_ms()); }

// Routings and the plan, built once on the main thread (every rank runs the same tokens: TP).
struct Plan {
    std::vector<Route> ra;                 // [layer] A's
    std::map<int64_t, std::vector<Route>> rb;   // [MB][layer]
};

// Per rank: results the main thread prints.
struct ArmRes { double a_us = 0, nb = 0, b_us = 0, s = 0; int npass = 0; };
struct RankOut {
    std::vector<double> ta;                          // [rep]
    std::map<int64_t, std::vector<double>> tb;       // [MB][rep]
    std::map<std::string, std::vector<ArmRes>> arm;  // [arm][rep]
    std::string ovcheck, queues;
    size_t dev_bytes = 0;
};
std::vector<RankOut> g_out(kWorld);
// Rank 0's per-rep estimates, read by every rank so the B pass counts agree (an all-reduce needs its peers).
std::map<std::string, int> g_npass;
std::mutex g_np_mu;

struct Ctx {
    int rank = 0, dev = 0;
    const Opt* o = nullptr;
    const Plan* pl = nullptr;
    std::vector<void*> allocs;
    size_t bytes = 0;
    void* dalloc(size_t n) {
        void* p = nullptr;
        n = (n + 255) / 256 * 256;
        ck(hipMalloc(&p, n < 256 ? 256 : n), "hipMalloc", dev);
        allocs.push_back(p);
        bytes += n;
        return p;
    }
    void put(void* d, const void* h, size_t n) { ck(hipMemcpy(d, h, n, hipMemcpyHostToDevice), "copy", dev); }
};

// A 64 MiB block of random bytes, tiled into every weight buffer (the trellis decode and the GGUF dequant do the same
// work whatever the bits; q6_K's d is set finite below).
std::vector<uint8_t> g_rblk;
void fill_random(Ctx& c, void* d, size_t n) {
    for (size_t at = 0; at < n; at += g_rblk.size()) c.put((char*)d + at, g_rblk.data(), std::min(g_rblk.size(), n - at));
}
void* f16_const(Ctx& c, size_t n, float v) {
    std::vector<uint16_t> h(n, rad_f32_to_f16(v));
    void* d = c.dalloc(n * 2);
    c.put(d, h.data(), n * 2);
    return d;
}
void* table_of(Ctx& c, void* base, int64_t n, size_t stride) {
    std::vector<void*> p((size_t)n);
    for (int64_t i = 0; i < n; ++i) p[(size_t)i] = (char*)base + (size_t)i * stride;
    void* d = c.dalloc(p.size() * sizeof(void*));
    c.put(d, p.data(), p.size() * sizeof(void*));
    return d;
}

const char* kdt_of(int bits) {
    static const char* const k[] = { "exl3k0", "exl3k1", "exl3k2", "exl3k3", "exl3k4", "exl3k5", "exl3k6" };
    return k[bits];
}

// The experts' planes of one pool (perfmoe --chain's tables): gate_up codes / suh / suh.1 / svh / down's suh, down codes / suh / svh.
struct Pool {
    std::array<std::array<RadTensor, 5>, 2> g;
    std::array<std::array<RadTensor, 3>, 2> d;
    std::array<void*, 16> w{};   // the dense weights (--dense-shapes: at most 16)
};
Pool make_pool(Ctx& c, const int64_t* Rg, const int64_t* Cg, const int64_t* Rd, const int64_t* Cd, const int64_t* nec) {
    const Opt& o = *c.o;
    Pool P;
    for (int k = 0; k < 2; ++k) {
        const size_t cbg = (size_t)(Rg[k] * Cg[k] * o.bits / 8), cbd = (size_t)(Rd[k] * Cd[k] * o.bits / 8);
        void* gc = c.dalloc(cbg * (size_t)nec[k]); fill_random(c, gc, cbg * (size_t)nec[k]);
        void* dc = c.dalloc(cbd * (size_t)nec[k]); fill_random(c, dc, cbd * (size_t)nec[k]);
        void* suh = f16_const(c, (size_t)(Cg[k] * nec[k]), 0.01f);
        void* suh1 = f16_const(c, (size_t)(Cg[k] * nec[k]), 0.01f);
        void* svh = f16_const(c, (size_t)(Rg[k] * nec[k]), 1.0f);
        void* dsuh = f16_const(c, (size_t)(Cd[k] * nec[k]), 0.01f);
        void* dsvh = f16_const(c, (size_t)(Rd[k] * nec[k]), 1.0f);
        P.g[(size_t)k][0] = TAB(table_of(c, gc, nec[k], cbg), kDtTrellis, { nec[k], Rg[k], Cg[k] });
        P.g[(size_t)k][1] = TAB(table_of(c, suh, nec[k], (size_t)Cg[k] * 2), RAD_F16, { nec[k], 1, Cg[k] });
        P.g[(size_t)k][2] = TAB(table_of(c, suh1, nec[k], (size_t)Cg[k] * 2), RAD_F16, { nec[k], 1, Cg[k] });
        P.g[(size_t)k][3] = TAB(table_of(c, svh, nec[k], (size_t)Rg[k] * 2), RAD_F16, { nec[k], Rg[k], 1 });
        P.g[(size_t)k][4] = TAB(table_of(c, dsuh, nec[k], (size_t)Cd[k] * 2), RAD_F16, { nec[k], 1, Cd[k] });
        P.d[(size_t)k][0] = TAB(table_of(c, dc, nec[k], cbd), kDtTrellis, { nec[k], Rd[k], Cd[k] });
        P.d[(size_t)k][1] = P.g[(size_t)k][4];
        P.d[(size_t)k][2] = TAB(table_of(c, dsvh, nec[k], (size_t)Rd[k] * 2), RAD_F16, { nec[k], Rd[k], 1 });
    }
    if (o.dense)
        for (int i = 0; i < kNDense; ++i) {
            const Dense& s = kDense[i];
            const int64_t rb = s.K / 256 * kQ6Blk;
            std::vector<uint8_t> h((size_t)(s.N * rb));
            for (size_t j = 0; j < h.size(); ++j) h[j] = g_rblk[j % g_rblk.size()];
            const uint16_t d16 = 0x211F;   // ~0.0102: finite products (r4d_selftest ds_gg_perf)
            for (int64_t n = 0; n < s.N; ++n)
                for (int64_t b = 0; b < s.K / 256; ++b) std::memcpy(&h[(size_t)(n * rb + b * kQ6Blk + 208)], &d16, 2);
            P.w[(size_t)i] = c.dalloc(h.size());
            c.put(P.w[(size_t)i], h.data(), h.size());
        }
    return P;
}

// The row parameter vectors live as long as the run (Call keeps a pointer).
std::vector<std::vector<RadParam>*> g_params;
std::mutex g_params_mu;
const std::vector<RadParam>* keep(std::vector<RadParam> p) {
    std::lock_guard<std::mutex> lk(g_params_mu);
    g_params.push_back(new std::vector<RadParam>(std::move(p)));
    return g_params.back();
}

void row_setup(Ctx& c, Call& k, int rank, int world) {
    k.rank = rank; k.world = world;
    if (k.row->init) {
        const int rc = k.row->init(k.p->data(), (int)k.p->size(), rank, world, &k.inst);
        if (rc != RAD_OK) die(std::string(k.row->name) + " init rc " + std::to_string(rc) + " (rank " + std::to_string(rank) + ")");
    }
    if (k.row->scratch) {
        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = k.t.data(); a.n_t = (int)k.t.size(); a.p = k.p->data(); a.n_p = (int)k.p->size();
        a.instance = k.inst; a.rank = rank; a.world_size = world;
        k.sb = k.row->scratch(&a);
        if (k.sb > 0) k.scratch = c.dalloc((size_t)k.sb);
    }
}

// The rows and parameter sets of M rows, as the served deepseek41 declares them: the routed experts exl3r4d _m16 up to 16
// rows, lever P's _dk pair at 17..48 (RADIANCE_DS41_MOE1=1: one MoE pass, the decode kernels over it), _pf above; the
// all-reduce one-shot below 524,289 elements, two-shot from it (the rows' own numel bounds, the two-shot's higher priority);
// gemm_nt_ggml's gemv row up to 16 rows, its tiled row above (both through gg_gemm_run's served forms by the env).
const char* exl3_band(int64_t M) { return M <= 16 ? "_m16" : M <= 48 ? "_dk" : "_pf"; }
const char* ar_row(int64_t numel) { return numel < 524289 ? "ar_oneshot_4rank_exact" : "ar_twoshot_4rank_exact"; }
struct Rows {
    const RadKernelInfo *gr = nullptr, *dr = nullptr, *arr = nullptr, *ggr = nullptr;
    std::vector<RadParam> g, d, ar;
    std::vector<std::vector<RadParam>> gg;
};
Rows rows_for(const Opt& o, int64_t M) {
    Rows R;
    int64_t Rg[2], Cg[2], Rd[2], Cd[2];
    for (int k = 0; k < 2; ++k) { Rg[k] = 2 * o.widths[(size_t)k]; Cg[k] = o.hidden; Rd[k] = o.hidden; Cd[k] = o.widths[(size_t)k]; }
    const std::string band = exl3_band(M);
    R.gr = row_of(0, o.exl3, ("exl3r4d_moe_gate_up" + band).c_str());
    R.dr = row_of(0, o.exl3, ("exl3r4d_moe_down" + band).c_str());
    R.arr = row_of(1, o.r4d, ar_row(M * o.hidden));
    R.ggr = row_of(1, o.r4d, M <= 16 ? "gemm_nt_ggml_qk256_gemv" : "gemm_nt_ggml_qk256_tiled");
    const char* kdt = kdt_of(o.bits);
    R.g = { PI("M", M), PI("N", Rg[0]), PI("K", Cg[0]), PI("n_expert", o.E), PI("top_k", o.topk), PS("dtype", kdt),
            PS("codebook", "mul1"), PI("N_odd", Rg[1]), PI("K_odd", Cg[1]), PI("h_rot", o.h_rot), PI("a_rot", 0) };
    R.d = { PI("M", M), PI("N", Rd[0]), PI("K", Cd[0]), PI("n_expert", o.E), PI("top_k", o.topk), PS("dtype", kdt),
            PS("codebook", "mul1"), PI("N_odd", Rd[1]), PI("K_odd", Cd[1]), PI("h_rot", o.h_rot) };
    R.ar = { PI("world_size", kWorld), PI("numel", M * o.hidden), PS("dtype", "bf16"), PI("exact", 1) };
    for (int i = 0; i < kNDense; ++i)
        R.gg.push_back({ PI("M", M), PI("N", kDense[i].N), PI("K", kDense[i].K), PS("type", "q6_K"), PS("dtype", "bf16") });
    return R;
}
// A row's declared constraints against a parameter set (what the engine's selector checks before it picks the row): "" or
// the first that fails. A ranged parameter is checked at both ends.
std::string constraints_fail(const RadKernelInfo* k, const std::vector<RadParam>& p) {
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint& c = k->constraints[i];
        if (!std::strncmp(c.key, "device.", 7)) continue;   // a device fact (the engine fills it from the card), not a parameter
        const RadParam* q = nullptr;
        for (const RadParam& x : p) if (!std::strcmp(x.key, c.key)) q = &x;
        if (!q) return std::string("parameter ") + c.key + " absent";
        if (c.op == RAD_C_IN) {   // a set of strings, or of integers written out
            const std::string v = q->kind == RAD_P_STR && q->sval ? std::string(q->sval)
                                : q->kind == RAD_P_INT ? std::to_string(q->ival) : std::string();
            const std::string set = std::string(" ") + (c.sval ? c.sval : "") + " ";
            if (v.empty() || set.find(" " + v + " ") == std::string::npos) return std::string(c.key) + " " + v + " not in {" + (c.sval ? c.sval : "") + "}";
            continue;
        }
        if (q->kind != RAD_P_INT && q->kind != RAD_P_RANGE) return std::string(c.key) + " is not an integer";
        for (long long v : { q->ival, q->kind == RAD_P_RANGE ? q->ihi : q->ival }) {
            const bool ok = c.op == RAD_C_EQ ? v == c.ival : c.op == RAD_C_LE ? v <= c.ival : c.op == RAD_C_GE ? v >= c.ival
                          : c.op == RAD_C_DIV ? (c.ival != 0 && v % c.ival == 0) : false;
            if (!ok) return std::string(c.key) + " " + std::to_string(v) + " fails " +
                            (c.op == RAD_C_EQ ? "==" : c.op == RAD_C_LE ? "<=" : c.op == RAD_C_GE ? ">=" : c.op == RAD_C_DIV ? "divisible by " : "?") +
                            std::to_string(c.ival);
        }
    }
    return "";
}

// Work of M rows over L layers on pools [p0, p0 + P): a layer's calls in issue order.
Work make_work(Ctx& c, const std::string& name, int64_t M, int L, const std::vector<Route>& routes, std::vector<Pool>& pools,
               int p0, std::map<std::string, Call>& inst_by_key) {
    const Opt& o = *c.o;
    Work W;
    W.name = name; W.M = M; W.L = L;
    const int64_t Tr = M * o.topk;
    const int64_t wmax = std::max(o.widths[0], o.widths[1]);
    const Rows R = rows_for(o, M);
    const RadKernelInfo* gr = R.gr;
    const RadKernelInfo* dr = R.dr;
    const std::vector<RadParam>* gp = keep(R.g);
    const std::vector<RadParam>* dp = keep(R.d);
    const Exl3Positions* gpos = exl3_positions(EXL3_GATE_UP);
    const Exl3Positions* dpos = exl3_positions(EXL3_DOWN);
    // activations (one set a Work: its layers run in order on one stream); a generator of this rank's own (threads)
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ (uint64_t)(c.rank + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
    auto lr = [&rs]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    std::vector<uint16_t> hx((size_t)(M * 8192));
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = rad_f32_to_f16((float)((int)(lr() % 2001) - 1000) / 4000.0f);
    void* xf16 = c.dalloc((size_t)(M * o.hidden * 2)); c.put(xf16, hx.data(), (size_t)(M * o.hidden * 2));
    std::vector<uint16_t> hb((size_t)(M * 8192));
    for (size_t i = 0; i < hb.size(); ++i) { const float f = (float)((int)(lr() % 2001) - 1000) / 4000.0f; uint32_t u; std::memcpy(&u, &f, 4); hb[i] = (uint16_t)(u >> 16); }
    void* xbf = c.dalloc((size_t)(M * 8192 * 2)); c.put(xbf, hb.data(), hb.size() * 2);
    void* ybf = c.dalloc((size_t)(M * 8192 * 2));
    void* h = c.dalloc((size_t)(Tr * wmax * 2));
    void* y = c.dalloc((size_t)(Tr * o.hidden * 2));
    void* arb = c.dalloc((size_t)(M * o.hidden * 2));
    ck(hipMemset(arb, 0, (size_t)(M * o.hidden * 2)), "memset", c.dev);
    W.dense_y = ybf; W.dense_yb = (size_t)(M * kDense[kNDense - 1].N * 2);
    W.chain_y = y; W.chain_yb = (size_t)(Tr * o.hidden * 2);
    // the all-reduce: the engine's band (one-shot up to 128 rows), an instance a Work (libr4d keys on numel: distinct M, distinct set)
    const char* arn = R.arr->name;
    Call ar;
    ar.row = R.arr;
    ar.p = keep(R.ar);
    ar.t = { T(arb, RAD_BF16, { M * o.hidden }) };
    {
        const std::string key = std::string(arn) + ":" + std::to_string(M);
        auto it = inst_by_key.find(key);
        if (it == inst_by_key.end()) { row_setup(c, ar, c.rank, kWorld); inst_by_key[key] = ar; }
        else ar = it->second;
    }
    // the dense rows (no instance): gemv up to 16 rows, else tiled (both go through gg_gemm_run's served forms)
    const RadKernelInfo* gg = R.ggr;
    std::vector<const std::vector<RadParam>*> ggp;
    for (int i = 0; i < kNDense; ++i) ggp.push_back(keep(R.gg[(size_t)i]));
    Call G0, D0;   // the expert rows' instance / scratch, shared by the Work's layers (as the engine: one resolved row)
    bool first = true;
    double live = 0;
    for (int l = 0; l < L; ++l) {
        const Route& rt = routes[(size_t)l];
        live += (double)rt.live;
        Pool& P = pools[(size_t)(p0 + l % o.pools)];
        void* ds = c.dalloc((size_t)Tr * 4); c.put(ds, rt.sorted.data(), (size_t)Tr * 4);
        void* doff = c.dalloc((size_t)(o.E + 1) * 4); c.put(doff, rt.off.data(), (size_t)(o.E + 1) * 4);
        const RadTensor st = T(ds, RAD_I32, { Tr }), ot = T(doff, RAD_I32, { o.E + 1 });
        std::vector<Call> calls;
        auto dense = [&](int i) {
            Call k;
            k.row = gg; k.p = ggp[(size_t)i];
            const int64_t rb = kDense[i].K / 256 * kQ6Blk;
            k.t = { T(xbf, RAD_BF16, { M, kDense[i].K }), T(P.w[(size_t)i], RAD_U8, { kDense[i].N, rb }), T(ybf, RAD_BF16, { M, kDense[i].N }) };
            calls.push_back(k);
        };
        if (o.dense) for (int i = 0; i < g_dense_ar; ++i) dense(i);
        calls.push_back(ar);
        if (o.dense) for (int i = g_dense_ar; i < kNDense; ++i) dense(i);
        Call G, D;
        G.row = gr; G.p = gp;
        G.t.assign((size_t)gpos->n_opd, RadTensor{});
        G.t[(size_t)gpos->x] = T(xf16, RAD_F16, { M, o.hidden }); G.t[(size_t)gpos->sorted] = st; G.t[(size_t)gpos->offset] = ot;
        G.t[(size_t)gpos->y] = T(h, RAD_F16, { Tr, wmax });
        D.row = dr; D.p = dp;
        D.t.assign((size_t)dpos->n_opd, RadTensor{});
        D.t[(size_t)dpos->x] = T(h, RAD_F16, { Tr, wmax }); D.t[(size_t)dpos->sorted] = st; D.t[(size_t)dpos->offset] = ot;
        D.t[(size_t)dpos->y] = T(y, RAD_BF16, { Tr, o.hidden });
        for (int k = 0; k < 2; ++k) {
            for (int q = 0; q < 5; ++q) G.t[(size_t)gpos->plane[k][q]] = P.g[(size_t)k][(size_t)q];
            const int dk[3] = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SVH };
            for (int q = 0; q < 3; ++q) D.t[(size_t)dpos->plane[k][dk[q]]] = P.d[(size_t)k][(size_t)q];
        }
        if (first) { row_setup(c, G, 0, 1); row_setup(c, D, 0, 1); G0 = G; D0 = D; first = false; }
        G.inst = G0.inst; G.scratch = G0.scratch; G.sb = G0.sb;
        D.inst = D0.inst; D.scratch = D0.scratch; D.sb = D0.sb;
        calls.push_back(G);
        calls.push_back(D);
        calls.push_back(ar);
        W.layer.push_back(calls);
    }
    W.live = live / L;
    return W;
}

int launch(Ctx& c, Call& k, hipStream_t s) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = k.t.data(); a.n_t = (int)k.t.size(); a.p = k.p->data(); a.n_p = (int)k.p->size();
    a.instance = k.inst; a.scratch = k.scratch; a.scratch_bytes = k.sb; a.rank = k.rank; a.world_size = k.world;
    const int rc = k.row->launch(&a, (RadStream)s);
    if (rc != RAD_OK && rc != RAD_LINK_ENDED) {
        std::printf("q2bench: rank %d: %s launch rc %d\n", c.rank, k.row->name, rc);
        g_fail.store(1);
        return rc;
    }
    return RAD_OK;
}
void run_work(Ctx& c, Work& W, hipStream_t s) {
    for (auto& L : W.layer)
        for (auto& k : L) launch(c, k, s);
}

std::string kfd_queues() {   // every KFD process's queues, by gpu id (inside a container getpid() is not KFD's host pid, so all are listed)
    const std::string root = "/sys/class/kfd/kfd/proc";
    DIR* pd = opendir(root.c_str());
    if (!pd) return "unreadable (" + root + ")";
    std::string s;
    int procs = 0;
    while (dirent* p = readdir(pd)) {
        if (p->d_name[0] == '.') continue;
        const std::string dir = root + "/" + p->d_name + "/queues";
        DIR* d = opendir(dir.c_str());
        if (!d) continue;
        std::map<std::string, int> by;
        int n = 0;
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            FILE* f = std::fopen((dir + "/" + e->d_name + "/gpuid").c_str(), "r");
            char g[64] = "?";
            if (f) { if (!std::fgets(g, sizeof g, f)) std::strcpy(g, "?"); std::fclose(f); }
            g[std::strcspn(g, "\n")] = 0;
            ++by[g]; ++n;
        }
        closedir(d);
        ++procs;
        s += std::string(s.empty() ? "" : "; ") + "pid " + p->d_name + " " + std::to_string(n) + " queues:";
        for (auto& kv : by) s += " gpu " + kv.first + " x" + std::to_string(kv.second);
    }
    closedir(pd);
    return std::to_string(procs) + " KFD process(es): " + s;
}

// The B passes inside A's window [0, a1] (ms from A's start event): a pass counts by the share of its own span inside the
// window; `inside` gets the spans (us) of the passes wholly inside. S = T_A / A' + nB x T_B / A' (all in the same unit).
double window_passes(double a1, const std::vector<double>& s0, const std::vector<double>& s1, std::vector<double>* inside) {
    double nb = 0;
    for (size_t i = 0; i < s0.size(); ++i) {
        const double lo = std::max(0.0, s0[i]), hi = std::min(a1, s1[i]);
        if (s1[i] > s0[i] && hi > lo) nb += (hi - lo) / (s1[i] - s0[i]);
        if (inside && s0[i] >= 0 && s1[i] <= a1) inside->push_back((s1[i] - s0[i]) * 1000.0);
    }
    return nb;
}
double s_of(double ta, double a_ov, double nb, double tb) { return ta / a_ov + nb * tb / a_ov; }

// --check: the window arithmetic on made-up timelines (host only)
int host_check() {
    int bad = 0;
    auto expect = [&](const char* what, double got, double want) {
        const bool ok = std::fabs(got - want) < 1e-9;
        bad += !ok;
        std::printf("CHECK %-58s %s (got %.6f, want %.6f)\n", what, ok ? "ok" : "FAILED", got, want);
    };
    {   // serial: B waits for A entirely -> no pass inside, S = T_A / A' = 1 when A' = T_A
        std::vector<double> s0 = { 100, 110 }, s1 = { 110, 120 };
        expect("serial (B after A): passes inside", window_passes(100, s0, s1, nullptr), 0.0);
        expect("serial: S", s_of(100, 100, 0, 10), 1.0);
    }
    {   // perfect overlap: A unchanged (100), ten 10-ms B passes inside -> S = 2
        std::vector<double> s0, s1;
        for (int i = 0; i < 12; ++i) { s0.push_back(10.0 * i); s1.push_back(10.0 * i + 10); }
        std::vector<double> in;
        expect("perfect overlap: passes inside", window_passes(100, s0, s1, &in), 10.0);
        expect("perfect overlap: passes wholly inside", (double)in.size(), 10.0);
        expect("perfect overlap: S", s_of(100, 100, 10, 10), 2.0);
    }
    {   // time-sliced: A 100 -> 150, B 10 -> 30 a pass inside, a pass straddling the end counts by its share
        std::vector<double> s0 = { 0, 30, 60, 90, 120, 150 }, s1 = { 30, 60, 90, 120, 160, 170 };
        expect("time-sliced: passes inside (4 + 30/40 straddling)", window_passes(150, s0, s1, nullptr), 4.75);
        expect("time-sliced: S", s_of(100, 150, 4.75, 10), 100.0 / 150 + 47.5 / 150);
    }
    {   // a pass that started before A (negative start) counts by its share too
        std::vector<double> s0 = { -5 }, s1 = { 5 };
        expect("a pass straddling A's start", window_passes(100, s0, s1, nullptr), 0.5);
    }
    std::printf("CHECK %d failed\n", bad);
    return bad ? 1 : 0;
}

double med(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void rank_main(Ctx* cp) {
    Ctx& c = *cp;
    const Opt& o = *c.o;
    ck(hipSetDevice(c.dev), "hipSetDevice", c.dev);
    int least = 0, greatest = 0;
    ck(hipDeviceGetStreamPriorityRange(&least, &greatest), "priority range", c.dev);
    hipStream_t qa, qal, qb, qbh;
    ck(hipStreamCreateWithPriority(&qa, hipStreamNonBlocking, 0), "stream qa", c.dev);
    ck(hipStreamCreateWithPriority(&qal, hipStreamNonBlocking, least), "stream qal", c.dev);
    ck(hipStreamCreateWithPriority(&qb, hipStreamNonBlocking, 0), "stream qb", c.dev);
    ck(hipStreamCreateWithPriority(&qbh, hipStreamNonBlocking, greatest), "stream qbh", c.dev);
    if (c.rank == 0) std::printf("q2bench: stream priorities: least %d, greatest %d (qa / qb 0, qal %d, qbh %d)\n", least, greatest, least, greatest);
    const int64_t nec[2] = { (o.E + 1) / 2, o.E / 2 };
    int64_t Rg[2], Cg[2], Rd[2], Cd[2];
    for (int k = 0; k < 2; ++k) { Rg[k] = 2 * o.widths[(size_t)k]; Cg[k] = o.hidden; Rd[k] = o.hidden; Cd[k] = o.widths[(size_t)k]; }
    std::vector<Pool> pools;
    for (int q = 0; q < 2 * o.pools; ++q) pools.push_back(make_pool(c, Rg, Cg, Rd, Cd, nec));
    beat("setup: pools");
    // the all-reduce instances rendezvous across ranks inside init (libr4d: arrival order, a 60-s timeout): every rank enters
    // them together, after its pools, in the same order
    g_bar.wait();
    std::map<std::string, Call> inst;
    Work A = make_work(c, "A", o.MA, o.LA, c.pl->ra, pools, 0, inst);
    std::map<int64_t, Work> B;
    for (int64_t m : o.MB) B[m] = make_work(c, "B" + std::to_string(m), m, o.LB, c.pl->rb.at(m), pools, o.pools, inst);
    ck(hipDeviceSynchronize(), "setup sync", c.dev);
    g_out[(size_t)c.rank].dev_bytes = c.bytes;
    if (c.rank == 0) {
        std::printf("q2bench: rank 0 holds %.2f GiB; A: %lld rows x %d layers, %zu launches a layer, %.1f live experts a layer;", c.bytes / 1073741824.0,
                    (long long)A.M, A.L, A.layer[0].size(), A.live);
        for (auto& kv : B) std::printf(" %s: %d layers, %.1f live;", kv.second.name.c_str(), kv.second.L, kv.second.live);
        std::printf("\n");
    }
    g_bar.wait();
    beat("warm");
    // events: a pool, reused each run
    std::vector<hipEvent_t> ev(512);
    for (auto& e : ev) ck(hipEventCreate(&e), "event", c.dev);
    auto ms = [&](hipEvent_t a, hipEvent_t b) { float x = 0; ck(hipEventElapsedTime(&x, a, b), "elapsed", c.dev); return (double)x; };
    // WARM: every work alone, back to back, until --warm-ms (clocks up; first-use code objects loaded)
    {
        // every rank runs the same rounds (each round's all-reduces need their peers): only rank 0's clock ends the loop
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            run_work(c, A, qa); ck(hipStreamSynchronize(qa), "warm A", c.dev);
            for (auto& kv : B) { run_work(c, kv.second, qb); ck(hipStreamSynchronize(qb), "warm B", c.dev); }
            run_work(c, A, qal); ck(hipStreamSynchronize(qal), "warm A low", c.dev);
            for (auto& kv : B) { run_work(c, kv.second, qbh); ck(hipStreamSynchronize(qbh), "warm B high", c.dev); }
            // every rank must leave the loop after the same number of rounds: rank 0 decides
            static std::atomic<int> stop{ 0 };
            if (c.rank == 0 && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= o.warm_ms) stop.store(1);
            g_bar.wait();
            const int s = stop.load();
            g_bar.wait();
            if (s) break;
        }
    }
    if (g_fail.load()) { g_bar.wait(); return; }
    RankOut& out = g_out[(size_t)c.rank];
    std::vector<char> ref_d, ref_y;
    auto snap = [&](std::vector<char>& d, std::vector<char>& y) {
        d.resize(A.dense_yb); y.resize(A.chain_yb);
        ck(hipMemcpy(d.data(), A.dense_y, A.dense_yb, hipMemcpyDeviceToHost), "copy", c.dev);
        ck(hipMemcpy(y.data(), A.chain_y, A.chain_yb, hipMemcpyDeviceToHost), "copy", c.dev);
    };
    for (int rep = 0; rep < o.reps; ++rep) {
        // A alone
        g_bar.wait(); beat("A alone");
        ck(hipEventRecord(ev[0], qa), "record", c.dev);
        run_work(c, A, qa);
        ck(hipEventRecord(ev[1], qa), "record", c.dev);
        ck(hipEventSynchronize(ev[1]), "sync", c.dev);
        out.ta.push_back(ms(ev[0], ev[1]) * 1000.0);
        if (rep == 0) snap(ref_d, ref_y);
        // B alone: 6 passes back to back, the median pass
        for (auto& kv : B) {
            g_bar.wait(); beat("B alone");
            ck(hipEventRecord(ev[0], qb), "record", c.dev);
            for (int i = 0; i < 6; ++i) { run_work(c, kv.second, qb); ck(hipEventRecord(ev[(size_t)i + 1], qb), "record", c.dev); }
            ck(hipEventSynchronize(ev[6]), "sync", c.dev);
            std::vector<double> v;
            for (int i = 0; i < 6; ++i) v.push_back(ms(ev[(size_t)i], ev[(size_t)i + 1]) * 1000.0);
            out.tb[kv.first].push_back(med(v));
        }
        // the overlap arms
        for (const std::string& arm : o.arms) {
            const char kind = arm[0];
            const int64_t m = std::atoll(arm.c_str() + 1);
            Work& Bw = B.at(m);
            hipStream_t sa = kind == 'l' || kind == 'x' ? qal : qa, sb = kind == 'h' || kind == 'x' ? qbh : qb;
            g_bar.wait();
            if (c.rank == 0) {
                const double ta = out.ta.back(), tb = out.tb.at(m).back();
                std::lock_guard<std::mutex> lk(g_np_mu);
                g_npass[arm] = std::min(250, (int)std::ceil(3.0 * ta / std::max(tb, 1.0)) + 2);
            }
            g_bar.wait(); beat("overlap");
            int np;
            { std::lock_guard<std::mutex> lk(g_np_mu); np = g_npass[arm]; }
            ck(hipEventRecord(ev[0], sa), "record", c.dev);
            run_work(c, A, sa);
            ck(hipEventRecord(ev[1], sa), "record", c.dev);
            for (int i = 0; i < np; ++i) {
                if (i >= 3) ck(hipEventSynchronize(ev[(size_t)(2 + 2 * (i - 3) + 1)]), "pace", c.dev);
                ck(hipEventRecord(ev[(size_t)(2 + 2 * i)], sb), "record", c.dev);
                run_work(c, Bw, sb);
                ck(hipEventRecord(ev[(size_t)(2 + 2 * i + 1)], sb), "record", c.dev);
                beat("overlap");
            }
            ck(hipStreamSynchronize(sa), "sync A", c.dev);
            ck(hipStreamSynchronize(sb), "sync B", c.dev);
            ArmRes r;
            const double a1 = ms(ev[0], ev[1]);
            r.a_us = a1 * 1000.0;
            std::vector<double> inside, s0, s1;
            for (int i = 0; i < np; ++i) { s0.push_back(ms(ev[0], ev[(size_t)(2 + 2 * i)])); s1.push_back(ms(ev[0], ev[(size_t)(2 + 2 * i + 1)])); }
            r.nb = window_passes(a1, s0, s1, &inside);
            r.npass = np;
            r.b_us = med(inside);
            r.s = s_of(out.ta.back(), r.a_us, r.nb, out.tb.at(m).back());
            out.arm[arm].push_back(r);
            if (rep == 0 && &arm == &o.arms[0]) {
                if (c.rank == 0) out.queues = kfd_queues();
                std::vector<char> d, y;
                snap(d, y);
                out.ovcheck = (d == ref_d ? std::string("dense y equal") : std::string("dense y DIFFERS")) + ", " +
                              (y == ref_y ? "expert y equal" : "expert y DIFFERS");
            }
            if (g_fail.load()) break;
        }
        if (g_fail.load()) break;
    }
    g_bar.wait();
    beat("done");
}

void print_plan(const Opt& o) {
    const int64_t nec[2] = { (o.E + 1) / 2, o.E / 2 };
    double eb = 0, db = 0;
    for (int k = 0; k < 2; ++k) {
        const double w = (double)o.widths[(size_t)k];
        // codes (gate_up 2w x hidden, down hidden x w at K bits) + the f16 planes (suh, suh.1, svh 2w, down's suh w, down's svh)
        eb += (double)nec[k] * (3.0 * w * o.hidden * o.bits / 8 + 6.0 * o.hidden + 6.0 * w);
    }
    for (int i = 0; i < kNDense && o.dense; ++i) db += (double)dense_bytes(kDense[i]);
    const int lpl = (o.dense ? kNDense : 0) + 4;
    std::printf("PLAN q2bench: TP4 (ranks = HIP devices 0..3), rank-0 widths %lld/%lld on every rank, hidden %lld, %lld experts top-%lld, K %d, h_rot %d\n",
                (long long)o.widths[0], (long long)o.widths[1], (long long)o.hidden, (long long)o.E, (long long)o.topk, o.bits, o.h_rot);
    std::printf("PLAN a layer: %d launches (%s2 x exl3r4d, 2 x all-reduce); weights a pool %.1f MB experts + %.1f MB dense\n", lpl,
                o.dense ? (std::to_string(kNDense) + " x gemm_nt_ggml q6_K, ").c_str() : "", eb / 1e6, db / 1e6);
    if (o.dense) {
        std::printf("PLAN dense (q6_K):");
        for (int i = 0; i < kNDense; ++i) std::printf("%s %lldx%lld", i == g_dense_ar ? " | AR |" : "", (long long)kDense[(size_t)i].N, (long long)kDense[(size_t)i].K);
        std::printf("%s\n", g_dense_ar == kNDense ? " | AR" : "");
    }
    std::printf("PLAN pools: %d for A + %d for B, %.2f GB a card of weights\n", o.pools, o.pools, 2.0 * o.pools * (eb + db) / 1e9);
    std::printf("PLAN A: %lld rows x %d layers (band %s, all-reduce %s %lld elements)\n", (long long)o.MA, o.LA, exl3_band(o.MA),
                ar_row(o.MA * o.hidden), (long long)(o.MA * o.hidden));
    for (int64_t m : o.MB)
        std::printf("PLAN B%lld: %lld rows x %d layers (band %s, all-reduce %s %lld elements, dense %s)\n", (long long)m, (long long)m, o.LB,
                    exl3_band(m), ar_row(m * o.hidden), (long long)(m * o.hidden),
                    m <= 16 ? "gemm_nt_ggml_qk256_gemv" : "gemm_nt_ggml_qk256_tiled");
    for (const std::string& a : o.arms)
        std::printf("PLAN arm %s: A on %s, B%s on %s\n", a.c_str(), a[0] == 'l' || a[0] == 'x' ? "qal (lowest)" : "qa (normal)", a.c_str() + 1,
                    a[0] == 'h' || a[0] == 'x' ? "qbh (highest)" : "qb (normal)");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered into the run's file: a crash keeps the lines before it
    Opt o;
    auto list = [](const char* s) { std::vector<std::string> v; std::string x = s; size_t p = 0;
        while (p <= x.size()) { size_t q = x.find(',', p); if (q == std::string::npos) q = x.size(); if (q > p) v.push_back(x.substr(p, q - p)); p = q + 1; }
        return v; };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
        if (a == "--exl3") o.exl3 = nx();
        else if (a == "--r4d") o.r4d = nx();
        else if (a == "--fn4x" || a == "--fn4r") {
            const std::string v = nx();
            const size_t eq = v.find('=');
            if (eq == std::string::npos) die(a + " is NAME=VALUE");
            (a == "--fn4x" ? o.fn4x : o.fn4r).push_back({ v.substr(0, eq), std::atoi(v.c_str() + eq + 1) });
        }
        else if (a == "--MA") o.MA = std::atoll(nx());
        else if (a == "--LA") o.LA = std::atoi(nx());
        else if (a == "--LB") o.LB = std::atoi(nx());
        else if (a == "--MB") { o.MB.clear(); for (auto& s : list(nx())) o.MB.push_back(std::atoll(s.c_str())); }
        else if (a == "--experts") o.E = std::atoll(nx());
        else if (a == "--topk") o.topk = std::atoll(nx());
        else if (a == "--hidden") o.hidden = std::atoll(nx());
        else if (a == "--widths") { auto v = list(nx()); if (v.size() != 2) die("--widths a,b"); o.widths = { std::atoll(v[0].c_str()), std::atoll(v[1].c_str()) }; }
        else if (a == "--bits") o.bits = std::atoi(nx());
        else if (a == "--h-rot") o.h_rot = std::atoi(nx());
        else if (a == "--pools") o.pools = std::atoi(nx());
        else if (a == "--reps") o.reps = std::atoi(nx());
        else if (a == "--warm-ms") o.warm_ms = (float)std::atof(nx());
        else if (a == "--dense") o.dense = std::atoi(nx()) != 0;
        else if (a == "--dense-shapes") {   // N:K,... (q6_K stand-ins), replacing the DeepSeek list
            kDense.clear();
            for (auto& v : list(nx())) {
                const size_t c2 = v.find(':');
                if (c2 == std::string::npos) die("--dense-shapes N:K,...");
                const int64_t N = std::atoll(v.c_str()), K = std::atoll(v.c_str() + c2 + 1);
                if (N <= 0 || K <= 0 || K % 256 || N > 8192 || K > 8192) die("--dense-shapes: N <= 8192, K a multiple of 256 <= 8192: " + v);
                kDense.push_back({ "dense", N, K });
            }
            if (kDense.empty() || kDense.size() > 16) die("--dense-shapes: 1..16 shapes");
            kNDense = (int)kDense.size();
            if (g_dense_ar > kNDense) g_dense_ar = kNDense;
        }
        else if (a == "--dense-ar") { g_dense_ar = std::atoi(nx()); if (g_dense_ar < 0 || g_dense_ar > kNDense) die("--dense-ar is 0..the dense shapes"); }
        else if (a == "--arms") o.arms = list(nx());
        else if (a == "--plan") o.plan = true;
        else if (a == "--rows-only") o.rows_only = true;
        else if (a == "--check") return host_check();
        else if (a == "--rows") {   // --rows LIB PREFIX: the rows of a library whose name starts with PREFIX, with their constraints
            const std::string lib = nx(), pre = nx();
            void* hh = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!hh) die("dlopen " + lib + ": " + dlerror());
            auto count = (int (*)())dlsym(hh, "rad_kernel_count");
            auto at = (const RadKernelInfo* (*)(int))dlsym(hh, "rad_kernel_at");
            if (!count || !at) die(lib + " is not a radiance kernel library");
            for (int r = 0; r < count(); ++r) {
                const RadKernelInfo* k = at(r);
                if (!k || !k->name || std::strncmp(k->name, pre.c_str(), pre.size())) continue;
                std::printf("ROW %s op %s prio %d:", k->name, k->op ? k->op : "?", k->priority);
                static const char* const opn[] = { "==", "<=", ">=", "%", "in" };
                for (int i = 0; i < k->n_constraints; ++i) {
                    const RadConstraint& c = k->constraints[i];
                    if (c.op == RAD_C_IN) std::printf(" %s in {%s}", c.key, c.sval ? c.sval : "");
                    else std::printf(" %s %s %lld", c.key, c.op >= 0 && c.op < 4 ? opn[c.op] : "?", c.ival);
                }
                std::printf("\n");
            }
            return 0;
        }
        else die("unknown argument " + a);
    }
    if (o.bits < 2 || o.bits > 6) die("--bits is 2..6");
    if (o.widths[0] % 128 || o.widths[1] % 128 || o.widths[0] <= 0 || o.widths[1] <= 0) die("--widths are multiples of 128");
    if (o.MA <= 0 || o.LA <= 0 || o.LB <= 0 || o.pools < 1 || o.reps < 1 || o.topk < 1 || o.topk > o.E) die("sizes must be positive (topk <= experts)");
    if (o.hidden % 256) die("--hidden is a multiple of 256");
    for (const std::string& a : o.arms) {
        if (a.size() < 2 || (a[0] != 'n' && a[0] != 'h' && a[0] != 'l' && a[0] != 'x')) die("an arm is n|h|l|x followed by a B row count: " + a);
        const int64_t m = std::atoll(a.c_str() + 1);
        if (std::find(o.MB.begin(), o.MB.end(), m) == o.MB.end()) die("arm " + a + " names a B row count not in --MB");
    }
    for (int64_t m : o.MB) if (m <= 0 || m == o.MA) die("--MB rows must be positive and differ from --MA (one all-reduce set a row count)");
    print_plan(o);
    if (o.plan) return 0;
    if (o.exl3.empty() || o.r4d.empty()) die("--exl3 <exl3r4d.so> and --r4d <libr4d.so> are required");
    {   // every row the configuration launches, resolved before any HIP call, its declared constraints and (exl3r4d) the
        // interface's own parameter parse checked on this host: a configuration the engine's selector would refuse fails here
        std::vector<int64_t> ms = o.MB;
        ms.push_back(o.MA);
        int bad = 0;
        for (int64_t m : ms) {
            const Rows R = rows_for(o, m);
            std::vector<std::pair<const RadKernelInfo*, const std::vector<RadParam>*>> calls = { { R.gr, &R.g }, { R.dr, &R.d }, { R.arr, &R.ar } };
            if (o.dense) for (const auto& g : R.gg) calls.push_back({ R.ggr, &g });
            std::string line;
            for (const auto& kv : calls) {
                std::string why = constraints_fail(kv.first, *kv.second);
                if (why.empty() && (kv.first == R.gr || kv.first == R.dr)) {
                    Exl3Geom geo;
                    const int rc = exl3_geom_parse(kv.second->data(), (int)kv.second->size(), kv.first == R.gr ? EXL3_GATE_UP : EXL3_DOWN,
                                                   exl3_geom_two_classes(kv.second->data(), (int)kv.second->size()), &geo);
                    if (rc != RAD_OK) why = "exl3_geom_parse rc " + std::to_string(rc);
                }
                if (!why.empty()) { ++bad; std::printf("q2bench: M %lld: %s REFUSES its parameters: %s\n", (long long)m, kv.first->name, why.c_str()); }
                if (line.find(kv.first->name) == std::string::npos) line += std::string(line.empty() ? "" : ", ") + kv.first->name;
            }
            std::printf("q2bench: M %lld rows (constraints + parse checked): %s\n", (long long)m, line.c_str());
        }
        if (bad) die(std::to_string(bad) + " row(s) refuse the configuration");
        if (o.rows_only) { std::printf("q2bench: --rows-only: every row resolved and accepted, no HIP call made\n"); return 0; }
    }
    for (int lib = 0; lib < 2; ++lib)
        for (const auto& f : lib ? o.fn4r : o.fn4x) {
            auto sw = (int (*)(const char*, int))dlsym(g_lib_h[lib], "rad_fn4_switch");
            auto val = (int (*)(const char*, int*))dlsym(g_lib_h[lib], "rad_fn4_value");
            int ap = -1;
            const int rc = sw ? sw(f.first.c_str(), f.second) : -999;
            if (val) (void)val(f.first.c_str(), &ap);
            std::printf("q2bench: %s fn4 %s = %d: rc %d, applied %d%s\n", lib ? "libr4d" : "exl3r4d", f.first.c_str(), f.second, rc, ap,
                        rc != 1 ? " (NOT SERVED by this library: the run goes on without it)" : "");
        }
    {   // the switches that pick the forms this bench times, as each library holds them now (compare with the engine's start)
        const char* const names[2][8] = { { "exl3_x3d", "exl3_dng", "exl3_xr", nullptr },
                                          { "ds41_wmk", "gg_pf", "gg_gemv", "gg_gemv_m16", "ar_xpush", "ds41_f32dec", "ds41_arcoded", nullptr } };
        std::string line = "Q2SW";
        for (int lib = 0; lib < 2; ++lib) {
            auto val = (int (*)(const char*, int*))dlsym(g_lib_h[lib], "rad_fn4_value");
            for (int i = 0; names[lib][i]; ++i) {
                int v = -99;
                const int rc = val ? val(names[lib][i], &v) : -999;
                line += std::string(" ") + names[lib][i] + "=" + (rc == 1 ? std::to_string(v) : std::string("n/a"));
            }
        }
        std::printf("%s\n", line.c_str());
    }
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < kWorld) die("needs 4 HIP devices (HIP_VISIBLE_DEVICES), found " + std::to_string(nd));
    for (int d = 0; d < kWorld; ++d) {
        hipDeviceProp_t p;
        ck(hipGetDeviceProperties(&p, d), "props", d);
        char bus[32] = { 0 };
        ck(hipDeviceGetPCIBusId(bus, sizeof bus, d), "bus id", d);
        std::printf("q2bench: rank %d = device %d pci %s %s, %d CUs\n", d, d, bus, p.gcnArchName, p.multiProcessorCount);
    }
    g_rblk.resize(64u << 20);
    for (size_t i = 0; i + 8 <= g_rblk.size(); i += 8) { const uint64_t v = rnd(); std::memcpy(&g_rblk[i], &v, 8); }
    Plan pl;
    for (int l = 0; l < o.LA; ++l) pl.ra.push_back(draw(o.MA, o.E, o.topk));
    for (int64_t m : o.MB) for (int l = 0; l < o.LB; ++l) pl.rb[m].push_back(draw(m, o.E, o.topk));
    std::vector<Ctx> ctx(kWorld);
    std::vector<std::thread> th;
    beat("setup");
    for (int r = 0; r < kWorld; ++r) { ctx[(size_t)r].rank = r; ctx[(size_t)r].dev = r; ctx[(size_t)r].o = &o; ctx[(size_t)r].pl = &pl; }
    for (int r = 0; r < kWorld; ++r) th.emplace_back(rank_main, &ctx[(size_t)r]);
    std::atomic<bool> done{ false };
    std::thread dog([&] {   // the watchdog: a phase without progress for 120 s (setup: 600 s) is a hang
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            const char* ph = g_phase.load();
            const long long lim = std::strncmp(ph, "setup", 5) ? 120000 : 600000;
            if (!done.load() && now_ms() - g_beat.load() > lim) {
                std::printf("Q2HANG no progress for %lld s in phase '%s'\n", lim / 1000, ph);
                std::fflush(stdout);
                std::_Exit(3);
            }
        }
    });
    for (auto& t : th) t.join();
    done.store(true);
    dog.join();
    if (g_fail.load()) { std::printf("q2bench: a launch failed\n"); return 1; }
    // REPORT: rank 0's clock; the ranks' spread of A alone as a sanity line
    const RankOut& r0 = g_out[0];
    for (int r = 0; r < kWorld; ++r)
        std::printf("Q2RANK %d: A alone median %.1f us, %.2f GiB on the card, OVCHECK %s\n", r, med(g_out[(size_t)r].ta),
                    g_out[(size_t)r].dev_bytes / 1073741824.0, g_out[(size_t)r].ovcheck.c_str());
    std::printf("Q2K %s\n", r0.queues.c_str());
    std::printf("Q2A A (%lld rows x %d layers) alone: median %.1f us over %d reps [", (long long)o.MA, o.LA, med(r0.ta), o.reps);
    for (double x : r0.ta) std::printf(" %.1f", x);
    std::printf(" ]\n");
    for (int64_t m : o.MB) {
        std::printf("Q2B B%lld (%d layers) alone: median %.1f us a pass [", (long long)m, o.LB, med(r0.tb.at(m)));
        for (double x : r0.tb.at(m)) std::printf(" %.1f", x);
        std::printf(" ]\n");
    }
    for (const std::string& arm : o.arms) {
        const auto& v = r0.arm.at(arm);
        const int64_t m = std::atoll(arm.c_str() + 1);
        std::vector<double> s, ax, bx, nb;
        for (size_t i = 0; i < v.size(); ++i) {
            s.push_back(v[i].s);
            ax.push_back(v[i].a_us / r0.ta[i]);
            bx.push_back(v[i].b_us / r0.tb.at(m)[i]);
            nb.push_back(v[i].nb);
            std::printf("Q2O %s rep %zu: A' %.1f us (x%.3f of alone %.1f), B passes inside A %.2f of %d issued, B' p50 %.1f us (x%.3f of alone %.1f), S %.3f\n",
                        arm.c_str(), i, v[i].a_us, ax.back(), r0.ta[i], v[i].nb, v[i].npass, v[i].b_us, bx.back(), r0.tb.at(m)[i], v[i].s);
        }
        std::printf("Q2S %-4s S median %.3f [%.3f, %.3f] | A x%.3f | B pass x%.3f | B passes inside A %.2f\n", arm.c_str(), med(s),
                    *std::min_element(s.begin(), s.end()), *std::max_element(s.begin(), s.end()), med(ax), med(bx), med(nb));
    }
    return 0;
}
