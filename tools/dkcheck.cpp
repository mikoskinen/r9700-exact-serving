// dkcheck -- lever P (ds41-p1pass, 13.11.2026): the routed EXL3 experts of a 17..48-token decode pass as ONE launch of the
// DECODE-BAND kernels (exl3r4d_launch at M > 16; the `_dk` rows) against today's issue, the same tokens in pieces of at
// most 16 through the `_m16` rows -- byte for byte per (token, slot): gate_up's h (f16) and down's y (bf16).
//
//   dkcheck --exl3 <exl3r4d.so> [--dtype exl3k2|exl3k2h] [--x3d V] [--full-gu ROW --full-dn ROW] [--mlo 17 --mhi 48]
//           [--draws D] [--rank r] [--seed S] [--perf ITERS [--perf-m 24,32,48] [--perf-layers L]]
//
// Geometry: DeepSeek V4.1 Flash at four ranks, expert parallel -- 384 experts, top 6, a rank owns 96 whole experts in two
// parity tables, gate_up [4608, 5120] (h_rot 1, a_rot 0, limit 10, bf16 token rows) and down [5120, 2304] (h_rot 1), as
// rad_block_moe_fp8.h declare_exl3 / issue_exl3 pass them. Weights: random codes records, random +-1 suh / suh.1, random
// svh; token rows random bf16. The routing is drawn model-wide (top 6 of 384 a token) and the rank's slots kept, sorted
// by expert then by flat index (moe_scatter's contract), so y has M * 6 rows and only off[96] of them are written.
//
// CASES a (dtype, x3d, M): --draws independent routings, one grouped routing (runs of 4 tokens sharing experts: a verify
// pass), and one adversarial routing (expert 0 takes every token: a run of M > 16 rows, several chunks; expert 1 takes
// tokens 0..16: a run of 17). Every local (token, slot) of the ONE launch is compared with the same (token, slot) of its
// piece's launch. Prints DKCHECK lines and DKVERDICT; exit 0 only when nothing differs and every launch returned RAD_OK.
//
// --perf ITERS: one card's time of the chain (gate_up -> down) a layer at M, ONE launch (full rows) against the pieces
// (m16 rows), order F P P F, over --perf-layers routings on one weight pool. A bench reading, not a server number.
#include "rad_abi.h"
#include "rad_plugin.h"
#include "exl3_abi.h"

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kDtTrellis = RAD_DT_PLUGIN_BASE + 0;   // kernels/exl3r4d/exl3r4d.h EXL3R4D_DT_TRELLIS
constexpr int64_t kEall = 384, kE = 96, kTop = 6, kD = 5120, kF = 2304, kPiece = 16;

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "dkcheck: %s\n", m.c_str());
    std::exit(2);
}
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) die(std::string(what) + ": " + hipGetErrorString(e));
}
void* dalloc(size_t n) {
    void* p = nullptr;
    ck(hipMalloc(&p, n < 256 ? 256 : n), "hipMalloc");
    return p;
}
void put(void* d, const void* h, size_t n) { ck(hipMemcpy(d, h, n, hipMemcpyHostToDevice), "copy to device"); }
void get(void* h, const void* d, size_t n) { ck(hipMemcpy(h, d, n, hipMemcpyDeviceToHost), "copy to host"); }

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
double unif() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

RadParam PI(const char* k, long long v) { RadParam p = RAD_INT(k, v); return p; }
RadParam PS(const char* k, const char* v) { RadParam p = RAD_STR(k, v); return p; }
RadParam PF(const char* k, double v) { RadParam p = RAD_F64(k, v); return p; }
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

void* g_lib = nullptr;
const RadKernelInfo* row_of(const char* name) {
    auto count = (int (*)())dlsym(g_lib, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(g_lib, "rad_kernel_at");
    if (!count || !at) die("not a radiance kernel library");
    for (int i = 0; i < count(); ++i)
        if (at(i) && at(i)->name && std::strcmp(at(i)->name, name) == 0) return at(i);
    die(std::string("no row ") + name);
}

void* random_dev(size_t n) {
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i + 8 <= n; i += 8) { const uint64_t v = rnd(); std::memcpy(&h[i], &v, 8); }
    for (size_t i = n / 8 * 8; i < n; ++i) h[i] = (uint8_t)rnd();
    void* d = dalloc(n);
    put(d, h.data(), n);
    return d;
}
// kind 1: +-1 (the EXL3 sign vectors); 2: uniform in (0.5, 1.5)
void* f16_dev(size_t n, int kind) {
    std::vector<uint16_t> h(n);
    for (auto& e : h) {
        const double u = unif();
        e = rad_f32_to_f16(kind == 1 ? (u < 0.5 ? -1.0f : 1.0f) : (float)(0.5 + u));
    }
    void* d = dalloc(n * 2);
    put(d, h.data(), n * 2);
    return d;
}
// n entries of `stride` bytes (a multiple of 256) in one buffer and a device pointer table onto them
template <class Gen> void* table_of(int64_t n, size_t stride, Gen gen) {
    void* base = gen((size_t)n * stride);
    std::vector<void*> p((size_t)n);
    for (int64_t i = 0; i < n; ++i) p[(size_t)i] = (char*)base + (size_t)i * stride;
    void* d = dalloc(p.size() * sizeof(void*));
    put(d, p.data(), p.size() * sizeof(void*));
    return d;
}
size_t up256(size_t b) { return (b + 255) / 256 * 256; }

// A model-wide routing of M tokens: ids[t][j], the token's top-6 experts (distinct), in rank order.
typedef std::vector<int32_t> Ids;
Ids draw_ids(int64_t M, int kind) {
    Ids id((size_t)(M * kTop));
    std::vector<double> logp((size_t)kEall), shared((size_t)kEall, 0.0);
    for (auto& v : logp) v = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::pair<double, int>> sc((size_t)kEall);
    for (int64_t t = 0; t < M; ++t) {
        if (kind == 1 && t % 4 == 0)
            for (auto& v : shared) v = -std::log(-std::log(unif()));
        const double mix = kind == 1 ? 0.53 : 0.0;
        for (int64_t e = 0; e < kEall; ++e)
            sc[(size_t)e] = { logp[(size_t)e] + mix * shared[(size_t)e] - (1 - mix) * std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + kTop, sc.end(),
                          [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
        for (int64_t j = 0; j < kTop; ++j) id[(size_t)(t * kTop + j)] = sc[(size_t)j].second;
    }
    return id;
}
// The adversarial routing on rank `base`: expert base + 0 takes every token (a run of M rows), base + 1 tokens 0..16
// (a run of 17), base + 2 token 16 alone and the last token (runs across the piece boundary); the rest as drawn.
Ids adversarial(int64_t M, int64_t base) {
    Ids id = draw_ids(M, 0);
    auto setj = [&](int64_t t, int64_t j, int32_t e) {
        for (int64_t q = 0; q < kTop; ++q)   // keep distinct: a slot already holding e takes slot j's expert
            if (q != j && id[(size_t)(t * kTop + q)] == e) id[(size_t)(t * kTop + q)] = id[(size_t)(t * kTop + j)];
        id[(size_t)(t * kTop + j)] = e;
    };
    for (int64_t t = 0; t < M; ++t) {
        setj(t, 0, (int32_t)base);
        if (t <= 16) setj(t, 1, (int32_t)(base + 1));
        if (t == 16 || t == M - 1) setj(t, 2, (int32_t)(base + 2));
    }
    for (int64_t t = 0; t < M; ++t)   // distinct experts a token (top-k picks each once)
        for (int64_t a = 0; a < kTop; ++a)
            for (int64_t b = a + 1; b < kTop; ++b)
                if (id[(size_t)(t * kTop + a)] == id[(size_t)(t * kTop + b)]) die("adversarial routing: a token's experts repeat");
    return id;
}
// The rank's sort of tokens [a, a + n): sorted flat indices (t - a) * top_k + j by local expert, offsets; pos[(t - a) * 6 + j]
// is the slot's sorted row, or -1 for a slot another rank owns.
struct Route {
    std::vector<int32_t> sorted, off, pos;
    int64_t live = 0, longest = 0;
};
Route route_of(const Ids& id, int64_t a, int64_t n, int64_t base) {
    Route r;
    r.sorted.assign((size_t)(n * kTop), 0);
    r.off.assign((size_t)kE + 1, 0);
    r.pos.assign((size_t)(n * kTop), -1);
    std::vector<std::vector<int32_t>> by((size_t)kE);
    for (int64_t t = 0; t < n; ++t)
        for (int64_t j = 0; j < kTop; ++j) {
            const int64_t e = id[(size_t)((a + t) * kTop + j)] - base;
            if (e >= 0 && e < kE) by[(size_t)e].push_back((int32_t)(t * kTop + j));
        }
    int32_t at = 0;
    for (int64_t e = 0; e < kE; ++e) {
        r.off[(size_t)e] = at;
        for (int32_t f : by[(size_t)e]) { r.sorted[(size_t)at] = f; r.pos[(size_t)f] = at; ++at; }
        r.live += !by[(size_t)e].empty();
        r.longest = std::max<int64_t>(r.longest, (int64_t)by[(size_t)e].size());
    }
    r.off[(size_t)kE] = at;
    return r;
}

struct Op {
    const RadKernelInfo* row = nullptr;
    void* instance = nullptr;
    void* scratch = nullptr;
    int64_t scratch_bytes = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string exl3, dtype = "exl3k2", full_gu = "exl3r4d_moe_gate_up_m16", full_dn = "exl3r4d_moe_down_m16", perf_m = "24,32,48";
    int x3d = 2, draws = 3, rank = 0, perf = 0, perf_layers = 40;
    int64_t mlo = 17, mhi = 48;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
        if (a == "--exl3") exl3 = nx();
        else if (a == "--dtype") dtype = nx();
        else if (a == "--x3d") x3d = std::atoi(nx());
        else if (a == "--full-gu") full_gu = nx();
        else if (a == "--full-dn") full_dn = nx();
        else if (a == "--mlo") mlo = std::atoll(nx());
        else if (a == "--mhi") mhi = std::atoll(nx());
        else if (a == "--draws") draws = std::atoi(nx());
        else if (a == "--rank") rank = std::atoi(nx());
        else if (a == "--seed") g_rs ^= (uint64_t)std::atoll(nx()) * 0x2545F4914F6CDD1Dull;
        else if (a == "--perf") perf = std::atoi(nx());
        else if (a == "--perf-m") perf_m = nx();
        else if (a == "--perf-layers") perf_layers = std::atoi(nx());
        else die("unknown argument " + a);
    }
    if (exl3.empty()) die("--exl3 <exl3r4d.so> is required");
    if (mlo < 1 || mhi > 96 || mlo > mhi) die("--mlo / --mhi: 1 <= mlo <= mhi <= 96");
    if (rank < 0 || rank > 3) die("--rank 0..3");
    const int bits = exl3_gemm_bits(dtype.c_str());
    if (bits < 0) die("--dtype " + dtype + " is not an EXL3 GEMM dtype");
    const int64_t k2 = exl3_k2(bits), base = rank * kE, half = kE / 2, Mmax = mhi;
    g_lib = dlopen(exl3.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) die(std::string("dlopen ") + exl3 + ": " + dlerror());
    {
        auto sw = (int (*)(const char*, int))dlsym(g_lib, "rad_fn4_switch");
        auto val = (int (*)(const char*, int*))dlsym(g_lib, "rad_fn4_value");
        if (!sw || !val) die("the library exports no fn4 hooks");
        int v = -1;
        if (sw("exl3_x3d", x3d) != 1 || val("exl3_x3d", &v) != 1 || v != x3d) die("fn4 exl3_x3d was not taken");
    }
    int dev = 0;
    hipDeviceProp_t prop;
    ck(hipGetDevice(&dev), "hipGetDevice");
    ck(hipGetDeviceProperties(&prop, dev), "props");
    char bus[32] = { 0 };
    ck(hipDeviceGetPCIBusId(bus, sizeof bus, dev), "bus id");
    std::printf("dkcheck: device pci %s %s; %s (K2 %lld), fn4 exl3_x3d %d, rank %d of 4 (experts %lld..%lld of %lld, top %lld), "
                "gate_up [%lld, %lld] down [%lld, %lld]; ONE launch rows %s / %s vs pieces of <= %lld through the _m16 rows\n",
                bus, prop.gcnArchName, dtype.c_str(), (long long)k2, x3d, rank, (long long)base, (long long)(base + kE - 1),
                (long long)kEall, (long long)kTop, (long long)(2 * kF), (long long)kD, (long long)kD, (long long)kF,
                full_gu.c_str(), full_dn.c_str(), (long long)kPiece);

    // ---- the weights: two parity classes of 48 experts
    const int64_t Rg = 2 * kF, Cg = kD, Rd = kD, Cd = kF;
    const size_t cbg = up256((size_t)(Rg * Cg * k2 / 16)), cbd = up256((size_t)(Rd * Cd * k2 / 16));
    RadTensor g_t[2][5], d_t[2][3];
    for (int c = 0; c < 2; ++c) {
        g_t[c][0] = TAB(table_of(half, cbg, [](size_t n) { return random_dev(n); }), kDtTrellis, { half, Rg, Cg });
        g_t[c][1] = TAB(table_of(half, up256((size_t)Cg * 2), [](size_t n) { return f16_dev(n / 2, 1); }), RAD_F16, { half, 1, Cg });
        g_t[c][2] = TAB(table_of(half, up256((size_t)Cg * 2), [](size_t n) { return f16_dev(n / 2, 1); }), RAD_F16, { half, 1, Cg });
        g_t[c][3] = TAB(table_of(half, up256((size_t)Rg * 2), [](size_t n) { return f16_dev(n / 2, 2); }), RAD_F16, { half, Rg, 1 });
        d_t[c][0] = TAB(table_of(half, cbd, [](size_t n) { return random_dev(n); }), kDtTrellis, { half, Rd, Cd });
        d_t[c][1] = TAB(table_of(half, up256((size_t)Cd * 2), [](size_t n) { return f16_dev(n / 2, 1); }), RAD_F16, { half, 1, Cd });
        d_t[c][2] = TAB(table_of(half, up256((size_t)Rd * 2), [](size_t n) { return f16_dev(n / 2, 2); }), RAD_F16, { half, Rd, 1 });
        g_t[c][4] = d_t[c][1];   // gate_up's dn_suh is down's suh run
    }
    // ---- the token rows (bf16, as the block's normed input), the outputs, the routing operands
    std::vector<uint16_t> xh((size_t)(Mmax * kD));
    for (auto& e : xh) e = rad_f32_to_bf16((float)(2 * unif() - 1));
    void* x = dalloc(xh.size() * 2);
    put(x, xh.data(), xh.size() * 2);
    const size_t hb = (size_t)(Mmax * kTop * kF * 2), yb = (size_t)(Mmax * kTop * kD * 2);
    void* h = dalloc(hb);
    void* y = dalloc(yb);
    void* dsorted = dalloc((size_t)(Mmax * kTop) * 4);
    void* doff = dalloc((size_t)(kE + 1) * 4);

    hipStream_t stream;
    ck(hipStreamCreate(&stream), "stream");
    auto params = [&](bool gu, int64_t M) {
        std::vector<RadParam> p = { PI("M", M), PI("N", gu ? Rg : Rd), PI("K", gu ? Cg : Cd), PI("n_expert", kE), PI("top_k", kTop),
                                    PS("dtype", dtype.c_str()), PS("codebook", "mul1"), PI("N_odd", gu ? Rg : Rd),
                                    PI("K_odd", gu ? Cg : Cd), PI("h_rot", 1) };
        if (gu) { p.push_back(PI("a_rot", 0)); p.push_back(PF("limit", 10.0)); }
        return p;
    };
    const Exl3Positions* gpos = exl3_positions(EXL3_GATE_UP);
    const Exl3Positions* dpos = exl3_positions(EXL3_DOWN);
    auto operands = [&](bool gu, int64_t a, int64_t n) {
        const Exl3Positions* pos = gu ? gpos : dpos;
        std::vector<RadTensor> t((size_t)pos->n_opd);
        t[(size_t)pos->x] = gu ? T((char*)x + (size_t)(a * kD * 2), RAD_BF16, { n, kD }) : T(h, RAD_F16, { n * kTop, kF });
        t[(size_t)pos->sorted] = T(dsorted, RAD_I32, { n * kTop });
        t[(size_t)pos->offset] = T(doff, RAD_I32, { kE + 1 });
        t[(size_t)pos->y] = gu ? T(h, RAD_F16, { n * kTop, kF }) : T(y, RAD_BF16, { n * kTop, kD });
        for (int c = 0; c < 2; ++c) {
            if (gu) for (int k = 0; k < 5; ++k) t[(size_t)pos->plane[c][k]] = g_t[c][k];
            else {
                const int dk[3] = { EXL3_P_CODES, EXL3_P_SUH, EXL3_P_SVH };
                for (int k = 0; k < 3; ++k) t[(size_t)pos->plane[c][dk[k]]] = d_t[c][k];
            }
        }
        return t;
    };
    auto setup = [&](const char* name, bool gu, int64_t Mtop) {
        Op o;
        o.row = row_of(name);
        const bool is_gu = !std::strcmp(o.row->op, EXL3_OP_GATE_UP) || !std::strcmp(o.row->op, EXL3_OP_GATE_UP_DK);
        const bool is_dn = !std::strcmp(o.row->op, EXL3_OP_DOWN) || !std::strcmp(o.row->op, EXL3_OP_DOWN_DK);
        if (gu ? !is_gu : !is_dn) die(std::string(name) + " is not the op asked for");
        std::vector<RadParam> p = params(gu, Mtop);
        if (o.row->init && o.row->init(p.data(), (int)p.size(), 0, 1, &o.instance) != RAD_OK) die(std::string(name) + ": init failed");
        std::vector<RadTensor> t = operands(gu, 0, Mtop);
        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; a.instance = o.instance;
        o.scratch_bytes = o.row->scratch ? o.row->scratch(&a) : 0;
        if (o.scratch_bytes > 0) o.scratch = dalloc((size_t)o.scratch_bytes);
        return o;
    };
    // the piece rows at their band's top (16), the ONE-launch rows at Mmax
    Op pg = setup("exl3r4d_moe_gate_up_m16", true, kPiece), pd = setup("exl3r4d_moe_down_m16", false, kPiece);
    Op fg = setup(full_gu.c_str(), true, Mmax), fd = setup(full_dn.c_str(), false, Mmax);
    std::printf("dkcheck: scratch bytes: pieces gate_up %lld down %lld; ONE launch (M %lld) gate_up %lld down %lld\n",
                (long long)pg.scratch_bytes, (long long)pd.scratch_bytes, (long long)Mmax, (long long)fg.scratch_bytes,
                (long long)fd.scratch_bytes);
    int launch_fail = 0;
    auto launch = [&](Op& o, bool gu, int64_t a, int64_t n) {
        std::vector<RadParam> p = params(gu, n);
        std::vector<RadTensor> t = operands(gu, a, n);
        RadArgs ar;
        std::memset(&ar, 0, sizeof ar);
        ar.t = t.data(); ar.n_t = (int)t.size(); ar.p = p.data(); ar.n_p = (int)p.size(); ar.world_size = 1;
        ar.instance = o.instance; ar.scratch = o.scratch; ar.scratch_bytes = o.scratch_bytes;
        const int rc = o.row->launch(&ar, (RadStream)stream);
        if (rc != RAD_OK) {
            if (launch_fail++ < 8) std::printf("DKFAIL %s launch at tokens [%lld, %lld) rc=%d\n", o.row->name, (long long)a, (long long)(a + n), rc);
        }
        return rc;
    };
    // one chain over tokens [a, a + n) with its sort; h and y of its n * 6 sorted rows copied back when asked
    auto chain = [&](Op& g, Op& d, const Route& r, int64_t a, int64_t n, std::vector<uint16_t>* hh, std::vector<uint16_t>* yh) {
        put(dsorted, r.sorted.data(), (size_t)(n * kTop) * 4);
        put(doff, r.off.data(), (size_t)(kE + 1) * 4);
        if (hh) { ck(hipMemset(h, 0x5a, (size_t)(n * kTop * kF * 2)), "memset h"); ck(hipMemset(y, 0x5a, (size_t)(n * kTop * kD * 2)), "memset y"); }
        int rc = launch(g, true, a, n);
        if (rc == RAD_OK) rc = launch(d, false, a, n);
        ck(hipStreamSynchronize(stream), "sync");
        if (hh) { hh->resize((size_t)(n * kTop * kF)); get(hh->data(), h, hh->size() * 2); }
        if (yh) { yh->resize((size_t)(n * kTop * kD)); get(yh->data(), y, yh->size() * 2); }
        return rc;
    };

    long long cases = 0, bad_cases = 0, slots = 0, h_diff = 0, y_diff = 0, long_runs = 0;
    if (perf == 0)
        for (int64_t M = mlo; M <= mhi; ++M)
            for (int dr = 0; dr < draws + 2; ++dr) {
                const int kind = dr < draws ? 0 : dr == draws ? 1 : 2;
                const Ids id = kind == 2 ? adversarial(M, base) : draw_ids(M, kind);
                const Route rf = route_of(id, 0, M, base);
                std::vector<uint16_t> hf, yf, hp, yp;
                long long cs = 0, ch = 0, cy = 0;
                int rc = chain(fg, fd, rf, 0, M, &hf, &yf);
                for (int64_t a = 0; a < M && rc == RAD_OK; a += kPiece) {
                    const int64_t n = std::min(kPiece, M - a);
                    const Route rp = route_of(id, a, n, base);
                    rc = chain(pg, pd, rp, a, n, &hp, &yp);
                    for (int64_t f = 0; f < n * kTop && rc == RAD_OK; ++f) {
                        const int32_t pp = rp.pos[(size_t)f], pf = rf.pos[(size_t)(a * kTop + f)];
                        if ((pp < 0) != (pf < 0)) die("the two sorts disagree on a slot's owner");
                        if (pp < 0) continue;
                        ++cs;
                        ch += std::memcmp(&hf[(size_t)pf * kF], &hp[(size_t)pp * kF], (size_t)kF * 2) != 0;
                        cy += std::memcmp(&yf[(size_t)pf * kD], &yp[(size_t)pp * kD], (size_t)kD * 2) != 0;
                    }
                }
                // a guard: the ONE launch's y is not all the fill pattern (it ran) on its first live row
                bool wrote = rf.off[(size_t)kE] == 0;
                for (int64_t q = 0; q < kD && !wrote; ++q) wrote = yf[(size_t)q] != 0x5a5a;
                ++cases;
                slots += cs; h_diff += ch; y_diff += cy;
                long_runs += rf.longest > kPiece;
                const bool ok = rc == RAD_OK && ch == 0 && cy == 0 && wrote && cs == rf.off[(size_t)kE];
                bad_cases += !ok;
                std::printf("DKCHECK %s x3d %d M %lld %s: live %lld longest run %lld, %lld slots, h differ %lld, y differ %lld%s -> %s\n",
                            dtype.c_str(), x3d, (long long)M, kind == 0 ? "draw" : kind == 1 ? "grouped" : "adversarial",
                            (long long)rf.live, (long long)rf.longest, cs, ch, cy, wrote ? "" : " (y NOT WRITTEN)",
                            ok ? "IDENTICAL" : "DIFFERS");
                std::fflush(stdout);
            }
    else {
        // ---- --perf: ms a "step" of perf_layers chains at M, ONE launch vs the pieces, F P P F
        std::vector<int64_t> Ms;
        for (const char* q = perf_m.c_str(); *q;) { char* e = nullptr; Ms.push_back(std::strtoll(q, &e, 10)); q = *e == ',' ? e + 1 : e; }
        for (int64_t M : Ms) {
            if (M < 1 || M > Mmax) die("--perf-m outside 1..--mhi");
            std::vector<Ids> ids;
            std::vector<Route> rfull;
            std::vector<std::vector<Route>> rpiece;
            double live_f = 0, live_p = 0;
            for (int l = 0; l < perf_layers; ++l) {
                ids.push_back(draw_ids(M, 0));
                rfull.push_back(route_of(ids.back(), 0, M, base));
                live_f += rfull.back().live;
                rpiece.emplace_back();
                for (int64_t a = 0; a < M; a += kPiece) {
                    rpiece.back().push_back(route_of(ids.back(), a, std::min(kPiece, M - a), base));
                    live_p += rpiece.back().back().live;
                }
            }
            // every routing's operands on the device once (the timed loop only launches)
            struct Dev { void* s; void* o; };
            auto dev_of = [&](const Route& r, int64_t n) {
                Dev d{ dalloc((size_t)(n * kTop) * 4), dalloc((size_t)(kE + 1) * 4) };
                put(d.s, r.sorted.data(), (size_t)(n * kTop) * 4);
                put(d.o, r.off.data(), (size_t)(kE + 1) * 4);
                return d;
            };
            std::vector<Dev> df;
            std::vector<std::vector<Dev>> dp;
            for (int l = 0; l < perf_layers; ++l) {
                df.push_back(dev_of(rfull[(size_t)l], M));
                dp.emplace_back();
                int i = 0;
                for (int64_t a = 0; a < M; a += kPiece, ++i) dp.back().push_back(dev_of(rpiece[(size_t)l][(size_t)i], std::min(kPiece, M - a)));
            }
            void* keep_s = dsorted; void* keep_o = doff;
            auto pass = [&](bool full) {
                for (int l = 0; l < perf_layers; ++l) {
                    if (full) {
                        dsorted = df[(size_t)l].s; doff = df[(size_t)l].o;
                        if (launch(fg, true, 0, M) == RAD_OK) launch(fd, false, 0, M);
                    } else {
                        int i = 0;
                        for (int64_t a = 0; a < M; a += kPiece, ++i) {
                            const int64_t n = std::min(kPiece, M - a);
                            dsorted = dp[(size_t)l][(size_t)i].s; doff = dp[(size_t)l][(size_t)i].o;
                            if (launch(pg, true, a, n) == RAD_OK) launch(pd, false, a, n);
                        }
                    }
                }
            };
            auto timed = [&](bool full) {
                ck(hipStreamSynchronize(stream), "sync");
                const auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < perf; ++it) pass(full);
                ck(hipStreamSynchronize(stream), "sync");
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / perf;
            };
            for (int w = 0; w < 3; ++w) { pass(true); pass(false); }
            ck(hipStreamSynchronize(stream), "sync");
            double f[4], p[4];
            for (int r = 0; r < 4; ++r) {   // F P P F, twice mirrored
                if (r % 2 == 0) { f[r] = timed(true); p[r] = timed(false); }
                else { p[r] = timed(false); f[r] = timed(true); }
            }
            std::sort(f, f + 4); std::sort(p, p + 4);
            const double fm = (f[1] + f[2]) / 2, pm = (p[1] + p[2]) / 2;
            std::printf("DKPERF %s x3d %d M %lld: %d chains a pass x %d iters; live experts a chain ONE launch %.1f, pieces %.1f | "
                        "ms a pass ONE launch %.3f [%.3f..%.3f], pieces %.3f [%.3f..%.3f] -> %+.3f ms (%+.1f %%)\n",
                        dtype.c_str(), x3d, (long long)M, perf_layers, perf, live_f / perf_layers, live_p / perf_layers,
                        fm, f[0], f[3], pm, p[0], p[3], fm - pm, 100.0 * (fm - pm) / pm);
            std::fflush(stdout);
            dsorted = keep_s; doff = keep_o;
        }
    }
    if (perf == 0) {
        const bool pass = bad_cases == 0 && launch_fail == 0 && cases > 0;
        std::printf("DKVERDICT %s x3d %d rank %d M %lld..%lld: %lld cases (%lld with a run above %lld rows), %lld (token, slot) rows "
                    "compared, h differ %lld, y differ %lld, launches failed %d -> %s\n",
                    dtype.c_str(), x3d, rank, (long long)mlo, (long long)mhi, cases, long_runs, (long long)kPiece, slots, h_diff,
                    y_diff, launch_fail, pass ? "PASS (byte-identical per row)" : "FAIL");
        return pass ? 0 : 1;
    }
    return launch_fail ? 1 : 0;
}
