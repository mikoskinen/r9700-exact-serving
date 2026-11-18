/* dbench (DRAFTER track, tp8 loop 18.11.2026) -- one card, the DSpark drafter's routed MXFP4 experts as one drafter pass
 * runs them: 3 stages x (gate_up 4608 x 5120 by token rows, down 5120 x 2304 by sorted rows), 32 experts a rank, top 3
 * of 128, `tokens` = 5 rows a sequence (engine_draft.cpp block_shape: DSpark proposes its whole block of 5). The calls
 * go through libr4d's own entries (moe_gemm_ggml type=mxfp4: r4d_rad_moe_gemm_mxfp4_gemv / _tiled, compiled from the
 * branch's r4d_ds41_mxfp4.hip / r4d_ds41_mxfp4_mr.hip with the plugin's flags), the stock path (drafter_mx off) vs the
 * drafter_mx rule vs pinned forms, every output byte-compared with the stock path's.
 *
 * Timing as the engine streams them: each stage's experts are their own weights (3 x 601 MB a rank), the 6 calls of a pass
 * issued back to back on one stream with an event between calls, `iters` passes; a call's time is its event interval
 * (median over the passes), the pass the sum. 1.8 GB a pass >> the 64 MB MALL, so every call reads its experts from DRAM.
 *
 * Routing: --route synth (default; r4d_selftest's mx_route, the 10.11. drkbench form: a sequence's rows draw 7 in 10 of
 * their experts from a pool of 6 of its own) or --route FILE, an R4D_MXDUMP capture (40 ints a record: n_expert, tokens,
 * T, off[0 .. n_expert]); a record's counts are kept and its slots assigned round robin over the tokens.
 *
 * usage: dbench [--tokens 40,80] [--iters 30] [--forms stock,rule,s1m8,...] [--route synth|FILE] [--rec-stride N]
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

/* r4d_emit.h's dispatch log (r4d_rows.cpp in the plugin). */
const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;

extern "C" int r4d_rad_moe_gemm_mxfp4_gemv(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_moe_gemm_mxfp4_tiled(const RadArgs* a, RadStream s);
extern "C" void r4d_mx_mr_pin(int on, int h, int r, int mt, int wn, int grid);
extern "C" const char* r4d_mx_mr_last();
extern "C" void r4d_mx_mr_cap(int tokens);
extern "C" void r4d_mx_gemv_order(int on);
void r4d_mx_mr_set(int on);

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

/* ---- routing */
struct Route { std::vector<int32_t> sorted, off; int64_t live = 0, experts = 0, rmax = 0; };
static constexpr int64_t kEG = 128, kE = 32, kTopK = 3;

static int g_pool = 6, g_pin10 = 7;   /* --pool / --pin10: a sequence's expert pool and the tenths of picks drawn from it */
static Route route_synth(int64_t tokens, int rank, uint64_t seed) {
    uint64_t rs = seed;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    std::vector<std::vector<int32_t>> by((size_t)kEG);
    std::vector<int32_t> pool((size_t)g_pool);
    for (int64_t t = 0; t < tokens; ++t) {
        if (t % 5 == 0)
            for (auto& p : pool) p = (int32_t)(rnd() % kEG);
        int32_t pick[3] = { -1, -1, -1 };
        for (int j = 0; j < kTopK; ++j) {
            int32_t e; int guard = 0;
            do { e = ((int)(rnd() % 10) < g_pin10) ? pool[(size_t)(rnd() % pool.size())] : (int32_t)(rnd() % kEG); }
            while ((e == pick[0] || e == pick[1] || e == pick[2]) && ++guard < 64);
            pick[j] = e;
            by[(size_t)e].push_back((int32_t)(t * kTopK + j));
        }
    }
    Route r; r.off.assign((size_t)(kE + 1), 0);
    for (int64_t e = 0; e < kE; ++e) {
        const auto& v = by[(size_t)(rank * kE + e)];
        r.off[(size_t)(e + 1)] = r.off[(size_t)e] + (int32_t)v.size();
        r.sorted.insert(r.sorted.end(), v.begin(), v.end());
        r.experts += !v.empty();
        r.rmax = std::max<int64_t>(r.rmax, (int64_t)v.size());
    }
    r.live = (int64_t)r.sorted.size();
    r.sorted.resize((size_t)(tokens * kTopK), -1);
    return r;
}

/* An R4D_MXDUMP record's counts: off[0..n] of this rank's experts; slots assigned so that expert e's k-th row is token
 * (c + k) mod tokens with distinct tokens an expert (round robin start c over the experts). */
static Route route_counts(const std::vector<int32_t>& off, int64_t tokens) {
    Route r; r.off = off;
    int64_t c = 0;
    for (int64_t e = 0; e + 1 < (int64_t)off.size(); ++e) {
        const int64_t n = off[(size_t)e + 1] - off[(size_t)e];
        for (int64_t k = 0; k < n; ++k) r.sorted.push_back((int32_t)(((c + k) % tokens) * kTopK + (e % kTopK)));
        c += n;
        r.experts += n > 0;
        r.rmax = std::max(r.rmax, n);
    }
    r.live = (int64_t)r.sorted.size();
    r.sorted.resize((size_t)(tokens * kTopK), -1);
    return r;
}

static int g_rec_start = 0;   /* --rec-start: the first matching record taken (then every `stride`-th) */
static std::vector<std::vector<int32_t>> read_records(const char* path, int64_t tokens, int want, int stride) {
    std::vector<std::vector<int32_t>> out;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    int32_t r[40];
    long long idx = 0;
    while (std::fread(r, 4, 40, f) == 40 && (int)out.size() < want) {
        if (r[0] == (int32_t)kE && r[1] == tokens) {
            if (idx >= g_rec_start && (idx - g_rec_start) % stride == 0) out.emplace_back(r + 3, r + 3 + kE + 1);
            ++idx;
        }
    }
    std::fclose(f);
    return out;
}

/* ---- weights: random MXFP4 blocks, exponents near 127 */
static void mx_blocks(std::vector<uint8_t>& w, int64_t rows, int64_t K, uint64_t seed) {
    uint64_t rs = seed;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    const int64_t nb = K / 32;
    w.resize((size_t)(rows * nb * 17));
    for (int64_t b = 0; b < rows * nb; ++b) {
        uint8_t* p = w.data() + b * 17;
        p[0] = (uint8_t)(118 + rnd() % 14);
        for (int i = 1; i < 17; ++i) p[i] = (uint8_t)rnd();
    }
}

struct Proj { const char* name; int64_t N, K; bool sorted_rows; };
static const Proj kProj[2] = { { "gate_up", 4608, 5120, false }, { "down", 5120, 2304, true } };

struct Call {          /* one (stage, projection) launch */
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    RadArgs a{};
    bool gemv = false;
    size_t ybytes = 0;
    void* y = nullptr;
    Route rt;
    int p = 0, stage = 0;
    int launch(hipStream_t s) {
        return gemv ? r4d_rad_moe_gemm_mxfp4_gemv(&a, (RadStream)s) : r4d_rad_moe_gemm_mxfp4_tiled(&a, (RadStream)s);
    }
};

struct Form { std::string name; int on, h, r, mt, wn, grid; };

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) v.push_back(s.substr(p, q - p)); p = q + 1; }
    return v;
}
static bool parse_form(const std::string& n0, Form* f) {
    f->name = n0; f->h = 4; f->r = 1; f->mt = 4; f->wn = 0; f->grid = 0;
    std::string n = n0;
    /* a trailing _gGRID: the grid cap */
    if (size_t q = n.find("_g"); q != std::string::npos) { f->grid = std::atoi(n.c_str() + q + 2); n = n.substr(0, q); }
    if (n == "stock") { f->on = 0; return true; }
    if (n == "rule") { f->on = -1; return true; }
    {   /* xRmMT[wWN]: the tiled order's forms with EX (the segment's rows only); dMT[wWN]: the 8-byte-row form */
        int a = 0, b = 0, c = 0;
        if (std::sscanf(n.c_str(), "d%dw%d", &b, &c) == 2) { f->on = 1; f->h = 2; f->r = 1; f->mt = b; f->wn = c; return true; }
        if (std::sscanf(n.c_str(), "d%d", &b) == 1) { f->on = 1; f->h = 2; f->r = 1; f->mt = b; return true; }
        int ab = 0;
        if (std::sscanf(n.c_str(), "a%dx%dm%d", &ab, &a, &b) == 3) { f->on = 1; f->h = 1 | (ab << 1); f->r = a; f->mt = b; return true; }
        if (std::sscanf(n.c_str(), "x%dm%dw%d", &a, &b, &c) == 3) { f->on = 1; f->h = 1; f->r = a; f->mt = b; f->wn = c; return true; }
        if (std::sscanf(n.c_str(), "x%dm%d", &a, &b) == 2) { f->on = 1; f->h = 1; f->r = a; f->mt = b; return true; }
    }
    /* sRmMT[wWN] (the tiled order's forms) or gHrRmMT[wWN] (the gemv band's) */
    int a = 0, b = 0, c = 0;
    if (std::sscanf(n.c_str(), "s%dm%dw%d", &a, &b, &c) == 3) { f->on = 1; f->h = 0; f->r = a; f->mt = b; f->wn = c; return true; }
    if (std::sscanf(n.c_str(), "s%dm%d", &a, &b) == 2) { f->on = 1; f->h = 0; f->r = a; f->mt = b; return true; }
    int d = 0;
    if (std::sscanf(n.c_str(), "g%dr%dm%dw%d", &a, &b, &c, &d) == 4) { f->on = 1; f->h = a; f->r = b; f->mt = c; f->wn = d; return true; }
    if (std::sscanf(n.c_str(), "g%dr%dm%d", &a, &b, &c) == 3) { f->on = 1; f->h = a; f->r = b; f->mt = c; return true; }
    return false;
}
static void apply_form(const Form& f) {
    if (f.on == 0) { r4d_mx_mr_pin(0, 0, 0, 0, 0, 0); r4d_mx_mr_set(0); }
    else if (f.on < 0) { r4d_mx_mr_set(1); r4d_mx_mr_pin(-1, 0, 0, 0, 0, 0); }
    else { r4d_mx_mr_set(1); r4d_mx_mr_pin(1, f.h, f.r, f.mt, f.wn, f.grid); }
}

int main(int argc, char** argv) {
    std::string toks = "40,80", forms = "stock,rule", route = "synth";
    int iters = 30, rec_stride = 2, stages = 3, cap = 0, gemv_order = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--tokens") toks = nx();
        else if (a == "--iters") iters = std::atoi(nx().c_str());
        else if (a == "--forms") forms = nx();
        else if (a == "--route") route = nx();
        else if (a == "--rec-stride") rec_stride = std::atoi(nx().c_str());
        else if (a == "--stages") stages = std::atoi(nx().c_str());
        else if (a == "--cap") cap = std::atoi(nx().c_str());
        else if (a == "--gemv-order") gemv_order = 1;
        else if (a == "--pool") g_pool = std::atoi(nx().c_str());
        else if (a == "--rec-start") g_rec_start = std::atoi(nx().c_str());
        else if (a == "--pin10") g_pin10 = std::atoi(nx().c_str());
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    std::vector<Form> fl;
    for (auto& n : split(forms)) { Form f; if (!parse_form(n, &f)) { std::fprintf(stderr, "bad form %s\n", n.c_str()); return 2; } fl.push_back(f); }
    if (fl.empty() || fl[0].on != 0) fl.insert(fl.begin(), Form{ "stock", 0, 0, 0, 0, 0, 0 });
    r4d_mx_mr_cap(cap);   /* 0: the shipped bound (48 tokens) */
    r4d_mx_gemv_order(gemv_order);   /* 1: every call in the gemv band's order (stock = mx_gemv_kernel at any tokens) */
    if (gemv_order) std::printf("gemv order at every token count: the reference is the stock gemv kernel\n");
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
    std::printf("synth routing: pool %d, %d/10 of picks from it\n", g_pool, g_pin10);
    std::printf("dbench: %s, %d WGPs, %d MHz, mem %d MHz x %d bit; route %s; iters %d; stages %d; forms' token cap %d\n", pr.gcnArchName,
                pr.multiProcessorCount, pr.clockRate / 1000, pr.memoryClockRate / 1000, pr.memoryBusWidth, route.c_str(), iters, stages, cap > 0 ? cap : 48);
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    /* weights: stage x projection, 32 experts each, a pointer table */
    void* wtab[3][2]; size_t ebytes[2];
    for (int st = 0; st < stages; ++st)
        for (int p = 0; p < 2; ++p) {
            const Proj& P = kProj[p];
            const int64_t rb = P.K / 32 * 17; ebytes[p] = (size_t)(P.N * rb);
            std::vector<uint8_t> hw;
            mx_blocks(hw, kE * P.N, P.K, 0xFEEDull ^ (uint64_t)P.N ^ ((uint64_t)st << 40));
            uint8_t* d = nullptr; CK(hipMalloc((void**)&d, hw.size()));
            CK(hipMemcpy(d, hw.data(), hw.size(), hipMemcpyHostToDevice));
            std::vector<void*> ptrs((size_t)kE);
            for (int64_t e = 0; e < kE; ++e) ptrs[(size_t)e] = d + e * (int64_t)ebytes[p];
            CK(hipMalloc(&wtab[st][p], kE * sizeof(void*)));
            CK(hipMemcpy(wtab[st][p], ptrs.data(), kE * sizeof(void*), hipMemcpyHostToDevice));
        }

    int fails = 0;
    for (auto& ts_ : split(toks)) {
        const int64_t tk = std::atoll(ts_.c_str());
        /* routing per stage */
        std::vector<Route> rts;
        if (route == "synth") {
            for (int st = 0; st < stages; ++st) rts.push_back(route_synth(tk, 0, 0xD5A2Cull + (uint64_t)tk + 1000003ull * (uint64_t)st));
        } else {
            auto recs = read_records(route.c_str(), tk, stages, rec_stride);
            if ((int)recs.size() < stages) { std::printf("tokens %lld: %zu records in %s (need %d)\n", (long long)tk, recs.size(), route.c_str(), stages); continue; }
            for (int st = 0; st < stages; ++st) rts.push_back(route_counts(recs[(size_t)st], tk));
        }
        std::vector<Call> calls;
        for (int st = 0; st < stages; ++st)
            for (int p = 0; p < 2; ++p) {
                const Proj& P = kProj[p];
                Call c; c.rt = rts[(size_t)st]; c.p = p; c.stage = st;
                const int64_t TR = (int64_t)c.rt.sorted.size(), AR = P.sorted_rows ? TR : tk, rb = P.K / 32 * 17;
                std::vector<uint16_t> hx((size_t)(AR * P.K));
                uint64_t rs = 0x777ull + (uint64_t)tk + 7919ull * (uint64_t)(2 * st + p);
                for (auto& v : hx) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; v = h_to_bf16((float)((int)(rs % 2001) - 1000) / 250.0f); }
                void *dx, *dso, *dof, *dy;
                CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
                CK(hipMalloc(&dso, (size_t)TR * 4)); CK(hipMemcpy(dso, c.rt.sorted.data(), (size_t)TR * 4, hipMemcpyHostToDevice));
                CK(hipMalloc(&dof, (size_t)(kE + 1) * 4)); CK(hipMemcpy(dof, c.rt.off.data(), (size_t)(kE + 1) * 4, hipMemcpyHostToDevice));
                c.ybytes = (size_t)(TR * P.N) * 2;
                CK(hipMalloc(&dy, c.ybytes)); c.y = dy;
                c.ts = { T(dx, RAD_BF16, { AR, P.K }), T(wtab[st][p], RAD_U8, { kE, P.N, rb }), T(dso, RAD_I32, { TR }),
                         T(dof, RAD_I32, { kE + 1 }), T(dy, RAD_BF16, { TR, P.N }) };
                c.ts[1].stride[0] = 0;   /* a table of expert pointers, as the engine hands them */
                c.ps = { PI("M", tk), PI("N", P.N), PI("K", P.K), PI("n_expert", kE), PI("top_k", kTopK),
                         PS("a_order", P.sorted_rows ? "sorted" : "token"), PS("type", "mxfp4"), PS("dtype", "bf16") };
                c.a.t = c.ts.data(); c.a.n_t = (int)c.ts.size(); c.a.p = c.ps.data(); c.a.n_p = (int)c.ps.size(); c.a.world_size = 1;
                c.gemv = tk <= 16 || gemv_order;
                calls.push_back(std::move(c));
            }
        for (auto& c : calls) { c.a.t = c.ts.data(); c.a.p = c.ps.data(); }
        std::printf("== tokens %lld (%lld sequences x 5):", (long long)tk, (long long)(tk / 5));
        for (int st = 0; st < stages; ++st) std::printf(" stage %d live %lld experts, %lld slots, run max %lld;", st, (long long)rts[(size_t)st].experts, (long long)rts[(size_t)st].live, (long long)rts[(size_t)st].rmax);
        std::printf("\n");
        std::vector<std::vector<uint8_t>> ref(calls.size());
        double stock_pass = 0;
        for (const Form& f : fl) {
            apply_form(f);
            /* correctness: each call once, bytes vs the stock path's */
            bool same = true; std::string formdesc;
            for (size_t i = 0; i < calls.size(); ++i) {
                CK(hipMemsetAsync(calls[i].y, 0xff, calls[i].ybytes, s));
                const int rc = calls[i].launch(s);
                CK(hipStreamSynchronize(s));
                if (rc != 0) { std::printf("  %s: call %zu rc %d\n", f.name.c_str(), i, rc); same = false; continue; }
                if (i == 0 && f.on != 0) formdesc = r4d_mx_mr_last();
                if (f.on != 0 && i > 0 && formdesc != r4d_mx_mr_last()) formdesc += std::string(" / ") + r4d_mx_mr_last();
                std::vector<uint8_t> got(calls[i].ybytes);
                CK(hipMemcpy(got.data(), calls[i].y, calls[i].ybytes, hipMemcpyDeviceToHost));
                if (f.on == 0) ref[i] = got;
                else if (got != ref[i]) same = false;
            }
            /* timing: passes back to back, an event before each call */
            const size_t nc = calls.size();
            std::vector<hipEvent_t> ev((nc + 1) * (size_t)iters);
            for (auto& e : ev) CK(hipEventCreate(&e));
            for (int w = 0; w < 5; ++w) for (auto& c : calls) c.launch(s);
            for (int it = 0; it < iters; ++it) {
                for (size_t i = 0; i < nc; ++i) { CK(hipEventRecord(ev[(size_t)it * (nc + 1) + i], s)); calls[i].launch(s); }
                CK(hipEventRecord(ev[(size_t)it * (nc + 1) + nc], s));
            }
            CK(hipStreamSynchronize(s));
            std::vector<std::vector<float>> t(nc);
            std::vector<float> pass;
            for (int it = 0; it < iters; ++it) {
                float tot = 0;
                for (size_t i = 0; i < nc; ++i) {
                    float ms = 0; CK(hipEventElapsedTime(&ms, ev[(size_t)it * (nc + 1) + i], ev[(size_t)it * (nc + 1) + i + 1]));
                    t[i].push_back(ms * 1000.f); tot += ms * 1000.f;
                }
                pass.push_back(tot);
            }
            for (auto& e : ev) CK(hipEventDestroy(e));
            auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return (double)v[v.size() / 2]; };
            const double pm = med(pass);
            if (f.on == 0) stock_pass = pm;
            double bytes = 0;
            for (auto& c : calls) bytes += (double)c.rt.experts * (double)ebytes[c.p];
            std::printf("  %-10s pass %8.1f us (x%.2f vs stock) %6.0f GB/s of live experts | bytes %s | %s\n", f.name.c_str(), pm,
                        stock_pass > 0 ? stock_pass / pm : 0.0, bytes / (pm * 1e3), f.on == 0 ? "reference" : same ? "IDENTICAL" : "DIFFER",
                        formdesc.c_str());
            std::printf("             calls us:");
            for (size_t i = 0; i < nc; ++i) {
                const double m = med(t[i]);
                std::printf(" s%d.%s %.1f (%.0f GB/s)", calls[i].stage, kProj[calls[i].p].name, m,
                            (double)calls[i].rt.experts * (double)ebytes[calls[i].p] / (m * 1e3));
            }
            std::printf("\n");
            if (f.on != 0 && !same) ++fails;
            std::fflush(stdout);
        }
        apply_form(fl[0]);
        for (auto& c : calls) { hipFree(c.ts[0].data); hipFree(c.ts[2].data); hipFree(c.ts[3].data); hipFree(c.y); }
    }
    std::printf("DBVERDICT %s\n", fails ? "SOME FORM DIFFERS" : "all forms byte-identical to the stock path");
    return fails ? 1 : 0;
}
