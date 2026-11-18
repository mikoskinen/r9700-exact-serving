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
extern "C" void r4d_mx_wm_slab(int bytes);
void r4d_mx_mr_set(int on);
/* The a8 arms (COORDINATOR 19:52): libr4d's moe_gemm_mxfp4a8 (r4d_moe_mxfp4.hip body(): canonical planes, E2M1 codes as exact
 * E4M3 bytes, fp8 WMMA, the E8M0 and the activation scales folded in f32 a 32-group) on activations quantised by
 * r4d_quant_act_fp8 (E4M3, an f32 scale a (row, 128 of K)). Speed and value movement only: not the stock bytes. */
extern "C" int r4d_rad_moe_mxfp4(const RadArgs* a, RadStream stream);
extern "C" int r4d_quant_act_fp8(long a, long q, long s, int M, int K, int x_ld, int q_ld, int s_ld, long stream);
static int g_a8 = 0;   /* 0 off; 1 quant + a8 GEMM; 2 a8 GEMM only (activations quantised before); 3 quant only */

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
static constexpr int64_t kEG = 128;
/* experts a rank and top-k: the drafter's 32 of 128 / 3 (default) or, --model main, the main model's TP4 rank: all 384 / 6 */
static int64_t kE = 32, kTopK = 3;

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
static std::vector<std::vector<int32_t>> read_records(const char* path, int64_t tokens, int want, int stride, int64_t ne = -1) {
    if (ne < 0) ne = kE;
    std::vector<std::vector<int32_t>> out;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    int32_t r[40];
    long long idx = 0;
    while (std::fread(r, 4, 40, f) == 40 && (int)out.size() < want) {
        if (r[0] == (int32_t)ne && r[1] == tokens) {
            if (idx >= g_rec_start && (idx - g_rec_start) % stride == 0) out.emplace_back(r + 3, r + 3 + ne + 1);
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
static Proj kProj[2] = { { "gate_up", 4608, 5120, false }, { "down", 5120, 2304, true } };
/* --model main (COORDINATOR 19:47): the main model's routed experts at one TP4 rank, a uniform width W of 2304 (default 576:
 * the average of the served 640 / 512 split, the same bytes): gate_up 2W x 5120, down 5120 x W, 384 experts, top 6. */
static void set_main(int64_t W) { kE = 384; kTopK = 6; kProj[0].N = 2 * W; kProj[1].K = W; }
/* --model dtp: the drafter's experts tensor-parallel instead of expert-parallel -- every rank holds a W-wide slice (default 576 =
 * 2304 / 4) of all 128 experts: gate_up 2W x 5120, down 5120 x W, top 3. */
static void set_dtp(int64_t W) { kE = 128; kTopK = 3; kProj[0].N = 2 * W; kProj[1].K = W; }

/* X7's EXL3R4D_ROUTEDUMP records (520 int32: n_expert, rows (= tokens x top_k), off[0 .. n_expert]) of the main model's
 * gate_up launches: records with n_expert == kE and rows == tokens x kTopK, from --rec-start, every stride-th. */
static std::vector<std::vector<int32_t>> read_records_x7(const char* path, int64_t tokens, int want, int stride, int start) {
    std::vector<std::vector<int32_t>> out;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    std::vector<int32_t> r(520);
    long long idx = 0;
    while (std::fread(r.data(), 4, 520, f) == 520 && (int)out.size() < want) {
        if (r[0] == (int32_t)kE && r[1] == tokens * kTopK) {
            if (idx >= start && (idx - start) % stride == 0) out.emplace_back(r.begin() + 2, r.begin() + 2 + kE + 1);
            ++idx;
        }
    }
    std::fclose(f);
    return out;
}

struct Call {          /* one (stage, projection) launch */
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    RadArgs a{};
    bool gemv = false;
    size_t ybytes = 0;
    void* y = nullptr;
    Route rt;
    int p = 0, stage = 0;
    /* a8: quantised activations, the canonical planes' pointer tables (even / odd experts), its own args */
    void *aq = nullptr, *as = nullptr, *w8[2] = { nullptr, nullptr }, *s8[2] = { nullptr, nullptr };
    int64_t AR = 0, K = 0, N = 0;
    std::vector<RadTensor> ts8;
    std::vector<RadParam> ps8;
    RadArgs a8{};
    int launch(hipStream_t s) {
        if (g_a8) {
            if (g_a8 != 2) {
                const int rc = r4d_quant_act_fp8((long)(uintptr_t)ts[0].data, (long)(uintptr_t)aq, (long)(uintptr_t)as, (int)AR, (int)K,
                                                 0, 0, 0, (long)(uintptr_t)s);
                if (rc) return rc;
            }
            if (g_a8 == 3) return 0;
            const int64_t TR = (int64_t)rt.sorted.size();
            ts8 = { T(aq, RAD_F8E4M3, { AR, K }), T(as, RAD_F32, { AR, K / 128 }), T(w8[0], RAD_FP4E2M1, { kE / 2, N, K }),
                    T(s8[0], RAD_U8, { kE / 2, N, K / 32 }), T(ts[2].data, RAD_I32, { TR }), T(ts[3].data, RAD_I32, { kE + 1 }),
                    T(w8[1], RAD_FP4E2M1, { kE / 2, N, K }), T(s8[1], RAD_U8, { kE / 2, N, K / 32 }), T(ts[4].data, RAD_BF16, { TR, N }) };
            ts8[2].stride[0] = 0; ts8[3].stride[0] = 0; ts8[6].stride[0] = 0; ts8[7].stride[0] = 0;   /* pointer tables */
            a8.t = ts8.data(); a8.n_t = (int)ts8.size(); a8.p = ps8.data(); a8.n_p = (int)ps8.size(); a8.world_size = 1;
            return r4d_rad_moe_mxfp4(&a8, (RadStream)s);
        }
        return gemv ? r4d_rad_moe_gemm_mxfp4_gemv(&a, (RadStream)s) : r4d_rad_moe_gemm_mxfp4_tiled(&a, (RadStream)s);
    }
};

struct Form { std::string name; int on, h, r, mt, wn, grid; };

/* ---- value comparison (a rule-B form against the stock bytes) */
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
struct DiffStats { long long n = 0, ndiff = 0; int maxulp = 0; double maxabs = 0, maxref = 0, sdiff = 0, sref = 0, adiff = 0, aref = 0; };
static void diff_stats(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, DiffStats* d, size_t nvals = 0) {
    const uint16_t* x = (const uint16_t*)a.data(); const uint16_t* y = (const uint16_t*)b.data();
    const size_t nv = nvals ? std::min(nvals, a.size() / 2) : a.size() / 2;
    for (size_t i = 0; i < nv; ++i) {
        const float fx = bf2f(x[i]), fy = bf2f(y[i]);
        ++d->n;
        if (x[i] != y[i]) {
            ++d->ndiff;
            auto ord = [](uint16_t v) { return (v & 0x8000) ? -(int)(v & 0x7fff) : (int)v; };
            d->maxulp = std::max(d->maxulp, std::abs(ord(x[i]) - ord(y[i])));
        }
        d->maxabs = std::max(d->maxabs, (double)std::fabs(fx - fy));
        d->maxref = std::max(d->maxref, (double)std::fabs(fx));
        d->sdiff += (double)(fx - fy) * (fx - fy); d->sref += (double)fx * fx;
        d->adiff += std::fabs((double)fx - fy); d->aref += std::fabs((double)fx);
    }
}

/* ---- rule B: the form against itself -- (1) the same call again, (2) each expert's rows reversed (every row in another
 * tile position with other co-rows; for down the activation rows move with their slots), (3) every 3rd live slot dropped:
 * each surviving row's bytes must equal its row in the first run. */
static int g_rb = 0;
static int rule_b_check(Call& c, hipStream_t s, const std::vector<uint8_t>& y0, long long* rows) {
    const Proj& P = kProj[c.p];
    const int64_t TR = (int64_t)c.rt.sorted.size(), N = P.N, K = P.K;
    const bool asort = P.sorted_rows;
    int fail = 0;
    auto run = [&](const std::vector<int32_t>& so, const std::vector<uint16_t>* xperm, std::vector<uint8_t>& out) {
        void* dso = c.ts[2].data; void* dx = c.ts[0].data;
        void* nso = nullptr; void* nx = nullptr;
        CK(hipMalloc(&nso, (size_t)TR * 4)); CK(hipMemcpy(nso, so.data(), (size_t)TR * 4, hipMemcpyHostToDevice));
        c.ts[2].data = nso;
        if (xperm) { CK(hipMalloc(&nx, xperm->size() * 2)); CK(hipMemcpy(nx, xperm->data(), xperm->size() * 2, hipMemcpyHostToDevice)); c.ts[0].data = nx; }
        c.a.t = c.ts.data();
        CK(hipMemsetAsync(c.y, 0xff, c.ybytes, s));
        c.launch(s); CK(hipStreamSynchronize(s));
        out.resize(c.ybytes); CK(hipMemcpy(out.data(), c.y, c.ybytes, hipMemcpyDeviceToHost));
        c.ts[2].data = dso; c.ts[0].data = dx; c.a.t = c.ts.data();
        hipFree(nso); if (nx) hipFree(nx);
    };
    /* (1) repeat */
    { std::vector<uint8_t> o; run(c.rt.sorted, nullptr, o); if (o != y0) ++fail; *rows += TR; }
    /* the permutation: reversed within each expert's live rows */
    std::vector<int64_t> pi((size_t)TR);
    for (int64_t i = 0; i < TR; ++i) pi[(size_t)i] = i;
    for (int64_t e = 0; e + 1 < (int64_t)c.rt.off.size(); ++e) {
        int64_t a = c.rt.off[(size_t)e], b = c.rt.off[(size_t)e + 1];
        for (int64_t i = a; i < b; ++i) pi[(size_t)i] = a + (b - 1 - i);
    }
    std::vector<uint16_t> hx;
    if (asort) { hx.resize((size_t)(TR * K)); CK(hipMemcpy(hx.data(), c.ts[0].data, hx.size() * 2, hipMemcpyDeviceToHost)); }
    for (int variant = 0; variant < 2; ++variant) {
        std::vector<int32_t> so((size_t)TR);
        std::vector<uint16_t> xp;
        for (int64_t i = 0; i < TR; ++i) so[(size_t)i] = c.rt.sorted[(size_t)pi[(size_t)i]];
        if (variant == 1) { int64_t k = 0; for (int64_t i = 0; i < TR; ++i) if (so[(size_t)i] >= 0 && (k++ % 3) == 2) so[(size_t)i] = -1; }
        if (asort) {
            xp.resize(hx.size());
            for (int64_t i = 0; i < TR; ++i) std::memcpy(&xp[(size_t)(i * K)], &hx[(size_t)(pi[(size_t)i] * K)], (size_t)K * 2);
        }
        std::vector<uint8_t> o; run(so, asort ? &xp : nullptr, o);
        for (int64_t i = 0; i < TR; ++i) {
            if (so[(size_t)i] < 0) continue;
            ++*rows;
            if (std::memcmp(&o[(size_t)(i * N * 2)], &y0[(size_t)(pi[(size_t)i] * N * 2)], (size_t)N * 2)) { ++fail; break; }
        }
    }
    return fail;
}

static std::string fmt_stage(int st, double mx, double mean) {
    char b[96]; std::snprintf(b, sizeof b, " s%d max %.1f mean %.1f;", st, mx, mean); return b;
}
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
    if (n == "mx2") { f->on = -2; return true; }   /* the switch drafter_mx 2, no pin */
    if (n == "a8") { f->on = -3; return true; }    /* fp8 quant + moe_gemm_mxfp4a8 a call */
    if (n == "a8g") { f->on = -4; return true; }   /* moe_gemm_mxfp4a8 only (activations quantised before) */
    if (n == "a8q") { f->on = -5; return true; }   /* the fp8 quant only */
    {   /* qNT: the long-run WMMA form (rule B) */
        int a = 0;
        if (std::sscanf(n.c_str(), "q%d", &a) == 1) { f->on = 1; f->h = 4096; f->r = 1; f->mt = a; return true; }
    }
    {   /* wWRnNT[pPD][wWN]: the WMMA form (rule B) */
        int a = 0, b = 0, c = 0, d = 0, ab = 0;
        if (std::sscanf(n.c_str(), "w%dn%dp%da%d", &a, &b, &d, &ab) == 4) { f->on = 1; f->h = 1024 + d + 16 * ab; f->r = a; f->mt = b; return true; }
        if (std::sscanf(n.c_str(), "w%dn%dp%dw%d", &a, &b, &d, &c) == 4) { f->on = 1; f->h = 1024 + d; f->r = a; f->mt = b; f->wn = c; return true; }
        if (std::sscanf(n.c_str(), "w%dn%dp%d", &a, &b, &d) == 3) { f->on = 1; f->h = 1024 + d; f->r = a; f->mt = b; return true; }
        if (std::sscanf(n.c_str(), "w%dn%dw%d", &a, &b, &c) == 3) { f->on = 1; f->h = 1024; f->r = a; f->mt = b; f->wn = c; return true; }
        if (std::sscanf(n.c_str(), "w%dn%d", &a, &b) == 2) { f->on = 1; f->h = 1024; f->r = a; f->mt = b; return true; }
    }
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
    g_a8 = f.on == -3 ? 1 : f.on == -4 ? 2 : f.on == -5 ? 3 : 0;
    if (g_a8) { r4d_mx_mr_pin(0, 0, 0, 0, 0, 0); r4d_mx_mr_set(0); return; }
    if (f.on == 0) { r4d_mx_mr_pin(0, 0, 0, 0, 0, 0); r4d_mx_mr_set(0); }
    else if (f.on == -2) { r4d_mx_mr_pin(0, 0, 0, 0, 0, 0); r4d_mx_mr_set(2); }
    else if (f.on < 0) { r4d_mx_mr_set(1); r4d_mx_mr_pin(-1, 0, 0, 0, 0, 0); }
    else { r4d_mx_mr_set(1); r4d_mx_mr_pin(1, f.h, f.r, f.mt, f.wn, f.grid); }
}

int main(int argc, char** argv) {
    std::string toks = "40,80", forms = "stock,rule", route = "synth";
    int iters = 30, rec_stride = 2, stages = 3, cap = 0, gemv_order = 0;
    std::string model = "drafter", wfile, route_x7;
    int64_t width = 576, wfw = 0;
    int quad = 0;
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
        else if (a == "--rb") g_rb = 1;
        else if (a == "--wslab") r4d_mx_wm_slab(std::atoi(nx().c_str()));
        else if (a == "--model") model = nx();
        else if (a == "--width") width = std::atoll(nx().c_str());
        else if (a == "--wfile") wfile = nx();
        else if (a == "--wfile-width") wfw = std::atoll(nx().c_str());
        else if (a == "--route-x7") route_x7 = nx();
        else if (a == "--quad") quad = 1;
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    if (model == "main") set_main(width);
    else if (model == "dtp") set_dtp(width);
    else if (model != "drafter") { std::fprintf(stderr, "--model is drafter, dtp or main\n"); return 2; }
    /* --quad (the EP vs TP question, DRAFTER round 3): 12 consecutive stage records of rank 0's drafter capture (stride --rec-stride)
     * stand for 3 stages x 4 ranks (stage st, rank r = record 4 st + r). drafter: 12 calls, each its own rank's 32 experts (EP:
     * a stage waits for the slowest of its 4); dtp: 3 calls over all 128 experts, the 4 records' counts side by side (TP). */
    if (quad) stages = model == "drafter" ? 12 : 3;
    if (stages > 12) { std::fprintf(stderr, "--stages <= 12\n"); return 2; }
    if (!route_x7.empty()) route = route_x7;
    std::printf("model %s: %lld experts a rank, top %lld, gate_up %lld x %lld, down %lld x %lld; weights %s\n", model.c_str(), (long long)kE,
                (long long)kTopK, (long long)kProj[0].N, (long long)kProj[0].K, (long long)kProj[1].N, (long long)kProj[1].K,
                wfile.empty() ? "random MXFP4 blocks" : wfile.c_str());
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

    bool want_a8 = false;
    for (auto& f : fl) want_a8 |= f.on <= -3;
    /* weights: stage x projection, 32 experts each, a pointer table; for the a8 arms also the canonical planes (E2M1 codes,
     * element 2i in the low nibble of byte i, [N, K/2]; E8M0 [N, K/32]) of the same values, even / odd experts' tables */
    void* wtab[12][2]; size_t ebytes[2];
    void* w8tab[12][2][2] = {}; void* s8tab[12][2][2] = {};
    /* --wfile (mxslice.py's canonical planes of one layer, expert after expert): the same weights at every stage, each stage its
     * own copy in device memory (no stage reads another's bytes from a cache) */
    std::vector<uint8_t> wf;
    if (!wfile.empty()) {
        FILE* f = std::fopen(wfile.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", wfile.c_str()); return 2; }
        const int64_t W = wfw > 0 ? wfw : kProj[1].K, per = 2 * W * (5120 / 2 + 5120 / 32) + 5120 * (W / 2 + W / 32);
        if (W < kProj[1].K) { std::fprintf(stderr, "--wfile-width %lld is below --width %lld\n", (long long)W, (long long)kProj[1].K); return 2; }
        wf.resize((size_t)(kE * per));
        if (std::fread(wf.data(), 1, wf.size(), f) != wf.size()) { std::fprintf(stderr, "%s: short (want %zu B)\n", wfile.c_str(), wf.size()); return 2; }
        std::fclose(f);
    }
    std::vector<uint8_t> hwf[2];   /* the file's weights as 17-byte blocks, built once a projection */
    for (int st = 0; st < stages; ++st)
        for (int p = 0; p < 2; ++p) {
            const Proj& P = kProj[p];
            const int64_t rb = P.K / 32 * 17; ebytes[p] = (size_t)(P.N * rb);
            std::vector<uint8_t> hwl;
            std::vector<uint8_t>& hw = wf.empty() ? hwl : hwf[p];
            if (wf.empty()) mx_blocks(hw, kE * P.N, P.K, 0xFEEDull ^ (uint64_t)P.N ^ ((uint64_t)st << 40));
            else if (hw.empty()) {
                /* the file holds width WF >= W: gate rows [0, W) and up rows [WF, WF + W) of its 2 WF; down's first W / 2 code
                 * bytes and W / 32 scales of each row */
                const int64_t W = kProj[1].K, WF = wfw > 0 ? wfw : W, per = 2 * WF * (5120 / 2 + 5120 / 32) + 5120 * (WF / 2 + WF / 32), nb = P.K / 32;
                const int64_t fcr = p == 0 ? 5120 / 2 : WF / 2, fsr = p == 0 ? 5120 / 32 : WF / 32;   /* the file's row bytes */
                hw.resize((size_t)(kE * P.N * nb * 17));
                for (int64_t e = 0; e < kE; ++e) {
                    const uint8_t* base = wf.data() + e * per + (p == 0 ? 0 : 2 * WF * (5120 / 2 + 5120 / 32));
                    const uint8_t* cd = base;
                    const uint8_t* sc = base + (p == 0 ? 2 * WF : 5120) * fcr;
                    for (int64_t n = 0; n < P.N; ++n)
                        for (int64_t b = 0; b < nb; ++b) {
                            const int64_t fr = p == 0 && n >= W ? n - W + WF : n;   /* the file's row */
                            uint8_t* q = hw.data() + ((e * P.N + n) * nb + b) * 17;
                            q[0] = sc[fr * fsr + b];
                            const uint8_t* c = cd + fr * fcr + b * 16;
                            uint8_t el[32];
                            for (int i = 0; i < 16; ++i) { el[2 * i] = c[i] & 15; el[2 * i + 1] = c[i] >> 4; }
                            for (int j = 0; j < 16; ++j) q[1 + j] = (uint8_t)(el[j] | (el[j + 16] << 4));
                        }
                }
            }
            uint8_t* d = nullptr; CK(hipMalloc((void**)&d, hw.size()));
            CK(hipMemcpy(d, hw.data(), hw.size(), hipMemcpyHostToDevice));
            std::vector<void*> ptrs((size_t)kE);
            for (int64_t e = 0; e < kE; ++e) ptrs[(size_t)e] = d + e * (int64_t)ebytes[p];
            CK(hipMalloc(&wtab[st][p], kE * sizeof(void*)));
            CK(hipMemcpy(wtab[st][p], ptrs.data(), kE * sizeof(void*), hipMemcpyHostToDevice));
            if (want_a8) {
                const int64_t nb = P.K / 32, cb = P.N * P.K / 2, sbb = P.N * nb;
                std::vector<uint8_t> codes((size_t)(kE * cb)), sc((size_t)(kE * sbb));
                for (int64_t e = 0; e < kE; ++e)
                    for (int64_t n = 0; n < P.N; ++n)
                        for (int64_t b = 0; b < nb; ++b) {
                            const uint8_t* q = hw.data() + ((e * P.N + n) * nb + b) * 17;
                            sc[(size_t)(e * sbb + n * nb + b)] = q[0];
                            uint8_t el[32];
                            for (int j = 0; j < 16; ++j) { el[j] = q[1 + j] & 15; el[j + 16] = q[1 + j] >> 4; }
                            uint8_t* o = codes.data() + e * cb + n * (P.K / 2) + b * 16;
                            for (int i = 0; i < 16; ++i) o[i] = (uint8_t)(el[2 * i] | (el[2 * i + 1] << 4));
                        }
                uint8_t *dc = nullptr, *ds = nullptr;
                CK(hipMalloc((void**)&dc, codes.size())); CK(hipMemcpy(dc, codes.data(), codes.size(), hipMemcpyHostToDevice));
                CK(hipMalloc((void**)&ds, sc.size())); CK(hipMemcpy(ds, sc.data(), sc.size(), hipMemcpyHostToDevice));
                for (int cls = 0; cls < 2; ++cls) {
                    std::vector<void*> pc, ps;
                    for (int64_t e = cls; e < kE; e += 2) { pc.push_back(dc + e * cb); ps.push_back(ds + e * sbb); }
                    CK(hipMalloc(&w8tab[st][p][cls], pc.size() * sizeof(void*)));
                    CK(hipMemcpy(w8tab[st][p][cls], pc.data(), pc.size() * sizeof(void*), hipMemcpyHostToDevice));
                    CK(hipMalloc(&s8tab[st][p][cls], ps.size() * sizeof(void*)));
                    CK(hipMemcpy(s8tab[st][p][cls], ps.data(), ps.size() * sizeof(void*), hipMemcpyHostToDevice));
                }
            }
        }

    int fails = 0;
    for (auto& ts_ : split(toks)) {
        const int64_t tk = std::atoll(ts_.c_str());
        /* routing per stage */
        std::vector<Route> rts;
        if (route == "synth") {
            for (int st = 0; st < stages; ++st) rts.push_back(route_synth(tk, 0, 0xD5A2Cull + (uint64_t)tk + 1000003ull * (uint64_t)st));
        } else if (quad) {
            auto recs = read_records(route.c_str(), tk, 12, rec_stride, 32);
            if ((int)recs.size() < 12) { std::printf("tokens %lld: %zu records (need 12)\n", (long long)tk, recs.size()); continue; }
            if (model == "drafter") for (int i = 0; i < 12; ++i) rts.push_back(route_counts(recs[(size_t)i], tk));
            else {
                for (int st = 0; st < 3; ++st) {
                    std::vector<int32_t> off(129, 0);
                    for (int r = 0; r < 4; ++r)
                        for (int e = 0; e < 32; ++e) {
                            const auto& q = recs[(size_t)(4 * st + r)];
                            off[(size_t)(32 * r + e + 1)] = off[(size_t)(32 * r + e)] + (q[(size_t)e + 1] - q[(size_t)e]);
                        }
                    if (off[128] > tk * kTopK) {   /* more slots than the tokens' picks: cut the last experts' rows */
                        std::printf("  stage %d: %d slots over 4 records > %lld, cut\n", st, off[128], (long long)(tk * kTopK));
                        for (auto& v : off) v = std::min<int32_t>(v, (int32_t)(tk * kTopK));
                    }
                    rts.push_back(route_counts(off, tk));
                }
            }
        } else {
            auto recs = route_x7.empty() ? read_records(route.c_str(), tk, stages, rec_stride)
                                         : read_records_x7(route.c_str(), tk, stages, rec_stride, g_rec_start);
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
                if (want_a8) {
                    c.AR = AR; c.K = P.K; c.N = P.N;
                    CK(hipMalloc(&c.aq, (size_t)(AR * P.K))); CK(hipMalloc(&c.as, (size_t)(AR * (P.K / 128)) * 4));
                    for (int cls = 0; cls < 2; ++cls) { c.w8[cls] = w8tab[st][p][cls]; c.s8[cls] = s8tab[st][p][cls]; }
                    c.ps8 = { PI("N", P.N), PI("K", P.K), PI("n_expert", kE), PI("top_k", kTopK), PI("group", 128),
                              PS("a_order", P.sorted_rows ? "sorted" : "token") };
                }
                calls.push_back(std::move(c));
            }
        for (auto& c : calls) { c.a.t = c.ts.data(); c.a.p = c.ps.data(); }
        if (model == "main") std::printf("== tokens %lld (main model rows, %lld slots):", (long long)tk, (long long)(tk * kTopK));
        else std::printf("== tokens %lld (%lld sequences x 5):", (long long)tk, (long long)(tk / 5));
        for (int st = 0; st < stages; ++st) std::printf(" stage %d live %lld experts, %lld slots, run max %lld;", st, (long long)rts[(size_t)st].experts, (long long)rts[(size_t)st].live, (long long)rts[(size_t)st].rmax);
        std::printf("\n");
        std::vector<std::vector<uint8_t>> ref(calls.size());
        double stock_pass = 0;
        for (const Form& f : fl) {
            apply_form(f);
            /* correctness: each call once, bytes vs the stock path's */
            bool same = true; std::string formdesc;
            DiffStats dst; long long rbn = 0; int rb_fail = 0;
            for (size_t i = 0; i < calls.size(); ++i) {
                CK(hipMemsetAsync(calls[i].y, 0xff, calls[i].ybytes, s));
                const int rc = calls[i].launch(s);
                CK(hipStreamSynchronize(s));
                if (rc != 0) { std::printf("  %s: call %zu rc %d\n", f.name.c_str(), i, rc); same = false; continue; }
                if (g_a8) { formdesc = g_a8 == 3 ? "r4d_quant_act_fp8 only" : "moe_gemm_mxfp4a8 (r4d_rad_moe_mxfp4)"; }
                else {
                    if (i == 0 && f.on != 0) formdesc = r4d_mx_mr_last();
                    if (f.on != 0 && i > 0 && formdesc != r4d_mx_mr_last()) formdesc += std::string(" / ") + r4d_mx_mr_last();
                }
                if (g_a8 == 3) continue;
                std::vector<uint8_t> got(calls[i].ybytes);
                CK(hipMemcpy(got.data(), calls[i].y, calls[i].ybytes, hipMemcpyDeviceToHost));
                if (f.on == 0) ref[i] = got;
                else if (got != ref[i]) {
                    same = false;
                    diff_stats(ref[i], got, &dst, (size_t)(calls[i].rt.live * kProj[calls[i].p].N));
                }
                if (f.on != 0 && g_rb && g_a8 != 2) rb_fail += rule_b_check(calls[i], s, got, &rbn);
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
            std::printf("             a layer-chain (gate_up + down) %.1f us; live-expert bytes a pass %.1f MB (%.2f MB a live expert-chain)\n",
                        pm / stages, bytes / 1e6, (double)(ebytes[0] + ebytes[1]) / 1e6);
            if (!same)
                std::printf("             vs stock: %lld of %lld values differ, max %d bf16 ulp, max |d| / max |ref| %.3g, mean |d| / mean |ref| %.3g, rms rel %.3g\n",
                            dst.ndiff, dst.n, dst.maxulp, dst.maxref > 0 ? dst.maxabs / dst.maxref : 0.0,
                            dst.aref > 0 ? dst.adiff / dst.aref : 0.0,
                            dst.n > 0 && dst.sref > 0 ? std::sqrt(dst.sdiff / dst.sref) : 0.0);
            if (f.on != 0 && g_rb)
                std::printf("             rule B: %s (repeat + rows reversed within experts + every 3rd slot dropped: %lld rows compared)\n",
                            rb_fail ? "FAIL" : "deterministic and placement-invariant", rbn);
            if (rb_fail) ++fails;
            std::printf("             calls us:");
            for (size_t i = 0; i < nc; ++i) {
                const double m = med(t[i]);
                std::printf(" s%d.%s %.1f (%.0f GB/s)", calls[i].stage, kProj[calls[i].p].name, m,
                            (double)calls[i].rt.experts * (double)ebytes[calls[i].p] / (m * 1e3));
            }
            std::printf("\n");
            if (quad && model == "drafter" && nc == 24) {   /* EP: a stage = the slowest of its 4 ranks */
                double crit = 0, mean = 0;
                std::string per;
                for (int st = 0; st < 3; ++st) {
                    double mx = 0, sm = 0;
                    for (int r = 0; r < 4; ++r) { const size_t i = (size_t)(2 * (4 * st + r)); const double v = med(t[i]) + med(t[i + 1]); mx = std::max(mx, v); sm += v; }
                    crit += mx; mean += sm / 4;
                    per += fmt_stage(st, mx, sm / 4);
                }
                std::printf("             EP (4 ranks' records a stage): critical (slowest rank a stage) %.1f us a pass, mean rank %.1f |%s\n", crit, mean, per.c_str());
            }
            if (f.on != 0 && !same && !((f.h >= 1024 || f.on == -2) && g_rb) && f.on > -3) ++fails;
            std::fflush(stdout);
        }
        apply_form(fl[0]);
        for (auto& c : calls) { hipFree(c.ts[0].data); hipFree(c.ts[2].data); hipFree(c.ts[3].data); hipFree(c.y); if (c.aq) hipFree(c.aq); if (c.as) hipFree(c.as); }
    }
    std::printf("DBVERDICT %s\n", fails ? "SOME FORM DIFFERS" : "all forms byte-identical to the stock path");
    return fails ? 1 : 0;
}
