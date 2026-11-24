// tests/r4d_emu/test_i8wide.cpp -- fn4 BI wide (14.11.2026): libr4d's gemm_nt_q_dec (gemm_i8a8_nt_m16 over 64-token
// groups on grid.y) run on the host by the SIMT emulator, against the narrow row at alone heights.
//
// The claim the engine relies on: a decode row's int8 linear bytes do not depend on the step's row count. Alone, a
// decode sequence's linears run gemm_i8a8_nt_m16 over its own 1 + n_spec rows (4 at depth 3); under load above 64 decode
// rows gemm_nt_q resolves to gemm_i8a8_tiled (an fma fold, different bytes), and gemm_nt_q_dec keeps the narrow fold.
//
//   ref    gemm_i8a8_nt_m16 over every 4-row slice (a sequence's alone launch), and over 1-row slices on one shape
//   dec    gemm_i8a8_nt_m16_dec over [0, M) at M from 1 to 132 (one shape) and at 64 / 65 / 96 / 100 / 128 (the
//          Flash-Next TP4 per-rank shapes: q|gate, k/v (split 5), attention out, GDN in / out, shared gate_up (split
//          2), shared down (K 640: the 128-wide K tile))
//   check  every element of every row equal to ref; at M <= 64 also equal to gemm_i8a8_nt_m16 at M (the same launch)
//   control gemm_i8a8_tiled over [0, M) at 96 / 128: its differing elements against ref are printed (the reason for the
//          op; not a pass condition)
//
// Weights, scales and activations are seeded bytes: the comparison needs only that both launches read the same bytes.
#include "hip/hip_runtime.h"   // tests/r4d_emu/emu: the emulator
#include "rad_abi.h"
#include "rad_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs*);
extern "C" int r4d_rad_gemm_i8a8_m16_dec(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_gemm_i8a8_m16_dec_scratch(const RadArgs*);
extern "C" int r4d_rad_gemm_i8a8_tiled(const RadArgs*, RadStream);
extern "C" void r4d_gemm_fp8a8_pin_pp(int);   /* lever F8P (Qwen loop round 37): -1 env, 0 today, 1 the twin */

static int g_fail = 0;
static void ok(const std::string& what, bool c, const std::string& d = "") {
    std::printf("  %s  %s%s%s\n", c ? "ok  " : "FAIL", what.c_str(), d.empty() ? "" : "  -- ", d.c_str());
    std::fflush(stdout);
    if (!c) ++g_fail;
}
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> sh) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = d; t.dtype = dt; t.rank = (uint32_t)sh.size();
    int64_t acc = 1;
    for (size_t i = 0; i < sh.size(); ++i) t.shape[i] = sh[i];
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= sh[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

template <class X>
struct Buf {   // 64-byte aligned, as the arena's buffers
    X* p = nullptr; size_t n = 0;
    explicit Buf(size_t n_) : n(n_) { p = (X*)std::aligned_alloc(64, ((n * sizeof(X) + 63) / 64) * 64); }
    ~Buf() { std::free(p); }
    Buf(const Buf&) = delete; Buf& operator=(const Buf&) = delete;
};

struct Shape { const char* name; int64_t N, K; };

struct Case {
    int64_t Mx, N, K, sk;
    Buf<int8_t> a, w; Buf<float> as; Buf<uint16_t> ws;
    Case(int64_t Mx_, int64_t N_, int64_t K_, uint32_t seed)
        : Mx(Mx_), N(N_), K(K_), sk(K_ / 128), a((size_t)(Mx_ * K_)), w((size_t)(N_ * K_)),
          as((size_t)(Mx_ * (K_ / 128))), ws((size_t)(N_ * (K_ / 128))) {
        for (size_t i = 0; i < a.n; ++i) a.p[i] = (int8_t)((int)(mix((uint32_t)i * 2u + 1u + seed) % 255u) - 127);
        for (size_t i = 0; i < w.n; ++i) w.p[i] = (int8_t)((int)(mix((uint32_t)i * 2u + 0x5bd1e995u + seed) % 255u) - 127);
        for (size_t i = 0; i < as.n; ++i)
            as.p[i] = std::ldexp(1.0f + (mix((uint32_t)i + 77 + seed) & 255) / 256.0f, -(int)(mix((uint32_t)i + 99) % 9) - 7);
        for (size_t i = 0; i < ws.n; ++i)
            ws.p[i] = to_bf(std::ldexp(1.0f + (mix((uint32_t)i + 5 + seed) & 127) / 128.0f, -(int)(mix((uint32_t)i + 7) % 7) - 6));
    }
    /* One launch over rows [m0, m0 + mm) into y (rows at their own offsets). */
    int go(int (*l)(const RadArgs*, RadStream), int64_t (*h)(const RadArgs*), int64_t m0, int64_t mm, uint16_t* y) {
        std::vector<RadTensor> ts = {
            T(a.p + m0 * K, RAD_I8, { mm, K }), T(as.p + m0 * sk, RAD_F32, { mm, sk }),
            T(w.p, RAD_I8, { N, K }), T(ws.p, RAD_BF16, { N, sk }),
            T(y + m0 * N, RAD_BF16, { mm, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }) };
        ts[5].rank = 0; ts[6].rank = 0;
        std::vector<RadParam> ps = { PI("M", mm), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "i8a8h") };
        RadArgs ra; std::memset(&ra, 0, sizeof ra);
        ra.t = ts.data(); ra.n_t = (int)ts.size(); ra.p = ps.data(); ra.n_p = (int)ps.size(); ra.world_size = 1;
        std::vector<char> scratch;
        if (h) {
            const int64_t need = h(&ra);
            if (need > 0) { scratch.assign((size_t)need + 64, (char)0x5a); ra.scratch = scratch.data(); ra.scratch_bytes = need; }
        }
        return l(&ra, nullptr);
    }
};

static size_t ndiff(const uint16_t* x, const uint16_t* y, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += x[i] != y[i];
    return d;
}

/* ref: the narrow row over q-row slices of [0, Mx) (a sequence's alone launch at depth q - 1). */
static bool ref_slices(Case& c, int64_t q, std::vector<uint16_t>& ref) {
    ref.assign((size_t)(c.Mx * c.N), 0xffff);
    for (int64_t m0 = 0; m0 < c.Mx; m0 += q)
        if (c.go(r4d_rad_gemm_i8a8_m16, r4d_rad_gemm_i8a8_m16_scratch, m0, std::min(q, c.Mx - m0), ref.data()) != 0)
            return false;
    return true;
}

static void shape_case(const Shape& s, const std::vector<int64_t>& Ms, bool control, bool rows1) {
    const int64_t Mx = *std::max_element(Ms.begin(), Ms.end());
    Case c(Mx, s.N, s.K, 0x1234u + (uint32_t)s.N);
    const std::string tag = std::string(s.name) + " N=" + std::to_string(s.N) + " K=" + std::to_string(s.K);
    std::vector<uint16_t> ref, ref1;
    if (!ref_slices(c, 4, ref)) { ok(tag + " ref (m16 over 4-row slices)", false, "launch failed"); return; }
    if (rows1) {
        if (!ref_slices(c, 1, ref1)) { ok(tag + " ref (1-row slices)", false, "launch failed"); return; }
        const size_t d = ndiff(ref.data(), ref1.data(), ref.size());
        ok(tag + " m16 4-row slices == 1-row slices", d == 0, std::to_string(d) + " differ");
    }
    for (int64_t M : Ms) {
        std::vector<uint16_t> y((size_t)(Mx * s.N), 0xffff);
        const int rc = c.go(r4d_rad_gemm_i8a8_m16_dec, r4d_rad_gemm_i8a8_m16_dec_scratch, 0, M, y.data());
        if (rc) { ok(tag + " dec M=" + std::to_string(M), false, "rc " + std::to_string(rc)); continue; }
        const size_t d = ndiff(y.data(), ref.data(), (size_t)(M * s.N));
        std::string extra;
        if (M <= 64) {
            std::vector<uint16_t> y2((size_t)(Mx * s.N), 0xffff);
            const int rc2 = c.go(r4d_rad_gemm_i8a8_m16, r4d_rad_gemm_i8a8_m16_scratch, 0, M, y2.data());
            const size_t d2 = rc2 ? (size_t)-1 : ndiff(y.data(), y2.data(), (size_t)(M * s.N));
            extra = ", m16 at M: " + std::to_string(d2) + " differ";
            if (d2 != 0) { ok(tag + " dec M=" + std::to_string(M) + " == m16 at M", false, extra); }
        }
        ok(tag + " dec M=" + std::to_string(M) + " == alone (4-row slices)", d == 0,
           std::to_string(d) + " of " + std::to_string(M * s.N) + " differ" + extra);
        if (control && M > 64) {
            std::vector<uint16_t> yt((size_t)(Mx * s.N), 0xffff);
            const int rct = c.go(r4d_rad_gemm_i8a8_tiled, nullptr, 0, M, yt.data());
            const size_t dt = rct ? (size_t)-1 : ndiff(yt.data(), ref.data(), (size_t)(M * s.N));
            std::printf("  info  %s tiled M=%lld vs alone: %zu of %lld elements differ (today's form above 64 rows)\n",
                        tag.c_str(), (long long)M, dt, (long long)(M * s.N));
        }
    }
}

/* lever F8P (Qwen loop round 37, QUEUE Q66): gemm_nt_q_dec with two weight K tiles in flight a wave (the twin kernel) against
 * today's, every element at each height; shapes cover 1 / 2 / 3 / 5 / 6 / 10 K tiles a block, FK 256 and 128, split K (k/v: a
 * slice of 2 tiles starting past 0) and the 64-token groups on grid.y. R4D_EMU_F8P_MUTATE=1 (the twin skips its refill) must
 * FAIL every shape with 3 or more tiles a block: the check sees the twin's own loop. */
static void f8p_case(const Shape& s, const std::vector<int64_t>& Ms) {
    const int64_t Mx = *std::max_element(Ms.begin(), Ms.end());
    Case c(Mx, s.N, s.K, 0x9e37u + (uint32_t)s.N + (uint32_t)s.K);
    const std::string tag = std::string("F8P ") + s.name + " N=" + std::to_string(s.N) + " K=" + std::to_string(s.K);
    for (int64_t M : Ms) {
        std::vector<uint16_t> y0((size_t)(Mx * s.N), 0xffff), y1((size_t)(Mx * s.N), 0xfffe);
        r4d_gemm_fp8a8_pin_pp(0);
        const int rc0 = c.go(r4d_rad_gemm_i8a8_m16_dec, r4d_rad_gemm_i8a8_m16_dec_scratch, 0, M, y0.data());
        const char* k0 = emu_last_kernel;
        r4d_gemm_fp8a8_pin_pp(1);
        const int rc1 = c.go(r4d_rad_gemm_i8a8_m16_dec, r4d_rad_gemm_i8a8_m16_dec_scratch, 0, M, y1.data());
        const char* k1 = emu_last_kernel;
        r4d_gemm_fp8a8_pin_pp(-1);
        if (rc0 || rc1) { ok(tag + " M=" + std::to_string(M), false, "rc " + std::to_string(rc0) + " / " + std::to_string(rc1)); continue; }
        const size_t d = ndiff(y0.data(), y1.data(), (size_t)(M * s.N));
        ok(tag + " M=" + std::to_string(M) + " twin == today", d == 0,
           std::to_string(d) + " of " + std::to_string(M * s.N) + " differ (" + (k1 ? k1 : "?") + " vs " + (k0 ? k0 : "?") + ")");
    }
}

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    /* every height 1..132 on a split shape (N 256, K 512: two K slices) */
    if (only.empty() || only == "sweep") {
        std::vector<int64_t> Ms;
        for (int64_t m = 1; m <= 132; ++m) Ms.push_back(m);
        shape_case({ "sweep k/v-like", 256, 512 }, Ms, false, true);
    }
    /* Flash-Next TP4 per-rank shapes (n_embd 2560) */
    if (only.empty() || only == "fn") {
        const std::vector<int64_t> Ms = { 4, 64, 65, 96, 100, 128 };
        const Shape sh[] = {
            { "k/v (split 5)", 256, 2560 },
            { "shared gate_up (split 2)", 1280, 2560 },
            { "shared down (K 640)", 2560, 640 },
            { "attn out", 2560, 1536 },
            { "q|gate (MW 16)", 3072, 2560 },
            { "gdn in (MW 16)", 4096, 2560 },
        };
        for (const Shape& s : sh) shape_case(s, Ms, true, false);
    }
    if (only.empty() || only == "f8p" || only == "f8pq") {
        const bool q = only == "f8pq";
        const std::vector<int64_t> Ms = q ? std::vector<int64_t>{ 5, 33, 65 } : std::vector<int64_t>{ 1, 5, 16, 17, 33, 48, 64, 65, 100, 128 };
        const Shape sh[] = {
            { "one tile", 256, 256 }, { "three 128-wide tiles", 256, 384 }, { "three tiles", 512, 768 },
            { "k/v (split 5)", 256, 2560 }, { "shared down (K 640, 5 x 128)", 2560, 640 },
            { "attn out / gdn out (6 tiles)", 2560, 1536 }, { "gdn in (MW 16, 10 tiles)", 4096, 2560 },
        };
        for (const Shape& s : sh) f8p_case(s, Ms);
    }
    std::printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
