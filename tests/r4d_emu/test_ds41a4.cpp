/* tests/r4d_emu: lever A4 (ds41-a4) -- the gg_gemv_m16 forms of lever D2's multi-row decode band (r4d_ds41_ggml_mr.hip:
 * the XP unit's kernels r4d_ds41_ggml_mr2.hip, the grid fit's and odd WN, H1R2, MT 4..16) on the host SIMT emulator, byte
 * for byte against the stock decode kernel (gg_gemv_kernel, r4d_gg_mr_emu_stock):
 *
 *   plain   r4d_gg_mr_gemm under every A4 pin (r4d_gg_mr_pin + r4d_gg_mr_pin_xp 1 / 2) and under the switch with a table
 *           (r4d_gg_mr_m16_load + r4d_gg_mr_m16_set), q5_K / q6_K / q8_0, bf16 and f32 outputs, at every DeepSeek-V4.1
 *           dense K (512 768 1280 2048 4096 5120 6144), N 45 and 77 (partial row groups, slots and grid-fit blocks), M 1..16
 *           and 17 / 24 / 48 (the XP forms' MT <= 16 tile above 16 rows); then the c8 profile's real TP4 (N, K) at M 5 / 8 /
 *           11 / 16 under the grid fit's XP form and the switch;
 *   pair    r4d_gg_mr_pair (bf16) / r4d_gg_mr_pair_f32 under the XP pins and the switch, against the two stock launches,
 *           K 4096 / 5120 at reduced N, M 1..16;
 *   folds   r4d_gg_mr_fold a / e under the XP pins and the switch, against the same fold in D2's own kernels (pin H4R1:
 *           tests/r4d_emu ds41fold2 holds D2's fold to the producer + the stock kernel), h and y, M 1..16.
 * Activations: random bf16 with zeros of both signs, subnormals, infinities and huge values (inf, inf - inf NaN sums).
 *
 *   r4d_emu_ds41a4 [quick]      quick: three K values, fewer M */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_mr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int r4d_gg_mr_emu_stock(const R4dGgMrCall& c, int rpw);
int r4d_gg_mr_pair_f32(RadStream s, const R4dGgMrCall& a, const R4dGgMrCall& b);
extern "C" void r4d_gg_mr_pin(int h, int r, int mt, int wn);
extern "C" void r4d_gg_mr_pin_xp(int v);
extern "C" const char* r4d_gg_mr_last();
extern "C" int r4d_gg_mr_m16_load(const char* path, int reset);
extern "C" int r4d_gg_mr_m16_forms();

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };

uint64_t g_rs = 0xA4A4A4A45EEDull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
uint16_t rnd_f16() {
    const uint32_t k = (uint32_t)(rnd() % 100);
    const uint16_t sign = (uint16_t)((rnd() & 1) << 15);
    uint16_t e;
    if (k < 8) e = 0;
    else if (k < 12) e = (uint16_t)(28 + rnd() % 3);
    else e = (uint16_t)(5 + rnd() % 8);
    return (uint16_t)(sign | (e << 10) | (uint16_t)(rnd() & 0x3ff));
}
void fill_row(uint16_t* x, int K, int mode, float scale = 4.0f) {
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 1000);
        if (c < 5) x[k] = 0x0000;
        else if (c < 10) x[k] = 0x8000;
        else if (c < 15) x[k] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
        else x[k] = f2bf(scale * (float)((int)(rnd() % 20001) - 10000) / 10000.0f);
    }
    if (mode == 1) x[rnd() % K] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | 0x7f80);
    if (mode == 2)
        for (int e = 0; e < 3; ++e)
            x[rnd() % K] = f2bf((rnd() & 1 ? -1.0f : 1.0f) * std::ldexp(1.0f + (float)(rnd() % 1000) / 1000.0f, 110 + (int)(rnd() % 17)));
}
struct Ty { int ty; const char* name; int qk, bb; };
const Ty kTypes[] = { { Q5_K, "q5_K", 256, 176 }, { Q6_K, "q6_K", 256, 210 }, { Q8_0, "q8_0", 32, 34 } };
std::vector<uint8_t> make_w(const Ty& t, int N, int K) {
    const long long rb = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> w((size_t)(N * rb));
    for (auto& b : w) b = (uint8_t)rnd();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / t.qk; ++b) {
            uint8_t* blk = w.data() + n * rb + (long long)b * t.bb;
            const uint16_t d = rnd_f16(), dm = rnd_f16();
            if (t.ty == Q6_K) std::memcpy(blk + 208, &d, 2);
            else { std::memcpy(blk, &d, 2); if (t.ty == Q5_K) std::memcpy(blk + 2, &dm, 2); }
        }
    return w;
}

int g_fail = 0, g_cases = 0;
/* h r mt wn xp; xp 16 = the switch with the test table (no pin) */
struct Pin { int h, r, mt, wn, xp; };
const Pin kPins[] = { { 1, 1, 0, -1, 1 }, { 1, 2, 0, 0, 1 }, { 1, 2, 0, -1, 1 }, { 1, 2, 6, 3, 1 }, { 2, 1, 0, 0, 1 },
                      { 2, 1, 0, -1, 1 }, { 2, 2, 0, -1, 1 }, { 2, 2, 8, 5, 1 }, { 4, 1, 0, -1, 1 }, { 4, 1, 12, 0, 1 },
                      { 4, 2, 0, 0, 1 }, { 4, 2, 0, -1, 1 }, { 4, 2, 4, 7, 1 }, { 2, 1, 0, -1, 2 }, { 4, 2, 0, -1, 2 },
                      { 1, 1, 0, 5, 2 }, { 0, 0, 0, 0, 16 } };
std::string pin_name(const Pin& p) {
    if (p.xp == 16) return "switch";
    char b[64];
    std::snprintf(b, sizeof b, "H%dR%d c%d w%d %s", p.h, p.r, p.mt, p.wn, p.xp == 1 ? "xp" : "a4");
    return b;
}
void set_pin(const Pin& p) {
    if (p.xp == 16) { r4d_gg_mr_pin(0, 0, 0, 0); r4d_gg_mr_pin_xp(0); r4d_gg_mr_m16_set(1); return; }
    r4d_gg_mr_m16_set(0);
    r4d_gg_mr_pin(p.h, p.r, p.mt, p.wn);
    r4d_gg_mr_pin_xp(p.xp);
}
void clear_pin() { r4d_gg_mr_pin(0, 0, 0, 0); r4d_gg_mr_pin_xp(0); r4d_gg_mr_m16_set(0); }

/* A table for the switch: every (type, N, K) the test runs, M bands 1-5 / 6-10 / 11-16 (pair 0 and 1) cycling through
 * forms, XP and D2 kernels, rule / fit / odd WN. */
void write_table(const char* path, const std::vector<int>& Ns, const std::vector<int>& Ks) {
    FILE* f = std::fopen(path, "w");
    const int forms[][5] = { { 1, 2, 0, -1, 1 }, { 2, 1, 8, 3, 1 }, { 4, 2, 0, 0, 1 }, { 2, 2, 0, -1, 0 }, { 4, 1, 6, 5, 1 },
                             { 1, 1, 0, 7, 1 } };
    int k = 0;
    for (const Ty& t : kTypes)
        for (int N : Ns)
            for (int K : Ks)
                for (int pair = 0; pair < 2; ++pair)
                    for (int b = 0; b < 3; ++b) {
                        const int lo = b == 0 ? 1 : b == 1 ? 6 : 11, hi = b == 0 ? 5 : b == 1 ? 10 : 16;
                        const int* fm = forms[k++ % 6];
                        std::fprintf(f, "%s %d %d %d %d %d %d %d %d %d %d\n", t.name, N, K, lo, hi, pair, fm[0], fm[1], fm[2], fm[3], fm[4]);
                    }
    std::fprintf(f, "# malformed on purpose:\nq9_K 1 1 1 1 0 1 1 0 0 1\n");
    std::fclose(f);
}

void check_plain(const Ty& t, int N, int K, int M, const std::vector<uint8_t>& w, const std::vector<Pin>& pins, bool f32) {
    std::vector<uint16_t> x((size_t)M * K);
    for (int m = 0; m < M; ++m) { const int r8 = (int)(rnd() % 8); fill_row(x.data() + (size_t)m * K, K, r8 == 0 ? 1 : r8 == 1 ? 2 : 0); }
    const int x_al = (((uintptr_t)x.data() & 15u) == 0) ? 1 : 0;
    const size_t ob = f32 ? 4 : 2;
    const long long w_ld = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> ref((size_t)M * N * ob, 0xA5), got((size_t)M * N * ob);
    R4dGgMrCall c{ t.ty, f32, x.data(), (long long)K, x_al, w.data(), w_ld, ref.data(), (long long)N, M, N, K };
    clear_pin();
    if (r4d_gg_mr_emu_stock(c, 1) != 0) { std::printf("FAIL stock launch %s K=%d M=%d\n", t.name, K, M); ++g_fail; return; }
    for (const Pin& p : pins) {
        std::fill(got.begin(), got.end(), 0xC3);
        c.y = got.data();
        set_pin(p);
        const int rc = r4d_gg_mr_gemm(nullptr, c);
        const std::string form = r4d_gg_mr_last();
        clear_pin();
        ++g_cases;
        if (rc != 0) { std::printf("FAIL %s%s N=%d K=%d M=%d %s: rc %d\n", t.name, f32 ? " f32" : "", N, K, M, pin_name(p).c_str(), rc); ++g_fail; continue; }
        if (got != ref) {
            size_t bad = 0, first = 0;
            for (size_t e = 0; e < got.size(); e += ob)
                if (std::memcmp(&got[e], &ref[e], ob)) { if (!bad) first = e / ob; ++bad; }
            std::printf("FAIL %s%s N=%d K=%d M=%d %s (%s): %zu of %zu outputs differ, first m=%zu n=%zu\n", t.name,
                        f32 ? " f32" : "", N, K, M, pin_name(p).c_str(), form.c_str(), bad, got.size() / ob, first / N, first % N);
            ++g_fail;
        }
    }
}

void check_pair(const Ty& t, int K, int NA, int NB, int M, const std::vector<Pin>& pins, bool f32) {
    const std::vector<uint8_t> wa = make_w(t, NA, K), wb = make_w(t, NB, K);
    const long long w_ld = (long long)K / t.qk * t.bb;
    std::vector<uint16_t> x((size_t)M * K);
    for (int m = 0; m < M; ++m) fill_row(x.data() + (size_t)m * K, K, (int)(rnd() % 6) == 0 ? 1 : 0);
    const size_t ob = f32 ? 4 : 2;
    std::vector<uint8_t> ra((size_t)M * NA * ob, 0x11), rb((size_t)M * NB * ob, 0x22);
    R4dGgMrCall a{ t.ty, f32, x.data(), (long long)K, 1, wa.data(), w_ld, ra.data(), (long long)NA, M, NA, K };
    R4dGgMrCall b{ t.ty, f32, x.data(), (long long)K, 1, wb.data(), w_ld, rb.data(), (long long)NB, M, NB, K };
    clear_pin();
    if (r4d_gg_mr_emu_stock(a, 1) != 0 || r4d_gg_mr_emu_stock(b, 1) != 0) { std::printf("FAIL pair stock %s\n", t.name); ++g_fail; return; }
    for (const Pin& p : pins) {
        std::vector<uint8_t> ga(ra.size(), 0x33), gb(rb.size(), 0x44);
        R4dGgMrCall a2 = a, b2 = b;
        a2.y = ga.data(); b2.y = gb.data();
        set_pin(p);
        const int rc = f32 ? r4d_gg_mr_pair_f32(nullptr, a2, b2) : r4d_gg_mr_pair(nullptr, a2, b2);
        const std::string form = r4d_gg_mr_last();
        clear_pin();
        ++g_cases;
        if (rc != 0 || ga != ra || gb != rb) {
            std::printf("FAIL pair %s%s K=%d %d+%d M=%d %s (%s): rc %d, A %s, B %s\n", t.name, f32 ? " f32" : "", K, NA, NB, M,
                        pin_name(p).c_str(), form.c_str(), rc, ga == ra ? "same" : "DIFFERS", gb == rb ? "same" : "DIFFERS");
            ++g_fail;
        }
    }
}

/* fold a (pro 1) / e (pro 2) under each pin vs the same fold in D2's kernels (pin H4R1) */
void check_fold(int pro, const Ty& t, int K, int N, int M, const std::vector<Pin>& pins) {
    const std::vector<uint8_t> w = make_w(t, N, K);
    const long long w_ld = (long long)K / t.qk * t.bb;
    const int xc = pro == 1 ? K : 2 * K;
    std::vector<uint16_t> x((size_t)M * xc + 16);
    for (int m = 0; m < M; ++m) fill_row(x.data() + (size_t)m * xc, xc, (int)(rnd() % 6) == 0 ? 1 : 0, pro == 2 && m % 3 == 1 ? 30.0f : 4.0f);
    std::vector<float> g((size_t)K);
    for (auto& v : g) { const uint32_t c = (uint32_t)(rnd() % 100); v = c < 5 ? 0.0f : c < 10 ? -0.5f : 0.75f + (float)(rnd() % 1000) / 2000.0f; }
    auto run = [&](const Pin* p, std::vector<uint16_t>& h, std::vector<uint16_t>& y) {
        R4dGgMrCall c{ t.ty, false, x.data(), (long long)xc, 1, w.data(), w_ld, y.data(), (long long)N, M, N, K };
        R4dGgMrFold f{};
        f.pro = pro; f.g = g.data(); f.eps = 1e-20f; f.wadd = 0.0f; f.mult = 1.0f;
        f.nw = 32;   /* r4d_act_threads(M <= 32, K) / 32 */
        f.limit = 10.0f;
        f.h = h.data(); f.h_ld = K; f.h_al = 1; f.u_al = (K % 8) == 0 ? 1 : 0;
        if (p) set_pin(*p);
        else { r4d_gg_mr_m16_set(0); r4d_gg_mr_pin(4, 1, 0, 0); r4d_gg_mr_pin_xp(0); }
        const int rc = r4d_gg_mr_fold(nullptr, c, f);
        clear_pin();
        return rc;
    };
    std::vector<uint16_t> hr((size_t)M * K, 0x5A5A), yr((size_t)M * N, 0xA5A5);
    if (run(nullptr, hr, yr) != 0) { std::printf("FAIL fold ref %s K=%d M=%d\n", t.name, K, M); ++g_fail; return; }
    for (const Pin& p : pins) {
        std::vector<uint16_t> h((size_t)M * K, 0xC3C3), y((size_t)M * N, 0x3C3C);
        const int rc = run(&p, h, y);
        ++g_cases;
        if (rc != 0 || h != hr || y != yr) {
            std::printf("FAIL fold %c %s K=%d N=%d M=%d %s (%s): rc %d, h %s, y %s\n", pro == 1 ? 'a' : 'e', t.name, K, N, M,
                        pin_name(p).c_str(), r4d_gg_mr_last(), rc, h == hr ? "same" : "DIFFERS", y == yr ? "same" : "DIFFERS");
            ++g_fail;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const std::vector<int> Ks = quick ? std::vector<int>{ 768, 1280, 5120 } : std::vector<int>{ 512, 768, 1280, 2048, 4096, 5120, 6144 };
    std::vector<int> Ms;
    for (int M = 1; M <= 16; ++M) if (!quick || M == 1 || M == 5 || M == 8 || M == 11 || M == 16) Ms.push_back(M);
    Ms.push_back(17); if (!quick) { Ms.push_back(24); Ms.push_back(48); }
    const std::vector<int> Ns = { 45, 77 };
    /* the switch's table: the test's keys plus the real shapes below, loaded as RADIANCE_DS41_GG_M16_FORMS would be */
    const char* tpath = "r4d_emu_ds41a4_table.txt";
    std::vector<int> tNs = Ns;
    for (int n : { 8192, 5120, 1280, 1024, 1536, 512 }) tNs.push_back(n);
    std::vector<int> tKs = Ks;
    for (int k : { 512, 768, 1280, 2048, 4096, 5120, 6144 }) { bool have = false; for (int q : tKs) have = have || q == k; if (!have) tKs.push_back(k); }
    write_table(tpath, tNs, tKs);
    const int loaded = r4d_gg_mr_m16_load(tpath, 1);
    if (loaded != -3) { std::printf("FAIL table load: %d (expected -3: one malformed line)\n", loaded); ++g_fail; }
    std::printf("table: %d entries\n", r4d_gg_mr_m16_forms());
    const std::vector<Pin> pins(std::begin(kPins), std::end(kPins));
    for (const Ty& t : kTypes)
        for (int K : Ks) {
            if (K % t.qk) continue;
            for (int N : Ns) {
                const std::vector<uint8_t> w = make_w(t, N, K);
                for (int M : Ms)
                    for (int f32 = 0; f32 < 2; ++f32) check_plain(t, N, K, M, w, pins, f32 != 0);
            }
            std::printf("plain %s K=%d: done (%d cases, %d failures so far)\n", t.name, K, g_cases, g_fail);
            std::fflush(stdout);
        }
    /* the c8 profile's real TP4 shapes at a few M: the grid fit's XP form, the H1R2 XP form and the switch */
    struct RS { int ty; int N, K; };
    const RS real[] = { { Q6_K, 8192, 1280 }, { Q5_K, 5120, 2048 }, { Q6_K, 1536, 5120 }, { Q8_0, 5120, 768 }, { Q6_K, 512, 5120 } };
    const std::vector<Pin> rpins = { { 4, 2, 0, -1, 1 }, { 1, 2, 0, -1, 1 }, { 0, 0, 0, 0, 16 } };
    for (const RS& r : real) {
        const Ty& t = r.ty == Q5_K ? kTypes[0] : r.ty == Q6_K ? kTypes[1] : kTypes[2];
        if (quick && r.N > 2048) continue;
        const std::vector<uint8_t> w = make_w(t, r.N, r.K);
        for (int M : { 5, 8, 11, 16 }) check_plain(t, r.N, r.K, M, w, rpins, false);
        std::printf("real %s %dx%d: done (%d cases, %d failures so far)\n", t.name, r.N, r.K, g_cases, g_fail);
        std::fflush(stdout);
    }
    for (const Ty& t : kTypes)
        for (int K : { 4096, 5120 }) {
            for (int M : Ms) {
                if (M > 16) continue;
                check_pair(t, K, 45, 21, M, pins, false);
                check_pair(t, K, 45, 21, M, pins, true);
            }
            std::printf("pair %s K=%d: done (%d cases, %d failures so far)\n", t.name, K, g_cases, g_fail);
            std::fflush(stdout);
        }
    for (int pro = 1; pro <= 2; ++pro)
        for (const Ty& t : kTypes)
            for (int K : pro == 1 ? std::vector<int>{ 1280 } : std::vector<int>{ 768, 512 }) {
                if (K % t.qk) continue;
                for (int M : Ms) if (M <= 16) check_fold(pro, t, K, 45, M, pins);
                std::printf("fold %c %s K=%d: done (%d cases, %d failures so far)\n", pro == 1 ? 'a' : 'e', t.name, K, g_cases, g_fail);
                std::fflush(stdout);
            }
    std::remove(tpath);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
