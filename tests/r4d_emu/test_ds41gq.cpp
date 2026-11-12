/* tests/r4d_emu: ds41-int11's lever 1U-GQ on the host SIMT emulator -- gemm_nt_ggml_pair's decode-band launch
 * (r4d_ds41_ggml_mr.hip r4d_gg_mr_pair: two calls of one type, K and M in one grid of lever D2's form) against
 *   (1) the same two calls as two launches of the form at the same pin (r4d_gg_mr_gemm), and
 *   (2) the stock decode kernel (gg_gemv_kernel, r4d_gg_mr_emu_stock), each call alone,
 * byte for byte, bf16 and f32 outputs, q5_K / q6_K / q8_0:
 *   - q_a + attn_kv: two weights over ONE input (a2 = a), separate outputs;
 *   - attention_output_a g0 + g1: two inputs (the column halves of one wider buffer), outputs into the column halves of
 *     one wider buffer (y / y2 at y_ld = 2 N, the other half and the padding left untouched);
 * at reduced N (45 + 21 rows: partial row groups and slots, so both calls' column ranges end mid-block) over the real K
 * values (5120, 4096) at M 1 / 2 / 3 / 6 / 8 / 16 / 48 and several pinned forms (each H / R / MT cap / WN), plus the
 * real TP4 shapes (1280 + 512 x 5120, 1024 + 1024 x 4096) at the table's / rule's form for M 1 / 6. Controls: the
 * pair with B given A's weight must differ from B's reference (the check sees B's rows), and a declined pair (types or
 * M differing) must say so.
 *
 *   r4d_emu_ds41gq [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_mr.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int r4d_gg_mr_emu_stock(const R4dGgMrCall& c, int rpw);
int r4d_gg_mr_pair_f32(RadStream s, const R4dGgMrCall& a, const R4dGgMrCall& b);
extern "C" void r4d_gg_mr_pin(int h, int r, int mt, int wn);
extern "C" const char* r4d_gg_mr_last();

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };   /* gg_gemm's GG_* codes */

uint64_t g_rs = 0x5EED00C0FFEEull;
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
void fill_row(uint16_t* x, int K) {
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 1000);
        if (c < 5) x[k] = 0x0000;
        else if (c < 10) x[k] = 0x8000;
        else if (c < 15) x[k] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
        else x[k] = f2bf((float)((int)(rnd() % 20001) - 10000) / 2500.0f);
    }
}
struct Ty { int ty; const char* name; int qk, bb; };
const Ty kTypes[] = { { Q5_K, "q5_K", 256, 176 }, { Q6_K, "q6_K", 256, 210 }, { Q8_0, "q8_0", 32, 34 } };
std::vector<uint8_t> make_w(const Ty& t, int N, int K) {
    const long long rb = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> w((size_t)(N * rb) + 64);
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
void check(bool ok, const std::string& what) {
    ++g_cases;
    if (!ok) ++g_fail;
    if (!ok) std::printf("FAIL %s\n", what.c_str());
}
std::string fmtp(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmtp(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

struct Pin { int h, r, mt, wn; };

/* One case: calls A (NA rows) and B (NB rows) of type t at K, M rows. shared_x: B reads A's input; else the inputs are
 * the two column halves of one [M, 2K] buffer. split_y: the outputs are the column halves of one [M, 2 N] buffer (NA ==
 * NB), else two buffers with a padded pitch. */
void one(const Ty& t, int K, int NA, int NB, int M, bool shared_x, bool split_y, const std::vector<Pin>& pins,
         bool mutate) {
    const std::vector<uint8_t> wa = make_w(t, NA, K), wb = make_w(t, NB, K);
    const long long w_ld = (long long)K / t.qk * t.bb;
    const int xw = shared_x ? K : 2 * K;
    std::vector<uint16_t> x((size_t)M * xw);
    for (int m = 0; m < M; ++m) fill_row(x.data() + (size_t)m * xw, xw);
    const uint16_t* xa = x.data();
    const uint16_t* xb = shared_x ? x.data() : x.data() + K;
    const int x_al = (((uintptr_t)xa & 15u) == 0 && ((xw * 2) & 15) == 0) ? 1 : 0;
    const int xb_al = (((uintptr_t)xb & 15u) == 0 && ((xw * 2) & 15) == 0) ? 1 : 0;
    for (int f32 = 0; f32 < 2; ++f32) {
        const size_t ob = f32 ? 4 : 2;
        /* output buffers: split_y = one [M, 2 NA + 3] buffer, else two [M, N + 5] buffers (padded pitch) */
        const long long ya_ld = split_y ? 2LL * NA + 3 : NA + 5, yb_ld = split_y ? ya_ld : NB + 5;
        const size_t ya_b = (size_t)M * ya_ld * ob, yb_b = split_y ? 0 : (size_t)M * yb_ld * ob;
        /* how: 0 stock, 1 two launches of the form, 2 the pair, 3 the pair with B on A's weight (the control) */
        auto run_ref = [&](std::vector<uint8_t>& Y, std::vector<uint8_t>& Y2, int how, const Pin* p) {
            Y.assign(ya_b, 0xA5);
            Y2.assign(split_y ? 0 : yb_b, 0xA5);
            uint8_t* ya = Y.data();
            uint8_t* yb = split_y ? Y.data() + (size_t)NA * ob : Y2.data();
            R4dGgMrCall a{ t.ty, f32 != 0, xa, (long long)xw, x_al, wa.data(), w_ld, ya, ya_ld, M, NA, K };
            R4dGgMrCall b{ t.ty, f32 != 0, xb, (long long)xw, xb_al, wb.data(), w_ld, yb, yb_ld, M, NB, K };
            if (how == 3) b.w = wa.data();   /* the control: B's rows from A's weight */
            if (p) r4d_gg_mr_pin(p->h, p->r, p->mt, p->wn);
            int rc = 0;
            if (how == 0) {        /* stock, each call alone */
                rc = r4d_gg_mr_emu_stock(a, 1) | r4d_gg_mr_emu_stock(b, 1);
            } else if (how == 1) { /* two launches of the form */
                rc = r4d_gg_mr_gemm(nullptr, a);
                rc |= r4d_gg_mr_gemm(nullptr, b);
            } else {               /* the pair (3: the control's) */
                rc = f32 ? r4d_gg_mr_pair_f32(nullptr, a, b) : r4d_gg_mr_pair(nullptr, a, b);
            }
            if (p) r4d_gg_mr_pin(0, 0, 0, 0);
            return rc;
        };
        std::vector<uint8_t> R, R2;
        if (run_ref(R, R2, 0, nullptr) != 0) { check(false, fmtp("%s stock launch K=%d M=%d", t.name, K, M)); continue; }
        std::vector<Pin> ps = pins;
        for (const Pin& p : ps) {
            const Pin* pp = p.h == 0 ? nullptr : &p;
            std::vector<uint8_t> T, T2, P, P2;
            const int r1 = run_ref(T, T2, 1, pp);
            const int r2 = run_ref(P, P2, 2, pp);
            const std::string lbl = fmtp("%s%s %s K=%d NA=%d NB=%d M=%d pin %d/%d/%d/%d (%s)", t.name, f32 ? " f32" : "",
                                         shared_x ? "one input" : "two inputs", K, NA, NB, M, p.h, p.r, p.mt, p.wn,
                                         r4d_gg_mr_last());
            check(r1 == 0 && r2 == 0, lbl + fmtp(": rc two %d pair %d", r1, r2));
            if (r1 || r2) continue;
            check(T == R && T2 == R2, lbl + ": two launches of the form != stock");
            const bool same = P == R && P2 == R2;
            check(same, lbl + ": the pair != stock (and the two launches)");
            if (!same) {
                size_t bad = 0;
                for (size_t e = 0; e < P.size(); e += ob) bad += std::memcmp(&P[e], &R[e], ob) != 0;
                for (size_t e = 0; e < P2.size(); e += ob) bad += std::memcmp(&P2[e], &R2[e], ob) != 0;
                std::printf("     %zu outputs differ\n", bad);
            }
        }
        if (mutate) {   /* the control: B from A's weight must differ (where NB rows of A exist) */
            std::vector<uint8_t> P, P2;
            const int rc = run_ref(P, P2, 3, nullptr);
            check(rc == 0 && !(P == R && P2 == R2),
                  fmtp("control: %s%s M=%d, B's rows from A's weight differ from B's reference", t.name, f32 ? " f32" : "", M));
            std::printf("control %s%s M=%d: B's rows from A's weight %s\n", t.name, f32 ? " f32" : "", M,
                        rc == 0 && !(P == R && P2 == R2) ? "caught" : "NOT caught");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const std::vector<int> Ms = quick ? std::vector<int>{ 1, 3, 8, 48 } : std::vector<int>{ 1, 2, 3, 6, 8, 16, 48 };
    const std::vector<Pin> pins = { { 0, 0, 0, 0 }, { -1, 0, 0, 0 }, { 1, 1, 0, 0 }, { 2, 1, 6, 2 }, { 2, 2, 0, 2 },
                                    { 4, 1, 8, 1 }, { 4, 2, 3, 0 }, { 4, 1, 0, 8 } };
    for (const Ty& t : kTypes) {
        for (int M : Ms) {
            /* q_a + kv: one input, K 5120 (reduced N) */
            one(t, 5120, 45, 21, M, true, false, pins, M == 3);
            /* the output groups: two inputs (column halves of [M, 2K]), the outputs the column halves of one buffer */
            one(t, 4096, 45, 45, M, false, true, pins, false);
        }
        std::printf("%s: %d cases, %d failures so far\n", t.name, g_cases, g_fail);
        std::fflush(stdout);
    }
    /* the real TP4 shapes at the table's / rule's form (no pin) */
    const std::vector<Pin> none = { { 0, 0, 0, 0 } };
    for (const Ty& t : kTypes) {
        if (quick && t.ty != Q6_K) continue;
        for (int M : { 1, 6 }) {
            one(t, 5120, 1280, 512, M, true, false, none, false);
            one(t, 4096, 1024, 1024, M, false, true, none, false);
        }
        std::printf("%s real shapes: %d cases, %d failures so far\n", t.name, g_cases, g_fail);
        std::fflush(stdout);
    }
    /* declines: types or M differing */
    {
        std::vector<uint16_t> x(5120 * 2);
        std::vector<uint8_t> w(64 * 5120), y(64 * 64 * 2);
        R4dGgMrCall a{ Q6_K, false, x.data(), 5120, 1, w.data(), 5120 / 256 * 210, y.data(), 64, 1, 8, 5120 };
        R4dGgMrCall b = a;
        b.ty = Q5_K;
        check(r4d_gg_mr_pair(nullptr, a, b) == R4D_GG_MR_DECLINE, "a pair of two types is declined");
        b = a;
        b.M = 2;
        check(r4d_gg_mr_pair(nullptr, a, b) == R4D_GG_MR_DECLINE, "a pair of two row counts is declined");
        b = a;
        b.K = 4096;
        check(r4d_gg_mr_pair(nullptr, a, b) == R4D_GG_MR_DECLINE, "a pair of two K is declined");
    }
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
