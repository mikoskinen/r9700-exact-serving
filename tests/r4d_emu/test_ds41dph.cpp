/* tests/r4d_emu: lever DPH's op (DRAFTER track round 68; libr4d mhc_post_collapse_rmsnorm_l256) on the host SIMT emulator
 * (tp8 loop QUEUE W14, DECODE round 54; the GPU twin: tools/dbench/dph on one card):
 *
 *   CHAIN  out and xc of mhc_post_collapse_rmsnorm_l256 against mhc_post -> mhc_collapse (both emulated), byte for byte.
 *   BI     y of the l256 op at M rows against the same rows of mhc_post_collapse_rmsnorm at MR = 160 rows (n 5120: 256
 *          threads a row, the fold whose order IS rmsnorm_bf16's 256-lane partition -- thread v adds columns v, v + 256, ..
 *          in order, then the same block_sum), and against the l256 op's own rows at MR: a row's bytes must not depend on
 *          how many rows the launch carries, at every r4d_act_threads width (M 1 / 3 / 16 / 32: 1024 threads; 33 / 48 /
 *          96 / 120: 512; 121 / 160: 256).
 *   CTRL   the stock op (mhc_post_collapse_rmsnorm, its norm folded over r4d_act_threads(M, n) threads: LEVERS BI item 4) at
 *          the same M against its own MR rows: it must differ somewhere over the cells (else the BI check cannot see a
 *          partition change), counted per cell, never a failure by itself.
 * Inputs: three kinds (typical rows; magnitudes 2^-60 .. 2^60 a row; specials: zeros, subnormals, +-inf and NaN payloads in a
 * few rows), f32 and bf16 norm weights, eps 1e-6 and 1e-20 (the DSpark tail's).
 *
 *   r4d_emu_ds41dph [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_mhc_post(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_collapse(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm_l256(const RadArgs*, RadStream);

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_cases;
    if (!ok) ++g_fail;
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
    std::fflush(stdout);
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
double uni() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
double gauss() {
    double u1 = uni(), u2 = uni();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16);   /* a NaN keeps its high payload bits */
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = const_cast<void*>(p);
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
RadArgs args(std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    return a;
}
/* values of rows [0, M) (cols a row) that differ bitwise */
long long ndiff_rows(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int64_t M, int64_t cols) {
    long long d = 0;
    for (int64_t i = 0; i < M * cols; ++i) d += a[(size_t)i] != b[(size_t)i];
    return d;
}
/* the same, but two NaNs of any payload / sign count as equal; *other = the differences that are not NaN vs NaN (an
 * infinity, a zero's sign, a value) */
bool bnan(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x007fu) != 0; }
long long ndiff_rows_nan(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int64_t M, int64_t cols) {
    long long d = 0;
    for (int64_t i = 0; i < M * cols; ++i) {
        const uint16_t x = a[(size_t)i], y = b[(size_t)i];
        d += x != y && !(bnan(x) && bnan(y));
    }
    return d;
}

constexpr int64_t kN = 5120, kHc = 4, kW = kHc * kN, kMR = 160;

struct In {
    std::vector<uint16_t> xin, h;
    std::vector<float> post, comb, pre, wf;
    std::vector<uint16_t> wb;
};
In make_inputs(int kind) {
    In in;
    in.xin.resize((size_t)(kMR * kN));
    in.h.resize((size_t)(kMR * kW));
    in.post.resize((size_t)(kMR * kHc));
    in.comb.resize((size_t)(kMR * kHc * kHc));
    in.pre.resize((size_t)(kMR * kHc));
    for (int64_t r = 0; r < kMR; ++r) {
        const double rs = kind == 1 ? std::ldexp(1.0, (int)(rnd() % 121) - 60) : std::ldexp(1.0, (int)(rnd() % 3) - 1);
        for (int64_t i = 0; i < kW; ++i) in.h[(size_t)(r * kW + i)] = bf((float)(gauss() * rs));
        for (int64_t i = 0; i < kN; ++i) in.xin[(size_t)(r * kN + i)] = bf((float)(gauss() * 0.3 * rs));
        for (int64_t k = 0; k < kHc; ++k) {
            in.post[(size_t)(r * kHc + k)] = (float)(0.5 + 0.2 * gauss());
            in.pre[(size_t)(r * kHc + k)] = (float)(0.1 + 0.9 * uni());
            for (int64_t j = 0; j < kHc; ++j) in.comb[(size_t)(r * kHc * kHc + j * kHc + k)] = (float)(0.3 * gauss());
        }
    }
    if (kind == 2) {   /* specials in rows 3 / 17 / 40 / 121 / 150: zeros, subnormals, +-inf, NaN payloads */
        const int64_t rows[] = { 3, 17, 40, 121, 150 };
        const uint16_t sp[] = { 0x0000, 0x8000, 0x0001, 0x8003, 0x7f80, 0xff80, 0x7fc1, 0xffa5 };
        for (int64_t r : rows) {
            const int64_t cnt = r == 3 ? kN : 6;   /* row 3: every column a zero or a subnormal (sum of squares 0 / tiny) */
            for (int64_t c = 0; c < cnt; ++c) {
                const int64_t i = r == 3 ? c : (int64_t)(rnd() % kN);
                in.xin[(size_t)(r * kN + i)] = r == 3 ? sp[c % 4] : sp[rnd() % 8];
                for (int64_t j = 0; j < kHc; ++j) in.h[(size_t)(r * kW + j * kN + i)] = r == 3 ? sp[(c + j) % 4] : sp[rnd() % 8];
            }
        }
    }
    in.wf.resize((size_t)kN);
    in.wb.resize((size_t)kN);
    for (int64_t i = 0; i < kN; ++i) {
        in.wf[(size_t)i] = (float)(1.0 + 0.2 * gauss());
        in.wb[(size_t)i] = bf(in.wf[(size_t)i]);
    }
    return in;
}

struct Out {
    std::vector<uint16_t> o, xc, y;
};
/* the fused op (l256 or stock) over the first M rows */
int fused(const In& in, int64_t M, bool l256, bool wbf, double eps, Out& r) {
    r.o.assign((size_t)(kMR * kW), 0xA5A5);
    r.xc.assign((size_t)(kMR * kN), 0xA5A5);
    r.y.assign((size_t)(kMR * kN), 0xA5A5);
    std::vector<RadTensor> t = { tens(in.xin.data(), RAD_BF16, { M, kN }), tens(in.h.data(), RAD_BF16, { M, kW }),
                                 tens(in.post.data(), RAD_F32, { M, kHc }), tens(in.comb.data(), RAD_F32, { M, kHc * kHc }),
                                 tens(r.o.data(), RAD_BF16, { M, kW }), tens(in.pre.data(), RAD_F32, { M, kHc }),
                                 tens(r.xc.data(), RAD_BF16, { M, kN }),
                                 wbf ? tens(in.wb.data(), RAD_BF16, { kN }) : tens(in.wf.data(), RAD_F32, { kN }),
                                 tens(r.y.data(), RAD_BF16, { M, kN }) };
    std::vector<RadParam> p = { pint("M", M), pint("n", kN), pint("hc", kHc), pf64("eps", eps), pstr("dtype", "bf16") };
    RadArgs a = args(t, p);
    return l256 ? r4d_rad_mhc_post_collapse_rmsnorm_l256(&a, nullptr) : r4d_rad_mhc_post_collapse_rmsnorm(&a, nullptr);
}
/* mhc_post then mhc_collapse over the first M rows (the chain's first two ops) */
int chain(const In& in, int64_t M, Out& r) {
    r.o.assign((size_t)(kMR * kW), 0xA5A5);
    r.xc.assign((size_t)(kMR * kN), 0xA5A5);
    std::vector<RadTensor> tp = { tens(in.xin.data(), RAD_BF16, { M, kN }), tens(in.h.data(), RAD_BF16, { M, kW }),
                                  tens(in.post.data(), RAD_F32, { M, kHc }), tens(in.comb.data(), RAD_F32, { M, kHc * kHc }),
                                  tens(r.o.data(), RAD_BF16, { M, kW }) };
    std::vector<RadParam> pp = { pint("M", M), pint("n", kN), pint("hc", kHc), pstr("dtype", "bf16") };
    RadArgs ap = args(tp, pp);
    const int rc = r4d_rad_mhc_post(&ap, nullptr);
    if (rc != RAD_OK) return rc;
    std::vector<RadTensor> tc = { tens(r.o.data(), RAD_BF16, { M, kW }), tens(in.pre.data(), RAD_F32, { M, kHc }),
                                  tens(r.xc.data(), RAD_BF16, { M, kN }) };
    RadArgs ac = args(tc, pp);
    return r4d_rad_mhc_collapse(&ac, nullptr);
}

long long g_ctrl = 0;

void cell_set(int kind, bool wbf, double eps, const std::vector<int64_t>& Ms) {
    const In in = make_inputs(kind);
    const char* kn = kind == 0 ? "typical" : kind == 1 ? "2^-60..2^60 a row" : "specials";
    Out refS, refL;   /* the stock op and the l256 op at MR rows (256 threads a row at n 5120) */
    const int rs = fused(in, kMR, false, wbf, eps, refS), rl = fused(in, kMR, true, wbf, eps, refL);
    check(rs == RAD_OK && rl == RAD_OK && ndiff_rows(refS.o, refL.o, kMR, kW) == 0 && ndiff_rows(refS.xc, refL.xc, kMR, kN) == 0 &&
              ndiff_rows(refS.y, refL.y, kMR, kN) == 0,
          fmt("BI  %s, w %s, eps %g: l256 = the stock op at M %lld (256 threads: rmsnorm_bf16's partition), out / xc / y", kn,
              wbf ? "bf16" : "f32", eps, (long long)kMR),
          fmt("rc %d %d; differing out %lld xc %lld y %lld", rs, rl, ndiff_rows(refS.o, refL.o, kMR, kW),
              ndiff_rows(refS.xc, refL.xc, kMR, kN), ndiff_rows(refS.y, refL.y, kMR, kN)));
    for (int64_t M : Ms) {
        Out C, L, S;
        const int rc = chain(in, M, C), r1 = fused(in, M, true, wbf, eps, L), r2 = fused(in, M, false, wbf, eps, S);
        /* against the separate ops a NaN's payload may differ on the emulator (the host's NaN propagation follows the
         * operand order, which the fused body and mhc_post_vec_kernel do not share); tools/dbench/dph holds them bitwise on
         * the card. Everything else bitwise. */
        const long long dO = ndiff_rows_nan(C.o, L.o, M, kW), dX = ndiff_rows_nan(C.xc, L.xc, M, kN);
        const long long pO = ndiff_rows(C.o, L.o, M, kW), pX = ndiff_rows(C.xc, L.xc, M, kN);
        check(rc == RAD_OK && r1 == RAD_OK && dO == 0 && dX == 0 && (kind == 2 || pO + pX == 0),
              fmt("CHAIN %s, w %s, eps %g, M %lld: l256's out / xc = mhc_post -> mhc_collapse%s", kn, wbf ? "bf16" : "f32", eps,
                  (long long)M, kind == 2 ? " (a NaN = a NaN)" : ""),
              fmt("rc %d %d; differing out %lld xc %lld (bitwise %lld / %lld)", rc, r1, dO, dX, pO, pX));
        const long long dY = ndiff_rows(L.y, refS.y, M, kN), dYl = ndiff_rows(L.y, refL.y, M, kN);
        const bool tail = L.y[(size_t)(M * kN)] == 0xA5A5 && L.xc[(size_t)(M * kN)] == 0xA5A5;   /* nothing past row M */
        check(r1 == RAD_OK && dY == 0 && dYl == 0 && tail,
              fmt("BI  %s, w %s, eps %g, M %lld: l256's y rows = the same rows at M %lld", kn, wbf ? "bf16" : "f32", eps,
                  (long long)M, (long long)kMR),
              fmt("differing vs stock@%lld %lld, vs l256@%lld %lld%s", (long long)kMR, dY, (long long)kMR, dYl,
                  tail ? "" : "; wrote past row M"));
        const long long dS = r2 == RAD_OK ? ndiff_rows(S.y, refS.y, M, kN) : -1;
        const long long dSo = r2 == RAD_OK ? ndiff_rows_nan(S.o, C.o, M, kW) + ndiff_rows_nan(S.xc, C.xc, M, kN) : -1;
        if (dS > 0) g_ctrl += dS;
        std::printf("ctrl %s, w %s, eps %g, M %lld: the stock op's y rows vs its own at M %lld: %lld value(s) differ (out / xc "
                    "vs the chain: %lld)\n",
                    kn, wbf ? "bf16" : "f32", eps, (long long)M, (long long)kMR, dS, dSo);
        check(dSo == 0, fmt("CHAIN %s, w %s, eps %g, M %lld: the stock op's out / xc = mhc_post -> mhc_collapse", kn,
                            wbf ? "bf16" : "f32", eps, (long long)M));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const std::vector<int64_t> all = { 1, 3, 16, 32, 33, 48, 96, 120, 121 }, few = { 1, 16, 32, 33, 96, 121 };
    cell_set(0, false, 1e-6, quick ? few : all);
    cell_set(1, false, 1e-20, quick ? few : all);
    cell_set(2, false, 1e-6, quick ? few : all);
    if (!quick) {
        cell_set(0, true, 1e-6, few);
        cell_set(1, true, 1e-6, few);
    }
    check(g_ctrl > 0, "CTRL the stock op's rows at M <= 120 differ from its own at M 160 somewhere (the BI check sees a partition)",
          fmt("%lld value(s)", g_ctrl));
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
