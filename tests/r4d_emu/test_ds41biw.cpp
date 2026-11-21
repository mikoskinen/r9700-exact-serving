/* tests/r4d_emu: lever BIW's ops (DRAFTER track round 75; libr4d mhc_post_collapse_rmsnorm_w1k, mhc_collapse_rmsnorm_w1k,
 * rmsnorm_rope_roundtrip_kv_store_w1k) on the host SIMT emulator (tp8 loop QUEUE W20, DECODE round 58; the GPU twin:
 * tools/dbench/biw on one card):
 *
 *   PIECES  deepseek41's BI issue cuts a decode segment of more than kBiNorm (32) rows into launches of <= 32 rows of the
 *           served op (r4d_act_threads: 1024 threads a row there); lever BIW issues ONE launch of the _w1k twin over up to
 *           64 rows instead. Every output of the twin over M rows (out / xc / y; x / y; y + the window cache) must equal the
 *           pieces' outputs byte for byte, at M 1 .. 96 (the twin's own domain is any M; deepseek41 issues it at 33 .. 64),
 *           and nothing past row M may be written.
 *   CTRL    the served op in ONE launch over the same M rows (> 32: 512 / 256 threads a row, another fold) against the pieces:
 *           for the trunk's norms (n 5120) it must differ somewhere over the cells (else the PIECES check could not see a width
 *           change); for the window key (n 512) only the 256-thread fold (M > 72) is another order (1024 / 512 threads: one order, the
 *           columns past 512 add zeros), so it is required over the full set and printed in quick.
 * Inputs: three kinds (typical rows; magnitudes 2^-60 .. 2^60 a row; specials: zeros, subnormals, +-inf and NaN payloads in
 * a few rows), f32 and bf16 norm weights; the window key at its serving geometry (n 512, gptj rot 64 at off 448, fp8 e4m3 /
 * ue8m0 in blocks of 32, block_size 16) with a bf16 and an fp8 (lever KVWIN) cache, slots unmapped / past the cache,
 * positions outside the angle table.
 *
 *   r4d_emu_ds41biw [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm_w1k(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_collapse_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_collapse_rmsnorm_w1k(const RadArgs*, RadStream);
extern "C" int r4d_rad_rmsnorm_rope_roundtrip_kv_store(const RadArgs*, RadStream);
extern "C" int r4d_rad_rmsnorm_rope_roundtrip_kv_store_w1k(const RadArgs*, RadStream);

namespace {

int g_cases = 0, g_fail = 0;
long long g_ctrl_pcn = 0, g_ctrl_cn = 0, g_ctrl_kv = 0;
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
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(u >> 16);
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
long long ndiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t from, size_t to) {
    long long d = 0;
    for (size_t i = from; i < to; ++i) d += a[i] != b[i];
    return d;
}
/* the launches deepseek41's BI issue makes today over rows [0, M): pieces of <= 32 rows (rows(kBiNorm, ..)) */
template <typename F>
int pieces(int64_t M, F one) {
    for (int64_t a = 0; a < M; a += 32) {
        const int rc = one(a, std::min<int64_t>(32, M - a));
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

/* ---- mhc_post_collapse_rmsnorm / mhc_collapse_rmsnorm over the trunk's streams (n 5120, hc 4) */
constexpr int64_t kN = 5120, kHc = 4, kW = kHc * kN, kMR = 100;

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
    if (kind == 2) {   /* specials in rows 3 / 17 / 40 / 63 / 70: zeros, subnormals, +-inf, NaN payloads */
        const int64_t rows[] = { 3, 17, 40, 63, 70 };
        const uint16_t sp[] = { 0x0000, 0x8000, 0x0001, 0x8003, 0x7f80, 0xff80, 0x7fc1, 0xffa5 };
        for (int64_t r : rows) {
            const int64_t cnt = r == 3 ? kN : 6;
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
    void reset() {
        o.assign((size_t)(kMR * kW), 0xA5A5);
        xc.assign((size_t)(kMR * kN), 0xA5A5);
        y.assign((size_t)(kMR * kN), 0xA5A5);
    }
};
using OpFn = int (*)(const RadArgs*, RadStream);
/* mhc_post_collapse_rmsnorm (or its twin) over rows [r0, r0 + M) of the inputs into the same rows of r */
int pcn(OpFn fn, const In& in, int64_t r0, int64_t M, bool wbf, Out& r) {
    std::vector<RadTensor> t = { tens(in.xin.data() + r0 * kN, RAD_BF16, { M, kN }), tens(in.h.data() + r0 * kW, RAD_BF16, { M, kW }),
                                 tens(in.post.data() + r0 * kHc, RAD_F32, { M, kHc }),
                                 tens(in.comb.data() + r0 * kHc * kHc, RAD_F32, { M, kHc * kHc }),
                                 tens(r.o.data() + r0 * kW, RAD_BF16, { M, kW }), tens(in.pre.data() + r0 * kHc, RAD_F32, { M, kHc }),
                                 tens(r.xc.data() + r0 * kN, RAD_BF16, { M, kN }),
                                 wbf ? tens(in.wb.data(), RAD_BF16, { kN }) : tens(in.wf.data(), RAD_F32, { kN }),
                                 tens(r.y.data() + r0 * kN, RAD_BF16, { M, kN }) };
    std::vector<RadParam> p = { pint("M", M), pint("n", kN), pint("hc", kHc), pf64("eps", 1e-6), pstr("dtype", "bf16") };
    RadArgs a = args(t, p);
    return fn(&a, nullptr);
}
/* mhc_collapse_rmsnorm (or its twin): h, pre -> x; w -> y, rows [r0, r0 + M) */
int cn(OpFn fn, const In& in, int64_t r0, int64_t M, bool wbf, Out& r) {
    std::vector<RadTensor> t = { tens(in.h.data() + r0 * kW, RAD_BF16, { M, kW }), tens(in.pre.data() + r0 * kHc, RAD_F32, { M, kHc }),
                                 tens(r.xc.data() + r0 * kN, RAD_BF16, { M, kN }),
                                 wbf ? tens(in.wb.data(), RAD_BF16, { kN }) : tens(in.wf.data(), RAD_F32, { kN }),
                                 tens(r.y.data() + r0 * kN, RAD_BF16, { M, kN }) };
    std::vector<RadParam> p = { pint("M", M), pint("n", kN), pint("hc", kHc), pf64("eps", 1e-6), pstr("dtype", "bf16") };
    RadArgs a = args(t, p);
    return fn(&a, nullptr);
}

void trunk_set(int kind, bool wbf, const std::vector<int64_t>& Ms) {
    const In in = make_inputs(kind);
    const char* kn = kind == 0 ? "typical" : kind == 1 ? "2^-60..2^60 a row" : "specials";
    for (int64_t M : Ms) {
        Out P, W, S;
        P.reset(); W.reset(); S.reset();
        int r1 = pieces(M, [&](int64_t a, int64_t n) { return pcn(r4d_rad_mhc_post_collapse_rmsnorm, in, a, n, wbf, P); });
        int r2 = pcn(r4d_rad_mhc_post_collapse_rmsnorm_w1k, in, 0, M, wbf, W);
        int r3 = pcn(r4d_rad_mhc_post_collapse_rmsnorm, in, 0, M, wbf, S);
        long long dO = ndiff(P.o, W.o, 0, W.o.size()), dX = ndiff(P.xc, W.xc, 0, W.xc.size()), dY = ndiff(P.y, W.y, 0, W.y.size());
        const bool tail = W.y[(size_t)(M * kN)] == 0xA5A5 && W.xc[(size_t)(M * kN)] == 0xA5A5 && W.o[(size_t)(M * kW)] == 0xA5A5;
        check(r1 == RAD_OK && r2 == RAD_OK && dO + dX + dY == 0 && tail,
              fmt("PIECES pcn %s, w %s, M %lld: mhc_post_collapse_rmsnorm_w1k in one launch = the <= 32-row pieces (out / xc / y)", kn,
                  wbf ? "bf16" : "f32", (long long)M),
              fmt("rc %d %d; differing out %lld xc %lld y %lld%s", r1, r2, dO, dX, dY, tail ? "" : "; wrote past row M"));
        if (r3 == RAD_OK) {
            const long long c = ndiff(P.y, S.y, 0, (size_t)(M * kN));
            g_ctrl_pcn += c;
            std::printf("ctrl pcn %s, w %s, M %lld: the served op in one launch vs the pieces: y %lld value(s) differ\n", kn,
                        wbf ? "bf16" : "f32", (long long)M, c);
        }
        P.reset(); W.reset(); S.reset();
        r1 = pieces(M, [&](int64_t a, int64_t n) { return cn(r4d_rad_mhc_collapse_rmsnorm, in, a, n, wbf, P); });
        r2 = cn(r4d_rad_mhc_collapse_rmsnorm_w1k, in, 0, M, wbf, W);
        r3 = cn(r4d_rad_mhc_collapse_rmsnorm, in, 0, M, wbf, S);
        dX = ndiff(P.xc, W.xc, 0, W.xc.size());
        dY = ndiff(P.y, W.y, 0, W.y.size());
        const bool tail2 = W.y[(size_t)(M * kN)] == 0xA5A5 && W.xc[(size_t)(M * kN)] == 0xA5A5;
        check(r1 == RAD_OK && r2 == RAD_OK && dX + dY == 0 && tail2,
              fmt("PIECES cn  %s, w %s, M %lld: mhc_collapse_rmsnorm_w1k in one launch = the <= 32-row pieces (x / y)", kn,
                  wbf ? "bf16" : "f32", (long long)M),
              fmt("rc %d %d; differing x %lld y %lld%s", r1, r2, dX, dY, tail2 ? "" : "; wrote past row M"));
        if (r3 == RAD_OK) {
            const long long c = ndiff(P.y, S.y, 0, (size_t)(M * kN));
            g_ctrl_cn += c;
            std::printf("ctrl cn  %s, w %s, M %lld: the served op in one launch vs the pieces: y %lld value(s) differ\n", kn,
                        wbf ? "bf16" : "f32", (long long)M, c);
        }
    }
}

/* ---- rmsnorm_rope_roundtrip_kv_store at the window key's geometry */
constexpr int64_t kKn = 512, kRot = 64, kOff = 448, kBs = 16, kTab = 300, kKM = 100;
constexpr int64_t kNb = kKM / kBs + 3;
int w8_bytes(int n) { return (n + n / 32 + 15) / 16 * 16; }   /* libr4d ds_w8_bytes: the fp8 window slot */

void kv_set(int kind, bool wbf, bool w8, const std::vector<int64_t>& Ms) {
    const char* kn = kind == 0 ? "typical" : kind == 1 ? "2^-60..2^60 a row" : "specials";
    std::vector<uint16_t> x((size_t)(kKM * kKn));
    for (int64_t m = 0; m < kKM; ++m) {
        const double rs = kind == 1 ? std::ldexp(1.0, (int)(rnd() % 121) - 60) : (m % 7 == 3 ? 1e-3 : m % 7 == 5 ? 40.0 : 1.0);
        for (int64_t i = 0; i < kKn; ++i) x[(size_t)(m * kKn + i)] = bf((float)(gauss() * rs));
    }
    if (kind == 2) {
        const uint16_t sp[] = { 0x0000, 0x8000, 0x0001, 0x8003, 0x7f80, 0xff80, 0x7fc1, 0xffa5 };
        for (int64_t i = 0; i < kKn; ++i) x[(size_t)(3 * kKn + i)] = sp[i % 4];
        for (int64_t r : { 17, 40, 63, 70 })
            for (int c = 0; c < 6; ++c) x[(size_t)(r * kKn + (int64_t)(rnd() % kKn))] = sp[rnd() % 8];
    }
    std::vector<float> wf((size_t)kKn);
    std::vector<uint16_t> wb((size_t)kKn);
    for (int64_t i = 0; i < kKn; ++i) {
        wf[(size_t)i] = (float)(0.5 + 0.1 * gauss());
        wb[(size_t)i] = bf(wf[(size_t)i]);
    }
    std::vector<float> cs((size_t)(kTab * kRot));
    for (int64_t p = 0; p < kTab; ++p)
        for (int64_t i = 0; i < kRot / 2; ++i) {
            const double a = (double)p * std::pow(10000.0, -2.0 * (double)i / (double)kRot);
            cs[(size_t)(p * kRot + i)] = (float)std::cos(a);
            cs[(size_t)(p * kRot + kRot / 2 + i)] = (float)std::sin(a);
        }
    std::vector<int32_t> pos((size_t)kKM), slot((size_t)kKM);
    std::vector<int32_t> perm((size_t)(kNb * kBs));
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int32_t)i;
    for (size_t i = perm.size() - 1; i > 0; --i) std::swap(perm[i], perm[(size_t)(rnd() % (i + 1))]);
    for (int64_t m = 0; m < kKM; ++m) {
        const int k = (int)(rnd() % 100);
        pos[(size_t)m] = k < 6 ? (int32_t)(kTab + k) : k < 9 ? -1 : (int32_t)(rnd() % kTab);
        slot[(size_t)m] = rnd() % 100 < 10 ? -1 : perm[(size_t)m];
    }
    slot[0] = perm[0];   /* row 0 mapped: an M 1 cell stores something */
    slot[(size_t)(kKM - 2)] = (int32_t)(kNb * kBs + 3);   /* past the cache */
    slot[(size_t)45] = (int32_t)(kNb * kBs + 1);
    const int64_t sl = w8 ? w8_bytes((int)kKn) / 2 : kKn;   /* bf16 slots an entry */
    const auto run = [&](OpFn fn, int64_t r0, int64_t M, std::vector<uint16_t>& y, std::vector<uint16_t>& kc) {
        std::vector<RadTensor> t = { tens(x.data() + r0 * kKn, RAD_BF16, { M, kKn }),
                                     wbf ? tens(wb.data(), RAD_BF16, { kKn }) : tens(wf.data(), RAD_F32, { kKn }),
                                     tens(y.data() + r0 * kKn, RAD_BF16, { M, kKn }), tens(cs.data(), RAD_F32, { kTab, kRot }),
                                     tens(pos.data() + r0, RAD_I32, { M }), tens(slot.data() + r0, RAD_I32, { M }),
                                     tens(kc.data(), RAD_BF16, { kNb, 1, kBs, sl }) };
        std::vector<RadParam> p = { pint("M", M), pint("n", kKn), pf64("eps", 1e-6), pint("n_head", 1), pint("head_dim", kKn),
                                    pint("rot", kRot), pint("off", kOff), pstr("mode", "gptj"), pint("inverse", 0),
                                    pint("block", 32), pstr("format", "fp8_e4m3"), pstr("scale", "ue8m0"), pstr("dtype", "bf16"),
                                    pint("block_size", kBs), pstr("kv_dtype", w8 ? "fp8_e4m3" : "bf16") };
        RadArgs a = args(t, p);
        return fn(&a, nullptr);
    };
    for (int64_t M : Ms) {
        std::vector<uint16_t> yP((size_t)(kKM * kKn), 0x1234), yW = yP, yS = yP;
        std::vector<uint16_t> kP((size_t)(kNb * kBs * sl), 0x5a5a), kW_ = kP, kS = kP;
        const int r1 = pieces(M, [&](int64_t a, int64_t n) { return run(r4d_rad_rmsnorm_rope_roundtrip_kv_store, a, n, yP, kP); });
        const int r2 = run(r4d_rad_rmsnorm_rope_roundtrip_kv_store_w1k, 0, M, yW, kW_);
        const int r3 = run(r4d_rad_rmsnorm_rope_roundtrip_kv_store, 0, M, yS, kS);
        const long long dY = ndiff(yP, yW, 0, yP.size()), dK = ndiff(kP, kW_, 0, kP.size());
        long long stored = 0;
        for (uint16_t v : kP) stored += v != 0x5a5a;
        check(r1 == RAD_OK && r2 == RAD_OK && dY + dK == 0 && stored > 0,
              fmt("PIECES kv  %s, w %s, cache %s, M %lld: rmsnorm_rope_roundtrip_kv_store_w1k in one launch = the <= 32-row "
                  "pieces (y + the window cache)", kn, wbf ? "bf16" : "f32", w8 ? "fp8" : "bf16", (long long)M),
              fmt("rc %d %d; differing y %lld, cache %lld (%lld stored)", r1, r2, dY, dK, stored));
        if (r3 == RAD_OK) {
            const long long c = ndiff(yP, yS, 0, (size_t)(M * kKn)) + ndiff(kP, kS, 0, kP.size());
            g_ctrl_kv += c;
            std::printf("ctrl kv  %s, w %s, cache %s, M %lld: the served op in one launch vs the pieces: %lld value(s) differ\n",
                        kn, wbf ? "bf16" : "f32", w8 ? "fp8" : "bf16", (long long)M, c);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const std::vector<int64_t> all = { 1, 3, 32, 33, 40, 48, 63, 64, 65, 96 }, few = { 1, 33, 48, 64, 96 };
    trunk_set(0, false, quick ? few : all);
    trunk_set(1, false, quick ? few : all);
    trunk_set(2, false, quick ? few : all);
    if (!quick) trunk_set(0, true, few);
    kv_set(0, false, false, quick ? few : all);
    kv_set(1, false, true, quick ? few : all);
    kv_set(2, true, true, few);
    if (!quick) kv_set(0, true, false, few);
    check(g_ctrl_pcn > 0, "CTRL pcn: the served op in one launch above 32 rows differs from the pieces somewhere",
          fmt("%lld value(s)", g_ctrl_pcn));
    check(g_ctrl_cn > 0, "CTRL cn: the served op in one launch above 32 rows differs from the pieces somewhere",
          fmt("%lld value(s)", g_ctrl_cn));
    /* the window key (n 512): the served op's fold at 1024 / 512 threads is the same order (columns 512 .. 1023 add zeros at the
     * end of block_sum), only 256 threads (M > 72) pair columns t and t + 256 first, and y then goes through the fp8 e4m3 round
     * trip -- so its one launch rarely shows a width (tools/dbench/biw on the card: 0 cells as well); printed, not required */
    if (quick)
        std::printf("info CTRL kv: the served op in one launch above 32 rows vs the pieces: %lld value(s) differ (n 512; quick: not "
                    "required)\n", g_ctrl_kv);
    else
        check(g_ctrl_kv > 0, "CTRL kv: the served op in one launch above 32 rows differs from the pieces somewhere (n 512: the "
                             "256-thread fold at M > 72)", fmt("%lld value(s)", g_ctrl_kv));
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
