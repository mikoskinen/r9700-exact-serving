/* tests/r4d_emu: lever A8D's folds (b) + (c) (DRAFTER track round 95) on the host SIMT emulator, each against the launches it
 * replaces in the DSpark stage, byte for byte (written by DECODE round 66 = QUEUE W33, the review of patch-r95-a8b.mbox):
 *
 *   (b) gemm_nt_q_pair_i8g (gemm_i8a8_pairq_kernel: both o_a groups in ONE launch, each split K as alone, the split finisher
 *       writing quant_act_i8g's codes of the bf16 sums) against the served four launches -- gemm_i8a8_nt_m16 (r4d_rad_gemm_i8a8_m16,
 *       the narrow int8 tile every <= 64-row A8D launch runs) per group over its K window of the activation pair into bf16 [M, N],
 *       then quant_act_i8g per group into its N columns of o_b's int8 pair: the codes and the f32 scales. N 256 / 1024, K 1024 /
 *       4096 (the second = an o_a group's [o_rank, gd] shape class at TP4), M 1 .. 64 (every narrow_ksplit row class), activation
 *       groups zero / tiny / large, wider q / scale pitches; refused: M 65, N not a multiple of 128, K < 512, group 64, unequal
 *       weight pitches.
 *   (c) mhc_post_collapse_rmsnorm_i8g (post_collapse_norm_q8_kernel) against mhc_post_collapse_rmsnorm (the served pcn_a, default
 *       form: r4d_act_threads' workgroup) then quant_act_i8g of its y: out, xc, y, the codes and the scales. n 5120 / 4096, hc 4,
 *       M 1 .. 160 (every r4d_act_threads width: 1024 / 512 / 256 threads, the one-pass and the two-pass y walks), f32 and bf16
 *       norm weights, eps 1e-6 / 1e-20, rows of every magnitude 2^-60 .. 2^60 and an all-zero row; refused: n % 128, n > 8192.
 *   r4d_emu_ds41a8b [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs*);
extern "C" int r4d_rad_gemm_nt_q_pair_i8g(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_gemm_nt_q_pair_i8g_scratch(const RadArgs*);
extern "C" int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm_i8g(const RadArgs*, RadStream);

namespace {

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
    std::fflush(stdout);
}
template <class T>
struct Al {   /* 64-byte aligned host buffer (the kernels' 16-byte operand rules) */
    T* p = nullptr;
    size_t n = 0;
    explicit Al(size_t k, T fill = T()) : n(k) {
        p = (T*)std::aligned_alloc(64, ((k * sizeof(T) + 63) / 64) * 64 + 64);
        for (size_t i = 0; i < k; ++i) p[i] = fill;
    }
    ~Al() { std::free(p); }
    Al(const Al&) = delete;
    Al& operator=(const Al&) = delete;
};
RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape, int64_t row_stride = 0) {
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
    if (row_stride && t.rank == 2) t.stride[0] = row_stride;
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* ---------------------------------------------------------------- (b) gemm_nt_q_pair_i8g */
struct PairCase {
    int M = 1, N = 256, K = 1024;
    int64_t qpad = 0, spad = 0;     /* q / scale row pitches beyond 2N / 2N / 128 */
    int group = 128;
    bool wpitch_differs = false;    /* b1 at a wider row pitch than b0: refused */
};

void pair_case(const std::string& name, const PairCase& c, uint32_t seed, bool expect_ok = true) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(-127, 127), pct(0, 99);
    std::uniform_real_distribution<float> sc(0.002f, 0.05f), wsc(0.5f, 2.0f);
    const int M = c.M, N = c.N, K = c.K, nkb = K / 128;
    const int64_t xq = 2LL * K, xs_ld = 2LL * nkb, qld = 2LL * N + c.qpad, sld = 2LL * N / 128 + c.spad;
    Al<int8_t> x((size_t)M * xq);
    Al<float> xs((size_t)M * xs_ld);
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < 2 * nkb; ++g) {
            const int k = pct(rng);   /* a 128-wide activation group: zero / tiny scale / large scale / typical */
            const float s = k < 5 ? 0.0f : k < 10 ? 1e-7f : k < 15 ? 40.0f : sc(rng);
            xs.p[(size_t)m * xs_ld + g] = s;
            for (int j = 0; j < 128; ++j) x.p[(size_t)m * xq + (size_t)g * 128 + j] = (int8_t)(k < 5 ? 0 : code(rng));
        }
    const int64_t wld1 = c.wpitch_differs ? K + 256 : K;
    Al<int8_t> b0((size_t)N * K), b1((size_t)N * wld1);
    Al<uint16_t> s0((size_t)N * nkb), s1((size_t)N * nkb);
    for (size_t i = 0; i < b0.n; ++i) b0.p[i] = (int8_t)code(rng);
    for (size_t i = 0; i < b1.n; ++i) b1.p[i] = (int8_t)code(rng);
    for (size_t i = 0; i < s0.n; ++i) s0.p[i] = bf(wsc(rng) * 0.01f);
    for (size_t i = 0; i < s1.n; ++i) s1.p[i] = bf(wsc(rng) * 0.01f);
    /* the fused op */
    Al<int8_t> q2((size_t)M * qld, (int8_t)0x55);
    Al<float> sc2((size_t)M * sld, -7.0f);
    RadTensor t2[8] = { tens(x.p, RAD_I8, { M, 2LL * K }), tens(xs.p, RAD_F32, { M, 2LL * nkb }),
                        tens(b0.p, RAD_I8, { N, K }), tens(s0.p, RAD_BF16, { N, nkb }),
                        tens(b1.p, RAD_I8, { N, K }, wld1), tens(s1.p, RAD_BF16, { N, nkb }),
                        tens(q2.p, RAD_I8, { M, 2LL * N }, qld), tens(sc2.p, RAD_F32, { M, 2LL * N / 128 }, sld) };
    const RadParam p2[5] = { pint("M", M), pint("N", N), pint("K", K), pint("group", c.group), pstr("dtype", "i8a8") };
    RadArgs a2{};
    a2.t = t2;
    a2.n_t = 8;
    a2.p = p2;
    a2.n_p = 5;
    const int64_t sb2 = r4d_rad_gemm_nt_q_pair_i8g_scratch(&a2);
    Al<uint8_t> scr2((size_t)(sb2 > 0 ? sb2 : 16));
    a2.scratch = scr2.p;
    a2.scratch_bytes = (uint64_t)(sb2 > 0 ? sb2 : 0);
    const int rc2 = r4d_rad_gemm_nt_q_pair_i8g(&a2, nullptr);
    if (!expect_ok) {
        expect(rc2 != RAD_OK, name, "the pair op served a geometry it must refuse", rc2, 0);
        long long w = 0;
        for (size_t i = 0; i < q2.n; ++i) w += q2.p[i] != (int8_t)0x55;
        expect(w == 0, name, "a refused call wrote q", w, 0);
        std::printf("%-58s refused (rc %d)\n", name.c_str(), rc2);
        return;
    }
    expect(rc2 == RAD_OK, name, "gemm_nt_q_pair_i8g", rc2, 0);
    expect(sb2 > 0, name, "the pair op's scratch query returned 0", sb2, 0);
    /* the served chain: per group gemm_i8a8_nt_m16 over its K window, then quant_act_i8g into its N columns */
    Al<int8_t> q1((size_t)M * qld, (int8_t)0x55);
    Al<float> sc1((size_t)M * sld, -7.0f);
    Al<uint16_t> y0((size_t)M * N, 0xA5A5), y1((size_t)M * N, 0xA5A5);
    for (int g = 0; g < 2; ++g) {
        Al<uint16_t>& y = g ? y1 : y0;
        RadTensor t[5] = { tens(x.p + (size_t)g * K, RAD_I8, { M, K }, xq), tens(xs.p + (size_t)g * nkb, RAD_F32, { M, nkb }, xs_ld),
                           g ? tens(b1.p, RAD_I8, { N, K }, wld1) : tens(b0.p, RAD_I8, { N, K }),
                           tens(g ? s1.p : s0.p, RAD_BF16, { N, nkb }), tens(y.p, RAD_BF16, { M, N }) };
        const RadParam p[5] = { pint("M", M), pint("N", N), pint("K", K), pint("group", 128), pstr("dtype", "i8a8") };
        RadArgs a{};
        a.t = t;
        a.n_t = 5;
        a.p = p;
        a.n_p = 5;
        const int64_t sb = r4d_rad_gemm_i8a8_m16_scratch(&a);
        Al<uint8_t> scr((size_t)(sb > 0 ? sb : 16));
        a.scratch = scr.p;
        a.scratch_bytes = (uint64_t)(sb > 0 ? sb : 0);
        expect(r4d_rad_gemm_i8a8_m16(&a, nullptr) == RAD_OK, name, "gemm_i8a8_nt_m16", g, 0);
        RadTensor tq[3] = { tens(y.p, RAD_BF16, { M, N }), tens(q1.p + (size_t)g * N, RAD_I8, { M, N }, qld),
                            tens(sc1.p + (size_t)g * N / 128, RAD_F32, { M, N / 128 }, sld) };
        const RadParam pq[4] = { pint("M", M), pint("n", N), pint("group", 128), pstr("dtype", "bf16") };
        RadArgs aq{};
        aq.t = tq;
        aq.n_t = 3;
        aq.p = pq;
        aq.n_p = 4;
        expect(r4d_rad_quant_act_i8g(&aq, nullptr) == RAD_OK, name, "quant_act_i8g", g, 0);
    }
    long long dq = 0, ds = 0, pad = 0, nz = 0;
    for (int m = 0; m < M; ++m) {
        for (int64_t j = 0; j < qld; ++j) {
            const size_t i = (size_t)m * qld + j;
            dq += q1.p[i] != q2.p[i];
            if (j >= 2LL * N) pad += q2.p[i] != (int8_t)0x55;
            else nz += q2.p[i] != 0;
        }
        for (int64_t j = 0; j < sld; ++j) {
            const size_t i = (size_t)m * sld + j;
            ds += std::memcmp(&sc1.p[i], &sc2.p[i], 4) != 0;
            if (j >= 2LL * N / 128) pad += std::memcmp(&sc2.p[i], &sc1.p[i], 4) != 0;
        }
    }
    expect(dq == 0, name, "the int8 codes differ from gemm_i8a8_nt_m16 + quant_act_i8g's", dq, 0);
    expect(ds == 0, name, "the scales differ from gemm_i8a8_nt_m16 + quant_act_i8g's", ds, 0);
    expect(pad == 0, name, "the pair op wrote past 2N codes / 2N / 128 scales", pad, 0);
    expect(nz > 0, name, "every code is 0 (the fixture reached nothing)", nz, 0);
    std::printf("%-58s M %2d N %4d K %4d (scratch %lld B): codes %lld, scales %lld differ; %lld nonzero codes\n", name.c_str(), M, N,
                K, (long long)sb2, dq, ds, nz);
    std::fflush(stdout);
}

/* ---------------------------------------------------------------- (c) mhc_post_collapse_rmsnorm_i8g */
constexpr int64_t kHc = 4;
struct PcnIn {
    int64_t n = 5120, MR = 160;
    std::vector<uint16_t> xin, h;
    std::vector<float> post, comb, pre, wf;
    std::vector<uint16_t> wb;
};
PcnIn pcn_inputs(int64_t n, int64_t MR, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    PcnIn in;
    in.n = n;
    in.MR = MR;
    const int64_t W = kHc * n;
    in.xin.resize((size_t)(MR * n));
    in.h.resize((size_t)(MR * W));
    in.post.resize((size_t)(MR * kHc));
    in.comb.resize((size_t)(MR * kHc * kHc));
    in.pre.resize((size_t)(MR * kHc));
    for (int64_t r = 0; r < MR; ++r) {
        const double rs = r % 7 == 3 ? std::ldexp(1.0, (int)(rng() % 121) - 60) : std::ldexp(1.0, (int)(rng() % 3) - 1);
        const bool zero = r == 5;   /* an all-zero row: sum of squares 0, every y 0, the quantiser's scale 1 */
        for (int64_t i = 0; i < W; ++i) in.h[(size_t)(r * W + i)] = zero ? 0 : bf((float)(nd(rng) * rs));
        for (int64_t i = 0; i < n; ++i) in.xin[(size_t)(r * n + i)] = zero ? 0 : bf((float)(nd(rng) * 0.3 * rs));
        for (int64_t k = 0; k < kHc; ++k) {
            in.post[(size_t)(r * kHc + k)] = (float)(0.5 + 0.2 * nd(rng));
            in.pre[(size_t)(r * kHc + k)] = (float)(0.1 + 0.9 * uni(rng));
            for (int64_t j = 0; j < kHc; ++j) in.comb[(size_t)(r * kHc * kHc + j * kHc + k)] = (float)(0.3 * nd(rng));
        }
    }
    in.wf.resize((size_t)n);
    in.wb.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        in.wf[(size_t)i] = (float)(1.0 + 0.2 * nd(rng));
        in.wb[(size_t)i] = bf(in.wf[(size_t)i]);
    }
    return in;
}

void pcn_case(const std::string& name, const PcnIn& in, int64_t M, bool wbf, double eps, bool expect_ok = true, int64_t n_over = 0) {
    const int64_t n = n_over ? n_over : in.n, W = kHc * n, ng = n / 128 + (n % 128 ? 1 : 0);
    std::vector<uint16_t> o1((size_t)(M * W), 0xA5A5), xc1((size_t)(M * n), 0xA5A5), y1((size_t)(M * n), 0xA5A5);
    std::vector<uint16_t> o2 = o1, xc2 = xc1, y2 = y1;
    Al<int8_t> q1((size_t)(M * n), (int8_t)0x55), q2((size_t)(M * n), (int8_t)0x55);
    Al<float> s1((size_t)(M * ng), -7.0f), s2((size_t)(M * ng), -7.0f);
    auto ins = [&](std::vector<uint16_t>& o, std::vector<uint16_t>& xc, std::vector<uint16_t>& y) {
        return std::vector<RadTensor>{ tens(in.xin.data(), RAD_BF16, { M, n }), tens(in.h.data(), RAD_BF16, { M, W }),
                                       tens(in.post.data(), RAD_F32, { M, kHc }), tens(in.comb.data(), RAD_F32, { M, kHc * kHc }),
                                       tens(o.data(), RAD_BF16, { M, W }), tens(in.pre.data(), RAD_F32, { M, kHc }),
                                       tens(xc.data(), RAD_BF16, { M, n }),
                                       wbf ? tens(in.wb.data(), RAD_BF16, { n }) : tens(in.wf.data(), RAD_F32, { n }),
                                       tens(y.data(), RAD_BF16, { M, n }) };
    };
    std::vector<RadParam> p = { pint("M", M), pint("n", n), pint("hc", kHc), pf64("eps", eps), pstr("dtype", "bf16") };
    /* the fused op */
    std::vector<RadTensor> t2 = ins(o2, xc2, y2);
    t2.push_back(tens(q2.p, RAD_I8, { M, n }));
    t2.push_back(tens(s2.p, RAD_F32, { M, ng }));
    RadArgs a2{};
    a2.t = t2.data();
    a2.n_t = (int)t2.size();
    a2.p = p.data();
    a2.n_p = (int)p.size();
    a2.world_size = 1;
    const int rc2 = r4d_rad_mhc_post_collapse_rmsnorm_i8g(&a2, nullptr);
    if (!expect_ok) {
        expect(rc2 != RAD_OK, name, "the fused norm served a geometry it must refuse", rc2, 0);
        long long w = 0;
        for (size_t i = 0; i < y2.size(); ++i) w += y2[i] != 0xA5A5;
        expect(w == 0, name, "a refused call wrote y", w, 0);
        std::printf("%-58s refused (rc %d)\n", name.c_str(), rc2);
        return;
    }
    expect(rc2 == RAD_OK, name, "mhc_post_collapse_rmsnorm_i8g", rc2, 0);
    /* the served pair: pcn_a, then quant_act_i8g of its y */
    std::vector<RadTensor> t1 = ins(o1, xc1, y1);
    RadArgs a1{};
    a1.t = t1.data();
    a1.n_t = (int)t1.size();
    a1.p = p.data();
    a1.n_p = (int)p.size();
    a1.world_size = 1;
    expect(r4d_rad_mhc_post_collapse_rmsnorm(&a1, nullptr) == RAD_OK, name, "mhc_post_collapse_rmsnorm", 0, 0);
    RadTensor tq[3] = { tens(y1.data(), RAD_BF16, { M, n }), tens(q1.p, RAD_I8, { M, n }), tens(s1.p, RAD_F32, { M, ng }) };
    const RadParam pq[4] = { pint("M", M), pint("n", n), pint("group", 128), pstr("dtype", "bf16") };
    RadArgs aq{};
    aq.t = tq;
    aq.n_t = 3;
    aq.p = pq;
    aq.n_p = 4;
    expect(r4d_rad_quant_act_i8g(&aq, nullptr) == RAD_OK, name, "quant_act_i8g", 0, 0);
    long long d_o = 0, d_x = 0, d_y = 0, d_q = 0, d_s = 0, nzq = 0;
    for (size_t i = 0; i < o1.size(); ++i) d_o += o1[i] != o2[i];
    for (size_t i = 0; i < xc1.size(); ++i) d_x += xc1[i] != xc2[i];
    for (size_t i = 0; i < y1.size(); ++i) d_y += y1[i] != y2[i];
    for (size_t i = 0; i < q1.n; ++i) { d_q += q1.p[i] != q2.p[i]; nzq += q2.p[i] != 0; }
    for (size_t i = 0; i < s1.n; ++i) d_s += std::memcmp(&s1.p[i], &s2.p[i], 4) != 0;
    expect(d_o == 0, name, "out differs from mhc_post_collapse_rmsnorm's", d_o, 0);
    expect(d_x == 0, name, "xc differs from mhc_post_collapse_rmsnorm's", d_x, 0);
    expect(d_y == 0, name, "y differs from mhc_post_collapse_rmsnorm's", d_y, 0);
    expect(d_q == 0, name, "the int8 codes differ from quant_act_i8g's of y", d_q, 0);
    expect(d_s == 0, name, "the scales differ from quant_act_i8g's of y", d_s, 0);
    expect(nzq > 0, name, "every code is 0 (the fixture reached nothing)", nzq, 0);
    std::printf("%-58s M %3lld n %5lld w %s eps %g: out %lld, xc %lld, y %lld, codes %lld, scales %lld differ\n", name.c_str(),
                (long long)M, (long long)n, wbf ? "bf16" : "f32", eps, d_o, d_x, d_y, d_q, d_s);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    /* (b) */
    PairCase c;
    const std::vector<int> Ms = quick ? std::vector<int>{ 1, 17, 33, 64 } : std::vector<int>{ 1, 2, 7, 16, 17, 24, 32, 33, 40, 48, 49, 64 };
    for (int M : Ms) {
        c = PairCase{};
        c.M = M;
        pair_case("(b) N 256 K 1024, M " + std::to_string(M), c, 1000u + (uint32_t)M);
    }
    for (int M : quick ? std::vector<int>{ 16 } : std::vector<int>{ 1, 16, 40, 64 }) {
        c = PairCase{};
        c.M = M; c.N = 1024; c.K = 4096;
        pair_case("(b) N 1024 K 4096 (an o_a group's class), M " + std::to_string(M), c, 2000u + (uint32_t)M);
    }
    c = PairCase{};
    c.M = 24; c.N = 384; c.K = 512; c.qpad = 64; c.spad = 3;
    pair_case("(b) N 384 K 512, q / scale pitches wider", c, 3000u);
    c = PairCase{}; c.M = 65;
    pair_case("(b) M 65 (past the narrow tile)", c, 3100u, false);
    c = PairCase{}; c.M = 8; c.N = 200;
    pair_case("(b) N 200 (not whole 128-column groups)", c, 3101u, false);
    c = PairCase{}; c.M = 8; c.K = 256;
    pair_case("(b) K 256 (< 512: no split)", c, 3102u, false);
    c = PairCase{}; c.M = 8; c.group = 64;
    pair_case("(b) group 64", c, 3103u, false);
    c = PairCase{}; c.M = 8; c.wpitch_differs = true;
    pair_case("(b) the two weights at different row pitches", c, 3104u, false);
    /* (c) */
    const PcnIn in = pcn_inputs(5120, 160, 77u);
    const std::vector<int64_t> Mc = quick ? std::vector<int64_t>{ 1, 33, 121 } : std::vector<int64_t>{ 1, 3, 16, 32, 33, 48, 96, 120, 121, 160 };
    for (int64_t M : Mc) pcn_case("(c) n 5120 hc 4, rows " + std::to_string(M), in, M, false, 1e-6);
    pcn_case("(c) n 5120 hc 4, rows 40, bf16 norm weight, eps 1e-20", in, 40, true, 1e-20);
    if (!quick) {
        const PcnIn in4 = pcn_inputs(4096, 130, 78u);
        for (int64_t M : { 2, 64, 130 }) pcn_case("(c) n 4096 hc 4, rows " + std::to_string(M), in4, M, true, 1e-6);
    }
    pcn_case("(c) n 5000 (not whole 128-column groups)", in, 4, false, 1e-6, false, 5000);
    {
        const PcnIn big = pcn_inputs(8320, 2, 79u);
        pcn_case("(c) n 8320 (past the LDS row)", big, 2, false, 1e-6, false);
    }
    std::printf("%d check(s), %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
