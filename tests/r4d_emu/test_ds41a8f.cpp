/* tests/r4d_emu: lever A8D's fold (a) (DRAFTER track round 94) on the host SIMT emulator -- libr4d rope_rows_roundtrip_i8g
 * (rope_rt8q_kernel: rope_rows_roundtrip's eight-columns-a-thread form with quant_act_i8g's int8 codes of the stored row on
 * its pass) against the two launches it replaces in the DSpark stage (DOF's o_f = rope_rows_roundtrip, then quant_act_i8g at
 * group 128), byte for byte: the rotated + round-tripped x, the int8 codes and the f32 scales. The chain under both
 * rope_rows_roundtrip forms (r4d_rope8_pin 0 / 1: the lane-group form and the eight-columns form; ROPE8 is BEST's). Heads 16
 * (TP4) / 64 (one rank), head_dim 512, rot 64 at off 448 (the model's), inverse 1; rows 1 .. 64 (and 130); positions in the
 * table, past it and negative (no rope there); all-zero, tiny and large 128-column groups; x / q row pitches wider than a row.
 * Refused: n % 128 != 0, block 16, neox. tp8 loop DECODE round 65 (QUEUE W32b, the review of patch-r94-a8d-fold.mbox). */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_rope_rows_roundtrip_i8g(const RadArgs*, RadStream);
extern "C" int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);
extern "C" void r4d_rope8_pin(int v);

namespace {

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
}
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
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }

struct Case {
    int nh = 16, hd = 512, rot = 64, off = 448, M = 1, block = 32;
    int64_t xpad = 0, qpad = 0;   /* row pitch beyond n (x in bf16 values, q in bytes) */
    const char* mode = "gptj";
};

std::vector<RadParam> rope_params(const Case& c) {
    return { pint("M", c.M), pint("n_head", c.nh), pint("head_dim", c.hd), pint("rot", c.rot), pint("off", c.off),
             pstr("mode", c.mode), pint("inverse", 1), pint("block", c.block), pstr("format", "fp8_e4m3"),
             pstr("scale", "ue8m0"), pstr("dtype", "bf16") };
}

/* one case: random x (groups zero / tiny / large), a cos_sin table of 3000 rows, positions (some past the table, some -1) */
void run(const std::string& name, const Case& c, uint32_t seed, bool expect_ok = true) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    const int64_t n = (int64_t)c.nh * c.hd, xp = n + c.xpad, qp = n + c.qpad, ng = n / 128 + (n % 128 ? 1 : 0);
    const int64_t tab = 3000;
    std::vector<float> cs((size_t)(tab * c.rot));
    for (int64_t p = 0; p < tab; ++p)
        for (int i = 0; i < c.rot / 2; ++i) {
            const double th = (double)p * std::pow(10000.0, -2.0 * i / c.rot);
            cs[(size_t)(p * c.rot + i)] = (float)std::cos(th);
            cs[(size_t)(p * c.rot + c.rot / 2 + i)] = (float)std::sin(th);
        }
    std::vector<int32_t> pos((size_t)c.M);
    for (int m = 0; m < c.M; ++m) {
        const int k = pct(rng);
        pos[(size_t)m] = k < 5 ? -1 : k < 10 ? (int32_t)(tab + 7) : (int32_t)(rng() % tab);
    }
    std::vector<uint16_t> x((size_t)(c.M * xp), 0x7f7f);
    for (int m = 0; m < c.M; ++m)
        for (int64_t g = 0; g < (n + 127) / 128; ++g) {
            const int k = pct(rng);
            const float sc = k < 6 ? 0.0f : k < 12 ? 1e-6f : k < 18 ? 300.0f : 1.0f;
            for (int64_t j = g * 128; j < std::min<int64_t>(n, (g + 1) * 128); ++j)
                x[(size_t)(m * xp + j)] = rad_f32_to_bf16(nd(rng) * sc);
        }
    const auto P = rope_params(c);
    /* the fused op */
    std::vector<uint16_t> x2 = x;
    std::vector<int8_t> q2((size_t)(c.M * qp), 0x55);
    std::vector<float> s2((size_t)(c.M * ng), -7.0f);
    RadTensor t2[5] = { tens(x2.data(), RAD_BF16, { c.M, n }, xp), tens(cs.data(), RAD_F32, { tab, c.rot }),
                        tens(pos.data(), RAD_I32, { c.M }), tens(q2.data(), RAD_I8, { c.M, n }, qp),
                        tens(s2.data(), RAD_F32, { c.M, ng }) };
    RadArgs a2{};
    a2.t = t2;
    a2.n_t = 5;
    a2.p = P.data();
    a2.n_p = (int)P.size();
    const int rc2 = r4d_rad_rope_rows_roundtrip_i8g(&a2, nullptr);
    if (!expect_ok) {
        expect(rc2 != RAD_OK, name, "the fused op served a geometry it must refuse", rc2, 0);
        long long w = 0;
        for (size_t i = 0; i < x.size(); ++i) w += x[i] != x2[i];
        expect(w == 0, name, "a refused call wrote x", w, 0);
        std::printf("%-44s refused (rc %d)\n", name.c_str(), rc2);
        return;
    }
    expect(rc2 == RAD_OK, name, "rope_rows_roundtrip_i8g", rc2, 0);
    for (int pin = 0; pin <= 1; ++pin) {
        r4d_rope8_pin(pin);
        std::vector<uint16_t> x1 = x;
        std::vector<int8_t> q1((size_t)(c.M * qp), 0x55);
        std::vector<float> s1((size_t)(c.M * ng), -7.0f);
        RadTensor t1[3] = { tens(x1.data(), RAD_BF16, { c.M, n }, xp), tens(cs.data(), RAD_F32, { tab, c.rot }),
                            tens(pos.data(), RAD_I32, { c.M }) };
        RadArgs a1{};
        a1.t = t1;
        a1.n_t = 3;
        a1.p = P.data();
        a1.n_p = (int)P.size();
        expect(r4d_rad_rope_rows_roundtrip(&a1, nullptr) == RAD_OK, name, "rope_rows_roundtrip", pin, 0);
        RadTensor tq[3] = { tens(x1.data(), RAD_BF16, { c.M, n }, xp), tens(q1.data(), RAD_I8, { c.M, n }, qp),
                            tens(s1.data(), RAD_F32, { c.M, ng }) };
        const RadParam pq[4] = { pint("M", c.M), pint("n", n), pint("group", 128), pstr("dtype", "bf16") };
        RadArgs aq{};
        aq.t = tq;
        aq.n_t = 3;
        aq.p = pq;
        aq.n_p = 4;
        expect(r4d_rad_quant_act_i8g(&aq, nullptr) == RAD_OK, name, "quant_act_i8g", pin, 0);
        long long dx = 0, dq = 0, ds = 0, rot_rows = 0;
        for (int m = 0; m < c.M; ++m) {
            for (int64_t j = 0; j < xp; ++j) dx += x1[(size_t)(m * xp + j)] != x2[(size_t)(m * xp + j)];   /* the pad too */
            for (int64_t j = 0; j < qp; ++j) dq += q1[(size_t)(m * qp + j)] != q2[(size_t)(m * qp + j)];
            for (int64_t g = 0; g < ng; ++g) ds += std::memcmp(&s1[(size_t)(m * ng + g)], &s2[(size_t)(m * ng + g)], 4) != 0;
            rot_rows += pos[(size_t)m] >= 0 && pos[(size_t)m] < tab;
        }
        expect(dx == 0, name, "x differs from the chain's", dx, pin);
        expect(dq == 0, name, "the int8 codes differ from quant_act_i8g's", dq, pin);
        expect(ds == 0, name, "the scales differ from quant_act_i8g's", ds, pin);
        long long moved = 0;
        for (size_t i = 0; i < x.size(); ++i) moved += x[i] != x1[i];
        expect(moved > 0, name, "the chain left x unchanged (the fixture reached nothing)", moved, 0);
        if (pin == 1)
            std::printf("%-44s rows %3d heads %2d n %5lld (pitch +%lld / +%lld), %lld roped rows: x %lld, codes %lld, scales %lld "
                        "differ (chain lane-group and ROPE8 forms)\n",
                        name.c_str(), c.M, c.nh, (long long)n, (long long)c.xpad, (long long)c.qpad, rot_rows, dx, dq, ds);
    }
    r4d_rope8_pin(-1);
}

}  // namespace

int main() {
    Case c;
    for (int M : { 1, 3, 17, 32, 40, 64, 130 }) {
        c.M = M;
        run("16 heads (TP4), rows " + std::to_string(M), c, 100 + M);
    }
    c.nh = 64;
    for (int M : { 1, 32 }) {
        c.M = M;
        run("64 heads (one rank), rows " + std::to_string(M), c, 300 + M);
    }
    c = Case{};
    c.M = 24; c.xpad = 64; c.qpad = 32;
    run("16 heads, 24 rows, x / q pitches wider", c, 400);
    c = Case{};
    c.M = 8; c.nh = 2; c.hd = 128; c.rot = 32; c.off = 96;
    run("2 heads x 128 (n 256), rot 32", c, 500);
    /* refusals */
    c = Case{};
    c.M = 4; c.nh = 1; c.hd = 64; c.rot = 32; c.off = 32;
    run("n 64 (not whole 128-column groups)", c, 600, false);
    c = Case{};
    c.M = 4; c.block = 16;
    run("block 16", c, 601, false);
    c = Case{};
    c.M = 4; c.mode = "neox";
    run("neox", c, 602, false);
    std::printf("%d check(s), %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
