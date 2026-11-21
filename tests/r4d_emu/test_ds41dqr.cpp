/* tests/r4d_emu: lever DQR's op (DRAFTER track round 69; libr4d attn_sparse_sink_dqr) on the host SIMT emulator (tp8 loop
 * QUEUE W15, DECODE round 56; the GPU twin: tools/dbench/dqr on one card). The DSpark stage's q_rope + attention as one
 * launch of the WMMA attention (ds41_attn_sparse_sink_wm_kernel QO), held byte for byte to the launches it replaces:
 *
 *   CHAIN   out_rope 0 (the form deepseek41 declares): attn_sparse_sink_dqr over the UNROTATED q = ds41_rope_rows (forward, at
 *           the rows' own positions) over a copy of q, then attn_sparse_sink at wm_from 1 (lever DAW: the WMMA form at every
 *           height) over the rotated copy -- every byte of out (sentinel-filled before each launch); out_rope 1 (offered by the
 *           op, not declared) = that chain, then ds41_rope_rows (inverse) + quant_roundtrip (fp8 e4m3 / ue8m0, blocks of 32),
 *           and = the same attention output through rope_rows_roundtrip.
 *   Q       q is not written by the op (every byte as before the launch).
 *   BI      each sequence alone (a launch over its own rows, its own cu_seqlens) = its rows inside the full pass.
 *   CTRL    the attention over the unrotated q differs from the chain somewhere (else CHAIN cannot see a missing rope); held
 *           at kinds 0 / 1 (kind 2's specials leave a NaN or an inf in nearly every head's q row: printed only).
 *   NEG     the op's bench ablation bit 1 (r4d_dqr_abl_pin 2: the output's rope + round trip left out) makes out_rope 1's
 *           CHAIN fail and leaves out_rope 0's: the checks see that part (q's part: CTRL). Bit 0 (documented as "skips q's rope")
 *           is read nowhere in the kernel at 577d3897 (only `a.abl & 2`): printed, not held.
 *   REFUSE  the forms the op does not serve return an error, never RAD_OK (the arch then keeps the chain): heads % 16 != 0,
 *           neox, wm_from 0 at a decode-band height, quant block 16, an fp8 window, rot / off not a multiple of 8.
 * Shapes: the DSpark query pass's sequences x block rows (1 x 5 .. 16 x 3: 5 .. 48 rows), 16 heads (a rank's), d 512, rot 64
 * at off 448 (gptj), cache blocks of 64 slots shuffled; each sequence's context drawn, its rows at their own positions
 * (ctx .. ctx + B - 1), the attention's positions every row at its block's last (apos), window 128 + B. Three input kinds:
 * typical; per-row magnitudes 2^-20 .. 2^20; specials (-0.0, +-0, +-inf, a NaN, subnormals in q -- one NaN in a rotated
 * column every ninth row -- and -0.0 / +-inf / a subnormal among the keys; every seventh row's own position past the angle
 * table: both ropes leave it). Tile skip 0 / 1 / 3 (r4d_attn_skip_pin: P32's key-tile skip, + rows longest first, + GL).
 *
 *   r4d_emu_ds41dqr [quick]      quick: kinds 0 and 2, tile skip 1 */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_dqr(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_rope_rows(const RadArgs*, RadStream);
extern "C" int r4d_rad_quant_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs*, RadStream);
extern "C" void r4d_attn_skip_pin(int v);
extern "C" void r4d_dqr_abl_pin(int v);

namespace {

constexpr int kH = 16, kD = 512, kRot = 64, kOff = kD - kRot, kHD = kH * kD, kBs = 64, kWin0 = 128;
constexpr int kCtxMax = 900, kTab = kCtxMax + 64, kMaxS = 16;
constexpr int kNwbt = (kCtxMax + 8 + kBs - 1) / kBs, kNblk = kMaxS * kNwbt + 8;

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
RadTensor none() { return RadTensor{}; }
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
int launch(int (*fn)(const RadArgs*, RadStream), std::vector<RadTensor> t, const std::vector<RadParam>& p) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    a.wgp_count = 32;
    return fn(&a, nullptr);
}

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
}
long long diff(const uint16_t* a, const uint16_t* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
    return d;
}

uint64_t g_rs = 0x9e3779b97f4a7c15ull;
uint32_t r32() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return (uint32_t)(g_rs >> 16); }
float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }
uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

/* The rope table (cos | sin a row, rope_table's plane), the sinks, the shuffled block table, the cache. */
struct World {
    std::vector<float> cs, sink;
    std::vector<int32_t> bt;
    std::vector<uint16_t> cache;
    World() : cs((size_t)kTab * kRot), sink(kH), bt((size_t)kMaxS * kNwbt), cache((size_t)kNblk * kBs * kD) {
        for (int p = 0; p < kTab; ++p)
            for (int i = 0; i < kRot / 2; ++i) {
                const double th = p * std::pow(10000.0, -2.0 * i / kRot);
                cs[(size_t)p * kRot + i] = (float)std::cos(th);
                cs[(size_t)p * kRot + kRot / 2 + i] = (float)std::sin(th);
            }
        for (auto& v : sink) v = uni(-1.0f, 1.0f);
        std::vector<int32_t> perm(kNblk);
        for (int i = 0; i < kNblk; ++i) perm[i] = i;
        for (int i = kNblk - 1; i > 0; --i) std::swap(perm[i], perm[(int)(r32() % (uint32_t)(i + 1))]);
        for (size_t i = 0; i < bt.size(); ++i) bt[i] = perm[i];
    }
    void fill(int kind) {
        for (auto& v : cache) v = bf(uni(-1.0f, 1.0f));
        if (kind == 2) {
            const uint16_t sp[] = { 0x8000, 0x7f80, 0xff80, 0x0001 };   /* -0.0, inf, -inf, a subnormal (no NaN key) */
            for (size_t i = 0; i < cache.size(); i += 4099) cache[i] = sp[r32() % 4];
        }
    }
};

/* A pass: S sequences of B rows. */
struct Pass {
    int S = 0, B = 0, M = 0;
    std::vector<uint16_t> q;
    std::vector<int32_t> wu, pos, qpos, cu;
};
Pass make(int S, int B, int kind) {
    Pass p;
    p.S = S;
    p.B = B;
    p.M = S * B;
    p.q.resize((size_t)p.M * kHD);
    for (int m = 0; m < p.M; ++m) {
        const float sc = kind == 1 ? std::ldexp(1.0f, (int)(r32() % 41) - 20) : 1.0f;
        for (int i = 0; i < kHD; ++i) p.q[(size_t)m * kHD + i] = bf(uni(-1.0f, 1.0f) * sc);
    }
    if (kind == 2) {
        const uint16_t sp[] = { 0x8000, 0x0000, 0x7f80, 0xff80, 0x7fc1, 0x0001, 0x8001, 0x7f7f };
        for (size_t i = 0; i < p.q.size(); i += 211) p.q[i] = sp[r32() % 8];
        for (int m = 1; m < p.M; m += 9) p.q[(size_t)m * kHD + kOff + 3] = 0x7fc1;   /* a NaN in a rotated column */
    }
    p.wu.resize(S);
    p.pos.resize(p.M);
    p.qpos.resize(p.M);
    p.cu.resize(S + 1);
    for (int s = 0; s < S; ++s) {
        const int ctx = 40 + (int)(r32() % (uint32_t)(kCtxMax - 60));
        p.wu[s] = ctx + B;
        p.cu[s] = s * B;
        for (int r = 0; r < B; ++r) {
            p.pos[(size_t)s * B + r] = ctx + B - 1;
            p.qpos[(size_t)s * B + r] = ctx + r;
        }
    }
    p.cu[S] = p.M;
    if (kind == 2)   /* rows whose own position is off the rope table: both ropes leave them */
        for (int m = 2; m < p.M; m += 7) p.qpos[m] = kTab + 3;
    return p;
}

struct Geo {   /* the op's parameters that the REFUSE cases change */
    int heads = kH, rot = kRot, off = kOff, block = 32, wm_from = 1;
    const char* mode = "gptj";
    bool w8 = false;
};

/* The attention's operands for sequences [s0, s0 + S) of the pass (the q rows from `q`), out into `out`; with `dqr` the op's
 * two more operands and parameters. */
int attn(int (*fn)(const RadArgs*, RadStream), const World& w, const Pass& p, int s0, int S, const uint16_t* q, uint16_t* out,
         const int32_t* cu, bool dqr, int out_rope = 0, const Geo& g = Geo()) {
    const int B = p.B, M = S * B, HD = g.heads * kD;
    std::vector<RadTensor> t = {
        tens(q, RAD_BF16, { M, HD }),
        tens(w.cache.data(), RAD_BF16, { kNblk, 1, kBs, g.w8 ? (kD + kD / 32 + 15) / 16 * 16 / 2 : kD }),
        tens(w.bt.data() + (size_t)s0 * kNwbt, RAD_I32, { S, kNwbt }),
        tens(p.wu.data() + s0, RAD_I32, { S }),
        none(), none(), none(),
        tens(p.pos.data() + (size_t)s0 * B, RAD_I32, { M }),
        tens(cu, RAD_I32, { S + 1 }),
        tens(w.sink.data(), RAD_F32, { g.heads }),
        tens(out, RAD_BF16, { M, HD }),
    };
    std::vector<RadParam> pr = { pint("M", M),           pint("heads", g.heads), pint("dim", kD),
                                 pint("window", kWin0 + B), pint("ratio", 0),  pint("topk", 0),
                                 pf64("scale", 1.0 / std::sqrt((double)kD)), pstr("dtype", "bf16"),
                                 pint("wm_from", g.wm_from) };
    if (g.w8) pr.push_back(pstr("window_format", "fp8_e4m3"));
    if (dqr) {
        t.push_back(tens(w.cs.data(), RAD_F32, { kTab, g.rot }));
        t.push_back(tens(p.qpos.data() + (size_t)s0 * B, RAD_I32, { M }));
        pr.push_back(pint("rot", g.rot));
        pr.push_back(pint("off", g.off));
        pr.push_back(pstr("mode", g.mode));
        pr.push_back(pint("block", g.block));
        pr.push_back(pstr("format", "fp8_e4m3"));
        pr.push_back(pstr("block_scale", "ue8m0"));
        pr.push_back(pint("out_rope", out_rope));
    }
    return launch(fn, t, pr);
}
int rope(const World& w, const Pass& p, uint16_t* x, int inverse) {
    return launch(r4d_rad_ds41_rope_rows,
                  { tens(x, RAD_BF16, { p.M, kHD }), tens(w.cs.data(), RAD_F32, { kTab, kRot }),
                    tens(p.qpos.data(), RAD_I32, { p.M }) },
                  { pint("M", p.M), pint("n_head", kH), pint("head_dim", kD), pint("rot", kRot), pint("off", kOff),
                    pstr("mode", "gptj"), pint("inverse", inverse), pstr("dtype", "bf16") });
}
int qrt(const Pass& p, uint16_t* x) {
    return launch(r4d_rad_quant_roundtrip, { tens(x, RAD_BF16, { p.M, kHD }) },
                  { pint("M", p.M), pint("n", kHD), pint("block", 32), pstr("format", "fp8_e4m3"), pstr("scale", "ue8m0"),
                    pstr("dtype", "bf16") });
}
int ort(const World& w, const Pass& p, uint16_t* x) {
    return launch(r4d_rad_rope_rows_roundtrip,
                  { tens(x, RAD_BF16, { p.M, kHD }), tens(w.cs.data(), RAD_F32, { kTab, kRot }),
                    tens(p.qpos.data(), RAD_I32, { p.M }) },
                  { pint("M", p.M), pint("n_head", kH), pint("head_dim", kD), pint("rot", kRot), pint("off", kOff),
                    pstr("mode", "gptj"), pint("inverse", 1), pint("block", 32), pstr("format", "fp8_e4m3"),
                    pstr("scale", "ue8m0"), pstr("dtype", "bf16") });
}

void run_case(const World& w, int S, int B, int kind, int skp) {
    const Pass p = make(S, B, kind);
    const int M = p.M;
    const size_t n = (size_t)M * kHD;
    const std::string name = "kind " + std::to_string(kind) + ", " + std::to_string(S) + " x " + std::to_string(B) + " = " +
                             std::to_string(M) + " rows, tile skip " + std::to_string(skp);
    r4d_attn_skip_pin(skp);
    /* the chain: rope a copy of q forward, the attention over it */
    std::vector<uint16_t> qr = p.q;
    expect(rope(w, p, qr.data(), 0) == RAD_OK, name, "ds41_rope_rows (q, forward)");
    std::vector<uint16_t> a(n, 0x5555);
    expect(attn(r4d_rad_ds41_attn_sparse_sink, w, p, 0, S, qr.data(), a.data(), p.cu.data(), false) == RAD_OK, name,
           "attn_sparse_sink (wm_from 1)");
    /* out_rope 1's chains: inverse rope + quant round trip, and rope_rows_roundtrip, each over a copy of a */
    std::vector<uint16_t> a3 = a, a2 = a;
    expect(rope(w, p, a3.data(), 1) == RAD_OK, name, "ds41_rope_rows (out, inverse)");
    expect(qrt(p, a3.data()) == RAD_OK, name, "quant_roundtrip");
    expect(ort(w, p, a2.data()) == RAD_OK, name, "rope_rows_roundtrip");
    /* the op over the unrotated q */
    const std::vector<uint16_t> q0 = p.q;
    std::vector<uint16_t> o0(n, 0x3333), o1(n, 0x2222), c(n, 0x1111);
    expect(attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, 0, S, p.q.data(), o0.data(), p.cu.data(), true, 0) == RAD_OK, name,
           "attn_sparse_sink_dqr out_rope 0");
    expect(attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, 0, S, p.q.data(), o1.data(), p.cu.data(), true, 1) == RAD_OK, name,
           "attn_sparse_sink_dqr out_rope 1");
    const long long dq = diff(p.q.data(), q0.data(), n);
    const long long d0 = diff(o0.data(), a.data(), n), d1 = diff(o1.data(), a3.data(), n), d12 = diff(o1.data(), a2.data(), n);
    /* control: the attention over the unrotated q */
    expect(attn(r4d_rad_ds41_attn_sparse_sink, w, p, 0, S, p.q.data(), c.data(), p.cu.data(), false) == RAD_OK, name,
           "attn_sparse_sink over the unrotated q");
    const long long dc = diff(c.data(), a.data(), n), dor = diff(a.data(), a3.data(), n);
    /* BI: every sequence alone (the first, the last and up to two between) vs its rows in the pass */
    long long dbi = 0;
    int nbi = 0;
    std::vector<int> pick = { 0, S / 3, (2 * S) / 3, S - 1 };
    std::sort(pick.begin(), pick.end());
    pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
    for (int s : pick) {
        const int32_t cu1[2] = { 0, B };
        std::vector<uint16_t> one((size_t)B * kHD, 0x4444);
        expect(attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, s, 1, p.q.data() + (size_t)s * B * kHD, one.data(), cu1, true, 0) ==
                   RAD_OK, name, "attn_sparse_sink_dqr, a sequence alone");
        dbi += diff(one.data(), o0.data() + (size_t)s * B * kHD, one.size());
        ++nbi;
    }
    long long nz = 0;
    for (size_t i = 0; i < n; ++i) nz += (o0[i] & 0x7fffu) != 0;
    expect(d0 == 0, name, "out_rope 0 differs from ds41_rope_rows (q) + attn_sparse_sink", d0, (long long)n);
    expect(d1 == 0, name, "out_rope 1 differs from the chain + ds41_rope_rows (inverse) + quant_roundtrip", d1, (long long)n);
    expect(d12 == 0, name, "out_rope 1 differs from the attention + rope_rows_roundtrip", d12, (long long)n);
    expect(dq == 0, name, "q was written by the op", dq, (long long)n);
    expect(dbi == 0, name, "a sequence alone differs from its rows in the pass", dbi, nbi);
    /* kind 2's specials saturate most rows (a NaN or an inf in every head's q row): its control is printed, not held */
    expect(kind == 2 || dc > 0, name, "CTRL: the attention over the unrotated q equals the chain (the rope check is blind)", dc, 0);
    expect(dor > 0, name, "CTRL: the output rope + round trip changed nothing", dor, 0);
    expect(nz > (long long)n / 2, name, "the output is mostly zero", nz, (long long)n);
    std::printf("%-44s out_rope 0 vs rope + attn %lld, out_rope 1 vs chain %lld / vs rope_rows_roundtrip %lld of %zu differ; "
                "q written %lld; %d sequences alone %lld; ctrl %lld\n",
                name.c_str(), d0, d1, d12, n, dq, nbi, dbi, dc);
}

/* NEG: the ablated op (bytes knowingly wrong) must differ from the chain in the part it leaves out, and only there. */
void neg_case(const World& w) {
    const Pass p = make(8, 2, 0);
    const size_t n = (size_t)p.M * kHD;
    std::vector<uint16_t> qr = p.q, a(n, 0x5555);
    expect(rope(w, p, qr.data(), 0) == RAD_OK, "NEG", "ds41_rope_rows (q, forward)");
    expect(attn(r4d_rad_ds41_attn_sparse_sink, w, p, 0, p.S, qr.data(), a.data(), p.cu.data(), false) == RAD_OK, "NEG", "attn");
    std::vector<uint16_t> a3 = a;
    expect(rope(w, p, a3.data(), 1) == RAD_OK && qrt(p, a3.data()) == RAD_OK, "NEG", "out chain");
    long long d[4][2] = {};
    for (int abl = 0; abl < 4; ++abl) {
        r4d_dqr_abl_pin(abl);
        std::vector<uint16_t> o0(n, 0x3333), o1(n, 0x2222);
        expect(attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, 0, p.S, p.q.data(), o0.data(), p.cu.data(), true, 0) == RAD_OK, "NEG",
               "dqr out_rope 0");
        expect(attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, 0, p.S, p.q.data(), o1.data(), p.cu.data(), true, 1) == RAD_OK, "NEG",
               "dqr out_rope 1");
        d[abl][0] = diff(o0.data(), a.data(), n);
        d[abl][1] = diff(o1.data(), a3.data(), n);
    }
    r4d_dqr_abl_pin(0);
    expect(d[0][0] == 0 && d[0][1] == 0, "NEG", "unablated op differs", d[0][0], d[0][1]);
    expect(d[2][0] == 0 && d[2][1] > 0, "NEG", "the output's rope left out: out_rope 0 must stay, out_rope 1 must differ", d[2][0], d[2][1]);
    std::printf("NEG  ablation 0 / 1 / 2 / 3: out_rope 0 differs %lld / %lld / %lld / %lld, out_rope 1 %lld / %lld / %lld / %lld of %zu\n",
                d[0][0], d[1][0], d[2][0], d[3][0], d[0][1], d[1][1], d[2][1], d[3][1], n);
}

void refuse_cases(const World& w) {
    const Pass p = make(2, 3, 0);
    const size_t n = (size_t)p.M * kHD;
    struct R {
        const char* what;
        Geo g;
    };
    std::vector<R> rs;
    { R r{ "heads 8 (heads % 16)", Geo() }; r.g.heads = 8; rs.push_back(r); }
    { R r{ "neox", Geo() }; r.g.mode = "neox"; rs.push_back(r); }
    { R r{ "wm_from 0 at 6 rows (the decode band: the scalar kernels)", Geo() }; r.g.wm_from = 0; rs.push_back(r); }
    { R r{ "quant block 16", Geo() }; r.g.block = 16; rs.push_back(r); }
    { R r{ "an fp8 window", Geo() }; r.g.w8 = true; rs.push_back(r); }
    { R r{ "rot 60 at off 452 (not 8-column chunks)", Geo() }; r.g.rot = 60; r.g.off = 452; rs.push_back(r); }
    std::vector<uint16_t> q((size_t)p.M * 16 * kD, 0x3c00), o((size_t)p.M * 16 * kD, 0x7777);
    for (const R& r : rs) {
        const int rc = attn(r4d_rad_ds41_attn_sparse_sink_dqr, w, p, 0, p.S, q.data(), o.data(), p.cu.data(), true, 0, r.g);
        expect(rc != RAD_OK, std::string("REFUSE ") + r.what, "served (RAD_OK) where the op must refuse", rc, 0);
        long long touched = 0;
        for (size_t i = 0; i < o.size(); ++i) touched += o[i] != 0x7777;
        expect(touched == 0, std::string("REFUSE ") + r.what, "a refused launch wrote out", touched, 0);
        std::printf("REFUSE %-58s rc %d, out bytes written %lld\n", r.what, rc, touched);
    }
    (void)n;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    World w;
    struct Sh { int S, B; };
    const Sh shapes[] = { { 1, 5 }, { 3, 5 }, { 8, 2 }, { 4, 4 }, { 16, 2 }, { 8, 5 }, { 16, 3 } };
    for (int kind = 0; kind < 3; ++kind) {
        if (quick && kind == 1) continue;
        w.fill(kind);
        for (const Sh& sh : shapes)
            for (int skp : { 1, 0, 3 }) {
                if (quick && skp != 1) continue;
                run_case(w, sh.S, sh.B, kind, skp);
            }
    }
    r4d_attn_skip_pin(1);
    w.fill(0);
    neg_case(w);
    refuse_cases(w);
    r4d_attn_skip_pin(-1);
    std::printf("ds41 lever DQR (attn_sparse_sink_dqr): %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
