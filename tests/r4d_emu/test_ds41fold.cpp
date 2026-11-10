/* tests/r4d_emu: lever K1's exact launch folds on DeepSeek-V4.1's attention path (10.11.2026), on the
 * host SIMT emulator, byte for byte against the separate rows they fold:
 *
 *   fold c  rmsnorm_rope_roundtrip_kv_store (r4d_ds41_elem.hip, norm_rope_rt_kernel under KV) against
 *           rmsnorm_rope_roundtrip (the same kernel without KV) followed by kv_store_bf16's copy of y's row
 *           (kv_store_kernel, r4d_model_bf16.hip, ported here: that unit is not in the emulator's build;
 *           the selftest's ds41 fold c case holds the fold to the device row itself). y and the whole
 *           cache (sentinel-filled, so a store outside the row's slot shows) must be equal, every byte,
 *           at the window key's geometry (n 512, rot 64 at off 448, fp8 e4m3 / ue8m0 in blocks of 32,
 *           block_size 16) for M 1..48 (the decode band), 64 and 130 (prefill heights), and at other
 *           geometries the row takes (f32 scale, blocks of 16, a serial block of 24, inverse, bf16 gains),
 *           with slots unmapped (-1) or past the cache, positions outside the angle table, and a NaN row;
 *   fold d  attn_sparse_sink_rope_roundtrip (r4d_ds41_sparse.hip, ds41_attn_dec_kernel under ORT, else
 *           the two launches) against attn_sparse_sink followed by rope_rows_roundtrip, and against
 *           attn_sparse_sink, ds41_rope_rows and quant_roundtrip (three rows): the decode band's forms
 *           (1-48 rows: the direct-PV head groups 1/2/4/8 and the tiled-PV form), bf16 and packed (nvfp4)
 *           latents, window-only (ratio 0) and compressed layers, rows whose position is past the angle
 *           table, the production geometry (inverse gptj rot 64 at off 448 of 512, fp8 e4m3 / ue8m0, block
 *           32) and others (forward, fp4 e2m1, off 0, rot 512), and the passes the fold leaves to the two
 *           launches (WMMA prefill at 64 rows, the scalar kernel, neox, blocks of 16). */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_rmsnorm_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_rmsnorm_rope_roundtrip_kv_store(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_rope_rows(const RadArgs*, RadStream);
extern "C" int r4d_rad_quant_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kSlots = 144;

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
RadTensor none() { return RadTensor{}; }
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
int launch(int (*fn)(const RadArgs*, RadStream), RadTensor* t, int nt, const std::vector<RadParam>& p) {
    RadArgs a{};
    a.t = t;
    a.n_t = nt;
    a.p = p.data();
    a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
}
template <class T>
long long diff(const std::vector<T>& a, const std::vector<T>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

/* A rope_table plane: cos then sin of `rot / 2` angles a position (any values do: the ops only read it). */
std::vector<float> angles(std::mt19937& rng, int64_t rows, int64_t rot) {
    std::uniform_real_distribution<float> u(-3.2f, 3.2f);
    std::vector<float> cs((size_t)(rows * rot));
    for (int64_t p = 0; p < rows; ++p)
        for (int64_t i = 0; i < rot / 2; ++i) {
            const float a = u(rng);
            cs[(size_t)(p * rot + i)] = std::cos(a);
            cs[(size_t)(p * rot + rot / 2 + i)] = std::sin(a);
        }
    return cs;
}

/* ============================================================ fold c */

struct KvCase {
    const char* name;
    int64_t M, n, rot, off, block;
    const char* fmt;
    const char* scale;
    int inverse;
    bool wf32;
};

/* kv_store_kernel (r4d_model_bf16.hip) at kv_heads 1, head_dim n / 2, k = y[:, 0, n/2), v = y[:, n/2, n). */
void kv_store_ref(const std::vector<uint16_t>& y, int64_t M, int64_t n, const std::vector<int32_t>& slot,
                  int64_t bs, int64_t nb, std::vector<uint16_t>& kv) {
    const int64_t hd = n / 2;
    for (int64_t t = 0; t < M; ++t) {
        const int32_t s = slot[(size_t)t];
        if (s < 0 || (uint32_t)s / (uint32_t)bs >= (uint32_t)nb) continue;
        const int64_t blk = s / bs, off = s % bs;
        const size_t base = ((size_t)blk * 1 * bs + (size_t)off) * 2 * hd;
        for (int64_t j = 0; j < hd; ++j) {
            kv[base + (size_t)j] = y[(size_t)(t * n + j)];
            kv[base + (size_t)(hd + j)] = y[(size_t)(t * n + hd + j)];
        }
    }
}

void kv_case(const KvCase& c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    const int64_t M = c.M, n = c.n, bs = 16, nb = M / bs + 3, tab = 300;
    const std::string name = std::string(c.name) + " M=" + std::to_string(M);
    /* x: a row a scale (some tiny, some large), one NaN row when M > 3 */
    std::vector<uint16_t> x((size_t)(M * n));
    for (int64_t m = 0; m < M; ++m) {
        const float sc = m % 7 == 3 ? 1e-3f : m % 7 == 5 ? 40.0f : 1.0f;
        for (int64_t i = 0; i < n; ++i) x[(size_t)(m * n + i)] = rad_f32_to_bf16(nd(rng) * sc);
    }
    if (M > 3) x[(size_t)(2 * n + 17)] = 0x7fc1;
    std::vector<float> wf((size_t)n);
    std::vector<uint16_t> wb((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        wf[(size_t)i] = 0.5f + 0.1f * nd(rng);
        wb[(size_t)i] = rad_f32_to_bf16(wf[(size_t)i]);
    }
    std::vector<float> cs = angles(rng, tab, c.rot);
    /* positions: mostly in the table, some past it or negative */
    std::vector<int32_t> pos((size_t)M);
    for (int64_t m = 0; m < M; ++m) {
        const int k = pct(rng);
        pos[(size_t)m] = k < 6 ? (int32_t)(tab + k) : k < 9 ? -1 : (int32_t)(rng() % tab);
    }
    /* slots: unique over the cache, some unmapped (-1), one past the cache */
    std::vector<int32_t> perm((size_t)(nb * bs));
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> slot((size_t)M);
    for (int64_t m = 0; m < M; ++m) slot[(size_t)m] = pct(rng) < 10 ? -1 : perm[(size_t)m];
    if (M > 4) slot[(size_t)(M - 2)] = (int32_t)(nb * bs + 3);
    std::vector<RadParam> p = { pint("M", M),
                                pint("n", n),
                                pf64("eps", 1e-6),
                                pint("n_head", 1),
                                pint("head_dim", n),
                                pint("rot", c.rot),
                                pint("off", c.off),
                                pstr("mode", "gptj"),
                                pint("inverse", c.inverse),
                                pint("block", c.block),
                                pstr("format", c.fmt),
                                pstr("scale", c.scale),
                                pstr("dtype", "bf16") };
    const void* w = c.wf32 ? (const void*)wf.data() : (const void*)wb.data();
    const uint32_t wdt = c.wf32 ? RAD_F32 : RAD_BF16;
    /* the separate rows */
    std::vector<uint16_t> y1((size_t)(M * n), 0x1234), kv1((size_t)(nb * bs * n), 0x5a5a);
    {
        RadTensor t[5] = { tens(x.data(), RAD_BF16, { M, n }), tens(w, wdt, { n }), tens(y1.data(), RAD_BF16, { M, n }),
                           tens(cs.data(), RAD_F32, { tab, c.rot }), tens(pos.data(), RAD_I32, { M }) };
        expect(launch(r4d_rad_rmsnorm_rope_roundtrip, t, 5, p) == RAD_OK, name, "rmsnorm_rope_roundtrip");
        kv_store_ref(y1, M, n, slot, bs, nb, kv1);
    }
    /* the fold */
    std::vector<uint16_t> y2((size_t)(M * n), 0x4321), kv2((size_t)(nb * bs * n), 0x5a5a);
    {
        std::vector<RadParam> pk = p;
        pk.push_back(pint("block_size", bs));
        pk.push_back(pstr("kv_dtype", "bf16"));
        RadTensor t[7] = { tens(x.data(), RAD_BF16, { M, n }), tens(w, wdt, { n }), tens(y2.data(), RAD_BF16, { M, n }),
                           tens(cs.data(), RAD_F32, { tab, c.rot }), tens(pos.data(), RAD_I32, { M }),
                           tens(slot.data(), RAD_I32, { M }), tens(kv2.data(), RAD_BF16, { nb, 1, bs, n }) };
        expect(launch(r4d_rad_rmsnorm_rope_roundtrip_kv_store, t, 7, pk) == RAD_OK, name,
               "rmsnorm_rope_roundtrip_kv_store");
    }
    const long long dy = diff(y1, y2), dk = diff(kv1, kv2);
    long long stored = 0;
    for (size_t i = 0; i < kv1.size(); ++i) stored += kv1[i] != 0x5a5a;
    expect(dy == 0, name, "y differs from rmsnorm_rope_roundtrip's", dy, (long long)y1.size());
    expect(dk == 0, name, "the cache differs from kv_store's", dk, (long long)kv1.size());
    expect(stored > 0, name, "the reference stored nothing (the fixture maps no slot)", stored, 0);
    std::printf("fold c  %-44s y differ %lld of %lld, cache differ %lld of %lld (%lld stored)\n", name.c_str(), dy,
                (long long)y1.size(), dk, (long long)kv1.size(), stored);
}

/* ============================================================ fold d (the fixture of test_ds41attn.cpp) */

void roundtrip16(uint16_t* v) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(rad_bf16_to_f32(v[i])));
    amax = std::max(amax, 6.0f * 0x1p-9f);
    const float scale = rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(amax / 6.0f));
    for (int i = 0; i < 16; ++i) {
        const float x = std::clamp(rad_bf16_to_f32(v[i]) / scale, -6.0f, 6.0f), ax = std::fabs(x);
        unsigned best = 0;
        for (unsigned c = 1; c < 8; ++c) {
            const float d = std::fabs(kMag[c] - ax), db = std::fabs(kMag[best] - ax);
            if (d < db || (d == db && (c & 1u) == 0u)) best = c;
        }
        v[i] = rad_f32_to_bf16((std::signbit(x) ? -1.0f : 1.0f) * kMag[best] * scale);
    }
}

struct Fix {
    int H = 0, ratio = 1, win = 128, topk = 64, S = 0, M = 0;
    int64_t nwblk = 0, ncblk = 0, nwbt = 0, ncbt = 0;
    const int bs = 16;
    std::vector<uint16_t> q, wc, cc, pk;
    std::vector<int32_t> wbt, wused, cbt, sel, pos, cu;
    std::vector<float> sink;
};

Fix make(uint32_t seed, int H, int ratio, const std::vector<std::pair<int, int>>& seqs, int topk = 64) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    Fix f;
    f.H = H;
    f.topk = ratio > 0 ? topk : 0;
    f.ratio = ratio;
    f.S = (int)seqs.size();
    f.cu.push_back(0);
    int64_t maxp = 0;
    for (const auto& s : seqs) {
        for (int i = 0; i < s.first; ++i) f.pos.push_back(s.second - s.first + 1 + i);
        f.cu.push_back(f.cu.back() + s.first);
        maxp = std::max<int64_t>(maxp, s.second + 1);
    }
    f.M = f.cu.back();
    auto bf = [&](float sc) { return rad_f32_to_bf16(nd(rng) * sc); };
    f.q.resize((size_t)f.M * H * kD);
    for (auto& x : f.q) x = bf(0.05f);
    f.nwbt = (maxp + f.bs - 1) / f.bs + 1;
    f.nwblk = f.S * f.nwbt + 3;
    std::vector<int32_t> perm((size_t)f.nwblk);
    for (int64_t i = 0; i < f.nwblk; ++i) perm[(size_t)i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    f.wbt.assign((size_t)(f.S * f.nwbt), -1);
    for (int s = 0; s < f.S; ++s)
        for (int64_t b = 0; b < f.nwbt; ++b) f.wbt[(size_t)(s * f.nwbt + b)] = perm[(size_t)(s * f.nwbt + b)];
    f.wc.resize((size_t)(f.nwblk * f.bs * kD));
    for (auto& x : f.wc) x = bf(1.0f);
    for (const auto& sq : seqs) f.wused.push_back((int32_t)(sq.second + 1));
    if (ratio > 0) {
        const int64_t nent = maxp / ratio + 1;
        f.ncbt = nent + 1;
        f.ncblk = f.S * f.ncbt + 5;
        std::vector<int32_t> cperm((size_t)f.ncblk);
        for (int64_t i = 0; i < f.ncblk; ++i) cperm[(size_t)i] = (int32_t)i;
        std::shuffle(cperm.begin(), cperm.end(), rng);
        f.cbt.assign((size_t)(f.S * f.ncbt), -1);
        for (int s = 0; s < f.S; ++s)
            for (int64_t c = 0; c < f.ncbt; ++c)
                f.cbt[(size_t)(s * f.ncbt + c)] = pct(rng) < 2 ? -1 : cperm[(size_t)(s * f.ncbt + c)];
        f.cc.resize((size_t)(f.ncblk * kD));
        for (int64_t e = 0; e < f.ncblk; ++e)
            for (int b = 0; b < kD / 16; ++b) {
                const int k = pct(rng);
                const float sc = k < 8 ? 1e-3f : k < 16 ? 0.02f : k < 18 ? 0.0f : 1.0f;
                uint16_t* v = &f.cc[(size_t)(e * kD + b * 16)];
                for (int i = 0; i < 16; ++i) v[i] = bf(sc);
                roundtrip16(v);
            }
        f.sel.resize((size_t)f.M * f.topk);
        for (int m = 0; m < f.M; ++m) {
            const int64_t reach = ((int64_t)f.pos[(size_t)m] + 1) / ratio;
            std::uniform_int_distribution<int64_t> any(0, std::max<int64_t>(0, reach - 1));
            for (int t = 0; t < f.topk; ++t)
                f.sel[(size_t)(m * f.topk + t)] = (reach > 0 && pct(rng) < 95) ? (int32_t)any(rng) : -1;
        }
    }
    f.sink.resize((size_t)H);
    for (auto& x : f.sink) x = nd(rng);
    return f;
}

int store_nv4(const Fix& f, uint16_t* packed) {
    std::vector<int32_t> page((size_t)f.ncblk);
    for (int64_t i = 0; i < f.ncblk; ++i) page[(size_t)i] = (int32_t)i;
    RadTensor t[3] = { tens(f.cc.data(), RAD_BF16, { f.ncblk, kD }), tens(page.data(), RAD_I32, { f.ncblk }),
                       tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }) };
    return launch(r4d_ds41_entry_store_nv4, t, 3,
                  { pint("M", f.ncblk), pint("n", kD), pint("block_size", f.ratio), pstr("dtype", "bf16"),
                    pstr("format", "nvfp4") });
}

struct Rt {   /* the rope and round trip's geometry */
    int rot = 64, off = 448, inverse = 1, block = 32;
    const char* mode = "gptj";
    const char* fmt = "fp8_e4m3";
    const char* scale = "ue8m0";
};

/* The attention's operands (+ cos_sin for the fold) and parameters (+ the fold's). */
int attn(int (*fn)(const RadArgs*, RadStream), const Fix& f, const uint16_t* packed, uint16_t* out, int64_t q_pitch,
         const uint16_t* q, const Rt* rt, const std::vector<float>* cs, int64_t tab) {
    RadTensor t[12] = {
        tens(q, RAD_BF16, { f.M, (int64_t)f.H * kD }, q_pitch),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { f.S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { f.S }),
        f.ratio == 0 ? none()
        : packed     ? tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
                     : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        f.ratio == 0 ? none() : tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        f.ratio == 0 ? none() : tens(f.sel.data(), RAD_I32, { f.M, f.topk }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out, RAD_BF16, { f.M, (int64_t)f.H * kD }),
        rt ? tens(cs->data(), RAD_F32, { tab, rt->rot }) : none(),
    };
    std::vector<RadParam> p = { pint("M", f.M),          pint("heads", f.H),   pint("dim", kD),
                                pint("window", f.win),   pint("ratio", f.ratio), pint("topk", f.topk),
                                pf64("scale", 1.0 / std::sqrt((double)kD)), pstr("dtype", "bf16") };
    if (packed) p.push_back(pstr("latent_format", "nvfp4"));
    if (rt) {
        p.push_back(pint("rot", rt->rot));
        p.push_back(pint("off", rt->off));
        p.push_back(pstr("mode", rt->mode));
        p.push_back(pint("inverse", rt->inverse));
        p.push_back(pint("block", rt->block));
        p.push_back(pstr("format", rt->fmt));
        p.push_back(pstr("block_scale", rt->scale));
    }
    return launch(fn, t, rt ? 12 : 11, p);
}

void ord_case(const std::string& name, const Fix& f, bool packed_lat, const Rt& rt, int64_t tab, int64_t q_pitch = 0) {
    std::mt19937 rng((uint32_t)f.M * 7919u + (uint32_t)rt.rot);
    const std::vector<float> cs = angles(rng, tab, rt.rot);
    std::vector<uint16_t> pk;
    if (packed_lat) {
        pk.assign((size_t)(f.ncblk * kSlots), 0x7777);
        expect(store_nv4(f, pk.data()) == RAD_OK, name, "entry_store nvfp4");
    }
    const uint16_t* pkp = packed_lat ? pk.data() : nullptr;
    const int64_t qp = q_pitch ? q_pitch : (int64_t)f.H * kD;
    std::vector<uint16_t> qv((size_t)(f.M * qp), 0);
    for (int m = 0; m < f.M; ++m) std::memcpy(&qv[(size_t)(m * qp)], &f.q[(size_t)m * f.H * kD], (size_t)f.H * kD * 2);
    const int64_t M = f.M, n = (int64_t)f.H * kD;
    const size_t no = (size_t)(M * n);
    /* the attention alone, then the two chains over copies of its output */
    std::vector<uint16_t> o0(no, 0x5555);
    expect(attn(r4d_rad_ds41_attn_sparse_sink, f, pkp, o0.data(), q_pitch, qv.data(), nullptr, nullptr, 0) == RAD_OK,
           name, "attn_sparse_sink");
    std::vector<uint16_t> o2 = o0, o3 = o0;
    const std::vector<RadParam> rp = { pint("M", M),           pint("n_head", f.H),       pint("head_dim", kD),
                                       pint("rot", rt.rot),    pint("off", rt.off),       pstr("mode", rt.mode),
                                       pint("inverse", rt.inverse), pint("block", rt.block), pstr("format", rt.fmt),
                                       pstr("scale", rt.scale), pstr("dtype", "bf16") };
    {
        RadTensor t[3] = { tens(o2.data(), RAD_BF16, { M, n }), tens(cs.data(), RAD_F32, { tab, rt.rot }),
                           tens(f.pos.data(), RAD_I32, { M }) };
        expect(launch(r4d_rad_rope_rows_roundtrip, t, 3, rp) == RAD_OK, name, "rope_rows_roundtrip");
    }
    {
        RadTensor t[3] = { tens(o3.data(), RAD_BF16, { M, n }), tens(cs.data(), RAD_F32, { tab, rt.rot }),
                           tens(f.pos.data(), RAD_I32, { M }) };
        expect(launch(r4d_rad_ds41_rope_rows, t, 3,
                      { pint("M", M), pint("n_head", f.H), pint("head_dim", kD), pint("rot", rt.rot),
                        pint("off", rt.off), pstr("mode", rt.mode), pint("inverse", rt.inverse), pstr("dtype", "bf16") }) ==
                   RAD_OK,
               name, "ds41_rope_rows");
        RadTensor tq[1] = { tens(o3.data(), RAD_BF16, { M, n }) };
        expect(launch(r4d_rad_quant_roundtrip, tq, 1,
                      { pint("M", M), pint("n", n), pint("block", rt.block), pstr("format", rt.fmt),
                        pstr("scale", rt.scale), pstr("dtype", "bf16") }) == RAD_OK,
               name, "quant_roundtrip");
    }
    /* the fold */
    std::vector<uint16_t> o1(no, 0x3333);
    expect(attn(r4d_rad_ds41_attn_sparse_sink_rope_roundtrip, f, pkp, o1.data(), q_pitch, qv.data(), &rt, &cs, tab) ==
               RAD_OK,
           name, "attn_sparse_sink_rope_roundtrip");
    const long long d12 = diff(o1, o2), d13 = diff(o1, o3), d23 = diff(o2, o3), d0 = diff(o0, o2);
    long long nz = 0;
    for (size_t i = 0; i < no; ++i) nz += (o1[i] & 0x7fffu) != 0;
    long long rows_out = 0;
    for (int m = 0; m < f.M; ++m) rows_out += f.pos[(size_t)m] >= tab;
    expect(d12 == 0, name, "the fold differs from attn + rope_rows_roundtrip", d12, (long long)no);
    expect(d13 == 0, name, "the fold differs from attn + rope_rows + quant_roundtrip", d13, (long long)no);
    expect(d23 == 0, name, "rope_rows_roundtrip differs from rope_rows + quant_roundtrip", d23, (long long)no);
    expect(d0 > 0, name, "the rope and round trip changed nothing (the fixture is too weak)", d0, 0);
    expect(nz > (long long)no / 2, name, "the output is mostly zero", nz, (long long)no);
    std::printf("fold d  %-66s rows %d heads %d ratio %d keys %d+%d %s%s rot %d off %d inv %d block %d %s/%s, "
                "%lld rows past the table: fold vs pair %lld, vs chain %lld of %lld (rope+rt changed %lld)\n",
                name.c_str(), f.M, f.H, f.ratio, f.win, f.topk, packed_lat ? "nvfp4" : "bf16",
                q_pitch ? " pitched q" : "", rt.rot, rt.off, rt.inverse, rt.block, rt.fmt, rt.scale, rows_out, d12, d13,
                (long long)no, d0);
}

}  // namespace

int main() {
    /* ---- fold c: the window key (n 512, rot 64 at off 448, fp8 / ue8m0 blocks of 32), every decode height */
    const KvCase win = { "window key", 0, 512, 64, 448, 32, "fp8_e4m3", "ue8m0", 0, true };
    uint32_t seed = 100;
    for (int64_t M = 1; M <= 48; ++M) {
        KvCase c = win;
        c.M = M;
        kv_case(c, seed++);
    }
    for (int64_t M : { 64, 130 }) {
        KvCase c = win;
        c.M = M;
        kv_case(c, seed++);
    }
    /* other geometries the row takes */
    kv_case({ "f32 scale, inverse", 9, 512, 64, 448, 32, "fp8_e4m3", "f32", 1, true }, seed++);
    kv_case({ "blocks of 16, bf16 gains", 12, 512, 64, 448, 16, "fp8_e4m3", "ue8m0", 0, false }, seed++);
    kv_case({ "serial block 24, n 384, rot 128 at 0", 7, 384, 128, 0, 24, "fp8_e4m3", "ue8m0", 0, true }, seed++);
    kv_case({ "n 2048 (two batches a thread)", 5, 2048, 64, 1984, 32, "fp8_e4m3", "ue8m0", 0, true }, seed++);

    /* ---- fold d: the production geometry over the decode band's forms */
    const Rt prod;
    ord_case("decode, 1 row, head group 1, window only", make(201, 8, 0, { { 1, 3000 } }), false, prod, 4000);
    ord_case("decode, 1 row, head group 1, ratio 4", make(202, 8, 4, { { 1, 3000 } }), false, prod, 4000);
    ord_case("decode, 1 row, nvfp4 latent, 512 selections", make(203, 8, 4, { { 1, 6000 } }, 512), true, prod, 7000);
    ord_case("decode, 4 rows, 16 heads, head group 2, two sequences", make(204, 16, 2, { { 2, 2000 }, { 2, 3500 } }),
             false, prod, 4000);
    ord_case("decode, 8 rows, head group 2, nvfp4", make(205, 8, 1, { { 8, 3001 } }), true, prod, 4000);
    ord_case("decode, 16 rows, head group 4, ratio 2", make(206, 8, 2, { { 4, 2000 }, { 12, 3500 } }), false, prod, 4000);
    ord_case("decode, 32 rows, head group 8", make(207, 8, 1, { { 32, 3999 } }), false, prod, 4000);
    ord_case("decode, 40 rows, tiled PV, ratio 2", make(208, 8, 2, { { 8, 2600 }, { 32, 3999 } }), false, prod, 4000);
    ord_case("decode, 48 rows, tiled PV, nvfp4, rows past the table", make(209, 8, 1, { { 48, 3999 } }), true, prod, 3980);
    ord_case("decode, 3 rows, window only, 16 heads", make(210, 16, 0, { { 3, 1000 } }), false, prod, 2000);
    /* other geometries the epilogue takes */
    Rt fwd = prod;
    fwd.inverse = 0;
    ord_case("decode, 2 rows, forward rope", make(211, 8, 1, { { 2, 1500 } }), false, fwd, 2000);
    Rt fp4 = prod;
    fp4.fmt = "fp4_e2m1";
    ord_case("decode, 5 rows, fp4 e2m1 / ue8m0", make(212, 8, 1, { { 5, 1500 } }), false, fp4, 2000);
    Rt f32s = prod;
    f32s.scale = "f32";
    ord_case("decode, 6 rows, fp8 / f32 scale", make(213, 8, 1, { { 6, 1500 } }), false, f32s, 2000);
    Rt off0 = prod;
    off0.off = 0;
    off0.rot = 512;
    ord_case("decode, 2 rows, rot 512 at 0", make(214, 8, 1, { { 2, 1500 } }), false, off0, 2000);
    /* the passes the fold leaves to the two launches */
    Rt neox = prod;
    neox.mode = "neox";
    ord_case("decode, 4 rows, neox (two launches)", make(215, 8, 1, { { 4, 1500 } }), false, neox, 2000);
    Rt b16 = prod;
    b16.block = 16;
    ord_case("decode, 4 rows, blocks of 16 (two launches)", make(216, 8, 1, { { 4, 1500 } }), false, b16, 2000);
    ord_case("WMMA prefill, 64 rows (two launches)", make(217, 16, 1, { { 64, 2047 } }), false, prod, 3000);
    ord_case("WMMA prefill, 64 rows, ratio 2, nvfp4 (two launches)", make(218, 16, 2, { { 64, 3071 } }), true, prod, 4000);
    {
        const Fix f = make(219, 4, 1, { { 50, 2999 } });
        ord_case("scalar kernel, 50 rows (two launches)", f, false, prod, 3000, (int64_t)f.H * kD + 2);
    }
    std::printf("ds41 K1 folds: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
