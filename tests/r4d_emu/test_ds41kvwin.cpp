/* tests/r4d_emu: lever KVWIN (10.11.2026) -- DeepSeek-V4.1's window keys stored as the fp8 window slot
 * (r4d_ds41_qrt.h ds_w8_*: 512 e4m3 codes and 16 scale bytes, 528 bytes instead of 1024) -- on the host SIMT
 * emulator, byte for byte against the bf16 window it replaces:
 *
 *   packer   window_store (r4d_ds41_elem.hip) over every block the window's round trip can store: a block is
 *            c_i 2^K with c_i e4m3 values and one K in [-22, 120] (an inf only at K = 120, NaN where an input
 *            was one); all 126 positive codes as the block's largest at every K, with random smaller codes,
 *            subnormal codes, zeros of both signs, and the round trip's own outputs over hand-made rows (inf
 *            and NaN inputs, values at the power-of-two boundaries of the scale rule, bf16 subnormals). Every
 *            packed value must decode (e4m3(code) x 2^(e-127) in f32) to the stored bf16 bit for bit (a NaN to a
 *            NaN), and libref's window_store must write the same bytes;
 *   fold c   rmsnorm_rope_roundtrip_kv_store kv_dtype fp8_e4m3 against rmsnorm_rope_roundtrip + window_store: y
 *            and the whole (sentinel-filled) cache the same bytes, M 1..48, 64, 130, unmapped and past-the-cache
 *            slots, a NaN row;
 *   readers  attn_sparse_sink window_format fp8_e4m3 over the packed window against the same op over the bf16
 *            window, every output bit for bit, in every form the op takes: the decode band's direct-PV head
 *            groups 1/2/4/8 and the tiled-PV forms, the scalar kernel, WMMA prefill; bf16 and nvfp4 latents,
 *            window-only layers (window 128 and the drafter's 133, nk == window); attn_sparse_sink_rope_roundtrip
 *            (lever K1 fold d) the same; libref the same;
 *   mutants  a code or scale byte changed in a key a row reads changes its output (kernel and libref); the
 *            packer's scale rule moved by one either way fails the decode on the packer's blocks.
 */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_window_store(const RadArgs*, RadStream);
extern "C" int r4d_rad_rmsnorm_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_rmsnorm_rope_roundtrip_kv_store(const RadArgs*, RadStream);
extern "C" int r4d_rad_quant_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kSlots = 144;     /* the latent; its nvfp4 entry in bf16 slots */
constexpr int kW8B = 528, kW8 = 264;      /* the fp8 window slot: bytes, bf16 slots */

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
int launch(RadLaunchFn fn, RadTensor* t, int nt, const std::vector<RadParam>& p) {
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
float f32_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint32_t bits_f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

struct Ref {
    RadLaunchFn attn = nullptr, wstore = nullptr;
};
Ref load_libref() {
    Ref r;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return r;
    void* h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return r;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
    if (open && open() < 0) return r;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return r;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (!std::strcmp(k->op, "attn_sparse_sink")) r.attn = k->launch;
        if (!std::strcmp(k->op, "window_store")) r.wstore = k->launch;
    }
    return r;
}

/* ---- the slot, on the host: a packed value, and the packer with its scale rule moved by `kshift` (0: the
 * kernels' rule, for the mutants +1 / -1) */
float w8_val(const uint8_t* e, int d) { return rad_fp8e4m3_to_f32(e[d]) * f32_bits((uint32_t)e[kD + d / 32] << 23); }
void w8_pack_host(const uint16_t* row, uint8_t* e, int kshift) {
    std::memset(e, 0, kW8B);
    for (int b = 0; b < kD / 32; ++b) {
        float am = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float v = std::fabs(rad_bf16_to_f32(row[b * 32 + i]));
            if (v > am) am = v;
        }
        const uint32_t u = bits_f32(am);
        int k;
        if (u >= 0x7f800000u) k = 120;
        else if (u == 0u) k = 0;
        else k = std::max(u < 0x00800000u ? -126 : (int)(u >> 23) - 135 + ((u & 0x7fffffu) > 0x600000u ? 1 : 0), -126);
        if (u != 0u) k = std::min(std::max(k + kshift, -126), 127);
        const uint32_t sc = (uint32_t)(k + 127);
        for (int i = 0; i < 32; ++i)
            e[b * 32 + i] = rad_f32_to_fp8e4m3(rad_bf16_to_f32(row[b * 32 + i]) * f32_bits((254u - sc) << 23));
        e[kD + b] = (uint8_t)sc;
    }
}
/* How many values of rows [M, 512] a packed cache (slot m = row m) does not give back bit for bit (a NaN for
 * a NaN counts as given back). */
long long decode_misses(const std::vector<uint16_t>& rows, const uint8_t* packed, int64_t M) {
    long long d = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int i = 0; i < kD; ++i) {
            const uint16_t want = rows[(size_t)(m * kD + i)];
            const float got = w8_val(packed + m * kW8B, i);
            const bool wnan = (want & 0x7fffu) > 0x7f80u;
            if (wnan) d += got == got;
            else d += bits_f32(got) != ((uint32_t)want << 16);
        }
    return d;
}
int window_store(RadLaunchFn fn, const std::vector<uint16_t>& rows, int64_t M, const std::vector<int32_t>& slot,
                 int64_t nb, uint16_t* cache) {
    RadTensor t[3] = { tens(rows.data(), RAD_BF16, { M, kD }), tens(slot.data(), RAD_I32, { M }),
                       tens(cache, RAD_BF16, { nb, 1, 16, kW8 }) };
    return launch(fn, t, 3, { pint("M", M), pint("n", kD), pint("block_size", 16), pstr("format", "fp8_e4m3"),
                              pstr("dtype", "bf16") });
}
/* quant_roundtrip(fp8_e4m3, ue8m0, 32) over rows [M, 512] in place: the window key's round trip (the rule the
 * fused chain and the fold run, r4d_ds41_qrt.h). */
int roundtrip_rows(std::vector<uint16_t>& rows, int64_t M) {
    RadTensor t[1] = { tens(rows.data(), RAD_BF16, { M, kD }) };
    return launch(r4d_rad_quant_roundtrip, t, 1, { pint("M", M), pint("n", kD), pint("block", 32),
                                                    pstr("format", "fp8_e4m3"), pstr("scale", "ue8m0"),
                                                    pstr("dtype", "bf16") });
}

/* ============================================================ the packer */

void packer_case(const std::string& name, const std::vector<uint16_t>& rows, const Ref& ref, bool mutants) {
    const int64_t M = (int64_t)rows.size() / kD, nb = (M + 15) / 16;
    std::vector<int32_t> slot((size_t)M);
    for (int64_t m = 0; m < M; ++m) slot[(size_t)m] = (int32_t)m;
    std::vector<uint16_t> pk((size_t)(nb * 16 * kW8), 0x7777), pr((size_t)(nb * 16 * kW8), 0x6666);
    expect(window_store(r4d_rad_window_store, rows, M, slot, nb, pk.data()) == RAD_OK, name, "window_store");
    const long long miss = decode_misses(rows, (const uint8_t*)pk.data(), M);
    expect(miss == 0, name, "a packed value does not decode to the stored bf16", miss, M * kD);
    /* the host port is the kernel, byte for byte */
    std::vector<uint8_t> hp((size_t)(M * kW8B));
    for (int64_t m = 0; m < M; ++m) w8_pack_host(&rows[(size_t)(m * kD)], &hp[(size_t)(m * kW8B)], 0);
    const long long dh = std::memcmp(hp.data(), pk.data(), hp.size()) != 0;
    expect(dh == 0, name, "the host port of the packer differs from the kernel");
    long long dr = -1;
    if (ref.wstore) {
        expect(window_store(ref.wstore, rows, M, slot, nb, pr.data()) == RAD_OK, name, "libref window_store");
        dr = std::memcmp(pr.data(), pk.data(), (size_t)(M * kW8B)) != 0;
        expect(dr == 0, name, "libref's packed slots differ from the kernel's");
    }
    long long nan = 0, inf = 0, sub = 0, zero = 0;
    for (uint16_t v : rows) {
        nan += (v & 0x7fffu) > 0x7f80u;
        inf += (v & 0x7fffu) == 0x7f80u;
        zero += (v & 0x7fffu) == 0;
    }
    for (int64_t m = 0; m < M; ++m)
        for (int i = 0; i < kD; ++i) sub += (((const uint8_t*)pk.data())[m * kW8B + i] & 0x78u) == 0;
    long long mut_up = -1, mut_dn = -1;
    if (mutants) {
        for (int ks : { 1, -1 }) {
            std::vector<uint8_t> mp((size_t)(M * kW8B));
            for (int64_t m = 0; m < M; ++m) w8_pack_host(&rows[(size_t)(m * kD)], &mp[(size_t)(m * kW8B)], ks);
            (ks > 0 ? mut_up : mut_dn) = decode_misses(rows, mp.data(), M);
        }
        expect(mut_up > 0, name, "MUTANT: the scale rule moved up by one still decodes every value", mut_up, 0);
        expect(mut_dn > 0, name, "MUTANT: the scale rule moved down by one still decodes every value", mut_dn, 0);
    }
    std::printf("packer  %-58s %lld rows (%lld NaN, %lld inf, %lld zero values; %lld subnormal codes): decode misses "
                "%lld, host port %s, libref %s; mutants k+1 / k-1 miss %lld / %lld\n",
                name.c_str(), (long long)M, nan, inf, zero, sub, miss, dh ? "DIFFERS" : "same",
                dr < 0 ? "skipped" : dr ? "DIFFERS" : "same", mut_up, mut_dn);
}

/* Every block the round trip can store: K in [-22, 120], the block's largest code c (all 126 positive finite
 * e4m3 values, either sign), the other 31 random codes at most |c| (some subnormal, some zero, -0), a NaN
 * code in some blocks; at K = 120 the codes from 256 up are inf (the f32 product overflows). */
std::vector<uint16_t> every_block(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pct(0, 99);
    std::vector<uint16_t> rows;
    std::vector<uint16_t> blk(32);
    int nblk = 0;
    for (int K = -22; K <= 120; ++K)
        for (uint32_t cmax = 1; cmax < 0x7f; ++cmax) {
            const float cm = rad_fp8e4m3_to_f32((uint8_t)cmax);
            for (int i = 0; i < 32; ++i) {
                uint32_t c;
                const int k = pct(rng);
                if (i == (int)(cmax % 32)) c = cmax;
                else if (k < 10) c = 0;
                else if (k < 25) c = (uint32_t)(rng() % 8);        /* subnormal codes */
                else {
                    do c = (uint32_t)(rng() % 0x7f); while (rad_fp8e4m3_to_f32((uint8_t)c) > cm);
                }
                if (rng() & 1u) c |= 0x80u;
                if (K == 40 && cmax == 64 && i == 3) c = 0x7f;       /* a NaN code */
                const float v = rad_fp8e4m3_to_f32((uint8_t)c) * f32_bits((uint32_t)(K + 127) << 23);
                blk[(size_t)i] = rad_f32_to_bf16(v);
            }
            rows.insert(rows.end(), blk.begin(), blk.end());
            ++nblk;
        }
    while (nblk % 16) {   /* whole rows */
        rows.insert(rows.end(), 32, 0);
        ++nblk;
    }
    return rows;
}

/* Hand-made rows through the round trip itself: inf and NaN inputs, bf16 subnormals, values at the
 * power-of-two boundaries of the rule (448 x 2^j and its neighbours), bf16's largest, mixed scales. */
std::vector<uint16_t> roundtrip_rows_case(uint32_t seed, int64_t M) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    std::vector<uint16_t> rows((size_t)(M * kD));
    for (int64_t m = 0; m < M; ++m)
        for (int b = 0; b < kD / 32; ++b) {
            uint16_t* v = &rows[(size_t)(m * kD + b * 32)];
            const int k = pct(rng);
            const int e = (int)(rng() % 250) - 130;                    /* a block scale 2^e */
            for (int i = 0; i < 32; ++i) {
                float x = nd(rng) * std::ldexp(1.0f, std::clamp(e, -126, 120));
                if (k < 4) x = 0.0f;
                else if (k < 8) x = std::ldexp(nd(rng), -130 - (int)(rng() % 3));   /* bf16 subnormal */
                else if (k < 12) x = (rng() & 1 ? 1.0f : -1.0f) * 448.0f * std::ldexp(1.0f, (int)(rng() % 60) - 30) *
                                     (1.0f + (float)((int)(rng() % 3) - 1) * 0x1p-7f);
                else if (k < 14) x = (rng() & 1 ? 1.0f : -1.0f) * 3.38e38f * (1.0f - 0x1p-6f * (float)(rng() % 4));
                v[i] = rad_f32_to_bf16(x);
            }
            if (k >= 14 && k < 16) v[rng() % 32] = 0x7f80;                  /* an inf input: a NaN block */
            if (k >= 16 && k < 18) v[rng() % 32] = 0xffc1;                  /* a NaN input */
            if (k >= 18 && k < 20) v[rng() % 32] = 0x8000;                  /* -0 */
        }
    expect(roundtrip_rows(rows, M) == RAD_OK, "round trip rows", "quant_roundtrip");
    return rows;
}

/* ============================================================ fold c, kv_dtype fp8_e4m3 */

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

void fold_case(int64_t M, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    const int64_t n = kD, bs = 16, nb = M / bs + 3, tab = 300, rot = 64, off = 448;
    const std::string name = "fold c fp8 M=" + std::to_string(M);
    std::vector<uint16_t> x((size_t)(M * n));
    for (int64_t m = 0; m < M; ++m) {
        const float sc = m % 7 == 3 ? 1e-3f : m % 7 == 5 ? 40.0f : 1.0f;
        for (int64_t i = 0; i < n; ++i) x[(size_t)(m * n + i)] = rad_f32_to_bf16(nd(rng) * sc);
    }
    if (M > 3) x[(size_t)(2 * n + 17)] = 0x7fc1;
    /* gains a block apart in scale, so the window's blocks are too (2^-60 .. 2^60) */
    std::vector<float> wf((size_t)n);
    for (int64_t i = 0; i < n; ++i) wf[(size_t)i] = (0.5f + 0.1f * nd(rng)) * std::ldexp(1.0f, (int)((i / 32) * 8) - 60);
    std::vector<float> cs = angles(rng, tab, rot);
    std::vector<int32_t> pos((size_t)M);
    for (int64_t m = 0; m < M; ++m) {
        const int k = pct(rng);
        pos[(size_t)m] = k < 6 ? (int32_t)(tab + k) : k < 9 ? -1 : (int32_t)(rng() % tab);
    }
    std::vector<int32_t> perm((size_t)(nb * bs));
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> slot((size_t)M);
    for (int64_t m = 0; m < M; ++m) slot[(size_t)m] = pct(rng) < 10 ? -1 : perm[(size_t)m];
    if (M > 4) slot[(size_t)(M - 2)] = (int32_t)(nb * bs + 3);
    std::vector<RadParam> p = { pint("M", M),        pint("n", n),        pf64("eps", 1e-6),   pint("n_head", 1),
                                pint("head_dim", n), pint("rot", rot),    pint("off", off),    pstr("mode", "gptj"),
                                pint("inverse", 0),  pint("block", 32),   pstr("format", "fp8_e4m3"),
                                pstr("scale", "ue8m0"), pstr("dtype", "bf16") };
    /* the chain: rmsnorm_rope_roundtrip, then window_store */
    std::vector<uint16_t> y1((size_t)(M * n), 0x1234), kv1((size_t)(nb * bs * kW8), 0x5a5a);
    {
        RadTensor t[5] = { tens(x.data(), RAD_BF16, { M, n }), tens(wf.data(), RAD_F32, { n }),
                           tens(y1.data(), RAD_BF16, { M, n }), tens(cs.data(), RAD_F32, { tab, rot }),
                           tens(pos.data(), RAD_I32, { M }) };
        expect(launch(r4d_rad_rmsnorm_rope_roundtrip, t, 5, p) == RAD_OK, name, "rmsnorm_rope_roundtrip");
        expect(window_store(r4d_rad_window_store, y1, M, slot, nb, kv1.data()) == RAD_OK, name, "window_store");
    }
    /* the fold */
    std::vector<uint16_t> y2((size_t)(M * n), 0x4321), kv2((size_t)(nb * bs * kW8), 0x5a5a);
    {
        std::vector<RadParam> pk = p;
        pk.push_back(pint("block_size", bs));
        pk.push_back(pstr("kv_dtype", "fp8_e4m3"));
        RadTensor t[7] = { tens(x.data(), RAD_BF16, { M, n }), tens(wf.data(), RAD_F32, { n }),
                           tens(y2.data(), RAD_BF16, { M, n }), tens(cs.data(), RAD_F32, { tab, rot }),
                           tens(pos.data(), RAD_I32, { M }), tens(slot.data(), RAD_I32, { M }),
                           tens(kv2.data(), RAD_BF16, { nb, 1, bs, kW8 }) };
        expect(launch(r4d_rad_rmsnorm_rope_roundtrip_kv_store, t, 7, pk) == RAD_OK, name,
               "rmsnorm_rope_roundtrip_kv_store fp8_e4m3");
    }
    const long long dy = diff(y1, y2), dk = diff(kv1, kv2);
    long long stored = 0;
    for (int64_t m = 0; m < M; ++m) stored += slot[(size_t)m] >= 0 && slot[(size_t)m] < nb * bs;
    /* the stored slots decode to y */
    long long miss = 0;
    for (int64_t m = 0; m < M; ++m) {
        const int32_t s = slot[(size_t)m];
        if (s < 0 || s >= nb * bs) continue;
        std::vector<uint16_t> row(y1.begin() + m * n, y1.begin() + (m + 1) * n);
        miss += decode_misses(row, (const uint8_t*)kv2.data() + (size_t)s * kW8B, 1);
    }
    expect(dy == 0, name, "y differs from rmsnorm_rope_roundtrip's", dy, (long long)y1.size());
    expect(dk == 0, name, "the cache differs from window_store's", dk, (long long)kv1.size());
    expect(miss == 0, name, "a stored slot does not decode to y", miss, stored * kD);
    expect(stored > 0, name, "no slot was stored", stored, 0);
    std::printf("fold c  %-24s y differ %lld of %lld, cache differ %lld of %lld (%lld rows stored, decode misses %lld)\n",
                name.c_str(), dy, (long long)y1.size(), dk, (long long)kv1.size(), stored, miss);
}

/* ============================================================ the readers (test_ds41attn.cpp's fixture) */

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
    std::vector<uint16_t> q, wc, wp, cc;
    std::vector<int32_t> wbt, wused, cbt, sel, pos, cu;
    std::vector<float> sink;
};

/* The window keys: the round trip's output over rows whose blocks span magnitudes (2^-24 .. 2^4: below the 1e-4
 * floor's subnormal codes up to the scores' useful range, so no key swamps the softmax; some zero), every slot of
 * every window block packed by window_store into wp. */
Fix make(uint32_t seed, int H, int ratio, const std::vector<std::pair<int, int>>& seqs, int topk = 64, int win = 128) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    Fix f;
    f.H = H;
    f.win = win;
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
    const int64_t nslot = f.nwblk * f.bs;
    f.wc.resize((size_t)(nslot * kD));
    for (int64_t r = 0; r < nslot; ++r)
        for (int b = 0; b < kD / 32; ++b) {
            const int k = pct(rng);
            const float sc = k < 6 ? 1e-6f : k < 10 ? 0.0f : k < 14 ? 0x1p-24f : std::ldexp(1.0f, (int)(rng() % 17) - 12);
            for (int i = 0; i < 32; ++i) f.wc[(size_t)(r * kD + b * 32 + i)] = bf(sc);
        }
    expect(roundtrip_rows(f.wc, nslot) == RAD_OK, "fixture", "the window keys' round trip");
    f.wp.assign((size_t)(nslot * kW8), 0x2468);
    {
        std::vector<int32_t> slot((size_t)nslot);
        for (int64_t i = 0; i < nslot; ++i) slot[(size_t)i] = (int32_t)i;
        expect(window_store(r4d_rad_window_store, f.wc, nslot, slot, f.nwblk, f.wp.data()) == RAD_OK, "fixture",
               "window_store");
        expect(decode_misses(f.wc, (const uint8_t*)f.wp.data(), nslot) == 0, "fixture", "a window key does not decode");
    }
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

/* attn_sparse_sink (rt null) or attn_sparse_sink_rope_roundtrip over the bf16 window (w8 false) or the packed
 * one (window_format fp8_e4m3, or `wfmt` named explicitly), the bf16 or nvfp4 latent. */
int attn(RadLaunchFn fn, const Fix& f, bool w8, const uint16_t* wp, const uint16_t* nv, uint16_t* out, int64_t q_pitch,
         const uint16_t* q, bool rt, const std::vector<float>* cs, int64_t tab, const char* wfmt = nullptr) {
    RadTensor t[12] = {
        tens(q, RAD_BF16, { f.M, (int64_t)f.H * kD }, q_pitch),
        w8 ? tens(wp, RAD_BF16, { f.nwblk, 1, f.bs, kW8 }) : tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { f.S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { f.S }),
        f.ratio == 0 ? none()
        : nv         ? tens(nv, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
                     : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        f.ratio == 0 ? none() : tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        f.ratio == 0 ? none() : tens(f.sel.data(), RAD_I32, { f.M, f.topk }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out, RAD_BF16, { f.M, (int64_t)f.H * kD }),
        rt ? tens(cs->data(), RAD_F32, { tab, 64 }) : none(),
    };
    std::vector<RadParam> p = { pint("M", f.M),          pint("heads", f.H),     pint("dim", kD),
                                pint("window", f.win),   pint("ratio", f.ratio), pint("topk", f.topk),
                                pf64("scale", 1.0 / std::sqrt((double)kD)), pstr("dtype", "bf16") };
    if (nv) p.push_back(pstr("latent_format", "nvfp4"));
    if (w8) p.push_back(pstr("window_format", "fp8_e4m3"));
    else if (wfmt) p.push_back(pstr("window_format", wfmt));
    if (rt) {
        p.push_back(pint("rot", 64));
        p.push_back(pint("off", 448));
        p.push_back(pstr("mode", "gptj"));
        p.push_back(pint("inverse", 1));
        p.push_back(pint("block", 32));
        p.push_back(pstr("format", "fp8_e4m3"));
        p.push_back(pstr("block_scale", "ue8m0"));
    }
    return launch(fn, t, rt ? 12 : 11, p);
}

/* A slot row 0 reads: its newest window key (position pos[0] in its sequence 0). */
int64_t slot_of_row0(const Fix& f) {
    const int64_t last = f.pos[(size_t)(f.cu[1] - 1)], first = last + 1 - f.wused[0];
    const int64_t r = f.pos[0] - first;
    return (int64_t)f.wbt[(size_t)(r / f.bs)] * f.bs + r % f.bs;
}

void read_case(const std::string& name, const Fix& f, bool nvlat, int64_t q_pitch, const Ref& ref, bool rt = false,
               bool mutants = false) {
    std::vector<uint16_t> nv;
    if (nvlat) {
        nv.assign((size_t)(f.ncblk * kSlots), 0x7777);
        expect(store_nv4(f, nv.data()) == RAD_OK, name, "entry_store nvfp4");
    }
    const uint16_t* nvp = nvlat ? nv.data() : nullptr;
    const int64_t qp = q_pitch ? q_pitch : (int64_t)f.H * kD;
    std::vector<uint16_t> qv((size_t)(f.M * qp), 0);
    for (int m = 0; m < f.M; ++m) std::memcpy(&qv[(size_t)(m * qp)], &f.q[(size_t)m * f.H * kD], (size_t)f.H * kD * 2);
    std::mt19937 rng((uint32_t)f.M * 7919u + 5u);
    const int64_t tab = 16384;
    const std::vector<float> cs = rt ? angles(rng, tab, 64) : std::vector<float>();
    const RadLaunchFn fn = rt ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip : r4d_rad_ds41_attn_sparse_sink;
    const size_t no = (size_t)f.M * f.H * kD;
    std::vector<uint16_t> o1(no, 0x5555), o2(no, 0x3333), o3(no, 0x1717);
    expect(attn(fn, f, false, nullptr, nvp, o1.data(), q_pitch, qv.data(), rt, &cs, tab) == RAD_OK, name, "attn bf16 window");
    expect(attn(fn, f, true, f.wp.data(), nvp, o2.data(), q_pitch, qv.data(), rt, &cs, tab) == RAD_OK, name,
           "attn fp8 window");
    expect(attn(fn, f, false, nullptr, nvp, o3.data(), q_pitch, qv.data(), rt, &cs, tab, "bf16") == RAD_OK, name,
           "attn window_format bf16 named");
    const long long dk = diff(o1, o2), dn = diff(o1, o3);
    long long nz = 0;
    for (size_t i = 0; i < no; ++i) nz += (o1[i] & 0x7fffu) != 0;
    expect(dk == 0, name, "the fp8 window's output differs from the bf16 window's", dk, (long long)no);
    expect(dn == 0, name, "window_format bf16 named differs from it unnamed", dn, (long long)no);
    expect(nz > (long long)no / 2, name, "the output is mostly zero (the fixture did not reach the keys)", nz, (long long)no);
    long long dr = -1, dkr = -1;
    if (ref.attn && !rt) {
        std::vector<uint16_t> r1(no, 0x1111), r2(no, 0x2222);
        expect(attn(ref.attn, f, false, nullptr, nvp, r1.data(), q_pitch, qv.data(), false, nullptr, 0) == RAD_OK, name,
               "libref attn bf16 window");
        expect(attn(ref.attn, f, true, f.wp.data(), nvp, r2.data(), q_pitch, qv.data(), false, nullptr, 0) == RAD_OK, name,
               "libref attn fp8 window");
        dr = diff(r1, r2);
        dkr = diff(r1, o1);
        expect(dr == 0, name, "libref: the fp8 window's output differs from the bf16 window's", dr, (long long)no);
    }
    long long mc = -1, ms = -1, mrc = -1;
    if (mutants) {   /* a code byte, then a scale byte, of the key row 0 reads last: its output must move */
        const int64_t s0 = slot_of_row0(f);
        std::vector<uint16_t> wm = f.wp;
        uint8_t* b = (uint8_t*)wm.data() + s0 * kW8B;
        int d = 0;   /* the key's largest value (its code times its block's scale): changing it moves the output */
        for (int i = 1; i < kD; ++i)
            if (std::fabs(w8_val(b, i)) > std::fabs(w8_val(b, d))) d = i;
        b[d] ^= 0x01u;
        Fix fm = f;
        fm.wp = wm;
        std::vector<uint16_t> om(no, 0x4444);
        expect(attn(fn, fm, true, fm.wp.data(), nvp, om.data(), q_pitch, qv.data(), rt, &cs, tab) == RAD_OK, name,
               "attn, a code byte changed");
        mc = diff(o2, om);
        expect(mc > 0, name, "MUTANT: a changed code byte left the output as it was", mc, 0);
        if (ref.attn && !rt) {
            std::vector<uint16_t> rm(no, 0x4444), r2(no, 0x2222);
            attn(ref.attn, f, true, f.wp.data(), nvp, r2.data(), q_pitch, qv.data(), false, nullptr, 0);
            attn(ref.attn, fm, true, fm.wp.data(), nvp, rm.data(), q_pitch, qv.data(), false, nullptr, 0);
            mrc = diff(r2, rm);
            expect(mrc > 0, name, "MUTANT: libref, a changed code byte left the output as it was", mrc, 0);
        }
        b[d] ^= 0x01u;
        b[kD + d / 32] = (uint8_t)(b[kD + d / 32] + 1);
        fm.wp = wm;
        expect(attn(fn, fm, true, fm.wp.data(), nvp, om.data(), q_pitch, qv.data(), rt, &cs, tab) == RAD_OK, name,
               "attn, a scale byte changed");
        ms = diff(o2, om);
        expect(ms > 0, name, "MUTANT: a changed scale byte left the output as it was", ms, 0);
    }
    std::printf("reader  %-62s rows %d heads %d ratio %d keys %d+%d %s%s%s: fp8 vs bf16 window %lld (kernel), %lld "
                "(libref), named bf16 %lld; kernel vs libref (info) %lld of %lld; mutants code / scale / libref code %lld / "
                "%lld / %lld\n",
                name.c_str(), f.M, f.H, f.ratio, f.win, f.topk, nvlat ? "nvfp4" : "bf16", q_pitch ? " pitched q" : "",
                rt ? " fold d" : "", dk, dr, dn, dkr, (long long)no, mc, ms, mrc);
}

}  // namespace

int main() {
    const Ref ref = load_libref();
    if (!ref.attn || !ref.wstore) std::printf("note: libref not loadable (set RADIANCE_HOME): its checks are skipped\n");

    /* ---- the packer */
    packer_case("every block the round trip can store (K -22..120 x 126 codes)", every_block(7), ref, true);
    packer_case("the round trip over hand-made rows (inf, NaN, boundaries)", roundtrip_rows_case(11, 96), ref, true);

    /* ---- fold c with the packed cache */
    uint32_t seed = 300;
    for (int64_t M = 1; M <= 48; ++M) fold_case(M, seed++);
    for (int64_t M : { 64, 130 }) fold_case(M, seed++);

    /* ---- the readers, every form */
    read_case("decode, 1 row, direct PV, head group 1", make(31, 8, 1, { { 1, 3000 } }), false, 0, ref, false, true);
    read_case("decode, 1 row, direct PV, nvfp4, 512 selections", make(39, 8, 4, { { 1, 6000 } }, 512), true, 0, ref);
    read_case("decode, 8 rows, direct PV, head group 2", make(36, 8, 1, { { 8, 3001 } }), false, 0, ref);
    read_case("decode, 8 rows, direct PV, head group 2, nvfp4", make(43, 8, 1, { { 8, 6000 } }, 1024), true, 0, ref);
    read_case("decode, 16 rows, direct PV, head group 4, ratio 2", make(37, 8, 2, { { 4, 2000 }, { 12, 3500 } }), false, 0,
              ref);
    read_case("decode, 16 rows, head group 4, nvfp4, ratio 2", make(44, 8, 2, { { 16, 9000 } }, 2048), true, 0, ref);
    read_case("decode, 32 rows, direct PV, head group 8", make(38, 8, 1, { { 32, 3999 } }), false, 0, ref);
    read_case("decode, 32 rows, head group 8, nvfp4", make(45, 8, 1, { { 32, 6000 } }, 1024), true, 0, ref);
    read_case("decode, 40 rows, tiled PV, ratio 2", make(32, 8, 2, { { 8, 2600 }, { 32, 3999 } }), false, 0, ref, false,
              true);
    read_case("decode, 48 rows, tiled PV, nvfp4", make(46, 8, 1, { { 48, 3999 } }), true, 0, ref);
    read_case("decode, 3 rows, window only (window 128)", make(47, 16, 0, { { 3, 1000 } }), false, 0, ref);
    read_case("decode, 40 rows, window only, drafter window 133 (nk == window)", make(48, 8, 0, { { 5, 700 }, { 5, 900 },
              { 5, 1100 }, { 5, 1300 }, { 5, 1500 }, { 5, 1700 }, { 5, 1900 }, { 5, 2100 } }, 0, 133), false, 0, ref);
    read_case("decode, 1 row, window only, window 133", make(49, 8, 0, { { 1, 500 } }, 0, 133), false, 0, ref);
    {
        const Fix f = make(33, 4, 1, { { 50, 2999 } });
        read_case("scalar kernel, 50 rows", f, false, (int64_t)f.H * kD + 2, ref, false, true);
        const Fix g = make(50, 4, 2, { { 50, 2999 } });
        read_case("scalar kernel, 50 rows, nvfp4, ratio 2", g, true, (int64_t)g.H * kD + 2, ref);
        const Fix h = make(51, 4, 0, { { 50, 999 } }, 0, 133);
        read_case("scalar kernel, 50 rows, window only 133", h, false, (int64_t)h.H * kD + 2, ref);
    }
    read_case("WMMA, 64 rows", make(34, 16, 1, { { 64, 2047 } }), false, 0, ref, false, true);
    read_case("WMMA, 64 rows, ratio 2, nvfp4", make(35, 16, 2, { { 64, 3071 } }), true, 0, ref);
    read_case("WMMA, 64 rows, window only 133", make(52, 16, 0, { { 64, 1500 } }, 0, 133), false, 0, ref);
    /* lever K1 fold d over the packed window: the decode band's fused epilogue, and its two-launch passes */
    read_case("fold d, decode, 1 row, head group 1", make(201, 8, 4, { { 1, 3000 } }), false, 0, ref, true, true);
    read_case("fold d, decode, 8 rows, head group 2, nvfp4", make(205, 8, 1, { { 8, 3001 } }), true, 0, ref, true);
    read_case("fold d, decode, 32 rows, head group 8", make(207, 8, 1, { { 32, 3999 } }), false, 0, ref, true);
    read_case("fold d, decode, 40 rows, tiled PV, ratio 2", make(208, 8, 2, { { 8, 2600 }, { 32, 3999 } }), false, 0, ref,
              true);
    read_case("fold d, decode, 3 rows, window only", make(210, 16, 0, { { 3, 1000 } }), false, 0, ref, true);
    read_case("fold d, WMMA 64 rows (two launches)", make(217, 16, 1, { { 64, 2047 } }), false, 0, ref, true);
    std::printf("ds41 KVWIN fp8 window: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
