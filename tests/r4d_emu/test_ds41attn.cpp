/* tests/r4d_emu: DeepSeek-V4.1's packed compressed latent (r4d_ds41_sparse.hip) on the host SIMT
 * emulator.
 *
 * The latent DeepSeek-V4 stores is quant_roundtrip(fp4_e2m1, e4m3, block 16) output (a host port of
 * libref's rule here, with blocks small enough that the e4m3 scale is subnormal). entry_store format
 * nvfp4 packs it (the kernel; libref's when loadable, byte for byte the kernel's), every packed entry
 * must unpack to the stored bf16 bits, and attn_sparse_sink with latent_format nvfp4 over the packed
 * cache must give the bf16 cache's output bit for bit in every form the op takes: the decode band's
 * kernels (1-32 rows: the direct-PV forms, head groups 1-8; 40 rows: the tiled-PV form), the scalar kernel (a pass whose
 * q rows the WMMA form refuses) and the WMMA prefill form (64 rows); and libref the same. */
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

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kSlots = 144;   /* the latent; its packed entry in bf16 slots (288 bytes) */

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

int g_fail = 0, g_check = 0;
void expect(bool ok, const char* name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name, what, a, b);
}

struct Ref {
    RadLaunchFn attn = nullptr, store = nullptr;
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
        if (!std::strcmp(k->op, "entry_store")) r.store = k->launch;
    }
    return r;
}

/* quant_roundtrip(fp4_e2m1, e4m3, 16) on one block of 16, in place (libref's rule). */
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

/* A step of one or two sequences: window caches, the compressed latent caches (bf16 and packed) at
 * `ratio`, selections, sinks. */
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
    f.topk = topk;
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
    /* window: every position a row may hold, blocks of bs, shuffled */
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
    for (const auto& sq : seqs) f.wused.push_back((int32_t)(sq.second + 1));   /* the table from position 0 */
    /* compressed: entry c of a sequence in block cbt[s][c] (block size ratio, one entry a block) */
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
    for (int64_t e = 0; e < f.ncblk; ++e) {
        /* a latent row: most blocks O(1), some tiny (subnormal e4m3 scales), one all zero */
        for (int b = 0; b < kD / 16; ++b) {
            const int k = pct(rng);
            const float sc = k < 8 ? 1e-3f : k < 16 ? 0.02f : k < 18 ? 0.0f : 1.0f;
            uint16_t* v = &f.cc[(size_t)(e * kD + b * 16)];
            for (int i = 0; i < 16; ++i) v[i] = bf(sc);
            roundtrip16(v);
        }
    }
    f.sel.resize((size_t)f.M * f.topk);
    for (int m = 0; m < f.M; ++m) {
        const int64_t reach = ((int64_t)f.pos[(size_t)m] + 1) / ratio;
        std::uniform_int_distribution<int64_t> any(0, std::max<int64_t>(0, reach - 1));
        for (int t = 0; t < f.topk; ++t)
            f.sel[(size_t)(m * f.topk + t)] = (reach > 0 && pct(rng) < 95) ? (int32_t)any(rng) : -1;
    }
    f.sink.resize((size_t)H);
    for (auto& x : f.sink) x = nd(rng);
    return f;
}

int store_nv4(RadLaunchFn fn, const Fix& f, uint16_t* packed) {
    std::vector<int32_t> page((size_t)f.ncblk);
    for (int64_t i = 0; i < f.ncblk; ++i) page[(size_t)i] = (int32_t)i;
    RadTensor t[3] = { tens(f.cc.data(), RAD_BF16, { f.ncblk, kD }), tens(page.data(), RAD_I32, { f.ncblk }),
                       tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }) };
    RadParam p[5] = { pint("M", f.ncblk), pint("n", kD), pint("block_size", f.ratio), pstr("dtype", "bf16"),
                      pstr("format", "nvfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 5;
    return fn(&a, nullptr);
}

/* attn_sparse_sink over the bf16 cache (packed null) or the packed one; q_pitch > H D gives q rows a
 * pitch the WMMA form refuses (the scalar kernel then). */
int attn(RadLaunchFn fn, const Fix& f, const uint16_t* packed, uint16_t* out, int64_t q_pitch, const uint16_t* q) {
    RadTensor t[11] = {
        tens(q, RAD_BF16, { f.M, (int64_t)f.H * kD }, q_pitch),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { f.S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { f.S }),
        packed ? tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
               : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        tens(f.sel.data(), RAD_I32, { f.M, f.topk }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out, RAD_BF16, { f.M, (int64_t)f.H * kD }),
    };
    std::vector<RadParam> p = { pint("M", f.M), pint("heads", f.H), pint("dim", kD), pint("window", f.win),
                                pint("ratio", f.ratio), pint("topk", f.topk), pf64("scale", 1.0 / std::sqrt((double)kD)),
                                pstr("dtype", "bf16") };
    if (packed) p.push_back(pstr("latent_format", "nvfp4"));
    RadArgs a{};
    a.t = t;
    a.n_t = 11;
    a.p = p.data();
    a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

void run_case(const char* name, const Fix& f, int64_t q_pitch, const Ref& ref) {
    std::vector<uint16_t> pk((size_t)(f.ncblk * kSlots), 0x7777), pr((size_t)(f.ncblk * kSlots), 0x6666);
    expect(store_nv4(r4d_ds41_entry_store_nv4, f, pk.data()) == RAD_OK, name, "entry_store nvfp4");
    int64_t dec = 0, pad = 0;
    for (int64_t e = 0; e < f.ncblk; ++e) {
        const uint8_t* b = (const uint8_t*)&pk[(size_t)(e * kSlots)];
        for (int d = 0; d < kD; ++d) {
            const unsigned c = (b[d / 2] >> (4 * (d & 1))) & 15u;
            const float mag = std::fabs((float)((c & 7u) < 4u ? 0.5f * (float)(c & 7u)
                                                               : (float)(2u << (((c & 7u) - 4u) >> 1)) * ((c & 1u) ? 1.5f : 1.0f)));
            const float v = mag * rad_fp8e4m3_to_f32(b[kD / 2 + d / 16]);
            uint32_t u;
            std::memcpy(&u, &v, 4);
            dec += (uint16_t)((u >> 16) | ((c & 8u) << 12)) != f.cc[(size_t)(e * kD + d)];
        }
        for (int z = kD / 2 + kD / 16; z < 2 * kSlots; ++z) pad += b[z] != 0;
    }
    expect(dec == 0, name, "a packed entry does not decode to the stored bf16 bits", dec, f.ncblk * kD);
    expect(pad == 0, name, "the pad bytes are not zero", pad, 0);
    if (ref.store) {
        expect(store_nv4(ref.store, f, pr.data()) == RAD_OK, name, "libref entry_store nvfp4");
        int64_t d = 0;
        for (size_t i = 0; i < pk.size(); ++i) d += pk[i] != pr[i];
        expect(d == 0, name, "libref's packed cache differs from the kernel's", d, (long long)pk.size());
    }
    /* q with the pitch asked for */
    const int64_t qp = q_pitch ? q_pitch : (int64_t)f.H * kD;
    std::vector<uint16_t> qv((size_t)(f.M * qp), 0);
    for (int m = 0; m < f.M; ++m)
        std::memcpy(&qv[(size_t)(m * qp)], &f.q[(size_t)m * f.H * kD], (size_t)f.H * kD * 2);
    const size_t no = (size_t)f.M * f.H * kD;
    std::vector<uint16_t> o1(no, 0x5555), o2(no, 0x3333);
    expect(attn(r4d_rad_ds41_attn_sparse_sink, f, nullptr, o1.data(), q_pitch, qv.data()) == RAD_OK, name, "attn bf16");
    expect(attn(r4d_rad_ds41_attn_sparse_sink, f, pk.data(), o2.data(), q_pitch, qv.data()) == RAD_OK, name, "attn nvfp4");
    int64_t dk = 0, nz = 0;
    for (size_t i = 0; i < no; ++i) {
        dk += o1[i] != o2[i];
        nz += (o1[i] & 0x7fffu) != 0;
    }
    expect(dk == 0, name, "the packed latent's output differs from the bf16 cache's", dk, (long long)no);
    expect(nz > (int64_t)no / 2, name, "the output is mostly zero (the fixture did not reach the keys)", nz, (long long)no);
    int64_t dr = -1, dkr = -1;
    if (ref.attn) {
        std::vector<uint16_t> r1(no, 0x1111), r2(no, 0x2222);
        expect(attn(ref.attn, f, nullptr, r1.data(), q_pitch, qv.data()) == RAD_OK, name, "libref attn bf16");
        expect(attn(ref.attn, f, pk.data(), r2.data(), q_pitch, qv.data()) == RAD_OK, name, "libref attn nvfp4");
        dr = 0;
        dkr = 0;
        for (size_t i = 0; i < no; ++i) {
            dr += r1[i] != r2[i];
            dkr += r1[i] != o1[i];
        }
        expect(dr == 0, name, "libref: the packed latent's output differs from the bf16 cache's", dr, (long long)no);
    }
    std::printf("%-46s rows %d heads %d ratio %d keys %d+%d: %lld entries packed (288 B each), outputs differ "
                "packed vs bf16: %lld (kernel), %lld (libref); kernel vs libref (bf16, info) %lld of %lld\n",
                name, f.M, f.H, f.ratio, f.win, f.topk, (long long)f.ncblk, (long long)dk, (long long)dr,
                (long long)dkr, (long long)no);
}

}  // namespace

int main() {
    const Ref ref = load_libref();
    if (!ref.attn || !ref.store) std::printf("note: libref not loadable (set RADIANCE_HOME): its checks are skipped\n");
    /* decode band: one row (8 heads: the direct-PV form) */
    run_case("decode, 1 row, direct PV", make(31, 8, 1, { { 1, 3000 } }), 0, ref);
    /* decode band, the other direct-PV forms (8 heads: 64, 128 and 256 row-heads -> head groups of
     * 2, 4 and 8; the packed PV ring in 16-key slots and its fences, 7.11.), and one row at the
     * model's 512 selections (16 packed blocks) */
    run_case("decode, 8 rows, direct PV, head group 2", make(36, 8, 1, { { 8, 3001 } }), 0, ref);
    run_case("decode, 16 rows, direct PV, head group 4, ratio 2", make(37, 8, 2, { { 4, 2000 }, { 12, 3500 } }), 0, ref);
    run_case("decode, 32 rows, direct PV, head group 8", make(38, 8, 1, { { 32, 3999 } }), 0, ref);
    run_case("decode, 1 row, direct PV, 512 selections", make(39, 8, 4, { { 1, 6000 } }, 512), 0, ref);
    /* 8.11. (v3): the packed direct PV a 32-key block a step (one register a key) and its LDS decode tables (one head a
     * PV thread: head groups 1, 2 and 4 at 1024 threads; 8 keeps the register decode), at 1024 and 2048 selections --
     * odd and even block counts, the head group 4 form at 2048 near the LDS bound */
    run_case("decode, 1 row, direct PV, 1024 selections, ratio 2", make(40, 8, 2, { { 1, 9000 } }, 1024), 0, ref);
    run_case("decode, 1 row, direct PV, 2048 selections", make(41, 8, 1, { { 1, 9000 } }, 2048), 0, ref);
    run_case("decode, 2 rows, 16 heads, direct PV, 2048 selections, ratio 2", make(42, 16, 2, { { 2, 12000 } }, 2048), 0, ref);
    run_case("decode, 8 rows, head group 2, 1024 selections", make(43, 8, 1, { { 8, 6000 } }, 1024), 0, ref);
    run_case("decode, 16 rows, head group 4, 2048 selections, ratio 2", make(44, 8, 2, { { 16, 9000 } }, 2048), 0, ref);
    run_case("decode, 32 rows, head group 8, 1024 selections", make(45, 8, 1, { { 32, 6000 } }, 1024), 0, ref);
    run_case("decode, 1 row, direct PV, 480 selections (15 packed blocks + window)", make(46, 8, 1, { { 1, 5000 } }, 480), 0, ref);
    /* decode band: 40 rows of two sequences, 8 heads (320 row-heads: the tiled-PV form), ratio 2 */
    run_case("decode, 40 rows, tiled PV, ratio 2", make(32, 8, 2, { { 8, 2600 }, { 32, 3999 } }), 0, ref);
    /* the scalar kernel: 50 rows whose q pitch (H D + 2) the WMMA form refuses */
    {
        const Fix f = make(33, 4, 1, { { 50, 2999 } });
        run_case("scalar kernel, 50 rows", f, (int64_t)f.H * kD + 2, ref);
    }
    /* WMMA prefill: 64 rows, 16 heads, ratio 1 and 2 */
    run_case("WMMA, 64 rows", make(34, 16, 1, { { 64, 2047 } }), 0, ref);
    run_case("WMMA, 64 rows, ratio 2", make(35, 16, 2, { { 64, 3071 } }), 0, ref);
    std::printf("ds41 packed latent: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
