/* tests/r4d_emu: lever DKV (14.11.2026) -- the packed compressed latent (RADIANCE_DS41_LATENT=nvfp4) read without
 * unpacking it in every attention, on the host SIMT emulator (r4d_ds41_sparse.hip).
 *
 * DKV-U: latent_unpack lays a pass's entries (the plan: each sequence's [n_s - t_s, n_s), n_s = (last position + 1) /
 *   ratio, the sequences taking the operand's rows in cu order) as bf16 rows; attn_sparse_sink_unpacked's WMMA form
 *   reads a 16-key tile the plan holds from those rows and unpacks any other tile itself (the v3 decode). Checked:
 *   the rows are the stored bf16 bits (every scale byte x every code, plus the fixture's entries), unmapped entries'
 *   rows are not written, and the attention equals the bf16 cache's attention bit for bit with the plan holding all,
 *   part (tiles mixed: some read the rows, some unpack) or none of the entries, over one and two sequences, ratio 1
 *   and 2, a PSK-style cut (the last rows of the unpacked segment, cu still the uncut one's), and the fold d form
 *   (attn_sparse_sink_rope_roundtrip_unpacked) against the fold over the bf16 cache. Mutations: one changed value in a
 *   held row changes the output (the rows are read); a held row of another cache's entries changes it.
 * DKV-F: RADIANCE_DS41_NVFAST=1 (set here before the first launch) -- attn_sparse_sink's own WMMA form decodes its
 *   packed tiles by the v3 decode: the bf16 cache's output bit for bit.
 * DKV-G: latent_gather writes the decode pass's valid selections (NVG's gather kernel as an op) and the gathered
 *   attentions (plain and fold d) over them equal the bf16 cache's, at 1..40 rows, one and two sequences; one changed
 *   gathered value changes the output; an invalid selection's row is not written. */
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

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_latent_unpack(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_latent_gather(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_unpacked(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_unpacked(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_gathered(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kSlots = 144, kRot = 64;

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

/* quant_roundtrip(fp4_e2m1, e4m3, 16) on one block of 16, in place (test_ds41attn's port of libref's rule). */
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
/* the stored bf16 of code c at scale byte sb (test_ds41attn's host decode) */
uint16_t host_val(unsigned c, unsigned sb) {
    const float mag = std::fabs((float)((c & 7u) < 4u ? 0.5f * (float)(c & 7u)
                                                       : (float)(2u << (((c & 7u) - 4u) >> 1)) * ((c & 1u) ? 1.5f : 1.0f)));
    const float v = mag * rad_fp8e4m3_to_f32(sb);
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t)((u >> 16) | ((c & 8u) << 12));
}

/* test_ds41attn's fixture: sequences (rows, last position), window caches, the compressed latent (bf16 and packed) at
 * `ratio`, selections in ascending order (index_topk's) with -1 after, sinks, and a cos_sin table for the fold. */
struct Fix {
    int H = 0, ratio = 1, win = 128, topk = 64, S = 0, M = 0;
    int64_t nwblk = 0, ncblk = 0, nwbt = 0, ncbt = 0, maxp = 0;
    const int bs = 16;
    std::vector<uint16_t> q, wc, cc, pk;
    std::vector<int32_t> wbt, wused, cbt, sel, pos, cu;
    std::vector<float> sink, cs;
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
    for (const auto& s : seqs) {
        for (int i = 0; i < s.first; ++i) f.pos.push_back(s.second - s.first + 1 + i);
        f.cu.push_back(f.cu.back() + s.first);
        f.maxp = std::max<int64_t>(f.maxp, s.second + 1);
    }
    f.M = f.cu.back();
    auto bf = [&](float sc) { return rad_f32_to_bf16(nd(rng) * sc); };
    f.q.resize((size_t)f.M * H * kD);
    for (auto& x : f.q) x = bf(0.05f);
    f.nwbt = (f.maxp + f.bs - 1) / f.bs + 1;
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
    const int64_t nent = f.maxp / ratio + 1;
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
    /* selections: distinct positions below the row's reach, ascending, -1 after (5 % of the slots) */
    f.sel.assign((size_t)f.M * f.topk, -1);
    for (int m = 0; m < f.M; ++m) {
        const int64_t reach = ((int64_t)f.pos[(size_t)m] + 1) / ratio;
        std::vector<int32_t> pick;
        if (reach > 0) {
            std::uniform_int_distribution<int64_t> any(0, reach - 1);
            for (int t = 0; t < f.topk; ++t)
                if (pct(rng) < 95) pick.push_back((int32_t)any(rng));
        }
        std::sort(pick.begin(), pick.end());
        pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
        for (size_t t = 0; t < pick.size(); ++t) f.sel[(size_t)m * f.topk + t] = pick[t];
    }
    f.sink.resize((size_t)H);
    for (auto& x : f.sink) x = nd(rng);
    f.cs.resize((size_t)(f.maxp + 1) * kRot);
    for (auto& x : f.cs) x = std::cos(nd(rng));
    std::vector<int32_t> page((size_t)f.ncblk);
    for (int64_t i = 0; i < f.ncblk; ++i) page[(size_t)i] = (int32_t)i;
    f.pk.assign((size_t)(f.ncblk * kSlots), 0x7777);
    RadTensor t[3] = { tens(f.cc.data(), RAD_BF16, { f.ncblk, kD }), tens(page.data(), RAD_I32, { f.ncblk }),
                       tens(f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }) };
    RadParam p[5] = { pint("M", f.ncblk), pint("n", kD), pint("block_size", f.ratio), pstr("dtype", "bf16"),
                      pstr("format", "nvfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 5;
    if (r4d_ds41_entry_store_nv4(&a, nullptr) != RAD_OK) std::printf("FAIL entry_store nvfp4 refused\n");
    return f;
}

/* A view of the fixture's rows [r0, r0 + n) with cu `cuv` (the attention's operands as the engine slices them). */
struct View {
    int r0 = 0, n = 0;
    std::vector<int32_t> cu;   /* empty: the fixture's */
};

/* the attention operands (11; 12 with cos_sin for the fold), then `extra` (unpacked / gathered) */
int attn(RadLaunchFn fn, const Fix& f, const View& v, bool packed, bool fold, uint16_t* out, const RadTensor* extra) {
    const std::vector<int32_t>& cu = v.cu.empty() ? f.cu : v.cu;
    const int S = (int)cu.size() - 1;
    std::vector<RadTensor> t = {
        tens(f.q.data() + (size_t)v.r0 * f.H * kD, RAD_BF16, { v.n, (int64_t)f.H * kD }),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { S }),
        packed ? tens(f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
               : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { S, f.ncbt }),
        tens(f.sel.data() + (size_t)v.r0 * f.topk, RAD_I32, { v.n, f.topk }),
        tens(f.pos.data() + v.r0, RAD_I32, { v.n }),
        tens(cu.data(), RAD_I32, { S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out, RAD_BF16, { v.n, (int64_t)f.H * kD }),
    };
    std::vector<RadParam> p = { pint("M", v.n), pint("heads", f.H), pint("dim", kD), pint("window", f.win),
                                pint("ratio", f.ratio), pint("topk", f.topk), pf64("scale", 1.0 / std::sqrt((double)kD)),
                                pstr("dtype", "bf16") };
    if (packed) p.push_back(pstr("latent_format", "nvfp4"));
    if (fold) {
        t.push_back(tens(f.cs.data(), RAD_F32, { f.maxp + 1, kRot }));
        p.push_back(pint("rot", kRot));
        p.push_back(pint("off", kD - kRot));
        p.push_back(pstr("mode", "gptj"));
        p.push_back(pint("inverse", 1));
        p.push_back(pint("block", 32));
        p.push_back(pstr("format", "fp8_e4m3"));
        p.push_back(pstr("block_scale", "ue8m0"));
    }
    if (extra) t.push_back(*extra);
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    return fn(&a, nullptr);
}

/* latent_unpack over view v into `u` [rows][w] */
int unpack(const Fix& f, const View& v, uint16_t* u, int64_t rows, int64_t w, const uint16_t* pkd = nullptr) {
    const std::vector<int32_t>& cu = v.cu.empty() ? f.cu : v.cu;
    const int S = (int)cu.size() - 1;
    RadTensor t[5] = { tens(pkd ? pkd : f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }),
                       tens(f.cbt.data(), RAD_I32, { S, f.ncbt }), tens(f.pos.data() + v.r0, RAD_I32, { v.n }),
                       tens(cu.data(), RAD_I32, { S + 1 }), tens(u, RAD_BF16, { rows, w }) };
    RadParam p[4] = { pint("M", v.n), pint("dim", kD), pint("ratio", f.ratio), pstr("latent_format", "nvfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 4;
    return r4d_rad_ds41_latent_unpack(&a, nullptr);
}
int gather(const Fix& f, const View& v, uint16_t* g, int64_t pitch) {
    const std::vector<int32_t>& cu = v.cu.empty() ? f.cu : v.cu;
    const int S = (int)cu.size() - 1;
    RadTensor t[5] = { tens(f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }),
                       tens(f.cbt.data(), RAD_I32, { S, f.ncbt }),
                       tens(f.sel.data() + (size_t)v.r0 * f.topk, RAD_I32, { v.n, f.topk }),
                       tens(cu.data(), RAD_I32, { S + 1 }), tens(g, RAD_BF16, { v.n, pitch }) };
    RadParam p[5] = { pint("M", v.n), pint("dim", kD), pint("ratio", f.ratio), pint("topk", f.topk),
                      pstr("latent_format", "nvfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 5;
    return r4d_rad_ds41_latent_gather(&a, nullptr);
}

int64_t ndiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    int64_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    return d;
}

/* the host plan (ds_lu_span's): sequence s -> [lo, hi) at off */
struct Span { int64_t lo, hi, off; };
std::vector<Span> host_plan(const Fix& f, const View& v, int64_t cap) {
    const std::vector<int32_t>& cu = v.cu.empty() ? f.cu : v.cu;
    const int S = (int)cu.size() - 1;
    std::vector<Span> sp;
    int64_t acc = 0;
    for (int s = 0; s < S; ++s) {
        int64_t n = 0;
        if (cu[(size_t)s + 1] > cu[(size_t)s]) {
            int64_t lr = cu[(size_t)s + 1] - 1;
            lr = lr < 0 ? 0 : (lr >= v.n ? v.n - 1 : lr);
            n = ((int64_t)f.pos[(size_t)(v.r0 + lr)] + 1) / f.ratio;
            n = std::min<int64_t>(n, f.ncbt * f.ratio / f.ratio * 1);   /* ncbt blocks of `ratio` positions, one entry each */
            n = std::min<int64_t>(n, f.ncbt);
        }
        const int64_t take = std::min(n, cap - acc);
        sp.push_back({ n - take, n, acc });
        acc += take;
    }
    return sp;
}

/* DKV-U over view v with a plan of `rows` x `per` entries; `unpack_view` (default v) is the view latent_unpack sees */
void case_u(const char* name, const Fix& f, const View& v, int64_t rows, int64_t per, bool fold, const View* uview = nullptr,
            bool mutate = false) {
    const View& uv = uview ? *uview : v;
    const size_t no = (size_t)v.n * f.H * kD;
    std::vector<uint16_t> ob(no, 0x5555), op(no, 0x3333), ou(no, 0x2222);
    const RadLaunchFn plain = fold ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip : r4d_rad_ds41_attn_sparse_sink;
    const RadLaunchFn unp = fold ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_unpacked : r4d_rad_ds41_attn_sparse_sink_unpacked;
    expect(attn(plain, f, v, false, fold, ob.data(), nullptr) == RAD_OK, name, "attn bf16");
    expect(attn(plain, f, v, true, fold, op.data(), nullptr) == RAD_OK, name, "attn nvfp4");
    const int64_t w = per * kD;
    std::vector<uint16_t> ubuf((size_t)(rows * w + 8), 0x4444);
    uint16_t* u = (uint16_t*)(((uintptr_t)ubuf.data() + 15) & ~(uintptr_t)15);
    for (int64_t i = 0; i < rows * w; ++i) u[i] = 0x4444;
    expect(unpack(f, uv, u, rows, w) == RAD_OK, name, "latent_unpack");
    /* the rows the plan holds: the cache's bf16 entries; unmapped ones and rows past the plan untouched */
    const std::vector<Span> sp = host_plan(f, uv, rows * per);
    int64_t bad = 0, held = 0, untouched = 0, lowest = 0;
    for (int s = 0; s < (int)sp.size(); ++s)
        for (int64_t c = sp[(size_t)s].lo; c < sp[(size_t)s].hi; ++c) {
            const int64_t row = sp[(size_t)s].off + c - sp[(size_t)s].lo;
            const int32_t blk = f.cbt[(size_t)(s * f.ncbt + (c * f.ratio) / f.ratio)];
            const uint16_t* r = u + row * kD;
            if (blk < 0 || blk >= f.ncblk) {
                for (int d = 0; d < kD; ++d) untouched += r[d] != 0x4444;
                continue;
            }
            ++held;
            for (int d = 0; d < kD; ++d) bad += r[d] != f.cc[(size_t)(blk * kD + d)];
            if (s == 0 && c == sp[0].lo) lowest = c;
        }
    int64_t used = 0;
    for (const Span& x : sp) used += x.hi - x.lo;
    int64_t past = 0;
    for (int64_t i = used * kD; i < rows * w; ++i) past += u[i] != 0x4444;
    expect(bad == 0, name, "a held row is not the entry's stored bf16 bits", bad, held * kD);
    expect(untouched == 0, name, "an unmapped entry's row was written", untouched, 0);
    expect(past == 0, name, "a row past the plan was written", past, 0);
    const RadTensor ut = tens(u, RAD_BF16, { rows, w });
    expect(attn(unp, f, v, true, fold, ou.data(), &ut) == RAD_OK, name, "attn unpacked");
    const int64_t d1 = ndiff(ob, op), d2 = ndiff(ob, ou);
    int64_t nz = 0;
    for (size_t i = 0; i < no; ++i) nz += (ob[i] & 0x7fffu) != 0;
    expect(d1 == 0, name, "the packed attention differs from the bf16 cache's", d1, (long long)no);
    expect(d2 == 0, name, "the unpacked attention differs from the bf16 cache's", d2, (long long)no);
    expect(nz > (int64_t)no / 2, name, "the output is mostly zero", nz, (long long)no);
    long long dm = -1, dx = -1;
    if (mutate && held > 0) {
        /* one value of the plan's first held row of sequence 0 (an entry the rows select, at a held tile, by the plan:
         * the lowest held entries are the oldest, selected by every row's first tiles) and a whole row from another
         * entry: each must change the output where a tile reads it */
        std::vector<uint16_t> om(no, 0x1111);
        int64_t mrow = -1;   /* row 0's first valid selection, held (the plan holds all of sequence 0 here) */
        for (int j = 0; j < f.topk && mrow < 0; ++j) {
            const int32_t c = f.sel[(size_t)v.r0 * f.topk + j];
            if (c < 0 || c >= f.ncbt || f.cbt[(size_t)c] < 0 || c < sp[0].lo || c >= sp[0].hi) continue;
            mrow = sp[0].off + c - sp[0].lo;
        }
        expect(mrow >= 0, name, "no held selection of row 0 to mutate", 0, 0);
        if (mrow < 0) mrow = 0;
        std::vector<uint16_t> saved(u + mrow * kD, u + (mrow + 1) * kD);
        for (int d = 0; d < kD; ++d) u[mrow * kD + d] = rad_f32_to_bf16(rad_bf16_to_f32(u[mrow * kD + d]) + 0.75f);
        expect(attn(unp, f, v, true, fold, om.data(), &ut) == RAD_OK, name, "attn unpacked (mutated)");
        dm = ndiff(ob, om);
        expect(dm > 0, name, "a changed held row did not change the output (the rows are not read)", dm, 0);
        std::memcpy(u + mrow * kD, saved.data(), kD * 2);
        std::vector<uint16_t> ok2(no, 0x1212);
        expect(attn(unp, f, v, true, fold, ok2.data(), &ut) == RAD_OK, name, "attn unpacked (restored)");
        dx = ndiff(ob, ok2);
        expect(dx == 0, name, "the restored rows' output differs", dx, 0);
    }
    std::printf("%-58s rows %d seqs %zu ratio %d topk %d plan %lld x %lld (held %lld entries, from %lld): packed vs bf16 "
                "%lld, unpacked vs bf16 %lld of %zu%s\n",
                name, v.n, sp.size(), f.ratio, f.topk, (long long)rows, (long long)per, (long long)held, (long long)lowest,
                (long long)d1, (long long)d2, no, mutate ? (dm > 0 ? "; mutation seen" : "; MUTATION NOT SEEN") : "");
}

void case_g(const char* name, const Fix& f, const View& v, bool fold, bool mutate = false) {
    const size_t no = (size_t)v.n * f.H * kD;
    std::vector<uint16_t> ob(no, 0x5555), og(no, 0x2222);
    const RadLaunchFn plain = fold ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip : r4d_rad_ds41_attn_sparse_sink;
    const RadLaunchFn gat = fold ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered : r4d_rad_ds41_attn_sparse_sink_gathered;
    expect(attn(plain, f, v, false, fold, ob.data(), nullptr) == RAD_OK, name, "attn bf16");
    const int64_t pitch = (int64_t)f.topk * kD;
    std::vector<uint16_t> gbuf((size_t)(v.n * pitch + 8), 0x4444);
    uint16_t* g = (uint16_t*)(((uintptr_t)gbuf.data() + 15) & ~(uintptr_t)15);
    for (int64_t i = 0; i < v.n * pitch; ++i) g[i] = 0x4444;
    expect(gather(f, v, g, pitch) == RAD_OK, name, "latent_gather");
    /* each valid selection's row = the entry; an invalid one's untouched */
    const std::vector<int32_t>& cu = v.cu.empty() ? f.cu : v.cu;
    int64_t bad = 0, inv = 0, valid = 0;
    for (int m = 0; m < v.n; ++m) {
        int s = 0;
        while (s + 1 < (int)cu.size() - 1 && cu[(size_t)s + 1] <= m) ++s;
        for (int j = 0; j < f.topk; ++j) {
            const int32_t c = f.sel[(size_t)(v.r0 + m) * f.topk + j];
            const uint16_t* r = g + m * pitch + (int64_t)j * kD;
            const bool ok = c >= 0 && (int64_t)c * f.ratio / f.ratio < f.ncbt && f.cbt[(size_t)(s * f.ncbt + c)] >= 0;
            if (!ok) {
                for (int d = 0; d < kD; ++d) inv += r[d] != 0x4444;
                continue;
            }
            ++valid;
            const int32_t blk = f.cbt[(size_t)(s * f.ncbt + c)];
            for (int d = 0; d < kD; ++d) bad += r[d] != f.cc[(size_t)(blk * kD + d)];
        }
    }
    expect(bad == 0, name, "a gathered row is not the entry's stored bf16 bits", bad, valid * kD);
    expect(inv == 0, name, "an invalid selection's row was written", inv, 0);
    const RadTensor gt = tens(g, RAD_BF16, { v.n, pitch });
    expect(attn(gat, f, v, true, fold, og.data(), &gt) == RAD_OK, name, "attn gathered");
    const int64_t d = ndiff(ob, og);
    expect(d == 0, name, "the gathered attention differs from the bf16 cache's", d, (long long)no);
    long long dm = -1;
    if (mutate) {
        int64_t mr = -1;
        for (int j = 0; j < f.topk && mr < 0; ++j)
            if (f.sel[(size_t)v.r0 * f.topk + j] >= 0 && f.cbt[(size_t)f.sel[(size_t)v.r0 * f.topk + j]] >= 0) mr = j;
        if (mr >= 0) {
            for (int dd = 0; dd < kD; ++dd) g[mr * kD + dd] = rad_f32_to_bf16(rad_bf16_to_f32(g[mr * kD + dd]) + 0.75f);
            std::vector<uint16_t> om(no, 0x1111);
            expect(attn(gat, f, v, true, fold, om.data(), &gt) == RAD_OK, name, "attn gathered (mutated)");
            dm = ndiff(ob, om);
            expect(dm > 0, name, "a changed gathered row did not change the output", dm, 0);
        }
    }
    std::printf("%-58s rows %d ratio %d topk %d: %lld valid selections gathered, gathered vs bf16 %lld of %zu%s\n", name,
                v.n, f.ratio, f.topk, (long long)valid, (long long)d, no,
                mutate ? (dm > 0 ? "; mutation seen" : "; MUTATION NOT SEEN") : "");
}

/* every scale byte x every code through latent_unpack's decode (and the WMMA form's v3 decode, via the plan's rows),
 * against the host decode; NaN scale bytes (0x7f, 0xff) reported, not required */
void case_decode_all() {
    const char* name = "v3 decode: 256 scale bytes x 16 codes";
    /* 8 entries of 32 blocks: block b of entry e has scale byte 32 e + b and codes 0..15 */
    const int E = 8;
    std::vector<uint8_t> pk((size_t)E * 288, 0);
    for (int e = 0; e < E; ++e)
        for (int b = 0; b < 32; ++b) {
            pk[(size_t)e * 288 + 256 + b] = (uint8_t)(32 * e + b);
            for (int i = 0; i < 16; ++i) {
                const int d = 16 * b + i;
                pk[(size_t)e * 288 + d / 2] |= (uint8_t)((unsigned)i << (4 * (d & 1)));
            }
        }
    std::vector<int32_t> cbt(E + 1), pos = { E - 1 }, cu = { 0, 1 };
    for (int e = 0; e <= E; ++e) cbt[(size_t)e] = e < E ? e : -1;
    std::vector<uint16_t> ubuf((size_t)E * kD + 8, 0x4444);
    uint16_t* u = (uint16_t*)(((uintptr_t)ubuf.data() + 15) & ~(uintptr_t)15);
    RadTensor t[5] = { tens(pk.data(), RAD_BF16, { E, 1, 1, kSlots }), tens(cbt.data(), RAD_I32, { 1, E + 1 }),
                       tens(pos.data(), RAD_I32, { 1 }), tens(cu.data(), RAD_I32, { 2 }),
                       tens(u, RAD_BF16, { E, kD }) };
    RadParam p[4] = { pint("M", 1), pint("dim", kD), pint("ratio", 1), pstr("latent_format", "nvfp4") };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p;
    a.n_p = 4;
    expect(r4d_rad_ds41_latent_unpack(&a, nullptr) == RAD_OK, name, "latent_unpack");
    int64_t bad = 0, nanb = 0;
    for (int e = 0; e < E; ++e)
        for (int d = 0; d < kD; ++d) {
            const unsigned sb = (unsigned)(32 * e + d / 16), c = (unsigned)(d % 16);
            const uint16_t want = host_val(c, sb), got = u[(size_t)e * kD + d];
            if ((sb & 0x7fu) == 0x7fu) nanb += got != want;
            else bad += got != want;
        }
    expect(bad == 0, name, "a decoded value differs from the host decode", bad, 254 * 16);
    std::printf("%-58s %lld of %d values differ (NaN scale bytes: %lld of 32 differ, not required)\n", name,
                (long long)bad, 254 * 16, (long long)nanb);
}

}  // namespace

int main() {
    setenv("RADIANCE_DS41_NVFAST", "1", 1);   /* lever DKV-F on for attn_sparse_sink's own WMMA form (read once) */
    case_decode_all();
    /* DKV-U, the WMMA form (> 48 rows), 16 heads */
    {
        const Fix f = make(51, 16, 1, { { 64, 2047 } });
        case_u("U: 64 rows, ratio 1, plan holds all", f, { 0, 64, {} }, 64, 32, false, nullptr, true);
        case_u("U: 64 rows, ratio 1, plan holds the newest 640 (tiles mixed)", f, { 0, 64, {} }, 64, 10, false);
        case_u("U: 64 rows, ratio 1, plan holds 64 (most tiles packed)", f, { 0, 64, {} }, 64, 1, false);
        case_u("U+fold d: 64 rows, ratio 1, plan holds all", f, { 0, 64, {} }, 64, 32, true, nullptr, true);
    }
    {
        const Fix f = make(52, 16, 2, { { 72, 3071 } }, 128);
        case_u("U: 72 rows, ratio 2, topk 128, plan holds all", f, { 0, 72, {} }, 72, 32, false, nullptr, true);
        /* a PSK-style cut: the last 56 rows, cu still the uncut segment's {0, 72}, the plan unpacked over the 72 */
        const View uncut{ 0, 72, { 0, 72 } }, cut{ 16, 56, { 0, 72 } };
        case_u("U: PSK cut (last 56 of 72 rows), plan from the uncut rows", f, cut, 72, 32, false, &uncut);
        case_u("U: PSK cut, plan unpacked over the cut rows into 72 rows", f, cut, 72, 32, false, nullptr, true);
        case_u("U+fold d: PSK cut, plan from the uncut rows", f, cut, 72, 32, true, &uncut);
    }
    {
        /* two sequences: the first takes most of the plan, the second what is left (part of its entries) */
        const Fix f = make(53, 16, 1, { { 30, 1500 }, { 40, 2600 } });
        case_u("U: 70 rows, two sequences, plan holds all", f, { 0, 70, {} }, 70, 64, false);
        case_u("U: 70 rows, two sequences, second only partly held", f, { 0, 70, {} }, 70, 24, false);
    }
    /* DKV-G, the decode band */
    {
        const Fix f = make(54, 16, 1, { { 1, 3000 } }, 512);
        case_g("G: 1 row, 512 selections", f, { 0, 1, {} }, false, true);
        case_g("G+fold d: 1 row, 512 selections", f, { 0, 1, {} }, true, true);
    }
    {
        const Fix f = make(55, 16, 2, { { 6, 2000 }, { 10, 3500 } }, 256);
        case_g("G: 16 rows, two sequences, ratio 2", f, { 0, 16, {} }, false);
        case_g("G+fold d: 16 rows, two sequences, ratio 2", f, { 0, 16, {} }, true);
    }
    {
        const Fix f = make(56, 16, 1, { { 8, 2600 }, { 32, 3999 } });
        case_g("G: 40 rows, tiled PV", f, { 0, 40, {} }, false, true);
        case_g("G+fold d: 40 rows", f, { 0, 40, {} }, true);
    }
    std::printf("ds41 DKV: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
