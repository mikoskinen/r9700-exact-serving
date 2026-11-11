/* tests/r4d_emu: lever NVG-fold (RADIANCE_DS41_NVGATHER_FOLD, 11.11.2026) on the host SIMT emulator.
 *
 * index_topk_nvg is index_topk whose final selection kernel (ds41_index_topk_nvg_kernel, or the split's
 * ds41_index_topk_split2_nvg_kernel) also unpacks each row's valid selections from the packed compressed latent into
 * `gathered`; attn_sparse_sink_gathered / attn_sparse_sink_rope_roundtrip_gathered read those rows instead of unpacking
 * the cache. For every case:
 *   sel        index_topk_nvg's selections are index_topk's, byte for byte (both selection forms: the one-block-a-row
 *              kernel and the chunk split; with and without keep, its kept walk);
 *   rows       every valid selection's gathered row is lever NVG's: the bytes ds41_latent_gather_nv4_kernel writes into
 *              attn_sparse_sink's scratch over the same selections (RADIANCE_DS41_NVGATHER=48 here), and the bf16
 *              cache's entry; an invalid selection's row (-1, past the table, an unmapped block) is written by neither;
 *   attention  attn_sparse_sink_gathered over those rows = attn_sparse_sink packed (the packed kernel) = with lever NVG's
 *              gather = over the bf16 cache; attn_sparse_sink_rope_roundtrip_gathered = attn_sparse_sink_rope_roundtrip
 *              (packed, NVG-gathered);
 *   mutations  the gathered attention reads the rows, not the cache: one gathered row changed, the rows of another
 *              layer's cache, and the rows of a stale selection each change its output (and the reference never);
 *              a gathered operand short of the pass's rows is refused.
 * Rows 1..40 of one or two sequences, ratio 1 / 2 / 4, topk 64 / 512 / 1024, 8 heads. */
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

extern "C" int r4d_rad_ds41_index_topk(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs*);
extern "C" int r4d_rad_ds41_index_topk_nvg(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_gathered(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered(const RadArgs*, RadStream);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs*, RadStream);

namespace {

constexpr int kD = 512, kSlots = 144;   /* the latent; its packed entry in bf16 slots (288 bytes) */
constexpr uint16_t kSentinel = 0x5A5A;

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

int g_fail = 0, g_check = 0;
void expect(bool ok, const std::string& name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name.c_str(), what, a, b);
}
int launch(int (*fn)(const RadArgs*, RadStream), RadTensor* t, int nt, const std::vector<RadParam>& p,
           void* scratch = nullptr, int64_t scratch_bytes = 0) {
    RadArgs a{};
    a.t = t;
    a.n_t = nt;
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = scratch;
    a.scratch_bytes = scratch_bytes;
    return fn(&a, nullptr);
}
long long diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += a[i] != b[i];
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}
/* a 16-byte aligned view into v (allocated with 8 spare elements) */
template <class T>
T* al16(std::vector<T>& v) {
    return (T*)(((uintptr_t)v.data() + 15) & ~(uintptr_t)15);
}

/* quant_roundtrip(fp4_e2m1, e4m3, 16) on one block of 16, in place (libref's rule; test_ds41attn's copy). */
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

/* One step: window caches, the compressed latent (bf16 and packed) at `ratio` with an entry a block, the indexer's
 * score plane over `C` positions (some -inf, some ties), optional keep words, sinks. `tab_frac` < 100 cuts the
 * compressed block table short (selections past it are not entries). */
struct Fix {
    int H = 8, ratio = 1, win = 128, K = 64, S = 0, M = 0, block = 0;
    int64_t C = 0, nwblk = 0, ncblk = 0, nwbt = 0, ncbt = 0;
    const int bs = 16;
    std::vector<uint16_t> q, wc, cc, pk;
    std::vector<int32_t> wbt, wused, cbt, pos, cu, keep;
    std::vector<float> sink, sc;
};

int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

int store_nv4(const Fix& f, const std::vector<uint16_t>& cc, uint16_t* packed) {
    std::vector<int32_t> page((size_t)f.ncblk);
    for (int64_t i = 0; i < f.ncblk; ++i) page[(size_t)i] = (int32_t)i;
    RadTensor t[3] = { tens(cc.data(), RAD_BF16, { f.ncblk, kD }), tens(page.data(), RAD_I32, { f.ncblk }),
                       tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }) };
    return launch(r4d_ds41_entry_store_nv4, t, 3,
                  { pint("M", f.ncblk), pint("n", kD), pint("block_size", f.ratio), pstr("dtype", "bf16"),
                    pstr("format", "nvfp4") });
}

void fill_latent(std::mt19937& rng, std::vector<uint16_t>& cc, int64_t entries) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    cc.resize((size_t)(entries * kD));
    for (int64_t e = 0; e < entries; ++e)
        for (int b = 0; b < kD / 16; ++b) {
            const int k = pct(rng);
            const float s = k < 8 ? 1e-3f : k < 16 ? 0.02f : k < 18 ? 0.0f : 1.0f;
            uint16_t* v = &cc[(size_t)(e * kD + b * 16)];
            for (int i = 0; i < 16; ++i) v[i] = rad_f32_to_bf16(nd(rng) * s);
            roundtrip16(v);
        }
}

Fix make(uint32_t seed, int ratio, const std::vector<std::pair<int, int>>& seqs, int K, int block = 0,
         int tab_frac = 100) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pct(0, 99);
    Fix f;
    f.K = K;
    f.ratio = ratio;
    f.block = block;
    f.S = (int)seqs.size();
    f.cu.push_back(0);
    int64_t maxp = 0;
    for (const auto& s : seqs) {
        for (int i = 0; i < s.first; ++i) f.pos.push_back(s.second - s.first + 1 + i);
        f.cu.push_back(f.cu.back() + s.first);
        maxp = std::max<int64_t>(maxp, s.second + 1);
    }
    f.M = f.cu.back();
    auto bf = [&](float s) { return rad_f32_to_bf16(nd(rng) * s); };
    f.q.resize((size_t)f.M * f.H * kD);
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
    /* compressed: entry c of a sequence in block cbt[s][c] (block size ratio, an entry a block); 2 % unmapped, and
     * `tab_frac` % of the entries in the table at all */
    const int64_t nent = maxp / ratio + 1;
    f.C = nent + 1;
    f.ncbt = std::max<int64_t>(1, (nent + 1) * tab_frac / 100);
    f.ncblk = f.S * f.ncbt + 5;
    std::vector<int32_t> cperm((size_t)f.ncblk);
    for (int64_t i = 0; i < f.ncblk; ++i) cperm[(size_t)i] = (int32_t)i;
    std::shuffle(cperm.begin(), cperm.end(), rng);
    f.cbt.assign((size_t)(f.S * f.ncbt), -1);
    for (int s = 0; s < f.S; ++s)
        for (int64_t c = 0; c < f.ncbt; ++c)
            f.cbt[(size_t)(s * f.ncbt + c)] = pct(rng) < 2 ? -1 : cperm[(size_t)(s * f.ncbt + c)];
    fill_latent(rng, f.cc, f.ncblk);
    f.pk.assign((size_t)(f.ncblk * kSlots), 0x7777);
    /* the indexer's scores: a few repeated values (ties), 3 % -inf */
    f.sc.resize((size_t)(f.M * f.C));
    for (auto& x : f.sc) {
        const int k = pct(rng);
        x = k < 3 ? -INFINITY : k < 10 ? 0.25f * (float)(k % 4) : nd(rng);
    }
    if (block > 0) {
        const int64_t kw = keep_words(f.C, block);
        f.keep.resize((size_t)(f.M * kw));
        std::uniform_int_distribution<uint32_t> any;
        for (auto& w : f.keep) w = (int32_t)(any(rng) | any(rng));   /* ~3/4 of the blocks kept */
    }
    f.sink.resize((size_t)f.H);
    for (auto& x : f.sink) x = nd(rng);
    return f;
}

std::vector<RadParam> topk_params(const Fix& f, bool nvg) {
    std::vector<RadParam> p = { pint("M", f.M), pint("width", f.C), pint("topk", f.K), pint("ratio", f.ratio) };
    if (f.block > 0) p.push_back(pint("block", f.block));
    if (nvg) {
        p.push_back(pint("dim", kD));
        p.push_back(pstr("latent_format", "nvfp4"));
    }
    return p;
}

/* index_topk (nvg false) or index_topk_nvg over f's plane into sel (and gathered, [rows >= M][K 512], pitch gp),
 * with the split's scratch when `split`. */
int topk(const Fix& f, bool nvg, bool split, int32_t* sel, const uint16_t* packed, uint16_t* gathered, int64_t grows,
         int64_t gp) {
    RadTensor t[8] = {
        tens(f.sc.data(), RAD_F32, { f.M, f.C }),
        f.block > 0 ? tens(f.keep.data(), RAD_I32, { f.M, keep_words(f.C, f.block) }) : none(),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(sel, RAD_I32, { f.M, f.K }),
        tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(gathered, RAD_BF16, { grows, (int64_t)f.K * kD }, gp),
    };
    const std::vector<RadParam> p = topk_params(f, nvg);
    std::vector<uint64_t> scr;
    void* sp = nullptr;
    int64_t need = 0;
    if (split) {
        RadArgs a{};
        a.p = p.data();
        a.n_p = (int)p.size();
        need = r4d_rad_ds41_index_topk_scratch(&a);
        scr.assign((size_t)(need / 8 + 2), 0xA5A5A5A5A5A5A5A5ull);
        sp = scr.data();
    }
    return launch(nvg ? r4d_rad_ds41_index_topk_nvg : r4d_rad_ds41_index_topk, t, nvg ? 8 : 4, p, sp, need);
}

struct Rt {   /* the rope and round trip's geometry (DeepSeek-V4.1's: 64 rotated columns at 448) */
    int rot = 64, off = 448;
    std::vector<float> cs;
    int64_t tab = 0;
};

/* attn_sparse_sink (rt null) or attn_sparse_sink_rope_roundtrip over the packed cache (bf16 cache when packed is
 * null), its gathered form when `gathered` is given ([rows][K 512], pitch gp). */
int attn(const Fix& f, const uint16_t* packed, const int32_t* sel, uint16_t* out, const Rt* rt,
         const uint16_t* gathered = nullptr, int64_t grows = 0, int64_t gp = 0, void* scratch = nullptr,
         int64_t scratch_bytes = 0) {
    std::vector<RadTensor> t = {
        tens(f.q.data(), RAD_BF16, { f.M, (int64_t)f.H * kD }),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { f.S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { f.S }),
        packed ? tens(packed, RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
               : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        tens(sel, RAD_I32, { f.M, f.K }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(out, RAD_BF16, { f.M, (int64_t)f.H * kD }),
    };
    if (rt) t.push_back(tens(rt->cs.data(), RAD_F32, { rt->tab, rt->rot }));
    if (gathered) t.push_back(tens(gathered, RAD_BF16, { grows, (int64_t)f.K * kD }, gp));
    std::vector<RadParam> p = { pint("M", f.M),        pint("heads", f.H),     pint("dim", kD),
                                pint("window", f.win), pint("ratio", f.ratio), pint("topk", f.K),
                                pf64("scale", 1.0 / std::sqrt((double)kD)), pstr("dtype", "bf16") };
    if (packed) p.push_back(pstr("latent_format", "nvfp4"));
    if (rt) {
        p.push_back(pint("rot", rt->rot));
        p.push_back(pint("off", rt->off));
        p.push_back(pstr("mode", "gptj"));
        p.push_back(pint("inverse", 1));
        p.push_back(pint("block", 32));
        p.push_back(pstr("format", "fp8_e4m3"));
        p.push_back(pstr("block_scale", "ue8m0"));
    }
    int (*fn)(const RadArgs*, RadStream) =
        gathered ? (rt ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered : r4d_rad_ds41_attn_sparse_sink_gathered)
                 : (rt ? r4d_rad_ds41_attn_sparse_sink_rope_roundtrip : r4d_rad_ds41_attn_sparse_sink);
    return launch(fn, t.data(), (int)t.size(), p, scratch, scratch_bytes);
}

/* The decode kernel's key setup on the host: selection j of row m -> its entry's block, or -1. */
int64_t entry_of(const Fix& f, const int32_t* sel, int64_t m, int64_t j) {
    int s = 0;
    while (s + 1 < f.S && f.cu[(size_t)(s + 1)] <= m) ++s;
    const int64_t c = sel[(size_t)(m * f.K + j)];
    const int64_t cbs = f.ratio;
    if (c < 0 || (c * f.ratio) / cbs >= f.ncbt) return -1;
    const int64_t blk = f.cbt[(size_t)(s * f.ncbt + (c * f.ratio) / cbs)];
    return blk >= 0 && blk < f.ncblk ? blk : -1;
}

void run_case(const std::string& name, Fix& f, bool split, bool ort) {
    expect(store_nv4(f, f.cc, f.pk.data()) == RAD_OK, name, "entry_store nvfp4");
    const int64_t M = f.M, K = f.K, rowe = K * kD, no = M * (int64_t)f.H * kD;
    /* ---- the selection: index_topk vs index_topk_nvg (gathered rows at a pitch past the row, 16-byte aligned) */
    const int64_t gp = rowe + 64;
    std::vector<int32_t> selA((size_t)(M * K), 0x1111), selB((size_t)(M * K), 0x2222);
    std::vector<uint16_t> gbuf((size_t)(M * gp + 8), kSentinel);
    uint16_t* G = al16(gbuf);
    expect(topk(f, false, split, selA.data(), nullptr, nullptr, 0, 0) == RAD_OK, name, "index_topk");
    expect(topk(f, true, split, selB.data(), f.pk.data(), G, M, gp) == RAD_OK, name, "index_topk_nvg");
    long long dsel = 0, nvalid = 0, ninval = 0, nneg = 0;
    for (size_t i = 0; i < selA.size(); ++i) dsel += selA[i] != selB[i];
    expect(dsel == 0, name, "index_topk_nvg's selection differs from index_topk's", dsel, (long long)selA.size());
    /* ---- lever NVG's gather over the same selections: attn_sparse_sink with the scratch its hook sizes (48 rows) */
    const int64_t need = (int64_t)48 * K * kD * 2;
    std::vector<uint16_t> sbuf((size_t)(need / 2 + 8), kSentinel);
    uint16_t* SC = al16(sbuf);
    std::vector<uint16_t> oN((size_t)no, 0x2222), oP((size_t)no, 0x3333), oB((size_t)no, 0x4444), oF((size_t)no, 0x5555);
    expect(attn(f, f.pk.data(), selA.data(), oN.data(), nullptr, nullptr, 0, 0, SC, need) == RAD_OK, name,
           "attn_sparse_sink, NVG gathered");
    /* ---- the rows: gathered = NVG's scratch = the bf16 cache's entry for every valid selection; neither writes the
     * others */
    long long dvs = 0, dvc = 0, wr_inv = 0, sw_inv = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < K; ++j) {
            const int64_t e = entry_of(f, selA.data(), m, j);
            const uint16_t* g = G + m * gp + j * kD;
            const uint16_t* s = SC + m * rowe + j * kD;
            nneg += selA[(size_t)(m * K + j)] < 0;
            if (e >= 0) {
                ++nvalid;
                for (int d = 0; d < kD; ++d) {
                    dvs += g[d] != s[d];
                    dvc += g[d] != f.cc[(size_t)(e * kD + d)];
                }
            } else {
                ++ninval;
                for (int d = 0; d < kD; ++d) {
                    wr_inv += g[d] != kSentinel;
                    sw_inv += s[d] != kSentinel;
                }
            }
        }
    long long pad = 0;   /* the pitch's tail past each row's K entries */
    for (int64_t m = 0; m < M; ++m)
        for (int64_t i = rowe; i < gp; ++i) pad += G[m * gp + i] != kSentinel;
    expect(nvalid > 0, name, "no valid selection (the fixture is too weak)");
    expect(dvs == 0, name, "a gathered row differs from lever NVG's scratch row", dvs, nvalid * kD);
    expect(dvc == 0, name, "a gathered row differs from the bf16 cache's entry", dvc, nvalid * kD);
    expect(wr_inv == 0, name, "index_topk_nvg wrote an invalid selection's row", wr_inv, ninval * kD);
    expect(sw_inv == 0, name, "lever NVG's gather wrote an invalid selection's row", sw_inv, ninval * kD);
    expect(pad == 0, name, "index_topk_nvg wrote past a row's K entries", pad, 0);
    /* ---- the attention: packed kernel = NVG = gathered = bf16 cache */
    expect(attn(f, f.pk.data(), selA.data(), oP.data(), nullptr) == RAD_OK, name, "attn_sparse_sink, packed");
    expect(attn(f, nullptr, selA.data(), oB.data(), nullptr) == RAD_OK, name, "attn_sparse_sink, bf16 cache");
    expect(attn(f, f.pk.data(), selB.data(), oF.data(), nullptr, G, M, gp) == RAD_OK, name, "attn_sparse_sink_gathered");
    const long long dFP = diff(oF, oP), dFN = diff(oF, oN), dFB = diff(oF, oB), dPB = diff(oP, oB);
    long long nz = 0;
    for (const uint16_t v : oF) nz += (v & 0x7fffu) != 0;
    expect(dFP == 0, name, "gathered attention differs from the packed kernel", dFP, no);
    expect(dFN == 0, name, "gathered attention differs from lever NVG's", dFN, no);
    expect(dFB == 0, name, "gathered attention differs from the bf16 cache's", dFB, no);
    expect(dPB == 0, name, "the packed kernel differs from the bf16 cache's (fixture)", dPB, no);
    expect(nz > no / 2, name, "the output is mostly zero (the fixture did not reach the keys)", nz, no);
    /* ---- fold d: attn_sparse_sink_rope_roundtrip, packed (NVG-gathered: the scratch is handed in) vs gathered */
    long long dR = -1, dR0 = -1;
    if (ort) {
        Rt rt;
        std::mt19937 rng((uint32_t)M * 7919u + 17u);
        std::uniform_real_distribution<float> u(-3.2f, 3.2f);
        rt.tab = 0;
        for (const int32_t p : f.pos) rt.tab = std::max<int64_t>(rt.tab, p + 1);
        rt.cs.resize((size_t)(rt.tab * rt.rot));
        for (int64_t p = 0; p < rt.tab; ++p)
            for (int i = 0; i < rt.rot / 2; ++i) {
                const float a = u(rng);
                rt.cs[(size_t)(p * rt.rot + i)] = std::cos(a);
                rt.cs[(size_t)(p * rt.rot + rt.rot / 2 + i)] = std::sin(a);
            }
        std::vector<uint16_t> r1((size_t)no, 0x1212), r2((size_t)no, 0x3434), r3((size_t)no, 0x5656);
        expect(attn(f, f.pk.data(), selA.data(), r1.data(), &rt, nullptr, 0, 0, SC, need) == RAD_OK, name,
               "attn_sparse_sink_rope_roundtrip, NVG gathered");
        expect(attn(f, f.pk.data(), selA.data(), r3.data(), &rt) == RAD_OK, name,
               "attn_sparse_sink_rope_roundtrip, packed");
        expect(attn(f, f.pk.data(), selB.data(), r2.data(), &rt, G, M, gp) == RAD_OK, name,
               "attn_sparse_sink_rope_roundtrip_gathered");
        dR = diff(r1, r2);
        dR0 = diff(r3, r2);
        expect(dR == 0, name, "gathered fold d differs from lever NVG's fold d", dR, no);
        expect(dR0 == 0, name, "gathered fold d differs from the packed fold d", dR0, no);
        expect(diff(r2, oF) > 0, name, "fold d changed nothing (the fixture is too weak)");
    }
    /* ---- mutations: the gathered attention must read the rows (each changes its output) */
    long long mu1 = -1, mu2 = -1, mu3 = -1;
    {
        /* (1) one valid selection's row: one value's low bit */
        std::vector<uint16_t> g2(gbuf.size(), kSentinel);
        uint16_t* G2 = al16(g2);
        std::memcpy(G2, G, (size_t)(M * gp) * 2);
        int64_t mm = -1, jj = -1;
        for (int64_t m = 0; m < M && mm < 0; ++m)
            for (int64_t j = 0; j < K; ++j) {
                const int64_t e = entry_of(f, selA.data(), m, j);
                if (e < 0) continue;
                const uint16_t* row = G2 + m * gp + j * kD;
                int d0 = -1;
                for (int d = 0; d < kD && d0 < 0; ++d)
                    if ((row[d] & 0x7fffu) > 0x3c00u) d0 = d;   /* |v| > 2^-7: a value that weighs */
                if (d0 < 0) continue;
                G2[m * gp + j * kD + d0] ^= 0x0040u;
                mm = m;
                jj = j;
                break;
            }
        std::vector<uint16_t> o((size_t)no, 0x6666);
        expect(attn(f, f.pk.data(), selA.data(), o.data(), nullptr, G2, M, gp) == RAD_OK, name, "mutation 1");
        mu1 = diff(o, oF);
        expect(mm >= 0 && mu1 > 0, name, "mutation 1: a changed gathered value did not change the output", mu1, mm);
    }
    {
        /* (2) another layer's cache: the same selections unpacked from a different latent (another kv source) */
        std::mt19937 rng2(4242u + (uint32_t)M);
        std::vector<uint16_t> cc2, pk2((size_t)(f.ncblk * kSlots), 0x7777);
        fill_latent(rng2, cc2, f.ncblk);
        expect(store_nv4(f, cc2, pk2.data()) == RAD_OK, name, "entry_store nvfp4 (other cache)");
        std::vector<int32_t> selC((size_t)(M * K), 0x3333);
        std::vector<uint16_t> g3((size_t)(M * gp + 8), kSentinel);
        expect(topk(f, true, split, selC.data(), pk2.data(), al16(g3), M, gp) == RAD_OK, name, "index_topk_nvg (other cache)");
        std::vector<uint16_t> o((size_t)no, 0x7777);
        expect(attn(f, f.pk.data(), selA.data(), o.data(), nullptr, al16(g3), M, gp) == RAD_OK, name, "mutation 2");
        mu2 = diff(o, oF);
        expect(mu2 > 0, name, "mutation 2: another cache's rows did not change the output", mu2, 0);
    }
    {
        /* (3) a stale selection: the rows of the previous token's scores (each row's plane rotated by one) */
        Fix f2 = f;
        for (int64_t m = 0; m < M; ++m) std::rotate(f2.sc.begin() + m * f.C, f2.sc.begin() + m * f.C + 1, f2.sc.begin() + (m + 1) * f.C);
        std::vector<int32_t> selD((size_t)(M * K), 0x4444);
        std::vector<uint16_t> g4((size_t)(M * gp + 8), kSentinel);
        expect(topk(f2, true, split, selD.data(), f.pk.data(), al16(g4), M, gp) == RAD_OK, name, "index_topk_nvg (stale)");
        std::vector<uint16_t> o((size_t)no, 0x8888);
        expect(attn(f, f.pk.data(), selA.data(), o.data(), nullptr, al16(g4), M, gp) == RAD_OK, name, "mutation 3");
        mu3 = diff(o, oF);
        expect(mu3 > 0, name, "mutation 3: a stale selection's rows did not change the output", mu3, 0);
    }
    /* ---- a gathered operand short of the pass's rows (or of the row's K entries) is refused */
    {
        std::vector<uint16_t> o((size_t)no, 0x9999);
        expect(attn(f, f.pk.data(), selA.data(), o.data(), nullptr, G, M - 1 > 0 ? M - 1 : 0, gp) != RAD_OK || M == 1,
               name, "a gathered operand of M - 1 rows was taken");
        std::vector<int32_t> selE((size_t)(M * K), 0);
        RadTensor t[8] = {
            tens(f.sc.data(), RAD_F32, { f.M, f.C }),
            f.block > 0 ? tens(f.keep.data(), RAD_I32, { f.M, keep_words(f.C, f.block) }) : none(),
            tens(f.pos.data(), RAD_I32, { f.M }),
            tens(selE.data(), RAD_I32, { f.M, f.K }),
            tens(f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio }),
            tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
            tens(f.cu.data(), RAD_I32, { f.S + 1 }),
            tens(G, RAD_BF16, { M, (int64_t)(f.K - 1) * kD }, gp),
        };
        expect(launch(r4d_rad_ds41_index_topk_nvg, t, 8, topk_params(f, true)) != RAD_OK, name,
               "index_topk_nvg took a gathered operand of K - 1 entries a row");
        std::vector<RadParam> p = topk_params(f, false);   /* no latent_format: refused */
        p.push_back(pint("dim", kD));
        t[7] = tens(G, RAD_BF16, { M, (int64_t)f.K * kD }, gp);
        expect(launch(r4d_rad_ds41_index_topk_nvg, t, 8, p) != RAD_OK, name, "index_topk_nvg took no latent_format");
    }
    std::printf("%-58s rows %lld ratio %d topk %lld%s%s: sel differ %lld; %lld valid / %lld invalid selections (%lld "
                "-1); rows vs NVG scratch %lld, vs bf16 cache %lld, invalid rows written %lld / %lld; attention "
                "gathered vs packed %lld, vs NVG %lld, vs bf16 %lld of %lld; fold d vs NVG %lld, vs packed %lld; "
                "mutations change %lld / %lld / %lld\n",
                name.c_str(), (long long)M, f.ratio, (long long)K, split ? ", split" : "", f.block ? ", keep" : "", dsel,
                nvalid, ninval, nneg, dvs, dvc, wr_inv, sw_inv, dFP, dFN, dFB, no, dR, dR0, mu1, mu2, mu3);
}

}  // namespace

int main() {
    setenv("RADIANCE_DS41_NVGATHER", "48", 1);   /* lever NVG on (read once, at the first launch): the reference rows */
    /* one decode row at the model's 512 selections, ratio 1 and 2: the one-block-a-row kernel, then the split */
    {
        Fix f = make(101, 1, { { 1, 3000 } }, 512);
        run_case("1 row, 512 of 3001, ratio 1", f, false, true);
    }
    {
        Fix f = make(102, 2, { { 1, 70000 } }, 512);
        run_case("1 row, 512 of 35001, ratio 2, split", f, true, true);
    }
    /* two sequences (ds_seq_of_row), the table cut short (selections past it), keep's kept walk */
    {
        Fix f = make(103, 1, { { 3, 2000 }, { 5, 4000 } }, 64, 8, 70);
        run_case("8 rows / 2 seqs, topk 64, keep block 8, table 70 %", f, false, true);
    }
    {
        Fix f = make(104, 1, { { 2, 33000 }, { 2, 40000 } }, 512, 8);
        run_case("4 rows / 2 seqs, 512 of 40001, keep block 8, split", f, true, false);
    }
    /* more rows than the split takes (one block a row), ratio 4, a short row (-1 fills), 1024 selections */
    {
        Fix f = make(105, 4, { { 24, 6000 }, { 16, 200 } }, 128);
        run_case("40 rows / 2 seqs, ratio 4, topk 128 (short rows: -1)", f, false, false);
    }
    {
        Fix f = make(106, 1, { { 2, 1500 } }, 1024);
        run_case("2 rows, 1024 of 1501", f, false, false);
    }
    std::printf("ds41 lever NVG-fold: %d checks, %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
