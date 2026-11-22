/* tests/r4d_emu: lever TSK (DRAFTER track round 15; fn4 ds41_tsk 1 / 2 / 3, served at 2) -- DeepSeek-V4.1's compressing
 * layers' decode-band attention as key chunks of kTskTpc 16-key tiles a workgroup (ds41_attn_tsk_kernel) and the partials'
 * merge (ds41_attn_tsk_merge_kernel), r4d_ds41_sparse.hip's tsk_emit -- on the host SIMT emulator. TSK's bytes are its own
 * (rule B against the stock decode forms: another summation order), so what it must hold is exactness UNDER LOAD
 * (batch invariance): a row's output is a function of its own q, keys and selections only.
 *   - taken: the op's scratch holds the chunk partials (written) and r4d_tsk_declines() does not move;
 *   - deterministic: the same call twice gives the same bytes;
 *   - batch invariant: every row of a pass gives the bytes it gives ALONE (a one-row pass of its own sequence), and inside
 *     another pass shape (the rows in reverse order, each its own sequence) -- at 1 .. 48 rows (1 .. 64 under band 64),
 *     1 .. 7 key chunks (above the band the pass takes the prefill-height form, the published split by pass height);
 *   - ds41_tsk 2 (TSKC: the merge grid past the full-card cliff) and 3 (+ TDD, which acts only on a gathered view) = 1's bytes;
 *   - the packed latent (nvfp4) = the bf16 cache's bytes (the packed entries decode to the stored bf16 bits);
 *   - not taken where it must not be: the switch at 0, a pass above the band (49 rows), scratch one byte short (declined
 *     loudly, counted) -- then the stock forms' bytes; TSK against them is printed (count, max |d|) and held within 1 %.
 * tp8 loop DECODE round 65 (QUEUE X13): no host test covered TSK before; the prerequisite of QUEUE D73 (q's rope inside
 * TSK's Q load). RADIANCE_DS41_TSK is read at load (g_tsk's initializer), so main() re-executes itself with it set. */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <unistd.h>

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
extern "C" int64_t r4d_rad_ds41_attn_sparse_sink_scratch(const RadArgs*);
extern "C" long long r4d_tsk_declines();
int r4d_tsk_get();
void r4d_tsk_set(int v);

namespace {

constexpr int kD = 512, kSlots = 144;   /* the latent; its packed entry in bf16 slots (288 bytes) */

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

int g_fail = 0, g_check = 0;
void expect(bool ok, const char* name, const char* what, long long a = 0, long long b = 0) {
    ++g_check;
    if (ok) return;
    ++g_fail;
    std::printf("FAIL %s: %s (%lld, %lld)\n", name, what, a, b);
}

/* quant_roundtrip(fp4_e2m1, e4m3, 16) on one block of 16, in place (libref's rule; test_ds41attn's) */
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

/* A step of sequences (rows a sequence, last position): window caches, the compressed latent caches (bf16 and packed) at
 * `ratio`, selections, sinks -- test_ds41attn's fixture; wused = the last position + 1, so a row's window starts at 0. */
struct Fix {
    int H = 0, ratio = 1, win = 128, topk = 64, S = 0, M = 0;
    int64_t nwblk = 0, ncblk = 0, nwbt = 0, ncbt = 0;
    const int bs = 16;
    std::vector<uint16_t> q, wc, cc, pk;
    std::vector<int32_t> wbt, wused, cbt, sel, pos, cu;
    std::vector<float> sink;
};

Fix make(uint32_t seed, int H, int ratio, const std::vector<std::pair<int, int>>& seqs, int topk) {
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

/* The rows `rows` of f as a pass of their own, each row its own sequence (its sequence's tables; wused so that its window
 * starts where it started in f: f's rows start at 0, so the row's own position + 1). The caches are f's. */
Fix slice(const Fix& f, const std::vector<int>& rows) {
    Fix g;
    g.H = f.H; g.ratio = f.ratio; g.win = f.win; g.topk = f.topk;
    g.nwblk = f.nwblk; g.ncblk = f.ncblk; g.nwbt = f.nwbt; g.ncbt = f.ncbt;
    g.wc = f.wc; g.cc = f.cc; g.pk = f.pk; g.sink = f.sink;
    g.S = (int)rows.size();
    g.M = g.S;
    g.cu.push_back(0);
    for (int r : rows) {
        int s = 0;
        while (f.cu[(size_t)s + 1] <= r) ++s;
        g.q.insert(g.q.end(), f.q.begin() + (size_t)r * f.H * kD, f.q.begin() + (size_t)(r + 1) * f.H * kD);
        g.pos.push_back(f.pos[(size_t)r]);
        g.cu.push_back(g.cu.back() + 1);
        g.wused.push_back(f.pos[(size_t)r] + 1);
        g.wbt.insert(g.wbt.end(), f.wbt.begin() + (size_t)(s * f.nwbt), f.wbt.begin() + (size_t)((s + 1) * f.nwbt));
        g.cbt.insert(g.cbt.end(), f.cbt.begin() + (size_t)(s * f.ncbt), f.cbt.begin() + (size_t)((s + 1) * f.ncbt));
        g.sel.insert(g.sel.end(), f.sel.begin() + (size_t)r * f.topk, f.sel.begin() + (size_t)(r + 1) * f.topk);
    }
    return g;
}

int store_nv4(const Fix& f, uint16_t* packed) {
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
    return r4d_ds41_entry_store_nv4(&a, nullptr);
}

std::vector<RadParam> params(const Fix& f, int64_t M, bool packed, int band) {
    std::vector<RadParam> p = { pint("M", M), pint("heads", f.H), pint("dim", kD), pint("window", f.win),
                                pint("ratio", f.ratio), pint("topk", f.topk), pf64("scale", 1.0 / std::sqrt((double)kD)),
                                pstr("dtype", "bf16") };
    if (packed) p.push_back(pstr("latent_format", "nvfp4"));
    if (band) p.push_back(pint("band", band));
    return p;
}

/* the op's scratch hook at the band's upper bound (TSK's chunk partials) */
int64_t scratch_need(const Fix& f, bool packed, int band) {
    std::vector<RadParam> p = params(f, band ? band : 48, packed, band);
    RadArgs a{};
    a.p = p.data();
    a.n_p = (int)p.size();
    return r4d_rad_ds41_attn_sparse_sink_scratch(&a);
}

/* attn_sparse_sink over the bf16 cache or the packed one, with `bytes` of scratch (filled 0x5a first); out filled first */
struct Run {
    std::vector<uint16_t> out;
    long long declines = 0;
    bool scratch_written = false;
    int rc = 0;
};
Run attn(const Fix& f, bool packed, int64_t bytes, int band = 0) {
    Run r;
    const size_t no = (size_t)f.M * f.H * kD;
    r.out.assign(no, 0x7b7b);
    std::vector<uint8_t> scr((size_t)bytes + 64, 0x5a);
    uint8_t* sp = (uint8_t*)(((uintptr_t)scr.data() + 15) & ~(uintptr_t)15);
    RadTensor t[11] = {
        tens(f.q.data(), RAD_BF16, { f.M, (int64_t)f.H * kD }),
        tens(f.wc.data(), RAD_BF16, { f.nwblk, 1, f.bs, kD }),
        tens(f.wbt.data(), RAD_I32, { f.S, f.nwbt }),
        tens(f.wused.data(), RAD_I32, { f.S }),
        packed ? tens(f.pk.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kSlots / f.ratio })
               : tens(f.cc.data(), RAD_BF16, { f.ncblk, 1, f.ratio, kD / f.ratio }),
        tens(f.cbt.data(), RAD_I32, { f.S, f.ncbt }),
        tens(f.sel.data(), RAD_I32, { f.M, f.topk }),
        tens(f.pos.data(), RAD_I32, { f.M }),
        tens(f.cu.data(), RAD_I32, { f.S + 1 }),
        tens(f.sink.data(), RAD_F32, { f.H }),
        tens(r.out.data(), RAD_BF16, { f.M, (int64_t)f.H * kD }),
    };
    std::vector<RadParam> p = params(f, f.M, packed, band);
    RadArgs a{};
    a.t = t;
    a.n_t = 11;
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = bytes > 0 ? sp : nullptr;
    a.scratch_bytes = bytes;
    const long long d0 = r4d_tsk_declines();
    r.rc = r4d_rad_ds41_attn_sparse_sink(&a, nullptr);
    r.declines = r4d_tsk_declines() - d0;
    for (int64_t i = 0; i < bytes && !r.scratch_written; ++i) r.scratch_written = sp[i] != 0x5a;
    return r;
}

long long ndiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += a[i] != b[i];
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}
/* rows of `part` (pass g of f's rows `rows`) against f's rows in `whole` */
long long rows_diff(const Fix& f, const std::vector<uint16_t>& whole, const std::vector<int>& rows, const std::vector<uint16_t>& part) {
    const size_t rw = (size_t)f.H * kD;
    long long d = 0;
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t e = 0; e < rw; ++e) d += whole[(size_t)rows[i] * rw + e] != part[i * rw + e];
    return d;
}

/* one fixture: every check above; `full` adds every row alone (the slow part) */
void run_case(const char* name, Fix f, bool full, int band = 0) {
    f.pk.assign((size_t)(f.ncblk * kSlots), 0x7777);
    expect(store_nv4(f, f.pk.data()) == RAD_OK, name, "entry_store nvfp4");
    const int64_t need = scratch_need(f, false, band), needp = scratch_need(f, true, band);
    const int bnd = band ? band : 48;
    const long nk = f.win + f.topk, nch = ((nk + 15) / 16 + 19) / 20;
    expect(need > 0 && need == needp, name, "the scratch hook does not size TSK's partials (bf16 / packed)", need, needp);
    const bool in_band = f.M <= bnd;

    r4d_tsk_set(1);
    expect(r4d_tsk_get() == 1, name, "ds41_tsk 1 not applied", r4d_tsk_get(), 1);
    const Run a1 = attn(f, false, need, band);
    const Run a1b = attn(f, false, need, band);
    const Run p1 = attn(f, true, need, band);
    expect(a1.rc == RAD_OK && a1b.rc == RAD_OK && p1.rc == RAD_OK, name, "attn (TSK 1)", a1.rc, p1.rc);
    expect(a1.declines == 0 && p1.declines == 0, name, "TSK declined a pass it serves", a1.declines, p1.declines);
    expect(a1.scratch_written == in_band, name, in_band ? "TSK not taken (scratch untouched)" : "TSK took a pass above the band",
           a1.scratch_written, in_band);
    const long long ddet = ndiff(a1.out, a1b.out), dpk = ndiff(a1.out, p1.out);
    expect(ddet == 0, name, "TSK not deterministic (two calls differ)", ddet, 0);
    expect(dpk == 0, name, "TSK over the packed latent differs from the bf16 cache's", dpk, 0);
    long long nz = 0;
    for (uint16_t v : a1.out) nz += (v & 0x7fffu) != 0;
    expect(nz > (long long)a1.out.size() / 2, name, "the output is mostly zero (the fixture did not reach the keys)", nz,
           (long long)a1.out.size());

    /* batch invariance: the rows in reverse order, each its own sequence (one pass), and (full) every row alone */
    std::vector<int> rev(f.M);
    for (int m = 0; m < f.M; ++m) rev[(size_t)m] = f.M - 1 - m;
    const Fix gr = slice(f, rev);
    const Run ar = attn(gr, false, scratch_need(gr, false, band), band);
    const long long drev = rows_diff(f, a1.out, rev, ar.out);
    expect(ar.rc == RAD_OK && drev == 0, name, "a row's bytes change with the pass's composition (reversed, own sequences)", drev, 0);
    /* (a pass above the band takes the prefill-height WMMA form, a one-row pass the decode band's: the published split of
     * the forms by pass height, not TSK's -- no row-alone check there) */
    long long dalone = 0, nalone = 0;
    for (int m = 0; m < f.M && in_band; ++m) {
        if (!full && m != 0 && m != f.M - 1 && m != f.M / 2) continue;
        const Fix g1 = slice(f, { m });
        const Run r1 = attn(g1, m % 2 == 1, scratch_need(g1, m % 2 == 1, band), band);   /* odd rows over the packed cache */
        expect(r1.rc == RAD_OK && r1.declines == 0 && r1.scratch_written, name, "a row alone: TSK not taken", m, r1.rc);
        dalone += rows_diff(f, a1.out, { m }, r1.out);
        ++nalone;
    }
    expect(dalone == 0, name, "a row alone gives other bytes than inside the pass", dalone, nalone);

    /* ds41_tsk 2 (TSKC) and 3 (+ TDD: no gathered view here) = 1's bytes */
    r4d_tsk_set(2);
    const Run a2 = attn(f, false, need, band);
    r4d_tsk_set(3);
    const Run a3 = attn(f, true, needp, band);
    const long long d2 = ndiff(a1.out, a2.out), d3 = ndiff(a1.out, a3.out);
    expect(a2.rc == RAD_OK && d2 == 0, name, "ds41_tsk 2 differs from 1", d2, a2.rc);
    expect(a3.rc == RAD_OK && d3 == 0, name, "ds41_tsk 3 (packed) differs from 1", d3, a3.rc);

    /* not taken: the switch at 0 (the stock forms), the scratch one byte short (declined, counted: the stock bytes) */
    r4d_tsk_set(0);
    const Run s0 = attn(f, false, need, band);
    expect(s0.rc == RAD_OK && !s0.scratch_written, name, "ds41_tsk 0 still wrote the scratch", s0.scratch_written, 0);
    r4d_tsk_set(2);
    long long dshort = -1;
    if (in_band) {
        const int64_t exact = (int64_t)f.M * nch * f.H * (kD + 2) * 4 + 256;   /* tsk_ws_bytes at this pass's M */
        const Run sh = attn(f, false, exact - 1, band);
        expect(sh.rc == RAD_OK && sh.declines == 1, name, "a pass with too little scratch was not declined once", sh.declines, 1);
        dshort = ndiff(sh.out, s0.out);
        expect(dshort == 0, name, "the declined pass's bytes are not the stock form's", dshort, 0);
        const Run ok = attn(f, false, exact, band);
        expect(ok.rc == RAD_OK && ok.declines == 0 && ndiff(ok.out, a1.out) == 0, name, "the exact scratch size not taken",
               ok.declines, 0);
    } else {
        expect(ndiff(a1.out, s0.out) == 0, name, "above the band: TSK's switch changed the bytes", ndiff(a1.out, s0.out), 0);
    }
    /* TSK vs the stock forms (rule B: info + a 1 % bound) */
    long long dst = 0;
    double maxd = 0.0, maxv = 0.0;
    for (size_t i = 0; i < a1.out.size(); ++i) {
        dst += a1.out[i] != s0.out[i];
        maxd = std::max(maxd, (double)std::fabs(rad_bf16_to_f32(a1.out[i]) - rad_bf16_to_f32(s0.out[i])));
        maxv = std::max(maxv, (double)std::fabs(rad_bf16_to_f32(s0.out[i])));
    }
    expect(maxd <= 0.01 * maxv, name, "TSK is more than 1 % of the output range from the stock forms", (long long)(maxd * 1e6),
           (long long)(maxv * 1e6));
    std::printf("%-44s rows %2d seqs %d heads %d keys %d+%d (%ld chunks)%s: TSK %s; det %lld, packed %lld, reversed %lld, alone %lld "
                "(%lld rows), tsk2 %lld, tsk3 %lld, short %lld; vs stock %lld of %zu differ, max |d| %.3g (range %.3g)\n",
                name, f.M, f.S, f.H, f.win, f.topk, nch, band ? " band 64" : "", in_band ? "taken" : "not taken (above band)",
                ddet, dpk, drev, dalone, nalone, d2, d3, dshort, dst, a1.out.size(), maxd, maxv);
}

}  // namespace

int main(int argc, char** argv) {
    /* g_tsk / tsk_env read RADIANCE_DS41_TSK while the program loads: set it (0 = sized, off) and start again */
    if (!std::getenv("RADIANCE_DS41_TSK")) {
        setenv("RADIANCE_DS41_TSK", "0", 1);
        execv("/proc/self/exe", argv);
        std::printf("FAIL: could not re-execute with RADIANCE_DS41_TSK=0\n");
        return 1;
    }
    if (std::strcmp(std::getenv("RADIANCE_DS41_TSK"), "0")) {
        std::printf("FAIL: RADIANCE_DS41_TSK=%s (this test wants 0 at load)\n", std::getenv("RADIANCE_DS41_TSK"));
        return 1;
    }
    const bool full = argc > 1 && !std::strcmp(argv[1], "--full");
    /* the served shape: 16 heads (64 / TP4), window 128 + 512 selections at ratio 4 (2 chunks of 20 tiles) */
    run_case("1 row, 512 selections, ratio 4", make(71, 16, 4, { { 1, 6000 } }, 512), true);
    run_case("8 rows (2 seqs x 4), 512 sel, ratio 4", make(72, 16, 4, { { 4, 5000 }, { 4, 7000 } }, 512), true);
    run_case("24 rows (6 seqs, depths 1-6), 512 sel", make(73, 16, 4, { { 1, 900 }, { 2, 3000 }, { 3, 4100 }, { 4, 5200 },
                                                                         { 6, 6300 }, { 8, 7400 } }, 512), full);
    run_case("48 rows (8 seqs x 6), 512 sel, ratio 4", make(74, 16, 4, { { 6, 2000 }, { 6, 2500 }, { 6, 3000 }, { 6, 3500 },
                                                                         { 6, 4000 }, { 6, 4500 }, { 6, 5000 }, { 6, 5500 } }, 512), full);
    /* one chunk (64 selections), several (1024 / 2048: 4 / 7 chunks, the last partial), early positions (selections -1) */
    run_case("4 rows, 64 sel, 1 chunk, ratio 1", make(75, 16, 1, { { 4, 400 } }, 64), true);
    run_case("6 rows, 1024 sel, 4 chunks, ratio 2", make(76, 16, 2, { { 3, 3000 }, { 3, 9000 } }, 1024), true);
    run_case("3 rows, 2048 sel, 7 chunks, ratio 4", make(77, 16, 4, { { 3, 12000 } }, 2048), true);
    run_case("5 rows near the start (few compressed entries)", make(78, 16, 4, { { 2, 20 }, { 3, 150 } }, 512), true);
    run_case("8 rows, 32 heads (2 head groups)", make(79, 32, 4, { { 8, 4000 } }, 512), full);
    /* the band: 49 rows (above 48: the stock forms whatever the switch), and 49 .. 64 under band 64 (D25 step 3) */
    run_case("49 rows (above the band)", make(80, 16, 4, { { 49, 6000 } }, 512), false);
    run_case("56 rows, band 64", make(81, 16, 4, { { 28, 5000 }, { 28, 6000 } }, 512), full, 64);
    std::printf("%d check(s), %d failure(s)\n", g_check, g_fail);
    return g_fail ? 1 : 0;
}
