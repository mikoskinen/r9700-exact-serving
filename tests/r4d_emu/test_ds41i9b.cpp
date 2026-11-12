/* tests/r4d_emu: ds41-int9b's three levers on the host SIMT emulator, each new form byte for byte against the stock form
 * it replaces (the GPU twin: r4d_selftest --case mhc_pcn_mix / ix_lds / ix_ni, at production sizes, in the queue item):
 *
 *   M1     mhc_post_collapse_rmsnorm_mix (r4d_ds41_elem.hip pcn_mix_body) against mhc_post_collapse_rmsnorm and then
 *          mhc_mix over its out, on the engine's buffers (post / comb read by the first and written by the second), at
 *          M 1 / 7 / 33 / 130 (r4d_act_threads' three norm widths) -- out, xc, y, pre, post, comb. Mutation controls: K1's
 *          slabs reversed (the projection's partial order) and K2's virtual-thread count (the norm's order) must differ.
 *   IX-LDS index_score under ix_lds 1 (ds41_index_score_lds_kernel) against 0 (the stock WMMA form): the whole plane, a
 *          64-row run (> 48: the WMMA forms) over MXFP4 keys, ratio 1 / 2, one sequence; two sequences in a run (the
 *          mixed-workgroup path) with padded rows. Mutation: the 8-step chain reversed must differ.
 *   IX-NI  under ix_ni 1 against 0: index_score's every row below its reach (WMMA, WMMA with keep, the scalar decode form
 *          with and without keep, padded rows), nothing written past it, index_topk's selection and index_candidates' keep
 *          over the two planes (index_candidates: the card only, r4d_selftest --case ix_ni -- its partial-wave ballot does
 *          not run here). Mutation: the last live entry unwritten.
 *
 *   r4d_emu_ds41i9b [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_mhc_mix(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_mhc_mix_scratch(const RadArgs*);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs*, RadStream);
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm_mix(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_mhc_post_collapse_rmsnorm_mix_scratch(const RadArgs*);
extern "C" int r4d_rad_ds41_index_score(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_index_candidates(const RadArgs*, RadStream);
extern "C" int r4d_rad_ds41_index_topk(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs*);
extern "C" void r4d_i9b_mutate(int which);
void r4d_ix_lds_set(int v);
void r4d_ix_ni_set(int v);

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "", bool detail_always = true) {
    ++g_cases;
    if (!ok) ++g_fail;
    const bool d = !detail.empty() && (detail_always || !ok);
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what.c_str(), d ? " -- " : "", d ? detail.c_str() : "");
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

uint64_t g_rs = 0x2545F4914F6CDD1Dull;
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
RadArgs args(std::vector<RadTensor>& t, std::vector<RadParam>& p, void* sc = nullptr, int64_t sb = 0) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    a.scratch = sc;
    a.scratch_bytes = sb;
    return a;
}
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

/* ============================================================== M1 */
void m1(int64_t M, bool mutations) {
    constexpr int64_t n = 5120, hc = 4, W = hc * n, MIX = hc * (hc + 2), iters = 20;
    auto fill = [&](std::vector<float>& v, size_t cnt, double sc, double off) {
        v.resize(cnt);
        for (auto& x : v) x = (float)(gauss() * sc + off);
    };
    std::vector<float> fn9, base9, scale9, fn10, base10, scale10, norm10;
    fill(fn9, (size_t)(MIX * W), 0.01, 0);
    fill(base9, MIX, 0.5, 0);
    fill(scale9, 3, 0.1, 0.5);
    fill(fn10, (size_t)(MIX * W), 0.01, 0);
    fill(base10, MIX, 0.5, 0);
    fill(scale10, 3, 0.1, 0.5);
    fill(norm10, n, 0.2, 1.0);
    std::vector<uint16_t> h1((size_t)(M * W)), xin((size_t)(M * n));
    for (int64_t r = 0; r < M; ++r) {
        const double rs = std::ldexp(1.0, (int)(rnd() % 3) - 1);
        for (int64_t i = 0; i < W; ++i) h1[(size_t)(r * W + i)] = bf((float)(gauss() * rs));
        for (int64_t i = 0; i < n; ++i) xin[(size_t)(r * n + i)] = bf((float)(gauss() * 0.3));
    }
    std::vector<float> pre_f((size_t)(M * hc)), post((size_t)(M * hc)), comb((size_t)(M * hc * hc));
    std::vector<RadParam> mps = { pint("M", M), pint("n", n), pint("hc", hc), pint("iterations", iters), pf64("eps", 1e-6),
                                  pf64("hc_eps", 1e-6), pstr("dtype", "bf16") };
    auto mix_ts = [&](const void* h, const void* fn, const void* sc, const void* bs, void* pre, void* po, void* cb) {
        return std::vector<RadTensor>{ tens(h, RAD_BF16, { M, W }), tens(fn, RAD_F32, { MIX, W }), tens(sc, RAD_F32, { 3 }),
                                       tens(bs, RAD_F32, { MIX }), tens(pre, RAD_F32, { M, hc }), tens(po, RAD_F32, { M, hc }),
                                       tens(cb, RAD_F32, { M, hc * hc }) };
    };
    std::vector<RadTensor> t9 = mix_ts(h1.data(), fn9.data(), scale9.data(), base9.data(), pre_f.data(), post.data(),
                                       comb.data());
    RadArgs a9 = args(t9, mps);
    std::vector<float> scr((size_t)(r4d_rad_mhc_mix_scratch(&a9) / 4 + 16));
    a9.scratch = scr.data();
    a9.scratch_bytes = (int64_t)scr.size() * 4;
    if (r4d_rad_mhc_mix(&a9, nullptr) != RAD_OK) { check(false, "M1 inputs (mhc_mix)"); return; }
    struct Out {
        std::vector<uint16_t> h0, xc, y;
        std::vector<float> pre, post, comb;
    };
    auto make = [&](Out& o) {
        o.h0.assign((size_t)(M * W), 0xA5A5);
        o.xc.assign((size_t)(M * n), 0xA5A5);
        o.y.assign((size_t)(M * n), 0xA5A5);
        o.pre.assign((size_t)(M * hc), -7.0f);
        o.post = post;
        o.comb = comb;
    };
    auto pcn_ts = [&](Out& o) {
        return std::vector<RadTensor>{ tens(xin.data(), RAD_BF16, { M, n }), tens(h1.data(), RAD_BF16, { M, W }),
                                       tens(o.post.data(), RAD_F32, { M, hc }), tens(o.comb.data(), RAD_F32, { M, hc * hc }),
                                       tens(o.h0.data(), RAD_BF16, { M, W }), tens(pre_f.data(), RAD_F32, { M, hc }),
                                       tens(o.xc.data(), RAD_BF16, { M, n }), tens(norm10.data(), RAD_F32, { n }),
                                       tens(o.y.data(), RAD_BF16, { M, n }) };
    };
    std::vector<RadParam> pps = { pint("M", M), pint("n", n), pint("hc", hc), pf64("eps", 1e-6), pstr("dtype", "bf16") };
    Out R, F;
    make(R);
    std::vector<RadTensor> tp = pcn_ts(R);
    RadArgs ap = args(tp, pps);
    const int rc1 = r4d_rad_mhc_post_collapse_rmsnorm(&ap, nullptr);
    std::vector<RadTensor> tm = mix_ts(R.h0.data(), fn10.data(), scale10.data(), base10.data(), R.pre.data(), R.post.data(),
                                       R.comb.data());
    RadArgs am = args(tm, mps, scr.data(), (int64_t)scr.size() * 4);
    const int rc2 = r4d_rad_mhc_mix(&am, nullptr);
    auto fused = [&](Out& o) {
        make(o);
        std::vector<RadTensor> tf = pcn_ts(o);
        tf.push_back(tens(fn10.data(), RAD_F32, { MIX, W }));
        tf.push_back(tens(scale10.data(), RAD_F32, { 3 }));
        tf.push_back(tens(base10.data(), RAD_F32, { MIX }));
        tf.push_back(tens(o.pre.data(), RAD_F32, { M, hc }));
        RadArgs af = args(tf, mps);
        std::vector<float> fs((size_t)(r4d_rad_mhc_post_collapse_rmsnorm_mix_scratch(&af) / 4 + 16));
        af.scratch = fs.data();
        af.scratch_bytes = (int64_t)fs.size() * 4;
        return r4d_rad_mhc_post_collapse_rmsnorm_mix(&af, nullptr);
    };
    const int rc3 = fused(F);
    const long long d = ndiff(R.h0, F.h0) + ndiff(R.xc, F.xc) + ndiff(R.y, F.y) + ndiff(R.pre, F.pre) +
                        ndiff(R.post, F.post) + ndiff(R.comb, F.comb);
    check(rc1 == RAD_OK && rc2 == RAD_OK && rc3 == RAD_OK && d == 0,
          fmt("M1 mhc_post_collapse_rmsnorm_mix byte for byte vs mhc_post_collapse_rmsnorm + mhc_mix, M=%lld", (long long)M),
          fmt("rc %d %d %d; differing: out %lld xc %lld y %lld pre %lld post %lld comb %lld", rc1, rc2, rc3,
              ndiff(R.h0, F.h0), ndiff(R.xc, F.xc), ndiff(R.y, F.y), ndiff(R.pre, F.pre), ndiff(R.post, F.post),
              ndiff(R.comb, F.comb)));
    if (!mutations) return;
    Out X;
    r4d_i9b_mutate(1);
    fused(X);
    const long long m1 = ndiff(R.pre, X.pre) + ndiff(R.post, X.post) + ndiff(R.comb, X.comb);
    r4d_i9b_mutate(2);
    fused(X);
    const long long m2 = ndiff(R.y, X.y);
    r4d_i9b_mutate(0);
    check(m1 > 0, fmt("M1 mutation 1 (K1's slabs reversed) caught, M=%lld", (long long)M), "no difference", false);
    check(m2 > 0, fmt("M1 mutation 2 (K2's virtual-thread count) caught, M=%lld", (long long)M), "no difference", false);
}

/* ============================================================== IX */
constexpr int64_t kH = 32, kD = 128, kKeyB = 80;
struct Seq {
    int64_t rows, pos0, ctx;
};
struct Ix {
    int64_t R = 0, T = 0, C = 0, ratio = 1, S = 0, nblk = 0, nbt = 0;
    std::vector<uint16_t> q, w;
    std::vector<uint8_t> kb;
    std::vector<int32_t> bt, pos, cu;
    void build(int64_t R_, int64_t ratio_, const std::vector<Seq>& sq) {
        R = R_;
        ratio = ratio_;
        S = (int64_t)sq.size();
        T = 0;
        int64_t maxreach = 0;
        nbt = 0;
        for (const Seq& s : sq) {
            T += s.rows;
            maxreach = std::max(maxreach, (s.pos0 + s.rows) / ratio);
            nbt = std::max(nbt, std::max(s.ctx, (s.pos0 + s.rows) / ratio) + 8);
        }
        C = maxreach + maxreach / 32 + 300 / ratio + 1;
        cu.assign((size_t)S + 1, 0);
        bt.assign((size_t)(S * nbt), -1);
        pos.assign((size_t)T, 0);
        int64_t r0 = 0, b0 = 0;
        for (int64_t si = 0; si < S; ++si) {
            const Seq& s = sq[(size_t)si];
            for (int64_t m = 0; m < s.rows; ++m) pos[(size_t)(r0 + m)] = (int32_t)(s.pos0 + m);
            r0 += s.rows;
            cu[(size_t)si + 1] = (int32_t)r0;
            for (int64_t j = 0; j < nbt; ++j) bt[(size_t)(si * nbt + j)] = j < s.ctx && rnd() % 97 != 5 ? (int32_t)(b0 + j) : -1;
            b0 += s.ctx;
        }
        nblk = b0;
        for (auto& v : bt)
            if (v >= 0) v = (int32_t)(((int64_t)v * 7919) % nblk);
        kb.assign((size_t)(nblk * kKeyB), 0);
        for (int64_t e = 0; e < nblk; ++e) {
            uint8_t* p = &kb[(size_t)(e * kKeyB)];
            for (int i = 0; i < 64; ++i) p[i] = (uint8_t)rnd();
            for (int i = 0; i < 4; ++i) p[64 + i] = (uint8_t)(110 + rnd() % 31);
        }
        /* full bf16 mantissas over a few binades and keys over 30 binades: dots that round in f32, so a reordered chain
         * shows (the fp4 grid's q would sum exactly in any order) */
        q.resize((size_t)(T * kH * kD));
        for (auto& x : q) x = bf((float)(gauss() * std::ldexp(1.0, (int)(rnd() % 7) - 3)));
        w.resize((size_t)(T * kH));
        for (auto& x : w) x = bf((float)(gauss() * 0.5));
    }
    int64_t reach(int64_t m) const { return m < T && m < R ? ((int64_t)pos[(size_t)m] + 1) / ratio : 0; }
    int score(std::vector<float>& sc, std::vector<int32_t>* keep, int64_t kw, int64_t block) {
        std::vector<RadTensor> t = { tens(q.data(), RAD_BF16, { T, kH * kD }), tens(w.data(), RAD_BF16, { T, kH }),
                                     tens(kb.data(), RAD_BF16, { nblk, 1, ratio, kKeyB / 2 / ratio }),
                                     tens(bt.data(), RAD_I32, { S, nbt }), tens(pos.data(), RAD_I32, { T }),
                                     tens(cu.data(), RAD_I32, { S + 1 }), tens(sc.data(), RAD_F32, { R, C }) };
        if (keep) t.push_back(tens(keep->data(), RAD_I32, { R, kw }));
        std::vector<RadParam> p = { pint("M", R), pint("heads", kH), pint("dim", kD), pint("ratio", ratio), pint("width", C),
                                    pf64("wscale", 1.0 / std::sqrt((double)kD * (double)kH)), pstr("dtype", "bf16"),
                                    pint("row_part", 0), pstr("key_format", "mxfp4") };
        if (keep) p.push_back(pint("block", block));
        RadArgs a = args(t, p);
        return r4d_rad_ds41_index_score(&a, nullptr);
    }
};
int64_t kwords(int64_t C, int64_t block) { return ((C + block - 1) / block + 31) / 32; }
const uint32_t kPoison = 0x7F000000u;
std::vector<float> poisoned(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) std::memcpy(&x, &kPoison, 4);
    return v;
}

void lds(const char* what, int64_t ratio, const std::vector<Seq>& sq, int64_t R, bool mutation) {
    Ix c;
    c.build(R, ratio, sq);
    std::vector<float> s0 = poisoned((size_t)(c.R * c.C)), s1 = s0;
    r4d_ix_ni_set(0);
    r4d_ix_lds_set(0);
    const int rc0 = c.score(s0, nullptr, 0, 0);
    r4d_ix_lds_set(1);
    const int rc1 = c.score(s1, nullptr, 0, 0);
    check(rc0 == RAD_OK && rc1 == RAD_OK && ndiff(s0, s1) == 0, fmt("IX-LDS index_score byte for byte vs the stock WMMA form, %s", what),
          fmt("rc %d %d, differing %lld", rc0, rc1, ndiff(s0, s1)));
    if (mutation) {
        std::vector<float> s2 = poisoned((size_t)(c.R * c.C));
        r4d_i9b_mutate(3);
        c.score(s2, nullptr, 0, 0);
        r4d_i9b_mutate(0);
        check(ndiff(s0, s2) > 0, fmt("IX-LDS mutation 3 (the chain reversed) caught, %s", what), "no difference", false);
    }
    r4d_ix_lds_set(0);
}

void ni(const char* what, int64_t ratio, const std::vector<Seq>& sq, int64_t R, bool keep, bool cand, bool mutations) {
    Ix c;
    c.build(R, ratio, sq);
    const int64_t block = 8, kw = kwords(c.C, block), rows = std::min(c.T, c.R), topk = 64;
    std::vector<int32_t> kp((size_t)(c.R * kw), 0);
    if (keep)
        for (int64_t m = 0; m < rows; ++m) {
            const int64_t nb = (c.reach(m) + block - 1) / block;
            for (int64_t b = 0; b < nb; ++b)
                if (rnd() % 4 == 0 || b == nb - 1) kp[(size_t)(m * kw + b / 32)] |= (int32_t)(1u << (b % 32));
        }
    std::vector<float> s0 = poisoned((size_t)(c.R * c.C)), s1 = s0;
    r4d_ix_lds_set(0);
    r4d_ix_ni_set(0);
    const int rc0 = c.score(s0, keep ? &kp : nullptr, kw, block);
    r4d_ix_ni_set(1);
    const int rc1 = c.score(s1, keep ? &kp : nullptr, kw, block);
    long long live = 0, dead = 0;
    for (int64_t m = 0; m < c.R; ++m)
        for (int64_t j = 0; j < c.C; ++j) {
            const size_t i = (size_t)(m * c.C + j);
            if (j < std::min(c.reach(m), c.C)) live += std::memcmp(&s0[i], &s1[i], 4) != 0;
            else dead += std::memcmp(&s1[i], &kPoison, 4) != 0;
        }
    check(rc0 == RAD_OK && rc1 == RAD_OK && live == 0, fmt("IX-NI index_score byte for byte below every row's reach, %s", what),
          fmt("rc %d %d, differing %lld", rc0, rc1, live));
    check(dead == 0, fmt("IX-NI index_score writes nothing past a row's reach, %s", what), fmt("%lld written", dead));
    auto topk_on = [&](std::vector<float>& sc, std::vector<int32_t>& sel) {
        sel.assign((size_t)(rows * topk), -9);
        std::vector<RadTensor> t = { tens(sc.data(), RAD_F32, { rows, c.C }), tens(kp.data(), RAD_I32, { rows, kw }),
                                     tens(c.pos.data(), RAD_I32, { rows }), tens(sel.data(), RAD_I32, { rows, topk }) };
        if (!keep) { t[1].data = nullptr; t[1].rank = 0; }
        std::vector<RadParam> p = { pint("M", rows), pint("width", c.C), pint("topk", topk), pint("ratio", c.ratio) };
        if (keep) p.push_back(pint("block", block));
        RadArgs a = args(t, p);
        std::vector<uint8_t> scr((size_t)r4d_rad_ds41_index_topk_scratch(&a) + 64);
        a.scratch = scr.data();
        a.scratch_bytes = (int64_t)scr.size() - 64;
        return r4d_rad_ds41_index_topk(&a, nullptr);
    };
    std::vector<int32_t> sel0, sel1;
    r4d_ix_ni_set(0);
    const int t0 = topk_on(s0, sel0);
    r4d_ix_ni_set(1);
    const int t1 = topk_on(s1, sel1);
    check(t0 == RAD_OK && t1 == RAD_OK && ndiff(sel0, sel1) == 0, fmt("IX-NI index_topk's selection byte for byte, %s", what),
          fmt("rc %d %d, differing %lld", t0, t1, ndiff(sel0, sel1)));
    auto cand_on = [&](std::vector<float>& sc, std::vector<int32_t>& ko) {
        ko.assign((size_t)(rows * kw), 0x55555555);
        std::vector<RadTensor> t = { tens(sc.data(), RAD_F32, { rows, c.C }), tens(c.pos.data(), RAD_I32, { rows }),
                                     tens(ko.data(), RAD_I32, { rows, kw }) };
        std::vector<RadParam> p = { pint("M", rows), pint("width", c.C), pint("block", block), pint("topk_blocks", 16),
                                    pint("ratio", c.ratio) };
        RadArgs a = args(t, p);
        return r4d_rad_ds41_index_candidates(&a, nullptr);
    };
    /* index_candidates does not run on the emulator: ds_emit's ballot runs on a partial wave past the last block, which the
     * emulator takes for a wave barrier not reached (ds41idx holds a host port of it for the same reason). The card runs it
     * (r4d_selftest --case ix_ni: its keep words over the ix_ni plane and mutation 5). */
    std::vector<int32_t> k0, k1;
    constexpr bool kCandOnEmu = false;
    if (cand && kCandOnEmu) {
        r4d_ix_ni_set(0);
        const int c0 = cand_on(s0, k0);
        r4d_ix_ni_set(1);
        const int c1 = cand_on(s1, k1);
        check(c0 == RAD_OK && c1 == RAD_OK && ndiff(k0, k1) == 0,
              fmt("IX-NI index_candidates' keep words byte for byte, %s", what), fmt("rc %d %d, differing %lld", c0, c1, ndiff(k0, k1)));
    }
    if (mutations) {
        std::vector<float> s2 = poisoned((size_t)(c.R * c.C));
        r4d_i9b_mutate(4);
        c.score(s2, nullptr, 0, 0);
        r4d_i9b_mutate(0);
        long long ld = 0;
        for (int64_t m = 0; m < c.R; ++m)
            for (int64_t j = 0; j < std::min(c.reach(m), c.C); ++j)
                ld += std::memcmp(&s0[(size_t)(m * c.C + j)], &s2[(size_t)(m * c.C + j)], 4) != 0;
        check(ld > 0, fmt("IX-NI mutation 4 (the last live entry unwritten) caught, %s", what), "no difference", false);
    }
    r4d_ix_ni_set(0);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    for (int64_t M : { 1, 7, 33, 130 }) m1(M, M == 33);
    if (!quick) m1(200, false);
    auto one = [](int64_t ctx, int64_t ratio, int64_t R) {   /* a run of R rows of a prefill chunk ending at ctx */
        return std::vector<Seq>{ { R, ctx - R, ctx / ratio + 16 } };
    };
    lds("ctx 2048, ratio 1, 64 rows", 1, one(2048, 1, 64), 64, true);
    lds("ctx 3000, ratio 2, 64 rows", 2, one(3000, 2, 64), 64, false);
    lds("two sequences in a 64-row run (rows 0-24 at ~2.5k, 25-49 at ~1.1k), 14 padded rows", 1,
        { { 25, 2500, 2600 }, { 25, 1100, 1200 } }, 64, false);
    if (!quick) lds("ctx 5000, ratio 1, 80 rows", 1, one(5000, 1, 80), 80, false);
    ni("WMMA form, ctx 2048, ratio 1", 1, one(2048, 1, 64), 64, false, true, true);
    ni("WMMA form, ctx 3000, ratio 2", 2, one(3000, 2, 64), 64, false, true, false);
    ni("WMMA form with keep, ctx 3000, ratio 1", 1, one(3000, 1, 64), 64, true, false, false);
    std::vector<Seq> dec;
    for (int i = 0; i < 6; ++i) dec.push_back({ 1, 400 + 500 * i + (int64_t)(rnd() % 97), 400 + 500 * i + 200 });
    ni("the scalar decode form, 6 rows of 6 sequences, ratio 1", 1, dec, 6, false, true, false);
    ni("the scalar decode form with keep, 6 rows, ratio 1", 1, dec, 6, true, false, false);
    ni("a run of 50 rows padded to 64, ratio 2", 2, { { 50, 2000, 1100 } }, 64, false, true, false);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
