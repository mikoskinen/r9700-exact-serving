/* r4d_selftest_i9b.inc -- ds41-int9b's byte checks on one card (included by r4d_selftest.cpp before device_phase; called
 * from it). The kbench2 checks (12.11.2026, queue-ar/dsmain/kb/bench) ported to the plugin's own rows, each new form held
 * to the stock form it replaces BYTE FOR BYTE, with mutation controls (r4d_i9b_mutate: a deliberately NOT exact variant
 * that a check must catch) and the stock / new form times as info lines:
 *
 *   --case mhc_pcn_mix (= the row mhc_post_collapse_rmsnorm_mix_bf16): lever M1, mhc_post_collapse_rmsnorm_mix vs
 *       mhc_post_collapse_rmsnorm + mhc_mix (the engine's buffers: post / comb read by the first, written by the second)
 *       at M 1 .. 4096 -- out, xc, y, pre, post, comb; mutations 1 (K1's slabs reversed) and 2 (K2's thread count).
 *   --case ix_lds (= ds41_index_score_d128_bf16): lever IX-LDSR2, index_score under fn4 ix_lds 1 vs 0 -- the whole
 *       [M, C] plane -- at 16k / 64k / 136k x ratio 1 / 2 (a 256-row run, MXFP4 keys, 32 heads), a run across two
 *       sequences with padded rows (the mixed-workgroup path); mutation 3 (the WMMA chain reversed).
 *   --case ix_ni (= ds41_index_candidates): lever IX-NI, under fn4 ix_ni 1 vs 0 -- index_score's [0, reach) of every row
 *       (WMMA, WMMA with keep, the scalar decode form with and without keep, padded rows), the dead region left as it
 *       was (poisoned beforehand), index_topk's selections and index_candidates' keep words over the two planes;
 *       mutations 4 (the last live entry unwritten) and 5 (candidates reading the full width of an ix_ni plane). */

namespace i9b {

static uint64_t g_rs = 0x2545F4914F6CDD1Dull;
static uint64_t rnd() {
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double uni() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
static double gauss() {
    double u1 = uni(), u2 = uni();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}
static uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
template <class V>
static void* up(const std::vector<V>& v) {
    void* d = dalloc(v.size() * sizeof(V));
    if (d) st_memcpy(d, v.data(), v.size() * sizeof(V), hipMemcpyHostToDevice);
    return d;
}
template <class V>
static std::vector<V> down(const void* d, size_t n) {
    std::vector<V> v(n);
    st_memcpy(v.data(), d, n * sizeof(V), hipMemcpyDeviceToHost);
    return v;
}
/* Words that differ between two device regions of `bytes` (a multiple of 2). */
static long long diff16(const void* a, const void* b, size_t bytes) {
    std::vector<uint16_t> x = down<uint16_t>(a, bytes / 2), y = down<uint16_t>(b, bytes / 2);
    long long d = 0;
    for (size_t i = 0; i < x.size(); ++i) d += x[i] != y[i];
    return d;
}
static RadArgs args(std::vector<RadTensor>& ts, std::vector<RadParam>& ps, void* sc = nullptr, int64_t sb = 0) {
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data();
    a.n_t = (int)ts.size();
    a.p = ps.data();
    a.n_p = (int)ps.size();
    a.world_size = 1;
    a.scratch = sc;
    a.scratch_bytes = sb;
    return a;
}
/* One launch, synchronised; its rc. */
static int run(const RadKernelInfo* r, std::vector<RadTensor>& ts, std::vector<RadParam>& ps, void* sc = nullptr,
               int64_t sb = 0) {
    RadArgs a = args(ts, ps, sc, sb);
    const int rc = r->launch(&a, (RadStream) nullptr);
    st_stream_sync(nullptr);
    return rc;
}
static void* scratch_for(const RadKernelInfo* r, std::vector<RadTensor>& ts, std::vector<RadParam>& ps, int64_t* sb) {
    *sb = 0;
    if (!r->scratch) return nullptr;
    RadArgs a = args(ts, ps);
    *sb = r->scratch(&a);
    return *sb > 0 ? dalloc((size_t)*sb) : nullptr;
}
static void info(const std::string& s) { std::printf("  info  %s\n", s.c_str()); }

/* ======================================================================= lever M1 */
static void m1_case(const Plugin& pl) {
    const RadKernelInfo* rmix = find(pl, "mhc_mix_bf16");
    const RadKernelInfo* rpcn = find(pl, "mhc_post_collapse_rmsnorm_bf16");
    const RadKernelInfo* rfus = find(pl, "mhc_post_collapse_rmsnorm_mix_bf16");
    auto mutate = (void (*)(int))dlsym(pl.h, "r4d_i9b_mutate");
    if (!rmix || !rpcn || !rfus || !mutate) {
        ok("mhc_post_collapse_rmsnorm_mix rows", false, "a row or r4d_i9b_mutate absent");
        return;
    }
    constexpr int64_t n = 5120, hc = 4, W = hc * n, MIX = hc * (hc + 2), iters = 20;
    const double eps = 1e-6, heps = 1e-6;
    g_rs = 0x5EED0001ull;
    auto fill = [&](std::vector<float>& v, size_t cnt, double sc, double off) {
        v.resize(cnt);
        for (auto& x : v) x = (float)(gauss() * sc + off);
    };
    std::vector<float> fn9, base9, scale9, fn10, base10, scale10, norm10;
    fill(fn9, (size_t)MIX * W, 0.01, 0);
    fill(base9, MIX, 0.5, 0);
    fill(scale9, 3, 0.1, 0.5);
    fill(fn10, (size_t)MIX * W, 0.01, 0);
    fill(base10, MIX, 0.5, 0);
    fill(scale10, 3, 0.1, 0.5);
    fill(norm10, n, 0.2, 1.0);
    const int64_t Ms[] = { 1, 8, 33, 130, 512, 1000, 4096 };
    for (int64_t M : Ms) {
        void *dfn9 = up(fn9), *dbase9 = up(base9), *dscale9 = up(scale9);
        void *dfn10 = up(fn10), *dbase10 = up(base10), *dscale10 = up(scale10), *dnorm = up(norm10);
        std::vector<uint16_t> h1((size_t)(M * W)), xin((size_t)(M * n));
        for (int64_t r = 0; r < M; ++r) {
            const double rs = std::ldexp(1.0, (int)(rnd() % 3) - 1);
            for (int64_t i = 0; i < W; ++i) h1[(size_t)(r * W + i)] = bf((float)(gauss() * rs));
            for (int64_t i = 0; i < n; ++i) xin[(size_t)(r * n + i)] = bf((float)(gauss() * 0.3));
        }
        void *dh1 = up(h1), *dxin = up(xin);
        void *pre_f = dalloc(M * hc * 4), *post = dalloc(M * hc * 4), *comb = dalloc(M * hc * hc * 4);
        if (!dfn9 || !dh1 || !pre_f || !post || !comb) { ok("mhc_pcn_mix alloc", false, "hipMalloc failed"); dfree_all(); return; }
        auto mix_ts = [&](void* h, void* fn, void* sc, void* bs, void* pre, void* po, void* cb) {
            return std::vector<RadTensor>{ T(h, RAD_BF16, { M, W }), T(fn, RAD_F32, { MIX, W }), T(sc, RAD_F32, { 3 }),
                                           T(bs, RAD_F32, { MIX }), T(pre, RAD_F32, { M, hc }), T(po, RAD_F32, { M, hc }),
                                           T(cb, RAD_F32, { M, hc * hc }) };
        };
        std::vector<RadParam> mps = { PI("M", M), PI("n", n), PI("hc", hc), PI("iterations", iters), PF("eps", eps),
                                      PF("hc_eps", heps), PS("dtype", "bf16") };
        /* the previous connection's gates (pcn's inputs): the stock mhc_mix over h1 with "blk.9"'s weights */
        std::vector<RadTensor> t9 = mix_ts(dh1, dfn9, dscale9, dbase9, pre_f, post, comb);
        int64_t sb9 = 0;
        void* scr = scratch_for(rmix, t9, mps, &sb9);
        if (run(rmix, t9, mps, scr, sb9) != RAD_OK) { ok("mhc_pcn_mix inputs (mhc_mix)", false, "rc"); dfree_all(); return; }
        const size_t bh = (size_t)(M * W * 2), bx = (size_t)(M * n * 2), bp = (size_t)(M * hc * 4), bc = (size_t)(M * hc * hc * 4);
        /* the two forms' outputs: R (the stock pair) and F (the fused op); post / comb start as the inputs in both */
        void *h0R = dalloc(bh), *xcR = dalloc(bx), *yR = dalloc(bx), *preR = dalloc(bp), *postR = dalloc(bp), *combR = dalloc(bc);
        void *h0F = dalloc(bh), *xcF = dalloc(bx), *yF = dalloc(bx), *preF = dalloc(bp), *postF = dalloc(bp), *combF = dalloc(bc);
        if (!combF) { ok("mhc_pcn_mix alloc", false, "hipMalloc failed"); dfree_all(); return; }
        auto pcn_ts = [&](void* po, void* cb, void* h0, void* xc, void* y) {
            return std::vector<RadTensor>{ T(dxin, RAD_BF16, { M, n }), T(dh1, RAD_BF16, { M, W }), T(po, RAD_F32, { M, hc }),
                                           T(cb, RAD_F32, { M, hc * hc }), T(h0, RAD_BF16, { M, W }), T(pre_f, RAD_F32, { M, hc }),
                                           T(xc, RAD_BF16, { M, n }), T(dnorm, RAD_F32, { n }), T(y, RAD_BF16, { M, n }) };
        };
        std::vector<RadParam> pps = { PI("M", M), PI("n", n), PI("hc", hc), PF("eps", eps), PS("dtype", "bf16") };
        std::vector<RadParam> fps = { PI("M", M), PI("n", n), PI("hc", hc), PI("iterations", iters), PF("eps", eps),
                                      PF("hc_eps", heps), PS("dtype", "bf16") };
        std::vector<RadTensor> tp = pcn_ts(postR, combR, h0R, xcR, yR);
        std::vector<RadTensor> tm = mix_ts(h0R, dfn10, dscale10, dbase10, preR, postR, combR);
        std::vector<RadTensor> tf = pcn_ts(postF, combF, h0F, xcF, yF);
        tf.push_back(T(dfn10, RAD_F32, { MIX, W }));
        tf.push_back(T(dscale10, RAD_F32, { 3 }));
        tf.push_back(T(dbase10, RAD_F32, { MIX }));
        tf.push_back(T(preF, RAD_F32, { M, hc }));
        int64_t sbf = 0;
        void* scf = scratch_for(rfus, tf, fps, &sbf);
        auto reset = [&](bool fused) {   /* post / comb back to the inputs, the outputs poisoned */
            void* po = fused ? postF : postR;
            void* cb = fused ? combF : combR;
            st_memcpy(po, post, bp, hipMemcpyDeviceToDevice);
            st_memcpy(cb, comb, bc, hipMemcpyDeviceToDevice);
            st_memset(fused ? h0F : h0R, 0xA5, bh);
            st_memset(fused ? xcF : xcR, 0xA5, bx);
            st_memset(fused ? yF : yR, 0xA5, bx);
            st_memset(fused ? preF : preR, 0xA5, bp);
        };
        reset(false);
        const int rc1 = run(rpcn, tp, pps), rc2 = run(rmix, tm, mps, scr, sb9);
        reset(true);
        const int rc3 = run(rfus, tf, fps, scf, sbf);
        if (rc1 != RAD_OK || rc2 != RAD_OK || rc3 != RAD_OK) {
            ok(fmt("mhc_post_collapse_rmsnorm_mix M=%lld launches", (long long)M).c_str(), false,
               fmt("rc pcn %d mix %d fused %d", rc1, rc2, rc3));
            dfree_all();
            continue;
        }
        const long long d_h0 = diff16(h0R, h0F, bh), d_xc = diff16(xcR, xcF, bx), d_y = diff16(yR, yF, bx);
        const long long d_pre = diff16(preR, preF, bp), d_po = diff16(postR, postF, bp), d_cb = diff16(combR, combF, bc);
        ok(fmt("mhc_post_collapse_rmsnorm_mix byte for byte vs mhc_post_collapse_rmsnorm + mhc_mix, M=%lld (out, xc, y, "
               "pre, post, comb)", (long long)M).c_str(),
           d_h0 + d_xc + d_y + d_pre + d_po + d_cb == 0,
           fmt("differing 16-bit words: out %lld xc %lld y %lld pre %lld post %lld comb %lld", d_h0, d_xc, d_y, d_pre, d_po,
               d_cb));
        if (M == 1000) {   /* mutation controls: each must be caught */
            mutate(1);
            reset(true);
            run(rfus, tf, fps, scf, sbf);
            const long long m1 = diff16(preR, preF, bp) + diff16(postR, postF, bp) + diff16(combR, combF, bc);
            mutate(2);
            reset(true);
            run(rfus, tf, fps, scf, sbf);
            const long long m2 = diff16(yR, yF, bx);
            mutate(0);
            ok(fmt("mutation 1 (K1's slabs in reverse order: the mix's partial order) caught, M=%lld", (long long)M).c_str(),
               m1 > 0, "pre/post/comb identical: the check would not see a reordered projection");
            ok(fmt("mutation 2 (K2 at another virtual-thread count: the norm's order) caught, M=%lld", (long long)M).c_str(),
               m2 > 0, "y identical: the check would not see a reordered norm");
        }
        if (M >= 512) {
            const int it = M >= 4096 ? 20 : 50;
            RadArgs ap = args(tp, pps), am = args(tm, mps, scr, sb9), af = args(tf, fps, scf, sbf);
            const double tpair = st_time_us([&](int) { rpcn->launch(&ap, nullptr); rmix->launch(&am, nullptr); }, it, nullptr);
            const double tfus = st_time_us([&](int) { rfus->launch(&af, nullptr); }, it, nullptr);
            info(fmt("mhc_pcn_mix M=%lld: pair (mhc_post_collapse_rmsnorm + mhc_mix) %.1f us, fused %.1f us (%+.1f %%)",
                     (long long)M, tpair, tfus, 100.0 * (tfus - tpair) / tpair));
        }
        dfree_all();
    }
}

/* ======================================================================= levers IX-LDSR2 / IX-NI */
constexpr int64_t kH = 32, kD = 128, kKeyB = 80;   /* heads, dim, MXFP4 key bytes */

struct IxSeq {
    int64_t rows, pos0, ctx;   /* its rows in the run, the first row's position, the entries mapped (>= reach) */
};
/* One indexer pass on the card: a run of R plane rows over a step of T (= sum of rows) rows of one or more sequences,
 * their keys in an MXFP4 cache (an entry a block, block size = ratio), ~1 % of the entries unmapped. */
struct IxCase {
    int64_t R = 0, T = 0, C = 0, ratio = 1, S = 0, nblk = 0, nbt = 0;
    void *q = nullptr, *w = nullptr, *kc = nullptr, *bt = nullptr, *pos = nullptr, *cu = nullptr;
    std::vector<int32_t> hpos;
    bool build(int64_t R_, int64_t ratio_, const std::vector<IxSeq>& sq) {
        R = R_;
        ratio = ratio_;
        S = (int64_t)sq.size();
        T = 0;
        int64_t maxreach = 0;
        nbt = 0;
        for (const IxSeq& s : sq) {
            T += s.rows;
            const int64_t reach = (s.pos0 + s.rows) / ratio;
            maxreach = std::max(maxreach, reach);
            nbt = std::max(nbt, std::max(s.ctx, reach) + 8);
        }
        /* the plane's width: the step's live bound, ~3 % of bucket slack past its furthest reach (deepseek41 `live`) */
        C = maxreach + maxreach / 32 + 2048 / ratio + 1;
        std::vector<int32_t> cuv((size_t)S + 1, 0), btv((size_t)(S * nbt), -1);
        hpos.assign((size_t)T, 0);
        int64_t r0 = 0, b0 = 0;
        for (int64_t si = 0; si < S; ++si) {
            const IxSeq& s = sq[(size_t)si];
            for (int64_t m = 0; m < s.rows; ++m) hpos[(size_t)(r0 + m)] = (int32_t)(s.pos0 + m);
            r0 += s.rows;
            cuv[(size_t)si + 1] = (int32_t)r0;
            for (int64_t j = 0; j < nbt; ++j)
                btv[(size_t)(si * nbt + j)] = j < s.ctx && rnd() % 97 != 5 ? (int32_t)(b0 + j) : -1;
            b0 += s.ctx;
        }
        nblk = b0;
        /* the block table points into a shuffled cache: entry j of sequence s -> block (b0 + j) * 7919 % nblk */
        for (auto& v : btv)
            if (v >= 0) v = (int32_t)(((int64_t)v * 7919) % nblk);
        std::vector<uint8_t> kb((size_t)(nblk * kKeyB), 0);
        for (int64_t e = 0; e < nblk; ++e) {
            uint8_t* p = &kb[(size_t)(e * kKeyB)];
            for (int i = 0; i < 64; ++i) p[i] = (uint8_t)rnd();
            for (int i = 0; i < 4; ++i) p[64 + i] = (uint8_t)(122 + rnd() % 9);
        }
        std::vector<uint16_t> qv((size_t)(T * kH * kD)), wv((size_t)(T * kH));
        for (auto& x : qv) {   /* e2m1-like values times a 32-block scale (the fp4 round trip's grid) */
            static const float g[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            x = bf((rnd() & 8 ? -1.0f : 1.0f) * g[rnd() & 7] * std::ldexp(1.0f, (int)(rnd() % 5) - 4));
        }
        for (auto& x : wv) x = bf((float)(gauss() * 0.5));
        q = up(qv);
        w = up(wv);
        kc = up(kb);
        bt = up(btv);
        pos = up(hpos);
        cu = up(cuv);
        return q && w && kc && bt && pos && cu;
    }
    int64_t reach(int64_t m) const { return m < T && m < R ? ((int64_t)hpos[(size_t)m] + 1) / ratio : 0; }
    std::vector<RadTensor> score_ts(void* sc, void* keep, int64_t kw) const {
        std::vector<RadTensor> t = { T_(q, RAD_BF16, { T, kH * kD }), T_(w, RAD_BF16, { T, kH }),
                                     T_(kc, RAD_BF16, { nblk, 1, ratio, kKeyB / 2 / ratio }), T_(bt, RAD_I32, { S, nbt }),
                                     T_(pos, RAD_I32, { T }), T_(cu, RAD_I32, { S + 1 }), T_(sc, RAD_F32, { R, C }) };
        if (keep) t.push_back(T_(keep, RAD_I32, { R, kw }));
        return t;
    }
    std::vector<RadParam> score_ps(int64_t block) const {
        std::vector<RadParam> p = { PI("M", R), PI("heads", kH), PI("dim", kD), PI("ratio", ratio), PI("width", C),
                                    PF("wscale", 1.0 / std::sqrt((double)kD * (double)kH)), PS("dtype", "bf16"),
                                    PI("row_part", 0), PS("key_format", "mxfp4") };
        if (block > 0) p.push_back(PI("block", block));
        return p;
    }
    static RadTensor T_(void* d, uint32_t dt, std::vector<int64_t> s) { return ::T(d, dt, std::move(s)); }
};
static int64_t kwords(int64_t C, int64_t block) { return ((C + block - 1) / block + 31) / 32; }

struct IxHooks {
    int (*sw)(const char*, int) = nullptr;
    void (*mutate)(int) = nullptr;
    const RadKernelInfo *score = nullptr, *cand = nullptr, *topk = nullptr;
    bool get(const Plugin& pl) {
        sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
        mutate = (void (*)(int))dlsym(pl.h, "r4d_i9b_mutate");
        score = find(pl, "ds41_index_score_d128_bf16");
        cand = find(pl, "ds41_index_candidates");
        topk = find(pl, "ds41_index_topk");
        return sw && mutate && score && cand && topk;
    }
};

static void lds_case(const Plugin& pl) {
    IxHooks hk;
    if (!hk.get(pl)) { ok("ix_lds hooks", false, "rad_fn4_switch / r4d_i9b_mutate / an index row absent"); return; }
    if (hk.sw("ix_lds", 1) != 1 || hk.sw("ix_lds", 0) != 1) { ok("ix_lds switch", false, "ix_lds not taken"); return; }
    hk.sw("ix_ni", 0);
    g_rs = 0x5EED0002ull;
    struct LC { const char* what; int64_t ratio; std::vector<IxSeq> sq; int64_t R; bool timed; };
    auto run7 = [](int64_t ctx, int64_t ratio) {   /* run 7 of a 4096-row prefill chunk ending at ctx (kbench2's) */
        return std::vector<IxSeq>{ { 256, ctx - 4096 + 7 * 256, ctx / ratio + 16 } };
    };
    const LC cases[] = {
        { "ctx 16384, ratio 1", 1, run7(16384, 1), 256, true },  { "ctx 16384, ratio 2", 2, run7(16384, 2), 256, true },
        { "ctx 65536, ratio 1", 1, run7(65536, 1), 256, true },  { "ctx 65536, ratio 2", 2, run7(65536, 2), 256, true },
        { "ctx 139264, ratio 1", 1, run7(139264, 1), 256, true }, { "ctx 139264, ratio 2", 2, run7(139264, 2), 256, true },
        { "two sequences in a run (rows 0-99 at ctx ~30k, 100-199 at ~9k), 56 padded rows, ratio 1", 1,
          { { 100, 30000, 30200 }, { 100, 9000, 9200 } }, 256, false },
    };
    for (const LC& lc : cases) {
        IxCase c;
        if (!c.build(lc.R, lc.ratio, lc.sq)) { ok("ix_lds alloc", false, "hipMalloc failed"); dfree_all(); return; }
        const size_t bytes = (size_t)(c.R * c.C * 4);
        void *s0 = dalloc(bytes), *s1 = dalloc(bytes);
        if (!s0 || !s1) { ok("ix_lds alloc", false, "hipMalloc failed"); dfree_all(); return; }
        std::vector<RadTensor> t0 = c.score_ts(s0, nullptr, 0), t1 = c.score_ts(s1, nullptr, 0);
        std::vector<RadParam> ps = c.score_ps(0);
        st_memset(s0, 0x7F, bytes);
        st_memset(s1, 0x7F, bytes);
        hk.sw("ix_lds", 0);
        const int rc0 = run(hk.score, t0, ps);
        hk.sw("ix_lds", 1);
        const int rc1 = run(hk.score, t1, ps);
        const long long d = diff16(s0, s1, bytes);
        ok(fmt("index_score ix_lds byte for byte vs the stock WMMA form, %s (the whole %lld x %lld plane)", lc.what,
               (long long)c.R, (long long)c.C).c_str(),
           rc0 == RAD_OK && rc1 == RAD_OK && d == 0, fmt("rc %d / %d, differing 16-bit words %lld", rc0, rc1, d));
        if (lc.ratio == 1 && lc.sq[0].pos0 == 65536 - 4096 + 7 * 256) {   /* the mutation control at 64k */
            hk.mutate(3);
            st_memset(s1, 0x7F, bytes);
            run(hk.score, t1, ps);
            hk.mutate(0);
            const long long dm = diff16(s0, s1, bytes);
            ok("mutation 3 (the LDS form's WMMA chain in reverse order) caught, ctx 65536 ratio 1", dm > 0,
               "the plane identical: the check would not see another dot order");
        }
        if (lc.timed) {
            RadArgs a0 = args(t0, ps), a1 = args(t1, ps);
            hk.sw("ix_lds", 0);
            const double u0 = st_time_us([&](int) { hk.score->launch(&a0, nullptr); }, 10, nullptr);
            hk.sw("ix_lds", 1);
            const double u1 = st_time_us([&](int) { hk.score->launch(&a1, nullptr); }, 10, nullptr);
            info(fmt("index_score %s, 256-row run: stock %.0f us, ix_lds %.0f us (%+.1f %%)", lc.what, u0, u1,
                     100.0 * (u1 - u0) / u0));
        }
        hk.sw("ix_lds", 0);
        dfree_all();
    }
}

/* Every row's [0, min(reach, C)) of two planes byte for byte, and how many words past it differ from the poison. */
static void ni_compare(const IxCase& c, int64_t rows, const void* s0, const void* s1, uint32_t poison, long long* live_diff,
                       long long* dead_written) {
    std::vector<uint32_t> a = down<uint32_t>(s0, (size_t)(c.R * c.C)), b = down<uint32_t>(s1, (size_t)(c.R * c.C));
    *live_diff = 0;
    *dead_written = 0;
    for (int64_t m = 0; m < rows; ++m) {
        const int64_t r = std::min(c.reach(m), c.C);
        for (int64_t j = 0; j < c.C; ++j) {
            const size_t i = (size_t)(m * c.C + j);
            if (j < r) *live_diff += a[i] != b[i];
            else *dead_written += b[i] != poison;
        }
    }
}

static void ni_case(const Plugin& pl) {
    IxHooks hk;
    if (!hk.get(pl)) { ok("ix_ni hooks", false, "rad_fn4_switch / r4d_i9b_mutate / an index row absent"); return; }
    if (hk.sw("ix_ni", 1) != 1 || hk.sw("ix_ni", 0) != 1) { ok("ix_ni switch", false, "ix_ni not taken"); return; }
    hk.sw("ix_lds", 0);
    g_rs = 0x5EED0003ull;
    const uint32_t poison = 0x7F000000u;   /* 1.7e38: a dead word read as a score would win every selection */
    const int64_t topk = 512, block = 8, kblocks = 64;
    struct NC { const char* what; int64_t ratio, R; std::vector<IxSeq> sq; bool keep, cand, mut; };
    auto run7 = [](int64_t ctx, int64_t ratio) { return std::vector<IxSeq>{ { 256, ctx - 4096 + 7 * 256, ctx / ratio + 16 } }; };
    std::vector<IxSeq> dec;   /* 8 decode rows of 8 sequences at 3k .. 60k */
    for (int i = 0; i < 8; ++i) dec.push_back({ 1, 3000 + 8000 * i + (int64_t)(rnd() % 997), (3000 + 8000 * i + 1000) });
    const NC cases[] = {
        { "WMMA form, ctx 16384, ratio 1", 1, 256, run7(16384, 1), false, true, true },
        { "WMMA form, ctx 16384, ratio 2", 2, 256, run7(16384, 2), false, false, false },
        { "WMMA form, ctx 65536, ratio 1", 1, 256, run7(65536, 1), false, true, false },
        { "WMMA form, ctx 65536, ratio 2", 2, 256, run7(65536, 2), false, false, false },
        { "WMMA form with keep, ctx 65536, ratio 1", 1, 256, run7(65536, 1), true, false, false },
        { "the scalar decode form, 8 rows of 8 sequences at 3k .. 60k, ratio 1", 1, 8, dec, false, true, false },
        { "the scalar decode form with keep, 8 rows, ratio 1", 1, 8, dec, true, false, false },
        { "a run of 200 rows padded to 256 (rows past the step), ratio 2", 2, 256, { { 200, 20000, 10200 } }, false, true,
          false },
    };
    for (const NC& nc : cases) {
        IxCase c;
        if (!c.build(nc.R, nc.ratio, nc.sq)) { ok("ix_ni alloc", false, "hipMalloc failed"); dfree_all(); return; }
        const int64_t rows = std::min(c.T, c.R), kw = kwords(c.C, block);
        const size_t bytes = (size_t)(c.R * c.C * 4);
        void *s0 = dalloc(bytes), *s1 = dalloc(bytes);
        void* keep = nc.keep ? dalloc((size_t)(c.R * kw * 4)) : nullptr;
        void *sel0 = dalloc((size_t)(rows * topk * 4)), *sel1 = dalloc((size_t)(rows * topk * 4));
        void *k0 = dalloc((size_t)(rows * kw * 4)), *k1 = dalloc((size_t)(rows * kw * 4));
        if (!s0 || !s1 || !sel0 || !sel1 || !k0 || !k1 || (nc.keep && !keep)) {
            ok("ix_ni alloc", false, "hipMalloc failed");
            dfree_all();
            return;
        }
        auto fillp = [&](void* p) {
            std::vector<uint32_t> v((size_t)(c.R * c.C), poison);
            st_memcpy(p, v.data(), bytes, hipMemcpyHostToDevice);
        };
        if (nc.keep) {   /* the candidate layer's keep: random blocks below reach, the newest pinned */
            std::vector<int32_t> kv((size_t)(c.R * kw), 0);
            for (int64_t m = 0; m < rows; ++m) {
                const int64_t nb = (c.reach(m) + block - 1) / block;
                for (int64_t b = 0; b < nb; ++b)
                    if (rnd() % 4 == 0 || b == nb - 1) kv[(size_t)(m * kw + b / 32)] |= (int32_t)(1u << (b % 32));
            }
            st_memcpy(keep, kv.data(), kv.size() * 4, hipMemcpyHostToDevice);
        }
        std::vector<RadTensor> t0 = c.score_ts(s0, keep, kw), t1 = c.score_ts(s1, keep, kw);
        std::vector<RadParam> ps = c.score_ps(nc.keep ? block : 0);
        fillp(s0);
        fillp(s1);
        hk.sw("ix_ni", 0);
        const int rc0 = run(hk.score, t0, ps);
        hk.sw("ix_ni", 1);
        const int rc1 = run(hk.score, t1, ps);
        long long ld = 0, dw = 0;
        ni_compare(c, c.R, s0, s1, poison, &ld, &dw);
        ok(fmt("index_score ix_ni byte for byte below every row's reach vs the stock op, %s", nc.what).c_str(),
           rc0 == RAD_OK && rc1 == RAD_OK && ld == 0, fmt("rc %d / %d, differing words below reach %lld", rc0, rc1, ld));
        ok(fmt("index_score ix_ni writes nothing past a row's reach, %s", nc.what).c_str(), dw == 0,
           fmt("%lld dead words written", dw));
        /* index_topk over the two planes (its rows: the step's), the keep where the layer reads one */
        auto topk_on = [&](void* sc, void* sel) {
            std::vector<RadTensor> t = { T(sc, RAD_F32, { rows, c.C }), T(keep, RAD_I32, { rows, kw }),
                                         T(c.pos, RAD_I32, { rows }), T(sel, RAD_I32, { rows, topk }) };
            if (!keep) absent(t[1]);
            t[0].stride[0] = c.C;
            std::vector<RadParam> p = { PI("M", rows), PI("width", c.C), PI("topk", topk), PI("ratio", c.ratio) };
            if (keep) p.push_back(PI("block", block));
            int64_t sb = 0;
            void* scr = scratch_for(hk.topk, t, p, &sb);
            return run(hk.topk, t, p, scr, sb);
        };
        hk.sw("ix_ni", 0);
        const int rt0 = topk_on(s0, sel0);
        hk.sw("ix_ni", 1);
        const int rt1 = topk_on(s1, sel1);
        const long long dsel = diff16(sel0, sel1, (size_t)(rows * topk * 4));
        ok(fmt("index_topk's selection byte for byte over the ix_ni plane, %s", nc.what).c_str(),
           rt0 == RAD_OK && rt1 == RAD_OK && dsel == 0, fmt("rc %d / %d, differing words %lld", rt0, rt1, dsel));
        auto cand_on = [&](void* sc, void* kout) {
            std::vector<RadTensor> t = { T(sc, RAD_F32, { rows, c.C }), T(c.pos, RAD_I32, { rows }),
                                         T(kout, RAD_I32, { rows, kw }) };
            std::vector<RadParam> p = { PI("M", rows), PI("width", c.C), PI("block", block), PI("topk_blocks", kblocks),
                                        PI("ratio", c.ratio) };
            return run(hk.cand, t, p);
        };
        if (nc.cand) {
            st_memset(k0, 0x55, (size_t)(rows * kw * 4));
            st_memset(k1, 0x55, (size_t)(rows * kw * 4));
            hk.sw("ix_ni", 0);
            const int rk0 = cand_on(s0, k0);
            hk.sw("ix_ni", 1);
            const int rk1 = cand_on(s1, k1);
            const long long dk = diff16(k0, k1, (size_t)(rows * kw * 4));
            ok(fmt("index_candidates' keep words byte for byte over the ix_ni plane (read below reach), %s", nc.what).c_str(),
               rk0 == RAD_OK && rk1 == RAD_OK && dk == 0, fmt("rc %d / %d, differing words %lld", rk0, rk1, dk));
        }
        if (nc.mut) {   /* mutation controls on the first case */
            hk.mutate(4);
            fillp(s1);
            run(hk.score, t1, ps);
            hk.mutate(0);
            long long ld4 = 0, dw4 = 0;
            ni_compare(c, c.R, s0, s1, poison, &ld4, &dw4);
            ok(fmt("mutation 4 (ix_ni leaving each row's last live entry unwritten) caught, %s", nc.what).c_str(), ld4 > 0,
               "below reach identical: the check would not see a lost entry");
            fillp(s1);
            run(hk.score, t1, ps);   /* the ix_ni plane again */
            hk.mutate(5);
            st_memset(k1, 0x55, (size_t)(rows * kw * 4));
            cand_on(s1, k1);
            hk.mutate(0);
            st_memset(k0, 0x55, (size_t)(rows * kw * 4));
            hk.sw("ix_ni", 0);
            cand_on(s0, k0);
            hk.sw("ix_ni", 1);
            const long long dk5 = diff16(k0, k1, (size_t)(rows * kw * 4));
            ok(fmt("mutation 5 (index_candidates reading the full width of an ix_ni plane) caught, %s", nc.what).c_str(),
               dk5 > 0, "keep identical: the check would not see dead words read");
        }
        if (nc.R == 256 && !nc.keep && c.T == c.R) {
            RadArgs a0 = args(t0, ps), a1 = args(t1, ps);
            hk.sw("ix_ni", 0);
            const double u0 = st_time_us([&](int) { hk.score->launch(&a0, nullptr); }, 10, nullptr);
            hk.sw("ix_ni", 1);
            const double u1 = st_time_us([&](int) { hk.score->launch(&a1, nullptr); }, 10, nullptr);
            info(fmt("index_score %s (plane width %lld = the step's live bound), 256-row run: stock %.0f us, ix_ni %.0f us "
                     "(%+.1f %%)", nc.what, (long long)c.C, u0, u1, 100.0 * (u1 - u0) / u0));
        }
        hk.sw("ix_ni", 0);
        dfree_all();
    }
}

}  // namespace i9b

static void i9b_cases(const Plugin& pl) {
    if (sel("mhc_pcn_mix") || sel("mhc_post_collapse_rmsnorm_mix_bf16")) i9b::m1_case(pl);
    if (sel("ix_lds") || sel("ds41_index_score_d128_bf16")) i9b::lds_case(pl);
    if (sel("ix_ni") || sel("ds41_index_candidates")) i9b::ni_case(pl);
}
