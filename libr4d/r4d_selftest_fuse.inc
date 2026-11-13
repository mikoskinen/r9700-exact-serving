/* r4d_selftest_fuse.inc -- ds41-fuse's one-card checks (included by r4d_selftest.cpp before device_phase; called from it):
 * every fused form held BYTE FOR BYTE to the stock launches it replaces, then timed per call against them (the chain's
 * launches back to back on one stream, so its dependent launch gaps are in its time):
 *
 *   --case rfuse (= the row router_gemm_f32_scored_topk_scatter_bf16): lever FUSE-R at DeepSeek-V4.1's router (K 5120, 384
 *       experts, top 6, sqrtsoftplus, norm, a 96-expert TP4 slice at bases 0 / 96 / 288): the stock chain gemm_nt_f32 ->
 *       router_scored_topk -> moe_scatter vs form 1 (gemm_nt_f32 -> router_scored_topk_scatter, no gate) vs form 2
 *       (router_gemm_f32_scored_topk_scatter, one launch) at M 1 .. 16, 24, 32, 40, 48, bf16 and f32 projections, random,
 *       tied (duplicated weight rows: equal logits, the lower expert must win) and NaN rows (no finite rank: expert 0 in
 *       every slot) -- logits, ids, weights, sorted_tok, expert_offset, expert_count, each with a guard tail; form 2
 *       launched twice in a row (the arrival counter must come back to zero); FUSEPERF lines at M 1 / 8 / 16 / 48 (warm =
 *       back to back, mean; cold = each launch after a 256 MiB read, median) and a FUSEGATE verdict per form: KEEP when
 *       the fused form is not slower than the chain (cold, M 1 / 8 / 16), else DROP.
 *   --case mixfin (= no row of its own; mhc_mix_bf16 under libr4d's fn4 ds41_mix_fin): lever FUSE-M, mhc_mix with
 *       ds41_mix_fin 1 vs 0 at M 1, 2, 3, 6, 8, 9, 16, 24, 48 (FIN taken: the counter must move) and 49 (above the band:
 *       the switch must not act) -- pre, post, comb and the scratch's partials; FUSEPERF / FUSEGATE as above at M 1 / 8 /
 *       16 / 48. NOTE the engine's schedule decides this lever, not this bench: in a recorded pass the finish already runs
 *       beside the next GEMV under the barrier that GEMV needs anyway (zzzaql's stamps). */

namespace fuse {

static uint64_t g_rs = 0x9E3779B97F4A7C15ull;
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
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
template <class V>
static void* up(const std::vector<V>& v) {
    void* d = dalloc(v.size() * sizeof(V));
    if (d) st_memcpy(d, v.data(), v.size() * sizeof(V), hipMemcpyHostToDevice);
    return d;
}
static std::vector<uint8_t> bytes(const void* d, size_t n) {
    std::vector<uint8_t> v(n);
    st_memcpy(v.data(), d, n, hipMemcpyDeviceToHost);
    return v;
}
static RadArgs args(std::vector<RadTensor>& ts, std::vector<RadParam>& ps, void* inst = nullptr, void* sc = nullptr,
                    int64_t sb = 0) {
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data();
    a.n_t = (int)ts.size();
    a.p = ps.data();
    a.n_p = (int)ps.size();
    a.world_size = 1;
    a.instance = inst;
    a.scratch = sc;
    a.scratch_bytes = sb;
    return a;
}
static void info(const std::string& s) { std::printf("  info  %s\n", s.c_str()); }
/* warm (back to back, mean) and cold (256 MiB read before each launch, median) */
template <class F>
static void time2(F&& f, int iters, double* warm, double* cold) {
    const int64_t c0 = g_cold;
    g_cold = 0;
    *warm = st_time_us([&](int) { f(); }, iters, nullptr);
    g_cold = 256ll << 20;
    *cold = st_time_us([&](int) { f(); }, iters / 2 > 20 ? iters / 2 : 20, nullptr);
    g_cold = c0;
}

/* ======================================================================= lever FUSE-R */
static void rf_case(const Plugin& pl) {
    const RadKernelInfo* rg = find(pl, "gemm_nt_f32_bf16");
    const RadKernelInfo* rs = find(pl, "router_scored_topk_f32");
    const RadKernelInfo* rc = find(pl, "moe_scatter_i32");
    const RadKernelInfo* f1 = find(pl, "router_scored_topk_scatter_f32");
    const RadKernelInfo* f2 = find(pl, "router_gemm_f32_scored_topk_scatter_bf16");
    if (!rg || !rs || !rc || !f1 || !f2 || !f2->init || !f2->fini) {
        ok("rfuse rows", false, "gemm_nt_f32_bf16 / router_scored_topk_f32 / moe_scatter_i32 / router_scored_topk_scatter_f32 / "
                                "router_gemm_f32_scored_topk_scatter_bf16 (with init / fini) absent");
        return;
    }
    void* inst = nullptr;
    const int irc = f2->init(nullptr, 0, 0, 1, &inst);
    ok("router_gemm_f32_scored_topk_scatter init takes an instance", irc == RAD_OK && inst != nullptr, fmt("rc %d", irc));
    if (irc != RAD_OK || !inst) return;
    constexpr int64_t K = 5120, NE = 384, TK = 6, NL = 96;
    const double scale = 1.5;
    struct Out { void *lg, *ids, *w, *so, *off, *cnt; };
    const size_t tail = 64;
    auto sizes = [&](int64_t M) {
        return std::array<size_t, 6>{ (size_t)(M * NE * 4) + tail, (size_t)(M * TK * 4) + tail, (size_t)(M * TK * 4) + tail,
                                      (size_t)(M * TK * 4) + tail, (size_t)(NL + 1) * 4 + tail, (size_t)NL * 4 + tail };
    };
    static const char* const kOut[6] = { "logits", "ids", "weights", "sorted_tok", "expert_offset", "expert_count" };
    int bad_total = 0, checks = 0;
    for (int wbf : { 1, 0 }) {
        for (int mode : { 0, 1, 2 }) {   /* random, tied, NaN rows */
            g_rs = 0x5EEDF00Dull + (uint64_t)(wbf * 7 + mode);
            std::vector<float> wf((size_t)(NE * K)), bias((size_t)NE);
            for (auto& v : wf) v = (float)(gauss() * 0.02);
            for (auto& v : bias) v = (float)(gauss() * 0.05);
            if (mode == 1)   /* experts 2j+1 copy 2j (equal logits); bias equal too: ties the lower index must win */
                for (int64_t e = 1; e < NE; e += 2) {
                    std::memcpy(&wf[(size_t)(e * K)], &wf[(size_t)((e - 1) * K)], (size_t)K * 4);
                    bias[(size_t)e] = bias[(size_t)e - 1];
                }
            std::vector<uint16_t> wb((size_t)(NE * K));
            for (size_t i = 0; i < wb.size(); ++i) {
                wb[i] = bf(wf[i]);
                if (wbf) {   /* the f32 values a bf16 weight stands for (the checkpoint's) */
                    uint32_t u = (uint32_t)wb[i] << 16;
                    std::memcpy(&wf[i], &u, 4);
                }
            }
            for (int64_t M : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 24, 32, 40, 48 }) {
                void* dw = wbf ? up(wb) : up(wf);   /* again each M: dfree_all below frees every allocation */
                void* db = up(bias);
                std::vector<uint16_t> x((size_t)(M * K));
                for (int64_t r = 0; r < M; ++r) {
                    const double rs_ = std::ldexp(1.0, (int)(rnd() % 5) - 2);
                    for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs_));
                    if (mode == 2 && (r % 3) == 1) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
                }
                void* dx = up(x);
                const auto sz = sizes(M);
                for (int64_t base : { (int64_t)0, (int64_t)96, (int64_t)288 }) {
                    Out o[3];
                    for (int f = 0; f < 3; ++f) {
                        void** p = &o[f].lg;
                        for (int j = 0; j < 6; ++j) { p[j] = dalloc(sz[(size_t)j]); if (p[j]) st_memset(p[j], 0xA5, sz[(size_t)j]); }
                    }
                    if (!dx || !dw || !db || !o[2].cnt) { ok("rfuse alloc", false, "hipMalloc failed"); dfree_all(); f2->fini(inst); return; }
                    auto gemm_t = [&](Out& q) {
                        return std::vector<RadTensor>{ T(dx, RAD_BF16, { M, K }), T(dw, wbf ? RAD_BF16 : RAD_F32, { NE, K }),
                                                       T(q.lg, RAD_F32, { M, NE }) };
                    };
                    std::vector<RadParam> pg = { PI("M", M), PI("N", NE), PI("K", K), PS("dtype", "bf16") };
                    std::vector<RadParam> psel = { PI("M", M), PI("n_expert", NE), PI("top_k", TK), PI("norm", 1),
                                                   PF("route_scale", scale), PS("score", "sqrtsoftplus"), PS("dtype", "f32") };
                    std::vector<RadParam> psort = { PI("M", M), PI("n_expert", NL), PI("top_k", TK), PI("expert_base", base) };
                    std::vector<RadParam> pf1 = { PI("M", M), PI("n_expert", NE), PI("n_local", NL), PI("top_k", TK), PI("norm", 1),
                                                  PF("route_scale", scale), PS("score", "sqrtsoftplus"), PS("dtype", "f32"),
                                                  PI("expert_base", base) };
                    std::vector<RadParam> pf2 = { PI("M", M), PI("K", K), PI("n_expert", NE), PI("n_local", NL), PI("top_k", TK),
                                                  PI("norm", 1), PF("route_scale", scale), PS("score", "sqrtsoftplus"),
                                                  PS("dtype", "bf16"), PI("expert_base", base) };
                    std::vector<RadTensor> tg0 = gemm_t(o[0]), tg1 = gemm_t(o[1]);
                    std::vector<RadTensor> tsel = { T(o[0].lg, RAD_F32, { M, NE }), T(db, RAD_F32, { NE }),
                                                    T(o[0].ids, RAD_I32, { M, TK }), T(o[0].w, RAD_F32, { M, TK }) };
                    std::vector<RadTensor> tsort = { T(o[0].ids, RAD_I32, { M, TK }), T(o[0].so, RAD_I32, { M * TK }),
                                                     T(o[0].off, RAD_I32, { NL + 1 }), T(o[0].cnt, RAD_I32, { NL }) };
                    std::vector<RadTensor> tf1 = { T(o[1].lg, RAD_F32, { M, NE }), T(db, RAD_F32, { NE }),
                                                   T(o[1].ids, RAD_I32, { M, TK }), T(o[1].w, RAD_F32, { M, TK }),
                                                   T(o[1].so, RAD_I32, { M * TK }), T(o[1].off, RAD_I32, { NL + 1 }),
                                                   T(o[1].cnt, RAD_I32, { NL }) };
                    std::vector<RadTensor> tf2 = { T(dx, RAD_BF16, { M, K }), T(dw, wbf ? RAD_BF16 : RAD_F32, { NE, K }),
                                                   T(db, RAD_F32, { NE }), T(o[2].lg, RAD_F32, { M, NE }),
                                                   T(o[2].ids, RAD_I32, { M, TK }), T(o[2].w, RAD_F32, { M, TK }),
                                                   T(o[2].so, RAD_I32, { M * TK }), T(o[2].off, RAD_I32, { NL + 1 }),
                                                   T(o[2].cnt, RAD_I32, { NL }) };
                    RadArgs ag0 = args(tg0, pg), ag1 = args(tg1, pg), asel = args(tsel, psel), asort = args(tsort, psort);
                    RadArgs af1 = args(tf1, pf1), af2 = args(tf2, pf2, inst);
                    auto chain = [&]() {
                        int r = rg->launch(&ag0, nullptr);
                        if (r == RAD_OK) r = rs->launch(&asel, nullptr);
                        if (r == RAD_OK) r = rc->launch(&asort, nullptr);
                        return r;
                    };
                    auto form1 = [&]() {
                        int r = rg->launch(&ag1, nullptr);
                        if (r == RAD_OK) r = f1->launch(&af1, nullptr);
                        return r;
                    };
                    auto form2 = [&]() { return f2->launch(&af2, nullptr); };
                    const int c0 = chain(), c1 = form1(), c2a = form2(), c2b = form2();   /* form 2 twice: the counter resets */
                    const bool ran = c0 == RAD_OK && c1 == RAD_OK && c2a == RAD_OK && c2b == RAD_OK &&
                                     st_stream_sync(nullptr) == hipSuccess;
                    int first1 = -1, first2 = -1;
                    for (int j = 0; ran && j < 6; ++j) {
                        const std::vector<uint8_t> r0 = bytes((&o[0].lg)[j], sz[(size_t)j]);
                        if (first1 < 0 && r0 != bytes((&o[1].lg)[j], sz[(size_t)j])) first1 = j;
                        if (first2 < 0 && r0 != bytes((&o[2].lg)[j], sz[(size_t)j])) first2 = j;
                    }
                    const bool same = ran && first1 < 0 && first2 < 0;
                    ++checks;
                    if (!same) ++bad_total;
                    /* one line a (weight, mode, M) at base 0, every failure named */
                    if (base == 0 || !same)
                        ok(fmt("rfuse %s weights, %s, M=%lld, base %lld: form 1 and form 2 byte for byte vs the chain "
                               "(logits, ids, weights, sorted, offsets, counts + tails)",
                               wbf ? "bf16" : "f32", mode == 0 ? "random" : mode == 1 ? "tied ranks" : "NaN rows",
                               (long long)M, (long long)base).c_str(),
                           same,
                           !ran ? fmt("rc chain %d form1 %d form2 %d / %d", c0, c1, c2a, c2b)
                                : fmt("form 1: %s, form 2: %s", first1 < 0 ? "same" : kOut[first1],
                                      first2 < 0 ? "same" : kOut[first2]));
                    /* timing: bf16 weights (production, RADIANCE_DS41_ROUTER_BF16), random, base 96 */
                    if (ran && wbf && mode == 0 && base == 96 && (M == 1 || M == 8 || M == 16 || M == 48)) {
                        double w0, k0, w1, k1, w2, k2;
                        time2(chain, 200, &w0, &k0);
                        time2(form1, 200, &w1, &k1);
                        time2(form2, 200, &w2, &k2);
                        std::printf("  FUSEPERF rfuse M=%lld chain(gemm+sel+sort) warm %.2f cold %.2f us | form1(gemm+selsort) "
                                    "warm %.2f cold %.2f us (%+.2f) | form2(one launch) warm %.2f cold %.2f us (%+.2f)\n",
                                    (long long)M, w0, k0, w1, k1, k1 - k0, w2, k2, k2 - k0);
                        if (M <= 16) {
                            std::printf("  FUSEGATE rfuse form1 M=%lld %s (cold %.2f vs chain %.2f us)\n", (long long)M,
                                        k1 <= k0 ? "KEEP" : "DROP", k1, k0);
                            std::printf("  FUSEGATE rfuse form2 M=%lld %s (cold %.2f vs chain %.2f us)\n", (long long)M,
                                        k2 <= k0 ? "KEEP" : "DROP", k2, k0);
                        }
                    }
                }
                dfree_all();
            }
        }
    }
    info(fmt("rfuse: %d of %d (weights, mode, M, base) checks byte-identical", checks - bad_total, checks));
    f2->fini(inst);
}

/* ======================================================================= lever FUSE-M */
static void mf_case(const Plugin& pl) {
    const RadKernelInfo* rmix = find(pl, "mhc_mix_bf16");
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    auto stats = (int (*)(long long*, int))dlsym(pl.h, "rad_fuse_stats");
    if (!rmix || !sw || !stats) {
        ok("mixfin rows", false, "mhc_mix_bf16 / rad_fn4_switch / rad_fuse_stats absent");
        return;
    }
    if (sw("ds41_mix_fin", 1) != 1 || sw("ds41_mix_fin", 0) != 1 || sw("ds41_mix_fin", 2) >= 0) {
        ok("ds41_mix_fin switch", false, "1 / 0 not taken or 2 not refused");
        return;
    }
    sw("ds41_mix_wv", 0);
    constexpr int64_t n = 5120, hc = 4, W = hc * n, MIX = hc * (hc + 2), S = MIX + 1, iters = 20;
    const int64_t chunks = (W + 1023) / 1024;
    g_rs = 0x5EED0F1Bull;
    std::vector<float> fn((size_t)(MIX * W)), base(MIX), scale(3);
    for (auto& x : fn) x = (float)(gauss() * 0.01);
    for (auto& x : base) x = (float)(gauss() * 0.5);
    for (auto& x : scale) x = (float)(gauss() * 0.1 + 0.5);
    for (int64_t M : { 1, 2, 3, 6, 8, 9, 16, 24, 48, 49 }) {
        void *dfn = up(fn), *dbase = up(base), *dscale = up(scale);
        std::vector<uint16_t> h((size_t)(M * W));
        for (int64_t r = 0; r < M; ++r) {
            const double rs_ = std::ldexp(1.0, (int)(rnd() % 5) - 2);
            for (int64_t i = 0; i < W; ++i) h[(size_t)(r * W + i)] = bf((float)(gauss() * rs_));
        }
        void* dh = up(h);
        const size_t bp = (size_t)(M * hc * 4), bc = (size_t)(M * hc * hc * 4), bs = (size_t)(M * S * chunks * 4);
        struct Out { void *pre, *post, *comb, *scr; };
        auto mk = [&]() { return Out{ dalloc(bp + 64), dalloc(bp + 64), dalloc(bc + 64), dalloc(bs + 4096) }; };
        Out R = mk(), F = mk();
        if (!dfn || !dh || !F.scr) { ok("mixfin alloc", false, "hipMalloc failed"); dfree_all(); return; }
        std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("hc", hc), PI("iterations", iters), PF("eps", 1e-6),
                                     PF("hc_eps", 1e-6), PS("dtype", "bf16") };
        auto ts = [&](Out& o) {
            return std::vector<RadTensor>{ T(dh, RAD_BF16, { M, W }), T(dfn, RAD_F32, { MIX, W }), T(dscale, RAD_F32, { 3 }),
                                           T(dbase, RAD_F32, { MIX }), T(o.pre, RAD_F32, { M, hc }),
                                           T(o.post, RAD_F32, { M, hc }), T(o.comb, RAD_F32, { M, hc * hc }) };
        };
        std::vector<RadTensor> t0 = ts(R), t1 = ts(F);
        RadArgs a0 = args(t0, ps, nullptr, R.scr, (int64_t)bs + 4096), a1 = args(t1, ps, nullptr, F.scr, (int64_t)bs + 4096);
        auto go = [&](Out& o, RadArgs& a, int fin) {
            st_memset(o.pre, 0xA5, bp + 64);
            st_memset(o.post, 0xA5, bp + 64);
            st_memset(o.comb, 0xA5, bc + 64);
            st_memset(o.scr, 0x5A, bs + 4096);
            sw("ds41_mix_fin", fin);
            const int rc = rmix->launch(&a, nullptr);
            st_stream_sync(nullptr);
            return rc;
        };
        long long s0[4] = { 0, 0, 0, 0 }, s1[4] = { 0, 0, 0, 0 };
        const int rc0 = go(R, a0, 0);
        stats(s0, 4);
        const int rc1 = go(F, a1, 1);
        stats(s1, 4);
        const int rc2 = go(F, a1, 1);   /* twice: the row groups' counters come back to zero */
        sw("ds41_mix_fin", 0);
        if (rc0 != RAD_OK || rc1 != RAD_OK || rc2 != RAD_OK) {
            ok(fmt("mixfin M=%lld launches", (long long)M).c_str(), false, fmt("rc stock %d fin %d / %d", rc0, rc1, rc2));
            dfree_all();
            continue;
        }
        const bool want = M <= 48;
        const bool took = s1[0] > s0[0];
        const bool same = bytes(R.pre, bp + 64) == bytes(F.pre, bp + 64) && bytes(R.post, bp + 64) == bytes(F.post, bp + 64) &&
                          bytes(R.comb, bc + 64) == bytes(F.comb, bc + 64) && bytes(R.scr, bs) == bytes(F.scr, bs);
        ok(fmt("mhc_mix ds41_mix_fin 1 byte for byte vs 0, M=%lld (pre, post, comb + tails, partials; FIN %s)", (long long)M,
               took ? "taken" : "not taken").c_str(),
           same && took == want, fmt("same %d, FIN launches +%lld (want %s)", same ? 1 : 0, s1[0] - s0[0], want ? "> 0" : "0"));
        if (M == 1 || M == 8 || M == 16 || M == 48) {
            double w0, k0, w1, k1;
            sw("ds41_mix_fin", 0);
            time2([&]() { return rmix->launch(&a0, nullptr); }, 200, &w0, &k0);
            sw("ds41_mix_fin", 1);
            time2([&]() { return rmix->launch(&a1, nullptr); }, 200, &w1, &k1);
            sw("ds41_mix_fin", 0);
            std::printf("  FUSEPERF mixfin M=%lld split+finish warm %.2f cold %.2f us | FIN warm %.2f cold %.2f us (%+.2f)\n",
                        (long long)M, w0, k0, w1, k1, k1 - k0);
            if (M <= 16)
                std::printf("  FUSEGATE mixfin M=%lld %s (cold %.2f vs pair %.2f us)\n", (long long)M,
                            k1 <= k0 ? "KEEP" : "DROP", k1, k0);
        }
        dfree_all();
    }
    sw("ds41_mix_fin", 0);
}

}  // namespace fuse

static void fuse_cases(const Plugin& pl) {
    if (sel("rfuse") || sel("router_gemm_f32_scored_topk_scatter_bf16")) fuse::rf_case(pl);
    if (sel("mixfin")) fuse::mf_case(pl);
}
