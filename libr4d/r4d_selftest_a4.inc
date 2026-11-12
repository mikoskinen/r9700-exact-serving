/* r4d_selftest_a4.inc -- lever A4 (ds41-a4): the gg_gemv_m16 forms on one card (included by r4d_selftest.cpp after
 * r4d_selftest_gq.inc). Two modes:
 *
 *   --mrcheck16 [types]   the on-card byte check (a screen's P0): every lever A4 form (kA4Pins: the XP unit's forms at every
 *       (H, R), the rule's and the grid fit's WN, split MT, odd WN; D2's kernels at the grid fit) and the switch itself
 *       (gg_gemv 2 + gg_gemv_m16 1 with the table the plugin loaded from RADIANCE_DS41_GG_M16_FORMS) against the stock
 *       decode band, byte for byte, at M 1..16:
 *         - gemm_nt_ggml / logits_gemm_ggml at every DeepSeek-V4.1 dense TP4 shape (ds_mr_check's, D2's forms included);
 *         - gemm_nt_ggml_pair (q_a + attn_kv, attention_output_a g0 + g1) against the two stock launches;
 *         - folds a / e (rmsnorm_gemm_nt_ggml, swiglu_clamped_gemm_nt_ggml) against their two launches
 *           (ds41_ggfold_case with kA4Pins and the switch among its modes; M 1..48 and two prefill heights).
 *       A4VERDICT at the end; any difference fails the run (exit 1).
 *   --perfm16 [Ms] [iters] [types] [--m16-out FILE]   the gate bench: us a call of D2's own pick (gg_gemv 2, no pin: the
 *       int12 form) and of each A4 candidate, rotated weight copies past 256 MiB (DRAM, as --perfmr's "rot"), two passes
 *       (the candidates forward, then backward), every candidate's bytes against D2's and the stock band's. Shapes: the
 *       c8 profile's dense GEMMs (rank 0 and ranks 1-3 widths, engram_kv at E2's 6400 rows, the head) and the two pair
 *       shapes. M16PERF a (type, shape, M) with every candidate, M16SUM (D2 vs the best), M16GATE at M 8 / 16 per shape,
 *       M16EST (the c8 / c16 profile's calls a step x the per-call saving), and the table: a key takes the fastest
 *       candidate that beat D2 by >= 3 % in BOTH passes with identical bytes; keys without such a candidate get no entry
 *       (D2's form stays). --m16-out writes the entries as RADIANCE_DS41_GG_M16_FORMS lines. */

namespace a4st {

struct Cand { int h, r, mt, wn, xp; };
static std::string cand_name(const Cand& c) {
    if (c.h == 0) return "d2";
    std::string s = fmt("H%dR%d", c.h, c.r);
    if (c.mt) s += fmt("c%d", c.mt);
    s += c.wn > 0 ? fmt("w%d", c.wn) : c.wn < 0 ? std::string("wfit") : std::string("");
    s += c.xp == 1 ? "-xp" : "-a4";
    return s;
}
/* The candidates at M rows: the XP forms at every (H, R) with the rule's WN and the grid fit, MT cap 8 above 8 rows (two
 * tiles), and D2's kernels at the grid fit's WN. */
static std::vector<Cand> cands(int64_t M) {
    std::vector<Cand> v;
    const int hr[][2] = { { 1, 1 }, { 1, 2 }, { 2, 1 }, { 2, 2 }, { 4, 1 }, { 4, 2 } };
    for (const auto& p : hr)
        for (int wn : { 0, -1 }) v.push_back({ p[0], p[1], 0, wn, 1 });
    if (M > 8)
        for (const auto& p : hr)
            if (!(p[0] == 4 && p[1] == 2)) v.push_back({ p[0], p[1], 8, -1, 1 });
    const int d2hr[][2] = { { 1, 1 }, { 2, 1 }, { 2, 2 }, { 4, 1 }, { 4, 2 } };
    for (const auto& p : d2hr) v.push_back({ p[0], p[1], 0, -1, 2 });
    return v;
}

struct Shape { const char* name; int64_t N, N2, K; const char* types; bool logits; int c8_q, c8_o; };
/* calls a c8 step on rank 0 by type (the 12.11. profile: attention q6_K 27 / q5_K 13 layers, the shared expert q6_K 27 /
 * q8_0 13; c8_q = the first type's, c8_o = the second's); the _w512 shapes are ranks 1-3's (0 on rank 0). */
static const Shape kShapes[] = {
    { "attn_q_b(fold a)", 8192, 0, 1280, "q5_K,q6_K", false, 13, 27 },
    { "q_a+attn_kv(pair)", 1280, 512, 5120, "q5_K,q6_K", false, 13, 27 },
    { "attn_output_b", 5120, 0, 2048, "q5_K,q6_K", false, 13, 27 },
    { "attn_output_a g0+g1(pair)", 1024, 1024, 4096, "q5_K,q6_K", false, 13, 27 },
    { "shexp_gate_up", 1536, 0, 5120, "q6_K,q8_0", false, 27, 13 },
    { "shexp_down(fold e)", 5120, 0, 768, "q6_K,q8_0", false, 27, 13 },
    { "shexp_gate_up_w512", 1024, 0, 5120, "q6_K,q8_0", false, 0, 0 },
    { "shexp_down_w512(fold e)", 5120, 0, 512, "q6_K,q8_0", false, 0, 0 },
    { "engram_kv_e2", 6400, 0, 6144, "q6_K", false, 2, 0 },
    { "head", 32320, 0, 5120, "q6_K", true, 1, 0 },
};

/* --mrcheck16's pair part: gemm_nt_ggml_pair under every A4 pin and the switch vs the two stock launches (gg_gemv 0). */
static void pair_check(const Plugin& pl, const MrHooks& hk, int (*sw)(const char*, int), long long* launches, long long* bad) {
    struct Sh { const char* what; int64_t NA, NB, K; bool two_in; };
    const Sh shapes[] = { { "q_a + attn_kv", 1280, 512, 5120, false }, { "attention_output_a g0 + g1", 1024, 1024, 4096, true } };
    std::vector<MrPin> modes(std::begin(kA4Pins), std::end(kA4Pins));
    modes.push_back({ 0, 0, 0, 0, 0, 16 });
    for (const MrTy& ty : kMrTypes) {
        const bool q32 = !std::strcmp(ty.name, "q8_0");
        const RadKernelInfo* rp = find(pl, q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256");
        const RadKernelInfo* rg = find(pl, q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv");
        if (!rp || !rg) { ok(fmt("a4 pair rows (%s)", ty.name).c_str(), false, "a row absent"); continue; }
        for (const Sh& sh : shapes) {
            dfree_all();
            std::vector<uint8_t> hwa, hwb;
            mr_weights(hwa, ty, sh.NA, sh.K, 0xA4A0001ull ^ (uint64_t)sh.NA);
            mr_weights(hwb, ty, sh.NB, sh.K, 0xA4A0002ull ^ (uint64_t)sh.NB);
            const int64_t rb = sh.K / ty.qk * ty.bb, xw = sh.two_in ? 2 * sh.K : sh.K, yw = sh.NA + sh.NB + 8;
            void *dwa = dalloc(hwa.size()), *dwb = dalloc(hwb.size());
            std::vector<uint16_t> hx;
            mr_acts(hx, xw, 0xA4C75ull ^ (uint64_t)xw);
            void* dx = dalloc(hx.size() * 2);
            const size_t yb = (size_t)48 * yw * 2;
            void *yS = dalloc(yb), *yP = dalloc(yb);
            if (!dwa || !dwb || !dx || !yS || !yP) { ok("a4 pair alloc", false, "hipMalloc failed"); ++*bad; dfree_all(); return; }
            st_memcpy(dwa, hwa.data(), hwa.size(), hipMemcpyHostToDevice);
            st_memcpy(dwb, hwb.data(), hwb.size(), hipMemcpyHostToDevice);
            st_memcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
            void* dx2 = sh.two_in ? (void*)((uint16_t*)dx + sh.K) : dx;
            long long nbad = 0, nl = 0;
            std::string first;
            for (int64_t M = 1; M <= 16; ++M) {
                std::vector<RadParam> pp = { PI("M", M), PI("N", sh.NA), PI("N2", sh.NB), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                std::vector<RadParam> pa = { PI("M", M), PI("N", sh.NA), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                std::vector<RadParam> pb = { PI("M", M), PI("N", sh.NB), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                /* the stock: two launches at gg_gemv 0 */
                hk.pin(0, 0, 0, 0); hk.pin_xp(0); sw("gg_gemv_m16", 0); sw("gg_gemv", 0);
                st_memset(yS, 0xA5, yb);
                {
                    std::vector<RadTensor> ta = { gqst::TP(dx, RAD_BF16, M, sh.K, xw), T(dwa, RAD_U8, { sh.NA, rb }), gqst::TP(yS, RAD_BF16, M, sh.NA, yw) };
                    std::vector<RadTensor> tb = { gqst::TP(dx2, RAD_BF16, M, sh.K, xw), T(dwb, RAD_U8, { sh.NB, rb }),
                                                  gqst::TP((uint16_t*)yS + sh.NA, RAD_BF16, M, sh.NB, yw) };
                    RadArgs a{}, b{};
                    a.t = ta.data(); a.n_t = 3; a.p = pa.data(); a.n_p = (int)pa.size(); a.world_size = 1;
                    b.t = tb.data(); b.n_t = 3; b.p = pb.data(); b.n_p = (int)pb.size(); b.world_size = 1;
                    const int rc = rg->launch(&a, nullptr) | rg->launch(&b, nullptr);
                    st_stream_sync(nullptr);
                    if (rc != RAD_OK) { ++nbad; if (first.empty()) first = fmt("stock rc %d at M=%lld", rc, (long long)M); continue; }
                }
                const std::vector<uint8_t> S = gqst::dn(yS, yb);
                for (const MrPin& p : modes) {
                    if (p.xp == 16) { hk.pin(0, 0, 0, 0); hk.pin_xp(0); sw("gg_gemv", 2); sw("gg_gemv_m16", 1); }
                    else { sw("gg_gemv", 0); mr_apply(hk, p); }
                    st_memset(yP, 0xA5, yb);
                    std::vector<RadTensor> t = { gqst::TP(dx, RAD_BF16, M, sh.K, xw), T(dwa, RAD_U8, { sh.NA, rb }),
                                                 gqst::TP(dx2, RAD_BF16, M, sh.K, xw), T(dwb, RAD_U8, { sh.NB, rb }),
                                                 gqst::TP(yP, RAD_BF16, M, sh.NA, yw), gqst::TP((uint16_t*)yP + sh.NA, RAD_BF16, M, sh.NB, yw) };
                    if (!sh.two_in) absent(t[2]);
                    RadArgs a{};
                    a.t = t.data(); a.n_t = (int)t.size(); a.p = pp.data(); a.n_p = (int)pp.size(); a.world_size = 1;
                    const int rc = rp->launch(&a, nullptr);
                    st_stream_sync(nullptr);
                    const std::string form = hk.last ? hk.last() : "?";
                    hk.pin(0, 0, 0, 0); hk.pin_xp(0); sw("gg_gemv_m16", 0); sw("gg_gemv", 0);
                    ++nl;
                    const std::vector<uint8_t> P = gqst::dn(yP, yb);
                    if (rc != RAD_OK || P != S) {
                        ++nbad;
                        if (first.empty())
                            first = fmt("M=%lld %s (%s): rc %d, %s", (long long)M, p.xp == 16 ? "gg_gemv_m16" : mr_pin_name(p).c_str(),
                                        form.c_str(), rc, P != S ? "bytes differ" : "bytes same");
                    }
                }
            }
            *launches += nl;
            *bad += nbad;
            ok(fmt("A4 gemm_nt_ggml_pair %s %s %lld+%lldx%lld M=1..16: %d forms + the switch == the two stock launches",
                   sh.what, ty.name, (long long)sh.NA, (long long)sh.NB, (long long)sh.K, (int)modes.size() - 1).c_str(),
               nbad == 0 && nl > 0, fmt("%lld launches, %lld differ%s%s", nl, nbad, first.empty() ? "" : "; first: ", first.c_str()));
        }
    }
    dfree_all();
}

/* --perfm16 */
static void perf(const Plugin& pl, const char* mlist, int iters, const char* tlist, const char* outpath) {
    if (iters <= 0) iters = 100;
    const MrHooks hk = mr_hooks(pl);
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    if (!hk.pin || !hk.pin_xp || !sw) { std::printf("--perfm16: r4d_gg_mr_pin_xp / rad_fn4_switch absent (a plugin without lever A4)\n"); ++g_fail; return; }
    std::vector<int64_t> Ms;
    { std::string m = mlist && *mlist ? mlist : "5,6,7,8,9,10,11,12,13,14,15,16"; size_t p = 0;
      while (p < m.size()) { size_t q = m.find(',', p); if (q == std::string::npos) q = m.size(); Ms.push_back(std::strtoll(m.substr(p, q - p).c_str(), nullptr, 10)); p = q + 1; } }
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    auto clear = [&]() { hk.pin(0, 0, 0, 0); hk.pin_xp(0); if (hk.pin_w64) hk.pin_w64(0); sw("gg_gemv_m16", 0); sw("gg_gemv", 0); };
    auto set = [&](const Cand& c) {
        clear();
        if (c.h == 0) { sw("gg_gemv", 2); return; }   /* D2's own pick (the int12 form) */
        hk.pin(c.h, c.r, c.mt, c.wn);
        hk.pin_xp(c.xp);
    };
    std::printf("lever A4 timing (rotated weight copies past 256 MiB, mean of %d launches, two passes): D2's pick (gg_gemv 2) vs "
                "the A4 candidates; bytes vs D2 and the stock band\n", iters);
    std::vector<std::string> table;
    long long ndiff = 0, nkeys = 0, nwin = 0;
    double est_c8 = 0.0, est_c16 = 0.0, d2_c8 = 0.0, d2_c16 = 0.0;
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const Shape& sh : kShapes) {
            if (!mr_in(sh.types, ty.name) || sh.K % ty.qk) continue;
            const bool pair = sh.N2 > 0, q32 = ty.qk == 32;
            const RadKernelInfo* rg = find(pl, sh.logits ? (q32 ? "logits_gemm_ggml_qk32_gemv" : "logits_gemm_ggml_qk256_gemv")
                                                         : (q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv"));
            const RadKernelInfo* rp = pair ? find(pl, q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256") : nullptr;
            if (!rg || (pair && !rp)) { std::printf("  %s %s: row absent\n", ty.name, sh.name); continue; }
            const int64_t rb = sh.K / ty.qk * ty.bb, wa = sh.N * rb, wb = sh.N2 * rb, wbytes = wa + wb;
            const int copies = (int)std::max<int64_t>(2, (256ll << 20) / wbytes + 1);
            std::vector<uint8_t> hwa, hwb;
            mr_weights(hwa, ty, sh.N, sh.K, 0xA4BE0001ull ^ (uint64_t)(sh.N * 7 + sh.K));
            if (pair) mr_weights(hwb, ty, sh.N2, sh.K, 0xA4BE0002ull ^ (uint64_t)(sh.N2 * 7 + sh.K));
            const int c8w = std::string(ty.name) == std::string(sh.types).substr(0, 4) ? sh.c8_q : sh.c8_o;
            for (int64_t M : Ms) {
                if (M < 1 || M > 16) continue;
                dfree_all();
                std::vector<uint16_t> hx;
                mr_acts(hx, sh.K, 0xA4ACull ^ (uint64_t)M);
                const size_t ob = sh.logits ? 4 : 2, yw = (size_t)(sh.N + sh.N2), yb = (size_t)M * yw * ob;
                void* d_x = dalloc((size_t)(M * sh.K) * 2);
                void* d_w = dalloc((size_t)wbytes * (size_t)copies);
                void* d_y = dalloc(yb);
                if (!d_x || !d_w || !d_y) { std::printf("  %s %s M=%lld: alloc failed\n", ty.name, sh.name, (long long)M); ++g_fail; continue; }
                st_memcpy(d_x, hx.data(), (size_t)(M * sh.K) * 2, hipMemcpyHostToDevice);
                for (int c = 0; c < copies; ++c) {
                    uint8_t* base = (uint8_t*)d_w + (size_t)c * (size_t)wbytes;
                    st_memcpy(base, hwa.data(), (size_t)wa, hipMemcpyHostToDevice);
                    if (pair) st_memcpy(base + wa, hwb.data(), (size_t)wb, hipMemcpyHostToDevice);
                }
                std::vector<RadParam> pg, pp, pa, pb;
                if (sh.logits) pg = { PI("M", M), PI("n_vocab", sh.N), PI("n_embd", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                else pg = { PI("M", M), PI("N", sh.N), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                pp = { PI("M", M), PI("N", sh.N), PI("N2", sh.N2), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                pa = pg;
                pb = { PI("M", M), PI("N", sh.N2), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                /* one launch of copy c: the pair's one launch, or the plain row */
                auto launch = [&](int c) {
                    uint8_t* base = (uint8_t*)d_w + (size_t)(c % copies) * (size_t)wbytes;
                    if (pair) {
                        std::vector<RadTensor> t = { gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base, RAD_U8, { sh.N, rb }),
                                                     gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base + wa, RAD_U8, { sh.N2, rb }),
                                                     gqst::TP(d_y, RAD_BF16, M, sh.N, (int64_t)yw),
                                                     gqst::TP((uint16_t*)d_y + sh.N, RAD_BF16, M, sh.N2, (int64_t)yw) };
                        absent(t[2]);
                        RadArgs a{};
                        a.t = t.data(); a.n_t = (int)t.size(); a.p = pp.data(); a.n_p = (int)pp.size(); a.world_size = 1;
                        return rp->launch(&a, nullptr);
                    }
                    std::vector<RadTensor> t = { T(d_x, RAD_BF16, { M, sh.K }), T(base, RAD_U8, { sh.N, rb }),
                                                 T(d_y, sh.logits ? RAD_F32 : RAD_BF16, { M, sh.N }) };
                    RadArgs a{};
                    a.t = t.data(); a.n_t = 3; a.p = pg.data(); a.n_p = (int)pg.size(); a.world_size = 1;
                    return rg->launch(&a, nullptr);
                };
                /* the stock band's bytes (gg_gemv 0; a pair: its fallback = the two stock launches) */
                clear();
                st_memset(d_y, 0xff, yb);
                int rc = launch(0);
                st_stream_sync(nullptr);
                std::vector<uint8_t> stock(yb);
                st_memcpy(stock.data(), d_y, yb, hipMemcpyDeviceToHost);
                std::vector<Cand> cs = cands(M);
                cs.insert(cs.begin(), Cand{ 0, 0, 0, 0, 0 });
                std::vector<double> t1(cs.size(), -1.0), t2(cs.size(), -1.0);
                std::vector<std::string> forms(cs.size());
                std::vector<int> same(cs.size(), 0);
                for (int pass = 0; pass < 2; ++pass)
                    for (size_t j0 = 0; j0 < cs.size(); ++j0) {
                        const size_t j = pass == 0 ? j0 : cs.size() - 1 - j0;
                        set(cs[j]);
                        st_memset(d_y, 0xff, yb);
                        rc = launch(0);
                        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) { clear(); continue; }
                        if (pass == 0) {
                            std::vector<uint8_t> got(yb);
                            st_memcpy(got.data(), d_y, yb, hipMemcpyDeviceToHost);
                            same[j] = got == stock ? 1 : 0;
                            forms[j] = hk.last ? hk.last() : "";
                        }
                        /* a candidate the form declines (no fit: "" as its form) ran the stock band: not a candidate */
                        if (j > 0 && forms[j].empty()) { clear(); continue; }
                        for (int i = 0; i < 20; ++i) launch(i);
                        const double us = st_time_us([&](int i) { launch(i); }, iters, nullptr);
                        (pass == 0 ? t1 : t2)[j] = us;
                        clear();
                    }
                ++nkeys;
                std::string line = fmt("M16PERF %s %s %lldx%lld%s M=%lld:", ty.name, sh.name, (long long)sh.N, (long long)sh.K,
                                       pair ? fmt("+%lld", (long long)sh.N2).c_str() : "", (long long)M);
                size_t best = 0;
                bool alls = true;
                for (size_t j = 0; j < cs.size(); ++j) {
                    const bool diff = t1[j] > 0 && !same[j];
                    alls = alls && !diff;
                    if (j > 0 && forms[j].empty()) { line += fmt(" %s declined;", cand_name(cs[j]).c_str()); continue; }
                    line += fmt(" %s %.1f/%.1f%s%s;", cand_name(cs[j]).c_str(), t1[j], t2[j],
                                forms[j].empty() ? "" : fmt(" [%s]", forms[j].c_str()).c_str(), diff ? " OUTPUT DIFFERS" : "");
                    if (j > 0 && same[j] && t1[j] > 0 && t2[j] > 0 && t1[j] < 0.97 * t1[0] && t2[j] < 0.97 * t2[0] &&
                        (best == 0 || t1[j] + t2[j] < t1[best] + t2[best]))
                        best = j;
                }
                std::printf("%s\n", line.c_str());
                if (!alls || !same[0]) { ++ndiff; ++g_fail; }
                const double d2us = 0.5 * (t1[0] + t2[0]), bus = best ? 0.5 * (t1[best] + t2[best]) : d2us;
                std::printf("M16SUM %s %s M=%lld: d2 %.1f us [%s] | best %s %.1f us [%s] | %+.1f %% | bytes %s\n", ty.name,
                            sh.name, (long long)M, d2us, forms[0].c_str(), best ? cand_name(cs[best]).c_str() : "d2", bus,
                            best ? forms[best].c_str() : forms[0].c_str(), 100.0 * (bus - d2us) / d2us,
                            alls && same[0] ? "identical" : "DIFFER");
                if (M == 8 || M == 16)
                    std::printf("M16GATE %s %s M=%lld: d2 %.1f us -> %.1f us (%s), %+.1f %%; c%d calls a step %d\n", ty.name,
                                sh.name, (long long)M, d2us, bus, best ? cand_name(cs[best]).c_str() : "no A4 win",
                                100.0 * (bus - d2us) / d2us, M == 8 ? 8 : 16, c8w);
                if (M == 8) { est_c8 += c8w * (d2us - bus); d2_c8 += c8w * d2us; }
                if (M == 16) { est_c16 += c8w * (d2us - bus); d2_c16 += c8w * d2us; }
                if (best) {
                    ++nwin;
                    const Cand& c = cs[best];
                    table.push_back(fmt("%s %lld %lld %lld %lld %d %d %d %d %d %d   # %s: d2 %.1f us -> %s %.1f us (%+.1f %%)",
                                        ty.name, (long long)sh.N, (long long)sh.K, (long long)M, (long long)M, pair ? 1 : 0,
                                        c.h, c.r, c.mt, c.wn, c.xp == 1 ? 1 : 0, sh.name, d2us, cand_name(c).c_str(), bus,
                                        100.0 * (bus - d2us) / d2us));
                }
                std::fflush(stdout);
            }
        }
    }
    clear();
    dfree_all();
    std::printf("M16EST rank 0 (bench us, the 12.11. profile's calls a step; fold a / e timed as their plain GEMMs): c8 at M 8 "
                "d2 %.0f us -> %.0f us a step (%.0f us saved); c16 at M 16 d2 %.0f us -> %.0f us (%.0f us saved)\n",
                d2_c8, d2_c8 - est_c8, est_c8, d2_c16, d2_c16 - est_c16, est_c16);
    std::printf("M16TABLE %zu entries (M16FORM lines: RADIANCE_DS41_GG_M16_FORMS's format)\n", table.size());
    for (const std::string& t : table) std::printf("M16FORM %s\n", t.c_str());
    if (outpath && *outpath) {
        FILE* f = std::fopen(outpath, "w");
        if (!f) { std::printf("--m16-out %s: cannot write\n", outpath); ++g_fail; }
        else {
            std::fprintf(f, "# lever A4 gg_gemv_m16 forms (r4d_selftest --perfm16, %d launches, two passes); type N K m_lo m_hi pair H R MT WN XP\n", iters);
            for (const std::string& t : table) std::fprintf(f, "%s\n", t.c_str());
            std::fclose(f);
        }
    }
    std::printf("M16VERDICT: %lld keys, %lld with an A4 win (>= 3 %% both passes), bytes %s\n", nkeys, nwin,
                ndiff ? fmt("DIFFER at %lld", ndiff).c_str() : "identical everywhere");
}

}  // namespace a4st

/* --mrcheck16 [types]: the plain shapes (ds_mr_check, M 1..16, D2 + A4 forms + the switch), the pair, folds a / e. */
static void a4_check(const Plugin& pl, const char* tlist) {
    const MrHooks hk = mr_hooks(pl);
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    if (!hk.pin || !hk.pin_xp || !sw) { std::printf("--mrcheck16: a plugin without lever A4\n"); ++g_fail; return; }
    const int f0 = g_fail;
    ds_mr_check(pl, tlist, 16, true);
    const int f1 = g_fail;
    long long pl_n = 0, pl_bad = 0;
    a4st::pair_check(pl, hk, sw, &pl_n, &pl_bad);
    const int f2 = g_fail;
    ds41_ggfold_case(pl, "rmsnorm_gemm_nt_ggml_qk256", 1);
    ds41_ggfold_case(pl, "rmsnorm_gemm_nt_ggml_qk32", 1);
    ds41_ggfold_case(pl, "swiglu_clamped_gemm_nt_ggml_qk256", 2);
    ds41_ggfold_case(pl, "swiglu_clamped_gemm_nt_ggml_qk32", 2);
    const int f3 = g_fail;
    std::printf("A4VERDICT: plain %s, pair %s (%lld launches, %lld differ), folds %s; table entries %d -> %s\n",
                f1 == f0 ? "PASS" : "FAIL", f2 == f1 ? "PASS" : "FAIL", pl_n, pl_bad, f3 == f2 ? "PASS" : "FAIL",
                hk.m16_forms ? hk.m16_forms() : -1, f3 == f0 ? "PASS (byte-identical)" : "FAIL");
}
