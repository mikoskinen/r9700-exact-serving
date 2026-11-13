/* r4d_selftest_wmg.inc -- the WMG gate bench (ds41-wmgate, 13.11.2026; a BENCH branch: no engine image carries it). Included by
 * r4d_selftest.cpp after r4d_selftest_a4.inc. The question: gemm_nt_ggml / logits_gemm_ggml's WMMA form (gg_wm_kernel, the
 * prefill rows' and the > 48-row steps' kernel) at DECODE heights M 1..48, where the engine runs the decode band (the stock
 * GEMV, lever D2's multi-row form, lever A4's M <= 16 forms). libr4d's bench pin r4d_gg_wm_pin(1) sends every M to the WMMA
 * branch of gg_gemm_run; r4d_gg_wm_count() counts its launches (a W arm that did not reach gg_launch_wm fails the run).
 *
 *   --perfwmg [Ms] [iters] [rounds] [types]   us a call, rotated weight copies past 256 MiB (as --perfm16; --cold MIB: the
 *       median past a flush), per (type, shape, M): D = the decode band as int16 runs it (gg_gemv 2 + gg_gemv_m16 1 with the
 *       loaded RADIANCE_DS41_GG_M16_FORMS table; the two pair shapes in gemm_nt_ggml_pair's ONE launch), W = the WMMA form
 *       (the pair shapes as TWO launches: there is no WMMA pair kernel), order D W W D a round. Shapes: lever A4's
 *       (a4st::kShapes: rank 0's, ranks 1-3's _w512, engram_kv at E2's rows, the head). WMGPERF lines (every round's
 *       values); the per-step sums are made from them outside (wmgsum.py).
 *   --wmgcheck [types]   bytes: does a row's WMMA output depend on M or on the row's place in the launch? alone[r] = the
 *       WMMA form at M 1 on row r (48 rows); then M 2..48 from row 0, windows starting at other rows (a row at another tile
 *       place), 49..130-row launches of permuted rows (other blocks), each row against alone[r]. Info: the outputs that
 *       differ between the WMMA form and the decode band at M 1 (the one-time reference change) and their size.
 *       WMGCHECK a (type, shape), WMGVERDICT. */

namespace wmg {

struct Hk {
    void (*wm_pin)(int) = nullptr;
    long long (*wm_count)() = nullptr;
    int (*sw)(const char*, int) = nullptr;
    MrHooks mr;
};
static bool hooks(const Plugin& pl, Hk& h, const char* who) {
    h.wm_pin = (void (*)(int))dlsym(pl.h, "r4d_gg_wm_pin");
    h.wm_count = (long long (*)())dlsym(pl.h, "r4d_gg_wm_count");
    h.sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    h.mr = mr_hooks(pl);
    if (!h.wm_pin || !h.wm_count || !h.sw || !h.mr.pin || !h.mr.pin_xp) {
        std::printf("%s: r4d_gg_wm_pin / r4d_gg_wm_count / rad_fn4_switch / r4d_gg_mr_pin absent (not the ds41-wmgate plugin)\n", who);
        ++g_fail;
        return false;
    }
    return true;
}
static void clear(const Hk& h) {
    h.wm_pin(0);
    h.mr.pin(0, 0, 0, 0); h.mr.pin_xp(0);
    if (h.mr.pin_w64) h.mr.pin_w64(0);
    h.sw("gg_gemv_m16", 0); h.sw("gg_gemv", 0);
}
static void set_dec(const Hk& h) { clear(h); h.sw("gg_gemv", 2); h.sw("gg_gemv_m16", 1); }   /* int16's decode band */
static void set_wm(const Hk& h) { clear(h); h.wm_pin(1); }

static std::vector<int64_t> ilist(const char* l, const char* dflt) {
    std::vector<int64_t> v;
    std::string m = l && *l ? l : dflt;
    size_t p = 0;
    while (p < m.size()) {
        size_t q = m.find(',', p);
        if (q == std::string::npos) q = m.size();
        v.push_back(std::strtoll(m.substr(p, q - p).c_str(), nullptr, 10));
        p = q + 1;
    }
    return v;
}
static float bf16f(uint16_t v) { const uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }

/* --perfwmg */
static void perf(const Plugin& pl, const char* mlist, int iters, int rounds, const char* tlist) {
    if (iters <= 0) iters = 100;
    if (rounds <= 0) rounds = 3;
    Hk h;
    if (!hooks(pl, h, "--perfwmg")) return;
    const std::vector<int64_t> Ms = ilist(mlist, "1,2,4,8,16,24,32,48");
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const char* mode = g_cold > 0 ? "cold" : "rot";
    std::printf("WMG gate timing (%s, %d launches a reading, %d rounds of D W W D): D = int16's decode band (gg_gemv 2 + gg_gemv_m16 1, "
                "table entries %d; pair shapes one launch), W = the WMMA form at the same M (pair shapes two launches)\n",
                g_cold > 0 ? fmt("cold, %lld MiB read before each, median", (long long)(g_cold >> 20)).c_str()
                           : "rotated weight copies past 256 MiB, mean",
                iters, rounds, h.mr.m16_forms ? h.mr.m16_forms() : -1);
    long long nkeys = 0, nbad = 0;
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const a4st::Shape& sh : a4st::kShapes) {
            if (!mr_in(sh.types, ty.name) || sh.K % ty.qk) continue;
            const bool pair = sh.N2 > 0, q32 = ty.qk == 32;
            const RadKernelInfo* rg = find(pl, sh.logits ? (q32 ? "logits_gemm_ggml_qk32_gemv" : "logits_gemm_ggml_qk256_gemv")
                                                         : (q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv"));
            const RadKernelInfo* rp = pair ? find(pl, q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256") : nullptr;
            if (!rg || (pair && !rp)) { std::printf("  %s %s: row absent\n", ty.name, sh.name); ++g_fail; continue; }
            const int64_t rb = sh.K / ty.qk * ty.bb, wa = sh.N * rb, wb = sh.N2 * rb, wbytes = wa + wb;
            const int copies = (int)std::max<int64_t>(2, (256ll << 20) / wbytes + 1);
            std::vector<uint8_t> hwa, hwb;
            mr_weights(hwa, ty, sh.N, sh.K, 0xA4BE0001ull ^ (uint64_t)(sh.N * 7 + sh.K));
            if (pair) mr_weights(hwb, ty, sh.N2, sh.K, 0xA4BE0002ull ^ (uint64_t)(sh.N2 * 7 + sh.K));
            const int calls = std::string(ty.name) == std::string(sh.types).substr(0, 4) ? sh.c8_q : sh.c8_o;
            for (int64_t M : Ms) {
                if (M < 1 || M > 48) continue;
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
                std::vector<RadParam> pg, pp, pb;
                if (sh.logits) pg = { PI("M", M), PI("n_vocab", sh.N), PI("n_embd", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                else pg = { PI("M", M), PI("N", sh.N), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                pp = { PI("M", M), PI("N", sh.N), PI("N2", sh.N2), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                pb = { PI("M", M), PI("N", sh.N2), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                /* one call of copy c. one_launch: the pair kernel (D); else the plain row, a pair as its two plain launches (W) */
                auto launch = [&](int c, bool one_launch) -> int {
                    uint8_t* base = (uint8_t*)d_w + (size_t)(c % copies) * (size_t)wbytes;
                    if (pair && one_launch) {
                        std::vector<RadTensor> t = { gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base, RAD_U8, { sh.N, rb }),
                                                     gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base + wa, RAD_U8, { sh.N2, rb }),
                                                     gqst::TP(d_y, RAD_BF16, M, sh.N, (int64_t)yw),
                                                     gqst::TP((uint16_t*)d_y + sh.N, RAD_BF16, M, sh.N2, (int64_t)yw) };
                        absent(t[2]);
                        RadArgs a{};
                        a.t = t.data(); a.n_t = (int)t.size(); a.p = pp.data(); a.n_p = (int)pp.size(); a.world_size = 1;
                        return rp->launch(&a, nullptr);
                    }
                    if (pair) {
                        std::vector<RadTensor> ta = { gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base, RAD_U8, { sh.N, rb }),
                                                      gqst::TP(d_y, RAD_BF16, M, sh.N, (int64_t)yw) };
                        std::vector<RadTensor> tb = { gqst::TP(d_x, RAD_BF16, M, sh.K, sh.K), T(base + wa, RAD_U8, { sh.N2, rb }),
                                                      gqst::TP((uint16_t*)d_y + sh.N, RAD_BF16, M, sh.N2, (int64_t)yw) };
                        RadArgs a{}, b{};
                        a.t = ta.data(); a.n_t = 3; a.p = pg.data(); a.n_p = (int)pg.size(); a.world_size = 1;
                        b.t = tb.data(); b.n_t = 3; b.p = pb.data(); b.n_p = (int)pb.size(); b.world_size = 1;
                        const int r1 = rg->launch(&a, nullptr);
                        const int r2 = rg->launch(&b, nullptr);
                        return r1 != RAD_OK ? r1 : r2;
                    }
                    std::vector<RadTensor> t = { T(d_x, RAD_BF16, { M, sh.K }), T(base, RAD_U8, { sh.N, rb }),
                                                 T(d_y, sh.logits ? RAD_F32 : RAD_BF16, { M, sh.N }) };
                    RadArgs a{};
                    a.t = t.data(); a.n_t = 3; a.p = pg.data(); a.n_p = (int)pg.size(); a.world_size = 1;
                    return rg->launch(&a, nullptr);
                };
                auto bytes = [&](bool one_launch, int* rc) {
                    st_memset(d_y, 0xff, yb);
                    *rc = launch(0, one_launch);
                    if (st_stream_sync(nullptr) != hipSuccess) *rc = -999;
                    std::vector<uint8_t> got(yb);
                    st_memcpy(got.data(), d_y, yb, hipMemcpyDeviceToHost);
                    return got;
                };
                int rc0 = 0, rcd = 0, rcw = 0;
                clear(h);
                const std::vector<uint8_t> stock = bytes(true, &rc0);       /* the stock band (a pair: its two stock launches inside) */
                set_dec(h);
                const std::vector<uint8_t> dec = bytes(true, &rcd);
                const std::string dform = h.mr.last ? h.mr.last() : "";
                set_wm(h);
                const long long c0 = h.wm_count();
                const std::vector<uint8_t> wm = bytes(false, &rcw);
                const long long took = h.wm_count() - c0;
                const bool dec_same = dec == stock, wm_ran = took == (pair ? 2 : 1);
                size_t ndiff = 0, nout = yb / ob;
                double maxrel = 0.0;
                for (size_t e = 0; e < yb; e += ob) {
                    if (!std::memcmp(&wm[e], &dec[e], ob)) continue;
                    ++ndiff;
                    float a, b;
                    if (ob == 4) { std::memcpy(&a, &wm[e], 4); std::memcpy(&b, &dec[e], 4); }
                    else { uint16_t ua, ub; std::memcpy(&ua, &wm[e], 2); std::memcpy(&ub, &dec[e], 2); a = bf16f(ua); b = bf16f(ub); }
                    if (std::isfinite(a) && std::isfinite(b)) {
                        const double den = std::max(std::fabs((double)a), std::fabs((double)b));
                        if (den > 0) maxrel = std::max(maxrel, std::fabs((double)a - (double)b) / den);
                    }
                }
                std::vector<double> td, tw;
                if (rc0 == RAD_OK && rcd == RAD_OK && rcw == RAD_OK && wm_ran) {
                    for (int r = 0; r < rounds; ++r)
                        for (int k = 0; k < 4; ++k) {
                            const bool w = k == 1 || k == 2;   /* D W W D */
                            if (w) set_wm(h); else set_dec(h);
                            for (int i = 0; i < 20; ++i) launch(i, !w);
                            const double us = st_time_us([&](int i) { launch(i, !w); }, iters, nullptr);
                            (w ? tw : td).push_back(us);
                        }
                }
                clear(h);
                ++nkeys;
                const bool good = !td.empty() && dec_same;
                if (!good) { ++nbad; ++g_fail; }
                auto stats = [](const std::vector<double>& v, double* mn, double* mx) {
                    double s = 0; *mn = 1e30; *mx = 0;
                    for (double x : v) { s += x; *mn = std::min(*mn, x); *mx = std::max(*mx, x); }
                    return v.empty() ? -1.0 : s / (double)v.size();
                };
                double dmn, dmx, wmn, wmx;
                const double dm = stats(td, &dmn, &dmx), wmn_ = stats(tw, &wmn, &wmx);
                std::string raw_d, raw_w;
                for (double x : td) raw_d += fmt(" %.1f", x);
                for (double x : tw) raw_w += fmt(" %.1f", x);
                std::printf("WMGPERF %s|%s|%s|%lld|%lld|%lld|M=%lld|calls=%d|D %.2f min %.2f max %.2f [%s]|W %.2f min %.2f max %.2f|W/D %.3f|"
                            "dec==stock %s|wm launches %lld|wm!=dec %zu of %zu (%.2f %%) maxrel %.3g|rc %d %d %d|D:%s|W:%s\n",
                            mode, ty.name, sh.name, (long long)sh.N, (long long)sh.N2, (long long)sh.K, (long long)M, calls, dm, dmn, dmx,
                            dform.c_str(), wmn_, wmn, wmx, dm > 0 ? wmn_ / dm : 0.0, dec_same ? "yes" : "NO", took, ndiff, nout,
                            nout ? 100.0 * (double)ndiff / (double)nout : 0.0, maxrel, rc0, rcd, rcw, raw_d.c_str(), raw_w.c_str());
                std::fflush(stdout);
            }
        }
    }
    clear(h);
    dfree_all();
    if (nkeys == 0) ++g_fail;
    std::printf("WMGVERDICT perf %s: %lld (type, shape, M) keys, %lld bad (a launch failed, the W arm missed the WMMA kernel, or D != the stock band)\n",
                mode, nkeys, nbad);
}

/* --wmgcheck */
static void check(const Plugin& pl, const char* tlist) {
    Hk h;
    if (!hooks(pl, h, "--wmgcheck")) return;
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const int64_t R = 48, RB = 130;
    const std::vector<int64_t> Ms = { 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 47, 48 };
    const int64_t wins[][2] = { { 1, 1 }, { 1, 16 }, { 5, 8 }, { 15, 2 }, { 16, 16 }, { 17, 16 }, { 7, 41 }, { 31, 17 }, { 33, 15 }, { 47, 1 }, { 40, 8 }, { 3, 45 } };
    const int64_t bigs[] = { 49, 64, 96, 130 };
    long long keys = 0, bad_keys = 0, launches = 0, rows_cmp = 0, rows_bad = 0;
    std::printf("WMG byte check: the WMMA form's row bytes at M 1 vs the same row inside M 2..48 (from row 0), inside windows at other "
                "starts, and inside 49..130-row launches of permuted rows\n");
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const MrShape& sh : kMrShapes) {
            if (sh.logits && std::strcmp(ty.name, "q6_K")) continue;
            if (sh.K % ty.qk) continue;
            if (!std::strcmp(sh.name, "engram_kv")) continue;   /* 25600 x 6144: the E2 width is covered by --perfwmg; the same kernel */
            MrLaunch L1;
            if (!mr_setup(L1, pl, sh, ty, 1)) { std::printf("  %s %s: row absent\n", ty.name, sh.name); ++g_fail; continue; }
            dfree_all();
            std::vector<uint8_t> hw;
            mr_weights(hw, ty, sh.N, sh.K, 0x9E3779B97F4A7C15ull ^ (uint64_t)(sh.N * 131 + sh.K) ^ 0x776du);
            std::vector<uint16_t> hx;
            mr_acts(hx, sh.K, 0x1234567ull ^ (uint64_t)sh.K ^ 0x776du);          /* 48 rows */
            std::vector<uint16_t> hb((size_t)(RB * sh.K));                        /* 130 rows: row i = row (7 i + 3) % 48 */
            for (int64_t i = 0; i < RB; ++i)
                std::memcpy(&hb[(size_t)(i * sh.K)], &hx[(size_t)(((7 * i + 3) % R) * sh.K)], (size_t)sh.K * 2);
            const size_t ob = sh.logits ? 4 : 2, rowb = (size_t)sh.N * ob;
            void* d_w = dalloc(hw.size());
            void* d_x = dalloc(hx.size() * 2);
            void* d_b = dalloc(hb.size() * 2);
            void* d_y = dalloc((size_t)RB * rowb);
            if (!d_w || !d_x || !d_b || !d_y) { std::printf("  FAIL %s %s: alloc\n", ty.name, sh.name); ++g_fail; continue; }
            st_memcpy(d_w, hw.data(), hw.size(), hipMemcpyHostToDevice);
            st_memcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(d_b, hb.data(), hb.size() * 2, hipMemcpyHostToDevice);
            int bad = 0;
            std::string fails;
            /* run M rows from x -> bytes */
            auto run = [&](void* x, int64_t M, std::vector<uint8_t>& out) -> bool {
                MrLaunch L;
                if (!mr_setup(L, pl, sh, ty, M)) return false;
                st_memset(d_y, 0xff, (size_t)M * rowb);
                const int rc = L.run(x, d_w, d_y);
                ++launches;
                if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return false;
                out.resize((size_t)M * rowb);
                st_memcpy(out.data(), d_y, (size_t)M * rowb, hipMemcpyDeviceToHost);
                return true;
            };
            /* the pin works: unpinned M 1 = the decode band, pinned M 1 = the WMMA kernel */
            std::vector<uint8_t> alone((size_t)R * rowb), one, decone;
            clear(h);
            bool okr = run(d_x, 1, decone);
            set_wm(h);
            const long long c0 = h.wm_count();
            for (int64_t r = 0; okr && r < R; ++r) {
                okr = run((uint16_t*)d_x + r * sh.K, 1, one);
                if (okr) std::memcpy(&alone[(size_t)r * rowb], one.data(), rowb);
            }
            if (!okr || h.wm_count() - c0 != R) { std::printf("  FAIL %s %s: M 1 launches (wm launches %lld of %lld)\n", ty.name, sh.name, h.wm_count() - c0, (long long)R); ++g_fail; ++bad_keys; ++keys; clear(h); continue; }
            /* info: the WMMA form vs the decode band on row 0 */
            size_t nd = 0, nfin = 0; double maxrel = 0, sumrel = 0;
            for (size_t e = 0; e < rowb; e += ob) {
                float a, b;
                if (ob == 4) { std::memcpy(&a, &alone[e], 4); std::memcpy(&b, &decone[e], 4); }
                else { uint16_t ua, ub; std::memcpy(&ua, &alone[e], 2); std::memcpy(&ub, &decone[e], 2); a = bf16f(ua); b = bf16f(ub); }
                if (std::memcmp(&alone[e], &decone[e], ob)) ++nd;
                if (std::isfinite(a) && std::isfinite(b)) {
                    const double den = std::max(std::fabs((double)a), std::fabs((double)b));
                    const double rel = den > 0 ? std::fabs((double)a - (double)b) / den : 0.0;
                    maxrel = std::max(maxrel, rel); sumrel += rel; ++nfin;
                }
            }
            auto cmp = [&](const std::vector<uint8_t>& got, int64_t M, const char* what, auto rowof) {
                for (int64_t m = 0; m < M; ++m) {
                    ++rows_cmp;
                    const int64_t r = rowof(m);
                    if (std::memcmp(&got[(size_t)m * rowb], &alone[(size_t)r * rowb], rowb)) {
                        ++rows_bad; ++bad;
                        size_t e = 0;
                        while (e < rowb && got[(size_t)m * rowb + e] == alone[(size_t)r * rowb + e]) ++e;
                        if (fails.size() < 600) fails += fmt(" %s M=%lld place %lld (row %lld) first output %zu;", what, (long long)M, (long long)m, (long long)r, e / ob);
                    }
                }
            };
            std::vector<uint8_t> got;
            for (int64_t M : Ms) {
                if (!run(d_x, M, got)) { ++bad; fails += fmt(" M=%lld rc;", (long long)M); continue; }
                cmp(got, M, "from0", [&](int64_t m) { return m; });
            }
            for (const auto& w : wins) {
                const int64_t s = w[0], M = w[1];
                if (!run((uint16_t*)d_x + s * sh.K, M, got)) { ++bad; fails += fmt(" win %lld+%lld rc;", (long long)s, (long long)M); continue; }
                cmp(got, M, "win", [&](int64_t m) { return s + m; });
            }
            const long long c1 = h.wm_count();
            for (int64_t M : bigs) {
                if (!run(d_b, M, got)) { ++bad; fails += fmt(" big M=%lld rc;", (long long)M); continue; }
                cmp(got, M, "big", [&](int64_t m) { return (7 * m + 3) % R; });
            }
            (void)c1;
            clear(h);
            ++keys;
            bad_keys += bad ? 1 : 0;
            g_fail += bad ? 1 : 0;
            std::printf("WMGCHECK %s %s %lldx%lld%s: %s | WMMA vs the decode band on row 0: %zu of %zu outputs differ (%.1f %%), rel diff mean %.3g max %.3g%s\n",
                        ty.name, sh.name, (long long)sh.N, (long long)sh.K, sh.logits ? " f32" : "",
                        bad ? "ROW BYTES DEPEND ON M / PLACE" : "every row = the row alone at every M and place", nd, (size_t)sh.N,
                        100.0 * (double)nd / (double)sh.N, nfin ? sumrel / (double)nfin : 0.0, maxrel, fails.c_str());
            std::fflush(stdout);
        }
    }
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    std::printf("WMGVERDICT check: %lld (type, shape) keys, %lld launches, %lld rows compared with the row alone, %lld differ in %lld keys -> %s\n",
                keys, launches, rows_cmp, rows_bad, bad_keys,
                keys == 0 ? "NO DATA" : rows_bad || bad_keys ? "FAIL (a row's WMMA bytes depend on M or its place)" : "PASS (M- and place-invariant, M 1..130)");
}

/* --perfwmk [iters] [rounds]: where the WMMA form's time at decode heights goes. (1) WMGSCAN: synthetic q6_K / q8_0 / q5_K shapes
 * N x K at M 1 / 16 / 48, W (the stock WMMA kernel) and D (int16's decode band), D W W D: the K -> 0 intercept (a launch's fixed
 * cost) and the slope a 128-value slab. (2) WMGFORM: lever GPF's tile geometries (r4d_gg_pf_pin codes, the prefill forms) at
 * M 1 / 16 on the real dense shapes, against the stock WMMA kernel, with their bytes. */
static void scan(const Plugin& pl, int iters, int rounds) {
    if (iters <= 0) iters = 100;
    if (rounds <= 0) rounds = 2;
    Hk h;
    if (!hooks(pl, h, "--perfwmk")) return;
    const GpHooks gp = gp_hooks(pl);
    const char* mode = g_cold > 0 ? "cold" : "rot";
    auto one = [&](const MrTy& ty, const char* name, int64_t N, int64_t K, int64_t M, bool forms) {
        MrShape sh{ name, N, K, ty.name, false };
        dfree_all();
        const int64_t rb = K / ty.qk * ty.bb, wbytes = N * rb;
        const int copies = (int)std::max<int64_t>(2, (256ll << 20) / wbytes + 1);
        std::vector<uint8_t> hw;
        mr_weights(hw, ty, N, K, 0x5CA40001ull ^ (uint64_t)(N * 7 + K));
        std::vector<uint16_t> hx;
        mr_acts(hx, K, 0x5CAull ^ (uint64_t)M);
        const size_t yb = (size_t)(M * N) * 2;
        void* d_x = dalloc((size_t)(M * K) * 2);
        void* d_w = dalloc((size_t)wbytes * (size_t)copies);
        void* d_y = dalloc(yb);
        if (!d_x || !d_w || !d_y) { std::printf("  %s %s: alloc failed\n", ty.name, name); ++g_fail; return; }
        st_memcpy(d_x, hx.data(), (size_t)(M * K) * 2, hipMemcpyHostToDevice);
        for (int c = 0; c < copies; ++c) st_memcpy((uint8_t*)d_w + (size_t)c * (size_t)wbytes, hw.data(), (size_t)wbytes, hipMemcpyHostToDevice);
        MrLaunch L;
        if (!mr_setup(L, pl, sh, ty, M)) { std::printf("  %s %s: row absent\n", ty.name, name); ++g_fail; return; }
        auto launch = [&](int i) { return L.run(d_x, (uint8_t*)d_w + (size_t)(i % copies) * (size_t)wbytes, d_y); };
        auto bytes = [&]() {
            st_memset(d_y, 0xff, yb);
            const int rc = L.run(d_x, d_w, d_y);
            std::vector<uint8_t> got(yb);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) { got.clear(); return got; }
            st_memcpy(got.data(), d_y, yb, hipMemcpyDeviceToHost);
            return got;
        };
        auto tm = [&]() { for (int i = 0; i < 20; ++i) launch(i); return st_time_us([&](int i) { launch(i); }, iters, nullptr); };
        if (!forms) {
            set_wm(h);
            const long long c0 = h.wm_count();
            const bool okw = !bytes().empty() && h.wm_count() - c0 == 1;
            double d = 0, w = 0;
            if (okw)
                for (int r = 0; r < rounds; ++r) {
                    set_dec(h); d += tm();
                    set_wm(h); w += tm(); w += tm();
                    set_dec(h); d += tm();
                }
            clear(h);
            if (!okw) ++g_fail;
            std::printf("WMGSCAN %s|%s|N=%lld|K=%lld|M=%lld|slabs=%lld|blocks=%lld|D %.2f|W %.2f|%s\n", mode, ty.name, (long long)N, (long long)K,
                        (long long)M, (long long)(K / 128), (long long)((N + 255) / 256), d / (2 * rounds), w / (2 * rounds), okw ? "ok" : "W DID NOT RUN");
        } else {
            if (!gp.pin || !gp.forms) { std::printf("  WMGFORM: r4d_gg_pf_pin absent\n"); ++g_fail; return; }
            set_wm(h);
            gp.pin(0);
            const std::vector<uint8_t> ref = bytes();
            std::string line = fmt("WMGFORM %s|%s|%s|%lldx%lld|M=%lld|", mode, ty.name, name, (long long)N, (long long)K, (long long)M);
            std::vector<int> codes = gp_forms(gp, false);
            codes.insert(codes.begin(), 0);
            std::vector<double> t(codes.size(), 0.0);
            std::vector<int> same(codes.size(), 0), ran(codes.size(), 0);
            for (size_t j = 0; j < codes.size(); ++j) {
                gp.pin(codes[j]);
                const std::vector<uint8_t> got = bytes();
                ran[j] = !got.empty();
                same[j] = ran[j] && got == ref;
            }
            for (int r = 0; r < rounds; ++r)
                for (size_t j0 = 0; j0 < codes.size(); ++j0) {
                    const size_t j = (r & 1) ? codes.size() - 1 - j0 : j0;
                    if (!ran[j]) continue;
                    gp.pin(codes[j]);
                    t[j] += tm();
                }
            gp.pin(0);
            clear(h);
            for (size_t j = 0; j < codes.size(); ++j)
                line += fmt(" %s %.1f%s;", gp_name(codes[j]).c_str(), ran[j] ? t[j] / rounds : -1.0, !ran[j] ? " (rc)" : same[j] ? "" : " (bytes differ)");
            std::printf("%s\n", line.c_str());
        }
        std::fflush(stdout);
    };
    std::printf("WMG scan (%s, %d launches a reading, %d rounds): the stock WMMA kernel W and int16's decode band D on synthetic shapes; then lever GPF's geometries at M 1 / 16\n", mode, iters, rounds);
    for (const MrTy& ty : kMrTypes)
        for (int64_t N : { 256ll, 1024ll, 4096ll })
            for (int64_t K : { 256ll, 512ll, 1280ll, 2560ll, 5120ll })
                for (int64_t M : { 1ll, 16ll, 48ll }) {
                    if (std::strcmp(ty.name, "q6_K") && (N != 1024 || M == 48)) continue;   /* the other types: one N, M 1 / 16 */
                    one(ty, "syn", N, K, M, false);
                }
    for (const MrTy& ty : kMrTypes)
        for (const MrShape& sh : kMrShapes) {
            if (sh.logits || !std::strcmp(sh.name, "engram_kv") || !mr_in(sh.types, ty.name)) continue;
            for (int64_t M : { 1ll, 16ll }) one(ty, sh.name, sh.N, sh.K, M, true);
        }
    clear(h);
    dfree_all();
    std::printf("WMGVERDICT scan done\n");
}

}  // namespace wmg
