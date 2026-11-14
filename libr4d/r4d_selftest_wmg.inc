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

/* ds41-wmk: --wmk CODE[,CODE...] -- the W arm is gg_wk_kernel (the part-order WMMA form, K split across waves) under the
 * bench pin r4d_gg_wk_pin(CODE) instead of the stock WMMA kernel; --perfwmg times the first code (the pair shapes ONE
 * launch), --wmgcheck checks every code and the codes against each other. */
static std::vector<int> g_wk_codes;
static int g_wk_cur = 0;
struct Hk {
    void (*wk_pin)(int) = nullptr;
    long long (*wk_count)() = nullptr;
    int (*wk_last)() = nullptr;
    void (*wm_pin)(int) = nullptr;
    long long (*wm_count)() = nullptr;
    int (*sw)(const char*, int) = nullptr;
    MrHooks mr;
};
static bool hooks(const Plugin& pl, Hk& h, const char* who) {
    h.wm_pin = (void (*)(int))dlsym(pl.h, "r4d_gg_wm_pin");
    h.wm_count = (long long (*)())dlsym(pl.h, "r4d_gg_wm_count");
    h.sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    h.wk_pin = (void (*)(int))dlsym(pl.h, "r4d_gg_wk_pin");
    h.wk_count = (long long (*)())dlsym(pl.h, "r4d_gg_wk_count");
    h.wk_last = (int (*)())dlsym(pl.h, "r4d_gg_wk_last");
    if (!g_wk_codes.empty() && (!h.wk_pin || !h.wk_count || !h.wk_last)) {
        std::printf("%s: r4d_gg_wk_pin absent (not the ds41-wmk plugin)\n", who);
        ++g_fail;
        return false;
    }
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
    if (h.wk_pin) h.wk_pin(0);
    h.sw("ds41_wmk", 0);   /* (a libr4d without lever WMK refuses it: harmless) */
    h.mr.pin(0, 0, 0, 0); h.mr.pin_xp(0);
    if (h.mr.pin_w64) h.mr.pin_w64(0);
    h.sw("gg_gemv_m16", 0); h.sw("gg_gemv", 0);
}
static void set_dec(const Hk& h) { clear(h); h.sw("gg_gemv", 2); h.sw("gg_gemv_m16", 1); }   /* int16's decode band */
/* --wmk 1048576: the W arm is the ENGINE's path (fn4 ds41_wmk 1 over int16's band: lever WMK's per-shape code table), not a pin */
constexpr int kWkEngine = 1048576;
static void set_wm(const Hk& h) {
    clear(h);
    /* 1048576 / 1048577 / 1048578: the engine's path under ds41_wmk 1 / 2 / 3 (FINDINGS F7: the r6 table / 70724 / r5's table) */
    if (g_wk_cur >= kWkEngine && g_wk_cur <= kWkEngine + 2) { h.sw("gg_gemv", 2); h.sw("gg_gemv_m16", 1); h.sw("ds41_wmk", 1 + (g_wk_cur - kWkEngine)); }
    else if (g_wk_cur) h.wk_pin(g_wk_cur);
    else h.wm_pin(1);
}
static void set_stock_wm(const Hk& h) { clear(h); h.wm_pin(1); }
static long long wcount(const Hk& h) { return g_wk_cur ? h.wk_count() : h.wm_count(); }

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
    g_wk_cur = g_wk_codes.empty() ? 0 : g_wk_codes[0];
    const bool wk = g_wk_cur != 0;
    if (wk) std::printf("WMK: the W arm is gg_wk_kernel under r4d_gg_wk_pin(%d) (pair shapes ONE launch)\n", g_wk_cur);
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
                const long long c0 = wcount(h);
                const std::vector<uint8_t> wm = bytes(wk, &rcw);
                const long long took = wcount(h) - c0;
                const int wkform = wk ? h.wk_last() : 0;
                const bool dec_same = dec == stock, wm_ran = took == (pair && !wk ? 2 : 1);
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
                            for (int i = 0; i < 20; ++i) launch(i, !w || wk);
                            const double us = st_time_us([&](int i) { launch(i, !w || wk); }, iters, nullptr);
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
                            "dec==stock %s|wm launches %lld wk %d|wm!=dec %zu of %zu (%.2f %%) maxrel %.3g|rc %d %d %d|D:%s|W:%s\n",
                            mode, ty.name, sh.name, (long long)sh.N, (long long)sh.N2, (long long)sh.K, (long long)M, calls, dm, dmn, dmx,
                            dform.c_str(), wmn_, wmn, wmx, dm > 0 ? wmn_ / dm : 0.0, dec_same ? "yes" : "NO", took, wkform, ndiff, nout,
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

/* --wmgcheck with --wmk codes (ds41-wmk): gg_wk_kernel under every code. Per (code, type, shape): a row alone (M 1) vs the
 * same row inside M 2..48 from row 0, windows at other starts, 49..130-row launches of permuted rows -- counted apart for
 * the finite rows, the inf row (row 5 holds one +inf) and the near-overflow row (row 17 holds three values of
 * 8.5e37..3.4e38). Info: the outputs of the 46 finite rows that differ from int16's decode band (the one-time reference
 * change) and their size. WMKCROSS: the codes' bytes against each other (one part order -> one set of bytes whatever
 * the geometry) and a P = 1 code against the stock WMMA kernel. WMKORDER: the part order itself, from the stock WMMA
 * kernel's f32 outputs on each part's columns folded on the host in ascending part order. */
static int wk_parts(int64_t K) { const int64_t S = K / 128; return S % 8 == 0 ? 8 : S % 4 == 0 ? 4 : S % 2 == 0 ? 2 : 1; }
static void order(const Plugin& pl, const Hk& h, long long* keys, long long* bad_keys) {
    const int64_t N = 328, M = 6;
    for (const MrTy& ty : kMrTypes) {
        const bool q32 = ty.qk == 32;
        const RadKernelInfo* rl = find(pl, q32 ? "logits_gemm_ggml_qk32_gemv" : "logits_gemm_ggml_qk256_gemv");
        if (!rl) { std::printf("  WMKORDER %s: logits row absent\n", ty.name); ++g_fail; continue; }
        for (int64_t K : { 2048ll, 4096ll, 6144ll, 1024ll, 512ll }) {
            if (K % ty.qk) continue;
            const int P = wk_parts(K);
            const int64_t Kp = K / P, rb = K / ty.qk * ty.bb, rbp = rb / P;
            if (Kp % 256) continue;   /* the stock WMMA kernel does not take the part */
            dfree_all();
            std::vector<uint8_t> hw;
            mr_weights(hw, ty, N, K, 0x0DE40001ull ^ (uint64_t)(N * 7 + K));
            std::vector<uint16_t> hx;
            mr_acts(hx, K, 0x0DEull ^ (uint64_t)K);
            void* d_x = dalloc((size_t)(M * K) * 2);
            void* d_w = dalloc(hw.size());
            void* d_wp = dalloc((size_t)(N * rbp));
            void* d_y = dalloc((size_t)(M * N) * 4);
            if (!d_x || !d_w || !d_wp || !d_y) { std::printf("  WMKORDER %s K=%lld: alloc\n", ty.name, (long long)K); ++g_fail; continue; }
            st_memcpy(d_x, hx.data(), (size_t)(M * K) * 2, hipMemcpyHostToDevice);
            st_memcpy(d_w, hw.data(), hw.size(), hipMemcpyHostToDevice);
            auto runl = [&](void* x, void* w, int64_t Kc, int64_t rbc, std::vector<float>& out) -> bool {
                std::vector<RadTensor> t = { gqst::TP(x, RAD_BF16, M, Kc, K), T(w, RAD_U8, { N, rbc }), T(d_y, RAD_F32, { M, N }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n_vocab", N), PI("n_embd", Kc), PS("type", ty.name), PS("dtype", "bf16") };
                RadArgs a{};
                a.t = t.data(); a.n_t = 3; a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
                st_memset(d_y, 0xff, (size_t)(M * N) * 4);
                const int rc = rl->launch(&a, nullptr);
                if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return false;
                out.resize((size_t)(M * N));
                st_memcpy(out.data(), d_y, (size_t)(M * N) * 4, hipMemcpyDeviceToHost);
                return true;
            };
            /* the host fold of the stock kernel's per-part f32 outputs */
            std::vector<float> tot, part;
            bool okp = true;
            g_wk_cur = 0;
            set_stock_wm(h);
            for (int p = 0; okp && p < P; ++p) {
                std::vector<uint8_t> hp((size_t)(N * rbp));
                for (int64_t n = 0; n < N; ++n) std::memcpy(&hp[(size_t)(n * rbp)], &hw[(size_t)(n * rb + p * rbp)], (size_t)rbp);
                st_memcpy(d_wp, hp.data(), hp.size(), hipMemcpyHostToDevice);
                const long long c0 = h.wm_count();
                okp = runl((uint16_t*)d_x + p * Kp, d_wp, Kp, rbp, part) && h.wm_count() - c0 == 1;
                if (!okp) break;
                if (p == 0) tot = part;
                else for (size_t i = 0; i < tot.size(); ++i) tot[i] = tot[i] + part[i];
            }
            std::string line = fmt("WMKORDER %s %lldx%lld M=%lld f32 | P %d parts of %lld |", ty.name, (long long)N, (long long)K, (long long)M, P, (long long)Kp);
            bool bad = !okp;
            if (!okp) line += " the stock part launches failed;";
            for (int code : g_wk_codes) {
                if (!okp) break;
                if (code & 16) continue;
                g_wk_cur = code;
                set_wm(h);
                std::vector<float> got;
                const long long c0 = h.wk_count();
                const bool okr = runl(d_x, d_w, K, rb, got) && h.wk_count() - c0 == 1;
                size_t dfin = 0, dinf = 0;
                if (okr)
                    for (size_t i = 0; i < got.size(); ++i) {
                        const bool same = !std::memcmp(&got[i], &tot[i], 4) || (std::isnan(got[i]) && std::isnan(tot[i]));
                        if (!same) { if ((int64_t)(i / (size_t)N) == 5) ++dinf; else ++dfin; }
                    }
                line += fmt(" code %d (kp*100+P %d): %s, finite rows %zu of %lld differ, inf row %zu of %lld;", code, okr ? h.wk_last() : -1,
                            okr ? "ran" : "LAUNCH FAILED", dfin, (long long)(5 * N), dinf, (long long)N);
                if (!okr || dfin || dinf) bad = true;
            }
            clear(h);
            ++*keys;
            if (bad) { ++*bad_keys; ++g_fail; }
            std::printf("%s %s\n", line.c_str(), bad ? "FAIL" : "ok");
            std::fflush(stdout);
        }
    }
    g_wk_cur = 0;
}

static void checkk(const Plugin& pl, const char* tlist) {
    Hk h;
    if (!hooks(pl, h, "--wmgcheck --wmk")) return;
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const int64_t R = 48, RB = 130;
    const std::vector<int64_t> Ms = { 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 47, 48 };
    const int64_t wins[][2] = { { 1, 1 }, { 1, 16 }, { 5, 8 }, { 15, 2 }, { 16, 16 }, { 17, 16 }, { 7, 41 }, { 31, 17 }, { 33, 15 }, { 47, 1 }, { 40, 8 }, { 3, 45 } };
    const int64_t bigs[] = { 49, 64, 96, 130 };
    long long keys = 0, bad_keys = 0, launches = 0, cmpc[3] = { 0, 0, 0 }, badc[3] = { 0, 0, 0 }, xkeys = 0, xbad = 0;
    std::printf("WMK byte check (gg_wk_kernel, codes"); for (int c : g_wk_codes) std::printf(" %d", c);
    std::printf("): a row's bytes at M 1 vs the same row inside M 2..48 (from row 0), windows at other starts, 49..130-row launches of permuted rows\n");
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const MrShape& sh : kMrShapes) {
            if (sh.logits && std::strcmp(ty.name, "q6_K")) continue;
            if (sh.K % ty.qk) continue;
            if (!std::strcmp(sh.name, "engram_kv")) continue;
            dfree_all();
            std::vector<uint8_t> hw;
            mr_weights(hw, ty, sh.N, sh.K, 0x9E3779B97F4A7C15ull ^ (uint64_t)(sh.N * 131 + sh.K) ^ 0x776du);
            std::vector<uint16_t> hx;
            mr_acts(hx, sh.K, 0x1234567ull ^ (uint64_t)sh.K ^ 0x776du);
            std::vector<uint16_t> hb((size_t)(RB * sh.K));
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
            /* int16's decode band on the 48 rows (a decode row's bytes do not depend on M: lever D2), the stock WMMA kernel's rows alone */
            std::vector<uint8_t> dec48, wmalone((size_t)R * rowb), one;
            g_wk_cur = 0;
            set_dec(h);
            bool okd = run(d_x, R, dec48);
            set_stock_wm(h);
            for (int64_t r = 0; okd && r < R; ++r) {
                okd = run((uint16_t*)d_x + r * sh.K, 1, one);
                if (okd) std::memcpy(&wmalone[(size_t)r * rowb], one.data(), rowb);
            }
            if (!okd) { std::printf("  FAIL %s %s: the decode band / stock WMMA launches\n", ty.name, sh.name); ++g_fail; continue; }
            std::vector<std::vector<uint8_t>> alones;
            for (int code : g_wk_codes) {
                g_wk_cur = code;
                set_wm(h);
                std::vector<uint8_t> alone((size_t)R * rowb);
                const long long c0 = h.wk_count();
                bool okr = true;
                for (int64_t r = 0; okr && r < R; ++r) {
                    okr = run((uint16_t*)d_x + r * sh.K, 1, one);
                    if (okr) std::memcpy(&alone[(size_t)r * rowb], one.data(), rowb);
                }
                const int form1 = h.wk_last();
                ++keys;
                if (!okr || h.wk_count() - c0 != R) {
                    std::printf("  FAIL code %d %s %s: M 1 launches (wk launches %lld of %lld)\n", code, ty.name, sh.name, h.wk_count() - c0, (long long)R);
                    ++g_fail; ++bad_keys; alones.push_back({}); continue;
                }
                /* info: the finite rows against the decode band */
                size_t nd = 0, nfin = 0, nout = 0; double maxrel = 0, sumrel = 0;
                for (int64_t r = 0; r < R; ++r) {
                    if (r == 5 || r == 17) continue;
                    for (size_t e = 0; e < rowb; e += ob) {
                        const size_t o = (size_t)r * rowb + e;
                        float a, b;
                        if (ob == 4) { std::memcpy(&a, &alone[o], 4); std::memcpy(&b, &dec48[o], 4); }
                        else { uint16_t ua, ub; std::memcpy(&ua, &alone[o], 2); std::memcpy(&ub, &dec48[o], 2); a = bf16f(ua); b = bf16f(ub); }
                        ++nout;
                        if (std::memcmp(&alone[o], &dec48[o], ob)) ++nd;
                        if (std::isfinite(a) && std::isfinite(b)) {
                            const double den = std::max(std::fabs((double)a), std::fabs((double)b));
                            const double rel = den > 0 ? std::fabs((double)a - (double)b) / den : 0.0;
                            maxrel = std::max(maxrel, rel); sumrel += rel; ++nfin;
                        }
                    }
                }
                long long kc[3] = { 0, 0, 0 }, kb[3] = { 0, 0, 0 };
                std::string fails;
                auto cmp = [&](const std::vector<uint8_t>& got, int64_t M, const char* what, auto rowof) {
                    for (int64_t m = 0; m < M; ++m) {
                        const int64_t r = rowof(m);
                        const int cl = r == 5 ? 1 : r == 17 ? 2 : 0;
                        ++kc[cl];
                        if (std::memcmp(&got[(size_t)m * rowb], &alone[(size_t)r * rowb], rowb)) {
                            ++kb[cl];
                            size_t e = 0;
                            while (e < rowb && got[(size_t)m * rowb + e] == alone[(size_t)r * rowb + e]) ++e;
                            if (fails.size() < 500) fails += fmt(" %s M=%lld place %lld (row %lld) first output %zu;", what, (long long)M, (long long)m, (long long)r, e / ob);
                        }
                    }
                };
                std::vector<uint8_t> got;
                int rcbad = 0, form16 = 0, form48 = 0;
                for (int64_t M : Ms) {
                    if (!run(d_x, M, got)) { ++rcbad; fails += fmt(" M=%lld rc;", (long long)M); continue; }
                    if (M == 16) form16 = h.wk_last();
                    if (M == 48) form48 = h.wk_last();
                    cmp(got, M, "from0", [&](int64_t m) { return m; });
                }
                for (const auto& w : wins) {
                    const int64_t st = w[0], M = w[1];
                    if (!run((uint16_t*)d_x + st * sh.K, M, got)) { ++rcbad; fails += fmt(" win %lld+%lld rc;", (long long)st, (long long)M); continue; }
                    cmp(got, M, "win", [&](int64_t m) { return st + m; });
                }
                for (int64_t M : bigs) {
                    if (!run(d_b, M, got)) { ++rcbad; fails += fmt(" big M=%lld rc;", (long long)M); continue; }
                    cmp(got, M, "big", [&](int64_t m) { return (7 * m + 3) % R; });
                }
                const bool bad = rcbad || kb[0] || kb[1] || kb[2];
                for (int c = 0; c < 3; ++c) { cmpc[c] += kc[c]; badc[c] += kb[c]; }
                bad_keys += bad ? 1 : 0;
                g_fail += bad ? 1 : 0;
                std::printf("WMKCHECK code %d %s %s %lldx%lld%s | forms kp*100+P M1 %d M16 %d M48 %d | rows differing from the row alone: finite %lld of %lld, inf row %lld of %lld, "
                            "near-overflow row %lld of %lld -> %s | vs int16's decode band on the 46 finite rows: %zu of %zu outputs differ (%.3f %%), rel diff mean %.3g max %.3g%s\n",
                            code, ty.name, sh.name, (long long)sh.N, (long long)sh.K, sh.logits ? " f32" : "", form1, form16, form48, kb[0], kc[0], kb[1], kc[1], kb[2], kc[2],
                            bad ? "ROW BYTES DEPEND ON M / PLACE" : "invariant", nd, nout, nout ? 100.0 * (double)nd / (double)nout : 0.0,
                            nfin ? sumrel / (double)nfin : 0.0, maxrel, fails.c_str());
                std::fflush(stdout);
                alones.push_back(std::move(alone));
            }
            /* the codes against each other, a P = 1 code against the stock WMMA kernel */
            std::string xl;
            bool xb = false;
            int base = -1;
            for (size_t i = 0; i < g_wk_codes.size(); ++i) {
                if (alones[i].empty()) continue;
                if (g_wk_codes[i] & 16) {
                    const bool same = alones[i] == wmalone;
                    xl += fmt(" code %d (P 1) == the stock WMMA kernel: %s;", g_wk_codes[i], same ? "yes" : "NO");
                    xb |= !same;
                } else if (base < 0) {
                    base = (int)i;
                } else {
                    const bool same = alones[i] == alones[(size_t)base];
                    xl += fmt(" code %d == code %d: %s;", g_wk_codes[i], g_wk_codes[(size_t)base], same ? "yes" : "NO");
                    xb |= !same;
                }
            }
            ++xkeys; xbad += xb ? 1 : 0; g_fail += xb ? 1 : 0;
            std::printf("WMKCROSS %s %s %lldx%lld (48 rows alone):%s\n", ty.name, sh.name, (long long)sh.N, (long long)sh.K, xl.c_str());
            std::fflush(stdout);
            clear(h);
        }
    }
    long long okeys = 0, obad = 0;
    order(pl, h, &okeys, &obad);
    g_wk_cur = 0;
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    std::printf("WMGVERDICT wmk check: %lld (code, type, shape) keys, %lld launches; rows compared with the row alone / differing: finite %lld / %lld, inf row %lld / %lld, "
                "near-overflow row %lld / %lld, %lld bad keys; cross-geometry %lld keys %lld bad; part order vs the host fold %lld keys %lld bad -> %s\n",
                keys, launches, cmpc[0], badc[0], cmpc[1], badc[1], cmpc[2], badc[2], bad_keys, xkeys, xbad, okeys, obad,
                keys == 0 ? "NO DATA" : (bad_keys || xbad || obad) ? "FAIL" : "PASS (M- and place-invariant M 1..130, geometry-invariant, the part order as defined)");
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

/* --wmkroute [types] (ds41-wmk2): lever WMK's engine path on the card. With int16's decode band set (gg_gemv 2 +
 * gg_gemv_m16 1) and fn4 ds41_wmk 1, every dense op row the engine issues -- gemm_nt_ggml, logits_gemm_ggml (f32 out),
 * gemm_nt_ggml_pair (a shared and a separate second activation), rmsnorm_gemm_nt_ggml (fold a), swiglu_clamped_gemm_nt_ggml
 * (fold e) -- at DeepSeek-V4.1's rank-0 / rank-1..3 shapes and M 1, 2, 5, 8, 16, 17, 24, 33, 47, 48, 49, 64, 97, 130 (row i of a
 * call holds content i % 49):
 *   route   rad_wmk_stats: the part-order launches a call issues -- one at every M (a pair above 48 rows: its two GEMMs;
 *           stage 3's prefill rows count as part-order launches), no decline; ds41_wmk 0: none;
 *   bytes   every row of every call = the same content alone (M 1) under the switch (y, y2; h for the folds) -- one
 *           arithmetic at every row count, prefill heights included;
 *           plain / logits / pair outputs = the bench pin r4d_gg_wk_pin(70724) at the same M (the same kernel and code);
 *           the folds' h = h under ds41_wmk 0 (the producer's bytes either way), and h, y = the producer's own row
 *           (rmsnorm_bf16 / swiglu_clamped_bf16) then gemm_nt_ggml under the switch (the fold in the form's staging);
 *   info    the form gg_pf picked at M 130 (r4d_gg_wkpf_last).
 * WMKROUTE a (type, op), WMGVERDICT wmkroute. */
static void route(const Plugin& pl, const char* tlist) {
    Hk h;
    if (!hooks(pl, h, "--wmkroute")) return;
    auto wst = (int (*)(long long*, int))dlsym(pl.h, "rad_wmk_stats");
    if (!wst || !h.wk_pin) { std::printf("--wmkroute: rad_wmk_stats / r4d_gg_wk_pin absent (not the ds41-wmk2 plugin)\n"); ++g_fail; return; }
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const int kCode = 70724;
    const std::vector<int64_t> Ms = { 1, 2, 5, 8, 16, 17, 24, 33, 47, 48, 49, 64, 97, 130 };
    const int64_t R = 49, RB = 2048;   /* RB: the prefill-height check of gg_wq (stage 3) runs a 2048-row call */
    auto wkpf_last = (int (*)())dlsym(pl.h, "r4d_gg_wkpf_last");
    auto pf_pin = (void (*)(int))dlsym(pl.h, "r4d_gg_pf_pin");
    auto wkpf_ts = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_ts");
    auto wkpf_follow = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_follow");
    struct Op { const char* name; int kind; int64_t N, N2, K; const char* types; bool a2; };   /* kind 0 gemm, 1 logits, 2 pair, 3 fold a, 4 fold e */
    const Op ops[] = {
        { "attn_q_b(fold a)", 3, 8192, 0, 1280, "q5_K,q6_K", false },  { "attn_q_b", 0, 8192, 0, 1280, "q5_K,q6_K", false },
        { "q_a+attn_kv(pair)", 2, 1280, 512, 5120, "q5_K,q6_K", false }, { "attn_output_a g0+g1(pair a2)", 2, 1024, 1024, 4096, "q5_K,q6_K", true },
        { "attn_output_b", 0, 5120, 0, 2048, "q5_K,q6_K", false },     { "shexp_gate_up", 0, 1536, 0, 5120, "q6_K,q8_0", false },
        { "shexp_gate_up_w512", 0, 1024, 0, 5120, "q6_K,q8_0", false }, { "shexp_down(fold e)", 4, 5120, 0, 768, "q6_K,q8_0", false },
        { "shexp_down_w512(fold e)", 4, 5120, 0, 512, "q6_K,q8_0", false }, { "engram_kv_e2", 0, 6400, 0, 6144, "q6_K", false },
        { "head", 1, 32320, 0, 5120, "q6_K", false },
    };
    long long keys = 0, bad_keys = 0, calls = 0;
    std::printf("WMK route check (fn4 ds41_wmk 1 over int16's decode band; bench pin code %d as the byte reference)\n", kCode);
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        const bool q32 = ty.qk == 32;
        for (const Op& op : ops) {
            if (!mr_in(op.types, ty.name) || op.K % ty.qk) continue;
            const char* rn = op.kind == 0 ? (q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv")
                           : op.kind == 1 ? (q32 ? "logits_gemm_ggml_qk32_gemv" : "logits_gemm_ggml_qk256_gemv")
                           : op.kind == 2 ? (q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256")
                           : op.kind == 3 ? (q32 ? "rmsnorm_gemm_nt_ggml_qk32" : "rmsnorm_gemm_nt_ggml_qk256")
                                          : (q32 ? "swiglu_clamped_gemm_nt_ggml_qk32" : "swiglu_clamped_gemm_nt_ggml_qk256");
            const RadKernelInfo* row = find(pl, rn);
            if (!row) { std::printf("  WMKROUTE %s %s: row %s absent\n", ty.name, op.name, rn); ++g_fail; ++bad_keys; ++keys; continue; }
            const RadKernelInfo* rprod = op.kind == 3 ? find(pl, "rmsnorm_bf16") : op.kind == 4 ? find(pl, "swiglu_clamped_bf16") : nullptr;
            const RadKernelInfo* rgem = find(pl, q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv");
            if (op.kind >= 3 && (!rprod || !rgem)) { std::printf("  WMKROUTE %s %s: producer / gemm row absent\n", ty.name, op.name); ++g_fail; ++bad_keys; ++keys; continue; }
            dfree_all();
            const int64_t K = op.K, N = op.N, N2 = op.N2, rb = K / ty.qk * ty.bb, xc = op.kind == 4 ? 2 * K : K;
            const int64_t NY = N + N2, hw_ = op.kind >= 3 ? K : 0;
            std::vector<uint8_t> hw, hw2;
            mr_weights(hw, ty, N, K, 0x57A0001ull ^ (uint64_t)(N * 7 + K));
            if (N2) mr_weights(hw2, ty, N2, K, 0x57A0002ull ^ (uint64_t)(N2 * 7 + K));
            std::vector<uint16_t> hx, hx2;
            mr_acts(hx, xc, 0x57ACull ^ (uint64_t)(xc + op.kind));
            hx.resize((size_t)(RB * xc));
            std::memcpy(&hx[(size_t)(48 * xc)], &hx[(size_t)(7 * xc)], (size_t)xc * 2);   /* row 48 = row 7's content */
            for (int64_t i = R; i < RB; ++i) std::memcpy(&hx[(size_t)(i * xc)], &hx[(size_t)((i % R) * xc)], (size_t)xc * 2);
            if (op.a2) {
                mr_acts(hx2, K, 0x57ADull ^ (uint64_t)K);
                hx2.resize((size_t)(RB * K));
                std::memcpy(&hx2[(size_t)(48 * K)], &hx2[(size_t)(11 * K)], (size_t)K * 2);
                for (int64_t i = R; i < RB; ++i) std::memcpy(&hx2[(size_t)(i * K)], &hx2[(size_t)((i % R) * K)], (size_t)K * 2);
            }
            std::vector<float> g((size_t)K);
            for (int64_t k = 0; k < K; ++k) g[(size_t)k] = (k % 7 == 0) ? 0.0f : (k % 5 == 0) ? -0.25f : 0.8f + 0.001f * (float)(k % 97);
            const size_t ob = op.kind == 1 ? 4 : 2, yrow = (size_t)NY * ob, hrow = (size_t)hw_ * 2;
            void* d_w = dalloc(hw.size() + hw2.size());
            void* d_x = dalloc(hx.size() * 2);
            void* d_x2 = op.a2 ? dalloc(hx2.size() * 2) : nullptr;
            void* d_g = dalloc((size_t)K * 4);
            void* d_y = dalloc((size_t)RB * yrow);
            void* d_h = hrow ? dalloc((size_t)RB * hrow) : nullptr;
            if (!d_w || !d_x || !d_g || !d_y || (op.a2 && !d_x2) || (hrow && !d_h)) { std::printf("  WMKROUTE %s %s: alloc\n", ty.name, op.name); ++g_fail; continue; }
            st_memcpy(d_w, hw.data(), hw.size(), hipMemcpyHostToDevice);
            if (N2) st_memcpy((uint8_t*)d_w + hw.size(), hw2.data(), hw2.size(), hipMemcpyHostToDevice);
            st_memcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
            if (op.a2) st_memcpy(d_x2, hx2.data(), hx2.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(d_g, g.data(), (size_t)K * 4, hipMemcpyHostToDevice);
            /* one call of M rows from activation row r0; out = y rows (then h rows) */
            auto call = [&](int64_t r0, int64_t M, std::vector<uint8_t>& out) -> int {
                uint16_t* x = (uint16_t*)d_x + r0 * xc;
                st_memset(d_y, 0xff, (size_t)M * yrow);
                if (hrow) st_memset(d_h, 0x5a, (size_t)M * hrow);
                int rc;
                if (op.kind == 0) {
                    rc = k1_launch(row, { T(x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16") });
                } else if (op.kind == 1) {
                    rc = k1_launch(row, { T(x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_y, RAD_F32, { M, N }) },
                                   { PI("M", M), PI("n_vocab", N), PI("n_embd", K), PS("type", ty.name), PS("dtype", "bf16") });
                } else if (op.kind == 2) {
                    std::vector<RadTensor> t = { gqst::TP(x, RAD_BF16, M, K, K), T(d_w, RAD_U8, { N, rb }),
                                                 gqst::TP(op.a2 ? (void*)((uint16_t*)d_x2 + r0 * K) : (void*)x, RAD_BF16, M, K, K),
                                                 T((uint8_t*)d_w + hw.size(), RAD_U8, { N2, rb }),
                                                 gqst::TP(d_y, RAD_BF16, M, N, NY), gqst::TP((uint16_t*)d_y + N, RAD_BF16, M, N2, NY) };
                    if (!op.a2) absent(t[2]);
                    rc = k1_launch(row, t, { PI("M", M), PI("N", N), PI("N2", N2), PI("K", K), PS("type", ty.name), PS("dtype", "bf16") });
                } else if (op.kind == 3) {
                    rc = k1_launch(row, { T(x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_g, RAD_F32, { K }),
                                          T(d_h, RAD_BF16, { M, K }), T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16"), PF("eps", 1e-6) });
                } else {
                    rc = k1_launch(row, { T(x, RAD_BF16, { M, 2 * K }), T(d_w, RAD_U8, { N, rb }), T(d_h, RAD_BF16, { M, K }),
                                          T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16"), PF("limit", 10.0) });
                }
                ++calls;
                if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return rc != RAD_OK ? rc : -999;
                out.resize((size_t)M * (yrow + hrow));
                st_memcpy(out.data(), d_y, (size_t)M * yrow, hipMemcpyDeviceToHost);
                if (hrow) st_memcpy(out.data() + (size_t)M * yrow, d_h, (size_t)M * hrow, hipMemcpyDeviceToHost);
                return RAD_OK;
            };
            auto yrow_of = [&](const std::vector<uint8_t>& o, int64_t m) { return std::vector<uint8_t>(o.begin() + m * yrow, o.begin() + (m + 1) * yrow); };
            auto hrow_of = [&](const std::vector<uint8_t>& o, int64_t M, int64_t m) {
                return std::vector<uint8_t>(o.begin() + M * yrow + m * hrow, o.begin() + M * yrow + (m + 1) * hrow);
            };
            auto stats = [&](long long* v) { for (int i = 0; i < 4; ++i) v[i] = 0; wst(v, 4); };
            auto wmk_on = [&](int on) { set_dec(h); h.sw("ds41_wmk", on); };
            std::string fails;
            long long bad = 0, rows_cmp = 0, rows_bad = 0, launch_bad = 0, decl = 0, pin_bad = 0, hoff_bad = 0, off_bad = 0, unf_bad = 0;
            /* a fold's two launches (the producer's row, then the gemm row on its h) under the current switch state */
            auto unfused = [&](int64_t M, std::vector<uint8_t>& out) -> int {
                st_memset(d_y, 0xff, (size_t)M * yrow);
                st_memset(d_h, 0x5a, (size_t)M * hrow);
                int rc = op.kind == 3
                             ? k1_launch(rprod, { T(d_x, RAD_BF16, { M, K }), T(d_g, RAD_F32, { K }), T(d_h, RAD_BF16, { M, K }) },
                                         { PI("M", M), PI("n", K), PF("eps", 1e-6), PS("dtype", "bf16") })
                             : k1_launch(rprod, { T(d_x, RAD_BF16, { M, 2 * K }), T(d_h, RAD_BF16, { M, K }) },
                                         { PI("M", M), PI("n", K), PF("limit", 10.0), PS("dtype", "bf16") });
                if (rc == RAD_OK)
                    rc = k1_launch(rgem, { T(d_h, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16") });
                if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return rc != RAD_OK ? rc : -999;
                out.resize((size_t)M * (yrow + hrow));
                st_memcpy(out.data(), d_y, (size_t)M * yrow, hipMemcpyDeviceToHost);
                st_memcpy(out.data() + (size_t)M * yrow, d_h, (size_t)M * hrow, hipMemcpyDeviceToHost);
                return RAD_OK;
            };
            int form130 = -1;
            /* the rows alone under the switch */
            wmk_on(1);
            std::vector<std::vector<uint8_t>> ay(R), ah(R);
            std::vector<uint8_t> o;
            for (int64_t r = 0; r < R; ++r) {
                long long v0[4], v1[4];
                stats(v0);
                const int rc = call(r, 1, o);
                stats(v1);
                if (rc != RAD_OK) { ++bad; if (fails.size() < 400) fails += fmt(" alone row %lld rc %d;", (long long)r, rc); continue; }
                if (v1[1] - v0[1] != 1) ++launch_bad;
                decl += v1[3] - v0[3];
                ay[(size_t)r] = yrow_of(o, 0);
                if (hrow) ah[(size_t)r] = hrow_of(o, 1, 0);
            }
            for (int64_t M : Ms) {
                if (M == 1) continue;
                wmk_on(1);
                long long v0[4], v1[4];
                stats(v0);
                const int rc = call(0, M, o);
                stats(v1);
                if (rc != RAD_OK) { ++bad; if (fails.size() < 400) fails += fmt(" M=%lld rc %d;", (long long)M, rc); continue; }
                const long long want_l = M > 48 && op.kind == 2 ? 2 : 1;
                if (v1[1] - v0[1] != want_l) { ++launch_bad; if (fails.size() < 400) fails += fmt(" M=%lld launches %lld;", (long long)M, v1[1] - v0[1]); }
                decl += v1[3] - v0[3];
                long long nb = 0;
                for (int64_t m = 0; m < M; ++m) {
                    bool same = yrow_of(o, m) == ay[(size_t)(m % R)];
                    if (hrow) same = same && hrow_of(o, M, m) == ah[(size_t)(m % R)];
                    if (!same) ++nb;
                }
                rows_cmp += M; rows_bad += nb;
                if (nb && fails.size() < 400) fails += fmt(" M=%lld %lld rows differ from alone;", (long long)M, nb);
                if (M == 130 && wkpf_last) form130 = wkpf_last();
                /* the bench pin's bytes (plain / logits / pair) and h under the switch off (folds) */
                if (op.kind <= 2 && M <= 48) {
                    h.sw("ds41_wmk", 0);
                    h.wk_pin(kCode);
                    std::vector<uint8_t> op_;
                    if (call(0, M, op_) != RAD_OK || op_ != o) { ++pin_bad; if (fails.size() < 400) fails += fmt(" M=%lld != pin;", (long long)M); }
                    h.wk_pin(0);
                }
                if (hrow) {   /* the fold = its two launches under the switch (y and h) */
                    std::vector<uint8_t> ou;
                    wmk_on(1);
                    if (unfused(M, ou) != RAD_OK || ou != o) { ++unf_bad; if (fails.size() < 400) fails += fmt(" M=%lld fold != two launches;", (long long)M); }
                }
                if (hrow && M <= 48) {
                    wmk_on(0);
                    std::vector<uint8_t> oo;
                    if (call(0, M, oo) != RAD_OK) ++hoff_bad;
                    else for (int64_t m = 0; m < M; ++m) if (hrow_of(oo, M, m) != hrow_of(o, M, m)) { ++hoff_bad; break; }
                }
                if (M == 16) {   /* the switch off: no part-order launch */
                    wmk_on(0);
                    std::vector<uint8_t> oo;
                    stats(v0);
                    call(0, M, oo);
                    stats(v1);
                    if (v1[1] - v0[1] != 0) ++off_bad;
                }
            }
            /* the prefill rows in every GPF form built with the part fold (pinned through gg_pf's bench pin): M 130 = alone */
            long long form_bad = 0;
            std::string forms_ok;
            if (pf_pin && op.kind <= 1) {
                for (int fc : { 0, 322, 422, 426, 326, 321, 222 })
                    for (int ts : { 1, 2 }) {   /* the running total in registers / in scratch (form 0, the stock kernel: registers) */
                        if (!wkpf_ts && ts == 2) continue;
                        wmk_on(1);
                        pf_pin(fc);
                        if (wkpf_follow) wkpf_follow(1);
                        if (wkpf_ts) wkpf_ts(ts);
                        std::vector<uint8_t> of;
                        const int rc = call(0, 130, of);
                        const int used = wkpf_last ? wkpf_last() : -1;
                        pf_pin(0);
                        if (wkpf_follow) wkpf_follow(0);
                        if (wkpf_ts) wkpf_ts(0);
                        long long nb = 0;
                        if (rc != RAD_OK) nb = -1;
                        else for (int64_t m = 0; m < 130; ++m) if (yrow_of(of, m) != ay[(size_t)(m % R)]) ++nb;
                        forms_ok += fmt(" %d%s:%s", fc, ts == 2 ? "s" : "r", nb == 0 && used == fc ? "ok" : nb < 0 ? "rc" : used != fc ? "notused" : "DIFF");
                        if (nb != 0 || used != fc) ++form_bad;
                    }
            }
            /* stage 3: W's own prefill forms (gg_wq_kernel 9xxx, W form 0 and the engine's table -1) at M 130 and 2048 = alone */
            auto wkpf_def = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_default");
            if (wkpf_def && op.kind <= 1) {
                for (int fc : { 9312, 9311, 9412, 9411, 9321, 9221, 9325, 9316, 9415, 9425, 9371, 9375, 9461, 9475, 9821, 9871, 0, -1 })
                    for (int64_t MM : { (int64_t)130, (int64_t)2048 }) {
                        wmk_on(1);
                        if (wkpf_follow) wkpf_follow(0);
                        wkpf_def(fc);
                        std::vector<uint8_t> of;
                        const int rc = call(0, MM, of);
                        const int used = wkpf_last ? wkpf_last() : -1;
                        wkpf_def(-1);
                        long long nb = 0;
                        if (rc != RAD_OK) nb = -1;
                        else for (int64_t m = 0; m < MM; ++m) if (yrow_of(of, m) != ay[(size_t)(m % R)]) ++nb;
                        const bool uok = fc < 0 || used == fc;
                        forms_ok += fmt(" %d@%lld:%s", fc, (long long)MM, nb == 0 && uok ? "ok" : nb < 0 ? "rc" : !uok ? "notused" : "DIFF");
                        if (nb != 0 || !uok) ++form_bad;
                    }
            }
            wmk_on(0);
            clear(h);
            const bool kb = bad || rows_bad || launch_bad || decl || pin_bad || hoff_bad || off_bad || unf_bad || form_bad;
            ++keys;
            if (kb) { ++bad_keys; ++g_fail; }
            std::printf("WMKROUTE %s %s %lldx%lld%s | rows vs alone (M 2..130): %lld of %lld differ | launches a call wrong %lld, declines %lld | "
                        "== bench pin %d: %s | h == switch off: %s | fold == its two launches: %s | off: no launch %s | M 130 GPF form %d | pinned forms at M 130:%s -> %s%s\n",
                        ty.name, op.name, (long long)NY, (long long)K, op.kind == 1 ? " f32" : "", rows_bad, rows_cmp, launch_bad, decl, kCode,
                        op.kind <= 2 ? (pin_bad ? "NO" : "yes") : "n/a", hrow ? (hoff_bad ? "NO" : "yes") : "n/a", hrow ? (unf_bad ? "NO" : "yes") : "n/a",
                        off_bad ? "NO" : "yes", form130, forms_ok.empty() ? " n/a" : forms_ok.c_str(),
                        kb ? "FAIL" : "ok", fails.c_str());
            std::fflush(stdout);
        }
    }
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    std::printf("WMGVERDICT wmkroute: %lld (type, op) keys, %lld calls, %lld bad -> %s\n", keys, calls, bad_keys,
                keys == 0 ? "NO DATA" : bad_keys ? "FAIL" : "PASS (every dense op row in the part order at M 1..130, row bytes M-invariant across the decode and prefill forms, = the bench pin)");
}

/* --wmkref [types] (ds41-wmk2, FINDINGS F2): the dense ops against an f64 HOST REFERENCE, so the part-order form is held to the
 * true value, not only to itself (rows = alone) or to the decode band. Weights: GGUF blocks with realistic scales (q6_K d
 * 2^-11 .. 2^-8, int8 sub-scales; q5_K d / dmin 2^-11 .. 2^-8, 6-bit scales / mins; q8_0 d 2^-9 .. 2^-6; random quants);
 * activations bf16 ~ N(0, 1). The reference dequantises with ggml's formulas on the host (f64) and sums x . w in f64; the
 * folds' reference reads the h the device wrote (the GEMM half alone). For each (type, op, M): per output the error
 * |y - ref| / sum_k |x_k w_k| (the sum's own scale: what any summation order may lose) -- mean and max -- for
 *   D  int16's decode band (gg_gemv 2 + gg_gemv_m16 1, ds41_wmk 0),  W  ds41_wmk 1, and (info)  S  the stock WMMA form at the same M
 *   (int16's prefill rows' arithmetic: the WMMA's own products and 16-term sums, the stock K order),
 * both halves of a pair, the head's f32 logits. A wrong operand / scale / half shows as an error near 1, rounding as ~1e-3
 * (bf16 output) or ~1e-6 (f32). WMKREF lines; WMGVERDICT wmkref: FAIL if W's max error exceeds 4x D's max + 1e-3. */
static double r_h2d(uint16_t h) { return (double)rad_f16_to_f32(h); }
static void r_deq(const MrTy& ty, const uint8_t* row, int64_t K, std::vector<double>& w) {
    w.assign((size_t)K, 0.0);
    const int64_t nb = K / ty.qk;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* q = row + b * ty.bb;
        double* y = w.data() + b * ty.qk;
        if (ty.qk == 32) {   /* q8_0 */
            uint16_t d; std::memcpy(&d, q, 2);
            for (int i = 0; i < 32; ++i) y[i] = r_h2d(d) * (double)(int8_t)q[2 + i];
        } else if (!std::strcmp(ty.name, "q6_K")) {
            const uint8_t* ql = q; const uint8_t* qh = q + 128; const int8_t* sc = (const int8_t*)(q + 192);
            uint16_t dh; std::memcpy(&dh, q + 208, 2);
            const double d = r_h2d(dh);
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[n + l] = d * sc[is] * q1;
                    y[n + l + 32] = d * sc[is + 2] * q2;
                    y[n + l + 64] = d * sc[is + 4] * q3;
                    y[n + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64; qh += 32; sc += 8;
            }
        } else {   /* q5_K */
            uint16_t dh, mh; std::memcpy(&dh, q, 2); std::memcpy(&mh, q + 2, 2);
            const double d = r_h2d(dh), dmin = r_h2d(mh);
            const uint8_t* scl = q + 4; const uint8_t* qh = q + 16; const uint8_t* ql = q + 48;
            auto sm = [&](int j, int& sc, int& m) {
                if (j < 4) { sc = scl[j] & 63; m = scl[j + 4] & 63; }
                else { sc = (scl[j + 4] & 0xF) | ((scl[j - 4] >> 6) << 4); m = (scl[j + 4] >> 4) | ((scl[j] >> 6) << 4); }
            };
            int is = 0; uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                int sc, m;
                sm(is, sc, m); const double d1 = d * sc, m1 = dmin * m;
                sm(is + 1, sc, m); const double d2 = d * sc, m2 = dmin * m;
                for (int l = 0; l < 32; ++l) y[j + l] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
    }
}
static void r_weights(std::vector<uint8_t>& hw, const MrTy& ty, int64_t N, int64_t K, uint64_t seed) {
    uint64_t rs = seed;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    const int64_t rb = K / ty.qk * ty.bb;
    hw.resize((size_t)(N * rb));
    for (size_t i = 0; i < hw.size(); i += 8) { const uint64_t v = rnd(); std::memcpy(&hw[i], &v, std::min<size_t>(8, hw.size() - i)); }
    auto f16p = [&](int e_lo, int e_hi) {   /* a positive normal f16 in [2^e_lo, 2^(e_hi+1)) */
        const int e = e_lo + (int)(rnd() % (uint64_t)(e_hi - e_lo + 1));
        return (uint16_t)(((e + 15) << 10) | (uint16_t)(rnd() & 0x3ff));
    };
    for (int64_t n = 0; n < N; ++n)
        for (int64_t b = 0; b < K / ty.qk; ++b) {
            uint8_t* blk = hw.data() + n * rb + b * ty.bb;
            if (!std::strcmp(ty.name, "q6_K")) { const uint16_t d = f16p(-11, -8); std::memcpy(blk + 208, &d, 2); }
            else if (!std::strcmp(ty.name, "q5_K")) { const uint16_t d = f16p(-11, -8), dm = f16p(-11, -8); std::memcpy(blk, &d, 2); std::memcpy(blk + 2, &dm, 2); }
            else { const uint16_t d = f16p(-9, -6); std::memcpy(blk, &d, 2); }
        }
}
static void refcheck(const Plugin& pl, const char* tlist) {
    Hk h;
    if (!hooks(pl, h, "--wmkref")) return;
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    struct Op { const char* name; int kind; int64_t N, N2, K; const char* types; bool a2; };   /* kind 0 gemm, 1 logits, 2 pair, 3 fold a, 4 fold e */
    const Op ops[] = {
        { "attn_q_b(fold a)", 3, 8192, 0, 1280, "q5_K,q6_K", false },  { "attn_q_b", 0, 8192, 0, 1280, "q5_K,q6_K", false },
        { "q_a+attn_kv(pair)", 2, 1280, 512, 5120, "q5_K,q6_K", false }, { "attn_output_a g0+g1(pair a2)", 2, 1024, 1024, 4096, "q5_K,q6_K", true },
        { "attn_output_b", 0, 5120, 0, 2048, "q5_K,q6_K", false },     { "shexp_gate_up", 0, 1536, 0, 5120, "q6_K,q8_0", false },
        { "shexp_down(fold e)", 4, 5120, 0, 768, "q6_K,q8_0", false }, { "shexp_down_w512(fold e)", 4, 5120, 0, 512, "q6_K,q8_0", false },
        { "engram_kv_e2", 0, 6400, 0, 6144, "q6_K", false },           { "head", 1, 32320, 0, 5120, "q6_K", false },
    };
    const int64_t Ms[] = { 1, 4, 16, 33, 48 };
    long long keys = 0, bad = 0;
    double worst_ratio = 0.0;
    std::printf("WMK f64 reference check: per output |y - ref| / sum|x w|, D = int16's decode band, W = ds41_wmk 1\n");
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        const bool q32 = ty.qk == 32;
        for (const Op& op : ops) {
            if (!mr_in(op.types, ty.name) || op.K % ty.qk) continue;
            const char* rn = op.kind == 0 ? (q32 ? "gemm_nt_ggml_qk32_gemv" : "gemm_nt_ggml_qk256_gemv")
                           : op.kind == 1 ? (q32 ? "logits_gemm_ggml_qk32_gemv" : "logits_gemm_ggml_qk256_gemv")
                           : op.kind == 2 ? (q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256")
                           : op.kind == 3 ? (q32 ? "rmsnorm_gemm_nt_ggml_qk32" : "rmsnorm_gemm_nt_ggml_qk256")
                                          : (q32 ? "swiglu_clamped_gemm_nt_ggml_qk32" : "swiglu_clamped_gemm_nt_ggml_qk256");
            const RadKernelInfo* row = find(pl, rn);
            if (!row) { std::printf("  WMKREF %s %s: row absent\n", ty.name, op.name); ++g_fail; continue; }
            dfree_all();
            const int64_t K = op.K, N = op.N, N2 = op.N2, NY = N + N2, rb = K / ty.qk * ty.bb, xc = op.kind == 4 ? 2 * K : K;
            const int64_t MX = 48;
            std::vector<uint8_t> hw, hw2;
            r_weights(hw, ty, N, K, 0x3EF0001ull ^ (uint64_t)(N * 7 + K));
            if (N2) r_weights(hw2, ty, N2, K, 0x3EF0002ull ^ (uint64_t)(N2 * 7 + K));
            std::mt19937 rng(0x5EED ^ (unsigned)(K + N));
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<uint16_t> hx((size_t)(MX * xc)), hx2(op.a2 ? (size_t)(MX * K) : 0);
            for (auto& v : hx) v = rad_f32_to_bf16(nd(rng) * (op.kind == 4 ? 3.0f : 1.0f));
            for (auto& v : hx2) v = rad_f32_to_bf16(nd(rng));
            std::vector<float> g((size_t)K);
            for (auto& v : g) v = 0.5f + 0.5f * std::fabs(nd(rng));
            /* the reference's dequantised rows (f64) */
            std::vector<std::vector<double>> wd((size_t)NY);
            for (int64_t n = 0; n < NY; ++n) r_deq(ty, n < N ? &hw[(size_t)(n * rb)] : &hw2[(size_t)((n - N) * rb)], K, wd[(size_t)n]);
            const size_t ob = op.kind == 1 ? 4 : 2;
            void* d_w = dalloc(hw.size() + hw2.size());
            void* d_x = dalloc(hx.size() * 2);
            void* d_x2 = op.a2 ? dalloc(hx2.size() * 2) : nullptr;
            void* d_g = dalloc((size_t)K * 4);
            void* d_y = dalloc((size_t)MX * NY * ob);
            void* d_h = op.kind >= 3 ? dalloc((size_t)MX * K * 2) : nullptr;
            if (!d_w || !d_x || !d_g || !d_y || (op.a2 && !d_x2) || (op.kind >= 3 && !d_h)) { std::printf("  WMKREF %s %s: alloc\n", ty.name, op.name); ++g_fail; continue; }
            st_memcpy(d_w, hw.data(), hw.size(), hipMemcpyHostToDevice);
            if (N2) st_memcpy((uint8_t*)d_w + hw.size(), hw2.data(), hw2.size(), hipMemcpyHostToDevice);
            st_memcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
            if (op.a2) st_memcpy(d_x2, hx2.data(), hx2.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(d_g, g.data(), (size_t)K * 4, hipMemcpyHostToDevice);
            auto call = [&](int64_t M) -> int {
                st_memset(d_y, 0xff, (size_t)M * NY * ob);
                if (d_h) st_memset(d_h, 0x5a, (size_t)M * K * 2);
                int rc;
                if (op.kind == 0)
                    rc = k1_launch(row, { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16") });
                else if (op.kind == 1)
                    rc = k1_launch(row, { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_y, RAD_F32, { M, N }) },
                                   { PI("M", M), PI("n_vocab", N), PI("n_embd", K), PS("type", ty.name), PS("dtype", "bf16") });
                else if (op.kind == 2) {
                    std::vector<RadTensor> t = { gqst::TP(d_x, RAD_BF16, M, K, K), T(d_w, RAD_U8, { N, rb }),
                                                 gqst::TP(op.a2 ? d_x2 : d_x, RAD_BF16, M, K, K), T((uint8_t*)d_w + hw.size(), RAD_U8, { N2, rb }),
                                                 gqst::TP(d_y, RAD_BF16, M, N, NY), gqst::TP((uint16_t*)d_y + N, RAD_BF16, M, N2, NY) };
                    if (!op.a2) absent(t[2]);
                    rc = k1_launch(row, t, { PI("M", M), PI("N", N), PI("N2", N2), PI("K", K), PS("type", ty.name), PS("dtype", "bf16") });
                } else if (op.kind == 3)
                    rc = k1_launch(row, { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_U8, { N, rb }), T(d_g, RAD_F32, { K }),
                                          T(d_h, RAD_BF16, { M, K }), T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16"), PF("eps", 1e-6) });
                else
                    rc = k1_launch(row, { T(d_x, RAD_BF16, { M, 2 * K }), T(d_w, RAD_U8, { N, rb }), T(d_h, RAD_BF16, { M, K }),
                                          T(d_y, RAD_BF16, { M, N }) },
                                   { PI("M", M), PI("N", N), PI("K", K), PS("type", ty.name), PS("dtype", "bf16"), PF("limit", 10.0) });
                return rc != RAD_OK ? rc : (st_stream_sync(nullptr) != hipSuccess ? -999 : RAD_OK);
            };
            /* sampled outputs: every row, columns n % NSTEP == 0 (about 1024 a call) and every column of the pair's second half */
            const int64_t NSTEP = std::max<int64_t>(1, N / 1024);
            std::vector<int64_t> cols;
            for (int64_t n = 0; n < NY; ++n) if (n >= N || n % NSTEP == 0) cols.push_back(n);
            /* x . w in f64 and sum |x w| for row content xi */
            auto refof = [&](const uint16_t* xi, int64_t n, double* mag) {
                const std::vector<double>& w = wd[(size_t)n];
                double ref = 0, mg = 0;
                for (int64_t k = 0; k < K; ++k) { const double a = (double)rad_bf16_to_f32(xi[k]); ref += a * w[(size_t)k]; mg += std::fabs(a * w[(size_t)k]); }
                *mag = mg;
                return ref;
            };
            /* the plain ops' references, once for the 48 rows (a call's row i holds row i's content at every M) */
            std::vector<double> R0, G0;
            if (op.kind <= 2) {
                R0.resize((size_t)(MX * cols.size())); G0.resize(R0.size());
                for (int64_t i = 0; i < MX; ++i)
                    for (size_t c = 0; c < cols.size(); ++c) {
                        const int64_t n = cols[c];
                        const uint16_t* xi = (op.a2 && n >= N) ? &hx2[(size_t)(i * K)] : &hx[(size_t)(i * xc)];
                        R0[(size_t)i * cols.size() + c] = refof(xi, n, &G0[(size_t)i * cols.size() + c]);
                    }
            }
            /* the error of one mode at M rows */
            auto measure = [&](int64_t M, double* mean, double* mx, double* rmax) -> bool {
                if (call(M) != RAD_OK) return false;
                std::vector<uint8_t> yb((size_t)M * NY * ob);
                st_memcpy(yb.data(), d_y, yb.size(), hipMemcpyDeviceToHost);
                std::vector<uint16_t> hb;
                if (d_h) { hb.resize((size_t)(M * K)); st_memcpy(hb.data(), d_h, hb.size() * 2, hipMemcpyDeviceToHost); }
                double s = 0, m = 0, r = 0; long long cnt = 0;
                for (int64_t i = 0; i < M; ++i)
                    for (size_t c = 0; c < cols.size(); ++c) {
                        const int64_t n = cols[c];
                        double ref, mag;
                        if (op.kind <= 2) { ref = R0[(size_t)i * cols.size() + c]; mag = G0[(size_t)i * cols.size() + c]; }
                        else ref = refof(&hb[(size_t)(i * K)], n, &mag);
                        double y;
                        if (ob == 4) { float f; std::memcpy(&f, &yb[(size_t)((i * NY + n) * 4)], 4); y = f; }
                        else { uint16_t u; std::memcpy(&u, &yb[(size_t)((i * NY + n) * 2)], 2); y = (double)rad_bf16_to_f32(u); }
                        const double e = mag > 0 ? std::fabs(y - ref) / mag : 0.0;
                        s += e; m = std::max(m, e); ++cnt;
                        if (std::fabs(ref) > 0) r = std::max(r, std::min(1.0, std::fabs(y - ref) / std::fabs(ref)));
                    }
                *mean = cnt ? s / (double)cnt : 0; *mx = m; *rmax = r;
                return true;
            };
            std::string line = fmt("WMKREF %s %s %lldx%lld%s |", ty.name, op.name, (long long)NY, (long long)K, ob == 4 ? " f32" : "");
            bool kbad = false;
            for (int64_t M : Ms) {
                double dm = 0, dx = 0, dr = 0, wm = 0, wx = 0, wr = 0, sm_ = 0, sx = 0, sr = 0;
                set_dec(h); h.sw("ds41_wmk", 0);
                const bool okd = measure(M, &dm, &dx, &dr);
                set_dec(h); h.sw("ds41_wmk", 1);
                const bool okw = measure(M, &wm, &wx, &wr);
                h.sw("ds41_wmk", 0);
                set_stock_wm(h);   /* S: the stock WMMA form (int16's prefill rows' arithmetic) at the same M; the folds' GEMM half */
                const bool oks = op.kind <= 2 || true ? measure(M, &sm_, &sx, &sr) : false;
                clear(h);
                const bool b = !okd || !okw || wx > 4.0 * dx + 1e-3;
                kbad |= b;
                if (okd && okw && dx > 0) worst_ratio = std::max(worst_ratio, wx / dx);
                line += fmt(" M%lld D mean %.2e max %.2e | W mean %.2e max %.2e (W/D max x%.2f) | S mean %.2e max %.2e%s;", (long long)M, dm, dx, wm, wx,
                            dx > 0 ? wx / dx : 0.0, oks ? sm_ : -1.0, oks ? sx : -1.0, b ? " BAD" : "");
            }
            clear(h);
            ++keys; bad += kbad; g_fail += kbad;
            std::printf("%s %s\n", line.c_str(), kbad ? "FAIL" : "ok");
            std::fflush(stdout);
        }
    }
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    std::printf("WMGVERDICT wmkref: %lld (type, op) keys, %lld bad, worst W/D max-error ratio %.2f -> %s\n", keys, bad, worst_ratio,
                keys == 0 ? "NO DATA" : bad ? "FAIL (W further from the f64 reference than int16's band)" : "PASS (W as close to f64 as int16's band)");
}

/* --perfwkpf [Ms] [iters] [types] (ds41-wmk2, stage 3 G3): the prefill rows' time, int16 vs lever WMK. Per (type, shape, M) at the
 * 2.25 .rad's dense shapes, rotated weights past 256 MiB, mean of `iters`: the stock WMMA kernel, each GPF form F (int16's arithmetic,
 * gg_pf pinned) and the same form under ds41_wmk 1 (the part order, r4d_gg_wkpf_run) with the running total in registers (r) and in
 * scratch (s). Bytes: every W arm = W on form 322 (one arithmetic whatever the form). GPWMK lines, WMGVERDICT wkpf. */
static void wkpf_perf(const Plugin& pl, const char* mlist, int iters, const char* tlist) {
    if (iters <= 0) iters = 20;
    Hk h;
    if (!hooks(pl, h, "--perfwkpf")) return;
    const GpHooks gp = gp_hooks(pl);
    auto wkpf_ts = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_ts");
    auto wkpf_last = (int (*)())dlsym(pl.h, "r4d_gg_wkpf_last");
    if (!gp.pin || !wkpf_ts || !wkpf_last) { std::printf("--perfwkpf: r4d_gg_pf_pin / r4d_gg_wkpf_ts absent\n"); ++g_fail; return; }
    auto wkpf_follow = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_follow");
    if (wkpf_follow) wkpf_follow(1);   /* W takes the pinned form */
    const std::vector<int64_t> Ms = gp_list(mlist, "4096,2048");
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const int forms[] = { 0, 322, 422, 426, 326, 321, 222, 414, 418, 214 };   /* 0: int16's stock kernel / W's stock kernel + fold; 414 / 418 / 214:
                                                                                  one column tile a wave, 16 waves (W only: int16's GPF has no such form) */
    auto wkpf_default = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_default");
    long long keys = 0, bad = 0;
    double lsum = 0; int nl = 0;
    std::printf("WMK prefill timing (rotated weights, mean of %d): stock / GPF form F (int16) / form under ds41_wmk 1 with the total in registers (r) and scratch (s)\n", iters);
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const MrShape& sh : kMrShapes) {
            if (sh.logits || !mr_in(sh.types, ty.name) || sh.K % ty.qk) continue;
            const int64_t rb = sh.K / ty.qk * ty.bb, wbytes = sh.N * rb;
            std::vector<uint8_t> hw;
            mr_weights(hw, ty, sh.N, sh.K, 0x51ED2701ull ^ (uint64_t)(sh.N * 7 + sh.K));
            const int copies = (int)std::max<int64_t>(2, (256ll << 20) / wbytes + 1);
            for (int64_t M : Ms) {
                if (M <= 48) continue;
                dfree_all();
                std::vector<uint16_t> hx;
                gp_acts(hx, M, sh.K, 0xABCDEFull ^ (uint64_t)M);
                const size_t mb = (size_t)(M * sh.N) * 2;
                void* d_x = dalloc((size_t)(M * sh.K) * 2);
                void* d_w = dalloc((size_t)wbytes * (size_t)copies);
                void* d_y = dalloc(mb);
                if (!d_x || !d_w || !d_y) { std::printf("  %s %s M=%lld: alloc failed\n", ty.name, sh.name, (long long)M); ++g_fail; continue; }
                st_memcpy(d_x, hx.data(), (size_t)(M * sh.K) * 2, hipMemcpyHostToDevice);
                for (int c = 0; c < copies; ++c) st_memcpy((uint8_t*)d_w + (size_t)c * (size_t)wbytes, hw.data(), (size_t)wbytes, hipMemcpyHostToDevice);
                MrLaunch L;
                if (!mr_setup(L, pl, sh, ty, M)) { std::printf("  %s %s: row absent\n", ty.name, sh.name); ++g_fail; continue; }
                std::vector<uint8_t> wref;
                /* arm: wmk 0 / 1, pf form (0 = stock), ts */
                auto arm = [&](int wmk, int form, int ts, double* us, bool* same, int* used) -> bool {
                    clear(h);
                    h.sw("ds41_wmk", wmk);
                    if (wmk && wkpf_default) { wkpf_default(form); if (wkpf_follow) wkpf_follow(0); }   /* W: exactly this form */
                    else { gp.pin(form); if (wkpf_follow) wkpf_follow(1); }
                    wkpf_ts(ts);
                    st_memset(d_y, 0xff, mb);
                    const int rc = L.run(d_x, d_w, d_y);
                    bool ok = rc == RAD_OK && st_stream_sync(nullptr) == hipSuccess;
                    *used = wmk ? wkpf_last() : form;
                    if (ok && wmk) {
                        std::vector<uint8_t> got(mb);
                        st_memcpy(got.data(), d_y, mb, hipMemcpyDeviceToHost);
                        if (wref.empty()) wref = got;
                        *same = got == wref;
                    } else {
                        *same = true;
                    }
                    if (ok) *us = st_time_us([&](int i) { L.run(d_x, (uint8_t*)d_w + (size_t)(i % copies) * (size_t)wbytes, d_y); }, iters, nullptr);
                    gp.pin(0); wkpf_ts(0); h.sw("ds41_wmk", 0);
                    if (wkpf_default) wkpf_default(-1);
                    return ok;
                };
                double st = 0; bool sm; int u;
                arm(0, 0, 0, &st, &sm, &u);
                const double fl = 2.0 * (double)M * (double)sh.N * (double)sh.K;
                std::string line = fmt("GPWMK %s %s %lldx%lld M=%lld: stock %.1f us (%.1f TF) |", ty.name, sh.name, (long long)sh.N, (long long)sh.K, (long long)M,
                                       st, st > 0 ? fl / (st * 1e6) : 0.0);
                /* s18: the same stock kernel held to fewer blocks a WGP by LDS alone (r4d_gg_wm_ldsx): the occupancy price */
                if (auto ldsx = (void (*)(int))dlsym(pl.h, "r4d_gg_wm_ldsx")) {
                    for (int lb : { 33000, 43776 }) {   /* LDS caps of 3 and 2 blocks a WGP (128 KiB) */
                        double sl = 0;
                        ldsx(lb);
                        arm(0, 0, 0, &sl, &sm, &u);
                        ldsx(0);
                        line += fmt(" stockL%d %.1f;", lb / 1024, sl);
                    }
                }
                double bestF = st, bestW = 1e30;
                int bF = 0, bW = 0;
                bool kb = false;
                for (int fc : forms) {
                    double uf = -1, ur = -1, us = -1; bool s1 = true, s2 = true; int u1 = 0, u2 = 0;
                    const bool w_only = fc == 414 || fc == 418 || fc == 214;
                    const bool okf = w_only ? true : arm(0, fc, 0, &uf, &sm, &u);
                    const bool okr = arm(1, fc, 1, &ur, &s1, &u1);
                    const bool oks = arm(1, fc, 2, &us, &s2, &u2);
                    if (!okf || !okr || !oks || !s1 || !s2 || u1 != fc || u2 != fc) kb = true;
                    if (fc == 0) us = -1;   /* the stock kernel has no scratch-total form */
                    if (okf && uf > 0 && uf < bestF) { bestF = uf; bF = fc; }
                    if (okr && ur > 0 && ur < bestW) { bestW = ur; bW = fc * 10 + 1; }
                    if (oks && us > 0 && us < bestW) { bestW = us; bW = fc * 10 + 2; }
                    line += fmt(" %d F %.1f W r %.1f s %.1f%s;", fc, uf, ur, us, (!s1 || !s2) ? " W BYTES DIFFER" : "");
                }
                ++keys; bad += kb; g_fail += kb;
                if (bestF > 0 && bestW < 1e29) { lsum += std::log(bestW / bestF); ++nl; }
                std::printf("%s | best int16 %d %.1f us, best W %d%s %.1f us, W/int16 x%.3f%s\n", line.c_str(), bF, bestF, bW / 10, bW % 10 == 1 ? "r" : "s", bestW,
                            bestF > 0 ? bestW / bestF : 0.0, kb ? " BAD" : "");
                std::fflush(stdout);
            }
        }
    }
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    if (wkpf_follow) wkpf_follow(0);
    std::printf("WMGVERDICT wkpf: %lld (type, shape, M) keys, %lld bad (a launch failed, a W form unused, or W bytes differ across forms); best W vs best int16 geomean x%.3f\n",
                keys, bad, nl ? std::exp(lsum / nl) : 0.0);
}

/* --perfwq [Ms] [iters] [types] [forms] (ds41-wmk2 stage 3, 14.11.2026): W's prefill forms against int16's stock WMMA kernel in ONE
 * session, per (type, shape, M) in the order stock, W form 0, the listed W forms (gg_wq_kernel 9xxx codes or GPF codes), stock again
 * (drift), rotated weights past 256 MiB, mean of `iters`. int16 also: GPF 426 / 422 (int16's P0 table forms for the q6_K / q8_0
 * shapes). Bytes: every W form == W form 0 (the part order). GPWQ lines (us, and x of the stock kernel's mean), WMGVERDICT wq with
 * the rank-0 weighted sums (q5_K attention x13 + q6_K attention x27, q6_K shared x27 + q8_0 shared x13, engram x2 a step). */
static void wq_perf(const Plugin& pl, const char* mlist, int iters, const char* tlist, const char* flist) {
    if (iters <= 0) iters = 20;
    Hk h;
    if (!hooks(pl, h, "--perfwq")) return;
    const GpHooks gp = gp_hooks(pl);
    auto wkpf_last = (int (*)())dlsym(pl.h, "r4d_gg_wkpf_last");
    auto wkpf_follow = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_follow");
    auto wkpf_default = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_default");
    auto wkpf_ts = (void (*)(int))dlsym(pl.h, "r4d_gg_wkpf_ts");
    if (!gp.pin || !wkpf_last || !wkpf_default || !wkpf_ts) { std::printf("--perfwq: r4d_gg_pf_pin / r4d_gg_wkpf_default absent\n"); ++g_fail; return; }
    const std::vector<int64_t> Ms = gp_list(mlist, "2048,4096");
    const std::string want = tlist && *tlist ? tlist : "q5_K,q6_K,q8_0";
    const std::vector<int64_t> forms = gp_list(flist, "9312,9311,9411,9412");
    long long keys = 0, bad = 0;
    std::map<int64_t, std::map<int64_t, double>> wsum;   /* M -> form (-1 stock, -2 int16 best, 0 W0, f) -> rank-0 weighted us */
    std::printf("WMK gg_wq prefill timing (rotated weights, %d timed after a >= 120 ms warm burst, the median of 3 alternating rounds): stock (int16) | W form 0 (stock + fold) | W forms\n", iters);
    for (const MrTy& ty : kMrTypes) {
        if (!mr_in(want.c_str(), ty.name)) continue;
        for (const MrShape& sh : kMrShapes) {
            if (sh.logits || !mr_in(sh.types, ty.name) || sh.K % ty.qk) continue;
            /* rank-0 weight of this (type, shape) a step: attention q5_K 13 / q6_K 27 layers, shared q6_K 27 / q8_0 13, engram 2 */
            const std::string nm = sh.name;
            const bool shexp = nm.rfind("shexp", 0) == 0, w512 = nm.find("w512") != std::string::npos, eng = nm.rfind("engram", 0) == 0;
            double wt = 0;
            if (eng) wt = 2;
            else if (shexp) wt = w512 ? 0 : (std::string(ty.name) == "q6_K" ? 27 : std::string(ty.name) == "q8_0" ? 13 : 0);
            else wt = std::string(ty.name) == "q5_K" ? 13 : std::string(ty.name) == "q6_K" ? 27 : 0;
            if (nm.find("attn_output_a") != std::string::npos) wt *= 2;   /* two groups a layer */
            const int64_t rb = sh.K / ty.qk * ty.bb, wbytes = sh.N * rb;
            std::vector<uint8_t> hw;
            mr_weights(hw, ty, sh.N, sh.K, 0x51ED2701ull ^ (uint64_t)(sh.N * 7 + sh.K));
            const int copies = (int)std::max<int64_t>(2, (256ll << 20) / wbytes + 1);
            for (int64_t M : Ms) {
                if (M <= 48) continue;
                dfree_all();
                std::vector<uint16_t> hx;
                gp_acts(hx, M, sh.K, 0xABCDEFull ^ (uint64_t)M);
                const size_t mb = (size_t)(M * sh.N) * 2;
                void* d_x = dalloc((size_t)(M * sh.K) * 2);
                void* d_w = dalloc((size_t)wbytes * (size_t)copies);
                void* d_y = dalloc(mb);
                if (!d_x || !d_w || !d_y) { std::printf("  %s %s M=%lld: alloc failed\n", ty.name, sh.name, (long long)M); ++g_fail; continue; }
                st_memcpy(d_x, hx.data(), (size_t)(M * sh.K) * 2, hipMemcpyHostToDevice);
                for (int c = 0; c < copies; ++c) st_memcpy((uint8_t*)d_w + (size_t)c * (size_t)wbytes, hw.data(), (size_t)wbytes, hipMemcpyHostToDevice);
                MrLaunch L;
                if (!mr_setup(L, pl, sh, ty, M)) { std::printf("  %s %s: row absent\n", ty.name, sh.name); ++g_fail; continue; }
                std::vector<uint8_t> wref;
                /* the arms: wmk 0 = int16 (form = gg_pf pin, 0 = the stock kernel); wmk 1 = W on exactly `form` */
                struct Arm { int wmk, form; std::vector<double> t; bool ok = true; };
                std::vector<Arm> arms;
                const bool q6 = std::string(ty.name) == "q6_K", q8 = std::string(ty.name) == "q8_0";
                arms.push_back({ 0, 0, {} });
                if (q6) arms.push_back({ 0, 426, {} });
                if (q8) arms.push_back({ 0, 422, {} });
                arms.push_back({ 1, 0, {} });
                for (int64_t f : forms) arms.push_back({ 1, (int)f, {} });
                auto set_arm = [&](const Arm& a) {
                    clear(h);
                    h.sw("ds41_wmk", a.wmk);
                    if (a.wmk) { wkpf_default(a.form); if (wkpf_follow) wkpf_follow(0); }
                    else gp.pin(a.form);
                    wkpf_ts(1);
                };
                auto unset_arm = [&]() { gp.pin(0); wkpf_ts(0); h.sw("ds41_wmk", 0); wkpf_default(-1); };
                bool kb = false;
                /* bytes and the form used, once an arm */
                for (Arm& a : arms) {
                    set_arm(a);
                    st_memset(d_y, 0xff, mb);
                    const int rc = L.run(d_x, d_w, d_y);
                    a.ok = rc == RAD_OK && st_stream_sync(nullptr) == hipSuccess;
                    const int used = a.wmk ? wkpf_last() : a.form;
                    if (a.ok && a.wmk) {
                        std::vector<uint8_t> got(mb);
                        st_memcpy(got.data(), d_y, mb, hipMemcpyDeviceToHost);
                        if (wref.empty()) wref = got;
                        if (got != wref) { a.ok = false; std::printf("  %s %s M=%lld: W form %d BYTES DIFFER from W form 0\n", ty.name, sh.name, (long long)M, a.form); }
                    }
                    if (a.wmk && used != a.form) { a.ok = false; std::printf("  %s %s M=%lld: W form %d NOT USED (%d)\n", ty.name, sh.name, (long long)M, a.form, used); }
                    kb |= !a.ok;
                    unset_arm();
                }
                /* the timing: the card held at its power cap (a warm burst of >= kWarmMs of the arm untimed, then `iters` timed), the
                 * arms in alternating order over kRounds rounds (forward, back, forward), the median a arm: a later arm no longer pays the
                 * clock drop of the ones before it (the first run of this mode read the same stock kernel 20-28 % apart within a key) */
                constexpr int kRounds = 3;
                constexpr double kWarmMs = 120.0;
                auto run_i = [&](int i) { L.run(d_x, (uint8_t*)d_w + (size_t)(i % copies) * (size_t)wbytes, d_y); };
                auto timed = [&]() -> double {
                    hipEvent_t e0, e1;
                    st_ev_create(&e0); st_ev_create(&e1);
                    st_ev_record(e0, nullptr); run_i(0); run_i(1); st_ev_record(e1, nullptr); st_ev_sync(e1);
                    float ms = 0.0f;
                    st_ev_elapsed(&ms, e0, e1);
                    const int nw = (int)std::min(20000.0, std::ceil(kWarmMs / std::max(0.002, (double)ms / 2.0)));
                    for (int i = 0; i < nw; ++i) run_i(i);
                    st_ev_record(e0, nullptr);
                    for (int i = 0; i < iters; ++i) run_i(i);
                    st_ev_record(e1, nullptr);
                    st_ev_sync(e1);
                    st_ev_elapsed(&ms, e0, e1);
                    st_ev_destroy(e0); st_ev_destroy(e1);
                    return (double)ms * 1000.0 / iters;
                };
                for (int r = 0; r < kRounds; ++r)
                    for (size_t k = 0; k < arms.size(); ++k) {
                        Arm& a = arms[r % 2 ? arms.size() - 1 - k : k];
                        if (!a.ok) continue;
                        set_arm(a);
                        a.t.push_back(timed());
                        unset_arm();
                    }
                auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                double st = med(arms[0].t), g426 = -1, g422 = -1, w0 = -1;
                const double s1 = arms[0].t.empty() ? -1 : *std::min_element(arms[0].t.begin(), arms[0].t.end());
                const double s2 = arms[0].t.empty() ? -1 : *std::max_element(arms[0].t.begin(), arms[0].t.end());
                std::vector<double> wf(forms.size(), -1);
                std::string fl;
                size_t ai = 1;
                if (q6) g426 = med(arms[ai++].t);
                if (q8) g422 = med(arms[ai++].t);
                w0 = med(arms[ai++].t);
                for (size_t i = 0; i < forms.size(); ++i, ++ai) {
                    wf[i] = med(arms[ai].t);
                    fl += fmt(" %lld %.1f;", (long long)forms[i], wf[i]);
                }
                double ib = st;   /* int16's best of stock / its P0 table's GPF form */
                if (g426 > 0) ib = std::min(ib, g426);
                if (g422 > 0) ib = std::min(ib, g422);
                double bw = w0; int64_t bf = 0;
                for (size_t i = 0; i < forms.size(); ++i) if (wf[i] > 0 && wf[i] < bw) { bw = wf[i]; bf = forms[i]; }
                wsum[M][-1] += wt * st; wsum[M][-2] += wt * ib; wsum[M][0] += wt * w0; wsum[M][-3] += wt * bw;
                for (size_t i = 0; i < forms.size(); ++i) wsum[M][forms[i]] += wt * wf[i];
                ++keys; bad += kb; g_fail += kb;
                std::printf("GPWQ %s %s %lldx%lld M=%lld w%g: stock %.1f us (min %.1f max %.1f)%s%s | W0 %.1f |%s | best W %lld %.1f = x%.3f of stock, x%.3f of int16 best%s\n",
                            ty.name, sh.name, (long long)sh.N, (long long)sh.K, (long long)M, wt, st, s1, s2,
                            g426 > 0 ? fmt(" | 426F %.1f", g426).c_str() : "", g422 > 0 ? fmt(" | 422F %.1f", g422).c_str() : "", w0, fl.c_str(),
                            (long long)bf, bw, st > 0 ? bw / st : 0.0, ib > 0 ? bw / ib : 0.0, kb ? " BAD" : "");
                std::fflush(stdout);
            }
        }
    }
    clear(h);
    dfree_all();
    if (keys == 0) ++g_fail;
    for (auto& mk : wsum) {
        auto& w = mk.second;
        std::string fl;
        for (int64_t f : forms) fl += fmt(" %lld %.2f ms (x%.3f);", (long long)f, w[f] / 1000.0, w[-1] > 0 ? w[f] / w[-1] : 0.0);
        std::printf("WQSUM M=%lld rank-0 dense prefill a step (weighted): stock %.2f ms | int16 best %.2f | W0 %.2f (x%.3f) |%s | best-per-shape W %.2f ms = x%.3f of stock, x%.3f of int16 best\n",
                    (long long)mk.first, w[-1] / 1000.0, w[-2] / 1000.0, w[0] / 1000.0, w[-1] > 0 ? w[0] / w[-1] : 0.0, fl.c_str(), w[-3] / 1000.0,
                    w[-1] > 0 ? w[-3] / w[-1] : 0.0, w[-2] > 0 ? w[-3] / w[-2] : 0.0);
    }
    std::printf("WMGVERDICT wq: %lld (type, shape, M) keys, %lld bad (a launch failed, a W form unused, or W bytes differ from W form 0)\n", keys, bad);
}

}  // namespace wmg
