/* r4d_selftest_i11.inc -- ds41-int11's byte checks on one card (included by r4d_selftest.cpp before device_phase; called
 * from it), each new form held to the stock form it replaces BYTE FOR BYTE, stock / new times as info lines:
 *
 *   --case mix_wv (= the row mhc_mix_bf16): lever 1U-MX, mhc_mix under libr4d's fn4 ds41_mix_wv 12 / 16 / 24 vs 0 at the
 *       decode heights M 1 / 2 / 3 / 6 / 8 (the split launch's R 1 form) and M 9 (above them: the switch must not act) --
 *       the partials in the scratch, pre, post and comb. The control: the same launch with the scratch's partials read
 *       before the finish must agree too (a wave that formed another dot's partial would show there first). */

namespace i11 {

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
static long long diff32(const void* a, const void* b, size_t bytes) {
    std::vector<uint32_t> x = down<uint32_t>(a, bytes / 4), y = down<uint32_t>(b, bytes / 4);
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
static int run(const RadKernelInfo* r, std::vector<RadTensor>& ts, std::vector<RadParam>& ps, void* sc = nullptr,
               int64_t sb = 0) {
    RadArgs a = args(ts, ps, sc, sb);
    const int rc = r->launch(&a, (RadStream) nullptr);
    st_stream_sync(nullptr);
    return rc;
}
static void info(const std::string& s) { std::printf("  info  %s\n", s.c_str()); }

/* ======================================================================= lever 1U-MX */
static void mx_case(const Plugin& pl) {
    const RadKernelInfo* rmix = find(pl, "mhc_mix_bf16");
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    auto stats = (int (*)(long long*, int))dlsym(pl.h, "rad_mx_stats");
    if (!rmix || !sw || !stats) {
        ok("mix_wv rows", false, "mhc_mix_bf16 / rad_fn4_switch / rad_mx_stats absent");
        return;
    }
    if (sw("ds41_mix_wv", 12) != 1 || sw("ds41_mix_wv", 0) != 1 || sw("ds41_mix_wv", 13) >= 0) {
        ok("ds41_mix_wv switch", false, "12 / 0 not taken or 13 not refused");
        return;
    }
    constexpr int64_t n = 5120, hc = 4, W = hc * n, MIX = hc * (hc + 2), S = MIX + 1, iters = 20;
    const int64_t chunks = (W + 1023) / 1024;
    g_rs = 0x5EED0011ull;
    std::vector<float> fn((size_t)(MIX * W)), base(MIX), scale(3);
    for (auto& x : fn) x = (float)(gauss() * 0.01);
    for (auto& x : base) x = (float)(gauss() * 0.5);
    for (auto& x : scale) x = (float)(gauss() * 0.1 + 0.5);
    void *dfn = up(fn), *dbase = up(base), *dscale = up(scale);
    for (int64_t M : { 1, 2, 3, 6, 8, 9 }) {
        std::vector<uint16_t> h((size_t)(M * W));
        for (int64_t r = 0; r < M; ++r) {
            const double rs = std::ldexp(1.0, (int)(rnd() % 5) - 2);
            for (int64_t i = 0; i < W; ++i) h[(size_t)(r * W + i)] = bf((float)(gauss() * rs));
        }
        void* dh = up(h);
        const size_t bp = (size_t)(M * hc * 4), bc = (size_t)(M * hc * hc * 4), bs = (size_t)(M * S * chunks * 4);
        struct Out { void *pre, *post, *comb, *scr; };
        auto mk = [&]() { return Out{ dalloc(bp), dalloc(bp), dalloc(bc), dalloc(bs + 4096) }; };
        Out R = mk();
        if (!dfn || !dh || !R.scr) { ok("mix_wv alloc", false, "hipMalloc failed"); dfree_all(); return; }
        std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("hc", hc), PI("iterations", iters), PF("eps", 1e-6),
                                     PF("hc_eps", 1e-6), PS("dtype", "bf16") };
        auto ts = [&](Out& o) {
            return std::vector<RadTensor>{ T(dh, RAD_BF16, { M, W }), T(dfn, RAD_F32, { MIX, W }), T(dscale, RAD_F32, { 3 }),
                                           T(dbase, RAD_F32, { MIX }), T(o.pre, RAD_F32, { M, hc }),
                                           T(o.post, RAD_F32, { M, hc }), T(o.comb, RAD_F32, { M, hc * hc }) };
        };
        auto go = [&](Out& o, int wv) {
            st_memset(o.pre, 0xA5, bp);
            st_memset(o.post, 0xA5, bp);
            st_memset(o.comb, 0xA5, bc);
            st_memset(o.scr, 0x5A, bs + 4096);
            sw("ds41_mix_wv", wv);
            std::vector<RadTensor> t = ts(o);
            return run(rmix, t, ps, o.scr, (int64_t)bs + 4096);
        };
        const int rc0 = go(R, 0);
        for (int wv : { 12, 16, 24 }) {
            Out F = mk();
            long long s0[2] = { 0, 0 }, s1[2] = { 0, 0 };
            stats(s0, 2);
            const int rc1 = go(F, wv);
            stats(s1, 2);
            if (rc0 != RAD_OK || rc1 != RAD_OK) {
                ok(fmt("mix_wv %d M=%lld launches", wv, (long long)M).c_str(), false, fmt("rc stock %d wide %d", rc0, rc1));
                continue;
            }
            const long long dp = diff32(R.scr, F.scr, bs), dpre = diff32(R.pre, F.pre, bp), dpo = diff32(R.post, F.post, bp),
                            dcb = diff32(R.comb, F.comb, bc);
            const bool took = s1[0] > s0[0];
            ok(fmt("mhc_mix ds41_mix_wv %d byte for byte vs 0, M=%lld (partials, pre, post, comb; wide form %s)", wv,
                   (long long)M, took ? "taken" : "not taken").c_str(),
               dp + dpre + dpo + dcb == 0 && took == (M <= 8),
               fmt("differing words: partials %lld pre %lld post %lld comb %lld; wide launches +%lld (want %s)", dp, dpre,
                   dpo, dcb, s1[0] - s0[0], M <= 8 ? "> 0" : "0"));
            if (M == 1 || M == 6) {
                std::vector<RadTensor> t0 = ts(R), t1 = ts(F);
                RadArgs a0 = args(t0, ps, R.scr, (int64_t)bs + 4096), a1 = args(t1, ps, F.scr, (int64_t)bs + 4096);
                sw("ds41_mix_wv", 0);
                const double u0 = st_time_us([&](int) { rmix->launch(&a0, nullptr); }, 200, nullptr);
                sw("ds41_mix_wv", wv);
                const double u1 = st_time_us([&](int) { rmix->launch(&a1, nullptr); }, 200, nullptr);
                info(fmt("mhc_mix M=%lld: WV 8 %.2f us, WV %d %.2f us (%+.1f %%)", (long long)M, u0, wv, u1,
                         100.0 * (u1 - u0) / u0));
            }
        }
        sw("ds41_mix_wv", 0);
    }
    dfree_all();
}

}  // namespace i11

static void i11_cases(const Plugin& pl) {
    if (sel("mix_wv") || sel("mhc_mix_bf16")) i11::mx_case(pl);
}
