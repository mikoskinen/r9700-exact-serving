/* r4d_selftest_p30o.inc -- tp8 loop P30o (19.11.2026): DeepSeek-V4.1's released block FP8 on one card (included by
 * r4d_selftest.cpp before device_phase, called from it):
 *
 *   --case fp8a8e8 (or the rows' names quant_act_fp8_ue8m0 / gemm_fp8a8e8_b32):
 *     1. quant_act_fp8_ue8m0 (the release's act_quant(x, 32, 'ue8m0')) against libref's row, BYTE FOR BYTE: codes and
 *        scales, at M 1 / 7 / 33 / 130 x K 1024 / 5120 / 6144, a pitched input (K + 64), a zero row, a huge and a tiny row;
 *     2. gemm_fp8a8e8_b32 (gemm_nt_q dtype fp8a8e8: E4M3 x E4M3, an E8M0 weight exponent per 32x32 block, a power-of-two
 *        activation scale per (row, 32), folded every 32 of K) against libref's gemm_nt_q at fp8a8e8 (quant_dot: per 32-group
 *        sub-sum ascending, acc += sub x sa x sb, one bf16 rounding): the count of bf16 outputs that differ and the worst
 *        relative error, at M 1 / 16 / 17 / 130 x (N, K) (64, 128) / (512, 1024) / (1280, 5120), plain and with a pitched
 *        activation (K + 128) and output (N + 64) (padding poisoned, checked untouched). The codes are multiples of 1/16 in
 *        [-4, 4] and the scales powers of two, which keeps every 32-term block sum exact in f32 in any order, so the card's
 *        WMMA order inside a block cannot round differently: PASS = 0 outputs differ;
 *     2b. the forms change no byte: 16-byte staging (r4d_fp8e8_pin_ld 16) at M tile 16 / 32 / 64 (r4d_fp8e8_pin_mt) into a
 *        re-poisoned output = the first form's (byte staging, tile 16), padding included;
 *     3. INFO: the GEMM at the newcomer's eight rank-0 shapes (run-p30a.sh's list and weights) x M 16 / 1100 / 4096 -- rows
 *        gemm_fp8a8e8_b32_first (byte staging, tile 16), gemm_fp8a8e8_b32 (16-byte staging, tile 16), _mt32 / _mt64 (16-byte, tile 32 / 64;
 *        above 16 rows) -- us a call over 10 timed launches after 2 warm ones, as `P30W name NxK M=m wW row us us` lines that
 *        tp8/devpf/scripts/p30read.py reads beside run-p30a.sh's (published B5SUM 101.16 / 356.15 ms at M 1100 / 4096).
 */

namespace p30ost {

/* the fp8a8e8 form's M tile (r4d_fp8_f32scale.hip: 16 = the first form, 32 / 64 = the wide tiles; the same bytes) */
static void pin_mt(const Plugin& pl, int mt) {
    auto f = (void (*)(int))dlsym(pl.h, "r4d_fp8e8_pin_mt");
    if (f) f(mt);
}
/* ... and its staging (4 = the first form's byte loads, 16 = 16-byte loads where aligned) */
static void pin_ld(const Plugin& pl, int ld) {
    auto f = (void (*)(int))dlsym(pl.h, "r4d_fp8e8_pin_ld");
    if (f) f(ld);
}

static const RadKernelInfo* oracle_of(void* ref, const char* op) {
    const auto count = (int (*)())dlsym(ref, "rad_kernel_count");
    const auto at = (const RadKernelInfo* (*)(int))dlsym(ref, "rad_kernel_at");
    for (int i = 0; count && at && i < count(); ++i)
        if (!std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}
static void* open_ref() {
    if (!g_plugin) return nullptr;
    std::string path(g_plugin);
    path = path.substr(0, path.find_last_of('/') + 1) + "libref.so";
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
static RadTensor TP(void* d, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch) {
    RadTensor t = T(d, dt, { rows, cols });
    t.stride[0] = pitch;
    return t;
}
/* a host buffer of `rows` x `pitch` elements with 128 guard bytes either side, poisoned */
struct HB {
    std::vector<uint8_t> b;
    void* p = nullptr;
    void* d = nullptr;
    size_t n = 0;
    void make(int64_t rows, int64_t pitch, uint32_t dt) {
        n = (size_t)(rows * pitch) * esz(dt);
        b.assign(n + 256, 0xa5);
        p = b.data() + 128;
    }
    bool up() {
        d = dalloc(b.size());
        if (!d) return false;
        st_memcpy(d, b.data(), b.size(), hipMemcpyHostToDevice);
        return true;
    }
    void* dd() const { return (uint8_t*)d + 128; }
    std::vector<uint8_t> down() const {
        std::vector<uint8_t> v(b.size());
        st_memcpy(v.data(), d, v.size(), hipMemcpyDeviceToHost);
        return v;
    }
};
static float val(uint32_t j, uint32_t salt) {
    return (float)((int)(st_mix32(j * 2654435761u + salt) % 2001u) - 1000) / 256.0f;
}
/* the GEMM's codes: multiples of 1/16 in [-4, 4] (E4M3 normals of exponent >= -4 after rounding), so a product's lowest bit
 * is >= 2^-14 and a 32-term block sum <= 2^9 in magnitude: exact in f32 in ANY order -- the card's WMMA order inside a block
 * cannot round differently from libref's ascending sum, and every output must be equal */
static float gval(uint32_t j, uint32_t salt) {
    return (float)((int)(st_mix32(j * 2654435761u + salt) % 129u) - 64) / 16.0f;
}

static void quant_case(const RadKernelInfo* row, const RadKernelInfo* oracle, int64_t M, int64_t K) {
    const int64_t xp = K + 64, G = 32;
    HB x, qh, sh;
    x.make(M, xp, RAD_BF16);
    qh.make(M, K, RAD_F8E4M3);
    sh.make(M, K / G, RAD_F32);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < K; ++c) {
            float v = val((uint32_t)(r * K + c), 0x51u);
            if (r == 1) v = 0.0f;                    /* a zero row: amax floored at 1e-4 */
            if (r == 2) v *= 3.0e4f;                 /* a huge row */
            if (r == 3) v *= 1.0e-6f;                /* a tiny row */
            if (c / G == 5 && r == 4) v = 0.0f;      /* one zero group inside a row */
            ((uint16_t*)x.p)[r * xp + c] = h_to_bf16(v);
        }
    std::vector<RadParam> ps = { PI("M", M), PI("n", K), PI("group", G), PS("dtype", "bf16") };
    /* the oracle on the host */
    std::vector<RadTensor> ht = { TP(x.p, RAD_BF16, M, K, xp), TP(qh.p, RAD_F8E4M3, M, K, K), TP(sh.p, RAD_F32, M, K / G, K / G) };
    RadArgs ha{};
    ha.t = ht.data();
    ha.n_t = (int)ht.size();
    ha.p = ps.data();
    ha.n_p = (int)ps.size();
    const int orc = oracle->launch(&ha, nullptr);
    /* the row on the card, from poisoned outputs */
    HB qd = qh, sd = sh;
    std::fill(qd.b.begin(), qd.b.end(), 0xa5);
    std::fill(sd.b.begin(), sd.b.end(), 0xa5);
    qd.p = qd.b.data() + 128;
    sd.p = sd.b.data() + 128;
    if (!x.up() || !qd.up() || !sd.up()) { ok("quant_act_fp8_ue8m0 allocation", false); return; }
    std::vector<RadTensor> dt = { TP(x.dd(), RAD_BF16, M, K, xp), TP(qd.dd(), RAD_F8E4M3, M, K, K),
                                  TP(sd.dd(), RAD_F32, M, K / G, K / G) };
    run_case(row, dt, ps);
    const std::vector<uint8_t> gq = qd.down(), gs = sd.down();
    int64_t dq = 0, ds = 0;
    for (size_t i = 0; i < qh.n; ++i) dq += gq[128 + i] != qh.b[128 + i];
    for (size_t i = 0; i < sh.n; ++i) ds += gs[128 + i] != sh.b[128 + i];
    bool guards = true;
    for (int i = 0; i < 128; ++i)
        guards = guards && gq[i] == 0xa5 && gq[gq.size() - 1 - i] == 0xa5 && gs[i] == 0xa5 && gs[gs.size() - 1 - i] == 0xa5;
    ok(fmt("quant_act_fp8_ue8m0 = libref byte for byte, M %lld K %lld (pitched input; zero / huge / tiny rows): %lld code "
           "bytes, %lld scale bytes differ%s", (long long)M, (long long)K, (long long)dq, (long long)ds,
           guards ? "" : "; GUARDS TOUCHED").c_str(),
       orc == RAD_OK && dq == 0 && ds == 0 && guards, orc == RAD_OK ? "" : "oracle rc " + std::to_string(orc));
    dfree_all();
}

/* the GEMM's operands, host bytes: a [M, K] E4M3 at pitch ap, its scale [M, K / 32] f32 (powers of two), w [N, K] E4M3,
 * its scale [N / 32, K / 32] E8M0, y [M, N] bf16 at pitch yp */
struct GemmIn {
    HB a, as, w, ws, y;
    int64_t M = 0, N = 0, K = 0, ap = 0, yp = 0;
    void make(int64_t M_, int64_t N_, int64_t K_, int64_t ap_, int64_t yp_, uint32_t salt) {
        M = M_; N = N_; K = K_; ap = ap_; yp = yp_;
        a.make(M, ap, RAD_F8E4M3);
        as.make(M, K / 32, RAD_F32);
        w.make(N, K, RAD_F8E4M3);
        ws.make(N / 32, K / 32, RAD_E8M0);
        y.make(M, yp, RAD_BF16);
        for (int64_t r = 0; r < M; ++r)
            for (int64_t c = 0; c < K; ++c)
                ((uint8_t*)a.p)[r * ap + c] = rad_f32_to_fp8e4m3(gval((uint32_t)(r * K + c), salt));
        for (int64_t i = 0; i < M * (K / 32); ++i)
            ((float*)as.p)[i] = std::ldexp(1.0f, -(int)(st_mix32((uint32_t)i ^ salt) % 6u) - 2);
        for (int64_t i = 0; i < N * K; ++i) ((uint8_t*)w.p)[i] = rad_f32_to_fp8e4m3(gval((uint32_t)i, salt ^ 0x9e37u));
        for (int64_t i = 0; i < (N / 32) * (K / 32); ++i) ((uint8_t*)ws.p)[i] = (uint8_t)(120u + st_mix32((uint32_t)i + salt) % 11u);
    }
    std::vector<RadTensor> host() {
        return { TP(a.p, RAD_F8E4M3, M, K, ap), TP(as.p, RAD_F32, M, K / 32, K / 32), TP(w.p, RAD_F8E4M3, N, K, K),
                 TP(ws.p, RAD_E8M0, N / 32, K / 32, K / 32), TP(y.p, RAD_BF16, M, N, yp) };
    }
    bool up() { return a.up() && as.up() && w.up() && ws.up() && y.up(); }
    std::vector<RadTensor> dev() {
        return { TP(a.dd(), RAD_F8E4M3, M, K, ap), TP(as.dd(), RAD_F32, M, K / 32, K / 32), TP(w.dd(), RAD_F8E4M3, N, K, K),
                 TP(ws.dd(), RAD_E8M0, N / 32, K / 32, K / 32), TP(y.dd(), RAD_BF16, M, N, yp) };
    }
};
static std::vector<RadParam> gemm_ps(int64_t M, int64_t N, int64_t K) {
    return { PI("M", M), PI("N", N), PI("K", K), PI("group", 32), PS("dtype", "fp8a8e8") };
}

static void gemm_case(const Plugin& pl, const RadKernelInfo* row, const RadKernelInfo* oracle, int64_t M, int64_t N, int64_t K,
                      bool pitched) {
    GemmIn g;
    g.make(M, N, K, pitched ? K + 128 : K, pitched ? N + 64 : N, (uint32_t)(M * 7919 + N * 31 + K));
    std::vector<RadParam> ps = gemm_ps(M, N, K);
    /* the oracle: libref's gemm_nt_q (7 slots: the two optional ones absent) into a copy of y */
    GemmIn hr = g;
    hr.a.p = hr.a.b.data() + 128; hr.as.p = hr.as.b.data() + 128; hr.w.p = hr.w.b.data() + 128;
    hr.ws.p = hr.ws.b.data() + 128; hr.y.p = hr.y.b.data() + 128;
    std::vector<RadTensor> ht = hr.host();
    RadTensor none;
    std::memset(&none, 0, sizeof none);
    ht.push_back(none);
    ht.push_back(none);
    RadArgs ha{};
    ha.t = ht.data();
    ha.n_t = (int)ht.size();
    ha.p = ps.data();
    ha.n_p = (int)ps.size();
    const int orc = oracle->launch(&ha, nullptr);
    if (!g.up()) { ok("gemm_fp8a8e8_b32 allocation", false); return; }
    std::vector<RadTensor> dt = g.dev();
    pin_mt(pl, 16);
    pin_ld(pl, 4);   /* the first form exactly: byte staging, 16-row tile */
    run_case(row, dt, ps);
    const std::vector<uint8_t> got = g.y.down();
    /* the other forms over the same operands into a re-poisoned y: every byte (padding included) = the first form's --
     * 16-byte staging at tile 16 / 32 / 64 */
    int64_t mtdiff[3] = { 0, 0, 0 };
    for (int t = 0; t < 3; ++t) {
        st_memcpy(g.y.d, g.y.b.data(), g.y.b.size(), hipMemcpyHostToDevice);
        pin_ld(pl, 16);
        pin_mt(pl, 16 << t);   /* 16, 32, 64 */
        run_case(row, dt, ps);
        const std::vector<uint8_t> g2 = g.y.down();
        for (size_t i = 0; i < got.size(); ++i) mtdiff[t] += g2[i] != got[i];
    }
    pin_mt(pl, 16);
    pin_ld(pl, 0);
    int64_t diff = 0, nonfin = 0;
    float worst = 0;
    bool guards = true;
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < N; ++c) {
            const size_t o = (size_t)(r * g.yp + c) * 2;
            uint16_t gv, wv;
            std::memcpy(&gv, got.data() + 128 + o, 2);
            std::memcpy(&wv, hr.y.b.data() + 128 + o, 2);
            if (gv != wv) {
                ++diff;
                const float a = h_bf16(gv), b = h_bf16(wv);
                if (!std::isfinite(a)) ++nonfin;
                worst = std::max(worst, std::fabs(a - b) / std::max(1e-30f, std::fabs(b)));
            }
        }
    for (int64_t r = 0; r < M && pitched; ++r)
        for (int64_t c = N; c < g.yp; ++c) {
            const size_t o = (size_t)(r * g.yp + c) * 2;
            guards = guards && got[128 + o] == 0xa5 && got[128 + o + 1] == 0xa5;
        }
    for (int i = 0; i < 128; ++i) guards = guards && got[i] == 0xa5 && got[got.size() - 1 - i] == 0xa5;
    ok(fmt("gemm_fp8a8e8_b32 vs libref gemm_nt_q fp8a8e8, M %lld N %lld K %lld%s: %lld of %lld bf16 outputs differ (worst rel %.3g)%s",
           (long long)M, (long long)N, (long long)K, pitched ? " (pitched a / y)" : "", (long long)diff, (long long)(M * N),
           worst, guards ? "" : "; PADDING / GUARDS TOUCHED").c_str(),
       orc == RAD_OK && diff == 0 && nonfin == 0 && guards, orc == RAD_OK ? "" : "oracle rc " + std::to_string(orc));
    ok(fmt("gemm_fp8a8e8_b32 16-byte staging at M tile 16 / 32 / 64 = the first form byte for byte, M %lld N %lld K %lld%s: "
           "%lld / %lld / %lld bytes differ", (long long)M, (long long)N, (long long)K, pitched ? " (pitched)" : "",
           (long long)mtdiff[0], (long long)mtdiff[1], (long long)mtdiff[2]).c_str(),
       mtdiff[0] == 0 && mtdiff[1] == 0 && mtdiff[2] == 0);
    dfree_all();
}

static void gemm_time(const Plugin& pl, const RadKernelInfo* row, const char* name, int64_t N, int64_t K, int64_t M, int w, int mt,
                      int ld) {
    const char* rn = ld == 4      ? "gemm_fp8a8e8_b32_first"
                     : mt == 64  ? "gemm_fp8a8e8_b32_mt64"
                     : mt == 32  ? "gemm_fp8a8e8_b32_mt32"
                                 : "gemm_fp8a8e8_b32";
    pin_mt(pl, mt);
    pin_ld(pl, ld);
    GemmIn g;
    g.make(M, N, K, K, N, 0x7u);
    std::vector<RadParam> ps = gemm_ps(M, N, K);
    if (!g.up()) { std::printf("  P30W %s %lldx%lld M=%lld w%d %s NA us (allocation)\n", name, (long long)N, (long long)K, (long long)M, w, rn); return; }
    std::vector<RadTensor> dt = g.dev();
    RadArgs a{};
    a.t = dt.data();
    a.n_t = (int)dt.size();
    a.p = ps.data();
    a.n_p = (int)ps.size();
    a.world_size = 1;
    int rc = RAD_OK;
    for (int i = 0; i < 2 && rc == RAD_OK; ++i) rc = row->launch(&a, nullptr);
    st_stream_sync(nullptr);
    const int it = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it && rc == RAD_OK; ++i) rc = row->launch(&a, nullptr);
    st_stream_sync(nullptr);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    const double tf = 2.0 * (double)M * (double)N * (double)K / (us * 1e6);
    if (rc == RAD_OK)
        std::printf("  P30W %s %lldx%lld M=%lld w%d %s %.1f us %.1f TF/s\n", name, (long long)N, (long long)K, (long long)M, w, rn, us,
                    tf);
    else
        std::printf("  P30W %s %lldx%lld M=%lld w%d %s NA us (rc %d)\n", name, (long long)N, (long long)K, (long long)M, w, rn, rc);
    pin_mt(pl, 16);
    pin_ld(pl, 0);
    dfree_all();
}

}  // namespace p30ost

static void p30o_cases(const Plugin& pl) {
    using namespace p30ost;
    /* each row's child runs its own check; the named case runs both and the times */
    const bool all = sel("fp8a8e8"), wq = all || sel("quant_act_fp8_ue8m0"), wg = all || sel("gemm_fp8a8e8_b32");
    if (!wq && !wg) return;
    if (!g_case) std::printf("\nP30o: the release's block FP8 (quant_act_fp8_ue8m0, gemm_fp8a8e8_b32) vs libref\n");
    const RadKernelInfo* rq = find(pl, "quant_act_fp8_ue8m0");
    const RadKernelInfo* rg = find(pl, "gemm_fp8a8e8_b32");
    void* ref = open_ref();
    const RadKernelInfo* oq = ref ? oracle_of(ref, "quant_act_fp8_ue8m0") : nullptr;
    const RadKernelInfo* og = ref ? oracle_of(ref, "gemm_nt_q") : nullptr;
    if (!rq || !rg || !oq || !og) {
        ok("P30o rows and libref oracles", false, fmt("quant %d gemm %d libref %d oracle quant %d gemm %d", rq != nullptr,
                                                       rg != nullptr, ref != nullptr, oq != nullptr, og != nullptr));
        if (ref) dlclose(ref);
        return;
    }
    if (wq)
        for (int64_t K : { 1024, 5120, 6144 })
            for (int64_t M : { 1, 7, 33, 130 }) quant_case(rq, oq, M, K);
    struct NK { int64_t N, K; };
    if (wg)
        for (const NK s : { NK{ 64, 128 }, NK{ 512, 1024 }, NK{ 1280, 5120 } })
            for (int64_t M : { 1, 16, 17, 130 })
                for (bool pitched : { false, true }) gemm_case(pl, rg, og, M, s.N, s.K, pitched);
    if (!all || !g_case) {
        dlclose(ref);
        return;
    }
    struct Sh { const char* n; int64_t N, K; int w; };
    const Sh shapes[] = { { "q_b", 8192, 1280, 40 }, { "o_b", 5120, 2048, 40 }, { "q_a", 1280, 5120, 40 }, { "o_a", 1024, 4096, 80 },
                          { "kv", 512, 5120, 40 }, { "shexp_gate_up", 1536, 5120, 40 }, { "shexp_down", 5120, 768, 40 },
                          { "engram_kv", 25600, 6144, 2 } };
    for (int64_t M : { 16, 1100, 4096 })
        for (int f = 0; f < 4; ++f) {   /* the first form (byte staging, tile 16), 16-byte staging at tile 16, 32, 64 */
            if (M <= 16 && f >= 2) continue;   /* a launch of <= 16 rows is one 16-row tile at any pin */
            for (const Sh& s : shapes) gemm_time(pl, rg, s.n, s.N, s.K, M, s.w, f == 3 ? 64 : f == 2 ? 32 : 16, f == 0 ? 4 : 16);
        }
    dlclose(ref);
}
