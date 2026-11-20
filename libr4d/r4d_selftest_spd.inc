/* r4d_selftest_spd.inc -- lever SPD-F's one-card check and bench (ds41-etp-spd, 15.11.2026; included by r4d_selftest.cpp after
 * r4d_selftest_fuse.inc; runs only when named: --case spdf32). Every decode-band form of the two f32 projections is held BYTE
 * FOR BYTE to the stock form at its height, then timed against it:
 *
 *   router_gemm_f32_scored_topk_scatter (ds41_rfuse 2, the trunk router at <= 48 rows): DeepSeek-V4.1's shape (K 5120, 384
 *       experts, top 6, sqrtsoftplus, norm), the expert-split layout (every expert local: n_local 384, base 0) and the
 *       expert-parallel one (n_local 96, base 96), bf16 weights (production) and f32, random / tied / NaN rows; every
 *       output (logits, ids, weights, sorted_tok, expert_offset, expert_count) with a guard tail; each form launched twice
 *       (the arrival counter must come back to zero).
 *   gemm_nt_f32 (the compressor's kv | gate, N 1024, K 5120; bf16 and f32 weights; and N 384 = the router's projection
 *       alone): y with a guard tail.
 * The forms come from the exported pin r4d_f32dec_pin (router form, gemm form; 0 = the stock form). SPDPERF lines: warm (back
 * to back, mean) and cold (each launch after a 256 MiB read, median), us a call, per (M, form). */

namespace spd {

using fuse::args;
using fuse::bf;
using fuse::bytes;
using fuse::gauss;
using fuse::rnd;
using fuse::time2;
using fuse::up;

struct RgsForm { int code, r, c, u, nw, rw = 1; };   /* rw: tp8 loop D28's row groups a workgroup (the 2-D wave map) */
static const RgsForm kRgs[] = { { 1, 8, 2, 8, 16 }, { 2, 8, 4, 8, 16 }, { 3, 4, 4, 8, 16 }, { 4, 4, 2, 8, 16 }, { 5, 2, 4, 16, 16 },
                                { 6, 4, 4, 8, 8 },  { 7, 2, 4, 16, 8 }, { 8, 4, 8, 8, 16 }, { 9, 2, 2, 16, 8 }, { 10, 2, 2, 16, 16 },
                                { 11, 8, 1, 8, 16 }, { 12, 4, 2, 16, 8 }, { 13, 1, 4, 16, 8 }, { 14, 2, 8, 16, 8 },
                                { 15, 4, 8, 8, 16, 4 }, { 16, 4, 4, 8, 16, 4 }, { 17, 4, 8, 8, 16, 2 }, { 18, 4, 4, 8, 8, 2 },
                                { 19, 4, 4, 8, 8 }, { 20, 4, 4, 16, 8 }, { 21, 4, 4, 8, 8 } };   /* 19 / 20: lever RGP (the selection a row group at a time);
                                                                      21: lever RGK (19 + the epilogue scores / keys) */
static const int kGemm[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

static void rgs_part(const Plugin& pl, void (*pin)(int, int)) {
    const RadKernelInfo* f2 = find(pl, "router_gemm_f32_scored_topk_scatter_bf16");
    if (!f2 || !f2->init || !f2->fini) { ok("spdf32 router row", false, "router_gemm_f32_scored_topk_scatter_bf16 absent"); return; }
    void* inst = nullptr;
    if (f2->init(nullptr, 0, 0, 1, &inst) != RAD_OK || !inst) { ok("spdf32 router init", false, "no instance"); return; }
    constexpr int64_t K = 5120, NE = 384, TK = 6;
    const double scale = 1.5;
    const size_t tail = 64;
    static const char* const kOut[6] = { "logits", "ids", "weights", "sorted_tok", "expert_offset", "expert_count" };
    int checks = 0, bad = 0;
    for (int wbf : { 1, 0 }) {
        for (int mode : { 0, 1, 2 }) {
            fuse::g_rs = 0x5EEDD00Dull + (uint64_t)(wbf * 7 + mode);
            std::vector<float> wf((size_t)(NE * K)), bias((size_t)NE);
            for (auto& v : wf) v = (float)(gauss() * 0.02);
            for (auto& v : bias) v = (float)(gauss() * 0.05);
            if (mode == 1)
                for (int64_t e = 1; e < NE; e += 2) {
                    std::memcpy(&wf[(size_t)(e * K)], &wf[(size_t)((e - 1) * K)], (size_t)K * 4);
                    bias[(size_t)e] = bias[(size_t)e - 1];
                }
            std::vector<uint16_t> wb((size_t)(NE * K));
            for (size_t i = 0; i < wb.size(); ++i) {
                wb[i] = bf(wf[i]);
                if (wbf) { uint32_t u = (uint32_t)wb[i] << 16; std::memcpy(&wf[i], &u, 4); }
            }
            for (int64_t M : { 1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 20, 24, 28, 32, 33, 36, 40, 44, 47, 48, 49, 56, 63, 64 }) {
                void* dw = wbf ? up(wb) : up(wf);
                void* db = up(bias);
                std::vector<uint16_t> x((size_t)(M * K));
                for (int64_t r = 0; r < M; ++r) {
                    const double rs_ = std::ldexp(1.0, (int)(rnd() % 5) - 2);
                    for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs_));
                    if (mode == 2 && (r % 3) == 1) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
                }
                void* dx = up(x);
                for (int lay : { 0, 1 }) {   /* 0: expert split (n_local 384, base 0); 1: expert parallel (96 at base 96) */
                    if (lay == 1 && (mode != 0 || wbf != 1)) continue;
                    const int64_t NL = lay ? 96 : NE, base = lay ? 96 : 0;
                    const std::array<size_t, 6> sz = { (size_t)(M * NE * 4) + tail, (size_t)(M * TK * 4) + tail,
                                                       (size_t)(M * TK * 4) + tail, (size_t)(M * TK * 4) + tail,
                                                       (size_t)(NL + 1) * 4 + tail, (size_t)NL * 4 + tail };
                    void* o[2][6];
                    for (int f = 0; f < 2; ++f)
                        for (int j = 0; j < 6; ++j) { o[f][j] = dalloc(sz[(size_t)j]); if (o[f][j]) st_memset(o[f][j], 0xA5, sz[(size_t)j]); }
                    if (!dx || !dw || !db || !o[1][5]) { ok("spdf32 alloc", false, "hipMalloc failed"); pin(-1, -1); dfree_all(); f2->fini(inst); return; }
                    std::vector<RadParam> pf2 = { PI("M", M), PI("K", K), PI("n_expert", NE), PI("n_local", NL), PI("top_k", TK),
                                                  PI("norm", 1), PF("route_scale", scale), PS("score", "sqrtsoftplus"),
                                                  PS("dtype", "bf16"), PI("expert_base", base) };
                    auto tens = [&](void* const* q) {
                        return std::vector<RadTensor>{ T(dx, RAD_BF16, { M, K }), T(dw, wbf ? RAD_BF16 : RAD_F32, { NE, K }),
                                                       T(db, RAD_F32, { NE }), T(q[0], RAD_F32, { M, NE }),
                                                       T(q[1], RAD_I32, { M, TK }), T(q[2], RAD_F32, { M, TK }),
                                                       T(q[3], RAD_I32, { M * TK }), T(q[4], RAD_I32, { NL + 1 }),
                                                       T(q[5], RAD_I32, { NL }) };
                    };
                    std::vector<RadTensor> t0 = tens(o[0]), t1 = tens(o[1]);
                    RadArgs a0 = args(t0, pf2, inst), a1 = args(t1, pf2, inst);
                    pin(0, -1);
                    const int r0 = f2->launch(&a0, nullptr);
                    st_stream_sync(nullptr);
                    std::vector<std::vector<uint8_t>> ref(6);
                    for (int j = 0; j < 6; ++j) ref[(size_t)j] = bytes(o[0][j], sz[(size_t)j]);
                    const bool timed = wbf && mode == 0 && lay == 0 &&
                                       (M == 4 || M == 8 || M == 16 || M == 24 || M == 32 || M == 40 || M == 47 || M == 48 || M == 56 || M == 64);
                    double sw = 0, sc = 0;
                    if (timed && r0 == RAD_OK) time2([&]() { return f2->launch(&a0, nullptr); }, 200, &sw, &sc);
                    std::string fails;
                    for (const RgsForm& fm : kRgs) {
                        for (int j = 0; j < 6; ++j) st_memset(o[1][j], 0xA5, sz[(size_t)j]);
                        pin(fm.code, -1);
                        const int ra = f2->launch(&a1, nullptr), rb = f2->launch(&a1, nullptr);   /* twice: the counter resets */
                        st_stream_sync(nullptr);
                        int first = -1;
                        for (int j = 0; r0 == RAD_OK && ra == RAD_OK && rb == RAD_OK && j < 6 && first < 0; ++j)
                            if (bytes(o[1][j], sz[(size_t)j]) != ref[(size_t)j]) first = j;
                        const bool same = r0 == RAD_OK && ra == RAD_OK && rb == RAD_OK && first < 0;
                        ++checks;
                        if (!same) {
                            ++bad;
                            fails += fmt(" form %d (%dx%d U%d NW%d): %s;", fm.code, fm.r, fm.c, fm.u, fm.nw,
                                         ra != RAD_OK || rb != RAD_OK || r0 != RAD_OK ? fmt("rc %d %d %d", r0, ra, rb).c_str()
                                                                                       : kOut[first]);
                        }
                        if (timed && same) {
                            double w = 0, c = 0;
                            time2([&]() { return f2->launch(&a1, nullptr); }, 200, &w, &c);
                            std::printf("  SPDPERF rgs M=%lld form %2d (%dx%d U%d NW%d RW%d) warm %7.2f cold %7.2f us | stock warm %7.2f "
                                        "cold %7.2f us | %+6.2f / %+6.2f\n",
                                        (long long)M, fm.code, fm.r, fm.c, fm.u, fm.nw, fm.rw, w, c, sw, sc, w - sw, c - sc);
                        }
                    }
                    pin(-1, -1);
                    ok(fmt("spdf32 router %s weights, %s, M=%lld, %s: %d forms byte for byte vs the stock form (6 outputs + tails)",
                           wbf ? "bf16" : "f32", mode == 0 ? "random" : mode == 1 ? "tied ranks" : "NaN rows", (long long)M,
                           lay ? "EP slice 96 @ 96" : "ETP 384 @ 0", (int)(sizeof(kRgs) / sizeof(kRgs[0]))).c_str(),
                       fails.empty(), fails);
                }
                dfree_all();
            }
        }
    }
    fuse::info(fmt("spdf32 router: %d of %d (weights, mode, M, layout, form) checks byte-identical", checks - bad, checks));
    f2->fini(inst);
}

static void gemm_part(const Plugin& pl, void (*pin)(int, int)) {
    const RadKernelInfo* rg = find(pl, "gemm_nt_f32_bf16");
    if (!rg) { ok("spdf32 gemm row", false, "gemm_nt_f32_bf16 absent"); return; }
    constexpr int64_t K = 5120;
    const size_t tail = 64;
    int checks = 0, bad = 0;
    for (int64_t N : { (int64_t)1024, (int64_t)384 }) {
        for (int wbf : { 1, 0 }) {
            if (N == 384 && !wbf) continue;
            fuse::g_rs = 0x5EEDC0DEull + (uint64_t)(wbf * 3 + N);
            std::vector<float> wf((size_t)(N * K));
            for (auto& v : wf) v = (float)(gauss() * 0.02);
            std::vector<uint16_t> wb((size_t)(N * K));
            for (size_t i = 0; i < wb.size(); ++i) {
                wb[i] = bf(wf[i]);
                if (wbf) { uint32_t u = (uint32_t)wb[i] << 16; std::memcpy(&wf[i], &u, 4); }
            }
            for (int64_t M : { 1, 2, 3, 4, 6, 8, 12, 16, 17, 24, 32, 33, 40, 47, 48, 63 }) {
                void* dw = wbf ? up(wb) : up(wf);
                std::vector<uint16_t> x((size_t)(M * K));
                for (int64_t r = 0; r < M; ++r) {
                    const double rs_ = std::ldexp(1.0, (int)(rnd() % 5) - 2);
                    for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs_));
                    if ((r % 5) == 3) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
                }
                void* dx = up(x);
                const size_t ys = (size_t)(M * N * 4) + tail;
                void* y0 = dalloc(ys);
                void* y1 = dalloc(ys);
                if (!dx || !dw || !y1) { ok("spdf32 gemm alloc", false, "hipMalloc failed"); pin(-1, -1); dfree_all(); return; }
                st_memset(y0, 0xA5, ys);
                std::vector<RadParam> pg = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
                std::vector<RadTensor> t0 = { T(dx, RAD_BF16, { M, K }), T(dw, wbf ? RAD_BF16 : RAD_F32, { N, K }), T(y0, RAD_F32, { M, N }) };
                std::vector<RadTensor> t1 = { T(dx, RAD_BF16, { M, K }), T(dw, wbf ? RAD_BF16 : RAD_F32, { N, K }), T(y1, RAD_F32, { M, N }) };
                RadArgs a0 = args(t0, pg), a1 = args(t1, pg);
                pin(-1, 0);
                const int r0 = rg->launch(&a0, nullptr);
                st_stream_sync(nullptr);
                const std::vector<uint8_t> ref = bytes(y0, ys);
                const bool timed = wbf && (M == 4 || M == 8 || M == 16 || M == 24 || M == 32 || M == 40 || M == 47);
                double sw = 0, sc = 0;
                if (timed && r0 == RAD_OK) time2([&]() { return rg->launch(&a0, nullptr); }, 200, &sw, &sc);
                std::string fails;
                for (int f : kGemm) {
                    st_memset(y1, 0xA5, ys);
                    pin(-1, f);
                    const int ra = rg->launch(&a1, nullptr);
                    st_stream_sync(nullptr);
                    const bool same = r0 == RAD_OK && ra == RAD_OK && bytes(y1, ys) == ref;
                    ++checks;
                    if (!same) { ++bad; fails += fmt(" form %d: %s;", f, ra != RAD_OK ? fmt("rc %d", ra).c_str() : "y differs"); }
                    if (timed && same) {
                        double w = 0, c = 0;
                        time2([&]() { return rg->launch(&a1, nullptr); }, 200, &w, &c);
                        std::printf("  SPDPERF gemm N=%lld M=%lld form %2d warm %8.2f cold %8.2f us | stock warm %8.2f cold %8.2f us | "
                                    "%+7.2f / %+7.2f\n",
                                    (long long)N, (long long)M, f, w, c, sw, sc, w - sw, c - sc);
                    }
                }
                pin(-1, -1);
                ok(fmt("spdf32 gemm_nt_f32 N=%lld %s weights M=%lld: %d forms byte for byte vs the stock form (y + tail)",
                       (long long)N, wbf ? "bf16" : "f32", (long long)M, (int)(sizeof(kGemm) / sizeof(kGemm[0]))).c_str(),
                   fails.empty(), fails);
                dfree_all();
            }
        }
    }
    fuse::info(fmt("spdf32 gemm: %d of %d (N, weights, M, form) checks byte-identical", checks - bad, checks));
}

/* ---- pool 4 measurement (no kernel change): the stock decode attention (ds41_attn_sparse_sink_d512_bf16, the TP4 rank's 16
 * heads, window 128, 512 selections at ratio 4, bf16 latent) at the profile's heights, to price lever 1U-ATT (a sequence's verify
 * rows sharing one load of their keys). k1_attn_fix's operands; per launch, warm (back to back) and cold (256 MiB read before
 * each), us a launch and a row. "shared" = every row of a sequence takes the selections of the sequence's first row (the most
 * key overlap verify rows can have: the window shifts by a row, the top-k the same); "own" = each row its own random top-k. */
static void attn_part(const Plugin& pl) {
    const RadKernelInfo* ra = find(pl, "ds41_attn_sparse_sink_d512_bf16");
    if (!ra) { ok("spdattn row", false, "ds41_attn_sparse_sink_d512_bf16 absent"); return; }
    struct Shape { const char* name; int S, R; bool shared; };
    const Shape shapes[] = { { "c32: 32 seqs x 1 row", 32, 1, false }, { "c16: 16 x 1", 16, 1, false },
                             { "48 seqs x 1 row", 48, 1, false },     { "c8 depth 5: 8 x 6, own top-k", 8, 6, false },
                             { "c8 depth 5: 8 x 6, shared top-k", 8, 6, true }, { "c8 depth 3: 8 x 4, own", 8, 4, false },
                             { "c8 depth 3: 8 x 4, shared", 8, 4, true },  { "1 seq x 48 rows, shared", 1, 48, true },
                             { "1 seq x 32 rows, shared", 1, 32, true },  { "c1 depth 4: 1 x 5, shared", 1, 5, true },
                             { "1 x 1", 1, 1, false } };
    std::mt19937 rng(4242);
    const int64_t D = 512;
    for (const Shape& sh : shapes) {
        std::vector<std::pair<int, int>> seqs;
        for (int s = 0; s < sh.S; ++s) seqs.push_back({ sh.R, 3000 + 190 * s + (int)(rng() % 150) });
        K1Attn f = k1_attn_fix(rng, 4, 512, seqs);
        if (sh.shared)
            for (int s = 0; s < f.S; ++s)
                for (int r = f.cu[(size_t)s] + 1; r < f.cu[(size_t)s + 1]; ++r)
                    std::memcpy(&f.sel[(size_t)r * f.topk], &f.sel[(size_t)f.cu[(size_t)s] * f.topk], (size_t)f.topk * 4);
        const int64_t M = f.M, H = f.H, n = H * D;
        auto upl = [&](const void* h, size_t b) {
            void* d = dalloc(b);
            if (d && b) st_memcpy(d, h, b, hipMemcpyHostToDevice);
            return d;
        };
        void* dq = upl(f.q.data(), f.q.size() * 2);
        void* dwc = upl(f.wc.data(), f.wc.size() * 2);
        void* dwbt = upl(f.wbt.data(), f.wbt.size() * 4);
        void* dwu = upl(f.wused.data(), f.wused.size() * 4);
        void* dpos = upl(f.pos.data(), f.pos.size() * 4);
        void* dcu = upl(f.cu.data(), f.cu.size() * 4);
        void* dsk = upl(f.sink.data(), f.sink.size() * 4);
        void* dcc = upl(f.cc.data(), f.cc.size() * 2);
        void* dcbt = upl(f.cbt.data(), f.cbt.size() * 4);
        void* dsel = upl(f.sel.data(), f.sel.size() * 4);
        void* dout = dalloc((size_t)(M * n * 2));
        if (!dq || !dwc || !dwbt || !dwu || !dpos || !dcu || !dsk || !dcc || !dcbt || !dsel || !dout) {
            ok("spdattn alloc", false, "hipMalloc failed"); dfree_all(); return; }
        std::vector<RadParam> pa = { PI("M", M), PI("heads", H), PI("dim", D), PI("window", f.win), PI("ratio", f.ratio),
                                     PI("topk", f.topk), PF("scale", 1.0 / std::sqrt((double)D)), PS("dtype", "bf16") };
        std::vector<RadTensor> ts = { T(dq, RAD_BF16, { M, n }), T(dwc, RAD_BF16, { f.nwblk, 1, f.bs, D }),
                                      T(dwbt, RAD_I32, { f.S, f.nwbt }), T(dwu, RAD_I32, { f.S }),
                                      T(dcc, RAD_BF16, { f.ncblk, 1, f.ratio, D / f.ratio }), T(dcbt, RAD_I32, { f.S, f.ncbt }),
                                      T(dsel, RAD_I32, { M, f.topk }), T(dpos, RAD_I32, { M }), T(dcu, RAD_I32, { f.S + 1 }),
                                      T(dsk, RAD_F32, { H }), T(dout, RAD_BF16, { M, n }) };
        const int rc = k1_launch(ra, ts, pa);
        if (rc != RAD_OK) { ok(fmt("spdattn %s", sh.name).c_str(), false, fmt("rc %d", rc)); dfree_all(); continue; }
        double w = 0, c = 0;
        g_k1_sync = false;
        time2([&]() { return k1_launch(ra, ts, pa); }, 200, &w, &c);
        g_k1_sync = true;
        std::printf("  SPDATTN %-34s M=%3lld: warm %8.2f us (%6.2f a row) cold %8.2f us (%6.2f a row)\n", sh.name, (long long)M, w,
                    w / (double)M, c, c / (double)M);
        dfree_all();
    }
}

static void spd_case(const Plugin& pl) {
    auto pin = (void (*)(int, int))dlsym(pl.h, "r4d_f32dec_pin");
    if (!pin) { ok("spdf32", false, "r4d_f32dec_pin absent (a plugin without lever SPD-F)"); return; }
    const char* part = std::getenv("SPD_PART");
    if (!part || !std::strcmp(part, "rgs") || !std::strcmp(part, "all")) rgs_part(pl, pin);
    if (!part || !std::strcmp(part, "gemm") || !std::strcmp(part, "all")) gemm_part(pl, pin);
    if (!part || !std::strcmp(part, "attn") || !std::strcmp(part, "all")) attn_part(pl);
    pin(-1, -1);
}

}  // namespace spd

static void spd_cases(const Plugin& pl) {
    if (g_case && sel("spdf32")) spd::spd_case(pl);
}
