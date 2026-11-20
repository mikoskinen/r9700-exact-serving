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

/* ---- lever RGM (DRAFTER track round 40, RULE B; SPD_PART=rgm): ds41_f32dec 10 forms every router logit on bf16 WMMA (gemm_nt_f32
 * code 40 at the router's shape, any M; the fused decode launch form 22 at 1..64 rows). Not the stock bytes, so the checks are rule B's:
 *   (a) gemm code 40 (the pin 40): repeat = the same bytes; rows of an M-row call = the same rows of a 2112-row call (M 1..1000);
 *       element loads (an x row pitch of K + 1, base off 16 bytes) = the 16-byte loads; the 96-column split (4 slices) = the 384-column
 *       call's columns; vs the stock chain: how far the logits move and how many rows change their top-6 set (information);
 *   (b) under the SWITCH (rad_fn4_switch ds41_f32dec 10, no pin): the fused launch (form 22, launched twice) = the chain gemm_nt_f32
 *       (code 40 by the rule) -> router_scored_topk -> moe_scatter, all 6 outputs + guard tails, expert split and EP slice, M 1..64;
 *       its logits = (a)'s rows;
 *   (c) SPDPERF: the fused launch at value 10 vs 9 (RGK) vs 1 (BEST's rule); gemm_nt_f32 at the router's prefill shapes value 10 vs
 *       value 1 + f32_pf 25 (BEST's prefill form). */
static void rgm_part(const Plugin& pl, void (*pin)(int, int)) {
    const RadKernelInfo* rg = find(pl, "gemm_nt_f32_bf16");
    const RadKernelInfo* rs = find(pl, "router_scored_topk_f32");
    const RadKernelInfo* rc = find(pl, "moe_scatter_i32");
    const RadKernelInfo* f2 = find(pl, "router_gemm_f32_scored_topk_scatter_bf16");
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    if (!rg || !rs || !rc || !f2 || !f2->init || !f2->fini || !sw) {
        ok("spdf32 rgm rows", false, "gemm_nt_f32_bf16 / router_scored_topk_f32 / moe_scatter_i32 / router_gemm_f32_scored_topk_scatter_bf16 / "
                                     "rad_fn4_switch absent");
        return;
    }
    /* RGM_SHAPE=NE,TK (DRAFTER round 41, lever RGD): another router shape -- 128,3 = the DSpark drafter's (128 experts, top 3), checked
     * under RGM_VAL (default 10; 12 = lever RGD, the value that takes the DSpark router onto the WMMA logits); unset: the trunk's 384,6 */
    int64_t NE = 384, TK = 6;
    if (const char* e = std::getenv("RGM_SHAPE")) {
        long long a = 0, b2 = 0;
        if (std::sscanf(e, "%lld,%lld", &a, &b2) == 2 && a >= 16 && a % 16 == 0 && a <= 384 && b2 >= 1 && b2 <= 8) { NE = a; TK = b2; }
    }
    const int VAL = [] { const char* e = std::getenv("RGM_VAL"); return e && *e ? std::atoi(e) : 10; }();
    if (NE != 384 || VAL != 10) std::printf("spdf32 rgm: router shape %lld experts, top %lld, value %d (RGM_SHAPE / RGM_VAL)\n", (long long)NE, (long long)TK, VAL);
    if (sw("ds41_f32dec", VAL) != 1) { ok(fmt("spdf32 rgm: ds41_f32dec %d taken", VAL).c_str(), false, "refused (a plugin without the lever)"); return; }
    sw("ds41_f32dec", 0);
    void* inst = nullptr;
    if (f2->init(nullptr, 0, 0, 1, &inst) != RAD_OK || !inst) { ok("spdf32 rgm init", false, "no instance"); return; }
    constexpr int64_t K = 5120, MR = 2112;
    const double scale = 1.5;
    const size_t tail = 64;
    fuse::g_rs = 0x5EEDB00Bull;
    std::vector<uint16_t> wb((size_t)(NE * K));
    for (auto& v : wb) v = bf((float)(gauss() * 0.02));
    std::vector<float> bias((size_t)NE);
    for (auto& v : bias) v = (float)(gauss() * 0.05);
    std::vector<uint16_t> x((size_t)(MR * K));
    for (int64_t r = 0; r < MR; ++r) {
        const double rs_ = std::ldexp(1.0, (int)(rnd() % 5) - 2);
        for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs_));
        if ((r % 97) == 5) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
    }
    /* the same rows at a pitch of K + 1, 8 elements in: every 16-byte load misaligned (code 40's element-load path) */
    std::vector<uint16_t> xu((size_t)(MR * (K + 1) + 16), 0);
    for (int64_t r = 0; r < MR; ++r)
        for (int64_t i = 0; i < K; ++i) xu[(size_t)(8 + 1 + r * (K + 1) + i)] = x[(size_t)(r * K + i)];
    void* dw = up(wb);
    void* db = up(bias);
    void* dx = up(x);
    void* dxu0 = up(xu);
    void* dxu = dxu0 ? (void*)((uint16_t*)dxu0 + 9) : nullptr;
    auto gemm = [&](void* xd, int64_t xpitch, int64_t M, const void* wd, int64_t N, void* yd) -> int {
        std::vector<RadParam> pg = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
        std::vector<RadTensor> t = { T(xd, RAD_BF16, { M, K }), T((void*)wd, RAD_BF16, { N, K }), T(yd, RAD_F32, { M, N }) };
        t[0].stride[0] = xpitch;
        RadArgs a = args(t, pg);
        return rg->launch(&a, nullptr);
    };
    const size_t yrs = (size_t)(MR * NE * 4);
    void* yref = dalloc(yrs + tail);
    void* yrep = dalloc(yrs + tail);
    if (!dw || !db || !dx || !dxu || !yref || !yrep) { ok("spdf32 rgm alloc", false, "hipMalloc failed"); dfree_all(); f2->fini(inst); return; }
    const bool po = std::getenv("RGM_PERFONLY") != nullptr;   /* only (c)'s gemm lines (a tile sweep: R4D_RGM_TILE a process) */
    std::vector<uint8_t> ref;
    if (!po) {
    /* (a) */
    pin(-1, 40);
    st_memset(yref, 0xA5, yrs + tail);
    st_memset(yrep, 0xA5, yrs + tail);
    const int ra = gemm(dx, K, MR, dw, NE, yref), rb = gemm(dx, K, MR, dw, NE, yrep);
    st_stream_sync(nullptr);
    ref = bytes(yref, yrs + tail);
    ok(fmt("spdf32 rgm gemm code 40: M %lld x N %lld launched twice, the same bytes (+ tail)", (long long)MR, (long long)NE).c_str(),
       ra == RAD_OK && rb == RAD_OK && ref == bytes(yrep, yrs + tail), fmt("rc %d %d", ra, rb));
    int checks = 0, bad = 0;
    /* rows [0, M) x columns [c0, c0 + N) of y (pitch N) vs the 2112-row reference; the guard tail untouched */
    auto same_rows = [&](void* yd, int64_t M, int64_t N, int64_t c0) -> std::string {
        const std::vector<uint8_t> got = bytes(yd, (size_t)(M * N * 4) + tail);
        for (int64_t r = 0; r < M; ++r)
            if (std::memcmp(&got[(size_t)(r * N * 4)], &ref[(size_t)((r * NE + c0) * 4)], (size_t)(N * 4)))
                return fmt("row %lld differs", (long long)r);
        for (size_t i = 0; i < tail; ++i)
            if (got[(size_t)(M * N * 4) + i] != 0xA5) return "guard tail written";
        return std::string();
    };
    {
        std::string fails;
        for (int64_t M : { 1, 2, 3, 4, 5, 8, 12, 16, 17, 24, 31, 32, 33, 40, 47, 48, 49, 56, 63, 64, 65, 96, 128, 255, 256, 1000 }) {
            for (int al : { 1, 0 }) {
                if (!al && !(M == 1 || M == 16 || M == 47 || M == 64 || M == 65 || M == 256)) continue;
                void* y = dalloc((size_t)(M * NE * 4) + tail);
                if (!y) { fails += " alloc;"; break; }
                st_memset(y, 0xA5, (size_t)(M * NE * 4) + tail);
                const int r = al ? gemm(dx, K, M, dw, NE, y) : gemm(dxu, K + 1, M, dw, NE, y);
                st_stream_sync(nullptr);
                const std::string d = r != RAD_OK ? fmt("rc %d", r) : same_rows(y, M, NE, 0);
                ++checks;
                if (!d.empty()) { ++bad; fails += fmt(" M %lld %s: %s;", (long long)M, al ? "16-byte loads" : "element loads", d.c_str()); }
            }
        }
        /* glm_router's TP4 column split (rad_glm_tp.h, from 256 rows): 4 slices of NE / 4 columns (96 at 384, 32 at 128) */
        const int64_t SC = NE / 4;
        for (int64_t c0 : { (int64_t)0, SC, 2 * SC, 3 * SC })
            for (int64_t M : { (int64_t)16, (int64_t)48, (int64_t)64, (int64_t)256, MR }) {
                void* y = dalloc((size_t)(M * SC * 4) + tail);
                if (!y) { fails += " alloc;"; break; }
                st_memset(y, 0xA5, (size_t)(M * SC * 4) + tail);
                const int r = gemm(dx, K, M, (const uint16_t*)dw + c0 * K, SC, y);
                st_stream_sync(nullptr);
                const std::string d = r != RAD_OK ? fmt("rc %d", r) : same_rows(y, M, SC, c0);
                ++checks;
                if (!d.empty()) { ++bad; fails += fmt(" split cols %lld.. M %lld: %s;", (long long)c0, (long long)M, d.c_str()); }
            }
        ok(fmt("spdf32 rgm gemm code 40 batch-invariant: rows of M 1..1000 (16-byte and element loads) and the column split's 4 slices "
               "(M 16..2112) = the 2112-row call's bytes (%d calls)", checks).c_str(),
           fails.empty(), fails);
    }
    /* vs the stock chain (information): M 64 */
    {
        pin(-1, 0);
        const int64_t M = 64;
        void* y = dalloc((size_t)(M * NE * 4));
        const int r = y ? gemm(dx, K, M, dw, NE, y) : -1;
        st_stream_sync(nullptr);
        if (r == RAD_OK) {
            std::vector<float> s((size_t)(M * NE)), g((size_t)(M * NE));
            st_memcpy(s.data(), y, s.size() * 4, hipMemcpyDeviceToHost);
            std::memcpy(g.data(), ref.data(), g.size() * 4);
            long nd = 0, rows_set = 0;
            double md = 0, mr = 0;
            for (int64_t i = 0; i < M; ++i) {
                std::vector<std::pair<double, int>> a, b;
                bool nan = false;
                for (int64_t e = 0; e < NE; ++e) {
                    const float u = s[(size_t)(i * NE + e)], v = g[(size_t)(i * NE + e)];
                    if (std::memcmp(&u, &v, 4)) ++nd;
                    if (std::isnan(u) || std::isnan(v)) { nan = true; continue; }
                    md = std::max(md, std::fabs((double)u - v));
                    if (u != 0) mr = std::max(mr, std::fabs(((double)u - v) / u));
                    auto sc_ = [&](double l) { return std::sqrt(l > 20 ? l : std::log1p(std::exp(l))) + bias[(size_t)e]; };
                    a.push_back({ -sc_(u), (int)e });
                    b.push_back({ -sc_(v), (int)e });
                }
                if (nan) continue;
                std::partial_sort(a.begin(), a.begin() + TK, a.end());
                std::partial_sort(b.begin(), b.begin() + TK, b.end());
                std::vector<int> sa, sb;
                for (int j = 0; j < TK; ++j) { sa.push_back(a[(size_t)j].second); sb.push_back(b[(size_t)j].second); }
                std::sort(sa.begin(), sa.end());
                std::sort(sb.begin(), sb.end());
                if (sa != sb) ++rows_set;
            }
            fuse::info(fmt("spdf32 rgm vs the stock chain (M 64, random rows): %ld of %lld logits differ, max |d| %.3g, max rel %.3g, rows "
                           "with another top-%lld set %ld (rows with a NaN skipped)",
                           nd, (long long)(M * NE), md, mr, (long long)TK, rows_set)
                           .c_str());
        }
    }
    /* (b) under the switch */
    pin(-1, -1);
    sw("ds41_f32dec", VAL);
    {
        static const char* const kOut[6] = { "logits", "ids", "weights", "sorted_tok", "expert_offset", "expert_count" };
        std::string fails;
        int nb = 0;
        for (int64_t M : { 1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 24, 32, 33, 40, 47, 48, 49, 56, 63, 64 }) {
            for (int lay : { 0, 1 }) {
                const int64_t NL = lay ? NE / 4 : NE, base = lay ? NE / 4 : 0;   /* EP slice: 96 @ 96 at 384, 32 @ 32 at 128 */
                const std::array<size_t, 6> sz = { (size_t)(M * NE * 4) + tail, (size_t)(M * TK * 4) + tail, (size_t)(M * TK * 4) + tail,
                                                   (size_t)(M * TK * 4) + tail, (size_t)(NL + 1) * 4 + tail, (size_t)NL * 4 + tail };
                void* o[2][6];
                bool okm = true;
                for (int f = 0; f < 2; ++f)
                    for (int j = 0; j < 6; ++j) { o[f][j] = dalloc(sz[(size_t)j]); if (o[f][j]) st_memset(o[f][j], 0xA5, sz[(size_t)j]); else okm = false; }
                if (!okm) { fails += " alloc;"; break; }
                std::vector<RadParam> psel = { PI("M", M), PI("n_expert", NE), PI("top_k", TK), PI("norm", 1),
                                               PF("route_scale", scale), PS("score", "sqrtsoftplus"), PS("dtype", "f32") };
                std::vector<RadParam> psort = { PI("M", M), PI("n_expert", NL), PI("top_k", TK), PI("expert_base", base) };
                std::vector<RadParam> pf2 = { PI("M", M), PI("K", K), PI("n_expert", NE), PI("n_local", NL), PI("top_k", TK),
                                              PI("norm", 1), PF("route_scale", scale), PS("score", "sqrtsoftplus"),
                                              PS("dtype", "bf16"), PI("expert_base", base) };
                std::vector<RadTensor> tsel = { T(o[0][0], RAD_F32, { M, NE }), T(db, RAD_F32, { NE }),
                                                T(o[0][1], RAD_I32, { M, TK }), T(o[0][2], RAD_F32, { M, TK }) };
                std::vector<RadTensor> tsort = { T(o[0][1], RAD_I32, { M, TK }), T(o[0][3], RAD_I32, { M * TK }),
                                                 T(o[0][4], RAD_I32, { NL + 1 }), T(o[0][5], RAD_I32, { NL }) };
                std::vector<RadTensor> tf2 = { T(dx, RAD_BF16, { M, K }), T(dw, RAD_BF16, { NE, K }), T(db, RAD_F32, { NE }),
                                               T(o[1][0], RAD_F32, { M, NE }), T(o[1][1], RAD_I32, { M, TK }),
                                               T(o[1][2], RAD_F32, { M, TK }), T(o[1][3], RAD_I32, { M * TK }),
                                               T(o[1][4], RAD_I32, { NL + 1 }), T(o[1][5], RAD_I32, { NL }) };
                RadArgs asel = args(tsel, psel), asort = args(tsort, psort), af2 = args(tf2, pf2, inst);
                int c0 = gemm(dx, K, M, dw, NE, o[0][0]);
                if (c0 == RAD_OK) c0 = rs->launch(&asel, nullptr);
                if (c0 == RAD_OK) c0 = rc->launch(&asort, nullptr);
                const int c1 = f2->launch(&af2, nullptr), c2 = f2->launch(&af2, nullptr);   /* twice: the counters reset */
                st_stream_sync(nullptr);
                int first = -1;
                for (int j = 0; c0 == RAD_OK && c1 == RAD_OK && c2 == RAD_OK && j < 6 && first < 0; ++j)
                    if (bytes(o[0][j], sz[(size_t)j]) != bytes(o[1][j], sz[(size_t)j])) first = j;
                const std::string dl = c0 == RAD_OK ? same_rows(o[0][0], M, NE, 0) : std::string("chain failed");
                ++nb;
                if (c0 != RAD_OK || c1 != RAD_OK || c2 != RAD_OK || first >= 0 || !dl.empty())
                    fails += fmt(" M %lld %s: %s;", (long long)M, lay ? "EP slice" : "ETP",
                                 c0 != RAD_OK || c1 != RAD_OK || c2 != RAD_OK ? fmt("rc %d %d %d", c0, c1, c2).c_str()
                                 : first >= 0                                ? kOut[first]
                                                                             : ("chain logits vs (a): " + dl).c_str());
            }
        }
        ok(fmt("spdf32 rgm under ds41_f32dec %d (%lld experts): the fused launch (form 22, twice) = gemm_nt_f32 (code 40 by the rule) -> "
               "router_scored_topk -> moe_scatter, 6 outputs + tails, ETP and EP slice, M 1..64 (%d cases); the chain's logits = (a)'s rows",
               VAL, (long long)NE, nb).c_str(),
           fails.empty(), fails);
    }
    /* (c) SPDPERF */
    {
        void* o[6];
        const int64_t MM = 64;
        const std::array<size_t, 6> sz = { (size_t)(MM * NE * 4), (size_t)(MM * TK * 4), (size_t)(MM * TK * 4), (size_t)(MM * TK * 4),
                                           (size_t)(NE + 1) * 4, (size_t)NE * 4 };
        for (int j = 0; j < 6; ++j) o[j] = dalloc(sz[(size_t)j]);
        for (int64_t M : { 1, 8, 16, 24, 32, 40, 48, 64 }) {
            std::vector<RadParam> pf2 = { PI("M", M), PI("K", K), PI("n_expert", NE), PI("n_local", NE), PI("top_k", TK),
                                          PI("norm", 1), PF("route_scale", scale), PS("score", "sqrtsoftplus"),
                                          PS("dtype", "bf16"), PI("expert_base", 0) };
            std::vector<RadTensor> tf2 = { T(dx, RAD_BF16, { M, K }), T(dw, RAD_BF16, { NE, K }), T(db, RAD_F32, { NE }),
                                           T(o[0], RAD_F32, { M, NE }), T(o[1], RAD_I32, { M, TK }), T(o[2], RAD_F32, { M, TK }),
                                           T(o[3], RAD_I32, { M * TK }), T(o[4], RAD_I32, { NE + 1 }), T(o[5], RAD_I32, { NE }) };
            RadArgs af2 = args(tf2, pf2, inst);
            /* the chain (three launches: what a router WITHOUT the fused op declared runs -- the DSpark stages' today): gemm_nt_f32 ->
             * router_scored_topk -> moe_scatter */
            std::vector<RadParam> psel = { PI("M", M), PI("n_expert", NE), PI("top_k", TK), PI("norm", 1),
                                           PF("route_scale", scale), PS("score", "sqrtsoftplus"), PS("dtype", "f32") };
            std::vector<RadParam> psort = { PI("M", M), PI("n_expert", NE), PI("top_k", TK), PI("expert_base", 0) };
            std::vector<RadTensor> tsel = { T(o[0], RAD_F32, { M, NE }), T(db, RAD_F32, { NE }), T(o[1], RAD_I32, { M, TK }),
                                            T(o[2], RAD_F32, { M, TK }) };
            std::vector<RadTensor> tsort = { T(o[1], RAD_I32, { M, TK }), T(o[3], RAD_I32, { M * TK }), T(o[4], RAD_I32, { NE + 1 }),
                                             T(o[5], RAD_I32, { NE }) };
            RadArgs asel = args(tsel, psel), asort = args(tsort, psort);
            auto chain3 = [&]() -> int {
                int r = gemm(dx, K, M, dw, NE, o[0]);
                if (r == RAD_OK) r = rs->launch(&asel, nullptr);
                if (r == RAD_OK) r = rc->launch(&asort, nullptr);
                return r;
            };
            {
                double cw[2] = { 0, 0 }, cc2[2] = { 0, 0 };
                for (int rep = 0; rep < 2; ++rep)
                    for (int i = 0; i < 2; ++i) {
                        sw("ds41_f32dec", i == 0 ? 1 : VAL);
                        double ww = 0, cc = 0;
                        time2(chain3, 200, &ww, &cc);
                        cw[i] += ww / 2;
                        cc2[i] += cc / 2;
                    }
                std::printf("  SPDPERF rgm chain M=%lld (gemm_nt_f32 + router_scored_topk + moe_scatter, 3 launches) value 1 %7.2f / %7.2f | "
                            "value %d %7.2f / %7.2f us (warm / cold)\n", (long long)M, cw[0], cc2[0], VAL, cw[1], cc2[1]);
            }
            double w[3] = { 0, 0, 0 }, c[3] = { 0, 0, 0 };
            const int vals[3] = { VAL, 9, 1 };
            for (int rep = 0; rep < 2; ++rep)
                for (int i = 0; i < 3; ++i) {
                    sw("ds41_f32dec", vals[i]);
                    double ww = 0, cc = 0;
                    time2([&]() { return f2->launch(&af2, nullptr); }, 200, &ww, &cc);
                    std::printf("  SPDPERF rgm fused M=%lld ds41_f32dec %2d rep %d warm %7.2f cold %7.2f us\n", (long long)M, vals[i], rep, ww, cc);
                    w[i] += ww / 2;
                    c[i] += cc / 2;
                }
            std::printf("  SPDPERF rgm fused M=%lld mean of 2: value %d %7.2f / %7.2f | 9 (RGK) %7.2f / %7.2f | 1 (BEST) %7.2f / %7.2f us "
                        "(warm / cold); %d vs 9 cold %+6.2f us\n",
                        (long long)M, VAL, w[0], c[0], w[1], c[1], w[2], c[2], VAL, c[0] - c[1]);
        }
    }
    }   /* !po */
    {
        for (int64_t N : { (int64_t)384, (int64_t)96, (int64_t)128 })
            for (int64_t M : { (int64_t)64, (int64_t)128, (int64_t)256, (int64_t)1024, MR }) {
                if ((NE == 384) != (N != 128)) break;   /* 384 / 96 for the trunk's shape, 128 for the DSpark router's */
                void* y = dalloc((size_t)(M * N * 4));
                if (!y) break;
                double w[2] = { 0, 0 }, c[2] = { 0, 0 };
                for (int rep = 0; rep < 2; ++rep)
                    for (int i = 0; i < 2; ++i) {
                        sw("ds41_f32dec", i == 0 ? VAL : 1);
                        sw("f32_pf", i == 0 ? 0 : 25);
                        double ww = 0, cc = 0;
                        time2([&]() { return gemm(dx, K, M, dw, N, y); }, 100, &ww, &cc);
                        w[i] += ww / 2;
                        c[i] += cc / 2;
                    }
                std::printf("  SPDPERF rgm gemm N=%lld M=%lld: value %d (code 40) warm %8.2f cold %8.2f | value 1 + f32_pf 25 warm %8.2f cold "
                            "%8.2f us | %+7.2f / %+7.2f\n",
                            (long long)N, (long long)M, VAL, w[0], c[0], w[1], c[1], w[0] - w[1], c[0] - c[1]);
            }
        /* RGM_COMP=1 (measurement only, no serving rule takes it): the ratio > 1 compressor's fused kv | gate projection (N 1024, K 5120, bf16
         * under ROUTER_BF16) in code 40 (the pin) vs today's rule (value 1: the decode band's forms; f32_pf 25 from 64 rows); + one
         * batch-invariance line (rows of 64 = the 2112-row call's) */
        if (const char* ce = std::getenv("RGM_COMP"); ce && ce[0] == '1') {
            constexpr int64_t NC = 1024;
            std::vector<uint16_t> wc((size_t)(NC * K));
            for (auto& v : wc) v = bf((float)(gauss() * 0.02));
            void* dwc = up(wc);
            void* yb = dalloc((size_t)(MR * NC * 4));
            void* ys = dalloc((size_t)(64 * NC * 4));
            if (dwc && yb && ys) {
                pin(-1, 40);
                const int r1 = gemm(dx, K, MR, dwc, NC, yb), r2 = gemm(dx, K, 64, dwc, NC, ys);
                st_stream_sync(nullptr);
                const bool same = r1 == RAD_OK && r2 == RAD_OK && bytes(ys, (size_t)(64 * NC * 4)) == bytes(yb, (size_t)(64 * NC * 4));
                ok("spdf32 rgm compressor shape (N 1024): code 40 rows of 64 = the 2112-row call's bytes", same, fmt("rc %d %d", r1, r2));
                /* value 11 (lever RGC): the compressor shape takes code 40 by the RULE (no pin) at decode and prefill heights alike */
                {
                    pin(-1, -1);
                    sw("ds41_f32dec", 11);
                    void* y2 = dalloc((size_t)(MR * NC * 4));
                    void* y3 = dalloc((size_t)(64 * NC * 4));
                    const int r3 = y2 ? gemm(dx, K, MR, dwc, NC, y2) : -1, r4 = y3 ? gemm(dx, K, 64, dwc, NC, y3) : -1;
                    st_stream_sync(nullptr);
                    const bool s11 = r3 == RAD_OK && r4 == RAD_OK && bytes(y2, (size_t)(MR * NC * 4)) == bytes(yb, (size_t)(MR * NC * 4)) &&
                                     bytes(y3, (size_t)(64 * NC * 4)) == bytes(ys, (size_t)(64 * NC * 4));
                    ok("spdf32 rgm value 11: the compressor shape's 2112- and 64-row calls by the rule = code 40's (pin) bytes", s11,
                       fmt("rc %d %d", r3, r4));
                    sw("ds41_f32dec", 1);
                }
                for (int64_t M : { (int64_t)16, (int64_t)32, (int64_t)48, (int64_t)64, (int64_t)256, (int64_t)1024, MR }) {
                    double w[2] = { 0, 0 }, c[2] = { 0, 0 };
                    for (int rep = 0; rep < 2; ++rep)
                        for (int i = 0; i < 2; ++i) {
                            pin(-1, i == 0 ? 40 : -1);
                            sw("ds41_f32dec", 1);
                            sw("f32_pf", i == 0 ? 0 : 25);
                            double ww = 0, cc = 0;
                            time2([&]() { return gemm(dx, K, M, dwc, NC, yb); }, 100, &ww, &cc);
                            w[i] += ww / 2;
                            c[i] += cc / 2;
                        }
                    std::printf("  SPDPERF rgm compressor N=1024 M=%lld: code 40 warm %8.2f cold %8.2f | value 1 + f32_pf 25 warm %8.2f cold %8.2f us | "
                                "%+7.2f / %+7.2f\n",
                                (long long)M, w[0], c[0], w[1], c[1], w[0] - w[1], c[0] - c[1]);
                }
                pin(-1, -1);
            }
        }
        sw("f32_pf", 0);
    }
    sw("ds41_f32dec", 0);
    pin(-1, -1);
    dfree_all();
    f2->fini(inst);
}

static void spd_case(const Plugin& pl) {
    auto pin = (void (*)(int, int))dlsym(pl.h, "r4d_f32dec_pin");
    if (!pin) { ok("spdf32", false, "r4d_f32dec_pin absent (a plugin without lever SPD-F)"); return; }
    const char* part = std::getenv("SPD_PART");
    if (!part || !std::strcmp(part, "rgs") || !std::strcmp(part, "all")) rgs_part(pl, pin);
    if (!part || !std::strcmp(part, "gemm") || !std::strcmp(part, "all")) gemm_part(pl, pin);
    if (!part || !std::strcmp(part, "attn") || !std::strcmp(part, "all")) attn_part(pl);
    if (part && (!std::strcmp(part, "rgm") || !std::strcmp(part, "all"))) rgm_part(pl, pin);
    pin(-1, -1);
}

}  // namespace spd

static void spd_cases(const Plugin& pl) {
    if (g_case && sel("spdf32")) spd::spd_case(pl);
}
