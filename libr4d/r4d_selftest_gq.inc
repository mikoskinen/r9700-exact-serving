/* r4d_selftest_gq.inc -- lever 1U-GQ's byte checks on one card (ds41-int11; included by r4d_selftest.cpp before
 * device_phase, called from it):
 *
 *   --case gq_pair (= the rows gemm_nt_ggml_pair_qk256 / _qk32): gemm_nt_ggml_pair under lever D2's form (gg_gemv 2,
 *       one launch) against the two gemm_nt_ggml launches under gg_gemv 2 and under gg_gemv 0 (the stock decode kernel),
 *       and the pair under gg_gemv 0 (its own fallback: the two launches), byte for byte, at DeepSeek-V4.1's TP4 shapes
 *       -- q_a + attn_kv (1280 + 512 rows x 5120, one input, two outputs) and attention_output_a g0 + g1 (1024 + 1024 x
 *       4096: the inputs and outputs the column halves of [M, 8192] / [M, 2048], as the engine's o / oa) -- in q5_K, q6_K
 *       and q8_0 at M 1 .. 48 (the column halves' other words and the padding left as they were). Control: the pair with
 *       the second weight replaced by the first must differ. Times (one launch vs two) as info lines at M 1 / 2 / 3 / 6. */

namespace gqst {

static RadTensor TP(void* d, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch) {
    RadTensor t = T(d, dt, { rows, cols });
    t.stride[0] = pitch;
    return t;
}
static std::vector<uint8_t> dn(const void* d, size_t n) {
    std::vector<uint8_t> v(n);
    st_memcpy(v.data(), d, n, hipMemcpyDeviceToHost);
    return v;
}

static void gq_case(const Plugin& pl) {
    auto sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    if (!sw) { ok("gq_pair hooks", false, "rad_fn4_switch absent"); return; }
    struct Sh { const char* what; int64_t NA, NB, K; bool two_in; };
    const Sh shapes[] = { { "q_a + attn_kv", 1280, 512, 5120, false }, { "attention_output_a g0 + g1", 1024, 1024, 4096, true } };
    const int64_t Ms[] = { 1, 2, 3, 4, 6, 8, 12, 16, 24, 48 };
    for (const MrTy& ty : kMrTypes) {
        const bool q32 = !std::strcmp(ty.name, "q8_0");
        const RadKernelInfo* rp = find(pl, q32 ? "gemm_nt_ggml_pair_qk32" : "gemm_nt_ggml_pair_qk256");
        const RadKernelInfo* rg = find(pl, q32 ? "gemm_nt_ggml_qk32_tiled" : "gemm_nt_ggml_qk256_tiled");
        if (!rp || !rg) { ok(fmt("gq_pair rows (%s)", ty.name).c_str(), false, "a row absent"); continue; }
        for (const Sh& sh : shapes) {
            std::vector<uint8_t> hwa, hwb;
            mr_weights(hwa, ty, sh.NA, sh.K, 0x51D0001ull ^ (uint64_t)sh.NA);
            mr_weights(hwb, ty, sh.NB, sh.K, 0x51D0002ull ^ (uint64_t)sh.NB);
            const int64_t rb = sh.K / ty.qk * ty.bb;
            void *dwa = dalloc(hwa.size()), *dwb = dalloc(hwb.size());
            if (!dwa || !dwb) { ok("gq_pair alloc", false, "hipMalloc failed"); dfree_all(); return; }
            st_memcpy(dwa, hwa.data(), hwa.size(), hipMemcpyHostToDevice);
            st_memcpy(dwb, hwb.data(), hwb.size(), hipMemcpyHostToDevice);
            /* the input: [48, K] (one input) or [48, 2K] (two: the column halves) */
            const int64_t xw = sh.two_in ? 2 * sh.K : sh.K;
            std::vector<uint16_t> hx;
            mr_acts(hx, xw, 0xAC75ull ^ (uint64_t)xw);
            void* dx = dalloc(hx.size() * 2);
            st_memcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
            void* dx2 = sh.two_in ? (void*)((uint16_t*)dx + sh.K) : dx;
            /* the outputs: [48, NA + NB + 8] -- y at column 0, y2 at column NA, 8 padding columns */
            const int64_t yw = sh.NA + sh.NB + 8;
            const size_t yb = (size_t)48 * yw * 2;
            void *yR = dalloc(yb), *yS = dalloc(yb), *yP = dalloc(yb), *yF = dalloc(yb), *yX = dalloc(yb);
            if (!yX) { ok("gq_pair alloc", false, "hipMalloc failed"); dfree_all(); return; }
            for (int64_t M : Ms) {
                std::vector<RadParam> pp = { PI("M", M), PI("N", sh.NA), PI("N2", sh.NB), PI("K", sh.K), PS("type", ty.name),
                                             PS("dtype", "bf16") };
                std::vector<RadParam> pa = { PI("M", M), PI("N", sh.NA), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                std::vector<RadParam> pb = { PI("M", M), PI("N", sh.NB), PI("K", sh.K), PS("type", ty.name), PS("dtype", "bf16") };
                auto ins = [&](void* y, void* w2) {   /* the pair's operands */
                    std::vector<RadTensor> t = { TP(dx, RAD_BF16, M, sh.K, xw), T(dwa, RAD_U8, { sh.NA, rb }),
                                                 TP(dx2, RAD_BF16, M, sh.K, xw), T(w2, RAD_U8, { sh.NB, rb }),
                                                 TP(y, RAD_BF16, M, sh.NA, yw),
                                                 TP((uint16_t*)y + sh.NA, RAD_BF16, M, sh.NB, yw) };
                    if (!sh.two_in) absent(t[2]);
                    return t;
                };
                auto pair = [&](void* y, void* w2) {
                    st_memset(y, 0xA5, yb);
                    std::vector<RadTensor> t = ins(y, w2);
                    RadArgs a{};
                    a.t = t.data(); a.n_t = (int)t.size(); a.p = pp.data(); a.n_p = (int)pp.size(); a.world_size = 1;
                    const int rc = rp->launch(&a, nullptr);
                    st_stream_sync(nullptr);
                    return rc;
                };
                auto two = [&](void* y) {
                    st_memset(y, 0xA5, yb);
                    std::vector<RadTensor> ta = { TP(dx, RAD_BF16, M, sh.K, xw), T(dwa, RAD_U8, { sh.NA, rb }),
                                                  TP(y, RAD_BF16, M, sh.NA, yw) };
                    std::vector<RadTensor> tb = { TP(dx2, RAD_BF16, M, sh.K, xw), T(dwb, RAD_U8, { sh.NB, rb }),
                                                  TP((uint16_t*)y + sh.NA, RAD_BF16, M, sh.NB, yw) };
                    RadArgs a{}, b{};
                    a.t = ta.data(); a.n_t = 3; a.p = pa.data(); a.n_p = (int)pa.size(); a.world_size = 1;
                    b.t = tb.data(); b.n_t = 3; b.p = pb.data(); b.n_p = (int)pb.size(); b.world_size = 1;
                    const int rc = rg->launch(&a, nullptr) | rg->launch(&b, nullptr);
                    st_stream_sync(nullptr);
                    return rc;
                };
                sw("gg_gemv", 0);
                const int r0 = two(yS), r1 = pair(yF, dwb);
                sw("gg_gemv", 2);
                const int r2 = two(yR), r3 = pair(yP, dwb);
                const std::vector<uint8_t> S = dn(yS, yb), R = dn(yR, yb), P = dn(yP, yb), F = dn(yF, yb);
                auto nd = [&](const std::vector<uint8_t>& u, const std::vector<uint8_t>& v) {
                    long long d = 0;
                    for (size_t i = 0; i < u.size(); i += 2) d += std::memcmp(&u[i], &v[i], 2) != 0;
                    return d;
                };
                const long long dP = nd(P, S), dR = nd(R, S), dF = nd(F, S);
                ok(fmt("gemm_nt_ggml_pair %s %s %lld+%lldx%lld M=%lld: one launch (gg_gemv 2) == two launches (gg_gemv 2) "
                       "== stock (gg_gemv 0) == the pair's fallback (gg_gemv 0)", sh.what, ty.name, (long long)sh.NA,
                       (long long)sh.NB, (long long)sh.K, (long long)M).c_str(),
                   r0 == RAD_OK && r1 == RAD_OK && r2 == RAD_OK && r3 == RAD_OK && dP + dR + dF == 0,
                   fmt("rc %d %d %d %d; differing words vs stock: pair %lld, two %lld, fallback %lld", r0, r1, r2, r3, dP,
                       dR, dF));
                if (M == 3) {   /* the control */
                    pair(yX, dwa);
                    const long long dX = nd(dn(yX, yb), S);
                    ok(fmt("gq_pair control %s %s M=3: the second weight replaced by the first is caught", sh.what,
                           ty.name).c_str(), dX > 0, "identical: the check would not see the second call's rows");
                }
                if (M == 1 || M == 2 || M == 3 || M == 6) {
                    std::vector<RadTensor> t = ins(yP, dwb);
                    RadArgs a{};
                    a.t = t.data(); a.n_t = (int)t.size(); a.p = pp.data(); a.n_p = (int)pp.size(); a.world_size = 1;
                    std::vector<RadTensor> ta = { TP(dx, RAD_BF16, M, sh.K, xw), T(dwa, RAD_U8, { sh.NA, rb }),
                                                  TP(yR, RAD_BF16, M, sh.NA, yw) };
                    std::vector<RadTensor> tb = { TP(dx2, RAD_BF16, M, sh.K, xw), T(dwb, RAD_U8, { sh.NB, rb }),
                                                  TP((uint16_t*)yR + sh.NA, RAD_BF16, M, sh.NB, yw) };
                    RadArgs ga{}, gb{};
                    ga.t = ta.data(); ga.n_t = 3; ga.p = pa.data(); ga.n_p = (int)pa.size(); ga.world_size = 1;
                    gb.t = tb.data(); gb.n_t = 3; gb.p = pb.data(); gb.n_p = (int)pb.size(); gb.world_size = 1;
                    const double u2 = st_time_us([&](int) { rg->launch(&ga, nullptr); rg->launch(&gb, nullptr); }, 200, nullptr);
                    const double u1 = st_time_us([&](int) { rp->launch(&a, nullptr); }, 200, nullptr);
                    std::printf("  info  gemm_nt_ggml_pair %s %s M=%lld: two launches %.2f us, one %.2f us (%+.1f %%)\n",
                                sh.what, ty.name, (long long)M, u2, u1, 100.0 * (u1 - u2) / u2);
                }
                sw("gg_gemv", 0);
            }
            dfree_all();
        }
    }
}

}  // namespace gqst

static void gq_cases(const Plugin& pl) {
    if (sel("gq_pair") || sel("gemm_nt_ggml_pair_qk256") || sel("gemm_nt_ggml_pair_qk32")) gqst::gq_case(pl);
}
