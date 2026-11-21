/* tools/dbench/dqr.cpp (DRAFTER track 21.11.2026 round 69, lever DQR) -- one card, through libr4d's own exported entries
 * (dlopen'd): the DSpark stage's query rope, attention and output rope / round trip (deepseek41.cpp dsp_query: q_rope ->
 * attn (lever DAW: wm_from 1, the WMMA form) -> o_rope + o_q, or DOF's o_f) vs ONE attn_sparse_sink_dqr (libr4d round 69):
 *
 *  A  bytes: chain 1 = ds41_rope_rows (forward, in place on a copy of q) -> attn_sparse_sink -> ds41_rope_rows (inverse) ->
 *     quant_roundtrip (BEST's MHR=4 form); chain 2 = the same with rope_rows_roundtrip (DOF, both ROPE8 forms) for the last
 *     two; vs attn_sparse_sink_dqr on the UNROTATED q: out byte-compared at the engine's row shapes (sequences x block rows),
 *     three input kinds (typical; per-row magnitudes 2^-20..2^20; NaN / inf / -0.0 sprinkled into q and the keys, some rows'
 *     positions off the rope table), at the attention's tile skip 0 / 1 / 3 (RADIANCE_DS41_ATTN_SKIP); q unchanged after it.
 *  C  batch invariance: each sequence alone (a pass of its own rows) vs inside the full pass, fused form.
 *  B  time: K issues back to back between two events (each launch behind the one before, as a recorded pass orders them),
 *     median of ITERS: chain 1, chain 2, the fused op, and the attention alone (the HIP launch path: the served gaps are
 *     ~4.2 us a dependent launch on top).
 * Usage: dqr LIBR4D [ITERS] [K] */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

using Entry = int (*)(const RadArgs*, RadStream);
using Pin = void (*)(int);

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static uint64_t g_rs = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return (uint32_t)(g_rs >> 16); }
static float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }
static uint16_t h_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static RadTensor NONE() { RadTensor t; std::memset(&t, 0, sizeof(t)); return t; }
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadParam PF(const char* k, double v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
struct Call {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
    void fix() { std::memset(&a, 0, sizeof(a)); a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; a.wgp_count = 32; }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dqr LIBR4D [ITERS] [K]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const int K = argc > 3 ? std::atoi(argv[3]) : 20;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto attn = (Entry)dlsym(lib, "r4d_rad_ds41_attn_sparse_sink");
    auto rope = (Entry)dlsym(lib, "r4d_rad_ds41_rope_rows");
    auto qrt = (Entry)dlsym(lib, "r4d_rad_quant_roundtrip");
    auto ort = (Entry)dlsym(lib, "r4d_rad_rope_rows_roundtrip");
    auto dqr = (Entry)dlsym(lib, "r4d_rad_ds41_attn_sparse_sink_dqr");
    auto skip_pin = (Pin)dlsym(lib, "r4d_attn_skip_pin");
    auto rope8_pin = (Pin)dlsym(lib, "r4d_rope8_pin");
    if (!attn || !rope || !qrt || !ort || !dqr || !skip_pin || !rope8_pin) {
        std::fprintf(stderr, "symbols missing (%p %p %p %p %p %p %p)\n", (void*)attn, (void*)rope, (void*)qrt, (void*)ort, (void*)dqr,
                     (void*)skip_pin, (void*)rope8_pin);
        return 2;
    }
    std::printf("dqr: %s, iters %d, K %d\n", argv[1], iters, K);
    const int H = 16, D = 512, ROT = 64, OFF = D - ROT, HD = H * D, BS = 64, WIN0 = 128;
    const int MAXM = 96, MAXS = 32, CTXMAX = 4096, TAB = CTXMAX + 64;
    const int nwbt = (CTXMAX + 8 + BS - 1) / BS, nblk = MAXS * nwbt + 8;
    hipStream_t st; CK(hipStreamCreate(&st));
    const RadStream rs = (RadStream)st;
    void *dq0, *dq1, *dc, *dbt, *dwu, *dpos, *dqpos, *dcu, *dsink, *dcs, *dA, *dB, *dC, *dS;
    const size_t qb = (size_t)MAXM * HD * 2;
    CK(hipMalloc(&dq0, qb)); CK(hipMalloc(&dq1, qb)); CK(hipMalloc(&dA, qb)); CK(hipMalloc(&dB, qb)); CK(hipMalloc(&dC, qb)); CK(hipMalloc(&dS, qb));
    CK(hipMalloc(&dc, (size_t)nblk * BS * D * 2)); CK(hipMalloc(&dbt, (size_t)MAXS * nwbt * 4)); CK(hipMalloc(&dwu, MAXS * 4));
    CK(hipMalloc(&dpos, MAXM * 4)); CK(hipMalloc(&dqpos, MAXM * 4)); CK(hipMalloc(&dcu, (MAXS + 1) * 4)); CK(hipMalloc(&dsink, H * 4));
    CK(hipMalloc(&dcs, (size_t)TAB * ROT * 4));
    {   /* the rope table (cos | sin a row, rope_table's plane) and the sinks */
        std::vector<float> cs((size_t)TAB * ROT), sk(H);
        for (int p = 0; p < TAB; ++p)
            for (int i = 0; i < ROT / 2; ++i) {
                const double th = p * std::pow(10000.0, -2.0 * i / ROT);
                cs[(size_t)p * ROT + i] = (float)std::cos(th);
                cs[(size_t)p * ROT + ROT / 2 + i] = (float)std::sin(th);
            }
        for (auto& v : sk) v = uni(-1.0f, 1.0f);
        CK(hipMemcpy(dcs, cs.data(), cs.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dsink, sk.data(), sk.size() * 4, hipMemcpyHostToDevice));
    }
    /* the cache: random keys in randomly placed blocks, the same for every case */
    std::vector<int32_t> perm(nblk);
    for (int i = 0; i < nblk; ++i) perm[i] = i;
    for (int i = nblk - 1; i > 0; --i) std::swap(perm[i], perm[(int)(r32() % (uint32_t)(i + 1))]);
    std::vector<int32_t> bt((size_t)MAXS * nwbt);
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = perm[i];
    CK(hipMemcpy(dbt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice));
    auto fill_cache = [&](int kind) {
        std::vector<uint16_t> hc((size_t)nblk * BS * D);
        for (auto& v : hc) v = h_bf16(uni(-1.0f, 1.0f));
        if (kind == 2) {
            const uint16_t sp[] = { 0x8000, 0x7f80, 0xff80, 0x0001 };   /* -0.0, inf, -inf, a subnormal (no NaN keys: a NaN row) */
            for (size_t i = 0; i < hc.size(); i += 4099) hc[i] = sp[r32() % 4];
        }
        CK(hipMemcpy(dc, hc.data(), hc.size() * 2, hipMemcpyHostToDevice));
    };
    struct Shape { int S, B; };
    const Shape shapes[] = { { 1, 5 }, { 3, 5 }, { 8, 2 }, { 8, 3 }, { 16, 2 }, { 8, 5 }, { 24, 2 }, { 16, 3 }, { 12, 4 }, { 16, 4 }, { 16, 5 }, { 32, 3 } };
    /* a case: S sequences of B rows (q random of `kind`), contexts drawn, apos = each row at its block's last position, qpos = the row's own */
    std::vector<int32_t> ctx(MAXS);
    auto set_case = [&](int S, int B, int kind) {
        const int M = S * B;
        std::vector<uint16_t> hq((size_t)M * HD);
        for (int m = 0; m < M; ++m) {
            const float sc = kind == 1 ? std::ldexp(1.0f, (int)(r32() % 41) - 20) : 1.0f;
            for (int i = 0; i < HD; ++i) hq[(size_t)m * HD + i] = h_bf16(uni(-1.0f, 1.0f) * sc);
        }
        if (kind == 2) {
            const uint16_t sp[] = { 0x8000, 0x0000, 0x7f80, 0xff80, 0x7fc1, 0x0001, 0x8001, 0x7f7f };
            for (size_t i = 0; i < hq.size(); i += 211) hq[i] = sp[r32() % 8];
            for (int m = 1; m < M; m += 9) hq[(size_t)m * HD + OFF + 3] = 0x7fc1;   /* a NaN in a rotated column */
        }
        std::vector<int32_t> wu(S), pos(M), qpos(M), cu(S + 1);
        for (int s = 0; s < S; ++s) {
            ctx[s] = 64 + (int)(r32() % (uint32_t)(CTXMAX - 200));
            wu[s] = ctx[s] + B;
            cu[s] = s * B;
            for (int r = 0; r < B; ++r) {
                pos[(size_t)s * B + r] = ctx[s] + B - 1;
                qpos[(size_t)s * B + r] = ctx[s] + r;
            }
        }
        cu[S] = M;
        if (kind == 2)   /* rows whose own position is off the rope table: both ropes skip them */
            for (int m = 2; m < M; m += 7) qpos[m] = TAB + 3;
        CK(hipMemcpy(dq0, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dwu, wu.data(), wu.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dpos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dqpos, qpos.data(), qpos.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dcu, cu.data(), cu.size() * 4, hipMemcpyHostToDevice));
    };
    /* the ops at M rows of S sequences, block B (window = 128 + B), q from qsrc, out into dout; positions / cu at offsets */
    auto mk_attn = [&](Call& c, int S, int B, void* qsrc, void* dout, int s0, bool fused, int orp = 1) {
        const int M = S * B;
        c.t = { T((char*)qsrc + (size_t)s0 * B * HD * 2, RAD_BF16, { M, HD }), T(dc, RAD_BF16, { nblk, 1, BS, D }),
                T((char*)dbt + (size_t)s0 * nwbt * 4, RAD_I32, { S, nwbt }), T((char*)dwu + (size_t)s0 * 4, RAD_I32, { S }), NONE(), NONE(),
                NONE(), T((char*)dpos + (size_t)s0 * B * 4, RAD_I32, { M }), T(dcu, RAD_I32, { S + 1 }), T(dsink, RAD_F32, { H }),
                T(dout, RAD_BF16, { M, HD }) };
        c.p = { PI("M", M), PI("heads", H), PI("dim", D), PI("window", WIN0 + B), PI("ratio", 0), PI("topk", 0),
                PF("scale", 1.0 / std::sqrt((double)D)), PS("dtype", "bf16"), PI("wm_from", 1) };
        if (fused) {
            c.t.push_back(T(dcs, RAD_F32, { TAB, ROT }));
            c.t.push_back(T((char*)dqpos + (size_t)s0 * B * 4, RAD_I32, { M }));
            for (auto q : { PI("rot", ROT), PI("off", OFF), PI("block", 32) }) c.p.push_back(q);
            c.p.push_back(PS("mode", "gptj")); c.p.push_back(PS("format", "fp8_e4m3")); c.p.push_back(PS("block_scale", "ue8m0"));
            c.p.push_back(PI("out_rope", orp));
        }
        c.fix();
    };
    auto mk_rope = [&](Call& c, int M, void* x, int inv) {
        c.t = { T(x, RAD_BF16, { M, HD }), T(dcs, RAD_F32, { TAB, ROT }), T(dqpos, RAD_I32, { M }) };
        c.p = { PI("M", M), PI("n_head", H), PI("head_dim", D), PI("rot", ROT), PI("off", OFF), PS("mode", "gptj"), PI("inverse", inv),
                PS("dtype", "bf16") };
        c.fix();
    };
    auto mk_qrt = [&](Call& c, int M, void* x) {
        c.t = { T(x, RAD_BF16, { M, HD }) };
        c.p = { PI("M", M), PI("n", HD), PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"), PS("dtype", "bf16") };
        c.fix();
    };
    auto mk_ort = [&](Call& c, int M, void* x) {
        c.t = { T(x, RAD_BF16, { M, HD }), T(dcs, RAD_F32, { TAB, ROT }), T(dqpos, RAD_I32, { M }) };
        c.p = { PI("M", M), PI("n_head", H), PI("head_dim", D), PI("rot", ROT), PI("off", OFF), PS("mode", "gptj"), PI("inverse", 1),
                PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"), PS("dtype", "bf16") };
        c.fix();
    };
    auto run = [&](Call& c, Entry f, const char* w) {
        const int rc = f(&c.a, rs);
        if (rc != 0) { std::fprintf(stderr, "%s rc %d\n", w, rc); std::exit(3); }
    };
    auto get = [&](void* d, size_t n) { std::vector<uint16_t> v(n); CK(hipStreamSynchronize(st)); CK(hipMemcpy(v.data(), d, n * 2, hipMemcpyDeviceToHost)); return v; };
    auto ndiff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long k = 0; for (size_t i = 0; i < a.size(); ++i) k += a[i] != b[i]; return k; };

    /* ---- A + C: bytes */
    long fails = 0, cmp = 0;
    const int skps[] = { 1, 0, 3 };
    for (int kind = 0; kind < 3; ++kind) {
        fill_cache(kind);
        for (const Shape& sh : shapes) {
            const int S = sh.S, B = sh.B, M = S * B;
            set_case(S, B, kind);
            const size_t n = (size_t)M * HD;
            const std::vector<uint16_t> q0 = get(dq0, n);
            for (int sk : skps) {
                skip_pin(sk);
                /* chain 1: q copy -> rope (forward) -> attn -> rope (inverse) -> quant_roundtrip */
                CK(hipMemcpyAsync(dq1, dq0, n * 2, hipMemcpyDeviceToDevice, st));
                Call cr, ca, co, cq, cf, ct;
                mk_rope(cr, M, dq1, 0); mk_attn(ca, S, B, dq1, dA, 0, false); mk_rope(co, M, dA, 1); mk_qrt(cq, M, dA);
                run(cr, rope, "rope q"); run(ca, attn, "attn"); run(co, rope, "o_rope"); run(cq, qrt, "o_q");
                const std::vector<uint16_t> a1 = get(dA, n);
                /* chain 2: the same attention output -> rope_rows_roundtrip (DOF) at both ROPE8 forms */
                Call ca2; mk_attn(ca2, S, B, dq1, dC, 0, false);
                long d2[2];
                for (int r8 = 0; r8 < 2; ++r8) {
                    rope8_pin(r8);
                    run(ca2, attn, "attn"); mk_ort(ct, M, dC); run(ct, ort, "o_f");
                    d2[r8] = ndiff(get(dC, n), a1);
                }
                rope8_pin(-1);
                /* q-only form (out_rope 0) vs rope q + attn (the attention's own output) */
                Call cq0; mk_attn(cq0, S, B, dq0, dS, 0, true, 0); run(cq0, dqr, "dqr q");
                Call ca3; mk_attn(ca3, S, B, dq1, dC, 0, false); run(ca3, attn, "attn");
                const long dqo = ndiff(get(dS, n), get(dC, n));
                /* fused, on the unrotated q */
                mk_attn(cf, S, B, dq0, dB, 0, true);
                run(cf, dqr, "dqr");
                const std::vector<uint16_t> b = get(dB, n);
                const long d = ndiff(b, a1), dq = ndiff(get(dq0, n), q0);
                /* C: each sequence alone (its own pass) vs inside the full pass */
                long dc_ = 0;
                for (int s = 0; s < S; s += (S > 4 ? S / 4 : 1)) {
                    std::vector<int32_t> cu1 = { 0, B };
                    void* dcu1; CK(hipMalloc(&dcu1, 8)); CK(hipMemcpy(dcu1, cu1.data(), 8, hipMemcpyHostToDevice));
                    Call c1; mk_attn(c1, 1, B, dq0, dS, s, true);
                    c1.t[8] = T(dcu1, RAD_I32, { 2 }); c1.fix();
                    run(c1, dqr, "dqr alone");
                    const std::vector<uint16_t> one = get(dS, (size_t)B * HD);
                    for (size_t i = 0; i < one.size(); ++i) dc_ += one[i] != b[(size_t)s * B * HD + i];
                    CK(hipFree(dcu1));
                }
                cmp += (long)n;
                const bool bad = d || d2[0] || d2[1] || dq || dc_ || dqo;
                fails += bad;
                std::printf("  kind %d %2d x %d = %2d rows skp %d: fused vs chain 1 %ld of %zu differ; q-only form vs rope q + attn %ld; o_f vs o_rope + o_q "
                            "%ld / %ld (ROPE8 0 / 1); q after the fused op %ld differ; alone vs in the pass %ld%s\n",
                            kind, S, B, M, sk, d, n, dqo, d2[0], d2[1], dq, dc_, bad ? "  <-- DIFFER" : "");
            }
            skip_pin(-1);
        }
    }
    std::printf("A/C: %ld of %d cells differ (%ld values compared a form)\n", fails, (int)(3 * sizeof(shapes) / sizeof(shapes[0]) * 3), cmp);

    /* ---- B: time, K issues back to back, median of iters (BEST's skip 1, ROPE8 1) */
    skip_pin(1); rope8_pin(1);
    fill_cache(0);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](auto body) {
        std::vector<float> v;
        for (int it = 0; it < iters + 3; ++it) {
            CK(hipEventRecord(e0, st));
            for (int k = 0; k < K; ++k) body();
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (it >= 3) v.push_back(ms * 1000.0f / K);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::printf("time (us an issue, median of %d x %d back to back; skip 1, ROPE8 1)\n", iters, K);
    std::printf("  %-8s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "rows", "attn", "rq+attn", "chain1", "chain2", "fused", "fused-c2", "q-form", "q-rq-at", "q+o_f-c2");
    const Shape tsh[] = { { 8, 2 }, { 8, 3 }, { 16, 2 }, { 8, 5 }, { 24, 2 }, { 16, 3 }, { 16, 4 }, { 16, 5 } };
    for (int rep = 0; rep < 2; ++rep)
        for (const Shape& sh : tsh) {
            const int S = sh.S, B = sh.B, M = S * B;
            set_case(S, B, 0);
            CK(hipMemcpy(dq1, dq0, (size_t)M * HD * 2, hipMemcpyDeviceToDevice));
            Call cr, ca, co, cq, ct, cf;
            mk_rope(cr, M, dq1, 0); mk_attn(ca, S, B, dq1, dA, 0, false); mk_rope(co, M, dA, 1); mk_qrt(cq, M, dA); mk_ort(ct, M, dA);
            mk_attn(cf, S, B, dq0, dB, 0, true);
            Call cg; mk_attn(cg, S, B, dq0, dA, 0, true, 0);
            for (int w = 0; w < 100; ++w) { run(cf, dqr, "warm"); run(cg, dqr, "warm"); }
            const float ta = timed([&]() { run(ca, attn, "attn"); });
            const float tr = timed([&]() { run(cr, rope, "r"); run(ca, attn, "a"); });
            const float t1 = timed([&]() { run(cr, rope, "r"); run(ca, attn, "a"); run(co, rope, "o"); run(cq, qrt, "q"); });
            const float t2 = timed([&]() { run(cr, rope, "r"); run(ca, attn, "a"); run(ct, ort, "f"); });
            const float tf = timed([&]() { run(cf, dqr, "dqr"); });
            const float tg = timed([&]() { run(cg, dqr, "dqr q"); });
            const float tgf = timed([&]() { run(cg, dqr, "dqr q"); run(ct, ort, "f"); });
            char nm[32]; std::snprintf(nm, sizeof nm, "%dx%d=%d", S, B, M);
            std::printf("  %-8s %9.2f %9.2f %9.2f %9.2f %9.2f %+9.2f %9.2f %+9.2f %+9.2f\n", nm, ta, tr, t1, t2, tf, tf - t2, tg, tg - tr, tgf - t2);
        }
    {   /* host enqueue cost a call (no sync inside), and the fused op at each tile skip: is the fused time the kernel's? */
        set_case(16, 2, 0);
        Call ca, cf; mk_attn(ca, 16, 2, dq1, dA, 0, false); mk_attn(cf, 16, 2, dq0, dB, 0, true);
        auto host = [&](Call& c, Entry f) {
            CK(hipStreamSynchronize(st));
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 200; ++k) run(c, f, "h");
            auto t1 = std::chrono::steady_clock::now();
            CK(hipStreamSynchronize(st));
            auto t2 = std::chrono::steady_clock::now();
            return std::make_pair(std::chrono::duration<double, std::micro>(t1 - t0).count() / 200,
                                  std::chrono::duration<double, std::micro>(t2 - t0).count() / 200);
        };
        for (int rep = 0; rep < 2; ++rep) {
            auto ha = host(ca, attn), hf = host(cf, dqr);
            std::printf("  host 16x2: attn enqueue %.2f / wall %.2f us a call; fused enqueue %.2f / wall %.2f\n", ha.first, ha.second, hf.first, hf.second);
        }
        for (int sk : { 0, 1, 3 }) {
            skip_pin(sk);
            const float ta = timed([&]() { run(ca, attn, "attn"); });
            const float tf = timed([&]() { run(cf, dqr, "dqr"); });
            std::printf("  skp %d 16x2: attn %.2f fused %.2f us\n", sk, ta, tf);
        }
        skip_pin(1);
        if (auto abl = (Pin)dlsym(lib, "r4d_dqr_abl_pin")) {   /* where the fused op's extra time goes (bytes then differ) */
            for (int v : { 0, 1, 2, 3, 0 }) {
                abl(v);
                const float tf = timed([&]() { run(cf, dqr, "dqr"); });
                std::printf("  ablation %d (bit 0: no q rope, bit 1: no output rope / round trip) 16x2: fused %.2f us\n", v, tf);
            }
            abl(0);
        }
        for (int k1 : { 1, 4 }) {
            std::vector<float> v;
            for (int it = 0; it < 100; ++it) {
                CK(hipEventRecord(e0, st)); for (int k = 0; k < k1; ++k) run(cf, dqr, "d"); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); v.push_back(ms * 1000.0f / k1);
            }
            std::sort(v.begin(), v.end());
            std::printf("  fused K %d: %.2f us an issue (p10 %.2f)\n", k1, v[50], v[10]);
        }
    }
    skip_pin(-1); rope8_pin(-1);
    return fails ? 1 : 0;
}
