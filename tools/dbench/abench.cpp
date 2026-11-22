/* abench (DRAFTER track, tp8 loop 18.11.2026 round 5) -- one card, the DSpark drafter stage's attention chain at a query
 * pass's rows, through libr4d's own entries (compiled from the branch's r4d_ds41_sparse.hip / r4d_ds41_elem.hip with the
 * plugin's flags):
 *   attn   attn_sparse_sink over the drafter's window (window 128 + B keys, B = rows a sequence; 16 heads a rank, the
 *          512-wide MQA latent, bf16 window cache, sinks), the rule and every decode-band form 0-6 pinned
 *          (r4d_attn_dec_pin; every form must give the rule's bytes)
 *   orope  ds41_rope_rows inverse over the output (o_rope), then
 *   oq     quant_roundtrip fp8_e4m3 / ue8m0 / 32 over the output (o_q) -- the served pair (deepseek41.cpp dsp_stage_ops)
 *   ort    rope_rows_roundtrip (the pair as one launch, the trunk's L.o_f) -- bytes vs orope + oq
 *   fold   attn_sparse_sink_rope_roundtrip (lever K1 fold d: the decode band's kernel with the pair in its epilogue) --
 *          bytes vs attn + orope + oq
 * Each op is timed by events around it inside a chain issued as serving issues it (attn -> orope -> oq), median over
 * `iters` chains; --cold MiB writes that many MiB between chains (the window cache then comes from DRAM).
 *
 * usage: abench [--rows 16] [--bpr 1] [--ctx 3000] [--bs 64] [--iters 300] [--cold 0] [--forms 0,1,2,3,4,5,6] */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;
/* Not reached on these paths (mhc_mix's FIN counters, SPD-F's router form): link stubs. */
unsigned int* r4d_8a8_split_ctr(RadStream, const void*) { return nullptr; }
int r4d_f32dec_rgs_form(int) { return 0; }

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs* a, RadStream s);
extern "C" void r4d_attn_dec_pin(int c);
extern "C" void r4d_qg8_pin(int v);
extern "C" void r4d_wq_pin(int v);
extern "C" void r4d_attn_wqa(int on);
extern "C" void r4d_attn_wm_from(int rows);   /* round 9 lever DAW: the WMMA form from that many rows (0: off) */
extern "C" int r4d_rad_ds41_rope_rows(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_quant_roundtrip(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs* a, RadStream s);

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

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

static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint64_t g_rs = 0x9e3779b97f4a7c15ull;
static float frand() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return (float)((g_rs >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

struct Call { std::vector<RadTensor> t; std::vector<RadParam> p; RadArgs a; void fix() { std::memset(&a, 0, sizeof(a)); a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; a.wgp_count = 32; } };

static std::vector<int> ints(const char* s) { std::vector<int> v; while (s && *s) { v.push_back(std::atoi(s)); s = std::strchr(s, ','); if (s) ++s; } return v; }

int main(int argc, char** argv) {
    int rows = 16, bpr = 1, ctx = 3000, bs = 64, iters = 300, coldmb = 0, kis = 20;
    std::vector<int> forms = { 0, 1, 2, 3, 4, 5, 6 };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)""; };
        if (k == "--rows") rows = std::atoi(nx());
        else if (k == "--bpr") bpr = std::atoi(nx());
        else if (k == "--ctx") ctx = std::atoi(nx());
        else if (k == "--bs") bs = std::atoi(nx());
        else if (k == "--iters") iters = std::atoi(nx());
        else if (k == "--cold") coldmb = std::atoi(nx());
        else if (k == "--forms") forms = ints(nx());
        else if (k == "--k") kis = std::atoi(nx());   /* round 7: issues between the two events (1 + --cold: one cold issue, as a served pass meets it) */
        else { std::fprintf(stderr, "unknown %s\n", k.c_str()); return 1; }
    }
    const int H = 16, D = 512, ROT = 64, OFF = D - ROT, HD = H * D, win = 128 + bpr;
    const int S = rows / bpr, M = S * bpr;
    const int used = ctx + bpr, nwbt = (used + bs - 1) / bs, nblk = S * nwbt + 8;
    const int maxctx = used + 64;
    std::printf("abench: rows %d (%d sequences x %d), heads %d x %d, window %d keys, ctx %d, cache blocks of %d, iters %d, cold %d MiB\n",
                M, S, bpr, H, D, win, ctx, bs, iters, coldmb);

    /* ---- inputs */
    std::vector<uint16_t> hq((size_t)M * HD), hc((size_t)nblk * bs * D);
    for (auto& v : hq) v = h_to_bf16(frand());
    for (auto& v : hc) v = h_to_bf16(frand());
    std::vector<int32_t> bt((size_t)S * nwbt), wu(S), pos(M), cu(S + 1);
    std::vector<int32_t> perm(nblk); for (int i = 0; i < nblk; ++i) perm[i] = i;
    for (int i = nblk - 1; i > 0; --i) { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; std::swap(perm[i], perm[(int)(g_rs % (uint64_t)(i + 1))]); }
    for (int s = 0; s < S; ++s) {
        for (int j = 0; j < nwbt; ++j) bt[(size_t)s * nwbt + j] = perm[(size_t)s * nwbt + j];
        wu[s] = used; cu[s] = s * bpr;
        for (int r = 0; r < bpr; ++r) pos[(size_t)s * bpr + r] = used - 1;   /* each row at its block's last position (apos) */
    }
    cu[S] = M;
    std::vector<float> sink(H), cs((size_t)maxctx * ROT);
    for (auto& v : sink) v = frand();
    for (int p = 0; p < maxctx; ++p)
        for (int i = 0; i < ROT / 2; ++i) {
            const double th = p * std::pow(10000.0, -2.0 * i / ROT);
            cs[(size_t)p * ROT + i] = (float)std::cos(th);
            cs[(size_t)p * ROT + ROT / 2 + i] = (float)std::sin(th);
        }
    void *dq, *dc, *dbt, *dwu, *dpos, *dcu, *dsink, *dcs, *dout, *dref, *dtmp, *dcold = nullptr;
    const size_t ob = (size_t)M * HD * 2;
    CK(hipMalloc(&dq, ob)); CK(hipMalloc(&dc, hc.size() * 2)); CK(hipMalloc(&dbt, bt.size() * 4)); CK(hipMalloc(&dwu, wu.size() * 4));
    CK(hipMalloc(&dpos, pos.size() * 4)); CK(hipMalloc(&dcu, cu.size() * 4)); CK(hipMalloc(&dsink, sink.size() * 4));
    CK(hipMalloc(&dcs, cs.size() * 4)); CK(hipMalloc(&dout, ob)); CK(hipMalloc(&dref, ob)); CK(hipMalloc(&dtmp, ob));
    if (coldmb > 0) CK(hipMalloc(&dcold, (size_t)coldmb << 20));
    CK(hipMemcpy(dq, hq.data(), ob, hipMemcpyHostToDevice)); CK(hipMemcpy(dc, hc.data(), hc.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dbt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dwu, wu.data(), wu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dpos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcu, cu.data(), cu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsink, sink.data(), sink.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcs, cs.data(), cs.size() * 4, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    const RadStream rs = (RadStream)st;

    /* ---- the calls */
    auto attn_ts = [&](void* out) {
        return std::vector<RadTensor>{ T(dq, RAD_BF16, { M, HD }), T(dc, RAD_BF16, { nblk, 1, bs, D }), T(dbt, RAD_I32, { S, nwbt }),
                                       T(dwu, RAD_I32, { S }), NONE(), NONE(), NONE(), T(dpos, RAD_I32, { M }), T(dcu, RAD_I32, { S + 1 }),
                                       T(dsink, RAD_F32, { H }), T(out, RAD_BF16, { M, HD }) };
    };
    const std::vector<RadParam> ap = { PI("M", M), PI("heads", H), PI("dim", D), PI("window", win), PI("ratio", 0), PI("topk", 0),
                                       PF("scale", 1.0 / std::sqrt((double)D)), PS("dtype", "bf16") };
    Call attn; attn.t = attn_ts(dout); attn.p = ap; attn.fix();
    Call fold; fold.t = attn_ts(dout); fold.t.push_back(T(dcs, RAD_F32, { maxctx, ROT })); fold.p = ap;
    for (auto q : { PI("rot", ROT), PI("off", OFF), PI("inverse", 1), PI("block", 32) }) fold.p.push_back(q);
    fold.p.push_back(PS("mode", "gptj")); fold.p.push_back(PS("format", "fp8_e4m3")); fold.p.push_back(PS("block_scale", "ue8m0")); fold.fix();
    Call orope; orope.t = { T(dout, RAD_BF16, { M, HD }), T(dcs, RAD_F32, { maxctx, ROT }), T(dpos, RAD_I32, { M }) };
    orope.p = { PI("M", M), PI("n_head", H), PI("head_dim", D), PI("rot", ROT), PI("off", OFF), PS("mode", "gptj"), PI("inverse", 1), PS("dtype", "bf16") }; orope.fix();
    Call oq; oq.t = { T(dout, RAD_BF16, { M, HD }) };
    oq.p = { PI("M", M), PI("n", HD), PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"), PS("dtype", "bf16") }; oq.fix();
    Call ort; ort.t = orope.t; ort.p = orope.p;
    for (auto q : { PI("block", 32) }) ort.p.push_back(q);
    ort.p.push_back(PS("format", "fp8_e4m3")); ort.p.push_back(PS("scale", "ue8m0")); ort.fix();

    auto run = [&](Call& c, int (*f)(const RadArgs*, RadStream)) {
        const int rc = f(&c.a, rs);
        if (rc != 0) { std::fprintf(stderr, "op rc %d\n", rc); std::exit(3); }
    };
    std::vector<uint16_t> hr(ob / 2), hx(ob / 2);
    auto grab = [&](std::vector<uint16_t>& h) { CK(hipStreamSynchronize(st)); CK(hipMemcpy(h.data(), dout, ob, hipMemcpyDeviceToHost)); };
    auto ndiff = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long n = 0; for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i]; return n; };

    /* ---- reference bytes: the rule's attention, then + orope + oq */
    r4d_attn_dec_pin(-1);
    run(attn, r4d_rad_ds41_attn_sparse_sink); grab(hr);
    std::vector<uint16_t> hat = hr;   /* the attention output */
    r4d_qg8_pin(0);
    run(orope, r4d_rad_ds41_rope_rows); run(oq, r4d_rad_quant_roundtrip); std::vector<uint16_t> hchain(ob / 2); grab(hchain);
    r4d_qg8_pin(-1);
    CK(hipMemcpy(dout, hat.data(), ob, hipMemcpyHostToDevice));
    run(orope, r4d_rad_ds41_rope_rows); run(oq, r4d_rad_quant_roundtrip); grab(hx);
    std::printf("  QG8 oq bytes vs the one-column kernel: %ld of %zu differ\n", ndiff(hx, hchain), hx.size());
    {   /* QG8 on hostile values: NaN, inf, subnormals, zeros, huge, a whole block of zeros */
        std::vector<uint16_t> hv(ob / 2);
        const uint16_t sp[] = { 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0001, 0x8001, 0x0000, 0x8000, 0x7f7f, 0xff7f, 0x0080, 0x3f80 };
        for (size_t i = 0; i < hv.size(); ++i) hv[i] = (i % 7 == 0) ? sp[(i / 7) % 12] : (i % 1024 < 32 ? 0 : h_to_bf16(frand() * ((i % 5) ? 1.0f : 1e-30f)));
        CK(hipMemcpy(dout, hv.data(), ob, hipMemcpyHostToDevice)); r4d_qg8_pin(0); run(oq, r4d_rad_quant_roundtrip); grab(hchain);
        CK(hipMemcpy(dout, hv.data(), ob, hipMemcpyHostToDevice)); r4d_qg8_pin(-1); run(oq, r4d_rad_quant_roundtrip); grab(hx);
        std::printf("  QG8 oq bytes vs the one-column kernel on special values: %ld differ\n", ndiff(hx, hchain));
        CK(hipMemcpy(dout, hat.data(), ob, hipMemcpyHostToDevice)); r4d_qg8_pin(0);
        run(orope, r4d_rad_ds41_rope_rows); run(oq, r4d_rad_quant_roundtrip); grab(hchain); r4d_qg8_pin(-1);
    }
    /* forms vs the rule */
    for (int f : forms) {
        r4d_attn_dec_pin(f); run(attn, r4d_rad_ds41_attn_sparse_sink); grab(hx);
        std::printf("  form %d attention bytes vs the rule: %ld of %zu differ\n", f, ndiff(hx, hat), hx.size());
        run(fold, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip); grab(hx);
        std::printf("  form %d fold d bytes vs attn + orope + oq: %ld differ\n", f, ndiff(hx, hchain));
    }
    {   /* round 9 lever DAW: the WMMA prefill form at these rows (rule B: not the rule's bits) -- values vs the rule, and its own determinism */
        r4d_attn_dec_pin(-1); r4d_attn_wm_from(1); run(attn, r4d_rad_ds41_attn_sparse_sink); grab(hx);
        std::vector<uint16_t> hw2(ob / 2); run(attn, r4d_rad_ds41_attn_sparse_sink); grab(hw2);
        auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; };
        double mx = 0, ma = 0, se = 0, sr = 0; long nn = 0;
        for (size_t i = 0; i < hx.size(); ++i) { const double r = bf(hat[i]), w = bf(hx[i]); mx = std::max(mx, std::fabs(w - r)); ma = std::max(ma, std::fabs(r)); se += (w - r) * (w - r); sr += r * r; nn += hx[i] != hat[i]; }
        std::printf("  DAW wmma form vs the rule: %ld of %zu bytes differ, max |d| %.3e (max |ref| %.3e), rms rel %.3e; repeat %ld differ\n", nn, hx.size(), mx, ma, std::sqrt(se / (sr > 0 ? sr : 1)), ndiff(hx, hw2));
        r4d_attn_wm_from(0);
        /* the engine's path: the op's own param wm_from 1 (deepseek41 under RADIANCE_DS41_DSPARK_AWM) -- the pin's bytes */
        Call aw; aw.t = attn_ts(dout); aw.p = ap; aw.p.push_back(PI("wm_from", 1)); aw.fix();
        run(aw, r4d_rad_ds41_attn_sparse_sink); grab(hw2);
        std::printf("  DAW op param wm_from 1 vs the pin: %ld differ\n", ndiff(hx, hw2));
        /* batch invariance: the first sequence's rows as a pass of their own (S 1) -- the same bytes as in the full pass */
        {
            const int M1 = bpr;
            std::vector<int32_t> cu1 = { 0, M1 };
            void* dcu1; CK(hipMalloc(&dcu1, 8)); CK(hipMemcpy(dcu1, cu1.data(), 8, hipMemcpyHostToDevice));
            Call a1; a1.t = { T(dq, RAD_BF16, { M1, HD }), T(dc, RAD_BF16, { nblk, 1, bs, D }), T(dbt, RAD_I32, { 1, nwbt }),
                              T(dwu, RAD_I32, { 1 }), NONE(), NONE(), NONE(), T(dpos, RAD_I32, { M1 }), T(dcu1, RAD_I32, { 2 }),
                              T(dsink, RAD_F32, { H }), T(dtmp, RAD_BF16, { M1, HD }) };
            a1.p = ap; a1.p[0] = PI("M", M1); a1.p.push_back(PI("wm_from", 1)); a1.fix();
            run(a1, r4d_rad_ds41_attn_sparse_sink); CK(hipStreamSynchronize(st));
            std::vector<uint16_t> h1((size_t)M1 * HD); CK(hipMemcpy(h1.data(), dtmp, h1.size() * 2, hipMemcpyDeviceToHost));
            long d1 = 0; for (size_t i = 0; i < h1.size(); ++i) d1 += h1[i] != hx[i];
            std::printf("  DAW batch invariance: sequence 0 alone (%d rows) vs in the %d-row pass: %ld of %zu differ\n", M1, M, d1, h1.size());
            CK(hipFree(dcu1));
        }
    }
    r4d_attn_dec_pin(-1);
    run(fold, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip); grab(hx);
    std::printf("  rule fold d bytes vs attn + orope + oq: %ld differ\n", ndiff(hx, hchain));
    CK(hipMemcpy(dout, hat.data(), ob, hipMemcpyHostToDevice));
    run(ort, r4d_rad_rope_rows_roundtrip); grab(hx);
    std::printf("  rope_rows_roundtrip bytes vs orope + oq: %ld differ\n", ndiff(hx, hchain));

    /* ---- timing: K issues of an op (or of a chain) back to back between two events, as a recorded pass streams them
     * (each launch waits for the one before: the dependent-launch gap is in the time), median over `iters` repeats. */
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto cold = [&]() { if (dcold) CK(hipMemsetAsync(dcold, (int)(g_rs & 0xff), (size_t)coldmb << 20, st)); };
    const int K = kis > 0 ? kis : 20;
    auto timed = [&](const char* name, int pin, auto body) {
        std::vector<float> v;
        r4d_attn_dec_pin(pin);
        for (int it = 0; it < iters + 3; ++it) {
            cold();
            CK(hipEventRecord(e0, st));
            for (int k = 0; k < K; ++k) body();
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (it >= 3) v.push_back(ms * 1000.0f / K);
        }
        r4d_attn_dec_pin(-1);
        std::sort(v.begin(), v.end());
        std::printf("  %-30s %8.2f us  (p10 %.2f, p90 %.2f)\n", name, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10]);
        return v[v.size() / 2];
    };
    std::printf("timing (us an issue, K %d back to back, median of %d)\n", K, iters);
    for (int w = 0; w < 200; ++w) run(attn, r4d_rad_ds41_attn_sparse_sink);   /* clocks up before the first row */
    CK(hipStreamSynchronize(st));
    auto A = [&]() { run(attn, r4d_rad_ds41_attn_sparse_sink); };
    auto R = [&]() { run(orope, r4d_rad_ds41_rope_rows); };
    auto Q = [&]() { run(oq, r4d_rad_quant_roundtrip); };
    auto RT = [&]() { run(ort, r4d_rad_rope_rows_roundtrip); };
    auto FD = [&]() { run(fold, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip); };
    timed("attn (rule)", -1, A);
    r4d_attn_wm_from(1); timed("attn DAW wmma form", -1, A); timed("attn DAW wmma + orope + oq", -1, [&]() { A(); R(); Q(); }); r4d_attn_wm_from(0);
    timed("attn (rule) again", -1, A); timed("attn (rule) + orope + oq", -1, [&]() { A(); R(); Q(); });
    for (int f : forms) { char nm[64]; std::snprintf(nm, sizeof nm, "attn form %d", f); timed(nm, f, A); }
    timed("orope", -1, R);
    r4d_qg8_pin(0); timed("oq (one column a thread)", -1, Q); r4d_qg8_pin(-1);
    timed("oq (QG8)", -1, Q);
    timed("ort (orope + oq fused)", -1, RT);
    timed("fold d (rule)", -1, FD);
    for (int f : forms) { char nm[64]; std::snprintf(nm, sizeof nm, "fold d form %d", f); timed(nm, f, FD); }
    {   /* quant_roundtrip by shape (o_q's [M, 8192] vs kv_q's [M, 512]) on the same buffer */
        const int shp[][2] = { { 16, 512 }, { 16, 8192 }, { 80, 512 }, { 80, 8192 }, { 8, 16384 }, { 32, 4096 }, { 128, 1024 },
                               { 32, 8192 }, { 48, 8192 }, { 64, 8192 }, { 160, 8192 }, { 1, 8192 }, { 4, 8192 }, { 8, 8192 } };
        for (auto& sh : shp) {
            const int mm = sh[0], nn = sh[1];
            {
                void* qb; CK(hipMalloc(&qb, (size_t)mm * nn * 2)); CK(hipMemset(qb, 0x3f, (size_t)mm * nn * 2));
                Call c; c.t = { T(qb, RAD_BF16, { mm, nn }) };
                c.p = { PI("M", mm), PI("n", nn), PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"), PS("dtype", "bf16") }; c.fix();
                char nm[64]; std::snprintf(nm, sizeof nm, "oq shape [%d, %d] col", mm, nn);
                r4d_qg8_pin(0); timed(nm, -1, [&]() { run(c, r4d_rad_quant_roundtrip); }); r4d_qg8_pin(-1);
                std::snprintf(nm, sizeof nm, "oq shape [%d, %d] QG8", mm, nn);
                timed(nm, -1, [&]() { run(c, r4d_rad_quant_roundtrip); });
                CK(hipFree(qb));
            }
        }
    }
    std::printf("chains (us a chain):\n");
    /* before (BEST's libr4d: the old attention rule, no dodge, no QG8) vs now, in process, A B A B */
    for (int rep = 0; rep < 2; ++rep) {
        r4d_attn_wqa(0); r4d_wq_pin(0); r4d_qg8_pin(0);
        timed("BEFORE: attn orope oq", -1, [&]() { A(); R(); Q(); });
        timed("BEFORE: attn", -1, A); timed("BEFORE: orope", -1, R); timed("BEFORE: oq", -1, Q);
        r4d_attn_wqa(1); r4d_wq_pin(-1); r4d_qg8_pin(-1);
        timed("NOW: attn orope oq", -1, [&]() { A(); R(); Q(); });
        timed("NOW: attn", -1, A); timed("NOW: orope", -1, R); timed("NOW: oq", -1, Q);
        timed("NOW: fold d", -1, FD);
        timed("NOW: attn ort", -1, [&]() { A(); RT(); });
    }
    r4d_qg8_pin(0); timed("served: attn orope oq", -1, [&]() { A(); R(); Q(); }); r4d_qg8_pin(-1);
    timed("attn orope oq(QG8)", -1, [&]() { A(); R(); Q(); });
    timed("attn + ort", -1, [&]() { A(); RT(); });
    timed("fold d", -1, FD);
    for (int f : forms) { char nm[64]; std::snprintf(nm, sizeof nm, "form %d: attn orope oq", f); timed(nm, f, [&]() { A(); R(); Q(); }); }
    return 0;
}
