/* tbench (DRAFTER track, tp8 loop 19.11.2026 round 13; round 15: the serving form of lever TSK, needs RADIANCE_DS41_TSK=0 at start) -- one card, the TRUNK's decode attention of one CSA layer at a lane's rows
 * (S sequences x B rows at consecutive positions), through libr4d's own entries (r4d_ds41_sparse.hip with the plugin's flags):
 *   served   latent_gather (nvfp4 compressed cache -> bf16 rows of the row's top-k entries) + attn_sparse_sink_rope_roundtrip_gathered
 *            (the decode band's exact kernel with the o rope + fp8 round trip in its epilogue) = lq167 / lq173's ds41_latent_gather_nv4 +
 *            ds41_attn_dec<8,512,2,4,..,ORT> pair; above 48 rows the engine runs it as band pieces (F58): --pieces P splits the rows
 *   packed   attn_sparse_sink_rope_roundtrip on the packed cache (the decode band's NV kernel, no gather)
 *   wmma     attn_sparse_sink's WMMA prefill form (ds41_attn_sparse_sink_wm_kernel, reads the packed cache) forced at these rows by the
 *            bench pin r4d_attn_wm_force (a bench-only patch, not in any image) + rope_rows_roundtrip -- NOT the decode band's bits
 *            (rule B): values vs served printed
 * 16 heads a rank (64 / TP4), the 512-wide MQA latent, window 128 (bf16 window cache), ratio --ratio (CSA 4), top-k --topk (512),
 * context --ctx tokens a sequence. Timing: K issues back to back between two events (dependent launches), median of --iters.
 * usage: tbench [--seqs 16] [--bpr 3] [--ctx 8192] [--ratio 4] [--topk 512] [--iters 200] [--cold 0] [--pieces 1] [--k 10]
 *              [--dxs f1,f2,..] (lever DXS forms, round 14: bytes vs served must be 0; times) [--skip-dsk] */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdlib.h>

#include "rad_abi.h"

const void** r4d_dispatch_log = nullptr;
int r4d_dispatch_cap = 0;
int r4d_dispatch_n = 0;
unsigned int* r4d_8a8_split_ctr(RadStream, const void*) { return nullptr; }
int r4d_f32dec_rgs_form(int) { return 0; }

extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_ds41_latent_gather(const RadArgs* a, RadStream s);
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs* a, RadStream s);
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs* a, RadStream s);
extern "C" int64_t r4d_rad_ds41_attn_sparse_sink_scratch(const RadArgs* a);
extern "C" void r4d_attn_dec_pin(int c);   /* the decode band's form 0-6 (-1: the rule) */
void r4d_tsk_set(int v);                   /* lever TSK (round 15): the fn4 switch ds41_tsk's setter (2 = TSKC, round 22) */
void r4d_tsk_bench_nomerge(int v);          /* round 22: ds41_tsk 1 without its merge launch (bench only: prices the merge) */
int r4d_tsk_get();
void r4d_tsk_bench_dedup(int v) __attribute__((weak));   /* round 38: TDD bound (WRONG bytes; bench branch only): TSK GA reads its sequence's first row's copy, the gather writes first rows */
void r4d_tdd_set(int v);                    /* round 38: lever TDD on / off (-1 = the switch: ds41_tsk 3) */
extern "C" long long r4d_tsk_declines();

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static RadTensor OFFS(RadTensor t, int64_t rows) {   /* a row slice of a rank-2 (or rank-1) tensor */
    t.data = (uint8_t*)t.data + rows * t.stride[0] * (t.dtype == RAD_I32 || t.dtype == RAD_F32 ? 4 : 2);
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadParam PF(const char* k, double v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static uint16_t h_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint64_t g_rs = 0x9e3779b97f4a7c15ull;
static uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
static float frand() { return (float)((rnd() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }
struct Call { std::vector<RadTensor> t; std::vector<RadParam> p; RadArgs a; void fix() { std::memset(&a, 0, sizeof(a)); a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; a.wgp_count = 32; } };

/* round 38: a READ flush (clean lines, as the experts' weight stream leaves L2 / MALL between two layers' attentions; a memset leaves
 * dirty lines whose write-back the next kernel pays) */
__global__ void tb_flush_read(const uint4* __restrict__ p, size_t n, uint4* __restrict__ sink) {
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x & acc.y & acc.z & acc.w) == 0xdeadbeefu) sink[0] = acc;
}
int main(int argc, char** argv) {
    int S = 16, bpr = 3, ctx = 8192, ratio = 4, topk = 512, iters = 200, coldmb = 0, pieces = 1, K = 10, decpins = 0, timing = 1, fast = 0, coldr = 0;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)"0"; };
        if (k == "--seqs") S = std::atoi(nx());
        else if (k == "--bpr") bpr = std::atoi(nx());
        else if (k == "--ctx") ctx = std::atoi(nx());
        else if (k == "--ratio") ratio = std::atoi(nx());
        else if (k == "--topk") topk = std::atoi(nx());
        else if (k == "--iters") iters = std::atoi(nx());
        else if (k == "--cold") coldmb = std::atoi(nx());
        else if (k == "--coldr") { coldmb = std::atoi(nx()); coldr = 1; }   /* round 38: the flush READS coldmb MiB instead of a memset */
        else if (k == "--pieces") pieces = std::atoi(nx());
        else if (k == "--k") K = std::atoi(nx());
        else if (k == "--no-timing") timing = 0;
        else if (k == "--decpins") decpins = 1;
        else if (k == "--fast") fast = 1;   /* round 38: timing = TSKC arms + the TDD bound only */
        else { std::fprintf(stderr, "unknown %s\n", k.c_str()); return 1; }
    }
    const int H = 16, D = 512, ROT = 64, OFF = D - ROT, HD = H * D, win = 128, bs = 64, cbs = 64;
    const int M = S * bpr, used = ctx + bpr, nwbt = (used + bs - 1) / bs, nblk = S * nwbt + 8;
    const int ncomp = used / ratio;                                     /* compressed entries a sequence */
    const int epb = cbs / ratio, ncbt = (ncomp + epb - 1) / epb, ncblk = S * ncbt + 8;
    const int nvb = 288, cslot = nvb / 2 / ratio;                      /* ds_nv4_bytes(512) = 288 */
    const int maxctx = used + 64;
    /* round 38: a context with fewer compressed entries than topk (the mb load's ~1.1-1.6k tokens): every row selects all of its reachable
     * entries in position order, -1 after (index_topk's output when reach <= K) */
    const bool allsel = ncomp < topk;
    std::printf("tbench: rows %d (%d sequences x %d), heads %d x %d, window %d, ratio %d, topk %d, ctx %d (%d compressed entries a sequence), iters %d, cold %d MiB, pieces %d\n",
                M, S, bpr, H, D, win, ratio, topk, ctx, ncomp, iters, coldmb, pieces);
    std::vector<uint16_t> hq((size_t)M * HD), hc((size_t)nblk * bs * D);
    for (auto& v : hq) v = h_to_bf16(frand());
    for (auto& v : hc) v = h_to_bf16(frand());
    std::vector<int32_t> bt((size_t)S * nwbt), wu(S), pos(M), cu(S + 1), cbt((size_t)S * ncbt), sel((size_t)M * topk);
    auto shuffled = [](int n) { std::vector<int32_t> p(n); for (int i = 0; i < n; ++i) p[i] = i; for (int i = n - 1; i > 0; --i) std::swap(p[i], p[(int)(rnd() % (uint64_t)(i + 1))]); return p; };
    const std::vector<int32_t> perm = shuffled(nblk), cperm = shuffled(ncblk);
    for (int s = 0; s < S; ++s) {
        for (int j = 0; j < nwbt; ++j) bt[(size_t)s * nwbt + j] = perm[(size_t)s * nwbt + j];
        for (int j = 0; j < ncbt; ++j) cbt[(size_t)s * ncbt + j] = cperm[(size_t)s * ncbt + j];
        wu[s] = used; cu[s] = s * bpr;
        /* the sequence's rows at consecutive positions; each row's top-k: a shared core + a few of its own (adjacent positions' selections overlap) */
        std::vector<int32_t> base = shuffled(ncomp);
        for (int r = 0; r < bpr; ++r) {
            const int m = s * bpr + r;
            pos[m] = used - bpr + r;
            if (allsel) {
                const int reach = std::min((pos[m] + 1) / ratio, ncomp);
                for (int j = 0; j < topk; ++j) sel[(size_t)m * topk + j] = j < reach ? j : -1;
                continue;
            }
            std::vector<int32_t> pick(base.begin(), base.begin() + topk);
            for (int q = 0; q < topk / 16; ++q) pick[(size_t)(rnd() % (uint64_t)topk)] = base[(size_t)(topk + (rnd() % (uint64_t)(ncomp - topk)))];
            std::sort(pick.begin(), pick.end()); pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
            while ((int)pick.size() < topk) { const int32_t x = base[(size_t)(rnd() % (uint64_t)ncomp)]; if (!std::binary_search(pick.begin(), pick.end(), x)) { pick.push_back(x); std::sort(pick.begin(), pick.end()); } }
            for (int j = 0; j < topk; ++j) sel[(size_t)m * topk + j] = pick[(size_t)j];
        }
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
    void *dq, *dc, *dbt, *dwu, *dpos, *dcu, *dsink, *dcs, *dout, *dcc, *dcbt, *dsel, *dg, *dx, *dpage, *dcold = nullptr;
    const size_t ob = (size_t)M * HD * 2, ccb = (size_t)ncblk * cbs * cslot * 2, gb = (size_t)M * topk * D * 2;
    CK(hipMalloc(&dq, ob)); CK(hipMalloc(&dc, hc.size() * 2)); CK(hipMalloc(&dbt, bt.size() * 4)); CK(hipMalloc(&dwu, wu.size() * 4));
    CK(hipMalloc(&dpos, pos.size() * 4)); CK(hipMalloc(&dcu, cu.size() * 4)); CK(hipMalloc(&dsink, sink.size() * 4));
    CK(hipMalloc(&dcs, cs.size() * 4)); CK(hipMalloc(&dout, ob)); CK(hipMalloc(&dcc, ccb)); CK(hipMalloc(&dcbt, cbt.size() * 4));
    CK(hipMalloc(&dsel, sel.size() * 4)); CK(hipMalloc(&dg, gb));
    if (coldmb > 0) CK(hipMalloc(&dcold, (size_t)coldmb << 20));
    CK(hipMemcpy(dq, hq.data(), ob, hipMemcpyHostToDevice)); CK(hipMemcpy(dc, hc.data(), hc.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dbt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dwu, wu.data(), wu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dpos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcu, cu.data(), cu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsink, sink.data(), sink.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcs, cs.data(), cs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dcbt, cbt.data(), cbt.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dsel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    const RadStream rs = (RadStream)st;
    auto run = [&](Call& c, int (*f)(const RadArgs*, RadStream), const char* what) {
        const int rc = f(&c.a, rs);
        if (rc != 0) { std::fprintf(stderr, "%s rc %d\n", what, rc); std::exit(3); }
    };
    /* the packed cache: every compressed entry stored by entry_store_nv4 from random bf16 (valid nvfp4 bytes); the cache viewed as
     * [entries, 1, ratio, cslot] (one entry = ratio rows of cslot slots), page = the physical entry */
    {
        const int NE = ncblk * epb;
        std::vector<uint16_t> hx((size_t)NE * D);
        for (auto& v : hx) v = h_to_bf16(frand());
        std::vector<int32_t> page(NE); for (int i = 0; i < NE; ++i) page[i] = i;
        CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dpage, page.size() * 4));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dpage, page.data(), page.size() * 4, hipMemcpyHostToDevice));
        Call es; es.t = { T(dx, RAD_BF16, { NE, D }), T(dpage, RAD_I32, { NE }), T(dcc, RAD_BF16, { NE, 1, ratio, cslot }) };
        es.p = { PI("n", D), PI("block_size", ratio), PS("format", "nvfp4") }; es.fix();
        run(es, r4d_ds41_entry_store_nv4, "entry_store_nv4"); CK(hipStreamSynchronize(st));
    }
    /* the calls over rows [r0, r0 + m) (whole sequences: r0, m multiples of bpr) */
    auto attn_ts = [&](int r0, int m, void* out) {
        const int s0 = r0 / bpr, sn = m / bpr;
        RadTensor tbt = T(dbt, RAD_I32, { sn, nwbt }); tbt.data = (int32_t*)dbt + (size_t)s0 * nwbt;
        RadTensor tcbt = T(dcbt, RAD_I32, { sn, ncbt }); tcbt.data = (int32_t*)dcbt + (size_t)s0 * ncbt;
        RadTensor twu = T(dwu, RAD_I32, { sn }); twu.data = (int32_t*)dwu + s0;
        return std::vector<RadTensor>{ OFFS(T(dq, RAD_BF16, { m, HD }), r0), T(dc, RAD_BF16, { nblk, 1, bs, D }), tbt, twu,
                                       T(dcc, RAD_BF16, { ncblk, 1, cbs, cslot }), tcbt, OFFS(T(dsel, RAD_I32, { m, topk }), r0),
                                       OFFS(T(dpos, RAD_I32, { m }), r0), T(dcu, RAD_I32, { sn + 1 }), T(dsink, RAD_F32, { H }),
                                       OFFS(T(out, RAD_BF16, { m, HD }), r0) };
    };
    /* cu of a piece: its own sequences from 0 (the engine's piece cu) */
    std::vector<void*> pcu;
    auto piece_cu = [&](int m) {
        std::vector<int32_t> c(m / bpr + 1); for (size_t i = 0; i < c.size(); ++i) c[i] = (int32_t)i * bpr;
        void* d; CK(hipMalloc(&d, c.size() * 4)); CK(hipMemcpy(d, c.data(), c.size() * 4, hipMemcpyHostToDevice)); pcu.push_back(d); return d;
    };
    auto ap = [&](int m) {
        return std::vector<RadParam>{ PI("M", m), PI("heads", H), PI("dim", D), PI("window", win), PI("ratio", ratio), PI("topk", topk),
                                      PF("scale", 1.0 / std::sqrt((double)D)), PS("dtype", "bf16"), PS("latent_format", "nvfp4") };
    };
    auto ortp = [&](std::vector<RadParam> p) {
        for (auto q : { PI("rot", ROT), PI("off", OFF), PI("inverse", 1), PI("block", 32) }) p.push_back(q);
        p.push_back(PS("mode", "gptj")); p.push_back(PS("format", "fp8_e4m3")); p.push_back(PS("block_scale", "ue8m0"));
        return p;
    };
    /* piece split: equal whole-sequence pieces (64 rows / 2 = 32 + 32) */
    std::vector<std::pair<int, int>> pc;
    { const int sp = (S + pieces - 1) / pieces; for (int s = 0; s < S; s += sp) pc.push_back({ s * bpr, std::min(sp, S - s) * bpr }); }
    std::vector<Call> G(pc.size()), A(pc.size()), P(pc.size()), W(1), R(1);
    for (size_t i = 0; i < pc.size(); ++i) {
        const int r0 = pc[i].first, m = pc[i].second;
        void* c = piece_cu(m);
        RadTensor tcbt = T(dcbt, RAD_I32, { m / bpr, ncbt }); tcbt.data = (int32_t*)dcbt + (size_t)(r0 / bpr) * ncbt;
        G[i].t = { T(dcc, RAD_BF16, { ncblk, 1, cbs, cslot }), tcbt, OFFS(T(dsel, RAD_I32, { m, topk }), r0), T(c, RAD_I32, { m / bpr + 1 }),
                   OFFS(T(dg, RAD_BF16, { m, (int64_t)topk * D }), r0) };
        G[i].p = { PI("dim", D), PI("ratio", ratio), PI("topk", topk), PS("latent_format", "nvfp4") }; G[i].fix();
        A[i].t = attn_ts(r0, m, dout); A[i].t[8] = T(c, RAD_I32, { m / bpr + 1 }); A[i].t.push_back(T(dcs, RAD_F32, { maxctx, ROT }));
        A[i].t.push_back(OFFS(T(dg, RAD_BF16, { m, (int64_t)topk * D }), r0)); A[i].p = ortp(ap(m)); A[i].fix();
        P[i].t = attn_ts(r0, m, dout); P[i].t[8] = T(c, RAD_I32, { m / bpr + 1 }); P[i].t.push_back(T(dcs, RAD_F32, { maxctx, ROT }));
        P[i].p = ortp(ap(m)); P[i].fix();
    }
    W[0].t = attn_ts(0, M, dout); W[0].t[8] = T(dcu, RAD_I32, { S + 1 }); W[0].p = ap(M); W[0].fix();
    R[0].t = { T(dout, RAD_BF16, { M, HD }), T(dcs, RAD_F32, { maxctx, ROT }), T(dpos, RAD_I32, { M }) };
    R[0].p = { PI("M", M), PI("n_head", H), PI("head_dim", D), PI("rot", ROT), PI("off", OFF), PS("mode", "gptj"), PI("inverse", 1), PS("dtype", "bf16"),
               PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0") }; R[0].fix();
    /* the ops' scratch: what the plugin's hook asks for at this band (NVG / TSK), as the arena would hold it */
    {
        Call hk; hk.p = ap(M); hk.fix();
        const int64_t need = r4d_rad_ds41_attn_sparse_sink_scratch(&hk.a);
        Call h48; h48.p = ap(48); h48.fix();
        std::printf("  scratch hook: %lld B at M %d, %lld B at M 48; lever TSK switch at start %d\n", (long long)need, M,
                    (long long)r4d_rad_ds41_attn_sparse_sink_scratch(&h48.a), r4d_tsk_get());
    }
    const size_t scrb = (size_t)M * 10 * H * (D + 2) * 4 + 4096 + (size_t)M * topk * D * 2;
    void* dscr; CK(hipMalloc(&dscr, scrb));
    for (size_t i = 0; i < pc.size(); ++i) { A[i].a.scratch = dscr; A[i].a.scratch_bytes = (int64_t)scrb; P[i].a.scratch = dscr; P[i].a.scratch_bytes = (int64_t)scrb; }
    W[0].a.scratch = dscr; W[0].a.scratch_bytes = (int64_t)scrb;
    auto served = [&]() { for (size_t i = 0; i < pc.size(); ++i) { run(G[i], r4d_rad_ds41_latent_gather, "latent_gather"); run(A[i], r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered, "attn gathered"); } };
    auto gather_only = [&]() { for (size_t i = 0; i < pc.size(); ++i) run(G[i], r4d_rad_ds41_latent_gather, "latent_gather"); };
    auto attn_only = [&]() { for (size_t i = 0; i < pc.size(); ++i) run(A[i], r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered, "attn gathered"); };
    auto packed = [&]() { for (size_t i = 0; i < pc.size(); ++i) run(P[i], r4d_rad_ds41_attn_sparse_sink_rope_roundtrip, "attn packed"); };
    /* the two-launch reference of the fused form: attn_sparse_sink (the packed cache, whole pass) then rope_rows_roundtrip */
    auto twol = [&]() { run(W[0], r4d_rad_ds41_attn_sparse_sink, "attn"); run(R[0], r4d_rad_rope_rows_roundtrip, "rope_rows_roundtrip"); };
    std::vector<uint16_t> h0(ob / 2), t1(ob / 2), t2(ob / 2), t3(ob / 2), t4(ob / 2), h4(ob / 2);
    auto grab = [&](std::vector<uint16_t>& h) { CK(hipStreamSynchronize(st)); CK(hipMemcpy(h.data(), dout, ob, hipMemcpyDeviceToHost)); };
    auto clr = [&]() { CK(hipMemset(dout, 0xff, ob)); };
    auto fnv = [](const std::vector<uint16_t>& h) { uint64_t x = 1469598103934665603ull; for (uint16_t v : h) { x ^= v; x *= 1099511628211ull; } return x; };
    auto diff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long d = 0; for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i]; return d; };
    r4d_tsk_set(0);
    clr(); served(); grab(h0); clr(); twol(); grab(h4);
    r4d_tsk_set(1);
    const int on = r4d_tsk_get();
    clr(); served(); grab(t1); clr(); served(); grab(t2); clr(); packed(); grab(t3); clr(); twol(); grab(t4);
    r4d_tsk_set(0);
    std::printf("  hash: served (stock) %016llx, stock two launches %016llx; TSK (switch %d) served %016llx packed %016llx two launches %016llx; declines %lld\n",
                (unsigned long long)fnv(h0), (unsigned long long)fnv(h4), on, (unsigned long long)fnv(t1), (unsigned long long)fnv(t3),
                (unsigned long long)fnv(t4), r4d_tsk_declines());
    {
        long nan1 = 0; double se = 0, sr = 0, mx = 0;
        for (size_t i = 0; i < h0.size(); ++i) {
            const double a = bf2f(h0[i]), b = bf2f(t1[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) { ++nan1; continue; }
            se += (a - b) * (a - b); sr += a * a; mx = std::max(mx, std::fabs(a - b));
        }
        std::printf("  TSK bytes: repeat %ld differ; packed (no gather) vs gathered %ld; fused merge vs two launches (TSK attn + rope_rows_rt) %ld; stock: "
                    "fused vs two launches %ld; TSK vs stock %ld of %zu differ (%.2f %%), max |d| %.3e, rms rel %.3e, non-finite %ld\n",
                    diff(t1, t2), diff(t1, t3), diff(t3, t4), diff(h0, h4), diff(t1, h0), t1.size(), 100.0 * diff(t1, h0) / t1.size(), mx,
                    std::sqrt(se / (sr > 0 ? sr : 1)), nan1);
    }
    {   /* lever TSKC (round 22, ds41_tsk 2): the merge past the full-card cliff must give ds41_tsk 1's bytes */
        std::vector<uint16_t> f1(ob / 2), f2(ob / 2), f3(ob / 2);
        r4d_tsk_set(2);
        const int on2 = r4d_tsk_get();
        clr(); served(); grab(f1); clr(); served(); grab(f2); clr(); packed(); grab(f3);
        r4d_tsk_set(0);
        std::printf("  TSKC (switch %d) hash %016llx: vs TSK two launches %ld differ, repeat %ld, packed vs gathered %ld; declines %lld\n", on2,
                    (unsigned long long)fnv(f1), diff(f1, t1), diff(f1, f2), diff(f1, f3), r4d_tsk_declines());
    }
    for (int sw : { 1, 2 }) {   /* batch invariance: each of sequences 0, S/2, S-1 alone (its bpr rows, its own cu) vs the same rows in the whole pass */
        r4d_tsk_set(sw);
        long dsum = 0;
        for (int s1 : { 0, S / 2, S - 1 }) {
            const int r0 = s1 * bpr;
            void* c1 = piece_cu(bpr);
            Call g1; RadTensor tcbt = T(dcbt, RAD_I32, { 1, ncbt }); tcbt.data = (int32_t*)dcbt + (size_t)s1 * ncbt;
            void* dg1; CK(hipMalloc(&dg1, (size_t)bpr * topk * D * 2)); pcu.push_back(dg1);
            g1.t = { T(dcc, RAD_BF16, { ncblk, 1, cbs, cslot }), tcbt, OFFS(T(dsel, RAD_I32, { bpr, topk }), r0), T(c1, RAD_I32, { 2 }),
                     T(dg1, RAD_BF16, { bpr, (int64_t)topk * D }) };
            g1.p = { PI("dim", D), PI("ratio", ratio), PI("topk", topk), PS("latent_format", "nvfp4") }; g1.fix();
            Call a1; a1.t = attn_ts(r0, bpr, dout); a1.t[8] = T(c1, RAD_I32, { 2 }); a1.t.push_back(T(dcs, RAD_F32, { maxctx, ROT }));
            a1.t.push_back(T(dg1, RAD_BF16, { bpr, (int64_t)topk * D })); a1.p = ortp(ap(bpr)); a1.fix();
            a1.a.scratch = dscr; a1.a.scratch_bytes = (int64_t)scrb;
            clr(); run(g1, r4d_rad_ds41_latent_gather, "gather1"); run(a1, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered, "attn1");
            std::vector<uint16_t> x(ob / 2); grab(x);
            long d = 0; for (size_t i = (size_t)r0 * HD; i < (size_t)(r0 + bpr) * HD; ++i) d += x[i] != t1[i];
            std::printf("  TSK (switch %d) batch invariance: sequence %d alone (%d rows) vs in the %d-row pass (switch 1): %ld of %d differ\n", sw, s1, bpr, M, d, bpr * HD);
            dsum += d;
        }
        r4d_tsk_set(0);
        std::printf("  TSK (switch %d) batch invariance total: %ld differ\n", sw, dsum);
    }
    if (decpins) for (int c = 0; c <= 6; ++c) {   /* the decode band's forms: bytes vs the rule (all the same bits by construction) */
        std::vector<uint16_t> x1(ob / 2);
        r4d_attn_dec_pin(c); clr(); served(); grab(x1); r4d_attn_dec_pin(-1);
        std::printf("  decode form %d bytes: gathered vs served (the rule) %ld differ\n", c, diff(x1, h0));
    }
    if (!timing) { for (void* d : pcu) CK(hipFree(d)); return 0; }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    if (dcold && coldr) CK(hipMemset(dcold, 0x5a, (size_t)coldmb << 20));
    auto cold = [&]() {
        if (!dcold) return;
        if (coldr) hipLaunchKernelGGL(tb_flush_read, dim3(1024), dim3(256), 0, st, (const uint4*)dcold, ((size_t)coldmb << 20) / 16, (uint4*)dcold);
        else CK(hipMemsetAsync(dcold, (int)(rnd() & 0xff), (size_t)coldmb << 20, st));
    };
    auto timed = [&](const char* name, auto body) {
        std::vector<float> v;
        for (int it = 0; it < iters + 3; ++it) {
            cold();
            CK(hipEventRecord(e0, st));
            for (int k = 0; k < K; ++k) body();
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (it >= 3) v.push_back(ms * 1000.0f / K);
        }
        std::sort(v.begin(), v.end());
        std::printf("  %-48s %8.2f us  (p10 %.2f, p90 %.2f)  %.2f us a row\n", name, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v[v.size() / 2] / M);
        return v[v.size() / 2];
    };
    for (int w = 0; w < 100; ++w) served();
    CK(hipStreamSynchronize(st));
    std::printf("timing (us a layer call, K %d back to back, median of %d); gathered rows %.1f MB, packed entries read %.1f MB\n", K, iters,
                gb / 1e6, (double)M * topk * nvb / 1e6);
    if (fast) {   /* round 38: lever TDD's bytes = TSKC's (rule A), repeat, each of three sequences alone vs in the pass */
        std::vector<uint16_t> f1(ob / 2), x1(ob / 2), x2(ob / 2), x3(ob / 2);
        r4d_tsk_set(2); r4d_tdd_set(0);
        clr(); served(); grab(f1);
        r4d_tdd_set(1);
        clr(); served(); grab(x1); clr(); served(); grab(x2); clr(); attn_only(); grab(x3);
        long bi = 0;
        for (int s1 : { 0, S / 2, S - 1 }) {   /* the sequence's rows as their own pass (gather + attention over its rows alone) */
            const int r0 = s1 * bpr;
            void* c1 = piece_cu(bpr);
            Call g1; RadTensor tcbt = T(dcbt, RAD_I32, { 1, ncbt }); tcbt.data = (int32_t*)dcbt + (size_t)s1 * ncbt;
            void* dg1; CK(hipMalloc(&dg1, (size_t)bpr * topk * D * 2)); pcu.push_back(dg1);
            g1.t = { T(dcc, RAD_BF16, { ncblk, 1, cbs, cslot }), tcbt, OFFS(T(dsel, RAD_I32, { bpr, topk }), r0), T(c1, RAD_I32, { 2 }),
                     T(dg1, RAD_BF16, { bpr, (int64_t)topk * D }) };
            g1.p = { PI("dim", D), PI("ratio", ratio), PI("topk", topk), PS("latent_format", "nvfp4") }; g1.fix();
            Call a1; a1.t = attn_ts(r0, bpr, dout); a1.t[8] = T(c1, RAD_I32, { 2 }); a1.t.push_back(T(dcs, RAD_F32, { maxctx, ROT }));
            a1.t.push_back(T(dg1, RAD_BF16, { bpr, (int64_t)topk * D })); a1.p = ortp(ap(bpr)); a1.fix();
            a1.a.scratch = dscr; a1.a.scratch_bytes = (int64_t)scrb;
            clr(); run(g1, r4d_rad_ds41_latent_gather, "gather1"); run(a1, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered, "attn1");
            std::vector<uint16_t> x(ob / 2); grab(x);
            for (size_t i = (size_t)r0 * HD; i < (size_t)(r0 + bpr) * HD; ++i) bi += x[i] != f1[i];
        }
        std::vector<uint16_t> x4(ob / 2), x5(ob / 2);
        r4d_tdd_set(-1); r4d_tsk_set(3);   /* the serving form: fn4 ds41_tsk 3 */
        const int on3 = r4d_tsk_get();
        clr(); served(); grab(x4);
        r4d_tsk_set(2); clr(); attn_only(); grab(x5);   /* switch back to 2 with the view still deduplicated: read through its map */
        r4d_tdd_set(0); r4d_tsk_set(0);
        std::printf("  TDD bytes: vs TSKC (served, gather + attn) %ld of %zu differ, repeat %ld, attn alone after the gather %ld, three sequences alone vs "
                    "the TSKC pass %ld; switch %d (ds41_tsk 3) vs TSKC %ld, switch 2 over a deduplicated view %ld; hash %016llx (TSKC %016llx); declines %lld\n",
                    diff(x1, f1), f1.size(), diff(x1, x2), diff(x3, f1), bi, on3, diff(x4, f1), diff(x5, f1),
                    (unsigned long long)fnv(x1), (unsigned long long)fnv(f1), r4d_tsk_declines());
    }
    for (int rep = 0; rep < 2 && fast; ++rep) {   /* round 38 */
        r4d_tsk_set(2);
        timed("TSKC: gather + attn gathered (ORT)", served);
        timed("  TSKC attn gathered alone (ORT)", attn_only);
        timed("  gather alone", gather_only);
        r4d_tdd_set(1);
        timed("TDD: gather + attn gathered (ORT)", served);
        timed("  TDD attn gathered alone (ORT)", attn_only);
        timed("  TDD gather alone", gather_only);
        r4d_tdd_set(0);
        if (r4d_tsk_bench_dedup) {
            r4d_tsk_bench_dedup(1);
            timed("TDD bound: gather + attn gathered (ORT)", served);
            timed("  TDD bound attn gathered alone (ORT)", attn_only);
            timed("  TDD bound gather alone", gather_only);
            r4d_tsk_bench_dedup(0);
        }
        r4d_tsk_set(0);
    }
    for (int rep = 0; rep < 2 && !fast; ++rep) {
        r4d_tsk_set(0);
        timed("stock: gather + attn gathered (ORT)", served);
        timed("  stock attn gathered alone (ORT)", attn_only);
        timed("  gather alone", gather_only);
        r4d_tsk_set(1);
        timed("TSK: gather + attn gathered (ORT)", served);
        timed("  TSK attn gathered alone (ORT)", attn_only);
        timed("  TSK packed (no gather, ORT)", packed);
        r4d_tsk_bench_nomerge(1);
        timed("  TSK attn gathered WITHOUT its merge (bench)", attn_only);
        r4d_tsk_bench_nomerge(2);
        timed("  TSK merge ALONE (bench)", attn_only);
        r4d_tsk_bench_nomerge(0);
        r4d_tsk_set(2);
        timed("TSKC: gather + attn gathered (ORT)", served);
        timed("  TSKC attn gathered alone (ORT)", attn_only);
        timed("  TSKC packed (no gather, ORT)", packed);
        r4d_tsk_set(0);
    }
    for (void* d : pcu) CK(hipFree(d));
    return 0;
}
