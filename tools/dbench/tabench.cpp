/* tabench (tp8 loop DECODE round 21, 19.11.2026) -- one card, the TRUNK's compressed-layer decode attention as served (lever DKV-G:
 * attn_sparse_sink_gathered over latent_gather's rows; 16 heads a rank, the 512-wide MQA latent, window 128 bf16 + topk gathered
 * entries, latent_format nvfp4), through libr4d's own entry, at decode heights: the rule and every decode-band form 0-6 pinned
 * (r4d_attn_dec_pin; every form must give the rule's bytes). Why: the follower's ds41_attn_dec at c32 is 4,376 us of a 39.5-ms
 * 48-row full lane step (lq167 stamps, 11 %), ~4x the bytes floor (48 rows x 640 keys x 1 KiB a layer, 22 layers, at ~640 GB/s
 * ~1.1 ms) -- does another form of the same bytes run it faster? --band 64 also times a 49..64-row pass in one launch (libr4d's
 * per-op band, D25 step 3).
 *
 * usage: tabench [--seqs 16] [--rpr 3] [--ctx 3000] [--topk 512] [--ratio 2] [--iters 200] [--k 22] [--cold 0] [--band 0]
 *        [--forms 0,1,2,3,4,5,6]   (rows = seqs x rpr; --k issues back to back between two events = the layers of a pass) */
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
unsigned int* r4d_8a8_split_ctr(RadStream, const void*) { return nullptr; }
int r4d_f32dec_rgs_form(int) { return 0; }

extern "C" int r4d_rad_ds41_attn_sparse_sink_gathered(const RadArgs* a, RadStream s);
extern "C" void r4d_attn_dec_pin(int c);

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadParam PF(const char* k, double v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static uint16_t h_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static uint64_t g_rs = 0x9e3779b97f4a7c15ull;
static uint64_t xr() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
static float frand() { return (float)((xr() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }
static std::vector<int> ints(const char* s) { std::vector<int> v; while (s && *s) { v.push_back(std::atoi(s)); s = std::strchr(s, ','); if (s) ++s; } return v; }

int main(int argc, char** argv) {
    int seqs = 16, rpr = 3, ctx = 3000, topk = 512, ratio = 2, iters = 200, K = 22, coldmb = 0, band = 0;
    std::vector<int> forms = { 0, 1, 2, 3, 4, 5, 6 };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)""; };
        if (k == "--seqs") seqs = std::atoi(nx());
        else if (k == "--rpr") rpr = std::atoi(nx());
        else if (k == "--ctx") ctx = std::atoi(nx());
        else if (k == "--topk") topk = std::atoi(nx());
        else if (k == "--ratio") ratio = std::atoi(nx());
        else if (k == "--iters") iters = std::atoi(nx());
        else if (k == "--k") K = std::atoi(nx());
        else if (k == "--cold") coldmb = std::atoi(nx());
        else if (k == "--band") band = std::atoi(nx());
        else if (k == "--forms") forms = ints(nx());
        else { std::fprintf(stderr, "unknown %s\n", k.c_str()); return 1; }
    }
    const int H = 16, D = 512, HD = H * D, win = 128, bs = 64, cbs = 64;
    const int S = seqs, M = S * rpr;
    const int used = ctx + rpr, nwbt = (used + bs - 1) / bs, nwblk = S * nwbt + 8;
    const int nent = used / ratio + 1, ncbt = (nent * ratio + cbs - 1) / cbs + 1, ncblk = S * ncbt + 8;
    const int cslot = ((D / 2 + D / 16 + 15) / 16 * 16) / 2;   /* ds_nv4_bytes(512) / 2 bf16 slots an entry */
    std::printf("tabench: rows %d (%d sequences x %d), heads %d x %d, window %d + topk %d gathered (ratio %d, nvfp4), ctx %d, K %d issues, "
                "iters %d, cold %d MiB, band %d\n", M, S, rpr, H, D, win, topk, ratio, ctx, K, iters, coldmb, band);
    std::vector<uint16_t> hq((size_t)M * HD), hw((size_t)nwblk * bs * D), hg((size_t)M * topk * D);
    for (auto& v : hq) v = h_to_bf16(frand() * 0.05f);
    for (auto& v : hw) v = h_to_bf16(frand());
    for (auto& v : hg) v = h_to_bf16(frand());
    std::vector<uint16_t> hc((size_t)ncblk * cbs * (cslot / ratio) * ratio, 0);   /* the packed cache: not read on the gathered path */
    std::vector<int32_t> wbt((size_t)S * nwbt), wu(S), pos(M), cu(S + 1), cbt((size_t)S * ncbt), sel((size_t)M * topk);
    std::vector<int32_t> perm(nwblk); for (int i = 0; i < nwblk; ++i) perm[i] = i;
    for (int i = nwblk - 1; i > 0; --i) std::swap(perm[i], perm[(int)(xr() % (uint64_t)(i + 1))]);
    std::vector<int32_t> cperm(ncblk); for (int i = 0; i < ncblk; ++i) cperm[i] = i;
    for (int i = ncblk - 1; i > 0; --i) std::swap(cperm[i], cperm[(int)(xr() % (uint64_t)(i + 1))]);
    for (int s = 0; s < S; ++s) {
        for (int j = 0; j < nwbt; ++j) wbt[(size_t)s * nwbt + j] = perm[(size_t)s * nwbt + j];
        for (int j = 0; j < ncbt; ++j) cbt[(size_t)s * ncbt + j] = cperm[(size_t)s * ncbt + j];
        wu[s] = used; cu[s] = s * rpr;
        for (int r = 0; r < rpr; ++r) pos[(size_t)s * rpr + r] = ctx + r;
    }
    cu[S] = M;
    for (int m = 0; m < M; ++m) {   /* topk distinct entries below the row's reach, ascending (index_topk's order) */
        const int reach = (pos[m] + 1) / ratio;
        std::vector<int32_t> pk;
        const int step = std::max(1, reach / topk);
        for (int c = (int)(xr() % (uint64_t)step); c < reach && (int)pk.size() < topk; c += step) pk.push_back(c);
        for (int t = 0; t < topk; ++t) sel[(size_t)m * topk + t] = t < (int)pk.size() ? pk[t] : -1;
    }
    std::vector<float> sink(H); for (auto& v : sink) v = frand();
    void *dq, *dw, *dc, *dg, *dwbt, *dwu, *dpos, *dcu, *dcbt, *dsel, *dsink, *dout, *dcold = nullptr;
    const size_t ob = (size_t)M * HD * 2;
    CK(hipMalloc(&dq, ob)); CK(hipMalloc(&dw, hw.size() * 2)); CK(hipMalloc(&dc, hc.size() * 2)); CK(hipMalloc(&dg, hg.size() * 2));
    CK(hipMalloc(&dwbt, wbt.size() * 4)); CK(hipMalloc(&dwu, wu.size() * 4)); CK(hipMalloc(&dpos, pos.size() * 4)); CK(hipMalloc(&dcu, cu.size() * 4));
    CK(hipMalloc(&dcbt, cbt.size() * 4)); CK(hipMalloc(&dsel, sel.size() * 4)); CK(hipMalloc(&dsink, sink.size() * 4)); CK(hipMalloc(&dout, ob));
    if (coldmb > 0) CK(hipMalloc(&dcold, (size_t)coldmb << 20));
    CK(hipMemcpy(dq, hq.data(), ob, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dc, hc.data(), hc.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, hg.data(), hg.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dwbt, wbt.data(), wbt.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dwu, wu.data(), wu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dpos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcu, cu.data(), cu.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dcbt, cbt.data(), cbt.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dsel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dsink, sink.data(), sink.size() * 4, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    const RadStream rs = (RadStream)st;
    std::vector<RadTensor> t = { T(dq, RAD_BF16, { M, HD }), T(dw, RAD_BF16, { nwblk, 1, bs, D }), T(dwbt, RAD_I32, { S, nwbt }),
                                 T(dwu, RAD_I32, { S }), T(dc, RAD_BF16, { ncblk, 1, cbs, cslot / ratio }), T(dcbt, RAD_I32, { S, ncbt }),
                                 T(dsel, RAD_I32, { M, topk }), T(dpos, RAD_I32, { M }), T(dcu, RAD_I32, { S + 1 }), T(dsink, RAD_F32, { H }),
                                 T(dout, RAD_BF16, { M, HD }), T(dg, RAD_BF16, { M, (int64_t)topk * D }) };
    std::vector<RadParam> p = { PI("M", M), PI("heads", H), PI("dim", D), PI("window", win), PI("ratio", ratio), PI("topk", topk),
                                PF("scale", 1.0 / std::sqrt((double)D)), PS("dtype", "bf16"), PS("latent_format", "nvfp4") };
    if (band > 0) p.push_back(PI("band", band));
    RadArgs a; std::memset(&a, 0, sizeof(a)); a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; a.wgp_count = 32;
    auto run = [&]() { const int rc = r4d_rad_ds41_attn_sparse_sink_gathered(&a, rs); if (rc != 0) { std::fprintf(stderr, "op rc %d\n", rc); std::exit(3); } };
    std::vector<uint16_t> hr(ob / 2), hx(ob / 2);
    auto grab = [&](std::vector<uint16_t>& h) { CK(hipStreamSynchronize(st)); CK(hipMemcpy(h.data(), dout, ob, hipMemcpyDeviceToHost)); };
    auto ndiff = [&](const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) { long n = 0; for (size_t i = 0; i < x.size(); ++i) n += x[i] != y[i]; return n; };
    r4d_attn_dec_pin(-1); run(); grab(hr);
    long nz = 0; for (uint16_t v : hr) nz += (v & 0x7fff) != 0;
    std::printf("  rule output: %ld of %zu values non-zero\n", nz, hr.size());
    for (int f : forms) {
        r4d_attn_dec_pin(f); run(); grab(hx);
        std::printf("  form %d bytes vs the rule: %ld of %zu differ\n", f, ndiff(hx, hr), hx.size());
    }
    r4d_attn_dec_pin(-1);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](const char* name, int pin) {
        std::vector<float> v;
        r4d_attn_dec_pin(pin);
        for (int it = 0; it < iters + 3; ++it) {
            if (dcold) CK(hipMemsetAsync(dcold, (int)(xr() & 0xff), (size_t)coldmb << 20, st));
            CK(hipEventRecord(e0, st));
            for (int k = 0; k < K; ++k) run();
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (it >= 3) v.push_back(ms * 1000.0f / K);
        }
        r4d_attn_dec_pin(-1);
        std::sort(v.begin(), v.end());
        std::printf("TABENCH rows %d seqs %d rpr %d band %d %-12s %8.2f us a launch (p10 %.2f, p90 %.2f)\n", M, S, rpr, band, name,
                    v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10]);
    };
    for (int w = 0; w < 200; ++w) run();
    CK(hipStreamSynchronize(st));
    for (int rep = 0; rep < 2; ++rep) {   /* A B A B: the rule, then every form, twice */
        timed("rule", -1);
        for (int f : forms) { char nm[32]; std::snprintf(nm, sizeof nm, "form %d", f); timed(nm, f); }
    }
    return 0;
}
