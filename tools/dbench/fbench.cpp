/* fbench (DRAFTER track, 18.11.2026 round 5) -- one card, gemm_nt_f32 (the DSpark drafter's router logits: x bf16 [M, 5120]
 * x the f32 ffn_gate_inp [128, 5120]^T -> f32 [M, 128]) through libr4d's own entry (r4d_glm.hip + r4d_ds41_f32pf.hip), every
 * decode-band form pinned (r4d_f32dec_pin(-1, f): 0 the stock rule, tile_wave 1..12, tiled 20..31) against the stock bytes,
 * K issues back to back between two events, median of `iters`, a cold write between samples (the card busy, as served).
 * usage: fbench [--rows 8,16,24,40,48] [--n 128] [--k 5120] [--iters 30] [--cold 64] */
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

extern "C" int r4d_rad_gemm_nt_f32(const RadArgs* a, RadStream s);
extern "C" void r4d_f32dec_pin(int rgs_form, int gemm_form);
int r4d_f32dec_set(int v);

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
static uint16_t h_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static uint64_t g_rs = 0x9e3779b97f4a7c15ull;
static float frand() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return (float)((g_rs >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }
static std::vector<int> ints(const char* s) { std::vector<int> v; while (s && *s) { v.push_back(std::atoi(s)); s = std::strchr(s, ','); if (s) ++s; } return v; }

int main(int argc, char** argv) {
    std::vector<int> rowsv = { 8, 16, 24, 40, 48 };
    int N = 128, K = 5120, iters = 30, coldmb = 64;
    std::vector<int> forms = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*)""; };
        if (k == "--rows") rowsv = ints(nx());
        else if (k == "--n") N = std::atoi(nx());
        else if (k == "--k") K = std::atoi(nx());
        else if (k == "--iters") iters = std::atoi(nx());
        else if (k == "--cold") coldmb = std::atoi(nx());
        else if (k == "--forms") forms = ints(nx());
    }
    const int MM = *std::max_element(rowsv.begin(), rowsv.end());
    std::vector<uint16_t> hx((size_t)MM * K); std::vector<float> hw((size_t)N * K);
    for (auto& v : hx) v = h_to_bf16(frand());
    for (auto& v : hw) v = frand() * 0.05f;
    void *dx, *dw, *dy, *dcold = nullptr;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dw, hw.size() * 4)); CK(hipMalloc(&dy, (size_t)MM * N * 4));
    if (coldmb > 0) CK(hipMalloc(&dcold, (size_t)coldmb << 20));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    std::printf("fbench: gemm_nt_f32 N %d K %d (f32 weight %.2f MB), K-repeat 20, median of %d, cold %d MiB\n", N, K, N * K * 4 / 1e6, iters, coldmb);
    for (int M : rowsv) {
        RadTensor t[3] = { T(dx, RAD_BF16, { M, K }), T(dw, RAD_F32, { N, K }), T(dy, RAD_F32, { M, N }) };
        RadParam p[3] = { PI("M", M), PI("N", N), PI("K", K) };
        RadArgs a; std::memset(&a, 0, sizeof a); a.t = t; a.n_t = 3; a.p = p; a.n_p = 3; a.world_size = 1; a.wgp_count = 32;
        std::vector<float> ref((size_t)M * N), out((size_t)M * N);
        r4d_f32dec_pin(-1, 0);
        if (r4d_rad_gemm_nt_f32(&a, (RadStream)st)) { std::printf("stock refused\n"); return 3; }
        CK(hipStreamSynchronize(st)); CK(hipMemcpy(ref.data(), dy, ref.size() * 4, hipMemcpyDeviceToHost));
        for (int w = 0; w < 300; ++w) r4d_rad_gemm_nt_f32(&a, (RadStream)st);
        CK(hipStreamSynchronize(st));
        std::printf("M %d:", M);
        for (int f : forms) {
            r4d_f32dec_pin(-1, f);
            CK(hipMemset(dy, 0xff, (size_t)M * N * 4));
            if (r4d_rad_gemm_nt_f32(&a, (RadStream)st)) { std::printf(" f%d:refused", f); continue; }
            CK(hipStreamSynchronize(st)); CK(hipMemcpy(out.data(), dy, out.size() * 4, hipMemcpyDeviceToHost));
            const bool same = std::memcmp(out.data(), ref.data(), out.size() * 4) == 0;
            std::vector<float> v;
            for (int it = 0; it < iters + 3; ++it) {
                if (dcold) CK(hipMemsetAsync(dcold, it & 0xff, (size_t)coldmb << 20, st));
                CK(hipEventRecord(e0, st));
                for (int k = 0; k < 20; ++k) r4d_rad_gemm_nt_f32(&a, (RadStream)st);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) v.push_back(ms * 1000.0f / 20);
            }
            std::sort(v.begin(), v.end());
            std::printf(" f%d %.1f%s", f, v[v.size() / 2], same ? "" : "(DIFF)");
        }
        {   /* the served rule: ds41_f32dec 1, no pin */
            r4d_f32dec_set(1); r4d_f32dec_pin(-1, -1);
            CK(hipMemset(dy, 0xff, (size_t)M * N * 4));
            r4d_rad_gemm_nt_f32(&a, (RadStream)st); CK(hipStreamSynchronize(st)); CK(hipMemcpy(out.data(), dy, out.size() * 4, hipMemcpyDeviceToHost));
            const bool same = std::memcmp(out.data(), ref.data(), out.size() * 4) == 0;
            std::vector<float> v;
            for (int it = 0; it < iters + 3; ++it) {
                if (dcold) CK(hipMemsetAsync(dcold, it & 0xff, (size_t)coldmb << 20, st));
                CK(hipEventRecord(e0, st));
                for (int k = 0; k < 20; ++k) r4d_rad_gemm_nt_f32(&a, (RadStream)st);
                CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it >= 3) v.push_back(ms * 1000.0f / 20);
            }
            std::sort(v.begin(), v.end());
            std::printf("  | ds41_f32dec 1 rule %.1f%s", v[v.size() / 2], same ? "" : "(DIFF)");
            r4d_f32dec_set(0);
        }
        std::printf("\n");
    }
    r4d_f32dec_pin(-1, -1);
    return 0;
}
