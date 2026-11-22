/* tests/r4d_emu: lever K1 folds a and e (10.11.2026), the activation's producer run in the staging of lever D2's
 * multi-row form (r4d_ds41_ggml_mr.hip, r4d_gg_mr_fold), on the host SIMT emulator, byte for byte against the two
 * launches they fold:
 *
 *   fold a  pro 1 (rmsnorm_gemm_nt_ggml's one launch) against rmsnorm_body (r4d_model_bf16.hip, ported here: that
 *           unit is not in the emulator's build; the sum of squares in kR4dSumLanes' partition, r4d_lanes_sumsq, the
 *           wave butterfly and block_sum's wave order at r4d_act_threads(M, n), rsqrtf as this build defines it, the
 *           product order and f32_to_bf16) -> h, then the stock decode kernel (gg_gemv_kernel, r4d_gg_mr_emu_stock)
 *           over that h -> y. The selftest's rmsnorm_gemm_nt_ggml cases hold the fold to the device rmsnorm row itself.
 *   fold e  pro 2 (swiglu_clamped_gemm_nt_ggml's one launch) against swiglu_clamped_kernel (r4d_glm.hip, in this
 *           build: r4d_rad_swiglu_clamped) -> h, then the stock decode kernel over h -> y.
 *
 * Both h (every byte, from a sentinel fill) and y must be equal. Types q5_K / q6_K / q8_0; fold a at DeepSeek-V4.1's
 * q_b K (1280, the q_a rank) and fold e at the shared expert's per-rank K (768 on rank 0, 512 on ranks 1-3), an N
 * that leaves partial row groups and slots (45) at every form, M 1..48, every (H, R) pair at its largest and a small
 * MT cap, at one, two and eight waves a block, the rule and the measured table (no pin); and the real N (8192 q_b,
 * 5120 shexp_down) at a few M under the rule. Rows: random bf16 with zeros of both signs, subnormals, rows of huge
 * values (squares and products past f32) and infinities, gains around 1 with a few negative and zero; gate / up
 * values past the clamp both ways. Unaligned activation and h rows on odd M (the 16-bit load / store paths).
 *
 *   r4d_emu_ds41fold2 [quick]      quick: fewer M, fewer forms */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_mr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int r4d_gg_mr_emu_stock(const R4dGgMrCall& c, int rpw);
extern "C" void r4d_gg_mr_pin(int h, int r, int mt, int wn);
extern "C" const char* r4d_gg_mr_last();
extern "C" int r4d_rad_swiglu_clamped(const RadArgs* a, RadStream stream);

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };   /* gg_gemm's GG_* codes */

uint64_t g_rs = 0x2545F4914F6CDD1Dull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
/* r4d_common.h's f32_to_bf16 (no NaN quieting) and bf16_to_f32 */
uint16_t f32_to_bf16(float f) { const uint32_t u = fbits(f); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float bf16_to_f32(uint16_t h) { return bitsf((uint32_t)h << 16); }
uint16_t f2bf(float f) {
    const uint32_t u = fbits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return f32_to_bf16(f);
}
/* the emulator's rsqrtf (tests/r4d_emu/emu/hip/hip_runtime.h), which the folded kernel calls here */
float emu_rsqrtf(float x) { return 1.0f / std::sqrt(x); }
/* r4d_common.h's r4d_act_threads */
int act_threads(int M, int K) {
    if (M <= 32) return 1024;
    const int lim = K >= 8192 ? (1 << 24) : (K >= 4096 ? 120 : 72);
    return M <= lim ? 512 : 256;
}

uint16_t rnd_f16() {
    const uint32_t k = (uint32_t)(rnd() % 100);
    const uint16_t sign = (uint16_t)((rnd() & 1) << 15);
    uint16_t e;
    if (k < 8) e = 0;
    else if (k < 12) e = (uint16_t)(28 + rnd() % 3);
    else e = (uint16_t)(5 + rnd() % 8);
    return (uint16_t)(sign | (e << 10) | (uint16_t)(rnd() & 0x3ff));
}
/* an activation row: values in [-scale, scale] with zeros of both signs and bf16 subnormals sprinkled; mode 1 adds an
 * infinity, mode 2 a few huge values (squares and products past f32) */
void fill_row(uint16_t* x, int K, int mode, float scale) {
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 1000);
        if (c < 5) x[k] = 0x0000;
        else if (c < 10) x[k] = 0x8000;
        else if (c < 15) x[k] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
        else x[k] = f2bf(scale * (float)((int)(rnd() % 20001) - 10000) / 10000.0f);
    }
    if (mode == 1) x[rnd() % K] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | 0x7f80);
    if (mode == 2)
        for (int e = 0; e < 3; ++e)
            x[rnd() % K] = f2bf((rnd() & 1 ? -1.0f : 1.0f) * std::ldexp(1.0f + (float)(rnd() % 1000) / 1000.0f, 64 + (int)(rnd() % 50)));
}

struct Ty { int ty; const char* name; int qk, bb; };
const Ty kTypes[] = { { Q5_K, "q5_K", 256, 176 }, { Q6_K, "q6_K", 256, 210 }, { Q8_0, "q8_0", 32, 34 } };

std::vector<uint8_t> make_w(const Ty& t, int N, int K) {
    const long long rb = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> w((size_t)((long long)N * rb));
    for (auto& b : w) b = (uint8_t)rnd();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / t.qk; ++b) {
            uint8_t* blk = w.data() + (long long)n * rb + (long long)b * t.bb;
            const uint16_t d = rnd_f16(), dm = rnd_f16();
            if (t.ty == Q6_K) std::memcpy(blk + 208, &d, 2);
            else { std::memcpy(blk, &d, 2); if (t.ty == Q5_K) std::memcpy(blk + 2, &dm, 2); }
        }
    return w;
}

/* rmsnorm_body (r4d_model_bf16.hip) at its launch's width, f32 gain: lane v < 256 sums the squares of columns v,
 * v + 256, ... in order; each wave of 32 lanes closes with r4d_wave_sum's butterfly (xor 16, 8, 4, 2, 1: an IEEE add
 * of the same two values in every lane); block_sum adds the waves' sums in order from 0 over all THR / 32 waves (the
 * waves past the eighth hold zero sums); s = rsqrtf(sum / n + eps); y = f32_to_bf16(x * s * (g + wadd) * mult). */
void rmsnorm_ref(const uint16_t* x, long long x_ld, const float* g, int M, int K, float eps, float wadd, float mult,
                 uint16_t* h, long long h_ld) {
    const int nw = act_threads(M, K) / 32;
    for (int m = 0; m < M; ++m) {
        const uint16_t* xr = x + (long long)m * x_ld;
        float lane[256];
        for (int v = 0; v < 256; ++v) {
            float ss = 0.0f;
            for (int c = v; c < K; c += 256) { const float t = bf16_to_f32(xr[c]); ss += t * t; }
            lane[v] = ss;
        }
        float ws[8];
        for (int w = 0; w < 8; ++w) {
            float a[32], b[32];
            for (int l = 0; l < 32; ++l) a[l] = lane[w * 32 + l];
            for (int X = 16; X >= 1; X >>= 1) {
                for (int l = 0; l < 32; ++l) b[l] = a[l] + a[l ^ X];
                std::memcpy(a, b, sizeof a);
            }
            ws[w] = a[0];
        }
        float s = 0.0f;
        for (int w = 0; w < nw; ++w) s += w < 8 ? ws[w] : 0.0f;
        const float sc = emu_rsqrtf(s / (float)K + eps);
        for (int k = 0; k < K; ++k) h[(long long)m * h_ld + k] = f32_to_bf16(bf16_to_f32(xr[k]) * sc * (g[k] + wadd) * mult);
    }
}

RadTensor t2(void* p, int64_t rows, int64_t cols, int64_t ld) {
    RadTensor t{};
    t.data = p;
    t.dtype = RAD_BF16;
    t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols;
    t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
/* swiglu_clamped_kernel through its row's entry: gate_up [M, 2K] (gu_ld a row) -> h [M, K] (h_ld a row) */
int swiglu_ref(const uint16_t* gu, long long gu_ld, int M, int K, float limit, uint16_t* h, long long h_ld) {
    RadTensor ts[2] = { t2((void*)gu, M, 2 * K, gu_ld), t2(h, M, K, h_ld) };
    RadParam ps[2] = {};
    ps[0].key = "n"; ps[0].kind = RAD_P_INT; ps[0].ival = K;
    ps[1].key = "limit"; ps[1].kind = RAD_P_F64; ps[1].dval = limit;
    RadArgs a{};
    a.t = ts; a.n_t = 2; a.p = ps; a.n_p = 2; a.world_size = 1;
    return r4d_rad_swiglu_clamped(&a, nullptr);
}

int g_fail = 0, g_cases = 0;
long long g_h_nan = 0, g_h_vals = 0;

struct Pin { int h, r, mt, wn; };

/* One (fold, type, K, N, M) case under each pin: the reference h and y once, then the fold under every pin. */
void run_case(int pro, const Ty& t, int K, int N, int M, const std::vector<uint8_t>& w, const std::vector<Pin>& pins) {
    const long long w_ld = (long long)K / t.qk * t.bb;
    const int xcols = pro == 1 ? K : 2 * K;
    /* odd M: the rows start one element in (the 16-bit load / store paths), even M: 16-byte aligned rows */
    const int off = (M & 1) ? 1 : 0;
    const long long x_ld = xcols + ((M % 3 == 0) ? 8 : 0);   /* a pitched row on every third M */
    std::vector<uint16_t> xb((size_t)(M * x_ld + 16));
    uint16_t* x = xb.data() + off;
    std::vector<float> g((size_t)K);
    const float limit = 10.0f;
    for (int m = 0; m < M; ++m) {
        const int r8 = (int)(rnd() % 8);
        if (pro == 1) {
            fill_row(x + (long long)m * x_ld, K, r8 == 0 ? 1 : r8 == 1 ? 2 : 0, (m % 5 == 2) ? 0.01f : 4.0f);
        } else {
            /* gate and up past the clamp both ways on some rows */
            fill_row(x + (long long)m * x_ld, K, r8 == 0 ? 1 : 0, (m % 4 == 1) ? 30.0f : 4.0f);
            fill_row(x + (long long)m * x_ld + K, K, r8 == 1 ? 1 : 0, (m % 4 == 3) ? 30.0f : 4.0f);
        }
    }
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 100);
        g[(size_t)k] = c < 3 ? 0.0f : c < 6 ? -0.0f : c < 10 ? -0.5f : 0.75f + (float)(rnd() % 1000) / 2000.0f;
    }
    const float eps = 1e-20f;   /* DeepSeek-V4.1's */
    const long long h_ld = K;
    std::vector<uint16_t> href((size_t)(M * h_ld + 16), 0x5A5A);
    uint16_t* hr = href.data() + off;
    if (pro == 1) rmsnorm_ref(x, x_ld, g.data(), M, K, eps, 0.0f, 1.0f, hr, h_ld);
    else if (swiglu_ref(x, x_ld, M, K, limit, hr, h_ld) != RAD_OK) { std::printf("FAIL swiglu_clamped ref launch\n"); ++g_fail; return; }
    for (int i = 0; i < M * K; ++i) { ++g_h_vals; const float v = bf16_to_f32(hr[i]); if (v != v) ++g_h_nan; }
    std::vector<uint16_t> yref((size_t)M * N, 0xA5A5);
    const int h_al = (((uintptr_t)hr & 15u) == 0 && h_ld % 8 == 0) ? 1 : 0;
    R4dGgMrCall cr{ t.ty, false, hr, h_ld, h_al, w.data(), w_ld, yref.data(), (long long)N, M, N, K };
    if (r4d_gg_mr_emu_stock(cr, 1) != 0) { std::printf("FAIL stock launch %s K=%d M=%d\n", t.name, K, M); ++g_fail; return; }
    const int x_al = (((uintptr_t)x & 15u) == 0 && x_ld % 8 == 0) ? 1 : 0;
    for (const Pin& p : pins) {
        std::vector<uint16_t> hb((size_t)(M * h_ld + 16), 0xC3C3), yb((size_t)M * N, 0x3C3C);
        uint16_t* hg = hb.data() + off;
        R4dGgMrCall c{ t.ty, false, x, x_ld, x_al, w.data(), w_ld, yb.data(), (long long)N, M, N, K };
        R4dGgMrFold f{};
        f.pro = pro;
        f.g = g.data(); f.eps = eps; f.wadd = 0.0f; f.mult = 1.0f; f.nw = act_threads(M, K) / 32;
        f.limit = limit;
        f.h = hg; f.h_ld = h_ld; f.h_al = h_al;
        f.u_al = (((uintptr_t)(x + K) & 15u) == 0 && x_ld % 8 == 0) ? 1 : 0;
        r4d_gg_mr_pin(p.h, p.r, p.mt, p.wn);
        const int rc = r4d_gg_mr_fold(nullptr, c, f);
        r4d_gg_mr_pin(0, 0, 0, 0);
        ++g_cases;
        if (rc != 0) {
            std::printf("FAIL fold %c %s K=%d N=%d M=%d pin %d/%d/%d/%d: rc %d\n", pro == 1 ? 'a' : 'e', t.name, K, N, M,
                        p.h, p.r, p.mt, p.wn, rc);
            ++g_fail;
            continue;
        }
        long long bh = 0, by = 0, fh = -1, fy = -1;
        for (long long i = 0; i < (long long)M * h_ld; ++i)
            if (hg[i] != hr[i]) { if (fh < 0) fh = i; ++bh; }
        for (long long i = 0; i < (long long)M * N; ++i)
            if (yb[(size_t)i] != yref[(size_t)i]) { if (fy < 0) fy = i; ++by; }
        if (bh || by) {
            std::printf("FAIL fold %c %s K=%d N=%d M=%d pin %d/%d/%d/%d (%s): h %lld of %lld differ (first m=%lld k=%lld: "
                        "%04x vs %04x), y %lld of %lld differ (first m=%lld n=%lld)\n", pro == 1 ? 'a' : 'e', t.name, K, N,
                        M, p.h, p.r, p.mt, p.wn, r4d_gg_mr_last(), bh, (long long)M * h_ld, fh < 0 ? -1 : fh / h_ld,
                        fh < 0 ? -1 : fh % h_ld, fh < 0 ? 0 : hg[fh], fh < 0 ? 0 : hr[fh], by, (long long)M * N,
                        fy < 0 ? -1 : fy / N, fy < 0 ? -1 : fy % N);
            ++g_fail;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    std::vector<int> Ms;
    if (quick) Ms = { 1, 3, 8, 9, 24, 33, 48 };
    else for (int M = 1; M <= 48; ++M) Ms.push_back(M);
    std::vector<Pin> pins = { { 1, 1, 0, 0 }, { 1, 1, 4, 1 }, { 2, 1, 0, 0 }, { 2, 1, 6, 2 }, { 2, 2, 0, 2 }, { 2, 2, 6, 0 },
                              { 4, 1, 0, 0 }, { 4, 1, 8, 1 }, { 4, 2, 0, 8 }, { 4, 2, 3, 0 }, { -1, 0, 0, 0 }, { 0, 0, 0, 0 } };
    if (quick) pins = { { 1, 1, 4, 1 }, { 2, 2, 0, 2 }, { 4, 2, 3, 0 }, { -1, 0, 0, 0 }, { 0, 0, 0, 0 } };
    const int N = 45;
    for (int pro = 1; pro <= 2; ++pro) {
        const std::vector<int> Ks = pro == 1 ? std::vector<int>{ 1280 } : std::vector<int>{ 768, 512 };
        for (const Ty& t : kTypes) {
            for (int K : Ks) {
                if (K % t.qk) continue;
                const std::vector<uint8_t> w = make_w(t, N, K);
                for (int M : Ms) run_case(pro, t, K, N, M, w, pins);
                /* the real shape under the rule and the table: q_b 8192 x 1280, shexp_down 5120 x K */
                const int NR = pro == 1 ? 8192 : 5120;
                const std::vector<uint8_t> wr = make_w(t, NR, K);
                for (int M : { 1, 6, 8, 25 }) run_case(pro, t, K, NR, M, wr, { { -1, 0, 0, 0 }, { 0, 0, 0, 0 } });
                std::printf("fold %c %s K=%d: done (%d cases, %d failures so far)\n", pro == 1 ? 'a' : 'e', t.name, K,
                            g_cases, g_fail);
                std::fflush(stdout);
            }
        }
    }
    std::printf("INFO reference h values: %lld, %lld of them NaN (inf rows through the norm / the clamp)\n", g_h_vals, g_h_nan);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
