/* tests/r4d_emu: lever VA "bf16" (RADIANCE_DS41_ROUTER_BF16) on the host SIMT emulator.
 *
 * DeepSeek-V4.1's MoE routers and its ratio > 1 compressors are BF16 in the checkpoint and ran as
 * f32 weights (gemm_nt_f32, r4d_glm.hip). Stored as bf16, the kernel widens each weight to the f32
 * value it stands for before the same fmaf in the same order (gemm_f32_wval), so the output must be
 * the f32 storage's byte for byte. Every case runs the same bf16-representable weights twice --
 * once as f32 (today), once as bf16 (the lever) -- at DeepSeek-V4.1's real shapes (K 5120: the
 * trunk router N 384, its TP4 column split N 96, the DSpark router N 128, the fused compressor
 * N 1024) and at row counts that reach every tile form the launcher picks (4x1 at N <= 8; 2x2 with
 * 16 steps in flight to 16 rows and below 64 at N < 128; 8x1 with 8 steps below 64; 8x4 from 64),
 * with K short of and past whole groups of steps. The outputs' pitch padding must stay untouched.
 * libavx's host form and libref's are held to their own f32 storage the same way (test_ds41vram is
 * a host binary: both are plain C++). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

namespace {

struct Shape {
    int M, N, K;
    const char* what;
};

RadTensor mat(void* d, uint32_t dt, int64_t r, int64_t c, int64_t pitch) {
    RadTensor t{};
    t.data = d;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = r;
    t.shape[1] = c;
    t.stride[0] = pitch;
    t.stride[1] = 1;
    return t;
}

/* One gemm_nt_f32 launch; y is [M, yp] filled with a sentinel first. */
int run(const std::vector<uint16_t>& x, int xp, const void* w, uint32_t wdt, int M, int N, int K,
        std::vector<float>& y, int yp) {
    y.assign((size_t)M * yp, -7.25f);   /* before the view: assign may move the storage */
    RadTensor t[3] = { mat((void*)x.data(), RAD_BF16, M, K, xp), mat((void*)w, wdt, N, K, K),
                       mat(y.data(), RAD_F32, M, N, yp) };
    RadParam p[4]{};
    p[0] = { "M", RAD_P_INT, M };
    p[1] = { "N", RAD_P_INT, N };
    p[2] = { "K", RAD_P_INT, K };
    p[3].key = "dtype";
    p[3].kind = RAD_P_STR;
    p[3].sval = "bf16";
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 4;
    return r4d_rad_gemm_nt_f32(&a, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    /* --quick: the decode shapes only (a few seconds); the default runs every shape. */
    const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.f, 1.f);
    const Shape shapes[] = {
        /* the trunk router, replicated: decode rows at each form, M 1..8 first */
        { 1, 384, 5120, "router" },   { 2, 384, 5120, "router" },   { 3, 384, 5120, "router" },
        { 5, 384, 5120, "router" },   { 8, 384, 5120, "router" },   { 16, 384, 5120, "router" },
        { 17, 384, 5120, "router" },  { 48, 384, 5120, "router" },  { 63, 384, 5120, "router" },
        { 64, 384, 5120, "router" },  { 70, 384, 5120, "router" },
        /* its TP4 column split (SplitF32Proj, from 256 rows: the part's N) */
        { 256, 96, 5120, "router part" }, { 33, 96, 5120, "router part" },
        /* the DSpark drafter's router */
        { 6, 128, 5120, "dspark router" }, { 48, 128, 5120, "dspark router" },
        /* the compressor's fused kv|gate projection, ratio 2 */
        { 1, 1024, 5120, "compressor" }, { 9, 1024, 5120, "compressor" }, { 65, 1024, 5120, "compressor" },
        /* K off the step groups, N at 8 and below (the 4x1 form), a tail column */
        { 7, 8, 5120, "4x1" }, { 5, 3, 1040, "4x1" }, { 20, 37, 1061, "tail" }, { 66, 131, 529, "tail" },
    };
    int cases = 0, fails = 0;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K, xp = K + 5, yp = N + 3;
        if (quick && M > 17) continue;
        std::vector<uint16_t> x((size_t)M * xp), wb((size_t)N * K);
        std::vector<float> wf((size_t)N * K);
        for (auto& v : x) v = f2b(nd(rng));
        /* bf16 weights at router magnitudes, a few signed zeros and subnormal-scale values among them */
        for (size_t i = 0; i < wb.size(); ++i) {
            float v = nd(rng) * 0.03f;
            if (i % 997 == 3) v = -0.0f;
            if (i % 1009 == 5) v = 1e-39f;
            wb[i] = f2b(v);
            wf[i] = b2f(wb[i]);
        }
        std::vector<float> y_f32, y_b16;
        ++cases;
        const int rf = run(x, xp, wf.data(), RAD_F32, M, N, K, y_f32, yp);
        const char* kf = emu_last_kernel;
        const int rb = run(x, xp, wb.data(), RAD_BF16, M, N, K, y_b16, yp);
        int bad = 0, pad = 0;
        if (rf == RAD_OK && rb == RAD_OK) {
            bad = std::memcmp(y_f32.data(), y_b16.data(), y_f32.size() * 4) != 0;
            for (int r = 0; r < M; ++r)
                for (int c = N; c < yp; ++c) pad += y_b16[(size_t)r * yp + c] != -7.25f;
        }
        std::printf("gemm_nt_f32 %-14s M %3d N %4d K %4d  %-38s f32 vs bf16 weights: %s%s (rc %d / %d)\n", sh.what,
                    M, N, K, kf ? kf : "?", rf || rb ? "LAUNCH FAILED" : bad ? "DIFFER" : "byte-identical",
                    pad ? ", PADDING TOUCHED" : "", rf, rb);
        if (rf != RAD_OK || rb != RAD_OK || bad || pad) ++fails;
    }
    std::printf("ds41 vram bf16: %d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
