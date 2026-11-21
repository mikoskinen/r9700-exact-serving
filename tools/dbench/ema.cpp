/* tools/dbench/ema.cpp (DRAFTER track 21.11.2026 round 59, lever EMA = DAR a) -- one card: the DSpark query pass's prologue
 * after the embedding, today's form vs lever EMA's, through libr4d's own exported launchers (dlopen'd):
 *
 *   today  hc x cast bf16 -> bf16 (r4d_rad_cast's body, r4d_p2p_copy2d): x [T, D] into h0's column strips [T, hc x D]
 *   EMA    gather_rows (r4d_gather_rows_bf16) x -> xa [S + 1, D] by ia = [1, 0, blk, 2 blk, ...], then hc x gather_rows
 *          xa -> h0's strips by ie[r] = r % blk ? 0 : 1 + r / blk (deepseek41.cpp dsp_declare's planes, the same formula)
 *
 * x holds the core's staging (rad_builder.h RadDrafterDecl::mask_token): an anchor row a sequence, then blk - 1 copies of one
 * mask row (bf16 patterns drawn from every exponent, NaN payloads included). The all-reduce between the two gathers is the
 * identity on one card (its per-element fold is the same at S + 1 rows as at T: r4d_ar_entry.hip). Checks: EMA's h0 ==
 * today's h0 byte for byte at every (S, blk); times: back-to-back on one stream, event-timed, p50 of 7 x ITERS.
 * Usage: ema LIBR4D [ITERS]. */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Copy2d = int (*)(long, long, long, long, long, long, long);
using Gather = int (*)(long, long, long, int, int, int, int, int, long);

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint16_t rbits() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint16_t)(rng >> 21); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ema LIBR4D [ITERS]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 400;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto copy2d = (Copy2d)dlsym(h, "r4d_p2p_copy2d");
    auto gather = (Gather)dlsym(h, "r4d_gather_rows_bf16");
    if (!copy2d || !gather) { std::fprintf(stderr, "symbols missing\n"); return 2; }
    const int D = 5120, HC = 4;
    struct Case { int S, blk; const char* what; };
    const Case cases[] = { { 16, 2, "c32 lane (16 x 2 short block)" }, { 8, 5, "c16 lane (8 x 5)" },
                           { 16, 5, "one machine c16 (16 x 5)" }, { 1, 5, "1 x 5" }, { 3, 2, "3 x 2" },
                           { 32, 5, "32 x 5 (Q 160)" } };
    hipStream_t st;
    hc(hipStreamCreate(&st), "stream");
    hipEvent_t e0, e1;
    hc(hipEventCreate(&e0), "ev");
    hc(hipEventCreate(&e1), "ev");
    int bad_cases = 0;
    for (const Case& k : cases) {
        const int S = k.S, blk = k.blk, T = S * blk, Q = T;
        std::vector<uint16_t> x((size_t)T * D), noise(D);
        for (auto& v : noise) v = rbits();
        for (int r = 0; r < T; ++r)
            for (int j = 0; j < D; ++j) x[(size_t)r * D + j] = r % blk ? noise[j] : rbits();
        std::vector<int32_t> ia(Q, 0), ie(Q, 0);
        ia[0] = 1;
        for (int s = 0; s + 1 < Q; ++s) ia[1 + s] = s * blk < Q ? s * blk : 0;
        for (int r = 0; r < Q; ++r) ie[r] = r % blk ? 0 : 1 + r / blk;
        void *dx, *dxa, *dh0, *dh1, *dia, *die;
        hc(hipMalloc(&dx, (size_t)T * D * 2), "x");
        hc(hipMalloc(&dxa, (size_t)(S + 1) * D * 2), "xa");
        hc(hipMalloc(&dh0, (size_t)T * HC * D * 2), "h0");
        hc(hipMalloc(&dh1, (size_t)T * HC * D * 2), "h1");
        hc(hipMalloc(&dia, (size_t)Q * 4), "ia");
        hc(hipMalloc(&die, (size_t)Q * 4), "ie");
        hc(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice), "x up");
        hc(hipMemcpy(dia, ia.data(), ia.size() * 4, hipMemcpyHostToDevice), "ia up");
        hc(hipMemcpy(die, ie.data(), ie.size() * 4, hipMemcpyHostToDevice), "ie up");
        hc(hipMemset(dh0, 0xA5, (size_t)T * HC * D * 2), "h0 set");
        hc(hipMemset(dh1, 0x5A, (size_t)T * HC * D * 2), "h1 set");
        const long sl = (long)(uintptr_t)st;
        auto today = [&] {
            for (int i = 0; i < HC; ++i)
                if (copy2d((long)(uintptr_t)dh0 + (long)i * D * 2, (long)(uintptr_t)dx, (long)HC * D * 2, (long)D * 2,
                           (long)D * 2, T, sl) != 0) { std::fprintf(stderr, "copy2d failed\n"); std::exit(3); }
        };
        auto ema = [&] {
            if (gather((long)(uintptr_t)dx, (long)(uintptr_t)dia, (long)(uintptr_t)dxa, S + 1, D, T, D, D, sl) != 0) {
                std::fprintf(stderr, "gather 1 failed\n"); std::exit(3);
            }
            for (int i = 0; i < HC; ++i)
                if (gather((long)(uintptr_t)dxa, (long)(uintptr_t)die, (long)(uintptr_t)dh1 + (long)i * D * 2, T, D, S + 1,
                           D, HC * D, sl) != 0) { std::fprintf(stderr, "gather 2 failed\n"); std::exit(3); }
        };
        today();
        ema();
        hc(hipStreamSynchronize(st), "sync");
        std::vector<uint16_t> a((size_t)T * HC * D), b((size_t)T * HC * D);
        hc(hipMemcpy(a.data(), dh0, a.size() * 2, hipMemcpyDeviceToHost), "h0 down");
        hc(hipMemcpy(b.data(), dh1, b.size() * 2, hipMemcpyDeviceToHost), "h1 down");
        long diff = 0, wrong_src = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        for (int r = 0; r < T; ++r)   /* today's h0 = x's row in every strip (the oracle of both) */
            for (int i = 0; i < HC; ++i)
                wrong_src += std::memcmp(&a[((size_t)r * HC + i) * D], &x[(size_t)r * D], (size_t)D * 2) != 0;
        auto timeit = [&](auto&& f) {
            std::vector<float> v;
            for (int rep = 0; rep < 7; ++rep) {
                for (int w = 0; w < 20; ++w) f();
                hc(hipEventRecord(e0, st), "rec");
                for (int it = 0; it < iters; ++it) f();
                hc(hipEventRecord(e1, st), "rec");
                hc(hipEventSynchronize(e1), "evsync");
                float ms = 0;
                hc(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                v.push_back(ms * 1000.f / iters);
            }
            std::sort(v.begin(), v.end());
            return v[3];
        };
        const float ta = timeit(today), tb = timeit(ema), ta2 = timeit(today), tb2 = timeit(ema);
        /* the all-reduce bytes (A, LEVERS RSF 204's four-card slope: 9.35 us + 0.0896 us a KB) */
        const double kb_t = (double)T * D * 2 / 1024, kb_e = (double)(S + 1) * D * 2 / 1024;
        std::printf("S %2d blk %d T %3d (%s): h0 EMA vs today %ld of %zu bf16 differ, today vs x rows %ld of %d strips wrong; "
                    "us a prologue (A B A B) today %.2f / %.2f (%d launches) EMA %.2f / %.2f (%d launches); "
                    "AR rows %d -> %d (A: %.1f -> %.1f us exact wire)\n",
                    S, blk, T, k.what, diff, a.size(), wrong_src, T * HC, ta, ta2, HC, tb, tb2, HC + 1, T, S + 1,
                    9.35 + 0.0896 * kb_t, 9.35 + 0.0896 * kb_e);
        bad_cases += diff != 0 || wrong_src != 0;
        for (void* p : { dx, dxa, dh0, dh1, dia, die }) (void)hipFree(p);
    }
    /* one gather_rows bf16 of M rows of 5120 (contiguous, a reversed index), back to back: the body libr4d picks at M */
    {
        const int Mmax = 64;
        void *gx, *gy, *gi;
        hc(hipMalloc(&gx, (size_t)Mmax * D * 2), "gx");
        hc(hipMalloc(&gy, (size_t)Mmax * D * 2), "gy");
        hc(hipMalloc(&gi, (size_t)Mmax * 4), "gi");
        std::vector<int32_t> id(Mmax);
        for (int i = 0; i < Mmax; ++i) id[i] = Mmax - 1 - i;
        hc(hipMemcpy(gi, id.data(), id.size() * 4, hipMemcpyHostToDevice), "gi up");
        std::printf("gather_rows bf16 n 5120, us a call back to back (p50 of 7 x %d):", iters);
        for (int M : { 1, 2, 4, 8, 16, 24, 32, 40, 48, 49, 64 }) {
            std::vector<float> v;
            for (int rep = 0; rep < 7; ++rep) {
                for (int w = 0; w < 20; ++w)
                    (void)gather((long)(uintptr_t)gx, (long)(uintptr_t)gi, (long)(uintptr_t)gy, M, D, Mmax, D, D, (long)(uintptr_t)st);
                hc(hipEventRecord(e0, st), "rec");
                for (int it = 0; it < iters; ++it)
                    (void)gather((long)(uintptr_t)gx, (long)(uintptr_t)gi, (long)(uintptr_t)gy, M, D, Mmax, D, D, (long)(uintptr_t)st);
                hc(hipEventRecord(e1, st), "rec");
                hc(hipEventSynchronize(e1), "evsync");
                float ms = 0;
                hc(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                v.push_back(ms * 1000.f / iters);
            }
            std::sort(v.begin(), v.end());
            std::printf(" M%d %.2f", M, v[3]);
        }
        std::printf("\n");
        for (void* p : { gx, gy, gi }) (void)hipFree(p);
    }
    std::printf("%s: %d of %zu cases differ\n", bad_cases ? "FAIL" : "PASS", bad_cases, sizeof(cases) / sizeof(cases[0]));
    return bad_cases ? 1 : 0;
}
