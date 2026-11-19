// What a dense k-quant prefill GEMM's K loop can reach on gfx1201 at the card's own clock and power limit, given the
// instruction mix its arithmetic needs: bf16 WMMA (the published dense forms' v_wmma_f32_16x16x16_bf16) with V plain
// VALU slots, L ds_load_b128, S ds_store_b128 per eight WMMA and a block barrier every BR groups of eight -- the mix
// oracle (tp8 loop A5b, audit DECISION item 5). wmmamix.hip asked the same of fp8 WMMA with zero operands; this
// asks it of bf16 at random operands (the power state of a real GEMM, so TFLOP/s compare with r4d_selftest --perfwq),
// and adds the k-quant fold as it is computed: F = 1 makes every WMMA a group's chain from its init (the published
// Q6_K form's CINIT, SPG 1) followed by eight fmas of its result into a running accumulator (the per-group scale fold,
// left to the compiler, which pairs them as VOPD where the banks allow).
//
// Every row is eight independent WMMA tiles a wave, every block resident (the grid is the occupancy answer times the
// WGP count), ~1 s timed after ~0.3 s warm; the result is TFLOP/s and SIMD clocks a WMMA from the elapsed time and
// the card's mean sclk (hwmon freq1_input, sampled every 20 ms). No dependencies but F's: a ceiling for a kernel with
// that mix, not a prediction -- a kernel far below its row is held back by latency or scheduling, not by its mix.
//
//   clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 wmmabf.hip -o wmmabf
//   ./wmmabf [z]          z: zero operands (the clock is then real; the multipliers barely switch)
//   ./wmmabf s SECS [MINW] [STOP]  the card's sclk / power over SECS (or until file STOP exists) while another process runs
//
// Rows (the published forms' mixes from their K loops, tp8/devpf/scripts/isaloop.py on ds41-pp-dev-pf b95aa87's
// r4d_ds41_wmk.hip code object, per WMMA: q6_K 326 12.18 VALU slots / 808 LDS B / a barrier per 24 WMMA; q8_0 414
// 10.70 / 544 / 16; q5_K 9411 15.64 / 1104 / 16) and candidate byte-identical q6_K geometries (A: the fold's 4 VOPD
// slots a WMMA are fixed by the part order; decode 10 slots a B fragment over NM row tiles; A + chain-init LDS reads
// 1536 B a row tile and K step over NT column tiles).
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <thread>
#include <vector>

typedef float  f8v  __attribute__((ext_vector_type(8)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));

template <int LDS, int V, int L, int S, int BR, int F>
__global__ __launch_bounds__(256) void mix(const uint32_t* src, float* out, int iters) {
    __shared__ __attribute__((aligned(16))) char pad[LDS];
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    pad[threadIdx.x] = (char)threadIdx.x;
    __syncthreads();
    const uint32_t* p = src + ((blockIdx.x & 63u) * 32u + lane) * 64u;
    v8bf a[8], b[4];   /* eight A and four B fragments rotating (registers kept for the fold's tiles) */
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        u4v ua = { p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3] };
        a[i] = __builtin_bit_cast(v8bf, ua);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u4v ub = { p[32 + 4 * i], p[33 + 4 * i], p[34 + 4 * i], p[35 + 4 * i] };
        b[i] = __builtin_bit_cast(v8bf, ub);
    }
    f8v c[8], acc[8], ci[2];
#pragma unroll
    for (int i = 0; i < 8; ++i) { c[i] = (f8v)(0.0f); acc[i] = (f8v)(0.0f); }
    ci[0] = (f8v)(-160.0f * (float)(lane & 7u));
    ci[1] = (f8v)(-160.0f * (float)(lane >> 3));
    float f[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) f[i] = (float)(threadIdx.x + i);
    const float kk = 1.0000001f, k2 = 1e-7f;
    float sc = 1.0f + (float)lane * 1e-6f;
    u4v r[4];   /* the loads' destinations rotate (a ds_load does not read its destination) */
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = u4v{};
    // A fragment read's pattern: lane row (l & 15) at a 272-byte pitch (the staged slab's), half k by lane half.
    const uint32_t la = (uint32_t)(uintptr_t)pad + (lane & 15u) * 272u + (lane >> 4) * 16u + (wave & 3u) * 4352u;
    const uint32_t sa = (uint32_t)(uintptr_t)pad + threadIdx.x * 16u;
    const u4v sv = { threadIdx.x, lane, wave, 7u };
    for (int it = 0; it < iters; ++it) {
        if constexpr (F) {   /* the chain inits are opaque to the compiler each step (no WMMA hoisted out of the loop) */
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int q = 0; q < 8; ++q) asm volatile("" : "+v"(ci[j][q]));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if constexpr (F) {
                c[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[i], b[(i + 3) & 3], ci[i & 1]);
            } else {
                c[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[i], b[(i + 3) & 3], c[i]);
            }
#pragma unroll
            for (int v = i * V / 8; v < (i + 1) * V / 8; ++v)
                asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(f[v & 15]) : "v"(kk), "v"(k2));
#pragma unroll
            for (int l = i * L / 8; l < (i + 1) * L / 8; ++l)
                asm volatile("ds_load_b128 %0, %1 offset:%2" : "=v"(r[l & 3]) : "v"(la), "i"((l % 8) * 32));
#pragma unroll
            for (int s = i * S / 8; s < (i + 1) * S / 8; ++s)
                asm volatile("ds_store_b128 %0, %1 offset:%2" ::"v"(sa), "v"(sv), "i"((s % 4) * 4096));
        }
        if constexpr (F) {
            /* the fold: acc = fma(P, scale, acc), eight a lane a WMMA, after the step's WMMAs (the stock order) */
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int q = 0; q < 8; ++q) acc[i][q] = __builtin_fmaf(c[i][q], sc, acc[i][q]);
            sc = __builtin_fmaf(sc, kk, k2);
        }
        if (L > 0 || S > 0) asm volatile("s_wait_dscnt 0x0" ::: "memory");
        if constexpr (BR > 0) {
            if (it % BR == BR - 1) __syncthreads();
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int t = 0; t < 8; ++t) s += c[i][t] + acc[i][t];
#pragma unroll
    for (int i = 0; i < 16; ++i) s += f[i];
#pragma unroll
    for (int i = 0; i < 4; ++i) s += (float)(r[i].x ^ r[i].w);
    s += (float)pad[(threadIdx.x * 7u) % LDS];
    if (s == -1.0f) out[0] = s;
}

static long read_long(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

static std::string g_hw;

template <int LDS, int V, int L, int S, int BR, int F>
static void run(const char* name, const uint32_t* d, float* o) {
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);
    int per = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, mix<LDS, V, L, S, BR, F>, 256, 0);
    const int blocks = per * prop.multiProcessorCount;
    const int iters = 8000;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    (void)hipEventRecord(e0);
    hipLaunchKernelGGL((mix<LDS, V, L, S, BR, F>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms1 = 0;
    (void)hipEventElapsedTime(&ms1, e0, e1);
    const int reps = (int)(1000.0f / (ms1 > 0.01f ? ms1 : 0.01f)) + 1;
    for (int r = 0; r < reps / 3; ++r)
        hipLaunchKernelGGL((mix<LDS, V, L, S, BR, F>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipDeviceSynchronize();
    std::atomic<bool> stop{ false };
    double mhz = 0, watt = 0;
    int n = 0;
    std::thread th([&] {
        while (!stop) {
            const long f = read_long(g_hw + "/freq1_input");
            if (f > 0) { mhz += f / 1e6; watt += read_long(g_hw + "/power1_average") / 1e6; ++n; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    (void)hipEventRecord(e0);
    for (int r = 0; r < reps; ++r)
        hipLaunchKernelGGL((mix<LDS, V, L, S, BR, F>), dim3(blocks), dim3(256), 0, 0, d, o, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    stop = true;
    th.join();
    float ms = 0;
    (void)hipEventElapsedTime(&ms, e0, e1);
    const double clk = n ? mhz / n : 0.0;
    const double wmma = 8.0 * iters * (double)blocks * 8.0 * reps;
    const double simds = 4.0 * prop.multiProcessorCount;   /* 64 CUs x 2 SIMDs; multiProcessorCount = 32 WGPs */
    const double clk_per = simds * (double)ms * 1e-3 * clk * 1e6 / wmma;
    const double tf = wmma * 8192.0 / ((double)ms * 1e9);
    std::printf("A5MIX %-34s %4.1f waves/SIMD %5.0f MHz %4.0f W %6.1f TFLOP/s %6.2f clk/WMMA/SIMD (%5.1f%% of 16)\n", name,
                (double)blocks * 8.0 / simds, clk, n ? watt / n : 0.0, tf, clk_per, 1600.0 / clk_per);
    std::fflush(stdout);
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
}

int main(int argc, char** argv) {
    const bool zero = argc > 1 && argv[1][0] == 'z';
    char bus[64] = {};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), 0) == hipSuccess) {
        for (char* q = bus; *q; ++q) *q = (char)std::tolower(*q);
        const std::string base = std::string("/sys/bus/pci/devices/") + bus + "/hwmon";
        if (DIR* dd = opendir(base.c_str())) {
            while (dirent* e = readdir(dd))
                if (!std::strncmp(e->d_name, "hwmon", 5)) { g_hw = base + "/" + e->d_name; break; }
            closedir(dd);
        }
    }
    if (g_hw.empty()) { std::printf("no hwmon node for device 0: the clock cannot be read\n"); return 1; }
    if (argc > 1 && argv[1][0] == 's') {
        /* `wmmabf s SECONDS [MINW]`: the card's own sclk / power while another process runs (run-a5.sh: during r4d_selftest
         * --perfwq), every 20 ms; the mean over samples at >= MINW W (default 120: the GEMMs, not the host gaps between them). */
        const double secs = argc > 2 ? std::atof(argv[2]) : 60.0, minw = argc > 3 ? std::atof(argv[3]) : 120.0;
        double mz = 0, wz = 0, ma = 0;
        long n = 0, nb = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const char* stopf = argc > 4 ? argv[4] : nullptr;   /* stop early once this file exists */
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
            if (stopf) {
                if (FILE* q = std::fopen(stopf, "r")) { std::fclose(q); break; }
            }
            const long f = read_long(g_hw + "/freq1_input"), pw = read_long(g_hw + "/power1_average");
            if (f > 0) {
                ++n;
                ma += f / 1e6;
                if (pw / 1e6 >= minw) { ++nb; mz += f / 1e6; wz += pw / 1e6; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::printf("A5CLK %ld samples (mean %.0f MHz); %ld at >= %.0f W: mean %.0f MHz %.1f W\n", n, n ? ma / n : 0.0, nb, minw,
                    nb ? mz / nb : 0.0, nb ? wz / nb : 0.0);
        return 0;
    }
    const size_t nw = 64 * 32 * 64;
    std::vector<uint32_t> h(nw);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (uint32_t)x; };
    for (size_t i = 0; i < nw; ++i) {
        uint32_t v = 0;
        for (int hh = 0; hh < 2; ++hh) {   /* bf16 near one (2^-2 .. 2^1), random sign and mantissa: nothing overflows */
            const uint32_t r = rnd();
            const uint32_t e = 125u + (r >> 16) % 4u;
            v |= (((r & 0x8000u) | (e << 7) | (r & 0x7Fu)) & 0xFFFFu) << (16 * hh);
        }
        h[i] = zero ? 0u : v;
    }
    uint32_t* d;
    float* o;
    (void)hipMalloc(&d, nw * 4);
    (void)hipMalloc(&o, 4);
    (void)hipMemcpy(d, h.data(), nw * 4, hipMemcpyHostToDevice);
    std::printf("A5MIX bf16 WMMA mixes (per 8 WMMA: V plain VALU, L ds_load_b128, S ds_store_b128, barrier every BR groups; F the fold), %s operands, sensors %s, cap %.0f W\n",
                zero ? "zero" : "random", g_hw.c_str(), (double)read_long(g_hw + "/power1_cap") / 1e6);
    /* occupancy: LDS 65536 = 2 blocks a WGP = 4 waves a SIMD (the published q6_K 326: 250 VGPRs, 2 blocks); 40000 = 6 */
    run<65536, 0, 0, 0, 0, 0>("wmma only @4w", d, o);
    run<40000, 0, 0, 0, 0, 0>("wmma only @6w", d, o);
    run<65536, 32, 0, 0, 0, 0>("V32 (4 a WMMA)", d, o);
    run<65536, 64, 0, 0, 0, 0>("V64 (8)", d, o);
    run<65536, 97, 0, 0, 0, 0>("V97 (12.2 = q6_K 326)", d, o);
    run<65536, 0, 6, 0, 0, 0>("L6 (384 B a WMMA)", d, o);
    run<65536, 0, 12, 0, 0, 0>("L12 (768 B = q6_K 326)", d, o);
    run<65536, 0, 16, 0, 0, 0>("L16 (1024 B)", d, o);
    run<65536, 0, 0, 0, 3, 0>("barrier / 24 WMMA", d, o);
    run<65536, 0, 0, 0, 0, 1>("F fold (8 fma a WMMA, chain init)", d, o);
    /* the published forms' mixes (isaloop.py) */
    run<65536, 97, 12, 1, 3, 0>("MIX q6_K 326 @4w", d, o);
    run<40000, 97, 12, 1, 3, 0>("MIX q6_K 326 @6w", d, o);
    run<65536, 86, 8, 0, 2, 0>("MIX q8_0 414 @4w", d, o);
    run<65536, 125, 16, 2, 2, 0>("MIX q5_K 9411 @4w", d, o);
    run<65536, 65, 12, 1, 3, 1>("MIX q6_K 326 as F + V65 @4w", d, o);
    /* candidate byte-identical q6_K geometries (the fold kept: F + the rest as plain VALU) */
    run<65536, 44, 12, 1, 4, 1>("CAND q6_K NM4 NT2 (F + V44, L12)", d, o);
    run<65536, 64, 6, 1, 4, 1>("CAND q6_K NM2 NT4 (F + V64, L6)", d, o);
    run<65536, 22, 11, 2, 4, 1>("CAND q6_K NM2 NT4 shared B (F + V22, L11)", d, o);
    run<65536, 0, 6, 0, 0, 1>("FLOOR q6_K fold + L6 only", d, o);
    run<40000, 0, 6, 0, 0, 1>("FLOOR q6_K fold + L6 only @6w", d, o);
    return 0;
}
