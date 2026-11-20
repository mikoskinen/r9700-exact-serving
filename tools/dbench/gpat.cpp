/* gpat (DRAFTER track 20.11.2026 round 30): the WEIGHT-LOAD PATTERN of the part-order dense form (gg_wk3_kernel, q8_0) alone --
 * the same grid, waves, rows and slabs as a call, only the loads (every word XOR-folded into one value, no math, no staging) --
 * in four stored layouts of the same bytes, timed by the card (wall_clock64 at each block's entry / exit; span = last exit -
 * first entry; a 256 MiB evicting read before every timed launch; p50 of R):
 *   sc  today's row-major GGUF q8_0 (34-B blocks): a lane (c16, h) loads, per block of its slab, the u16 scale + two 8-B words
 *       at +2 + 8h and +18 + 8h of its row (Wq<GG_Q8_0>::load1: 2-byte-aligned 8-B loads, 16 rows a wave);
 *   sp  split planes: quants row-major [N][K] with a lane's 16 bytes of a block contiguous and 16-B aligned + the scales in a
 *       plane of their own [N][K / 32] (one aligned 8-B load a lane a slab);
 *   il  tile-interleaved: a (16-row tile, slab) is 2,176 contiguous bytes, lane L's 64 quant bytes at 64 L, the 16 rows' 4 scales
 *       after them (lane c16 loads 8 B): 4 x 16-B + 1 x 8-B load a lane a slab, every load contiguous across the wave;
 *   st  a plain stream of the same bytes at the same grid (the floor).
 * Shapes = the drafter's stage linears + main_proj + the head (q6_K's 210-B blocks: only st / il-q6 approximations).
 *   gpat [R] [M (row tiles 1..3: only the grid's KP rule)] */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ __launch_bounds__(256) void flush(const uint4* p, size_t n, uint32_t* o) {
    uint32_t a = 0;
    for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) o[0] = a;
}
struct G { const uint8_t* w; long long N, K; int KP; uint32_t* o; unsigned long long* st; };
typedef __attribute__((address_space(1))) const uint16_t* gu16;
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(1))) const u32x2* gu64a2 __attribute__((aligned(2)));

/* MODE 0 sc, 1 sp, 2 il; a block = KP runs x NS = 8 / KP column groups of waves, a wave 16 rows, its run's S / KP slabs */
template <int MODE>
__global__ __launch_bounds__(256) void pat(G g) {
    const unsigned long long t0 = wall_clock64();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, KP = g.KP, NS = 8 / KP;
    const int kp = wave / NS, ns = wave % NS, c16 = lane & 15, h = lane >> 4;
    const long long S = g.K / 128, SW = S / KP, sb = kp * SW;
    const long long n = (long long)blockIdx.x * 16 * NS + ns * 16 + c16;
    const long long nn = n < g.N ? n : g.N - 1;
    const long long tile = nn / 16;
    uint32_t a = 0;
    for (long long t = 0; t < SW; ++t) {
        const long long s = sb + t;
        if constexpr (MODE == 0) {
            const uint8_t* row = g.w + nn * (g.K / 32 * 34);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint8_t* b = row + (4 * s + i) * 34;
                const uint16_t sc = *(gu16)(const void*)b;
                const u32x2 q0 = *(gu64a2)(const void*)(b + 2 + 8 * h);
                const u32x2 q1 = *(gu64a2)(const void*)(b + 18 + 8 * h);
                a ^= sc ^ q0.x ^ q0.y ^ q1.x ^ q1.y;
            }
        } else if constexpr (MODE == 1) {
            const uint8_t* q = g.w + nn * g.K + s * 128 + 32 * 0 + 16 * h;   /* block i at + 32 i: a lane's 16 B contiguous */
#pragma unroll
            for (int i = 0; i < 4; ++i) { const uint4 v = *(const uint4*)(q + 32 * i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
            const uint2 sc = *(const uint2*)(g.w + g.N * g.K + nn * (g.K / 32 * 2) + s * 8);
            a ^= sc.x ^ sc.y;
        } else {
            const uint8_t* tb = g.w + (tile * S + s) * 2176;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const uint4 v = *(const uint4*)(tb + 64 * lane + 16 * i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
            const uint2 sc = *(const uint2*)(tb + 2048 + 8 * c16);
            a ^= sc.x ^ sc.y;
        }
    }
    if (a == 0x12345678u) g.o[0] = a;
    __syncthreads();
    if (threadIdx.x == 0) { g.st[2 * blockIdx.x] = t0; g.st[2 * blockIdx.x + 1] = wall_clock64(); }
}

/* q6_K (the head, 210-B blocks; a slab = half a block: ql 64 B at 64 c, qh 32 B at 128 + 32 c, scales 8 B at 192 + 8 c, d at 208):
 * MODE 0 sc6 = Wq<GG_Q6_K>::load (4 + 2 + 1 2-B-aligned 8-B loads + a u16 a lane), MODE 1 al6 = the same bytes through 16-B ALIGNED
 * loads (a lane takes its half of each region: ql 32 B + qh 16 B + the scale / d words, each range rounded out to 16-B words),
 * MODE 2 il6 = a tile-interleaved (16 rows x slab = 1,680 contiguous bytes) form. NT tiles a wave, KP 1. */
template <int MODE, int NT>
__global__ __launch_bounds__(256) void pat6(G g) {
    const unsigned long long t0 = wall_clock64();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, c16 = lane & 15, h = lane >> 4;
    const long long S = g.K / 128, RB = g.K / 256 * 210;
    uint32_t a = 0;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const long long n = ((long long)blockIdx.x * 8 + wave) * 16 * NT + nt * 16 + c16;
        const long long nn = n < g.N ? n : g.N - 1, tile = nn / 16;
        const uint8_t* row = g.w + nn * RB;
        for (long long s = 0; s < S; ++s) {
            const uint8_t* b = row + (s >> 1) * 210;
            const int c = (int)(s & 1);
            if constexpr (MODE == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { const u32x2 q = *(gu64a2)(const void*)(b + 64 * c + 32 * (i >> 1) + 16 * (i & 1) + 8 * h); a ^= q.x ^ q.y; }
#pragma unroll
                for (int i = 0; i < 2; ++i) { const u32x2 q = *(gu64a2)(const void*)(b + 128 + 32 * c + 16 * i + 8 * h); a ^= q.x ^ q.y; }
                const u32x2 sc = *(gu64a2)(const void*)(b + 192 + 8 * c);
                a ^= sc.x ^ sc.y ^ *(gu16)(const void*)(b + 208);
            } else if constexpr (MODE == 1) {
                auto rng = [&](long long off, int len) {   /* the 16-B words covering [off, off + len) */
                    const uintptr_t lo = ((uintptr_t)(b + off)) & ~(uintptr_t)15, hi = ((uintptr_t)(b + off + len) + 15) & ~(uintptr_t)15;
                    for (uintptr_t p = lo; p < hi; p += 16) { const uint4 v = *(const uint4*)p; a ^= v.x ^ v.y ^ v.z ^ v.w; }
                };
                rng(64 * c + 32 * h, 32);
                rng(128 + 32 * c + 16 * h, 16);
                if (h == 0) rng(192 + 8 * c, 8); else rng(208, 2);
            } else {
                const uint8_t* tb = g.w + (tile * S + s) * 1680;
                for (int i = lane; i < 105; i += 32) { const uint4 v = *(const uint4*)(tb + 16 * i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
            }
        }
    }
    if (a == 0x12345678u) g.o[0] = a;
    __syncthreads();
    if (threadIdx.x == 0) { g.st[2 * blockIdx.x] = t0; g.st[2 * blockIdx.x + 1] = wall_clock64(); }
}
__global__ __launch_bounds__(256) void rd(const uint4* p, long long n16, uint32_t* o, unsigned long long* st) {
    const unsigned long long t0 = wall_clock64();
    const long long per = (n16 + gridDim.x - 1) / gridDim.x;
    const long long b0 = (long long)blockIdx.x * per, b1 = std::min(b0 + per, n16);
    uint32_t a = 0;
    for (long long i = b0 + threadIdx.x; i < b1; i += 256) { uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) o[0] = a;
    __syncthreads();
    if (threadIdx.x == 0) { st[2 * blockIdx.x] = t0; st[2 * blockIdx.x + 1] = wall_clock64(); }
}
int main(int argc, char** argv) {
    const int R = argc > 1 ? std::atoi(argv[1]) : 30;
    const size_t FB = 256ull << 20;
    uint4* f; uint8_t* w; uint32_t* o; unsigned long long* st;
    CK(hipMalloc((void**)&f, FB)); CK(hipMemset(f, 1, FB));
    CK(hipMalloc((void**)&w, 160ull << 20)); CK(hipMemset(w, 3, 160ull << 20));
    CK(hipMalloc((void**)&o, 64)); CK(hipMalloc((void**)&st, 2 * 8192 * 8));
    std::vector<unsigned long long> hs(2 * 8192);
    struct Sh { const char* name; long long N, K; int KP; };
    /* KP as gg_wk3 takes it at 17..32 rows under ds41_wmk 11 (G2: P / 2; o_b KP 1 by DQ8's bit 24; q_b P 2 -> KP 1) */
    const Sh shapes[] = { { "q_a+kv 1792x5120", 1792, 5120, 4 }, { "q_b 8192x1280", 8192, 1280, 1 }, { "o_a 2048x4096", 2048, 4096, 4 },
                          { "o_b 5120x2048 KP1", 5120, 2048, 1 }, { "o_b 5120x2048 KP4", 5120, 2048, 4 },
                          { "main_proj 5120x15360", 5120, 15360, 4 }, { "q_a+kv KP1", 1792, 5120, 1 }, { "q_a+kv KP2", 1792, 5120, 2 } };
    auto span = [&](int G_) {
        CK(hipMemcpy(hs.data(), st, 2 * G_ * 8, hipMemcpyDeviceToHost));
        unsigned long long a = ~0ull, b = 0;
        for (int i = 0; i < G_; ++i) { a = std::min(a, hs[2 * i]); b = std::max(b, hs[2 * i + 1]); }
        return (double)(b - a) / 100.0;
    };
    for (const Sh& sh : shapes) {
        const double bytes = (double)sh.N * (sh.K / 32) * 34;
        const int NS = 8 / sh.KP, grid = (int)((sh.N + 16 * NS - 1) / (16 * NS));
        G g{ w, sh.N, sh.K, sh.KP, o, st };
        std::string line;
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<double> sp;
            for (int r = 0; r < R + 2; ++r) {
                flush<<<1024, 256>>>(f, FB / 16, o);
                if (mode == 0) pat<0><<<grid, 256>>>(g);
                else if (mode == 1) pat<1><<<grid, 256>>>(g);
                else if (mode == 2) pat<2><<<grid, 256>>>(g);
                else rd<<<grid, 256>>>((const uint4*)w, (long long)(bytes / 16), o, st);
                CK(hipDeviceSynchronize());
                if (r >= 2) sp.push_back(span(grid));
            }
            std::sort(sp.begin(), sp.end());
            const double us = sp[sp.size() / 2];
            static const char* nm[] = { "sc", "sp", "il", "st" };
            line += std::string(" ") + nm[mode] + " " + std::to_string(us).substr(0, 6) + " us " + std::to_string(bytes / 1e3 / us).substr(0, 5) + " GB/s |";
        }
        std::printf("GPAT %-22s %6.2f MB grid %4d KP %d:%s\n", sh.name, bytes / 1e6, grid, sh.KP, line.c_str());
        std::fflush(stdout);
    }
    for (int nt6 : { 2, 1 }) {
        const long long N6 = 32320, K6 = 5120;
        const double bytes = (double)N6 * (K6 / 256) * 210;
        const int grid = (int)((N6 + 128 * nt6 - 1) / (128 * nt6));
        G g{ w, N6, K6, 1, o, st };
        std::string line;
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<double> sp;
            for (int r = 0; r < R + 2; ++r) {
                flush<<<1024, 256>>>(f, FB / 16, o);
                if (mode == 3) rd<<<grid, 256>>>((const uint4*)w, (long long)(bytes / 16), o, st);
                else if (nt6 == 2) { if (mode == 0) pat6<0, 2><<<grid, 256>>>(g); else if (mode == 1) pat6<1, 2><<<grid, 256>>>(g); else pat6<2, 2><<<grid, 256>>>(g); }
                else { if (mode == 0) pat6<0, 1><<<grid, 256>>>(g); else if (mode == 1) pat6<1, 1><<<grid, 256>>>(g); else pat6<2, 1><<<grid, 256>>>(g); }
                CK(hipDeviceSynchronize());
                if (r >= 2) sp.push_back(span(grid));
            }
            std::sort(sp.begin(), sp.end());
            const double us = sp[sp.size() / 2];
            static const char* nm[] = { "sc6", "al6", "il6", "st" };
            line += std::string(" ") + nm[mode] + " " + std::to_string(us).substr(0, 6) + " us " + std::to_string(bytes / 1e3 / us).substr(0, 5) + " GB/s |";
        }
        std::printf("GPAT6 head 32320x5120 q6_K %6.2f MB grid %4d NT %d:%s\n", bytes / 1e6, grid, nt6, line.c_str());
        std::fflush(stdout);
    }
    return 0;
}
