/* patd (DRAFTER track 19.11.2026 round 12): one card, the read patterns for lever DEP's down call (the DSpark drafter's
 * routed MXFP4 down under DEP: K 576 = 18 blocks, rows of 306 B, 5120 rows an expert = 1.57 MB, ~33 live of 128 a stage),
 * timed as GB/s of the live experts' bytes. A block owns (expert, 16 WN rows); a wave owns 16 rows (one WMMA tile).
 *  W U:  lane (n, h) reads the 16-byte dword window at (row + 17 b) & ~3 + 4 h, U blocks in flight (xm_wm WR 1 today)
 *  C:    the wave's 16 rows as one contiguous range (16 x 306 = 4896 B), 16-B aligned chunks, lane j + 32 i, all in flight
 *  CL:   C, then the chunks into LDS (a wave's own region) and the 18 windows a lane read back from LDS (the loader form)
 *  B:    the block's 16 WN rows as one contiguous range, chunks over all threads, all in flight
 * 3 stages' own weights (3 x 128 experts = 602 MB >> the 64 MB MALL), a pass = 3 launches. usage: patd [live] [iters]
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); return 2; } } while (0)

constexpr int kRows = 5120, kRB = 306, kUK = 18, kE = 128;
typedef __attribute__((address_space(1))) const uint32_t* gu32;
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u4 ld16(const uint8_t* p) {
    const gu32 g = (gu32)(const void*)p;
    u4 v; v[0] = g[0]; v[1] = g[1]; v[2] = g[2]; v[3] = g[3]; return v;
}
__device__ __forceinline__ u4 ld16a(const uint8_t* p) { return *(const u4*)(const void*)p; }

template <int P, int U, int WN>
__global__ __launch_bounds__(256) void patd(const uint8_t* const* ex, uint32_t* out) {
    constexpr int RBK = 16 * WN, NRG = kRows / RBK;
    const int item = blockIdx.x, e = item / NRG, rg = item % NRG;
    const uint8_t* base = ex[e] + (size_t)rg * RBK * kRB;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nl = lane & 15, h = lane >> 4;
    uint32_t acc = 0;
    if constexpr (P == 1) {
        const uint8_t* r0 = base + (size_t)(wave * 16 + nl) * kRB;
        for (int b = 0; b < kUK; b += U) {
            u4 v[U];
#pragma unroll
            for (int k = 0; k < U; ++k) {
                const int bb = b + k < kUK ? b + k : kUK - 1;
                v[k] = ld16((const uint8_t*)(((uintptr_t)(r0 + 17 * bb) & ~(uintptr_t)3) + 4 * h));
            }
#pragma unroll
            for (int k = 0; k < U; ++k) acc ^= v[k][0] ^ v[k][1] ^ v[k][2] ^ v[k][3];
        }
    } else if constexpr (P == 2 || P == 3) {
        /* the wave's range [g0, g0 + 4896) as aligned 16-B chunks from ga = g0 & ~15 (+1 chunk: the windows read <= 3 B past) */
        const uint8_t* g0 = base + (size_t)(wave * 16) * kRB;
        const uintptr_t ga = (uintptr_t)g0 & ~(uintptr_t)15;
        const int nc = (int)(((uintptr_t)g0 + 16 * kRB + 3 - ga + 15) / 16);   /* <= 308 */
        constexpr int NCL = (16 * kRB + 3 + 15 + 15) / 16 / 32 + 1;            /* chunks a lane, 10 */
        u4 v[NCL];
#pragma unroll
        for (int i = 0; i < NCL; ++i) {
            const int c = lane + 32 * i;
            v[i] = c < nc ? ld16a((const uint8_t*)(ga + 16 * (uintptr_t)c)) : u4{0u, 0u, 0u, 0u};
        }
        if constexpr (P == 2) {
#pragma unroll
            for (int i = 0; i < NCL; ++i) acc ^= v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
        } else {
            extern __shared__ u4 lds[];
            uint8_t* reg = (uint8_t*)lds + wave * (NCL * 32 * 16);
#pragma unroll
            for (int i = 0; i < NCL; ++i) {
                const int c = lane + 32 * i;
                if (c < nc) *(u4*)(reg + 16 * c) = v[i];
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            const uintptr_t rrow = (uintptr_t)g0 + (uintptr_t)nl * kRB;
#pragma unroll
            for (int b = 0; b < kUK; ++b) {
                const uintptr_t pa = ((rrow + 17 * b) & ~(uintptr_t)3) + 4 * h;
                const uint32_t* q = (const uint32_t*)(reg + (pa - ga));
                acc ^= q[0] ^ q[1] ^ q[2] ^ q[3];
            }
        }
    } else {
        /* the block's 16 WN rows as one range, aligned chunks over all threads */
        const uintptr_t ga = (uintptr_t)base & ~(uintptr_t)15;
        const int nc = (int)(((uintptr_t)base + RBK * kRB + 3 - ga + 15) / 16);
        constexpr int NT = 32 * WN, NCL = (RBK * kRB + 3 + 30) / 16 / NT + 1;
        u4 v[NCL];
#pragma unroll
        for (int i = 0; i < NCL; ++i) {
            const int c = (int)threadIdx.x + NT * i;
            v[i] = c < nc ? ld16a((const uint8_t*)(ga + 16 * (uintptr_t)c)) : u4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < NCL; ++i) acc ^= v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

int main(int argc, char** argv) {
    const int L = argc > 1 ? std::atoi(argv[1]) : 33, IT = argc > 2 ? std::atoi(argv[2]) : 20;
    const size_t eb = (size_t)kRows * kRB;
    uint8_t* w; CK(hipMalloc((void**)&w, eb * kE * 3 + 64));
    CK(hipMemset(w, 3, eb * kE * 3 + 64));
    uint32_t* o; CK(hipMalloc((void**)&o, 64));
    std::vector<const uint8_t*> hp;
    for (int st = 0; st < 3; ++st)
        for (int i = 0; i < L; ++i) hp.push_back(w + eb * (size_t)(st * kE + (i * 37 + st * 11) % kE));
    const uint8_t** dp; CK(hipMalloc((void**)&dp, hp.size() * sizeof(void*)));
    CK(hipMemcpy(dp, hp.data(), hp.size() * sizeof(void*), hipMemcpyHostToDevice));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    auto run = [&](const char* nm, void (*k)(const uint8_t* const*, uint32_t*), int wn, unsigned lds) {
        const unsigned grid = (unsigned)(L * (kRows / (16 * wn)));
        for (int w0 = 0; w0 < 3; ++w0) for (int st = 0; st < 3; ++st) k<<<grid, 32 * wn, lds>>>(dp + st * L, o);
        CK(hipEventRecord(a));
        for (int it = 0; it < IT; ++it) for (int st = 0; st < 3; ++st) k<<<grid, 32 * wn, lds>>>(dp + st * L, o);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        std::printf("%-10s WN%d %7.1f us a launch (%5u blocks, lds %5u) = %6.1f GB/s of live bytes\n", nm, wn, ms * 1000 / (3 * IT), grid,
                    lds, 3.0 * IT * L * eb / (ms * 1e6));
        return 0;
    };
    const unsigned CLW = 10 * 32 * 16;   /* LDS a wave for CL */
    for (int rep = 0; rep < 2; ++rep) {
        run("W U2", patd<1, 2, 8>, 8, 0); run("W U9", patd<1, 9, 8>, 8, 0); run("W U18", patd<1, 18, 8>, 8, 0);
        run("C", patd<2, 1, 8>, 8, 0); run("C", patd<2, 1, 4>, 4, 0); run("C", patd<2, 1, 2>, 2, 0);
        run("CL", patd<3, 1, 8>, 8, 8 * CLW); run("CL", patd<3, 1, 4>, 4, 4 * CLW); run("CL", patd<3, 1, 2>, 2, 2 * CLW);
        run("B", patd<4, 1, 8>, 8, 0); run("B", patd<4, 1, 4>, 4, 0);
        run("W U2", patd<1, 2, 4>, 4, 0); run("W U2", patd<1, 2, 2>, 2, 0);
    }
    return 0;
}
