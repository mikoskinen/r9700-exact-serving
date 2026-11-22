/* pat (DRAFTER track 18.11.2026): one card, the read patterns a 16-row-tile MXFP4 expert kernel can use on the
 * drafter's gate_up experts (rows of 160 blocks x 17 B = 2720 B, 4608 rows an expert, 12 live of 32), timed as
 * GB/s of the live experts' bytes. Each block of 4 waves owns (expert, 128 rows); a wave owns 32 rows (2 tiles of 16).
 *  P1 U: lane (n, h) reads the 16-byte dword window at (row + 17 b) & ~3 + 4 h, U blocks in flight (the WMMA form today)
 *  P2 U: lane (n, h) reads 64 contiguous bytes of its row a step (the line half h of 128 B), U steps in flight
 *  P3:   lane (n, h) reads 272 contiguous bytes of its row a step (17 x 16 B; half h of 544 B = 32 blocks)
 *  P4 U: the block's 128 rows as one contiguous range (348 KB), 16 B a lane, U in flight (the coalesced bound)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); return 2; } } while (0)

constexpr int kRows = 4608, kRB = 2720, kUK = 160;
typedef __attribute__((address_space(1))) const uint32_t* gu32;
typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ u4 ld16(const uint8_t* p) {
    const gu32 g = (gu32)(const void*)p;
    u4 v; v[0] = g[0]; v[1] = g[1]; v[2] = g[2]; v[3] = g[3]; return v;
}

template <int P, int U>
__global__ __launch_bounds__(256) void pat(const uint8_t* const* ex, int nexp, uint32_t* out) {
    const int item = blockIdx.x, e = item / (kRows / 128), rg = item % (kRows / 128);
    const uint8_t* base = ex[e] + (size_t)rg * 128 * kRB;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nl = lane & 15, h = lane >> 4;
    uint32_t acc = 0;
    if constexpr (P == 1) {
        const uint8_t* r0 = base + (size_t)(wave * 32 + nl) * kRB;
        const uint8_t* r1 = r0 + 16 * kRB;
        for (int b = 0; b < kUK; b += U) {
            u4 v[U][2];
#pragma unroll
            for (int k = 0; k < U; ++k) {
                const int bb = b + k < kUK ? b + k : kUK - 1;
                v[k][0] = ld16((const uint8_t*)(((uintptr_t)(r0 + 17 * bb) & ~(uintptr_t)3) + 4 * h));
                v[k][1] = ld16((const uint8_t*)(((uintptr_t)(r1 + 17 * bb) & ~(uintptr_t)3) + 4 * h));
            }
#pragma unroll
            for (int k = 0; k < U; ++k) acc ^= v[k][0][0] ^ v[k][0][1] ^ v[k][0][2] ^ v[k][0][3] ^ v[k][1][0] ^ v[k][1][1] ^ v[k][1][2] ^ v[k][1][3];
        }
    } else if constexpr (P == 2) {
        const uint8_t* r0 = base + (size_t)(wave * 32 + nl) * kRB;
        const uint8_t* r1 = r0 + 16 * kRB;
        constexpr int NS = (kRB + 127) / 128;
        for (int s = 0; s < NS; s += U) {
            u4 v[U][2][4];
#pragma unroll
            for (int k = 0; k < U; ++k)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int off = min((s + k) * 128 + 64 * h + 16 * j, kRB - 16);
                    v[k][0][j] = ld16(r0 + off);
                    v[k][1][j] = ld16(r1 + off);
                }
#pragma unroll
            for (int k = 0; k < U; ++k)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc ^= v[k][0][j][0] ^ v[k][0][j][3] ^ v[k][1][j][1] ^ v[k][1][j][2];
        }
    } else if constexpr (P == 3) {
        const uint8_t* r0 = base + (size_t)(wave * 32 + nl) * kRB;
        const uint8_t* r1 = r0 + 16 * kRB;
        for (int s = 0; s < kRB; s += 544) {
            u4 v[2][17];
#pragma unroll
            for (int j = 0; j < 17; ++j) {
                const int off = min(s + 272 * h + 16 * j, kRB - 16);
                v[0][j] = ld16(r0 + off);
                v[1][j] = ld16(r1 + off);
            }
#pragma unroll
            for (int j = 0; j < 17; ++j) acc ^= v[0][j][0] ^ v[0][j][3] ^ v[1][j][1] ^ v[1][j][2];
        }
    } else if constexpr (P == 5 || P == 6) {
        /* P5 U: lane (n, h) reads the 136 contiguous bytes of its half of a 16-block super-step (34 dwords: 8 x 16 B + 8 B),
         * U super-steps in flight; P6: the same with one row tile a wave (waves own 16 rows; 8 waves a block) */
        constexpr int WR = P == 5 ? 2 : 1;
        const uint8_t* r[WR];
#pragma unroll
        for (int t = 0; t < WR; ++t) r[t] = base + (size_t)((P == 5 ? wave * 32 : wave * 16) + nl + 16 * t) * kRB;
        for (int s = 0; s < kRB; s += 272 * U) {
            u4 v[U][WR][8]; uint2 q[U][WR];
#pragma unroll
            for (int k = 0; k < U; ++k)
#pragma unroll
                for (int t = 0; t < WR; ++t) {
                    const int o = min(s + 272 * k, kRB - 272) + 136 * h;
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[k][t][j] = ld16(r[t] + o + 16 * j);
                    const gu32 g = (gu32)(const void*)(r[t] + o + 128);
                    q[k][t].x = g[0]; q[k][t].y = g[1];
                }
#pragma unroll
            for (int k = 0; k < U; ++k)
#pragma unroll
                for (int t = 0; t < WR; ++t) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc ^= v[k][t][j][0] ^ v[k][t][j][3];
                    acc ^= q[k][t].x ^ q[k][t].y;
                }
        }
    } else {
        const size_t n = (size_t)128 * kRB / 16;
        for (size_t i = threadIdx.x; i < n; i += 128 * U) {
            u4 v[U];
#pragma unroll
            for (int k = 0; k < U; ++k) { const size_t j = i + 128 * k < n ? i + 128 * k : n - 1; v[k] = ld16(base + 16 * j); }
#pragma unroll
            for (int k = 0; k < U; ++k) acc ^= v[k][0] ^ v[k][1] ^ v[k][2] ^ v[k][3];
        }
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

int main() {
    const int E = 32, L = 12;
    const size_t eb = (size_t)kRows * kRB;
    uint8_t* w; CK(hipMalloc((void**)&w, eb * E * 3));
    CK(hipMemset(w, 3, eb * E * 3));
    uint32_t* o; CK(hipMalloc((void**)&o, 64));
    /* 3 stages' live experts (12 of 32, scattered), a pass = 3 launches */
    std::vector<const uint8_t*> hp;
    for (int st = 0; st < 3; ++st)
        for (int i = 0; i < L; ++i) hp.push_back(w + eb * (size_t)(st * E + (i * 7 + st * 3) % E));
    const uint8_t** dp; CK(hipMalloc((void**)&dp, hp.size() * sizeof(void*)));
    CK(hipMemcpy(dp, hp.data(), hp.size() * sizeof(void*), hipMemcpyHostToDevice));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const unsigned grid = L * (kRows / 128);
    auto run = [&](const char* nm, void (*k)(const uint8_t* const*, int, uint32_t*), unsigned thr = 128) {
        for (int w0 = 0; w0 < 3; ++w0) for (int st = 0; st < 3; ++st) k<<<grid, thr>>>(dp + st * L, L, o);
        CK(hipEventRecord(a));
        for (int it = 0; it < 10; ++it) for (int st = 0; st < 3; ++st) k<<<grid, thr>>>(dp + st * L, L, o);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b));
        std::printf("%-8s %7.1f us a launch (%u blocks) = %6.1f GB/s\n", nm, ms * 1000 / 30, grid, 30.0 * L * eb / (ms * 1e6));
        return 0;
    };
    run("P1 U2", pat<1, 2>); run("P1 U4", pat<1, 4>); run("P1 U8", pat<1, 8>);
    run("P2 U1", pat<2, 1>); run("P2 U2", pat<2, 2>);
    run("P3", pat<3, 1>);
    run("P4 U2", pat<4, 2>); run("P4 U4", pat<4, 4>); run("P4 U8", pat<4, 8>);
    run("P1 U4", pat<1, 4>); run("P4 U4", pat<4, 4>);
    run("P5 U1", pat<5, 1>); run("P5 U2", pat<5, 2>); run("P6 U1", pat<6, 1>, 256); run("P6 U2", pat<6, 2>, 256);
    return 0;
}
