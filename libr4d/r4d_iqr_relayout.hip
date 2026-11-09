/* r4d_iqr_relayout.hip -- trellis stored form written on the card from the container's planes.
 *
 * The bit order is r4d_iqr_layout.cpp's walk_tile, not a second layout. A unit is unit_bytes()
 * bytes (4 * lane_bits bits). Lane l of unit (tile, K-block) starts at that bit offset plus
 * l * lane_bits. Within the lane, column e's 16 codes sit at codes 16e..16e+15, bottom-up
 * (row 15 of the tile first). The table is copied at aux_off(), and a mad form's A and B
 * follow it. A mismatch here is a wrong answer, so the host relayout remains the fallback
 * whenever this returns anything but RAD_OK.
 */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_iqr.h"

#include <cstdio>
#include <cstring>

namespace {

using namespace r4d_iqr;

__device__ uint32_t load_bits(const uint8_t* row, uint32_t o, uint32_t n) {
    const uint8_t* b = row + (o >> 3);
    const uint32_t sh = o & 7u;
    const uint32_t bytes = (sh + n + 7u) >> 3;
    uint32_t w = 0;
    for (uint32_t i = 0; i < bytes; ++i) w |= (uint32_t)b[i] << (8u * i);
    return (w >> sh) & ((1u << n) - 1u);
}

__device__ void store_bits(uint8_t* dst, uint64_t o, uint32_t v, uint32_t n) {
    uint8_t* b = dst + (o >> 3);
    const uint32_t sh = (uint32_t)(o & 7ull);
    const uint32_t bytes = (sh + n + 7u) >> 3;
    v <<= sh;
    for (uint32_t i = 0; i < bytes; ++i) b[i] |= (uint8_t)(v >> (8u * i));
}

__global__ void iqr_relayout_trellis_kernel(const uint8_t* __restrict__ codes, uint8_t* __restrict__ dst,
                                            uint32_t N, uint32_t K, uint32_t bits, uint32_t nkb,
                                            uint32_t lbits, uint64_t unit_bits, uint64_t aux,
                                            uint32_t entries, const uint8_t* __restrict__ table,
                                            const uint32_t* __restrict__ mad) {
    const uint32_t rt = blockIdx.x;
    if (rt * 16u >= N) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t row_bytes = K * bits / 8u;
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const uint64_t l0 = ((uint64_t)rt * nkb + kb) * unit_bits + (uint64_t)lane * lbits;
        const uint32_t k0 = kb * 128u + 4u * lane;
        for (uint32_t e = 0; e < 4u; ++e) {
            for (uint32_t q = 0; q < 16u; ++q) {
                const uint32_t r = rt * 16u + 15u - q;
                const uint32_t c = k0 + e;
                const uint32_t code = load_bits(codes + (size_t)r * row_bytes, c * bits, bits);
                store_bits(dst, l0 + (uint64_t)(e * 16u + q) * bits, code, bits);
            }
        }
    }
    if (rt == 0 && lane == 0) {
        for (uint32_t i = 0; i < entries; ++i) dst[aux + i] = table[i];
        if (mad) {
            uint32_t* m = (uint32_t*)(dst + aux + entries);
            m[0] = mad[0];
            m[1] = mad[1];
        }
    }
}

}  /* namespace */

extern "C" int r4d_iqr_relayout_gpu(const void* codes, void* dst, int64_t N, int64_t K, uint32_t bits,
                                   uint32_t L, uint32_t entries, const void* table, const void* mad,
                                   int64_t dst_bytes, void* stream) {
    using namespace r4d_iqr;
    if (!codes || !dst || !table || N <= 0 || K <= 0 || (N % 16) || (K % 128) || bits == 0 || bits > 8)
        return RAD_E_SHAPE;
    if (L == 0 || L > 24 || entries != (1u << L)) return RAD_E_SHAPE;
    Form f;
    f.kind = mad ? kTrellisMad : kTrellis;
    f.bits = bits;
    f.L = L;
    const int64_t wbytes = weight_bytes(f, N, K);
    const int64_t packed = N * (K * (int64_t)bits / 8);
    if (dst_bytes < wbytes || packed != codes_bytes(f, N, K)) return RAD_E_SHAPE;
    const int64_t tab_bytes = entries;
    const int64_t need = packed + tab_bytes + (mad ? 8 : 0);
    /* File reads run on many threads at once (mover.cpp). A shared scratch would mix two experts
     * and the next token would not be the one the CPU unpack would have produced. */
    struct Buf { uint8_t* p = nullptr; int64_t cap = 0; };
    static thread_local Buf scratch, outb;
    static thread_local hipStream_t mine = nullptr;
    if (!mine && hipStreamCreate(&mine) != hipSuccess) return RAD_E_STATE;
    if (scratch.cap < need) {
        if (scratch.p) hipFree(scratch.p);
        if (hipMalloc(&scratch.p, (size_t)need) != hipSuccess) { scratch.p = nullptr; scratch.cap = 0; return RAD_E_NOMEM; }
        scratch.cap = need;
    }
    if (outb.cap < wbytes) {
        if (outb.p) hipFree(outb.p);
        if (hipMalloc(&outb.p, (size_t)wbytes) != hipSuccess) { outb.p = nullptr; outb.cap = 0; return RAD_E_NOMEM; }
        outb.cap = wbytes;
    }
    auto st = mine;
    if (hipMemcpyAsync(scratch.p, codes, (size_t)packed, hipMemcpyHostToDevice, st) != hipSuccess)
        return RAD_E_STATE;
    if (hipMemcpyAsync(scratch.p + packed, table, (size_t)tab_bytes, hipMemcpyHostToDevice, st) != hipSuccess)
        return RAD_E_STATE;
    if (mad && hipMemcpyAsync(scratch.p + packed + tab_bytes, mad, 8, hipMemcpyHostToDevice, st) != hipSuccess)
        return RAD_E_STATE;
    if (hipMemsetAsync(outb.p, 0, (size_t)wbytes, st) != hipSuccess) return RAD_E_STATE;
    iqr_relayout_trellis_kernel<<<(uint32_t)(N / 16), 32, 0, st>>>(
        scratch.p, outb.p, (uint32_t)N, (uint32_t)K, bits, (uint32_t)(K / 128), lane_bits(f),
        (uint64_t)unit_bytes(f) * 8ull, (uint64_t)aux_off(f, N, K), entries, scratch.p + packed,
        mad ? (const uint32_t*)(scratch.p + packed + tab_bytes) : nullptr);
    if (hipMemcpyAsync(dst, outb.p, (size_t)wbytes, hipMemcpyDeviceToHost, st) != hipSuccess)
        return RAD_E_STATE;
    if (hipGetLastError() != hipSuccess) return RAD_E_STATE;
    if (hipStreamSynchronize(st) != hipSuccess) return RAD_E_STATE;
    if (dst_bytes > wbytes) std::memset((uint8_t*)dst + wbytes, 0, (size_t)(dst_bytes - wbytes));
    static int once = 0;
    if (!once) { once = 1; fprintf(stderr, "I iqr file relayout: gpu wrote the window\n"); }
    return RAD_OK;
}
