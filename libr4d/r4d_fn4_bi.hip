/* r4d_fn4_bi.hip -- bi_rowhash: a 64-bit fingerprint of each row's bytes (fn4 BI's engine tracer, 11.11.2026; ported from
 * the ds41-bi line's tracer, radiance-bi 060dff0).
 *
 * Lane t folds its 16-bit units u (at index i) as the XOR of a mix of (i, u); the workgroup XORs the lanes' folds -- an
 * order-free reduction, so the value is a function of the row's bytes alone, whatever the launch's grid. A debug op: the
 * qwen4exp architecture declares it only under RADIANCE_FN4_BITRACE, never in a served pass otherwise.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_fn4bi {

struct BiArgs {
    const uint8_t* x;  long long xs;   /* rows' bytes, row pitch in bytes */
    long long      nu;                 /* 16-bit units a row */
    int            odd;                /* a trailing odd byte */
    uint32_t*      h;  long long hs;   /* [rows][2] i32, row pitch in elements */
    long long      rows;
};

__device__ __forceinline__ unsigned long long bi_mix(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

__global__ __launch_bounds__(256) void bi_rowhash_kernel(BiArgs A) {
    __shared__ unsigned long long sh[256];
    for (long long r = blockIdx.x; r < A.rows; r += gridDim.x) {
        const uint16_t* p = (const uint16_t*)(A.x + r * A.xs);
        unsigned long long acc = 0;
        for (long long i = threadIdx.x; i < A.nu; i += 256)
            acc ^= bi_mix(((unsigned long long)i << 16 | p[i]) ^ 0x9E3779B97F4A7C15ull);
        if (A.odd && threadIdx.x == 0) acc ^= bi_mix(((unsigned long long)A.nu << 16 | A.x[r * A.xs + 2 * A.nu]) ^ 0x51ED27ull);
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (int w = 128; w > 0; w >>= 1) {
            if ((int)threadIdx.x < w) sh[threadIdx.x] ^= sh[threadIdx.x + w];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const unsigned long long v = bi_mix(sh[0] ^ (unsigned long long)(2 * A.nu + A.odd));
            A.h[r * A.hs] = (uint32_t)v;
            A.h[r * A.hs + 1] = (uint32_t)(v >> 32);
        }
        __syncthreads();
    }
}

}  /* namespace r4d_fn4bi */

extern "C" {

/* operands: x (any dtype, rows of contiguous trailing extents), h (i32 [rows][2], out) */
int r4d_rad_bi_rowhash(const RadArgs* a, RadStream stream) {
    using namespace r4d_fn4bi;
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &h = a->t[1];
    if (x.rank < 1 || h.rank != 2 || h.dtype != RAD_I32 || h.shape[1] != 2 || h.stride[1] != 1) return RAD_E_SHAPE;
    const long long rows = (long long)x.shape[0];
    if (rows < 1 || (long long)h.shape[0] != rows) return RAD_E_SHAPE;
    long long inner = 1;
    for (uint32_t d = x.rank; d-- > 1;) {
        if ((long long)x.stride[d] != inner) return RAD_E_SHAPE;
        inner *= (long long)x.shape[d];
    }
    const long long eb = (long long)rad_dtype_bytes(x.dtype, 1);
    if (eb < 1) return RAD_E_DTYPE;
    const long long bytes = inner * eb, xs = (x.rank > 1 ? (long long)x.stride[0] : 1) * eb;
    if (((uintptr_t)x.data & 1u) || (xs & 1)) return RAD_E_ALIGN;
    BiArgs P{};
    P.x = (const uint8_t*)x.data; P.xs = xs; P.nu = bytes / 2; P.odd = (int)(bytes & 1);
    P.h = (uint32_t*)h.data; P.hs = (long long)h.stride[0]; P.rows = rows;
    const unsigned blocks = (unsigned)(rows < 4096 ? rows : 4096);
    R4D_KLAUNCH(R4D_KNAME(bi_rowhash_kernel), dim3(blocks), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
