// r4d_prefetch.hip - r4d_rad_prefetch: read a weight's leading bytes into the on-die cache.
//
// A decode step at a few rows streams each weight once from memory, and the all-reduce between two
// GEMMs leaves memory idle for its whole round trip: ~20 us at a DFlash2 verify's 8 rows on 4x
// R9700. A read issued on the second lane in that window puts the next GEMM's first bytes in the
// 64 MB Infinity Cache, and the GEMM then streams them from there. On four cards (a microbenchmark
// of the 8-row two-shot and a GEMM-like stream of a cold 29 MB weight) 16 MB read beside the
// all-reduce took a layer from 84.0 us to 70.1; the same read on the same stream gained nothing.
//
// It writes nothing at all. Each lane folds its loads into one word and hands it to an empty
// `asm volatile`, which the compiler can neither drop nor see into, so the loads stay.
//
// Nothing here reaches HIP's runtime: no allocation at launch and no `__device__` word, which HIP's
// runtime registers and places itself (the engine records only fat binaries and functions). Such a
// word cost ~8% of a DFlash2 step on 4x R9700, for a reason not pinned down.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include <cstdint>

namespace {

/* The reads of one lane: eight 16-byte loads in flight (issued together, then folded), the grid's stride apart, then the
 * tail one at a time. DRAFTER track 22.11.2026 round 85 (lever PFX, drafter/kit/mes/pfx.cpp on one R9700): one load in
 * flight a lane read 29.8 MB in 65.9 us over five 64-block launches (452 GB/s), eight in flight in 48.4 us over one launch
 * (616 GB/s); a stride of whole 16-byte words a lane either way, so every byte read is the same set. */
__device__ __forceinline__ unsigned pf_lane(const int4* __restrict__ p, size_t n16) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nt  = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    size_t i = tid;
    for (; i + 7 * nt < n16; i += 8 * nt) {
        int4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = p[i + (size_t)u * nt];
#pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= (unsigned)(v[u].x ^ v[u].y ^ v[u].z ^ v[u].w);
    }
    for (; i < n16; i += nt) {
        const int4 v = p[i];
        acc ^= (unsigned)(v.x ^ v.y ^ v.z ^ v.w);
    }
    return acc;
}

/* Up to eight weights a launch (lever PFX: one dispatch beside a collective instead of one a weight -- on one R9700 five 64-block
 * launches of the same 29.8 MB ended 14.1 us past a 53-us window where one launch ended 4.4 us past it: each further unordered
 * dispatch's completion is retired in order, ~2.5 us apart). The weights one after another, every lane through each. */
struct PfSet {
    const int4* p[8];
    size_t n16[8];
    int n;
};

__global__ __launch_bounds__(256) void prefetch_kernel(PfSet s) {
    unsigned acc = 0;
    for (int j = 0; j < s.n; ++j) acc ^= pf_lane(s.p[j], s.n16[j]);
    asm volatile("" ::"v"(acc));
}

/* The chained form (RadKernelChainFn): it waits for the kernels before it on its queue, as every
 * chained form does, but adds its blocks to the counter at once, before reading: nothing after it
 * reads anything it writes, so the next kernel need not wait for its reads. Issued on lane 0
 * between a GEMM and the collective after it (where launches chain), it starts reading as the
 * GEMM's last block ends, with no event between the queues and no barrier bit on the GEMM. */
__global__ __launch_bounds__(256) void r4dchain_prefetch(PfSet s) {
    r4d_chain_wait();
    r4d_chain_done();
    unsigned acc = 0;
    for (int j = 0; j < s.n; ++j) acc ^= pf_lane(s.p[j], s.n16[j]);
    asm volatile("" ::"v"(acc));
}

}  // namespace

extern "C" int r4d_rad_layout_any(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*) {
    return RAD_LAYOUT_ANY;
}

extern "C" int r4d_rad_prefetch(const RadArgs* a, RadStream stream) {
    const RadTensor* w0 = r4d_opt(a, 0);
    if (!w0 || !w0->data) return R4D_NO(RAD_E_INVAL);
    static const char* const kBytes[8] = { "bytes", "bytes1", "bytes2", "bytes3", "bytes4", "bytes5", "bytes6", "bytes7" };
    PfSet s{};
    for (int j = 0; j < 8 && j < a->n_t; ++j) {
        const RadTensor* w = r4d_opt(a, j);
        if (!w || !w->data) continue;
        long long bytes = r4d_geti_or(a, kBytes[j], 0);
        if (bytes <= 0) continue;
        // Never past what the tensor says it holds, where its dtype says how wide an element is.
        const int bits = rad_dtype_bits(w->dtype);
        if (bits > 0) {
            long long n = 1;
            for (uint32_t i = 0; i < w->rank; ++i) n *= w->shape[i];
            const long long have = n * bits / 8;
            if (have > 0 && bytes > have) bytes = have;
        }
        if ((uintptr_t)w->data % 16) return R4D_NO(RAD_E_ALIGN);
        if (bytes / 16 <= 0) continue;
        s.p[s.n] = (const int4*)w->data;
        s.n16[s.n] = (size_t)(bytes / 16);
        ++s.n;
    }
    if (s.n == 0) return RAD_OK;
    int nb = r4d_cfg_prefetch_nb(a);
    if (nb < 1) nb = 64;
    const bool ch = r4d_geti_or(a, "chain", 0) != 0;
    return R4D_EMIT(stream, (ch ? r4dchain_prefetch : prefetch_kernel), dim3((unsigned)nb), dim3(256), 0, s);
}
