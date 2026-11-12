#pragma once
// r4d_ar_ll_dev.h: lever A3's device side (ds41-ll; the rule and the layout's description: r4d_ar_ll.h). The kernel
// body of the LL all-reduce in both forms and both sources (the plain all-reduce's input, moe_gather's words), shared
// by libr4d's launcher (r4d_ar_ll.hip) and tests/r4d_emu (test_arll.cpp), which runs the phases rank by rank.
//
// A unit: bits 0..47 three bf16 values (element 3j, 3j+1, 3j+2 of a 24-value group), bits 48..63 the launch's tag.
// A group: three 16-byte words, eight units. A region of W words: its G = ceil(W / 3) groups as eight planes, unit j
// of group g at [j * G + g] (a wave's stores of one plane are contiguous). A slot (parity p, source rank s) of an
// area: area + (p * 4 + s) * slot_units; the two-shot's region A at its start, region B 8 * G units on.
//
// Phases (R4dArllArgs::phases; a launch runs all of them, the emulator one at a time over the four ranks):
//   1  push: every group of the block's range to every peer (one-shot: the message; two-shot: each peer's shard into
//      its region A, this rank's own shard kept -- the gather stores its words into y)
//   2  receive and fold: each group's units from every peer, in ascending rank with this rank's own words (from the
//      input, or y for the gather), rounded once; the output stored; the two-shot also pushes the reduced group into
//      every peer's region B
//   4  two-shot only: the peers' reduced shards from region B into the output
// Every unit read is cleared (r4d_ar_ll.h: the area is zero between launches).
#include "r4d_ar_ll.h"
#include "r4d_ar_wide.h"   /* r4d_ar_to_f / r4d_ar_from_f (ar_vsum's arithmetic), R4D_AR_WIDE_MAX_BLOCKS, r4d_ar_seq.h */
#include "r4d_ar_hc.h"     /* R4dArGather, r4d_ar_gather_list / r4d_ar_gather_word_list */

/* wall_clock64 ticks (100 MHz): 2 s, as the push check's (r4d_ar_seq.h). A unit that never comes is a lockstep fault. */
#ifndef R4D_ARLL_DEADLINE
#define R4D_ARLL_DEADLINE 200000000ull
#endif

struct R4dArllArgs {
    const int4* in4;              /* plain: the input (this rank's own term); gather: unused */
    int4* out4;                   /* the output (plain: may be in4; gather: y) */
    unsigned long long* peer[3];  /* the peers' LL areas, ascending global rank without this one */
    unsigned long long* mine;     /* this rank's LL area */
    unsigned int* seq;            /* the instance's sequence counters (the parity words at [nbw, 2 nbw)) */
    long long* gate;              /* the engine's link word (r4d_ar_seq.h), or null */
    unsigned int* gate_ctr;
    long long slot_units;         /* units a (parity, source) slot */
    int n16;                      /* the message's 16-byte words */
    int rank;
    int wpr;                      /* gather: 16-byte words a row */
    unsigned test_tag;            /* 0: the launch count from the counters; else this one (tests/r4d_emu) */
    int phases;                   /* 7 in a launch */
};

/* The launch's count on the instance and its slot parity: r4d_ar_seq_take's parity half (every launch advances all
 * nbw parity words once, so a block's word is the launch count), without the handshake counter, which only the
 * flag kernels use: they take the same values on every rank whether LL launches ran in between or not. */
__device__ __forceinline__ void r4d_arll_take(unsigned int* __restrict__ ctrs, unsigned int nbw, unsigned int& s_cnt,
                                              unsigned int& s_slot) {
    const unsigned int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    unsigned int* const par = ctrs + nbw;
    if (tid == 0) {
        const unsigned int v = atomicAdd(&par[b], 1u) + 1u;
        s_cnt = v;
        s_slot = v & 1u;
    }
    for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* Element m (0 .. 23) of a group's three words, as 12 dwords. */
__device__ __forceinline__ unsigned r4d_arll_el(const unsigned* d, int m) {
    return (d[m >> 1] >> (16 * (m & 1))) & 0xffffu;
}
/* Three words -> eight units with `tag`. */
__device__ __forceinline__ void r4d_arll_pack(const int4* w, unsigned tag, unsigned long long* u) {
    unsigned d[12];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        d[4 * q] = (unsigned)w[q].x; d[4 * q + 1] = (unsigned)w[q].y;
        d[4 * q + 2] = (unsigned)w[q].z; d[4 * q + 3] = (unsigned)w[q].w;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
        u[j] = (unsigned long long)r4d_arll_el(d, 3 * j) | ((unsigned long long)r4d_arll_el(d, 3 * j + 1) << 16) |
               ((unsigned long long)r4d_arll_el(d, 3 * j + 2) << 32) | ((unsigned long long)(tag & 0xffffu) << 48);
}
/* Eight units -> three words (the tags dropped). */
__device__ __forceinline__ void r4d_arll_unpack(const unsigned long long* u, int4* w) {
    unsigned d[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        const int m0 = 2 * i, m1 = 2 * i + 1;
        const unsigned e0 = (unsigned)(u[m0 / 3] >> (16 * (m0 % 3))) & 0xffffu;
        const unsigned e1 = (unsigned)(u[m1 / 3] >> (16 * (m1 % 3))) & 0xffffu;
        d[i] = e0 | (e1 << 16);
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        w[q].x = (int)d[4 * q]; w[q].y = (int)d[4 * q + 1]; w[q].z = (int)d[4 * q + 2]; w[q].w = (int)d[4 * q + 3];
    }
}

/* One group's units into a region: eight system-scope 8-byte stores (written through, so nothing waits in this card's
 * L2 for a release). */
__device__ __forceinline__ void r4d_arll_put8(unsigned long long* reg, long long G, long long g,
                                              const unsigned long long* u) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
        __hip_atomic_store(&reg[j * G + g], u[j], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
/* One pass over a group's units still `need`ed (bit j: unit j): each one carrying `tag` is taken into u[j] and its bit
 * cleared. Returns what is still needed. */
__device__ __forceinline__ unsigned r4d_arll_try8(const unsigned long long* reg, long long G, long long g,
                                                  unsigned tag, unsigned need, unsigned long long* u) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
        if (need & (1u << j)) {
            const unsigned long long v =
                __hip_atomic_load(&reg[j * G + g], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if ((unsigned)(v >> 48) == (tag & 0xffffu)) {
                u[j] = v;
                need &= ~(1u << j);
            }
        }
    return need;
}
/* A group's eight units, waited for, then cleared to 0 (the area's state between launches). */
__device__ __forceinline__ void r4d_arll_get8(unsigned long long* reg, long long G, long long g, unsigned tag,
                                              unsigned long long* u) {
    unsigned need = r4d_arll_try8(reg, G, g, tag, 0xffu, u);
    unsigned spins = 0;
    unsigned long long t0 = 0;
    while (need) {
        __builtin_amdgcn_s_sleep(1);
        if (++spins == 64u) t0 = wall_clock64();
        else if (spins > 64u && wall_clock64() - t0 > R4D_ARLL_DEADLINE) __builtin_trap();   /* lockstep fault */
        need = r4d_arll_try8(reg, G, g, tag, need, u);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
        __hip_atomic_store(&reg[j * G + g], 0ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
/* acc[8 q + l] += element l of word q (ar_vsum's term). */
__device__ __forceinline__ void r4d_arll_add(float* acc, const int4* w) {
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        const int4 v = w[q];
        const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
        for (int l = 0; l < 8; ++l) acc[8 * q + l] += r4d_ar_to_f(pv[l]);
    }
}
/* acc -> three words, rounded once (ar_vsum's store). */
__device__ __forceinline__ void r4d_arll_round(const float* acc, int4* w) {
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        int4 o;
        __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
        for (int l = 0; l < 8; ++l) r4d_ar_from_f(acc[8 * q + l], po[l]);
        w[q] = o;
    }
}

/* The kernel body. GATHER: the words are moe_gather's (the slot list in LDS for every row, as the two-shot gather
 * builds it), y = out4; else the input's. TWO: the two-shot form. */
template <bool GATHER, bool TWO>
__device__ __forceinline__ void r4d_arll_body(const R4dArllArgs& A, const R4dArGather& G) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    __shared__ unsigned int s_cnt, s_slot;
    const long long gate_word = r4d_ar_link_word(A.gate, (unsigned int)tid);
    if (A.test_tag) {
        if (tid == 0) { s_cnt = A.test_tag; s_slot = A.test_tag & 1u; }
        __syncthreads();
    } else {
        r4d_arll_take(A.seq, R4D_AR_WIDE_MAX_BLOCKS, s_cnt, s_slot);
    }
    const unsigned tag = r4d_arll_tag16(s_cnt);
    const long long par = (long long)s_slot;
    const int rank = A.rank, n16 = A.n16;

    const int* li = nullptr;
    const float* lw = nullptr;
    const int* ln = nullptr;
    const float* lg = nullptr;
    if constexpr (GATHER) {
        __shared__ int   s_i[R4D_ARGL_TAB];
        __shared__ float s_w[R4D_ARGL_TAB];
        __shared__ int   s_n[R4D_ARGL_ROWS];
        __shared__ float s_g[R4D_ARGL_ROWS];
        r4d_ar_gather_list(G, s_i, s_w, s_n, s_g);
        li = s_i; lw = s_w; ln = s_n; lg = s_g;
    }
    /* word gw of this rank's message */
    auto word = [&](int gw) -> int4 {
        if constexpr (GATHER) {
            const int K = G.top_k, row = gw / A.wpr, i0 = (gw - row * A.wpr) * 8;
            return r4d_ar_gather_word_list(G, li + row * K, lw + row * K, ln[row], G.sh ? lg[row] : 0.0f, row, i0);
        } else {
            return A.in4[gw];
        }
    };
    int4 zero;
    zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
    unsigned long long* const mine = A.mine + par * 4 * A.slot_units;   /* + source * slot_units */
    auto peer_slot = [&](int p) { return A.peer[p] + (par * 4 + rank) * A.slot_units; };
    auto mine_slot = [&](int src) { return mine + (long long)src * A.slot_units; };

    if constexpr (!TWO) {
        const int W = n16;
        const long long Gn = r4d_arll_groups(W);
        const long long w = (Gn + nb - 1) / nb, g0 = (long long)b * w;
        const long long g1 = g0 + w < Gn ? g0 + w : Gn;
        if (A.phases & 1)
            for (long long g = g0 + tid; g < g1; g += nt) {
                int4 v[3];
                unsigned long long u[8];
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    const long long k = 3 * g + q;
                    v[q] = k < W ? word((int)k) : zero;
                    if constexpr (GATHER) if (k < W) A.out4[k] = v[q];   /* y keeps this rank's own term */
                }
                r4d_arll_pack(v, tag, u);
#pragma unroll
                for (int p = 0; p < 3; ++p) r4d_arll_put8(peer_slot(p), Gn, g, u);
            }
        if (A.phases & 2)
            for (long long g = g0 + tid; g < g1; g += nt) {
                float acc[24];
#pragma unroll
                for (int e = 0; e < 24; ++e) acc[e] = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    int4 v[3];
                    if (r == rank) {
#pragma unroll
                        for (int q = 0; q < 3; ++q) {
                            const long long k = 3 * g + q;
                            v[q] = k < W ? (GATHER ? A.out4[k] : A.in4[k]) : zero;
                        }
                    } else {
                        unsigned long long u[8];
                        r4d_arll_get8(mine_slot(r), Gn, g, tag, u);
                        r4d_arll_unpack(u, v);
                    }
                    r4d_arll_add(acc, v);
                }
                int4 o[3];
                r4d_arll_round(acc, o);
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    const long long k = 3 * g + q;
                    if (k < W) A.out4[k] = o[q];
                }
            }
    } else {
        const int sh = n16 / 4;
        const long long Gs = r4d_arll_groups(sh), offB = 8 * Gs;
        const long long w = (Gs + nb - 1) / nb, g0 = (long long)b * w;
        const long long g1 = g0 + w < Gs ? g0 + w : Gs;
        if (A.phases & 1)
            for (long long g = g0 + tid; g < g1; g += nt)
                for (int o = 0; o < 4; ++o) {
                    int4 v[3];
#pragma unroll
                    for (int q = 0; q < 3; ++q) {
                        const long long k = 3 * g + q;
                        v[q] = k < sh ? word((int)(o * sh + k)) : zero;
                        if constexpr (GATHER) if (o == rank && k < sh) A.out4[o * sh + k] = v[q];
                    }
                    if (o == rank) continue;
                    unsigned long long u[8];
                    r4d_arll_pack(v, tag, u);
                    r4d_arll_put8(peer_slot(o < rank ? o : o - 1), Gs, g, u);   /* the owner's region A */
                }
        if (A.phases & 2)
            for (long long g = g0 + tid; g < g1; g += nt) {
                float acc[24];
#pragma unroll
                for (int e = 0; e < 24; ++e) acc[e] = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    int4 v[3];
                    if (r == rank) {
#pragma unroll
                        for (int q = 0; q < 3; ++q) {
                            const long long k = 3 * g + q;
                            v[q] = k < sh ? (GATHER ? A.out4[rank * sh + k] : A.in4[rank * sh + k]) : zero;
                        }
                    } else {
                        unsigned long long u[8];
                        r4d_arll_get8(mine_slot(r), Gs, g, tag, u);
                        r4d_arll_unpack(u, v);
                    }
                    r4d_arll_add(acc, v);
                }
                int4 o[3];
                r4d_arll_round(acc, o);
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    const long long k = 3 * g + q;
                    if (k < sh) A.out4[rank * sh + k] = o[q];
                }
                unsigned long long u[8];
                r4d_arll_pack(o, tag, u);
#pragma unroll
                for (int p = 0; p < 3; ++p) r4d_arll_put8(peer_slot(p) + offB, Gs, g, u);   /* every peer's region B */
            }
        if (A.phases & 4)
            for (long long g = g0 + tid; g < g1; g += nt)
                for (int o = 0; o < 4; ++o) {
                    if (o == rank) continue;
                    unsigned long long u[8];
                    int4 v[3];
                    r4d_arll_get8(mine_slot(o) + offB, Gs, g, tag, u);
                    r4d_arll_unpack(u, v);
#pragma unroll
                    for (int q = 0; q < 3; ++q) {
                        const long long k = 3 * g + q;
                        if (k < sh) A.out4[o * sh + k] = v[q];
                    }
                }
    }
    /* Every unit this block reads has landed (and its own stores are on their way): the engine's hold on the link
     * ends with the last block, as the flag kernels end it after their last handshake. */
    if (A.gate) {
        __syncthreads();
        r4d_ar_link_landed(A.gate, gate_word, A.gate_ctr, (unsigned int)nb, (unsigned int)tid);
    }
}
