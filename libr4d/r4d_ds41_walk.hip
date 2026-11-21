// r4d_ds41_walk.hip -- lever W1: DeepSeek-V4.1's DSpark Markov walk over the whole vocabulary, one walk step on
// one rank's vocabulary shard (op dspark_walk_step; specs and row r4d_ds41_walk.inc, operands
// r4d_ds41_walk_shapes.inc).
//
// The DSpark head proposes a block of `steps` tokens; position l's token is
//
//   tok[l] = argmax_v  logits[l][v] + sum_r pred[prev][r] * succ[v][r]       prev = anchor (l = 0), tok[l-1]
//
// over the WHOLE vocabulary (the reference, exllamav3 and vLLM), ties to the lower id. Today's walk
// (dflash_select over row_topk's candidates) re-ranks only the head's top 32. Under tensor parallelism the
// head's logits are vocabulary-sharded, so each step is a per-rank partial argmax and a cross-rank max:
//
//   dspark_walk_step(step l):  prev = the max of step l-1's gathered keys (the anchor at l = 0); the rank's
//                              block 0 writes it to tokens[s][l-1]; every column j of this rank's shard is
//                              scored and the shard's best key max-ed into keys[s] (pre-zeroed);
//   all_gather(keys)        :  the plugin issues it between steps (world > 1), so step l+1 reads every rank's;
//   dspark_walk_step(steps) :  only the last token's merge (tokens[s][steps-1]).
//
// THE ARITHMETIC, fixed so a host reference reproduces it bit for bit (tests/r4d_emu/test_ds41walk.cpp):
//   acc = 0; for r = 0 .. R-1: acc = fma(pred[prev][r], succ[v][r], acc)   (one thread a token, r ascending)
//   score = logits[l][v] + acc                                            (f32, round to nearest)
// pred / succ are bf16, widened exactly; an id outside [0, n_vocab) as predecessor contributes zeros. A NaN
// score is never chosen; -0 counts as +0; no valid score at all gives token 0.
//
// THE KEY: (ordered bits of the score) << 32 | (~id), an unsigned 64-bit integer whose order is the score's
// with ties to the lower id, so the rank-local reduction is one atomicMax per (block, sequence) and the
// cross-rank one a max over the gathered keys: both are order-independent, so every rank derives the same
// token from the same gathered bytes. Key 0 is "none" (a valid key's high word is at least 0x007fffff).
// A key travels in a 16-byte SLOT (R4D_DS41_WALK_SLOT f32: the key, then 8 bytes nobody reads), because the
// card's all-gather moves whole 16-byte words a row (r4d_ar_gather_admit.h); keys is [M, 4] f32 and the
// gathered prev [M, world * 4] f32, rank w's key of sequence s at prev's u64 [s][2 w].
//
// One thread a vocabulary column, 128 a block, every sequence of the call in passes of eight (their
// predecessor rows staged in LDS as f32); form 0: a thread reads its succ row once a pass in 16-byte pieces; form 1
// (`form` 1, lever W1b): the block's rows come through LDS as whole 128-byte lines, the same chain a thread.
#include <hip/hip_runtime.h>
#include "r4d_ika.h"   // lever IKA (DRAFTER track round 50): R4D_IKA_ENTRY()
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"   // r4d_bf2f, r4d_u16, r4d_u32x4, R4dV8
#include "r4d_plugin.h"

namespace {

constexpr int kWalkThreads = 128;
constexpr int kWalkWaves = kWalkThreads / 32;
constexpr int kWalkSeqs = 8;          // sequences a pass (SEQS 16 only as a bench pin)
constexpr int kWalkMaxRank = 512;

__device__ __forceinline__ unsigned walk_ord(unsigned b) {
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
/* The key of `score` at global id `gid`; 0 for a NaN score. */
__device__ __forceinline__ unsigned long long walk_key(float score, int gid) {
    unsigned b = __float_as_uint(score);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0ull;   // NaN: never chosen
    if ((b << 1) == 0u) b = 0u;                          // -0 -> +0
    return ((unsigned long long)walk_ord(b) << 32) | (unsigned long long)(~(unsigned)gid);
}
__device__ __forceinline__ int walk_token(unsigned long long k) {
    return k ? (int)(~(unsigned)(k & 0xffffffffull)) : 0;
}
__device__ __forceinline__ unsigned long long walk_umax(unsigned long long a, unsigned long long b) {
    return a > b ? a : b;
}
/* The wave's largest key (wave32), every lane. */
__device__ __forceinline__ unsigned long long walk_wave_max(unsigned long long k) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned hi = __shfl_xor((unsigned)(k >> 32), o, 32);
        const unsigned lo = __shfl_xor((unsigned)(k & 0xffffffffull), o, 32);
        k = walk_umax(k, ((unsigned long long)hi << 32) | lo);
    }
    return k;
}

/* STAGED (form 1, lever W1b): the block's 128 succ rows go through LDS in 64-element chunks, each chunk read from
 * global memory as whole 128-byte lines (16 bytes a lane, eight lanes a row) one chunk ahead in registers, then
 * every thread runs ITS id's fma chain over the chunk from LDS: the same chain, r ascending, as form 0 (which reads
 * its own row in 16-byte pieces, 512 bytes apart across a wave's lanes), so the same bits. The LDS rows are 33 dwords
 * apart, so a wave's 32 rows sit in 32 different banks on the reads and a piece's four dwords on the writes. */
constexpr int kStageElems = 64;                       // bf16 a row a chunk
constexpr int kStagePitch = kStageElems / 2 + 1;      // dwords a row in LDS (33: bank-free)
constexpr int kStagePieces = kWalkThreads * kStageElems / 8 / kWalkThreads;   // 16-byte pieces a thread a chunk (8)
constexpr int kStageMaxRank = 256;                    // form 1's predecessor rows in LDS (8 KB): more LDS, fewer waves

template <bool STAGED, int SEQS = kWalkSeqs>
__global__ __launch_bounds__(kWalkThreads) void dspark_walk_step_kernel(
    const float* __restrict__ logits, long lpitch, const int* __restrict__ anchor, long astride,
    const unsigned long long* __restrict__ prev, long ppitch, int world,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    unsigned long long* __restrict__ keys, long kpitch, int* __restrict__ tokens, long tpitch,
    int S, int step, int steps, int R, int V, int N, int voff) { R4D_IKA_ENTRY();
    __shared__ float su[SEQS * (STAGED ? kStageMaxRank : kWalkMaxRank)];
    __shared__ int sprev[SEQS];
    __shared__ unsigned long long sbest[SEQS * kWalkWaves];
    __shared__ unsigned sw[STAGED ? kWalkThreads * kStagePitch : 1];
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int j = blockIdx.x * kWalkThreads + tid;
    const bool live = step < steps && j < N;

    for (int g0 = 0; g0 < S; g0 += SEQS) {
        const int ng = S - g0 < SEQS ? S - g0 : SEQS;
        if (tid < ng) {
            const int s = g0 + tid;
            int p;
            if (step == 0) {
                p = anchor[(long)s * astride];
            } else {
                unsigned long long k = 0ull;
                for (int w = 0; w < world; ++w) k = walk_umax(k, prev[(long)s * ppitch + 2 * w]);
                p = walk_token(k);
                if (blockIdx.x == 0) tokens[(long)s * tpitch + step - 1] = p;
            }
            sprev[tid] = p;
        }
        if (step == steps) continue;   // the last merge only (one block): nothing to score
        __syncthreads();
        for (int i = tid; i < ng * R; i += kWalkThreads) {
            const int sl = i / R, r = i - sl * R;
            const int p = sprev[sl];
            su[sl * R + r] = (p >= 0 && p < V) ? r4d_bf2f(pred[(long)p * R + r]) : 0.0f;
        }
        __syncthreads();

        float acc[SEQS];
#pragma unroll
        for (int sl = 0; sl < SEQS; ++sl) acc[sl] = 0.0f;
        if constexpr (STAGED) {
            const int base = blockIdx.x * kWalkThreads;
            const int nrows = N - base < kWalkThreads ? N - base : kWalkThreads;
            const int nch = (R + kStageElems - 1) / kStageElems;
            r4d_u32x4 pre[kStagePieces];
            /* chunk c's pieces of this thread: piece q = p * 128 + tid -> row q / 8, 16-byte column q % 8, so a wave
             * reads four rows' 128-byte chunk lines whole; R a multiple of 32 makes a chunk 4 or 8 pieces a row. */
            auto load = [&](int c) {
                const int n4 = (R - c * kStageElems) / 8 < 8 ? (R - c * kStageElems) / 8 : 8;
#pragma unroll
                for (int p = 0; p < kStagePieces; ++p) {
                    const int q = p * kWalkThreads + tid, row = q >> 3, c4 = q & 7;
                    r4d_u32x4 z = { 0u, 0u, 0u, 0u };
                    if (row < nrows && c4 < n4)
                        z = *(const r4d_u32x4*)(succ + (long)(voff + base + row) * R + c * kStageElems + c4 * 8);
                    pre[p] = z;
                }
            };
            load(0);
            for (int c = 0; c < nch; ++c) {
                __syncthreads();   // the previous chunk's reads are done
#pragma unroll
                for (int p = 0; p < kStagePieces; ++p) {
                    const int q = p * kWalkThreads + tid, row = q >> 3, c4 = q & 7;
                    unsigned* d = sw + row * kStagePitch + c4 * 4;
                    d[0] = pre[p].x; d[1] = pre[p].y; d[2] = pre[p].z; d[3] = pre[p].w;
                }
                if (c + 1 < nch) load(c + 1);
                __syncthreads();
                if (live) {
                    const int nd = (R - c * kStageElems) / 2 < kStageElems / 2 ? (R - c * kStageElems) / 2 : kStageElems / 2;
                    const unsigned* wr = sw + tid * kStagePitch;
                    for (int dd = 0; dd < nd; ++dd) {
                        const unsigned w2 = wr[dd];
                        const float w0 = r4d_bf2f((r4d_u16)(w2 & 0xffffu)), w1 = r4d_bf2f((r4d_u16)(w2 >> 16));
                        const int r = c * kStageElems + 2 * dd;
#pragma unroll
                        for (int sl = 0; sl < SEQS; ++sl) {
                            if (sl < ng) {
                                acc[sl] = __fmaf_rn(su[sl * R + r], w0, acc[sl]);
                                acc[sl] = __fmaf_rn(su[sl * R + r + 1], w1, acc[sl]);
                            }
                        }
                    }
                }
            }
        } else if (live) {
            const r4d_u16* __restrict__ w = succ + (long)(voff + j) * R;
            /* R is a multiple of 32: four 16-byte pieces of the row in flight. */
#pragma unroll 4
            for (int r0 = 0; r0 < R; r0 += 8) {
                R4dV8 v;
                v.v = *(const r4d_u32x4*)(w + r0);
                float wf[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) wf[q] = r4d_bf2f(v.h[q]);
#pragma unroll
                for (int sl = 0; sl < SEQS; ++sl) {
                    if (sl < ng) {
                        const float* u = su + sl * R + r0;
#pragma unroll
                        for (int q = 0; q < 8; ++q) acc[sl] = __fmaf_rn(u[q], wf[q], acc[sl]);
                    }
                }
            }
        }
#pragma unroll
        for (int sl = 0; sl < SEQS; ++sl) {
            if (sl < ng) {
                unsigned long long k = 0ull;
                if (live) k = walk_key(logits[(long)((g0 + sl) * steps + step) * lpitch + j] + acc[sl], voff + j);
                k = walk_wave_max(k);
                if (lane == 0) sbest[sl * kWalkWaves + wave] = k;
            }
        }
        __syncthreads();
        if (tid < ng) {
            unsigned long long k = 0ull;
            for (int w = 0; w < kWalkWaves; ++w) k = walk_umax(k, sbest[tid * kWalkWaves + w]);
            if (k) atomicMax(&keys[(long)(g0 + tid) * kpitch], k);
        }
        __syncthreads();   // sprev / su / sbest are the next pass's
    }
}

/* DRAFTER track 18.11.2026 (RADIANCE_DS41_WALK_WM=1 at load, rule B): the walk's scores as a WMMA product. The fma chains
 * (S x N x R per step, each with an LDS read of the predecessor value) bound form 0 at S 16 (~135-150 us a step, the shard's
 * succ rows are 16.5 MB = ~27 us). Here a wave takes 16 vocabulary columns at a time: A = the sequences' predecessor rows
 * (16 rows, bf16 as stored), B = the columns' succ rows, C = 16 x 16 dot products by v_wmma_f32_16x16x16_bf16 over R = 256
 * in 16 K-steps; lane half h holds r = 128 h + 8 j .. + 7 of K-step j in both operands (any order both agree on), so a lane
 * reads its succ row half as 256 contiguous bytes. score = logits + C (one f32 add), the key as walk_key. Not form 0's chain
 * order: rule B (deterministic; a sequence's scores depend only on its own predecessor row, so batch-invariant). */
typedef __bf16 walk_v8bf __attribute__((ext_vector_type(8)));
typedef float walk_v8f __attribute__((ext_vector_type(8)));

constexpr int kWmR = 256;
__global__ __launch_bounds__(kWalkThreads) void dspark_walk_wm_kernel(
    const float* __restrict__ logits, long lpitch, const int* __restrict__ anchor, long astride,
    const unsigned long long* __restrict__ prev, long ppitch, int world,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    unsigned long long* __restrict__ keys, long kpitch, int* __restrict__ tokens, long tpitch,
    int S, int step, int steps, int V, int N, int voff) { R4D_IKA_ENTRY();
    __shared__ int sprev[16];
    __shared__ unsigned long long sbest[16 * kWalkWaves];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, nl = lane & 15, h = lane >> 4;
    const int ntile = (N + 15) >> 4;
    for (int g0 = 0; g0 < S; g0 += 16) {
        const int ng = S - g0 < 16 ? S - g0 : 16;
        if (tid < ng) {
            const int s = g0 + tid;
            int p;
            if (step == 0) {
                p = anchor[(long)s * astride];
            } else {
                unsigned long long k = 0ull;
                for (int w = 0; w < world; ++w) k = walk_umax(k, prev[(long)s * ppitch + 2 * w]);
                p = walk_token(k);
                if (blockIdx.x == 0) tokens[(long)s * tpitch + step - 1] = p;
            }
            sprev[tid] = p;
        }
        if (step == steps) continue;
        __syncthreads();
        /* A: this lane's sequence row (nl), its half h, all 16 K-steps */
        r4d_u32x4 af[16];
        {
            const int p = nl < ng ? sprev[nl] : -1;
            const bool ok = p >= 0 && p < V;
            const r4d_u16* pr = pred + (long)(ok ? p : 0) * kWmR + 128 * h;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                r4d_u32x4 z = { 0u, 0u, 0u, 0u };
                if (ok) z = *(const r4d_u32x4*)(pr + 8 * j);
                af[j] = z;
            }
        }
        unsigned long long best[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) best[i] = 0ull;
        for (int t = blockIdx.x * kWalkWaves + wave; t < ntile; t += gridDim.x * kWalkWaves) {
            const int v = t * 16 + nl;
            const bool vl = v < N;
            const r4d_u16* sr = succ + (long)(voff + (vl ? v : 0)) * kWmR + 128 * h;
            r4d_u32x4 bf[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) bf[j] = *(const r4d_u32x4*)(sr + 8 * j);
            walk_v8f c = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int j = 0; j < 16; ++j)
                c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(walk_v8bf, af[j]), __builtin_bit_cast(walk_v8bf, bf[j]), c);
            /* D lane L: column (vocabulary) L % 16, rows (sequences) 8 (L / 16) + i */
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int sl = 8 * h + i;
                if (vl && sl < ng) {
                    const float sc = logits[(long)((g0 + sl) * steps + step) * lpitch + v] + c[i];
                    best[i] = walk_umax(best[i], walk_key(sc, voff + v));
                }
            }
        }
        /* each sequence's max over the half-wave's 16 columns, then the waves, then the block's atomicMax */
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned long long k = best[i];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const unsigned hi = __shfl_xor((unsigned)(k >> 32), o, 32);
                const unsigned lo = __shfl_xor((unsigned)(k & 0xffffffffull), o, 32);
                k = walk_umax(k, ((unsigned long long)hi << 32) | lo);
            }
            if (nl == 0) sbest[(8 * h + i) * kWalkWaves + wave] = k;
        }
        __syncthreads();
        if (tid < ng) {
            unsigned long long k = 0ull;
            for (int w = 0; w < kWalkWaves; ++w) k = walk_umax(k, sbest[tid * kWalkWaves + w]);
            if (k) atomicMax(&keys[(long)(g0 + tid) * kpitch], k);
        }
        __syncthreads();
    }
}

}  // namespace

/* ------------------------------------------------------------------ plugin entry */
static int g_walk_seqs = 0;   /* bench hook (DRAFTER track): 0 the rule, 8 / 16 pinned */
extern "C" void r4d_walk_seqs(int n) { g_walk_seqs = n == 8 || n == 16 ? n : 0; }
/* RADIANCE_DS41_WALK_WM=1 (DRAFTER track, rule B): form 0 calls at R 256 take dspark_walk_wm_kernel; bench hook r4d_walk_wm. */
static int walk_wm_env() {
    const char* v = std::getenv("RADIANCE_DS41_WALK_WM");
    const int on = v && *v == '1' && !v[1];
    if (on) std::fprintf(stderr, "radiance: libr4d: dspark_walk_step takes the WMMA form at R 256 (RADIANCE_DS41_WALK_WM=1: rule B, the drafter's own reference)\n");
    return on;
}
static int g_walk_wm = walk_wm_env();
extern "C" void r4d_walk_wm(int on) { g_walk_wm = on ? 1 : 0; }
enum { WK_LOGITS = 0, WK_ANCHOR, WK_PREV, WK_PRED, WK_SUCC, WK_KEYS, WK_TOKENS };

/* An f32 [rows, 2 * width] plane read as [rows, width] u64 (a key slot is two of them): its row pitch in u64,
 * or -1. */
static long long wk_u64_plane(const RadTensor* t, long long rows, long long width) {
    if (!t || !t->data || t->dtype != RAD_F32 || t->rank != 2) return -1;
    if (t->shape[0] < rows || t->shape[1] < 2 * width) return -1;
    const long long p = r4d_row_pitch(t);
    if (p < 2 * width || (p & 1) || ((uintptr_t)t->data & 7)) return -1;
    return p / 2;
}

extern "C" int r4d_rad_dspark_walk_step(const RadArgs* a, RadStream stream) {
    const RadTensor* logits = r4d_opt(a, WK_LOGITS);
    const RadTensor* anchor = r4d_opt(a, WK_ANCHOR);
    const RadTensor* prev = r4d_opt(a, WK_PREV);
    const RadTensor* pred = r4d_opt(a, WK_PRED);
    const RadTensor* succ = r4d_opt(a, WK_SUCC);
    const RadTensor* keys = r4d_opt(a, WK_KEYS);
    const RadTensor* tokens = r4d_opt(a, WK_TOKENS);
    if (!logits || !anchor || !pred || !succ || !tokens) return R4D_NO(RAD_E_INVAL);

    const long long step = r4d_geti_or(a, "step", -1), steps = r4d_geti_or(a, "steps", 0);
    const long long R = r4d_geti_or(a, "rank", 0), V = r4d_geti_or(a, "n_vocab", 0);
    const long long N = r4d_geti_or(a, "N", 0), voff = r4d_geti_or(a, "vocab_off", 0);
    const long long world = r4d_geti_or(a, "world", 1), astride = r4d_geti_or(a, "anchor_stride", 1);
    /* form 0: a thread reads its own succ row (16-byte pieces); 1: the block's rows staged through LDS as whole lines
     * (lever W1b). The same fma chain a id, the same bits. */
    const long long form = r4d_geti_or(a, "form", 0);
    if (form < 0 || form > 1) return R4D_NO(RAD_E_INVAL);
    if (steps < 1 || steps > INT_MAX || step < 0 || step > steps || R < 32 || R > kWalkMaxRank || R % 32 ||
        V < 1 || V > INT_MAX - 1 || N < 0 || voff < 0 || voff + N > V || world < 1 || world > 64 || astride < 1)
        return R4D_NO(RAD_E_INVAL);

    if (tokens->dtype != RAD_I32 || tokens->rank != 2 || tokens->shape[1] < steps) return R4D_NO(RAD_E_SHAPE);
    const long long S = tokens->shape[0], tpitch = r4d_row_pitch(tokens);
    if (S < 1 || S > INT_MAX / steps || tpitch < steps) return R4D_NO(RAD_E_SHAPE);
    if (anchor->dtype != RAD_I32 || !r4d_contig(anchor) || r4d_numel(anchor) < (S - 1) * astride + 1)
        return R4D_NO(RAD_E_SHAPE);
    if (pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 || !r4d_contig(pred) || !r4d_contig(succ) ||
        r4d_numel(pred) < V * R || r4d_numel(succ) < V * R || ((uintptr_t)succ->data & 15))
        return R4D_NO(RAD_E_SHAPE);
    long long ppitch = 0, kpitch = 0, lpitch = 0;
    if (step > 0 && (ppitch = wk_u64_plane(prev, S, 2 * world)) < 0) return R4D_NO(RAD_E_SHAPE);
    if (step < steps) {
        if ((kpitch = wk_u64_plane(keys, S, 2)) < 0) return R4D_NO(RAD_E_SHAPE);
        if (logits->dtype != RAD_F32 || logits->rank != 2 || logits->shape[0] < S * steps ||
            logits->shape[1] < N || (lpitch = r4d_row_pitch(logits)) < N || lpitch < 1)
            return R4D_NO(RAD_E_SHAPE);
    }
    /* A rank whose shard is empty still merges and writes the tokens: one block. */
    const unsigned blocks = step < steps && N > 0 ? (unsigned)((N + kWalkThreads - 1) / kWalkThreads) : 1u;
    const float* lg = (const float*)logits->data;
    const unsigned long long* pv = step > 0 ? (const unsigned long long*)prev->data : nullptr;
    unsigned long long* ks = step < steps ? (unsigned long long*)keys->data : nullptr;
    /* DRAFTER track 18.11.2026: passes of 16 sequences (bench hook r4d_walk_seqs pins 8 / 16; each sequence's chain is the same,
     * so the same bits); the rule keeps 8 */
    if (g_walk_wm && form == 0 && R == kWmR) {
        const unsigned wb = step < steps && N > 0 ? (unsigned)(((N + 15) / 16 + kWalkWaves * 4 - 1) / (kWalkWaves * 4)) : 1u;
        hipLaunchKernelGGL(dspark_walk_wm_kernel, dim3(wb), dim3(kWalkThreads), 0, (hipStream_t)stream,
                           lg, (long)lpitch, (const int*)anchor->data, (long)astride, pv, (long)ppitch, (int)world,
                           (const r4d_u16*)pred->data, (const r4d_u16*)succ->data, ks, (long)kpitch,
                           (int*)tokens->data, (long)tpitch, (int)S, (int)step, (int)steps, (int)V, (int)N, (int)voff);
        return R4D_RAW((int)hipGetLastError());
    }
    const int seqs = g_walk_seqs > 0 ? g_walk_seqs : kWalkSeqs;   /* passes of 16: -10 % at S 16, +32 % at S 32 (wbench): pin only */
    if (seqs == 16) {
        if (form == 1 && R <= kStageMaxRank)
            hipLaunchKernelGGL((dspark_walk_step_kernel<true, 16>), dim3(blocks), dim3(kWalkThreads), 0, (hipStream_t)stream,
                               lg, (long)lpitch, (const int*)anchor->data, (long)astride, pv, (long)ppitch, (int)world,
                               (const r4d_u16*)pred->data, (const r4d_u16*)succ->data, ks, (long)kpitch,
                               (int*)tokens->data, (long)tpitch, (int)S, (int)step, (int)steps, (int)R, (int)V, (int)N,
                               (int)voff);
        else
            hipLaunchKernelGGL((dspark_walk_step_kernel<false, 16>), dim3(blocks), dim3(kWalkThreads), 0, (hipStream_t)stream,
                               lg, (long)lpitch, (const int*)anchor->data, (long)astride, pv, (long)ppitch, (int)world,
                               (const r4d_u16*)pred->data, (const r4d_u16*)succ->data, ks, (long)kpitch,
                               (int*)tokens->data, (long)tpitch, (int)S, (int)step, (int)steps, (int)R, (int)V, (int)N,
                               (int)voff);
        return R4D_RAW((int)hipGetLastError());
    }
    /* form 1 holds the predecessor rows of a rank up to 256 (DeepSeek-V4.1's); past it form 0 runs, the same bits */
    if (form == 1 && R <= kStageMaxRank)
        hipLaunchKernelGGL(dspark_walk_step_kernel<true>, dim3(blocks), dim3(kWalkThreads), 0, (hipStream_t)stream,
                           lg, (long)lpitch, (const int*)anchor->data, (long)astride, pv, (long)ppitch, (int)world,
                           (const r4d_u16*)pred->data, (const r4d_u16*)succ->data, ks, (long)kpitch,
                           (int*)tokens->data, (long)tpitch, (int)S, (int)step, (int)steps, (int)R, (int)V, (int)N,
                           (int)voff);
    else
        hipLaunchKernelGGL(dspark_walk_step_kernel<false>, dim3(blocks), dim3(kWalkThreads), 0, (hipStream_t)stream,
                           lg, (long)lpitch, (const int*)anchor->data, (long)astride, pv, (long)ppitch, (int)world,
                           (const r4d_u16*)pred->data, (const r4d_u16*)succ->data, ks, (long)kpitch,
                           (int*)tokens->data, (long)tpitch, (int)S, (int)step, (int)steps, (int)R, (int)V, (int)N,
                           (int)voff);
    return R4D_RAW((int)hipGetLastError());
}
