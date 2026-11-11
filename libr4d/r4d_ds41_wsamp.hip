// r4d_ds41_wsamp.hip -- lever SD: sampled drafting on lever W1's DSpark Markov walk over the whole vocabulary, one walk
// step on one rank's vocabulary shard (op dspark_walk_sample_step; specs and row r4d_ds41_wsamp.inc, operands
// r4d_ds41_wsamp_shapes.inc; the law and the state r4d_ds41_wsamp_math.h).
//
// deepseek41 issues it as it issues dspark_walk_step (r4d_ds41_walk.hip), `steps` + 1 times a query pass with an
// all_gather of the per-rank keys between steps, under fn4 draft_sample >= 1 (--draft-coupling on):
//
//   step l (0 < l <= steps): every block merges step l-1's gathered keys (world ranks x the eight best a rank) into the
//       eight best of the whole vocabulary, and the sequence's position l-1 (round l) becomes r4d_wsamp_step's token:
//       x ~ Q for a sampled row, the best key's id for a greedy one (lever W1's token). Block 0 writes the token to
//       tokens[s][l-1] and the round's state to state[l-1][s]; every block takes x as the predecessor;
//   step l < steps: every column of the shard is scored as lever W1 scores it (form 0: a thread reads its own succ row,
//       the same fma chain, so the same bits) and keyed as W1 keys it; a block's eight best a sequence go to its
//       scratch row (a wave's eight by repeated wave maxima, then the block's from the four waves'), and the LAST block
//       to arrive (an agent-scope counter, the split-K arrival of r4d_lora_bf16.hip) merges every block's eight into
//       the rank's eight best, keys[s][0 .. 8) (descending, 0 = none), which the plugin gathers before step l + 1;
//   step steps: the last position's merge only (one block).
//
// Keys travel in 64-byte rows (eight u64 = four whole 16-byte words: libr4d's all-gather moves whole words a row).
// Everything a block derives from the gathered bytes is a function of those bytes, so every block and every rank
// takes the same x, and the token, the state and the predecessor agree.
#include <hip/hip_runtime.h>
#include <climits>
#include <cstdint>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"   // r4d_bf2f, r4d_u16, r4d_u32x4, R4dV8
#include "r4d_plugin.h"
#include "r4d_ds41_wsamp_math.h"

namespace {

constexpr int kWsThreads = 128;
constexpr int kWsWaves = kWsThreads / 32;
constexpr int kWsSeqs = 8;          // sequences a pass
constexpr int kWsMaxRank = 512;
constexpr int kWsK = R4D_WSAMP_K;   // keys a sequence a rank (and a block)

__device__ __forceinline__ unsigned long long ws_umax(unsigned long long a, unsigned long long b) {
    return a > b ? a : b;
}
/* The wave's largest key (wave32), every lane. */
__device__ __forceinline__ unsigned long long ws_wave_max(unsigned long long k) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned hi = __shfl_xor((unsigned)(k >> 32), o, 32);
        const unsigned lo = __shfl_xor((unsigned)(k & 0xffffffffull), o, 32);
        k = ws_umax(k, ((unsigned long long)hi << 32) | lo);
    }
    return k;
}
/* The wave's eight greatest keys over one key a lane: lane i returns the i-th (descending), lanes 8 .. 31 and the
 * rounds past the last live key 0. Keys are unique (the id is in them), so each round's maximum is one lane's. */
__device__ __forceinline__ unsigned long long ws_wave_top8(unsigned long long k, int lane) {
    unsigned long long mine = 0ull;
    for (int i = 0; i < kWsK; ++i) {
        const unsigned long long m = ws_wave_max(k);
        if (lane == i) mine = m;
        if (m == 0ull) break;   // the same m on every lane: a uniform exit
        if (k == m) k = 0ull;
    }
    return mine;
}

/* lane `src`'s key, every lane */
__device__ __forceinline__ unsigned long long ws_shfl64(unsigned long long k, int src) {
    const unsigned hi = (unsigned)__shfl((int)(unsigned)(k >> 32), src, 32);
    const unsigned lo = (unsigned)__shfl((int)(unsigned)(k & 0xffffffffull), src, 32);
    return ((unsigned long long)hi << 32) | lo;
}

struct WsArgs {
    const float* logits; long lpitch;
    const int* anchor; long astride;
    const unsigned long long* prev; long ppitch;   // [S][world * 8] u64 (an f32 plane read as u64)
    const r4d_u16* pred; const r4d_u16* succ;
    unsigned long long* keys; long kpitch;          // [S][8] u64
    int* tokens; long tpitch;
    unsigned long long* scr;                        // [S][nblk][8] u64
    unsigned* cnt;                                  // the step's arrival counter
    const int* seq;                                 // the query batch's card a row (RadBatch::seq_ids)
    const uint32_t* draw; long dpitch;              // [S][RAD_MTP_COUPLE_DRAW_WORDS]
    const float* qt; long qstride;                  // [S] q_temp a row (the request's temperature x the scale)
    uint32_t* state; long st_round, st_row;         // [rounds][rows][RAD_MTP_COUPLE_STATE_WORDS]
    int S, step, steps, R, V, N, voff, world, nblk;
};

__global__ __launch_bounds__(kWsThreads) void dspark_wsamp_kernel(WsArgs A) {
    __shared__ float su[kWsSeqs * kWsMaxRank];
    __shared__ int sprev[kWsSeqs];
    __shared__ unsigned long long sw8[kWsSeqs * kWsWaves * kWsK];
    __shared__ unsigned sarr;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int j = blockIdx.x * kWsThreads + tid;
    const int R = A.R, step = A.step, steps = A.steps;
    const bool live = step < steps && j < A.N;

    for (int g0 = 0; g0 < A.S; g0 += kWsSeqs) {
        const int ng = A.S - g0 < kWsSeqs ? A.S - g0 : kWsSeqs;
        /* The predecessors: the anchor at step 0, else position step-1's token from the gathered keys, a wave a
         * sequence (wave w takes w, w + 4). */
        if (step == 0) {
            if (tid < ng) sprev[tid] = A.anchor[(long)(g0 + tid) * A.astride];
        } else {
            for (int sl = wave; sl < ng; sl += kWsWaves) {
                const int s = g0 + sl;
                const int nk = A.world * kWsK;
                const unsigned long long k = lane < nk ? A.prev[(long)s * A.ppitch + lane] : 0ull;
                const unsigned long long t = ws_wave_top8(k, lane);
                uint64_t top[kWsK];   // this sequence's eight, descending, on every lane
#pragma unroll
                for (int i = 0; i < kWsK; ++i) top[i] = ws_shfl64(t, i);
                if (lane == 0) {
                    const RadMtpCoupleDraw d = *reinterpret_cast<const RadMtpCoupleDraw*>(A.draw + (long)s * A.dpitch);
                    const float q = A.qt ? A.qt[(long)s * A.qstride] : 1.0f;
                    RadMtpCoupleState st;
                    const int32_t x = r4d_wsamp_step(top, q, d, d.card == A.seq[s], step, (int64_t)A.V, &st);
                    if (blockIdx.x == 0) {
                        A.tokens[(long)s * A.tpitch + step - 1] = x;
                        *reinterpret_cast<RadMtpCoupleState*>(A.state + (long)(step - 1) * A.st_round +
                                                              (long)s * A.st_row) = st;
                    }
                    sprev[sl] = x;
                }
            }
        }
        if (step == steps) continue;   // the last merge only (one block): nothing to score
        __syncthreads();
        for (int i = tid; i < ng * R; i += kWsThreads) {
            const int sl = i / R, r = i - sl * R;
            const int p = sprev[sl];
            su[sl * R + r] = (p >= 0 && p < A.V) ? r4d_bf2f(A.pred[(long)p * R + r]) : 0.0f;
        }
        __syncthreads();

        /* lever W1's form 0, the same chain a column */
        float acc[kWsSeqs];
#pragma unroll
        for (int sl = 0; sl < kWsSeqs; ++sl) acc[sl] = 0.0f;
        if (live) {
            const r4d_u16* __restrict__ w = A.succ + (long)(A.voff + j) * R;
#pragma unroll 4
            for (int r0 = 0; r0 < R; r0 += 8) {
                R4dV8 v;
                v.v = *(const r4d_u32x4*)(w + r0);
                float wf[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) wf[q] = r4d_bf2f(v.h[q]);
#pragma unroll
                for (int sl = 0; sl < kWsSeqs; ++sl) {
                    if (sl < ng) {
                        const float* u = su + sl * R + r0;
#pragma unroll
                        for (int q = 0; q < 8; ++q) acc[sl] = __fmaf_rn(u[q], wf[q], acc[sl]);
                    }
                }
            }
        }
        /* Each wave's eight a sequence, then the block's eight from the four waves' 32 (wave w takes w, w + 4). */
        for (int sl = 0; sl < ng; ++sl) {
            unsigned long long k = 0ull;
            if (live) k = r4d_wsamp_key(A.logits[(long)((g0 + sl) * steps + step) * A.lpitch + j] + acc[sl], A.voff + j);
            const unsigned long long t = ws_wave_top8(k, lane);
            if (lane < kWsK) sw8[(sl * kWsWaves + wave) * kWsK + lane] = t;
        }
        __syncthreads();
        for (int sl = wave; sl < ng; sl += kWsWaves) {
            const unsigned long long k = sw8[sl * kWsWaves * kWsK + lane];   // 32 = kWsWaves * kWsK keys
            const unsigned long long t = ws_wave_top8(k, lane);
            if (lane < kWsK) A.scr[((long)(g0 + sl) * A.nblk + blockIdx.x) * kWsK + lane] = t;
        }
        __syncthreads();   // sprev / su / sw8 are the next pass's
    }
    if (step == steps) return;

    /* The rank's eight a sequence, in the last block to arrive. Every block's scratch stores are fenced before its
     * arrival; the counter is left zero (the plugin also zeroes it every query pass). */
    __threadfence();
    __syncthreads();
    if (tid == 0) sarr = __hip_atomic_fetch_add(A.cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (sarr != (unsigned)(A.nblk - 1)) return;
    if (tid == 0) A.cnt[0] = 0u;
    __threadfence();
    const int nk = A.nblk * kWsK;
    for (int s = wave; s < A.S; s += kWsWaves) {
        const unsigned long long* src = A.scr + (long)s * nk;
        uint64_t loc[kWsK];
#pragma unroll
        for (int i = 0; i < kWsK; ++i) loc[i] = 0ull;
        for (int q = lane; q < nk; q += 32) r4d_wsamp_insert(loc, src[q]);
        unsigned long long mine = 0ull;
        for (int i = 0; i < kWsK; ++i) {
            const unsigned long long h = loc[0];
            const unsigned long long m = ws_wave_max(h);
            if (lane == i) mine = m;
            if (m == 0ull) break;
            if (h == m) {
#pragma unroll
                for (int t = 0; t < kWsK - 1; ++t) loc[t] = loc[t + 1];
                loc[kWsK - 1] = 0ull;
            }
        }
        if (lane < kWsK) A.keys[(long)s * A.kpitch + lane] = mine;
    }
}

}  // namespace

/* ------------------------------------------------------------------ plugin entry */
enum { WS_LOGITS = 0, WS_ANCHOR, WS_PREV, WS_PRED, WS_SUCC, WS_KEYS, WS_TOKENS, WS_SCR, WS_CNT, WS_SEQ, WS_DRAW,
       WS_QT, WS_STATE };

/* An f32 [rows, >= 2 * width] plane read as [rows, width] u64: its row pitch in u64, or -1. */
static long long ws_u64_plane(const RadTensor* t, long long rows, long long width) {
    if (!t || !t->data || t->dtype != RAD_F32 || t->rank != 2) return -1;
    if (t->shape[0] < rows || t->shape[1] < 2 * width) return -1;
    const long long p = r4d_row_pitch(t);
    if (p < 2 * width || (p & 1) || ((uintptr_t)t->data & 7)) return -1;
    return p / 2;
}

extern "C" int r4d_rad_dspark_walk_sample_step(const RadArgs* a, RadStream stream) {
    const RadTensor* logits = r4d_opt(a, WS_LOGITS);
    const RadTensor* anchor = r4d_opt(a, WS_ANCHOR);
    const RadTensor* prev = r4d_opt(a, WS_PREV);
    const RadTensor* pred = r4d_opt(a, WS_PRED);
    const RadTensor* succ = r4d_opt(a, WS_SUCC);
    const RadTensor* keys = r4d_opt(a, WS_KEYS);
    const RadTensor* tokens = r4d_opt(a, WS_TOKENS);
    const RadTensor* scr = r4d_opt(a, WS_SCR);
    const RadTensor* cnt = r4d_opt(a, WS_CNT);
    const RadTensor* seq = r4d_opt(a, WS_SEQ);
    const RadTensor* draw = r4d_opt(a, WS_DRAW);
    const RadTensor* qt = r4d_opt(a, WS_QT);
    const RadTensor* state = r4d_opt(a, WS_STATE);
    if (!logits || !anchor || !pred || !succ || !tokens || !seq || !draw || !state) return R4D_NO(RAD_E_INVAL);

    const long long step = r4d_geti_or(a, "step", -1), steps = r4d_geti_or(a, "steps", 0);
    const long long R = r4d_geti_or(a, "rank", 0), V = r4d_geti_or(a, "n_vocab", 0);
    const long long N = r4d_geti_or(a, "N", 0), voff = r4d_geti_or(a, "vocab_off", 0);
    const long long world = r4d_geti_or(a, "world", 1), astride = r4d_geti_or(a, "anchor_stride", 1);
    if (steps < 1 || steps > RAD_MTP_COUPLE_MAX_ROUNDS || step < 0 || step > steps || R < 32 || R > kWsMaxRank ||
        R % 32 || V < 1 || V > INT_MAX - 1 || N < 0 || voff < 0 || voff + N > V || world < 1 || world > 4 ||
        astride < 1)
        return R4D_NO(RAD_E_INVAL);

    if (tokens->dtype != RAD_I32 || tokens->rank != 2 || tokens->shape[1] < steps) return R4D_NO(RAD_E_SHAPE);
    const long long S = tokens->shape[0], tpitch = r4d_row_pitch(tokens);
    if (S < 1 || S > INT_MAX / steps || tpitch < steps) return R4D_NO(RAD_E_SHAPE);
    if (anchor->dtype != RAD_I32 || !r4d_contig(anchor) || r4d_numel(anchor) < (S - 1) * astride + 1)
        return R4D_NO(RAD_E_SHAPE);
    if (pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 || !r4d_contig(pred) || !r4d_contig(succ) ||
        r4d_numel(pred) < V * R || r4d_numel(succ) < V * R || ((uintptr_t)succ->data & 15))
        return R4D_NO(RAD_E_SHAPE);
    /* the batch's cards, the draws, the q_temp column (absent: 1.0) and the state plane [rounds, rows, words] */
    if ((seq->dtype != RAD_I32 && seq->dtype != RAD_U32) || !r4d_contig(seq) || r4d_numel(seq) < S)
        return R4D_NO(RAD_E_SHAPE);
    if ((draw->dtype != RAD_U32 && draw->dtype != RAD_I32) || draw->rank != 2 || draw->shape[0] < S ||
        draw->shape[1] < (int64_t)RAD_MTP_COUPLE_DRAW_WORDS || ((uintptr_t)draw->data & 7) ||
        (r4d_row_pitch(draw) & 1))
        return R4D_NO(RAD_E_SHAPE);
    const long long qstride = 1;   /* q_temp: S contiguous f32 ([rows, 1] or [rows]) */
    if (qt && (!qt->data || qt->dtype != RAD_F32 || !r4d_contig(qt) || r4d_numel(qt) < S)) return R4D_NO(RAD_E_SHAPE);
    if ((state->dtype != RAD_U32 && state->dtype != RAD_I32) || state->rank != 3 || !r4d_contig(state) ||
        state->shape[0] < steps || state->shape[1] < S || state->shape[2] != (int64_t)RAD_MTP_COUPLE_STATE_WORDS ||
        ((uintptr_t)state->data & 7))
        return R4D_NO(RAD_E_SHAPE);
    long long ppitch = 0, kpitch = 0, lpitch = 0;
    const long long nblk = step < steps && N > 0 ? (N + kWsThreads - 1) / kWsThreads : 1;
    if (step > 0 && (ppitch = ws_u64_plane(prev, S, (long long)world * kWsK)) < 0) return R4D_NO(RAD_E_SHAPE);
    if (step < steps) {
        if ((kpitch = ws_u64_plane(keys, S, kWsK)) < 0) return R4D_NO(RAD_E_SHAPE);
        if (logits->dtype != RAD_F32 || logits->rank != 2 || logits->shape[0] < S * steps ||
            logits->shape[1] < N || (lpitch = r4d_row_pitch(logits)) < N || lpitch < 1)
            return R4D_NO(RAD_E_SHAPE);
        /* the scratch: [S * nblk, 16] f32 tight (eight u64 a block a sequence); the counter one u32 */
        if (!scr || !scr->data || scr->dtype != RAD_F32 || !r4d_contig(scr) || r4d_numel(scr) < S * nblk * 2 * kWsK ||
            ((uintptr_t)scr->data & 7))
            return R4D_NO(RAD_E_SHAPE);
        if (!cnt || !cnt->data || r4d_numel(cnt) < 1 || ((uintptr_t)cnt->data & 3)) return R4D_NO(RAD_E_SHAPE);
    }
    WsArgs A{};
    A.logits = (const float*)logits->data; A.lpitch = (long)lpitch;
    A.anchor = (const int*)anchor->data; A.astride = (long)astride;
    A.prev = step > 0 ? (const unsigned long long*)prev->data : nullptr; A.ppitch = (long)ppitch;
    A.pred = (const r4d_u16*)pred->data; A.succ = (const r4d_u16*)succ->data;
    A.keys = step < steps ? (unsigned long long*)keys->data : nullptr; A.kpitch = (long)kpitch;
    A.tokens = (int*)tokens->data; A.tpitch = (long)tpitch;
    A.scr = step < steps ? (unsigned long long*)scr->data : nullptr;
    A.cnt = step < steps ? (unsigned*)cnt->data : nullptr;
    A.seq = (const int*)seq->data;
    A.draw = (const uint32_t*)draw->data; A.dpitch = (long)r4d_row_pitch(draw);
    A.qt = qt ? (const float*)qt->data : nullptr; A.qstride = (long)qstride;
    A.state = (uint32_t*)state->data;
    A.st_row = (long)RAD_MTP_COUPLE_STATE_WORDS;
    A.st_round = (long)(state->shape[1] * (int64_t)RAD_MTP_COUPLE_STATE_WORDS);
    A.S = (int)S; A.step = (int)step; A.steps = (int)steps; A.R = (int)R; A.V = (int)V; A.N = (int)N;
    A.voff = (int)voff; A.world = (int)world; A.nblk = (int)nblk;
    hipLaunchKernelGGL(dspark_wsamp_kernel, dim3((unsigned)nblk), dim3(kWsThreads), 0, (hipStream_t)stream, A);
    return R4D_RAW((int)hipGetLastError());
}
