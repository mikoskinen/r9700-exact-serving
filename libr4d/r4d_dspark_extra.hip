/* r4d_dspark_extra.hip -- two small ops beside DSpark's Markov walk (arch/deepseek41), one workgroup a
 * sequence, written against r4d_launch.h so tests/r4d_emu runs the same source on the host.
 *
 * dspark_conf: the release's confidence head (inference/model.py DSparkConfidenceHead, mtp.{n-1}.
 * confidence_head.proj, [1, D + R], no bias) over a query pass's rows:
 *
 *   conf[s, j] = bias + sum_k w[k] * x[s * steps + j][k] + sum_r w[D + r] * pred[pid(s, j)][r]
 *   pid(s, 0) = anchor[s * anchor_stride]      the block's anchor
 *   pid(s, j) = tokens[s, j - 1]               the walk's own proposal before row j
 *
 * x is the head's input before its final norm (forward_head's hc_pre output), pred the Markov
 * head's predecessor table (markov_head.embed, the walk's W1): row j's input is cat(x_j, W1[the token
 * row j predicts after]), the token proposal j then follows. The value is a logit; the reference
 * applies nothing to it, exllamav3 1.6.5 keeps the longest prefix with sigmoid >= 0.5. Here it only
 * says how many proposals a sequence verifies (the engine's dspark_conf switch): it changes no
 * proposal and no emitted token. An out-of-range predecessor contributes nothing. f32 throughout;
 * thread t sums k = t, t + 256, ... then its predecessor column t (t < R), then a fixed tree.
 *
 * dspark_alt: a sequence's next-best first proposals, each continued by the walk, for the engine's
 * shadow of a position-1 tree (no proposal it makes is verified). Row 0's candidates are scored as
 * dflash_select scores them (unary + <pred[anchor], succ[cand]>); the `n_alt` best whose id is valid
 * and not the walk's own first proposal (tokens[s, 0]), ties to the lower candidate index, are the
 * branches' first tokens; each branch then walks rows 1 .. steps - 1 greedily, its own pick the next
 * predecessor (dflash_select's rule). alt[s, a * steps + l] is branch a's token at position l; a
 * branch with no candidate left is -1 throughout. Eight lanes a candidate (k = lane, lane + 8, ...),
 * then the eight partials in order: the order differs from dflash_select's wave sum, so a near-tie
 * may rank otherwise there; an alternative only feeds a measurement.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_dspx {

typedef unsigned short u16;
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float((unsigned)v << 16); }

constexpr int kThreads = 256;
constexpr int kMaxK = 32;      /* candidates; eight lanes each */
constexpr int kMaxR = 512;     /* the predecessor row in LDS */
constexpr int kMaxAlt = 4;

struct ConfArgs {
    const u16* x; long long x_ld;
    const int* anchor; long long astr;
    const int* tok; long long t_ld;
    const u16* pred;
    const float* w;
    float* out; long long o_ld;
    int steps, D, R, V;
    float bias;
};

__global__ __launch_bounds__(256) void conf_kernel(ConfArgs A) {
    __shared__ float part[kThreads];
    __shared__ int spid;
    const int s = (int)blockIdx.x, j = (int)blockIdx.y, t = (int)threadIdx.x;
    if (t == 0) spid = j == 0 ? A.anchor[(long long)s * A.astr] : A.tok[(long long)s * A.t_ld + j - 1];
    __syncthreads();
    const u16* xr = A.x + ((long long)s * A.steps + j) * A.x_ld;
    float acc = 0.0f;
    for (int k = t; k < A.D; k += kThreads) acc += A.w[k] * bf2f(xr[k]);
    const int pid = spid;
    if (pid >= 0 && pid < A.V) {
        const u16* pr = A.pred + (long long)pid * A.R;
        for (int r = t; r < A.R; r += kThreads) acc += A.w[A.D + r] * bf2f(pr[r]);
    }
    part[t] = acc;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o >>= 1) {
        if (t < o) part[t] += part[t + o];
        __syncthreads();
    }
    if (t == 0) A.out[(long long)s * A.o_ld + j] = part[0] + A.bias;
}

struct AltArgs {
    const int* cand; const float* unary; long long c_ld;
    const int* anchor; long long astr;
    const int* tok; long long t_ld;
    const u16* pred; const u16* succ;
    int* out; long long o_ld;
    int steps, K, R, V, n_alt;
};

/* Every candidate of row `row` scored against predecessor `pid` into sc[0..K): eight lanes a
 * candidate, partials summed in lane order. A candidate outside [0, V) scores its unary term. */
__device__ __forceinline__ void alt_score(const AltArgs& A, long long row, int pid, float* sa, float* p8,
                                          float* sc) {
    const int t = (int)threadIdx.x;
    const bool pok = pid >= 0 && pid < A.V;
    for (int r = t; r < A.R; r += kThreads) sa[r] = pok ? bf2f(A.pred[(long long)pid * A.R + r]) : 0.0f;
    __syncthreads();
    const int c = t >> 3, l8 = t & 7;
    float acc = 0.0f;
    if (c < A.K) {
        const int id = A.cand[row * A.c_ld + c];
        if (id >= 0 && id < A.V) {
            const u16* sr = A.succ + (long long)id * A.R;
            for (int r = l8; r < A.R; r += 8) acc += sa[r] * bf2f(sr[r]);
        }
    }
    p8[t] = acc;
    __syncthreads();
    if (l8 == 0 && c < A.K) {
        float v = A.unary[row * A.c_ld + c];
        float d = 0.0f;
        for (int i = 0; i < 8; ++i) d += p8[t + i];
        sc[c] = v + d;
    }
    __syncthreads();
}

__global__ __launch_bounds__(256) void alt_kernel(AltArgs A) {
    __shared__ float sa[kMaxR];
    __shared__ float p8[kThreads];
    __shared__ float sc[kMaxK];
    __shared__ int first[kMaxAlt];
    __shared__ int spid;
    const int s = (int)blockIdx.x, t = (int)threadIdx.x;
    const long long row0 = (long long)s * A.steps;
    alt_score(A, row0, A.anchor[(long long)s * A.astr], sa, p8, sc);
    if (t == 0) {
        const int d1 = A.tok[(long long)s * A.t_ld];
        bool used[kMaxK];
        for (int c = 0; c < A.K; ++c) {
            const int id = A.cand[row0 * A.c_ld + c];
            used[c] = id < 0 || id >= A.V || id == d1;
        }
        for (int a = 0; a < A.n_alt; ++a) {
            int best = -1;
            float bv = -__builtin_inff();
            for (int c = 0; c < A.K; ++c)
                if (!used[c] && (best < 0 || sc[c] > bv)) { bv = sc[c]; best = c; }
            if (best >= 0) used[best] = true;
            first[a] = best >= 0 ? A.cand[row0 * A.c_ld + best] : -1;
        }
    }
    __syncthreads();
    for (int a = 0; a < A.n_alt; ++a) {
        const int f = first[a];
        int* o = A.out + (long long)s * A.o_ld + (long long)a * A.steps;
        if (f < 0) {
            for (int l = t; l < A.steps; l += kThreads) o[l] = -1;
            continue;                               /* uniform: `first` is shared */
        }
        if (t == 0) { o[0] = f; spid = f; }
        __syncthreads();
        for (int l = 1; l < A.steps; ++l) {
            alt_score(A, row0 + l, spid, sa, p8, sc);
            if (t == 0) {
                int best = 0;
                float bv = -__builtin_inff();
                for (int c = 0; c < A.K; ++c)
                    if (sc[c] > bv) { bv = sc[c]; best = c; }
                const int tok = A.cand[(row0 + l) * A.c_ld + best];
                o[l] = tok;
                spid = tok;
            }
            __syncthreads();
        }
    }
}

/* [rows, width] at a row pitch, the last axis dense; -1 when not. */
static long long plane(const RadTensor* t, uint32_t dtype, long long width, long long rows) {
    if (!t || !t->data || t->dtype != dtype) return -1;
    if (t->rank == 1) return (t->shape[0] >= width * rows && t->stride[0] == 1) ? width : -1;
    if (t->rank != 2 || t->shape[1] < width || t->shape[0] < rows || t->stride[1] != 1) return -1;
    return t->stride[0] >= width ? t->stride[0] : -1;
}

static long long numel(const RadTensor* t) {
    long long n = 1;
    for (uint32_t i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

}  /* namespace r4d_dspx */

/* x, anchor, tokens, pred w, w, out. M is the output's row count (the band's bound arrives as the
 * parameter), steps rows of x a sequence. */
extern "C" int r4d_rad_dspark_conf(const RadArgs* a, RadStream stream) {
    using namespace r4d_dspx;
    if (!a || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *x = rad_arg_in(a, 0), *anc = rad_arg_in(a, 1), *tok = rad_arg_in(a, 2);
    const RadTensor *pr = rad_arg_in(a, 3), *w = rad_arg_in(a, 4), *out = rad_arg_in(a, 5);
    if (!x || !anc || !tok || !pr || !w || !out) return RAD_E_INVAL;
    const long long steps = rad_args_geti_or(a, "steps", 0), D = rad_args_geti_or(a, "n_embd", 0);
    const long long R = rad_args_geti_or(a, "rank", 0), V = rad_args_geti_or(a, "n_vocab", 0);
    const long long astr = rad_args_geti_or(a, "anchor_stride", 1);
    if (out->rank < 1) return RAD_E_SHAPE;
    const long long M = out->shape[0];
    if (M <= 0) return RAD_OK;
    if (steps < 1 || D < 1 || R < 1 || V < 1 || astr < 1) return RAD_E_INVAL;
    if (x->dtype != RAD_BF16 || pr->dtype != RAD_BF16 || w->dtype != RAD_F32 || out->dtype != RAD_F32 ||
        (anc->dtype != RAD_I32 && anc->dtype != RAD_U32) || (tok->dtype != RAD_I32 && tok->dtype != RAD_U32))
        return RAD_E_DTYPE;
    const long long x_ld = plane(x, RAD_BF16, D, M * steps);
    const long long o_ld = plane(out, RAD_F32, steps, M);
    const long long t_ld = steps > 1 ? plane(tok, tok->dtype, steps - 1, M) : 1;
    if (x_ld < 0 || o_ld < 0 || t_ld < 0) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(w) < D + R || numel(pr) < V * R) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(pr) || !rad_tensor_is_contiguous(w) || !rad_tensor_is_contiguous(anc))
        return RAD_E_STRIDE;
    double bias = 0.0;
    for (int i = 0; i < a->n_p; ++i)
        if (a->p[i].key && !std::strcmp(a->p[i].key, "bias"))
            bias = a->p[i].kind == RAD_P_F64 ? a->p[i].dval : (double)a->p[i].ival;
    ConfArgs A{};
    A.x = (const u16*)x->data; A.x_ld = x_ld;
    A.anchor = (const int*)anc->data; A.astr = astr;
    A.tok = (const int*)tok->data; A.t_ld = t_ld;
    A.pred = (const u16*)pr->data;
    A.w = (const float*)w->data;
    A.out = (float*)out->data; A.o_ld = o_ld;
    A.steps = (int)steps; A.D = (int)D; A.R = (int)R; A.V = (int)V;
    A.bias = (float)bias;
    R4D_KLAUNCH(R4D_KNAME(r4d_dspx::conf_kernel), dim3((unsigned)M, (unsigned)steps), dim3(kThreads),
                (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* cand, unary, anchor, tokens, pred w, succ w, alt. */
extern "C" int r4d_rad_dspark_alt(const RadArgs* a, RadStream stream) {
    using namespace r4d_dspx;
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *cand = rad_arg_in(a, 0), *un = rad_arg_in(a, 1), *anc = rad_arg_in(a, 2);
    const RadTensor *tok = rad_arg_in(a, 3), *pr = rad_arg_in(a, 4), *sc = rad_arg_in(a, 5);
    const RadTensor *out = rad_arg_in(a, 6);
    if (!cand || !un || !anc || !tok || !pr || !sc || !out) return RAD_E_INVAL;
    const long long steps = rad_args_geti_or(a, "steps", 0), K = rad_args_geti_or(a, "top_k", 0);
    const long long R = rad_args_geti_or(a, "rank", 0), V = rad_args_geti_or(a, "n_vocab", 0);
    const long long astr = rad_args_geti_or(a, "anchor_stride", 1), n_alt = rad_args_geti_or(a, "n_alt", 0);
    if (out->rank < 1) return RAD_E_SHAPE;
    const long long M = out->shape[0];
    if (M <= 0) return RAD_OK;
    if (steps < 1 || K < 1 || K > kMaxK || R < 1 || R > kMaxR || V < 1 || astr < 1 || n_alt < 1 ||
        n_alt > kMaxAlt)
        return RAD_E_INVAL;
    if (cand->dtype != RAD_I32 || un->dtype != RAD_F32 || out->dtype != RAD_I32 || pr->dtype != RAD_BF16 ||
        sc->dtype != RAD_BF16 || (anc->dtype != RAD_I32 && anc->dtype != RAD_U32) ||
        (tok->dtype != RAD_I32 && tok->dtype != RAD_U32))
        return RAD_E_DTYPE;
    const long long c_ld = plane(cand, RAD_I32, K, M * steps), u_ld = plane(un, RAD_F32, K, M * steps);
    const long long t_ld = plane(tok, tok->dtype, 1, M), o_ld = plane(out, RAD_I32, n_alt * steps, M);
    if (c_ld < 0 || u_ld < 0 || t_ld < 0 || o_ld < 0) return RAD_E_SHAPE;
    if (c_ld != u_ld) return RAD_E_STRIDE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(pr) < V * R || numel(sc) < V * R) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(pr) || !rad_tensor_is_contiguous(sc) || !rad_tensor_is_contiguous(anc))
        return RAD_E_STRIDE;
    AltArgs A{};
    A.cand = (const int*)cand->data; A.unary = (const float*)un->data; A.c_ld = c_ld;
    A.anchor = (const int*)anc->data; A.astr = astr;
    A.tok = (const int*)tok->data; A.t_ld = t_ld;
    A.pred = (const u16*)pr->data; A.succ = (const u16*)sc->data;
    A.out = (int*)out->data; A.o_ld = o_ld;
    A.steps = (int)steps; A.K = (int)K; A.R = (int)R; A.V = (int)V; A.n_alt = (int)n_alt;
    R4D_KLAUNCH(R4D_KNAME(r4d_dspx::alt_kernel), dim3((unsigned)M), dim3(kThreads), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
