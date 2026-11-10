// r4d_tree.hip -- a tree verify's bookkeeping for DeepSeek-V4.1's DSpark (lever T1): the rows' tree
// table (tree_rows), the context pass's commits of the accepted path's per-node state
// (tree_commit_entries, tree_commit_rows) and the drafter's tree proposal (dspark_tree_assemble).
// libref/ref_tree.cpp is the oracle, element for element; docs/OPS.md "Tree verify (DSpark)".
//
// All four are a handful of integers or one entry a row: one thread a row (tree_rows, assemble), one
// workgroup a row (entries), one workgroup a sequence with its rows serial (rows: a later move may
// read what an earlier one left, so the order is the contract).

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <climits>
#include <cstdint>

namespace {

constexpr int kTrThreads = 256;

/* ---- tree_rows: one thread a row. The walk to the root through spec_parent (a parent row inside
 * the sequence, before the row), at most width - 3 steps; a root -1; anything else an ordinary row. */
struct TreeRowsArgs {
    const int32_t* sp;
    const int32_t* cu;
    const int32_t* pos;
    int32_t* tt;
    int32_t* spos;
    int32_t* tmark;
    long tr;              /* tt's row pitch */
    int M, S, W;
};

__global__ __launch_bounds__(kTrThreads) void tree_rows_kernel(TreeRowsArgs a) {
    const int m = (int)(blockIdx.x * kTrThreads + threadIdx.x);
    if (m >= a.M) return;
    int s = -1;
    for (int i = 0; i < a.S; ++i)
        if (m >= a.cu[i] && m < a.cu[i + 1]) { s = i; break; }
    const int dmax = a.W - 3;
    int d = -1, root = -1;
    if (s >= 0) {
        const int b0 = a.cu[s];
        int r = m, n = 0;
        bool ok = true;
        for (;;) {
            const int p = a.sp[r];
            if (p == -1) { root = r; break; }
            if (p < 0 || n >= dmax) { ok = false; break; }
            const int pr = b0 + p;
            if (pr < b0 || pr >= r) { ok = false; break; }
            r = pr;
            ++n;
        }
        if (ok) d = n;
    }
    int32_t* row = a.tt + (long)m * a.tr;
    for (int c = 0; c < a.W; ++c) row[c] = -1;
    if (d >= 0) {
        row[0] = d;
        row[1] = root;
        int r = m;
        const int b0 = a.cu[s];
        for (int k = d; k >= 0; --k) {
            row[2 + k] = r;
            if (k > 0) r = b0 + a.sp[r];
        }
    }
    a.spos[m] = d >= 0 ? a.pos[root] + (m - root) : a.pos[m];
    a.tmark[m] = d >= 1 ? 1 : 0;
}

/* ---- tree_commit_entries: one workgroup a row; 16-byte words when both planes allow, else 2-byte. */
struct CommitEntArgs {
    const int32_t* tmark;
    const int32_t* pos;
    const int32_t* slot;
    const unsigned char* scratch;
    unsigned char* cache;
    long bytes;           /* an entry */
    long nblk;            /* cache blocks */
    int n, ratio;
};

__global__ __launch_bounds__(kTrThreads) void tree_commit_entries_kernel(CommitEntArgs a) {
    const int m = (int)blockIdx.x;
    if (m >= a.n) return;
    if (a.tmark[m] != 1) return;
    const int s = a.slot[m], p = a.pos[m];
    if (s < 0 || p < 0 || (p + 1) % a.ratio != 0) return;
    const long blk = s / a.ratio;
    if (blk >= a.nblk) return;
    const unsigned char* src = a.scratch + (long)m * a.bytes;
    unsigned char* dst = a.cache + blk * a.bytes;
    if (a.bytes % 16 == 0 && ((uintptr_t)src & 15) == 0 && ((uintptr_t)dst & 15) == 0) {
        for (long i = threadIdx.x; i < a.bytes / 16; i += kTrThreads)
            ((uint4*)dst)[i] = ((const uint4*)src)[i];
    } else {
        for (long i = threadIdx.x; i < a.bytes; i += kTrThreads) dst[i] = src[i];
    }
}

/* ---- tree_commit_rows: one workgroup a sequence, the rows in order, a row's words across the
 * threads (a barrier between rows: the next move may read a row this one wrote). */
struct CommitRowsArgs {
    const int32_t* tmark;
    const int32_t* pos;
    const int32_t* cu;
    const int32_t* slot;
    const int32_t* sidx;
    uint32_t* state;
    long sidx_ld, per;    /* sidx's row pitch; words a state slot */
    int M, S, TR, n, ring, off, slots;
};

__global__ __launch_bounds__(kTrThreads) void tree_commit_rows_kernel(CommitRowsArgs a) {
    const int s = (int)blockIdx.x;
    if (s >= a.S) return;
    const int r0 = a.cu[s], r1 = a.cu[s + 1];
    if (r1 <= r0 + 1 || r0 < 0 || r0 >= a.M) return;
    const int si = a.sidx[(long)s * a.sidx_ld];
    if (si < 0 || si >= a.slots) return;
    const int p0 = a.pos[r0];
    uint32_t* b = a.state + (long)si * a.per;
    for (int m = r0 + 1; m < r1 && m < a.M && m < a.TR; ++m) {
        if (a.tmark[m] != 1 || a.slot[m] < 0) continue;
        const int t = a.pos[m] - p0, j = m - r0;
        if (t < 0) continue;
        const long src = a.ring > 0 ? (long)(p0 + j) % a.ring : (long)a.off + j;
        const long dst = a.ring > 0 ? (long)(p0 + t) % a.ring : (long)a.off + t;
        if (src == dst || src < 0 || dst < 0 || (src + 1) * a.n > a.per || (dst + 1) * a.n > a.per) continue;
        for (int e = threadIdx.x; e < a.n; e += kTrThreads) b[dst * a.n + e] = b[src * a.n + e];
        __syncthreads();
    }
}

/* ---- dspark_tree_assemble: one thread a (sequence, node). */
struct AsmArgs {
    const int32_t* tok;
    const int32_t* alt;
    int32_t* prop;
    int32_t* par;
    long t_ld, a_ld, p_ld, q_ld;
    int S, K, B, P, na;
};

__global__ __launch_bounds__(kTrThreads) void dspark_tree_assemble_kernel(AsmArgs a) {
    const long g = (long)blockIdx.x * kTrThreads + threadIdx.x;
    if (g >= (long)a.S * a.P) return;
    const int s = (int)(g / a.P), n = (int)(g % a.P) + 1;
    const int32_t* tk = a.tok + (long)s * a.t_ld;
    int t, p;
    if (n <= a.K) {
        t = tk[n - 1];
        p = n - 1;
    } else if (n - a.K - 1 < a.B * a.K) {
        const int x = n - a.K - 1, k = x / a.B + 1, b = x % a.B;
        t = b < a.na ? a.alt[(long)s * a.a_ld + (long)b * a.K + k - 1] : -1;
        if (t < 0) t = tk[k - 1];
        p = k == 1 ? 0 : n - a.B;
    } else {
        t = tk[a.K - 1];
        p = n - 1;
    }
    a.prop[(long)s * a.p_ld + n - 1] = t;
    a.par[(long)s * a.q_ld + n - 1] = p;
}

}  // namespace

/* spec_parent, cu_seqlens, positions -> tree [M, width], spos, tmark. */
extern "C" int r4d_rad_tree_rows(const RadArgs* a, RadStream stream) {
    const RadTensor *sp = r4d_opt(a, 0), *cu = r4d_opt(a, 1), *po = r4d_opt(a, 2);
    const RadTensor *tt = r4d_opt(a, 3), *spos = r4d_opt(a, 4), *tm = r4d_opt(a, 5);
    if (!sp || !cu || !po || !tt || !spos || !tm) return R4D_NO(RAD_E_INVAL);
    const long long W = r4d_geti_or(a, "width", 0), M = r4d_numel(sp);
    if (M <= 0 || W < 3 || W > 64 || tt->rank != 2 || tt->shape[1] != W || tt->shape[0] < M ||
        r4d_numel(po) < M || r4d_numel(spos) < M || r4d_numel(tm) < M || r4d_numel(cu) < 2 || M > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (sp->dtype != RAD_I32 || cu->dtype != RAD_I32 || po->dtype != RAD_I32 || tt->dtype != RAD_I32 ||
        spos->dtype != RAD_I32 || tm->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(sp) || !r4d_contig(cu) || !r4d_contig(po) || !r4d_contig(spos) || !r4d_contig(tm) ||
        tt->stride[1] != 1)
        return R4D_NO(RAD_E_STRIDE);
    TreeRowsArgs A{};
    A.sp = (const int32_t*)sp->data;
    A.cu = (const int32_t*)cu->data;
    A.pos = (const int32_t*)po->data;
    A.tt = (int32_t*)tt->data;
    A.spos = (int32_t*)spos->data;
    A.tmark = (int32_t*)tm->data;
    A.tr = (long)tt->stride[0];
    A.M = (int)M;
    A.S = (int)(r4d_numel(cu) - 1);
    A.W = (int)W;
    return R4D_EMIT(stream, tree_rows_kernel, dim3((unsigned)((M + kTrThreads - 1) / kTrThreads)),
                    dim3(kTrThreads), 0, A);
}

/* tmark [TR], positions [M], slot_mapping [M], scratch [TR, ...] -> cache [blocks, ...] (inout). */
extern "C" int r4d_rad_tree_commit_entries(const RadArgs* a, RadStream stream) {
    const RadTensor *tm = r4d_opt(a, 0), *po = r4d_opt(a, 1), *sl = r4d_opt(a, 2), *sc = r4d_opt(a, 3);
    const RadTensor* ca = r4d_opt(a, 4);
    if (!tm || !po || !sl || !sc || !ca) return R4D_NO(RAD_E_INVAL);
    const long long ratio = r4d_geti_or(a, "ratio", 0), M = r4d_numel(po);
    if (M <= 0 || ratio < 1 || r4d_numel(sl) < M || sc->rank < 2 || ca->rank < 2 || sc->dtype != ca->dtype ||
        !r4d_contig(sc) || !r4d_contig(ca) || !r4d_contig(tm) || !r4d_contig(po) || !r4d_contig(sl))
        return R4D_NO(RAD_E_SHAPE);
    if (tm->dtype != RAD_I32 || po->dtype != RAD_I32 || sl->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long per = r4d_numel(sc) / sc->shape[0];
    if (per != r4d_numel(ca) / ca->shape[0] || per < 1) return R4D_NO(RAD_E_SHAPE);
    const long long bytes = per * (long long)(rad_dtype_bits(sc->dtype) / 8);
    if (bytes < 1) return R4D_NO(RAD_E_DTYPE);
    const long long TR = r4d_numel(tm) < sc->shape[0] ? r4d_numel(tm) : sc->shape[0];
    const long long n = M < TR ? M : TR;
    if (n <= 0) return RAD_OK;
    CommitEntArgs A{};
    A.tmark = (const int32_t*)tm->data;
    A.pos = (const int32_t*)po->data;
    A.slot = (const int32_t*)sl->data;
    A.scratch = (const unsigned char*)sc->data;
    A.cache = (unsigned char*)ca->data;
    A.bytes = (long)bytes;
    A.nblk = (long)ca->shape[0];
    A.n = (int)n;
    A.ratio = (int)ratio;
    return R4D_EMIT(stream, tree_commit_entries_kernel, dim3((unsigned)n), dim3(kTrThreads), 0, A);
}

/* tmark [TR], positions [M], cu_seqlens, slot_mapping [M], sidx -> state (inout, 4-byte words). */
extern "C" int r4d_rad_tree_commit_rows(const RadArgs* a, RadStream stream) {
    const RadTensor *tm = r4d_opt(a, 0), *po = r4d_opt(a, 1), *cu = r4d_opt(a, 2), *sl = r4d_opt(a, 3);
    const RadTensor *sx = r4d_opt(a, 4), *st = r4d_opt(a, 5);
    if (!tm || !po || !cu || !sl || !sx || !st) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), ring = r4d_geti_or(a, "ring", 0);
    const long long off = r4d_geti_or(a, "off", 0), M = r4d_numel(po);
    if (M <= 0 || n < 1 || ring < 0 || off < 0 || r4d_numel(sl) < M || r4d_numel(cu) < 2 || st->rank < 1 ||
        !r4d_contig(st) || !r4d_contig(tm) || !r4d_contig(po) || !r4d_contig(cu) || !r4d_contig(sl))
        return R4D_NO(RAD_E_SHAPE);
    if (rad_dtype_bits(st->dtype) != 32 || tm->dtype != RAD_I32 || po->dtype != RAD_I32 ||
        cu->dtype != RAD_I32 || sl->dtype != RAD_I32 || sx->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    const long long S = r4d_numel(cu) - 1;
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < S || S > INT_MAX || M > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    CommitRowsArgs A{};
    A.tmark = (const int32_t*)tm->data;
    A.pos = (const int32_t*)po->data;
    A.cu = (const int32_t*)cu->data;
    A.slot = (const int32_t*)sl->data;
    A.sidx = (const int32_t*)sx->data;
    A.state = (uint32_t*)st->data;
    A.sidx_ld = (long)sxp;
    A.per = (long)(r4d_numel(st) / st->shape[0]);
    A.M = (int)M;
    A.S = (int)S;
    A.TR = (int)(r4d_numel(tm) < INT_MAX ? r4d_numel(tm) : INT_MAX);
    A.n = (int)n;
    A.ring = (int)ring;
    A.off = (int)off;
    A.slots = (int)st->shape[0];
    return R4D_EMIT(stream, tree_commit_rows_kernel, dim3((unsigned)S), dim3(kTrThreads), 0, A);
}

/* tokens [S, K], alt [S, n_alt * K] -> prop [S, pitch], par [S, pitch]. */
extern "C" int r4d_rad_dspark_tree_assemble(const RadArgs* a, RadStream stream) {
    const RadTensor *tok = r4d_opt(a, 0), *alt = r4d_opt(a, 1), *prop = r4d_opt(a, 2), *par = r4d_opt(a, 3);
    if (!tok || !alt || !prop || !par) return R4D_NO(RAD_E_INVAL);
    const long long K = r4d_geti_or(a, "K", 0), B = r4d_geti_or(a, "B", -1);
    const long long P = r4d_geti_or(a, "pitch", 0), na = r4d_geti_or(a, "n_alt", 0);
    if (K < 1 || B < 0 || P < K || na < 0) return R4D_NO(RAD_E_SHAPE);
    long long S = 0;
    const long long t_ld = r4d_rows_pitch(tok, K, &S);
    if (S <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Sa = 0, Sp = 0, Sq = 0;
    const long long a_ld = na > 0 ? r4d_rows_pitch(alt, na * K, &Sa) : 1;
    const long long p_ld = r4d_rows_pitch(prop, P, &Sp), q_ld = r4d_rows_pitch(par, P, &Sq);
    if (t_ld < 0 || a_ld < 0 || p_ld < 0 || q_ld < 0 || (na > 0 && Sa < S) || Sp < S || Sq < S)
        return R4D_NO(RAD_E_SHAPE);
    if (tok->dtype != RAD_I32 || alt->dtype != RAD_I32 || prop->dtype != RAD_I32 || par->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    AsmArgs A{};
    A.tok = (const int32_t*)tok->data;
    A.alt = (const int32_t*)alt->data;
    A.prop = (int32_t*)prop->data;
    A.par = (int32_t*)par->data;
    A.t_ld = (long)t_ld;
    A.a_ld = (long)a_ld;
    A.p_ld = (long)p_ld;
    A.q_ld = (long)q_ld;
    A.S = (int)S;
    A.K = (int)K;
    A.B = (int)B;
    A.P = (int)P;
    A.na = (int)na;
    const long long total = S * P;
    return R4D_EMIT(stream, dspark_tree_assemble_kernel, dim3((unsigned)((total + kTrThreads - 1) / kTrThreads)),
                    dim3(kTrThreads), 0, A);
}
