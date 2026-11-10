/* ref_tree.cpp -- a tree verify's bookkeeping for DeepSeek-V4.1's DSpark (lever T1): the rows' tree
 * table, the context pass's commits of what the accepted path left in the plugin's own state, and
 * the drafter's tree proposal. docs/OPS.md "Tree verify (DSpark)"; libr4d's r4d_tree.hip is the
 * device form, element for element.
 *
 * A tree verify (RadBatch::spec_parent, rad_drafter_tree) runs a sequence's proposal as a tree in its
 * rows: node j at row j, rotated at the anchor + its depth, its K and V stored at the anchor + j. The
 * plugin's tree-aware ops read a row's ancestors through `tree_rows`' table; after the acceptance
 * the context pass (ADV_CTXTREE: the root at the anchor, accepted path node t at anchor + t + 1, a
 * rejected row stored nowhere) moves the accepted path's per-node state into place. */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

int64_t ld(const RadTensor* t, int64_t lin) { return ld_int(t->dtype, t->data, offlin(t, lin)); }

/* Element offset of [r][c] of a tensor read as [rows, cols] (cols its last extent). */
int64_t at2(const RadTensor* t, int64_t r, int64_t c, int64_t cols) {
    return rowoff(t, r, cols) + c * laststride(t);
}

}  // namespace

/* `spec_parent`, `cu_seqlens`, `positions` -> `tree` [M, width], `spos` [M], `tmark` [M]
 *
 * Row m of sequence s (cu): spec_parent -1 is a root (depth 0); p >= 0 names the parent row cu[s] + p,
 * which must precede m in s and be a tree row itself; anything else (-2, a parent outside the
 * sequence or not before the row, a depth past width - 3) is an ordinary row. Columns: 0 the depth
 * (-1 ordinary), 1 the root's row (-1), 2 + d the ancestor at depth d (d = 0 .. depth, the row itself
 * last), the rest -1. spos: positions[root] + (m - root) for a tree row, positions[m] otherwise.
 * tmark: 1 for a tree row of depth >= 1. */
int ref_tree_rows(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *sp = t_in(a, 0), *cu = t_in(a, 1), *po = t_in(a, 2);
    const RadTensor *tt = t_in(a, 3), *spos = t_in(a, 4), *tm = t_in(a, 5);
    if (!sp || !cu || !po || !tt || !spos || !tm) return RAD_E_INVAL;
    const int64_t W = p_int(a, "width", 0);
    const int64_t M = band_rows(numel(sp), p_int(a, "M", 0));
    if (M <= 0 || W < 3 || W > 64) return RAD_E_SHAPE;
    if (numel(po) < M || numel(tt) < M * W || numel(spos) < M || numel(tm) < M || numel(cu) < 2)
        return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    const int64_t dmax = W - 3;
    std::vector<int64_t> chain((size_t)(dmax + 1));
    for (int64_t m = 0; m < M; ++m) {
        /* the walk to the root, or -1 */
        int64_t s = -1;
        for (int64_t i = 0; i < S; ++i)
            if (m >= cu_at(cu, i) && m < cu_at(cu, i + 1)) { s = i; break; }
        int64_t d = -1, root = -1;
        if (s >= 0) {
            const int64_t b0 = cu_at(cu, s);
            int64_t r = m, n = 0;
            bool ok = true;
            for (;;) {
                const int64_t p = ld(sp, r);
                if (p == -1) { root = r; break; }
                if (p < 0 || n >= dmax) { ok = false; break; }
                const int64_t pr = b0 + p;
                if (pr < b0 || pr >= r) { ok = false; break; }
                r = pr;
                ++n;
            }
            if (ok) d = n;
        }
        const int64_t tb = rowoff(tt, m, W), ts = laststride(tt);
        for (int64_t c = 0; c < W; ++c) st_int(tt->dtype, tt->data, tb + c * ts, -1);
        if (d >= 0) {
            /* the ancestors, deepest first along the walk */
            int64_t r = m;
            const int64_t b0 = cu_at(cu, s);
            for (int64_t k = d; k >= 0; --k) {
                chain[(size_t)k] = r;
                if (k > 0) r = b0 + ld(sp, r);
            }
            st_int(tt->dtype, tt->data, tb, d);
            st_int(tt->dtype, tt->data, tb + ts, root);
            for (int64_t k = 0; k <= d; ++k) st_int(tt->dtype, tt->data, tb + (2 + k) * ts, chain[(size_t)k]);
        }
        const int64_t pm = ld(po, m);
        st_int(spos->dtype, spos->data, offlin(spos, m), d >= 0 ? ld(po, root) + (m - root) : pm);
        st_int(tm->dtype, tm->data, offlin(tm, m), d >= 1 ? 1 : 0);
    }
    return RAD_OK;
}

/* `tmark` [TR], `positions` [M], `slot_mapping` [M], `scratch` [TR, ...] -> `cache` [blocks, ...]
 *
 * The context pass after a tree verify: a row m < min(M, TR) that was a tree row of depth >= 1
 * (tmark), is on the accepted path (slot >= 0) and completed a block of `ratio` positions
 * ((positions[m] + 1) % ratio == 0) copies its entry, scratch block m, to the cache block that slot
 * names (slot / ratio). Bytes: scratch and cache share the entry's format. */
int ref_tree_commit_entries(const RadArgs* a, RadStream) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *tm = t_in(a, 0), *po = t_in(a, 1), *sl = t_in(a, 2), *sc = t_in(a, 3);
    const RadTensor* ca = t_in(a, 4);
    if (!tm || !po || !sl || !sc || !ca) return RAD_E_INVAL;
    const int64_t ratio = p_int(a, "ratio", 0);
    const int64_t M = band_rows(numel(po), p_int(a, "M", 0));
    if (M <= 0 || ratio < 1 || numel(sl) < M || sc->rank < 2 || ca->rank < 2 || sc->dtype != ca->dtype ||
        !rad_tensor_is_contiguous(sc) || !rad_tensor_is_contiguous(ca))
        return RAD_E_SHAPE;
    const int64_t TR = std::min(numel(tm), sc->shape[0]);
    const int64_t per = numel(sc) / sc->shape[0];
    if (per != numel(ca) / ca->shape[0] || per < 1) return RAD_E_SHAPE;
    const int64_t bytes = per * ((int64_t)rad_dtype_bits(sc->dtype) / 8);
    if (bytes < 1) return RAD_E_DTYPE;
    const int64_t n = std::min(M, TR);
    for (int64_t m = 0; m < n; ++m) {
        if (ld(tm, m) != 1) continue;
        const int64_t s = ld(sl, m), p = ld(po, m);
        if (s < 0 || p < 0 || (p + 1) % ratio != 0) continue;
        const int64_t blk = s / ratio;
        if (blk >= ca->shape[0]) continue;
        std::memcpy((char*)ca->data + blk * bytes, (const char*)sc->data + m * bytes, (size_t)bytes);
    }
    return RAD_OK;
}

/* `tmark` [TR], `positions` [M], `cu_seqlens`, `slot_mapping` [M], `sidx` -> `state`:inout
 *
 * The context pass after a tree verify: per sequence s (root row r0 = cu[s]), the rows m after r0 in
 * order that were tree rows of depth >= 1 (tmark) and are on the accepted path (slot >= 0): with
 * t = positions[m] - positions[r0] (its depth) and j = m - r0 (its row), state row src moves to row
 * dst of the sequence's slot sidx[s] (column 0): ring mode (`ring` > 0) src = (positions[r0] + j) %
 * ring, dst = (positions[r0] + t) % ring; linear mode src = off + j, dst = off + t. A row is `n`
 * 4-byte words (f32 or i32), row r at word r * n of the slot. Rows in order, so every move reads
 * what no earlier move wrote (src >= dst). A sequence with a negative slot moves nothing. */
int ref_tree_commit_rows(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *tm = t_in(a, 0), *po = t_in(a, 1), *cu = t_in(a, 2), *sl = t_in(a, 3);
    const RadTensor *sx = t_in(a, 4), *st = t_in(a, 5);
    if (!tm || !po || !cu || !sl || !sx || !st) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), ring = p_int(a, "ring", 0), off = p_int(a, "off", 0);
    const int64_t M = band_rows(numel(po), p_int(a, "M", 0));
    if (M <= 0 || n < 1 || ring < 0 || off < 0 || numel(sl) < M || numel(cu) < 2 || st->rank < 1 ||
        !rad_tensor_is_contiguous(st))
        return RAD_E_SHAPE;
    if (rad_dtype_bits(st->dtype) != 32) return RAD_E_DTYPE;
    const int64_t S = numel(cu) - 1, slots = st->shape[0], per = numel(st) / slots;
    if ((sx->rank >= 2 ? sx->shape[0] : numel(sx)) < S) return RAD_E_SHAPE;
    const int64_t TR = numel(tm);
    uint32_t* w = (uint32_t*)st->data;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = cu_at(cu, s), r1 = cu_at(cu, s + 1);
        if (r1 <= r0 + 1 || r0 < 0 || r0 >= M) continue;
        const int64_t si = ld_int(sx->dtype, sx->data, sx->rank >= 2 ? s * sx->stride[0] : offlin(sx, s));
        if (si < 0 || si >= slots) continue;
        const int64_t p0 = ld(po, r0);
        for (int64_t m = r0 + 1; m < r1 && m < M && m < TR; ++m) {
            if (ld(tm, m) != 1 || ld(sl, m) < 0) continue;
            const int64_t t = ld(po, m) - p0, j = m - r0;
            if (t < 0) continue;
            const int64_t src = ring > 0 ? (p0 + j) % ring : off + j;
            const int64_t dst = ring > 0 ? (p0 + t) % ring : off + t;
            if (src == dst || src < 0 || dst < 0 || (src + 1) * n > per || (dst + 1) * n > per) continue;
            uint32_t* b = w + si * per;
            for (int64_t e = 0; e < n; ++e) b[dst * n + e] = b[src * n + e];
        }
    }
    return RAD_OK;
}

/* `tokens` [S, K], `alt` [S, n_alt * K] -> `prop` [S, pitch], `par` [S, pitch]
 *
 * DSpark's tree proposal: nodes 1..K the walk (node j's parent j - 1), then the B alternative branches
 * at position 1 depth-major -- node K + (k - 1) B + b + 1 is branch b's node at depth k (1..K), its
 * parent 0 at depth 1 and the node B before it after; an alternative token below zero (no candidate,
 * or b >= n_alt) is the walk's token at that depth. Nodes past K + B K (B < 3): the walk's last token,
 * parent the node before. par[n - 1] is node n's parent (0 the root). */
int ref_dspark_tree_assemble(const RadArgs* a, RadStream) {
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor *tok = t_in(a, 0), *alt = t_in(a, 1), *prop = t_in(a, 2), *par = t_in(a, 3);
    if (!tok || !alt || !prop || !par) return RAD_E_INVAL;
    const int64_t K = p_int(a, "K", 0), B = p_int(a, "B", -1), P = p_int(a, "pitch", 0);
    const int64_t na = p_int(a, "n_alt", 0);
    const int64_t S = K > 0 ? band_rows(numel(tok) / K, p_int(a, "M", 0)) : 0;
    if (S <= 0 || K < 1 || B < 0 || P < K || na < 0 || numel(alt) < S * na * K || numel(prop) < S * P ||
        numel(par) < S * P)
        return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s) {
        auto tk = [&](int64_t k) { return ld_int(tok->dtype, tok->data, at2(tok, s, k, K)); };
        for (int64_t n = 1; n <= P; ++n) {
            int64_t t, p;
            if (n <= K) {
                t = tk(n - 1);
                p = n - 1;
            } else if (n - K - 1 < B * K) {
                const int64_t x = n - K - 1, k = x / B + 1, b = x % B;
                t = b < na ? ld_int(alt->dtype, alt->data, at2(alt, s, b * K + k - 1, na * K)) : -1;
                if (t < 0) t = tk(k - 1);
                p = k == 1 ? 0 : n - B;
            } else {
                t = tk(K - 1);
                p = n - 1;
            }
            st_int(prop->dtype, prop->data, at2(prop, s, n - 1, P), t);
            st_int(par->dtype, par->data, at2(par, s, n - 1, P), p);
        }
    }
    return RAD_OK;
}
