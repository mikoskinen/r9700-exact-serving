/* ref_tree.h -- a tree verify's row table (tree_rows' `tree` operand) as libref's tree-aware ops read
 * it (attn_sparse_sink_tree, index_score_tree, qsa_work_tree, ngram_ids_tree). libr4d reads it by the
 * same rules, element for element (r4d_tree.h), so the two agree on any operand, a malformed one
 * included.
 *
 * Row m's table row: [0] its node's depth d (a tree row needs 0 <= d <= width - 3), [1] the absolute
 * row r0 of its root, [2 + x] the absolute row of its ancestor at depth x (x = 0 .. d). A row whose
 * depth or root is out of range is an ordinary row; an ancestor out of range is no row (the key it
 * would name is no key). The anchor A is positions[r0]; the ancestor at position q (A <= q <= A + d)
 * is the one at depth q - A, and its K / V / entry sits at slot position A + (row - r0). */
#pragma once
#include "ref_common.h"

namespace ref {

struct RefTree {
    const RadTensor* tt = nullptr;    /* [rows, width] i32, null: no tree */
    const RadTensor* pos = nullptr;   /* the step's positions (absolute rows) */
    int64_t W = 0, rows = 0, npos = 0;

    RefTree() = default;
    RefTree(const RadTensor* t, const RadTensor* p) : tt(t && t->data ? t : nullptr), pos(p) {
        if (tt) {
            W = tt->rank >= 2 ? tt->shape[tt->rank - 1] : 0;
            rows = W > 0 ? numel(tt) / W : 0;
        }
        npos = numel(pos);
    }
    int64_t cell(int64_t m, int64_t c) const {
        return ld_int(tt->dtype, tt->data, rowoff(tt, m, W) + c * laststride(tt));
    }
    /* Whether row m is a tree row; its depth and root. */
    bool row(int64_t m, int64_t* d, int64_t* r0) const {
        if (!tt || W < 3 || m < 0 || m >= rows) return false;
        const int64_t dd = cell(m, 0);
        if (dd < 0 || dd > W - 3) return false;
        const int64_t rr = cell(m, 1);
        if (rr < 0 || rr >= npos) return false;
        *d = dd;
        *r0 = rr;
        return true;
    }
    /* Row m's ancestor at depth x (row m a tree row of depth d), or -1. */
    int64_t anc(int64_t m, int64_t d, int64_t x) const {
        if (x < 0 || x > d) return -1;
        const int64_t r = cell(m, 2 + x);
        return r >= 0 && r < npos ? r : -1;
    }
    int64_t at(int64_t r) const { return ld_int(pos->dtype, pos->data, offlin(pos, r)); }
};

}  // namespace ref
