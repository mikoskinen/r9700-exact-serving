/* ref_dspark.cpp -- DSpark beside its Markov walk: the confidence head (dspark_conf) and the walk's
 * next-best first proposals each continued by the walk (dspark_alt). docs/OPS.md "DSpark";
 * libr4d's r4d_dspark_extra.hip is the device form. */
#include "ref_common.h"
#include "ref_ops.h"

#include <cmath>
#include <vector>

using namespace ref;

/* `x`, `anchor`, `tokens`, `pred`w, `w` -> `conf`
 *
 *   conf[s, j] = bias + sum_k w[k] x[s*steps+j][k] + sum_r w[n_embd + r] pred[pid][r]
 *   pid = anchor[s * anchor_stride] at j = 0, tokens[s, j-1] after; an out-of-range pid adds nothing.
 *
 * f32, ascending k then r. Rows from `conf` [M, steps]. */
int ref_dspark_conf(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *anc = &a->t[1], *tok = &a->t[2], *pred = &a->t[3];
    const RadTensor *w = &a->t[4], *out = &a->t[5];
    if (!x->data || !anc->data || !tok->data || !pred->data || !w->data || !out->data) return RAD_E_INVAL;
    const int64_t steps = p_int(a, "steps", 0), D = p_int(a, "n_embd", 0), R = p_int(a, "rank", 0);
    const int64_t nv = p_int(a, "n_vocab", 0), astr = p_int(a, "anchor_stride", 1);
    const float bias = p_f32(a, "bias", 0.0f);
    const int64_t M = steps > 0 ? band_rows(numel(out) / steps, p_int(a, "M", 0)) : 0;
    if (M <= 0 || steps <= 0 || D <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(x) < M * steps * D || numel(w) < D + R || numel(pred) < nv * R) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || (steps > 1 && numel(tok) < M * (steps - 1))) return RAD_E_SHAPE;
    const int64_t xs = laststride(x), ps = laststride(pred), os = laststride(out);
    const int64_t tcols = tok->rank >= 2 ? tok->shape[tok->rank - 1] : steps;
    REF_PAR_FOR(M * steps * (D + R))
    for (int64_t m = 0; m < M; ++m)
        for (int64_t j = 0; j < steps; ++j) {
            const int64_t pid = j == 0 ? ld_int(anc->dtype, anc->data, offlin(anc, m * astr))
                                       : ld_int(tok->dtype, tok->data, rowoff(tok, m, tcols) + (j - 1) * laststride(tok));
            const int64_t xb = rowoff(x, m * steps + j, D);
            float acc = 0.0f;
            for (int64_t k = 0; k < D; ++k) acc += ldt(w, offlin(w, k)) * ldt(x, xb + k * xs);
            if (pid >= 0 && pid < nv) {
                const int64_t pb = rowoff(pred, pid, R);
                for (int64_t r = 0; r < R; ++r) acc += ldt(w, offlin(w, D + r)) * ldt(pred, pb + r * ps);
            }
            stt(out, rowoff(out, m, steps) + j * os, acc + bias);
        }
    return RAD_OK;
}

/* `cand`, `unary`, `anchor`, `tokens`, `pred`w, `succ`w -> `alt`
 *
 * Row 0's candidates scored as dflash_select scores them; the `n_alt` best whose id is valid and not
 * tokens[s, 0] (ties to the lower index) start the branches, each walked greedily over rows
 * 1..steps-1 (dflash_select's rule: argmax, ties to the lower index, the pick the next predecessor).
 * alt[s, a*steps + l]; a branch with no candidate is -1 throughout. */
int ref_dspark_alt(const RadArgs* a, RadStream) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[2], *tok = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6];
    if (!cand->data || !un->data || !anc->data || !tok->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;
    const int64_t steps = p_int(a, "steps", 0), K = p_int(a, "top_k", 0), R = p_int(a, "rank", 0);
    const int64_t nv = p_int(a, "n_vocab", 0), astr = p_int(a, "anchor_stride", 1);
    const int64_t na = p_int(a, "n_alt", 0);
    const int64_t M = steps > 0 && na > 0 ? band_rows(numel(out) / (na * steps), p_int(a, "M", 0)) : 0;
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0 || na <= 0) return RAD_E_SHAPE;
    if (numel(cand) < M * steps * K || numel(un) < M * steps * K || numel(tok) < M) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);
    const int64_t ps = laststride(pred), sxs = laststride(succ);
    const int64_t tcols = tok->rank >= 2 ? tok->shape[tok->rank - 1] : 1;
    REF_PAR_FOR(M * steps * K * R * (na + 1))
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> ax((size_t)R), sc((size_t)K);
        auto score = [&](int64_t row, int64_t pid) {
            const bool ok = pid >= 0 && pid < nv;
            for (int64_t i = 0; i < R; ++i) ax[(size_t)i] = ok ? ldt(pred, rowoff(pred, pid, R) + i * ps) : 0.0f;
            const int64_t cb = rowoff(cand, row, K), ub = rowoff(un, row, K);
            for (int64_t c = 0; c < K; ++c) {
                const int64_t id = ld_int(cand->dtype, cand->data, cb + c * cs);
                float d = 0.0f;
                if (id >= 0 && id < nv) {
                    const int64_t sb = rowoff(succ, id, R);
                    for (int64_t i = 0; i < R; ++i) d += ax[(size_t)i] * ldt(succ, sb + i * sxs);
                }
                sc[(size_t)c] = ldt(un, ub + c * us) + d;
            }
        };
        const int64_t row0 = m * steps;
        score(row0, ld_int(anc->dtype, anc->data, offlin(anc, m * astr)));
        const int64_t d1 = ld_int(tok->dtype, tok->data, rowoff(tok, m, tcols));
        const int64_t cb0 = rowoff(cand, row0, K);
        std::vector<bool> used((size_t)K);
        for (int64_t c = 0; c < K; ++c) {
            const int64_t id = ld_int(cand->dtype, cand->data, cb0 + c * cs);
            used[(size_t)c] = id < 0 || id >= nv || id == d1;
        }
        std::vector<int64_t> first((size_t)na, -1);
        for (int64_t b = 0; b < na; ++b) {
            int64_t best = -1;
            for (int64_t c = 0; c < K; ++c)
                if (!used[(size_t)c] && (best < 0 || sc[(size_t)c] > sc[(size_t)best])) best = c;
            if (best >= 0) {
                used[(size_t)best] = true;
                first[(size_t)b] = ld_int(cand->dtype, cand->data, cb0 + best * cs);
            }
        }
        const int64_t ob = rowoff(out, m, na * steps);
        for (int64_t b = 0; b < na; ++b) {
            int64_t pid = first[(size_t)b];
            for (int64_t l = 0; l < steps; ++l) {
                if (first[(size_t)b] < 0) { st_int(out->dtype, out->data, ob + (b * steps + l) * os, -1); continue; }
                if (l > 0) {
                    score(row0 + l, pid);
                    int64_t best = 0;
                    float bv = -INFINITY;
                    for (int64_t c = 0; c < K; ++c)
                        if (sc[(size_t)c] > bv) { bv = sc[(size_t)c]; best = c; }
                    pid = ld_int(cand->dtype, cand->data, rowoff(cand, row0 + l, K) + best * cs);
                }
                st_int(out->dtype, out->data, ob + (b * steps + l) * os, pid);
            }
        }
    }
    return RAD_OK;
}
