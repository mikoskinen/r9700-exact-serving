/* r4d_ds41_wsamp_math.h -- lever SD (sampled drafting) on lever W1's full-vocabulary DSpark walk: the arithmetic one
 * walk position turns into a sampled draft and the coupled state the verify reads (abi/rad_spechub.h). Header-only and
 * stateless, host and device, so the kernel (r4d_ds41_wsamp.hip), the emulator test (tests/r4d_emu/test_ds41wsamp.cpp)
 * and the host exactness test (tests/sd_exact_test.cpp) share one definition.
 *
 * THE LAW Q. At draft position l (round l + 1) the walk scores every vocabulary id v with
 *     score[v] = logits[l][v] + sum_r pred[prev][r] * succ[v][r]           (lever W1's arithmetic, bit for bit)
 * and lever W1 takes the argmax. Sampled drafting takes the EIGHT best keys over the whole vocabulary (the key is
 * W1's: ordered score bits << 32 | ~id, so "best" is the greatest score, ties to the lower id; NaN never) and
 *     Q = FP64 softmax of (score / q_temp) over those eight                 (rad_mtp_couple_build_q's arithmetic)
 *     x ~ Q at the draw's uniform (seed, base, RAD_MTP_COUPLE_KEY_DRAFT + round)   (rad_mtp_couple_pick)
 * q_temp is the request's temperature times the switch's scale (draft_sample 1: 1.0), so Q carries the temperature
 * the target's law carries. Q is stored exactly in the state (ids, q, x), which is all the maximal coupling needs to be
 * exact: the verify accepts x with probability min(1, P(x) / Q(x)) and otherwise draws from norm(max(0, P - Q))
 * (rad_mtp_couple_decide, the spechub plugin's mtp_couple_verify), so the emitted token is distributed as P whatever
 * Q is. Q depends on the committed text and on x_0 .. x_{l-1} only, never on the verify's randomness.
 *
 *   mode 0 (a greedy request, temperature 0 / top_k 1):  status 0, x = the best key's id -- exactly lever W1's token
 *       (no valid key at all: 0, W1's rule), so temperature 0 under draft_sample 1 drafts what draft_sample 0 drafts;
 *   mode 1, the card's own draw and a Q that builds:     status 1, x ~ Q;
 *   mode 1 otherwise (a draw naming another card, q_temp not a positive finite number, a +inf score, no live key):
 *       status -1, x = the greedy token, and the verify applies the ordinary rule to it (the engine binds it as
 *       malformed and the sampler does not couple it). */
#ifndef R4D_DS41_WSAMP_MATH_H
#define R4D_DS41_WSAMP_MATH_H

#include <math.h>
#include <stdint.h>

#include "rad_spechub.h"

#if defined(__HIPCC__)
#define R4D_WS_HD __host__ __device__
#else
#define R4D_WS_HD
#endif

/* Keys a position keeps: the coupled state holds eight. */
#define R4D_WSAMP_K 8

/* A walk key's id (key 0, "none": -1) and its score (the inverse of r4d_ds41_walk.hip's walk_ord: the high word is
 * the score's bits with the sign flipped for a positive number and all bits flipped for a negative one). */
static inline R4D_WS_HD int32_t r4d_wsamp_key_id(uint64_t k) {
    return k ? (int32_t)(~(uint32_t)(k & 0xffffffffull)) : -1;
}
static inline R4D_WS_HD float r4d_wsamp_key_score(uint64_t k) {
    const uint32_t o = (uint32_t)(k >> 32);
    const uint32_t b = (o & 0x80000000u) ? (o & 0x7fffffffu) : ~o;
    union { uint32_t u; float f; } c;
    c.u = b;
    return c.f;
}
/* The key of `score` at id `gid` (lever W1's walk_key): 0 for a NaN score, -0 as +0. */
static inline R4D_WS_HD uint64_t r4d_wsamp_key(float score, int32_t gid) {
    union { float f; uint32_t u; } c;
    c.f = score;
    uint32_t b = c.u;
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0ull;
    if ((b << 1) == 0u) b = 0u;
    const uint32_t o = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t)o << 32) | (uint64_t)(~(uint32_t)gid);
}

/* Insert `k` into the descending list t[0, 8) (0 = empty slot); a key not above t[7] is dropped. Keys are unique (the
 * id is in them), so the list is the eight greatest of everything inserted, in order, whatever the order. */
static inline R4D_WS_HD void r4d_wsamp_insert(uint64_t* t, uint64_t k) {
    if (k <= t[R4D_WSAMP_K - 1]) return;
    int i = R4D_WSAMP_K - 1;
    while (i > 0 && t[i - 1] < k) { t[i] = t[i - 1]; --i; }
    t[i] = k;
}

/* One position: `top` the eight best keys, descending (0 = none). Fills `*s` (draw, pos 0, round, status, x, count,
 * ids, q) and returns the proposed token. `q_temp` as above; `card_ok` whether the draw names the card the row runs
 * on. */
static inline R4D_WS_HD int32_t r4d_wsamp_step(const uint64_t* top, float q_temp, const RadMtpCoupleDraw& d,
                                               bool card_ok, int round, int64_t vocab, RadMtpCoupleState* s) {
    const int32_t greedy = top[0] ? r4d_wsamp_key_id(top[0]) : 0;
    s->draw = d;
    s->pos = 0;
    s->round = round;
    s->count = 0;
    for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    if (d.mode != 1) {
        s->status = 0;
        s->x = greedy;
        return greedy;
    }
    s->status = -1;
    if (card_ok && q_temp > 0.0f && q_temp <= 3.402823466e38f) {
        int32_t kid[8];
        float   kv[8];
        for (int i = 0; i < 8; ++i) {
            kid[i] = top[i] ? r4d_wsamp_key_id(top[i]) : -1;
            kv[i]  = top[i] ? r4d_wsamp_key_score(top[i]) / q_temp : -INFINITY;
        }
        if (rad_mtp_couple_build_q(kid, kv, vocab, s)) {
            const double u = rad_spechub_uniform(d.seed, d.base, RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)round);
            s->x = s->ids[rad_mtp_couple_pick(*s, u)];
            s->status = 1;
            return s->x;
        }
        s->count = 0;
        for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    }
    s->x = greedy;
    return greedy;
}

#undef R4D_WS_HD
#endif
