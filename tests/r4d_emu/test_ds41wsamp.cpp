/* tests/r4d_emu: lever SD (sampled drafting, fn4 draft_sample) -- dspark_walk_sample_step (libr4d/r4d_ds41_wsamp.hip),
 * the W1 walk over the whole vocabulary SAMPLED, on the host SIMT emulator.
 *
 * deepseek41 issues it `steps` + 1 times a query pass with an all_gather of the per-rank keys (eight u64 a sequence a
 * rank) between steps. This test runs that sequence over W = 1 .. 4 ranks (the shards as deepseek41 splits the
 * vocabulary; the gather as all_gather moves the bytes: rank w's eight keys of sequence s at [s][8 w .. 8 w + 8)) and
 * holds every rank's tokens AND every round's coupled state to a plain host walk byte for byte:
 *
 *   prev = anchor; for l: score[v] = logits[l][v] + fma-chain_r(pred[prev][r], succ[v][r]) for every v (zeros for a
 *     predecessor outside the vocabulary); top = the eight greatest keys (r4d_wsamp_key: score bits, ties to the lower
 *     id, NaN never); x = r4d_wsamp_step(top, q_temp, draw, draw.card == card, l + 1, V, &state[l]); prev = x
 *
 * and, separately, the greedy rows to lever W1's own walk (the plain argmax walk of test_ds41walk.cpp): a mode-0 draw
 * drafts exactly what dspark_walk_step drafts, which is why temperature 0 under draft_sample 1 is draft for draft
 * today's. Cases: random tables and logits (bf16-valued logits: equal scores are common) at ranks 32 / 64 / 256, one to
 * eleven sequences (two passes of eight), shards of one to several blocks, sampled and greedy rows mixed, a draw naming
 * another card (status -1, the greedy token), a q_temp that is not a positive finite number (status -1), NaN / -inf /
 * +inf logits, anchors outside the vocabulary. Each case also checks that the tokens' padding columns, the state rows
 * past the batch and the arrival counters (zero after every step) are left as they should be.
 *
 * Then the law: one position, 6000 seeds, x's frequencies against the Q the state stores (chi-square, the p-value
 * printed), and Q itself against an independent FP64 softmax of the eight best scores at q_temp. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "r4d_ds41_wsamp_math.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int64_t per_rank(int64_t total, int world) {   /* arch/common/rad_arch.h vocab_per_rank */
    if (total % world == 0) return total / world;
    const int64_t per = (total + world - 1) / world;
    return (per + 127) / 128 * 128;
}
int64_t sh_begin(int64_t total, int world, int rank) {
    const int64_t b = per_rank(total, world) * rank;
    return b < total ? b : total;
}
int64_t sh_extent(int64_t total, int world, int rank) {
    const int64_t b = sh_begin(total, world, rank), e = std::min(total, b + per_rank(total, world));
    return e > b ? e - b : 0;
}

constexpr int kSW = RAD_MTP_COUPLE_STATE_WORDS;
constexpr int kDW = RAD_MTP_COUPLE_DRAW_WORDS;

struct Walk {
    int V = 0, R = 0, S = 0, steps = 5, astride = 5;
    const uint16_t* w1 = nullptr;          /* pred [V, R] */
    const uint16_t* w2 = nullptr;          /* succ [V, R] */
    std::vector<float> logits;             /* [S * steps, V] */
    std::vector<int32_t> anchor;           /* [(S - 1) * astride + 1] */
    std::vector<RadMtpCoupleDraw> draw;    /* [S] */
    std::vector<int32_t> card;             /* [S] the batch's seq_ids */
    std::vector<float> qt;                 /* [S] */
};

float score_of(const Walk& w, const float* u, int s, int l, int v) {
    const uint16_t* row = w.w2 + (size_t)v * w.R;
    float acc = 0.0f;
    for (int r = 0; r < w.R; ++r) acc = std::fma(u[r], b2f(row[r]), acc);
    return w.logits[(size_t)(s * w.steps + l) * w.V + v] + acc;
}

/* The plain sampled walk: tokens [S * steps], states [steps][S]. */
void ref_walk(const Walk& w, std::vector<int32_t>& tok, std::vector<RadMtpCoupleState>& st) {
    tok.assign((size_t)w.S * w.steps, 0);
    st.assign((size_t)w.S * w.steps, RadMtpCoupleState{});
    std::vector<float> u((size_t)w.R);
    for (int s = 0; s < w.S; ++s) {
        int32_t prev = w.anchor[(size_t)s * w.astride];
        for (int l = 0; l < w.steps; ++l) {
            const bool ok = prev >= 0 && prev < w.V;
            for (int r = 0; r < w.R; ++r) u[(size_t)r] = ok ? b2f(w.w1[(size_t)prev * w.R + r]) : 0.0f;
            uint64_t top[8] = {};
            for (int v = 0; v < w.V; ++v) r4d_wsamp_insert(top, r4d_wsamp_key(score_of(w, u.data(), s, l, v), v));
            RadMtpCoupleState& o = st[(size_t)l * w.S + s];
            const int32_t x = r4d_wsamp_step(top, w.qt[(size_t)s], w.draw[(size_t)s], w.draw[(size_t)s].card == w.card[(size_t)s],
                                             l + 1, w.V, &o);
            tok[(size_t)s * w.steps + l] = x;
            prev = x;
        }
    }
}

/* Lever W1's walk (the argmax, ties to the lower id, NaN skipped, nothing -> 0), test_ds41walk.cpp's ref_walk. */
std::vector<int32_t> w1_walk(const Walk& w) {
    std::vector<int32_t> out((size_t)w.S * w.steps);
    std::vector<float> u((size_t)w.R);
    for (int s = 0; s < w.S; ++s) {
        int32_t prev = w.anchor[(size_t)s * w.astride];
        for (int l = 0; l < w.steps; ++l) {
            const bool ok = prev >= 0 && prev < w.V;
            for (int r = 0; r < w.R; ++r) u[(size_t)r] = ok ? b2f(w.w1[(size_t)prev * w.R + r]) : 0.0f;
            bool have = false;
            float best = 0.0f;
            int32_t t = 0;
            for (int v = 0; v < w.V; ++v) {
                float sc = score_of(w, u.data(), s, l, v);
                if (std::isnan(sc)) continue;
                if (sc == 0.0f) sc = 0.0f;
                if (!have || sc > best) { have = true; best = sc; t = v; }
            }
            out[(size_t)s * w.steps + l] = t;
            prev = t;
        }
    }
    return out;
}

RadTensor mat(void* d, uint32_t dt, int64_t r, int64_t c, int64_t pitch) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c;
    t.stride[0] = pitch; t.stride[1] = 1;
    return t;
}
RadTensor vec(void* d, uint32_t dt, int64_t n) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 1;
    t.shape[0] = n; t.stride[0] = 1;
    return t;
}
RadTensor cube(void* d, uint32_t dt, int64_t a, int64_t b, int64_t c) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 3;
    t.shape[0] = a; t.shape[1] = b; t.shape[2] = c;
    t.stride[0] = b * c; t.stride[1] = c; t.stride[2] = 1;
    return t;
}
RadParam pint(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }

/* The sharded sampled walk over W ranks as deepseek41 issues it: every rank's tokens (pitch steps + 2, padding -77)
 * and states ([steps][S + 1 rows][38]: the row past the batch must stay untouched). False on a refused launch or a
 * counter left nonzero. */
bool sharded_walk(const Walk& w, int W, std::vector<std::vector<int32_t>>& tok,
                  std::vector<std::vector<RadMtpCoupleState>>& st, std::string* why) {
    const int S = w.S, steps = w.steps, tp = steps + 2, rows = S + 1;
    std::vector<std::vector<float>> lg((size_t)W);
    std::vector<int64_t> beg((size_t)W), ext((size_t)W), lp((size_t)W);
    for (int r = 0; r < W; ++r) {
        beg[(size_t)r] = sh_begin(w.V, W, r);
        ext[(size_t)r] = sh_extent(w.V, W, r);
        lp[(size_t)r] = std::max<int64_t>(1, ext[(size_t)r] + 3);
        lg[(size_t)r].assign((size_t)(S * steps) * lp[(size_t)r], 1e30f);
        for (int i = 0; i < S * steps; ++i)
            for (int64_t j = 0; j < ext[(size_t)r]; ++j)
                lg[(size_t)r][(size_t)(i * lp[(size_t)r] + j)] = w.logits[(size_t)i * w.V + beg[(size_t)r] + j];
    }
    tok.assign((size_t)W, std::vector<int32_t>((size_t)S * tp, -77));
    RadMtpCoupleState poison{};
    std::memset(&poison, 0xa5, sizeof poison);
    st.assign((size_t)W, std::vector<RadMtpCoupleState>((size_t)steps * rows, poison));
    std::vector<uint32_t> draws((size_t)S * kDW);
    std::memcpy(draws.data(), w.draw.data(), (size_t)S * sizeof(RadMtpCoupleDraw));
    std::vector<float> qt((size_t)S);
    for (int s = 0; s < S; ++s) qt[(size_t)s] = w.qt[(size_t)s];
    /* keys [S][8] u64 = [S, 16] f32; gath [S][8 W] u64 = [S, 16 W] f32; scratch [S * nblk][8] u64 */
    std::vector<std::vector<uint64_t>> keys((size_t)W, std::vector<uint64_t>((size_t)S * 8));
    std::vector<uint64_t> gath((size_t)S * W * 8, 0);
    std::vector<std::vector<uint64_t>> scr((size_t)W);
    std::vector<uint32_t> cnt((size_t)W * 4 * steps, 0);
    for (int step = 0; step <= steps; ++step) {
        for (int r = 0; r < W; ++r) {
            const int64_t nblk = std::max<int64_t>(1, (ext[(size_t)r] + 127) / 128);
            scr[(size_t)r].assign((size_t)(S * nblk * 8), 0x5a5a5a5a5a5a5a5aull);   /* every slot written before it is read */
            std::fill(keys[(size_t)r].begin(), keys[(size_t)r].end(), 0x7777777777777777ull);
            uint32_t* c = cnt.data() + ((size_t)r * steps + (size_t)std::min(step, steps - 1)) * 4;
            RadTensor t[13] = {
                mat(lg[(size_t)r].data(), RAD_F32, (int64_t)S * steps, ext[(size_t)r], lp[(size_t)r]),
                vec((void*)w.anchor.data(), RAD_I32, (int64_t)w.anchor.size()),
                step > 0 ? mat(gath.data(), RAD_F32, S, 16 * W, 16 * W) : RadTensor{},
                mat((void*)w.w1, RAD_BF16, w.V, w.R, w.R),
                mat((void*)w.w2, RAD_BF16, w.V, w.R, w.R),
                step < steps ? mat(keys[(size_t)r].data(), RAD_F32, S, 16, 16) : RadTensor{},
                mat(tok[(size_t)r].data(), RAD_I32, S, steps, tp),
                step < steps ? mat(scr[(size_t)r].data(), RAD_F32, S * nblk, 16, 16) : RadTensor{},
                step < steps ? mat(c, RAD_F32, 1, 4, 4) : RadTensor{},
                vec((void*)w.card.data(), RAD_I32, S),
                mat(draws.data(), RAD_U32, S, kDW, kDW),
                mat(qt.data(), RAD_F32, S, 1, 1),
                cube(st[(size_t)r].data(), RAD_U32, steps, rows, kSW),
            };
            RadParam p[10] = { pint("M", S), pint("step", step), pint("steps", steps), pint("rank", w.R),
                               pint("n_vocab", w.V), pint("N", ext[(size_t)r]), pint("vocab_off", beg[(size_t)r]),
                               pint("world", W), pint("anchor_stride", w.astride), RadParam{} };
            p[9].key = "dtype"; p[9].kind = RAD_P_STR; p[9].sval = "f32";
            RadArgs a{};
            a.t = t; a.n_t = 13; a.p = p; a.n_p = 10;
            const int rc = r4d_rad_dspark_walk_sample_step(&a, nullptr);
            if (rc != RAD_OK) {
                if (why) *why = "rank " + std::to_string(r) + " step " + std::to_string(step) + " rc " + std::to_string(rc);
                return false;
            }
            if (step < steps && c[0] != 0u) {
                if (why) *why = "rank " + std::to_string(r) + " step " + std::to_string(step) + ": counter left " + std::to_string(c[0]);
                return false;
            }
        }
        if (step < steps)   /* all_gather, row-interleaved: y[s][w] = rank w's eight keys of s */
            for (int s = 0; s < S; ++s)
                for (int r = 0; r < W; ++r)
                    for (int q = 0; q < 8; ++q)
                        gath[(size_t)((s * W + r) * 8 + q)] = keys[(size_t)r][(size_t)(s * 8 + q)];
    }
    return true;
}

int g_cases = 0, g_fails = 0;

bool same_state(const RadMtpCoupleState& a, const RadMtpCoupleState& b) { return !std::memcmp(&a, &b, sizeof a); }

void check(const char* what, const Walk& w, std::initializer_list<int> worlds) {
    std::vector<int32_t> want;
    std::vector<RadMtpCoupleState> wst;
    ref_walk(w, want, wst);
    /* the greedy rows are lever W1's walk */
    const std::vector<int32_t> w1 = w1_walk(w);
    int gbad = 0, ngreedy = 0, nsamp = 0, nmal = 0;
    for (int s = 0; s < w.S; ++s)
        for (int l = 0; l < w.steps; ++l) {
            const RadMtpCoupleState& o = wst[(size_t)l * w.S + s];
            if (o.status == 0) { ++ngreedy; gbad += o.x != want[(size_t)s * w.steps + l]; }
            if (o.status == 1) { ++nsamp; if (!rad_mtp_couple_state_valid(o, w.V)) ++gbad; }
            if (o.status == -1) ++nmal;
        }
    /* the whole walk of a greedy-only copy equals W1's */
    Walk g = w;
    for (auto& d : g.draw) d.mode = 0;
    std::vector<int32_t> gw;
    std::vector<RadMtpCoupleState> gst;
    ref_walk(g, gw, gst);
    const bool w1same = gw == w1;
    for (int W : worlds) {
        ++g_cases;
        std::vector<std::vector<int32_t>> tok;
        std::vector<std::vector<RadMtpCoupleState>> st;
        std::string why;
        int bad = 0, sbad = 0, pad = 0;
        const bool ran = sharded_walk(w, W, tok, st, &why);
        if (ran)
            for (int r = 0; r < W; ++r) {
                for (int s = 0; s < w.S; ++s) {
                    for (int l = 0; l < w.steps; ++l) {
                        bad += tok[(size_t)r][(size_t)s * (w.steps + 2) + l] != want[(size_t)s * w.steps + l];
                        sbad += !same_state(st[(size_t)r][(size_t)l * (w.S + 1) + s], wst[(size_t)l * w.S + s]);
                    }
                    for (int l = w.steps; l < w.steps + 2; ++l) pad += tok[(size_t)r][(size_t)s * (w.steps + 2) + l] != -77;
                }
                RadMtpCoupleState poison{};
                std::memset(&poison, 0xa5, sizeof poison);
                for (int l = 0; l < w.steps; ++l) pad += !same_state(st[(size_t)r][(size_t)l * (w.S + 1) + w.S], poison);
            }
        const bool ok = ran && !bad && !sbad && !pad && !gbad && w1same;
        std::printf("%-54s V %5d R %3d S %2d W %d: %s (rows: %d sampled, %d greedy, %d status -1)", what, w.V, w.R, w.S, W,
                    !ran ? "LAUNCH REFUSED" : bad ? "TOKENS DIFFER" : sbad ? "STATES DIFFER" : pad ? "PADDING TOUCHED"
                    : gbad ? "REFERENCE STATE INVALID" : !w1same ? "GREEDY ROWS DIFFER FROM W1"
                    : "byte-identical to the plain walk, tokens and states; greedy rows = W1",
                    nsamp, ngreedy, nmal);
        if (!ran) std::printf(" (%s)", why.c_str());
        if (bad || sbad) std::printf(" (%d token(s), %d state(s))", bad, sbad);
        std::printf("\n");
        if (!ok) ++g_fails;
    }
}

uint16_t f2bh(float f) { return f2b(f); }

void fill_random(Walk& w, std::vector<uint16_t>& w1, std::vector<uint16_t>& w2, std::mt19937& rng, float ts, float ls,
                 bool bad_anchor, int greedy_every, float qt) {
    std::normal_distribution<float> nd(0.f, 1.f);
    w1.resize((size_t)w.V * w.R);
    w2.resize((size_t)w.V * w.R);
    for (auto& x : w1) x = f2bh(nd(rng) * ts);
    for (auto& x : w2) x = f2bh(nd(rng) * ts);
    w.w1 = w1.data();
    w.w2 = w2.data();
    w.logits.resize((size_t)w.S * w.steps * w.V);
    for (auto& x : w.logits) x = rb(nd(rng) * ls);
    w.anchor.assign((size_t)(w.S - 1) * w.astride + 1, 0);
    std::uniform_int_distribution<int> id(0, w.V - 1);
    for (int s = 0; s < w.S; ++s) w.anchor[(size_t)s * w.astride] = id(rng);
    if (bad_anchor && w.S > 1) {
        w.anchor[0] = -1;
        w.anchor[(size_t)(w.S - 1) * w.astride] = w.V + 3;
    }
    w.draw.assign((size_t)w.S, RadMtpCoupleDraw{});
    w.card.assign((size_t)w.S, 0);
    w.qt.assign((size_t)w.S, qt);
    for (int s = 0; s < w.S; ++s) {
        RadMtpCoupleDraw& d = w.draw[(size_t)s];
        d.seed = 1234 + 7919ull * (uint64_t)rng();
        d.base = 100 + (uint64_t)(rng() % 5000);
        d.seq = 1000 + (uint64_t)s;
        d.card = (int32_t)((s * 3 + 1) % 16);
        d.mode = greedy_every > 0 && s % greedy_every == greedy_every - 1 ? 0u : 1u;
        w.card[(size_t)s] = d.card;
    }
}

/* Regularised upper incomplete gamma Q(a, x) (series / continued fraction), for the chi-square p-value. */
double gammq(double a, double x) {
    if (x <= 0) return 1.0;
    const double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) { ap += 1; del *= x / ap; sum += del; if (std::fabs(del) < std::fabs(sum) * 1e-15) break; }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2;
        d = an * d + b; if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) break;
    }
    return std::exp(-x + a * std::log(x) - gln) * h;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
    std::mt19937 rng(4141);

    /* ---- random tables and logits; every third row greedy where asked */
    struct Rnd { int V, R, S; float ts, ls; bool bad; int greedy_every; float qt; };
    const Rnd rnd[] = { { 1000, 32, 1, 0.5f, 2.f, false, 0, 1.0f },  { 1537, 64, 3, 0.4f, 3.f, true, 3, 1.0f },
                        { 2048, 256, 8, 0.4f, 2.5f, false, 3, 0.7f }, { 3001, 256, 11, 0.4f, 2.5f, true, 4, 1.0f },
                        { 777, 512, 2, 0.3f, 1.f, false, 0, 1.3f },   { 1200, 96, 5, 0.4f, 0.5f, true, 1, 1.0f } };
    for (const Rnd& c : rnd)
        for (int seed = 0; seed < (quick ? 1 : 2); ++seed) {
            Walk w;
            w.V = c.V; w.R = c.R; w.S = c.S;
            std::vector<uint16_t> w1, w2;
            fill_random(w, w1, w2, rng, c.ts, c.ls, c.bad, c.greedy_every, c.qt);
            check(c.greedy_every == 1 ? "random, every row greedy (temperature 0)" : c.bad ? "random, anchors out of vocab"
                                                                                        : "random", w, { 1, 2, 3, 4 });
        }

    /* ---- a draw naming another card, and q_temp 0 / -1 / inf / NaN: status -1, the greedy token */
    {
        Walk w;
        w.V = 1300; w.R = 64; w.S = 6;
        std::vector<uint16_t> w1, w2;
        fill_random(w, w1, w2, rng, 0.4f, 2.f, false, 0, 1.0f);
        w.card[1] = w.draw[1].card + 1;
        w.qt[2] = 0.0f;
        w.qt[3] = -1.0f;
        w.qt[4] = std::numeric_limits<float>::infinity();
        w.qt[5] = std::numeric_limits<float>::quiet_NaN();
        std::vector<int32_t> want;
        std::vector<RadMtpCoupleState> wst;
        ref_walk(w, want, wst);
        int m1 = 0;
        for (int s = 1; s < 6; ++s)
            for (int l = 0; l < w.steps; ++l) m1 += wst[(size_t)l * w.S + s].status == -1;
        ++g_cases;
        std::printf("%-54s rows 1..5 every round status -1: %s\n", "another card, q_temp 0 / -1 / inf / NaN (reference)",
                    m1 == 5 * w.steps ? "yes" : "NO");
        if (m1 != 5 * w.steps) ++g_fails;
        check("another card, q_temp 0 / -1 / inf / NaN", w, { 1, 2, 4 });
    }
    /* ---- NaN, -inf and +inf logits */
    {
        Walk w;
        w.V = 1800; w.R = 32; w.S = 3;
        std::vector<uint16_t> w1, w2;
        fill_random(w, w1, w2, rng, 0.4f, 2.f, false, 3, 1.0f);
        for (size_t i = 0; i < w.logits.size(); i += 37) w.logits[i] = std::numeric_limits<float>::quiet_NaN();
        for (size_t i = 5; i < w.logits.size(); i += 53) w.logits[i] = -std::numeric_limits<float>::infinity();
        float* all_nan = w.logits.data() + (size_t)(1 * w.steps + 2) * w.V;
        for (int v = 0; v < w.V; ++v) all_nan[v] = std::numeric_limits<float>::quiet_NaN();
        float* all_inf = w.logits.data() + (size_t)(0 * w.steps + 1) * w.V;
        for (int v = 0; v < w.V; ++v) all_inf[v] = -std::numeric_limits<float>::infinity();
        all_inf[1500] = std::numeric_limits<float>::quiet_NaN();
        w.logits[(size_t)(0 * w.steps + 3) * w.V + 700] = std::numeric_limits<float>::infinity();   /* +inf: Q refused */
        check("NaN / -inf / +inf logits, a row of NaN, a row of -inf", w, { 1, 2, 3, 4 });
    }

    /* ---- the law: x ~ Q, Q = softmax(score / q_temp) over the eight best */
    {
        Walk w;
        w.V = 600; w.R = 32; w.S = 1; w.steps = 1; w.astride = 1;
        std::vector<uint16_t> w1, w2;
        fill_random(w, w1, w2, rng, 0.3f, 1.2f, false, 0, 0.8f);
        /* scores spread over the top eight */
        std::vector<float> u((size_t)w.R);
        for (int r = 0; r < w.R; ++r) u[(size_t)r] = b2f(w1[(size_t)w.anchor[0] * w.R + r]);
        uint64_t top[8] = {};
        for (int v = 0; v < w.V; ++v) r4d_wsamp_insert(top, r4d_wsamp_key(score_of(w, u.data(), 0, 0, v), v));
        /* an independent FP64 softmax of the eight */
        double sc[8], mx = -1e300, z = 0;
        for (int i = 0; i < 8; ++i) { sc[i] = (double)(r4d_wsamp_key_score(top[i]) / 0.8f); mx = std::max(mx, sc[i]); }
        double q[8];
        for (int i = 0; i < 8; ++i) z += (q[i] = std::exp(sc[i] - mx));
        for (int i = 0; i < 8; ++i) q[i] /= z;
        const int N = quick ? 2000 : 6000;
        std::vector<double> cnt(8, 0);
        double qdev = 0;
        int invalid = 0;
        for (int n = 0; n < N; ++n) {
            w.draw[0].seed = 99 + 31ull * (uint64_t)n;
            std::vector<std::vector<int32_t>> tok;
            std::vector<std::vector<RadMtpCoupleState>> st;
            std::string why;
            if (n < 3 && !sharded_walk(w, 2, tok, st, &why)) { ++invalid; continue; }   /* the kernel itself, a few */
            RadMtpCoupleState o;
            const int32_t x = r4d_wsamp_step(top, 0.8f, w.draw[0], true, 1, w.V, &o);
            if (n < 3 && (tok[0][0] != x || !same_state(st[0][0], o))) ++invalid;
            if (o.status != 1 || !rad_mtp_couple_state_valid(o, w.V)) { ++invalid; continue; }
            for (int i = 0; i < 8; ++i) {
                if (o.ids[i] != r4d_wsamp_key_id(top[i])) ++invalid;
                qdev = std::max(qdev, std::fabs(o.q[i] - q[i]));
                if (o.ids[i] == x) cnt[(size_t)i] += 1;
            }
        }
        double chi = 0;
        int df = -1;
        for (int i = 0; i < 8; ++i)
            if (q[i] * N >= 5) { chi += (cnt[(size_t)i] - q[i] * N) * (cnt[(size_t)i] - q[i] * N) / (q[i] * N); ++df; }
        const double pv = df > 0 ? gammq(df / 2.0, chi / 2.0) : 1.0;
        ++g_cases;
        const bool ok = !invalid && qdev < 1e-12 && pv > 1e-4;
        std::printf("%-54s %d draws: chi2 %.2f df %d p %.4f; Q vs an FP64 softmax max |dq| %.2e; ids = the eight best: %s\n",
                    "x ~ Q (one position, seeds varied)", N, chi, df, pv, qdev, invalid ? "NO" : "yes");
        if (!ok) ++g_fails;
    }
    std::printf("ds41 wsamp: %d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
