/* tests/sd_exact_test.cpp -- lever SD (sampled drafting, fn4 draft_sample): the emitted token's law is the target's.
 *
 * Lever SD drafts x ~ Q (the W1 walk's eight best keys, softmax at q_temp: libr4d/r4d_ds41_wsamp_math.h) and verifies
 * with the maximal coupling (abi/rad_spechub.h rad_mtp_couple_decide, which the spechub plugin's mtp_couple_verify
 * computes on the card): x is kept with probability min(1, P(x) / Q(x)), otherwise the token is drawn from
 * norm(max(0, P - Q)). Acceptance is then by agreement (core/sample/accept.h): a draft is accepted while the verify's
 * token equals it. Every uniform is the engine's: the draft's at (seed, base, KEY_DRAFT + round), the verify's at
 * (seed, pos, KEY_ACCEPT) and (seed, pos, KEY_RESID), the ordinary pick's (a position with no bound state) a separate
 * stream. Seeds vary over the draws, as requests and positions do.
 *
 * Large-N frequency checks, each a chi-square goodness of fit against the exact P (p-value printed, required > 1e-4,
 * i.e. a false alarm about once in 10^4 runs; the seeds are fixed, so a run is reproducible):
 *   residual        P and Q overlap with P(x) < Q(x) on some of Q's tokens: rejections, residual draws
 *   all accepted    P = Q on Q's support (the same scores at the same temperature): every draft is kept
 *   q zeros         Q's eight cover a part of P's support (P has mass where Q has none), Q with fewer than eight live
 *                   keys (-inf scores) and an entry whose weight underflows to zero
 *   top-p           P is the target's law after a top-p cut (the candidate plane the pick reads), Q puts mass on
 *                   tokens the cut removed: never emitted
 *   chain           two positions, Q_2 and P_2 depending on the token before; the emitted pair's law is P_1 x P_2
 *                   with acceptance by agreement and the second position's ordinary draw after a rejection
 * and the reduction at temperature 0: a greedy draw (mode 0) proposes the best key's id -- lever W1's token -- with
 * status 0, which the sampler never couples (rad_mtp_couple_binds), so the verify is today's argmax match. */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"
#include "r4d_ds41_wsamp_math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

double gammq(double a, double x) {   /* regularised upper incomplete gamma */
    if (x <= 0) return 1.0;
    const double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 2000; ++n) { ap += 1; del *= x / ap; sum += del; if (std::fabs(del) < std::fabs(sum) * 1e-15) break; }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
    for (int i = 1; i < 2000; ++i) {
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

/* chi-square of counts against probabilities (cells with expectation < 5 pooled into one); p-value */
double chi_p(const std::vector<double>& cnt, const std::vector<double>& p, double n, double* chi_out, int* df_out,
             double* outside) {
    double chi = 0, pool_o = 0, pool_e = 0, out = 0;
    int cells = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const double e = p[i] * n;
        if (p[i] == 0) { out += cnt[i]; continue; }
        if (e < 5) { pool_o += cnt[i]; pool_e += e; continue; }
        chi += (cnt[i] - e) * (cnt[i] - e) / e;
        ++cells;
    }
    if (pool_e > 0) { chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e; ++cells; }
    const int df = cells - 1;
    if (chi_out) *chi_out = chi;
    if (df_out) *df_out = df;
    if (outside) *outside = out;
    return df > 0 ? gammq(df / 2.0, chi / 2.0) : 1.0;
}

/* The target's law over a candidate plane, as rad_mtp_couple_decide sees it: FP64 softmax of the values; ids < 0 and
 * -inf carry no mass. P over the vocabulary [0, V). */
std::vector<double> law_of(const std::vector<int32_t>& ids, const std::vector<float>& val, int V) {
    std::vector<double> p((size_t)V, 0.0);
    float top = -INFINITY;
    for (size_t j = 0; j < ids.size(); ++j) if (ids[j] >= 0 && val[j] > top) top = val[j];
    double z = 0;
    for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] >= 0 && val[j] != -INFINITY) { const double w = std::exp((double)val[j] - top); p[(size_t)ids[j]] += w; z += w; }
    for (auto& x : p) x /= z;
    return p;
}

/* The walk's eight best keys from scores over [0, V) (-inf allowed, NaN skipped, as the walk keys them). */
void top8(const std::vector<float>& score, uint64_t* top) {
    for (int i = 0; i < 8; ++i) top[i] = 0;
    for (size_t v = 0; v < score.size(); ++v) r4d_wsamp_insert(top, r4d_wsamp_key(score[v], (int32_t)v));
}

RadMtpCoupleDraw draw_of(uint64_t seed, uint64_t base, uint64_t seq, int32_t card, uint32_t mode) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = seq; d.card = card; d.mode = mode;
    return d;
}

/* One position as the engine runs it: x ~ Q from `score` (q_temp), the coupled verify against the candidate plane
 * (ids, val) at output position `pos`. Returns the emitted token; *x the draft, *acc whether it was kept. */
int32_t one_position(const std::vector<float>& score, float q_temp, const std::vector<int32_t>& ids,
                     const std::vector<float>& val, int V, uint64_t seed, uint64_t base, uint64_t pos, int round,
                     int32_t* x, bool* acc, RadMtpCoupleState* st_out = nullptr) {
    uint64_t top[8];
    top8(score, top);
    RadMtpCoupleState st;
    *x = r4d_wsamp_step(top, q_temp, draw_of(seed, base, 7, 3, 1), true, round, V, &st);
    if (st_out) *st_out = st;
    if (st.status != 1) return -2;
    int32_t out = -1;
    if (!rad_mtp_couple_decide(st, ids.data(), val.data(), (int64_t)ids.size(), V,
                               rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                               rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), &out))
        return -3;
    *acc = out == *x;
    return out;
}

/* The ordinary pick's law at a position with no bound state: inverse CDF over P with an independent uniform. */
int32_t pick_from(const std::vector<double>& p, double u) {
    double c = u;
    int32_t last = -1;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0) { last = (int32_t)i; c -= p[i]; if (c < 0) return (int32_t)i; }
    return last;
}

struct Freq {
    std::vector<double> cnt;
    double acc = 0, n = 0, bad = 0;
};

/* N positions of one (score, q_temp, plane) case; seeds vary. */
Freq run_case(const std::vector<float>& score, float q_temp, const std::vector<int32_t>& ids,
              const std::vector<float>& val, int V, int N, uint64_t salt) {
    Freq f;
    f.cnt.assign((size_t)V, 0);
    for (int n = 0; n < N; ++n) {
        const uint64_t seed = 0x5d1ull + salt * 0x9e3779b97f4a7c15ull + (uint64_t)n * 0x2545f4914f6cdd1dull;
        const uint64_t base = 1000 + (uint64_t)(n % 977), pos = base + 1;
        int32_t x = -1;
        bool acc = false;
        const int32_t y = one_position(score, q_temp, ids, val, V, seed, base, pos, 1, &x, &acc);
        if (y < 0 || y >= V) { f.bad += 1; continue; }
        f.cnt[(size_t)y] += 1;
        f.acc += acc;
        f.n += 1;
    }
    return f;
}

/* Expected acceptance: sum_y min(P(y), Q(y)) with the stored Q. */
double expect_acc(const std::vector<float>& score, float q_temp, const std::vector<double>& p, int V) {
    uint64_t top[8];
    top8(score, top);
    RadMtpCoupleState st;
    (void)r4d_wsamp_step(top, q_temp, draw_of(1, 1, 7, 3, 1), true, 1, V, &st);
    double a = 0;
    for (int k = 0; k < st.count; ++k) a += std::min(p[(size_t)st.ids[k]], st.q[k]);
    return a;
}

void report(const char* what, const Freq& f, const std::vector<double>& p, double acc_want) {
    double chi = 0, out = 0;
    int df = 0;
    const double pv = chi_p(f.cnt, p, f.n, &chi, &df, &out);
    const double acc = f.acc / f.n, se = std::sqrt(acc_want * (1 - acc_want) / f.n);
    std::printf("  %-14s N %.0f: chi2 %.2f df %d p-value %.4f; emitted outside P's support %.0f; acceptance %.4f "
                "(sum min(P, Q) %.4f, %.1f sigma); malformed %.0f\n",
                what, f.n, chi, df, pv, out, acc, acc_want, se > 0 ? (acc - acc_want) / se : 0.0, f.bad);
    CHECK(f.bad == 0);
    CHECK(out == 0);
    CHECK(pv > 1e-4);
    CHECK(se > 0 ? std::fabs(acc - acc_want) < 5 * se + 1e-12 : std::fabs(acc - acc_want) < 1e-12);
}

}  // namespace

/* P and Q overlap; Q over-weights some tokens (rejections and residual draws). */
TEST(sd_the_emitted_token_is_distributed_as_p_with_rejections) {
    const int V = 40, N = 400000;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> score((size_t)V), tv((size_t)V);
    for (int v = 0; v < V; ++v) { score[(size_t)v] = 1.5f * nd(rng); tv[(size_t)v] = score[(size_t)v] + 1.2f * nd(rng); }
    std::vector<int32_t> ids;
    std::vector<float> val;
    for (int v = 0; v < 30; ++v) { ids.push_back(v); val.push_back(tv[(size_t)v]); }   /* the plane: 30 candidates */
    const std::vector<double> p = law_of(ids, val, V);
    const Freq f = run_case(score, 1.0f, ids, val, V, N, 1);
    report("residual", f, p, expect_acc(score, 1.0f, p, V));
    CHECK(f.acc / f.n < 0.97);   /* rejections happened */
}

/* P = Q on Q's support: every draft is kept, and the emitted law is still P. */
TEST(sd_every_draft_is_accepted_when_p_equals_q) {
    const int V = 64, N = 300000;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> score((size_t)V, -INFINITY);
    for (int v = 0; v < V; ++v) score[(size_t)v] = 2.f * nd(rng);
    uint64_t top[8];
    top8(score, top);
    /* the target's plane: exactly the eight at the same values / temperature, so pq == q bit for bit */
    const float qt = 0.75f;
    std::vector<int32_t> ids;
    std::vector<float> val;
    for (int i = 0; i < 8; ++i) { ids.push_back(r4d_wsamp_key_id(top[i])); val.push_back(r4d_wsamp_key_score(top[i]) / qt); }
    const std::vector<double> p = law_of(ids, val, V);
    const Freq f = run_case(score, qt, ids, val, V, N, 2);
    report("all accepted", f, p, 1.0);
    CHECK(f.acc == f.n);
}

/* Q has zero mass where P has mass: Q's eight cover part of P's support; fewer than eight live keys; an underflow. */
TEST(sd_q_with_zeros_keeps_the_law) {
    const int V = 50, N = 400000;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    /* (a) P over 25 candidates, Q's eight among the walk's best of the whole vocabulary */
    {
        std::vector<float> score((size_t)V), tv((size_t)V);
        for (int v = 0; v < V; ++v) { score[(size_t)v] = 1.2f * nd(rng); tv[(size_t)v] = 0.8f * nd(rng); }
        std::vector<int32_t> ids;
        std::vector<float> val;
        for (int v = 0; v < 25; ++v) { ids.push_back(v); val.push_back(tv[(size_t)v]); }
        const std::vector<double> p = law_of(ids, val, V);
        report("q zeros (a)", run_case(score, 1.0f, ids, val, V, N, 3), p, expect_acc(score, 1.0f, p, V));
    }
    /* (b) three live keys (the rest -inf), one of them 2000 below the best: its weight underflows, so Q has two */
    {
        std::vector<float> score((size_t)V, -INFINITY);
        score[4] = 1.0f; score[9] = 0.3f; score[17] = -2000.0f;
        uint64_t top[8];
        top8(score, top);
        RadMtpCoupleState st;
        (void)r4d_wsamp_step(top, 1.0f, draw_of(5, 5, 7, 3, 1), true, 1, V, &st);
        CHECK_EQ(st.status, 1);
        CHECK_EQ(st.count, 2);
        CHECK(rad_mtp_couple_state_valid(st, V));
        std::vector<int32_t> ids;
        std::vector<float> val;
        for (int v = 0; v < 12; ++v) { ids.push_back(v); val.push_back(0.4f * nd(rng)); }
        const std::vector<double> p = law_of(ids, val, V);
        report("q zeros (b)", run_case(score, 1.0f, ids, val, V, 1000000, 4), p, expect_acc(score, 1.0f, p, V));
        report("q zeros (b')", run_case(score, 1.0f, ids, val, V, 1000000, 40), p, expect_acc(score, 1.0f, p, V));
    }
}

/* P after a top-p cut: the plane holds only the kept candidates; Q puts mass on removed ones, never emitted. */
TEST(sd_top_p_truncated_target_law_is_kept) {
    const int V = 60, N = 400000;
    std::mt19937 rng(14);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> score((size_t)V), tv((size_t)V);
    for (int v = 0; v < V; ++v) { tv[(size_t)v] = 1.3f * nd(rng); score[(size_t)v] = tv[(size_t)v] + 0.7f * nd(rng); }
    /* top-p 0.8 over the target's full softmax, as the sampler's cut keeps the smallest prefix reaching it */
    std::vector<int32_t> order((size_t)V);
    for (int v = 0; v < V; ++v) order[(size_t)v] = v;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return tv[(size_t)a] > tv[(size_t)b]; });
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp((double)tv[(size_t)v] - tv[(size_t)order[0]]);
    std::vector<int32_t> ids;
    std::vector<float> val;
    double cum = 0;
    for (int i = 0; i < V && cum < 0.8; ++i) {
        const int v = order[(size_t)i];
        cum += std::exp((double)tv[(size_t)v] - tv[(size_t)order[0]]) / z;
        ids.push_back(v);
        val.push_back(tv[(size_t)v]);
    }
    const std::vector<double> p = law_of(ids, val, V);
    /* the drafter favours three tokens the cut removed (its law is not the target's), so Q has mass outside P */
    for (int i = (int)ids.size(), k = 0; i < V && k < 3; ++i, ++k) score[(size_t)order[(size_t)(V - 1 - k)]] += 6.0f;
    uint64_t top[8];
    top8(score, top);
    int outside = 0;
    for (int i = 0; i < 8; ++i) outside += p[(size_t)r4d_wsamp_key_id(top[i])] == 0;
    std::printf("  top-p 0.8 keeps %zu of %d candidates; %d of Q's eight are outside it\n", ids.size(), V, outside);
    CHECK(outside > 0);
    report("top-p", run_case(score, 1.0f, ids, val, V, N, 5), p, expect_acc(score, 1.0f, p, V));
}

/* Two positions: Q_2 depends on the draft x_1, P_2 on the emitted y_1. Acceptance by agreement; after a rejection the
 * next position is the target's ordinary draw (the next step's first row). The emitted pair ~ P_1 x P_2(. | y_1). */
TEST(sd_a_chain_of_two_positions_keeps_the_joint_law) {
    const int V = 6, N = 600000;
    std::mt19937 rng(15);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> s1((size_t)V), t1((size_t)V);
    std::vector<std::vector<float>> s2((size_t)V, std::vector<float>((size_t)V)), t2 = s2;
    for (int v = 0; v < V; ++v) { s1[(size_t)v] = nd(rng); t1[(size_t)v] = nd(rng); }
    for (int a = 0; a < V; ++a)
        for (int v = 0; v < V; ++v) { s2[(size_t)a][(size_t)v] = 1.5f * nd(rng); t2[(size_t)a][(size_t)v] = 1.5f * nd(rng); }
    std::vector<int32_t> ids;
    for (int v = 0; v < V; ++v) ids.push_back(v);
    const std::vector<double> p1 = law_of(ids, t1, V);
    std::vector<std::vector<double>> p2((size_t)V);
    for (int a = 0; a < V; ++a) p2[(size_t)a] = law_of(ids, t2[(size_t)a], V);
    std::vector<double> cnt((size_t)V * V, 0), pj((size_t)V * V, 0);
    for (int a = 0; a < V; ++a)
        for (int b = 0; b < V; ++b) pj[(size_t)a * V + b] = p1[(size_t)a] * p2[(size_t)a][(size_t)b];
    double both = 0, bad = 0;
    for (int n = 0; n < N; ++n) {
        const uint64_t seed = 0xc4a1ull + (uint64_t)n * 0x9e3779b97f4a7c15ull;
        const uint64_t base = 500, pos = 501;   /* the drafts' coordinate, and position 1's (round 1 at pos, round 2 at pos + 1) */
        /* the drafts: x1 ~ Q1, then x2 ~ Q2(. | x1), each at its round's uniform */
        uint64_t k1[8], k2[8];
        top8(s1, k1);
        RadMtpCoupleState q1, q2;
        const int32_t x1 = r4d_wsamp_step(k1, 1.0f, draw_of(seed, base, 7, 3, 1), true, 1, V, &q1);
        top8(s2[(size_t)x1], k2);
        const int32_t x2 = r4d_wsamp_step(k2, 1.0f, draw_of(seed, base, 7, 3, 1), true, 2, V, &q2);
        (void)x2;
        /* the verify: row 0 coupled to q1 at pos; row 1 coupled to q2 at pos + 1, read only while row 0 agreed */
        int32_t y1 = -1, y2 = -1;
        if (!rad_mtp_couple_decide(q1, ids.data(), t1.data(), V, V, rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                                   rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), &y1)) { ++bad; continue; }
        if (y1 == x1) {
            if (!rad_mtp_couple_decide(q2, ids.data(), t2[(size_t)x1].data(), V, V,
                                       rad_spechub_uniform(seed, pos + 1, RAD_SAMPLE_KEY_ACCEPT),
                                       rad_spechub_uniform(seed, pos + 1, RAD_SAMPLE_KEY_RESID), &y2)) { ++bad; continue; }
            both += y2 == x2;
        } else {
            /* rejected at row 0: the next step samples position 2 itself (no bound state there: the ordinary pick) */
            y2 = pick_from(p2[(size_t)y1], rad_spechub_uniform(seed, pos + 1, RAD_SAMPLE_KEY_PICK + 0x77ull));
        }
        cnt[(size_t)y1 * V + y2] += 1;
    }
    double chi = 0, out = 0;
    int df = 0;
    const double pv = chi_p(cnt, pj, N - bad, &chi, &df, &out);
    std::printf("  %-14s N %d: chi2 %.2f df %d p-value %.4f over the %d x %d pairs; both drafts kept %.4f\n", "chain",
                N, chi, df, pv, V, V, both / N);
    CHECK(bad == 0);
    CHECK(pv > 1e-4);
}

/* Temperature 0: a greedy draw proposes the best key's id (W1's token: greatest score, ties to the lower id, -0 = +0,
 * NaN never, nothing at all -> 0) with status 0, and the sampler never binds it. */
TEST(sd_temperature_zero_reduces_to_the_greedy_walk) {
    std::mt19937 rng(16);
    std::normal_distribution<float> nd(0.f, 1.f);
    int checked = 0;
    for (int t = 0; t < 2000; ++t) {
        const int V = 5 + (int)(rng() % 200);
        std::vector<float> score((size_t)V);
        for (auto& x : score) x = std::round(nd(rng) * 4.f) / 4.f;   /* ties are common */
        if (t % 7 == 0) score[(size_t)(rng() % V)] = NAN;
        if (t % 11 == 0) score[(size_t)(rng() % V)] = -0.0f;
        if (t % 13 == 0) for (auto& x : score) x = NAN;
        int32_t want = 0;
        bool have = false;
        float best = 0;
        for (int v = 0; v < V; ++v) {
            float s = score[(size_t)v];
            if (std::isnan(s)) continue;
            if (s == 0.0f) s = 0.0f;
            if (!have || s > best) { have = true; best = s; want = v; }
        }
        uint64_t top[8];
        top8(score, top);
        RadMtpCoupleState st;
        const RadMtpCoupleDraw d = draw_of(1234, 77, 9, 2, 0);
        const int32_t x = r4d_wsamp_step(top, 1.0f, d, true, 3, V, &st);
        CHECK_EQ(x, want);
        CHECK_EQ(st.status, 0);
        CHECK_EQ(st.x, want);
        CHECK_EQ(st.count, 0);
        CHECK(!std::memcmp(&st.draw, &d, sizeof d));
        CHECK(!rad_mtp_couple_binds(st, 9, 1234, 0, 2, x, V));
        ++checked;
    }
    std::printf("  greedy draws: %d random walks' positions, every token = W1's argmax, status 0, never bound\n", checked);
}

/* A sampled draw that cannot sample (another card, q_temp not a positive finite number, a +inf score, nothing live)
 * keeps the greedy token with status -1, which the sampler does not couple. */
TEST(sd_a_draw_that_cannot_sample_keeps_the_greedy_token) {
    std::vector<float> score = { 0.5f, 2.0f, -1.0f, 1.0f };
    uint64_t top[8];
    top8(score, top);
    RadMtpCoupleState st;
    CHECK_EQ(r4d_wsamp_step(top, 1.0f, draw_of(1, 1, 1, 1, 1), false, 1, 4, &st), 1);
    CHECK_EQ(st.status, -1);
    for (float q : { 0.0f, -1.0f, INFINITY, NAN }) {
        CHECK_EQ(r4d_wsamp_step(top, q, draw_of(1, 1, 1, 1, 1), true, 1, 4, &st), 1);
        CHECK_EQ(st.status, -1);
        CHECK(!rad_mtp_couple_binds(st, 1, 1, 0, 0, 1, 4));
    }
    score[3] = INFINITY;
    top8(score, top);
    CHECK_EQ(r4d_wsamp_step(top, 1.0f, draw_of(1, 1, 1, 1, 1), true, 1, 4, &st), 3);
    CHECK_EQ(st.status, -1);
    std::vector<float> none = { NAN, NAN };
    top8(none, top);
    CHECK_EQ(r4d_wsamp_step(top, 1.0f, draw_of(1, 1, 1, 1, 1), true, 1, 2, &st), 0);
    CHECK_EQ(st.status, -1);
    /* and a good one binds at its row's coordinate */
    std::vector<float> ok = { 0.5f, 2.0f, -1.0f, 1.0f };
    top8(ok, top);
    const int32_t x = r4d_wsamp_step(top, 1.0f, draw_of(42, 10, 5, 1, 1), true, 2, 4, &st);
    CHECK_EQ(st.status, 1);
    CHECK(rad_mtp_couple_state_valid(st, 4));
    st.pos = rad_mtp_couple_verify_pos(30, 2);
    CHECK(rad_mtp_couple_binds(st, 5, 42, 30, 1, x, 4));
}

/* The key round trip: id and score back out of every key, NaN never keyed, -0 keyed as +0. */
TEST(sd_walk_keys_round_trip) {
    std::mt19937 rng(17);
    const float specials[] = { 0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY, 1e-45f, -1e-45f, 3.4e38f, -3.4e38f };
    int n = 0;
    for (int t = 0; t < 200000; ++t) {
        float f;
        if (t < 10) f = specials[t];
        else { uint32_t u = (uint32_t)rng(); std::memcpy(&f, &u, 4); }
        const int32_t id = (int32_t)(rng() % 2000000);
        const uint64_t k = r4d_wsamp_key(f, id);
        if (std::isnan(f)) { CHECK_EQ(k, 0ull); continue; }
        CHECK(k != 0ull);
        CHECK_EQ(r4d_wsamp_key_id(k), id);
        const float g = r4d_wsamp_key_score(k);
        CHECK(g == f && (f != 0.0f || !std::signbit(g)));
        ++n;
    }
    CHECK_EQ(r4d_wsamp_key_id(0), -1);
    std::printf("  %d keys round-tripped\n", n);
}

RAD_TEST_MAIN()
