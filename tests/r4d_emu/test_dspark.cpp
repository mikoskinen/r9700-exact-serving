/* tests/r4d_emu: DSpark beside its walk (r4d_dspark_extra.hip) on the host SIMT emulator.
 *
 * dspark_conf against the formula, dspark_alt against the walk written out here: every value is a
 * small multiple of a power of two, so every sum is exact in f32 whatever its order and the device
 * must match bit for bit (ties included: the alternative ranks and the walk break them on the lower
 * candidate index). Pitched planes, out-of-range ids (an unfilled candidate, a padding anchor), the
 * walk's own first proposal left out, a branch with no candidate left, and the outputs' padding
 * untouched. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void t2(RadTensor& t, void* p, uint32_t dt, long long rows, long long cols, long long ld) {
    t = RadTensor{};
    t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = rows; t.shape[1] = cols; t.stride[0] = ld; t.stride[1] = 1;
}
static void t1(RadTensor& t, void* p, uint32_t dt, long long n) {
    t = RadTensor{};
    t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
}

static int conf_cases(std::mt19937& rng, int& cases) {
    int fails = 0;
    struct Sh { int M, steps, D, R, V, astr, xpad, opad; float bias; };
    const Sh shapes[] = { { 1, 5, 5120, 256, 900, 5, 0, 0, 0.0f }, { 3, 5, 5120, 256, 900, 5, 8, 3, 0.25f },
                          { 8, 5, 640, 64, 300, 5, 0, 1, -1.5f }, { 2, 1, 300, 32, 50, 1, 4, 0, 0.0f },
                          { 5, 7, 1000, 300, 77, 7, 2, 2, 0.5f } };
    std::uniform_int_distribution<int> small(-15, 15), tiny(-7, 7);
    for (const Sh& sh : shapes) {
        const int M = sh.M, S = sh.steps, D = sh.D, R = sh.R, V = sh.V;
        const int xld = D + sh.xpad, old = S + sh.opad, tld = S;   /* tokens [M, S] */
        std::vector<uint16_t> x((size_t)M * S * xld), pred((size_t)V * R);
        std::vector<float> w((size_t)(D + R)), out((size_t)M * old, 777.0f), ref((size_t)M * S);
        std::vector<int32_t> anc((size_t)M * sh.astr), tok((size_t)M * tld);
        for (auto& v : x) v = f2b((float)tiny(rng) * 0.125f);
        for (auto& v : pred) v = f2b((float)tiny(rng) * 0.125f);
        for (auto& v : w) v = (float)small(rng) * 0.0625f;
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < sh.astr; ++i) anc[(size_t)m * sh.astr + i] = (int32_t)(rng() % V);
            if (m == 1) anc[(size_t)m * sh.astr] = -1;          /* a padding anchor: no Markov term */
            for (int j = 0; j < tld; ++j) tok[(size_t)m * tld + j] = (int32_t)(rng() % (V + 3)) - 1;
        }
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < S; ++j) {
                const int pid = j == 0 ? anc[(size_t)m * sh.astr] : tok[(size_t)m * tld + j - 1];
                double acc = sh.bias;
                for (int k = 0; k < D; ++k) acc += (double)w[k] * b2f(x[((size_t)m * S + j) * xld + k]);
                if (pid >= 0 && pid < V)
                    for (int r = 0; r < R; ++r) acc += (double)w[D + r] * b2f(pred[(size_t)pid * R + r]);
                ref[(size_t)m * S + j] = (float)acc;
            }
        RadTensor t[6];
        t2(t[0], x.data(), RAD_BF16, (long long)M * S, D, xld);
        t1(t[1], anc.data(), RAD_I32, (long long)M * sh.astr);
        t2(t[2], tok.data(), RAD_I32, M, tld, tld);
        t2(t[3], pred.data(), RAD_BF16, V, R, R);
        t1(t[4], w.data(), RAD_F32, D + R);
        t2(t[5], out.data(), RAD_F32, M, S, old);
        RadParam p[8]{};
        p[0] = { "M", RAD_P_INT, M }; p[1] = { "steps", RAD_P_INT, S }; p[2] = { "n_embd", RAD_P_INT, D };
        p[3] = { "rank", RAD_P_INT, R }; p[4] = { "n_vocab", RAD_P_INT, V };
        p[5] = { "anchor_stride", RAD_P_INT, sh.astr };
        p[6].key = "bias"; p[6].kind = RAD_P_F64; p[6].dval = sh.bias;
        p[7].key = "dtype"; p[7].kind = RAD_P_STR; p[7].sval = "bf16";
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 8;
        const int rc = r4d_rad_dspark_conf(&a, nullptr);
        bool ok = rc == 0;
        for (int m = 0; m < M && ok; ++m) {
            for (int j = 0; j < S && ok; ++j) ok = out[(size_t)m * old + j] == ref[(size_t)m * S + j];
            for (int j = S; j < old && ok; ++j) ok = out[(size_t)m * old + j] == 777.0f;
        }
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("FAIL dspark_conf M %d steps %d D %d R %d rc %d: out[0] %.9g ref %.9g\n", M, S, D, R, rc,
                        out[0], ref[0]);
        }
    }
    return fails;
}

/* The walk written out: row 0 scored against the anchor, the n_alt best valid candidates other than
 * the walk's own first proposal, then dflash_select's greedy rule from each. */
static void alt_model(const std::vector<int32_t>& cand, const std::vector<float>& un, int c_ld,
                      const std::vector<int32_t>& anc, int astr, const std::vector<int32_t>& tok,
                      const std::vector<uint16_t>& pred, const std::vector<uint16_t>& succ, int M, int S,
                      int K, int R, int V, int na, std::vector<int32_t>& out) {
    out.assign((size_t)M * na * S, 0);
    auto score = [&](long long row, int pid, std::vector<float>& sc) {
        sc.assign((size_t)K, 0.0f);
        for (int c = 0; c < K; ++c) {
            const int id = cand[(size_t)row * c_ld + c];
            double d = 0.0;
            if (id >= 0 && id < V && pid >= 0 && pid < V)
                for (int r = 0; r < R; ++r) d += (double)b2f(pred[(size_t)pid * R + r]) * b2f(succ[(size_t)id * R + r]);
            sc[(size_t)c] = (float)((double)un[(size_t)row * c_ld + c] + d);
        }
    };
    std::vector<float> sc;
    for (int m = 0; m < M; ++m) {
        const long long row0 = (long long)m * S;
        score(row0, anc[(size_t)m * astr], sc);
        std::vector<bool> used((size_t)K);
        for (int c = 0; c < K; ++c) {
            const int id = cand[(size_t)row0 * c_ld + c];
            used[(size_t)c] = id < 0 || id >= V || id == tok[(size_t)m * S];
        }
        for (int a = 0; a < na; ++a) {
            int best = -1;
            for (int c = 0; c < K; ++c)
                if (!used[(size_t)c] && (best < 0 || sc[(size_t)c] > sc[(size_t)best])) best = c;
            int32_t* o = &out[((size_t)m * na + a) * S];
            if (best < 0) { for (int l = 0; l < S; ++l) o[l] = -1; continue; }
            used[(size_t)best] = true;
            int pid = cand[(size_t)row0 * c_ld + best];
            o[0] = pid;
            std::vector<float> s2;
            for (int l = 1; l < S; ++l) {
                score(row0 + l, pid, s2);
                int b = 0;
                for (int c = 1; c < K; ++c)
                    if (s2[(size_t)c] > s2[(size_t)b]) b = c;
                pid = cand[(size_t)(row0 + l) * c_ld + b];
                o[l] = pid;
            }
        }
    }
}

static int alt_cases(std::mt19937& rng, int& cases) {
    int fails = 0;
    struct Sh { int M, S, K, R, V, na, astr, cpad, opad, mode; };
    /* mode 0 random, 1 ties everywhere, 2 a row 0 with only invalid ids or the first proposal */
    const Sh shapes[] = { { 1, 5, 32, 256, 600, 2, 5, 0, 0, 0 }, { 4, 5, 32, 256, 600, 3, 5, 3, 2, 1 },
                          { 8, 5, 16, 64, 120, 2, 5, 0, 1, 0 }, { 3, 4, 8, 32, 40, 4, 4, 1, 0, 2 },
                          { 2, 1, 5, 96, 60, 2, 1, 0, 3, 1 } };
    std::uniform_int_distribution<int> tiny(-7, 7), small(-15, 15);
    for (const Sh& sh : shapes) {
        const int M = sh.M, S = sh.S, K = sh.K, R = sh.R, V = sh.V, na = sh.na;
        const int c_ld = K + sh.cpad, o_ld = na * S + sh.opad;
        std::vector<int32_t> cand((size_t)M * S * c_ld), anc((size_t)M * sh.astr), tok((size_t)M * S);
        std::vector<float> un((size_t)M * S * c_ld);
        std::vector<uint16_t> pred((size_t)V * R), succ((size_t)V * R);
        for (auto& v : pred) v = f2b((float)tiny(rng) * 0.125f);
        for (auto& v : succ) v = f2b((float)(sh.mode == 1 ? (int)(rng() % 2) : tiny(rng)) * 0.125f);
        for (int m = 0; m < M; ++m) {
            anc[(size_t)m * sh.astr] = (int32_t)(rng() % V);
            for (int l = 0; l < S; ++l) {
                std::vector<int32_t> ids;
                for (int c = 0; c < K; ++c) {
                    int32_t id;
                    do { id = (int32_t)(rng() % V); } while (std::find(ids.begin(), ids.end(), id) != ids.end());
                    ids.push_back(id);
                }
                for (int c = 0; c < K; ++c) {
                    int32_t id = ids[(size_t)c];
                    if (rng() % 9 == 0) id = -1;                    /* an unfilled slot */
                    cand[((size_t)m * S + l) * c_ld + c] = id;
                    un[((size_t)m * S + l) * c_ld + c] = sh.mode == 1 ? 1.0f : (float)small(rng) * 0.0625f;
                }
            }
            tok[(size_t)m * S] = cand[((size_t)m * S) * c_ld + (int)(rng() % K)];
            if (sh.mode == 2 && m == 1) {                          /* nothing left after the first proposal */
                for (int c = 0; c < K; ++c) cand[((size_t)m * S) * c_ld + c] = c == 2 ? 11 : -1;
                tok[(size_t)m * S] = 11;
            }
        }
        std::vector<int32_t> ref, out((size_t)M * o_ld, 4242);
        alt_model(cand, un, c_ld, anc, sh.astr, tok, pred, succ, M, S, K, R, V, na, ref);
        RadTensor t[7];
        t2(t[0], cand.data(), RAD_I32, (long long)M * S, K, c_ld);
        t2(t[1], un.data(), RAD_F32, (long long)M * S, K, c_ld);
        t1(t[2], anc.data(), RAD_I32, (long long)M * sh.astr);
        t2(t[3], tok.data(), RAD_I32, M, S, S);
        t2(t[4], pred.data(), RAD_BF16, V, R, R);
        t2(t[5], succ.data(), RAD_BF16, V, R, R);
        t2(t[6], out.data(), RAD_I32, M, na * S, o_ld);
        RadParam p[8]{};
        p[0] = { "M", RAD_P_INT, M }; p[1] = { "steps", RAD_P_INT, S }; p[2] = { "top_k", RAD_P_INT, K };
        p[3] = { "rank", RAD_P_INT, R }; p[4] = { "n_vocab", RAD_P_INT, V };
        p[5] = { "anchor_stride", RAD_P_INT, sh.astr }; p[6] = { "n_alt", RAD_P_INT, na };
        p[7].key = "dtype"; p[7].kind = RAD_P_STR; p[7].sval = "bf16";
        RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 8;
        const int rc = r4d_rad_dspark_alt(&a, nullptr);
        bool ok = rc == 0;
        int bad_m = -1;
        for (int m = 0; m < M && ok; ++m) {
            for (int i = 0; i < na * S && ok; ++i) ok = out[(size_t)m * o_ld + i] == ref[(size_t)m * na * S + i];
            for (int i = na * S; i < o_ld && ok; ++i) ok = out[(size_t)m * o_ld + i] == 4242;
            if (!ok) bad_m = m;
        }
        if (ok && sh.mode == 2) ok = ref[(size_t)1 * na * S] == -1;   /* the empty branch is reported */
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("FAIL dspark_alt M %d S %d K %d R %d na %d mode %d rc %d (sequence %d)\n", M, S, K, R, na,
                        sh.mode, rc, bad_m);
            if (bad_m >= 0)
                for (int i = 0; i < na * S; ++i)
                    std::printf("  [%d] dev %d ref %d\n", i, out[(size_t)bad_m * o_ld + i], ref[(size_t)bad_m * na * S + i]);
        }
    }
    return fails;
}

int main() {
    std::mt19937 rng(41);
    int cases = 0;
    const int f1 = conf_cases(rng, cases);
    const int f2 = alt_cases(rng, cases);
    std::printf("dspark_conf / dspark_alt: %d cases, %d failure(s)\n", cases, f1 + f2);
    return (f1 + f2) != 0;
}
