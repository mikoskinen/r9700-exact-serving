/* tests/ds41_cone_test.cpp -- lever PSK's row plans in deepseek41 (arch/deepseek41/ds41_psk_cone.h), host arithmetic only:
 *
 *   rows / plan_end   the cut rows of a layer and the stager's layer end, as design 1 / 2 issue them; lever PSK-L20
 *                     (ds41-pskl20) starts the scan at the last kv source (20).
 *   the cone rule     a row-level model of a prompt prefilled in chunks under design 1 / 2 (targets: the prompt's end,
 *                     every K-th grid point, or every chunk end), window 128, 40 layers, every row through layer 19
 *                     (layer 20 too with PSK-L20 off): a row's window key at layer l is right iff its layer input is; its
 *                     output iff its input and every key of its window at l are. Checked at every chunk: every row's
 *                     layer-20 input right (the latent / index keys come from it); at every target the window state of
 *                     every layer whole (the window - 1 keys the next position reads); at the prompt's end the last
 *                     row's layer-39 output and the DSpark taps (inputs of 37-39) of its last 129 rows. With PSK-L20 on
 *                     as off; and a control (layer 20's cone without its window of slack) fails, so the model has teeth.
 *   pfctx_runs        lever PSK-DS: the context pass's runs -- every entry's last window + 1 rows covered, runs sorted,
 *                     disjoint, each >= 160 rows, fewer rows than the pass; no plan where nothing is skipped, the
 *                     entries cannot be told apart, or the pass is short. */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_psk_cone.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kNL = 40, kW = 128, kFirst = 21, kL20 = 20;

/* The prefill of one prompt of N positions in `chunks` ([s, e) each, in order) under the targets `tg` (sorted chunk ends,
 * the prompt's end among them); `l20` lever PSK-L20; `c20` layer 20's cone (0: the rule's). Returns "" or what broke. */
std::string cone_model(int64_t N, const std::vector<std::pair<int64_t, int64_t>>& chunks, const std::vector<int64_t>& tg,
                       bool l20, int64_t c20 = 0) {
    std::vector<std::vector<uint8_t>> key((size_t)kNL, std::vector<uint8_t>((size_t)N, 0));
    std::vector<uint8_t> in((size_t)N), out((size_t)N);
    std::vector<int32_t> pre((size_t)N + 1);
    std::vector<std::vector<uint8_t>> tap(3, std::vector<uint8_t>((size_t)N, 0));
    const int cut0 = l20 ? kL20 : kFirst;
    for (const auto& ch : chunks) {
        const int64_t s = ch.first, e = ch.second, n = e - s;
        /* the first target past the chunk's first position */
        int64_t T1 = -1;
        for (int64_t t : tg)
            if (t > s) { T1 = t; break; }
        if (T1 < e) return "target list does not cover chunk " + std::to_string(s);
        const int32_t ahead = (int32_t)(T1 - e);
        for (int64_t p = s; p < e; ++p) in[(size_t)p] = 1;   /* the embedding: right */
        for (int l = 0; l < kNL; ++l) {
            int64_t a = s;   /* computed rows [a, e) */
            if (l >= cut0) {
                const int64_t C = (l == kL20 && c20 > 0) ? c20 : ds41cone::cone(kNL, kW, l);
                a = e - ds41cone::rows(C, ahead, n);
            }
            if (l == 20)   /* the latent / index keys: every row's layer-20 input */
                for (int64_t p = s; p < e; ++p)
                    if (!in[(size_t)p]) return "layer-20 input wrong at " + std::to_string(p);
            if (l >= 37)   /* the tap: the layer's input over every row of a layer the pass runs */
                for (int64_t p = s; p < e; ++p) tap[(size_t)(l - 37)][(size_t)p] = a < e && in[(size_t)p];
            /* the keys first (the pass stores a row's window key before any row's attention reads it) */
            for (int64_t p = a; p < e; ++p) key[(size_t)l][(size_t)p] = in[(size_t)p];
            const int64_t w0 = std::max<int64_t>(0, s - kW);
            pre[(size_t)w0] = 0;
            for (int64_t p = w0; p < e; ++p) pre[(size_t)p + 1] = pre[(size_t)p] + key[(size_t)l][(size_t)p];
            for (int64_t p = s; p < e; ++p) {
                if (p < a) { out[(size_t)p] = 0; continue; }
                const int64_t q0 = std::max<int64_t>(0, p - kW + 1);
                out[(size_t)p] = in[(size_t)p] && pre[(size_t)p + 1] - pre[(size_t)q0] == p + 1 - q0;
            }
            for (int64_t p = s; p < e; ++p) in[(size_t)p] = out[(size_t)p];
        }
        if (std::binary_search(tg.begin(), tg.end(), e))
            for (int l = 0; l < kNL; ++l)
                for (int64_t q = std::max<int64_t>(0, e - kW + 1); q < e; ++q)
                    if (!key[(size_t)l][(size_t)q])
                        return "target " + std::to_string(e) + ": layer " + std::to_string(l) + " key " + std::to_string(q) + " wrong";
        if (e == N) {
            if (!in[(size_t)N - 1]) return "the last row's output wrong";
            for (int64_t p = std::max<int64_t>(0, N - kW - 1); p < N; ++p)
                for (int k = 0; k < 3; ++k)
                    if (!tap[(size_t)k][(size_t)p]) return "tap " + std::to_string(37 + k) + " wrong at " + std::to_string(p);
        }
    }
    return "";
}

std::vector<std::pair<int64_t, int64_t>> chunked(int64_t N, int64_t first, int64_t rest) {
    std::vector<std::pair<int64_t, int64_t>> c;
    for (int64_t s = 0; s < N;) {
        const int64_t e = std::min(N, s + (c.empty() ? first : rest));
        c.push_back({ s, e });
        s = e;
    }
    return c;
}

/* design 2's targets: every K-th chunk end (the checkpoint grid's K-th, 2K-th, ... point) and the prompt's end */
std::vector<int64_t> targets_k(const std::vector<std::pair<int64_t, int64_t>>& c, int64_t N, int K) {
    std::vector<int64_t> t;
    for (size_t i = 0; i < c.size(); ++i)
        if (c[i].second == N || (i + 1) % (size_t)K == 0) t.push_back(c[i].second);
    return t;
}
std::vector<int64_t> targets_all(const std::vector<std::pair<int64_t, int64_t>>& c) {
    std::vector<int64_t> t;
    for (const auto& ch : c) t.push_back(ch.second);
    return t;
}

}  // namespace

TEST(cone_rows_and_cones_are_design_2s) {
    CHECK_EQ(ds41cone::cone(kNL, kW, 21), (int64_t)2464);
    CHECK_EQ(ds41cone::cone(kNL, kW, 20), (int64_t)2592);
    CHECK_EQ(ds41cone::cone(kNL, kW, 39), (int64_t)160);
    for (int64_t n : { 1, 100, 160, 161, 2048, 2592, 4096 })
        for (int32_t ah : { 0, 1, 100, 2400, 2463, 2464, 2591, 2592, 5000, ds41cone::kNoHint })
            for (int l = 20; l < kNL; ++l) {
                const int64_t C = ds41cone::cone(kNL, kW, l), r = ds41cone::rows(C, ah, n);
                const int64_t want = ah == ds41cone::kNoHint ? n : ah >= C ? 0 : std::min(n, std::max(C - ah, (int64_t)160));
                CHECK_EQ(r, want);
            }
    /* nested: layer 20's cut rows hold layer 21's, and above the floor exactly one window more */
    for (int64_t n = 1; n <= 4160; n += 7)
        for (int32_t ah = 0; ah <= 6000; ah += 13) {
            const int64_t r20 = ds41cone::rows(ds41cone::cone(kNL, kW, 20), ah, n), r21 = ds41cone::rows(ds41cone::cone(kNL, kW, 21), ah, n);
            CHECK(r20 >= r21);
            if (r21 > 0 && r21 < n && ds41cone::cone(kNL, kW, 21) - ah > 160) CHECK_EQ(r20, std::min(n, r21 + kW));
        }
}

TEST(cone_plan_end_starts_at_the_split_layer) {
    const int32_t rw[1] = { 4096 };
    struct C { int32_t ah; int from21, from20; };
    for (const C c : { C{ 0, kNL, kNL }, C{ 2463, 22, 22 }, C{ 2464, 21, 21 }, C{ 2500, 21, 21 },
                       C{ 2591, 21, 21 }, C{ 2592, 21, 20 }, C{ 9000, 21, 20 } }) {
        const int32_t ah[1] = { c.ah };
        const int e21 = ds41cone::plan_end(kNL, kW, kFirst, 1, ah, rw), e20 = ds41cone::plan_end(kNL, kW, kL20, 1, ah, rw);
        CHECK_EQ(e21, c.from21);
        CHECK_EQ(e20, c.from20);
    }
    /* two sequences: the pass ends where neither runs a row */
    const int32_t ah2[2] = { 9000, 2500 }, rw2[2] = { 4096, 2048 };
    CHECK_EQ(ds41cone::plan_end(kNL, kW, kL20, 2, ah2, rw2), 21);
    const int32_t ah3[2] = { 9000, 9000 };
    CHECK_EQ(ds41cone::plan_end(kNL, kW, kL20, 2, ah3, rw2), 20);
    /* an unplanned sequence (no hint resolved to "every row") runs every layer */
    const int32_t ah4[1] = { ds41cone::kNoHint };
    CHECK_EQ(ds41cone::plan_end(kNL, kW, kL20, 1, ah4, rw), kNL);
}

TEST(cone_rule_holds_with_the_split_layer) {
    struct P { int64_t N, first, rest; };
    int checked = 0;
    for (const P p : { P{ 136216, 2048, 4096 }, P{ 64000, 2048, 4096 }, P{ 16000, 2048, 4096 }, P{ 9000, 2048, 2048 },
                       P{ 4096 * 5 + 50, 4096, 4096 }, P{ 2048 * 3 + 130, 2048, 2048 }, P{ 2600, 2600, 2600 },
                       P{ 300, 300, 300 }, P{ 5000, 2048, 1000 } }) {
        const auto c = chunked(p.N, p.first, p.rest);
        for (const auto& tg : { targets_all(c), targets_k(c, p.N, 4), targets_k(c, p.N, 8), targets_k(c, p.N, 1000) })
            for (bool l20 : { false, true }) {
                const std::string r = cone_model(p.N, c, tg, l20);
                if (!r.empty()) fprintf(stderr, "N %lld chunks %lld/%lld targets %zu l20 %d: %s\n", (long long)p.N,
                                        (long long)p.first, (long long)p.rest, tg.size(), (int)l20, r.c_str());
                CHECK(r.empty());
                ++checked;
            }
    }
    /* random chunkings (pieces of 16 .. 4096 rows, as mixed steps cut them), targets every K-th grid end or every end */
    std::mt19937 rng(20261112);
    for (int it = 0; it < 24; ++it) {
        const int64_t N = 3000 + (int64_t)(rng() % 30000);
        std::vector<std::pair<int64_t, int64_t>> c;
        for (int64_t s = 0; s < N;) {
            const int64_t e = std::min(N, s + 16 + (int64_t)(rng() % 4081));
            c.push_back({ s, e });
            s = e;
        }
        const auto tg = it % 2 ? targets_all(c) : std::vector<int64_t>{ N };
        for (bool l20 : { false, true }) {
            const std::string r = cone_model(N, c, tg, l20);
            if (!r.empty()) fprintf(stderr, "random %d N %lld l20 %d: %s\n", it, (long long)N, (int)l20, r.c_str());
            CHECK(r.empty());
            ++checked;
        }
    }
    CHECK(checked > 100);
}

TEST(cone_rule_control_layer_20_without_its_slack_fails) {
    /* layer 20 cut to layer 21's cone (no window of slack): layer 21's first rows read layer-20 outputs whose window at 20
     * was not stored -- the model must say so, or it proves nothing */
    const auto c = chunked(16000, 2048, 4096);
    CHECK(cone_model(16000, c, targets_all(c), true).empty());
    CHECK(!cone_model(16000, c, targets_all(c), true, ds41cone::cone(kNL, kW, 21)).empty());
}

namespace {
/* positions for entries of the given lengths, each at its own base (far apart: told apart) */
std::vector<int32_t> pos_of(const std::vector<int64_t>& len, std::vector<int64_t>* starts = nullptr) {
    std::vector<int32_t> p;
    int32_t base = 1000;
    for (int64_t n : len) {
        if (starts) starts->push_back((int64_t)p.size());
        for (int64_t j = 0; j < n; ++j) p.push_back(base + (int32_t)j);
        base += (int32_t)n + 100000;
    }
    return p;
}
std::string runs_ok(const std::vector<int64_t>& len, int64_t keep, int64_t mr, const std::vector<std::pair<int64_t, int64_t>>& ru,
                    int64_t kept) {
    std::vector<int64_t> st;
    const auto p = pos_of(len, &st);
    const int64_t T = (int64_t)p.size();
    std::vector<uint8_t> cov((size_t)T, 0);
    int64_t tot = 0, last = -1;
    for (const auto& u : ru) {
        if (u.first < 0 || u.second <= 0 || u.first + u.second > T) return "run out of range";
        if (u.first <= last) return "runs overlap or unsorted";
        if (u.second < mr) return "run shorter than min";
        for (int64_t r = u.first; r < u.first + u.second; ++r) cov[(size_t)r] = 1;
        tot += u.second;
        last = u.first + u.second - 1;
    }
    if (tot != kept) return "kept count";
    if (tot >= T) return "nothing skipped";
    for (size_t i = 0; i < len.size(); ++i)
        for (int64_t r = st[i] + std::max<int64_t>(0, len[i] - keep); r < st[i] + len[i]; ++r)
            if (!cov[(size_t)r]) return "entry " + std::to_string(i) + " row " + std::to_string(r) + " not run";
    return "";
}
}  // namespace

TEST(pfctx_runs_keep_each_entrys_last_window_in_runs_of_160) {
    std::vector<std::pair<int64_t, int64_t>> ru;
    int64_t kept = 0;
    {   /* one 4096-row chunk: its last 129 rows, widened back to 160 */
        const auto p = pos_of({ 4096 });
        CHECK(ds41cone::pfctx_runs(p.data(), 4096, 1, 129, 160, &ru, &kept));
        REQUIRE_EQ(ru.size(), (size_t)1);
        CHECK_EQ(ru[0].first, (int64_t)3936);
        CHECK_EQ(ru[0].second, (int64_t)160);
        CHECK_EQ(kept, (int64_t)160);
    }
    {   /* decode rows ahead of a chunk and a short prefill after it: the decode run widened forward to 160 */
        const std::vector<int64_t> len = { 6, 6, 2048, 30 };
        const auto p = pos_of(len);
        CHECK(ds41cone::pfctx_runs(p.data(), (int64_t)p.size(), 4, 129, 160, &ru, &kept));
        CHECK_EQ(runs_ok(len, 129, 160, ru, kept), std::string());
        REQUIRE_EQ(ru.size(), (size_t)2);
        CHECK_EQ(ru[0].first, (int64_t)0);
        CHECK_EQ(ru[0].second, (int64_t)160);
        CHECK_EQ(ru[1].first + ru[1].second, (int64_t)2090);
    }
    {   /* every entry within the window: the pass as it is */
        const auto p = pos_of({ 6, 129, 100, 5 });
        CHECK(!ds41cone::pfctx_runs(p.data(), (int64_t)p.size(), 4, 129, 160, &ru, &kept));
        CHECK_EQ(kept, (int64_t)p.size());
    }
    {   /* two entries whose positions continue each other cannot be told apart: no plan */
        std::vector<int32_t> p;
        for (int32_t j = 0; j < 4000; ++j) p.push_back(j);   /* entry 0: 0..1999, entry 1: 2000..3999 */
        CHECK(!ds41cone::pfctx_runs(p.data(), 4000, 2, 129, 160, &ru, &kept));
        CHECK(!ds41cone::pfctx_runs(nullptr, 4000, 1, 129, 160, &ru, &kept));
    }
    {   /* a short pass, or a long entry whose cut would leave less than a run */
        const auto p = pos_of({ 150 });
        CHECK(!ds41cone::pfctx_runs(p.data(), 150, 1, 129, 160, &ru, &kept));
        const auto q = pos_of({ 200 });
        CHECK(ds41cone::pfctx_runs(q.data(), 200, 1, 129, 160, &ru, &kept));
        CHECK_EQ(kept, (int64_t)160);
    }
    /* random steps: decode entries (1 .. 6 rows) then prefill pieces (1 .. 4160 rows) */
    std::mt19937 rng(1112);
    int cut = 0;
    for (int it = 0; it < 4000; ++it) {
        std::vector<int64_t> len;
        const int nd = (int)(rng() % 9), np = 1 + (int)(rng() % 3);
        for (int i = 0; i < nd; ++i) len.push_back(1 + (int64_t)(rng() % 6));
        for (int i = 0; i < np; ++i) len.push_back(1 + (int64_t)(rng() % (it % 3 ? 4160 : 300)));
        const auto p = pos_of(len);
        const bool c = ds41cone::pfctx_runs(p.data(), (int64_t)p.size(), (int64_t)len.size(), 129, 160, &ru, &kept);
        if (c) {
            ++cut;
            const std::string r = runs_ok(len, 129, 160, ru, kept);
            if (!r.empty()) fprintf(stderr, "random %d: %s\n", it, r.c_str());
            CHECK(r.empty());
        } else {
            /* no plan: nothing could be skipped (every entry within the window) or a run would cover the pass */
            CHECK_EQ(kept, (int64_t)p.size());
        }
    }
    CHECK(cut > 1000);
}
RAD_TEST_MAIN()
