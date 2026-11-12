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
 *   cone replay       lever PSK-CR (ds41-pskcr): a second row model -- a prompt whose global KV the prefix cache holds up
 *                     to `at` (the deepest grid point under a hit at any H), started at the grid point at or below
 *                     at - C_0 with NO state (every window key before it wrong, the Engram token window too unless it
 *                     is 0), its chunks up to `at` cut at EVERY layer to at's cone with no layer-20 need (no store),
 *                     then the prompt's rest as design 2 (K4 / K8 / tail-only targets); the Engram layers 1 / 14 read a
 *                     row's last 4 tokens through the token window. Checked: at `at` every layer's window keys, and from
 *                     there every chunk's layer-20 inputs, every target's window state, the end's output and taps -- for
 *                     hits at every grid point / off-grid position of several prompts, random chunkings of the replay
 *                     and of the rest, and shared heads (the second / third prompt replay at the head's grid point, the
 *                     third's snapshot at the head is whole and a fourth resumed from it is right). Controls that must
 *                     fail: a replay cone without its window of slack, a replay starting one chunk short of C_0, and
 *                     no replay at all (the fresh state read at `at`).
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

namespace {

/* ------------------------------------------------------------------ lever PSK-CR: the cone replay's row model */
constexpr int64_t kGO = 2048, kGS = 4096;   /* the production grid: --checkpoint-interval 2048, --prefill-span 2 */
constexpr int kNgram = 4;                   /* the Engram's n-gram (its token window: ngram - 1 ids) */
int64_t gfloor(int64_t p) { return p < kGO ? 0 : kGO + (p - kGO) / kGS * kGS; }
/* the cold chunker's chunks of [a, e) on the grid (a on it): to the next grid point, then 4096 each */
std::vector<std::pair<int64_t, int64_t>> grid_chunks(int64_t a, int64_t e) {
    std::vector<std::pair<int64_t, int64_t>> c;
    for (int64_t s = a; s < e;) {
        const int64_t nx = s < kGO ? kGO : gfloor(s) + kGS;
        const int64_t t = std::min(e, nx);
        c.push_back({ s, t });
        s = t;
    }
    return c;
}
/* the chunks with a boundary at `p` too (a request's checkpoint_at splits the chunk it falls in, plan_chunk) */
std::vector<std::pair<int64_t, int64_t>> split_at(std::vector<std::pair<int64_t, int64_t>> c, int64_t p) {
    for (size_t i = 0; i < c.size(); ++i)
        if (c[i].first < p && p < c[i].second) {
            const int64_t e = c[i].second;
            c[i].second = p;
            c.insert(c.begin() + (long)i + 1, { p, e });
            break;
        }
    return c;
}
std::vector<std::pair<int64_t, int64_t>> random_chunks(int64_t a, int64_t e, std::mt19937& rng, int64_t lo = 16) {
    std::vector<std::pair<int64_t, int64_t>> c;
    for (int64_t s = a; s < e;) {
        const int64_t t = std::min(e, s + lo + (int64_t)(rng() % (uint32_t)(4097 - lo)));
        c.push_back({ s, t });
        s = t;
    }
    return c;
}
/* a request's targets past `at` up to N: the end, the tail snapshot (the grid point under N - 16), every K-th grid point
 * (K 0: tail-only), checkpoint_at (> 0) */
std::vector<int64_t> cr_targets(int64_t at, int64_t N, int K, int64_t ckpt_at = 0) {
    std::vector<int64_t> t{ N };
    const int64_t tail = gfloor(N - 16);
    if (tail > at && tail >= kGO) t.push_back(tail);
    for (int64_t i = 1; K > 0; ++i) {
        const int64_t q = kGO + (i - 1) * kGS;
        if (q >= N) break;
        if (q > at && i % K == 0) t.push_back(q);
    }
    if (ckpt_at > at && ckpt_at < N) t.push_back(ckpt_at);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    return t;
}

struct CrRun {
    int64_t N = 0, at = 0, from = 0;
    std::vector<std::pair<int64_t, int64_t>> replay, rest;   /* chunks: [from, at) and [at, N) */
    std::vector<int64_t> tg;                                /* the rest's targets (N among them) */
    bool l20 = false;
    int64_t cone_sub = 0;       /* a control: every replay cone this many rows short */
    bool restored = false;      /* resumed from a snapshot at `from` (= at): the window and token window whole there */
};
/* Returns "" or what broke. A row's Engram ids (layers 1 and 14, in front of the block) are right iff its ngram - 1
 * predecessors are known: from 0 (the eos window, as cold), or inside rows the sequence ran since its start, or a
 * restored token window. */
std::string cr_model(const CrRun& R) {
    const int64_t N = R.N;
    std::vector<std::vector<uint8_t>> key((size_t)kNL, std::vector<uint8_t>((size_t)N, 0));
    std::vector<uint8_t> in((size_t)N), out((size_t)N);
    std::vector<int32_t> pre((size_t)N + 1);
    std::vector<std::vector<uint8_t>> tap(3, std::vector<uint8_t>((size_t)N, 0));
    if (R.restored)
        for (int l = 0; l < kNL; ++l)
            for (int64_t q = std::max<int64_t>(0, R.from - kW + 1); q < R.from; ++q) key[(size_t)l][(size_t)q] = 1;
    const auto ids_ok = [&](int64_t p) { return R.from == 0 || R.restored || p - (kNgram - 1) >= R.from; };
    const int cut0 = R.l20 ? kL20 : kFirst;
    std::vector<std::pair<std::pair<int64_t, int64_t>, bool>> all;
    for (const auto& c : R.replay) all.push_back({ c, true });
    for (const auto& c : R.rest) all.push_back({ c, false });
    int64_t prev = R.from;
    for (const auto& ce : all) {
        const int64_t s = ce.first.first, e = ce.first.second, n = e - s;
        const bool rp = ce.second;
        if (s != prev) return "chunks not contiguous at " + std::to_string(s);
        prev = e;
        if (rp && e > R.at) return "a replay chunk crosses at";
        int32_t ahead = 0;
        if (rp) {
            ahead = (int32_t)(R.at - e);
        } else {
            int64_t T1 = -1;
            for (int64_t t : R.tg)
                if (t > s) { T1 = t; break; }
            if (T1 < e) return "target list does not cover chunk " + std::to_string(s);
            ahead = (int32_t)(T1 - e);
        }
        for (int64_t p = s; p < e; ++p) in[(size_t)p] = 1;   /* the embedding: right */
        for (int l = 0; l < kNL; ++l) {
            int64_t a = s;   /* computed rows [a, e) */
            if (rp || l >= cut0) {
                const int64_t C = ds41cone::cone(kNL, kW, l) - (rp ? R.cone_sub : 0);
                a = e - ds41cone::rows(C, ahead, n);
            }
            if (!rp && l == 20)   /* the latent / index keys this chunk stores: every row's layer-20 input */
                for (int64_t p = s; p < e; ++p)
                    if (!in[(size_t)p]) return "layer-20 input wrong at " + std::to_string(p);
            if (l == 1 || l == 14)   /* the Engram block: the row's own ids */
                for (int64_t p = s; p < e; ++p) in[(size_t)p] = in[(size_t)p] && ids_ok(p);
            if (l >= 37)
                for (int64_t p = s; p < e; ++p) tap[(size_t)(l - 37)][(size_t)p] = a < e && in[(size_t)p];
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
        const bool tgt = (rp && e == R.at) || (!rp && std::binary_search(R.tg.begin(), R.tg.end(), e));
        if (tgt) {
            for (int l = 0; l < kNL; ++l)
                for (int64_t q = std::max<int64_t>(0, e - kW + 1); q < e; ++q)
                    if (!key[(size_t)l][(size_t)q])
                        return std::string(rp ? "replay end " : "target ") + std::to_string(e) + ": layer " +
                               std::to_string(l) + " key " + std::to_string(q) + " wrong";
            if (rp)   /* the DSpark taps of the replay end's last window + 1 rows come out too */
                for (int64_t p = std::max<int64_t>(0, e - kW - 1); p < e; ++p)
                    for (int k = 0; k < 3; ++k)
                        if (!tap[(size_t)k][(size_t)p]) return "replay end tap " + std::to_string(37 + k) + " wrong";
        }
        if (e == N) {
            if (!in[(size_t)N - 1]) return "the last row's output wrong";
            for (int64_t p = std::max<int64_t>(0, N - kW - 1); p < N; ++p)
                for (int k = 0; k < 3; ++k)
                    if (!tap[(size_t)k][(size_t)p]) return "tap " + std::to_string(37 + k) + " wrong at " + std::to_string(p);
        }
    }
    if (prev != N) return "chunks end at " + std::to_string(prev);
    return "";
}
/* The scheduler's replay for a hit of `H` tokens on a prompt of N (Scheduler::psk_cr_resume): at = the grid point under
 * min(H, N - 1), from = the grid point under at - C_0; {at <= 0}: none. */
CrRun cr_plan(int64_t N, int64_t H, int K, bool l20) {
    CrRun R;
    R.N = N;
    R.at = gfloor(std::min(H, N - 1));
    R.from = gfloor(std::max<int64_t>(0, R.at - ds41cone::cone(kNL, kW, 0)));
    R.replay = grid_chunks(R.from, R.at);
    R.rest = grid_chunks(R.at, N);
    R.tg = cr_targets(R.at, N, K);
    R.l20 = l20;
    return R;
}

}  // namespace

TEST(cone_replay_restores_every_window_at_its_end) {
    CHECK_EQ(ds41cone::cone(kNL, kW, 0), (int64_t)5152);
    int checked = 0;
    /* hits at every grid point and between them, K4 / K8 / tail-only, PSK-L20 off and on */
    for (int64_t N : { (int64_t)136216, (int64_t)64000, (int64_t)25600 + 3000, (int64_t)9000, (int64_t)6200 })
        for (int K : { 4, 8, 0 })
            for (bool l20 : { false, true })
                for (int64_t H = 2048; H < N; H += (N > 70000 ? 2048 * 5 + 77 : 1024 + 31)) {
                    const CrRun R = cr_plan(N, H, K, l20);
                    if (R.at <= 0) continue;
                    const std::string r = cr_model(R);
                    if (!r.empty())
                        fprintf(stderr, "N %lld H %lld at %lld from %lld K %d l20 %d: %s\n", (long long)N, (long long)H,
                                (long long)R.at, (long long)R.from, K, (int)l20, r.c_str());
                    CHECK(r.empty());
                    ++checked;
                }
    /* random chunkings of the replay and of the rest (pieces of 16 .. 4096 rows, as mixed steps cut them) */
    std::mt19937 rng(20261112 + 3);
    for (int it = 0; it < 60; ++it) {
        const int64_t N = 7000 + (int64_t)(rng() % 60000), H = 2048 + (int64_t)(rng() % (uint32_t)(N - 2048));
        CrRun R = cr_plan(N, H, (int)(it % 3 == 0 ? 0 : it % 3 == 1 ? 4 : 8), it % 2 == 1);
        if (R.at <= 0) continue;
        if (it % 4 != 3) R.replay = random_chunks(R.from, R.at, rng, it % 4 == 2 ? 1 : 16);
        if (it % 4 != 0) {   /* the rest in random pieces: targets at the end, or at every piece's end */
            R.rest = random_chunks(R.at, N, rng);
            R.tg.clear();
            if (it % 2) for (const auto& c : R.rest) R.tg.push_back(c.second);
            else R.tg.push_back(N);
        }
        const std::string r = cr_model(R);
        if (!r.empty()) fprintf(stderr, "random %d N %lld H %lld: %s\n", it, (long long)N, (long long)H, r.c_str());
        CHECK(r.empty());
        ++checked;
    }
    CHECK(checked > 150);
}

TEST(cone_replay_serves_a_shared_head_and_its_snapshot) {
    /* prompts B and C share A's head of h tokens, then diverge: each hits at h, replays at the grid point under it and
     * goes on; C takes the head's snapshot (checkpoint_at = branch_at = h, a target) -- its window state at h is whole,
     * so D, resumed from that snapshot (no replay), is right too */
    for (int64_t h : { (int64_t)25600, (int64_t)14336, (int64_t)30000, (int64_t)6144, (int64_t)2100 })
        for (int K : { 4, 8, 0 })
            for (bool l20 : { false, true }) {
                const int64_t N = h + 9000;
                CrRun B = cr_plan(N, h, K, l20);
                CHECK(cr_model(B).empty());
                CrRun C = B;
                C.tg = cr_targets(C.at, N, K, h);
                C.rest = split_at(C.rest, h);
                CHECK(std::binary_search(C.tg.begin(), C.tg.end(), h) || h == C.at);
                const std::string rc = cr_model(C);
                if (!rc.empty()) fprintf(stderr, "head %lld K %d: C %s\n", (long long)h, K, rc.c_str());
                CHECK(rc.empty());
                CrRun D;
                D.N = N + 500;
                D.at = D.from = h;
                D.restored = true;
                D.rest = grid_chunks(h, D.N);
                D.tg = cr_targets(h, D.N, K);
                D.l20 = l20;
                const std::string rd = cr_model(D);
                if (!rd.empty()) fprintf(stderr, "head %lld K %d: D %s\n", (long long)h, K, rd.c_str());
                CHECK(rd.empty());
            }
}

TEST(cone_replay_controls_fail) {
    /* each control breaks one premise of the replay, and the model must say so -- or it proves nothing */
    const int64_t N = 40000, H = 30000;
    CrRun R = cr_plan(N, H, 4, true);
    CHECK_EQ(R.at, (int64_t)26624);
    CHECK_EQ(R.from, (int64_t)18432);
    CHECK(cr_model(R).empty());
    {   /* a cone without its window of slack */
        CrRun c = R;
        c.cone_sub = kW;
        CHECK(!cr_model(c).empty());
    }
    {   /* a replay one chunk short (from at - 4096 < at - C_0: layer 0's cone not reached) */
        CrRun c = R;
        c.from = R.at - kGS;
        c.replay = grid_chunks(c.from, c.at);
        CHECK(!cr_model(c).empty());
    }
    {   /* no replay: the fresh state read at `at` */
        CrRun c = R;
        c.from = c.at;
        c.replay.clear();
        CHECK(!cr_model(c).empty());
    }
    {   /* a replay from inside C_0 (off the grid): C_0 itself holds; the last start that holds lies within the cones'
         * slack (C_l - C_(l+1) = window, a row reads window - 1 back; the DSpark taps' 129 rows bind at 70), and every start
         * past it fails */
        CrRun c = R;
        const int64_t C0 = ds41cone::cone(kNL, kW, 0);
        int64_t last_ok = -1;
        for (int64_t d = 0; d <= 256; d += 1) {
            c.from = R.at - C0 + d;
            c.replay = { { c.from, R.at - kGS }, { R.at - kGS, R.at } };
            if (cr_model(c).empty()) last_ok = d;
        }
        CHECK_EQ(last_ok >= 0, true);
        CHECK(last_ok < kW);   /* 70: the taps' 129 rows at layers 37..39 bind first */
        fprintf(stderr, "cone replay: starts up to %lld rows inside C_0 hold, none past them\n", (long long)last_ok);
        for (int64_t d = last_ok + 1; d <= 256; d += 17) {
            c.from = R.at - C0 + d;
            c.replay = { { c.from, R.at - kGS }, { R.at - kGS, R.at } };
            CHECK(!cr_model(c).empty());
        }
    }
}

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

/* lever PSK-CR: a replay entry is cut from layer 0 -- its first chunk (the target one chunk past its end) runs layers
 * 0..8 only (C_8 = 4128 > 4096 >= C_9), its last every layer; beside a planned entry cut from 21 the pass runs at least
 * through 20, and an entry cut from 0 with no hint resolved runs every layer */
TEST(cone_plan_end_from_a_replay) {
    const int32_t rw[2] = { 4096, 4096 };
    const int32_t from0[2] = { 0, 0 }, mixed[2] = { 0, kFirst };
    struct C { int32_t ah; int want; };
    for (const C c : { C{ 4096, 9 }, C{ 0, kNL }, C{ 1000, 33 }, C{ 5152, 0 }, C{ 5100, 1 } }) {
        const int32_t ah[1] = { c.ah };
        CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 1, ah, rw, from0), c.want);
    }
    const int32_t ah2[2] = { 4096, 9000 };
    CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 2, ah2, rw, mixed), kFirst);
    const int32_t ah3[2] = { 4096, 2500 };
    CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 2, ah3, rw, mixed), 21);
    const int32_t ah4[2] = { 4096, 1000 };
    CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 2, ah4, rw, mixed), 33);
    /* the same as plan_end where every entry starts at one layer */
    for (int32_t a = 0; a < 6000; a += 97) {
        const int32_t ah[2] = { a, a / 2 }, f21[2] = { kFirst, kFirst }, f20[2] = { kL20, kL20 };
        CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 2, ah, rw, f21), ds41cone::plan_end(kNL, kW, kFirst, 2, ah, rw));
        CHECK_EQ(ds41cone::plan_end_from(kNL, kW, 2, ah, rw, f20), ds41cone::plan_end(kNL, kW, kL20, 2, ah, rw));
    }
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
