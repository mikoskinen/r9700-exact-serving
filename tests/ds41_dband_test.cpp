/* tests/ds41_dband_test.cpp -- tp8 loop FINDINGS F58: a decode segment above the decode band (kBiBand = 48 rows) in
 * deepseek41's step_bi, issued in band pieces (arch/deepseek41/ds41_dec_band.h) under RADIANCE_DS41_BI_DBAND=1; host
 * arithmetic only.
 *
 * libr4d's attn_sparse_sink and index_score take their scalar decode kernels (libref's bits) up to 48 rows a launch and
 * WMMA forms above (other bits), so a decode row keeps its alone bytes only in a launch of <= 48 rows. The attention
 * finds a row's sequence in the launch's cu_seqlens (libref's seq_of_row: cu[s] <= row < cu[s+1], rows from the
 * launch's first), so a piece is whole sequences [s0, s1) at rows [r0, r0 + n) read through cu[s0 .. s1] - cu[s0].
 * Checked over decode layouts of 49 .. ~380 rows (uniform drafting depths and random ones, 1 .. 6 rows a sequence):
 * the pieces cover the segment's rows once, in order, each whole sequences of <= 48 rows (<= 49 cu words: bi_dcu's
 * row), as few as greedy packing gives; every row maps to its own sequence through its piece's relative cu; the
 * indexer's runs at the band (ds41_idx_runs at 48 rows) are <= 48 rows and cover the segment. The sequences' first rows
 * from the host positions, and the ambiguous case (two sequences whose positions continue) refused. Controls that must
 * fail: today's issue (one attention launch and one indexer run over the segment) leaves the band, and a later piece
 * read through the batch's absolute cu maps its rows to the wrong sequences.
 */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_dec_band.h"
#include "../arch/deepseek41/ds41_index_split.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kBand = 48;

/* libref's seq_of_row over cu[0 .. S]: the s with cu[s] <= row < cu[s + 1] (bisection as the kernels do) */
int seq_of_row(const std::vector<int64_t>& cu, int64_t row) {
    int lo = 0, hi = (int)cu.size() - 2;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu[(size_t)mid] <= row) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* A decode layout: rows a sequence -> the sequences' first rows (st) and cu (dn + 1 words) */
struct Layout {
    std::vector<int64_t> st, cu;
    int64_t dt = 0;
};
Layout layout(const std::vector<int64_t>& rows) {
    Layout L;
    L.cu.push_back(0);
    for (int64_t r : rows) {
        L.st.push_back(L.dt);
        L.dt += r;
        L.cu.push_back(L.dt);
    }
    return L;
}

/* "" when the pieces are the band issue of the layout, else what broke */
std::string check(const Layout& L, const std::vector<Ds41DecPiece>& p) {
    if (p.empty()) return "no pieces";
    int64_t r = 0, s = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        const Ds41DecPiece& q = p[j];
        if (q.r0 != r || q.s0 != s) return "piece " + std::to_string(j) + " does not continue the last";
        if (q.n < 1 || q.n > kBand) return "piece " + std::to_string(j) + " of " + std::to_string(q.n) + " rows";
        if (q.s1 <= q.s0 || q.s1 - q.s0 + 1 > kBand + 1) return "piece " + std::to_string(j) + " sequences";
        if (L.cu[(size_t)q.s0] != q.r0 || L.cu[(size_t)q.s1] != q.r0 + q.n) return "piece " + std::to_string(j) + " cuts a sequence";
        /* the piece's cu as bi_dcu row j holds it (piece 0: the batch's own, cu[0] = 0) */
        std::vector<int64_t> rc;
        for (int64_t k = q.s0; k <= q.s1; ++k) rc.push_back(L.cu[(size_t)k] - L.cu[(size_t)q.s0]);
        for (int64_t m = 0; m < q.n; ++m)
            if (seq_of_row(rc, m) + q.s0 != seq_of_row(L.cu, q.r0 + m))
                return "piece " + std::to_string(j) + " row " + std::to_string(m) + " maps to another sequence";
        r += q.n;
        s = q.s1;
    }
    if (r != L.dt || s != (int64_t)L.st.size()) return "the pieces do not cover the segment";
    return "";
}

}  // namespace

TEST(uniform_depth3_lane_of_16_is_two_even_pieces) {
    const Layout L = layout(std::vector<int64_t>(16, 4));   /* 16 x (1 + 3) = 64 rows: lq150's x4d3 step */
    std::vector<Ds41DecPiece> p;
    CHECK_EQ(ds41_dec_pieces(L.st, L.dt, kBand, p), 0);
    CHECK_EQ(p.size(), (size_t)2);
    CHECK_EQ(p[0].n, 32);
    CHECK_EQ(p[1].n, 32);
    CHECK_EQ(p[1].s0, 8);
    CHECK_EQ(check(L, p), std::string(""));
    /* 15 x 4 = 60: 32 + 28 (greedy at 48: 48 + 12) */
    const Layout L2 = layout(std::vector<int64_t>(15, 4));
    CHECK_EQ(ds41_dec_pieces(L2.st, L2.dt, kBand, p), 0);
    CHECK_EQ(p.size(), (size_t)2);
    CHECK_EQ(p[0].n, 32);
    CHECK_EQ(p[1].n, 28);
    CHECK_EQ(check(L2, p), std::string(""));
}

TEST(a_segment_in_the_band_takes_no_pieces) {
    std::vector<Ds41DecPiece> p;
    for (int64_t q = 1; q <= 6; ++q)
        for (int64_t dn = 1; dn * q <= kBand; ++dn) {
            const Layout L = layout(std::vector<int64_t>((size_t)dn, q));
            CHECK_EQ(ds41_dec_pieces(L.st, L.dt, kBand, p), 0);
            CHECK(p.empty());   /* today's one launch: the decode band already */
        }
}

TEST(random_layouts_every_row_in_the_band_at_its_own_sequence) {
    std::mt19937_64 rng(0xF58);
    int layouts = 0;
    std::vector<Ds41DecPiece> p, g;
    for (int it = 0; it < 4000; ++it) {
        const int64_t dn = 9 + (int64_t)(rng() % 56);   /* 9 .. 64 sequences */
        std::vector<int64_t> rows((size_t)dn);
        const bool uni = rng() % 3 == 0;
        const int64_t q = 1 + (int64_t)(rng() % 6);
        for (auto& r : rows) r = uni ? q : 1 + (int64_t)(rng() % 6);
        const Layout L = layout(rows);
        if (L.dt <= kBand) continue;
        ++layouts;
        CHECK_EQ(ds41_dec_pieces(L.st, L.dt, kBand, p), 0);
        CHECK_EQ(check(L, p), std::string(""));
        ds41_dec_greedy(L.st, L.dt, kBand, g);
        CHECK_EQ(p.size(), g.size());   /* as few as greedy packing gives */
        CHECK(p.size() >= (size_t)((L.dt + kBand - 1) / kBand));
    }
    CHECK(layouts > 3000);
}

TEST(starts_from_host_positions_and_the_ambiguous_case) {
    /* 16 sequences x 4 rows at their own context positions */
    std::vector<int32_t> pos;
    std::vector<int64_t> want, st;
    for (int s = 0; s < 16; ++s) {
        want.push_back((int64_t)pos.size());
        for (int k = 0; k < 4; ++k) pos.push_back(1000 * (s + 1) + 37 * s + k);
    }
    CHECK(ds41_dec_starts(pos.data(), (int64_t)pos.size(), 16, st));
    CHECK(st == want);
    /* sequence 5 starts where 4 ends + 1: the jumps give 15 starts -> refused (the caller reads cu back) */
    for (int k = 0; k < 4; ++k) pos[(size_t)(5 * 4 + k)] = pos[(size_t)(4 * 4 + 3)] + 1 + k;
    CHECK(!ds41_dec_starts(pos.data(), (int64_t)pos.size(), 16, st));
    CHECK(st.empty());
    CHECK(!ds41_dec_starts(nullptr, 64, 16, st));
}

TEST(indexer_runs_at_the_band_stay_in_the_decode_form) {
    for (int64_t T = kBand + 1; T <= 384; ++T) {
        const Ds41IdxRuns r = ds41_idx_runs(T, kBand, 1 << 20, 4, 1, false);
        CHECK(!r.split);
        CHECK(r.R <= kDs41IdxFormRows);
        CHECK_EQ(r.n, (T + kBand - 1) / kBand);
        CHECK(r.n * r.R >= T && (r.n - 1) * r.R < T);   /* run p: rows [p R, p R + R), the last may be shorter */
        CHECK_EQ(r.p0, 0);
        CHECK_EQ(r.p1, r.n);
    }
}

TEST(dense_band_cuts_cover_the_run_within_the_band) {
    /* tp8 loop D22: a decode-only dense run in launches of <= band rows; band 48 is today's issue, 64 the WMK dense band */
    std::vector<std::pair<int64_t, int64_t>> c;
    CHECK_EQ(ds41_band_cuts(64, 48, c), 2);
    CHECK(c[0] == std::make_pair((int64_t)0, (int64_t)48) && c[1] == std::make_pair((int64_t)48, (int64_t)16));
    CHECK_EQ(ds41_band_cuts(64, 64, c), 1);
    CHECK(c[0] == std::make_pair((int64_t)0, (int64_t)64));
    CHECK_EQ(ds41_band_cuts(48, 64, c), 1);
    CHECK_EQ(ds41_band_cuts(0, 64, c), 0);
    for (int64_t band = 48; band <= 64; ++band)
        for (int64_t n = 1; n <= 300; ++n) {
            ds41_band_cuts(n, band, c);
            int64_t r = 0;
            bool ok = true;
            for (const auto& p : c) {
                ok = ok && p.first == r && p.second >= 1 && p.second <= band;
                r += p.second;
            }
            CHECK(ok && r == n);
            CHECK_EQ((int64_t)c.size(), (n + band - 1) / band);
        }
}

TEST(dband2_key_tells_every_piece_layout_apart) {
    /* RADIANCE_DS41_BI_DBAND=2: uneven layouts recorded by their pieces. Over every layout of 16 sequences of 3 or 4 rows
     * (2^16): the same pieces give the same key, different pieces different keys, never 0 */
    std::vector<std::pair<uint64_t, std::vector<int64_t>>> seen;
    std::vector<Ds41DecPiece> p;
    int layouts = 0;
    for (uint32_t bits = 0; bits < (1u << 16); ++bits) {
        std::vector<int64_t> rows;
        for (int s = 0; s < 16; ++s) rows.push_back(bits >> s & 1 ? 4 : 3);
        const Layout L = layout(rows);
        if (L.dt <= kBand) continue;
        ++layouts;
        ds41_dec_pieces(L.st, L.dt, kBand, p);
        std::vector<int64_t> sig{ L.dt };
        for (const Ds41DecPiece& q : p) { sig.push_back(q.r0); sig.push_back(q.s0); }
        const uint64_t k = ds41_dec_pieces_key(p);
        CHECK(k != 0);
        bool dup = false;
        for (const auto& e : seen)
            if (e.second == sig) { dup = true; CHECK(e.first == k); break; }
        if (!dup) seen.push_back({ k, sig });
    }
    CHECK(layouts > 60000);
    /* distinct piece layouts at the same row count never share a key */
    int clash = 0;
    for (size_t i = 0; i < seen.size(); ++i)
        for (size_t j = i + 1; j < seen.size(); ++j)
            clash += seen[i].second[0] == seen[j].second[0] && seen[i].first == seen[j].first;
    CHECK_EQ(clash, 0);
    CHECK(seen.size() >= 10);
}

TEST(control_todays_issue_leaves_the_band_and_absolute_cu_misplaces_rows) {
    /* today: the attention once a segment, the indexer at irows (>= 96) -- one launch of 64 rows each */
    const Layout L = layout(std::vector<int64_t>(16, 4));
    const std::vector<Ds41DecPiece> one{ { 0, L.dt, 0, 16 } };
    CHECK(!check(L, one).empty());
    const Ds41IdxRuns r = ds41_idx_runs(L.dt, 256, 1 << 20, 4, 1, false);
    CHECK(r.R > kDs41IdxFormRows);
    /* piece 1 read through the batch's cu[8 .. 16] as is (absolute rows 32 .. 64): its first row maps to sequence 8 only
     * by accident of the bisection's floor, its last rows to sequence 8 too -- not their own */
    std::vector<Ds41DecPiece> p;
    ds41_dec_pieces(L.st, L.dt, kBand, p);
    CHECK_EQ(p.size(), (size_t)2);
    std::vector<int64_t> abs_cu(L.cu.begin() + p[1].s0, L.cu.begin() + p[1].s1 + 1);
    int wrong = 0;
    for (int64_t m = 0; m < p[1].n; ++m) wrong += seq_of_row(abs_cu, m) + p[1].s0 != seq_of_row(L.cu, p[1].r0 + m);
    CHECK(wrong >= 28);
}

TEST(d22_the_dense_band_follows_ds41_wmk_and_a_64_row_run_is_one_launch) {
    /* tp8 loop D22 (DECODE round 18): off WMK the decode band whatever the env; ds41_wmk 1 .. 4 the env's band; ds41_wmk 5
     * (libr4d's decode band to 64 rows) 64 whatever the env */
    for (int64_t env : { (int64_t)48, (int64_t)56, (int64_t)64 }) {
        CHECK_EQ(ds41_dense_band(0, env, kBand), kBand);
        CHECK_EQ(ds41_dense_band(-1, env, kBand), kBand);
        for (int w = 1; w <= 4; ++w) CHECK_EQ(ds41_dense_band(w, env, kBand), env);
        CHECK_EQ(ds41_dense_band(5, env, kBand), (int64_t)64);
        /* tp8 loop D25 step 2 (DECODE round 21): ds41_wmk 6 / 7 (5 + another 49..64-row launch code in libr4d) keep 5's band */
        CHECK_EQ(ds41_dense_band(6, env, kBand), (int64_t)64);
        CHECK_EQ(ds41_dense_band(7, env, kBand), (int64_t)64);
        CHECK_EQ(ds41_dense_band(8, env, kBand), (int64_t)64);   /* mode 8: 5 + the per-shape 49..64-row table */
        CHECK_EQ(ds41_dense_band(9, env, kBand), (int64_t)64);   /* mode 9 (D27): 8 + the per-shape 33..48-row table */
        CHECK_EQ(ds41_dense_band(10, env, kBand), (int64_t)64);  /* mode 10 (tp8 loop D34): 9 without its two q_b entries */
    }
    /* at the band of 64 every decode-only run of 1 .. 64 rows is ONE launch over all its rows; at 48 a run of 49 .. 64 is
     * 48 + the rest (today's two weight reads); past 64 the 64 band cuts at 64 */
    std::vector<std::pair<int64_t, int64_t>> c;
    for (int64_t n = 1; n <= 64; ++n) {
        CHECK_EQ(ds41_band_cuts(n, ds41_dense_band(5, kBand, kBand), c), (int64_t)1);
        CHECK(c[0].first == 0 && c[0].second == n);
        for (int w = 6; w <= 10; ++w) {
            CHECK_EQ(ds41_band_cuts(n, ds41_dense_band(w, kBand, kBand), c), (int64_t)1);
            CHECK(c[0].first == 0 && c[0].second == n);
        }
        CHECK_EQ(ds41_band_cuts(n, ds41_dense_band(2, kBand, kBand), c), n > kBand ? (int64_t)2 : (int64_t)1);
        if (n > kBand) CHECK(c[0].second == kBand && c[1].first == kBand && c[1].second == n - kBand);
    }
    CHECK_EQ(ds41_band_cuts(100, 64, c), (int64_t)2);
    CHECK(c[1].first == 64 && c[1].second == 36);
}

RAD_TEST_MAIN()
