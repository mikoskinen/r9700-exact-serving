/* tests/ds41_engdd_test.cpp -- lever E1 (ds41-e1, fn4 ds41_engdd): the Engram kv dedup's plan and its scatter, on the host.
 *
 * arch/deepseek41/ds41_engram_dedup.h plans one Engram layer of a step: the launches of more than 48 rows pool their rows,
 * the pool's distinct id rows are looked up, copied and multiplied once, and gather_rows writes every pooled row's kv row
 * from its unique row; launches of <= 48 rows keep their own GEMM. Here the card's per-row work (lookup + copy + GEMM
 * above the band: a function of the row's ids alone) is a host stand-in -- a bf16 table row an id, a fixed-order f32 dot
 * product a column rounded to bf16 -- and ds41_engdd::emulate runs the deduplicated pass as deepseek41 issues it (compact
 * ids, the kept launches' GEMMs into their step rows, the unique rows' GEMM into the scratch plane, each pooled run
 * gathered through the map). Checked:
 *   - the plan: every pooled row maps to a unique row with the same H ids (byte for byte), the unique rows are pairwise
 *     different and in first-occurrence order, every row is covered once (kept launches + pooled runs), kept launches
 *     are the <= 48-row ones, a deduplicated pool has > 48 unique rows;
 *   - the pass: the deduplicated kv rows equal the undeduplicated ones byte for byte -- one launch over the step (step()),
 *     BI's launches (decode pieces of <= 48, a short prefill segment, wide prefill segments, several pooled runs), heavy
 *     hash-slot collisions, rows equal but for one id;
 *   - the guards: <= 48 unique rows in a pool of wider launches -> kGuard (48 included, 49 deduplicates), too few
 *     repeats -> kLowShare (the 1/16 boundary both sides), no launch above the band -> kNone; nothing planned then;
 *   - MUTATIONS: the same check fails for a scatter that is wrong -- a map entry pointing at another unique row, a gather
 *     run shifted by one row, a compact row taking another step row's ids, a kept launch written one row off -- so a
 *     broken scatter cannot pass this test;
 *   - tp8 loop P48b (ds41_engdd 2, the dedup on the Engram's int8 twin): with a GEMM whose row bytes depend on the launch's
 *     rows up to 64 (the twin's narrow band: a stand-in that changes a row's low bits by M there) the plan at floor 64 keeps
 *     every launch of <= 64 rows, pools only wider ones, refuses a pool of <= 64 unique rows, and the pass equals the
 *     by-launch pass byte for byte; at today's floor 48 the same launches differ (a 49..64-row launch pooled, a pool of
 *     49..64 unique rows run in the narrow band) -- the negative controls; pool_twin: all / none / split. */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_engram_dedup.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace ds41_engdd;

constexpr int64_t kDim = 8;     /* table row width (the 1017: 256) */
constexpr int64_t kW = 48;      /* kv columns (the 1017: 25600) */

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even */
    return (uint16_t)(u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

/* The card's work on one row, a function of its ids alone: the table rows of its H ids (bf16), then y[n] = bf16(sum_k
 * x[k] w[n][k]) in a fixed order. */
struct RowFn {
    int64_t H;
    std::vector<float> w;   /* [kW, H kDim] */
    explicit RowFn(int64_t h) : H(h), w((size_t)(kW * h * kDim)) {
        for (size_t i = 0; i < w.size(); ++i) w[i] = (float)((int64_t)(mix(i + 77) % 2001) - 1000) / 1000.0f;
    }
    void operator()(const int32_t* ids, uint16_t* out) const {
        std::vector<uint16_t> x((size_t)(H * kDim));
        for (int64_t h = 0; h < H; ++h)
            for (int64_t d = 0; d < kDim; ++d)
                x[(size_t)(h * kDim + d)] =
                    to_bf16((float)((int64_t)(mix(((uint64_t)(uint32_t)ids[h] << 8) ^ (uint64_t)d) % 4001) - 2000) / 500.0f);
        for (int64_t n = 0; n < kW; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < H * kDim; ++k) acc += from_bf16(x[(size_t)k]) * w[(size_t)(n * H * kDim + k)];
            out[n] = to_bf16(acc);
        }
    }
};

/* [T, H] ids: `distinct` distinct rows, each row drawn from them (a long-tailed draw: a few rows repeat often). */
std::vector<int32_t> make_ids(int64_t T, int64_t H, int64_t distinct, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::vector<int32_t> base((size_t)(distinct * H));
    for (auto& v : base) v = (int32_t)(g() % 99092);
    std::vector<int32_t> ids((size_t)(T * H));
    for (int64_t r = 0; r < T; ++r) {
        const uint64_t u = g();
        const uint64_t hot = (uint64_t)(distinct < 10 ? distinct : 10);
        const int64_t k = (int64_t)(u % 5 == 0 ? (u >> 3) % hot : (u >> 3) % (uint64_t)distinct);
        std::memcpy(ids.data() + r * H, base.data() + k * H, (size_t)H * 4);
    }
    return ids;
}

/* The undeduplicated pass: every row's kv row from its own ids. */
std::vector<uint16_t> reference(const std::vector<int32_t>& ids, int64_t T, const RowFn& f) {
    std::vector<uint16_t> kv((size_t)(T * kW), 0);
    for (int64_t r = 0; r < T; ++r) f(ids.data() + r * f.H, kv.data() + r * kW);
    return kv;
}

/* The deduplicated pass of plan `p` (ds41_engdd::emulate) against the reference: the kv rows equal byte for byte. */
bool same_pass(const Plan& p, const std::vector<int32_t>& ids, const RowFn& f, const std::vector<uint16_t>& ref) {
    std::vector<uint16_t> kv((size_t)(p.T * kW), 0xdead);
    std::vector<int32_t> cids;
    std::vector<uint16_t> ku;
    emulate(p, ids.data(), kW, [&](const int32_t* row, uint16_t* out) { f(row, out); }, kv.data(), cids, ku);
    return std::memcmp(kv.data(), ref.data(), kv.size() * 2) == 0;
}

/* The plan's structure: every pooled row maps to a unique row with its own ids, unique rows distinct and in first-
 * occurrence order, kept launches <= 48 rows, every row covered once, > 48 unique rows. */
bool plan_ok(const Plan& p, const std::vector<int32_t>& ids, const std::vector<Launch>& ls) {
    const int64_t H = p.H;
    if (p.out != kDedup || p.uniq <= kFormRows || p.rows != p.u0 + p.uniq || (int64_t)p.src.size() != p.rows) return false;
    std::vector<int> cover((size_t)p.T, 0);
    for (const Kept& k : p.kept) {
        if (k.n > kFormRows) return false;
        for (int64_t i = 0; i < k.n; ++i) {
            ++cover[(size_t)(k.r0 + i)];
            if (p.src[(size_t)(k.c0 + i)] != k.r0 + i || p.map[(size_t)(k.r0 + i)] != -1) return false;
        }
    }
    int64_t pooled = 0;
    for (const auto& u : p.runs)
        for (int64_t r = u.first; r < u.first + u.second; ++r) {
            ++cover[(size_t)r];
            ++pooled;
            const int32_t s = p.map[(size_t)r];
            if (s < 0 || s >= p.uniq) return false;
            const int32_t fr = p.src[(size_t)(p.u0 + s)];
            if (fr > r || std::memcmp(ids.data() + (int64_t)fr * H, ids.data() + r * H, (size_t)H * 4) != 0) return false;
        }
    for (const Launch& l : ls)
        for (int64_t r = l.r0; r < l.r0 + l.n; ++r)
            if (cover[(size_t)r] != 1) return false;
    for (int64_t a = 0; a < p.uniq; ++a) {
        if (a > 0 && p.src[(size_t)(p.u0 + a)] <= p.src[(size_t)(p.u0 + a - 1)]) return false;
        for (int64_t b = a + 1; b < p.uniq; ++b)
            if (std::memcmp(ids.data() + (int64_t)p.src[(size_t)(p.u0 + a)] * H,
                            ids.data() + (int64_t)p.src[(size_t)(p.u0 + b)] * H, (size_t)H * 4) == 0)
                return false;
    }
    return pooled == p.pooled;
}

}  // namespace

TEST(engdd_one_launch_over_the_step_is_exact_and_maps_every_row_to_its_own_ids) {
    const int64_t T = 4096, H = 24;
    const RowFn f(H);
    for (const int64_t distinct : { 120, 900, 3000 }) {
        const std::vector<int32_t> ids = make_ids(T, H, distinct, 1000 + (uint64_t)distinct);
        const std::vector<Launch> ls = { { 0, T } };
        Plan p;
        REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p) == kDedup);
        CHECK(plan_ok(p, ids, ls));
        CHECK(p.kept.empty());
        CHECK_EQ(p.u0, (int64_t)0);
        CHECK_EQ(p.runs.size(), (size_t)1);
        CHECK(p.uniq <= distinct);
        CHECK(same_pass(p, ids, f, reference(ids, T, f)));
    }
}

TEST(engdd_bi_launches_keep_the_band_and_dedup_the_wide_ones_exactly) {
    /* BI's issue at an Engram layer: the decode segment (60 rows: pieces 48 + 12), a short prefill segment (30), wide
     * prefill segments (1000, 900), another short one (40) between them, a last wide one (2000) */
    const int64_t H = 24;
    const std::vector<Launch> ls = { { 0, 48 }, { 48, 12 }, { 60, 30 }, { 90, 1000 }, { 1090, 40 }, { 1130, 900 },
                                     { 2030, 2000 } };
    const int64_t T = 4030;
    const RowFn f(H);
    const std::vector<int32_t> ids = make_ids(T, H, 700, 7);
    Plan p;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p) == kDedup);
    CHECK(plan_ok(p, ids, ls));
    CHECK_EQ(p.kept.size(), (size_t)4);                 /* 48, 12, 30, 40 */
    CHECK_EQ(p.u0, (int64_t)(48 + 12 + 30 + 40));
    CHECK_EQ(p.runs.size(), (size_t)2);                 /* [90, 1090) and [1130, 4030) */
    CHECK_EQ(p.runs[0].first, (int64_t)90);
    CHECK_EQ(p.runs[1].first, (int64_t)1130);
    CHECK_EQ(p.runs[1].second, (int64_t)2900);
    CHECK_EQ(p.pooled, (int64_t)3900);
    for (const Kept& k : p.kept) CHECK(k.n <= kFormRows);
    CHECK(same_pass(p, ids, f, reference(ids, T, f)));
}

TEST(engdd_rows_equal_but_for_one_id_stay_apart_under_slot_collisions) {
    /* 2000 distinct rows that share every id but the last, each repeated: the hash table probes through many occupied
     * slots, and a row is only merged with a whole-row equal one */
    const int64_t T = 4000, H = 24;
    std::vector<int32_t> ids((size_t)(T * H));
    for (int64_t r = 0; r < T; ++r) {
        for (int64_t h = 0; h < H - 1; ++h) ids[(size_t)(r * H + h)] = 12345 + (int32_t)h;
        ids[(size_t)(r * H + H - 1)] = (int32_t)((r * 7919) % 2000);
    }
    const std::vector<Launch> ls = { { 0, T } };
    Plan p;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p) == kDedup);
    CHECK_EQ(p.uniq, (int64_t)2000);
    CHECK(plan_ok(p, ids, ls));
    const RowFn f(H);
    CHECK(same_pass(p, ids, f, reference(ids, T, f)));
}

TEST(engdd_guards_the_band_the_share_and_a_pool_of_none) {
    const int64_t H = 4, T = 1000;
    const std::vector<Launch> one = { { 0, T } };
    auto rows_of = [&](int64_t distinct) {
        std::vector<int32_t> ids((size_t)(T * H));
        for (int64_t r = 0; r < T; ++r)
            for (int64_t h = 0; h < H; ++h) ids[(size_t)(r * H + h)] = (int32_t)((r % distinct) * 10 + h);
        return ids;
    };
    Plan p;
    /* <= 48 unique rows from a launch of > 48: the decode kernels would take them -- kept whole */
    for (const int64_t u : { 1, 40, 48 }) {
        const std::vector<int32_t> ids = rows_of(u);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p) == kGuard);
        CHECK_EQ(p.uniq, u);
        CHECK(p.src.empty() && p.kept.empty() && p.rows == 0);
    }
    {
        const std::vector<int32_t> ids = rows_of(49);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p) == kDedup);
        CHECK_EQ(p.uniq, (int64_t)49);
        CHECK(plan_ok(p, ids, one));
    }
    /* the share: 1000 pooled rows deduplicate at >= 63 repeated (63 x 16 = 1008 >= 1000), not at 62 */
    {
        std::vector<int32_t> ids = rows_of(1000 - 63);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p) == kDedup);
        ids = rows_of(1000 - 62);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p) == kLowShare);
        CHECK(p.src.empty() && p.rows == 0);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p, 0) == kDedup);   /* min_share_div 0: any repeat */
        ids = rows_of(1000);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p, 0) == kLowShare);   /* no repeat at all */
    }
    /* every launch inside the band: no pool */
    {
        const std::vector<int32_t> ids = rows_of(3);
        const std::vector<Launch> band = { { 0, 48 }, { 48, 48 }, { 96, 30 } };
        CHECK(plan(ids.data(), 126, H, band.data(), band.size(), p) == kNone);
        CHECK_EQ(p.pooled, (int64_t)0);
    }
}

TEST(engdd_mutations_of_the_scatter_are_caught) {
    const int64_t H = 24;
    const std::vector<Launch> ls = { { 0, 48 }, { 48, 20 }, { 68, 1500 }, { 1568, 30 }, { 1598, 1000 } };
    const int64_t T = 2598;
    const RowFn f(H);
    const std::vector<int32_t> ids = make_ids(T, H, 400, 99);
    const std::vector<uint16_t> ref = reference(ids, T, f);
    Plan p;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p) == kDedup);
    REQUIRE(same_pass(p, ids, f, ref));
    REQUIRE(plan_ok(p, ids, ls));
    /* a pooled row's map entry at another unique row (different ids) */
    {
        Plan q = p;
        const int64_t r = q.runs[1].first + 17;
        q.map[(size_t)r] = (q.map[(size_t)r] + 1) % (int32_t)q.uniq;
        CHECK(!same_pass(q, ids, f, ref));
        CHECK(!plan_ok(q, ids, ls));
    }
    /* a gather run shifted by one row (rows [r0 + 1, r0 + n] written from map[r0 ..]) */
    {
        Plan q = p;
        std::vector<int32_t> m2 = q.map;
        for (int64_t i = q.runs[0].second - 1; i >= 0; --i) m2[(size_t)(q.runs[0].first + i + 1)] = q.map[(size_t)(q.runs[0].first + i)];
        q.map = m2;
        CHECK(!same_pass(q, ids, f, ref));
    }
    /* a compact unique row taking another step row's ids (a row whose ids differ) */
    {
        Plan q = p;
        const size_t c = (size_t)(q.u0 + 5);
        int32_t other = -1;
        for (int32_t r = 0; r < (int32_t)T && other < 0; ++r)
            if (std::memcmp(ids.data() + (int64_t)r * H, ids.data() + (int64_t)p.src[c] * H, (size_t)H * 4) != 0) other = r;
        REQUIRE(other >= 0);
        q.src[c] = other;
        CHECK(!same_pass(q, ids, f, ref));
        CHECK(!plan_ok(q, ids, ls));
    }
    /* a kept launch written one row off (its GEMM's y at r0 + 1) */
    {
        Plan q = p;
        q.kept[1].r0 += 1;
        CHECK(!same_pass(q, ids, f, ref));
        CHECK(!plan_ok(q, ids, ls));
    }
    /* the guard lifted: a pool of <= 48 unique rows "deduplicated" is refused by the structure check */
    {
        Plan q = p;
        q.uniq = kFormRows;
        CHECK(!plan_ok(q, ids, ls));
    }
}

namespace {

/* tp8 loop P48b: the int8 twin's GEMM stand-in -- M <= 64 (the narrow band, gemm_i8a8_nt_m16: K split by M) changes a row's
 * bytes by M (here its low bits), above 64 every row is its own (the tiled band) */
constexpr int64_t kNarrow = 64;
void row_m(const RowFn& f, const int32_t* ids, uint16_t* out, int64_t M) {
    f(ids, out);
    if (M <= kNarrow)
        for (int64_t n = 0; n < kW; ++n) out[n] ^= (uint16_t)(1 + (M & 7));
}
/* the by-launch pass: every launch's rows at that launch's M */
std::vector<uint16_t> reference_m(const std::vector<int32_t>& ids, const std::vector<Launch>& ls, int64_t T, const RowFn& f) {
    std::vector<uint16_t> kv((size_t)(T * kW), 0);
    for (const Launch& l : ls)
        for (int64_t r = l.r0; r < l.r0 + l.n; ++r) row_m(f, ids.data() + r * f.H, kv.data() + r * kW, l.n);
    return kv;
}
/* the deduplicated pass with M: a kept launch at its own rows, the unique rows' GEMM at M = uniq, the pooled rows gathered */
bool same_pass_m(const Plan& p, const std::vector<int32_t>& ids, const RowFn& f, const std::vector<uint16_t>& ref) {
    std::vector<uint16_t> kv((size_t)(p.T * kW), 0xdead);
    std::vector<int32_t> cids((size_t)(p.rows * p.H));
    compact_ids(p, ids.data(), cids.data());
    for (const Kept& k : p.kept)
        for (int64_t i = 0; i < k.n; ++i) row_m(f, cids.data() + (k.c0 + i) * p.H, kv.data() + (k.r0 + i) * kW, k.n);
    std::vector<uint16_t> ku((size_t)(p.uniq * kW), 0);
    for (int64_t u = 0; u < p.uniq; ++u) row_m(f, cids.data() + (p.u0 + u) * p.H, ku.data() + u * kW, p.uniq);
    for (const auto& run : p.runs)
        for (int64_t i = 0; i < run.second; ++i) {
            const int32_t s = p.map[(size_t)(run.first + i)];
            if (s < 0 || s >= p.uniq) return false;
            std::memcpy(kv.data() + (run.first + i) * kW, ku.data() + (int64_t)s * kW, (size_t)kW * 2);
        }
    return std::memcmp(kv.data(), ref.data(), kv.size() * 2) == 0;
}

}  // namespace

TEST(engdd_p48b_twin_floor_keeps_the_narrow_band_and_dedups_the_tiled_launches_exactly) {
    /* BI's launches at an Engram layer with the twin: decode pieces 48 + 12, prompt segments 30, 60 (inside 49..64), 1000,
     * 64 (the band's edge), 900, 65 (just above it) */
    const int64_t H = 24;
    const std::vector<Launch> ls = { { 0, 48 }, { 48, 12 }, { 60, 30 }, { 90, 60 }, { 150, 1000 }, { 1150, 64 }, { 1214, 900 },
                                     { 2114, 65 } };
    const int64_t T = 2179;
    const RowFn f(H);
    const std::vector<int32_t> ids = make_ids(T, H, 500, 48);
    const std::vector<uint16_t> ref = reference_m(ids, ls, T, f);
    Plan p;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p, kMinShareDiv, kNarrow) == kDedup);
    CHECK_EQ(p.kept.size(), (size_t)5);                 /* 48, 12, 30, 60, 64 */
    for (const Kept& k : p.kept) CHECK(k.n <= kNarrow);
    CHECK_EQ(p.pooled, (int64_t)(1000 + 900 + 65));
    CHECK(p.uniq > kNarrow);
    CHECK(same_pass_m(p, ids, f, ref));
    /* negative control: today's floor 48 pools the 60- and 64-row launches into the tiled unique GEMM */
    Plan q;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), q) == kDedup);
    CHECK_EQ(q.pooled, (int64_t)(60 + 1000 + 64 + 900 + 65));
    CHECK(!same_pass_m(q, ids, f, ref));
}

TEST(engdd_p48b_twin_floor_refuses_a_pool_of_64_unique_rows) {
    const int64_t H = 4, T = 1000;
    const std::vector<Launch> one = { { 0, T } };
    auto rows_of = [&](int64_t distinct) {
        std::vector<int32_t> ids((size_t)(T * H));
        for (int64_t r = 0; r < T; ++r)
            for (int64_t h = 0; h < H; ++h) ids[(size_t)(r * H + h)] = (int32_t)((r % distinct) * 10 + h);
        return ids;
    };
    const RowFn f(H);
    Plan p;
    for (const int64_t u : { 49, 60, 64 }) {
        const std::vector<int32_t> ids = rows_of(u);
        CHECK(plan(ids.data(), T, H, one.data(), one.size(), p, kMinShareDiv, kNarrow) == kGuard);
        CHECK(p.src.empty() && p.kept.empty() && p.rows == 0);
        /* negative control: at floor 48 these pools dedup, and their unique GEMM runs in the narrow band */
        Plan q;
        REQUIRE(plan(ids.data(), T, H, one.data(), one.size(), q) == kDedup);
        CHECK(!same_pass_m(q, ids, f, reference_m(ids, one, T, f)));
    }
    const std::vector<int32_t> ids = rows_of(65);
    REQUIRE(plan(ids.data(), T, H, one.data(), one.size(), p, kMinShareDiv, kNarrow) == kDedup);
    CHECK_EQ(p.uniq, (int64_t)65);
    CHECK(same_pass_m(p, ids, f, reference_m(ids, one, T, f)));
}

TEST(engdd_p48b_the_unique_gemm_takes_the_twin_only_when_every_pooled_launch_does) {
    const std::vector<Launch> ls = { { 0, 48 }, { 48, 60 }, { 108, 1000 }, { 1108, 70 }, { 1178, 900 } };
    const int64_t T = 2078;
    const int kEng = 8;
    /* decode piece 0, a short prompt segment with the twin, the pooled prompt segments with the twin: all -> the class */
    { const int tw[] = { 0, 8, 8 | 1, 8, 8 | 4 }; CHECK_EQ(pool_twin(ls.data(), tw, ls.size(), T, kEng, kNarrow), kEng); }
    /* no pooled launch takes it (only the kept 60-row one): none */
    { const int tw[] = { 0, 8, 1, 0, 4 }; CHECK_EQ(pool_twin(ls.data(), tw, ls.size(), T, kEng, kNarrow), 0); }
    /* the pooled launches split (the 70-row one below a twin row floor): -1, the pass keeps its by-launch issue */
    { const int tw[] = { 0, 8, 8, 0, 8 }; CHECK_EQ(pool_twin(ls.data(), tw, ls.size(), T, kEng, kNarrow), -1); }
    /* at floor 48 the 60-row launch is pooled too: its class counts */
    { const int tw[] = { 0, 0, 8, 8, 8 }; CHECK_EQ(pool_twin(ls.data(), tw, ls.size(), T, kEng, kFormRows), -1); }
    /* a launch out of range is not pooled */
    { const std::vector<Launch> bad = { { 0, 100 }, { 2000, 500 } }; const int tw[] = { 8, 0 };
      CHECK_EQ(pool_twin(bad.data(), tw, bad.size(), 2100, kEng, kNarrow), kEng); }
}

TEST(engdd_compact_ids_are_the_source_rows) {
    const int64_t H = 6, T = 300;
    const std::vector<int32_t> ids = make_ids(T, H, 80, 3);
    const std::vector<Launch> ls = { { 0, 20 }, { 20, 280 } };
    Plan p;
    REQUIRE(plan(ids.data(), T, H, ls.data(), ls.size(), p) == kDedup);
    std::vector<int32_t> c((size_t)(p.rows * H));
    compact_ids(p, ids.data(), c.data());
    for (int64_t r = 0; r < p.rows; ++r)
        CHECK(std::memcmp(c.data() + r * H, ids.data() + (int64_t)p.src[(size_t)r] * H, (size_t)H * 4) == 0);
    CHECK_EQ(p.u0, (int64_t)20);
    for (int64_t r = 0; r < 20; ++r) CHECK_EQ(p.src[(size_t)r], (int32_t)r);
}

RAD_TEST_MAIN()
