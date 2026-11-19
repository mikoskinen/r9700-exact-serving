/* tests/ds41_dkv_held_test.cpp -- tp8 loop D25 step 1 (RADIANCE_DS41_DKV_GPIECE=1): lever DKV's gathered latent rows held a
 * decode band piece (arch/deepseek41/ds41_dkv_held.h); host arithmetic only.
 *
 * The gather copies the same selected entries wherever it sits, so the rule under test is the hold: a hit only for an entry
 * whose rows no later gather overwrote. Checked: a 64-row segment as two 32-row pieces over a layer walk whose kv sources
 * repeat (DeepSeek-V4.1's compressing layers read the latest source at or below them) -- each piece gathers once a source and
 * hits at every later layer of it (today's one-entry hold re-gathers both pieces at every layer: the control); a new source
 * drops exactly the rows it overwrites; a model of the buffer (which (source, piece) each row holds) agrees with every hit.
 */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_dkv_held.h"

#include <cstdint>
#include <random>
#include <vector>

namespace {

TEST(two_pieces_gather_once_a_source_and_hit_at_its_later_layers) {
    /* layers 0..39 by their kv source (every 4 layers a new source) and one index source */
    std::vector<Ds41Held> held;
    int gathers = 0, hits = 0, one_entry_gathers = 0;
    Ds41Held one;   /* today's single entry, for the control */
    for (int l = 0; l < 40; ++l) {
        const int src = l / 4 * 4;
        for (int p = 0; p < 2; ++p) {
            Ds41Held k;
            k.src = src; k.idx = 0; k.s0 = p * 8; k.s1 = p * 8 + 8; k.r0 = p * 32; k.n = 32;
            if (ds41_held_take(held, k)) ++hits;
            else ++gathers;
            CHECK(held.size() <= 2);
            /* control: one entry, every piece at rows [0, 32) */
            Ds41Held c = k;
            c.r0 = 0;
            if (!(one.src == c.src && one.idx == c.idx && one.s0 == c.s0 && one.s1 == c.s1 && one.r0 == c.r0 && one.n == c.n)) {
                ++one_entry_gathers;
                one = c;
            }
        }
    }
    CHECK_EQ(gathers, 20);          /* 10 sources x 2 pieces */
    CHECK_EQ(hits, 60);
    CHECK_EQ(one_entry_gathers, 80);   /* the control re-gathers every piece at every layer */
}

TEST(a_hit_is_only_ever_rows_nothing_overwrote) {
    std::mt19937_64 rng(25);
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<Ds41Held> held;
        /* the buffer model: per row, the key that last gathered it (src, idx, s0, s1, r0, n) */
        std::vector<Ds41Held> row(64);
        for (int step = 0; step < 60; ++step) {
            Ds41Held k;
            k.src = (int)(rng() % 4); k.idx = (int)(rng() % 2);
            const int64_t r0 = (int64_t)(rng() % 4) * 16, n = 16 + (int64_t)(rng() % 3) * 16;
            if (r0 + n > 64) continue;
            k.r0 = r0; k.n = n; k.s0 = r0 / 4; k.s1 = (r0 + n) / 4;
            const bool hit = ds41_held_take(held, k);
            bool intact = true;
            for (int64_t r = r0; r < r0 + n; ++r) {
                const Ds41Held& h = row[(size_t)r];
                intact = intact && h.src == k.src && h.idx == k.idx && h.s0 == k.s0 && h.s1 == k.s1 && h.r0 == k.r0 && h.n == k.n;
            }
            if (hit) CHECK(intact);   /* never a hit on rows another gather overwrote */
            if (!hit)
                for (int64_t r = r0; r < r0 + n; ++r) row[(size_t)r] = k;
            /* held entries never overlap */
            for (size_t i = 0; i < held.size(); ++i)
                for (size_t j = i + 1; j < held.size(); ++j)
                    CHECK(!(held[i].r0 < held[j].r0 + held[j].n && held[j].r0 < held[i].r0 + held[i].n));
        }
    }
}

}  // namespace

RAD_TEST_MAIN()
