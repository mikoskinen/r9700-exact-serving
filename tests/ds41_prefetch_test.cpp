/* tests/ds41_prefetch_test.cpp -- tp8 loop D24 (fn4 ds41_prefetch, RADIANCE_DS41_PREFETCH): the pass and site rules of
 * deepseek41's weight reads beside the decode all-reduces (arch/deepseek41/ds41_prefetch.h); host arithmetic only.
 *
 * libr4d's `prefetch` reads a weight's leading bytes and writes nothing (docs/OPS.md), so the reads cannot change a value;
 * what must hold is WHERE they are issued. Checked over every switch value, pass kind and height 0 .. 80 rows: a pass takes
 * the switch's bits only where the reads were declared, it is a trunk pass (no drafter pass, no encoder / generation pass),
 * every sequence decodes (no prefill sequence: a MIXED step and a prompt step never take the reads, rad_arch.h's rule) and
 * it has 1 .. 64 rows; site 0 (beside a layer's attention all-reduce) reads that layer's mix fn at bit 0 and its router +
 * shared gate_up at bit 1, never the next layer's; site 1 (beside the MoE all-reduce) reads the next layer's mix fn at bit
 * 0 and its comp_w (only at a kv source) + wq_a + wkv at bit 2, nothing where the pass ends at a crossing / no next held
 * layer. Controls that must fail: a rule that took a MIXED step's decode rows, or the published prefetch_beside's own
 * 16-row ceiling (kPrefetchRows) in place of 64, would leave the 17 .. 64-row decode lane steps (c32's 48 / 64) unread.
 */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_prefetch.h"

#include <cstdint>

namespace {

using namespace ds41_pf;

TEST(a_pass_takes_the_bits_only_when_declared_trunk_all_decode_and_within_64_rows) {
    long long took = 0, refused = 0;
    for (int sw = 0; sw <= 8; ++sw)
        for (int ok = 0; ok <= 1; ++ok)
            for (int dp : { 0, 1, -1 })
                for (int enc = 0; enc <= 1; ++enc)
                    for (int64_t T = 0; T <= 80; ++T)
                        for (int64_t S : { 1, 4, 8, 16 })
                            for (int kind = 0; kind < 3; ++kind) {   /* 0 decode, 1 prefill, 2 MIXED (decode rows + a prompt) */
                                const int64_t dn = kind == 0 ? S : kind == 1 ? 0 : S - 1;
                                const int64_t dt = kind == 0 ? T : kind == 1 ? 0 : (T > 1 ? T / 2 : 0);
                                const int got = pass_bits(sw, ok != 0, dp, enc != 0, T, S, dn, dt);
                                const bool want = ok && sw >= 1 && sw <= 7 && dp == 0 && !enc && T >= 1 && T <= 64 && kind == 0;
                                CHECK_EQ(got, want ? sw : 0);
                                if (want) ++took;
                                else ++refused;
                                /* controls: a MIXED or prompt step never; > 64 rows never; a drafter pass never */
                                if (kind != 0 || T > 64 || dp != 0) CHECK_EQ(got, 0);
                            }
    CHECK(took > 0);
    CHECK(refused > took);
    /* the decode lane steps the pair runs: c16 lanes of 8 (<= 32 rows), c32 lanes of 16 at depth 2 / 3 (48 / 64 rows) */
    for (int64_t T : { 1, 8, 16, 17, 24, 32, 33, 48, 49, 64 }) CHECK_EQ(pass_bits(7, true, 0, false, T, 16, 16, T), 7);
    /* the control: the published prefetch_beside ceiling (16 rows) would not take 17 .. 64 */
    int above16 = 0;
    for (int64_t T = 17; T <= 64; ++T) above16 += pass_bits(1, true, 0, false, T, 16, 16, T) == 1 ? 1 : 0;
    CHECK_EQ(above16, 48);
}

TEST(each_site_reads_its_own_weights_by_bit) {
    for (int bits = 0; bits <= 7; ++bits)
        for (int src = 0; src <= 1; ++src)
            for (int next = 0; next <= 1; ++next) {
                const unsigned a = site_reads(bits, 0, src != 0, next != 0);
                const unsigned m = site_reads(bits, 1, src != 0, next != 0);
                /* site 0: this layer's FFN side only, whatever the next layer is */
                CHECK_EQ(a, ((bits & 1) ? (unsigned)kMixF : 0u) | ((bits & 2) ? (unsigned)(kRouter | kShGu) : 0u));
                CHECK_EQ(a & (unsigned)(kMixA | kComp | kQa | kKv), 0u);
                /* site 1: the next layer's attention side, nothing without a next layer on this node */
                unsigned want = 0;
                if (next) {
                    if (bits & 1) want |= kMixA;
                    if (bits & 4) want |= (src ? (unsigned)kComp : 0u) | kQa | kKv;
                }
                CHECK_EQ(m, want);
                CHECK_EQ(m & (unsigned)(kMixF | kRouter | kShGu), 0u);
                if (!src) CHECK_EQ(m & (unsigned)kComp, 0u);
                CHECK_EQ(site_reads(bits, 2, src != 0, next != 0), 0u);   /* no third site */
            }
    /* bit 2 alone never reads at site 0; bit 1 alone never at site 1 */
    CHECK_EQ(site_reads(4, 0, true, true), 0u);
    CHECK_EQ(site_reads(2, 1, true, true), 0u);
}

}  // namespace

RAD_TEST_MAIN()
