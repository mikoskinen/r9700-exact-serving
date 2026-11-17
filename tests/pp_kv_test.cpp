/* tests/pp_kv_test.cpp -- the two-node pipeline's KV capacity (tp8 S4, review F1), with the real
 * KVManager (core/mem/kv.h): W(0,20) in miniature.
 *
 *   leader   : r1 FULL block 1 real (its layers' source), r2 FULL block 2 zero (the follower's)
 *   follower : r1 zero,                                   r2 real, 3 layers (dearer a token)
 *
 * The block ids are the leader's (one scheduler); each node carves its own groups and zeroes the
 * other's. Engine::pp_kv_agree has both nodes exchange paged_tokens() before the carve and cap at the
 * least (the one-machine rule for unequal ranks, Engine::configure_kv), so every group -- a zeroed one
 * too -- holds the same block count on both nodes (rendezvous 2 compares them), and no id the leader
 * hands out lies past the follower's pool. The review's pp_kvcap_test measured the earlier cap (a
 * loan after the carve) handing out r2 id 1,927 for a 677-block follower pool; this is its fixed case
 * as a regression test. */
#include "rad_test.h"

#include "device/device.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "rad_core.h"

#include <algorithm>
#include <vector>

namespace {

using namespace rad;

enum : int32_t { G_R1 = 1, G_R2 = 2 };

std::vector<KVGroupInfo> groups(bool r1_zero, bool r2_zero) {
    std::vector<KVGroupInfo> g(1);   /* handle 0: the null group */
    KVGroupInfo a;
    a.name = "kv_compressed_r1";
    a.decl.kind = RAD_KV_FULL;
    a.decl.dtype = RAD_BF16;
    a.decl.n_head_kv = 1;
    a.decl.head_dim = 64;
    a.block_size = 1;
    a.layers = { 20 };
    a.zero = r1_zero;
    g.push_back(a);
    KVGroupInfo b = a;
    b.name = "kv_compressed_r2";
    b.block_size = 2;
    b.layers = { 2, 8, 14 };
    b.zero = r2_zero;
    g.push_back(b);
    return g;
}

struct Node {
    Pools pools;
    KVManager kv;
    Config cfg;
    int plan(const std::vector<KVGroupInfo>& g) {
        cfg.vram_weights_mib = 1;
        cfg.vram_kv_mib = 1;
        cfg.gpu_headroom_mib = 1;
        cfg.host_pool_mib = 0;
        cfg.max_seqs = 8;
        cfg.prefix_cache = false;
        cfg.checkpoint_slots = 0;
        RAD_TRY(pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
        return kv.plan_groups(g, cfg);
    }
};

/* Engine::pp_kv_agree's rule: the least of the two, -1 meaning "pages nothing itself". */
int64_t agree(int64_t a, int64_t b) {
    if (a < 0) return b;
    if (b < 0) return a;
    return std::min(a, b);
}

TEST(both_nodes_cap_before_the_carve_and_hold_the_same_blocks_in_every_group) {
    Node lead, fol;
    REQUIRE(lead.plan(groups(false, true)) == RAD_OK);
    REQUIRE(fol.plan(groups(true, false)) == RAD_OK);
    const int64_t tl = lead.kv.paged_tokens(lead.pools.kv().avail());
    const int64_t tf = fol.kv.paged_tokens(fol.pools.kv().avail());
    fprintf(stderr, "    paged tokens before the carve: leader %lld, follower %lld\n", (long long)tl, (long long)tf);
    CHECK(tf < tl);   /* the follower's group is dearer: it limits */
    const int64_t least = agree(tl, tf);
    lead.kv.cap_paged_tokens(least);
    fol.kv.cap_paged_tokens(least);
    CHECK_OK(lead.kv.carve(lead.pools.kv()));
    CHECK_OK(fol.kv.carve(fol.pools.kv()));
    for (int32_t g : { G_R1, G_R2 }) {
        CHECK_EQ(lead.kv.plan(g)->n_blocks, fol.kv.plan(g)->n_blocks);
        CHECK(lead.kv.plan(g)->n_blocks > 0);
    }
    /* The leader's scheduler way: sequences grow by odd chunks until admission refuses. */
    const int S = (int)lead.cfg.max_seqs;
    std::vector<int64_t> len((size_t)S, 0);
    for (int s = 0; s < S; ++s) REQUIRE(lead.kv.add_sequence((uint64_t)(100 + s)) >= 0);
    int64_t total = 0, max_r1 = -1, max_r2 = -1;
    for (int round = 0, refused = 0; round < 100000 && refused < S; ++round) {
        refused = 0;
        for (int s = 0; s < S; ++s) {
            const int64_t want = len[(size_t)s] + 257;
            if (lead.kv.ensure((uint64_t)(100 + s), want) < 0) { ++refused; continue; }
            total += want - len[(size_t)s];
            len[(size_t)s] = want;
            for (int32_t b : lead.kv.block_table((uint64_t)(100 + s), G_R1)) max_r1 = std::max<int64_t>(max_r1, b);
            for (int32_t b : lead.kv.block_table((uint64_t)(100 + s), G_R2)) max_r2 = std::max<int64_t>(max_r2, b);
        }
    }
    fprintf(stderr, "    leader admitted %lld tokens (cap %lld); highest ids r1 %lld of %lld, r2 %lld of the follower's %lld\n",
            (long long)total, (long long)least, (long long)max_r1, (long long)fol.kv.plan(G_R1)->n_blocks,
            (long long)max_r2, (long long)fol.kv.plan(G_R2)->n_blocks);
    CHECK(total <= least);
    CHECK(max_r1 < fol.kv.plan(G_R1)->n_blocks);
    CHECK(max_r2 < fol.kv.plan(G_R2)->n_blocks);
    CHECK_EQ(lead.kv.loan_tokens(), 0);   /* no loan: nothing to recall */
}

TEST(a_node_that_pages_nothing_itself_takes_the_other_nodes_count) {
    CHECK_EQ(agree(-1, 1354), 1354);
    CHECK_EQ(agree(4064, -1), 4064);
    CHECK_EQ(agree(-1, -1), -1);
    CHECK_EQ(agree(4064, 1354), 1354);
}

}  /* namespace */

RAD_TEST_MAIN()
