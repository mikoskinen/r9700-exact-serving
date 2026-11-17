/* tests/ds41_layers_test.cpp -- tp8 S3's layer ranges in deepseek41 (arch/deepseek41/ds41_layers.h), host arithmetic only:
 *
 *   parse     RADIANCE_DS41_LAYERS / _MODE: unset, empty and every-layer specs are the published engine (on false); the
 *             ranges of the plan's splits; malformed specs refused (order, overlap, bounds, syntax); adjacent ranges merged;
 *             the leader is the node that holds the last layer; next() skips the gap.
 *   needs     against DeepSeek-V4.1-Flash 1017's geometry (40 layers; ratios 0 0 2x18 1x20; kv sources 2 8 14 20; index
 *             sources 2 8 14 20 24 28 32 36; candidate layer 20; Engram 1 14; DSpark taps 37 38 39): W(0,20)'s two nodes
 *             read nothing of each other (the clean cut, PLAN.md section 3); W(10,32)'s follower reads source 8's latent and
 *             index layer 8's selections at layers 10..13; its leader reads source 20's latent and index keys at 32..39 and
 *             the candidate layer's keep words at the index layers 32 and 36 (PLAN.md review rows 5-7); copies() lists the
 *             foreign sources a strict node holds a copy of.
 *   partition every split's two nodes hold every layer exactly once. */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_layers.h"

#include <string>
#include <vector>

namespace {

using namespace rad::arch::ds41layers;

/* DeepSeek-V4.1-Flash 1017 (rad-info of the 2.25 container: compress_ratios, kv_source_layer_ids, index_source_layer_ids,
 * candidate_source_layer_id, engram_layer_ids, dspark_target_layer_ids), src_of / idx_of as deepseek41's read_cfg derives
 * them: the latest kv / index source at or below a compressing layer. */
Geo flash1017() {
    Geo g;
    g.n_layer = 40;
    g.ratio.assign(40, 0);
    for (int l = 2; l < 20; ++l) g.ratio[(size_t)l] = 2;
    for (int l = 20; l < 40; ++l) g.ratio[(size_t)l] = 1;
    g.is_src.assign(40, 0);
    g.is_idx.assign(40, 0);
    for (int l : { 2, 8, 14, 20 }) g.is_src[(size_t)l] = 1;
    for (int l : { 2, 8, 14, 20, 24, 28, 32, 36 }) g.is_idx[(size_t)l] = 1;
    g.src_of.assign(40, -1);
    g.idx_of.assign(40, -1);
    int s = -1, x = -1;
    for (int l = 0; l < 40; ++l) {
        if (g.is_src[(size_t)l]) s = l;
        if (g.is_idx[(size_t)l]) x = l;
        if (g.ratio[(size_t)l] <= 0) continue;
        g.src_of[(size_t)l] = s;
        g.idx_of[(size_t)l] = x;
    }
    g.cand = 20;
    g.ple = { 1, 14 };
    g.taps = { 37, 38, 39 };
    return g;
}

Spec spec(const char* s, const char* mode = nullptr, int n = 40) {
    Spec o;
    std::string err;
    const int rc = parse(s, mode, n, &o, &err);
    CHECK_EQ(rc, 0);
    if (rc) std::fprintf(stderr, "  parse(%s): %s\n", s ? s : "null", err.c_str());
    return o;
}
bool refused(const char* s, const char* mode = nullptr, int n = 40) {
    Spec o;
    std::string err;
    const bool r = parse(s, mode, n, &o, &err) < 0 && !err.empty();
    return r;
}
int count(const std::vector<Dep>& v, Need k, int from) {
    int n = 0;
    for (const Dep& d : v) n += d.kind == k && d.from == from;
    return n;
}

}  // namespace

TEST(layers_unset_empty_and_every_layer_are_the_published_engine) {
    CHECK(!spec(nullptr).on);
    CHECK(!spec("").on);
    const Spec all = spec("0:40");
    CHECK(!all.on);
    CHECK_EQ(all.text, std::string("0:40"));
    CHECK(!spec("0:20,20:40").on);      /* adjacent ranges merge: every layer */
    CHECK(!spec("0:40", "strict").on);  /* every layer in either mode */
    for (int l = 0; l < 40; ++l) CHECK(all.held(l));
    CHECK(needs(flash1017(), all, true).empty());
}

TEST(layers_parse_the_plans_splits) {
    const Spec b = spec("0:20"), a = spec("20:40");
    CHECK(b.on && a.on);
    CHECK(!b.strict && !a.strict);
    CHECK_EQ(b.n_own, 20);
    CHECK_EQ(a.n_own, 20);
    CHECK(!b.leader);
    CHECK(a.leader);
    CHECK_EQ(b.first, 0);
    CHECK_EQ(b.last, 19);
    CHECK_EQ(a.first, 20);
    CHECK_EQ(a.last, 39);
    const Spec w = spec("0:10,32:40", "strict"), f = spec("10:32", "strict");
    CHECK(w.strict && f.strict);
    CHECK_EQ(w.n_own, 18);
    CHECK_EQ(f.n_own, 22);
    CHECK(w.leader && !f.leader);
    CHECK_EQ(w.text, std::string("0:10,32:40"));
    CHECK_EQ(w.next[9], 32);   /* the gap passes the streams from layer 9 to 32 */
    CHECK_EQ(w.next[39], -1);
    CHECK_EQ(w.next[0], 1);
    CHECK_EQ(f.next[31], -1);
    CHECK_EQ(f.first, 10);
    CHECK(!w.held(10) && !w.held(31) && w.held(9) && w.held(32));
    const Spec m = spec("0:5,5:10,30:40");   /* adjacent ranges merge */
    CHECK_EQ(m.ranges.size(), (size_t)2);
    CHECK_EQ(m.text, std::string("0:10,30:40"));
}

TEST(layers_malformed_specs_are_refused) {
    for (const char* s : { "20:10", "0:41", "5:5", "-1:3", "0:20,10:30", "20:40,0:10", "0:20,", ",0:20", "0-20", "a:b", "0:20x",
                           "0:", ":20", "0:20;20:40", " 0:20" })
        CHECK(refused(s));
    CHECK(refused("0:20", "both"));
    CHECK(!refused("0:20", "proxy"));
    CHECK(!refused("0:20", ""));
    CHECK(refused("0:1", nullptr, 0));
}

TEST(layers_w0_20_is_a_clean_cut) {
    const Geo g = flash1017();
    for (const char* mode : { "proxy", "strict" }) {
        const Spec b = spec("0:20", mode), a = spec("20:40", mode);
        CHECK(needs(g, b, false).empty());
        CHECK(needs(g, a, true).empty());
        CHECK(copies(needs(g, a, true)).empty());
    }
}

TEST(layers_w10_32_reads_across_the_nodes) {
    const Geo g = flash1017();
    const Spec f = spec("10:32", "strict"), a = spec("0:10,32:40", "strict");
    const std::vector<Dep> nf = needs(g, f, false), na = needs(g, a, true);
    /* the follower: layers 10..13 read source 8's latent and index layer 8's selections (rows 5 and 7 of PLAN.md's review) */
    CHECK_EQ(count(nf, Need::Latent, 8), 4);
    CHECK_EQ(count(nf, Need::Sel, 8), 4);
    CHECK_EQ(count(nf, Need::IndexKeys, 8), 0);
    CHECK_EQ((int)nf.size(), 8);
    for (const Dep& d : nf) CHECK(d.layer >= 10 && d.layer <= 13);
    /* the leader: layers 32..39 read source 20's latent, the index layers 32 / 36 its index keys and layer 20's keep words */
    CHECK_EQ(count(na, Need::Latent, 20), 8);
    CHECK_EQ(count(na, Need::IndexKeys, 20), 2);
    CHECK_EQ(count(na, Need::Keep, 20), 2);
    CHECK_EQ(count(na, Need::Sel, 20), 0);   /* 33-35 and 37-39 read 32's and 36's, held */
    CHECK_EQ(count(na, Need::Tap, 37) + count(na, Need::Tap, 38) + count(na, Need::Tap, 39), 0);
    const std::vector<Copy> cf = copies(nf), ca = copies(na);
    CHECK_EQ(cf.size(), (size_t)1);
    CHECK(cf.size() == 1 && cf[0].layer == 8 && cf[0].latent && !cf[0].index);   /* "the source-8 copy needs only the latent" */
    CHECK_EQ(ca.size(), (size_t)1);
    CHECK(ca.size() == 1 && ca[0].layer == 20 && ca[0].latent && ca[0].index);   /* "a source-20 copy (544 B)" */
    CHECK_EQ(describe(nf), std::string("layers 10..13 read layer 8's compressed latent; layers 10..13 read layer 8's selections"));
    CHECK_EQ(describe(na), std::string("layers 32..39 read layer 20's compressed latent; layers 32, 36 read layer 20's index keys; "
                                       "layers 32, 36 read layer 20's keep words"));
}

TEST(layers_t_moves_the_reads) {
    const Geo g = flash1017();
    /* t = 34 (the drafting lanes' cut): the leader 0:10,34:40 reads index 32's selections at 34, 35 (no index layer held
     * past 32 before 36) and source 20 throughout; the follower 10:34 holds 32 */
    const Spec a = spec("0:10,34:40", "strict"), f = spec("10:34", "strict");
    const std::vector<Dep> na = needs(g, a, true), nf = needs(g, f, false);
    CHECK_EQ(count(na, Need::Sel, 32), 2);
    CHECK_EQ(count(na, Need::Latent, 20), 6);
    CHECK_EQ(count(na, Need::IndexKeys, 20), 1);   /* layer 36 */
    CHECK_EQ(count(nf, Need::Latent, 8), 4);
    /* the drafter's taps on the follower: a leader 38:40 lacks tap 37 */
    const Spec l38 = spec("38:40", "strict");
    CHECK_EQ(count(needs(g, l38, true), Need::Tap, 37), 1);
}

TEST(layers_every_split_partitions_the_layers) {
    const char* pairs[][2] = { { "0:20", "20:40" }, { "0:10,32:40", "10:32" }, { "0:10,33:40", "10:33" },
                               { "0:10,34:40", "10:34" }, { "0:9,32:40", "9:32" }, { "0:11,30:40", "11:30" } };
    for (const auto& p : pairs) {
        const Spec x = spec(p[0], "strict"), y = spec(p[1], "strict");
        CHECK(x.on && y.on);
        CHECK(x.leader != y.leader);
        for (int l = 0; l < 40; ++l) CHECK_EQ((int)x.held(l) + (int)y.held(l), 1);
    }
}

/* tp8 S6: the crossings of a general cut (PLAN.md section 3: X1 + layer 8's selections + new source-8 latent entries; X2 +
 * the last index layer's selections, the keep words, new source-20 entries) */
namespace {
struct Want {
    int h, t;
    bool r1;
    int sel1, keep1, src1;
    bool lat1, idx1;
    int sel2, keep2, src2;
    bool lat2, idx2;
};
}  // namespace

TEST(layers_s6_crossings_of_the_plan_cuts) {
    const Geo g = flash1017();
    const Want ws[] = {
        /* W(0,20): the clean cut, nothing but the embedding's output and the residual */
        { 0, 20, false, -1, -1, -1, false, false, -1, -1, -1, false, false },
        /* W(10,32): X1 source 8's latent (layers 10..13 attend over it) + index 8's selections; X2 source 20's latent and
         * index keys (index layers 32, 36) + the candidate layer's keep words; layer 32 selects for itself */
        { 10, 32, true, 8, -1, 8, true, false, -1, 20, 20, true, true },
        /* W(10,33) / W(10,34): + index 32's selections into the leader's 33.. */
        { 10, 33, true, 8, -1, 8, true, false, 32, 20, 20, true, true },
        { 10, 34, true, 8, -1, 8, true, false, 32, 20, 20, true, true },
        /* W(8,34): layer 8 is its own source and index layer, so X1 is the residual alone */
        { 8, 34, true, -1, -1, -1, false, false, 32, 20, 20, true, true },
        /* W(9,32): layer 9 reads source 8's latent and index 8's selections */
        { 9, 32, true, 8, -1, 8, true, false, -1, 20, 20, true, true },
        /* W(22,30): the follower's index layers 24, 28 read source 20's index keys and keep; X2 index 28's selections */
        { 22, 30, true, 20, 20, 20, true, true, 28, -1, -1, false, false },
        /* W(1,20): an Engram layer as the follower's first (its boundary is mhc_post alone, S4's form) */
        { 1, 20, true, -1, -1, -1, false, false, -1, -1, -1, false, false },
    };
    for (const Want& w : ws) {
        Crossing x[2];
        std::string err;
        CHECK_EQ(crossings(g, w.h, w.t, x, &err), 0);
        if (!err.empty()) std::fprintf(stderr, "  W(%d,%d): %s\n", w.h, w.t, err.c_str());
        CHECK_EQ(x[0].at, w.h);
        CHECK_EQ(x[1].at, w.t);
        CHECK_EQ(x[0].residual, w.r1);
        CHECK(x[1].residual);
        CHECK_EQ(x[0].sel, w.sel1);
        CHECK_EQ(x[0].keep, w.keep1);
        CHECK_EQ(x[0].src, w.src1);
        CHECK_EQ(x[0].latent, w.lat1);
        CHECK_EQ(x[0].index, w.idx1);
        CHECK_EQ(x[1].sel, w.sel2);
        CHECK_EQ(x[1].keep, w.keep2);
        CHECK_EQ(x[1].src, w.src2);
        CHECK_EQ(x[1].latent, w.lat2);
        CHECK_EQ(x[1].index, w.idx2);
    }
}

TEST(layers_s6_cuts_the_geometry_refuses) {
    const Geo g = flash1017();
    Crossing x[2];
    std::string err;
    CHECK(crossings(g, 0, 38, x, &err) < 0 && err.find("DSpark") != std::string::npos);   /* tap 37 on the follower */
    CHECK(crossings(g, 10, 40, x, &err) < 0);   /* the leader must hold the last layer */
    CHECK(crossings(g, 12, 12, x, &err) < 0);
    CHECK(crossings(g, -1, 20, x, &err) < 0);
    /* every cut 0 <= h < t <= 37 is allowed (one source, one selection layer, one keep a crossing on this geometry) */
    int ok = 0;
    for (int h = 0; h < 40; ++h)
        for (int t = h + 1; t < 40; ++t) ok += crossings(g, h, t, x, &err) == 0;
    CHECK_EQ(ok, 37 * 38 / 2);   /* t = 1 .. 37, h = 0 .. t - 1 */
}

TEST(layers_s6_binds_hold_or_copy) {
    const Geo g = flash1017();
    Crossing x[2];
    std::string err;
    CHECK_EQ(crossings(g, 10, 32, x, &err), 0);
    const Spec f = spec("10:32", "strict"), a = spec("0:10,32:40", "strict");
    const Binds bf = binds(g, f, &x[0]), ba = binds(g, a, &x[1]);
    /* the follower: source 8's latent copy (not its index keys), 14 and 20 its own; source 2 unbound */
    CHECK(!bf.comp[2] && bf.comp[8] && bf.comp[14] && bf.comp[20]);
    CHECK(!bf.index[2] && !bf.index[8] && bf.index[14] && bf.index[20]);
    CHECK(!bf.tail[2] && !bf.tail[8] && bf.tail[14] && !bf.tail[20]);
    CHECK(!bf.comp_zero_r[2] && !bf.comp_zero_r[1] && !bf.index_zero_r[2] && !bf.index_zero_r[1]);
    /* the leader: 2 and 8 its own, source 20's latent + index keys copied, 14 unbound */
    CHECK(ba.comp[2] && ba.comp[8] && !ba.comp[14] && ba.comp[20]);
    CHECK(ba.index[2] && ba.index[8] && !ba.index[14] && ba.index[20]);
    CHECK(ba.tail[2] && ba.tail[8] && !ba.tail[14] && !ba.tail[20]);
    /* W(0,20): the clean cut zeroes the other ratio on each node (S4's groups) */
    CHECK_EQ(crossings(g, 0, 20, x, &err), 0);
    const Binds b0 = binds(g, spec("0:20", "strict"), &x[0]), b1 = binds(g, spec("20:40", "strict"), &x[1]);
    CHECK(b0.comp_zero_r[1] && b0.index_zero_r[1] && !b0.comp_zero_r[2]);
    CHECK(b1.comp_zero_r[2] && b1.index_zero_r[2] && !b1.comp_zero_r[1]);
}

RAD_TEST_MAIN()
