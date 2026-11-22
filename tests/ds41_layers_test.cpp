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

/* (review of S6) Every allowed cut: each crossing carries exactly what the receiving node's layers read of the sending node
 * (every need of the receiver is the crossing's source / selection / keep, and each part it carries is read), and what it
 * carries is the sending node's LAST writer of that plane below the boundary -- the latent / index-key / page planes hold
 * the last kv source the sender computed, the selection plane its last index layer's, so a crossing of an earlier one would
 * ship another layer's bytes. */
TEST(layers_s6_every_cut_crosses_the_senders_last_writer) {
    const Geo g = flash1017();
    int cuts = 0, with_src = 0, with_sel = 0, with_keep = 0;
    for (int h = 0; h < 40; ++h)
        for (int t = h + 1; t <= 37; ++t) {
            Crossing x[2];
            std::string err;
            if (crossings(g, h, t, x, &err) != 0) {
                std::fprintf(stderr, "  W(%d,%d): %s\n", h, t, err.c_str());
                CHECK(false);
                continue;
            }
            ++cuts;
            const std::string fs = std::to_string(h) + ":" + std::to_string(t);
            const std::string ls = (h > 0 ? "0:" + std::to_string(h) + "," : std::string()) + std::to_string(t) + ":40";
            const Spec fol = spec(fs.c_str(), "strict"), lead = spec(ls.c_str(), "strict");
            for (int k = 0; k < 2; ++k) {
                const Crossing& c = x[k];
                const Spec& snd = k == 0 ? lead : fol;
                const Spec& rcv = k == 0 ? fol : lead;
                const std::vector<Dep> nd = needs(g, rcv, k == 1);
                bool src_read = false, lat_read = false, idx_read = false, sel_read = false, keep_read = false;
                for (const Dep& d : nd) {
                    CHECK(snd.held(d.from));   /* a need of the receiver is the sender's layer */
                    if (d.kind == Need::Latent) { CHECK_EQ(d.from, c.src); src_read = lat_read = true; }
                    if (d.kind == Need::IndexKeys) { CHECK_EQ(d.from, c.src); src_read = idx_read = true; }
                    if (d.kind == Need::Sel) { CHECK_EQ(d.from, c.sel); sel_read = true; }
                    if (d.kind == Need::Keep) { CHECK_EQ(d.from, c.keep); keep_read = true; }
                    CHECK(d.kind != Need::Tap);
                }
                CHECK_EQ(c.src >= 0, src_read);
                CHECK_EQ(c.latent, lat_read);
                CHECK_EQ(c.index, idx_read);
                CHECK_EQ(c.sel >= 0, sel_read);
                CHECK_EQ(c.keep >= 0, keep_read);
                const int lo = k == 0 ? 0 : h;   /* the sender's layers below the boundary c.at */
                if (c.src >= 0) {
                    ++with_src;
                    CHECK(snd.held(c.src) && g.is_src[(size_t)c.src]);
                    for (int l = std::max(lo, c.src + 1); l < c.at; ++l) CHECK(!(snd.held(l) && g.is_src[(size_t)l]));
                }
                if (c.sel >= 0) {
                    ++with_sel;
                    CHECK(snd.held(c.sel) && g.is_idx[(size_t)c.sel]);
                    for (int l = std::max(lo, c.sel + 1); l < c.at; ++l) CHECK(!(snd.held(l) && g.is_idx[(size_t)l]));
                }
                if (c.keep >= 0) {
                    ++with_keep;
                    CHECK_EQ(c.keep, g.cand);
                    CHECK(snd.held(c.keep));
                }
            }
        }
    CHECK_EQ(cuts, 37 * 38 / 2);
    std::fprintf(stderr, "  %d cuts: %d crossings carry a kv source's entries, %d selections, %d keep words\n", cuts, with_src,
                 with_sel, with_keep);
    CHECK(with_src > 0 && with_sel > 0 && with_keep > 0);
}

/* (review of S6) A geometry where the sender computes another kv source between the one that crosses and the boundary (two
 * ratios interleaved: layer 1 a ratio-2 source, layer 2 a ratio-4 source, layers 3..6 ratio 2, 7..9 ratio 4): W(3,7)'s X1
 * would ship source 2's entries as source 1's -- refused by name. W(2,7) (source 2 on the follower) and W(3,8) are allowed
 * or refused by the other rules only. */
TEST(layers_s6_a_crossing_behind_another_source_is_refused) {
    Geo g;
    g.n_layer = 10;
    g.ratio = { 0, 2, 4, 2, 2, 2, 2, 4, 4, 4 };
    g.is_src = { 0, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
    g.is_idx.assign(10, 0);
    g.src_of = { -1, 1, 2, 1, 1, 1, 1, 2, 2, 2 };
    g.idx_of.assign(10, -1);
    Crossing x[2];
    std::string err;
    CHECK(crossings(g, 3, 7, x, &err) < 0);
    std::fprintf(stderr, "  W(3,7): %s\n", err.c_str());
    CHECK(err.find("computes kv source 2 after it") != std::string::npos);
    err.clear();
    CHECK_EQ(crossings(g, 2, 7, x, &err), 0);   /* the leader's last source is 1; source 2 is the follower's own */
    CHECK_EQ(x[0].src, 1);
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

/* tp8 M56: lanes issue the leader's pass as a head ([0, h), X1) and a tail (X2, [t, 40)) with other lanes' parts
 * between them, so a leader layer >= t must not read the selections / keep words of a leader layer < h. The plan's
 * cuts read none; W(22,30) does (index layers 32 / 36 read the candidate layer 20's keep words), and so does W(22,23)
 * (layer 23 reads index layer 20's selections). */
TEST(layers_m56_lanes_head_tail_reads) {
    const Geo g = flash1017();
    for (const auto& c : std::vector<std::pair<int, int>>{ { 0, 20 }, { 10, 32 }, { 10, 33 }, { 10, 34 }, { 8, 34 },
                                                           { 9, 32 }, { 0, 37 } })
        CHECK(head_tail_reads(g, c.first, c.second).empty());
    const std::vector<Dep> a = head_tail_reads(g, 22, 30);
    REQUIRE(!a.empty());
    bool keep32 = false, keep36 = false;
    for (const Dep& d : a) {
        CHECK(d.from < 22);
        keep32 = keep32 || (d.layer == 32 && d.from == 20 && d.kind == Need::Keep);
        keep36 = keep36 || (d.layer == 36 && d.from == 20 && d.kind == Need::Keep);
    }
    CHECK(keep32 && keep36);
    const std::vector<Dep> b = head_tail_reads(g, 22, 23);
    REQUIRE(!b.empty());
    CHECK_EQ(b[0].layer, 23);
    CHECK_EQ(b[0].from, 20);
    CHECK(b[0].kind == Need::Sel);
    /* every cut the geometry allows: a read is always of a leader layer below h by a leader layer at or past t */
    int clean = 0;
    for (int t = 1; t <= 37; ++t)
        for (int h = 0; h < t; ++h) {
            const std::vector<Dep> v = head_tail_reads(g, h, t);
            for (const Dep& d : v) CHECK(d.from < h && d.layer >= t);
            if (v.empty()) ++clean;
        }
    fprintf(stderr, "    %d of %d cuts W(h, t <= 37) can run lanes\n", clean, 37 * 38 / 2);
    CHECK(clean > 0);
}

/* tp8 dev M4 (loop FINDINGS F12): a follower segment boundary s of W(h, t) saves what [s, t) reads of [h, s) -- X1 of
 * W(s, t) without its kv source (the follower holds its caches) -- AND what the pass's X2 ships that a layer below s
 * computed: its source's entries, its selections / keep words. The card defect: W(10,32)'s X2 ships source 20's latent and
 * index keys, the boundaries at 22 .. 30 dropped them, a decode pass between the segments rewrote the planes, and the
 * leader's copy of source 20 took that pass's rows. Every cut, every boundary. */
TEST(layers_m4_segment_boundaries_keep_what_x2_ships) {
    const Geo g = flash1017();
    /* the card cut first: W(10,32) at k 2 / 4 (lq28 / lq29: boundaries 12 .. 30 / 14 .. 30) */
    for (int s = 11; s < 32; ++s) {
        Crossing c;
        std::string err;
        CHECK_EQ(segment_crossing(g, 10, s, 32, &c, &err), 0);
        CHECK(c.residual);
        CHECK_EQ(c.src, s > 20 ? 20 : -1);
        CHECK_EQ(c.latent, s > 20);
        CHECK_EQ(c.index, s > 20);
        CHECK_EQ(c.keep, s > 20 ? 20 : -1);
    }
    int pairs = 0, x2_src = 0, x2_keep = 0, x2_sel = 0;
    for (int t = 2; t <= 37; ++t)
        for (int h = 0; h < t; ++h) {
            Crossing whole[2];
            std::string err;
            REQUIRE(crossings(g, h, t, whole, &err) == 0);
            const Crossing& x2 = whole[1];
            for (int s = h + 1; s < t; ++s) {
                Crossing c, at_s[2];
                REQUIRE(segment_crossing(g, h, s, t, &c, &err) == 0);
                REQUIRE(crossings(g, s, t, at_s, &err) == 0);
                ++pairs;
                CHECK(c.residual);
                CHECK_EQ(c.at, s);
                /* the kv source: X2's when its layer is below s, never X1-of-W(s, t)'s (the node's own caches) */
                if (x2.src >= 0 && x2.src < s) {
                    ++x2_src;
                    CHECK_EQ(c.src, x2.src);
                    CHECK_EQ(c.latent, x2.latent);
                    CHECK_EQ(c.index, x2.index);
                    /* no source in [s, t): the planes keep it from the save to X2's send */
                    for (int l = s; l < t; ++l) CHECK(!g.is_src[(size_t)l]);
                } else {
                    CHECK_EQ(c.src, -1);
                    CHECK(!c.latent && !c.index);
                }
                /* the selections / keep words: what [s, t) reads below s, and what X2 ships below s -- one layer each */
                if (at_s[0].sel >= 0) CHECK_EQ(c.sel, at_s[0].sel);
                if (x2.sel >= 0 && x2.sel < s) { ++x2_sel; CHECK_EQ(c.sel, x2.sel); }
                if (at_s[0].sel < 0 && !(x2.sel >= 0 && x2.sel < s)) CHECK_EQ(c.sel, -1);
                if (at_s[0].keep >= 0) CHECK_EQ(c.keep, at_s[0].keep);
                if (x2.keep >= 0 && x2.keep < s) { ++x2_keep; CHECK_EQ(c.keep, x2.keep); }
                if (at_s[0].keep < 0 && !(x2.keep >= 0 && x2.keep < s)) CHECK_EQ(c.keep, -1);
                /* every read at or after s of a plane a layer in [h, s) wrote is saved: [s, t)'s selections / keep words,
                 * X2's three */
                for (int l = s; l < t; ++l) {
                    if (g.ratio[(size_t)l] <= 0) continue;
                    const int ix = g.idx_of[(size_t)l];
                    if (!g.is_idx[(size_t)l] && ix >= h && ix < s) CHECK_EQ(c.sel, ix);
                    if (g.is_idx[(size_t)l] && g.cand >= h && g.cand < s && g.cand < l &&
                        g.ratio[(size_t)g.cand] == g.ratio[(size_t)l])
                        CHECK_EQ(c.keep, g.cand);
                }
            }
        }
    std::fprintf(stderr, "  %d (cut, boundary) pairs: X2's source entries saved at %d, its selections at %d, its keep at %d\n",
                 pairs, x2_src, x2_sel, x2_keep);
    CHECK(x2_src > 0 && x2_keep > 0);
    /* outside (h, t) is refused */
    Crossing c;
    std::string err;
    CHECK(segment_crossing(g, 10, 10, 32, &c, &err) < 0);
    CHECK(segment_crossing(g, 10, 32, 32, &c, &err) < 0);
}

/* tp8 dev M9: a leader segment boundary s of W(h, t) -- in its head (0 < s < h) or its tail (t < s <= the first tap) -- saves
 * what the part's layers from s read below s (the residual, the selections, the keep words; no kv source: the leader's caches
 * and copies) and, in the head, what X1 ships that a layer below s computed (its source's entries, its selections / keep
 * words). The card cut first (W(10,32): head boundaries 1 .. 9, tail 33 .. 37), then every cut and boundary against the plane
 * rule; a tail boundary past the first tap and boundaries outside the parts are refused. */
TEST(layers_m9_lead_segment_boundaries_keep_what_the_part_reads) {
    const Geo g = flash1017();
    for (int s = 1; s < 10; ++s) {
        Crossing c;
        std::string err;
        CHECK_EQ(lead_segment_crossing(g, 10, s, 32, &c, &err), 0);
        CHECK(c.residual);
        CHECK_EQ(c.src, s > 8 ? 8 : -1);           /* X1 ships source 8's latent: saved past layer 8 */
        CHECK_EQ(c.latent, s > 8);
        CHECK(!c.index);
        CHECK_EQ(c.sel, s >= 3 && s <= 7 ? 2 : s == 9 ? 8 : -1);   /* layers s..7 read index 2's; 9 reads 8's (X1's too) */
        CHECK_EQ(c.keep, -1);
    }
    for (int s = 33; s <= 37; ++s) {
        Crossing c;
        std::string err;
        CHECK_EQ(lead_segment_crossing(g, 10, s, 32, &c, &err), 0);
        CHECK(c.residual);
        CHECK_EQ(c.src, -1);
        CHECK_EQ(c.sel, s <= 35 ? 32 : s == 37 ? 36 : -1);   /* 33..35 read index 32's, 37..39 index 36's */
        CHECK_EQ(c.keep, s <= 36 ? 20 : -1);               /* index layer 36 reads candidate 20's keep words (X2's) */
    }
    int pairs = 0, x1_src = 0, x1_sel = 0;
    for (int t = 2; t <= 37; ++t)
        for (int h = 0; h < t; ++h) {
            Crossing whole[2];
            std::string err;
            REQUIRE(crossings(g, h, t, whole, &err) == 0);
            const Crossing& x1 = whole[0];
            std::vector<int> bs;
            for (int s = 1; s < h; ++s) bs.push_back(s);
            for (int s = t + 1; s <= 37; ++s) bs.push_back(s);
            for (int s : bs) {
                const bool head = s < h;
                Crossing c;
                REQUIRE(lead_segment_crossing(g, h, s, t, &c, &err) == 0);
                ++pairs;
                CHECK(c.residual);
                CHECK_EQ(c.at, s);
                if (head && x1.src >= 0 && x1.src < s) {
                    ++x1_src;
                    CHECK_EQ(c.src, x1.src);
                    CHECK_EQ(c.latent, x1.latent);
                    CHECK_EQ(c.index, x1.index);
                    for (int l = s; l < h; ++l) CHECK(!g.is_src[(size_t)l]);   /* the planes keep it from the save to X1 */
                } else {
                    CHECK_EQ(c.src, -1);
                    CHECK(!c.latent && !c.index);
                }
                if (head && x1.sel >= 0 && x1.sel < s) { ++x1_sel; CHECK_EQ(c.sel, x1.sel); }
                if (head && x1.keep >= 0 && x1.keep < s) CHECK_EQ(c.keep, x1.keep);
                /* every read at or after s, inside the part, of a plane a layer below s wrote is saved -- and nothing else */
                int want_sel = head && x1.sel >= 0 && x1.sel < s ? x1.sel : -1, want_keep = head && x1.keep >= 0 && x1.keep < s ? x1.keep : -1;
                for (int l = s; l < (head ? h : 40); ++l) {
                    if (g.ratio[(size_t)l] <= 0) continue;
                    const int ix = g.idx_of[(size_t)l];
                    if (!g.is_idx[(size_t)l] && ix >= 0 && ix < s) { CHECK_EQ(c.sel, ix); want_sel = ix; }
                    if (g.is_idx[(size_t)l] && g.cand >= 0 && g.cand < s && g.cand < l && g.ratio[(size_t)g.cand] == g.ratio[(size_t)l]) {
                        CHECK_EQ(c.keep, g.cand);
                        want_keep = g.cand;
                    }
                }
                CHECK_EQ(c.sel, want_sel);
                CHECK_EQ(c.keep, want_keep);
            }
        }
    std::fprintf(stderr, "  %d (cut, leader boundary) pairs: X1's source entries saved at %d, its selections at %d\n", pairs,
                 x1_src, x1_sel);
    CHECK(pairs > 0 && x1_src > 0 && x1_sel > 0);
    Crossing c;
    std::string err;
    CHECK(lead_segment_crossing(g, 10, 38, 32, &c, &err) < 0);   /* past the first tap (37) */
    CHECK(lead_segment_crossing(g, 10, 10, 32, &c, &err) < 0);   /* X1's own layer */
    CHECK(lead_segment_crossing(g, 10, 20, 32, &c, &err) < 0);   /* the follower's */
    CHECK(lead_segment_crossing(g, 10, 0, 32, &c, &err) < 0);
}

/* tp8 loop P28: a band pass of W(10,32) at k = 1..4 -- the leader's head over [0, 10 + k), the follower's from 10 + k. Its X1
 * carries the follower's copy of source 8's latent (layers 10..13 attend over it in every later decode pass), the band
 * layers' window keys, and what [10 + k, 32) reads below 10 + k: index 8's selections and source 8's latent while 10 + k <= 13
 * (the same source as the copy: one plane), the residual alone at 14 (layer 14 is its own source and index layer). X2 is
 * W(10,32)'s. k = 5 holds layer 14 (Engram + kv source): refused, as is every band that is not 1 <= k < t - h, one with a
 * source, a tap, or one whose X2 would change; on every cut of the model each accepted band's X1 crosses the sender's last
 * writers (crossings()' rule) and its window list is exactly the band. */
TEST(layers_p28_band_passes_of_w10_32) {
    const Geo g = flash1017();
    for (int k = 1; k <= 4; ++k) {
        Band b;
        std::string err;
        CHECK_EQ(band_crossing(g, 10, k, 32, &b, &err), 0);
        if (!err.empty()) std::fprintf(stderr, "  band k %d: %s\n", k, err.c_str());
        CHECK_EQ(b.at, 10 + k);
        CHECK_EQ(b.copy_src, 8);
        CHECK(b.copy_latent);
        CHECK(!b.copy_index);
        CHECK_EQ((int)b.window.size(), k);
        CHECK_EQ(b.window.front(), 10);
        CHECK_EQ(b.window.back(), 10 + k - 1);
        CHECK(b.x1.residual);
        if (k < 4) {
            CHECK_EQ(b.x1.src, 8);
            CHECK(b.x1.latent);
            CHECK_EQ(b.x1.sel, 8);
        } else {
            CHECK_EQ(b.x1.src, -1);
            CHECK_EQ(b.x1.sel, -1);
            CHECK_EQ(b.x1.keep, -1);
        }
    }
    Band b;
    std::string err;
    CHECK(band_crossing(g, 10, 5, 32, &b, &err) < 0);          /* layer 14: Engram + kv source */
    CHECK(err.find("layer 14") != std::string::npos);
    CHECK(band_crossing(g, 10, 0, 32, &b, &err) < 0);
    CHECK(band_crossing(g, 10, 22, 32, &b, &err) < 0);         /* the follower would keep no layer */
    CHECK_EQ(band_crossing(g, 0, 1, 20, &b, &err), 0);         /* W(0,20)'s band [0, 1): the embedding's layer, X1 the residual at 1 */
    CHECK_EQ(b.copy_src, -1);
    CHECK(band_crossing(g, 0, 2, 20, &b, &err) < 0);           /* layer 1 is Engram */
    CHECK(band_crossing(g, 34, 2, 38, &b, &err) < 0);          /* 37 is a tap (and t >= the first tap is crossings()' refusal) */
    int accepted = 0, refused_n = 0;
    for (int h = 0; h < 39; ++h)
        for (int t = h + 2; t < 40; ++t)
            for (int k = 1; h + k < t; ++k) {
                Band x;
                std::string e;
                if (band_crossing(g, h, k, t, &x, &e) < 0) { ++refused_n; CHECK(!e.empty()); continue; }
                ++accepted;
                for (int l = h; l < h + k; ++l) {
                    CHECK(!(g.is_src[(size_t)l] && g.ratio[(size_t)l] > 0));
                    CHECK(l != 1 && l != 14 && l < 37);
                }
                CHECK_EQ((int)x.window.size(), k);
                Crossing c2[2];
                CHECK_EQ(crossings(g, h, t, c2, &e), 0);
                CHECK_EQ(x.copy_src, c2[0].src);
                CHECK(x.x1.src < 0 || x.x1.src == x.copy_src || x.copy_src < 0);
            }
    std::fprintf(stderr, "  band passes over every cut: %d accepted, %d refused\n", accepted, refused_n);
    CHECK(accepted > 0 && refused_n > 0);
}

/* tp8 loop D39: the TAIL band of W(10,32) at j = 1 -- the follower stops at 31 (X2 of W(10,31): the residual, source 20's latent +
 * index keys as W(10,32)'s X2, keep 20, and index 28's selections, which layer 31 reads and W(10,32)'s X2 does not carry: the band's
 * second X2 form) and the leader's tail runs [31, 40); layer 31 holds no Engram, tap or kv source.
 * j = 2 is refused (this form), as are bands holding a source (W(10,21): layer 20), Engram (W(0,15): layer 14) or a tap, and
 * one that leaves the follower no layer; over every cut each accepted band is [t - 1, t) with X2 = W(h, t)'s besides the residual. */
TEST(layers_d39_tail_band_of_w10_32) {
    const Geo g = flash1017();
    TailBand b;
    std::string err;
    CHECK_EQ(tail_band_crossing(g, 10, 32, 1, &b, &err), 0);
    if (!err.empty()) std::fprintf(stderr, "  tail band W(10,32) j 1: %s\n", err.c_str());
    CHECK_EQ(b.at, 31);
    CHECK_EQ((int)b.window.size(), 1);
    CHECK_EQ(b.window.front(), 31);
    CHECK(b.x2.residual);
    Crossing c2[2];
    CHECK_EQ(crossings(g, 10, 32, c2, &err), 0);
    CHECK_EQ(b.x2.src, c2[1].src);
    CHECK_EQ(b.x2.latent, c2[1].latent);
    CHECK_EQ(b.x2.index, c2[1].index);
    CHECK_EQ(b.x2.sel, 28);   /* the band layer's selections: the second X2 form */
    CHECK_EQ(c2[1].sel, -1);
    CHECK_EQ(b.x2.keep, 20);
    std::fprintf(stderr, "  tail band W(10,32) j 1: X2 of W(10,31) src %d latent %d index %d sel %d keep %d\n", b.x2.src, (int)b.x2.latent,
                 (int)b.x2.index, b.x2.sel, b.x2.keep);
    CHECK(tail_band_crossing(g, 10, 32, 2, &b, &err) < 0);    /* this form: j = 1 */
    CHECK(tail_band_crossing(g, 10, 21, 1, &b, &err) < 0);    /* layer 20 is a kv source */
    CHECK(err.find("layer 20") != std::string::npos);
    CHECK(tail_band_crossing(g, 0, 15, 1, &b, &err) < 0);     /* layer 14: Engram + kv source */
    CHECK(tail_band_crossing(g, 10, 11, 1, &b, &err) < 0);    /* the follower would keep no layer */
    int accepted = 0, refused_n = 0;
    for (int h = 0; h < 39; ++h)
        for (int t = h + 2; t < 40; ++t) {
            TailBand x;
            std::string e;
            if (tail_band_crossing(g, h, t, 1, &x, &e) < 0) { ++refused_n; CHECK(!e.empty()); continue; }
            ++accepted;
            const int l = t - 1;
            CHECK(!(g.is_src[(size_t)l] && g.ratio[(size_t)l] > 0));
            CHECK(l != 1 && l != 14 && l < 37);
            CHECK_EQ(x.at, l);
            Crossing c[2];
            CHECK_EQ(crossings(g, h, t, c, &e), 0);
            CHECK_EQ(x.x2.src, c[1].src);
        }
    std::fprintf(stderr, "  tail bands over every cut: %d accepted, %d refused\n", accepted, refused_n);
    CHECK(accepted > 0 && refused_n > 0);
}

RAD_TEST_MAIN()

/* tp8 loop P41: the head band on the follower at W(10,32) -- layer 9 (no Engram, tap or kv source). A reverse pass's X1 is
 * W(9,32)'s: the residual into layer 9 with source 8's latent copy and index 8's selections, as W(10,32)'s X1 carries them;
 * a normal prompt pass's X1 is W(10,32)'s + layer 9's window keys (P28's band X1 of [9, 10)). Refused where layer h - 1 is an
 * Engram layer, a kv source or a tap (h = 2, 9, 15, 21, 38), and h < 2; over every cut each accepted band's two X1 forms carry
 * the same source / selections / keep and its X2 is W(h, t)'s. */
TEST(layers_p41_head_band_on_the_follower) {
    const Geo g = flash1017();
    RBand r;
    std::string err;
    CHECK_EQ(head_rband_crossing(g, 10, 32, &r, &err), 0);
    if (!err.empty()) std::fprintf(stderr, "  rband W(10,32): %s\n", err.c_str());
    CHECK_EQ(r.l, 9);
    CHECK_EQ(r.x1r.at, 9);
    CHECK(r.x1r.residual);
    CHECK_EQ(r.x1r.src, 8);
    CHECK(r.x1r.latent);
    CHECK_EQ(r.x1r.sel, 8);
    CHECK_EQ(r.x1k.x1.at, 10);
    CHECK_EQ(r.x1k.x1.src, r.x1r.src);
    CHECK_EQ(r.x1k.x1.sel, r.x1r.sel);
    CHECK_EQ(r.x1k.copy_src, 8);
    CHECK_EQ((int)r.x1k.window.size(), 1);
    CHECK_EQ(r.x1k.window.front(), 9);
    std::fprintf(stderr, "  rband W(10,32): X1 reverse at %d src %d latent %d sel %d keep %d; normal + window key of layer %d\n", r.x1r.at,
                 r.x1r.src, (int)r.x1r.latent, r.x1r.sel, r.x1r.keep, r.x1k.window.front());
    CHECK(head_rband_crossing(g, 9, 32, &r, &err) < 0);     /* layer 8: a kv source */
    CHECK(err.find("layer 8") != std::string::npos);
    CHECK(head_rband_crossing(g, 15, 32, &r, &err) < 0);    /* layer 14: Engram + kv source */
    CHECK(head_rband_crossing(g, 2, 20, &r, &err) < 0);     /* layer 1: Engram */
    CHECK(head_rband_crossing(g, 1, 20, &r, &err) < 0);     /* the leader would keep no layer */
    int accepted = 0, refused_n = 0;
    for (int h = 0; h < 39; ++h)
        for (int t = h + 1; t < 40; ++t) {
            RBand x;
            std::string e;
            if (head_rband_crossing(g, h, t, &x, &e) < 0) { ++refused_n; CHECK(!e.empty()); continue; }
            ++accepted;
            const int l = h - 1;
            CHECK(!(g.is_src[(size_t)l] && g.ratio[(size_t)l] > 0));
            CHECK(l != 1 && l != 14 && l < 37);
            Crossing c[2];
            std::string e2;
            CHECK_EQ(crossings(g, h, t, c, &e2), 0);
            CHECK_EQ(x.x1r.src, c[0].src);
            CHECK_EQ(x.x1r.sel, c[0].sel);
            CHECK_EQ(x.x1r.keep, c[0].keep);
            CHECK_EQ(x.x1k.x1.src, c[0].src);
        }
    CHECK(accepted > 0);
    CHECK(refused_n > 0);
    std::fprintf(stderr, "  rband over every cut: %d accepted, %d refused\n", accepted, refused_n);
}
