/* tests/ds41_sp_test.cpp -- tp8 loop P20 round 2: RADIANCE_DS41_SPHC's layer plan (arch/deepseek41/ds41_sp_plan.h), host
 * arithmetic only.
 *
 * A row-validity model of deepseek41's step_bi issue for a lone prefill pass: per buffer (the streams h0 / h1, the mixes
 * pre_attn + post + comb, pre_ffn, xn, the FFN input / MoE delta act.x, attn_out) whether every row holds the pass's value
 * on every rank (ALL) or only each rank's quarter (Q). Each op of the issue reads at its rows -- an SP op at the quarter
 * (needs Q or ALL), any other over the layer's rows (needs ALL: a cone row can be any rank's) -- in step_bi's order: the
 * crossing in (its boundary op), a PSK skip, the lazy streams' gather at a non-SP layer, the Engram gate, the in-body
 * collapse, the attention mix, xn's gather before the attention, the attention's reduce-scatter / all-reduce, pcn_f + the
 * FFN input's gather, the FFN mix, the MoE's reduce-scatter / all-reduce, the crossing out (X1 reads its live set whole;
 * a body pass's X2 carries no residual row), the next layer's pcn_a or post, and the final norm. Checked for the
 * pass shapes the two-node engine and one machine run (one machine PSK off / body / cone, the leader's head and the
 * follower at W(10,32) and W(0,20), the leader's tail): every read sees valid rows, and the plan takes the layers it
 * should (counts). Controls that must fail: the plan without its Engram rule (an in-body collapse after an SP layer
 * reads pre_ffn's quarter), and the issue without the lazy streams' gather.
 */
#include "rad_test.h"

#include "../arch/deepseek41/ds41_sp_plan.h"
#include "../arch/deepseek41/ds41_index_split.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr int kLayers = 40, kL20 = 20, kPskFirst = 21;

enum Cut { kUncut, kCone, kL20None, kSkipped };   /* a layer's rows in the pass (lever PSK) */

struct Pass {
    int lo = 0, hi = kLayers;   /* the layers this part issues [lo, hi) */
    bool recv = false;          /* a crossing in at lo (the follower's X1 at h > 0, the leader's X2 at t) */
    bool send = false;          /* X1 out after hi - 1 (the leader's head at h > 0) */
    bool x2 = false;            /* X2 out after hi - 1 (the follower) */
    bool psk = false;           /* lever PSK cuts this pass */
    bool cone = false;          /* ... and the pass holds the prompt's end (cone rows from layer 20) */
    bool x1_sh = false;         /* tp8 loop P20 round 3: X1 leaves row-sharded (each rank its rows of the live set) */
    bool x1r_sh = false;        /* round 4: X1 lands row-sharded at an SP receiving layer (each rank copies its rows) */
    bool x2_sh = false;         /* tp8 loop P36: X2 leaves the follower row-sharded (each rank its rows of the live set) */
    /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG): the follower's pass in segments of `seg` layers (0: whole), each boundary saved
     * after its last layer and restored before the next segment's first, other passes between them (every buffer rewritten);
     * the plan is the whole part's. seg_sh: an SP layer's boundary saved / restored row-sharded (2), else whole (1). */
    int seg = 0;
    bool seg_sh = false;
    /* P25b (RADIANCE_DS41_SPHC_SEG=3): layers (kL20, open_to) keep every row of the pass (its cone covers the prompt): uncut */
    int open_to = 0;
};

bool engram(int l) { return l == 1 || l == 14; }
bool tap(int l) { return l >= 37 && l <= 39; }
bool pcn_a(int l) { return l > 0 && !engram(l); }   /* the layer's collapse is issued at the previous layer's end */
Cut cut_of(const Pass& p, int l) {
    if (!p.psk || l < kL20) return kUncut;
    if (l > kL20 && l < p.open_to) return kUncut;   /* P25b */
    if (l == kL20) return p.cone ? kCone : kL20None;
    return p.cone ? kCone : kSkipped;
}

/* the plan as step_bi builds it (may_cut: psk && l >= 20) */
std::vector<uint8_t> plan_of(const Pass& p, bool engram_rule = true) {
    std::vector<ds41sp::Layer> L((size_t)kLayers);
    for (int i = p.lo; i < p.hi; ++i) {
        ds41sp::Layer& d = L[(size_t)i];
        d.issued = true;
        if (i + 1 < p.hi) {
            d.next = i + 1;
            d.next_pcn_a = pcn_a(i + 1) || !engram_rule;
        }
        const bool send_after = (p.send || p.x2) && i == p.hi - 1;
        d.out_sharded = ((p.send && p.x1_sh) || (p.x2 && p.x2_sh)) && i == p.hi - 1;   /* P36: X2 too */
        d.self_ok = !tap(i) && !(send_after && !d.out_sharded) && !(p.psk && i >= kL20 && !(i > kL20 && i < p.open_to));
    }
    std::vector<uint8_t> spv;
    ds41sp::plan(L, spv);
    return spv;
}

enum V { kNone, kQ, kAll };
struct State {
    V h0 = kAll, h1 = kNone, mix = kNone, pre_f = kAll, xn = kNone, act = kNone, attn = kNone;
};

/* the issue's row model; "" = every read valid, else the first bad read. seg_rx_rule (P25): a segment's boundary op over
 * the quarter where the layer before the boundary is SP (step_bi's seg_hsh), not only where this layer is */
std::string run(const Pass& p, const std::vector<uint8_t>& spv, bool lazy_gather = true, bool seg_rx_rule = true) {
    State s;
    State saved;   /* P25: what a boundary's save holds (validity of act.x, h1, the mixes, pre_ffn) */
    std::string err;
    const auto need = [&](V have, bool sp, const char* what, int l) {
        if (!err.empty()) return;
        if (have == kAll || (sp && have == kQ)) return;
        err = std::string(what) + " at layer " + std::to_string(l) + (sp ? " (quarter)" : " (whole)") + " reads " +
              (have == kQ ? "a quarter" : "nothing");
    };
    if (p.recv) {   /* the crossing's live set: act.x, h1, post, comb, pre_f whole on every rank (round 4: the quarter) */
        s.act = s.h1 = s.mix = s.pre_f = p.x1r_sh && spv[(size_t)p.lo] ? kQ : kAll;
        s.h0 = kNone;
    }
    bool rows_left = true;   /* the layer has rows in this pass (the final norm's) */
    for (int l = p.lo; l < p.hi; ++l) {
        const bool sp = spv[(size_t)l] != 0;
        const Cut ct = cut_of(p, l);
        if (p.seg > 0 && l > p.lo && (l - p.lo) % p.seg == 0) {   /* P25: the restore, then its boundary op */
            s = State{};
            s.h0 = s.xn = s.attn = kNone;
            s.act = saved.act; s.h1 = saved.h1; s.mix = saved.mix; s.pre_f = saved.pre_f;
            const bool rx = sp || (seg_rx_rule && spv[(size_t)l - 1] != 0);
            const Cut cp = cut_of(p, l - 1);
            if (cp == kSkipped || cp == kL20None) {
                /* the layer before ran no row past its all-row part: the boundary op has no row (and this layer none) */
            } else if (pcn_a(l)) {
                need(s.act, rx, "pcn_a (segment)", l); need(s.h1, rx, "pcn_a (segment)", l);
                need(s.mix, rx, "pcn_a (segment)", l); need(s.pre_f, rx, "pcn_a (segment)", l);
                s.h0 = s.xn = rx ? kQ : kAll;
            } else {
                need(s.act, rx, "post (segment)", l); need(s.h1, rx, "post (segment)", l);
                need(s.mix, rx, "post (segment)", l);
                s.h0 = rx ? kQ : kAll;
            }
        }
        if (p.recv && l == p.lo) {   /* the receive's boundary op */
            if (pcn_a(l)) {
                need(s.act, sp, "pcn_a (crossing)", l); need(s.h1, sp, "pcn_a (crossing)", l);
                need(s.mix, sp, "pcn_a (crossing)", l); need(s.pre_f, sp, "pcn_a (crossing)", l);
                s.h0 = s.xn = sp ? kQ : kAll;
            } else {
                need(s.act, sp, "post (crossing)", l); need(s.h1, sp, "post (crossing)", l);
                need(s.mix, sp, "post (crossing)", l);
                s.h0 = sp ? kQ : kAll;
            }
        }
        const bool seg_end = p.seg > 0 && l + 1 < p.hi && (l + 1 - p.lo) % p.seg == 0;   /* P25: a boundary after l */
        if (ct == kSkipped) {   /* no row here: the crossing out carries no residual row */
            rows_left = false;
            if (seg_end) saved = s;   /* P25: its boundary's save, whole (no SP layer is cut) */
            continue;
        }
        /* the lazy gather */
        if (lazy_gather && s.h0 == kQ && !sp && ct != kL20None) {
            s.h0 = kAll;
            s.mix = kNone;   /* the fused mix's rows forgotten: mix_a over every row below */
        }
        if (engram(l)) {   /* the gate: h0 -> h1 -> h0 */
            need(s.h0, sp, "Engram gate", l);
            s.h0 = sp ? kQ : kAll;
        }
        if (!pcn_a(l)) {   /* the in-body collapse */
            need(s.h0, sp, "cn_a", l);
            need(s.pre_f, sp, "cn_a", l);
            s.xn = sp ? kQ : kAll;
        }
        if (ct != kL20None) {   /* the attention mix (over the layer's rows) */
            need(s.h0, sp, "mix_a", l);
            s.mix = sp ? kQ : kAll;
        }
        if (s.xn == kQ) s.xn = kAll;   /* xn's gather before the attention */
        need(s.xn, false, "attention / compressor (xn)", l);
        if (ct == kL20None) {   /* the split layer with no cone row: its all-row part only */
            rows_left = false;
            if (seg_end) saved = s;   /* P25: its boundary's save, whole */
            continue;
        }
        s.attn = sp ? kQ : kAll;   /* the reduce-scatter, or the all-reduce */
        need(s.attn, sp, "pcn_f (attn_out)", l); need(s.h0, sp, "pcn_f (h0)", l); need(s.mix, sp, "pcn_f (mixes)", l);
        s.h1 = sp ? kQ : kAll;
        s.act = kAll;   /* the FFN input (gathered at SP) */
        need(s.h1, sp, "mix_f", l);
        s.mix = s.pre_f = sp ? kQ : kAll;
        need(s.act, false, "MoE input", l);
        s.act = sp ? kQ : kAll;   /* its delta reduce-scattered, or all-reduced */
        if (l == p.hi - 1 && p.send) {   /* X1: the live set whole from rank 0, or each rank its rows (row-sharded) */
            const bool q = p.x1_sh && sp;
            need(s.act, q, "X1 act.x", l); need(s.h1, q, "X1 h1", l);
            need(s.mix, q, "X1 post / comb", l); need(s.pre_f, q, "X1 pre_ffn", l);
            return err;
        }
        if (l == p.hi - 1 && p.x2) {   /* X2: the live set whole from rank 0, or each rank its rows (P36 row-sharded) */
            const bool q = p.x2_sh && sp;
            need(s.act, q, "X2 act.x", l); need(s.h1, q, "X2 h1", l);
            need(s.mix, q, "X2 post / comb", l); need(s.pre_f, q, "X2 pre_ffn", l);
            return err;
        }
        if (seg_end) {
            /* P25: the boundary's save (the boundary op's inputs, before it), then other passes rewrite every buffer: a
             * row-sharded save holds only each rank's quarter (it reads that quarter), a whole one what the rows hold */
            const bool q = p.seg_sh && sp;
            if (q) {
                need(s.act, true, "segment save act.x", l); need(s.h1, true, "segment save h1", l);
                need(s.mix, true, "segment save post / comb", l); need(s.pre_f, true, "segment save pre_ffn", l);
            }
            const auto keep = [&](V v) { return q && v != kNone ? kQ : v; };
            saved.act = keep(s.act); saved.h1 = keep(s.h1); saved.mix = keep(s.mix); saved.pre_f = keep(s.pre_f);
            continue;
        }
        if (l + 1 < p.hi) {   /* the next layer's collapse (pcn_a) or post */
            need(s.act, sp, "next pcn_a / post", l); need(s.h1, sp, "next pcn_a / post", l);
            need(s.mix, sp, "next pcn_a / post", l);
            if (pcn_a(l + 1)) {
                need(s.pre_f, sp, "next pcn_a", l);
                s.xn = sp ? kQ : kAll;
            }
            s.h0 = sp ? kQ : kAll;
        } else {   /* the node's last layer: post / the final collapse + norm over its rows */
            need(s.act, false, "final", l); need(s.h1, false, "final", l); need(s.mix, false, "final", l);
            need(s.pre_f, false, "final", l);
        }
    }
    (void)rows_left;
    return err;
}

int count(const std::vector<uint8_t>& v) {
    int n = 0;
    for (uint8_t b : v) n += b;
    return n;
}

std::vector<Pass> shapes() {
    std::vector<Pass> v;
    Pass a;   /* one machine, PSK off: every layer's rows */
    v.push_back(a);
    a.psk = true;   /* one machine, a body chunk (no cone row) */
    v.push_back(a);
    a.cone = true;   /* one machine, the prompt's end */
    v.push_back(a);
    Pass h;   /* the leader's head at W(10,32): layers 0-9, X1 after 9 */
    h.hi = 10; h.send = true; h.psk = true;
    v.push_back(h);
    Pass f;   /* the follower at W(10,32): X1 in at 10, X2 out after 31 */
    f.lo = 10; f.hi = 32; f.recv = true; f.x2 = true; f.psk = true;
    v.push_back(f);
    f.cone = true;
    v.push_back(f);
    Pass f0;   /* the follower at W(0,20): layers 0-19 from X1 = the embedding (no crossing live set: the prologue) */
    f0.hi = 20; f0.x2 = true; f0.psk = true;
    v.push_back(f0);
    Pass t;   /* the leader's tail at W(10,32): X2 in at 32, layers 32-39 (cone rows) */
    t.lo = 32; t.recv = true; t.psk = true; t.cone = true;
    v.push_back(t);
    Pass hs = h;   /* the leader's head at W(10,32) with X1 row-sharded (round 3) */
    hs.x1_sh = true;
    v.push_back(hs);
    Pass fq = f;   /* the follower at W(10,32), X1 landing row-sharded (round 4), body and cone */
    fq.x1r_sh = true;
    fq.cone = false;
    v.push_back(fq);
    fq.cone = true;
    v.push_back(fq);
    Pass f15;   /* W(15,35)'s follower: no Engram layer, X1 in at 15 */
    f15.lo = 15; f15.hi = 35; f15.recv = true; f15.x2 = true; f15.psk = true; f15.cone = true;
    v.push_back(f15);
    return v;
}

}  // namespace

TEST(plan_every_read_valid_on_every_pass_shape) {
    for (const Pass& p : shapes()) {
        const std::vector<uint8_t> spv = plan_of(p);
        const std::string e = run(p, spv);
        if (!e.empty()) std::fprintf(stderr, "pass [%d, %d) psk %d cone %d: %s\n", p.lo, p.hi, p.psk, p.cone, e.c_str());
        CHECK_EQ(e, std::string(""));
    }
}

TEST(plan_takes_the_layers_it_should) {
    const std::vector<Pass> v = shapes();
    /* one machine PSK off: 0..36 (37-39 tap, 39 the last) */
    CHECK_EQ(count(plan_of(v[0])), 37);
    /* one machine PSK: 0..19 (20 the split layer, 21.. cut) */
    CHECK_EQ(count(plan_of(v[1])), 20);
    CHECK_EQ(count(plan_of(v[2])), 20);
    /* the leader's head at W(10,32): 0..8 (9 sends X1) */
    CHECK_EQ(count(plan_of(v[3])), 9);
    /* the follower at W(10,32): 10..19, body and cone */
    CHECK_EQ(count(plan_of(v[4])), 10);
    CHECK_EQ(count(plan_of(v[5])), 10);
    /* W(0,20)'s follower: 0..18 (19 sends X2) */
    CHECK_EQ(count(plan_of(v[6])), 19);
    /* the leader's tail: cone layers, none */
    CHECK_EQ(count(plan_of(v[7])), 0);
    /* the leader's head with X1 row-sharded: 0..9 (no gather before X1) */
    CHECK_EQ(count(plan_of(v[8])), 10);
    /* the follower with X1 landing row-sharded: 10..19 */
    CHECK_EQ(count(plan_of(v[9])), 10);
    CHECK_EQ(count(plan_of(v[10])), 10);
    /* W(15,35)'s follower: 15..19 */
    CHECK_EQ(count(plan_of(v[11])), 5);
}

TEST(plan_control_without_the_engram_rule_fails) {
    /* an Engram layer that is not SP after an SP one: here the leader's head cut at layer 2 (layer 1 = the last before X1,
     * Engram, not SP; layer 0 SP only if the rule is dropped) */
    Pass h;
    h.hi = 2; h.send = true; h.psk = true;
    CHECK_EQ(count(plan_of(h)), 0);                 /* the rule: layer 0 stays whole */
    CHECK_EQ(run(h, plan_of(h)), std::string(""));
    const std::vector<uint8_t> bad = plan_of(h, /*engram_rule=*/false);
    CHECK_EQ(count(bad), 1);
    CHECK(!run(h, bad).empty());                    /* layer 1's in-body collapse reads pre_ffn's quarter */
    /* and the follower whose last layer before X2 is Engram 14 (W(10,15)) */
    Pass f;
    f.lo = 10; f.hi = 15; f.recv = true; f.x2 = true; f.psk = true;
    CHECK_EQ(run(f, plan_of(f)), std::string(""));
    CHECK(!run(f, plan_of(f, false)).empty());
}

/* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG): the follower at W(10,32) and W(15,35) in one-layer (and two-layer) segments, the
 * boundaries whole (1) and row-sharded (2), X1 whole and row-sharded, body and cone: every read valid and the whole part's
 * plan (10 / 5 layers) -- the segmented issue is the whole pass's with a save / restore at each boundary. Control: the
 * boundary op taken over every row where only the layer before the boundary is SP (19 -> 20 at W(10,32)) reads a quarter. */
TEST(plan_segment_passes_every_read_valid) {
    int checked = 0;
    for (Pass f : { shapes()[4], shapes()[5], shapes()[9], shapes()[10], shapes()[11] })
        for (int seg : { 1, 2 })
            for (bool sh : { false, true }) {
                f.seg = seg;
                f.seg_sh = sh;
                const std::vector<uint8_t> spv = plan_of(f);
                const std::string e = run(f, spv);
                if (!e.empty())
                    std::fprintf(stderr, "segments of %d [%d, %d) sh %d x1r %d cone %d: %s\n", seg, f.lo, f.hi, sh, f.x1r_sh,
                                 f.cone, e.c_str());
                CHECK_EQ(e, std::string(""));
                CHECK_EQ(count(spv), f.lo == 15 ? 5 : 10);
                ++checked;
            }
    CHECK_EQ(checked, 20);
    Pass f = shapes()[4];
    f.seg = 1;
    for (bool sh : { false, true }) {
        f.seg_sh = sh;
        CHECK(!run(f, plan_of(f), true, /*seg_rx_rule=*/false).empty());
    }
}

/* tp8 loop D40 (RADIANCE_DS41_SPHC_LSEG=1, fn4 pp_lead_seg + 512): the LEADER's head at W(10,32) in LSP's segments -- every 1 / 2 /
 * 4 / 5 layers (pp_lead_seg strides 1, 2, 4 and 5 over the declared boundaries 1..9), the boundaries saved / restored whole, X1
 * whole and row-sharded: every read valid and the whole head's plan (9 / 10 layers) in every form -- the segmented issue is the
 * whole head's with a save / restore at each boundary (P25's rule on the leader). Control: the boundary op taken over every row
 * where only the layer before the boundary is SP (X1 whole: boundary 9) reads a quarter. */
TEST(plan_leader_head_segment_passes_every_read_valid) {
    int checked = 0;
    for (Pass h : { shapes()[3], shapes()[8] })
        for (int seg : { 1, 2, 4, 5 }) {
            h.seg = seg;
            h.seg_sh = false;
            const std::vector<uint8_t> spv = plan_of(h);
            const std::string e = run(h, spv);
            if (!e.empty()) std::fprintf(stderr, "leader head in segments of %d, x1_sh %d: %s\n", seg, h.x1_sh, e.c_str());
            CHECK_EQ(e, std::string(""));
            CHECK_EQ(count(spv), h.x1_sh ? 10 : 9);
            ++checked;
        }
    CHECK_EQ(checked, 8);
    /* the control needs a boundary whose layer is whole after an SP one: X1 whole, boundary 9 (stride 1; layer 9 sends X1) */
    Pass h = shapes()[3];
    h.seg = 1;
    CHECK(!run(h, plan_of(h), true, /*seg_rx_rule=*/false).empty());
    h = shapes()[8];   /* X1 row-sharded: every head layer SP, the rule changes nothing */
    h.seg = 2;
    CHECK(run(h, plan_of(h), true, /*seg_rx_rule=*/false).empty());
}

/* tp8 loop P25b (RADIANCE_DS41_SPHC_SEG=3): the follower's newcomer pass whose cone keeps every row of layers 21-31 (a ~1.1k
 * prompt: C_31 = 160 + 8 x 128 = 1,184) or of 21-27 only (a ~1.6k prompt: C_28 = 1,568), in 1- / 2-layer segments, boundaries whole /
 * sharded, X1 whole / sharded: every read valid; the plan takes 10-19 + 21-30 (20) or 10-19 + 21-27 (17) -- 20 (the split layer)
 * and 31 (before X2) stay whole. */
TEST(plan_segment_passes_with_uncut_deep_layers_every_read_valid) {
    int checked = 0;
    for (int open_to : { 32, 28 })
        for (Pass f : { shapes()[5], shapes()[10] })
            for (int seg : { 1, 2 })
                for (bool sh : { false, true }) {
                    f.seg = seg;
                    f.seg_sh = sh;
                    f.open_to = open_to;
                    const std::vector<uint8_t> spv = plan_of(f);
                    const std::string e = run(f, spv);
                    if (!e.empty())
                        std::fprintf(stderr, "open to %d, segments of %d, sh %d x1r %d: %s\n", open_to, seg, sh, f.x1r_sh, e.c_str());
                    CHECK_EQ(e, std::string(""));
                    CHECK_EQ(count(spv), open_to == 32 ? 20 : 17);
                    CHECK_EQ((int)spv[20], 0);
                    CHECK_EQ((int)spv[31], 0);
                    ++checked;
                }
    CHECK_EQ(checked, 16);
}

/* tp8 loop P25c (RADIANCE_DS41_SPHC_SEG=4): the same uncut deep layers in the follower's LONE prefill pass (a cold prompt's last
 * chunk, no segment: X1 in at 10, X2 out after 31), X1 whole / row-sharded: every read valid; 10-19 + 21-30 (20) or 10-19 + 21-27
 * (17) SP, 20 and 31 whole; control: the same pass with P25c off (open_to 0) takes 10-19 only (P20's lone pass). The 16k cell's
 * chunk 7 (1,441 rows, the tail snapshot 417 rows before the end): C_l + 417 >= 1,441 through layer 31 -> open to 32. */
TEST(plan_lone_passes_with_uncut_deep_layers_every_read_valid) {
    int checked = 0;
    for (int open_to : { 32, 28 })
        for (Pass f : { shapes()[5], shapes()[10] }) {
            f.seg = 0;
            f.open_to = open_to;
            const std::vector<uint8_t> spv = plan_of(f);
            const std::string e = run(f, spv);
            if (!e.empty()) std::fprintf(stderr, "lone, open to %d, x1r %d: %s\n", open_to, f.x1r_sh, e.c_str());
            CHECK_EQ(e, std::string(""));
            CHECK_EQ(count(spv), open_to == 32 ? 20 : 17);
            CHECK_EQ((int)spv[20], 0);
            CHECK_EQ((int)spv[31], 0);
            for (int l = 10; l < 20; ++l) CHECK_EQ((int)spv[(size_t)l], 1);
            f.open_to = 0;
            CHECK_EQ(count(plan_of(f)), 10);
            CHECK_EQ(run(f, plan_of(f)), std::string(""));
            ++checked;
        }
    CHECK_EQ(checked, 4);
    /* the lone pass without the lazy gather (the streams whole again at the split layer 20 after SP 19, and at 31 after 30) */
    Pass f = shapes()[5];
    f.open_to = 32;
    CHECK(!run(f, plan_of(f), /*lazy_gather=*/false).empty());
}

/* tp8 loop P36 (RADIANCE_DS41_SPHC_X2=1): X2 leaves the follower row-sharded, so the plan may take layer 31 too -- the segment
 * and lone follower passes of P25b / P25c with every deep layer uncut (open to 32), X1 whole / landing sharded, boundaries whole /
 * sharded: every read valid and 31 SP (21 = 10-19 + 21-31; 20, the split layer, whole); open to 28 (31 cut by PSK) keeps 31
 * whole (17, as without P36); a body pass (PSK cuts 20-31 to nothing) is unchanged (10 SP). Control: X2 whole after an SP 31
 * (the shard asked for but the plan's layer read whole) is a bad read. */
TEST(plan_x2_row_sharded_takes_layer_31_every_read_valid) {
    int checked = 0;
    for (int open_to : { 32, 28 })
        for (Pass f : { shapes()[5], shapes()[10] })
            for (int seg : { 0, 1, 2 })
                for (bool sh : { false, true }) {
                    f.seg = seg;
                    f.seg_sh = sh;
                    f.open_to = open_to;
                    f.x2_sh = true;
                    const std::vector<uint8_t> spv = plan_of(f);
                    const std::string e = run(f, spv);
                    if (!e.empty())
                        std::fprintf(stderr, "P36 open to %d, segments of %d, sh %d x1r %d: %s\n", open_to, seg, sh, f.x1r_sh, e.c_str());
                    CHECK_EQ(e, std::string(""));
                    CHECK_EQ(count(spv), open_to == 32 ? 21 : 17);
                    CHECK_EQ((int)spv[20], 0);
                    CHECK_EQ((int)spv[31], open_to == 32 ? 1 : 0);
                    ++checked;
                }
    CHECK_EQ(checked, 24);
    Pass b = shapes()[4];   /* a body pass: no cone row past 20 */
    b.x2_sh = true;
    CHECK_EQ(count(plan_of(b)), 10);
    CHECK_EQ(run(b, plan_of(b)), std::string(""));
    /* control: the plan made 31 SP for a sharded X2, the issue sent it whole */
    Pass f = shapes()[5];
    f.open_to = 32;
    f.x2_sh = true;
    const std::vector<uint8_t> spv = plan_of(f);
    f.x2_sh = false;
    CHECK(!run(f, spv).empty());
}

TEST(plan_control_without_the_lazy_gather_fails) {
    int failed = 0;
    for (const Pass& p : shapes()) failed += !run(p, plan_of(p), /*lazy_gather=*/false).empty();
    CHECK(failed >= 4);   /* every shape with a non-SP layer reading the streams after SP ones */
}

/* ---- tp8 loop P47 (RADIANCE_DS41_CAND_SPLIT): the candidate layer's indexer split where its keep words cross, its keep rows
 * all-gathered (deepseek41 step_bi / the plain issue: index_candidates into the rank's keep_part at (p - p0) R, then keep_gather
 * lays rank r's q R rows at s.r0 + r q R). A model of the keep rows, word = the row's own value: the unsplit layer (today's
 * form where keep crosses: every rank every run not skipped by PSK) against the split + gather, at 4 ranks, for every deal
 * the issue takes (irows 96 / 256, plain / balanced, the declared run ops), segment starts 0 / 48 (a BI step's decode rows
 * first), and PSK cut rows at the candidate layer, a reindex layer and X2's boundary (cones shrink with the layer). Checked:
 * every row a later reader takes -- a reindex layer's runs under the same deal (not skipped by its cut) on EVERY rank, X2's live
 * rows [lo, end) on rank 0 -- holds the same word in both forms; the gather stays inside keep's declared rows (sel's rows) and
 * the part / numel declared; rows before the segment are untouched. Controls that must fail: no gather (rank-local keep, today's
 * split at a layer whose keep crosses), and the part laid at another rank's offset (ceil(T / world) rows a rank). */
namespace p47 {

constexpr int kWorld = 4;
constexpr int64_t kStale = -7;   /* a word no run wrote */

struct Deal { int64_t irows, max_tok, live; bool balance; };

int64_t qmax_of(const Deal& d) {
    const int64_t iparts = (d.max_tok + d.irows - 1) / d.irows;
    return (iparts + kWorld - 1) / kWorld;
}

/* The keep rows each rank holds after the candidate layer; `form` 0 = unsplit (every rank, every run), 1 = split + gather,
 * 2 = split without the gather (control), 3 = split, the part at r ceil(T / world) (control). Returns false when a gather
 * leaves the declared rows (keep_rows) or the part (qmax irows). */
bool candidate_layer(const Deal& d, int64_t s_r0, int64_t T, int64_t cut0, int form, int64_t keep_rows,
                     std::vector<std::vector<int64_t>>& keep) {
    keep.assign(kWorld, std::vector<int64_t>((size_t)keep_rows, kStale));
    const int64_t qm = qmax_of(d), iparts = (d.max_tok + d.irows - 1) / d.irows;
    std::vector<std::vector<int64_t>> part(kWorld, std::vector<int64_t>((size_t)(qm * d.irows), kStale));
    std::vector<Ds41IdxRuns> rus(kWorld);
    for (int r = 0; r < kWorld; ++r) {
        const Ds41IdxRuns ru = ds41_idx_runs(T, d.irows, d.live, kWorld, r, form != 0, kDs41SplitMinLive,
                                             form != 0 && d.balance, iparts);
        rus[(size_t)r] = ru;
        for (int64_t p = ru.p0; p < ru.p1; ++p) {
            const int64_t q0 = p * ru.R, qn = std::min(ru.R, T - q0);
            if (ds41_idx_run_skipped(s_r0, q0, qn, cut0)) continue;
            for (int64_t i = 0; i < qn; ++i) {
                if (ru.split && form != 2) {
                    const int64_t at = (p - ru.p0) * ru.R + i;
                    if (at >= (int64_t)part[(size_t)r].size()) return false;
                    part[(size_t)r][(size_t)at] = s_r0 + q0 + i;
                } else {
                    keep[(size_t)r][(size_t)(s_r0 + q0 + i)] = s_r0 + q0 + i;
                }
            }
        }
    }
    if (form == 1 || form == 3) {
        const Ds41IdxRuns& r0 = rus[0];
        if (!r0.split) return true;
        const int64_t qR = r0.q * r0.R;
        if (qR > qm * d.irows) return false;   /* the issued numel against the declared one */
        const int64_t off = form == 1 ? qR : (T + kWorld - 1) / kWorld;
        if (s_r0 + (int64_t)kWorld * qR > keep_rows) return false;
        for (int dst = 0; dst < kWorld; ++dst)
            for (int r = 0; r < kWorld; ++r)
                for (int64_t j = 0; j < qR && s_r0 + r * off + j < keep_rows; ++j)
                    keep[(size_t)dst][(size_t)(s_r0 + r * off + j)] = part[(size_t)r][(size_t)j];
    }
    return true;
}

/* The rows a reader takes: a reindex layer (cut c) reads its runs' rows on every rank under the same deal; X2 reads [lo, end)
 * on rank 0. Returns the number of reads whose word differs from the row's own value. */
int64_t bad_reads(const Deal& d, int64_t s_r0, int64_t T, int64_t cut_reindex, int64_t x2_lo,
                  const std::vector<std::vector<int64_t>>& keep, bool split) {
    int64_t bad = 0;
    const int64_t iparts = (d.max_tok + d.irows - 1) / d.irows;
    for (int r = 0; r < kWorld; ++r) {
        const Ds41IdxRuns ru = ds41_idx_runs(T, d.irows, d.live, kWorld, r, split, kDs41SplitMinLive, split && d.balance, iparts);
        for (int64_t p = ru.p0; p < ru.p1; ++p) {
            const int64_t q0 = p * ru.R, qn = std::min(ru.R, T - q0);
            if (ds41_idx_run_skipped(s_r0, q0, qn, cut_reindex)) continue;
            for (int64_t i = 0; i < qn; ++i) bad += keep[(size_t)r][(size_t)(s_r0 + q0 + i)] != s_r0 + q0 + i;
        }
    }
    for (int64_t t = x2_lo; t < s_r0 + T; ++t) bad += keep[0][(size_t)t] != t;
    return bad;
}

}   // namespace p47

TEST(p47_candidate_split_keep_gather_holds_every_read_row_as_unsplit) {
    using namespace p47;
    long long cases = 0, split_cases = 0, gathered_rows = 0;
    const int64_t cones[4] = { 2592, 2080, 1568, 1056 };   /* C_l = 160 + (39 - l) 128 at layers 20 / 24 / 28 / 32 */
    for (const int64_t irows : { 96, 256 })
        for (const bool balance : { false, true })
            for (const int64_t s_r0 : { 0, 48 })
                for (int64_t T = 97; T <= 4160 - s_r0; T += (T < 600 ? 37 : 251)) {
                    const Deal d{ irows, 4160, 70000, balance };
                    const int64_t qm = qmax_of(d);
                    /* keep's rows under the env = sel's rows (BI / PSK declared) */
                    const int64_t keep_rows = std::max<int64_t>(4160, kWorld * qm * irows) + kWorld * irows + 64;
                    for (int ahead = 0; ahead < 3; ++ahead) {
                        /* the cut rows: the pass's end minus each layer's cone (ahead > 0: a second-to-last pass, its cone
                         * rows shifted by a target past its end, as design 2 plans them; clamped to the segment) */
                        const int64_t end = s_r0 + T, shift = ahead * 640;
                        int64_t c[4];
                        for (int k = 0; k < 4; ++k) c[k] = std::min(end - 1, std::max(s_r0, end + shift - cones[k]));
                        std::vector<std::vector<int64_t>> ref, got;
                        CHECK(candidate_layer(d, s_r0, T, c[0], 0, keep_rows, ref));
                        CHECK(candidate_layer(d, s_r0, T, c[0], 1, keep_rows, got));
                        const Ds41IdxRuns ru = ds41_idx_runs(T, irows, d.live, kWorld, 0, true, kDs41SplitMinLive, balance,
                                                             (4160 + irows - 1) / irows);
                        for (int k = 1; k < 4; ++k) {
                            CHECK_EQ(bad_reads(d, s_r0, T, c[k], c[3], ref, true), 0);
                            CHECK_EQ(bad_reads(d, s_r0, T, c[k], c[3], got, true), 0);
                        }
                        for (int r = 0; r < kWorld; ++r)
                            for (int64_t t = 0; t < s_r0; ++t) CHECK_EQ(got[(size_t)r][(size_t)t], kStale);
                        ++cases;
                        if (ru.split) {
                            ++split_cases;
                            gathered_rows += kWorld * ru.q * ru.R;
                        }
                    }
                }
    CHECK(split_cases > 100);
    fprintf(stderr, "p47: %lld cases (%lld split), %lld rows gathered\n", cases, split_cases, gathered_rows);
}

TEST(p47_control_without_the_gather_or_at_another_offset_fails) {
    using namespace p47;
    const Deal d{ 96, 4160, 70000, true };
    const int64_t qm = qmax_of(d), keep_rows = std::max<int64_t>(4160, kWorld * qm * 96) + kWorld * 96 + 64;
    int failed_local = 0, failed_off = 0, n = 0;
    for (int64_t T : { 1441, 2048, 2162, 3001 }) {
        const int64_t end = T, c0 = std::max<int64_t>(0, end - 2592), c3 = std::max<int64_t>(0, end - 1056),
                      c1 = std::max<int64_t>(0, end - 2080);
        std::vector<std::vector<int64_t>> local, off;
        CHECK(candidate_layer(d, 0, T, c0, 2, keep_rows, local));
        CHECK(candidate_layer(d, 0, T, c0, 3, keep_rows, off));
        failed_local += bad_reads(d, 0, T, c1, c3, local, true) > 0;
        failed_off += bad_reads(d, 0, T, c1, c3, off, true) > 0;
        ++n;
    }
    CHECK_EQ(failed_local, n);   /* rank 0's X2 rows include other ranks' runs */
    CHECK(failed_off >= 3);      /* the part at ceil(T / 4) a rank lands rows off their place */
}

TEST(p47_keep_pitch_pads_to_whole_16_byte_words_only_when_asked) {
    CHECK_EQ(ds41_keep_pitch(4097, false), 4097);   /* 1M context, block 8: 16,388 B a row as X2 carries it today */
    CHECK_EQ(ds41_keep_pitch(4097, true), 4100);
    CHECK_EQ(ds41_keep_pitch(2048, true), 2048);
    CHECK_EQ(ds41_keep_pitch(0, true), 0);
    for (int64_t k = 1; k < 64; ++k) {
        const int64_t p = ds41_keep_pitch(k, true);
        CHECK(p >= k && p < k + 4 && (p * 4) % 16 == 0);
    }
    CHECK(ds41_idx_run_skipped(100, 0, 50, 150));
    CHECK(!ds41_idx_run_skipped(100, 0, 51, 150));
}

RAD_TEST_MAIN()
