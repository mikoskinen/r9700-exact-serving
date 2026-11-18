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
};

bool engram(int l) { return l == 1 || l == 14; }
bool tap(int l) { return l >= 37 && l <= 39; }
bool pcn_a(int l) { return l > 0 && !engram(l); }   /* the layer's collapse is issued at the previous layer's end */
Cut cut_of(const Pass& p, int l) {
    if (!p.psk || l < kL20) return kUncut;
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
        d.out_sharded = p.send && p.x1_sh && i == p.hi - 1;
        d.self_ok = !tap(i) && !(send_after && !d.out_sharded) && !(p.psk && i >= kL20);
    }
    std::vector<uint8_t> spv;
    ds41sp::plan(L, spv);
    return spv;
}

enum V { kNone, kQ, kAll };
struct State {
    V h0 = kAll, h1 = kNone, mix = kNone, pre_f = kAll, xn = kNone, act = kNone, attn = kNone;
};

/* the issue's row model; "" = every read valid, else the first bad read */
std::string run(const Pass& p, const std::vector<uint8_t>& spv, bool lazy_gather = true) {
    State s;
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
        if (ct == kSkipped) {   /* no row here: the crossing out carries no residual row */
            rows_left = false;
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
        if (l == p.hi - 1 && p.x2) {   /* X2 at a cone pass: the cone rows' live set whole */
            need(s.act, false, "X2 act.x", l); need(s.h1, false, "X2 h1", l);
            need(s.mix, false, "X2 post / comb", l); need(s.pre_f, false, "X2 pre_ffn", l);
            return err;
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

TEST(plan_control_without_the_lazy_gather_fails) {
    int failed = 0;
    for (const Pass& p : shapes()) failed += !run(p, plan_of(p), /*lazy_gather=*/false).empty();
    CHECK(failed >= 4);   /* every shape with a non-SP layer reading the streams after SP ones */
}

RAD_TEST_MAIN()
