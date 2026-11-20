/* tests/pp_switch_test.cpp -- the two-node pipeline's runtime switches (tp8 S4, review F2 / F3;
 * core/runtime/fn4_switch.h): what the leader ships, what the follower takes, and how a request for a
 * lever only the other node declares is judged.
 *
 *   follow     the follower requests only the switches that act on its own passes or placement, each
 *              once per leader request; a value it refuses (a lever it does not declare) is skipped
 *              and named, never a failure -- the published switch file's 49 non-zero values included,
 *              in a drafter-less node's registration state (the review's pp_fn4_follower_test scenario)
 *   requests   apply_pending counts every request with its value as asked
 *   peer       on the leader, a plugin switch it refuses is taken when the follower takes it (the
 *              leader holds no Engram layer: ds41_e2 is the follower's); both refusing is a refusal
 *              naming both; a core switch is never asked about
 * The process loads no plugin, so every plugin switch is "no loaded plugin serves it" here. */
#include "rad_test.h"

#include "runtime/fn4_switch.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace rad;

/* The published DeepSeek switches.json's non-zero values (hero2/raw/ds-h210/run-used/deepseek, the 49
 * the review counted; ds41_psk / ds41_bi as s4-run.sh sets them, 0). */
const std::map<std::string, int>& published() {
    static const std::map<std::string, int> m = {
        { "gg_gemv", 2 }, { "aql_acq", 2 }, { "ds41_kvfold", 1 }, { "ds41_ofold", 1 }, { "ds41_onefill", 1 },
        { "draft_adaptive", 1 }, { "draft_depth0", 3 }, { "draft_hybrid_match", 8 }, { "drafter_mx", 1 },
        { "gg_pf", 2 }, { "f32_pf", 25 }, { "dspark_walk", 1 }, { "ds41_argather", 48 }, { "idx_split", 1 },
        { "ds41_qnfold", 1 }, { "ds41_swfold", 1 }, { "draft_ctxskip", 1 }, { "dspark_conf", 1 },
        { "dspark_conf_rows", 1 }, { "draft_lend", 1 }, { "draft_probe_backoff", 3 }, { "draft_hybrid", 1 },
        { "draft_occ_k", 32 }, { "ds41_psk_k", 8 }, { "draft_occ", 13 }, { "sched_bi_chunk", 3 },
        { "sched_noa", 2 }, { "dspark_pfctx", 1 }, { "ds41_engdd", 1 }, { "mhc_fuse", 1 },
        { "sched_row_cap", 48 }, { "ds41_gq", 3 }, { "ds41_e2", 1 }, { "sched_cb_hold_k", 2 },
        { "exl3_x3d", 2 }, { "ar_xpush", 1 }, { "gg_gemv_m16", 1 }, { "place_swap_lent", 1 },
        { "heat_min_gain_x100", 75 }, { "heat_hyst_x100", 125 }, { "ds41_rfuse", 2 }, { "route_copy_ov", 1 },
        { "ds41_wmk", 2 }, { "ds41_moe1", 1 }, { "sched_tail_grid", 512 }, { "draft_dcap", 3 },
        { "exl3_dng", 2 }, { "ds41_f32dec", 1 }, { "exl3_xr", 3 },
    };
    return m;
}

bool scheduler_or_drafter(const std::string& n) {
    return n.rfind("draft_", 0) == 0 || n.rfind("dspark_conf", 0) == 0 || n.rfind("sched_", 0) == 0 ||
           n == "arena_d0" || n == "ds41_psk_k" || n == "stage_short" || n == "stage_pick" ||
           n == "stage_full_min" || n == "stage_full_pick" || n == "dspark_alt";
}

TEST(the_follower_takes_only_its_own_switches_and_never_fails_on_a_refusal) {
    /* A drafter-less node, as a follower registers itself (scheduler.cpp, engine_draft.cpp). */
    fn4::draft_register(false, false);
    fn4::hybrid_register(false);
    fn4::dspark_register(false, false, 0, 0);
    /* The leader's request record: one request each of the published values. */
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    for (const auto& [n, v] : published()) {
        const int i = fn4::switch_index(n);
        REQUIRE(i >= 0);
        count[(size_t)i] = 1;
        value[(size_t)i] = v;
    }
    std::vector<std::string> skipped;
    const int asked = fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped);
    /* Every request is marked taken, whether or not this node acts on it. */
    for (int i = 0; i < fn4::kSwitches; ++i) CHECK_EQ(taken[(size_t)i], count[(size_t)i]);
    /* No scheduler or drafter switch is requested on the follower -- in particular none of the review's
     * refused draft_hybrid / dspark_conf / dspark_conf_rows -- and none is even named as skipped. */
    for (const std::string& s : skipped) {
        const std::string n = s.substr(0, s.find('='));
        if (scheduler_or_drafter(n)) fprintf(stderr, "    asked for a leader-only switch: %s\n", s.c_str());
        CHECK(!scheduler_or_drafter(n));
    }
    int follower_kind = 0;
    for (const auto& [n, v] : published()) follower_kind += fn4::follower_switch(fn4::switch_index(n)) ? 1 : 0;
    fprintf(stderr, "    %zu published values: %d act on a follower's passes / placement; %d requested, %zu skipped "
            "(no plugin in this process; placement unregistered)\n", published().size(), follower_kind, asked,
            skipped.size());
    CHECK_EQ(asked + (int)skipped.size(), follower_kind);
    CHECK(fn4::follower_switch(fn4::switch_index("ds41_e2")));
    CHECK(fn4::follower_switch(fn4::switch_index("aql_acq")));
    CHECK(!fn4::follower_switch(fn4::switch_index("draft_hybrid")));
    CHECK(!fn4::follower_switch(fn4::switch_index("dspark_conf")));
    CHECK(!fn4::follower_switch(fn4::switch_index("sched_noa")));
    /* aql_acq is the core's and acts on every pass: requested and applied. */
    int sw[fn4::kSwitches];
    (void)fn4::apply_pending(1, sw);
    CHECK_EQ(fn4::value(fn4::kAqlAcq), 2);
    /* The same record again: nothing new is requested. */
    skipped.clear();
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 0);
    CHECK(skipped.empty());
    /* A second request of the same switch (the count advances) is taken again. */
    const int acq = fn4::kAqlAcq;
    count[(size_t)acq] = 2;
    value[(size_t)acq] = 1;
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 1);
    (void)fn4::apply_pending(2, sw);
    CHECK_EQ(fn4::value(acq), 1);
    /* tp8 loop U1: aql_hdp (the HDP flush in front of every doorbell) is the core's on both nodes: the leader's request is
     * taken and applied here, and its undo too. */
    const int hdp = fn4::kAqlHdp;
    CHECK(fn4::follower_switch(hdp));
    CHECK_EQ(fn4::aql_hdp(), 0);
    count[(size_t)hdp] = 1;
    value[(size_t)hdp] = 1;
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 1);
    (void)fn4::apply_pending(3, sw);
    CHECK_EQ(fn4::value(hdp), 1);
    CHECK_EQ(fn4::aql_hdp(), 1);
    count[(size_t)hdp] = 2;
    value[(size_t)hdp] = 0;
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 1);
    (void)fn4::apply_pending(4, sw);
    CHECK_EQ(fn4::aql_hdp(), 0);
}

TEST(apply_pending_records_each_request_as_asked) {
    std::vector<uint32_t> n0(fn4::kSwitches), n1(fn4::kSwitches);
    std::vector<int32_t> v0(fn4::kSwitches), v1(fn4::kSwitches);
    fn4::requests(n0.data(), v0.data());
    std::string why;
    CHECK(fn4::request("aql_acq", 2, &why));
    CHECK(fn4::request("route_copy_ov", 1, &why));
    int sw[fn4::kSwitches];
    (void)fn4::apply_pending(3, sw);
    fn4::requests(n1.data(), v1.data());
    CHECK_EQ(n1[(size_t)fn4::kAqlAcq], n0[(size_t)fn4::kAqlAcq] + 1);
    CHECK_EQ(v1[(size_t)fn4::kAqlAcq], 2);
    CHECK_EQ(n1[(size_t)fn4::kRouteCopyOv], n0[(size_t)fn4::kRouteCopyOv] + 1);
    for (int i = 0; i < fn4::kSwitches; ++i)
        if (i != fn4::kAqlAcq && i != fn4::kRouteCopyOv) CHECK_EQ(n1[(size_t)i], n0[(size_t)i]);
    /* A plugin switch nothing here serves is refused before it is queued: not counted. */
    CHECK(!fn4::request("ds41_e2", 1, &why));
    fn4::requests(n0.data(), v0.data());
    CHECK_EQ(n0[(size_t)fn4::kDs41E2], n1[(size_t)fn4::kDs41E2]);
}

TEST(the_leader_takes_a_plugin_switch_the_follower_serves_and_refuses_one_neither_takes) {
    int asks = 0;
    std::string last;
    fn4::set_peer_check([&](const std::string& n, int v, std::string* why) {
        ++asks;
        last = n + "=" + std::to_string(v);
        if (n == "ds41_e2" && v == 1) return 1;          /* the follower holds the Engram layers */
        *why = n + "=" + std::to_string(v) + " refused: the plugin that serves it does not know that value";
        return -1;
    });
    std::string why;
    /* This node (no plugin) refuses ds41_e2; the follower takes it: the pipeline does. */
    CHECK(fn4::check("ds41_e2", 1, &why));
    CHECK_EQ(asks, 1);
    CHECK(last == "ds41_e2=1");
    /* Neither takes it: refused, both reasons named. */
    CHECK(!fn4::check("gg_gemv", 7, &why));
    CHECK(why.find("no loaded plugin serves it") != std::string::npos);
    CHECK(why.find("the pipeline's other node refuses it too") != std::string::npos);
    CHECK_EQ(asks, 2);
    /* A core switch is the leader's own verdict: no ask. */
    CHECK(fn4::check("aql_acq", 1, &why));
    CHECK(!fn4::check("aql_acq", 9, &why));
    CHECK_EQ(asks, 2);
    /* A follower that does not answer (the link down, a timeout): refused, saying so. */
    fn4::set_peer_check([&](const std::string&, int, std::string* why) { *why = "no answer"; return 0; });
    CHECK(!fn4::check("ds41_e2", 1, &why));
    CHECK(why.find("did not answer") != std::string::npos);
    /* Requested through the pipeline: queued and counted on the leader, whose plugin then refuses
     * it at apply (named in its log); the follower takes it from the request record. */
    fn4::set_peer_check([&](const std::string&, int, std::string*) { return 1; });
    std::vector<uint32_t> n0(fn4::kSwitches), n1(fn4::kSwitches);
    std::vector<int32_t> v0(fn4::kSwitches), v1(fn4::kSwitches);
    fn4::requests(n0.data(), v0.data());
    CHECK(fn4::request("ds41_e2", 1, &why));
    int sw[fn4::kSwitches];
    (void)fn4::apply_pending(4, sw);
    fn4::requests(n1.data(), v1.data());
    CHECK_EQ(n1[(size_t)fn4::kDs41E2], n0[(size_t)fn4::kDs41E2] + 1);
    CHECK_EQ(v1[(size_t)fn4::kDs41E2], 1);
    /* The leader reports the follower's values (GET's "peer"), only the follower's own switches. */
    std::vector<int32_t> fol(fn4::kSwitches, 0);
    fol[(size_t)fn4::kDs41E2] = 1;
    fol[(size_t)fn4::kDraftOcc] = 0;
    fn4::note_peer(fol.data(), fn4::kSwitches);
    const std::string js = fn4::state_json();
    const size_t at = js.find("\"peer\": {\"role\": \"follower\", \"switches\": {");
    CHECK(at != std::string::npos);
    const std::string peer = at == std::string::npos ? "" : js.substr(at, js.find('}', at) - at);
    CHECK(peer.find("\"ds41_e2\": 1") != std::string::npos);
    CHECK(peer.find("draft_occ") == std::string::npos);
    fn4::set_peer_check(nullptr);
    /* One machine again (no peer): the plugin switch is this node's verdict alone. */
    CHECK(!fn4::check("ds41_e2", 1, &why));
}

/* tp8 loop P25s: the SP route's in-process switches (ds41_sphc, ds41_sphc_attn, ds41_sphc_seg) are deepseek41's -- plugin
 * switches that act on a follower's passes (the follower takes them from the leader's record), appended after
 * draft_short_blk (DECODE's D23, which took 82 first); with no plugin here serving them a request is refused and nothing is queued, and GET reports -1 (none
 * serves them); the follower's record of them is taken (marked) and named as skipped, never a failure. */
TEST(the_sp_route_switches_are_deepseek41s_and_a_followers) {
    CHECK_EQ(fn4::switch_index("ds41_sphc"), (int)fn4::kDs41Sphc);
    CHECK_EQ(fn4::switch_index("ds41_sphc_attn"), (int)fn4::kDs41SphcAttn);
    CHECK_EQ(fn4::switch_index("ds41_sphc_seg"), (int)fn4::kDs41SphcSeg);
    CHECK_EQ((int)fn4::kDs41Sphc, (int)fn4::kDs41AttnDec + 1);   /* after DECODE's D23 draft_short_blk (82), D24 ds41_prefetch (83) and D26 ds41_attn_dec (84) since the 19.11. merges */
    CHECK_EQ((int)fn4::kPpBgRoom, (int)fn4::kDs41SphcSeg + 1);   /* tp8 loop P23s after them: the leader's own */
    CHECK_EQ(fn4::switch_index("pp_bg_room"), (int)fn4::kPpBgRoom);
    CHECK_EQ(fn4::switch_index("pp_bg_join_ms"), (int)fn4::kPpBgJoinMs);
    CHECK(!fn4::follower_switch(fn4::kPpBgRoom));
    CHECK(!fn4::follower_switch(fn4::kPpBgJoinMs));
    {
        std::string e;
        CHECK(!fn4::check("pp_bg_room", 100, &e));      /* 0 or 256 .. 65536 */
        CHECK(fn4::check("pp_bg_room", 2600, &e));
        CHECK(!fn4::check("pp_bg_join_ms", 60001, &e));
    }
    for (const int i : { (int)fn4::kDs41Sphc, (int)fn4::kDs41SphcAttn, (int)fn4::kDs41SphcSeg }) {
        CHECK(fn4::follower_switch(i));
        CHECK_EQ(fn4::value(i), -1);
    }
    std::vector<uint32_t> n0(fn4::kSwitches), n1(fn4::kSwitches);
    std::vector<int32_t> v0(fn4::kSwitches), v1(fn4::kSwitches);
    fn4::requests(n0.data(), v0.data());
    std::string why;
    CHECK(!fn4::request("ds41_sphc", 2048, &why));
    CHECK(!fn4::request("ds41_sphc_seg", 0, &why));
    fn4::requests(n1.data(), v1.data());
    CHECK_EQ(n1[(size_t)fn4::kDs41Sphc], n0[(size_t)fn4::kDs41Sphc]);
    CHECK_EQ(n1[(size_t)fn4::kDs41SphcSeg], n0[(size_t)fn4::kDs41SphcSeg]);
    CHECK(fn4::state_json().find("\"ds41_sphc\": -1, \"ds41_sphc_attn\": -1, \"ds41_sphc_seg\": -1") != std::string::npos);
    /* the follower's side: the leader's record asks ds41_sphc_seg 0 -- taken (marked), skipped here (no plugin) */
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    count[(size_t)fn4::kDs41SphcSeg] = 1;
    std::vector<std::string> skipped;
    (void)fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped);
    CHECK_EQ(taken[(size_t)fn4::kDs41SphcSeg], 1u);
    bool named = false;
    for (const std::string& s : skipped) named = named || s.rfind("ds41_sphc_seg", 0) == 0;
    CHECK(named);
}

/* tp8 loop P30t: ds41_twin is switch 90, right after pp_bg_join_ms, deepseek41's (a plugin switch the follower takes with the
 * leader's, not the scheduler's); with no plugin serving it a request is refused at the step release and nothing is queued,
 * and GET reports -1; the follower's record of it is taken and named as skipped. */
TEST(ds41_twin_is_a_follower_plugin_switch_after_pp_bg_join_ms) {
    const int i = fn4::switch_index("ds41_twin");
    REQUIRE(i == (int)fn4::kDs41Twin);
    CHECK_EQ(i, (int)fn4::kPpBgJoinMs + 1);
    CHECK_EQ(i, 90);
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    CHECK(std::string(fn4::switch_name(i)) == "ds41_twin");
    CHECK_EQ(fn4::value(i), -1);
    std::vector<uint32_t> n0(fn4::kSwitches), n1(fn4::kSwitches);
    std::vector<int32_t> v0(fn4::kSwitches), v1(fn4::kSwitches);
    fn4::requests(n0.data(), v0.data());
    std::string why;
    CHECK(!fn4::request("ds41_twin", 0, &why));
    fn4::requests(n1.data(), v1.data());
    CHECK_EQ(n1[(size_t)i], n0[(size_t)i]);
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    count[(size_t)i] = 1;
    value[(size_t)i] = 1;
    std::vector<std::string> skipped;
    (void)fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped);
    CHECK_EQ(taken[(size_t)i], 1u);
    bool named = false;
    for (const std::string& s : skipped) named = named || s.rfind("ds41_twin", 0) == 0;
    CHECK(named);
}

/* tp8 loop P47: ds41_csplit is switch 103, right after aql_hdp, deepseek41's (the follower takes it with the leader's: the
 * candidate layer it changes is the follower's); with no plugin serving it the follower's record of it is taken and named as
 * skipped. */
TEST(ds41_csplit_is_a_follower_plugin_switch_after_aql_hdp) {
    const int i = fn4::switch_index("ds41_csplit");
    REQUIRE(i == (int)fn4::kDs41Csplit);
    CHECK_EQ(i, (int)fn4::kAqlHdp + 1);
    CHECK(fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    count[(size_t)i] = 1;
    value[(size_t)i] = 0;
    std::vector<std::string> skipped;
    (void)fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped);
    CHECK_EQ(taken[(size_t)i], 1u);
    bool named = false;
    for (const std::string& s : skipped) named = named || s.rfind("ds41_csplit", 0) == 0;
    CHECK(named);
}

/* tp8 loop D56: sched_lane_room is switch 104, after ds41_csplit, the leader's scheduler's alone: a follower takes the leader's
 * record of it (the count) without asking for it, and nothing is skipped. */
TEST(sched_lane_room_is_the_leaders_alone_after_ds41_csplit) {
    const int i = fn4::switch_index("sched_lane_room");
    REQUIRE(i == (int)fn4::kSchedLaneRoom);
    CHECK_EQ(i, (int)fn4::kDs41Csplit + 1);
    CHECK_EQ(i + 2, (int)fn4::kSwitches);   /* tp8 loop P46: pp_send_ash 105 after it */
    CHECK(!fn4::follower_switch(i));
    CHECK(!fn4::sched_switch(i));
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    count[(size_t)i] = 1;
    value[(size_t)i] = 1;
    std::vector<std::string> skipped;
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 0);
    CHECK_EQ(taken[(size_t)i], 1u);
    CHECK(skipped.empty());
    CHECK_EQ(fn4::sched_lane_room(), 0);
}

/* tp8 loop P46: pp_send_ash is switch 105, after sched_lane_room, BOTH nodes' (each its own send rings): a follower asks for the
 * leader's request; a follower started without RADIANCE_PP_SEND_ASH refuses 1 by name (skipped, never a failure) and keeps 0. */
TEST(pp_send_ash_follows_the_leaders_request_after_sched_lane_room) {
    const int i = fn4::switch_index("pp_send_ash");
    REQUIRE(i == (int)fn4::kPpSendAsh);
    CHECK_EQ(i, (int)fn4::kSchedLaneRoom + 1);
    CHECK_EQ(i + 1, (int)fn4::kSwitches);
    CHECK(fn4::follower_switch(i));
    std::vector<uint32_t> count(fn4::kSwitches, 0), taken(fn4::kSwitches, 0);
    std::vector<int32_t> value(fn4::kSwitches, 0);
    count[(size_t)i] = 1;
    value[(size_t)i] = 1;
    std::vector<std::string> skipped;
    CHECK_EQ(fn4::follow(count.data(), value.data(), fn4::kSwitches, taken.data(), &skipped), 0);
    CHECK_EQ(taken[(size_t)i], 1u);
    REQUIRE_EQ(skipped.size(), (size_t)1);
    CHECK(skipped[0].find("pp_send_ash=1") != std::string::npos);
    CHECK(skipped[0].find("RADIANCE_PP_SEND_ASH=N") != std::string::npos);
    CHECK_EQ(fn4::pp_send_ash(), 0);
}

}  /* namespace */

RAD_TEST_MAIN()
