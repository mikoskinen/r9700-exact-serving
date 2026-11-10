/* draft_conf_test.cpp -- the confidence gate's rule and the tracker (core/sched/draft_conf.h): the
 * dspark_conf values, the longest confident prefix, the calibration counts, and the shadow tree held
 * to the tokens the target emits across verifies. */
#include "rad_test.h"

#include "runtime/fn4_switch.h"
#include "sched/draft_conf.h"

#include <cmath>
#include <limits>
#include <string>

using namespace rad;

TEST(policy_values) {
    dconf::Policy p = dconf::policy_of(0);
    CHECK_EQ(p.mode, 0);
    p = dconf::policy_of(1);
    CHECK_EQ(p.mode, 1);
    CHECK(p.replace);
    CHECK_NEAR(p.thr, 0.5, 1e-12);
    CHECK_NEAR(p.logit, 0.0f, 1e-6f);
    p = dconf::policy_of(30);
    CHECK_EQ(p.mode, 1);
    CHECK(p.replace);
    CHECK_NEAR(p.logit, std::log(0.3 / 0.7), 1e-6);
    p = dconf::policy_of(101);
    CHECK_EQ(p.mode, 1);
    CHECK(!p.replace);
    CHECK_NEAR(p.thr, 0.5, 1e-12);
    p = dconf::policy_of(170);
    CHECK(!p.replace);
    CHECK_NEAR(p.thr, 0.7, 1e-12);
    CHECK_EQ(dconf::policy_of(2).mode, 2);
    /* values outside the switch's set read as off */
    CHECK_EQ(dconf::policy_of(3).mode, 0);
    CHECK_EQ(dconf::policy_of(100).mode, 0);
    CHECK_EQ(dconf::policy_of(196).mode, 0);
    for (int v : { 0, 1, 2, 5, 50, 95, 101, 105, 150, 195 }) CHECK(fn4::dspark_conf_value_ok(v));
    for (int v : { -1, 3, 4, 96, 100, 102, 104, 196, 200 }) CHECK(!fn4::dspark_conf_value_ok(v));
}

TEST(prefix_is_the_longest_confident_run) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float a[5] = { 1.0f, 0.5f, -0.1f, 2.0f, 3.0f };
    CHECK_EQ(dconf::prefix_len(a, 5, dconf::policy_of(1)), 2);    /* sigmoid >= 0.5: logit >= 0 */
    CHECK_EQ(dconf::prefix_len(a, 5, dconf::policy_of(40)), 5);   /* logit >= -0.405 */
    CHECK_EQ(dconf::prefix_len(a, 5, dconf::policy_of(80)), 0);   /* logit >= 1.386 */
    CHECK_EQ(dconf::prefix_len(a, 1, dconf::policy_of(1)), 1);
    CHECK_EQ(dconf::prefix_len(a, 0, dconf::policy_of(1)), 0);
    const float b[3] = { 4.0f, nan, 4.0f };
    CHECK_EQ(dconf::prefix_len(b, 3, dconf::policy_of(1)), 1);    /* a NaN stops the prefix */
}

/* The JSON's numbers in order after a key, for the checks below. */
static std::string after(const std::string& j, const std::string& key) {
    const size_t at = j.find("\"" + key + "\": ");
    if (at == std::string::npos) return "";
    const size_t s = at + key.size() + 4;
    size_t e = s;
    int depth = 0;
    for (; e < j.size(); ++e) {
        if (j[e] == '[') ++depth;
        if (j[e] == ']') { if (--depth == 0) { ++e; break; } }
        if (depth == 0 && (j[e] == ',' || j[e] == '}')) break;
    }
    return j.substr(s, e - s);
}

TEST(tracker_shadow_follows_the_emitted_tokens) {
    dconf::Tracker t;
    const int32_t alt1[10] = { 10, 11, 12, 13, 14,   20, 21, 22, 23, 24 };
    /* Verify 1: the chain's first draft rejected, the target emitted 20 = branch 1's first token. */
    t.proposed(7, 1, nullptr, 0, alt1, 2, 5, -1);
    const int32_t e1[1] = { 20 };
    t.verified(7, 2, 3, 0, e1, 1);
    /* Verify 2 (a new proposal, its chain took two drafts): 21, 22 continue branch 1, 99 ends it. */
    const int32_t alt2[10] = { 50, 51, 52, 53, 54,   60, 61, 62, 63, 64 };
    t.proposed(7, 2, nullptr, 0, alt2, 2, 5, -1);
    const int32_t e2[3] = { 21, 22, 99 };
    t.verified(7, 3, 3, 2, e2, 3);
    /* Another sequence: the chain took nothing and no branch starts with the target's token. */
    t.proposed(8, 2, nullptr, 0, alt2, 2, 5, -1);
    const int32_t e3[1] = { 77 };
    t.verified(8, 3, 1, 0, e3, 1);
    /* A third: branch 0 matches all five, across two later verifies. */
    t.proposed(9, 3, nullptr, 0, alt1, 2, 5, -1);
    const int32_t e4[1] = { 10 };
    t.verified(9, 4, 2, 0, e4, 1);
    const int32_t e5[2] = { 11, 12 };
    t.verified(9, 5, 0, 0, e5, 2);
    const int32_t e6[3] = { 13, 14, 15 };
    t.verified(9, 6, 0, 0, e6, 3);
    /* A verify that carried no draft: counted apart, no shadow. */
    t.proposed(10, 6, nullptr, 0, alt1, 2, 5, 0);
    const int32_t e7[1] = { 10 };
    t.verified(10, 7, 0, 0, e7, 1);
    const std::string j = t.json();
    CHECK_EQ(after(j, "verifies"), std::string("4"));
    CHECK_EQ(after(j, "chain_accepted"), std::string("2"));
    CHECK_EQ(after(j, "p1_miss"), std::string("3"));
    CHECK_EQ(after(j, "zero_draft"), std::string("1"));
    CHECK_EQ(after(j, "alt_hit"), std::string("[1, 1]"));
    /* branch 1 matched 3 (counts with 2 branches only), branch 0 matched 5 (with 1 or 2) */
    CHECK_EQ(after(j, "gain"), std::string("[[1, 2, 3, 4, 5], [2, 4, 6, 7, 8]]"));
    if (after(j, "gain") != "[[1, 2, 3, 4, 5], [2, 4, 6, 7, 8]]") fprintf(stderr, "    %s\n", j.c_str());
}

TEST(tracker_closes_an_open_branch_when_its_sequence_goes_quiet) {
    dconf::Tracker t;
    const int32_t alt[5] = { 5, 6, 7, 8, 9 };
    t.proposed(3, 10, nullptr, 0, alt, 1, 5, -1);
    const int32_t e[1] = { 5 };
    t.verified(3, 11, 1, 0, e, 1);
    t.purge(11 + 10, 64);
    CHECK_EQ(after(t.json(), "gain"), std::string("[[0, 0, 0, 0, 0]]"));
    t.purge(11 + 65, 64);   /* the request ended: the branch closes where it stood (1 token) */
    CHECK_EQ(after(t.json(), "gain"), std::string("[[1, 1, 1, 1, 1]]"));
}

TEST(tracker_calibration_and_cuts) {
    dconf::Tracker t;
    const float lg[5] = { 2.0f, -1.0f, 0.0f, 3.0f, 3.0f };   /* sigmoid deciles 8, 2, 5, 9, 9 */
    t.proposed(1, 1, lg, 5, nullptr, 0, 0, 3);
    const int32_t e[2] = { 40, 41 };
    t.verified(1, 2, 3, 1, e, 2);   /* position 0 taken, 1 rejected, 2 never reached */
    const std::string j = t.json();
    CHECK_EQ(after(j, "cuts"), std::string("1"));
    CHECK_EQ(after(j, "cut_drafts"), std::string("3"));
    CHECK_EQ(after(j, "cut_hist"), std::string("[0, 0, 0, 1]"));
    CHECK_EQ(after(j, "calib"),
             std::string("[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0]], "
                         "[[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]"));
    /* a verify with no proposal pending changes nothing */
    t.verified(1, 3, 2, 2, e, 2);
    CHECK_EQ(after(t.json(), "calib"), after(j, "calib"));
}

RAD_TEST_MAIN()
