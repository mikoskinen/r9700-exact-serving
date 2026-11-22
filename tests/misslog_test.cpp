/* misslog_test.cpp -- RADIANCE_DEBUG_MISS_LOG (place/misslog.h): a pass's tier snapshot, the drain's
 * classing of each histogram by it, the lines and the file. Pure host code: the engine's drain
 * hands MissRank the same rows these cases hand it (runtime_test drives it from the route ring). */
#include "rad_test.h"

#include "place/misslog.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace rad;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> lines_of(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string l; std::getline(ss, l);) v.push_back(l);
    return v;
}

/* A rank with `n_trunk` trunk layers and one head layer of `E` experts; unit = layer * E + e, except
 * the holes named. Unit u is (u + 1) KiB. */
struct Toy {
    int n_layer, n_trunk;
    int64_t E;
    std::vector<Tier> tier;
    MissRank mr;

    Toy(int trunk, int64_t e, std::vector<int> holes = {}) : n_layer(trunk + 1), n_trunk(trunk), E(e) {
        const size_t n = (size_t)n_layer * (size_t)E;
        tier.assign(n, Tier::VRAM);
        std::vector<int32_t> unit_of(n);
        std::vector<int64_t> bytes(n);
        for (size_t u = 0; u < n; ++u) {
            unit_of[u] = (int32_t)u;
            bytes[u] = 1024 * (int64_t)(u + 1);
        }
        for (int h : holes) unit_of[(size_t)h] = -1;
        std::vector<uint8_t> cls(n, (uint8_t)kMlVram);
        mr.init(0, n_layer, n_trunk, E, unit_of, bytes, cls);
    }
    int32_t unit(int layer, int e) const { return (int32_t)(layer * E + e); }
    void issue(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode, int64_t n_tok) {
        mr.issued(step, draft_pass, n_seq, n_seq_decode, n_tok,
                  [this](int32_t u) { return tier[(size_t)u]; });
    }
    void credit(int layer, int step, std::vector<int32_t> counts, int reports, int now_step) {
        mr.credit(layer, step, counts.data(), (int64_t)counts.size(), reports, now_step,
                  [this](int32_t u) { return tier[(size_t)u]; });
    }
    std::vector<std::string> drain() {
        mr.end_drain();
        std::string s;
        mr.take(&s);
        return lines_of(s);
    }
};

}  /* namespace */

TEST(the_variable_unset_or_empty_names_no_file) {
    unsetenv(kMissLogEnv);
    CHECK(miss_log_path() == nullptr);
    setenv(kMissLogEnv, "", 1);
    CHECK(miss_log_path() == nullptr);
    setenv(kMissLogEnv, "/x/miss.log", 1);
    REQUIRE(miss_log_path() != nullptr);
    CHECK_EQ(std::string(miss_log_path()), std::string("/x/miss.log"));
    unsetenv(kMissLogEnv);
}

TEST(trunk_kinds_follow_the_batch_shape) {
    CHECK_EQ(ml_trunk_kind(4, 0, 4096), (int)kMlPrefill);
    CHECK_EQ(ml_trunk_kind(4, 4, 4), (int)kMlDecode);
    CHECK_EQ(ml_trunk_kind(4, 4, 8), (int)kMlVerify);
    CHECK_EQ(ml_trunk_kind(4, 3, 900), (int)kMlMixed);
    CHECK_EQ(std::string(ml_kind_name(kMlVerify)), std::string("verify"));
    CHECK_EQ(std::string(ml_kind_name(kMlDraft)), std::string("draft"));
}

/* A decode step and its draft passes: each layer classed by the tiers its own pass read, a move
 * published after the issue counted in `mv` and not in the classes, a hole counted as other, and
 * the head's layer on a line of its own carrying the last draft pass's shape. */
TEST(a_histogram_is_classed_by_its_own_pass_snapshot) {
    Toy t(2, 4, { 1 * 4 + 3 });                     /* (layer 1, expert 3) has no unit */
    t.tier[(size_t)t.unit(0, 1)] = Tier::HostPinned;
    t.tier[(size_t)t.unit(0, 3)] = Tier::SSD;
    t.tier[(size_t)t.unit(1, 2)] = Tier::HostPinned;
    for (int e = 0; e < 4; ++e) t.tier[(size_t)t.unit(2, e)] = Tier::HostPinned;

    t.issue(5, 0, 2, 2, 2);
    t.mr.ran(5, 0, false);
    t.issue(5, -1, 2, 2, 4);                        /* the history pass */
    t.issue(5, 2, 2, 2, 2);                         /* draft round 2: the bank holds this one */
    /* The dispatch behind step 5 promotes (0, 1) and demotes (0, 0): the drain's tiers. */
    t.tier[(size_t)t.unit(0, 1)] = Tier::VRAM;
    t.tier[(size_t)t.unit(0, 0)] = Tier::HostPinned;

    t.credit(0, 5, { 1, 2, 0, 1 }, 1, 6);
    t.credit(1, 5, { 0, 1, 1, 4 }, 1, 6);
    t.credit(2, 5, { 0, 3, 0, 0 }, 2, 6);
    const std::vector<std::string> v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)2);
    /* live 6: vram (0,0) (1,1); host (0,1) (1,2); other (0,3) and the hole. hkib (2 + 7) KiB. */
    CHECK_EQ(v[0], std::string("5 decode 0 2 2 1 0 0 6 2 2 2 2 9 2 0:1.1.1,1.1.1"));
    /* (2, 1) is unit 9: 10 KiB. */
    CHECK_EQ(v[1], std::string("5 draft 0 2 2 2 0 0 1 0 1 0 1 10 0 2:0.1"));
    CHECK_EQ(t.mr.last_step(), 5);
}

/* An armed prefill stager copies the host-pool experts to VRAM: no zero-copy count. Verify, prefill
 * and mixed passes are named from the batch. */
TEST(staged_passes_have_no_zero_copy_count_and_kinds_are_named) {
    Toy t(2, 4);
    t.tier[(size_t)t.unit(0, 2)] = Tier::HostPinned;
    t.issue(10, 0, 2, 2, 4);                        /* verify */
    t.mr.ran(10, 0, false);
    t.issue(11, 0, 1, 0, 4096);                     /* prefill */
    t.mr.ran(11, 0, true);
    t.issue(12, 0, 3, 2, 1000);                     /* mixed */
    t.mr.ran(12, 0, false);
    for (int s = 10; s <= 12; ++s) t.credit(0, s, { 1, 0, 1, 0 }, 1, 13);
    const std::vector<std::string> v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)3);
    /* (0, 2) is unit 2: 3 KiB. */
    CHECK_EQ(v[0], std::string("10 verify 0 4 2 1 0 0 2 1 1 0 1 3 0 0:1.1"));
    CHECK_EQ(v[1], std::string("11 prefill 0 4096 1 1 1 0 2 1 1 0 -1 3 0 0:1.1"));
    CHECK_EQ(v[2], std::string("12 mixed 0 1000 3 1 0 0 2 1 1 0 1 3 0 0:1.1"));
}

/* A histogram whose pass left no snapshot is classed by the drain's tiers and tagged with the lag;
 * at shutdown the lag is -1. A layer between two drained ones is "-". */
TEST(a_histogram_without_a_snapshot_is_tagged_with_its_lag) {
    Toy t(3, 2);
    t.tier[(size_t)t.unit(2, 1)] = Tier::HostPinned;
    t.credit(0, 40, { 1, 1 }, 1, 42);
    t.credit(2, 40, { 0, 3 }, 1, 42);
    std::vector<std::string> v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)1);
    CHECK_EQ(v[0], std::string("40 unknown 0 0 0 1 0 2 3 2 1 0 1 6 -1 0:2,-,0.1"));

    t.credit(1, 41, { 2, 0 }, 1, -1);
    v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)1);
    CHECK_EQ(v[0], std::string("41 unknown 0 0 0 1 0 -1 1 1 0 0 0 0 -1 1:1"));
}

/* A layer drained after its step's line was written makes another line with only that layer, still
 * classed by the step's snapshot. */
TEST(a_late_layer_makes_a_line_of_its_own) {
    Toy t(2, 2);
    t.issue(3, 0, 1, 1, 1);
    t.credit(0, 3, { 1, 0 }, 1, 4);
    std::vector<std::string> v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)1);
    CHECK_EQ(v[0], std::string("3 decode 0 1 1 1 0 0 1 1 0 0 0 0 0 0:1"));
    t.tier[(size_t)t.unit(1, 1)] = Tier::HostPinned;   /* moved after the pass */
    t.credit(1, 3, { 0, 1 }, 1, 5);
    v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)1);
    CHECK_EQ(v[0], std::string("3 decode 0 1 1 1 0 0 1 1 0 0 0 0 1 1:1"));
}

/* A histogram older than the step now in its ring slot does not evict that step's record. */
TEST(an_old_histogram_does_not_evict_a_newer_pass) {
    Toy t(1, 2);
    t.tier[(size_t)t.unit(0, 1)] = Tier::HostPinned;
    t.issue(71, 0, 1, 1, 1);                          /* slot 71 % 64 = 7 */
    t.tier[(size_t)t.unit(0, 1)] = Tier::VRAM;
    t.credit(0, 7, { 0, 1 }, 1, 72);                  /* step 7: no record, drain's tiers */
    t.credit(0, 71, { 0, 1 }, 1, 72);                 /* step 71: its own snapshot */
    const std::vector<std::string> v = t.drain();
    REQUIRE_EQ(v.size(), (size_t)2);
    CHECK_EQ(v[0], std::string("7 unknown 0 0 0 1 0 65 1 1 0 0 0 0 -1 0:1"));
    CHECK_EQ(v[1], std::string("71 decode 0 1 1 1 0 0 1 0 1 0 1 2 1 0:0.1"));
}

/* The file: a header at start, lines the writer thread flushes while the engine runs, profile
 * markers, and the summary at close. A path that cannot be written fails start(). */
TEST(the_file_has_a_header_flushed_lines_markers_and_a_summary) {
    const std::string path = radtest::tmp_path("miss.log");
    MlInfo info;
    info.model = "toy";
    info.arch = "deepseek41";
    info.tp = 2;
    info.n_layer = 2;
    info.n_trunk = 1;
    info.n_expert = 2;
    MissLog ml;
    ml.init(path, info);
    REQUIRE_EQ(ml.n_rank(), 2);
    std::vector<Tier> tier = { Tier::VRAM, Tier::HostPinned, Tier::HostPinned, Tier::HostPinned };
    for (int r = 0; r < 2; ++r)
        ml.rank(r)->init(r, 2, 1, 2, { 0, 1, 2, 3 }, { 1024, 2048, 4096, 4096 },
                         { kMlVram, kMlHost, kMlHost, kMlHost });
    CHECK_OK(ml.start(20));
    std::string s = slurp(path);
    CHECK(s.rfind("# radiance-miss-log/1\n", 0) == 0);
    CHECK(s.find("# model toy arch deepseek41 tp 2 routed_layers 2 trunk_layers 1 experts 2") !=
          std::string::npos);
    CHECK(s.find("# rank 1 expert_units 4 at_start vram 1 host 3 other 0 unit_bytes_mean 2816") !=
          std::string::npos);

    const auto tier_of = [&](int32_t u) { return tier[(size_t)u]; };
    for (int step = 1; step <= 3; ++step)
        for (int r = 0; r < 2; ++r) {
            MissRank* m = ml.rank(r);
            m->issued(step, 0, 1, 1, 1, tier_of);
            m->ran(step, 0, false);
            m->issued(step, -1, 1, 1, 2, tier_of);
            const int32_t c0[2] = { 1, 1 }, c1[2] = { 0, 2 };
            m->credit(0, step, c0, 2, 1, step + 1, tier_of);
            m->credit(1, step, c1, 2, 1, step + 1, tier_of);
            m->end_drain();
        }
    ml.profile_mark(2, false);
    /* The writer flushes on its own. */
    bool seen = false;
    for (int i = 0; i < 200 && !seen; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        seen = slurp(path).find("3 draft 1 2 1 1 0 0 1 0 1 0 1 4 0 1:0.1") != std::string::npos;
    }
    CHECK(seen);
    s = slurp(path);
    CHECK(s.find("2 decode 0 1 1 1 0 0 2 1 1 0 1 2 0 0:1.1\n") != std::string::npos);
    CHECK(s.find("# profile 1 step 2\n") != std::string::npos);

    ml.profile_mark(ml.last_step(), true);
    ml.close("# heat rank 0 dispatches 3\n");
    s = slurp(path);
    CHECK(s.find("# profile 2 step 3 final\n") != std::string::npos);
    CHECK(s.find("# heat rank 0 dispatches 3\n") != std::string::npos);
    CHECK(s.find("# summary rank 0 kind decode lines 3 live 6 vram 3 host 3 other 0 zc_lines 3 "
                 "zc 3 hkib 6 mv 0 lag_lines 0\n") != std::string::npos);
    CHECK(s.find("# summary rank 1 kind draft lines 3 live 3 vram 0 host 3 other 0 zc_lines 3 "
                 "zc 3 hkib 12 mv 0 lag_lines 0\n") != std::string::npos);
    CHECK(s.find("# filed rank 1 trunk_passes 3 draft_passes 3 histograms_without_trunk_pass 0\n") !=
          std::string::npos);
    /* Every data line has the 16 fields the header names. */
    int data = 0;
    for (const std::string& l : lines_of(s)) {
        if (l.empty() || l[0] == '#') continue;
        ++data;
        std::stringstream ls(l);
        int fields = 0;
        for (std::string f; ls >> f;) ++fields;
        CHECK_EQ(fields, 16);
    }
    CHECK_EQ(data, 12);

    MissLog bad;
    bad.init(radtest::tmp_path("no/such/dir/miss.log"), info);
    for (int r = 0; r < 2; ++r) bad.rank(r)->init(r, 2, 1, 2, { 0, 1, 2, 3 }, { 1, 1, 1, 1 }, {});
    CHECK(bad.start() < 0);
}

/* The size the caller asked to keep reasonable: 1000 steps x 4 ranks x 43 trunk layers and a head,
 * 96 experts a rank, ~20 live a layer (8 users verifying one draft each at top-6 over 4 ranks),
 * one in twelve of them in the host pool. */
TEST(a_thousand_decode_steps_at_tp4_stay_near_a_megabyte) {
    const int L = 44, T = 43;
    const int64_t E = 96;
    const size_t n = (size_t)L * (size_t)E;
    std::vector<int32_t> unit_of(n);
    std::vector<int64_t> bytes(n, 9 << 20);
    for (size_t u = 0; u < n; ++u) unit_of[u] = (int32_t)u;
    std::vector<Tier> tier(n, Tier::VRAM);
    std::mt19937 rng(7);
    for (size_t u = 0; u < n; ++u)
        if (rng() % 12 == 0) tier[u] = Tier::HostPinned;
    const auto tier_of = [&](int32_t u) { return tier[(size_t)u]; };

    size_t total = 0;
    std::vector<int32_t> counts((size_t)E);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 4; ++r) {
        MissRank m;
        m.init(r, L, T, E, unit_of, bytes, std::vector<uint8_t>(n, (uint8_t)kMlVram));
        for (int step = 0; step < 1000; ++step) {
            m.issued(step, 0, 8, 8, 16, tier_of);
            m.ran(step, 0, false);
            m.issued(step, -1, 8, 8, 16, tier_of);
            for (int l = 0; l < L; ++l) {
                std::fill(counts.begin(), counts.end(), 0);
                for (int k = 0; k < 24; ++k) ++counts[(size_t)(rng() % (uint32_t)E)];
                m.credit(l, step, counts.data(), E, 1, step + 1, tier_of);
            }
            m.end_drain();
        }
        std::string s;
        m.take(&s);
        total += s.size();
        CHECK_EQ(lines_of(s).size(), (size_t)2000);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  1000 steps x 4 ranks: %.2f MB of lines, %.1f us a (step, rank) on this host\n",
                (double)total / 1e6, us / 4000.0);
    CHECK(total < (size_t)2'000'000);
}

RAD_TEST_MAIN()
