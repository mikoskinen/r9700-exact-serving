/* draft_conf.h -- a chain block drafter's confidence plane and alternative branches, on the host
 * (ds41-acc: DeepSeek-V4.1's DSpark, levers C1 and T1; the engine's `dspark_conf` / `dspark_alt` switches,
 * core/runtime/fn4_switch.h).
 *
 * The confidence plane (rad_drafter_confidence) gives each proposal j of a drafting sequence a logit.
 * Policy: the sequence verifies the longest prefix whose sigmoid clears a threshold (exllamav3 1.6.5's
 * rule for DSpark; the release's inference/model.py computes the logit and applies nothing). That is a
 * row count, so the engine needs it before it builds the verify; it changes no proposal and no emitted
 * token, which follow core/sample/accept.h as ever.
 *
 * Tracker: what the plane and the branches would have bought, counted against what the target emitted.
 *   calibration  per draft position, the verified positions by the sigmoid's decile: reached, accepted.
 *   shadow       the alternative branches (rad_drafter_alternatives) against a verify whose chain took no
 *                draft: when one branch's first token is the token the target emitted there, the branch is
 *                held to the next tokens the target emits until it disagrees or ends. A tree that verified
 *                the first `n` branches to depth L beside the chain would have emitted min(match, L) more
 *                tokens on that verify: greedy acceptance follows the path the target's own tokens take, so
 *                the count is exact for that verify (the steps after it would be cut differently).
 * The step thread feeds it; /stats reads it under the lock. */
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace dconf {

/* A dspark_conf value read: 0 off, 1 gate, 2 observe. `replace`: the gate's prefix replaces the
 * adaptive window's stop; else it is capped by it. `logit`: the threshold as a logit. */
struct Policy {
    int    mode = 0;
    bool   replace = true;
    double thr = 0.5;
    float  logit = 0.0f;
};
Policy policy_of(int v);

/* The longest prefix of `n` logits whose sigmoid is at least the policy's threshold. */
int prefix_len(const float* logits, int n, const Policy& p);

class Tracker {
public:
    enum { kMaxLen = 16, kMaxAlt = 8, kDeciles = 10 };

    /* A proposal published for `seq`'s next verify (step `step`): its logits (null: none) and branches
     * (null: none), `n_alt` of `len` tokens, -1 where there is none. `cut` is the verify count the gate
     * set, -1 when it set none. */
    void proposed(uint64_t seq, int step, const float* logits, int n_logits, const int32_t* alt, int n_alt,
                  int len, int cut);
    /* `seq`'s verify: `drafted` drafts carried, `accepted` taken, `emitted` the tokens it emitted in
     * order (the accepted drafts, then the target's own). */
    void verified(uint64_t seq, int step, int drafted, int accepted, const int32_t* emitted, int n_emitted);
    /* Records idle for `age` steps are closed (a finished request): its open branches end where they are. */
    void purge(int step, int age = 64);
    /* dspark_conf_rows: a cycle under a load bound (`bounded`), and whether its load was above it (`gated`). */
    void load_gate(bool bounded, bool gated);
    std::string json() const;

private:
    struct Open { int a = 0, matched = 0, len = 0; int32_t want[kMaxLen] = {}; };
    struct Rec {
        int step = 0;
        bool pending = false;
        int n_logits = 0, n_alt = 0, len = 0, cut = -1;
        float logits[kMaxLen] = {};
        bool has_logits = false;
        int32_t alt[kMaxAlt * kMaxLen] = {};
        std::vector<Open> open;
    };
    void close(const Open& o);

    mutable std::mutex mu_;
    std::map<uint64_t, Rec> recs_;
    /* calibration: [position][decile] */
    int64_t reached_[kMaxLen][kDeciles] = {}, hit_[kMaxLen][kDeciles] = {};
    int64_t cuts_ = 0, cut_hist_[kMaxLen + 1] = {}, cut_drafts_ = 0, cut_full_ = 0;
    int64_t lg_bounded_ = 0, lg_gated_ = 0;     /* dspark_conf_rows: cycles under a bound, cycles above it */
    /* shadow */
    int64_t sh_verifies_ = 0, sh_chain_acc_ = 0, sh_p1_miss_ = 0, sh_zero_ = 0;
    int64_t sh_hit_[kMaxAlt] = {};
    int64_t sh_gain_[kMaxAlt][kMaxLen] = {};   /* [branches used - 1][depth - 1] */
    int     sh_n_alt_ = 0, sh_len_ = 0;
};

Tracker& tracker();

}  // namespace dconf
}  // namespace rad
