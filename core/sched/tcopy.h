/* tcopy.h -- tp8 loop D46: which sequences' TAIL BAND state the leader holds, so a decode lane's step may run as a tail band
 * pass (fn4 pp_tband 3). The plugin declares its copy's form (abi/rad_pp.h rad_pp_tail_copy):
 *   form 1      every pass that is no tail band pass carries the band layers' KV rows in X2 and the leader stores them: the
 *               leader holds the band layers' WHOLE state of a sequence whose every row was written so (llama_fp8's stand-in);
 *   form R >= 2 only a pass whose entries are ALL decode entries carries them (a decode pass is never row-cut, so its rows are
 *               the rows its band layers store); the band layers keep a WINDOW of the last R positions (deepseek41: a layer
 *               past the last kv source keeps only its window keys, window + max_spec slots), so a sequence is held again once
 *               the first row of a step lies R positions past its last row that was not written on the leader.
 * A tail band pass writes the band layers on the leader itself (its tail runs them), so it counts as written. A snapshot
 * restore or a cone replay (positions the follower's state came from elsewhere) is a gap at its context. Gap[seq]: -1 = every
 * row so far written on the leader; p >= 0 = the position after the last row that was not. Pure functions of the step's
 * entries, so sched_test holds them; the engine keeps the map (engine_lanes.cpp pp_tband_note / pp_tband_step). */
#pragma once
#include "batch.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rad::tcopy {

using Gap = std::unordered_map<uint64_t, int64_t>;

/* whether this step's rows are written on the leader: a tail band pass, or a pass the plugin's copy rides */
inline bool written(int form, bool band, const std::vector<StepEntry>& es) {
    if (band || form == 1) return true;
    if (form < 2) return false;
    for (const StepEntry& e : es)
        if (e.is_prefill) return false;
    return true;
}

/* after a step's head decided whether it is a tail band pass */
inline void note(Gap& g, int form, bool band, const std::vector<StepEntry>& es) {
    if (form <= 0) return;
    const bool w = written(form, band, es);
    for (const StepEntry& e : es) {
        const int64_t ctx = e.ctx_len, end = (int64_t)e.ctx_len + e.n_tokens;
        const auto it = g.find(e.seq);
        if (e.ckpt_restore >= 0 || e.cr_replay) {   /* the positions before this step came from a snapshot / the prefix cache */
            g[e.seq] = w ? ctx : end;
        } else if (!w) {
            g[e.seq] = end;
        } else if (it == g.end()) {
            g[e.seq] = ctx == 0 ? -1 : ctx;   /* first seen with context: its earlier rows are not known to be on the leader */
        }
    }
    if (g.size() > 65536) g.clear();   /* never finished: forgotten (their steps then keep W(h, t)'s) */
}

/* whether every sequence of a step has its band state on the leader (the step may run as a tail band pass): a fresh prompt
 * row (ctx 0) is written by the band pass itself */
inline bool held(const Gap& g, int form, const std::vector<StepEntry>& es) {
    if (form <= 0) return false;
    for (const StepEntry& e : es) {
        if (e.ckpt_restore >= 0 || e.cr_replay) return false;
        if (e.is_prefill && e.ctx_len == 0) continue;
        const auto it = g.find(e.seq);
        if (it == g.end()) return false;
        if (it->second < 0) continue;
        if (form < 2 || (int64_t)e.ctx_len - it->second < form) return false;
    }
    return true;
}

}  // namespace rad::tcopy
