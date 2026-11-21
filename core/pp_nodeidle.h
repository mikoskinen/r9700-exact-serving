/* pp_nodeidle.h -- tp8 dev D8 (RADIANCE_PP_NODE_IDLE=1, either node of a two-node pipeline; unset: nothing here runs).
 *
 * Where rank 0's card spends the wall. Every decode lever since the loop's PROFILE 20:58 is judged by node idle (FINDINGS
 * F5's one-stream bound, the M4 / M5b readings, F11), but the screen sees only the leader's host gaps > 100 ms, and the
 * card stamps (RADIANCE_AQL_STAMP) do not see a MIXED or prefill pass (unrecorded). Here card-local events go onto rank
 * 0's stream before each part's release and after its ranks issued it (the leader's heads, tails with their drafter and
 * the model parts issued ahead; the follower's passes and segments): an event completes when the stream reaches it, so
 * begin -> end is the part's card time and end -> the next begin what the card did between parts (idle while the host had
 * not issued the next one, or work outside the brackets: restores, the KV check, stream probes). A part's card time also
 * holds the card's own wait for the other node's crossing -- stage_recv spins on the card -- so rank 0's receives time it
 * (RadPpRing::wait_on / wait_ticks, abi/rad_pp.h) and the line takes it out: "working" = parts - receive waits, as a
 * share of the covered time (parts + gaps; a gap > 5 s is the server without work, left out). One line a minute and the
 * run's at stop. Events come from a pool (at most kCap in flight) and are read in stream order with
 * rad_event_query, so nothing waits on them.
 *
 * tp8 loop D45 (RADIANCE_PP_NODE_IDLE=2; =1 is the line above alone): the same card time split BY STRETCH -- with a BG step
 * in flight vs without -- in a second line beside each node-idle line. The engine says which stretch it is in: the leader
 * by its round's state (a BG lane busy or holding a newcomer: the D32 BG-rate line's definition, so the two read the same
 * stretches), the follower while a BG lane's frame is its pass (its body, segments and the passes run between them).
 * A part goes to the stretch of its begin, a gap between parts to the stretch at the part's end before it, and rank 0's
 * receive waits to the stretch at the mark before them (the rings' counters read at each mark: host time, so a wait that
 * straddles a change goes to one side). Why: FINDINGS DECODE 19:23 read WHO IS BOUND WHEN (clean stretches leader-bound,
 * BG stretches follower-bound with the leader 24-30 % idle) from stamped runs only (2.9 GB a node, stamps cost c16
 * ~-3.8 %); D39 (the tail band) and every later BG-balancing lever targets exactly that split. RADIANCE_PP_NODE_IDLE_S=N
 * (1..600, either value; default 60) sets the lines' period, so a 30-60 s cell holds whole windows ("the last N s"; the
 * text stays "the last minute" at 60).
 *
 * tp8 loop D68 (RADIANCE_PP_NODE_IDLE_PARTS=1 beside NODE_IDLE=1 / 2, either node; unset: nothing below is kept or printed): the
 * same parts' card time BY KIND AND SEQUENCES -- the engine tags each part before its begin with what it is (the leader's head,
 * head segment, tail model part, tail segment, tail = the tail's model part + the head's logits + its sampler (or the sampler
 * alone when the model part went ahead), then the drafter + step end as a part of its own -- the one place the setting changes
 * the parts: with it on, a tail is two parts, so the D8 / D45 lines count one part more a tail and the card's time between the
 * sampler and the drafter's first kernel as a gap; the follower's pass or segment), its batch's phase (D decode only / P prefill / M mixed) and sequences / rows; a part's ms goes to (stretch,
 * kind, phase, sequences) and one more line a period prints each key's p50 / p25 / p75 ms, count and mean rows (keys with >= 3
 * parts), clean stretch first. Why: DECODE round 54 (lq229's step log) found the bound node's lane part growing with SEQUENCES
 * more than with rows (c4 -> c8: ~+2.3 ms a lane part a sequence beyond rows and context) and could name it only with a stamped
 * start; this splits it at zero card cost on any start -- the leader's head (trunk layers [0, h)), its tail (trunk layers
 * [t, n) + the head's logits + sampler) and its drafter, and the follower's pass (trunk layers [h, t) only). A part's card
 * time holds the card's wait for the other node's crossing (as above): the leader's head part receives nothing. Log only: the
 * issue is the same. */
#pragma once

#include "rad_internal.h"   /* RAD_INFO, rad_mono_ns, rad_device.h */
#include "rad_pp.h"
#include "device/device.h"   /* rad_event_create_local */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace rad {

struct PpNodeIdle {
    static constexpr size_t kCap = 4096;
    static constexpr float kIdleMs = 5000.0f;   /* a gap this long is no work at all (between the test's cells) */
    struct Rec { RadEvent ev; bool begin; bool bg; uint32_t tag = 0; int32_t rows = 0; };
    /* tp8 loop D68: a part's kind (the tag's low byte), set by the engine before the part's begin */
    enum : uint32_t { kPartOther = 0, kPartHead = 1, kPartHeadSeg = 2, kPartTailModel = 3, kPartTailSeg = 4, kPartTail = 5,
                      kPartPass = 6, kPartSeg = 7, kPartSampler = 8, kPartDrafter = 9 };
    struct PAcc { std::vector<float> ms; double rows = 0; };
    struct Win {
        uint64_t t0 = 0;
        double part_ms = 0, gap_ms = 0, gap_max = 0, idle_ms = 0;
        int64_t parts = 0, gaps_1ms = 0, idles = 0;
        uint64_t wait0 = 0, waits0 = 0;   /* the rings' counters at the window's start */
        /* tp8 loop D45: by stretch, [0] without a BG step in flight, [1] with one */
        double sp_part[2] = { 0, 0 }, sp_gap[2] = { 0, 0 };
        int64_t sp_parts[2] = { 0, 0 };
        uint64_t sp_wait[2] = { 0, 0 };    /* receive-wait ticks */
        std::map<uint32_t, PAcc> pt;       /* tp8 loop D68: (stretch << 31 | phase << 24 | sequences << 8 | kind) -> the parts' ms */
    };
    /* tp8 loop D68 (RADIANCE_PP_NODE_IDLE_PARTS=1): parts_on keeps the parts by kind; tag() names the next begin's part */
    bool parts_on = false;
    uint32_t next_tag = 0;
    int32_t next_rows = 0;
    static constexpr size_t kPartKeep = 65536;   /* ms kept a key and window (the run's too): past it only the count grows */
    void tag(uint32_t kind, int phase, int64_t n_seq, int64_t n_tok) {
        if (!parts_on) return;
        const uint32_t ph = phase == 1 ? 1u : phase == 2 ? 2u : 0u;   /* RAD_PHASE_DECODE 1 / MIXED 2 / PREFILL 0 */
        const uint32_t ns = (uint32_t)std::min<int64_t>(std::max<int64_t>(n_seq, 0), 65535);
        next_tag = (ph << 24) | (ns << 8) | (kind & 0xffu);
        next_rows = (int32_t)std::min<int64_t>(std::max<int64_t>(n_tok, 0), 1 << 30);
    }
    /* tp8 loop D45 (RADIANCE_PP_NODE_IDLE=2): the stretch the engine is in -- the leader sets `bg` each round, the follower
     * counts BG frames in pass (`bg_depth`); split prints the second line; period_s the lines' period */
    bool split = false;
    int period_s = 60;
    bool bg = false;
    int bg_depth = 0;
    bool sp_prev = false;
    uint64_t sp_wt_last = 0;
    bool bg_now() const { return bg || bg_depth > 0; }
    /* tp8 loop D51 (fn4 pp_tband 5; RADIANCE_PP_LOAD_REPORT_MS on the follower): the card's RECENT working share -- cumulative
     * part / gap ms (as the lines count them) and the receive waits, sampled at every mark while recent_keep_ns > 0 (0 = off,
     * nothing kept: today); recent_pm(w) = (parts - waits) / (parts + gaps) between the newest sample and the last one at least
     * w old, in permille, -1 while under a quarter of w is covered */
    struct Smp { uint64_t t; double part, gap, wait; };
    uint64_t recent_keep_ns = 0;
    double cum_part = 0, cum_gap = 0;
    std::deque<Smp> smp;
    int recent_pm(uint64_t w_ns) const {
        if (smp.size() < 2) return -1;
        const Smp& b = smp.back();
        const Smp* a = &smp.front();
        for (const Smp& x : smp) {
            if (x.t + w_ns > b.t) break;
            a = &x;
        }
        const double cov = (b.part - a->part) + (b.gap - a->gap);
        if (cov <= 0 || cov * 4e6 < (double)w_ns) return -1;
        const double wk = (b.part - a->part) - std::max(0.0, b.wait - a->wait);
        return (int)std::lround(std::min(1000.0, std::max(0.0, wk * 1000.0 / cov)));
    }
    const char* node = "";                /* "leader" / "follower" */
    const char* crossing = "";            /* what rank 0 receives: "X2" / "X1" */
    RadStream s = nullptr;
    std::vector<RadPpRing*> rings;        /* this node's receive rings, every lane's */
    std::vector<RadEvent> pool;
    size_t made = 0;
    std::deque<Rec> recs;
    Rec last{ nullptr, false };
    Win win, run;
    int64_t dropped = 0, unread = 0;
    /* tp8 loop DECODE round 20 (FINDINGS 06:17, lq166: a follower whose cards ran 11-161 MiB from full stopped its queues
     * under KFD evictions): this node's cards' VRAM by the driver's counter (dev_vram_used: every process's bytes), sampled
     * at most once a second at the marks; the least free a card had in the window and the run. A card whose counter is not
     * readable (the host backend) is named so. Read-only: nothing here allocates or waits. */
    static constexpr int64_t kVramLowMiB = 64;
    std::vector<int> devs;               /* the node's ranks' devices, rank order */
    std::vector<int64_t> totals;         /* their VRAM bytes (rad_dev_props), -1 unknown */
    std::vector<int64_t> wmin, rmin;     /* the least free bytes seen in the window / the run, -1 none read */
    uint64_t vram_t = 0;
    void vram_sample(bool force = false) {
        const uint64_t now = rad_mono_ns();
        if (devs.empty() || (!force && now - vram_t < 1000000000ull)) return;
        vram_t = now;
        if (wmin.size() != devs.size()) { wmin.assign(devs.size(), -1); rmin.assign(devs.size(), -1); }
        for (size_t i = 0; i < devs.size(); ++i) {
            int64_t used = 0;
            if (i >= totals.size() || totals[i] <= 0 || dev_vram_used(devs[i], &used) != RAD_OK) continue;
            const int64_t fr = totals[i] - used;
            if (wmin[i] < 0 || fr < wmin[i]) wmin[i] = fr;
            if (rmin[i] < 0 || fr < rmin[i]) rmin[i] = fr;
        }
    }
    void vram_note(const std::vector<int64_t>& m, const char* what) const {
        if (devs.empty()) return;
        std::string per;
        int low = 0, read = 0;
        for (size_t i = 0; i < devs.size(); ++i) {
            const int64_t v = i < m.size() ? m[i] : -1;
            per += (i ? ", card " : "card ") + std::to_string(devs[i]) + " " +
                   (v < 0 ? std::string("not readable") : std::to_string(v >> 20) + " MiB");
            if (v >= 0) {
                ++read;
                if ((v >> 20) < kVramLowMiB) ++low;
            }
        }
        RAD_INFO("two-node pipeline, %s node VRAM (RADIANCE_PP_NODE_IDLE=1): the least free a card had over %s, by the driver's "
                 "counter: %s%s", node, what, per.c_str(),
                 !read ? " (no card's counter readable here)"
                       : low ? (std::string(" -- ") + std::to_string(low) + " card(s) under " + std::to_string(kVramLowMiB) +
                                " MiB: nothing more builds on first use there without the driver evicting (passes not "
                                "recorded, queues stopped)").c_str()
                             : "");
    }

    ~PpNodeIdle() {
        for (RadEvent e : pool) rad_event_destroy(e);
        for (const Rec& r : recs) rad_event_destroy(r.ev);
        if (last.ev) rad_event_destroy(last.ev);
    }
    /* the rings are the core's (pinned host memory): count from now on */
    void arm() {
        for (RadPpRing* r : rings)
            if (r) r->wait_on = 1;
        start(win);
        start(run);
        uint64_t n = 0, tp = 0;
        waits(&sp_wt_last, &n, &tp);
    }
    void waits(uint64_t* ticks, uint64_t* n, uint64_t* tpms) const {
        *ticks = *n = 0;
        *tpms = 0;
        for (const RadPpRing* r : rings) {
            if (!r) continue;
            *ticks += __atomic_load_n(&r->wait_ticks, __ATOMIC_ACQUIRE);
            *n += __atomic_load_n(&r->waits, __ATOMIC_ACQUIRE);
            if (r->wait_tpms) *tpms = r->wait_tpms;
        }
    }
    void start(Win& w) {
        w = Win{};
        w.t0 = rad_mono_ns();
        uint64_t tp = 0;
        waits(&w.wait0, &w.waits0, &tp);
    }
    RadEvent take() {
        if (!pool.empty()) { RadEvent e = pool.back(); pool.pop_back(); return e; }
        if (made >= kCap) return nullptr;
        RadEvent e = nullptr;
        if (rad_event_create_local(&e) < 0) return nullptr;
        ++made;
        return e;
    }
    /* on rank 0's stream, with its rank thread parked (before a release / after the join) */
    void mark(bool begin) {
        drain();
        vram_sample();
        const bool now_bg = bg_now();
        if (split) {   /* tp8 loop D45: the receive waits since the last mark, to the stretch in effect since then */
            uint64_t wt = 0, n = 0, tp = 0;
            waits(&wt, &n, &tp);
            const uint64_t d = wt >= sp_wt_last ? wt - sp_wt_last : 0;
            sp_wt_last = wt;
            win.sp_wait[sp_prev] += d;
            run.sp_wait[sp_prev] += d;
            sp_prev = now_bg;
        }
        if (recent_keep_ns) {   /* tp8 loop D51: a sample of the cumulative counts (the drain above brought them up to date) */
            uint64_t wt = 0, n = 0, tp = 0;
            waits(&wt, &n, &tp);
            const uint64_t now = rad_mono_ns();
            smp.push_back({ now, cum_part, cum_gap, tp ? (double)wt / (double)tp : 0.0 });
            while (smp.size() > 2 && (smp.front().t + recent_keep_ns < now || smp.size() > 8192)) smp.pop_front();
        }
        RadEvent e = take();
        if (!e) { ++dropped; return; }
        if (rad_event_record(e, s) < 0) { pool.push_back(e); ++dropped; return; }
        recs.push_back({ e, begin, now_bg, begin ? next_tag : 0u, begin ? next_rows : 0 });
        if (begin) next_tag = 0, next_rows = 0;   /* tp8 loop D68: an untagged part is 'other' */
    }
    void begin() { mark(true); }
    void end() { mark(false); }
    /* the records whose events completed, in stream order: the time since the one before */
    void drain() {
        while (!recs.empty() && rad_event_query(recs.front().ev) == 1) {
            const Rec r = recs.front();
            recs.pop_front();
            if (last.ev) {
                float ms = 0.0f;
                if (rad_event_elapsed_ms(last.ev, r.ev, &ms) < 0) ++unread;
                else if (last.begin) {   /* a part (a begin with no end before the next begin: still that part) */
                    cum_part += ms;   /* tp8 loop D51 */
                    for (Win* w : { &win, &run }) {
                        w->part_ms += ms;
                        ++w->parts;
                        w->sp_part[last.bg] += ms;   /* tp8 loop D45: the stretch at the part's begin */
                        ++w->sp_parts[last.bg];
                        if (parts_on) {   /* tp8 loop D68 */
                            PAcc& a = w->pt[(last.bg ? 1u << 31 : 0u) | last.tag];
                            if (a.ms.size() < kPartKeep) a.ms.push_back(ms);
                            a.rows += last.rows;
                        }
                    }
                } else if (r.begin) {    /* between two parts; one over kIdleMs is the server without work, left out */
                    if (ms <= kIdleMs) cum_gap += ms;   /* tp8 loop D51 */
                    for (Win* w : { &win, &run }) {
                        if (ms > kIdleMs) { w->idle_ms += ms; ++w->idles; continue; }
                        w->gap_ms += ms;
                        w->sp_gap[last.bg] += ms;    /* tp8 loop D45: the stretch at the end of the part before */
                        if (ms > 1.0f) ++w->gaps_1ms;
                        w->gap_max = std::max(w->gap_max, (double)ms);
                    }
                }
                pool.push_back(last.ev);
            }
            last = r;
        }
    }
    void note(const Win& w, const char* what) {
        uint64_t wt = 0, wn = 0, tpms = 0;
        waits(&wt, &wn, &tpms);
        const double wall = (double)(rad_mono_ns() - w.t0) / 1e6;
        const double wait_ms = tpms ? (double)(wt - w.wait0) / (double)tpms : 0.0;
        const int64_t nw = (int64_t)(wn - w.waits0);
        const double cov = w.part_ms + w.gap_ms;   /* the time the card had parts to run or was between two of them */
        const auto pct = [&](double ms) { return cov > 0 ? ms * 100.0 / cov : 0.0; };
        RAD_INFO("two-node pipeline, %s node idle (RADIANCE_PP_NODE_IDLE=%d, tp8 dev D8): rank 0's card over %s (%.1f s wall, "
                 "%.1f s covered; %lld gaps > %.0f s left out, %.1f s): in parts %.1f s (%.1f %%, %lld parts), of which waiting "
                 "on the card for the other node's %s %.1f s (%.1f %%, %lld receives); between parts %.1f s (%.1f %%: idle, or "
                 "work outside the parts; %lld gaps > 1 ms, the longest %.1f ms) -> working %.1f %%; events dropped %lld, unread "
                 "%lld", node, split ? 2 : 1, what, wall / 1000.0, cov / 1000.0, (long long)w.idles, kIdleMs / 1000.0, w.idle_ms / 1000.0,
                 w.part_ms / 1000.0, pct(w.part_ms), (long long)w.parts, crossing, wait_ms / 1000.0, pct(wait_ms), (long long)nw,
                 w.gap_ms / 1000.0, pct(w.gap_ms), (long long)w.gaps_1ms, w.gap_max, pct(w.part_ms - wait_ms),
                 (long long)dropped, (long long)unread);
        if (!split) return;
        std::string sp;
        for (int k = 1; k >= 0; --k) {   /* no receive yet (no tick rate): its waits read 0 */
            const double c = w.sp_part[k] + w.sp_gap[k], wm = tpms ? (double)w.sp_wait[k] / (double)tpms : 0.0;
            const auto q = [&](double ms) { return c > 0 ? ms * 100.0 / c : 0.0; };
            char b[400];
            std::snprintf(b, sizeof b, "%s %.1f s covered (%.1f %% of it): in parts %.1f %% (%lld parts), waiting on the card for the "
                          "other node's %s %.1f %%, between parts %.1f %% -> working %.1f %%", k ? "with a BG step in flight" : "; without",
                          c / 1000.0, pct(c), q(w.sp_part[k]), (long long)w.sp_parts[k], crossing, q(wm), q(w.sp_gap[k]),
                          q(w.sp_part[k] - wm));
            sp += b;
        }
        RAD_INFO("two-node pipeline, %s node idle by stretch (RADIANCE_PP_NODE_IDLE=2, tp8 loop D45): rank 0's card over %s: %s", node,
                 what, sp.c_str());
    }
    /* tp8 loop D68: "head D 4 seq: p50 20.31 ms (p25 19.90, p75 20.88; 494 parts, 23.6 rows)" for each key with >= 3 parts */
    static const char* kind_name(uint32_t k) {
        switch (k) {
        case kPartHead: return "head";
        case kPartHeadSeg: return "head segment";
        case kPartTailModel: return "tail model part";
        case kPartTailSeg: return "tail segment";
        case kPartTail: return "tail (+ logits, sampler)";
        case kPartPass: return "pass";
        case kPartSeg: return "segment";
        case kPartSampler: return "sampler";
        case kPartDrafter: return "drafter + step end";
        default: return "other";
        }
    }
    void parts_note(Win& w, const char* what) const {
        if (!parts_on) return;
        for (int bg = 0; bg <= 1; ++bg) {
            std::string o;
            int64_t n_all = 0;
            for (auto& [key, a] : w.pt) {
                if ((key >> 31) != (uint32_t)bg || a.ms.size() < 3) continue;
                std::vector<float> v = a.ms;
                std::sort(v.begin(), v.end());
                const auto q = [&](double f) { return (double)v[std::min(v.size() - 1, (size_t)(f * (double)(v.size() - 1) + 0.5))]; };
                const uint32_t k = key & 0xffu, ns = (key >> 8) & 0xffffu, ph = (key >> 24) & 0x3u;
                char b[256];
                std::snprintf(b, sizeof b, "%s%s %c %u seq: p50 %.2f ms (p25 %.2f, p75 %.2f; %zu parts, %.1f rows)", o.empty() ? "" : "; ",
                              kind_name(k), ph == 1 ? 'D' : ph == 2 ? 'M' : 'P', ns, q(0.5), q(0.25), q(0.75), v.size(),
                              a.rows / (double)v.size());
                o += b;
                n_all += (int64_t)v.size();
            }
            if (o.empty()) continue;
            RAD_INFO("two-node pipeline, %s node parts by kind and sequences (RADIANCE_PP_NODE_IDLE_PARTS=1, tp8 loop D68): rank 0's card "
                     "over %s, %s (%lld parts; D decode only, P prefill, M mixed; a part's ms holds its wait for the other node's %s): %s",
                     node, what, bg ? "with a BG step in flight" : "without a BG step in flight", (long long)n_all, crossing, o.c_str());
        }
    }
    void minute() {
        if (rad_mono_ns() - win.t0 < (uint64_t)period_s * 1000000000ull) return;
        drain();
        const std::string what = period_s == 60 ? std::string("the last minute") : "the last " + std::to_string(period_s) + " s";
        note(win, what.c_str());
        parts_note(win, what.c_str());   /* tp8 loop D68 */
        vram_sample(true);
        vram_note(wmin, what.c_str());
        std::fill(wmin.begin(), wmin.end(), (int64_t)-1);
        start(win);
    }
    void stop() {
        drain();
        note(run, "the run");
        parts_note(run, "the run");   /* tp8 loop D68 */
        vram_sample(true);
        vram_note(rmin, "the run");
    }
};

}  // namespace rad
