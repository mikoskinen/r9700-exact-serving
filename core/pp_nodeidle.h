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
 * rad_event_query, so nothing waits on them. */
#pragma once

#include "rad_internal.h"   /* RAD_INFO, rad_mono_ns, rad_device.h */
#include "rad_pp.h"
#include "device/device.h"   /* rad_event_create_local */

#include <algorithm>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace rad {

struct PpNodeIdle {
    static constexpr size_t kCap = 4096;
    static constexpr float kIdleMs = 5000.0f;   /* a gap this long is no work at all (between the test's cells) */
    struct Rec { RadEvent ev; bool begin; };
    struct Win {
        uint64_t t0 = 0;
        double part_ms = 0, gap_ms = 0, gap_max = 0, idle_ms = 0;
        int64_t parts = 0, gaps_1ms = 0, idles = 0;
        uint64_t wait0 = 0, waits0 = 0;   /* the rings' counters at the window's start */
    };
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
        RadEvent e = take();
        if (!e) { ++dropped; return; }
        if (rad_event_record(e, s) < 0) { pool.push_back(e); ++dropped; return; }
        recs.push_back({ e, begin });
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
                    for (Win* w : { &win, &run }) { w->part_ms += ms; ++w->parts; }
                } else if (r.begin) {    /* between two parts; one over kIdleMs is the server without work, left out */
                    for (Win* w : { &win, &run }) {
                        if (ms > kIdleMs) { w->idle_ms += ms; ++w->idles; continue; }
                        w->gap_ms += ms;
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
        RAD_INFO("two-node pipeline, %s node idle (RADIANCE_PP_NODE_IDLE=1, tp8 dev D8): rank 0's card over %s (%.1f s wall, "
                 "%.1f s covered; %lld gaps > %.0f s left out, %.1f s): in parts %.1f s (%.1f %%, %lld parts), of which waiting "
                 "on the card for the other node's %s %.1f s (%.1f %%, %lld receives); between parts %.1f s (%.1f %%: idle, or "
                 "work outside the parts; %lld gaps > 1 ms, the longest %.1f ms) -> working %.1f %%; events dropped %lld, unread "
                 "%lld", node, what, wall / 1000.0, cov / 1000.0, (long long)w.idles, kIdleMs / 1000.0, w.idle_ms / 1000.0,
                 w.part_ms / 1000.0, pct(w.part_ms), (long long)w.parts, crossing, wait_ms / 1000.0, pct(wait_ms), (long long)nw,
                 w.gap_ms / 1000.0, pct(w.gap_ms), (long long)w.gaps_1ms, w.gap_max, pct(w.part_ms - wait_ms),
                 (long long)dropped, (long long)unread);
    }
    void minute() {
        if (rad_mono_ns() - win.t0 < 60ull * 1000000000ull) return;
        drain();
        note(win, "the last minute");
        vram_sample(true);
        vram_note(wmin, "the last minute");
        std::fill(wmin.begin(), wmin.end(), (int64_t)-1);
        start(win);
    }
    void stop() {
        drain();
        note(run, "the run");
        vram_sample(true);
        vram_note(rmin, "the run");
    }
};

}  // namespace rad
