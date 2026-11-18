/* pp_pftl_fol.h -- tp8 loop P22 (RADIANCE_PP_PF_TIMELINE=1 on the follower): the follower's side of a lone cold prompt's
 * timeline. The leader's line (engine_lanes.cpp PfTimeline) times its heads and how long each tail waited on the card for X2,
 * not what this node did meanwhile; here card-local events go onto rank 0's stream around every lone prefill pass (A before its
 * release, B after its join -- the pass's X1 receive is its first op, so B - A is the pass's card time WITH its wait for X1),
 * and the receive rings' D8 counters (RadPpRing::wait_ticks / waits, armed here as RADIANCE_PP_NODE_IDLE arms them) give the
 * prompt's X1 waits. A pass at context 0 opens a prompt; the prompt's line is printed once a run frame that is not one of its
 * passes arrives (its first decode step, the next prompt) and every event of it has completed -- the leader ships that frame
 * only after the prompt's last tail, which waited for this node's last X2, so by then the passes have run. Events come from a
 * pool (at most kCap) and are read with rad_event_query: nothing waits on them. One prompt at a time: two cold prompts whose
 * chunks interleave are read as one. */
#pragma once

#include "rad_internal.h"   /* RAD_INFO, fmt, rad_device.h */
#include "rad_pp.h"
#include "device/device.h"   /* rad_event_create_local */

#include <algorithm>
#include <climits>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace rad {

struct PfFolTimeline {
    static constexpr size_t kCap = 1024;
    struct Pass { RadEvent a = nullptr, b = nullptr; int64_t rows = 0, ctx = 0; int lane = -1; int32_t ahead = INT32_MIN;
                  float gap = -1.0f, ms = -1.0f; };
    RadStream s = nullptr;
    std::vector<RadPpRing*> rings;   /* this node's X1 receive rings, every lane's */
    std::vector<RadEvent> pool;
    size_t made = 0;
    std::vector<Pass> ps;            /* the open prompt's passes */
    size_t next = 0;                 /* the first pass whose events are not read yet */
    RadEvent prev_b = nullptr;       /* the last read pass's end (the gap before the next) */
    bool open = false, closing = false;
    uint64_t w0 = 0, n0 = 0;         /* the rings' wait counters at the prompt's first pass */
    uint64_t w1 = 0, n1 = 0, tp1 = 0; /* ... and when the prompt closed (before the next pass's issue: its receive not in) */
    int64_t at = 0;                  /* the open prompt's rows so far: the next pass's context */
    int64_t dropped = 0, unread = 0, prompts = 0;

    ~PfFolTimeline() {
        for (RadEvent e : pool) rad_event_destroy(e);
        for (Pass& p : ps) { if (p.a) rad_event_destroy(p.a); if (p.b) rad_event_destroy(p.b); }
        if (prev_b) rad_event_destroy(prev_b);
    }
    void waits(uint64_t* ticks, uint64_t* n, uint64_t* tpms) const {
        *ticks = *n = *tpms = 0;
        for (const RadPpRing* r : rings) {
            if (!r) continue;
            *ticks += __atomic_load_n(&r->wait_ticks, __ATOMIC_ACQUIRE);
            *n += __atomic_load_n(&r->waits, __ATOMIC_ACQUIRE);
            if (r->wait_tpms) *tpms = r->wait_tpms;
        }
    }
    void arm() { for (RadPpRing* r : rings) if (r) r->wait_on = 1; }
    RadEvent take() {
        if (!pool.empty()) { RadEvent e = pool.back(); pool.pop_back(); return e; }
        if (made >= kCap) return nullptr;
        RadEvent e = nullptr;
        if (rad_event_create_local(&e) < 0) return nullptr;
        ++made;
        return e;
    }
    RadEvent mark() {
        RadEvent e = take();
        if (!e) { ++dropped; return nullptr; }
        if (rad_event_record(e, s) < 0) { pool.push_back(e); ++dropped; return nullptr; }
        return e;
    }
    /* before a lone prefill pass's release (rank 0's thread parked): context 0 opens a prompt (closing one still open) */
    /* `first`: the batch's context is 0 (RadBatch::max_ctx_len is a band bound, so the context is counted here) */
    void begin(int64_t rows, bool first, int lane, int32_t ahead) {
        if (first || !open) {
            close();
            flush(true);
            open = true;
            closing = false;
            at = 0;
            uint64_t tp = 0;
            waits(&w0, &n0, &tp);
        }
        Pass p;
        p.rows = rows;
        p.ctx = at;
        at += rows;
        p.lane = lane;
        p.ahead = ahead;
        p.a = mark();
        ps.push_back(p);
    }
    /* after the pass's join */
    void end() {
        if (!ps.empty() && !ps.back().b) ps.back().b = mark();
        flush(false);
    }
    /* a run frame that is not a lone prefill pass: the open prompt is over */
    void close() {
        if (!open || closing) return;
        closing = true;
        waits(&w1, &n1, &tp1);
    }
    /* read the completed events in stream order; print the prompt once it is closing and every pass is read (`force`: now,
     * whatever is unread -- a new prompt or the stop) */
    void flush(bool force) {
        while (next < ps.size()) {
            Pass& p = ps[next];
            if (!p.b || !p.a) { ++unread; ++next; continue; }
            if (rad_event_query(p.b) != 1) break;
            float ms = 0.0f;
            if (rad_event_elapsed_ms(p.a, p.b, &ms) == 0) p.ms = ms; else ++unread;
            if (prev_b && rad_event_elapsed_ms(prev_b, p.a, &ms) == 0) p.gap = ms;
            if (prev_b) pool.push_back(prev_b);
            prev_b = p.b;
            p.b = nullptr;
            pool.push_back(p.a);
            p.a = nullptr;
            ++next;
        }
        if (!open || !closing || (next < ps.size() && !force)) return;
        print();
        for (Pass& p : ps) { if (p.a) pool.push_back(p.a); if (p.b) pool.push_back(p.b); }
        ps.clear();
        next = 0;
        if (prev_b) { pool.push_back(prev_b); prev_b = nullptr; }
        open = closing = false;
    }
    void print() {
        if (ps.empty()) return;
        ++prompts;
        const uint64_t wt = w1, wn = n1, tpms = tp1;
        const double wait_ms = tpms ? (double)(wt - w0) / (double)tpms : 0.0;
        int64_t rows = 0;
        double part = 0, gap = 0;
        int nread = 0;
        for (const Pass& p : ps) {
            rows += p.rows;
            if (p.ms >= 0) { part += p.ms; ++nread; }
            if (p.gap >= 0) gap += p.gap;
        }
        const size_t n = ps.size();
        const float l1 = n >= 1 ? ps[n - 1].ms : -1.0f, l2 = n >= 2 ? ps[n - 2].ms : -1.0f;
        RAD_INFO("follower prefill timeline (RADIANCE_PP_PF_TIMELINE=1, tp8 loop P22): prompt %lld, %zu passes, %lld rows (context "
                 "%lld .. %lld); rank 0's card: passes %.1f ms (%d read; %.2f ms a 1k rows), of which waiting for X1 %.1f ms (%lld "
                 "receives on the rings), between passes %.1f ms; the last two passes %.1f / %.1f ms; events dropped %lld, unread "
                 "%lld", (long long)prompts, n, (long long)rows, (long long)ps[0].ctx, (long long)(ps[n - 1].ctx + ps[n - 1].rows),
                 part, nread, rows > 0 ? part * 1000.0 / (double)rows : 0.0, wait_ms, (long long)(wn - n0), gap, (double)l2,
                 (double)l1, (long long)dropped, (long long)unread);
        /* every pass, 16 a line: index rows@context lane ahead: idle before | pass ms (with its X1 wait) */
        for (size_t k0 = 0; k0 < n; k0 += 16) {
            std::string t;
            for (size_t k = k0; k < n && k < k0 + 16; ++k) {
                const Pass& p = ps[k];
                t += fmt("%s%zu:%lld@%lld l%d a%s %.1f|%.1f", k > k0 ? "  " : "", k, (long long)p.rows, (long long)p.ctx, p.lane,
                         p.ahead == INT32_MIN ? "-" : std::to_string(p.ahead).c_str(), (double)p.gap, (double)p.ms);
            }
            RAD_INFO("follower prefill timeline prompt %lld passes %zu-%zu (rows@context lane ahead: idle before | pass ms with "
                     "its X1 wait; -1 = not read): %s", (long long)prompts, k0, std::min(n, k0 + 16) - 1, t.c_str());
        }
    }
};

}  // namespace rad
