/* pp_pftl_fol.h -- tp8 loop P22 (RADIANCE_PP_PF_TIMELINE=1 on the follower): the follower's side of a lone cold prompt's
 * timeline. The leader's line (engine_lanes.cpp PfTimeline) times its heads and how long each tail waited on the card for X2,
 * not what this node did meanwhile; here card-local events go onto rank 0's stream around every lone prefill pass (A before its
 * release, B after its join -- the pass's X1 receive is its first op, so B - A is the pass's card time WITH its wait for X1),
 * and the receive rings' D8 counters (RadPpRing::wait_ticks / waits, armed here as RADIANCE_PP_NODE_IDLE arms them) give the
 * prompt's X1 waits. A pass at context 0 opens a prompt; the prompt's line is printed once a run frame that is not one of its
 * passes arrives (its first decode step, the next prompt) and every event of it has completed -- the leader ships that frame
 * only after the prompt's last tail, which waited for this node's last X2, so by then the passes have run. Events come from a
 * pool (at most kCap) and are read with rad_event_query: nothing waits on them. One prompt at a time: two cold prompts whose
 * chunks interleave are read as one.
 *
 * tp8 loop P22b: a SPLIT lone prefill pass (a BG newcomer's body in layer segments, RADIANCE_PP_SEG_LAYERS, with other lanes'
 * decode passes run between them) cannot be read that way -- one A / B pair around it holds those passes too, and a decode frame
 * run between its segments (or the next BG lane's newcomer) closed the open prompt before its events were in (M lq151: '0 read,
 * unread 126'). Such a pass gets its own record: events around EVERY segment (A_k before its release, B_k after its join: on
 * the in-order stream B_k - A_k is segment k's card time, B_k .. A_k+1 the passes run between), its own lane's X1 ring counter
 * (the receive is segment 0's first op; read once segment 0 has run and before the lane's next pass is out), and one line once
 * every event has completed, in P22's form (kit p27read.py reads it: "passes" = the segments' own ms) with the segments, the
 * passes between and the span; a summary line a minute and the run's at stop (rows, own ms, X1 wait a newcomer). */
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

    /* tp8 loop P22b: a split lone prefill pass's record */
    struct SegRec {
        int64_t rows = 0;
        bool first = false;              /* at context 0 */
        int lane = -1;
        int32_t ahead = INT32_MIN;
        std::vector<RadEvent> a, b;      /* around each segment */
        RadPpRing* ring = nullptr;       /* its lane's X1 receive ring */
        uint64_t w0 = 0, n0 = 0, w1 = 0, n1 = 0, tpms = 0;
        bool x1 = false;                 /* w1 / n1 read */
        int64_t between = 0, gen = 0;    /* passes run between its segments; its lane's pass count at its begin */
        bool issued = false;
    };
    struct SegStat { double own, x1, s0, btw, span; int64_t rows, between; };
    std::deque<SegRec> sq;
    bool seg_open = false;               /* a split pass's segments are going out (a pass between them is not one) */
    std::vector<int64_t> lane_gen;       /* passes begun a lane (P22's and P22b's) */
    std::vector<SegStat> win, all;       /* read records: this minute's, the run's */
    uint64_t win_t0 = 0;
    int64_t seg_unread_ev = 0, seg_unread_x1 = 0;

    ~PfFolTimeline() {
        for (RadEvent e : pool) rad_event_destroy(e);
        for (Pass& p : ps) { if (p.a) rad_event_destroy(p.a); if (p.b) rad_event_destroy(p.b); }
        if (prev_b) rad_event_destroy(prev_b);
        for (SegRec& r : sq) {
            for (RadEvent e : r.a) if (e) rad_event_destroy(e);
            for (RadEvent e : r.b) if (e) rad_event_destroy(e);
        }
    }
    int64_t& gen_of(int lane) {
        const size_t i = lane < 0 ? 0 : (size_t)lane;
        if (lane_gen.size() <= i) lane_gen.resize(i + 1, 0);
        return lane_gen[i];
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
        ++gen_of(lane);
    }
    /* tp8 loop P22b: a split lone prefill pass begins (rank 0's thread parked; its segments follow) */
    void seg_begin(int64_t rows, bool first, int lane, int32_t ahead, RadPpRing* ring) {
        SegRec r;
        r.rows = rows;
        r.first = first;
        r.lane = lane;
        r.ahead = ahead;
        r.ring = ring;
        if (ring) {
            r.w0 = __atomic_load_n(&ring->wait_ticks, __ATOMIC_ACQUIRE);
            r.n0 = __atomic_load_n(&ring->waits, __ATOMIC_ACQUIRE);
        }
        r.gen = ++gen_of(lane);
        sq.push_back(std::move(r));
        seg_open = true;
    }
    /* before segment k's release / after its join */
    void seg_a() { if (seg_open && !sq.empty()) sq.back().a.push_back(mark()); }
    void seg_b() { if (seg_open && !sq.empty()) sq.back().b.push_back(mark()); }
    /* every segment is out: `between` other passes ran between them */
    void seg_end(int64_t between) {
        if (!seg_open || sq.empty()) return;
        seg_open = false;
        SegRec& r = sq.back();
        r.issued = true;
        r.between = between;
        seg_x1(r);
        seg_flush(false);
    }
    /* the lane's X1 counter once segment 0 (its receive) has run, and only while no later pass of the lane is out */
    void seg_x1(SegRec& r) {
        if (r.x1 || !r.ring || r.b.empty() || !r.b[0] || rad_event_query(r.b[0]) != 1 || gen_of(r.lane) != r.gen) return;
        r.w1 = __atomic_load_n(&r.ring->wait_ticks, __ATOMIC_ACQUIRE);
        r.n1 = __atomic_load_n(&r.ring->waits, __ATOMIC_ACQUIRE);
        r.tpms = r.ring->wait_tpms;
        r.x1 = true;
    }
    static double p50(std::vector<double> v) {
        if (v.empty()) return -1.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    void seg_summary(const std::vector<SegStat>& v, const char* what) {
        if (v.empty()) return;
        std::vector<double> own, x1, s0, btw, span, rows, between;
        double own_sum = 0, x1_sum = 0;
        int64_t rsum = 0, nx1 = 0;
        for (const SegStat& s : v) {
            own.push_back(s.own); own_sum += s.own; s0.push_back(s.s0); btw.push_back(s.btw); span.push_back(s.span);
            rows.push_back((double)s.rows); between.push_back((double)s.between); rsum += s.rows;
            if (s.x1 >= 0) { x1.push_back(s.x1); x1_sum += s.x1; ++nx1; }
        }
        RAD_INFO("follower BG bodies (RADIANCE_PP_PF_TIMELINE=1, tp8 loop P22b) %s: %zu newcomers, %lld rows (p50 %.0f); rank 0's "
                 "card a newcomer: its own segments p50 %.1f / mean %.1f ms (%.2f ms a 1k rows), of which waiting for X1 p50 %.1f / "
                 "mean %.1f ms (%lld read), segment 0 p50 %.1f ms; other passes between its segments p50 %.0f, the card between "
                 "them p50 %.1f ms; span p50 %.1f ms; events unread %lld, X1 unread %lld",
                 what, v.size(), (long long)rsum, p50(rows), p50(own), own_sum / (double)v.size(),
                 rsum > 0 ? own_sum * 1000.0 / (double)rsum : 0.0, p50(x1), nx1 ? x1_sum / (double)nx1 : -1.0, (long long)nx1,
                 p50(s0), p50(between), p50(btw), p50(span), (long long)seg_unread_ev, (long long)seg_unread_x1);
    }
    /* read the completed split passes in issue order (`force`: now, whatever is unread -- the stop) */
    void seg_flush(bool force) {
        while (!sq.empty()) {
            SegRec& r = sq.front();
            if (!r.issued) break;
            seg_x1(r);
            RadEvent last = nullptr;
            for (RadEvent e : r.b) if (e) last = e;
            if (!force && last && rad_event_query(last) != 1) break;
            auto ok = [](RadEvent e) { return e && rad_event_query(e) == 1; };
            const size_t n = std::min(r.a.size(), r.b.size());
            double own = 0, btw = 0, s0 = -1, span = -1;
            int nread = 0;
            float ms = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                if (ok(r.a[k]) && ok(r.b[k]) && rad_event_elapsed_ms(r.a[k], r.b[k], &ms) == 0) {
                    own += ms;
                    ++nread;
                    if (k == 0) s0 = ms;
                } else {
                    ++seg_unread_ev;
                }
                if (k + 1 < n && ok(r.b[k]) && ok(r.a[k + 1]) && rad_event_elapsed_ms(r.b[k], r.a[k + 1], &ms) == 0) btw += ms;
            }
            if (n > 0 && ok(r.a[0]) && ok(r.b[n - 1]) && rad_event_elapsed_ms(r.a[0], r.b[n - 1], &ms) == 0) span = ms;
            double x1 = -1.0;
            if (r.x1 && r.tpms) x1 = (double)(r.w1 - r.w0) / (double)r.tpms;
            else ++seg_unread_x1;
            ++prompts;
            RAD_INFO("follower prefill timeline (RADIANCE_PP_PF_TIMELINE=1, tp8 loop P22b): prompt %lld, 1 passes, %lld rows (%s) in "
                     "%zu segments, lane %d, ahead %s; rank 0's card: passes %.1f ms (%d read; %.2f ms a 1k rows) = its own segments, "
                     "of which waiting for X1 %.1f ms (%lld receives on the rings) and segment 0 (the receive + its first layer) "
                     "%.1f ms; %lld other passes ran between its segments, the card %.1f ms between them; span %.1f ms; events "
                     "dropped %lld", (long long)prompts, (long long)r.rows, r.first ? "context 0" : "a resumed prompt", n, r.lane,
                     r.ahead == INT32_MIN ? "-" : std::to_string(r.ahead).c_str(), own, nread,
                     r.rows > 0 ? own * 1000.0 / (double)r.rows : 0.0, x1, r.x1 ? (long long)(r.n1 - r.n0) : -1LL, s0,
                     (long long)r.between, btw, span, (long long)dropped);
            if (nread == (int)n && n > 0) {
                const SegStat s{ own, x1, s0, btw, span, r.rows, r.between };
                if (win.empty()) win_t0 = rad_mono_ns();
                win.push_back(s);
                all.push_back(s);
            }
            for (RadEvent e : r.a) if (e) pool.push_back(e);
            for (RadEvent e : r.b) if (e) pool.push_back(e);
            sq.pop_front();
        }
        if (!win.empty() && rad_mono_ns() - win_t0 >= 60ull * 1000000000ull) {
            seg_summary(win, fmt("in the last %.0f s", (double)(rad_mono_ns() - win_t0) / 1e9).c_str());
            win.clear();
        }
    }
    /* the run's end: whatever is open is read now, the minute's and the run's summaries */
    void stop() {
        close();
        flush(true);
        seg_open = false;
        for (SegRec& r : sq) r.issued = true;
        seg_flush(true);
        if (!win.empty()) seg_summary(win, fmt("in the last %.0f s", (double)(rad_mono_ns() - win_t0) / 1e9).c_str());
        win.clear();
        seg_summary(all, "over the run");
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
        seg_flush(false);   /* tp8 loop P22b: the split passes read on their own (never forced by a cold prompt) */
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
