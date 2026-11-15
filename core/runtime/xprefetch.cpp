/* xprefetch.cpp -- the expert prefetch's host half: Ctx::xp_*. See xprefetch.h for what it does. */
#include "ctx.h"
#include "xp_state.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <immintrin.h>

#if !(defined(RAD_HAVE_HIP) && RAD_HAVE_HIP)
/* A build without HIP has no kernels to launch: rad_xp_supported is 0 there, so nothing calls these. */
extern "C" int rad_xp_logits_launch(const uint16_t*, const uint16_t*, float*, uint32_t, uint32_t, uint32_t, void*) {
    return -1;
}
extern "C" int rad_xp_plan_launch(const float*, uint32_t, uint32_t, uint32_t, void** const*, const int32_t*,
                                  const int32_t*, rad::XpPools, uint64_t, uint64_t, rad::XpRec*, int32_t*,
                                  rad::XpBox*, uint32_t, rad::XpStats*, void*) {
    return -1;
}
extern "C" int rad_xp_step_launch(const rad::XpRec*, const int32_t*, const rad::XpRec*, const int32_t*,
                                  const uint32_t*, uint32_t, void*) {
    return -1;
}
extern "C" int rad_xpf_count_launch(const uint16_t*, uint32_t, uint32_t, uint32_t, uint32_t, const float*, int32_t*,
                                    void*) {
    return -1;
}
extern "C" int rad_xpf_plan_launch(int32_t*, uint32_t, void** const*, const int32_t*, const int32_t*, rad::XpPools,
                                   uint64_t, uint64_t, uint32_t, rad::XpfRec*, int32_t*, rad::XpfBox*, uint32_t,
                                   rad::XpfStats*, void*) {
    return -1;
}
extern "C" int rad_xpf_step_launch(const rad::XpfRec*, const int32_t*, const uint32_t*, uint32_t, rad::XpfStats*, uint32_t*,
                                   void*) {
    return -1;
}
extern "C" int rad_xpf_restore_launch(rad::XpfRec*, int32_t*, rad::XpfStats*, void*) { return -1; }
#endif

namespace rad {

namespace {

/* A pool class wider than this holds whole tables (the token embedding read in place), not experts. */
constexpr uint64_t kXpMaxUnit = (uint64_t)64 << 20;
/* Lists the queue may be behind before a request is dropped. Nothing else throttles a lagging
 * queue: its copies would land too late anyway, and an ever-growing queue would end up polling
 * marks already returned to the pool. */
constexpr int64_t kXpBehind = 3;
/* Dispatches after a dropped request's mark before its signal may be reused; far more than the
 * pass that completes it. */
constexpr int64_t kXpMarkHold = 4096;
/* Payoff. A used copy saves link time under the GEMM; an unused one costs link time the GEMMs
 * wanted, and the prediction always costs a little. On Qwen3.8-Flash-Next break-even is near a
 * quarter of copies used (--tp 1 used half and decode was 3.6% faster; --tp 2, where the pool holds
 * rarely routed experts, used a sixth and was 4-6% slower). Every kXpWindow predictions the share
 * used over the last window is judged; below kXpMinUse the prefetch rests for kXpRest passes. */
constexpr int64_t  kXpWindow    = 2048;
constexpr double   kXpMinUse    = 0.30;
constexpr uint32_t kXpMinJudged = 512;
constexpr int64_t  kXpRest      = 8192;

}  // namespace

void Ctx::xp_free(XpState& x) {
    if (x.stream) rad_stream_sync(x.stream);
    if (x.ring) rad_dev_free_uncached(x.ring);
    for (void* p : {(void*)x.rec, (void*)x.nrec, (void*)x.stats, (void*)x.logits, (void*)x.maps,
                    (void*)x.places})
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    for (void* p : {(void*)x.box, (void*)x.done, (void*)x.stats_h})
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (x.tail) rad_event_destroy(x.tail);
    if (x.sync) rad_event_destroy(x.sync);
    if (x.stream) rad_stream_destroy(x.stream);
}

/* At bringup, after placement has made its pinned pools: allocates everything, or nothing and logs
 * why the prefetch is off. */
void Ctx::xp_open(const uint64_t* lo, const uint64_t* hi, const uint64_t* stride, int n) {
    if (xp_ || device_is_host() || route_.empty()) return;
    if (!rad_xp_supported(device_)) {
        RAD_INFO("rank %d: expert prefetch off: the device backend has no DMA engine to spare for it",
                 rank_);
        return;
    }
    auto x = std::make_unique<XpState>();
    x->device = device_;
    for (int i = 0; i < n && x->pools.n < kXpMaxPools; ++i) {
        if (stride[i] == 0 || stride[i] > kXpMaxUnit || hi[i] <= lo[i]) continue;
        x->pools.lo[x->pools.n] = lo[i];
        x->pools.hi[x->pools.n] = hi[i];
        x->pools.stride[x->pools.n] = stride[i];
        x->slot = std::max(x->slot, stride[i]);
        ++x->pools.n;
    }
    if (x->pools.n == 0) return;
    const int64_t nl = (int64_t)route_.size();
    const int64_t ring = (int64_t)(kXpGroups * kXpSlots * x->slot);
    if (rad_dev_set(device_) < 0) return;
    x->ring   = (char*)rad_dev_alloc_uncached(ring);
    x->rec    = (XpRec*)rad_dev_alloc((int64_t)sizeof(XpRec) * kXpGroups * kXpSlots, RAD_MEM_DEVICE);
    x->nrec   = (int32_t*)rad_dev_alloc(4 * kXpGroups, RAD_MEM_DEVICE);
    x->stats  = (XpStats*)rad_dev_alloc((int64_t)sizeof(XpStats), RAD_MEM_DEVICE);
    x->logits = (float*)rad_dev_alloc(4 * (int64_t)kXpGroups * kXpMaxRows * kXpMaxNe, RAD_MEM_DEVICE);
    x->maps   = (void**)rad_dev_alloc(nl * kXpMaxNe * kXpParts * (int64_t)sizeof(void*), RAD_MEM_DEVICE);
    x->places = (int32_t*)rad_dev_alloc(nl * kXpMaxNe * 4, RAD_MEM_DEVICE);
    x->box    = (XpBox*)rad_dev_alloc((int64_t)sizeof(XpBox) * kXpGroups, RAD_MEM_HOST_PINNED);
    x->done   = (uint32_t*)rad_dev_alloc(4 * kXpGroups, RAD_MEM_HOST_PINNED);
    x->stats_h = (XpStats*)rad_dev_alloc((int64_t)sizeof(XpStats), RAD_MEM_HOST_PINNED);
    const bool ok = x->ring && x->rec && x->nrec && x->stats && x->logits && x->maps && x->places && x->box &&
                    x->done && x->stats_h && rad_xp_stream_create(&x->stream) >= 0 &&
                    rad_event_create_local(&x->tail) >= 0 && rad_event_create_local(&x->sync) >= 0 &&
                    rad_memset_async(x->nrec, 0, 4 * kXpGroups, stream_) >= 0 &&
                    rad_memset_async(x->stats, 0, sizeof(XpStats), stream_) >= 0 && rad_stream_sync(stream_) >= 0;
    if (!ok) {
        const char* why = rad_dev_last_error();
        RAD_WARN("rank %d: expert prefetch off: %s", rank_, why && *why ? why : "no memory for it");
        xp_free(*x);
        return;
    }
    std::memset((void*)x->box, 0, sizeof(XpBox) * kXpGroups);
    std::memset(x->done, 0, 4 * kXpGroups);
    x->layers.resize((size_t)nl);
    XpState* xs = x.get();
    x->thr = std::thread([this, xs] { xp_serve(*xs); });
    RAD_INFO("rank %d: expert prefetch on: %d copies a routed layer by DMA, ring %s", rank_, kXpSlots,
             humanb(ring).c_str());
    xp_ = std::move(x);
}

void Ctx::xp_close() {
    if (!xp_) return;
    XpState& x = *xp_;
    /* Nothing may touch the ring when it is freed: issued compute work can read it through entries
     * still pointing at it, and queued DMA lists write it until they finish. */
    if (stream_) rad_stream_sync(stream_);
    if (lane1_) rad_stream_sync(lane1_);
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.stop = true;
    }
    x.cv.notify_all();
    if (x.thr.joinable()) x.thr.join();
    const auto t0 = std::chrono::steady_clock::now();
    while (rad_xp_behind(x.device) > 0) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            RAD_WARN("rank %d: the prefetch DMA queue still has %lld lists after 5 s; its ring is freed anyway",
                     rank_, (long long)rad_xp_behind(x.device));
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    /* Summary: copies planned, and how many restored copies the step routed. */
    if (x.stream && rad_memcpy_async(x.stats_h, x.stats, sizeof(XpStats), x.stream) >= 0 &&
        rad_stream_sync(x.stream) >= 0 && x.stats_h->planned)
        RAD_INFO("rank %d: expert prefetch: %u copies planned, %u of %u looked back on routed (%.1f%%), %lld "
                 "requests dropped behind a slow queue, %lld rests", rank_, x.stats_h->planned, x.stats_h->routed,
                 x.stats_h->copied, x.stats_h->copied ? 100.0 * x.stats_h->routed / x.stats_h->copied : 0.0,
                 (long long)x.dropped.load(), (long long)x.rests);
    xp_free(x);
    xp_.reset();
}

/* Host thread: hands each request's mailbox to the DMA queue once the plan has written it. */
void Ctx::xp_serve(XpState& x) {
    for (;;) {
        XpReq r;
        {
            std::unique_lock<std::mutex> lk(x.mu);
            x.cv.wait(lk, [&] { return x.stop.load() || !x.q.empty(); });
            if (x.stop) return;
            r = x.q.front();
            x.q.pop_front();
        }
        /* Return the marks of finished lists to the pool. */
        const int64_t behind = rad_xp_behind(x.device);
        while (!x.marks.empty() && x.marks.front().second <= x.lists - behind) {
            rad_ride_release(x.marks.front().first, nullptr, 0);
            x.marks.pop_front();
        }
        /* The mailbox reaches the request number once the plan has run; a higher number means a
         * later request overwrote it and this one is stale. */
        volatile XpBox* b = x.box + r.g;
        const auto t0 = std::chrono::steady_clock::now();
        bool got = true;
        while ((int32_t)(b->seq - r.seq) < 0) {
            _mm_pause();
            if (x.stop || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
                got = false;
                break;
            }
        }
        if (got && b->seq != r.seq) got = false;
        std::atomic_thread_fence(std::memory_order_acquire);
        uint64_t src[kXpSlots], dst[kXpSlots], bytes[kXpSlots];
        int n = 0;
        if (got) {
            n = std::min<int>((int)b->n, kXpSlots);
            for (int i = 0; i < n && got; ++i) {
                /* A job seen before its own stores landed carries an older tag. */
                const auto t1 = std::chrono::steady_clock::now();
                while (b->jobs[i].tag != r.seq &&
                       std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(100))
                    _mm_pause();
                if (b->jobs[i].tag != r.seq) { got = false; break; }
                std::atomic_thread_fence(std::memory_order_acquire);
                src[i] = b->jobs[i].src;
                dst[i] = b->jobs[i].dst;
                bytes[i] = b->jobs[i].bytes;
            }
        }
        if (!got || n == 0 || behind > kXpBehind) {
            if (got && n > 0) ++x.dropped;
            if (r.mark) rad_ride_release(r.mark, r.ms, kXpMarkHold);
            continue;
        }
        if (rad_xp_list(x.device, r.mark, n, src, dst, bytes, x.done + r.g, r.seq, 0) == RAD_OK) {
            ++x.lists;
            if (r.mark) x.marks.emplace_back(r.mark, x.lists);
        } else {
            if (r.mark) rad_ride_release(r.mark, r.ms, kXpMarkHold);
            RAD_WARN("rank %d: the prefetch DMA queue refused a list", rank_);
        }
    }
}

/* Builds the layer's expert -> table-entry map; false until all its tables are bound. An expert
 * whose weights are not each in exactly one table is left out. */
bool Ctx::xp_bind(XpLayer& L, int layer) {
    const RadPrefetchNext& nx = L.next;
    std::vector<void**> hm((size_t)kXpMaxNe * kXpParts, nullptr);
    std::vector<int32_t> hp((size_t)kXpMaxNe, -1);
    for (int64_t e = 0; e < nx.n_expert; ++e) {
        void** ent[kXpParts] = {};
        bool usable = true;
        for (int64_t k = 0; k < nx.n_parts && usable; ++k) {
            const rad_weight w = nx.parts[e * nx.n_parts + k];
            if (!w) continue;
            if ((size_t)w >= wmemb_head_.size() || wmemb_head_[w] < 0) { usable = false; break; }
            const WMemb& m = wmemb_[(size_t)wmemb_head_[w]];
            if (m.next >= 0) { usable = false; break; }
            const WeightTable& t = wtabs_[(size_t)m.slot];
            if (!t.dev) return false;
            ent[k] = (void**)t.dev + m.entry;
        }
        if (usable)
            for (int k = 0; k < kXpParts; ++k) hm[(size_t)(e * kXpParts + k)] = ent[k];
        hp[(size_t)e] = nx.count_at ? nx.count_at[e] : (int32_t)e;
    }
    XpState& x = *xp_;
    void** map = x.maps + (size_t)layer * kXpMaxNe * kXpParts;
    int32_t* place = x.places + (size_t)layer * kXpMaxNe;
    /* Runs once per layer at first issue, so this host wait is off the step path. */
    if (rad_memcpy_async(map, hm.data(), (int64_t)hm.size() * (int64_t)sizeof(void*), stream_) < 0 ||
        rad_memcpy_async(place, hp.data(), (int64_t)hp.size() * 4, stream_) < 0 || rad_stream_sync(stream_) < 0)
        return false;
    L.map = map;
    L.place = place;
    L.bound = true;
    return true;
}

/* Judges the share of copies used from the counters sent to the host one window ago, then sends the
 * next window's. The first window after a rest is not judged (its counters predate the rest), so
 * each rest is followed by one full fresh window. */
void Ctx::xp_judge(XpState& x) {
    if (x.have_snap) {
        const volatile XpStats* h = x.stats_h;
        const uint32_t copied = h->copied, routed = h->routed;
        const uint32_t dc = copied - x.snap_copied, dr = routed - x.snap_routed;
        x.snap_copied = copied;
        x.snap_routed = routed;
        if (x.judge && dc >= kXpMinJudged && (double)dr < kXpMinUse * (double)dc) {
            x.rest_until = x.passes + kXpRest;
            x.judge = false;
            if (x.rests++ == 0)
                RAD_INFO("rank %d: expert prefetch resting: the steps used %.0f%% of its last %u copies, under the "
                         "%.0f%% that pays for them; it measures again every %lld passes", rank_,
                         100.0 * dr / dc, dc, 100.0 * kXpMinUse, (long long)kXpRest);
        } else {
            x.judge = true;
        }
    }
    if (rad_memcpy_async(x.stats_h, x.stats, sizeof(XpStats), x.stream) >= 0) x.have_snap = true;
}

/* When recording, each of the three calls becomes its own tape step and is re-run at every play:
 * its stream work and rides are issued afresh, not recorded. */
int Ctx::xp_predict(const RadPrefetchNext* nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    /* Too wide to predict; keep the tape free of an empty step. */
    if (!xp_ || !nx || rows <= 0 || rows > kXpMaxRows) return RAD_OK;
    if (!rec_) return xp_predict_run(*nx, h, r0, rows, n_embd);
    TapeXp p;
    p.next = *nx;
    p.h = h;
    p.r0 = r0;
    p.rows = rows;
    p.n_embd = n_embd;
    TapeStep st;
    st.kind = kTapeXp;
    st.a = (int32_t)rec_->xp.size();
    rec_->xp.push_back(p);
    return tape_host(st, [&] { return xp_predict_run(*nx, h, r0, rows, n_embd); });
}

int Ctx::xp_mark() {
    /* lever XPF: a deferred plan's copies start behind the last routed down issued before its RAD_XPF_GO */
    if (xpf_ && !rec_ && xpf_->pend.valid) {
        XpfState& x = *xpf_;
        void* mark = nullptr;
        if (rad_ride(cur_stream(), &mark) == RAD_OK) {
            if (x.pend.mark) rad_ride_release(x.pend.mark, x.pend.ms, kXpMarkHold);
            x.pend.mark = mark;
            x.pend.ms = cur_stream();
        }
    }
    if (!xp_) return RAD_OK;
    if (!rec_) return xp_mark_run();
    TapeStep st;
    st.kind = kTapeXpMark;
    return tape_host(st, [&] { return xp_mark_run(); });
}

int Ctx::xp_step(int layer) {
    /* lever XPF: a layer an xpf plan was made for this pass (live passes only: a pass that plans is never recorded) */
    if (xpf_ && !rec_ && layer >= 0 && (xpf_->wl[0] == layer || xpf_->wl[1] == layer)) {
        const int rc = xpf_step_run(layer);
        if (rc < 0) return rc;
    }
    if (!xp_) return RAD_OK;
    if (!rec_) return xp_step_run(layer);
    TapeStep st;
    st.kind = kTapeXpStep;
    st.a = layer;
    return tape_host(st, [&] { return xp_step_run(layer); });
}

int Ctx::xp_predict_run(const RadPrefetchNext& nx, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd) {
    XpState& x = *xp_;
    const int layer = nx.layer;
    if (rows <= 0 || rows > kXpMaxRows || nx.top_k <= 0 || nx.top_k > kXpMaxK || nx.n_expert <= 0 ||
        nx.n_expert > kXpMaxNe || nx.n_parts <= 0 || nx.n_parts > kXpParts || !nx.parts || n_embd % 256 ||
        n_embd > 4096 || layer < 0 || layer >= (int)x.layers.size()) {
        /* Said once a rank (lever VA "xpoff"): a router input this predictor never takes means the
         * prefetch copies nothing for the model, and its ring is VRAM the slab could hold. */
        if ((n_embd % 256 || n_embd > 4096) && !x.said_width) {
            x.said_width = true;
            RAD_INFO("rank %d: expert prefetch: the router input is %lld columns, which the predictor does not "
                     "take (a multiple of 256, at most 4096): it copies nothing for this model "
                     "(RADIANCE_XPREFETCH=0 returns its ring to the slab)", rank_, (long long)n_embd);
        }
        return RAD_OK;
    }
    XpLayer& L = x.layers[(size_t)layer];
    if (!L.bound) {
        L.next = nx;
        if (!xp_bind(L, layer)) return RAD_OK;
    }
    const void* w = weight_ptr(nx.router);
    const char* hb = (const char*)buf_ptr(h);
    if (!w || !hb) return RAD_OK;
    const RadStream cs = cur_stream();
    /* A new pass waits for the prefetch stream's work from the last pass, so that stream is never a
     * pass behind: the records and mailboxes a request reuses are three layers old. */
    if (layer <= x.last_layer) {
        if (x.tail_live) {
            RAD_TRY(rad_event_record(x.tail, x.stream));
            RAD_TRY(rad_event_wait(cs, x.tail));
            x.tail_live = false;
        }
        ++x.passes;
    }
    x.last_layer = layer;
    if (x.passes < x.rest_until) return RAD_OK;
    /* The router input is read in place, riding behind the next compute dispatch (the router, which
     * reads it) rather than via a barrier packet between two kernels. */
    void* rt = nullptr;
    if (rad_ride(cs, &rt) == RAD_OK) {
        RAD_TRY(rad_ride_wait(x.stream, rt));
    } else {
        RAD_TRY(rad_event_record(x.sync, cs));
        RAD_TRY(rad_event_wait(x.stream, x.sync));
    }
    /* An earlier request still waiting for its mark is sent now, unmarked. */
    if (x.pend.valid) {
        {
            std::lock_guard<std::mutex> lk(x.mu);
            x.q.push_back(x.pend);
        }
        x.cv.notify_one();
        x.pend.valid = false;
    }
    const int g = layer % kXpGroups;
    const uint32_t seq = ++x.seq[g];
    const uint16_t* hg = (const uint16_t*)(hb + r0 * n_embd * 2);
    float* lg = x.logits + (size_t)g * kXpMaxRows * kXpMaxNe;
    const int32_t* cnt = route_n_expert_ > 0 && nx.n_expert <= route_n_expert_
                             ? route_dev_ + (int64_t)layer * route_n_expert_ : nullptr;
    const uint64_t ring = (uint64_t)(uintptr_t)(x.ring + (size_t)g * kXpSlots * x.slot);
    if (rad_xp_logits_launch(hg, (const uint16_t*)w, lg, (uint32_t)rows, (uint32_t)n_embd, (uint32_t)nx.n_expert,
                             (void*)x.stream) ||
        rad_xp_plan_launch(lg, (uint32_t)rows, (uint32_t)nx.n_expert, (uint32_t)nx.top_k, (void** const*)L.map,
                           L.place, cnt, x.pools, ring, x.slot, x.rec + (size_t)g * kXpSlots, x.nrec + g,
                           (XpBox*)rad_dev_device_ptr(x.box + g), seq, x.stats, (void*)x.stream))
        return RAD_E_DEVICE;
    x.tail_live = true;
    if (++x.calls % kXpWindow == 0) xp_judge(x);
    x.pend.g = g;
    x.pend.seq = seq;
    x.pend.mark = nullptr;
    x.pend.ms = nullptr;
    x.pend.valid = true;
    L.want = true;
    L.g = g;
    L.seq = seq;
    return RAD_OK;
}

int Ctx::xp_mark_run() {
    XpState& x = *xp_;
    if (!x.pend.valid) return RAD_OK;
    /* Refused only while another ride waits on the stream; the list then starts unmarked. */
    void* mark = nullptr;
    if (rad_ride(cur_stream(), &mark) == RAD_OK) {
        x.pend.mark = mark;
        x.pend.ms = cur_stream();
    }
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.q.push_back(x.pend);
    }
    x.cv.notify_one();
    x.pend.valid = false;
    return RAD_OK;
}

int Ctx::xp_step_run(int layer) {
    XpState& x = *xp_;
    XpLayer* L = layer >= 0 && (size_t)layer < x.layers.size() ? &x.layers[(size_t)layer] : nullptr;
    const bool patch = L && L->want;
    if (!patch && x.restore_g < 0) return RAD_OK;
    const XpRec* rrec = x.restore_g >= 0 ? x.rec + (size_t)x.restore_g * kXpSlots : nullptr;
    const int32_t* rn = x.restore_g >= 0 ? x.nrec + x.restore_g : nullptr;
    const XpRec* prec = patch ? x.rec + (size_t)L->g * kXpSlots : nullptr;
    const int32_t* pn = patch ? x.nrec + L->g : nullptr;
    const uint32_t* done = patch ? (const uint32_t*)rad_dev_device_ptr(x.done + L->g) : nullptr;
    if (rad_xp_step_launch(rrec, rn, prec, pn, done, patch ? L->seq : 0, (void*)cur_stream())) return RAD_E_DEVICE;
    x.restore_g = patch ? L->g : -1;
    if (L) L->want = false;
    return RAD_OK;
}

/* ================================================================== lever XPF (xprefetch.h) */

void Ctx::xpf_free(XpfState& x) {
    if (x.ring) rad_dev_free_uncached(x.ring);
    for (void* p : {(void*)x.rec, (void*)x.nrec, (void*)x.cnt, (void*)x.stats, (void*)x.maps, (void*)x.places})
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    for (void* p : {(void*)x.box, (void*)x.done, (void*)x.started, (void*)x.stats_h})
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

void Ctx::xpf_open(const uint64_t* lo, const uint64_t* hi, const uint64_t* stride, int n, int64_t ring) {
    if (xpf_ || device_is_host() || route_.empty() || ring <= 0) return;
    if (!rad_xp_supported(device_)) {
        RAD_INFO("rank %d: lever XPF off: the device backend has no DMA engine to spare for it", rank_);
        return;
    }
    auto x = std::make_unique<XpfState>();
    x->device = device_;
    for (int i = 0; i < n && x->pools.n < kXpMaxPools; ++i) {
        if (stride[i] == 0 || stride[i] > kXpMaxUnit || hi[i] <= lo[i]) continue;
        x->pools.lo[x->pools.n] = lo[i];
        x->pools.hi[x->pools.n] = hi[i];
        x->pools.stride[x->pools.n] = stride[i];
        x->slot = std::max(x->slot, stride[i]);
        ++x->pools.n;
    }
    if (x->pools.n == 0 || x->slot == 0) return;
    x->slots_max = (int)std::min<int64_t>(kXpfMaxSlots, ring / 2 / (int64_t)x->slot);
    if (x->slots_max < 1) {
        RAD_WARN("rank %d: lever XPF off: %lld ring bytes hold no two copies of %lld B", rank_, (long long)ring,
                 (long long)x->slot);
        return;
    }
    x->ring_bytes = ring / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT * 2;
    x->group_bytes = x->ring_bytes / 2;
    x->slots_max = kXpfMaxSlots;   /* the bytes bound the copies (packed by slot size); this only bounds the records */
    const int64_t nl = (int64_t)route_.size();
    if (rad_dev_set(device_) < 0) return;
    x->ring    = (char*)rad_dev_alloc_uncached(x->ring_bytes);
    x->rec     = (XpfRec*)rad_dev_alloc((int64_t)sizeof(XpfRec) * 2 * kXpfMaxSlots, RAD_MEM_DEVICE);
    x->nrec    = (int32_t*)rad_dev_alloc(4 * 2, RAD_MEM_DEVICE);
    x->cnt     = (int32_t*)rad_dev_alloc(4 * kXpMaxNe, RAD_MEM_DEVICE);
    x->stats   = (XpfStats*)rad_dev_alloc((int64_t)sizeof(XpfStats), RAD_MEM_DEVICE);
    x->maps    = (void**)rad_dev_alloc(nl * kXpMaxNe * kXpfParts * (int64_t)sizeof(void*), RAD_MEM_DEVICE);
    x->places  = (int32_t*)rad_dev_alloc(nl * kXpMaxNe * 4, RAD_MEM_DEVICE);
    x->box     = (XpfBox*)rad_dev_alloc((int64_t)sizeof(XpfBox) * 2, RAD_MEM_HOST_PINNED);
    x->done    = (uint32_t*)rad_dev_alloc(4 * 2 * kXpfChunks, RAD_MEM_HOST_PINNED);
    x->started = (uint32_t*)rad_dev_alloc(4 * 2, RAD_MEM_HOST_PINNED);
    x->stats_h = (XpfStats*)rad_dev_alloc((int64_t)sizeof(XpfStats), RAD_MEM_HOST_PINNED);
    const bool ok = x->ring && x->rec && x->nrec && x->cnt && x->stats && x->maps && x->places && x->box && x->done &&
                    x->started && x->stats_h && rad_memset_async(x->nrec, 0, 4 * 2, stream_) >= 0 &&
                    rad_memset_async(x->cnt, 0, 4 * kXpMaxNe, stream_) >= 0 &&
                    rad_memset_async(x->stats, 0, sizeof(XpfStats), stream_) >= 0 && rad_stream_sync(stream_) >= 0;
    if (!ok) {
        const char* why = rad_dev_last_error();
        RAD_WARN("rank %d: lever XPF off: %s", rank_, why && *why ? why : "no memory for it");
        xpf_free(*x);
        return;
    }
    std::memset((void*)x->box, 0, sizeof(XpfBox) * 2);
    std::memset(x->done, 0, 4 * 2 * kXpfChunks);
    std::memset(x->started, 0, 4 * 2);
    std::memset((void*)x->stats_h, 0, sizeof(XpfStats));
    x->layers.resize((size_t)nl);
    XpfState* xs = x.get();
    x->thr = std::thread([this, xs] { xpf_serve(*xs); });
    RAD_INFO("rank %d: lever XPF on: the prefill prefetch's ring %s (2 groups of %s: copies packed by their pool slot, "
             "the largest %lld B), DMA lists of %d copies", rank_, humanb(x->ring_bytes).c_str(),
             humanb(x->group_bytes).c_str(), (long long)x->slot, kXpfChunk);
    xpf_ = std::move(x);
}

void Ctx::xpf_close() {
    if (!xpf_) return;
    XpfState& x = *xpf_;
    if (stream_) rad_stream_sync(stream_);
    if (lane1_) rad_stream_sync(lane1_);
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.stop = true;
    }
    x.cv.notify_all();
    if (x.thr.joinable()) x.thr.join();
    const auto t0 = std::chrono::steady_clock::now();
    while (rad_xp_behind(x.device) > 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (stream_ && rad_memcpy_async(x.stats_h, x.stats, sizeof(XpfStats), stream_) >= 0 && rad_stream_sync(stream_) >= 0)
        RAD_INFO("rank %d: lever XPF: %u plans, %u copies planned of %u candidates, %u landed in time and patched, %u of "
                 "those routed; %lld DMA lists, %lld requests dropped (queue behind), %lld late mailboxes", rank_,
                 x.stats_h->plans, x.stats_h->planned, x.stats_h->cands, x.stats_h->patched, x.stats_h->used,
                 (long long)x.lists.load(), (long long)x.dropped.load(), (long long)x.late.load());
    xpf_free(x);
    xpf_.reset();
}

/* Host thread: each plan's mailbox, once written, to the prefetch DMA queue as lists of kXpfChunk copies, each list
 * writing the request number to its chunk's done word. */
void Ctx::xpf_serve(XpfState& x) {
    for (;;) {
        XpfReq r;
        {
            std::unique_lock<std::mutex> lk(x.mu);
            x.cv.wait(lk, [&] { return x.stop.load() || !x.q.empty(); });
            if (x.stop) return;
            r = x.q.front();
            x.q.pop_front();
        }
        /* return the marks of finished lists to the pool */
        {
            const int64_t behind = rad_xp_behind(x.device);
            while (!x.marks.empty() && x.marks.front().second <= x.lists.load() - behind) {
                rad_ride_release(x.marks.front().first, nullptr, 0);
                x.marks.pop_front();
            }
        }
        void* mark = r.mark;   /* the first list polls it; returned once a list holds it, or here */
        const auto drop_mark = [&] {
            if (mark) rad_ride_release(mark, r.ms, kXpMarkHold);
            mark = nullptr;
        };
        volatile XpfBox* b = x.box + r.g;
        const auto t0 = std::chrono::steady_clock::now();
        bool got = true;
        while ((int32_t)(b->seq - r.seq) < 0) {
            _mm_pause();
            if (x.stop || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) { got = false; break; }
        }
        if (!got || b->seq != r.seq) { ++x.late; drop_mark(); continue; }
        std::atomic_thread_fence(std::memory_order_acquire);
        const int n = std::min<int>((int)b->n, kXpfMaxSlots);
        volatile uint32_t* st = x.started + r.g;
        for (int c0 = 0; c0 < n; c0 += kXpfChunk) {
            /* at most two of this plan's lists in the queue, and none once its layer's routed GEMMs began: a copy that
             * lands after that is never read and its bytes share the link with the GEMMs' own reads */
            const auto t2 = std::chrono::steady_clock::now();
            bool late = false;
            while (c0 >= 2 * kXpfChunk && rad_xp_behind(x.device) >= 2) {
                if (*st == r.seq) { late = true; break; }
                if (x.stop || std::chrono::steady_clock::now() - t2 > std::chrono::milliseconds(200)) { late = true; break; }
                _mm_pause();
            }
            if (late || *st == r.seq) { ++x.late; break; }
            /* a queue still busy with older lists: the rest of this plan would land too late */
            if (rad_xp_behind(x.device) > 2 * kXpfChunks) { ++x.dropped; break; }
            uint64_t src[kXpfChunk], dst[kXpfChunk], bytes[kXpfChunk];
            int m = 0;
            bool ok = true;
            for (int j = c0; j < std::min(n, c0 + kXpfChunk) && ok; ++j) {
                const auto t1 = std::chrono::steady_clock::now();
                while (b->jobs[j].tag != r.seq && std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(100))
                    _mm_pause();
                if (b->jobs[j].tag != r.seq) { ok = false; break; }
                std::atomic_thread_fence(std::memory_order_acquire);
                src[m] = b->jobs[j].src;
                dst[m] = b->jobs[j].dst;
                bytes[m] = b->jobs[j].bytes;
                ++m;
            }
            if (!ok || m == 0) { ++x.late; break; }
            if (rad_xp_list(x.device, mark, m, src, dst, bytes, x.done + r.g * kXpfChunks + c0 / kXpfChunk, r.seq, 0) !=
                RAD_OK) {
                RAD_WARN("rank %d: lever XPF: the prefetch DMA queue refused a list", rank_);
                break;
            }
            ++x.lists;
            if (mark) {
                x.marks.emplace_back(mark, x.lists.load());
                mark = nullptr;   /* later lists queue behind the first on the same engine */
            }
        }
        drop_mark();
    }
}

/* The layer's expert -> table-entry map, every entry naming one of the unit's weights (a weight in two tables: two
 * entries, at most kXpfParts); an expert with more is left out. */
bool Ctx::xpf_bind(XpfLayer& L, const RadPrefetchNext& nx, int layer) {
    L.tried = true;
    if (nx.n_expert <= 0 || nx.n_expert > kXpMaxNe || nx.n_parts <= 0 || !nx.parts) return false;
    std::vector<void**> hm((size_t)kXpMaxNe * kXpfParts, nullptr);
    std::vector<int32_t> hp((size_t)kXpMaxNe, -1);
    for (int64_t e = 0; e < nx.n_expert; ++e) {
        void* ent[kXpfParts] = {};
        int k = 0;
        bool usable = true;
        for (int64_t p = 0; p < nx.n_parts && usable; ++p) {
            const rad_weight w = nx.parts[e * nx.n_parts + p];
            if (!w) continue;
            if ((size_t)w >= wmemb_head_.size() || wmemb_head_[w] < 0) { usable = false; break; }
            for (int32_t mi = wmemb_head_[w]; mi >= 0 && usable; mi = wmemb_[(size_t)mi].next) {
                const WMemb& m = wmemb_[(size_t)mi];
                const WeightTable& t = wtabs_[(size_t)m.slot];
                if (!t.dev) return false;
                if (k >= kXpfParts) { usable = false; break; }
                ent[k++] = (void**)t.dev + m.entry;
            }
        }
        if (usable)
            for (int q = 0; q < kXpfParts; ++q) hm[(size_t)(e * kXpfParts + q)] = (void**)ent[q];
        hp[(size_t)e] = nx.count_at ? nx.count_at[e] : (int32_t)e;
    }
    XpfState& x = *xpf_;
    void** map = x.maps + (size_t)layer * kXpMaxNe * kXpfParts;
    int32_t* place = x.places + (size_t)layer * kXpMaxNe;
    if (rad_memcpy_async(map, hm.data(), (int64_t)hm.size() * (int64_t)sizeof(void*), stream_) < 0 ||
        rad_memcpy_async(place, hp.data(), (int64_t)hp.size() * 4, stream_) < 0 || rad_stream_sync(stream_) < 0)
        return false;
    L.map = map;
    L.place = place;
    L.bound = true;
    return true;
}

int Ctx::xpf_plan(const RadPrefetchNext* nx, const void* logits, int64_t rows, int64_t ld, const void* bias, int kp,
                  int slots, unsigned flags) {
    if (!xpf_) return RAD_E_UNSUPPORTED;
    if (rec_) return RAD_OK;   /* never in a recorded pass (the architecture keys planning passes 0) */
    XpfState& x = *xpf_;
    const RadStream cs = cur_stream();
    if (!nx && (flags & RAD_XPF_GO)) {   /* the deferred plan's copies may go once the last down marked has run */
        if (x.pend.valid) {
            {
                std::lock_guard<std::mutex> lk(x.mu);
                x.q.push_back(x.pend);
            }
            x.cv.notify_one();
            x.pend = XpfReq{};
        }
        return RAD_OK;
    }
    if (!nx) {   /* the pass's end: both groups back to the pool */
        if (x.pend.valid) {   /* a deferred plan never let go: its copies are not needed any more */
            if (x.pend.mark) rad_ride_release(x.pend.mark, x.pend.ms, kXpMarkHold);
            x.pend = XpfReq{};
        }
        for (int g = 0; g < 2; ++g)
            if (x.dirty[g]) {
                if (rad_xpf_restore_launch(x.rec + (size_t)g * kXpfMaxSlots, x.nrec + g, x.stats, (void*)cs))
                    return RAD_E_DEVICE;
                x.dirty[g] = false;
            }
        x.wl[0] = x.wl[1] = -1;
        return RAD_OK;
    }
    const int layer = nx->layer;
    if (layer < 0 || (size_t)layer >= x.layers.size() || !logits || rows <= 0 || ld < nx->n_expert || kp <= 0 ||
        kp > kXpfMaxK || nx->n_expert > kXpMaxNe)
        return RAD_E_INVAL;
    XpfLayer& L = x.layers[(size_t)layer];
    if (!L.bound) {
        if (L.tried) return RAD_OK;
        if (!xpf_bind(L, *nx, layer)) {
            RAD_WARN("rank %d: lever XPF: layer %d's experts could not be mapped to their table entries; no prefetch for it",
                     rank_, layer);
            return RAD_OK;
        }
    }
    const int g = x.g ^ 1;
    const uint32_t seq = ++x.seq[g];
    const int sl = std::max(0, std::min(slots, x.slots_max));
    const int32_t* rcnt = route_n_expert_ > 0 && nx->n_expert <= route_n_expert_
                              ? route_dev_ + (int64_t)layer * route_n_expert_ : nullptr;
    const uint64_t ring = (uint64_t)(uintptr_t)(x.ring + (size_t)g * (size_t)x.group_bytes);
    if (rad_xpf_count_launch((const uint16_t*)logits, (uint32_t)rows, (uint32_t)ld, (uint32_t)nx->n_expert, (uint32_t)kp,
                             (const float*)bias, x.cnt, (void*)cs) ||
        rad_xpf_plan_launch(x.cnt, (uint32_t)nx->n_expert, (void** const*)L.map, L.place, rcnt, x.pools, ring,
                            (uint64_t)x.group_bytes, (uint32_t)sl, x.rec + (size_t)g * kXpfMaxSlots, x.nrec + g,
                            (XpfBox*)rad_dev_device_ptr(x.box + g), seq, x.stats, (void*)cs))
        return RAD_E_DEVICE;
    XpfReq rq;
    rq.g = g;
    rq.seq = seq;
    rq.valid = true;
    if (flags & RAD_XPF_DEFER) {
        if (x.pend.valid) {   /* an older deferred plan nobody let go: dropped */
            if (x.pend.mark) rad_ride_release(x.pend.mark, x.pend.ms, kXpMarkHold);
            ++x.dropped;
        }
        x.pend = rq;
    } else {
        {
            std::lock_guard<std::mutex> lk(x.mu);
            x.q.push_back(rq);
        }
        x.cv.notify_one();
    }
    x.g = g;
    x.dirty[g] = true;
    x.wl[g] = layer;
    x.wseq[g] = seq;
    /* every 2048 plans: the counters the last snapshot brought (one window late), then the next snapshot, behind the plan */
    if (++x.plans % 2048 == 0) {
        XpfStats h;
        std::memcpy(&h, (const void*)x.stats_h, sizeof h);
        if (x.plans > 2048)
            RAD_INFO("rank %d: lever XPF: %u plans, %u copies planned of %u candidates, %u landed in time (%.1f%%), %u of "
                     "those routed (%.1f%%); %lld DMA lists, %lld plans dropped behind the queue, %lld late mailboxes", rank_,
                     h.plans, h.planned, h.cands, h.patched, h.planned ? 100.0 * h.patched / h.planned : 0.0, h.used,
                     h.patched ? 100.0 * h.used / h.patched : 0.0, (long long)x.lists.load(), (long long)x.dropped.load(),
                     (long long)x.late.load());
        (void)rad_memcpy_async(x.stats_h, x.stats, sizeof(XpfStats), cs);
    }
    return RAD_OK;
}

int Ctx::xpf_step_run(int layer) {
    XpfState& x = *xpf_;
    const int g = x.wl[0] == layer ? 0 : 1;
    if (rad_xpf_step_launch(x.rec + (size_t)g * kXpfMaxSlots, x.nrec + g,
                            (const uint32_t*)rad_dev_device_ptr(x.done + g * kXpfChunks), x.wseq[g], x.stats,
                            (uint32_t*)rad_dev_device_ptr(x.started + g), (void*)cur_stream()))
        return RAD_E_DEVICE;
    x.wl[g] = -1;
    return RAD_OK;
}

}  /* namespace rad */
