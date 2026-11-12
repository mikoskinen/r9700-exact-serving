/* r4d_selftest_arll.inc -- ds41-ll's lever A3 / 1U-AR on FOUR cards (included by r4d_selftest.cpp before device_phase;
 * called from it beside the four-rank all-reduce rows):
 *
 *   --case ar_ll: LL areas forced on for the instances this case builds (r4d_arll_test_settings, as RADIANCE_AR_LL would
 *     at load), then
 *     SWITCH  ar_ll refuses -1 and 49, takes 0, 1 and 48 (read back).
 *     AR      the row ar_oneshot_4rank_exact through its own entry (deepseek41's decode all-reduce: the one-shot, the
 *             two-shot from ts_min), [T, 5120] bf16 at T 1 2 6 7 8 9 16 24 47 48 49 64, three inputs (activation-like rows
 *             with exact zeros, ar_rough_value's 25 binades, ar_coded_value's edges with an infinity), T 16 out of place:
 *             ar_ll 0 and ar_ll 1 == the fp32 ascending-rank fold rounded once on every rank, bit for bit, so ON == OFF;
 *             the counters (rad_arll_stats) prove the wire: OFF no LL launch; ON the one-shot at T <= 6, the two-shot at
 *             7 .. 48 on every rank, T 49 / 64 plain, none declined.
 *     GATHER  moe_gather_all_reduce_4rank's D3 case (moe_gather_ar_case: M 1 .. 48, 56, 64 of 5120 with top_k 6 of 384,
 *             generic n 2048 arms, M 300; one-shot and two-shot launches each) under ar_ll 1 against its slot-order
 *             oracle; the counters show the gather's LL launches.
 *     STRESS  1000 launches back to back in batches of eight (out of place, every launch its own input), heights drawn
 *             from 1 .. 64 (LL to 48, plain past it, on the same instance: mixed parities, forms and wires), each
 *             launch's output == its fold on every rank: 0 bad; then 1000 LL launches at T 7 / 8 / 9 only (the c8 band).
 *     TIME    (INFO) T 1 / 8 / 16 / 48 x 5120, the plain all-reduce and the gather (deepseek41's MoE shape), ar_ll 0 vs 1:
 *             isolated -- the four rank threads released together, the slowest rank's launch-to-sync time, median of
 *             60 -- and back to back (40 launches, the slowest rank's mean). "us a call" lines for the screen's P0. */
#include <chrono>
#include "r4d_ar_ll.h"

namespace arll {

constexpr int64_t kW = 5120;   /* deepseek41's residual width */

static RadTensor arll_tensor(void* d, int64_t rows) { return T(d, RAD_BF16, { rows, kW }); }

static float act_value(int rank, int64_t i, uint64_t salt = 0) {
    uint64_t h = (uint64_t)i * 0x9E3779B97F4A7C15ull ^ ((uint64_t)rank + 7) * 0xD1B54A32D192ED03ull ^ salt * 0x2545F4914F6CDD1Dull;
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
    uint64_t r = (uint64_t)(i / kW) * 0x94D049BB133111EBull ^ salt;
    r ^= r >> 31;
    if ((h >> 60) == 0) return 0.0f;
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += (double)((h >> (13 * k)) & 0x1fff) / 4096.0 - 1.0;
    return (float)std::ldexp(s * (1.0 + 0.25 * rank), (int)(r % 8) - 4);
}
/* 2 act, 0 ar_rough_value, 1 ar_coded_value */
static float value(int pat, int rank, int64_t i) { return pat == 2 ? act_value(rank, i) : ar_value(pat, rank, i); }
static const char* pat_name(int pat) { return pat == 2 ? "act" : pat == 1 ? "coded-form edges" : "rough (25 binades)"; }

struct Hooks {
    int (*sw)(const char*, int) = nullptr;
    int (*val)(const char*, int*) = nullptr;
    int (*stats)(long long*, int) = nullptr;
    void (*settings)(int, int, int) = nullptr;
};
struct Ctr { long long v[7] = {}; };
static Ctr snap(const Hooks& k) { Ctr c; if (k.stats) k.stats(c.v, 7); return c; }
static Ctr diff(const Ctr& a, const Ctr& b) { Ctr d; for (int i = 0; i < 7; ++i) d.v[i] = b.v[i] - a.v[i]; return d; }
static std::string ctrs(const Ctr& d) {
    return fmt("LL one-shot %lld, two-shot %lld (rows %lld, gather %lld), declined %lld, plain %lld, LL all ranks %lld",
               d.v[0], d.v[1], d.v[2], d.v[3], d.v[4], d.v[5], d.v[6]);
}

/* Four rank threads released together, again and again (a sense-reversing spin barrier). */
struct Barrier {
    std::atomic<int> n{0}, gen{0};
    void wait(int world) {
        const int g = gen.load(std::memory_order_acquire);
        if (n.fetch_add(1, std::memory_order_acq_rel) + 1 == world) {
            n.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_acq_rel);
        } else {
            while (gen.load(std::memory_order_acquire) == g) std::this_thread::yield();
        }
    }
};

/* ---- the plain all-reduce: identity runs, stress, timing */
struct Job {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0, pat = 2;
    std::vector<int64_t> rows;           /* identity: the launches, T rows of kW each */
    int oop = -1;                        /* index into rows launched out of place */
    int stress = 0;                      /* > 0: that many stress launches (rows drawn from stress_lo .. stress_hi) */
    int stress_lo = 1, stress_hi = 64;
    std::vector<int64_t> trows;          /* timing sizes */
    int iso = 60, bb = 40;
    Barrier* bar = nullptr;
    std::vector<double>* iso_us = nullptr;   /* [timing size][iteration] this rank's launch-to-sync, us (flattened) */
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<int64_t> bad, first;
    std::vector<uint64_t> hash;
    int64_t stress_bad = 0, stress_first = -1, stress_launches = 0;
    std::vector<double> bb_us;
};

static void worker(Job* t) {
    int64_t most = 64 * kW;
    for (int64_t T : t->rows) most = std::max(most, T * kW);
    std::vector<RadParam> ps = { PI("world_size", 4), PI("numel", most), PS("dtype", "bf16"), PI("exact", 1) };
    constexpr int kBatch = 8;
    void* in[kBatch] = {};
    void* out[kBatch] = {};
    if (st_set_device(t->dev) != hipSuccess) t->rc_init = RAD_E_DEVICE;
    else t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, 4, &t->inst);
    for (int b = 0; t->rc_init == RAD_OK && b < kBatch; ++b)
        if (hipMalloc(&in[b], (size_t)most * 2) != hipSuccess || hipMalloc(&out[b], (size_t)most * 2) != hipSuccess)
            t->rc_launch = RAD_E_NOMEM;
    auto launch = [&](void* x, void* y, int64_t T) {
        RadTensor ts[2] = { arll_tensor(x, T), arll_tensor(y, T) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = y ? 2 : 1;
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst; a.rank = t->rank; a.world_size = 4;
        const int rc = t->row->launch(&a, nullptr);
        return rc == RAD_LINK_ENDED ? RAD_OK : rc;
    };
    std::vector<uint16_t> h((size_t)most);
    auto check = [&](const uint16_t* o, int64_t n, int pat, uint64_t salt, int64_t* bad, int64_t* first, uint64_t* fnv) {
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < 4; ++r)
                acc += h_bf16(h_to_bf16(pat < 0 ? act_value(r, i, salt) : value(pat, r, i)));
            if (o[(size_t)i] != h_to_bf16(acc)) { if (*first < 0) *first = i; ++*bad; }
            if (fnv) *fnv = (*fnv ^ o[(size_t)i]) * 1099511628211ull;
        }
    };
    for (size_t k = 0; t->rc_init == RAD_OK && t->rc_launch == RAD_OK && k < t->rows.size(); ++k) {
        const int64_t T = t->rows[k], n = T * kW;
        const bool oop = (int)k == t->oop;
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(value(t->pat, t->rank, i));
        st_memcpy(in[0], h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        if (oop) st_memset(out[0], 0, (size_t)n * 2);
        const int rc = launch(in[0], oop ? out[0] : nullptr, T);
        if (rc != RAD_OK) { t->rc_launch = rc; break; }
        st_stream_sync(nullptr);
        st_memcpy(h.data(), oop ? out[0] : in[0], (size_t)n * 2, hipMemcpyDeviceToHost);
        int64_t bad = 0, first = -1;
        uint64_t fnv = 1469598103934665603ull;
        check(h.data(), n, t->pat, 0, &bad, &first, &fnv);
        t->bad.push_back(bad); t->first.push_back(first); t->hash.push_back(fnv);
    }
    /* stress: batches of kBatch launches back to back, each its own input (salted), then all checked */
    if (t->stress > 0 && t->rc_init == RAD_OK && t->rc_launch == RAD_OK) {
        uint64_t rs = 0x5DEECE66Dull;   /* the same sequence on every rank: the ranks must launch the same heights */
        std::vector<std::vector<uint16_t>> hb(kBatch, std::vector<uint16_t>((size_t)most));
        for (int base = 0; base < t->stress && t->rc_launch == RAD_OK; base += kBatch) {
            int64_t Ts[kBatch];
            const int nbat = std::min(kBatch, t->stress - base);
            for (int b = 0; b < nbat; ++b) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                Ts[b] = t->stress_lo + (int64_t)(rs % (uint64_t)(t->stress_hi - t->stress_lo + 1));
                const uint64_t salt = (uint64_t)(base + b) + 1;
                for (int64_t i = 0; i < Ts[b] * kW; ++i) hb[(size_t)b][(size_t)i] = h_to_bf16(act_value(t->rank, i, salt));
                st_memcpy(in[b], hb[(size_t)b].data(), (size_t)(Ts[b] * kW) * 2, hipMemcpyHostToDevice);
            }
            st_stream_sync(nullptr);
            for (int b = 0; b < nbat && t->rc_launch == RAD_OK; ++b) t->rc_launch = launch(in[b], out[b], Ts[b]);
            st_stream_sync(nullptr);
            for (int b = 0; b < nbat && t->rc_launch == RAD_OK; ++b) {
                st_memcpy(hb[(size_t)b].data(), out[b], (size_t)(Ts[b] * kW) * 2, hipMemcpyDeviceToHost);
                int64_t bad = 0, first = -1;
                check(hb[(size_t)b].data(), Ts[b] * kW, -1, (uint64_t)(base + b) + 1, &bad, &first, nullptr);
                if (bad && t->stress_first < 0) t->stress_first = (int64_t)(base + b);
                t->stress_bad += bad;
                ++t->stress_launches;
            }
        }
    }
    /* timing: isolated (released together) and back to back, out of place. Every rank thread passes every barrier, a failed
     * one too (it launches nothing more), so the others are never left waiting at one. */
    const bool up = t->rc_init == RAD_OK && t->rc_launch == RAD_OK;
    for (size_t k = 0; k < t->trows.size(); ++k) {
        const int64_t T = t->trows[k], n = T * kW;
        if (up && t->rc_launch == RAD_OK) {
            for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(act_value(t->rank, i));
            st_memcpy(in[0], h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
            for (int w = 0; w < 3 && t->rc_launch == RAD_OK; ++w) t->rc_launch = launch(in[0], out[0], T);
            st_stream_sync(nullptr);
        }
        for (int it = 0; it < t->iso; ++it) {
            t->bar->wait(4);
            if (!up || t->rc_launch != RAD_OK) continue;
            const auto t0 = std::chrono::steady_clock::now();
            t->rc_launch = launch(in[0], out[0], T);
            st_stream_sync(nullptr);
            const auto t1 = std::chrono::steady_clock::now();
            t->iso_us->push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        t->bar->wait(4);
        if (!up || t->rc_launch != RAD_OK) continue;
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < t->bb && t->rc_launch == RAD_OK; ++it) t->rc_launch = launch(in[0], out[0], T);
        st_stream_sync(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        t->bb_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / t->bb);
    }
    for (int b = 0; b < kBatch; ++b) {
        if (in[b]) st_free(in[b]);
        if (out[b]) st_free(out[b]);
    }
}

/* One run of four rank threads under ar_ll `v`; jobs filled by `fill`. */
template <class F>
static bool run(const RadKernelInfo* row, const std::vector<int>& devs, const Hooks& hk, int v, std::vector<Job>* jobs,
                Ctr* d, std::string* why, const F& fill) {
    const int sr = hk.sw("ar_ll", v);
    int applied = -2;
    hk.val("ar_ll", &applied);
    if (sr != 1 || applied != v) { *why = fmt("rad_fn4_switch(ar_ll, %d) = %d, applied %d", v, sr, applied); return false; }
    const Ctr c0 = snap(hk);
    jobs->assign(4, Job());
    for (int r = 0; r < 4; ++r) {
        Job& j = (*jobs)[(size_t)r];
        j.row = row; j.rank = r; j.dev = devs[(size_t)r];
        fill(j);
    }
    std::vector<std::thread> th;
    for (int r = 0; r < 4; ++r) th.emplace_back(worker, &(*jobs)[(size_t)r]);
    for (std::thread& x : th) x.join();
    *d = diff(c0, snap(hk));
    std::string rcs;
    bool up = true;
    for (Job& j : *jobs) {
        up = up && j.rc_init == RAD_OK && j.rc_launch == RAD_OK;
        rcs += fmt("%d/%d ", j.rc_init, j.rc_launch);
        if (row->fini && j.inst) row->fini(j.inst);
        j.inst = nullptr;
    }
    if (!up) *why = "init/launch rc " + rcs;
    return up;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* T(...) under a name the gather worker's local T (its slot count) does not shadow */
static RadTensor T_(void* d, uint32_t dt, std::vector<int64_t> shape) { return T(d, dt, std::move(shape)); }

/* ---- the gather's timing: deepseek41's MoE shape at M rows, ar_ll 0 vs 1 */
struct GJob {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0;
    int64_t M = 0;
    int iso = 60, bb = 40;
    Barrier* bar = nullptr;
    std::vector<double> iso_us;
    double bb_us = 0.0;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    void* inst = nullptr;
};
static void gworker(GJob* t) {
    const int64_t n = kW, k = 6, E = 384, M = t->M, T = M * k, El = E / 4;
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    std::vector<RadParam> ps = { PI("world_size", 4), PI("M", M), PI("n", n), PI("top_k", k), PS("dtype", "bf16"),
                                 PS("act", "none") };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, 4, &t->inst);
    if (t->rc_init != RAD_OK) {   /* pass the barriers the others wait at, launching nothing */
        for (int it = 0; it <= t->iso; ++it) t->bar->wait(4);
        return;
    }
    /* the routing: the same on every rank (seeded by M), the rank's own experts' slots sorted */
    uint64_t rs = 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    std::vector<int32_t> ex((size_t)T), sorted((size_t)T, -1);
    for (int64_t tok = 0; tok < M; ++tok)
        for (int64_t j = 0; j < k; ++j) {
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                e = (int32_t)(rnd() % (uint64_t)E);
                dup = false;
                for (int64_t q = 0; q < j; ++q) dup = dup || ex[(size_t)(tok * k + q)] == e;
            }
            ex[(size_t)(tok * k + j)] = e;
        }
    int64_t pos = 0;
    for (int32_t e = (int32_t)(t->rank * El); e < (int32_t)((t->rank + 1) * El); ++e)
        for (int64_t i = 0; i < T; ++i)
            if (ex[(size_t)i] == e) sorted[(size_t)pos++] = (int32_t)i;
    std::vector<uint16_t> ye((size_t)(T * n)), sh((size_t)(M * n)), sg((size_t)M, h_to_bf16(1.0f));
    std::vector<float> ew((size_t)T);
    for (size_t i = 0; i < ye.size(); ++i) ye[i] = h_to_bf16(act_value(t->rank, (int64_t)i));
    for (size_t i = 0; i < sh.size(); ++i) sh[i] = h_to_bf16(act_value(t->rank + 4, (int64_t)i));
    for (auto& w : ew) w = (float)(rnd() % 100000 + 1) / 40000.0f;
    void *dye = nullptr, *dew = nullptr, *dso = nullptr, *dsh = nullptr, *dsg = nullptr, *dy = nullptr;
    if (hipMalloc(&dye, ye.size() * 2) != hipSuccess || hipMalloc(&dew, ew.size() * 4) != hipSuccess ||
        hipMalloc(&dso, sorted.size() * 4) != hipSuccess || hipMalloc(&dsh, sh.size() * 2) != hipSuccess ||
        hipMalloc(&dsg, sg.size() * 2) != hipSuccess || hipMalloc(&dy, (size_t)(M * n) * 2) != hipSuccess) {
        t->rc_launch = RAD_E_NOMEM;
        for (int it = 0; it <= t->iso; ++it) t->bar->wait(4);
    } else {
        st_memcpy(dye, ye.data(), ye.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(dew, ew.data(), ew.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(dso, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(dsh, sh.data(), sh.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(dsg, sg.data(), sg.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = { T_(dye, RAD_BF16, { T, n }), T_(dew, RAD_F32, { M, k }), T_(dso, RAD_I32, { T }),
                                      T_(dsh, RAD_BF16, { M, n }), T_(dsg, RAD_BF16, { M, 1 }), T_(dy, RAD_BF16, { M, n }) };
        auto launch = [&]() {
            RadArgs a;
            std::memset(&a, 0, sizeof(a));
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps.data(); a.n_p = (int)ps.size();
            a.instance = t->inst; a.rank = t->rank; a.world_size = 4;
            const int rc = t->row->launch(&a, nullptr);
            return rc == RAD_LINK_ENDED ? RAD_OK : rc;
        };
        for (int w = 0; w < 3 && t->rc_launch == RAD_OK; ++w) t->rc_launch = launch();
        st_stream_sync(nullptr);
        for (int it = 0; it < t->iso; ++it) {
            t->bar->wait(4);
            if (t->rc_launch != RAD_OK) continue;
            const auto t0 = std::chrono::steady_clock::now();
            t->rc_launch = launch();
            st_stream_sync(nullptr);
            t->iso_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        t->bar->wait(4);
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < t->bb && t->rc_launch == RAD_OK; ++it) t->rc_launch = launch();
        st_stream_sync(nullptr);
        t->bb_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / t->bb;
    }
    for (void* p : { dye, dew, dso, dsh, dsg, dy }) if (p) st_free(p);
}
static void time_gather(const RadKernelInfo* grow, const std::vector<int>& devs, const Hooks& hk, int64_t M, double* iso,
                        double* bb, std::string* why, int v) {
    hk.sw("ar_ll", v);
    Barrier bar;
    std::vector<GJob> j(4);
    std::vector<std::thread> th;
    for (int r = 0; r < 4; ++r) {
        j[(size_t)r].row = grow; j[(size_t)r].rank = r; j[(size_t)r].dev = devs[(size_t)r]; j[(size_t)r].M = M;
        j[(size_t)r].bar = &bar;
    }
    for (int r = 0; r < 4; ++r) th.emplace_back(gworker, &j[(size_t)r]);
    for (std::thread& x : th) x.join();
    bool up = true;
    for (GJob& x : j) {
        up = up && x.rc_init == RAD_OK && x.rc_launch == RAD_OK;
        if (grow->fini && x.inst) grow->fini(x.inst);
    }
    if (!up) { *why = fmt("rc %d/%d %d/%d %d/%d %d/%d", j[0].rc_init, j[0].rc_launch, j[1].rc_init, j[1].rc_launch,
                          j[2].rc_init, j[2].rc_launch, j[3].rc_init, j[3].rc_launch); *iso = *bb = 0; return; }
    std::vector<double> slow;
    for (size_t it = 0; it < j[0].iso_us.size(); ++it) {
        double m = 0.0;
        for (int r = 0; r < 4; ++r) if (j[(size_t)r].iso_us.size() > it) m = std::max(m, j[(size_t)r].iso_us[it]);
        slow.push_back(m);
    }
    *iso = median(slow);
    *bb = 0.0;
    for (const GJob& x : j) *bb = std::max(*bb, x.bb_us);
}

static void arll_case(const Plugin& pl, const std::vector<int>& devs) {
    const char* name = "ar_ll";
    const RadKernelInfo* row = find(pl, "ar_oneshot_4rank_exact");
    const RadKernelInfo* grow = find(pl, "moe_gather_all_reduce_4rank");
    if (!row || !grow) { ok(name, false, "row ar_oneshot_4rank_exact / moe_gather_all_reduce_4rank absent"); return; }
    if ((int)devs.size() < 4) { skip(name, "needs 4 gfx1201 cards"); return; }
    if (!ar_peer_path_ok(devs, 4)) { skip(name, "no peer path between the cards (device.p2p)"); return; }
    Hooks hk;
    hk.sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    hk.val = (int (*)(const char*, int*))dlsym(pl.h, "rad_fn4_value");
    hk.stats = (int (*)(long long*, int))dlsym(pl.h, "rad_arll_stats");
    hk.settings = (void (*)(int, int, int))dlsym(pl.h, "r4d_arll_test_settings");
    if (!hk.sw || !hk.val || !hk.stats || !hk.settings) {
        ok("ar_ll hooks", false, "rad_fn4_switch / rad_fn4_value / rad_arll_stats / r4d_arll_test_settings absent");
        return;
    }
    hk.settings(1, 0, 0);   /* LL areas for every instance built from here on, the grid rule */
    {
        int v0 = -2, v1 = -2, v48 = -2;
        const int a = hk.sw("ar_ll", -1), b = hk.sw("ar_ll", 49), c = hk.sw("ar_ll", 1);
        hk.val("ar_ll", &v1);
        const int e = hk.sw("ar_ll", 48);
        hk.val("ar_ll", &v48);
        const int f = hk.sw("ar_ll", 0);
        hk.val("ar_ll", &v0);
        ok("ar_ll: the switch refuses -1 and 49, takes 1, 48 and 0 (read back)",
           a < 0 && b < 0 && c == 1 && e == 1 && f == 1 && v1 == 1 && v48 == 48 && v0 == 0,
           fmt("rc %d / %d / %d / %d / %d, values %d %d %d", a, b, c, e, f, v1, v48, v0));
    }
    /* AR: identity, OFF and ON */
    const std::vector<int64_t> rows = { 1, 2, 6, 7, 8, 9, 16, 24, 47, 48, 49, 64 };
    const int oop = 6;   /* T 16 */
    int64_t n_one = 0, n_two = 0, n_ll_rows = 0, n_plain = 0;
    for (int64_t T : rows) {
        if (T > 48) { ++n_plain; continue; }
        n_ll_rows += T;
        if (T * kW >= 32768) ++n_two; else ++n_one;
    }
    for (int pat : { 2, 0, 1 }) {
        std::vector<Job> off, on;
        Ctr d0, d1;
        std::string w0, w1;
        auto fill = [&](Job& j) { j.pat = pat; j.rows = rows; j.oop = oop; };
        const bool r0 = run(row, devs, hk, 0, &off, &d0, &w0, fill);
        const bool r1 = r0 && run(row, devs, hk, 1, &on, &d1, &w1, fill);
        ok(fmt("ar_ll AR %s: init (peer rendezvous, LL areas) and launches, ar_ll 0 then 1", pat_name(pat)).c_str(), r0 && r1,
           r0 ? w1 : w0);
        if (!r0 || !r1) break;
        for (size_t k = 0; k < rows.size(); ++k) {
            int64_t b0 = 0, b1 = 0, f0 = -1, f1 = -1;
            int hdiff = 0;
            for (int r = 0; r < 4; ++r) {
                const Job &a = off[(size_t)r], &b = on[(size_t)r];
                if (a.bad.size() <= k || b.bad.size() <= k) { b0 = b1 = -1; break; }
                b0 += a.bad[k]; b1 += b.bad[k];
                if (f0 < 0) f0 = a.first[k];
                if (f1 < 0) f1 = b.first[k];
                hdiff += a.hash[k] != b.hash[k];
            }
            const int64_t T = rows[k];
            const char* form = T > 48 ? "plain (past 48 rows)" : T * kW >= 32768 ? "LL two-shot" : "LL one-shot";
            ok(fmt("ar_ll AR %s T %lld x 5120 (%s, ON: %s): OFF and ON == fp32 ascending-rank fold on every rank, bit for "
                   "bit; ON == OFF", pat_name(pat), (long long)T, (int)k == oop ? "out of place" : "in place", form).c_str(),
               b0 == 0 && b1 == 0 && hdiff == 0,
               fmt("OFF %lld wrong (first %lld), ON %lld wrong (first %lld), %d rank hash(es) differ", (long long)b0,
                   (long long)f0, (long long)b1, (long long)f1, hdiff));
        }
        /* The "plain" counter counts launches of at most 48 rows only (rad_arll_stats [5]): the T 49 / 64 launches never reach the
         * LL pick, so they count nowhere -- 10 plain OFF, 0 ON (gpu02 12.11. 19:38: the first P0 expected 12 / 2 here). */
        ok(fmt("ar_ll AR %s counters OFF: no LL launch, %lld plain of <= 48 rows", pat_name(pat), (long long)(n_one + n_two)).c_str(),
           d0.v[0] == 0 && d0.v[1] == 0 && d0.v[6] == 0 && d0.v[4] == 0 && d0.v[5] == n_one + n_two, ctrs(d0));
        ok(fmt("ar_ll AR %s counters ON: %lld one-shot, %lld two-shot on rank 0 (%lld rows), 4 x that over the ranks, the "
               "%lld past 48 rows plain (uncounted), none declined", pat_name(pat), (long long)n_one, (long long)n_two,
               (long long)n_ll_rows, (long long)n_plain).c_str(),
           d1.v[0] == n_one && d1.v[1] == n_two && d1.v[2] == n_ll_rows && d1.v[6] == 4 * (n_one + n_two) &&
               d1.v[5] == 0 && d1.v[4] == 0 && d1.v[3] == 0,
           ctrs(d1));
    }
    /* GATHER: the D3 case under ar_ll 1 */
    {
        hk.sw("ar_ll", 1);
        const Ctr c0 = snap(hk);
        std::printf("  ar_ll GATHER: moe_gather_all_reduce_4rank's case under ar_ll 1 (lines below)\n");
        moe_gather_ar_case(pl, devs, "moe_gather_all_reduce_4rank");
        const Ctr d = diff(c0, snap(hk));
        ok("ar_ll GATHER counters: the gather's launches of <= 48 rows went LL (none declined)",
           d.v[3] > 0 && d.v[3] == d.v[0] + d.v[1] && d.v[4] == 0, ctrs(d));
        hk.sw("ar_ll", 0);
    }
    /* STRESS */
    for (int band = 0; band < 2; ++band) {
        std::vector<Job> st;
        Ctr d;
        std::string w;
        const int lo = band ? 7 : 1, hi = band ? 9 : 64;
        const bool r = run(row, devs, hk, 1, &st, &d, &w, [&](Job& j) {
            j.pat = 2; j.stress = 1000; j.stress_lo = lo; j.stress_hi = hi;
        });
        int64_t bad = 0, launches = 0, first = -1;
        for (const Job& j : st) { bad += j.stress_bad; launches += j.stress_launches; if (first < 0) first = j.stress_first; }
        ok(fmt("ar_ll STRESS %s: 1000 launches a rank back to back (batches of 8, each its own input), heights %d .. %d: "
               "every output == its fold, 0 bad", band ? "c8 band" : "mixed LL + plain", lo, hi).c_str(),
           r && bad == 0 && launches == 4000 && d.v[0] + d.v[1] > 0,
           r ? fmt("%lld wrong element(s) (first at launch %lld), %lld launches checked; %s", (long long)bad,
                   (long long)first, (long long)launches, ctrs(d).c_str())
             : w);
    }
    hk.sw("ar_ll", 0);
    /* TIME */
    {
        const std::vector<int64_t> tr = { 1, 8, 16, 48 };
        double iso[2][4] = {}, bb[2][4] = {};
        bool up = true;
        for (int v = 0; v < 2 && up; ++v) {
            std::vector<double> iso_us[4];
            Barrier bar;
            std::vector<Job> j;
            Ctr d;
            std::string w;
            int rk = 0;
            up = run(row, devs, hk, v, &j, &d, &w, [&](Job& x) { x.trows = tr; x.bar = &bar; x.iso_us = &iso_us[rk++]; });
            if (!up) { std::printf("  info  ar_ll timing not run: %s\n", w.c_str()); break; }
            const size_t per = (size_t)j[0].iso;
            for (size_t k = 0; k < tr.size(); ++k) {
                std::vector<double> slow;
                for (size_t it = 0; it < per; ++it) {
                    double m = 0.0;
                    for (int r = 0; r < 4; ++r)
                        if (iso_us[r].size() > k * per + it) m = std::max(m, iso_us[r][k * per + it]);
                    slow.push_back(m);
                }
                iso[v][k] = median(slow);
                for (int r = 0; r < 4; ++r)
                    if (j[(size_t)r].bb_us.size() > k) bb[v][k] = std::max(bb[v][k], j[(size_t)r].bb_us[k]);
            }
        }
        for (size_t k = 0; up && k < tr.size(); ++k)
            std::printf("  info  ar_ll time AR T %lld x 5120 bf16 (%s): isolated %.1f -> %.1f us a call (%+.1f %%), back to "
                        "back %.1f -> %.1f us (%+.1f %%) [ar_ll 0 -> 1, the slowest rank]\n",
                        (long long)tr[k], tr[k] * kW >= 32768 ? "two-shot" : "one-shot", iso[0][k], iso[1][k],
                        iso[0][k] > 0 ? 100.0 * (iso[1][k] / iso[0][k] - 1.0) : 0.0, bb[0][k], bb[1][k],
                        bb[0][k] > 0 ? 100.0 * (bb[1][k] / bb[0][k] - 1.0) : 0.0);
        for (int64_t M : tr) {
            double i0 = 0, b0 = 0, i1 = 0, b1 = 0;
            std::string w;
            time_gather(grow, devs, hk, M, &i0, &b0, &w, 0);
            if (w.empty()) time_gather(grow, devs, hk, M, &i1, &b1, &w, 1);
            if (!w.empty()) { std::printf("  info  ar_ll time GATHER M %lld not run: %s\n", (long long)M, w.c_str()); continue; }
            std::printf("  info  ar_ll time GATHER M %lld x 5120 top_k 6 (%s): isolated %.1f -> %.1f us a call (%+.1f %%), "
                        "back to back %.1f -> %.1f us (%+.1f %%) [ar_ll 0 -> 1, the slowest rank]\n",
                        (long long)M, M * kW >= 32768 ? "two-shot" : "one-shot", i0, i1, i0 > 0 ? 100.0 * (i1 / i0 - 1.0) : 0.0,
                        b0, b1, b0 > 0 ? 100.0 * (b1 / b0 - 1.0) : 0.0);
        }
    }
    hk.sw("ar_ll", 0);
    hk.settings(0, 0, 0);
}

}  // namespace arll

static void arll_cases(const Plugin& pl, const std::vector<int>& devs) {
    if (!sel("ar_ll")) return;
    if (!g_case) std::printf("\ncross-rank, lever A3 (ar_ll: the flags in the data words at decode sizes)\n");
    std::printf("  ar_ll\n");
    arll::arll_case(pl, devs);
}
