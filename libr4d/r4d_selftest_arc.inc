/* r4d_selftest_arc.inc -- ds41-int10's lever ARC on FOUR cards (included by r4d_selftest.cpp before device_phase; called
 * from it beside the four-rank all-reduce rows):
 *
 *   --case ar_arcoded (or ds41_arcoded): the row ar_twoshot_4rank_exact (deepseek41's prefill all-reduce, no overlap) through
 *       its own entry, libr4d fn4 ds41_arcoded 0 (the plain wire) and 1024 (the coded wire from 1024 rows), [T, 5120] bf16
 *       operands as deepseek41 issues them, T 1023 / 1024 / 1100 / 2048 / 4096 / 4160 (below the threshold, at it, the grids
 *       8 / 16 / 24, max-num-batched-tokens), T 1100 out of place; three inputs (activation-like rows with exact zeros,
 *       ar_rough_value's 25 binades: mostly raw groups, ar_coded_value's edge groups: range 14 / 15, a subnormal, an inf,
 *       zeros, the top exponents). Every rank, every size, BOTH settings == the fp32 ascending-rank fold rounded once, bit
 *       for bit (so ON == OFF), plus each output's hash ON vs OFF. The counters (rad_arc_stats) prove the wire: OFF no coded
 *       launch, ON one a size from 1024 rows (on every rank) and the T 1023 launch plain below the threshold. The threshold's
 *       edge at 1100 (1024 / 1099 plain, 1100 coded); the switch's refusals (-1, 1048577). INFO: back-to-back times out of
 *       place at T 1100 / 2048 / 4096, OFF vs ON (the slowest rank's mean). */
#include <chrono>
#include "r4d_ar_arc.h"   /* lever ARC's rule: the grid a size takes (r4d_arc_grid) */

namespace arc {

constexpr int64_t kW = 5120;   /* deepseek41's residual width */

/* deepseek41's operand: a [rows, 5120] bf16 slice (the entry counts rows over the last axis) */
static RadTensor arc_tensor(void* d, int64_t rows) { return T(d, RAD_BF16, { rows, kW }); }

/* activation-like: a hashed near-gaussian (four uniforms), a per-row scale 2^-4 .. 2^3, 1 in 16 exactly zero */
static float act_value(int rank, int64_t i) {
    uint64_t h = (uint64_t)i * 0x9E3779B97F4A7C15ull ^ ((uint64_t)rank + 7) * 0xD1B54A32D192ED03ull;
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
    uint64_t r = (uint64_t)(i / kW) * 0x94D049BB133111EBull;
    r ^= r >> 31;
    if ((h >> 60) == 0) return 0.0f;
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += (double)((h >> (13 * k)) & 0x1fff) / 4096.0 - 1.0;
    return (float)std::ldexp(s * (1.0 + 0.25 * rank), (int)(r % 8) - 4);
}
/* 2 act, 0 ar_rough_value, 1 ar_coded_value */
static float value(int pat, int rank, int64_t i) { return pat == 2 ? act_value(rank, i) : ar_value(pat, rank, i); }
static const char* pat_name(int pat) { return pat == 2 ? "act" : pat == 1 ? "coded-form edges" : "rough (25 binades)"; }

struct Job {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0, pat = 2;
    std::vector<int64_t> rows;           /* the launches, T rows of kW each */
    int oop = -1;                        /* index into rows launched out of place */
    std::vector<int64_t> trows;          /* INFO timing: out of place, back to back */
    int iters = 20;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<int64_t> bad, first;     /* per size: elements != the fold, the first */
    std::vector<uint64_t> hash;          /* per size: FNV-1a over the output's bits */
    std::vector<double> us;              /* per timing size: mean us a launch */
};

static void worker(Job* t) {
    int64_t most = 0;
    for (int64_t T : t->rows) most = std::max(most, T * kW);
    for (int64_t T : t->trows) most = std::max(most, T * kW);
    std::vector<RadParam> ps = { PI("world_size", 4), PI("numel", most), PS("dtype", "bf16"), PI("exact", 1) };
    void* in = nullptr;
    void* out = nullptr;
    if (st_set_device(t->dev) != hipSuccess) t->rc_init = RAD_E_DEVICE;
    else t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, 4, &t->inst);
    if (t->rc_init == RAD_OK && (hipMalloc(&in, (size_t)most * 2) != hipSuccess || hipMalloc(&out, (size_t)most * 2) != hipSuccess))
        t->rc_launch = RAD_E_NOMEM;
    auto launch = [&](int64_t T, bool oop) {
        RadTensor ts[2] = { arc_tensor(in, T), arc_tensor(out, T) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = oop ? 2 : 1;
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst; a.rank = t->rank; a.world_size = 4;
        return t->row->launch(&a, nullptr);
    };
    std::vector<uint16_t> h((size_t)most);
    for (size_t k = 0; t->rc_init == RAD_OK && t->rc_launch == RAD_OK && k < t->rows.size(); ++k) {
        const int64_t T = t->rows[k], n = T * kW;
        const bool oop = (int)k == t->oop;
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(value(t->pat, t->rank, i));
        st_memcpy(in, h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        if (oop) st_memset(out, 0, (size_t)n * 2);
        const int rc = launch(T, oop);
        if (rc != RAD_OK) { t->rc_launch = rc; break; }
        st_stream_sync(nullptr);
        st_memcpy(h.data(), oop ? out : in, (size_t)n * 2, hipMemcpyDeviceToHost);
        int64_t bad = 0, first = -1;
        uint64_t fnv = 1469598103934665603ull;
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < 4; ++r) acc += h_bf16(h_to_bf16(value(t->pat, r, i)));
            if (h[(size_t)i] != h_to_bf16(acc)) { if (first < 0) first = i; ++bad; }
            fnv = (fnv ^ h[(size_t)i]) * 1099511628211ull;
        }
        t->bad.push_back(bad); t->first.push_back(first); t->hash.push_back(fnv);
    }
    for (size_t k = 0; t->rc_init == RAD_OK && t->rc_launch == RAD_OK && k < t->trows.size(); ++k) {
        const int64_t T = t->trows[k], n = T * kW;
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(act_value(t->rank, i));
        st_memcpy(in, h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        for (int w = 0; w < 3 && t->rc_launch == RAD_OK; ++w) t->rc_launch = launch(T, true);
        st_stream_sync(nullptr);   /* after the warm launches the ranks are in step (each waits on its peers) */
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < t->iters && t->rc_launch == RAD_OK; ++it) t->rc_launch = launch(T, true);
        st_stream_sync(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        t->us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / t->iters);
    }
    if (in) st_free(in);
    if (out) st_free(out);
}

struct Hooks {
    int (*sw)(const char*, int) = nullptr;
    int (*val)(const char*, int*) = nullptr;
    int (*stats)(long long*, int) = nullptr;
};
struct Ctr { long long v[7] = {}; };
static Ctr snap(const Hooks& k) { Ctr c; if (k.stats) k.stats(c.v, 7); return c; }

/* one run of four rank threads under ds41_arcoded `thr` */
static bool run(const Plugin& pl, const RadKernelInfo* row, const std::vector<int>& devs, const Hooks& hk, int thr, int pat,
                const std::vector<int64_t>& rows, int oop, const std::vector<int64_t>& trows, std::vector<Job>* jobs,
                Ctr* d, std::string* why) {
    (void)pl;
    const int sr = hk.sw("ds41_arcoded", thr);
    int applied = -2;
    hk.val("ds41_arcoded", &applied);
    if (sr != 1 || applied != thr) { *why = fmt("rad_fn4_switch(ds41_arcoded, %d) = %d, applied %d", thr, sr, applied); return false; }
    const Ctr c0 = snap(hk);
    jobs->assign(4, Job());
    std::vector<std::thread> th;
    for (int r = 0; r < 4; ++r) {
        Job& j = (*jobs)[(size_t)r];
        j.row = row; j.rank = r; j.dev = devs[(size_t)r]; j.pat = pat; j.rows = rows; j.oop = oop; j.trows = trows;
    }
    for (int r = 0; r < 4; ++r) th.emplace_back(worker, &(*jobs)[(size_t)r]);
    for (std::thread& x : th) x.join();
    const Ctr c1 = snap(hk);
    for (int i = 0; i < 7; ++i) d->v[i] = c1.v[i] - c0.v[i];
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

static std::string ctrs(const Ctr& d) {
    return fmt("coded %lld (rows %lld, all ranks %lld), plain %lld (rows %lld), below %lld, declined %lld", d.v[0], d.v[1], d.v[6],
               d.v[2], d.v[3], d.v[4], d.v[5]);
}

static void arc_case(const Plugin& pl, const std::vector<int>& devs) {
    const char* name = "ar_arcoded";
    const RadKernelInfo* row = find(pl, "ar_twoshot_4rank_exact");
    if (!row) { ok(name, false, "row ar_twoshot_4rank_exact absent"); return; }
    if ((int)devs.size() < 4) { skip(name, "needs 4 gfx1201 cards"); return; }
    if (!ar_peer_path_ok(devs, 4)) { skip(name, "no peer path between the cards (device.p2p)"); return; }
    Hooks hk;
    hk.sw = (int (*)(const char*, int))dlsym(pl.h, "rad_fn4_switch");
    hk.val = (int (*)(const char*, int*))dlsym(pl.h, "rad_fn4_value");
    hk.stats = (int (*)(long long*, int))dlsym(pl.h, "rad_arc_stats");
    if (!hk.sw || !hk.val || !hk.stats) { ok("ar_arcoded hooks", false, "rad_fn4_switch / rad_fn4_value / rad_arc_stats absent"); return; }
    {
        int v = -2;
        const int a = hk.sw("ds41_arcoded", -1), b = hk.sw("ds41_arcoded", 1048577), c = hk.sw("ds41_arcoded", 0);
        hk.val("ds41_arcoded", &v);
        ok("ar_arcoded: the switch refuses -1 and 1048577, takes 0 (read back 0)", a < 0 && b < 0 && c == 1 && v == 0,
           fmt("rc %d / %d / %d, value %d", a, b, c, v));
    }
    const std::vector<int64_t> rows = { 1023, 1024, 1100, 2048, 4096, 4160 };
    const int oop = 2;
    int64_t from = 0, from_rows = 0;
    for (int64_t T : rows) if (T >= 1024) { ++from; from_rows += T; }
    for (int pat : { 2, 0, 1 }) {
        std::vector<Job> off, on;
        Ctr d0, d1;
        std::string w0, w1;
        const bool r0 = run(pl, row, devs, hk, 0, pat, rows, oop, {}, &off, &d0, &w0);
        const bool r1 = r0 && run(pl, row, devs, hk, 1024, pat, rows, oop, {}, &on, &d1, &w1);
        ok(fmt("ar_arcoded %s: init (peer rendezvous) and launches, ds41_arcoded 0 then 1024", pat_name(pat)).c_str(), r0 && r1,
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
            const bool coded = rows[k] >= 1024;
            ok(fmt("ar_arcoded %s T %lld x 5120 (%s, %s under 1024): OFF and ON == fp32 ascending-rank fold on every rank, bit for "
                   "bit; ON == OFF", pat_name(pat), (long long)rows[k], (int)k == oop ? "out of place" : "in place",
                   coded ? fmt("coded at %d blocks", r4d_arc_grid(rows[k] * kW)).c_str() : "plain").c_str(),
               b0 == 0 && b1 == 0 && hdiff == 0,
               fmt("OFF %lld wrong (first %lld), ON %lld wrong (first %lld), %d rank hash(es) differ", (long long)b0, (long long)f0,
                   (long long)b1, (long long)f1, hdiff));
        }
        ok(fmt("ar_arcoded %s counters OFF: no coded launch, %d plain", pat_name(pat), (int)rows.size()).c_str(),
           d0.v[0] == 0 && d0.v[6] == 0 && d0.v[2] == (long long)rows.size() && d0.v[4] == 0 && d0.v[5] == 0, ctrs(d0));
        ok(fmt("ar_arcoded %s counters ON (1024): %lld coded on rank 0 and 4 x that over the ranks, the T 1023 launch plain below "
               "the threshold, none declined", pat_name(pat), (long long)from).c_str(),
           d1.v[0] == from && d1.v[1] == from_rows && d1.v[6] == 4 * from && d1.v[2] == (long long)rows.size() - from &&
               d1.v[4] == (long long)rows.size() - from && d1.v[5] == 0,
           ctrs(d1));
    }
    {   /* the threshold's edge */
        std::vector<Job> j;
        Ctr d;
        std::string w;
        const std::vector<int64_t> er = { 1024, 1099, 1100 };
        const bool r = run(pl, row, devs, hk, 1100, 2, er, -1, {}, &j, &d, &w);
        int64_t bad = 0;
        for (const Job& x : j) for (int64_t b : x.bad) bad += b;
        ok("ar_arcoded act at ds41_arcoded 1100: T 1024 / 1099 plain, 1100 coded; all == the fold", r && bad == 0 && d.v[0] == 1 &&
               d.v[1] == 1100 && d.v[4] == 2 && d.v[2] == 2,
           r ? fmt("%lld wrong; %s", (long long)bad, ctrs(d).c_str()) : w);
    }
    {   /* INFO: back-to-back, out of place */
        std::vector<Job> a, b;
        Ctr da, db;
        std::string wa, wb;
        const std::vector<int64_t> tr = { 1100, 2048, 4096 };
        const bool ra = run(pl, row, devs, hk, 0, 2, {}, -1, tr, &a, &da, &wa);
        const bool rb = ra && run(pl, row, devs, hk, 1024, 2, {}, -1, tr, &b, &db, &wb);
        if (ra && rb) {
            for (size_t k = 0; k < tr.size(); ++k) {
                double ua = 0.0, ub = 0.0;
                for (int r = 0; r < 4; ++r) {
                    if (a[(size_t)r].us.size() > k) ua = std::max(ua, a[(size_t)r].us[k]);
                    if (b[(size_t)r].us.size() > k) ub = std::max(ub, b[(size_t)r].us[k]);
                }
                std::printf("  info  ar_arcoded time T %lld x 5120 bf16, back to back out of place, %d launches, the slowest rank's "
                            "mean: plain %.1f us, coded (%d blocks) %.1f us = %+.1f %%\n", (long long)tr[k], a[0].iters, ua,
                            r4d_arc_grid(tr[k] * kW), ub, ua > 0 ? 100.0 * (ub / ua - 1.0) : 0.0);
            }
            std::printf("  info  ar_arcoded timing counters: OFF %s | ON %s\n", ctrs(da).c_str(), ctrs(db).c_str());
        } else {
            std::printf("  info  ar_arcoded timing not run: %s\n", (ra ? wb : wa).c_str());
        }
    }
    hk.sw("ds41_arcoded", 0);
}

}  // namespace arc

static void arc_cases(const Plugin& pl, const std::vector<int>& devs) {
    if (!sel("ar_arcoded") && !sel("ds41_arcoded")) return;
    if (!g_case) std::printf("\ncross-rank, lever ARC (ds41_arcoded: the coded wire at overlap 0)\n");
    std::printf("  ar_arcoded\n");
    arc::arc_case(pl, devs);
}
