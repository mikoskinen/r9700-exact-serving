/* r4d_selftest_arf.inc -- lever DAR b (DRAFTER track 21.11.2026 round 58) on FOUR cards (included by r4d_selftest.cpp after
 * r4d_selftest_arc.inc; called beside arc_cases):
 *
 *   --case ar_fp8: the row ar_twoshot_4rank_exact (the DSpark drafter's [rows, 5120] bf16 reductions) through its own entry,
 *       launch parameter `wire` 0 (the exact wire) and 1 (the fp8 wire, r4d_ar_fp8.h: E4M3 + an E8M0 scale a 16-byte word,
 *       both pushes), rows 1 / 2 / 8 / 16 / 24 / 32 / 40 / 48 / 64, in place, activation-like values that are a function of
 *       (rank, row, column) only. wire 0 == the fp32 ascending-rank fold rounded once, bit for bit; wire 1 == the host fold
 *       of the fp8 wire (r4d_arf_sw_*: dequant(quant(sum of dequant(quant(x_r))))) bit for bit on EVERY rank at every row
 *       count (so batch-invariant: a row's bits do not depend on the rows beside it), the four ranks' hashes equal, and
 *       rad_arf_calls counts one fp8 launch a rank a size. INFO: the fp8 values' distance from the exact ones; back-to-back
 *       in-place times at 16 / 32 / 40 rows (the c32 / c16 drafter heights), the slowest rank's mean, exact at the served
 *       decode grid vs fp8 at wire_nb x wire_nt 2 x 1024 / 4 x 1024 / 4 x 256 / 8 x 256 / 8 x 512 / 16 x 256. */
#include <chrono>
#include "r4d_ar_fp8.h"   /* the host codec (the reference) */

namespace arf {

constexpr int64_t kW = 5120;

static float act(int rank, int64_t row, int64_t col) {
    uint64_t h = ((uint64_t)row * 0x9E3779B97F4A7C15ull + (uint64_t)col * 0xC2B2AE3D27D4EB4Full) ^ ((uint64_t)rank + 7) * 0xD1B54A32D192ED03ull;
    h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
    uint64_t r = (uint64_t)row * 0x94D049BB133111EBull + (uint64_t)rank * 0x2545F4914F6CDD1Dull;
    r ^= r >> 31;
    if ((h >> 60) == 0) return 0.0f;
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += (double)((h >> (13 * k)) & 0x1fff) / 4096.0 - 1.0;
    return (float)std::ldexp(s, (int)(r % 8) - 4);
}

struct Job {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0;
    std::vector<int64_t> rows;
    int wire = 0, nb = 16, nt = 256;
    std::vector<int64_t> trows;
    int iters = 200;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<int64_t> bad;
    std::vector<uint64_t> hash;
    std::vector<double> us;
    std::vector<uint16_t> last;   /* the largest size's output (the distance INFO) */
};

static void worker(Job* t) {
    int64_t most = 64;
    for (int64_t T : t->trows) most = std::max(most, T);
    std::vector<RadParam> ps = { PI("world_size", 4), PI("numel", most * kW), PS("dtype", "bf16"), PI("exact", 1) };
    void* buf = nullptr;
    if (st_set_device(t->dev) != hipSuccess) t->rc_init = RAD_E_DEVICE;
    else t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, 4, &t->inst);
    if (t->rc_init == RAD_OK && hipMalloc(&buf, (size_t)most * kW * 2) != hipSuccess) t->rc_launch = RAD_E_NOMEM;
    std::vector<RadParam> lp = ps;
    lp.push_back(PI("wire", t->wire)); lp.push_back(PI("wire_nb", t->nb)); lp.push_back(PI("wire_nt", t->nt));
    auto launch = [&](int64_t nr) {
        RadTensor ts[1] = { T(buf, RAD_BF16, { nr, kW }) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = 1;
        a.p = lp.data(); a.n_p = (int)lp.size();
        a.instance = t->inst; a.rank = t->rank; a.world_size = 4;
        return t->row->launch(&a, nullptr);
    };
    std::vector<uint16_t> h((size_t)most * kW), ref((size_t)most * kW);
    for (size_t k = 0; t->rc_init == RAD_OK && t->rc_launch == RAD_OK && k < t->rows.size(); ++k) {
        const int64_t T = t->rows[k], n = T * kW;
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(act(t->rank, i / kW, i % kW));
        st_memcpy(buf, h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        const int rc = launch(T);
        if (rc != RAD_OK && rc != RAD_LINK_ENDED) { t->rc_launch = rc; break; }
        st_stream_sync(nullptr);
        st_memcpy(h.data(), buf, (size_t)n * 2, hipMemcpyDeviceToHost);
        int64_t bad = 0;
        uint64_t fnv = 1469598103934665603ull;
        for (int64_t g = 0; g < n / 8; ++g) {
            float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int r = 0; r < 4; ++r) {
                float v[8];
                for (int l = 0; l < 8; ++l) v[l] = h_bf16(h_to_bf16(act(r, (g * 8 + l) / kW, (g * 8 + l) % kW)));
                if (t->wire == 1) { unsigned char c[8], s; r4d_arf_sw_enc8(v, c, s); r4d_arf_sw_dec8(c, s, v); }
                for (int l = 0; l < 8; ++l) acc[l] += v[l];
            }
            if (t->wire == 1) { unsigned char c[8], s; r4d_arf_sw_enc8(acc, c, s); r4d_arf_sw_dec8(c, s, acc); }
            for (int l = 0; l < 8; ++l) {
                const int64_t i = g * 8 + l;
                if (h[(size_t)i] != h_to_bf16(acc[l])) ++bad;
                fnv = (fnv ^ h[(size_t)i]) * 1099511628211ull;
            }
        }
        t->bad.push_back(bad); t->hash.push_back(fnv);
        if (k + 1 == t->rows.size()) t->last.assign(h.begin(), h.begin() + n);
    }
    for (size_t k = 0; t->rc_init == RAD_OK && t->rc_launch == RAD_OK && k < t->trows.size(); ++k) {
        const int64_t T = t->trows[k], n = T * kW;
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(act(t->rank, i / kW, i % kW));
        st_memcpy(buf, h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        for (int w = 0; w < 5 && (t->rc_launch == RAD_OK || t->rc_launch == RAD_LINK_ENDED); ++w) t->rc_launch = launch(T);
        st_stream_sync(nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < t->iters && (t->rc_launch == RAD_OK || t->rc_launch == RAD_LINK_ENDED); ++it) t->rc_launch = launch(T);
        st_stream_sync(nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        if (t->rc_launch == RAD_LINK_ENDED) t->rc_launch = RAD_OK;
        t->us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / t->iters);
    }
    if (t->rc_launch == RAD_LINK_ENDED) t->rc_launch = RAD_OK;
    if (buf) st_free(buf);
}

static bool run(const RadKernelInfo* row, const std::vector<int>& devs, int wire, int nb, int nt, const std::vector<int64_t>& rows,
                const std::vector<int64_t>& trows, std::vector<Job>* jobs, std::string* why) {
    jobs->assign(4, Job());
    std::vector<std::thread> th;
    for (int r = 0; r < 4; ++r) {
        Job& j = (*jobs)[(size_t)r];
        j.row = row; j.rank = r; j.dev = devs[(size_t)r]; j.rows = rows; j.trows = trows; j.wire = wire; j.nb = nb; j.nt = nt;
    }
    for (int r = 0; r < 4; ++r) th.emplace_back(worker, &(*jobs)[(size_t)r]);
    for (std::thread& x : th) x.join();
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

static void arf_case(const Plugin& pl, const std::vector<int>& devs) {
    const char* name = "ar_fp8";
    const RadKernelInfo* row = find(pl, "ar_twoshot_4rank_exact");
    if (!row) { ok(name, false, "row ar_twoshot_4rank_exact absent"); return; }
    if ((int)devs.size() < 4) { skip(name, "needs 4 gfx1201 cards"); return; }
    if (!ar_peer_path_ok(devs, 4)) { skip(name, "no peer path between the cards (device.p2p)"); return; }
    long (*calls)() = (long (*)())dlsym(pl.h, "rad_arf_calls");
    if (!calls) { ok("ar_fp8 hooks", false, "rad_arf_calls absent"); return; }
    const std::vector<int64_t> rows = { 1, 2, 8, 16, 24, 32, 40, 48, 64 };
    std::vector<Job> x, f;
    std::string wx, wf;
    const long c0 = calls();
    const bool rx = run(row, devs, 0, 2, 1024, rows, {}, &x, &wx);
    const long c1 = calls();
    const bool rf = rx && run(row, devs, 1, 16, 256, rows, {}, &f, &wf);
    const long c2 = calls();
    ok("ar_fp8: init (peer rendezvous) and launches, wire 0 then wire 1 (16 x 256)", rx && rf, rx ? wf : wx);
    if (!rx || !rf) return;
    for (size_t k = 0; k < rows.size(); ++k) {
        int64_t bx = 0, bf = 0;
        int hx = 0, hf = 0;
        for (int r = 0; r < 4; ++r) {
            bx += x[(size_t)r].bad[k]; bf += f[(size_t)r].bad[k];
            hx += x[(size_t)r].hash[k] != x[0].hash[k]; hf += f[(size_t)r].hash[k] != f[0].hash[k];
        }
        ok(fmt("ar_fp8 T %lld x 5120: wire 0 == the exact fold, wire 1 == the fp8 wire's host fold, on every rank bit for bit; "
               "the ranks' hashes equal", (long long)rows[k]).c_str(),
           bx == 0 && bf == 0 && hx == 0 && hf == 0,
           fmt("wire 0 %lld wrong, wire 1 %lld wrong, rank hashes differing %d / %d", (long long)bx, (long long)bf, hx, hf));
    }
    ok(fmt("ar_fp8 counter: wire 0 no fp8 launch, wire 1 one a rank a size (%d)", (int)(4 * rows.size())).c_str(),
       c1 - c0 == 0 && c2 - c1 == (long)(4 * rows.size()), fmt("wire 0 %ld, wire 1 %ld", c1 - c0, c2 - c1));
    /* round 59 (DAR b DEFECT fix): the row a small message selects in serving, ar_oneshot_4rank_exact, takes the fp8 form at
     * EVERY size under `wire` 1 (round 58's entry took it only from ts_min = 32,768 elements: T 1 / 2 here reduced exact,
     * so a drafter row's bits depended on its pass's row count) */
    if (const RadKernelInfo* r1 = find(pl, "ar_oneshot_4rank_exact")) {
        std::vector<Job> f1;
        std::string w1;
        const long d0 = calls();
        const bool ro = run(r1, devs, 1, 16, 256, rows, {}, &f1, &w1);
        const long d1 = calls();
        ok("ar_fp8: the one-shot row (ar_oneshot_4rank_exact) at wire 1: init and launches", ro, w1);
        if (ro) {
            for (size_t k = 0; k < rows.size(); ++k) {
                int64_t bf = 0;
                int hf = 0, same = 0;
                for (int r = 0; r < 4; ++r) {
                    bf += f1[(size_t)r].bad[k];
                    hf += f1[(size_t)r].hash[k] != f1[0].hash[k];
                    same += f1[(size_t)r].hash[k] == f[(size_t)r].hash[k];
                }
                ok(fmt("ar_fp8 one-shot row T %lld x 5120, wire 1: == the fp8 wire's host fold on every rank, = the two-shot row's "
                       "bytes (one form at every size: batch-invariant)", (long long)rows[k]).c_str(),
                   bf == 0 && hf == 0 && same == 4,
                   fmt("%lld wrong, rank hashes differing %d, ranks equal to the two-shot row %d of 4", (long long)bf, hf, same));
            }
            ok(fmt("ar_fp8 one-shot row counter: one fp8 launch a rank a size (%d)", (int)(4 * rows.size())).c_str(),
               d1 - d0 == (long)(4 * rows.size()), fmt("%ld", d1 - d0));
        }
    } else {
        ok(name, false, "row ar_oneshot_4rank_exact absent");
    }
    {
        double se = 0, sx = 0, mx = 0;
        const std::vector<uint16_t>&a = x[0].last, &b = f[0].last;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            const double u = h_bf16(a[i]), v = h_bf16(b[i]);
            se += (v - u) * (v - u); sx += u * u; mx = std::max(mx, std::fabs(v - u));
        }
        std::printf("  info  ar_fp8 values at 64 rows: rms(fp8 - exact) / rms(exact) %.5f, max |d| %.4g\n", sx > 0 ? std::sqrt(se / sx) : 0.0, mx);
    }
    const std::vector<int64_t> tr = { 16, 32, 40 };
    struct G { int wire, nb, nt; };
    const G gs[] = { { 0, 0, 1024 }, { 1, 2, 1024 }, { 1, 4, 1024 }, { 1, 4, 256 }, { 1, 8, 256 }, { 1, 8, 512 }, { 1, 16, 256 } };
    for (const G& g : gs) {
        std::vector<Job> j;
        std::string w;
        if (!run(row, devs, g.wire, g.nb, g.nt, {}, tr, &j, &w)) { std::printf("  info  ar_fp8 timing wire %d not run: %s\n", g.wire, w.c_str()); continue; }
        std::printf("  info  ar_fp8 time, back to back in place, %d launches, the slowest rank's mean: %s", j[0].iters,
                    g.wire ? fmt("fp8 %d x %d", g.nb, g.nt).c_str() : "exact (served decode grid)");
        for (size_t k = 0; k < tr.size(); ++k) {
            double u = 0.0;
            for (int r = 0; r < 4; ++r) if (j[(size_t)r].us.size() > k) u = std::max(u, j[(size_t)r].us[k]);
            std::printf("  T %lld %.1f us", (long long)tr[k], u);
        }
        std::printf("\n");
    }
}

}  // namespace arf

static void arf_cases(const Plugin& pl, const std::vector<int>& devs) {
    if (!sel("ar_fp8")) return;
    if (!g_case) std::printf("\ncross-rank, lever DAR b (the fp8 wire of the drafter's all-reduces)\n");
    std::printf("  ar_fp8\n");
    arf::arf_case(pl, devs);
}
