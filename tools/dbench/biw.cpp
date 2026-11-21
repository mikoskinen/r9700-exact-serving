/* tools/dbench/biw.cpp (DRAFTER track 21.11.2026 round 75, lever BIW) -- one card, through libr4d's own exported entries
 * (dlopen'd): the main model's three fused norms that deepseek41's BI issue cuts into <= 32-row pieces at a decode segment
 * of more rows (kBiNorm: mhc_post_collapse_rmsnorm = pcn_a / pcn_f, mhc_collapse_rmsnorm = cn_a, and the window key's
 * rmsnorm_rope_roundtrip_kv_store = kv_fs) vs ONE launch of the op's _w1k form (the 1024-thread workgroup a row at every M):
 *
 *  A  bytes: every output (out / xc / y; x / y; y + the cache) of the pieces (rows [0, 32), [32, 64), .. each its own launch,
 *     as rows(kBiNorm, ..)) vs the _w1k form over all M rows, at M 1 .. 96, on random rows, rows of large / tiny magnitudes,
 *     and rows holding -0.0 / inf / NaN. Control: the served op in ONE launch over M rows (its width follows M: it must differ
 *     somewhere above 32 rows, or the check could not see a width change).
 *  B  time: the pieces back to back on one stream vs the _w1k launch (HIP launch path, event-timed, p50 of 7 x ITERS);
 *     the served pieces' cost is read from stamps (a piece past the first: +3.1..+7.6 us a run on lq231's c16 trunk).
 * Usage: biw LIBR4D [ITERS] [W8 (0 bf16 cache, 1 fp8 window)] */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_abi.h"

using Entry = int (*)(const RadArgs*, RadStream);

static void hc(hipError_t e, const char* w) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e)); std::exit(2); }
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t r32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float uni(float lo, float hi) { return lo + (hi - lo) * (float)(r32() & 0xffffff) / 16777216.0f; }
static uint16_t h_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

static RadTensor T2(void* d, uint32_t dt, int64_t r, int64_t c, int64_t ld) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor T1(void* d, uint32_t dt, int64_t n) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
static RadTensor T4(void* d, uint32_t dt, int64_t a, int64_t b, int64_t c, int64_t e) {
    RadTensor t; std::memset(&t, 0, sizeof(t));
    t.data = d; t.dtype = dt; t.rank = 4; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; t.shape[3] = e;
    t.stride[3] = 1; t.stride[2] = e; t.stride[1] = c * e; t.stride[0] = b * c * e;
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadParam PF(const char* k, double v) { RadParam p; std::memset(&p, 0, sizeof(p)); p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

struct Call {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
    void fix() { a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1; }
};
static double p50(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: biw LIBR4D [ITERS] [W8]\n"); return 2; }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const bool w8 = argc > 3 && std::atoi(argv[3]) != 0;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto pcn = (Entry)dlsym(lib, "r4d_rad_mhc_post_collapse_rmsnorm");
    auto pcnw = (Entry)dlsym(lib, "r4d_rad_mhc_post_collapse_rmsnorm_w1k");
    auto cn = (Entry)dlsym(lib, "r4d_rad_mhc_collapse_rmsnorm");
    auto cnw = (Entry)dlsym(lib, "r4d_rad_mhc_collapse_rmsnorm_w1k");
    auto kv = (Entry)dlsym(lib, "r4d_rad_rmsnorm_rope_roundtrip_kv_store");
    auto kvw = (Entry)dlsym(lib, "r4d_rad_rmsnorm_rope_roundtrip_kv_store_w1k");
    if (!pcn || !pcnw || !cn || !cnw || !kv || !kvw) {
        std::fprintf(stderr, "symbols missing (%p %p %p %p %p %p)\n", (void*)pcn, (void*)pcnw, (void*)cn, (void*)cnw, (void*)kv, (void*)kvw);
        return 2;
    }
    std::printf("biw: %s, window cache %s\n", argv[1], w8 ? "fp8_e4m3" : "bf16");
    const int D = 5120, HC = 4, MAXM = 96, HD = 512, ROT = 64, BS = 64, CTX = 4096;
    const double eps = 1e-6;
    hipStream_t st; hc(hipStreamCreate(&st), "stream");
    hipEvent_t e0, e1; hc(hipEventCreate(&e0), "ev"); hc(hipEventCreate(&e1), "ev");
    const size_t nx = (size_t)MAXM * D, nh = nx * HC;
    uint16_t *dx, *dh, *dk;                                  /* inputs: x [M, D], h [M, 4D], the window key's kv [M, HD] */
    float *dpo, *dcb, *dpre, *dw, *dwk, *dcs;
    int32_t *dpos, *dslot;
    hc(hipMalloc(&dx, nx * 2), "m"); hc(hipMalloc(&dh, nh * 2), "m"); hc(hipMalloc(&dk, (size_t)MAXM * HD * 2), "m");
    hc(hipMalloc(&dpo, (size_t)MAXM * HC * 4), "m"); hc(hipMalloc(&dcb, (size_t)MAXM * HC * HC * 4), "m");
    hc(hipMalloc(&dpre, (size_t)MAXM * HC * 4), "m"); hc(hipMalloc(&dw, (size_t)D * 4), "m"); hc(hipMalloc(&dwk, (size_t)HD * 4), "m");
    hc(hipMalloc(&dcs, (size_t)CTX * ROT * 4), "m");
    hc(hipMalloc(&dpos, (size_t)MAXM * 4), "m"); hc(hipMalloc(&dslot, (size_t)MAXM * 4), "m");
    /* outputs, A = the pieces, B = the _w1k launch, C = the served op in one launch (control) */
    uint16_t *dO[3], *dXc[3], *dY[3], *dKy[3], *dKc[3];
    const int kcw = w8 ? (((HD + HD / 32) + 15) / 16 * 16) / 2 : HD;   /* the cache entry in bf16 slots (ds_w8_bytes(n) / 2) */
    const int NB = MAXM / BS + 4;                                        /* cache blocks */
    const size_t kcn = (size_t)NB * BS * kcw;
    for (int i = 0; i < 3; ++i) {
        hc(hipMalloc(&dO[i], nh * 2), "m"); hc(hipMalloc(&dXc[i], nx * 2), "m"); hc(hipMalloc(&dY[i], nx * 2), "m");
        hc(hipMalloc(&dKy[i], (size_t)MAXM * HD * 2), "m"); hc(hipMalloc(&dKc[i], kcn * 2), "m");
    }

    auto fill_inputs = [&](int kind) {
        std::vector<uint16_t> hx(nx), hh(nh), hk((size_t)MAXM * HD);
        std::vector<float> hpo((size_t)MAXM * HC), hcb((size_t)MAXM * HC * HC), hpre((size_t)MAXM * HC), hw(D), hwk(HD),
            hcs((size_t)CTX * ROT);
        std::vector<int32_t> hpos(MAXM), hslot(MAXM);
        for (int m = 0; m < MAXM; ++m) {
            const float sc = kind == 1 ? std::ldexp(1.0f, (int)(r32() % 121) - 60) : 1.0f;
            for (int i = 0; i < D; ++i) hx[(size_t)m * D + i] = h_bf16(uni(-3.0f, 3.0f) * sc);
            for (int i = 0; i < HC * D; ++i) hh[(size_t)m * HC * D + i] = h_bf16(uni(-8.0f, 8.0f) * sc);
            for (int i = 0; i < HD; ++i) hk[(size_t)m * HD + i] = h_bf16(uni(-4.0f, 4.0f) * sc);
            for (int k = 0; k < HC; ++k) { hpo[(size_t)m * HC + k] = uni(0.0f, 2.0f); hpre[(size_t)m * HC + k] = uni(0.0f, 1.0f); }
            for (int k = 0; k < HC * HC; ++k) hcb[(size_t)m * HC * HC + k] = uni(0.0f, 1.0f);
            hpos[m] = (int32_t)(r32() % CTX);
            hslot[m] = (m % 7 == 6) ? -1 : (int32_t)((m * 37 + 11) % (NB * BS));   /* a negative slot: no store */
        }
        for (int m = 0; m < MAXM; ++m)   /* distinct slots: a later row must not overwrite an earlier one's entry */
            for (int j = 0; j < m; ++j) if (hslot[m] >= 0 && hslot[m] == hslot[j]) hslot[m] = -1;
        if (kind == 2) {
            const uint16_t sp[] = { 0x8000, 0x0000, 0x7f80, 0xff80, 0x7fc1, 0x0001, 0x8001, 0x7f7f };
            for (size_t i = 0; i < nx; i += 97) hx[i] = sp[r32() % 8];
            for (size_t i = 0; i < nh; i += 389) hh[i] = sp[r32() % 8];
            for (size_t i = 0; i < hk.size(); i += 53) hk[i] = sp[r32() % 8];
            for (int m = 3; m < MAXM; m += 11) {
                for (int i = 0; i < D; ++i) hx[(size_t)m * D + i] = 0;
                for (int i = 0; i < HD; ++i) hk[(size_t)m * HD + i] = 0;
            }
        }
        for (int i = 0; i < D; ++i) hw[i] = uni(0.5f, 1.5f);
        for (int i = 0; i < HD; ++i) hwk[i] = uni(0.5f, 1.5f);
        for (int p = 0; p < CTX; ++p)
            for (int i = 0; i < ROT / 2; ++i) {
                const double th = p * std::pow(10000.0, -2.0 * i / ROT);
                hcs[(size_t)p * ROT + i] = (float)std::cos(th); hcs[(size_t)p * ROT + ROT / 2 + i] = (float)std::sin(th);
            }
        hc(hipMemcpy(dx, hx.data(), nx * 2, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dh, hh.data(), nh * 2, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dpo, hpo.data(), hpo.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dcb, hcb.data(), hcb.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dpre, hpre.data(), hpre.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dwk, hwk.data(), hwk.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dcs, hcs.data(), hcs.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dpos, hpos.data(), hpos.size() * 4, hipMemcpyHostToDevice), "cp");
        hc(hipMemcpy(dslot, hslot.data(), hslot.size() * 4, hipMemcpyHostToDevice), "cp");
    };
    /* one call of each op over rows [r0, r0 + M) into output set o */
    auto c_pcn = [&](Call& c, int r0, int M, int o) {
        c.t = { T2(dx + (size_t)r0 * D, RAD_BF16, M, D, D), T2(dh + (size_t)r0 * HC * D, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D),
                T2(dpo + (size_t)r0 * HC, RAD_F32, M, HC, HC), T2(dcb + (size_t)r0 * HC * HC, RAD_F32, M, HC * HC, HC * HC),
                T2(dO[o] + (size_t)r0 * HC * D, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D), T2(dpre + (size_t)r0 * HC, RAD_F32, M, HC, HC),
                T2(dXc[o] + (size_t)r0 * D, RAD_BF16, M, D, D), T1(dw, RAD_F32, D), T2(dY[o] + (size_t)r0 * D, RAD_BF16, M, D, D) };
        c.p = { PI("M", M), PI("n", D), PI("hc", HC), PF("eps", eps), PS("dtype", "bf16") }; c.fix();
    };
    auto c_cn = [&](Call& c, int r0, int M, int o) {
        c.t = { T2(dh + (size_t)r0 * HC * D, RAD_BF16, M, (int64_t)HC * D, (int64_t)HC * D), T2(dpre + (size_t)r0 * HC, RAD_F32, M, HC, HC),
                T2(dXc[o] + (size_t)r0 * D, RAD_BF16, M, D, D), T1(dw, RAD_F32, D), T2(dY[o] + (size_t)r0 * D, RAD_BF16, M, D, D) };
        c.p = { PI("M", M), PI("n", D), PI("hc", HC), PF("eps", eps), PS("dtype", "bf16") }; c.fix();
    };
    auto c_kv = [&](Call& c, int r0, int M, int o) {
        RadTensor cs = T2(dcs, RAD_F32, CTX, ROT, ROT);
        c.t = { T2(dk + (size_t)r0 * HD, RAD_BF16, M, HD, HD), T1(dwk, RAD_F32, HD), T2(dKy[o] + (size_t)r0 * HD, RAD_BF16, M, HD, HD),
                cs, T1(dpos + r0, RAD_I32, M), T1(dslot + r0, RAD_I32, M), T4(dKc[o], RAD_BF16, NB, 1, BS, kcw) };
        c.p = { PI("M", M), PI("n", HD), PF("eps", eps), PI("n_head", 1), PI("head_dim", HD), PI("rot", ROT), PI("off", HD - ROT),
                PS("mode", "gptj"), PI("inverse", 0), PI("block", 32), PS("format", "fp8_e4m3"), PS("scale", "ue8m0"),
                PS("dtype", "bf16"), PI("block_size", BS), PS("kv_dtype", w8 ? "fp8_e4m3" : "bf16") };
        c.fix();
    };
    auto get = [&](const uint16_t* d, size_t n) { std::vector<uint16_t> v(n); hc(hipMemcpy(v.data(), d, n * 2, hipMemcpyDeviceToHost), "cp"); return v; };
    auto ndiff = [](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) { long k = 0; for (size_t i = 0; i < a.size(); ++i) k += a[i] != b[i]; return k; };
    const int rows[] = { 1, 5, 16, 31, 32, 33, 40, 43, 47, 48, 49, 56, 63, 64, 65, 80, 96 };
    const int nrows = (int)(sizeof rows / sizeof rows[0]);
    int fails = 0, ctl_seen[3] = { 0, 0, 0 };
    long tot = 0;
    const char* opn[3] = { "mhc_post_collapse_rmsnorm", "mhc_collapse_rmsnorm", "rmsnorm_rope_roundtrip_kv_store" };
    for (int kind = 0; kind < 3; ++kind) {
        fill_inputs(kind);
        int fk[3] = { 0, 0, 0 };
        for (int M : rows) {
            for (int op = 0; op < 3; ++op) {
                for (int o = 0; o < 3; ++o) {
                    hc(hipMemset(dO[o], 0x5a, nh * 2), "ms"); hc(hipMemset(dXc[o], 0xa5, nx * 2), "ms"); hc(hipMemset(dY[o], 0xa5, nx * 2), "ms");
                    hc(hipMemset(dKy[o], 0x3c, (size_t)MAXM * HD * 2), "ms"); hc(hipMemset(dKc[o], 0x00, kcn * 2), "ms");
                }
                int rc = 0;
                Entry base = op == 0 ? pcn : op == 1 ? cn : kv, w1k = op == 0 ? pcnw : op == 1 ? cnw : kvw;
                auto mk = [&](Call& c, int r0, int m, int o) { if (op == 0) c_pcn(c, r0, m, o); else if (op == 1) c_cn(c, r0, m, o); else c_kv(c, r0, m, o); };
                for (int a = 0; a < M; a += 32) { Call c; mk(c, a, std::min(32, M - a), 0); rc |= base(&c.a, (RadStream)st); }   /* the pieces */
                { Call c; mk(c, 0, M, 1); rc |= w1k(&c.a, (RadStream)st); }                                                      /* one _w1k */
                { Call c; mk(c, 0, M, 2); rc |= base(&c.a, (RadStream)st); }                                                     /* control */
                hc(hipStreamSynchronize(st), "sync");
                long d1 = 0, d2 = 0; size_t nv = 0;
                auto cmp = [&](uint16_t* const* arr, size_t n) {
                    const auto a = get(arr[0], n), b = get(arr[1], n), c = get(arr[2], n);
                    d1 += ndiff(a, b); d2 += ndiff(a, c); nv += n;
                };
                if (op == 0) { cmp(dO, (size_t)M * HC * D); cmp(dXc, (size_t)M * D); cmp(dY, (size_t)M * D); }
                else if (op == 1) { cmp(dXc, (size_t)M * D); cmp(dY, (size_t)M * D); }
                else { cmp(dKy, (size_t)M * HD); cmp(dKc, kcn); }
                tot += (long)nv;
                const bool ok = !d1 && !rc;
                fk[op] += !ok;
                if (d2) ctl_seen[op]++;
                if (!ok || (kind == 0 && (M == 33 || M == 48 || M == 64)))
                    std::printf("A  kind %d M %2d %-32s rc %d: pieces vs _w1k %ld of %zu values differ (%s); served op in one launch "
                                "vs the pieces %ld (control)\n", kind, M, opn[op], rc, d1, nv, ok ? "IDENTICAL" : "DIFFER", d2);
            }
        }
        std::printf("A  kind %d (%s): cells where _w1k differs from the pieces: pcn %d, cn %d, kv %d of %d each\n", kind,
                    kind == 0 ? "typical" : kind == 1 ? "magnitudes 2^-60 .. 2^60 a row" : "specials", fk[0], fk[1], fk[2], nrows);
        fails += fk[0] + fk[1] + fk[2];
    }
    std::printf("A  %ld values compared; _w1k vs the pieces: %d cells differ -> %s; control (served op in one launch differs from the "
                "pieces somewhere): pcn %d cells, cn %d, kv %d\n", tot, fails, fails ? "FAIL" : "PASS", ctl_seen[0], ctl_seen[1], ctl_seen[2]);

    fill_inputs(0);
    for (int M : { 32, 33, 40, 43, 48, 56, 64 }) {
        auto timed = [&](auto&& body) {
            std::vector<double> v;
            for (int r = 0; r < 7; ++r) {
                body(); hc(hipStreamSynchronize(st), "s");
                hc(hipEventRecord(e0, st), "e");
                for (int i = 0; i < iters; ++i) body();
                hc(hipEventRecord(e1, st), "e"); hc(hipEventSynchronize(e1), "e");
                float ms = 0; hc(hipEventElapsedTime(&ms, e0, e1), "el");
                v.push_back(1000.0 * ms / iters);
            }
            return p50(v);
        };
        double tp[3], tw[3], tc[3];
        for (int op = 0; op < 3; ++op) {
            Entry base = op == 0 ? pcn : op == 1 ? cn : kv, w1k = op == 0 ? pcnw : op == 1 ? cnw : kvw;
            auto mk = [&](Call& c, int r0, int m, int o) { if (op == 0) c_pcn(c, r0, m, o); else if (op == 1) c_cn(c, r0, m, o); else c_kv(c, r0, m, o); };
            std::vector<Call> pc((size_t)(M + 31) / 32);
            for (int a = 0, i = 0; a < M; a += 32, ++i) mk(pc[(size_t)i], a, std::min(32, M - a), 0);
            Call cw, cc; mk(cw, 0, M, 1); mk(cc, 0, M, 2);
            tp[op] = timed([&] { for (auto& c : pc) base(&c.a, (RadStream)st); });
            tw[op] = timed([&] { w1k(&cw.a, (RadStream)st); });
            tc[op] = timed([&] { base(&cc.a, (RadStream)st); });
        }
        std::printf("B  M %2d: us back to back (p50 of 7 x %d): pcn pieces %.2f -> _w1k %.2f (served op one launch %.2f) | cn %.2f -> %.2f "
                    "(%.2f) | kv %.2f -> %.2f (%.2f)\n", M, iters, tp[0], tw[0], tc[0], tp[1], tw[1], tc[1], tp[2], tw[2], tc[2]);
    }
    std::printf("biw: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
