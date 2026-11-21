/* tests/r4d_emu: qfn-on-ds lever P-2 (Qwen loop round 11, QUEUE Q8) on the host SIMT emulator (the GPU twin: r4d_selftest
 * --perfatwalk's byte line) -- the q-tiled union walk r4d_attn_walk_kernel<R, M> (libr4d/r4d_attn_walk.h) against today's
 * sparse prefill attention row, r4d_attn_decode_kernel<1, 16, 256, 6, 4, fp8, G8_D_OPT | O_V8> at one split (the kernel
 * attn_decode_gq_h256_gqa6_bs4_fp8kv launches for a prefill piece: each query row its own "sequence", its table its QSA
 * selection), byte for byte over the split-KV partials both write (f16 O / l, then m and l a row) -- the merge reads only
 * those, so equal partials give equal o / codes / scales.
 *   Selections like the real ones: consecutive rows share a sliding band of pages plus a few scattered ones, ascending, with
 *   one entry out of order near the end and the row's own page last; contexts = full pages (a full last tile) or the own
 *   page partly used (a partial last tile: the cache path's clamps); rows not a multiple of R; R 4 / 8 / 16, M 7 / 15
 *   (the ring's spill-over: tiles read from the cache because a page left the ring or came out of order); the plain union and
 *   the re-load lists (every full tile from the ring).
 * NEGATIVE CONTROLS: the compare finds a one-code change in one selected page of the cache (walk on the changed cache vs
 * today on the original), and one changed union position (a tile read from a wrong ring slot).
 *
 *   r4d_emu_qwalk [quick] */
#include <hip/hip_runtime.h>

#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"
#define R4D_WALK_KERNEL_ONLY 1
#include "r4d_attn_walk.h"
#include "r4d_attn_walk_sched.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

/* G8_D_OPT | O_V8 (r4d_attn_paged_gqa8.hip: the one-wave decode workgroup of the gqa6 / page-4 unit) */
constexpr int kOptToday = 3429947 | O_V8;
constexpr int HD = 256, NQ = 6, BS = 4, ROWB = 2 * HD;

int g_cases = 0, g_fail = 0;
std::mt19937 rng(20261121);

struct Sel {
    int S = 0, W = 0, NB = 0;
    std::vector<int32_t> table, used;   /* [S][W], [S] */
};

/* rows r = 0 .. S - 1 at positions P0 + r; each row ~n pages: a band near its position's window, scattered pages, one
 * entry out of order, its own page last */
Sel make_sel(int S, int P0, int n, bool partial) {
    Sel s;
    s.S = S;
    s.NB = (P0 + S) / BS + 1;
    s.W = n + 2;
    s.table.assign((size_t)S * s.W, 0);
    s.used.assign(S, 0);
    std::uniform_int_distribution<int> any(0, (P0 + S) / BS - 1);
    std::vector<int> scat(n / 3);
    for (auto& x : scat) x = any(rng);
    for (int r = 0; r < S; ++r) {
        const int pos = P0 + r, own = pos / BS;
        std::vector<int> v;
        const int band0 = std::max(0, own - n / 2 - 7 + (r / 5) % 9);
        for (int k = 0; k < n / 2; ++k) if (band0 + k < own) v.push_back(band0 + k);
        for (size_t k = 0; k < scat.size(); ++k)
            if ((k + r / 3) % 4 != 0 && scat[k] < own) v.push_back(scat[k]);       /* shared with the neighbours, mostly */
        while ((int)v.size() < n - 1) { const int x = any(rng) % std::max(1, own); v.push_back(x); }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        if ((int)v.size() > n - 1) v.resize(n - 1);
        if (v.size() >= 4 && (r % 3) != 2) std::swap(v[v.size() - 1], v[v.size() / 3]);   /* one entry out of order */
        v.push_back(own);
        for (size_t k = 0; k < v.size(); ++k) s.table[(size_t)r * s.W + k] = v[k];
        s.used[r] = partial ? (int)(v.size() - 1) * BS + pos % BS + 1 : (int)v.size() * BS;
    }
    return s;
}

struct Ops {
    std::vector<uint16_t> q;
    std::vector<uint8_t> kv;
    std::vector<float> kd;
};

Ops make_ops(const Sel& s) {
    Ops o;
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    o.q.resize((size_t)s.S * NQ * HD);
    for (auto& x : o.q) { const float f = u(rng); uint32_t b; std::memcpy(&b, &f, 4); x = (uint16_t)(b >> 16); }
    o.kv.resize((size_t)s.NB * BS * ROWB);
    /* E4M3 codes of magnitude 1/8 .. 2 (exponent field 4 .. 8): every key's score in a few units, so the softmax weighs
     * every tile (random bytes reach 448 and leave one key a row: a change elsewhere would not show) */
    std::uniform_int_distribution<int> sg(0, 1), ex(4, 8), mt(0, 7);
    for (auto& x : o.kv) x = (uint8_t)((sg(rng) << 7) | (ex(rng) << 3) | mt(rng));
    o.kd.resize(s.S);
    for (auto& x : o.kd) x = 0.75f + 0.25f * (u(rng) + 2.f) / 2.f;
    return o;
}

R4DArgs args(const Sel& s, const Ops& o, std::vector<uint8_t>& kv, std::vector<uint8_t>& scratch) {
    R4DArgs a;
    std::memset(&a, 0, sizeof(a));
    a.q = o.q.data();
    a.kv = kv.data();
    a.block_table = s.table.data();
    a.seqused_k = s.used.data();
    a.k_descale = o.kd.data();
    a.num_seqs = s.S; a.q_len = 1; a.q_heads = NQ; a.kv_heads = 1; a.head_dim = HD; a.block_size = BS; a.max_blocks = s.W;
    a.kv_block_stride = BS * ROWB; a.kv_head_stride = BS * ROWB;
    a.scale = 1.0f / 16.0f;
    a.splits = 1; a.max_ctx = s.W * BS; a.causal = 1;
    const size_t nrow = (size_t)s.S * NQ;
    scratch.assign(nrow * (HD * 2 + 8), 0xab);
    a.scratch = scratch.data();
    return a;
}

/* the load lists of groups of R rows (libr4d/r4d_attn_walk_sched.h): the plain union, or with re-loads */
void unions(const Sel& s, int R, int M, bool rl, std::vector<int32_t>& ul, std::vector<int32_t>& uo, std::vector<int32_t>& up) {
    static std::vector<int32_t> lastp;
    ul.clear(); uo.assign(1, 0); up.assign((size_t)s.S * s.W, -1);
    for (int r0 = 0; r0 < s.S; r0 += R) {
        r4d_walk_group_list(s.table.data(), s.W, s.used.data(), r0, std::min(s.S, r0 + R), M, rl, ul, up, lastp);
        uo.push_back((int32_t)ul.size());
    }
}

std::string fmt_case(int S, int P0, int n, bool partial, int R, int M, const char* what) {
    char b[400];
    std::snprintf(b, sizeof b, "rows %d from %d, ~%d pages a row, %s last page, R %d M %d: %s", S, P0, n,
                  partial ? "partial" : "full", R, M, what);
    return b;
}

void check(bool ok, const std::string& name) {
    ++g_cases;
    if (!ok) ++g_fail;
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", name.c_str());
}

std::vector<uint8_t> today(const Sel& s, const Ops& o) {
    std::vector<uint8_t> kv = o.kv, sc;
    R4DArgs a = args(s, o, kv, sc);
    hipLaunchKernelGGL((r4d_attn_decode_kernel<1, 16, 256, 6, 4, 0, kOptToday>), dim3(1, 1, s.S), dim3(32), 0, nullptr, a, 1);
    return sc;
}

void launch_walk(int R, int M, const R4DArgs& a, int groups, const int* ul, const int* uo, const int* up, int* ws) {
#define WL(RR, MM) if (R == RR && M == MM) { hipLaunchKernelGGL((r4d_attn_walk_kernel<RR, MM>), dim3(groups, 1, 1), \
                                                               dim3((RR + r4dwalk::NLW) * 32), 0, nullptr, a, ul, uo, up, ws, 0); return; }
    WL(4, 15) WL(8, 15) WL(16, 15) WL(8, 7) WL(16, 7)
#undef WL
    std::printf("  no instance R %d M %d\n", R, M);
}

std::vector<uint8_t> walk(const Sel& s, const Ops& o, int R, int M, bool rl, int* ws,
                          const std::vector<uint8_t>* kv_over = nullptr, int mutate_up = -1) {
    std::vector<uint8_t> kv = kv_over ? *kv_over : o.kv, sc;
    R4DArgs a = args(s, o, kv, sc);
    std::vector<int32_t> ul, uo, up;
    unions(s, R, M, rl, ul, uo, up);
    if (mutate_up >= 0) {   /* row mutate_up's entries one place off in its union: its ring tiles read wrong slots */
        for (int b = 0; b < s.W; ++b)
            if (up[(size_t)mutate_up * s.W + b] > 0) up[(size_t)mutate_up * s.W + b] -= 1;
    }
    launch_walk(R, M, a, (s.S + R - 1) / R, ul.data(), uo.data(), up.data(), ws);
    return sc;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    std::printf("r4d_emu_qwalk: lever P-2's union walk vs today's sparse prefill attention row, partials byte for byte\n");
    struct C { int S, P0, n; bool partial; };
    std::vector<C> cs = { { 21, 900, 60, false }, { 37, 1500, 90, true } };
    if (!quick) cs.push_back({ 33, 2600, 130, true });
    const int cfg[][3] = { { 4, 15, 0 }, { 8, 15, 0 }, { 16, 15, 0 }, { 8, 7, 0 }, { 16, 7, 0 }, { 8, 15, 1 }, { 16, 15, 1 },
                           { 16, 7, 1 } };
    for (const C& c : cs) {
        const Sel s = make_sel(c.S, c.P0, c.n, c.partial);
        const Ops o = make_ops(s);
        const std::vector<uint8_t> ref = today(s, o);
        const std::vector<uint8_t> ref2 = today(s, o);
        check(ref == ref2, fmt_case(c.S, c.P0, c.n, c.partial, 0, 0, "today twice (the compare's control)"));
        for (const auto& cf : cfg) {
            int ws[2] = { 0, 0 };
            const std::vector<uint8_t> got = walk(s, o, cf[0], cf[1], cf[2] != 0, ws);
            size_t bad = 0;
            for (size_t i = 0; i < ref.size(); ++i) bad += got[i] != ref[i];
            char tail[160];
            /* the re-load lists leave the cache only the partial last tiles (one a row whose context is not whole tiles) */
            int part = 0;
            for (int r = 0; r < s.S; ++r) part += s.used[r] % 16 != 0;
            std::snprintf(tail, sizeof tail, "walk%s = today (%zu of %zu partial bytes differ; tiles ring %d / cache %d)",
                          cf[2] ? " re-load" : "", bad, ref.size(), ws[0], ws[1]);
            check(bad == 0 && ws[0] > 0 && (cf[2] ? ws[1] == part : ws[1] > 0), fmt_case(c.S, c.P0, c.n, c.partial, cf[0], cf[1], tail));
        }
        /* negative controls */
        {
            std::vector<uint8_t> kv2 = o.kv;
            const int pg = s.table[(size_t)5 * s.W + 3];
            kv2[(size_t)pg * BS * ROWB + 1 * ROWB + 17] ^= 0x10;   /* a K code of row 5's 4th page, token 1 */
            int ws[2] = { 0, 0 };
            const std::vector<uint8_t> got = walk(s, o, 8, 15, true, ws, &kv2);
            check(got != ref, fmt_case(c.S, c.P0, c.n, c.partial, 8, 15, "NEGATIVE: one cache code changed -> partials differ"));
        }
        {
            int ws[2] = { 0, 0 };
            const std::vector<uint8_t> got = walk(s, o, 8, 15, true, ws, nullptr, 3);
            check(got != ref, fmt_case(c.S, c.P0, c.n, c.partial, 8, 15, "NEGATIVE: one union position off -> partials differ"));
        }
    }
    std::printf("r4d_emu_qwalk: %d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
