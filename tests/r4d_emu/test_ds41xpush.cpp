/* tests/r4d_emu: XPUSH (ds41-int13, libr4d fn4 ar_xpush; ported from fn4-x3-sp3 808866c2) on the host SIMT emulator.
 *
 * The exact two-shot's first hop in the interleaved form (r4d_ar_xpush.h's r4d_ar_push1_xpush, what
 * r4d_ar_twoshot_wide_exact_body's push1 runs at R4D_AR8_FORM_XPUSH) against the lockstep loop it replaces (the body's
 * loop, copied here verbatim as the reference): over every block of a launch and every piece of a block, as the kernel
 * splits the shard space (w words a block, pw a piece), each peer's region A must hold exactly the words lockstep stores
 * -- the source shard's words [0, len(owner)) -- and nothing past them (junk guard words untouched), byte for byte.
 * The transport (peer pushes, flags, handshakes) needs four cards: r4d_selftest --case ar_twoshot_4rank_exact with
 * RADIANCE_FN4_AR_XPUSH 0 and 1 (the queue item's P0).
 *   WIDTHS   3, 4 and 8 ranks (every rank as the sender); even shards (r4d_ar_shard_table without weights: the
 *            four-rank exact rows' only split) and weighted ones (uneven lengths, empty shards, a single owner).
 *   GRIDS    1 .. 24 blocks, 64 and 256 threads, 1 .. 4 pieces (pieces past a block's words included: empty pieces).
 *   CONTROL  a lockstep run one word short must be caught (the comparison is not vacuous).
 *
 *   r4d_emu_ds41xpush */
#include <hip/hip_runtime.h>

#include "r4d_ar_shards.h"
#include "r4d_ar_xpush.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_cases;
    if (!ok) ++g_fail;
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
    std::fflush(stdout);
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

constexpr int kJunk = 0x7e57a5a5;

/* One launch's phase 1 as the body splits it: block b owns words [w0, w1) of the shard space, piece j of it
 * [w0 + j * pw, + pw). XP: r4d_ar_push1_xpush; else the body's lockstep loop (FORM == LOCKSTEP: p = i), copied.
 * short_by > 0 shortens lockstep's limit (the control). */
template <int NPEERS, bool XP>
__global__ void push1_kernel(const int4* in4, int4* const* pa, const int* peer_rank, const int* off, int sh_max,
                             int pieces, int short_by) {
    const int b = (int)blockIdx.x, nb = (int)gridDim.x, tid = (int)threadIdx.x, nt = (int)blockDim.x;
    const int w = (sh_max + nb - 1) / nb;
    const int w0 = b * w;
    int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
    const int pw = (w + pieces - 1) / pieces;
    for (int j = 0; j < pieces; ++j) {
        const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
        if (XP) {
            r4d_ar_push1_xpush<NPEERS>(in4, pa, peer_rank, off, p0, p1, tid, nt);
        } else {
            for (int i = 0; i < NPEERS; ++i) {
                const int p = i;
                const int owner = peer_rank[p];
                const int4* s1 = in4 + (size_t)off[owner];
                int4* dst = pa[p];
                const int lim = (p1 < off[owner + 1] - off[owner] ? p1 : off[owner + 1] - off[owner]) - short_by;
                for (int k = p0 + tid; k < lim; k += nt) dst[k] = s1[k];
            }
        }
    }
}

std::mt19937_64 g_rng(0x58505553ull);

struct Run {
    std::vector<std::vector<int4>> region;   /* a peer's region A (capacity sh_max + guard) */
};

/* One width / sender / split / grid: XPUSH vs lockstep vs the oracle. */
template <int WS>
bool one(const char* split, const int* weights, long n16, int rank, int nb, int nt, int pieces, int short_by,
         std::string* why) {
    constexpr int NPEERS = WS - 1;
    long offl[WS + 1];
    const long sh_max = r4d_ar_shard_table(n16, WS, weights, offl);
    int off[WS + 1];
    for (int s = 0; s <= WS; ++s) off[s] = (int)offl[s];
    int peer_rank[NPEERS];
    for (int i = 0; i < NPEERS; ++i) peer_rank[i] = i < rank ? i : i + 1;
    std::vector<int4> in((size_t)n16);
    for (auto& v : in) v = int4{ (int)g_rng(), (int)g_rng(), (int)g_rng(), (int)g_rng() };
    if (nb > sh_max && sh_max > 0) nb = (int)sh_max;   /* the launcher's clamp */
    if (nb < 1) nb = 1;
    const size_t cap = (size_t)sh_max + 37;
    auto launch = [&](bool xp, int sb, std::vector<std::vector<int4>>& reg) {
        reg.assign(NPEERS, std::vector<int4>(cap, int4{ kJunk, kJunk, kJunk, kJunk }));
        int4* pa[NPEERS];
        for (int i = 0; i < NPEERS; ++i) pa[i] = reg[(size_t)i].data();
        if (xp)
            R4D_EMIT(nullptr, (push1_kernel<NPEERS, true>), dim3((unsigned)nb), dim3((unsigned)nt), 0,
                     (const int4*)in.data(), (int4* const*)pa, (const int*)peer_rank, (const int*)off, (int)sh_max,
                     pieces, sb);
        else
            R4D_EMIT(nullptr, (push1_kernel<NPEERS, false>), dim3((unsigned)nb), dim3((unsigned)nt), 0,
                     (const int4*)in.data(), (int4* const*)pa, (const int*)peer_rank, (const int*)off, (int)sh_max,
                     pieces, sb);
    };
    std::vector<std::vector<int4>> xp, ls;
    launch(true, 0, xp);
    launch(false, short_by, ls);
    auto same = [](const int4& a, const int4& b) { return std::memcmp(&a, &b, sizeof a) == 0; };
    long diff_ls = 0, bad_oracle = 0, first = -1;
    for (int i = 0; i < NPEERS; ++i) {
        const int owner = peer_rank[i];
        const long len = off[owner + 1] - off[owner];
        for (size_t k = 0; k < cap; ++k) {
            const int4 want = (long)k < len ? in[(size_t)off[owner] + k] : int4{ kJunk, kJunk, kJunk, kJunk };
            if (!same(xp[(size_t)i][k], ls[(size_t)i][k])) { ++diff_ls; if (first < 0) first = (long)k; }
            if (!same(xp[(size_t)i][k], want)) ++bad_oracle;
        }
    }
    *why = fmt("%s, %d ranks, sender %d, %ld words (shard max %ld), %d x %d, pieces %d: %ld words differ from lockstep "
               "(first at %ld), %ld from the oracle", split, WS, rank, n16, sh_max, nb, nt, pieces, diff_ls, first,
               bad_oracle);
    return diff_ls == 0 && bad_oracle == 0;
}

template <int WS>
void width(const char* split, const int* weights, const std::vector<long>& sizes) {
    int n = 0, ok = 0;
    std::string last_bad, d;
    for (long n16 : sizes)
        for (int rank = 0; rank < WS; ++rank)
            for (int nb : { 1, 3, 8, 24 })
                for (int nt : { 64, 256 })
                    for (int pieces : { 1, 2, 4 }) {
                        if (nb * nt > 4096 && n16 < 64) continue;
                        ++n;
                        if (one<WS>(split, weights, n16, rank, nb, nt, pieces, 0, &d)) ++ok;
                        else if (last_bad.empty()) last_bad = d;
                    }
    check(ok == n, fmt("%d ranks, %s: XPUSH's first hop == lockstep's == the source shards, byte for byte, guards "
                       "untouched (%d / %d launches)", WS, split, ok, n), last_bad);
}

}  // namespace

int main() {
    /* the four-rank exact rows' messages: whole words dividing by 4 -- 2 rows of 5120 bf16 (1280 words), 26 rows
     * (16640: the XPUSH floor's first DeepSeek-V4.1 height), a small 4-word message, 2048 x 5120 / 64 (an uneven
     * block split), and odd counts the even split hands out unevenly at 3 / 8 ranks */
    const std::vector<long> sizes4 = { 4, 1280, 16640, 20480, 4100 };
    width<4>("even", nullptr, sizes4);
    width<3>("even", nullptr, { 1, 2, 7, 1281, 16641 });
    width<8>("even", nullptr, { 8, 13, 1287, 16640 });
    static const int w8a[8] = { 0, 3, 1, 0, 5, 1, 2, 4 }, w8b[8] = { 1, 0, 0, 0, 0, 0, 0, 0 },
                     w4a[4] = { 4, 1, 0, 4 }, w3a[3] = { 2, 0, 5 };
    width<8>("weights 0,3,1,0,5,1,2,4", w8a, { 72, 1287, 8256 });
    width<8>("weights 1,0,0,0,0,0,0,0 (a single owner)", w8b, { 9, 1280 });
    width<4>("weights 4,1,0,4", w4a, { 5, 1281, 16641 });
    width<3>("weights 2,0,5", w3a, { 3, 1283 });
    {   /* the control: lockstep one word short of each shard's piece end must differ */
        std::string d;
        const bool same = one<4>("even", nullptr, 1280, 1, 3, 256, 2, 1, &d);
        check(!same, "control: a lockstep run one word short is caught (the comparison sees a difference)", d);
    }
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
