/* tests/r4d_emu: lever A3 / 1U-AR (ds41-ll, libr4d fn4 ar_ll) on the host SIMT emulator.
 *
 *   RULE    r4d_ar_ll.h's host rule: the switch's values and bounds, groups / units / the slot an instance sizes, the
 *           fit of each form, the grid, the tag (never 0), the pick at deepseek41's heights (rows 1 .. 48 of 5120 under
 *           ts_min 32768: the one-shot to 6 rows, the two-shot from 7; 49 rows and up, a non-bf16 instance, no area, a
 *           message past the element bound or a gather list past LDS: plain, "declined" where the switch covered it).
 *   PACK    a group's three words through r4d_arll_pack / r4d_arll_unpack give back every bit (random bits, NaN
 *           payloads, -0, subnormals, infinities), the tag in bits 48..63 of every unit.
 *   WIRE    the LL kernel body (r4d_ar_ll_dev.h, the code libr4d's r4d_ar_ll.hip launches) for four ranks on four LL
 *           areas, run phase by phase (push on every rank, then receive / fold [/ phase-2 push] on every rank, then the
 *           two-shot's phase-2 receive): every rank's output against the host fold of the four inputs (fp32 from 0.0f
 *           in ascending rank, rounded once -- ar_vsum's), both forms, in place and out of place, grids 1 .. 7 blocks
 *           of 32 .. 128 threads, at deepseek41's heights (1 .. 48 rows of 5120) and odd widths (a last group of one or
 *           two words); the gather form (moe_gather's words from the slot list) against the fold of each rank's
 *           partial formed by the one-shot gather's word over the inverse table (r4d_ar_gather_word<true>, test_argts
 *           holds it to moe_gather). After every launch every LL area is zero again (the readers' clears), so a run of
 *           launches of mixed sizes, forms and parities on the same areas is checked too; output guards untouched.
 *   CONTROLS  a unit with another launch's tag, or 0, is not taken (r4d_arll_try8); a data bit flipped in a pushed unit
 *           before the receive changes the fold (the comparison sees it); a unit left behind is seen by the zero check.
 *
 *   r4d_emu_arll */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

/* test-local: the f16 names r4d_ar_wide.h's overloads take, and its host extent check (unused here) */
struct __half { unsigned short x; };
inline float __half2float(__half h) {
    const unsigned s = (h.x >> 15) & 1u, e = (h.x >> 10) & 31u, m = h.x & 1023u;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 1024u), (int)e - 25);
    return s ? -v : v;
}
inline __half __float2half(float) { return __half{ 0 }; }
typedef void* hipDeviceptr_t;
inline hipError_t hipMemGetAddressRange(hipDeviceptr_t*, size_t*, hipDeviceptr_t) { return 1; }

#define R4D_ARLL_DEADLINE 200000ull   /* the emulator's clock ticks a read: a unit that never comes traps soon */
#include "r4d_ar_ll_dev.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_cases;
    if (!ok) ++g_fail;
    if (!ok || std::getenv("ARLL_VERBOSE"))
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

float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2b(float f) { __hip_bfloat16 b(f); return b.data; }   /* the emulator's (ROCm's) narrow, as the kernel's */

template <bool GATHER, bool TWO>
__global__ void ll_kernel(R4dArllArgs A, R4dArGather G) { r4d_arll_body<GATHER, TWO>(A, G); }

/* the one-shot gather's word over the inverse table: each rank's partial, independently of the list form */
__global__ void inv_word_kernel(R4dArGather G, const int* inv, int4* y4, int n16, int wpr) {
    for (int k = (int)(blockIdx.x * blockDim.x + threadIdx.x); k < n16; k += (int)(blockDim.x * gridDim.x)) {
        const int row = k / wpr, i0 = (k - row * wpr) * 8;
        y4[k] = r4d_ar_gather_word<true>(G, inv, nullptr, nullptr, nullptr, row, 0, i0);
    }
}

/* ---------------------------------------------------------------- RULE */
void rule() {
    check(r4d_arll_known(0) && r4d_arll_known(1) && r4d_arll_known(48) && !r4d_arll_known(-1) && !r4d_arll_known(49),
          "rule: ar_ll takes 0 .. 48 (-1 and 49 refused)");
    check(r4d_arll_bound(0) == 0 && r4d_arll_bound(1) == 48 && r4d_arll_bound(2) == 2 && r4d_arll_bound(48) == 48,
          "rule: 0 off, 1 = 48 rows, N = N rows");
    check(r4d_arll_groups(1) == 1 && r4d_arll_groups(3) == 1 && r4d_arll_groups(4) == 2 && r4d_arll_groups(640) == 214,
          "rule: three words a group, the last padded");
    check(r4d_arll_need(R4D_ARLL_ONESHOT, 640) == 8 * 214 && r4d_arll_need(R4D_ARLL_TWOSHOT, 640) == 16 * 54,
          "rule: units a slot -- one-shot 8 a group of the message, two-shot 16 a group of a shard");
    const long long su = r4d_arll_slot_units(21299200LL, 2);   /* deepseek41's all-reduce bound: 4160 rows of 5120 */
    check(su % 64 == 0 && su >= r4d_arll_need(R4D_ARLL_ONESHOT, 32768) && su >= r4d_arll_need(R4D_ARLL_TWOSHOT, 32768) &&
              su < r4d_arll_need(R4D_ARLL_ONESHOT, 32768) + 64,
          "rule: the slot at a 4160-row bound holds both forms at the element bound", fmt("%lld units", su));
    check(r4d_arll_slot_units(4, 2) == 0, "rule: no area under a word");
    const long long s4 = r4d_arll_slot_units(4 * 5120, 2);
    check(r4d_arll_fits(R4D_ARLL_ONESHOT, 4 * 5120, 2, s4) && r4d_arll_fits(R4D_ARLL_TWOSHOT, 4 * 5120, 2, s4) &&
              !r4d_arll_fits(R4D_ARLL_ONESHOT, 5 * 5120, 2, s4) && r4d_arll_fits(R4D_ARLL_TWOSHOT, 5 * 5120, 2, s4),
          "rule: an instance bounded at 4 rows fits both forms to 4 rows (the two-shot further: its regions are a shard)");
    check(!r4d_arll_fits(R4D_ARLL_TWOSHOT, 8 * 3, 2, 1 << 20) && !r4d_arll_fits(R4D_ARLL_ONESHOT, 262145, 2, 1 << 20) &&
              !r4d_arll_fits(R4D_ARLL_ONESHOT, 7, 2, 1 << 20) && r4d_arll_fits(R4D_ARLL_ONESHOT, 8, 2, 1 << 20),
          "rule: the two-shot needs whole words a rank; the element bound; whole 16-byte words");
    {
        int nb = 0, nt = 0;
        r4d_arll_grid(R4D_ARLL_ONESHOT, 640, 64, 0, 0, &nb, &nt);
        bool g = nb == 2 && nt == 128;
        r4d_arll_grid(R4D_ARLL_TWOSHOT, 8 * 640, 64, 0, 0, &nb, &nt);
        g = g && nb == 4 && nt == 128;
        r4d_arll_grid(R4D_ARLL_TWOSHOT, 48 * 640, 64, 0, 0, &nb, &nt);
        g = g && nb == 20;
        r4d_arll_grid(R4D_ARLL_ONESHOT, 32768, 64, 0, 0, &nb, &nt);
        g = g && nb == 32;
        r4d_arll_grid(R4D_ARLL_ONESHOT, 32768, 8, 0, 0, &nb, &nt);
        g = g && nb == 8;
        r4d_arll_grid(R4D_ARLL_ONESHOT, 640, 64, 5, 64, &nb, &nt);
        g = g && nb == 5 && nt == 64;
        r4d_arll_grid(R4D_ARLL_ONESHOT, 3, 64, 5, 64, &nb, &nt);
        g = g && nb == 1;
        check(g, "rule: grid -- a thread a group, 128 a block, 1 .. 32 blocks, the capacity, pins, never past the groups");
    }
    {
        bool t = true;
        for (unsigned l : { 0u, 1u, 65534u, 65535u, 65536u, 131070u, 0xffffffffu }) {
            const unsigned g = r4d_arll_tag16(l);
            t = t && g >= 1 && g <= 65535;
        }
        t = t && r4d_arll_tag16(1) != r4d_arll_tag16(2) && r4d_arll_tag16(1) != r4d_arll_tag16(3);
        check(t, "rule: the tag is 1 .. 65535 (never a cleared unit's 0), launches L and L + 2 differ");
    }
    {   /* deepseek41's heights under ts_min 32768 */
        int one = 0, two = 0, bad = 0;
        for (long long T = 1; T <= 64; ++T) {
            const long long numel = T * 5120;
            const int twoshot = numel >= 32768 && (numel * 2 / 16) % 4 == 0;
            bool dec = false;
            const int p = r4d_arll_pick(1, T, numel, 2, su, twoshot, 1, &dec);
            if (T <= 48) {
                if (p == 1 + R4D_ARLL_ONESHOT && T <= 6 && !dec) ++one;
                else if (p == 1 + R4D_ARLL_TWOSHOT && T >= 7 && !dec) ++two;
                else ++bad;
            } else if (p != 0 || dec) {
                ++bad;
            }
            bool d0 = true;
            if (r4d_arll_pick(0, T, numel, 2, su, twoshot, 1, &d0) != 0 || d0) ++bad;
            if (T <= 48) {
                bool d1 = false;
                if (r4d_arll_pick(8, T, numel, 2, su, twoshot, 1, &d1) != (T <= 8 ? p : 0) || d1) ++bad;
            }
        }
        check(one == 6 && two == 42 && bad == 0,
              "rule: ar_ll 1 at 1 .. 64 rows of 5120 -- one-shot 1 .. 6, two-shot 7 .. 48, plain from 49; ar_ll 0 plain; "
              "ar_ll 8 to 8 rows",
              fmt("one %d two %d bad %d", one, two, bad));
        bool d = false, ok = true;
        ok = ok && r4d_arll_pick(1, 8, 8 * 5120, 0, su, 1, 1, &d) == 0 && d;       /* not bf16 */
        ok = ok && r4d_arll_pick(1, 8, 8 * 5120, 2, 0, 1, 1, &d) == 0 && d;        /* no area */
        ok = ok && r4d_arll_pick(1, 8, 8 * 5120, 2, su, 1, 0, &d) == 0 && d;       /* gather list past LDS */
        ok = ok && r4d_arll_pick(1, 48, 48 * 8192, 2, su, 1, 1, &d) == 0 && d;     /* past the element bound */
        ok = ok && r4d_arll_pick(1, 1, 5120, 2, su, 0, 1, &d) == 1 + R4D_ARLL_ONESHOT && !d;
        check(ok, "rule: declined -- not bf16, no area, a gather list past LDS, past the element bound");
        check(r4d_arll_list_ok(48, 6) && r4d_arll_list_ok(256, 8) && !r4d_arll_list_ok(257, 1) && !r4d_arll_list_ok(48, 64),
              "rule: the gather's slot list bounds (256 rows, 2048 slots)");
    }
}

/* ---------------------------------------------------------------- PACK */
void pack(std::mt19937& rng) {
    int bad = 0, tagbad = 0;
    for (int it = 0; it < 4000; ++it) {
        int4 w[3], back[3];
        uint32_t d[12];
        for (int i = 0; i < 12; ++i) {
            d[i] = (uint32_t)rng();
            if (it % 7 == 0 && i % 3 == 0) d[i] = 0x7fc1ffffu;   /* NaN payloads */
            if (it % 11 == 0 && i % 4 == 1) d[i] = 0x80000001u;  /* -0 and a subnormal */
            if (it % 13 == 0 && i % 5 == 2) d[i] = 0xff807f80u;  /* -inf, +inf */
        }
        std::memcpy(w, d, 48);
        const unsigned tag = r4d_arll_tag16((unsigned)it * 2654435761u);
        unsigned long long u[8];
        r4d_arll_pack(w, tag, u);
        for (int j = 0; j < 8; ++j) tagbad += (unsigned)(u[j] >> 48) != tag;
        r4d_arll_unpack(u, back);
        bad += std::memcmp(w, back, 48) != 0;
    }
    check(bad == 0 && tagbad == 0, "pack: 4000 groups of three words through the units and back, bit for bit; every unit "
                                   "carries the tag", fmt("%d groups differ, %d units with another tag", bad, tagbad));
}

/* ---------------------------------------------------------------- WIRE */
struct Areas {
    long long su = 0;
    std::vector<std::vector<unsigned long long>> a;   /* 4 areas of 2 x 4 x su units */
    explicit Areas(long long slot_units) : su(slot_units), a(4, std::vector<unsigned long long>((size_t)(8 * slot_units), 0ull)) {}
    long long nonzero() const {
        long long z = 0;
        for (const auto& v : a) for (unsigned long long x : v) z += x != 0;
        return z;
    }
};

/* Gather operands for one rank, deepseek41's MoE shape. */
struct GatherRank {
    std::vector<uint16_t> ye, sh, sg;
    std::vector<int32_t> sorted, inv;
    R4dArGather G;
};
struct GatherShape {
    int M = 0, n = 5120, k = 6, E = 384;
    std::vector<float> ewf;
    GatherRank r[4];
};
void make_gather(GatherShape& s, std::mt19937& rng) {
    const int M = s.M, n = s.n, k = s.k, T = M * k, El = s.E / 4;
    std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
    std::vector<int32_t> ex((size_t)T, -1);
    for (int tok = 0; tok < M; ++tok)
        for (int j = 0; j < k; ++j) {
            if (rng() % 31 == 0) continue;
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                e = (int32_t)(rng() % (uint32_t)s.E);
                dup = false;
                for (int q = 0; q < j; ++q) dup = dup || ex[(size_t)(tok * k + q)] == e;
            }
            ex[(size_t)(tok * k + j)] = e;
        }
    s.ewf.resize((size_t)T);
    for (auto& w : s.ewf) w = (float)(rng() % 100000 + 1) / 40000.0f;
    for (int r = 0; r < 4; ++r) {
        GatherRank& g = s.r[r];
        g.sorted.assign((size_t)T, -1);
        g.inv.assign((size_t)T, -1);
        int pos = 0;
        for (int32_t e = r * El; e < (r + 1) * El; ++e)
            for (int i = 0; i < T; ++i)
                if (ex[(size_t)i] == e) g.sorted[(size_t)pos++] = i;
        for (int i = 0; i < pos; ++i) g.inv[(size_t)g.sorted[(size_t)i]] = i;
        g.ye.resize((size_t)T * n);
        for (auto& v : g.ye) v = f2b(ud(rng));
        g.sh.resize((size_t)M * n);
        g.sg.assign((size_t)M, f2b(1.0f));   /* deepseek41's ungated shared arm: a gate plane of ones, "none" */
        for (auto& v : g.sh) v = f2b(ud(rng) * 2.0f);
        std::memset(&g.G, 0, sizeof g.G);
        g.G.ye = g.ye.data();
        g.G.ew = reinterpret_cast<const unsigned short*>(s.ewf.data());
        g.G.sorted = g.sorted.data();
        g.G.sh = g.sh.data();
        g.G.sg = g.sg.data();
        g.G.ye_ld = n; g.G.ew_ld = k; g.G.sh_ld = n; g.G.sg_ld = 1;
        g.G.top_k = k; g.G.T = T; g.G.sig = 0; g.G.ew_f32 = 1;
    }
}

constexpr uint16_t kJunk = 0x7e57;

/* One LL all-reduce on four emulated ranks. in[r]: rank r's input (plain); out[r]: its output buffer (n16 * 8 bf16 + 8
 * guard words; for in place, out[r] is in[r]). flip: (rank, source, unit index) -- one bit of a pushed unit flipped
 * between the push and the receive (a control), -1 = none. */
template <bool GATHER, bool TWO>
void run_ll(Areas& ar, int n16, int nb, int nt, unsigned tag, std::vector<std::vector<uint16_t>>& in,
            std::vector<std::vector<uint16_t>>& out, bool in_place, const GatherShape* gs, int flip_rank = -1) {
    auto launch = [&](int r, int phases) {
        R4dArllArgs A;
        std::memset(&A, 0, sizeof A);
        A.in4 = GATHER ? nullptr : reinterpret_cast<const int4*>(in[(size_t)r].data());
        A.out4 = reinterpret_cast<int4*>(in_place && !GATHER ? in[(size_t)r].data() : out[(size_t)r].data());
        int p = 0;
        for (int q = 0; q < 4; ++q)
            if (q != r) A.peer[p++] = ar.a[(size_t)q].data();
        A.mine = ar.a[(size_t)r].data();
        A.slot_units = ar.su;
        A.n16 = n16;
        A.rank = r;
        A.wpr = gs ? gs->n / 8 : 640;
        A.test_tag = tag;
        A.phases = phases;
        R4dArGather G;
        std::memset(&G, 0, sizeof G);
        if (GATHER) G = gs->r[r].G;
        emu_emit("ll_kernel", &ll_kernel<GATHER, TWO>, dim3((unsigned)nb), dim3((unsigned)nt), 0, A, G);
    };
    for (int r = 0; r < 4; ++r) launch(r, 1);
    if (flip_rank >= 0) {   /* control: one data bit of a unit pushed into flip_rank's slot from its first peer */
        const int src = flip_rank == 0 ? 1 : 0;
        unsigned long long* slot = ar.a[(size_t)flip_rank].data() + ((long long)(tag & 1u) * 4 + src) * ar.su;
        slot[0] ^= 1ull << 14;   /* the top exponent bit of element 0 */
    }
    for (int r = 0; r < 4; ++r) launch(r, 2);
    if (TWO)
        for (int r = 0; r < 4; ++r) launch(r, 4);
}

/* The host fold of four bf16 vectors: fp32 from 0.0f in ascending rank, rounded once. */
std::vector<uint16_t> fold(const std::vector<std::vector<uint16_t>>& p, size_t n) {
    std::vector<uint16_t> o(n);
    for (size_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int r = 0; r < 4; ++r) acc += b2f(p[(size_t)r][i]);
        o[i] = f2b(acc);
    }
    return o;
}

uint16_t rnd_bf16(std::mt19937& rng, int kind) {
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    switch (kind) {
        case 0: return f2b(ud(rng));
        case 1: {   /* any bits but NaN: infinities, subnormals, -0 (two NaN inputs' payload depends on the add's operand
                     * order, which a compiler may swap; real activations carry none, and the card checks avoid them too) */
            uint16_t h = (uint16_t)rng();
            if ((h & 0x7f80u) == 0x7f80u && (h & 0x7fu)) h &= 0xff80u;
            return h;
        }
        default: { const uint32_t r = rng() % 8; return r == 0 ? 0x8000 : r == 1 ? 0x0001 : f2b(std::ldexp(ud(rng), (int)(rng() % 60) - 30)); }
    }
}

/* One plain launch checked; returns false on a difference. */
template <bool TWO>
bool plain_case(Areas& ar, int n16, int nb, int nt, unsigned tag, bool in_place, int kind, std::mt19937& rng,
                std::string* why, int flip = -1) {
    const size_t n = (size_t)n16 * 8;
    std::vector<std::vector<uint16_t>> in(4, std::vector<uint16_t>(n + 64)), out(4, std::vector<uint16_t>(n + 64, kJunk));
    for (auto& v : in) {
        for (size_t i = 0; i < n; ++i) v[i] = rnd_bf16(rng, kind);
        for (size_t i = n; i < n + 64; ++i) v[i] = kJunk;
    }
    const auto ref = fold(in, n);
    run_ll<false, TWO>(ar, n16, nb, nt, tag, in, out, in_place, nullptr, flip);
    long long diff = 0, first = -1, guard = 0;
    for (int r = 0; r < 4; ++r) {
        const auto& o = in_place ? in[(size_t)r] : out[(size_t)r];
        for (size_t i = 0; i < n; ++i)
            if (o[i] != ref[i]) { if (first < 0) first = (long long)i; ++diff; }
        for (size_t i = n; i < n + 64; ++i) guard += o[i] != kJunk;
    }
    const long long nz = ar.nonzero();
    if (flip >= 0) {   /* the control must differ, and leaves nothing behind */
        if (diff == 0) { *why = "the flipped unit did not change the fold"; return false; }
        return nz == 0;
    }
    if (diff || guard || nz) {
        *why = fmt("%lld element(s) differ (first %lld), %lld guard word(s) written, %lld unit(s) left in the areas", diff,
                   first, guard, nz);
        return false;
    }
    return true;
}

void wire_plain(std::mt19937& rng) {
    const long long su = r4d_arll_slot_units(64LL * 5120, 2);
    Areas ar(su);
    unsigned tag = 1;
    /* deepseek41's heights, both forms where they apply, every grid in the list, in place and out */
    struct G { int nb, nt; };
    const G grids[] = { { 0, 0 }, { 1, 32 }, { 3, 64 }, { 7, 128 } };
    int ran = 0, failed = 0;
    std::string first;
    for (int T : { 1, 2, 5, 6, 7, 8, 9, 12, 16, 24, 32, 47, 48 }) {
        const int n16 = T * 640;
        for (int form = 0; form < 2; ++form) {
            if (form == R4D_ARLL_TWOSHOT && n16 % 4) continue;
            for (const G& g : grids) {
                int nb = g.nb, nt = g.nt;
                if (nb == 0) r4d_arll_grid(form, n16, 64, 0, 0, &nb, &nt);
                for (int ip = 0; ip < 2; ++ip) {
                    std::string why;
                    const int kind = (T + ip) % 3;
                    const bool ok = form == R4D_ARLL_TWOSHOT ? plain_case<true>(ar, n16, nb, nt, tag++, ip, kind, rng, &why)
                                                             : plain_case<false>(ar, n16, nb, nt, tag++, ip, kind, rng, &why);
                    ++ran;
                    if (!ok) {
                        ++failed;
                        if (first.empty()) first = fmt("T %d %s grid %dx%d %s: %s", T, form ? "two-shot" : "one-shot", nb, nt,
                                                       ip ? "in place" : "out of place", why.c_str());
                    }
                }
            }
        }
    }
    check(failed == 0, fmt("wire: plain all-reduce, %d launches (T 1 .. 48 x 5120, both forms, grids rule / 1x32 / 3x64 / "
                           "7x128, in place + out of place, three value kinds) == the host fold on every rank, areas zero "
                           "after each", ran),
          first);
    /* odd widths: a last group of one or two words (n16 % 3 = 1, 2), shards likewise */
    failed = 0; ran = 0; first.clear();
    for (int n16 : { 1, 2, 4, 5, 8, 13, 20, 44, 100, 260, 644 }) {
        for (int form = 0; form < 2; ++form) {
            if (form == R4D_ARLL_TWOSHOT && (n16 % 4 || n16 < 4)) continue;
            int nb = 0, nt = 0;
            r4d_arll_grid(form, n16, 64, 0, 0, &nb, &nt);
            for (int pin : { 0, 2 }) {
                const int b = pin ? std::min(pin, (int)r4d_arll_groups(form ? n16 / 4 : n16)) : nb;
                std::string why;
                const bool ok = form ? plain_case<true>(ar, n16, b, 32, tag++, false, 1, rng, &why)
                                     : plain_case<false>(ar, n16, b, 32, tag++, false, 1, rng, &why);
                ++ran;
                if (!ok && first.empty()) first = fmt("n16 %d %s: %s", n16, form ? "two-shot" : "one-shot", why.c_str());
                failed += !ok;
            }
        }
    }
    check(failed == 0, fmt("wire: odd widths (a last group of 1 or 2 words), %d launches == the fold", ran), first);
    /* controls */
    {
        std::string why;
        const bool c1 = plain_case<false>(ar, 640, 2, 128, tag++, false, 0, rng, &why, 2);
        const bool c2 = plain_case<true>(ar, 8 * 640, 4, 128, tag++, false, 0, rng, &why, 1);
        check(c1 && c2, "control: a data bit flipped in a pushed unit changes the fold (one-shot, two-shot)", why);
    }
    {   /* a launch count past 65535 and the wrap of the tag: still the fold */
        std::string why;
        bool ok = true;
        for (unsigned t : { 65534u, 65535u, 65536u, 65537u, 0xfffffffdu, 0xfffffffeu })   /* t + 1 != 0: 0 takes the counters */
            ok = ok && plain_case<true>(ar, 7 * 640, 3, 64, t, false, 0, rng, &why) &&
                 plain_case<false>(ar, 3 * 640, 3, 64, t + 1, false, 0, rng, &why);
        check(ok, "wire: launch counts around the tag's wrap (65534 .. 65538, 2^32 - 3 .. 2^32 - 1) == the fold", why);
    }
}

void wire_gather(std::mt19937& rng) {
    const long long su = r4d_arll_slot_units(64LL * 5120, 2);
    Areas ar(su);
    unsigned tag = 7;
    int ran = 0, failed = 0;
    std::string first;
    for (int M : { 1, 2, 3, 6, 7, 8, 9, 16, 33, 48 }) {
        GatherShape gs;
        gs.M = M;
        make_gather(gs, rng);
        const int n16 = M * gs.n / 8;
        /* each rank's partial by the inverse-table word, then the fold */
        std::vector<std::vector<uint16_t>> part(4, std::vector<uint16_t>((size_t)n16 * 8));
        for (int r = 0; r < 4; ++r)
            emu_emit("inv_word_kernel", &inv_word_kernel, dim3((unsigned)((n16 + 255) / 256)), dim3(256), 0, gs.r[r].G,
                     (const int*)gs.r[r].inv.data(), reinterpret_cast<int4*>(part[(size_t)r].data()), n16, gs.n / 8);
        const auto ref = fold(part, (size_t)n16 * 8);
        for (int form = 0; form < 2; ++form) {
            if (form == R4D_ARLL_TWOSHOT && n16 % 4) continue;
            for (int pin : { 0, 3 }) {
                int nb = 0, nt = 0;
                r4d_arll_grid(form, n16, 64, pin, pin ? 64 : 0, &nb, &nt);
                std::vector<std::vector<uint16_t>> in, out(4, std::vector<uint16_t>((size_t)n16 * 8 + 64, kJunk));
                if (form) run_ll<true, true>(ar, n16, nb, nt, tag++, in, out, false, &gs);
                else run_ll<true, false>(ar, n16, nb, nt, tag++, in, out, false, &gs);
                long long diff = 0, guard = 0, fi = -1;
                for (int r = 0; r < 4; ++r) {
                    for (size_t i = 0; i < (size_t)n16 * 8; ++i)
                        if (out[(size_t)r][i] != ref[i]) { if (fi < 0) fi = (long long)i; ++diff; }
                    for (size_t i = (size_t)n16 * 8; i < (size_t)n16 * 8 + 64; ++i) guard += out[(size_t)r][i] != kJunk;
                }
                const long long nz = ar.nonzero();
                ++ran;
                if (diff || guard || nz) {
                    ++failed;
                    if (first.empty())
                        first = fmt("M %d %s grid %dx%d: %lld differ (first %lld), %lld guard, %lld units left", M,
                                    form ? "two-shot" : "one-shot", nb, nt, diff, fi, guard, nz);
                }
            }
        }
    }
    check(failed == 0, fmt("wire: moe_gather_all_reduce, %d launches (M 1 .. 48 x 5120, top_k 6 of 384, f32 weights, the "
                           "ungated shared arm; both forms, rule and pinned grids) == the fold of the inverse-table "
                           "partials on every rank, areas zero after each", ran),
          first);
}

/* A run of launches of mixed sizes, forms and parities on one set of areas (the clears keep them apart). */
void wire_mixed(std::mt19937& rng) {
    const long long su = r4d_arll_slot_units(48LL * 5120, 2);
    Areas ar(su);
    int failed = 0;
    std::string first;
    for (unsigned t = 1; t <= 60; ++t) {
        const int T = 1 + (int)(rng() % 48);
        const int n16 = T * 640;
        const bool two = T >= 7;
        int nb = 0, nt = 0;
        r4d_arll_grid(two, n16, 64, 0, 0, &nb, &nt);
        std::string why;
        const bool ok = two ? plain_case<true>(ar, n16, nb, nt, t, (t & 2) != 0, (int)(t % 3), rng, &why)
                            : plain_case<false>(ar, n16, nb, nt, t, (t & 2) != 0, (int)(t % 3), rng, &why);
        if (!ok && first.empty()) first = fmt("launch %u T %d: %s", t, T, why.c_str());
        failed += !ok;
    }
    check(failed == 0, "wire: 60 launches of random heights 1 .. 48 back to back on one set of areas (alternating parities) "
                       "== the fold", first);
}

void controls() {
    std::vector<unsigned long long> reg(8 * 4, 0ull);
    unsigned long long u[8] = {};
    int4 w[3];
    for (int q = 0; q < 3; ++q) { w[q].x = 1 + q; w[q].y = 2; w[q].z = 3; w[q].w = 4; }
    unsigned long long p[8];
    r4d_arll_pack(w, r4d_arll_tag16(5), p);
    for (int j = 0; j < 8; ++j) reg[(size_t)j * 4 + 1] = p[j];
    const unsigned m_ok = r4d_arll_try8(reg.data(), 4, 1, r4d_arll_tag16(5), 0xffu, u);
    const unsigned m_old = r4d_arll_try8(reg.data(), 4, 1, r4d_arll_tag16(7), 0xffu, u);
    const unsigned m_zero = r4d_arll_try8(reg.data(), 4, 0, r4d_arll_tag16(5), 0xffu, u);
    reg[(size_t)3 * 4 + 1] = 0;
    const unsigned m_one = r4d_arll_try8(reg.data(), 4, 1, r4d_arll_tag16(5), 0xffu, u);
    check(m_ok == 0 && m_old == 0xffu && m_zero == 0xffu && m_one == 0x08u,
          "control: try8 takes only units with the launch's tag (another launch's tag, a cleared 0, one missing unit)",
          fmt("masks %02x %02x %02x %02x", m_ok, m_old, m_zero, m_one));
}

}  // namespace

int main() {
    std::mt19937 rng(0xA3u);
    rule();
    pack(rng);
    controls();
    wire_plain(rng);
    wire_gather(rng);
    wire_mixed(rng);
    std::printf("r4d_emu_arll: %d checks, %d failure(s)\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
