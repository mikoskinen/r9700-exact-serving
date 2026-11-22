/* tests/r4d_emu: lever ARC (ds41-int10, libr4d fn4 ds41_arcoded) on the host SIMT emulator.
 *
 *   RULE   r4d_ar_arc.h's pick, host code: the switch's values, the operand's rows, the threshold (plain below it and at
 *          every decode height, the coded wire from it), the grid by the message against the ARW table (12.11.2026: C8 best
 *          at T 1100, C16 at T 1500 / 2048, C24 at T 4096, rows of 5120 bf16), the instance's block capacity, a declined
 *          coded form or a `coded` 0 parameter.
 *   CODEC  the coded wire's codec itself (r4d_ar_coded.h, the device code r4d_ar_twoshot_coded_body runs), byte for byte:
 *          a wave's chunks through r4d_arc_put / r4d_arc_get give back every input word, NaN payloads, -0, subnormals and
 *          infinities included, on four patterns (activation-like rows of 5120, the same with exact zeros and -0, the
 *          coded form's edge groups -- an exponent range of exactly 14 (coded) and 15 (raw), a subnormal, +-inf, NaNs, zero
 *          groups, the top and bottom normal exponents --, and random bits); the region's bytes stay inside
 *          r4d_arc_region_bytes (guard bytes past it untouched).
 *   FOLD   the coded two-shot's data path for one shard at four ranks, the transport aside (four cards: r4d_selftest
 *          --case ar_arcoded): the three peers' inputs coded into regions A, the owner's reduce (fp32, ascending rank from
 *          0.0f, the owner's own input raw), the result coded into region B and decoded -- against the plain wire's fold of
 *          the raw inputs (the same loop without the codec) and a host fold, every owner, byte for byte (the owner's own
 *          words and the peers' decoded ones).
 *   MUTATIONS (the comparisons see a difference): one bit of a coded group's packed word, a coded group's base byte, a raw
 *          group's shadow word -- each changed between the coding and the reduce -- must make the coded fold differ.
 *
 *   r4d_emu_ds41arc */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

/* the emulator's runtime has int4 but not HIP's make_int4, which the codec calls */
inline int4 make_int4(int a, int b, int c, int d) { return int4{ a, b, c, d }; }

#include "r4d_ar_coded.h"
#include "r4d_ar_arc.h"

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
/* the emulator's __hip_bfloat16(float): round to nearest even, a NaN kept quiet */
uint16_t f2b(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* ------------------------------------------------------------------------------------------------ RULE */
void rule() {
    const long long D = 5120;
    bool ok = !r4d_arc_known(-1) && r4d_arc_known(0) && r4d_arc_known(1) && r4d_arc_known(R4D_ARC_MAX) &&
              !r4d_arc_known((long long)R4D_ARC_MAX + 1);
    check(ok, "rule: ds41_arcoded takes 0 .. 1048576 (-1 and 1048577 refused)");
    ok = r4d_arc_rows(1100 * D, 2, D) == 1100 && r4d_arc_rows(4160 * D, 2, D) == 4160 && r4d_arc_rows(8 * D, 2, D) == 8 &&
         r4d_arc_rows(1100 * D, 1, 1100 * D) == 1 && r4d_arc_rows(1100 * D, 3, D) == 1100 && r4d_arc_rows(5, 2, 0) == 1;
    check(ok, "rule: rows = elements over the last axis ([T, 5120] -> T; a rank-1 operand is one row)");
    /* switch off: the plain wire at every height */
    int coded = 0;
    for (long long T = 1; T <= 4160; ++T) coded += r4d_arc_pick(0, T, T * D, 1, 1, 64) != 0;
    check(coded == 0, "rule: ds41_arcoded 0 -> the plain wire at every height 1 .. 4160", fmt("%d coded", coded));
    /* the default threshold: plain below 1024 rows (every decode / verify height), coded from it */
    int below = 0, from = 0;
    for (long long T = 1; T <= 4160; ++T) {
        const int nb = r4d_arc_pick(R4D_ARC_DEFAULT, T, T * D, 1, 1, 64);
        if (T < 1024) below += nb != 0;
        else from += nb == 0;
    }
    check(below == 0 && from == 0, "rule: ds41_arcoded 1024 -> plain at 1 .. 1023 rows (decode 8 / 16 / 48 included), coded "
                                   "at 1024 .. 4160", fmt("%d coded below, %d plain from 1024", below, from));
    /* the grid against the ARW table's best variant at each measured height, and the band edges */
    struct G { long long T; int nb; const char* why; };
    const G g[] = { { 1024, 8, "threshold" }, { 1100, 8, "ARW C8 -17.0 %" }, { 1299, 8, "edge" }, { 1300, 16, "edge" },
                    { 1500, 16, "ARW C16 -17.2 %" }, { 2048, 16, "ARW C16 -16.2 %" }, { 3071, 16, "edge" },
                    { 3072, 24, "edge" }, { 4096, 24, "ARW C24 -15.0 %" }, { 4160, 24, "max-num-batched-tokens" } };
    std::string bad;
    for (const G& x : g) {
        const int nb = r4d_arc_pick(R4D_ARC_DEFAULT, x.T, x.T * D, 1, 1, 64);
        if (nb != x.nb) bad += fmt(" T %lld -> %d (want %d, %s)", x.T, nb, x.nb, x.why);
    }
    check(bad.empty(), "rule: the grid -- 8 blocks below 1300 rows of 5120, 16 below 3072, 24 from there (ARW's best "
                       "variant at T 1100 / 1500 / 2048 / 4096)", bad);
    /* the capacity, a declined form, `coded` 0, a threshold above the step */
    ok = r4d_arc_pick(1024, 4096, 4096 * D, 1, 1, 16) == 16 && r4d_arc_pick(1024, 4096, 4096 * D, 1, 1, 4) == 4 &&
         r4d_arc_pick(1024, 1100, 1100 * D, 1, 1, 4) == 4 && r4d_arc_pick(1024, 4096, 4096 * D, 1, 1, 0) == 0;
    check(ok, "rule: never past the instance's block capacity (R4D_AR_MAX_BLOCKS 16 / 4)");
    ok = r4d_arc_pick(1024, 4096, 4096 * D, 0, 1, 64) == 0 && r4d_arc_pick(1024, 4096, 4096 * D, 1, 0, 64) == 0 &&
         r4d_arc_pick(4161, 4160, 4160 * D, 1, 1, 64) == 0 && r4d_arc_pick(1, 1, D, 1, 1, 64) == 8;
    check(ok, "rule: a declined coded form or `coded` 0 -> plain; a threshold above the step -> plain; 1 -> every launch");
}

/* ------------------------------------------------------------------------------------------------ the kernels */
/* chunks [0, n16) of `in` into one region (SL = n16), a wave's 32 chunks a step, as the body's phase 1 */
__global__ void put_kernel(const int4* in, char* region, int n16) {
    const int tid = (int)threadIdx.x, nt = (int)blockDim.x, lane = tid & 31;
    R4dArcRegion dst[1] = { r4d_arc_region(region, n16) };
    for (int k = tid; k - lane < n16; k += nt) {
        const bool valid = k < n16;
        const int4 v = valid ? in[k] : make_int4(0, 0, 0, 0);
        r4d_arc_put<1>(dst, k - lane, r4d_arc_enc(v, valid), v, valid);
    }
}
/* chunks back from a region, as the body's assemble */
__global__ void get_kernel(const char* region, int4* out, int n16) {
    const int tid = (int)threadIdx.x, nt = (int)blockDim.x, lane = tid & 31;
    const R4dArcRegion src = r4d_arc_region(region, n16);
    for (int k = tid; k - lane < n16; k += nt) {
        const bool valid = k < n16;
        const int4 v = r4d_arc_get(src, k - lane, valid);
        if (valid) out[k] = v;
    }
}
/* the owner's reduce: coded (peers from their regions A, the owner raw), result into its own out and region B -- the
 * body's middle phase -- and the plain wire's fold of the raw inputs beside it */
__global__ void reduce_kernel(const int4* in0, const int4* in1, const int4* in2, const int4* in3, const char* a0,
                              const char* a1, const char* a2, const char* a3, int owner, int4* own_out, char* regb,
                              int4* plain_out, int n16) {
    const int tid = (int)threadIdx.x, nt = (int)blockDim.x, lane = tid & 31;
    const int4* in[4] = { in0, in1, in2, in3 };
    R4dArcRegion srcA[4] = { r4d_arc_region(a0, n16), r4d_arc_region(a1, n16), r4d_arc_region(a2, n16),
                             r4d_arc_region(a3, n16) };
    R4dArcRegion dstB[1] = { r4d_arc_region(regb, n16) };
    for (int k = tid; k - lane < n16; k += nt) {
        const bool valid = k < n16;
        const int k0 = k - lane;
        float acc[8], pacc[8];
        for (int j = 0; j < 8; ++j) acc[j] = pacc[j] = 0.0f;
        for (int r = 0; r < 4; ++r) {
            const int4 v = (r == owner) ? (valid ? in[r][k] : make_int4(0, 0, 0, 0)) : r4d_arc_get(srcA[r], k0, valid);
            const int4 p = valid ? in[r][k] : make_int4(0, 0, 0, 0);
            const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
            const __hip_bfloat16* pp = reinterpret_cast<const __hip_bfloat16*>(&p);
            for (int j = 0; j < 8; ++j) { acc[j] += (float)pv[j]; pacc[j] += (float)pp[j]; }
        }
        int4 o, q;
        __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
        __hip_bfloat16* pq = reinterpret_cast<__hip_bfloat16*>(&q);
        for (int j = 0; j < 8; ++j) { po[j] = (__hip_bfloat16)acc[j]; pq[j] = (__hip_bfloat16)pacc[j]; }
        if (valid) { own_out[k] = o; plain_out[k] = q; }
        r4d_arc_put<1>(dstB, k0, r4d_arc_enc(o, valid), o, valid);
    }
}

/* ------------------------------------------------------------------------------------------------ data */
std::mt19937_64 g_rng(0xA2C0DEDull);
double gauss() {
    std::normal_distribution<double> n(0.0, 1.0);
    return n(g_rng);
}
/* one rank's message of n values */
std::vector<uint16_t> pattern(const char* kind, int rank, size_t n) {
    std::vector<uint16_t> v(n);
    if (!std::strcmp(kind, "act") || !std::strcmp(kind, "actz")) {
        const bool z = kind[3] == 'z';
        std::uniform_real_distribution<double> u(-4.0, 4.0), p(0.0, 1.0);
        double scale = 1.0;
        for (size_t i = 0; i < n; ++i) {
            if (i % 5120 == 0) scale = std::ldexp(1.0, (int)std::floor(u(g_rng)));   /* a row's scale */
            float x = (float)(gauss() * scale * (1.0 + 0.25 * rank));
            if (z) { const double q = p(g_rng); if (q < 0.25) x = 0.0f; else if (q < 0.30) x = -0.0f; }
            v[i] = f2b(x);
        }
        return v;
    }
    if (!std::strcmp(kind, "bin")) {
        for (size_t i = 0; i < n; ++i) v[i] = (uint16_t)(g_rng() & 0xffffu);
        return v;
    }
    /* edge: the group kind by the group (64 values) index */
    for (size_t i = 0; i < n; ++i) {
        const size_t g = i / 64, j = i % 64;
        const uint16_t sg = (j & 1) ? 0x8000u : 0u;
        uint16_t w = (uint16_t)(sg | ((uint16_t)(120 + (g_rng() % 8)) << 7) | (uint16_t)(g_rng() & 0x7f));   /* near 1 */
        switch (g % 14) {
        case 1: w = (uint16_t)(sg | ((uint16_t)(100 + (j % 15)) << 7) | (j & 0x7f)); break;    /* range exactly 14: coded */
        case 2: w = (uint16_t)(sg | ((uint16_t)(100 + (j % 16)) << 7) | (j & 0x7f)); break;    /* range 15: raw */
        case 3: if (j == 17) w = (uint16_t)(0x0005u | sg); break;                               /* a subnormal: raw */
        case 4: if (j == 40) w = rank == 0 ? 0x7f80u : 0xff80u; break;                         /* +-inf: raw */
        case 5: if (j == 9) w = (uint16_t)(rank == 1 ? 0x7fc1u : rank == 2 ? 0xffa3u : 0x7f81u); break;   /* NaNs: raw */
        case 6: w = 0x0000u; break;                                                             /* all zeros: coded */
        case 7: w = (j % 3) ? 0x8000u : 0x0000u; break;                                         /* +-0: coded */
        case 8: w = (uint16_t)(sg | (254u << 7) | (g_rng() & 0x7f)); if (j % 4 == 0) w = (uint16_t)(sg | (241u << 7)); break;   /* top: coded */
        case 9: w = (uint16_t)(sg | (1u << 7) | (g_rng() & 0x7f)); if (j % 5 == 0) w = (uint16_t)(sg | (14u << 7)); break;     /* bottom: coded */
        case 10: if (j % 2) w = (uint16_t)(sg | 0u); break;                                     /* half zeros: coded */
        case 11: w = (uint16_t)(sg | (127u << 7) | (j & 0x7f)); if (j == 63) w = (uint16_t)(141u << 7); break;   /* range 14 at the top */
        case 12: w = (uint16_t)(sg | (127u << 7) | (j & 0x7f)); if (j == 63) w = (uint16_t)(142u << 7); break;   /* range 15 at the top */
        default: break;
        }
        v[i] = w;
    }
    return v;
}

/* a region buffer with guard bytes past r4d_arc_region_bytes */
struct Region {
    std::vector<unsigned char> b;
    size_t bytes = 0;
    explicit Region(int n16) {
        bytes = r4d_arc_region_bytes(n16);
        b.assign(bytes + 512, 0xA5);
        /* 256-aligned start inside the buffer */
        off = (256 - ((uintptr_t)b.data() & 255)) & 255;
    }
    size_t off = 0;
    char* p() { return reinterpret_cast<char*>(b.data() + off); }
    bool guard_ok() const {
        for (size_t i = off + bytes; i < b.size(); ++i) if (b[i] != 0xA5) return false;
        for (size_t i = 0; i < off; ++i) if (b[i] != 0xA5) return false;
        return true;
    }
};
long first_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) if (a[i] != b[i]) return (long)i;
    return a.size() == b.size() ? -1 : (long)std::min(a.size(), b.size());
}

/* ------------------------------------------------------------------------------------------------ CODEC */
void codec(const char* kind, int n16, int threads) {
    const size_t n = (size_t)n16 * 8;
    std::vector<uint16_t> in = pattern(kind, 1, n), out(n, 0x7e57);
    Region rg(n16);
    R4D_EMIT(nullptr, put_kernel, dim3(1), dim3((unsigned)threads), 0, reinterpret_cast<const int4*>(in.data()), rg.p(), n16);
    R4D_EMIT(nullptr, get_kernel, dim3(1), dim3((unsigned)threads), 0, (const char*)rg.p(), reinterpret_cast<int4*>(out.data()), n16);
    /* groups sent raw / coded (the base bytes: a byte a group, 0xFF raw) */
    long raw = 0, cod = 0;
    const unsigned char* bases = reinterpret_cast<const unsigned char*>(rg.p()) + (size_t)12 * n16;
    for (int g = 0; g < (n16 + 7) / 8; ++g) (bases[g] == 0xFF ? raw : cod)++;
    const long d = first_diff(in, out);
    check(d < 0 && rg.guard_ok(), fmt("codec %s: %d chunks (%zu values) through put / get at %d threads == the input, bit for "
                                      "bit; region inside r4d_arc_region_bytes", kind, n16, n, threads),
          fmt("%ld coded groups, %ld raw%s%s", cod, raw, d >= 0 ? fmt(", first diff at %ld: %04x vs %04x", d, in[(size_t)d], out[(size_t)d]).c_str() : "",
              rg.guard_ok() ? "" : ", GUARD BYTES CHANGED"));
    if (!std::strcmp(kind, "edge")) {
        /* the 14-binade rule on the edge groups: kind g % 14 (one rank's own values) */
        std::string bad;
        for (int g = 0; g < (n16 + 7) / 8 && g < 14 * 4; ++g) {
            const int k = g % 14;
            const bool want_raw = k == 2 || k == 3 || k == 5 || k == 12 || (k == 4);
            /* kind 4 holds an inf on every rank (0x7f80 / 0xff80) */
            if ((bases[g] == 0xFF) != want_raw) bad += fmt(" group %d (kind %d) %s", g, k, bases[g] == 0xFF ? "raw" : "coded");
        }
        check(bad.empty(), "codec edge: range 14 coded, range 15 raw (low and top), a subnormal / an inf / a NaN raw, zero "
                           "and +-0 groups, the top and bottom normal exponents coded", bad);
    }
}

/* ------------------------------------------------------------------------------------------------ FOLD */
/* mut: 0 none, 1 a packed word's bit in a coded group of a peer's region A, 2 a coded group's base byte, 3 a raw group's
 * shadow word (each in rank `mr`'s region, between the coding and the reduce) */
bool fold(const char* kind, int n16, int threads, int owner, int mut, std::string* detail) {
    const size_t n = (size_t)n16 * 8;
    std::vector<std::vector<uint16_t>> in(4);
    for (int r = 0; r < 4; ++r) in[(size_t)r] = pattern(kind, r, n);
    std::vector<Region> A;
    A.reserve(4);
    for (int r = 0; r < 4; ++r) A.emplace_back(n16);
    Region B(n16);
    for (int r = 0; r < 4; ++r)
        if (r != owner)
            R4D_EMIT(nullptr, put_kernel, dim3(1), dim3((unsigned)threads), 0, reinterpret_cast<const int4*>(in[(size_t)r].data()),
                     A[(size_t)r].p(), n16);
    if (mut) {
        const int mr = (owner + 1) % 4;
        char* p = A[(size_t)mr].p();
        unsigned char* bases = reinterpret_cast<unsigned char*>(p) + (size_t)12 * n16;
        int g = -1;
        for (int i = 0; i < (n16 + 7) / 8 && g < 0; ++i) if ((bases[i] == 0xFF) == (mut == 3)) g = i;
        if (g < 0) { *detail = "no group of the kind the mutation needs"; return false; }
        if (mut == 1) reinterpret_cast<unsigned*>(p)[(size_t)g * 8 * 3 + 1] ^= 0x10u;   /* chunk g*8's dword 1: a mantissa bit */
        if (mut == 2) bases[g] = (unsigned char)(bases[g] + 1);
        if (mut == 3) reinterpret_cast<unsigned*>(p + r4d_arc_shadow_off(n16))[(size_t)g * 8 * 4] ^= 0x00400000u;   /* the second value's top mantissa bit */
    }
    std::vector<uint16_t> own(n, 0x7e57), plain(n, 0x7e57), peer(n, 0x7e57);
    R4D_EMIT(nullptr, reduce_kernel, dim3(1), dim3((unsigned)threads), 0,
             reinterpret_cast<const int4*>(in[0].data()), reinterpret_cast<const int4*>(in[1].data()),
             reinterpret_cast<const int4*>(in[2].data()), reinterpret_cast<const int4*>(in[3].data()),
             (const char*)A[0].p(), (const char*)A[1].p(), (const char*)A[2].p(), (const char*)A[3].p(), owner,
             reinterpret_cast<int4*>(own.data()), B.p(), reinterpret_cast<int4*>(plain.data()), n16);
    R4D_EMIT(nullptr, get_kernel, dim3(1), dim3((unsigned)threads), 0, (const char*)B.p(), reinterpret_cast<int4*>(peer.data()), n16);
    /* the host fold: fp32 from 0.0f in ascending rank, rounded once */
    std::vector<uint16_t> host(n);
    for (size_t i = 0; i < n; ++i) {
        float a = 0.0f;
        for (int r = 0; r < 4; ++r) a += b2f(in[(size_t)r][i]);
        host[i] = f2b(a);
    }
    bool guards = B.guard_ok();
    for (auto& r : A) guards = guards && r.guard_ok();
    const long d1 = first_diff(own, plain), d2 = first_diff(peer, plain);
    /* the host fold's NaN payload / sign is the host's (x86 keeps an operand's, in whichever order the compiler put the
     * adds), so a NaN result matches any NaN there; everything else bit for bit. The kernels' two folds above: strict. */
    long d3 = -1, nans = 0;
    for (size_t i = 0; i < n && d3 < 0; ++i) {
        const bool pn = (plain[i] & 0x7fffu) > 0x7f80u, hn = (host[i] & 0x7fffu) > 0x7f80u;
        nans += pn;
        if (pn != hn || (!pn && plain[i] != host[i])) d3 = (long)i;
    }
    *detail = fmt("owner's words vs plain first diff %ld, peers' decoded vs plain %ld, plain vs host fold %ld (%ld NaN results)%s",
                  d1, d2, d3, nans, guards ? "" : ", GUARD BYTES CHANGED");
    return d1 < 0 && d2 < 0 && d3 < 0 && guards;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    rule();
    /* 4 rows of 5120 (2560 chunks: 80 waves), 1056 chunks (33 waves: a partial last pass at 256 threads), 1000 chunks
     * (not whole waves: lanes past the end, which the coded form's launcher never sends but the codec takes) */
    for (const char* k : { "act", "actz", "edge", "bin" }) {
        codec(k, 2560, 256);
        codec(k, 1056, 256);
    }
    codec("edge", 1000, 128);
    codec("act", 2560, 1024);
    for (const char* k : { "act", "actz", "edge", "bin" })
        for (int owner = 0; owner < 4; ++owner) {
            std::string d;
            const bool ok = fold(k, 2560, 256, owner, 0, &d);
            check(ok, fmt("fold %s, owner rank %d, 2560 chunks: coded two-shot data path == the plain fold == the host fold, "
                          "byte for byte", k, owner), d);
        }
    {
        std::string d;
        check(fold("act", 1056, 1024, 2, 0, &d), "fold act, owner 2, 1056 chunks at 1024 threads", d);
    }
    for (int m = 1; m <= 3; ++m) {
        std::string d;
        const bool same = fold("edge", 1056, 256, 1, m, &d);
        check(!same, fmt("mutation %d (%s) caught: the coded fold differs", m,
                         m == 1 ? "a packed word's mantissa bit in a coded group" : m == 2 ? "a coded group's base byte"
                                                                                          : "a raw group's shadow word"), d);
    }
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
