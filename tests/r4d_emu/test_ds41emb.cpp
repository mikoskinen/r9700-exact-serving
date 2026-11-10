/* tests/r4d_emu: lever EAR (deepseek41's RADIANCE_DS41_EMB_REP / switch ds41_embrep) on the host SIMT emulator, byte for
 * byte.
 *
 * Today's embedding at TP 4 (deepseek41.cpp step() and dsp_query()): each rank looks the step's tokens up in its vocab
 * shard of the table (embed_lookup_ggml, vocab_offset = the shard's first row; gg_embed_kernel writes a zero row, +0.0,
 * for every id outside it), an exact all-reduce folds the four ranks' bf16 rows in fp32 from 0.0f in ascending rank
 * order and rounds once (ar_vsum, r4d_ar_oneshot_Nrank_exact.hip; the two-shot's reduce2 and the coded two-shot's
 * reduce, r4d_ar_twoshot_Nrank_exact.hip, fold the same way; r4d_ar_wide.h's r4d_ar_to_f / r4d_ar_from_f convert), and
 * each of the hc = 4 stream copies is a byte copy of it (cast_bf16_bf16, a 2-D copy). The lever: every rank looks the
 * tokens up in the WHOLE table (vocab_offset 0), fills a row of +0.0 (fill), and writes each hc copy as add(row, zero
 * row) (binary_kernel, r4d_model_bf16.hip, b broadcast down the rows: fp32 u + v rounded by r4d_common.h's
 * f32_to_bf16). Both give the owner's value v, and +0.0 where v is -0.0, which a q6_K row holds wherever q = 32 under
 * a negative d * sc (Gg<GG_Q6_K>::dec: ds * (float)(q - 32)) -- so the lookup alone would NOT be the all-reduce's
 * bytes, and the add is what makes it so.
 *
 * Checked here:
 *   (0) every finite, non-subnormal bf16 value and both zeros, at each of the four owner ranks: the all-reduce's fold of
 *       (v at the owner, +0 elsewhere) == add(v, +0) == v with -0 made +0. (Subnormals and NaN / infinity are left
 *       out: the 2.25 model's table holds none -- (3) checks every block -- and a device may flush subnormals.)
 *   (1) gg_embed_kernel itself (r4d_ds41_ggml_mr.hip's copy of r4d_ds41_ggml.hip's source, r4d_gg_embed_emu) for both
 *       lookups; the hc stream plane [M, 4 * E] after the shard path (4 lookups, the fold, 4 copies) == after the
 *       whole-table path (1 lookup, the zero row, 4 adds), every byte, with guard words past it; the owner's shard row
 *       == the whole-table row before the add, the other shards' rows +0.0; how many elements the lookup alone would
 *       get wrong (its -0.0) is counted, and the case fails if no -0.0 was produced at all.
 *       Synthetic tables: q6_K (n_embd 5120, the 2.25 table's type), q5_K and q8_0; random block bytes with finite
 *       scales of both signs, zero and subnormal scales, many q6_K codes 32; ids over every shard edge, padding (-1),
 *       ids past the table; M = the trunk's decode / verify / prefill heights (1 .. 4160, the --max-num-batched-tokens
 *       the items run; 4160 rows also crosses the launch's 4096-block grid) and the DSpark query pass's (block 6 x 1..8
 *       sequences). The trunk and the drafter issue the same three ops on the same table, differing only in M and in
 *       the buffers (xcol / h0 vs dspark_collapsed / the drafter's h0), so these heights cover both call sites.
 *   (2) the REAL table when given: `r4d_emu_ds41emb real FILE.rad [ENTRY]` (token_embd.weight, 129280 x 5120 q6_K in
 *       the 2.25 model): its three planes read from the container and packed into llama.cpp's blocks as the loader's
 *       relayout does (pack_q6_K), the packing held to the planes (every pool row's lookup == bf16(code * sc * d),
 *       sign of zero included); the four shards are the deepseek41 split (shard_extent), and the ids are every shard's
 *       first and last two rows, the 64 rows with the most -0.0, 192 random rows, padding and an id past the table, at
 *       the heights of (1) up to 4160.
 *   (3) with the real table, every block's d is finite (so is every value: |sc| <= 128, |q - 32| <= 32); a value is then
 *       +-0 or at least 2^-24 in magnitude, never a bf16 subnormal. */
#include "rad_abi.h"
#include "rad_format.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "r4d_emit.h"
#include "r4d_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int r4d_gg_embed_emu(int ty, const int32_t* tok, const uint8_t* w, long long w_ld, uint16_t* x, long long x_ld, int M,
                     int E, int V, long long voff, float mult);

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };   /* r4d_ds41_ggml.hip's GG_* codes */
constexpr int kRanks = 4, kHc = 4;
constexpr int kGuard = 64;
constexpr uint16_t kJunk = 0x7e57;

/* ar_vsum's element (r4d_ar_oneshot_Nrank_exact.hip) at T = __hip_bfloat16: acc from 0.0f, r4d_ar_to_f = (float)x over
 * the ranks in ascending global rank (src[] is pre-ordered so on every rank), r4d_ar_from_f = (__hip_bfloat16)acc. */
__device__ __forceinline__ uint16_t ar_elem(const uint16_t* const* src, long long k) {
    float acc = 0.0f;
    for (int r = 0; r < kRanks; ++r) {
        __hip_bfloat16 v;
        std::memcpy(&v, &src[r][k], 2);
        acc += (float)v;
    }
    const __hip_bfloat16 o = (__hip_bfloat16)acc;
    uint16_t y;
    std::memcpy(&y, &o, 2);
    return y;
}
__global__ void fold_kernel(const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3, uint16_t* y,
                            long long n) {
    const uint16_t* src[kRanks] = { p0, p1, p2, p3 };
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (long long)gridDim.x * blockDim.x)
        y[k] = ar_elem(src, k);
}

/* binary_kernel (r4d_model_bf16.hip) as r4d_rad_add launches it here: a block a row, mul 0, b broadcast down the rows
 * (b_ld 0, b_step 1), div 1; f32_to_bf16 / bf16_to_f32 are r4d_common.h's own. */
__global__ void add_kernel(const uint16_t* a, const uint16_t* b, uint16_t* y, uint32_t n, uint32_t a_ld, uint32_t b_ld,
                           uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* ar = a + (size_t)m * a_ld;
    const uint16_t* br = b + (size_t)m * b_ld;
    uint16_t* yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float u = bf16_to_f32(ar[i]), v = bf16_to_f32(br[i]);
        const float r = u + v;
        yr[i] = f32_to_bf16(r);
    }
}
/* fill's bf16 row of value 0.0 */
__global__ void fill_kernel(uint16_t* y, uint32_t n) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) y[i] = f32_to_bf16(0.0f);
}

uint16_t host_fold(const uint16_t* v) {
    const uint16_t* src[kRanks] = { &v[0], &v[1], &v[2], &v[3] };
    return ar_elem(src, 0);
}
uint16_t host_add0(uint16_t h) { return f32_to_bf16(bf16_to_f32(h) + bf16_to_f32(f32_to_bf16(0.0f))); }

/* (0): every finite, non-subnormal bf16 and both zeros, at every owner rank. */
int check_values(long* n_checked) {
    int fails = 0;
    long n = 0;
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        const uint32_t e = (h >> 7) & 0xffu, mant = h & 0x7fu;
        if (e == 0xffu || (e == 0 && mant != 0)) continue;   /* NaN / infinity, subnormal */
        const uint16_t want = h == 0x8000u ? 0 : (uint16_t)h;
        const uint16_t add = host_add0((uint16_t)h);
        for (int r = 0; r < kRanks; ++r) {
            uint16_t v[kRanks] = { 0, 0, 0, 0 };
            v[r] = (uint16_t)h;
            const uint16_t ar = host_fold(v);
            ++n;
            if (ar != want || add != want) {
                if (++fails <= 8)
                    std::printf("FAIL value %04x at owner rank %d: all-reduce %04x, add(+0) %04x, want %04x\n", h, r,
                                ar, add, want);
            }
        }
    }
    *n_checked = n;
    return fails;
}

struct Ty { int code, qk, bb; const char* name; };
constexpr Ty kTypes[] = { { Q6_K, 256, 210, "q6_K" }, { Q5_K, 256, 176, "q5_K" }, { Q8_0, 32, 34, "q8_0" } };

/* arch/common/rad_arch.h's vocabulary split (vocab_per_rank / shard_extent / shard_begin). */
int64_t vocab_per_rank(int64_t total, int world) {
    if (total % world == 0) return total / world;
    const int64_t per = (total + world - 1) / world;
    return (per + 127) / 128 * 128;
}
int64_t shard_begin(int64_t total, int world, int rank) {
    const int64_t beg = vocab_per_rank(total, world) * rank;
    return beg < total ? beg : total;
}
int64_t shard_extent(int64_t total, int world, int rank) {
    const int64_t per = vocab_per_rank(total, world), beg = per * rank;
    const int64_t end = beg + per < total ? beg + per : total;
    return end > beg ? end - beg : 0;
}

/* A finite f16: random sign, exponent and mantissa, with zeros of both signs and subnormals now and then. */
uint16_t rand_f16(std::mt19937& rng, int emax) {
    const uint32_t r = rng();
    const uint16_t sign = (uint16_t)((r & 1u) << 15);
    switch ((r >> 1) % 32) {
    case 0: return sign;                                            /* +-0 */
    case 1: return (uint16_t)(sign | ((r >> 8) & 0x3ffu));          /* subnormal */
    default: {
        const uint16_t e = (uint16_t)(1 + (r >> 8) % (uint32_t)emax);
        return (uint16_t)(sign | (e << 10) | ((r >> 16) & 0x3ffu));
    }
    }
}

/* V rows of E values in llama.cpp's blocks: random bytes, every f16 field finite (exponents up to 2^-1, far from bf16
 * overflow; the 2.25 table's |d| <= 8.2e-4); q6_K blocks get many q = 32 codes (a value of 0) so -0.0 is common. */
std::vector<uint8_t> make_table(const Ty& t, int V, int E, std::mt19937& rng) {
    const int nb = E / t.qk;
    std::vector<uint8_t> w((size_t)V * nb * t.bb);
    for (auto& b : w) b = (uint8_t)rng();
    auto put16 = [&](size_t o, uint16_t h) { w[o] = (uint8_t)h; w[o + 1] = (uint8_t)(h >> 8); };
    for (size_t blk = 0; blk < (size_t)V * nb; ++blk) {
        const size_t o = blk * (size_t)t.bb;
        if (t.code == Q8_0) {
            put16(o, rand_f16(rng, 14));
        } else if (t.code == Q5_K) {
            put16(o, rand_f16(rng, 14));
            put16(o + 2, rand_f16(rng, 14));
        } else {
            put16(o + 208, rand_f16(rng, 14));
            /* a quarter of the low nibbles 0 and half the 2-bit high fields 10b: q = 32 often */
            for (int l = 0; l < 128; ++l)
                if (rng() % 4 == 0) w[o + l] = (uint8_t)(w[o + l] & 0xf0);
            for (int l = 0; l < 64; ++l)
                if (rng() % 2 == 0) w[o + 128 + l] = 0xaa;
        }
    }
    return w;
}

struct Totals { int cases = 0, fails = 0; long elems = 0, negz = 0, naive = 0; };

/* One step of M tokens through both paths over table `w` (V rows of E values, type t), compared. */
void run_case(const Ty& t, const uint8_t* w, int V, int E, const std::vector<int32_t>& tok, const char* what,
              Totals& tot) {
    const int M = (int)tok.size();
    const long long w_ld = (long long)(E / t.qk) * t.bb;
    const size_t n = (size_t)M * E, nh = (size_t)M * kHc * E;
    /* the shard path: each rank's lookup in its shard, the exact all-reduce's fold, the hc copies (byte copies) */
    std::vector<std::vector<uint16_t>> part(kRanks, std::vector<uint16_t>(n + kGuard, kJunk));
    int64_t beg[kRanks], ext[kRanks];
    for (int r = 0; r < kRanks; ++r) {
        beg[r] = shard_begin(V, kRanks, r);
        ext[r] = shard_extent(V, kRanks, r);
        const int rc = r4d_gg_embed_emu(t.code, tok.data(), w + (size_t)beg[r] * w_ld, w_ld, part[(size_t)r].data(), E,
                                        M, E, (int)ext[r], beg[r], 1.0f);
        if (rc != 0) { std::printf("FAIL %s %s: rank %d's shard lookup returned %d\n", t.name, what, r, rc); ++tot.fails; return; }
    }
    std::vector<uint16_t> xcol(n + kGuard, kJunk), h0a(nh + kGuard, kJunk);
    {
        const unsigned g = (unsigned)std::min<size_t>((n + 255) / 256, 1024);
        R4D_EMIT(nullptr, fold_kernel, dim3(g), dim3(256), 0, part[0].data(), part[1].data(), part[2].data(),
                 part[3].data(), xcol.data(), (long long)n);
    }
    for (int i = 0; i < kHc; ++i)
        for (int m = 0; m < M; ++m)
            std::memcpy(&h0a[(size_t)m * kHc * E + (size_t)i * E], &xcol[(size_t)m * E], (size_t)E * 2);
    /* the whole-table path: one lookup, the zero row, an add a hc copy */
    std::vector<uint16_t> full(n + kGuard, kJunk), zero((size_t)E + kGuard, kJunk), h0b(nh + kGuard, kJunk);
    const int rc = r4d_gg_embed_emu(t.code, tok.data(), w, w_ld, full.data(), E, M, E, V, 0, 1.0f);
    if (rc != 0) { std::printf("FAIL %s %s: the whole-table lookup returned %d\n", t.name, what, rc); ++tot.fails; return; }
    R4D_EMIT(nullptr, fill_kernel, dim3(1), dim3(256), 0, zero.data(), (uint32_t)E);
    for (int i = 0; i < kHc; ++i)
        R4D_EMIT(nullptr, add_kernel, dim3((unsigned)M), dim3(256), 0, full.data(), zero.data(), h0b.data() + (size_t)i * E,
                 (uint32_t)E, (uint32_t)E, 0u, (uint32_t)(kHc * E));
    ++tot.cases;
    long d1 = -1, d2 = -1, negz = 0, naive = 0;
    for (size_t i = 0; i < nh; ++i)
        if (h0a[i] != h0b[i]) { d1 = (long)i; break; }
    for (size_t i = 0; i < n; ++i) {
        negz += full[i] == 0x8000u;
        naive += full[i] != xcol[i];
        const int m = (int)(i / (size_t)E);
        const int64_t id = tok[(size_t)m];
        int owner = -1;
        for (int r = 0; r < kRanks; ++r)
            if (id >= beg[r] && id < beg[r] + ext[r]) owner = r;
        for (int r = 0; r < kRanks; ++r) {
            const uint16_t want = r == owner ? full[i] : 0;
            if (part[(size_t)r][i] != want && d2 < 0) d2 = (long)i;
        }
    }
    bool guards = true;
    for (size_t i = 0; i < (size_t)kGuard; ++i)
        guards = guards && h0a[nh + i] == kJunk && h0b[nh + i] == kJunk && full[n + i] == kJunk &&
                 xcol[n + i] == kJunk && zero[(size_t)E + i] == kJunk && part[0][n + i] == kJunk &&
                 part[3][n + i] == kJunk;
    tot.elems += (long)nh;
    tot.negz += negz;
    tot.naive += naive;
    if (d1 >= 0 || d2 >= 0 || !guards) {
        ++tot.fails;
        std::printf("FAIL %s %s E=%d V=%d M=%d: hc plane whole+add vs shards+fold+copy first diff %ld (%04x vs %04x), "
                    "shard rows vs the whole row first diff %ld, guards %d\n", t.name, what, E, V, M, d1,
                    d1 >= 0 ? h0b[(size_t)d1] : 0, d1 >= 0 ? h0a[(size_t)d1] : 0, d2, (int)guards);
    }
}

/* The trunk's heights (decode at 1..8 sequences, NST 5 verify at 6 a sequence, prefill chunks up to 4160) and the
 * DSpark query pass's (block 6 x 1..8 sequences). */
const int kTrunkM[] = { 1, 2, 3, 4, 5, 8, 11, 16, 24, 64, 200 };
const int kDraftM[] = { 6, 12, 18, 30, 48 };

std::vector<int32_t> batch(const std::vector<int32_t>& pool, int M, int start) {
    std::vector<int32_t> tok((size_t)M);
    for (int m = 0; m < M; ++m) tok[(size_t)m] = pool[(size_t)(start + m) % pool.size()];
    return tok;
}

void run_synthetic(bool quick, Totals& tot) {
    std::mt19937 rng(4141);
    for (const Ty& t : kTypes) {
        const int E = t.code == Q6_K ? 5120 : 512, Vs = 37, V = kRanks * Vs;
        const std::vector<uint8_t> w = make_table(t, V, E, rng);
        /* the pool: every shard edge, padding and past the table first, then the vocabulary in a random order */
        std::vector<int32_t> pool = { 0, Vs - 1, Vs, 2 * Vs - 1, 2 * Vs, 3 * Vs - 1, 3 * Vs, V - 1, -1, V, V + 7 };
        std::vector<int32_t> rest((size_t)V);
        for (int i = 0; i < V; ++i) rest[(size_t)i] = i;
        std::shuffle(rest.begin(), rest.end(), rng);
        pool.insert(pool.end(), rest.begin(), rest.end());
        int start = 0;
        for (int M : kTrunkM) {
            if (quick && M > 48) continue;
            run_case(t, w.data(), V, E, batch(pool, M, start), "trunk", tot);
            start += M;
        }
        for (int M : kDraftM) {
            run_case(t, w.data(), V, E, batch(pool, M, start), "drafter", tot);
            start += M;
        }
        /* the prefill chunk height, past the launch's 4096-block grid: the small-row type only (time) */
        if (!quick && t.code == Q8_0) run_case(t, w.data(), V, E, batch(pool, 4160, 0), "trunk prefill", tot);
    }
}

/* The .rad entry `name`'s planes (rad_format.h: header, directory, plane table, strings), each read whole. */
bool read_planes(const char* path, const char* name, int64_t* rows, int64_t* cols, std::vector<std::vector<uint8_t>>* pl) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    auto rd = [&](void* dst, size_t n, uint64_t off) {
        return fseeko(f, (off_t)off, SEEK_SET) == 0 && std::fread(dst, 1, n, f) == n;
    };
    RadFileHeader h{};
    bool ok = rd(&h, sizeof h, 0) && h.magic == RAD_MAGIC && h.version == RAD_FORMAT_VER;
    std::vector<char> str(ok ? (size_t)h.str_bytes : 0);
    ok = ok && rd(str.data(), str.size(), h.str_off) && !str.empty() && str.back() == 0;
    std::vector<RadFileEntry> dir(ok ? (size_t)h.dir_count : 0);
    ok = ok && rd(dir.data(), dir.size() * sizeof(RadFileEntry), h.dir_off);
    const RadFileEntry* e = nullptr;
    for (const RadFileEntry& d : dir)
        if (ok && d.name < str.size() && !std::strcmp(str.data() + d.name, name)) e = &d;
    ok = ok && e && e->rank == 2 && e->n_planes == 3;
    std::vector<RadFilePlane> pt(3);
    ok = ok && rd(pt.data(), 3 * sizeof(RadFilePlane), h.plane_off + e->plane_first * sizeof(RadFilePlane));
    if (ok) {
        *rows = e->shape[0];
        *cols = e->shape[1];
        pl->assign(3, {});
        for (int i = 0; i < 3 && ok; ++i) {
            (*pl)[(size_t)i].resize((size_t)pt[(size_t)i].bytes);
            ok = rd((*pl)[(size_t)i].data(), (size_t)pt[(size_t)i].bytes, pt[(size_t)i].offset);
        }
    }
    std::fclose(f);
    return ok;
}

/* libquant's pack_q6_K (lq_ggml_planes.cpp), which the loader runs per row at load (r4d_ds41_ggml_relayout): the
 * planes (codes i6 = q - 32, lowest bits first; scale i8 [1x16]; scale.1 f16 [1x256] = d) -> the row's native blocks. */
void pack_q6_K_row(const uint8_t* codes, const uint8_t* sc, const uint8_t* d, int nb, uint8_t* out) {
    uint64_t acc = 0;
    int n = 0;
    for (int i = 0; i < nb; ++i) {
        uint8_t* x = out + (size_t)i * 210;
        std::memset(x, 0, 192);
        for (int k = 0; k < 256; ++k) {
            while (n < 6) { acc |= (uint64_t)*codes++ << n; n += 8; }
            const uint32_t q = ((uint32_t)(acc & 63u) + 32u) & 63u;
            acc >>= 6;
            n -= 6;
            const int h = k / 128, t = (k % 128) / 32, l = k % 32;
            uint8_t& lo = x[64 * h + l + 32 * (t & 1)];
            lo |= (uint8_t)(t < 2 ? (q & 15) : (q & 15) << 4);
            x[128 + 32 * h + l] |= (uint8_t)((q >> 4) << (2 * t));
        }
        std::memcpy(x + 192, sc + 16 * i, 16);
        x[208] = d[2 * i];
        x[209] = d[2 * i + 1];
    }
}

float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t u;
    if (e == 0) {
        if (!m) u = s;
        else { float v = std::ldexp((float)m, -24); std::memcpy(&u, &v, 4); u |= s; }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

int run_real(const char* path, const char* name, Totals& tot) {
    const Ty& t = kTypes[0];
    int64_t V64 = 0, E64 = 0;
    std::vector<std::vector<uint8_t>> pl;
    if (!read_planes(path, name, &V64, &E64, &pl)) {
        std::printf("FAIL real: no 2-D three-plane entry '%s' in %s\n", name, path);
        return 1;
    }
    const int V = (int)V64, E = (int)E64, nb = E / t.qk;
    const size_t crow = (size_t)E * 6 / 8, srow = (size_t)E / 16, drow = (size_t)nb * 2;
    if (E % t.qk || pl[0].size() != (size_t)V * crow || pl[1].size() != (size_t)V * srow || pl[2].size() != (size_t)V * drow) {
        std::printf("FAIL real: '%s' is %d x %d with planes of %zu / %zu / %zu bytes, not q6_K's i6 / i8 [1x16] / f16 "
                    "[1x256]\n", name, V, E, pl[0].size(), pl[1].size(), pl[2].size());
        return 1;
    }
    const size_t w_ld = (size_t)nb * t.bb;
    std::vector<uint8_t> w((size_t)V * w_ld);
    for (int r = 0; r < V; ++r)
        pack_q6_K_row(pl[0].data() + r * crow, pl[1].data() + r * srow, pl[2].data() + r * drow, nb, w.data() + r * w_ld);
    /* (3) every block's d finite; zero d counted (a value is then +-0, never subnormal); per row, the values that are
     * -0.0 by the planes (code 0 under a negative sc * d, or d = -0 / sc = 0 sign products) */
    long nonfin = 0, dzero = 0, negz_vals = 0;
    std::vector<std::pair<int, int>> rowneg((size_t)V);
    for (int r = 0; r < V; ++r) {
        int cnt = 0;
        const uint8_t* c = pl[0].data() + r * crow;
        uint64_t acc = 0;
        int n = 0;
        for (int b = 0; b < nb; ++b) {
            const uint16_t dh = (uint16_t)(pl[2][r * drow + 2 * b] | (pl[2][r * drow + 2 * b + 1] << 8));
            if ((dh & 0x7c00u) == 0x7c00u) ++nonfin;
            if ((dh & 0x7fffu) == 0) ++dzero;
            const float d = f16_to_f32(dh);
            for (int k = 0; k < 256; ++k) {
                while (n < 6) { acc |= (uint64_t)*c++ << n; n += 8; }
                const int code = (int)(acc & 63u) - ((acc & 32u) ? 64 : 0);
                acc >>= 6;
                n -= 6;
                const float v = (float)code * (float)(int8_t)pl[1][r * srow + 16 * b + k / 16] * d;   /* the affine order */
                if (v == 0.0f && std::signbit(v)) ++cnt;
            }
        }
        negz_vals += cnt;
        rowneg[(size_t)r] = { cnt, r };
    }
    std::printf("lever EAR real table %s '%s': %d x %d q6_K planes -> native blocks, %ld blocks: d non-finite %ld, d zero "
                "%ld; %ld values dequantise to -0.0\n", path, name, V, E, (long)V * nb, nonfin, dzero, negz_vals);
    int fails = 0;
    if (nonfin) {
        std::printf("FAIL real: %ld blocks' d is not finite -- the identity (0) does not cover that table\n", nonfin);
        ++fails;
    }
    /* the pool, owners in turn: every shard's first and last two rows, the 64 rows with the most -0.0, 192 random rows;
     * then padding and an id past the table */
    std::vector<std::vector<int32_t>> by(kRanks);
    for (int r = 0; r < kRanks; ++r) {
        const int64_t b = shard_begin(V, kRanks, r), e = b + shard_extent(V, kRanks, r);
        for (int64_t id : { b, b + 1, e - 2, e - 1 }) by[(size_t)r].push_back((int32_t)id);
    }
    std::vector<std::pair<int, int>> top = rowneg;
    std::partial_sort(top.begin(), top.begin() + 64, top.end(), [](auto a, auto b) { return a.first > b.first; });
    std::mt19937 rng(4142);
    std::vector<int32_t> extra;
    for (int i = 0; i < 64; ++i) extra.push_back(top[(size_t)i].second);
    for (int i = 0; i < 192; ++i) extra.push_back((int32_t)(rng() % (uint32_t)V));
    for (int32_t id : extra)
        for (int r = 0; r < kRanks; ++r)
            if (id >= shard_begin(V, kRanks, r) && id < shard_begin(V, kRanks, r) + shard_extent(V, kRanks, r))
                by[(size_t)r].push_back(id);
    std::vector<int32_t> pool;
    for (size_t k = 0;; ++k) {
        bool any = false;
        for (int r = 0; r < kRanks; ++r)
            if (k < by[(size_t)r].size()) { pool.push_back(by[(size_t)r][k]); any = true; }
        if (!any) break;
    }
    /* the packing, held to the planes: every pool row's lookup (gg_embed_kernel) == bf16(code * sc * d), sign of zero
     * included */
    {
        const int P = (int)pool.size();
        std::vector<uint16_t> x((size_t)P * E, kJunk);
        if (r4d_gg_embed_emu(t.code, pool.data(), w.data(), (long long)w_ld, x.data(), E, P, E, V, 0, 1.0f) != 0) {
            std::printf("FAIL real: the pool lookup's launch failed\n");
            return fails + 1;
        }
        long bad = 0;
        for (int m = 0; m < P; ++m) {
            const int r = pool[(size_t)m];
            const uint8_t* c = pl[0].data() + r * crow;
            uint64_t acc = 0;
            int n = 0;
            for (int k = 0; k < E; ++k) {
                while (n < 6) { acc |= (uint64_t)*c++ << n; n += 8; }
                const int code = (int)(acc & 63u) - ((acc & 32u) ? 64 : 0);
                acc >>= 6;
                n -= 6;
                const int b = k / 256;
                const float d = f16_to_f32((uint16_t)(pl[2][r * drow + 2 * b] | (pl[2][r * drow + 2 * b + 1] << 8)));
                const float v = (float)code * (float)(int8_t)pl[1][r * srow + k / 16] * d;
                bad += x[(size_t)m * E + k] != f32_to_bf16(v);
            }
        }
        std::printf("lever EAR real table: the native blocks' lookup vs the planes' affine decode over %d rows: %ld "
                    "differing value(s)\n", P, bad);
        if (bad) ++fails;
    }
    pool.push_back(-1);
    pool.push_back(V);
    std::printf("lever EAR real table: pool of %zu ids (owners %zu / %zu / %zu / %zu; the most -0.0 in a row: %d), the "
                "first rows %d %d %d %d\n", pool.size(), by[0].size(), by[1].size(), by[2].size(), by[3].size(),
                top[0].first, pool[0], pool[1], pool[2], pool[3]);
    int start = 0;
    for (int M : kTrunkM) { run_case(t, w.data(), V, E, batch(pool, M, start), "real trunk", tot); start += M; }
    for (int M : kDraftM) { run_case(t, w.data(), V, E, batch(pool, M, start), "real drafter", tot); start += M; }
    run_case(t, w.data(), V, E, batch(pool, (int)pool.size(), 0), "real whole pool", tot);
    run_case(t, w.data(), V, E, batch(pool, 4160, 0), "real trunk prefill", tot);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const bool real = argc > 1 && !std::strcmp(argv[1], "real");
    if (real && argc != 3 && argc != 4) {
        std::printf("usage: %s [quick] | real FILE.rad [ENTRY (token_embd.weight)]\n", argv[0]);
        return 2;
    }
    long nval = 0;
    int fails = check_values(&nval);
    std::printf("lever EAR values: %ld (every finite non-subnormal bf16 and both zeros x 4 owner ranks): the all-reduce's "
                "fold == add(+0) == the value with -0 as +0, %d failure(s)\n", nval, fails);
    Totals tot;
    if (real) {
        fails += run_real(argv[2], argc > 3 ? argv[3] : "token_embd.weight", tot);
    } else {
        run_synthetic(quick, tot);
    }
    fails += tot.fails;
    std::printf("lever EAR %s: %d case(s), %ld hc-plane element(s): whole-table lookup + zero row + add vs shard lookups + "
                "the exact all-reduce's fold + copies, %d failure(s) in all; the lookups wrote %ld -0.0 value(s), and "
                "the lookup alone (no add) would differ from the all-reduce at %ld element(s)\n",
                real ? "real table" : "synthetic", tot.cases, tot.elems, fails, tot.negz, tot.naive);
    if (!fails && tot.negz == 0) {
        std::printf("FAIL: no -0.0 was produced, so the add's purpose was not exercised\n");
        return 1;
    }
    return fails ? 1 : 0;
}
