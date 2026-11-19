/* tests/r4d_emu: tp8 loop P30t-psk (19.11.2026) on the host SIMT emulator -- what deepseek41's int8 twin at lever PSK's layers
 * rests on (RADIANCE_DS41_TWIN bit 64, arch/deepseek41/ds41_psk_cone.h twin_cut_safe):
 *   1. gemm_i8a8_tiled -- the GEMM every twin launch of more than 64 rows runs (gemm_nt_q i8a8's band above the narrow tile)
 *      -- is ROW-INVARIANT: a launch over a sub-range [r0, r0 + m) of a pass's rows (lever PSK's cut = a chunk's last >= 160
 *      rows, an SP quarter, a row alone) writes, for those rows, the bytes the whole launch writes; and both are the host fold
 *      (int32 sums within a 128-K block, then out = fma(sum, a_scale x w_scale, out) block by block in K order, bf16 RNE) bit
 *      for bit -- which also holds the emulated iu8 WMMA (emu.cpp) to the operand layout the kernel reads.
 *   2. quant_act_i8g, the twin's activation quantiser, per (row, 128 columns): a sub-range launch = the whole launch's rows.
 *   3. (information, no check) gemm_i8a8_nt_m16, the narrow band (M <= 64): its K split follows M, so how many of a row's
 *      outputs differ from the tiled / host fold at M 1 .. 64 -- the band a cut never enters (lever PSK's floor is 160).
 * Shapes: N 256 / K 512 at M 300, N 128 / K 1024 at M 520; rows of zeros, large / tiny activation scales.
 *
 *   r4d_emu_ds41tw */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_fp8_args.h"
#include "r4d_fp8_frag.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N, int a_ld, int sa_ld,
                                   int w_ld, int sw_ld, long stream);
extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs*);
extern "C" int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_cases;
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL %s\n", what.c_str());
    }
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t r = 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)((u + r) >> 16);
}
float bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void mat(RadTensor& t, void* d, uint32_t dt, long long r, long long c, long long pitch) {
    std::memset(&t, 0, sizeof t);
    t.data = d;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = r;
    t.shape[1] = c;
    t.stride[0] = pitch;
    t.stride[1] = 1;
}

/* 16-byte aligned storage, as the arena's */
template <class T>
struct Al {
    std::vector<uint8_t> raw;
    T* p = nullptr;
    explicit Al(size_t n) : raw(n * sizeof(T) + 64, 0) { p = (T*)(((uintptr_t)raw.data() + 15) & ~(uintptr_t)15); }
};

struct Case {
    int M, N, K;
    std::vector<int8_t> a, w;   /* row-major [M][K], [N][K] */
    std::vector<float> as;      /* [M][K / 128] */
    std::vector<uint16_t> ws;   /* [N][K / 128] bf16 */
};

Case make(int M, int N, int K, std::mt19937& rng) {
    Case c{ M, N, K, {}, {}, {}, {} };
    const int nkb = K / 128;
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    c.a.resize((size_t)M * K);
    c.w.resize((size_t)N * K);
    c.as.resize((size_t)M * nkb);
    c.ws.resize((size_t)N * nkb);
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k) c.a[(size_t)r * K + k] = (int8_t)(r % 37 == 5 ? 0 : code(rng));
    for (auto& v : c.w) v = (int8_t)code(rng);
    for (int r = 0; r < M; ++r)
        for (int b = 0; b < nkb; ++b)
            c.as[(size_t)r * nkb + b] = r % 11 == 3 ? 1.0f : (r % 13 == 4 ? 37.5f * u(rng) : 1e-3f + 0.05f * u(rng));
    for (auto& v : c.ws) v = bf16_of(1e-3f + 0.02f * u(rng));
    return c;
}

/* the weight in fragment order (r4d_fp8_frag.h, the layout hook's walk) and its scale plane tile-major [N/16][K/128][16] */
void planes(const Case& c, Al<uint8_t>& wf, Al<uint16_t>& sf) {
    const int64_t ksuper = c.K / r4d_fp8::kSuperK, nkb = c.K / 128;
    int64_t at = 0;
    for (int64_t rt = 0; rt < c.N / r4d_fp8::kFragRows; ++rt)
        for (int64_t ks = 0; ks < ksuper; ++ks)
            for (int64_t lane = 0; lane < r4d_fp8::kTileBytes / r4d_fp8::kLaneBytes; ++lane)
                for (int64_t t = 0; t < r4d_fp8::kLaneBytes / 8; ++t)
                    for (int64_t j = 0; j < 8; ++j, ++at) {
                        int64_t row, k;
                        r4d_fp8::frag_src(rt, ks, lane, t, j, &row, &k);
                        wf.p[at] = (uint8_t)c.w[(size_t)(row * c.K + k)];
                    }
    for (int64_t r = 0; r < c.N; ++r)
        for (int64_t kb = 0; kb < nkb; ++kb) sf.p[((r / 16) * nkb + kb) * 16 + r % 16] = c.ws[(size_t)(r * nkb + kb)];
}

/* the host fold: row r, column n */
uint16_t host_fold(const Case& c, int r, int n) {
    const int nkb = c.K / 128;
    float out = 0.f;
    for (int kb = 0; kb < nkb; ++kb) {
        int s = 0;
        for (int k = kb * 128; k < kb * 128 + 128; ++k) s += (int)c.a[(size_t)r * c.K + k] * (int)c.w[(size_t)n * c.K + k];
        const float sc = c.as[(size_t)r * nkb + kb] * bf16_val(c.ws[(size_t)n * nkb + kb]);
        out = std::fmaf((float)s, sc, out);
    }
    return bf16_of(out);
}

/* gemm_i8a8_tiled over rows [r0, r0 + m) of the case into y (pitch N) */
int tiled(const Case& c, const Al<uint8_t>& wf, const Al<uint16_t>& sf, Al<int8_t>& a, Al<float>& as, int r0, int m,
          uint16_t* y) {
    const int nkb = c.K / 128;
    return r4d_gemm_i8a8_tiled((long)(uintptr_t)(a.p + (size_t)r0 * c.K), (long)(uintptr_t)(as.p + (size_t)r0 * nkb),
                               (long)(uintptr_t)wf.p, (long)(uintptr_t)sf.p, (long)(uintptr_t)y, m, c.K, c.N, c.K, nkb, 0,
                               nkb, 0);
}

void tiled_cases(std::mt19937& rng) {
    struct S { int M, N, K; std::vector<std::pair<int, int>> subs; };
    /* (r0, m): lever PSK's cuts (a chunk's last 160 / 161 / 288 rows), the band's floor (65 rows), an SP quarter
     * (ceil(T / 4) and the last rank's rest), a 128-row tile's edges, one row */
    const S shapes[] = {
        { 300, 256, 512, { { 140, 160 }, { 139, 161 }, { 12, 288 }, { 0, 65 }, { 235, 65 }, { 0, 75 }, { 225, 75 },
                           { 127, 2 }, { 299, 1 }, { 64, 128 } } },
        { 520, 128, 1024, { { 360, 160 }, { 0, 130 }, { 390, 130 }, { 232, 288 }, { 1, 519 } } },
    };
    for (const S& sh : shapes) {
        const Case c = make(sh.M, sh.N, sh.K, rng);
        const int nkb = c.K / 128;
        Al<uint8_t> wf((size_t)c.N * c.K);
        Al<uint16_t> sf((size_t)c.N * nkb);
        planes(c, wf, sf);
        Al<int8_t> a((size_t)c.M * c.K);
        Al<float> as((size_t)c.M * nkb);
        std::memcpy(a.p, c.a.data(), c.a.size());
        std::memcpy(as.p, c.as.data(), c.as.size() * 4);
        const std::string tag = "N " + std::to_string(c.N) + " K " + std::to_string(c.K) + " M " + std::to_string(c.M);
        Al<uint16_t> whole((size_t)c.M * c.N);
        check(tiled(c, wf, sf, a, as, 0, c.M, whole.p) == 0, tag + ": the whole launch runs");
        long long bad = 0, nz = 0;
        for (int r = 0; r < c.M; ++r)
            for (int n = 0; n < c.N; ++n) {
                const uint16_t h = host_fold(c, r, n), g = whole.p[(size_t)r * c.N + n];
                bad += h != g;
                nz += g != 0;
                if (h != g && bad <= 3)
                    std::printf("    %s row %d col %d: kernel %04x host %04x\n", tag.c_str(), r, n, (unsigned)g, (unsigned)h);
            }
        check(bad == 0, tag + ": the whole launch = the host fold bit for bit (" + std::to_string(bad) + " differ)");
        check(nz > (long long)c.M * c.N / 2, tag + ": outputs not mostly zero (the case reads something)");
        for (const auto& [r0, m] : sh.subs) {
            Al<uint16_t> part((size_t)m * c.N);
            for (size_t i = 0; i < (size_t)m * c.N; ++i) part.p[i] = 0xa5a5;
            const std::string st = tag + " rows [" + std::to_string(r0) + ", " + std::to_string(r0 + m) + ")";
            check(tiled(c, wf, sf, a, as, r0, m, part.p) == 0, st + ": the sub-range launch runs");
            check(std::memcmp(part.p, whole.p + (size_t)r0 * c.N, (size_t)m * c.N * 2) == 0,
                  st + ": = the whole launch's rows, byte for byte (row-invariant)");
        }
    }
}

int quant(int M, int n, const uint16_t* x, uint8_t* q, float* s) {
    RadTensor t[3];
    mat(t[0], (void*)x, RAD_BF16, M, n, n);
    mat(t[1], q, RAD_I8, M, n, n);
    mat(t[2], s, RAD_F32, M, n / 128, n / 128);
    RadParam p[4]{};
    p[0] = { "M", RAD_P_INT, M };
    p[1] = { "n", RAD_P_INT, n };
    p[2] = { "group", RAD_P_INT, 128 };
    p[3].key = "dtype";
    p[3].kind = RAD_P_STR;
    p[3].sval = "bf16";
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 4;
    return r4d_rad_quant_act_i8g(&a, nullptr);
}

void quant_cases(std::mt19937& rng) {
    const int M = 300, n = 512;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x((size_t)M * n);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < n; ++c) x[(size_t)r * n + c] = bf16_of(nd(rng) * (r % 9 == 2 ? 0.f : r % 5 == 1 ? 700.f : 1.f));
    std::vector<uint8_t> q((size_t)M * n);
    std::vector<float> s((size_t)M * n / 128);
    check(quant(M, n, x.data(), q.data(), s.data()) == RAD_OK, "quant_act_i8g M 300: the whole launch");
    for (const auto& [r0, m] : std::vector<std::pair<int, int>>{ { 140, 160 }, { 0, 65 }, { 299, 1 }, { 75, 75 } }) {
        std::vector<uint8_t> qp((size_t)m * n);
        std::vector<float> sp((size_t)m * n / 128);
        const std::string st = "quant_act_i8g rows [" + std::to_string(r0) + ", " + std::to_string(r0 + m) + ")";
        check(quant(m, n, x.data() + (size_t)r0 * n, qp.data(), sp.data()) == RAD_OK, st + ": runs");
        check(std::memcmp(qp.data(), q.data() + (size_t)r0 * n, qp.size()) == 0 &&
                  std::memcmp(sp.data(), s.data() + (size_t)r0 * n / 128, sp.size() * 4) == 0,
              st + ": codes + scales = the whole launch's rows");
    }
}

/* information: the narrow tile at M rows (its own K split) against the host fold */
void narrow_info(std::mt19937& rng) {
    const Case c = make(64, 256, 2048, rng);
    const int nkb = c.K / 128;
    Al<uint8_t> wf((size_t)c.N * c.K);
    Al<uint16_t> sf((size_t)c.N * nkb);
    planes(c, wf, sf);
    for (int M : { 1, 8, 16, 33, 64 }) {
        Al<int8_t> a((size_t)M * c.K);
        Al<float> as((size_t)M * nkb);
        std::memcpy(a.p, c.a.data(), (size_t)M * c.K);
        std::memcpy(as.p, c.as.data(), (size_t)M * nkb * 4);
        Al<uint16_t> y((size_t)M * c.N);
        RadTensor t[5];
        mat(t[F_A], a.p, RAD_I8, M, c.K, c.K);
        mat(t[F_ASCALE], as.p, RAD_F32, M, nkb, nkb);
        mat(t[F_B], wf.p, RAD_I8, c.N, c.K, c.K);
        mat(t[F_BSCALE], sf.p, RAD_BF16, c.N, nkb, nkb);
        mat(t[F_Y], y.p, RAD_BF16, M, c.N, c.N);
        RadParam p[5]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "N", RAD_P_INT, c.N };
        p[2] = { "K", RAD_P_INT, c.K };
        p[3] = { "group", RAD_P_INT, 128 };
        p[4].key = "dtype";
        p[4].kind = RAD_P_STR;
        p[4].sval = "i8a8";
        RadArgs ar{};
        ar.t = t;
        ar.n_t = 5;
        ar.p = p;
        ar.n_p = 5;
        const int64_t sb = r4d_rad_gemm_i8a8_m16_scratch(&ar);
        Al<uint8_t> scr((size_t)(sb > 0 ? sb : 16));
        ar.scratch = scr.p;
        ar.scratch_bytes = (uint64_t)(sb > 0 ? sb : 0);
        const int rc = r4d_rad_gemm_i8a8_m16(&ar, nullptr);
        long long bad = 0;
        for (int r = 0; r < M && rc == RAD_OK; ++r)
            for (int n = 0; n < c.N; ++n) bad += host_fold(c, r, n) != y.p[(size_t)r * c.N + n];
        std::printf("  info: gemm_i8a8_nt_m16 N %d K %d M %d (scratch %lld B): rc %d, %lld of %lld outputs differ from the "
                    "tiled / host fold\n", c.N, c.K, M, (long long)sb, rc, bad, (long long)M * c.N);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261119u);
    tiled_cases(rng);
    quant_cases(rng);
    narrow_info(rng);
    std::printf("r4d_emu_ds41tw: %d check(s), %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
