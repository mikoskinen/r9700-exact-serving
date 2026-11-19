/* tests/r4d_emu: tp8 loop P30f (FINDINGS F63, 19.11.2026) on the host SIMT emulator -- the two facts deepseek41's int8
 * attention_output_a / attention_output_b form rests on:
 *   1. libr4d's int8 / fp8 GEMMs read their operands through r4d_fp8_args.h's fp8_ops, which takes C at pitch N only:
 *      an attention_output_a group's column slice of attn_o_lora ([M, 1024] at pitch 2048, TP4's two groups a rank) is
 *      refused RAD_E_STRIDE (what the cards printed at step 0: 'gemm_i8a8_nt_m16 refused op gemm_nt_q ... non-contiguous
 *      operand'), the group's own [M, 1024] buffer is taken, and the activation pair may sit at any pitch (a K = 2048
 *      slice of the model's 5120-wide pair);
 *   2. quant_act_i8g / had_quant_act_i8g are per (row, 128 columns), so o_b's quantiser run ONCE A GROUP -- each over its
 *      group's own contiguous buffer, or (a GGUF group kept in place) over attn_o_lora's column slice, into that group's
 *      columns of the pair -- writes the codes and scales of one launch over the concatenation, byte for byte; columns of
 *      the pair past K are untouched (guards).
 * Shapes: TP4's 2 x 1024 (K 2048) at M 1 / 3 / 48 / 64 / 130, and 4 x 256, 3 x 384, 2 x 128; plain and rotated forms;
 * rows of zeros (scale 1 path), large and tiny magnitudes, one 128-group of zeros inside a row.
 *
 *   r4d_emu_ds41p30f */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_fp8_args.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);
extern "C" int r4d_rad_had_quant_act_i8g(const RadArgs*, RadStream);
/* fp8_ops checks the library's block against R4D_FP8_BLOCK; r4d_gemm_fp8a8.hip (not in this build) defines it */
extern "C" int r4d_gemm_fp8a8_block(void) { return R4D_FP8_BLOCK; }

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

/* one quantiser launch: x [M, n] at x_pitch (elements) -> q [M, n] at q_pitch bytes, s [M, n / 128] at s_pitch */
int quant(bool rot, int M, int n, const uint16_t* x, long long xp, uint8_t* q, long long qp, float* s, long long sp) {
    RadTensor t[3];
    mat(t[0], (void*)x, RAD_BF16, M, n, xp);
    mat(t[1], q, RAD_I8, M, n, qp);
    mat(t[2], s, RAD_F32, M, n / 128, sp);
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
    return rot ? r4d_rad_had_quant_act_i8g(&a, nullptr) : r4d_rad_quant_act_i8g(&a, nullptr);
}

void split_cases(std::mt19937& rng) {
    struct S { int M, gk, gr; };
    const S shapes[] = { { 1, 1024, 2 }, { 3, 1024, 2 }, { 48, 1024, 2 }, { 64, 1024, 2 }, { 130, 1024, 2 },
                         { 17, 256, 4 }, { 9, 384, 3 }, { 5, 128, 2 } };
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const S& sh : shapes) {
        const int M = sh.M, gk = sh.gk, gr = sh.gr, k = gk * gr, kb = k / 128;
        const long long QP = k + 3 * 128, SP = QP / 128;   /* the pair's pitch: wider than K, as the model's 5120 */
        std::vector<uint16_t> x((size_t)M * k);
        for (int r = 0; r < M; ++r) {
            const float mag = r % 7 == 3 ? 0.f : (r % 5 == 1 ? 900.f : (r % 5 == 2 ? 1e-3f : 1.f));
            for (int c = 0; c < k; ++c) {
                float v = nd(rng) * mag;
                if (r % 4 == 2 && c / 128 == kb / 2) v = 0.f;   /* one zero group inside a row */
                x[(size_t)r * k + c] = bf16_of(v);
            }
        }
        std::vector<std::vector<uint16_t>> g((size_t)gr, std::vector<uint16_t>((size_t)M * gk));
        for (int j = 0; j < gr; ++j)
            for (int r = 0; r < M; ++r)
                std::memcpy(&g[(size_t)j][(size_t)r * gk], &x[(size_t)r * k + (size_t)j * gk], (size_t)gk * 2);
        for (int rot = 0; rot < 2; ++rot) {
            std::vector<uint8_t> qr((size_t)M * QP, 0xa5), qa((size_t)M * QP, 0xa5), qb((size_t)M * QP, 0xa5);
            std::vector<float> sr((size_t)M * SP, -7.25f), sa((size_t)M * SP, -7.25f), sb((size_t)M * SP, -7.25f);
            const std::string tag = std::string(rot ? "had_quant_act_i8g" : "quant_act_i8g") + " M " + std::to_string(M) +
                                    " " + std::to_string(gr) + " x " + std::to_string(gk);
            check(quant(rot, M, k, x.data(), k, qr.data(), QP, sr.data(), SP) == RAD_OK, tag + ": the whole launch");
            bool okA = true, okB = true;
            for (int j = 0; j < gr; ++j) {
                const size_t c0 = (size_t)j * gk, b0 = c0 / 128;
                /* A: every group from its own buffer */
                okA = okA && quant(rot, M, gk, g[(size_t)j].data(), gk, qa.data() + c0, QP, sa.data() + b0, SP) == RAD_OK;
                /* B: odd groups in place (attn_o_lora's column slice), even ones from their own buffers */
                okB = okB && (j % 2 ? quant(rot, M, gk, x.data() + c0, k, qb.data() + c0, QP, sb.data() + b0, SP)
                                    : quant(rot, M, gk, g[(size_t)j].data(), gk, qb.data() + c0, QP, sb.data() + b0, SP)) ==
                                 RAD_OK;
            }
            check(okA && okB, tag + ": the group launches");
            check(qa == qr && std::memcmp(sa.data(), sr.data(), sa.size() * 4) == 0,
                  tag + ": groups from their own buffers = one launch over the concatenation (codes + scales, guards)");
            check(qb == qr && std::memcmp(sb.data(), sr.data(), sb.size() * 4) == 0,
                  tag + ": mixed own buffers / in-place column slices = one launch (codes + scales, guards)");
            bool guards = true;
            for (int r = 0; r < M; ++r) {
                for (long long c = k; c < QP; ++c) guards = guards && qr[(size_t)r * QP + c] == 0xa5;
                for (long long c = kb; c < SP; ++c) guards = guards && sr[(size_t)r * SP + c] == -7.25f;
            }
            check(guards, tag + ": the pair's columns past K untouched");
            bool nz = false;
            for (int r = 0; r < M && !nz; ++r)
                for (int c = 0; c < k && !nz; ++c) nz = qr[(size_t)r * QP + c] != 0;
            check(nz || M == 0, tag + ": codes not all zero (the case reads something)");
        }
    }
}

void fp8ops_cases() {
    /* gemm_nt_q i8a8 over attention_output_a's group: N 1024, K 4096; o_b: N 5120, K 2048 over the 5120-wide pair */
    const int M = 8;
    struct G { int N, K; long long yp, ap; bool ok; const char* what; };
    const G gs[] = {
        { 1024, 4096, 2048, 5120, false, "o_a group 0 into its column slice of attn_o_lora (pitch 2048): refused (F63)" },
        { 1024, 4096, 1024, 5120, true, "o_a group into its own [M, 1024] buffer: taken" },
        { 5120, 2048, 5120, 5120, true, "o_b over a K 2048 slice of the 5120-wide pair: taken (the pair has a pitch)" },
        { 1280, 5120, 1280, 5120, true, "q_a into q_lora: taken" },
        { 25600, 6144, 25600, 6144, true, "an int8 Engram kv over its own 6144-wide pair: taken" },
    };
    for (const G& gg : gs) {
        const int N = gg.N, K = gg.K, nkb = K / 128;
        std::vector<int8_t> a((size_t)M * gg.ap + 64), w((size_t)N * K + 64);
        std::vector<float> as((size_t)M * 64);
        std::vector<uint16_t> ws((size_t)N * nkb), y((size_t)M * gg.yp + 64);
        RadTensor t[5];
        mat(t[F_A], a.data(), RAD_I8, M, K, gg.ap);
        mat(t[F_ASCALE], as.data(), RAD_F32, M, nkb, 64);
        mat(t[F_B], w.data(), RAD_I8, N, K, K);
        mat(t[F_BSCALE], ws.data(), RAD_BF16, N, nkb, nkb);
        mat(t[F_Y], y.data(), RAD_BF16, M, N, gg.yp);
        /* 16-byte aligned data, as the arena's */
        t[F_A].data = (void*)(((uintptr_t)a.data() + 15) & ~(uintptr_t)15);
        t[F_B].data = (void*)(((uintptr_t)w.data() + 15) & ~(uintptr_t)15);
        RadArgs ar{};
        ar.t = t;
        ar.n_t = 5;
        Fp8Ops g;
        const int rc = fp8_ops(&ar, &g, 1, FP8_OPS_ROW_SCALE);
        check(gg.ok ? rc == RAD_OK : rc == RAD_E_STRIDE,
              std::string(gg.what) + " (fp8_ops " + std::to_string(rc) + ")");
        if (gg.ok && rc == RAD_OK) check(g.M == M && g.N == N && g.K == K && g.a_ld == gg.ap, std::string(gg.what) + ": M / N / K / a pitch");
    }
}

}  // namespace

int main() {
    std::mt19937 rng(30);
    split_cases(rng);
    fp8ops_cases();
    std::printf("%d check(s), %d failure(s)\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
