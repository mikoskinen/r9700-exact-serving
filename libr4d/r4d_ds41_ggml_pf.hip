/* r4d_ds41_ggml_pf.hip -- lever GPF: gemm_nt_ggml / logits_gemm_ggml's prefill rows (above the decode band) on the
 * stock WMMA kernel's arithmetic in other tile geometries. Off unless switched on (r4d_ds41_ggml_pf.h); gg_gemm
 * calls it ahead of the stock WMMA launch.
 *
 * ---- the stock kernel (r4d_ds41_ggml.hip, gg_wm_kernel, dense rows)
 *
 *  A block is BM = 48 activation rows (three 16-row tiles) by 256 weight rows (8 waves x 2 column tiles of 16). The
 *  activation slab (48 rows x 128 values, bf16) is staged in LDS a slab ahead, with its rows' scale-group sums
 *  (times XGK) for the K-quants. Each wave decodes its own two column tiles' blocks straight into B fragments and
 *  runs, per 16-value K step J, NM x 2 WMMAs (NM live row tiles): P = wmma(A(mt, J), B(nt, J), c0) chained over a
 *  scale group (SPG steps; c0 = 0, or the staged group sum for Q6_K's CINIT), then acc = fma(P, scale, acc) (and,
 *  Q5_K, acc = fma(groupsum, min, acc)) once a group. Profiled at DeepSeek-V4.1-Flash's 128k prefill (dsm-225, rank
 *  0): 254 ms of a 1206 ms step at 49-65 TFLOP/s a shape, ~1/3 of the bf16 WMMA rate.
 *
 *  Per K step a wave issues 6 WMMAs against ~68 VALU (Q6_K NM 3: 2 B fragments decoded, ~36 plain VALU, 48 scale
 *  fmas, most of them dual-issued) and 9 LDS fragment reads (3 A fragments, and 2 x 3 for Q6_K's chain inits: each
 *  lane reads its 8 rows' sums), so both the VALU slots and LDS bandwidth bind near the WMMA work.
 *
 * ---- the forms (gg_pf_kernel<T, MT, NT, WN, OT, ABL, FI>)
 *
 *  MT 16-row tiles a block (BM = 16 MT), WN waves a block, NT column tiles a wave (BN = 16 NT WN weight rows a block).
 *  FI 1 interleaves a scale group's folds with the step's WMMAs (row tile mt's WMMAs, then tile mt - 1's fmas), so two
 *  tiles' partials are live instead of NM: the registers MT 4 needs without spilling.
 *  More row tiles a wave share each decoded B fragment (decode a WMMA 1 / NM); more column tiles a wave share each
 *  A fragment and chain init read (LDS bytes a WMMA ~ 1 / NT); fewer waves a block, more blocks a WGP. Everything
 *  else is the stock kernel's code: the same staging (16-byte chunks, the same group-sum order and DPP pairs), the
 *  same weight words (Wq<T>::load / reload, EARLY / ROLL sets), the same B decode (Wq<T>::bfrag), the same scale /
 *  min arithmetic, the same step order and the same epilogue.
 *
 * ---- exactness: each output (row r, weight row n) is the same sequence of operations as in gg_wm_kernel
 *
 *  Output (r, n) lives in the lane of n's column tile, in the accumulator of r's row tile, whatever the geometry: its
 *  A fragment for step J of slab s is the staged row r's values 128 s + 16 J + 8 h .. + 7 (xoff), its B fragment n's
 *  decoded values at the same K (bfrag<J> of n's words for slab s), its chain per scale group starts at the same c0
 *  (the staged group sum, computed by the same 8-value add order and the same DPP pairs: a chunk's lane pair /
 *  quad is the same since every block's thread count is a multiple of 32), runs the same SPG WMMAs, and is folded
 *  by the same fma(s) in ascending group and slab order. The tile geometry, the block's thread count and the grid
 *  only change which lane / block holds it. Rows past M stage the block's last live row (as the stock's dead rows
 *  stage a live one) and write nothing; the last weight row is re-read past N and nothing is written there.
 *  r4d_selftest --gpcheck byte-compares every form against the stock kernel at every DeepSeek-V4.1 dense shape and
 *  type at M 2048 / 4096 and odd tails; tests/r4d_emu's ds41pf runs both kernels on the host emulator.
 *
 *  ABL != 0 (bench only, never a rule): timing ablations whose outputs DIFFER -- 1 no chain-init / min group-sum
 *  LDS reads (c0 = 0, min term 0), 2 no B decode (raw words as fragments), 4 no per-group fma (one chain a slab). */
#define R4D_GG_BLOCKS_ONLY 1
#define R4D_GG_WM_SRC 1
#include "r4d_ds41_ggml.hip"
#include "r4d_ds41_ggml_pf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <int T, int MT, int NT, int WN, typename OT, int ABL, int FI>
__global__ __launch_bounds__(WN * 32) void gg_pf_kernel(GgTiledArgs A) {
    using W = Wq<T>;
    using Pf = typename W::Pf;
    constexpr int NTH = WN * 32;
    constexpr int BM = 16 * MT, BN = 16 * NT * WN, WC = 16 * NT;   /* WC weight rows a wave */
    constexpr int XC = BM * 16 / NTH;                               /* staged 16-byte chunks a thread a slab */
    static_assert(XC * NTH == BM * 16, "the slab's chunks split evenly over the block");
    constexpr int XE = BM * kWmXP;                                  /* bf16 a staged slab */
    constexpr int XGE = W::XG > 0 ? kWmS / W::XG * BM : 0;          /* floats a slab's group sums */
    constexpr bool NOCI = (ABL & 1) != 0, NODEC = (ABL & 2) != 0, NOFMA = (ABL & 4) != 0;
#ifdef R4D_EMU_HOST   /* tests/r4d_emu: the launch's dynamic LDS */
    uint8_t* const gg_smem = (uint8_t*)emu_dyn_lds();
#else
    extern __shared__ __attribute__((aligned(16))) uint8_t gg_smem[];
#endif
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    /* Dense rows: the block's live rows are r0 .. r1 - 1 (the stock's s_tok = i there, -1 past), so its live 16-row
     * tiles are the first ceil((r1 - r0) / 16) (the stock's s_nmt) and a dead row stages r1 - 1 (its s_any). */
    const int r0 = (int)blockIdx.x * BM, r1 = min(r0 + BM, A.M);
    const int nmt = (r1 - r0 + 15) >> 4;
    const int tany = r1 - 1;
    const int n0 = (int)blockIdx.y * BN;
    W::table(gg_smem);
    uint16_t* Xs = (uint16_t*)(void*)(gg_smem + W::TB);
    float* Xg = (float*)(void*)(gg_smem + W::TB + 2 * XE * 2);
    const uint8_t* wb = A.w;
    const int h = lane >> 4, c16 = lane & 15;
    const uint8_t* wr[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int n = n0 + wave * WC + nt * 16 + c16;
        wr[nt] = wb + (long long)(n < A.N ? n : A.N - 1) * A.w_ld;
    }
    /* Activation staging as the stock's: XC 16-byte chunks a thread a slab (row q >> 4, values 8 (q & 15)). */
    const uint16_t* xsrc[XC];
    int xdo[XC];
#pragma unroll
    for (int i = 0; i < XC; ++i) {
        const int q = tid + NTH * i, row = q >> 4, kc = (q & 15) * 8;
        const int r = r0 + row;
        xsrc[i] = A.x + (long long)(r < r1 ? r : tany) * A.x_ld + kc;
        xdo[i] = row * kWmXP + kc;
    }
    r4d_u32x4 xr[XC];
    auto xload = [&](int s) {
#pragma unroll
        for (int i = 0; i < XC; ++i) xr[i] = *(gg_gu128)(const void*)(xsrc[i] + s * kWmS);
    };
    auto xstore = [&](int b) {
        uint16_t* xs = Xs + b * XE;
#pragma unroll
        for (int i = 0; i < XC; ++i) {
            *(r4d_u32x4*)(void*)(xs + xdo[i]) = xr[i];
            if constexpr (W::XG > 0) {
                const int q = tid + NTH * i;
                float sm = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) sm += bf16w_lo(xr[i][k]) + bf16w_hi(xr[i][k]);
                sm += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(sm), 0xB1, 0xF, 0xF, false));
                if constexpr (W::XG == 32)
                    sm += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(sm), 0x4E, 0xF, 0xF, false));
                Xg[b * XGE + ((q & 15) * 8 / W::XG) * BM + (q >> 4)] = W::XGK * sm;
            }
        }
    };
    const int S = A.K / kWmS;
    /* The stock's L2 prefetch three slabs ahead, over the wave's WC weight rows. */
    const uint8_t* prow = gg_sptr(wb + (long long)(n0 + __builtin_amdgcn_readfirstlane(wave) * WC) * A.w_ld);
    const int prmax = __builtin_amdgcn_readfirstlane(A.N - 1 - (n0 + wave * WC));
    const long long psb = A.w_ld / S;
    const int pfl = (int)min(31LL, (2 * psb + 127) / 128);
    auto prefetch = [&](int s) {
        const uint8_t* p = prow + (long long)min(s + 3, S - 1) * psb;
#pragma unroll 1
        for (int i = 0; i < WC; ++i) {
            __builtin_amdgcn_s_prefetch_data(p, pfl);
            if (i < prmax) p += A.w_ld;
        }
    };
    const uint16_t* xa = Xs + c16 * kWmXP;
    const float* xgl = Xg + 8 * h;
    auto run = [&](auto nmc) {
        constexpr int NM = decltype(nmc)::value;
        auto live = [&](int mt) { return !W::GUARD || mt < nmt; };
        gg_v8f acc[NM][NT];
#pragma unroll
        for (int i = 0; i < NM; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[i][j] = (gg_v8f)(0.0f);
        uint32_t rE[2][NT][W::RAW], rR[NT][W::RAW];
        xload(0);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            uint32_t t[W::RAW];
            W::load(wr[nt], 0, h, t);
#pragma unroll
            for (int k = 0; k < W::RAW; ++k) {
                if ((W::EARLY >> k) & 1u) rE[0][nt][k] = t[k];
                else rR[nt][k] = t[k];
            }
        }
        xstore(0);
        xload(1);
        r4d_u32x4 bf[2][NT];
        Pf pf[NT];
        lds_barrier();
        auto bfr = [&](auto jc, const uint32_t* rc, const Pf& p, int s) -> r4d_u32x4 {
            constexpr int J = decltype(jc)::value;
            if constexpr (NODEC) {   /* ablation: the raw words as the fragment */
                r4d_u32x4 b;
                b[0] = rc[(4 * J) % W::RAW]; b[1] = rc[(4 * J + 1) % W::RAW];
                b[2] = rc[(4 * J + 2) % W::RAW]; b[3] = rc[(4 * J + 3) % W::RAW];
                return b;
            } else {
                return W::template bfrag<J>(rc, h, p, s);
            }
        };
        {
            uint32_t rc[NT][W::RAW];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[0][nt][k] : rR[nt][k];
            if constexpr (W::PIPE) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) W::template pre<0>(rc[nt], h, gg_smem, pf[nt]);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) bf[0][nt] = bfr(std::integral_constant<int, 0>{}, rc[nt], pf[nt], 0);
            }
        }
        auto slab = [&](auto pc, int s) {
            constexpr int PB = decltype(pc)::value;
            uint32_t rc[NT][W::RAW];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[PB][nt][k] : rR[nt][k];
            const int sn = s + 1 < S ? s + 1 : s;
            if constexpr (PB == 0) prefetch(s);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint32_t t[W::RAW];
                W::load(wr[nt], sn, h, t);
#pragma unroll
                for (int k = 0; k < W::RAW; ++k)
                    if ((W::EARLY >> k) & 1u) rE[PB ^ 1][nt][k] = t[k];
            }
            const uint16_t* xb = xa + PB * XE;
            const float* xg = xgl + PB * XGE;
            gg_v8f P[NM][NT];
            r4d_u32x4 af[2][NM];
            gg_v8f ci[NM];
#pragma unroll
            for (int mt = 0; mt < NM; ++mt) {
                if (live(mt)) {
                    af[0][mt] = *(const r4d_u32x4*)(const void*)(xb + mt * 16 * kWmXP + W::xoff(0, h));
                    if constexpr (W::CINIT && W::PIPE) {
                        if constexpr (NOCI) ci[mt] = (gg_v8f)(0.0f);
                        else ci[mt] = *(const gg_v8f*)(const void*)(xg + mt * 16);
                    }
                }
            }
            gg_for<8>([&](auto jc) {
                constexpr int J = decltype(jc)::value, C = J & 1;
                /* (a) */
                if constexpr (!W::PIPE) {
                    if constexpr (J < 7) {
#pragma unroll
                        for (int mt = 0; mt < NM; ++mt)
                            if (live(mt))
                                af[C ^ 1][mt] = *(const r4d_u32x4*)(const void*)(xb + mt * 16 * kWmXP + W::xoff(J + 1, h));
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) bf[C][nt] = bfr(jc, rc[nt], pf[nt], s);
                } else if constexpr (J < 7) {
#pragma unroll
                    for (int mt = 0; mt < NM; ++mt)
                        if (live(mt))
                            af[C ^ 1][mt] = *(const r4d_u32x4*)(const void*)(xb + mt * 16 * kWmXP + W::xoff(J + 1, h));
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) W::template pre<J + 1>(rc[nt], h, gg_smem, pf[nt]);
                } else {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                        for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[PB ^ 1][nt][k] : rR[nt][k];
                        W::template pre<0>(rc[nt], h, gg_smem, pf[nt]);
                    }
                }
                gg_v8f g8[NM];
                if constexpr (W::MINS && W::PIPE && J % W::SPG == W::SPG - 1) {
#pragma unroll
                    for (int mt = 0; mt < NM; ++mt)
                        if (live(mt)) {
                            if constexpr (NOCI) g8[mt] = (gg_v8f)(0.0f);
                            else g8[mt] = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM + mt * 16);
                        }
                }
                constexpr bool FOLD = !NOFMA && J % W::SPG == W::SPG - 1;   /* this step closes a scale group */
                float sc[NT], mn[NT];
                /* The group's scales (and mins): the stock's (e) expressions on the same words. */
                auto scales = [&]() {
                    if constexpr (FOLD) {
                        if constexpr (J < 7 || !W::PIPE) {
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) {
                                sc[nt] = W::scale(rc[nt], J / W::SPG, s);
                                if constexpr (W::MINS) mn[nt] = W::mins(rc[nt], J / W::SPG, s);
                            }
                        } else {   /* rc is the next slab's now: this slab's words */
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) {
                                uint32_t ro[W::RAW];
#pragma unroll
                                for (int k = 0; k < W::RAW; ++k) ro[k] = ((W::EARLY >> k) & 1u) ? rE[PB][nt][k] : rR[nt][k];
                                sc[nt] = W::scale(ro, J / W::SPG, s);
                                if constexpr (W::MINS) mn[nt] = W::mins(ro, J / W::SPG, s);
                            }
                        }
                    }
                };
                /* One row tile's fold: acc = fma(P, scale, acc) (and Q5_K's acc = fma(groupsum, min, acc)), per output in
                 * the stock's order. */
                auto fold = [&](int mt) {
                    if (!live(mt)) return;
                    if constexpr (W::MINS && !W::PIPE) {
                        if constexpr (NOCI) g8[mt] = (gg_v8f)(0.0f);
                        else g8[mt] = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM + mt * 16);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            acc[mt][nt][q] = __builtin_fmaf(P[mt][nt][q], sc[nt], acc[mt][nt][q]);
                            if constexpr (W::MINS) acc[mt][nt][q] = __builtin_fmaf(g8[mt][q], mn[nt], acc[mt][nt][q]);
                        }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) gg_pin8(acc[mt][nt]);
                };
                /* (b) one row tile's WMMAs */
                auto mma = [&](int mt) {
                    if (live(mt)) {
                        if constexpr (NOFMA) {   /* ablation: one chain a slab, no per-group fold */
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt)
                                P[mt][nt] = gg_wmma(af[C][mt], bf[C][nt], J == 0 ? acc[mt][nt] : P[mt][nt]);
                        } else if constexpr (J % W::SPG == 0) {
                            gg_v8f c0 = (gg_v8f)(0.0f);
                            if constexpr (W::CINIT && W::PIPE) c0 = ci[mt];
                            else if constexpr (W::CINIT && !NOCI) c0 = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM + mt * 16);
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) P[mt][nt] = gg_wmma(af[C][mt], bf[C][nt], c0);
                        } else {
#pragma unroll
                            for (int nt = 0; nt < NT; ++nt) P[mt][nt] = gg_wmma(af[C][mt], bf[C][nt], P[mt][nt]);
                        }
                    }
                };
                /* (c) */
                auto cnext = [&]() {
                    if constexpr (W::CINIT && W::PIPE && J < 7 && (J + 1) % W::SPG == 0) {
#pragma unroll
                        for (int mt = 0; mt < NM; ++mt)
                            if (live(mt)) {
                                if constexpr (NOCI) ci[mt] = (gg_v8f)(0.0f);
                                else ci[mt] = *(const gg_v8f*)(const void*)(xg + ((J + 1) * 16 / W::XG) * BM + mt * 16);
                            }
                    }
                };
                /* (d) */
                auto bnext = [&]() {
                    if constexpr (W::PIPE) {
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                            bf[C ^ 1][nt] = bfr(std::integral_constant<int, (J + 1) & 7>{}, rc[nt], pf[nt], J < 7 ? s : sn);
                    }
                };
                if constexpr (!FI) {
                    /* the stock's order: every row tile's WMMAs, (c), (d), then the scales and every tile's fold */
#pragma unroll
                    for (int mt = 0; mt < NM; ++mt) mma(mt);
                    cnext();
                    bnext();
                    if constexpr (FOLD) {
                        scales();
#pragma unroll
                        for (int mt = 0; mt < NM; ++mt) fold(mt);
                    } else if constexpr (NOFMA && J == 7) {
#pragma unroll
                        for (int mt = 0; mt < NM; ++mt)
                            if (live(mt))
#pragma unroll
                                for (int nt = 0; nt < NT; ++nt) acc[mt][nt] = P[mt][nt];
                    }
                } else {
                    /* FI: row tile mt's WMMAs, then tile mt - 1's fold (its WMMAs a tile back), so a step keeps two tiles'
                     * partials live instead of NM; the last tile's fold after (c) and (d). Each output's operations are
                     * the same; only the order across outputs changes. */
                    scales();
#pragma unroll
                    for (int mt = 0; mt < NM; ++mt) {
                        mma(mt);
                        if constexpr (FOLD) {
                            if (mt > 0) fold(mt - 1);
                        }
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cnext();
                    bnext();
                    if constexpr (FOLD) fold(NM - 1);
                    if constexpr (NOFMA && J == 7) {
#pragma unroll
                        for (int mt = 0; mt < NM; ++mt)
                            if (live(mt))
#pragma unroll
                                for (int nt = 0; nt < NT; ++nt) acc[mt][nt] = P[mt][nt];
                    }
                }
                if constexpr (W::ROLL) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) W::template reload<J>(wr[nt], sn, h, rR[nt]);
                }
                if constexpr (J == 3) {
                    xstore(PB ^ 1);
                    xload(s + 2 < S ? s + 2 : S - 1);
                }
                __builtin_amdgcn_sched_barrier(0);
            });
            lds_barrier();
        };
        for (int s = 0; s < S; s += 2) {   /* S is even (K % 256 == 0) */
            slab(std::integral_constant<int, 0>{}, s);
            slab(std::integral_constant<int, 1>{}, s + 1);
        }
        gg_for<MT>([&](auto mc) {
            constexpr int mt = decltype(mc)::value;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int n = n0 + wave * WC + nt * 16 + c16;
                if (n >= A.N) continue;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int lr = mt * 16 + 8 * h + q, row = r0 + lr;
                    if (row >= r1) continue;
                    float v = 0.0f;
                    if constexpr (mt < NM) v = acc[mt][nt][q];
                    gg_put((OT*)A.y + (long long)row * A.y_ld + n, v);
                }
            }
        });
    };
    if constexpr (W::GUARD) {
        run(std::integral_constant<int, MT>{});
    } else {
        gg_for<MT>([&](auto ic) {
            constexpr int NM = decltype(ic)::value + 1;
            if (nmt == NM) run(std::integral_constant<int, NM>{});
        });
    }
}

/* LDS of a form: the table, the two staged slabs, their group sums (gg_wm_lds at BM = 16 MT). */
template <int T>
constexpr unsigned pf_lds(int mt) {
    return (unsigned)(Wq<T>::TB + 2 * 16 * mt * kWmXP * 2 + (Wq<T>::XG > 0 ? 2 * (kWmS / Wq<T>::XG) * 16 * mt * 4 : 0));
}

/* ---- the forms built: (MT, NT, WN, FI) for every type, and the ablations on the stock geometry. A form's code is
 * 100 MT + 10 NT + WN / 4 + 4 FI (322 = the stock geometry and order, 326 the same with the folds interleaved), plus
 * 1000 ABL for an ablation. (6.11 compile: MT 4 NT 2 WN 4, MT 2 NT 4 WN 4 and MT 3 NT 4 WN 4 spill 9-480 registers in
 * the stock order; not built.) */
#define PF_FORMS(X) X(3, 2, 8, 0) X(4, 2, 8, 0) X(2, 2, 8, 0) X(3, 2, 4, 0) X(2, 4, 8, 0) X(3, 2, 8, 1) X(4, 2, 8, 1) X(2, 4, 8, 1)
#define PF_ABLS(X) X(1) X(2) X(4)

constexpr int pf_code(int mt, int nt, int wn, int fi, int abl) { return 1000 * abl + 100 * mt + 10 * nt + wn / 4 + 4 * fi; }

bool pf_form_known(int code) {
    const int abl = code / 1000, g = code % 1000;
#define PF_K(mt, nt, wn, fi) if (abl == 0 && g == pf_code(mt, nt, wn, fi, 0)) return true;
    PF_FORMS(PF_K)
#undef PF_K
#define PF_KA(a) if (abl == a && g == pf_code(3, 2, 8, 0, 0)) return true;
    PF_ABLS(PF_KA)
#undef PF_KA
    return false;
}

template <int T, int MT, int NT, int WN, int FI, int ABL>
int pf_launch_f(RadStream s, bool f32out, const GgTiledArgs& A) {
    const dim3 grid((unsigned)((A.M + 16 * MT - 1) / (16 * MT)), (unsigned)((A.N + 16 * NT * WN - 1) / (16 * NT * WN)));
    if (grid.y > 65535u) return R4D_NO(RAD_E_SHAPE);
    const unsigned lds = pf_lds<T>(MT);
    if (f32out) return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, float, ABL, FI>), grid, dim3(32 * WN), lds, A);
    return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, uint16_t, ABL, FI>), grid, dim3(32 * WN), lds, A);
}

template <int T>
int pf_launch_t(RadStream s, int code, bool f32out, const GgTiledArgs& A) {
#define PF_L(mt, nt, wn, fi) if (code == pf_code(mt, nt, wn, fi, 0)) return pf_launch_f<T, mt, nt, wn, fi, 0>(s, f32out, A);
    PF_FORMS(PF_L)
#undef PF_L
#ifndef R4D_EMU_HOST   /* the ablations: card timing only */
#define PF_LA(a) if (code == pf_code(3, 2, 8, 0, a)) return pf_launch_f<T, 3, 2, 8, 0, a>(s, f32out, A);
    PF_ABLS(PF_LA)
#undef PF_LA
#endif
    return R4D_GG_PF_DECLINE;
}

int pf_launch(RadStream s, int ty, int code, bool f32out, const GgTiledArgs& A) {
    switch (ty) {
    case GG_Q8_0: return pf_launch_t<GG_Q8_0>(s, code, f32out, A);
    case GG_Q5_K: return pf_launch_t<GG_Q5_K>(s, code, f32out, A);
    case GG_Q6_K: return pf_launch_t<GG_Q6_K>(s, code, f32out, A);
    default: return R4D_GG_PF_DECLINE;
    }
}

/* ---- the rule: a (type, N, K) -> form table. RADIANCE_DS41_GG_PF_TABLE at load ("q6_K:8192x1280=422,..." -- the
 * bench's GPTAB line; an entry naming an unknown form is skipped, named on stderr) overrides the built-in one, which
 * is empty until a bench has been read: a shape with no entry runs the stock kernel. */
struct PfEnt { int ty, N, K, code; };
const PfEnt kPfBuiltin[] = { { -1, 0, 0, 0 } };

std::vector<PfEnt> pf_table_env() {
    std::vector<PfEnt> t;
    const char* v = std::getenv("RADIANCE_DS41_GG_PF_TABLE");
    if (!v || !*v) return t;
    std::string s(v);
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        const std::string e = s.substr(p, q - p);
        p = q + 1;
        char tn[16] = {0};
        int N = 0, K = 0, code = 0;
        if (std::sscanf(e.c_str(), "%15[^:]:%dx%d=%d", tn, &N, &K, &code) != 4 || !pf_form_known(code) || code >= 1000) {
            std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_GG_PF_TABLE entry '%s' skipped (type:NxK=form)\n", e.c_str());
            continue;
        }
        const int ty = !std::strcmp(tn, "q8_0") ? GG_Q8_0 : !std::strcmp(tn, "q5_K") ? GG_Q5_K : !std::strcmp(tn, "q6_K") ? GG_Q6_K : -1;
        if (ty < 0) {
            std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_GG_PF_TABLE entry '%s' skipped (type)\n", e.c_str());
            continue;
        }
        t.push_back({ ty, N, K, code });
    }
    std::fprintf(stderr, "radiance: libr4d: lever GPF form table from RADIANCE_DS41_GG_PF_TABLE: %zu entries\n", t.size());
    return t;
}
const std::vector<PfEnt>& pf_table() {
    static const std::vector<PfEnt> t = pf_table_env();
    return t;
}
int pf_rule(int ty, int N, int K) {
    const std::vector<PfEnt>& t = pf_table();
    if (!t.empty()) {
        for (const PfEnt& e : t)
            if (e.ty == ty && e.N == N && e.K == K) return e.code;
        return 0;
    }
    for (const PfEnt& e : kPfBuiltin)
        if (e.ty == ty && e.N == N && e.K == K) return e.code;
    return 0;
}

bool pf_value_known(int v) { return v == 0 || v == 1 || (v < 1000 && pf_form_known(v)); }

int pf_env() {
    const char* v = std::getenv("RADIANCE_DS41_GG_PF");
    if (!v || !*v) return 0;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (*end || !pf_value_known((int)x)) {
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_GG_PF=%s is not 0 / 1 / a form code: the stock prefill WMMA kernel\n", v);
        return 0;
    }
    if (x)
        std::fprintf(stderr, "radiance: libr4d: gemm_nt_ggml's prefill rows take lever GPF's forms (RADIANCE_DS41_GG_PF=%ld, the fn4 switch gg_pf)\n", x);
    return (int)x;
}
std::atomic<int> g_pf_on{ pf_env() };
int g_pf_pin = 0;   /* bench pin: a form code (ablations included), -1 the rule; 0 none */
char g_pf_last[96];

}  // namespace

int r4d_gg_pf_on() { return g_pf_pin != 0 ? 1 : (g_pf_on.load(std::memory_order_relaxed) != 0); }
int r4d_gg_pf_known(int v) { return pf_value_known(v) ? 1 : 0; }
int r4d_gg_pf_set(int v) {
    if (!pf_value_known(v)) return -1;
    g_pf_on.store(v, std::memory_order_relaxed);
    return 1;
}
int r4d_gg_pf_get() { return g_pf_on.load(std::memory_order_relaxed); }

int r4d_gg_pf_gemm(RadStream s, const R4dGgPfCall& c) {
    if (c.ty != GG_Q5_K && c.ty != GG_Q6_K && c.ty != GG_Q8_0) return R4D_GG_PF_DECLINE;
    if (c.M < 1 || c.N < 1 || c.K < 2 * kWmS || c.K % (2 * kWmS)) return R4D_GG_PF_DECLINE;
    int code = g_pf_pin != 0 ? g_pf_pin : g_pf_on.load(std::memory_order_relaxed);
    if (code == -1 || code == 1) code = pf_rule(c.ty, c.N, c.K);
    if (code == 0 || !pf_form_known(code)) return R4D_GG_PF_DECLINE;
    GgTiledArgs A{};
    A.x = c.x; A.x_ld = c.x_ld;
    A.w = c.w; A.wtab = nullptr; A.w_ld = c.w_ld; A.w_es = 0;
    A.y = c.y; A.y_ld = c.y_ld;
    A.M = c.M; A.N = c.N; A.K = c.K;
    A.x_al = 1; A.tab_bytes = 0;
    std::snprintf(g_pf_last, sizeof g_pf_last, "form %d", code);
    return pf_launch(s, c.ty, code, c.f32out, A);
}

/* r4d_gg_pf_pin (bench / test hook, r4d_selftest --gpcheck / --perfgp): 0 clears the pin (the runtime switch decides,
 * off by default), -1 forces the rule, a form code (ablations included) pins that form for every shape. */
extern "C" void r4d_gg_pf_pin(int code) { g_pf_pin = (code == -1 || pf_form_known(code)) ? code : 0; }
/* The last launch's form ("form 422"), for the bench's lines. */
extern "C" const char* r4d_gg_pf_last() { return g_pf_last; }
/* The forms this build has, as codes (the bench iterates them): writes up to `cap`, returns the count. */
extern "C" int r4d_gg_pf_forms(int* out, int cap) {
    int n = 0;
#define PF_C(mt, nt, wn, fi) if (n < cap) out[n] = pf_code(mt, nt, wn, fi, 0); ++n;
    PF_FORMS(PF_C)
#undef PF_C
#define PF_CA(a) if (n < cap) out[n] = pf_code(3, 2, 8, 0, a); ++n;
    PF_ABLS(PF_CA)
#undef PF_CA
    return n;
}

#ifdef R4D_EMU_HOST
/* tests/r4d_emu (ds41pf): the stock WMMA kernel's launch as gg_gemm issues it (gg_launch_wm_t), from this unit's
 * copy of gg_wm_kernel -- the same source as the plugin's. */
template <int T>
int pf_emu_stock_t(const R4dGgPfCall& c, const GgTiledArgs& A) {
    constexpr int MT = 3;
    const dim3 grid((unsigned)((c.M + 47) / 48), (unsigned)((c.N + kWmBN - 1) / kWmBN));
    const unsigned lds = pf_lds<T>(MT);
    if (c.f32out) return R4D_EMIT(nullptr, (gg_wm_kernel<T, MT, false, float>), grid, dim3(256), lds, A);
    return R4D_EMIT(nullptr, (gg_wm_kernel<T, MT, false, uint16_t>), grid, dim3(256), lds, A);
}
int r4d_gg_pf_emu_stock(const R4dGgPfCall& c) {
    GgTiledArgs A{};
    A.x = c.x; A.x_ld = c.x_ld;
    A.w = c.w; A.wtab = nullptr; A.w_ld = c.w_ld; A.w_es = 0;
    A.y = c.y; A.y_ld = c.y_ld;
    A.M = c.M; A.N = c.N; A.K = c.K;
    A.x_al = 1; A.tab_bytes = 0;
    switch (c.ty) {
    case GG_Q5_K: return pf_emu_stock_t<GG_Q5_K>(c, A);
    case GG_Q6_K: return pf_emu_stock_t<GG_Q6_K>(c, A);
    case GG_Q8_0: return pf_emu_stock_t<GG_Q8_0>(c, A);
    default: return -1;
    }
}
#endif
