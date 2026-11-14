/* tests/ds41_etp_test.cpp -- lever 1U-ETP (ds41-etp): DeepSeek-V4.1's routed EXL3 experts as width slices of every expert
 * on every rank, host arithmetic only:
 *
 *   slices        arch/deepseek41/ds41_etp.h etp_slice: 2304 columns at four ranks are 5/5/4/4 128-blocks for the even
 *                 experts and 4/4/5/5 for the odd ones; every class's slices tile the width once; refusals.
 *   shares        the loader's cut (core/format/share.cpp weight_share) of each plane the block declares
 *                 (rad_block_moe_fp8.h declare_exl3 under ETP), on the real canonical EXL3 encodings at K 2 (u2 [1x1])
 *                 and K 2.5 (u5 pair symbols [1x2]): gate_up's codes and svh = rows [lo, hi) of the gate half and the
 *                 same rows of the up half; suh / suh.1 whole; down's codes = columns [lo, hi) of every row on whole
 *                 bytes (pair symbols at K 2.5); down's suh = [lo, hi); down's svh whole.
 *   symbols       a random canonical plane cut by those rectangles holds, symbol for symbol, the slice an independent
 *                 bit reader takes from the whole plane (row j of the slice = row lo + j of the gate / up half;
 *                 column j = column lo + j), at both K. */
#include "rad_test.h"
#include "format/share.h"
#include "../arch/deepseek41/ds41_etp.h"
#include "../arch/common/exl3/exl3_encoding.h"

#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace rad;
using rad::arch::ds41etp::etp_slice;

namespace {

constexpr int64_t kD = 5120, kF = 2304;

RadEncoding exl3_enc(int bits, bool stacked) {
    RadEncoding e;
    if (exl3_encoding_make(bits, EXL3_CB_MUL1, stacked ? 1 : 0, &e) != RAD_OK) rad_enc_clear(&e);
    return e;
}

/* A single-plane view as declare_exl3 declares it under ETP: plane `role` of the entry, declared [rows, cols] (this
 * rank's), the shard, gate_up's two stacked parts, the span in the plane's units. */
WeightInfo view(const RadEncoding& e, const char* role, int64_t rows, int64_t cols, int shard, int64_t parts,
                int64_t lo, int64_t hi) {
    WeightInfo w;
    w.enc = e;
    w.enc_known = true;
    w.decl.dtype = e.plane[rad_enc_find(&e, role)].dtype;
    w.decl.rank = 2;
    w.decl.shape[0] = rows;
    w.decl.shape[1] = cols;
    w.decl.shard = shard;
    if (parts > 0) {
        w.decl.n_row_parts = 2;
        w.decl.row_parts[0] = w.decl.row_parts[1] = parts;
    }
    if (shard != RAD_SHARD_NONE) {
        w.shard_lo = lo;
        w.shard_hi = hi;
    }
    w.n_sel = 1;
    w.sel[0] = rad_enc_find(&e, role);
    w.sel_rows[0] = rows;
    w.sel_cols[0] = cols;
    return w;
}

int share(const WeightInfo& w, int64_t R, int64_t C, int rank, PlaneShare* out, std::string* why) {
    const int64_t shape[2] = { R, C };
    return weight_share(w, w.enc, 2, shape, 4, rank, out, why);
}

/* An independent reader of a canonical codes row: symbol j at bits [floor(j K2 / 2), floor((j + 1) K2 / 2)),
 * LSB-first (exl3_decode.h, the codes plane). */
uint32_t sym(const uint8_t* row, int k2, int64_t j) {
    const int64_t o = j * k2 / 2, n = (j + 1) * k2 / 2 - o;
    uint32_t v = 0;
    for (int64_t b = 0; b < n; ++b) v |= (uint32_t)((row[(o + b) >> 3] >> ((o + b) & 7)) & 1u) << b;
    return v;
}

/* the share's bytes out of a dense canonical plane, as the loader lays them end to end */
std::vector<uint8_t> cut(const std::vector<uint8_t>& plane, const PlaneShare& s) {
    std::vector<uint8_t> out;
    for (const PlaneShare::Rect& q : s.rects)
        for (int64_t r = 0; r < q.rows; ++r) {
            const uint8_t* p = plane.data() + (q.row0 + r) * s.row_bytes + q.byte0;
            out.insert(out.end(), p, p + q.bytes);
        }
    return out;
}

}  // namespace

TEST(etp_slices_are_5544_even_and_4455_odd_and_tile_the_width) {
    const int64_t want_lo[2][4] = { { 0, 640, 1280, 1792 }, { 0, 512, 1024, 1664 } };
    const int64_t want_hi[2][4] = { { 640, 1280, 1792, 2304 }, { 512, 1024, 1664, 2304 } };
    for (int r = 0; r < 4; ++r) {
        int64_t lo[2], hi[2];
        REQUIRE(etp_slice(kF, 4, r, lo, hi));
        for (int q = 0; q < 2; ++q) {
            CHECK_EQ(lo[q], want_lo[q][r]);
            CHECK_EQ(hi[q], want_hi[q][r]);
        }
        /* an even / odd pair is 9 blocks on every rank */
        CHECK_EQ((hi[0] - lo[0]) + (hi[1] - lo[1]), (int64_t)(9 * 128));
    }
    for (int world = 2; world <= 8; ++world)
        for (int q = 0; q < 2; ++q) {
            int64_t at = 0;
            for (int r = 0; r < world; ++r) {
                int64_t lo[2], hi[2];
                REQUIRE(etp_slice(kF, world, r, lo, hi));
                CHECK_EQ(lo[q], at);
                CHECK(hi[q] > lo[q]);
                CHECK_EQ((hi[q] - lo[q]) % 128, (int64_t)0);
                at = hi[q];
            }
            CHECK_EQ(at, kF);
        }
    int64_t lo[2], hi[2];
    CHECK(!etp_slice(kF, 1, 0, lo, hi));          /* one rank: nothing to split */
    CHECK(!etp_slice(2300, 4, 0, lo, hi));        /* not whole blocks */
    CHECK(!etp_slice(384, 4, 0, lo, hi));         /* fewer blocks than ranks */
    CHECK(!etp_slice(kF, 4, 4, lo, hi));          /* no such rank */
}

/* The rectangles of every plane declare_exl3 declares, at K 2 and K 2.5, every rank and class. */
TEST(etp_shares_cut_gate_up_rows_and_down_columns_on_whole_bytes) {
    for (int bits : { 2, EXL3_HALF(2) }) {
        const RadEncoding gu = exl3_enc(bits, true), dn = exl3_enc(bits, false);
        REQUIRE(gu.n_planes == 4);
        REQUIRE(dn.n_planes == 3);
        const int64_t cpe = exl3_bits_half(bits) ? 2 : 1, k2 = exl3_k2(bits);
        for (int r = 0; r < 4; ++r) {
            int64_t lo[2], hi[2];
            REQUIRE(etp_slice(kF, 4, r, lo, hi));
            for (int q = 0; q < 2; ++q) {
                const int64_t W = hi[q] - lo[q];
                PlaneShare s[RAD_ENC_MAX_PLANES];
                std::string why;
                /* gate_up codes: rows of both halves, every byte of each row */
                REQUIRE_EQ(share(view(gu, "codes", 2 * W, kD / cpe, RAD_SHARD_ROW, W, lo[q], hi[q]), 2 * kF, kD, r, s,
                                 &why), RAD_OK);
                REQUIRE_EQ(s[0].rects.size(), (size_t)2);
                CHECK_EQ(s[0].rects[0].row0, lo[q]);
                CHECK_EQ(s[0].rects[1].row0, kF + lo[q]);
                CHECK_EQ(s[0].rects[0].rows, W);
                CHECK_EQ(s[0].rects[0].byte0, (int64_t)0);
                CHECK_EQ(s[0].rects[0].bytes, s[0].row_bytes);
                CHECK_EQ(s[0].row_bytes, kD * k2 / 16);
                CHECK_EQ(s[0].rows, 2 * W);
                /* gate_up svh: the same rows, one f16 each */
                REQUIRE_EQ(share(view(gu, "svh", 2 * W, 1, RAD_SHARD_ROW, W, lo[q], hi[q]), 2 * kF, kD, r, s, &why),
                           RAD_OK);
                REQUIRE_EQ(s[0].rects.size(), (size_t)2);
                CHECK_EQ(s[0].rects[0].row0, lo[q]);
                CHECK_EQ(s[0].rects[1].row0, kF + lo[q]);
                CHECK_EQ(s[0].rects[1].bytes, (int64_t)2);
                /* gate_up suh / suh.1: whole */
                for (const char* role : { "suh", "suh.1" }) {
                    REQUIRE_EQ(share(view(gu, role, 1, kD, RAD_SHARD_NONE, 0, 0, 0), 2 * kF, kD, r, s, &why), RAD_OK);
                    REQUIRE_EQ(s[0].rects.size(), (size_t)1);
                    CHECK_EQ(s[0].rects[0].bytes, 2 * kD);
                }
                /* down codes: K columns [lo, hi) of every row, on whole bytes (pair symbols at K 2.5) */
                REQUIRE_EQ(share(view(dn, "codes", kD, W / cpe, RAD_SHARD_COL, 0, lo[q] / cpe, hi[q] / cpe), kD, kF, r, s,
                                 &why), RAD_OK);
                REQUIRE_EQ(s[0].rects.size(), (size_t)1);
                CHECK_EQ(s[0].rects[0].row0, (int64_t)0);
                CHECK_EQ(s[0].rects[0].rows, kD);
                CHECK_EQ(s[0].rects[0].byte0, lo[q] * k2 / 16);
                CHECK_EQ(s[0].rects[0].bytes, W * k2 / 16);
                CHECK_EQ(s[0].row_bytes, kF * k2 / 16);
                /* down suh: [lo, hi); svh: whole */
                REQUIRE_EQ(share(view(dn, "suh", 1, W, RAD_SHARD_COL, 0, lo[q], hi[q]), kD, kF, r, s, &why), RAD_OK);
                REQUIRE_EQ(s[0].rects.size(), (size_t)1);
                CHECK_EQ(s[0].rects[0].byte0, 2 * lo[q]);
                CHECK_EQ(s[0].rects[0].bytes, 2 * W);
                REQUIRE_EQ(share(view(dn, "svh", kD, 1, RAD_SHARD_NONE, 0, 0, 0), kD, kF, r, s, &why), RAD_OK);
                CHECK_EQ(s[0].rects[0].rows, kD);
                CHECK_EQ(s[0].rects[0].bytes, (int64_t)2);
                /* a span the declaration does not match is refused, not read */
                CHECK_EQ(share(view(dn, "codes", kD, W / cpe + 64, RAD_SHARD_COL, 0, lo[q] / cpe, hi[q] / cpe), kD, kF, r,
                               s, &why), RAD_E_SHAPE);
            }
        }
    }
}

/* Symbol for symbol: a random canonical plane cut by the shares is the slice an independent reader takes. */
TEST(etp_cut_planes_hold_the_slice_symbol_for_symbol) {
    std::mt19937_64 rng(20261114);
    for (int bits : { 2, EXL3_HALF(2) }) {
        const RadEncoding gu = exl3_enc(bits, true), dn = exl3_enc(bits, false);
        const int64_t cpe = exl3_bits_half(bits) ? 2 : 1, k2 = exl3_k2(bits);
        const int64_t gu_rb = kD * k2 / 16, dn_rb = kF * k2 / 16;
        std::vector<uint8_t> gplane((size_t)(2 * kF * gu_rb)), dplane((size_t)(kD * dn_rb));
        for (uint8_t& b : gplane) b = (uint8_t)rng();
        for (uint8_t& b : dplane) b = (uint8_t)rng();
        int64_t bad = 0, seen = 0;
        for (int r = 0; r < 4; ++r) {
            int64_t lo[2], hi[2];
            REQUIRE(etp_slice(kF, 4, r, lo, hi));
            for (int q = 0; q < 2; ++q) {
                const int64_t W = hi[q] - lo[q];
                PlaneShare s[RAD_ENC_MAX_PLANES];
                std::string why;
                REQUIRE_EQ(share(view(gu, "codes", 2 * W, kD / cpe, RAD_SHARD_ROW, W, lo[q], hi[q]), 2 * kF, kD, r, s,
                                 &why), RAD_OK);
                const std::vector<uint8_t> g = cut(gplane, s[0]);
                REQUIRE_EQ((int64_t)g.size(), 2 * W * gu_rb);
                for (int64_t i = 0; i < 2 * W; i += 37) {   /* slice row i = gate / up row lo + i % W of its half */
                    const int64_t src = (i < W ? 0 : kF) + lo[q] + (i % W);
                    for (int64_t j = 0; j < kD; j += 13) {
                        ++seen;
                        bad += sym(g.data() + i * gu_rb, (int)k2, j) != sym(gplane.data() + src * gu_rb, (int)k2, j);
                    }
                }
                REQUIRE_EQ(share(view(dn, "codes", kD, W / cpe, RAD_SHARD_COL, 0, lo[q] / cpe, hi[q] / cpe), kD, kF, r, s,
                                 &why), RAD_OK);
                const std::vector<uint8_t> d = cut(dplane, s[0]);
                const int64_t srb = W * k2 / 16;
                REQUIRE_EQ((int64_t)d.size(), kD * srb);
                for (int64_t i = 0; i < kD; i += 29)        /* slice column j = column lo + j of the same row */
                    for (int64_t j = 0; j < W; ++j) {
                        ++seen;
                        bad += sym(d.data() + i * srb, (int)k2, j) != sym(dplane.data() + i * dn_rb, (int)k2, lo[q] + j);
                    }
            }
        }
        CHECK(seen > 100000);
        CHECK_EQ(bad, (int64_t)0);
    }
}

RAD_TEST_MAIN()
