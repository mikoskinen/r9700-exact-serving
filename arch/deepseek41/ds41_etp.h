/* ds41_etp.h -- lever 1U-ETP's slice arithmetic in deepseek41 (expert tensor parallel routed experts,
 * RADIANCE_DS41_ETP), pure host arithmetic so tests/ds41_etp_test.cpp checks it without a card.
 *
 *   etp_slice   this rank's columns [lo[q], hi[q]) of a routed expert of parity class q (0 even, 1 odd): the
 *               intermediate width's nb 128-blocks split over the ranks, the even experts giving the first
 *               nb % world ranks one block more and the odd experts the last nb % world ranks, so every rank holds
 *               nb / world blocks on average over an even / odd pair. DeepSeek-V4.1-Flash at four ranks: 2304 =
 *               18 blocks -> even 5/5/4/4 (640 / 640 / 512 / 512 wide), odd 4/4/5/5.
 *
 * The block (rad_block_moe_fp8.h, declare_exl3) declares class q's gate_up rows [lo, hi) of the gate half and the same
 * rows of the up half, and down's K columns [lo, hi), for every expert of that parity. */
#ifndef RAD_ARCH_DS41_ETP_H
#define RAD_ARCH_DS41_ETP_H

#include <cstdint>

namespace rad {
namespace arch {
namespace ds41etp {

constexpr int64_t kBlock = 128;

/* false where the width is no whole number of 128-blocks, there are fewer blocks than ranks, or one rank */
inline bool etp_slice(int64_t n_ff, int world, int rank, int64_t lo[2], int64_t hi[2]) {
    if (world < 2 || rank < 0 || rank >= world || n_ff <= 0 || n_ff % kBlock || n_ff / kBlock < world) return false;
    const int64_t nb = n_ff / kBlock, base = nb / world, extra = nb % world;
    for (int q = 0; q < 2; ++q) {
        int64_t at = 0, mine = 0;
        for (int r = 0; r <= rank; ++r) {
            const int64_t n = base + ((q == 0 ? r < extra : r >= world - extra) ? 1 : 0);
            if (r == rank) mine = n;
            else at += n;
        }
        lo[q] = at * kBlock;
        hi[q] = (at + mine) * kBlock;
    }
    return true;
}

}  // namespace ds41etp
}  // namespace arch
}  // namespace rad

#endif  // RAD_ARCH_DS41_ETP_H
