#pragma once
// r4d_ar_arc.h: lever ARC (ds41-int10, libr4d fn4 "ds41_arcoded") -- which launch of the four-rank exact two-shot
// (row ar_twoshot_4rank_exact, r4d_rad_ar_twoshot_4rank_exact) takes the coded wire, and at what grid. Host code only,
// plain C++, so tests/r4d_emu holds the rule itself (test_ds41arc.cpp).
//
// The coded wire (r4d_ar_coded.h, r4d_ar_twoshot_coded_body) sends ~12.2 bits a bf16 value instead of 16 and decodes the
// same values, so the reduce folds the same numbers in the same order: the plain wire's bytes. Until ds41-int10 only a
// launch at `overlap` 1 (the lane row, a reduction beside compute) took it, at 4 x 1024; deepseek41 declares its
// all-reduces without overlap, so it always ran the plain wire.
//
// ds41_arcoded N (0 = off, the plain wire exactly as before; 1 .. R4D_ARC_MAX): a launch at overlap 0 whose operand has
// at least N rows takes the coded wire, where the coded form fits the message (r4d_ar_twoshot_coded_ok: bf16, shards of
// whole 256-value waves, two coded regions in a slot) and no `coded` 0 launch parameter asks for the plain one; the rest
// stays plain. Rows = the operand's elements over its last axis (deepseek41's all-reduces carry [rows, 5120] slices; a
// rank-1 operand counts as one row). Every rank sees the same message and the same switch (fn4 applies a change while
// every rank thread is parked), so every rank takes the same wire.
//
// The grid (the ARW item, gpu02 12.11.2026, runs/xdsm/arw/arwsum.txt: libr4d's coded launcher against the plain row at T
// rows x 5120 bf16, activation-like data, 100 isolated launches a variant, medians; 189 byte comparisons, 0 differences):
//   T 1100: plain 956.0 us; coded C8 793.3 (-17.0 %), C16 -15.9 %, C24 -15.2 %, the lane's 4 x 1024 -8.8 %
//   T 1500: plain 1312.4 us; C16 1086.2 (-17.2 %), C8 -16.5 %, C24 -16.0 %
//   T 2048: plain 1724.8 us; C16 1446.0 (-16.2 %), C8 -15.9 %, C24 -14.7 %
//   T 4096: plain 3376.9 us; C24 2869.6 (-15.0 %), C16 -14.3 %, C8 -12.9 %
//   decode sizes (T 8 / 16 / 48) coded SLOWER at every grid (+2.3 .. +152.8 %): the default threshold keeps them plain
//   (deepseek41's decode all-reduces take the one-shot row anyway, below 524288 elements).
// So 8 blocks below 1300 rows' worth of elements, 16 below 3072 rows' worth, 24 from there (1024 threads a block, the
// coded launcher's form); never past the instance's block capacity.
#include <cstdint>

enum {
    R4D_ARC_MAX = 1048576,        /* the largest threshold ds41_arcoded takes */
    R4D_ARC_DEFAULT = 1024,       /* the screen's ON value: coded from 1024 rows (ARW: -17 % at 1100, no point below) */
    R4D_ARC_THREADS = 1024,       /* threads a block (the coded launcher's form; below 1024 it codes too slowly) */
};
/* Element bounds of the grid rule: 1300 and 3072 rows of 5120 (deepseek41's residual width). */
static constexpr long long kArcGrid8Below = 1300LL * 5120LL;    /* 6656000 */
static constexpr long long kArcGrid16Below = 3072LL * 5120LL;   /* 15728640 */

/* A switch value the plugin takes: 0 (off) or 1 .. R4D_ARC_MAX. */
static inline bool r4d_arc_known(long long v) { return v >= 0 && v <= R4D_ARC_MAX; }

/* The operand's rows: its elements over its last axis; a rank-1 (or empty-axis) operand is one row. */
static inline long long r4d_arc_rows(long long numel, int rank, long long last) {
    return rank >= 2 && last > 0 ? numel / last : 1;
}

/* The coded launch's block count for a message of `numel` elements (the ARW table above). */
static inline int r4d_arc_grid(long long numel) {
    if (numel < kArcGrid8Below) return 8;
    if (numel < kArcGrid16Below) return 16;
    return 24;
}

/* The wire a launch at overlap 0 takes: 0 = plain, else the coded launch's block count. thr = ds41_arcoded (0 off),
 * coded_ok = r4d_ar_twoshot_coded_ok for the message and the instance's slot, coded_param = the `coded` launch parameter
 * (absent = 1), max_blocks = the instance's block capacity (flags / counters sized for it; R4D_AR_MAX_BLOCKS lowers it). */
static inline int r4d_arc_pick(long long thr, long long rows, long long numel, int coded_ok, long long coded_param,
                               int max_blocks) {
    if (thr <= 0 || rows < thr || !coded_ok || coded_param == 0 || max_blocks < 1) return 0;
    const int nb = r4d_arc_grid(numel);
    return nb < max_blocks ? nb : max_blocks;
}
