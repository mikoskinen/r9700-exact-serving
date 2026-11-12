#pragma once
// r4d_ar_ll.h: lever A3 / 1U-AR (ds41-ll, libr4d fn4 "ar_ll") -- the four-rank exact all-reduce at decode sizes with
// its flags in its data words (LL, "low latency"), for the plain all-reduce (row ar_oneshot_4rank_exact, which hands
// off to the exact two-shot from ts_min) and moe_gather_all_reduce_4rank. Host code only, plain C++, so tests/r4d_emu
// holds the rule itself; the device side is r4d_ar_ll_dev.h.
//
// What changes is how the bytes and their flags travel, nothing else. Today a block pushes its words into each peer's
// scratch, drains its stores, stores a flag word into each peer and spins on its own flags (r4d_ar_handshake): the
// drain and the flag's own trip across the link sit between the last data word and the reduce, once a phase (the
// one-shot has one, the two-shot two). Here every 8-byte unit of the wire carries three bf16 values and a 16-bit tag
// (bits 48..63, never 0), stored with one system-scope 8-byte store; the reader spins on the units themselves (an
// 8-byte load is one copy of an 8-byte store) and reduces each group the moment its units are in. No flag, no drain.
// The cost is the wire: 8 bytes for 6 of data (4/3 of the plain bytes).
//
// The fold is the plain kernels' fold: per element 0.0f + r0 + r1 + r2 + r3 in fp32 (ascending global rank, this
// rank's own term from its input), rounded once to bf16 (r4d_ar_to_f / r4d_ar_from_f, ar_vsum's loop). The form is
// today's form: the one-shot below the tuned ts_min, the two-shot (reduce-scatter, then the reduced shard to every
// peer) from it, as r4d_rad_ar_oneshot_4rank_exact / r4d_rad_moe_gather_ar_4rank choose; the gather's words are
// moe_gather's (r4d_ar_gather_word_list, the two-shot gather's words). So a launch's bytes are the plain launch's.
//
// Where the units go. Each instance built while RADIANCE_AR_LL is set (any value) gets its own LL area on every rank:
// fine-grained VRAM, zeroed, written only by LL launches -- 2 parities x 4 sources x slot_units units. The slot parity is
// the launch's (r4d_ar_seq.h's parity counters, shared with every kernel on the instance), so launch L+1 never writes
// where launch L still reads. A reader CLEARS every unit it has consumed (a system-scope store of 0), so after each
// launch the whole area is zero again: a unit can only hold this launch's value or 0, whatever sizes and forms ran
// before. The 16-bit tag (1 + launch count mod 65535) is a second check on top. The next writer of a cleared unit is a
// peer's launch L+2, which starts only after it has seen this rank's L+1 bytes, which this rank sends only after its
// launch L (and its clears) completed.
//
// ar_ll (0 = off, today's wire exactly; 1 = LL for messages of at most R4D_ARLL_MAX_ROWS rows; N in 2 .. 48 = at most N
// rows): rows = the operand's elements over its last axis (deepseek41's [rows, 5120] slices), the gather's M. A launch
// takes LL only on an instance with an LL area, bf16, at most R4D_ARLL_MAX_ELEMS elements, where its form's units fit
// the slot (two-shot: whole words a rank); anything else runs the plain kernel ("declined" in the counters). Every rank
// sees the same message, the same switch (fn4 applies a change with every rank parked) and the same instance, so every
// rank takes the same wire.
#include <cstdint>

/* The two helpers the kernel body uses too (r4d_ar_ll_dev.h): host and device under HIP, plain C++ elsewhere. */
#if defined(__HIP__) || defined(__HIPCC__)
#define R4D_ARLL_HD __host__ __device__
#else
#define R4D_ARLL_HD
#endif

enum {
    R4D_ARLL_MAX_ROWS = 48,        /* ar_ll 1: LL up to 48 rows (decode and verify heights) */
    R4D_ARLL_MAX_ELEMS = 262144,   /* an LL message's element bound (51 rows of 5120); sizes the area */
    R4D_ARLL_THREADS = 128,        /* threads a block: one group (3 words, 8 units) a thread a phase */
    R4D_ARLL_MAX_BLOCKS = 32,      /* <= R4D_AR_WIDE_MAX_BLOCKS, whose parity words every launch advances */
    R4D_ARLL_UNITS = 8,            /* units a group */
    R4D_ARLL_WORDS = 3,            /* 16-byte words a group (24 bf16 = 8 units x 3) */
};
/* The gather's slot list (r4d_ar_gather_list, every row's live slots in a block's LDS): r4d_ar_hc.h's R4D_ARGL_ROWS /
 * R4D_ARGL_TAB, restated so host code needs no device header (r4d_ar_ll.hip asserts they agree). */
enum { R4D_ARLL_LIST_ROWS = 256, R4D_ARLL_LIST_TAB = 2048 };
static inline bool r4d_arll_list_ok(long long M, long long top_k) {
    return M >= 1 && M <= R4D_ARLL_LIST_ROWS && top_k >= 1 && M * top_k <= R4D_ARLL_LIST_TAB;
}
/* A launch's form: the one-shot (below ts_min) or the two-shot (from it). */
enum { R4D_ARLL_ONESHOT = 0, R4D_ARLL_TWOSHOT = 1 };

/* A switch value libr4d takes: 0 (off) .. 48. */
static inline bool r4d_arll_known(long long v) { return v >= 0 && v <= R4D_ARLL_MAX_ROWS; }
/* The row bound a value means: 0 off, 1 -> 48, N -> N. */
static inline int r4d_arll_bound(int v) { return v <= 0 ? 0 : v == 1 ? (int)R4D_ARLL_MAX_ROWS : v; }

/* Groups over W 16-byte words: three words a group, the last one padded. */
static R4D_ARLL_HD inline long long r4d_arll_groups(long long w) { return (w + R4D_ARLL_WORDS - 1) / R4D_ARLL_WORDS; }
/* Units a form needs in one (parity, source) slot for an n16-word message: the one-shot one region over the message,
 * the two-shot regions A (phase 1) and B (phase 2) over a shard each. */
static inline long long r4d_arll_need(int form, long long n16) {
    return form == R4D_ARLL_TWOSHOT ? 2LL * R4D_ARLL_UNITS * r4d_arll_groups(n16 / 4)
                                    : (long long)R4D_ARLL_UNITS * r4d_arll_groups(n16);
}
/* An instance's slot, in units, for messages up to `max_numel` elements of `esize` bytes (both forms at the bound),
 * rounded to 64 units. 0: no LL area (the bound is under a word). */
static inline long long r4d_arll_slot_units(long long max_numel, int esize) {
    const long long e = max_numel < (long long)R4D_ARLL_MAX_ELEMS ? max_numel : (long long)R4D_ARLL_MAX_ELEMS;
    const long long n16 = e * esize / 16;
    if (n16 < 1) return 0;
    long long u = r4d_arll_need(R4D_ARLL_ONESHOT, n16);
    const long long t = r4d_arll_need(R4D_ARLL_TWOSHOT, n16 - n16 % 4);
    if (t > u) u = t;
    return (u + 63) / 64 * 64;
}
/* The form fits: the message within the element bound, the two-shot's words split evenly over the four ranks (whole
 * words, at least one a rank), the units in the slot. */
static inline bool r4d_arll_fits(int form, long long numel, int esize, long long slot_units) {
    if (numel < 1 || numel > R4D_ARLL_MAX_ELEMS || slot_units <= 0 || (numel * esize) % 16) return false;
    const long long n16 = numel * esize / 16;
    if (form == R4D_ARLL_TWOSHOT && (n16 % 4 || n16 < 4)) return false;
    return r4d_arll_need(form, n16) <= slot_units;
}
/* The grid: a thread a group of the region a block sweeps (the message, or a shard), R4D_ARLL_THREADS a block,
 * 1 .. R4D_ARLL_MAX_BLOCKS blocks, never past the instance's block capacity. nb / nt > 0 pin it (r4d_selftest). */
static inline void r4d_arll_grid(int form, long long n16, int max_blocks, int pin_nb, int pin_nt, int* nb, int* nt) {
    const long long w = form == R4D_ARLL_TWOSHOT ? n16 / 4 : n16;
    const long long g = r4d_arll_groups(w);
    int t = pin_nt > 0 ? pin_nt : (int)R4D_ARLL_THREADS;
    if (t < 32) t = 32;
    if (t > 1024) t = 1024;
    long long b = pin_nb > 0 ? pin_nb : (g + t - 1) / t;
    if (b > R4D_ARLL_MAX_BLOCKS) b = R4D_ARLL_MAX_BLOCKS;
    if (b > g) b = g;
    if (b > max_blocks) b = max_blocks;
    if (b < 1) b = 1;
    *nb = (int)b;
    *nt = t;
}
/* The tag a launch's units carry: 1 + (its launch count on the instance) mod 65535 -- never 0, the cleared unit. */
static R4D_ARLL_HD inline unsigned r4d_arll_tag16(unsigned launch) { return 1u + launch % 65535u; }

/* Which launch takes LL, on the host: 0 = the plain kernel; else 1 + the form. v = ar_ll; rows / numel = the message;
 * esize = the instance's element bytes (2 = bf16, the only LL dtype); slot_units = the instance's LL slot (0 = none);
 * twoshot = today's form for this message (numel >= ts_min and the two-shot takes it); list_ok = the gather's slot
 * list fits a block's LDS (1 for the plain all-reduce). *declined is set where the switch covers the message but LL
 * cannot take it. */
static inline int r4d_arll_pick(int v, long long rows, long long numel, int esize, long long slot_units, int twoshot,
                                int list_ok, bool* declined) {
    *declined = false;
    const int bound = r4d_arll_bound(v);
    if (bound <= 0 || rows < 1 || rows > bound) return 0;
    const int form = twoshot ? R4D_ARLL_TWOSHOT : R4D_ARLL_ONESHOT;
    if (esize != 2 || !list_ok || !r4d_arll_fits(form, numel, esize, slot_units)) {
        *declined = true;
        return 0;
    }
    return 1 + form;
}
