/* ds41_prefetch.h -- tp8 loop D24 (fn4 ds41_prefetch, RADIANCE_DS41_PREFETCH): which passes and which sites take deepseek41's
 * weight reads beside the decode all-reduces (libr4d `prefetch` through rad_arch.h's prefetch_beside). Pure rules, so
 * tests/ds41_prefetch_test.cpp checks them on the host (the host backend serves no `prefetch`: the reads are never declared
 * there, so the plugin's issue of them runs on a card only).
 *
 * The rules: a pass takes the reads only where they were declared, the switch is >= 1, it is a trunk pass (no drafter
 * pass, no encoder pass) and EVERY row decodes -- no prefill sequence: rad_arch.h's rule, "prompt steps never take the
 * reads" (beside prefill-sized all-reduces they exposed the four-rank slot-reuse race; a MIXED step carries a prompt's rows
 * in its collectives) -- and it has 1 .. kRows rows (64: the deepest decode lane step, 16 sequences x 4 rows). Site 0 is
 * right before a layer's attention all-reduce and reads that layer's FFN side; site 1 is right before a layer's MoE
 * all-reduce and reads the NEXT layer's attention side, only where the pass goes on to that layer on this node.
 */
#pragma once

#include <cstdint>

namespace ds41_pf {

constexpr int64_t kRows = 64;

/* The reads, as a mask (site_reads). */
enum Read : unsigned {
    kMixF   = 1u << 0,   /* site 0, bit 0: the layer's hc_f_fn (the fused post + collapse + norm + mix after the reduction) */
    kRouter = 1u << 1,   /* site 0, bit 1: the router projection */
    kShGu   = 1u << 2,   /* site 0, bit 1: the shared arm's gate_up */
    kMixA   = 1u << 3,   /* site 1, bit 0: the next layer's hc_a_fn */
    kComp   = 1u << 4,   /* site 1, bit 2: the next layer's comp_w (a kv source only) */
    kQa     = 1u << 5,   /* site 1, bit 2: the next layer's wq_a */
    kKv     = 1u << 6,   /* site 1, bit 2: the next layer's wkv */
};

/* A pass's bits: `sw` (the switch, 0 .. 7) where `ok` (reads declared), a trunk pass (draft_pass 0, not an encoder pass)
 * whose every row decodes (n_seq_dec == n_seq, n_tok_dec == n_tok) of 1 .. kRows rows; else 0. */
inline int pass_bits(int sw, bool ok, int draft_pass, bool enc, int64_t n_tok, int64_t n_seq, int64_t n_seq_dec,
                     int64_t n_tok_dec) {
    if (!ok || sw <= 0 || sw > 7 || draft_pass != 0 || enc) return 0;
    if (n_tok <= 0 || n_tok > kRows || n_seq <= 0) return 0;
    return n_seq_dec == n_seq && n_tok_dec == n_tok ? sw : 0;
}

/* The reads a site issues at `bits` (site 0: this layer's FFN side; site 1: the next layer's attention side, `src` = that
 * layer is a kv source, `next` = the pass goes on to it on this node: no crossing after this layer and a next held layer). */
inline unsigned site_reads(int bits, int site, bool src, bool next) {
    unsigned r = 0;
    if (site == 0) {
        if (bits & 1) r |= kMixF;
        if (bits & 2) r |= kRouter | kShGu;
        return r;
    }
    if (site != 1 || !next) return 0;
    if (bits & 1) r |= kMixA;
    if (bits & 4) r |= (src ? kComp : 0u) | kQa | kKv;
    return r;
}

}  // namespace ds41_pf
