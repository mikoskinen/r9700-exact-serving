/* xprefetch.h -- next-layer expert prefetch at decode: what the host half (xprefetch.cpp) and the
 * kernels (xprefetch.hip) share.
 *
 * Under expert-tiered placement, a routed layer's GEMMs read off-card experts from the pinned host
 * pool across the link. A layer's router input predicts the next routed layer's choice well, so at
 * each routed layer the prefetch applies the next layer's router to this layer's router input,
 * copies the likeliest pinned-pool experts onto the card with the DMA engine, and points the next
 * layer's weight tables at the copies just before its GEMMs.
 *
 * Results never change: a copy holds the bytes of the pool slot it came from, and an entry is
 * redirected only where it still names that slot (compare-and-swap; a heat-engine move wins) and
 * only once the copy has landed. Late copies go unused; nothing waits for them.
 *
 * Pieces, per routed layer l predicting l+1:
 *   - at l's router input (Ctx::xp_predict): on its own stream, riding behind the compute dispatch
 *     that reads the input, xp_logits applies l+1's router and xp_plan writes the chosen copies to
 *     a mailbox in pinned host memory;
 *   - after l's routed down GEMM (Ctx::xp_mark): a ride on the compute stream that the DMA list
 *     polls, so the copies cross the link under l+1's attention, not under l's streamed experts;
 *   - a host thread hands each mailbox to the prefetch DMA queue: wait for the mark, copy, then
 *     write the request number to the group's done word;
 *   - before l+1's routed GEMMs (Ctx::xp_step, compute stream): restore the previous layer's
 *     entries (its GEMMs are earlier in the stream) and redirect l+1's if the done word matches.
 * The ring holds kXpGroups layers of kXpSlots copies; a group is reused three layers later, after
 * the step that restored its entries. */
#pragma once
#include <cstdint>

namespace rad {

constexpr int kXpSlots    = 2;    /* copies a layer: the link fits about two units between the marks */
constexpr int kXpGroups   = 3;
constexpr int kXpMaxRows  = 16;   /* decode passes; a wider pass makes no prediction */
constexpr int kXpMaxK     = 16;
constexpr int kXpMaxNe    = 512;
constexpr int kXpMaxPools = 4;
constexpr int kXpParts    = 4;    /* the weights one expert's unit is made of */

/* One copied unit: its table entries, what they held and what they hold while patched, and where the
 * layer's routing counts say whether the step used it. */
struct XpRec {
    void**         ent[kXpParts];
    void*          oldv[kXpParts];
    void*          newv[kXpParts];
    const int32_t* cnt;
    int32_t        place;
    int32_t        pad;
};
struct XpJob {
    uint64_t src, dst, bytes, tag;    /* tag: the request's number, written after the rest */
};
/* A group's mailbox, in pinned host memory: xp_plan writes the jobs, then `seq`. */
struct XpBox {
    XpJob             jobs[kXpSlots];
    int32_t           n;
    volatile uint32_t seq;
};
struct XpPools {
    uint64_t lo[kXpMaxPools], hi[kXpMaxPools], stride[kXpMaxPools];
    int32_t  n;
};
/* Restored records: [0] units copied, [1] of them the step routed; [2] units planned. */
struct XpStats {
    unsigned int copied, routed, planned, pad;
};

/* Device bytes the prefetch takes for `layers` routed layers with largest expert unit `unit` bytes:
 * the ring and its side buffers (exactly what Ctx::xp_open allocates), each rounded to the card's
 * 2 MiB allocation grain. */
inline int64_t xp_device_bytes(int64_t unit, int64_t layers) {
    constexpr int64_t kGrain = (int64_t)2 << 20;
    auto grains = [](int64_t b) { return (b + kGrain - 1) / kGrain * kGrain; };
    const int64_t ring  = (int64_t)kXpGroups * kXpSlots * unit;
    const int64_t small = (int64_t)sizeof(XpRec) * kXpGroups * kXpSlots + 4 * kXpGroups + (int64_t)sizeof(XpStats) +
                          4 * (int64_t)kXpGroups * kXpMaxRows * kXpMaxNe + layers * kXpMaxNe * kXpParts * 8 +
                          layers * kXpMaxNe * 4;
    return grains(ring) + grains(small);
}

/* ---- lever XPF (ds41-xpf, SPEED-E): a newcomer pass's cross-layer expert prefetch -------------------------------
 *
 * A newcomer / MIXED pass (hundreds of prefill rows) reads ~64 host-pool units a routed layer a rank (about a third of
 * the layer's pooled units), in place, inside its routed GEMMs, where the link is already the bound; between two layers'
 * routed GEMMs (the attention block, the collectives, the router) the link carries only the all-reduces. Here, at a
 * point the architecture picks before layer l's router (deepseek41: l's attention input, the ffn proxy), the layer's
 * router over a proxy input gives each row's likely experts; the most-picked pinned-pool units are copied by the
 * prefetch DMA queue into the prefill ring while the attention block runs, in chunks that each write their own done
 * word, and l's routed GEMMs read the copies that landed (rad_expert_prefetch_step patches the entries chunk by chunk,
 * compare-and-swap: a heat-engine move wins). Late chunks stay unused; nothing waits for them.
 *
 * Results never change: a copy holds the pool slot's bytes, the kernels and operands are the same; the prediction (its
 * input, width, budget) only picks which bytes cross the link before the GEMMs instead of inside them.
 *
 * The ring is two groups of slots; layers alternate. A group's patches are restored by the next plan into that group
 * (the layer two back, whose GEMMs ran before it in stream order) and by the pass-end call (rad_xpf_plan with next
 * null). Sized by RADIANCE_XPF_MIB (VRAM budget term); unset: no ring, no queue, nothing runs. */
constexpr int kXpfParts     = 8;     /* table entries an EXL3 unit is redirected through (7 planes, down's suh twice) */
constexpr int kXpfMaxSlots  = 128;   /* copies a layer at most */
constexpr int kXpfChunk     = 8;     /* copies a DMA list (its own done word) */
constexpr int kXpfChunks    = kXpfMaxSlots / kXpfChunk;
constexpr int kXpfMaxK      = 16;
struct XpfRec {
    void**         ent[kXpfParts];
    void*          oldv[kXpfParts];
    void*          newv[kXpfParts];
    const int32_t* cnt;
    int32_t        place;
    int32_t        pad;
};
struct XpfBox {
    XpJob             jobs[kXpfMaxSlots];
    int32_t           n;
    volatile uint32_t seq;
};
/* [0] planned (copies asked), [1] patched (copies that landed in time, units), [2] used (patched units the layer
 * routed, counted at the restore), [3] plans, [4] candidates seen (pooled units with a pick), [5] restores. */
struct XpfStats {
    unsigned int planned, patched, used, plans, cands, restores, pad[2];
};

}  /* namespace rad */

extern "C" int rad_xpf_count_launch(const uint16_t* lg, uint32_t rows, uint32_t ld, uint32_t ne, uint32_t kp,
                                    const float* bias, int32_t* cnt, void* stream);
extern "C" int rad_xpf_plan_launch(int32_t* cnt, uint32_t ne, void** const* map, const int32_t* place,
                                   const int32_t* rcnt, rad::XpPools pools, uint64_t ring, uint64_t rs, uint32_t slots,
                                   rad::XpfRec* rec, int32_t* nrec, rad::XpfBox* box, uint32_t seq,
                                   rad::XpfStats* stats, void* stream);
extern "C" int rad_xpf_step_launch(const rad::XpfRec* rec, const int32_t* nrec, const uint32_t* done, uint32_t seq,
                                   rad::XpfStats* stats, void* stream);
extern "C" int rad_xpf_restore_launch(rad::XpfRec* rec, int32_t* nrec, rad::XpfStats* stats, void* stream);

extern "C" int rad_xp_logits_launch(const uint16_t* h, const uint16_t* w, float* logits, uint32_t rows,
                                    uint32_t n_embd, uint32_t ne, void* stream);
extern "C" int rad_xp_plan_launch(const float* logits, uint32_t rows, uint32_t ne, uint32_t k,
                                  void** const* map, const int32_t* place, const int32_t* cnt,
                                  rad::XpPools pools, uint64_t ring, uint64_t ring_stride, rad::XpRec* rec,
                                  int32_t* nrec, rad::XpBox* box, uint32_t seq, rad::XpStats* stats,
                                  void* stream);
extern "C" int rad_xp_step_launch(const rad::XpRec* rrec, const int32_t* rn, const rad::XpRec* prec,
                                  const int32_t* pn, const uint32_t* done, uint32_t seq, void* stream);
