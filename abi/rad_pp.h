/* rad_pp.h -- the two-node pipeline (tp8 S4, docs/OPS.md "Stage crossings"): what an architecture
 * plugin and a kernel library see of it.
 *
 * One engine runs on two machines as two processes, each a tensor-parallel group of its own: the
 * leader (node 0) holds the scheduler, the sampler, the embedding and the head and some layers, the
 * follower (node 1) the other layers. A step crosses the machine boundary twice:
 *
 *   X1  leader -> follower  what the follower's first layer reads (the leader's embedding output)
 *   X2  follower -> leader  what the leader's next layer reads after the follower's last one (the
 *                           inputs of the op at the cut, so no arithmetic moves across it)
 *
 * The core owns one ring a crossing on each node, in pinned host memory every local card reads and
 * writes over its own link: the sender's `stage_send` packs the live rows into its send ring and
 * publishes them with `ready`; the core's link moves the payload into the receiver's ring and
 * publishes it with `landed`; every receiving rank's `stage_recv` waits for `landed`, copies the
 * rows into its buffers and counts itself in `taken`. The counters are read and written at run time
 * by the kernels themselves, so a recorded pass replays without a host-side argument changing.
 *
 * A single-machine engine (the published path) has no rings: rad_pp_info answers nodes 1, or is
 * absent altogether in an engine built before this header, which a plugin sees as a null weak
 * symbol. Nothing here is part of the ABI layout fingerprint (rad_abi_layout_rows.h): a kernel
 * library built without it still loads, and one built with it loads into an older engine.
 */
#ifndef RAD_PP_H
#define RAD_PP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAD_PP_RING_MAGIC 0x31474e4952505052ull /* "RPPRING1", little endian */
/* The payload starts this far into the ring, so the header's counters own their cache lines. */
#define RAD_PP_RING_HDR   4096
#define RAD_PP_MAX_RANKS  8
/* The most buffers one crossing carries (stage_send's x0..x11). tp8 S6: a crossing of a general cut
 * W(h, t) carries the residual stream's five parts and the receiving layers' reads of the other node
 * (a kv source's new entries and their pages, the selections, the keep words): up to ten. Each part
 * has its own rows (the residual's are the step's, a compressor's entries its work rows); the
 * header's `rows` is part 0's. */
#define RAD_PP_MAX_PARTS  12
/* Each part of a payload starts on this boundary. */
#define RAD_PP_PART_ALIGN 256
/* A payload ends in a trailer of this many bytes at the aligned end of its parts: its first word is
 * the payload's number (the sender's `ready` + 1), which the receiver compares with the payload it
 * waits for, so a stale or torn payload whose header happens to match is refused (review F6).
 * `bytes` counts it; a ring holds it (the core sizes rings for it). */
#define RAD_PP_TRAILER    64
/* The trailer's second word (tp8 S6): the payload's layout, rad_pp_layout over its parts' bytes, so a
 * receiver whose parts differ in size from the sender's (two nodes out of step) is refused even where
 * the totals agree. The link moves the trailer with the payload; it carries no per-part sizes. */
static inline uint64_t rad_pp_layout(const uint64_t* bytes, int n) {
    uint64_t h = 1469598103934665603ull;
    int k;
    for (k = 0; k < n; ++k) h = (h ^ bytes[k]) * 1099511628211ull;
    return h ^ (uint64_t)n;
}

enum { RAD_PP_X1 = 0, RAD_PP_X2 = 1, RAD_PP_N_RINGS = 2 };

/* tp8 S5: lanes (--pp-lanes). The leader's scheduler runs up to this many independent step streams; a
 * sequence is bound to one for life and each lane has at most one step in flight, so the two nodes
 * work at once on different lanes. Every lane has its own pair of rings (X1, X2), so each ring keeps
 * the one-payload-in-flight invariant of a single lane. The core addresses a lane's ring as
 * ring index lane * RAD_PP_N_RINGS + crossing. */
#define RAD_PP_MAX_LANES  4
#define RAD_PP_MAX_RING_IX (RAD_PP_MAX_LANES * RAD_PP_N_RINGS)

/* tp8 S5: which part of the leader's pass a step issues. A lane's step flows leader -> follower ->
 * leader, so the leader's pass is issued in two parts with other lanes' parts between them:
 *   RAD_PP_PART_WHOLE  the pass as one (one machine; a pipeline of one lane; the follower always)
 *   RAD_PP_PART_HEAD   the leader's part up to and including stage_send X1
 *   RAD_PP_PART_TAIL   the leader's part from stage_recv X2 on (the head and sampler follow it)
 * A plugin reads it at issue (RadPpInfo::part); the core keys recorded passes by it. */
enum { RAD_PP_PART_WHOLE = 0, RAD_PP_PART_HEAD = 1, RAD_PP_PART_TAIL = 2, RAD_PP_PART_SEG = 3 };

/* tp8 dev M4 (RADIANCE_PP_SEG_LAYERS=k on the follower; unset: nothing here): a follower pass of many rows is issued in
 * layer segments [seg_l0, seg_l1) of its [h, t), with other lanes' whole passes between them on the same stream
 *   RAD_PP_PART_SEG  layers [seg_l0, seg_l1) only: from X1 (seg_l0 == h) or from the segment boundary's restore, to X2
 *                    (seg_l1 == t) or to the next boundary's save
 * A boundary s is a cut on one node: the segment before it saves what a cut W(s, t) would carry into the follower's layer s
 * (the boundary op's inputs, the selections, the keep words; no kv entries: the node owns its caches) and the segment
 * from it restores those bytes, each rank its own, through the stage ops in loop mode (`loop` 1: the ring operand is a
 * buffer of this rank, the plugin's, which the send fills and publishes to itself and the receive copies back: no link,
 * no peer, no host memory). The plugin names its boundaries at declare with rad_pp_segments; an engine that has none
 * never issues RAD_PP_PART_SEG. */
#define RAD_PP_MAX_SEGS 16

/* One crossing's ring on one node. `ready` is the sender's, `landed` and `taken` the receiver's;
 * a node holds each ring in one role. Every counter is the count of payloads, starting at 0. */
typedef struct RadPpRing {
    volatile uint64_t ready;          /* send ring: payloads stage_send has published */
    volatile uint64_t sent;           /* send ring: payloads the link has taken off the ring */
    /* tp8 S5: the host backend's stage ops note when they published / took the last payload
     * (CLOCK_MONOTONIC ns; 0 where no op notes it, as on a card), for the stage busy stamps. */
    volatile uint64_t ready_ns;       /* send ring: when the sender published `ready` */
    uint64_t          pad0[5];
    volatile uint64_t landed;         /* receive ring: payloads the link has written in full */
    volatile uint64_t taken_ns;       /* receive ring: when rank 0 took the last payload */
    uint64_t          pad1[6];
    volatile uint64_t abort;          /* nonzero: the link is down; stage_recv fails the pass. A stage
                                       * op that gives up stores 2 (abort seen), 3 (its deadline), 4 (a
                                       * payload of another step's shape), 5 (a stale payload: its
                                       * trailer is not the number waited for) */
    uint64_t          pad2[7];
    volatile uint64_t taken[RAD_PP_MAX_RANKS]; /* receive ring: payloads rank r has copied out */
    uint64_t magic;                   /* RAD_PP_RING_MAGIC once the core has set the ring up */
    uint64_t cap;                     /* payload capacity in bytes */
    uint64_t rows;                    /* rows of the last payload (the step's n_tok) */
    uint64_t bytes;                   /* bytes of the last payload */
    uint64_t n_ranks;                 /* local ranks that read a receive ring */
    uint64_t deadline_ms;             /* how long stage_recv waits for a live peer before failing */
    uint64_t id;                      /* RAD_PP_X1 or RAD_PP_X2 */
    uint64_t parts;                   /* parts of the last payload, for the receiver's check */
    uint64_t part_bytes[RAD_PP_MAX_PARTS]; /* each part's bytes in the last payload */
    /* tp8 loop P8 (RADIANCE_PP_SEND_SPLIT, a send ring only; 0 from an engine before it): who copies a payload into the
     * ring. 0: rank `sender` alone (as before). 1: every rank of the node its share -- the payload's 256-byte units split
     * evenly over `send_ranks`, each rank from its own copy of the parts (the residual stream and what a crossing carries
     * are replicated over the ranks) -- and rank `sender` publishes once every rank's `done` names the payload. 2 (a
     * check of 1): rank `sender` copies the whole payload, then every other rank compares its share of its own parts
     * with the ring's bytes before the publish, and sums the 4-byte words that differ. The payload's bytes are rank
     * `sender`'s under 0 and 2. */
    uint64_t send_split;
    uint64_t send_ranks;                       /* the ranks of this node (1 and 2) */
    volatile uint64_t done[RAD_PP_MAX_RANKS];  /* payloads rank r has copied (1) or compared (2) its share of */
    volatile uint64_t staged;                  /* (2) payloads rank `sender` has copied in full, not yet published */
    volatile uint64_t diff_words[RAD_PP_MAX_RANKS]; /* (2) 4-byte words of rank r's share that differed, summed */
    volatile uint64_t cmp_words[RAD_PP_MAX_RANKS];  /* (2) 4-byte words rank r compared, summed */
    volatile uint64_t diff_part[RAD_PP_MAX_RANKS][RAD_PP_MAX_PARTS]; /* (2) rank r's differing words by part, summed */
    /* tp8 loop P10 (RADIANCE_PP_SEND_ASYNC=N, a send ring only; 0 from an engine before it): a payload of at least
     * `async_rows` rows (part 0's) is packed by rank `sender` into `stage_dev` (a buffer of `stage_cap` bytes on that
     * rank's card, one a node, the core's) on the pass's stream, and copied from there into the ring + published on a
     * side stream of the card, so the pass's stream goes on with the next pass while the payload crosses PCIe. The
     * ring's bytes are the same words in the same places (the parts' bytes, packed at the payload's offsets). The core
     * issues every pass of >= async_rows rows live (never recorded), so a side-stream launch is never left out of a
     * tape. 0 = every payload on the pass's stream (as before). */
    uint64_t async_rows;
    uint64_t stage_dev;
    uint64_t stage_cap;
    volatile uint64_t async_sends;             /* payloads sent the async way (the sender's host thread counts) */
    /* tp8 loop P10b (RADIANCE_PP_RECV_ASYNC=N, a receive ring only; 0 from an engine before it): a payload of at least
     * `recv_rows` rows is copied from the ring into rank r's card staging buffer `recv_stage[r]` (`recv_cap` bytes, the
     * core's) by a launch on the card's side stream issued with the pass -- it waits on the card for the payload to
     * land, checks it as stage_recv does and counts `taken[r]` -- and the pass's stream only unpacks it (D2D) when the
     * side stream is done; so a payload that lands while the card still runs the previous pass crosses PCIe beside it.
     * The parts get the same bytes. The core issues every pass of >= recv_rows rows live (never recorded). */
    uint64_t recv_rows;
    uint64_t recv_stage[RAD_PP_MAX_RANKS];
    uint64_t recv_cap;
    volatile uint64_t recv_async[RAD_PP_MAX_RANKS];   /* payloads rank r took the async way (its host thread counts) */
} RadPpRing;

#if defined(__cplusplus)
static_assert(sizeof(RadPpRing) <= RAD_PP_RING_HDR, "the ring's header fits before its payload");
#endif

/* A payload's layout: part i at RAD_PP_RING_HDR + the aligned sum of the parts before it. */
static inline uint64_t rad_pp_align(uint64_t n) {
    return (n + RAD_PP_PART_ALIGN - 1) / RAD_PP_PART_ALIGN * RAD_PP_PART_ALIGN;
}

typedef struct RadPpInfo {
    int32_t nodes;        /* 1: one machine, the published engine; 2: the two-node pipeline */
    int32_t rank;         /* this node: 0 the leader, 1 the follower */
    int32_t cut_h;        /* W(h, t): the leader holds layers [0, h) and [t, n_layer), */
    int32_t cut_t;        /* the follower [h, t); bring-up is W(0, 20) */
    int32_t link;         /* 0 kernel TCP, 1 RDMA verbs */
    /* tp8 S5 (zero from an engine before lanes: one lane, the whole pass): */
    int32_t lanes;        /* --pp-lanes: the leader's step streams (1 = one step in flight) */
    int32_t lane;         /* the lane of the pass being issued now (every rank of a node issues one at a time) */
    int32_t part;         /* RAD_PP_PART_*: which part of the leader's pass is being issued */
    /* tp8 dev M4 (zero from an engine before segments): RAD_PP_PART_SEG's layers [seg_l0, seg_l1) */
    int32_t seg_l0;
    int32_t seg_l1;
    int32_t reserved[6];
    /* This node's ring of each crossing (its send ring or its receive ring) for the lane of the pass
     * being issued, null until the core has allocated them, which is after declare and before the
     * first step. A plugin reads it at every issue: each lane has its own rings. */
    RadPpRing* ring[RAD_PP_N_RINGS];
} RadPpInfo;

/* Exported by an engine that knows the pipeline; returns 0 and fills `out` (nodes 1 when the
 * engine runs on one machine). Weak, so a plugin loads into an engine without it and reads a null
 * pointer there: one machine. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
int rad_pp_info(RadPpInfo* out);

/* The plugin names, at declare (every rank, every sizing declare; the largest wins), the bytes one
 * row of crossing `ring` carries: the sum of its parts' row bytes, before the 256-byte alignment
 * of each part. The core sizes each ring for --max-num-batched-tokens such rows and the
 * rendezvous compares the two nodes' numbers. A pipeline whose plugin names none refuses to
 * start. Weak, like rad_pp_info. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
void rad_pp_crossing(int ring, int64_t row_bytes, int parts);

/* tp8 dev M4: the plugin names, at declare (every rank, every declare; the last wins), the segment boundaries it declared
 * on this node (layers s, ascending, inside the follower's (h, t); n 0: none). Weak, like rad_pp_info. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
void rad_pp_segments(const int32_t* bounds, int n);

static inline void rad_pp_declare_segments(const int32_t* bounds, int n) {
    if (rad_pp_segments) rad_pp_segments(bounds, n);
}

/* tp8 S6: the plugin names, at declare (rank 0's real declare; repeats are ignored), each KV cache
 * layer both nodes hold of one source -- the owner its own, the receiver the copy a crossing fills:
 * `group` (the rad_kvgroup), `slot` (the layer's place among the group's bound layers on THIS node, in
 * ascending layer order: the builder sorts them)
 * and `layer` (the model layer, which names it on both). RADIANCE_PP_KV_CHECK=N then has the two
 * nodes compare a CRC-32C of that layer's blocks every N trunk steps (debug). Weak, like
 * rad_pp_info. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
void rad_pp_kv_copy(int32_t group, int32_t slot, int32_t layer);

static inline void rad_pp_declare_kv_copy(int32_t group, int32_t slot, int32_t layer) {
    if (rad_pp_kv_copy) rad_pp_kv_copy(group, slot, layer);
}

/* The node's pipeline as the plugin should act on it: nodes 1 when the engine has no pipeline or
 * predates it. */
static inline RadPpInfo rad_pp_query(void) {
    RadPpInfo p;
    unsigned i;
    for (i = 0; i < sizeof p; ++i) ((unsigned char*)&p)[i] = 0;
    p.nodes = 1;
    if (rad_pp_info && rad_pp_info(&p) != 0) p.nodes = 1;
    if (p.nodes < 1) p.nodes = 1;
    return p;
}

/* rad_pp_crossing where the engine has it. */
static inline void rad_pp_declare_crossing(int ring, int64_t row_bytes, int parts) {
    if (rad_pp_crossing) rad_pp_crossing(ring, row_bytes, parts);
}

#ifdef __cplusplus
}
#endif

#endif /* RAD_PP_H */
