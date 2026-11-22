/* rad_qsa.h -- Qwen4-Exp's QSA indexer: decides which blocks of key history a full-attention
 * layer's attention reads.
 *
 * docs/QSA.md has the design; this is the declaration side. The kernels are libr4d's
 * `qsa_block_key`, `qsa_score` and `qsa_select`; the sparse attention is the ordinary `attn_paged`
 * issued with the selection in place of the sequence's block table (both plain operands, so core/
 * knows nothing of QSA).
 *
 * Contents: the projection and query path, the block-key path with its two states, and the score
 * and selection. Below `budget + ratio - 1` tokens of context every visible block is selected, so
 * the sparse path must reproduce the dense path's text byte for byte; that is the test for any
 * change here.
 */
#ifndef RAD_QSA_H
#define RAD_QSA_H

#include "rad_arch.h"
#include "rad_fp8.h"

#include <atomic>

namespace rad {
namespace arch {

/* fn4 qsa_split (16.11.2026, qwen/QSA-SPLIT.md; switch qsa_split / RADIANCE_FN4_QSA_SPLIT, a mask; 0 = today). Every rank
 * holds the whole projection, the block keys and the score plane, and today runs the score and the selection over every
 * sequence's whole history. Bit 1: rank r of the group scores and selects over its own block range of each sequence
 * (libr4d r4d_qsa_part.h: cuts of the sequence's complete-block count, so a row's range never depends on the other rows),
 * an all_gather moves each rank's block-ordered top-k list to every rank, and one merge a query rebuilds qsa_select's table
 * byte for byte (a block in the global top-k is in its range's top-k; the merge keeps qsa_select's threshold, tie rule and
 * table order). Bit 2: the same chunked selection on one card (parts 1) at any decode height, for steps bit 1 does not
 * take. Steps of at most `split_rows` rows from `split_min` live blocks (RADIANCE_FN4_QSA_SPLIT_MIN, blocks; 0) take it;
 * the choice reads host values only (rows, the context bound, the switch), the same on every rank, and every path gives
 * the same bytes, so a sequence's selection is the same alone and beside any batch. RADIANCE_FN4_QSA_SPLIT set at start
 * (any value above 0) declares both forms and starts at that value. */
inline std::atomic<int>& fn4_qsasplit_flag() {
    static std::atomic<int> f{[] { const char* e = std::getenv("RADIANCE_FN4_QSA_SPLIT");
                                   const int v = e && *e ? std::atoi(e) : 0;
                                   return v < 0 ? 0 : v > 3 ? 3 : v; }()};
    return f;
}
/* qfn-on-ds PF probe QSP (Qwen loop round 27, lever QRS's bound; sp_route bit 16384 through the plugin's switch): a prefill
 * step's (more than 256 rows) score + select issued TWICE -- the second pair rewrites the score plane and the selection from
 * the same inputs, so the same bytes; the step's added time is one copy's marginal cost on the compute lane, the bound of
 * scoring and selecting a quarter of the rows a rank (every rank today scores and selects every row of the step). */
inline std::atomic<int>& fn4_qsaprobe_flag() {
    static std::atomic<int> f{0};
    return f;
}
inline int fn4_qsasplit_declared() {
    static const int d = [] { const char* e = std::getenv("RADIANCE_FN4_QSA_SPLIT");
                              return (e && *e && std::atoi(e) > 0) ? 3 : 0; }();
    return d;
}
inline int64_t fn4_qsasplit_min() {
    static const int64_t v = [] { const char* e = std::getenv("RADIANCE_FN4_QSA_SPLIT_MIN");
                                  const long long x = e && *e ? std::atoll(e) : 0;
                                  return (int64_t)(x < 0 ? 0 : x); }();
    return v;
}

struct QsaIndexer {
    struct Config {
        int64_t     heads    = 0;   /* indexer_n_heads      -- 4 */
        int64_t     head_dim = 0;   /* indexer_head_dim     -- 128 */
        int64_t     ratio    = 0;   /* indexer_compress_ratio -- 4 tokens a block */
        int64_t     budget   = 0;   /* indexer_budget       -- 2048 tokens, so 512 blocks */
        int64_t     rot      = 0;   /* the layer's rotary width, shared by the indexer */
        const char* mode     = "neox";
        /* Checkpoint names. Absent means this layer has no indexer and declare() is a no-op. */
        const char* qk_proj  = nullptr;
        const char* q_norm   = nullptr;
        const char* k_norm   = nullptr;
        /* Derived; set by the caller because they size buffers it declares. */
        int64_t     topk       = 0;   /* budget / ratio -- blocks the selection keeps */
        int64_t     max_blocks = 0;   /* max_ctx / ratio -- the widest block table */
        int64_t     max_work   = 0;   /* ceil(max_tok / ratio) + max_seqs */
        /* The tail's ring length, `ratio + max_spec`. A ring of `ratio` indexed by `pos % ratio`
         * fails under speculation: the next step restarts at the first rejected position, so
         * overwritten slots are needed again (at ratio 4, depth 3, positions 9..12 write slots
         * 1,2,3,0 and 12 overwrites the slot of 8; accepting one then rebuilds block 2 from token
         * 12, corrupting that block key permanently). A step can still owe positions
         * [p0 - ratio + 1, p0 + n_spec], so `ratio + max_spec` slots keep every owed position. */
        int64_t     ring       = 0;   /* ratio + max_spec */
        /* The attention group, whose page ids the selection outputs. The block-key group is
         * declared on first use (see declare()). */
        rad_kvgroup kv_attn    = 0;
        /* The tail's group: one RAD_KV_LINEAR state per sequence per layer, `ring` rows of
         * `head_dim` (a KV group because it is per-sequence state; see Wire). */
        rad_kvgroup kv_tail    = 0;
        /* Whether anything reads the selection. False on the dense path; the stateful block-key
         * half still runs so the sparse path can later read the history, but score and select (the
         * indexer's largest ops) are skipped. Declared either way, so the graph and buffer plan do
         * not depend on it. */
        bool        emit_sel   = true;
        /* The rows the score plane was declared with, 0 for max_tok. Fewer only where a step
         * wider than them is one sequence (RadBuildCtx::max_mixed_tok); such a step scores and
         * selects in windows of this many rows (step()). */
        int64_t     plane_rows = 0;
        /* fn4 qsa_split: the forms declared (a mask, fn4_qsasplit_declared), the most query rows a split step has (the
         * lists' height) and the fewest live blocks it splits from. 0 = not declared. */
        int         split      = 0;
        int64_t     split_rows = 0;
        int64_t     split_min  = 0;
    };

    /* `qk` is the whole projection (`heads` query heads, then one raw key head); `q` is the query
     * heads normed and rotated, contiguous.
     *
     * The raw key is not normed or rotated here: it is pooled over `ratio` tokens first, and the
     * pooled result is normed and rotated at the block's first token's position (`qsa_block_key`),
     * unlike the q path. */
    struct Wire {
        rad_buf qk = 0;   /* [max_tok, (heads + 1) * head_dim] */
        rad_buf q  = 0;   /* [max_tok, heads * head_dim] */
        /* These buffers are within-step scratch shared by every layer. The tail is not here: it
         * carries an unfinished block's raw index keys across steps, per layer (each layer has its
         * own `index_qk_proj`, the MTP head included) and per sequence (batch rows are step-local
         * and shift between steps). A KV group has a stable per-sequence slot, zeroed on hand-out
         * and checkpointed with the linear state by the prefix cache, so the tail is
         * `Config::kv_tail` (RAD_KV_LINEAR), bound to this layer in declare(). */
        rad_buf stage = 0;   /* [max_work * ratio, head_dim] */
        rad_buf page  = 0;   /* [max_work]      i32 */
        rad_buf bpos  = 0;   /* [max_work] i32, or [max_work, 3] under M-RoPE */
        rad_buf nc    = 0;   /* [max_seqs]      i32 */
        /* The selection, one row per query, not per sequence (the same at decode; at prefill each
         * query of a chunk selects from the shared history, docs/QSA.md, "Per-query selection"). `sel` and
         * `seqused` replace the sequence's block table and length for the sparse attention.
         *
         * `score` is the largest: max_tok * max_ctx/ratio floats, 134 MB at a 2048-token chunk
         * and 64K context, one buffer for the whole model. A step of one sequence is tiled on the
         * query axis when it is wider than the plane (Config::plane_rows): a window of it is a
         * sequence of its own rows. A step of several is not: that would need a per-token
         * sequence id operand (from qsa_work), since `cu` is absolute over the batch. */
        rad_buf score = 0;   /* [max_tok, max_blocks] f32 */
        rad_buf sel   = 0;   /* [max_tok, topk + 1]   i32 */
        rad_buf nsel  = 0;   /* [max_tok]             i32 */
        rad_buf sequ  = 0;   /* [max_tok]             i32 */
        /* The model's cos/sin plane, [rope_table_rows, rot], or 0 if it builds none. Filled once a
         * step by `rope_table` for the attention block; the indexer rotates at the same `rot`,
         * `theta` and `scale`. Without it the unfused per-head norms run. */
        rad_buf cos_sin = 0;
        /* fn4 qsa_split: candidate lists (r4d_qsa_part.h, LW = 4 + 2 topk words): `qlist` [split_rows, 8 LW] a query's
         * chunks, `qpart` [split_rows, LW] this rank's list, `qgath` [split_rows, world LW] every rank's after the gather. */
        rad_buf qlist = 0, qpart = 0, qgath = 0;
    };

    Geom       g{};
    Config     c{};
    Wire       w{};
    int        layer = 0;
    rad_weight  w_qn = 0, w_kn = 0;
    LinearFP8   proj{};
    rad_kvgroup kv_bk = 0;
    rad_op      op_norm = 0, op_rope = 0, op_qprep = 0;
    /* Most rows `op_qprep` may be issued over (0 when not declared): `qk_fuse_rows`, a correctness
     * bound, since the cos/sin table is filled only for steps at or under it. */
    int64_t     qprep_rows = 0;
    rad_op      op_work = 0, op_bkey = 0, op_tail = 0, op_score = 0, op_select = 0;
    /* fn4 qsa_split: the rank's score and part, the gather, the one-card part, the merge */
    rad_op      op_score_part = 0, op_selp = 0, op_qgath = 0, op_selp1 = 0, op_selm = 0;

    bool on() const { return op_rope != 0; }

    /* `bk` is the block-key KV group, created on first use and shared by all QSA layers (hence
     * in/out). It must be declared immediately before `qsa_block_key`: `Builder::find_kv_consumer`
     * binds a group to the first later-declared op with a `block_size`. */
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, const ActFP8& in, rad_kvgroup* bk);
    /* `win_cu`, device: {0, plane_rows}, the one sequence of a window of a step wider than the
     * score plane (Config::plane_rows); null where no step is. */
    void step(RadCtx* c, const ActFP8& in, const RadBatch* batch,
              const int32_t* win_cu = nullptr) const;
    /* Stores the pass's block keys and tail exactly as step() does, with no query, score or
     * selection (see AttnGatedFP8::store_kv). */
    void store(RadCtx* c, const ActFP8& in, const RadBatch* batch) const;
    bool store_keys(RadCtx* c, const RadBatch* batch) const;
};

inline int QsaIndexer::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                               const Config& cfg, const Wire& wire, const ActFP8& in,
                               rad_kvgroup* bk) {
    g = geom; c = cfg; w = wire; layer = l;
    if (!c.qk_proj || !c.q_norm) return RAD_OK;          /* no indexer on this layer */
    if (c.heads <= 0 || c.head_dim <= 0 || c.ratio <= 0 || c.budget <= 0) return RAD_E_INVAL;
    if (!w.qk || !w.q) return RAD_E_INVAL;
    /* The layer's rotary width must fit inside the index head (as the checkpoint's validator also
     * requires). */
    if (c.rot < 0 || c.rot > c.head_dim || (c.rot & 1)) return RAD_E_SHAPE;

    const int64_t nw = (c.heads + 1) * c.head_dim;       /* the projection width */

    RAD_ARCH_TRY(proj.declare(b, nm, g, nm.f("blk.%d.qsa_qk", l), nw, g.n_embd,
                              RAD_SHARD_NONE, grp_layer(l), { c.qk_proj }, in, w.qk));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_q_norm.weight", l), c.q_norm));
    w_qn = decl_w(b, nm.f("blk.%d.qsa_q_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_qn) return RAD_E_INVAL;

    /* Unfused, one norm op is issued once per head: in `qk` (`heads + 1` heads wide) the query
     * heads are 128 apart within a token and 640 apart across tokens, which no rank-2 view can
     * express, but a one-head column slice can. It writes a contiguous `q` that the rope takes
     * whole. For 128-wide rows these launches are nearly all overhead.
     *
     * `op_qprep` fuses them (five launches to one) using `qk_norm_rope_gate`'s k side: it reads
     * [tokens, heads, head_dim] at the operand's row pitch, writes heads contiguously, and rounds
     * the norm to bf16 before rotating. With no gate, `n_head` 0 selects that half and passes no
     * q operands. It needs the model's cos/sin table (same `rot`, `theta`, `scale` as the
     * attention block) and is used only up to `qk_fuse_rows`, a correctness bound: the table is
     * filled only for steps that small, so a larger step would rotate by stale angles. The bound
     * comes from the same function as the attention block's. */
    const bool qprep_neox = !std::strcmp(c.mode, "neox") || !std::strcmp(c.mode, "mrope") ||
                            g.rope_mc;
    qprep_rows = qk_fuse_rows(g);
    if (w.cos_sin && qprep_neox && qprep_rows > 0 && (c.head_dim % 32) == 0 &&
        rad_op_declared(b, "qk_norm_rope_gate"))
        op_qprep = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, qprep_rows),
                                    RAD_INT("head_dim", c.head_dim), RAD_INT("n_head", 0),
                                    RAD_INT("n_head_kv", c.heads), RAD_INT("rot", c.rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_qn)),
                     {w.qk, w.cos_sin}, {w.q});

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_qn)),
                {w.qk}, {w.q});
    op_rope = rw(b, decl_rope(b, g, g.max_tok, c.head_dim, c.heads, 0, c.rot, c.mode),
                {w.q}, {w.q});

    /* ---- the block-key path. Nothing below reads anything above except `w.qk`'s raw-key half. */
    if (!c.k_norm || !w.stage || !c.topk || !c.max_blocks || !c.max_work)
        return RAD_E_INVAL;
    if (!c.kv_tail || c.ring < c.ratio) return RAD_E_INVAL;
    /* This layer's tail state: the caller's group holds one state per sequence per bound layer. */
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, c.kv_tail));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_k_norm.weight", l), c.k_norm));
    w_kn = decl_w(b, nm.f("blk.%d.qsa_k_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_kn) return RAD_E_INVAL;

    op_work = rw(b, RAD_OP(b, "qsa_work",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                               RAD_INT("seqs", g.max_seqs),
                               RAD_INT("work", c.max_work), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.qk}, {w.stage, w.page, w.bpos, w.nc});

    /* The block-key group, declared immediately before its consumer (see the declaration). One KV
     * head, `block_size` = ratio and `head_dim` = n / (2 * ratio) make a page hold exactly one
     * block key of `n` values, one page per `ratio` tokens. */
    if (*bk == 0) {
        if ((c.head_dim % (2 * c.ratio)) != 0) return RAD_E_SHAPE;
        RadKVGroupDecl d{};
        d.kind      = RAD_KV_FULL;
        d.dtype     = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim  = c.head_dim / (2 * c.ratio);
        *bk = rad_decl_kv_group(b, nm.f("kv_qsa_bkey"), &d);
        if (!*bk) return RAD_E_INVAL;
        /* qsa_score reads every complete block of the history each step, so this group stays resident
         * and holds every sequence's context while a sibling streams (kvstream/carve.h). */
        RAD_ARCH_TRY(rad_kv_group_dense_read(b, *bk));
    }
    kv_bk = *bk;
    /* Under M-RoPE a block key rotates at its first token's three components, which qsa_work
     * writes into a [work, 3] `bpos` when given the step's rotary planes. */
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, c.max_work), RAD_INT("n", c.head_dim),
                                RAD_INT("ratio", c.ratio), RAD_INT("rotary_dim", c.rot),
                                RAD_INT("block_size", c.ratio),
                                RAD_F64("eps", g.eps), RAD_F64("theta", g.theta),
                                RAD_F64("scale", g.rope_scale), RAD_F64("wadd", g.wadd_qk),
                                RAD_STR("dtype", g.dtype), RAD_STR("mode", g.rope_mode),
                                RAD_STR("sections", g.rope_sections) };
        const int n_ps = (int)(sizeof ps / sizeof ps[0]) - (g.rope_mc ? 0 : 2);
        op_bkey = rw(b, decl_rotary(b, g, "qsa_block_key", ps, n_ps, &w_kn, 1),
                     {w.stage, w.page, w.bpos}, {});
    }
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv_bk));

    op_tail = rw(b, RAD_OP(b, "qsa_tail_store",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                               RAD_INT("seqs", g.max_seqs),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.qk}, {});

    /* M is queries for both, up to a whole prefill chunk: the selection is per query. */
    op_score = rw(b, RAD_OP(b, "qsa_score",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                                RAD_INT("heads", c.heads), RAD_INT("blocks", c.max_blocks),
                                RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.q, w.nc}, {w.score});
    op_select = rw(b, RAD_OP(b, "qsa_select",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                 RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                 RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs)),
                      RAD_NOWEIGHTS),
                  {w.score, w.nc}, {w.sel, w.nsel, w.sequ});

    /* fn4 qsa_split: both forms, declared only when the start asked for them (the graph and buffers otherwise today's). */
    if (c.split && c.split_rows > 0 && w.qlist) {
        const int64_t lw = 4 + 2 * c.topk;
        op_selp1 = rw(b, RAD_OP(b, "qsa_select_part",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.split_rows),
                                    RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                    RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs),
                                    RAD_INT("parts", 1), RAD_INT("part", 0)),
                         RAD_NOWEIGHTS),
                     {w.score, w.nc}, {w.qlist});
        if (!op_selp1) return RAD_E_INVAL;
        if (g.world > 1 && w.qpart && w.qgath && c.head_dim == 128 && c.heads <= 4) {
            op_score_part = rw(b, RAD_OP(b, "qsa_score_part",
                                  RAD_PARAMS(RAD_RANGE("M", 1, c.split_rows), RAD_INT("n", c.head_dim),
                                             RAD_INT("heads", c.heads), RAD_INT("blocks", c.max_blocks),
                                             RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype),
                                             RAD_INT("parts", g.world), RAD_INT("part", g.rank)),
                                  RAD_NOWEIGHTS),
                              {w.q, w.nc}, {w.score});
            op_selp = rw(b, RAD_OP(b, "qsa_select_part",
                            RAD_PARAMS(RAD_RANGE("M", 1, c.split_rows),
                                       RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                       RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs),
                                       RAD_INT("parts", g.world), RAD_INT("part", g.rank)),
                            RAD_NOWEIGHTS),
                        {w.score, w.nc}, {w.qlist, w.qpart});
            op_qgath = rw(b, RAD_OP(b, "all_gather",
                             RAD_PARAMS(RAD_INT("world_size", g.world),
                                        RAD_INT("numel", c.split_rows * lw),
                                        RAD_STR("dtype", "f32"), RAD_INT("row", lw)),
                             RAD_NOWEIGHTS),
                         {w.qpart}, {w.qgath});
            if (!op_score_part || !op_selp || !op_qgath) return RAD_E_INVAL;
        }
        op_selm = RAD_OP(b, "qsa_select_merge",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.split_rows),
                                    RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                    RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs)),
                         RAD_NOWEIGHTS);
        if (!op_selm) return RAD_E_INVAL;
        {
            /* Arrays rather than rw(): the read set depends on which lists exist (rad_fp8.h: NormQuantFP8). */
            rad_buf rd[3] = { w.qlist, w.nc, w.qgath };
            rad_buf wr[3] = { w.sel, w.nsel, w.sequ };
            rad_op_reads (b, op_selm, rd, w.qgath && op_qgath ? 3 : 2);
            rad_op_writes(b, op_selm, wr, 3);
        }
    }

    /* When the attention group streams, the attention block resolves `sel` to ring slots after its
     * layer's store (AttnGatedFP8::step); prefill runs on the stage and never resolves. */
    return RAD_OK;
}

/* Stores the block keys of the blocks the pass completes and the tail of raw keys after them,
 * from the projection's raw-key head. False when the step has no batch for the indexer's
 * groups. */
inline bool QsaIndexer::store_keys(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t nw = (c.heads + 1) * c.head_dim;
    const int64_t koff = c.heads * c.head_dim;   /* the raw-key head, after the query heads */
    const RadKVGroupBatch* bkb = kv_batch(batch, kv_bk);
    const RadKVGroupBatch* tlb = kv_batch(batch, c.kv_tail);
    if (!bkb || !tlb) return false;
    /* Column 0 of the state row is the committed slot; the row is wider when speculating (linear
     * groups get a second copy), so use the batch's pitch. */
    const int64_t tw = tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;

    /* Invariant: the work list runs before the tail store. The gather reads a completing block's
     * raw keys, up to `ratio - 1` of which are in the tail from earlier steps and are about to be
     * overwritten. The ring also spaces overwrites by `ring` positions, but the order still decides
     * what a straddling block reads. */
    const int64_t work = (T + c.ratio - 1) / c.ratio + S;
    /* Block positions: one component per block, or three under M-RoPE with planes. Under M-RoPE
     * `bpos` is [max_work, 3]; this step passes [work, 3], or only column 0 when the pass has no
     * planes (qsa_work then writes component 0 and the block key reads the one-component form). */
    const bool mc = g.rope_mc && batch->rope_pos;
    RadOperand bpos = g.rope_mc ? RAD_B_COL(w.bpos, 0, mc ? 3 : 1) : brows(w.bpos, work);
    if (g.rope_mc) bpos.rows = work;
    RAD_ISSUE_N(c_, op_work, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index, RAD_I32, S, tw),
                praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                mc ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE,
                brows(w.stage, work * c.ratio), brows(w.page, work), bpos,
                brows(w.nc, S));
    RAD_ISSUE_N(c_, op_bkey, work,
                brows(w.stage, work * c.ratio), RAD_W(w_kn),
                brows(w.page, work), bpos,
                kv_cache(kv_bk, layer));
    RAD_ISSUE_N(c_, op_tail, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index, RAD_I32, S, tw));
    return true;
}

inline void QsaIndexer::store(RadCtx* c_, const ActFP8& in, const RadBatch* batch) const {
    if (!on()) return;
    proj.step(c_, in, w.qk, batch->n_tok);
    (void)store_keys(c_, batch);
}

inline void QsaIndexer::step(RadCtx* c_, const ActFP8& in, const RadBatch* batch,
                             const int32_t* win_cu) const {
    if (!on()) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t qw = c.heads * c.head_dim;
    const int64_t nw = (c.heads + 1) * c.head_dim;

    proj.step(c_, in, w.qk, T);
    /* One fused launch or `heads` + 1 unfused. The fused form takes the whole projection at its
     * row pitch and writes the query heads contiguous and rotated. `n_head` 0 is the k-only form:
     * no q_gate, q_w or q_out, and `gate_out` absent. `T <= qprep_rows` is a correctness bound
     * (see declare()). */
    if (op_qprep && T <= qk_fuse_now(qprep_rows) && !rope_mixed(batch)) {
        RAD_ISSUE_N(c_, op_qprep, T,
                    RAD_NONE, brows(w.qk, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_NONE, RAD_W(w_qn),
                    RAD_NONE, brows(w.q, T), RAD_NONE);
    } else {
        for (int64_t h = 0; h < c.heads; ++h)
            RAD_ISSUE_N(c_, op_norm, T,
                        bcol_at(w.qk, 0, nw, h * c.head_dim, c.head_dim, T), RAD_W(w_qn),
                        bcol_at(w.q,  0, qw, h * c.head_dim, c.head_dim, T));
        RAD_ISSUE_N(c_, op_rope, T, brows(w.q, T), rope_posmc(g, batch, T));
    }

    /* The keys first: the selection below reads the blocks they complete. */
    const RadKVGroupBatch* bkb = kv_batch(batch, kv_bk);
    const RadKVGroupBatch* atb = kv_batch(batch, c.kv_attn);
    if (!atb || !store_keys(c_, batch)) return;

    /* Score and select per query: both ops take `cu`, the block table stays per sequence, and `m`
     * is a query row, so each query of a prefill chunk selects from the shared history, as the
     * checkpoint does. */
    if (!c.emit_sel) return;

    /* Narrow the score columns to the live history. Both kernels size their grid from the
     * operand's width (the live count is device data), so the full `max_ctx` width would make a
     * 2000-token prompt walk 50000 block slots; the pitch stays declared, only the column count
     * moves.
     *
     * This must be an upper bound: an underestimate silently drops blocks. A query has at most
     * (position + 1) / ratio complete blocks; positions are bounded by the host context bound plus
     * the step's tokens plus the draft window; the +1 keeps `n < nb` so the partial page `bt[n]`
     * is readable. */
    const int64_t reach = (int64_t)batch->max_ctx_len + batch->n_tok + batch->n_spec + c.ratio;
    int64_t live = reach / c.ratio + 1;
    if (live > c.max_blocks) live = c.max_blocks;
    if (live < 1) live = 1;

    /* A step wider than the score plane is one sequence (Config::plane_rows), scored and selected
     * in windows of the plane's rows from the front, the last one ending at the step's end and
     * overlapping the one before. A window is handed as a sequence of its own rows (`win_cu` =
     * {0, rows}). Every operand a query row reads is its own (query, position, selection) and the
     * block table and complete-block count are the step's, so each row is scored and selected as
     * in one pass over the step, and a row two windows share is written twice with the same
     * bytes. */
    const int64_t PR = c.plane_rows > 0 ? c.plane_rows : T;

    /* fn4 qsa_split (see fn4_qsasplit_flag). Host values only: every rank takes the same path. */
    const int sf = c.split ? (fn4_qsasplit_flag().load(std::memory_order_relaxed) & c.split) : 0;
    if (sf && T <= c.split_rows && T <= PR && live >= c.split_min && op_selm) {
        const int64_t lw = 4 + 2 * c.topk;
        const RadOperand sco  = bcol_at(w.score, 0, c.max_blocks, 0, live, T);
        const RadOperand atbt = praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch);
        if ((sf & 1) && op_score_part && op_selp && op_qgath) {
            /* The widest range a rank owns, in chunks of at most ~4096 blocks (8 at most: <= 4099 at 524,288 tokens). */
            const int64_t width = (live + g.world - 1) / g.world + 15;
            int64_t nl = (width + 4095) / 4096;
            if (nl < 1) nl = 1;
            if (nl > 8) nl = 8;
            RAD_ISSUE_N(c_, op_score_part, T,
                        brows(w.q, T), kv_cache(kv_bk, layer),
                        praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                        praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S), sco);
            RAD_ISSUE_N(c_, op_selp, T,
                        sco, atbt, brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens, RAD_I32, S + 1),
                        bcol_at(w.qlist, 0, 8 * lw, 0, nl * lw, T), brows(w.qpart, T));
            RAD_ISSUE(c_, op_qgath, brows(w.qpart, T), brows(w.qgath, T));
            RAD_ISSUE_N(c_, op_selm, T,
                        brows(w.qgath, T), atbt, brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens, RAD_I32, S + 1),
                        brows(w.sel, T), brows(w.nsel, T), brows(w.sequ, T));
            return;
        }
        if ((sf & 2) && op_selp1 && live <= 8 * 8192) {
            int64_t nl = (live + 4095) / 4096;
            if (nl < 1) nl = 1;
            if (nl > 8) nl = 8;
            const RadOperand lists = bcol_at(w.qlist, 0, 8 * lw, 0, nl * lw, T);
            RAD_ISSUE_N(c_, op_score, T,
                        brows(w.q, T), kv_cache(kv_bk, layer),
                        praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                        praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S), sco);
            RAD_ISSUE_N(c_, op_selp1, T,
                        sco, atbt, brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens, RAD_I32, S + 1), lists, RAD_NONE);
            RAD_ISSUE_N(c_, op_selm, T,
                        lists, atbt, brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens, RAD_I32, S + 1),
                        brows(w.sel, T), brows(w.nsel, T), brows(w.sequ, T));
            return;
        }
    }

    if (T > PR) {
        if (S != 1 || !win_cu) {
            rad_step_fail(c_, "qsa: a step wider than the score plane must be one sequence with "
                              "its window bounds (RadBuildCtx::max_mixed_tok)");
            return;
        }
        const int reps = (T > 256 && fn4_qsaprobe_flag().load(std::memory_order_relaxed)) ? 2 : 1;   /* PF probe QSP */
        for (int64_t w0 = 0;;) {
          for (int rep = 0; rep < reps; ++rep) {
            RAD_ISSUE_N(c_, op_score, PR,
                        brow_slice(w.q, w0, PR, qw), kv_cache(kv_bk, layer),
                        praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                        praw(win_cu, RAD_I32, 2), brows(w.nc, S),
                        bcol_at(w.score, 0, c.max_blocks, 0, live, PR));
            RAD_ISSUE_N(c_, op_select, PR,
                        bcol_at(w.score, 0, c.max_blocks, 0, live, PR),
                        praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch),
                        brows(w.nc, S), praw(batch->positions + w0, RAD_I32, PR),
                        praw(win_cu, RAD_I32, 2),
                        brow_slice(w.sel, w0, PR, c.topk + 1), brow_slice(w.nsel, w0, PR, 1),
                        brow_slice(w.sequ, w0, PR, 1));
          }
            if (w0 + PR >= T) break;
            w0 = w0 + PR + PR <= T ? w0 + PR : T - PR;
        }
        return;
    }
    const int reps = (T > 256 && fn4_qsaprobe_flag().load(std::memory_order_relaxed)) ? 2 : 1;   /* PF probe QSP */
    for (int rep = 0; rep < reps; ++rep) {
    RAD_ISSUE_N(c_, op_score, T,
                brows(w.q, T), kv_cache(kv_bk, layer),
                praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                bcol_at(w.score, 0, c.max_blocks, 0, live, T));
    RAD_ISSUE_N(c_, op_select, T,
                bcol_at(w.score, 0, c.max_blocks, 0, live, T),
                praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch),
                brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                brows(w.sel, T), brows(w.nsel, T), brows(w.sequ, T));
    }

    /* RADIANCE_DEBUG_QSA: prints the first few selections (nc, nsel, sequ, leading sel and score
     * entries) to tell a divergence from intended dropping (`nc` above `topk`). Synchronises the
     * stream; off unless set. */
    static const bool dbg = std::getenv("RADIANCE_DEBUG_QSA") != nullptr;
    if (dbg) {
        static std::atomic<int> seen{0};
        if (seen.fetch_add(1) < 6) {
            rad_stream_sync(rad_stream(c_));
            std::vector<int32_t> hnc(1), hns(1), hsu(1), hsel(8);
            std::vector<float>   hsc(8);
            rad_memcpy_async(hnc.data(), rad_buf_ptr(c_, w.nc),   4, rad_stream(c_));
            rad_memcpy_async(hns.data(), rad_buf_ptr(c_, w.nsel), 4, rad_stream(c_));
            rad_memcpy_async(hsu.data(), rad_buf_ptr(c_, w.sequ), 4, rad_stream(c_));
            rad_memcpy_async(hsel.data(), rad_buf_ptr(c_, w.sel), 32, rad_stream(c_));
            rad_memcpy_async(hsc.data(), rad_buf_ptr(c_, w.score), 32, rad_stream(c_));
            rad_stream_sync(rad_stream(c_));
            std::fprintf(stderr, "D qsa L%d nc=%d nsel=%d sequ=%d sel[%d %d %d %d] "
                                 "score[%.4g %.4g %.4g %.4g]\n", layer,
                         hnc[0], hns[0], hsu[0], hsel[0], hsel[1], hsel[2], hsel[3],
                         hsc[0], hsc[1], hsc[2], hsc[3]);
        }
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_QSA_H */
