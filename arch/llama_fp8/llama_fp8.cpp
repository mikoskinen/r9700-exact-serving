/* llama_fp8 -- the plain LLaMA graph with block-scaled FP8 weights and FP8 activations.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * `model_type: llama` with `LlamaForCausalLM`, as MiniCPM5-2B reports: 42 layers, hidden 2048, 16
 * query heads over 2 KV heads at head_dim 128, SwiGLU at 6144, RMS norms, full rotary at theta
 * 5e6, an untied 130560-row head, and no biases. Every layer is full attention; unlike
 * arch/qwen35_fp8 there is no gated delta net, layer schedule, output gate or per-head q/k norm.
 *
 * ============================== TWO CHECKPOINT FORMATS, ONE GRAPH ==============================
 *
 * The selection key is the (architecture, quantisation) pair (spec §2.4) and one .so claims one
 * pair, so this source builds twice:
 *
 *   llama_fp8        quant "fp8_e4m3"    the checkpoint holds E4M3 codes and a `weight_scale_inv`
 *                                        plane per linear, as the engine reads them.
 *   llama_dense_fp8  quant ""            the checkpoint is dense bf16, and a recipe quantises each
 *                                        linear to those two planes at convert time
 *                                        (data/recipes/minicpm5-2b-dspark.recipe: amax per 128x128
 *                                        block, scale = amax/448 rounded to bf16).
 *
 * MiniCPM5-2B needs the second, since it is published only in bf16. Both builds declare the same
 * weights and ops, so nothing downstream can tell which one ran.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 *   E4M3 + a bf16 [N/128][K/128] plane   self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj
 *   BF16, no plane                       every norm and the embedding
 *
 * The lm_head is declared at whatever the model holds: bf16 as llama checkpoints ship it, or block
 * fp8 where a recipe quantised it (the widest GEMM at half the bytes), and the logits kernel for
 * that encoding is selected. Both vocabulary edges are row-sharded with the vocabulary.
 */
#ifndef LLAMA_DENSE_SRC
#define LLAMA_DENSE_SRC 0
#endif
#ifndef LLAMA_NS
#define LLAMA_NS llama_fp8
#endif
#ifndef LLAMA_QUANT
#define LLAMA_QUANT "fp8_e4m3"
#endif

#include <cstdlib>
#include <cstring>
#include <vector>

#include "rad_pp.h"

#include <arch/rad_arch.h>
#include <arch/rad_block_attn_fp8.h>
#include <arch/rad_block_dspark.h>
#include <arch/rad_block_mlp_fp8.h>

namespace LLAMA_NS {

using namespace rad::arch;

struct Layer {
    AttnFP8 attn;
    MlpFP8  mlp;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t n_rot = 0;
    int64_t n_main = 0;

    rad_kvgroup kv_full = 0;

    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActFP8  a_h{};
    rad_buf b_qkv = 0;
    rad_buf b_q   = 0;
    ActFP8  a_attn{};
    rad_buf b_gate_up = 0;
    ActFP8  a_ffn{};

    /* Auxiliary hidden-state taps, declared only when the container carries a drafter that wants
     * them. A DSpark or DFlash drafter reads the residual stream at several intermediate layers
     * and projects the concatenation down (`fc` is [n_embd, n_taps * n_embd]), so the input must
     * be one contiguous row. The trunk's one residual buffer is overwritten by every layer, so
     * each tap's rows are copied into its column slice as they pass. Same mechanism as
     * arch/qwen35_fp8. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    /* The drafter, declared only when the container carries one and this serve speculates. Its
     * layers get absolute indices after the trunk's, so the KV group's layer_slot lookup finds
     * them and its weights land in layer groups of their own. */
    Dspark       ds{};
    Dspark::Wire dsw{};
    rad_kvgroup  kv_draft = 0;
    bool         have_ds = false;
    /* tp8 loop D38a: a pipeline follower beside the leader's drafter: no taps, no drafter, the drafter's KV group declared
     * zeroed (deepseek41's pp_dspark_zero: the nodes' groups stay one list) */
    bool         pp_draft_zero = false;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;

    std::vector<Layer> layers;

    /* tp8 S4: a node of a two-node pipeline (abi/rad_pp.h), W(h, t): the leader (rank 0) holds
     * the embedding, layers [0, h) and [t, n) and the head, the follower layers [h, t). Off (pp
     * false) on one machine, and then nothing below is declared or issued. A crossing carries what
     * the receiving side's next op reads: the residual stream, and where that op folds the
     * previous block's delta into it (layer_fold's mix_fold) the delta too, so the receiving
     * layer's norm adds them as it does in one process. */
    bool     pp = false;
    int      pp_rank = 0, pp_h = 0, pp_t = 0;
    std::vector<char> held;              /* per trunk layer: declared and run on this node */
    rad_op   op_x1 = 0, op_x2 = 0;       /* stage_send / stage_recv of each crossing */
    int      parts_x1 = 0, parts_x2 = 0; /* 1: the stream; 2: the stream and the pending delta */
    RadPpRing* ring[RAD_PP_N_RINGS] = {};
    /* tp8 S6, a host test of the KV copies (RADIANCE_LLAMA_PP_KVCOPY=1, W(h > 0, t); unset: nothing here): X1 also
     * carries layer h - 1's fused q|k|v rows (the inputs of its kv_store) and the follower stores them into a copy of
     * that layer's cache with the same op at the same slots, as deepseek41's copied sources are (a llama layer reads
     * no other layer's cache, so nothing reads the copy: the core's RADIANCE_PP_KV_CHECK compares it with the owner's).
     * RADIANCE_PP_TEST_KVCOPY_SKIP=N (follower) leaves the copy out at its N-th trunk pass, for the check's own test. */
    bool     kvcopy = false;
    rad_op   op_kvcopy = 0;
    int64_t  kvcopy_skip = -1, kvcopy_pass = 0;
    /* tp8 loop P28, a host test of the engine's band passes (RADIANCE_LLAMA_PP_BAND=1 on both nodes, W(h > 0, t), t > h + 1;
     * unset: nothing here): the leader also declares layer h (the band of one layer, whose decode owner stays the
     * follower); on a band pass (RadPpInfo.reserved[0] == 1) its head runs [0, h + 1) and sends X1 after layer h -- the
     * stream (+ the delta) and layer h's fused q|k|v rows -- and the follower stores layer h's k / v into its own cache
     * at the batch's slots (the owner's op), then runs [h + 1, t). Every other pass is W(h, t)'s. */
    int      band = 0;
    int      parts_x1b = 0;
    bool     band_skip = false;   /* RADIANCE_PP_TEST_BAND_SKIP=1 (follower): the store left out -- the test's negative control */
    rad_op   op_x1b = 0, op_band_st = 0;
    /* tp8 loop D39, a host test of the engine's TAIL band (RADIANCE_LLAMA_PP_TBAND=1 on both nodes, W(h, t), t - 1 > h; unset:
     * nothing here): the leader also declares layer t - 1 (whose decode owner stays the follower); on a tail band pass
     * (RadPpInfo.reserved[2] == 1) the follower runs [h, t - 1), sends X2 there, and then only layer t - 1's k / v into its own
     * cache from its own rows (the layer's norm, q|k|v projection, rope and kv_store: the ops the layer issues first, at the same
     * rows); the leader's tail runs [t - 1, n). Every other pass is W(h, t)'s. */
    int      tband = 0;
    int      parts_x2b = 0;
    bool     tband_skip = false;  /* RADIANCE_PP_TEST_TBAND_SKIP=1 (follower): the k / v stores left out -- the negative control */
    rad_op   op_x2b = 0;
    /* tp8 dev 1a (RADIANCE_PP_FOLLOWER_EMBED=1 on both nodes, W(0, t) only; unset: nothing here): the follower looks up
     * the step's embedding itself (the same op over the same table, the vocab-sharded rows completed by the same
     * all-reduce) where it took X1, and the leader's head part issues neither the lookup nor X1: the follower starts a
     * lane step from its build, never from the leader's stream. */
    bool     fe = false;
    /* tp8 dev D10 round 2 (RADIANCE_PP_ISSUE_AHEAD=1 with fe; unset: nothing here): X1 is declared as the step's token ids
     * alone (the leader's batch token ids on its card, ADV_TRUNK's for a step issued ahead of its collect), the leader's head
     * part sends it and the follower takes it into its own batch's token ids before its lookup, so a step the leader ships
     * before it knows the ids (its frame's are the host's) embeds the card's. */
    bool     x1_ids = false;
    /* tp8 dev M4 (RADIANCE_PP_SEG_LAYERS=k on the follower, --pp-lanes > 1; unset: nothing here): the follower's [h, t) in
     * segments of k layers; at each boundary s the segment before it saves the stream (and the pending delta where layer s
     * folds it) into this rank's ring buffer and the segment from s takes them back (the stage ops' loop mode), so other
     * lanes' passes may run between the two. */
    std::vector<int32_t> seg_b;           /* the boundaries, ascending, in (h, t) */
    std::vector<rad_op>  seg_send, seg_recv;
    std::vector<int>     seg_parts;       /* 1 or 2, per boundary */
    rad_buf  b_segring = 0;
    rad_buf  b_segring2 = 0;   /* tp8 loop D32 (RADIANCE_PP_LEAD_SEG_RINGS=2, the leader): the odd lanes' BG segments */
    int64_t  segring_bytes = 0;
};

static Model g_model[MAX_RANKS];

/* tp8 loop D33 (RADIANCE_PP_SEG_MAX=N, as deepseek41's; unset 16 = the bound before D33): at most N - 1 segment boundaries on
 * this node, 2 <= N <= RAD_PP_MAX_SEGS -- the host tests' stand-in for the follower's end lump */
static int pp_seg_cap() {
    const char* e = std::getenv("RADIANCE_PP_SEG_MAX");
    const int n = e && *e ? std::atoi(e) : 16;
    return std::min(std::max(n, 2), (int)RAD_PP_MAX_SEGS);
}

/* The quantisation descriptor is the selection key but is checked anyway: `--arch` can override
 * the architecture id at convert time, and a plugin handed the wrong checkpoint should say which
 * half disagrees rather than fail at a name map. */
static int check_format(const RadModelMeta* meta) {
    const char* q = meta->quant ? meta->quant : "";
#if LLAMA_DENSE_SRC
    if (*q) {
        fprintf(stderr, "radiance: llama_dense_fp8 serves an unquantised checkpoint and this one "
                        "declares '%s'; llama_fp8 serves it\n", q);
        return RAD_E_UNSUPPORTED;
    }
#else
    if (std::strncmp(q, "fp8", 3) != 0) {
        fprintf(stderr, "radiance: llama_fp8 serves a block-scaled fp8 checkpoint and this one "
                        "declares '%s'; llama_dense_fp8 serves a dense bf16 one\n",
                *q ? q : "(unquantised)");
        return RAD_E_UNSUPPORTED;
    }
#endif
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: llama has no routed feed-forward and this checkpoint declares "
                        "%lld experts\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The DSpark drafter. Its Geom is the trunk's with the drafter's layer geometry from `draft.*`
 * substituted; the vocabulary, deployment bounds, rank and wire stay the trunk's.
 *
 * Nothing is defaulted quietly: a `draft.` config that does not describe a drafter this file
 * implements is refused by name, because a silently ignored scale factor or head type would only
 * lower acceptance (spec §17).
 *
 * The confidence head (`AcceptRatePredictor`) is deliberately ignored: it drives adaptive block
 * length, this engine's depth is fixed at declare, and it changes nothing the drafter proposes
 * (rad_block_dspark.h). */
static int declare_dspark(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Plain RMS norm: the reference uses Qwen3RMSNorm (`model_type: qwen3`). It matches this
     * MiniCPM5 target, but it is the drafter's convention, set on the drafter's own Geom. */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DSpark"));

    Dspark::Config dc{};
    dc.n_taps = m.n_taps;
    dc.rank   = rad_meta_geti(meta, "draft.markov_rank", 0);
    dc.causal = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.sliding_window", 0);
    /* How many candidates the Markov walk re-ranks. The reference biases all 130560 logits and
     * takes the argmax; this re-ranks the base head's top K, the head's only approximation, which
     * can cost acceptance but not correctness (rad_block_dspark.h). The selector bounds K at 32;
     * the default 16 is 8 KiB of codebook per position, and a checkpoint may ask for more. */
    dc.top_k  = rad_meta_geti(meta, "draft.markov_top_k", 16);
    if (dc.top_k < 1 || dc.top_k > 32) {
        fprintf(stderr, "radiance: draft.markov_top_k is %lld; the Markov walk scores at most 32 "
                        "candidates a position\n", (long long)dc.top_k);
        return RAD_E_UNSUPPORTED;
    }

    /* The block is both the number of noise rows and the number of drafts: every row, the anchor
     * included, predicts the token after it. So the depth is `block_size`, where a DFlash2
     * container's is `block_size - 1` (see RadDrafterDecl::block). A wrong value runs but accepts
     * nothing.
     *
     * A block-diffusion drafter is trained for one block length and has no partial block, so any
     * other depth is refused. The probe reports `draft_depth_fixed`, so the engine normally picks
     * this already. */
    const long long trained = rad_meta_geti(meta, "draft.block_size", 0);
    if (trained < 2) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no draft.block_size; the "
                        "number of masked positions it denoises is not a number this engine may "
                        "choose.\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec != trained) {
        fprintf(stderr, "radiance: the DSpark drafter was trained at block_size %lld, which is "
                        "%lld draft tokens, and this serve asks for %d. A block-diffusion drafter "
                        "denoises a fixed number of mask positions; run it at %lld.\n",
                trained, trained, ctx->max_spec, trained);
        return RAD_E_UNSUPPORTED;
    }
    dc.block = trained;

    /* Two variants this implementation does not apply; each changes the proposals, so neither is
     * ignored quietly. */
    const char* mh = rad_meta_gets(meta, "draft.markov_head_type", "vanilla");
    if (dc.rank <= 0 || !mh || std::strcmp(mh, "vanilla") != 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint has markov_rank %lld and "
                        "markov_head_type '%s'. Only the rank-N 'vanilla' bigram head is "
                        "implemented, and it is what the published checkpoints ship.\n",
                (long long)dc.rank, mh ? mh : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    if (dc.rank % 32 || dc.rank > 512) {
        fprintf(stderr, "radiance: markov_rank %lld; the walk that applies the bigram bias takes "
                        "a multiple of 32 up to 512\n", (long long)dc.rank);
        return RAD_E_UNSUPPORTED;
    }

    /* DSpark's published config sets `sliding_window: null`, so this is normally a full cache. */
    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    /* Activations. The context pass has one row per committed token over a whole prefill chunk;
     * the query pass one row per draft position, `block` per sequence; the two are sized apart.
     * All are small: the drafter's cost is its weights. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dspark::Wire& dw = m.dsw;
    dw.src  = m.b_dfsrc;
    dw.c    = decl_b(b, nm.f("ds_c"),  dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck   = decl_b(b, nm.f("ds_ck"), dg.act_dtype, {dg.max_tok, kvd});
    dw.cv   = decl_b(b, nm.f("ds_cv"), dg.act_dtype, {dg.max_tok, kvd});
    /* FP8 copies for the context pass. `srcq` is the largest (20 MiB at a 2048-token chunk, five
     * taps, hidden 2048) and halves the bytes of the context pass's dominant GEMM. */
    dw.srcq = decl_b(b, nm.f("ds_srcq"), RAD_F8E4M3, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs = decl_b(b, nm.f("ds_srcs"), RAD_F32,
                     {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq   = decl_b(b, nm.f("ds_cq"), RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs   = decl_b(b, nm.f("ds_cs"), RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});

    dw.x      = decl_b(b, nm.f("ds_x"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.x    = decl_b(b, nm.f("ds_h"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.q    = decl_b(b, nm.f("ds_h.q"), RAD_F8E4M3, {qr, dg.n_embd});
    dw.h.s    = decl_b(b, nm.f("ds_h.s"), RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.q      = decl_b(b, nm.f("ds_q"), dg.act_dtype, {qr, qd});
    dw.k      = decl_b(b, nm.f("ds_k"), dg.act_dtype, {qr, kvd});
    dw.v      = decl_b(b, nm.f("ds_v"), dg.act_dtype, {qr, kvd});
    dw.attn.x = decl_b(b, nm.f("ds_attn"), dg.act_dtype, {qr, qd});
    dw.attn.q = decl_b(b, nm.f("ds_attn.q"), RAD_F8E4M3, {qr, qd});
    dw.attn.s = decl_b(b, nm.f("ds_attn.s"), RAD_F32,    {qr, fp8_blocks(qd)});
    dw.gate_up = decl_b(b, nm.f("ds_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn.x  = decl_b(b, nm.f("ds_ffn"), dg.act_dtype, {qr, dg.n_ff});
    dw.ffn.q  = decl_b(b, nm.f("ds_ffn.q"), RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffn.s  = decl_b(b, nm.f("ds_ffn.s"), RAD_F32,    {qr, fp8_blocks(dg.n_ff)});

    dw.sel_h  = decl_b(b, nm.f("ds_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.dq     = decl_b(b, nm.f("ds_dq"), RAD_I8,  {sr, dg.n_embd});
    dw.ds     = decl_b(b, nm.f("ds_ds"), RAD_F32, {sr, 1});
    /* Group-major, [n_embd/group][ceil(M/16)*16]: quant_act_i8's asum plane, whose first
     * dimension is the K group, not the token. */
    dw.dsum   = decl_b(b, nm.f("ds_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* This rank's plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded; asking it keeps the buffer consistent with the op declared over it. */
    dw.dlog   = decl_b(b, nm.f("ds_dlog"), dg.act_dtype, {sr, Dspark::head_rows(dg)});
    dw.cand   = decl_b(b, nm.f("dspark_cand"), RAD_I32, {sr, dc.top_k});
    dw.unary  = decl_b(b, nm.f("ds_unary"), RAD_F32, {sr, dc.top_k});
    if (Dspark::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("ds_dpair"), RAD_F32, {sr, dc.top_k, 2});
        dw.dgath = decl_b(b, nm.f("ds_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.top_k, 2});
    }
    /* The proposal buffer (RadDrafterDecl::proposal): a block pass proposes through the
     * drafter's own walk, never through the sampler's token buffer. */
    dw.tokens = decl_b(b, nm.f("dspark_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    /* The same lm_head sources as the trunk's head (embedding as the tied alternative), so both
     * project onto one vocabulary. */
    RAD_ARCH_TRY(m.ds.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), nm.ckpt("model.embed_tokens.weight"),
                              dw));
    m.have_ds = true;

    /* Block drafter: a context pass mirroring the trunk step, then one pass filling the whole
     * block. `block` is `depth`, not `depth + 1` (this drafter's training convention). `window`
     * must be the one this plugin gated its own attention with, since the scheduler stages the
     * block pass against it. */
    RadDrafterDecl drf{};
    drf.name           = "dspark";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.mask_token_id", -1);
    if (drf.mask_token < 0)
        drf.mask_token = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no mask_token_id, which every "
                        "unfilled block position embeds\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    rad_note(b, "DSpark drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "over %lld noise row(s), the anchor included, proposes %lld tokens",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.block, (long long)dc.steps());
    rad_note(b, "the rank-%lld Markov bigram head walks `unary + <markov_w1[prev], "
                "markov_w2[cand]>` over the draft head's top %lld, greedily; the confidence head "
                "(adaptive block length) is not applied",
             (long long)dc.rank, (long long)dc.top_k);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the activation dtype: everything but the two GEMM operands runs at it. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    Geom&  g  = m.g;
    Names& nm = m.nm;
    m.n_main  = g.n_layer;
    m.held.assign((size_t)m.n_main, 1);
    {
        const RadPpInfo pp = rad_pp_query();
        if (pp.nodes == 2) {
            if (pp.cut_h < 0 || pp.cut_t <= pp.cut_h || pp.cut_t > m.n_main) {
                fprintf(stderr, "radiance: llama: --pp-cut %d,%d is not a range of this model's %lld "
                                "layers\n", pp.cut_h, pp.cut_t, (long long)m.n_main);
                return RAD_E_INVAL;
            }
            /* World 2 defers each block's all-reduce into the next block's norm (layer_fold), so the
             * delta at the cut would be each rank's own partial: not yet carried. */
            if (g.world == 2) {
                fprintf(stderr, "radiance: llama: a two-node pipeline at --tp 2 is not supported (the "
                                "cut would carry each rank's partial sum); use --tp 1 or 4\n");
                return RAD_E_UNSUPPORTED;
            }
            m.pp = true;
            m.pp_rank = pp.rank;
            m.pp_h = pp.cut_h;
            m.pp_t = pp.cut_t;
            int64_t n_held = 0;
            for (int64_t l = 0; l < m.n_main; ++l) {
                m.held[(size_t)l] = ((l >= m.pp_h && l < m.pp_t) == (m.pp_rank == 1)) ? 1 : 0;
                n_held += m.held[(size_t)l];
            }
            /* Every node binds a layer to the KV group, whose block size comes off its attention
             * op (rad_kv_block_size): a leader of W(0, n) would hold none. */
            if (n_held == 0) {
                fprintf(stderr, "radiance: llama: --pp-cut %d,%d leaves the %s no trunk layer; each node "
                                "holds at least one\n", m.pp_h, m.pp_t, m.pp_rank == 0 ? "leader" : "follower");
                return RAD_E_UNSUPPORTED;
            }
            const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
            m.fe = fe && !std::strcmp(fe, "1") && m.pp_h == 0;
            /* tp8 loop P28 test stand-in: the band of one layer (layer h declared on the leader too) */
            if (const char* bd = std::getenv("RADIANCE_LLAMA_PP_BAND"); bd && !std::strcmp(bd, "1") && m.pp_h > 0 &&
                m.pp_h + 1 < m.pp_t && !m.fe && pp.lanes > 1) {
                m.band = 1;
                if (m.pp_rank == 0) m.held[(size_t)m.pp_h] = 1;
            }
            /* tp8 loop D39 test stand-in: the tail band of one layer (layer t - 1 declared on the leader too) */
            if (const char* tb = std::getenv("RADIANCE_LLAMA_PP_TBAND"); tb && !std::strcmp(tb, "1") && !m.band &&
                m.pp_t - 1 > m.pp_h && !m.fe && pp.lanes > 1) {
                m.tband = 1;
                if (m.pp_rank == 0) m.held[(size_t)m.pp_t - 1] = 1;
            }
            /* tp8 dev D10 round 2: beside the follower's own lookup, X1 is the step's token ids (the follower reads them) */
            const char* ia = std::getenv("RADIANCE_PP_ISSUE_AHEAD");
            m.x1_ids = m.fe && ia && !std::strcmp(ia, "1");
            const char* sk = std::getenv("RADIANCE_PP_SEG_LAYERS");
            const int k = sk && *sk ? std::atoi(sk) : 0;
            if (m.pp_rank == 1 && k > 0 && pp.lanes > 1)
                for (int x = m.pp_h + k; x < m.pp_t && (int)m.seg_b.size() < pp_seg_cap() - 1; x += k) m.seg_b.push_back(x);
            /* tp8 dev M9 (RADIANCE_PP_LEAD_SEG=k on the leader): its head [0, h) and its tail [t, n) in segments of k layers
             * -- boundaries k, 2k .. below h, and t + k, t + 2k .. below n (none past the drafter's first tap: the taps
             * and the drafter after the tail must be one segment's) */
            const char* lk = std::getenv("RADIANCE_PP_LEAD_SEG");
            const int kl = lk && *lk ? std::atoi(lk) : 0;
            if (m.pp_rank == 0 && kl > 0 && pp.lanes > 1) {
                for (int x = kl; x < m.pp_h && (int)m.seg_b.size() < pp_seg_cap() - 1; x += kl) m.seg_b.push_back(x);
                for (int x = m.pp_t + kl; x < (int)m.n_main && (int)m.seg_b.size() < pp_seg_cap() - 1; x += kl)
                    m.seg_b.push_back(x);
            }
        }
    }

    /* The rotary width is the whole head unless `partial_rotary_factor` says otherwise (Phi-style
     * models set it). A config that states nothing means the full head. */
    {
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        m.n_rot = prf > 0.0 ? (int64_t)(g.head_dim * prf) : g.head_dim;
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: llama: rotary width %lld is not an even width inside head_dim "
                        "%lld\n", (long long)m.n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* ---- the KV cache. One group: every layer is full attention. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* ---- activations. */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    m.b_x      = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd));
    m.b_qkv    = decl_b(b, nm.f("attn_qkv"), g.act_dtype, {g.max_tok, q_dim + 2 * kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim));
    m.b_gate_up = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff));
    m.b_hout   = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* ---- the drafter's taps, only when the container carries a drafter that reads them and the
     * deployment speculates: the buffer is one residual stream per tap, 40 MiB at a 2048-token
     * chunk with five taps and hidden 2048.
     *
     * Two key spellings: a SpecForge DFlash config nests the wiring under `dflash_config`, while
     * the published DSpark configs (openbmb/MiniCPM5-2B-DSpark and siblings) put it at the top
     * level. The flat key is read first. */
    if (rad_ctx_draft_head(ctx)) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
            const char* ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
            if (!ids) ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
            m.n_taps = parse_layer_ids(ids, m.n_main, m.tap_layer, Model::MAX_TAPS);
            /* tp8 loop D38a: in a two-node pipeline the drafter is the leader's -- its context pass reads the taps after the
             * tail, so every tap must be a leader layer ([0, h) or [t, n)); the follower declares no tap and no drafter,
             * only the drafter's KV group, zeroed (below, where the leader declares the drafter) */
            if (m.pp && m.n_taps > 0) {
                for (int t = 0; t < m.n_taps; ++t)
                    if (m.tap_layer[t] >= m.pp_h && m.tap_layer[t] < m.pp_t) {
                        fprintf(stderr, "radiance: llama: the drafter reads trunk layer %d, which W(%d,%d) gives the follower; "
                                        "in a two-node pipeline every tap must be a leader layer\n",
                                m.tap_layer[t], m.pp_h, m.pp_t);
                        return RAD_E_UNSUPPORTED;
                    }
                if (m.pp_rank == 1) {
                    m.n_taps = 0;
                    m.pp_draft_zero = true;
                }
            }
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("draft_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            } else if (!m.pp_draft_zero) {
                fprintf(stderr, "radiance: llama: the container carries a '%s' drafter but names "
                                "no trunk layer this model has; it is paired with a different "
                                "target\n", da);
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    const bool shell = !m.pp || m.pp_rank == 0;   /* embedding, head and logits: the leader's */
    /* tp8 dev 1a: the follower's own lookup -- the table, the op and its all-reduce, declared where one process
     * declares them (before the layers), nothing of the head */
    if (!shell && m.fe) {
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    }
    if (shell) {
    m.b_logits = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded with the vocabulary. A tied head is the
     * alternative source: without `lm_head.weight` the embedding table serves as both, as in small
     * LLaMA derivatives. The head is declared at its codes' dtype (bf16, or E4M3 where a recipe
     * made it block fp8), queried after its name map entry. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* Always exact: each token's embedding comes from one rank and the others contribute zero, so
     * a lossy wire would only add error. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    }

    AttnFP8::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qkv = m.b_qkv; aw.q = m.b_q; aw.attn = m.a_attn;
    MlpFP8::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;

    m.layers.resize((size_t)m.n_main);
    /* tp8 dev M9: no leader tail boundary past the drafter's first tap (the taps and the drafter after the tail are one
     * segment's: another lane's tail between two segments would rewrite the tap buffer) */
    if (m.pp && m.pp_rank == 0 && m.n_taps > 0) {
        int top = (int)m.n_main;
        for (int t = 0; t < m.n_taps; ++t) top = std::min(top, m.tap_layer[t]);
        std::vector<int32_t> keep;
        for (int32_t x : m.seg_b)
            if (x < m.pp_h || x <= top) keep.push_back(x);
        m.seg_b.swap(keep);
    }
    /* tp8 dev M4: the segment boundaries' parts (as a crossing into layer s carries: the stream, and the delta where s folds
     * it), this rank's ring buffer for them (header + the widest boundary's parts at max_tok rows + the trailer), and each
     * boundary's save (after layer s - 1) and restore (before layer s) in loop mode */
    if (!m.seg_b.empty()) {
        const int64_t row = g.n_embd * (int64_t)rad_dtype_bytes(g.act_dtype, 1);
        int widest = 1;
        for (int32_t x : m.seg_b) {
            const int np = layer_fold(g, x, m.n_main, m.tap_layer, m.n_taps, m.op_embed_ar != 0).mix_fold ? 2 : 1;
            m.seg_parts.push_back(np);
            widest = std::max(widest, np);
        }
        const int64_t part = (g.max_tok * row + RAD_PP_PART_ALIGN - 1) / RAD_PP_PART_ALIGN * RAD_PP_PART_ALIGN;
        m.segring_bytes = RAD_PP_RING_HDR + part * widest + RAD_PP_TRAILER;
        m.b_segring = decl_b(b, nm.f("pp_segring"), RAD_U8, {m.segring_bytes}, RAD_BUF_PERSIST);
        const char* rr = std::getenv("RADIANCE_PP_LEAD_SEG_RINGS");
        if (m.pp_rank == 0 && rr && !std::strcmp(rr, "2"))
            m.b_segring2 = decl_b(b, nm.f("pp_segring2"), RAD_U8, {m.segring_bytes}, RAD_BUF_PERSIST);
    }
    auto seg_op = [&](size_t k, bool send) {
        std::vector<rad_buf> pb = { m.b_x };
        if (m.seg_parts[k] == 2) pb.push_back(m.a_h.x);
        rad_op h = RAD_OP(b, send ? "stage_send" : "stage_recv",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("ring", 2), RAD_INT("parts", (int64_t)pb.size()),
                                     RAD_INT("loop", 1)),
                          RAD_NOWEIGHTS);
        if (h == RAD_NULL_HANDLE) return h;
        if (send) rad_op_reads(b, h, pb.data(), (int)pb.size());
        else      rad_op_writes(b, h, pb.data(), (int)pb.size());
        return h;
    };
    m.seg_send.assign(m.seg_b.size(), 0);
    m.seg_recv.assign(m.seg_b.size(), 0);
    for (int64_t l = 0; l < m.n_main; ++l) {
        if (!m.held[(size_t)l]) continue;
        for (size_t k = 0; k < m.seg_b.size(); ++k)
            if (m.seg_b[k] == (int32_t)l) {
                if (!(m.seg_send[k] = seg_op(k, true)) || !(m.seg_recv[k] = seg_op(k, false))) return RAD_E_INVAL;
            }
        Layer& lay = m.layers[(size_t)l];
        /* Residual fold and collective placement: rad_fp8.h, layer_fold. */
        const LayerFold f = layer_fold(g, l, m.n_main, m.tap_layer, m.n_taps, m.op_embed_ar != 0);

        const char* pre = "model.layers";
        AttnFP8::Src as{};
        as.norm = nm.ckpt("%s.%lld.input_layernorm.weight",  pre, (long long)l);
        as.q    = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
        as.k    = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
        as.v    = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
        as.o    = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                      f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));

        MlpFP8::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, true, f.mlp_add,
                                     f.mlp_ar_in, f.mlp_ar_out));
    }

    /* tp8 S4: the crossings. X1 enters the follower's first layer h, X2 the leader's layer t;
     * each carries the stream and, where that layer's norm folds the previous block's delta in,
     * the delta (a_h.x) -- the layer is declared with the same fold as in one process, so it adds
     * the same two operands. Declared on both nodes in the same shape, so the rendezvous can
     * compare them. */
    if (m.pp) {
        auto parts_at = [&](int64_t first) {
            if (first <= 0 || first >= m.n_main) return 1;
            return layer_fold(g, first, m.n_main, m.tap_layer, m.n_taps, m.op_embed_ar != 0).mix_fold ? 2 : 1;
        };
        m.parts_x1 = parts_at(m.pp_h);
        m.parts_x2 = parts_at(m.pp_t);
        const char* kc = std::getenv("RADIANCE_LLAMA_PP_KVCOPY");
        m.kvcopy = kc && !std::strcmp(kc, "1") && m.pp_h > 0;
        const int64_t row = g.n_embd * (int64_t)rad_dtype_bytes(rad_dtype_parse(g.dtype), 1);
        const int64_t qkv_row = (g.q_dim() + 2 * g.kv_dim()) * (int64_t)rad_dtype_bytes(g.act_dtype, 1);
        if (m.x1_ids) {   /* tp8 dev D10 round 2: the ids alone, 4 B a row */
            m.parts_x1 = 1;
            rad_pp_declare_crossing(RAD_PP_X1, 4, 1);
            rad_pp_declare_x1_ids(1);
        } else {
        int64_t x1_row = row * m.parts_x1 + (m.kvcopy ? qkv_row : 0);
        int x1_parts = m.parts_x1 + (m.kvcopy ? 1 : 0);
        if (m.band) {   /* tp8 loop P28 stand-in: the band X1 (at h + 1) may be the wider one */
            m.parts_x1b = parts_at(m.pp_h + 1);
            x1_row = std::max(x1_row, row * m.parts_x1b + qkv_row);
            x1_parts = std::max(x1_parts, m.parts_x1b + 1);
            rad_pp_declare_band(1);
        }
        rad_pp_declare_crossing(RAD_PP_X1, x1_row, x1_parts);
        }
        if (m.tband) {   /* tp8 loop D39 stand-in: X2 at t - 1 may be the wider one */
            m.parts_x2b = parts_at(m.pp_t - 1);
            rad_pp_declare_crossing(RAD_PP_X2, row * std::max(m.parts_x2, m.parts_x2b), std::max(m.parts_x2, m.parts_x2b));
            rad_pp_declare_tail_band(1);
        } else {
            rad_pp_declare_crossing(RAD_PP_X2, row * m.parts_x2, m.parts_x2);
        }
        auto stage = [&](const char* op, int ring, int parts, bool send) {
            const bool kv = ring == RAD_PP_X1 && m.kvcopy;
            std::vector<rad_buf> pb = { m.b_x };
            if (parts == 2) pb.push_back(m.a_h.x);
            if (kv) pb.push_back(m.b_qkv);   /* tp8 S6 test: layer h - 1's q|k|v rows */
            const bool ids = ring == RAD_PP_X1 && m.x1_ids;   /* tp8 dev D10 round 2: the batch's ids, no buffer */
            if (ids) pb.clear();
            const int np = ids ? 1 : (int)pb.size();
            rad_op h = RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("ring", ring),
                                                RAD_INT("parts", np)),
                              RAD_NOWEIGHTS);
            if (h == RAD_NULL_HANDLE || ids) return h;
            if (send) rad_op_reads(b, h, pb.data(), np);
            else      rad_op_writes(b, h, pb.data(), np);
            return h;
        };
        const bool lead = m.pp_rank == 0;
        m.op_x1 = stage(lead ? "stage_send" : "stage_recv", RAD_PP_X1, m.parts_x1, lead);
        m.op_x2 = stage(lead ? "stage_recv" : "stage_send", RAD_PP_X2, m.parts_x2, !lead);
        if (m.tband) {   /* tp8 loop D39 stand-in: the tail band's X2 (the stream (+ the delta) at t - 1) */
            m.op_x2b = stage(lead ? "stage_recv" : "stage_send", RAD_PP_X2, m.parts_x2b, !lead);
            if (m.op_x2b == RAD_NULL_HANDLE) return RAD_E_INVAL;
            const char* ts = std::getenv("RADIANCE_PP_TEST_TBAND_SKIP");
            m.tband_skip = !lead && ts && !std::strcmp(ts, "1");
        }
        if (m.band) {   /* tp8 loop P28 stand-in: the band X1 = the stream (+ the delta) at h + 1 + layer h's q|k|v rows */
            std::vector<rad_buf> pb = { m.b_x };
            if (m.parts_x1b == 2) pb.push_back(m.a_h.x);
            pb.push_back(m.b_qkv);
            m.op_x1b = RAD_OP(b, lead ? "stage_send" : "stage_recv",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("ring", RAD_PP_X1), RAD_INT("parts", (int)pb.size())),
                              RAD_NOWEIGHTS);
            if (m.op_x1b == RAD_NULL_HANDLE) return RAD_E_INVAL;
            if (lead) rad_op_reads(b, m.op_x1b, pb.data(), (int)pb.size());
            else      rad_op_writes(b, m.op_x1b, pb.data(), (int)pb.size());
            if (!lead) {
                m.op_band_st = rw(b, RAD_OP(b, "kv_store",
                                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                                 RAD_INT("n_head_kv", g.n_head_kv),
                                                 RAD_INT("block_size", rad_kv_block_size(b, m.kv_full)),
                                                 RAD_STR("kv_dtype", g.kv_dtype)),
                                      RAD_NOWEIGHTS),
                                  {m.b_qkv}, {});
                if (m.op_band_st == RAD_NULL_HANDLE) return RAD_E_INVAL;
                const char* bs = std::getenv("RADIANCE_PP_TEST_BAND_SKIP");
                m.band_skip = bs && !std::strcmp(bs, "1");
            }
        }
        if (m.kvcopy) {
            /* the follower's copy of layer h - 1's cache; a group's bound layers are sorted at the end of declare
             * (rad_builder), so the copy is the follower's slot 0 and the leader's own its slot h - 1 */
            if (!lead) {
                RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)m.pp_h - 1, m.kv_full));
                m.op_kvcopy = rw(b, RAD_OP(b, "kv_store",
                                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                                RAD_INT("n_head_kv", g.n_head_kv),
                                                RAD_INT("block_size", rad_kv_block_size(b, m.kv_full)),
                                                RAD_STR("kv_dtype", g.kv_dtype)),
                                     RAD_NOWEIGHTS),
                                 {m.b_qkv}, {});
                if (m.op_kvcopy == RAD_NULL_HANDLE) return RAD_E_INVAL;
                const char* sk = std::getenv("RADIANCE_PP_TEST_KVCOPY_SKIP");
                m.kvcopy_skip = sk && *sk ? std::atoll(sk) : -1;
            }
            rad_pp_declare_kv_copy((int32_t)m.kv_full, lead ? m.pp_h - 1 : 0, m.pp_h - 1);
        }
        rad_note(b, "two-node pipeline (tp8 S4): the %s of W(%d,%d) holds layers %s; X1 carries %d and "
                    "X2 %d part(s) of [rows, %lld] %s", lead ? "leader" : "follower", m.pp_h, m.pp_t,
                 lead ? "outside the cut, the embedding and the head" : "inside the cut", m.parts_x1,
                 m.parts_x2, (long long)g.n_embd, g.dtype);
        if (m.fe && m.x1_ids)
            rad_note(b, "two-node pipeline (tp8 dev 1a + D10 round 2, RADIANCE_PP_FOLLOWER_EMBED=1 + RADIANCE_PP_ISSUE_AHEAD=1): "
                        "the follower looks the step's embedding up itself from the token ids X1 carries (the leader's card's), "
                        "which it takes first; the leader's head part issues no lookup, only X1");
        else if (m.fe)
            rad_note(b, "two-node pipeline (tp8 dev 1a, RADIANCE_PP_FOLLOWER_EMBED=1): the follower looks the step's "
                        "embedding up itself; X1 is declared and never issued, the leader's head part issues no lookup");
        if (!m.seg_b.empty()) {
            std::string bs;
            for (int32_t x : m.seg_b) bs += (bs.empty() ? "" : ", ") + std::to_string(x);
            if (lead)
                rad_note(b, "two-node pipeline (tp8 dev M9, RADIANCE_PP_LEAD_SEG): a BG lane's head / tail parts may run in "
                            "segments at layers %s, each boundary's stream saved and restored on this rank (%s %lld-byte ring "
                            "buffer%s)", bs.c_str(), m.b_segring2 ? "two" : "a", (long long)m.segring_bytes,
                         m.b_segring2 ? "s: RADIANCE_PP_LEAD_SEG_RINGS=2, tp8 loop D32" : "");
            else
            rad_note(b, "two-node pipeline (tp8 dev M4, RADIANCE_PP_SEG_LAYERS): the follower's passes may run in segments "
                        "at layers %s, each boundary's stream saved and restored on this rank (a %lld-byte ring buffer)",
                     bs.c_str(), (long long)m.segring_bytes);
            rad_pp_declare_segments(m.seg_b.data(), (int)m.seg_b.size());
        }
    }

    if (shell) {
    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.a_h.x});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});
    }

    /* The drafter is declared last, after `op_logits`. Do not move it.
     *
     * Declaration order is the buffer planner's liveness: a transient lives over [first_def,
     * last_use] in op indices, and buffers with disjoint ranges may share bytes
     * (core/build/rad_bufplan.cpp). The drafter's passes run after the whole trunk, so they must
     * be declared after it; declared earlier, the trunk's tail ops would look live after the
     * drafter, and trunk and drafter buffers live in one step would share bytes.
     *
     * The failure is silent: the trunk output stays correct (row 0 of a verify step is computed
     * before the clobber) but every draft is rejected. The op oracle cannot see it, since device
     * and host compute the same wrong answer from a clobbered input. It can also depend on world
     * size, because sharding changes buffer sizes and so the packer's offsets. */
    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dspark(b, meta, ctx, m, (int)m.n_main));
    if (m.pp_draft_zero) {   /* tp8 loop D38a: the follower's zeroed copy of the leader's drafter KV group */
        Geom dg{};
        RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DSpark"));
        const int64_t win = rad_meta_geti(meta, "draft.use_sliding_window", 0) ? rad_meta_geti(meta, "draft.sliding_window", 0) : 0;
        RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, win, (int)m.n_main, &m.kv_draft));
        /* its block size off a store declared right after it, never issued (deepseek41's window_key_store) */
        const int64_t kvd = dg.n_head_kv * dg.head_dim;
        const rad_buf kz = decl_b(b, nm.f("pp_draft_kz"), dg.act_dtype, {g.max_tok, kvd});
        const rad_op st = rw(b, RAD_OP(b, "kv_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", dg.head_dim),
                                           RAD_INT("n_head_kv", dg.n_head_kv), RAD_INT("block_size", RAD_KV_BLOCK),
                                           RAD_STR("kv_dtype", dg.kv_dtype)),
                                RAD_NOWEIGHTS),
                             {kz, kz}, {});
        if (!st || rad_kv_block_size(b, m.kv_draft) < 1 || rad_kv_group_zero(b, m.kv_draft) < 0) return RAD_E_INVAL;
    }

    rad_note(b, "rank %d/%d: %lld layers, all full attention; n_head=%lld n_head_kv=%lld "
                "head_dim=%lld rot=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)m.n_rot, (long long)g.n_ff,
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
#if LLAMA_DENSE_SRC
    rad_note(b, "a dense bf16 checkpoint, every linear quantised by the recipe to E4M3 with a "
                "bf16 [N/%d,K/%d] scale plane (scale = block amax / 448, rounded to bf16)",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
#else
    rad_note(b, "weights are block-scaled fp8: E4M3 [N,K] with a bf16 [N/%d,K/%d] scale plane, "
                "one stored weight taken by gemm_nt_q as two operands",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
#endif
    rad_note(b, "bf16 in the checkpoint: every norm, the embedding and the lm_head, which is "
                "declared at what the container holds");
    if (m.n_taps > 0) {
        char ids[64] = {};
        int  off = 0;
        for (int t = 0; t < m.n_taps && off < (int)sizeof(ids) - 8; ++t)
            off += snprintf(ids + off, sizeof(ids) - (size_t)off, t ? " %d" : "%d",
                            m.tap_layer[t]);
        rad_note(b, "the drafter reads the residual stream after layer(s) %s into a [%lld, %lld] "
                    "buffer; those layers keep their own residual add",
                 ids, (long long)g.max_tok, (long long)((int64_t)m.n_taps * g.n_embd));
    }
    rad_note(b, "q, k and v are one fused [%lld, %lld] projection, rotated in place by `rope` "
                "(q heads, then k heads); no per-head q/k norm and no output gate",
             (long long)(q_dim + 2 * kv_dim), (long long)g.n_embd);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    /* A drafter pass; the sign gives the kind. Positive is the query pass, proposing the whole
     * block at once; negative is the context pass, turning the tokens the trunk just committed
     * into the drafter's own K and V. Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_ds) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` block, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.ds.step(c, batch);
        else                       m.ds.context(c, batch);
        return;
    }

    const int64_t T = batch->n_tok;
    /* tp8 S5: the pass's lane (its rings) and, on the leader, its part: a lane step's leader pass is
     * issued as its head (up to and including X1's send) and its tail (from X2's receive), with
     * other lanes' passes between them (abi/rad_pp.h). One lane, the follower: the whole pass. */
    RadPpInfo ppi{};
    if (m.pp) ppi = rad_pp_query();
    const int part = m.pp && m.pp_rank == 0 ? ppi.part : RAD_PP_PART_WHOLE;
    const bool run_head = part != RAD_PP_PART_TAIL, run_tail = part != RAD_PP_PART_HEAD;
    /* tp8 loop P28 stand-in: a band pass crosses at h + 1 */
    const int bnd = m.band && ppi.reserved[0] == m.band ? m.band : 0;
    const int64_t he = m.pp_h + bnd;
    /* tp8 loop D39 stand-in: a tail band pass crosses at t - 1 (the follower's parts end there) */
    const int tbd = m.tband && ppi.reserved[2] == m.tband ? m.tband : 0;
    const int64_t te = m.pp_t - tbd;
    /* tp8 dev M4: a follower segment issues layers [s0, s1) of its [h, t): X1 (or the lookup) only in the first, X2 only in
     * the last, a boundary's restore / save at its ends */
    const bool seg = m.pp && m.pp_rank == 1 && ppi.part == RAD_PP_PART_SEG && !m.seg_b.empty();
    const int64_t s0 = seg ? ppi.seg_l0 : he, s1 = seg ? ppi.seg_l1 : te;
    const bool seg_first = !seg || s0 == he;
    /* tp8 dev M9: a leader segment [l0, l1) of its head / tail part: the lookup only in the head's first, X1 only in its
     * last, X2 only in the tail's first, the final norm and the head only in its last (RAD_PP_SEG_END), a boundary's restore
     * / save at its ends */
    const bool lseg = m.pp && m.pp_rank == 0 && (part == RAD_PP_PART_HEAD || part == RAD_PP_PART_TAIL) && ppi.seg_l1 > 0 &&
                      !m.seg_b.empty();
    const int64_t l0 = lseg ? ppi.seg_l0 : 0, l1 = lseg ? std::min<int64_t>(ppi.seg_l1, m.n_main) : m.n_main;
    /* tp8 loop D40 stand-in: the engine's SP word (RadPpInfo.reserved[1], fn4 pp_lead_seg + 512) comes only with a leader HEAD
     * segment (deepseek41 plans the whole head's SP there); this model has no SP and only checks where it lands */
    if (m.pp && ppi.reserved[1] != 0 && !(lseg && part == RAD_PP_PART_HEAD)) {
        rad_step_fail(c, "llama: the engine's SP word (tp8 loop D40) on a pass that is not a leader head segment");
        return;
    }
    const bool lfirst = !lseg || l0 == (part == RAD_PP_PART_HEAD ? 0 : m.pp_t);
    /* tp8 loop D36: a band pass's head ends at h + 1 (its last segment [s, h + 1), RADIANCE_PP_LEAD_SEG_BAND=1) */
    const bool llast = !lseg || (part == RAD_PP_PART_HEAD ? l1 == he : l1 >= m.n_main);
    /* tp8 loop D32: the leader's ring of the pass's lane (lane % 2 with two rings; the engine pairs them the same way) */
    const rad_buf sring = m.b_segring2 && (ppi.lane & 1) ? m.b_segring2 : m.b_segring;
    auto seg_issue = [&](int64_t at, bool send) {
        for (size_t k = 0; k < m.seg_b.size(); ++k) {
            if (m.seg_b[k] != (int32_t)at) continue;
            if (m.seg_parts[k] == 2)
                RAD_ISSUE(c, send ? m.seg_send[k] : m.seg_recv[k], brows(sring, m.segring_bytes), brows(m.b_x, T),
                          brows(m.a_h.x, T));
            else
                RAD_ISSUE(c, send ? m.seg_send[k] : m.seg_recv[k], brows(sring, m.segring_bytes), brows(m.b_x, T));
            return true;
        }
        rad_step_fail(c, "llama: a segment boundary the follower did not declare");
        return false;
    };
    /* tp8 S4: a crossing's issue: the core's ring, then the stream and, folded, the delta. The ring
     * is the pass's lane's (tp8 S5), read at every issue. */
    auto cross = [&](rad_op op, int ring, int parts) {
        RadPpRing* rg = ppi.ring[ring];
        if (!rg) {
            rad_step_fail(c, "two-node pipeline: the core set up no ring for this crossing");
            return;
        }
        const bool kv = ring == RAD_PP_X1 && m.kvcopy;   /* tp8 S6 test: + layer h - 1's q|k|v rows */
        if (ring == RAD_PP_X1 && bnd) {   /* tp8 loop P28 stand-in: the band X1 */
            if (m.parts_x1b == 2)
                RAD_ISSUE(c, m.op_x1b, praw(rg, RAD_U8, 1), brows(m.b_x, T), brows(m.a_h.x, T), brows(m.b_qkv, T));
            else
                RAD_ISSUE(c, m.op_x1b, praw(rg, RAD_U8, 1), brows(m.b_x, T), brows(m.b_qkv, T));
            return;
        }
        if (ring == RAD_PP_X1 && m.x1_ids)   /* tp8 dev D10 round 2: the step's token ids alone */
            RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), praw(batch->token_ids, RAD_I32, batch->n_tok));
        else if (parts == 2 && kv)
            RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), brows(m.b_x, T), brows(m.a_h.x, T), brows(m.b_qkv, T));
        else if (parts == 2)
            RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), brows(m.b_x, T), brows(m.a_h.x, T));
        else if (kv)
            RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), brows(m.b_x, T), brows(m.b_qkv, T));
        else
            RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), brows(m.b_x, T));
    };
    /* tp8 dev D10 round 2: the follower takes X1 (the step's token ids) into its batch before its lookup reads them */
    if (m.x1_ids && m.pp_rank == 1 && seg_first) cross(m.op_x1, RAD_PP_X1, m.parts_x1);
    /* tp8 dev 1a: under m.fe the follower's lookup, the leader's none */
    if (((!m.pp || m.pp_rank == 0) && run_head && !m.fe && lfirst) || (m.fe && m.pp_rank == 1 && seg_first)) {
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar && !ar_wire_is_exact(m.g, batch->n_tok))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    }
    /* The follower starts at its first layer from the leader's X1 (tp8 S6 test: then stores layer h - 1's k / v
     * into its copy, as the leader's layer stored them, at the same slots). */
    if (m.pp && m.pp_rank == 1 && seg_first) {
        if (!m.fe) cross(m.op_x1, RAD_PP_X1, m.parts_x1);
        if (bnd && !m.band_skip) {   /* tp8 loop P28 stand-in: layer h's k / v into this node's own cache, the owner's op at the batch's slots */
            const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
            const int64_t q_dim = m.g.q_dim(), kv_dim = m.g.kv_dim();
            RAD_ISSUE(c, m.op_band_st, bcol(m.b_qkv, q_dim, kv_dim, T), bcol(m.b_qkv, q_dim + kv_dim, kv_dim, T),
                      praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(m.kv_full, (int)m.pp_h));
        }
        if (m.op_kvcopy && m.kvcopy_pass++ != m.kvcopy_skip) {
            const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
            const int64_t q_dim = m.g.q_dim(), kv_dim = m.g.kv_dim();
            RAD_ISSUE(c, m.op_kvcopy, bcol(m.b_qkv, q_dim, kv_dim, T), bcol(m.b_qkv, q_dim + kv_dim, kv_dim, T),
                      praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(m.kv_full, (int)m.pp_h - 1));
        }
    }

    for (size_t li = 0; li < m.layers.size(); ++li) {
        if (m.pp) {
            /* The leader hands over before layer h and takes the follower's result back at t; the
             * follower's last layer hands its result back. */
            if (m.pp_rank == 0 && (int64_t)li == he) {
                if (run_head && (!m.fe || m.x1_ids)) cross(m.op_x1, RAD_PP_X1, m.parts_x1);
                if (!run_tail) return;   /* tp8 S5: the head part ends with X1's send */
                if (lfirst) {   /* tp8 dev M9: a later tail segment: its restore; tp8 loop D39: a tail band's X2 at t - 1 */
                    if (tbd) cross(m.op_x2b, RAD_PP_X2, m.parts_x2b);
                    else cross(m.op_x2, RAD_PP_X2, m.parts_x2);
                }
            }
            if (!m.held[li]) continue;
            /* tp8 loop P28 stand-in: the leader's band layer runs only in a band pass's head */
            if (m.band && m.pp_rank == 0 && (int64_t)li >= m.pp_h && (int64_t)li < m.pp_h + m.band && !(bnd && run_head)) continue;
            /* ... and a band pass's follower starts at h + 1 */
            if (bnd && m.pp_rank == 1 && (int64_t)li < he) continue;
            /* tp8 loop D39 stand-in: the leader's layer t - 1 runs only in a tail band pass's tail; that pass's follower ends at t - 1 */
            if (m.tband && m.pp_rank == 0 && (int64_t)li == m.pp_t - 1 && !(tbd && run_tail)) continue;
            if (tbd && m.pp_rank == 1 && (int64_t)li >= te) continue;
            /* tp8 S5: the tail part starts at X2; the head part ran the layers before h */
            if (m.pp_rank == 0 && !run_head && (int64_t)li < m.pp_h) continue;
            /* tp8 dev M4: a segment's layers only; its first from the boundary's restore */
            if (seg && ((int64_t)li < s0 || (int64_t)li >= s1)) continue;
            if (seg && (int64_t)li == s0 && !seg_first && !seg_issue(s0, false)) return;
            /* tp8 dev M9: the same for a leader segment */
            if (lseg && ((int64_t)li < l0 || (int64_t)li >= l1)) continue;
            if (lseg && (int64_t)li == l0 && !lfirst && !seg_issue(l0, false)) return;
        }
        m.layers[li].attn.step(c, batch);
        m.layers[li].mlp.step(c, batch);
        /* tp8 dev M4: a segment that ends before t saves the boundary's live set and ends */
        if (seg && (int64_t)li == s1 - 1 && s1 < te) {
            seg_issue(s1, true);
            return;
        }
        /* tp8 dev M9: a leader segment that ends at a boundary (not X1's, not the head's end) saves and ends */
        if (lseg && (int64_t)li == l1 - 1 && !llast) {
            seg_issue(l1, true);
            return;
        }
        /* The tap, into its column slice. Runs on prefill and decode alike: the drafter
         * conditions on every committed token's residual stream. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    if (m.pp && m.pp_rank == 1) {
        if (!tbd) {
            cross(m.op_x2, RAD_PP_X2, m.parts_x2);
            return;
        }
        /* tp8 loop D39 stand-in: X2 at t - 1, then layer t - 1's k / v from this node's own rows into its own cache (the
         * owner's ops, as the whole layer issues them first) */
        cross(m.op_x2b, RAD_PP_X2, m.parts_x2b);
        if (!m.tband_skip) m.layers[(size_t)te].attn.step_kv_only(c, batch);
        return;
    }
    /* A cut at the last layer: the leader's X2 arrives after the loop. */
    if (m.pp && m.pp_rank == 0 && m.pp_h >= (int)m.layers.size()) {
        if (run_head) cross(m.op_x1, RAD_PP_X1, m.parts_x1);
        if (!run_tail) return;
        cross(m.op_x2, RAD_PP_X2, m.parts_x2);
    }
    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.a_h.x, batch->n_tok));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The speculation depth this container supports: zero for a plain llama container (no MTP head
 * in this family), or the trained block length of a DSpark drafter added with `--draft-model`.
 *
 * `draft_depth` is `block_size`, not `block_size - 1` as in arch/qwen35_fp8: DSpark's anchor row
 * predicts too. `draft_depth_fixed` forbids any other depth, since the drafter was trained for
 * exactly one block length. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
        const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
        if (bs > 1) {
            out->draft_depth       = (int)bs;
            out->draft_depth_fixed = 1;
        }
    }
    return RAD_OK;
}

}  /* namespace LLAMA_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* One more level of expansion so `#NS` sees the expanded namespace. */
#define LLAMA_PROBE_X(NS)        RAD_ARCH_PROBE(NS)
#define LLAMA_PLUGIN_X(NS, ...)  RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
LLAMA_PROBE_X(LLAMA_NS)
#if LLAMA_DENSE_SRC
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama) from a dense bf16 checkpoint, made block-scaled fp8 by a recipe "
               "at convert time and served fp8 weights / fp8 activations")
#else
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama), block-scaled fp8 weights and fp8 activations")
#endif
#endif
