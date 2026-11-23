/* qwen4exp_fp8 -- Qwen4-Exp (`model_type: qwen4_exp`), the architecture of Qwen3.8-Flash-Next.
 * The checkpoint is dense bf16, quantised ahead of time by a recipe
 * (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe is the served one): int8 or fp8 trunk linears with
 * matching activations, four-bit routed experts, or bf16 throughout.
 *
 * ============================== RELATION TO QWEN3.5 MOE ==============================
 *
 * In transformers it is `Qwen4ExpTextModel`, a subclass of `Qwen3_5MoeTextModel`, which this tree
 * already serves (arch/qwen35moe_fp8). Most of this file composes the same blocks:
 *
 *   reused                     the Gated DeltaNet (conv width 4, key/value head dim 128), gated
 *                              full attention (`attn_output_gate`: q_proj twice as wide, half of
 *                              it a sigmoid gate on the output), partial rotary (64 of 256 dims),
 *                              the routed feed-forward with a gated shared expert, the MTP
 *                              position.
 *   new                        the gated residual (arch/common/rad_block_hc.h), the n-gram / PLE
 *                              layer (arch/common/rad_block_ple.h), QSA's indexer
 *                              (arch/common/rad_qsa.h), stacked dense expert tensors
 *                              (tools/rad_convert.cpp, slice_stacked), gqa 12, and the delta net's
 *                              sigmoid output gate.
 *
 * Qwen3.8-Flash-Next: 48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE), hidden
 * 2560, hc_count 4 so the residual stream is 10240, 512 experts of 640 at top-10 plus a shared 640,
 * 24 query heads and 2 KV heads at head_dim 256, vocab 248320.
 *
 * ============================== THE RESIDUAL STREAM IS FOUR TIMES AS WIDE ==============================
 *
 * The checkpoint has no `input_layernorm`, `post_attention_layernorm` or final `norm`. Instead
 * every block has a gated residual connection: a grouped norm over the 10240-wide stream, a
 * low-rank data-dependent read gate that mixes the four sub-streams down to 2560, and four
 * data-dependent scalars that add the block's output back into all four. There are 96, plus a 97th
 * with no write half that collapses the stream for the lm_head.
 *
 * arch/common/rad_block_hc.h holds the arithmetic and docs/OPS.md's `hc_read` / `hc_write` rows
 * the schema. The blocks do not own their norms: this plugin passes `Src::norm = nullptr`, the fp8
 * blocks' "input already normed and quantised" mode, which forgoes folding the residual add and
 * the all-reduce into the next norm. The all-reduce rides in the gated residual's write instead,
 * where a kernel serves that (HyperConn::Config::reduce, see declare()).
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * The checkpoint is bf16 throughout (no `quantization_config`), so the quantisation descriptor is
 * empty and the served formats come from a recipe (data/recipes/qwen4exp-*.recipe). The plugin
 * declares each weight at its encoding, and blocks that read several formats choose their ops by
 * it:
 *
 *   block fp8 or int8  self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv, in_proj_z, out_proj},
 *                      the shared expert's three (LinearFP8); the indexer's qk projection
 *                      unless the recipe leaves it bf16, as the served one does
 *   fp8, four-bit      every routed expert's gate_up and down, each layer and projection in its
 *   or bf16            own form (MoeFP8::probe)
 *   E4M3 rows or bf16  each connection's input_mix_weight_{down,up}, whose lowrank of 320 is 2.5
 *                      scale blocks and so is scaled a row group at a time (HyperConn), and the
 *                      MTP head's fc_hidden / fc_embedding
 *   fp8, int8, bf16    the lm_head (whichever logits_gemm kernel reads it)
 *   BF16               every norm gain, A_log, dt_bias, conv1d, in_proj_a / in_proj_b (48 rows is
 *                      not a whole scale block), the router, shared_expert_gate, embed_tokens,
 *                      the PLE projections and inject weights
 *
 * ============================== HOW TENSOR PARALLEL SPLITS AN EXPERT ==============================
 *
 * `moe_intermediate_size` is 640 = 5 x 128. An even tp2 share, 320, is not a whole number of
 * 128-column blocks, on which the fp8 scale block, the 4-bit scale group and the rotation sit. So
 * each rank takes three blocks of the experts of one parity and two of the other's; the router is
 * replicated so both ranks route alike, each rank computes its slice of every routed expert, and
 * the block's existing end-of-block all-reduce sums the partials, with no all-to-all. See the
 * section in declare().
 *
 * ============================== PLE AND QSA ==============================
 *
 * PLE, the n-gram embedding, is 51B of the model's 180B parameters: a 320,001,536-row table of
 * 160-dim embeddings, gathered 16 rows per token by a hash of the last three token ids, read once
 * at layer 1 and injected into the wide stream.
 *
 * QSA's indexer picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens
 * per query and attends only to those. Below `indexer_budget + indexer_compress_ratio - 1` = 2051
 * tokens of context every complete block is selected, so QSA equals dense causal attention bit for
 * bit; the engine issues the cheaper dense attention there, and it serves as the oracle for QSA.
 * Above it the selection is per query on both prefill and decode (docs/QSA.md,
 * arch/common/rad_qsa.h).
 *
 * The vision tower is declared through arch/common/rad_block_vit.h, only when the container
 * carries one and the deployment asked for media; rad-convert writes only declared weights.
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>   /* getpid: fn4 BI's tracer files */

#include <arch/rad_arch.h>
#include <rad_prefill_split.h>
#include <arch/rad_block_attn_gated_fp8.h>
#include <arch/rad_block_gdn_fp8.h>
#include <arch/rad_block_hc.h>
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_block_mtp_hc.h>
#include <arch/rad_block_ple.h>
#include <arch/rad_qsa.h>
#include <arch/rad_block_vit.h>

#include "qfn_layers.h"   /* Qwen tp8 S1: a node's trunk-layer ranges (RADIANCE_FN4_LAYERS / _MODE) */
#include "rad_pp.h"       /* Qwen tp8 S2: the two-node pipeline (rad_pp_info, the stage crossings) */

#include <chrono>
#include <thread>

namespace qwen4exp_fp8 {

/* fn4 (9.11.2026): an environment switch set and not 0. */
static bool fn4_env_on(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && *v != '0';
}

using namespace rad::arch;

/* fn4 SP-serial (14.11.2026, Qwen's newcomer step): RADIANCE_FN4_SPSERIAL set at start declares the sequence-parallel gated
 * residual for a serial step's prefill rows (Model::spser_ok; the start value is the switch's), switch "sp_serial" picks it a
 * step: 1 one run over the step, 2 two halves whose collectives run on the second lane beside the other half's write and read
 * (falls back to 1 where the halves do not shard). Unset: nothing declared, today's ops and bytes. */
static bool fn4_spser_env() { return fn4_env_on("RADIANCE_FN4_SPSERIAL"); }
static std::atomic<int>& fn4_spser_flag() {
    static std::atomic<int> f{[] { const char* e = std::getenv("RADIANCE_FN4_SPSERIAL");
                                   const int v = e && *e ? std::atoi(e) : 0; return v >= 2 ? 2 : v > 0 ? 1 : 0; }()};
    return f;
}

struct Layer {
    int          full = 0;
    HyperConn    hc_mix;     /* in front of the attention or the delta net */
    HyperConn    hc_ffn;     /* in front of the routed feed-forward */
    AttnGatedFP8 attn;
    QsaIndexer   qsa;        /* full-attention layers only; inert when the checkpoint has none */
    GdnFP8       gdn;
    MoeFP8       mlp;
    /* The control vector's projection, applied to the wide stream right after this layer's
     * feed-forward write -- where llama.cpp's `build_cvec(res_hc, il)` sits in the `qwen4exp`
     * graph. Inert (no op) when the deployment has no vector or this layer is outside its range
     * (arch/common/rad_block_hc.h). */
    HcProject    proj;
};

/* fn4 BI's engine tracer (RADIANCE_FN4_BITRACE=<dir>, 11.11.2026; the ds41-bi tracer's form, radiance-bi 060dff0): a 64-bit
 * fingerprint of every traced row of every traced buffer after the op that wrote it (libr4d bi_rowhash), copied to the host
 * at the step's end and written, for the rows at positions below RADIANCE_FN4_BITRACE_PMAX (default 256) of the sequences it
 * follows (a request first seen at position 0 with a query of at most RADIANCE_FN4_BITRACE_PLEN rows, default 128: the
 * greedy set's prompts, not mb's streams nor a long prefill), to <dir>/bt-r<rank>-<pid>.bin (names in .names). A run alone
 * and one under load, matched by a request's (position, token) rows, name the first op whose output differs (bitrace.py).
 * Unset: nothing declared, nothing issued -- today's graph and passes. Set: the passes are not recorded (ctx.cpp), every
 * traced block-internal buffer is zeroed over the step's rows at its start (a row a step's path does not write then hashes
 * alike in both runs), and every step synchronises once at its end. */
struct BiTr {
    bool on = false;
    std::string dir;
    int64_t pmax = 256, plen = 128, pmin = 0;   /* pmin: RADIANCE_FN4_BITRACE_PMIN (15.11.2026), rows at positions >= it */
    rad_buf plane = 0;              /* [max_tok, 2 * cap] i32: row r's point i at columns 2i, 2i + 1 */
    int64_t cap = 0;                /* points a step */
    struct Tb {
        rad_op  op = 0;             /* bi_rowhash over this buffer */
        int64_t pitch = 0;          /* elements a row */
        int64_t esz = 0;            /* bytes an element */
        bool    clear = false;      /* zeroed over the step's rows at its start */
    };
    std::map<rad_buf, Tb> tb;
    int n = 0;                      /* points this step */
    std::map<std::string, int> at;  /* this step's point by name: a point issued a piece at a time keeps one */
    std::vector<std::string> names;
    std::vector<uint8_t> kind;      /* 0: the step's token rows; 1: its output rows (out_ids) */
    /* RadBatch::seq_ids are StepSlot indices (core/sched/batch.cpp: e.card), reused by the next request on the slot: a
     * request is told apart by its context length (a fresh one: first seen, a prefill whose context does not continue the
     * slot's last step, or a context that shrank) and written under a serial of its own. */
    struct Card { uint32_t serial = 0; bool follow = false; int64_t ctx = -1, end = -1; };
    std::map<int32_t, Card> card;
    uint32_t serials = 0;
    uint64_t names_key = 0;
    FILE* f = nullptr;
    FILE* fn = nullptr;
    std::vector<uint32_t> host;
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnFP8::Config gcfg{};
    MoeFP8::Config moecfg{};
    /* The trunk's linears are int8 (LinearFP8::i8), as read from the first delta-net layer's input
     * projection before the activation buffers are declared. */
    bool           trunk_i8 = false;
    HyperConn::Config hccfg{};
    QsaIndexer::Config qsacfg{};
    QsaIndexer::Wire   qsaw{};
    rad_kvgroup        kv_qsa_bkey = 0;   /* created by the first QSA layer's declare() */

    int64_t n_rot = 0;

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;
    /* PLE lives on exactly one layer; -1 when the checkpoint has none or this build does not
     * declare that layer. */
    PleLayer    ple;
    int64_t     ple_layer = -1;
    /* Qwen tp8 S5 prerequisite (round 31): the checkpoint's PLE layer even where this node does not hold it (a pipeline
     * follower declares that layer's two state groups as the leader does, pp_ple_groups) */
    int64_t     ple_layer_model = -1;

    /* The wide stream and the narrow block input. `b_h` is the residual stream, hc * n_embd per
     * token; `a_x` is what a connection's read produces and where its block writes the delta. One
     * of each for the whole model: a read is immediately followed by its block and its write, so
     * nothing in `a_x` or `b_inj` outlives the write that consumes it. Declaration order determines
     * the planner's liveness (see the note at declare()), so this holds by this file's order. */
    rad_buf b_h = 0;
    ActFP8  a_x{};
    rad_buf b_inj = 0;
    /* fn4 W1b (RADIANCE_FN4_HC_SS=1): the trunk connections' norm-scale handoff, [T, 8] f32 (HyperConn::Wire::ss).
     * Declared only on request, so the default plan is unchanged; with it, buffers under its 128 KiB (at T 4096) may
     * land elsewhere in the arena (the packer places largest first), so A/B it in process (hc_readv bit 3). */
    rad_buf b_hss = 0;

    /* the attention block's own activations, shared across the 12 attention layers */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_rope_cs = 0, b_gate = 0;
    ActFP8  a_attn{};

    /* the delta net's */
    ActFP8  a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0, b_gdec = 0, b_beta = 0, b_kkt = 0;

    /* the routed feed-forward's */
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_einv = 0;   /* fn4 F2a: the sort's inverse slot table (RADIANCE_FN4_GATHER_INV) */
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0, b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_eff{}, a_sff{};
    /* The rotated codes of the block input, for the routed arm only: `q` and `s` are used and `x`
     * stays zero, since the bf16 source is `a_x`, unrotated. Declared only when the container asks
     * for a rotation; zero handles tell the MoE block the pair is absent. */
    ActFP8  a_hr{};

    rad_buf b_hout = 0, b_logits = 0;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h). Declared only when
     * the deployment speculates, since its ~2.5 GiB of weights would otherwise be loaded and never
     * issued. rad-convert declares at a non-zero max_spec so the container carries everything the
     * model can serve. */
    bool        have_mtp = false;
    MtpHcBlock  mtp{};
    rad_buf     b_mtp_hin = 0, b_mtp_hw = 0, b_mtp_e = 0, b_mtp_hout = 0;
    rad_buf     b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf     b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;
    /* --draft-coupling: [max_seqs] draws staged before the serial passes and [max_spec,
     * max_seqs] states read back after them. PERSIST: the engine writes them from the host and the
     * head writes the states on the card, across a step boundary. */
    rad_buf     b_mtp_couple_draws = 0, b_mtp_couple_state = 0;
    rad_buf     b_mtp_dpair = 0, b_mtp_dgath = 0;

    HcEnter   enter;
    HyperConn mixer;            /* the 97th connection: no write half */
    /* Whether this rank has said that it pipelines a prefill step (step_pipelined), once. */
    bool      pipe_noted = false;
    bool      nowait_noted = false;   /* PF probe NOWAIT (sp_route bit 256) said once */
    bool      psp_noted = false;      /* lever PSP (sp_route bit 512) said once */
    bool      gpf_noted = false;      /* lever GPF (sp_route bit 4096) said once */
    int       pair_noted = 0;   /* fn4 P2x: the mode last noted (taken), -1 refused noted; P2S: 1000 (mode + 1) + layers paired */
    /* Whether the deployment spans (RadBuildCtx::max_mixed_tok below a max_tok wider than a MoE
     * pass): a step wider than a pass is then one sequence, the connections read it a pass at a
     * time and the indexer scores it in windows. `pipe_on` adds ranks whose blocks issue their own
     * all-reduce at such a step (declare): it is pipelined (step_pipelined) and its all-reduces
     * are declared at `overlap` 1. Without a span neither is set and none of these paths is
     * declared or taken. A sizing declare runs at a max_tok no wider than a pass and takes both
     * from the real declare, so it declares the same ops. */
    bool      span_on = false;
    bool      pipe_on = false;
    /* The pass a span cuts a wide step into (the pipelined step's pieces, the connections' read width, the indexer's
     * window): the MoE's rows, except under fn4 P2x (RADIANCE_FN4_P2X=1 at start), where the MoE is declared two
     * pieces tall (moecfg.rows = 2 pass_rows) so a pipelined step's pieces can share one routed pass (switch moe_pair). */
    int64_t   pass_rows = 0;
    /* fn4 SP-hc (RADIANCE_FN4_SPHC=1 at start declares; switch hc_sp picks it a pipelined prefill step): the trunk's
     * all-reduces as reduce-scatter (sp_rs) + all-gather of the next block input (sp_ag), the connections' write and
     * read on this rank's quarter of each piece between them, the codes redone on every row (HyperConn::codes), and the
     * wide stream gathered raw after the last layer (sp_agh). */
    rad_op    sp_rs = 0, sp_ag = 0, sp_agh = 0;
    /* fn4 SP-route (16.11.2026, qwen/prefill; RADIANCE_FN4_SP_ROUTE declares, switch sp_route): the router logits' all-gather
     * of an SP-hc piece (raw rows of n_expert_all), declared beside SP-hc */
    rad_op    sp_agl = 0;
    bool      sp_on = false;
    /* fn4 KVCAP C1 (15.11.2026, qwen/KV-BOUNDS.md): the most rows one sp_agh launch carries, its declared bound. Both issue
     * sites carry at most a pass or an SP-serial segment (a pipelined piece <= pass_rows; an SP-serial run's T <=
     * spser_rows <= 128 kSpTiles), so the bound is min(T, max(pass_rows, 128 kSpTiles)) rows of the wide stream, not the
     * whole step's T: libr4d sizes the gather's peer scratch (8 slots + locpack, hipMalloc outside the VRAM budget) by the
     * declared numel bound, 1,370 MiB a card at T = 8192. Every issue checks it (sp_agh_fits) and fails the step loudly
     * past it. */
    int64_t   sp_agh_rows = 0;
    /* fn4 SP-serial (RADIANCE_FN4_SPSERIAL at start): sp_rs / sp_ag / sp_agh, every trunk connection's codes op and its
     * K-pinned part reads (HyperConn::op_read_kt) for segments up to spser_rows rows; spser_noted: said once. */
    bool      spser_ok = false;
    int64_t   spser_rows = 0;
    int       spser_noted = 0;
    /* qfn-on-ds lever LSK (Qwen loop round 27, QUEUE Q51; RADIANCE_FN4_SPSER_LANE=1 at start): SP-serial's OWN three
     * collectives, sp_rs / sp_ag / sp_agh's twins with lane_kind 1 -- libr4d launches them with the lane form's second pair
     * (switches ar_laneblocks_s / ar_lanethreads_s), so MIXED / serial steps and lone pipelined prefill can run different
     * lane forms (rows lane5 / lane6: 12 x 256 -0.40..-0.49 % a cold prompt; lanm1: the MIXED step +0.8..+1.4 % at it).
     * The same params otherwise, so libr4d serves them from the same instances (no scratch). 0 = step_serial_sp takes
     * the shared ops (today). */
    rad_op    ss_rs = 0, ss_ag = 0, ss_agh = 0;
    /* {0, pass, T} for a step wider than a pass (pipe_bounds): the windows of the indexer's score
     * plane and the delta net's pieces. A transient (declare), uploaded by every step that reads
     * it -- the first two from the pinned host copy `pipe_cu_h`, made once. */
    rad_buf   b_pipe_cu = 0;
    int32_t*  pipe_cu_h = nullptr;
    bool      pipe_cu_tried = false;

    std::vector<Layer> layers;

    rad_weight w_tok = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_rope_cs = 0, op_gather = 0, op_logits = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows over
     * the token embeddings, declared only when the container carries a tower and the deployment
     * asked for media (RadBuildCtx::max_enc_patches). */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    BiTr        bt{};   /* fn4 BI's engine tracer: inert unless RADIANCE_FN4_BITRACE */
    /* fn4 BI-dual (RADIANCE_FN4_X3DUAL=1 at start, switch exl3_dual): the step's row bands (libr4d bi_band from
     * cu_seqlens: 1 for a row whose sequence has at most 16 rows in the step) for every trunk MoE layer's EXL3 down
     * (exl3 interface v12), so a mixed step's decode rows get the decode band's bytes in the prefill band's pass.
     * PERSIST: the downs read it all step, and their declarations do not name it. */
    rad_buf     b_band = 0;
    rad_op      op_band = 0;
    /* fn4 BI (switch hc_bi, RADIANCE_FN4_HC_BI): this step's read segments -- [0, decode rows), then each prefill
     * sequence's rows (host positions: a jump starts the next), as bounds {0, .., T} -- so libr4d's hc_read takes every
     * decode-class run (at most 64 rows) in a read of its own, with its one-family form (r4d_hc_bf16.hip hc_bi). */
    bool        bi_on = false;
    int         bi_mode = 0;     /* fn4 hc_bi's value this step (3: decode rows read <= kBiDecRows at a time; 5: as 3,
                                  * a connection with the bi_dec twin reads them in one launch) */
    int64_t     bi_td = 0;       /* this step's decode rows */
    int64_t     bi_qd = 1;       /* rows a decode sequence (1 + n_spec) */
    std::vector<int64_t> bi_cut;
    /* qfn-on-ds lever HCP (Qwen loop round 23, sp_route bit 8192; bi_segments sets it a step): a prefill segment's
     * connection reads past the prompt's first interval at the K split of a FULL piece (HyperConn::op_read_sp), whatever
     * the segment's rows -- hcp_pos0[i] the first position of segment [bi_cut[i], bi_cut[i + 1]) (-1: a decode segment or
     * no host positions); hcp_pos_step a pure prefill step's first position (bi_cut empty: one read). hcp_n: reads pinned
     * (rank 0's engine line). */
    bool        hcp = false;
    std::vector<int64_t> hcp_pos0;
    int64_t     hcp_pos_step = -1;
    mutable uint64_t hcp_n = 0;
    bool        hcp_noted = false;

    /* Qwen tp8 S1 (Qwen loop round 30, QUEUE Q58; qfn_layers.h): this node's trunk-layer ranges (RADIANCE_FN4_LAYERS /
     * _MODE). lr.on false = every layer, the published engine, and nothing below differs. Proxy: the node declares, loads and
     * runs only its layers (weights, experts, KV binds; the PLE where its layer is held), the embedding / final mixer / MTP
     * head / lm_head as the shell; its first layer reads the embedding's stream (the synthetic boundary input) and a gap
     * between two of its ranges passes the wide stream on. lr_next[l]: the next held layer after l (-1 after the last). */
    rad::arch::qfnlayers::Spec lr{};
    std::vector<int> lr_next;
    int lr_first = 0, lr_last = -1;   /* the first / last held layer (every layer: 0 / n_layer - 1) */
    /* Qwen tp8 S2 (Qwen loop round 30): a node of the two-node pipeline (--pp-nodes 2, abi/rad_pp.h), W(h, t): the leader
     * (pp_rank 0) holds [0, h) and [t, n_layer) with the embedding, the PLE layer, the final mixer, lm_head and the MTP
     * head; the follower [h, t) and nothing else. m.lr is the node's strict spec. The crossings carry the wide stream
     * (+ the hc_ss handoff row where declared): X1 after layer h - 1's write (leader -> follower), X2 after layer t - 1's
     * (follower -> leader). op_x1 / op_x2: this node's stage_send / stage_recv of each. */
    bool pp = false;
    int  pp_rank = 0, pp_h = 0, pp_t = 0;
    rad_op op_x1 = 0, op_x2 = 0;
    int  pp_parts = 0;
    int64_t pp_row = 0;
    /* Qwen tp8 S6 (Qwen loop round 32; the DS line's tp8 dev M4 in llama_fp8's form): RADIANCE_PP_SEG_LAYERS=k on the follower
     * with --pp-lanes > 1 -- its [h, t) in segments of k layers, other lanes' passes between them (the core's follower loop).
     * At a boundary s the segment before it saves this rank's wide stream (every row as this rank holds it: an SP form's
     * quarter rows and the rest as they are) into the rank's own ring buffer after layer s - 1's writes, and the segment from s
     * takes it back and reads layer s as the pass reads it after a write it cannot hand its scales (the stage ops' loop mode:
     * card memory, no link, no gather). seg_b: the boundaries, ascending, in (h, t); unset: empty, nothing declared or issued. */
    std::vector<int32_t> seg_b;
    std::vector<rad_op>  seg_save, seg_load;
    rad_buf seg_ring = 0;
    int64_t seg_ring_bytes = 0;
};

static Model g_model[MAX_RANKS];

/* Qwen tp8 S1: whether this node holds trunk layer l (every layer when RADIANCE_FN4_LAYERS is unset) */
static bool lr_held(const Model& m, int64_t l) { return m.lr.held((int)l); }
/* the layer the step runs after l: l + 1, or (layer ranges) the next layer this node holds; -1 after the last */
static int64_t lr_after(const Model& m, int64_t l) {
    if (!m.lr.on) return l + 1 < (int64_t)m.layers.size() ? l + 1 : -1;
    return l >= 0 && l < (int64_t)m.lr_next.size() ? (int64_t)m.lr_next[(size_t)l] : -1;
}
/* Qwen tp8 S2: the embedding, the stream's entry, the final mixer, lm_head and the MTP head -- one machine, a proxy node, or
 * the leader of the two-node pipeline (never its follower) */
static bool lr_shell(const Model& m) { return !m.pp || m.pp_rank == 0; }
/* Qwen tp8 S6: the core's bound on a node's segments (RADIANCE_PP_SEG_MAX, default 16 = 15 boundaries; tp8 loop D33) */
static int pp_seg_cap() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_PP_SEG_MAX");
        const int n = e && *e ? std::atoi(e) : 16;
        return std::min(std::max(n, 2), (int)RAD_PP_MAX_SEGS);
    }();
    return v;
}

/* qfn-on-ds lever TIP (Qwen loop round 21; RADIANCE_FN4_TIP=1 at start, the engine's too): a prompt's tail snapshot at the
 * START OF THE LAST PIECE of its last pipelined prefill step, taken inside the step instead of cutting the step there (lever
 * G at sched_tail_grid 2048 cuts it: the tail then runs as a lone step and pays its fixed cost, +40..+60 ms a cold prompt).
 * The engine names the row (rad_tip_set, on the rank's thread around its issue); the delta net's convolution, solve and scan
 * and the PLE's two windowed ops run [0, row) and [row, T) in two launches and stage each state row between them
 * (rad_arch.h TipStage) -- the same arithmetic: the scan's state is fp32 in the cache and across its chunks, every window is
 * what its launch's last rows leave, and the row is a multiple of every chunk / block / ratio. The QSA tail ring is not read
 * by a resume at a multiple of its ratio (4), so it is the one group left as the step leaves it (rad_tip_free). Kept out of
 * Model (declare resets that): the pinned ring the bounds are uploaded from and their device copy live for the process. */
struct TipRank {
    int64_t   row = 0;
    int     (*stage)(void*, int, int, void*) = nullptr;
    void*     u = nullptr;
    int32_t*  ring_h = nullptr;   /* 64 x {0, row, R, row}: a slot is rewritten 64 TIP steps after its copy was issued */
    int32_t*  cu_d = nullptr;     /* {0, row, T, R, row} on the card (lever TIF: cu_d + 3 = piece 1 up to the row) */
    int64_t   k = 0;
    bool      tried = false;
    bool      noted = false;
};
static TipRank g_tipr[MAX_RANKS];
/* qfn-on-ds Qwen tp8 S7b (round 43; RADIANCE_PP_HEAD_SNAP=1 on the leader, core Engine::head_stage): the engine names "a chunk-end
 * snapshot in this HEAD part" with rad_tip_set row -1; pp_cross hands every head layer's state rows to `stage` before X1's send
 * (the rows the step-end save would read for layers [0, h) had no other lane's head run after this one), and the pass is never
 * recorded while it is named (pass_key). */
struct HeadSnapRank {
    int  (*stage)(void*, int, int, void*) = nullptr;
    void* u = nullptr;
};
static HeadSnapRank g_hsr[MAX_RANKS];
static bool fn4_tip_env() {
    static const bool on = fn4_env_on("RADIANCE_FN4_TIP");
    return on;
}
/* The step's TipStage, its bounds uploaded on the step's stream; false (and the step failed) where the row is not the start
 * of a piece of this pipelined step of one prefill sequence. */
static bool tip_stage_for(RadCtx* c, const RadBatch* batch, const Model& m, TipStage* out) {
    *out = TipStage{};
    const int rk = rad_rank(c);
    TipRank& t = g_tipr[rk];
    if (t.row <= 0) return true;
    const int64_t T = batch->n_tok, R = m.pass_rows;
    /* lever TIF (Qwen loop round 23): also a 64-aligned row inside the step's LAST piece (the delta net's scan chunks and the
     * PLE's windows hand on there as at a piece start; the scheduler plans it under sched_tail_inner 3) */
    const bool tif_row = R > 0 && t.row % 64 == 0 && t.row > (T - 1) / R * R && T > R;
    if (batch->n_seq != 1 || batch->phase != RAD_PHASE_PREFILL || batch->draft_pass != 0 || R <= 0 ||
        (t.row % R != 0 && !tif_row) || t.row >= T || !t.stage) {
        char why[256];
        snprintf(why, sizeof why, "qwen4exp: lever TIP: a snapshot at row %lld of a %lld-row step (%lld sequences, phase %d) "
                                  "is neither the start of a piece nor a 64-aligned row of the last piece of a pipelined "
                                  "prefill step of one sequence (pieces of %lld)",
                 (long long)t.row, (long long)T, (long long)batch->n_seq, batch->phase, (long long)R);
        rad_step_fail(c, why);
        return false;
    }
    if (!t.tried) {
        t.tried = true;
        t.ring_h = (int32_t*)rad_dev_alloc(64 * 4 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        t.cu_d = (int32_t*)rad_dev_alloc(8 * sizeof(int32_t), RAD_MEM_DEVICE);
    }
    if (!t.ring_h || !t.cu_d) {
        rad_step_fail(c, "qwen4exp: lever TIP: no pinned / device memory for the snapshot row's bounds");
        return false;
    }
    /* {0, row} -> cu_d[0..1], the step's T -> cu_d[2], and (lever TIF) {R, row} -> cu_d[3..4] */
    int32_t* h = t.ring_h + 4 * (t.k++ % 64);
    h[0] = 0;
    h[1] = (int32_t)t.row;
    h[2] = (int32_t)R;
    h[3] = (int32_t)t.row;
    const RadStream s = rad_stream(c);
    if (rad_memcpy_async(t.cu_d, h, 2 * sizeof(int32_t), s) != RAD_OK ||
        rad_memcpy_async(t.cu_d + 2, batch->cu_seqlens + 1, sizeof(int32_t), s) != RAD_OK ||
        rad_memcpy_async(t.cu_d + 3, h + 2, 2 * sizeof(int32_t), s) != RAD_OK) {
        rad_step_fail(c, "qwen4exp: lever TIP: the snapshot row's bounds could not be uploaded");
        return false;
    }
    out->row = t.row;
    out->cu = t.cu_d;
    out->cu_in = t.cu_d + 3;
    out->stage = t.stage;
    out->u = t.u;
    if (!t.noted && rk == 0) {
        t.noted = true;
        fprintf(stderr, "radiance: qwen4exp: lever TIP: a %lld-row prefill step takes its tail snapshot at its row %lld (the "
                        "start of its last piece) inside the step: the delta net and the PLE run their state ops in two "
                        "launches and stage the rows between them\n", (long long)T, (long long)t.row);
    }
    return true;
}

/* fn4 BI-dual: switch exl3_dual's value (RADIANCE_FN4_X3DUAL starts it at 1; declared ranks only issue it). */
static std::atomic<int>& fn4_dual_flag() {
    static std::atomic<int> f{fn4_env_on("RADIANCE_FN4_X3DUAL") ? 1 : 0};
    return f;
}
static std::atomic<int> g_dual_declared{0};
/* fn4 BI: switch hc_bi's value (RADIANCE_FN4_HC_BI starts it; libr4d takes the same switch for its forms). */
static std::atomic<int>& fn4_hcbi_flag() {
    static std::atomic<int> f{[] { const char* e = std::getenv("RADIANCE_FN4_HC_BI");
                                   const int v = e && *e ? std::atoi(e) : 0; return v == 5 ? 5 : v == 3 ? 3 : v == 2 ? 2 : v > 0 ? 1 : 0; }()};
    return f;
}
/* fn4 BI: switch bi_seg's value (RADIANCE_FN4_BI_SEG starts it): the trunk's linears cut at the step's segments. */
static std::atomic<int>& fn4_biseg_flag() {
    static std::atomic<int> f{[] { const char* e = std::getenv("RADIANCE_FN4_BI_SEG");
                                   const int v = e && *e ? std::atoi(e) : 0; return v == 3 ? 3 : v == 2 ? 2 : v > 0 ? 1 : 0; }()};
    return f;
}
/* fn4 BI: the step's segments (Model::bi_cut) when hc_bi or bi_seg is on; the reads take them under hc_bi
 * (bi_read), every LinearFP8 under bi_seg (rad::arch::bi_cuts, this rank thread's, cleared at a step without). */
static std::atomic<int>& fn4_sproute_flag();
static void bi_segments(Model& m, const RadBatch* batch) {
    m.bi_mode = fn4_hcbi_flag().load(std::memory_order_relaxed);
    m.bi_on = m.bi_mode > 0;
    m.bi_td = 0;
    const bool seg = fn4_biseg_flag().load(std::memory_order_relaxed) > 0;
    m.bi_cut.clear();
    bi_cuts() = BiCuts{};
    bi_wide_rows() = 0;
    /* lever HCP (sp_route bit 8192): on for a trunk step with host positions (the core always hands them) */
    m.hcp = (fn4_sproute_flag().load(std::memory_order_relaxed) & 8192) != 0 && batch->positions_h &&
            batch->draft_pass == 0 && !batch->enc;
    m.hcp_pos0.clear();
    m.hcp_pos_step = -1;
    const int64_t T = batch->n_tok;
    int64_t sd = 0, td = 0;
    batch_split(batch, &sd, &td);
    if (m.hcp && td == 0 && T > 0) m.hcp_pos_step = batch->positions_h[0];   /* a pure prefill step (the fallback reads) */
    if (!m.bi_on && !seg) return;
    m.bi_td = td;
    m.bi_qd = sd > 0 && td % sd == 0 ? td / sd : 1;   /* a decode sequence's rows (1 + n_spec) */
    /* fn4 BI wide (bi_wide bit 1): the decode rows above the narrow band through the linears' gemm_nt_q_dec */
    if (seg && td > kBiNarrowRows && fn4_biwide(1)) bi_wide_rows() = td;
    m.bi_cut.push_back(0);
    if (td > 0 && td < T) m.bi_cut.push_back(td);
    if (batch->positions_h)
        for (int64_t r = td + 1; r < T; ++r)
            if (batch->positions_h[r] != batch->positions_h[r - 1] + 1) m.bi_cut.push_back(r);
    m.bi_cut.push_back(T);
    if (seg && m.bi_cut.size() > 2) bi_cuts() = BiCuts{m.bi_cut.data(), (int)m.bi_cut.size()};
    if (m.hcp)
        for (size_t i = 0; i + 1 < m.bi_cut.size(); ++i)
            m.hcp_pos0.push_back(m.bi_cut[i] >= td ? (int64_t)batch->positions_h[m.bi_cut[i]] : -1);
}
/* lever HCP (Qwen loop round 23, sp_route bit 8192; QUEUE Q44 lever TIF's premise): libr4d's hc_read splits the down product's
 * K by the launch's row tiles (r4d_hc_bf16.hip hc_down_kchunk; at lowrank 320 / HN 10,240: 2 ways from 1,537 rows, 3 at
 * 1,025-1,536, 4 at 769-1,024, 9 at 257-384, 13 at 129-256), so a prompt's rows read inside a 1,200-row last piece, a full
 * piece or a 267-row tail step sum K in three orders, and every row after differs -- the reason a prefix hit is exact only where the cold run cut its chunk (resume_cold), and lever
 * G's 256 cut moves the cold's bytes (g2k1 / tip0: lA 9 / 12). HCP reads every prefill segment that starts past the prompt's
 * first interval (its first position >= a pass's rows) and has more than 64 rows (the 128-row tile, kTileMinRows) through
 * op_read_sp: kc_rows = the full piece's rows (16 tiles), the split a full piece reads with today, whatever the segment's,
 * whole or a rank's part (libr4d nrt_k: a part of > 64 rows at the 128-row tile). Full pieces keep their bytes; a shorter
 * piece, a tail step and a hit's rows take the full piece's order, so where a row lies in which piece no longer moves it. A
 * prompt's first interval (every newcomer shorter than a pass) is never pinned: no resume point lies inside it. Off = today. */
static bool hcp_seg_ok(const Model& m, int64_t a) {
    if (!m.hcp) return false;
    const int64_t R = m.pass_rows > 0 ? m.pass_rows : 2048;
    if (m.bi_cut.size() < 2 || m.hcp_pos0.size() + 1 != m.bi_cut.size()) return m.hcp_pos_step >= R;
    for (size_t i = 0; i + 1 < m.bi_cut.size(); ++i)
        if (a >= m.bi_cut[i] && a < m.bi_cut[i + 1]) return m.hcp_pos0[i] >= R;
    return false;
}
/* a prefill read of rows [a, a + n) of segment row `a0`'s: through op_read_sp where HCP pins it */
static bool hcp_pins(const Model& m, const HyperConn& h, int64_t a0, int64_t n) {
    return n > 64 && h.op_read_sp && n <= h.c.read_rows && hcp_seg_ok(m, a0);
}
static void hcp_read(RadCtx* c, const Model& m, const HyperConn& h, int64_t T, int64_t a, int64_t n, bool ss) {
    if (hcp_pins(m, h, a, n)) { h.read_sp(c, T, a, n, ss); ++m.hcp_n; return; }
    h.read(c, T, a, n, ss);
}
/* fn4 BI (switch bi_seg 2): the lm_head at each output row's ALONE height class. logits_gemm (fp8 / i8 head) runs a
 * matvec at M <= 2 and WMMA from M = 3; within WMMA a row's bytes do not depend on M (lm_wmma_body: each 128-column scale
 * block's WMMA partial, then fmaf into the row's accumulator, blocks in K order whatever MT / FKB), so only the class
 * matters. Alone, a prompt's last row is M = 1 and a decode sequence's verify rows M = 1 + n_spec. The outputs are the
 * decode rows (core/sched/batch.cpp: a decode entry gives its 1 + n_spec rows, decode entries first) then one row a prompt
 * that finishes this step: the decode rows go in one launch when 1 + n_spec >= 3 (else a launch a sequence, at its own
 * M), each prompt's row in a launch of its own. Returns false (the caller issues one launch) when off, or when the step's
 * outputs do not have that form (uneven verify blocks, scoring rows). 12.11.2026, fn4 180: the logits were the first
 * differing op in 19 of 40 probes, every one a prompt's last row in a mixed step.
 * bi_seg 3 (20.11.2026, qwen-port round 3, s5-bt: logits first-differing in 2 of 20 probes, a row verified at q 3 beside
 * load vs q 2 alone): the decode rows in ONE launch whenever they are 3 or more, at any verify depth. A decode row's alone
 * class is that of its alone step at the published depth (MTP 3, draft_adaptive 0: q 4, WMMA); a step whose depth a
 * row cap lowered (CAP48 / RADIANCE_DECODE_ROW_CAP_UNIFORM: q 2 or 3) then keeps the WMMA class instead of a matvec a
 * sequence. Only steps below q 3 with two or more decode sequences change; a lone sequence below q 3 (cap <= 3) keeps
 * its M = qd launch. */
static bool bi_logits(RadCtx* c, const Model& m, const RadBatch* batch) {
    const int bsm = fn4_biseg_flag().load(std::memory_order_relaxed);
    if (bsm < 2) return false;
    const int64_t n = batch->n_out;
    if (n <= 1) return false;                      /* one row: already its alone class */
    int64_t sd = 0, td = 0;
    batch_split(batch, &sd, &td);
    const int64_t qd = (int64_t)batch->n_spec + 1;
    if (td < 0 || td > n || sd < 0 || td != sd * qd) return false;
    if (n - td > batch->n_seq - sd) return false;  /* more rows than finishing prompts can give */
    const int64_t D = m.g.n_embd, V = m.g.n_vocab;
    auto one = [&](int64_t r0, int64_t k) {
        RAD_ISSUE_N(c, m.op_logits, k, brow_slice(m.b_hout, r0, k, D), RAD_W(m.w_lm_head),
                    brow_slice(m.b_logits, r0, k, V));
    };
    if (td > 0) {
        if (qd >= 3 || (bsm >= 3 && td >= 3)) one(0, td);
        else for (int64_t s = 0; s < sd; ++s) one(s * qd, qd);
    }
    for (int64_t r = td; r < n; ++r) one(r, 1);
    return true;
}
/* A connection's read over rows [r0, r0 + n), cut at the step's segments under hc_bi. */
/* fn4 BI hc_bi 3 (12.11.2026, 193's trace: L00.rd_mix first-differing in 18 of 20 c4 probes): Flash-Next's mixing matrices
 * are E4M3, which hc_bi 1 / 2 do not cover (libr4d: !f8) -- its decode forms move with the row count (G to 8 rows, S 9..16,
 * tiled from 17) while a sequence's own decode read is 4 rows (G). Under 3 the decode rows are read at most kBiDecRows at a
 * time (form G, its rows' bytes the same at every group size; the up product B3 the same), each prefill sequence alone as
 * under 1 / 2. A pure decode step is cut too. libr4d keeps today's forms at 3. */
constexpr int64_t kBiDecRows = 8;
static void bi_read_seg(RadCtx* c, const Model& m, const HyperConn& hcx, int64_t T, int64_t a, int64_t b, bool ss) {
    const bool dec = m.bi_mode >= 3 && a < m.bi_td;
    if (!dec) { hcp_read(c, m, hcx, T, a, b - a, ss); return; }   /* lever HCP: past the first interval, a full piece's K split */
    /* hc_bi 5 (12.11.2026, 199: hc_bi 3's 8-row reads cost 15.7 % at c16): the decode rows in one launch through the
     * twin (libr4d bi_dec: form G in 8-row groups, B3 in 16-row groups; each row's bytes as at 3). */
    if (m.bi_mode == 5 && hcx.op_read_bd) {
        const int64_t pc = hcx.bd_piece();   /* 64, or bd_rows under bi_wide bit 2 */
        for (int64_t p = a; p < b; p += pc)
            hcx.read_bd(c, p, std::min<int64_t>(pc, b - p), ss);
        return;
    }
    /* A connection without the inject gains (the final read: libr4d's form G wants them) takes the per-row kernels
     * below kDecMinRows (8) and the tiled form from 8: its decode rows go a sequence at a time (1 + n_spec rows, the
     * alone read; x02 12.11.: 'final' first-differing in 20 of 20 c4 probes with 8-row pieces). */
    const int64_t pc = hcx.c.inject ? kBiDecRows : std::max<int64_t>(1, std::min<int64_t>(m.bi_qd, kBiDecRows - 1));
    for (int64_t p = a; p < b; p += pc) hcx.read(c, T, p, std::min(pc, b - p), ss);
}
static void bi_read(RadCtx* c, const Model& m, const HyperConn& hcx, int64_t T, int64_t r0, int64_t n, bool ss = false) {
    if (!m.bi_on || (m.bi_cut.size() <= 2 && !(m.bi_mode >= 3 && m.bi_td > 0))) { hcp_read(c, m, hcx, T, r0, n, ss); return; }
    for (size_t i = 0; i + 1 < m.bi_cut.size(); ++i) {
        const int64_t a = std::max(m.bi_cut[i], r0), b = std::min(m.bi_cut[i + 1], r0 + n);
        if (b > a) bi_read_seg(c, m, hcx, T, a, b, ss);
    }
}
/* read_pieces under hc_bi: each segment in pieces of the read's band. */
static void bi_read_pieces(RadCtx* c, const Model& m, const HyperConn& hcx, int64_t T, bool ss = false) {
    if (!m.bi_on || (m.bi_cut.size() <= 2 && !(m.bi_mode >= 3 && m.bi_td > 0))) {
        if (!m.hcp) { hcx.read_pieces(c, T, ss); return; }
        const int64_t pr = hcx.c.read_rows;   /* lever HCP: read_pieces' pieces, each through hcp_read */
        if (pr <= 0 || T <= pr) { hcp_read(c, m, hcx, T, 0, T, ss); return; }
        for (int64_t r0 = 0; r0 < T; r0 += pr) hcp_read(c, m, hcx, T, r0, T - r0 < pr ? T - r0 : pr, ss);
        return;
    }
    const int64_t piece = hcx.c.read_rows > 0 ? hcx.c.read_rows : T;
    for (size_t i = 0; i + 1 < m.bi_cut.size(); ++i)
        for (int64_t a = m.bi_cut[i]; a < m.bi_cut[i + 1]; a += piece)
            bi_read_seg(c, m, hcx, T, a, std::min(a + piece, m.bi_cut[i + 1]), ss);
}

/* ============================== fn4 BI's engine tracer (BiTr) ============================== */
static const char* bt_dir() {
    static const char* d = [] {
        const char* e = std::getenv("RADIANCE_FN4_BITRACE");
        return e && *e && std::strcmp(e, "0") ? e : (const char*)nullptr;
    }();
    return d;
}
/* At the end of a declare: the plane and one bi_rowhash a traced buffer. Declared after everything the model declared, so
 * every traced buffer stays live to the end of the program and shares no storage with a later op's under tracing. */
static int bt_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (!bt_dir()) return RAD_OK;
    auto& t = m.bt;
    const Geom& g = m.g;
    t.on = true;
    t.dir = bt_dir();
    if (const char* e = std::getenv("RADIANCE_FN4_BITRACE_PMAX")) t.pmax = std::max<int64_t>(1, std::atoll(e));
    if (const char* e = std::getenv("RADIANCE_FN4_BITRACE_PLEN")) t.plen = std::max<int64_t>(1, std::atoll(e));
    if (const char* e = std::getenv("RADIANCE_FN4_BITRACE_PMIN")) t.pmin = std::max<int64_t>(0, std::atoll(e));
    t.cap = 64 + 24 * (int64_t)m.layers.size();
    const int64_t MT = g.max_tok;
    t.plane = decl_b(b, m.nm.f("bi_trace_plane"), RAD_I32, {MT, 2 * t.cap}, RAD_BUF_PERSIST);
    if (!t.plane) return RAD_E_INVAL;
    auto add = [&](rad_buf buf, int64_t pitch, uint32_t dtype, bool clear, int64_t rows) {
        if (!buf || pitch <= 0 || t.tb.count(buf)) return;
        BiTr::Tb e;
        e.pitch = pitch;
        e.esz = rad_dtype_bytes(dtype, 1);
        e.clear = clear;
        e.op = rw(b, RAD_OP(b, "bi_rowhash", RAD_PARAMS(RAD_RANGE("M", 1, rows)), RAD_NOWEIGHTS), {buf}, {t.plane});
        t.tb[buf] = e;
    };
    const uint32_t A = g.act_dtype;
    const int64_t HN = m.hccfg.hc * g.n_embd, D = g.n_embd;
    add(m.b_h, HN, A, false, MT);
    add(m.a_x.x, D, A, false, MT);
    add(m.trunk_i8 ? m.a_x.q8 : m.a_x.q, D, m.trunk_i8 ? RAD_I8 : RAD_F8E4M3, true, MT);
    add(m.a_hr.q, D, RAD_F8E4M3, true, MT);
    add(m.b_qg, g.n_head * 2 * g.head_dim, A, true, MT);
    add(m.b_k, g.kv_dim(), A, true, MT);
    add(m.b_v, g.kv_dim(), A, true, MT);
    add(m.b_q, g.q_dim(), A, true, MT);
    add(m.a_attn.x, g.q_dim(), A, true, MT);
    add(m.a_in.x, m.gcfg.conv_dim() + m.gcfg.v_dim(), A, true, MT);
    add(m.b_ab, m.gcfg.ab_dim(), A, true, MT);
    add(m.b_gq, m.gcfg.n_head_k * m.gcfg.head_k, A, true, MT);
    add(m.b_gk, m.gcfg.n_head_k * m.gcfg.head_k, A, true, MT);
    add(m.b_gv, m.gcfg.v_dim(), A, true, MT);
    add(m.a_go.x, m.gcfg.v_dim(), A, true, MT);
    add(m.b_rlogits, m.moecfg.n_expert_all, A, true, MT);
    add(m.b_eids, m.moecfg.top_k, RAD_I32, true, MT);
    add(m.b_ew, m.moecfg.top_k, A, true, MT);
    add(m.b_sout, D, A, true, MT);
    add(m.b_logits, g.n_vocab, RAD_F32, false, g.max_logit_rows);
    for (const auto& kv : t.tb)
        if (!kv.second.op || kv.second.esz < 1) {
            fprintf(stderr, "radiance: qwen4exp: fn4 BI's tracer: bi_rowhash did not resolve (a libr4d with r4d_fn4_bi?)\n");
            return RAD_E_UNSUPPORTED;
        }
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: qwen4exp: fn4 BI's tracer ON (RADIANCE_FN4_BITRACE=%s): %zu traced buffers, %lld points a "
                        "step at most, rows at positions %lld .. %lld of sequences first seen at 0 (or, with PMIN, at a prefix "
                        "hit at or below PMIN) with at most %lld query rows; passes not recorded\n",
                t.dir.c_str(), t.tb.size(), (long long)t.cap, (long long)t.pmin, (long long)t.pmax, (long long)t.plen);
    return RAD_OK;
}
/* This step's point `name` (layer l, -1 for none): its index, allocated on first use, or -1 when the plane is full. */
static int bt_slot(BiTr& t, const char* name, int l, int kind) {
    char nm[64];
    if (l >= 0) snprintf(nm, sizeof nm, "L%02d.%s", l, name);
    else snprintf(nm, sizeof nm, "%s", name);
    const auto it = t.at.find(nm);
    if (it != t.at.end()) return it->second;
    if (t.n >= t.cap) return -1;
    t.at[nm] = t.n;
    t.names.emplace_back(nm);
    t.kind.push_back((uint8_t)kind);
    return t.n++;
}
/* One point (or one piece of one): rows [r0, r0 + rows) of `buf` as the last op left them. */
static void bt_pt(RadCtx* c, Model& m, rad_buf buf, int64_t r0, int64_t rows, const char* name, int l, int kind = 0) {
    auto& t = m.bt;
    if (!t.on || !buf || rows <= 0) return;
    const auto it = t.tb.find(buf);
    if (it == t.tb.end() || !it->second.op) return;
    const int s = bt_slot(t, name, l, kind);
    if (s < 0) return;
    RAD_ISSUE_N(c, it->second.op, rows, brow_slice(buf, r0, rows, it->second.pitch),
                bcol_at(t.plane, r0, 2 * t.cap, 2 * (int64_t)s, 2, rows));
}
/* The step's start: no points yet, the block-internal buffers zeroed over the step's rows. */
static void bt_begin(RadCtx* c, Model& m, int64_t T) {
    auto& t = m.bt;
    if (!t.on) return;
    t.n = 0;
    t.at.clear();
    t.names.clear();
    t.kind.clear();
    const RadStream st = rad_stream(c);
    for (const auto& kv : t.tb)
        if (kv.second.clear)
            if (void* p = rad_buf_ptr(c, kv.first)) rad_memset_async(p, 0, T * kv.second.pitch * kv.second.esz, st);
}
/* The step's end: the plane's rows and the batch's fields to the host (one synchronisation), the selected rows to the file. */
static void bt_flush(RadCtx* c, Model& m, const RadBatch* batch) {
    auto& t = m.bt;
    if (!t.on || t.n == 0) return;
    const int64_t T = batch->n_tok, S = batch->n_seq, NO = batch->n_out > 0 ? batch->n_out : 0, W = 2 * t.cap;
    const int64_t HR = std::min<int64_t>(std::max(T, NO), m.g.max_tok);
    const RadStream st = rad_stream(c);
    std::vector<int32_t> tok((size_t)T), pos((size_t)T), cu((size_t)S + 1), sid((size_t)S), ql((size_t)S),
        cl((size_t)S), oid((size_t)NO);
    auto cp = [&](void* dst, const void* src, int64_t bytes) {
        if (src && bytes > 0) rad_memcpy_async(dst, src, bytes, st);
    };
    cp(tok.data(), batch->token_ids, T * 4);
    cp(pos.data(), batch->positions, T * 4);
    cp(cu.data(), batch->cu_seqlens, (S + 1) * 4);
    cp(sid.data(), batch->seq_ids, S * 4);
    cp(ql.data(), batch->q_lens, S * 4);
    cp(cl.data(), batch->ctx_lens, S * 4);
    cp(oid.data(), batch->out_ids, NO * 4);
    t.host.resize((size_t)HR * (size_t)W);
    cp(t.host.data(), rad_buf_ptr(c, t.plane), HR * W * 4);
    rad_stream_sync(st);
    const int rank = rad_rank(c);
    if (!t.f) {
        char path[1024];
        snprintf(path, sizeof path, "%s/bt-r%d-%d.bin", t.dir.c_str(), rank, (int)getpid());
        t.f = fopen(path, "ab");
        snprintf(path, sizeof path, "%s/bt-r%d-%d.names", t.dir.c_str(), rank, (int)getpid());
        t.fn = fopen(path, "a");
        if (!t.f || !t.fn) {
            fprintf(stderr, "radiance: qwen4exp: fn4 BI's tracer cannot open %s: off\n", path);
            t.on = false;
            return;
        }
    }
    uint64_t key = 1469598103934665603ull;
    for (size_t i = 0; i < t.names.size(); ++i) {
        for (const char ch : t.names[i]) key = (key ^ (uint8_t)ch) * 1099511628211ull;
        key = (key ^ (uint64_t)(t.kind[i] + 1)) * 1099511628211ull;
    }
    if (key != t.names_key) {
        fprintf(t.fn, "N %016llx %d\n", (unsigned long long)key, t.n);
        for (int i = 0; i < t.n; ++i) fprintf(t.fn, "%s %d\n", t.names[(size_t)i].c_str(), (int)t.kind[(size_t)i]);
        fflush(t.fn);
        t.names_key = key;
    }
    int64_t seqdec = 0, tokdec = 0;
    batch_split(batch, &seqdec, &tokdec);
    /* the requests followed: decided at a request's first step (its first row at position 0, a short query); each
     * sequence's slot id replaced by its request's serial */
    std::vector<uint8_t> fol((size_t)S, 0);
    for (int64_t s = 0; s < S; ++s) {
        const auto it = t.card.find(sid[(size_t)s]);
        BiTr::Card& k = t.card[sid[(size_t)s]];
        const int64_t cx = cl[(size_t)s];
        const bool prefill = s >= seqdec;
        if (it == t.card.end() || cx < k.ctx || (prefill && (cx == 0 || cx != k.end))) {
            const int32_t r = cu[(size_t)s];
            k.serial = ++t.serials;
            /* Qwen loop F25 (21.11.2026): with RADIANCE_FN4_BITRACE_PMIN set, a request whose first step starts at a prefix hit
             * (0 < context <= PMIN, its first row at that context) is followed too, so a hit sibling's rows at PMIN.. are traced
             * beside its cold twin's; PMIN unset (0) = the cold-only rule as before */
            const bool cold0 = cx == 0 && r >= 0 && r < T && pos[(size_t)r] == 0;
            const bool hit = t.pmin > 0 && cx > 0 && cx <= t.pmin && r >= 0 && r < T && pos[(size_t)r] == cx;
            k.follow = (cold0 || hit) && ql[(size_t)s] <= t.plen;
        }
        k.ctx = cx;
        k.end = cx + ql[(size_t)s];
        fol[(size_t)s] = k.follow;
        sid[(size_t)s] = (int32_t)k.serial;
    }
    std::vector<int32_t> sel, selq;
    for (int64_t r = 0, sq = 0; r < T; ++r) {
        while (sq + 1 < S && cu[(size_t)sq + 1] <= r) ++sq;
        if (fol[(size_t)sq] && pos[(size_t)r] >= t.pmin && pos[(size_t)r] < t.pmax) {
            sel.push_back((int32_t)r);
            selq.push_back((int32_t)sq);
        }
    }
    if (sel.empty()) return;
    struct {
        char magic[4];
        int32_t rank;
        int64_t step;
        int32_t T, S, NO, NP, nsel, DT, SD, draft;
        uint64_t names;
    } h{ { 'B', 'I', 'T', '2' }, rank, (int64_t)batch->step, (int32_t)T, (int32_t)S, (int32_t)NO, t.n,
         (int32_t)sel.size(), (int32_t)tokdec, (int32_t)seqdec, (int32_t)batch->n_spec, key };
    static_assert(sizeof h == 56, "bitrace.py reads a 56-byte record header");
    fwrite(&h, sizeof h, 1, t.f);
    fwrite(cu.data(), 4, cu.size(), t.f);
    fwrite(sid.data(), 4, sid.size(), t.f);
    fwrite(ql.data(), 4, ql.size(), t.f);
    fwrite(cl.data(), 4, cl.size(), t.f);
    fwrite(oid.data(), 4, oid.size(), t.f);
    std::vector<uint32_t> rec;
    for (size_t k = 0; k < sel.size(); ++k) {
        const int32_t r = sel[k];
        const int32_t hd[4] = { r, pos[(size_t)r], tok[(size_t)r], selq[k] };
        fwrite(hd, 4, 4, t.f);
        rec.assign((size_t)t.n * 2, 0);
        int32_t orow = -1;
        for (int64_t j = 0; j < NO; ++j)
            if (oid[(size_t)j] == r) orow = (int32_t)j;
        for (int i = 0; i < t.n; ++i) {
            const int64_t row = t.kind[(size_t)i] ? orow : r;
            if (row < 0 || row >= HR) continue;
            const size_t a = (size_t)row * (size_t)W + 2 * (size_t)i;
            rec[(size_t)i * 2] = t.host[a];
            rec[(size_t)i * 2 + 1] = t.host[a + 1];
        }
        fwrite(rec.data(), 4, rec.size(), t.f);
    }
    fflush(t.f);
}
/* A connection read's outputs over rows [r0, r0 + n): the block input, its codes, and (`hr`, the feed-forward's read) the
 * rotated codes. */
static void bt_rd(RadCtx* c, Model& m, int64_t r0, int64_t n, const char* name, int l, bool hr) {
    if (!m.bt.on) return;
    char nm[48];
    bt_pt(c, m, m.a_x.x, r0, n, name, l);
    snprintf(nm, sizeof nm, "%s_c", name);
    bt_pt(c, m, m.trunk_i8 ? m.a_x.q8 : m.a_x.q, r0, n, nm, l);
    if (hr) {
        snprintf(nm, sizeof nm, "%s_hr", name);
        bt_pt(c, m, m.a_hr.q, r0, n, nm, l);
    }
}
/* The mixer block's own buffers over the step's rows, after the block. */
static void bt_blk(RadCtx* c, Model& m, const Layer& L, int64_t T, int l) {
    if (!m.bt.on) return;
    if (L.full) {
        bt_pt(c, m, m.b_qg, 0, T, "qg", l);
        bt_pt(c, m, m.b_k, 0, T, "k", l);
        bt_pt(c, m, m.b_v, 0, T, "v", l);
        bt_pt(c, m, m.b_q, 0, T, "q", l);
        bt_pt(c, m, m.a_attn.x, 0, T, "att", l);
    } else {
        bt_pt(c, m, m.a_in.x, 0, T, "g_in", l);
        bt_pt(c, m, m.b_ab, 0, T, "g_ab", l);
        bt_pt(c, m, m.b_gq, 0, T, "g_q", l);
        bt_pt(c, m, m.b_gk, 0, T, "g_k", l);
        bt_pt(c, m, m.b_gv, 0, T, "g_v", l);
        bt_pt(c, m, m.a_go.x, 0, T, "g_o", l);
    }
}
/* The routed feed-forward's routing and shared arm over the step's rows (global rows in every form). */
static void bt_moe(RadCtx* c, Model& m, int64_t T, int l) {
    if (!m.bt.on) return;
    bt_pt(c, m, m.b_rlogits, 0, T, "router", l);
    bt_pt(c, m, m.b_eids, 0, T, "ids", l);
    bt_pt(c, m, m.b_ew, 0, T, "ew", l);
    bt_pt(c, m, m.b_sout, 0, T, "sout", l);
}


/* The n-gram table's row count, by prime search.
 *
 * Each of the `heads` hash heads has its own vocabulary: the 1st, 2nd, ... nth prime above
 * `ngram_vocab_size_base - 1`; the table is their sum, padded to `divisor`
 * (`Qwen4ExpTextNGramEmbedding.__init__`). The per-head sizes are in the checkpoint
 * (`ngram_heads_vocab_sizes`) and the served hash reads them there; only the total is derived,
 * because a weight's shape must be declared before any tensor is opened. rad-convert checks it:
 * a multi-source concat's declared leading extent must equal its sources'.
 *
 * Cost: 16 primes near 2e7 by trial division, once at declare. */
static bool ngram_is_prime(int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}

static int64_t ngram_table_rows(int64_t base, int64_t heads, int64_t divisor) {
    if (base <= 1 || heads <= 0) return 0;
    int64_t total = 0, p = base - 1;
    for (int64_t i = 0; i < heads; ++i) {
        do { ++p; } while (!ngram_is_prime(p));
        total += p;
    }
    return divisor > 1 ? ((total + divisor - 1) / divisor) * divisor : total;
}

/* A rank's place in the shared expert's uneven block split, where the first places take one
 * block more. MoeFP8::declare hands a plain expert's extra 64-wide tiles to the first ranks by
 * index; the shared expert's extra blocks start at the rank after the last of those, so the two
 * remainders land on different ranks when there are enough. (Routed experts leave no remainder:
 * the parity split at two ranks and the four-class split at four are even.) At four ranks and 640
 * columns (ten tiles, five blocks) ranks 0 and 1 take a tile more and rank 2 the block; at two the
 * place is the rank. */
static int64_t shared_place(int64_t n_ff_all, int world, int rank) {
    const int64_t first = (n_ff_all / 64) % world;
    return (rank - first + world) % world;
}

/* Columns [lo, hi) of the slice at `place` when `nb` fp8 blocks split over `world` places, the
 * first nb % world places taking one block more. */
static void block_slice(int64_t nb, int world, int64_t place, int64_t* lo, int64_t* hi) {
    const int64_t base = nb / world, extra = nb % world;
    const int64_t b0 = place * base + std::min<int64_t>(place, extra);
    *lo = b0 * RAD_FP8_BLOCK;
    *hi = (b0 + base + (place < extra ? 1 : 0)) * RAD_FP8_BLOCK;
}

/* Qwen tp8 S5 prerequisite (Qwen loop round 31, found by reading the S2 host declares: W(2, 26)'s leader declared 7 KV groups,
 * its follower 5). The engine indexes KV groups by POSITION on both nodes -- the rendezvous compares 'kv group N' (name, kind,
 * block, window, states) and the step batch layout, the follower replays the leader's block tables, state slots, state clears
 * and checkpoint saves / restores by group index (core/engine_node.cpp pp_connect's second rendezvous, pp_ck_apply) -- so a
 * follower declares the PLE layer's two per-sequence state groups (kv_ple_tok, kv_ple_conv) at the leader's position (the PLE
 * layer's turn in the layer loop: after the upfront groups, before the first QSA layer's kv_qsa_bkey), with the leader's
 * geometry (arch/common/rad_block_ple.h PleLayer::declare, its "two rolling windows") and bound to that layer. Not zeroed: a
 * zero group carves no memory and the follower's checkpoint copy of a stateful group needs an address (core/mem/kv.cpp
 * checkpoint_copy: "group %d has no address"); the follower holds the windows unread (240 KiB + 20 B a state a rank, and
 * the same in each snapshot slot). arch_test flashnext_pp_nodes compares both nodes' group lists. */
static int pp_ple_groups(RadBuilder* b, const RadModelMeta* meta, Model& m, int layer) {
    const Geom& g = m.g;
    const int64_t ngram = meta_int2(meta, "qwen4exp.ngram_size", "ngram_size", 0);
    const int64_t width = meta_int2(meta, "qwen4exp.ple_conv_kernel_size", "ple_conv_kernel_size", 0);
    if (ngram < 2 || width <= 1 || m.hccfg.hc <= 1) return RAD_E_INVAL;
    RadKVGroupDecl td{};
    td.kind = RAD_KV_CONV;
    td.dtype = RAD_I32;
    td.conv_width = ngram;
    td.n_head_kv = 1;
    td.head_dim = 1;
    const rad_kvgroup kv_tok = rad_decl_kv_group(b, m.nm.f("kv_ple_tok"), &td);
    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = g.act_dtype;
    cd.conv_width = (width - 1) * ngram + 1;   /* PleLayer::Config::hist() + 1, the dilation being the n-gram size */
    cd.n_head_kv = 1;
    cd.head_dim = m.hccfg.hc * g.n_embd;
    const rad_kvgroup kv_conv = rad_decl_kv_group(b, m.nm.f("kv_ple_conv"), &cd);
    if (!kv_tok || !kv_conv) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_tok));
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_conv));
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    /* A sizing declare writes scratch: the step reads g_model, filled by the real declare at the
     * real max_tok (RadBuildCtx::shape_probe). */
    static thread_local Model probe_model;
    Model& m = ctx->shape_probe ? probe_model : g_model[ctx->rank];
    /* The reset drops the pinned window copy (pipe_bounds); free it first -- a re-declare of the
     * same rank (every rebuild, every test case) otherwise overwrites the pointer and leaks the
     * pin. The copy is read only by a step's memcpys, and no step runs across a declare. */
    if (m.pipe_cu_h) rad_dev_free(m.pipe_cu_h, RAD_MEM_HOST_PINNED);
    m = Model{};
    m.nm = Names(ctx->scope ? ctx->scope : "");
    Names& nm = m.nm;

    /* arch_splits_ff: the plugin splits the routed experts' width (640 = five fp8 blocks) itself,
     * by whole blocks and parity, at any world up to the block count (including three, which 640
     * does not divide), so geom_from must not refuse it. attn_world_ok: this plugin's attention
     * blocks (rad_block_attn_gated_fp8.h) serve a smaller attention world when the two KV heads do
     * not divide the world. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true));
    Geom& g = m.g;

    /* Qwen4-Exp's norms are all `x_hat * (1 + w)` (`Qwen3NextRMSNorm.forward` is
     * `_norm(x.float()) * (1.0 + weight.float())`, weight initialised to zeros); the delta net's
     * gated output norm is plain `weight * x` (initialised to ones). Taken from the reference. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    /* ---- the gated residual's geometry. */
    m.hccfg.hc      = meta_int2(meta, "qwen4exp.hc_count",   "hc_count",   0);
    m.hccfg.lowrank = meta_int2(meta, "qwen4exp.hc_lowrank", "hc_lowrank", 0);
    if (m.hccfg.hc <= 1 || m.hccfg.lowrank <= 0) {
        fprintf(stderr, "radiance: qwen4exp: the container does not describe the gated residual "
                        "(hc_count %lld, hc_lowrank %lld). Every norm in this architecture is one "
                        "of these connections, so there is nothing to fall back to.\n",
                (long long)m.hccfg.hc, (long long)m.hccfg.lowrank);
        return RAD_E_INVAL;
    }
    /* Every block's all-reduce rides in the following write where a kernel serves that
     * (HyperConn::Config::reduce): the block's partial sum is the write's input, so the pair is one
     * launch. Each block is told below whether its connection's fused op resolved. */
    m.hccfg.reduce = 1;
    const int64_t HN = m.hccfg.hc * g.n_embd;

    /* ---- partial rotary: 0.25 of head_dim 256 is 64; the other 192 dims pass through. */
    {
        const double pr = rad_meta_getf(meta, "text_config.rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "partial_rotary_factor", 1.0)));
        m.n_rot = (int64_t)((double)g.head_dim * pr);
        if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot % 2)) {
            fprintf(stderr, "radiance: qwen4exp: partial_rotary_factor %.6g of head_dim %lld is "
                            "%lld rotary dims, which is not an even count in (0, head_dim]\n",
                    pr, (long long)g.head_dim, (long long)m.n_rot);
            return RAD_E_INVAL;
        }
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));
    /* There is deliberately no serve-time layer slice. A model too large for the device needs a
     * converter flag that writes a named partial container, not a switch that silently serves less
     * than the model. */

    /* ---- the layer schedule. */
    const int64_t interval = meta_int2(meta, "qwen4exp.full_attention_interval",
                                       "full_attention_interval", 0);
    if (interval <= 1) {
        fprintf(stderr, "radiance: qwen4exp: full_attention_interval is %lld; this is a hybrid "
                        "architecture and the interval decides which layers are linear\n",
                (long long)interval);
        return RAD_E_INVAL;
    }
    /* The config file says "full_attention" for the QSA layers (HuggingFace's post_init rewrites
     * them to "qwen_sparse_attention" in memory), so both spellings mean a full layer. */
    RAD_ARCH_TRY(check_layer_schedule("qwen4exp", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval,
                                      {"full_attention", "qwen_sparse_attention"}));

    /* ---- Qwen tp8 S1 (qfn_layers.h): this node's trunk-layer ranges. Not the slice the note above refuses: a node of the
     * two-machine engine (tp8), or its stage timed alone (proxy). Unset, or every layer: m.lr.on stays false and nothing
     * below differs from the published declare. */
    {
        std::string err;
        /* Qwen tp8 S2: a pipeline node's ranges come from its cut (--pp-nodes 2 --pp-cut h,t), strict (the DS line's form) */
        const RadPpInfo pp = rad_pp_query();
        if (pp.nodes == 2) {
            const int n = (int)g.n_layer;
            if (std::getenv("RADIANCE_FN4_LAYERS") && *std::getenv("RADIANCE_FN4_LAYERS")) {
                fprintf(stderr, "radiance: qwen4exp: RADIANCE_FN4_LAYERS and --pp-nodes 2 both name the layers; unset the former\n");
                return RAD_E_INVAL;
            }
            /* the leader holds layer 0 (the embedding's stream, the PLE's token ids) and the last layer (the mixer, the head) */
            if (pp.cut_h <= 0 || pp.cut_t <= pp.cut_h || pp.cut_t >= n || (pp.rank != 0 && pp.rank != 1)) {
                fprintf(stderr, "radiance: qwen4exp: --pp-cut %d,%d (node %d): the follower's layers must lie inside (0, %d) "
                                "and leave the leader layer 0 and the last one\n", pp.cut_h, pp.cut_t, pp.rank, n);
                return RAD_E_INVAL;
            }
            const std::string spec = pp.rank == 1 ? std::to_string(pp.cut_h) + ":" + std::to_string(pp.cut_t)
                                                  : "0:" + std::to_string(pp.cut_h) + "," + std::to_string(pp.cut_t) + ":" +
                                                        std::to_string(n);
            if (rad::arch::qfnlayers::parse(spec.c_str(), "strict", n, &m.lr, &err) < 0 || !m.lr.on ||
                m.lr.leader != (pp.rank == 0)) {
                fprintf(stderr, "radiance: qwen4exp: --pp-cut %d,%d: %s\n", pp.cut_h, pp.cut_t,
                        err.empty() ? "not a two-node split of the trunk" : err.c_str());
                return RAD_E_INVAL;
            }
            m.pp = true;
            m.pp_rank = pp.rank;
            m.pp_h = pp.cut_h;
            m.pp_t = pp.cut_t;
            /* Qwen tp8 S6: the follower's segment boundaries -- every k-th layer after h, at most RADIANCE_PP_SEG_MAX - 1 (the
             * core's bound, default 15). Any layer is one: what crosses a cut is the wide stream alone (S3a), on one node too. */
            const char* sk = std::getenv("RADIANCE_PP_SEG_LAYERS");
            const int k = sk && *sk ? std::atoi(sk) : 0;
            if (m.pp_rank == 1 && k > 0 && pp.lanes > 1)
                for (int x = m.pp_h + k; x < m.pp_t && (int)m.seg_b.size() < pp_seg_cap() - 1; x += k) m.seg_b.push_back(x);
        } else if (rad::arch::qfnlayers::parse(std::getenv("RADIANCE_FN4_LAYERS"), std::getenv("RADIANCE_FN4_LAYERS_MODE"),
                                               (int)g.n_layer, &m.lr, &err) < 0) {
            fprintf(stderr, "radiance: qwen4exp: %s\n", err.c_str());
            return RAD_E_INVAL;
        }
        if (m.lr.on && m.lr.strict && !m.pp) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_FN4_LAYERS=%s with RADIANCE_FN4_LAYERS_MODE=strict: a node's own part "
                            "of W(h, t) is declared by the two-node engine (--pp-nodes 2 --pp-rank r --pp-cut h,t), not by "
                            "the env -- run the stage alone with the proxy mode\n", m.lr.text.c_str());
            return RAD_E_UNSUPPORTED;
        }
        m.lr_first = 0;
        m.lr_last = (int)g.n_layer - 1;
        if (m.lr.on) {
            m.lr_first = m.lr.first;
            m.lr_last = m.lr.last;
            m.lr_next.assign((size_t)g.n_layer, -1);
            int nx = -1;
            for (int l = (int)g.n_layer - 1; l >= 0; --l) {
                m.lr_next[(size_t)l] = nx;
                if (m.lr.held(l)) nx = l;
            }
        }
    }

    /* ---- the delta net. */
    {
        GdnFP8::Config& gc = m.gcfg;
        const int64_t nk = meta_int2(meta, "qwen4exp.ssm.key_head_count",
                                     "linear_num_key_heads", 0);
        const int64_t nv = meta_int2(meta, "qwen4exp.ssm.value_head_count",
                                     "linear_num_value_heads", 0);
        gc.head_k     = meta_int2(meta, "qwen4exp.ssm.key_head_dim",   "linear_key_head_dim",   0);
        gc.head_v     = meta_int2(meta, "qwen4exp.ssm.value_head_dim", "linear_value_head_dim", 0);
        gc.conv_width = meta_int2(meta, "qwen4exp.ssm.conv_kernel",    "linear_conv_kernel_dim", 0);
        gc.chunk      = 64;
        /* `output_gate_type` is a checkpoint property: Qwen3.5/3.6 leave it unset and mean silu,
         * Qwen4-Exp states "sigmoid". libr4d's gated norm has both. */
        const char* ogt = rad_meta_gets(meta, "text_config.output_gate_type",
                          rad_meta_gets(meta, "output_gate_type", nullptr));
        if (!ogt || !*ogt) ogt = rad_meta_gets(meta, "text_config.hidden_act",
                                rad_meta_gets(meta, "hidden_act", "silu"));
        gc.act = ogt;
        if (nk <= 0 || nv <= 0 || gc.head_k <= 0 || gc.head_v <= 0 || gc.conv_width <= 0) {
            fprintf(stderr, "radiance: qwen4exp: the container does not describe the linear "
                            "layers (key heads %lld, value heads %lld, key width %lld, value "
                            "width %lld, conv width %lld)\n",
                    (long long)nk, (long long)nv, (long long)gc.head_k, (long long)gc.head_v,
                    (long long)gc.conv_width);
            return RAD_E_INVAL;
        }
        /* Heads split evenly when they can and unevenly otherwise: 16 K heads over three ranks
         * are 6, 5 and 5, each rank's V heads the same multiple of its K heads (48 / 16 = 3 ->
         * 18, 15, 15), declared as spans of the container's rows (rad_block_gdn_fp8.h,
         * Config::k_first). Nothing is padded; the kernels take any head count. A V count that is
         * not a multiple of the K count is refused. */
        gc.n_head_k_all = nk;
        gc.n_head_v_all = nv;
        if (g.world > 1 && (nk % g.world != 0 || nv % g.world != 0)) {
            if (nv % nk != 0) {
                fprintf(stderr, "radiance: qwen4exp: %lld value heads over %lld key heads do not "
                                "group, so an uneven split over %d ranks is not defined\n",
                        (long long)nv, (long long)nk, g.world);
                return RAD_E_INVAL;
            }
            const int64_t ratio = nv / nk, base = nk / g.world, rem = nk % g.world;
            /* Extra heads go first to ranks without attention (rad_arch.h remainder_place);
             * with none, this is the rank's own place. */
            const int64_t pl = remainder_place(g, g.rank, 0);
            gc.n_head_k = base + (pl < rem ? 1 : 0);
            gc.k_first  = pl * base + (pl < rem ? pl : rem);
            gc.n_head_v = ratio * gc.n_head_k;
            gc.v_first  = ratio * gc.k_first;
            rad_note(b, "rank %d takes delta-net K heads [%lld, %lld) of %lld and V heads "
                        "[%lld, %lld) of %lld", g.rank,
                    (long long)gc.k_first, (long long)(gc.k_first + gc.n_head_k), (long long)nk,
                    (long long)gc.v_first, (long long)(gc.v_first + gc.n_head_v), (long long)nv);
        } else {
            int s;
            if ((s = divide_or_fail(nk, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0) return s;
            if ((s = divide_or_fail(nv, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0) return s;
        }
    }

    /* ---- the routed feed-forward.
     *
     * n_ff is `moe_intermediate_size`, divided by world_size when the world divides it and whole
     * otherwise (geom_from's `arch_splits_ff`). At more than one rank the split below sets the
     * rank's width, by whole fp8 blocks and parity (see the file header). */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert    = meta_int2(meta, "qwen4exp.expert_count",      "num_experts", 0);
        mc.top_k       = meta_int2(meta, "qwen4exp.expert_used_count", "num_experts_per_tok", 0);
        mc.n_ff_exp    = g.n_ff;
        mc.n_ff_shared = meta_int2(meta, "qwen4exp.expert_shared_feed_forward_length",
                                   "shared_expert_intermediate_size", 0);
        /* The shared expert is split by whole fp8 scale blocks: evenly when possible, otherwise
         * as evenly as whole blocks allow, by the rank's place (shared_place). Its 640 columns are
         * five blocks: three on rank 0 and two on rank 1; at four ranks two on rank 2 and one on
         * each other. See MoeFP8::Config::shared_lo. */
        if (mc.n_ff_shared > 0 && g.world > 1 && (mc.n_ff_shared / g.world) % RAD_FP8_BLOCK == 0) {
            int s;
            if ((s = divide_or_fail(mc.n_ff_shared, g.world,
                                    "shared_expert_intermediate_size", &mc.n_ff_shared)) < 0)
                return s;
        } else if (mc.n_ff_shared > 0 && g.world > 1) {
            if (mc.n_ff_shared % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: the shared expert is %lld columns, which is not a whole "
                                "number of %d-column fp8 scale blocks, so it cannot be split "
                                "across %d ranks.\n", (long long)mc.n_ff_shared, RAD_FP8_BLOCK,
                        g.world);
                return RAD_E_INVAL;
            }
            const int64_t nb = mc.n_ff_shared / RAD_FP8_BLOCK;
            block_slice(nb, g.world, shared_place(g.n_ff_all, g.world, g.rank),
                        &mc.shared_lo, &mc.shared_hi);
            mc.n_ff_shared = mc.shared_hi - mc.shared_lo;
            rad_note(b, "the shared expert does not split evenly into %d-column fp8 scale blocks: "
                        "rank %d computes columns [%lld, %lld) of it, %lld of %lld blocks, and the "
                        "all-reduce sums the partials", RAD_FP8_BLOCK, g.rank,
                     (long long)mc.shared_lo, (long long)mc.shared_hi,
                     (long long)(mc.n_ff_shared / RAD_FP8_BLOCK), (long long)nb);
        }
        mc.norm_topk   = 1;
        /* GPTQ calibration run, enabled by the same setting that names the output directory (a
         * calibration with nowhere to write would be a 1.34 GB allocation and two launches per
         * layer for nothing). A separate run from serving; see MoeFP8::Config::calib. */
        mc.calib       = calib_dir() != nullptr ? 1 : 0;
        /* Tokens per pass. 2048 is the linear-state checkpoint interval the scheduler clamps the
         * prefill chunk to, so the shipped configuration is one pass. It also sizes four buffers at
         * `rows * top_k`, so a smaller max_tok lowers it. */
        mc.rows        = g.max_tok < 2048 ? g.max_tok : 2048;
        m.pass_rows    = mc.rows;
        /* fn4 P2x: the expert buffers two pieces tall (the pieces as before) */
        if (fn4_env_on("RADIANCE_FN4_P2X") && g.max_tok >= 2 * mc.rows) mc.rows = 2 * mc.rows;
        if (mc.n_expert <= 0 || mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: qwen4exp: the container describes %lld expert(s) with "
                            "top_k %lld\n", (long long)mc.n_expert, (long long)mc.top_k);
            return RAD_E_INVAL;
        }
        /* ---- the experts' stored format (block fp8, four-bit, four-bit rotated, or plain bf16),
         * read from one expert's encoding (MoeFP8::probe) before anything is sized by it: the
         * split between ranks, the rotated copy of the block input, the connection in front of
         * every routed block, and for bf16 every linear in the model. The encoding comes from the
         * container, checkpoint or recipe being applied, so a container is always served in the
         * format it holds.
         *
         * The probed expert is the first one this rank owns under expert parallelism: the answer
         * decides whether that split is used, and the expert is declared on this rank either way
         * (whole, or a slice under the parity split), so the block skips exactly the map the probe
         * made. */
        mc.probed_expert = (g.world > 1 && mc.n_expert % g.world == 0)
                               ? (int64_t)g.rank * (mc.n_expert / g.world) : 0;
        {
            MoeFP8::Src ps{};
            RAD_ARCH_TRY(moe_expert_sources(b, nm, "format",
                                            "model.language_model.layers.0.mlp.experts", &ps));
            RAD_ARCH_TRY(MoeFP8::probe(b, nm, 0, ps, &mc));
        }
        /* qfn-on-ds: EXL3 experts take the Qwen line's declare (MoeFP8::Config::exl3_v12: protected experts by place,
         * the shared routing map, the fused decode chain, the v12 band); the DeepSeek line's form is deepseek41's. */
        mc.exl3_v12 = mc.expert_exl3 ? 1 : 0;
        /* Plain experts mean a container converted without a quantising recipe, so every linear
         * is plain too: the blocks issue bf16 GEMMs over bf16 activations and nothing quantises
         * (Geom::w_bf16). Each linear refuses a quantised encoding then. The converse is allowed:
         * a quantised container may keep a linear plain (the QSA indexer's projection), which
         * LinearFP8 reads as bf16. */
        g.w_bf16 = mc.expert_bf16 != 0;
        /* The connections in front of the blocks then emit no codes, since nothing reads them. */
        m.hccfg.quant = g.w_bf16 ? 0 : 1;
        /* ============================== EVERY RANK HOLDS A SLICE OF EVERY EXPERT ==============================
         *
         * An expert is 640 columns, five 128-column blocks (the fp8 scale block, the 4-bit scale
         * group and the rotation width), so two ranks cannot halve one.
         *
         * The split is by parity: each parity's five blocks are cut contiguously over the ranks,
         * and the rank taking the extra block alternates with the expert's parity
         * (MoeFP8::Config::ff_lo). At two ranks, rank r's slice of expert e is the first three
         * blocks when e % 2 == r and the last two otherwise. The router is replicated, every rank
         * computes its slice of every routed expert, and the block's all-reduce sums the partials.
         *
         * At four ranks parity alone would give ranks 0 and 1 three blocks of every expert pair
         * and ranks 2 and 3 two, leaving 2 and 3 waiting in the all-reduce. So the extra block
         * rotates over four places by expert number mod 4 (four slice tables, `moe_gemm_q_mod4`),
         * and every rank computes five blocks of every four experts. A layer keeping two experts
         * plain bf16 has 510 quantised ones, and its first two classes are one expert longer.
         * Other remainders stay apart: the plain experts' extra tiles go to ranks 0 and 1
         * (MoeFP8::declare), the shared expert's extra block to rank 2 (shared_place).
         *
         * Not expert parallelism (rank r owning experts [r*E/W, (r+1)*E/W) whole): bytes per rank
         * are equal on average, but routing is random, and a verify step's forty routed slots leave
         * one rank about five experts ahead, costing about a millisecond per step at one sequence
         * in the all-reduce wait. With parity the ranks differ by a fifth of an expert per unit of
         * even/odd imbalance. Whole experts would also break the recipe's pairing of protected
         * experts, which keeps each rank's quantised-expert count even for the two parity tables.
         *
         * A calibration run does split whole experts per rank: the down projection's Hessian spans
         * an expert's whole K, and a slice would lack the cross terms. Whole experts give each rank
         * the full Hessian of every expert it holds, which the packer sums (rad_arch.h,
         * calib_dir). */
        mc.n_expert_all = mc.n_expert;
        /* bf16 experts are also split whole per rank: a slice of a bf16 expert is two row ranges
         * and a column range, which the file tier cannot read as one run, and a bf16 model is
         * served to calibrate and compare, not for decode speed. EXL3 experts likewise
         * (MoeFP8::declare_exl3): the radiance-exl3 ops run an expert whole, at their one-rank
         * geometry, so at two or more ranks each rank serves its own experts (expert parallel)
         * and the block's all-reduce sums the ranks' outputs -- with the routing imbalance the
         * note above prices (about a millisecond a step at one sequence). */
        if (g.world > 1 && (mc.calib || mc.expert_bf16 || mc.expert_exl3)) {
            int s;
            int64_t per = 0;
            if ((s = divide_or_fail(mc.n_expert_all, g.world, "expert_count", &per)) < 0) return s;
            mc.n_ff_exp    = g.n_ff_all;
            mc.n_expert    = per;
            mc.expert_base = (int64_t)g.rank * per;
            rad_note(b, "%s: rank %d owns experts [%lld, %lld) of %lld whole.",
                     mc.calib ? "calibration (each expert's Hessian is over its whole width)"
                              : mc.expert_exl3 ? "EXL3 experts (expert parallel)" : "bf16 experts", g.rank,
                     (long long)mc.expert_base, (long long)(mc.expert_base + mc.n_expert),
                     (long long)mc.n_expert_all);
        } else if (g.world > 1) {
            if (g.n_ff_all % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld columns, which is not a whole "
                                "number of %d-column blocks and cannot be split across ranks.\n",
                        (long long)g.n_ff_all, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            const int64_t nb = g.n_ff_all / RAD_FP8_BLOCK;
            if (nb % g.world == 0) {
                mc.n_ff_exp = g.n_ff_all / g.world;
            } else if (g.world <= nb) {
                /* The uneven split is two slices per rank: each parity's blocks are cut
                 * contiguously across the ranks and the rank taking the odd block alternates with
                 * the parity, so slices are whole blocks and every block has exactly one owner.
                 * Widths are ceil(nb/W) and floor(nb/W). Every rank must get at least one block of
                 * each parity, since an empty slice still joins the all-reduce, so the world may
                 * not exceed nb. */
                for (int q = 0; q < 2; ++q) {
                    /* This rank's place in expert parity q's split: first for its own parity.
                     * Experts do not go remainder-first (rad_arch.h remainder_place): an expert
                     * block is a whole GEMM over every routed token, and a rank taking the extra
                     * block of every class becomes the slowest rank (at three ranks on
                     * Flash-Next, a 50/25/25 split cost 25% per step single-stream). Only the
                     * delta net's remainder, a head's worth of work, goes to the rank without
                     * attention. */
                    block_slice(nb, g.world, (g.rank - q + g.world) % g.world, &mc.ff_lo[q],
                                &mc.ff_hi[q]);
                }
                mc.n_ff_exp = std::max(mc.ff_hi[0] - mc.ff_lo[0], mc.ff_hi[1] - mc.ff_lo[1]);
                /* At four ranks, four classes by expert number mod 4 (MoeFP8::Config::ff_lo4),
                 * used by every layer instead: the same cut with the rank's place shifted by the
                 * class, so the extra block rotates over all four ranks. Every block still has one
                 * owner. */
                if (g.world == 4) {
                    for (int q = 0; q < 4; ++q) {
                        block_slice(nb, 4, (g.rank - q + 4) % 4, &mc.ff_lo4[q], &mc.ff_hi4[q]);
                        mc.n_ff_exp = std::max(mc.n_ff_exp, mc.ff_hi4[q] - mc.ff_lo4[q]);
                    }
                    rad_note(b, "routed experts at four classes: rank %d computes columns [%lld, "
                                "%lld) [%lld, %lld) [%lld, %lld) [%lld, %lld) of the experts "
                                "numbered 0, 1, 2 and 3 mod 4, %lld blocks per four experts like "
                                "every rank",
                             g.rank, (long long)mc.ff_lo4[0], (long long)mc.ff_hi4[0],
                             (long long)mc.ff_lo4[1], (long long)mc.ff_hi4[1],
                             (long long)mc.ff_lo4[2], (long long)mc.ff_hi4[2],
                             (long long)mc.ff_lo4[3], (long long)mc.ff_hi4[3], (long long)nb);
                } else {
                    rad_note(b, "routed experts: rank %d computes columns [%lld, %lld) of each "
                                "even expert and [%lld, %lld) of each odd one, of %lld; %lld "
                                "blocks of %d do not split evenly over %lld ranks, so the extra "
                                "block alternates with the expert's parity",
                             g.rank, (long long)mc.ff_lo[0], (long long)mc.ff_hi[0],
                             (long long)mc.ff_lo[1], (long long)mc.ff_hi[1],
                             (long long)g.n_ff_all, (long long)nb, RAD_FP8_BLOCK,
                             (long long)g.world);
                }
            } else {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld blocks of %d, which %d ranks "
                                "cannot each take a whole block of in both parities. Lower the rank "
                                "count, or serve whole experts (a bf16 container or a calibration "
                                "run does).\n",
                        (long long)nb, RAD_FP8_BLOCK, g.world);
                return RAD_E_INVAL;
            }
        }
    }

    /* ---- the n-gram embedding's layer and the indexer's geometry, each noted where it changes
     * what this build computes. */
    {
        const char* ple = rad_meta_gets(meta, "text_config.ple_layer_ids",
                          rad_meta_gets(meta, "ple_layer_ids", nullptr));
        if (ple && *ple) {
            /* One-indexed: `ple_layer_ids: [2]` is layer_idx 1 (the reference uses
             * `config.ple_layer_ids.index(layer_idx + 1)` and validates [1, num_hidden_layers]). */
            m.ple_layer = (int64_t)std::strtoll(ple, nullptr, 10) - 1;
            if (m.ple_layer >= 0 && m.ple_layer < g.n_layer) m.ple_layer_model = m.ple_layer;
            if (m.ple_layer < 0 || m.ple_layer >= g.n_layer) {
                /* Outside the declared layers: the n-gram embedding is not in this build, and the
                 * note says so rather than dropping it silently. */
                rad_note(b, "ple_layer_ids = %s is outside the %lld layers declared, so the n-gram "
                            "embedding is not in this build.", ple, (long long)g.n_layer);
                m.ple_layer = -1;
            } else if (m.pp && [&] {   /* Qwen tp8 S2: the PLE layer below the cut, so X1 carries no token ids */
                           int h = 0, t = 0;
                           std::string err;
                           if (rad::arch::qfnlayers::pair_of(m.lr, (int)g.n_layer, (int)m.ple_layer, &h, &t, &err) == 0) return false;
                           fprintf(stderr, "radiance: qwen4exp: --pp-cut %d,%d: %s\n", m.pp_h, m.pp_t, err.c_str());
                           return true;
                       }()) {
                return RAD_E_UNSUPPORTED;
            } else if (!lr_held(m, m.ple_layer)) {
                /* Qwen tp8 S1: the n-gram embedding is its layer's; a node that does not hold that layer declares none of it */
                rad_note(b, "ple_layer_ids = %s: layer %lld is another node's (RADIANCE_FN4_LAYERS=%s), so this node declares "
                            "no n-gram embedding.", ple, (long long)m.ple_layer, m.lr.text.c_str());
                m.ple_layer = -1;
            }
        }
    }
    {
        const int64_t budget = meta_int2(meta, "qwen4exp.indexer_budget", "indexer_budget", 0);
        const int64_t ratio  = meta_int2(meta, "qwen4exp.indexer_compress_ratio",
                                         "indexer_compress_ratio", 0);
        m.qsacfg.budget   = budget;
        m.qsacfg.ratio    = ratio;
        m.qsacfg.heads    = meta_int2(meta, "qwen4exp.indexer_n_heads", "indexer_n_heads", 0);
        m.qsacfg.head_dim = meta_int2(meta, "qwen4exp.indexer_head_dim", "indexer_head_dim", 0);
        m.qsacfg.rot        = m.n_rot;
        m.qsacfg.topk       = (ratio > 0) ? budget / ratio : 0;
        m.qsacfg.max_blocks = (ratio > 0) ? (g.max_ctx + ratio - 1) / ratio : 0;
        m.qsacfg.max_work   = (ratio > 0) ? (g.max_tok + ratio - 1) / ratio + g.max_seqs : 0;
        /* The tail's ring (see QsaIndexer::Config::ring), sized for `max_spec`, the deepest
         * speculation any step can carry. */
        m.qsacfg.ring       = ratio + g.max_spec;
        if (budget > 0 && ratio > 0) {
            /* The exactness bound: with `pos + 1 <= budget + ratio - 1` tokens visible there are
             * at most budget/ratio complete blocks, the indexer's top-k keeps all of them, and the
             * tail is always attended, so QSA equals dense causal attention bit for bit. */
            const int64_t exact_to = budget + ratio - 1;
            /* The context bound is not capped: prefill selects per query exactly as decode does
             * (arch/common/rad_qsa.h, docs/QSA.md). There is deliberately no force-dense switch,
             * since above the bound dense attention is not the trained model. */
            if (g.max_ctx > exact_to)
                rad_note(b, "QSA: sparse past %lld tokens of context, %lld blocks of %lld per "
                            "query at prefill and decode; at or below that the engine issues the "
                            "bit-identical dense attention",
                         (long long)exact_to, (long long)(budget / ratio), (long long)ratio);
            else
                rad_note(b, "QSA: dense attention is exact to %lld tokens of context and "
                            "--max-model-len is %lld, so the indexer selects every visible block",
                         (long long)exact_to, (long long)g.max_ctx);
        }
    }

    rad_note(b, "gated residual: %lld streams of %lld (%lld wide), low-rank gate %lld; %lld "
                "connections plus the mixer",
             (long long)m.hccfg.hc, (long long)g.n_embd, (long long)HN,
             (long long)m.hccfg.lowrank, (long long)(2 * g.n_layer));

    /* ---- the KV groups: the same three as qwen35_fp8. */
    {
        RadKVGroupDecl fd{};
        fd.kind      = RAD_KV_FULL;
        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
        fd.n_head_kv = g.n_head_kv;
        fd.head_dim  = g.head_dim;
        /* --kv-cache-dtype fp8_v4 / k6_v4 (RadBuildCtx::kv_quant): V as int4 in groups of 32 with
         * f16 scales, K E4M3 as above or int6 the same way (libr4d's gqa12 and four-rank gqa6
         * rows for these layers). A token's row is 25 or 22 * head_dim / 16 bytes instead of
         * 2 * head_dim, so the group is declared at half that and the core sizes pages from it;
         * only those kernels read the layout. The MTP head's layer shares this group. The engine
         * asks only for a layout probe() offered. */
        const int kvq = fd.dtype == RAD_F8E4M3 ? ctx->kv_quant : RAD_KVQ_NONE;
        const char* kvq_name = kvq == RAD_KVQ_V4 ? "fp8_v4" : kvq == RAD_KVQ_K6V4 ? "k6_v4"
                                                                                 : nullptr;
        if (kvq != RAD_KVQ_NONE && !kvq_name) {
            fprintf(stderr, "radiance: KV cache layout %d is not one this plugin knows\n", kvq);
            return RAD_E_UNSUPPORTED;
        }
        /* A rank's ratio: twelve at one and two ranks, six at four (each KV head replicated
         * across a rank pair). */
        if (kvq_name && !(g.head_dim == 256 && (g.gqa() == 12 || g.gqa() == 6))) {
            fprintf(stderr, "radiance: --kv-cache-dtype %s has libr4d rows at head_dim 256 and a "
                            "rank's gqa 12 or 6 only; this rank is head_dim %lld, gqa %lld\n",
                    kvq_name, (long long)g.head_dim, (long long)g.gqa());
            return RAD_E_UNSUPPORTED;
        }
        const int64_t row16 = kvq == RAD_KVQ_V4 ? 25 : kvq == RAD_KVQ_K6V4 ? 22 : 32;
        fd.head_dim = g.head_dim * row16 / 32;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
        g.kv_dtype = kvq_name ? kvq_name : rad_dtype_name(fd.dtype);
        if (kvq_name)
            rad_note(b, "kv cache %s: %s K, int4 V, in groups of 32 with f16 scales, %lld bytes a "
                        "token and head instead of %lld", kvq_name,
                     kvq == RAD_KVQ_V4 ? "E4M3" : "int6", (long long)(g.head_dim * row16 / 16),
                     (long long)(2 * g.head_dim));

        RadKVGroupDecl sd{};
        sd.kind         = RAD_KV_LINEAR;
        sd.dtype        = RAD_F32;   /* f16 trades draft acceptance for a sliver of step time */
        sd.n_head_kv    = m.gcfg.n_head_v;
        sd.state_dim[0] = m.gcfg.head_v;
        sd.state_dim[1] = m.gcfg.head_k;
        m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

        RadKVGroupDecl cd{};
        cd.kind       = RAD_KV_CONV;
        cd.dtype      = RAD_BF16;
        cd.n_head_kv  = 2 * m.gcfg.n_head_k + m.gcfg.n_head_v;
        cd.head_dim   = m.gcfg.head_k;
        cd.conv_width = m.gcfg.conv_width;
        m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);
        if (!m.kv_full || !m.kv_state || !m.kv_conv) return RAD_E_INVAL;

        /* ---- the QSA indexer's tail, which is state, not a buffer.
         *
         * It carries an unfinished block's raw index keys across steps, so it is indexed by
         * sequence and layer, which a step-local buffer cannot express (arch/common/rad_qsa.h,
         * Wire). As a linear group the engine keeps a slot stable for the sequence's life, zeroes
         * it on allocation, and saves and restores it with the linear state on prefix-cache
         * checkpoints, so a resume onto a half-built block reads defined data.
         *
         * `n_head_kv` 1 and `state_dim` {ring, head_dim}: `ring` raw keys, a few KiB per layer per
         * sequence. */
        if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0 && m.qsacfg.ring > 0) {
            RadKVGroupDecl td{};
            td.kind         = RAD_KV_LINEAR;
            td.dtype        = g.act_dtype;
            td.n_head_kv    = 1;
            td.state_dim[0] = m.qsacfg.ring;
            td.state_dim[1] = m.qsacfg.head_dim;
            m.qsacfg.kv_tail = rad_decl_kv_group(b, nm.f("kv_qsa_tail"), &td);
            if (!m.qsacfg.kv_tail) return RAD_E_INVAL;
        }
    }

    /* ---- the trunk's stored format, read from the first delta-net layer's input projection as
     * the experts' is from one expert. The activation buffers below carry an int8 code pair only
     * when the trunk's linears are int8 (LinearFP8::i8), so it must be known first. The probe maps
     * the weight through the layer's own LinearFP8, which then does not map it again. */
    m.layers.resize((size_t)g.n_layer);
    {
        const char* pre0 = "model.language_model.layers";
        for (int64_t l = 0; l < g.n_layer; ++l) {
            if (is_full_attention(l, interval)) continue;
            /* Qwen tp8 S1: the first delta-net layer this node holds (the probe maps that layer's weight) */
            if (!lr_held(m, l)) continue;
            RAD_ARCH_TRY(m.layers[(size_t)l].gdn.in.probe(
                b, nm, g, nm.f("blk.%lld.ssm_inz", (long long)l),
                { nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre0, (long long)l),
                  nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight", pre0, (long long)l) }));
            m.trunk_i8 = m.layers[(size_t)l].gdn.in.probed_i8;
            break;
        }
        bool gdn_held = !m.lr.on;
        for (int64_t l = 0; l < g.n_layer && !gdn_held; ++l) gdn_held = lr_held(m, l) && !is_full_attention(l, interval);
        if (!gdn_held) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_FN4_LAYERS=%s holds no delta-net layer: the trunk's stored format is "
                            "read from one (Qwen tp8 S1 runs ranges with at least one)\n", m.lr.text.c_str());
            return RAD_E_UNSUPPORTED;
        }
    }

    /* ---- buffers. */
    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = m.gcfg.v_dim();
    const int64_t conv_dim = m.gcfg.conv_dim();
    const int64_t T        = g.max_tok;
    const int64_t ER       = m.moecfg.rows * m.moecfg.top_k;

    m.b_h   = decl_b(b, nm.f("h_wide"), g.act_dtype, {T, HN});
    m.a_x.x = decl_b(b, nm.f("x"),      g.act_dtype, {T, g.n_embd});
    if (ctx->shape_probe) {
        m.span_on = g_model[ctx->rank].span_on;
        m.pipe_on = g_model[ctx->rank].pipe_on;
    } else {
        m.span_on = ctx->max_mixed_tok > 0 && ctx->max_mixed_tok < g.max_tok &&
                    g.max_tok > m.pass_rows;
        /* Only where the blocks issue their own all-reduce at a step wider than a pass: at two
         * ranks the gated residual's fused write takes every step's (ar_exact_write_rows), so such
         * a step runs serially. */
        m.pipe_on = m.span_on && g.world > 1 && ar_exact_write_rows(g) < g.max_tok;
    }
    /* The pipelined step's all-reduces run on the second lane, in place over `x`, so it is the one
     * buffer that lane touches; it is live over the whole program anyway. */
    if (m.pipe_on) RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_x.x));
    /* A transient although only its last entry changes per step: a persistent buffer sits in the
     * fixed region under every transient, and one more alignment unit there moves every
     * activation by 256 bytes, which on Flash-Next at four ranks slows the MoE gather by 73 % and
     * a prefill by 3 %. Every step that reads it rewrites it (pipe_bounds). */
    if (m.span_on)   /* lever QRS (RADIANCE_FN4_QRS=1): + {0, pass / 4} at [4, 5], a rank's quarter window (rad_qsa.h) */
        m.b_pipe_cu = decl_b(b, nm.f("pipe_cu"), RAD_I32, {(fn4_qrs_declared() && g.world == 4) ? 6 : 4});
    /* Under a span every connection reads a step wider than a pass a pass at a time
     * (HyperConn::read_pieces, and the pipelined step's own reads), so the reads are declared no
     * wider than a pass and their scratch is sized for that band. */
    if (m.span_on) m.hccfg.read_rows = m.pass_rows;
    RAD_ARCH_TRY(m.a_x.declare_qs(b, nm, g, "x", g.n_embd, 0, m.trunk_i8));
    /* The connections write int8 codes themselves when every reader of their codes is an int8
     * linear: the trunk's are, and the routed experts then read the rotated copy or the bf16
     * plane. When experts read the plain E4M3 codes, the connections keep those and the int8
     * linears quantise for themselves. */
    /* fn4 E12 (10.11.2026): EXL3 experts read the bf16 plane as bf16 ones do (MoeFP8::act_bf16), so with
     * RADIANCE_FN4_X3CODES=1 their connections write the int8 codes too and the int8 linears stop quantising for
     * themselves (148's stamps: quant_act_fp8<false, true> 1.64 ms of a 35.9 ms c16 pass, plus its launches). The same
     * bytes: hc_quant_group<true> and quant_act_fp8_body<false, true> take one amax a 128-group of the same bf16 x,
     * scale m * (1 / 127), and rintf of the quotient clamped to +-127. */
    const bool x3codes = m.moecfg.expert_exl3 && fn4_env_on("RADIANCE_FN4_X3CODES");
    m.hccfg.codes_i8 = (m.trunk_i8 && (m.moecfg.expert_rot || m.moecfg.expert_bf16 || x3codes)) ? 1 : 0;
    m.a_x.q8_fed     = m.hccfg.codes_i8 != 0;
    m.b_inj = decl_b(b, nm.f("hc_inj"), g.act_dtype, {T, m.hccfg.hc});

    m.b_qg     = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {T, g.n_head, 2 * g.head_dim});
    m.b_k      = decl_b(b, nm.f("attn_k"),   g.act_dtype, {T, kv_dim});
    m.b_v      = decl_b(b, nm.f("attn_v"),   g.act_dtype, {T, kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {T, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {T, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, m.trunk_i8));
    /* The gate epilogue in front of the int8 out projection writes its int8 codes itself. */
    m.a_attn.q8_fed = m.trunk_i8;
    /* The bf16 gate's own plane (AttnGatedFP8::Wire::gate); the fp8 path fuses it away. */
    if (g.w_bf16) m.b_gate = decl_b(b, nm.f("attn_gate"), g.act_dtype, {T, q_dim});
    /* The QSA indexer's planes, shared by every full-attention layer like `attn_out`. Declared
     * only when the checkpoint carries an indexer; otherwise the layer's declare() is a no-op. */
    if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0) {
        m.qsaw.qk = decl_b(b, nm.f("qsa_qk"), g.act_dtype,
                           { g.max_tok, (m.qsacfg.heads + 1) * m.qsacfg.head_dim });
        m.qsaw.q  = decl_b(b, nm.f("qsa_q"),  g.act_dtype,
                           { g.max_tok, m.qsacfg.heads * m.qsacfg.head_dim });
        const int64_t hd = m.qsacfg.head_dim, rt = m.qsacfg.ratio, wk = m.qsacfg.max_work;
        m.qsaw.stage = decl_b(b, nm.f("qsa_stage"), g.act_dtype, { wk * rt, hd });
        m.qsaw.page  = decl_b(b, nm.f("qsa_page"),  RAD_I32, { wk });
        /* Three components a block under M-RoPE (arch/common/rad_qsa.h). */
        m.qsaw.bpos  = g.rope_mc ? decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk, 3 })
                                 : decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk });
        m.qsaw.nc    = decl_b(b, nm.f("qsa_nc"),    RAD_I32, { g.max_seqs });
        /* A pass tall under a span whose shared steps fit in a pass (RadBuildCtx::
         * max_mixed_tok): a wider step is one sequence and scores and selects in windows of a
         * pass (QsaIndexer::step, with the bounds in b_pipe_cu). The plane is the largest buffer
         * the span would otherwise grow, max_ctx / ratio floats a query row (512 MiB more at 4096
         * rows and 256K of context). */
        const int64_t qsa_rows = (m.span_on && ctx->max_mixed_tok <= m.pass_rows)
                                     ? m.pass_rows : g.max_tok;
        m.qsacfg.plane_rows = qsa_rows < g.max_tok ? qsa_rows : 0;
        /* One row per query, not per sequence (arch/common/rad_qsa.h). The score plane is the
         * only buffer whose size is the product of two deployment bounds, so an oversized one is
         * refused here, naming the two flags, rather than failing later as an allocation. */
        const int64_t sc_elems = qsa_rows * m.qsacfg.max_blocks;
        if (sc_elems > (int64_t)512 * 1024 * 1024) {
            fprintf(stderr,
                "radiance: qwen4exp: the QSA score plane would be %lld MiB -- %lld rows "
                "(--max-num-batched-tokens %lld) by --max-model-len %lld / compress ratio %lld. "
                "Lower either.\n",
                (long long)(sc_elems * 4 / (1024 * 1024)), (long long)qsa_rows,
                (long long)g.max_tok, (long long)g.max_ctx, (long long)m.qsacfg.ratio);
            return RAD_E_UNSUPPORTED;
        }
        m.qsaw.score = decl_b(b, nm.f("qsa_score"), RAD_F32,
                              { qsa_rows, m.qsacfg.max_blocks });
        m.qsaw.sel   = decl_b(b, nm.f("qsa_sel"),   RAD_I32,
                              { g.max_tok, m.qsacfg.topk + 1 });
        m.qsaw.nsel  = decl_b(b, nm.f("qsa_nsel"),  RAD_I32, { g.max_tok });
        m.qsaw.sequ  = decl_b(b, nm.f("qsa_sequ"),  RAD_I32, { g.max_tok });
        if (!m.qsaw.qk || !m.qsaw.q || !m.qsaw.stage || !m.qsaw.page ||
            !m.qsaw.bpos || !m.qsaw.nc || !m.qsaw.score || !m.qsaw.sel || !m.qsaw.nsel ||
            !m.qsaw.sequ)
            return RAD_E_INVAL;
        /* fn4 qsa_split (arch/common/rad_qsa.h, qwen/QSA-SPLIT.md): the candidate lists, declared only when the start asks
         * for the split. A split step has at most one row a verify position a sequence. */
        if (fn4_qsasplit_declared() && m.qsacfg.topk > 0) {
            const int64_t lw = 4 + 2 * m.qsacfg.topk;
            const int64_t rows = g.max_seqs * (1 + (int64_t)g.max_spec);
            m.qsacfg.split      = fn4_qsasplit_declared();
            m.qsacfg.split_rows = rows;
            m.qsacfg.split_min  = fn4_qsasplit_min();
            m.qsaw.qlist = decl_b(b, nm.f("qsa_lists"), RAD_I32, { rows, 8 * lw });
            if (g.world > 1) {
                m.qsaw.qpart = decl_b(b, nm.f("qsa_part"), RAD_I32, { rows, lw });
                m.qsaw.qgath = decl_b(b, nm.f("qsa_gath"), RAD_I32, { rows, (int64_t)g.world * lw });
            }
            if (!m.qsaw.qlist || (g.world > 1 && (!m.qsaw.qpart || !m.qsaw.qgath))) return RAD_E_INVAL;
            if (g.rank == 0)
                fprintf(stderr, "radiance: qwen4exp: fn4 qsa_split declared (start value %d): steps of <= %lld rows from %lld "
                                "live blocks; lists %lld + %lld + %lld KiB a card\n",
                        fn4_qsasplit_flag().load(), (long long)rows, (long long)m.qsacfg.split_min,
                        (long long)(rows * 8 * lw * 4 / 1024), (long long)(g.world > 1 ? rows * lw * 4 / 1024 : 0),
                        (long long)(g.world > 1 ? rows * g.world * lw * 4 / 1024 : 0));
        }
        /* qfn-on-ds lever QRS (Qwen loop round 29, QUEUE Q53; arch/common/rad_qsa.h fn4_qrs_flag): a rank's quarter of each
         * score-plane window, staged a window a slot and gathered on the second lane -- declared only when the start asks
         * (RADIANCE_FN4_QRS=1) and a step can be wider than the plane (a span); sp_route bit 32768 turns it on in process. */
        if (fn4_qrs_declared() && g.world == 4 && m.b_pipe_cu && m.qsacfg.plane_rows > 0 &&
            m.qsacfg.plane_rows % g.world == 0 && m.qsacfg.topk > 0) {
            const int64_t q = m.qsacfg.plane_rows / g.world;
            const int64_t wins = (g.max_tok + m.qsacfg.plane_rows - 1) / m.qsacfg.plane_rows;
            m.qsacfg.qrs_rows = q;
            m.qsacfg.qrs_wins = wins;
            m.qsaw.qrs_sel  = decl_b(b, nm.f("qsa_qrs_sel"),  RAD_I32, { wins * q, m.qsacfg.topk + 1 });
            m.qsaw.qrs_nsel = decl_b(b, nm.f("qsa_qrs_nsel"), RAD_I32, { wins * q });
            m.qsaw.qrs_sequ = decl_b(b, nm.f("qsa_qrs_sequ"), RAD_I32, { wins * q });
            if (!m.qsaw.qrs_sel || !m.qsaw.qrs_nsel || !m.qsaw.qrs_sequ) return RAD_E_INVAL;
            /* the gathers run on the second lane: every buffer they touch is live for the program (rad_buf_concurrent) */
            for (rad_buf cb : { m.qsaw.qrs_sel, m.qsaw.qrs_nsel, m.qsaw.qrs_sequ, m.qsaw.sel, m.qsaw.nsel, m.qsaw.sequ })
                RAD_ARCH_TRY(rad_buf_concurrent(b, cb));
            if (g.rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: qwen4exp: lever QRS declared (RADIANCE_FN4_QRS; sp_route bit 32768): a step wider "
                                "than the %lld-row score plane scores and selects %lld rows a rank a window and gathers the "
                                "selections on the second lane; staging %lld KiB a card for %lld windows\n",
                        (long long)m.qsacfg.plane_rows, (long long)q,
                        (long long)(wins * q * (m.qsacfg.topk + 3) * 4 / 1024), (long long)wins);
        }
    }

    m.a_in.x = decl_b(b, nm.f("gdn_in"), g.act_dtype, {T, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab   = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {T, m.gcfg.ab_dim()});
    m.b_gq   = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gk   = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gv   = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    /* Both f32, not the activation dtype; `gdn_conv_prep` requires it and otherwise fails the
     * first step with "dtype this kernel does not serve". The log decay is a cumulative sum whose
     * differences the chunked scan exponentiates, and bf16's eight mantissa bits are not enough for
     * it or for beta. */
    m.b_gdec = decl_b(b, nm.f("gdn_gdec"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_beta = decl_b(b, nm.f("gdn_beta"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_kkt  = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16, {T, m.gcfg.n_head_v, m.gcfg.chunk});
    m.a_go.x = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim, 0, m.trunk_i8));
    /* The delta net's output epilogue also writes its int8 codes; its prefill quantiser is the
     * int8 one. */
    m.a_go.q8_fed = m.trunk_i8;

    /* The model's expert count, not this rank's: the router is replicated and scores every
     * expert. */
    m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {T, m.moecfg.n_expert_all});
    m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {T, m.moecfg.top_k});
    m.b_ew      = decl_b(b, nm.f("moe_ew"),     g.act_dtype, {T, m.moecfg.top_k});
    m.b_eoff    = decl_b(b, nm.f("moe_eoff"),   RAD_I32,     {m.moecfg.n_expert + 1});
    /* PERSIST, not transient. `moe_scatter` writes the per-expert histogram and no declared op
     * reads it (rad_route_report is a handoff, not an op), so as a transient liveness would let a
     * later buffer alias it, and a wide prefill chunk would corrupt the counts (even negative).
     * One int32 per expert. */
    m.b_ecnt    = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,     {m.moecfg.n_expert},
                         RAD_BUF_PERSIST);
    m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {ER});
    /* fn4 F2a, RADIANCE_FN4_GATHER_INV=1: the sort's inverse slot table, so the routed gather is
     * moe_gather_inv's one launch instead of moe_gather's three (MoeFP8: router_gemm_topk_scatter_inv
     * writes it at decode, moe_scatter_inv above the fused router's rows). Unset: not declared. */
    /* RADIANCE_FN4_SWITCHABLE=1 declares it too, with both paths (MoeFP8::inv_switchable), so the
     * runtime switch (rad_fn4_switch "gather_inv") changes nothing in the buffer plan. */
    if (fn4_env_on("RADIANCE_FN4_GATHER_INV") || fn4_env_on("RADIANCE_FN4_SWITCHABLE"))
        m.b_einv = decl_b(b, nm.f("moe_inverse"), RAD_I32, {ER});
    m.b_egu     = decl_b(b, nm.f("moe_egu"),    g.act_dtype, {ER, 2 * m.moecfg.n_ff_exp});
    m.a_eff.x   = decl_b(b, nm.f("moe_eff"),    g.act_dtype, {ER, m.moecfg.n_ff_exp});
    /* ER rows, not max_tok: an expert plane has one row per (token, slot). declare_qs's default
     * would make it `top_k` times too short and the gated quantiser would overrun it (see
     * ActFP8::declare_qs). */
    RAD_ARCH_TRY(m.a_eff.declare_qs(b, nm, g, "moe_eff", m.moecfg.n_ff_exp, ER));
    if (m.moecfg.expert_rot)
        RAD_ARCH_TRY(m.a_hr.declare_qs(b, nm, g, "moe_hr", g.n_embd));
    m.b_edn     = decl_b(b, nm.f("moe_edn"),    g.act_dtype, {ER, g.n_embd});
    m.b_emoe    = decl_b(b, nm.f("moe_out"),    g.act_dtype, {T, g.n_embd});
    if (m.moecfg.n_ff_shared > 0) {
        m.b_sgu   = decl_b(b, nm.f("moe_sgu"),   g.act_dtype, {T, 2 * m.moecfg.n_ff_shared});
        m.a_sff.x = decl_b(b, nm.f("moe_sff"),   g.act_dtype, {T, m.moecfg.n_ff_shared});
        RAD_ARCH_TRY(m.a_sff.declare_qs(b, nm, g, "moe_sff", m.moecfg.n_ff_shared, 0,
                                         m.trunk_i8));
        /* The shared expert's gated quantiser in front of its int8 down projection writes its codes
     * too. */
        m.a_sff.q8_fed = m.trunk_i8;
        m.b_sout  = decl_b(b, nm.f("moe_sout"),  g.act_dtype, {T, g.n_embd});
        m.b_sgate = decl_b(b, nm.f("moe_sgate"), g.act_dtype, {T, 1});
    }

    m.b_hout   = decl_b(b, nm.f("hout"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded as in qwen35_fp8. This checkpoint ships
     * `lm_head.weight` (tie_word_embeddings is false); the alternative source covers a smaller
     * family member that ties them.
     *
     * The head is declared over all its planes at the codes' dtype (bf16 as shipped, block fp8
     * where a recipe makes it so); the selector then picks a `logits_gemm` kernel whose layout
     * reads that encoding. The name maps come first, since they are how the head is queried. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));
    /* Qwen tp8 S2: the embedding, its all-reduce, the stream's entry and lm_head are the leader's (lr_shell); a pipeline
     * follower declares none of them (its stream arrives in X1) */
    if (lr_shell(m)) {
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    if (!m.w_tok || !m.w_lm_head) return RAD_E_INVAL;
    /* The rank's rows, stated explicitly: a vocabulary the world does not divide (248320 at three
     * ranks) is split by rad_arch.h's rule (ceil to a 128-row block, last rank short), not the
     * core's even share. Where it divides, the span equals the even share. */
    if (g.world > 1) {
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok,     g.vocab_off, g.vocab_off + g.n_vocab));
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));
    }

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                  RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                  RAD_INT("vocab_offset", g.vocab_off)),
                       RAD_WEIGHTS(m.w_tok)),
                   {}, {m.a_x.x});
    /* The encoder's rows overwrite the lookup before its reduction (see MediaRows). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0 && !m.lr.on;
    /* Qwen tp8 S1: not on a node with layer ranges (the tower is the leader's, and a proxy times the trunk's stage) */
    if (m.lr.on && VisionTower::present(meta) && ctx->max_enc_patches > 0)
        rad_note(b, "pictures: off on a node with layer ranges (RADIANCE_FN4_LAYERS=%s)", m.lr.text.c_str());
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, T, g.n_embd, m.a_x.x));
    /* Always exact: this sums disjoint rows, a token's embedding coming from exactly one rank. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", g.n_embd, T * g.n_embd),
                                         RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                              RAD_NOWEIGHTS),
                          {m.a_x.x}, {m.a_x.x});

    /* The stream's initial value: hc identical copies of the embedding. */
    RAD_ARCH_TRY(m.enter.declare(b, g, m.hccfg.hc, m.a_x.x, m.b_h));
    }   /* lr_shell */

    /* The control vector's per-token switch, when this deployment found one (HcProject). All zeros
     * is the stock model, so a request that does not ask for the vector is unaffected. */
    const rad_buf proj_on = HcProject::declare_mask(b, nm.f("projection_on"), g, ctx->cvec);

    /* ---- the layers.
     *
     * Declaration order is the buffer planner's liveness: a transient lives over [first
     * definition, last use] in op indices, and transients with disjoint ranges may share bytes
     * (core/build/rad_bufplan.cpp). The buffers above are shared by every layer, so the order
     * below must match step()'s issue order (read, block, write, ...). Declaring a connection out
     * of order would alias buffers live in the same step, silently corrupting the output.
     */
    m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)),
                                    RAD_INT("rot", m.n_rot), RAD_F64("theta", g.theta),
                                    RAD_F64("scale", g.rope_scale), RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32,
                                               {rope_table_rows(g), m.n_rot})});
    /* Qwen tp8 S3a: on the follower no MTP head reads the table after the trunk, so its planned life ends at the last
     * attention block -- declared BEFORE that layer's indexer, which is issued first: an indexer buffer first written there
     * (a node with one QSA layer: arch_test's W(2, 6) follower) could share the table's bytes. Kept out of the pool there
     * (<= 67 MB at 262k context; the follower holds no embedding or lm_head). */
    if (m.pp && m.pp_rank == 1 && m.b_rope_cs) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_rope_cs));

    AttnGatedFP8::Wire aw{};
    aw.x = m.b_h; aw.h = m.a_x; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn; aw.gate = m.b_gate;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;
    /* The indexer reads the same rope table: its rotation uses `m.n_rot`, `g.theta` and
     * `g.rope_scale`, as the table does, which saves the trunk layers and the head's passes a norm
     * and rope each. Set here because the table is declared after `m.qsaw`'s buffers;
     * `tw.qsa = m.qsaw` below passes it on to the MTP head. */
    m.qsaw.cos_sin = aw.cos_sin;

    GdnFP8::Wire gw{};
    gw.x = m.b_h; gw.h = m.a_x; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;

    MoeFP8::Wire mw{};
    mw.x = m.b_h; mw.h = m.a_x; mw.hr = m.a_hr;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt; mw.inv = m.b_einv;
    mw.egu = m.b_egu; mw.eff = m.a_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;

    HyperConn::Wire hw{};
    hw.h = m.b_h; hw.x = m.a_x; hw.inj = m.b_inj;
    if (fn4_env_on("RADIANCE_FN4_HC_SS")) {
        m.b_hss = decl_b(b, nm.f("hc_ss"), RAD_F32, {T, 8});
        hw.ss = m.b_hss;
    }

    m.layers.resize((size_t)g.n_layer);
    const char* pre = "model.language_model.layers";
    /* The pipelined step's all-reduces run beside the other piece's work, so each block declares
     * its collective a second time at `overlap` 1, for the rows that leave the CUs to that work
     * (the lane rows of libr4d/r4d_rows.cpp). */
    const int pipe_overlap = m.pipe_on ? 1 : 0;
    /* Qwen tp8 S2 / S3a: a pipeline node's stage ops over the wide stream, declared in issue order -- the follower's X1
     * receive before its first layer (its first definition of the stream), the leader's X1 send + X2 receive before layer t,
     * the follower's X2 send after its last layer -- since declaration order is the planner's liveness. The hc_ss handoff
     * does not cross (S3a): the receiving read computes the scales itself, the same bits as a read handed them (docs/OPS.md
     * `ss`: 'same bits as the read's pass'), so the crossing is the stream alone, 20,480 B a row. */
    std::vector<rad_buf> pp_bufs = { m.b_h };
    if (m.pp) {
        m.pp_parts = (int)pp_bufs.size();
        m.pp_row = rad::arch::qfnlayers::crossing_row_bytes(m.hccfg.hc, g.n_embd, false);
    }
    auto pp_stage = [&](bool send, int ring) {
        rad_op h = RAD_OP(b, send ? "stage_send" : "stage_recv",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("ring", ring), RAD_INT("parts", m.pp_parts)),
                          RAD_NOWEIGHTS);
        if (h == RAD_NULL_HANDLE) return h;
        if (send) rad_op_reads(b, h, pp_bufs.data(), (int)pp_bufs.size());
        else      rad_op_writes(b, h, pp_bufs.data(), (int)pp_bufs.size());
        return h;
    };
    /* Qwen tp8 S6: this rank's segment ring buffer (the loop-mode header, the wide stream at max_tok rows, the trailer; kept
     * across the passes run between two segments) and each boundary's save / restore, declared where they are issued (the
     * save after layer s - 1, the restore before layer s) */
    if (m.pp && !m.seg_b.empty()) {
        m.seg_ring_bytes = RAD_PP_RING_HDR + (int64_t)rad_pp_align((uint64_t)(g.max_tok * m.pp_row)) + RAD_PP_TRAILER;
        m.seg_ring = decl_b(b, nm.f("pp_segring"), RAD_U8, {m.seg_ring_bytes}, RAD_BUF_PERSIST);
        m.seg_save.assign(m.seg_b.size(), 0);
        m.seg_load.assign(m.seg_b.size(), 0);
    }
    auto seg_op = [&](bool save) {
        rad_op h = RAD_OP(b, save ? "stage_send" : "stage_recv",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("ring", 2), RAD_INT("parts", m.pp_parts),
                                     RAD_INT("loop", 1)),
                          RAD_NOWEIGHTS);
        if (h == RAD_NULL_HANDLE) return h;
        if (save) rad_op_reads(b, h, pp_bufs.data(), (int)pp_bufs.size());
        else      rad_op_writes(b, h, pp_bufs.data(), (int)pp_bufs.size());
        return h;
    };
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        if (m.pp && m.pp_rank == 1 && l == m.lr_first) m.op_x1 = pp_stage(false, RAD_PP_X1);   /* the follower takes X1 */
        for (size_t k = 0; k < m.seg_b.size(); ++k)                                            /* Qwen tp8 S6 */
            if (m.seg_b[k] == (int32_t)l && (!(m.seg_save[k] = seg_op(true)) || !(m.seg_load[k] = seg_op(false)))) {
                fprintf(stderr, "radiance: qwen4exp: RADIANCE_PP_SEG_LAYERS: the segment boundary at layer %lld did not declare "
                                "(no stage_send / stage_recv in loop mode: a kernel library without the two-node pipeline's "
                                "stage ops)\n", (long long)l);
                return RAD_E_UNSUPPORTED;
            }
        if (m.pp && m.pp_rank == 0 && l == m.pp_t) {                                           /* the leader's cut */
            m.op_x1 = pp_stage(true, RAD_PP_X1);
            m.op_x2 = pp_stage(false, RAD_PP_X2);
        }
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        lay.attn.ar_overlap = lay.gdn.ar_overlap = lay.mlp.ar_overlap = pipe_overlap;
        /* Qwen tp8 S1: another node's layer -- no weights, experts, KV binds or ops here (its PLE: m.ple_layer is -1); a pipeline
         * follower still declares the PLE layer's two state groups at their position (S5 prerequisite, pp_ple_groups) */
        if (!lr_held(m, l)) {
            if (m.pp && m.pp_rank == 1 && l == m.ple_layer_model) RAD_ARCH_TRY(pp_ple_groups(b, meta, m, (int)l));
            continue;
        }

        /* PLE comes first, before the attention connection reads the stream: the reference adds
         * it into `hidden_states` at the top of the decoder layer, so the following gated residual
         * normalises a stream that already includes the n-gram features. It is declared here, not
         * hoisted, so its buffers' liveness starts where they are used. */
        if (l == m.ple_layer) {
            PleLayer::Config pc{};
            pc.hc         = m.hccfg.hc;
            pc.embed_dim  = meta_int2(meta, "qwen4exp.ple_embed_dim", "ple_embed_dim", 0);
            pc.ngram      = meta_int2(meta, "qwen4exp.ngram_size", "ngram_size", 0);
            const int64_t per_ngram =
                meta_int2(meta, "qwen4exp.heads_per_ngram", "heads_per_ngram", 0);
            pc.heads      = (pc.ngram - 1) * per_ngram;
            pc.eos        = meta_int2(meta, "qwen4exp.eos_token_id", "eos_token_id", -1);
            pc.shards     = meta_int2(meta, "qwen4exp.split_ngram_parts", "split_ngram_parts", 0);
            pc.conv_width = meta_int2(meta, "qwen4exp.ple_conv_kernel_size",
                                      "ple_conv_kernel_size", 0);
            /* The dilation is the n-gram size (`conv_dilation = config.ngram_size` in the
             * reference), not a separate key and not 1. */
            pc.dilation   = pc.ngram;
            pc.rows       = ngram_table_rows(
                meta_int2(meta, "qwen4exp.ngram_vocab_size_base", "ngram_vocab_size_base", 0),
                pc.heads,
                meta_int2(meta, "qwen4exp.make_ngram_vocab_size_divisible_by",
                          "make_ngram_vocab_size_divisible_by", 1));

            PleLayer::Wire pw{};
            pw.h = m.b_h;

            PleLayer::Src ps{};
            ps.shard_prefix = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_embedding.shard_",
                                      pre, (long long)l);
            ps.shard_suffix = ".weight";
            ps.mult         = nm.ckpt("%s.%lld.ple.ple_embedding.layer_multipliers",
                                      pre, (long long)l);
            ps.vocab_sizes  = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_vocab_sizes",
                                      pre, (long long)l);
            ps.offsets      = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_offsets",
                                      pre, (long long)l);
            ps.key_proj     = nm.ckpt("%s.%lld.ple.key_proj.weight",   pre, (long long)l);
            ps.value_proj   = nm.ckpt("%s.%lld.ple.value_proj.weight", pre, (long long)l);
            ps.norm_key     = nm.ckpt("%s.%lld.ple.norm_key.weight",   pre, (long long)l);
            ps.norm_query   = nm.ckpt("%s.%lld.ple.norm_query.weight", pre, (long long)l);
            ps.norm_conv    = nm.ckpt("%s.%lld.ple.norm_conv.weight",  pre, (long long)l);
            ps.conv1d       = nm.ckpt("%s.%lld.ple.conv1d.weight",     pre, (long long)l);
            RAD_ARCH_TRY(m.ple.declare(b, nm, g, pc, (int)l, pw, ps));
        }

        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("%s.%lld.attn_hyper_connection.hc_norm.weight", pre, (long long)l);
        mixs.mix_down = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        mixs.mix_up   = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        mixs.inject_w = nm.ckpt("%s.%lld.attn_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        HyperConn::Config mcfg = m.hccfg;
        mcfg.sp_twins = fn4_spser_env() && g.world == 4 ? 1 : 0;   /* fn4 SP-serial: the K-pinned part reads */
        RAD_ARCH_TRY(lay.hc_mix.declare(b, nm, g, nm.f("blk.%lld.attn_hc", (long long)l),
                                        mcfg, hw, mixs));

        /* `Src::norm = nullptr` is the blocks' "input already normed and quantised" mode, and
         * `add_out = false` stops them adding into the stream; the connection's write half does
         * that with the four gains the read produced. `fold` and `ar_in` are off because both fuse
         * into a norm these blocks do not have. */
        if (lay.full) {
            AttnGatedFP8::Src as{};
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
            /* The KV page is the QSA compress block, so the indexer's selection is a paged block
             * table rather than a new attention kernel (docs/QSA.md); the attention rows must
             * declare the same 4 or nothing resolves. Page size only affects the block-table walk:
             * pages of 4 and 16 cost the same at long context and give identical text. */
            lay.attn.kv_block = 4;
            /* The selection must be set before the attention is declared: `attn_paged` declares
             * `sel` and `seqused` as reads, and a read declared after the op does not count
             * (rad_block_attn_gated_fp8.h). The indexer itself is declared after the attention
             * block, where the sparse attention reads its ops.
             *
             * Below the indexer's budget the selection is every visible block, so sparse and dense
             * paths must give identical text; any difference is a defect. */
            if (m.qsaw.qk) {
                lay.attn.qsa_sel      = m.qsaw.sel;
                lay.attn.qsa_nsel     = m.qsaw.nsel;
                lay.attn.qsa_sequ     = m.qsaw.sequ;
                lay.attn.qsa_topk     = m.qsacfg.topk;
                /* The exactness bound, below which prefill issues the cheaper dense attention.
                 * Decode ignores it and always takes the sparse path, keeping that path exercised
                 * on every step. */
                lay.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            /* An attention-zero rank keeps no attention cache: its block is a fill of zeros
             * (rad_block_attn_gated_fp8.h), and the indexer reads only the shared block table. The
             * call is idempotent. */
            if (g.attn_zero && m.kv_full) RAD_ARCH_TRY(rad_kv_group_zero(b, m.kv_full));
            lay.attn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          false, false, kArNone, lay.hc_mix.op_ar_write != 0));
            /* The indexer, after the attention block so its ops sit where the sparse attention
             * reads them. It reads the same normed fp8 input as the projections
             * (arch/common/rad_qsa.h, docs/QSA.md). */
            if (m.qsaw.qk) {
                QsaIndexer::Config qc = m.qsacfg;
                qc.mode    = lay.attn.mode;
                qc.qk_proj = nm.ckpt("%s.%lld.self_attn.indexer.index_qk_proj.weight",
                                     pre, (long long)l);
                qc.q_norm  = nm.ckpt("%s.%lld.self_attn.indexer.q_layernorm.weight",
                                     pre, (long long)l);
                qc.k_norm  = nm.ckpt("%s.%lld.self_attn.indexer.k_layernorm.weight",
                                     pre, (long long)l);
                qc.kv_attn  = m.kv_full;
                qc.emit_sel = lay.attn.qsa_sel != 0;
                RAD_ARCH_TRY(lay.qsa.declare(b, nm, g, (int)l, qc, m.qsaw, aw.h,
                                             &m.kv_qsa_bkey));
            }
        } else {
            GdnFP8::Src gs{};
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",   pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",   pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",   pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",      pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",              pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",            pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",        pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            lay.gdn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, m.gcfg, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         false, false, kArNone, lay.hc_mix.op_ar_write != 0));
        }

        HyperConn::Src ffns{};
        ffns.norm     = nm.ckpt("%s.%lld.mlp_hyper_connection.hc_norm.weight", pre, (long long)l);
        ffns.mix_down = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        ffns.mix_up   = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        ffns.inject_w = nm.ckpt("%s.%lld.mlp_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        /* The rotated expert plane's input codes come out of this read, beside the plain ones
         * the shared arm takes, so the routed arm declares no quantiser of its own. */
        HyperConn::Config fcfg = m.hccfg;
        fcfg.sp_twins = fn4_spser_env() && g.world == 4 ? 1 : 0;   /* fn4 SP-serial */
        HyperConn::Wire   fw   = hw;
        if (m.moecfg.expert_rot) { fcfg.rotate = 1; fw.rq = m.a_hr.q; fw.rs = m.a_hr.s; }
        RAD_ARCH_TRY(lay.hc_ffn.declare(b, nm, g, nm.f("blk.%lld.ffn_hc", (long long)l),
                                        fcfg, fw, ffns));

        MoeFP8::Src ms{};
        ms.cfg              = m.moecfg;
        ms.router           = nm.ckpt("%s.%lld.mlp.gate.weight", pre, (long long)l);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "layer",
                                        nm.ckpt("%s.%lld.mlp.experts", pre, (long long)l), &ms));
        if (m.moecfg.n_ff_shared > 0) {
            ms.shared      = nm.ckpt("%s.%lld.mlp.shared_expert", pre, (long long)l);
            ms.shared_gate = nm.ckpt("%s.%lld.mlp.shared_expert_gate.weight", pre, (long long)l);
        }
        lay.mlp.ar_out_take = lay.hc_ffn.ar_take();
        lay.mlp.hr_in = fcfg.rotate != 0;
        lay.mlp.inv_switchable = m.b_einv && fn4_env_on("RADIANCE_FN4_SWITCHABLE");   /* fn4 F2a */
        if (lay.mlp.inv_switchable)   /* the path the switch starts on: RADIANCE_FN4_GATHER_INV */
            fn4_gather_inv_flag().store(fn4_env_on("RADIANCE_FN4_GATHER_INV") ? 1 : 0,
                                        std::memory_order_relaxed);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, false, false, kArNone,
                                     lay.hc_ffn.op_ar_write != 0));
        /* The write behind the block takes its gather where it can, declared after the block so
         * the buffers it reads are the block's outputs in declaration order. */
        {
            HyperConn::GatherSrc gs{};
            gs.ye     = lay.mlp.w.edn;
            gs.ew     = lay.mlp.w.ew;
            gs.sorted = lay.mlp.w.sorted;
            if (lay.mlp.gfold) { gs.sh = lay.mlp.w.sout; gs.sg = lay.mlp.w.sgate; }
            gs.top_k  = lay.mlp.c.top_k;
            gs.rows   = lay.mlp.c.rows;
            RAD_ARCH_TRY(lay.hc_ffn.declare_gather(b, gs));
            lay.mlp.gather_out_rows  = lay.hc_ffn.gather_rows;
            lay.mlp.gather_out_rows6 = lay.hc_ffn.gather_rows6;
        }

        /* The control vector's projection for this layer, if the vector steers it. */
        RAD_ARCH_TRY(lay.proj.declare(b, nm.f("blk.%lld.proj", (long long) l), g, l, ctx->cvec,
                                      proj_on, m.b_h, m.hccfg.hc));
    }

    /* Each routed layer predicts the next one's experts at its router input (MoeFP8::xp_next). */
    if (m.pp && m.pp_rank == 1) m.op_x2 = pp_stage(true, RAD_PP_X2);   /* the follower's result back (Qwen tp8 S3a) */
    /* (Qwen tp8 S1: the next layer this node runs) */
    for (int64_t l = 0; l + 1 < (int64_t)m.layers.size(); ++l)
        if (lr_held(m, l) && lr_after(m, l) >= 0) m.layers[(size_t)l].mlp.xp_next = &m.layers[(size_t)lr_after(m, l)].mlp;
    /* qfn-on-ds P2S: with the two-piece buffers declared, say at start whether a layer's MoE takes a joint pass of two
     * pieces (MoeFP8::pair_ok) and, if not, the first condition that refuses it */
    if (fn4_env_on("RADIANCE_FN4_P2X") && g.rank == 0 && !m.layers.empty() && m.pass_rows > 0) {
        const char* why = m.layers[(size_t)m.lr_first].mlp.pair_why(2 * m.pass_rows, m.pass_rows);
        fprintf(stderr, "radiance: qwen4exp: P2S: a joint routed pass of two %lld-row pieces (MoE rows %lld): %s\n",
                (long long)m.pass_rows, (long long)m.moecfg.rows, why ? why : "allowed");
    }

    /* fn4 SP-hc (test builds): declared only on request, at four ranks under the pipelined prefill. */
    /* fn4 KVCAP C1: the wide stream's all-gather bounded by the rows one launch carries (Model::sp_agh_rows) */
    const int64_t agh_rows = std::max<int64_t>(4, std::min<int64_t>(T, std::max<int64_t>(m.pass_rows,
                                                                                          128LL * HyperConn::kSpTiles)));
    if (fn4_env_on("RADIANCE_FN4_SPHC") && g.world == 4 && m.pipe_on) {
        const int64_t HN = m.hccfg.hc * g.n_embd;
        m.sp_rs = rw(b, RAD_OP(b, "reduce_scatter_rows",
                         RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                    RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", g.wire_exact ? 1 : 0),
                                    RAD_INT("overlap", 1)),
                         RAD_NOWEIGHTS),
                     {m.a_x.x}, {m.a_x.x});
        m.sp_ag = rw(b, RAD_OP(b, "all_gather_rows",
                         RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                    RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", g.wire_exact ? 1 : 0),
                                    RAD_INT("overlap", 1)),
                         RAD_NOWEIGHTS),
                     {m.a_x.x}, {m.a_x.x});
        m.sp_agh = rw(b, RAD_OP(b, "all_gather_rows",
                          RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * HN, agh_rows * HN),
                                     RAD_STR("dtype", g.dtype), RAD_INT("row", HN), RAD_INT("raw", 1),
                                     RAD_INT("overlap", 1)),
                          RAD_NOWEIGHTS),
                      {m.b_h}, {m.b_h});
        m.sp_agh_rows = agh_rows;
        if (fn4_env_on("RADIANCE_FN4_SP_ROUTE") && g.wire_exact && m.b_rlogits) {   /* fn4 SP-route */
            const int64_t NE = m.moecfg.n_expert_all;
            if (NE % 32 == 0) {
                m.sp_agl = rw(b, RAD_OP(b, "all_gather_rows",
                                  RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * NE, T * NE),
                                             RAD_STR("dtype", g.dtype), RAD_INT("row", NE), RAD_INT("raw", 1),
                                             RAD_INT("overlap", 1)),
                                  RAD_NOWEIGHTS),
                              {m.b_rlogits}, {m.b_rlogits});
                if (m.sp_agl) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_rlogits));
            }
            if (g.rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: qwen4exp: fn4 SP-route %s (RADIANCE_FN4_SP_ROUTE; switch sp_route: the router GEMM on "
                                "each rank's quarter of an SP-hc piece, the logits gathered raw)\n",
                        m.sp_agl ? "declared" : "NOT served -- all_gather_rows over the logits did not resolve");
        }
        bool ok = m.sp_rs && m.sp_ag && m.sp_agh;
        const int64_t band = m.hccfg.read_rows > 0 ? m.hccfg.read_rows : T;
        for (Layer& lay : m.layers) {
            if (!lr_held(m, &lay - m.layers.data())) continue;   /* Qwen tp8 S1: another node's layer */
            ok = ok && lay.hc_mix.declare_codes(b, band) == RAD_OK && lay.hc_ffn.declare_codes(b, band) == RAD_OK;
            /* the exact form: every connection has its K-split-pinned quarter read (HyperConn::op_read_sp) */
            ok = ok && lay.hc_mix.op_read_sp && lay.hc_ffn.op_read_sp;
        }
        m.sp_on = ok && m.sp_agh_rows >= std::min<int64_t>(T, m.pass_rows);   /* fn4 KVCAP C1: a piece fits the gather */
        if (g.rank == 0)
            fprintf(stderr, "radiance: qwen4exp: fn4 SP-hc %s (RADIANCE_FN4_SPHC; switch hc_sp picks it a pipelined "
                            "prefill step)\n", ok ? "declared" : "NOT served -- reduce_scatter_rows / all_gather_rows / "
                            "hc_codes missing");
    }

    /* fn4 SP-serial (14.11.2026): the same collectives and codes as SP-hc (declared here when SP-hc is not), and on every
     * trunk connection the part reads pinned to a segment's row-tile count (HyperConn::op_read_kt, declared with the
     * connections). A serial step's prefill rows then take the sequence-parallel residual (step_serial_sp). */
    if (fn4_spser_env() && g.world == 4) {
        const int64_t HN = m.hccfg.hc * g.n_embd;
        if (!m.sp_rs)
            m.sp_rs = rw(b, RAD_OP(b, "reduce_scatter_rows",
                             RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                        RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", g.wire_exact ? 1 : 0),
                                        RAD_INT("overlap", 1)),
                             RAD_NOWEIGHTS),
                         {m.a_x.x}, {m.a_x.x});
        if (!m.sp_ag)
            m.sp_ag = rw(b, RAD_OP(b, "all_gather_rows",
                             RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                        RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", g.wire_exact ? 1 : 0),
                                        RAD_INT("overlap", 1)),
                             RAD_NOWEIGHTS),
                         {m.a_x.x}, {m.a_x.x});
        if (!m.sp_agh) {
            m.sp_agh = rw(b, RAD_OP(b, "all_gather_rows",
                              RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * HN, agh_rows * HN),
                                         RAD_STR("dtype", g.dtype), RAD_INT("row", HN), RAD_INT("raw", 1),
                                         RAD_INT("overlap", 1)),
                              RAD_NOWEIGHTS),
                          {m.b_h}, {m.b_h});
            m.sp_agh_rows = agh_rows;
        }
        bool ok = m.sp_rs && m.sp_ag && m.sp_agh && g.wire_exact;
        const int64_t band = m.hccfg.read_rows > 0 ? m.hccfg.read_rows : T;
        /* the longest segment every connection has a twin for */
        int tmax = HyperConn::kSpTiles;
        for (Layer& lay : m.layers) {
            if (!lr_held(m, &lay - m.layers.data())) continue;   /* Qwen tp8 S1: another node's layer */
            for (HyperConn* h : { &lay.hc_mix, &lay.hc_ffn }) {
                if (h->c.quant && !h->op_codes) ok = ok && h->declare_codes(b, band) == RAD_OK;
                int t = 1;
                while (t + 1 <= HyperConn::kSpTiles && h->op_read_kt[t + 1]) ++t;
                tmax = std::min(tmax, t);
            }
        }
        m.spser_rows = 128LL * tmax;
        m.spser_ok = ok && tmax >= 2 && !m.layers.empty() &&
                     m.sp_agh_rows >= std::min<int64_t>(T, m.spser_rows);   /* fn4 KVCAP C1: a run fits the gather */
        /* lever LSK: SP-serial's own collectives with lane_kind 1 (every param else = the shared op's: the same libr4d
         * instance); a library that does not take the param leaves them 0 (the shared ops, today) */
        if (m.spser_ok && fn4_env_on("RADIANCE_FN4_SPSER_LANE")) {
            m.ss_rs = rw(b, RAD_OP(b, "reduce_scatter_rows",
                             RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                        RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", 1),
                                        RAD_INT("overlap", 1), RAD_INT("lane_kind", 1)),
                             RAD_NOWEIGHTS),
                         {m.a_x.x}, {m.a_x.x});
            m.ss_ag = rw(b, RAD_OP(b, "all_gather_rows",
                             RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * g.n_embd, T * g.n_embd),
                                        RAD_STR("dtype", g.dtype), RAD_INT("row", g.n_embd), RAD_INT("raw", 1),
                                        RAD_INT("overlap", 1), RAD_INT("lane_kind", 1)),
                             RAD_NOWEIGHTS),
                         {m.a_x.x}, {m.a_x.x});
            m.ss_agh = rw(b, RAD_OP(b, "all_gather_rows",
                              RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", 4 * HN, m.sp_agh_rows * HN),
                                         RAD_STR("dtype", g.dtype), RAD_INT("row", HN), RAD_INT("raw", 1),
                                         RAD_INT("overlap", 1), RAD_INT("lane_kind", 1)),
                              RAD_NOWEIGHTS),
                          {m.b_h}, {m.b_h});
            if (!(m.ss_rs && m.ss_ag && m.ss_agh)) m.ss_rs = m.ss_ag = m.ss_agh = 0;
            if (g.rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: qwen4exp: lever LSK (RADIANCE_FN4_SPSER_LANE): SP-serial's own collectives %s "
                                "(lane_kind 1: libr4d's lane pair ar_laneblocks_s / ar_lanethreads_s, 0 there = the first "
                                "pair's)\n", m.ss_rs ? "declared" : "NOT served -- the shared ops stay");
        }
        if (g.rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: qwen4exp: fn4 SP-serial %s (RADIANCE_FN4_SPSERIAL; switch sp_serial picks it a serial "
                            "step: prefill segments up to %lld rows, K split pinned to the segment's row tiles)\n",
                    m.spser_ok ? "declared" : "NOT served -- a collective, the codes or the pinned reads did not resolve "
                                              "(or the wire is not exact)", (long long)m.spser_rows);
    }
    if (m.sp_agh && g.rank == 0 && !ctx->shape_probe) {
        const int64_t HN = m.hccfg.hc * g.n_embd;
        fprintf(stderr, "radiance: qwen4exp: fn4 KVCAP C1: the wide stream's all-gather (sp_agh) is declared for %lld rows "
                        "(%lld elements; the step is %lld rows)%s\n", (long long)m.sp_agh_rows,
                (long long)(m.sp_agh_rows * HN), (long long)T,
                m.sp_agh_rows * HN == T * g.n_embd ? ": the same bound as the trunk's all-reduces, so libr4d serves it from "
                                                     "their peer scratch instance" : "");
    }

    /* ---- Qwen tp8 S2: the two crossings of W(h, t). Each carries what the next layer's connection read takes after a
     * layer's hc_ffn.write: the wide stream's rows [T, hc x n_embd] bf16 (20,480 B a row on Flash-Next) and, where the writes
     * hand their norm scales on (RADIANCE_FN4_HC_SS: hc_ss, [T, 8] f32), that row too -- the same two operands the read takes
     * in one process, so no arithmetic moves across the cut. X1 after layer h - 1 (leader -> follower), X2 after layer
     * t - 1 (follower -> leader); declared alike on both nodes so the rendezvous compares them. Every Flash-Next state is
     * its own layer's and the PLE layer sits below h (its token ids stay on the leader), so nothing else crosses. */
    if (m.pp) {   /* (the stage ops are declared where they are issued: in the layer loop above, Qwen tp8 S3a) */
        rad_pp_declare_crossing(RAD_PP_X1, m.pp_row, m.pp_parts);
        rad_pp_declare_crossing(RAD_PP_X2, m.pp_row, m.pp_parts);
        if (!m.op_x1 || !m.op_x2) {
            fprintf(stderr, "radiance: qwen4exp: --pp-nodes 2: stage_send / stage_recv did not resolve (a kernel library with "
                            "the stage ops: libr4d r4d_stage.hip, libref ref_stage.cpp)\n");
            return RAD_E_UNSUPPORTED;
        }
        /* Qwen tp8 S6: the follower's segment boundaries, for the core's follower loop (every rank, every declare) */
        if (!m.seg_b.empty()) {
            std::string bs;
            for (int32_t x : m.seg_b) bs += (bs.empty() ? "" : ", ") + std::to_string(x);
            rad_note(b, "two-node pipeline (Qwen tp8 S6, RADIANCE_PP_SEG_LAYERS): the follower's passes may run in segments at "
                        "layers %s of W(%d,%d), each boundary's wide stream saved and restored on this rank (a %lld-byte ring "
                        "buffer, loop mode)", bs.c_str(), m.pp_h, m.pp_t, (long long)m.seg_ring_bytes);
            if (g.rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: qwen4exp: two-node pipeline (Qwen tp8 S6, RADIANCE_PP_SEG_LAYERS): the follower's "
                                "passes may run in segments at layers %s of W(%d,%d), each boundary's wide stream saved and "
                                "restored on this rank (a %lld-byte ring buffer, loop mode; no gather)\n", bs.c_str(), m.pp_h,
                        m.pp_t, (long long)m.seg_ring_bytes);
            rad_pp_declare_segments(m.seg_b.data(), (int)m.seg_b.size());
        }
    }

    /* ---- the 97th connection: no write half, no fp8 copy. It produces the bf16 hidden state the
     * lm_head reads; its `hc_norm` takes the place of a final norm (the checkpoint has no
     * `model.language_model.norm.weight`). (Qwen tp8 S2: the leader's, with the gather and lm_head) */
    if (lr_shell(m)) {
    {
        HyperConn::Config mc = m.hccfg;
        mc.inject = 0;
        mc.quant  = 0;
        HyperConn::Wire mixw{};
        mixw.h = m.b_h; mixw.x.x = m.a_x.x;       /* q/s deliberately absent */
        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("model.language_model.hyper_connection_mixer.hc_norm.weight");
        mixs.mix_down = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_down.weight");
        mixs.mix_up   = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_up.weight");
        RAD_ARCH_TRY(m.mixer.declare(b, nm, g, nm.f("output_hc"), mc, mixw, mixs));
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {m.a_x.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_lm_head)),
                    {m.b_hout}, {m.b_logits});
    if (!m.op_embed || !m.op_gather || !m.op_logits) return RAD_E_INVAL;
    }   /* lr_shell */
    /* Qwen tp8 S1 / S2: the node's line (rank 0's real declare; a note in the graph) */
    if (m.lr.on && m.pp) {
        char line[1024];
        snprintf(line, sizeof line,
                 "Qwen tp8 S2: node %d of the two-node pipeline W(%d, %d) (strict; Qwen tp8 S3 / S3b-SP: every step form by part): layers %s, %d of "
                 "%lld; %s; crossings X1 / X2 of %d part(s), %lld B a row (the wide stream%s)",
                 m.pp_rank, m.pp_h, m.pp_t, m.lr.text.c_str(), m.lr.n_own, (long long)g.n_layer,
                 m.pp_rank == 0 ? "the leader: the embedding, the PLE layer, the final mixer, lm_head and the MTP head"
                                : "the follower: no embedding, mixer, head or drafter",
                 m.pp_parts, (long long)m.pp_row, "");
        rad_note(b, "%s", line);
        if (g.rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: qwen4exp: %s\n", line);
    } else if (m.lr.on) {
        int full = 0;
        for (int64_t l = 0; l < g.n_layer; ++l)
            if (lr_held(m, l)) full += m.layers[(size_t)l].full;
        char line[1024];
        snprintf(line, sizeof line,
                 "Qwen tp8 S1: layer ranges RADIANCE_FN4_LAYERS=%s (proxy: the stage alone): %d of %lld trunk layers (%d "
                 "attention, %d delta net, each with its routed experts), layers %d..%d%s; n-gram embedding %s; embedding, "
                 "final mixer, lm_head and the MTP head declared as the proxy's shell; its first layer reads the embedding's "
                 "stream, a gap passes the wide stream on; the crossing a row %lld B",
                 m.lr.text.c_str(), m.lr.n_own, (long long)g.n_layer, full, m.lr.n_own - full, m.lr.first, m.lr.last,
                 m.lr.leader ? " (holds the last layer)" : "", m.ple_layer >= 0 ? "held" : "not on this node",
                 (long long)rad::arch::qfnlayers::crossing_row_bytes(m.hccfg.hc, g.n_embd, false));
        rad_note(b, "%s", line);
        if (g.rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: qwen4exp: %s\n", line);
    }

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h).
     *
     * Declared after op_logits; do not move it. Declaration order is the planner's liveness, and a
     * drafter declared before the trunk's tail ops would share a trunk buffer with one of the
     * head's, giving correct output with zero acceptance.
     *
     * Declared only when the checkpoint carries a head and the deployment speculates; at
     * max_spec 0 the weights stay off the card (the container still carries them, since
     * rad-convert declares at a non-zero max_spec). */
    const int64_t nextn = meta_int2(meta, "qwen4exp.mtp_num_hidden_layers",
                                    "mtp_num_hidden_layers", 0);
    if (nextn > 0 && rad_ctx_draft_head(ctx) && lr_shell(m)) {   /* (Qwen tp8 S2: the leader's) */
        const char* mpre = "mtp";
        /* Sized at max_tok, not max_logit_rows: the history pass that fills the head's own K and V
         * has one row per token over a whole prefill chunk. */
        m.b_mtp_hin  = decl_b(b, nm.f("mtp_hin"),  g.act_dtype, {T, HN});
        m.b_mtp_hw   = decl_b(b, nm.f("mtp_hw"),   g.act_dtype, {T, HN});
        m.b_mtp_e    = decl_b(b, nm.f("mtp_e"),    g.act_dtype, {T, g.n_embd});
        m.b_mtp_hout = decl_b(b, nm.f("mtp_hout"), g.act_dtype, {g.max_logit_rows, g.n_embd});
        /* The 2-bit draft head, when the model holds one: asked once here, since the answer
         * decides the head's buffers and the block reads it back from them. */
        if (mtp_draft_head_q2(b, MtpHcBlock::head_name(nm), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,  {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32, {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32, {g.n_embd / 128, dmpad});
            /* This rank's rows, not the model's: the 2-bit head is vocab-parallel and the merge
             * reassembles it (rad_arch.h). */
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype,
                                  {g.max_logit_rows, vocab_head_rows(g)});
            if (vocab_head_sharded(g)) {
                m.b_mtp_dpair = decl_b(b, nm.f("mtp_dpair"), RAD_F32,
                                       {g.max_logit_rows, kDraftTopR, 2});
                m.b_mtp_dgath = decl_b(b, nm.f("mtp_dgath"), RAD_F32,
                                       {g.max_logit_rows, (int64_t)g.world * kDraftTopR, 2});
            }
            /* The proposal: a draft round's token comes from row_topk, never the sampler.
             * kDraftTopR columns, of which a draft reads column 0; the rest are the coarse head's
             * other candidates, which `logit_rerank` scores first, and the width also gives the
             * vocab-parallel all-gather a 16-byte row (rad_block_mtp_hc.h, kDraftTopR). */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32, {g.max_logit_rows, kDraftTopR});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32, {g.max_logit_rows, kDraftTopR});

            /* --draft-coupling (RadBuildCtx::draft_coupling). Q is the law of the 2-bit candidate
             * head's reranked top-8; a plain bf16 head leaves one candidate and nothing to couple
             * over. The engine checks the drafter shape (a linear serial chain). */
            if (ctx->draft_coupling) {
                if (!m.b_mtp_didx || g.max_spec < 1 || g.max_spec > RAD_MTP_COUPLE_MAX_ROUNDS) {
                    rad_note(b, "--draft-coupling on requires the 2-bit candidate head's reranked "
                                "top8 and max_spec 1..%d",
                             RAD_MTP_COUPLE_MAX_ROUNDS);
                    return RAD_E_UNSUPPORTED;
                }
                m.b_mtp_couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                              {g.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS},
                                              RAD_BUF_PERSIST);
                m.b_mtp_couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                              {g.max_spec, g.max_seqs,
                                               RAD_MTP_COUPLE_STATE_WORDS},
                                              RAD_BUF_PERSIST);
                if (!m.b_mtp_couple_draws || !m.b_mtp_couple_state) return RAD_E_INVAL;
                rad_note(b, "MTP coupled: the head samples x~Q from its reranked top8 each round "
                            "(FP64), and the sampler couples the target's own draw to it; %lld "
                            "round(s), fixed-seed tokens are NOT preserved",
                         (long long)g.max_spec);
            }
        }

        MtpHcBlock::Wire tw{};
        tw.src  = m.b_h;        tw.hin = m.b_mtp_hin;  tw.hw = m.b_mtp_hw;
        tw.e    = m.b_mtp_e;    tw.inj = m.b_inj;      tw.hout = m.b_mtp_hout;
        tw.logits = m.b_logits; tw.x = m.a_x;
        tw.dq   = m.b_mtp_dq;   tw.ds = m.b_mtp_ds;    tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.dpair = m.b_mtp_dpair; tw.dgath = m.b_mtp_dgath;
        tw.couple_draws = m.b_mtp_couple_draws; tw.couple_state = m.b_mtp_couple_state;
        /* The head reuses the trunk's working buffers, all dead by the time a draft round runs (a
         * second set would be 150 MiB). Only the wide stream and the carry are its own, since
         * they must survive the round that produced them. */
        tw.attn = aw; tw.attn.x = m.b_mtp_hw;
        tw.mlp  = mw; tw.mlp.x  = m.b_mtp_hw;
        /* The QSA wire is within-step scratch and a draft round is its own step, so the head
         * shares all of it. The QSA tail is per-layer state (`kv_qsa_tail`), not in the wire, and
         * declare() binds the head's layer to it like any other. */
        tw.qsa  = m.qsaw;

        MtpHcBlock::Src ts{};
        ts.fc_hidden          = nm.ckpt("%s.fc_hidden.weight", mpre);
        ts.fc_embedding       = nm.ckpt("%s.fc_embedding.weight", mpre);
        ts.pre_norm_hidden    = nm.ckpt("%s.pre_fc_norm_hidden.weight", mpre);
        ts.pre_norm_embedding = nm.ckpt("%s.pre_fc_norm_embedding.weight", mpre);

        ts.hc_attn.norm     = nm.ckpt("%s.layers.0.attn_hyper_connection.hc_norm.weight", mpre);
        ts.hc_attn.mix_down = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_attn.mix_up   = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_attn.inject_w = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        ts.hc_mlp.norm      = nm.ckpt("%s.layers.0.mlp_hyper_connection.hc_norm.weight", mpre);
        ts.hc_mlp.mix_down  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_mlp.mix_up    = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_mlp.inject_w  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        /* The head's own mixer has no `block_inject_weight`, like the trunk's 97th connection:
         * nothing follows it to write back into. */
        ts.hc_mix.norm      = nm.ckpt("%s.hyper_connection_mixer.hc_norm.weight", mpre);
        ts.hc_mix.mix_down  = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_down.weight",
                                      mpre);
        ts.hc_mix.mix_up    = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_up.weight", mpre);

        ts.attn.qg     = nm.ckpt("%s.layers.0.self_attn.q_proj.weight", mpre);
        ts.attn.k      = nm.ckpt("%s.layers.0.self_attn.k_proj.weight", mpre);
        ts.attn.v      = nm.ckpt("%s.layers.0.self_attn.v_proj.weight", mpre);
        ts.attn.q_norm = nm.ckpt("%s.layers.0.self_attn.q_norm.weight", mpre);
        ts.attn.k_norm = nm.ckpt("%s.layers.0.self_attn.k_norm.weight", mpre);
        ts.attn.o      = nm.ckpt("%s.layers.0.self_attn.o_proj.weight", mpre);

        ts.mlp.cfg             = m.moecfg;
        ts.mlp.router          = nm.ckpt("%s.layers.0.mlp.gate.weight", mpre);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "mtp", nm.ckpt("%s.layers.0.mlp.experts", mpre),
                                        &ts.mlp));
        if (m.moecfg.n_ff_shared > 0) {
            ts.mlp.shared      = nm.ckpt("%s.layers.0.mlp.shared_expert", mpre);
            ts.mlp.shared_gate = nm.ckpt("%s.layers.0.mlp.shared_expert_gate.weight", mpre);
        }

        /* Set before declare(), as for the trunk: `attn_paged` declares `sel` and `seqused` as
         * reads, which must precede the op. */
        m.mtp.attn.kv_block = 4;
        if (m.qsaw.qk) {
            m.mtp.attn.qsa_sel      = m.qsaw.sel;
            m.mtp.attn.qsa_nsel     = m.qsaw.nsel;
            m.mtp.attn.qsa_sequ     = m.qsaw.sequ;
            m.mtp.attn.qsa_topk     = m.qsacfg.topk;
            m.mtp.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            ts.qsa         = m.qsacfg;
            ts.qsa.mode    = m.mtp.attn.mode;
            ts.qsa.qk_proj = nm.ckpt("%s.layers.0.self_attn.indexer.index_qk_proj.weight", mpre);
            ts.qsa.q_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.q_layernorm.weight", mpre);
            ts.qsa.k_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.k_layernorm.weight", mpre);
            ts.qsa.kv_attn  = m.kv_full;
            ts.qsa.emit_sel = m.mtp.attn.qsa_sel != 0;
        }

        m.mtp.media = m.media;
        /* qfn-on-ds lever MSP (Qwen loop round 20; RADIANCE_FN4_MSP=1 declares, sp_route bit 1024 picks it): a prompt's
         * mid-chunk history pass of four read pieces runs the head's per-row work on this rank's piece, then gathers the
         * block input raw (SP-hc's all_gather_rows over a_x.x, the head's w.x) and redoes its codes (MtpHcBlock::step_msp) */
        if (fn4_env_on("RADIANCE_FN4_MSP") && g.world == 4 && m.sp_ag && (m.sp_on || m.spser_ok) && g.wire_exact &&
            m.hccfg.read_rows > 0 && (int64_t)g.world * m.hccfg.read_rows <= T) {
            m.mtp.msp_ag   = m.sp_ag;
            m.mtp.msp_rows = m.hccfg.read_rows;
        }
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)g.n_layer, m.kv_full, &m.kv_qsa_bkey, m.n_rot,
                                   m.w_tok, m.w_lm_head, m.hccfg, g.max_spec, tw, ts));
        if (fn4_env_on("RADIANCE_FN4_MSP") && g.rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: qwen4exp: lever MSP (RADIANCE_FN4_MSP; sp_route bit 1024): %s\n",
                    !m.mtp.msp_ag ? "NOT served -- needs four ranks, SP-hc / SP-serial served (RADIANCE_FN4_SPHC or _SPSERIAL), "
                                    "the exact wire, a read piece (--prefill-span) and the connection's codes op"
                    : m.mtp.msp_rs ? "a prompt's mid-chunk MTP history pass of four read pieces runs its gather, mtp_enter and "
                                     "connection read on this rank's piece, then gathers the block input raw and redoes its codes "
                                     "(bit 2048: its embedding by a raw reduce-scatter onto the piece)"
                                   : "a prompt's mid-chunk MTP history pass of four read pieces runs its gather, mtp_enter and "
                                     "connection read on this rank's piece, then gathers the block input raw and redoes its codes "
                                     "(bit 2048 not served: the embedding's reduce-scatter did not resolve)");
        /* A head pass refills the rope table ahead of the head's ops (step), while the carry
         * `mtp_hin` still holds the rows the previous pass left. The refill is the trunk's
         * `rope_table` op, so the table's planned life would end at its last declared reader,
         * which is in the trunk when no head op reads the table (no fused prologue), and the
         * planner could then put the carry in the table's bytes. Declared as written by the
         * head's first op, the table is live where the carry is. */
        if (m.op_rope_cs && rad_op_writes(b, m.mtp.op_gather, &m.b_rope_cs, 1) < 0)
            return RAD_E_INVAL;
        m.have_mtp = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal.
         * The 2-bit head writes its own argmax into `mtp_didx`; without it `mtp_didx` is 0 and the
         * bf16 head goes through the ordinary sampler chain. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = kDraftTopR;   /* the pick is column 0; see kDraftTopR */
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        rad_note(b, "MTP head: one gated-attention layer with 512 routed experts over the wide "
                    "stream, %d serial draft round(s)", ctx->max_spec);
    } else if (nextn > 0) {
        rad_note(b, "the checkpoint's `mtp.` head is not loaded: this deployment does not "
                    "speculate");
    }

    /* The vision tower, last: an encoder pass issues nothing else, so its activations share the
     * arena with every buffer above. Its weight groups start past every text layer and the MTP
     * head. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    /* fn4 BI-dual (RADIANCE_FN4_X3DUAL=1): the step's row bands, for the EXL3 experts' downs */
    if (fn4_env_on("RADIANCE_FN4_X3DUAL") && m.moecfg.expert_exl3) {
        m.b_band = decl_b(b, nm.f("x3_band"), RAD_I32, {g.max_tok}, RAD_BUF_PERSIST);
        m.op_band = m.b_band ? rw(b, RAD_OP(b, "bi_band",
                                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("band", 16)),
                                            RAD_NOWEIGHTS),
                                  {}, {m.b_band})
                             : 0;
        if (!m.op_band) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_FN4_X3DUAL: bi_band did not resolve (a libr4d with r4d_fn4_bi?)\n");
            return RAD_E_UNSUPPORTED;
        }
        if (!ctx->shape_probe) g_dual_declared.store(1, std::memory_order_relaxed);
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: qwen4exp: fn4 BI-dual declared (RADIANCE_FN4_X3DUAL, switch exl3_dual now %d): the "
                            "EXL3 downs take the step's row bands (sequences of at most 16 rows: the decode band's sums)\n",
                    fn4_dual_flag().load(std::memory_order_relaxed));
    }
    /* fn4 BI's tracer, after everything (RADIANCE_FN4_BITRACE; unset: nothing declared). */
    RAD_ARCH_TRY(bt_declare(b, m, ctx));
    return RAD_OK;
}

/* ============================== A prefill step wider than one MoE pass ==============================
 *
 * Every layer of a prefill chunk makes two all-reduces, the mixer's and the MoE's, each in a hard
 * chain: mixer -> all-reduce -> write -> read -> MoE -> all-reduce -> write -> read. At four ranks
 * a chunk's 97 tiered-int8 two-shots are about a quarter of its time, with nothing beside them on
 * one stream.
 *
 * Under --prefill-span the scheduler hands a prefill alone in its step several checkpoint
 * intervals, each piece the interval-sized chunk a span of 1 makes. Everything a piece can run by
 * itself runs a piece at a time, and every all-reduce goes out on the second lane, beside the
 * other piece's work on the first:
 *
 *   the mixer       the delta net runs whole a piece at a time (projection, convolution, solve and
 *                   scan over the piece's bounds, the state handed on through the cache as between
 *                   two chunks), and the attention runs its gated sparse path a piece at a time;
 *                   each hands its first piece's all-reduce off behind that piece's out projection
 *                   (ArPieces), so it runs beside the second piece's mixer;
 *   the MoE         a pass a piece: the mixer's all-reduce of piece 1 runs beside piece 0's write,
 *                   read and pass, the MoE's of piece 0 beside piece 1's;
 *   the last one    the MoE's all-reduce of piece 1 runs beside piece 0's write and the next
 *                   layer's read.
 *
 * The indexer's selection and the embedding run over the whole step.
 *
 * A piece computes the bytes the narrower chunk does: every all-reduce message has a piece's rows,
 * so the tiered-int8 shards and their owners are the ones a span of 1 sends; every connection read
 * is a piece, so its down product splits K the same way (HyperConn::read_pieces); every MoE pass
 * and delta-net piece is a chunk's, and attention rows are their own. The first interval of a
 * prompt is never spanned (Scheduler::plan_chunk), so the dense attention below QSA's exactness
 * bound is taken where it was.
 *
 * The second lane's requirements (Ctx::lane1_) hold: every collective of the step is on it,
 * including the embedding's; none carries an arrival counter; and the one buffer it touches is out
 * of the sharing pool (declare). The collectives there are declared at `overlap` 1, whose rows
 * leave the CUs to the work beside them. */
/* Qwen tp8 S3a (Qwen loop round 30, WIP): this pass's part of the two-node pipeline (abi/rad_pp.h, read at issue as
 * llama_fp8 :1007-1013): the leader issues its HEAD ([0, h) up to and including X1's send), its TAIL (from X2's receive:
 * [t, n) + the mixer and the head) or both (one lane: the whole pass); the follower always its whole pass (X1's receive,
 * [h, t), X2's send). Off (one machine, a proxy): head and tail. */
struct PpPass {
    bool on = false;
    bool head = true, tail = true;
    RadPpInfo info{};
    /* Qwen tp8 S6: the follower's layers this part issues, [s0, s1) of its [h, t): the whole pass, or a segment the core
     * asked for (RAD_PP_PART_SEG) -- X1's receive only where s0 = h, else boundary s0's restore; X2's send only where s1 = t,
     * else boundary s1's save */
    int  s0 = 0, s1 = 0;
    bool seg_in = false, seg_out = false;
};
static PpPass pp_pass(const Model& m) {
    PpPass p;
    if (!m.pp) return p;
    p.on = true;
    p.info = rad_pp_query();
    const int part = m.pp_rank == 0 ? p.info.part : RAD_PP_PART_WHOLE;
    p.head = part != RAD_PP_PART_TAIL;
    p.tail = part != RAD_PP_PART_HEAD;
    p.s0 = m.pp_h;
    p.s1 = m.pp_t;
    if (m.pp_rank == 1 && p.info.part == RAD_PP_PART_SEG && !m.seg_b.empty()) {
        p.s0 = p.info.seg_l0;
        p.s1 = p.info.seg_l1;
        p.seg_in = p.s0 != m.pp_h;
        p.seg_out = p.s1 != m.pp_t;
    }
    return p;
}
/* Qwen tp8 S6: a part's layers are a segment this node declared: [h, t) or bounded by its boundaries */
static bool pp_seg_ok(const Model& m, const PpPass& pp) {
    auto is_b = [&](int s) { return std::find(m.seg_b.begin(), m.seg_b.end(), (int32_t)s) != m.seg_b.end(); };
    return pp.s0 < pp.s1 && pp.s0 >= m.pp_h && pp.s1 <= m.pp_t && (pp.s0 == m.pp_h || is_b(pp.s0)) &&
           (pp.s1 == m.pp_t || is_b(pp.s1));
}
/* Qwen tp8 S6: boundary s's save (after layer s - 1's writes) or restore (before layer s's read): this rank's wide stream,
 * every row of the step as this rank holds it, into / out of its own ring buffer (the stage ops' loop mode) */
static bool pp_seg_io(RadCtx* c, const Model& m, int64_t s, bool save, int64_t T) {
    for (size_t k = 0; k < m.seg_b.size(); ++k) {
        if (m.seg_b[k] != (int32_t)s) continue;
        const rad_op op = save ? m.seg_save[k] : m.seg_load[k];
        if (!op || !m.seg_ring) break;
        RAD_ISSUE(c, op, brows(m.seg_ring, m.seg_ring_bytes), brows(m.b_h, T));
        return true;
    }
    rad_step_fail(c, "qwen4exp: two-node pipeline (Qwen tp8 S6): a segment boundary this node did not declare");
    return false;
}
/* One crossing's stage op over the wide stream's T rows (the receiver's read recomputes the hc_ss scales: same bits) */
static bool pp_cross(RadCtx* c, const Model& m, const PpPass& pp, rad_op op, int ring, int64_t T) {
    RadPpRing* rg = ring >= 0 && ring < RAD_PP_N_RINGS ? pp.info.ring[ring] : nullptr;
    if (!rg || !op) {
        rad_step_fail(c, "qwen4exp: two-node pipeline: the core set up no ring for this crossing (or the stage op is not declared)");
        return false;
    }
    /* Qwen tp8 S7b: the head layers' state rows into the chunk-end snapshot, before X1 leaves (every head layer's writes joined:
     * each form joins lane 1 before its X1) -- a failed copy is counted by the engine, which refuses that snapshot's save */
    if (ring == RAD_PP_X1 && m.pp_rank == 0 && pp.head) {
        const HeadSnapRank& hs = g_hsr[rad_rank(c)];
        if (hs.stage) {
            const RadStream st = rad_stream(c);
            for (int64_t l = 0; l < m.pp_h && l < (int64_t)m.layers.size(); ++l)
                if (!m.layers[(size_t)l].full) {
                    (void)hs.stage(hs.u, (int)m.kv_state, (int)l, st);
                    (void)hs.stage(hs.u, (int)m.kv_conv, (int)l, st);
                }
            if (m.ple_layer >= 0 && m.ple_layer < m.pp_h) {
                (void)hs.stage(hs.u, (int)m.ple.kv_tok, (int)m.ple_layer, st);
                (void)hs.stage(hs.u, (int)m.ple.kv_conv, (int)m.ple_layer, st);
            }
        }
    }
    RAD_ISSUE(c, op, praw(rg, RAD_U8, 1), brows(m.b_h, T));
    return true;
}

static bool pipelined(const Model& m, int64_t T) {
    const int64_t R = m.pass_rows;
    if (!m.pipe_on || R <= 0 || T <= R) return false;
    /* (Qwen tp8 S3b: a pipeline follower has no embedding, so no embedding all-reduce: its step starts at X1's receive) */
    if (m.moecfg.calib || dbg_resid_on() || (!m.op_embed_ar && !(m.pp && m.pp_rank == 1))) return false;
    for (const Layer& l : m.layers) {
        if (!lr_held(m, &l - m.layers.data())) continue;   /* Qwen tp8 S1: another node's layer */
        if (!(l.full ? l.attn.issues_ar(T) : l.gdn.issues_ar(T))) return false;
        if (!l.mlp.issues_ar(T, R)) return false;
    }
    return true;
}

/* The mixer's reduction of a piece the block has finished (ArPieces), on the second lane behind
 * the out projection that wrote it. */
struct MixAr {
    RadCtx*      c = nullptr;
    const Layer* l = nullptr;
    /* fn4 SP-hc: the reduce-scatter in place of the all-reduce for pieces of at least sp_min rows (0: off) */
    rad_op       rs = 0;
    rad_buf      x = 0;
    int64_t      n_embd = 0, sp_min = 0;
    /* fn4 PF marks (qwen/prefill; sp_route bit 8): record lane-1 mark (r0 / rows) behind each handed piece's reduction,
     * and the bit in *mk, so the step waits for that piece alone (Ctx::lane_join 0x100 / 0x200) */
    uint32_t*    mk = nullptr;
    int64_t      rows = 0;
};
static void mix_ar_piece(void* u, int64_t r0, int64_t n) {
    const MixAr& s = *(const MixAr*)u;
    rad_lane(s.c, 1);
    rad_lane_join(s.c, 0, 1);
    if (s.rs && n >= s.sp_min) RAD_ISSUE_N(s.c, s.rs, n * s.n_embd, brow_slice(s.x, r0, n, s.n_embd), RAD_NONE);
    else if (s.l->full) s.l->attn.issue_ar(s.c, r0, n);
    else           s.l->gdn.issue_ar(s.c, r0, n);
    if (s.mk && s.rows > 0 && r0 / s.rows < 8) {
        const int k = (int)(r0 / s.rows);
        if (rad_lane_join(s.c, 1, 0x100 + k) == RAD_OK) *s.mk |= 1u << k;
    }
    rad_lane(s.c, 0);
}

/* The bounds a step wider than a pass is cut by, on the card: {0, pass, T}. The first two are the
 * window of the indexer's score plane (QsaIndexer::step, Config::plane_rows) and the delta net's
 * first piece (ArPieces::cu), copied from the pinned host copy; T is copied on the card from the
 * step's cu_seqlens. Both are issued on the step's stream ahead of every layer by each step of one
 * sequence wider than a pass, since the buffer is a transient (declare). Null where no step can be
 * wider than a pass (no b_pipe_cu) or the pinned copy could not be allocated. */
static const int32_t* pipe_bounds(RadCtx* c, const RadBatch* batch, Model& m) {
    if (!m.b_pipe_cu) return nullptr;
    if (!m.pipe_cu_tried) {
        m.pipe_cu_tried = true;
        m.pipe_cu_h = (int32_t*)rad_dev_alloc(4 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        if (m.pipe_cu_h) {
            m.pipe_cu_h[0] = 0;
            m.pipe_cu_h[1] = (int32_t)m.pass_rows;
            m.pipe_cu_h[2] = 0;                                   /* lever QRS: a rank's quarter window {0, q} */
            m.pipe_cu_h[3] = (int32_t)m.qsacfg.qrs_rows;
        } else if (m.g.rank == 0) {
            fprintf(stderr, "radiance: qwen4exp: no pinned host memory for the window bounds; a "
                            "step wider than a pass cannot run\n");
        }
    }
    if (!m.pipe_cu_h) return nullptr;
    int32_t* d = (int32_t*)rad_buf_ptr(c, m.b_pipe_cu);
    if (batch->n_tok > m.pass_rows && batch->n_seq == 1) {
        const RadStream s = rad_stream(c);
        if (rad_memcpy_async(d, m.pipe_cu_h, 2 * sizeof(int32_t), s) != RAD_OK ||
            rad_memcpy_async(d + 2, batch->cu_seqlens + 1, sizeof(int32_t), s) != RAD_OK)
            return nullptr;
        if (m.qsacfg.qrs_rows > 0 && rad_memcpy_async(d + 4, m.pipe_cu_h + 2, 2 * sizeof(int32_t), s) != RAD_OK)
            return nullptr;   /* lever QRS's {0, q} (declared {6} under RADIANCE_FN4_QRS) */
    }
    return d;
}

/* fn4 P2x (declared with RADIANCE_FN4_P2X=1: the expert buffers two pieces tall): switch "moe_pair" 0 / 1 / 2 puts a
 * pipelined prefill step's two pieces through ONE routed-expert pass (MoeFP8::pair_routed) -- 1 with each piece's
 * shared arm after it (beside piece 0's MoE all-reduce), 2 with the shared arms inside it (pass()'s interleave);
 * RADIANCE_FN4_MOE_PAIR=<n> starts it at n.
 * qfn-on-ds lever P2S (Qwen loop round 17, QUEUE Q38): under hc_sp 2 (every piece SP) a step of an EVEN number of full pieces
 * runs its pieces' routed experts in PAIRS ((0, 1), (2, 3)): 1 / 2 the joint pass (as above; RADIANCE_FN4_P2X=1 declares the
 * MoE two pieces tall), 3 = the pair ORDER with each piece's own pass (exact by construction, no P2X needed: the order's own
 * cost, the arm that splits the joint form's net into its order and its kernel saving). See step_pipelined's P2S branch. */
static std::atomic<int>& fn4_pair_flag() {
    static std::atomic<int> f{[] {
        const char* v = std::getenv("RADIANCE_FN4_MOE_PAIR");
        const int n = v && *v ? std::atoi(v) : 0;
        return n >= 0 && n <= 4 ? n : 0;
    }()};
    return f;
}
/* fn4 SP-hc SPEED PROBE (10.11.2026, measurement only -- WRONG OUTPUT): the sequence-parallel gated residual's compute
 * without its collectives' new form. With the flag on, every hc write / read between two blocks of a pipelined prefill
 * step runs on this rank's quarter of the piece's rows only (rows [r0 + rank q, r0 + (rank + 1) q), q = ceil(rn /
 * world)), the all-reduces stay as they are (a two-shot = the reduce-scatter + all-gather sequence-parallel would
 * issue, the same bytes), so a step's critical path is SP's -- reduce, the hc ops on T / world rows, gather -- less the
 * gather's re-quantisation; the other ranks' rows of the block input stay stale. Decode and serial steps unchanged.
 * Core switch "hc_spprobe" -> this plugin's rad_fn4_switch; RADIANCE_FN4_SPPROBE=1 at start. */
static std::atomic<int>& fn4_spprobe_flag() {
    static std::atomic<int> f{fn4_env_on("RADIANCE_FN4_SPPROBE") ? 1 : 0};
    return f;
}
/* fn4 SP-hc proper (declared with RADIANCE_FN4_SPHC=1, Model::sp_on): switch "hc_sp" 0 / 1 picks it for the pipelined
 * prefill steps; RADIANCE_FN4_HC_SP=1 starts it on. Pieces under kSpMinRows rows keep the all-reduce form. */
static std::atomic<int>& fn4_sp_flag() {
    static std::atomic<int> f{fn4_env_on("RADIANCE_FN4_HC_SP") ? 1 : 0};
    return f;
}
/* fn4 SP-PLE (16.11.2026, qwen/SP-SERIAL-EXACT.md; switch "sp_ple", RADIANCE_FN4_SP_PLE at start; 0 = today): in a pipelined
 * SP-hc step the layers before the PLE's (Flash-Next: blk.0) run replicated, as step_serial_sp runs them. Today SP-hc
 * splits blk.0 too, so when the PLE runs (over every row: its gate reads the wide stream, its dilated convolution reads the
 * nine rows before each one) a rank holds blk.0's update only on its own quarters: every other row of `h` is still the
 * step's entry stream. Two things then take those stale rows: the convolution of the first nine rows of each quarter
 * (their taps reach into the previous rank's quarter), and the window the step leaves for the next one (the chunk's
 * last nine rows' conv input, which only the rank owning them computes from the real stream -- every other rank keeps a
 * different window, and the next step's first nine rows read it). 1: blk.0 with its all-reduces and whole-row reads /
 * writes, so the PLE sees the whole stream on every rank; SP from blk.1's read on. */
static std::atomic<int>& fn4_spple_flag() {
    static std::atomic<int> f{fn4_env_on("RADIANCE_FN4_SP_PLE") ? 1 : 0};
    return f;
}
/* fn4 SP-route (16.11.2026, qwen/prefill; switch "sp_route", RADIANCE_FN4_SP_ROUTE at start declares it and gives its value;
 * 0 = today): in a pipelined hc_sp 2 step whose pieces all take SP, each rank runs the router GEMM on its own quarter of a
 * piece right after the quarter's connection read (MoeFP8::route_rows), the logits rows go to every rank with a raw
 * all-gather on the second lane beside it (sp_agl, after the piece's block-input gather), and the piece's MoE pass skips its
 * GEMM (pass(..., routed)): the router's 2048-row GEMM, replicated on all four ranks today, becomes one 512-row GEMM a rank.
 * The logits bytes are the same (a row's logits do not depend on the launch's other rows), so the top-k, the scatter and
 * everything after them are today's. */
static std::atomic<int>& fn4_sproute_flag() {
    static std::atomic<int> f{fn4_env_on("RADIANCE_FN4_SP_ROUTE") ? 1 : 0};
    return f;
}
constexpr int64_t kSpMinRows = 64;
/* fn4 KVCAP C1: an sp_agh launch of `rows` rows fits the bound its peer scratch was declared for (Model::sp_agh_rows);
 * past it the step fails with the reason, before the launch (the core would refuse the count as outside the op's bucket
 * table, and the AR entry as larger than the instance's max_numel). */
static bool sp_agh_fits(RadCtx* c, const Model& m, int64_t rows) {
    if (rows <= m.sp_agh_rows) return true;
    char why[384];
    snprintf(why, sizeof why,
             "qwen4exp: fn4 KVCAP C1: the wide stream's all-gather (sp_agh) was asked for %lld rows, past the %lld rows "
             "its peer scratch is declared for (min(max_tok, max(pass rows %lld, SP-serial segment %lld))): a step shape "
             "the bound does not cover",
             (long long)rows, (long long)m.sp_agh_rows, (long long)m.pass_rows, 128LL * HyperConn::kSpTiles);
    rad_step_fail(c, why);
    return false;
}

static void step_pipelined(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu, const PpPass& pp = PpPass{}) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, R = m.pass_rows;
    const int64_t K = (T + R - 1) / R;
    auto r0 = [&](int64_t p) { return p * R; };
    auto rn = [&](int64_t p) { return T - p * R < R ? T - p * R : R; };
    /* fn4 P2x: the two pieces' routed experts in one pass (moe_pair 1 / 2), where the MoE was declared for it */
    const int pmode = fn4_pair_flag().load(std::memory_order_relaxed);
    /* qfn-on-ds P2S: under hc_sp 2 the pairs ride the SP order (pair_sp, the branch below); the K = 2 form above keeps the
     * stock order (SP off) and takes only modes 1 / 2 when hc_sp is not 2 */
    /* qfn-on-ds lever PSP (sp_route bit 512, below): also an odd piece count -- pairs (0, 1), (2, 3), .. and the last piece
     * single (the loop's paired(); moe_pair 4's tail runs the same mix of joint and single passes) */
    const bool psp_k = (fn4_sproute_flag().load(std::memory_order_relaxed) & 512) != 0;
    /* (Qwen tp8 S3b-SP, round 31: a pipeline node runs the SP forms too; a crossing gathers the wide stream's quarters first,
     * since stage_send launches on ONE rank and carries whole rows -- the cut below and the step's end) */
    const bool pair_sp = pmode > 0 && m.sp_on && g.world == 4 && fn4_sp_flag().load(std::memory_order_relaxed) == 2 &&
                         K >= 2 && (K % 2 == 0 || psp_k) && (pmode == 3 || m.moecfg.rows >= 2 * R);
    const bool pair_on = pmode > 0 && pmode < 3 && !pair_sp && K == 2 && m.moecfg.rows >= T;
    /* fn4 SP-hc: sp, the real sequence-parallel residual (reduce-scatter, the connections on this rank's quarter, the
     * all-gather and the codes redone); spp, the speed probe (the quarter only, all-reduces kept, wrong output). A
     * piece takes SP when it has kSpMinRows rows; its quarter is the reduce-scatter's shard: q = ceil(rn / world),
     * rows [r0 + min(rank q, rn), r0 + min((rank + 1) q, rn)). */
    const int  spv = fn4_sp_flag().load(std::memory_order_relaxed);
    const bool sp  = m.sp_on && g.world == 4 && !pair_on && spv > 0;
    /* hc_sp 2 (12.11.2026): SP's issue order software-pipelined -- piece p + 1's connections beside piece p's all-gather and
     * piece p + 1's all-gather beside piece p's codes and MoE (hc_sp 1 waits on every gather right after issuing it: three
     * of four a layer on lane 0's path at two pieces). The same ops on the same rows: only the order of issue moves. */
    const bool spo = sp && spv == 2;
    const bool spp = !sp && !pp.on && g.world > 1 && fn4_spprobe_flag().load(std::memory_order_relaxed);
    /* fn4 PF marks (16.11.2026, qwen/prefill; sp_route bit 8): in the hc_sp 2 order, lane 0 waits for the one collective
     * an op reads (piece p's mixer reduction before its connection write; piece p's gathers before its codes) instead of
     * everything the second lane holds. The same ops on the same rows; only where lane 0 waits moves. */
    const bool pmk = (fn4_sproute_flag().load(std::memory_order_relaxed) & 8) != 0;
    /* fn4 SPO-tail (sp_route bit 16): the hc_sp 2 order also where only the last piece is partial */
    const bool spo_tail = spo && (fn4_sproute_flag().load(std::memory_order_relaxed) & 16) != 0;
    /* fn4 PF-ORDER (sp_route bit 32): see the layer loop */
    const bool pford = (fn4_sproute_flag().load(std::memory_order_relaxed) & 32) != 0;
    /* PF probe NOWAIT (qfn-on-ds, Qwen loop round 13, QUEUE Q9's upper-bound read; sp_route bit 256; WRONG BYTES, timed
     * only): inside this step lane 0 never waits for lane 1 (wait1 / waitm issue no join); lane 1 still orders every
     * collective behind the lane-0 mark it reads (on1's join 0 -> 1, the handed pieces' mix_ar_piece), and the step ends
     * with one join 1 -> 0. So a step takes max(lane 0's own chain, lane 1's chain behind it): what lane 0 costs if its
     * waits on the collectives were filled -- the ceiling of a second prompt's pieces in the step (Q9) and of every form
     * that hides the comm lane's waits. Lane 0 reads x before its reduction lands (garbage rows, finite); a start that
     * takes it reads timing only (no identity cell, nonce prompts). */
    const bool nowait = (fn4_sproute_flag().load(std::memory_order_relaxed) & 256) != 0;
    /* qfn-on-ds lever PSP (Qwen loop round 19, sp_route bit 512; QUEUE Q39): the step's PARTIAL last piece takes SP too, so the
     * hc_sp 2 order (and P2S's pairs: a full piece with the partial one in one routed pass) covers every prompt's last cold
     * chunk (the prefill timeline, R42: a chunk with a partial piece 109-121 ms a 1k rows against 95-96 for span chunks). Its
     * quarter's reads go through the SP-serial twins (HyperConn::read_kt, the K split pinned to the PIECE's own row tiles), which
     * is what reading the partial piece whole gives (spser_run's rule: every rank's part more than 64 rows, a twin for the
     * piece's tiles); the other ops are per row. So the same bytes as today; off = today. */
    const bool psp = psp_k;
    /* fn4 SP-hc on the exact wire: only a FULL piece (rn == R) takes SP, its quarter's reads through op_read_sp (the K split
     * pinned to R, byte-identical to the piece read whole); a shorter last piece reads whole as today. */
    /* fn4 SP-PLE (switch sp_ple): the layers before the PLE's run without SP (lsp), so the PLE reads the whole stream on
     * every rank. spL: the layer being issued; a connection read for the next layer takes that layer's (spN below). */
    const int64_t spl0 = sp && fn4_spple_flag().load(std::memory_order_relaxed) > 0 && m.ple_layer > 0 ? m.ple_layer : 0;
    auto lsp = [&](int64_t li) { return sp && li >= spl0; };
    bool spL = sp;
    auto psp_ok = [&](int64_t p) {   /* lever PSP: the partial last piece's SP, where its parts read as the piece whole */
        if (!psp || p != K - 1 || p == 0 || rn(p) >= R || rn(p) % g.world != 0 || rn(p) / g.world <= 64) return false;
        if (m.layers.empty()) return false;
        /* every connection declares the same twins (Qwen tp8 S1: the first and last layer this node holds) */
        for (const Layer* l : { &m.layers[(size_t)m.lr_first], &m.layers[(size_t)m.lr_last] })
            if (!l->hc_mix.read_kt_op(rn(p)) || !l->hc_ffn.read_kt_op(rn(p))) return false;
        return true;
    };
    auto sp_pm = [&](int64_t p, bool on) { return on && rn(p) >= kSpMinRows && (rn(p) == R || psp_ok(p)); };
    auto sp_p = [&](int64_t p) { return sp_pm(p, spL); };
    auto hq0m = [&](int64_t p, bool on) {
        if (!(spp || sp_pm(p, on))) return r0(p);
        const int64_t q = (rn(p) + g.world - 1) / g.world, lo = (int64_t)g.rank * q;
        return r0(p) + (lo < rn(p) ? lo : rn(p) - 1);
    };
    auto hqnm = [&](int64_t p, bool on) {
        if (!(spp || sp_pm(p, on))) return rn(p);
        const int64_t q = (rn(p) + g.world - 1) / g.world, lo = (int64_t)g.rank * q;
        const int64_t n = rn(p) - lo < q ? rn(p) - lo : q;
        return n > 0 ? n : (int64_t)0;
    };
    auto hq0 = [&](int64_t p) { return hq0m(p, spL); };
    auto hqn = [&](int64_t p) { return hqnm(p, spL); };
    /* a connection's write / read on piece p's rows of this rank (none when the quarter is empty: SP only) */
    auto hw = [&](const HyperConn& h, int64_t p) { return hqn(p) > 0 ? h.write(c, T, hq0(p), hqn(p)) : false; };
    auto hrm = [&](const HyperConn& h, int64_t p, bool ss, bool on) {
        if (hqnm(p, on) <= 0) return;
        /* lever HCP: a partial piece's rows (a rank's part under PSP, or the piece whole) at the full piece's split */
        if (sp_pm(p, on) && rn(p) != R) {
            if (hcp_pins(m, h, r0(p), hqnm(p, on))) { h.read_sp(c, T, hq0m(p, on), hqnm(p, on), ss); ++m.hcp_n; }
            else h.read_kt(c, hq0m(p, on), hqnm(p, on), rn(p), ss);   /* lever PSP: the piece's own split */
        }
        else if (sp_pm(p, on) && h.op_read_sp) h.read_sp(c, T, hq0m(p, on), hqnm(p, on), ss);
        else hcp_read(c, m, h, T, hq0m(p, on), hqnm(p, on), ss);
    };
    auto hr = [&](const HyperConn& h, int64_t p, bool ss) { hrm(h, p, ss, spL); };
    /* SP's collectives on piece p of a_x (the block output; the next block input) and of the wide stream */
    auto rs_p  = [&](int64_t p) { RAD_ISSUE_N(c, m.sp_rs, rn(p) * g.n_embd, brow_slice(m.a_x.x, r0(p), rn(p), g.n_embd), RAD_NONE); };
    auto ag_p  = [&](int64_t p) { RAD_ISSUE_N(c, m.sp_ag, rn(p) * g.n_embd, brow_slice(m.a_x.x, r0(p), rn(p), g.n_embd), RAD_NONE); };
    auto agh_p = [&](int64_t p) {
        const int64_t HN = m.hccfg.hc * g.n_embd;
        if (!sp_agh_fits(c, m, rn(p))) return;   /* fn4 KVCAP C1 */
        RAD_ISSUE_N(c, m.sp_agh, rn(p) * HN, brow_slice(m.b_h, r0(p), rn(p), HN), RAD_NONE);
    };
    /* A collective on the second lane, ordered behind lane 0 where it reads what lane 0 just
     * wrote (a join to a lane not yet built is a no-op). */
    auto on1 = [&](bool behind0, auto&& issue) {
        rad_lane(c, 1);
        if (behind0) rad_lane_join(c, 0, 1);
        issue();
        rad_lane(c, 0);
    };
    auto wait1 = [&] { if (!nowait) rad_lane_join(c, 1, 0); };
    if (psp && !m.psp_noted && g.rank == 0 && K >= 2 && rn(K - 1) < R) {
        m.psp_noted = true;
        fprintf(stderr, "radiance: qwen4exp: lever PSP (sp_route bit 512): a %lld-row step's partial last piece of %lld rows %s\n",
                (long long)T, (long long)rn(K - 1),
                psp_ok(K - 1) ? "takes SP (its quarter's reads pinned to the piece's own K split, the SP-serial twins), so the "
                                "hc_sp 2 order and P2S's pairs cover it"
                              : "stays whole (a part of 64 rows or fewer, a row count the four ranks do not share evenly, or no "
                                "SP-serial twin for its tiles: RADIANCE_FN4_SPSERIAL=1 at start declares them)");
    }
    if (nowait && !m.nowait_noted && g.rank == 0) {
        m.nowait_noted = true;
        fprintf(stderr, "radiance: qwen4exp: PF probe NOWAIT (sp_route bit 256): a pipelined prefill step's lane 0 does not wait "
                        "for lane 1 (one join at the step's end); WRONG BYTES, timing only\n");
    }
    if (!m.pipe_noted) {
        m.pipe_noted = true;
        if (g.rank == 0)
            fprintf(stderr, "radiance: qwen4exp: prefill steps wider than a MoE pass are pipelined: "
                            "%lld rows in %lld pieces of %lld, all-reduces on the second lane\n",
                    (long long)T, (long long)K, (long long)R);
    }

    /* qfn-on-ds lever TIP: the tail snapshot inside this step, at the start of its last piece (g_tipr) */
    TipStage tip{};
    if (!tip_stage_for(c, batch, m, &tip)) return;
    const TipStage* tipp = tip.on() ? &tip : nullptr;

    /* Qwen tp8 S3b: the embedding .. the stream's entry are the leader's head; the follower starts at X1's receive */
    const bool lead_tail_only = pp.on && m.pp_rank == 0 && !pp.head;
    if (!pp.on || (m.pp_rank == 0 && pp.head)) {
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    on1(true, [&] { RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE); });
    wait1();
    bt_pt(c, m, m.a_x.x, 0, T, "emb", -1);
    if (m.ple_layer >= 0) m.ple.ids(c, batch, tipp);
    m.enter.step(c, T);
    bt_pt(c, m, m.b_h, 0, T, "enter", -1);
    } else if (m.pp_rank == 1 && !(pp.seg_in ? pp_seg_io(c, m, pp.s0, false, T)      /* Qwen tp8 S6: boundary s0's restore */
                                             : pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T))) {
        return;
    }
    if (m.op_rope_cs && T <= qk_fuse_now(qk_fuse_rows(g)) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    /* The delta net runs a piece at a time over the bounds (ArPieces::cu) in a step of one
     * prefill sequence in two pieces. */
    const int32_t* gcu = pcu && K == 2 && batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL
                       ? pcu : nullptr;
    /* qfn-on-ds lever GPF (Qwen loop round 22, sp_route bit 4096; test tip5's sibling hits): that two-piece path projects
     * each piece alone, so a second piece of <= 64 rows (libr4d's narrow band) takes gemm_nt_q m16's split-K arithmetic for
     * the delta net's in-projection and gemm_nt m16 for its a|b, where a step of three or more pieces (the cold run of the
     * same rows) projects them inside its T-row tiled launch: a hit resuming one piece before a short last piece answered
     * otherwise than its cold. On: the piece path only where the second piece is wider than the band (ArPieces::min_last);
     * a (R, <= 64) step runs the delta net whole, as K >= 3 does. Changes the bytes of such steps only (T in (R, R + 64]). */
    const bool gpf = (fn4_sproute_flag().load(std::memory_order_relaxed) & 4096) != 0;
    if (gpf && gcu && T - R <= kBiNarrowRows && !m.gpf_noted && g.rank == 0) {
        m.gpf_noted = true;
        fprintf(stderr, "radiance: qwen4exp: lever GPF (sp_route bit 4096): a %lld-row step's last piece of %lld rows is in the "
                        "narrow band, so its delta net runs whole (the projections over all %lld rows, as a step of three "
                        "pieces), not a piece at a time\n",
                (long long)T, (long long)(T - R), (long long)T);
    }

    const int64_t n = (int64_t)m.layers.size();
    /* Qwen tp8 S1: the first layer this node holds (0 without ranges); S3b: a leader's tail part starts at X2's receive + t;
     * S6: a follower's segment at s0 */
    const bool fol = pp.on && m.pp_rank == 1;
    const int64_t l0 = lead_tail_only ? (int64_t)m.pp_t : fol ? (int64_t)pp.s0 : (int64_t)m.lr_first;
    if (lead_tail_only && !pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T)) return;
    if (n > 0 && pp.seg_in) {
        /* Qwen tp8 S6: a segment from boundary s0 -- each rank's wide stream as its save left it (an SP piece's quarter rows
         * this rank wrote, the rest as they were), read as the pass reads a layer after a write that hands it no scales (the
         * PLE's form below: every piece's read on this rank's rows, an SP piece's block input gathered and coded): the same
         * ops on the same bytes as the read layer s0 - 1's writes would have issued; the read computes the norm scales itself,
         * the same bits as the write's (docs/OPS.md `ss`); no gather of the stream at the boundary */
        const HyperConn& h0 = m.layers[(size_t)l0].hc_mix;
        const bool sp0 = lsp(l0);
        for (int64_t p = 0; p < K; ++p) {
            if (sp_pm(p, sp0)) hrm(h0, p, false, sp0); else bi_read(c, m, h0, T, r0(p), rn(p));
            bt_rd(c, m, r0(p), rn(p), "rd_mix", (int)l0, false);
            if (sp_pm(p, sp0)) on1(true, [&] { ag_p(p); });
        }
        if (sp0) {
            wait1();
            for (int64_t p = 0; p < K; ++p)
                if (sp_pm(p, sp0)) h0.codes(c, r0(p), rn(p));
        }
    } else if (n > 0) {
        if (m.ple_layer == l0) { m.ple.step(c, batch, tipp); bt_pt(c, m, m.b_h, 0, T, "ple", (int)l0); }
        for (int64_t p = 0; p < K; ++p) {
            bi_read(c, m, m.layers[(size_t)l0].hc_mix, T, r0(p), rn(p));
            bt_rd(c, m, r0(p), rn(p), "rd_mix", (int)l0, false);
        }
    }
    /* qfn-on-ds P2S note (F50): the layers that take the pairs in this step, and the first that does not with its reason */
    int p2s_n = 0;
    int64_t p2s_no = -1;
    const char* p2s_why = nullptr;
    for (int64_t li = 0; li < n; ++li) {
        if (!lr_held(m, li)) continue;   /* Qwen tp8 S1: another node's layer (the stream passes on) */
        if (lead_tail_only && li < m.pp_t) continue;   /* Qwen tp8 S3b: the head part ran these */
        if (fol && (li < pp.s0 || li >= pp.s1)) continue;   /* Qwen tp8 S6: another segment's layer */
        const Layer& l = m.layers[(size_t)li];
        const int64_t lnx = lr_after(m, li);   /* the next layer this node runs (li + 1 without ranges), -1 after the last */
        spL = lsp(li);
        const bool spN = lsp(lnx >= 0 ? lnx : li + 1);   /* the next layer's connection read (fn4 SP-PLE) */
        /* The mixer finishes its output a piece at a time where its last stages are per row and
         * hands the first pieces' reductions to the second lane as it goes (ArPieces), so the
         * first piece's all-reduce runs beside the next piece's attention or out projection. */
        MixAr ms{c, &l};
        if (spL) { ms.rs = m.sp_rs; ms.x = m.a_x.x; ms.n_embd = g.n_embd; ms.sp_min = kSpMinRows; }
        /* fn4 PF marks: this layer runs the hc_sp 2 order (every piece SP) and the switch asks for the precise waits */
        uint32_t mk = 0;
        /* qfn-on-ds P2S (QUEUE Q38): this layer runs the hc_sp 2 order with paired routed passes -- every piece SP and full,
         * not under SP-route (sp_route bit 1), and the MoE block's forms allow a joint pass (modes 1 / 2) */
        const bool pair_sp_now = pair_sp && spo &&
                                 !(m.sp_agl && (fn4_sproute_flag().load(std::memory_order_relaxed) & 1)) &&
                                 [&] { for (int64_t q = 0; q < K; ++q) if (!sp_p(q)) return false; return true; }() &&
                                 (pmode == 3 || (l.mlp.pair_ok(2 * R, R) &&
                                                 (rn(K - 1) == R || K % 2 != 0 || l.mlp.pair_ok(R + rn(K - 1), R))));   /* PSP's partial pair */
        /* F50: counted over the step's layers and noted after the loop (layer 0 runs without SP under sp_ple 1, so a note
         * read at layer 0 alone said "today's order" while layers 1-47 paired) */
        if (pair_sp && g.rank == 0) {
            if (pair_sp_now) {
                ++p2s_n;
            } else if (p2s_no < 0) {
                p2s_no = li;
                p2s_why = !spo ? "not the hc_sp 2 order"
                          : (m.sp_agl && (fn4_sproute_flag().load(std::memory_order_relaxed) & 1)) ? "SP-route"
                          : [&] { for (int64_t q = 0; q < K; ++q) if (!sp_p(q)) return false; return true; }()
                              ? (l.mlp.pair_why(2 * R, R) ? l.mlp.pair_why(2 * R, R) : "?")
                              : "a piece not SP";
            }
        }
        const bool pmk_l = pmk && spo && !pair_on && !pair_sp_now && K >= 2 && K <= 8 &&
                           [&] { for (int64_t q = 0; q < K; ++q) if (!sp_p(q)) return false; return true; }();
        if (pmk_l) { ms.mk = &mk; ms.rows = R; }
        ArPieces pc{};
        pc.rows = R; pc.done = mix_ar_piece; pc.u = &ms; pc.cu = gcu;
        if (gpf) pc.min_last = kBiNarrowRows;   /* lever GPF: a (R, <= 64)-row step's mixer runs whole, as at K >= 3 */
        /* fn4 PF-ORDER (sp_route bit 32, with the marks): a delta-net layer hands only piece 0's reduction; the step issues
         * the others beside the pieces' connections (ar_mix below), so the second lane takes piece 0's gather before
         * piece 1's and 2's mixer reductions instead of after them (the delta net hands every piece at its very end) */
        if (pmk_l && pford && !l.full) pc.max_hand = 1;
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch, false, &pc); }
        else        l.gdn.step(c, batch, false, &pc, tipp);
        bt_blk(c, m, l, T, (int)li);   /* lane 0's buffers; lane 1 reduces only `x` */
        auto ar_mix = [&](int64_t p) {
            if (sp_p(p))     rs_p(p);
            else if (l.full) l.attn.issue_ar(c, r0(p), rn(p));
            else             l.gdn.issue_ar(c, r0(p), rn(p));
        };
        /* A block that handed nothing leaves the first piece's all-reduce to be issued here. */
        if (pc.handed == 0) on1(true, [&] { ar_mix(0); });
        auto mark1 = [&](int k) { if (k < 16 && rad_lane_join(c, 1, 0x100 + k) == RAD_OK) mk |= 1u << k; };
        auto waitm = [&](int k) {
            if (k < 16 && (mk & (1u << k))) { if (!nowait) rad_lane_join(c, 0x200 + k, 0); mk &= ~(1u << k); } else wait1();
        };
        if (pmk_l && pc.handed == 0) mark1(0);
        if (!pmk_l) wait1();   /* fn4 PF marks: the spo branch below waits piece by piece */
        int64_t issued = pc.handed > 0 ? pc.handed : 1;
        int64_t sp_pend = -1;   /* fn4 SP-hc: a piece whose MoE reduce-scatter is still to be issued */
        int64_t pp_lo = 0, pp_n = 0;   /* qfn-on-ds P2S: the last pass's pieces [pp_lo, pp_lo + pp_n) still to be reduce-scattered */
        bool pp_mark = false;          /* P2S: the writes wait for piece K - 2's reduce-scatter by mark 15 */
        bool behind = pc.handed == 0;     /* lane 1 already behind the whole mixer */
        const bool pair_now = pair_on && l.mlp.pair_ok(T, R);
        if (pair_on && li == l0 && g.rank == 0 && m.pair_noted != (pair_now ? pmode : -1)) {
            m.pair_noted = pair_now ? pmode : -1;
            fprintf(stderr, pair_now ? "radiance: qwen4exp: fn4 P2x: a pipelined step's two pieces share one routed pass "
                                       "(moe_pair %d, %lld rows, MoE rows %lld)\n"
                                     : "radiance: qwen4exp: fn4 P2x: moe_pair %d asked, but the MoE block's forms do not "
                                       "allow the pair at %lld rows (MoE rows %lld): per-piece passes\n",
                    pmode, (long long)T, (long long)m.moecfg.rows);
        }
        if (pair_now) {
            /* fn4 P2x: piece 0's connections beside piece 1's mixer all-reduce, then piece 1's, then one routed pass
             * over both; piece 0's shared arm (mode 1) + gather, its MoE all-reduce on lane 1 beside piece 1's shared
             * arm + gather, then piece 1's beside the writes below (as the per-piece form's last one). */
            if (1 >= issued) {
                on1(!behind, [&] { ar_mix(1); });
                behind = true;
                issued = 2;
            }
            const bool ss0 = l.hc_mix.write(c, T, r0(0), rn(0));
            bt_pt(c, m, m.b_h, r0(0), rn(0), "h_mix", (int)li);
            bi_read(c, m, l.hc_ffn, T, r0(0), rn(0), ss0);
            bt_rd(c, m, r0(0), rn(0), "rd_ffn", (int)li, true);
            wait1();
            const bool ss1 = l.hc_mix.write(c, T, r0(1), rn(1));
            bt_pt(c, m, m.b_h, r0(1), rn(1), "h_mix", (int)li);
            bi_read(c, m, l.hc_ffn, T, r0(1), rn(1), ss1);
            bt_rd(c, m, r0(1), rn(1), "rd_ffn", (int)li, true);
            l.mlp.pair_routed(c, T, R, pmode == 2);
            l.mlp.pair_finish(c, T, r0(0), rn(0), pmode != 2);
            on1(true, [&] { l.mlp.issue_ar(c, r0(0), rn(0)); });
            l.mlp.pair_finish(c, T, r0(1), rn(1), pmode != 2);
            wait1();
            on1(true, [&] { l.mlp.issue_ar(c, r0(1), rn(1)); });
        } else if (pair_sp_now) {
            /* qfn-on-ds lever P2S (QUEUE Q38): the hc_sp 2 order (the branch below) with the pieces' routed experts in pairs. Per
             * piece p: piece p + 1's connections beside piece p's gather, the wait, piece p + 1's gather, piece p's codes; a
             * pair's first piece then waits, and at its second the pending MoE reduce-scatters (the previous pass's pieces) go
             * out beside this pass -- one routed pass over both pieces (moe_pair 1 / 2: every touched expert read and decoded
             * once for 2 R rows), or each piece's own pass in the same order (3: exact by construction, the order's own cost).
             * 4: the joint pair only where two more pieces follow it (K = 4: (0, 1), then 2 and 3 single), so the last
             * pieces keep today's tail. The last pass's reduce-scatters go out after the loop; with two, the writes wait for
             * the first by a mark. The same ops on the same rows as the branch below; a joint pass's slot outputs are the
             * per-piece pass's bytes (perfmoe --split-seed, LEVERS P2x row 134 (162)). */
            if (1 >= issued) { on1(!behind, [&] { ar_mix(1); }); behind = true; issued = 2; }
            {
                const bool ss0 = hw(l.hc_mix, 0);
                bt_pt(c, m, m.b_h, r0(0), rn(0), "h_mix", (int)li);
                hr(l.hc_ffn, 0, ss0);
                bt_rd(c, m, r0(0), rn(0), "rd_ffn", (int)li, true);
            }
            wait1();
            on1(true, [&] { ag_p(0); });
            auto paired = [&](int64_t q) {   /* pieces q (even) and q + 1 share a pass */
                return q % 2 == 0 && q + 1 < K && (pmode != 4 || q + 3 < K);
            };
            for (int64_t p = 0; p < K; ++p) {
                if (p + 1 < K) {
                    if (p + 2 < K && p + 2 >= issued) { on1(!behind, [&] { ar_mix(p + 2); }); behind = true; issued = p + 3; }
                    const bool ss = hw(l.hc_mix, p + 1);
                    bt_pt(c, m, m.b_h, r0(p + 1), rn(p + 1), "h_mix", (int)li);
                    hr(l.hc_ffn, p + 1, ss);
                    bt_rd(c, m, r0(p + 1), rn(p + 1), "rd_ffn", (int)li, true);
                }
                wait1();
                if (p + 1 < K) on1(true, [&] { ag_p(p + 1); });
                const bool first = paired(p), second = p >= 1 && paired(p - 1);
                if (!first && pp_n > 0) {   /* the previous pass's reduce-scatters beside this pass */
                    const int64_t q = pp_lo, nq = pp_n;
                    on1(true, [&] { for (int64_t j = 0; j < nq; ++j) rs_p(q + j); });
                    pp_n = 0;
                }
                l.hc_ffn.codes(c, r0(p), rn(p));
                if (first) continue;
                if (second && pmode != 3) {
                    /* PN: the pair's rows -- 2 R, or R + the partial last piece under lever PSP (pair_routed_at's second piece
                     * is N - R rows) */
                    const int64_t P0 = r0(p - 1), PN = rn(p - 1) + rn(p);
                    l.mlp.pair_routed_at(c, T, P0, PN, R, pmode == 2);
                    l.mlp.pair_finish(c, T, P0, rn(p - 1), pmode != 2, P0, PN);
                    l.mlp.pair_finish(c, T, r0(p), rn(p), pmode != 2, P0, PN);
                } else {
                    if (second) l.mlp.pass(c, T, r0(p - 1), rn(p - 1), false);
                    l.mlp.pass(c, T, r0(p), rn(p), false);
                }
                pp_lo = second ? p - 1 : p;
                pp_n = second ? 2 : 1;
            }
        } else if (spo && [&] { for (int64_t q = 0; q < K; ++q) if (!sp_p(q)) return false; return K >= 2; }()) {
            /* fn4 hc_sp 2: every piece SP. Piece 0's connections (piece 1's mixer reduce-scatter beside them), one wait for
             * both reduce-scatters (hw of piece 1 reads piece 1's), then per piece p: piece p + 1's connections beside piece
             * p's gather, the wait, piece p + 1's gather and piece p - 1's MoE reduce-scatter beside piece p's codes + MoE.
             * Piece p + 2's mixer reduce-scatter goes out before piece p + 1's write, so the wait at p covers it. */
            if (1 >= issued) { on1(!behind, [&] { ar_mix(1); }); behind = true; issued = 2; if (pmk_l) mark1(1); }
            /* fn4 SP-route: the router on this rank's quarter right after its read (lane 0), its logits gathered behind the
             * piece's block-input gather (lane 1), the pass without its GEMM */
            /* fn4 PF probe (qwen/prefill, measurement of where the pipelined step's time goes; exact: the repeats rewrite the
             * same bytes): sp_route bit 2 = every block-input gather issued twice on lane 1 (+1 AG a piece and layer), bit 4 =
             * every quarter connection read issued twice on lane 0 (+1 hc read a piece and layer). */
            const int sprv = fn4_sproute_flag().load(std::memory_order_relaxed);
            const bool rsp = m.sp_agl && (sprv & 1);
            const bool prb_ag = (sprv & 2) != 0, prb_hr = (sprv & 4) != 0;
            const int64_t NE = m.moecfg.n_expert_all;
            bool routed[8] = { false, false, false, false, false, false, false, false };
            auto rq = [&](int64_t p) { if (rsp && p < 8) routed[p] = l.mlp.route_rows(c, rn(p), hq0(p), hqn(p)); };
            auto agl = [&](int64_t p) {
                if (p < 8 && routed[p])
                    on1(true, [&] { RAD_ISSUE_N(c, m.sp_agl, rn(p) * NE, brow_slice(m.b_rlogits, r0(p), rn(p), NE), RAD_NONE); });
            };
            {
                if (pmk_l) waitm(0);   /* fn4 PF marks: piece 0's mixer reduction only */
                const bool ss0 = hw(l.hc_mix, 0);
                bt_pt(c, m, m.b_h, r0(0), rn(0), "h_mix", (int)li);
                hr(l.hc_ffn, 0, ss0);
                if (prb_hr) hr(l.hc_ffn, 0, ss0);
                bt_rd(c, m, r0(0), rn(0), "rd_ffn", (int)li, true);
                rq(0);
            }
            if (!pmk_l) wait1();
            on1(true, [&] { ag_p(0); });
            if (prb_ag) on1(true, [&] { ag_p(0); });
            agl(0);
            if (pmk_l) mark1(8);
            for (int64_t p = 0; p < K; ++p) {
                if (p + 1 < K) {
                    if (p + 2 < K && p + 2 >= issued) {
                        on1(!behind, [&] { ar_mix(p + 2); }); behind = true; issued = p + 3;
                        if (pmk_l) mark1((int)p + 2);
                    }
                    if (pmk_l) waitm((int)p + 1);   /* fn4 PF marks: piece p + 1's mixer reduction */
                    const bool ss = hw(l.hc_mix, p + 1);
                    bt_pt(c, m, m.b_h, r0(p + 1), rn(p + 1), "h_mix", (int)li);
                    hr(l.hc_ffn, p + 1, ss);
                    if (prb_hr) hr(l.hc_ffn, p + 1, ss);
                    bt_rd(c, m, r0(p + 1), rn(p + 1), "rd_ffn", (int)li, true);
                    rq(p + 1);
                }
                if (pmk_l) waitm(8 + (int)p); else wait1();   /* fn4 PF marks: piece p's gathers only */
                if (p + 1 < K) {
                    on1(true, [&] { ag_p(p + 1); });
                    if (prb_ag) on1(true, [&] { ag_p(p + 1); });
                    agl(p + 1);
                    if (pmk_l) mark1(8 + (int)p + 1);
                }
                if (sp_pend >= 0) { const int64_t q = sp_pend; on1(true, [&] { rs_p(q); }); sp_pend = -1; }
                l.hc_ffn.codes(c, r0(p), rn(p));
                l.mlp.pass(c, T, r0(p), rn(p), false, p < 8 && routed[p]);
                sp_pend = p;
            }
        } else if (spo_tail && K >= 2 && !sp_p(K - 1) && [&] { for (int64_t q = 0; q + 1 < K; ++q) if (!sp_p(q)) return false; return true; }()) {
            /* fn4 SPO-tail (16.11.2026, qwen/prefill; sp_route bit 16): the hc_sp 2 order for a step whose last piece is
             * shorter than a pass (every prompt's last step; a Claude Code follow-up's 2-8k chunk): the full pieces as the
             * branch above runs them, the partial piece's connections (whole-piece forms, no SP) beside the last full
             * piece's gather, its pass and MoE all-reduce after the full pieces' passes. The same ops on the same rows as
             * the loop below; only the issue order of the full pieces' gathers and the partial piece's reads moves. */
            if (1 >= issued) { on1(!behind, [&] { ar_mix(1); }); behind = true; issued = 2; }
            {
                const bool ss0 = hw(l.hc_mix, 0);
                bt_pt(c, m, m.b_h, r0(0), rn(0), "h_mix", (int)li);
                hr(l.hc_ffn, 0, ss0);
                bt_rd(c, m, r0(0), rn(0), "rd_ffn", (int)li, true);
            }
            wait1();
            on1(true, [&] { ag_p(0); });
            for (int64_t p = 0; p < K; ++p) {
                if (p + 1 < K) {
                    if (p + 2 < K && p + 2 >= issued) { on1(!behind, [&] { ar_mix(p + 2); }); behind = true; issued = p + 3; }
                    const bool ss = hw(l.hc_mix, p + 1);
                    bt_pt(c, m, m.b_h, r0(p + 1), rn(p + 1), "h_mix", (int)li);
                    if (sp_p(p + 1)) hr(l.hc_ffn, p + 1, ss);
                    else bi_read(c, m, l.hc_ffn, T, hq0(p + 1), hqn(p + 1), ss);
                    bt_rd(c, m, r0(p + 1), rn(p + 1), "rd_ffn", (int)li, true);
                }
                wait1();
                if (p + 1 < K && sp_p(p + 1)) on1(true, [&] { ag_p(p + 1); });
                if (sp_pend >= 0) { const int64_t q = sp_pend; on1(true, [&] { rs_p(q); }); sp_pend = -1; }
                if (sp_p(p)) {
                    l.hc_ffn.codes(c, r0(p), rn(p));
                    l.mlp.pass(c, T, r0(p), rn(p), false);
                    sp_pend = p;
                } else {
                    l.mlp.pass(c, T, r0(p), rn(p), false);
                    wait1();
                    on1(true, [&] { l.mlp.issue_ar(c, r0(p), rn(p)); });
                }
            }
        } else
        for (int64_t p = 0; p < K; ++p) {
            if (p + 1 < K && p + 1 >= issued) {
                on1(!behind, [&] { ar_mix(p + 1); });
                behind = true;
                issued = p + 2;
            }
            const bool ss_mix = hw(l.hc_mix, p);
            bt_pt(c, m, m.b_h, r0(p), rn(p), "h_mix", (int)li);
            if (sp_p(p)) hr(l.hc_ffn, p, ss_mix); else bi_read(c, m, l.hc_ffn, T, hq0(p), hqn(p), ss_mix);
            bt_rd(c, m, r0(p), rn(p), "rd_ffn", (int)li, true);
            if (sp_p(p)) {
                /* fn4 SP-hc: every rank's quarter of the MoE input gathered and coded on every row; the previous
                 * piece's MoE reduce-scatter is issued only after that wait, so it runs beside this piece's MoE (as the
                 * stock all-reduce does) instead of queueing ahead of the gather on the one collective lane. */
                on1(true, [&] { ag_p(p); });
                wait1();
                if (sp_pend >= 0) { const int64_t q = sp_pend; on1(true, [&] { rs_p(q); }); sp_pend = -1; }
                l.hc_ffn.codes(c, r0(p), rn(p));
                l.mlp.pass(c, T, r0(p), rn(p), false);
                sp_pend = p;
                continue;
            }
            if (sp_pend >= 0) { wait1(); const int64_t q = sp_pend; on1(true, [&] { rs_p(q); }); sp_pend = -1; }
            l.mlp.pass(c, T, r0(p), rn(p), false);
            /* Lane 0 waits here for the next piece's mixer all-reduce and the last piece's MoE
             * one, both issued before this pass; lane 1 then takes this piece's behind its
             * gather. */
            wait1();
            on1(true, [&] { l.mlp.issue_ar(c, r0(p), rn(p)); });
        }
        if (sp_pend >= 0) { wait1(); const int64_t q = sp_pend; on1(true, [&] { rs_p(q); }); sp_pend = -1; }
        if (pp_n > 0) {   /* P2S: the last pass's reduce-scatters; with two, mark the first for the writes below */
            wait1();
            const int64_t q = pp_lo;
            on1(true, [&] { rs_p(q); });
            if (pp_n == 2) {
                mark1(15);   /* waitm(15) below falls back to the whole-lane wait if the mark was not taken */
                pp_mark = true;
                on1(true, [&] { rs_p(q + 1); });
            }
            pp_n = 0;
        }
        l.mlp.route_report(c);
        bt_moe(c, m, T, (int)li);   /* lane 1 reduces only `x` */
        /* The writes, and the next layer's reads beside the last piece's all-reduce. PLE adds into
         * every row of the stream before the next layer reads it, so there the reads wait. */
        const Layer* nx = lnx >= 0 ? &m.layers[(size_t)lnx] : nullptr;   /* (Qwen tp8 S1: the next held layer) */
        const bool ple_next = nx && lnx == m.ple_layer;
        /* Qwen tp8 S3b: the leader's cut (layer h - 1 -> t): the writes, then X1 out and X2 in, then layer t's piece reads */
        const bool cut = pp.on && m.pp_rank == 0 && lnx == m.pp_t;
        /* Qwen tp8 S6: the follower's segment ends at boundary s1 after this layer's writes (each rank its rows): the save */
        const bool seg_end = fol && pp.seg_out && lnx == pp.s1;
        for (int64_t p = 0; p < K; ++p) {
            if (pp_mark && p == K - 2) { waitm(15); pp_mark = false; }   /* P2S: piece K - 2's reduce-scatter */
            if (p == K - 1) wait1();
            const bool ss_ffn = hw(l.hc_ffn, p);
            /* The projection of this piece, before the next layer reads it. */
            const bool projected = l.proj.step(c, batch, r0(p), rn(p));
            bt_pt(c, m, m.b_h, r0(p), rn(p), "h_ffn", (int)li);
            if (nx && !ple_next && !cut && !seg_end) {
                if (sp_pm(p, spN)) hrm(nx->hc_mix, p, ss_ffn && !projected, spN);
                else bi_read(c, m, nx->hc_mix, T, hq0m(p, spN), hqnm(p, spN), ss_ffn && !projected);
                bt_rd(c, m, r0(p), rn(p), "rd_mix", (int)lnx, false);   /* lane 1 holds the last piece's rows only */
                if (sp_pm(p, spN)) on1(true, [&] { ag_p(p); });   /* fn4 SP-hc: the next block's input, every quarter */
            }
        }
        if (cut) {   /* every write on lane 0 and lane 1 joined (the last write waited) */
            /* Qwen tp8 S3b-SP: under SP-hc layer h - 1 wrote each rank's quarter rows of an SP piece only, and stage_send
             * copies rank `sender`'s stream: the quarters gathered raw first (the step end's gather, exact), lane 0 waits */
            if (pp.head && sp) {
                bool any = false;
                for (int64_t p = 0; p < K; ++p)
                    if (sp_p(p)) { on1(true, [&] { agh_p(p); }); any = true; }
                if (any) wait1();
            }
            if (pp.head && !pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T)) return;
            if (!pp.tail) return;
            if (!pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T)) return;
            for (int64_t p = 0; p < K; ++p) {
                bi_read(c, m, nx->hc_mix, T, r0(p), rn(p));
                bt_rd(c, m, r0(p), rn(p), "rd_mix", (int)lnx, false);
            }
            continue;
        }
        if (seg_end) {   /* every write on lane 0 and lane 1 joined (the last write waited); nothing of lane 1 after it */
            if (nowait) rad_lane_join(c, 1, 0);
            pp_seg_io(c, m, pp.s1, true, T);
            return;
        }
        if (ple_next) {
            m.ple.step(c, batch, tipp);
            bt_pt(c, m, m.b_h, 0, T, "ple", (int)lnx);
            for (int64_t p = 0; p < K; ++p) {
                if (sp_pm(p, spN)) hrm(nx->hc_mix, p, false, spN); else bi_read(c, m, nx->hc_mix, T, r0(p), rn(p));
                bt_rd(c, m, r0(p), rn(p), "rd_mix", (int)lnx, false);
                if (sp_pm(p, spN)) on1(true, [&] { ag_p(p); });
            }
        }
        /* fn4 SP-hc: the gathered inputs' codes on every row, before the next block reads them */
        if (spN && nx && spo && !ple_next) {
            /* hc_sp 2: the writes' last-piece wait already joined every gather but the last piece's: their codes beside it */
            for (int64_t p = 0; p + 1 < K; ++p)
                if (sp_pm(p, spN)) nx->hc_mix.codes(c, r0(p), rn(p));
            wait1();
            if (sp_pm(K - 1, spN)) nx->hc_mix.codes(c, r0(K - 1), rn(K - 1));
        } else if (spN && nx) {
            wait1();
            for (int64_t p = 0; p < K; ++p)
                if (sp_pm(p, spN)) nx->hc_mix.codes(c, r0(p), rn(p));
        }
    }
    /* qfn-on-ds P2S note (F50): once a change of (mode, layers paired) */
    if (pair_sp && g.rank == 0 && m.pair_noted != 1000 * (pmode + 1) + p2s_n) {
        m.pair_noted = 1000 * (pmode + 1) + p2s_n;
        fprintf(stderr, "radiance: qwen4exp: P2S: moe_pair %d under hc_sp 2 (%s): a %lld-row step's %lld pieces -- %d of %lld layers "
                        "in pairs%s%s%s%s\n",
                pmode, pmode == 3 ? "the pair order, each piece's own pass" : pmode == 4 ? "one joint pass, the last two pieces single"
                                                                         : "one joint pass a pair",
                (long long)T, (long long)K, p2s_n, (long long)(m.lr.on ? m.lr.n_own : n), p2s_no >= 0 ? "; the first in today's order: layer " : "",
                p2s_no >= 0 ? std::to_string(p2s_no).c_str() : "", p2s_no >= 0 ? " -- " : "", p2s_no >= 0 && p2s_why ? p2s_why : "");
    }
    /* fn4 SP-hc: after the last layer every rank holds only its quarters of the wide stream; the output mixer and the
     * MTP head read every row of it, so it is gathered raw (exact) here. */
    if (sp) {
        for (int64_t p = 0; p < K; ++p)
            if (sp_p(p)) on1(true, [&] { agh_p(p); });
        wait1();
    }
    if (nowait) rad_lane_join(c, 1, 0);   /* PF probe NOWAIT: the step's one wait for lane 1 */
    /* Qwen tp8 S3b: the follower's result back after its last layer's writes (the last waited for lane 1) */
    if (pp.on && m.pp_rank == 1) pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T);
}

/* The trunk one stream at a time over the whole step: every step a span of 1 makes, and every
 * other where step_pipelined's conditions do not hold. */
static void step_serial(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu, const PpPass& pp = PpPass{}) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;
    /* Qwen tp8 S3a: the embedding, the PLE's ids and the stream's entry are the leader's head; the follower's stream is X1 */
    if (!pp.on || (m.pp_rank == 0 && pp.head)) {
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    /* Unconditional, unlike qwen35_fp8's: the first block has no norm to fold it into, only a
     * gated-residual read. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    bt_pt(c, m, m.a_x.x, 0, T, "emb", -1);
    /* The PLE's hash, ahead of every layer: its host gather waits for it, and the layers issued
     * between the two keep the device busy while the host gathers (rad_block_ple.h). */
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    bt_pt(c, m, m.b_h, 0, T, "enter", -1);
    } else if (m.pp_rank == 1 && !(pp.seg_in ? pp_seg_io(c, m, pp.s0, false, T)      /* Qwen tp8 S6: boundary s0's restore */
                                             : pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T))) {
        return;
    }

    if (m.op_rope_cs && T <= qk_fuse_now(qk_fuse_rows(g)) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    /* fn4 W1b: whether the stream's last change was a write that stored the norm scales for its rows. (Qwen tp8 S6: a
     * segment's first read after its restore computes them itself, the same bits) */
    bool ss_ok = false;
    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        /* Qwen tp8 S1: another node's layer -- the stream passes on (the write's norm scales stay its rows', as a crossing
         * that carries them) */
        if (!lr_held(m, li)) continue;
        if (pp.on && m.pp_rank == 1 && (li < pp.s0 || li >= pp.s1)) continue;   /* Qwen tp8 S6: another segment's layer */
        if (pp.on && m.pp_rank == 0) {   /* Qwen tp8 S3a: the leader's parts */
            if (li < m.pp_h && !pp.head) continue;
            if (li == m.pp_t) {   /* the cut: X1 out after layer h - 1's write, X2 in before layer t's read */
                if (pp.head && !pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T)) return;
                if (!pp.tail) return;
                if (!pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T)) return;
                ss_ok = false;   /* the read computes the norm scales itself: the same bits as the write's (docs/OPS.md `ss`) */
            }
        }
        const Layer& l = m.layers[(size_t)li];
        /* Before the attention connection reads the stream, which is where the reference puts it. */
        if (li == m.ple_layer) { m.ple.step(c, batch); ss_ok = false; bt_pt(c, m, m.b_h, 0, T, "ple", (int)li); }
        bi_read_pieces(c, m, l.hc_mix, T, ss_ok);
        bt_rd(c, m, 0, T, "rd_mix", (int)li, false);
        /* The indexer runs first, since the attention reads its selection. The block's input is
         * already normed and quantised by the connection (`ext_in`), so `w.h` is final. */
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch); }
        else        l.gdn.step(c, batch);
        bt_blk(c, m, l, T, (int)li);
        /* fn4 tracer (13.11., 222): where the next write takes the all-reduce (four ranks, <= arw_rows rows: ar_hc_write) a_x.x
         * holds this rank's PARTIAL, else the reduced sum -- named apart ("blkp" / "moep") so an alone step and a wider one are
         * never compared across the two meanings; the write's output (h_mix / h_ffn) is the comparable point. */
        bt_pt(c, m, m.a_x.x, 0, T, l.hc_mix.takes_ar(T) ? "blkp" : "blk", (int)li);
        const bool ss_mix = l.hc_mix.write(c, T, 0, T);
        bt_pt(c, m, m.b_h, 0, T, "h_mix", (int)li);
        dbg_resid(c, (int)li, "mix", g.n_embd, m.b_h, m.a_x.x);
        bi_read_pieces(c, m, l.hc_ffn, T, ss_mix);
        bt_rd(c, m, 0, T, "rd_ffn", (int)li, true);
        static const bool bt_sp_ser = [] { const char* e = std::getenv("RADIANCE_FN4_BITRACE_SP"); return e && *e && *e != '0'; }();
        if (m.bt.on && bt_sp_ser && !l.mlp.op_add) {
            /* tracer only (RADIANCE_FN4_BITRACE_SP, qwen/SP-SERIAL-EXACT.md): the same passes and all-reduces as mlp.step,
             * the block's partial traced before its all-reduce (step_serial_sp's moe_part) */
            for (int64_t r0 = 0; r0 < T; r0 += l.mlp.c.rows)
                l.mlp.pass(c, T, r0, std::min<int64_t>(l.mlp.c.rows, T - r0), false);
            bt_pt(c, m, m.a_x.x, 0, T, "moe_part", (int)li);
            for (int64_t r0 = 0; r0 < T; r0 += l.mlp.c.rows) {
                const int64_t n = std::min<int64_t>(l.mlp.c.rows, T - r0);
                if (l.mlp.issues_ar(T, n)) l.mlp.issue_ar(c, r0, n, false);
            }
            l.mlp.route_report(c);
        } else
        l.mlp.step(c, batch);
        bt_moe(c, m, T, (int)li);
        bt_pt(c, m, m.a_x.x, 0, T, l.hc_ffn.takes_ar(T) ? "moep" : "moe", (int)li);
        ss_ok = l.hc_ffn.write(c, T, 0, T);
        /* The request's control vector, on the stream the write just finished. */
        if (l.proj.step(c, batch, 0, T)) ss_ok = false;
        bt_pt(c, m, m.b_h, 0, T, "h_ffn", (int)li);
        dbg_resid(c, (int)li, "ffn", g.n_embd, m.b_h, m.a_x.x);
    }
    /* Qwen tp8 S3a: the follower's result back to the leader after its last layer's write (S6: a segment's end, its save) */
    if (pp.on && m.pp_rank == 1) {
        if (pp.seg_out) pp_seg_io(c, m, pp.s1, true, T);
        else            pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T);
    }
}

/* ============================== fn4 SP-serial (14.11.2026) ==============================
 *
 * The newcomer step (MIXED: a ~1.1k-row prompt beside the decode rows) and a lone short prefill run step_serial, where every
 * rank reduces every row (one all-reduce a connection) and then writes and reads the gated residual on ALL rows: 43.0 of a
 * c16 newcomer step's 170 ms on rank 0 (P1, qwen/build2/STAGES.md). SP-hc (step_pipelined, hc_sp) already splits that work
 * for prompts wider than a pass. Here the same split for a serial step, connection by connection:
 *
 *   all-reduce of the block's partials            ->  reduce-scatter (sp_rs: whole-row shards of q = ceil(T / 4) rows)
 *   write + read on all T rows                    ->  write + read on this rank's q rows
 *   (the read's outputs on every rank)            ->  all-gather of the block input (sp_ag), its codes redone on every row
 *
 * and once after the last layer the wide stream's all-gather (sp_agh: the output mixer and the MTP head read every row).
 *
 * The same bytes as step_serial, op by op:
 *   - reduce-scatter + all-gather on the exact wire fold each element as the exact all-reduce does (fp32 from 0, ascending
 *     rank, one rounding; r4d_ar_sp_rows.hip), and the gather copies;
 *   - hc_write is per row;
 *   - a rank's rows of the decode segment are read as step_serial reads them (bi_read_seg: the bi_dec twin, each row its own
 *     sequence's alone read at any launch height); a whole prefill segment inside the rank's rows reads whole, as today;
 *     a PART of a prefill segment of h rows reads through HyperConn::read_kt: kc_rows = 128 ceil(h / 128) pins the down
 *     product's K split to the one the segment's read whole takes (a part of > 64 rows is at the same 128-row tile), the
 *     wide / PF2 forms are the same bytes at one split, the norm scales one sum in every form (or the write's `ss`);
 *   - hc_codes is hc_read's own codes stage over the gathered x (hc_quant_group: the same amax, scale and packing as the
 *     read's finisher and the decode form's B3).
 * So a step takes it only when every rank's part of every prefill segment is whole or more than 64 rows, every segment is
 * within the pinned reads' band. The PLE reads every row of the wide stream and its convolution mixes rows, so the layers
 * before it (Flash-Next's PLE is blk.1: layer 0) run replicated as step_serial runs them and the split starts at its layer's
 * read. Steps without a prefill segment (decode) and everything else keep step_serial. */
struct SpSer {
    struct Part { int64_t a = 0, n = 0, seg = 0; bool dec = false; };
    /* a run of rows [r0, r0 + n) the collectives shard whole (q = ceil(n / 4) rows a rank): this rank's rows and parts */
    struct Run { int64_t r0 = 0, n = 0, lo = 0, hi = 0; std::vector<Part> mine; };
    int  nrun = 0;      /* 1: the step as one run (sp_serial 1); 2: two halves, pipelined on the second lane (sp_serial 2) */
    Run  run[2];
};
/* The parts of run [r0, r0 + n) on every rank: false (the caller keeps step_serial, or one run) when a rank would have no rows
 * or a prefill segment's part on any rank is neither whole nor more than 64 rows (or past the pinned twins). Decided alike on
 * every rank. */
static bool spser_run(const Model& m, int64_t r0, int64_t n, SpSer::Run& run, bool* any_pf) {
    const Geom& g = m.g;
    const int64_t q = (n + g.world - 1) / g.world;
    run = SpSer::Run{};
    run.r0 = r0; run.n = n;
    for (int r = 0; r < g.world; ++r) {
        const int64_t lo = r0 + std::min<int64_t>((int64_t)r * q, n), hi = r0 + std::min<int64_t>((int64_t)(r + 1) * q, n);
        if (hi <= lo) return false;
        if (r == g.rank) { run.lo = lo; run.hi = hi; }
        for (size_t i = 0; i + 1 < m.bi_cut.size(); ++i) {
            const int64_t c0 = m.bi_cut[i], c1 = m.bi_cut[i + 1];
            const int64_t a = std::max(c0, lo), b = std::min(c1, hi);
            if (b <= a) continue;
            const bool dec = m.bi_mode >= 3 && a < m.bi_td;   /* bi_read_seg's own rule */
            const int64_t seg = c1 - c0;
            if (!dec) {
                *any_pf = true;
                if (seg > m.spser_rows) return false;
                if (b - a < seg && (b - a <= 64 || !m.layers[(size_t)m.lr_first].hc_mix.read_kt_op(seg))) return false;
            }
            if (r == g.rank) { SpSer::Part p; p.a = a; p.n = b - a; p.seg = seg; p.dec = dec; run.mine.push_back(p); }
        }
    }
    return true;
}
static bool spser_plan(const Model& m, const RadBatch* batch, SpSer& sp) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;
    const int mode = fn4_spser_flag().load(std::memory_order_relaxed);
    if (!m.spser_ok || g.world != 4 || mode <= 0) return false;
    /* (media rows are scattered into the embedding before the first read, which every rank still holds whole) */
    /* fn4 BI tracer + SP-serial (16.11.2026, qwen/KV-STREAM-EXACT.md 5d): RADIANCE_FN4_BITRACE_SP=1 keeps SP-serial on under the
     * tracer (a rank's hc rows outside its part are then not this step's: compare a part's rows on its own rank only) */
    static const bool bt_sp = [] { const char* e = std::getenv("RADIANCE_FN4_BITRACE_SP"); return e && *e && *e != '0'; }();
    /* (Qwen tp8 S3b-SP: a pipeline follower has no embedding, so no embedding all-reduce: its step starts at X1's receive) */
    if ((m.bt.on && !bt_sp) || dbg_resid_on() || m.moecfg.calib || (!m.op_embed_ar && !(m.pp && m.pp_rank == 1))) return false;
    /* the PLE's convolution mixes rows of the stream: the layers up to it run replicated (step_serial_sp), so it must leave
     * some to split */
    /* (Qwen tp8 S1: a layer this node runs after the PLE's) */
    if (m.layers.empty() || (m.ple_layer >= 0 && lr_after(m, m.ple_layer) < 0)) return false;
    if (!m.bi_on || m.bi_mode < 3 || m.bi_cut.size() < 2) return false;   /* the reads cut at the step's segments */
    if (T < 4 * 65 || T > m.spser_rows) return false;
    for (const Layer& l : m.layers) {
        if (!lr_held(m, &l - m.layers.data())) continue;   /* Qwen tp8 S1: another node's layer */
        if (!(l.full ? l.attn.issues_ar(T) : l.gdn.issues_ar(T))) return false;
        if (!l.mlp.issues_ar(T, std::min<int64_t>(T, l.mlp.c.rows))) return false;
    }
    /* sp_serial 2: two halves (the first a multiple of 4 rows, at least half), when both shard; else one run */
    if (mode == 2 && m.pipe_on) {
        bool pf0 = false, pf1 = false;
        const int64_t T1 = ((T / 2 + 3) / 4) * 4;
        if (T1 < T && spser_run(m, 0, T1, sp.run[0], &pf0) && spser_run(m, T1, T - T1, sp.run[1], &pf1) && (pf0 || pf1)) {
            sp.nrun = 2;
            return true;
        }
    }
    bool pf = false;
    if (!spser_run(m, 0, T, sp.run[0], &pf) || !pf) return false;
    sp.nrun = 1;
    return true;
}

/* diagnostic only (qwen/SP-SERIAL-EXACT.md): RADIANCE_FN4_SPDIAG bit mask, tracer starts only (host syncs) */
static int spdiag() {
    static const int v = [] { const char* e = std::getenv("RADIANCE_FN4_SPDIAG"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}
static void step_serial_sp(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu, const SpSer& sp,
                           const PpPass& pp = PpPass{}) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, D = g.n_embd, HN = m.hccfg.hc * g.n_embd;
    if (m.spser_noted != sp.nrun) {
        m.spser_noted = sp.nrun;
        if (g.rank == 0)
            fprintf(stderr, "radiance: qwen4exp: fn4 SP-serial: a serial step's prefill rows take the sequence-parallel gated "
                            "residual (%s; first at %lld rows, %zu part(s) on rank 0)\n",
                    sp.nrun == 2 ? "two halves, collectives on the second lane" : "one run", (long long)T, sp.run[0].mine.size());
    }
    const bool l2 = sp.nrun == 2;   /* sp_serial 2: every collective of the step on the second lane (Ctx::lane1_'s rule) */
    auto on1 = [&](bool behind0, auto&& issue) {
        if (!l2) { issue(); return; }
        rad_lane(c, 1);
        if (behind0) rad_lane_join(c, 0, 1);
        issue();
        rad_lane(c, 0);
    };
    auto wait1 = [&] { if (l2) rad_lane_join(c, 1, 0); };
    if (spdiag() & 1) {   /* diagnostic: both lanes drained on the host before the step */
        rad_lane(c, 1); rad_stream_sync(rad_stream(c)); rad_lane(c, 0); rad_stream_sync(rad_stream(c));
    }
    /* Qwen tp8 S3b-SP (round 31): this pass's part of the two-node pipeline, as step_serial's -- the embedding .. the entry are
     * the leader's head, the follower starts at X1's receive, a leader's tail part at X2's receive + layer t's read */
    const bool lead_tail_only = pp.on && m.pp_rank == 0 && !pp.head;
    if (!pp.on || (m.pp_rank == 0 && pp.head)) {
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    on1(true, [&] { RAD_ISSUE_N(c, m.op_embed_ar, T * D, brows(m.a_x.x, T), RAD_NONE); });
    if (spdiag() & 16) {   /* diagnostic: one more all-reduce on the instance (the embedding's again, of zeros added:
                            * a_x.x unchanged in value? no -- the embedding AR sums disjoint rows, so a second AR would
                            * multiply; use the gather of the whole step instead, which copies every rank's own rows) */
        const SpSer::Run allr = [&] { SpSer::Run r; r.r0 = 0; r.n = T; return r; }();
        (void)allr;
        RAD_ISSUE_N(c, m.sp_ag, T * D, brow_slice(m.a_x.x, 0, T, D), RAD_NONE);   /* after the AR every rank holds every row: a no-op copy */
    }
    wait1();
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    } else if (m.pp_rank == 1 && !(pp.seg_in ? pp_seg_io(c, m, pp.s0, false, T)      /* Qwen tp8 S6: boundary s0's restore */
                                             : pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T))) {
        return;
    }
    if (m.op_rope_cs && T <= qk_fuse_now(qk_fuse_rows(g)) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
    if (lead_tail_only && !pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T)) return;

    const int64_t band = m.hccfg.read_rows > 0 ? m.hccfg.read_rows : g.max_tok;
    /* lever LSK: SP-serial's own collectives (lane_kind 1) where declared, else the shared ops */
    const rad_op o_rs = m.ss_rs ? m.ss_rs : m.sp_rs, o_ag = m.ss_ag ? m.ss_ag : m.sp_ag, o_agh = m.ss_agh ? m.ss_agh : m.sp_agh;
    auto rs = [&](const SpSer::Run& r) { RAD_ISSUE_N(c, o_rs, r.n * D, brow_slice(m.a_x.x, r.r0, r.n, D), RAD_NONE); };
    auto ag = [&](const SpSer::Run& r) { RAD_ISSUE_N(c, o_ag, r.n * D, brow_slice(m.a_x.x, r.r0, r.n, D), RAD_NONE); };
    auto codes = [&](const HyperConn& h, int64_t a, int64_t n) {
        for (int64_t r0 = a; r0 < a + n; r0 += band) h.codes(c, r0, std::min(band, a + n - r0));
    };
    /* a connection's read on this rank's rows of a run: the decode rows and whole segments as step_serial reads them, a
     * segment's part through the K-pinned twin */
    auto rd = [&](const HyperConn& h, const SpSer::Run& r, bool ss) {
        for (const SpSer::Part& p : r.mine) {
            if (p.dec)             bi_read_seg(c, m, h, T, p.a, p.a + p.n, ss);
            else if (p.n == p.seg) hcp_read(c, m, h, T, p.a, p.n, ss);   /* lever HCP: the full piece's split */
            else if (hcp_pins(m, h, p.a, p.n)) { h.read_sp(c, T, p.a, p.n, ss); ++m.hcp_n; }
            else                   h.read_kt(c, p.a, p.n, p.seg, ss);
        }
    };
    /* One connection between two blocks: reduce-scatter the block's partials, this rank's rows written (+ `after`, e.g. the
     * control vector) and read by the next connection `nx` (none after the last layer), the block input gathered and coded.
     * Two runs: run 1's reduce-scatter beside run 0's write + read, run 0's gather beside run 1's. */
    /* tnm / tli: the BI tracer's point for the write's output on this rank's rows (step_serial's "h_mix" / "h_ffn"; only
     * under RADIANCE_FN4_BITRACE_SP, qwen/KV-STREAM-EXACT.md 5d) and, after the gather, the block input on every row */
    auto conn = [&](const HyperConn& wr, const HyperConn* nx, auto&& after, const char* tnm = nullptr, int tli = -1) {
        if (sp.nrun == 1) {
            const SpSer::Run& r = sp.run[0];
            rs(r);
            if (tnm) bt_pt(c, m, m.a_x.x, r.lo, r.hi - r.lo, tli >= 0 && tnm[2] == 'm' ? "blk" : "moe", tli);
            bool ss = wr.write(c, T, r.lo, r.hi - r.lo);
            if (after(r)) ss = false;
            if (tnm) bt_pt(c, m, m.b_h, r.lo, r.hi - r.lo, tnm, tli);
            if (nx) {
                rd(*nx, r, ss);
                /* diagnostic only (RADIANCE_FN4_SPDUMP=<dir>, RADIANCE_FN4_SPDUMP_T=<rows>): the block input of every row
                 * before and after the gather, bytes to <dir>/r<rank>-s<step>-<point>.bin (qwen/SP-SERIAL-EXACT.md) */
                static const char* sdd = [] { const char* e = std::getenv("RADIANCE_FN4_SPDUMP"); return e && *e ? e : (const char*)nullptr; }();
                static const int64_t sdt = [] { const char* e = std::getenv("RADIANCE_FN4_SPDUMP_T"); return e && *e ? (int64_t)std::atoll(e) : (int64_t)0; }();
                const bool sdump = sdd && T == sdt && tli == 1;
                auto dumpx = [&](const char* what) {
                    if (!sdump) return;
                    std::vector<uint16_t> hb((size_t)(T * D));
                    const RadStream st = rad_stream(c);
                    rad_memcpy_async(hb.data(), rad_buf_ptr(c, m.a_x.x), (size_t)(T * D * 2), st);
                    rad_stream_sync(st);
                    char fn[600];
                    snprintf(fn, sizeof fn, "%s/r%d-s%lld-L%02d-%s-%s.bin", sdd, g.rank, (long long)batch->step, tli, tnm ? tnm : "x", what);
                    if (FILE* f = fopen(fn, "wb")) { fwrite(hb.data(), 2, hb.size(), f); fclose(f); }
                };
                dumpx("pre_ag");
                /* the read's outputs on this rank's rows (step_serial's rd_ffn / rd_mix), then the gathered + coded block
                 * input on every row (rdg_*: compare with step_serial's rd_* on all rows) */
                const bool fnx = tnm && tnm[2] == 'm';
                const int  lnx = fnx ? tli : tli + 1;
                if (tnm) bt_rd(c, m, r.lo, r.hi - r.lo, fnx ? "rd_ffn" : "rd_mix", lnx, fnx);
                ag(r);
                if (spdiag() & 2) ag(r);   /* diagnostic: the same gather a second time */
                dumpx("post_ag");
                codes(*nx, 0, T);
                dumpx("post_codes");
                if (sdump) {
                    /* the codes the shaders wrote from the gathered rows (their view of x), rows [0, T) */
                    const rad_buf qb = nx->c.codes_i8 ? m.a_x.q8 : m.a_x.q, sb = nx->c.codes_i8 ? m.a_x.s8 : m.a_x.s;
                    std::vector<uint8_t> hq((size_t)(T * D));
                    std::vector<float> hs((size_t)(T * (D / 128)));
                    const RadStream st = rad_stream(c);
                    rad_memcpy_async(hq.data(), rad_buf_ptr(c, qb), (size_t)(T * D), st);
                    rad_memcpy_async(hs.data(), rad_buf_ptr(c, sb), hs.size() * 4, st);
                    rad_stream_sync(st);
                    char fn[600];
                    snprintf(fn, sizeof fn, "%s/r%d-s%lld-L%02d-%s-codes%s.bin", sdd, g.rank, (long long)batch->step, tli, tnm ? tnm : "x",
                             nx->c.codes_i8 ? "i8" : "f8");
                    if (FILE* f = fopen(fn, "wb")) { fwrite(hq.data(), 1, hq.size(), f); fwrite(hs.data(), 4, hs.size(), f); fclose(f); }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    dumpx("post_codes_100ms");
                }
                if (tnm) bt_rd(c, m, 0, T, fnx ? "rdg_ffn" : "rdg_mix", lnx, fnx);
            }
            return;
        }
        auto half = [&](const SpSer::Run& r) {
            bool ss = wr.write(c, T, r.lo, r.hi - r.lo);
            if (after(r)) ss = false;
            if (nx) rd(*nx, r, ss);
        };
        on1(true, [&] { rs(sp.run[0]); });
        wait1();                                       /* run 0 reduced */
        on1(false, [&] { rs(sp.run[1]); });            /* run 1's reduce-scatter beside run 0's write + read */
        half(sp.run[0]);
        wait1();                                       /* run 1 reduced: lane 1 holds nothing else yet */
        if (nx) on1(true, [&] { ag(sp.run[0]); });     /* run 0's gather beside run 1's write + read */
        half(sp.run[1]);
        wait1();                                       /* run 0 gathered (lane 1 holds nothing after it) */
        if (!nx) return;
        on1(true, [&] { ag(sp.run[1]); });
        codes(*nx, sp.run[0].r0, sp.run[0].n);         /* run 0's codes beside run 1's gather */
        wait1();
        codes(*nx, sp.run[1].r0, sp.run[1].n);
    };
    auto none = [](const SpSer::Run&) { return false; };
    const int64_t nl = (int64_t)m.layers.size();
    /* The layers before the PLE's (Flash-Next: blk.1) run replicated as step_serial runs them: the PLE reads every row of the
     * wide stream and its dilated convolution mixes rows, so the stream must be whole on every rank when it runs. Their
     * all-reduces: the block's own (one run), or reduce-scatter + all-gather of the whole step on the second lane (two runs:
     * every collective there; the raw halves fold as the exact all-reduce does). */
    const int64_t spl0 = m.ple_layer > 0 ? m.ple_layer : 0;
    const SpSer::Run all = [&] { SpSer::Run r; r.r0 = 0; r.n = T; return r; }();
    auto ar_all = [&] { on1(true, [&] { rs(all); }); on1(false, [&] { ag(all); }); wait1(); };
    /* The first read on every row: nothing is reduced before it, every rank holds the whole embedding. (Qwen tp8 S1: the
     * first layer this node holds; a proxy's synthetic boundary input) */
    /* (Qwen tp8 S3b-SP: a leader's tail part reads layer t first, from X2's whole rows) */
    const bool fol = pp.on && m.pp_rank == 1;
    const int64_t l0 = lead_tail_only ? (int64_t)m.pp_t : fol ? (int64_t)pp.s0 : (int64_t)m.lr_first;
    if (pp.seg_in) {
        /* Qwen tp8 S6: a segment from boundary s0 -- each rank's wide stream as its save left it (its run rows written, the
         * rest as they were): layer s0's read as conn() issues it after layer s0 - 1's write on this rank's rows of each run,
         * the block input gathered and coded on every row; the read computes the norm scales itself (the same bits as the
         * write's, docs/OPS.md `ss`); no gather of the stream at the boundary */
        const HyperConn& h0 = m.layers[(size_t)l0].hc_mix;
        if (sp.nrun == 1) {
            rd(h0, sp.run[0], false);
            ag(sp.run[0]);
            codes(h0, 0, T);
        } else {   /* conn()'s two-run order without the reduce-scatters and the writes */
            rd(h0, sp.run[0], false);
            on1(true, [&] { ag(sp.run[0]); });
            rd(h0, sp.run[1], false);
            wait1();
            on1(true, [&] { ag(sp.run[1]); });
            codes(h0, sp.run[0].r0, sp.run[0].n);
            wait1();
            codes(h0, sp.run[1].r0, sp.run[1].n);
        }
    } else {
    if (m.ple_layer == l0) m.ple.step(c, batch);
    bi_read_pieces(c, m, m.layers[(size_t)l0].hc_mix, T, false);
    }
    /* the wide stream's runs gathered raw on every rank (exact): before the output mixer / the MTP head, and before a crossing
     * (stage_send copies rank `sender`'s rows, and each rank wrote only its run rows) */
    auto gather_h = [&] {
        for (int k = 0; k < sp.nrun; ++k)
            on1(true, [&] {
                if (!sp_agh_fits(c, m, sp.run[k].n)) return;   /* fn4 KVCAP C1 */
                RAD_ISSUE_N(c, o_agh, sp.run[k].n * HN, brow_slice(m.b_h, sp.run[k].r0, sp.run[k].n, HN), RAD_NONE);
            });
        wait1();
    };
    for (int64_t li = 0; li < nl; ++li) {
        if (!lr_held(m, li)) continue;   /* Qwen tp8 S1: another node's layer (the stream passes on) */
        if (lead_tail_only && li < m.pp_t) continue;   /* Qwen tp8 S3b-SP: the head part ran these */
        if (fol && (li < pp.s0 || li >= pp.s1)) continue;   /* Qwen tp8 S6: another segment's layer */
        const Layer& l = m.layers[(size_t)li];
        const int64_t lnx = lr_after(m, li);   /* the next layer this node runs, -1 after the last */
        /* Qwen tp8 S3b-SP: the leader's cut (layer h - 1 -> t): the write on this rank's rows, the gather, X1 out, X2 in,
         * layer t's read on every row (as the step's first read) */
        const bool cut = pp.on && m.pp_rank == 0 && lnx == m.pp_t;
        /* Qwen tp8 S6: the follower's segment ends at boundary s1 after this layer's write on this rank's rows: the save */
        const bool seg_end = fol && pp.seg_out && lnx == pp.s1;
        if (cut && li < spl0) {
            rad_step_fail(c, "qwen4exp: two-node pipeline (Qwen tp8 S3b-SP): the cut after a layer SP-serial runs replicated "
                             "(before the PLE's) is not built");
            return;
        }
        if (li < spl0) {
            if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch, !l2); }
            else        l.gdn.step(c, batch, !l2);
            if (l2) ar_all();
            const bool ss_mix = l.hc_mix.write(c, T, 0, T);
            bt_pt(c, m, m.b_h, 0, T, "h_mix", (int)li);
            bi_read_pieces(c, m, l.hc_ffn, T, ss_mix);
            if (!l2) l.mlp.step(c, batch);
            else {
                for (int64_t r0 = 0; r0 < T; r0 += l.mlp.c.rows)
                    l.mlp.pass(c, T, r0, std::min<int64_t>(l.mlp.c.rows, T - r0), false);
                l.mlp.route_report(c);
                ar_all();
            }
            bool ss = l.hc_ffn.write(c, T, 0, T);
            if (l.proj.step(c, batch, 0, T)) ss = false;
            bt_pt(c, m, m.b_h, 0, T, "h_ffn", (int)li);
            if (lnx == m.ple_layer) { m.ple.step(c, batch); ss = false; }
            bi_read_pieces(c, m, m.layers[(size_t)lnx].hc_mix, T, ss);   /* (li < the PLE's layer: lnx exists) */
            continue;
        }
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch, false); }
        else        l.gdn.step(c, batch, false);
        conn(l.hc_mix, &l.hc_ffn, none, m.bt.on ? "h_mix" : nullptr, (int)li);
        for (int64_t r0 = 0; r0 < T; r0 += l.mlp.c.rows)
            l.mlp.pass(c, T, r0, std::min<int64_t>(l.mlp.c.rows, T - r0), false);
        l.mlp.route_report(c);
        if (m.bt.on) { bt_moe(c, m, T, (int)li); bt_pt(c, m, m.a_x.x, 0, T, "moe_part", (int)li); }   /* routing, shared arm, partial */
        conn(l.hc_ffn, lnx >= 0 && !cut && !seg_end ? &m.layers[(size_t)lnx].hc_mix : nullptr,
             [&](const SpSer::Run& r) { return l.proj.step(c, batch, r.r0, r.n); }, m.bt.on ? "h_ffn" : nullptr, (int)li);
        if (seg_end) {   /* (two runs: conn joined lane 1 after the last write) */
            pp_seg_io(c, m, pp.s1, true, T);
            return;
        }
        if (cut) {
            if (pp.head) {
                gather_h();
                if (!pp_cross(c, m, pp, m.op_x1, RAD_PP_X1, T)) return;
            }
            if (!pp.tail) return;
            if (!pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T)) return;
            bi_read_pieces(c, m, m.layers[(size_t)lnx].hc_mix, T, false);   /* the read computes the norm scales itself */
        }
    }
    /* the output mixer and the MTP head read every row of the wide stream (Qwen tp8 S3b-SP: the follower's X2 too) */
    gather_h();
    bt_pt(c, m, m.b_h, 0, T, "h_end", -1);   /* the gathered wide stream (BI tracer under RADIANCE_FN4_BITRACE_SP) */
    if (pp.on && m.pp_rank == 1) pp_cross(c, m, pp, m.op_x2, RAD_PP_X2, T);
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;

    /* Qwen tp8 S3a / S3b / S3b-SP: a pipeline node runs its part through the step form one machine would (step_pipelined,
     * step_serial_sp, step_serial); no encoder pass, no drafter pass on the follower, no snapshot inside a step */
    const PpPass pp = pp_pass(m);
    if (pp.on && (batch->enc || (m.pp_rank == 1 && batch->draft_pass != 0) || g_tipr[rad_rank(c)].row > 0)) {
        rad_step_fail(c, "qwen4exp: two-node pipeline (Qwen tp8 S3a): an encoder pass, a drafter pass on the follower or a "
                         "snapshot inside a step is not built");
        return;
    }
    /* Qwen tp8 S6: a follower segment the core asked for must be one this node declared; the leader's segments (the DS
     * line's M9, RADIANCE_PP_LEAD_SEG) are not built here, so the leader declares no boundary and is never sent one */
    if (pp.on && m.pp_rank == 1 && !pp_seg_ok(m, pp)) {
        rad_step_fail(c, "qwen4exp: two-node pipeline (Qwen tp8 S6): a segment of layers this node did not declare");
        return;
    }
    if (pp.on && m.pp_rank == 0 && (pp.info.part == RAD_PP_PART_SEG || pp.info.seg_l1 > 0)) {
        rad_step_fail(c, "qwen4exp: two-node pipeline (Qwen tp8 S6): a leader segment (RADIANCE_PP_LEAD_SEG) is not built");
        return;
    }

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* An MTP pass runs only the head. The trunk already left its wide stream in `h`; the head
     * turns the rows it wants into a draft token or one more position of its own attention
     * history, by the sign of `draft_pass` (rad_block_mtp_hc.h). Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `mtp.` head, or declare ran at max_spec 0");
            return;
        }
        /* Fill the head's own cos/sin rows. A draft round runs past the committed end (round k
         * writes K and V at position head_i + k - 1), at rows the preceding trunk step never
         * wrote. Without this, later rounds would rotate by stale rows: only acceptance drops, and
         * because the indexer reads the same table, the head's QSA selection and so the accepted
         * text shift at tie level at long context. A row depends only on its position, so
         * re-filling is idempotent. */
        if (m.op_rope_cs && T <= qk_fuse_now(qk_fuse_rows(g)) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
        m.mtp.qsa_win_cu = pipe_bounds(c, batch, m);
        m.mtp.msp_now    = (fn4_sproute_flag().load(std::memory_order_relaxed) & 1024) != 0;   /* lever MSP */
        m.mtp.msp_rs_now = (fn4_sproute_flag().load(std::memory_order_relaxed) & 2048) != 0;   /* MSP's embedding reduce-scatter */
        m.mtp.step(c, batch);
        return;
    }

    bt_begin(c, m, T);
    bi_segments(m, batch);   /* fn4 BI (hc_bi): the step's read segments */
    {   /* fn4 BI-dual: the step's row bands, and every trunk MoE layer's downs told (exl3_dual) */
        const bool dual = m.op_band && fn4_dual_flag().load(std::memory_order_relaxed) > 0;
        if (dual)
            RAD_ISSUE_N(c, m.op_band, T, praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(m.b_band, T));
        if (m.op_band)
            for (const Layer& l : m.layers)
                if (lr_held(m, &l - m.layers.data())) l.mlp.set_band(dual ? m.b_band : 0);   /* (Qwen tp8 S1: held layers) */
    }
    /* qfn-on-ds lever X3GD: the step's longest sequence, for the fused EXL3 chain past 16 rows (rad_block_moe_fp8.h) */
    for (const Layer& l : m.layers)
        if (lr_held(m, &l - m.layers.data())) l.mlp.set_qmax(batch->max_q_len);
    const int32_t* pcu = pipe_bounds(c, batch, m);
    SpSer sp;
    if (g_tipr[rad_rank(c)].row > 0 && !pipelined(m, T)) {   /* lever TIP: only a pipelined step stages its rows */
        rad_step_fail(c, "qwen4exp: lever TIP: a snapshot inside a step that is not pipelined (one pass or less, or the "
                         "pipelined form refused): its state rows would be the step end's");
        return;
    }
    if (pp.on) {   /* Qwen tp8 S3a: a pipeline node's part; the mixer and the head after the leader's tail only */
        if (pipelined(m, T))               step_pipelined(c, batch, m, pcu, pp);     /* Qwen tp8 S3b: the pipelined form by part */
        else if (spser_plan(m, batch, sp)) step_serial_sp(c, batch, m, pcu, sp, pp);   /* S3b-SP: SP-serial by part */
        else                               step_serial(c, batch, m, pcu, pp);
        if (!(m.pp_rank == 0 && pp.tail)) {   /* the follower, or a head part: nothing after its send */
            bt_flush(c, m, batch);
            bi_cuts() = BiCuts{};
            bi_wide_rows() = 0;
            return;
        }
    }
    else if (pipelined(m, T))          step_pipelined(c, batch, m, pcu);
    else if (spser_plan(m, batch, sp)) step_serial_sp(c, batch, m, pcu, sp);   /* fn4 SP-serial */
    else                               step_serial(c, batch, m, pcu);

    if (!pipelined(m, T) && !sp.nrun) bt_pt(c, m, m.b_h, 0, T, "h_end", -1);   /* step_serial's wide stream (step_serial_sp: its own) */
    bi_read_pieces(c, m, m.mixer, T);
    bt_pt(c, m, m.a_x.x, 0, T, "final", -1);

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        if (!bi_logits(c, m, batch))
            RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                        brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                        brows(m.b_logits, batch->n_out));
        bt_pt(c, m, m.b_logits, 0, batch->n_out, "logits", -1, 1);
    }
    if (m.hcp && !m.hcp_noted && m.g.rank == 0 && (m.hcp_n > 0 || !m.mixer.op_read_sp)) {   /* lever HCP: said once */
        m.hcp_noted = true;
        if (m.mixer.op_read_sp)
            fprintf(stderr, "radiance: qwen4exp: lever HCP (sp_route bit 8192): a prefill segment past the prompt's first interval "
                            "reads its connections at a full piece's K split (op_read_sp, kc_rows %lld) whatever its rows -- first "
                            "in a %lld-row step at position %lld, %llu reads pinned so far\n",
                    (long long)m.mixer.c.read_rows, (long long)batch->n_tok, (long long)m.hcp_pos_step,
                    (unsigned long long)m.hcp_n);
        else
            fprintf(stderr, "radiance: qwen4exp: lever HCP (sp_route bit 8192) is INERT: the connections have no op_read_sp "
                            "(RADIANCE_FN4_SPHC=1 at start with a span declares it)\n");
    }
    bt_flush(c, m, batch);
    bi_cuts() = BiCuts{};   /* fn4 BI: the cuts are this trunk step's only */
    bi_wide_rows() = 0;
}

/* The default draft depth, used by `--num-speculative-tokens auto` and by rad-convert.
 *
 * 3 is an operating point, not a model property (`mtp_num_hidden_layers` only says the one-layer
 * head exists), so `draft_depth_fixed` stays 0. Throughput peaks at 3 on this model: per-position
 * acceptance stays high at every depth, but each extra verify row costs real time because the
 * weights do not fit device memory (more experts streamed, more n-gram rows faulted from the
 * mapped table). */
uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    if (!c || !batch) return 0;
    /* lever TIP: a step with a snapshot inside issues per its row (two launches a state op and staging calls), which no key
     * carries: never recorded, never a replay */
    if (g_tipr[rad_rank(c)].row > 0) return 0;
    if (g_hsr[rad_rank(c)].stage) return 0;   /* Qwen tp8 S7b: a head part that stages a snapshot's rows */
    if (batch->enc) return 1;
    /* Mixed batches split the query rows into decode and prefill subsets before attention; the
     * whole-batch host summary cannot key those subset geometries exactly. */
    if (batch->phase == RAD_PHASE_MIXED) return 0;

    Model& m = g_model[rad_rank(c)];
    /* Qwen tp8 S3c (round 31): a pipeline node's passes are recorded like one machine's. What a node issues beyond the shape this
     * key reads is (a) its part -- the core's Ctx::pass_key mixes RadPpInfo.part (HEAD / TAIL) itself, WHOLE (one lane, the
     * follower) mixes nothing -- and (b) its lane's rings, which the stage ops take as operands: each lane's batch is its own
     * slab, whose pointers the core keys (engine_lanes.cpp), so a lane never plays another's tape; the ring counters are read
     * at run time (r4d_stage.hip), so a played crossing waits for its own payload. The node, the cut and the held layers are
     * fixed at declare. (Was 0 on a node since S3a.) */
    /* Decode, verify and draft passes (the MTP head attends over the same KV group): every
     * attention launch covers all the pass's sequences, and the shape fields the core keys (n_tok,
     * n_seq, max_q_len, n_spec, draft_pass) fix each launch's q_len band and units. The live KV
     * length enters a split launcher's count only through the cap of eight 48-key tiles a split
     * (rad_prefill_splits_law; every head_dim 256 launcher has P_TILE 48), and from 64 splits up
     * the cap no longer binds. Keying that cap level lets these passes be recorded while each launch
     * still splits by its exact live length: answering 0 left them unrecorded (issue #402), and the
     * core's keyed bound for a plugin without the callback rounds the length up a bucket, which split
     * a short single-stream verify into several partials with a merge (x1 203 against 219 tok/s). */
    if (batch->phase != RAD_PHASE_PREFILL || batch->draft_pass != 0 || batch->n_spec > 0) {
        if (!m.kv_full || !used_max_h || !batch->kv) return 0;
        int group = -1;
        for (int i = 0; i < batch->n_kv_groups; ++i)
            if (batch->kv[i].group == m.kv_full) { group = i; break; }
        if (group < 0) return 2;  // no full-attention KV in this pass: nothing splits by length
        const int used = used_max_h[group];
        if (used <= 0) return 0;  // Missing host summary is not the same as an empty KV cache.
        constexpr int tile = 48;
        const int ctx = (m.g.max_ctx > 0 && used >= m.g.max_ctx) ? (int)m.g.max_ctx : used;
        const int cap = std::min(((ctx + tile - 1) / tile) / 8, 64);
        /* fn4 att_bi (12.11.2026, 205: "draft pass 0 at 16 tokens issued differently from its recording" -- a decode
         * sequence dense when recorded, sparse when replayed): under att_bi each sequence of a multi-sequence ordinary pass
         * takes the dense or sparse path by ITS OWN context against qsa_exact_to (rad_block_attn_gated_fp8.h), so the
         * launches depend on every sequence's side of that line, which the length cap does not carry. The key takes the
         * pattern; without host lengths the pass is not recorded. */
        const int abf = rad::arch::fn4_attbi_flag().load(std::memory_order_relaxed);
        if (batch->draft_pass == 0 && batch->n_seq > 1 && abf > 0) {
            const int32_t* su = batch->kv[group].seqused_h;
            int64_t ex = 0;
            for (const Layer& layer : m.layers)   /* (Qwen tp8 S1: a held layer's) */
                if (layer.full && lr_held(m, &layer - m.layers.data())) { ex = layer.attn.qsa_exact_to; break; }
            if (ex > 0) {
                if (!su) return 0;
                /* att_bi 4: the pattern by each sequence's alone bound (fn4_attbi_sparse), its rows the pass's uniform
                 * decode / verify rows; a pass whose rows do not split evenly is not recorded */
                const int64_t rows = batch->n_tok % batch->n_seq == 0 ? batch->n_tok / batch->n_seq : 0;
                if (abf >= 4 && rows <= 0) return 0;
                uint64_t h = 1469598103934665603ull;
                for (int64_t s = 0; s < batch->n_seq; ++s) {
                    const bool sp = abf >= 4 ? rad::arch::fn4_attbi_sparse(abf, (int64_t)su[s], rows, ex, m.g.max_ctx)
                                             : (int64_t)su[s] > ex;
                    h = (h ^ (uint64_t)(sp ? 2 : 1)) * 1099511628211ull;
                }
                if (abf >= 4) h ^= 0x9e3779b97f4a7c15ull;   /* not a 1..3 pattern's key */
                return (h & ~0xffull) | (uint64_t)(3 + cap);
            }
        }
        return 3 + (uint64_t)cap;
    }

    bool has_full_attention = false;
    for (const Layer& layer : m.layers) {
        if (!layer.full || !lr_held(m, &layer - m.layers.data())) continue;   /* (Qwen tp8 S1: held layers) */
        if (layer.attn.kv_block != 4) return 0;
        has_full_attention = true;
    }
    if (!has_full_attention) return 1;
    if (m.g.head_dim != 256 || (m.g.gqa() != 6 && m.g.gqa() != 12) || !m.kv_full ||
        !used_max_h || !batch->kv || batch->n_kv_groups <= 0)
        return 0;

    int group = -1;
    for (int i = 0; i < batch->n_kv_groups; ++i)
        if (batch->kv[i].group == m.kv_full) { group = i; break; }
    if (group < 0) return 0;

    /* Qwen always passes cu_seqlens, so fill_paged takes q_len = min(band, total_q). That equals
     * n_tok only for one sequence; a ragged or multi-sequence launch's band is not the host group
     * max, and mixed decode-prefix / prefill-suffix launches use the suffix's own KV lengths. */
    if (batch->n_seq != 1 || batch->n_tok <= 0 || m.g.max_ctx <= 0) return 0;
    const int q_len = (int)batch->n_tok;
    const int total_q = (int)batch->n_tok;
    const int num_seqs = 1;
    const int kv_heads = (int)m.g.n_head_kv;
    const int used = used_max_h[group];
    if (used <= 0) return 0;  // Missing host summary is not the same as an empty KV cache.
    const int wgps = rad_ctx_wgp_count(c);
    if (wgps <= 0) return 0;
    constexpr int tile = 48, nwarps = 24;
    const int bq = nwarps * 16 / m.g.gqa();
    const int splits = rad_prefill_splits_for_live(bq, tile, q_len, total_q, num_seqs,
                                                    kv_heads, m.g.max_ctx, used, wgps);
    return rad_prefill_grid_key(q_len, bq, splits, kv_heads, num_seqs, nwarps * 32);
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    /* The gated attention block declares the streamed store and the resolve
     * (rad_block_attn_gated_fp8.h). */
    out->kv_stream_ok = 1;
    out->draft_depth = 0;
    if (meta_int2(meta, "qwen4exp.mtp_num_hidden_layers", "mtp_num_hidden_layers", 0) > 0)
        out->draft_depth = 3;
    out->shape_probe_ok = 1;      /* declare() writes scratch under shape_probe */
    /* --kv-cache-dtype fp8_v4 / k6_v4: libr4d has their rows for the QSA layers' geometry
     * (head_dim 256, twelve queries per KV head; gqa6 with 4-token pages at four ranks), so other
     * geometries are refused at startup rather than in declare. */
    {
        const long long hd  = meta->head_dim ? meta->head_dim
                                             : (meta->n_head ? meta->n_embd / meta->n_head : 0);
        const long long nkv = meta->n_head_kv ? meta->n_head_kv : meta->n_head;
        if (hd == 256 && nkv > 0 && meta->n_head == 12 * nkv)
            out->kv_quant_ok = (1 << RAD_KVQ_V4) | (1 << RAD_KVQ_K6V4);
    }
    /* The gated residual applies a control vector after every block's write (HcProject). */
    out->control_vector_ok = 1;
    /* Chained launches pay with the MTP drafter: the speculative step's small kernels wait on the
     * previous one on the card instead of behind a queue barrier (libr4d/r4d_chain.h). On 2x R9700,
     * sampled at temperature 1 with a 25k prompt, MTP depth 3, with the chained hc_read forms:
     * 1/2/4/8/16 streams 174/291/458/657/851 -> 207/339/493/647/843 tok/s. */
    out->chain_pays = 0;   /* fn4 9.11.: off -- with the MTP drafter chained launches cost draft acceptance under
                            * concurrent multi-turn load (113: cc_sim u16 0.259 against 0.351) and gain <= 1 % */
    /* fn4 10.11.: RADIANCE_FN4_CHAIN=1 at start turns them on again for an A/B across restarts (the core decides
     * chaining at bring-up and the rows resolve with it, so it is not a runtime switch); natural-mode data
     * since showed no acceptance cost (0.922 vs 0.920). */
    if (fn4_env_on("RADIANCE_FN4_CHAIN")) out->chain_pays = 1;
    return RAD_OK;
}

/* lever TIP (the engine's hooks, core/runtime/fn4_switch.cpp): rad_tip_set names the snapshot row of the step this rank
 * issues next (0 after it); -1 where the start did not declare the lever (RADIANCE_FN4_TIP) or the rank has no pipelined
 * prefill. rad_tip_free lists the stateful groups a resume at a piece start does not read (the QSA tail ring), which the
 * engine saves as the step leaves them. */
int tip_set(int rank, long long row, int (*stage)(void*, int, int, void*), void* u) {
    if (rank < 0 || rank >= MAX_RANKS) return -1;
    /* Qwen tp8 S7b: row -1 = a chunk-end snapshot staged in this leader's HEAD part (pp_cross); a head layer with attention keeps
     * a QSA tail ring this does not stage: refused */
    if (row == -1) {
        const Model& m = g_model[rank];
        if (!m.pp || m.pp_rank != 0 || m.pp_h <= 0 || !stage) return -1;
        for (int64_t l = 0; l < m.pp_h && l < (int64_t)m.layers.size(); ++l)
            if (m.layers[(size_t)l].full) return -1;
        g_hsr[rank].stage = stage;
        g_hsr[rank].u = u;
        return 1;
    }
    if (row == 0) g_hsr[rank] = HeadSnapRank{};
    if (!fn4_tip_env()) return -1;
    if (!g_model[rank].pipe_on && row != 0) return -1;
    TipRank& t = g_tipr[rank];
    t.row = row > 0 ? (int64_t)row : 0;
    t.stage = row > 0 ? stage : nullptr;
    t.u = row > 0 ? u : nullptr;
    return 1;
}
int tip_free(int rank, int* groups, int max) {
    if (rank < 0 || rank >= MAX_RANKS || !fn4_tip_env() || !g_model[rank].pipe_on) return -1;
    int n = 0;
    if (g_model[rank].qsacfg.kv_tail && groups && n < max) groups[n++] = (int)g_model[rank].qsacfg.kv_tail;
    return n;
}

}  /* namespace qwen4exp_fp8 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwen4exp_fp8)
RAD_ARCH_PLUGIN(qwen4exp_fp8, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next): hybrid Gated DeltaNet / gated attention, a gated "
                "residual four streams wide and 512 routed experts, from a dense bf16 checkpoint "
                "quantised ahead of time by a recipe")
/* fn4 runtime switch (the core's runtime/fn4_switch.h, between steps, every rank parked): "gather_inv"
 * picks MoeFP8's inverse twins when the blocks were declared switchable (RADIANCE_FN4_SWITCHABLE=1 at
 * start); refused (-1) otherwise. */
/* qfn-on-ds: this plugin's switches are the Qwen line's, which the core checks by its own ranges and never asks a plugin
 * about; for every other name (the ds41 line's plugin switches, checked through rad_fn4_accepts) it is not the owner. */
extern "C" int rad_fn4_accepts(const char*, int) { return 0; }
extern "C" int rad_fn4_switch(const char* name, int value) {
    if (name && std::strcmp(name, "moe_pair") == 0) {   /* fn4 P2x (step_pipelined; RADIANCE_FN4_P2X=1 declares) */
        if (value < 0 || value > 4) return -1;
        /* qfn-on-ds P2S: 3 (the pair order, each piece's own pass) needs no two-piece buffers; 1 / 2 / 4 run a joint pass */
        if (value && value != 3 && !qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_P2X")) return -1;
        qwen4exp_fp8::fn4_pair_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "gnq8") == 0) {   /* qfn-on-ds lever GNQ8: the GDN prefill gated norm writes the codes (RADIANCE_FN4_GNQ8 declares) */
        if (value < 0 || value > 1) return -1;
        if (value && !rad::arch::fn4_gnq8_declared()) return -1;
        rad::arch::fn4_gnq8_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "arw_rows") == 0) {   /* fn4 ARW: the fused exact write's row cap (0 = none) */
        if (value < 0 || value > 4096) return -1;
        rad::arch::fn4_arw_rows().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "kvs_bi") == 0) {   /* fn4 kvs_bi: KV streaming exact under load (bit 1 here, bit 2 the core's) */
        if (value < 0 || value > 3) return -1;
        rad::arch::fn4_kvsbi_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "qsa_split") == 0) {   /* fn4 qsa_split (rad_qsa.h; RADIANCE_FN4_QSA_SPLIT declares) */
        if (value < 0 || value > 3) return -1;
        if (value & ~rad::arch::fn4_qsasplit_declared()) return -1;
        rad::arch::fn4_qsasplit_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "att_bi") == 0) {   /* fn4 BI: a step's prefill sequences' attention, each alone-shaped */
        if (value < 0 || value > 6) return -1;   /* 3: + the dense decode sequences in one pinned launch; 4: + the alone bound's path; 5: + dense steps; 6: + the depth-3 sparse law at every depth */
        rad::arch::fn4_attbi_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "bi_wide") == 0) {   /* fn4 BI wide: the forms above 64 decode rows (mask; RADIANCE_FN4_BI_WIDE declares) */
        if (value < 0 || value > 15) return -1;
        if (value & ~rad::arch::fn4_biwide_start()) return -1;   /* only the forms the start value declared */
        rad::arch::fn4_biwide_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "bi_seg") == 0) {   /* fn4 BI: the trunk's linears cut at the step's segments (2: + the lm_head; 3: + its decode rows in one launch at any depth) */
        if (value < 0 || value > 3) return -1;
        qwen4exp_fp8::fn4_biseg_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "hc_bi") == 0) {   /* fn4 BI: the reads cut at the step's segments (libr4d takes it too) */
        if (value < 0 || value > 5 || value == 4) return -1;
        qwen4exp_fp8::fn4_hcbi_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "sp_serial") == 0) {   /* fn4 SP-serial (RADIANCE_FN4_SPSERIAL declares; 2 two halves on the second lane) */
        if (value < 0 || value > 2) return -1;
        if (value && !qwen4exp_fp8::fn4_spser_env()) return -1;
        qwen4exp_fp8::fn4_spser_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "hc_spprobe") == 0) {   /* fn4 SP-hc speed probe (step_pipelined) */
        if (value < 0 || value > 1) return -1;
        qwen4exp_fp8::fn4_spprobe_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "mtp_flat") == 0) {   /* fn4 MTP-FLAT (RADIANCE_FN4_MTPFLAT=1 declares) */
        if (value < 0 || value > 1) return -1;
        if (value && !qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_MTPFLAT")) return -1;
        rad::arch::fn4_mtpflat_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "mtp_dec") == 0) {   /* qfn-on-ds lever MTD: the MTP draft passes' decode forms (rad_block_mtp_hc.h) */
        if (value < 0 || value > 31) return -1;   /* bits 4 / 8 / 16: libr4d's (levers MTE / W2K / W2N) */
        rad::arch::fn4_mtpdec_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "sp_ple") == 0) {   /* fn4 SP-PLE (qwen/SP-SERIAL-EXACT.md): SP-hc keeps the layers before the PLE's replicated */
        if (value < 0 || value > 1) return -1;
        qwen4exp_fp8::fn4_spple_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (name && std::strcmp(name, "sp_route") == 0) {   /* fn4 SP-route (qwen/prefill; RADIANCE_FN4_SP_ROUTE declares bit 1); bits 2 / 4: PF probes */
        if (value < 0 || value > 65535) return -1;   /* 32768: lever QRS (the QSA indexer's windows a quarter a rank + gathers; the same bytes; RADIANCE_FN4_QRS=1 declares); 16384: PF probe QSP (a > 256-row step's QSA score + select twice; the same bytes; lever QRS's bound); 8192: lever HCP (a prefill segment past the first interval reads its connections at a full piece's K split, op_read_sp; other bytes for partial pieces / tail and hit steps); bit 16: SPO-tail, 32: PF-ORDER, 64: PF probe ATP (attention twice), 128: PF probe X3P (EXL3 experts twice), 256: PF probe NOWAIT (lane 0 never waits for lane 1 in a pipelined step; wrong bytes), 512: lever PSP (the partial last piece SP + P2S's partial pair; the same bytes), 1024: lever MSP (a mid-chunk MTP history pass's per-row work a piece a rank + the block input gathered; the same bytes; RADIANCE_FN4_MSP=1 declares), 2048: with 1024, MSP's embedding by a raw reduce-scatter onto the rank's piece instead of the all-reduce (the same bytes), 4096: lever GPF (a two-piece step's mixer a piece at a time only where its last piece is wider than 64 rows; other bytes for (R, <= 64)-row steps) */
        if ((value & 1) && !qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_SP_ROUTE")) return -1;
        qwen4exp_fp8::fn4_sproute_flag().store(value, std::memory_order_relaxed);
        rad::arch::fn4_attprobe_flag().store((value & 64) ? 1 : 0, std::memory_order_relaxed);
        rad::arch::fn4_x3probe_flag().store((value & 128) ? 1 : 0, std::memory_order_relaxed);
        rad::arch::fn4_qsaprobe_flag().store((value & 16384) ? 1 : 0, std::memory_order_relaxed);   /* PF probe QSP */
        rad::arch::fn4_qrs_flag().store((value & 32768) ? 1 : 0, std::memory_order_relaxed);       /* lever QRS */
        return 1;
    }
    if (name && std::strcmp(name, "hc_sp") == 0) {   /* fn4 SP-hc proper (step_pipelined; RADIANCE_FN4_SPHC=1 declares; 2 pipelined order) */
        if (value < 0 || value > 2) return -1;
        if (value && !qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_SPHC")) return -1;
        qwen4exp_fp8::fn4_sp_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (!name || std::strcmp(name, "gather_inv") != 0) return 0;
    if (!qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_SWITCHABLE")) return -1;
    rad::arch::fn4_gather_inv_flag().store(value ? 1 : 0, std::memory_order_relaxed);
    return 1;
}
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch,
                                      const int32_t* used_max_h) {
    return qwen4exp_fp8::pass_key(c, batch, used_max_h);
}
extern "C" int rad_tip_set(int rank, long long row, int (*stage)(void*, int, int, void*), void* u) {
    return qwen4exp_fp8::tip_set(rank, row, stage, u);
}
extern "C" int rad_tip_free(int rank, int* groups, int max) { return qwen4exp_fp8::tip_free(rank, groups, max); }
#endif
