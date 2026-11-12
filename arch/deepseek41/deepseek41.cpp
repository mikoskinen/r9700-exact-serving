/* deepseek41 -- DeepSeek-V4.1 (GGUF `general.architecture: deepseek41`, the layout llama.cpp's
 * converter writes): `hc` hyper-connection residual streams over latent attention layers that
 * attend a sliding window of raw keys plus, past the first layers, a two-level sparse top-k of
 * compressed keys; sqrt(softplus)-routed MoE with one shared expert; Engram n-gram lookups written
 * into the streams at a few layers. Every dimension comes from the checkpoint's metadata, so the
 * same graph serves the released model and any reduced one (tiny/ds41-tiny.gguf in the port).
 *
 * The numerics follow the official inference/model.py (DeepSeek's release): the window keys, the
 * compressed keys and the index keys are quantised and dequantised exactly where the reference
 * does it (quant_roundtrip), so the bf16 caches hold the reference's values.
 *
 * Per layer, in issue order (docs: the port's PLAN.md, "Phase 2 plumbing design"):
 *   engram (at ple layers)   token_map -> ngram_ids -> embed_lookup -> cast to the card -> gemm ->
 *                            stream_gate_add (every layer's ngram_ids issued before the first
 *                            block, see step())
 *   attention connection     mhc_mix (this layer's gates) ; mhc_collapse (the previous gates)
 *   attention                q path, window key into the window cache, the compressor at a
 *                            kv-source layer (qsa_work / softmax_pool, or one entry a token at
 *                            ratio 1), the indexer at an index-source layer, attn_sparse_sink,
 *                            the inverse rotation, the grouped output projection ; mhc_post
 *   ffn connection           mhc_mix ; mhc_collapse ; MoeFP8 (norm, router, experts) ; mhc_post
 * then mhc_collapse with the last gates, the final norm and the head.
 *
 * The checkpoint's DSpark drafter (mtp.0 .. mtp.{n-1}, when the container carries its stages as
 * blk.{n_layer}.. with `nextn_predict_layers`) runs on the block-drafter protocol: see struct Dsp. */
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_deepseek41_chat.h>

#include "ds41_engram_dedup.h"
#include "ds41_index_split.h"
#include "ds41_psk_cone.h"
#include "ds41_walk_geom.h"
#include "rad_spechub.h"   /* lever SD: the coupling wire (RAD_MTP_COUPLE_*), --draft-coupling */

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>   /* getpid: lever BI's tracer files */

/* The engine's runtime switches (abi/rad_runtime.h), weak: a host without them (an older core, a tool)
 * reads every switch as off. The DSpark extras (ds41-acc) are issued only while theirs is on. */
extern "C" int rad_fn4_get(const char* name, int* value) __attribute__((weak));
/* ...and the drafter's extras' declarations (abi/rad_builder.h), weak too: a host built before them (rad-info,
 * rad-convert or rad-kbench of an older image) still loads this plugin, and a serve there declares no extras. */
extern "C" int rad_drafter_confidence(RadBuilder* b, rad_buf conf, int64_t pitch) __attribute__((weak));
extern "C" int rad_drafter_alternatives(RadBuilder* b, rad_buf alt, int64_t n_alt, int64_t pitch) __attribute__((weak));
static int ds41_switch(const char* name) {
    int v = 0;
    return rad_fn4_get && rad_fn4_get(name, &v) > 0 ? v : 0;
}

namespace deepseek41 {
using namespace rad::arch;

/* RAD_ARCH_TRY that names the call that failed: a refused declaration otherwise surfaces only as
 * the plugin's return code. */
#define DS_TRY(e)                                                                                \
    do {                                                                                         \
        const int rc_ = (e);                                                                     \
        if (rc_ < 0) {                                                                           \
            fprintf(stderr, "radiance: deepseek41: %s refused (%d)\n", #e, rc_);                  \
            return rc_;                                                                          \
        }                                                                                        \
    } while (0)

/* ------------------------------------------------------------------ the configuration */
struct Cfg {
    int64_t n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, rope = 0, q_lora = 0;
    int64_t o_groups = 0, o_rank = 0, window = 0;
    int64_t n_expert = 0, top_k = 0, n_ff = 0, n_shared = 0;
    double  route_scale = 1.0, swiglu_limit = 0.0;
    int     norm_topk = 1;
    int64_t ih = 0, idim = 0, itopk = 0, cand = -1, cand_blocks = 0, cand_block = 0;
    int64_t hc = 0, hc_iters = 0;
    double  hc_eps = 0, eps = 0;
    double  theta = 0, comp_theta = 0, yarn_factor = 0, yarn_low = 0, yarn_high = 0;
    std::vector<int64_t> ratio, kv_src, idx_src, ple;
    int64_t ngram = 0, ple_heads = 0, ple_dim = 0, ple_rows = 0, pad = 0;
    int64_t n_vocab = 0;
    /* derived, per layer */
    std::vector<int> src_of, idx_of;
    bool is_src(int l) const { for (int64_t s : kv_src) if (s == l) return true; return false; }
    bool is_idx(int l) const { for (int64_t s : idx_src) if (s == l) return true; return false; }
    int  ple_index(int l) const {
        for (size_t i = 0; i < ple.size(); ++i) if (ple[i] == l) return (int)i;
        return -1;
    }
};

static std::string key(const char* k) { return std::string("deepseek41.") + k; }
/* DeepSeek's release states the same values in its config.json (text_config's keys, which the
 * core also reads bare). Each deepseek41.* key read here and the release's name for it; a key the
 * release does not state (the Engram's bucket sizes, derived from its primes) has none. */
static const char* release_key(const char* k) {
    static const struct { const char* gguf; const char* hf; } t[] = {
        { "rope.dimension_count", "qk_rope_head_dim" },
        { "attention.q_lora_rank", "q_lora_rank" },
        { "attention.output_group_count", "o_groups" },
        { "attention.output_lora_rank", "o_lora_rank" },
        { "attention.sliding_window", "sliding_window" },
        { "expert_feed_forward_length", "moe_intermediate_size" },
        { "expert_weights_scale", "routed_scaling_factor" },
        { "expert_weights_norm", "norm_topk_prob" },
        { "swiglu_clamp_exp", "swiglu_limit" },
        { "attention.indexer.head_count", "index_n_heads" },
        { "attention.indexer.key_length", "index_head_dim" },
        { "attention.indexer.top_k", "index_topk" },
        { "attention.indexer.candidate_source_layer", "candidate_source_layer_id" },
        { "attention.indexer.candidate_top_k_blocks", "candidate_topk_blocks" },
        { "attention.indexer.candidate_block_size", "candidate_block_size" },
        { "hyper_connection.count", "hc_mult" },
        { "hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters" },
        { "hyper_connection.epsilon", "hc_eps" },
        { "attention.compress_rope_freq_base", "compress_rope_theta" },
        { "attention.compress_ratios", "compress_ratios" },
        { "attention.kv_source_layers", "kv_source_layer_ids" },
        { "attention.indexer.source_layers", "index_source_layer_ids" },
        { "ple.layers", "engram_layer_ids" },
        { "ple.ngram_size", "engram_max_ngram_size" },
        { "ple.heads_per_ngram", "engram_n_heads" },
        { "embedding_length_per_layer_input", "engram_head_dim" },
        { "engram.pad_token_id", "engram_pad_token_id" },
        { "nextn_predict_layers", "num_nextn_predict_layers" },
        { "dspark.block_size", "dspark_block_size" },
        { "dspark.noise_token_id", "dspark_noise_token_id" },
        { "dspark.target_layers", "dspark_target_layer_ids" },
        { "dspark.markov_rank", "dspark_markov_rank" },
        { "dspark.expert_count", "dspark_n_routed_experts" },
        { "dspark.expert_used_count", "dspark_num_experts_per_tok" } };
    for (const auto& e : t)
        if (std::strcmp(e.gguf, k) == 0) return e.hf;
    return nullptr;
}
/* deepseek41.<k>, else the release's spelling of it; null when neither is there. */
static const char* gs(const RadModelMeta* m, const char* k) {
    const char* s = rad_meta_gets(m, key(k).c_str(), nullptr);
    if (!s)
        if (const char* r = release_key(k)) s = rad_meta_gets(m, r, nullptr);
    return s;
}
static int64_t gi(const RadModelMeta* m, const char* k, int64_t d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtoll(s, nullptr, 10) : d;
}
static double gf(const RadModelMeta* m, const char* k, double d) {
    const char* s = gs(m, k);
    return s && *s ? std::strtod(s, nullptr) : d;
}
/* Whether the checkpoint is DeepSeek's release (config.json, inference/model.py's tensor names)
 * rather than a GGUF in llama.cpp's names: it states no deepseek41.block_count. */
static bool is_release(const RadModelMeta* m) {
    return rad_meta_gets(m, key("block_count").c_str(), nullptr) == nullptr &&
           rad_meta_gets(m, "num_hidden_layers", nullptr) != nullptr;
}
/* An integer array spelled out, comma-separated (core/format/checkpoint.cpp, meta_from_gguf) or
 * space-separated (a config.json array). */
static std::vector<int64_t> int_list(const char* s) {
    std::vector<int64_t> v;
    if (!s || !*s || *s == '[') return v;
    while (*s) {
        char* e = nullptr;
        v.push_back(std::strtoll(s, &e, 10));
        if (e == s) return {};
        s = *e == ',' ? e + 1 : e;
    }
    return v;
}
static std::vector<int64_t> gl(const RadModelMeta* m, const char* k) { return int_list(gs(m, k)); }

static int read_cfg(Cfg& c, const RadModelMeta* m) {
    c.n_layer = m->n_layers;
    c.n_embd = m->n_embd;
    c.n_head = m->n_head;
    c.head_dim = m->head_dim;
    c.n_vocab = m->n_vocab;
    c.rope = gi(m, "rope.dimension_count", 0);
    c.q_lora = gi(m, "attention.q_lora_rank", 0);
    c.o_groups = gi(m, "attention.output_group_count", 0);
    c.o_rank = gi(m, "attention.output_lora_rank", 0);
    c.window = gi(m, "attention.sliding_window", 0);
    c.n_expert = m->n_expert;
    c.top_k = m->n_expert_used;
    c.n_ff = gi(m, "expert_feed_forward_length", 0);
    c.n_shared = m->n_expert_shared;
    c.route_scale = gf(m, "expert_weights_scale", 1.0);
    c.norm_topk = (int)gi(m, "expert_weights_norm", 1);
    {
        /* One clamp for every layer and arm; a per-layer schedule is refused. */
        const std::vector<int64_t> dummy;
        const char* s = gs(m, "swiglu_clamp_exp");
        c.swiglu_limit = s && *s && *s != '[' ? std::atof(s) : 10.0;
    }
    c.ih = gi(m, "attention.indexer.head_count", 0);
    c.idim = gi(m, "attention.indexer.key_length", 0);
    c.itopk = gi(m, "attention.indexer.top_k", 0);
    c.cand = gi(m, "attention.indexer.candidate_source_layer", -1);
    c.cand_blocks = gi(m, "attention.indexer.candidate_top_k_blocks", 0);
    c.cand_block = gi(m, "attention.indexer.candidate_block_size", 0);
    c.hc = gi(m, "hyper_connection.count", 0);
    c.hc_iters = gi(m, "hyper_connection.sinkhorn_iterations", 0);
    c.hc_eps = gf(m, "hyper_connection.epsilon", 0.0);
    c.eps = m->rms_eps;
    c.theta = m->rope_theta;
    c.comp_theta = gf(m, "attention.compress_rope_freq_base", 0.0);
    c.ratio = gl(m, "attention.compress_ratios");
    c.kv_src = gl(m, "attention.kv_source_layers");
    c.idx_src = gl(m, "attention.indexer.source_layers");
    c.ple = gl(m, "ple.layers");
    c.ngram = gi(m, "ple.ngram_size", 0);
    c.ple_heads = (c.ngram - 1) * gi(m, "ple.heads_per_ngram", 0);
    c.ple_dim = gi(m, "embedding_length_per_layer_input", 0);
    c.pad = gi(m, "engram.pad_token_id", -1);
    for (int64_t v : gl(m, "ple.head_vocab_sizes")) c.ple_rows += v;
    /* The release states each Engram table's rows; their sum is the joined table's. */
    if (!c.ple_rows)
        for (int64_t v : int_list(rad_meta_gets(m, "engram_num_embeddings", nullptr))) c.ple_rows += v;
    /* The compressed layers' rotary: YaRN with the bounds the reference derives from the
     * training length (model.py, precompute_freqs_cis). */
    c.yarn_factor = rad_meta_getf(m, "rope_scaling.factor", 0.0);
    const double orig = rad_meta_getf(m, "rope_scaling.original_max_position_embeddings", 0.0);
    const double bf = rad_meta_getf(m, "rope_scaling.beta_fast", 32.0);
    const double bs = rad_meta_getf(m, "rope_scaling.beta_slow", 1.0);
    if (c.yarn_factor > 1.0 && orig > 0 && c.comp_theta > 0 && c.rope > 0) {
        auto corr = [&](double r) {
            return (double)c.rope * std::log(orig / (r * 2.0 * M_PI)) / (2.0 * std::log(c.comp_theta));
        };
        c.yarn_low = std::max(std::floor(corr(bf)), 0.0);
        c.yarn_high = std::min(std::ceil(corr(bs)), (double)c.rope - 1);
    }

    const struct { const char* what; int64_t v; } need[] = {
        { "block_count", c.n_layer }, { "embedding_length", c.n_embd },
        { "attention.head_count", c.n_head }, { "attention.key_length", c.head_dim },
        { "rope.dimension_count", c.rope }, { "attention.q_lora_rank", c.q_lora },
        { "attention.output_group_count", c.o_groups }, { "attention.output_lora_rank", c.o_rank },
        { "attention.sliding_window", c.window }, { "expert_count", c.n_expert },
        { "expert_used_count", c.top_k }, { "expert_feed_forward_length", c.n_ff },
        { "hyper_connection.count", c.hc }, { "hyper_connection.sinkhorn_iterations", c.hc_iters },
        { "attention.indexer.head_count", c.ih }, { "attention.indexer.key_length", c.idim },
        { "attention.indexer.top_k", c.itopk } };
    for (const auto& n : need)
        if (n.v <= 0) {
            fprintf(stderr, "radiance: deepseek41: the checkpoint has no deepseek41.%s\n", n.what);
            return RAD_E_UNSUPPORTED;
        }
    /* The kv and index sources may be absent where no layer compresses (a model cut to its first
     * layers); a compressing layer without one is refused below. */
    if ((int64_t)c.ratio.size() < c.n_layer) {
        fprintf(stderr, "radiance: deepseek41: compress_ratios has %zu values for %lld layers\n",
                c.ratio.size(), (long long)c.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    if (c.n_shared != 1 || c.head_dim % 2 || c.rope > c.head_dim || c.rope > c.idim ||
        (c.n_head * c.head_dim) % c.o_groups || c.n_expert % 2 || c.hc > 8) {
        fprintf(stderr, "radiance: deepseek41: unsupported geometry (shared experts %lld, head_dim "
                        "%lld, rope %lld, groups %lld, experts %lld, hc %lld)\n",
                (long long)c.n_shared, (long long)c.head_dim, (long long)c.rope,
                (long long)c.o_groups, (long long)c.n_expert, (long long)c.hc);
        return RAD_E_UNSUPPORTED;
    }
    /* Each compressing layer reads the latest kv source and index source at or below it; their
     * ratios must agree. */
    c.src_of.assign((size_t)c.n_layer, -1);
    c.idx_of.assign((size_t)c.n_layer, -1);
    int s = -1, x = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        if (c.is_src(l)) s = l;
        if (c.is_idx(l)) x = l;
        const int64_t r = c.ratio[(size_t)l];
        if (r <= 0) continue;
        if (s < 0 || x < 0 || c.ratio[(size_t)s] != r || c.ratio[(size_t)x] != r ||
            c.head_dim % (2 * r) || c.idim % (2 * r)) {
            fprintf(stderr, "radiance: deepseek41: layer %d (ratio %lld) has no kv or index source "
                            "of its ratio\n", l, (long long)r);
            return RAD_E_UNSUPPORTED;
        }
        c.src_of[(size_t)l] = s;
        c.idx_of[(size_t)l] = x;
    }
    if (!c.ple.empty() && (c.ngram < 2 || c.ple_heads <= 0 || c.ple_dim <= 0 || c.pad < 0 ||
                           c.ple_rows <= 0)) {
        fprintf(stderr, "radiance: deepseek41: the Engram layout is incomplete\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the model */
struct Layer {
    int l = 0;
    int64_t r = 0;            /* compress ratio, 0 for window only */
    bool src = false, idx = false;
    /* weights */
    rad_weight hc_a_fn = 0, hc_a_scale = 0, hc_a_base = 0, hc_f_fn = 0, hc_f_scale = 0,
               hc_f_base = 0;
    rad_weight attn_norm = 0, wq_a = 0, q_norm = 0, wq_b = 0, wkv = 0, kv_norm = 0, sinks = 0,
               wo_b = 0;
    std::vector<rad_weight> wo_a;
    rad_weight comp_w = 0, comp_norm = 0, ik_w = 0, ik_norm = 0, iq_w = 0, iproj = 0;
    rad_weight e_mult = 0, e_vsz = 0, e_off = 0, e_kv = 0, e_gate = 0;
    /* ops, in issue order */
    rad_op e_ids = 0, e_emb = 0, e_copy = 0, e_gemm = 0, e_gate_op = 0, e_back = 0;
    /* This Engram layer's row ids [T, heads]: one buffer a layer, since step() writes every
     * layer's ids before the first block runs (see there). */
    rad_buf ids = 0;
    /* Lever E1 (RADIANCE_DS41_ENGDD declared): the lookup over the compact ids `edd_ids` ([T, heads] i32, host), the kv
     * GEMM into Model::ekv_u, the gather of every pooled row's kv row from it through `edd_map` ([T] i32, host). Both
     * written by the plugin at issue and persistent, one a layer: a layer's gather (and a lookup the card ran) may still
     * be queued when the next layer's are written. 0 where not declared. */
    rad_op e_emb_dd = 0, e_gemm_u = 0, e_scat = 0;
    rad_buf edd_ids = 0, edd_map = 0;
    rad_op mix_a = 0, col_a = 0, norm_a = 0;
    rad_op q_a = 0, q_an = 0, q_b = 0, q_rope = 0;
    rad_op kv_g = 0, kv_n = 0, kv_rope = 0, kv_q = 0, kv_st = 0;
    rad_op c_gemm = 0, c_work = 0, c_pool = 0, c_norm = 0, c_tab = 0;
    rad_op k_gemm = 0, k_norm = 0, k_rope = 0, k_q = 0, k_st = 0;
    rad_op l_rope = 0, l_q = 0, l_st = 0, c_tail = 0;
    rad_op i_q = 0, i_rope = 0, i_qq = 0, i_w = 0, i_score = 0, i_cand = 0, i_topk = 0;
    /* index_score over run p of a step's rows (row_part p); i_runs[0] is i_score. i_keep: these
     * runs take the candidate layer's keep (Model::ikeep). */
    std::vector<rad_op> i_runs;
    bool i_keep = false;
    rad_op attn = 0, o_rope = 0, o_q = 0, o_b = 0, post_a = 0, attn_fill = 0, ar_attn = 0;
    std::vector<rad_op> o_a;
    /* Lever K1's folds (k1_read_env), 0 where not declared: kv_fs = kv_f + kv_st
     * (rmsnorm_rope_roundtrip_kv_store), attn_rt = attn + o_f, or attn + o_rope + o_q
     * (attn_sparse_sink_rope_roundtrip). Issued for those when their switch is 1. */
    rad_op kv_fs = 0, attn_rt = 0;
    /* Lever K1 fold a (RADIANCE_DS41_QNFOLD): q_an + q_b as rmsnorm_gemm_nt_ggml (the norm in q_b's staging; it also
     * writes qrn, which the indexer's i_q reads), declared beside the pair where q_b is a GGUF weight. */
    rad_op q_bf = 0;
    /* The fused chains (fused()), each 0 where its chain above is declared instead: kv_n + kv_rope
     * + kv_q, k_norm + k_rope + k_q, o_rope + o_q, l_rope + l_q, i_rope + i_qq, col_a + norm_a,
     * col_f + norm_f. */
    rad_op kv_f = 0, k_f = 0, o_f = 0, l_f = 0, i_f = 0, cn_a = 0, cn_f = 0;
    /* mhc_post into the next sublayer's collapse + norm (post_collapse_norm()): pcn_f = post_a +
     * col_f + norm_f, issued for post_a; pcn_a = the previous layer's post_f + this layer's col_a +
     * norm_a, issued for that post_f (so before this layer's Engram-free tap and mix_a). */
    rad_op pcn_f = 0, pcn_a = 0;
    /* ar_attn + pcn_f as one four-rank collective (ar_mhc_post_collapse_rmsnorm, RADIANCE_DS41_AR_PCN=1), issued
     * instead of the pair for a step ar_wide_write_exact takes; 0 = the pair. */
    rad_op ar_pcn_f = 0;
    /* The previous layer's MoE all-reduce + pcn_a as one collective (RADIANCE_DS41_AR_PCN=2), issued in place of
     * that block's all-reduce and pcn_a for a step ar_wide_write_exact takes; the block leaves the all-reduce to it
     * (its ar_out, kArOutExact, set when this is declared). 0 = the pair. */
    rad_op ar_pcn_a = 0;
    rad_weight ffn_norm = 0;
    rad_op mix_f = 0, col_f = 0, norm_f = 0, post_f = 0;
    MoeFP8 moe{};
    /* A DSpark target layer: the hc mean of its input streams (after its Engram) into column slice
     * `tap_col` of the drafter's source rows. */
    rad_op tap = 0;
    int tap_col = -1;
    /* A DSpark stage's context pass: its window key from the projected trunk rows. */
    rad_op d_kv = 0, d_kvn = 0, d_kvrope = 0, d_kvq = 0, d_kvst = 0;
};

/* The DSpark block drafter (the checkpoint's mtp.0 .. mtp.{n-1}; official inference/model.py DSparkBlock,
 * forward_spec), on the engine's block-drafter protocol (docs/PLUGIN.md, RAD_DRAFT_BLOCK):
 *
 *   trunk step    each target layer's input streams (after its Engram) collapse to their hc mean
 *                 (mhc_contract) into one column slice of `src` [T, n_taps * D]: model.py's main_hiddens.
 *   context pass  one row per trunk row: main_x = main_norm(main_proj(src)); per stage the window key
 *                 kv_norm(wkv(main_x)), rotated at the row's position, fp8 round trip, into the stage's
 *                 layer of the drafter's window cache. The stages' blocks never run over the context
 *                 (DSparkBlock.forward at start_pos 0 only seeds the cache, and each decode step writes the
 *                 committed token's key from its main_x).
 *   query pass    `block` rows a sequence, [anchor, noise x (block - 1)] at the anchor's position on: the
 *                 trunk's embedding into hc copies, identity pre gates, then each stage a full V4 block
 *                 (hc mix, window-only latent attention, MoE) whose attention reads the last `window`
 *                 context keys before the anchor plus all `block` keys of its own rows, non-causal
 *                 (get_dspark_topk_idxs): attn_sparse_sink over the drafter's cache with window
 *                 `window + block` and every row placed at its block's last position (iota + add over
 *                 the positions), while rope uses each row's own. The head: hc collapse with the last
 *                 stage's ffn gates, the stage's norm, the trunk's lm_head; the Markov bias
 *                 W2 . W1[prev] walked greedily over the head's top `cand` (dflash_select, as
 *                 arch/common/rad_block_dspark.h) into the proposal buffer. Every row predicts the token
 *                 after it (forward_head: logits[i] + bias(output_ids[i]) -> output_ids[i + 1]), so the
 *                 block is the depth.
 *   confidence    (ds41-acc, lever C1) the release's confidence head (mtp.{n-1}.confidence_head.proj,
 *                 [1, D + rank], no bias), which no container carries: read at declare from the sidecar
 *                 RADIANCE_DS41_DSPARK_CONF names (D + rank f32, an optional bias after them). Rank 0 then
 *                 declares dspark_conf after the walk, its logit a row from the collapse's output before
 *                 the norm and the walk's predecessor's Markov row (forward_head's
 *                 confidence_head(x, markov_embed)), into the plane rad_drafter_confidence names; the
 *                 engine's dspark_conf switch issues it and cuts each verify to the longest confident
 *                 prefix (exllamav3's rule). It changes no proposal.
 *   branches      (ds41-acc, lever T1) RADIANCE_DS41_DSPARK_ALT=N (1..4): rank 0 declares dspark_alt, the
 *                 walk's N next-best first proposals each walked on (rad_drafter_alternatives), which the
 *                 engine's dspark_alt switch issues and compares with the target's tokens: a shadow of a
 *                 tree at position 1, measured, never verified.
 *   not applied   the bias from outside the head's top `cand` (the walk's only approximation, acceptance
 *                 only). */
struct Dsp {
    bool on = false;
    int base = 0, n_stage = 0;            /* stages at absolute layers base .. base + n_stage - 1 */
    int64_t block = 0, noise = -1, rank = 0, n_expert = 0, top_k = 0, cand = 0, Q = 0;
    int64_t ne_rank = 0, ne_base = 0;     /* this rank's routed experts [ne_base, ne_base + ne_rank) */
    std::vector<int64_t> taps;            /* target layers, ascending (main_hiddens' order) */
    rad_kvgroup kv = 0;
    int64_t kv_bs = 0;
    /* the trunk's taps and the context pass, max_tok rows */
    rad_buf src = 0, c = 0, cn = 0, ckv = 0, ckvn = 0;
    /* the query pass, Q rows */
    rad_buf x = 0, xn = 0, h0 = 0, h1 = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv_ = 0, kvn = 0, o = 0, oa = 0, attn_out = 0, poff = 0, apos = 0;
    rad_buf fin = 0, rows = 0, logits = 0, cand_ = 0, unary = 0, pairs = 0, gath = 0, tokens = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    rad_weight fc = 0, main_norm = 0, norm = 0, mk1 = 0, mk2 = 0;
    rad_op c_tab = 0, c_fc = 0, c_norm = 0, sizing = 0;
    rad_op embed = 0, embed_ar = 0, expand = 0, pre0 = 0, pre1 = 0, q_tab = 0, poff_op = 0, apos_op = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head = 0, topk = 0, pgather = 0, merge = 0, select = 0;
    /* lever W1, the full-vocabulary walk (RADIANCE_DS41_DSPARK_WALK; walk_read_env): world > 1, a dspark_walk_step a
     * step + the last merge, the keys a step [block * wseqs, 4] (a u64 in a 16-byte slot, ds41_walk_geom.h) zeroed by
     * wfill and gathered by wag into wgath [block * wseqs, world * 4]; world 1, dflash_markov_greedy_f32 (wmarkov)
     * with its tile scratch. */
    int64_t wseqs = 0;
    rad_buf wkeys = 0, wgath = 0, wts = 0, wtt = 0;
    rad_op wfill = 0, wag = 0, wmarkov = 0;
    std::vector<rad_op> wstep;
    /* lever W1b: the scoring steps again at form 1 (succ staged through LDS as whole lines; the same bits), issued
     * while dspark_walk is 2 (the last merge is wstep's) */
    std::vector<rad_op> wstep2;
    /* lever SD, sampled drafting (--draft-coupling on; fn4 draft_sample, the engine's): the walk of W1 sampled, a
     * dspark_walk_sample_step a step + the last merge (libr4d r4d_ds41_wsamp.hip). The coupling planes the engine finds
     * by name -- draws [max_seqs, 8] u32 it stages, q_temp [max_seqs, 1] f32 it stages (the request's temperature x the
     * switch's scale), state [block, max_seqs, 38] u32 it reads back -- and the walk's own: eight u64 keys a sequence a
     * step [block * sseqs, 16] gathered into [block * sseqs, 16 * world], each block's eight [sseqs * snblk, 16], and an
     * arrival counter a step [block, 4] zeroed by sfill every pass. */
    int64_t sseqs = 0, snblk = 0;
    rad_buf sdraw = 0, sqt = 0, sstate = 0, skeys = 0, sgath = 0, sscr = 0, scnt = 0;
    rad_op sfill = 0, sag = 0;
    std::vector<rad_op> sstep;
    std::vector<Layer> st;
    /* The confidence head (lever C1): the sidecar's D + rank weights and bias, the device copy
     * (`cwb`, uploaded at this rank's first query pass), the plane; the branches (lever T1). Rank 0. */
    std::vector<float> cw;
    double cbias = 0.0;
    std::string conf_path;
    int64_t n_alt = 0;
    rad_buf cwb = 0, conf = 0, alt = 0;
    rad_op conf_op = 0, alt_op = 0;
    bool cw_up = false;
};

/* A dense linear's int8 activation pair, keyed by the GEMM op a linear() declared: the quantiser
 * writes the pair and the GEMM reads it. One pair for the model's widest activation keeps a
 * quantised trunk's arena small; a GEMM's own columns are sliced at issue. */
struct IqrQOp {
    rad_op  op = 0;
    int64_t K  = 0;
};

/* Lever BI's tracer (RADIANCE_DS41_BITRACE=<dir>, ds41-bi 11.11.2026): a 64-bit fingerprint of every traced row of
 * every traced buffer after the op that wrote it (bi_rowhash, libr4d), copied to the host at the step's end and written,
 * for the rows at positions below RADIANCE_DS41_BITRACE_PMAX (default 640), to <dir>/bt-r<rank>-<pid>.bin (names in
 * .names). A run alone and one under load, matched by a request's (position, token) rows, name the first op whose
 * output differs (tools/bitrace.py). Unset: nothing declared, nothing issued -- today's graph and passes. Set: the
 * passes are not recorded (ctx.cpp) and every step synchronises once at its end. */
struct BiTr {
    bool on = false;
    std::string dir;
    int64_t pmax = 640;
    int64_t pmin = 0;               /* RADIANCE_DS41_BITRACE_PMIN: rows at positions [pmin, pmax) */
    int lmax = 1 << 30;             /* RADIANCE_DS41_BITRACE_LMAX: the layers' points below this layer only (head points kept) */
    rad_buf plane = 0;              /* [cap * max_tok, 2] i32 */
    int64_t cap = 0;                /* points a step */
    std::map<rad_buf, rad_op> op;   /* bi_rowhash over each traced buffer */
    int n = 0;                      /* points issued this step */
    std::vector<std::string> names; /* this step's points */
    std::vector<uint8_t> kind;      /* 0: the step's token rows; 1: its output rows (out_ids) */
    uint64_t names_key = 0;
    FILE* f = nullptr;
    FILE* fn = nullptr;
    std::vector<uint32_t> host;
};

struct Model {
    Names nm{};
    Geom g{};
    Cfg c{};
    std::vector<Layer> layers;
    int64_t work = 0, cmax = 0;
    /* --calib-layers, hi exclusive. Both 0: the tap is off. */
    int calib_lo = 0, calib_hi = 0;
    /* The indexer scores a step's rows in runs of at most `irows` rows into one [irows, cmax] score
     * plane (index_runs()), at most `iparts` runs a step; `kwords` is index_candidates' keep, a bit
     * a block of the candidate layer's width, i32 words a row (0: the model has no candidates). */
    int64_t irows = 0, iparts = 0, kwords = 0;
    /* Lever VA "arena" (RADIANCE_DS41_ARENA_DECODE_ROWS, ds41_arena_decode_rows()): the rows the
     * score plane and the selections are declared at in a sizing declare below irows (a level for
     * steps of at most that many rows, which are one run, never split). 0: irows, as the real
     * declare (and every declare with the switch unset). */
    int64_t score_rows = 0;
    /* The indexer's two exact restrictions (index_mode(), RADIANCE_DS41_INDEX at declare):
     * `ikeep` scores a layer past the candidate layer at the candidate layer's kept blocks only
     * (index_score's keep: the same selection, docs/OPS.md); `isplit` deals a step's runs over the
     * attention ranks (ds41_index_split.h), each rank scoring and selecting its own runs into
     * `sel_part` (`qmax` runs of `irows` rows at most), and `sel_gather` joins the parts into `sel`. */
    bool ikeep = false, isplit = false;
    int64_t qmax = 0;
    /* The thresholds of the two (index_mode()): a layer past the candidate layer restricts its
     * index_score to the kept blocks only from `keep_min_live` positions on (below, the kept share
     * is more than half the row and the restricted WMMA form costs about what the unrestricted one
     * does: 6.11. card, 92 % kept -> 105 %); the split deals runs from `split_min_live` on. Both
     * exact either way: index_topk reads the kept blocks only, and the split's gather lays every
     * run where the unsplit step does. */
    int64_t keep_min_live = 0, split_min_live = kDs41SplitMinLive;
    /* RADIANCE_DS41_IDXKEY=mxfp4: the index caches hold each key packed (index_slots(): 80 bytes for 128
     * values instead of 256; entry_store format / index_score key_format mxfp4). Lossless: the keys are
     * quant_roundtrip(fp4_e2m1, ue8m0, 32) values, and every score is the bf16 cache's bit for bit. */
    bool ikmx = false;
    /* RADIANCE_DS41_CTAIL=full: the compressor tail's chain over the whole work plane every step (before
     * 6.11.); unset: over the step's bound. */
    bool ctail_full = false;
    /* RADIANCE_DS41_LATENT=nvfp4: the compressed latent caches hold each 512-value entry packed (288
     * bytes instead of 1024: e2m1 codes, an e4m3 scale a block of 16; entry_store format /
     * attn_sparse_sink latent_format nvfp4). Lossless: the latent is quant_roundtrip(fp4_e2m1, e4m3, 16)
     * output, which the packing reproduces bit for bit (docs/OPS.md "Packed compressed latent"). */
    bool lnv = false;
    /* RADIANCE_DS41_KVWIN_FP8=1 (lever KVWIN; "trunk": the trunk's window only): the window caches -- the trunk's
     * kv_window (w8) and the DSpark drafter's kv_dspark (w8d) -- hold each 512-value key packed (528 bytes instead
     * of 1024: e4m3 codes, a scale byte a block of 32; window_store / rmsnorm_rope_roundtrip_kv_store kv_dtype /
     * attn_sparse_sink window_format fp8_e4m3). Lossless: the window key is quant_roundtrip(fp8_e4m3, ue8m0, 32)
     * output over the whole row, which the packing reproduces bit for bit (docs/OPS.md window_store). */
    bool w8 = false, w8d = false;
    rad_buf sel_part = 0;
    rad_op sel_gather = 0;
    /* The attention world (geometry()): attention ranks [0, aw), this rank's index `ar` in it and
     * its `gr` output groups; `zero` for an attention-zero rank. */
    int aw = 1, ar = 0;
    int64_t gr = 0;
    bool zero = false;
    /* This rank's columns [sh_lo, sh_hi) of the shared expert when its width does not split
     * evenly into whole blocks (shared_slice); 0, 0 is the even split. */
    int64_t sh_lo = 0, sh_hi = 0;
    /* buffers */
    rad_buf h0 = 0, h1 = 0, xcol = 0, xn = 0, pre_a = 0, pre_f = 0, post = 0, comb = 0;
    rad_buf qr = 0, qrn = 0, q = 0, kv = 0, kvn = 0, o = 0, oa = 0, attn_out = 0;
    rad_buf kvs = 0, stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, lat = 0, latn = 0,
            ik = 0, ikn = 0;
    rad_buf iq = 0, iw = 0, score = 0, keep = 0, sel = 0, cs_a = 0, cs_b = 0;
    rad_buf ctok = 0, emb = 0, embd = 0, ekv = 0;
    /* Lever E1: the unique rows' kv [T, (hc + 1) D]; `edd_ev` recorded behind the step's ngram_ids (the host reads the
     * ids once it completed), `edd_rec` / `edd_synced` this step's record and wait, `edd_counted` the step counted. */
    rad_buf ekv_u = 0;
    RadEvent edd_ev = nullptr;
    bool edd_rec = false, edd_synced = false, edd_counted = false;
    ds41_engdd::Plan edd_plan;
    std::vector<ds41_engdd::Launch> edd_ls;
    rad_buf final_hidden = 0, logit_input = 0, logits = 0;
    ActFP8 act{};
    MoeFP8::Wire mw{};
    /* KV groups */
    rad_kvgroup win = 0, tok = 0;
    std::map<int64_t, rad_kvgroup> comp, index, tail;
    /* The GGUF type of each linear weight the model ships in llama.cpp's blocks (lw()). */
    std::map<rad_weight, const char*> wty;
    /* IQR dense weights: lw()'s code view is `w`, its scale view `wsc`; `wdt` is the GEMM's dtype
     * (`i8a8h`, or an asymmetric rotated form) and `wrb` a trellis's rows per int8 scale. */
    std::map<rad_weight, rad_weight>   wsc;
    std::map<rad_weight, const char*>  wdt;
    std::map<rad_weight, uint32_t>     wcod;   /* codes' dtype: RAD_I8/RAD_U5 for IQR, F8E4M3 for fp8 */
    std::map<rad_weight, int64_t>      wrb;
    /* The quantiser each dense GEMM waits for (linear()): the op and the K its pair was declared at. */
    std::map<rad_op, IqrQOp>           qops;
    rad_buf iqr_q8 = 0, iqr_s8 = 0;   /* [max_tok, widest K] i8; [max_tok, widest K / 128] f32 */
    int64_t win_bs = 0;
    /* model-wide weights and ops */
    rad_weight token = 0, head = 0, final_gain = 0, tmap = 0, ple_table = 0;
    /* The Engram table stored as affine planes (a GGUF's IQ4_NL): views gathered in place by
     * embed_lookup_planes, `ple_group` values a scale; 0 for a plain bf16 table. */
    rad_weight ple_codes = 0, ple_scale = 0, ple_tab = 0;
    int64_t ple_group = 0;
    rad_op embed = 0, expand = 0, pre0 = 0, pre1 = 0, rope_a = 0, rope_b_step = 0, tok_map = 0;
    rad_op contract = 0, fnorm = 0, gather = 0, head_op = 0, embed_ar = 0;
    /* Lever K1 fold f (RADIANCE_DS41_ONEFILL): the step's plane of ones for every MoE layer's ungated shared
     * gate (mw.sgate), filled once at the step's start while the ds41_onefill switch is 1; 0 = not declared. */
    rad_op sone = 0;
    rad_op cn_fin = 0;   /* contract + fnorm fused (fused()), or 0 */
    rad_op pcn_fin = 0;  /* the last post_f + contract + fnorm fused, or 0 */
    /* The last layer's MoE all-reduce + pcn_fin as one collective (RADIANCE_DS41_AR_PCN=2), or 0. */
    rad_op ar_pcn_fin = 0;
    BiTr bt{};
    /* Lever BI (step_bi): each prefill segment's cu view {0, rows} (sub_i32 of cu at its sequence), a row a segment;
     * the row pitch of every buffer step_bi slices. */
    rad_buf bi_cu = 0;
    rad_op bi_sub = 0;
    std::map<rad_buf, int64_t> bp;
    /* Lever PSK: the first layer a prefill segment is cut at (the last kv source + 1); 0 where PSK is not declared. */
    int psk_first = 0;
    /* Lever PSK-L20 (ds41-pskl20): the last kv source (20 of the 1017's 40), split under ds41_psk_l20 1 -- its all-row
     * part over every row, its cone part over C_l rows as the layers past it; -1 where the split is not available. */
    int psk_l20 = -1;
    Dsp dsp{};
    /* The checkpoint is DeepSeek's release (is_release): the name map reads its tensors by
     * inference/model.py's names (ckpt_name), `n_stage` DSpark stages (mtp.*) after the trunk. */
    bool release = false;
    int n_stage = 0;
};
static std::unique_ptr<Model> models[MAX_RANKS];
/* The model whose name map is being declared (declare() on this thread). */
static thread_local Model* naming = nullptr;

/* The checkpoint tensor a declared weight is read from. The declared names are llama.cpp's GGUF
 * names (`blk.N.attn_q_a.weight`, the drafter's stages as blk.{n_layer + s}), which a GGUF ships as
 * they are. DeepSeek's release ships the same tensors under inference/model.py's names, and
 * llama.cpp's converter (conversion/deepseek41.py, DeepseekV4Model._map_dsv4_tensor_name) only
 * renames them: no permutation, split or transpose. So this is a table of renames; the joins and
 * products the converter makes (the experts' stacks, the Engram tables, the Engram gate) are maps
 * of their own at their call sites. A name the table does not know is returned unchanged. */
static const char* ckpt_name(const char* declared) {
    Model* m = naming;
    if (!m || !m->release) return declared;
    static const struct { const char* gguf; const char* rel; } layer_t[] = {
        { "attn_norm.weight", "attn_norm.weight" },
        { "ffn_norm.weight", "ffn_norm.weight" },
        { "hc_attn_fn.weight", "hc_attn_fn" }, { "hc_attn_base.weight", "hc_attn_base" },
        { "hc_attn_scale.weight", "hc_attn_scale" }, { "hc_ffn_fn.weight", "hc_ffn_fn" },
        { "hc_ffn_base.weight", "hc_ffn_base" }, { "hc_ffn_scale.weight", "hc_ffn_scale" },
        { "attn_sinks.weight", "attn.attn_sink" },
        { "attn_q_a.weight", "attn.wq_a.weight" },
        { "attn_q_a_norm.weight", "attn.q_norm.weight" },
        { "attn_q_b.weight", "attn.wq_b.weight" },
        { "attn_kv.weight", "attn.wkv.weight" },
        { "attn_kv_a_norm.weight", "attn.kv_norm.weight" },
        { "attn_output_a.weight", "attn.wo_a.weight" },
        { "attn_output_b.weight", "attn.wo_b.weight" },
        { "attn_compressor_kv.weight", "attn.compressor.wkv.weight" },
        { "attn_compressor_gate.weight", "attn.compressor.wgate.weight" },
        { "attn_compressor_norm.weight", "attn.compressor.norm.weight" },
        { "indexer.attn_k.weight", "attn.indexer.wk.weight" },
        { "indexer.k_norm.weight", "attn.indexer.k_norm.weight" },
        { "indexer.attn_q_b.weight", "attn.indexer.wq_b.weight" },
        { "indexer.proj.weight", "attn.indexer.weights_proj.weight" },
        { "ffn_gate_inp.weight", "ffn.gate.weight" },
        { "exp_probs_b.bias", "ffn.gate.bias" },
        { "ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight" },
        { "ffn_up_shexp.weight", "ffn.shared_experts.w3.weight" },
        { "ffn_down_shexp.weight", "ffn.shared_experts.w2.weight" },
        { "engram_kv.weight", "engram.wkv.weight" } };
    const int last = m->n_stage - 1;
    const std::string d = declared;
    if (d == "token_embd.weight") return "embed.weight";
    if (d == "output.weight") return "head.weight";
    if (d == "output_norm.weight") return "norm.weight";
    if (d == "deepseek41.engram.token_map") return "engram.token_map";
    if (d == "dspark.fc.weight") return "mtp.0.main_proj.weight";
    if (d == "dspark.enc.output_norm.weight") return "mtp.0.main_norm.weight";
    if (d == "dspark.output_norm.weight") return m->nm.ckpt("mtp.%d.norm.weight", last);
    if (d == "dspark.markov_w1.weight") return m->nm.ckpt("mtp.%d.markov_head.embed.weight", last);
    if (d == "dspark.markov_w2.weight") return m->nm.ckpt("mtp.%d.markov_head.head.weight", last);
    int l = -1, at = 0;
    if (std::sscanf(declared, "blk.%d.%n", &l, &at) != 1 || at <= 0) return declared;
    const char* suffix = declared + at;
    for (const auto& e : layer_t)
        if (std::strcmp(e.gguf, suffix) == 0)
            return l < m->c.n_layer ? m->nm.ckpt("layers.%d.%s", l, e.rel)
                                    : m->nm.ckpt("mtp.%d.%s", l - (int)m->c.n_layer, e.rel);
    return declared;
}

static rad_op op(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                 std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                 std::initializer_list<rad_buf> wr) {
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    if (!h) std::fprintf(stderr, "radiance: deepseek41: op %s refused\n", name);
    return rw(b, h, rd, wr);
}
static rad_weight wt(RadBuilder* b, const char* name, uint32_t dt,
                     std::initializer_list<int64_t> shape, RadWeightGroup grp,
                     int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
    return decl_w(b, name, dt, shape, access, RAD_SHARD_NONE, grp);
}
/* A weight split over the ranks: this rank's [lo, hi) of its split dimension (rows for
 * RAD_SHARD_ROW, columns for RAD_SHARD_COL), declared at `shape`, which must already be that
 * extent. An explicit span because the attention world can be smaller than the world. */
static rad_weight wts(RadBuilder* b, const Geom& g, const char* name, uint32_t dt,
                      std::initializer_list<int64_t> shape, RadWeightGroup grp, int shard, int64_t lo,
                      int64_t hi, int access = RAD_ACCESS_PER_TOKEN) {
    if (map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
    const rad_weight w = decl_w(b, name, dt, shape, access, g.world > 1 ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (g.world > 1 && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    return w;
}
/* A linear's weight, [n, k] at this rank's extents: llama.cpp's blocks when the model ships it
 * so (a GGUF weight, imported as its type's planes; rad_ggml.h), declared at the codes' dtype and
 * read by gemm_nt_ggml, else bf16. A block's values run along k, so k and a column span must cut
 * whole blocks. With `shard` and a world above one, this rank holds [lo, hi) of the split
 * dimension. */
static rad_weight lw(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                     RadWeightGroup grp, int shard = RAD_SHARD_NONE, int64_t lo = 0, int64_t hi = 0,
                     bool mapped = false, int access = RAD_ACCESS_PER_TOKEN) {
    if (!mapped && map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
    RadEncoding e{};
    int64_t blk = 0;
    const bool known = weight_enc(b, name, &e);
    const char* ty = known ? ggml_type_of(e, &blk) : nullptr;
    const bool split = m.g.world > 1 && shard != RAD_SHARD_NONE;
    /* A container written in lane order: its bytes are already the GEMM's stored form. */
    if (known && rad_enc_is(&e, "stored")) {
        std::string base(name);
        const bool suffix = base.size() > 6 && base.compare(base.size() - 6, 6, ".weight") == 0;
        if (suffix) base.resize(base.size() - 6);
        const char* sn = m.nm.f("%s.scale", base.c_str());
        const rad_weight w = decl_view(b, name, nullptr, "codes", RAD_U8, { n, k }, access,
                                       split ? shard : RAD_SHARD_NONE, grp);
        const rad_weight ws = decl_view(b, sn, name, "scale", RAD_U8, { n, k / 128 }, access,
                                        split ? shard : RAD_SHARD_NONE, grp);
        if (!w || !ws) {
            fprintf(stderr, "radiance: deepseek41: stored IQR weight %s refused\n", name);
            return 0;
        }
        m.wsc[w] = ws;
        m.wcod[w] = RAD_U8;
        m.wdt[w] = "i8a8h";
        m.wrb[w] = 0;
        return w;
    }
    const W8hForm form = known ? linear_w8h_form(e) : W8hForm::None;
    const bool w8 = known && linear_is_w8(e);
    const bool w8h = known && linear_is_w8h(e);
    const bool wnh = known && linear_is_wnh(e);
    /* A code plane narrower than bf16 in an affine/trellis encoding is an IQR view even when
     * linear_w8h_form() does not name its layout (e.g. an unsigned table with a zero point).
     * A named GGML block (q6_K and the rest) is not that view: its scale is the block's own
     * planes, and gemm_nt_ggml / embed_lookup_ggml read them. */
    const bool narrow = !ty && known && (rad_enc_is(&e, "affine") || rad_enc_is(&e, "trellis")) &&
                        e.n_planes >= 2 && e.plane[0].dtype != RAD_BF16 && e.plane[0].dtype != RAD_F32;
    /* A rotated or int8 narrow weight: lw()'s codes are a view of the stored planes, its scale
     * plane a second view; the GEMM decodes them where it reads them (libr4d's IQR rows). */
    if (!ty && (form != W8hForm::None || w8 || w8h || wnh || narrow)) {
        const bool trellis = form == W8hForm::Trellis;
        const RadEncPlane* tpl = trellis ? rad_enc_plane(&e, "trellis") : nullptr;
        const int64_t wb = trellis && tpl ? tpl->block[1] : 0;
        if ((n % 16) || (k % (trellis ? 256 : 128))) {
            fprintf(stderr, "radiance: deepseek41: %s is an IQR weight; a dense IQR GEMM takes rows in "
                            "16s and K in whole 256s, this rank reads %lldx%lld\n",
                    name, (long long)n, (long long)k);
            return 0;
        }
        if (split && shard == RAD_SHARD_COL && (lo % 128 || hi % 128)) {
            fprintf(stderr, "radiance: deepseek41: %s is an IQR weight, and this rank's column slice "
                            "[%lld, %lld) cuts a 128-column scale group\n",
                    name, (long long)lo, (long long)hi);
            return 0;
        }
        const char* cv = form == W8hForm::Zero    ? "codes,zero"
                       : form == W8hForm::Table ? "codes,table"
                       : form == W8hForm::Grid  ? "codes,grid"
                       : form == W8hForm::Trellis
                             ? (linear_w8h_mad(e) ? "codes,trellis,trellis.mad" : "codes,trellis")
                       : wnh                    ? "codes,zero"
                                                : "codes";
        std::string base(name);
        const bool suffix = base.size() > 6 && base.compare(base.size() - 6, 6, ".weight") == 0;
        if (suffix) base.resize(base.size() - 6);
        const char* sn = m.nm.f("%s.scale", base.c_str());
        const bool rowf = form != W8hForm::None && linear_w8h_rowf(e);
        const bool two = form != W8hForm::None && linear_w8h_scale2(e);
        const char* sv = wnh ? "scale,zero" : rowf ? "scale,scale.col" : two ? "scale,scale.1" : "scale";
        const uint32_t sdt = wnh ? (uint32_t)RAD_F16 : rowf ? (uint32_t)RAD_F32
                           : two ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
        const bool logical = wnh || rowf || two;
        const rad_weight w = decl_view(b, name, nullptr, cv, e.plane[0].dtype, { n, k }, access,
                                       split ? shard : RAD_SHARD_NONE, grp);
        const rad_weight ws = decl_view(b, sn, name, sv, sdt,
                                        { n, logical ? k : k / RAD_FP8_BLOCK }, access,
                                        split ? shard : RAD_SHARD_NONE, grp);
        if (!w || !ws) {
            fprintf(stderr, "radiance: deepseek41: IQR weight %s refused\n", name);
            return 0;
        }
        if (split && (rad_weight_shard_span(b, w, lo, hi) != RAD_OK ||
                      rad_weight_shard_span(b, ws, logical || shard == RAD_SHARD_ROW ? lo : lo / RAD_FP8_BLOCK,
                                            logical || shard == RAD_SHARD_ROW ? hi : hi / RAD_FP8_BLOCK) != RAD_OK)) {
            fprintf(stderr, "radiance: deepseek41: %s's IQR span [%lld, %lld) refused\n",
                    name, (long long)lo, (long long)hi);
            return 0;
        }
        m.wsc[w] = ws;
        m.wcod[w] = (uint32_t)e.plane[0].dtype;
        m.wdt[w] = wnh ? (e.plane[0].dtype == RAD_U5 ? "w5a8i8" : "w4a8i8_asym")
                       : form != W8hForm::None ? "i8a8h" : "i8a8";
        m.wrb[w] = wb;
        return w;
    }
    const bool split_ggml = split;
    if (ty && (k % blk || (split_ggml && shard == RAD_SHARD_COL && (lo % blk || hi % blk)))) {
        fprintf(stderr, "radiance: deepseek41: %s is GGUF %s, whose blocks cover %lld of K; this rank "
                        "reads %lld columns%s\n", name, ty, (long long)blk, (long long)k,
                split_ggml && shard == RAD_SHARD_COL ? " of a slice" : "");
        return 0;
    }
    const rad_weight w = decl_w(b, name, ty ? ggml_codes_dtype(e) : (uint32_t)RAD_BF16, { n, k },
                                access, split_ggml ? shard : RAD_SHARD_NONE, grp);
    if (!w) {
        fprintf(stderr, "radiance: deepseek41: weight %s refused\n", name);
        return 0;
    }
    if (split_ggml && rad_weight_shard_span(b, w, lo, hi) != RAD_OK) {
        fprintf(stderr, "radiance: deepseek41: %s's span [%lld, %lld) refused\n", name, (long long)lo,
                (long long)hi);
        return 0;
    }
    if (ty) m.wty[w] = ty;
    return w;
}
/* y = x . w^T over lw()'s weight: for a GGUF block it is gemm_nt_ggml, for an IQR view gemm_nt_q
 * at the weight's code dtype over a rotated int8 activation pair, otherwise gemm_nt at bf16. */
static rad_op linear(RadBuilder* b, Model& m, int64_t rows, rad_weight w, int64_t n,
                     int64_t k, rad_buf x, rad_buf y) {
    const auto s = m.wsc.find(w);
    if (s != m.wsc.end()) {
        const char* qdt = m.wdt.at(w);
        const bool rot = std::strcmp(qdt, "i8a8") != 0;
        const rad_op qop = op(b, rot ? "had_quant_act_i8g" : "quant_act_i8g",
                              { RAD_RANGE("M", 1, rows), RAD_INT("n", k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("dtype", "bf16") },
                              {}, { x }, { m.iqr_q8, m.iqr_s8 });
        std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", qdt) };
        const int64_t wrb = m.wrb.at(w);
        if (wrb > 1) ps.push_back(RAD_INT("w_rb", wrb));
        std::vector<rad_weight> wts = { w, s->second };
        const rad_op h = op(b, "gemm_nt_q", ps, wts, { m.iqr_q8, m.iqr_s8 }, { y });
        if (qop && h) m.qops[h] = IqrQOp{ qop, k };
        return h;
    }
    const auto t = m.wty.find(w);
    if (t != m.wty.end())
        return op(b, "gemm_nt_ggml",
                  { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("type", t->second),
                    RAD_STR("dtype", m.g.dtype) },
                  { w }, { x }, { y });
    return op(b, "gemm_nt",
              { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("dtype", m.g.dtype) },
              { w }, { x }, { y });
}
/* Issue a dense linear: its quantiser first when lw() found an IQR weight, then its GEMM. The pair
 * is one buffer for the whole model, so the quantiser and GEMM read the same columns of it.
 * gemm_nt_q's schema is a, a_scale, b, b_scale, y, then optional a_sum and b_ref (7 slots, 5
 * required). Issuing the bf16 list (activation, weight, y) is a step-0 schema refusal. i8a8 and
 * i8a8h do not read the trailing pair; pass them as nulls so a later grid fails closed. */
static void lin_issue(RadCtx* x, Model& m, rad_op h, int64_t rows, RadOperand a, rad_weight w,
                      RadOperand y) {
    const auto q = m.qops.find(h);
    if (q != m.qops.end()) {
        const int64_t K = q->second.K;
        const RadOperand q8 = bcol(m.iqr_q8, 0, K, rows);
        const RadOperand s8 = bcol(m.iqr_s8, 0, K / RAD_FP8_BLOCK, rows);
        RAD_ISSUE_N(x, q->second.op, rows, a, q8, s8);
        const auto sc = m.wsc.find(w);
        RAD_ISSUE_N(x, h, rows, q8, s8, RAD_W(w), RAD_W(sc->second), y, RAD_NONE, RAD_NONE);
        return;
    }
    RAD_ISSUE_N(x, h, rows, a, RAD_W(w), y);
}

static rad_op rmsnorm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                      rad_buf x, rad_buf y) {
    return op(b, "rmsnorm",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                RAD_STR("dtype", "bf16") },
              { w }, { x }, { y });
}
static rad_op rope_rows(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                        int inverse, rad_buf x, rad_buf cs) {
    return op(b, "ds41_rope_rows",
              { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                RAD_INT("inverse", inverse), RAD_STR("dtype", "bf16") },
              {}, { x, cs }, { x });
}
static rad_op roundtrip(RadBuilder* b, int64_t rows, int64_t n, int64_t block, const char* fmt,
                        const char* scale, rad_buf x) {
    return op(b, "quant_roundtrip",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block", block),
                RAD_STR("format", fmt), RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
              {}, { x }, { x });
}
/* format: null (bf16 entries) or "mxfp4" (the packed index-key form, docs/OPS.md entry_store) */
static rad_op entry_store(RadBuilder* b, int64_t rows, int64_t n, int64_t bs, rad_buf x,
                          rad_buf page, const char* format = nullptr) {
    std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("block_size", bs),
                                 RAD_STR("dtype", "bf16") };
    if (format) ps.push_back(RAD_STR("format", format));
    return op(b, "entry_store", ps, {}, { x, page }, {});
}

/* ---- fused chains (docs/OPS.md rmsnorm_rope_roundtrip, rope_rows_roundtrip,
 * mhc_collapse_rmsnorm): one launch for a chain of two or three dependent ops on the same rows,
 * the same bytes as the chain (a kernel library's `replaces`). A decode step pays a fixed cost per
 * dependent launch, so this is most of what the small ops cost there. They are device-only (libr4d
 * declares them; libref has no row): each declares the fused op only where a loaded library
 * declares and serves it, and returns 0 otherwise, and the caller then declares the chain. */
static rad_op fused(RadBuilder* b, const char* name, std::vector<RadParam> ps,
                    std::vector<rad_weight> ws, std::initializer_list<rad_buf> rd,
                    std::initializer_list<rad_buf> wr) {
    if (!rad_op_declared(b, name)) return 0;
    const rad_op h = rad_decl_op(b, name, ps.data(), (int)ps.size(),
                                 ws.empty() ? nullptr : ws.data(), (int)ws.size());
    if (!h || !rad_op_resolved(b, h)) return 0;
    return rw(b, h, rd, wr);
}
/* rmsnorm (x, w -> y) then rope_rows and roundtrip over y, as one op: one head of n columns. */
static rad_op norm_rope_rt(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n,
                           int64_t block, const char* fmt, const char* scale, rad_buf x, rad_buf y,
                           rad_buf cs) {
    return fused(b, "rmsnorm_rope_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", m.c.eps),
                   RAD_INT("n_head", 1), RAD_INT("head_dim", n), RAD_INT("rot", m.c.rope),
                   RAD_INT("off", n - m.c.rope), RAD_STR("mode", "gptj"), RAD_INT("inverse", 0),
                   RAD_INT("block", block), RAD_STR("format", fmt), RAD_STR("scale", scale),
                   RAD_STR("dtype", "bf16") },
                 { w }, { x, cs }, { y });
}
/* rope_rows then roundtrip over all heads * hd columns of x, in place, as one op. */
static rad_op rope_rt(RadBuilder* b, const Model& m, int64_t rows, int64_t heads, int64_t hd,
                      int inverse, int64_t block, const char* fmt, const char* scale, rad_buf x,
                      rad_buf cs) {
    return fused(b, "rope_rows_roundtrip",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n_head", heads), RAD_INT("head_dim", hd),
                   RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope), RAD_STR("mode", "gptj"),
                   RAD_INT("inverse", inverse), RAD_INT("block", block), RAD_STR("format", fmt),
                   RAD_STR("scale", scale), RAD_STR("dtype", "bf16") },
                 {}, { x, cs }, { x });
}
/* mhc_collapse (h, pre -> x) then rmsnorm (x, w -> y), as one op. */
static rad_op collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, rad_buf h,
                            rad_buf pre, rad_buf x, rad_buf y) {
    return fused(b, "mhc_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { h, pre }, { x, y });
}
/* mhc_post (in, h, post, comb -> out), mhc_collapse (out, pre -> x), rmsnorm (x, w -> y), as one
 * op. */
static rad_op post_collapse_norm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w,
                                 rad_buf in, rad_buf h, rad_buf out, rad_buf pre, rad_buf x,
                                 rad_buf y) {
    return fused(b, "mhc_post_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rows), RAD_INT("n", m.c.n_embd), RAD_INT("hc", m.c.hc),
                   RAD_F64("eps", m.c.eps), RAD_STR("dtype", "bf16") },
                 { w }, { in, h, m.post, m.comb, pre }, { out, x, y });
}

/* The shared expert is split by width, and a rank's K slice of its down projection must fall on
 * the encoding's block boundary along K: 256 columns for the GGUF k- and i-quant superblocks, 128 for
 * block fp8, 32 for legacy GGUF and MXFP4, 1 for bf16. A slice on a 256-column boundary is on every
 * one's. The even split is kept wherever each rank's share is whole 256-column blocks (and wherever
 * the width is no whole number of them, where the encoding decides at declare, as before); otherwise
 * the width splits by whole blocks, the first nb % world ranks one block more. DeepSeek-V4.1-Flash's
 * 2304 columns are nine blocks: three ranks take three each, and four ranks 768/512/512/512 where
 * the even 576 would cut a superblock. Each rank computes its gate/up columns and K slice of down,
 * and the all-reduce sums the partials (MoeFP8::Config::shared_lo). */
static constexpr int64_t kSharedBlock = 256;

/* RADIANCE_DS41_ARGATHER: the MoE gather and its all-reduce as one kernel (moe_gather_all_reduce,
 * MoeFP8::Config::gather_ar_rows) for a step of at most this many tokens at four ranks; unset, empty,
 * 0 or off is the two kernels, on is 4 (a decode step of up to four sequences, where the unfused
 * all-reduce is the one-shot too). Byte-identical either way.
 *
 * RADIANCE_DS41_ARGATHER_MAX=<rows> (1..256, read at declare; lever D3): with ARGATHER on (or a
 * number) the fused op is declared up to this many rows instead, so a step of up to <rows> tokens
 * (c8, a mixed step that small, a drafting verify) takes it. Unset: today's bound, nothing else
 * declared. libr4d's moe_gather_all_reduce runs the one-shot below the all-reduce's ts_min (32768
 * elements, 7 rows of 5120) and from it the gather in front of the exact two-shot (the transport the
 * unfused all-reduce takes there). Every form folds a token's slots on its rank in ascending slot
 * order, rounds that partial (with the shared arm's) to bf16 and sums the four ranks' partials in
 * fp32 in ascending rank order, rounded once: the bytes of moe_gather + all_reduce at every row count.
 * The in-process switch ds41_argather (POST /admin/fn4, 0..the declared bound) lowers the bound the
 * issue takes; the ops above it stay declared. */
static int64_t gather_ar_max() {
    const char* e = std::getenv("RADIANCE_DS41_ARGATHER_MAX");
    if (!e || !*e) return 0;
    const long n = std::strtol(e, nullptr, 10);
    return n > 0 ? std::min<long>(n, 256) : 0;
}
static int64_t gather_ar_rows() {
    const char* e = std::getenv("RADIANCE_DS41_ARGATHER");
    if (!e || !*e) return 0;
    const std::string v(e);
    if (v == "off" || v == "0" || v == "none") return 0;
    if (gather_ar_max() > 0) return gather_ar_max();
    if (v == "on") return 4;
    const long n = std::strtol(e, nullptr, 10);
    return n > 0 ? std::min<long>(n, 64) : 0;
}

/* RADIANCE_DS41_SHEXP_LANE=1 (or on; lever F4): at a decode or verify step of one pass (at most
 * kMoeShexpLaneRows rows) every trunk MoE layer issues its shared expert (gate, gate_up, the clamped
 * activation, down) on the second lane beside the router, the sort and the routed EXL3 experts, and
 * lane 0 joins it back in front of the gather (MoeFP8::shexp_lane). Nothing is declared for it; the
 * gather combines the arms as before, so the bytes are the same. Unset or anything else: one lane,
 * as before. The in-process switch ds41_shexp_lane (0 / 1) changes it at a step release. Read at the
 * first declare. */
static int shexp_lane_env() {
    const char* e = std::getenv("RADIANCE_DS41_SHEXP_LANE");
    return e && (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1 : 0;
}

/* The in-process switches this plugin serves (core/runtime/fn4_switch.h; the core finds the hooks
 * below through the plugin's handle): "ds41_argather" -- the fused gather's live row bound, 0..the
 * declared bound (gather_ar_rows), the issue taking moe_gather_all_reduce at T <= it and moe_gather
 * + all_reduce above (MoeFP8::gar_live); "ds41_shexp_lane" -- 0 / 1, the shared expert on the second
 * lane (MoeFP8::shexp_lane), 1 only where a trunk layer's arm is lane-safe. Applied between steps
 * with every rank thread parked; the core's epoch then re-records every pass. Process-wide: every
 * rank's blocks read the same words. */
static std::atomic<int> g_sw_gar{-1};          /* -1: the declared bound */
static std::atomic<int> g_sw_gar_decl{0};      /* the largest bound a layer declared */
static std::atomic<int> g_sw_shexp{0};
static std::atomic<int> g_sw_shexp_ok{0};      /* a trunk layer's shared arm is lane-safe */
static std::atomic<int> g_sw_shexp_env{-1};    /* RADIANCE_DS41_SHEXP_LANE, read once */
/* Lever K1's exact launch folds (10.11.2026), each declared beside the ops it folds only when its environment
 * asks (RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL = 1 or on, read at the first declare; unset: nothing more is
 * declared, the issue is today's, and the switch takes only 0), and picked per step by its switch:
 *   ds41_kvfold   fold c: the window key's rmsnorm_rope_roundtrip + kv_store as rmsnorm_rope_roundtrip_kv_store
 *                 (Layer::kv_fs), one launch writing y and the cache slot with the same bytes;
 *   ds41_ofold    fold d: attn_sparse_sink + rope_rows_roundtrip (the output's inverse rope and fp8 round trip)
 *                 as attn_sparse_sink_rope_roundtrip (Layer::attn_rt), the decode band in one launch;
 *   ds41_onefill  fold f: the ungated shared expert's gate plane of ones filled once a step (Model::sone) instead
 *                 of by each MoE layer (MoeFP8 shared_one_external, live through sone_live).
 * `_ok`: a layer (or the model) declared the fold, so 1 is accepted. Process-wide, as the switches above. */
static std::atomic<int> g_sw_kvf{0}, g_sw_kvf_ok{0}, g_sw_kvf_env{-1};
static std::atomic<int> g_sw_of{0}, g_sw_of_ok{0}, g_sw_of_env{-1};
static std::atomic<int> g_sw_one{0}, g_sw_one_ok{0}, g_sw_one_env{-1};
/* Folds a and e (lever K1, 10.11.2026, the same rules), each one launch where lever D2's multi-row form is on
 * (gg_gemv 2) in the decode band, the two launches elsewhere (libr4d's rows):
 *   ds41_qnfold   fold a: the query's rmsnorm (q_a_norm) in q_b's GEMV staging, rmsnorm_gemm_nt_ggml (Layer::q_bf),
 *                 which also writes qrn. 1 = every layer that declared it: at a layer with the indexer it is issued at
 *                 q_an's point (i_q, which reads qrn, behind it), elsewhere at q_b's; 2 = only the layers without the
 *                 indexer (an indexer layer keeps q_an + q_b, i_q beside q_b as today). RADIANCE_DS41_QNFOLD=1 (or
 *                 on, or 2) declares it and starts at that value.
 *   ds41_swfold   fold e: the shared expert's swiglu_clamped in its down GEMV's staging, swiglu_clamped_gemm_nt_ggml
 *                 (MoeFP8::op_swfold, issued at the down's point; it also writes the clamped product's plane).
 *                 RADIANCE_DS41_SWFOLD=1 (or on). Trunk layers only, as fold f. */
static std::atomic<int> g_sw_qnf{0}, g_sw_qnf_ok{0}, g_sw_qnf_env{-1};
static std::atomic<int> g_sw_swf{0}, g_sw_swf_ok{0}, g_sw_swf_env{-1};
/* 1 for "1" or "on", values up to `max` as themselves, else 0. */
static int k1_env(const char* name, int max = 1) {
    const char* e = std::getenv(name);
    if (!e) return 0;
    if (!std::strcmp(e, "1") || !std::strcmp(e, "on")) return 1;
    for (int v = 2; v <= max; ++v)
        if (e[0] == (char)('0' + v) && !e[1]) return v;
    return 0;
}
struct K1Switch {
    const char* name;
    std::atomic<int>* val;
    std::atomic<int>* ok;
    int max;                      /* the largest value taken where declared */
};
static const K1Switch kK1Switches[] = { { "ds41_kvfold", &g_sw_kvf, &g_sw_kvf_ok, 1 },
                                        { "ds41_ofold", &g_sw_of, &g_sw_of_ok, 1 },
                                        { "ds41_onefill", &g_sw_one, &g_sw_one_ok, 1 },
                                        { "ds41_qnfold", &g_sw_qnf, &g_sw_qnf_ok, 2 },
                                        { "ds41_swfold", &g_sw_swf, &g_sw_swf_ok, 1 } };
static const K1Switch* k1_switch(const char* name) {
    for (const K1Switch& k : kK1Switches)
        if (!std::strcmp(name, k.name)) return &k;
    return nullptr;
}
/* Once, at the first declare: a later declare (a sizing probe, a re-declare) keeps what the switch set since. */
static void k1_read_env() {
    const std::pair<std::atomic<int>*, const char*> e[] = { { &g_sw_kvf_env, "RADIANCE_DS41_KVFOLD" },
                                                            { &g_sw_of_env, "RADIANCE_DS41_OFOLD" },
                                                            { &g_sw_one_env, "RADIANCE_DS41_ONEFILL" },
                                                            { &g_sw_qnf_env, "RADIANCE_DS41_QNFOLD" },
                                                            { &g_sw_swf_env, "RADIANCE_DS41_SWFOLD" } };
    std::atomic<int>* const v[] = { &g_sw_kvf, &g_sw_of, &g_sw_one, &g_sw_qnf, &g_sw_swf };
    for (int i = 0; i < 5; ++i) {
        int unread = -1;
        if (e[i].first->compare_exchange_strong(unread, k1_env(e[i].second, kK1Switches[i].max)))
            v[i]->store(e[i].first->load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
}
/* Lever W1 (10.11.2026): the DSpark Markov walk over the whole vocabulary (dspark_walk_step, libr4d r4d_ds41_walk.hip),
 * the reference's walk, declared beside today's (the head's top `cand` re-ranked by dflash_select) only when
 * RADIANCE_DS41_DSPARK_WALK is set at the first declare: 0 / off declares it and starts at 0, 1 / on declares it and
 * starts at 1; unset (or another value) declares nothing more, the issue is today's and the switch takes only 0.
 * Switch "dspark_walk" (fn4): 0 the top-K re-rank, 1 the full-vocabulary walk, 2 the same walk with libr4d's form 1
 * (lever W1b: each block's succ rows staged through LDS as whole lines; the same fma chain an id, so the same bits and
 * tokens as 1; RADIANCE_DS41_DSPARK_WALK=2 starts there); picked per query pass. The drafter
 * only proposes and the trunk verifies, so the greedy text does not depend on it. `_ok`: a drafter declared the
 * walk. Process-wide, as the switches above. */
static std::atomic<int> g_sw_walk{0}, g_sw_walk_ok{0}, g_sw_walk2_ok{0}, g_sw_walk_env{-2};   /* env: -2 unread, -1 unset */
static void walk_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_DSPARK_WALK");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1 : !std::strcmp(e, "2") ? 2
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_walk_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_walk.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_WALK=%s is not 0 / off, 1 / on or 2: the "
                            "full-vocabulary walk is not declared\n", e);
    }
}
/* Lever BI (batch invariance, ds41-bi 11.11.2026; USER RULING: an answer that changes under load vs alone is a BUG).
 * Switch "ds41_bi" (fn4): 1 = a step whose rows belong to more than one shape class is issued in segments by each row's
 * own sequence (step_bi): the decode rows as one segment, each prefill sequence as its own, so every row-banded op
 * (gemm_nt_ggml's decode band <= 48 rows vs WMMA, attn_sparse_sink's and index_score's decode band <= 48 vs WMMA, the
 * fused norms' r4d_act_threads widths 1024 <= 32 rows, 512, 256) sees the rows its sequence gives it alone; the MoE one
 * pass a segment. 0 = one launch over the step (today). RADIANCE_DS41_BI=0 / off declares it and starts at 0, 1 / on
 * at 1; unset declares nothing (the switch takes only 0, step() is today's). `_ok`: a declare had it on. */
static std::atomic<int> g_sw_bi{0}, g_sw_bi_ok{0}, g_sw_bi_env{-2};   /* env: -2 unread, -1 unset */
static void bi_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_BI");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1 : (e[0] >= '2' && e[0] <= '4' && !e[1]) ? e[0] - '0'
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_bi_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_bi.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_BI=%s is not 0 / off, 1 / on or 2: lever BI is not declared\n", e);
    }
}
static bool bi_declared() { return g_sw_bi_env.load(std::memory_order_relaxed) >= 0; }
/* ds41_bi 2 (arm BI-P): as 1, and the routed MoE takes the decode segment in passes of at most this many rows -- the EXL3
 * experts' decode band (radiance-exl3 exl3r4d: <= 16 tokens a pass the decode kernels, above the prefill form). A row's
 * decode-band bytes do not depend on the pass's other rows (fn4 177b ds25: M-invariant from 1 to 16, split-K 2 at <= 2 not
 * taken at DeepSeek's widths), and a sequence alone passes <= 1 + n_spec (6) rows, so every decode row gets its alone bytes
 * whatever the step's decode rows (a c4 / c8 NST 5 verify: 24-48). Costs a MoE pass (router, gather, all-reduce, the
 * shared arm) per extra piece. */
static constexpr int64_t kBiMoe = 16;
/* The values: 1 = the segments (B); 2 = B + the MoE pieces (P); 3 = B, 4 = P, each with the cheap issue (C): the A1 chains
 * op by op across the segments (each weight's two reads back to back, so the decode rows' GEMV finds it in the last-level
 * cache) and the per-row ops that never flip (rmsnorm q_an / kv_n, ds41_rope_rows, rope_rows_roundtrip, quant_roundtrip,
 * the window-key store) once over the step. Every value gives a row its alone bytes. */
static bool bi_pieces(int v) { return v == 2 || v == 4; }
static bool bi_cheap(int v) { return v >= 3; }
static constexpr int64_t kBiBand = 48;   /* gemm_nt_ggml / attn_sparse_sink / index_score: the decode band's top */
static constexpr int64_t kBiNorm = 32;   /* the fused norms: r4d_act_threads' 1024-thread width up to this many rows */

/* Lever PSK (prefill skip above the last cache source, ds41-psk 11.11.2026). A layer past the last kv source (the
 * largest of attention.kv_source_layers: 20 of the 1017's 40) writes nothing a later token reads but its own window keys
 * (kv_window, the attention's reach: `window` keys with the row's own) and, at the DSpark target layers (37-39), the
 * drafter's tapped rows: the compressed latent, the index keys and the compressor tails are written at the kv sources
 * only (layer(): L.src), the indexer's candidates at the candidate layer (20), the Engram's token history from the
 * tokens alone (step(): e_ids, before the blocks). So a prefill chunk needs every row through the last source, and a
 * layer l past it only the rows its chunk's END reads: the last C_l = kPskBase + (n_layer - 1 - l) * window rows of
 * each prefill sequence's chunk. A row's output at l is right where its window's keys at l are -- the outputs of l - 1
 * at the window rows before it -- so C_l - C_{l+1} = window covers each layer's reach; the last layer keeps kPskBase
 * rows: >= window keys for the next token, the logits row, and C_{n_layer-2} - (window - 1) = kPskBase + 1 tapped rows
 * for the drafter's window + 1 (129) context keys. Every chunk computes its own end cone, so the window state at every
 * chunk END is the full prefill's: the checkpoint snapshots (a model with linear state checkpoints at chunk ends only,
 * Scheduler::plan_chunk), the next chunk, the first decode and the drafter's context read nothing else. The rows outside
 * the cone keep a lower layer's stream; their window keys at l are never read (a computed row reads <= window - 1
 * positions back, all inside the cone of the layer below), nor their taps (the drafter reads its last window + 1 keys).
 * Each cut prefill segment is at least kPskBase (> 120) rows, above every row-count form switch of the trunk's kernels
 * (decode bands <= 48, fused-norm widths <= 120, EXL3 decode band <= 16, index_topk's split <= 16). The cut steps run
 * through step_bi (one segment a prefill sequence, its cu view; attn_sparse_sink clamps a sequence's last row to the
 * launch's, and the cut rows end at it). Switch ds41_psk (fn4): 1 = the cone, 0 = every row (today).
 * RADIANCE_DS41_PSK=0 / off declares it and starts at 0, 1 / on at 1; unset declares nothing (1 refused, today's issue).
 * Cut: a step with one prefill sequence and no decode rows (any ds41_bi), and the prefill segments of a step ds41_bi
 * segments. Not cut: a step asking more logits rows than its decode rows + one a prefill sequence (prompt logprobs),
 * score-only / generator / encoder / plugin-word passes, a calibration (--calib-layers) serve.
 * The indexer at a cut layer (ds41-psk-fix cf7bbc7, carried into ds41-psk2): the candidate layer's keep (index_keep) is
 * rank-local -- under the row split each rank's index_candidates writes only its own runs' rows, and keep is not gathered
 * -- so a cut layer's indexer keeps the step's deal (ds41_idx_runs over the uncut segment, as every uncut layer) and skips
 * the runs wholly before its cut rows; re-dealt over the cut rows, a rank would read keep rows another rank wrote (r00y's
 * trace: L28.sel). Design 2 the same: a segment cut to its target's cone keeps its uncut segment's deal; a segment with no
 * row at the layer runs no indexer run there. */
static std::atomic<int> g_sw_psk{0}, g_sw_psk_ok{0}, g_sw_psk_env{-2};   /* env: -2 unread, -1 unset */
static std::atomic<long long> g_psk_steps{0}, g_psk_rows{0};
static void psk_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_PSK");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0
                          : !std::strcmp(e, "2") ? 2 : !std::strcmp(e, "3") ? 3 : -1;
    if (g_sw_psk_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_psk.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PSK=%s is not 0 / off, 1 / on, 2 or 3: lever PSK is not "
                            "declared\n", e);
    }
}
static bool psk_declared() { return g_sw_psk_env.load(std::memory_order_relaxed) >= 0; }
static constexpr int64_t kPskBase = 160;

/* Lever PSK design 2 (anchors, ds41-psk2 11.11.2026). ds41_psk 2 (3: + an anchor at the first checkpoint position at
 * or past 32768) hands the scheduler the choice of where the window state of layers past the last kv source must be
 * whole: its TARGETS, per prefill sequence -- the prompt's end, its tail snapshot, every ds41_psk_k-th checkpoint
 * position of the chunk grid, a request's own checkpoint_at (Scheduler::psk_next_target); the scheduler reserves window
 * snapshots at targets only. Each step, before its release, the core hands this plugin (rad_psk_plan, below) for every
 * prefill sequence of the step `ahead` = T1 - e: T1 the first target past the chunk's first position, e the chunk's end
 * (kPskNoHint: a sequence the scheduler did not latch, which runs design 1's cone under ds41_psk >= 1). A cone reaches
 * C_l rows back from its target at layer l, so the chunk runs at layer l its last psk_rows(C_l, ahead, n) rows: none
 * when T1 - C_l >= e, else from max(T1 - C_l, chunk start) to the chunk's end, at least kPskBase (>= 121) rows. ahead 0
 * (every chunk end a target) is design 1 to the row. A target's cone crossing the chunk start was run by the chunks
 * before it (the targets are fixed per request), so its rows read whole windows. A layer no segment runs a row of is
 * skipped whole (no op, no collective; the stager is told the pass's last layer, Engine::psk_release). The K/V store
 * range per layer is psk_kv_cone, a parameter apart from the forward range psk_cone (today the same). */
static constexpr int32_t kPskNoHint = INT32_MIN;
static std::atomic<int> g_sw_psk2_ok{0}, g_psk_first{0}, g_psk_nlayer{0}, g_psk_window{0};
static std::atomic<long long> g_psk_hsteps{0}, g_psk_skips{0}, g_psk_hbad{0};
/* Live counters for /stats (rad_psk_stats, ds41-int7-pskglue; rank 0's): layer-rows not run at the first nine cut layers
 * (21..29 on the 1017) and at the rest (30..39), and passes with a cut (design 1 or 2). */
static std::atomic<long long> g_psk_rows_lo{0}, g_psk_rows_hi{0}, g_psk_cut_passes{0};
struct PskPlan {
    int step = -1;               /* the trunk pass the plan is for (RadBatch::step); -1 none */
    long long dec_tok = 0;
    std::vector<int32_t> ahead, rows;   /* per prefill sequence, in the step's order */
    bool l20 = false;            /* lever PSK-L20 (ds41_psk_l20) as the plan's layer end was found under it */
    /* lever PSK-CR (ds41-pskcr, rad_psk_plan2): per prefill sequence bit 0 = a cone-replay chunk (empty: none) */
    std::vector<int32_t> flags;
};
static PskPlan g_psk_plan;     /* written by rad_psk_plan with every rank thread parked, read by the ranks' issue */
static int64_t psk_cone(int64_t n_layer, int64_t window, int64_t l) { return kPskBase + (n_layer - 1 - l) * window; }
static int64_t psk_kv_cone(int64_t n_layer, int64_t window, int64_t l) { return psk_cone(n_layer, window, l); }
/* The rows a prefill segment of `n` rows runs at a layer of cone C, its next target `ahead` rows past its end. */
static int64_t psk_rows(int64_t C, int32_t ahead, int64_t n) {
    if (ahead == kPskNoHint) return n;
    if ((int64_t)ahead >= C) return 0;
    return std::min(n, std::max(C - (int64_t)ahead, kPskBase));
}
/* The plan for this trunk pass, or null. */
static const PskPlan* psk_hint(const RadBatch* b) {
    return b && b->draft_pass == 0 && g_psk_plan.step >= 0 && g_psk_plan.step == b->step ? &g_psk_plan : nullptr;
}
/* A prefill sequence's `ahead`: the plan's, else design 1's (0) under ds41_psk >= 1, else no cut. */
static int32_t psk_ahead(const PskPlan* p, size_t i) {
    if (p && i < p->ahead.size() && p->ahead[i] != kPskNoHint) return p->ahead[i];
    return g_sw_psk.load(std::memory_order_relaxed) >= 1 ? 0 : kPskNoHint;
}

/* Lever PSK-L20 (ds41-pskl20 12.11.2026; ARCH-AUDIT-20261112 lever 1). The official model is a causal encoder-decoder
 * (tech report 2.2): layers 0-19 the encoder, 20-39 the decoder, and the decoder's global KV -- the compressed latent and
 * the index keys -- is projected from the encoder's final hidden state, layer 20's attention input. So of layer 20 (the
 * last kv source, L.src, and the indexer's candidate layer) only that projection needs every row: the collapse + norm
 * into xn, the compressor chain over the step (c_gemm .. c_norm, the tails), the index-key chain and the latent /
 * index-key stores (l_st / k_st). Its query path, indexer scoring / candidates / selection, attention, o-projection,
 * MoE and window-key store run over the cone C_20 = kPskBase + (n_layer - 1 - 20) x window rows exactly as layers 21-39
 * do over theirs (design 1 / 2 alike: psk_rows(C_20, ahead, n)). Why exact: rows outside C_20 reach a later layer only
 * through the latent / index keys, which still come from every row's layer-20 input; a row's window key at 20 is a
 * function of its own xn, so every cone row's key is right, and the first window - 1 cone rows (whose earlier window keys
 * at 20 were not stored) are outside C_21 = C_20 - window; the candidate keep a reindex layer (24..36) reads is per row
 * and its rows lie in C_20 (the cf7bbc7 deal: the cut indexer keeps the uncut segment's runs). Switch ds41_psk_l20
 * (fn4): 1 = split layer 20 wherever lever PSK cuts (ds41_psk >= 1), 0 = every row as before (today);
 * RADIANCE_DS41_PSK_L20=0 / 1 sets the start value (unset: 0). `_ok`: a declare found layer last-source split-safe
 * (lever PSK declared, no Engram or DSpark tap at it, the candidate layer at or below it). */
static std::atomic<int> g_sw_psk_l20{0}, g_sw_psk_l20_ok{0}, g_sw_psk_l20_env{-2};   /* env: -2 unread, -1 unset */
static std::atomic<int> g_psk_l20_layer{-1};   /* the split layer (Model::psk_l20) for rad_psk_plan; -1 none */
/* Live counters (rank 0's; /stats "psk"): passes whose layer 20 was cut, its rows not run in the cone part, passes
 * whose layer 20 ran its all-row part only (no query / attention / MoE row), planned passes the stager was told end at
 * layer 20 (its routed experts not staged: rad_psk_plan). */
static std::atomic<long long> g_psk_l20_passes{0}, g_psk_l20_rows{0}, g_psk_l20_empty{0}, g_psk_l20_end{0};
static void psk_l20_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_PSK_L20");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_psk_l20_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_psk_l20.store(v, std::memory_order_relaxed);
        if (e && v < 0) fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PSK_L20=%s is not 0 / off or 1 / on: ignored\n", e);
    }
}
/* Lever PSK-CR (ds41-pskcr 12.11.2026; ARCH-AUDIT-20261112 lever 3): the exact form of the official bounded replay (tech
 * report 3.2.2). A prefix hit with no snapshot at the deepest grid point H under its match (the cold run's chunk end
 * there) resumes at H by a cone replay: the scheduler adopts the cached global KV (latent + index keys, every row's: the
 * only state the decoder keeps of a row a window cannot reach) up to H, starts the sequence at the grid point at or below
 * H - reach (reach = C_0 = kPskBase + (n_layer - 1) x window, 5152 on the 1017) with no state restored, and runs the cold
 * run's grid chunks up to H flagged replay (rad_psk_plan2): every segment flagged runs EVERY layer, encoder layers
 * 0..20 too, over its rows in H's cone only -- psk_rows(C_l, H - chunk end, n), at least kPskBase rows (as design 2 cuts
 * 21..39 toward a target) -- and stores no global KV: the batch maps a replay entry's full-group slots and linear-state
 * slots to -1 (core sched/batch.cpp), and a pass of replay chunks only skips the kv sources' compressor chain whole.
 * Why the window state at H is a cold prefill's: layer 0's input is every row's embedding; the window store at l writes
 * a row's key from its own layer input; a row's output at l reads the keys of its window - 1 predecessors at l, the
 * cached global KV at positions <= its own (attention: the compressed latent of src_of[l]; the indexer: src_of[l]'s
 * index keys -- the cache holds every row of both, so a skipped store leaves nothing a replay row reads unwritten),
 * and, at a reindex layer, the candidate keep layer 20 wrote for the same row in the same pass (the cf7bbc7 deal: a cut
 * layer keeps the step's uncut deal at every cut layer, 0..20 included, so the same rank reads it). So a layer's input
 * right on [H - C_l, H) gives its keys right there and its output right on [H - C_l + window - 1, H) superset of
 * [H - C_(l+1), H): by induction every layer's window keys of [H - window + 1, H) and the 37..39 inputs (the DSpark
 * taps) of the last kPskBase rows are right, whatever the windows held at the replay's start. The Engram block reads
 * token ids only (its rows' n-gram history lies inside the replay from layer 1's cone on; the conv state at H is the
 * last ngram - 1 tokens, written by the replay's last chunk). The compressor tail (ratio 2 at 2..19) is not rebuilt:
 * H is a multiple of every ratio, so the chunk from H reads none of it. Rows run in launches of >= kPskBase rows (the
 * form band of the uncut 4096-row launch: fused norms > 120, decode bands 48, EXL3 16), and on the cold run's own
 * chunks, so each step's deal is the cold step's. Switch ds41_psk_cr (fn4): 1 lets the scheduler replay, 0 = today;
 * RADIANCE_DS41_PSK_CR=0 / 1 sets the start value (unset: 0); 1 refused where design 2 is not available. */
static std::atomic<int> g_sw_psk_cr{0}, g_sw_psk_cr_env{-2};   /* env: -2 unread, -1 unset */
/* Live counters (rank 0's; /stats "psk" cr_*): replay passes cut, their layer-rows run, their layer-rows not run below
 * the first PSK layer (m.psk_first), kv-source compressor chains skipped, replay passes beside other rows (decode or
 * another prefill: the chain ran, its stores masked by the batch). */
static std::atomic<long long> g_cr_passes{0}, g_cr_rows_run{0}, g_cr_rows_skip{0}, g_cr_chains{0}, g_cr_mixed{0};
static void psk_cr_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_PSK_CR");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_psk_cr_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_psk_cr.store(v, std::memory_order_relaxed);
        if (e && v < 0) fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PSK_CR=%s is not 0 / off or 1 / on: ignored\n", e);
    }
}
static bool psk_cr_flag(const PskPlan* p, size_t i) {
    return p && i < p->flags.size() && (p->flags[i] & 1) != 0;
}

/* Lever PSK-DS (ds41-pskl20 12.11.2026; ARCH-AUDIT-20261112 lever 2). The DSpark context pass (dsp_context, draft_pass
 * -1, one row a trunk row) writes each row's context key into the drafter's window cache; a query pass reads the last
 * c.window keys before its anchor (drf.window = c.window + 1 with the anchor), the anchor at or past its sequence's
 * committed end; prefix hits leave the drafter's cache a hole and recompute its window (draft refill: a hit ends
 * window + 1 tokens before the prompt's end), and no checkpoint carries it on this model (the window snapshot rides
 * only without linear state). So of a context pass only each entry's last c.window + 1 rows can still be read -- across
 * chunks too: a chunk's rows before its own last window + 1 lie more than a window before the sequence's end -- and
 * dspark_pfctx 1 runs the pass over those rows (ds41cone::pfctx_runs: each run >= kPskBase rows, widened over skipped
 * rows where shorter, so every launch is in the form band today's one launch over T >= kPskBase rows had: gemm_nt_ggml's
 * WMMA (> 48), rmsnorm's fixed partition, per-row rope / round trip / window store). Target output never reads a draft,
 * so the target is exact by construction; the drafter's readable window is byte-for-byte today's. 0 = every row (today).
 * RADIANCE_DS41_DSPARK_PFCTX=0 / 1 sets the start value (unset: 0). `_ok`: a declare has the context pass's ops. */
static std::atomic<int> g_sw_pfctx{0}, g_sw_pfctx_ok{0}, g_sw_pfctx_env{-2};   /* env: -2 unread, -1 unset */
/* Live counters (rank 0's; /stats "dspark_pfctx"): context passes cut, their rows not run, the rows they ran. */
static std::atomic<long long> g_pfctx_passes{0}, g_pfctx_skipped{0}, g_pfctx_run{0};
static std::atomic<int> g_pfctx_window{0};   /* c.window at the declare that set `_ok` (rad_arch_pass_key's plan) */
static void pfctx_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_DSPARK_PFCTX");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_pfctx_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_pfctx.store(v, std::memory_order_relaxed);
        if (e && v < 0) fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_PFCTX=%s is not 0 / off or 1 / on: ignored\n", e);
    }
}
/* Lever PSK-DS: this context pass's runs (`runs` [r0, n), `*kept` their rows), or false: every row, the pass as today. */
static bool pfctx_plan(const RadBatch* b, int64_t window, std::vector<std::pair<int64_t, int64_t>>* runs, int64_t* kept) {
    if (!b || b->draft_pass >= 0 || g_sw_pfctx.load(std::memory_order_relaxed) <= 0 || window <= 0) return false;
    return ds41cone::pfctx_runs(b->positions_h, b->n_tok, b->n_seq, window + 1, kPskBase, runs, kept);
}

/* Lever E1 (ds41-e1, 12.11.2026; audit lever 5): an Engram layer's lookup, copy and kv GEMM over a step's UNIQUE n-gram
 * id rows, every row's kv row then gathered from its unique row (ds41_engram_dedup.h: why it is exact, the plan, the
 * 48-row guard). Switch "ds41_engdd" (fn4): 1 = deduplicate a step with a prefill sequence (a pass with no prefill
 * sequence, the drafter's passes and every launch of <= 48 rows are issued as today), 0 = today's issue.
 * RADIANCE_DS41_ENGDD=0 / off declares it and starts at 0, 1 / on at 1; unset declares nothing (1 refused, the program
 * and the issue are today's). Declared: per Engram layer a second lookup (the compact ids), a GEMM into a scratch plane
 * (engram_kv_u) and a gather_rows back into engram_kv; host buffers for the compact ids and each layer's map. A pass it
 * may act on is not recorded (rad_arch_pass_key): its row counts come from the step's ids. `_ok`: a declare had it. */
static std::atomic<int> g_sw_engdd{0}, g_sw_engdd_ok{0}, g_sw_engdd_env{-2};   /* env: -2 unread, -1 unset */
/* Live counters (/stats "engdd", rad_engdd_stats; rank 0's): Engram layer passes deduplicated, rows not looked up /
 * copied / multiplied, rows pooled in those passes, layer passes the 48-row guard kept whole, layer passes below the
 * repeat share, steps with a deduplicated layer, passes with no launch above the band. */
static std::atomic<long long> g_edd_passes{0}, g_edd_rows{0}, g_edd_pooled{0}, g_edd_guard{0}, g_edd_low{0}, g_edd_steps{0},
    g_edd_nopool{0};
static void engdd_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_ENGDD");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_engdd_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_engdd.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_ENGDD=%s is not 0 / off or 1 / on: lever E1 is not declared\n", e);
    }
}
static bool engdd_declared() { return g_sw_engdd_env.load(std::memory_order_relaxed) >= 0; }
/* A pass lever E1 may deduplicate: the switch on where declared, a trunk pass with a prefill sequence wider than the band
 * (decode / verify passes keep today's issue and their recordings). The same test keys the pass (rad_arch_pass_key). */
static bool engdd_may_act(const RadBatch* b) {
    return b && g_sw_engdd.load(std::memory_order_relaxed) >= 1 && g_sw_engdd_ok.load(std::memory_order_relaxed) &&
           b->draft_pass == 0 && b->n_tok > ds41_engdd::kFormRows && b->n_seq > b->n_seq_decode && !b->enc && !b->gen;
}

static void sw_note_decl(int64_t rows) {
    int cur = g_sw_gar_decl.load(std::memory_order_relaxed);
    while ((int64_t)cur < rows && !g_sw_gar_decl.compare_exchange_weak(cur, (int)rows)) {}
}

/* idx_split: the indexer's balanced deal (lever I1), below index_runs(). */
static int idx_fn4_accepts(const char* name, int value);
static int idx_fn4_switch(const char* name, int value);
static int idx_fn4_value(const char* name, int* applied);

extern "C" int rad_fn4_accepts(const char* name, int value) {
    if (!name) return 0;
    if (const int r = idx_fn4_accepts(name, value)) return r;
    if (!std::strcmp(name, "ds41_argather"))
        return value >= 0 && value <= g_sw_gar_decl.load(std::memory_order_relaxed) ? 1 : -1;
    if (!std::strcmp(name, "ds41_shexp_lane"))
        return value == 0 || (value == 1 && g_sw_shexp_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (const K1Switch* k = k1_switch(name))
        return value == 0 || (value >= 1 && value <= k->max && k->ok->load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "dspark_walk"))
        return value == 0 || (value == 1 && g_sw_walk_ok.load(std::memory_order_relaxed)) ||
               (value == 2 && g_sw_walk2_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_bi"))
        return value == 0 || (value >= 1 && value <= 4 && g_sw_bi_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_psk"))   /* 4 (lever PSK-CR): tail-only anchors, design 2 here */
        return value == 0 || (value == 1 && g_sw_psk_ok.load(std::memory_order_relaxed)) ||
               ((value == 2 || value == 3 || value == 4) && g_sw_psk2_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_psk_cr"))
        return value == 0 || (value == 1 && g_sw_psk2_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_psk_l20"))
        return value == 0 || (value == 1 && g_sw_psk_l20_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "dspark_pfctx"))
        return value == 0 || (value == 1 && g_sw_pfctx_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_engdd"))
        return value == 0 || (value == 1 && g_sw_engdd_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    return 0;
}
extern "C" int rad_fn4_switch(const char* name, int value) {
    if (name && !std::strcmp(name, "idx_split")) return idx_fn4_switch(name, value);
    const int ok = rad_fn4_accepts(name, value);
    if (ok <= 0) return ok;
    if (!std::strcmp(name, "ds41_argather")) g_sw_gar.store(value, std::memory_order_relaxed);
    else if (const K1Switch* k = k1_switch(name)) k->val->store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "dspark_walk")) g_sw_walk.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_bi")) g_sw_bi.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk")) g_sw_psk.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk_l20")) g_sw_psk_l20.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "dspark_pfctx")) g_sw_pfctx.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk_cr")) g_sw_psk_cr.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_engdd")) g_sw_engdd.store(value, std::memory_order_relaxed);
    else g_sw_shexp.store(value, std::memory_order_relaxed);
    return 1;
}
extern "C" int rad_fn4_value(const char* name, int* applied) {
    if (!name || !applied) return 0;
    if (idx_fn4_value(name, applied)) return 1;
    if (!std::strcmp(name, "ds41_argather")) {
        const int v = g_sw_gar.load(std::memory_order_relaxed);
        *applied = v >= 0 ? v : g_sw_gar_decl.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_shexp_lane")) {
        *applied = g_sw_shexp.load(std::memory_order_relaxed);
        return 1;
    }
    if (const K1Switch* k = k1_switch(name)) {
        *applied = k->val->load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_bi")) {
        *applied = g_sw_bi.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_psk")) {
        *applied = g_sw_psk.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_psk_l20")) {
        *applied = g_sw_psk_l20.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "dspark_pfctx")) {
        *applied = g_sw_pfctx.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_psk_cr")) {
        *applied = g_sw_psk_cr.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ds41_engdd")) {
        *applied = g_sw_engdd.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "dspark_walk")) {
        *applied = g_sw_walk.load(std::memory_order_relaxed);
        return 1;
    }
    return 0;
}

static void shared_slice(int64_t n_ff, int world, int rank, int64_t* lo, int64_t* hi) {
    *lo = *hi = 0;
    if (world <= 1 || n_ff % kSharedBlock) return;
    if (n_ff % world == 0 && (n_ff / world) % kSharedBlock == 0) return;
    const int64_t nb = n_ff / kSharedBlock, base = nb / world, extra = nb % world;
    const int64_t b0 = rank * base + std::min<int64_t>(rank, extra);
    *lo = b0 * kSharedBlock;
    *hi = (b0 + base + (rank < extra ? 1 : 0)) * kSharedBlock;
}

static int geometry(Model& m, const RadBuildCtx* ctx) {
    auto& g = m.g;
    const auto& c = m.c;
    g.n_layer = c.n_layer;
    g.n_embd = c.n_embd;
    g.head_dim = c.head_dim;
    const int world = ctx->world_size ? ctx->world_size : 1;
    g.world = world;
    g.rank = ctx->rank;
    /* The attention world: the largest d <= world that divides both the output groups and the
     * heads; the ranks past it are attention-zero (Geom::attn_world). There is one latent KV head,
     * so geom_from's KV-head rule does not apply: three ranks attend on two, four on four. */
    int aw = 1;
    for (int d = world; d >= 1; --d)
        if (c.o_groups % d == 0 && c.n_head % d == 0) {
            aw = d;
            break;
        }
    g.attn_world = aw;
    g.attn_zero = ctx->rank >= aw;
    g.attn_rank = g.attn_zero ? aw - 1 : ctx->rank;
    m.aw = aw;
    m.ar = g.attn_rank;
    m.zero = g.attn_zero;
    m.gr = c.o_groups / aw;
    g.n_head_all = c.n_head;
    g.n_head = c.n_head / aw;
    g.n_head_kv = g.n_head_kv_all = 1;
    /* Routed experts are expert-parallel: each rank holds n_expert / world whole experts, an even
     * count (the grouped GEMM takes two parity tables), and the router stays replicated. Whole
     * experts are what the bf16 and quantised expert paths both serve, at any world (an expert's
     * width split into whole quant blocks holds at three ranks but not at four). The shared expert
     * is split by width. */
    int64_t sh_lo = 0, sh_hi = 0;
    shared_slice(c.n_ff, world, ctx->rank, &sh_lo, &sh_hi);
    if (c.n_expert % world || (c.n_expert / world) % 2 || (sh_hi == 0 && c.n_ff % world)) {
        fprintf(stderr, "radiance: deepseek41: %lld experts of width %lld do not split over %d ranks "
                        "(an even count of whole experts each, and the shared expert's width)\n",
                (long long)c.n_expert, (long long)c.n_ff, world);
        return RAD_E_UNSUPPORTED;
    }
    g.n_ff_all = c.n_ff;
    /* This rank's width of the shared expert (its buffers and the MoE block's n_ff_shared). */
    g.n_ff = sh_hi ? sh_hi - sh_lo : c.n_ff / world;
    m.sh_lo = sh_lo;
    m.sh_hi = sh_hi;
    g.n_vocab_all = c.n_vocab;
    g.n_vocab = shard_extent(c.n_vocab, world, ctx->rank);
    g.vocab_off = shard_begin(c.n_vocab, world, ctx->rank);
    g.max_tok = ctx->max_tok;
    g.max_ctx = ctx->max_ctx;
    g.max_seqs = ctx->max_seqs;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = std::min<int64_t>(g.max_tok, g.max_seqs * (g.max_spec + 1));
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.eps = (float)c.eps;
    g.theta = (float)c.theta;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;
    g.w_bf16 = true;
    g.wire_exact = 1;
    return g.max_tok < 1 || g.max_ctx < 1 || g.max_seqs < 1 ? RAD_E_INVAL : RAD_OK;
}

/* index_candidates' keep words a row (docs/OPS.md): a bit a block of `block` positions. */
static int64_t keep_words(int64_t width, int64_t block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

/* The indexer's score plane is a row's score of every compressed position, [rows, cmax] f32 with
 * cmax = max_ctx / (the smallest ratio) + 1: at ratio 1, 8192-row chunks and 256k of context a
 * whole step's plane would be 8 GiB, and the candidates' mask (live from the candidate layer to the
 * last that reads it) as much again. So a step is scored in runs of rows into one plane of at most
 * kScorePlane bytes (but never fewer than kScoreRunMin rows), each run selected from before the next
 * overwrites it, and the mask is a bit a block (index_candidates).
 *
 * A step of T rows takes n = ceil(T / irows) runs of R = ceil(T / n) rows. Rows are independent,
 * so every run is the unsplit op's rows bit for bit, provided each run is scored by the kernel the
 * whole step would have taken: kernels pick their form by the pass height (libr4d's index_score
 * runs WMMA above 48 rows, a different dot order from the decode form's), and a run of a split
 * step is over irows / 2 >= 128 rows, as tall as any form's threshold. A step of at most irows
 * rows is one run, the op as before. */
static constexpr int64_t kScorePlane = 256ll << 20;
static constexpr int64_t kScoreRunMin = 256;
static void index_runs(Model& m) {
    const auto& g = m.g;
    const auto& c = m.c;
    m.irows = std::min(g.max_tok, std::max(kScoreRunMin, kScorePlane / (4 * m.cmax)));
    m.iparts = (g.max_tok + m.irows - 1) / m.irows;
    m.kwords = 0;
    if (c.cand >= 0 && c.cand < c.n_layer && c.cand_block > 0 && c.ratio[(size_t)c.cand] > 0)
        m.kwords = keep_words(g.max_ctx / c.ratio[(size_t)c.cand] + 1, c.cand_block);
}

/* The row split's balanced deal (ds41_index_split.h; lever I1) as an in-process switch, "idx_split": 0 = the plain
 * deal (n = ceil(T / irows) runs of ceil(T / n) rows, ceil(n / world) a rank: a mixed step of 2048 prefill rows and
 * a few decode rows is 9 runs dealt 3 / 3 / 3 / 0), 1 = the balanced deal where the world does not divide n (every
 * rank an equal block of ceil(T / world) rows in equal runs: 12 runs, 3 a rank). Exact either way: a row's score,
 * candidates and selection are its own whichever run holds it, and the gather lays every row where the plain deal
 * does (the header). RADIANCE_DS41_IDX_BALANCE=1 sets 1 at the first declare (unset or anything else: 0, the issue
 * as before, byte for byte); POST /admin/fn4 {"idx_split": v} changes it at a step release with every rank thread
 * parked (core/runtime/fn4_switch.h), so every rank deals each step under one value. Nothing is declared for it: a
 * balanced step fits the plain deal's run ops (row_part < iparts), part, sel and score plane, or it takes the plain
 * deal. 1 is refused while no real declare has the split on (one rank, RADIANCE_DS41_INDEX without split, ...).
 * Only a step of more than irows rows at split_min_live positions is split, so decode and verify steps never see it. */
static std::atomic<int> g_idx_bal{-1};        /* -1: RADIANCE_DS41_IDX_BALANCE not read yet */
static std::atomic<int> g_idx_bal_ok{0};      /* a real declare has the row split on */
static std::atomic<int> g_idx_bal_noted{0};   /* rank 0 said so at its first balanced step */
static void idx_balance_env() {
    int unread = -1;
    const char* e = std::getenv("RADIANCE_DS41_IDX_BALANCE");
    g_idx_bal.compare_exchange_strong(unread, e && !std::strcmp(e, "1") ? 1 : 0);
}
static bool idx_balance() { return g_idx_bal.load(std::memory_order_relaxed) == 1; }
/* The hooks' idx_split part (rad_fn4_* below): 1 / 0 / -1 as the core reads them. */
static int idx_fn4_accepts(const char* name, int value) {
    if (!name || std::strcmp(name, "idx_split")) return 0;
    return value == 0 || (value == 1 && g_idx_bal_ok.load(std::memory_order_relaxed)) ? 1 : -1;
}
static int idx_fn4_switch(const char* name, int value) {
    const int ok = idx_fn4_accepts(name, value);
    if (ok > 0) g_idx_bal.store(value, std::memory_order_relaxed);
    return ok;
}
static int idx_fn4_value(const char* name, int* applied) {
    if (!name || !applied || std::strcmp(name, "idx_split")) return 0;
    *applied = idx_balance() ? 1 : 0;
    return 1;
}

/* RADIANCE_DS41_INDEX: a list of `keep` (the candidate restriction) and `split` (the row split
 * over the ranks), or `off`; unset or empty is `keep,split`. Both are exact -- every selection is the one
 * the unrestricted, unsplit indexer makes (docs/OPS.md index_score; ds41_index_split.h) -- so the
 * switch is for an A/B in one image. The split needs every rank to attend (no attention-zero rank,
 * whose step skips the indexer and could not join the gather) and a step of two runs or more. */
static void index_mode(Model& m) {
    const char* e = std::getenv("RADIANCE_DS41_INDEX");
    const std::string v = e && *e ? e : "keep,split";   /* set but empty (a compose pass-through) = unset */
    const bool off = v == "off" || v == "0" || v == "none";
    m.ikeep = !off && v.find("keep") != std::string::npos && m.kwords > 0;
    m.isplit = !off && v.find("split") != std::string::npos && m.g.world > 1 && m.aw == m.g.world &&
               m.iparts >= 2;
    m.qmax = m.isplit ? (m.iparts + m.aw - 1) / m.aw : 0;
    /* RADIANCE_DS41_KEEP_MIN_LIVE / RADIANCE_DS41_SPLIT_MIN_LIVE (positions; unset or empty: twice
     * the candidate positions, 2 x topk_blocks x block = 32768 at DeepSeek-V4.1, and 4096). */
    auto envi = [](const char* n, int64_t d) {
        const char* v = std::getenv(n);
        return v && *v ? std::max<int64_t>(0, std::atoll(v)) : d;
    };
    m.keep_min_live = envi("RADIANCE_DS41_KEEP_MIN_LIVE", 2 * m.c.cand_blocks * std::max<int64_t>(1, m.c.cand_block));
    m.split_min_live = envi("RADIANCE_DS41_SPLIT_MIN_LIVE", kDs41SplitMinLive);
    const char* ct = std::getenv("RADIANCE_DS41_CTAIL");
    m.ctail_full = ct && !std::strcmp(ct, "full");
    const char* kx = std::getenv("RADIANCE_DS41_IDXKEY");
    m.ikmx = kx && !std::strcmp(kx, "mxfp4") && m.c.idim % 32 == 0;
    const char* lx = std::getenv("RADIANCE_DS41_LATENT");
    m.lnv = lx && !std::strcmp(lx, "nvfp4") && m.c.window % 32 == 0;
    idx_balance_env();
    m.w8 = m.w8d = false;
    if (const char* wx = std::getenv("RADIANCE_DS41_KVWIN_FP8"); wx && *wx && std::strcmp(wx, "0") && std::strcmp(wx, "off")) {
        /* the packed slot's readers take the 512-wide latent; the trunk's window keys end where its selections
         * start, so a multiple of 32 there (the kernels' key blocks) */
        m.w8 = m.c.head_dim == 512 && m.c.window % 32 == 0;
        m.w8d = m.c.head_dim == 512 && std::strcmp(wx, "trunk") != 0;
    }
}

/* RADIANCE_DS41_ARENA_DECODE_ROWS=<rows> (lever VA "arena"): the core adds an activation-arena level
 * below its 256-row floor for decode steps of at most <rows> rows (Engine::probe_arena_levels, which
 * reads the same variable), and a sizing declare below irows declares the indexer's score plane, its
 * selections and the split's part at its own rows instead of the real declare's capacity (Model::
 * score_rows): a step of at most irows rows is one run of exactly its rows (ds41_idx_runs), never
 * split, and every index op is issued over bcol / brow_slice views of those rows, so the plane needs
 * no more. Placement only: the kernels, their launch shapes and their operands' extents are the real
 * plan's (core/build/rad_bufplan.cpp plan_level refuses anything but fewer rows). At 512k context the
 * plane is 2 MiB a row, 512 MiB at 256 rows. Unset or 0: every sizing declare as today. */
static int64_t ds41_arena_decode_rows() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_ARENA_DECODE_ROWS");
        return e && *e ? std::max<int64_t>(0, std::atoll(e)) : (int64_t)0;
    }();
    return v;
}

/* RADIANCE_DS41_AR_PCN (default off until its TP4 A/B): at four ranks with the exact wire, a decode or verify step's
 * all-reduce and the mhc_post_collapse_rmsnorm after it as one collective (ar_mhc_post_collapse_rmsnorm, byte-identical
 * to the pair). "1" = at the attention's all-reduce (its post + the ffn collapse + norm); "2" = there and at the MoE's
 * (its post + the next layer's attention collapse + norm, or the last layer's + the final norm: the block leaves its
 * all-reduce to it, MoeFP8::ar_out / kArOutExact); unset or "0" = the pairs. */
static int ds41_ar_pcn_mode() {
    static const int m = [] {
        const char* e = std::getenv("RADIANCE_DS41_AR_PCN");
        return !e || !e[0] || e[1] ? 0 : e[0] == '1' ? 1 : e[0] == '2' ? 2 : 0;
    }();
    return m;
}
static int ds41_ar_pcn() { return ds41_ar_pcn_mode() >= 1; }
static int ds41_ar_pcn_moe() { return ds41_ar_pcn_mode() == 2; }

/* RADIANCE_DS41_ROUTER_FUSE: "1" = the selection and the sort as one launch (router_scored_topk_scatter,
 * where a loaded library declares it and the pass fits the sort's one workgroup; byte-identical to
 * router_scored_topk + moe_scatter); unset or anything else = the two launches. Default off since
 * 10.11. (gpu02 item 77: c4 -1.0..-1.4 %, 1 user 0 %: rejected, the switch kept). */
static int ds41_router_fuse() {
    static const int on = [] { const char* e = std::getenv("RADIANCE_DS41_ROUTER_FUSE"); return e && e[0] == '1' && !e[1]; }();
    return on ? 1 : 0;
}

/* RADIANCE_DS41_ROUTER_BF16=1 (lever VA "bf16"; read at declare): the f32 weights whose checkpoint is
 * BF16 -- every MoE router projection (trunk and DSpark, [n_expert, n_embd]) and the ratio > 1
 * compressors' fused kv|gate projection ([2 hd, n_embd]) -- are stored as bf16 instead of f32, half
 * the bytes (~150 + 30 MiB a card). Exact: the checkpoint holds them as BF16 (the release's
 * ffn.gate.weight and attn.compressor.wkv / wgate; the container's f32 router planes have zero low
 * halves), the loader stores those values unchanged (a narrowing of an exact bf16 value is that
 * value; the compressor's bf16 plane is no longer widened), and gemm_nt_f32 widens each weight to
 * that same f32 value before the same fmaf in the same order (libr4d gemm_f32_wval, held byte for
 * byte to the f32 storage by tests/r4d_emu/test_ds41vram.cpp). Unset or anything but "1": f32. */
static int ds41_router_bf16() {
    static const int on = [] { const char* e = std::getenv("RADIANCE_DS41_ROUTER_BF16"); return e && e[0] == '1' && !e[1]; }();
    return on ? 1 : 0;
}

/* A compressed latent entry's bf16 slots: its hd values, or its packed bytes / 2 (nvfp4). */
static int64_t latent_slots(const Model& m, int64_t hd) {
    return m.lnv ? (hd / 2 + hd / 16 + 15) / 16 * 16 / 2 : hd;
}

/* An index cache entry's bf16 slots: the key's idim values, or its packed bytes / 2 (mxfp4). */
static int64_t index_slots(const Model& m) {
    return m.ikmx ? (m.c.idim / 2 + m.c.idim / 32 + 15) / 16 * 16 / 2 : m.c.idim;
}

static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, W = m.work;
    const int64_t HN = c.hc * D, HD = g.n_head * c.head_dim;
    auto bf = [&](const char* n, std::initializer_list<int64_t> s) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, s);
    };
    m.h0 = bf("streams", { T, HN });
    m.h1 = bf("streams_next", { T, HN });
    m.xcol = bf("collapsed", { T, D });
    m.xn = bf("normed", { T, D });
    m.pre_a = decl_b(b, nm.f("pre_attn"), RAD_F32, { T, c.hc });
    m.pre_f = decl_b(b, nm.f("pre_ffn"), RAD_F32, { T, c.hc });
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, c.hc });
    m.comb = decl_b(b, nm.f("comb"), RAD_F32, { T, c.hc * c.hc });
    m.qr = bf("q_lora", { T, c.q_lora });
    m.qrn = bf("q_lora_normed", { T, c.q_lora });
    m.q = bf("q", { T, HD });
    m.kv = bf("window_kv", { T, c.head_dim });
    m.kvn = bf("window_kv_normed", { T, c.head_dim });
    m.o = bf("attn_o", { T, HD });
    m.oa = bf("attn_o_lora", { T, m.gr * c.o_rank });
    m.attn_out = bf("attn_out", { T, D });
    m.kvs = decl_b(b, nm.f("comp_kv_score"), RAD_F32, { T, 2 * c.head_dim });
    m.stage = decl_b(b, nm.f("comp_stage"), RAD_F32, { 2 * W, 2 * c.head_dim });
    m.page = decl_b(b, nm.f("comp_page"), RAD_I32, { W });
    m.bpos = decl_b(b, nm.f("comp_bpos"), RAD_I32, { W });
    m.nc = decl_b(b, nm.f("comp_nc"), RAD_I32, { g.max_seqs });
    const int64_t LR = std::max(T, W);
    m.pooled = bf("comp_pooled", { LR, c.head_dim });
    m.lat = bf("comp_latent", { LR, c.head_dim });
    m.latn = bf("comp_latent_normed", { LR, c.head_dim });
    m.ik = bf("index_k", { LR, c.idim });
    m.ikn = bf("index_k_normed", { LR, c.idim });
    m.iq = bf("index_q", { T, c.ih * c.idim });
    m.iw = bf("index_w", { T, c.ih });
    /* score_rows (lever VA "arena", a sizing declare below irows): no split at such a step, so the
     * selections are its rows too. */
    m.score = decl_b(b, nm.f("index_score"), RAD_F32, { m.score_rows ? m.score_rows : m.irows, m.cmax });
    if (m.kwords > 0) m.keep = decl_b(b, nm.f("index_keep"), RAD_I32, { T, m.kwords });
    /* Split: the gather lays aw parts of qmax runs of at most irows rows, rank r's at r qmax R. */
    /* lever BI: a prefill segment's split runs are laid from its first row, so up to a decode segment's rows (48)
     * past the unsplit plane; lever PSK: a cut segment's runs from its cut row, up to the ranks' runs past the step */
    const int64_t sel_rows = (m.isplit && !m.score_rows ? std::max(T, (int64_t)m.aw * m.qmax * m.irows) : T) +
                             (bi_declared() || psk_declared() ? (int64_t)m.aw * m.irows + 64 : 0);
    m.sel = decl_b(b, nm.f("index_sel"), RAD_I32, { sel_rows, c.itopk });
    if (m.isplit)
        m.sel_part = decl_b(b, nm.f("index_sel_part"), RAD_I32,
                            { m.score_rows ? std::min(m.score_rows, m.qmax * m.irows) : m.qmax * m.irows, c.itopk });
    m.cs_a = decl_b(b, nm.f("rope_window"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    m.cs_b = decl_b(b, nm.f("rope_compressed"), RAD_F32, { g.max_ctx, c.rope }, RAD_BUF_PERSIST);
    if (!c.ple.empty()) {
        m.ctok = decl_b(b, nm.f("engram_tokens"), RAD_I32, { T });
        /* The gathered rows are host domain, as rad_block_ple.h's: a host gather (the table on the
         * disk or RAM tier) whose operands are all host-arena buffers joins on the last device op
         * that touched the host arena, where a device-domain operand drains the stream
         * (core/runtime/issue.cpp). A cast copies them onto the card for the GEMM, which reads
         * each row band once per column block. */
        m.emb = decl_b(b, nm.f("engram_rows"), RAD_BF16, { T, c.ple_heads * c.ple_dim },
                       RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        m.embd = bf("engram_rows_dev", { T, c.ple_heads * c.ple_dim });
        m.ekv = bf("engram_kv", { T, (c.hc + 1) * D });
        /* lever E1 (RADIANCE_DS41_ENGDD declared): the unique rows' kv plane (the layers declare their compact ids and maps) */
        if (engdd_declared()) m.ekv_u = bf("engram_kv_u", { T, (c.hc + 1) * D });
    }
    /* The routed FFN's wires (rad_block_moe_fp8.h), bf16 throughout. */
    /* bf16 linears: each ActFP8 carries only its activation (`x`); the plugin norms the FFN input
     * itself (MoeFP8's external-input form) and the block leaves its delta in `act.x`. */
    m.act.x = bf("ffn_normed", { T, D });
    /* A quantised dense trunk reads this pair, one f32 scale a 128 columns, shared by every dense
     * IQR GEMM. The activation quantiser (had_quant_act_i8g) checks the code and scale planes
     * against the linear's own K, and a row slice does not narrow the column, so the pair is the
     * hidden width -- the width every dense linear but the Engram gather reads. An Engram linear
     * wider than that declares its own pair. */
    const int64_t QK = D;
    m.iqr_q8 = decl_b(b, nm.f("iqr_q8"), RAD_I8, { T, QK });
    m.iqr_s8 = decl_b(b, nm.f("iqr_s8"), RAD_F32, { T, fp8_blocks(QK) });
    /* The shared expert's input is the same normed activation; its own quantiser writes the pair. */
    m.act.q8 = m.iqr_q8;
    m.act.s8 = m.iqr_s8;
    auto& mw = m.mw;
    mw.x = m.xcol;
    mw.h = m.act;
    mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, c.n_expert });
    mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, c.top_k });
    mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, c.top_k });
    mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { c.n_expert + 1 });
    mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { c.n_expert }, RAD_BUF_PERSIST);
    mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { T * c.top_k });
    mw.egu = bf("expert_gate_up", { T * c.top_k, 2 * c.n_ff });
    mw.eff.x = bf("expert_ff", { T * c.top_k, c.n_ff });
    mw.edn = bf("expert_down", { T * c.top_k, D });
    /* the shared expert: this rank's width; none where its slice is empty (shared_slice) */
    if (g.n_ff > 0) {
        mw.sgu = bf("shared_gate_up", { T, 2 * g.n_ff });
        mw.sff.x = bf("shared_ff", { T, g.n_ff });
    }
    mw.sout = bf("shared_out", { T, D });
    mw.sgate = bf("shared_one", { T, 1 });
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, D }, RAD_BUF_PERSIST);
    {   /* lever BI: the row pitch of each per-token buffer step_bi slices (its declared width) */
        const std::pair<rad_buf, int64_t> bp[] = {
            { m.h0, HN }, { m.h1, HN }, { m.xcol, D }, { m.xn, D }, { m.pre_a, c.hc }, { m.pre_f, c.hc }, { m.post, c.hc },
            { m.comb, c.hc * c.hc }, { m.qr, c.q_lora }, { m.qrn, c.q_lora }, { m.q, HD }, { m.kv, c.head_dim },
            { m.kvn, c.head_dim }, { m.o, HD }, { m.oa, m.gr * c.o_rank }, { m.attn_out, D }, { m.kvs, 2 * c.head_dim },
            { m.iq, c.ih * c.idim }, { m.iw, c.ih }, { m.sel, c.itopk }, { m.act.x, D }, { m.final_hidden, D },
            { m.embd, c.ple_heads * c.ple_dim }, { m.ekv, (c.hc + 1) * D }, { m.ctok, 1 }, { m.iqr_q8, QK },
            { m.ekv_u, (c.hc + 1) * D },
            { m.iqr_s8, fp8_blocks(QK) }, { m.mw.sgate, 1 } };
        for (const auto& e : bp)
            if (e.first) m.bp[e.first] = e.second;
    }
    m.logit_input = bf("logit_input", { g.max_logit_rows, D });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    return rad_declare_logits(b, m.logits);
}

/* The caches. The window holds the whole rotated latent per slot (K then V halves); a compressed or
 * index group holds one entry a block of `ratio` slots (entry_store fixes the block size); the
 * compressor's tail is a per-sequence ring of r + max_spec f32 rows. Declared identically on every
 * rank (an attention-zero rank zeroes them). */
/* A window key's bf16 slots: hd, or (lever KVWIN) the fp8 window slot's (hd + hd/32) bytes rounded up to 16. */
static int64_t window_slots(int64_t hd, bool w8) { return w8 ? (hd + hd / 32 + 15) / 16 * 16 / 2 : hd; }
static rad_kvgroup win_group(RadBuilder* b, Model& m) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_WINDOW;
    d.dtype = RAD_BF16;
    d.window = m.c.window + m.g.max_spec;
    d.n_head_kv = 1;
    d.head_dim = window_slots(m.c.head_dim, m.w8) / 2;
    return rad_decl_kv_group(b, m.nm.f("kv_window"), &d);
}
/* The window key's store into a window group: kv_store of its K and V halves, or (lever KVWIN) window_store's
 * packed slot. Declared right after the group, its block_size sets the group's. */
static rad_op window_key_store(RadBuilder* b, int64_t rows, int64_t hd, bool w8, rad_buf x) {
    if (w8)
        return op(b, "window_store",
                  { RAD_RANGE("M", 1, rows), RAD_INT("n", hd), RAD_INT("block_size", 16),
                    RAD_STR("format", "fp8_e4m3"), RAD_STR("dtype", "bf16") },
                  {}, { x }, {});
    return op(b, "kv_store",
              { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                RAD_INT("block_size", 16), RAD_STR("kv_dtype", "bf16") },
              {}, { x }, {});
}
static rad_kvgroup tail_group(RadBuilder* b, Model& m, int64_t r) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = 1;
    d.state_dim[0] = r + m.g.max_spec;
    d.state_dim[1] = 2 * m.c.head_dim;
    return rad_decl_kv_group(b, m.nm.f("kv_comp_tail_r%lld", (long long)r), &d);
}
static rad_kvgroup full_group(RadBuilder* b, Model& m, const char* fmt, int64_t n, int64_t r) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = n / r / 2;
    return rad_decl_kv_group(b, m.nm.f(fmt, (long long)r), &d);
}

/* The routed FFN's checkpoint tensors at (absolute) layer `l`: a GGUF's stacks (ffn_gate_exps,
 * ffn_up_exps, ffn_down_exps, each [n_expert, ...]), or the release's experts one by one
 * (`<layer>.ffn.experts.<e>.w1/w3/w2`, gate/up/down as model.py's Expert names them). */
static void moe_names(Model& m, int l, MoeFP8::Src* s) {
    auto& nm = m.nm;
    s->router = ckpt_name(nm.f("blk.%d.ffn_gate_inp.weight", l));
    s->router_bias = ckpt_name(nm.f("blk.%d.exp_probs_b.bias", l));
    s->shared_mlp_gate = ckpt_name(nm.f("blk.%d.ffn_gate_shexp.weight", l));
    s->shared_mlp_up = ckpt_name(nm.f("blk.%d.ffn_up_shexp.weight", l));
    s->shared_mlp_down = ckpt_name(nm.f("blk.%d.ffn_down_shexp.weight", l));
    if (m.release) {
        s->experts = l < m.c.n_layer ? nm.ckpt("layers.%d.ffn.experts", l)
                                     : nm.ckpt("mtp.%d.ffn.experts", l - (int)m.c.n_layer);
        s->expert_gate = "w1";
        s->expert_up = "w3";
        s->expert_down = "w2";
        return;
    }
    s->experts_gate = nm.f("blk.%d.ffn_gate_exps.weight", l);
    s->experts_up = nm.f("blk.%d.ffn_up_exps.weight", l);
    s->experts_down = nm.f("blk.%d.ffn_down_exps.weight", l);
}

/* ------------------------------------------------------------------ one layer */
/* RADIANCE_DS41_AR_PCN=2: the MoE all-reduce of layer `P` and the mhc_post_collapse_rmsnorm after it (the next
 * layer's pcn_a with gain `gw` into `y`, or pcn_fin into the final hidden) as one four-rank collective, declared up to
 * ar_wide_write_rows (decode and verify steps). Where it resolves, `P`'s MoE block leaves its all-reduce to the caller
 * at the T ar_exact_write_takes (= ar_wide_write_exact at four ranks), the same T at which the issue takes `*out`. */
static int ar_pcn_moe_declare(RadBuilder* b, Model& m, Layer& P, int64_t T, rad_weight gw, rad_buf y, rad_op* out) {
    const auto& g = m.g;
    const auto& c = m.c;
    *out = 0;
    if (!ds41_ar_pcn_moe() || g.world != 4 || !P.moe.op_ar || P.moe.ar_out) return RAD_OK;
    const int64_t rr = std::min<int64_t>(T, ar_wide_write_rows(g));
    if (rr < 1) return RAD_OK;
    *out = fused(b, "ar_mhc_post_collapse_rmsnorm",
                 { RAD_RANGE("M", 1, rr), RAD_INT("n", c.n_embd), RAD_INT("hc", c.hc), RAD_F64("eps", c.eps),
                   RAD_INT("world_size", g.world), RAD_STR("dtype", "bf16"), RAD_INT("wire", 0) },
                 { gw }, { m.act.x, m.h1, m.post, m.comb, m.pre_f }, { m.act.x, m.h0, m.xcol, y });
    if (*out) {
        P.moe.ar_out = true;
        P.moe.ar_out_take = kArOutExact;
    }
    return RAD_OK;
}

static int layer(RadBuilder* b, Model& m, Layer& L, int l) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int64_t T = g.max_tok, D = c.n_embd, HD = g.n_head * c.head_dim, W = m.work;
    const int64_t hd = c.head_dim, mix = c.hc * (c.hc + 2);
    const auto grp = grp_layer(l);
    L.l = l;
    L.r = c.ratio[(size_t)l];
    L.src = c.is_src(l);
    L.idx = c.is_idx(l);
    auto w = [&](const char* fmt, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wt(b, nm.f(fmt, l), dt, shape, grp);
    };

    /* ---- Engram, on the streams in front of the block */
    const int pi = c.ple_index(l);
    if (pi >= 0) {
        const int64_t H = c.ple_heads;
        /* The hash constants: GGUF metadata arrays (llama.cpp's converter computes them), or the
         * release's, which the core derives from its config.json and tokenizer as
         * inference/engram.py does at load (core/format/engram.h). One layout either way. */
        const char* tab = m.release ? "engram." : "deepseek41.ple.";
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_mult", l),
                        { { nm.ckpt("%slayer_multipliers", tab), (int)(pi * c.ngram), c.ngram } }));
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_vsz", l),
                        { { nm.ckpt("%shead_vocab_sizes", tab), (int)(pi * H), H } }));
        DS_TRY(map_rows(b, nm.f("blk.%d.engram_off", l),
                        { { nm.ckpt("%shead_offsets", tab), (int)(pi * H), H } }));
        L.e_mult = decl_w(b, nm.f("blk.%d.engram_mult", l), RAD_I64, { c.ngram },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        L.e_vsz = decl_w(b, nm.f("blk.%d.engram_vsz", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_off = decl_w(b, nm.f("blk.%d.engram_off", l), RAD_I64, { H }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
        L.e_kv = lw(b, m, nm.f("blk.%d.engram_kv.weight", l), (c.hc + 1) * D, H * c.ple_dim, grp);
        if (m.release) {
            /* The release ships the gate as two factors, k_weight and q_weight, that the model only
             * multiplies (model.py Engram): one gain, as llama.cpp's converter folds it. */
            const char* gn = nm.f("blk.%d.engram_gate.weight", l);
            DS_TRY(map_mul(b, gn, nm.ckpt("layers.%d.engram.q_weight", l),
                           nm.ckpt("layers.%d.engram.k_weight", l)));
            L.e_gate = decl_w(b, gn, RAD_F32, { c.hc * D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        } else {
            L.e_gate = w("blk.%d.engram_gate.weight", RAD_F32, { c.hc * D });
        }
        if (!m.tok) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_CONV;
            d.dtype = RAD_I32;
            d.conv_width = c.ngram;
            d.n_head_kv = 1;
            d.head_dim = 1;
            m.tok = rad_decl_kv_group(b, nm.f("kv_engram_tokens"), &d);
            if (!m.tok) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.tok));
        /* Host domain, like the rows (see m.emb), and persistent rather than liveness-shared:
         * liveness runs over the declared op order, and step() writes these ids before the first
         * block, far ahead of this layer's gather. */
        L.ids = decl_b(b, nm.f("blk.%d.engram_ids", l), RAD_I32, { T, H }, RAD_BUF_PERSIST,
                       RAD_DOMAIN_HOST);
        L.e_ids = op(b, "ngram_ids",
                     { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("ngram", c.ngram),
                       RAD_INT("eos", c.pad) },
                     { L.e_mult, L.e_vsz, L.e_off }, { m.ctok }, { L.ids });
        if (m.ple_group > 0 && m.ple_tab)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_INT("group", m.ple_group),
                           RAD_STR("dtype", "bf16") },
                         { m.ple_codes, m.ple_scale, m.ple_tab }, { L.ids }, { m.emb });
        else if (m.ple_group > 0)
            L.e_emb = op(b, "embed_lookup_planes",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_INT("group", m.ple_group),
                           RAD_STR("dtype", "bf16") },
                         { m.ple_codes, m.ple_scale }, { L.ids }, { m.emb });
        else
            L.e_emb = op(b, "embed_lookup",
                         { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim),
                           RAD_INT("n_vocab", c.ple_rows), RAD_STR("dtype", "bf16") },
                         { m.ple_table }, { L.ids }, { m.emb });
        /* Lever E1: the same lookup over the layer's compact ids, declared before the copy so the rows' plane stays live
         * from it to the copy (the order engdd_issue issues them in). Only for a GGUF-block kv weight (gemm_nt_ggml: the
         * per-row form above the band ds41_engram_dedup.h rests on). */
        const bool edd = m.ekv_u && m.wty.count(L.e_kv) && !m.wsc.count(L.e_kv);
        if (edd) {
            L.edd_ids = decl_b(b, nm.f("blk.%d.engram_ids_dd", l), RAD_I32, { T, H }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
            if (L.edd_ids) m.bp[L.edd_ids] = H;
            if (m.ple_group > 0)
                L.e_emb_dd = op(b, "embed_lookup_planes",
                                { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim), RAD_INT("n_vocab", c.ple_rows),
                                  RAD_INT("group", m.ple_group), RAD_STR("dtype", "bf16") },
                                m.ple_tab ? std::vector<rad_weight>{ m.ple_codes, m.ple_scale, m.ple_tab }
                                          : std::vector<rad_weight>{ m.ple_codes, m.ple_scale },
                                { L.edd_ids }, { m.emb });
            else
                L.e_emb_dd = op(b, "embed_lookup",
                                { RAD_RANGE("M", 1, T * H), RAD_INT("n_embd", c.ple_dim), RAD_INT("n_vocab", c.ple_rows),
                                  RAD_STR("dtype", "bf16") },
                                { m.ple_table }, { L.edd_ids }, { m.emb });
        }
        L.e_copy = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", H * c.ple_dim), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.emb }, { m.embd });
        L.e_gemm = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.embd, m.ekv);
        /* Lever E1: the GEMM over the unique rows into engram_kv_u, then every pooled row's kv row gathered from it
         * (gather_rows, a byte copy) through this layer's map */
        if (edd) {
            L.edd_map = decl_b(b, nm.f("blk.%d.engram_dd_map", l), RAD_I32, { T }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
            if (L.edd_map) m.bp[L.edd_map] = 1;
            L.e_gemm_u = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.embd, m.ekv_u);
            L.e_scat = op(b, "gather_rows", { RAD_RANGE("M", 1, T), RAD_INT("n", (c.hc + 1) * D), RAD_STR("dtype", "bf16") },
                          {}, { m.ekv_u, L.edd_map }, { m.ekv });
        }
        L.e_gate_op = op(b, "stream_gate_add",
                         { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                           RAD_F64("eps", c.eps), RAD_F64("gate_eps", 1e-6),
                           RAD_STR("dtype", "bf16") },
                         { L.e_gate }, { m.h0, m.ekv }, { m.h1 });
        /* The block reads the streams from h0: the gated streams go back there. */
        L.e_back = op(b, "cast",
                      { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc * D), RAD_STR("from", "bf16"),
                        RAD_STR("to", "bf16") },
                      {}, { m.h1 }, { m.h0 });
    }

    /* ---- a DSpark target: the attention input's hc mean (model.py main_hiddens), after the Engram */
    if (m.dsp.on && !m.zero)   /* the context pass is an attention rank's */
        for (size_t j = 0; j < m.dsp.taps.size(); ++j)
            if (m.dsp.taps[j] == l) {
                L.tap_col = (int)j;
                L.tap = op(b, "mhc_contract",
                           { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                           {}, { m.h0 }, { m.dsp.src });
                if (!L.tap) return RAD_E_INVAL;
            }

    /* ---- the attention connection */
    L.hc_a_fn = w("blk.%d.hc_attn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_a_scale = w("blk.%d.hc_attn_scale.weight", RAD_F32, { 3 });
    L.hc_a_base = w("blk.%d.hc_attn_base.weight", RAD_F32, { mix });
    const std::vector<RadParam> mixp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                         RAD_INT("iterations", c.hc_iters), RAD_F64("eps", c.eps),
                                         RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") };
    const std::vector<RadParam> hcp = { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                        RAD_STR("dtype", "bf16") };
    L.mix_a = op(b, "mhc_mix", mixp, { L.hc_a_fn, L.hc_a_scale, L.hc_a_base }, { m.h0 },
                 { m.pre_a, m.post, m.comb });
    if (!m.zero) {
        L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
        /* The previous layer's post_f writes h0 straight into this collapse unless an Engram
         * block (above) rewrites h0 between them. */
        if (l > 0 && !L.e_ids)
            L.pcn_a = post_collapse_norm(b, m, T, L.attn_norm, m.act.x, m.h1, m.h0, m.pre_f,
                                         m.xcol, m.xn);
        if (!L.pcn_a) L.cn_a = collapse_norm(b, m, T, L.attn_norm, m.h0, m.pre_f, m.xcol, m.xn);
    }
    if (!L.cn_a && !L.pcn_a) L.col_a = op(b, "mhc_collapse", hcp, {}, { m.h0, m.pre_f }, { m.xcol });
    if (L.pcn_a && l > 0) DS_TRY(ar_pcn_moe_declare(b, m, m.layers[(size_t)l - 1], T, L.attn_norm, m.xn, &L.ar_pcn_a));
    if (m.zero) {
        /* Attention-zero (Geom::attn_world): no heads on this rank. Its caches are declared and
         * bound as on an attention rank, the weightless stores that fix their block sizes declared
         * but never issued, then zeroed (rad_kv_group_zero); the block's output is a fill of zeros
         * the all-reduce adds. */
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = window_key_store(b, T, hd, m.w8, m.kvn);
        if (!m.win_bs && (m.win_bs = rad_kv_block_size(b, m.win)) < 1) return RAD_E_INVAL;
        DS_TRY(rad_bind_layer_kv(b, l, m.win));
        DS_TRY(rad_kv_group_zero(b, m.win));
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            if (r > 1) {
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                /* Per-sequence state is kept on every rank (only paged groups zero): this rank's
                 * copy of the ring is small and never written. */
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
            }
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", index_slots(m), r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            DS_TRY(rad_kv_group_zero(b, m.index[r]));
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", latent_slots(m, hd), r))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, m.lnv ? "nvfp4" : nullptr);
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            DS_TRY(rad_kv_group_zero(b, m.comp[r]));
        }
        L.attn_fill = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { m.attn_out });
    } else {
        if (!L.cn_a && !L.pcn_a) L.norm_a = rmsnorm(b, m, T, L.attn_norm, D, m.xcol, m.xn);

        /* ---- attention: the query, the window key */
        L.wq_a = lw(b, m, nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D, grp);
        L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
        L.wq_b = lw(b, m, nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, grp, RAD_SHARD_ROW,
                    m.ar * HD, (m.ar + 1) * HD);
        L.q_a = linear(b, m, T, L.wq_a, c.q_lora, D, m.xn, m.qr);
        L.q_an = rmsnorm(b, m, T, L.q_norm, c.q_lora, m.qr, m.qrn);
        L.q_b = linear(b, m, T, L.wq_b, HD, c.q_lora, m.qrn, m.q);
        /* Lever K1 fold a (RADIANCE_DS41_QNFOLD): the pair as one op beside it, where q_b is a GGUF weight (gemm_nt_ggml,
         * not an IQR view) and a library serves the fold; the pair stays declared for the steps the switch leaves to it. */
        if (L.q_an && L.q_b && g_sw_qnf_env.load(std::memory_order_relaxed) > 0 && !m.wsc.count(L.wq_b)) {
            const auto ty = m.wty.find(L.wq_b);
            if (ty != m.wty.end())
                L.q_bf = fused(b, "rmsnorm_gemm_nt_ggml",
                               { RAD_RANGE("M", 1, T), RAD_INT("N", HD), RAD_INT("K", c.q_lora), RAD_STR("type", ty->second),
                                 RAD_STR("dtype", m.g.dtype), RAD_F64("eps", m.c.eps) },
                               { L.wq_b, L.q_norm }, { m.qr }, { m.qrn, m.q });
            if (L.q_bf) g_sw_qnf_ok.store(1, std::memory_order_relaxed);
        }
        const rad_buf cs = L.r > 0 ? m.cs_b : m.cs_a;
        L.q_rope = rope_rows(b, m, T, g.n_head, hd, 0, m.q, cs);
        L.wkv = lw(b, m, nm.f("blk.%d.attn_kv.weight", l), hd, D, grp);
        L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
        L.kv_g = linear(b, m, T, L.wkv, hd, D, m.xn, m.kv);
        L.kv_f = norm_rope_rt(b, m, T, L.kv_norm, hd, 32, "fp8_e4m3", "ue8m0", m.kv, m.kvn, cs);
        if (!L.kv_f) {
            L.kv_n = rmsnorm(b, m, T, L.kv_norm, hd, m.kv, m.kvn);
            L.kv_rope = rope_rows(b, m, T, 1, hd, 0, m.kvn, cs);
            L.kv_q = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", m.kvn);
        }
        if (!m.win && !(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = window_key_store(b, T, hd, m.w8, m.kvn);
        if (!m.win_bs) {
            m.win_bs = rad_kv_block_size(b, m.win);
            if (m.win_bs < 1) return RAD_E_INVAL;
        }
        DS_TRY(rad_bind_layer_kv(b, l, m.win));
        /* Lever K1 fold c (RADIANCE_DS41_KVFOLD): kv_f and kv_st as one op where a library serves it, declared
         * after the window group's block size is fixed (its block_size must not claim a group: kv_st did). */
        if (L.kv_f && g_sw_kvf_env.load(std::memory_order_relaxed) > 0) {
            L.kv_fs = fused(b, "rmsnorm_rope_roundtrip_kv_store",
                            { RAD_RANGE("M", 1, T), RAD_INT("n", hd), RAD_F64("eps", m.c.eps), RAD_INT("n_head", 1),
                              RAD_INT("head_dim", hd), RAD_INT("rot", m.c.rope), RAD_INT("off", hd - m.c.rope),
                              RAD_STR("mode", "gptj"), RAD_INT("inverse", 0), RAD_INT("block", 32),
                              RAD_STR("format", "fp8_e4m3"), RAD_STR("scale", "ue8m0"), RAD_STR("dtype", "bf16"),
                              RAD_INT("block_size", m.win_bs), RAD_STR("kv_dtype", m.w8 ? "fp8_e4m3" : "bf16") },
                            { L.kv_norm }, { m.kv, cs }, { m.kvn });
            if (L.kv_fs) g_sw_kvf_ok.store(1, std::memory_order_relaxed);
        }

        /* ---- the compressor and the index keys, at a kv source */
        const int64_t r = L.r;
        if (L.src) {
            const int64_t rows = r > 1 ? W : T;
            L.comp_norm = w("blk.%d.attn_compressor_norm.weight", RAD_F32, { hd });
            if (r > 1) {
                const char* cn = nm.f("blk.%d.attn_compressor_kvg.weight", l);
                DS_TRY(map_concat(b, cn, { ckpt_name(nm.f("blk.%d.attn_compressor_kv.weight", l)),
                                           ckpt_name(nm.f("blk.%d.attn_compressor_gate.weight", l)) }));
                L.comp_w = decl_w(b, cn, ds41_router_bf16() ? RAD_BF16 : RAD_F32, { 2 * hd, D },
                                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
                L.c_gemm = op(b, "gemm_nt_f32",
                              { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * hd), RAD_INT("K", D),
                                RAD_STR("dtype", "bf16") },
                              { L.comp_w }, { m.xn }, { m.kvs });
                if (!m.tail.count(r) && !(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
                DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
                L.c_work = op(b, "qsa_work",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_INT("work", W), RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, { m.stage, m.page, m.bpos, m.nc });
                L.c_pool = op(b, "softmax_pool",
                              { RAD_RANGE("M", 1, W), RAD_INT("n", hd), RAD_INT("ratio", r),
                                RAD_STR("dtype", "bf16") },
                              {}, { m.stage, m.page }, { m.pooled });
                L.c_norm = rmsnorm(b, m, W, L.comp_norm, hd, m.pooled, m.lat);
            } else {
                L.comp_w = lw(b, m, nm.f("blk.%d.attn_compressor_kv.weight", l), hd, D, grp);
                L.c_gemm = linear(b, m, T, L.comp_w, hd, D, m.xn, m.pooled);
                L.c_norm = rmsnorm(b, m, T, L.comp_norm, hd, m.pooled, m.lat);
            }
            /* The compressed rows' angles, at each entry's first-token position. */
            L.c_tab = op(b, "rope_table",
                         { RAD_RANGE("M", 1, rows), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                           RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                           RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                           RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                           RAD_F64("yarn_magnitude", 1.0) },
                         {}, { m.bpos }, { m.cs_b });
            /* Index keys from the unrotated latent, then the latent itself rotated and quantised. */
            L.ik_w = lw(b, m, nm.f("blk.%d.indexer.attn_k.weight", l), c.idim, hd, grp);
            L.ik_norm = w("blk.%d.indexer.k_norm.weight", RAD_F32, { c.idim });
            L.k_gemm = linear(b, m, rows, L.ik_w, c.idim, hd, m.lat, m.ik);
            L.k_f = norm_rope_rt(b, m, rows, L.ik_norm, c.idim, 32, "fp4_e2m1", "ue8m0", m.ik, m.ikn,
                                 m.cs_b);
            if (!L.k_f) {
                L.k_norm = rmsnorm(b, m, rows, L.ik_norm, c.idim, m.ik, m.ikn);
                L.k_rope = rope_rows(b, m, rows, 1, c.idim, 0, m.ikn, m.cs_b);
                L.k_q = roundtrip(b, rows, c.idim, 32, "fp4_e2m1", "ue8m0", m.ikn);
            }
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", index_slots(m), r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            L.l_f = rope_rt(b, m, rows, 1, hd, 0, 16, "fp4_e2m1", "e4m3", m.lat, m.cs_b);
            if (!L.l_f) {
                L.l_rope = rope_rows(b, m, rows, 1, hd, 0, m.lat, m.cs_b);
                L.l_q = roundtrip(b, rows, hd, 16, "fp4_e2m1", "e4m3", m.lat);
            }
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", latent_slots(m, hd), r))) return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, m.lnv ? "nvfp4" : nullptr);
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
            if (r > 1)
                L.c_tail = op(b, "qsa_tail_store",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", 2 * hd), RAD_INT("ratio", r),
                                RAD_INT("ring", r + g.max_spec), RAD_INT("seqs", g.max_seqs),
                                RAD_STR("dtype", "f32") },
                              {}, { m.kvs }, {});
        }

        /* ---- the indexer, at an index source */
        if (L.idx && r > 0) {
            L.iq_w = lw(b, m, nm.f("blk.%d.indexer.attn_q_b.weight", l), c.ih * c.idim, c.q_lora, grp);
            L.iproj = lw(b, m, nm.f("blk.%d.indexer.proj.weight", l), c.ih, D, grp);
            L.i_q = linear(b, m, T, L.iq_w, c.ih * c.idim, c.q_lora, m.qrn, m.iq);
            L.i_f = rope_rt(b, m, T, c.ih, c.idim, 0, 32, "fp4_e2m1", "ue8m0", m.iq, m.cs_b);
            if (!L.i_f) {
                L.i_rope = rope_rows(b, m, T, c.ih, c.idim, 0, m.iq, m.cs_b);
                L.i_qq = roundtrip(b, T, c.ih * c.idim, 32, "fp4_e2m1", "ue8m0", m.iq);
            }
            L.i_w = linear(b, m, T, L.iproj, c.ih, D, m.xn, m.iw);
            const int64_t width = g.max_ctx / r + 1, IR = m.irows;
            /* A layer past the candidate layer scores only its kept blocks (m.ikeep): index_topk
             * reads no other position of the plane, so the selection is the same set. */
            const bool kept = m.ikeep && c.cand >= 0 && c.cand < l && m.keep && c.ratio[(size_t)c.cand] == r;
            L.i_keep = kept;
            /* One index_score a run of rows (index_runs()): run p scores step rows [p R, p R + R)
             * into the score plane's first R rows. */
            for (int64_t p = 0; p < m.iparts; ++p) {
                std::vector<RadParam> sp = { RAD_RANGE("M", 1, IR), RAD_INT("heads", c.ih), RAD_INT("dim", c.idim),
                                             RAD_INT("ratio", r), RAD_INT("width", width),
                                             RAD_F64("wscale", 1.0 / std::sqrt((double)c.idim * (double)c.ih)),
                                             RAD_STR("dtype", "bf16"), RAD_INT("row_part", p) };
                if (kept) sp.push_back(RAD_INT("block", c.cand_block));
                if (m.ikmx) sp.push_back(RAD_STR("key_format", "mxfp4"));
                const rad_op h = op(b, "index_score", sp, {},
                                    kept ? std::initializer_list<rad_buf>{ m.iq, m.iw, m.keep }
                                         : std::initializer_list<rad_buf>{ m.iq, m.iw },
                                    { m.score });
                if (!h) return RAD_E_INVAL;
                L.i_runs.push_back(h);
            }
            L.i_score = L.i_runs[0];
            /* The split's join, declared at the first indexer (before any attention reads sel). A
             * gather no library serves (no peer access, say) leaves the indexer unsplit. */
            if (m.isplit && !m.sel_gather) {
                m.sel_gather = op(b, "all_gather",
                                  { RAD_INT("world_size", g.world), RAD_INT("numel", m.qmax * IR * c.itopk),
                                    RAD_STR("dtype", "f32") },
                                  {}, { m.sel_part }, { m.sel });
                if (!m.sel_gather) {
                    m.isplit = false;
                    rad_note(b, "deepseek41: the indexer's row split is off: no all_gather of its selections "
                                "resolves at %d ranks", g.world);
                }
            }
            if (l == c.cand) {
                if (!m.keep) return RAD_E_INVAL;
                L.i_cand = op(b, "index_candidates",
                              { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                RAD_INT("block", c.cand_block), RAD_INT("topk_blocks", c.cand_blocks),
                                RAD_INT("ratio", r) },
                              {}, { m.score }, { m.keep });
                if (!L.i_cand) return RAD_E_INVAL;
            }
            const bool uses = c.cand >= 0 && c.cand < l;
            if (uses && (!m.keep || c.ratio[(size_t)c.cand] != r)) {
                fprintf(stderr, "radiance: deepseek41: layer %d reads the candidate layer's blocks at "
                                "another ratio (%lld) or there are none\n", l, (long long)r);
                return RAD_E_UNSUPPORTED;
            }
            std::vector<RadParam> tp = { RAD_RANGE("M", 1, IR), RAD_INT("width", width),
                                         RAD_INT("topk", c.itopk), RAD_INT("ratio", r) };
            if (uses) tp.push_back(RAD_INT("block", c.cand_block));
            /* Split, a rank's selections go to its part, which the gather joins into sel. */
            const rad_buf sel_out = m.isplit ? m.sel_part : m.sel;
            L.i_topk = op(b, "index_topk", tp, {},
                          uses ? std::initializer_list<rad_buf>{ m.score, m.keep }
                               : std::initializer_list<rad_buf>{ m.score },
                          { sel_out });
            if (L.i_topk && m.isplit && rw(b, L.i_topk, {}, { m.sel }) == 0) return RAD_E_INVAL;
            if (!L.i_topk) return RAD_E_INVAL;
        }

        /* ---- the sparse attention and the output projection */
        L.sinks = wts(b, g, nm.f("blk.%d.attn_sinks.weight", l), RAD_F32, { g.n_head }, grp, RAD_SHARD_ROW,
                      m.ar * g.n_head, (m.ar + 1) * g.n_head);
        std::vector<RadParam> ap = { RAD_RANGE("M", 1, T), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                                     RAD_INT("window", c.window), RAD_INT("ratio", r),
                                     RAD_INT("topk", r > 0 ? c.itopk : 0),
                                     RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") };
        if (m.lnv && r > 0) ap.push_back(RAD_STR("latent_format", "nvfp4"));
        if (m.w8) ap.push_back(RAD_STR("window_format", "fp8_e4m3"));
        L.attn = op(b, "attn_sparse_sink", ap,
                    { L.sinks },
                    r > 0 ? std::initializer_list<rad_buf>{ m.q, m.sel }
                          : std::initializer_list<rad_buf>{ m.q },
                    { m.o });
        L.o_f = rope_rt(b, m, T, g.n_head, hd, 1, 32, "fp8_e4m3", "ue8m0", m.o, cs);
        if (!L.o_f) {
            L.o_rope = rope_rows(b, m, T, g.n_head, hd, 1, m.o, cs);
            L.o_q = roundtrip(b, T, HD, 32, "fp8_e4m3", "ue8m0", m.o);
        }
        /* Lever K1 fold d (RADIANCE_DS41_OFOLD): the attention and its output's inverse rope + round trip as one
         * op where a library serves it (the decode band in one launch); the pair above stays declared. */
        if (g_sw_of_env.load(std::memory_order_relaxed) > 0) {
            std::vector<RadParam> fp = ap;
            fp.push_back(RAD_INT("rot", m.c.rope));
            fp.push_back(RAD_INT("off", hd - m.c.rope));
            fp.push_back(RAD_STR("mode", "gptj"));
            fp.push_back(RAD_INT("inverse", 1));
            fp.push_back(RAD_INT("block", 32));
            fp.push_back(RAD_STR("format", "fp8_e4m3"));
            fp.push_back(RAD_STR("block_scale", "ue8m0"));
            if (r > 0)
                L.attn_rt = fused(b, "attn_sparse_sink_rope_roundtrip", fp, { L.sinks }, { m.q, m.sel, cs }, { m.o });
            else
                L.attn_rt = fused(b, "attn_sparse_sink_rope_roundtrip", fp, { L.sinks }, { m.q, cs }, { m.o });
            if (L.attn_rt) g_sw_of_ok.store(1, std::memory_order_relaxed);
        }
        /* This rank's output groups, each a block of the block-diagonal wo_a. */
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) {
            const int gi_ = (int)(m.ar * m.gr + j);
            const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
            DS_TRY(map_rows(b, n, { { ckpt_name(nm.f("blk.%d.attn_output_a.weight", l)), (int)(gi_ * c.o_rank),
                                            c.o_rank } }));
            L.wo_a.push_back(lw(b, m, n, c.o_rank, gd, grp, RAD_SHARD_NONE, 0, 0, true));
            if (!L.wo_a.back()) return RAD_E_INVAL;
            L.o_a.push_back(linear(b, m, T, L.wo_a.back(), c.o_rank, gd, m.o, m.oa));
        }
        const int64_t ok = m.gr * c.o_rank;
        L.wo_b = lw(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL,
                    m.ar * ok, (m.ar + 1) * ok);
        L.o_b = linear(b, m, T, L.wo_b, D, ok, m.oa, m.attn_out);
    }
    /* The heads' partial outputs (zeros on an attention-zero rank) summed over every rank. */
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                         RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { m.attn_out }, { m.attn_out });
    L.post_a = op(b, "mhc_post", hcp, {}, { m.attn_out, m.h0, m.post, m.comb }, { m.h1 });

    /* ---- the ffn connection and the MoE */
    L.hc_f_fn = w("blk.%d.hc_ffn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_f_scale = w("blk.%d.hc_ffn_scale.weight", RAD_F32, { 3 });
    L.hc_f_base = w("blk.%d.hc_ffn_base.weight", RAD_F32, { mix });
    L.mix_f = op(b, "mhc_mix", mixp, { L.hc_f_fn, L.hc_f_scale, L.hc_f_base }, { m.h1 },
                 { m.pre_f, m.post, m.comb });
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    L.pcn_f = post_collapse_norm(b, m, T, L.ffn_norm, m.attn_out, m.h0, m.h1, m.pre_a, m.xcol,
                                 m.act.x);
    if (!L.pcn_f) L.cn_f = collapse_norm(b, m, T, L.ffn_norm, m.h1, m.pre_a, m.xcol, m.act.x);
    if (!L.pcn_f && !L.cn_f) {
        L.col_f = op(b, "mhc_collapse", hcp, {}, { m.h1, m.pre_a }, { m.xcol });
        L.norm_f = rmsnorm(b, m, T, L.ffn_norm, D, m.xcol, m.act.x);
    }
    /* The all-reduce + pcn_f as one collective where both exist (ds41_ar_pcn); the pair stays declared for the
     * steps it does not take. */
    if (L.ar_attn && L.pcn_f && ds41_ar_pcn() && g.world == 4) {
        const int64_t rr = std::min<int64_t>(T, ar_wide_write_rows(g));
        if (rr >= 1)
            L.ar_pcn_f = fused(b, "ar_mhc_post_collapse_rmsnorm",
                               { RAD_RANGE("M", 1, rr), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_F64("eps", c.eps),
                                 RAD_INT("world_size", g.world), RAD_STR("dtype", "bf16"), RAD_INT("wire", 0) },
                               { L.ffn_norm }, { m.attn_out, m.h0, m.post, m.comb, m.pre_a },
                               { m.attn_out, m.h1, m.xcol, m.act.x });
    }
    MoeFP8::Src s;
    s.norm = nullptr;
    moe_names(m, l, &s);
    s.cfg.n_expert_all = c.n_expert;
    s.cfg.n_expert = c.n_expert / g.world;
    s.cfg.expert_base = g.rank * s.cfg.n_expert;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = c.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.shared_lo = m.sh_lo;
    s.cfg.shared_hi = m.sh_hi;
    s.cfg.rows = T;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.router_bf16 = ds41_router_bf16();
    s.cfg.router_fuse = ds41_router_fuse();
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    s.cfg.gather_ar_rows = gather_ar_rows();
    s.cfg.calib = (m.calib_hi > m.calib_lo && l >= m.calib_lo && l < m.calib_hi) ? 1 : 0;
    /* Lever K1 fold f: the step's ones plane (Model::sone) where declared; the layer's own fill stays declared
     * and serves while the ds41_onefill switch is 0 (sone_live). */
    s.cfg.shared_one_external = m.sone && g.n_ff > 0 ? 1 : 0;
    /* Lever K1 fold e: the shared arm's clamped SwiGLU + GGUF down as one op beside the pair (RADIANCE_DS41_SWFOLD). */
    s.cfg.shared_swfold = g_sw_swf_env.load(std::memory_order_relaxed) > 0 && g.n_ff > 0 ? 1 : 0;
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    DS_TRY(L.moe.declare(b, nm, g, l, m.mw, s, false, false));
    /* the in-process switches (rad_fn4_*): the fused gather's live bound, the shared arm's lane */
    L.moe.gar_live = &g_sw_gar;
    L.moe.shexp_lane = &g_sw_shexp;
    if (s.cfg.shared_one_external) {
        L.moe.sone_live = &g_sw_one;
        if (L.moe.gfold) g_sw_one_ok.store(1, std::memory_order_relaxed);
    }
    L.moe.swfold_live = &g_sw_swf;
    if (L.moe.op_swfold) g_sw_swf_ok.store(1, std::memory_order_relaxed);
    sw_note_decl(L.moe.gar_rows);
    if (L.moe.shexp_lane_ok) g_sw_shexp_ok.store(1, std::memory_order_relaxed);
    L.post_f = op(b, "mhc_post", hcp, {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
    return RAD_OK;
}

/* ------------------------------------------------------------------ the DSpark drafter (Dsp) */
/* Whether this checkpoint and deployment get the drafter (m.dsp.on); an error only for a serve the
 * drafter cannot run as asked. The checkpoint states it by `nextn_predict_layers` (the stages ship as
 * blk.{n_layer}.. ) beside the 1017's `dspark.*` keys; the published UD GGUFs carry the keys but no
 * stages, and get no drafter. The official checkpoint's own stages (mtp.*, fp8 / MXFP4) reach a GGUF
 * trunk as one more split shard in these names, re-spelled value for value (BF16 linears, Q8_0 experts)
 * by a graft outside the tree, so they arrive here as any GGUF weight does. */
static int dsp_config(RadBuilder* b, Model& m, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    auto& d = m.dsp;
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t ns = gi(meta, "nextn_predict_layers", 0), blk = gi(meta, "dspark.block_size", 0);
    if (blk <= 0) return RAD_OK;
    if (ns <= 0) {
        rad_note(b, "the checkpoint names a DSpark drafter (dspark.block_size %lld) but carries none of its "
                    "stages (no nextn_predict_layers): no drafter", (long long)blk);
        return RAD_OK;
    }
    if (!rad_ctx_draft_head(ctx)) return RAD_OK;   /* no speculation, or prompt lookup alone */
    if (ctx->max_spec != blk) {
        fprintf(stderr, "radiance: deepseek41: the DSpark drafter proposes a whole block of %lld tokens "
                        "(dspark.block_size, the length it was trained at) and this serve asks for %d; run "
                        "--num-speculative-tokens %lld (or auto)\n", (long long)blk, ctx->max_spec, (long long)blk);
        return RAD_E_UNSUPPORTED;
    }
    d.n_expert = gi(meta, "dspark.expert_count", c.n_expert);
    d.top_k = gi(meta, "dspark.expert_used_count", c.top_k);
    d.rank = gi(meta, "dspark.markov_rank", 0);
    d.noise = gi(meta, "dspark.noise_token_id", -1);
    /* The walk's candidates from the head's top: 32, the most the walk takes (the container states 16, its converter's
     * default). Measured 10.11.2026 on EXL3 2.25, TP4, NST 5 (same image, 16 vs 32): tokens a step 1-user code 4.00 ->
     * 4.52, c4 9.51 -> 9.76, c8 19.66 -> 20.09 at the same step time. */
    d.cand = 32;
    /* RADIANCE_DS41_DSPARK_CAND (1..32) overrides it: the Markov walk's candidates from the head's top (the reference
     * biases the whole vocabulary; the walk re-ranks this many). The drafter only proposes and the trunk verifies, so
     * the trunk's greedy text does not depend on it. */
    if (const char* e = std::getenv("RADIANCE_DS41_DSPARK_CAND")) {
        const long v = std::strtol(e, nullptr, 10);
        if (v >= 1 && v <= 32) d.cand = v;
    }
    /* Lever C1: the confidence head from the sidecar (the containers carry no confidence_head). A
     * named file that does not read as D + rank f32 values (and an optional bias) is refused: the
     * serve asked for the head by name. */
    d.cw.clear();
    d.cbias = 0.0;
    d.conf_path.clear();
    if (const char* e = std::getenv("RADIANCE_DS41_DSPARK_CONF"); e && *e && g.rank == 0) {
        const int64_t want = c.n_embd + gi(meta, "dspark.markov_rank", 0);
        FILE* f = std::fopen(e, "rb");
        std::vector<float> v;
        if (f) {
            float x = 0.0f;
            while (std::fread(&x, 4, 1, f) == 1 && (int64_t)v.size() <= want) v.push_back(x);
            std::fclose(f);
        }
        if (want <= c.n_embd || ((int64_t)v.size() != want && (int64_t)v.size() != want + 1)) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_CONF=%s: %s; the DSpark confidence head is "
                            "%lld f32 values (n_embd %lld + markov rank %lld) and an optional bias\n", e,
                    f ? (std::to_string(v.size()) + " values").c_str() : "cannot be read", (long long)want, (long long)c.n_embd,
                    (long long)(want - c.n_embd));
            return RAD_E_INVAL;
        }
        if ((int64_t)v.size() == want + 1) {
            d.cbias = v.back();
            v.pop_back();
        }
        d.cw = std::move(v);
        d.conf_path = e;
    }
    /* Lever T1: the shadow tree's branches. */
    d.n_alt = 0;
    if (const char* e = std::getenv("RADIANCE_DS41_DSPARK_ALT"); e && *e && g.rank == 0) {
        const long v = std::strtol(e, nullptr, 10);
        if (v < 0 || v > 4) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_ALT=%s: the shadow tree takes 1..4 "
                            "branches (0 or unset: none)\n", e);
            return RAD_E_INVAL;
        }
        d.n_alt = v;
    }
    d.taps = gl(meta, "dspark.target_layers");
    std::sort(d.taps.begin(), d.taps.end());
    bool ok = !d.taps.empty() && d.noise >= 0 && d.noise < c.n_vocab && d.rank > 0 && d.rank % 32 == 0 &&
              d.rank <= 512 && d.cand >= 1 && d.cand <= 32 && d.n_expert > 0 && d.top_k > 0 &&
              d.top_k <= d.n_expert;
    for (int64_t t : d.taps) ok = ok && t >= 0 && t < c.n_layer;
    if (!ok) {
        fprintf(stderr, "radiance: deepseek41: the dspark.* configuration does not describe a drafter this "
                        "plugin runs (target layers %zu, noise id %lld, markov rank %lld (a multiple of 32 up "
                        "to 512), candidates %lld (1..32), experts %lld top-%lld)\n",
                d.taps.size(), (long long)d.noise, (long long)d.rank, (long long)d.cand,
                (long long)d.n_expert, (long long)d.top_k);
        return RAD_E_UNSUPPORTED;
    }
    /* Experts are expert-parallel as in the trunk, an even count of whole experts a rank (the grouped GEMM
     * takes two parity tables), but not necessarily the same count: the 1017's 128 over three ranks are
     * 44 / 42 / 42 (the even floor of the share each, then two more to the first ranks until all are
     * placed). The router is replicated, so routing needs no agreement on the split. */
    {
        const int64_t per = (d.n_expert / g.world) & ~(int64_t)1;
        int64_t rem = d.n_expert - per * g.world, base = 0;
        for (int r = 0; r <= g.rank; ++r) {
            const int64_t n = per + (rem >= 2 ? 2 : 0);
            rem -= rem >= 2 ? 2 : 0;
            if (r == g.rank) {
                d.ne_rank = n;
                d.ne_base = base;
            }
            base += n;
        }
    }
    if (d.n_expert % 2 || d.ne_rank < 2) {
        fprintf(stderr, "radiance: deepseek41: the DSpark drafter's %lld experts do not split over %d ranks "
                        "(an even count of whole experts each); serving without it\n",
                (long long)d.n_expert, g.world);
        rad_note(b, "DSpark drafter NOT declared: %lld experts over %d ranks", (long long)d.n_expert, g.world);
        d.taps.clear();
        return RAD_OK;
    }
    if (d.ne_rank * g.world != d.n_expert)
        rad_note(b, "DSpark drafter experts split unevenly: rank %d holds %lld of %lld from %lld", g.rank,
                 (long long)d.ne_rank, (long long)d.n_expert, (long long)d.ne_base);
    d.block = blk;
    d.base = (int)c.n_layer;
    d.n_stage = (int)ns;
    d.Q = std::min<int64_t>(g.max_seqs * blk, g.max_tok);
    d.on = d.Q >= blk;
    return RAD_OK;
}

/* One stage's weights (blk.{base + s}.*, the trunk's suffixes) and its context-pass key ops. */
static int dsp_stage_weights(RadBuilder* b, Model& m, Layer& L, int s) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    const int l = m.dsp.base + s;
    const int64_t D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim, mix = c.hc * (c.hc + 2);
    const auto grp = grp_layer(l);
    L.l = l;
    L.r = 0;
    auto w = [&](const char* fmt, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wt(b, nm.f(fmt, l), dt, shape, grp);
    };
    L.hc_a_fn = w("blk.%d.hc_attn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_a_scale = w("blk.%d.hc_attn_scale.weight", RAD_F32, { 3 });
    L.hc_a_base = w("blk.%d.hc_attn_base.weight", RAD_F32, { mix });
    L.hc_f_fn = w("blk.%d.hc_ffn_fn.weight", RAD_F32, { mix, c.hc * D });
    L.hc_f_scale = w("blk.%d.hc_ffn_scale.weight", RAD_F32, { 3 });
    L.hc_f_base = w("blk.%d.hc_ffn_base.weight", RAD_F32, { mix });
    L.ffn_norm = w("blk.%d.ffn_norm.weight", RAD_F32, { D });
    if (!L.hc_a_fn || !L.hc_a_scale || !L.hc_a_base || !L.hc_f_fn || !L.hc_f_scale || !L.hc_f_base || !L.ffn_norm)
        return RAD_E_INVAL;
    if (m.zero) return RAD_OK;
    L.attn_norm = w("blk.%d.attn_norm.weight", RAD_F32, { D });
    L.wq_a = lw(b, m, nm.f("blk.%d.attn_q_a.weight", l), c.q_lora, D, grp);
    L.q_norm = w("blk.%d.attn_q_a_norm.weight", RAD_F32, { c.q_lora });
    L.wq_b = lw(b, m, nm.f("blk.%d.attn_q_b.weight", l), HD, c.q_lora, grp, RAD_SHARD_ROW, m.ar * HD,
                (m.ar + 1) * HD);
    L.wkv = lw(b, m, nm.f("blk.%d.attn_kv.weight", l), hd, D, grp);
    L.kv_norm = w("blk.%d.attn_kv_a_norm.weight", RAD_F32, { hd });
    L.sinks = wts(b, g, nm.f("blk.%d.attn_sinks.weight", l), RAD_F32, { g.n_head }, grp, RAD_SHARD_ROW,
                  m.ar * g.n_head, (m.ar + 1) * g.n_head);
    const int64_t gd = HD / m.gr;
    for (int j = 0; j < m.gr; ++j) {
        const int gi_ = (int)(m.ar * m.gr + j);
        const char* n = nm.f("blk.%d.attn_output_a.g%d.weight", l, gi_);
        DS_TRY(map_rows(b, n, { { ckpt_name(nm.f("blk.%d.attn_output_a.weight", l)), (int)(gi_ * c.o_rank), c.o_rank } }));
        L.wo_a.push_back(lw(b, m, n, c.o_rank, gd, grp, RAD_SHARD_NONE, 0, 0, true));
        if (!L.wo_a.back()) return RAD_E_INVAL;
    }
    const int64_t ok = m.gr * c.o_rank;
    L.wo_b = lw(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL, m.ar * ok,
                (m.ar + 1) * ok);
    return L.attn_norm && L.wq_a && L.q_norm && L.wq_b && L.wkv && L.kv_norm && L.sinks && L.wo_b ? RAD_OK
                                                                                                  : RAD_E_INVAL;
}

/* One stage's query-pass block: Layer::layer()'s window-only path over the drafter's rows, its
 * attention over the drafter's cache. */
static int dsp_stage_ops(RadBuilder* b, Model& m, Layer& L) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    auto& d = m.dsp;
    const int l = L.l;
    const int64_t Q = d.Q, D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim;
    const std::vector<RadParam> mixp = { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                         RAD_INT("iterations", c.hc_iters), RAD_F64("eps", c.eps),
                                         RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") };
    const std::vector<RadParam> hcp = { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc),
                                        RAD_STR("dtype", "bf16") };
    L.mix_a = op(b, "mhc_mix", mixp, { L.hc_a_fn, L.hc_a_scale, L.hc_a_base }, { d.h0 }, { d.pre_a, d.post, d.comb });
    L.col_a = op(b, "mhc_collapse", hcp, {}, { d.h0, d.pre_f }, { d.x });
    if (m.zero) {
        L.attn_fill = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { d.attn_out });
    } else {
        L.norm_a = rmsnorm(b, m, Q, L.attn_norm, D, d.x, d.xn);
        L.q_a = linear(b, m, Q, L.wq_a, c.q_lora, D, d.xn, d.qr);
        L.q_an = rmsnorm(b, m, Q, L.q_norm, c.q_lora, d.qr, d.qrn);
        L.q_b = linear(b, m, Q, L.wq_b, HD, c.q_lora, d.qrn, d.q);
        L.q_rope = rope_rows(b, m, Q, g.n_head, hd, 0, d.q, m.cs_a);
        L.kv_g = linear(b, m, Q, L.wkv, hd, D, d.xn, d.kv_);
        L.kv_n = rmsnorm(b, m, Q, L.kv_norm, hd, d.kv_, d.kvn);
        L.kv_rope = rope_rows(b, m, Q, 1, hd, 0, d.kvn, m.cs_a);
        L.kv_q = roundtrip(b, Q, hd, 32, "fp8_e4m3", "ue8m0", d.kvn);
        L.kv_st = m.w8d ? op(b, "window_store",
                             { RAD_RANGE("M", 1, Q), RAD_INT("n", hd), RAD_INT("block_size", d.kv_bs),
                               RAD_STR("format", "fp8_e4m3"), RAD_STR("dtype", "bf16") },
                             {}, { d.kvn }, {})
                        : op(b, "kv_store",
                             { RAD_RANGE("M", 1, Q), RAD_INT("head_dim", hd / 2), RAD_INT("n_head_kv", 1),
                               RAD_INT("block_size", d.kv_bs), RAD_STR("kv_dtype", "bf16") },
                             {}, { d.kvn }, {});
        /* The window of `c.window` context keys before the anchor plus the block's own `block` keys,
         * for every row: each row is placed at its block's last position (apos). */
        std::vector<RadParam> dap = { RAD_RANGE("M", 1, Q), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                                      RAD_INT("window", c.window + d.block), RAD_INT("ratio", 0), RAD_INT("topk", 0),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") };
        if (m.w8d) dap.push_back(RAD_STR("window_format", "fp8_e4m3"));
        L.attn = op(b, "attn_sparse_sink", dap, { L.sinks }, { d.q, d.apos }, { d.o });
        L.o_rope = rope_rows(b, m, Q, g.n_head, hd, 1, d.o, m.cs_a);
        L.o_q = roundtrip(b, Q, HD, 32, "fp8_e4m3", "ue8m0", d.o);
        const int64_t gd = HD / m.gr;
        for (int j = 0; j < m.gr; ++j) L.o_a.push_back(linear(b, m, Q, L.wo_a[(size_t)j], c.o_rank, gd, d.o, d.oa));
        L.o_b = linear(b, m, Q, L.wo_b, D, m.gr * c.o_rank, d.oa, d.attn_out);
        if (!L.norm_a || !L.q_a || !L.q_an || !L.q_b || !L.q_rope || !L.kv_g || !L.kv_n || !L.kv_rope ||
            !L.kv_st || !L.attn || !L.o_rope || !L.o_b)
            return RAD_E_INVAL;
    }
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, Q * D), RAD_STR("dtype", "bf16"),
                         RAD_INT("exact", g.wire_exact), RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { d.attn_out }, { d.attn_out });
    L.post_a = op(b, "mhc_post", hcp, {}, { d.attn_out, d.h0, d.post, d.comb }, { d.h1 });
    L.mix_f = op(b, "mhc_mix", mixp, { L.hc_f_fn, L.hc_f_scale, L.hc_f_base }, { d.h1 }, { d.pre_f, d.post, d.comb });
    L.col_f = op(b, "mhc_collapse", hcp, {}, { d.h1, d.pre_a }, { d.x });
    L.norm_f = rmsnorm(b, m, Q, L.ffn_norm, D, d.x, d.act.x);
    MoeFP8::Src s;
    s.norm = nullptr;
    moe_names(m, l, &s);
    s.cfg.n_expert_all = d.n_expert;
    s.cfg.n_expert = d.ne_rank;
    s.cfg.expert_base = d.ne_base;
    s.cfg.probed_expert = s.cfg.expert_base;
    s.cfg.top_k = d.top_k;
    s.cfg.n_ff_exp = c.n_ff;
    s.cfg.n_ff_shared = g.n_ff;
    s.cfg.shared_lo = m.sh_lo;
    s.cfg.shared_hi = m.sh_hi;
    s.cfg.rows = Q;
    s.cfg.norm_topk = c.norm_topk;
    s.cfg.sigmoid_router = 1;
    s.cfg.router_score = "sqrtsoftplus";
    s.cfg.router_bf16 = ds41_router_bf16();
    s.cfg.router_fuse = ds41_router_fuse();
    s.cfg.route_scale = c.route_scale;
    s.cfg.swiglu_limit = c.swiglu_limit;
    s.cfg.shared_ungated = 1;
    s.cfg.gather_ar_rows = gather_ar_rows();
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    /* Lever Z0-A (ds41-z0): the block's sigmoid router refuses rows above its geometry's max_tok
     * (rad_block_glm_router.h, GlmRouter::declare: c.rows > g.max_tok -> RAD_E_INVAL). A sizing declare below Q (the
     * 8-row rung at NST 5) keeps the head's real Q rows (the shape_probe block in declare()), so the drafter's block
     * takes max_tok = max(T, Q). The real declare has Q <= T (dsp_config: Q = min(max_seqs x block, max_tok)), so its
     * geometry, graph and parameters are today's. */
    Geom gq = g;
    gq.max_tok = std::max<int64_t>(g.max_tok, Q);
    DS_TRY(L.moe.declare(b, nm, gq, l, d.mw, s, false, false));
    L.moe.gar_live = &g_sw_gar;   /* the same live bound as the trunk's (rad_fn4_*); one lane always */
    sw_note_decl(L.moe.gar_rows);
    L.post_f = op(b, "mhc_post", hcp, {}, { d.act.x, d.h1, d.post, d.comb }, { d.h0 });
    return L.mix_a && L.col_a && L.post_a && L.mix_f && L.col_f && L.norm_f && L.post_f ? RAD_OK : RAD_E_INVAL;
}

/* Everything of the drafter past the trunk's head (declaration order is the buffer planner's
 * liveness: the drafter's passes run after the whole trunk step, as arch/llama_fp8 declares its own). */
static int dsp_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    auto& g = m.g;
    auto& nm = m.nm;
    const auto& c = m.c;
    auto& d = m.dsp;
    const int64_t T = g.max_tok, Q = d.Q, D = c.n_embd, HD = g.n_head * c.head_dim, hd = c.head_dim;
    const int64_t R = d.cand;
    auto bf = [&](const char* n, std::initializer_list<int64_t> s) { return decl_b(b, nm.f("%s", n), RAD_BF16, s); };

    /* ---- buffers: the context pass (a trunk step's rows), the query pass (Q rows), the head */
    d.c = bf("dspark_main", { T, D });
    d.cn = bf("dspark_main_normed", { T, D });
    d.ckv = bf("dspark_ctx_kv", { T, hd });
    d.ckvn = bf("dspark_ctx_kv_normed", { T, hd });
    d.x = bf("dspark_collapsed", { Q, D });
    d.xn = bf("dspark_normed", { Q, D });
    d.h0 = bf("dspark_streams", { Q, c.hc * D });
    d.h1 = bf("dspark_streams_next", { Q, c.hc * D });
    d.pre_a = decl_b(b, nm.f("dspark_pre_attn"), RAD_F32, { Q, c.hc });
    d.pre_f = decl_b(b, nm.f("dspark_pre_ffn"), RAD_F32, { Q, c.hc });
    d.post = decl_b(b, nm.f("dspark_post"), RAD_F32, { Q, c.hc });
    d.comb = decl_b(b, nm.f("dspark_comb"), RAD_F32, { Q, c.hc * c.hc });
    d.qr = bf("dspark_q_lora", { Q, c.q_lora });
    d.qrn = bf("dspark_q_lora_normed", { Q, c.q_lora });
    d.q = bf("dspark_q", { Q, HD });
    d.kv_ = bf("dspark_kv", { Q, hd });
    d.kvn = bf("dspark_kv_normed", { Q, hd });
    d.o = bf("dspark_attn_o", { Q, HD });
    d.oa = bf("dspark_attn_o_lora", { Q, m.gr * c.o_rank });
    d.attn_out = bf("dspark_attn_out", { Q, D });
    d.poff = decl_b(b, nm.f("dspark_pos_offset"), RAD_I32, { Q, 1 });
    d.apos = decl_b(b, nm.f("dspark_attn_pos"), RAD_I32, { Q, 1 });
    d.act.x = bf("dspark_ffn_normed", { Q, D });
    auto& mw = d.mw;
    mw.x = d.x;
    mw.h = d.act;
    mw.logits = decl_b(b, nm.f("dspark_router_logits"), RAD_F32, { Q, d.n_expert });
    mw.ids = decl_b(b, nm.f("dspark_expert_ids"), RAD_I32, { Q, d.top_k });
    mw.ew = decl_b(b, nm.f("dspark_expert_weights"), RAD_F32, { Q, d.top_k });
    mw.eoff = decl_b(b, nm.f("dspark_expert_offsets"), RAD_I32, { d.n_expert + 1 });
    mw.ecnt = decl_b(b, nm.f("dspark_expert_counts"), RAD_I32, { d.n_expert }, RAD_BUF_PERSIST);
    mw.sorted = decl_b(b, nm.f("dspark_expert_sorted"), RAD_I32, { Q * d.top_k });
    mw.egu = bf("dspark_expert_gate_up", { Q * d.top_k, 2 * c.n_ff });
    mw.eff.x = bf("dspark_expert_ff", { Q * d.top_k, c.n_ff });
    mw.edn = bf("dspark_expert_down", { Q * d.top_k, D });
    if (g.n_ff > 0) {
        mw.sgu = bf("dspark_shared_gate_up", { Q, 2 * g.n_ff });
        mw.sff.x = bf("dspark_shared_ff", { Q, g.n_ff });
    }
    mw.sout = bf("dspark_shared_out", { Q, D });
    mw.sgate = bf("dspark_shared_one", { Q, 1 });
    d.fin = bf("dspark_final", { Q, D });
    d.rows = bf("dspark_head_in", { Q, D });
    d.logits = decl_b(b, nm.f("dspark_logits"), RAD_F32, { Q, g.n_vocab });
    d.cand_ = decl_b(b, nm.f("dspark_cand"), RAD_I32, { Q, R });
    d.unary = decl_b(b, nm.f("dspark_unary"), RAD_F32, { Q, R });
    if (g.world > 1) {
        d.pairs = decl_b(b, nm.f("dspark_pairs"), RAD_F32, { Q, R, 2 });
        d.gath = decl_b(b, nm.f("dspark_pairs_all"), RAD_F32, { Q, (int64_t)g.world * R, 2 });
    }
    /* The proposal buffer (RadDrafterDecl::proposal), `block` tokens a sequence. */
    d.tokens = decl_b(b, nm.f("dspark_tokens"), RAD_I32, { g.max_seqs, d.block });
    if (!d.c || !d.cn || !d.ckv || !d.ckvn || !d.x || !d.h0 || !d.logits || !d.tokens) return RAD_E_INVAL;

    /* ---- weights */
    d.st.resize((size_t)d.n_stage);
    for (int s = 0; s < d.n_stage; ++s) DS_TRY(dsp_stage_weights(b, m, d.st[(size_t)s], s));
    const int64_t nt = (int64_t)d.taps.size();
    /* The context pass runs on the attention ranks only (an attention-zero rank keeps no keys). */
    if (!m.zero) {
        d.fc = lw(b, m, nm.f("dspark.fc.weight"), D, nt * D, grp_model());
        d.main_norm = wt(b, nm.f("dspark.enc.output_norm.weight"), RAD_F32, { D }, grp_model());
        if (!d.fc || !d.main_norm) return RAD_E_INVAL;
    }
    d.norm = wt(b, nm.f("dspark.output_norm.weight"), RAD_F32, { D }, grp_model());
    /* The Markov head's tables, [vocab, rank] each and replicated: the walk gathers rows by global id. */
    d.mk1 = wt(b, nm.f("dspark.markov_w1.weight"), RAD_BF16, { c.n_vocab, d.rank }, grp_model(), RAD_ACCESS_VOCAB);
    d.mk2 = wt(b, nm.f("dspark.markov_w2.weight"), RAD_BF16, { c.n_vocab, d.rank }, grp_model(), RAD_ACCESS_VOCAB);
    if (!d.norm || !d.mk1 || !d.mk2) return RAD_E_INVAL;

    /* ---- the context pass, and the drafter's window cache: the context keys of every committed
     * position plus, past them, the block keys a query pass writes (the next context pass overwrites
     * them). Declared right before the first store, whose block_size fixes the group's. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_WINDOW;
    kd.dtype = RAD_BF16;
    kd.window = c.window + d.block + g.max_spec;
    kd.n_head_kv = 1;
    kd.head_dim = window_slots(hd, m.w8d) / 2;
    if (m.zero) {
        if (!(d.kv = rad_decl_kv_group(b, nm.f("kv_dspark"), &kd))) return RAD_E_INVAL;
        d.sizing = window_key_store(b, T, hd, m.w8d, d.ckvn);   /* fixes the block size; never issued */
    } else {
        d.c_tab = op(b, "rope_table",
                     { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta),
                       RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") },
                     {}, {}, { m.cs_a });
        d.c_fc = linear(b, m, T, d.fc, D, nt * D, d.src, d.c);
        d.c_norm = rmsnorm(b, m, T, d.main_norm, D, d.c, d.cn);
        for (auto& L : d.st) {
            L.d_kv = linear(b, m, T, L.wkv, hd, D, d.cn, d.ckv);
            L.d_kvn = rmsnorm(b, m, T, L.kv_norm, hd, d.ckv, d.ckvn);
            L.d_kvrope = rope_rows(b, m, T, 1, hd, 0, d.ckvn, m.cs_a);
            L.d_kvq = roundtrip(b, T, hd, 32, "fp8_e4m3", "ue8m0", d.ckvn);
            if (!d.kv && !(d.kv = rad_decl_kv_group(b, nm.f("kv_dspark"), &kd))) return RAD_E_INVAL;
            L.d_kvst = window_key_store(b, T, hd, m.w8d, d.ckvn);
            if (!L.d_kv || !L.d_kvn || !L.d_kvrope || !L.d_kvst) return RAD_E_INVAL;
        }
        if (!d.c_tab || !d.c_fc || !d.c_norm) return RAD_E_INVAL;
        /* lever PSK-DS (dspark_pfctx): the context pass's ops are declared; the line only when its env is set */
        if (!ctx->shape_probe && c.window > 0) {
            g_pfctx_window.store((int)c.window, std::memory_order_relaxed);
            g_sw_pfctx_ok.store(1, std::memory_order_relaxed);
            if (ctx->rank == 0 && g_sw_pfctx_env.load(std::memory_order_relaxed) >= 0)
                fprintf(stderr, "radiance: deepseek41: lever PSK-DS (RADIANCE_DS41_DSPARK_PFCTX, fn4 dspark_pfctx, now %d): "
                                "a DSpark context pass runs each entry's last %lld rows only (the drafter's window + 1), "
                                "in runs of >= %lld rows\n",
                        g_sw_pfctx.load(std::memory_order_relaxed), (long long)c.window + 1, (long long)kPskBase);
        }
    }
    d.kv_bs = rad_kv_block_size(b, d.kv);
    if (d.kv_bs < 1) return RAD_E_INVAL;
    for (const auto& L : d.st) DS_TRY(rad_bind_layer_kv(b, L.l, d.kv));
    if (m.zero) DS_TRY(rad_kv_group_zero(b, d.kv));

    /* ---- the query pass: [anchor, noise ...] embedded by the trunk's table into hc copies */
    const auto tt = m.wty.find(m.token);
    const auto ts2 = m.wsc.find(m.token);
    const auto tc2 = m.wcod.find(m.token);
    if (ts2 != m.wsc.end() && tc2 != m.wcod.end() && tc2->second != RAD_F8E4M3)
        d.embed = op(b, "embed_lookup_iqr",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token, ts2->second }, {}, { d.x });
    else if (ts2 != m.wsc.end() && tc2 != m.wcod.end() && tc2->second == RAD_F8E4M3)
        d.embed = op(b, "embed_lookup_q",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off),
                       RAD_F64("wscale", 1.0) },
                     { m.token, ts2->second }, {}, { d.x });
    else if (tt != m.wty.end())
        d.embed = op(b, "embed_lookup_ggml",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("type", tt->second), RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { d.x });
    else
        d.embed = op(b, "embed_lookup",
                     { RAD_RANGE("M", 1, Q), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { d.x });
    if (g.world > 1)
        d.embed_ar = op(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, Q * D), RAD_STR("dtype", "bf16"),
                          RAD_INT("exact", 1) },
                        {}, { d.x }, { d.x });
    d.expand = op(b, "cast", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("from", "bf16"), RAD_STR("to", "bf16") },
                  {}, { d.x }, { d.h0 });
    d.pre0 = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", c.hc), RAD_STR("dtype", "f32"), RAD_F64("value", 0.0) },
                {}, {}, { d.pre_f });
    d.pre1 = op(b, "fill", { RAD_RANGE("M", 1, Q), RAD_INT("n", 1), RAD_STR("dtype", "f32"), RAD_F64("value", 1.0) },
                {}, {}, { d.pre_f });
    d.q_tab = op(b, "rope_table",
                 { RAD_RANGE("M", 1, Q), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta), RAD_F64("scale", 1.0),
                   RAD_STR("dtype", "f32") },
                 {}, {}, { m.cs_a });
    /* apos = position + (block - 1 - row % block): each row at its block's last position. */
    d.poff_op = op(b, "iota", { RAD_RANGE("M", 1, Q), RAD_INT("start", d.block - 1), RAD_INT("step", -1),
                                RAD_INT("period", d.block) },
                   {}, {}, { d.poff });
    d.apos_op = op(b, "add", { RAD_RANGE("M", 1, Q), RAD_INT("n", 1), RAD_STR("dtype", "i32") }, {}, { d.poff },
                   { d.apos });
    if (!d.embed || !d.expand || !d.pre0 || !d.pre1 || !d.q_tab || !d.poff_op || !d.apos_op) return RAD_E_INVAL;
    for (auto& L : d.st) DS_TRY(dsp_stage_ops(b, m, L));

    /* ---- the head: the last stage's ffn gates collapse the streams, its norm, the trunk's lm_head */
    d.contract = op(b, "mhc_collapse", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                    {}, { d.h0, d.pre_f }, { d.x });
    d.fnorm = rmsnorm(b, m, Q, d.norm, D, d.x, d.fin);
    d.gather = op(b, "gather_rows", { RAD_RANGE("M", 1, Q), RAD_INT("n", D), RAD_STR("dtype", "bf16") }, {},
                  { d.fin }, { d.rows });
    const auto ht = m.wty.find(m.head);
    const auto hs2 = m.wsc.find(m.head);
    const auto hc2 = m.wcod.find(m.head);
    if (hs2 != m.wsc.end() && hc2 != m.wcod.end() && hc2->second == RAD_I8)
        d.head = op(b, "logits_gemm_iqr",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D),
                      RAD_STR("dtype", "bf16") },
                    { m.head, hs2->second }, { d.rows }, { d.logits });
    else if (hs2 != m.wsc.end() && hc2 != m.wcod.end())
        d.head = op(b, "logits_gemm",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D),
                      RAD_STR("dtype", "bf16") },
                    { m.head }, { d.rows }, { d.logits });
    else if (ht != m.wty.end())
        d.head = op(b, "logits_gemm_ggml",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D),
                      RAD_STR("type", ht->second), RAD_STR("dtype", "bf16") },
                    { m.head }, { d.rows }, { d.logits });
    else
        d.head = op(b, "logits_gemm",
                    { RAD_RANGE("M", 1, Q), RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                    { m.head }, { d.rows }, { d.logits });
    /* The candidates: this rank's top-R of its vocabulary shard, merged to the global top-R across
     * ranks (ties to the lower id, as an unsharded top-R). */
    if (g.world > 1) {
        d.topk = op(b, "row_topk", { RAD_RANGE("M", 1, Q), RAD_INT("N", g.n_vocab), RAD_INT("R", R),
                                     RAD_INT("vocab_off", g.vocab_off), RAD_STR("dtype", "f32") },
                    {}, { d.logits }, { d.cand_, d.unary, d.pairs });
        d.pgather = op(b, "all_gather", { RAD_INT("world_size", g.world), RAD_INT("numel", Q * R * 2),
                                          RAD_STR("dtype", "f32"), RAD_INT("row", R * 2) },
                       {}, { d.pairs }, { d.gath });
        d.merge = op(b, "row_topk_merge", { RAD_RANGE("M", 1, Q), RAD_INT("R", R), RAD_INT("world_size", g.world),
                                            RAD_STR("dtype", "f32") },
                     {}, { d.gath }, { d.cand_, d.unary });
        if (!d.pgather || !d.merge) return RAD_E_INVAL;
    } else {
        d.topk = op(b, "row_topk", { RAD_RANGE("M", 1, Q), RAD_INT("N", g.n_vocab), RAD_INT("R", R), RAD_STR("dtype", "f32") },
                    {}, { d.logits }, { d.cand_, d.unary });
    }
    /* The Markov walk: logits[i] + W2 . W1[prev(i)], prev(0) the anchor, greedily (dflash_select with no
     * hidden projection is exactly that bigram over the candidates). */
    d.select = op(b, "dflash_select",
                  { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", d.block), RAD_INT("top_k", R),
                    RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", d.block),
                    RAD_STR("dtype", "bf16") },
                  { d.mk1, d.mk2 }, { d.cand_, d.unary }, { d.tokens });
    if (!d.contract || !d.fnorm || !d.gather || !d.head || !d.topk || !d.select) return RAD_E_INVAL;
    /* Lever W1: the full-vocabulary walk beside it, when RADIANCE_DS41_DSPARK_WALK asked at the first declare. Every
     * rank scores its own vocabulary shard (the head's logits are vocab-sharded) and the ranks agree on each step's
     * token through one all_gather of a u64 key (in a 16-byte slot) a sequence (dspark_walk_step: the score's ordered bits and the id's
     * complement, ties to the lower id), so a step costs a shard scan and a gather; the last merge writes the last
     * token. One rank: the reference op dflash_markov_greedy_f32 over the whole vocabulary. Declared after the top-K
     * walk, so the planner keeps the logits live to the last step. */
    if (g_sw_walk_env.load(std::memory_order_relaxed) >= 0) {
        d.wseqs = d.Q / d.block;
        const int64_t KR = ds41walk::plane_rows(d.block, d.wseqs);
        bool wok = true;
        if (g.world > 1 && !rad_op_declared(b, "dspark_walk_step")) {
            wok = false;   /* no loaded kernel plugin serves it (libr4d does): today's walk only */
        } else if (g.world > 1) {
            /* 16-byte key slots (ds41_walk_geom.h): the card's all-gather moves whole 16-byte words a row. */
            d.wkeys = decl_b(b, nm.f("dspark_walk_keys"), RAD_F32, { KR, ds41walk::keys_cols() });
            d.wgath = decl_b(b, nm.f("dspark_walk_keys_all"), RAD_F32, { KR, ds41walk::gath_cols(g.world) });
            d.wfill = op(b, "fill", { RAD_RANGE("M", 1, KR), RAD_INT("n", ds41walk::keys_cols()), RAD_STR("dtype", "f32"),
                                      RAD_F64("value", 0.0) },
                         {}, {}, { d.wkeys });
            d.wag = op(b, "all_gather", { RAD_INT("world_size", g.world), RAD_INT("numel", ds41walk::gather_numel(d.wseqs)),
                                          RAD_STR("dtype", "f32"), RAD_INT("row", ds41walk::gather_row()) },
                       {}, { d.wkeys }, { d.wgath });
            for (int64_t l = 0; l <= d.block; ++l) {
                const rad_op w = op(b, "dspark_walk_step",
                                    { RAD_RANGE("M", 1, d.wseqs), RAD_INT("step", l), RAD_INT("steps", d.block),
                                      RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("N", g.n_vocab),
                                      RAD_INT("vocab_off", g.vocab_off), RAD_INT("world", g.world),
                                      RAD_INT("anchor_stride", d.block), RAD_STR("dtype", "f32") },
                                    { d.mk1, d.mk2 }, { d.logits, d.wgath, d.wkeys }, { d.wkeys, d.tokens });
                wok = wok && w && rad_op_resolved(b, w);   /* declared, and a kernel serves it on this device */
                d.wstep.push_back(w);
            }
            /* W1b: the scoring steps at form 1 (a libr4d without `form` refuses them: only dspark_walk 2 is lost) */
            bool w2ok = true;
            for (int64_t l = 0; l < d.block; ++l) {
                const rad_op w = op(b, "dspark_walk_step",
                                    { RAD_RANGE("M", 1, d.wseqs), RAD_INT("step", l), RAD_INT("steps", d.block),
                                      RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("N", g.n_vocab),
                                      RAD_INT("vocab_off", g.vocab_off), RAD_INT("world", g.world),
                                      RAD_INT("anchor_stride", d.block), RAD_STR("dtype", "f32"), RAD_INT("form", 1) },
                                    { d.mk1, d.mk2 }, { d.logits, d.wgath, d.wkeys }, { d.wkeys, d.tokens });
                w2ok = w2ok && w && rad_op_resolved(b, w);
                d.wstep2.push_back(w);
            }
            if (!w2ok) d.wstep2.clear();
            wok = wok && d.wkeys && d.wgath && d.wfill && d.wag && rad_op_resolved(b, d.wfill) && rad_op_resolved(b, d.wag);
        } else {
            const int64_t tiles = (c.n_vocab + 63) / 64;
            d.wts = decl_b(b, nm.f("dspark_walk_tile_scores"), RAD_F32, { d.wseqs, tiles });
            d.wtt = decl_b(b, nm.f("dspark_walk_tile_tokens"), RAD_I32, { d.wseqs, tiles });
            d.wmarkov = op(b, "dflash_markov_greedy_f32",
                           { RAD_RANGE("M", 1, d.wseqs), RAD_INT("steps", d.block), RAD_INT("rank", d.rank),
                             RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", d.block), RAD_STR("dtype", "bf16") },
                           { d.mk1, d.mk2 }, { d.logits }, { d.wts, d.wtt, d.tokens });
            wok = d.wts && d.wtt && d.wmarkov && rad_op_resolved(b, d.wmarkov);
        }
        if (wok) {
            g_sw_walk_ok.store(1, std::memory_order_relaxed);
            if (!d.wstep2.empty()) g_sw_walk2_ok.store(1, std::memory_order_relaxed);
        } else {
            /* Anything declared stays unissued: the switch keeps refusing 1 and the walk is today's. */
            d.wfill = d.wag = d.wmarkov = 0;
            d.wstep.clear();
            d.wstep2.clear();
            rad_note(b, "DSpark lever W1 NOT declared: no kernel serves %s here; the top-%lld walk only",
                     g.world > 1 ? "dspark_walk_step" : "dflash_markov_greedy_f32", (long long)R);
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_WALK is set but no loaded kernel serves %s: the "
                                "full-vocabulary walk is not declared (dspark_walk takes only 0)\n",
                        g.world > 1 ? "dspark_walk_step" : "dflash_markov_greedy_f32");
        }
    }

    /* Lever SD (11.11.2026, the user's GO: distribution-exact, not byte-identical at temperature > 0): under
     * --draft-coupling on (RadBuildCtx::draft_coupling) the W1 walk again, SAMPLED -- at each position the eight best
     * keys of the whole vocabulary, Q their FP64 softmax at the request's temperature, x ~ Q at the request's seed
     * (r4d_ds41_wsamp_math.h), the state the sampler's coupled verify (mtp_couple_verify: accept min(1, P / Q), else
     * the residual) reads. Issued in place of the walk while the engine's fn4 draft_sample is >= 1; a greedy request's
     * row drafts W1's token (the best key), so temperature 0 drafts what draft_sample 0 drafts. Planes named as the MTP
     * head's, which is how the engine finds them; every rank declares them (the engine stages the draws on all). */
    if (ctx->draft_coupling) {
        const char* why = nullptr;
        if (g.world < 2 || g.world > 4) why = "it runs over 2 .. 4 vocabulary shards (TP2 .. TP4)";
        else if (d.block > RAD_MTP_COUPLE_MAX_ROUNDS) why = "the block is longer than the coupling's rounds";
        else if (!rad_op_declared(b, "dspark_walk_sample_step")) why = "no loaded kernel plugin serves dspark_walk_sample_step";
        if (!why) {
            d.sseqs = d.Q / d.block;
            d.snblk = (g.n_vocab + 127) / 128;
            const int64_t KR = d.block * d.sseqs, SR = 16;
            /* The steps first, without their buffers, so a device none of them resolves on declares no coupling plane
             * (the engine then refuses --draft-coupling on by name instead of serving it uncoupled). */
            const rad_weight ws[2] = { d.mk1, d.mk2 };
            bool sok = true;
            for (int64_t l = 0; l <= d.block; ++l) {
                const RadParam ps[] = { RAD_RANGE("M", 1, d.sseqs), RAD_INT("step", l), RAD_INT("steps", d.block),
                                        RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("N", g.n_vocab),
                                        RAD_INT("vocab_off", g.vocab_off), RAD_INT("world", g.world),
                                        RAD_INT("anchor_stride", d.block), RAD_STR("dtype", "f32") };
                const rad_op w = rad_decl_op(b, "dspark_walk_sample_step", ps, (int)(sizeof ps / sizeof ps[0]), ws, 2);
                sok = sok && w && rad_op_resolved(b, w);
                d.sstep.push_back(w);
            }
            if (!sok) why = "no kernel serves dspark_walk_sample_step on this device";
            if (!why) {
                d.sdraw = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                 { g.max_seqs, (int64_t)RAD_MTP_COUPLE_DRAW_WORDS }, RAD_BUF_PERSIST);
                d.sqt = decl_b(b, nm.f("mtp_couple_qtemp"), RAD_F32, { g.max_seqs, 1 }, RAD_BUF_PERSIST);
                d.sstate = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                  { d.block, g.max_seqs, (int64_t)RAD_MTP_COUPLE_STATE_WORDS }, RAD_BUF_PERSIST);
                d.skeys = decl_b(b, nm.f("dspark_wsamp_keys"), RAD_F32, { KR, SR });
                d.sgath = decl_b(b, nm.f("dspark_wsamp_keys_all"), RAD_F32, { KR, SR * g.world });
                d.sscr = decl_b(b, nm.f("dspark_wsamp_scratch"), RAD_F32, { d.sseqs * d.snblk, SR });
                d.scnt = decl_b(b, nm.f("dspark_wsamp_count"), RAD_F32, { d.block, 4 });
                if (!d.sdraw || !d.sqt || !d.sstate || !d.skeys || !d.sgath || !d.sscr || !d.scnt) return RAD_E_INVAL;
                d.sfill = op(b, "fill", { RAD_RANGE("M", 1, d.block), RAD_INT("n", 4), RAD_STR("dtype", "f32"),
                                          RAD_F64("value", 0.0) },
                             {}, {}, { d.scnt });
                d.sag = op(b, "all_gather", { RAD_INT("world_size", g.world), RAD_INT("numel", d.sseqs * SR),
                                              RAD_STR("dtype", "f32"), RAD_INT("row", SR) },
                           {}, { d.skeys }, { d.sgath });
                if (!d.sfill || !d.sag || !rad_op_resolved(b, d.sfill) || !rad_op_resolved(b, d.sag))
                    return RAD_E_UNSUPPORTED;   /* the stock fill / gather: every device that runs the walk has them */
                for (const rad_op w : d.sstep)
                    rw(b, w, { d.logits, d.sgath, d.skeys, d.sscr, d.scnt, d.sdraw, d.sqt, d.sstate },
                       { d.skeys, d.tokens, d.sscr, d.scnt, d.sstate });
            }
        }
        if (why) {
            d.sfill = d.sag = 0;
            d.sstep.clear();
            /* No planes named mtp_couple_*: the engine refuses --draft-coupling on by name at bring-up. */
            rad_note(b, "DSpark lever SD (sampled drafting) NOT declared: %s", why);
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: --draft-coupling on: DSpark lever SD (sampled drafting) NOT "
                                "declared: %s\n", why);
        } else {
            rad_note(b, "DSpark lever SD: the sampled full-vocabulary walk declared (dspark_walk_sample_step a step over "
                        "%d shard(s) + an all_gather of eight keys a sequence, %lld sequences a pass, %lld blocks a shard); "
                        "fn4 draft_sample (the engine's) issues it", g.world, (long long)d.sseqs, (long long)d.snblk);
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: DSpark lever SD: the sampled full-vocabulary walk declared "
                                "(dspark_walk_sample_step a step over %d rank(s) + all_gather of the eight best keys, %lld "
                                "sequences a pass); x ~ Q = softmax(score / (temperature x scale)) over the eight best, "
                                "the maximal coupling at the verify; issued while fn4 draft_sample >= 1 "
                                "(RADIANCE_DRAFT_SAMPLE at start)\n",
                        g.world, (long long)d.sseqs);
        }
    }

    /* Lever C1 (rank 0): the confidence head after the walk, over the collapse's output before its
     * norm (d.x, which nothing after the collapse writes) and the walk's predecessors; its weights in a
     * persistent buffer the first query pass fills from the sidecar. */
    if (!d.cw.empty() && !rad_drafter_confidence) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_CONF: this host has no rad_drafter_confidence; "
                        "the confidence head is not declared\n");
        d.cw.clear();
    }
    if (d.n_alt > 0 && !rad_drafter_alternatives) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_ALT: this host has no rad_drafter_alternatives; "
                        "the branches are not declared\n");
        d.n_alt = 0;
    }
    if (!d.cw.empty()) {
        const int64_t W = (int64_t)d.cw.size();
        d.cwb  = decl_b(b, nm.f("dspark_conf_w"), RAD_F32, { 1, W }, RAD_BUF_PERSIST);
        d.conf = decl_b(b, nm.f("dspark_conf"), RAD_F32, { g.max_seqs, d.block }, RAD_BUF_PERSIST);
        if (!d.cwb || !d.conf) return RAD_E_INVAL;
        d.conf_op = op(b, "dspark_conf",
                       { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", d.block), RAD_INT("n_embd", D),
                         RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", d.block),
                         RAD_F64("bias", d.cbias), RAD_STR("dtype", "bf16") },
                       { d.mk1 }, { d.x, d.tokens, d.cwb }, { d.conf });
        if (!d.conf_op) return RAD_E_INVAL;
    }
    /* Lever T1 (rank 0): the walk's next-best first proposals, each walked on. */
    if (d.n_alt > 0) {
        d.alt = decl_b(b, nm.f("dspark_alt"), RAD_I32, { g.max_seqs, d.n_alt * d.block }, RAD_BUF_PERSIST);
        if (!d.alt) return RAD_E_INVAL;
        d.alt_op = op(b, "dspark_alt",
                      { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", d.block), RAD_INT("top_k", R),
                        RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", d.block),
                        RAD_INT("n_alt", d.n_alt), RAD_STR("dtype", "bf16") },
                      { d.mk1, d.mk2 }, { d.cand_, d.unary, d.tokens }, { d.alt });
        if (!d.alt_op) return RAD_E_INVAL;
    }

    RadDrafterDecl drf{};
    drf.name = "dspark";
    drf.kind = RAD_DRAFT_BLOCK;
    drf.depth = ctx->max_spec;
    drf.block = d.block;
    drf.proposal = d.tokens;
    drf.proposal_pitch = d.block;
    /* The query block's first row reads back `window` keys before itself. */
    drf.window = c.window + 1;
    drf.mask_token = (int32_t)d.noise;
    DS_TRY(rad_declare_drafter(b, &drf));
    if (d.conf) DS_TRY(rad_drafter_confidence(b, d.conf, d.block));
    if (d.alt) DS_TRY(rad_drafter_alternatives(b, d.alt, d.n_alt, d.block));
    if (d.conf || d.alt)
        rad_note(b, "DSpark extras (ds41-acc): %s%s%s", d.conf ? "the confidence head (sidecar " : "",
                 d.conf ? d.conf_path.c_str() : "", d.conf ? ", dspark_conf)" : "");
    if (d.alt) rad_note(b, "DSpark extras (ds41-acc): %lld shadow branch(es) at position 1 (dspark_alt)", (long long)d.n_alt);
    rad_note(b, "DSpark drafter: %d stages at absolute layers %d..%d (window-only V4 blocks, %lld experts top-%lld), "
                "taps before %zu trunk layer(s), one pass over [anchor, noise %lld x %lld] proposes %lld tokens; "
                "attention over %lld context keys + the block's own; Markov rank %lld walked over the head's top %lld; "
                "the confidence head %s",
             d.n_stage, d.base, d.base + d.n_stage - 1, (long long)d.n_expert, (long long)d.top_k, d.taps.size(),
             (long long)d.noise, (long long)(d.block - 1), (long long)d.block, (long long)c.window,
             (long long)d.rank, (long long)R,
             d.conf ? "is declared (its gate is the engine's dspark_conf switch)" : "is not applied");
    /* The same on stderr once (rank 0's real declare), so a serve log says which walk ran. */
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: DSpark drafter: %d stages, %lld experts top-%lld, block %lld; Markov rank %lld "
                        "walked over the head's top %lld (RADIANCE_DS41_DSPARK_CAND)\n",
                d.n_stage, (long long)d.n_expert, (long long)d.top_k, (long long)d.block, (long long)d.rank, (long long)R);
    if (ctx->rank == 0 && !ctx->shape_probe && (d.conf || d.alt))
        fprintf(stderr, "radiance: deepseek41: DSpark extras (ds41-acc): confidence head %s%s%s, shadow branches %lld "
                        "(switches dspark_conf / dspark_alt; neither changes a proposal)\n",
                d.conf ? "from " : "off", d.conf ? d.conf_path.c_str() : "",
                d.conf ? (d.cbias != 0.0 ? " with a bias" : " (no bias, as the release)") : "", (long long)d.n_alt);
    if (d.wfill || d.wmarkov) {
        rad_note(b, "DSpark lever W1: the full-vocabulary Markov walk declared (%s), dspark_walk %d",
                 d.wfill ? "a dspark_walk_step a step on the rank's shard + an all_gather of the keys" : "dflash_markov_greedy_f32",
                 g_sw_walk.load(std::memory_order_relaxed));
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: DSpark lever W1: the full-vocabulary Markov walk declared (%s over %d "
                            "rank(s), %lld sequences a pass); fn4 dspark_walk=%d (0 the head's top %lld re-ranked, 1 the "
                            "whole vocabulary, 2 the same with succ staged through LDS -- W1b, %s; RADIANCE_DS41_DSPARK_WALK)\n",
                    d.wfill ? "dspark_walk_step a step + all_gather" : "dflash_markov_greedy_f32", g.world,
                    (long long)d.wseqs, g_sw_walk.load(std::memory_order_relaxed), (long long)R,
                    d.wstep2.empty() ? "not declared" : "declared");
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ declare */
static int bt_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx);   /* lever BI's tracer, below */
static int psk_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx);  /* lever PSK, below */
static int bi_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx);   /* lever BI, below */
static void engdd_declare(Model& m, const RadBuildCtx* ctx);               /* lever E1, below */
static int kvr_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx);  /* lever KVR, below */

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16) {
        fprintf(stderr, "radiance: deepseek41: its caches hold bf16 (lossless for the fp8 and fp4 "
                        "values the model writes); run with --kv-cache-dtype bf16\n");
        return RAD_E_UNSUPPORTED;
    }
    /* A sizing declare (RadBuildCtx::shape_probe, the core's arena levels) writes scratch: the
     * step reads models[], filled by the real declare at the real max_tok, and the probes run on
     * parallel threads. probe() offers them only where levels_ok(). */
    static thread_local std::unique_ptr<Model> probe_model;
    if (ctx->shape_probe && !models[ctx->rank]) return RAD_E_INVAL;   /* no real declare to size against */
    auto& slot = ctx->shape_probe ? probe_model : models[ctx->rank];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    /* RADIANCE_DS41_SHEXP_LANE once, at the first declare: a later one (a sizing probe, a re-declare)
     * keeps what the ds41_shexp_lane switch has set since. */
    {
        int unread = -1;
        if (g_sw_shexp_env.compare_exchange_strong(unread, shexp_lane_env()))
            g_sw_shexp.store(g_sw_shexp_env.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    k1_read_env();
    walk_read_env();
    bi_read_env();
    psk_read_env();
    psk_l20_read_env();
    pfctx_read_env();
    psk_cr_read_env();
    engdd_read_env();
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    DS_TRY(read_cfg(m.c, meta));
    m.release = is_release(meta);
    m.n_stage = (int)gi(meta, "nextn_predict_layers", 0);
    naming = &m;
    DS_TRY(geometry(m, ctx));
    DS_TRY(dsp_config(b, m, meta, ctx));
    const auto& c = m.c;
    auto& g = m.g;
    int64_t rmin = 0;
    for (int l = 0; l < c.n_layer; ++l)
        if (c.ratio[(size_t)l] > 0 && (rmin == 0 || c.ratio[(size_t)l] < rmin)) rmin = c.ratio[(size_t)l];
    m.work = g.max_tok + g.max_seqs;
    m.cmax = g.max_ctx / std::max<int64_t>(rmin, 1) + 1;
    index_runs(m);
    index_mode(m);
    if (ctx->shape_probe) {
        /* A level shrinks only what the step indexes by its own rows T. Everything the step
         * addresses by a capacity of the real declare keeps the real size: the work plane m.work
         * (comp_stage is written rows x r > T at a small step: at r 128 and 8 sequences a decode
         * step's 9 blocks are 1152 stage rows), the score runs (irows), the split's parts
         * (iparts, qmax: rank r's selections land at r x qmax x irows), the keep words. */
        const Model& full = *models[ctx->rank];
        m.work = full.work;
        m.irows = full.irows;
        m.iparts = full.iparts;
        m.kwords = full.kwords;
        m.ikeep = full.ikeep;
        m.isplit = full.isplit;
        m.qmax = full.qmax;
        /* Lever Z0-A (ds41-z0): the DSpark head's query rows Q (max_seqs x block, capped by max_tok: dsp_config) size
         * FIXED parameters -- the candidates' all_gather numel Q x R x 2, the walk's wseqs = Q / block and its gather --
         * so a level below Q (the 8-row rung at NST 5: Q 8 instead of 40, numel 512 instead of 2560) declared a different
         * graph and was refused ("no level for 8-token steps"). A level keeps the real Q: the head's buffers keep their real
         * rows (not larger, so plan_level takes them), and the head never runs at such a level (Engine::arena_step gives
         * a step below the largest decode rung only when it verifies nothing and runs no head). */
        m.dsp.Q = full.dsp.Q;
        m.dsp.on = full.dsp.on;
        /* Lever VA "arena": below irows the plane and the selections follow the level's rows. */
        if (ds41_arena_decode_rows() > 0 && g.max_tok < m.irows) m.score_rows = g.max_tok;
    }
    DS_TRY(buffers(b, m));
    /* The drafter's source rows, one hc-mean row a target layer, written by the trunk step. */
    if (m.dsp.on && !m.zero &&
        !(m.dsp.src = decl_b(b, nm.f("dspark_src"), RAD_BF16, { g.max_tok, (int64_t)m.dsp.taps.size() * c.n_embd })))
        return RAD_E_INVAL;

    const int64_t T = g.max_tok, D = c.n_embd;
    /* The vocabulary is split over the ranks (shard_extent): the lookup gives zeros outside this
     * rank's rows and an all-reduce completes it; the head's logits stay sharded for the sampler. */
    const int64_t v0 = g.vocab_off, v1 = g.vocab_off + g.n_vocab;
    m.token = lw(b, m, nm.f("token_embd.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1,
                 false, RAD_ACCESS_VOCAB);
    m.head = lw(b, m, nm.f("output.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1, false,
                RAD_ACCESS_VOCAB);
    if (!m.token || !m.head) return RAD_E_INVAL;
    m.final_gain = wt(b, nm.f("output_norm.weight"), RAD_F32, { D }, grp_model());
    /* A GGUF's table in llama.cpp's blocks is gathered from them (embed_lookup_ggml). */
    const auto tt = m.wty.find(m.token);
    const auto ts = m.wsc.find(m.token);
    const auto tc = m.wcod.find(m.token);
    if (ts != m.wsc.end() && tc != m.wcod.end() && tc->second != RAD_F8E4M3)
        m.embed = op(b, "embed_lookup_iqr",
                     { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token, ts->second }, {}, { m.xcol });
    else if (ts != m.wsc.end() && tc != m.wcod.end() && tc->second == RAD_F8E4M3)
        m.embed = op(b, "embed_lookup_q",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off),
                       RAD_F64("wscale", 1.0) },
                     { m.token, ts->second }, {}, { m.xcol });
    else if (tt != m.wty.end())
        m.embed = op(b, "embed_lookup_ggml",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("type", tt->second), RAD_STR("dtype", "bf16"),
                       RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { m.xcol });
    else
        m.embed = op(b, "embed_lookup",
                     { RAD_RANGE("M", 1, T), RAD_INT("n_embd", D), RAD_INT("n_vocab", g.n_vocab),
                       RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", g.vocab_off) },
                     { m.token }, {}, { m.xcol });
    if (g.world > 1)
        m.embed_ar = op(b, "all_reduce",
                        { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1) },
                        {}, { m.xcol }, { m.xcol });
    m.expand = op(b, "cast",
                  { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("from", "bf16"),
                    RAD_STR("to", "bf16") },
                  {}, { m.xcol }, { m.h0 });
    /* Layer 0's attention collapses with one-hot gates: stream 0. */
    m.pre0 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", c.hc), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 0.0) },
                {}, {}, { m.pre_f });
    m.pre1 = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 1.0) },
                {}, {}, { m.pre_f });
    /* Lever K1 fold f (RADIANCE_DS41_ONEFILL): the MoE layers' ungated shared gate, a plane of bf16 ones, filled
     * once at the step's start -- the op each layer issues (MoeFP8 op_sgate: fill n 1, value 1, the block's
     * dtype) over the step's rows. Only where this rank has a shared slice (the layers then take it). */
    if (g_sw_one_env.load(std::memory_order_relaxed) > 0 && g.n_ff > 0 && m.mw.sgate) {
        m.sone = op(b, "fill", { RAD_RANGE("M", 1, T), RAD_INT("n", 1), RAD_STR("dtype", g.dtype),
                                 RAD_F64("value", 1.0) },
                    {}, {}, { m.mw.sgate });
        if (!m.sone) return RAD_E_INVAL;
    }
    m.rope_a = op(b, "rope_table",
                  { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.theta),
                    RAD_F64("scale", 1.0), RAD_STR("dtype", "f32") },
                  {}, {}, { m.cs_a });
    m.rope_b_step = op(b, "rope_table",
                       { RAD_RANGE("M", 1, T), RAD_INT("rot", c.rope), RAD_F64("theta", c.comp_theta),
                         RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
                         RAD_F64("yarn_factor", c.yarn_factor > 1.0 ? c.yarn_factor : 0.0),
                         RAD_F64("yarn_low", c.yarn_low), RAD_F64("yarn_high", c.yarn_high),
                         RAD_F64("yarn_magnitude", 1.0) },
                       {}, {}, { m.cs_b });
    if (!c.ple.empty()) {
        m.tmap = wt(b, "deepseek41.engram.token_map", RAD_I32, { c.n_vocab }, grp_model());
        /* Bf16, or a single-level affine encoding (codes, a scale per group, an optional
         * codebook: a GGUF's IQ4_NL, Q8_0, Q4_0) read in place from its stored planes, since the
         * table (103 GiB in the 1017) is too large to relayout or copy; it stays on whatever tier
         * holds the container's bytes. */
        const char* tn = nm.f("per_layer_token_embd.weight");
        if (m.release) {
            /* Every Engram layer's table, joined in layer order (the GGUF's one table). */
            for (size_t i = 0; i < c.ple.size(); i += 8) {
                std::vector<const char*> parts;
                for (size_t k = i; k < c.ple.size() && k < i + 8; ++k)
                    parts.push_back(nm.ckpt("layers.%lld.engram.embed.weight", (long long)c.ple[k]));
                RadNameMap mp = map_base(tn, RAD_MAP_CONCAT);
                for (const char* p : parts) mp.src[mp.n_src++] = p;
                DS_TRY(rad_decl_name_map(b, &mp));
            }
        } else {
            DS_TRY(map_copy(b, tn, tn));
        }
        RadEncoding te{};
        const bool known = weight_enc(b, tn, &te);
        if (!known || (rad_enc_is(&te, "plain") && te.n_planes == 1)) {
            m.ple_table = decl_w(b, tn, RAD_BF16, { c.ple_rows, c.ple_dim }, RAD_ACCESS_VOCAB,
                                 RAD_SHARD_NONE, grp_model());
            if (!m.ple_table) return RAD_E_INVAL;
        } else {
            const RadEncPlane* cp = rad_enc_plane(&te, "codes");
            const RadEncPlane* sp = rad_enc_plane(&te, "scale");
            const RadEncPlane* tp = rad_enc_plane(&te, "table");
            const int64_t G = sp ? sp->block[1] : 0;
            if (!rad_enc_is(&te, "affine") || te.transform[0] || !cp || !sp || cp->block[0] != 1 ||
                cp->block[1] != 1 || sp->block[0] != 1 || G <= 0 || c.ple_dim % G ||
                te.n_planes != 2 + (tp ? 1 : 0)) {
                char en[256];
                rad_enc_format(&te, en, sizeof en);
                fprintf(stderr, "radiance: deepseek41: the Engram table is %s; it is read as bf16 or "
                                "as codes with one scale per group of a row and an optional "
                                "codebook\n", en);
                return RAD_E_DTYPE;
            }
            m.ple_group = G;
            m.ple_codes = decl_view(b, tn, nullptr, "codes", cp->dtype, { c.ple_rows, c.ple_dim },
                                    RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
            m.ple_scale = decl_view(b, nm.f("per_layer_token_embd.scale"), tn, "scale", sp->dtype,
                                    { c.ple_rows, c.ple_dim / G }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                    grp_model());
            if (tp)
                m.ple_tab = decl_view(b, nm.f("per_layer_token_embd.table"), tn, "table", tp->dtype,
                                      { tp->extent[0], tp->extent[1] }, RAD_ACCESS_PER_TOKEN,
                                      RAD_SHARD_NONE, grp_model());
            if (!m.ple_codes || !m.ple_scale || (tp && !m.ple_tab)) return RAD_E_INVAL;
        }
        m.tok_map = op(b, "token_map", { RAD_RANGE("M", 1, T), RAD_INT("n_vocab", c.n_vocab) },
                       { m.tmap }, {}, { m.ctok });
    }
    m.layers.resize((size_t)c.n_layer);
    if (ctx->calib_dir && ctx->calib_dir[0]) {
        m.calib_lo = ctx->calib_layer_lo;
        m.calib_hi = ctx->calib_layer_hi > ctx->calib_layer_lo ? ctx->calib_layer_hi : (int)c.n_layer;
    }
    for (int l = 0; l < c.n_layer; ++l) DS_TRY(layer(b, m, m.layers[(size_t)l], l));
    m.pcn_fin = post_collapse_norm(b, m, T, m.final_gain, m.act.x, m.h1, m.h0, m.pre_f, m.xcol,
                                   m.final_hidden);
    if (m.pcn_fin && c.n_layer > 0)
        DS_TRY(ar_pcn_moe_declare(b, m, m.layers[(size_t)c.n_layer - 1], T, m.final_gain, m.final_hidden,
                                  &m.ar_pcn_fin));
    if (!m.pcn_fin)
        m.cn_fin = collapse_norm(b, m, T, m.final_gain, m.h0, m.pre_f, m.xcol, m.final_hidden);
    if (!m.pcn_fin && !m.cn_fin) {
        m.contract = op(b, "mhc_collapse",
                        { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                        {}, { m.h0, m.pre_f }, { m.xcol });
        m.fnorm = rmsnorm(b, m, T, m.final_gain, D, m.xcol, m.final_hidden);
    }
    m.gather = op(b, "gather_rows",
                  { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                  {}, { m.final_hidden }, { m.logit_input });
    const auto ht = m.wty.find(m.head);
    const auto hs = m.wsc.find(m.head);
    const auto hc = m.wcod.find(m.head);
    if (hs != m.wsc.end() && hc != m.wcod.end() && hc->second == RAD_I8)
        m.head_op = op(b, "logits_gemm_iqr",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                       { m.head, hs->second }, { m.logit_input }, { m.logits });
    else if (hs != m.wsc.end() && hc != m.wcod.end())
        m.head_op = op(b, "logits_gemm",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    else if (ht != m.wty.end())
        m.head_op = op(b, "logits_gemm_ggml",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("type", ht->second), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    else
        m.head_op = op(b, "logits_gemm",
                       { RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                         RAD_INT("n_embd", D), RAD_STR("dtype", "bf16") },
                       { m.head }, { m.logit_input }, { m.logits });
    /* The drafter after everything the trunk step issues (llama_fp8's rule: declaration order is
     * liveness, and its passes run behind the step). */
    if (m.dsp.on) DS_TRY(dsp_declare(b, m, ctx));
    rad_note(b, "DeepSeek-V4.1: %lld layers, hc %lld, window %lld, %zu kv sources, %zu index "
                "sources, %lld experts top-%lld, Engram at %zu layers",
             (long long)c.n_layer, (long long)c.hc, (long long)c.window, c.kv_src.size(),
             c.idx_src.size(), (long long)c.n_expert, (long long)c.top_k, c.ple.size());
    int n_arpcn = 0, n_arpcn_moe = m.ar_pcn_fin ? 1 : 0;
    for (const Layer& Lx : m.layers) {
        n_arpcn += Lx.ar_pcn_f ? 1 : 0;
        n_arpcn_moe += Lx.ar_pcn_a ? 1 : 0;
    }
    rad_note(b, "deepseek41 indexer: candidate restriction %s from %lld positions, row split %s from %lld (runs of "
                "%lld rows, %lld a rank at most; RADIANCE_DS41_INDEX, _KEEP_MIN_LIVE, _SPLIT_MIN_LIVE), keys %s "
                "(%lld bytes an entry; RADIANCE_DS41_IDXKEY), latent %s (%lld bytes an entry; RADIANCE_DS41_LATENT), "
                "compressor tail %s (RADIANCE_DS41_CTAIL), router selection+sort %s (RADIANCE_DS41_ROUTER_FUSE), "
                "attention all-reduce + post-collapse-norm %s at %d layers, MoE all-reduce + next collapse-norm "
                "%s at %d layers (RADIANCE_DS41_AR_PCN)",
             m.ikeep ? "on" : "off", (long long)m.keep_min_live, m.isplit && m.sel_gather ? "on" : "off",
             (long long)m.split_min_live, (long long)m.irows, (long long)m.qmax, m.ikmx ? "mxfp4" : "bf16",
             (long long)(2 * index_slots(m)), m.lnv ? "nvfp4" : "bf16", (long long)(2 * latent_slots(m, c.head_dim)),
             m.ctail_full ? "full" : "bound", ds41_router_fuse() ? "fused" : "separate", n_arpcn ? "fused" : "separate",
             n_arpcn, n_arpcn_moe ? "fused" : "separate", n_arpcn_moe);
    if (ctx->rank == 0 && !ctx->shape_probe && g.world == 4 && gather_ar_rows() > 0) {
        if (gather_ar_max() > 0)
            fprintf(stderr, "radiance: deepseek41: the MoE gather and its all-reduce as one kernel up to %lld "
                            "tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX; %d layers declare "
                            "it up to %d rows; the one-shot below the all-reduce's ts_min, the two-shot from it; "
                            "switch ds41_argather)\n", (long long)gather_ar_rows(),
                    (int)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.moe.op_gar != 0; }),
                    g_sw_gar_decl.load(std::memory_order_relaxed));
        else
            fprintf(stderr, "radiance: deepseek41: the MoE gather and its all-reduce as one kernel up to %lld "
                            "tokens a step (RADIANCE_DS41_ARGATHER)\n", (long long)gather_ar_rows());
    }
    if (ctx->rank == 0 && !ctx->shape_probe && g_sw_shexp.load(std::memory_order_relaxed) > 0)
        fprintf(stderr, "radiance: deepseek41: the shared expert on the second lane beside the routed experts "
                        "at a decode or verify step of up to %lld rows, joined before the gather "
                        "(RADIANCE_DS41_SHEXP_LANE; switch ds41_shexp_lane): %d of %zu layers lane-safe\n",
                (long long)kMoeShexpLaneRows,
                (int)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.moe.shexp_lane_ok; }),
                m.layers.size());
    if (ctx->rank == 0 && !ctx->shape_probe &&
        (g_sw_kvf_env.load(std::memory_order_relaxed) > 0 || g_sw_of_env.load(std::memory_order_relaxed) > 0 ||
         g_sw_one_env.load(std::memory_order_relaxed) > 0 || g_sw_qnf_env.load(std::memory_order_relaxed) > 0 ||
         g_sw_swf_env.load(std::memory_order_relaxed) > 0))
        {
        /* folds a / e (named only when their environment asked, so the c / d / f wording above is unchanged) */
        char qnote[256] = "", snote[200] = "";
        if (g_sw_qnf_env.load(std::memory_order_relaxed) > 0)
            std::snprintf(qnote, sizeof qnote, ", the query norm folded into q_b's GEMV staging in %d of %zu layers (%d "
                          "with the indexer; RADIANCE_DS41_QNFOLD; ds41_qnfold=%d)",
                          (int)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.q_bf != 0; }),
                          m.layers.size(),
                          (int)std::count_if(m.layers.begin(), m.layers.end(),
                                             [](const Layer& L) { return L.q_bf != 0 && L.i_score != 0; }),
                          g_sw_qnf.load(std::memory_order_relaxed));
        if (g_sw_swf_env.load(std::memory_order_relaxed) > 0)
            std::snprintf(snote, sizeof snote, ", the shared expert's clamped SwiGLU folded into its down GEMV's "
                          "staging in %d MoE layers (RADIANCE_DS41_SWFOLD; ds41_swfold=%d)",
                          (int)std::count_if(m.layers.begin(), m.layers.end(),
                                             [](const Layer& L) { return L.moe.op_swfold != 0; }),
                          g_sw_swf.load(std::memory_order_relaxed));
        fprintf(stderr, "radiance: deepseek41: lever K1 folds: window key norm+rope+round trip+store in one launch "
                        "at %d of %zu layers (RADIANCE_DS41_KVFOLD; switch ds41_kvfold=%d), attention + output "
                        "rope+round trip at %d (RADIANCE_DS41_OFOLD; ds41_ofold=%d), the shared gate's ones once a "
                        "step %s at %d MoE layers (RADIANCE_DS41_ONEFILL; ds41_onefill=%d)%s%s\n",
                (int)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.kv_fs != 0; }),
                m.layers.size(), g_sw_kvf.load(std::memory_order_relaxed),
                (int)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.attn_rt != 0; }),
                g_sw_of.load(std::memory_order_relaxed), m.sone ? "declared" : "not declared",
                (int)std::count_if(m.layers.begin(), m.layers.end(),
                                   [](const Layer& L) { return L.moe.sone_live != nullptr; }),
                g_sw_one.load(std::memory_order_relaxed), qnote, snote);
    }
    /* Lever KVWIN (RADIANCE_DS41_KVWIN_FP8): said only when asked for, so an unset start logs what it did. */
    if (const char* wx = std::getenv("RADIANCE_DS41_KVWIN_FP8"); wx && *wx && std::strcmp(wx, "0") && std::strcmp(wx, "off")) {
        const int64_t wb = 2 * window_slots(c.head_dim, true), pb = 2 * window_slots(c.head_dim, false);
        rad_note(b, "lever KVWIN: window keys fp8_e4m3, %lld bytes a key instead of %lld: kv_window %s, kv_dspark %s "
                    "(RADIANCE_DS41_KVWIN_FP8=%s)",
                 (long long)wb, (long long)pb, m.w8 ? "packed" : "bf16",
                 !m.dsp.on ? "none" : m.w8d ? "packed" : "bf16", wx);
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: lever KVWIN: the window caches hold each key as fp8_e4m3 codes and a "
                            "scale byte a block of 32, %lld bytes a key and layer instead of %lld: kv_window %s at %zu "
                            "layers, kv_dspark %s (RADIANCE_DS41_KVWIN_FP8=%s; window %lld, latent %lld wide)\n",
                    (long long)wb, (long long)pb, m.w8 ? "packed" : "bf16 (the window is not a multiple of 32)",
                    (size_t)std::count_if(m.layers.begin(), m.layers.end(), [](const Layer& L) { return L.kv_st != 0; }),
                    !m.dsp.on ? "not declared" : m.w8d ? "packed" : "bf16", wx,
                    (long long)c.window, (long long)c.head_dim);
    }
    /* The same on stderr once (rank 0's real declare), so a serve log says which indexer ran. */
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41 indexer: candidate restriction %s from %lld positions, row split %s "
                        "from %lld (runs of %lld rows, %lld a rank at most; RADIANCE_DS41_INDEX, _KEEP_MIN_LIVE, "
                        "_SPLIT_MIN_LIVE), keys %s (%lld bytes an entry; RADIANCE_DS41_IDXKEY), latent %s (%lld bytes an "
                        "entry; RADIANCE_DS41_LATENT), compressor tail %s (RADIANCE_DS41_CTAIL), router selection+sort %s "
                        "(RADIANCE_DS41_ROUTER_FUSE), attention all-reduce + post-collapse-norm %s at %d layers, "
                        "MoE all-reduce + next collapse-norm %s at %d layers (RADIANCE_DS41_AR_PCN)\n",
                m.ikeep ? "on" : "off", (long long)m.keep_min_live, m.isplit && m.sel_gather ? "on" : "off",
                (long long)m.split_min_live, (long long)m.irows, (long long)m.qmax, m.ikmx ? "mxfp4" : "bf16",
                (long long)(2 * index_slots(m)), m.lnv ? "nvfp4" : "bf16", (long long)(2 * latent_slots(m, c.head_dim)),
                m.ctail_full ? "full" : "bound", ds41_router_fuse() ? "fused" : "separate", n_arpcn ? "fused" : "separate",
                n_arpcn, n_arpcn_moe ? "fused" : "separate", n_arpcn_moe);
    /* The row split's deal (idx_split): the value at start and what 1 does at this geometry, on a mixed step of
     * 8 runs' rows and 4 decode rows (2052 at 256-row runs). */
    if (!ctx->shape_probe) {
        const bool split = m.isplit && m.sel_gather;
        if (split) g_idx_bal_ok.store(1, std::memory_order_relaxed);
        const int64_t ex = 8 * m.irows + 4;
        const Ds41IdxRuns pl = ds41_idx_runs(ex, m.irows, m.split_min_live, m.aw, 0, split, m.split_min_live);
        const Ds41IdxRuns bl = ds41_idx_runs(ex, m.irows, m.split_min_live, m.aw, 0, split, m.split_min_live, true,
                                             m.iparts);
        char what[512];
        if (split)
            snprintf(what, sizeof what, "1 = an equal block of rows a rank where %d ranks do not divide a step's runs, "
                     "within the %lld declared runs (a %lld-row step: %lld runs of %lld, %lld a rank, instead of %lld "
                     "runs of %lld dealt %lld a rank, the last rank %lld)", m.aw, (long long)m.iparts, (long long)ex,
                     (long long)bl.n, (long long)bl.R, (long long)bl.q, (long long)pl.n, (long long)pl.R,
                     (long long)pl.q, (long long)std::max<int64_t>(0, pl.n - (int64_t)(m.aw - 1) * pl.q));
        else
            snprintf(what, sizeof what, "the row split is off, so every step is the unsplit runs and 1 is refused");
        rad_note(b, "deepseek41 indexer deal: idx_split %d (RADIANCE_DS41_IDX_BALANCE at start, POST /admin/fn4 "
                    "idx_split): %s", idx_balance() ? 1 : 0, what);
        if (ctx->rank == 0)
            fprintf(stderr, "radiance: deepseek41 indexer deal: idx_split %d (RADIANCE_DS41_IDX_BALANCE at start, "
                            "POST /admin/fn4 idx_split): %s\n", idx_balance() ? 1 : 0, what);
    }
    DS_TRY(bi_declare(b, m, ctx));
    DS_TRY(psk_declare(b, m, ctx));
    engdd_declare(m, ctx);
    DS_TRY(bt_declare(b, m, ctx));
    DS_TRY(kvr_declare(b, m, ctx));
    return RAD_OK;
}

/* ------------------------------------------------------------------ lever BI's tracer */
static const char* bt_dir() {
    static const char* d = [] {
        const char* e = std::getenv("RADIANCE_DS41_BITRACE");
        return e && *e && std::strcmp(e, "0") ? e : (const char*)nullptr;
    }();
    return d;
}
/* At the end of a declare: the plane and one bi_rowhash a traced buffer (read after everything the model declared, so
 * no traced buffer shares memory with a later op's under tracing). */
/* Lever BI at the end of a declare (RADIANCE_DS41_BI set): each prefill segment's cu view plane and its op. */
static int bi_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (!bi_declared()) return RAD_OK;
    /* The all-reduce + post-collapse-norm collectives (RADIANCE_DS41_AR_PCN) take the step's rows whole; step_bi issues
     * the all-reduce and the fused norms apart, so the two are not combined: lever BI stays undeclared (1 refused). */
    bool arpcn = m.ar_pcn_fin != 0;
    for (const Layer& L : m.layers) arpcn = arpcn || L.ar_pcn_f || L.ar_pcn_a;
    if (arpcn) {
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: lever BI not declared: RADIANCE_DS41_AR_PCN's collectives take a step's "
                            "rows whole (unset it to use RADIANCE_DS41_BI)\n");
        return RAD_OK;
    }
    m.bi_cu = decl_b(b, m.nm.f("bi_cu"), RAD_I32, { m.g.max_seqs + 1, 2 }, RAD_BUF_PERSIST);
    if (!m.bi_cu) return RAD_E_INVAL;
    m.bi_sub = op(b, "sub_i32", { RAD_RANGE("M", 1, 2) }, {}, {}, { m.bi_cu });
    if (!m.bi_sub) return RAD_E_UNSUPPORTED;
    if (!ctx->shape_probe) g_sw_bi_ok.store(1, std::memory_order_relaxed);
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now %d): a step of more "
                        "than one shape class runs in segments by sequence (decode rows <= %lld a dense / attention / "
                        "indexer launch, <= %lld a fused norm; each prefill sequence at its own rows; the MoE a pass a "
                        "segment); a pass with two or more prefill sequences is not recorded\n",
                g_sw_bi.load(std::memory_order_relaxed), (long long)kBiBand, (long long)kBiNorm);
    return RAD_OK;
}

/* Lever PSK at the end of a declare (RADIANCE_DS41_PSK set): the first layer a prefill segment is cut at -- past the
 * last kv source, where no layer is a kv source (L.src: the compressor, the latent and index caches, the tails) and none
 * the indexer's candidate layer (its keep is every row's). Not with RADIANCE_DS41_AR_PCN's collectives (they take a
 * step's rows whole; step_bi issues apart), a calibration tap, or no window. */
static int psk_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    m.psk_first = 0;
    if (!psk_declared()) return RAD_OK;
    const auto& c = m.c;
    int last = -1;
    for (int64_t s : c.kv_src) last = std::max(last, (int)s);
    const int first = last + 1;
    bool ok = last >= 0 && first < (int)c.n_layer && c.window > 0 && m.calib_hi <= m.calib_lo && c.cand < first;
    for (const Layer& L : m.layers) ok = ok && !(L.l >= first && L.src);
    bool arpcn = m.ar_pcn_fin != 0;
    for (const Layer& L : m.layers) arpcn = arpcn || L.ar_pcn_f || L.ar_pcn_a;
    if (!ok || arpcn) {
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: lever PSK not declared: %s\n",
                    arpcn ? "RADIANCE_DS41_AR_PCN's collectives take a step's rows whole (unset it to use RADIANCE_DS41_PSK)"
                          : "no layer past the last kv source is cut-safe here (kv sources, candidate layer, window, "
                            "calibration)");
        return RAD_OK;
    }
    m.psk_first = first;
    (void)b;
    /* lever PSK-L20 (ds41_psk_l20): the last kv source split -- its all-row part (collapse + norm, compressor, latent /
     * index-key stores) over every row, the rest over its cone; available where no Engram block or DSpark tap sits at
     * it (they would need every row there too) and the indexer's candidate layer is at or below it (the keep a later
     * reindex layer reads is written there or earlier; its rows lie in the cone) */
    {
        const Layer& L = m.layers[(size_t)last];
        if (L.l == last && L.src && !L.e_ids && !L.tap && c.cand <= last) m.psk_l20 = last;
    }
    if (!ctx->shape_probe) {
        if (m.psk_l20 >= 0) {
            g_sw_psk_l20_ok.store(1, std::memory_order_relaxed);
            g_psk_l20_layer.store(m.psk_l20, std::memory_order_relaxed);
        }
        g_sw_psk_ok.store(1, std::memory_order_relaxed);
        /* design 2 (ds41_psk 2 / 3): the scheduler's plan per step; a step with decode rows or two prefill sequences
         * is cut in BI's segments (step_bi), which need BI's cu views (RADIANCE_DS41_BI declared) */
        g_psk_first.store(first, std::memory_order_relaxed);
        g_psk_nlayer.store((int)c.n_layer, std::memory_order_relaxed);
        g_psk_window.store((int)c.window, std::memory_order_relaxed);
        if (m.bi_sub) g_sw_psk2_ok.store(1, std::memory_order_relaxed);
    }
    /* RADIANCE_DS41_PSK=2 / 3 (the start value, whatever rank declares first): said once on rank 0; without BI's cu views
     * design 2 is not available and the switch starts at 1 */
    const int env = g_sw_psk_env.load(std::memory_order_relaxed);
    if (ctx->rank == 0 && !ctx->shape_probe && env >= 2)
        fprintf(stderr, "radiance: deepseek41: lever PSK design 2 (anchors, RADIANCE_DS41_PSK=%d): layers %d..%lld run only "
                        "the cones of the scheduler's targets (the prompt's end, its tail snapshot, every ds41_psk_k-th "
                        "checkpoint position%s, a request's checkpoint_at); %s\n",
                env, first, (long long)c.n_layer - 1, env == 3 ? ", the first at or past 32768" : "",
                m.bi_sub ? "window snapshots at targets only" : "NOT available: RADIANCE_DS41_BI is not declared -- starts at 1");
    if (!ctx->shape_probe && !m.bi_sub && g_sw_psk.load(std::memory_order_relaxed) >= 2)
        g_sw_psk.store(1, std::memory_order_relaxed);
    /* lever PSK-CR rides on design 2's plan: no BI cu views, no replay */
    if (!ctx->shape_probe && !m.bi_sub && g_sw_psk_cr.load(std::memory_order_relaxed) >= 1)
        g_sw_psk_cr.store(0, std::memory_order_relaxed);
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now %d): a prefill "
                        "sequence's chunk runs layers %d..%lld over its last %lld + (%lld - l) x %lld rows only (layer %d: "
                        "%lld, layer %lld: %lld); every row through layer %d (the last kv source)\n",
                g_sw_psk.load(std::memory_order_relaxed), first, (long long)c.n_layer - 1, (long long)kPskBase,
                (long long)c.n_layer - 1, (long long)c.window, first,
                (long long)(kPskBase + (c.n_layer - 1 - first) * c.window), (long long)c.n_layer - 1, (long long)kPskBase,
                last);
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s\n",
                g_sw_psk_cr.load(std::memory_order_relaxed),
                m.bi_sub ? ("a planned prefix hit with no snapshot at its grid point H resumes there by a cone replay "
                            "from H - " + std::to_string(kPskBase + (c.n_layer - 1) * c.window) + ": every layer over "
                            "H's cone, no global-KV store; ds41_psk 4 = tail-only anchors").c_str()
                          : "NOT available: RADIANCE_DS41_BI is not declared");
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s\n",
                g_sw_psk_l20.load(std::memory_order_relaxed),
                m.psk_l20 >= 0 ? ("layer " + std::to_string(m.psk_l20) + " split under lever PSK -- its collapse + norm, "
                                  "compressor and latent / index-key stores over every row, its query, indexer, attention, "
                                  "MoE and window store over its last " +
                                  std::to_string(kPskBase + (c.n_layer - 1 - m.psk_l20) * c.window) + " rows").c_str()
                               : "NOT available (an Engram block or a DSpark tap at the last kv source, or the candidate "
                                 "layer above it)");
    return RAD_OK;
}

/* Lever E1 at the end of a declare (RADIANCE_DS41_ENGDD set): every Engram layer declared its deduplicated ops (layer():
 * a GGUF-block kv weight), else the lever is not available here (1 refused, the switch reads 0). */
static void engdd_declare(Model& m, const RadBuildCtx* ctx) {
    if (!engdd_declared() || ctx->shape_probe) return;
    int n = 0, want = 0;
    for (const Layer& L : m.layers) {
        if (!L.e_ids) continue;
        ++want;
        n += L.e_emb_dd && L.e_gemm_u && L.e_scat && L.edd_ids && L.edd_map ? 1 : 0;
    }
    const bool ok = want > 0 && n == want && m.ekv_u;
    if (!ok) {
        g_sw_engdd.store(0, std::memory_order_relaxed);
        if (ctx->rank == 0)
            fprintf(stderr, "radiance: deepseek41: lever E1 not declared: %d of %d Engram layers have a GGUF-block kv weight "
                            "(gemm_nt_ggml) -- the switch stays 0\n", n, want);
        return;
    }
    g_sw_engdd_ok.store(1, std::memory_order_relaxed);
    if (ctx->rank == 0)
        fprintf(stderr, "radiance: deepseek41: lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now %d): the Engram "
                        "lookup / copy / kv GEMM (%lld x %lld) of a step with a prefill sequence over its unique n-gram id "
                        "rows at %d Engram layer(s), then gather_rows back; launches of <= %lld rows as today, and none "
                        "when the unique rows would be <= %lld or fewer than 1/%lld of the pooled rows repeat; such a pass "
                        "is not recorded\n",
                g_sw_engdd.load(std::memory_order_relaxed), (long long)((m.c.hc + 1) * m.c.n_embd),
                (long long)(m.c.ple_heads * m.c.ple_dim), want, (long long)ds41_engdd::kFormRows,
                (long long)ds41_engdd::kFormRows, (long long)ds41_engdd::kMinShareDiv);
}

/* Lever KVR (ds41-kvr, 12.11.2026): RADIANCE_DS41_KV_REPLICATED=1 declares the prefix-cached groups -- kv_index_r* and
 * kv_compressed_r*, the only groups an entry holds (the window and the DSpark ring are windowed, the compressor tail is
 * linear state in the checkpoint store) -- rad_kv_group_replicated, so a host-tier slot carries them once, at its front,
 * written by rank 0 and read back by every rank (core/runtime/tierexec.h, TierExec::slot_layout). Storage only: no op,
 * buffer or kernel changes, and every rank still computes and stores its own copy. The claim holds because every rank
 * writes the same bytes into these groups: the stores (the compressor GEMM, the pool / norm, the index-key GEMM, the
 * norm / rope / round trips and entry_store, declare at :1971-2031, issued over the step's whole compressed rows on every
 * rank in both step paths) read only the normed residual stream, which the exact wire (g.wire_exact, :1371: every word
 * folded in fp32 in ascending rank order whoever reduces it) leaves bit-identical on every rank, through replicated
 * weights (lw's RAD_SHARD_NONE); lever PSK cuts only layers past the last kv source; the packed forms write every byte
 * of an entry (the mxfp4 / nvfp4 zero pad). RADIANCE_DEBUG_TIER_VERIFY checks it on a live server. Declared only when no
 * rank is attention-zero (aw == world, :1323-1330): a zero rank holds none of these groups, and a replicated group must
 * be one every rank holds alike. A declare-time layout (the slot's size and the host arena are fixed at bringup), so a
 * start-time env, not a fn4 switch. Unset or 0 declares nothing: today's slots, byte for byte. */
static std::atomic<int> g_kvr_env{-2};   /* -2 unread, -1 unset or not understood, 0 off, 1 on */
static int kvr_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_KV_REPLICATED");
    const int v = !e || !*e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                                  : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_kvr_env.compare_exchange_strong(unread, v) && e && *e && v < 0)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_KV_REPLICATED=%s is not 0 / off or 1 / on: lever KVR is not "
                        "declared\n", e);
    return g_kvr_env.load(std::memory_order_relaxed);
}
static int kvr_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (kvr_read_env() != 1) return RAD_OK;
    const bool say = ctx->rank == 0 && !ctx->shape_probe;
    if (m.g.world <= 1 || m.aw != m.g.world) {
        if (say)
            fprintf(stderr, "radiance: deepseek41: lever KVR not declared (RADIANCE_DS41_KV_REPLICATED=1): %s\n",
                    m.g.world <= 1 ? "one rank holds every slot alone"
                                   : "attention-zero ranks hold none of the cached groups, so no group is alike on every rank");
        return RAD_OK;
    }
    std::string names;
    for (const auto* groups : { &m.index, &m.comp })
        for (const auto& kv : *groups) {
            DS_TRY(rad_kv_group_replicated(b, kv.second));
            names += (names.empty() ? "" : ", ") +
                     std::string(m.nm.f(groups == &m.index ? "kv_index_r%lld" : "kv_compressed_r%lld", (long long)kv.first));
        }
    if (!ctx->shape_probe)
        rad_note(b, "lever KVR: %s replicated on every rank (RADIANCE_DS41_KV_REPLICATED=1): a host-tier slot holds them once",
                 names.c_str());
    if (say)
        fprintf(stderr, "radiance: deepseek41: lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1): %s hold the same bytes on "
                        "all %d ranks (rad_kv_group_replicated), so a host-tier slot holds them once instead of %d times; "
                        "start-time only\n",
                names.c_str(), m.g.world, m.g.world);
    return RAD_OK;
}

static int bt_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (!bt_dir()) return RAD_OK;
    auto& t = m.bt;
    t.on = true;
    t.dir = bt_dir();
    if (const char* e = std::getenv("RADIANCE_DS41_BITRACE_PMAX")) t.pmax = std::max<int64_t>(1, std::atoll(e));
    if (const char* e = std::getenv("RADIANCE_DS41_BITRACE_PMIN")) t.pmin = std::max<int64_t>(0, std::atoll(e));
    if (const char* e = std::getenv("RADIANCE_DS41_BITRACE_LMAX")) t.lmax = std::max(0, std::atoi(e));
    t.cap = 64 + 24 * (int64_t)m.c.n_layer;
    const int64_t MT = m.g.max_tok;
    t.plane = decl_b(b, m.nm.f("bi_trace_plane"), RAD_I32, { t.cap * MT, 2 }, RAD_BUF_PERSIST);
    if (!t.plane) return RAD_E_INVAL;
    auto add = [&](rad_buf buf, int64_t rows) {
        if (!buf || t.op.count(buf)) return;
        t.op[buf] = op(b, "bi_rowhash", { RAD_RANGE("M", 1, rows) }, {}, { buf }, { t.plane });
    };
    for (rad_buf buf : { m.h0, m.h1, m.xn, m.qr, m.q, m.kvn, m.iq, m.iw, m.sel, m.o, m.oa, m.attn_out, m.act.x,
                         m.mw.logits, m.mw.ids, m.mw.ew, m.mw.sout, m.final_hidden })
        add(buf, MT);
    add(m.logits, m.g.max_logit_rows);
    for (const auto& kv : t.op)
        if (!kv.second) return RAD_E_UNSUPPORTED;
    if (ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever BI's tracer ON (RADIANCE_DS41_BITRACE=%s): %zu traced buffers, %lld "
                        "points a step at most, rows at positions %lld..%lld, layers below %d; passes not recorded\n",
                t.dir.c_str(), t.op.size(), (long long)t.cap, (long long)t.pmin, (long long)t.pmax - 1, t.lmax);
    return RAD_OK;
}
/* One point: `rows` rows of `buf` as the last op left them. */
static void bt_pt(RadCtx* x, Model& m, rad_buf buf, int64_t rows, const char* name, int l, int kind = 0) {
    auto& t = m.bt;
    if (!t.on || !buf || rows <= 0 || t.n >= t.cap || l >= t.lmax) return;
    const auto it = t.op.find(buf);
    if (it == t.op.end() || !it->second) return;
    RAD_ISSUE_N(x, it->second, rows, brows(buf, rows), brow_slice(t.plane, (int64_t)t.n * m.g.max_tok, rows, 2));
    char nm[64];
    if (l >= 0) snprintf(nm, sizeof nm, "L%02d.%s", l, name);
    else snprintf(nm, sizeof nm, "%s", name);
    t.names.emplace_back(nm);
    t.kind.push_back((uint8_t)kind);
    ++t.n;
}
static void bt_begin(Model& m) {
    if (!m.bt.on) return;
    m.bt.n = 0;
    m.bt.names.clear();
    m.bt.kind.clear();
}
/* The step's end: the plane and the batch's rows to the host (one synchronisation), the selected rows to the file. */
static void bt_flush(RadCtx* x, Model& m, const RadBatch* batch) {
    auto& t = m.bt;
    if (!t.on || t.n == 0) return;
    const int64_t T = batch->n_tok, S = batch->n_seq, NO = batch->n_out > 0 ? batch->n_out : 0, MT = m.g.max_tok;
    const RadStream st = rad_stream(x);
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
    t.host.resize((size_t)t.n * (size_t)MT * 2);
    cp(t.host.data(), rad_buf_ptr(x, t.plane), (int64_t)t.n * MT * 8);
    rad_stream_sync(st);
    const int rank = rad_rank(x);
    if (!t.f) {
        char path[1024];
        snprintf(path, sizeof path, "%s/bt-r%d-%d.bin", t.dir.c_str(), rank, (int)getpid());
        t.f = fopen(path, "ab");
        snprintf(path, sizeof path, "%s/bt-r%d-%d.names", t.dir.c_str(), rank, (int)getpid());
        t.fn = fopen(path, "a");
        if (!t.f || !t.fn) {
            fprintf(stderr, "radiance: deepseek41: lever BI's tracer cannot open %s: off\n", path);
            t.on = false;
            return;
        }
    }
    uint64_t key = 1469598103934665603ull;
    for (size_t i = 0; i < t.names.size(); ++i) {
        for (const char c : t.names[i]) key = (key ^ (uint8_t)c) * 1099511628211ull;
        key = (key ^ (uint64_t)(t.kind[i] + 1)) * 1099511628211ull;
    }
    if (key != t.names_key) {
        fprintf(t.fn, "N %016llx %d\n", (unsigned long long)key, t.n);
        for (int i = 0; i < t.n; ++i) fprintf(t.fn, "%s %d\n", t.names[(size_t)i].c_str(), (int)t.kind[(size_t)i]);
        fflush(t.fn);
        t.names_key = key;
    }
    std::vector<int32_t> sel;
    for (int64_t r = 0; r < T; ++r)
        if (pos[(size_t)r] >= t.pmin && pos[(size_t)r] < t.pmax) sel.push_back((int32_t)r);
    int64_t seqdec = 0, tokdec = 0;
    batch_split(batch, &seqdec, &tokdec);
    struct {
        char magic[4];
        int32_t rank;
        int64_t step;
        int32_t T, S, NO, NP, nsel, DT, SD, draft;
        uint64_t names;
    } h{ { 'B', 'I', 'T', '2' }, rank, (int64_t)batch->step, (int32_t)T, (int32_t)S, (int32_t)NO, t.n,
         (int32_t)sel.size(), (int32_t)tokdec, (int32_t)seqdec, (int32_t)batch->n_spec, key };
    fwrite(&h, sizeof h, 1, t.f);
    fwrite(cu.data(), 4, cu.size(), t.f);
    fwrite(sid.data(), 4, sid.size(), t.f);
    fwrite(ql.data(), 4, ql.size(), t.f);
    fwrite(cl.data(), 4, cl.size(), t.f);
    fwrite(oid.data(), 4, oid.size(), t.f);
    std::vector<uint32_t> rec;
    for (const int32_t r : sel) {
        int32_t sq = 0;
        while (sq + 1 <= S && cu[(size_t)sq + 1] <= r) ++sq;
        const int32_t hd[4] = { r, pos[(size_t)r], tok[(size_t)r], sq };
        fwrite(hd, 4, 4, t.f);
        rec.assign((size_t)t.n * 2, 0);
        int32_t orow = -1;
        for (int64_t j = 0; j < NO; ++j)
            if (oid[(size_t)j] == r) orow = (int32_t)j;
        for (int i = 0; i < t.n; ++i) {
            const int64_t row = t.kind[(size_t)i] ? orow : r;
            if (row < 0) continue;
            const size_t at = ((size_t)i * (size_t)MT + (size_t)row) * 2;
            rec[(size_t)i * 2] = t.host[at];
            rec[(size_t)i * 2 + 1] = t.host[at + 1];
        }
        fwrite(rec.data(), 4, rec.size(), t.f);
    }
    fflush(t.f);
}

/* ------------------------------------------------------------------ step */
/* The drafter's context pass (draft_pass < 0): the trunk step's rows, in order. */
static void dsp_context(RadCtx* x, Model& m, const RadBatch* batch) {
    const auto& d = m.dsp;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, hd = m.c.head_dim;
    if (T <= 0 || m.zero) return;   /* an attention-zero rank keeps no keys (its cache is zeroed) */
    const auto* kb = kv_batch(batch, d.kv);
    if (!kb || !kb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the DSpark cache has no slots on a context pass");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    /* Lever PSK-DS (dspark_pfctx 1, ds41-pskl20): only the rows a drafter can still read -- each entry's last
     * c.window + 1 -- in runs of >= kPskBase rows (pfctx_plan; rad_arch_pass_key leaves such a pass unrecorded). Every
     * run's launches are in the band today's one launch over T (>= kPskBase) rows was: gemm_nt_ggml's WMMA form (> 48
     * rows), rmsnorm's fixed 256-lane partition, per-row rope / fp8 round trip / window store; so a computed row's
     * context key is today's byte for byte, and the rows not run are keys no query can reach (dsp_context's note above
     * pfctx_read_env). The angle table is written over every row as today (the trunk's same angles, rope_a). */
    {
        static thread_local std::vector<std::pair<int64_t, int64_t>> pr;
        int64_t kept = T;
        if (pfctx_plan(batch, m.c.window, &pr, &kept)) {
            if (rad_rank(x) == 0) {
                const long long np = g_pfctx_passes.fetch_add(1, std::memory_order_relaxed) + 1;
                g_pfctx_skipped.fetch_add(T - kept, std::memory_order_relaxed);
                g_pfctx_run.fetch_add(kept, std::memory_order_relaxed);
                if (np == 1 || np % 4096 == 0) {
                    std::string rs;
                    for (const auto& u : pr)
                        rs += std::string(rs.empty() ? "" : ", ") + std::to_string(u.first) + "+" + std::to_string(u.second);
                    fprintf(stderr, "radiance: deepseek41: lever PSK-DS: %lld context passes cut, %lld rows not run (this "
                                    "pass: %lld rows, %lld entries -> runs [%s])\n", np,
                            g_pfctx_skipped.load(std::memory_order_relaxed), (long long)T, (long long)batch->n_seq, rs.c_str());
                }
            }
            const int64_t D = m.c.n_embd, NT = (int64_t)d.taps.size() * D;
            auto S = [&](rad_buf h, int64_t pitch, int64_t r0, int64_t n) {
                return r0 ? brow_slice(h, r0, n, pitch) : brows(h, n);
            };
            RAD_ISSUE_N(x, d.c_tab, T, pos, brows(m.cs_a, g.max_ctx));
            for (const auto& u : pr) lin_issue(x, m, d.c_fc, u.second, S(d.src, NT, u.first, u.second), d.fc, S(d.c, D, u.first, u.second));
            for (const auto& u : pr)
                RAD_ISSUE_N(x, d.c_norm, u.second, S(d.c, D, u.first, u.second), RAD_W(d.main_norm), S(d.cn, D, u.first, u.second));
            for (const auto& L : d.st) {
                for (const auto& u : pr)
                    lin_issue(x, m, L.d_kv, u.second, S(d.cn, D, u.first, u.second), L.wkv, S(d.ckv, hd, u.first, u.second));
                for (const auto& u : pr)
                    RAD_ISSUE_N(x, L.d_kvn, u.second, S(d.ckv, hd, u.first, u.second), RAD_W(L.kv_norm), S(d.ckvn, hd, u.first, u.second));
                for (const auto& u : pr)
                    RAD_ISSUE_N(x, L.d_kvrope, u.second, S(d.ckvn, hd, u.first, u.second), brows(m.cs_a, g.max_ctx),
                                praw(batch->positions + u.first, RAD_I32, u.second));
                if (L.d_kvq)
                    for (const auto& u : pr) RAD_ISSUE_N(x, L.d_kvq, u.second, S(d.ckvn, hd, u.first, u.second));
                for (const auto& u : pr) {
                    const RadOperand sl = praw(kb->slot_mapping + u.first, RAD_I32, u.second);
                    if (m.w8d)
                        RAD_ISSUE_N(x, L.d_kvst, u.second, S(d.ckvn, hd, u.first, u.second), sl, kv_cache(d.kv, L.l));
                    else
                        RAD_ISSUE_N(x, L.d_kvst, u.second, bcol_at(d.ckvn, u.first, hd, 0, hd / 2, u.second),
                                    bcol_at(d.ckvn, u.first, hd, hd / 2, hd / 2, u.second), sl, kv_cache(d.kv, L.l));
                }
            }
            return;
        }
    }
    RAD_ISSUE_N(x, d.c_tab, T, pos, brows(m.cs_a, g.max_ctx));
    lin_issue(x, m, d.c_fc, T, brows(d.src, T), d.fc, brows(d.c, T));
    RAD_ISSUE_N(x, d.c_norm, T, brows(d.c, T), RAD_W(d.main_norm), brows(d.cn, T));
    for (const auto& L : d.st) {
        lin_issue(x, m, L.d_kv, T, brows(d.cn, T), L.wkv, brows(d.ckv, T));
        RAD_ISSUE_N(x, L.d_kvn, T, brows(d.ckv, T), RAD_W(L.kv_norm), brows(d.ckvn, T));
        RAD_ISSUE_N(x, L.d_kvrope, T, brows(d.ckvn, T), brows(m.cs_a, g.max_ctx), pos);
        if (L.d_kvq) RAD_ISSUE_N(x, L.d_kvq, T, brows(d.ckvn, T));
        if (m.w8d)   /* window_store: the row packed into its slot (lever KVWIN) */
            RAD_ISSUE_N(x, L.d_kvst, T, brows(d.ckvn, T), praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
        else
            RAD_ISSUE_N(x, L.d_kvst, T, bcol(d.ckvn, 0, hd / 2, T), bcol(d.ckvn, hd / 2, hd / 2, T),
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
    }
}

/* The drafter's query pass (draft_pass > 0): `block` rows a drafting sequence, the anchor first. */
static void dsp_query(RadCtx* x, Model& m, const RadBatch* batch) {
    auto& d = m.dsp;   /* cw_up moves once (the confidence head's weights) */
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    if (T <= 0 || S <= 0) return;
    if (T != S * d.block || T > d.Q) {
        rad_step_fail(x, "deepseek41: a DSpark query pass staged at a block this drafter has no ops for");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* kb = kv_batch(batch, d.kv);
    if (!m.zero && (!kb || !kb->slot_mapping || !kb->block_table || !kb->seqused)) {
        rad_step_fail(x, "deepseek41: the DSpark cache has no block table on a query pass");
        return;
    }

    RAD_ISSUE_N(x, d.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(d.x, T));
    if (d.embed_ar) RAD_ISSUE_N(x, d.embed_ar, T * D, brows(d.x, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i) RAD_ISSUE_N(x, d.expand, T, brows(d.x, T), bcol(d.h0, i * D, D, T));
    RAD_ISSUE_N(x, d.pre0, T, brows(d.pre_f, T));
    RAD_ISSUE_N(x, d.pre1, T, bcol(d.pre_f, 0, 1, T));
    RAD_ISSUE_N(x, d.q_tab, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, d.poff_op, T, brows(d.poff, T));
    RAD_ISSUE_N(x, d.apos_op, T, pos, brows(d.poff, T), brows(d.apos, T));

    for (const auto& L : d.st) {
        RAD_ISSUE_N(x, L.mix_a, T, brows(d.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale), RAD_W(L.hc_a_base),
                    brows(d.pre_a, T), brows(d.post, T), brows(d.comb, T));
        RAD_ISSUE_N(x, L.col_a, T, brows(d.h0, T), brows(d.pre_f, T), brows(d.x, T));
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(d.attn_out, T));
        } else {
            RAD_ISSUE_N(x, L.norm_a, T, brows(d.x, T), RAD_W(L.attn_norm), brows(d.xn, T));
            lin_issue(x, m, L.q_a, T, brows(d.xn, T), L.wq_a, brows(d.qr, T));
            RAD_ISSUE_N(x, L.q_an, T, brows(d.qr, T), RAD_W(L.q_norm), brows(d.qrn, T));
            lin_issue(x, m, L.q_b, T, brows(d.qrn, T), L.wq_b, brows(d.q, T));
            RAD_ISSUE_N(x, L.q_rope, T, brows(d.q, T), brows(m.cs_a, g.max_ctx), pos);
            lin_issue(x, m, L.kv_g, T, brows(d.xn, T), L.wkv, brows(d.kv_, T));
            RAD_ISSUE_N(x, L.kv_n, T, brows(d.kv_, T), RAD_W(L.kv_norm), brows(d.kvn, T));
            RAD_ISSUE_N(x, L.kv_rope, T, brows(d.kvn, T), brows(m.cs_a, g.max_ctx), pos);
            if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(d.kvn, T));
            /* The block's keys past the committed end; the next context pass overwrites them. */
            if (m.w8d)
                RAD_ISSUE_N(x, L.kv_st, T, brows(d.kvn, T), praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
            else
                RAD_ISSUE_N(x, L.kv_st, T, bcol(d.kvn, 0, hd / 2, T), bcol(d.kvn, hd / 2, hd / 2, T),
                            praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
            RAD_ISSUE_N(x, L.attn, T, brows(d.q, T), kv_cache(d.kv, L.l),
                        praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch), praw(kb->seqused, RAD_I32, S),
                        RAD_NONE, RAD_NONE, RAD_NONE, brows(d.apos, T), cu, RAD_W(L.sinks), brows(d.o, T));
            RAD_ISSUE_N(x, L.o_rope, T, brows(d.o, T), brows(m.cs_a, g.max_ctx), pos);
            if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(d.o, T));
            const int64_t gd = HD / m.gr;
            for (int j = 0; j < m.gr; ++j)
                lin_issue(x, m, L.o_a[(size_t)j], T, bcol(d.o, j * gd, gd, T), L.wo_a[(size_t)j],
                          bcol(d.oa, j * c.o_rank, c.o_rank, T));
            lin_issue(x, m, L.o_b, T, brows(d.oa, T), L.wo_b, brows(d.attn_out, T));
        }
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(d.attn_out, T), RAD_NONE);
        RAD_ISSUE_N(x, L.post_a, T, brows(d.attn_out, T), brows(d.h0, T), brows(d.post, T), brows(d.comb, T),
                    brows(d.h1, T));
        RAD_ISSUE_N(x, L.mix_f, T, brows(d.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                    brows(d.pre_f, T), brows(d.post, T), brows(d.comb, T));
        RAD_ISSUE_N(x, L.col_f, T, brows(d.h1, T), brows(d.pre_a, T), brows(d.x, T));
        RAD_ISSUE_N(x, L.norm_f, T, brows(d.x, T), RAD_W(L.ffn_norm), brows(d.act.x, T));
        L.moe.step(x, batch);
        RAD_ISSUE_N(x, L.post_f, T, brows(d.act.x, T), brows(d.h1, T), brows(d.post, T), brows(d.comb, T),
                    brows(d.h0, T));
    }

    RAD_ISSUE_N(x, d.contract, T, brows(d.h0, T), brows(d.pre_f, T), brows(d.x, T));
    RAD_ISSUE_N(x, d.fnorm, T, brows(d.x, T), RAD_W(d.norm), brows(d.fin, T));
    const int64_t N = batch->n_out;
    if (N <= 0) return;
    RAD_ISSUE_N(x, d.gather, N, brows(d.fin, T), praw(batch->out_ids, RAD_I32, N), brows(d.rows, N));
    RAD_ISSUE_N(x, d.head, N, brows(d.rows, N), RAD_W(m.head), brows(d.logits, N));
    /* Lever W1: the full-vocabulary walk while the dspark_walk switch is 1 (declared: RADIANCE_DS41_DSPARK_WALK).
     * ds41-int3: the walk no longer returns before the DSpark extras below -- the confidence head (dspark_conf, rank 0)
     * reads the walk's proposals in d.tokens whichever walk wrote them, so with dspark_walk >= 1 and dspark_conf on the
     * plane the engine cuts by is this pass's. dspark_alt continues the top-K walk's candidates (cand_ / unary), which
     * the full-vocabulary walk does not compute: it is not issued while that walk runs. Every pass with the walk off,
     * or with the walk on and no extra switched on, issues exactly what it issued before. */
    const int wsw = g_sw_walk.load(std::memory_order_relaxed);
    /* Lever SD: the sampled walk while the engine's draft_sample is >= 1 (declared under --draft-coupling on). It
     * replaces whichever walk dspark_walk names (a greedy row drafts W1's token); the engine stages the draws and the
     * q_temp column before this pass and reads the states back behind it. */
    const bool sampled = !d.sstep.empty() && ds41_switch("draft_sample") >= 1 && N == S * d.block && S <= d.sseqs;
    const bool walked = sampled || ((d.wfill || d.wmarkov) && wsw >= 1 && N == S * d.block && S <= d.wseqs);
    if (sampled) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        const int64_t SS = d.sseqs, SR = 16, GW = SR * g.world;
        RAD_ISSUE_N(x, d.sfill, d.block, brows(d.scnt, d.block));
        for (int64_t l = 0; l <= d.block; ++l) {
            const bool last = l == d.block;
            RAD_ISSUE_N(x, d.sstep[(size_t)l], S, brows(d.logits, N), anchor,
                        l ? brow_slice(d.sgath, (l - 1) * SS, S, GW) : RAD_NONE, RAD_W(d.mk1), RAD_W(d.mk2),
                        last ? RAD_NONE : brow_slice(d.skeys, l * SS, S, SR), brows(d.tokens, S),
                        last ? RAD_NONE : brows(d.sscr, S * d.snblk), last ? RAD_NONE : brow_slice(d.scnt, l, 1, 4),
                        praw(batch->seq_ids, RAD_I32, S), brows(d.sdraw, S), brows(d.sqt, S), RAD_B(d.sstate));
            if (!last) RAD_ISSUE(x, d.sag, brow_slice(d.skeys, l * SS, S, SR), brow_slice(d.sgath, l * SS, S, GW));
        }
    } else if (walked && d.wmarkov) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        RAD_ISSUE_N(x, d.wmarkov, S, brows(d.logits, N), anchor, RAD_W(d.mk1), RAD_W(d.mk2), brows(d.wts, S),
                    brows(d.wtt, S), brows(d.tokens, S));
    } else if (walked) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        const int64_t WS = d.wseqs, KC = ds41walk::keys_cols(), GW = ds41walk::gath_cols(g.world);
        const int64_t KR = ds41walk::plane_rows(d.block, WS);
        RAD_ISSUE_N(x, d.wfill, KR, brows(d.wkeys, KR));
        for (int64_t l = 0; l < d.block; ++l) {
            const int64_t r0 = ds41walk::step_row(l, WS);
            RAD_ISSUE_N(x, (wsw == 2 && !d.wstep2.empty() ? d.wstep2 : d.wstep)[(size_t)l], S, brows(d.logits, N), anchor,
                        l ? brow_slice(d.wgath, ds41walk::step_row(l - 1, WS), S, GW) : RAD_NONE, RAD_W(d.mk1),
                        RAD_W(d.mk2), brow_slice(d.wkeys, r0, S, KC), brows(d.tokens, S));
            RAD_ISSUE(x, d.wag, brow_slice(d.wkeys, r0, S, KC), brow_slice(d.wgath, r0, S, GW));
        }
        RAD_ISSUE_N(x, d.wstep[(size_t)d.block], S, brows(d.logits, N), anchor,
                    brow_slice(d.wgath, ds41walk::step_row(d.block - 1, WS), S, GW), RAD_W(d.mk1), RAD_W(d.mk2),
                    RAD_NONE, brows(d.tokens, S));
    } else {
        if (d.merge) {
            RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), brows(d.pairs, N));
            RAD_ISSUE(x, d.pgather, brows(d.pairs, N), brows(d.gath, N));
            RAD_ISSUE_N(x, d.merge, N, brows(d.gath, N), brows(d.cand_, N), brows(d.unary, N));
        } else {
            RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), RAD_NONE);
        }
        RAD_ISSUE_N(x, d.select, S, brows(d.cand_, N), brows(d.unary, N), RAD_NONE, praw(batch->token_ids, RAD_I32, T),
                    RAD_W(d.mk1), RAD_W(d.mk2), brows(d.tokens, S));
    }
    /* The DSpark extras (ds41-acc, rank 0), each only while its switch is on: the switches move at a
     * step release with a new epoch, so a pass recorded without them is never played with them. */
    if (d.conf_op && ds41_switch("dspark_conf") > 0) {
        if (!d.cw_up) {   /* the sidecar's weights, once, landed before the head is issued */
            void* p = rad_buf_ptr(x, d.cwb);
            if (!p || rad_memcpy_async(p, d.cw.data(), (int64_t)d.cw.size() * 4, rad_stream(x)) < 0 ||
                rad_stream_sync(rad_stream(x)) < 0) {
                rad_step_fail(x, "deepseek41: the DSpark confidence head's weights could not be copied to the card");
                return;
            }
            d.cw_up = true;
        }
        RAD_ISSUE_N(x, d.conf_op, S, brows(d.x, T), praw(batch->token_ids, RAD_I32, T), brows(d.tokens, S),
                    RAD_W(d.mk1), brows(d.cwb, 1), brows(d.conf, S));
    }
    if (d.alt_op && !walked && ds41_switch("dspark_alt") > 0)
        RAD_ISSUE_N(x, d.alt_op, S, brows(d.cand_, N), brows(d.unary, N), praw(batch->token_ids, RAD_I32, T),
                    brows(d.tokens, S), RAD_W(d.mk1), RAD_W(d.mk2), brows(d.alt, S));
}

/* The ffn connection's mix, issued by the MoE block behind its routed gate_up (MoeFP8::Beside). */
struct MixBeside {
    const Layer* L;
    const Model* m;
    int64_t T;
};
static void mix_f_beside(RadCtx* x, void* a) {
    const MixBeside& p = *(const MixBeside*)a;
    const Layer& L = *p.L;
    const Model& m = *p.m;
    const int64_t T = p.T;
    RAD_ISSUE_N(x, L.mix_f, T, brows(m.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                RAD_W(L.hc_f_base), brows(m.pre_f, T), brows(m.post, T), brows(m.comb, T));
}

/* ------------------------------------------------------------------ lever BI: the step in segments */
/* A segment: rows [r0, r0 + n) of sequences [s0, s1); `dec` the step's decode rows (sequences [0, n_seq_decode)), else
 * ONE prefill sequence. `cu`: the segment's cu_seqlens view -- the batch's own at s0 = 0 (decode rows start the step),
 * else row `k` of bi_cu, {0, n}, written by bi_sub at the step's start. */
struct BiSeg {
    int64_t r0 = 0, n = 0, s0 = 0, s1 = 0;
    bool dec = false;
    int k = -1;   /* bi_cu's row (a prefill segment), -1 for the batch's cu */
};
static std::atomic<long long> g_bi_steps{0}, g_bi_sync{0}, g_bi_wide{0}, g_bi_moe{0};
/* The step's segments under ds41_bi; false when one launch over the step is the per-sequence arithmetic already (one
 * segment, decode rows within the norms' width): step() then runs today's issue. Prefill sequences' bounds from the
 * host mirror of positions (each sequence's rows are consecutive positions; a jump starts the next); when that does not
 * give the step's prefill sequence count (two sequences whose positions continue), cu_seqlens is read back (a sync). */
static bool bi_segments(RadCtx* x, const Model& m, const RadBatch* b, std::vector<BiSeg>& sg) {
    sg.clear();
    const int64_t T = b->n_tok, S = b->n_seq;
    int64_t Dn = 0, DT = 0;
    batch_split(b, &Dn, &DT);
    const int64_t P = S - Dn;
    if (DT > 0) sg.push_back({ 0, DT, 0, Dn, true, -1 });
    if (P == 1) sg.push_back({ DT, T - DT, Dn, S, false, -1 });
    if (P >= 2) {
        std::vector<int64_t> st{ DT };
        if (b->positions_h)
            for (int64_t r = DT + 1; r < T; ++r)
                if (b->positions_h[r] != b->positions_h[r - 1] + 1) st.push_back(r);
        if ((int64_t)st.size() != P) {
            std::vector<int32_t> cu((size_t)S + 1);
            rad_memcpy_async(cu.data(), b->cu_seqlens, (S + 1) * 4, rad_stream(x));
            rad_stream_sync(rad_stream(x));
            st.clear();
            for (int64_t s = Dn; s < S; ++s) st.push_back(cu[(size_t)s]);
            g_bi_sync.fetch_add(1, std::memory_order_relaxed);
        }
        for (int64_t i = 0; i < P; ++i) {
            const int64_t r1 = i + 1 < P ? st[(size_t)i + 1] : T;
            sg.push_back({ st[(size_t)i], r1 - st[(size_t)i], Dn + i, Dn + i + 1, false, -1 });
        }
    }
    int k = 0;
    for (BiSeg& s : sg)
        if (!s.dec) s.k = s.s0 == 0 ? -1 : k++;
    if (DT > kBiBand) g_bi_wide.fetch_add(1, std::memory_order_relaxed);   /* > 48 decode rows: one segment past the band */
    (void)m;
    const int bi = g_sw_bi.load(std::memory_order_relaxed);
    return sg.size() > 1 || (sg.size() == 1 && sg[0].dec && (sg[0].n > kBiNorm || (bi_pieces(bi) && sg[0].n > kBiMoe)));
}

/* Lever PSK: may this step's prefill segments be cut? (the switch, a declare that has it, and no pass that wants a
 * prefill row's logits beyond its last: n_out within the decode rows + one a prefill sequence) */
static bool psk_step_ok(const Model& m, const RadBatch* b, int64_t prefill_seqs, int64_t decode_rows) {
    return m.psk_first > 0 && (g_sw_psk.load(std::memory_order_relaxed) > 0 || psk_hint(b)) && b->draft_pass == 0 && !b->enc &&
           b->n_mm_rows == 0 && b->n_bidi == 0 && !b->gen && b->n_plugin_words == 0 && !b->score_only &&
           prefill_seqs > 0 && b->n_out <= decode_rows + prefill_seqs;
}

/* Lever E1: the core's rad_dev_host_wrote (core/device/device.h; the radiance binary exports it): the host wrote memory a
 * kernel issued after it reads, with no host wait between -- the next dispatch on every queue acquires at system scope.
 * Weak: a host without it (a test loading the plugin) runs on, and lever E1 then relies on the host lookup's own call. */
extern "C" __attribute__((weak)) void rad_dev_host_wrote(void);

/* Lever E1: behind the step's ngram_ids (both step paths, before the blocks), the event the host waits on before it reads
 * the ids (engdd_issue) -- on a pass lever E1 may act on only. */
static void engdd_mark(RadCtx* x, Model& m, const RadBatch* batch) {
    m.edd_rec = m.edd_synced = m.edd_counted = false;
    if (!engdd_may_act(batch)) return;
    if (!m.edd_ev && rad_event_create(&m.edd_ev) < 0) {
        m.edd_ev = nullptr;
        return;
    }
    m.edd_rec = rad_event_record(m.edd_ev, rad_stream(x)) >= 0;
}

/* Lever E1: layer L's Engram lookup, copy and kv GEMM over the step's unique id rows, when its plan deduplicates
 * (ds41_engram_dedup.h); false = nothing issued, the caller issues today's. `ls`: today's kv GEMM launches over the
 * layer's T rows. The ids are read on the host once the event behind the step's ngram_ids completed -- the point the
 * host lookup joins on today -- and the compact ids and the map are written before anything that reads them is issued. */
static bool engdd_issue(RadCtx* x, Model& m, Layer& L, const RadBatch* batch, int64_t T,
                        const std::vector<ds41_engdd::Launch>& ls) {
    using namespace ds41_engdd;
    if (!m.edd_rec || !L.e_emb_dd || !L.e_gemm_u || !L.e_scat || !L.edd_ids || !L.edd_map || !engdd_may_act(batch))
        return false;
    const auto& c = m.c;
    const int64_t H = c.ple_heads;
    if (!m.edd_synced) {
        if (rad_event_sync(m.edd_ev) < 0) {
            m.edd_rec = false;
            return false;
        }
        m.edd_synced = true;
    }
    const int32_t* ids = static_cast<const int32_t*>(rad_buf_ptr(x, L.ids));
    int32_t* cids = static_cast<int32_t*>(rad_buf_ptr(x, L.edd_ids));
    int32_t* map = static_cast<int32_t*>(rad_buf_ptr(x, L.edd_map));
    if (!ids || !cids || !map) return false;
    Plan& p = m.edd_plan;
    const Outcome o = plan(ids, T, H, ls.data(), ls.size(), p);
    const bool r0 = rad_rank(x) == 0;
    if (o != kDedup) {
        if (r0) (o == kGuard ? g_edd_guard : o == kLowShare ? g_edd_low : g_edd_nopool).fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    compact_ids(p, ids, cids);
    std::memcpy(map, p.map.data(), (size_t)T * sizeof(int32_t));
    if (rad_dev_host_wrote) rad_dev_host_wrote();
    auto RS = [&](rad_buf h, int64_t a, int64_t n) { return a == 0 ? brows(h, n) : brow_slice(h, a, n, m.bp.at(h)); };
    /* the lookup over the compact rows (the kept launches' rows, then the unique rows), their copy to the card */
    if (m.ple_group > 0)
        RAD_ISSUE_N(x, L.e_emb_dd, p.rows * H, brows(L.edd_ids, p.rows), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                    m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, p.rows));
    else
        RAD_ISSUE_N(x, L.e_emb_dd, p.rows * H, brows(L.edd_ids, p.rows), RAD_W(m.ple_table), brows(m.emb, p.rows));
    RAD_ISSUE_N(x, L.e_copy, p.rows, brows(m.emb, p.rows), brows(m.embd, p.rows));
    /* the kept launches as today (their own rows: the same form), into their step rows */
    for (const Kept& k : p.kept) lin_issue(x, m, L.e_gemm, k.n, RS(m.embd, k.c0, k.n), L.e_kv, RS(m.ekv, k.r0, k.n));
    /* the unique rows, one launch of > 48 rows, then each pooled run gathered from them */
    lin_issue(x, m, L.e_gemm_u, p.uniq, RS(m.embd, p.u0, p.uniq), L.e_kv, brows(m.ekv_u, p.uniq));
    for (const auto& u : p.runs)
        RAD_ISSUE_N(x, L.e_scat, u.second, brows(m.ekv_u, p.uniq), RS(L.edd_map, u.first, u.second),
                    RS(m.ekv, u.first, u.second));
    if (r0) {
        g_edd_passes.fetch_add(1, std::memory_order_relaxed);
        g_edd_rows.fetch_add(p.pooled - p.uniq, std::memory_order_relaxed);
        g_edd_pooled.fetch_add(p.pooled, std::memory_order_relaxed);
        if (!m.edd_counted) {
            m.edd_counted = true;
            g_edd_steps.fetch_add(1, std::memory_order_relaxed);
        }
        const long long n = g_edd_passes.load(std::memory_order_relaxed);
        if (n == 1 || n % 256 == 0)
            fprintf(stderr, "radiance: deepseek41: lever E1: %lld Engram layer passes deduplicated (%lld steps), %lld of %lld "
                            "pooled rows not computed; %lld kept whole by the %lld-row guard, %lld below 1/%lld repeated, %lld "
                            "with no launch above the band (this pass: layer %d, %lld rows, %lld pooled -> %lld unique, %zu "
                            "kept launch(es), %zu gather(s))\n",
                    n, g_edd_steps.load(std::memory_order_relaxed), g_edd_rows.load(std::memory_order_relaxed),
                    g_edd_pooled.load(std::memory_order_relaxed), g_edd_guard.load(std::memory_order_relaxed),
                    (long long)kFormRows, g_edd_low.load(std::memory_order_relaxed), (long long)kMinShareDiv,
                    g_edd_nopool.load(std::memory_order_relaxed), L.l, (long long)T, (long long)p.pooled, (long long)p.uniq,
                    p.kept.size(), p.runs.size());
    }
    return true;
}

/* `psk_only`: a one-prefill-sequence step step() hands over for lever PSK alone (no BI segmenting; the BI stats skip it) */
static void step_bi(RadCtx* x, Model& m, const RadBatch* batch, const std::vector<BiSeg>& sg, bool psk_only = false) {
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    /* lever PSK (psk_step_ok): the prefill segments past m.psk_first cut to their cone, layer by layer (below) */
    int64_t psk_pre = 0, psk_dec = 0;
    for (const BiSeg& s : sg) (s.dec ? psk_dec : psk_pre) += s.dec ? s.n : 1;
    const bool psk = psk_step_ok(m, batch, psk_pre, psk_dec);
    /* design 2: the scheduler's plan for this pass, its entries matched to the prefill segments in order */
    const PskPlan* hint = psk_hint(batch);
    if (hint) {
        bool ok = psk && (int64_t)hint->ahead.size() == psk_pre && (int64_t)hint->rows.size() == psk_pre;
        size_t i = 0;
        for (const BiSeg& s : sg)
            if (!s.dec) { ok = ok && i < hint->rows.size() && (int64_t)hint->rows[i] == s.n; ++i; }
        if (!ok) {
            /* a plan this pass cannot follow (a core bug, or a pass PSK does not cut): every row runs, said once */
            if (rad_rank(x) == 0 && g_psk_hbad.fetch_add(1, std::memory_order_relaxed) == 0)
                fprintf(stderr, "radiance: deepseek41: lever PSK design 2: VIRHE step %d's plan (%zu prefill sequences) "
                                "does not match the pass (%lld prefill segments%s) -- every row runs\n", batch->step,
                        hint->ahead.size(), (long long)psk_pre, psk ? "" : ", a pass PSK does not cut");
            hint = nullptr;
        } else if (rad_rank(x) == 0) {
            g_psk_hsteps.fetch_add(1, std::memory_order_relaxed);
        }
    }
    /* lever PSK-CR: the prefill segments that are cone-replay chunks (the plan's flags, in the step's order), and whether
     * the pass is replay chunks only (no decode row, every prefill segment replays): then no row of it stores global KV,
     * and the kv sources' compressor chain is skipped whole. Beside other rows the chain runs over the step as ever and
     * the batch's -1 slots keep the replay rows' stores off the cache. */
    static thread_local std::vector<uint8_t> crf;
    crf.assign((size_t)psk_pre, 0);
    bool cr_any = false, cr_all = hint != nullptr && psk_dec == 0 && psk_pre > 0;
    for (size_t i = 0; i < crf.size(); ++i) {
        crf[i] = psk_cr_flag(hint, i) ? 1 : 0;
        cr_any = cr_any || crf[i];
        cr_all = cr_all && crf[i];
    }
    cr_all = cr_all && cr_any;
    if (cr_any && rad_rank(x) == 0) {
        g_cr_passes.fetch_add(1, std::memory_order_relaxed);
        if (!cr_all) g_cr_mixed.fetch_add(1, std::memory_order_relaxed);
    }
    if (!psk_only) {   /* rank 0, every 512 segmented steps: the counts (MoE passes a layer summed over those steps: the launch cost of P) */
        const long long n = g_bi_steps.load(std::memory_order_relaxed);
        if (n > 0 && n % 512 == 0 && rad_rank(x) == 0)
            fprintf(stderr, "radiance: deepseek41: lever BI stats: %lld segmented steps, %lld MoE passes a layer in them (%.2f a "
                            "step), %lld cu read-backs, %lld steps with > 48 decode rows; ds41_bi %d\n",
                    n, g_bi_moe.load(std::memory_order_relaxed),
                    (double)g_bi_moe.load(std::memory_order_relaxed) / (double)n, g_bi_sync.load(std::memory_order_relaxed),
                    g_bi_wide.load(std::memory_order_relaxed), g_sw_bi.load(std::memory_order_relaxed));
    }
    if (!psk_only && g_bi_steps.fetch_add(1, std::memory_order_relaxed) == 0 && rad_rank(x) == 0) {
        std::string seg;
        for (const BiSeg& s : sg) seg += std::string(seg.empty() ? "" : ", ") + (s.dec ? "decode " : "prefill ") +
                                         std::to_string(s.n);
        fprintf(stderr, "radiance: deepseek41: lever BI (ds41_bi %d): the first segmented step, %lld rows as [%s]\n",
                g_sw_bi.load(std::memory_order_relaxed), (long long)T, seg.c_str());
    }
    bt_begin(m);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* wb = kv_batch(batch, m.win);
    if (!wb || !wb->block_table || !wb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the window group has no block table");
        return;
    }
    /* row slices of the per-token buffers (their declared pitch) */
    auto R = [&](rad_buf h, int64_t r0, int64_t n) {
        if (r0 == 0) return brows(h, n);
        return brow_slice(h, r0, n, m.bp.at(h));
    };
    auto C = [&](rad_buf h, int64_t first, int64_t width, int64_t r0, int64_t n) {
        return bcol_at(h, r0, m.bp.at(h), first, width, n);
    };
    auto P = [&](const int32_t* p, int64_t r0, int64_t n) { return praw(p + r0, RAD_I32, n); };
    /* lever PSK: the step's own segments (the indexer's deal at every layer, cut or not: ds41-psk-fix cf7bbc7) */
    const std::vector<BiSeg>& sg_step = sg;
    /* lever PSK: the layer's segments (`cur`: the step's, or `psk_sg` with the prefill segments cut) and the runs of
     * consecutive rows they cover (`runs`: [0, T) uncut) */
    static thread_local std::vector<BiSeg> psk_sg;
    static thread_local std::vector<std::pair<int64_t, int64_t>> runs;
    const std::vector<BiSeg>* cur = &sg;
    const auto set_runs = [&] {
        runs.clear();
        for (const BiSeg& s : *cur) {
            if (!runs.empty() && runs.back().first + runs.back().second == s.r0) runs.back().second += s.n;
            else runs.push_back({ s.r0, s.n });
        }
    };
    set_runs();
    /* fn over the layer's rows: each run of consecutive rows in one launch */
    auto whole = [&](const auto& fn) {
        for (const auto& u : runs) fn(u.first, u.second);
    };
    /* fn over the step's rows by segment, the decode segment in launches of at most `cap` rows */
    auto rows = [&](int64_t cap, const auto& fn) {
        for (const BiSeg& s : *cur) {
            if (!s.dec || s.n <= cap) { fn(s.r0, s.n); continue; }
            for (int64_t a = 0; a < s.n; a += cap) fn(s.r0 + a, std::min(cap, s.n - a));
        }
    };
    /* a segment's cu view and per-sequence slices */
    auto segcu = [&](const BiSeg& s) {
        if (s.k < 0) return praw(batch->cu_seqlens + s.s0, RAD_I32, s.s1 - s.s0 + 1);
        return brow_slice(m.bi_cu, s.k, 1, 2);
    };
    /* the prefill segments' cu views, once a step: {cu[s0] - cu[s0], cu[s0 + 1] - cu[s0]} */
    for (const BiSeg& s : sg)
        if (s.k >= 0)
            RAD_ISSUE_N(x, m.bi_sub, 2, praw(batch->cu_seqlens + s.s0, RAD_I32, 2),
                        praw(batch->cu_seqlens + s.s0, RAD_I32, 1), brow_slice(m.bi_cu, s.k, 1, 2));

    /* the step's head ops: as step() (per-token, over the step) */
    RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
    if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i)
        RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
    RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
    RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    if (m.sone && g_sw_one.load(std::memory_order_relaxed) > 0) RAD_ISSUE_N(x, m.sone, T, brows(m.mw.sgate, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap), brows(m.ctok, T));
    for (auto& L : m.layers) {
        if (!L.e_ids) continue;
        const int l = L.l;
        const auto* tk = kv_batch(batch, m.tok);
        const int32_t* tk_idx = tk ? tk->state_index : nullptr;
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        const int64_t Pn = S - Dn;
        if (Dn > 0)
            RAD_ISSUE_N(x, L.e_ids, DT, brows(m.ctok, DT), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens, RAD_I32, Dn + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off), praw(tk_idx, RAD_I32, Dn),
                        praw(batch->ctx_lens, RAD_I32, Dn),
                        praw(batch->num_accepted, RAD_I32, Dn), brows(L.ids, DT), RAD_NONE);
        if (Pn > 0)
            RAD_ISSUE_N(x, L.e_ids, T, brows(m.ctok, T), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens + Dn, RAD_I32, Pn + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off),
                        praw(tk_idx ? tk_idx + Dn : nullptr, RAD_I32, Pn),
                        praw(batch->ctx_lens + Dn, RAD_I32, Pn), RAD_NONE, brows(L.ids, T),
                        RAD_NONE);
    }
    engdd_mark(x, m, batch);   /* lever E1: the ids' event (a pass it may act on) */

    /* lever PSK-L20: the split layer's all-row part (its collapse + norm) over the STEP's segments / runs, as rows() /
     * whole() issue it in an uncut pass */
    auto rows_step = [&](int64_t cap, const auto& fn) {
        for (const BiSeg& s : sg_step) {
            if (!s.dec || s.n <= cap) { fn(s.r0, s.n); continue; }
            for (int64_t a = 0; a < s.n; a += cap) fn(s.r0 + a, std::min(cap, s.n - a));
        }
    };
    auto whole_step = [&](const auto& fn) {
        int64_t a = 0, n = 0;
        for (const BiSeg& s : sg_step) {
            if (n > 0 && a + n == s.r0) { n += s.n; continue; }
            if (n > 0) fn(a, n);
            a = s.r0;
            n = s.n;
        }
        if (n > 0) fn(a, n);
    };
    /* lever PSK-L20 (ds41_psk_l20 1, only where lever PSK cuts this pass): a planned pass takes the value its plan was
     * made under (rad_psk_plan: the stager's layer end), any other the switch's -- the same, read at one release */
    const bool l20_on = psk && m.psk_l20 >= 0 && m.psk_l20 < m.psk_first &&
                        (hint ? hint->l20 : g_sw_psk_l20.load(std::memory_order_relaxed) > 0);
    /* lever PSK-CR: an all-replay pass stores no global KV, so the split layer is a plain cut layer there (no all-row
     * part) */
    for (auto& L : m.layers) {
        const int l = L.l;
        const int64_t r = L.r;
        /* lever PSK: from m.psk_first on, each prefill segment runs its last psk_rows(C_l, ahead, n) rows -- design 1
         * (ahead 0): its last C_l; design 2: the cone of its next target, none when that does not reach the chunk.
         * Lever PSK-L20: the last kv source too, as the split layer (`l20`): the cut below is its cone part's; its all-row
         * part (the collapse + norm here, the compressor chain and the latent / index-key stores over the step's rows
         * below) runs whatever the cone, so the layer is never skipped whole */
        bool cut = false;
        const bool l20 = l20_on && l == m.psk_l20 && !cr_all;
        /* lever PSK-CR: a replay segment is cut at every layer (its rows outside the end's cone feed nothing the end
         * reads, and it stores no global KV); any other segment from m.psk_first (or the split layer) on as before */
        if (psk && (l >= m.psk_first || l20 || cr_any)) {
            const int64_t C = psk_cone(c.n_layer, c.window, l);
            /* the K/V store range (psk_kv_cone, today = C): the forward rows' window keys come from it */
            const int64_t CK = psk_kv_cone(c.n_layer, c.window, l);
            (void)CK;
            psk_sg.clear();
            size_t pi = 0;
            for (const BiSeg& s0 : sg) {
                if (s0.dec) { psk_sg.push_back(s0); continue; }
                const size_t ip = pi++;
                const bool cr_seg = ip < crf.size() && crf[ip];
                if (!(l >= m.psk_first || l20 || cr_seg)) { psk_sg.push_back(s0); continue; }   /* not cut here */
                const int64_t n = psk_rows(C, psk_ahead(hint, ip), s0.n);
                if (cr_seg && rad_rank(x) == 0) {
                    g_cr_rows_run.fetch_add(std::max<int64_t>(n, 0), std::memory_order_relaxed);
                    if (l < m.psk_first && !l20 && n < s0.n)
                        g_cr_rows_skip.fetch_add(s0.n - std::max<int64_t>(n, 0), std::memory_order_relaxed);
                }
                if (n >= s0.n) { psk_sg.push_back(s0); continue; }
                if (rad_rank(x) == 0) {
                    g_psk_rows.fetch_add(s0.n - n, std::memory_order_relaxed);
                    if (l20) g_psk_l20_rows.fetch_add(s0.n - n, std::memory_order_relaxed);
                    else if (l >= m.psk_first)
                        (l < m.psk_first + 9 ? g_psk_rows_lo : g_psk_rows_hi).fetch_add(s0.n - n, std::memory_order_relaxed);
                }
                cut = true;
                if (n <= 0) continue;   /* design 2: no row of this segment at this layer */
                BiSeg s = s0;
                s.r0 += s0.n - n;
                s.n = n;
                psk_sg.push_back(s);
            }
        }
        if (cut != (cur != &sg)) {
            cur = cut ? &psk_sg : &sg;
            set_runs();
        } else if (cut) {
            set_runs();
        }
        /* lever PSK-L20: the split layer cut (its cone part over fewer rows); none of its cone part when no segment has a
         * row there -- the all-row part still runs, every cone-part op below issues over the empty segments / runs */
        const bool l20_none = l20 && cut && cur->empty();
        if (l20 && cut && rad_rank(x) == 0) {
            g_psk_l20_passes.fetch_add(1, std::memory_order_relaxed);
            if (l20_none) g_psk_l20_empty.fetch_add(1, std::memory_order_relaxed);
        }
        /* design 2: a layer no segment runs a row of is skipped whole (and so is every layer after it: the cones shrink
         * with l) -- no op, no collective; the runs stay empty for the final norm */
        if (cut && cur->empty() && !l20) {
            if (rad_rank(x) == 0) g_psk_skips.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const std::vector<BiSeg>& sg = *cur;   /* the layer's segments, below */
        if (L.e_ids) {
            const int64_t H = c.ple_heads;
            /* lever E1: the lookup, copy and GEMM over the unique id rows of the launches below that are wider than the
             * band (the decode segment's pieces and a short prefill segment keep their own launch) */
            m.edd_ls.clear();
            rows(kBiBand, [&](int64_t r0, int64_t n) { m.edd_ls.push_back({ r0, n }); });
            if (!engdd_issue(x, m, L, batch, T, m.edd_ls)) {
                if (m.ple_group > 0)
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                                m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, T));
                else
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
                RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
                /* the Engram GEMM (gemm_nt_ggml, banded) by segment */
                rows(kBiBand, [&](int64_t r0, int64_t n) { lin_issue(x, m, L.e_gemm, n, R(m.embd, r0, n), L.e_kv, R(m.ekv, r0, n)); });
            }
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        bt_pt(x, m, m.h0, T, "h0", l);
        if (L.tap) RAD_ISSUE_N(x, L.tap, T, brows(m.h0, T), bcol(m.dsp.src, L.tap_col * D, D, T));
        const bool late = !m.zero;
        const auto mix_a = [&] {
            whole([&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.mix_a, n, R(m.h0, r0, n), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                            RAD_W(L.hc_a_base), R(m.pre_a, r0, n), R(m.post, r0, n), R(m.comb, r0, n));
            });
        };
        /* the collapse (+ norm: a fused norm, by segment within its 32-row width); lever PSK-L20: at the split layer over
         * the step's own segments, the uncut pass's launches -- xn on every row is the compressor's input */
        const auto cn_a = [&](int64_t r0, int64_t n) {
            RAD_ISSUE_N(x, L.cn_a, n, R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(L.attn_norm), R(m.xn, r0, n));
        };
        const auto col_a = [&](int64_t r0, int64_t n) {
            RAD_ISSUE_N(x, L.col_a, n, R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n));
        };
        if (L.cn_a) {
            if (l20) rows_step(kBiNorm, cn_a);
            else rows(kBiNorm, cn_a);
        } else if (L.col_a) {
            if (l20) whole_step(col_a);
            else whole(col_a);
        }
        if (!late) mix_a();
        if (m.zero) {
            whole([&](int64_t r0, int64_t n) { RAD_ISSUE_N(x, L.attn_fill, n, R(m.attn_out, r0, n)); });
        } else {
            if (L.norm_a) {
                const auto norm_a = [&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, L.norm_a, n, R(m.xcol, r0, n), RAD_W(L.attn_norm), R(m.xn, r0, n));
                };
                if (l20) whole_step(norm_a);
                else whole(norm_a);
            }
            bt_pt(x, m, m.xn, T, "xn", l);
            const rad_buf cs = r > 0 ? m.cs_b : m.cs_a;
            const bool ctail = L.src && r > 1;
            int64_t crows = T, tpitch = 1;
            RadOperand page = RAD_NONE, bpos = pos;
            const auto* cb = L.src ? kv_batch(batch, m.comp[r]) : nullptr;
            const auto* tb = ctail ? kv_batch(batch, m.tail[r]) : nullptr;
            if (L.src) {
                const auto* ib = kv_batch(batch, m.index[r]);
                if (!cb || !ib || !cb->block_table || !ib->block_table) {
                    rad_step_fail(x, "deepseek41: a compressed group has no block table");
                    return;
                }
                page = praw(cb->slot_mapping, RAD_I32, T);
                if (ctail) {
                    if (!tb || !tb->state_index) {
                        rad_step_fail(x, "deepseek41: the compressor tail has no state slots");
                        return;
                    }
                    tpitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    crows = m.ctail_full ? m.work
                                         : std::min<int64_t>(m.work, std::max<int64_t>((T + r - 1) / r + S, g.max_seqs));
                    page = brows(m.page, crows);
                    bpos = brows(m.bpos, crows);
                }
            }
            /* A1: the query, window key and index query / weight chains by segment (each row-banded op at its
             * segment's rows); their reads of the angles (cs) stay ahead of c_tab's rewrite below. */
            const int qnf = L.q_bf ? g_sw_qnf.load(std::memory_order_relaxed) : 0;
            const bool qfold = qnf == 1 || (qnf == 2 && !L.i_score);
            const bool kvfold = L.kv_fs && g_sw_kvf.load(std::memory_order_relaxed) > 0;
            const bool cheap = bi_cheap(g_sw_bi.load(std::memory_order_relaxed));
            /* the fused window-key norm over a segment (decode rows within 32) */
            const auto kvn_seg = [&](const BiSeg& s) {
                const auto kvn = [&](int64_t a0, int64_t an) {
                    if (kvfold)
                        RAD_ISSUE_N(x, L.kv_fs, an, R(m.kv, a0, an), RAD_W(L.kv_norm), R(m.kvn, a0, an),
                                    brows(cs, g.max_ctx), P(batch->positions, a0, an), P(wb->slot_mapping, a0, an),
                                    kv_cache(m.win, l));
                    else
                        RAD_ISSUE_N(x, L.kv_f, an, R(m.kv, a0, an), RAD_W(L.kv_norm), R(m.kvn, a0, an),
                                    brows(cs, g.max_ctx), P(batch->positions, a0, an));
                };
                if (s.dec && s.n > kBiNorm)
                    for (int64_t a = 0; a < s.n; a += kBiNorm) kvn(s.r0 + a, std::min(kBiNorm, s.n - a));
                else
                    kvn(s.r0, s.n);
            };
            if (cheap) {   /* C: op by op across the segments; the never-flipping per-row ops once over the step */
                for (const BiSeg& s : sg) lin_issue(x, m, L.q_a, s.n, R(m.xn, s.r0, s.n), L.wq_a, R(m.qr, s.r0, s.n));
                for (const BiSeg& s : sg) lin_issue(x, m, L.kv_g, s.n, R(m.xn, s.r0, s.n), L.wkv, R(m.kv, s.r0, s.n));
                if (L.i_score)
                    for (const BiSeg& s : sg) lin_issue(x, m, L.i_w, s.n, R(m.xn, s.r0, s.n), L.iproj, R(m.iw, s.r0, s.n));
                if (qfold)
                    for (const BiSeg& s : sg)
                        RAD_ISSUE_N(x, L.q_bf, s.n, R(m.qr, s.r0, s.n), RAD_W(L.wq_b), RAD_W(L.q_norm), R(m.qrn, s.r0, s.n),
                                    R(m.q, s.r0, s.n));
                else
                    whole([&](int64_t r0, int64_t n) {
                        RAD_ISSUE_N(x, L.q_an, n, R(m.qr, r0, n), RAD_W(L.q_norm), R(m.qrn, r0, n));
                    });
                if (kvfold || L.kv_f)
                    for (const BiSeg& s : sg) kvn_seg(s);
                else
                    whole([&](int64_t r0, int64_t n) {
                        RAD_ISSUE_N(x, L.kv_n, n, R(m.kv, r0, n), RAD_W(L.kv_norm), R(m.kvn, r0, n));
                    });
                if (L.i_score)
                    for (const BiSeg& s : sg) lin_issue(x, m, L.i_q, s.n, R(m.qrn, s.r0, s.n), L.iq_w, R(m.iq, s.r0, s.n));
                if (!qfold)
                    for (const BiSeg& s : sg) lin_issue(x, m, L.q_b, s.n, R(m.qrn, s.r0, s.n), L.wq_b, R(m.q, s.r0, s.n));
                whole([&](int64_t r0, int64_t n) {
                    const RadOperand pn = P(batch->positions, r0, n);
                    if (!L.kv_f) RAD_ISSUE_N(x, L.kv_rope, n, R(m.kvn, r0, n), brows(cs, g.max_ctx), pn);
                    if (!L.src && L.i_score) {
                        if (L.i_f) RAD_ISSUE_N(x, L.i_f, n, R(m.iq, r0, n), brows(m.cs_b, g.max_ctx), pn);
                        else RAD_ISSUE_N(x, L.i_rope, n, R(m.iq, r0, n), brows(m.cs_b, g.max_ctx), pn);
                        if (L.i_qq) RAD_ISSUE_N(x, L.i_qq, n, R(m.iq, r0, n));
                    }
                    RAD_ISSUE_N(x, L.q_rope, n, R(m.q, r0, n), brows(cs, g.max_ctx), pn);
                    if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, n, R(m.kvn, r0, n));
                    if (!kvfold) {
                        if (m.w8)
                            RAD_ISSUE_N(x, L.kv_st, n, R(m.kvn, r0, n), P(wb->slot_mapping, r0, n), kv_cache(m.win, l));
                        else
                            RAD_ISSUE_N(x, L.kv_st, n, C(m.kvn, 0, hd / 2, r0, n), C(m.kvn, hd / 2, hd / 2, r0, n),
                                        P(wb->slot_mapping, r0, n), kv_cache(m.win, l));
                    }
                });
            }
            for (const BiSeg& s : sg) {
                if (cheap) break;
                const int64_t r0 = s.r0, n = s.n;
                lin_issue(x, m, L.q_a, n, R(m.xn, r0, n), L.wq_a, R(m.qr, r0, n));
                lin_issue(x, m, L.kv_g, n, R(m.xn, r0, n), L.wkv, R(m.kv, r0, n));
                if (L.i_score) lin_issue(x, m, L.i_w, n, R(m.xn, r0, n), L.iproj, R(m.iw, r0, n));
                if (qfold)
                    RAD_ISSUE_N(x, L.q_bf, n, R(m.qr, r0, n), RAD_W(L.wq_b), RAD_W(L.q_norm), R(m.qrn, r0, n), R(m.q, r0, n));
                else
                    RAD_ISSUE_N(x, L.q_an, n, R(m.qr, r0, n), RAD_W(L.q_norm), R(m.qrn, r0, n));
                /* the window key's norm (+ rope + round trip [+ store]): a fused norm, decode rows within 32 */
                const auto kvn = [&](int64_t a0, int64_t an) {
                    if (kvfold)
                        RAD_ISSUE_N(x, L.kv_fs, an, R(m.kv, a0, an), RAD_W(L.kv_norm), R(m.kvn, a0, an),
                                    brows(cs, g.max_ctx), P(batch->positions, a0, an), P(wb->slot_mapping, a0, an),
                                    kv_cache(m.win, l));
                    else if (L.kv_f)
                        RAD_ISSUE_N(x, L.kv_f, an, R(m.kv, a0, an), RAD_W(L.kv_norm), R(m.kvn, a0, an),
                                    brows(cs, g.max_ctx), P(batch->positions, a0, an));
                    else
                        RAD_ISSUE_N(x, L.kv_n, an, R(m.kv, a0, an), RAD_W(L.kv_norm), R(m.kvn, a0, an));
                };
                if (s.dec && n > kBiNorm)
                    for (int64_t a = 0; a < n; a += kBiNorm) kvn(r0 + a, std::min(kBiNorm, n - a));
                else
                    kvn(r0, n);
                if (L.i_score) lin_issue(x, m, L.i_q, n, R(m.qrn, r0, n), L.iq_w, R(m.iq, r0, n));
                if (!qfold) lin_issue(x, m, L.q_b, n, R(m.qrn, r0, n), L.wq_b, R(m.q, r0, n));
                if (!L.kv_f) RAD_ISSUE_N(x, L.kv_rope, n, R(m.kvn, r0, n), brows(cs, g.max_ctx), P(batch->positions, r0, n));
                if (!L.src && L.i_score) {
                    if (L.i_f) RAD_ISSUE_N(x, L.i_f, n, R(m.iq, r0, n), brows(m.cs_b, g.max_ctx), P(batch->positions, r0, n));
                    else RAD_ISSUE_N(x, L.i_rope, n, R(m.iq, r0, n), brows(m.cs_b, g.max_ctx), P(batch->positions, r0, n));
                    if (L.i_qq) RAD_ISSUE_N(x, L.i_qq, n, R(m.iq, r0, n));
                }
                RAD_ISSUE_N(x, L.q_rope, n, R(m.q, r0, n), brows(cs, g.max_ctx), P(batch->positions, r0, n));
                if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, n, R(m.kvn, r0, n));
                if (!kvfold) {
                    if (m.w8)
                        RAD_ISSUE_N(x, L.kv_st, n, R(m.kvn, r0, n), P(wb->slot_mapping, r0, n), kv_cache(m.win, l));
                    else
                        RAD_ISSUE_N(x, L.kv_st, n, C(m.kvn, 0, hd / 2, r0, n), C(m.kvn, hd / 2, hd / 2, r0, n),
                                    P(wb->slot_mapping, r0, n), kv_cache(m.win, l));
                }
            }
            /* B: the compressor chain over the step, as step() (per entry; no row-banded op). Lever PSK-CR: an
             * all-replay pass stores no global KV (the cache holds every row's), so the chain is skipped whole; the
             * index query rope (A2) reads the angles rope_b_step wrote for the step's positions -- c_tab writes the
             * same rope_table bytes at the same positions (one pure function of the position, the same parameters) */
            const bool chain = L.src && !cr_all;
            if (L.src && cr_all && rad_rank(x) == 0) g_cr_chains.fetch_add(1, std::memory_order_relaxed);
            if (chain)
                RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w), brows(ctail ? m.kvs : m.pooled, T));
            if (ctail && chain)
                RAD_ISSUE_N(x, L.c_work, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                            praw2(tb->state_index, RAD_I32, S, tpitch),
                            praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), RAD_NONE,
                            brows(m.stage, crows * r), brows(m.page, crows), brows(m.bpos, crows), brows(m.nc, S));
            if (ctail && chain)
                RAD_ISSUE_N(x, L.c_pool, crows, brows(m.stage, crows * r), brows(m.pooled, crows), brows(m.page, crows));
            if (ctail && chain && L.c_tail)
                RAD_ISSUE_N(x, L.c_tail, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                            praw2(tb->state_index, RAD_I32, S, tpitch));
            if (chain) RAD_ISSUE_N(x, L.c_norm, crows, brows(m.pooled, crows), RAD_W(L.comp_norm), brows(m.lat, crows));
            if (chain) lin_issue(x, m, L.k_gemm, crows, brows(m.lat, crows), L.ik_w, brows(m.ik, crows));
            if (chain) RAD_ISSUE_N(x, L.c_tab, crows, bpos, brows(m.cs_b, g.max_ctx));
            if (chain) {
                if (L.k_f)
                    RAD_ISSUE_N(x, L.k_f, crows, brows(m.ik, crows), RAD_W(L.ik_norm), brows(m.ikn, crows),
                                brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.k_norm, crows, brows(m.ik, crows), RAD_W(L.ik_norm), brows(m.ikn, crows));
            }
            /* A2: a source layer's index query rope reads the angles c_tab rewrote */
            if (L.src && L.i_score && cheap) {
                if (L.i_f) RAD_ISSUE_N(x, L.i_f, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
                else RAD_ISSUE_N(x, L.i_rope, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
            } else if (L.src && L.i_score)
                for (const BiSeg& s : sg) {
                    if (L.i_f) RAD_ISSUE_N(x, L.i_f, s.n, R(m.iq, s.r0, s.n), brows(m.cs_b, g.max_ctx), P(batch->positions, s.r0, s.n));
                    else RAD_ISSUE_N(x, L.i_rope, s.n, R(m.iq, s.r0, s.n), brows(m.cs_b, g.max_ctx), P(batch->positions, s.r0, s.n));
                }
            if (chain) {
                if (L.l_f) RAD_ISSUE_N(x, L.l_f, crows, brows(m.lat, crows), brows(m.cs_b, g.max_ctx), bpos);
                else RAD_ISSUE_N(x, L.l_rope, crows, brows(m.lat, crows), brows(m.cs_b, g.max_ctx), bpos);
                if (!L.k_f) RAD_ISSUE_N(x, L.k_rope, crows, brows(m.ikn, crows), brows(m.cs_b, g.max_ctx), bpos);
            }
            if (L.src && L.i_score && L.i_qq && cheap)
                RAD_ISSUE_N(x, L.i_qq, T, brows(m.iq, T));
            else if (L.src && L.i_score && L.i_qq)
                for (const BiSeg& s : sg) RAD_ISSUE_N(x, L.i_qq, s.n, R(m.iq, s.r0, s.n));
            if (chain && L.l_q) RAD_ISSUE_N(x, L.l_q, crows, brows(m.lat, crows));
            if (chain && L.k_q) RAD_ISSUE_N(x, L.k_q, crows, brows(m.ikn, crows));
            if (chain) RAD_ISSUE_N(x, L.l_st, crows, brows(m.lat, crows), page, kv_cache(m.comp[r], l));
            if (chain) RAD_ISSUE_N(x, L.k_st, crows, brows(m.ikn, crows), page, kv_cache(m.index[r], l));
            /* the indexer: score / candidates / selection by segment, each segment's runs from its own first row */
            if (L.i_score) {
                const int src = c.src_of[(size_t)l];
                const auto* ib = kv_batch(batch, m.index[r]);
                const int64_t live = std::min<int64_t>(
                    g.max_ctx / r + 1, ((int64_t)batch->max_ctx_len + batch->max_q_len) / r + 1);
                const int64_t kw = keep_words(live, c.cand_block);
                const bool uses = c.cand >= 0 && c.cand < l;
                const bool split = m.isplit && m.sel_gather;
                const bool rkeep = L.i_keep && live >= m.keep_min_live;
                size_t sj = 0;
                for (size_t si = 0; si < sg.size(); ++si) {
                    /* lever PSK (ds41-psk-fix cf7bbc7): a cut prefill segment keeps the step's deal (sg_step: every
                     * uncut layer's). The candidate layer's keep rows are rank-local -- each rank writes only its own
                     * runs' rows, and keep is not gathered -- so a row's selection must run on the rank that wrote its
                     * keep, at the same run height. Runs wholly before the layer's cut rows are skipped; a run reaching
                     * them runs whole (its rows before the cut are not read at this layer). Design 2 drops a segment
                     * with no row at the layer, so the step's segment is found by its sequence (s0, dec; the order is
                     * kept). Uncut: sg[si] is sg_step[si], nothing skipped, the issue as before. */
                    while (sj < sg_step.size() && !(sg_step[sj].s0 == sg[si].s0 && sg_step[sj].dec == sg[si].dec)) ++sj;
                    const BiSeg& s = sj < sg_step.size() ? sg_step[sj] : sg[si];
                    const int64_t cut0 = sg[si].r0;
                    const Ds41IdxRuns ru = ds41_idx_runs(s.n, m.irows, live, m.aw, m.ar, split && !s.dec, m.split_min_live,
                                                         split && !s.dec && idx_balance(), (int64_t)L.i_runs.size());
                    for (int64_t p = ru.p0; p < ru.p1; ++p) {
                        const int64_t q0 = p * ru.R, qn = std::min(ru.R, s.n - q0);
                        if (s.r0 + q0 + qn <= cut0) continue;
                        RAD_ISSUE_N(x, L.i_runs[(size_t)p], ru.R, R(m.iq, s.r0, s.n), R(m.iw, s.r0, s.n),
                                    kv_cache(m.index[r], src),
                                    praw2(ib->block_table + s.s0 * ib->block_table_pitch, RAD_I32, s.s1 - s.s0,
                                          ib->block_table_pitch),
                                    P(batch->positions, s.r0, s.n), segcu(s), bcol(m.score, 0, live, ru.R),
                                    rkeep ? bcol_at(m.keep, s.r0 + q0, m.kwords, 0, kw, qn) : RAD_NONE);
                        if (L.i_cand)
                            RAD_ISSUE_N(x, L.i_cand, qn, bcol(m.score, 0, live, qn), P(batch->positions, s.r0 + q0, qn),
                                        bcol_at(m.keep, s.r0 + q0, m.kwords, 0, kw, qn));
                        RAD_ISSUE_N(x, L.i_topk, qn, bcol(m.score, 0, live, qn),
                                    uses ? bcol_at(m.keep, s.r0 + q0, m.kwords, 0, kw, qn) : RAD_NONE,
                                    P(batch->positions, s.r0 + q0, qn),
                                    ru.split ? brow_slice(m.sel_part, (p - ru.p0) * ru.R, qn, c.itopk)
                                             : brow_slice(m.sel, s.r0 + q0, qn, c.itopk));
                    }
                    if (ru.split)
                        RAD_ISSUE(x, m.sel_gather, brows(m.sel_part, ru.q * ru.R),
                                  brow_slice(m.sel, s.r0, (int64_t)m.aw * ru.q * ru.R, c.itopk));
                }
            }
            bt_pt(x, m, m.qr, T, "qr", l);
            bt_pt(x, m, m.q, T, "q", l);
            bt_pt(x, m, m.kvn, T, "kvn", l);
            if (L.i_score) {
                bt_pt(x, m, m.iq, T, "iq", l);
                bt_pt(x, m, m.iw, T, "iw", l);
                bt_pt(x, m, m.sel, T, "sel", l);
            }
            /* the attention by segment (its band at the segment's rows), its sequences' slices */
            const bool ofold = L.attn_rt && g_sw_of.load(std::memory_order_relaxed) > 0;
            const rad_op ah = ofold ? L.attn_rt : L.attn;
            const RadOperand acs = ofold ? brows(cs, g.max_ctx) : RAD_NONE;
            for (const BiSeg& s : sg) {
                const int64_t ns = s.s1 - s.s0;
                const RadOperand wbt = praw2(wb->block_table + s.s0 * wb->block_table_pitch, RAD_I32, ns, wb->block_table_pitch);
                const RadOperand wused = praw(wb->seqused + s.s0, RAD_I32, ns);
                if (r > 0) {
                    const int src = c.src_of[(size_t)l];
                    const auto* cbb = kv_batch(batch, m.comp[r]);
                    const RadOperand cbt = praw2(cbb->block_table + s.s0 * cbb->block_table_pitch, RAD_I32, ns,
                                                 cbb->block_table_pitch);
                    if (ofold)
                        RAD_ISSUE_N(x, ah, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, kv_cache(m.comp[r], src),
                                    cbt, R(m.sel, s.r0, s.n), P(batch->positions, s.r0, s.n), segcu(s), RAD_W(L.sinks),
                                    R(m.o, s.r0, s.n), acs);
                    else
                        RAD_ISSUE_N(x, L.attn, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused,
                                    kv_cache(m.comp[r], src), cbt, R(m.sel, s.r0, s.n), P(batch->positions, s.r0, s.n),
                                    segcu(s), RAD_W(L.sinks), R(m.o, s.r0, s.n));
                } else if (ofold) {
                    RAD_ISSUE_N(x, ah, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, RAD_NONE, RAD_NONE, RAD_NONE,
                                P(batch->positions, s.r0, s.n), segcu(s), RAD_W(L.sinks), R(m.o, s.r0, s.n), acs);
                } else {
                    RAD_ISSUE_N(x, L.attn, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, RAD_NONE, RAD_NONE,
                                RAD_NONE, P(batch->positions, s.r0, s.n), segcu(s), RAD_W(L.sinks), R(m.o, s.r0, s.n));
                }
            }
            if (ofold) {
            } else if (L.o_f) {
                whole([&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, L.o_f, n, R(m.o, r0, n), brows(cs, g.max_ctx), P(batch->positions, r0, n));
                });
            } else {
                whole([&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, L.o_rope, n, R(m.o, r0, n), brows(cs, g.max_ctx), P(batch->positions, r0, n));
                    if (L.o_q) RAD_ISSUE_N(x, L.o_q, n, R(m.o, r0, n));
                });
            }
            bt_pt(x, m, m.o, T, "o", l);
            const int64_t gd = HD / m.gr;
            rows(kBiBand, [&](int64_t r0, int64_t n) {
                for (int j = 0; j < m.gr; ++j)
                    lin_issue(x, m, L.o_a[(size_t)j], n, C(m.o, j * gd, gd, r0, n), L.wo_a[(size_t)j],
                              C(m.oa, j * c.o_rank, c.o_rank, r0, n));
            });
            bt_pt(x, m, m.oa, T, "oa", l);
            if (late) mix_a();
            rows(kBiBand, [&](int64_t r0, int64_t n) {
                lin_issue(x, m, L.o_b, n, R(m.oa, r0, n), L.wo_b, R(m.attn_out, r0, n));
            });
            bt_pt(x, m, m.attn_out, T, "attn_out", l);
        }
        /* the attention's all-reduce over the step (exact: 0 + r0 + .. + r3 a value), then post / collapse / norm (a
         * fused norm) by segment; ar_pcn_f (the all-reduce with it, RADIANCE_DS41_AR_PCN) is not taken under ds41_bi */
        if (L.ar_attn)
            whole([&](int64_t r0, int64_t n) { RAD_ISSUE_N(x, L.ar_attn, n * D, R(m.attn_out, r0, n), RAD_NONE); });
        if (L.pcn_f)
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.pcn_f, n, R(m.attn_out, r0, n), R(m.h0, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h1, r0, n), R(m.pre_a, r0, n), R(m.xcol, r0, n), RAD_W(L.ffn_norm), R(m.act.x, r0, n));
            });
        else
            whole([&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.post_a, n, R(m.attn_out, r0, n), R(m.h0, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h1, r0, n));
            });
        if (L.pcn_f) {
        } else if (L.cn_f) {
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.cn_f, n, R(m.h1, r0, n), R(m.pre_a, r0, n), R(m.xcol, r0, n), RAD_W(L.ffn_norm),
                            R(m.act.x, r0, n));
            });
        } else {
            whole([&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.col_f, n, R(m.h1, r0, n), R(m.pre_a, r0, n), R(m.xcol, r0, n));
                RAD_ISSUE_N(x, L.norm_f, n, R(m.xcol, r0, n), RAD_W(L.ffn_norm), R(m.act.x, r0, n));
            });
        }
        bt_pt(x, m, m.h1, T, "h1", l);
        bt_pt(x, m, m.act.x, T, "ffn_in", l);
        /* the MoE: one pass a segment (the router, the experts at the segment's rows, the shared arm, the gather and
         * the all-reduce over its rows); the ffn mix beside the first (r0 = 0) */
        MixBeside mb{ &L, &m, T };
        MoeFP8::Beside bs;
        bs.fn = mix_f_beside;
        bs.arg = &mb;
        /* lever PSK: a cut layer's ffn mix over its runs, here (the block issues a beside only in a pass from row 0) */
        const MoeFP8::Beside* bsp = &bs;
        if (cut) {
            whole([&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.mix_f, n, R(m.h1, r0, n), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                            R(m.pre_f, r0, n), R(m.post, r0, n), R(m.comb, r0, n));
            });
            bsp = nullptr;
        }
        {   /* ds41_bi 2: the decode segment in pieces within the EXL3 decode band */
            const int64_t piece = bi_pieces(g_sw_bi.load(std::memory_order_relaxed)) ? kBiMoe : ((int64_t)1 << 40);
            for (const BiSeg& s : sg) {
                if (!s.dec || s.n <= piece) {
                    L.moe.pass(x, T, s.r0, s.n, true, bsp);
                    if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                for (int64_t a = 0; a < s.n; a += piece) {
                    L.moe.pass(x, T, s.r0 + a, std::min(piece, s.n - a), true, bsp);
                    if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        /* lever PSK-L20: a split layer that ran no MoE pass reports nothing (as a layer skipped whole: the router's
         * counts are another layer's) */
        if (!l20_none) L.moe.route_report(x);
        bt_pt(x, m, m.mw.logits, T, "router", l);
        bt_pt(x, m, m.mw.ids, T, "ids", l);
        bt_pt(x, m, m.mw.ew, T, "ew", l);
        bt_pt(x, m, m.mw.sout, T, "sout", l);
        bt_pt(x, m, m.act.x, T, "moe_out", l);
        /* post_f, or post_f with the next collapse + norm (a fused norm, by segment); never the AR_PCN forms here */
        const Layer* nx = L.l + 1 < c.n_layer ? &m.layers[(size_t)L.l + 1] : nullptr;
        if (nx && nx->pcn_a)
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, nx->pcn_a, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(nx->attn_norm), R(m.xn, r0, n));
            });
        else if (!nx && m.pcn_fin)
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, m.pcn_fin, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(m.final_gain),
                            R(m.final_hidden, r0, n));
            });
        else
            whole([&](int64_t r0, int64_t n) {
                RAD_ISSUE_N(x, L.post_f, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h0, r0, n));
            });
    }
    if (psk && cur != &sg && rad_rank(x) == 0) {   /* lever PSK: a cut step (stats; the first one's cut rows) */
        g_psk_cut_passes.fetch_add(1, std::memory_order_relaxed);
        const long long n = g_psk_steps.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == 1 || n % 256 == 0) {
            std::string seg;
            for (const BiSeg& s : sg) seg += std::string(seg.empty() ? "" : ", ") + (s.dec ? "decode " : "prefill ") +
                                             std::to_string(s.n);
            /* design 2's counters only once a planned pass ran (design 1's line is today's text) */
            std::string d2;
            if (const long long h = g_psk_hsteps.load(std::memory_order_relaxed))
                d2 = "; design 2: " + std::to_string(h) + " planned passes, " +
                     std::to_string(g_psk_skips.load(std::memory_order_relaxed)) + " layers skipped whole";
            fprintf(stderr, "radiance: deepseek41: lever PSK: %lld cut steps, %lld layer-rows not run (this step: %lld rows "
                            "as [%s]; layer %d runs %lld a cut prefill segment, layer %lld %lld%s)\n", n,
                    g_psk_rows.load(std::memory_order_relaxed), (long long)T, seg.c_str(), m.psk_first,
                    (long long)(kPskBase + (c.n_layer - 1 - m.psk_first) * c.window), (long long)c.n_layer - 1,
                    (long long)kPskBase, d2.c_str());
        }
    }
    if (m.pcn_fin) {
    } else if (m.cn_fin) {
        rows(kBiNorm, [&](int64_t r0, int64_t n) {
            RAD_ISSUE_N(x, m.cn_fin, n, R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(m.final_gain),
                        R(m.final_hidden, r0, n));
        });
    } else {
        whole([&](int64_t r0, int64_t n) {
            RAD_ISSUE_N(x, m.contract, n, R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n));
            RAD_ISSUE_N(x, m.fnorm, n, R(m.xcol, r0, n), RAD_W(m.final_gain), R(m.final_hidden, r0, n));
        });
    }
    bt_pt(x, m, m.final_hidden, T, "final", -1);
    if (batch->n_out > 0) {
        RAD_ISSUE_N(x, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        /* the head (logits_gemm_ggml, banded): the sampled rows in launches within the decode band */
        for (int64_t a = 0; a < batch->n_out; a += kBiBand) {
            const int64_t n = std::min(kBiBand, batch->n_out - a);
            RAD_ISSUE_N(x, m.head_op, n, a ? brow_slice(m.logit_input, a, n, D) : brows(m.logit_input, n), RAD_W(m.head),
                        a ? brow_slice(m.logits, a, n, g.n_vocab) : brows(m.logits, n));
        }
        bt_pt(x, m, m.logits, batch->n_out, "logits", -1, 1);
    }
    bt_flush(x, m, batch);
}

static void step(RadCtx* x, const RadBatch* batch) {
    auto& m = *models[rad_rank(x)];
    /* A drafter pass: the sign gives the kind (rad_runtime.h); zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.dsp.on) {
            rad_step_fail(x, "deepseek41: a drafter pass and no DSpark drafter is declared (the checkpoint "
                             "carries none, or declare ran at max_spec 0)");
            return;
        }
        if (batch->draft_pass > 0) dsp_query(x, m, batch);
        else dsp_context(x, m, batch);
        return;
    }
    /* lever PSK design 2: a pass the scheduler planned (rad_psk_plan) is always cut, so a step of more than one segment
     * runs in BI's segments even under ds41_bi 0 (then as ds41_bi 1: segments, no pieces) */
    const bool psk_plan = m.psk_first > 0 && psk_hint(batch) != nullptr;
    /* lever BI (ds41_bi 1): a step of more than one shape class runs in segments (step_bi) */
    if (m.bi_sub && (g_sw_bi.load(std::memory_order_relaxed) > 0 || psk_plan)) {
        static thread_local std::vector<BiSeg> sg;
        if (bi_segments(x, m, batch, sg)) {
            step_bi(x, m, batch, sg);
            return;
        }
    }
    /* lever PSK (ds41_psk 1): a step of one prefill sequence and no decode rows, longer than the last layer's cone, runs
     * through step_bi as that one segment, cut from m.psk_first on (a one-segment step_bi issues step()'s ops on the same
     * rows below the cut); design 2: any planned one-sequence prefill step */
    if (m.psk_first > 0 && batch->draft_pass == 0 && batch->n_seq == 1 &&
        ((batch->n_tok > kPskBase && g_sw_psk.load(std::memory_order_relaxed) > 0) || psk_plan)) {
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        if (Dn == 0 && psk_step_ok(m, batch, 1, 0)) {
            static thread_local std::vector<BiSeg> ps(1);
            ps[0] = BiSeg{ 0, batch->n_tok, 0, 1, false, -1 };
            step_bi(x, m, batch, ps, true);
            return;
        }
    }
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    bt_begin(m);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const auto* wb = kv_batch(batch, m.win);
    if (!wb || !wb->block_table || !wb->slot_mapping) {
        rad_step_fail(x, "deepseek41: the window group has no block table");
        return;
    }
    const RadOperand wbt = praw2(wb->block_table, RAD_I32, S, wb->block_table_pitch);
    /* The window group's table is rebased at the sequence's first held block; its seqused counts
     * from there, and attn_sparse_sink recovers the base from it (docs/OPS.md). */
    const RadOperand wused = praw(wb->seqused, RAD_I32, S);

    RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
    if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    for (int64_t i = 0; i < c.hc; ++i)
        RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
    RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
    RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    if (m.sone && g_sw_one.load(std::memory_order_relaxed) > 0) RAD_ISSUE_N(x, m.sone, T, brows(m.mw.sgate, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap),
                    brows(m.ctok, T));

    /* The Engram's row ids are a function of the batch's tokens alone (token_map, each Engram
     * layer's token history), never of an activation, so every Engram layer's ngram_ids is issued
     * here, before the first block, each into its own host buffer (rad_block_ple.h's ids() does
     * the same). A host-site gather (the table on the disk or RAM tier) joins on the last device
     * op that touched the host arena: these few ops, or the previous Engram layer's copy of its
     * rows. It then reads its rows while the device runs the blocks the host has already issued,
     * where ngram_ids at its layer (and device-domain rows) made the device drain before every
     * gather and idle through it. */
    for (auto& L : m.layers) {
        if (!L.e_ids) continue;
        const int l = L.l;
        const auto* tk = kv_batch(batch, m.tok);
        const int32_t* tk_idx = tk ? tk->state_index : nullptr;
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        const int64_t P = S - Dn;
        if (Dn > 0)
            RAD_ISSUE_N(x, L.e_ids, DT, brows(m.ctok, DT), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens, RAD_I32, Dn + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off), praw(tk_idx, RAD_I32, Dn),
                        praw(batch->ctx_lens, RAD_I32, Dn),
                        praw(batch->num_accepted, RAD_I32, Dn), brows(L.ids, DT), RAD_NONE);
        if (P > 0)
            RAD_ISSUE_N(x, L.e_ids, T, brows(m.ctok, T), kv_cache(m.tok, l),
                        praw(batch->cu_seqlens + Dn, RAD_I32, P + 1), RAD_W(L.e_mult),
                        RAD_W(L.e_vsz), RAD_W(L.e_off),
                        praw(tk_idx ? tk_idx + Dn : nullptr, RAD_I32, P),
                        praw(batch->ctx_lens + Dn, RAD_I32, P), RAD_NONE, brows(L.ids, T),
                        RAD_NONE);
    }
    engdd_mark(x, m, batch);   /* lever E1: the ids' event (a pass it may act on) */

    for (auto& L : m.layers) {
        const int l = L.l;
        const int64_t r = L.r;
        if (L.e_ids) {
            const int64_t H = c.ple_heads;
            /* lever E1: the step's rows are one kv GEMM launch below; deduplicated when its plan says so */
            m.edd_ls.assign(1, ds41_engdd::Launch{ 0, T });
            if (!engdd_issue(x, m, L, batch, T, m.edd_ls)) {
                /* T rows of operands; the gather's band is T * H, one table row per head per token,
                 * read through the ids and rows as flat (rad_block_ple.h does the same). */
                if (m.ple_group > 0)
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                                m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, T));
                else
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
                RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
                lin_issue(x, m, L.e_gemm, T, brows(m.embd, T), L.e_kv, brows(m.ekv, T));
            }
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        bt_pt(x, m, m.h0, T, "h0", l);
        if (L.tap) RAD_ISSUE_N(x, L.tap, T, brows(m.h0, T), bcol(m.dsp.src, L.tap_col * D, D, T));
        /* Issue order of the connections' mixes (Ctx::haz_mark: a recorded pass leaves a dispatch
         * unordered when it touches nothing in flight). Each connection's collapse reads the gates
         * the previous connection's mix wrote, never what its own mix writes, and nothing reads a
         * mix's output before the next mhc_post. So, same kernels on the same operands:
         * - the collapse goes first. Where it is its own op (no fused mhc_post before it: the
         *   first layer, after an Engram block, the attention-zero rank) the mix's second launch,
         *   ordered anyway, is then the barrier the norm needs;
         * - on an attention rank the attention mix goes between the output projection's groups
         *   and attn_output_b, where its launches overlap those GEMVs instead of running nearly
         *   alone at the layer's start; on the attention-zero rank it follows the collapse. */
        const bool late = !m.zero;
        const auto mix_a = [&] {
            RAD_ISSUE_N(x, L.mix_a, T, brows(m.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                        RAD_W(L.hc_a_base), brows(m.pre_a, T), brows(m.post, T), brows(m.comb, T));
        };
        const auto col_a = [&] {
            if (L.cn_a)
                RAD_ISSUE_N(x, L.cn_a, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                            RAD_W(L.attn_norm), brows(m.xn, T));
            else if (L.col_a)   /* 0 under pcn_a: issued for the previous post_f */
                RAD_ISSUE_N(x, L.col_a, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
        };
        col_a();
        if (!late) mix_a();
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(m.attn_out, T));
        } else {
            if (L.norm_a)
                RAD_ISSUE_N(x, L.norm_a, T, brows(m.xcol, T), RAD_W(L.attn_norm), brows(m.xn, T));
            bt_pt(x, m, m.xn, T, "xn", l);
            const rad_buf cs = r > 0 ? m.cs_b : m.cs_a;
            /* The compressor (at a kv source) and the indexer (at an index source) read xn and qrn
             * as the query and key chains do, and their only order with them is the compressed
             * angles: c_tab rewrites cs_b after the query and the window key read it, and at a
             * source the index query's rope reads it after. Each of their ops goes between the
             * query/key pairs at the first point its inputs are issued, so a recorded pass overlaps
             * three or four chains instead of running the compressor and the indexer alone after
             * the window key. Same kernels on the same operands. */
            const bool ctail = L.src && r > 1;
            int64_t rows = T, tpitch = 1;
            RadOperand page = RAD_NONE, bpos = pos;
            const auto* cb = L.src ? kv_batch(batch, m.comp[r]) : nullptr;
            const auto* tb = ctail ? kv_batch(batch, m.tail[r]) : nullptr;
            if (L.src) {
                const auto* ib = kv_batch(batch, m.index[r]);
                if (!cb || !ib || !cb->block_table || !ib->block_table) {
                    rad_step_fail(x, "deepseek41: a compressed group has no block table");
                    return;
                }
                page = praw(cb->slot_mapping, RAD_I32, T);
                if (ctail) {
                    if (!tb || !tb->state_index) {
                        rad_step_fail(x, "deepseek41: the compressor tail has no state slots");
                        return;
                    }
                    tpitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
                    /* The step's completed blocks are at most ceil(T / r) + S (one a `r` tokens plus one
                     * straddling block a sequence; r4d_qsa_work_bf16.hip's host bound), and qsa_work needs
                     * one item a declared sequence for its counts: so the compressor chain runs over that
                     * many rows, not the work plane's m.work (8200 at a decode step's one token).
                     * RADIANCE_DS41_CTAIL=full keeps m.work (ctail_mode()). Items past the real count are
                     * page -1 either way. */
                    rows = m.ctail_full ? m.work
                                        : std::min<int64_t>(m.work, std::max<int64_t>((T + r - 1) / r + S, g.max_seqs));
                    page = brows(m.page, rows);
                    bpos = brows(m.bpos, rows);
                }
            }
            /* The query and key chains share only xn. Issued alternately, each op's predecessor
             * is in the other chain, so a recorded pass can leave it unordered behind it (the
             * tape's hazard check) and the two chains' dispatches overlap; one after the other,
             * every op but kv_g waits for the one before. Same kernels on the same operands. */
            const auto q_a = [&] { lin_issue(x, m, L.q_a, T, brows(m.xn, T), L.wq_a, brows(m.qr, T)); };
            /* Lever K1 fold a (switch ds41_qnfold): q_an + q_b as q_bf, at q_an's point where the indexer's i_q reads qrn
             * (value 1), else at q_b's point; value 2 leaves the indexer's layers to the pair. */
            const int qnf = L.q_bf ? g_sw_qnf.load(std::memory_order_relaxed) : 0;
            const bool qfold = qnf == 1 || (qnf == 2 && !L.i_score);
            const bool qfold_at_an = qfold && L.i_score;
            const auto q_fold = [&] {
                RAD_ISSUE_N(x, L.q_bf, T, brows(m.qr, T), RAD_W(L.wq_b), RAD_W(L.q_norm), brows(m.qrn, T), brows(m.q, T));
            };
            const auto q_an = [&] {
                if (qfold_at_an) q_fold();
                else if (!qfold) RAD_ISSUE_N(x, L.q_an, T, brows(m.qr, T), RAD_W(L.q_norm), brows(m.qrn, T));
            };
            const auto q_b = [&] {
                if (qfold) { if (!qfold_at_an) q_fold(); }
                else lin_issue(x, m, L.q_b, T, brows(m.qrn, T), L.wq_b, brows(m.q, T));
            };
            const auto q_rope = [&] { RAD_ISSUE_N(x, L.q_rope, T, brows(m.q, T), brows(cs, g.max_ctx), pos); };
            const auto kv_g = [&] { lin_issue(x, m, L.kv_g, T, brows(m.xn, T), L.wkv, brows(m.kv, T)); };
            /* The window key's norm, rope and round trip: one op (kv_f) where declared, else
             * kv_n here and kv_rope / kv_q at the next two points. Lever K1 fold c (switch ds41_kvfold):
             * kv_fs at kv_f's point, which also writes the cache slot, and no kv_st. */
            const bool kvfold = L.kv_fs && g_sw_kvf.load(std::memory_order_relaxed) > 0;
            const auto kv_n = [&] {
                if (kvfold)
                    RAD_ISSUE_N(x, L.kv_fs, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T),
                                brows(cs, g.max_ctx), pos, praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
                else if (L.kv_f)
                    RAD_ISSUE_N(x, L.kv_f, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T),
                                brows(cs, g.max_ctx), pos);
                else
                    RAD_ISSUE_N(x, L.kv_n, T, brows(m.kv, T), RAD_W(L.kv_norm), brows(m.kvn, T));
            };
            const auto kv_rope = [&] {
                if (!L.kv_f) RAD_ISSUE_N(x, L.kv_rope, T, brows(m.kvn, T), brows(cs, g.max_ctx), pos);
            };
            const auto kv_q = [&] {
                if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(m.kvn, T));
            };
            const auto kv_st = [&] {
                if (kvfold) return;   /* kv_fs stored the row */
                if (m.w8)             /* window_store: the row packed into its slot (lever KVWIN) */
                    RAD_ISSUE_N(x, L.kv_st, T, brows(m.kvn, T), praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
                else
                    RAD_ISSUE_N(x, L.kv_st, T, bcol(m.kvn, 0, hd / 2, T), bcol(m.kvn, hd / 2, hd / 2, T),
                                praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
            };
            const auto c_gemm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_gemm, T, brows(m.xn, T), RAD_W(L.comp_w),
                                brows(ctail ? m.kvs : m.pooled, T));
            };
            const auto c_work = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_work, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch),
                                praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), RAD_NONE,
                                brows(m.stage, rows * r), brows(m.page, rows), brows(m.bpos, rows),
                                brows(m.nc, S));
            };
            const auto c_pool = [&] {
                if (ctail)
                    RAD_ISSUE_N(x, L.c_pool, rows, brows(m.stage, rows * r), brows(m.pooled, rows),
                                brows(m.page, rows));
            };
            const auto c_norm = [&] {
                if (L.src)
                    RAD_ISSUE_N(x, L.c_norm, rows, brows(m.pooled, rows), RAD_W(L.comp_norm),
                                brows(m.lat, rows));
            };
            const auto c_tab = [&] {
                if (L.src) RAD_ISSUE_N(x, L.c_tab, rows, bpos, brows(m.cs_b, g.max_ctx));
            };
            const auto k_gemm = [&] {
                if (L.src) lin_issue(x, m, L.k_gemm, rows, brows(m.lat, rows), L.ik_w, brows(m.ik, rows));
            };
            /* The index key's norm, rope and round trip (k_f where declared); the latent's rope and
             * round trip (l_f where declared); the index query's (i_f where declared). Each part
             * function issues the fused op at the chain's first point and nothing at the others. */
            const auto k_norm = [&] {
                if (!L.src) return;
                if (L.k_f)
                    RAD_ISSUE_N(x, L.k_f, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows),
                                brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.k_norm, rows, brows(m.ik, rows), RAD_W(L.ik_norm), brows(m.ikn, rows));
            };
            const auto k_rope = [&] {
                if (L.src && !L.k_f) RAD_ISSUE_N(x, L.k_rope, rows, brows(m.ikn, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto k_q = [&] {
                if (L.src && L.k_q) RAD_ISSUE_N(x, L.k_q, rows, brows(m.ikn, rows));
            };
            const auto k_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.k_st, rows, brows(m.ikn, rows), page, kv_cache(m.index[r], l));
            };
            const auto l_rope = [&] {
                if (!L.src) return;
                if (L.l_f)
                    RAD_ISSUE_N(x, L.l_f, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
                else
                    RAD_ISSUE_N(x, L.l_rope, rows, brows(m.lat, rows), brows(m.cs_b, g.max_ctx), bpos);
            };
            const auto l_q = [&] {
                if (L.src && L.l_q) RAD_ISSUE_N(x, L.l_q, rows, brows(m.lat, rows));
            };
            const auto l_st = [&] {
                if (L.src) RAD_ISSUE_N(x, L.l_st, rows, brows(m.lat, rows), page, kv_cache(m.comp[r], l));
            };
            const auto c_tail = [&] {
                if (ctail && L.c_tail)
                    RAD_ISSUE_N(x, L.c_tail, T, brows(m.kvs, T), pos, cu, kv_cache(m.tail[r], l),
                                praw2(tb->state_index, RAD_I32, S, tpitch));
            };
            const auto i_q = [&] {
                if (L.i_score) lin_issue(x, m, L.i_q, T, brows(m.qrn, T), L.iq_w, brows(m.iq, T));
            };
            const auto i_rope = [&] {
                if (!L.i_score) return;
                if (L.i_f)
                    RAD_ISSUE_N(x, L.i_f, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
                else
                    RAD_ISSUE_N(x, L.i_rope, T, brows(m.iq, T), brows(m.cs_b, g.max_ctx), pos);
            };
            const auto i_qq = [&] {
                if (L.i_score && L.i_qq) RAD_ISSUE_N(x, L.i_qq, T, brows(m.iq, T));
            };
            const auto i_w = [&] {
                if (L.i_score) lin_issue(x, m, L.i_w, T, brows(m.xn, T), L.iproj, brows(m.iw, T));
            };
            const auto i_score = [&] {   /* the scores, the candidates and the selection */
                if (!L.i_score) return;
                const int src = c.src_of[(size_t)l];
                const auto* ib = kv_batch(batch, m.index[r]);
                const int64_t live = std::min<int64_t>(
                    g.max_ctx / r + 1, ((int64_t)batch->max_ctx_len + batch->max_q_len) / r + 1);
                const int64_t kw = keep_words(live, c.cand_block);
                const bool uses = c.cand >= 0 && c.cand < l;
                /* Run by run (index_runs()): score R rows into the plane, then select from them
                 * before the next run overwrites it. Rows are independent, so each row's score,
                 * candidates and selection are the whole step's. Split (m.isplit, a step of two
                 * runs or more at a long enough context), this rank takes runs [p0, p1) into its
                 * part and the gather joins every rank's part into sel (ds41_index_split.h). A
                 * layer past the candidate layer scores its kept blocks only (L.i_keep). Under the
                 * idx_split switch (idx_balance()) a split step whose runs the ranks do not divide
                 * takes the balanced deal: an equal block of rows a rank, in runs within the
                 * declared ones -- the same loop and gather below, only R, q and the runs' bounds move. */
                const bool split = m.isplit && m.sel_gather;
                const Ds41IdxRuns ru = ds41_idx_runs(T, m.irows, live, m.aw, m.ar, split, m.split_min_live,
                                                     split && idx_balance(), (int64_t)L.i_runs.size());
                if (ru.balanced && m.ar == 0 && !g_idx_bal_noted.exchange(1, std::memory_order_relaxed)) {
                    const Ds41IdxRuns pl = ds41_idx_runs(T, m.irows, live, m.aw, 0, split, m.split_min_live);
                    fprintf(stderr, "radiance: deepseek41 indexer deal: idx_split 1 took its first step: %lld rows as "
                                    "%lld runs of %lld, %lld a rank (the plain deal: %lld runs of %lld, %lld a "
                                    "rank, the last rank %lld)\n",
                            (long long)T, (long long)ru.n, (long long)ru.R, (long long)ru.q, (long long)pl.n,
                            (long long)pl.R, (long long)pl.q,
                            (long long)std::max<int64_t>(0, pl.n - (int64_t)(m.aw - 1) * pl.q));
                }
                /* the restriction below its threshold is the unrestricted op (the optional keep left out) */
                const bool rkeep = L.i_keep && live >= m.keep_min_live;
                for (int64_t p = ru.p0; p < ru.p1; ++p) {
                    const int64_t r0 = p * ru.R, rows = std::min(ru.R, T - r0);
                    const RadOperand prow = praw(batch->positions + r0, RAD_I32, rows);
                    RAD_ISSUE_N(x, L.i_runs[(size_t)p], ru.R, brows(m.iq, T), brows(m.iw, T),
                                kv_cache(m.index[r], src),
                                praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), pos, cu,
                                bcol(m.score, 0, live, ru.R),
                                rkeep ? bcol_at(m.keep, r0, m.kwords, 0, kw, rows) : RAD_NONE);
                    if (L.i_cand)
                        RAD_ISSUE_N(x, L.i_cand, rows, bcol(m.score, 0, live, rows), prow,
                                    bcol_at(m.keep, r0, m.kwords, 0, kw, rows));
                    RAD_ISSUE_N(x, L.i_topk, rows, bcol(m.score, 0, live, rows),
                                uses ? bcol_at(m.keep, r0, m.kwords, 0, kw, rows) : RAD_NONE, prow,
                                ru.split ? brow_slice(m.sel_part, (p - ru.p0) * ru.R, rows, c.itopk)
                                         : brow_slice(m.sel, r0, rows, c.itopk));
                }
                if (ru.split)
                    RAD_ISSUE(x, m.sel_gather, brows(m.sel_part, ru.q * ru.R),
                              brows(m.sel, (int64_t)m.aw * ru.q * ru.R));
            };

            c_gemm();
            q_a(); kv_g();
            i_w(); c_work();
            q_an(); kv_n();
            i_q(); c_pool();
            q_b(); kv_rope();
            if (!L.src) i_rope();   /* at a source it reads the angles c_tab rewrites */
            c_tail(); c_norm();
            q_rope(); kv_q();
            if (!L.src) i_qq();
            k_gemm(); c_tab();
            kv_st();
            k_norm();
            if (L.src) i_rope();
            l_rope(); k_rope();
            if (L.src) i_qq();
            l_q(); k_q(); l_st(); k_st(); i_score();
            bt_pt(x, m, m.qr, T, "qr", l);
            bt_pt(x, m, m.q, T, "q", l);
            bt_pt(x, m, m.kvn, T, "kvn", l);
            if (L.i_score) {
                bt_pt(x, m, m.iq, T, "iq", l);
                bt_pt(x, m, m.iw, T, "iw", l);
                bt_pt(x, m, m.sel, T, "sel", l);
            }

            /* Lever K1 fold d (switch ds41_ofold): attn_rt for the attention and its output's rope + round trip. */
            const bool ofold = L.attn_rt && g_sw_of.load(std::memory_order_relaxed) > 0;
            const rad_op ah = ofold ? L.attn_rt : L.attn;
            const RadOperand acs = ofold ? brows(cs, g.max_ctx) : RAD_NONE;
            if (r > 0) {
                const int src = c.src_of[(size_t)l];
                const auto* cb = kv_batch(batch, m.comp[r]);
                if (ofold)
                    RAD_ISSUE_N(x, ah, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused, kv_cache(m.comp[r], src),
                                praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), brows(m.sel, T), pos, cu,
                                RAD_W(L.sinks), brows(m.o, T), acs);
                else
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused,
                            kv_cache(m.comp[r], src),
                            praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), brows(m.sel, T),
                            pos, cu, RAD_W(L.sinks), brows(m.o, T));
            } else if (ofold) {
                RAD_ISSUE_N(x, ah, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused, RAD_NONE, RAD_NONE, RAD_NONE,
                            pos, cu, RAD_W(L.sinks), brows(m.o, T), acs);
            } else {
                RAD_ISSUE_N(x, L.attn, T, brows(m.q, T), kv_cache(m.win, l), wbt, wused, RAD_NONE,
                            RAD_NONE, RAD_NONE, pos, cu, RAD_W(L.sinks), brows(m.o, T));
            }
            if (ofold) {
                /* attn_rt rotated and rounded o */
            } else if (L.o_f) {
                RAD_ISSUE_N(x, L.o_f, T, brows(m.o, T), brows(cs, g.max_ctx), pos);
            } else {
                RAD_ISSUE_N(x, L.o_rope, T, brows(m.o, T), brows(cs, g.max_ctx), pos);
                if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(m.o, T));
            }
            bt_pt(x, m, m.o, T, "o", l);
            const int64_t gd = HD / m.gr;
            for (int j = 0; j < m.gr; ++j)
                lin_issue(x, m, L.o_a[(size_t)j], T, bcol(m.o, j * gd, gd, T), L.wo_a[(size_t)j],
                          bcol(m.oa, j * c.o_rank, c.o_rank, T));
            bt_pt(x, m, m.oa, T, "oa", l);
            if (late) mix_a();
            lin_issue(x, m, L.o_b, T, brows(m.oa, T), L.wo_b, brows(m.attn_out, T));
            bt_pt(x, m, m.attn_out, T, "attn_out", l);
        }
        if (L.ar_pcn_f && ar_wide_write_exact(m.g, T))   /* the all-reduce, post_a, col_f and norm_f in one */
            RAD_ISSUE_N(x, L.ar_pcn_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T));
        else {
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(m.attn_out, T), RAD_NONE);
        if (L.pcn_f)   /* post_a, col_f and norm_f in one */
            RAD_ISSUE_N(x, L.pcn_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T));
        else
            RAD_ISSUE_N(x, L.post_a, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T));
        }

        /* The ffn connection's mix is likewise needed first by the next mhc_post, so the MoE block
         * issues it (MoeFP8::Beside) behind its routed and shared gate_up, where a recorded pass
         * overlaps it with those GEMMs instead of with the collapse and the norm (or, under the
         * fused post, with nothing). It reads h1 and writes what post_a (fused or not) has read. */
        const auto col_f = [&] {
            if (L.pcn_f) {
            } else if (L.cn_f) {
                RAD_ISSUE_N(x, L.cn_f, T, brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                            RAD_W(L.ffn_norm), brows(m.act.x, T));
            } else {
                RAD_ISSUE_N(x, L.col_f, T, brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T));
                RAD_ISSUE_N(x, L.norm_f, T, brows(m.xcol, T), RAD_W(L.ffn_norm), brows(m.act.x, T));
            }
        };
        col_f();
        bt_pt(x, m, m.h1, T, "h1", l);
        bt_pt(x, m, m.act.x, T, "ffn_in", l);
        MixBeside mb{ &L, &m, T };
        MoeFP8::Beside bs;
        bs.fn = mix_f_beside;
        bs.arg = &mb;
        L.moe.step(x, batch, bs);
        bt_pt(x, m, m.mw.logits, T, "router", l);
        bt_pt(x, m, m.mw.ids, T, "ids", l);
        bt_pt(x, m, m.mw.ew, T, "ew", l);
        bt_pt(x, m, m.mw.sout, T, "sout", l);
        bt_pt(x, m, m.act.x, T, "moe_out", l);
        /* post_f, or post_f with the next collapse + norm (the next layer's pcn_a, or pcn_fin); under
         * RADIANCE_DS41_AR_PCN=2 with the block's all-reduce in front (the block left it to us: ar_out). */
        const Layer* nx = L.l + 1 < c.n_layer ? &m.layers[(size_t)L.l + 1] : nullptr;
        const bool arw = ar_wide_write_exact(m.g, T);
        if (nx && nx->ar_pcn_a && arw)
            RAD_ISSUE_N(x, nx->ar_pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(nx->attn_norm), brows(m.xn, T));
        else if (!nx && m.ar_pcn_fin && arw)
            RAD_ISSUE_N(x, m.ar_pcn_fin, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(m.final_gain), brows(m.final_hidden, T));
        else if (nx && nx->pcn_a)
            RAD_ISSUE_N(x, nx->pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(nx->attn_norm), brows(m.xn, T));
        else if (!nx && m.pcn_fin)
            RAD_ISSUE_N(x, m.pcn_fin, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(m.final_gain), brows(m.final_hidden, T));
        else
            RAD_ISSUE_N(x, L.post_f, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T));
    }
    if (m.pcn_fin) {
    } else if (m.cn_fin) {
        RAD_ISSUE_N(x, m.cn_fin, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                    RAD_W(m.final_gain), brows(m.final_hidden, T));
    } else {
        RAD_ISSUE_N(x, m.contract, T, brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T));
        RAD_ISSUE_N(x, m.fnorm, T, brows(m.xcol, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    }
    bt_pt(x, m, m.final_hidden, T, "final", -1);
    if (batch->n_out > 0) {
        RAD_ISSUE_N(x, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(x, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
        bt_pt(x, m, m.logits, batch->n_out, "logits", -1, 1);
    }
    bt_flush(x, m, batch);
}

/* RADIANCE_DS41_LEVELS=1: offer the core's arena levels (RadArchProbe::shape_probe_ok), which lend
 * the activation arena's idle top to the expert slab at small steps and arm the prefill stager
 * (core/place/stager.h) -- placement only, so exact, provided no step writes past what a level
 * keeps. Unset (today) or RADIANCE_DS41_CTAIL=full: refused, since the full compressor tail runs
 * every step over the work plane's capacity rather than the step's rows. */
static bool levels_ok() {
    const char* e = std::getenv("RADIANCE_DS41_LEVELS");
    if (!(e && (!std::strcmp(e, "1") || !std::strcmp(e, "on")))) return false;
    const char* ct = std::getenv("RADIANCE_DS41_CTAIL");
    return !(ct && !std::strcmp(ct, "full"));
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (levels_ok()) out->shape_probe_ok = 1;   /* declare() writes scratch under shape_probe */
    /* A DSpark drafter proposes the block it was trained at, every row predicting (block = depth). */
    const int64_t ns = gi(meta, "nextn_predict_layers", 0), blk = gi(meta, "dspark.block_size", 0);
    if (ns > 0 && blk > 0) {
        out->draft_depth = (int)blk;
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

/* DSML tool calls with typed argument tags, which the template analysis cannot state. */
static const RadChatFormat* chat_format(const RadModelMeta*) { return rad::arch::deepseek41_chat_format(); }
}  // namespace deepseek41

#ifndef RAD_ARCH_NO_EXPORTS
/* Lever BI: a pass whose segments come from its prefill sequences' own lengths (two or more prefill sequences under
 * ds41_bi 1) is not recorded -- the key below does not hold those lengths; every other pass keys as before (a constant).
 * No DeepSeek kernel reads RadArgs::kv_used_max_h, the one thing the callback's presence changes in the issue path. */
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* b, const int32_t* kv_used_max_h) {
    (void)c;
    (void)kv_used_max_h;
    /* Lever PSK design 2: a pass the scheduler planned (rad_psk_plan) is shaped by its targets, which no key holds: not
     * recorded */
    if (deepseek41::psk_hint(b)) return 0;
    /* Lever E1: a pass it may deduplicate issues row counts read from the step's ids at issue: not recorded */
    if (deepseek41::engdd_may_act(b)) return 0;
    if (b && b->draft_pass == 0 && deepseek41::g_sw_bi.load(std::memory_order_relaxed) > 0) {
        int64_t Dn = 0, DT = 0;
        rad::arch::batch_split(b, &Dn, &DT);
        if (b->n_seq - Dn >= 2) return 0;
    }
    /* Lever PSK-DS (dspark_pfctx 1): a context pass cut to its entries' readable rows is shaped by their lengths, which
     * the key does not hold: not recorded. A pass it leaves whole issues as today and keys as today. */
    if (b && b->draft_pass < 0 && deepseek41::g_sw_pfctx.load(std::memory_order_relaxed) > 0) {
        static thread_local std::vector<std::pair<int64_t, int64_t>> ru;
        int64_t kept = 0;
        if (deepseek41::pfctx_plan(b, deepseek41::g_pfctx_window.load(std::memory_order_relaxed), &ru, &kept)) return 0;
    }
    return 1;
}

/* Lever PSK design 2 (ds41-psk2): the core, at a step's release with every rank thread parked (Engine::psk_release), hands
 * the scheduler's plan for trunk pass `step`: per prefill sequence in the step's order its `ahead` (the first target past
 * the chunk's first position, less the chunk's end; kPskNoHint: not planned) and its chunk rows, and the step's decode
 * rows. Returns the first layer from which no row of the step runs (the pass issues nothing from it on; the stager stages
 * no layer from it), n_layer when every layer runs, -1 when lever PSK is not declared here (the plan is not kept). Under
 * lever PSK-L20 that layer can be the split one: no row in its cone, so no routed expert of it or after it is read, while
 * its all-row part (collapse + norm, compressor, latent / index-key stores: no routed expert) still issues. */
/* Lever PSK's live counters for /stats (ds41-int7-pskglue), rank 0's, cumulative: out[0] trunk passes that followed a
 * scheduler plan, [1] passes with a cut (design 1 or 2), [2] layers skipped whole (design 2), [3] layer-rows not run at
 * the first nine cut layers (21..29), [4] at the rest (30..39), [5] plans dropped as not matching their pass. Returns
 * how many it wrote (6), 0 when lever PSK is not declared here. */
extern "C" int rad_psk_stats(long long* out, int n) {
    /* lever PSK-CR (ds41-pskcr): [10] replay passes cut, [11] their layer-rows run, [12] their layer-rows not run below
     * the first PSK layer, [13] kv-source compressor chains skipped, [14] replay passes beside other rows, [15] 0 */
    using namespace deepseek41;
    if (!out || n < 6 || g_psk_first.load(std::memory_order_relaxed) <= 0) return 0;
    out[0] = g_psk_hsteps.load(std::memory_order_relaxed);
    out[1] = g_psk_cut_passes.load(std::memory_order_relaxed);
    out[2] = g_psk_skips.load(std::memory_order_relaxed);
    out[3] = g_psk_rows_lo.load(std::memory_order_relaxed);
    out[4] = g_psk_rows_hi.load(std::memory_order_relaxed);
    out[5] = g_psk_hbad.load(std::memory_order_relaxed);
    if (n < 10) return 6;
    /* lever PSK-L20 (ds41-pskl20): [6] passes whose split layer was cut, [7] its rows not run in the cone part, [8]
     * passes it ran its all-row part only, [9] planned passes told to end at it (its routed experts not staged) */
    out[6] = g_psk_l20_passes.load(std::memory_order_relaxed);
    out[7] = g_psk_l20_rows.load(std::memory_order_relaxed);
    out[8] = g_psk_l20_empty.load(std::memory_order_relaxed);
    out[9] = g_psk_l20_end.load(std::memory_order_relaxed);
    if (n < 16) return 10;
    out[10] = g_cr_passes.load(std::memory_order_relaxed);
    out[11] = g_cr_rows_run.load(std::memory_order_relaxed);
    out[12] = g_cr_rows_skip.load(std::memory_order_relaxed);
    out[13] = g_cr_chains.load(std::memory_order_relaxed);
    out[14] = g_cr_mixed.load(std::memory_order_relaxed);
    out[15] = 0;
    return 16;
}

/* Lever E1's live counters for /stats "engdd" (ds41-e1), rank 0's, cumulative: out[0] Engram layer passes deduplicated,
 * [1] rows not looked up / copied / multiplied in them, [2] rows pooled in them, [3] layer passes the 48-row guard kept
 * whole, [4] layer passes below the repeat share, [5] steps with a deduplicated layer, [6] layer passes with no launch
 * above the band. Returns how many it wrote (7), 0 when lever E1 is not declared here. */
extern "C" int rad_engdd_stats(long long* out, int n) {
    using namespace deepseek41;
    if (!out || n < 7 || !g_sw_engdd_ok.load(std::memory_order_relaxed)) return 0;
    out[0] = g_edd_passes.load(std::memory_order_relaxed);
    out[1] = g_edd_rows.load(std::memory_order_relaxed);
    out[2] = g_edd_pooled.load(std::memory_order_relaxed);
    out[3] = g_edd_guard.load(std::memory_order_relaxed);
    out[4] = g_edd_low.load(std::memory_order_relaxed);
    out[5] = g_edd_steps.load(std::memory_order_relaxed);
    out[6] = g_edd_nopool.load(std::memory_order_relaxed);
    return 7;
}

static int psk_plan_impl(int step, long long dec_tok, int n, const int32_t* ahead, const int32_t* rows,
                         const int32_t* flags) {
    using namespace deepseek41;
    const int first = g_psk_first.load(std::memory_order_relaxed), nl = g_psk_nlayer.load(std::memory_order_relaxed);
    const int w = g_psk_window.load(std::memory_order_relaxed);
    if (first <= 0 || nl <= first || n < 0 || (n > 0 && (!ahead || !rows))) {
        g_psk_plan.step = -1;
        return -1;
    }
    g_psk_plan.step = step;
    g_psk_plan.dec_tok = dec_tok;
    g_psk_plan.ahead.assign(ahead, ahead + n);
    g_psk_plan.rows.assign(rows, rows + n);
    /* lever PSK-CR: the replay flags (none from rad_psk_plan) */
    g_psk_plan.flags.clear();
    bool cr = false;
    for (int i = 0; flags && i < n; ++i) cr = cr || (flags[i] & 1) != 0;
    if (cr) g_psk_plan.flags.assign(flags, flags + n);
    /* lever PSK-L20: the split layer's cone part is cut as the layers past it; its all-row part stages no routed expert,
     * so the scan starts there and a pass with no row in its cone ends at it (its experts not staged) */
    const int l20 = g_psk_l20_layer.load(std::memory_order_relaxed);
    g_psk_plan.l20 = l20 >= 0 && l20 < first && g_sw_psk_l20.load(std::memory_order_relaxed) > 0;
    if (dec_tok > 0 || n == 0) return nl;
    static std::vector<int32_t> ah, from;   /* the scheduler thread's, every rank parked */
    ah.resize((size_t)n);
    for (int i = 0; i < n; ++i) ah[(size_t)i] = psk_ahead(&g_psk_plan, (size_t)i);
    const int cut0 = g_psk_plan.l20 ? l20 : first;
    if (!cr) {
        const int end = ds41cone::plan_end(nl, w, cut0, n, ah.data(), rows);
        if (g_psk_plan.l20 && end == l20) g_psk_l20_end.fetch_add(1, std::memory_order_relaxed);
        return end;
    }
    /* lever PSK-CR: a replay entry is cut from layer 0 (its last chunk's experts staged only where its cone reaches) */
    from.resize((size_t)n);
    for (int i = 0; i < n; ++i) from[(size_t)i] = (flags[i] & 1) ? 0 : cut0;
    return ds41cone::plan_end_from(nl, w, n, ah.data(), rows, from.data());
}

extern "C" int rad_psk_plan(int step, long long dec_tok, int n, const int32_t* ahead, const int32_t* rows) {
    return psk_plan_impl(step, dec_tok, n, ahead, rows, nullptr);
}

/* Lever PSK-CR (ds41-pskcr): rad_psk_plan with each prefill sequence's flags (bit 0: a cone-replay chunk, cut at every
 * layer to its `ahead`'s cone, no global-KV store). */
extern "C" int rad_psk_plan2(int step, long long dec_tok, int n, const int32_t* ahead, const int32_t* rows,
                             const int32_t* flags) {
    return psk_plan_impl(step, dec_tok, n, ahead, rows, flags);
}

/* Lever PSK-CR: the rows before its end a cone replay must start at or below -- layer 0's cone, kPskBase + (n_layer - 1)
 * x window (5152 on the 1017) -- or -1 where design 2's plan (and so a replay) is not available. */
extern "C" int rad_psk_cr_reach(void) {
    using namespace deepseek41;
    const int nl = g_psk_nlayer.load(std::memory_order_relaxed), w = g_psk_window.load(std::memory_order_relaxed);
    if (!g_sw_psk2_ok.load(std::memory_order_relaxed) || nl <= 0 || w <= 0) return -1;
    return (int)psk_cone(nl, w, 0);
}

/* Lever PSK-DS's live counters for /stats "dspark_pfctx" (ds41-pskl20), rank 0's, cumulative: out[0] the switch
 * (dspark_pfctx), [1] context passes cut, [2] their rows not run, [3] the rows they ran. Returns how many it wrote (4),
 * 0 when no declare has the context pass here. */
extern "C" int rad_dspark_pfctx_stats(long long* out, int n) {
    using namespace deepseek41;
    if (!out || n < 4 || !g_sw_pfctx_ok.load(std::memory_order_relaxed)) return 0;
    out[0] = g_sw_pfctx.load(std::memory_order_relaxed);
    out[1] = g_pfctx_passes.load(std::memory_order_relaxed);
    out[2] = g_pfctx_skipped.load(std::memory_order_relaxed);
    out[3] = g_pfctx_run.load(std::memory_order_relaxed);
    return 4;
}

RAD_ARCH_PROBE(deepseek41)
RAD_ARCH_CHAT_FORMAT(deepseek41)
RAD_ARCH_PLUGIN(deepseek41, "deepseek41", "", "0.1.0",
                "DeepSeek-V4.1: hyper-connection streams, window + sparse compressed latent "
                "attention, sqrt(softplus) MoE, Engram")
#endif
