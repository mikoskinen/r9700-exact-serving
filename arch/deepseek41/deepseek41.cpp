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
#include <arch/rad_block_ds41_vision.h>
#include <arch/rad_block_vit.h>   /* MediaRows */
#include <arch/rad_deepseek41_chat.h>

#include "ds41_dec_band.h"   /* tp8 loop F58: a decode segment above the band, in band pieces */
#include "ds41_engram_dedup.h"
#include "ds41_index_split.h"
#include "ds41_layers.h"
#include "rad_pp.h"        /* tp8 S4: the two-node pipeline (rad_pp_info, the stage crossings) */
#include "ds41_psk_cone.h"
#include "ds41_sp_plan.h"   /* tp8 loop P20 round 2: RADIANCE_DS41_SPHC's layers */
#include "ds41_etp.h"
#include "ds41_walk_geom.h"
#include "ds41_prefetch.h"   /* tp8 loop D24: the weight reads' pass / site rules */
#include "ds41_dkv_held.h"   /* tp8 loop D25: the gathered latent rows held a decode band piece */
#include "rad_spechub.h"   /* lever SD: the coupling wire (RAD_MTP_COUPLE_*), --draft-coupling */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
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
extern "C" int rad_drafter_short_block(RadBuilder* b, int64_t block) __attribute__((weak));   /* tp8 loop D18 */
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
    /* Pictures (Model::media): an Engram layer's value rows zeroed at the media rows before its gate */
    rad_op mm_fill = 0, mm_put = 0;
    /* This Engram layer's row ids [T, heads]: one buffer a layer, since step() writes every
     * layer's ids before the first block runs (see there). */
    rad_buf ids = 0;
    /* Lever E1 (RADIANCE_DS41_ENGDD declared): the lookup over the compact ids `edd_ids` ([T, heads] i32, host), the kv
     * GEMM into Model::ekv_u, the gather of every pooled row's kv row from it through `edd_map` ([T] i32, host). Both
     * written by the plugin at issue and persistent, one a layer: a layer's gather (and a lookup the card ran) may still
     * be queued when the next layer's are written. 0 where not declared. */
    rad_op e_emb_dd = 0, e_gemm_u = 0, e_scat = 0;
    rad_buf edd_ids = 0, edd_map = 0;
    /* Lever 1U-E2 (RADIANCE_DS41_E2 declared): this rank's column part of the kv GEMV (gemm_nt_ggml, N / world, over a row
     * view of e_kv) into Model::ekv_p and the row-interleaved all_gather into Model::ekv; 0 where not declared. */
    rad_op e_gemm_p = 0, e_gath = 0;
    /* Lever 1U-GQ (RADIANCE_DS41_GQ declared): q_a + kv_g as one gemm_nt_ggml_pair (qkv_pair), and the two
     * attention_output_a groups as one (oa_pair); 0 where not declared. */
    rad_op qkv_pair = 0, oa_pair = 0;
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
    /* tp8 loop P13b (RADIANCE_DS41_PIPE2_ATTN): ar_attn declared a second time at `overlap` 1 (the narrow launch beside
     * compute) for piece 0 of a pipelined pass; 0 otherwise */
    rad_op ar_attn_ov = 0;
    std::vector<rad_op> o_a;
    /* tp8 loop P30f: the buffer each attention_output_a group writes -- 0 = its column slice of attn_o_lora (today); an IQR
     * group (a weight sidecar's int8 dense) writes its own [T, o_rank] buffer, since libr4d's int8 GEMMs take C at pitch N
     * only (r4d_fp8_args.h), and o_b's quantiser then reads the groups one by one (IqrQOp::parts) */
    std::vector<rad_buf> oa_dst;
    /* tp8 loop P30t: every attention_output_a group and attention_output_b have a TWIN (Model::twin) -- the twin groups write
     * Model::oa_g (as P30f's IQR groups) and o_b's twin quantises them a group at a time; a prompt segment issues the four
     * together (oa_fn / ob_fn), a decode row the GGUF ops over attn_o_lora as today */
    bool oa_tw = false;
    /* Lever K1's folds (k1_read_env), 0 where not declared: kv_fs = kv_f + kv_st
     * (rmsnorm_rope_roundtrip_kv_store), attn_rt = attn + o_f, or attn + o_rope + o_q
     * (attn_sparse_sink_rope_roundtrip). Issued for those when their switch is 1. */
    rad_op kv_fs = 0, attn_rt = 0;
    /* Lever DKV (dkv_read_env), 0 where not declared: lu_unpack = latent_unpack into Model::lu, attn_lu / attn_rtlu = attn /
     * attn_rt reading it; lgath = latent_gather into Model::selkv, attn_g / attn_rtg = attn / attn_rt reading that. */
    rad_op lu_unpack = 0, attn_lu = 0, attn_rtlu = 0, lgath = 0, attn_g = 0, attn_rtg = 0;
    /* attn_lu / attn_rtlu with plan_full 1 (the plan holds the whole pass: the LDS-free form), issued where the host's
     * positions say so */
    rad_op attn_luf = 0, attn_rtluf = 0;
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
    /* Lever M1 (RADIANCE_DS41_MHC_FUSE declared): pcn_a + this layer's mix_a as one op (mhc_post_collapse_rmsnorm_mix),
     * declared beside pcn_a, issued in its place for a launch the mhc_fuse switch takes (mhc_fuse_min). 0 elsewhere. */
    rad_op pcn_mix = 0;
    /* ... and the ffn connection's: pcn_f + this layer's mix_f (the 1017's 40 layers) */
    rad_op pcn_mix_f = 0;
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
    /* tp8 loop D24 (RADIANCE_DS41_PREFETCH declared; fn4 ds41_prefetch): libr4d `prefetch` reads of this layer's weights
     * (each weight's first RADIANCE_DS41_PREFETCH_MB), 0 where not declared: pf_f / pf_r / pf_s = hc_f_fn, the router
     * projection, the shared arm's gate_up (beside this layer's attention all-reduce); pf_a / pf_c / pf_q / pf_kv =
     * hc_a_fn, comp_w, wq_a, wkv (beside the previous held layer's MoE all-reduce). */
    rad_op pf_f = 0, pf_r = 0, pf_s = 0, pf_a = 0, pf_c = 0, pf_q = 0, pf_kv = 0;
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
    /* tp8 loop D18, the short block (RADIANCE_DS41_DSPARK_SHORT=S, rad_drafter_short_block; --draft-short-from picks it
     * per step): S rows a drafting sequence, the anchor first, each row proposing the token after it (S proposals, the
     * front of each proposal row). The ops whose shape names the block, declared again at S: the rows' offsets to the
     * block's last position, each stage's window attention (window + S keys), the walks (top-K, W1, SD), the confidence
     * head; every other op of the query pass is per row. No set: the drafter serves its own block only.
     * tp8 loop D19: RADIANCE_DS41_DSPARK_SHORT=S1,S2,... declares one set a length (the engine's RADIANCE_DRAFT_SHORT_TIERS
     * picks among them by load); a pass of S rows a sequence issues the set of S. */
    struct Short {
        int64_t S = 0;
        rad_op poff = 0, select = 0, wmarkov = 0, conf = 0;
        std::vector<rad_op> attn, wstep, sstep;
    };
    std::vector<int64_t> sblks;   /* the env's lengths, in its order */
    std::vector<Short> sh;        /* the declared sets (empty: none, or not served -- sbwhy) */
};

/* A dense linear's int8 activation pair, keyed by the GEMM op a linear() declared: the quantiser
 * writes the pair and the GEMM reads it. One pair for the model's widest activation keeps a
 * quantised trunk's arena small; a GEMM's own columns are sliced at issue. */
struct IqrQOp {
    rad_op  op = 0;
    int64_t K  = 0;
    /* the pair this linear's quantiser writes and its GEMM reads (Model::iqr_q8 / iqr_s8, or a wider pair: P30f) */
    rad_buf q8 = 0, s8 = 0;
    /* tp8 loop P30f: o_b over IQR attention_output_a groups -- one quantiser a group (`parts[j]`, over `psrc[j]`: the
     * group's own buffer, or attn_o_lora's column slice for a GGUF group when `pcol[j]`), each writing its `pk` columns of
     * the pair (a per-(row, 128-column) quantiser: the codes and scales of the concatenation, byte for byte); `apitch` =
     * the declared pitch of the activation the caller hands lin_issue (its row offset is read off it) */
    std::vector<rad_op>  parts;
    std::vector<rad_buf> psrc;
    std::vector<char>    pcol;
    int64_t pk = 0, apitch = 0;
    /* tp8 loop P30o: the pair's K elements a scale (32 for the release's block FP8, else RAD_FP8_BLOCK) */
    int64_t grp = RAD_FP8_BLOCK;
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
    /* Pictures (rad_block_ds41_vision.h), when the container carries the tower and the engine asked
     * for an encoder budget. The tower runs on encoder passes; a trunk step with media rows writes
     * them over the token embeddings (MediaRows, before the vocabulary all-reduce), gives those rows
     * DeepSeek-V4.1's image routing bias (MoeFP8::Vl, each MoE layer's bias_vl) and keeps them out of
     * the Engram (model.py: image tokens take no part in an n-gram and get no Engram contribution):
     * their compressed token is the pad the n-gram look-back stops at (`mm_eos` scattered into
     * engram_tokens), and their Engram value rows are zero (`mm_zero` scattered into engram_kv's
     * value columns), so the gate adds nothing. A step without media rows issues none of it. */
    bool media = false;
    Ds41Vision vit{};
    MediaRows mrows{};
    rad_op mm_iota = 0, mm_tok_put = 0, mm_zero_fill = 0, mm_kv_put = 0;
    rad_buf mm_eos = 0, mm_zero = 0;
    rad_op vl_gather = 0, vl_scatter = 0;
    rad_buf vl_ids = 0, vl_ew = 0, vl_tid = 0, vl_tew = 0;
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
    /* Lever DKV (dkv_read_env): lu [T][lu_e hd] (the plan's rows of a prefill pass), selkv [selkv_rows][itopk hd] (a decode
     * pass's gathered selections); 0: not declared. */
    rad_buf lu = 0, selkv = 0;
    int64_t lu_e = 0, selkv_rows = 0;
    /* tp8 loop D25 (RADIANCE_DS41_DKV_GPIECE=1): selkv's rows (each band piece gathers at its own rows); = selkv_rows otherwise */
    int64_t selkv_cap = 0;
    /* tp8 loop D25 step 3 (RADIANCE_DS41_DKV_GPIECE=2): the gather and its attentions are declared to selkv_cap rows with libr4d's
     * per-op band 64, so a decode segment of 49 .. 64 rows runs them ONCE instead of in F58's band pieces */
    bool selkv_one = false;
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
    /* Lever 1U-ETP (etp_env, geometry()): the trunk's routed experts as width slices of every expert on every rank */
    bool etp = false;
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
    /* Lever 1U-E2: a rank's column part of the kv rows [e2_rows, e2_part] (bf16), the part's width (N / world), and a
     * weight row's stored bytes (the view's row step); 0 where not declared. */
    rad_buf ekv_p = 0;
    int64_t e2_rows = 0, e2_part = 0, e2_rb = 0;
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
    /* tp8 loop P30o: an E4M3 weight with an E8M0 exponent per 32 x 32 block (the release's block FP8, a weight sidecar of the
     * official dense bytes): its scale group along K (32); absent = RAD_FP8_BLOCK */
    std::map<rad_weight, int64_t>      wgrp;
    /* The quantiser each dense GEMM waits for (linear()): the op and the K its pair was declared at. */
    std::map<rad_op, IqrQOp>           qops;
    rad_buf iqr_q8 = 0, iqr_s8 = 0;   /* [max_tok, widest K] i8; [max_tok, widest K / 128] f32 */
    int64_t iqr_qk = 0;               /* iqr_q8's declared width (buffers(): the hidden width) */
    /* tp8 loop P30f: the pair of an IQR linear wider than iqr_qk (an int8 Engram kv, K = ple_heads x ple_dim = 6144 > 5120),
     * declared at the first such linear; its width */
    rad_buf iqr_q8w = 0, iqr_s8w = 0;
    int64_t iqr_qkw = 0;
    /* tp8 loop P30f: the IQR attention_output_a groups' own [max_tok, o_rank] outputs, one a group, shared by the layers */
    std::vector<rad_buf> oa_g;
    int oa_said = 0;
    /* tp8 loop P30t (RADIANCE_DS41_TWIN=mask): a GGUF dense weight's TWIN -- a weight sidecar's '<name>#pf' entry (rad-sidecar
     * --twin), an IQR form (int8 / the release's block FP8) resident BESIDE the GGUF weight -- and the twin's ops: `twop` maps a
     * GGUF linear's op to the twin's GEMM (whose quantiser is in `qops`), `twcls` the twin GEMM's class bit (twin_class);
     * lin_issue takes the twin for a PROMPT segment's rows (t_twin, set by step_bi's by-segment issue), decode rows stay GGUF */
    std::map<rad_weight, rad_weight> twin;
    std::map<rad_weight, int>        twwcls;
    std::map<rad_op, rad_op>         twop;
    std::map<rad_op, int>            twcls;
    int twin_mask = 0;                /* the classes some layer here twinned */
    /* tp8 loop P30o: the fp8 activation pair the release's block-FP8 linears read (quant_act_fp8_ue8m0: E4M3 codes, a power-of-two
     * f32 scale per 32 of K) -- [max_tok, the hidden width] and [max_tok, width / 32], declared at the first such linear; a wider
     * one (the Engram kv) beside it */
    rad_buf f8_q = 0, f8_s = 0, f8_qw = 0, f8_sw = 0;
    int64_t f8_qkw = 0;
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
    /* tp8 loop F58 (RADIANCE_DS41_BI_DBAND=1): a decode band piece's cu, cu[s0 .. s1] - cu[s0] (row j = piece j, at most
     * kBiBand + 1 words), and its sub_i32; 0 unless declared */
    rad_buf bi_dcu = 0;
    rad_op bi_dsub = 0;
    /* A sizing declare (RadBuildCtx::shape_probe): the real declare's model of this rank; null in the real declare. */
    const Model* sized_against = nullptr;
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
    /* tp8 loop P13 (RADIANCE_DS41_PIPE2=N, pipe2_env(); read at declare): an uncut layer of a lone prefill pass of >= `pipe2`
     * rows runs its MoE in two row pieces, piece 0's all-reduce on the second lane beside piece 1's MoE (step_bi,
     * pipe2_cut()). 0 = off: nothing new declared or issued. */
    int64_t pipe2 = 0;
    /* tp8 loop P13b (RADIANCE_DS41_PIPE2_ATTN=1, with pipe2): the same layers' attention all-reduce in the two pieces too,
     * piece 0's on the second lane beside piece 1's output projection (step_bi) */
    bool pipe2_attn = false;
    /* tp8 loop P13c (RADIANCE_DS41_PIPE2_CONE=1, with pipe2): PSK's cut (cone) layers too, where the layer's rows are one prefill
     * segment of >= pipe2 rows (the drain's last chunks: the cone rows of layers 21 .. ~32 at pipe2 1024); not the split layer */
    bool pipe2_cone = false;
    /* tp8 loop P15 (RADIANCE_DS41_PIPE2_TBO=1 / 2 / 3, with pipe2 + pipe2_attn, every rank attending): the other two piece
     * all-reduces of a pipelined layer on the second lane too -- bit 1 piece 1's attention all-reduce beside MoE pass 0,
     * bit 2 piece 1's MoE all-reduce beside the next layer's piece-0 per-row work (step_bi, tbo_*); 0 = off */
    int pipe2_tbo = 0;
    /* tp8 loop P20 round 2 (RADIANCE_DS41_SPHC=N, sphc_env(); read at declare): a lone prefill pass of >= `sphc` rows runs
     * the mHC stream ops of its uncut layers sequence-parallel -- each rank the per-row ops over its quarter of the rows
     * between a reduce-scatter (the sublayer's all-reduce's first half) and an all-gather of the normed input (its second
     * half) -- (step_bi, sp_*). 0 = off: nothing new declared or issued. `sphc_eng`: an Engram layer's lookup / copy / kv GEMM
     * over the quarter too (RADIANCE_DS41_SPHC_ENGRAM, default on). The ops (fused(): 0 where no library serves them):
     * reduce-scatters over attn_out and the MoE's delta (act.x), all-gathers of the normed inputs (xn, act.x) and of the
     * streams (h0, at a non-SP layer after SP ones), all raw, on the all-reduce's instance. */
    int64_t sphc = 0;
    bool sphc_eng = false;
    rad_op sp_rs_a = 0, sp_rs_f = 0, sp_ag_x = 0, sp_ag_f = 0, sp_ag_h = 0;
    /* tp8 loop P20 round 3 (RADIANCE_DS41_SPHC_X1=1, with sphc, the leader at h > 0): X1's stage_send with its residual parts
     * row-sharded (`shard`: every rank writes its quarter rows, rank 0 the rest) -- the leader's last head layer then SP too,
     * no gather of the streams before X1. Round 4, the follower at h > 0: X1's stage_recv with `shard` (each rank copies
     * only its quarter rows of those parts: its SP layer h reads no other) */
    rad_op pp_x1_sh = 0;
    /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG=1 / 2, with sphc, the follower with segment boundaries; sphc_seg_env()): a
     * follower SEGMENT pass of one prefill segment (a BG newcomer's body in one-layer segments, other lanes' passes between
     * them) is SP over the layers the whole follower part's plan makes SP -- every segment computes that plan from the batch
     * and the declare, so the two sides of a boundary agree: after an SP layer the boundary holds the streams row-sharded,
     * and the next segment issues its boundary op over this rank's quarter (as a whole pass does at the layer's end). 1: the
     * boundary saved / restored whole (today's bytes: the quarter is in them); 2: row-sharded (the stage ops' loop mode with
     * `shard` + `ranks`: each rank its quarter rows of act.x / h1 / post / comb / pre_ffn, the rest whole: seg_*_sh). 0 = off.
     * 3 (P25b): + a layer past PSK's first whose cone keeps every row of the segment pass is planned uncut. 4 (P25c, PREFILL
     * round 18): + the same for the follower's LONE prefill passes (a cold prompt's last chunk: no segment, X1 in, X2 out),
     * which are SP from round 4 only up to PSK's first layer. */
    int sp_seg = 0;
    bool sp_lone = false;   /* P25c declared here (RADIANCE_DS41_SPHC_SEG=4 on the follower; kept if boundaries fall back to 1) */
    std::vector<rad_op> seg_send_sh, seg_recv_sh;
    /* tp8 loop P26 (RADIANCE_DS41_SPHC_ATTN=1, with sphc; sphc_attn_env(), read at declare): at an SP layer whose xn no
     * compressor / indexer reads, the attention's unsharded input GEMMs (q_a, kv: every rank holds them whole) over this
     * rank's quarter, then all-gathers of their outputs (qr, kv) in place of xn's -- the gathers raw, on the all-reduce's
     * instance (`bound`). 0 = off: nothing new declared or issued. */
    rad_op sp_ag_qr = 0, sp_ag_kv = 0;
    /* tp8 loop P26b (RADIANCE_DS41_SPHC_ATTN=2): at an SP kv-source layer the compressor's kv | gate GEMM over the quarter and
     * its f32 output kvs gathered raw (libr4d sp_ag_rows_4rank_f32: the words on the bf16 instance) */
    rad_op sp_ag_kvs = 0;
    /* tp8 S3 (ds41_layers.h): this node's layer ranges (RADIANCE_DS41_LAYERS, _MODE); lr.on false = every layer, the
     * published engine. `lr_deps`: what its layers read that another node computes (strict mode: the copies declared). */
    ds41layers::Spec lr{};
    std::vector<ds41layers::Dep> lr_deps;
    /* tp8 S4: a node of a two-node pipeline (rad_pp_info): `lr` is then the node's strict part of W(pp_h, pp_t) (the
     * leader layers [0, h) and [t, n) with the embedding, the head and the drafter; the follower [h, t)), and the step
     * runs it with the crossings: X1 = the embedding's output (xcol) into the follower's prologue, X2 = the inputs of the
     * boundary op at layer t (ds41_cut_liveset: act.x, h1, post, comb, pre_f) back to the leader, which issues that op
     * as one process does. Off (pp false) on one machine. */
    bool pp = false;
    int  pp_rank = 0, pp_h = 0, pp_t = 0;
    rad_op pp_x1 = 0, pp_x2 = 0;          /* this node's stage op of each crossing */
    rad_op pp_post = 0;                   /* the leader: layer t-1's mhc_post where no fused pcn_a resolves */
    RadPpRing* pp_ring[RAD_PP_N_RINGS] = {};
    Dsp  pp_dsp{};                        /* the follower: the drafter it does not declare (its KV group is zeroed) */
    std::vector<int64_t> pp_zero_r;       /* the ratios whose index groups this node zeroed (pp_copy_layer) */
    std::vector<int64_t> pp_zero_cr;      /* ... whose latent groups this node zeroed */
    /* tp8 S6: a general cut W(h, t) (ds41_layers.h crossings(): both nodes hold both crossings' descriptions). X1 enters
     * the follower's layer h, X2 the leader's layer t; each carries the boundary op's inputs (X1 at h = 0: xcol) and what
     * the receiving layers read of the sending node: one kv source's new entries, which the receiver stores into its copy
     * of that source's cache with the same entry_store at the same pages (pp_cp_*), the selections of the last index layer
     * before the boundary, the candidate layer's keep words. `pp_binds`: the source layers this node binds (held or
     * copied). `pp_unsplit_cand`: the candidate layer's indexer runs unsplit on its node when its keep words cross, so
     * the one sending rank holds every row's (the split keeps a run's keep rows on the rank that ran it). */
    ds41layers::Crossing pp_cx[2]{};
    ds41layers::Binds pp_binds{};
    rad_op pp_cp_lat = 0, pp_cp_idx = 0;  /* the receiver's copy stores (the crossing into this node) */
    bool pp_unsplit_cand = false;
    /* tp8 dev 1a (RADIANCE_PP_FOLLOWER_EMBED=1 on both nodes, W(0, t) only; unset: nothing here): the follower looks the
     * step's embedding up itself -- the same table and op as the leader's, the vocab-sharded rows completed by the same
     * exact all-reduce, into xcol, where it took X1 -- and the leader sends no X1 (its own lookup stays: its xcol is what
     * the prologue's expand reads at every part, as before). A pipeline node serves no pictures (lr_check), so xcol is the
     * lookup alone on both. */
    bool pp_fe = false;
    /* tp8 dev D10 round 2 (RADIANCE_PP_ISSUE_AHEAD=1 on both nodes, W(0, t) only; unset: nothing here): X1 carries the step's
     * token ids as its last part (the sender's batch token ids on its card: for a step the leader issued ahead of its collect,
     * ADV_TRUNK's, behind the sampler), and the follower takes X1 at the top of its pass, into its own batch's token ids,
     * before its lookup (pp_fe) and its Engram layers' token_map read them -- its frame's ids are the host's, which a step
     * issued ahead does not know yet. Under pp_fe X1 is the ids alone (4 B a row), else the lookup + the ids. At h > 0 (only
     * where the follower holds an Engram layer) X1 stays where it is, at layer h, with the ids as its last part, and the
     * follower's token-id ops (token_map, its Engram layers' ngram_ids, lever E1's mark) are issued right after it instead
     * of in the prologue (pp_ids_late): the same ops on the same inputs, later on the same stream, inside ctok's live range. */
    bool pp_ids = false;
    /* tp8 dev M4 (RADIANCE_PP_SEG_LAYERS=k on the follower, --pp-lanes > 1; unset: nothing here): the follower's [h, t) in
     * segments of k layers. A boundary s is a cut on this node: the segment before it saves what X1 of W(s, t) would carry
     * into layer s -- the boundary op's inputs, the selections, the keep words; no kv entries, the node holds its caches and
     * copies -- into this rank's ring buffer (the stage ops' loop mode: each rank its own bytes back), and the segment from s
     * takes them back and issues the boundary op as a receiving node does. `seg_cx`: X1 of W(s, t) without its source, plus
     * what this pass's X2 ships that layers below s computed -- its kv source's entries over the work rows, its selections /
     * keep words below s (ds41layers::segment_crossing; loop FINDINGS F12: without them a pass between the segments rewrote
     * the source planes X2 reads at t). */
    std::vector<int32_t> seg_b;
    std::vector<ds41layers::Crossing> seg_cx;
    std::vector<rad_op> seg_send, seg_recv, seg_post;
    rad_buf seg_ring = 0;
    int64_t seg_ring_bytes = 0;
    /* tp8 dev M9 (RADIANCE_PP_LEAD_SEG=k on the leader): seg_b / seg_cx are the leader's -- head boundaries in (0, h), tail
     * ones in (t, n) (ds41layers::lead_segment_crossing) -- and a head / tail part with seg_l1 > 0 issues one segment */
    bool seg_lead = false;
    /* tp8 loop P28 (RADIANCE_DS41_PP_BAND=k on both nodes, --pp-lanes > 1; unset: nothing here): the newcomer's own cut, the
     * c32 direction. W(h, t) decodes as always; a band pass (a BG step of fresh prompts, the engine's choice) runs the
     * leader's head over [0, h + k) and the follower from h + k (ds41layers::band_crossing: its X1 = W(h + k, t)'s + the
     * follower's source copy + the band layers' window keys). Round 2a: the band is parsed and checked on both nodes and
     * the leader DECLARES the band layers (weights, ops, their window binding) -- no pass issues them yet (both step paths
     * skip a layer that is not m.lr's, and m.lr stays the decode set). `band_decl`: per layer, declared for the band here. */
    int band_k = 0;
    ds41layers::Band band{};
    std::vector<char> band_decl;
    /* round 2b (declare only): each band layer's window-key rows (persist, both nodes), the leader's copies of them out of
     * window_kv_normed after the layer's key chain (a bf16 cast, byte copy), the band X1 (the leader's send after layer h + k - 1,
     * the follower's receive before layer h + k) and the follower's window stores of the received rows (window_key_store, the
     * owner's op, issued into band layer j's window pages) */
    std::vector<rad_buf> band_key;
    std::vector<rad_op> band_cp, band_st;
    rad_op pp_x1b = 0;
};
static std::unique_ptr<Model> models[MAX_RANKS];
/* The model whose name map is being declared (declare() on this thread). */
static thread_local Model* naming = nullptr;

/* ------------------------------------------------------------------ tp8 S3: the node's layer ranges (ds41_layers.h) */
/* The layer the step runs after `l`: l + 1, or (layer ranges) the next layer this node holds; null after the last. */
/* A formatted std::string (the node lines). */
static std::string fmt_s(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt_s(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}
/* tp8 loop P28 round 2c: the pass being issued is a band pass (pp_part reads the engine's band word, RadPpInfo.reserved[0]:
 * the band's k, 0 none); set per pass, read by the issue helpers below */
static thread_local bool t_band = false;
static const Layer* next_layer(const Model& m, int l) {
    if (!m.lr.on) return l + 1 < m.c.n_layer ? &m.layers[(size_t)l + 1] : nullptr;
    /* tp8 loop P28 round 2c: a band pass's leader runs [h - 1, h + k - 1) into the band's next layer, as one process */
    if (t_band && m.pp_rank == 0 && m.band_k > 0 && l >= m.band.h - 1 && l < m.band.at - 1) return &m.layers[(size_t)l + 1];
    const int n = m.lr.next[(size_t)l];
    return n >= 0 ? &m.layers[(size_t)n] : nullptr;
}
/* The embedding, the final norm and the head (and the logits the engine samples): every layer, a proxy node (it runs
 * alone), or the leader of a strict one. */
static bool lr_shell(const Model& m) { return !m.lr.on || !m.lr.strict || m.lr.leader; }
/* The Engram tables and token map: every layer, or a node that holds an Engram layer. */
static bool lr_engram(const Model& m) {
    if (!m.lr.on) return true;
    for (int64_t l : m.c.ple)
        if (m.lr.held((int)l)) return true;
    return false;
}
/* Proxy: the node's first layer past 0 starts as layer 0 does (the embedding over the streams, one-hot gates), so its
 * attention connection is the plain collapse + norm, not the post of a layer the node does not run. */
static bool lr_synth_start(const Model& m, int l) { return m.lr.on && !m.lr.strict && l > 0 && l == m.lr.first; }
/* The index plane's keep words: some held layer is the candidate layer or an index layer past it. */
static bool lr_keep_used(const Model& m) {
    if (!m.lr.on) return true;
    const auto& c = m.c;
    for (int l = 0; l < (int)c.n_layer; ++l)
        if (m.lr.held(l) && c.cand >= 0 && (l == c.cand || (c.is_idx(l) && l > c.cand))) return true;
    return false;
}
static ds41layers::Geo lr_geo(const Model& m) {
    const auto& c = m.c;
    ds41layers::Geo g;
    g.n_layer = (int)c.n_layer;
    g.ratio = c.ratio;
    g.src_of = c.src_of;
    g.idx_of = c.idx_of;
    g.is_src.assign((size_t)c.n_layer, 0);
    g.is_idx.assign((size_t)c.n_layer, 0);
    for (int l = 0; l < (int)c.n_layer; ++l) {
        g.is_src[(size_t)l] = c.is_src(l) ? 1 : 0;
        g.is_idx[(size_t)l] = c.is_idx(l) ? 1 : 0;
    }
    g.cand = (int)c.cand;
    for (int64_t l : c.ple) g.ple.push_back((int)l);
    /* (a pipeline follower keeps the leader's drafter in pp_dsp: the taps are the model's either way) */
    for (int64_t t : (m.dsp.taps.empty() ? m.pp_dsp.taps : m.dsp.taps)) g.taps.push_back((int)t);
    return g;
}

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
/* tp8 loop P30t (QUEUE P30t, FINDINGS F65): the int8 / fp8 TWIN -- RADIANCE_DS41_TWIN=mask declares, beside a GGUF dense
 * linear, the IQR form a weight sidecar holds under '<name>#pf' (rad-sidecar --twin: the engine serves it BESIDE the container's
 * <name>, both resident), and a PROMPT segment's rows of that linear run on the twin (its quantiser + gemm_nt_q) while decode rows
 * keep the published GGUF launch (no change to a decode row's bytes by construction). Classes (mask bits): 1 = attn_q_a +
 * attn_kv, 2 = attn_q_b, 4 = attn_output_a (every group) + attn_output_b, 8 = engram_kv (lever E1's dedup issue keeps GGUF), 16 =
 * the shared expert's gate_up + down (MoeFP8 Config::shared_twin: plain int8 twins, the arm on lane 0 for a twin pass), 32 = any
 * other dense name. RADIANCE_DS41_TWIN_ROWS=N (default 1): a prompt segment of fewer rows keeps GGUF. In process: fn4
 * ds41_twin = mask + 256 x rows (rows 0 = the env's), a subset of the env's mask (0 = every row GGUF = the published program).
 * A prompt row's bytes are then its own segment's (M = the segment's rows, issued by segment; a segment's sub-range -- an SP
 * quarter, a P13 / P15 piece -- keeps GGUF), so they do not depend on the other rows of the pass: rule B (own reference). */
static constexpr const char kTwinSfx[] = "#pf";
enum : int { kTwQkv = 1, kTwQb = 2, kTwOab = 4, kTwEng = 8, kTwShx = 16, kTwOther = 32, kTwAll = 63 };
static int twin_env() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_TWIN");
        if (!e || !*e) return 0;
        const long x = std::strtol(e, nullptr, 0);
        return x > 0 && x <= kTwAll ? (int)x : 0;
    }();
    return v;
}
static int64_t twin_rows_env() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_TWIN_ROWS");
        const long long x = e && *e ? std::strtoll(e, nullptr, 10) : 1;
        return (int64_t)(x >= 1 && x <= 65535 ? x : 1);
    }();
    return v;
}
static std::atomic<int> g_sw_twin{-1};      /* fn4 ds41_twin's value; -1 = the env's */
static std::atomic<int> g_twin_decl{0};     /* the classes some declare here twinned */
static std::atomic<long long> g_tw_launch{0}, g_tw_rows{0};
static thread_local int t_twin = 0;          /* the twin classes the segment being issued takes (TwinScope) */
static thread_local bool t_twin_sub = false; /* issuing a sub-range of a segment (step_bi's with_rows): no twin */
/* the classes the layer being issued keeps on GGUF in every pass (step_bi, at each layer): where a lone pass may issue a class's
 * GEMM over a SUB-range of its segment (an SP quarter: P26's q_a / kv, the SP Engram kv; a P13 / P15 piece) that class stays
 * GGUF in every context, so a prompt's bytes do not depend on whether it ran alone or beside other prompts (rule B) */
static thread_local int t_tw_off = 0;
static int twin_class(const char* name) {
    if (!name) return kTwOther;
    if (std::strstr(name, "indexer.")) return kTwOther;
    if (std::strstr(name, ".attn_q_a.") || std::strstr(name, ".attn_kv.")) return kTwQkv;
    if (std::strstr(name, ".attn_q_b.")) return kTwQb;
    if (std::strstr(name, ".attn_output_a") || std::strstr(name, ".attn_output_b.")) return kTwOab;
    if (std::strstr(name, ".engram_kv.")) return kTwEng;
    if (std::strstr(name, "_shexp.")) return kTwShx;
    return kTwOther;
}
static int twin_bi_now();   /* ds41_bi's value (below g_sw_bi): no twin under ds41_bi 0, where step() issues every pass */
static int twin_mask_now() {
    if (twin_bi_now() <= 0) return 0;
    const int d = g_twin_decl.load(std::memory_order_relaxed);
    const int v = g_sw_twin.load(std::memory_order_relaxed);
    return (v < 0 ? twin_env() : (v & 255)) & d;
}
static int64_t twin_rows_now() {
    const int v = g_sw_twin.load(std::memory_order_relaxed);
    const int r = v < 0 ? 0 : (v >> 8);
    return r > 0 ? r : twin_rows_env();
}
/* The twin classes for the rows a by-segment launch issues: a prompt segment (not decode) of at least twin_rows_now() rows,
 * outside a sub-range; restored at the scope's end so nothing outside a segment's launch takes the twin. */
/* the shared expert's twin for the MoE pass being issued (MoeFP8::twin_now) */
static bool twin_shx_now() { return (t_twin & kTwShx) != 0; }
struct TwinScope {
    int prev;
    TwinScope(bool prompt, int64_t n) : prev(t_twin) {
        t_twin = 0;
        if (prompt && !t_twin_sub) {
            const int mk = twin_mask_now() & ~t_tw_off;
            if (mk && n >= twin_rows_now()) t_twin = mk;
        }
    }
    ~TwinScope() { t_twin = prev; }
    TwinScope(const TwinScope&) = delete;
    TwinScope& operator=(const TwinScope&) = delete;
};
static rad_weight lw(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                     RadWeightGroup grp, int shard = RAD_SHARD_NONE, int64_t lo = 0, int64_t hi = 0,
                     bool mapped = false, int access = RAD_ACCESS_PER_TOKEN);
/* A GGUF weight's twin, when the env asks for its class and a sidecar holds '<name>#pf' in an IQR form: declared as its own
 * weight (lw() over that name) and recorded in Model::twin. False: the twin exists but cannot be served (refused at declare). */
static bool twin_decl(RadBuilder* b, Model& m, rad_weight w, const char* name, int64_t n, int64_t k, RadWeightGroup grp,
                      int shard, int64_t lo, int64_t hi, int access) {
    const int cls = twin_class(name);
    if (!(twin_env() & cls)) return true;
    /* the trunk's layers only (the drafter's blk.40+ linears are DLW's, never twinned) */
    int bl = -1;
    if (std::sscanf(name, "blk.%d.", &bl) == 1 && (bl < 0 || bl >= (int)m.c.n_layer)) return true;
    const std::string tn = std::string(name) + kTwinSfx;
    RadEncoding te{};
    if (!weight_enc(b, tn.c_str(), &te)) return true;   /* no twin: the GGUF weight alone, as today */
    const char* tname = m.nm.ckpt("%s", tn.c_str());   /* name is scoped already: no second prefix */
    const rad_weight tw = lw(b, m, tname, n, k, grp, shard, lo, hi, false, access);
    if (!tw || !m.wsc.count(tw) || m.wty.count(tw)) {
        fprintf(stderr, "radiance: deepseek41: tp8 loop P30t: %s's twin %s is not an IQR / block-FP8 form deepseek41 serves\n",
                name, tname);
        return false;
    }
    m.twin[w] = tw;
    m.twwcls[tw] = cls;
    return true;
}
static rad_weight lw(RadBuilder* b, Model& m, const char* name, int64_t n, int64_t k,
                     RadWeightGroup grp, int shard, int64_t lo, int64_t hi,
                     bool mapped, int access) {
    if (!mapped && map_copy(b, name, ckpt_name(name)) != RAD_OK) return 0;
    RadEncoding e{};
    int64_t blk = 0;
    const bool known = weight_enc(b, name, &e);
    const char* ty = known ? ggml_type_of(e, &blk) : nullptr;
    const bool split = m.g.world > 1 && shard != RAD_SHARD_NONE;
    /* tp8 loop P30o: the release's block FP8 (a sidecar of the official dense bytes: affine {codes fp8_e4m3 [1x1], scale e8m0
     * [32x32]}, what rad-convert's importer gives DeepSeek-V4's E4M3 linears): the codes as stored and the E8M0 block exponents,
     * read by gemm_nt_q dtype fp8a8e8 after quant_act_fp8_ue8m0 (the release's act_quant(x, 32, 'ue8m0')). A rank's span must
     * be whole 32-blocks on the split axis. */
    if (!ty && known && rad_enc_is(&e, "affine") && e.n_planes == 2 && e.plane[0].dtype == RAD_F8E4M3) {
        const RadEncPlane* sp = rad_enc_plane(&e, "scale");
        if (sp && sp->dtype == RAD_E8M0 && sp->block[0] == 32 && sp->block[1] == 32) {
            const int64_t G = 32;
            if (n % G || k % G || (split && (lo % G || hi % G))) {
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30o: %s is fp8_e4m3*e8m0[32x32]; this rank reads %lldx%lld%s, "
                                "which cuts a 32 x 32 scale block\n", name, (long long)n, (long long)k,
                        split ? " of a slice" : "");
                return 0;
            }
            std::string base(name);
            if (base.size() > 6 && base.compare(base.size() - 6, 6, ".weight") == 0) base.resize(base.size() - 6);
            const char* sn = m.nm.f("%s.scale", base.c_str());
            const rad_weight w = decl_view(b, name, nullptr, "codes", RAD_F8E4M3, { n, k }, access,
                                           split ? shard : RAD_SHARD_NONE, grp);
            const rad_weight ws = decl_view(b, sn, name, "scale", RAD_E8M0, { n / G, k / G }, access,
                                            split ? shard : RAD_SHARD_NONE, grp);
            if (!w || !ws) {
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30o: block-FP8 weight %s refused\n", name);
                return 0;
            }
            if (split && (rad_weight_shard_span(b, w, lo, hi) != RAD_OK ||
                          rad_weight_shard_span(b, ws, lo / G, hi / G) != RAD_OK)) {
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30o: %s's span [%lld, %lld) refused\n", name,
                        (long long)lo, (long long)hi);
                return 0;
            }
            m.wsc[w] = ws;
            m.wcod[w] = RAD_F8E4M3;
            m.wdt[w] = "fp8a8e8";
            m.wrb[w] = 0;
            m.wgrp[w] = G;
            return w;
        }
    }
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
    if (ty && twin_env() && !twin_decl(b, m, w, name, n, k, grp, shard, lo, hi, access)) return 0;   /* tp8 loop P30t */
    return w;
}
/* y = x . w^T over lw()'s weight: for a GGUF block it is gemm_nt_ggml, for an IQR view gemm_nt_q
 * at the weight's code dtype over a rotated int8 activation pair, otherwise gemm_nt at bf16. */
/* An IQR linear's quantiser over n columns of x into the pair (q8, s8). */
static int64_t iqr_grp(const Model& m, rad_weight w) {
    const auto g = m.wgrp.find(w);
    return g == m.wgrp.end() ? (int64_t)RAD_FP8_BLOCK : g->second;
}
static bool iqr_fp8(const Model& m, rad_weight w) { return std::strcmp(m.wdt.at(w), "fp8a8e8") == 0; }
static rad_op iqr_quant(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n, rad_buf x,
                        rad_buf q8, rad_buf s8) {
    /* tp8 loop P30o: the release's block FP8 reads the release's act_quant */
    const bool f8 = iqr_fp8(m, w);
    const bool rot = !f8 && std::strcmp(m.wdt.at(w), "i8a8") != 0;
    return op(b, f8 ? "quant_act_fp8_ue8m0" : rot ? "had_quant_act_i8g" : "quant_act_i8g",
              { RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("group", iqr_grp(m, w)), RAD_STR("dtype", "bf16") },
              {}, { x }, { q8, s8 });
}
/* An IQR linear's GEMM over the pair (q8, s8) into y. */
static rad_op iqr_gemm(RadBuilder* b, const Model& m, int64_t rows, rad_weight w, int64_t n, int64_t k,
                       rad_buf q8, rad_buf s8, rad_buf y) {
    std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k),
                                 RAD_INT("group", iqr_grp(m, w)), RAD_STR("dtype", m.wdt.at(w)) };
    const int64_t wrb = m.wrb.at(w);
    if (wrb > 1) ps.push_back(RAD_INT("w_rb", wrb));
    std::vector<rad_weight> wts = { w, m.wsc.at(w) };
    return op(b, "gemm_nt_q", ps, wts, { q8, s8 }, { y });
}
/* The pair an IQR linear of K = k reads: the model's (the hidden width), or -- tp8 loop P30f -- for a wider linear (an int8
 * Engram kv: K = ple_heads x ple_dim = 6144 > 5120) its own pair, declared at the first such linear. Before P30f a wider
 * linear sliced k columns out of the hidden-width pair: rows of a [rows, k] view at the pair's pitch overlap (a silent
 * defect, never served: the 2.25 file's dense are GGUF). False: refused. */
static bool iqr_pair(RadBuilder* b, Model& m, int64_t k, rad_buf* q8, rad_buf* s8, rad_weight w = 0) {
    if (w && iqr_fp8(m, w)) {   /* tp8 loop P30o: the fp8 pair, at the hidden width or wider */
        const int64_t G = iqr_grp(m, w);
        const bool wide = k > m.iqr_qk;
        rad_buf& q = wide ? m.f8_qw : m.f8_q;
        rad_buf& s = wide ? m.f8_sw : m.f8_s;
        if (!q) {
            const int64_t wk = wide ? k : m.iqr_qk;
            q = decl_b(b, m.nm.f(wide ? "f8_q_wide" : "f8_q"), RAD_F8E4M3, { m.g.max_tok, wk });
            s = decl_b(b, m.nm.f(wide ? "f8_s_wide" : "f8_s"), RAD_F32, { m.g.max_tok, (wk + G - 1) / G });
            if (wide) m.f8_qkw = k;
            if (m.g.rank == 0)
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30o: the release's block-FP8 linears read an fp8 activation pair "
                                "%lld wide (E4M3 + a power-of-two scale a %lld)\n", (long long)wk, (long long)G);
        }
        if (!q || !s || (wide && k > m.f8_qkw)) {
            fprintf(stderr, "radiance: deepseek41: tp8 loop P30o: no fp8 activation pair for a linear of K %lld\n", (long long)k);
            return false;
        }
        *q8 = q;
        *s8 = s;
        return true;
    }
    if (k <= m.iqr_qk) {
        *q8 = m.iqr_q8;
        *s8 = m.iqr_s8;
        return true;
    }
    if (!m.iqr_q8w) {
        m.iqr_q8w = decl_b(b, m.nm.f("iqr_q8_wide"), RAD_I8, { m.g.max_tok, k });
        m.iqr_s8w = decl_b(b, m.nm.f("iqr_s8_wide"), RAD_F32, { m.g.max_tok, fp8_blocks(k) });
        m.iqr_qkw = k;
        if (m.g.rank == 0)
            fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: an IQR linear of K %lld reads its own activation pair "
                            "(the model's is %lld wide)\n", (long long)k, (long long)m.iqr_qk);
    }
    if (!m.iqr_q8w || !m.iqr_s8w || k > m.iqr_qkw) {
        fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: no activation pair for an IQR linear of K %lld (the wide "
                        "pair is %lld)\n", (long long)k, (long long)m.iqr_qkw);
        return false;
    }
    *q8 = m.iqr_q8w;
    *s8 = m.iqr_s8w;
    return true;
}
static rad_op linear(RadBuilder* b, Model& m, int64_t rows, rad_weight w, int64_t n,
                     int64_t k, rad_buf x, rad_buf y, bool twin_ok = true) {
    const auto s = m.wsc.find(w);
    if (s != m.wsc.end()) {
        rad_buf q8 = 0, s8 = 0;
        if (!iqr_pair(b, m, k, &q8, &s8, w)) return 0;
        const rad_op qop = iqr_quant(b, m, rows, w, k, x, q8, s8);
        const rad_op h = iqr_gemm(b, m, rows, w, n, k, q8, s8, y);
        if (qop && h) {
            IqrQOp e;
            e.op = qop;
            e.K = k;
            e.q8 = q8;
            e.s8 = s8;
            e.grp = iqr_grp(m, w);
            m.qops[h] = e;
        }
        return h;
    }
    const auto t = m.wty.find(w);
    if (t != m.wty.end()) {
        const rad_op h = op(b, "gemm_nt_ggml",
                            { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("type", t->second),
                              RAD_STR("dtype", m.g.dtype) },
                            { w }, { x }, { y });
        /* tp8 loop P30t: the twin's quantiser + GEMM beside it, over the same x into the same y (o_a / o_b: their caller) */
        const auto tw = twin_ok && h ? m.twin.find(w) : m.twin.end();
        if (tw != m.twin.end()) {
            const rad_op th = linear(b, m, rows, tw->second, n, k, x, y, false);
            if (!th) return 0;
            m.twop[h] = th;
            m.twcls[th] = m.twwcls.at(tw->second);
            m.twin_mask |= m.twcls[th];
        }
        return h;
    }
    return op(b, "gemm_nt",
              { RAD_RANGE("M", 1, rows), RAD_INT("N", n), RAD_INT("K", k), RAD_STR("dtype", m.g.dtype) },
              { w }, { x }, { y });
}
/* Issue a dense linear: its quantiser first when lw() found an IQR weight, then its GEMM. The pair
 * is one buffer for the whole model, so the quantiser and GEMM read the same columns of it.
 * gemm_nt_q's schema is a, a_scale, b, b_scale, y, then optional a_sum and b_ref (7 slots, 5
 * required). Issuing the bf16 list (activation, weight, y) is a step-0 schema refusal. i8a8 and
 * i8a8h do not read the trailing pair; pass them as nulls so a later grid fails closed. */
/* tp8 loop P30t: whether the launch of GGUF op h for the segment being issued takes its twin (t_twin's classes) */
static bool tw_take(const Model& m, rad_op h) {
    if (!t_twin) return false;
    const auto t = m.twop.find(h);
    if (t == m.twop.end()) return false;
    const auto c = m.twcls.find(t->second);
    return c != m.twcls.end() && (t_twin & c->second);
}
static void lin_issue(RadCtx* x, Model& m, rad_op h, int64_t rows, RadOperand a, rad_weight w,
                      RadOperand y) {
    if (t_twin && tw_take(m, h)) {   /* tp8 loop P30t: a prompt segment's rows on the twin */
        const auto tw = m.twin.find(w);
        if (tw != m.twin.end()) {
            h = m.twop.at(h);
            w = tw->second;
            if (rad_rank(x) == 0) {
                const long long nl = g_tw_launch.fetch_add(1, std::memory_order_relaxed) + 1;
                const long long nr = g_tw_rows.fetch_add(rows, std::memory_order_relaxed) + rows;
                if (nl == 1 || nl % 16384 == 0)
                    fprintf(stderr, "radiance: deepseek41: tp8 loop P30t: %lld twin launches (%lld rows) on prompt segments; "
                                    "ds41_twin mask %d, from %lld rows\n", nl, nr, twin_mask_now(), (long long)twin_rows_now());
            }
        }
    }
    const auto q = m.qops.find(h);
    if (q != m.qops.end()) {
        const IqrQOp& e = q->second;
        const int64_t K = e.K;
        const rad_buf pq = e.q8 ? e.q8 : m.iqr_q8, ps = e.s8 ? e.s8 : m.iqr_s8;
        const int64_t G = e.grp > 0 ? e.grp : (int64_t)RAD_FP8_BLOCK;   /* tp8 loop P30o: 32 for the release's block FP8 */
        const RadOperand q8 = bcol(pq, 0, K, rows);
        const RadOperand s8 = bcol(ps, 0, K / G, rows);
        if (!e.parts.empty()) {
            /* tp8 loop P30f: o_b over IQR attention_output_a groups -- each group's quantiser over its own columns of
             * the pair, at the rows `a` names (a row view of attn_o_lora: the row offset is read off it) */
            if (a.cols != 0 || e.apitch <= 0 || a.offset % e.apitch) {
                rad_step_fail(x, "deepseek41: tp8 loop P30f: o_b's grouped quantiser takes a row view of attn_o_lora");
                return;
            }
            const int64_t r0 = a.offset / e.apitch, pk = e.pk, pb = pk / G;
            for (size_t j = 0; j < e.parts.size(); ++j) {
                const int64_t c0 = (int64_t)j * pk;
                const RadOperand src = e.pcol[j] ? bcol_at(e.psrc[j], r0, e.apitch, c0, pk, rows)
                                     : r0 ? brow_slice(e.psrc[j], r0, rows, pk) : brows(e.psrc[j], rows);
                RAD_ISSUE_N(x, e.parts[j], rows, src, bcol(pq, c0, pk, rows), bcol(ps, c0 / G, pb, rows));
            }
        } else {
            RAD_ISSUE_N(x, e.op, rows, a, q8, s8);
        }
        const auto sc = m.wsc.find(w);
        RAD_ISSUE_N(x, h, rows, q8, s8, RAD_W(w), RAD_W(sc->second), y, RAD_NONE, RAD_NONE);
        return;
    }
    RAD_ISSUE_N(x, h, rows, a, RAD_W(w), y);
}
/* tp8 loop P30f: o_b's linear when attention_output_a groups are IQR (L.oa_dst): libr4d's int8 / fp8 GEMMs write C at pitch
 * N only (r4d_fp8_args.h fp8_ops), so an IQR group cannot write its column slice of attn_o_lora (`oa`, [T, k]: TP4's two
 * groups at pitch 2048; FINDINGS F63: gemm_i8a8_nt_m16 refused it at step 0) -- each writes its own [T, gk] buffer, and
 * o_b's activation quantiser runs once a group over that group's gk columns into the same columns of the pair. The
 * quantiser is per (row, 128 columns) and gk is a multiple of 128, so the codes and scales are the concatenation's byte for
 * byte, and the GEMM reads the pair as linear() declares it. A GGUF group keeps its column slice, read in place. An IQR
 * group in front of a GGUF o_b (which reads the concatenation) is refused here. */
static rad_op linear_oab(RadBuilder* b, Model& m, const Layer& L, int64_t rows, rad_weight w, int64_t n, int64_t k,
                         int64_t gk, rad_buf oa, rad_buf y, const std::vector<rad_buf>* groups = nullptr) {
    /* tp8 loop P30t: `groups` = the twin o_a groups' buffers (Model::oa_g), in place of L.oa_dst */
    const std::vector<rad_buf>& dst = groups ? *groups : L.oa_dst;
    if (!m.wsc.count(w)) {
        fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: layer %d's attention_output_a groups are IQR but its "
                        "attention_output_b is not (it reads the groups' concatenation): give both in the sidecar\n", L.l);
        return 0;
    }
    if (gk % iqr_grp(m, w) || (int64_t)dst.size() * gk != k) {
        fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: layer %d's o_b over %zu groups of %lld columns (K %lld) "
                        "cannot quantise a group at a time\n", L.l, dst.size(), (long long)gk, (long long)k);
        return 0;
    }
    rad_buf q8 = 0, s8 = 0;
    if (!iqr_pair(b, m, k, &q8, &s8, w)) return 0;
    IqrQOp e;
    e.K = k;
    e.q8 = q8;
    e.s8 = s8;
    e.grp = iqr_grp(m, w);
    e.pk = gk;
    e.apitch = k;
    for (size_t j = 0; j < dst.size(); ++j) {
        const rad_buf src = dst[j] ? dst[j] : oa;
        const rad_op qj = iqr_quant(b, m, rows, w, gk, src, q8, s8);
        if (!qj) return 0;
        e.parts.push_back(qj);
        e.psrc.push_back(src);
        e.pcol.push_back(dst[j] ? 0 : 1);
    }
    e.op = e.parts[0];
    const rad_op h = iqr_gemm(b, m, rows, w, n, k, q8, s8, y);
    if (h) m.qops[h] = e;
    return h;
}
static bool oa_split(const Layer& L) {
    for (const rad_buf d : L.oa_dst)
        if (d) return true;
    return false;
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
/* Pictures: the zero rows an Engram layer's media rows take their value columns from, a scatter at a time. */
static constexpr int64_t kMmZeroRows = 64;

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
/* Lever 1U-ETP (ds41-etp, 14.11.2026): RADIANCE_DS41_ETP=1 (or on) at start puts the trunk's routed EXL3 experts in
 * EXPERT TENSOR PARALLEL form instead of expert parallel: every rank holds a width slice of ALL n_expert experts of a
 * layer (gate_up by its intermediate 128-blocks, the gate rows and the same up rows; down by the same K 128-blocks)
 * instead of n_expert / world whole ones, and runs every routed slot of every token on its slice. Each rank's routed
 * partial is then a K slice of every slot's down, folded by the existing gather (a token's slots in ascending slot
 * order, f32, rounded to bf16 with the shared arm's) and summed by the existing exact all-reduce (four rank partials in
 * fp32, ascending rank order, one rounding). No collective is added and the all-reduce keeps its size.
 *
 * Slices (etp_slice): the intermediate width's nb 128-blocks (DeepSeek-V4.1-Flash: 2304 = 18) split by parity class
 * as the two-class expert ops take them (expert e at entry e / 2 of class e % 2): the even experts give the first
 * nb % world ranks one block more, the odd experts the last nb % world ranks -- at four ranks 5/5/4/4 blocks (640 /
 * 512 wide) for the even experts and 4/4/5/5 for the odd ones, 4.5 blocks a rank on average. So ranks 0-1 hold even
 * experts 640 wide and odd ones 512 wide, ranks 2-3 the other way round.
 *
 * Prefill, newcomer and mixed steps run on the same placement through the same op pair at slice width (the M > 16
 * forms take the same per-class widths as the decode band), so there is one stored form and nothing to switch.
 *
 * Exactness: a reference change against expert parallel, once. gate_up's outputs are whole dot products over K 5120
 * on one rank (an output 128-block's workgroup is the same code over the same codes), so h is the same; down's K
 * 2304 becomes four partial dot products (K 512..640) whose f16 / output-rotation / svh / bf16 roundings happen per
 * rank, the gather folds every slot's partial on every rank (EP: only the slots whose expert is the rank's own) and
 * the all-reduce sums four partials that each hold a quarter of all six slots (EP: each the whole of its own slots).
 * Then deterministic and batch-invariant like EP: every row's arithmetic is its own (per (token, slot) decode rows,
 * a token's slots in a fixed order, the rank fold in a fixed order).
 *
 * Unset / 0 / off: expert parallel, today's program and bytes. Anything else is refused at start. The DSpark drafter's
 * MoE stays expert parallel either way. */
static int etp_env() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_ETP");
        if (!e || !*e || !std::strcmp(e, "0") || !std::strcmp(e, "off")) return 0;
        if (!std::strcmp(e, "1") || !std::strcmp(e, "on")) return 1;
        return -1;
    }();
    return v;
}
using ds41etp::etp_slice;   /* ds41_etp.h: this rank's slice of an expert of each parity class */
/* Lever XR (ds41-etp-xr, 14.11.2026): RADIANCE_DS41_XRING=<MiB> (1 .. 1024) at start, with RADIANCE_DS41_ETP, declares
 * each trunk layer's EXL3 gate_up with interface v10's `xring` -- that much copy ring in its prefill scratch. exl3r4d's
 * map form then can (fn4 exl3_xr bit 2) copy the routed experts whose slices sit in the pinned host pool onto the card
 * inside the launch while the resident ones compute, and read the copies: the same bytes, a placement. The ring is
 * charged to the arena like the rest of the scratch (fewer expert slots). Unset / 0: nothing declared (today's ops);
 * anything else is refused at start. */
static int64_t xring_env() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_XRING");
        if (!e || !*e) return (int64_t)0;
        char* end = nullptr;
        const long n = std::strtol(e, &end, 10);
        if (!std::isdigit((unsigned char)e[0]) || (end && *end) || n < 0 || n > 1024) return (int64_t)-1;
        return (int64_t)n;
    }();
    return v;
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
/* tp8 loop P25s: the SP route's switches (ds41_sphc / ds41_sphc_attn / ds41_sphc_seg; below sphc_attn_env) */
static std::atomic<int> g_sw_sphc{0}, g_sw_sphc_decl{0};
static std::atomic<int> g_sw_spattn{-1}, g_sw_spattn_decl{0};
static std::atomic<int> g_sw_spseg{-1}, g_sw_spseg_decl{0};
static void sw_atomic_max(std::atomic<int>& a, int v) {
    int cur = a.load(std::memory_order_relaxed);
    while (cur < v && !a.compare_exchange_weak(cur, v)) {}
}
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
/* Lever DMR (DRAFTER track 18.11.2026 round 6): the DSpark query pass's mHC order. Switch "dspark_mhr" (fn4, default 0 =
 * today's issue; RADIANCE_DS41_DSPARK_MHR=1 starts at 1): 1 issues each connection's collapse + norm AHEAD of its mix
 * (col_a, norm_a, then mix_a; col_f, norm_f, then mix_f), so the mix -- read only by the connection's mhc_post, after
 * the attention / the MoE -- is no longer on the path to q_a / the router (lq146 c16ak, 8 sequences: mix_split +
 * mix_finish + their launch gaps ~28 us a connection on the critical path, 6 a pass). The same ops on the same
 * operands (col_a reads pre_f and h0, mix_a writes pre_a / post / comb; col_f reads pre_a and h1, mix_f writes pre_f /
 * post / comb: no buffer one writes is one the other reads), so the same bytes. Read at issue (a flip re-records).
 * 2 = 1 + each stage's window key (kv_n, kv_rope, kv_q, kv_st: four launches on the path to the attention) as the trunk's
 * lever K1 fold c, rmsnorm_rope_roundtrip_kv_store (each row at its own position, as kv_rope; the same bytes and slot). */
static std::atomic<int> g_sw_dmr{-1};
static int dmr_now() {
    int v = g_sw_dmr.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_DSPARK_MHR");
        int unread = -1;
        g_sw_dmr.compare_exchange_strong(unread, e && (!std::strcmp(e, "1") || !std::strcmp(e, "2")) ? e[0] - '0' : 0);
        v = g_sw_dmr.load(std::memory_order_relaxed);
    }
    return v;
}
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
static int twin_bi_now() { return g_sw_bi.load(std::memory_order_relaxed); }   /* tp8 loop P30t */
static void bi_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_BI");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1 : (e[0] >= '2' && e[0] <= '6' && !e[1]) ? e[0] - '0'
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
/* 5 / 6 = 3 / 4 with the MoE's all-reduce whole (lever A9, ds41-nc 12.11.2026, = ds41-bi 7cfb77b's BI-d on int12): every MoE
 * pass of a segmented step gathers its rows without reducing, then ONE all-reduce over the step's rows [0, T) (a step of
 * one pass keeps its own gather + all-reduce / ARGATHER, as today). Exact: the exact wire's all-reduce folds each element
 * 0 + r0 + r1 + r2 + r3 in fp32 in ascending rank order whatever the rows a launch carries and whoever owns the shard
 * (r4d_ar_twoshot_Nrank_exact.hip: "the bits match the one-shot's"), the gather is per row, and ARGATHER (a lone decode
 * step) is gather + all-reduce byte for byte (lever D3) -- so a row's bytes are its per-pass all-reduce's, i.e. ds41_bi
 * 3 / 4's, i.e. its alone bytes. Taken only where the block reduces its gathered delta in one collective (no moe /
 * shared all-reduce, combine or add; no consumer taking the all-reduce, no gather folded into it). */
/* 7 (lever PMM, ds41-pf2 17.11.2026) = 6 + the routed MoE of consecutive prefill segments as ONE pass: a segmented step's
 * prefill segments of at least kPmmMin rows each, whose rows follow one another (every uncut layer; a cut layer where a
 * later segment lies wholly in its cone), take one MoE pass over their rows together instead of a pass each -- so a host-
 * resident expert two prompts both route crosses the link once a layer, not once a prompt (s8mix: a MIXED step with a
 * second prefill segment read 6.6-8.4 GB a rank from host against 3.0-4.4 GB with one). Exact: every op of the pass at
 * prefill height writes a row's bytes whatever the pass's other rows -- the router chain's forms (FUSE-R: "every form writes
 * the separate ops' bytes at any row count"), the shared arm's dense GGUF GEMMs under ds41_wmk (one part order at every row
 * count), the EXL3 experts' prefill band above kBiMoe rows (a prompt's rows read the same bytes in any chunking: lever TI's
 * cold texts and turn-2 resumes IDENTICAL), the gather per row, the exact wire's all-reduce per element -- and each segment
 * alone is above every row band (kPmmMin >= kBiBand, kBiMoe). The decode segment keeps its pieces. */
static constexpr int64_t kPmmMin = 128;
static bool bi_pieces(int v) { return v == 2 || v == 4 || v == 6 || v == 7; }
static bool bi_cheap(int v) { return v >= 3; }
static bool bi_onear(int v) { return v >= 5; }
static bool bi_pmm(int v) { return v == 7; }
static std::atomic<long long> g_pmm_steps{0}, g_pmm_merged{0};   /* lever PMM: steps with a merged pass; passes saved (layer 0) */
static constexpr int64_t kBiBand = 48;   /* gemm_nt_ggml / attn_sparse_sink / index_score: the decode band's top */
static constexpr int64_t kBiNorm = 32;   /* the fused norms: r4d_act_threads' 1024-thread width up to this many rows */
/* tp8 loop F58 (19.11.2026): RADIANCE_DS41_BI_DBAND=1 (read with RADIANCE_DS41_BI at the first declare; both nodes) -- a
 * decode segment of more than kBiBand rows (depth-3 drafting at lanes of 16: 16 x 4 = 64 rows under fn4 sched_row_cap
 * 64) runs its attention in pieces of whole decode sequences of at most kBiBand rows (ds41_dec_band.h: as few as greedy
 * packing gives, as even as that count allows) and its indexer in runs of at most kBiBand rows (ds41_idx_runs at that
 * height), so both stay in libr4d's scalar decode kernels (libref's bits) instead of the WMMA forms above 48 rows, which
 * are NOT those bits (r4d_ds41_sparse.hip head note) -- lq150 loadid c32 9 / 20 DIFFERENT at sched_row_cap 64 (F58). A
 * piece past the first reads its sequences' cu relative to its first row: `bi_dcu` row j = cu[s0 .. s1] - cu[s0], one
 * sub_i32 a piece at the step's start (bi_dsub, beside lever BI's bi_sub). Unset or anything but 1: nothing declared,
 * today's issue (a decode segment above the band in one attention launch and one indexer run). */
static std::atomic<int> g_bi_dband{-1};   /* -1 unread, 0 off, 1 on (RADIANCE_DS41_BI_DBAND=1) */
static std::atomic<long long> g_dband_steps{0}, g_dband_pieces{0}, g_dband_runs{0}, g_dband_sync{0}, g_dband_over{0};
/* 2 = 1, and a pass whose decode sequences carry different row counts is recorded too, keyed by its pieces (rows and
 * sequences of each: rad_arch_pass_key); at 1 such a pass is issued live (only the uniform layout records). */
static std::atomic<long long> g_dband_live{0}, g_dband_keyed{0};
static bool bi_dband_env() {
    int v = g_bi_dband.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_BI_DBAND");
        const int w = e && (e[0] == '1' || e[0] == '2') && !e[1] ? e[0] - '0' : 0;
        if (e && !w && std::strcmp(e, "0"))
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_BI_DBAND=%s is not 1 or 2: the decode band pieces are not declared\n", e);
        int unread = -1;
        g_bi_dband.compare_exchange_strong(unread, w);
        v = g_bi_dband.load(std::memory_order_relaxed);
    }
    return v > 0;
}

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
/* written by rad_psk_plan with every rank thread parked, read by the ranks' issue. (Review of tp8 S6, for S5's lanes:) one
 * plan a step in flight, by step number -- with lanes a lane step's leader tail is issued after other lanes' heads have
 * registered their steps' plans, and the one global plan then named another step: the tail found no plan and cut its
 * prefill segments otherwise than the follower, which had the step's plan in its frame. One step in flight (one machine,
 * one lane): the latest step's, as the one plan was. */
static constexpr int kPskPlans = 8;
static PskPlan g_psk_plans[kPskPlans];
static PskPlan& psk_slot(int step) { return g_psk_plans[(unsigned)step % (unsigned)kPskPlans]; }
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
    if (!b || b->draft_pass != 0 || b->step < 0) return nullptr;
    const PskPlan& p = psk_slot(b->step);
    return p.step >= 0 && p.step == b->step ? &p : nullptr;
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
/* Lever TI (ds41-pf2, fn4 sched_tail_inner): each rank's mapped Engram ids of the trunk step it issued last (m.ctok, one i32
 * a row in the batch's row order: token_map's output, media rows written over it) and the Engram window's ids (ngram - 1),
 * for the engine's snapshot inside a step (rad_ckpt_conv_src below). */
static constexpr int kTiRanks = 16;
static std::atomic<const void*> g_ti_ctok[kTiRanks];
static std::atomic<int> g_ti_ctx{0};
/* tp8 S6: a node with layer ranges that holds no Engram layer keeps the Engram tokens' CONV group (every group of the
 * published declare) but no mapped ids: it cannot patch a snapshot taken inside a step, so it answers not capable and
 * the leader's scheduler takes lever TI's switch as 0 there (exact either way: TI places snapshots only). */
static std::atomic<int> g_ti_nocap{0};
static void ti_note(RadCtx* x, const Model& m) {
    const int rk = rad_rank(x);
    if (!m.ctok || rk < 0 || rk >= kTiRanks) return;
    g_ti_ctok[rk].store(rad_buf_ptr(x, m.ctok), std::memory_order_release);
    g_ti_ctx.store((int)m.c.ngram - 1, std::memory_order_release);
}
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

/* Lever M1 (ds41-int9b, 12.11.2026; audit lever 4, kbench2 F1R8: a pair 1874.7 -> 1384.1 us at 4096 rows, ~-1.28 s a
 * 136k prefill): the attention connection's pcn_a (the previous layer's mhc_post with this layer's collapse + norm,
 * issued at the end of the previous layer) and this layer's mix_a (mhc_mix over the streams pcn_a just wrote) as ONE op,
 * mhc_post_collapse_rmsnorm_mix (libr4d: the mix's projection formed from the post's output in LDS in mhc_mix's
 * per-partial order, mhc_mix's finish, the norm in pcn's order -- the two ops' bytes, r4d_ds41_elem.hip pcn_mix_body).
 * Switch "mhc_fuse" (fn4): 0 off; 1 = a pcn_a launch of at least kMhcFuseRows rows is issued as the fused op and the
 * layer's mix_a is not issued for its rows; N in 2 .. 1048576 the same from N rows (the fused op loses below a few
 * hundred rows: kbench2 +84 % at 8 rows, +83 % at 1). RADIANCE_DS41_MHC_FUSE=0 / off | 1 / on | N declares it (the
 * fused op beside every pcn_a: 39 of the 1017's 40 layers, not layer 0 or an Engram layer) and starts at that value;
 * unset declares nothing (>= 1 refused; the program and the issue today's). Why exact: every output of the fused op is
 * the pair's bytes; mhc_mix's arithmetic is a row's own (mix_split_kernel: a partial has the same bits for any R, WV and
 * row count; mix_finish a wave a row), so mixing some of a layer's rows inside the previous layer's pcn_a launches and
 * the rest in its own mix_a launches gives every row today's mix_a bytes; the mix's outputs (pre_a, post, comb) are read
 * by nothing between pcn_a and mix_a (step()'s "Issue order of the connections' mixes": the collapse reads the previous
 * mix's gates, nothing reads a mix's output before the next mhc_post; the tap, the norm, the query / key / indexer read
 * h0 and xn); only the issue order moves (the mix before the attention instead of before attn_output_b). Under lever PSK
 * a fused launch can mix rows the layer's cut drops: they feed nothing later (the cone rule) and nothing reads them. */
static constexpr int64_t kMhcFuseRows = 512;
static std::atomic<int> g_sw_mhc{0}, g_sw_mhc_ok{0}, g_sw_mhc_env{-2}, g_mhc_noted{0};   /* env: -2 unread, -1 unset */
/* live counters (rank 0's, at issue: a recorded pass counts when it is recorded): fused launches, their rows, pcn_a
 * launches the switch left unfused (below its rows) */
static std::atomic<long long> g_mhc_pairs{0}, g_mhc_rows{0}, g_mhc_small{0};
static void mhc_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_MHC_FUSE");
    int v = -1;
    if (e) {
        if (!std::strcmp(e, "on")) v = 1;
        else if (!std::strcmp(e, "off")) v = 0;
        else {
            char* end = nullptr;
            const long n = std::strtol(e, &end, 10);
            if (end != e && !*end && n >= 0 && n <= 1048576) v = (int)n;
        }
    }
    if (g_sw_mhc_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_mhc.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_MHC_FUSE=%s is not 0 / off, 1 / on or a row count up to "
                            "1048576: lever M1 is not declared\n", e);
    }
}
static bool mhc_declared() { return g_sw_mhc_env.load(std::memory_order_relaxed) >= 0; }
/* Levers 1U-GQ and 1U-E2 (ds41-int11, from the i9p 1u profile 12.11.2026), each declared only when its environment asks
 * (read at the first declare; unset: nothing more is declared, the issue is today's and >= 1 is refused) and picked per
 * pass by its fn4 switch:
 *   ds41_gq   (RADIANCE_DS41_GQ=1|2|3, a mask; = K1 fold b): bit 0 the query's q_a and the latent kv projection -- two
 *             GEMVs over the attention norm's output -- as one gemm_nt_ggml_pair launch (Layer::qkv_pair), bit 1
 *             attention_output_a's two group GEMVs as one (Layer::oa_pair). libr4d runs the pair in lever D2's multi-row
 *             form, each weight's rows in their own blocks of one grid with D2's per-output order (every D2 form gives a
 *             row the same bytes), in the decode band; elsewhere (or with D2 off) the two launches.
 *   ds41_e2   (RADIANCE_DS41_E2=1; audit lever 6): the Engram kv projection at a decode-band pass computed by output
 *             columns over the ranks: rank r issues gemm_nt_ggml over rows [r N / W, (r + 1) N / W) of the replicated
 *             weight (an operand view: nothing is loaded twice) into engram_kv_part, then one row-interleaved all_gather
 *             (row N / W) writes engram_kv whole on every rank (Layer::e_gemm_p, Model::e_gath). Exact: each output is
 *             gemm_nt_ggml's per-output sequence (the decode band's forms give a row the same bytes at any N), and the
 *             gather only copies.
 * `_ok`: at least one layer declared the lever, so >= 1 is accepted. Counters (rank 0, at issue: a recorded pass counts
 * when it is recorded) for GET "i11" (rad_i11_stats). */
static std::atomic<int> g_sw_gq{0}, g_sw_gq_ok{0}, g_sw_gq_env{-2};   /* env: -2 unread, -1 unset, else the mask */
static std::atomic<int> g_sw_e2{0}, g_sw_e2_ok{0}, g_sw_e2_env{-2};
static std::atomic<int> g_i11_noted{0};
static std::atomic<long long> g_gq_layers{0}, g_gq_qkv{0}, g_gq_oa{0}, g_e2_layers{0}, g_e2_splits{0}, g_e2_rows{0};
static void i11_read_env() {
    auto rd = [](const char* name, int maxv, std::atomic<int>& env, std::atomic<int>& sw) {
        int unread = -2;
        const char* e = std::getenv(name);
        int v = -1;
        if (e) {
            if (!std::strcmp(e, "on")) v = maxv;
            else if (!std::strcmp(e, "off")) v = 0;
            else {
                char* end = nullptr;
                const long n = std::strtol(e, &end, 10);
                if (end != e && !*end && n >= 0 && n <= maxv) v = (int)n;
            }
        }
        if (env.compare_exchange_strong(unread, v)) {
            if (v >= 0) sw.store(v, std::memory_order_relaxed);
            if (e && v < 0)
                fprintf(stderr, "radiance: deepseek41: %s=%s is not 0 .. %d (or on / off): not declared\n", name, e, maxv);
        }
    };
    rd("RADIANCE_DS41_GQ", 3, g_sw_gq_env, g_sw_gq);
    rd("RADIANCE_DS41_E2", 1, g_sw_e2_env, g_sw_e2);
}
static bool gq_declared() { return g_sw_gq_env.load(std::memory_order_relaxed) >= 0; }
static bool e2_declared() { return g_sw_e2_env.load(std::memory_order_relaxed) >= 0; }
/* the live mask / flag, 0 where nothing was declared */
static int gq_now() { return g_sw_gq_ok.load(std::memory_order_relaxed) ? g_sw_gq.load(std::memory_order_relaxed) : 0; }
static bool e2_now() { return g_sw_e2_ok.load(std::memory_order_relaxed) && g_sw_e2.load(std::memory_order_relaxed) > 0; }
/* Lever FUSE-R (ds41-fuse, 13.11.2026; zzzaql's AQL-stamped c8 on int14: gemm_nt_f32_tile -> router_scored 173 us and
 * router_scored -> scatter_small 172 us of dependent-launch gaps a decode step, both on the critical path): the trunk MoE
 * layers' router chain in fewer launches, each form the stock code with only the launch boundaries removed (libr4d
 * r4d_ds41_elem.hip):
 *   ds41_rfuse 1   the selection and the sort as router_scored_topk_scatter (no gate): router_scored_row a row a wave and
 *                  the stock small sort, in one workgroup (the op ROUTER_FUSE took, there with the gate);
 *   ds41_rfuse 2   the projection too, as router_gemm_f32_scored_topk_scatter: gemm_nt_f32's tile body, the GEMM's last
 *                  workgroup (arrival counter, an op instance's) runs 1's body -- at a pass of <= 48 rows (the decode
 *                  band; a larger pass takes 1's form where it fits the sort, else the separate ops);
 *   0              the separate ops (gemm_nt_f32, router_scored_topk, moe_scatter), today's issue.
 * Every form writes the separate ops' bytes at any row count (each row's logits, selection and its slots' places are its
 * own; the sort is stable and its result does not depend on the workgroup's wave count). RADIANCE_DS41_RFUSE=0 / 1 / 2
 * declares the fused forms beside every trunk MoE layer's router (MoeFP8 Config::rfuse_decl) and starts at that value;
 * unset declares nothing (>= 1 refused; the program and the issue today's). Counters (rank 0, at issue: a recorded pass
 * counts when it is recorded) for GET /admin/fn4 "fuse": passes issued in form 1 / form 2 (rad_rfuse_stats). */
static std::atomic<int> g_sw_rf{0}, g_sw_rf_ok{0}, g_sw_rf_env{-2}, g_rf_noted{0};   /* ok: bit 0 form 1, bit 1 form 2 */
static std::atomic<long long> g_rf_count[2] = { {0}, {0} };
static std::atomic<int> g_rf_layers{0};
static void rf_read_env() {
    int unread = -2;
    const char* e = std::getenv("RADIANCE_DS41_RFUSE");
    int v = -1;
    if (e) {
        if (!std::strcmp(e, "0") || !std::strcmp(e, "off")) v = 0;
        else if (!std::strcmp(e, "1")) v = 1;
        else if (!std::strcmp(e, "2") || !std::strcmp(e, "on")) v = 2;
    }
    if (g_sw_rf_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_rf.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_RFUSE=%s is not 0 / 1 / 2 (or off / on): lever FUSE-R is "
                            "not declared\n", e);
    }
}
static bool rf_declared() { return g_sw_rf_env.load(std::memory_order_relaxed) >= 0; }
/* Lever P (ds41-p1pass, 13.11.2026; SCAL32 on int16: a decode step of 17 .. 32 rows pays a second <= 16-row MoE piece,
 * +10.6 .. 11 ms a step at 24 / 32 rows -- c24 231.8 tok/s against c16's 236.0). Under ds41_bi 2 / 4 / 6 a decode segment
 * above kBiMoe rows is cut into MoE passes of <= 16 rows because the EXL3 experts' M > 16 form (the MTX GEMM) rounds
 * differently from the decode band, and every piece repeats the router, the shared arm, the gather and (bi < 5) the
 * all-reduce, decodes again every expert two pieces share, and loses the fused gather + all-reduce. Switch "ds41_moe1"
 * (fn4): 1 = a step that is ONE decode segment of 17 .. kMoe1Max rows runs its MoE as ONE pass whose routed experts go
 * through moe_exl3_gate_up_dk / moe_exl3_down_dk (exl3/exl3_abi.h; exl3r4d ds41-p1pass: the decode kernels over the whole
 * pass, an expert decoded once, a run 16 rows at a time) -- per (token, slot) the bytes of the <= 16-row pieces (on the
 * card: tools/dkcheck, M 17 .. 48, K 2 / 2.5, 608 cases, 0 rows differ). Everything else in the pass is per row and is
 * what a prefill segment of that height already runs: the router's projection, selection and stable sort (FUSE-R's forms
 * are the stock chain's bytes at any row count), the shared arm's decode-band GEMVs (M-invariant to kBiBand, as the
 * attention's), the gather (a token's slots folded in ascending slot order, which is ascending expert order in either
 * sort) and the exact all-reduce. So a decode row keeps its alone bytes. 0 = the pieces (today).
 * NOT taken, the pieces instead: a step with a prefill segment (a newcomer step: above 1024 tokens the prefill stager is
 * armed and keys a layer on gate_up, which this pass does not issue), a segment above kMoe1Max, a PSK-cut layer, a layer
 * whose library does not serve the ops. RADIANCE_DS41_MOE1=0 / off declares the ops beside every trunk layer's EXL3
 * pair and starts at 0, 1 / on at 1; unset declares nothing (1 refused; the program and the issue are today's). */
static constexpr int64_t kMoe1Max = 48;
/* tp8 loop D21 (18.11.2026): RADIANCE_DS41_MOE1_MAX=N (49 .. 64; read with RADIANCE_DS41_MOE1 at the first declare) raises
 * lever P's ceiling from kMoe1Max to N rows -- the dk pair declared over 17 .. N, a decode step of up to N rows one MoE pass.
 * For depth-3 drafting at lanes of 16 (16 x 4 = 64 rows) with fn4 sched_row_cap N, which otherwise runs four 16-row pieces a
 * layer. Needs an exl3r4d whose _dk rows serve N (built with EXL3R4D_DK_MAX >= N: the constraint is the only change, the
 * kernels take a run 16 rows a pass at any height); with the published library the 49 .. N band has no kernel and the start
 * stops at its declare. The rows' bytes are the pieces' as at <= 48 (exl3r4d rows.c's lever P note; tools/dkcheck on a card at
 * 49 .. N). The dense and attention ops of such a step stay in <= kBiBand-row launches (BI's rows(kBiBand)) and its gather +
 * all-reduce goes unfused above RADIANCE_DS41_ARGATHER_MAX: both the decode band's bytes. Unset, 48 or anything else: 48
 * (the published declare). */
static std::atomic<int64_t> g_p1_max{kMoe1Max};
static int64_t moe1_max() { return g_p1_max.load(std::memory_order_relaxed); }
static std::atomic<int> g_sw_p1{0}, g_sw_p1_ok{0}, g_sw_p1_env{-2}, g_p1_noted{0};
static std::atomic<long long> g_p1_steps{0}, g_p1_rows{0};
static void p1_read_env() {
    int unread = -2;
    if (g_sw_p1_env.load(std::memory_order_relaxed) == -2) {   /* first declare: the ceiling with the switch's own env */
        const char* mx = std::getenv("RADIANCE_DS41_MOE1_MAX");
        if (mx && *mx) {
            char* end = nullptr;
            const long long v = std::strtoll(mx, &end, 10);
            if (end && !*end && v >= 49 && v <= 64) g_p1_max.store(v, std::memory_order_relaxed);
            else if (!(end && !*end && v == kMoe1Max))
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_MOE1_MAX=%s is not 49 .. 64: lever P stays at %lld rows\n", mx,
                        (long long)kMoe1Max);
        }
    }
    const char* e = std::getenv("RADIANCE_DS41_MOE1");
    const int v = !e ? -1 : (!std::strcmp(e, "1") || !std::strcmp(e, "on")) ? 1
                          : (!std::strcmp(e, "0") || !std::strcmp(e, "off")) ? 0 : -1;
    if (g_sw_p1_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_p1.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_MOE1=%s is not 0 / off or 1 / on: lever P is not declared\n", e);
    }
}
static bool p1_declared() { return g_sw_p1_env.load(std::memory_order_relaxed) >= 0; }

/* tp8 loop D24 (19.11.2026, DECODE round 20): the decode pass's next weights read into the on-die cache beside its
 * all-reduces -- libr4d's `prefetch` (docs/OPS.md: a weight's first `bytes`, 16 B a lane over 64 workgroups, writes
 * nothing; device-only) through rad_arch.h's prefetch_beside (the second lane behind a join from lane 0: deepseek41 sets
 * no chain_pays, so launches do not chain). A decode-height all-reduce leaves the card's memory idle for its round trip;
 * the op right after each reads a weight cold: the fused post + collapse + norm + mix (mhc_fuse) its connection's mix fn
 * (24 x 20480 f32, 1.97 MB), then the router / shared arm (after the attention's) or the q / kv projections (after the
 * MoE's). RADIANCE_DS41_PREFETCH=0 .. 7 declares every read at every held layer and starts fn4 ds41_prefetch at that
 * value: bit 0 the mix fns (hc_f_fn beside the attention all-reduce, the next layer's hc_a_fn beside the MoE one), bit 1
 * the router projection + the shared gate_up beside the attention all-reduce, bit 2 the next layer's comp_w (a kv
 * source) / wq_a / wkv beside the MoE all-reduce. RADIANCE_DS41_PREFETCH_MB=1 .. 64 (default 8): a read's cap a weight.
 * Only a trunk pass whose every row decodes, of at most kPfRows rows (prompt steps never take the reads: rad_arch.h).
 * Unset: nothing declared or issued, the switch takes only 0. Exact: the op reads weights and writes nothing. */
static constexpr int64_t kPfRows = ds41_pf::kRows;
static std::atomic<int> g_sw_pf{0}, g_sw_pf_ok{0}, g_sw_pf_env{-2}, g_pf_noted{0};
static std::atomic<int64_t> g_pf_mb{8};
static std::atomic<long long> g_pf_passes{0}, g_pf_reads{0};
static void pf_read_env() {
    int unread = -2;
    if (g_sw_pf_env.load(std::memory_order_relaxed) != -2) return;
    const char* mb = std::getenv("RADIANCE_DS41_PREFETCH_MB");
    if (mb && *mb) {
        char* end = nullptr;
        const long long v = std::strtoll(mb, &end, 10);
        if (end && !*end && v >= 1 && v <= 64) g_pf_mb.store(v, std::memory_order_relaxed);
        else fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PREFETCH_MB=%s is not 1 .. 64: the reads take 8 MB a weight\n", mb);
    }
    const char* e = std::getenv("RADIANCE_DS41_PREFETCH");
    const int v = !e ? -1 : (e[0] >= '0' && e[0] <= '7' && !e[1]) ? e[0] - '0' : -1;
    if (g_sw_pf_env.compare_exchange_strong(unread, v)) {
        if (v >= 0) g_sw_pf.store(v, std::memory_order_relaxed);
        if (e && v < 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PREFETCH=%s is not 0 .. 7: lever D24 is not declared\n", e);
    }
}
static bool pf_declared() { return g_sw_pf_env.load(std::memory_order_relaxed) >= 0; }

/* Lever DKV (14.11.2026, the dense (nvfp4) latent without its slowdown; RADIANCE_DS41_LATENT=nvfp4 only, read once at the
 * first declare; unset: nothing declared or issued differently):
 *   RADIANCE_DS41_LATUNPACK=<E> (1..256): Model::lu ([T][E hd] bf16, E entries a row of the step: 32 KiB a row at 32) holds
 *     a prefill pass's packed compressed entries unpacked once a kv source and segment (latent_unpack, issued before the
 *     first attention over that source's cache: the attentions of layers 2-7 / 8-13 / 14-19 / 20-39 each read one cache),
 *     and those attentions take them (attn_sparse_sink(_rope_roundtrip)_unpacked: the WMMA form reads a 16-key tile the
 *     plan holds as plain bf16 rows -- instead of every row's workgroup unpacking its 512 selections itself, in every
 *     layer -- and a tile with an entry the plan lacks unpacks itself as before);
 *   RADIANCE_DS41_LATGATHER=1: Model::selkv ([min(T, 48)][itopk hd] bf16, 24 MiB) holds the decode pass's selected entries
 *     unpacked once an index source group (latent_gather = lever NVG's gather kernel, before the first attention of the
 *     group: the attentions of [index source x, next index source) read x's selection over one kv source), and those
 *     attentions take them (attn_sparse_sink(_rope_roundtrip)_gathered: lever NVG's plain-kernel pass) -- 8 gathers a step
 *     instead of NVG's 38.
 * Exact: the rows hold the entries' stored bf16 bits (the v3 decode, NVG's), each written after its kv source's store and
 * before every attention that reads it, under a key (kv source, index source, segment rows) checked at every issue -- a
 * miss unpacks again -- and the attentions run the same WMMAs / the same plain kernel over the same values. */
static std::atomic<int> g_dkv_lu{-1}, g_dkv_g{-1};
static std::atomic<long long> g_dkv_lu_n{0}, g_dkv_lu_hit{0}, g_dkv_g_n{0}, g_dkv_g_hit{0};
static void dkv_read_env() {
    int unread = -1;
    const char* e = std::getenv("RADIANCE_DS41_LATUNPACK");
    char* end = nullptr;
    const long v = e && *e ? std::strtol(e, &end, 10) : 0;
    const int lu = end && !*end && v >= 1 && v <= 256 ? (int)v : 0;
    if (g_dkv_lu.compare_exchange_strong(unread, lu) && e && *e && !lu)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LATUNPACK=%s is not 1..256: lever DKV-U is not declared\n", e);
    unread = -1;
    const char* g = std::getenv("RADIANCE_DS41_LATGATHER");
    const int gv = g && (!std::strcmp(g, "1") || !std::strcmp(g, "on")) ? 1 : 0;
    g_dkv_g.compare_exchange_strong(unread, gv);
}
/* The pass's held rows (one rank thread a ctx): which segment's plan m.lu holds for which kv source (latent_unpack issued
 * this pass), which pass's selections m.selkv holds for which (index source, kv source). The issue order is the stream's,
 * so a held entry is valid from its launch to the end of the pass (the caches it reads are written only at their source,
 * before it); every pass starts empty. */
struct DkvHeld {
    int src = -1, idx = -1;
    int64_t s0 = -1, s1 = -1, r0 = -1, n = 0;
};
static thread_local std::vector<DkvHeld> t_dkv_lu;
static thread_local DkvHeld t_dkv_g;
/* tp8 loop D25 step 1 (RADIANCE_DS41_DKV_GPIECE=1, read at the first declare; both nodes): the held gathers a band piece
 * (ds41_dkv_held.h), each at its piece's rows of a selkv of up to kDkvGpRows rows; unset: t_dkv_g as today */
static thread_local std::vector<Ds41Held> t_dkv_gl;
static constexpr int64_t kDkvGpRows = 64;
static std::atomic<int> g_dkv_gp{-1};
static std::atomic<long long> g_dkv_gp_hit{0}, g_dkv_gp_n{0};
static std::atomic<long long> g_dkv_one{0};   /* step 3: decode segments above the band whose attention ran as one launch (rank 0, a layer) */
static bool dkv_gpiece_env() {
    int v = g_dkv_gp.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_DKV_GPIECE");
        /* 2 (tp8 loop D25 step 3): 1's 64-row gathered latent + the decode segment's gather and attention as one launch */
        const int w = e && (e[0] == '1' || e[0] == '2') && !e[1] ? e[0] - '0' : 0;
        if (e && !w && std::strcmp(e, "0"))
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DKV_GPIECE=%s is not 0, 1 or 2: the piece gathers are not declared\n", e);
        int unread = -1;
        g_dkv_gp.compare_exchange_strong(unread, w);
        v = g_dkv_gp.load(std::memory_order_relaxed);
    }
    return v > 0;
}
/* this rank thread's counts: unpacks, plan hits, gathers, gather hits; rank 0 notes them at its 1st, 10th, 100th ... launch */
static thread_local long long t_dkv_ct[4] = { 0, 0, 0, 0 };
static void dkv_note(RadCtx* x, const char* what, int l, int64_t n, int64_t hr0, int64_t hn) {
    if (rad_rank(x) != 0) return;
    const long long k = t_dkv_ct[0] + t_dkv_ct[2];
    long long p = 1;
    while (p < k) p *= 10;
    if (p != k) return;
    fprintf(stderr, "radiance: deepseek41: lever DKV (rank 0, launch %lld: %s at layer %d, %lld rows, plan rows [%lld, %lld)): "
                    "latent_unpack %lld (plan hits %lld), latent_gather %lld (hits %lld)\n",
            k, what, l, (long long)n, (long long)hr0, (long long)(hr0 + hn), t_dkv_ct[0], t_dkv_ct[1], t_dkv_ct[2],
            t_dkv_ct[3]);
}
/* RADIANCE_DS41_DKV_LOG=1 (read once): rank 0 notes every pass of at least 2048 rows when the next one starts -- the
 * attentions it ran over the plan's / the gathered rows, the taller ones it left to today's op, the unpacks and gathers */
static thread_local long long t_dkv_pass[6] = { 0, 0, 0, 0, 0, 0 };   /* plan, gathered, tall today, unpacks, gathers, rows */
static thread_local long long t_dkv_full = 0;                          /* plan attentions in the LDS-free form (plan_full) */
static bool dkv_log_on() {
    static const bool on = [] { const char* e = std::getenv("RADIANCE_DS41_DKV_LOG"); return e && *e && *e != '0'; }();
    return on;
}
static void dkv_pass_reset(RadCtx* x = nullptr, int64_t rows = 0) {
    if (x && rad_rank(x) == 0 && dkv_log_on() && t_dkv_pass[5] >= 2048 &&
        t_dkv_pass[0] + t_dkv_pass[1] + t_dkv_pass[2] + t_dkv_pass[3] + t_dkv_pass[4] > 0)
        fprintf(stderr, "radiance: deepseek41: lever DKV pass of %lld rows: %lld attentions over the plan's rows (%lld of them "
                        "plan_full), %lld over gathered rows, %lld taller ones by today's op; %lld latent_unpack, %lld "
                        "latent_gather\n",
                t_dkv_pass[5], t_dkv_pass[0], t_dkv_full, t_dkv_pass[1], t_dkv_pass[2], t_dkv_pass[3], t_dkv_pass[4]);
    for (long long& v : t_dkv_pass) v = 0;
    t_dkv_full = 0;
    t_dkv_pass[5] = rows;
    t_dkv_lu.clear();
    t_dkv_g = DkvHeld{};
    t_dkv_gl.clear();
}
/* The fewest rows a pcn_a launch is fused at under the switch; 0 = none. */
static int64_t mhc_fuse_min() {
    if (!g_sw_mhc_ok.load(std::memory_order_relaxed)) return 0;
    const int v = g_sw_mhc.load(std::memory_order_relaxed);
    return v <= 0 ? 0 : v == 1 ? kMhcFuseRows : (int64_t)v;
}
/* fn over the rows of [r0, r0 + n) that the fused launches `done` (disjoint) did not mix, a run at a time */
template <class F>
static void mhc_rest(std::vector<std::pair<int64_t, int64_t>>& done, int64_t r0, int64_t n, const F& fn) {
    if (done.empty()) { fn(r0, n); return; }
    std::sort(done.begin(), done.end());
    int64_t a = r0;
    const int64_t e = r0 + n;
    for (const auto& d : done) {
        const int64_t d0 = d.first, d1 = d.first + d.second;
        if (d1 <= a || d0 >= e) continue;
        if (d0 > a) fn(a, d0 - a);
        a = std::max(a, d1);
        if (a >= e) break;
    }
    if (a < e) fn(a, e - a);
}

/* The rows a step's fused launches mixed (each rank thread its own; set where the fused op is issued, read where the
 * mix is): t_mhc_a the attention mix of layer t_mhc_a_l (fused into the previous layer's pcn_a), t_mhc_f this layer's
 * ffn mix (fused into its pcn_f; cleared at each layer's ffn connection). */
static thread_local std::vector<std::pair<int64_t, int64_t>> t_mhc_a, t_mhc_f;
static thread_local int t_mhc_a_l = -1;
static void mhc_note(RadCtx* x, int64_t n) {
    if (rad_rank(x) != 0) return;
    g_mhc_pairs.fetch_add(1, std::memory_order_relaxed);
    g_mhc_rows.fetch_add(n, std::memory_order_relaxed);
}

/* Lever IX-NI's live count (ds41-int9b): libr4d serves fn4 ix_ni (index_score writes no -inf past a row's reach;
 * index_candidates then reads below it only); deepseek41 takes the switch too, only to count what the issue leaves
 * unwritten (rank 0, at issue: a recorded pass counts when it is recorded): index_score runs issued under it, and the
 * dead bytes, 4 x (live - min(live, reach)) for each row of a run's plane (a row past the step: the whole live width).
 * RADIANCE_DS41_IX_NI=1: the start value (libr4d reads the same). */
static std::atomic<int> g_sw_ixni{-1};
static std::atomic<long long> g_ixni_runs{0}, g_ixni_dead{0};
static int ixni_on() {
    int v = g_sw_ixni.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_IX_NI");
        int unread = -1;
        g_sw_ixni.compare_exchange_strong(unread, e && e[0] == '1' && !e[1] ? 1 : 0);
        v = g_sw_ixni.load(std::memory_order_relaxed);
    }
    return v;
}
/* one run's count: R plane rows, `rows` of them the step's from host positions `hp` (null: runs only), width `live` */
static void ixni_note(RadCtx* x, const int32_t* hp, int64_t rows, int64_t R, int64_t live, int64_t ratio) {
    if (ixni_on() <= 0 || rad_rank(x) != 0) return;
    g_ixni_runs.fetch_add(1, std::memory_order_relaxed);
    if (!hp || ratio < 1) return;
    long long dead = 0;
    for (int64_t m = 0; m < R; ++m) {
        const int64_t reach = m < rows ? ((int64_t)hp[m] + 1) / ratio : 0;
        dead += (long long)(live - std::min<int64_t>(live, std::max<int64_t>(reach, 0)));
    }
    g_ixni_dead.fetch_add(4 * dead, std::memory_order_relaxed);
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
    /* tp8 S6: a two-node pipeline node runs BI segments and PSK passes (step_bi crosses at the cut, per node; S4 refused
     * them here) */
    if (const int r = idx_fn4_accepts(name, value)) return r;
    if (!std::strcmp(name, "ds41_argather"))
        return value >= 0 && value <= g_sw_gar_decl.load(std::memory_order_relaxed) ? 1 : -1;
    if (!std::strcmp(name, "ds41_shexp_lane"))   /* 2: lever F4P (ds41-pf2), prefill-height passes only */
        return value == 0 || ((value == 1 || value == 2) && g_sw_shexp_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (const K1Switch* k = k1_switch(name))
        return value == 0 || (value >= 1 && value <= k->max && k->ok->load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "dspark_walk"))
        return value == 0 || (value == 1 && g_sw_walk_ok.load(std::memory_order_relaxed)) ||
               (value == 2 && g_sw_walk2_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "dspark_mhr")) return value >= 0 && value <= 2 ? 1 : -1;   /* lever DMR (2: + the key's fold c) */
    if (!std::strcmp(name, "ds41_bi"))   /* 5 / 6: lever A9 (one MoE all-reduce a segmented step); 7: lever PMM */
        return value == 0 || (value >= 1 && value <= 7 && g_sw_bi_ok.load(std::memory_order_relaxed)) ? 1 : -1;
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
    if (!std::strcmp(name, "mhc_fuse"))   /* lever M1 */
        return value == 0 || (value >= 1 && value <= 1048576 && g_sw_mhc_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ix_ni")) return value == 0 || value == 1 ? 1 : -1;   /* lever IX-NI: counted here, libr4d's */
    if (!std::strcmp(name, "ds41_gq"))   /* lever 1U-GQ */
        return value == 0 || (value >= 1 && value <= 3 && g_sw_gq_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_e2"))   /* lever 1U-E2 */
        return value == 0 || (value == 1 && g_sw_e2_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_prefetch"))   /* tp8 loop D24: >= 1 needs the reads declared */
        return value == 0 || (value >= 1 && value <= 7 && g_sw_pf_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_moe1"))   /* lever P: 1 needs the dk ops declared at some trunk layer */
        return value == 0 || (value == 1 && g_sw_p1_ok.load(std::memory_order_relaxed)) ? 1 : -1;
    if (!std::strcmp(name, "ds41_rfuse"))   /* lever FUSE-R: 1 needs form 1 declared somewhere, 2 form 2 */
        return value == 0 || (value == 1 && (g_sw_rf_ok.load(std::memory_order_relaxed) & 1)) ||
               (value == 2 && (g_sw_rf_ok.load(std::memory_order_relaxed) & 2)) ? 1 : -1;
    /* tp8 loop P25s: the SP route's switches take away only what the env declared (0 = as declared for ds41_sphc) */
    if (!std::strcmp(name, "ds41_sphc"))
        return value == 0 || (g_sw_sphc_decl.load(std::memory_order_relaxed) > 0 &&
                              value >= g_sw_sphc_decl.load(std::memory_order_relaxed) && value <= (1 << 24)) ? 1 : -1;
    /* the levels: 0 .. the env's (both nodes carry the same env; a node whose declare holds less -- the leader has no
     * segment passes -- takes the value and issues what it holds) */
    if (!std::strcmp(name, "ds41_sphc_attn"))
        return value >= 0 && value <= g_sw_spattn_decl.load(std::memory_order_relaxed) ? 1 : -1;
    if (!std::strcmp(name, "ds41_sphc_seg"))
        return value >= 0 && value <= g_sw_spseg_decl.load(std::memory_order_relaxed) ? 1 : -1;
    /* tp8 loop P30t: mask + 256 x rows, the mask a subset of RADIANCE_DS41_TWIN's classes (0: every row GGUF) */
    if (!std::strcmp(name, "ds41_twin"))
        return value >= 0 && ((value & 255) & ~twin_env()) == 0 && (value >> 8) <= 65535 ? 1 : -1;
    return 0;
}
extern "C" int rad_fn4_switch(const char* name, int value) {
    if (name && !std::strcmp(name, "idx_split")) return idx_fn4_switch(name, value);
    const int ok = rad_fn4_accepts(name, value);
    if (ok <= 0) return ok;
    if (!std::strcmp(name, "ds41_argather")) g_sw_gar.store(value, std::memory_order_relaxed);
    else if (const K1Switch* k = k1_switch(name)) k->val->store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "dspark_walk")) g_sw_walk.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "dspark_mhr")) g_sw_dmr.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_bi")) g_sw_bi.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk")) g_sw_psk.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk_l20")) g_sw_psk_l20.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "dspark_pfctx")) g_sw_pfctx.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_psk_cr")) g_sw_psk_cr.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_engdd")) g_sw_engdd.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "mhc_fuse")) g_sw_mhc.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ix_ni")) g_sw_ixni.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_gq")) g_sw_gq.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_e2")) g_sw_e2.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_rfuse")) g_sw_rf.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_moe1")) g_sw_p1.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_prefetch")) g_sw_pf.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_sphc")) g_sw_sphc.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_sphc_attn")) g_sw_spattn.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_sphc_seg")) g_sw_spseg.store(value, std::memory_order_relaxed);
    else if (!std::strcmp(name, "ds41_twin")) g_sw_twin.store(value, std::memory_order_relaxed);   /* tp8 loop P30t */
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
    if (!std::strcmp(name, "mhc_fuse")) {
        *applied = g_sw_mhc.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "ix_ni")) {
        *applied = ixni_on();
        return 1;
    }
    if (!std::strcmp(name, "dspark_walk")) {
        *applied = g_sw_walk.load(std::memory_order_relaxed);
        return 1;
    }
    if (!std::strcmp(name, "dspark_mhr")) {
        *applied = dmr_now();
        return 1;
    }
    if (!std::strcmp(name, "ds41_gq")) {
        *applied = gq_now();
        return 1;
    }
    if (!std::strcmp(name, "ds41_e2")) {
        *applied = e2_now() ? 1 : 0;
        return 1;
    }
    if (!std::strcmp(name, "ds41_moe1")) {   /* what the issue reads: 0 where nothing was declared */
        *applied = g_sw_p1_ok.load(std::memory_order_relaxed) ? g_sw_p1.load(std::memory_order_relaxed) : 0;
        return 1;
    }
    if (!std::strcmp(name, "ds41_rfuse")) {   /* what the issue reads: 0 where nothing was declared */
        *applied = g_sw_rf_ok.load(std::memory_order_relaxed) ? g_sw_rf.load(std::memory_order_relaxed) : 0;
        return 1;
    }
    if (!std::strcmp(name, "ds41_prefetch")) {   /* tp8 loop D24: what the issue reads, 0 where nothing was declared */
        *applied = g_sw_pf_ok.load(std::memory_order_relaxed) ? g_sw_pf.load(std::memory_order_relaxed) : 0;
        return 1;
    }
    /* tp8 loop P25s: what the issue reads (the threshold in rows, 0 where nothing is SP; the levels) */
    if (!std::strcmp(name, "ds41_sphc")) {
        const int d = g_sw_sphc_decl.load(std::memory_order_relaxed), v = g_sw_sphc.load(std::memory_order_relaxed);
        *applied = d > 0 && v > d ? v : d;
        return 1;
    }
    if (!std::strcmp(name, "ds41_sphc_attn")) {
        const int d = g_sw_spattn_decl.load(std::memory_order_relaxed), v = g_sw_spattn.load(std::memory_order_relaxed);
        *applied = v >= 0 && v < d ? v : d;
        return 1;
    }
    if (!std::strcmp(name, "ds41_sphc_seg")) {
        const int d = g_sw_spseg_decl.load(std::memory_order_relaxed), v = g_sw_spseg.load(std::memory_order_relaxed);
        *applied = v >= 0 && v < d ? v : d;
        return 1;
    }
    if (!std::strcmp(name, "ds41_twin")) {   /* tp8 loop P30t: what the issue reads (classes held here) + 256 x the rows set */
        const int v = g_sw_twin.load(std::memory_order_relaxed);
        *applied = twin_mask_now() + 256 * (v < 0 ? 0 : (v >> 8));
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
    /* Lever 1U-ETP: refused where the width does not cut into a slice a rank (etp_slice) */
    if (etp_env() < 0) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_ETP=%s is not 0 / off or 1 / on\n", std::getenv("RADIANCE_DS41_ETP"));
        return RAD_E_INVAL;
    }
    m.etp = etp_env() > 0 && world > 1;
    if (xring_env() < 0) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_XRING=%s is not 0 .. 1024 (MiB)\n", std::getenv("RADIANCE_DS41_XRING"));
        return RAD_E_INVAL;
    }
    if (xring_env() > 0 && m.etp && ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: lever XR (RADIANCE_DS41_XRING): %lld MiB of copy ring in each trunk gate_up's prefill "
                        "scratch (exl3r4d fn4 exl3_xr; the same bytes either way)\n", (long long)xring_env());
    if (m.etp) {
        int64_t lo[2], hi[2];
        if (!etp_slice(c.n_ff, world, ctx->rank, lo, hi)) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_ETP: an expert of width %lld does not cut into whole 128-column "
                            "blocks for %d ranks\n", (long long)c.n_ff, world);
            return RAD_E_UNSUPPORTED;
        }
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: lever 1U-ETP (RADIANCE_DS41_ETP): every routed expert on every rank as a width "
                            "slice -- rank 0 even experts [%lld, %lld), odd [%lld, %lld) of %lld; the gather folds every slot on every "
                            "rank and the exact all-reduce sums the four partials (a reference change against expert parallel)\n",
                    (long long)lo[0], (long long)hi[0], (long long)lo[1], (long long)hi[1], (long long)c.n_ff);
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
/* Lever SPF-IR (RADIANCE_DS41_IROWS_MIN=<rows>, read once at the first declare): the runs' floor instead of
 * kScoreRunMin. This model's index layers run at ratio 1, so cmax = max_ctx + 1 and the plane is 4 bytes x max_ctx a
 * row: at --max-model-len 1048576 the 256-row floor makes it 1 GiB a card (512 MiB at 524288), and the arena's every
 * level from 256 rows up carries it. The plane only has to hold one run, so a lower floor is placement only, provided
 * each run keeps the kernel form the unsplit step takes: a split step (T > irows rows, n = ceil(T / irows) >= 2 runs)
 * has runs of R = ceil(T / n) > T irows / (T + irows) > irows / 2 rows, so a floor >= 2 kDs41IdxFormRows (96) keeps
 * every run above the decode form's 48 rows, in the WMMA form the unsplit step (T > irows >= 96 rows) takes; index_score
 * computes each row alone in that form (ds41_index_split.h), index_candidates / index_topk select a row exactly
 * whatever rows share a launch, and the selections land at the same rows (the split's own argument). Accepted: 96 ..
 * 256 (a number outside, or junk: warned, kScoreRunMin). Unset: kScoreRunMin, every declare as before. */
static constexpr int64_t kScoreRunFloorMin = 2 * kDs41IdxFormRows;
static int64_t ds41_irows_min() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_IROWS_MIN");
        if (!e || !*e) return kScoreRunMin;
        char* end = nullptr;
        const long long r = std::strtoll(e, &end, 10);
        if (!end || *end || r < kScoreRunFloorMin || r > kScoreRunMin) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_IROWS_MIN=%s is not a row count in %lld .. %lld; "
                            "the indexer's runs keep their %lld-row floor\n",
                    e, (long long)kScoreRunFloorMin, (long long)kScoreRunMin, (long long)kScoreRunMin);
            return kScoreRunMin;
        }
        return (int64_t)r;
    }();
    return v;
}
static void index_runs(Model& m) {
    const auto& g = m.g;
    const auto& c = m.c;
    m.irows = std::min(g.max_tok, std::max(ds41_irows_min(), kScorePlane / (4 * m.cmax)));
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

/* tp8 loop P13 -- RADIANCE_DS41_PIPE2=N (read at declare; unset, empty, 0 or anything not a number = off, today's declare
 * and issue byte for byte): a lone prefill pass of >= N rows (N >= 2 x kPipe2Piece; "1" = 2048) runs the MoE of each
 * uncut layer in two row pieces cut at a multiple of kPipe2Align near the middle, both pieces >= kPipe2Piece rows:
 * the two pieces' router projections and their all_gathers first (lane 0, as the pass issues them), then MoE pass 0,
 * its all-reduce on the second lane beside MoE pass 1, then pass 1's all-reduce, lane 0 waiting for it before the
 * layer goes on. Why: a node's prefill pass is its TP4 all-reduces ~25 % of the time with the card idle on its stream
 * (LEVERS PA / L7); on a pipeline node every expert is in VRAM, so the link carries only the collectives (L7's loss on
 * one node was the expert staging beside the lane-1 AR and the router gathers queued behind it).
 *
 * EXACT by construction (the published rows' bytes): a piece issues the whole pass's own MoE ops over a row range of the
 * same operands (router projection + all_gather, selection + sort, the EXL3 experts in the map form, the shared arm, the
 * gather), each computing a row from that row alone by the form the whole pass takes at prefill height (kPipe2Piece is
 * above every row-count choice: gemm_nt_f32's split from 256, the map form from 2 T > 3 n_expert, the act widths from
 * 121); the exact wire's all-reduce folds each element in ascending rank order at any message size, and its overlap
 * instance (`overlap` 1, MoeFP8::op_ar_ov) is the same fold in a narrower launch. Only uncut layers (not PSK's cone
 * layers) and only passes of one prefill segment. Collectives stay totally ordered on every rank: piece 0's all-reduce
 * goes to lane 1 behind everything lane 0 issued, lane 0 then issues piece 1's pass, which holds no collective (its
 * router was projected before), piece 1's all-reduce follows on lane 1, and lane 0 joins lane 1 before its next op. */
static constexpr int64_t kPipe2Align = 256, kPipe2Piece = 512;
static int64_t pipe2_env() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_PIPE2");
        if (!e || !*e) return (int64_t)0;
        char* end = nullptr;
        const long long n = std::strtoll(e, &end, 10);
        if (!end || *end || n <= 0) return (int64_t)0;
        return n == 1 ? (int64_t)2048 : std::max<int64_t>((int64_t)n, 2 * kPipe2Piece);
    }();
    return v;
}
static std::atomic<long long> g_pipe2_passes{0}, g_pipe2_layers{0}, g_pipe2_rows{0}, g_pipe2_serial{0};
static std::atomic<long long> g_pipe2_attn{0};   /* P13b: layer-passes whose attention all-reduce ran in the pieces */
/* tp8 loop P20 round 2 -- RADIANCE_DS41_SPHC=N (read at declare; unset, empty, 0 or not a number = off, today's declare and
 * issue byte for byte; "1" = 2048): SEQUENCE-PARALLEL mHC STREAMS in a lone prefill pass of >= N rows (one prefill segment,
 * no decode row, no media row, not a follower / leader segment). Today every rank runs every mHC op (mhc_post / collapse /
 * rmsnorm / mhc_mix, their fused forms, Engram's stream_gate_add) over ALL rows of the pass on the replicated 4 x 5120
 * streams after a full all-reduce. Here an SP layer's sublayer output is REDUCE-SCATTERED by whole rows (rank r keeps rows
 * [r q, (r + 1) q), q = ceil(T / 4): the all-reduce's first half, the same fp32 ascending-rank fold = the same bytes), the
 * mHC ops run over the rank's quarter only, and the next sublayer's normed input is ALL-GATHERED raw (the all-reduce's
 * second half: the same words). The streams stay sharded between sublayers and layers; a later non-SP layer (PSK's cone /
 * split layers with cone rows, the leader's last layer before X1, a DSpark tap layer, the node's last layer) first
 * gathers the streams (all_gather_rows of h0, rows of 4 x 5120) and re-mixes its attention coefficients over every row.
 *
 * EXACT by construction: the mHC ops are per row (a wave / a workgroup a row; lever BI's r4d_selftest --bicheck holds them
 * at any row range), so a row's bytes at a quarter launch are its bytes at the whole one; the reduce-scatter + raw gather
 * is the exact all-reduce's per-element arithmetic (r4d_selftest --case sp_rows_exact); the Engram lookup / copy / kv
 * GEMM are per row (gemm_nt_ggml's prefill form at >= 512 rows). A layer is SP only where every op that reads the streams
 * between it and the next gather is an SP op (spv below), so no rank reads a row it did not write. */
static int64_t sphc_env() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_SPHC");
        if (!e || !*e) return (int64_t)0;
        char* end = nullptr;
        const long long n = std::strtoll(e, &end, 10);
        if (!end || *end || n <= 0) return (int64_t)0;
        return n == 1 ? (int64_t)2048 : std::max<int64_t>((int64_t)n, 512);
    }();
    return v;
}
static bool sphc_eng_env() {
    static const bool on = [] { const char* e = std::getenv("RADIANCE_DS41_SPHC_ENGRAM"); return !(e && e[0] == '0' && !e[1]); }();
    return on;
}
static std::atomic<long long> g_sp_passes{0}, g_sp_layers{0}, g_sp_rows{0}, g_sp_gathers{0}, g_sp_eng{0}, g_sp_lone{0};
/* tp8 loop P20 round 3 -- RADIANCE_DS41_SPHC_X1=1 (with RADIANCE_DS41_SPHC, read at declare; default 0 = round 2's X1): an SP
 * pass's X1 leaves the leader row-sharded -- each rank writes its quarter rows of the residual parts (act.x, h1, post, comb,
 * pre_ffn) into the send ring, rank 0 the selections / keep words / kv entries whole (stage_send `shard`, split 3); the
 * payload's bytes are the replicated send's (every row's value is the one every rank would hold after the gather), so the
 * follower's receive is unchanged. The leader's layer h - 1 is then an SP layer and the streams' gather before it goes. */
static bool sphc_x1_env() {
    static const bool on = [] { const char* e = std::getenv("RADIANCE_DS41_SPHC_X1"); return e && e[0] == '1' && !e[1]; }();
    return on;
}
/* tp8 loop P25 -- RADIANCE_DS41_SPHC_SEG=1 / 2 (with RADIANCE_DS41_SPHC, the follower, read at declare; Model::sp_seg): SP in a
 * follower segment pass. EXACT by construction: the same SP ops over the same rows as the whole pass's (a segment boundary is
 * transparent to the plan, and the next segment issues the boundary op the whole pass issues at the layer's end, over the
 * same quarter); the boundary's save / restore moves bytes only (2: only the quarter SP layers read). 3 = P25b, 4 = P25c (Model::
 * sp_seg). Anything else 0. */
static int sphc_seg_env() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_SPHC_SEG");
        return e && (e[0] >= '1' && e[0] <= '4') && !e[1] ? e[0] - '0' : 0;
    }();
    return v;
}
static std::atomic<long long> g_sp_seg_passes{0}, g_sp_seg_layers{0}, g_sp_seg_sh{0};
/* tp8 loop P25c: lone follower prefill passes with a layer past PSK's first planned SP, and those layer-passes */
static std::atomic<long long> g_sp_cone_passes{0}, g_sp_cone_layers{0};
static std::atomic<long long> g_sp_x1{0};
/* tp8 loop P26 -- RADIANCE_DS41_SPHC_ATTN=1 (with RADIANCE_DS41_SPHC, read at declare; default 0 = round 4's SP layers): the
 * attention's input GEMMs q_a and kv are declared unsharded (every rank holds the whole weight and, today, runs it over every
 * row of an SP layer after the gather of xn: four times the work). Here, at an SP layer where nothing else reads xn (no
 * compressor chain, no indexer projection: DeepSeek-V4.1-Flash's layers 0-19 but 2 / 8 / 14), each rank runs q_a and kv over
 * its quarter of the rows and the outputs qr (q_lora a row) and kv (head_dim a row) are all-gathered raw instead of xn
 * (n_embd a row). The ops after them (q_b's norm + GEMM, the window key's norm / rope / round trip / store, the ropes) read
 * every row as today. EXACT by construction: a row's gemm_nt_ggml bytes are the same at any M and any place in the launch
 * above the decode band (lever WMG's byte check: 28 keys, 25,928 rows, 0 differ; lever GPF's forms keep the per-output
 * order) and a quarter of an SP pass is >= 128 rows (SPHC >= 512); the gathers move the owner's words. */
static int sphc_attn_env() {
    /* 1 = q_a / kv (P26); 2 = + the compressor at the kv-source layers (P26b); anything else 0 */
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_SPHC_ATTN");
        return e && (e[0] == '1' || e[0] == '2') && !e[1] ? e[0] - '0' : 0;
    }();
    return v;
}
static std::atomic<long long> g_sp_attn{0}, g_sp_comp{0};
/* tp8 loop P25s (PREFILL round 17) -- the SP route's in-process switches, so ONE start serves both arms of an A / B (TESTQUEUE
 * 2b: the control beside RADIANCE_DS41_SPHC=1024 SPHC_ATTN=1 SPHC_SEG=3) on the same image, cards and load. Each only takes
 * away what the start's env declared, and its default is that env's (a start that sets none of them is today's):
 *   ds41_sphc       0 = the declared threshold (RADIANCE_DS41_SPHC); N >= it = a pass is SP from N rows (N above
 *                   --max-num-batched-tokens: no pass is) -- 2048 at a 1024 declare = BEST's SP passes (the cold chunks)
 *   ds41_sphc_attn  0 .. the declared level (P26 1, P26b 2): 0 = q_a / kv after xn's gather, as a start without it
 *   ds41_sphc_seg   0 .. the declared level (P25 1 / 2, P25b 3): 0 = the follower's segment passes whole, as without it
 * EXACT whatever the values: SP is the same fold and bytes as the replicated pass (P20), X1 / X2 carry every row whichever side
 * shards (a sharded send's payload is the whole send's; a whole receive copies every row), and a segment boundary's form is one
 * pass's own (a switch change keeps its place in the follower's frame order, never between a pass's segments). Applied between
 * steps like every switch; both nodes take it from the leader's switch frame. */
/* what a pass issues with: the declare's, less what the switch takes away */
static int64_t sphc_now(const Model& m) {
    const int v = g_sw_sphc.load(std::memory_order_relaxed);
    return m.sphc > 0 && v > m.sphc ? (int64_t)v : m.sphc;
}
static int spattn_now(const Model& m) {
    const int d = m.sp_ag_qr ? (m.sp_ag_kvs ? 2 : 1) : 0, v = g_sw_spattn.load(std::memory_order_relaxed);
    return v >= 0 && v < d ? v : d;
}
static int spseg_now(const Model& m) {
    const int v = g_sw_spseg.load(std::memory_order_relaxed);
    return v >= 0 && v < m.sp_seg ? v : m.sp_seg;
}
/* tp8 loop P25c: the lone follower passes' uncut deep layers -- declared (RADIANCE_DS41_SPHC_SEG=4) and the switch at 4 */
static bool splone_now(const Model& m) {
    const int v = g_sw_spseg.load(std::memory_order_relaxed);
    return m.sp_lone && (v < 0 || v >= 4);
}
/* tp8 loop P13b -- RADIANCE_DS41_PIPE2_ATTN=1 (read at declare, only with RADIANCE_DS41_PIPE2 on): a pipelined layer's
 * attention output too: the first group projections and the attention mix as today, o_b of piece 0, piece 0's
 * attention all-reduce on the second lane behind lane 0 and beside piece 1's o_a + o_b (no collective), lane 0
 * behind lane 1, piece 1's all-reduce; then post / collapse / norm over the step as today. The same ops over row
 * ranges of the same operands (gemm_nt_ggml's 128-row WMMA tiles over the whole K above 48 rows: per row), the exact
 * wire's fold per element. Anything else: off. */
static bool pipe2_cone_env() {
    static const bool on = [] { const char* e = std::getenv("RADIANCE_DS41_PIPE2_CONE"); return e && e[0] == '1' && !e[1]; }();
    return on;
}
static bool pipe2_attn_env() {
    static const bool on = [] { const char* e = std::getenv("RADIANCE_DS41_PIPE2_ATTN"); return e && e[0] == '1' && !e[1]; }();
    return on;
}
/* tp8 loop P15 -- RADIANCE_DS41_PIPE2_TBO=1 / 2 / 3 (read at declare, only with RADIANCE_DS41_PIPE2 + _ATTN on and every
 * rank attending; anything else = 0, the P13 issue byte for byte): the two piece all-reduces P13a / P13b leave serial on
 * lane 0 go to the second lane as well, each beside work of the OTHER piece (the two-micro-batch overlap inside a node):
 *   bit 1 (attention side): after piece 1's o_a + o_b, lane 0 joins piece 0's attention all-reduce, issues piece 0's
 *     post / collapse / norm (+ mix) and its router projection + all_gather; then piece 1's attention all-reduce on lane 1
 *     BESIDE MoE pass 0; lane 0 joins it, issues piece 1's post / collapse / norm and router; piece 0's MoE all-reduce on
 *     lane 1 beside MoE pass 1 (as P13a), lane 0 joins it;
 *   bit 2 (layer boundary): piece 1's MoE all-reduce on lane 1 BESIDE the next layer's post / collapse / norm (+ mix) of
 *     piece 0 and that layer's per-row attention inputs over piece 0 (q_a, kv, the indexer's projections, q_b, the window
 *     key's norm / rope / store, the ropes); lane 0 then joins it and issues the same over piece 1, and the compressor
 *     chain, the indexer and the attention as today over the step -- only between two adjacent uncut layers this node
 *     issues in the same pass, with no crossing between them and no Engram block / tap / receive in the second.
 * EXACT by construction: every op is the one P13a / P13b (or the whole pass) issues, over the same row ranges of the same
 * operands (per-row ops: the fused mhc forms work in 8-row groups and a piece starts at a multiple of 256); the collectives
 * are the same messages, each behind a join on its lane, so they stay totally ordered on every rank (never two in
 * flight); the buffers a reordered op reads or writes after the other piece's ops are kept out of the sharing pool
 * (rad_buf_concurrent: h0 + pre_attn for bit 1, streams_next + pre_ffn + collapsed for bit 2). */
/* bit 4 (P16, after the cards' lq50: P13a's two MoE passes cost a 4,096-row head +15 ms, more than its lane-1 all-reduce
 * hid): the MoE of such a layer runs WHOLE (one pass, its rows' experts read once), only its all-reduce in the two halves
 * -- half 0 on lane 0, half 1 on lane 0 too or, with bit 2, on lane 1 beside the next layer's piece-0 inputs; bit 1 is
 * not taken with bit 4 (it needs MoE pass 0). TBO=4: P13b's attention pieces with the MoE whole; TBO=6: + bit 2. */
static int pipe2_tbo_env() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_PIPE2_TBO");
        if (!e || !e[0] || e[1] || e[0] < '1' || e[0] > '7') return 0;
        return e[0] - '0';
    }();
    return v;
}
/* P15's counters (rank 0): layer-passes with bit 1's order, layer boundaries with bit 2's, fallbacks (a deferral the next
 * layer could not take, finished at once) */
static std::atomic<long long> g_tbo_attn{0}, g_tbo_xl{0}, g_tbo_flush{0}, g_tbo_whole{0};   /* + bit 4: layer-passes whose MoE ran whole */
/* P15 bit 2: the layer whose piece 0 the previous layer prepared (its pcn_mix over [p0, p0 + pa)), piece 1's MoE all-reduce
 * still on lane 1; -1 = none. Per rank thread, reset at every pass's start. */
struct TboDefer { int l = -1; int64_t p0 = 0, pa = 0, pn = 0; };
static thread_local TboDefer t_tbo;

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

/* Lever 1U-GQ (ds41-int11) at a layer's declare (RADIANCE_DS41_GQ declared): gemm_nt_ggml_pair beside q_a + kv_g (both
 * GEMVs of the attention norm's output `xn`, into `qr` and `kv`) where both weights are GGUF blocks of one type
 * (gemm_nt_ggml, not an IQR view), and beside the two attention_output_a groups (`o`'s two column halves into `oa`'s)
 * where the rank holds two of one type. fused(): 0 where no library serves the op (a host deployment), and the
 * separate ops stay declared for every pass the switch leaves to them. Where it is declared: the pair is issued where
 * the first of its two ops is (q_a's point, the first group's), so it is declared at or before the point its writes
 * begin in issue order -- the trunk's after kv_g (the trunk issues q_a and kv_g back to back, and its qr / kv live over
 * every layer), the drafter's right after its q_a (the drafter issues kv_g after q_an / q_b / q_rope: declared there,
 * kv_'s planned life starts before them and no query-chain buffer can share its bytes); the groups' after both. */
static const char* gq_type(const Model& m, rad_weight w1, rad_weight w2) {
    const auto t1 = m.wty.find(w1), t2 = m.wty.find(w2);
    if (t1 == m.wty.end() || t2 == m.wty.end() || m.wsc.count(w1) || m.wsc.count(w2)) return nullptr;
    return std::strcmp(t1->second, t2->second) ? nullptr : t1->second;
}
static void gq_qkv(RadBuilder* b, Model& m, Layer& L, int64_t rows, rad_buf xn, rad_buf qr, rad_buf kv) {
    if (!gq_declared() || !L.q_a || !L.wkv) return;   /* kv_g: declared beside by the caller */
    const char* ty = gq_type(m, L.wq_a, L.wkv);
    if (!ty) return;
    L.qkv_pair = fused(b, "gemm_nt_ggml_pair",
                       { RAD_RANGE("M", 1, rows), RAD_INT("N", m.c.q_lora), RAD_INT("N2", m.c.head_dim),
                         RAD_INT("K", m.c.n_embd), RAD_STR("type", ty), RAD_STR("dtype", m.g.dtype) },
                       { L.wq_a, L.wkv }, { xn }, { qr, kv });
}
static void gq_oa(RadBuilder* b, Model& m, Layer& L, int64_t rows, int64_t gd, rad_buf o, rad_buf oa) {
    if (!gq_declared() || m.gr != 2 || L.o_a.size() != 2 || L.wo_a.size() != 2 || !L.o_a[0] || !L.o_a[1]) return;
    const char* ty = gq_type(m, L.wo_a[0], L.wo_a[1]);
    if (!ty) return;
    L.oa_pair = fused(b, "gemm_nt_ggml_pair",
                      { RAD_RANGE("M", 1, rows), RAD_INT("N", m.c.o_rank), RAD_INT("N2", m.c.o_rank), RAD_INT("K", gd),
                        RAD_STR("type", ty), RAD_STR("dtype", m.g.dtype) },
                      { L.wo_a[0], L.wo_a[1] }, { o }, { oa });
}
/* At issue: gq_qkv_on -- q_a + kv_g over n rows as the pair (the switch's bit 0, the layer declared it, n in the band);
 * else the caller issues the two ops. gq_oa_on the same for the output groups (bit 1). */
static bool gq_qkv_on(const Layer& L, int64_t n) { return L.qkv_pair && n >= 1 && n <= kBiBand && (gq_now() & 1); }
static bool gq_oa_on(const Layer& L, int64_t n) { return L.oa_pair && n >= 1 && n <= kBiBand && (gq_now() & 2); }
static void gq_qkv_issue(RadCtx* x, const Model& m, const Layer& L, int64_t n, RadOperand xn, RadOperand qr,
                         RadOperand kv) {
    RAD_ISSUE_N(x, L.qkv_pair, n, xn, RAD_W(L.wq_a), RAD_NONE, RAD_W(L.wkv), qr, kv);
    if (m.g.rank == 0) g_gq_qkv.fetch_add(1, std::memory_order_relaxed);
}
/* ... and the two attention_output_a groups (o's column halves -> oa's) under bit 1 */
static void gq_oa_issue(RadCtx* x, const Model& m, const Layer& L, int64_t n, RadOperand o0, RadOperand o1,
                        RadOperand y0, RadOperand y1) {
    RAD_ISSUE_N(x, L.oa_pair, n, o0, RAD_W(L.wo_a[0]), o1, RAD_W(L.wo_a[1]), y0, y1);
    if (m.g.rank == 0) g_gq_oa.fetch_add(1, std::memory_order_relaxed);
}

/* Lever 1U-E2 (ds41-int11): a GGUF weight row's stored bytes (gemm_nt_ggml's native blocks: the row step of a row view);
 * 0 for a type this build cannot name. */
static int64_t ggml_row_bytes(const char* ty, int64_t k) {
#ifdef RAD_ARCH_HAVE_GGML
    for (uint32_t t = 0; ty && t < 64; ++t) {
        const lq::ggml::PlaneKind* pk = lq::ggml::plane_kind(t);
        if (pk && pk->name && !std::strcmp(pk->name, ty) && pk->block > 0 && k % pk->block == 0)
            return k / pk->block * (int64_t)pk->block_bytes;
    }
#endif
    (void)ty;
    (void)k;
    return 0;
}

/* Lever 1U-E2 at an Engram layer (RADIANCE_DS41_E2 declared, world > 1): this rank's part GEMV over rows
 * [rank n / world, (rank + 1) n / world) of the replicated kv weight (issued through a row view, so nothing is loaded
 * twice) into the shared part buffer, and the row-interleaved all_gather of the parts into engram_kv, both at the decode
 * band's heights (<= kBiBand rows). Left undeclared (the whole GEMV only) where the weight is not a GGUF block type, n
 * does not split evenly into whole 16-byte rows, or no library serves the gather. */
static int e2_layer(RadBuilder* b, Model& m, Layer& L, int64_t T, int64_t n, int64_t k) {
    const auto& g = m.g;
    const auto ty = m.wty.find(L.e_kv);
    if (ty == m.wty.end() || m.wsc.count(L.e_kv) || n % g.world || ((n / g.world) * 2) % 16) return RAD_OK;
    const int64_t P = n / g.world, R = std::min<int64_t>(T, kBiBand), rb = ggml_row_bytes(ty->second, k);
    if (rb <= 0) return RAD_OK;
    if (!m.ekv_p) {
        m.ekv_p = decl_b(b, m.nm.f("engram_kv_part"), RAD_BF16, { R, P });
        if (!m.ekv_p) return RAD_E_INVAL;
        m.e2_rows = R;
        m.e2_part = P;
        m.e2_rb = rb;
    }
    if (m.e2_part != P || m.e2_rb != rb) return RAD_OK;   /* one part shape for the model's Engram layers */
    const rad_op gp = op(b, "gemm_nt_ggml",
                         { RAD_RANGE("M", 1, R), RAD_INT("N", P), RAD_INT("K", k), RAD_STR("type", ty->second),
                           RAD_STR("dtype", g.dtype) },
                         { L.e_kv }, { m.embd }, { m.ekv_p });
    if (!gp) return RAD_OK;
    const rad_op ga = op(b, "all_gather",
                         { RAD_INT("world_size", g.world), RAD_RANGE("numel", P, R * P), RAD_STR("dtype", "bf16"),
                           RAD_INT("row", P) },
                         {}, { m.ekv_p }, { m.ekv });
    if (!ga) return RAD_OK;
    L.e_gemm_p = gp;
    L.e_gath = ga;
    return RAD_OK;
}

/* Lever 1U-E2 at issue: the Engram kv GEMV of n rows (a -> y, engram_kv rows) as this rank's column part and the gather
 * when the switch is on, the layer declared it and n is in the band; else the whole GEMV (lin_issue). Exact either way:
 * each output keeps gemm_nt_ggml's per-output sequence (the band's forms give a row the same bytes at any N), and the
 * gather moves the parts' bytes into their columns. */
static void e2_gemm(RadCtx* x, Model& m, const Layer& L, int64_t n, RadOperand a, RadOperand y) {
    if (L.e_gemm_p && L.e_gath && n >= 1 && n <= m.e2_rows && e2_now() && !tw_take(m, L.e_gemm)) {   /* P30t: the twin */
        RadOperand w = RAD_W(L.e_kv);
        w.offset = (int64_t)m.g.rank * m.e2_part * m.e2_rb;   /* u8 elements of the stored rows = bytes */
        w.rows = m.e2_part;
        RAD_ISSUE_N(x, L.e_gemm_p, n, a, w, brows(m.ekv_p, n));
        RAD_ISSUE_N(x, L.e_gath, n * m.e2_part, brows(m.ekv_p, n), y);
        if (m.g.rank == 0) {
            g_e2_splits.fetch_add(1, std::memory_order_relaxed);
            g_e2_rows.fetch_add(n, std::memory_order_relaxed);
        }
        return;
    }
    lin_issue(x, m, L.e_gemm, n, a, L.e_kv, y);
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
    /* (layer ranges: only where a held layer writes or reads it) */
    if (m.kwords > 0 && lr_keep_used(m)) m.keep = decl_b(b, nm.f("index_keep"), RAD_I32, { T, m.kwords });
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
    if (!c.ple.empty() && lr_engram(m)) {   /* (layer ranges: a node that holds an Engram layer) */
        /* tp8 loop FINDINGS F43: lever TI's snapshot inside a step patches its CONV window from these ids AFTER the pass
         * (rad_ckpt_conv_src: Engine::save_checkpoints behind the leader's tail part, pp_ck_apply behind the follower's
         * pass), past the last declared use -- the last held Engram layer's ngram_ids. With layer ranges that use comes
         * early (W(10,32): op 242 of the follower's 1,529 at layer 14, op 61 of the leader's at layer 1) and the packer
         * lays later transients over the ids at every arena level (index_score, engram_kv, comp_stage; on the leader
         * dspark_*), so the patch copied their bytes: a resume from such a snapshot answered differently on the pair. So
         * a node with layer ranges keeps them for the whole step (PERSIST, T x 4 B = 16.25 KiB). One process (every
         * layer) keeps the published declare: there no transient shares their bytes at any level (host declare with
         * RADIANCE_DEBUG_GRAPH_ALIAS=engram_tokens, 18.11.). */
        m.ctok = decl_b(b, nm.f("engram_tokens"), RAD_I32, { T }, m.lr.on ? RAD_BUF_PERSIST : RAD_BUF_TRANSIENT);
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
    m.iqr_qk = QK;
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
    /* Lever DKV (dkv_read_env): the plan's rows of a prefill pass (E entries a row of the step) and a decode pass's
     * gathered selections, over the packed latent only */
    m.lu = m.selkv = 0;
    m.lu_e = m.selkv_rows = m.selkv_cap = 0;
    if (m.lnv && c.head_dim == 512 && c.itopk > 0) {
        if (const int e = g_dkv_lu.load(std::memory_order_relaxed); e > 0) {
            m.lu_e = e;
            m.lu = bf("comp_latent_unpacked", { T, (int64_t)e * c.head_dim });
            if (m.lu) m.bp[m.lu] = (int64_t)e * c.head_dim;
        }
        if (g_dkv_g.load(std::memory_order_relaxed) > 0) {
            m.selkv_rows = std::min<int64_t>(T, kBiBand);
            /* tp8 loop D25: with RADIANCE_DS41_DKV_GPIECE=1 the buffer holds a 64-row segment's pieces side by side; the ops'
             * row range (selkv_rows) stays the band's, so every launch is the one today issues */
            m.selkv_cap = dkv_gpiece_env() ? std::min<int64_t>(T, kDkvGpRows) : m.selkv_rows;
            /* the env alone (not T): the sizing declare at a smaller max_tok must declare the same parameter lists */
            m.selkv_one = g_dkv_gp.load(std::memory_order_relaxed) == 2;
            m.selkv = bf("comp_latent_gathered", { m.selkv_cap, c.itopk * c.head_dim });
        }
    }
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
        /* Lever 1U-E2: the decode band's split by output columns (e2_issue), beside the whole GEMV */
        if (e2_declared() && g.world > 1) DS_TRY(e2_layer(b, m, L, T, (c.hc + 1) * D, H * c.ple_dim));
        /* Lever E1: the GEMM over the unique rows into engram_kv_u, then every pooled row's kv row gathered from it
         * (gather_rows, a byte copy) through this layer's map */
        if (edd) {
            L.edd_map = decl_b(b, nm.f("blk.%d.engram_dd_map", l), RAD_I32, { T }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
            if (L.edd_map) m.bp[L.edd_map] = 1;
            L.e_gemm_u = linear(b, m, T, L.e_kv, (c.hc + 1) * D, H * c.ple_dim, m.embd, m.ekv_u);
            L.e_scat = op(b, "gather_rows", { RAD_RANGE("M", 1, T), RAD_INT("n", (c.hc + 1) * D), RAD_STR("dtype", "bf16") },
                          {}, { m.ekv_u, L.edd_map }, { m.ekv });
        }
        /* Pictures: the media rows' value columns zeroed in front of the gate, so it adds nothing there
         * (model.py Engram's token_mask); fill and scatter at this layer, so their liveness is the layer's. */
        if (m.media) {
            L.mm_fill = op(b, "fill", { RAD_RANGE("M", 1, kMmZeroRows), RAD_INT("n", D), RAD_STR("dtype", "bf16"),
                                        RAD_F64("value", 0.0) },
                           {}, {}, { m.mm_zero });
            L.mm_put = op(b, "scatter_rows", { RAD_RANGE("M", 1, kMmZeroRows), RAD_INT("n", D), RAD_STR("dtype", "bf16") },
                          {}, { m.mm_zero, m.ekv }, { m.ekv });
            if (!L.mm_fill || !L.mm_put) return RAD_E_INVAL;
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
         * block (above) rewrites h0 between them (or, a proxy node's first layer, nothing ran before it here). */
        if (l > 0 && !L.e_ids && !lr_synth_start(m, l))
            L.pcn_a = post_collapse_norm(b, m, T, L.attn_norm, m.act.x, m.h1, m.h0, m.pre_f,
                                         m.xcol, m.xn);
        /* lever M1 (RADIANCE_DS41_MHC_FUSE): pcn_a + mix_a as one op, beside the pair (fused(): 0 where no library
         * declares and serves it, and the lever is then not available) */
        if (L.pcn_a && mhc_declared() && c.hc == 4 && D % 1024 == 0)
            L.pcn_mix = fused(b, "mhc_post_collapse_rmsnorm_mix",
                              { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_INT("iterations", c.hc_iters),
                                RAD_F64("eps", c.eps), RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") },
                              { L.attn_norm, L.hc_a_fn, L.hc_a_scale, L.hc_a_base },
                              { m.act.x, m.h1, m.post, m.comb, m.pre_f }, { m.h0, m.xcol, m.xn, m.pre_a, m.post, m.comb });
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
        gq_qkv(b, m, L, T, m.xn, m.qr, m.kv);   /* lever 1U-GQ: q_a + kv_g as one op, beside them */
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
            /* tp8 S4 hw (layer ranges / a pipeline node only): k_f / l_f / k_rope / l_rope are issued with comp_bpos
             * (the entries' positions) but declare only cs_b; with every layer a later layer's c_work / c_tab keeps
             * comp_bpos alive, on a node whose last compressing layer is this one the plan ended its life at this
             * c_tab and the step refused (M gpu03 17.11. s3hw-B-3, RADIANCE_DS41_LAYERS=0:20: "operand 4 names buffer
             * 'comp_bpos' ... ops 123..891 and this is op 893"; the S3 hardware agent's bpos patch, applied here for
             * the W(0,20) follower). Unset: nothing declared here (the published plan). */
            if (m.lr.on && L.k_f && rw(b, L.k_f, { m.bpos }, {}) == 0) return RAD_E_INVAL;
            if (!L.k_f) {
                L.k_norm = rmsnorm(b, m, rows, L.ik_norm, c.idim, m.ik, m.ikn);
                L.k_rope = rope_rows(b, m, rows, 1, c.idim, 0, m.ikn, m.cs_b);
                if (m.lr.on && L.k_rope && rw(b, L.k_rope, { m.bpos }, {}) == 0) return RAD_E_INVAL;
                L.k_q = roundtrip(b, rows, c.idim, 32, "fp4_e2m1", "ue8m0", m.ikn);
            }
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", index_slots(m), r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
            L.l_f = rope_rt(b, m, rows, 1, hd, 0, 16, "fp4_e2m1", "e4m3", m.lat, m.cs_b);
            if (m.lr.on && L.l_f && rw(b, L.l_f, { m.bpos }, {}) == 0) return RAD_E_INVAL;   /* tp8 S4 hw: as k_f */
            if (!L.l_f) {
                L.l_rope = rope_rows(b, m, rows, 1, hd, 0, m.lat, m.cs_b);
                if (m.lr.on && L.l_rope && rw(b, L.l_rope, { m.bpos }, {}) == 0) return RAD_E_INVAL;
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
            /* Lever DKV: the fold d form over the plan's / the gathered rows (beside it) */
            if (r > 0 && L.attn_rt && m.lu) {
                L.attn_rtlu = fused(b, "attn_sparse_sink_rope_roundtrip_unpacked", fp, { L.sinks }, { m.q, m.sel, cs, m.lu },
                                    { m.o });
                std::vector<RadParam> ff = fp;
                ff.push_back(RAD_INT("plan_full", 1));
                L.attn_rtluf = fused(b, "attn_sparse_sink_rope_roundtrip_unpacked", ff, { L.sinks }, { m.q, m.sel, cs, m.lu },
                                     { m.o });
            }
            if (r > 0 && L.attn_rt && m.selkv) {
                std::vector<RadParam> gf = fp;
                gf[0] = RAD_RANGE("M", 1, m.selkv_one ? m.selkv_cap : m.selkv_rows);
                if (m.selkv_one) gf.push_back(RAD_INT("band", kDkvGpRows));   /* tp8 loop D25 step 3: libr4d's decode band to 64 */
                L.attn_rtg = fused(b, "attn_sparse_sink_rope_roundtrip_gathered", gf, { L.sinks },
                                   { m.q, m.sel, cs, m.selkv }, { m.o });
            }
        }
        /* Lever DKV (dkv_read_env): latent_unpack / latent_gather over this layer's kv source's packed cache, and the
         * attention over their rows, beside today's (issued where dkv_attn finds them) */
        if (r > 0 && m.lu) {
            L.lu_unpack = fused(b, "latent_unpack",
                                { RAD_RANGE("M", 1, T), RAD_INT("dim", hd), RAD_INT("ratio", r),
                                  RAD_STR("latent_format", "nvfp4") },
                                {}, {}, { m.lu });
            L.attn_lu = fused(b, "attn_sparse_sink_unpacked", ap, { L.sinks }, { m.q, m.sel, m.lu }, { m.o });
            std::vector<RadParam> af = ap;
            af.push_back(RAD_INT("plan_full", 1));
            L.attn_luf = fused(b, "attn_sparse_sink_unpacked", af, { L.sinks }, { m.q, m.sel, m.lu }, { m.o });
        }
        if (r > 0 && m.selkv) {
            L.lgath = fused(b, "latent_gather",
                            { RAD_RANGE("M", 1, m.selkv_one ? m.selkv_cap : m.selkv_rows), RAD_INT("dim", hd), RAD_INT("ratio", r),
                              RAD_INT("topk", c.itopk), RAD_STR("latent_format", "nvfp4") },
                            {}, { m.sel }, { m.selkv });
            std::vector<RadParam> gp = ap;
            gp[0] = RAD_RANGE("M", 1, m.selkv_one ? m.selkv_cap : m.selkv_rows);
            if (m.selkv_one) gp.push_back(RAD_INT("band", kDkvGpRows));   /* tp8 loop D25 step 3 */
            L.attn_g = fused(b, "attn_sparse_sink_gathered", gp, { L.sinks }, { m.q, m.sel, m.selkv }, { m.o });
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
            rad_buf oy = m.oa;
            if (m.wsc.count(L.wo_a.back())) {   /* tp8 loop P30f: an IQR group writes its own buffer (linear_oab) */
                if (m.oa_g.size() != (size_t)m.gr) {
                    m.oa_g.assign((size_t)m.gr, 0);
                    for (int i = 0; i < m.gr; ++i) {
                        m.oa_g[(size_t)i] = decl_b(b, nm.f("attn_o_lora_g%d", i), RAD_BF16, { T, c.o_rank });
                        if (!m.oa_g[(size_t)i]) return RAD_E_INVAL;
                        m.bp[m.oa_g[(size_t)i]] = c.o_rank;
                    }
                }
                if (L.oa_dst.size() != (size_t)m.gr) L.oa_dst.assign((size_t)m.gr, 0);
                oy = L.oa_dst[(size_t)j] = m.oa_g[(size_t)j];
            }
            L.o_a.push_back(linear(b, m, T, L.wo_a.back(), c.o_rank, gd, m.o, oy, false));   /* P30t: twins below */
        }
        gq_oa(b, m, L, T, gd, m.o, m.oa);   /* lever 1U-GQ: the two groups as one op, beside them */
        const int64_t ok = m.gr * c.o_rank;
        L.wo_b = lw(b, m, nm.f("blk.%d.attn_output_b.weight", l), D, ok, grp, RAD_SHARD_COL,
                    m.ar * ok, (m.ar + 1) * ok);
        if (oa_split(L)) {
            L.o_b = linear_oab(b, m, L, T, L.wo_b, D, ok, c.o_rank, m.oa, m.attn_out);
            if (!L.o_b) return RAD_E_INVAL;
            if (m.g.rank == 0 && !m.oa_said++)
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: IQR attention_output_a groups write their own "
                                "[%lld, %lld] buffers and o_b quantises them a group at a time (libr4d's int8 GEMMs "
                                "write C at pitch N only)\n", (long long)T, (long long)c.o_rank);
        } else {
            L.o_b = linear(b, m, T, L.wo_b, D, ok, m.oa, m.attn_out, false);   /* P30t: its twin below */
        }
        /* tp8 loop P30t: the output projections' twins, all or none: each twin group writes its own [T, o_rank] buffer
         * (Model::oa_g: libr4d's int8 / fp8 GEMMs write C at pitch N only), o_b's twin quantises those a group at a time
         * (P30f's form); a prompt segment issues the four, a decode row the GGUF ops over attn_o_lora */
        int ntw = m.twin.count(L.wo_b) ? 1 : 0;
        for (const rad_weight w : L.wo_a) ntw += m.twin.count(w) ? 1 : 0;
        if (ntw == (int)L.wo_a.size() + 1 && !oa_split(L)) {
            if (m.oa_g.size() != (size_t)m.gr) {
                m.oa_g.assign((size_t)m.gr, 0);
                for (int i = 0; i < m.gr; ++i) {
                    m.oa_g[(size_t)i] = decl_b(b, nm.f("attn_o_lora_g%d", i), RAD_BF16, { T, c.o_rank });
                    if (!m.oa_g[(size_t)i]) return RAD_E_INVAL;
                    m.bp[m.oa_g[(size_t)i]] = c.o_rank;
                }
            }
            for (int j = 0; j < m.gr; ++j) {
                const rad_weight tw = m.twin.at(L.wo_a[(size_t)j]);
                const rad_op th = linear(b, m, T, tw, c.o_rank, gd, m.o, m.oa_g[(size_t)j], false);
                if (!th) return RAD_E_INVAL;
                m.twop[L.o_a[(size_t)j]] = th;
                m.twcls[th] = kTwOab;
            }
            const rad_op thb = linear_oab(b, m, L, T, m.twin.at(L.wo_b), D, ok, c.o_rank, m.oa, m.attn_out, &m.oa_g);
            if (!thb) return RAD_E_INVAL;
            m.twop[L.o_b] = thb;
            m.twcls[thb] = kTwOab;
            m.twin_mask |= kTwOab;
            L.oa_tw = true;
        } else if (ntw > 0 && m.g.rank == 0) {
            fprintf(stderr, "radiance: deepseek41: tp8 loop P30t: layer %d holds %d of the %zu output-projection twins (every "
                            "attention_output_a group + attention_output_b, or none%s): its output projections stay GGUF\n",
                    L.l, ntw, L.wo_a.size() + 1, oa_split(L) ? "; and its groups are IQR already" : "");
        }
    }
    /* The heads' partial outputs (zeros on an attention-zero rank) summed over every rank. */
    if (g.world > 1)
        L.ar_attn = op(b, "all_reduce",
                       { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                         RAD_INT("min_bytes", g.wire_min_bytes) },
                       {}, { m.attn_out }, { m.attn_out });
    /* tp8 loop P13b: the same collective at `overlap` 1 for RADIANCE_DS41_PIPE2_ATTN's piece 0 (libr4d's narrow coded launch
     * of the same fold, beside piece 1's output projection) */
    if (g.world > 1 && m.pipe2 > 0 && m.pipe2_attn)
        L.ar_attn_ov = op(b, "all_reduce",
                          { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                            RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                            RAD_INT("min_bytes", g.wire_min_bytes), RAD_INT("overlap", 1) },
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
    /* lever M1: pcn_f + mix_f as one op, beside the pair */
    if (L.pcn_f && mhc_declared() && c.hc == 4 && D % 1024 == 0)
        L.pcn_mix_f = fused(b, "mhc_post_collapse_rmsnorm_mix",
                            { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_INT("iterations", c.hc_iters),
                              RAD_F64("eps", c.eps), RAD_F64("hc_eps", c.hc_eps), RAD_STR("dtype", "bf16") },
                            { L.ffn_norm, L.hc_f_fn, L.hc_f_scale, L.hc_f_base },
                            { m.attn_out, m.h0, m.post, m.comb, m.pre_a }, { m.h1, m.xcol, m.act.x, m.pre_f, m.post, m.comb });
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
    /* Lever 1U-ETP (etp_env): every expert on every rank, as this rank's width slice by parity class */
    if (m.etp) {
        int64_t lo[2], hi[2];
        if (!etp_slice(c.n_ff, g.world, g.rank, lo, hi)) return RAD_E_UNSUPPORTED;
        s.cfg.n_expert = c.n_expert;
        s.cfg.expert_base = 0;
        s.cfg.probed_expert = 0;
        for (int q = 0; q < 2; ++q) {
            s.cfg.ff_lo[q] = lo[q];
            s.cfg.ff_hi[q] = hi[q];
        }
        s.cfg.n_ff_exp = std::max(hi[0] - lo[0], hi[1] - lo[1]);
        s.cfg.exl3_xring = xring_env() > 0 ? xring_env() : 0;   /* lever XR */
    }
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
    s.cfg.shared_twin = (twin_env() & kTwShx) && g.n_ff > 0 ? 1 : 0;   /* tp8 loop P30t */
    /* Lever FUSE-R: the router chain's fused forms beside the separate ops (RADIANCE_DS41_RFUSE set) */
    s.cfg.rfuse_decl = rf_declared() ? 1 : 0;
    /* Lever P: the decode-band-bytes EXL3 pair over 17 .. kMoe1Max rows beside gate_up / down (RADIANCE_DS41_MOE1 set) */
    s.cfg.exl3_dk_rows = p1_declared() ? moe1_max() : 0;
    if (m.sized_against) {   /* a sizing declare: the pair exactly where the real declare's layer has it (also at an 8-row level) */
        const Model& full = *m.sized_against;
        s.cfg.exl3_dk_rows = (size_t)l < full.layers.size() ? full.layers[(size_t)l].moe.dk_asked : 0;
        s.cfg.exl3_dk_probe = 1;
    }
    DS_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
    /* tp8 loop P13: the MoE's all-reduce declared a second time at `overlap` 1 (op_ar_ov: the same fold, the narrow launch
     * beside other work) for RADIANCE_DS41_PIPE2's piece 0 on the second lane */
    L.moe.ar_overlap = m.pipe2 > 0 ? 1 : 0;
    DS_TRY(L.moe.declare(b, nm, g, l, m.mw, s, false, false));
    /* Pictures: the image rows' routing bias (model.py Gate.bias_vl), its selection and the byte copies over the
     * router's picks (MoeFP8::Vl), declared at this layer so their liveness is the layer's */
    if (m.media) {
        auto& v = L.moe.vl;
        const char* bn = nm.f("blk.%d.exp_probs_b.bias_vl", l);
        DS_TRY(map_copy(b, bn, nm.ckpt("layers.%d.ffn.gate.bias_vl", l)));
        v.bias = decl_w(b, bn, RAD_F32, { c.n_expert }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        if (!v.bias) return RAD_E_INVAL;
        v.select = op(b, "router_scored_topk",
                      { RAD_RANGE("M", 1, T), RAD_INT("n_expert", c.n_expert), RAD_INT("top_k", c.top_k),
                        RAD_INT("norm", c.norm_topk), RAD_F64("route_scale", c.route_scale),
                        RAD_STR("score", "sqrtsoftplus"), RAD_STR("dtype", "f32") },
                      { v.bias }, { m.mw.logits }, { m.vl_ids, m.vl_ew });
        v.gather = op(b, "gather_rows", { RAD_RANGE("M", 1, T), RAD_INT("n", 4 * c.top_k), RAD_STR("dtype", "u8") },
                      {}, { m.vl_ids, m.vl_ew }, { m.vl_tid, m.vl_tew });
        v.scatter = op(b, "scatter_rows", { RAD_RANGE("M", 1, T), RAD_INT("n", 4 * c.top_k), RAD_STR("dtype", "u8") },
                       {}, { m.vl_tid, m.vl_tew, m.mw.ids, m.mw.ew }, { m.mw.ids, m.mw.ew });
        v.ids = m.vl_ids;
        v.ew = m.vl_ew;
        v.tid = m.vl_tid;
        v.tew = m.vl_tew;
        if (!v.select || !v.gather || !v.scatter) return RAD_E_INVAL;
    }
    if (p1_declared() && L.moe.dk_takes(kBiMoe + 1)) g_sw_p1_ok.store(1, std::memory_order_relaxed);
    /* the in-process switches (rad_fn4_*): the fused gather's live bound, the shared arm's lane */
    L.moe.gar_live = &g_sw_gar;
    L.moe.shexp_lane = &g_sw_shexp;
    if (L.moe.sh_tw) {   /* tp8 loop P30t: the shared arm's int8 twins */
        L.moe.twin_now = &twin_shx_now;
        m.twin_mask |= kTwShx;
    }
    if (s.cfg.shared_one_external) {
        L.moe.sone_live = &g_sw_one;
        if (L.moe.gfold) g_sw_one_ok.store(1, std::memory_order_relaxed);
    }
    L.moe.swfold_live = &g_sw_swf;
    if (L.moe.op_swfold) g_sw_swf_ok.store(1, std::memory_order_relaxed);
    if (rf_declared()) {
        L.moe.rfuse_live = &g_sw_rf;
        L.moe.rf_count = g_rf_count;
        if (L.moe.op_rs_fused) g_sw_rf_ok.fetch_or(1, std::memory_order_relaxed);
        if (L.moe.op_rgs_fused) g_sw_rf_ok.fetch_or(2, std::memory_order_relaxed);
    }
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
    /* tp8 loop D18: the short block's rows (every rank: the walks' collectives run on all). */
    d.sblks.clear();
    d.sh.clear();
    if (const char* e = std::getenv("RADIANCE_DS41_DSPARK_SHORT"); e && *e && std::strcmp(e, "0") != 0) {
        /* tp8 loop D19: a comma list, each length once */
        const char* q = e;
        bool bad = false;
        while (*q && !bad) {
            char* end = nullptr;
            const long v = std::strtol(q, &end, 10);
            if (end == q || (*end && *end != ',') || v < 1 || v >= blk ||
                std::find(d.sblks.begin(), d.sblks.end(), (int64_t)v) != d.sblks.end() || d.sblks.size() >= 4) {
                bad = true;
                break;
            }
            d.sblks.push_back(v);
            q = *end ? end + 1 : end;
            if (*end && !*q) bad = true;
        }
        if (bad || d.sblks.empty()) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_SHORT=%s: the DSpark short block takes 1 .. %lld "
                            "rows (the drafter's own block is %lld; 0 or unset: none; tp8 loop D19: a comma list of "
                            "distinct lengths, at most 4)\n", e, (long long)(blk - 1), (long long)blk);
            return RAD_E_INVAL;
        }
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
        /* lever 1U-GQ: the pair is issued at q_a's point, ahead of q_an / q_b / q_rope (kv_g's own point is after
         * them), so it is declared here: d.kv_'s planned life then starts before q_an, and no buffer of the query
         * chain can share its bytes. kv_g itself is declared below as before (the stage fails without it). */
        gq_qkv(b, m, L, Q, d.xn, d.qr, d.kv_);
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
        /* lever DMR 2 (dspark_mhr 2): the window key's norm, rope, round trip and store as ONE op -- the trunk's lever K1
         * fold c (rmsnorm_rope_roundtrip_kv_store, served at ds41_kvfold 1 on the trunk's own window: the same row
         * positions, the same cache layout and kv_dtype), declared beside the four ops; 0 where no library serves it.
         * tp8 loop (DECODE merge, 19.11.): declared only while RADIANCE_DS41_DSPARK_MHR is set (any value; 0 starts the switch at
         * 0), so an unset start declares the published program; dspark_mhr 2 without the env finds no fold and keeps the four. */
        L.kv_fs = !std::getenv("RADIANCE_DS41_DSPARK_MHR") ? 0 : fused(b, "rmsnorm_rope_roundtrip_kv_store",
                        { RAD_RANGE("M", 1, Q), RAD_INT("n", hd), RAD_F64("eps", c.eps), RAD_INT("n_head", 1),
                          RAD_INT("head_dim", hd), RAD_INT("rot", c.rope), RAD_INT("off", hd - c.rope),
                          RAD_STR("mode", "gptj"), RAD_INT("inverse", 0), RAD_INT("block", 32),
                          RAD_STR("format", "fp8_e4m3"), RAD_STR("scale", "ue8m0"), RAD_STR("dtype", "bf16"),
                          RAD_INT("block_size", d.kv_bs), RAD_STR("kv_dtype", m.w8d ? "fp8_e4m3" : "bf16") },
                        { L.kv_norm }, { d.kv_, m.cs_a }, { d.kvn });
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
        for (int j = 0; j < m.gr; ++j) {
            if (m.wsc.count(L.wo_a[(size_t)j])) {   /* tp8 loop P30f: the trunk's grouped form only */
                fprintf(stderr, "radiance: deepseek41: tp8 loop P30f: the drafter's attention_output_a group %d is IQR; "
                                "its column slice of the drafter's attn_o_lora is not served by libr4d's int8 GEMMs\n", j);
                return RAD_E_UNSUPPORTED;
            }
            L.o_a.push_back(linear(b, m, Q, L.wo_a[(size_t)j], c.o_rank, gd, d.o, d.oa));
        }
        gq_oa(b, m, L, Q, gd, d.o, d.oa);   /* lever 1U-GQ */
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

    /* tp8 loop D18: the short block's ops (declared after every other DSpark op: what they read stays live to here). */
    const char* sbwhy = nullptr;
    static thread_local char sbwhy_s[160];
    if (!d.sblks.empty() && d.n_alt > 0) sbwhy = "not with RADIANCE_DS41_DSPARK_ALT (its branches walk the drafter's own block)";
    for (size_t si_ = 0; !sbwhy && si_ < d.sblks.size(); ++si_) {
        const int64_t SB = d.sblks[si_];
        Dsp::Short z;
        z.S = SB;
        if (!sbwhy) {
            /* apos = position + (SB - 1 - row % SB): each row at its short block's last position */
            z.poff = op(b, "iota", { RAD_RANGE("M", 1, Q), RAD_INT("start", SB - 1), RAD_INT("step", -1),
                                        RAD_INT("period", SB) },
                           {}, {}, { d.poff });
            if (!z.poff || !rad_op_resolved(b, z.poff)) sbwhy = "no kernel serves iota at the short block";
        }
        /* each stage's attention: the window's context keys before the anchor plus the short block's own SB keys */
        for (size_t i = 0; !sbwhy && i < d.st.size(); ++i) {
            const Layer& L = d.st[i];
            if (m.zero) {
                z.attn.push_back(0);   /* attn_fill in its place, as the drafter's own block */
                continue;
            }
            std::vector<RadParam> dap = { RAD_RANGE("M", 1, Q), RAD_INT("heads", g.n_head), RAD_INT("dim", hd),
                                          RAD_INT("window", c.window + SB), RAD_INT("ratio", 0), RAD_INT("topk", 0),
                                          RAD_F64("scale", 1.0 / std::sqrt((double)hd)), RAD_STR("dtype", "bf16") };
            if (m.w8d) dap.push_back(RAD_STR("window_format", "fp8_e4m3"));
            const rad_op a = op(b, "attn_sparse_sink", dap, { L.sinks }, { d.q, d.apos }, { d.o });
            if (!a || !rad_op_resolved(b, a)) sbwhy = "no kernel serves attn_sparse_sink at the short block's window";
            z.attn.push_back(a);
        }
        if (!sbwhy) {
            z.select = op(b, "dflash_select",
                             { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", SB), RAD_INT("top_k", R),
                               RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", SB),
                               RAD_STR("dtype", "bf16") },
                             { d.mk1, d.mk2 }, { d.cand_, d.unary }, { d.tokens });
            if (!z.select || !rad_op_resolved(b, z.select)) sbwhy = "no kernel serves dflash_select at the short block";
        }
        /* W1 at SB steps on the same planes (step l's rows [l * wseqs, ...), SB * wseqs <= block * wseqs) */
        if (!sbwhy && d.wfill) {
            for (int64_t l = 0; l <= SB && !sbwhy; ++l) {
                const rad_op w = op(b, "dspark_walk_step",
                                    { RAD_RANGE("M", 1, d.wseqs), RAD_INT("step", l), RAD_INT("steps", SB),
                                      RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("N", g.n_vocab),
                                      RAD_INT("vocab_off", g.vocab_off), RAD_INT("world", g.world),
                                      RAD_INT("anchor_stride", SB), RAD_STR("dtype", "f32") },
                                    { d.mk1, d.mk2 }, { d.logits, d.wgath, d.wkeys }, { d.wkeys, d.tokens });
                if (!w || !rad_op_resolved(b, w)) sbwhy = "dspark_walk_step at the short block: no kernel serves it";
                z.wstep.push_back(w);
            }
        } else if (!sbwhy && d.wmarkov) {
            z.wmarkov = op(b, "dflash_markov_greedy_f32",
                              { RAD_RANGE("M", 1, d.wseqs), RAD_INT("steps", SB), RAD_INT("rank", d.rank),
                                RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", SB), RAD_STR("dtype", "bf16") },
                              { d.mk1, d.mk2 }, { d.logits }, { d.wts, d.wtt, d.tokens });
            if (!z.wmarkov || !rad_op_resolved(b, z.wmarkov)) sbwhy = "dflash_markov_greedy_f32 at the short block";
        }
        /* SD at SB steps on the same planes (keys [l * sseqs, ...), the state's first SB rounds, the counter's first SB rows) */
        if (!sbwhy && !d.sstep.empty()) {
            const rad_weight ws[2] = { d.mk1, d.mk2 };
            for (int64_t l = 0; l <= SB && !sbwhy; ++l) {
                const RadParam ps[] = { RAD_RANGE("M", 1, d.sseqs), RAD_INT("step", l), RAD_INT("steps", SB),
                                        RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("N", g.n_vocab),
                                        RAD_INT("vocab_off", g.vocab_off), RAD_INT("world", g.world),
                                        RAD_INT("anchor_stride", SB), RAD_STR("dtype", "f32") };
                const rad_op w = rad_decl_op(b, "dspark_walk_sample_step", ps, (int)(sizeof ps / sizeof ps[0]), ws, 2);
                if (!w || !rad_op_resolved(b, w)) {
                    sbwhy = "dspark_walk_sample_step at the short block: no kernel serves it";
                } else {
                    rw(b, w, { d.logits, d.sgath, d.skeys, d.sscr, d.scnt, d.sdraw, d.sqt, d.sstate },
                       { d.skeys, d.tokens, d.sscr, d.scnt, d.sstate });
                }
                z.sstep.push_back(w);
            }
        }
        if (!sbwhy && d.conf_op) {
            z.conf = op(b, "dspark_conf",
                           { RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", SB), RAD_INT("n_embd", D),
                             RAD_INT("rank", d.rank), RAD_INT("n_vocab", c.n_vocab), RAD_INT("anchor_stride", SB),
                             RAD_F64("bias", d.cbias), RAD_STR("dtype", "bf16") },
                           { d.mk1 }, { d.x, d.tokens, d.cwb }, { d.conf });
            if (!z.conf || !rad_op_resolved(b, z.conf)) sbwhy = "no kernel serves dspark_conf at the short block";
        }
        if (sbwhy && d.sblks.size() > 1) {   /* tp8 loop D19: which length */
            std::snprintf(sbwhy_s, sizeof sbwhy_s, "%s (short block %lld)", sbwhy, (long long)SB);
            sbwhy = sbwhy_s;
        }
        if (!sbwhy) d.sh.push_back(std::move(z));
    }
    {
        if (!sbwhy && !d.sblks.empty() && !rad_drafter_short_block) sbwhy = "this host has no rad_drafter_short_block (ABI 27)";
        if (sbwhy) d.sh.clear();
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
    /* tp8 loop D18 (D19: one call a length) */
    for (const Dsp::Short& z : d.sh) {
        const int64_t sblk = z.S;
        DS_TRY(rad_drafter_short_block(b, sblk));
        rad_note(b, "DSpark short block (tp8 loop D18, RADIANCE_DS41_DSPARK_SHORT=%lld): [anchor, noise x %lld] proposes %lld "
                    "token(s); attention over %lld context keys + its own %lld; walks / confidence head at %lld steps; the "
                    "engine's --draft-short-from picks it per step", (long long)sblk, (long long)(sblk - 1),
                 (long long)sblk, (long long)c.window, (long long)sblk, (long long)sblk);
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: DSpark short block (tp8 loop D18): %lld row(s) a sequence = %lld proposal(s) "
                            "(the drafter's own block %lld), its query pass at %lld rows a sequence whenever the engine's "
                            "--draft-short-from picks it (RADIANCE_DS41_DSPARK_SHORT=%lld)\n",
                    (long long)sblk, (long long)sblk, (long long)d.block, (long long)sblk, (long long)sblk);
    }
    if (d.sh.empty() && sbwhy) {
        rad_note(b, "DSpark short block (tp8 loop D18) NOT declared: %s", sbwhy);
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DSPARK_SHORT is set but the short block is NOT declared: %s\n",
                    sbwhy);
    }
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
static void mhc_declare(Model& m, const RadBuildCtx* ctx);                  /* lever M1, below */
static void i11_declare(Model& m, const RadBuildCtx* ctx);                  /* levers 1U-GQ / 1U-E2, below */
static void rf_declare(Model& m, const RadBuildCtx* ctx);                   /* lever FUSE-R, below */
static void p1_declare(Model& m, const RadBuildCtx* ctx);                   /* lever P, below */
static void pf_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx);   /* tp8 loop D24, below */

/* tp8 S3 (ds41_layers.h): RADIANCE_DS41_LAYERS / _MODE read against the model (read_cfg has run). Unset, or every layer:
 * m.lr.on stays false and nothing below this differs from the published declare. Refused where a lever's shape assumes
 * every layer (the AR_PCN collectives pair a layer with the one before it; --calib-layers is the calibration's own range). */
static int lr_read(Model& m, const RadBuildCtx* ctx) {
    std::string err;
    /* tp8 S4: a pipeline node's ranges come from its cut (--pp-cut), strict. */
    const RadPpInfo pp = rad_pp_query();
    if (pp.nodes == 2) {
        const int n = (int)m.c.n_layer;
        if (std::getenv("RADIANCE_DS41_LAYERS") && *std::getenv("RADIANCE_DS41_LAYERS")) {
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LAYERS and --pp-nodes 2 both name the layers; unset the former\n");
            return RAD_E_INVAL;
        }
        /* The leader holds the head, so it must hold the last layer (the S3 leader rule). */
        if (pp.cut_h < 0 || pp.cut_t <= pp.cut_h || pp.cut_t >= n) {
            fprintf(stderr, "radiance: deepseek41: --pp-cut %d,%d: the follower's layers must lie inside [0, %d) and leave "
                            "the leader the last one\n", pp.cut_h, pp.cut_t, n);
            return RAD_E_INVAL;
        }
        std::string spec;
        if (pp.rank == 1) spec = std::to_string(pp.cut_h) + ":" + std::to_string(pp.cut_t);
        else {
            if (pp.cut_h > 0) spec = "0:" + std::to_string(pp.cut_h) + ",";
            spec += std::to_string(pp.cut_t) + ":" + std::to_string(n);
        }
        if (ds41layers::parse(spec.c_str(), "strict", n, &m.lr, &err) < 0) {
            fprintf(stderr, "radiance: deepseek41: --pp-cut %d,%d: %s\n", pp.cut_h, pp.cut_t, err.c_str());
            return RAD_E_INVAL;
        }
        m.pp = true;
        m.pp_rank = pp.rank;
        m.pp_h = pp.cut_h;
        m.pp_t = pp.cut_t;
        {
            const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
            m.pp_fe = fe && !std::strcmp(fe, "1") && m.pp_h == 0;
            const char* ia = std::getenv("RADIANCE_PP_ISSUE_AHEAD");
            /* tp8 dev D10 round 2: at W(0, t), and at h > 0 where the follower holds an Engram layer (W(10,32): 14) -- both
             * nodes know it from the cut and the model; W(15,35)'s follower reads no id and keeps round 1's X1 */
            bool fol_engram = false;
            for (int l = m.pp_h; l < m.pp_t; ++l) fol_engram = fol_engram || m.c.ple_index(l) >= 0;
            m.pp_ids = ia && !std::strcmp(ia, "1") && (m.pp_h == 0 || fol_engram);
        }
        if (m.lr.leader != (pp.rank == 0)) return RAD_E_STATE;
    } else if (ds41layers::parse(std::getenv("RADIANCE_DS41_LAYERS"), std::getenv("RADIANCE_DS41_LAYERS_MODE"), (int)m.c.n_layer,
                          &m.lr, &err) < 0) {
        fprintf(stderr, "radiance: deepseek41: %s\n", err.c_str());
        return RAD_E_INVAL;
    }
    if (!m.lr.on) {
        if (!m.lr.text.empty() && ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LAYERS=%s holds every layer: the published engine\n",
                    m.lr.text.c_str());
        return RAD_OK;
    }
    const char* why = ds41_ar_pcn_mode() != 0                  ? "RADIANCE_DS41_AR_PCN's collectives pair a layer with the one before it"
                      : (ctx->calib_dir && ctx->calib_dir[0]) ? "calibration records its own layers (--calib-layers)"
                      : (m.pp && m.media)                     ? "pictures (the encoder's rows are not shipped to the follower)"
                                                              : nullptr;
    if (why) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LAYERS=%s is not built with this: %s\n", m.lr.text.c_str(), why);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* After dsp_config: the drafter only where its taps are held (and, strict, on the leader), then what the node's layers
 * read from another node. A proxy runs alone, so it is refused when that is anything (S6 builds the copies). */
static int lr_settle(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (!m.lr.on) return RAD_OK;
    bool taps = !m.dsp.taps.empty();
    for (int64_t t : m.dsp.taps) taps = taps && m.lr.held((int)t);
    const bool want = m.lr.strict ? m.lr.leader : taps;   /* strict: the leader's drafter, whatever it needs */
    if (m.dsp.on && !want) {
        rad_note(b, "DSpark drafter NOT declared on this node (RADIANCE_DS41_LAYERS=%s): %s", m.lr.text.c_str(),
                 m.lr.strict ? "the leader's (the node that holds the last layer)" : "its taps are another node's layers");
        if (m.pp) m.pp_dsp = m.dsp;   /* tp8 S4: its KV group is declared and zeroed here (pp_dspark_zero) */
        m.dsp = Dsp{};
    }
    m.lr_deps = ds41layers::needs(lr_geo(m), m.lr, m.dsp.on || (m.lr.strict && m.lr.leader && !m.dsp.taps.empty()));
    /* tp8 S6: the crossings of W(h, t) carry what each node's layers read of the other (crossings(): a source's new
     * entries for the receiver's copy, the selections, the keep words); a cut the geometry does not allow is refused */
    if (m.pp) {
        std::string err;
        if (ds41layers::crossings(lr_geo(m), m.pp_h, m.pp_t, m.pp_cx, &err) < 0) {
            fprintf(stderr, "radiance: deepseek41: --pp-cut %d,%d: %s\n", m.pp_h, m.pp_t, err.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const ds41layers::Crossing& in = m.pp_cx[m.pp_rank == 1 ? RAD_PP_X1 : RAD_PP_X2];
        m.pp_binds = ds41layers::binds(lr_geo(m), m.lr, &in);
        /* tp8 dev M4: the follower's segment boundaries -- every k-th layer after h whose cut W(s, t) the geometry allows */
        if (const char* sk = std::getenv("RADIANCE_PP_SEG_LAYERS"); sk && *sk && m.pp_rank == 1 && rad_pp_query().lanes > 1) {
            const int k = std::atoi(sk);
            for (int x = m.pp_h + std::max(k, 1); k > 0 && x < m.pp_t && (int)m.seg_b.size() < RAD_PP_MAX_SEGS - 1; x += k) {
                /* loop FINDINGS F12: X1 of W(x, t) without its source, plus what this pass's X2 ships that layers below x
                 * computed (ds41layers::segment_crossing) */
                ds41layers::Crossing sc;
                std::string e2;
                if (ds41layers::segment_crossing(lr_geo(m), m.pp_h, x, m.pp_t, &sc, &e2) < 0) {
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_PP_SEG_LAYERS=%d: no segment boundary at layer %d (%s)\n", k,
                            x, e2.c_str());
                    continue;
                }
                m.seg_b.push_back(x);
                m.seg_cx.push_back(sc);
            }
        }
        /* tp8 dev M9: the leader's boundaries -- every k-th layer of its head [0, h) and of its tail [t, n) (none past the
         * first DSpark tap) the geometry allows; not beside lever AR_PCN (a MoE all-reduce left to the next layer's fused
         * collapse would cross a boundary unreduced) */
        if (const char* lk = std::getenv("RADIANCE_PP_LEAD_SEG"); lk && *lk && m.pp_rank == 0 && rad_pp_query().lanes > 1) {
            const int k = std::atoi(lk);
            if (k > 0 && ds41_ar_pcn_moe()) {
                fprintf(stderr, "radiance: deepseek41: RADIANCE_PP_LEAD_SEG=%d is not taken beside RADIANCE_DS41_AR_PCN=2 (tp8 dev "
                                "M9)\n", k);
            } else if (k > 0) {
                const ds41layers::Geo geo = lr_geo(m);
                int top = (int)m.c.n_layer - 1;
                for (int tp : geo.taps) top = std::min(top, tp);
                std::vector<int> xs;
                for (int x = k; x < m.pp_h; x += k) xs.push_back(x);
                for (int x = m.pp_t + k; x <= top; x += k) xs.push_back(x);
                for (int x : xs) {
                    if ((int)m.seg_b.size() >= RAD_PP_MAX_SEGS - 1) break;
                    ds41layers::Crossing sc;
                    std::string e2;
                    if (ds41layers::lead_segment_crossing(geo, m.pp_h, x, m.pp_t, &sc, &e2) < 0) {
                        fprintf(stderr, "radiance: deepseek41: RADIANCE_PP_LEAD_SEG=%d: no leader boundary at layer %d (%s)\n", k,
                                x, e2.c_str());
                        continue;
                    }
                    m.seg_b.push_back(x);
                    m.seg_cx.push_back(sc);
                }
                m.seg_lead = !m.seg_b.empty();
            }
        }
        /* tp8 loop P28 round 2a: the band (both nodes read the same value; a band X1's layout word catches a mismatch) */
        if (const char* bk = std::getenv("RADIANCE_DS41_PP_BAND"); bk && *bk && std::atoi(bk) != 0) {
            const int k = std::atoi(bk);
            std::string e2;
            if (rad_pp_query().lanes <= 1) {
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PP_BAND=%d is not taken: a band pass is a BG lane's step "
                                "(--pp-lanes > 1) (tp8 loop P28)\n", k);
            } else if (ds41layers::band_crossing(lr_geo(m), m.pp_h, k, m.pp_t, &m.band, &e2) < 0) {
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PP_BAND=%d is not taken: %s (tp8 loop P28)\n", k, e2.c_str());
                m.band = ds41layers::Band{};
            } else {
                m.band_k = k;
                m.band_decl.assign((size_t)m.c.n_layer, 0);
                if (m.pp_rank == 0)
                    for (int l : m.band.window) m.band_decl[(size_t)l] = 1;
            }
        }
        /* the candidate layer's keep words cross: it is this node's, and its indexer runs unsplit here (every rank then
         * holds every row's keep, so rank 0's crossing carries them all; the split is exact either way) */
        for (const ds41layers::Crossing& cx : m.pp_cx)
            if (cx.keep >= 0 && m.lr.held(cx.keep)) m.pp_unsplit_cand = true;
        /* tp8 S5 + S6: lanes issue the leader's pass as a head and a tail with other lanes' parts between them; what the
         * tail reads of the head must not be an activation (ds41layers::head_tail_reads) */
        if (m.pp_rank == 0 && rad_pp_query().lanes > 1) {
            const std::vector<ds41layers::Dep> ht = ds41layers::head_tail_reads(lr_geo(m), m.pp_h, m.pp_t);
            if (!ht.empty()) {
                fprintf(stderr, "radiance: deepseek41: --pp-cut %d,%d with --pp-lanes %d: the leader's layer %d reads the %s "
                                "of its layer %d, which a lane step's head part computes; other lanes' parts run between the "
                                "head and the tail and rewrite it -- start with --pp-lanes 1, or a cut whose follower holds "
                                "that layer\n", m.pp_h, m.pp_t, rad_pp_query().lanes, ht[0].layer,
                        ds41layers::need_name(ht[0].kind), ht[0].from);
                return RAD_E_UNSUPPORTED;
            }
        }
    }
    if (!m.lr.strict && !m.lr_deps.empty()) {
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LAYERS=%s (proxy, this node alone): %s -- another node computes "
                        "these (tp8 S6: the KV copies, selections and keep words across the nodes). A proxy runs only a node "
                        "whose layers read none of them (W(0,20)'s 0:20 and 20:40); RADIANCE_DS41_LAYERS_MODE=strict declares "
                        "this node's part with copies of the foreign kv sources (declare only)\n",
                m.lr.text.c_str(), ds41layers::describe(m.lr_deps).c_str());
        return RAD_E_UNSUPPORTED;
    }
    (void)ctx;
    return RAD_OK;
}

/* A strict node's copy of a foreign kv source's caches at that source's layer `l`: the group (shared with the node's own
 * sources of that ratio), the entry store that fixes its block size (declared, never issued here: S6's import fills it),
 * the layer's binding. Nothing for any other layer the node does not hold. */
static int pp_copy_layer(RadBuilder* b, Model& m, Layer& L, int l);
static int lr_copy_layer(RadBuilder* b, Model& m, Layer& L, int l) {
    L.l = l;
    if (!m.lr.on || !m.lr.strict) return RAD_OK;
    if (m.pp) return pp_copy_layer(b, m, L, l);
    const auto& c = m.c;
    for (const ds41layers::Copy& cp : ds41layers::copies(m.lr_deps)) {
        if (cp.layer != l) continue;
        const int64_t r = c.ratio[(size_t)l], rows = r > 1 ? m.work : m.g.max_tok, hd = c.head_dim;
        L.r = r;
        if (r <= 0) return RAD_E_INVAL;
        if (cp.index) {
            if (!m.index.count(r) && !(m.index[r] = full_group(b, m, "kv_index_r%lld", index_slots(m), r))) return RAD_E_INVAL;
            L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr);
            if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
        }
        if (cp.latent) {
            if (!m.comp.count(r) && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", latent_slots(m, hd), r)))
                return RAD_E_INVAL;
            L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, m.lnv ? "nvfp4" : nullptr);
            if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
            DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
        }
    }
    return RAD_OK;
}

/* tp8 S4: a pipeline node declares every KV group of the published declare, in its order, so the leader's block tables
 * (its scheduler's, the only one) serve both nodes and the step batch is laid out alike (the rendezvous compares the
 * layout's hash). At a layer this node does not hold: the groups the published declare makes there, in its order (the
 * Engram tokens, the window, the source's tail, index and latent), each paged one with the store that fixes its block size
 * (declared, never issued), bound at that layer and zeroed: no pool is carved for it here. The per-sequence groups (the
 * Engram tokens, a tail) cannot be zeroed and are bound there, a few KiB a sequence. */
static int pp_copy_layer(RadBuilder* b, Model& m, Layer& L, int l) {
    const auto& c = m.c;
    const int64_t T = m.g.max_tok, hd = c.head_dim;
    if (std::find(c.ple.begin(), c.ple.end(), (int64_t)l) != c.ple.end() && !m.tok) {
        RadKVGroupDecl d{};
        d.kind = RAD_KV_CONV;
        d.dtype = RAD_I32;
        d.conv_width = c.ngram;
        d.n_head_kv = 1;
        d.head_dim = 1;
        if (!(m.tok = rad_decl_kv_group(b, m.nm.f("kv_engram_tokens"), &d))) return RAD_E_INVAL;
        DS_TRY(rad_bind_layer_kv(b, l, m.tok));
    }
    if (!m.win) {
        if (!(m.win = win_group(b, m))) return RAD_E_INVAL;
        L.kv_st = window_key_store(b, T, hd, m.w8, m.kvn);   /* fixes the block size; never issued here */
        if ((m.win_bs = rad_kv_block_size(b, m.win)) < 1) return RAD_E_INVAL;
        /* The window group is this node's too: its own layers bind it below; no layer of another node is bound. */
    }
    if (!c.is_src(l)) return RAD_OK;
    const int64_t r = c.ratio[(size_t)l], rows = r > 1 ? m.work : T;
    if (r <= 0) return RAD_E_INVAL;
    L.r = r;
    if (r > 1 && !m.tail.count(r)) {
        if (!(m.tail[r] = tail_group(b, m, r))) return RAD_E_INVAL;
        DS_TRY(rad_bind_layer_kv(b, l, m.tail[r]));
    }
    /* tp8 S6 (ds41layers::binds): this unheld source is bound to a group when a crossing into this node copies it (its
     * entries land here at the owner's pages); a group none of this node's layers reads or writes is bound at its first
     * source and zeroed (S4's rule; W(0,t)'s other ratio). A group that is read here but neither held nor copied at this
     * source is declared here (the published order) and bound where a held or copied source of it is. */
    const auto& pb = m.pp_binds;
    const bool zi = (size_t)r < pb.index_zero_r.size() && pb.index_zero_r[(size_t)r];
    const bool zc = (size_t)r < pb.comp_zero_r.size() && pb.comp_zero_r[(size_t)r];
    const bool new_index = !m.index.count(r), new_comp = !m.comp.count(r);
    if (new_index && !(m.index[r] = full_group(b, m, "kv_index_r%lld", index_slots(m), r))) return RAD_E_INVAL;
    if (new_index) {
        L.k_st = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr);   /* the block size; never issued */
        if (rad_kv_block_size(b, m.index[r]) != r) return RAD_E_INVAL;
    }
    if ((new_index && zi) || pb.index[(size_t)l]) DS_TRY(rad_bind_layer_kv(b, l, m.index[r]));
    if (new_index && zi) {
        DS_TRY(rad_kv_group_zero(b, m.index[r]));
        m.pp_zero_r.push_back(r);
    }
    if (new_comp && !(m.comp[r] = full_group(b, m, "kv_compressed_r%lld", latent_slots(m, hd), r))) return RAD_E_INVAL;
    if (new_comp) {
        L.l_st = entry_store(b, rows, hd, r, m.lat, m.page, m.lnv ? "nvfp4" : nullptr);   /* the block size; never issued */
        if (rad_kv_block_size(b, m.comp[r]) != r) return RAD_E_INVAL;
    }
    if ((new_comp && zc) || pb.comp[(size_t)l]) DS_TRY(rad_bind_layer_kv(b, l, m.comp[r]));
    if (new_comp && zc) {
        DS_TRY(rad_kv_group_zero(b, m.comp[r]));
        m.pp_zero_cr.push_back(r);
    }
    return RAD_OK;
}

/* tp8 S4: the follower's copy of the drafter's KV group (the leader's drafter declares it last; same geometry, same
 * stage layers), zeroed. */
static int pp_dspark_zero(RadBuilder* b, Model& m) {
    if (!m.pp || m.pp_rank != 1 || !m.pp_dsp.on) return RAD_OK;
    const auto& c = m.c;
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_WINDOW;
    kd.dtype = RAD_BF16;
    kd.window = c.window + m.pp_dsp.block + m.g.max_spec;
    kd.n_head_kv = 1;
    kd.head_dim = window_slots(c.head_dim, m.w8d) / 2;
    const rad_kvgroup kv = rad_decl_kv_group(b, m.nm.f("kv_dspark"), &kd);
    if (!kv) return RAD_E_INVAL;
    if (!window_key_store(b, m.g.max_tok, c.head_dim, m.w8d, m.kvn)) return RAD_E_INVAL;   /* the block size; never issued */
    if (rad_kv_block_size(b, kv) < 1) return RAD_E_INVAL;
    for (int s = 0; s < m.pp_dsp.n_stage; ++s) DS_TRY(rad_bind_layer_kv(b, m.pp_dsp.base + s, kv));
    return rad_kv_group_zero(b, kv);
}

/* tp8 S4 (the plan's ds41_cut_liveset): the buffers a crossing carries. X1: the embedding's output (xcol, after its
 * all-reduce), which the follower's prologue expands into the streams as one process does. X2: what the op at the
 * leader's layer t reads that the follower's layers wrote -- the inputs of the boundary op, mhc_post_collapse_rmsnorm
 * (layer t-1's post + layer t's collapse + norm: act.x, h1, post, comb, pre_f), which also covers the unfused form
 * (mhc_post over act.x, h1, post, comb, then the collapse over h0 and pre_f). Every part [rows, width], dense. */
/* tp8 S6: a general cut's crossing adds what the receiving layers read of the sending node (Model::pp_cx): the
 * selections of the crossing's index layer (sel, itopk i32 a row), the candidate layer's keep words (keep, kwords i32 a
 * row), and one kv source's new entries -- the inputs of its entry stores, the compressed latent (lat) and / or the
 * index keys (ikn) over the compressor's work rows, with their pages (page, ratio > 1: the compressor tail's; ratio 1
 * stores at the batch's slot mapping, which both nodes have). The receiver issues the same entry_store into its copy of
 * that source's cache at the same pages, so the copy is the owner's bytes (ds41_layers.h crossings()). `kind` says
 * which rows a part has at issue: the step's (or, under BI / PSK, the boundary's live rows [lo, T)), or the work rows. */
/* tp8 dev D10 round 2: `Ids` = the step's token ids (the batch's, no buffer of the plugin's: b 0), X1's last part under
 * m.pp_ids -- under pp_fe X1's only one. */
enum class CutRows { Step, Live, Work, Ids };
struct CutPart { rad_buf b; const char* name; int64_t row_bytes; CutRows rows; int64_t pitch; /* elements a row */ };
static std::vector<CutPart> ds41_cut_liveset_cx(const Model& m, const ds41layers::Crossing& cx);
static std::vector<CutPart> ds41_cut_liveset(const Model& m, int ring) {
    if (ring != RAD_PP_X1 || !m.pp_ids) return ds41_cut_liveset_cx(m, m.pp_cx[ring]);
    std::vector<CutPart> v;
    if (!m.pp_fe) v = ds41_cut_liveset_cx(m, m.pp_cx[ring]);
    v.push_back({ 0, "ids", 4, CutRows::Ids, 1 });
    return v;
}
/* tp8 dev M4: the rings of a segment boundary k in pp_recv_at / pp_send_after / pp_cross (the core's are 0 and 1) */
static constexpr int kSegRecv = 100, kSegSend = 200;
static std::vector<CutPart> ds41_cut_liveset_cx(const Model& m, const ds41layers::Crossing& cx) {
    const auto& c = m.c;
    const int64_t D = c.n_embd, H = c.hc;
    std::vector<CutPart> v;
    if (!cx.residual) v.push_back({ m.xcol, "xcol", D * 2, CutRows::Step, D });
    else
        v = { { m.act.x, "act.x", D * 2, CutRows::Live, D }, { m.h1, "h1", H * D * 2, CutRows::Live, H * D },
              { m.post, "post", H * 4, CutRows::Live, H }, { m.comb, "comb", H * H * 4, CutRows::Live, H * H },
              { m.pre_f, "pre_f", H * 4, CutRows::Live, H } };
    if (cx.sel >= 0) v.push_back({ m.sel, "sel", c.itopk * 4, CutRows::Live, c.itopk });
    if (cx.keep >= 0) v.push_back({ m.keep, "keep", m.kwords * 4, CutRows::Live, m.kwords });
    if (cx.src >= 0) {
        if (cx.latent) v.push_back({ m.lat, "lat", c.head_dim * 2, CutRows::Work, c.head_dim });
        if (cx.index) v.push_back({ m.ikn, "ikn", c.idim * 2, CutRows::Work, c.idim });
        if (c.ratio[(size_t)cx.src] > 1) v.push_back({ m.page, "page", 4, CutRows::Work, 1 });
    }
    return v;
}
/* tp8 loop P28 round 2b: a band pass's X1 parts -- X1 of W(h + k, t), the follower's source copy X1 of W(h, t) carries
 * (where X1 of W(h + k, t) does not carry that source already), then each band layer's window-key rows */
static std::vector<CutPart> ds41_band_liveset(const Model& m) {
    const auto& c = m.c;
    std::vector<CutPart> v = ds41_cut_liveset_cx(m, m.band.x1);
    if (m.band.copy_src >= 0 && m.band.x1.src < 0) {
        if (m.band.copy_latent) v.push_back({ m.lat, "lat", c.head_dim * 2, CutRows::Work, c.head_dim });
        if (m.band.copy_index) v.push_back({ m.ikn, "ikn", c.idim * 2, CutRows::Work, c.idim });
        if (c.ratio[(size_t)m.band.copy_src] > 1) v.push_back({ m.page, "page", 4, CutRows::Work, 1 });
    }
    for (size_t j = 0; j < m.band_key.size(); ++j)
        v.push_back({ m.band_key[j], "bkey", c.head_dim * 2, CutRows::Live, c.head_dim });
    return v;
}
/* The rows a kv source's entry stores run over in a step of T rows and S sequences (the compressor chain's: ratio > 1
 * the bound on its completed blocks, step()'s `rows`; ratio 1 every row). */
static int64_t pp_work_rows(const Model& m, int64_t r, int64_t T, int64_t S) {
    if (r <= 1) return T;
    return m.ctail_full ? m.work : std::min<int64_t>(m.work, std::max<int64_t>((T + r - 1) / r + S, m.g.max_seqs));
}
/* A crossing's bytes a row for the core's ring (rad_pp_crossing: capacity = this x --max-num-batched-tokens): the work
 * rows' parts at their largest count, spread over the step's rows. */
static int64_t cut_row_bytes(const Model& m, const std::vector<CutPart>& v, int ring) {
    const int64_t T = std::max<int64_t>(m.g.max_tok, 1);
    const ds41layers::Crossing& cx = m.pp_cx[ring];
    const int64_t wr = cx.src >= 0 ? pp_work_rows(m, m.c.ratio[(size_t)cx.src], T, m.g.max_seqs) : T;
    int64_t n = 0, w = 0;
    for (const CutPart& p : v) (p.rows == CutRows::Work ? w : n) += p.row_bytes;
    return n + (w * wr + T - 1) / T;
}
/* The stage op of crossing `ring` on this node, its liveness against the parts (declared where it is issued: the sender
 * after its last layer before the crossing, the receiver before its first layer after it). */
static rad_op pp_stage_op(RadBuilder* b, const Model& m, int ring, bool shard = false) {
    const bool sends = (ring == RAD_PP_X1) == (m.pp_rank == 0);
    const std::vector<CutPart> v = ds41_cut_liveset(m, ring);
    rad_op h = 0;
    if (shard) {
        /* tp8 loop P20 round 3: the residual parts (the Live rows of act.x / h1 / post / comb / pre_ffn) row-sharded */
        int64_t mask = 0;
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i].rows == CutRows::Live && (v[i].b == m.act.x || v[i].b == m.h1 || v[i].b == m.post || v[i].b == m.comb ||
                                               v[i].b == m.pre_f))
                mask |= (int64_t)1 << i;
        if (mask == 0) return 0;
        const char* nm = sends ? "stage_send" : "stage_recv";   /* round 4: the follower's quarter receive */
        const RadParam ps[] = { RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("ring", ring), RAD_INT("parts", (int64_t)v.size()),
                                RAD_INT("shard", mask) };
        h = rad_op_declared(b, nm) ? rad_decl_op(b, nm, ps, 4, nullptr, 0) : 0;
        if (!h || !rad_op_resolved(b, h)) return 0;
    } else {
        h = op(b, sends ? "stage_send" : "stage_recv",
               { RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("ring", ring), RAD_INT("parts", (int64_t)v.size()) }, {}, {}, {});
    }
    if (!h) return h;
    std::vector<rad_buf> bufs;
    for (const CutPart& p : v)
        if (p.b) bufs.push_back(p.b);   /* tp8 dev D10 round 2: the token ids are the batch's, not a buffer */
    if (bufs.empty()) return h;
    if (sends) rad_op_reads(b, h, bufs.data(), (int)bufs.size());
    else       rad_op_writes(b, h, bufs.data(), (int)bufs.size());
    return h;
}

/* tp8 loop P28 round 2b: crossing `ring`'s stage op over an explicit parts list (the band X1) */
static rad_op pp_stage_op_v(RadBuilder* b, const Model& m, int ring, const std::vector<CutPart>& v) {
    const bool sends = (ring == RAD_PP_X1) == (m.pp_rank == 0);
    rad_op h = op(b, sends ? "stage_send" : "stage_recv",
                  { RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("ring", ring), RAD_INT("parts", (int64_t)v.size()) }, {}, {}, {});
    if (!h) return h;
    std::vector<rad_buf> bufs;
    for (const CutPart& p : v)
        if (p.b) bufs.push_back(p.b);
    if (bufs.empty()) return h;
    if (sends) rad_op_reads(b, h, bufs.data(), (int)bufs.size());
    else       rad_op_writes(b, h, bufs.data(), (int)bufs.size());
    return h;
}

/* tp8 dev M4: segment boundary k's save (`sends`, after layer s - 1) or restore (before layer s): the stage op in loop mode
 * over this rank's ring buffer */
static rad_op pp_seg_stage_op(RadBuilder* b, const Model& m, size_t k, bool sends, bool shard = false) {
    const std::vector<CutPart> v = ds41_cut_liveset_cx(m, m.seg_cx[k]);
    rad_op h = 0;
    if (shard) {
        /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG=2): the residual parts (the Live rows of act.x / h1 / post / comb / pre_ffn)
         * row-sharded over the node's ranks, the rest whole -- pp_stage_op's mask, in loop mode */
        int64_t mask = 0;
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i].rows == CutRows::Live && (v[i].b == m.act.x || v[i].b == m.h1 || v[i].b == m.post || v[i].b == m.comb ||
                                               v[i].b == m.pre_f))
                mask |= (int64_t)1 << i;
        if (mask == 0) return 0;
        const char* nm = sends ? "stage_send" : "stage_recv";
        const RadParam ps[] = { RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("ring", 2), RAD_INT("parts", (int64_t)v.size()),
                                RAD_INT("loop", 1), RAD_INT("shard", mask), RAD_INT("ranks", m.g.world) };
        h = rad_op_declared(b, nm) ? rad_decl_op(b, nm, ps, 6, nullptr, 0) : 0;
        if (!h || !rad_op_resolved(b, h)) return 0;
    } else {
        h = op(b, sends ? "stage_send" : "stage_recv",
               { RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("ring", 2), RAD_INT("parts", (int64_t)v.size()), RAD_INT("loop", 1) },
               {}, {}, {});
    }
    if (!h) return h;
    std::vector<rad_buf> bufs;
    for (const CutPart& p : v) bufs.push_back(p.b);
    if (sends) rad_op_reads(b, h, bufs.data(), (int)bufs.size());
    else       rad_op_writes(b, h, bufs.data(), (int)bufs.size());
    return h;
}

/* tp8 S6: the receiver's copy stores -- the entry_store of the copied source (the owner's l_st / k_st: same op, same
 * parameters), declared after the crossing that carries their inputs, issued right after it into this node's copy of
 * that source's cache (bound at the source layer by pp_copy_layer). */
static int pp_copy_declare(RadBuilder* b, Model& m, int ring) {
    const ds41layers::Crossing& cx = m.pp_cx[ring];
    if (cx.src < 0) return RAD_OK;
    const auto& c = m.c;
    const int64_t r = c.ratio[(size_t)cx.src], rows = r > 1 ? m.work : m.g.max_tok;
    if (r <= 0) return RAD_E_INVAL;
    if (cx.latent && !(m.pp_cp_lat = entry_store(b, rows, c.head_dim, r, m.lat, m.page, m.lnv ? "nvfp4" : nullptr)))
        return RAD_E_INVAL;
    if (cx.index && !(m.pp_cp_idx = entry_store(b, rows, c.idim, r, m.ikn, m.page, m.ikmx ? "mxfp4" : nullptr)))
        return RAD_E_INVAL;
    return RAD_OK;
}

/* The node's line (rank 0's real declare, and a note in the graph): its layers, roles, expert units a rank (what the S3
 * pass criterion checks resident: the mover's slab slots against it, no pinned pool slot), its kv sources and copies. */
static void lr_note(RadBuilder* b, const Model& m, const RadBuildCtx* ctx) {
    if (!m.lr.on) return;
    const auto& c = m.c;
    int moe = 0, eng = 0;
    std::string own_src;
    for (const Layer& L : m.layers) {
        if (!m.lr.held(L.l)) continue;
        moe += L.moe.op_router ? 1 : 0;
        if (L.src) own_src += (own_src.empty() ? "" : " ") + std::to_string(L.l);
        eng += L.e_ids ? 1 : 0;
    }
    const int64_t per = m.etp ? c.n_expert : c.n_expert / std::max(m.g.world, 1);
    const int64_t trunk = (int64_t)moe * per, draft = m.dsp.on ? (int64_t)m.dsp.n_stage * m.dsp.ne_rank : 0;
    std::string cps;
    for (const ds41layers::Copy& cp : ds41layers::copies(m.lr_deps))
        cps += (cps.empty() ? "" : ", ") + std::to_string(cp.layer) + (cp.latent && cp.index ? " (latent + index keys)"
                                                                       : cp.latent           ? " (latent)"
                                                                                             : " (index keys)");
    char line[2048];
    std::snprintf(line, sizeof line,
                  "layer ranges RADIANCE_DS41_LAYERS=%s (%s): %d of %lld trunk layers, %s; embedding + final norm "
                  "+ head %s; DSpark drafter %s; Engram layers %d; kv sources held [%s]%s%s; routed expert units a rank "
                  "%lld trunk (%d MoE layers x %lld) + %lld drafter = %lld -- all resident iff the mover's slab slots >= "
                  "%lld and its pinned pool slots are 0 (--host-pool-mib 0)%s%s",
                  m.lr.text.c_str(), m.lr.strict ? "strict: this node's part only, declare only" : "proxy: the stage alone",
                  m.lr.n_own, (long long)c.n_layer, m.lr.leader ? "the leader" : "a follower",
                  m.lr.leader ? "held (the leader's)" : lr_shell(m) ? "declared as the proxy's shell (the leader's)"
                  : m.pp_fe ? "not declared but the embedding (RADIANCE_PP_FOLLOWER_EMBED=1: this node looks it up, tp8 dev 1a)"
                            : "not declared",
                  m.dsp.on ? "declared" : "not declared", eng, own_src.c_str(), cps.empty() ? "" : "; copies of foreign sources ",
                  cps.c_str(), (long long)trunk, moe, (long long)per, (long long)draft, (long long)(trunk + draft),
                  (long long)(trunk + draft), m.lr_deps.empty() ? "" : "; reads from another node: ",
                  m.lr_deps.empty() ? "" : ds41layers::describe(m.lr_deps).c_str());
    rad_note(b, "%s", line);
    if (ctx->rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: deepseek41: %s\n", line);
    /* tp8 loop P28 round 2a: the band's form and what it costs this node (declare only) */
    if (m.band_k > 0) {
        const ds41layers::Band& bd = m.band;
        int bmoe = 0;
        for (int l : bd.window) bmoe += m.layers[(size_t)l].moe.op_router ? 1 : 0;
        char bl[1024];
        std::snprintf(bl, sizeof bl,
                      "RADIANCE_DS41_PP_BAND=%d (tp8 loop P28 round 2a, declare only): a band pass of W(%d,%d) would run the leader's "
                      "head over [0, %d) and the follower from %d; its X1 = W(%d,%d)'s (%s%s%s%s) + the follower's copy of %s + the "
                      "window keys of layers %d..%d; this node %s -- no pass issues a band layer in this build",
                      m.band_k, m.pp_h, m.pp_t, bd.at, bd.at, bd.at, m.pp_t, bd.x1.residual ? "the residual" : "xcol",
                      bd.x1.sel >= 0 ? ", selections" : "", bd.x1.keep >= 0 ? ", keep words" : "",
                      bd.x1.src >= 0 ? ", a source's entries" : "",
                      bd.copy_src >= 0 ? (bd.copy_latent ? "a kv source's latent" : "a kv source's index keys") : "no source",
                      bd.h, bd.at - 1,
                      m.pp_rank == 0 ? fmt_s("declares the band's %zu layers too (%d MoE layers: +%lld routed expert units a rank, their "
                                           "window binding)", bd.window.size(), bmoe,
                                           (long long)bmoe * (m.etp ? c.n_expert : c.n_expert / std::max(m.g.world, 1))).c_str()
                                     : "holds them already (their decode owner)");
        rad_note(b, "%s", bl);
        if (ctx->rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: deepseek41: %s\n", bl);
    }
    if (m.pp) {
        std::string x1, x2;
        for (const CutPart& p : ds41_cut_liveset(m, RAD_PP_X1)) x1 += fmt_s("%s%s %lld B", x1.empty() ? "" : ", ", p.name, (long long)p.row_bytes);
        for (const CutPart& p : ds41_cut_liveset(m, RAD_PP_X2)) x2 += fmt_s("%s%s %lld B", x2.empty() ? "" : ", ", p.name, (long long)p.row_bytes);
        /* tp8 S6: what the copies bind here (sources held or copied) */
        std::string cp;
        const ds41layers::Crossing& in = m.pp_cx[m.pp_rank == 1 ? RAD_PP_X1 : RAD_PP_X2];
        if (in.src >= 0)
            cp = fmt_s("; a copy of source %d's %s (stored here from X%d at the owner's pages)", in.src,
                       in.latent && in.index ? "latent + index keys" : in.latent ? "latent" : "index keys", in.at == m.pp_h ? 1 : 2);
        if (m.pp_unsplit_cand) cp += fmt_s("; layer %lld's indexer unsplit (its keep words cross)", (long long)c.cand);
        const bool clean = m.pp_h == 0 && m.pp_cx[RAD_PP_X2].sel < 0 && m.pp_cx[RAD_PP_X2].keep < 0 &&
                           m.pp_cx[RAD_PP_X2].src < 0;
        if (clean)   /* W(0,t), a clean cut: S4's line, byte for byte (its declare is S4's) */
            std::snprintf(line, sizeof line,
                          "two-node pipeline (tp8 S4): the %s of W(%d,%d); X1 = %s (%lld B a row), X2 = %s (%lld B a row, the "
                          "inputs of layer %d's mhc_post_collapse_rmsnorm); every KV group of the published declare in its "
                          "order, the ones no held layer reads zeroed%s",
                          m.pp_rank == 0 ? "leader" : "follower", m.pp_h, m.pp_t, x1.c_str(),
                          (long long)cut_row_bytes(m, ds41_cut_liveset(m, RAD_PP_X1), RAD_PP_X1), x2.c_str(),
                          (long long)cut_row_bytes(m, ds41_cut_liveset(m, RAD_PP_X2), RAD_PP_X2), m.pp_t,
                          m.pp_rank == 1 && m.pp_dsp.on ? "; the leader's drafter KV group zeroed here" : "");
        else
            std::snprintf(line, sizeof line,
                          "two-node pipeline (tp8 S6): the %s of W(%d,%d); X1 = %s (%lld B a row), X2 = %s (%lld B a row); the "
                          "residual parts are the inputs of the boundary layer's mhc_post_collapse_rmsnorm (layers %d and %d); "
                          "every KV group of the published declare in its order, the ones no held layer reads zeroed%s%s",
                          m.pp_rank == 0 ? "leader" : "follower", m.pp_h, m.pp_t, x1.c_str(),
                          (long long)cut_row_bytes(m, ds41_cut_liveset(m, RAD_PP_X1), RAD_PP_X1), x2.c_str(),
                          (long long)cut_row_bytes(m, ds41_cut_liveset(m, RAD_PP_X2), RAD_PP_X2), m.pp_h, m.pp_t,
                          m.pp_rank == 1 && m.pp_dsp.on ? "; the leader's drafter KV group zeroed here" : "", cp.c_str());
        rad_note(b, "%s", line);
        if (ctx->rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: deepseek41: %s\n", line);
        /* tp8 dev M4: the follower's segment boundaries (RADIANCE_PP_SEG_LAYERS) */
        if (!m.seg_b.empty()) {
            std::string sb;
            for (size_t k = 0; k < m.seg_b.size(); ++k) {
                std::string ps;
                for (const CutPart& p : ds41_cut_liveset_cx(m, m.seg_cx[k])) ps += fmt_s("%s%s", ps.empty() ? "" : "+", p.name);
                sb += fmt_s("%s%d (%s)", sb.empty() ? "" : ", ", (int)m.seg_b[k], ps.c_str());
            }
            if (m.seg_lead)
                std::snprintf(line, sizeof line,
                              "two-node pipeline (tp8 dev M9, RADIANCE_PP_LEAD_SEG): a BG lane's leader head / tail parts may run in "
                              "segments at layers %s, each boundary's live set saved and restored on this rank (a %lld-byte ring "
                              "buffer a card)", sb.c_str(), (long long)m.seg_ring_bytes);
            else
            std::snprintf(line, sizeof line,
                          "two-node pipeline (tp8 dev M4, RADIANCE_PP_SEG_LAYERS): the follower's passes may run in segments at "
                          "layers %s, each boundary's live set saved and restored on this rank (a %lld-byte ring buffer a card)",
                          sb.c_str(), (long long)m.seg_ring_bytes);
            rad_note(b, "%s", line);
            if (ctx->rank == 0 && !ctx->shape_probe) fprintf(stderr, "radiance: deepseek41: %s\n", line);
            rad_pp_declare_segments(m.seg_b.data(), (int)m.seg_b.size());
        }
        /* tp8 S6: the copied sources' layers both nodes hold, for the core's KV-copy check (RADIANCE_PP_KV_CHECK): the
         * group, the layer's place among this node's bound layers of it (ascending: the held sources and the copy), the
         * layer */
        if (ctx->rank == 0 && !ctx->shape_probe)
            for (const ds41layers::Crossing& cx : m.pp_cx) {
                if (cx.src < 0) continue;
                const int64_t r = c.ratio[(size_t)cx.src];
                for (int which = 0; which < 2; ++which) {
                    if (which == 0 ? !cx.latent : !cx.index) continue;
                    const std::vector<char>& bd = which == 0 ? m.pp_binds.comp : m.pp_binds.index;
                    int slot = 0;
                    for (int l = 0; l < cx.src; ++l) slot += bd[(size_t)l] && c.ratio[(size_t)l] == r;
                    const auto it = which == 0 ? m.comp.find(r) : m.index.find(r);
                    if (it != (which == 0 ? m.comp.end() : m.index.end()))
                        rad_pp_declare_kv_copy((int32_t)it->second, slot, cx.src);
                }
            }
    }
}

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
    mhc_read_env();
    i11_read_env();
    rf_read_env();
    p1_read_env();
    pf_read_env();
    dkv_read_env();
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    DS_TRY(read_cfg(m.c, meta));
    DS_TRY(lr_read(m, ctx));   /* tp8 S3: the node's layer ranges (unset: every layer) */
    m.release = is_release(meta);
    m.n_stage = (int)gi(meta, "nextn_predict_layers", 0);
    naming = &m;
    DS_TRY(geometry(m, ctx));
    DS_TRY(dsp_config(b, m, meta, ctx));
    DS_TRY(lr_settle(b, m, ctx));
    const auto& c = m.c;
    auto& g = m.g;
    int64_t rmin = 0;
    for (int l = 0; l < c.n_layer; ++l)
        if (c.ratio[(size_t)l] > 0 && (rmin == 0 || c.ratio[(size_t)l] < rmin)) rmin = c.ratio[(size_t)l];
    m.work = g.max_tok + g.max_seqs;
    m.cmax = g.max_ctx / std::max<int64_t>(rmin, 1) + 1;
    index_runs(m);
    index_mode(m);
    /* tp8 loop P13: RADIANCE_DS41_PIPE2 (pipe2_env) where the ranks reduce on the exact wire and a pass holds two pieces;
     * a sizing declare (below) takes the real declare's value, so its levels declare the same ops */
    if (!ctx->shape_probe && pipe2_env() > 0) {
        const char* why = g.world < 2 ? "one rank, no all-reduce to overlap"
                        : !g.wire_exact ? "the all-reduce wire is not exact"
                        : g.max_tok < pipe2_env() ? "--max-num-batched-tokens is below its rows" : nullptr;
        m.pipe2 = why ? 0 : pipe2_env();
        m.pipe2_attn = m.pipe2 > 0 && pipe2_attn_env();
        m.pipe2_cone = m.pipe2 > 0 && pipe2_cone_env();
        /* P15: every rank attends (no attention-zero rank: the same per-rank ops order everywhere) */
        m.pipe2_tbo = m.pipe2 > 0 && m.pipe2_attn && m.aw == g.world ? pipe2_tbo_env() : 0;
        if (ctx->rank == 0) {
            if (why)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2=%lld off (tp8 loop P13): %s\n",
                        (long long)pipe2_env(), why);
            else
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2=%lld (tp8 loop P13): an uncut layer of a lone "
                                "prefill pass of >= %lld rows runs its MoE in two pieces (cut at a multiple of %lld, each >= "
                                "%lld rows), piece 0's all-reduce on the second lane beside piece 1's MoE; the same ops and "
                                "bytes as the whole pass\n",
                        (long long)pipe2_env(), (long long)m.pipe2, (long long)kPipe2Align, (long long)kPipe2Piece);
            if (!why && m.pipe2_cone)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2_CONE=1 (tp8 loop P13c): lever PSK's cut layers too, where "
                                "a layer's rows are one prefill segment of >= %lld rows (not the split layer)\n",
                        (long long)m.pipe2);
            if (!why && m.pipe2_attn)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2_ATTN=1 (tp8 loop P13b): those layers' attention "
                                "all-reduce in the two pieces too, piece 0's on the second lane beside piece 1's output "
                                "projection\n");
            if (!why && pipe2_tbo_env() > 0) {
                if (m.pipe2_tbo > 0)
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2_TBO=%d (tp8 loop P15 / P16):%s%s%s; the same ops and "
                                    "bytes, the collectives in one order\n",
                            m.pipe2_tbo,
                            (m.pipe2_tbo & 4) ? " the MoE of a pipelined layer WHOLE (one pass), its all-reduce in the two halves"
                                              : "",
                            (m.pipe2_tbo & 1) && !(m.pipe2_tbo & 4) ? " piece 1's attention all-reduce on the second lane beside "
                                                                      "MoE pass 0"
                                                                    : "",
                            (m.pipe2_tbo & 2) ? ", piece 1's MoE all-reduce on the second lane beside the next layer's piece-0 "
                                                "inputs"
                                              : "");
                else
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2_TBO=%d off (tp8 loop P15): %s\n", pipe2_tbo_env(),
                            !m.pipe2_attn ? "it needs RADIANCE_DS41_PIPE2_ATTN=1" : "an attention-zero rank");
            }
        }
    }
    /* tp8 loop P20 round 2: RADIANCE_DS41_SPHC (sphc_env) where four ranks reduce on the exact wire (the reduce-scatter's
     * fold is the exact all-reduce's) and every rank attends; a sizing declare (below) takes the real declare's value */
    if (!ctx->shape_probe && sphc_env() > 0) {
        const char* why = g.world != 4 ? "not four ranks (the rows collectives are four-rank)"
                        : !g.wire_exact ? "the all-reduce wire is not exact"
                        : m.aw != g.world ? "an attention-zero rank"
                        : m.pipe2 > 0 ? "RADIANCE_DS41_PIPE2 is on (its pieces and the lane-1 all-reduces are another form)"
                        : g.max_tok < sphc_env() ? "--max-num-batched-tokens is below its rows"
                        : g.max_tok % 4 ? "--max-num-batched-tokens is not a multiple of four (the streams' gather rides "
                                          "the all-reduce's instance)"
                        : (c.n_embd % 32) ? "the hidden width is not a multiple of 32" : nullptr;
        m.sphc = why ? 0 : sphc_env();
        m.sphc_eng = m.sphc > 0 && sphc_eng_env();
        if (ctx->rank == 0) {
            if (why)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC=%lld off (tp8 loop P20): %s\n", (long long)sphc_env(),
                        why);
            else
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC=%lld (tp8 loop P20): a lone prefill pass of >= %lld "
                                "rows runs its uncut layers' mHC stream ops over each rank's quarter of the rows (reduce-"
                                "scatter / all-gather raw in place of the all-reduce: the same fold and bytes)%s\n",
                        (long long)sphc_env(), (long long)m.sphc,
                        m.sphc_eng ? "; an Engram layer's lookup, copy and kv GEMM over the quarter too"
                                   : "; Engram lookups whole (RADIANCE_DS41_SPHC_ENGRAM=0)");
        }
    }
    if (ctx->shape_probe) {
        m.sized_against = models[ctx->rank].get();   /* lever P: a level declares the dk pair where the real declare did */
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
        m.pipe2 = full.pipe2;   /* tp8 loop P13: the same overlap all-reduces and concurrent buffer as the real declare */
        m.pipe2_attn = full.pipe2_attn;
        m.pipe2_cone = full.pipe2_cone;
        m.pipe2_tbo = full.pipe2_tbo;   /* P15: the same concurrent buffers as the real declare */
        m.sphc = full.sphc;             /* tp8 loop P20: the same rows collectives as the real declare */
        m.sphc_eng = full.sphc_eng;
        m.sp_seg = full.sp_seg;         /* tp8 loop P25: the same segment boundary ops as the real declare */
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
    /* tp8 loop P13: the buffer RADIANCE_DS41_PIPE2's lane-1 all-reduce reduces in place beside lane 0's work (the MoE's
     * output rows, act.x) stays out of the sharing pool (Ctx::lane1_ rule 3) */
    if (m.pipe2 > 0) DS_TRY(rad_buf_concurrent(b, m.act.x));
    if (m.pipe2 > 0 && m.pipe2_attn) DS_TRY(rad_buf_concurrent(b, m.attn_out));   /* P13b's lane-1 attention all-reduce */
    /* P15 (tp8 loop): what a piece-1 op reads or writes after the other piece's ops ran in between, out of the sharing pool --
     * bit 1: piece 1's post / collapse / norm (pcn_f) after MoE pass 0 reads h0 + pre_attn (dead after pcn_f in the declared
     * order, so the MoE's transients could share their bytes); bit 2: the next layer's pcn_mix over piece 1 after that layer's
     * piece-0 attention inputs reads streams_next + pre_ffn and writes collapsed (dead around pcn_mix in the declared order,
     * so q_lora / window_kv / ... could share their bytes) */
    if ((m.pipe2_tbo & 1) && !(m.pipe2_tbo & 4)) {
        DS_TRY(rad_buf_concurrent(b, m.h0));
        DS_TRY(rad_buf_concurrent(b, m.pre_a));
    }
    if (m.pipe2_tbo & 2) {
        DS_TRY(rad_buf_concurrent(b, m.h1));
        DS_TRY(rad_buf_concurrent(b, m.pre_f));
        DS_TRY(rad_buf_concurrent(b, m.xcol));
    }
    /* Pictures: the container's tower with an encoder budget (rad_block_ds41_vision.h; Model::media). Not on a node with
     * layer ranges (tp8 S3: the tower is the leader's, and its rows' Engram masks are another node's layers). */
    m.media = Ds41Vision::present(meta) && ctx->max_enc_patches > 0 && !m.lr.on;
    if (m.lr.on && Ds41Vision::present(meta) && ctx->max_enc_patches > 0)
        rad_note(b, "pictures: off on a node with layer ranges (RADIANCE_DS41_LAYERS=%s)", m.lr.text.c_str());
    if (m.media) {
        const int64_t Tm = g.max_tok, K = c.top_k;
        m.vl_ids = decl_b(b, nm.f("vl_ids"), RAD_I32, { Tm, K });
        m.vl_ew  = decl_b(b, nm.f("vl_ew"), RAD_F32, { Tm, K });
        m.vl_tid = decl_b(b, nm.f("vl_tid"), RAD_U8, { Tm, 4 * K });
        m.vl_tew = decl_b(b, nm.f("vl_tew"), RAD_U8, { Tm, 4 * K });
        if (!m.vl_ids || !m.vl_ew || !m.vl_tid || !m.vl_tew) return RAD_E_INVAL;
        if (!c.ple.empty()) {
            m.mm_eos  = decl_b(b, nm.f("mm_eos"), RAD_I32, { Tm });
            m.mm_zero = decl_b(b, nm.f("mm_zero"), RAD_BF16, { kMmZeroRows, c.n_embd });
            if (!m.mm_eos || !m.mm_zero) return RAD_E_INVAL;
        }
    }
    /* The drafter's source rows, one hc-mean row a target layer, written by the trunk step. */
    if (m.dsp.on && !m.zero &&
        !(m.dsp.src = decl_b(b, nm.f("dspark_src"), RAD_BF16, { g.max_tok, (int64_t)m.dsp.taps.size() * c.n_embd })))
        return RAD_E_INVAL;

    const int64_t T = g.max_tok, D = c.n_embd;
    /* The vocabulary is split over the ranks (shard_extent): the lookup gives zeros outside this
     * rank's rows and an all-reduce completes it; the head's logits stay sharded for the sampler. */
    const int64_t v0 = g.vocab_off, v1 = g.vocab_off + g.n_vocab;
    /* tp8 S3: the embedding, the head and the final norm's gain where lr_shell() (not on a strict follower); tp8 dev 1a:
     * the embedding alone on a follower under pp_fe */
    const bool shell = lr_shell(m);
    if (shell || m.pp_fe) {
        m.token = lw(b, m, nm.f("token_embd.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1,
                     false, RAD_ACCESS_VOCAB);
        if (shell)
            m.head = lw(b, m, nm.f("output.weight"), g.n_vocab, D, grp_model(), RAD_SHARD_ROW, v0, v1, false,
                        RAD_ACCESS_VOCAB);
        if (!m.token || (shell && !m.head)) return RAD_E_INVAL;
        if (shell) m.final_gain = wt(b, nm.f("output_norm.weight"), RAD_F32, { D }, grp_model());
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
        /* Pictures: the encoder's rows over the lookup, before the vocabulary all-reduce completes it (rank 0 holds the
         * rows, every other rank zeros: RadBatch::mm_embd). */
        if (m.media && shell) DS_TRY(m.mrows.declare(b, T, D, m.xcol));
        if (g.world > 1)
            m.embed_ar = op(b, "all_reduce",
                            { RAD_INT("world_size", g.world), RAD_RANGE("numel", D, T * D),
                              RAD_STR("dtype", "bf16"), RAD_INT("exact", 1) },
                            {}, { m.xcol }, { m.xcol });
    }
    /* tp8 S4: X1, the embedding's output: the leader sends it after its all-reduce, the follower's prologue starts from
     * it where one process's starts from the lookup. Both nodes name both crossings' rows (rad_pp_crossing). */
    if (m.pp) {
        /* tp8 S6: at h > 0 X1 is the residual after layer h - 1, declared there (the layer loop below) */
        if (m.pp_h == 0 && !(m.pp_x1 = pp_stage_op(b, m, RAD_PP_X1))) return RAD_E_INVAL;
        /* tp8 loop P28 round 2b: the band's key rows (both nodes) before the ring sizes: a band X1 may be the wider one */
        if (m.band_k > 0) {
            for (size_t j = 0; j < m.band.window.size(); ++j)
                m.band_key.push_back(decl_b(b, nm.f("pp_band_key%zu", j), RAD_BF16, { T, c.head_dim }, RAD_BUF_PERSIST));
            m.band_cp.assign(m.band_key.size(), 0);
            m.band_st.assign(m.band_key.size(), 0);
        }
        for (int ring : { (int)RAD_PP_X1, (int)RAD_PP_X2 }) {
            const std::vector<CutPart> v = ds41_cut_liveset(m, ring);
            int64_t rb = cut_row_bytes(m, v, ring), np = (int64_t)v.size();
            if (ring == RAD_PP_X1 && m.band_k > 0) {
                const std::vector<CutPart> vb = ds41_band_liveset(m);
                rb = std::max(rb, cut_row_bytes(m, vb, ring));
                np = std::max<int64_t>(np, (int64_t)vb.size());
            }
            rad_pp_declare_crossing(ring, rb, (int)np);
        }
        if (m.pp_ids) rad_pp_declare_x1_ids(1);   /* tp8 dev D10 round 2: X1's token ids, taken first by the follower */
        if (m.band_k > 0) {
            if (m.pp_ids) {   /* tp8 loop P28: a band X1 carries no token ids (round 2c's form) */
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PP_BAND=%d is not taken beside RADIANCE_PP_ISSUE_AHEAD=1 (X1's "
                                "token ids; tp8 loop P28)\n", m.band_k);
            } else {
                rad_pp_declare_band(m.band_k);   /* tp8 loop P28: the engine may mark band passes */
            }
        }
    }
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
    /* tp8 loop P20 round 2 (RADIANCE_DS41_SPHC): the rows collectives, raw (exact), in place, each over its buffer -- the
     * attention output's and the MoE delta's reduce-scatter, the normed inputs' all-gathers (row D), the streams' all-gather
     * (row hc x D, `bound` the all-reduce's T x D: the same instance); none where any does not resolve */
    if (m.sphc > 0) {
        const auto rows_op = [&](const char* name, int64_t row, rad_buf bf, bool wide) {
            std::vector<RadParam> ps = { RAD_INT("world_size", g.world), RAD_RANGE("numel", row, T * row),
                                         RAD_STR("dtype", "bf16"), RAD_INT("row", row), RAD_INT("raw", 1) };
            /* `bound` is a plain parameter: a sizing declare (an arena level) names the real declare's, the same instance */
            if (wide) ps.push_back(RAD_INT("bound", (ctx->shape_probe && m.sized_against ? m.sized_against->g.max_tok : T) * D));
            const rad_op h = rad_op_declared(b, name) ? rad_decl_op(b, name, ps.data(), (int)ps.size(), nullptr, 0) : 0;
            if (!h || !rad_op_resolved(b, h)) return (rad_op)0;
            return rw(b, h, { bf }, { bf });
        };
        m.sp_rs_a = rows_op("reduce_scatter_rows", D, m.attn_out, false);
        m.sp_rs_f = rows_op("reduce_scatter_rows", D, m.act.x, false);
        m.sp_ag_x = rows_op("all_gather_rows", D, m.xn, false);
        m.sp_ag_f = rows_op("all_gather_rows", D, m.act.x, false);
        m.sp_ag_h = rows_op("all_gather_rows", c.hc * D, m.h0, true);
        if (!m.sp_rs_a || !m.sp_rs_f || !m.sp_ag_x || !m.sp_ag_f || !m.sp_ag_h) {
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC off (tp8 loop P20): the rows collectives do not "
                                "resolve here (reduce_scatter_rows / all_gather_rows: rs %d %d, ag %d %d %d)\n",
                        m.sp_rs_a != 0, m.sp_rs_f != 0, m.sp_ag_x != 0, m.sp_ag_f != 0, m.sp_ag_h != 0);
            m.sphc = 0;
        }
        /* tp8 loop P26 (RADIANCE_DS41_SPHC_ATTN=1): the gathers of q_a's and kv's outputs (rows of q_lora / head_dim, both
         * multiples of 32), `bound` the all-reduce's T x D so they ride its instance (its slot holds a longer shard) */
        if (m.sphc > 0 && sphc_attn_env()) {
            m.sp_ag_qr = (c.q_lora % 32 || c.head_dim % 32) ? 0 : rows_op("all_gather_rows", c.q_lora, m.qr, true);
            m.sp_ag_kv = m.sp_ag_qr ? rows_op("all_gather_rows", c.head_dim, m.kv, true) : 0;
            if (!m.sp_ag_kv) m.sp_ag_qr = 0;
            /* tp8 loop P26b (=2): the compressor's f32 kvs [T][2 x head_dim], `bound` in f32 elements = half the
             * all-reduce's T x D (libr4d's f32 row rides the bf16 instance of twice its bound) */
            if (m.sp_ag_qr && sphc_attn_env() >= 2 && (2 * c.head_dim) % 16 == 0 && D % 2 == 0) {
                const int64_t row = 2 * c.head_dim;
                std::vector<RadParam> ps = { RAD_INT("world_size", g.world), RAD_RANGE("numel", row, T * row),
                                             RAD_STR("dtype", "f32"), RAD_INT("row", row), RAD_INT("raw", 1),
                                             RAD_INT("bound", (ctx->shape_probe && m.sized_against ? m.sized_against->g.max_tok : T) *
                                                                  D / 2) };
                const rad_op h = rad_op_declared(b, "all_gather_rows")
                                     ? rad_decl_op(b, "all_gather_rows", ps.data(), (int)ps.size(), nullptr, 0) : 0;
                m.sp_ag_kvs = h && rad_op_resolved(b, h) ? rw(b, h, { m.kvs }, { m.kvs }) : 0;
            }
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_ATTN=%d (tp8 loop P26): %s%s\n", sphc_attn_env(),
                        m.sp_ag_qr ? "an SP layer whose xn no compressor / indexer reads runs q_a and kv over each rank's "
                                     "quarter and gathers their outputs (qr, kv) in place of xn"
                                   : "off -- the gathers of qr / kv do not resolve here",
                        sphc_attn_env() < 2 || !m.sp_ag_qr ? ""
                        : m.sp_ag_kvs ? "; an SP kv-source layer runs the compressor's kv | gate GEMM over the quarter and "
                                        "gathers its f32 output (P26b)"
                                      : "; P26b off -- the f32 gather of the compressor's output does not resolve here");
        }
    }
    /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG): the follower's segment passes SP; 2's row-sharded boundary ops are declared with
     * the boundaries below (a library that does not serve them leaves 1) */
    if (sphc_seg_env() && !ctx->shape_probe) {
        const char* why = m.sphc <= 0 ? "RADIANCE_DS41_SPHC is off" : !m.pp || m.pp_rank != 1 ? "not a pipeline follower"
                        : m.seg_b.empty() ? "no segment boundaries (RADIANCE_PP_SEG_LAYERS)" : m.seg_lead ? "leader segments" : nullptr;
        m.sp_seg = why ? 0 : sphc_seg_env();
        /* P25c: a lone pass needs no segment boundary, only the follower's SP passes */
        m.sp_lone = sphc_seg_env() >= 4 && m.sphc > 0 && m.pp && m.pp_rank == 1 && !m.seg_lead;
        if (ctx->rank == 0 && why)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_SEG=%d off (tp8 loop P25): %s%s\n", sphc_seg_env(), why,
                    m.sp_lone ? "; P25c (lone follower passes) stays on" : "");
    }
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
    if (!c.ple.empty() && lr_engram(m)) {   /* (layer ranges: a node that holds an Engram layer) */
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
        /* Pictures: the media rows' compressed token is the pad the n-gram look-back stops at (engram.py's DEAD,
         * read as `eos` by ngram_ids), written over token_map's ids before any ngram_ids reads them. */
        if (m.media) {
            m.mm_iota = op(b, "iota", { RAD_RANGE("M", 1, T), RAD_INT("start", c.pad), RAD_INT("step", 0) },
                           {}, {}, { m.mm_eos });
            m.mm_tok_put = op(b, "scatter_rows", { RAD_RANGE("M", 1, T), RAD_INT("n", 4), RAD_STR("dtype", "u8") },
                              {}, { m.mm_eos, m.ctok }, { m.ctok });
            if (!m.mm_iota || !m.mm_tok_put) return RAD_E_INVAL;
        }
    }
    m.layers.resize((size_t)c.n_layer);
    if (ctx->calib_dir && ctx->calib_dir[0]) {
        m.calib_lo = ctx->calib_layer_lo;
        m.calib_hi = ctx->calib_layer_hi > ctx->calib_layer_lo ? ctx->calib_layer_hi : (int)c.n_layer;
    }
    /* tp8 S3: only the layers this node holds (every layer when RADIANCE_DS41_LAYERS is unset); strict, a copy of a
     * foreign kv source's caches at its layer (lr_copy_layer) */
    /* tp8 dev M4: this rank's segment ring buffer: the header, every boundary's widest live set at max_tok rows, the trailer */
    if (!m.seg_b.empty()) {
        int64_t widest = 0;
        for (const ds41layers::Crossing& cx : m.seg_cx) {
            int64_t n = 0;
            /* loop FINDINGS F12: X2's source entries over the work rows (at most m.work) */
            for (const CutPart& p : ds41_cut_liveset_cx(m, cx))
                n += rad_pp_align((uint64_t)((p.rows == CutRows::Work ? std::max<int64_t>(T, m.work) : T) * p.row_bytes));
            widest = std::max(widest, n);
        }
        m.seg_ring_bytes = RAD_PP_RING_HDR + widest + RAD_PP_TRAILER;
        m.seg_ring = decl_b(b, m.nm.f("pp_segring"), RAD_U8, { m.seg_ring_bytes }, RAD_BUF_PERSIST);
        m.seg_send.assign(m.seg_b.size(), 0);
        m.seg_recv.assign(m.seg_b.size(), 0);
        m.seg_post.assign(m.seg_b.size(), 0);
        m.seg_send_sh.assign(m.seg_b.size(), 0);   /* tp8 loop P25: 2's row-sharded twins */
        m.seg_recv_sh.assign(m.seg_b.size(), 0);
    }
    for (int l = 0; l < c.n_layer; ++l) {
        /* tp8 dev M4: a segment boundary at l: the restore and its mhc_post before layer l (as a receiving node's) */
        for (size_t k = 0; k < m.seg_b.size(); ++k) {
            if (m.seg_b[k] != l) continue;
            if (!(m.seg_recv[k] = pp_seg_stage_op(b, m, k, false))) return RAD_E_INVAL;
            if (m.sp_seg >= 2) m.seg_recv_sh[k] = pp_seg_stage_op(b, m, k, false, true);   /* tp8 loop P25 */
            m.seg_post[k] = op(b, "mhc_post", { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                               {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
            if (!m.seg_post[k]) return RAD_E_INVAL;
        }
        /* tp8 S4 / S6: a node takes its crossing in before its first layer after the cut -- the leader X2 before layer t,
         * the follower X1 before layer h (h > 0) -- then stores the copied source's entries into its copy and, where the
         * boundary layer has no fused pcn_a (an Engram layer, or no library serves it), issues the previous layer's
         * mhc_post itself (pp_post). The sender sends after its last layer before the cut: the leader X1 after layer h - 1
         * (h > 0), the follower X2 after layer t - 1. */
        const bool recv_here = m.pp && ((m.pp_rank == 0 && l == m.pp_t) || (m.pp_rank == 1 && m.pp_h > 0 && l == m.pp_h));
        if (recv_here) {
            const int ring = m.pp_rank == 0 ? RAD_PP_X2 : RAD_PP_X1;
            rad_op& xo = ring == RAD_PP_X2 ? m.pp_x2 : m.pp_x1;
            if (!(xo = pp_stage_op(b, m, ring))) return RAD_E_INVAL;
            /* tp8 loop P20 round 4: the follower's quarter receive (RADIANCE_DS41_SPHC_X1), for an SP pass's X1 */
            if (ring == RAD_PP_X1 && m.sphc > 0 && sphc_x1_env()) {
                m.pp_x1_sh = pp_stage_op(b, m, RAD_PP_X1, true);
                if (ctx->rank == 0 && !ctx->shape_probe)
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_X1=1 (tp8 loop P20 round 4): %s\n",
                            m.pp_x1_sh ? "an SP pass's X1 lands row-sharded (each rank copies its quarter rows of act.x / h1 / "
                                         "post / comb / pre_ffn, the rest whole)"
                                       : "off -- the row-sharded stage_recv does not resolve here");
            }
            DS_TRY(pp_copy_declare(b, m, ring));
            m.pp_post = op(b, "mhc_post", { RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_INT("hc", c.hc), RAD_STR("dtype", "bf16") },
                           {}, { m.act.x, m.h1, m.post, m.comb }, { m.h0 });
            if (!m.pp_post) return RAD_E_INVAL;
        }
        /* tp8 loop P28 round 2b: the follower takes a band pass's X1 before layer h + k, then stores the band layers' window
         * keys (the source copy rides pp_cp_lat / pp_cp_idx as X1's does) */
        if (m.band_k > 0 && m.pp_rank == 1 && l == m.band.at) {
            if (!(m.pp_x1b = pp_stage_op_v(b, m, RAD_PP_X1, ds41_band_liveset(m)))) return RAD_E_INVAL;
            for (size_t j = 0; j < m.band_key.size(); ++j)
                if (!(m.band_st[j] = window_key_store(b, T, c.head_dim, m.w8, m.band_key[j]))) return RAD_E_INVAL;
        }
        if (m.lr.held(l)) DS_TRY(layer(b, m, m.layers[(size_t)l], l));
        else if (m.band_k > 0 && m.band_decl[(size_t)l]) DS_TRY(layer(b, m, m.layers[(size_t)l], l));   /* tp8 loop P28: the band */
        else DS_TRY(lr_copy_layer(b, m, m.layers[(size_t)l], l));
        /* tp8 loop P28 round 2b: the leader's copy of band layer l's key rows, and the band X1 after its last band layer */
        if (m.band_k > 0 && m.pp_rank == 0 && m.band_decl[(size_t)l]) {
            const size_t j = (size_t)(l - m.band.h);
            m.band_cp[j] = op(b, "cast", { RAD_RANGE("M", 1, T), RAD_INT("n", c.head_dim), RAD_STR("from", "bf16"), RAD_STR("to", "bf16") },
                              {}, { m.kvn }, { m.band_key[j] });
            if (!m.band_cp[j]) return RAD_E_INVAL;
            if (l == m.band.at - 1 && !(m.pp_x1b = pp_stage_op_v(b, m, RAD_PP_X1, ds41_band_liveset(m)))) return RAD_E_INVAL;
        }
        if (m.pp && m.pp_rank == 1 && l == m.pp_t - 1 && !(m.pp_x2 = pp_stage_op(b, m, RAD_PP_X2))) return RAD_E_INVAL;
        for (size_t k = 0; k < m.seg_b.size(); ++k)   /* tp8 dev M4: the save after layer s - 1 */
            if (m.seg_b[k] == l + 1) {
                if (!(m.seg_send[k] = pp_seg_stage_op(b, m, k, true))) return RAD_E_INVAL;
                if (m.sp_seg >= 2) m.seg_send_sh[k] = pp_seg_stage_op(b, m, k, true, true);   /* tp8 loop P25 */
            }
        if (m.pp && m.pp_rank == 0 && m.pp_h > 0 && l == m.pp_h - 1 && !(m.pp_x1 = pp_stage_op(b, m, RAD_PP_X1)))
            return RAD_E_INVAL;
        /* tp8 loop P20 round 3: its row-sharded twin (RADIANCE_DS41_SPHC_X1), for an SP pass's X1 */
        if (m.pp && m.pp_rank == 0 && m.pp_h > 0 && l == m.pp_h - 1 && m.sphc > 0 && sphc_x1_env()) {
            m.pp_x1_sh = pp_stage_op(b, m, RAD_PP_X1, true);
            if (ctx->rank == 0 && !ctx->shape_probe)
                fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_X1=1 (tp8 loop P20 round 3): %s\n",
                        m.pp_x1_sh ? "an SP pass's X1 leaves row-sharded (each rank its quarter rows of act.x / h1 / post / "
                                     "comb / pre_ffn, rank 0 the rest); the leader's layer before X1 runs SP, no streams' gather"
                                   : "off -- the row-sharded stage_send does not resolve here");
        }
    }
    /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG): 2 needs every boundary's row-sharded save and restore, else 1 */
    if (m.sp_seg >= 2)
        for (size_t k = 0; k < m.seg_b.size(); ++k)
            if (!m.seg_send_sh[k] || !m.seg_recv_sh[k]) {
                m.sp_seg = 1;
                break;
            }
    if (sphc_seg_env() && m.sp_seg > 0 && ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_SEG=%d (tp8 loop P25): a follower segment pass of one prefill "
                        "segment of >= %lld rows is SP where the whole follower part's plan is (the next segment issues the "
                        "boundary op over each rank's quarter); %s\n",
                sphc_seg_env(), (long long)m.sphc,
                m.sp_seg >= 2 ? (m.sp_seg >= 3 ? "its boundaries saved / restored row-sharded (each rank its quarter rows of act.x / "
                                                 "h1 / post / comb / pre_ffn, the rest whole); P25b: a layer past PSK's first whose "
                                                 "cone keeps every row of the pass is planned uncut"
                                               : "its boundaries saved / restored row-sharded (each rank its quarter rows of act.x / h1 / "
                                                 "post / comb / pre_ffn, the rest whole)")
                : sphc_seg_env() >= 2 ? "boundaries WHOLE: the row-sharded loop stage ops do not resolve here (as 1)"
                                      : "boundaries saved / restored whole");
    if (m.sp_lone && ctx->rank == 0 && !ctx->shape_probe)
        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_SEG=4 (tp8 loop P25c): a LONE follower prefill pass of one "
                        "prefill segment of >= %lld rows (a cold prompt's chunk) plans a layer past PSK's first SP where its cone "
                        "keeps every row of the pass (P25b's rule; ds41_sphc_seg 3 or lower = off)\n",
                (long long)m.sphc);
    /* tp8 loop P25s: what the SP route's switches may take away -- this declare's threshold and levels (every rank the same) */
    if (!ctx->shape_probe && m.sphc > 0) {
        sw_atomic_max(g_sw_sphc_decl, (int)m.sphc);
        sw_atomic_max(g_sw_spattn_decl, sphc_attn_env());   /* the env's levels: the same on both nodes */
        sw_atomic_max(g_sw_spseg_decl, sphc_seg_env());
        if (ctx->rank == 0)
            fprintf(stderr, "radiance: deepseek41: the SP route's switches (tp8 loop P25s): ds41_sphc %lld (a pass SP from that "
                            "many rows; up to 16777216 = none), ds41_sphc_attn 0..%d, ds41_sphc_seg 0..%d -- each at this "
                            "start's env, and lower only takes away what the env declared (this node issues P26 at %d, "
                            "segment SP at %d at most)\n",
                    (long long)m.sphc, sphc_attn_env(), sphc_seg_env(), m.sp_ag_qr ? (m.sp_ag_kvs ? 2 : 1) : 0, m.sp_seg);
    }
    /* tp8 loop P30t: the twins this declare holds, and what ds41_twin may take (the env's classes this node twinned) */
    if (twin_env() && !ctx->shape_probe) {
        int nw = 0;
        std::map<int, int> per;
        for (const auto& [h, th] : m.twop) {
            (void)h;
            ++per[m.twcls.at(th)];
            ++nw;
        }
        for (const Layer& L : m.layers)
            if (L.moe.sh_tw) {   /* the shared arm's two */
                per[kTwShx] += 2;
                nw += 2;
            }
        int cur = g_twin_decl.load(std::memory_order_relaxed);
        while (!g_twin_decl.compare_exchange_weak(cur, cur | m.twin_mask)) {}
        if (ctx->rank == 0) {
            std::string pl;
            for (const auto& [cl, n] : per) pl += fmt_s("%s%d x class %d", pl.empty() ? "" : ", ", n, cl);
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_TWIN=%d (tp8 loop P30t): %d twin GEMM(s) beside their GGUF "
                            "linears (%s; %zu twin weights), prompt segments of >= %lld rows take them (RADIANCE_DS41_TWIN_ROWS), "
                            "decode rows the GGUF launch; switch ds41_twin = mask + 256 x rows, a subset of %d (0 = all GGUF)%s\n",
                    twin_env(), nw, pl.empty() ? "none: no '<name>#pf' entry of the env's classes" : pl.c_str(), m.twin.size(),
                    (long long)twin_rows_env(), m.twin_mask,
                    (twin_env() & kTwEng) ? "; the Engram twin only where lever E1's dedup does not act (ds41_engdd 0)" : "");
        }
    }
    /* tp8 S3: the final norm, the gather and the head where lr_shell() (after the node's last layer: the leader's, or the
     * proxy's shell); none on a strict follower, whose last layer's output is the boundary */
    if (lr_shell(m)) {
        m.pcn_fin = post_collapse_norm(b, m, T, m.final_gain, m.act.x, m.h1, m.h0, m.pre_f, m.xcol,
                                       m.final_hidden);
        if (m.pcn_fin && c.n_layer > 0)
            DS_TRY(ar_pcn_moe_declare(b, m, m.layers[(size_t)(m.lr.on ? m.lr.last : c.n_layer - 1)], T, m.final_gain,
                                      m.final_hidden, &m.ar_pcn_fin));
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
    }
    /* The drafter after everything the trunk step issues (llama_fp8's rule: declaration order is
     * liveness, and its passes run behind the step). */
    if (m.dsp.on) DS_TRY(dsp_declare(b, m, ctx));
    DS_TRY(pp_dspark_zero(b, m));   /* tp8 S4: the follower's zeroed copy of the leader's drafter KV group */
    /* The vision tower last, its weight groups past the drafter's stages; an encoder pass issues it alone. */
    if (m.media) {
        DS_TRY(m.vit.declare(b, meta, ctx, nm, D, (int)c.n_layer + std::max(m.n_stage, 0)));
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: pictures: the vision tower (%lld blocks, %lld patches a pass), the "
                            "image rows' routing bias at %lld MoE layers, Engram masked at the image rows\n",
                    (long long)m.vit.depth, (long long)m.vit.P, (long long)c.n_layer);
    }
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
    /* Lever DKV: what the declare took (rank 0's real declare) */
    if (ctx->rank == 0 && !ctx->shape_probe && (g_dkv_lu.load(std::memory_order_relaxed) > 0 ||
                                                g_dkv_g.load(std::memory_order_relaxed) > 0)) {
        int nu = 0, nl = 0, nlr = 0, ng = 0, nga = 0, ngr = 0, nc = 0;
        for (const Layer& L : m.layers) {
            nu += L.lu_unpack != 0;
            nl += L.attn_lu != 0;
            nlr += L.attn_rtlu != 0;
            ng += L.lgath != 0;
            nga += L.attn_g != 0;
            ngr += L.attn_rtg != 0;
            nc += L.r > 0;
        }
        if (!m.lnv)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_LATUNPACK / RADIANCE_DS41_LATGATHER need the packed latent "
                            "(RADIANCE_DS41_LATENT=nvfp4): lever DKV is not declared\n");
        else
            fprintf(stderr, "radiance: deepseek41: lever DKV: of %d compressing layers, latent_unpack %d, "
                            "attn_sparse_sink_unpacked %d, its fold d form %d (%lld entries a row: %.2f MiB at %lld rows); "
                            "latent_gather %d, attn_sparse_sink_gathered %d, its fold d form %d (%lld rows: %.2f MiB)\n",
                    nc, nu, nl, nlr, (long long)m.lu_e,
                    (double)(T * m.lu_e * c.head_dim * 2) / (1024.0 * 1024.0), (long long)T, ng, nga, ngr,
                    (long long)m.selkv_rows, (double)(m.selkv_rows * c.itopk * c.head_dim * 2) / (1024.0 * 1024.0));
        if (m.lnv && m.selkv && m.selkv_cap > m.selkv_rows && !m.selkv_one)
            fprintf(stderr, "radiance: deepseek41: tp8 loop D25 (RADIANCE_DS41_DKV_GPIECE=1): the gathered latent holds %lld rows "
                            "(%.2f MiB): a decode segment's band pieces (F58, above %lld rows) gather at their own rows, each held "
                            "for the layers of its kv source; every launch stays at <= %lld rows\n",
                    (long long)m.selkv_cap, (double)(m.selkv_cap * c.itopk * c.head_dim * 2) / (1024.0 * 1024.0),
                    (long long)kBiBand, (long long)m.selkv_rows);
        if (m.lnv && m.selkv && m.selkv_one)
            fprintf(stderr, "radiance: deepseek41: tp8 loop D25 step 3 (RADIANCE_DS41_DKV_GPIECE=2): the gathered latent holds %lld rows "
                            "(%.2f MiB) and latent_gather / attn_sparse_sink_gathered (+ its fold d form) are declared to %lld rows "
                            "with libr4d's band %lld: a decode segment above %lld rows gathers and attends ONCE at a compressed layer "
                            "(the scalar decode kernels, each row its own bytes); window-only layers keep F58's pieces\n",
                    (long long)m.selkv_cap, (double)(m.selkv_cap * c.itopk * c.head_dim * 2) / (1024.0 * 1024.0),
                    (long long)m.selkv_cap, (long long)m.selkv_cap, (long long)kBiBand);
    }
    /* Lever SPF-IR, only where RADIANCE_DS41_IROWS_MIN moved the floor (unset: no line, as before). */
    if (ctx->rank == 0 && !ctx->shape_probe && ds41_irows_min() != kScoreRunMin)
        fprintf(stderr, "radiance: deepseek41: lever SPF-IR (RADIANCE_DS41_IROWS_MIN=%lld): the indexer scores in runs of "
                        "at most %lld rows (the %lld-row floor would give %lld), %lld runs declared: score plane %.2f MiB "
                        "a card instead of %.2f\n",
                (long long)ds41_irows_min(), (long long)m.irows, (long long)kScoreRunMin,
                (long long)std::min(g.max_tok, std::max(kScoreRunMin, kScorePlane / (4 * m.cmax))), (long long)m.iparts,
                (double)(m.irows * m.cmax * 4) / (1024.0 * 1024.0),
                (double)(std::min(g.max_tok, std::max(kScoreRunMin, kScorePlane / (4 * m.cmax))) * m.cmax * 4) /
                    (1024.0 * 1024.0));
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
    mhc_declare(m, ctx);
    i11_declare(m, ctx);
    rf_declare(m, ctx);
    p1_declare(m, ctx);
    pf_declare(b, m, ctx);
    DS_TRY(bt_declare(b, m, ctx));
    DS_TRY(kvr_declare(b, m, ctx));
    lr_note(b, m, ctx);   /* tp8 S3: the node's line (nothing when RADIANCE_DS41_LAYERS is unset) */
    if (m.lr.on && !c.ple.empty() && !lr_engram(m) && !ctx->shape_probe) g_ti_nocap.store(1, std::memory_order_release);
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
    if (bi_dband_env()) {   /* tp8 loop F58: a decode band piece's cu a row */
        m.bi_dcu = decl_b(b, m.nm.f("bi_dcu"), RAD_I32, { std::max<int64_t>(1, m.g.max_seqs), kBiBand + 1 }, RAD_BUF_PERSIST);
        if (!m.bi_dcu) return RAD_E_INVAL;
        m.bi_dsub = op(b, "sub_i32", { RAD_RANGE("M", 1, kBiBand + 1) }, {}, {}, { m.bi_dcu });
        if (!m.bi_dsub) return RAD_E_UNSUPPORTED;
        if (ctx->rank == 0 && !ctx->shape_probe)
            fprintf(stderr, "radiance: deepseek41: tp8 loop F58 (RADIANCE_DS41_BI_DBAND=1): a decode segment above %lld rows "
                            "runs its attention in pieces of whole sequences of <= %lld rows and its indexer in runs of <= "
                            "%lld rows (the decode kernels' bytes)\n",
                    (long long)kBiBand, (long long)kBiBand, (long long)kBiBand);
    }
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
        /* tp8 S6: a pipeline node takes the model's split layer whichever node holds it, so both nodes accept
         * ds41_psk_l20 alike and cut every layer as one process does (the leader of W(10,t) does not hold 20) */
        if (m.pp && m.psk_l20 < 0) {
            bool tap = false;
            for (int64_t t : m.dsp.taps.empty() ? m.pp_dsp.taps : m.dsp.taps) tap = tap || t == last;
            const bool eng = std::find(c.ple.begin(), c.ple.end(), (int64_t)last) != c.ple.end();
            if (c.is_src(last) && !eng && !tap && c.cand <= last) m.psk_l20 = last;
        }
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

/* Lever M1 at the end of a declare (RADIANCE_DS41_MHC_FUSE set): the layers that declared the fused op beside pcn_a; none
 * (no library serves it: a host deployment) -> not available (>= 1 refused, the switch reads 0). */
static void mhc_declare(Model& m, const RadBuildCtx* ctx) {
    if (!mhc_declared() || ctx->shape_probe) return;
    int n = 0, want = 0;
    for (const Layer& L : m.layers) {
        want += (L.pcn_a ? 1 : 0) + (L.pcn_f ? 1 : 0);
        n += (L.pcn_mix ? 1 : 0) + (L.pcn_mix_f ? 1 : 0);
    }
    if (n == 0) {
        g_sw_mhc.store(0, std::memory_order_relaxed);
        if (ctx->rank == 0 && !g_mhc_noted.exchange(1, std::memory_order_relaxed))
            fprintf(stderr, "radiance: deepseek41: lever M1 not declared: no layer's mhc_post_collapse_rmsnorm_mix resolves "
                            "(%d pcn_a / pcn_f pairs) -- the switch stays 0\n", want);
        return;
    }
    g_sw_mhc_ok.store(1, std::memory_order_relaxed);
    if (ctx->rank == 0 && !g_mhc_noted.exchange(1, std::memory_order_relaxed))
        fprintf(stderr, "radiance: deepseek41: lever M1 (mhc_fuse) declared (RADIANCE_DS41_MHC_FUSE, fn4 mhc_fuse, now %d): "
                        "mhc_post_collapse_rmsnorm_mix beside %d of %d mhc_post_collapse_rmsnorm (pcn_a / pcn_f: the "
                        "post, the next connection's collapse + norm and its mhc_mix in one op); 1 = launches of >= %lld "
                        "rows fused, N >= 2 = of >= N rows\n",
                g_sw_mhc.load(std::memory_order_relaxed), n, want, (long long)kMhcFuseRows);
}

/* Lever P at the end of a declare (RADIANCE_DS41_MOE1 set): the trunk MoE layers that declared the dk pair. Taken only
 * where EVERY trunk MoE layer has it (a step is one pass at every layer or at none: one rule for the whole step), else
 * the switch stays 0 and 1 is refused. */
static void p1_declare(Model& m, const RadBuildCtx* ctx) {
    if (ctx->shape_probe || !p1_declared()) return;
    int moe = 0, dk = 0;
    int64_t rows = 0;
    for (const Layer& L : m.layers) {
        if (!L.moe.op_router) continue;
        ++moe;
        if (L.moe.dk_takes(kBiMoe + 1)) { ++dk; rows = L.moe.dk_rows; }
    }
    const int ok = moe > 0 && dk == moe ? 1 : 0;
    g_sw_p1_ok.store(ok, std::memory_order_relaxed);
    if (!ok) g_sw_p1.store(0, std::memory_order_relaxed);
    if (ctx->rank != 0 || g_p1_noted.exchange(1, std::memory_order_relaxed)) return;
    fprintf(stderr, "radiance: deepseek41: lever P (ds41_moe1, RADIANCE_DS41_MOE1, now %d) %s: moe_exl3_gate_up_dk / "
                    "moe_exl3_down_dk beside the EXL3 pair of %d of %d trunk MoE layers, 17 .. %lld rows -- a step that "
                    "is one decode segment of that height runs ONE MoE pass (the decode band's bytes per row) instead of "
                    "<= %lld-row pieces\n",
            g_sw_p1.load(std::memory_order_relaxed),
            ok ? "declared" : "not declared (no library serves the ops at every layer)", dk, moe, (long long)rows,
            (long long)kBiMoe);
}

/* tp8 loop D24 at the end of a declare (RADIANCE_DS41_PREFETCH set): a `prefetch` of each read weight at every layer this
 * node holds, where a library serves the op and the deployment did not turn the reads off (--weight-prefetch off); none
 * declared (the host backend: libref serves no `prefetch`) -> the switch reads 0 and >= 1 is refused. */
static void pf_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    if (ctx->shape_probe || !pf_declared()) return;
    const int64_t cap = g_pf_mb.load(std::memory_order_relaxed) << 20;
    int n[3] = { 0, 0, 0 }, layers = 0;
    for (Layer& L : m.layers) {
        if (!m.lr.held(L.l) || ctx->weight_prefetch_off) continue;
        const auto d = [&](rad_weight w, int grp) -> rad_op {
            const rad_op h = w ? decl_prefetch(b, w, cap) : 0;
            if (h) ++n[grp];
            return h;
        };
        L.pf_f = d(L.hc_f_fn, 0);
        L.pf_a = d(L.hc_a_fn, 0);
        if (L.moe.op_router) {
            L.pf_r = d(L.moe.c.sigmoid_router ? L.moe.glm_router.projection : L.moe.w_router, 1);
            L.pf_s = d(L.moe.sh_gate_up.w, 1);
        }
        if (L.src) L.pf_c = d(L.comp_w, 2);
        L.pf_q = d(L.wq_a, 2);
        L.pf_kv = d(L.wkv, 2);
        if (L.pf_f || L.pf_a || L.pf_r || L.pf_s || L.pf_c || L.pf_q || L.pf_kv) ++layers;
    }
    const int ok = n[0] + n[1] + n[2] > 0 ? 1 : 0;
    g_sw_pf_ok.store(ok, std::memory_order_relaxed);
    if (!ok) g_sw_pf.store(0, std::memory_order_relaxed);
    if (ctx->rank != 0 || g_pf_noted.exchange(1, std::memory_order_relaxed)) return;
    fprintf(stderr, "radiance: deepseek41: tp8 loop D24 (ds41_prefetch, RADIANCE_DS41_PREFETCH, now %d) %s: at %d layers, "
                    "the first %lld MB of each weight read into the on-die cache beside a decode-only pass's all-reduces "
                    "(<= %lld rows): bit 0 the mix fns (%d reads), bit 1 the router + shared gate_up (%d), bit 2 the next "
                    "layer's comp_w / wq_a / wkv (%d)\n",
            g_sw_pf.load(std::memory_order_relaxed),
            ok ? "declared" : (ctx->weight_prefetch_off ? "not declared (--weight-prefetch off)"
                                                         : "not declared (no library serves `prefetch`)"),
            layers, (long long)(cap >> 20), (long long)kPfRows, n[0], n[1], n[2]);
}

/* Lever FUSE-R at the end of a declare (RADIANCE_DS41_RFUSE set): the trunk MoE layers that declared each form; a form no
 * layer declared (no library serves the op) is refused, and the switch falls back to the highest form declared. */
static void rf_declare(Model& m, const RadBuildCtx* ctx) {
    if (ctx->shape_probe || !rf_declared()) return;
    int moe = 0, rs = 0, rgs = 0;
    for (const Layer& L : m.layers) {
        if (!L.moe.op_router) continue;
        ++moe;
        rs += L.moe.op_rs_fused ? 1 : 0;
        rgs += L.moe.op_rgs_fused ? 1 : 0;
    }
    const int ok = (rs > 0 ? 1 : 0) | (rgs > 0 ? 2 : 0);
    g_sw_rf_ok.store(ok, std::memory_order_relaxed);
    const int v = g_sw_rf.load(std::memory_order_relaxed);
    if ((v == 2 && !(ok & 2)) || (v >= 1 && !ok)) g_sw_rf.store(ok & 2 ? 2 : ok & 1 ? 1 : 0, std::memory_order_relaxed);
    if (ctx->rank != 0) return;
    g_rf_layers.store(rgs > rs ? rgs : rs, std::memory_order_relaxed);
    if (g_rf_noted.exchange(1, std::memory_order_relaxed)) return;
    fprintf(stderr, "radiance: deepseek41: lever FUSE-R (ds41_rfuse, RADIANCE_DS41_RFUSE, now %d) %s: beside the router of %d "
                    "MoE layers -- router_scored_topk_scatter (form 1, the selection + the sort) at %d, "
                    "router_gemm_f32_scored_topk_scatter (form 2, the projection too, <= 48 rows) at %d\n",
            g_sw_rf.load(std::memory_order_relaxed), ok ? "declared" : "not declared (no library serves the forms)", moe,
            rs, rgs);
}

/* Levers 1U-GQ / 1U-E2 at the end of a declare (their environment set): the layers that declared them; none (no library
 * serves the pair op / the gather, or the shapes do not split) -> not available (>= 1 refused, the switch reads 0). */
static void i11_declare(Model& m, const RadBuildCtx* ctx) {
    if (ctx->shape_probe || (!gq_declared() && !e2_declared())) return;
    int gq = 0, qkv = 0, oa = 0, e2 = 0, eng = 0, dqkv = 0, doa = 0;
    for (const Layer& L : m.layers) {
        qkv += L.qkv_pair ? 1 : 0;
        oa += L.oa_pair ? 1 : 0;
        gq += L.qkv_pair || L.oa_pair ? 1 : 0;
        e2 += L.e_gemm_p && L.e_gath ? 1 : 0;
        eng += L.e_ids ? 1 : 0;
    }
    for (const Layer& L : m.dsp.st) {   /* the drafter's query-pass stages (lever 1U-GQ there too) */
        dqkv += L.qkv_pair ? 1 : 0;
        doa += L.oa_pair ? 1 : 0;
        gq += L.qkv_pair || L.oa_pair ? 1 : 0;
    }
    if (gq_declared()) {
        g_sw_gq_ok.store(gq > 0 ? 1 : 0, std::memory_order_relaxed);
        if (gq == 0) g_sw_gq.store(0, std::memory_order_relaxed);
    }
    if (e2_declared()) {
        g_sw_e2_ok.store(e2 > 0 ? 1 : 0, std::memory_order_relaxed);
        if (e2 == 0) g_sw_e2.store(0, std::memory_order_relaxed);
    }
    if (ctx->rank != 0) return;
    g_gq_layers.store(gq_declared() ? gq : 0, std::memory_order_relaxed);
    g_e2_layers.store(e2_declared() ? e2 : 0, std::memory_order_relaxed);
    if (g_i11_noted.exchange(1, std::memory_order_relaxed)) return;
    if (gq_declared())
        fprintf(stderr, "radiance: deepseek41: lever 1U-GQ (ds41_gq, RADIANCE_DS41_GQ, now %d) %s: gemm_nt_ggml_pair beside "
                        "q_a + kv at %d layers and beside attention_output_a's two groups at %d of %zu layers; in the "
                        "drafter's %zu stage(s) %d / %d; bit 0 q_a + kv, bit 1 the groups, decode band only\n",
                gq_now(), gq > 0 ? "declared" : "not declared (no library serves gemm_nt_ggml_pair)", qkv, oa,
                m.layers.size(), m.dsp.st.size(), dqkv, doa);
    if (e2_declared())
        fprintf(stderr, "radiance: deepseek41: lever 1U-E2 (ds41_e2, RADIANCE_DS41_E2, now %d) %s: the Engram kv GEMV by "
                        "output columns at %d of %d Engram layers -- %lld of the %lld columns a rank over a row view of the "
                        "weight (%lld bytes a row), then all_gather (row %lld), at <= %lld rows\n",
                e2_now() ? 1 : 0, e2 > 0 ? "declared" : "not declared", e2, eng, (long long)m.e2_part,
                (long long)(m.e2_part * m.g.world), (long long)m.e2_rb, (long long)m.e2_part, (long long)m.e2_rows);
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
            /* tp8 S4: a group this pipeline node zeroed holds nothing here to replicate (S6: index / latent apart) */
            const auto& zr = groups == &m.index ? m.pp_zero_r : m.pp_zero_cr;
            if (std::find(zr.begin(), zr.end(), kv.first) != zr.end()) continue;
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

/* Lever DMR: a stage's ffn mix beside its MoE's routed gate_up (the trunk's MixBeside on the drafter's buffers). */
struct DspMixBeside {
    const Layer* L;
    const Dsp* d;
    int64_t T;
};
static void dsp_mix_f_beside(RadCtx* x, void* a) {
    const DspMixBeside& p = *(const DspMixBeside*)a;
    const Layer& L = *p.L;
    const Dsp& d = *p.d;
    const int64_t T = p.T;
    RAD_ISSUE_N(x, L.mix_f, T, brows(d.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                brows(d.pre_f, T), brows(d.post, T), brows(d.comb, T));
}

/* The drafter's query pass (draft_pass > 0): `block` rows a drafting sequence, the anchor first. */
static void dsp_query(RadCtx* x, Model& m, const RadBatch* batch) {
    auto& d = m.dsp;   /* cw_up moves once (the confidence head's weights) */
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    if (T <= 0 || S <= 0) return;
    /* tp8 loop D18: the pass's block, the drafter's own or its short one (every drafting sequence carries the same) */
    /* (D19: the set of the pass's length among the declared ones) */
    const Dsp::Short* lo = nullptr;
    for (const Dsp::Short& z : d.sh)
        if (z.S != d.block && T == S * z.S) { lo = &z; break; }
    const int64_t blk = lo ? lo->S : d.block;
    if ((!lo && T != S * d.block) || T > d.Q) {
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
    RAD_ISSUE_N(x, lo ? lo->poff : d.poff_op, T, brows(d.poff, T));
    RAD_ISSUE_N(x, d.apos_op, T, pos, brows(d.poff, T), brows(d.apos, T));

    /* lever DMR (dmr_now): collapse + norm ahead of the connection's mix; the attention-zero rank keeps today's order */
    const int dmrv = m.zero ? 0 : dmr_now();
    const bool dmr = dmrv >= 1;
    for (const auto& L : d.st) {
        const size_t si = (size_t)(&L - d.st.data());   /* tp8 loop D18: the stage's short-block attention */
        const auto mix_a = [&] {
            RAD_ISSUE_N(x, L.mix_a, T, brows(d.h0, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale), RAD_W(L.hc_a_base),
                        brows(d.pre_a, T), brows(d.post, T), brows(d.comb, T));
        };
        if (!dmr) mix_a();
        const bool kvfold = dmrv == 2 && L.kv_fs;   /* lever DMR 2: the window key as the trunk's fold c */
        RAD_ISSUE_N(x, L.col_a, T, brows(d.h0, T), brows(d.pre_f, T), brows(d.x, T));
        if (m.zero) {
            RAD_ISSUE_N(x, L.attn_fill, T, brows(d.attn_out, T));
        } else {
            RAD_ISSUE_N(x, L.norm_a, T, brows(d.x, T), RAD_W(L.attn_norm), brows(d.xn, T));
            if (dmr) mix_a();
            const bool gqp = gq_qkv_on(L, T);   /* lever 1U-GQ: q_a + kv_g as one launch at q_a's point */
            if (gqp) gq_qkv_issue(x, m, L, T, brows(d.xn, T), brows(d.qr, T), brows(d.kv_, T));
            else lin_issue(x, m, L.q_a, T, brows(d.xn, T), L.wq_a, brows(d.qr, T));
            RAD_ISSUE_N(x, L.q_an, T, brows(d.qr, T), RAD_W(L.q_norm), brows(d.qrn, T));
            lin_issue(x, m, L.q_b, T, brows(d.qrn, T), L.wq_b, brows(d.q, T));
            RAD_ISSUE_N(x, L.q_rope, T, brows(d.q, T), brows(m.cs_a, g.max_ctx), pos);
            if (!gqp) lin_issue(x, m, L.kv_g, T, brows(d.xn, T), L.wkv, brows(d.kv_, T));
            if (kvfold) {   /* lever DMR 2: kv_n + kv_rope + kv_q + kv_st as the trunk's fold c, one launch */
                RAD_ISSUE_N(x, L.kv_fs, T, brows(d.kv_, T), RAD_W(L.kv_norm), brows(d.kvn, T), brows(m.cs_a, g.max_ctx), pos,
                            praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
            } else {
                RAD_ISSUE_N(x, L.kv_n, T, brows(d.kv_, T), RAD_W(L.kv_norm), brows(d.kvn, T));
                RAD_ISSUE_N(x, L.kv_rope, T, brows(d.kvn, T), brows(m.cs_a, g.max_ctx), pos);
                if (L.kv_q) RAD_ISSUE_N(x, L.kv_q, T, brows(d.kvn, T));
                /* The block's keys past the committed end; the next context pass overwrites them. */
                if (m.w8d)
                    RAD_ISSUE_N(x, L.kv_st, T, brows(d.kvn, T), praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
                else
                    RAD_ISSUE_N(x, L.kv_st, T, bcol(d.kvn, 0, hd / 2, T), bcol(d.kvn, hd / 2, hd / 2, T),
                                praw(kb->slot_mapping, RAD_I32, T), kv_cache(d.kv, L.l));
            }
            RAD_ISSUE_N(x, lo ? lo->attn[si] : L.attn, T, brows(d.q, T), kv_cache(d.kv, L.l),
                        praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch), praw(kb->seqused, RAD_I32, S),
                        RAD_NONE, RAD_NONE, RAD_NONE, brows(d.apos, T), cu, RAD_W(L.sinks), brows(d.o, T));
            RAD_ISSUE_N(x, L.o_rope, T, brows(d.o, T), brows(m.cs_a, g.max_ctx), pos);
            if (L.o_q) RAD_ISSUE_N(x, L.o_q, T, brows(d.o, T));
            const int64_t gd = HD / m.gr;
            if (gq_oa_on(L, T))   /* lever 1U-GQ */
                gq_oa_issue(x, m, L, T, bcol(d.o, 0, gd, T), bcol(d.o, gd, gd, T), bcol(d.oa, 0, c.o_rank, T),
                            bcol(d.oa, c.o_rank, c.o_rank, T));
            else
                for (int j = 0; j < m.gr; ++j)
                    lin_issue(x, m, L.o_a[(size_t)j], T, bcol(d.o, j * gd, gd, T), L.wo_a[(size_t)j],
                              bcol(d.oa, j * c.o_rank, c.o_rank, T));
            lin_issue(x, m, L.o_b, T, brows(d.oa, T), L.wo_b, brows(d.attn_out, T));
        }
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(d.attn_out, T), RAD_NONE);
        RAD_ISSUE_N(x, L.post_a, T, brows(d.attn_out, T), brows(d.h0, T), brows(d.post, T), brows(d.comb, T),
                    brows(d.h1, T));
        const auto mix_f = [&] {
            RAD_ISSUE_N(x, L.mix_f, T, brows(d.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                        brows(d.pre_f, T), brows(d.post, T), brows(d.comb, T));
        };
        if (!dmr) mix_f();
        RAD_ISSUE_N(x, L.col_f, T, brows(d.h1, T), brows(d.pre_a, T), brows(d.x, T));
        RAD_ISSUE_N(x, L.norm_f, T, brows(d.x, T), RAD_W(L.ffn_norm), brows(d.act.x, T));
        if (dmr) {
            /* the trunk's own order (step(): col_f, norm_f, then the MoE with mix_f beside its routed gate_up) */
            DspMixBeside mb{ &L, &d, T };
            MoeFP8::Beside bs;
            bs.fn = dsp_mix_f_beside;
            bs.arg = &mb;
            L.moe.step(x, batch, bs);
        } else {
            L.moe.step(x, batch);
        }
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
    /* tp8 loop D18: at the short block every walk is its own op set at blk steps, on the same planes */
    const std::vector<rad_op>& sst = lo ? lo->sstep : d.sstep;
    const bool wdecl = lo ? (!lo->wstep.empty() || lo->wmarkov) : (d.wfill || d.wmarkov);
    const bool sampled = !sst.empty() && ds41_switch("draft_sample") >= 1 && N == S * blk && S <= d.sseqs;
    const bool walked = sampled || (wdecl && wsw >= 1 && N == S * blk && S <= d.wseqs);
    if (sampled) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        const int64_t SS = d.sseqs, SR = 16, GW = SR * g.world;
        RAD_ISSUE_N(x, d.sfill, blk, brows(d.scnt, blk));
        for (int64_t l = 0; l <= blk; ++l) {
            const bool last = l == blk;
            RAD_ISSUE_N(x, sst[(size_t)l], S, brows(d.logits, N), anchor,
                        l ? brow_slice(d.sgath, (l - 1) * SS, S, GW) : RAD_NONE, RAD_W(d.mk1), RAD_W(d.mk2),
                        last ? RAD_NONE : brow_slice(d.skeys, l * SS, S, SR), brows(d.tokens, S),
                        last ? RAD_NONE : brows(d.sscr, S * d.snblk), last ? RAD_NONE : brow_slice(d.scnt, l, 1, 4),
                        praw(batch->seq_ids, RAD_I32, S), brows(d.sdraw, S), brows(d.sqt, S), RAD_B(d.sstate));
            if (!last) RAD_ISSUE(x, d.sag, brow_slice(d.skeys, l * SS, S, SR), brow_slice(d.sgath, l * SS, S, GW));
        }
    } else if (walked && (lo ? lo->wmarkov : d.wmarkov)) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        RAD_ISSUE_N(x, lo ? lo->wmarkov : d.wmarkov, S, brows(d.logits, N), anchor, RAD_W(d.mk1), RAD_W(d.mk2),
                    brows(d.wts, S), brows(d.wtt, S), brows(d.tokens, S));
    } else if (walked) {
        const RadOperand anchor = praw(batch->token_ids, RAD_I32, T);
        const int64_t WS = d.wseqs, KC = ds41walk::keys_cols(), GW = ds41walk::gath_cols(g.world);
        const int64_t KR = ds41walk::plane_rows(blk, WS);
        /* the scoring steps: W1b's form 1 while dspark_walk is 2 at the drafter's own block; the short block's form 0 */
        const std::vector<rad_op>& wst = lo ? lo->wstep : (wsw == 2 && !d.wstep2.empty() ? d.wstep2 : d.wstep);
        RAD_ISSUE_N(x, d.wfill, KR, brows(d.wkeys, KR));
        for (int64_t l = 0; l < blk; ++l) {
            const int64_t r0 = ds41walk::step_row(l, WS);
            RAD_ISSUE_N(x, wst[(size_t)l], S, brows(d.logits, N), anchor,
                        l ? brow_slice(d.wgath, ds41walk::step_row(l - 1, WS), S, GW) : RAD_NONE, RAD_W(d.mk1),
                        RAD_W(d.mk2), brow_slice(d.wkeys, r0, S, KC), brows(d.tokens, S));
            RAD_ISSUE(x, d.wag, brow_slice(d.wkeys, r0, S, KC), brow_slice(d.wgath, r0, S, GW));
        }
        RAD_ISSUE_N(x, (lo ? lo->wstep : d.wstep)[(size_t)blk], S, brows(d.logits, N), anchor,
                    brow_slice(d.wgath, ds41walk::step_row(blk - 1, WS), S, GW), RAD_W(d.mk1), RAD_W(d.mk2),
                    RAD_NONE, brows(d.tokens, S));
    } else {
        if (d.merge) {
            RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), brows(d.pairs, N));
            RAD_ISSUE(x, d.pgather, brows(d.pairs, N), brows(d.gath, N));
            RAD_ISSUE_N(x, d.merge, N, brows(d.gath, N), brows(d.cand_, N), brows(d.unary, N));
        } else {
            RAD_ISSUE_N(x, d.topk, N, brows(d.logits, N), brows(d.cand_, N), brows(d.unary, N), RAD_NONE);
        }
        RAD_ISSUE_N(x, lo ? lo->select : d.select, S, brows(d.cand_, N), brows(d.unary, N), RAD_NONE,
                    praw(batch->token_ids, RAD_I32, T), RAD_W(d.mk1), RAD_W(d.mk2), brows(d.tokens, S));
    }
    /* The DSpark extras (ds41-acc, rank 0), each only while its switch is on: the switches move at a
     * step release with a new epoch, so a pass recorded without them is never played with them. */
    if ((lo ? lo->conf : d.conf_op) && ds41_switch("dspark_conf") > 0) {
        if (!d.cw_up) {   /* the sidecar's weights, once, landed before the head is issued */
            void* p = rad_buf_ptr(x, d.cwb);
            if (!p || rad_memcpy_async(p, d.cw.data(), (int64_t)d.cw.size() * 4, rad_stream(x)) < 0 ||
                rad_stream_sync(rad_stream(x)) < 0) {
                rad_step_fail(x, "deepseek41: the DSpark confidence head's weights could not be copied to the card");
                return;
            }
            d.cw_up = true;
        }
        RAD_ISSUE_N(x, lo ? lo->conf : d.conf_op, S, brows(d.x, T), praw(batch->token_ids, RAD_I32, T),
                    brows(d.tokens, S), RAD_W(d.mk1), brows(d.cwb, 1), brows(d.conf, S));
    }
    if (d.alt_op && !lo && !walked && ds41_switch("dspark_alt") > 0)
        RAD_ISSUE_N(x, d.alt_op, S, brows(d.cand_, N), brows(d.unary, N), praw(batch->token_ids, RAD_I32, T),
                    brows(d.tokens, S), RAD_W(d.mk1), RAD_W(d.mk2), brows(d.alt, S));
}

/* The ffn connection's mix, issued by the MoE block behind its routed gate_up (MoeFP8::Beside). */
struct MixBeside {
    const Layer* L;
    const Model* m;
    int64_t T;
};
/* The ffn mix over the step's rows -- lever M1: those its pcn_f launches did not mix (t_mhc_f), a run a launch */
static void mix_f_beside(RadCtx* x, void* a) {
    const MixBeside& p = *(const MixBeside*)a;
    const Layer& L = *p.L;
    const Model& m = *p.m;
    const int64_t T = p.T;
    if (t_mhc_f.empty()) {
        RAD_ISSUE_N(x, L.mix_f, T, brows(m.h1, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                    RAD_W(L.hc_f_base), brows(m.pre_f, T), brows(m.post, T), brows(m.comb, T));
        return;
    }
    mhc_rest(t_mhc_f, 0, T, [&](int64_t r0, int64_t n) {
        auto R = [&](rad_buf h) { return r0 == 0 ? brows(h, n) : brow_slice(h, r0, n, m.bp.at(h)); };
        RAD_ISSUE_N(x, L.mix_f, n, R(m.h1), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base), R(m.pre_f), R(m.post),
                    R(m.comb));
    });
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
static std::atomic<long long> g_bi_onear{0};   /* lever A9: segmented steps whose MoE took one all-reduce (layer 0) */
/* Lever WMK in the engine (ds41-wmk2 stage 3, 14.11.2026): libr4d's fn4 ds41_wmk 1 runs every dense GGUF GEMM (gemm_nt_ggml,
 * its pairs and folds) in ONE part-order arithmetic at every row count, so a row's bytes no longer depend on the launch's
 * other rows or on the row count (the 48 / 49 decode-band flip is gone for these ops). step_bi then issues a layer's dense GGUF
 * GEMMs once over each run of consecutive rows that holds a prefill segment -- the decode rows of a MIXED step ride the prefill
 * launch instead of a decode-segment launch of their own (one weight read fewer a layer and op) -- and a run of decode rows
 * only keeps today's pieces (rows(kBiBand), the decode forms: the faster geometry there; the same bytes). Read at issue: a
 * flip of the switch bumps the epoch, so recorded passes re-record. Off (or libr4d without the switch): today's issue. */
static std::atomic<long long> g_wmkw_steps{0}, g_wmkw_runs{0}, g_wmkw_rows{0};
static bool wmk_whole_on() { return ds41_switch("ds41_wmk") >= 1; }   /* 1, or F7's A/B codes 2 / 3 (the same bytes) */
/* tp8 loop D22 (19.11.2026; the 64-row decode step's dense, the DRAFTER's lq150 lanecmp item 2): RADIANCE_DS41_DENSE_BAND=N
 * (49 .. 64, read once; unset or anything else: kBiBand, today) -- under lever WMK (ds41_wmk >= 1: every dense GGUF GEMM one
 * part order at every row count, so a launch's height does not move a row's bytes) a decode-only run of up to N rows issues
 * each dense GGUF GEMM through dense() (wo_a, wo_b, the Engram GEMM: today rows of 48 + the rest; q_a / kv / iproj / q_b /
 * iq_w go whole a segment already) and the logits head as ONE launch instead of 48 + the rest, each weight read once a
 * pass. It pays only with a libr4d whose decode-band G2 (gg_wk3) serves N rows (4 row tiles = 64 rows; today it declines
 * above 3 and gemm_nt_ggml takes the slower prefill form gg_pf at 49 .. 64: the same bytes). Off WMK: kBiBand. */
static int64_t dense_band_env() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_DS41_DENSE_BAND");
        if (!e || !*e) return kBiBand;
        char* end = nullptr;
        const long long n = std::strtoll(e, &end, 10);
        if (end && !*end && n >= kBiBand + 1 && n <= 64) return (int64_t)n;
        if (!(end && !*end && n == kBiBand))
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_DENSE_BAND=%s is not 49 .. 64: the dense decode band stays "
                            "%lld rows\n", e, (long long)kBiBand);
        return kBiBand;
    }();
    return v;
}
static std::atomic<long long> g_dband64_runs{0};   /* D22: decode-only dense runs above kBiBand issued as one launch (rank 0) */
/* the weight is a GGUF block weight served by gemm_nt_ggml (not IQR, not bf16 gemm_nt) */
static bool gguf_dense(const Model& m, rad_weight w) { return m.wty.count(w) && !m.wsc.count(w); }
/* The step's segments under ds41_bi; false when one launch over the step is the per-sequence arithmetic already (one
 * segment, decode rows within the norms' width): step() then runs today's issue. Prefill sequences' bounds from the
 * host mirror of positions (each sequence's rows are consecutive positions; a jump starts the next); when that does not
 * give the step's prefill sequence count (two sequences whose positions continue), cu_seqlens is read back (a sync). */
/* Lever DKV (dkv_read_env): layer L's attention over a pass's rows through the DKV ops where they serve it; false =
 * nothing issued (the caller issues today's op). `a`: the attention's operands as today's issue hands them -- q, window
 * cache, window table, window used, compressed cache, its table, sel, positions, cu_seqlens, sinks, out (then cos_sin
 * for the fold d form, `ofold`). `dec`: a decode pass or segment (the gathered rows, at most selkv_rows rows); else a
 * prefill one (the plan's rows, more than the decode band's rows). s0 / s1: its sequences; n: its rows at this layer;
 * hr0 / hn: the rows of m.lu its plan may own (the segment's rows before a PSK cut: the cut keeps its last rows, so the
 * plan's last positions -- the only rows it reads -- are the same). */
static bool dkv_attn(RadCtx* x, Model& m, const Layer& L, bool ofold, bool dec, int64_t s0, int64_t s1, int64_t n,
                     int64_t hr0, int64_t hn, std::vector<RadOperand> a, int64_t last_pos = -1) {
    const auto& c = m.c;
    const int l = L.l;
    if (L.r <= 0 || !m.lnv || a.size() < 11 || (ofold && a.size() < 12)) return false;
    const int src = c.src_of[(size_t)l], idx = c.idx_of[(size_t)l];
    /* tp8 loop D25 (selkv_cap > selkv_rows: RADIANCE_DS41_DKV_GPIECE=1): a piece gathers at its own rows `hr0` of the segment
     * and is held by its key beside the other pieces' (ds41_dkv_held.h); the launches are today's at every row count */
    const bool gp = m.selkv_cap > m.selkv_rows;
    /* step 3 (RADIANCE_DS41_DKV_GPIECE=2): the ops take up to selkv_cap rows in the decode band (libr4d's per-op band) */
    if (gp && dec && m.selkv && L.lgath && (ofold ? L.attn_rtg : L.attn_g) && n >= 1 &&
        n <= (m.selkv_one ? m.selkv_cap : m.selkv_rows) && hr0 >= 0 && hr0 + n <= m.selkv_cap) {
        const RadOperand gv = hr0 == 0 ? brows(m.selkv, n) : brow_slice(m.selkv, hr0, n, c.itopk * c.head_dim);
        Ds41Held k;
        k.src = src;
        k.idx = idx;
        k.s0 = s0;
        k.s1 = s1;
        k.r0 = hr0;
        k.n = n;
        if (ds41_held_take(t_dkv_gl, k)) {
            g_dkv_g_hit.fetch_add(1, std::memory_order_relaxed);
            if (rad_rank(x) == 0) g_dkv_gp_hit.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[3];
        } else {
            const RadOperand go[] = { a[4], a[5], a[6], a[8], gv };
            rad_issue(x, L.lgath, go, 5, n);
            g_dkv_g_n.fetch_add(1, std::memory_order_relaxed);
            if (rad_rank(x) == 0) g_dkv_gp_n.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[2];
            ++t_dkv_pass[4];
            dkv_note(x, "gather", l, n, hr0, n);
        }
        a.push_back(gv);
        rad_issue(x, ofold ? L.attn_rtg : L.attn_g, a.data(), (int)a.size(), n);
        ++t_dkv_pass[1];
        return true;
    }
    if (!gp && dec && m.selkv && L.lgath && (ofold ? L.attn_rtg : L.attn_g) && n >= 1 && n <= m.selkv_rows) {
        DkvHeld& h = t_dkv_g;
        const RadOperand gv = brows(m.selkv, n);
        if (h.src == src && h.idx == idx && h.s0 == s0 && h.s1 == s1 && h.r0 == hr0 && h.n == n) {
            g_dkv_g_hit.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[3];
        } else {
            const RadOperand go[] = { a[4], a[5], a[6], a[8], gv };
            rad_issue(x, L.lgath, go, 5, n);
            h.src = src;
            h.idx = idx;
            h.s0 = s0;
            h.s1 = s1;
            h.r0 = hr0;
            h.n = n;
            g_dkv_g_n.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[2];
            ++t_dkv_pass[4];
            dkv_note(x, "gather", l, n, 0, 0);
        }
        a.push_back(gv);
        rad_issue(x, ofold ? L.attn_rtg : L.attn_g, a.data(), (int)a.size(), n);
        ++t_dkv_pass[1];
        return true;
    }
    if (!dec && m.lu && L.lu_unpack && (ofold ? L.attn_rtlu : L.attn_lu) && n > kBiBand && hn >= n) {
        const RadOperand uv = hr0 == 0 ? brows(m.lu, hn) : brow_slice(m.lu, hr0, hn, m.bp.at(m.lu));
        bool held = false;
        for (const DkvHeld& h : t_dkv_lu)
            held = held || (h.src == src && h.s0 == s0 && h.s1 == s1 && h.r0 == hr0 && h.n == hn);
        if (held) {
            g_dkv_lu_hit.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[1];
        } else {
            /* a plan over other rows of m.lu that overlap these no longer holds its rows */
            t_dkv_lu.erase(std::remove_if(t_dkv_lu.begin(), t_dkv_lu.end(),
                                          [&](const DkvHeld& h) { return h.r0 < hr0 + hn && hr0 < h.r0 + h.n; }),
                           t_dkv_lu.end());
            const RadOperand uo[] = { a[4], a[5], a[7], a[8], uv };
            rad_issue(x, L.lu_unpack, uo, 5, n);
            DkvHeld h;
            h.src = src;
            h.s0 = s0;
            h.s1 = s1;
            h.r0 = hr0;
            h.n = hn;
            t_dkv_lu.push_back(h);
            g_dkv_lu_n.fetch_add(1, std::memory_order_relaxed);
            ++t_dkv_ct[0];
            ++t_dkv_pass[3];
            dkv_note(x, "unpack", l, n, hr0, hn);
        }
        a.push_back(uv);
        /* one sequence whose last position the host holds: the plan holds every entry it may select when (last + 1) /
         * ratio fits the plan's rows (the device plan's reach is at most that) -> the LDS-free form */
        const bool full = last_pos >= 0 && s1 - s0 == 1 && (last_pos + 1) / L.r <= m.lu_e * hn;
        const rad_op fo = ofold ? L.attn_rtluf : L.attn_luf;
        rad_issue(x, full && fo ? fo : (ofold ? L.attn_rtlu : L.attn_lu), a.data(), (int)a.size(), n);
        ++t_dkv_pass[0];
        if (full && fo) ++t_dkv_full;
        return true;
    }
    if (n > kBiBand) ++t_dkv_pass[2];
    return false;
}

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

/* ------------------------------------------------------------------ pictures in a trunk step (Model::media) */
/* Every MoE layer's image rows for this step (MoeFP8::Vl): the media rows, or none. Set at every trunk step, so a step
 * without media rows leaves the routing exactly as before. */
static void media_route(Model& m, const RadBatch* batch) {
    if (!m.media) return;
    const bool on = batch->n_mm_rows > 0 && batch->mm_rows;
    for (auto& L : m.layers) {
        L.moe.vl_rows = on ? batch->mm_rows : nullptr;
        L.moe.vl_n = on ? batch->n_mm_rows : 0;
    }
}
/* The media rows' compressed tokens: the n-gram pad (engram.py's DEAD), over token_map's ids. */
static void media_tokens(RadCtx* x, Model& m, const RadBatch* batch, int64_t T) {
    if (!m.mm_tok_put || batch->n_mm_rows <= 0 || !batch->mm_rows) return;
    const int64_t n = batch->n_mm_rows;
    void* eos = rad_buf_ptr(x, m.mm_eos);
    void* tok = rad_buf_ptr(x, m.ctok);
    if (!eos || !tok) {
        rad_step_fail(x, "deepseek41: the media rows' token buffers have no address");
        return;
    }
    RAD_ISSUE_N(x, m.mm_iota, n, brows(m.mm_eos, n));
    RAD_ISSUE_N(x, m.mm_tok_put, n, praw2(eos, RAD_U8, n, 4), praw(batch->mm_rows, RAD_I32, n),
                praw2(tok, RAD_U8, T, 4));
}
/* An Engram layer's value rows at the media rows: zero, so its gate adds nothing there. */
static void media_engram(RadCtx* x, Model& m, const Layer& L, const RadBatch* batch, int64_t T) {
    if (!L.mm_put || batch->n_mm_rows <= 0 || !batch->mm_rows) return;
    const int64_t n = batch->n_mm_rows, D = m.c.n_embd, hc = m.c.hc;
    RAD_ISSUE_N(x, L.mm_fill, kMmZeroRows, brows(m.mm_zero, kMmZeroRows));
    for (int64_t a = 0; a < n; a += kMmZeroRows) {
        const int64_t k = std::min<int64_t>(kMmZeroRows, n - a);
        RAD_ISSUE_N(x, L.mm_put, k, brows(m.mm_zero, k), praw(batch->mm_rows + a, RAD_I32, k),
                    bcol(m.ekv, hc * D, D, T));
    }
}

/* ------------------------------------------------------------------ tp8 S4 / S6: the crossings in a step */
/* Crossing `ring`'s issue on this node: the core's ring, then the parts (ds41_cut_liveset) -- the residual stream, the
 * selections and the keep words over the boundary's live rows [lo, hi) (step(): the whole step; step_bi: the rows of the
 * boundary layer's segments, which every layer after it runs a subset of), the copied source's entries over its work
 * rows. Both nodes compute lo / hi and the work rows from the same batch, so the two ops agree (the trailer's layout
 * word checks it). */
static bool pp_cross(RadCtx* x, Model& m, const RadBatch* batch, int ring, int64_t lo, int64_t hi, bool sh = false) {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    if (ring >= kSegRecv) {   /* tp8 dev M4: a segment boundary's save / restore, this rank's ring buffer, the live rows */
        const bool send = ring >= kSegSend;
        const size_t k = (size_t)(ring - (send ? kSegSend : kSegRecv));
        if (k >= m.seg_b.size() || !(send ? m.seg_send[k] : m.seg_recv[k])) {
            rad_step_fail(x, "deepseek41: a segment boundary the follower did not declare");
            return false;
        }
        /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG=2): the row-sharded twin, over the pass's rows [0, T) (an SP boundary's) */
        const rad_op sop = !sh ? (send ? m.seg_send[k] : m.seg_recv[k])
                         : k < m.seg_send_sh.size() ? (send ? m.seg_send_sh[k] : m.seg_recv_sh[k]) : 0;
        if (!sop || (sh && (lo != 0 || hi != T))) {
            rad_step_fail(x, "deepseek41: RADIANCE_DS41_SPHC_SEG: a row-sharded segment boundary not declared or not over the "
                             "pass's rows (a plan defect)");
            return false;
        }
        if (hi > T || hi <= lo) {
            lo = std::max<int64_t>(0, std::min(lo, T - 1));
            hi = lo + 1;
        }
        lo = std::max<int64_t>(0, lo);
        static thread_local std::vector<RadOperand> so;
        so.clear();
        so.push_back(brows(m.seg_ring, m.seg_ring_bytes));
        /* loop FINDINGS F12: X2's source entries below the boundary over their work rows, as X2 ships them */
        const ds41layers::Crossing& sc = m.seg_cx[k];
        const int64_t swr = sc.src >= 0 ? pp_work_rows(m, m.c.ratio[(size_t)sc.src], T, S) : T;
        for (const CutPart& p : ds41_cut_liveset_cx(m, sc)) {
            if (p.rows == CutRows::Step) so.push_back(brows(p.b, T));
            else if (p.rows == CutRows::Work) so.push_back(brows(p.b, swr));
            else so.push_back(lo == 0 ? brows(p.b, hi) : brow_slice(p.b, lo, hi - lo, p.pitch));
        }
        rad_issue(x, sop, so.data(), (int)so.size(), T);
        if (sh && rad_rank(x) == 0) g_sp_seg_sh.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    /* tp8 loop P28 round 2c: a band pass's X1 is the band form (whole: never row-sharded) */
    const bool bx = ring == RAD_PP_X1 && t_band;
    if (bx && sh) {
        rad_step_fail(x, "deepseek41: RADIANCE_DS41_PP_BAND: a band pass's X1 does not cross row-sharded (a plan defect)");
        return false;
    }
    const rad_op op_ = bx ? m.pp_x1b : ring == RAD_PP_X1 ? (sh && m.pp_x1_sh ? m.pp_x1_sh : m.pp_x1) : m.pp_x2;   /* P20 r3: `sh` */
    /* the ring read at every issue (tp8 S5: each lane has its own) */
    m.pp_ring[ring] = rad_pp_query().ring[ring];
    if (!m.pp_ring[ring] || !op_) {
        rad_step_fail(x, "deepseek41: two-node pipeline: the core set up no ring for a crossing");
        return false;
    }
    const ds41layers::Crossing& cx = m.pp_cx[ring];
    /* the work rows of the source the crossing carries (a band X1's: its copy's, the same source as X1 of W(h, t)) */
    const int wsrc = bx ? (m.band.x1.src >= 0 ? m.band.x1.src : m.band.copy_src) : cx.src;
    const int64_t wr = wsrc >= 0 ? pp_work_rows(m, m.c.ratio[(size_t)wsrc], T, S) : T;
    if (hi > T || hi <= lo) {   /* no live row: one row crosses (the op's range starts at 1; the receiver reads none) */
        lo = std::max<int64_t>(0, std::min(lo, T - 1));
        hi = lo + 1;
    }
    lo = std::max<int64_t>(0, lo);
    static thread_local std::vector<RadOperand> o;
    o.clear();
    o.push_back(praw(m.pp_ring[ring], RAD_U8, 1));
    for (const CutPart& p : bx ? ds41_band_liveset(m) : ds41_cut_liveset(m, ring)) {
        if (p.rows == CutRows::Ids) o.push_back(praw(batch->token_ids, RAD_I32, T));   /* tp8 dev D10 round 2 */
        else if (p.rows == CutRows::Step) o.push_back(brows(p.b, T));
        else if (p.rows == CutRows::Work) o.push_back(brows(p.b, wr));
        else o.push_back(lo == 0 ? brows(p.b, hi) : brow_slice(p.b, lo, hi - lo, p.pitch));
    }
    rad_issue(x, op_, o.data(), (int)o.size(), T);
    return true;
}
/* The receiver's copy stores of the crossing into it (pp_copy_declare): the owner's entry stores over the same rows at
 * the same pages (ratio > 1: the shipped tail pages; ratio 1: the batch's slot mapping), into this node's copy. `store`:
 * the owner stored this pass (step_bi: not an all-replay pass, lever PSK-CR). */
static bool pp_copy_issue(RadCtx* x, Model& m, const RadBatch* batch, int ring, bool store) {
    if (ring >= kSegRecv) return true;   /* tp8 dev M4: a segment boundary carries no kv entries */
    const ds41layers::Crossing& cx = m.pp_cx[ring];
    if (cx.src < 0 || !store) return true;
    const int64_t T = batch->n_tok, S = batch->n_seq, r = m.c.ratio[(size_t)cx.src];
    const int64_t wr = pp_work_rows(m, r, T, S);
    RadOperand page = brows(m.page, wr);
    if (r <= 1) {
        const auto* cb = kv_batch(batch, m.comp[r]);
        if (!cb || !cb->slot_mapping) {
            rad_step_fail(x, "deepseek41: two-node pipeline: the copied source's group has no slot mapping");
            return false;
        }
        page = praw(cb->slot_mapping, RAD_I32, T);
    }
    if (m.pp_cp_lat) RAD_ISSUE_N(x, m.pp_cp_lat, wr, brows(m.lat, wr), page, kv_cache(m.comp[r], cx.src));
    if (m.pp_cp_idx) RAD_ISSUE_N(x, m.pp_cp_idx, wr, brows(m.ikn, wr), page, kv_cache(m.index[r], cx.src));
    return true;
}
/* tp8 loop P28 round 2c: after a band pass's X1 on the follower, the band layers' window keys into their window pages (the
 * owner's store op over the pass's rows, the window group's slot mapping) */
static bool band_store_issue(RadCtx* x, Model& m, const RadBatch* batch) {
    if (!t_band || m.pp_rank != 1 || m.band_k <= 0) return true;
    const int64_t T = batch->n_tok, hd = m.c.head_dim;
    const auto* wb = kv_batch(batch, m.win);
    if (!wb || !wb->slot_mapping) {
        rad_step_fail(x, "deepseek41: RADIANCE_DS41_PP_BAND: the window group has no slot mapping");
        return false;
    }
    for (size_t j = 0; j < m.band_st.size(); ++j) {
        const int l = m.band.h + (int)j;
        if (!m.band_st[j]) {
            rad_step_fail(x, "deepseek41: RADIANCE_DS41_PP_BAND: a band layer's store was not declared");
            return false;
        }
        if (m.w8)
            RAD_ISSUE_N(x, m.band_st[j], T, brows(m.band_key[j], T), praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
        else
            RAD_ISSUE_N(x, m.band_st[j], T, bcol(m.band_key[j], 0, hd / 2, T), bcol(m.band_key[j], hd / 2, hd / 2, T),
                        praw(wb->slot_mapping, RAD_I32, T), kv_cache(m.win, l));
    }
    return true;
}
/* This node's crossing in at layer l (the leader X2 at t, the follower X1 at h > 0), or -1 */
/* tp8 dev M4: the follower segment this rank's thread issues (pp_part sets it at a pass's start): [t_seg0, t_seg1) */
static thread_local bool t_seg = false;
static thread_local int t_seg0 = 0, t_seg1 = 0;
/* tp8 loop P28 round 2c: where X1 crosses in the pass being issued: h, or h + k on a band pass */
static int pp_he(const Model& m) { return m.pp_h + (t_band ? m.band_k : 0); }
static int seg_index(const Model& m, int s) {
    for (size_t k = 0; k < m.seg_b.size(); ++k)
        if (m.seg_b[k] == s) return (int)k;
    return -1;
}
/* ... whether this pass issues the follower's first layers (X1, the lookup): the whole pass, or its first segment */
static bool pp_seg_first(const Model& m) { return !t_seg || t_seg0 == pp_he(m); }
/* tp8 dev D10 round 2: the follower at h > 0 under pp_ids issues its token-id ops after X1 at layer h (in the pass's first
 * segment; a later segment's prologue issues them as before: X1 is in its batch by then) */
static bool pp_ids_late(const Model& m) { return m.pp && m.pp_ids && m.pp_rank == 1 && m.pp_h > 0 && pp_seg_first(m); }
/* tp8 dev M9: a leader segment's boundary at its start / end: not the part's own ends (layer 0 / X1 at h; X2 at t / the
 * final norm at n) */
static bool pp_lead_bound(const Model& m, int s) {
    return s > 0 && s != m.pp_h && s != pp_he(m) && s != m.pp_t && s < (int)m.c.n_layer;
}
static int pp_recv_at(const Model& m, int l) {
    if (!m.pp) return -1;
    if (m.pp_rank == 0 && l == m.pp_t && !(t_seg && t_seg0 != m.pp_t)) return RAD_PP_X2;   /* tp8 dev M9: a later tail segment restores */
    if (t_seg && m.pp_rank == 1 && l == t_seg0 && t_seg0 > pp_he(m)) {   /* tp8 dev M4: a segment from boundary s */
        const int k = seg_index(m, l);
        return k >= 0 ? kSegRecv + k : -1;
    }
    if (t_seg && m.pp_rank == 0 && l == t_seg0 && pp_lead_bound(m, t_seg0)) {   /* tp8 dev M9: a leader segment from s */
        const int k = seg_index(m, l);
        return k >= 0 ? kSegRecv + k : -1;
    }
    if (m.pp_rank == 1 && m.pp_h > 0 && l == pp_he(m)) return RAD_PP_X1;   /* tp8 loop P28: h + k on a band pass */
    return -1;
}
/* ... and out after layer l (the follower X2 after t - 1, the leader X1 after h - 1 when h > 0), or -1 */
static int pp_send_after(const Model& m, int l) {
    if (!m.pp) return -1;
    if (t_seg && m.pp_rank == 1 && l == t_seg1 - 1 && t_seg1 < m.pp_t) {   /* tp8 dev M4: a segment to boundary s */
        const int k = seg_index(m, t_seg1);
        return k >= 0 ? kSegSend + k : -1;
    }
    if (t_seg && m.pp_rank == 0 && l == t_seg1 - 1 && pp_lead_bound(m, t_seg1)) {   /* tp8 dev M9: a leader segment to s */
        const int k = seg_index(m, t_seg1);
        return k >= 0 ? kSegSend + k : -1;
    }
    if (m.pp_rank == 1 && l == m.pp_t - 1) return RAD_PP_X2;
    if (m.pp_rank == 0 && m.pp_h > 0 && l == pp_he(m) - 1) return RAD_PP_X1;   /* tp8 loop P28: after h + k - 1 on a band pass */
    return -1;
}
/* tp8 S5 + S6: the part of the leader's pass being issued (abi/rad_pp.h). With lanes the leader issues a lane step's pass
 * as its head -- the prologue, layers [0, h) and X1's send after layer h - 1 (at h = 0: the lookup and X1) -- and its tail
 * -- the prologue's per-step tables again (expand, the mixes, the rope tables, the token map: functions of the batch), X2's
 * receive at layer t and everything after -- with other lanes' parts between them on the stream. What the tail reads of
 * the head crosses nothing: the residual and the follower's reads come back in X2; the head's own kv / state writes are
 * paged per sequence; the selections / keep words its layers >= t read are never the head's (lanes_head_tail_reads,
 * checked at declare). Whole: one machine, one lane, the follower. */
static int pp_part(const Model& m) {
    t_seg = false;
    t_band = false;
    if (!m.pp) return RAD_PP_PART_WHOLE;
    const RadPpInfo p = rad_pp_query();
    /* tp8 loop P28 round 2c: the engine's band word (0 from an engine before P28: never a band pass) */
    t_band = m.band_k > 0 && p.reserved[0] == m.band_k;
    if (m.pp_rank != 0) {
        /* tp8 dev M4: a follower segment [seg_l0, seg_l1) at the boundaries this node declared */
        if (p.part != RAD_PP_PART_SEG || m.seg_b.empty()) return RAD_PP_PART_WHOLE;
        t_seg = true;
        t_seg0 = p.seg_l0;
        t_seg1 = p.seg_l1;
        return RAD_PP_PART_SEG;
    }
    /* tp8 dev M9: a leader head / tail segment [seg_l0, seg_l1) at the boundaries this node declared (RAD_PP_SEG_END: n) */
    if ((p.part == RAD_PP_PART_HEAD || p.part == RAD_PP_PART_TAIL) && p.seg_l1 > 0 && m.seg_lead) {
        t_seg = true;
        t_seg0 = p.seg_l0;
        t_seg1 = std::min<int>(p.seg_l1, (int)m.c.n_layer);
    }
    return p.part;
}
/* ... and whether the part issues layer l (the head [0, h), the tail [t, n), a follower segment its layers; the whole pass
 * every held layer; tp8 dev M9: a leader segment its layers of its part) */
static bool pp_part_layer(const Model& m, int part, int l) {
    if (part == RAD_PP_PART_SEG) return l >= t_seg0 && l < t_seg1;
    if (t_seg && (part == RAD_PP_PART_HEAD || part == RAD_PP_PART_TAIL) && (l < t_seg0 || l >= t_seg1)) return false;
    if (part == RAD_PP_PART_HEAD) return l < pp_he(m);
    if (part == RAD_PP_PART_TAIL) return l >= m.pp_t;
    return !(t_band && m.pp_rank == 1) || l >= pp_he(m);   /* tp8 loop P28: a band pass's follower from h + k */
}
/* tp8 dev M9: whether this leader part issues the head's first layers (the lookup): not a head segment from a boundary */
static bool pp_lead_first(const Model& m) { return !(t_seg && m.pp_rank == 0 && t_seg0 > 0); }
/* The receive's boundary op where the layer has no fused pcn_a: X1 / X2's, or a segment boundary's (tp8 dev M4) */
static rad_op pp_post_op(const Model& m, int ring) {
    return ring >= kSegRecv && ring < kSegSend ? m.seg_post[(size_t)(ring - kSegRecv)] : m.pp_post;
}

/* tp8 loop D24 (fn4 ds41_prefetch): this pass's bits, 0 = no reads -- a trunk pass (no drafter pass, no encoder) whose
 * every row decodes (batch_split: no prefill sequence), of 1 .. kPfRows rows, with the reads declared. Read at issue, so a
 * recorded pass keeps what it was recorded with (a switch flip bumps the epoch: the passes re-record). */
static int pf_bits(const RadBatch* b) {
    const bool ok = g_sw_pf_ok.load(std::memory_order_relaxed) != 0;
    const int v = g_sw_pf.load(std::memory_order_relaxed);
    if (!ok || v <= 0) return 0;
    int64_t Dn = 0, DT = 0;
    batch_split(b, &Dn, &DT);
    return ds41_pf::pass_bits(v, ok, b->draft_pass, b->enc != 0 || b->gen != 0, b->n_tok, b->n_seq, Dn, DT);
}
static void pf_read(RadCtx* x, rad_op op, rad_weight w, int& n) {
    if (!op || !w) return;
    prefetch_beside(x, op, w, 1);   /* a height of 1: pf_bits decided the pass qualifies (kPfRows, not kPrefetchRows) */
    ++n;
}
static void pf_note(RadCtx* x, int n) {
    if (n <= 0 || rad_rank(x) != 0) return;
    g_pf_reads.fetch_add(n, std::memory_order_relaxed);
}
/* Beside this layer's attention all-reduce (issued right before it): the FFN side the fused post + mix and the MoE read
 * first -- hc_f_fn (bit 0), the router projection and the shared arm's gate_up (bit 1). */
static void pf_attn(RadCtx* x, const Layer& L, int bits) {
    const unsigned r = ds41_pf::site_reads(bits, 0, L.src, true);
    int n = 0;
    if (r & ds41_pf::kMixF) pf_read(x, L.pf_f, L.hc_f_fn, n);
    if (r & ds41_pf::kRouter) pf_read(x, L.pf_r, L.moe.c.sigmoid_router ? L.moe.glm_router.projection : L.moe.w_router, n);
    if (r & ds41_pf::kShGu) pf_read(x, L.pf_s, L.moe.sh_gate_up.w, n);
    pf_note(x, n);
}
/* Beside a layer's MoE all-reduce (MoeFP8::Beside::before_ar, right before the pass's gather + all-reduce): the next layer
 * this pass issues -- its mix fn (bit 0; read by the fused post + collapse + norm + mix right after the reduction), its
 * comp_w at a kv source, wq_a and wkv (bit 2). Nothing where the layer ends this node's part of the pass (a crossing). */
struct PfBeside {
    const Model* m = nullptr;
    const Layer* L = nullptr;
    int bits = 0;
};
static void pf_moe_beside(RadCtx* x, void* a) {
    const PfBeside& p = *(const PfBeside*)a;
    const Model& m = *p.m;
    const Layer* nx = pp_send_after(m, p.L->l) >= 0 ? nullptr : next_layer(m, p.L->l);
    const unsigned r = ds41_pf::site_reads(p.bits, 1, nx && nx->src, nx && m.lr.held(nx->l));
    if (!r) return;
    int n = 0;
    if (r & ds41_pf::kMixA) pf_read(x, nx->pf_a, nx->hc_a_fn, n);
    if (r & ds41_pf::kComp) pf_read(x, nx->pf_c, nx->comp_w, n);
    if (r & ds41_pf::kQa) pf_read(x, nx->pf_q, nx->wq_a, n);
    if (r & ds41_pf::kKv) pf_read(x, nx->pf_kv, nx->wkv, n);
    pf_note(x, n);
}
/* one line now and then (rank 0, at issue: a recorded pass counts when it is recorded) */
static void pf_pass_note(RadCtx* x, const RadBatch* b, int bits) {
    if (bits <= 0 || rad_rank(x) != 0) return;
    const long long n = g_pf_passes.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || n % 512 == 0)
        fprintf(stderr, "radiance: deepseek41: tp8 loop D24 (ds41_prefetch %d): %lld decode passes issued %lld weight reads "
                        "beside their all-reduces (rank 0, issued or recorded; this one %lld rows)\n",
                bits, n, g_pf_reads.load(std::memory_order_relaxed), (long long)b->n_tok);
}

/* `psk_only`: a one-prefill-sequence step step() hands over for lever PSK alone (no BI segmenting; the BI stats skip it) */
static void step_bi(RadCtx* x, Model& m, const RadBatch* batch, const std::vector<BiSeg>& sg, bool psk_only = false) {
    dkv_pass_reset(x, batch->n_tok);   /* lever DKV: nothing held at a pass's start (each part of a lane step's pass too) */
    const int pfb = pf_bits(batch);   /* tp8 loop D24: 0 = no weight reads beside the all-reduces */
    pf_pass_note(x, batch, pfb);
    const int part = pp_part(m);       /* tp8 S5 + S6: the leader's head / tail part, or the whole pass */
    const bool run_head = part != RAD_PP_PART_TAIL;
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
        /* tp8 loop D22: its own line beside, once a decode-only dense run above the band went as one launch */
        if (n > 0 && n % 512 == 0 && rad_rank(x) == 0 && g_dband64_runs.load(std::memory_order_relaxed) > 0)
            fprintf(stderr, "radiance: deepseek41: tp8 loop D22 (RADIANCE_DS41_DENSE_BAND=%lld, or 64 under ds41_wmk 5): %lld decode-only "
                            "dense GGUF runs above %lld rows issued as one launch (layer-ops summed)\n", (long long)dense_band_env(),
                    g_dband64_runs.load(std::memory_order_relaxed), (long long)kBiBand);
        /* lever A9 (ds41_bi 5 / 6): its own line beside, once a step took it (the line above stays int12's) */
        if (n > 0 && n % 512 == 0 && rad_rank(x) == 0 && g_bi_onear.load(std::memory_order_relaxed) > 0)
            fprintf(stderr, "radiance: deepseek41: lever A9 stats: %lld of %lld segmented steps with one MoE all-reduce; "
                            "ds41_bi %d\n", g_bi_onear.load(std::memory_order_relaxed), n,
                    g_sw_bi.load(std::memory_order_relaxed));
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
    t_mhc_a.clear();   /* lever M1: nothing fused yet this step */
    t_mhc_a_l = -1;
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
    /* lever WMK (ds41_wmk 1, wmk_whole_on): a dense GGUF GEMM over each run of consecutive rows that holds a prefill segment in
     * ONE launch (the decode rows ride it), a decode-only run as rows(kBiBand); off: rows(kBiBand) (`wmkw` false) or the caller's
     * own by-segment issue. `seg`: the caller's per-segment form (q_a / kv / q_b / the indexer's linears: by segment, uncut). */
    const bool wmkw = wmk_whole_on();
    /* tp8 loop D22 (RADIANCE_DS41_DENSE_BAND): a decode-only run's dense GGUF GEMM band under WMK (kBiBand otherwise); DECODE
     * round 18: libr4d's fn4 ds41_wmk 5 (its decode band's part-order launch up to 64 rows) takes the band to 64 in process, so
     * one start A/Bs ds41_wmk 2 / 5 (a switch flip bumps the epoch: recorded passes re-record) */
    const int64_t dband = wmkw ? ds41_dense_band(ds41_switch("ds41_wmk"), dense_band_env(), kBiBand) : kBiBand;
    auto dense = [&](bool gg, bool seg, const auto& fn) {
        const int64_t band = gg ? dband : kBiBand;
        const auto by_seg = [&](int64_t a0, int64_t a1) {
            for (const BiSeg& s : *cur) {
                if (s.r0 < a0 || s.r0 >= a1) continue;
                if (seg || !s.dec || s.n <= band) {
                    if (!seg && s.dec && s.n > kBiBand && rad_rank(x) == 0) g_dband64_runs.fetch_add(1, std::memory_order_relaxed);
                    const TwinScope tws(!s.dec, s.n);   /* tp8 loop P30t: a prompt segment's launch may take its twin */
                    fn(s.r0, s.n);
                    continue;
                }
                static thread_local std::vector<std::pair<int64_t, int64_t>> bc;
                ds41_band_cuts(s.n, band, bc);
                for (const auto& c : bc) fn(s.r0 + c.first, c.second);
            }
        };
        if (!wmkw || !gg) { by_seg(0, T); return; }
        /* tp8 loop P30t: a run holding a prompt segment that takes the twin is issued by segment (its prompt rows on the twin,
         * its decode rows in their own GGUF launches) */
        const int tw_mk = t_twin_sub ? 0 : twin_mask_now();
        const int64_t tw_r = tw_mk ? twin_rows_now() : 0;
        for (const auto& u : runs) {
            int64_t dn = 0, pn = 0;
            bool tw_run = false;
            for (const BiSeg& s : *cur)
                if (s.r0 >= u.first && s.r0 < u.first + u.second) {
                    (s.dec ? dn : pn) += s.n;
                    tw_run = tw_run || (tw_mk && !s.dec && s.n >= tw_r);
                }
            if (tw_run) {
                by_seg(u.first, u.first + u.second);
                continue;
            }
            /* a mixed run in one launch where that is the faster geometry: the run fits the decode band (one <= 48-row launch for
             * both), or its prefill rows are past the band already (they run a prefill form anyway; the decode rows add <= their
             * share of it). A short prefill tail next to decode rows (pn <= 48 < dn + pn) keeps two decode-band launches: one
             * prefill-form launch of 49..96 rows is slower than both (K-serial tiles). */
            if (dn > 0 && pn > 0 && (pn > kBiBand || dn + pn <= kBiBand)) {   /* a mixed run: one launch */
                fn(u.first, u.second);
                if (rad_rank(x) == 0) {
                    g_wmkw_runs.fetch_add(1, std::memory_order_relaxed);
                    g_wmkw_rows.fetch_add(u.second, std::memory_order_relaxed);
                }
                continue;
            }
            by_seg(u.first, u.first + u.second);
        }
    };
    if (wmkw && rad_rank(x) == 0) {
        bool pre = false, dec = false;
        for (const BiSeg& s : sg) (s.dec ? dec : pre) = true;
        if (pre && dec) {
            const long long n = g_wmkw_steps.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n == 1 || n % 256 == 0)
                fprintf(stderr, "radiance: deepseek41: lever WMK (ds41_wmk 1): %lld mixed steps issued their dense GGUF GEMMs over "
                                "whole runs (%lld run launches a layer-op, %lld rows); this step %lld rows\n",
                        n, g_wmkw_runs.load(std::memory_order_relaxed), g_wmkw_rows.load(std::memory_order_relaxed),
                        (long long)T);
        }
    }
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
    /* tp8 loop F58 (RADIANCE_DS41_BI_DBAND=1, m.bi_dsub): the decode segment's band pieces when it is above kBiBand rows
     * (`dap`, empty otherwise: today's issue), and piece j's cu view (dcu): piece 0 starts at row 0, so the batch's own
     * cu[0 .. s1] is its view; a later piece reads bi_dcu row j = cu[s0 .. s1] - cu[s0], written here once a step. The
     * sequences' first rows: every decode sequence at max_q_len_decode rows (the recorded passes: rad_arch_pass_key keys
     * only that layout), else the host positions' jumps, else cu read back (a sync, as bi_segments' prefill rule). */
    static thread_local std::vector<Ds41DecPiece> dap;
    static thread_local std::vector<int64_t> dst;
    dap.clear();
    if (m.bi_dsub && !psk_only && !sg.empty() && sg[0].dec && sg[0].n > kBiBand && sg[0].r0 == 0 && sg[0].s0 == 0) {
        const int64_t dn = sg[0].s1 - sg[0].s0, dt = sg[0].n, q = batch->max_q_len_decode;
        dst.clear();
        const bool uni = q > 0 && dn * q == dt;
        const bool pos_ok = !uni && ds41_dec_starts(batch->positions_h, dt, dn, dst);
        if (!uni && rad_rank(x) == 0)   /* rad_arch_pass_key's verdict on this layout: recorded by its pieces, or live */
            (g_bi_dband.load(std::memory_order_relaxed) >= 2 && pos_ok ? g_dband_keyed : g_dband_live)
                .fetch_add(1, std::memory_order_relaxed);
        if (uni)
            for (int64_t s = 0; s < dn; ++s) dst.push_back(s * q);
        else if (!pos_ok) {
            std::vector<int32_t> cuh((size_t)dn + 1);
            rad_memcpy_async(cuh.data(), batch->cu_seqlens, (dn + 1) * 4, rad_stream(x));
            rad_stream_sync(rad_stream(x));
            for (int64_t s = 0; s < dn; ++s) dst.push_back(cuh[(size_t)s]);
            if (rad_rank(x) == 0) g_dband_sync.fetch_add(1, std::memory_order_relaxed);
        }
        const int64_t over = ds41_dec_pieces(dst, dt, kBiBand, dap);
        bool fits = over == 0 && (int64_t)dap.size() <= std::max<int64_t>(1, m.g.max_seqs);
        for (const Ds41DecPiece& p : dap) fits = fits && p.s1 - p.s0 + 1 <= kBiBand + 1;
        if (!fits) {
            if (rad_rank(x) == 0) g_dband_over.fetch_add(1, std::memory_order_relaxed);
            dap.clear();
        }
        for (size_t j = 1; j < dap.size(); ++j) {
            const int64_t ns = dap[j].s1 - dap[j].s0;
            RAD_ISSUE_N(x, m.bi_dsub, ns + 1, praw(batch->cu_seqlens + dap[j].s0, RAD_I32, ns + 1),
                        praw(batch->cu_seqlens + dap[j].s0, RAD_I32, 1), bcol_at(m.bi_dcu, (int64_t)j, kBiBand + 1, 0, ns + 1, 1));
        }
        if (!dap.empty() && rad_rank(x) == 0) {
            const long long n = g_dband_steps.fetch_add(1, std::memory_order_relaxed) + 1;
            g_dband_pieces.fetch_add((long long)dap.size(), std::memory_order_relaxed);
            if (n == 1 || n % 4096 == 0)
                fprintf(stderr, "radiance: deepseek41: tp8 loop F58 (RADIANCE_DS41_BI_DBAND=%d): %lld decode segments above %lld "
                                "rows in band pieces (%lld pieces, %lld indexer runs a layer-op summed; %lld cu read-backs, %lld "
                                "left whole; uneven layouts %lld issued live, %lld recorded by their pieces); this one %lld rows of "
                                "%lld sequences in %zu pieces\n",
                        g_bi_dband.load(std::memory_order_relaxed), n, (long long)kBiBand,
                        g_dband_pieces.load(std::memory_order_relaxed), g_dband_runs.load(std::memory_order_relaxed),
                        g_dband_sync.load(std::memory_order_relaxed), g_dband_over.load(std::memory_order_relaxed),
                        g_dband_live.load(std::memory_order_relaxed), g_dband_keyed.load(std::memory_order_relaxed),
                        (long long)dt, (long long)dn, dap.size());
            if (m.selkv_one && (n == 1 || n % 4096 == 0))
                fprintf(stderr, "radiance: deepseek41: tp8 loop D25 step 3 (RADIANCE_DS41_DKV_GPIECE=2): %lld decode-segment attentions "
                                "above %lld rows ran as ONE gather + ONE launch (a layer each, rank 0), the rest in the pieces above\n",
                        g_dkv_one.load(std::memory_order_relaxed), (long long)kBiBand);
        }
    }
    auto dcu = [&](size_t j) {
        const int64_t ns = dap[j].s1 - dap[j].s0;
        if (j == 0) return praw(batch->cu_seqlens, RAD_I32, ns + 1);
        return bcol_at(m.bi_dcu, (int64_t)j, kBiBand + 1, 0, ns + 1, 1);
    };

    /* the step's head ops: as step() (per-token, over the step); tp8 S6: on a pipeline node as step() does them -- the
     * leader's lookup, X1 at W(0, t), the streams where the node's first layer is layer 0's successor of the lookup */
    media_route(m, batch);
    /* tp8 dev 1a: under pp_fe the follower's own lookup in place of X1; the leader keeps its own (a few dispatches, 0.06-0.10
     * ms a lane step on cards) so its xcol holds what it held before -- the prologue's expand reads it at every part -- and
     * only X1's send goes */
    /* tp8 dev D10 round 2 (m.pp_ids): the follower takes X1 -- with the step's token ids, into its batch's -- before its
     * lookup and its token_map read them; the leader sends it after its own lookup, under pp_fe too (the ids alone) */
    const bool x1_top = m.pp && m.pp_ids && m.pp_h == 0 && m.pp_rank == 1 && pp_seg_first(m);
    if (x1_top && !pp_cross(x, m, batch, RAD_PP_X1, 0, T)) return;
    if (((!m.pp || m.pp_rank == 0) && run_head && pp_lead_first(m)) || (m.pp_fe && m.pp_rank == 1 && pp_seg_first(m))) {
        RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
        m.mrows.step(x, batch, m.xcol, T);
        if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    }
    if (m.pp && m.pp_h == 0 && run_head && (!m.pp_fe || m.pp_ids) && !x1_top && pp_seg_first(m) &&
        !pp_cross(x, m, batch, RAD_PP_X1, 0, T))
        return;
    if (part == RAD_PP_PART_HEAD && m.pp_h == 0) return;   /* tp8 S5: at W(0, t) the head part ends with X1's send */
    if (!(m.pp && m.pp_rank == 1 && m.pp_h > 0)) {
        for (int64_t i = 0; i < c.hc; ++i)
            RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
        RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
        RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    }
    if (m.sone && g_sw_one.load(std::memory_order_relaxed) > 0) RAD_ISSUE_N(x, m.sone, T, brows(m.mw.sgate, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    /* tp8 dev D10 round 2: the token-id ops -- here, or after X1 at layer h on the follower at h > 0 under pp_ids */
    auto tok_ops = [&]() {
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap), brows(m.ctok, T));
    media_tokens(x, m, batch, T);
    ti_note(x, m);   /* lever TI */
    for (auto& L : m.layers) {
        /* tp8 S5 + S6: an Engram layer's ids (and its token window's update) in the part that issues the layer only */
        if (!L.e_ids || !pp_part_layer(m, part, L.l)) continue;
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
    };
    if (!pp_ids_late(m)) tok_ops();

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
    /* tp8 S6: the live rows [lo, hi) of the segments `cur` names (a crossing's residual / selection / keep rows) */
    const auto live_span = [&](int64_t* lo, int64_t* hi) {
        *lo = T;
        *hi = 0;
        for (const BiSeg& s : *cur) {
            if (s.n <= 0) continue;
            *lo = std::min(*lo, s.r0);
            *hi = std::max(*hi, s.r0 + s.n);
        }
    };
    /* tp8 loop P15 (RADIANCE_DS41_PIPE2_TBO, m.pipe2_tbo): `fn` over the rows [r0, r0 + n) of the layer's one prefill
     * segment only -- `cur` / `runs` name that piece while fn issues (the rows() / whole() / dense() above then launch over
     * it as they launch over the segment), then the layer's own again */
    static thread_local std::vector<BiSeg> tbo_sg;
    const auto with_rows = [&](int64_t r0, int64_t n, const auto& fn) {
        const std::vector<BiSeg>* keep = cur;
        BiSeg s = cur->empty() ? BiSeg{} : cur->front();
        s.r0 = r0;
        s.n = n;
        tbo_sg.assign(1, s);
        cur = &tbo_sg;
        set_runs();
        const bool sub0 = t_twin_sub;
        t_twin_sub = true;   /* tp8 loop P30t: a sub-range of a segment (SP quarter / piece) keeps GGUF */
        fn();
        t_twin_sub = sub0;
        cur = keep;
        set_runs();
    };
    /* P15: layer N's post / collapse / norm (+ its attention mix: lever M1) over the rows `cur` names -- the end-of-layer
     * issue below for the next layer (nx->pcn_a), as one function so bit 2 issues it over piece 0 there and piece 1 at N */
    const auto pcn_next = [&](const Layer& N, bool clear) {
        const int64_t fmin = N.pcn_mix ? mhc_fuse_min() : 0;
        if (clear) {
            t_mhc_a.clear();
            t_mhc_a_l = N.l;
        }
        rows(kBiNorm, [&](int64_t r0, int64_t n) {
            if (fmin > 0 && n >= fmin) {
                RAD_ISSUE_N(x, N.pcn_mix, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(N.attn_norm), R(m.xn, r0, n),
                            RAD_W(N.hc_a_fn), RAD_W(N.hc_a_scale), RAD_W(N.hc_a_base), R(m.pre_a, r0, n));
                t_mhc_a.push_back({ r0, n });
                mhc_note(x, n);
                return;
            }
            if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
            RAD_ISSUE_N(x, N.pcn_a, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                        R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(N.attn_norm), R(m.xn, r0, n));
        });
    };
    /* P15 bit 2: finish a deferral the layer it named cannot take in its order (a core / plan surprise; never by design):
     * lane 0 behind piece 1's MoE all-reduce, then that layer's pcn over piece 1 -- every row's bytes as ever, the overlap
     * lost; counted and said once */
    t_tbo = TboDefer{};
    const auto tbo_flush = [&] {
        if (t_tbo.l < 0) return;
        const TboDefer d = t_tbo;
        t_tbo.l = -1;
        rad_lane_join(x, 1, 0);
        with_rows(d.p0 + d.pa, d.pn - d.pa, [&] { pcn_next(m.layers[(size_t)d.l], false); });
        if (rad_rank(x) == 0 && g_tbo_flush.fetch_add(1, std::memory_order_relaxed) == 0)
            fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2_TBO (tp8 loop P15): VIRHE step %d: layer %d could not "
                            "take its piece-0 start in order -- finished at once (the same bytes, no overlap); counted in the "
                            "P13 stats line\n", batch->step, d.l);
    };
    /* tp8 loop P20 round 2 (RADIANCE_DS41_SPHC, m.sphc): this pass's sequence-parallel layers. The pass: one prefill
     * segment of the whole step, >= m.sphc rows, no media row, issued whole on this node (not a follower / leader segment),
     * the tracer off; every rank decides the same from the batch and the declare. Its shard: rows [sp0, sp0 + spn), q =
     * ceil(T / 4) a rank (the rows collectives' cut). A layer is SP (spv) where it is held in this part, no PSK cut can
     * reach it (may_cut: decided from the plan before its rows are), it is not a DSpark tap or the layer before this node's
     * crossing out, its MoE reduces its delta in ONE all-reduce the plugin may take (own_ar false), and the next layer this
     * part issues is either one whose collapse the previous layer issues (pcn_a: the gathers then make every row whole
     * again where needed) or SP itself (an Engram layer collapses in its own body from pre_ffn, which an SP layer writes
     * by shards). h_sh / xn_sh: the streams / xn hold only this rank's rows now (gathered before the next reader). */
    const auto may_cut = [&](int l) {
        return psk && (l >= m.psk_first || (l20_on && l == m.psk_l20 && !cr_all) || cr_any);
    };
    /* tp8 loop P25 (RADIANCE_DS41_SPHC_SEG, m.sp_seg): a follower SEGMENT pass of such a step too -- its plan is the whole
     * follower part's [h, t), which every segment of the step computes the same (segment boundaries are transparent to it),
     * so a boundary after an SP layer holds the streams sharded on both its sides */
    const int sp_seg = spseg_now(m);   /* tp8 loop P25s: the switch's level (<= the declare's) */
    const int sp_attn = spattn_now(m);
    const int64_t sp_thr = sphc_now(m);
    const bool sp_segp = sp_seg > 0 && t_seg && part == RAD_PP_PART_SEG && m.pp && m.pp_rank == 1;
    const bool sp_pass = m.sphc > 0 && T >= sp_thr && (T + 3) / 4 * 4 <= g.max_tok && sg.size() == 1 && !sg[0].dec &&
                         sg[0].r0 == 0 && sg[0].n == T && (sp_segp || (!t_seg && part != RAD_PP_PART_SEG)) &&
                         !(batch->n_mm_rows > 0 && batch->mm_rows) && !m.bt.on &&
                         !m.zero && !t_band;   /* tp8 loop P28: a band pass runs whole-row forms */
    const auto sp_in_part = [&](int i) { return sp_segp ? i >= pp_he(m) && i < m.pp_t : pp_part_layer(m, part, i); };
    /* tp8 loop P25b (RADIANCE_DS41_SPHC_SEG=3, a follower segment pass): a layer past m.psk_first whose cone keeps every row
     * of the pass's one prefill segment (psk_rows over C_l and the plan's `ahead`, as the cut below computes it) runs uncut,
     * so the plan may take it; the split layer (PSK-L20) and replay chunks stay as may_cut says. The cut below re-checks.
     * tp8 loop P25c (RADIANCE_DS41_SPHC_SEG=4): the same rule for the follower's LONE prefill pass (a cold prompt's chunk). */
    const bool sp_lonep = splone_now(m) && !t_seg && part != RAD_PP_PART_SEG && m.pp && m.pp_rank == 1;
    const auto sp_may_cut = [&](int l) {
        if (!may_cut(l)) return false;
        if (!((sp_segp && sp_seg >= 3) || sp_lonep) || cr_any || sg.size() != 1 || sg[0].dec || l < m.psk_first) return true;
        if (l20_on && l == m.psk_l20 && !cr_all) return true;
        return psk_rows(psk_cone(c.n_layer, c.window, l), psk_ahead(hint, 0), sg[0].n) < sg[0].n;
    };
    const int64_t spq = (T + 3) / 4;
    const int64_t sp0 =std::min<int64_t>((int64_t)rad_rank(x) * spq, T), spn = std::max<int64_t>(0, std::min(spq, T - sp0));
    static thread_local std::vector<uint8_t> spv;
    static thread_local std::vector<ds41sp::Layer> spl;
    spv.assign(m.layers.size(), 0);
    int sp_n = 0, sp_ga = 0, sp_lo = -1, sp_hi = -1, sp_en = 0, sp_at = 0, sp_ac = 0;   /* sp_at / sp_ac: P26's / P26b's layers */
    int sp_deep = 0;   /* tp8 loop P25c: SP layers past PSK's first */
    if (sp_pass) {
        spl.assign(m.layers.size(), ds41sp::Layer{});
        for (int i = 0; i < (int)m.layers.size(); ++i) {
            const Layer& L = m.layers[(size_t)i];
            ds41sp::Layer& d = spl[(size_t)i];
            d.issued = m.lr.held(L.l) && sp_in_part(i) && L.l == i;
            if (!d.issued) continue;
            const Layer* nx = next_layer(m, L.l);
            const int ni = nx ? (int)(nx - m.layers.data()) : -1;
            if (nx && ni == nx->l && m.lr.held(nx->l) && sp_in_part(ni)) {
                d.next = ni;
                d.next_pcn_a = nx->pcn_a != 0;
            }
            const MoeFP8& mo = L.moe;
            /* tp8 loop P20 round 3: the leader's layer before X1 where X1 can leave row-sharded; P25: a segment pass plans
             * the follower part's only crossing out (X2 after t - 1), not its segment's boundary */
            const int so = sp_segp ? (L.l == m.pp_t - 1 ? RAD_PP_X2 : -1) : pp_send_after(m, L.l);
            d.out_sharded = so == RAD_PP_X1 && m.pp_x1_sh != 0;
            d.self_ok = L.ar_attn && !L.tap && (so < 0 || d.out_sharded) && !sp_may_cut(L.l) && !mo.c.calib && !mo.op_moe_ar &&
                        !mo.op_shared_ar && !mo.op_combine && !mo.op_add && mo.issues_ar(T, T);
        }
        ds41sp::plan(spl, spv);
    }
    bool h_sh = false, xn_sh = false;
    /* tp8 loop P20: the SP line at the pass's every exit (a pipeline part returns at its crossing out, before the loop's end:
     * lq82 showed no line on either node) -- rank 0, the first and every 64th SP pass */
    const auto sp_report = [&] {
            if (!(sp_pass && rad_rank(x) == 0)) return;
            if (sp_segp) {   /* tp8 loop P25: a segment pass -- its own line (the first and every 512th), not the lone passes' */
                const long long sn = g_sp_seg_passes.fetch_add(1, std::memory_order_relaxed) + 1;
                const long long sl = g_sp_seg_layers.fetch_add(sp_n, std::memory_order_relaxed) + sp_n;
                if (sn == 1 || sn % 512 == 0)
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_SEG=%d (tp8 loop P25): %lld follower segment passes "
                                    "of one prefill segment ran %lld layer-passes sequence-parallel, %lld boundaries row-sharded "
                                    "(save + restore); this one step %d, %lld rows, layers [%d, %d), %d SP, %d gathers of the "
                                    "streams\n",
                            sp_seg, sn, sl, g_sp_seg_sh.load(std::memory_order_relaxed), batch->step, (long long)T, t_seg0,
                            t_seg1, sp_n, sp_ga);
                return;
            }
            static thread_local int t_sp_step = -1;
            const long long lone = g_sp_lone.fetch_add(1, std::memory_order_relaxed) + 1;
            if (sp_n > 0 && batch->step != t_sp_step) {
                t_sp_step = batch->step;
                g_sp_layers.fetch_add(sp_n, std::memory_order_relaxed);
                g_sp_rows.fetch_add(T, std::memory_order_relaxed);
                g_sp_gathers.fetch_add(sp_ga, std::memory_order_relaxed);
                g_sp_eng.fetch_add(sp_en, std::memory_order_relaxed);
                const long long at = g_sp_attn.fetch_add(sp_at, std::memory_order_relaxed) + sp_at;
                const long long ac = g_sp_comp.fetch_add(sp_ac, std::memory_order_relaxed) + sp_ac;
                const long long n = g_sp_passes.fetch_add(1, std::memory_order_relaxed) + 1;
                if (sp_lonep && sp_deep > 0) {   /* tp8 loop P25c: its own line, the first and every 64th such pass */
                    const long long cn = g_sp_cone_passes.fetch_add(1, std::memory_order_relaxed) + 1;
                    const long long cl = g_sp_cone_layers.fetch_add(sp_deep, std::memory_order_relaxed) + sp_deep;
                    if (cn == 1 || cn % 64 == 0)
                        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC_SEG=4 (tp8 loop P25c): %lld lone follower prefill "
                                        "passes ran %lld layer-passes past PSK's first sequence-parallel (their cones keep every row); "
                                        "this one step %d, %lld rows, layers %d..%d SP (%d, %d past PSK's first)\n",
                                cn, cl, batch->step, (long long)T, sp_lo, sp_hi, sp_n, sp_deep);
                }
                char p26[256] = "";
                if (m.sp_ag_qr)   /* tp8 loop P26: appended only with RADIANCE_DS41_SPHC_ATTN=1 / 2 */
                    std::snprintf(p26, sizeof p26, "; RADIANCE_DS41_SPHC_ATTN (tp8 loop P26): %lld layer-passes ran q_a / kv "
                                                   "over the quarter, this one %d", at, sp_at);
                if (m.sp_ag_kvs) {   /* P26b */
                    const size_t u = std::strlen(p26);
                    std::snprintf(p26 + u, sizeof p26 - u, "; %lld source layer-passes the compressor over the quarter "
                                                           "(P26b), this one %d", ac, sp_ac);
                }
                if (n == 1 || n % 64 == 0)
                    fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_SPHC (tp8 loop P20): %lld prefill passes ran %lld "
                                    "layer-passes sequence-parallel (%lld rows, the streams gathered %lld times, %lld Engram "
                                    "layer-passes over the quarter) of %lld lone prefill passes of >= %lld rows, %lld X1s row-sharded; "
                                    "this one %lld rows, layers %d..%d SP (%d), %d gathers of the streams, part %d%s\n",
                            n, g_sp_layers.load(std::memory_order_relaxed), g_sp_rows.load(std::memory_order_relaxed),
                            g_sp_gathers.load(std::memory_order_relaxed), g_sp_eng.load(std::memory_order_relaxed), lone,
                            (long long)sp_thr, g_sp_x1.load(std::memory_order_relaxed), (long long)T, sp_lo, sp_hi, sp_n, sp_ga,
                            part, p26);
            }
    };
    struct SpEnd {
        const decltype(sp_report)& f;
        ~SpEnd() { f(); }
    } sp_end{ sp_report };
    const auto sp_gather_h = [&] {   /* the streams whole again on every rank; the next mix over every row */
        RAD_ISSUE_N(x, m.sp_ag_h, T * c.hc * D, brows(m.h0, T), RAD_NONE);
        t_mhc_a.clear();
        t_mhc_a_l = -1;
        h_sh = false;
        ++sp_ga;
    };
    const auto sp_gather_x = [&] {   /* xn whole again on every rank (the attention's inputs, the compressor) */
        if (!xn_sh) return;
        RAD_ISSUE_N(x, m.sp_ag_x, T * D, brows(m.xn, T), RAD_NONE);
        xn_sh = false;
    };
    for (auto& L : m.layers) {
        /* tp8 loop P30t: the classes this layer keeps on GGUF (t_tw_off): every class under P13 / P15's pieces; q_a + kv where
         * P26 may issue them over an SP quarter (its static conditions at this switch level); the Engram kv under the SP Engram */
        t_tw_off = !m.twin_mask ? 0
                 : (m.pipe2 > 0 ? kTwAll : 0) | (m.sphc_eng ? kTwEng : 0) |
                   (bi_pmm(g_sw_bi.load(std::memory_order_relaxed)) ? kTwShx : 0) |
                   (m.sp_ag_qr && m.sp_ag_kv && sp_attn >= 1 && !L.norm_a && !L.src && !L.i_score ? kTwQkv : 0);
        /* tp8 S3 / S6: another node's layer issues nothing, but its cut is computed below (the rows a crossing at the
         * next held layer carries are the previous layer's) */
        /* tp8 S5 + S6: a lane step's leader part issues its own layers; the other part's are as another node's (their
         * cut is computed, nothing issued) */
        /* tp8 loop P28 round 2c: a band pass's leader issues the band layers too */
        const bool band_l = t_band && m.pp_rank == 0 && m.band_k > 0 && m.band_decl[(size_t)L.l];
        const bool held = (m.lr.held(L.l) || band_l) && pp_part_layer(m, part, (int)(&L - m.layers.data()));
        const int l = held ? L.l : (int)(&L - m.layers.data());
        const int64_t r = held ? L.r : c.ratio[(size_t)l];
        /* tp8 S6: the receiver's first layer after the cut starts from the crossing: its live rows are the previous
         * layer's segments (`cur`, not yet this layer's), over which one process issues the boundary op at the end of the
         * previous layer -- issued the same here (rows(kBiNorm) as below), after this node's copy of the source's
         * entries (stored unless the pass is all replay chunks, which store none: lever PSK-CR). */
        if (const int ring = held ? pp_recv_at(m, l) : -1; ring >= 0) {
            int64_t lo = 0, hi = 0;
            live_span(&lo, &hi);
            /* tp8 loop P20 round 4: X1 into an SP layer lands row-sharded (each rank its quarter of the residual parts) */
            const bool x1r_sh = ring == RAD_PP_X1 && m.pp && m.pp_rank == 1 && m.pp_x1_sh && held &&
                                spv[(size_t)(&L - m.layers.data())] != 0 && lo == 0 && hi == T;
            /* tp8 loop P25: a segment boundary after an SP layer (the plan's layer before this one) holds the streams
             * row-sharded: the boundary op over this rank's quarter, as the whole pass issues it at that layer's end; 2 restores
             * only the quarter */
            const int li = (int)(&L - m.layers.data());
            const bool seg_hsh = ring >= kSegRecv && ring < kSegSend && sp_segp && li > 0 && spv[(size_t)li - 1] != 0;
            if (seg_hsh && (lo != 0 || hi != T)) {
                rad_step_fail(x, "deepseek41: RADIANCE_DS41_SPHC_SEG: an SP segment boundary's rows are not the pass's (a plan "
                                 "defect)");
                return;
            }
            if (!pp_cross(x, m, batch, ring, lo, hi, x1r_sh || (seg_hsh && sp_seg >= 2))) return;
            if (x1r_sh && rad_rank(x) == 0) g_sp_x1.fetch_add(1, std::memory_order_relaxed);
            if (!pp_copy_issue(x, m, batch, ring, !cr_all)) return;
            if (ring == RAD_PP_X1 && !band_store_issue(x, m, batch)) return;   /* tp8 loop P28 round 2c */
            if (ring == RAD_PP_X1 && pp_ids_late(m)) tok_ops();   /* tp8 dev D10 round 2: X1's ids are in the batch now */
            const bool sp_rx = held && (spv[(size_t)li] != 0 || seg_hsh);
            if (sp_rx && L.pcn_a) {   /* tp8 loop P20: the crossing's rows whole on every rank, this rank's quarter collapsed */
                with_rows(sp0, spn, [&] { pcn_next(L, true); });
                h_sh = xn_sh = true;
            } else if (sp_rx) {
                with_rows(sp0, spn, [&] {
                    whole([&](int64_t r0, int64_t n) {
                        RAD_ISSUE_N(x, pp_post_op(m, ring), n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n),
                                    R(m.comb, r0, n), R(m.h0, r0, n));
                    });
                });
                h_sh = true;
            } else if (L.pcn_a) {
                const int64_t fmin = L.pcn_mix ? mhc_fuse_min() : 0;
                t_mhc_a.clear();
                t_mhc_a_l = L.l;
                rows(kBiNorm, [&](int64_t r0, int64_t n) {
                    if (fmin > 0 && n >= fmin) {
                        RAD_ISSUE_N(x, L.pcn_mix, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                                    R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(L.attn_norm), R(m.xn, r0, n),
                                    RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale), RAD_W(L.hc_a_base), R(m.pre_a, r0, n));
                        t_mhc_a.push_back({ r0, n });
                        mhc_note(x, n);
                        return;
                    }
                    if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
                    RAD_ISSUE_N(x, L.pcn_a, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                                R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(L.attn_norm), R(m.xn, r0, n));
                });
            } else {
                whole([&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, pp_post_op(m, ring), n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                                R(m.h0, r0, n));
                });
            }
        }
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
                if (cr_seg && held && rad_rank(x) == 0) {
                    g_cr_rows_run.fetch_add(std::max<int64_t>(n, 0), std::memory_order_relaxed);
                    if (l < m.psk_first && !l20 && n < s0.n)
                        g_cr_rows_skip.fetch_add(s0.n - std::max<int64_t>(n, 0), std::memory_order_relaxed);
                }
                if (n >= s0.n) { psk_sg.push_back(s0); continue; }
                if (held && rad_rank(x) == 0) {
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
        if (cut && held && spv[(size_t)(&L - m.layers.data())] != 0) {   /* tp8 loop P25b: the plan took a layer PSK cuts */
            rad_step_fail(x, sp_lonep ? "deepseek41: RADIANCE_DS41_SPHC_SEG=4 (P25c): a lone pass's SP layer is cut by lever PSK "
                                        "(a plan defect)"
                                      : "deepseek41: RADIANCE_DS41_SPHC_SEG=3: an SP layer is cut by lever PSK (a plan defect)");
            return;
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
        if (l20 && cut && held && rad_rank(x) == 0) {
            g_psk_l20_passes.fetch_add(1, std::memory_order_relaxed);
            if (l20_none) g_psk_l20_empty.fetch_add(1, std::memory_order_relaxed);
        }
        /* design 2: a layer no segment runs a row of is skipped whole (and so is every layer after it: the cones shrink
         * with l) -- no op, no collective; the runs stay empty for the final norm */
        if (cut && cur->empty() && !l20) {
            if (held && rad_rank(x) == 0) g_psk_skips.fetch_add(1, std::memory_order_relaxed);
            /* tp8 S6: the sender's last layer before the cut sends its crossing whatever rows it ran (none here: the
             * receiver's layers run none either, and the copied source's entries cross all the same) */
            if (const int ring = held ? pp_send_after(m, l) : -1; ring >= 0) {
                if (!pp_cross(x, m, batch, ring, T, 0)) return;
                if (ring != RAD_PP_X1 || part == RAD_PP_PART_HEAD) {   /* tp8 S5: the head part ends with X1's send (tp8 dev M4: a
                                                                       * segment with its boundary's save) */
                    bt_flush(x, m, batch);
                    return;
                }
            }
            continue;
        }
        if (!held) continue;   /* tp8 S3 / S6: another node's layer: its cut only */
        const std::vector<BiSeg>& sg = *cur;   /* the layer's segments, below */
        /* tp8 loop P20: this layer SP -- or, where an SP layer left the streams sharded and this one reads them, the streams
         * gathered first (not the split layer with no cone row: it reads only xn, gathered below) */
        const bool sp_l = spv[(size_t)(&L - m.layers.data())] != 0;
        if (h_sh && !sp_l && !(l20 && cut && cur->empty())) sp_gather_h();
        if (sp_l) {
            ++sp_n;
            if (sp_lo < 0) sp_lo = l;
            sp_hi = l;
            if (l >= m.psk_first) ++sp_deep;   /* tp8 loop P25b / P25c: a layer PSK could cut, planned SP (uncut) */
        }
        /* tp8 loop P13b (RADIANCE_DS41_PIPE2_ATTN): the attention all-reduce of an uncut layer of a lone prefill pass in
         * the pieces [0, pa) [pa, T) (pa as the MoE's cut below), piece 0's on lane 1 beside piece 1's output projection;
         * decided from the batch and the declare alone, so every rank issues the same collectives */
        /* P13c (RADIANCE_DS41_PIPE2_CONE): a cut layer's one prefill segment [p0, p0 + pn) as well (not the split layer);
         * uncut, the segment is the step's [0, T) */
        const bool p2_layer = m.pipe2 > 0 && (!cut || (m.pipe2_cone && !l20)) && sg.size() == 1 && !sg[0].dec &&
                              sg[0].n >= m.pipe2 && (cut || (sg[0].r0 == 0 && sg[0].n == T));
        const int64_t p0 = p2_layer ? sg[0].r0 : 0, pn = p2_layer ? sg[0].n : T;
        int64_t pa = 0;
        if (p2_layer && m.pipe2_attn && L.ar_attn && L.ar_attn_ov) {
            pa = (pn / 2 + kPipe2Align - 1) / kPipe2Align * kPipe2Align;
            if (pa < kPipe2Piece || pn - pa < kPipe2Piece) pa = 0;
        }
        const auto ar_a_piece0 = [&] {   /* lane 1 behind everything lane 0 issued (piece 0's rows of attn_out written) */
            rad_lane_join(x, 0, 1);
            rad_lane(x, 1);
            RAD_ISSUE_N(x, L.ar_attn_ov, pa * D, R(m.attn_out, p0, pa), RAD_NONE);
            rad_lane(x, 0);
        };
        /* tp8 loop P13's MoE cut of this layer (the MoE section below decides it the same way; 0 = the MoE runs whole) */
        const auto moe_pc = [&]() -> int64_t {
            if (!p2_layer) return 0;
            const int64_t q = (pn / 2 + kPipe2Align - 1) / kPipe2Align * kPipe2Align;
            const MoeFP8& mo = L.moe;
            if (q < kPipe2Piece || pn - q < kPipe2Piece || mo.c.calib || mo.op_moe_ar || mo.op_shared_ar || mo.op_combine ||
                mo.op_add || !mo.op_ar_ov || !mo.issues_ar(T, pn) || !mo.issues_ar(T, q) || !mo.issues_ar(T, pn - q))
                return 0;
            return q;
        };
        /* tp8 loop P15 bit 1 (RADIANCE_DS41_PIPE2_TBO): this layer's attention + MoE pieces are the same rows ([p0, p0 + pa),
         * the rest), so piece 1's attention all-reduce can go beside MoE pass 0; decided from the batch and the declare */
        const bool tbo_a = (m.pipe2_tbo & 1) && !(m.pipe2_tbo & 4) && pa > 0 && L.pcn_f && !m.bt.on && moe_pc() == pa;
        /* P15 bit 2: a deferral for another layer, or one this layer cannot take in its order (never by design) */
        if (t_tbo.l >= 0 && (t_tbo.l != l || cut || !p2_layer || p0 != t_tbo.p0 || pn != t_tbo.pn)) tbo_flush();
        if (L.e_ids && sp_l) {
            /* tp8 loop P20: an SP Engram layer -- the gate (+ its copy back) over this rank's quarter of the streams; with
             * m.sphc_eng the lookup, the copy and the kv GEMM over the quarter too (each a row's own: the rows the gate
             * reads), lever E1's dedup otherwise as today over every row */
            const int64_t H = c.ple_heads;
            m.edd_ls.clear();
            const bool e_gg = gguf_dense(m, L.e_kv);
            dense(e_gg, false, [&](int64_t r0, int64_t n) { m.edd_ls.push_back({ r0, n }); });
            const bool edd_ok = !m.sphc_eng && engdd_issue(x, m, L, batch, T, m.edd_ls);
            if (!edd_ok && m.sphc_eng && spn > 0) {
                const RadOperand ids = brow_slice(L.ids, sp0, spn, H);
                if (m.ple_group > 0)
                    RAD_ISSUE_N(x, L.e_emb, spn * H, ids, RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                                m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, spn));
                else
                    RAD_ISSUE_N(x, L.e_emb, spn * H, ids, RAD_W(m.ple_table), brows(m.emb, spn));
                RAD_ISSUE_N(x, L.e_copy, spn, brows(m.emb, spn), R(m.embd, sp0, spn));
                with_rows(sp0, spn, [&] {
                    dense(e_gg, false, [&](int64_t r0, int64_t n) { e2_gemm(x, m, L, n, R(m.embd, r0, n), R(m.ekv, r0, n)); });
                });
                ++sp_en;
            } else if (!edd_ok) {
                if (m.ple_group > 0)
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                                m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, T));
                else
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
                RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
                dense(e_gg, false, [&](int64_t r0, int64_t n) { e2_gemm(x, m, L, n, R(m.embd, r0, n), R(m.ekv, r0, n)); });
            }
            if (spn > 0) {
                RAD_ISSUE_N(x, L.e_gate_op, spn, R(m.h0, sp0, spn), C(m.ekv, 0, c.hc * D, sp0, spn),
                            C(m.ekv, c.hc * D, D, sp0, spn), RAD_W(L.e_gate), R(m.h1, sp0, spn));
                RAD_ISSUE_N(x, L.e_back, spn, R(m.h1, sp0, spn), R(m.h0, sp0, spn));
            }
            h_sh = true;
        } else if (L.e_ids) {
            const int64_t H = c.ple_heads;
            /* lever E1: the lookup, copy and GEMM over the unique id rows of the launches below that are wider than the
             * band (the decode segment's pieces and a short prefill segment keep their own launch) */
            m.edd_ls.clear();
            const bool e_gg = gguf_dense(m, L.e_kv);   /* lever WMK: a mixed run's Engram rows in one launch (dense()) */
            dense(e_gg, false, [&](int64_t r0, int64_t n) { m.edd_ls.push_back({ r0, n }); });
            if (!engdd_issue(x, m, L, batch, T, m.edd_ls)) {
                if (m.ple_group > 0)
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_codes), RAD_W(m.ple_scale),
                                m.ple_tab ? RAD_W(m.ple_tab) : RAD_NONE, brows(m.emb, T));
                else
                    RAD_ISSUE_N(x, L.e_emb, T * H, brows(L.ids, T), RAD_W(m.ple_table), brows(m.emb, T));
                RAD_ISSUE_N(x, L.e_copy, T, brows(m.emb, T), brows(m.embd, T));
                /* the Engram GEMM (gemm_nt_ggml, banded) by segment */
                dense(e_gg, false, [&](int64_t r0, int64_t n) { e2_gemm(x, m, L, n, R(m.embd, r0, n), R(m.ekv, r0, n)); });
            }
            media_engram(x, m, L, batch, T);
            RAD_ISSUE_N(x, L.e_gate_op, T, brows(m.h0, T), bcol(m.ekv, 0, c.hc * D, T),
                        bcol(m.ekv, c.hc * D, D, T), RAD_W(L.e_gate), brows(m.h1, T));
            RAD_ISSUE_N(x, L.e_back, T, brows(m.h1, T), brows(m.h0, T));
        }
        bt_pt(x, m, m.h0, T, "h0", l);
        if (L.tap) RAD_ISSUE_N(x, L.tap, T, brows(m.h0, T), bcol(m.dsp.src, L.tap_col * D, D, T));
        const bool late = !m.zero;
        const auto mix_a = [&] {
            /* lever M1: the rows the previous layer's fused pcn_a launches did not mix */
            if (t_mhc_a_l != l) t_mhc_a.clear();
            whole([&](int64_t r0w, int64_t nw) {
                mhc_rest(t_mhc_a, r0w, nw, [&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, L.mix_a, n, R(m.h0, r0, n), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                                RAD_W(L.hc_a_base), R(m.pre_a, r0, n), R(m.post, r0, n), R(m.comb, r0, n));
                });
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
        if (L.cn_a && sp_l) {   /* tp8 loop P20: this rank's quarter; xn gathered before the attention reads it */
            with_rows(sp0, spn, [&] { rows(kBiNorm, cn_a); });
            h_sh = xn_sh = true;
        } else if (L.cn_a) {
            if (l20) rows_step(kBiNorm, cn_a);
            else rows(kBiNorm, cn_a);
        } else if (L.col_a) {
            if (l20) whole_step(col_a);
            else whole(col_a);
        }
        /* tp8 loop P20: the attention mix over this rank's quarter at an SP layer */
        const auto mix_a_sp = [&] {
            if (sp_l) with_rows(sp0, spn, mix_a);
            else mix_a();
        };
        if (!late) mix_a_sp();
        if (m.zero) {
            whole([&](int64_t r0, int64_t n) { RAD_ISSUE_N(x, L.attn_fill, n, R(m.attn_out, r0, n)); });
        } else {
            /* tp8 loop P26 (RADIANCE_DS41_SPHC_ATTN=1, m.sp_ag_qr): an SP layer whose xn only q_a and kv read (no compressor
             * chain: L.src; no indexer projection: L.i_score) runs both over this rank's quarter and gathers their outputs
             * instead of xn; everything below reads qr / kv over every row as today (a1_cheap skips the two GEMMs) */
            const bool cheap0 = bi_cheap(g_sw_bi.load(std::memory_order_relaxed));
            const bool p26 = sp_l && xn_sh && m.sp_ag_qr && m.sp_ag_kv && sp_attn >= 1 && cheap0 && !L.norm_a && !L.src &&
                             !L.i_score &&
                             L.q_a && L.kv_g && gguf_dense(m, L.wq_a) && gguf_dense(m, L.wkv) && !p2_layer && t_tbo.l != l;
            if (p26) {
                with_rows(sp0, spn, [&] {
                    dense(true, true, [&](int64_t r0, int64_t n) {
                        lin_issue(x, m, L.q_a, n, R(m.xn, r0, n), L.wq_a, R(m.qr, r0, n));
                    });
                    dense(true, true, [&](int64_t r0, int64_t n) {
                        lin_issue(x, m, L.kv_g, n, R(m.xn, r0, n), L.wkv, R(m.kv, r0, n));
                    });
                });
                RAD_ISSUE_N(x, m.sp_ag_qr, T * c.q_lora, brows(m.qr, T), RAD_NONE);
                RAD_ISSUE_N(x, m.sp_ag_kv, T * c.head_dim, brows(m.kv, T), RAD_NONE);
                xn_sh = false;   /* xn past this rank's quarter is stale; nothing reads it before the next layer writes it */
                ++sp_at;
            }
            sp_gather_x();   /* tp8 loop P20: xn whole on every rank before its first reader */
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
            const bool cheap = cheap0;   /* tp8 loop P26: read once above */
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
            /* C: op by op across the segments; the never-flipping per-row ops once over the step (over `cur` / `runs`: P15
             * issues it over one piece at a time) */
            const auto a1_cheap = [&] {
                /* lever WMK (dense(), ds41_wmk 1): the dense GGUF linears once over a mixed run; off: by segment, as before */
                const bool qkv_gg = gguf_dense(m, L.wq_a) && gguf_dense(m, L.wkv);
                if (!p26) {   /* tp8 loop P26: issued over the quarter above */
                dense(qkv_gg, true, [&](int64_t r0, int64_t n) {   /* lever 1U-GQ: q_a + kv_g as one launch where the switch takes them */
                    if (!tw_take(m, L.q_a) && gq_qkv_on(L, n))   /* tp8 loop P30t: a twin prompt segment: the twins */
                        gq_qkv_issue(x, m, L, n, R(m.xn, r0, n), R(m.qr, r0, n), R(m.kv, r0, n));
                    else
                        lin_issue(x, m, L.q_a, n, R(m.xn, r0, n), L.wq_a, R(m.qr, r0, n));
                });
                dense(qkv_gg, true, [&](int64_t r0, int64_t n) {
                    if (tw_take(m, L.q_a) || !gq_qkv_on(L, n)) lin_issue(x, m, L.kv_g, n, R(m.xn, r0, n), L.wkv, R(m.kv, r0, n));
                });
                }
                if (L.i_score)
                    dense(gguf_dense(m, L.iproj), true, [&](int64_t r0, int64_t n) {
                        lin_issue(x, m, L.i_w, n, R(m.xn, r0, n), L.iproj, R(m.iw, r0, n));
                    });
                if (qfold)   /* rmsnorm_gemm_nt_ggml: rmsnorm's row arithmetic is the row's at any width (kR4dSumLanes) */
                    dense(gguf_dense(m, L.wq_b), true, [&](int64_t r0, int64_t n) {
                        if (tw_take(m, L.q_b)) {   /* tp8 loop P30t: the norm alone (the fold's qrn bytes), then q_b's twin */
                            RAD_ISSUE_N(x, L.q_an, n, R(m.qr, r0, n), RAD_W(L.q_norm), R(m.qrn, r0, n));
                            lin_issue(x, m, L.q_b, n, R(m.qrn, r0, n), L.wq_b, R(m.q, r0, n));
                            return;
                        }
                        RAD_ISSUE_N(x, L.q_bf, n, R(m.qr, r0, n), RAD_W(L.wq_b), RAD_W(L.q_norm), R(m.qrn, r0, n), R(m.q, r0, n));
                    });
                else
                    whole([&](int64_t r0, int64_t n) {
                        RAD_ISSUE_N(x, L.q_an, n, R(m.qr, r0, n), RAD_W(L.q_norm), R(m.qrn, r0, n));
                    });
                if (kvfold || L.kv_f)
                    for (const BiSeg& s : *cur) kvn_seg(s);   /* = the layer's segments (sg), or P15's piece */
                else
                    whole([&](int64_t r0, int64_t n) {
                        RAD_ISSUE_N(x, L.kv_n, n, R(m.kv, r0, n), RAD_W(L.kv_norm), R(m.kvn, r0, n));
                    });
                if (L.i_score)
                    dense(gguf_dense(m, L.iq_w), true, [&](int64_t r0, int64_t n) {
                        lin_issue(x, m, L.i_q, n, R(m.qrn, r0, n), L.iq_w, R(m.iq, r0, n));
                    });
                if (!qfold)
                    dense(gguf_dense(m, L.wq_b), true, [&](int64_t r0, int64_t n) {
                        lin_issue(x, m, L.q_b, n, R(m.qrn, r0, n), L.wq_b, R(m.q, r0, n));
                    });
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
            };
            if (cheap && t_tbo.l == l) {
                /* tp8 loop P15 bit 2: the previous layer issued this layer's pcn over piece 0 beside its piece-1 MoE
                 * all-reduce (lane 1, in flight): piece 0's attention inputs beside it too, then lane 0 behind it, this
                 * layer's pcn over piece 1 and piece 1's inputs; the compressor chain, the indexer and the attention
                 * below over the step as ever */
                const TboDefer d = t_tbo;
                t_tbo.l = -1;
                with_rows(d.p0, d.pa, a1_cheap);
                rad_lane_join(x, 1, 0);
                with_rows(d.p0 + d.pa, d.pn - d.pa, [&] { pcn_next(L, false); });
                with_rows(d.p0 + d.pa, d.pn - d.pa, a1_cheap);
                if (rad_rank(x) == 0) g_tbo_xl.fetch_add(1, std::memory_order_relaxed);
            } else if (cheap) {
                tbo_flush();   /* (nothing pending: a no-op) */
                a1_cheap();
            } else {
                tbo_flush();
            }
            for (const BiSeg& s : sg) {
                if (cheap) break;
                const int64_t r0 = s.r0, n = s.n;
                const TwinScope tws(!s.dec, n);   /* tp8 loop P30t */
                if (!tw_take(m, L.q_a) && gq_qkv_on(L, n)) {   /* lever 1U-GQ */
                    gq_qkv_issue(x, m, L, n, R(m.xn, r0, n), R(m.qr, r0, n), R(m.kv, r0, n));
                } else {
                    lin_issue(x, m, L.q_a, n, R(m.xn, r0, n), L.wq_a, R(m.qr, r0, n));
                    lin_issue(x, m, L.kv_g, n, R(m.xn, r0, n), L.wkv, R(m.kv, r0, n));
                }
                if (L.i_score) lin_issue(x, m, L.i_w, n, R(m.xn, r0, n), L.iproj, R(m.iw, r0, n));
                const bool qf = qfold && !tw_take(m, L.q_b);   /* tp8 loop P30t: a twin q_b takes the norm alone */
                if (qf)
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
                if (!qf) lin_issue(x, m, L.q_b, n, R(m.qrn, r0, n), L.wq_b, R(m.q, r0, n));
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
            /* tp8 loop P26b (RADIANCE_DS41_SPHC_ATTN=2, m.sp_ag_kvs): at an SP kv-source layer the compressor's kv | gate
             * GEMM (unsharded gemm_nt_f32: every form the stock per-output fmaf chains, so a row's bytes at any M) over this
             * rank's quarter of the gathered xn, then its f32 output gathered raw before the chain reads every row */
            if (chain && ctail && sp_l && m.sp_ag_kvs && sp_attn >= 2) {   /* P25s: the switch's level */
                if (spn > 0) RAD_ISSUE_N(x, L.c_gemm, spn, R(m.xn, sp0, spn), RAD_W(L.comp_w), R(m.kvs, sp0, spn));
                RAD_ISSUE_N(x, m.sp_ag_kvs, T * 2 * hd, brows(m.kvs, T), RAD_NONE);
                ++sp_ac;
            } else if (chain)
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
                /* tp8 S6: the candidate layer unsplit where its keep words cross (Model::pp_unsplit_cand) */
                const bool split = m.isplit && m.sel_gather && !(m.pp_unsplit_cand && l == c.cand);
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
                    Ds41IdxRuns ru = ds41_idx_runs(s.n, m.irows, live, m.aw, m.ar, split && !s.dec, m.split_min_live,
                                                   split && !s.dec && idx_balance(), (int64_t)L.i_runs.size());
                    /* tp8 loop F58: the decode segment above the band in runs of <= kBiBand rows (the decode kernel) */
                    if (s.dec && !dap.empty() && ru.R > kBiBand) {
                        const Ds41IdxRuns rd = ds41_idx_runs(s.n, kBiBand, live, m.aw, m.ar, false);
                        if (rd.n <= (int64_t)L.i_runs.size()) {
                            ru = rd;
                            if (rad_rank(x) == 0) g_dband_runs.fetch_add(rd.n, std::memory_order_relaxed);
                        }
                    }
                    for (int64_t p = ru.p0; p < ru.p1; ++p) {
                        const int64_t q0 = p * ru.R, qn = std::min(ru.R, s.n - q0);
                        if (s.r0 + q0 + qn <= cut0) continue;
                        ixni_note(x, batch->positions_h ? batch->positions_h + s.r0 + q0 : nullptr, qn, ru.R, live, r);
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
            /* tp8 loop F58: one segment (or decode band piece) with its cu view `cuv` */
            const auto attn_one = [&](const BiSeg& s, const RadOperand cuv) {
                const int64_t ns = s.s1 - s.s0;
                const RadOperand wbt = praw2(wb->block_table + s.s0 * wb->block_table_pitch, RAD_I32, ns, wb->block_table_pitch);
                const RadOperand wused = praw(wb->seqused + s.s0, RAD_I32, ns);
                if (r > 0) {
                    const int src = c.src_of[(size_t)l];
                    const auto* cbb = kv_batch(batch, m.comp[r]);
                    const RadOperand cbt = praw2(cbb->block_table + s.s0 * cbb->block_table_pitch, RAD_I32, ns,
                                                 cbb->block_table_pitch);
                    /* lever DKV: a prefill segment's plan rows are its uncut rows (sg_step's segment of the same
                     * sequences, when this one is its last rows) */
                    if (m.lu || m.selkv) {
                        int64_t hr0 = s.r0, hn = s.n;
                        if (!s.dec)
                            for (const BiSeg& u : sg_step)
                                if (!u.dec && u.s0 == s.s0 && u.s1 == s.s1 && u.r0 <= s.r0 && u.r0 + u.n == s.r0 + s.n) {
                                    hr0 = u.r0;
                                    hn = u.n;
                                    break;
                                }
                        std::vector<RadOperand> da = { R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused,
                                                       kv_cache(m.comp[r], src), cbt, R(m.sel, s.r0, s.n),
                                                       P(batch->positions, s.r0, s.n), cuv, RAD_W(L.sinks),
                                                       R(m.o, s.r0, s.n) };
                        if (ofold) da.push_back(acs);
                        const int64_t lp = batch->positions_h && s.n > 0 ? (int64_t)batch->positions_h[s.r0 + s.n - 1] : -1;
                        if (dkv_attn(x, m, L, ofold, s.dec, s.s0, s.s1, s.n, hr0, hn, std::move(da), lp)) return;
                    }
                    if (ofold)
                        RAD_ISSUE_N(x, ah, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, kv_cache(m.comp[r], src),
                                    cbt, R(m.sel, s.r0, s.n), P(batch->positions, s.r0, s.n), cuv, RAD_W(L.sinks),
                                    R(m.o, s.r0, s.n), acs);
                    else
                        RAD_ISSUE_N(x, L.attn, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused,
                                    kv_cache(m.comp[r], src), cbt, R(m.sel, s.r0, s.n), P(batch->positions, s.r0, s.n),
                                    cuv, RAD_W(L.sinks), R(m.o, s.r0, s.n));
                } else if (ofold) {
                    RAD_ISSUE_N(x, ah, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, RAD_NONE, RAD_NONE, RAD_NONE,
                                P(batch->positions, s.r0, s.n), cuv, RAD_W(L.sinks), R(m.o, s.r0, s.n), acs);
                } else {
                    RAD_ISSUE_N(x, L.attn, s.n, R(m.q, s.r0, s.n), kv_cache(m.win, l), wbt, wused, RAD_NONE, RAD_NONE,
                                RAD_NONE, P(batch->positions, s.r0, s.n), cuv, RAD_W(L.sinks), R(m.o, s.r0, s.n));
                }
            };
            for (const BiSeg& s : sg) {
                if (!(s.dec && !dap.empty())) {
                    attn_one(s, segcu(s));
                    continue;
                }
                /* tp8 loop D25 step 3 (RADIANCE_DS41_DKV_GPIECE=2): a decode segment of <= selkv_cap rows at a layer lever DKV-G
                 * serves (a compressed layer: dkv_attn's gathered branch, whose ops carry libr4d's band 64) runs ONE gather + ONE
                 * attention over all its rows -- the scalar decode kernels, each row's alone bytes as in a piece; any other layer
                 * (window-only, or a form DKV-G does not serve) keeps F58's pieces */
                if (m.selkv_one && r > 0 && m.lnv && m.selkv && L.lgath && (ofold ? L.attn_rtg : L.attn_g) && s.r0 == 0 &&
                    s.n <= m.selkv_cap) {
                    attn_one(s, segcu(s));
                    if (rad_rank(x) == 0) g_dkv_one.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                /* tp8 loop F58: the decode segment above the band in its pieces, each in the decode kernels */
                for (size_t j = 0; j < dap.size(); ++j) attn_one(BiSeg{ dap[j].r0, dap[j].n, dap[j].s0, dap[j].s1, true, -1 }, dcu(j));
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
            bool oa_gg = true;
            for (int j = 0; j < m.gr; ++j) oa_gg = oa_gg && gguf_dense(m, L.wo_a[(size_t)j]);
            const auto oa_fn = [&](int64_t r0, int64_t n) {
                /* tp8 loop P30t: a twin prompt segment's groups write Model::oa_g, which o_b's twin reads (ob_fn: lin_issue) */
                const bool otw = L.oa_tw && tw_take(m, L.o_a[0]);
                if (otw) {
                    for (int j = 0; j < m.gr; ++j)
                        lin_issue(x, m, L.o_a[(size_t)j], n, C(m.o, j * gd, gd, r0, n), L.wo_a[(size_t)j],
                                  R(m.oa_g[(size_t)j], r0, n));
                    return;
                }
                if (gq_oa_on(L, n)) {   /* lever 1U-GQ: the two groups as one launch */
                    gq_oa_issue(x, m, L, n, C(m.o, 0, gd, r0, n), C(m.o, gd, gd, r0, n), C(m.oa, 0, c.o_rank, r0, n),
                                C(m.oa, c.o_rank, c.o_rank, r0, n));
                    return;
                }
                for (int j = 0; j < m.gr; ++j) {
                    const rad_buf od = (size_t)j < L.oa_dst.size() ? L.oa_dst[(size_t)j] : 0;   /* tp8 loop P30f */
                    lin_issue(x, m, L.o_a[(size_t)j], n, C(m.o, j * gd, gd, r0, n), L.wo_a[(size_t)j],
                              od ? R(od, r0, n) : C(m.oa, j * c.o_rank, c.o_rank, r0, n));
                }
            };
            const auto ob_fn = [&](int64_t r0, int64_t n) {
                lin_issue(x, m, L.o_b, n, R(m.oa, r0, n), L.wo_b, R(m.attn_out, r0, n));
            };
            if (pa > 0) {
                /* tp8 loop P13b: piece 0's projections, the mix (over the step, as today: after the groups), piece 0's
                 * o_b, its all-reduce on lane 1, then piece 1's projections on lane 0 beside it (no collective in them; a
                 * lone prefill segment is one launch of each in dense(), so these are its launches over the two row
                 * ranges) */
                oa_fn(p0, pa);
                if (late) mix_a();
                ob_fn(p0, pa);
                ar_a_piece0();
                oa_fn(p0 + pa, pn - pa);
                ob_fn(p0 + pa, pn - pa);
                bt_pt(x, m, oa_split(L) ? 0 : m.oa, T, "oa", l);
            } else {
                dense(oa_gg, false, oa_fn);
                bt_pt(x, m, oa_split(L) ? 0 : m.oa, T, "oa", l);
                if (late) mix_a_sp();
                dense(gguf_dense(m, L.wo_b), false, ob_fn);
            }
            bt_pt(x, m, m.attn_out, T, "attn_out", l);
        }
        /* the attention's all-reduce over the step (exact: 0 + r0 + .. + r3 a value), then post / collapse / norm (a
         * fused norm) by segment; ar_pcn_f (the all-reduce with it, RADIANCE_DS41_AR_PCN) is not taken under ds41_bi */
        if (L.ar_attn && (pfb & 3)) pf_attn(x, L, pfb);   /* tp8 loop D24: beside the attention all-reduce below */
        if (L.ar_attn && pa > 0 && tbo_a) {
            /* tp8 loop P15 bit 1: lane 0 behind piece 0's all-reduce (lane 1) for piece 0's pcn below; piece 1's
             * all-reduce goes to lane 1 in the MoE section, beside MoE pass 0 (no attention-zero rank under P15) */
            rad_lane_join(x, 1, 0);
            if (rad_rank(x) == 0) {
                g_pipe2_attn.fetch_add(1, std::memory_order_relaxed);
                g_tbo_attn.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (L.ar_attn && pa > 0) {
            /* tp8 loop P13b: an attention-zero rank (its attn_out zeros, no projection) issues piece 0's all-reduce here,
             * so every rank issues the same collectives in the same order; lane 0 behind lane 1, piece 1's */
            if (m.zero) ar_a_piece0();
            rad_lane_join(x, 1, 0);
            RAD_ISSUE_N(x, L.ar_attn, (pn - pa) * D, R(m.attn_out, p0 + pa, pn - pa), RAD_NONE);
            if (rad_rank(x) == 0) g_pipe2_attn.fetch_add(1, std::memory_order_relaxed);
        } else if (L.ar_attn && sp_l) {
            /* tp8 loop P20: the all-reduce's first half -- this rank's quarter of the sum, the same fold */
            RAD_ISSUE_N(x, m.sp_rs_a, T * D, brows(m.attn_out, T), RAD_NONE);
        } else if (L.ar_attn)
            whole([&](int64_t r0, int64_t n) { RAD_ISSUE_N(x, L.ar_attn, n * D, R(m.attn_out, r0, n), RAD_NONE); });
        t_mhc_f.clear();
        /* post / collapse / norm (+ the ffn mix: lever M1) over the rows `cur` names (P15 bit 1: one piece at a time) */
        const auto pcn_f_issue = [&] {
        if (L.pcn_f) {
            /* lever M1: a launch of >= mhc_fuse_min() rows as pcn_f + mix_f in one op (its rows' mix_f then not issued) */
            const int64_t fmin = L.pcn_mix_f ? mhc_fuse_min() : 0;
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                if (fmin > 0 && n >= fmin) {
                    RAD_ISSUE_N(x, L.pcn_mix_f, n, R(m.attn_out, r0, n), R(m.h0, r0, n), R(m.post, r0, n),
                                R(m.comb, r0, n), R(m.h1, r0, n), R(m.pre_a, r0, n), R(m.xcol, r0, n), RAD_W(L.ffn_norm),
                                R(m.act.x, r0, n), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                                R(m.pre_f, r0, n));
                    t_mhc_f.push_back({ r0, n });
                    mhc_note(x, n);
                    return;
                }
                if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
                RAD_ISSUE_N(x, L.pcn_f, n, R(m.attn_out, r0, n), R(m.h0, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h1, r0, n), R(m.pre_a, r0, n), R(m.xcol, r0, n), RAD_W(L.ffn_norm), R(m.act.x, r0, n));
            });
        } else
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
        };
        if (tbo_a) with_rows(p0, pa, pcn_f_issue);   /* P15 bit 1: piece 0 now, piece 1 after MoE pass 0 (below) */
        else if (sp_l) {
            /* tp8 loop P20: post / collapse / norm (+ the ffn mix) over this rank's quarter, the FFN input gathered raw */
            with_rows(sp0, spn, pcn_f_issue);
            RAD_ISSUE_N(x, m.sp_ag_f, T * D, brows(m.act.x, T), RAD_NONE);
            h_sh = true;
        } else
            pcn_f_issue();
        bt_pt(x, m, m.h1, T, "h1", l);
        bt_pt(x, m, m.act.x, T, "ffn_in", l);
        /* the MoE: one pass a segment (the router, the experts at the segment's rows, the shared arm, the gather and
         * the all-reduce over its rows); the ffn mix beside the first (r0 = 0) */
        MixBeside mb{ &L, &m, T };
        MoeFP8::Beside bs;
        bs.fn = mix_f_beside;
        bs.arg = &mb;
        PfBeside pfm{ &m, &L, pfb };   /* tp8 loop D24: the next layer's reads beside this MoE all-reduce */
        if (pfb & 5) {
            bs.before_ar = pf_moe_beside;
            bs.ar_arg = &pfm;
        }
        /* lever PSK: a cut layer's ffn mix over its runs, here (the block issues a beside only in a pass from row 0) */
        const MoeFP8::Beside* bsp = &bs;
        const auto mix_f_rest = [&] {
            whole([&](int64_t r0w, int64_t nw) {
                mhc_rest(t_mhc_f, r0w, nw, [&](int64_t r0, int64_t n) {   /* lever M1: the rows not mixed in pcn_f */
                    RAD_ISSUE_N(x, L.mix_f, n, R(m.h1, r0, n), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale), RAD_W(L.hc_f_base),
                                R(m.pre_f, r0, n), R(m.post, r0, n), R(m.comb, r0, n));
                });
            });
        };
        if (tbo_a) {
            bsp = nullptr;   /* P15 bit 1: the ffn mix's rest after piece 1's pcn (MoE section), cut or not */
        } else if (sp_l) {
            with_rows(sp0, spn, mix_f_rest);   /* tp8 loop P20: this rank's quarter, before the MoE */
            bsp = nullptr;
        } else if (cut) {
            mix_f_rest();
            bsp = nullptr;
        }
        /* tp8 loop P13 (RADIANCE_DS41_PIPE2, pipe2_env): an uncut layer of a lone prefill pass of >= m.pipe2 rows -- the
         * pass today issues ONE MoE pass over its rows with its own all-reduce (every ds41_bi form below does, for one
         * prefill segment) -- runs it in two pieces [0, pc) and [pc, T): both pieces' router projections + all_gathers
         * first (lane 0, nothing on lane 1), pass 0, its all-reduce on lane 1 behind lane 0 and beside pass 1 (no
         * collective inside: projected, own_ar false), then lane 0 joins lane 1 and issues pass 1's all-reduce. */
        int64_t pc = 0;
        if (p2_layer) {
            pc = (pn / 2 + kPipe2Align - 1) / kPipe2Align * kPipe2Align;
            const MoeFP8& mo = L.moe;
            if (pc < kPipe2Piece || pn - pc < kPipe2Piece || mo.c.calib || mo.op_moe_ar || mo.op_shared_ar ||
                mo.op_combine || mo.op_add || !mo.op_ar_ov || !mo.issues_ar(T, pn) || !mo.issues_ar(T, pc) ||
                !mo.issues_ar(T, pn - pc))
                pc = 0;
            if (pc == 0 && rad_rank(x) == 0) g_pipe2_serial.fetch_add(1, std::memory_order_relaxed);
        }
        if (pc > 0) {
            const int64_t n1 = pn - pc, q1 = p0 + pc;
            const bool pr0 = L.moe.route_splits(pc), pr1 = L.moe.route_splits(n1);
            if (m.pipe2_tbo & 4) {
                /* tp8 loop P16 (RADIANCE_DS41_PIPE2_TBO bit 4): the MoE pass whole -- the ops the published pass issues
                 * (router, experts, shared arm on its lane, gather, the ffn mix beside), without its all-reduce -- then
                 * the all-reduce over [p0, q1) here on lane 0 and over [q1, p0 + pn) below (lane 1 with bit 2) */
                L.moe.pass(x, T, p0, pn, false, bsp);
                L.moe.issue_ar(x, p0, pc, false);
                if (rad_rank(x) == 0) g_tbo_whole.fetch_add(1, std::memory_order_relaxed);
            } else if (tbo_a && pc == pa) {
                /* tp8 loop P15 bit 1: piece 0's router (projection + all_gather; lane 1 idle: piece 0's attention
                 * all-reduce was joined above), then lane 1 behind lane 0 (piece 1's o_b rows and the gather) and piece
                 * 1's attention all-reduce there, BESIDE MoE pass 0 (no collective inside: projected, own_ar false); lane
                 * 0 behind it for piece 1's pcn, the ffn mix's rest and piece 1's router; then P13a's order: piece 0's
                 * MoE all-reduce on lane 1 beside MoE pass 1, lane 0 behind it */
                if (pr0) {
                    L.moe.route_part(x, p0, pc);
                    L.moe.route_gather(x, p0, pc);
                }
                rad_lane_join(x, 0, 1);
                rad_lane(x, 1);
                RAD_ISSUE_N(x, L.ar_attn_ov, n1 * D, R(m.attn_out, q1, n1), RAD_NONE);
                rad_lane(x, 0);
                L.moe.pass(x, T, p0, pc, false, nullptr, false, pr0, true);
                rad_lane_join(x, 1, 0);
                with_rows(q1, n1, pcn_f_issue);
                mix_f_rest();
                if (pr1) {
                    L.moe.route_part(x, q1, n1);
                    L.moe.route_gather(x, q1, n1);
                }
                rad_lane_join(x, 0, 1);   /* lane 1 behind everything lane 0 issued: pass 0's gather wrote rows [p0, q1) */
                rad_lane(x, 1);
                L.moe.issue_ar(x, p0, pc, true);   /* the overlap instance, beside pass 1 */
                rad_lane(x, 0);
                L.moe.pass(x, T, q1, n1, false, nullptr, false, pr1, true);
                rad_lane_join(x, 1, 0);   /* lane 0 behind piece 0's all-reduce (and a lane-1 shared arm) */
            } else {
            if (pr0) {
                L.moe.route_part(x, p0, pc);
                L.moe.route_gather(x, p0, pc);
            }
            if (pr1) {
                L.moe.route_part(x, q1, n1);
                L.moe.route_gather(x, q1, n1);
            }
            L.moe.pass(x, T, p0, pc, false, bsp, false, pr0, true);
            rad_lane_join(x, 0, 1);   /* lane 1 behind everything lane 0 issued: pass 0's gather wrote rows [p0, q1) */
            rad_lane(x, 1);
            L.moe.issue_ar(x, p0, pc, true);   /* the overlap instance, beside pass 1 */
            rad_lane(x, 0);
            L.moe.pass(x, T, q1, n1, false, nullptr, false, pr1, true);
            rad_lane_join(x, 1, 0);   /* lane 0 behind piece 0's all-reduce (and a lane-1 shared arm) */
            }
            /* piece 1's MoE all-reduce: tp8 loop P15 bit 2 -- on lane 1 beside the next layer's piece-0 pcn and attention
             * inputs, where that layer is the adjacent uncut one this node issues in this pass with nothing between (no
             * crossing, no receive, no Engram block / tap, its pcn fused) and the attention inputs take the per-segment
             * (cheap) form; else on lane 0 here (P13a) */
            const Layer* tnx = next_layer(m, L.l);
            const bool nx_cut = psk && (L.l + 1 >= m.psk_first || (l20_on && L.l + 1 == m.psk_l20 && !cr_all) || cr_any);
            const bool tbo_x = (m.pipe2_tbo & 2) && !cut && !m.bt.on && p0 == 0 && pn == T &&
                               bi_cheap(g_sw_bi.load(std::memory_order_relaxed)) && pp_send_after(m, L.l) < 0 && tnx &&
                               tnx->l == L.l + 1 && pp_part_layer(m, part, tnx->l) && pp_recv_at(m, tnx->l) < 0 &&
                               tnx->pcn_a && tnx->pcn_mix && !tnx->e_ids && !tnx->tap && !nx_cut && !m.media;
            if (tbo_x) {
                rad_lane_join(x, 0, 1);   /* lane 1 behind pass 1's gather (rows [q1, p0 + pn)) */
                rad_lane(x, 1);
                L.moe.issue_ar(x, q1, n1, true);   /* the overlap instance, beside the next layer's piece 0 */
                rad_lane(x, 0);
                t_tbo = TboDefer{ tnx->l, p0, pc, pn };
            } else {
                L.moe.issue_ar(x, q1, n1, false);
            }
            if (l == 0) g_bi_moe.fetch_add(2, std::memory_order_relaxed);
            if (rad_rank(x) == 0) {
                static thread_local int t_p2_step = -1;
                g_pipe2_layers.fetch_add(1, std::memory_order_relaxed);
                if (batch->step != t_p2_step) {
                    t_p2_step = batch->step;
                    g_pipe2_rows.fetch_add(pn, std::memory_order_relaxed);
                    const long long n = g_pipe2_passes.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (n == 1 || n % 64 == 0)
                        fprintf(stderr, "radiance: deepseek41: RADIANCE_DS41_PIPE2 (tp8 loop P13): %lld prefill passes ran "
                                        "their uncut layers' MoE in two pieces with piece 0's all-reduce on the second lane "
                                        "(%lld layer-passes, %lld rows; this one %lld rows as %lld + %lld from layer %d); "
                                        "%lld layer-passes of lone prefills ran whole (a piece below %lld rows or a reducing "
                                        "block); attention all-reduces in the pieces (RADIANCE_DS41_PIPE2_ATTN) %lld; "
                                        "RADIANCE_DS41_PIPE2_TBO (P15): piece 1's attention all-reduce beside MoE pass 0 %lld, "
                                        "piece 1's MoE all-reduce beside the next layer's piece 0 %lld, finished at once %lld; of the layer-passes "
                                        "above, the MoE ran WHOLE with its all-reduce in two halves (bit 4) %lld\n",
                                n, g_pipe2_layers.load(std::memory_order_relaxed), g_pipe2_rows.load(std::memory_order_relaxed),
                                (long long)pn, (long long)pc, (long long)n1, l, g_pipe2_serial.load(std::memory_order_relaxed),
                                (long long)kPipe2Piece, g_pipe2_attn.load(std::memory_order_relaxed),
                                g_tbo_attn.load(std::memory_order_relaxed), g_tbo_xl.load(std::memory_order_relaxed),
                                g_tbo_flush.load(std::memory_order_relaxed), g_tbo_whole.load(std::memory_order_relaxed));
                }
            }
        } else if (sp_l) {
            /* tp8 loop P20: the MoE over every row as the pass issues it (one pass from row 0: router, experts, shared arm,
             * gather), its all-reduce's first half by the plugin -- this rank's quarter of the delta */
            {
                const TwinScope tws(sg.size() == 1 && !sg[0].dec, T);   /* tp8 loop P30t: the lone prompt segment's pass */
                L.moe.pass(x, T, 0, T, false, nullptr);
            }
            RAD_ISSUE_N(x, m.sp_rs_f, T * D, brows(m.act.x, T), RAD_NONE);
            if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
        } else
        {   /* ds41_bi 2 / 4 / 6: the decode segment in pieces within the EXL3 decode band; 5 / 6 (lever A9): one all-reduce */
            const int biv = g_sw_bi.load(std::memory_order_relaxed);
            const int64_t piece = bi_pieces(biv) ? kBiMoe : ((int64_t)1 << 40);
            /* lever P (ds41_moe1 1): the step is ONE decode segment above a piece and within the dk ops' rows -- its MoE
             * as one pass, the routed experts through the decode-band-bytes ops (the pieces' bytes per row); the pass
             * is the whole step (r0 0, T rows), so it takes the fused gather + all-reduce and reports its routing as a
             * one-pass decode step does */
            const bool p1 = g_sw_p1.load(std::memory_order_relaxed) > 0 && g_sw_p1_ok.load(std::memory_order_relaxed) &&
                            sg.size() == 1 && sg[0].dec && sg[0].r0 == 0 && sg[0].n == T && sg[0].n > piece &&
                            sg[0].n <= moe1_max() && !cut && L.moe.dk_takes(sg[0].n);
            if (p1) {
                L.moe.pass(x, T, 0, T, true, bsp, true);
                if (l == 0) {
                    g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                    if (rad_rank(x) == 0) {
                        g_p1_rows.fetch_add(T, std::memory_order_relaxed);
                        const long long n = g_p1_steps.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (n == 1 || n % 256 == 0)   /* a c24 block is ~500 such steps: two lines a block show it engaged */
                            fprintf(stderr, "radiance: deepseek41: lever P (ds41_moe1 1): %lld decode steps of 17 .. %lld "
                                            "rows issued their MoE as ONE pass (%lld rows in all; this one %lld rows)\n",
                                    n, (long long)moe1_max(), g_p1_rows.load(std::memory_order_relaxed), (long long)T);
                    }
                }
            } else if (!bi_onear(biv)) {   /* ds41_bi 1 .. 4: int12's issue, unchanged */
                for (const BiSeg& s : sg) {
                    if (!s.dec || s.n <= piece) {
                        const TwinScope tws(!s.dec, s.n);   /* tp8 loop P30t: a prompt segment's shared arm on its twin */
                        L.moe.pass(x, T, s.r0, s.n, true, bsp);
                        if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    for (int64_t a = 0; a < s.n; a += piece) {
                        L.moe.pass(x, T, s.r0 + a, std::min(piece, s.n - a), true, bsp);
                        if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            } else {
                /* the passes (first row, rows) in the order 1 .. 4 issue them */
                static thread_local std::vector<std::pair<int64_t, int64_t>> passes;
                static thread_local std::vector<int64_t> pass_pf;   /* tp8 loop P30t: a pass that is one whole prompt segment: its rows */
                passes.clear();
                pass_pf.clear();
                /* lever PMM (ds41_bi 7): a prefill segment of >= kPmmMin rows right after the previous one (also of
                 * >= kPmmMin rows) joins its pass */
                const bool pmm = bi_pmm(biv);
                bool prev_pf = false;
                int64_t merged = 0;
                for (const BiSeg& s : sg) {
                    const bool pf = !s.dec && s.n >= kPmmMin;
                    if (pmm && pf && prev_pf && passes.back().first + passes.back().second == s.r0) {
                        passes.back().second += s.n;
                        pass_pf.back() = 0;
                        ++merged;
                        continue;
                    }
                    prev_pf = pf;
                    if (!s.dec || s.n <= piece) {
                        passes.emplace_back(s.r0, s.n);
                        pass_pf.push_back(s.dec ? 0 : s.n);
                        continue;
                    }
                    for (int64_t a = 0; a < s.n; a += piece) {
                        passes.emplace_back(s.r0 + a, std::min(piece, s.n - a));
                        pass_pf.push_back(0);
                    }
                }
                if (merged > 0 && l == 0 && rad_rank(x) == 0) {
                    const long long n = g_pmm_steps.fetch_add(1, std::memory_order_relaxed) + 1;
                    g_pmm_merged.fetch_add(merged, std::memory_order_relaxed);
                    if (n == 1 || n % 256 == 0)
                        fprintf(stderr, "radiance: deepseek41: lever PMM (ds41_bi 7): %lld segmented steps ran their consecutive "
                                        "prefill segments' MoE as one pass at layer 0 (%lld passes fewer; this one %lld)\n",
                                n, g_pmm_merged.load(std::memory_order_relaxed), (long long)merged);
                }
                /* one all-reduce over the step's rows: only where the block reduces its gathered delta in ONE collective
                 * over h.x (issue_ar), nothing takes it downstream, and the gather is the block's own; a PSK-cut layer
                 * runs fewer rows than T (its passes do not cover [0, T)), so each pass keeps its own all-reduce there */
                const bool onear = passes.size() > 1 && !cut && !L.moe.op_moe_ar && !L.moe.op_shared_ar &&
                                   !L.moe.op_combine && !L.moe.op_add && !L.moe.ar_out && L.moe.issues_ar(T, T) &&
                                   !gather_taken(m.g, T, L.moe.gather_out_rows, L.moe.gather_out_rows6);
                for (size_t pi = 0; pi < passes.size(); ++pi) {
                    const auto& ps = passes[pi];
                    const TwinScope tws(pass_pf[pi] > 0, pass_pf[pi]);   /* tp8 loop P30t: a whole prompt segment's pass */
                    L.moe.pass(x, T, ps.first, ps.second, !onear, bsp);
                    if (l == 0) g_bi_moe.fetch_add(1, std::memory_order_relaxed);
                }
                if (onear) {
                    if (bsp && bsp->before_ar) bsp->before_ar(x, bsp->ar_arg);   /* tp8 loop D24 */
                    L.moe.issue_ar(x, 0, T, false);
                    if (l == 0) g_bi_onear.fetch_add(1, std::memory_order_relaxed);
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
        /* tp8 S6: the sender's last layer before the cut ends at the boundary op's inputs, which the crossing carries over
         * this layer's live rows (the follower's t - 1: X2, then its pass ends; the leader's h - 1: X1, then its next
         * layer is t) */
        /* tp8 loop P28 round 2c: a band pass's leader keeps band layer l's key rows (window_kv_normed = the stored key: kvfold's
         * output, or kv_f + kv_q's) for the band X1 */
        if (band_l) {
            const size_t j = (size_t)(L.l - m.band.h);
            RAD_ISSUE_N(x, m.band_cp[j], T, brows(m.kvn, T), brows(m.band_key[j], T));
        }
        if (const int ring = pp_send_after(m, L.l); ring >= 0) {
            int64_t lo = 0, hi = 0;
            live_span(&lo, &hi);
            /* tp8 loop P20 round 3: an SP layer's X1 leaves row-sharded (its rows are the pass's [0, T): the shard's rows) */
            const bool x1_sh = sp_l && ring == RAD_PP_X1 && m.pp_x1_sh;
            if (x1_sh && (lo != 0 || hi != T)) {
                rad_step_fail(x, "deepseek41: RADIANCE_DS41_SPHC_X1: an SP pass's X1 rows are not the pass's (a plan defect)");
                return;
            }
            /* tp8 loop P25: an SP layer before its segment's boundary -- the next segment takes the streams sharded (2 saves
             * only this rank's quarter; 1 the whole rows, the quarter among them) */
            const bool seg_sh = sp_l && sp_segp && ring >= kSegSend;
            if (seg_sh && (lo != 0 || hi != T)) {
                rad_step_fail(x, "deepseek41: RADIANCE_DS41_SPHC_SEG: an SP segment boundary's rows are not the pass's (a plan "
                                 "defect)");
                return;
            }
            if (sp_l && !x1_sh && !seg_sh) {
                rad_step_fail(x, "deepseek41: RADIANCE_DS41_SPHC: an SP layer before a whole crossing (a plan defect)");
                return;
            }
            if (!pp_cross(x, m, batch, ring, lo, hi, x1_sh || (seg_sh && sp_seg >= 2))) return;
            if (seg_sh) h_sh = false;   /* the segment ends here; the next one restores the quarter */
            if (x1_sh) {
                h_sh = false;   /* nothing on this node reads the residual parts after X1 (the head part ends; a whole pass
                                 * continues at layer t from X2) */
                if (rad_rank(x) == 0) g_sp_x1.fetch_add(1, std::memory_order_relaxed);
            }
            if (ring == RAD_PP_X1 && part != RAD_PP_PART_HEAD) continue;   /* tp8 S5: the head part ends here */
            bt_flush(x, m, batch);
            return;
        }
        /* post_f, or post_f with the next collapse + norm (a fused norm, by segment); never the AR_PCN forms here */
        const Layer* nx = next_layer(m, L.l);   /* tp8 S3: the next layer this node holds (a gap passes the streams on) */
        if (nx && t_tbo.l >= 0 && t_tbo.l == nx->l) {
            /* tp8 loop P15 bit 2 (set in the MoE section above): the next layer's pcn over piece 0 only, beside piece 1's
             * MoE all-reduce on lane 1; that layer issues piece 1's after joining it */
            const TboDefer d = t_tbo;
            with_rows(d.p0, d.pa, [&] { pcn_next(*nx, true); });
        } else if (nx && nx->pcn_a && sp_l) {
            with_rows(sp0, spn, [&] { pcn_next(*nx, true); });   /* tp8 loop P20: this rank's quarter */
            h_sh = xn_sh = true;
        } else if (nx && !nx->pcn_a && sp_l) {
            with_rows(sp0, spn, [&] {   /* tp8 loop P20: the post into an Engram layer's streams, this rank's quarter */
                whole([&](int64_t r0, int64_t n) {
                    RAD_ISSUE_N(x, L.post_f, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                                R(m.h0, r0, n));
                });
            });
            h_sh = true;
        } else if (nx && nx->pcn_a) {
            /* lever M1: a launch of >= mhc_fuse_min() rows as pcn_a + the next layer's mix_a in one op */
            const int64_t fmin = nx->pcn_mix ? mhc_fuse_min() : 0;
            t_mhc_a.clear();
            t_mhc_a_l = nx->l;
            rows(kBiNorm, [&](int64_t r0, int64_t n) {
                if (fmin > 0 && n >= fmin) {
                    RAD_ISSUE_N(x, nx->pcn_mix, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                                R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(nx->attn_norm), R(m.xn, r0, n),
                                RAD_W(nx->hc_a_fn), RAD_W(nx->hc_a_scale), RAD_W(nx->hc_a_base), R(m.pre_a, r0, n));
                    t_mhc_a.push_back({ r0, n });
                    mhc_note(x, n);
                    return;
                }
                if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
                RAD_ISSUE_N(x, nx->pcn_a, n, R(m.act.x, r0, n), R(m.h1, r0, n), R(m.post, r0, n), R(m.comb, r0, n),
                            R(m.h0, r0, n), R(m.pre_f, r0, n), R(m.xcol, r0, n), RAD_W(nx->attn_norm), R(m.xn, r0, n));
            });
        } else if (!nx && m.pcn_fin)
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
    tbo_flush();   /* tp8 loop P15: nothing pending after the last layer by design (a deferral names an adjacent layer) */
    /* tp8 loop P20: what an SP pass left sharded is read by nothing after it (spv's rule; the split layer with no cone row
     * reads xn alone) -- xn here would be a defect of the plan, gathered so no reader can see a quarter */
    sp_gather_x();
    /* (the SP line: sp_end, at every exit of the pass -- a pipeline part returns at its crossing out) */
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
        /* the head (logits_gemm_ggml, banded): the sampled rows in launches within the decode band (tp8 loop D22: the
         * dense band RADIANCE_DS41_DENSE_BAND under WMK for a GGUF head) */
        const int64_t hb = wmkw && gguf_dense(m, m.head) ? dband : kBiBand;
        static thread_local std::vector<std::pair<int64_t, int64_t>> hc;
        ds41_band_cuts(batch->n_out, hb, hc);
        for (const auto& c : hc) {
            const int64_t a = c.first, n = c.second;
            RAD_ISSUE_N(x, m.head_op, n, a ? brow_slice(m.logit_input, a, n, D) : brows(m.logit_input, n), RAD_W(m.head),
                        a ? brow_slice(m.logits, a, n, g.n_vocab) : brows(m.logits, n));
        }
        bt_pt(x, m, m.logits, batch->n_out, "logits", -1, 1);
    }
    bt_flush(x, m, batch);
}

static void step(RadCtx* x, const RadBatch* batch) {
    auto& m = *models[rad_rank(x)];
    /* tp8 S3: a strict node is this node's part of a two-node engine; run as one only by the pipeline (tp8 S4) */
    if (m.lr.on && m.lr.strict && !m.pp) {
        rad_step_fail(x, "deepseek41: RADIANCE_DS41_LAYERS_MODE=strict declares this node's part of a two-node engine "
                         "only; this build has no node link (tp8 S4) -- run the stage alone with the proxy mode");
        return;
    }
    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(x, batch);
        return;
    }
    dkv_pass_reset(x, batch->n_tok);   /* lever DKV: nothing held at a pass's start */
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
    /* tp8 S6: a pipeline node runs BI's segments and PSK's cut passes too (step_bi crosses at the cut, per node) */
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
    /* tp8 loop P30t: with a twin class on, a lone prompt pass (one prefill sequence, no decode rows) runs through step_bi as its
     * one segment -- the issue a multi-prompt pass gives the same prompt -- so its rows take the same twin forms alone and beside
     * other prompts (step() issues no twin); a one-segment step_bi issues step()'s ops on the same rows (lever PSK's route above).
     * Every twin class off (ds41_twin 0, or ds41_bi 0): step() as today. */
    if (m.bi_sub && twin_mask_now() && batch->draft_pass == 0 && batch->n_seq == 1) {
        int64_t Dn = 0, DT = 0;
        batch_split(batch, &Dn, &DT);
        if (Dn == 0) {
            static thread_local std::vector<BiSeg> ps1(1);
            ps1[0] = BiSeg{ 0, batch->n_tok, 0, 1, false, -1 };
            step_bi(x, m, batch, ps1, true);
            return;
        }
    }
    const auto& c = m.c;
    const auto& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, D = c.n_embd;
    const int64_t HD = g.n_head * c.head_dim, hd = c.head_dim;
    bt_begin(m);
    const int pfb = pf_bits(batch);   /* tp8 loop D24: 0 = no weight reads beside the all-reduces */
    pf_pass_note(x, batch, pfb);
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

    /* tp8 S5 + S6: the part of the leader's pass this issue is (pp_part): its head -- the prologue, layers [0, h) and X1's
     * send -- or its tail -- the prologue's per-step tables again, X2's receive at layer t and everything after. One lane,
     * the follower, one machine: the whole pass. */
    const int part = pp_part(m);
    if (t_band) {   /* tp8 loop P28 round 2c: the band's crossing and casts are step_bi's (lever BI / PSK route every band pass there) */
        rad_step_fail(x, "deepseek41: RADIANCE_DS41_PP_BAND: a band pass reached the unsegmented step (ds41_bi 0 and no PSK plan)");
        return;
    }
    const bool run_head = part != RAD_PP_PART_TAIL;
    media_route(m, batch);
    /* tp8 dev 1a: under pp_fe the follower's own lookup in place of X1; the leader keeps its own (its xcol as before) and
     * only X1's send goes */
    /* tp8 dev D10 round 2 (m.pp_ids): the follower takes X1 -- with the step's token ids, into its batch's -- before its
     * lookup and its token_map read them; the leader sends it after its own lookup, under pp_fe too (the ids alone) */
    const bool x1_top = m.pp && m.pp_ids && m.pp_h == 0 && m.pp_rank == 1 && pp_seg_first(m);
    if (x1_top && !pp_cross(x, m, batch, RAD_PP_X1, 0, T)) return;
    if (((!m.pp || m.pp_rank == 0) && run_head && pp_lead_first(m)) || (m.pp_fe && m.pp_rank == 1 && pp_seg_first(m))) {
    RAD_ISSUE_N(x, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.xcol, T));
    m.mrows.step(x, batch, m.xcol, T);
    if (m.embed_ar) RAD_ISSUE_N(x, m.embed_ar, T * D, brows(m.xcol, T), RAD_NONE);
    }
    /* X1 at W(0, t): the leader's lookup out, the follower's in, where one process's lookup was (tp8 S6: at h > 0 X1 is
     * the residual after layer h - 1, below, and the follower's layer h starts from it: no streams of its own here) */
    if (m.pp && m.pp_h == 0 && run_head && (!m.pp_fe || m.pp_ids) && !x1_top && pp_seg_first(m) &&
        !pp_cross(x, m, batch, RAD_PP_X1, 0, T))
        return;
    if (part == RAD_PP_PART_HEAD && m.pp_h == 0) return;   /* tp8 S5: at W(0, t) the head part ends with X1's send */
    if (!(m.pp && m.pp_rank == 1 && m.pp_h > 0)) {
    for (int64_t i = 0; i < c.hc; ++i)
        RAD_ISSUE_N(x, m.expand, T, brows(m.xcol, T), bcol(m.h0, i * D, D, T));
    RAD_ISSUE_N(x, m.pre0, T, brows(m.pre_f, T));
    RAD_ISSUE_N(x, m.pre1, T, bcol(m.pre_f, 0, 1, T));
    }
    if (m.sone && g_sw_one.load(std::memory_order_relaxed) > 0) RAD_ISSUE_N(x, m.sone, T, brows(m.mw.sgate, T));
    RAD_ISSUE_N(x, m.rope_a, T, pos, brows(m.cs_a, g.max_ctx));
    RAD_ISSUE_N(x, m.rope_b_step, T, pos, brows(m.cs_b, g.max_ctx));
    /* tp8 dev D10 round 2: the token-id ops -- here, or after X1 at layer h on the follower at h > 0 under pp_ids */
    auto tok_ops = [&]() {
    if (m.tok_map)
        RAD_ISSUE_N(x, m.tok_map, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tmap),
                    brows(m.ctok, T));
    media_tokens(x, m, batch, T);
    ti_note(x, m);   /* lever TI */

    /* The Engram's row ids are a function of the batch's tokens alone (token_map, each Engram
     * layer's token history), never of an activation, so every Engram layer's ngram_ids is issued
     * here, before the first block, each into its own host buffer (rad_block_ple.h's ids() does
     * the same). A host-site gather (the table on the disk or RAM tier) joins on the last device
     * op that touched the host arena: these few ops, or the previous Engram layer's copy of its
     * rows. It then reads its rows while the device runs the blocks the host has already issued,
     * where ngram_ids at its layer (and device-domain rows) made the device drain before every
     * gather and idle through it. */
    for (auto& L : m.layers) {
        /* tp8 S5 + S6: an Engram layer's ids (and its token window's update) in the part that issues the layer only */
        if (!L.e_ids || !pp_part_layer(m, part, L.l)) continue;
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
    };
    if (!pp_ids_late(m)) tok_ops();
    t_mhc_a.clear();           /* lever M1: nothing fused yet this step */
    t_mhc_a_l = -1;

    for (auto& L : m.layers) {
        if (!m.lr.held(L.l)) continue;   /* tp8 S3: another node's layer */
        if (!pp_part_layer(m, part, L.l)) continue;   /* tp8 S5 + S6: the other part of a lane step's leader pass */
        const int l = L.l;
        const int64_t r = L.r;
        /* tp8 S4 / S6: the receiver's first layer after the cut (the leader's t, the follower's h > 0) starts from the
         * crossing with the op that ends the layer before it in one process: the fused post + collapse + norm (its M1 form
         * where that is what one process takes), else mhc_post, after which this layer's own collapse runs as below. The
         * same choice as the end of a layer (below). The copied source's entries are stored into this node's copy first. */
        if (const int ring = pp_recv_at(m, l); ring >= 0) {
            if (!pp_cross(x, m, batch, ring, 0, T)) return;
            if (!pp_copy_issue(x, m, batch, ring, true)) return;
            if (ring == RAD_PP_X1 && pp_ids_late(m)) tok_ops();   /* tp8 dev D10 round 2: X1's ids are in the batch now */
            if (L.pcn_a) {
                const int64_t fmin = L.pcn_mix ? mhc_fuse_min() : 0;
                t_mhc_a.clear();
                t_mhc_a_l = L.l;
                if (fmin > 0 && T >= fmin) {
                    RAD_ISSUE_N(x, L.pcn_mix, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                                brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                                RAD_W(L.attn_norm), brows(m.xn, T), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                                RAD_W(L.hc_a_base), brows(m.pre_a, T));
                    t_mhc_a.push_back({ 0, T });
                    mhc_note(x, T);
                } else {
                    if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
                    RAD_ISSUE_N(x, L.pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                                brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                                RAD_W(L.attn_norm), brows(m.xn, T));
                }
            } else {
                RAD_ISSUE_N(x, pp_post_op(m, ring), T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                            brows(m.comb, T), brows(m.h0, T));
            }
        }
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
                e2_gemm(x, m, L, T, brows(m.embd, T), brows(m.ekv, T));
            }
            media_engram(x, m, L, batch, T);
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
            /* lever M1: the rows the previous layer's fused pcn_a did not mix (here all or none) */
            if (t_mhc_a_l != l) t_mhc_a.clear();
            mhc_rest(t_mhc_a, 0, T, [&](int64_t r0, int64_t n) {
                auto Rs = [&](rad_buf h) { return r0 == 0 ? brows(h, n) : brow_slice(h, r0, n, m.bp.at(h)); };
                RAD_ISSUE_N(x, L.mix_a, n, Rs(m.h0), RAD_W(L.hc_a_fn), RAD_W(L.hc_a_scale),
                            RAD_W(L.hc_a_base), Rs(m.pre_a), Rs(m.post), Rs(m.comb));
            });
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
                /* tp8 S6: the candidate layer unsplit where its keep words cross (Model::pp_unsplit_cand) */
                const bool split = m.isplit && m.sel_gather && !(m.pp_unsplit_cand && l == c.cand);
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
                    ixni_note(x, batch->positions_h ? batch->positions_h + r0 : nullptr, rows, ru.R, live, r);
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
            if (gq_qkv_on(L, T))   /* lever 1U-GQ: q_a + kv_g as one launch */
                gq_qkv_issue(x, m, L, T, brows(m.xn, T), brows(m.qr, T), brows(m.kv, T));
            else {
                q_a(); kv_g();
            }
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
            /* lever DKV: the whole pass as one segment (decode-band rows: the gathered rows; taller: the plan's) */
            std::vector<RadOperand> da;
            if (r > 0 && (m.lu || m.selkv)) {
                const int src = c.src_of[(size_t)l];
                const auto* cb = kv_batch(batch, m.comp[r]);
                da = { brows(m.q, T), kv_cache(m.win, l), wbt, wused, kv_cache(m.comp[r], src),
                       praw2(cb->block_table, RAD_I32, S, cb->block_table_pitch), brows(m.sel, T), pos, cu,
                       RAD_W(L.sinks), brows(m.o, T) };
                if (ofold) da.push_back(acs);
            }
            const int64_t lp = batch->positions_h && S == 1 && T > 0 ? (int64_t)batch->positions_h[T - 1] : -1;
            if (!da.empty() && dkv_attn(x, m, L, ofold, T <= kBiBand, 0, S, T, 0, T, std::move(da), lp)) {
            } else if (r > 0) {
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
            if (gq_oa_on(L, T))   /* lever 1U-GQ: the two groups as one launch */
                gq_oa_issue(x, m, L, T, bcol(m.o, 0, gd, T), bcol(m.o, gd, gd, T), bcol(m.oa, 0, c.o_rank, T),
                            bcol(m.oa, c.o_rank, c.o_rank, T));
            else
                for (int j = 0; j < m.gr; ++j) {
                    const rad_buf od = (size_t)j < L.oa_dst.size() ? L.oa_dst[(size_t)j] : 0;   /* tp8 loop P30f */
                    lin_issue(x, m, L.o_a[(size_t)j], T, bcol(m.o, j * gd, gd, T), L.wo_a[(size_t)j],
                              od ? brows(od, T) : bcol(m.oa, j * c.o_rank, c.o_rank, T));
                }
            bt_pt(x, m, oa_split(L) ? 0 : m.oa, T, "oa", l);
            if (late) mix_a();
            lin_issue(x, m, L.o_b, T, brows(m.oa, T), L.wo_b, brows(m.attn_out, T));
            bt_pt(x, m, m.attn_out, T, "attn_out", l);
        }
        t_mhc_f.clear();   /* lever M1: this layer's ffn mix not fused yet */
        if ((L.ar_attn || L.ar_pcn_f) && (pfb & 3)) pf_attn(x, L, pfb);   /* tp8 loop D24: beside the all-reduce below */
        if (L.ar_pcn_f && ar_wide_write_exact(m.g, T))   /* the all-reduce, post_a, col_f and norm_f in one */
            RAD_ISSUE_N(x, L.ar_pcn_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T));
        else {
        if (L.ar_attn) RAD_ISSUE_N(x, L.ar_attn, T * D, brows(m.attn_out, T), RAD_NONE);
        const int64_t fmin_f = L.pcn_mix_f ? mhc_fuse_min() : 0;
        if (L.pcn_f && fmin_f > 0 && T >= fmin_f) {   /* lever M1: post_a, col_f, norm_f and the ffn mix in one */
            RAD_ISSUE_N(x, L.pcn_mix_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T), RAD_W(L.hc_f_fn), RAD_W(L.hc_f_scale),
                        RAD_W(L.hc_f_base), brows(m.pre_f, T));
            t_mhc_f.push_back({ 0, T });
            mhc_note(x, T);
        } else if (L.pcn_f) {   /* post_a, col_f and norm_f in one */
            if (fmin_f > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
            RAD_ISSUE_N(x, L.pcn_f, T, brows(m.attn_out, T), brows(m.h0, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h1, T), brows(m.pre_a, T), brows(m.xcol, T),
                        RAD_W(L.ffn_norm), brows(m.act.x, T));
        } else
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
        PfBeside pfm{ &m, &L, pfb };   /* tp8 loop D24: the next layer's reads beside this MoE all-reduce */
        if (pfb & 5) {
            bs.before_ar = pf_moe_beside;
            bs.ar_arg = &pfm;
        }
        L.moe.step(x, batch, bs);
        bt_pt(x, m, m.mw.logits, T, "router", l);
        bt_pt(x, m, m.mw.ids, T, "ids", l);
        bt_pt(x, m, m.mw.ew, T, "ew", l);
        bt_pt(x, m, m.mw.sout, T, "sout", l);
        bt_pt(x, m, m.act.x, T, "moe_out", l);
        /* post_f, or post_f with the next collapse + norm (the next layer's pcn_a, or pcn_fin); under
         * RADIANCE_DS41_AR_PCN=2 with the block's all-reduce in front (the block left it to us: ar_out). */
        const Layer* nx = next_layer(m, L.l);   /* tp8 S3: the next layer this node holds (a gap passes the streams on) */
        /* tp8 S4 / S6: the sender's last layer before the cut ends at the boundary op's inputs, which the crossing carries
         * (the follower's t - 1: X2, then its pass ends; the leader's h - 1: X1, then its next layer is t) */
        if (const int ring = pp_send_after(m, L.l); ring >= 0) {
            if (!pp_cross(x, m, batch, ring, 0, T)) return;
            if (ring == RAD_PP_X1 && part != RAD_PP_PART_HEAD) continue;   /* tp8 S5: the head part ends here */
            bt_flush(x, m, batch);
            return;
        }
        const bool arw = ar_wide_write_exact(m.g, T);
        if (nx && nx->ar_pcn_a && arw)
            RAD_ISSUE_N(x, nx->ar_pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(nx->attn_norm), brows(m.xn, T));
        else if (!nx && m.ar_pcn_fin && arw)
            RAD_ISSUE_N(x, m.ar_pcn_fin, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                        brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                        RAD_W(m.final_gain), brows(m.final_hidden, T));
        else if (nx && nx->pcn_a) {
            const int64_t fmin = nx->pcn_mix ? mhc_fuse_min() : 0;
            t_mhc_a.clear();
            t_mhc_a_l = nx->l;
            if (fmin > 0 && T >= fmin) {   /* lever M1: pcn_a + the next layer's mix_a in one */
                RAD_ISSUE_N(x, nx->pcn_mix, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                            brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                            RAD_W(nx->attn_norm), brows(m.xn, T), RAD_W(nx->hc_a_fn), RAD_W(nx->hc_a_scale),
                            RAD_W(nx->hc_a_base), brows(m.pre_a, T));
                t_mhc_a.push_back({ 0, T });
                mhc_note(x, T);
            } else {
                if (fmin > 0 && rad_rank(x) == 0) g_mhc_small.fetch_add(1, std::memory_order_relaxed);
                RAD_ISSUE_N(x, nx->pcn_a, T, brows(m.act.x, T), brows(m.h1, T), brows(m.post, T),
                            brows(m.comb, T), brows(m.h0, T), brows(m.pre_f, T), brows(m.xcol, T),
                            RAD_W(nx->attn_norm), brows(m.xn, T));
            }
        } else if (!nx && m.pcn_fin)
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
    /* The tower's span row, so a pass size the core picks itself is whole rows (16380 of 16384). */
    if (Ds41Vision::present(meta)) out->enc_merge = Ds41Vision::merge_of(meta);
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
    uint64_t geo = 1;   /* the pass's geometry beyond the core's key: a constant, or (tp8 loop F58) its band pieces */
    /* Lever PSK design 2: a pass the scheduler planned (rad_psk_plan) is shaped by its targets, which no key holds: not
     * recorded */
    if (deepseek41::psk_hint(b)) return 0;
    /* Lever E1: a pass it may deduplicate issues row counts read from the step's ids at issue: not recorded */
    if (deepseek41::engdd_may_act(b)) return 0;
    if (b && b->draft_pass == 0 && deepseek41::g_sw_bi.load(std::memory_order_relaxed) > 0) {
        int64_t Dn = 0, DT = 0;
        rad::arch::batch_split(b, &Dn, &DT);
        if (b->n_seq - Dn >= 2) return 0;
        /* tp8 loop F58 (RADIANCE_DS41_BI_DBAND=1): a decode segment above the band issues its attention in pieces of whole
         * sequences, shaped by their row counts -- recorded only where every decode sequence has max_q_len_decode rows
         * (the key above holds that layout); any other layout is issued live */
        const int dband = deepseek41::g_bi_dband.load(std::memory_order_relaxed);
        if (dband > 0 && DT > deepseek41::kBiBand && (int64_t)b->max_q_len_decode * Dn != DT) {
            /* RADIANCE_DS41_BI_DBAND=2: keyed by the pieces step_bi issues (the same host positions, the same packing);
             * positions that do not give the layout (step_bi reads cu back) stay live */
            if (dband < 2) return 0;
            static thread_local std::vector<int64_t> st;
            static thread_local std::vector<Ds41DecPiece> pc;
            if (!ds41_dec_starts(b->positions_h, DT, Dn, st) || ds41_dec_pieces(st, DT, deepseek41::kBiBand, pc) != 0 ||
                pc.empty())
                return 0;
            geo = ds41_dec_pieces_key(pc);
        }
    }
    /* Lever PSK-DS (dspark_pfctx 1): a context pass cut to its entries' readable rows is shaped by their lengths, which
     * the key does not hold: not recorded. A pass it leaves whole issues as today and keys as today. */
    if (b && b->draft_pass < 0 && deepseek41::g_sw_pfctx.load(std::memory_order_relaxed) > 0) {
        static thread_local std::vector<std::pair<int64_t, int64_t>> ru;
        int64_t kept = 0;
        if (deepseek41::pfctx_plan(b, deepseek41::g_pfctx_window.load(std::memory_order_relaxed), &ru, &kept)) return 0;
    }
    return geo;
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

/* Levers M1 / IX-NI's live counters (ds41-int9b, GET /admin/fn4 "i9b"), rank 0's, cumulative, counted at issue (a
 * recorded pass when it is recorded): out[0] lever M1 declared and served here (1 / 0), [1] fused launches
 * (mhc_post_collapse_rmsnorm_mix), [2] their rows, [3] launches of the pair the switch left unfused (below its rows),
 * [4] index_score runs issued under ix_ni, [5] the plane bytes those runs leave unwritten (past each row's reach).
 * Returns how many it wrote (6). */
extern "C" int rad_i9b_stats(long long* out, int n) {
    using namespace deepseek41;
    if (!out || n < 6) return 0;
    out[0] = g_sw_mhc_ok.load(std::memory_order_relaxed);
    out[1] = g_mhc_pairs.load(std::memory_order_relaxed);
    out[2] = g_mhc_rows.load(std::memory_order_relaxed);
    out[3] = g_mhc_small.load(std::memory_order_relaxed);
    out[4] = g_ixni_runs.load(std::memory_order_relaxed);
    out[5] = g_ixni_dead.load(std::memory_order_relaxed);
    return 6;
}

/* Levers 1U-GQ / 1U-E2's live counters for GET /admin/fn4 "i11" (ds41-int11), rank 0's, cumulative, counted at issue (a
 * recorded pass when it is recorded): out[0] layers that declared GQ, [1] q_a + kv pair launches, [2] attention_output_a
 * pair launches, [3] layers that declared E2, [4] E2 split launches (part GEMV + gather), [5] their rows. Returns 6. */
/* Lever FUSE-R's live counters for GET /admin/fn4 "fuse" (ds41-fuse), rank 0's, cumulative, counted at issue (a recorded pass
 * when it is recorded): out[0] the forms declared (bit 0 form 1, bit 1 form 2), [1] MoE layers that declared one, [2] passes
 * issued in form 1, [3] in form 2. Returns 4. */
extern "C" int rad_rfuse_stats(long long* out, int n) {
    using namespace deepseek41;
    if (!out || n < 4) return 0;
    out[0] = g_sw_rf_ok.load(std::memory_order_relaxed);
    out[1] = g_rf_layers.load(std::memory_order_relaxed);
    out[2] = g_rf_count[0].load(std::memory_order_relaxed);
    out[3] = g_rf_count[1].load(std::memory_order_relaxed);
    return 4;
}
extern "C" int rad_i11_stats(long long* out, int n) {
    using namespace deepseek41;
    if (!out || n < 6) return 0;
    out[0] = g_gq_layers.load(std::memory_order_relaxed);
    out[1] = g_gq_qkv.load(std::memory_order_relaxed);
    out[2] = g_gq_oa.load(std::memory_order_relaxed);
    out[3] = g_e2_layers.load(std::memory_order_relaxed);
    out[4] = g_e2_splits.load(std::memory_order_relaxed);
    out[5] = g_e2_rows.load(std::memory_order_relaxed);
    return 6;
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
    if (step < 0) return -1;
    PskPlan& g_psk_plan = psk_slot(step);   /* (this step's slot) */
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

/* Lever TI (ds41-pf2, fn4 sched_tail_inner): the source of the Engram CONV window at a row inside a trunk step, for a
 * snapshot the engine saves there (Engine::save_checkpoints). rank -1 is the engine's probe: 1 = a snapshot inside a step
 * is whole here -- the compressor tails (the LINEAR groups, kv_comp_tail_r*) at lever G's tail grid are never read by the
 * chunk a resume starts there (the grid is a multiple of every compress ratio: the chunk from there compresses its own
 * rows only, as lever PSK-CR's replay relies on), and the Engram window (kv_engram_tokens, CONV: ngram_ids leaves a
 * prefill's last ngram - 1 mapped ids at offset zero) is read back from the step's mapped ids. rank >= 0: that rank's
 * mapped ids of its last issued trunk step (device, one i32 a row) and the window's ids; 0 when none is known. */
extern "C" int rad_ckpt_conv_src(int rank, const void** ids, int* ctx) {
    using namespace deepseek41;
    if (ids) *ids = nullptr;
    if (ctx) *ctx = 0;
    if (rank < 0) return g_ti_nocap.load(std::memory_order_acquire) ? 0 : 1;
    if (rank >= kTiRanks || !ids || !ctx) return 0;
    *ids = g_ti_ctok[rank].load(std::memory_order_acquire);
    *ctx = g_ti_ctx.load(std::memory_order_acquire);
    return *ids && *ctx > 0 ? 1 : 0;
}

RAD_ARCH_PROBE(deepseek41)
RAD_ARCH_CHAT_FORMAT(deepseek41)
RAD_ARCH_PLUGIN(deepseek41, "deepseek41", "", "0.1.0",
                "DeepSeek-V4.1: hyper-connection streams, window + sparse compressed latent "
                "attention, sqrt(softplus) MoE, Engram")
#endif
