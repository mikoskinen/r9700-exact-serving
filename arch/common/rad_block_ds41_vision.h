/* rad_block_ds41_vision.h -- DeepSeek-V4.1's vision tower and aligner (the release's
 * inference/vision.py), served by deepseek41.
 *
 * What it computes for one picture of n_vit_h x n_vit_w patches (image_processor.py's load_image):
 *
 *   x  = patches @ Wp^T + bp                     PatchEmbed: a patch row is (channel, y, x), 588 wide
 *   32 times:
 *     x += wo(attn(rope(q), rope(k), v)) + bo, q | k | v = wqkv(RMS(x)) + bqkv
 *                                                full attention over the picture's patches; the 2-D
 *                                                rotary is `rope`'s axial mode over NeoX pairs
 *                                                (i, i + 32) of the 64-wide head, the row on the
 *                                                first 16 frequencies and the column on the next 16,
 *                                                each theta^(-j / 16) (get_vision_cos_sin)
 *     x += w2(silu(g) * u), g | u = w1(RMS(x))   no biases, no clamp
 *   y  = RMS(x)                                  vision.norm (every RMSNorm eps 1e-6, its gain f32
 *                                                in the reference, bf16 values in the checkpoint)
 *   the Aligner: y viewed [n_vit_h, n_vit_w, 1024], zero-padded right and bottom to multiples of 3,
 *   F.unfold 3 x 3 stride 3 (columns (channel, kernel row, kernel column)), then
 *   rows = w2(gelu(w1 . unfold + b1)) + b2       exact GELU
 *
 * and merge_image_embeddings: the language model reads [image_start] ([row] * w + [image_newline])
 * * h [image_end], the three learned rows of the checkpoint around the aligner's rows.
 *
 * How the engine stages it (core/mm Family::DeepSeek41): a picture is one segment of 9 patches a
 * span row, so the encoder's output is the whole span, one row a placeholder. The real patches
 * come first, raster order; the rest of the segment is zero patches. Row-wise ops run over the
 * whole pass, attention over each segment's real patches only (cu {0, n_vit_h * n_vit_w} in
 * plane 3), so a pad patch is never a key. After the final norm a gather (plane 2) writes each
 * span row's 3 x 3 taps as 9 consecutive rows, zero where the aligner pads or the row is a start,
 * newline or end; conv2d_unfold over a 3-wide view (GLM-5.3's trick) turns them into the unfold's
 * column order; the aligner runs over every span row; then the start, newline and end rows are
 * overwritten with the learned rows (embed_lookup over a [3, 5120] table, scatter_rows).
 *
 * The patch embedding is conv2d with a 1 x 1 kernel: libr4d's biased GEMM wants K a multiple of
 * 8 and the patch is 588 wide.
 *
 * Every rank runs the whole tower with replicated weights, RAD_ACCESS_PER_REQUEST (so
 * --vision-tower-placement ram pins it in the host pool). Activations are sized by the pass
 * (RadBuildCtx::max_enc_patches), except in a sizing declare (shape_probe). */
#ifndef RAD_BLOCK_DS41_VISION_H
#define RAD_BLOCK_DS41_VISION_H

#include "rad_arch.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct Ds41Vision {
    int64_t depth = 0, width = 0, heads = 0, head_dim = 0, ff = 0, patch = 0, patch_dim = 0;
    int64_t ratio = 0, mm = 0, n_embd = 0, P = 0, R = 0;
    double  eps = 1e-6, theta = 10000.0;
    bool    on = false;

    struct Layer {
        rad_weight n1 = 0, qkv_w = 0, qkv_b = 0, wo_w = 0, wo_b = 0, n2 = 0, w1 = 0, w2 = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_rope = 0, op_attn = 0, op_wo = 0, op_n2 = 0, op_w1 = 0;
        rad_op op_act = 0, op_w2 = 0, op_res = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, b_patch = 0, w_norm = 0, a_w1 = 0, a_b1 = 0, a_w2 = 0, a_b2 = 0;
    rad_weight w_rows = 0;   /* [3, n_embd]: image_start, image_newline, image_end */
    rad_op op_patch = 0, op_norm = 0, op_gather = 0, op_unfold = 0, op_a1 = 0, op_gelu = 0;
    rad_op op_a2 = 0, op_spec = 0, op_spec_put = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_gu = 0, b_ff = 0, b_mo = 0, b_post = 0;
    rad_buf b_g = 0, b_unf = 0, b_a1 = 0, b_a2 = 0, b_out = 0, b_sp = 0;
    char    sec[32] = {};

    /* Whether the container states DeepSeek-V4.1's tower. */
    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "vision_config.model_type", "");
        return t && !std::strcmp(t, "deepseek_v41_vision");
    }
    /* A span row's patches: the aligner's 3 x 3. */
    static int64_t merge_of(const RadModelMeta* m) {
        const int64_t r = rad_meta_geti(m, "vision_config.downsample_ratio", 0);
        return r > 0 ? r * r : 0;
    }

    /* One segment of a pass, from its real grid (RadBatch::enc_seg_h / _w). */
    struct Seg { int64_t nvh, nvw, h, w, n_tok, P, real, n_sp, cu, r0; };
    static Seg seg_of(int64_t nvh, int64_t nvw, int64_t ratio) {
        Seg s{};
        s.nvh = nvh; s.nvw = nvw;
        s.h = (nvh + ratio - 1) / ratio;
        s.w = (nvw + ratio - 1) / ratio;
        s.n_tok = s.h * (s.w + 1) + 2;
        s.P = ratio * ratio * s.n_tok;
        s.real = nvh * nvw;
        s.n_sp = s.h + 2;
        return s;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int Ds41Vision::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                               Names& nm, int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;
    depth    = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
    width    = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads    = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
    ff       = rad_meta_geti(m, "vision_config.intermediate_size", 0);
    patch    = rad_meta_geti(m, "vision_config.patch_size", 0);
    ratio    = rad_meta_geti(m, "vision_config.downsample_ratio", 0);
    theta    = rad_meta_getf(m, "vision_config.rope_theta", 10000.0);
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || ratio <= 0 ||
        width % heads || (width / heads) % 4 || !(theta > 0)) {
        fprintf(stderr, "radiance: deepseek41: the container's vision_config is incomplete (layers %lld, "
                        "hidden %lld, heads %lld, intermediate %lld, patch %lld, downsample %lld)\n",
                (long long)depth, (long long)width, (long long)heads, (long long)ff,
                (long long)patch, (long long)ratio);
        return RAD_E_INVAL;
    }
    n_embd    = model_embd;
    head_dim  = width / heads;
    patch_dim = 3 * patch * patch;
    mm        = ratio * ratio;
    /* get_vision_cos_sin(dim = head_dim / 2): head_dim / 4 frequencies an axis */
    std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(head_dim / 4), (long long)(head_dim / 4));
    P = ctx->max_enc_patches;
    const int64_t max_seg = P;
    if (ctx->shape_probe) {
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm * ratio, r / (mm * ratio) * (mm * ratio));
    }
    /* whole span rows, and the unfold's 3-wide view needs P / 3 whole */
    if (P % mm) return RAD_E_SHAPE;
    R = P / mm;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    auto copy = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                    RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    w_patch = copy(nm.f("v.patch.weight"), nm.ckpt("vision.patch_embed.proj.weight"),
                   { width, patch_dim }, grp_model());
    b_patch = copy(nm.f("v.patch.bias"), nm.ckpt("vision.patch_embed.proj.bias"), { width },
                   grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        auto src = [&](const char* s) { return nm.ckpt("vision.blocks.%d.%s", li, s); };
        auto dn  = [&](const char* s) { return nm.f("v.blk.%d.%s", li, s); };
        l.n1    = copy(dn("norm1.weight"), src("norm1.weight"), { width }, g);
        l.qkv_w = copy(dn("wqkv.weight"), src("attn.wqkv.weight"), { 3 * width, width }, g);
        l.qkv_b = copy(dn("wqkv.bias"), src("attn.wqkv.bias"), { 3 * width }, g);
        l.wo_w  = copy(dn("wo.weight"), src("attn.wo.weight"), { width, width }, g);
        l.wo_b  = copy(dn("wo.bias"), src("attn.wo.bias"), { width }, g);
        l.n2    = copy(dn("norm2.weight"), src("norm2.weight"), { width }, g);
        /* w1 is the gate and the up projection stacked, gate first (MLP.forward's chunk(2)) */
        l.w1    = copy(dn("w1.weight"), src("mlp.w1.weight"), { 2 * ff, width }, g);
        l.w2    = copy(dn("w2.weight"), src("mlp.w2.weight"), { width, ff }, g);
        if (!l.n1 || !l.qkv_w || !l.qkv_b || !l.wo_w || !l.wo_b || !l.n2 || !l.w1 || !l.w2)
            return RAD_E_INVAL;
    }
    w_norm = copy(nm.f("v.norm.weight"), nm.ckpt("vision.norm.weight"), { width }, grp_model());
    a_w1   = copy(nm.f("v.aligner.w1.weight"), nm.ckpt("aligner.w1.weight"), { n_embd, mm * width },
                  grp_model());
    a_b1   = copy(nm.f("v.aligner.w1.bias"), nm.ckpt("aligner.w1.bias"), { n_embd }, grp_model());
    a_w2   = copy(nm.f("v.aligner.w2.weight"), nm.ckpt("aligner.w2.weight"), { n_embd, n_embd },
                  grp_model());
    a_b2   = copy(nm.f("v.aligner.w2.bias"), nm.ckpt("aligner.w2.bias"), { n_embd }, grp_model());
    /* The span's three learned rows as one table, in the order the processor numbers them. */
    {
        const char* tn = nm.f("v.span_rows.weight");
        if (map_concat(b, tn, { nm.ckpt("image_start"), nm.ckpt("image_newline"),
                                nm.ckpt("image_end") }) < 0)
            return RAD_E_INVAL;
        w_rows = decl_w(b, tn, RAD_BF16, { 3, n_embd }, ACC, RAD_SHARD_NONE, grp_model());
    }
    if (!w_patch || !b_patch || !w_norm || !a_w1 || !a_b1 || !a_w2 || !a_b2 || !w_rows)
        return RAD_E_INVAL;

    b_x    = decl_b(b, nm.f("v.x"),    RAD_BF16, { P, width });
    b_h    = decl_b(b, nm.f("v.h"),    RAD_BF16, { P, width });
    b_qkv  = decl_b(b, nm.f("v.qkv"),  RAD_BF16, { P, 3 * width });
    b_att  = decl_b(b, nm.f("v.att"),  RAD_BF16, { P, width });
    b_gu   = decl_b(b, nm.f("v.gu"),   RAD_BF16, { P, 2 * ff });
    b_ff   = decl_b(b, nm.f("v.ff"),   RAD_BF16, { P, ff });
    b_mo   = decl_b(b, nm.f("v.mo"),   RAD_BF16, { P, width });
    b_post = decl_b(b, nm.f("v.post"), RAD_BF16, { P, width });
    /* the gathered taps, declared ratio patches a row: conv2d_unfold reads it so */
    b_g    = decl_b(b, nm.f("v.taps"), RAD_BF16, { P / ratio, ratio * width });
    b_unf  = decl_b(b, nm.f("v.unf"),  RAD_BF16, { R, mm * width });
    b_a1   = decl_b(b, nm.f("v.a1"),   RAD_BF16, { R, n_embd });
    b_a2   = decl_b(b, nm.f("v.a2"),   RAD_BF16, { R, n_embd });
    b_out  = decl_b(b, nm.f("v.out"),  RAD_BF16, { R, n_embd });
    b_sp   = decl_b(b, nm.f("v.span_sp"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_qkv || !b_att || !b_gu || !b_ff || !b_mo || !b_post || !b_g || !b_unf ||
        !b_a1 || !b_a2 || !b_out || !b_sp)
        return RAD_E_INVAL;

    auto gemm_b = [&](int64_t rows, int64_t N, int64_t K, rad_weight w, rad_weight bw) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                      RAD_WEIGHTS(w, bw));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) -> rad_op {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto rms = [&](int64_t rows, rad_weight w) -> rad_op {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                 RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };

    /* PatchEmbed as a 1 x 1 convolution: one time row a patch, width 1, 588 channels in */
    op_patch = rw(b, RAD_OP(b, "conv2d",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("width", 1),
                                       RAD_INT("cin", patch_dim), RAD_INT("cout", width),
                                       RAD_INT("kh", 1), RAD_INT("kw", 1), RAD_INT("stride", 1),
                                       RAD_INT("pad", 0), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_patch, b_patch)),
                  {}, { b_x });
    for (Layer& l : L) {
        l.op_n1   = rw(b, rms(P, l.n1), { b_x }, { b_h });
        l.op_qkv  = rw(b, gemm_b(P, 3 * width, width, l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.op_rope = rw(b, RAD_OP(b, "rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                            RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                            RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                                            RAD_STR("mode", "axial"),
                                            RAD_INT("rotary_dim", head_dim),
                                            RAD_STR("sections", sec)),
                                 RAD_NOWEIGHTS),
                       { b_qkv }, { b_qkv });
        l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                            RAD_INT("max_seqlen", max_seg)),
                                 RAD_NOWEIGHTS),
                       { b_qkv }, { b_att });
        l.op_wo   = rw(b, gemm_b(P, width, width, l.wo_w, l.wo_b), { b_att, b_x }, { b_x });
        l.op_n2   = rw(b, rms(P, l.n2), { b_x }, { b_h });
        l.op_w1   = rw(b, gemm(P, 2 * ff, width, l.w1), { b_h }, { b_gu });
        l.op_act  = rw(b, RAD_OP(b, "silu_mul",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { b_gu }, { b_ff });
        l.op_w2   = rw(b, gemm(P, width, ff, l.w2), { b_ff }, { b_mo });
        l.op_res  = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { b_x, b_mo }, { b_x });
        if (!l.op_n1 || !l.op_qkv || !l.op_rope || !l.op_attn || !l.op_wo || !l.op_n2 || !l.op_w1 ||
            !l.op_act || !l.op_w2 || !l.op_res)
            return RAD_E_INVAL;
    }
    op_norm   = rw(b, rms(P, w_norm), { b_x }, { b_post });
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { b_post }, { b_g });
    /* 3 patches a time row, a 3 x 3 kernel at stride 3: one row a span row, columns (channel,
     * kernel row, kernel column), F.unfold's */
    op_unfold = rw(b, RAD_OP(b, "conv2d_unfold",
                             RAD_PARAMS(RAD_RANGE("M", 1, P / ratio), RAD_INT("width", ratio),
                                        RAD_INT("cin", width), RAD_INT("kh", ratio),
                                        RAD_INT("kw", ratio), RAD_INT("stride", ratio),
                                        RAD_INT("pad", 0), RAD_INT("cu_div", ratio),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { b_g }, { b_unf });
    op_a1   = rw(b, gemm_b(R, n_embd, mm * width, a_w1, a_b1), { b_unf }, { b_a1 });
    op_gelu = rw(b, RAD_OP(b, "gelu",
                           RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 { b_a1 }, { b_a2 });
    op_a2   = rw(b, gemm_b(R, n_embd, n_embd, a_w2, a_b2), { b_a2 }, { b_out });
    op_spec = rw(b, RAD_OP(b, "embed_lookup",
                           RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n_embd", n_embd),
                                      RAD_INT("n_vocab", 3), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w_rows)),
                 {}, { b_sp });
    op_spec_put = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { b_sp, b_out }, { b_out });
    if (!op_patch || !op_norm || !op_gather || !op_unfold || !op_a1 || !op_gelu || !op_a2 ||
        !op_spec || !op_spec_put)
        return RAD_E_INVAL;

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch %lld x %lld, "
                    "aligner %lld x %lld -> %lld; %lld patches a pass -> %lld span rows",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)patch, (long long)patch, (long long)ratio,
                 (long long)ratio, (long long)n_embd, (long long)P, (long long)R);
    }
    on = true;
    return RAD_OK;
}

inline void Ds41Vision::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    if (!on || Pn <= 0 || Pn > P || Pn % mm || S <= 0 || !batch->enc_seg_h || !batch->enc_seg_w) {
        rad_step_fail(c, "deepseek41: an encoder pass its vision tower cannot serve: no tower, more "
                         "patches than declared, not whole span rows, or no segment grids");
        return;
    }
    /* The segments' layout from their real grids (core/mm ds41_span_planes). */
    std::vector<Seg> sg((size_t)S);
    int64_t at = 0;
    for (int64_t s = 0; s < S; ++s) {
        Seg q = seg_of(batch->enc_seg_h[s], batch->enc_seg_w[s], ratio);
        q.cu = at;
        q.r0 = at / mm;
        at += q.P;
        sg[(size_t)s] = q;
    }
    if (at != Pn) {
        rad_step_fail(c, "deepseek41: an encoder pass whose segments are not DeepSeek-V4.1 picture spans "
                         "(the staged patches and the grids disagree)");
        return;
    }
    const int64_t Rn = Pn / mm;
    const int32_t* plane2 = batch->enc_coord + 2 * Pn;
    const int32_t* plane3 = batch->enc_coord + 3 * Pn;
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand rc  = praw2(batch->enc_coord, RAD_I32, 2, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), cu, RAD_NONE, brows(b_x, Pn), RAD_W(b_patch));
    for (const Layer& l : L) {
        RAD_ISSUE_N(c, l.op_n1, Pn, brows(b_x, Pn), RAD_W(l.n1), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.op_qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.op_rope, Pn, brows(b_qkv, Pn), rc);
        /* each picture's real patches attend to each other only */
        for (const Seg& q : sg)
            RAD_ISSUE_N(c, l.op_attn, q.real, bcol_at(b_qkv, q.cu, 3 * width, 0, width, q.real),
                        bcol_at(b_qkv, q.cu, 3 * width, width, width, q.real),
                        bcol_at(b_qkv, q.cu, 3 * width, 2 * width, width, q.real),
                        praw(plane3 + q.cu + 2 * q.n_sp, RAD_I32, 2),
                        brow_slice(b_att, q.cu, q.real, width));
        RAD_ISSUE_N(c, l.op_wo, Pn, brows(b_att, Pn), RAD_W(l.wo_w), RAD_W(l.wo_b), brows(b_x, Pn),
                    brows(b_x, Pn));
        RAD_ISSUE_N(c, l.op_n2, Pn, brows(b_x, Pn), RAD_W(l.n2), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.op_w1, Pn, brows(b_h, Pn), RAD_W(l.w1), brows(b_gu, Pn));
        RAD_ISSUE_N(c, l.op_act, Pn, brows(b_gu, Pn), brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.op_w2, Pn, brows(b_ff, Pn), RAD_W(l.w2), brows(b_mo, Pn));
        RAD_ISSUE_N(c, l.op_res, Pn, brows(b_x, Pn), brows(b_mo, Pn), brows(b_x, Pn));
    }
    RAD_ISSUE_N(c, op_norm, Pn, brows(b_x, Pn), RAD_W(w_norm), brows(b_post, Pn));
    /* each span row's taps, 9 consecutive rows, zero where padded or not an image row */
    for (const Seg& q : sg)
        RAD_ISSUE_N(c, op_gather, q.P, brow_slice(b_post, q.cu, q.P, width),
                    praw(plane2 + q.cu, RAD_I32, q.P),
                    brow_slice(b_g, q.cu / ratio, q.P / ratio, ratio * width));
    RAD_ISSUE_N(c, op_unfold, Pn / ratio, brows(b_g, Pn / ratio), cu, RAD_NONE, brows(b_unf, Rn));
    RAD_ISSUE_N(c, op_a1, Rn, brows(b_unf, Rn), RAD_W(a_w1), RAD_W(a_b1), RAD_NONE, brows(b_a1, Rn));
    RAD_ISSUE_N(c, op_gelu, Rn, brows(b_a1, Rn), brows(b_a2, Rn));
    RAD_ISSUE_N(c, op_a2, Rn, brows(b_a2, Rn), RAD_W(a_w2), RAD_W(a_b2), RAD_NONE, brows(b_out, Rn));
    /* the start, newline and end rows: the learned rows over the aligner's */
    for (const Seg& q : sg) {
        RAD_ISSUE_N(c, op_spec, q.n_sp, praw(plane3 + q.cu, RAD_I32, q.n_sp), RAD_W(w_rows),
                    brows(b_sp, q.n_sp));
        RAD_ISSUE_N(c, op_spec_put, q.n_sp, brows(b_sp, q.n_sp),
                    praw(plane3 + q.cu + q.n_sp, RAD_I32, q.n_sp),
                    brow_slice(b_out, q.r0, q.n_tok, n_embd));
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DS41_VISION_H */
