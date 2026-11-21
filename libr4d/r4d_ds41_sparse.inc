/* r4d_ds41_sparse.inc -- DeepSeek-V4.1's sparse ops in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose. */
#if defined(R4D_DS41_SPECS)

/* libref's schemas (libref/ref_registry.cpp), field for field. */
static const RadParamSpec pDs41AttnSparseSink[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INTO("wm_from") };   /* wm_from: lever DAW (attn_body) */
static const RadOperandSpec oDs41AttnSparseSink[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },
};
/* attn_sparse_sink + rope_rows_roundtrip over its out (lever K1 fold d): the attention's parameters, then
 * rope_rows' rot / off / mode / inverse (n_head = heads, head_dim = dim) and quant_roundtrip's block / format,
 * its scale named block_scale; the attention's operands, then the rope's cos_sin. */
static const RadParamSpec pDs41AttnSparseSinkRt[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INT("rot"), P_INT("off"), P_STR("mode"), P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("block_scale") };
static const RadOperandSpec oDs41AttnSparseSinkRt[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
};
/* lever DQR (DRAFTER track 21.11.2026 round 69; device-only, no libref schema): q's forward rope, attn_sparse_sink, then the
 * inverse rope + round trip of its out, one launch -- the attention's parameters (wm_from: lever DAW, the WMMA form at every
 * height; the op is served only there), then rope_rows' rot / off / mode (q forward, out inverse: no `inverse`) and
 * quant_roundtrip's block / format / block_scale; the attention's operands, then cos_sin and the rows' own positions. */
static const RadParamSpec pDs41AttnSparseSinkDqr[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_INT("wm_from"), P_INT("rot"),
    P_INT("off"), P_STR("mode"), P_INT("block"), P_STR("format"), P_STR("block_scale"), P_INT("out_rope") };
static const RadOperandSpec oDs41AttnSparseSinkDqr[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
    { "row_positions", RAD_OPD_IN, 0 },
};
/* Levers DKV-U / DKV-G (RADIANCE_DS41_LATUNPACK / RADIANCE_DS41_LATGATHER; device-only, no libref schema): the packed
 * latent's entries unpacked (latent_unpack: the plan's rows of a pass; latent_gather: a decode-band pass's selections),
 * and the two attentions with the rows they read appended. */
/* the unpacked attentions' parameters: attn_sparse_sink's (resp. the fold's), then plan_full (optional; 1: the caller knows
 * the plan holds every entry the pass may select -- the LDS-free form, the plain form's occupancy; any value gives the
 * same output) */
static const RadParamSpec pDs41AttnSparseSinkLu[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INTO("plan_full") };
static const RadParamSpec pDs41AttnSparseSinkRtLu[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INT("rot"), P_INT("off"), P_STR("mode"), P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("block_scale"), P_INTO("plan_full") };
/* tp8 loop D25 step 3: the gathered attentions' parameters = the plain ones + band (optional; 49 .. 64: a pass of at most `band` rows
 * runs the scalar decode kernels -- each row its alone bytes -- instead of the WMMA form above 48 rows; absent: 48, today) */
static const RadParamSpec pDs41AttnSparseSinkG[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INTO("band") };
static const RadParamSpec pDs41AttnSparseSinkRtG[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format"),
    P_STRO("window_format"), P_INT("rot"), P_INT("off"), P_STR("mode"), P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("block_scale"), P_INTO("band") };
static const RadParamSpec pDs41LatUnpack[] = { P_INT("M"), P_INT("dim"), P_INT("ratio"), P_STR("latent_format") };
static const RadOperandSpec oDs41LatUnpack[] = {
    { "comp_cache", RAD_OPD_IN, 0 }, { "comp_bt", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "unpacked", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41LatGather[] = { P_INT("M"), P_INT("dim"), P_INT("ratio"), P_INT("topk"),
    P_STR("latent_format") };
static const RadOperandSpec oDs41LatGather[] = {
    { "comp_cache", RAD_OPD_IN, 0 }, { "comp_bt", RAD_OPD_IN, 0 }, { "sel", RAD_OPD_IN, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "gathered", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec oDs41AttnSparseSinkLu[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "unpacked", RAD_OPD_IN, 0 },
};
static const RadOperandSpec oDs41AttnSparseSinkRtLu[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
    { "unpacked", RAD_OPD_IN, 0 },
};
static const RadOperandSpec oDs41AttnSparseSinkGth[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "gathered", RAD_OPD_IN, 0 },
};
static const RadOperandSpec oDs41AttnSparseSinkRtGth[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
    { "gathered", RAD_OPD_IN, 0 },
};
static const RadParamSpec pDs41IndexScore[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("ratio"), P_INT("width"), P_F64("wscale"), P_STR("dtype"), P_INTO("row_part"),
    P_INTO("block"), P_STRO("key_format") };
static const RadOperandSpec oDs41IndexScore[] = {
    { "q", RAD_OPD_IN, 0 },  { "w", RAD_OPD_IN, 0 },         { "index_cache", RAD_OPD_IN, 0 },
    { "bt", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "score", RAD_OPD_OUT, 0 }, { "keep", RAD_OPD_IN, 1 },
};
static const RadParamSpec pDs41IndexCand[] = { P_INT("M"), P_INT("width"), P_INT("block"),
    P_INT("topk_blocks"), P_INT("ratio") };
static const RadOperandSpec oDs41IndexCand[] = {
    { "score", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "keep", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41IndexTopk[] = { P_INT("M"), P_INT("width"), P_INT("topk"), P_INT("ratio"),
    P_INTO("block") };
static const RadOperandSpec oDs41IndexTopk[] = {
    { "score", RAD_OPD_IN, 0 }, { "keep", RAD_OPD_IN, 1 }, { "positions", RAD_OPD_IN, 0 },
    { "sel", RAD_OPD_OUT, 0 },
};

/* r4d_ds41_sparse.hip's launch checks. The latent is 512 wide (16-byte loads of 512/ratio-wide
 * compressed slots: ratio <= 64), at most 1024 window and 2048 selected keys a row (the per-row
 * key list and scores live in LDS). M is any: the grid loops over rows. */
static const RadConstraint cDs41AttnSparseSink[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 0),   C_LE("ratio", 64), C_GE("topk", 0),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41AttnSparseSinkRt[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 0),   C_LE("ratio", 64), C_GE("topk", 0),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"), C_GE("rot", 2), C_DIV("rot", 2), C_GE("off", 0), C_GE("inverse", 0),
    C_LE("inverse", 1), C_IN("mode", "gptj neox"), C_GE("block", 1), C_IN("format", "fp8_e4m3 fp4_e2m1"),
    C_IN("block_scale", "ue8m0 e4m3 f32"),
};
static const RadConstraint cDs41AttnSparseSinkDqr[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_EQ("ratio", 0),   C_EQ("topk", 0),  C_IN("dtype", "bf16"), C_GE("wm_from", 1), C_GE("rot", 2), C_DIV("rot", 2),
    C_GE("off", 0), C_DIV("off", 2), C_IN("mode", "gptj"), C_EQ("block", 32), C_IN("format", "fp8_e4m3 fp4_e2m1"),
    C_IN("block_scale", "ue8m0 e4m3 f32"), C_GE("out_rope", 0), C_LE("out_rope", 1),
};
/* 128-wide index keys held in registers, a slot dim/ratio wide read 16 bytes at a time. */
static const RadConstraint cDs41IndexScore[] = {
    C_GE("M", 1),     C_GE("heads", 1), C_EQ("dim", 128),      C_GE("ratio", 1), C_LE("ratio", 16),
    C_GE("width", 1), C_LE("width", 1073741823), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41IndexCand[] = {
    C_GE("M", 1),          C_GE("width", 1), C_LE("width", 1073741823), C_GE("block", 1),
    C_GE("topk_blocks", 1), C_GE("ratio", 1),
};
static const RadConstraint cDs41IndexTopk[] = {
    C_GE("M", 1), C_GE("width", 1), C_LE("width", 1073741823), C_GE("topk", 1),
    C_LE("topk", 1073741823), C_GE("ratio", 1),
};
/* Levers DKV-U / DKV-G: the packed 512-wide latent (ratio dividing its 144 slots); the gathered attentions are the
 * decode band's (<= 48 rows), the unpacked ones any height (the plan serves the WMMA form's). */
static const RadConstraint cDs41LatUnpack[] = {
    C_GE("M", 1), C_EQ("dim", 512), C_GE("ratio", 1), C_LE("ratio", 64), C_IN("latent_format", "nvfp4"),
};
static const RadConstraint cDs41LatGather[] = {
    C_GE("M", 1), C_LE("M", 64),   /* tp8 loop D25 step 3: 64 (a depth-3 decode segment of 16 x 4); the kernel takes any height */ C_EQ("dim", 512), C_GE("ratio", 1), C_LE("ratio", 64), C_GE("topk", 1),
    C_LE("topk", 2048), C_IN("latent_format", "nvfp4"),
};
static const RadConstraint cDs41AttnSparseSinkLu[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 1),   C_LE("ratio", 64), C_GE("topk", 1),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"), C_IN("latent_format", "nvfp4"),
};
static const RadConstraint cDs41AttnSparseSinkRtLu[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 1),   C_LE("ratio", 64), C_GE("topk", 1),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"), C_IN("latent_format", "nvfp4"), C_GE("rot", 2), C_DIV("rot", 2), C_GE("off", 0),
    C_GE("inverse", 0), C_LE("inverse", 1), C_IN("mode", "gptj neox"), C_GE("block", 1),
    C_IN("format", "fp8_e4m3 fp4_e2m1"), C_IN("block_scale", "ue8m0 e4m3 f32"),
};
static const RadConstraint cDs41AttnSparseSinkGth[] = {
    C_GE("M", 1),       C_LE("M", 64),   /* tp8 loop D25 step 3: to 64 (above 48 rows the decode band only under `band`) */    C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 1),   C_LE("ratio", 64), C_GE("topk", 1),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"), C_IN("latent_format", "nvfp4"),
};
static const RadConstraint cDs41AttnSparseSinkRtGth[] = {
    C_GE("M", 1),       C_LE("M", 64),   /* tp8 loop D25 step 3 */    C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 1),   C_LE("ratio", 64), C_GE("topk", 1),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"), C_IN("latent_format", "nvfp4"), C_GE("rot", 2), C_DIV("rot", 2), C_GE("off", 0),
    C_GE("inverse", 0), C_LE("inverse", 1), C_IN("mode", "gptj neox"), C_GE("block", 1),
    C_IN("format", "fp8_e4m3 fp4_e2m1"), C_IN("block_scale", "ue8m0 e4m3 f32"),
};

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("attn_sparse_sink", pDs41AttnSparseSink, oDs41AttnSparseSink,
       "Latent attention (key = value, dim wide) with a per-head sink over the window cache's keys "
       "at [pos-window+1, pos] (the window table rebased at first = pos(last row of s) + 1 - "
       "window_used[s]: block wbt[s][(j-first)/bs], cache [blocks,1,bs,dim]) and, with ratio > 0, "
       "the selected compressed entries c (tokens [c ratio, (c+1) ratio), block cbt[s][c ratio/cbs], "
       "cache [blocks,1,cbs,dim/ratio]). f32 scores q.k*scale, max floored at -1e30, "
       "out = sum p k / (sum p + exp(sink - max)), rounded once; a row with no key is zero. "
       "latent_format nvfp4: the compressed cache holds each entry packed (entry_store format nvfp4: "
       "e2m1 codes, an e4m3 scale a block of 16, 288 bytes for 512 values), the window a multiple of "
       "32; every output is the bf16 cache's, bit for bit, where the entries are "
       "quant_roundtrip(fp4_e2m1, e4m3, 16) values. window_format fp8_e4m3: the window cache holds each key "
       "packed (window_store format fp8_e4m3: e4m3 codes, a scale byte a block of 32, 528 bytes for 512 "
       "values; a value is e4m3(code) x 2^(e-127) in f32), the window a multiple of 32 when ratio > 0; every "
       "output is the bf16 cache's, bit for bit, where the keys are quant_roundtrip(fp8_e4m3, ue8m0, 32) "
       "values window_store packed."),
SCHEMA("attn_sparse_sink_rope_roundtrip", pDs41AttnSparseSinkRt, oDs41AttnSparseSinkRt,
       "attn_sparse_sink (operands q .. out, its parameters), then rope_rows_roundtrip over out (n_head = "
       "heads, head_dim = dim, rot, off, mode, inverse, block, format, scale = block_scale) with cos_sin at "
       "the attention's positions: out as the two ops leave it. DeepSeek-V4.1's attention output, "
       "de-rotated and quantise-dequantised before wo_a."),
SCHEMA("attn_sparse_sink_dqr", pDs41AttnSparseSinkDqr, oDs41AttnSparseSinkDqr,
       "ds41_rope_rows (n_head = heads, head_dim = dim, rot, off, mode, inverse 0) over q at row_positions, then "
       "attn_sparse_sink (operands q .. out, its parameters; ratio 0: the window alone), then with out_rope 1 "
       "rope_rows_roundtrip (inverse 1, block, format, scale = block_scale) over out at row_positions with cos_sin: out as "
       "those ops leave it; q is not written. The DSpark drafter stage's query rope, attention (and output rope / round trip)."),
SCHEMA("index_score", pDs41IndexScore, oDs41IndexScore,
       "Per row and compressed position j: sum_h relu(q_h . k_j) w_h with k_j entry j of the paged "
       "index cache and w_h = round(w * wscale); the dot, relu(dot) w_h and the f32 head sum each "
       "rounded once to dtype. "
       "j >= (pos+1)/ratio or an unmapped block scores -inf. score is f32 [M, width]. With row_part "
       "p, score row i is step row p M + i (q, w, positions the step's; past the step -inf). With "
       "keep (index_candidates' bits over blocks of `block`, keep row i for score row i) only the kept "
       "blocks are scored, the rest -inf unread; a kept score is the unkept op's, bit for bit. "
       "key_format mxfp4: the cache holds each entry packed (entry_store format mxfp4: e2m1 codes, a ue8m0 "
       "scale a block of 32, 80 bytes for 128 values) and every score is the bf16 cache's, bit for bit, "
       "where the keys are quant_roundtrip(fp4_e2m1, ue8m0, 32) values."),
SCHEMA("index_candidates", pDs41IndexCand, oDs41IndexCand,
       "Blocks of `block` positions scored by their maximum, the newest reachable block pinned to "
       "+inf, the topk_blocks highest (ties to the lower block) kept unless -inf; keep is a bit a "
       "block, i32 [M, ceil(ceil(width/block)/32)], bit b%32 of word b/32 set when block b is kept."),
SCHEMA("index_topk", pDs41IndexTopk, oDs41IndexTopk,
       "The topk highest-scoring passed positions (j < (pos+1)/ratio), inside keep when given "
       "(index_candidates' bits over blocks of `block`), never -inf, ties to the lower position, in "
       "ascending order, then -1."),
SCHEMA("latent_unpack", pDs41LatUnpack, oDs41LatUnpack,
       "The packed compressed cache's entries a pass may select (latent_format nvfp4, dim 512): for each sequence s "
       "in cu order, entries [n_s - t_s, n_s) where n_s = (pos(last row of s) + 1) / ratio (at most the block "
       "table's reach) and t_s = min(n_s, what the earlier sequences left of the operand's rows x (width / dim) "
       "entries), unpacked to bf16 at consecutive dim-wide rows of `unpacked`; an unmapped entry's row is not "
       "written. A pass of more than 256 sequences writes nothing. Lever DKV-U (device-only)."),
SCHEMA("latent_gather", pDs41LatGather, oDs41LatGather,
       "Each row's selection j that attn_sparse_sink's key setup finds in the packed compressed cache (latent_format "
       "nvfp4, dim 512; comp_cache / comp_bt / cu_seqlens as that op's) unpacked to bf16 at gathered[row][j dim .. "
       "j dim + dim); the other selections' rows are not written. Lever DKV-G (device-only)."),
SCHEMA("attn_sparse_sink_unpacked", pDs41AttnSparseSinkLu, oDs41AttnSparseSinkLu,
       "attn_sparse_sink (operands q .. out, its parameters; latent_format nvfp4) whose WMMA passes read a selected "
       "entry from `unpacked` (latent_unpack's rows for the same cache, block table, positions' last rows and "
       "cu_seqlens) where its 16-key tile is held there: the same out. plan_full 1: the caller knows the plan holds every "
       "entry the pass may select (the LDS-free form; any value the same out). Lever DKV-U (device-only)."),
SCHEMA("attn_sparse_sink_rope_roundtrip_unpacked", pDs41AttnSparseSinkRtLu, oDs41AttnSparseSinkRtLu,
       "attn_sparse_sink_rope_roundtrip (operands q .. cos_sin, its parameters) with the attention's WMMA passes "
       "reading `unpacked` as attn_sparse_sink_unpacked: the same out. Lever DKV-U (device-only)."),
SCHEMA("attn_sparse_sink_gathered", pDs41AttnSparseSinkG, oDs41AttnSparseSinkGth,
       "attn_sparse_sink (operands q .. out, its parameters; latent_format nvfp4) whose decode-band pass reads the "
       "selected entries from `gathered` (latent_gather's rows for the same sel, cache, block table and "
       "cu_seqlens) instead of unpacking the cache: the same out. band 49 .. 64 (optional): passes of at most band "
       "rows in the decode band (each row its own bytes, as a pass of <= 48 rows), not attn_sparse_sink's WMMA form "
       "above 48 rows. Lever DKV-G (device-only)."),
SCHEMA("attn_sparse_sink_rope_roundtrip_gathered", pDs41AttnSparseSinkRtG, oDs41AttnSparseSinkRtGth,
       "attn_sparse_sink_rope_roundtrip (operands q .. cos_sin, its parameters) with the selected entries read from "
       "`gathered`, as attn_sparse_sink_gathered (band as there): the same out. Lever DKV-G (device-only)."),

#elif defined(R4D_DS41_ROWS)

{ "ds41_attn_sparse_sink_d512_bf16", "attn_sparse_sink", "attn",
  "latent attention (key = value) over the window keys and the selected compressed entries, "
  "per-head sinks; libref's key order and roundings (glibc expf algorithm), so libref's bits, "
  "up to 48 rows; taller passes on WMMA (online softmax, P as bf16 hi + lo), not bit-exact",
  "dim 512, any heads, window <= 1024, topk <= 2048, ratio 0..64; one workgroup a (d half, head "
  "group of 2/4/8, row), the group picked by row count; > 48 rows a (16-head tile, row), four "
  "waves of 128 latent columns, 16 keys a step",
  "bf16 q, caches and out; f32 sinks; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSink), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink, r4d_rad_ds41_attn_sparse_sink_scratch, nullptr, nullptr,
  r4d_shape_attn_sparse_sink },
{ "ds41_attn_sparse_sink_rope_rt_d512_bf16", "attn_sparse_sink_rope_roundtrip", "attn",
  "attn_sparse_sink with rope_rows_roundtrip of its output: up to 48 rows the decode band's kernel with the "
  "inverse rope and the round trip in its epilogue (one launch, the two ops' bytes); taller passes, and a "
  "geometry the epilogue does not take, the two ops' launches",
  "attn_sparse_sink's; the fused epilogue for gptj, off even, block 32 (a half-wave's column pairs)",
  "bf16 q, caches and out; f32 sinks and cos_sin; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkRt), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip, r4d_rad_ds41_attn_sparse_sink_scratch, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_rope_roundtrip, nullptr, r4d_fuse_attn_sparse_sink_rope_roundtrip },
/* lever DQR (DRAFTER track 21.11.2026 round 69). No `replaces`: the arch issues it in place of the chain (deepseek41, the DSpark
 * stages; RADIANCE_DS41_DSPARK_DQR). tools/dbench/dqr holds it to the chain byte for byte and batch-invariant. */
{ "ds41_attn_sparse_sink_dqr_d512_bf16", "attn_sparse_sink_dqr", "attn",
  "the WMMA attention (ds41_attn_sparse_sink_wm_kernel, lever DAW's form) with q's rope applied to its loaded Q words and the "
  "output's inverse rope + round trip in its epilogue: one launch, the three ops' bytes",
  "dim 512, any heads, window <= 1024, ratio 0, wm_from >= 1, gptj, off even, block 32; the plain bf16 window only",
  "bf16 q, window cache and out; f32 sinks and cos_sin; i32 tables and positions",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkDqr), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_dqr, nullptr, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_dqr },
{ "ds41_index_score_d128_bf16", "index_score", "attn",
  "indexer score sum_h relu(q_h.k_j) w_h over a paged index cache, libref's roundings and order "
  "(> 48 rows: the dot on WMMA, its summation order the only change)",
  "dim 128, any heads, ratio 1..16, row_part; a thread an entry (1, 2, 4 or 8 waves an entry "
  "group by grid size), -inf past (pos+1)/ratio without reading keys; > 48 rows and <= 32 heads a "
  "wave a row over 128 entries (with keep: over 1024 entries, its unkept 16-entry tiles skipped)",
  "bf16 q, w and cache; i32 tables and positions; f32 score",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexScore), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_score, nullptr, nullptr, nullptr,
  r4d_shape_index_score },
{ "ds41_index_candidates", "index_candidates", "select",
  "two-level top-k level one: block maxima, newest block pinned, exact top topk_blocks by radix "
  "select, ties to the lower block",
  "any width and block; one 1024-thread block a row",
  "f32 score, i32 positions and keep (a bit a block, a wave's ballot a word)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexCand), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_candidates, nullptr, nullptr, nullptr,
  r4d_shape_index_candidates },
{ "ds41_index_topk", "index_topk", "select",
  "exact top-k passed positions by radix select, ties to the lower position, ascending, -1 fill",
  "any width and topk; one 1024-thread block a row, or at <= 16 rows over >= 4 chunks of 8192 "
  "positions (topk <= 2048) a chunk-parallel pass and a merge (scratch)",
  "f32 score, i32 keep (a bit a block), positions and sel",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexTopk), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_topk, r4d_rad_ds41_index_topk_scratch, nullptr, nullptr,
  r4d_shape_index_topk },
{ "ds41_latent_unpack_d512", "latent_unpack", "attn",
  "the packed latent's entries a pass may select, unpacked once into the plan's rows (lever DKV-U): a wave an "
  "entry, lane L its block L (lever NVG's gather expressions), grid stride",
  "dim 512 nvfp4, ratio <= 64, <= 256 sequences a pass",
  "packed bf16-slot cache; i32 tables, positions, cu_seqlens; bf16 unpacked",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41LatUnpack), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_latent_unpack, nullptr, nullptr, nullptr,
  r4d_shape_latent_unpack },
{ "ds41_latent_gather_d512", "latent_gather", "attn",
  "a decode-band pass's selected packed entries unpacked once (lever DKV-G): lever NVG's gather kernel as an op",
  "dim 512 nvfp4, <= 48 rows, topk <= 2048; a wave an entry, lane L its block L",
  "packed bf16-slot cache; i32 tables, selections, cu_seqlens; bf16 gathered",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41LatGather), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_latent_gather, nullptr, nullptr, nullptr,
  r4d_shape_latent_gather },
{ "ds41_attn_sparse_sink_unpacked_d512_bf16", "attn_sparse_sink_unpacked", "attn",
  "attn_sparse_sink over the packed latent whose WMMA passes read the latent_unpack plan's rows as plain bf16 tiles "
  "(a tile with an entry the plan lacks unpacks itself, by the v3 decode); the decode band as attn_sparse_sink",
  "attn_sparse_sink's, latent_format nvfp4; <= 256 sequences a pass for the plan",
  "bf16 q, caches, unpacked and out; f32 sinks; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkLu), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_unpacked, r4d_rad_ds41_attn_sparse_sink_scratch, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_unpacked },
{ "ds41_attn_sparse_sink_rope_rt_unpacked_d512_bf16", "attn_sparse_sink_rope_roundtrip_unpacked", "attn",
  "attn_sparse_sink_rope_roundtrip whose taller passes' attention reads the latent_unpack plan's rows as "
  "attn_sparse_sink_unpacked, then rope_rows_roundtrip's launch; the decode band as the fold",
  "attn_sparse_sink_rope_roundtrip's, latent_format nvfp4",
  "bf16 q, caches, unpacked and out; f32 sinks and cos_sin; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkRtLu), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_unpacked, r4d_rad_ds41_attn_sparse_sink_scratch,
  nullptr, nullptr, r4d_shape_attn_sparse_sink_rope_roundtrip_unpacked },
{ "ds41_attn_sparse_sink_gathered_d512_bf16", "attn_sparse_sink_gathered", "attn",
  "attn_sparse_sink over the packed latent with its selections already unpacked (latent_gather): the decode band's "
  "plain kernel over the gathered rows, no gather launch; libref's bits as attn_sparse_sink's",
  "attn_sparse_sink's decode band (<= 48 rows), latent_format nvfp4",
  "bf16 q, caches, gathered and out; f32 sinks; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkGth), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_gathered, r4d_rad_ds41_attn_sparse_sink_scratch, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_gathered },
{ "ds41_attn_sparse_sink_rope_rt_gathered_d512_bf16", "attn_sparse_sink_rope_roundtrip_gathered", "attn",
  "attn_sparse_sink_rope_roundtrip over the packed latent with its selections already unpacked (latent_gather): the "
  "decode band's plain kernel with the fused epilogue over the gathered rows, no gather launch",
  "attn_sparse_sink_rope_roundtrip's decode band (<= 48 rows), latent_format nvfp4",
  "bf16 q, caches, gathered and out; f32 sinks and cos_sin; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkRtGth), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip_gathered, r4d_rad_ds41_attn_sparse_sink_scratch,
  nullptr, nullptr, r4d_shape_attn_sparse_sink_rope_roundtrip_gathered },

#endif
