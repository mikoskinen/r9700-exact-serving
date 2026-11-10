/* r4d_ds41_sparse.inc -- DeepSeek-V4.1's sparse ops in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose. */
#if defined(R4D_DS41_SPECS)

/* libref's schemas (libref/ref_registry.cpp), field for field. */
static const RadParamSpec pDs41AttnSparseSink[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"), P_STRO("latent_format") };
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
    P_INT("rot"), P_INT("off"), P_STR("mode"), P_INT("inverse"), P_INT("block"), P_STR("format"),
    P_STR("block_scale") };
static const RadOperandSpec oDs41AttnSparseSinkRt[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "cos_sin", RAD_OPD_IN, 0 },
};
static const RadParamSpec pDs41IndexScore[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("ratio"), P_INT("width"), P_F64("wscale"), P_STR("dtype"), P_INTO("row_part"),
    P_INTO("block"), P_STRO("key_format") };
static const RadOperandSpec oDs41IndexScore[] = {
    { "q", RAD_OPD_IN, 0 },  { "w", RAD_OPD_IN, 0 },         { "index_cache", RAD_OPD_IN, 0 },
    { "bt", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "score", RAD_OPD_OUT, 0 }, { "keep", RAD_OPD_IN, 1 },
};
/* The tree verify's variants (lever T1): the op's operands, then the row table and the in-tree entries. */
static const RadOperandSpec oDs41AttnSparseSinkTree[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },       { "tree", RAD_OPD_IN, 0 },
    { "tree_cache", RAD_OPD_IN, 1 },
};
static const RadOperandSpec oDs41IndexScoreTree[] = {
    { "q", RAD_OPD_IN, 0 },  { "w", RAD_OPD_IN, 0 },         { "index_cache", RAD_OPD_IN, 0 },
    { "bt", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "score", RAD_OPD_OUT, 0 }, { "keep", RAD_OPD_IN, 1 },  { "tree", RAD_OPD_IN, 0 },
    { "tree_cache", RAD_OPD_IN, 1 },
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
       "quant_roundtrip(fp4_e2m1, e4m3, 16) values."),
SCHEMA("attn_sparse_sink_rope_roundtrip", pDs41AttnSparseSinkRt, oDs41AttnSparseSinkRt,
       "attn_sparse_sink (operands q .. out, its parameters), then rope_rows_roundtrip over out (n_head = "
       "heads, head_dim = dim, rot, off, mode, inverse, block, format, scale = block_scale) with cos_sin at "
       "the attention's positions: out as the two ops leave it. DeepSeek-V4.1's attention output, "
       "de-rotated and quantise-dequantised before wo_a."),
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
SCHEMA("attn_sparse_sink_tree", pDs41AttnSparseSink, oDs41AttnSparseSinkTree,
       "attn_sparse_sink over a tree verify: a tree row (tree_rows' table) of sequence s, anchor A = "
       "positions[root], reads its window table from first = A + rows(s) - window_used[s] and a window "
       "position j >= A at the slot position A + (its ancestor at depth j - A) - root; at depth >= 1 a "
       "compressed entry c completing on its path is, at ratio 1 (c >= A + 1), the entry at its "
       "ancestor's slot position, at ratio > 1 ((c + 1) ratio - 1 >= A + 1) block `ancestor` of "
       "tree_cache ([rows, 1, ratio, slots / ratio], the comp cache's format). Every other key and row "
       "is attn_sparse_sink's, bit for bit (the chain the row's path is)."),
SCHEMA("index_score_tree", pDs41IndexScore, oDs41IndexScoreTree,
       "index_score over a tree verify: a tree row's entry j completing on its path reads its key as "
       "attn_sparse_sink_tree reads that entry (ratio 1: at its ancestor's slot position; ratio > 1: "
       "block `ancestor` of tree_cache, the index cache's format); everything else is index_score's."),
SCHEMA("index_candidates", pDs41IndexCand, oDs41IndexCand,
       "Blocks of `block` positions scored by their maximum, the newest reachable block pinned to "
       "+inf, the topk_blocks highest (ties to the lower block) kept unless -inf; keep is a bit a "
       "block, i32 [M, ceil(ceil(width/block)/32)], bit b%32 of word b/32 set when block b is kept."),
SCHEMA("index_topk", pDs41IndexTopk, oDs41IndexTopk,
       "The topk highest-scoring passed positions (j < (pos+1)/ratio), inside keep when given "
       "(index_candidates' bits over blocks of `block`), never -inf, ties to the lower position, in "
       "ascending order, then -1."),

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
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink, nullptr, nullptr, nullptr,
  r4d_shape_attn_sparse_sink },
{ "ds41_attn_sparse_sink_rope_rt_d512_bf16", "attn_sparse_sink_rope_roundtrip", "attn",
  "attn_sparse_sink with rope_rows_roundtrip of its output: up to 48 rows the decode band's kernel with the "
  "inverse rope and the round trip in its epilogue (one launch, the two ops' bytes); taller passes, and a "
  "geometry the epilogue does not take, the two ops' launches",
  "attn_sparse_sink's; the fused epilogue for gptj, off even, block 32 (a half-wave's column pairs)",
  "bf16 q, caches and out; f32 sinks and cos_sin; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSinkRt), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_rope_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_rope_roundtrip, nullptr, r4d_fuse_attn_sparse_sink_rope_roundtrip },
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
{ "ds41_attn_sparse_sink_tree_d512_bf16", "attn_sparse_sink_tree", "attn",
  "attn_sparse_sink over a tree verify's rows (lever T1): libref's key order and roundings at every "
  "height (the scalar forms only), a tree row's keys through its path, so the chain its path is, bit "
  "for bit",
  "attn_sparse_sink's scalar forms (decode band up to 48 rows, the head-group kernel past it); the key "
  "list read through the row table",
  "bf16 q, caches, in-tree entries and out; f32 sinks; i32 tables, positions, selections, row table",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSink), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink_tree, nullptr, nullptr, nullptr,
  r4d_shape_attn_sparse_sink_tree },
{ "ds41_index_score_tree_d128_bf16", "index_score_tree", "attn",
  "index_score over a tree verify's rows (lever T1): the scalar kernels at every height (libref's "
  "roundings and order), a tree row's in-path entries read through the row table",
  "index_score's scalar and kept kernels",
  "bf16 q, w, cache and in-tree keys; i32 tables, positions and row table; f32 score",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexScore), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_score_tree, nullptr, nullptr, nullptr,
  r4d_shape_index_score_tree },
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

#endif
