/* r4d_ds41_elem.inc -- DeepSeek-V4.1's elem ops in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose.
 *
 * The specs and docs are libref's (libref/ref_registry.cpp) field for field. The constraints are
 * what each launch in r4d_ds41_elem.hip accepts; a condition between parameters (n % block,
 * off + rot <= head_dim) is refused there and by libref alike. */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41TokenMap[] = { P_INT("M"), P_INT("n_vocab") };
static const RadOperandSpec oDs41TokenMap[] = {
    { "ids", RAD_OPD_IN, 0 }, { "table", RAD_OPD_WEIGHT, 0 }, { "mapped", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41TokenMap[] = { C_GE("M", 1), C_GE("n_vocab", 1) };

/* lever BI's tracer (debug): a fingerprint of each row of any operand */
static const RadParamSpec pBiRowHash[] = { P_INT("M") };
static const RadOperandSpec oBiRowHash[] = { { "x", RAD_OPD_IN, 0 }, { "h", RAD_OPD_OUT, 0 } };
static const RadConstraint cBiRowHash[] = { C_GE("M", 1) };

static const RadParamSpec pDs41StreamGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                                P_F64("gate_eps"), P_STR("dtype") };
static const RadOperandSpec oDs41StreamGate[] = {
    { "h", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 },       { "v", RAD_OPD_IN, 0 },
    { "w", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41StreamGate[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 1), C_LE("hc", 16), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41RopeRows[] = { P_INT("M"),   P_INT("n_head"), P_INT("head_dim"),
                                              P_INT("rot"), P_INT("off"),    P_STR("mode"),
                                              P_INT("inverse"), P_STR("dtype") };
static const RadOperandSpec oDs41RopeRows[] = {
    { "x", RAD_OPD_INOUT, 0 }, { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
static const RadConstraint cDs41RopeRows[] = {
    C_GE("M", 1),       C_GE("n_head", 1),   C_GE("head_dim", 2), C_GE("rot", 2),
    C_DIV("rot", 2),    C_GE("off", 0),      C_GE("inverse", 0),  C_LE("inverse", 1),
    C_IN("mode", "gptj neox"), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41QuantRoundtrip[] = { P_INT("M"),      P_INT("n"),
                                                    P_INT("block"),  P_STR("format"),
                                                    P_STR("scale"),  P_STR("dtype") };
static const RadOperandSpec oDs41QuantRoundtrip[] = { { "x", RAD_OPD_INOUT, 0 } };
/* One row a format, so each states the scales that format takes. */
static const RadConstraint cDs41QuantFp8[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block", 1), C_IN("format", "fp8_e4m3"),
    C_IN("scale", "ue8m0 f32"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41QuantFp4[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block", 1), C_IN("format", "fp4_e2m1"),
    C_IN("scale", "ue8m0 e4m3"), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41MhcMix[] = { P_INT("M"),          P_INT("n"),   P_INT("hc"),
                                            P_INT("iterations"), P_F64("eps"), P_F64("hc_eps"),
                                            P_STR("dtype") };
static const RadOperandSpec oDs41MhcMix[] = {
    { "h", RAD_OPD_IN, 0 },      { "fn", RAD_OPD_WEIGHT, 0 }, { "scale", RAD_OPD_WEIGHT, 0 },
    { "base", RAD_OPD_WEIGHT, 0 }, { "pre", RAD_OPD_OUT, 0 }, { "post", RAD_OPD_OUT, 0 },
    { "comb", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41MhcMix[] = {
    C_GE("M", 1),          C_GE("n", 1),           C_GE("hc", 2), C_LE("hc", 8),
    C_GE("iterations", 1), C_LE("iterations", 256), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41MhcCollapse[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
                                                 P_STR("dtype") };
static const RadOperandSpec oDs41MhcCollapse[] = {
    { "h", RAD_OPD_IN, 0 }, { "pre", RAD_OPD_IN, 0 }, { "x", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41MhcCollapse[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41SoftmaxPool[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                                 P_STR("dtype") };
static const RadOperandSpec oDs41SoftmaxPool[] = {
    { "stage", RAD_OPD_IN, 0 }, { "out", RAD_OPD_OUT, 0 }, { "page", RAD_OPD_IN, 1 },
};
static const RadConstraint cDs41SoftmaxPool[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("ratio", 1), C_LE("ratio", 64), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41EntryStore[] = { P_INT("M"), P_INT("n"), P_INT("block_size"),
                                                P_STR("dtype"), P_STRO("format") };
static const RadOperandSpec oDs41EntryStore[] = {
    { "x", RAD_OPD_IN, 0 }, { "page", RAD_OPD_IN, 0 }, { "cache", RAD_OPD_INOUT, 0 },
};
static const RadConstraint cDs41EntryStore[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block_size", 1), C_IN("dtype", "bf16"),
};

/* window_store (lever KVWIN): a key row packed into the fp8 window slot of a paged cache, kv_store's slot rule. */
static const RadParamSpec pDs41WindowStore[] = { P_INT("M"), P_INT("n"), P_INT("block_size"), P_STR("format"),
                                                 P_STR("dtype") };
static const RadOperandSpec oDs41WindowStore[] = {
    { "x", RAD_OPD_IN, 0 }, { "slot_mapping", RAD_OPD_IN, 0 }, { "kv_cache", RAD_OPD_INOUT, 0 },
};
static const RadConstraint cDs41WindowStore[] = {
    C_GE("M", 1), C_GE("n", 32), C_LE("n", 16384), C_DIV("n", 32), C_GE("block_size", 1),
    C_IN("format", "fp8_e4m3"), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41RouterScored[] = { P_INT("M"),           P_INT("n_expert"),
                                                  P_INT("top_k"),       P_INT("norm"),
                                                  P_F64("route_scale"), P_STR("score"),
                                                  P_STR("dtype") };
static const RadOperandSpec oDs41RouterScored[] = {
    { "logits", RAD_OPD_IN, 0 }, { "bias", RAD_OPD_WEIGHT, 0 },
    { "ids", RAD_OPD_OUT, 0 },   { "weights", RAD_OPD_OUT, 0 },
};
/* top_k <= 32: round j parks its pick on lane j of the row's wave. */
static const RadConstraint cDs41RouterScored[] = {
    C_GE("M", 1),     C_GE("n_expert", 1), C_LE("n_expert", 512),
    C_GE("top_k", 1), C_LE("top_k", 32),   C_IN("score", "sigmoid sqrtsoftplus softmax"),
    C_IN("dtype", "f32"),
};

/* router_scored_topk and moe_scatter as one op (router_sigmoid_topk_scatter's operands, r4d_rows.cpp
 * oRouterSigmoidScatter, with the score named), and the `fill` of an ungated shared arm's gate. */
static const RadParamSpec pDs41RouterScoredScatter[] = {
    P_INT("M"),     P_INT("n_expert"),     P_INTO("n_local"),  P_INT("top_k"),     P_INT("norm"),
    P_F64("route_scale"), P_STR("score"),  P_STR("dtype"),     P_INTO("expert_base"),
    P_STRO("protect") };
static const RadOperandSpec oDs41RouterScoredScatter[] = {
    { "logits", RAD_OPD_IN, 0 },        { "bias", RAD_OPD_WEIGHT, 0 },
    { "ids", RAD_OPD_OUT, 0 },          { "weights", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 },
    { "expert_count", RAD_OPD_OUT, 0 }, { "gate", RAD_OPD_OUT, 1 },
    { "inverse", RAD_OPD_OUT, 1 }
};
/* The sort's single workgroup: M * top_k <= 1024 (refused above by the launch and the shape). */
static const RadConstraint cDs41RouterScoredScatter[] = {
    C_GE("M", 1),     C_LE("M", 1024),     C_GE("n_expert", 1), C_LE("n_expert", 512),
    C_GE("top_k", 1), C_LE("top_k", 32),   C_IN("score", "sigmoid sqrtsoftplus softmax"),
    C_IN("dtype", "f32"),
};

/* lever FUSE-R (ds41-fuse): gemm_nt_f32 (the router's projection), router_scored_topk and moe_scatter in one launch, no
 * gate. The decode band: M <= 48 (the sort's one workgroup and the GEMM's decode tile forms), <= 384 experts,
 * DeepSeek's sqrtsoftplus. */
static const RadParamSpec pDs41RouterGemmScored[] = {
    P_INT("M"),     P_INT("K"),           P_INT("n_expert"), P_INTO("n_local"), P_INT("top_k"), P_INT("norm"),
    P_F64("route_scale"), P_STR("score"), P_STR("dtype"),    P_INTO("expert_base"), P_STRO("protect") };
static const RadOperandSpec oDs41RouterGemmScored[] = {
    { "a", RAD_OPD_IN, 0 },             { "b", RAD_OPD_WEIGHT, 0 },        { "bias", RAD_OPD_WEIGHT, 0 },
    { "logits", RAD_OPD_OUT, 0 },       { "ids", RAD_OPD_OUT, 0 },         { "weights", RAD_OPD_OUT, 0 },
    { "sorted_tok", RAD_OPD_OUT, 0 },   { "expert_offset", RAD_OPD_OUT, 0 }, { "expert_count", RAD_OPD_OUT, 0 },
    { "inverse", RAD_OPD_OUT, 1 } };
static const RadConstraint cDs41RouterGemmScored[] = {
    C_GE("M", 1),     C_LE("M", 48),     C_GE("K", 1),   C_LE("K", 16384), C_GE("n_expert", 1), C_LE("n_expert", 384),
    C_GE("top_k", 1), C_LE("top_k", 32), C_IN("score", "sqrtsoftplus"),    C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41EmbedPlanes[] = { P_INT("M"),     P_INT("n_embd"),
                                                 P_INT("n_vocab"), P_INT("group"),
                                                 P_STR("dtype"), P_INTO("vocab_offset"),
                                                 P_F64O("mult") };
static const RadOperandSpec oDs41EmbedPlanes[] = {
    { "tokens", RAD_OPD_IN, 0 },     { "codes", RAD_OPD_WEIGHT, 0 },
    { "scale", RAD_OPD_WEIGHT, 0 },  { "table", RAD_OPD_WEIGHT, 1 },
    { "x", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41EmbedPlanes[] = {
    C_GE("M", 1), C_GE("n_embd", 1), C_GE("n_vocab", 1), C_GE("group", 1), C_IN("dtype", "bf16"),
};

/* The fused chains. n <= 16384: the norm's row passes between the steps in LDS. One row a
 * quant_roundtrip format, as there. */
static const RadParamSpec pDs41NormRopeRt[] = {
    P_INT("M"),     P_INT("n"),        P_F64("eps"),   P_INT("n_head"),  P_INT("head_dim"),
    P_INT("rot"),   P_INT("off"),      P_STR("mode"),  P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("scale"),   P_STR("dtype"), P_F64O("wadd"),   P_F64O("mult"),
};
static const RadOperandSpec oDs41NormRopeRt[] = {
    { "x", RAD_OPD_IN, 0 },       { "w", RAD_OPD_WEIGHT, 0 },    { "y", RAD_OPD_OUT, 0 },
    { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
#define C_DS41_ROPE                                                                                \
    C_GE("M", 1), C_GE("n_head", 1), C_GE("head_dim", 2), C_GE("rot", 2), C_DIV("rot", 2),         \
        C_GE("off", 0), C_GE("inverse", 0), C_LE("inverse", 1), C_IN("mode", "gptj neox"),          \
        C_GE("block", 1), C_IN("dtype", "bf16")
static const RadConstraint cDs41NormRopeRtFp8[] = {
    C_DS41_ROPE, C_GE("n", 1), C_LE("n", 16384), C_IN("format", "fp8_e4m3"),
    C_IN("scale", "ue8m0 f32"),
};
static const RadConstraint cDs41NormRopeRtFp4[] = {
    C_DS41_ROPE, C_GE("n", 1), C_LE("n", 16384), C_IN("format", "fp4_e2m1"),
    C_IN("scale", "ue8m0 e4m3"),
};
/* rmsnorm_rope_roundtrip + kv_store of y's row (lever K1 fold c): the chain's parameters, then kv_store's
 * block_size and kv_dtype; y's row is one cache entry, its K half then its V half (kv_store at n_head_kv 1,
 * head_dim n / 2: the cache [n_blocks, 1, block_size, n]). */
static const RadParamSpec pDs41NormRopeRtKv[] = {
    P_INT("M"),     P_INT("n"),        P_F64("eps"),   P_INT("n_head"),  P_INT("head_dim"),
    P_INT("rot"),   P_INT("off"),      P_STR("mode"),  P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("scale"),   P_STR("dtype"), P_INT("block_size"), P_STR("kv_dtype"),
    P_F64O("wadd"), P_F64O("mult"),
};
static const RadOperandSpec oDs41NormRopeRtKv[] = {
    { "x", RAD_OPD_IN, 0 },       { "w", RAD_OPD_WEIGHT, 0 },    { "y", RAD_OPD_OUT, 0 },
    { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "slot_mapping", RAD_OPD_IN, 0 },
    { "kv_cache", RAD_OPD_INOUT, 0 },
};
static const RadConstraint cDs41NormRopeRtKvFp8[] = {
    C_DS41_ROPE, C_GE("n", 2), C_LE("n", 16384), C_DIV("n", 2), C_IN("format", "fp8_e4m3"),
    C_IN("scale", "ue8m0 f32"), C_GE("block_size", 1), C_IN("kv_dtype", "bf16 fp8_e4m3"),
};

static const RadParamSpec pDs41RopeRt[] = {
    P_INT("M"),   P_INT("n_head"), P_INT("head_dim"), P_INT("rot"),    P_INT("off"),
    P_STR("mode"), P_INT("inverse"), P_INT("block"),  P_STR("format"), P_STR("scale"),
    P_STR("dtype"),
};
static const RadOperandSpec oDs41RopeRt[] = {
    { "x", RAD_OPD_INOUT, 0 }, { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
static const RadConstraint cDs41RopeRtFp8[] = {
    C_DS41_ROPE, C_IN("format", "fp8_e4m3"), C_IN("scale", "ue8m0 f32"),
};
static const RadConstraint cDs41RopeRtFp4[] = {
    C_DS41_ROPE, C_IN("format", "fp4_e2m1"), C_IN("scale", "ue8m0 e4m3"),
};
#undef C_DS41_ROPE

static const RadParamSpec pDs41CollapseNorm[] = { P_INT("M"),     P_INT("n"),     P_INT("hc"),
                                                  P_F64("eps"),   P_STR("dtype"), P_F64O("wadd"),
                                                  P_F64O("mult") };
static const RadOperandSpec oDs41CollapseNorm[] = {
    { "h", RAD_OPD_IN, 0 },     { "pre", RAD_OPD_IN, 0 }, { "x", RAD_OPD_OUT, 0 },
    { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41CollapseNorm[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};
static const RadOperandSpec oDs41PostCollapseNorm[] = {
    { "x", RAD_OPD_IN, 0 },   { "h", RAD_OPD_IN, 0 },  { "post", RAD_OPD_IN, 0 },
    { "comb", RAD_OPD_IN, 0 }, { "out", RAD_OPD_OUT, 0 }, { "pre", RAD_OPD_IN, 0 },
    { "xc", RAD_OPD_OUT, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
/* mhc_post_bf16's range (cMhcPlain). */
static const RadConstraint cDs41PostCollapseNorm[] = {
    C_GE("M", 1),  C_LE("M", 65535), C_GE("n", 1),          C_LE("n", 16384),
    C_GE("hc", 2), C_LE("hc", 8),    C_IN("dtype", "bf16"),
};
/* lever M1 (ds41-int9b): mhc_post_collapse_rmsnorm, then mhc_mix over its out -- the first's parameters and operands, then
 * mhc_mix's iterations / hc_eps and fn, scale, base and pre (`eps` is both ops': DeepSeek-V4.1 gives the norm and the mix
 * rms_norm_eps); post and comb are read as the post's gates and written as the mix's. */
static const RadParamSpec pDs41PcnMix[] = { P_INT("M"),     P_INT("n"),          P_INT("hc"),
                                            P_INT("iterations"), P_F64("eps"),   P_F64("hc_eps"),
                                            P_STR("dtype"), P_F64O("wadd"),      P_F64O("mult") };
static const RadOperandSpec oDs41PcnMix[] = {
    { "x", RAD_OPD_IN, 0 },       { "h", RAD_OPD_IN, 0 },       { "post", RAD_OPD_INOUT, 0 },
    { "comb", RAD_OPD_INOUT, 0 }, { "out", RAD_OPD_OUT, 0 },    { "pre", RAD_OPD_IN, 0 },
    { "xc", RAD_OPD_OUT, 0 },     { "w", RAD_OPD_WEIGHT, 0 },   { "y", RAD_OPD_OUT, 0 },
    { "fn", RAD_OPD_WEIGHT, 0 },  { "scale", RAD_OPD_WEIGHT, 0 }, { "base", RAD_OPD_WEIGHT, 0 },
    { "pre_o", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41PcnMix[] = {
    C_GE("M", 1),          C_LE("M", 65535),        C_GE("n", 1024), C_LE("n", 16384), C_DIV("n", 1024),
    C_GE("hc", 4),         C_LE("hc", 4),           C_GE("iterations", 1), C_LE("iterations", 256),
    C_IN("dtype", "bf16"),
};

#elif defined(R4D_DS41_SCHEMAS)

    SCHEMA("bi_rowhash", pBiRowHash, oBiRowHash,
           "h[m] = a 64-bit fingerprint of row m's bytes (two i32 a row): debug only, lever BI's tracer."),
    SCHEMA("token_map", pDs41TokenMap, oDs41TokenMap,
           "mapped[m] = table[ids[m]], -1 for an id outside [0, n_vocab): ids through an id table."),
    SCHEMA("stream_gate_add", pDs41StreamGate, oDs41StreamGate,
           "Per row and stream s: dot = sum h w k * rsqrt(mean h^2 + eps) rsqrt(mean k^2 + eps) / sqrt(n), "
           "gate = sigmoid(copysign(sqrt(max(|dot|, gate_eps)), dot)), out = h + gate v; f32, rounded "
           "once. v is one [n] row a token shared by the hc streams; w is f32 [hc*n]."),
    SCHEMA("ds41_rope_rows", pDs41RopeRows, oDs41RopeRows,
           "rope from a rope_table plane over columns [off, off+rot) of each head_dim-wide head of x, in "
           "place: gptj pairs adjacent columns, neox i with i+rot/2; (a,b)(c+is) in f32, rounded once; "
           "inverse 1 conjugates the rotation. A negative or out-of-table position leaves its row."),
    SCHEMA("quant_roundtrip", pDs41QuantRoundtrip, oDs41QuantRoundtrip,
           "Block-scaled quantise then dequantise in place: per `block` values of a row, amax, a scale "
           "(fp8_e4m3: amax floored at 1e-4, ue8m0 = power of two at or above amax*f32(1/448), f32 = "
           "amax*f32(1/448); fp4_e2m1: ue8m0 from amax floored at 6*2^-126 times f32(1/6), e4m3 = "
           "fp8(max(amax, 6*2^-9)/6)), codes of clamp(x/scale) (fp8 RNE saturating, e2m1 ties to the "
           "even code), times scale in f32, rounded once to dtype."),
    SCHEMA("mhc_mix", pDs41MhcMix, oDs41MhcMix,
           "mhc_pre without the collapse: the same RMS-scaled projection, gates and Sinkhorn mixing, with "
           "the pre gates written to `pre` (f32 [M, hc]) instead of collapsing h. For models that collapse "
           "a sublayer's input with gates an earlier sublayer computed (DeepSeek-V4.1); mhc_collapse "
           "applies them."),
    SCHEMA("mhc_collapse", pDs41MhcCollapse, oDs41MhcCollapse,
           "x[i] = sum over hc streams of pre[stream] * h[stream, i], accumulated in f32 and rounded once "
           "to dtype. pre is f32 [M, hc]; with every pre 1/hc it is mhc_contract."),
    SCHEMA("softmax_pool", pDs41SoftmaxPool, oDs41SoftmaxPool,
           "Channel-wise softmax pooling of `ratio` rows: stage [M*ratio, 2n] f32 holds each entry's "
           "values then scores; out[w,c] = sum_i softmax_i(score[i,c]) value[i,c] in f32, rounded once. "
           "An entry whose optional page is negative is not written."),
    SCHEMA("entry_store", pDs41EntryStore, oDs41EntryStore,
           "x[w] (n values) into the whole of block page[w] of a paged cache [blocks, heads, block_size, "
           "slot], heads*block_size*slot == n, rounded once to the cache dtype; a negative page writes "
           "nothing. Declared right after its group, its block_size sets the group's. format mxfp4 (n a "
           "multiple of 32): each block of 32 as quant_roundtrip(fp4_e2m1, ue8m0) scales it, packed -- n/2 "
           "bytes of e2m1 codes (value 2i low nibble of byte i), n/32 scale bytes, zeros to 16 -- into "
           "heads*block_size*slot bf16 slots of that many bytes; exact for values that rule produced. format "
           "nvfp4 (n a multiple of 32): blocks of 16 with an e4m3 scale (n/2 code bytes, n/16 scale bytes, "
           "zeros to 16), the scale searched among amax/c (c = 6, 4, 3, 2, 1.5, 1, 0.5) for one that gives "
           "every value back exactly, else quant_roundtrip(fp4_e2m1, e4m3)'s."),
    SCHEMA("window_store", pDs41WindowStore, oDs41WindowStore,
           "x[m] (n values, n a multiple of 32) packed into slot slot_mapping[m] of a paged cache [n_blocks, 1, "
           "block_size, slot] (kv_store's slot rule: a negative slot or one past the cache stores nothing). format "
           "fp8_e4m3 (the fp8 window slot, heads*slot*2 = (n + n/32) bytes rounded up to 16): n e4m3 codes (value d "
           "at byte d), then a scale byte e a block of 32 (scale 2^(e-127)), then zeros. A block's k: 120 if its amax "
           "(NaN as 0) is inf, 0 if it is 0, else the least k with amax <= 448 2^k, at least -126; e = k + 127; code "
           "= fp8(v 2^-k) (RNE, saturating). A value reads back as e4m3(code) x 2^(e-127) in f32: exactly the stored "
           "bf16 where the row is quant_roundtrip(fp8_e4m3, ue8m0, 32) output (DeepSeek-V4.1's window key)."),
    SCHEMA("router_scored_topk", pDs41RouterScored, oDs41RouterScored,
           "router_sigmoid_topk with the score function a required name: sigmoid, sqrtsoftplus "
           "(sqrt(softplus(logit)), softplus linear above 20) or softmax over all n_expert logits."),
    SCHEMA("router_scored_topk_scatter", pDs41RouterScoredScatter, oDs41RouterScoredScatter,
           "`router_scored_topk` then `moe_scatter` over its ids, in one launch, and with `gate` present "
           "the ungated shared arm's `fill` of its gate with 1; with `inverse` the sort is "
           "`moe_scatter_inv` (router_sigmoid_topk_scatter with the score named). Every output is the "
           "bytes the separate launches write, including a row with fewer finite ranks than top_k. "
           "Bounded by the sort's single-workgroup form: M * top_k <= 1024, refused above."),
    SCHEMA("router_gemm_f32_scored_topk_scatter", pDs41RouterGemmScored, oDs41RouterGemmScored,
           "`gemm_nt_f32` (a [M, K] bf16 x b [n_expert, K] f32 -> logits f32, the router's projection), then "
           "`router_scored_topk_scatter` over its logits (no gate), in one launch: the GEMM's last workgroup to "
           "finish ranks and sorts. Every output is the bytes the separate launches write. Device-only; "
           "M <= 48, n_expert <= 384, score sqrtsoftplus."),
    SCHEMA("embed_lookup_planes", pDs41EmbedPlanes, oDs41EmbedPlanes,
           "A gather from a table stored as single-level affine planes, read in place: x[m, i] = v * "
           "scale[id, i / group] * mult, v = table[code] when a table is given, else the code. Codes are i8, "
           "u8, or 4-bit elements of a bit stream along the row, lowest bits first; a table is indexed by an "
           "unsigned code and holds every value. Padding, bounds and vocab_offset as embed_lookup."),
    SCHEMA("rmsnorm_rope_roundtrip", pDs41NormRopeRt, oDs41NormRopeRt,
           "rmsnorm (x, w -> y), then rope_rows over y, then quant_roundtrip over y: y as the three ops "
           "leave it, each step rounded to dtype where that op stores. n = n_head * head_dim. A key "
           "normalised, rotated and quantise-dequantised before its cache (DeepSeek-V4.1)."),
    SCHEMA("rmsnorm_rope_roundtrip_kv_store", pDs41NormRopeRtKv, oDs41NormRopeRtKv,
           "rmsnorm_rope_roundtrip (x, w -> y, cos_sin, positions), then kv_store of y's row into the paged "
           "cache at its slot (slot_mapping; negative or past the cache: no store), the row one entry of a "
           "[n_blocks, 1, block_size, n] cache (kv_store at n_head_kv 1, head_dim n / 2: K = y[0, n/2), "
           "V = y[n/2, n)): y and the cache as the two ops leave them. A key normalised, rotated, "
           "quantise-dequantised and stored in one pass (DeepSeek-V4.1's window key). kv_dtype fp8_e4m3: the "
           "cache is window_store's fp8 window slot ([n_blocks, 1, block_size, (n + n/32) rounded up to 16 bytes "
           "/ 2]) and y's row is stored there as window_store packs it."),
    SCHEMA("rope_rows_roundtrip", pDs41RopeRt, oDs41RopeRt,
           "rope_rows then quant_roundtrip over the n_head * head_dim columns of x, in place: x as the "
           "two ops leave it, each step rounded to dtype where that op stores."),
    SCHEMA("mhc_collapse_rmsnorm", pDs41CollapseNorm, oDs41CollapseNorm,
           "mhc_collapse (h, pre -> x), then rmsnorm (x, w -> y): x and y as the two ops leave them, x "
           "rounded to dtype before the norm reads it."),
    SCHEMA("mhc_post_collapse_rmsnorm", pDs41CollapseNorm, oDs41PostCollapseNorm,
           "mhc_post (x, h, post, comb -> out), then mhc_collapse (out, pre -> xc), then rmsnorm (xc, "
           "w -> y): out, xc and y as the three ops leave them, each rounded to dtype before the next "
           "step reads it. out must not alias h."),
    SCHEMA("mhc_post_collapse_rmsnorm_mix", pDs41PcnMix, oDs41PcnMix,
           "mhc_post_collapse_rmsnorm (x, h, post, comb -> out; out, pre -> xc; xc, w -> y), then mhc_mix over out "
           "(fn, scale, base -> pre_o, post, comb): every output as the two ops leave them. post and comb are read "
           "as the post's gates and then written as the mix's. A sublayer's output into the next connection's "
           "collapse, norm and gates in one op (DeepSeek-V4.1's attention connection, lever M1)."),

#elif defined(R4D_DS41_ROWS)

/* ---------------------------------- DeepSeek-V4.1's elem ops (r4d_ds41_elem.hip) -------------
 * R4D_PRIO_ONLY throughout: nothing else in this plugin implements them. */
{ "bi_rowhash", "bi_rowhash", "debug",
  "a 64-bit fingerprint of each row's bytes (XOR of mixed 16-bit units; order-free)", "any M; a workgroup a row",
  "any x dtype (2-byte aligned rows), i32 pairs h", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cBiRowHash), NOTUNE,
  nullptr, nullptr, r4d_rad_bi_rowhash, nullptr, nullptr, nullptr, nullptr },
{ "token_map_i32", "token_map", "embed",
  "ids through an id table, -1 outside [0, n_vocab)", "any M; one thread an id",
  "i32 ids, table and mapped", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41TokenMap), NOTUNE,
  nullptr, nullptr, r4d_rad_token_map, nullptr, nullptr, nullptr, r4d_shape_token_map },
{ "stream_gate_add_bf16", "stream_gate_add", "norm",
  "Engram's normalised-dot sigmoid gate, out = h + gate v per stream",
  "any M and n, hc 1..16; one workgroup a (row, stream)",
  "bf16 h, k, v and out (rows may be pitched column slices), f32 w and arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41StreamGate), NOTUNE,
  nullptr, nullptr, r4d_rad_stream_gate_add, nullptr, nullptr, nullptr,
  r4d_shape_stream_gate_add },
{ "rope_rows_bf16", "ds41_rope_rows", "attn",
  "rotary from a rope_table plane over columns [off, off+rot) of each head, in place",
  "any M, heads and head_dim, gptj or neox, inverse 0/1; one thread a pair",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRows),
  NOTUNE, nullptr, nullptr, r4d_rad_ds41_rope_rows, nullptr, nullptr, nullptr, r4d_shape_rope_rows },
{ "quant_roundtrip_fp8_bf16", "quant_roundtrip", "quant",
  "block-scaled E4M3 quantise-dequantise in place, ue8m0 or f32 scale",
  "any M, n a multiple of block; a lane group a block up to 32 (powers of two), else a thread",
  "bf16 x, f32 arithmetic", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41QuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_quant_roundtrip },
{ "quant_roundtrip_fp4_bf16", "quant_roundtrip", "quant",
  "block-scaled e2m1 quantise-dequantise in place, ue8m0 or e4m3 scale",
  "any M, n a multiple of block; a lane group a block up to 32 (powers of two), else a thread",
  "bf16 x, f32 arithmetic", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41QuantFp4), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_quant_roundtrip },
{ "mhc_mix_bf16", "mhc_mix", "norm",
  "mHC RMS-scaled projection, gates and Sinkhorn mixing, pre gates out",
  "any M and n, hc 2..8, iterations 1..256; 1024-column chunk partials in scratch, then a wave a "
  "row (one wave a row walking the chunks without scratch, the same bits)",
  "bf16 h, f32 fn, scale, base, pre, post and comb", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41MhcMix), NOTUNE, nullptr, nullptr, r4d_rad_mhc_mix, r4d_rad_mhc_mix_scratch, nullptr,
  nullptr, r4d_shape_mhc_mix },
{ "mhc_collapse_bf16", "mhc_collapse", "norm", "x = sum over streams of pre[j] h[j]",
  "any M and n, hc 2..8; eight columns a thread on 16-byte rows", "bf16 h and x, f32 pre",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41MhcCollapse), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_collapse, nullptr, nullptr, nullptr, r4d_shape_mhc_collapse },
{ "softmax_pool_bf16", "softmax_pool", "attn",
  "channel-wise softmax pooling of ratio rows into one entry",
  "any M and n, ratio 1..64; one thread a (entry, channel)", "f32 stage, bf16 out, i32 page",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41SoftmaxPool), NOTUNE,
  nullptr, nullptr, r4d_rad_softmax_pool, nullptr, nullptr, nullptr, r4d_shape_softmax_pool },
{ "entry_store_bf16", "entry_store", "attn", "one entry a block into a paged cache",
  "any M, n and block_size; one thread an element", "bf16 x and cache, i32 page",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41EntryStore), NOTUNE,
  nullptr, nullptr, r4d_rad_entry_store, nullptr, nullptr, nullptr, r4d_shape_entry_store },
{ "window_store_fp8_bf16", "window_store", "attn",
  "a key row packed into the fp8 window slot (e4m3 codes, a ue8m0-style scale byte a block of 32) at its slot",
  "any M, n a multiple of 32 up to 16384, any block_size; 256 threads a row, a wave a block, a byte a thread",
  "bf16 x, i32 slot mapping, the cache's bytes in bf16 slots", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41WindowStore), NOTUNE, nullptr, nullptr, r4d_rad_window_store, nullptr, nullptr, nullptr,
  r4d_shape_window_store },
{ "router_scored_topk_f32", "router_scored_topk", "moe",
  "correction-bias routing over sigmoid, sqrt(softplus) or softmax scores",
  "single group, experts <= 512, top_k <= 32; one wave a row",
  "f32 logits, bias and weights; i32 ids", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41RouterScored), NOTUNE, nullptr, nullptr, r4d_rad_router_scored_topk, nullptr, nullptr,
  nullptr, r4d_shape_router_scored_topk },
{ "router_scored_topk_scatter_f32", "router_scored_topk_scatter", "moe",
  "router_scored_topk, the stable counting sort and the ungated shared gate in one launch",
  "single group, experts <= 512, top_k <= 32, M*top_k <= 1024; one workgroup, a wave a row",
  "f32 logits, bias and weights; i32 ids, sorted slots, offsets and counts; bf16 gate",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RouterScoredScatter), NOTUNE, nullptr, nullptr,
  r4d_rad_router_scored_topk_scatter, nullptr, nullptr, nullptr,
  r4d_shape_router_scored_topk_scatter, nullptr, r4d_fuse_router_scored_topk_scatter },
{ "router_gemm_f32_scored_topk_scatter_bf16", "router_gemm_f32_scored_topk_scatter", "moe",
  "the router's f32 projection, router_scored_topk and the stable counting sort in one launch -- the GEMM's last "
  "workgroup ranks and sorts (lever FUSE-R)",
  "M <= 48, M*top_k <= 1024, n_expert <= 384, top_k <= 32, K <= 16384; the GEMM a wave a tile (8 / 16 waves a "
  "workgroup), the selection a wave a row",
  "bf16 activation, f32 or bf16 projection, f32 bias; f32 logits and weights, i32 ids, sorted slots, offsets and "
  "counts",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RouterGemmScored), NOTUNE,
  r4d_rad_router_gemm_scored_scatter_init, r4d_rad_router_gemm_scored_scatter_fini,
  r4d_rad_router_gemm_scored_scatter, nullptr, nullptr, nullptr,
  r4d_shape_router_gemm_f32_scored_topk_scatter, nullptr, r4d_fuse_router_gemm_f32_scored_topk_scatter },
{ "embed_lookup_planes_bf16", "embed_lookup_planes", "vocab",
  "embedding gather from a table's affine planes, read in place (no layout hook)",
  "any M, n_embd a multiple of group; one thread an element of a gathered row",
  "i8 u8 i4 u4 codes; f32 f16 bf16 fp8_e4m3 e8m0 scale; optional i8 u8 f32 f16 bf16 fp8_e4m3 "
  "table; i32 tokens; bf16 x",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41EmbedPlanes), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_planes, nullptr, nullptr, nullptr,
  r4d_shape_embed_lookup_planes },
/* The fused chains: one launch for two or three of the rows above (and rmsnorm_bf16), the same
 * bytes (`replaces`, r4d_fuse.cpp). */
{ "rmsnorm_rope_roundtrip_fp8_bf16", "rmsnorm_rope_roundtrip", "norm",
  "rmsnorm, rope_rows and an E4M3 quant_roundtrip of a row in one pass",
  "any M, n <= 16384; rmsnorm_bf16's workgroup a row (r4d_act_threads), the row in LDS between steps",
  "bf16 x and y, f32 or bf16 w, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41NormRopeRtFp8), NOTUNE, nullptr, nullptr, r4d_rad_rmsnorm_rope_roundtrip, nullptr,
  nullptr, nullptr, r4d_shape_rmsnorm_rope_roundtrip, nullptr, r4d_fuse_rmsnorm_rope_roundtrip },
{ "rmsnorm_rope_roundtrip_fp4_bf16", "rmsnorm_rope_roundtrip", "norm",
  "rmsnorm, rope_rows and an e2m1 quant_roundtrip of a row in one pass",
  "any M, n <= 16384; rmsnorm_bf16's workgroup a row (r4d_act_threads), the row in LDS between steps",
  "bf16 x and y, f32 or bf16 w, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41NormRopeRtFp4), NOTUNE, nullptr, nullptr, r4d_rad_rmsnorm_rope_roundtrip, nullptr,
  nullptr, nullptr, r4d_shape_rmsnorm_rope_roundtrip, nullptr, r4d_fuse_rmsnorm_rope_roundtrip },
/* No `replaces`: kv_store reads y's two halves as its k and v, column slices of one operand that a chain's
 * wiring cannot name. The selftest (ds41 fold c case) and tests/r4d_emu ds41fold hold it to the two rows. */
{ "rmsnorm_rope_roundtrip_kv_store_fp8_bf16", "rmsnorm_rope_roundtrip_kv_store", "norm",
  "rmsnorm, rope_rows and an E4M3 quant_roundtrip of a row, and kv_store of the row, in one pass",
  "any M, n <= 16384 even; rmsnorm_bf16's workgroup a row (r4d_act_threads), the row in LDS between steps, "
  "each stored bf16 also written to the row's cache slot",
  "bf16 x, y and cache, f32 or bf16 w, f32 cos_sin, i32 positions and slot mapping", RAD_DOMAIN_DEVICE,
  R4D_PRIO_ONLY, ROW(cDs41NormRopeRtKvFp8), NOTUNE, nullptr, nullptr,
  r4d_rad_rmsnorm_rope_roundtrip_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_rmsnorm_rope_roundtrip_kv_store },
{ "rope_rows_roundtrip_fp8_bf16", "rope_rows_roundtrip", "attn",
  "rope_rows then an E4M3 quant_roundtrip, in place, in one pass",
  "any M; a thread a column, the gptj partner by a lane shuffle (head_dim and off even, block a "
  "power of two <= 32); else rope_rows and quant_roundtrip as two launches",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRtFp8),
  NOTUNE, nullptr, nullptr, r4d_rad_rope_rows_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_rope_rows_roundtrip, nullptr, r4d_fuse_rope_rows_roundtrip },
{ "rope_rows_roundtrip_fp4_bf16", "rope_rows_roundtrip", "attn",
  "rope_rows then an e2m1 quant_roundtrip, in place, in one pass",
  "any M; a thread a column, the gptj partner by a lane shuffle (head_dim and off even, block a "
  "power of two <= 32); else rope_rows and quant_roundtrip as two launches",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRtFp4),
  NOTUNE, nullptr, nullptr, r4d_rad_rope_rows_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_rope_rows_roundtrip, nullptr, r4d_fuse_rope_rows_roundtrip },
{ "mhc_collapse_rmsnorm_bf16", "mhc_collapse_rmsnorm", "norm",
  "x = sum over streams of pre[j] h[j], then y = rmsnorm(x), in one pass",
  "any M and n, hc 2..8; rmsnorm_bf16's workgroup a row (r4d_act_threads)",
  "bf16 h, x and y, f32 pre, f32 or bf16 w", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41CollapseNorm), NOTUNE, nullptr, nullptr, r4d_rad_mhc_collapse_rmsnorm, nullptr,
  nullptr, nullptr, r4d_shape_mhc_collapse_rmsnorm, nullptr, r4d_fuse_mhc_collapse_rmsnorm },
{ "mhc_post_collapse_rmsnorm_bf16", "mhc_post_collapse_rmsnorm", "norm",
  "out = post[k] x + sum_j comb[j,k] h[j], xc = sum_k pre[k] out[k], y = rmsnorm(xc), in one pass",
  "M 1..65535, n 1..16384, hc 2..8; rmsnorm_bf16's workgroup a row (r4d_act_threads)",
  "bf16 x, h, out, xc and y, f32 post, comb and pre, f32 or bf16 w", RAD_DOMAIN_DEVICE,
  R4D_PRIO_ONLY, ROW(cDs41PostCollapseNorm), NOTUNE, nullptr, nullptr,
  r4d_rad_mhc_post_collapse_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_mhc_post_collapse_rmsnorm, nullptr, r4d_fuse_mhc_post_collapse_rmsnorm },
/* lever M1 (ds41-int9b). No `replaces`: the arch issues it in place of the pair (deepseek41, RADIANCE_DS41_MHC_FUSE /
 * fn4 mhc_fuse). r4d_selftest --case mhc_pcn_mix and tests/r4d_emu ds41i9b hold it to the two rows byte for byte. */
{ "mhc_post_collapse_rmsnorm_mix_bf16", "mhc_post_collapse_rmsnorm_mix", "norm",
  "mhc_post_collapse_rmsnorm, then mhc_mix over its out: three launches, the mix's projection formed from the post's "
  "output in LDS",
  "M 1..65535, hc 4, n a multiple of 1024 up to 16384; 8 rows x 1024 columns a workgroup, mhc_mix's finish kernel, a "
  "workgroup a row for the norm (r4d_act_threads' order); mhc_mix's partials in scratch",
  "bf16 x, h, out, xc and y, f32 post, comb, pre, w, fn, scale, base and pre_o", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41PcnMix), NOTUNE, nullptr, nullptr, r4d_rad_mhc_post_collapse_rmsnorm_mix,
  r4d_rad_mhc_post_collapse_rmsnorm_mix_scratch, nullptr, nullptr, r4d_shape_mhc_post_collapse_rmsnorm_mix },

#endif
