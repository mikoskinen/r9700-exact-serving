/* r4d_ds41_mxfp4.inc -- gemm_nt_ggml and moe_gemm_ggml rows over GGUF MXFP4 blocks (type=mxfp4) in r4d_rows.cpp,
 * included three times: R4D_DS41_SPECS (constraints, file scope), R4D_DS41_SCHEMAS (none: the ops' schemas are
 * r4d_ds41_ggml.inc's, libref's field for field) and R4D_DS41_ROWS (entries of kKernels). No include guard, on
 * purpose.
 *
 * Kernels in r4d_ds41_mxfp4.hip. Each op has a weight-streaming `_gemv` row for the decode band (M <= 16; for the
 * grouped GEMM M counts tokens) above a `_tiled` row that serves any M, both only at type=mxfp4 (r4d_ds41_ggml's
 * rows take the other types), K whole blocks of 32. Every row relayouts a GGUF import's planes (fp4_e2m1 codes,
 * e8m0 per 32) into llama.cpp's native MXFP4 blocks with r4d_ds41_ggml_layout.cpp's hooks (libref's bytes and tag)
 * and also reads the float logical form, which is what rad-kbench's random operands pass. Built only with
 * libquant's sources (R4D_HAVE_GGML), which the layout needs. */
#if defined(R4D_DS41_SPECS)

static const RadConstraint cDs41MxGemv[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxTiled[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxMoeGemv[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41MxMoeTiled[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
/* moe_gemm_mxil (lever WIL): drafter_mx's token bound (kXmMaxTokens), whole 16-row groups */
static const RadConstraint cDs41MxIl[] = {
    C_GE("M", 1), C_LE("M", 512), C_GE("N", 16), C_DIV("N", 16), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
/* moe_gemm_mxil_glu (lever GSE, DRAFTER track round 91): moe_gemm_mxil's gate_up form (K from 2048) with swiglu_clamped in its
 * epilogue -- N is the gate_up's (gate rows then up rows), y [M x top_k, N / 2]; libr4d only. Its parameters are moe_gemm_ggml's
 * and swiglu_clamped's limit. Entry and operand description live in r4d_ds41_mxfp4.hip / r4d_ds41_mxfp4_shapes.inc, declared
 * here (not in r4d_ds41_mxfp4.h, which r4d_plugin.h carries into every unit). */
static const RadConstraint cDs41MxIlGlu[] = {
    C_GE("M", 1), C_LE("M", 512), C_GE("N", 32), C_DIV("N", 16), C_GE("K", 2048), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", "mxfp4"), C_IN("dtype", "bf16"),
};
static const RadParamSpec pDs41MxIlGlu[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"), P_INT("top_k"),
    P_STR("a_order"), P_STR("type"), P_STR("dtype"), P_F64("limit") };
extern "C" int r4d_rad_moe_mxil_glu(const RadArgs*, RadStream);
extern "C" int r4d_shape_moe_gemm_mxil_glu(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

#elif defined(R4D_DS41_SCHEMAS)

/* gemm_nt_ggml's and moe_gemm_ggml's schemas are r4d_ds41_ggml.inc's. moe_gemm_mxil (lever WIL, libr4d only) takes
 * moe_gemm_ggml's parameters and operands. */
SCHEMA("moe_gemm_mxil", pDs41MoeGemmGgml, oDs41MoeGemmGgml,
       "moe_gemm_ggml at type=mxfp4 with the weight stored tile-interleaved (libr4d's own layout, tag r4d_mxfp4_il16: "
       "16-row x 1-block tiles of 272 bytes, the native blocks' bytes moved): the same values and arithmetic as "
       "moe_gemm_ggml under drafter_mx 2 / 4 / 5 (rule B against the stock kernels), so its outputs are those forms' "
       "byte for byte. The DSpark drafter's routed experts under RADIANCE_DS41_DSPARK_WIL=1."),
SCHEMA("moe_gemm_mxil_glu", pDs41MxIlGlu, oDs41MoeGemmGgml,
       "moe_gemm_mxil's gate_up (gate rows [0, N / 2), up rows [N / 2, N), the same tile-interleaved stored form and layout "
       "hooks) with swiglu_clamped (limit) applied in the GEMM's epilogue: y [M x top_k, N / 2] holds the bytes moe_gemm_mxil "
       "followed by swiglu_clamped would write into the down projection's input (lever GSE). The DSpark drafter's routed "
       "gate_up under RADIANCE_DS41_DSPARK_WIL=1 + RADIANCE_DS41_GSE=1."),

#elif defined(R4D_DS41_ROWS)
#if defined(R4D_HAVE_GGML)

#define R4D_DS41_MX_HOOKS(shape)                                                                   \
    r4d_ds41_ggml_layout, r4d_ds41_ggml_relayout, shape, r4d_ds41_ggml_unrelayout

{ "gemm_nt_ggml_mxfp4_gemv", "gemm_nt_ggml", "gemm",
  "y = a . w^T over GGUF MXFP4 blocks, the weight streamed once: a wave per weight row (two from 3 rows on wide "
  "weights), lanes over its 17-byte blocks (four in flight, each read as the five aligned dwords covering it), "
  "the activation chunk staged in LDS as f32, a wave butterfly per sum",
  "M 1..16 (eight rows a pass), any N, K a multiple of 32", "bf16 a and y; mxfp4 blocks (or the f32 logical "
  "form), kvalues_mxfp4 x 2^(e - 128) as llama.cpp dequantises, f32 products and sum",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxGemv), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_mxfp4_gemv, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_gemm_nt_mxfp4) },
{ "gemm_nt_ggml_mxfp4_tiled", "gemm_nt_ggml", "gemm",
  "y = a . w^T over GGUF MXFP4 blocks in 128 x 128 tiles: the weight tile dequantised into LDS once a block-wide "
  "slab, the next slab's blocks prefetched, f32 8 x 8 register tiles",
  "any M, any N, K a multiple of 32", "bf16 a and y; mxfp4 blocks (or the f32 logical form), f32 products and sum",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41MxTiled), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_mxfp4_tiled, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_gemm_nt_mxfp4) },
{ "moe_gemm_ggml_mxfp4_gemv", "moe_gemm_ggml", "moe",
  "the grouped GEMM over GGUF MXFP4 experts, each live expert's weight streamed once: a block scans the offsets "
  "for the experts with rows, then a wave per weight row (two on experts of 1024 rows or more) dots it with that "
  "expert's gathered rows (eight a pass)",
  "M 1..16 tokens, any N, K a multiple of 32, up to 1024 experts", "bf16 a and y; mxfp4 experts (table or "
  "stacked, or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxMoeGemv), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_mxfp4_gemv, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_moe_gemm_mxfp4) },
{ "moe_gemm_ggml_mxfp4_tiled", "moe_gemm_ggml", "moe",
  "the grouped GEMM over GGUF MXFP4 experts in 128-row (or 32-row) x 128-column tiles of one expert's run (a "
  "block finds its expert by scanning the offsets), gemm_nt_ggml_mxfp4_tiled's slab loop",
  "any M, any N, K a multiple of 32, up to 1024 experts", "bf16 a and y; mxfp4 experts (or the f32 logical form), "
  "i32 routing",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41MxMoeTiled), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_mxfp4_tiled, nullptr, R4D_DS41_MX_HOOKS(r4d_shape_moe_gemm_mxfp4) },

{ "moe_gemm_mxil_wmma", "moe_gemm_mxil", "moe",
  "the grouped GEMM over tile-interleaved MXFP4 experts (lever WIL): a wave a 16-row group, one 64-bit load a lane a "
  "block (256 contiguous bytes a wave) and the group's 16 exponents, two v_wmma_f32_16x16x16_bf16 a block and one fma "
  "of its scale; K below 2048 the sequential-tile form (four 16-row tiles a wave, the activations staged once an item)",
  "M 1..512 tokens, N a multiple of 16, K a multiple of 32, up to 1024 experts", "bf16 a and y; mxfp4 experts "
  "stored tile-interleaved (table or stacked), i32 routing",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxIl), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_mxil, nullptr,
  r4d_ds41_mxil_layout, r4d_ds41_mxil_relayout, r4d_shape_moe_gemm_mxfp4, r4d_ds41_mxil_unrelayout },
{ "moe_gemm_mxil_glu_wmma", "moe_gemm_mxil_glu", "moe",
  "moe_gemm_mxil_wmma's gate_up kernel (lever WCN's loop) with a wave's 16 WMMA columns as 8 gate rows and their 8 up rows, "
  "each lane reading its row from the row's tile-interleaved group; the gate lane takes the up sums by DPP and stores "
  "swiglu_clamped of the two bf16-rounded values (lever GSE)",
  "M 1..512 tokens, N a multiple of 16 (the gate_up's), K from 2048, a multiple of 32, up to 1024 experts", "bf16 a and y "
  "(y [T, N / 2]); mxfp4 experts stored tile-interleaved (table or stacked), i32 routing, f64 limit > 0",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41MxIlGlu), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_mxil_glu, nullptr,
  r4d_ds41_mxil_layout, r4d_ds41_mxil_relayout, r4d_shape_moe_gemm_mxil_glu, r4d_ds41_mxil_unrelayout },

#undef R4D_DS41_MX_HOOKS
#endif  /* R4D_HAVE_GGML */
#endif
