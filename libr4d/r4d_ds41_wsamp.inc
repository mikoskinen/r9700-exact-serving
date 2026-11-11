/* r4d_ds41_wsamp.inc -- lever SD's op, dspark_walk_sample_step, in r4d_rows.cpp, included three times: R4D_DS41_SPECS
 * (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (its kSchemas entry) and R4D_DS41_ROWS (its
 * kKernels entry). No include guard, on purpose.
 *
 * One step of DeepSeek-V4.1's DSpark Markov walk over the whole vocabulary on one rank's vocabulary shard, SAMPLED
 * (r4d_ds41_wsamp.hip says the steps, r4d_ds41_wsamp_math.h the law and the state). Declared by deepseek41 only under
 * --draft-coupling on; libr4d is its only implementation (tests/r4d_emu/test_ds41wsamp holds the kernel to a host
 * reference walk bit for bit). */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41WSamp[] = {
    P_INT("M"), P_INT("step"), P_INT("steps"), P_INT("rank"), P_INT("n_vocab"), P_INT("N"),
    P_INT("vocab_off"), P_INT("world"), P_INTO("anchor_stride"), P_STR("dtype"),
};
static const RadOperandSpec oDs41WSamp[] = {
    { "logits", RAD_OPD_IN, 0 },     { "anchor", RAD_OPD_IN, 0 },    { "prev", RAD_OPD_IN, 1 },
    { "pred", RAD_OPD_WEIGHT, 0 },   { "succ", RAD_OPD_WEIGHT, 0 },  { "keys", RAD_OPD_INOUT, 1 },
    { "tokens", RAD_OPD_OUT, 0 },    { "scratch", RAD_OPD_INOUT, 1 }, { "count", RAD_OPD_INOUT, 1 },
    { "seq_ids", RAD_OPD_IN, 0 },    { "draw", RAD_OPD_IN, 0 },      { "q_temp", RAD_OPD_IN, 1 },
    { "state", RAD_OPD_INOUT, 0 },
};
static const RadConstraint cDs41WSamp[] = {
    C_GE("M", 1), C_GE("steps", 1), C_LE("steps", 8), C_GE("step", 0), C_GE("rank", 32), C_LE("rank", 512),
    C_DIV("rank", 32), C_GE("n_vocab", 1), C_GE("N", 0), C_GE("vocab_off", 0), C_GE("world", 1), C_LE("world", 4),
    C_IN("dtype", "f32"),
};

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("dspark_walk_sample_step", pDs41WSamp, oDs41WSamp,
       "One step of DSpark's Markov walk over the whole vocabulary, SAMPLED (lever SD), on one rank's vocabulary shard "
       "[vocab_off, vocab_off + N): step l merges step l-1's gathered keys `prev` (eight u64 keys a rank, an f32 "
       "[M, 16 * world] plane) into the eight best of the vocabulary and turns position l-1 into a token: for a draw of "
       "mode 1 on its own card x ~ Q, Q the FP64 softmax of score / q_temp over those eight, drawn at (draw.seed, "
       "draw.base, KEY_DRAFT + l); for mode 0 the best key's id (dspark_walk_step's token); tokens[s][l-1] and the "
       "round's coupled state state[l-1][s] (abi/rad_spechub.h) written. Every column's score is dspark_walk_step's "
       "(logits + the fma chain over pred[x] . succ[v], r ascending) keyed as it keys; each block's eight best a "
       "sequence go to `scratch` and the last block to arrive (`count`, one u32, zero at entry and left zero) writes "
       "the shard's eight to keys[s] (an f32 [M, 16] plane). step = steps only merges the last position."),

#elif defined(R4D_DS41_ROWS)

{ "dspark_walk_sample_step_f32", "dspark_walk_sample_step", "dflash",
  "one step of the full-vocabulary DSpark Markov walk, sampled (lever SD): a thread a column, 128 a block, every "
  "sequence in passes of eight (their predecessor rows in LDS, dspark_walk_step's form 0 chain); a block's eight best "
  "keys a sequence by repeated wave maxima, the shard's eight in the last block to arrive; the merge and the draw "
  "x ~ Q a wave a sequence (lane 0 runs r4d_wsamp_step)",
  "rank a multiple of 32 up to 512, vocab_off + N <= n_vocab, world <= 4, steps <= 8, succ 16-byte aligned, keys "
  "[M, 16] / prev [M, 16 * world] f32 planes of eight u64 keys, scratch [M * ceil(N / 128), 16] f32, count one u32 "
  "zeroed before the first step, draw [M, 8] u32, state [steps, rows >= M, 38] u32",
  "f32 logits, bf16 pred / succ (widened exactly), f32 fma in rank order, u64 keys, FP64 Q (rad_mtp_couple_build_q), "
  "53-bit uniforms (rad_spechub_uniform), i32 tokens",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41WSamp), NOTUNE,
  nullptr, nullptr, r4d_rad_dspark_walk_sample_step, nullptr, nullptr, nullptr,
  r4d_shape_dspark_walk_sample_step },

#endif
