/* r4d_ds41_walk.inc -- lever W1's op, dspark_walk_step, in r4d_rows.cpp, included three times: R4D_DS41_SPECS
 * (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (its kSchemas entry) and R4D_DS41_ROWS
 * (its kKernels entry). No include guard, on purpose.
 *
 * One step of DeepSeek-V4.1's DSpark Markov walk over the whole vocabulary on one rank's vocabulary shard
 * (r4d_ds41_walk.hip says the arithmetic and the key). Declared by deepseek41 only under
 * RADIANCE_DS41_DSPARK_WALK; libr4d is its only implementation (no libref oracle: tests/r4d_emu/test_ds41walk
 * holds the kernel to a host reference walk bit for bit). */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41Walk[] = {
    P_INT("M"), P_INT("step"), P_INT("steps"), P_INT("rank"), P_INT("n_vocab"), P_INT("N"),
    P_INT("vocab_off"), P_INT("world"), P_INTO("anchor_stride"), P_STR("dtype"), P_INTO("form"),
};
static const RadOperandSpec oDs41Walk[] = {
    { "logits", RAD_OPD_IN, 0 },     { "anchor", RAD_OPD_IN, 0 }, { "prev", RAD_OPD_IN, 1 },
    { "pred", RAD_OPD_WEIGHT, 0 },   { "succ", RAD_OPD_WEIGHT, 0 }, { "keys", RAD_OPD_INOUT, 1 },
    { "tokens", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41Walk[] = {
    C_GE("M", 1), C_GE("steps", 1), C_GE("step", 0), C_GE("rank", 32), C_LE("rank", 512), C_DIV("rank", 32),
    C_GE("n_vocab", 1), C_GE("N", 0), C_GE("vocab_off", 0), C_GE("world", 1), C_LE("world", 64),
    C_IN("dtype", "f32"),
};
/* lever WSP (DRAFTER track round 84): dspark_walk_part, the WMMA form's step 0 in two parts */
static const RadParamSpec pDs41WalkPart[] = {
    P_INT("M"), P_INT("part"), P_INT("steps"), P_INT("rank"), P_INT("n_vocab"), P_INT("N"), P_INT("vocab_off"),
    P_INTO("anchor_stride"), P_STR("dtype"),
};
static const RadOperandSpec oDs41WalkPart[] = {
    { "logits", RAD_OPD_IN, 1 },     { "anchor", RAD_OPD_IN, 1 }, { "pred", RAD_OPD_WEIGHT, 0 },
    { "succ", RAD_OPD_WEIGHT, 0 },   { "acc", RAD_OPD_INOUT, 0 }, { "keys", RAD_OPD_INOUT, 1 },
};
static const RadConstraint cDs41WalkPart[] = {
    C_GE("M", 1), C_GE("part", 1), C_LE("part", 2), C_GE("steps", 1), C_GE("rank", 256), C_LE("rank", 256), C_GE("n_vocab", 1),
    C_GE("N", 1), C_GE("vocab_off", 0), C_IN("dtype", "f32"),
};

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("dspark_walk_step", pDs41Walk, oDs41Walk,
       "One step of DSpark's greedy Markov walk over the whole vocabulary, on one rank's vocabulary shard "
       "[vocab_off, vocab_off + N): step l's predecessor is the max of step l-1's gathered keys `prev` "
       "(16-byte slots, the u64 key first: an f32 [M, 4 * world] plane; the anchor at l = 0), written to tokens[s][l-1]; every "
       "column's score logits[s * steps + l][j] + sum_r pred[prev][r] * succ[vocab_off + j][r] (fma, r "
       "ascending) is keyed (ordered score bits << 32 | ~id: ties to the lower id, NaN never) and the shard's "
       "best max-ed into keys[s] (the u64 of a 16-byte slot, an f32 [M, 4] plane, zeroed before). step = steps only merges the "
       "last token. `form` (default 0) picks how a block reads succ (1: through LDS as whole lines); the bits are the "
       "same."),
SCHEMA("dspark_walk_part", pDs41WalkPart, oDs41WalkPart,
       "dspark_walk_step's step 0 in two parts (lever WSP), the WMMA form (rank 256) only: part 1 stores the products "
       "sum_r pred[anchor][r] * succ[vocab_off + j][r] of every sequence and shard column (the WMMA form's accumulator, f32) in "
       "acc [M, N] (and, keys given, zeros in all its [rows, 4] slots: the walk's fill); part 2 keys logits[s * steps][j] + acc[s][j] and max-es the shard's best into keys[s] (pre-zeroed) as step "
       "0 does. The same keys as dspark_walk_step's step 0 in the WMMA form, bit for bit."),

#elif defined(R4D_DS41_ROWS)

{ "dspark_walk_step_f32", "dspark_walk_step", "dflash",
  "one step of the full-vocabulary DSpark Markov walk on a vocabulary shard: a thread a column, 128 a block, "
  "every sequence in passes of eight (their predecessor rows in LDS), the shard's best key a sequence by a wave "
  "max and one atomicMax a block; form 0 a thread reads its succ row in 16-byte pieces, form 1 the block's rows "
  "come through LDS as whole 128-byte lines (the same fma chain a thread)",
  "rank a multiple of 32 up to 512, vocab_off + N <= n_vocab, succ 16-byte aligned, keys [M, 4] / prev [M, 4 * world] "
  "f32 planes of 16-byte key slots, keys zeroed before the step",
  "f32 logits, bf16 pred / succ (widened exactly), f32 fma in rank order, u64 keys in 16-byte slots of f32 planes, i32 tokens",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41Walk), NOTUNE,
  nullptr, nullptr, r4d_rad_dspark_walk_step, nullptr, nullptr, nullptr,
  r4d_shape_dspark_walk_step },
{ "dspark_walk_part_f32", "dspark_walk_part", "dflash",
  "lever WSP: the WMMA walk's step 0 split at its product -- part 1 (anchors' pred rows x the shard's succ rows, 16 WMMAs a "
  "16-column tile, acc stored f32) can run before the head; part 2 (a thread a column, logits + acc keyed, one atomicMax a "
  "block and sequence) after it",
  "rank 256 (the WMMA form), vocab_off + N <= n_vocab, succ / pred 16-byte aligned, acc [M, >= N] f32, keys [M, 4] f32 slots "
  "zeroed before part 2; refused unless RADIANCE_DS41_WALK_WM=1 (the steps' form)",
  "f32 logits / acc, bf16 pred / succ (WMMA bf16 x bf16 -> f32 in the WMMA form's K order), u64 keys in 16-byte slots",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41WalkPart), NOTUNE,
  nullptr, nullptr, r4d_rad_dspark_walk_part, nullptr, nullptr, nullptr,
  r4d_shape_dspark_walk_part },

#endif
