/* r4d_ds41_walk_shapes.inc -- operand descriptions of dspark_walk_step (lever W1); included at the end of
 * r4d_shapes.cpp for its helpers (opd, opd_idx, R4D_PICK, pi).
 *
 * In schema order: logits [M * steps, N] f32, anchor (ids < n_vocab at anchor_stride), prev [M, 2 * world]
 * (the previous step's gathered keys, read as u64), pred / succ [n_vocab, rank] bf16, keys [M, 2] (this
 * step's key a sequence, read as u64 and max-ed into), tokens [M, steps] i32. Random keys are legal input:
 * a key decodes to some id, and an id outside [0, n_vocab) contributes zeros. */
extern "C" int r4d_shape_dspark_walk_step(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), steps = pi(p, n_p, "steps"), R = pi(p, n_p, "rank");
    const int64_t V = pi(p, n_p, "n_vocab"), N = pi(p, n_p, "N"), voff = pi(p, n_p, "vocab_off", 0);
    const int64_t world = pi(p, n_p, "world"), astr = pi(p, n_p, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || R <= 0 || V <= 0 || N <= 0 || voff < 0 || voff + N > V || world <= 0 || astr < 1)
        return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M * steps, N),           /* logits, this rank's shard */
        opd_idx((M - 1) * astr + 1, 0, V),              /* anchor */
        opd((uint32_t)RAD_F32, M, 2 * world),           /* prev, gathered keys */
        opd((uint32_t)RAD_BF16, V, R),                  /* pred */
        opd((uint32_t)RAD_BF16, V, R),                  /* succ */
        opd((uint32_t)RAD_F32, M, 2),                   /* keys */
        opd_idx(M, steps, V),                           /* tokens */
    };
    R4D_PICK(l);
}
