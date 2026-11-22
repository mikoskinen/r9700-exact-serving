/* r4d_ds41_wsamp_shapes.inc -- operand descriptions of dspark_walk_sample_step (lever SD); included at the end of
 * r4d_shapes.cpp for its helpers (opd, opd_idx, R4D_PICK, pi).
 *
 * In schema order: logits [M * steps, N] f32, anchor (ids < n_vocab at anchor_stride), prev [M, 16 * world] (the
 * previous step's gathered keys, eight u64 a rank), pred / succ [n_vocab, rank] bf16, keys [M, 16], tokens [M, steps]
 * i32, scratch [M * ceil(N / 128), 16] f32, count [1, 4] (one u32, zero), seq_ids [M] i32, draw [M, 8] u32, q_temp
 * [M, 1] f32 in (0, 1), state [steps, M, 38] u32. Random keys, draws and states are legal input: a key decodes to some
 * id (outside [0, n_vocab) it contributes zeros and refuses Q), a draw naming another card or of another mode keeps
 * the greedy token. */
extern "C" int r4d_shape_dspark_walk_sample_step(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), steps = pi(p, n_p, "steps"), R = pi(p, n_p, "rank");
    const int64_t V = pi(p, n_p, "n_vocab"), N = pi(p, n_p, "N"), voff = pi(p, n_p, "vocab_off", 0);
    const int64_t world = pi(p, n_p, "world"), astr = pi(p, n_p, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || R <= 0 || V <= 0 || N <= 0 || voff < 0 || voff + N > V || world <= 0 || astr < 1)
        return RAD_E_UNSUPPORTED;
    const int64_t nblk = (N + 127) / 128;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M * steps, N),           /* logits, this rank's shard */
        opd_idx((M - 1) * astr + 1, 0, V),              /* anchor */
        opd((uint32_t)RAD_F32, M, 16 * world),          /* prev, gathered keys */
        opd((uint32_t)RAD_BF16, V, R),                  /* pred */
        opd((uint32_t)RAD_BF16, V, R),                  /* succ */
        opd((uint32_t)RAD_F32, M, 16),                  /* keys, eight a sequence */
        opd_idx(M, steps, V),                           /* tokens */
        opd((uint32_t)RAD_F32, M * nblk, 16),           /* scratch */
        opd_idx(1, 4, 1, 0),                            /* count: zeros */
        opd_idx(M, 0, 64),                              /* seq_ids */
        opd((uint32_t)RAD_U32, M, 8),                   /* draw */
        in_unit(opd((uint32_t)RAD_F32, M, 1)),          /* q_temp */
        opd((uint32_t)RAD_U32, steps, M, 38),           /* state */
    };
    R4D_PICK(l);
}
