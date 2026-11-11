/* r4d_ds41_wsamp.h -- lever SD, sampled drafting on lever W1's full-vocabulary DSpark walk (op dspark_walk_sample_step):
 * the launch and the shape hook; included by r4d_plugin.h inside its extern "C".
 *
 * Kernel in r4d_ds41_wsamp.hip, the law and the state in r4d_ds41_wsamp_math.h, specs and row in r4d_ds41_wsamp.inc,
 * operands in r4d_ds41_wsamp_shapes.inc. */
#ifndef R4D_DS41_WSAMP_H
#define R4D_DS41_WSAMP_H

/* Keys a rank (and a block) keeps a sequence a step, and their row in the f32 key planes (eight u64). */
#define R4D_DS41_WSAMP_KEYS 8
#define R4D_DS41_WSAMP_ROW 16

int r4d_rad_dspark_walk_sample_step(const RadArgs*, RadStream);
int r4d_shape_dspark_walk_sample_step(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

#endif
