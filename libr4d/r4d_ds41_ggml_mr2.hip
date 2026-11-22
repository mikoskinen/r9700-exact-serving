/* r4d_ds41_ggml_mr2.hip -- lever A4 (ds41-a4): lever D2's multi-row decode-band kernels (r4d_ds41_ggml_mr.hip) in their
 * XP forms (gg_mr_body XP = true: each slab's activation loads issued one slab ahead into registers, mr_fetch / mr_put),
 * the plain kernel, lever 1U-GQ's pair kernel and lever K1's folds a / e, at MR2_FORMS' (H, R, MT). A unit of its own so
 * D2's unit compiles as before (its kernels are XP = false, the template default) and the two compile in parallel.
 * Reached through r4d_gg_mr_launch_xp from D2's unit only: the gg_gemv_m16 switch's table or the bench's XP pin. */
#define R4D_GG_MR_XP 1
#include "r4d_ds41_ggml_mr.hip"
