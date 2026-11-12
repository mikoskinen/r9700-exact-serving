/* r4d_ds41_gg_m16_forms.h -- lever A4 (ds41-a4): the gg_gemv_m16 switch's compiled table (r4d_ds41_ggml_mr.hip,
 * mr2_tab). Lines as RADIANCE_DS41_GG_M16_FORMS's (r4d_selftest --perfm16 writes them): type, N, K, M band, pair,
 * then the form (H, R, MT cap, WN: 0 D2's rule, -1 the grid fit, 1..8; XP 0 / 1). The file's entries come first.
 *
 * EMPTY on ds41-a4's first image: the forms are measured on the card first (the screen's P0 writes the file the
 * server reads); a measured table is compiled in once it has been read. The sentinel (H 0) never matches. */
#define R4D_GG_M16_FORMS_N 0
static const Mr2Tab kMrM16Forms[] = {
    /* type      N      K   M lo  M hi  pair   H  R  MT  WN  XP */
    { -1,        0,     0,    0,    0,    0,   0, 0,  0,  0,  0 },   /* sentinel */
};
