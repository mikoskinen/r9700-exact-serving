// exl3r4d_ika.hip -- lever IKA's list for this plugin (radiance core/device/aql.cpp ika_kernel reads it by dlsym on exl3r4d.so; r4d_ika.h):
// the kernels holding R4D_IKA_ENTRY() (an agent acquire fence at entry, an agent release at every exit), so the engine may play
// their recorded packets without the packet processor's release, and without its acquire where Ctx::haz_mark allows. Empty unless
// built with -DR4D_IKA=1 in the HIP flags (DRAFTER track round 50; inserted by drafter/kit/ika_apply.py).
#ifndef R4D_IKA
#define R4D_IKA 0
#endif
namespace {
const char* const kExl3Ika[] = {
#if R4D_IKA
    "exl3r4d_epi_kernel",
    "exl3r4d_fused_kernel",
    "exl3r4d_gemm_dense_kernel",
    "exl3r4d_gemm_fused_kernel",
    "exl3r4d_gemm_k25_kernel",
    "exl3r4d_gemm_kernel",
    "exl3r4d_gemm_pdn_kernel",
    "exl3r4d_gemm_pdx_kernel",
    "exl3r4d_gemm_pu_kernel",
    "exl3r4d_gemm_wq_kernel",
    "exl3r4d_gemm_x3d_kernel",
    "exl3r4d_gemm_xs_kernel",
    "exl3r4d_opstamp_kernel",
    "exl3r4d_pf_fused_kernel",
    "exl3r4d_pf_gemm_kernel",
    "exl3r4d_pf_gemm_x3d_kernel",
    "exl3r4d_pf_grp_kernel",
    "exl3r4d_pf_map_kernel",
    "exl3r4d_pf_mapprep_kernel",
    "exl3r4d_pf_pair_kernel",
    "exl3r4d_pf_prep_kernel",
    "exl3r4d_prep_e_kernel",
    "exl3r4d_prep_grid_kernel",
    "exl3r4d_prep_id_kernel",
    "exl3r4d_prep_kernel",
    "exl3r4d_prep_row_kernel",
    "exl3r4d_routedump_kernel",
    "exl3r4d_wgp_probe_kernel",
#endif
    nullptr};
}  // namespace
extern "C" __attribute__((visibility("default"))) const char* const* r4d_ika_kernels(int* n) {
    if (n) *n = (int)(sizeof(kExl3Ika) / sizeof(kExl3Ika[0])) - 1;
    return kExl3Ika;
}
