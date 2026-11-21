// r4d_ika_nochain.inc -- lever CIK-X (tp8 loop R49c, PREFILL round 49): the IKA families that keep IKA's plain entry in a CIK-X build
// (R4D_IKA_ENTRY_X() of r4d_ika_x.h in their definitions; CMake R4D_IKA_CHAIN_X). Chosen from the code objects (tp8/devpf/r49/cgscan.py, IKA
// df1b0322 vs CHAIN 796ed36f): more than half of a family's instantiations change beyond the entry in the CHAIN build -- >= +4 vector
// loads, >= +8 VGPRs, any VGPR spill / scratch, or >= +4 v_readfirstlane. Each name must be IKA-listed (r4d_ika_list.inc) and every
// definition of it must use R4D_IKA_ENTRY_X() (tp8/devpf/r49/cikxcheck.py checks both).
    "alt_kernel",
    "ds41_attn_dec_kernel",
    "ds41_index_score_kept_kernel",
    "ds41_index_score_kernel",
    "ds41_index_score_wm_kernel",
    "ds41_latent_gather_nv4_kernel",
    "ds41_latent_unpack_nv4_kernel",
    "mhc_post_vec_kernel",
    "mix_finish_kernel",
    "mix_fused_kernel",
    "norm_rope_rt_kernel",
    "xm_a8_kernel",
    "xm_gemv_kernel",
    "xm_seq_kernel",
    "xm_wl_kernel",
