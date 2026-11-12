/* qwen4exp_x3 -- the qwen4exp_fp8 plugin claiming the "exl3" quantisation descriptor.
 *
 * A container rad-convert writes from an EXL3 checkpoint (quantization_config.quant_method exl3: xeon's
 * qwen3.8-flash-next-exl3.rad, its routed experts EXL3 codes, everything else bf16) carries the descriptor
 * "exl3", and the selection key (spec §2.4) is (architecture, descriptor): qwen4exp_fp8 answers "" only, so
 * the engine refused the container ("no architecture plugin claims 'qwen4exp' at quantisation 'exl3'"). The
 * plugin itself already serves that container's formats -- the routed experts through rad_block_moe_fp8.h's
 * EXL3 path (MoeFP8::probe, MoE format 12), bf16 linears through LinearFP8's plain-bf16 form -- so this
 * unit is qwen4exp_fp8.cpp, included whole with its exports off, under a second key. Nothing else differs:
 * declare, step, probe and the pass key are qwen4exp_fp8's. */
#define RAD_ARCH_NO_EXPORTS
#include "../qwen4exp_fp8/qwen4exp_fp8.cpp"
#undef RAD_ARCH_NO_EXPORTS

namespace qwen4exp_x3 {
using qwen4exp_fp8::declare;
using qwen4exp_fp8::probe;
using qwen4exp_fp8::step;
}  // namespace qwen4exp_x3

RAD_ARCH_PROBE(qwen4exp_x3)
RAD_ARCH_PLUGIN(qwen4exp_x3, "qwen4exp", "exl3", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next) from an EXL3 checkpoint: qwen4exp_fp8 under the exl3 "
                "descriptor (EXL3 routed experts, bf16 elsewhere)")
/* fn4 runtime switches (the core's runtime/fn4_switch.h, between steps, every rank parked): "exl3_dual" 0 / 1 -- fn4
 * BI-dual (the EXL3 downs take the step's row bands; RADIANCE_FN4_X3DUAL=1 declares); "moe_pair" 0 / 1 / 2 --
 * fn4 P2x (qwen4exp_fp8's step_pipelined); "exl3_mapshare" 0 / 1 --
 * MoeFP8's EXL3 ops with or without the pass's shared routing map (fn4_exl3_map_flag); "exl3_gd" 0 / 1 -- the decode
 * band's fused moe_exl3_gate_up_down in place of the split pair (fn4_exl3_gd_flag; rad_block_moe_fp8.h). Every other
 * name is not this plugin's (0). */
extern "C" int rad_fn4_switch(const char* name, int value) {
    if (!name) return 0;
    if (std::strcmp(name, "moe_pair") == 0) {   /* fn4 P2x (qwen4exp_fp8's step_pipelined; RADIANCE_FN4_P2X=1 declares) */
        if (value < 0 || value > 2) return -1;
        if (value && !qwen4exp_fp8::fn4_env_on("RADIANCE_FN4_P2X")) return -1;
        qwen4exp_fp8::fn4_pair_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (std::strcmp(name, "bi_seg") == 0) {   /* fn4 BI: the trunk's linears cut at the step's segments */
        if (value < 0 || value > 1) return -1;
        qwen4exp_fp8::fn4_biseg_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (std::strcmp(name, "hc_bi") == 0) {   /* fn4 BI: the reads cut at the step's segments (libr4d takes it too) */
        if (value < 0 || value > 2) return -1;
        qwen4exp_fp8::fn4_hcbi_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    if (std::strcmp(name, "exl3_dual") == 0) {   /* fn4 BI-dual (qwen4exp_fp8's step; RADIANCE_FN4_X3DUAL=1 declares) */
        if (value < 0 || value > 1) return -1;
        if (value && !qwen4exp_fp8::g_dual_declared.load(std::memory_order_relaxed)) return -1;
        qwen4exp_fp8::fn4_dual_flag().store(value, std::memory_order_relaxed);
        return 1;
    }
    const bool map = std::strcmp(name, "exl3_mapshare") == 0, gd = std::strcmp(name, "exl3_gd") == 0;
    if (!map && !gd) return 0;
    if (value < 0 || value > 1) return -1;
    (map ? rad::arch::fn4_exl3_map_flag() : rad::arch::fn4_exl3_gd_flag()).store(value, std::memory_order_relaxed);
    return 1;
}
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    return qwen4exp_fp8::pass_key(c, batch, used_max_h);
}
