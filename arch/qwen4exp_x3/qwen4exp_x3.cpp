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
/* fn4 runtime switch (the core's runtime/fn4_switch.h, between steps, every rank parked): "exl3_mapshare" 0 / 1 --
 * MoeFP8's EXL3 ops with or without the pass's shared routing map (rad_block_moe_fp8.h, fn4_exl3_map_flag). Every
 * other name is not this plugin's (0). */
extern "C" int rad_fn4_switch(const char* name, int value) {
    if (!name || std::strcmp(name, "exl3_mapshare") != 0) return 0;
    if (value < 0 || value > 1) return -1;
    rad::arch::fn4_exl3_map_flag().store(value, std::memory_order_relaxed);
    return 1;
}
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {
    return qwen4exp_fp8::pass_key(c, batch, used_max_h);
}
