/* fn4_switch.h -- runtime switches for in-process A/B blocks (test builds only).
 *
 * Ported to the DeepSeek-V4.1 line (ds41-gg, 10.11.2026) from fn4-c16 2d532ecc / 91f566cd: the core machinery only
 * (the registry, POST /admin/fn4, the apply at the step release, the epoch in the pass key, draft_adaptive,
 * epoch_bump), without fn4's qwen4exp and hc_* switches, so an A/B of an exact lever runs in one process and one
 * load instead of two starts (start-to-start spread is ~1-1.5 % on one config).
 *
 * A switch is set from the environment at start and may be changed while the server runs through POST /admin/fn4.
 * A change is queued by the server thread and applied by the scheduler thread at the next step release
 * (Engine::prepare_release), while every rank thread is parked, so all ranks issue each step under one setting.
 * Every applied change advances an epoch that Ctx::pass_key mixes in, so no recorded pass from before the change
 * is played after it. That holds for an architecture with a pass-key callback and for one without (deepseek41
 * exports none; the core keys its decode, verify and draft passes by the bucketed bound): the epoch is mixed into
 * the core's own key after the callback's part, whichever applies. At epoch 0 nothing is mixed in, so with the
 * environment unset and no request every key is today's to the bit and the same passes are recorded.
 *
 *   draft_adaptive  scheduler: --draft-window adaptive (1) or fixed (0); -1 = the configured window
 *   gg_gemv         libr4d (rad_fn4_switch): gemm_nt_ggml's decode band (up to 48 rows), 0 the stock     RADIANCE_DS41_GG_GEMV
 *                   rows-a-wave / grid rule, 1 the measured per-(type, N, K, M) form table
 *                   (r4d_ds41_gg_forms.h); the forms differ only in which wave and block compute an output,
 *                   never in its arithmetic, so the bytes are the same either way
 *   aql_acq         core (Ctx::haz_mark + the AQL replay): the address-based acquire rule, 0 off, 1 rule,
 *                   2 rule-w; -1 until set = RADIANCE_AQL_ACQ=rule | rule-w at start (ds41-ar-acq 4c9e6c2's
 *                   start-time rule, now read through here). A change re-records every pass (the epoch), since
 *                   haz_mark marks the records at recording. RADIANCE_AQL_ACQ=none stays start-time only.
 *
 * "epoch_bump": 1 in a request starts a new epoch without changing a switch, so an A/A block pair re-records
 * exactly as an A/B pair does.
 *
 * A plugin takes part by exporting  extern "C" int rad_fn4_switch(const char* name, int value):
 * 1 = applied, 0 = not its switch, < 0 = refused. The kernel and architecture plugins are dlopen'ed RTLD_LOCAL, so
 * the core finds the hook in every loaded object through its own handle (fn4_switch.cpp). */
#ifndef RAD_RUNTIME_FN4_SWITCH_H
#define RAD_RUNTIME_FN4_SWITCH_H

#include <cstdint>
#include <string>

namespace rad {
namespace fn4 {

enum Switch { kDraftAdaptive = 0, kGgGemv, kAqlAcq, kSwitches };

const char* switch_name(int i);
int  switch_index(const std::string& name);   /* -1 when unknown */
int  value(int i);                            /* the applied value */
uint64_t epoch();                             /* applied changes so far; 0 until the first */
/* The AQL acquire rule: 0 off, 1 rule, 2 rule-w -- the aql_acq switch once set, else RADIANCE_AQL_ACQ (rule |
 * rule-w) at start. aql_acq_none: RADIANCE_AQL_ACQ=none (the unsafe experiment; start-time only). */
int  aql_acq_rule();
bool aql_acq_none();

/* Whether a request may set `name` to `value` (epoch_bump included); false with the reason. */
bool check(const std::string& name, int value, std::string* err);
/* Server thread: queue a value; false (with the reason) for an unknown switch or a bad value. */
bool request(const std::string& name, int value, std::string* err);
/* Scheduler thread, every rank thread parked: apply what is queued. Returns true when anything
 * changed; *draft_adaptive is the draft window to set: 1 adaptive, 0 fixed, -1 back to the
 * configured one, kNoChange when it did not change. */
constexpr int kNoChange = -2;
bool apply_pending(int step, int* draft_adaptive);
/* {"switches": {...}, "pending": {...}, "epoch": n, "applied_step": n, "plugins": n} */
std::string state_json();

}  // namespace fn4
}  // namespace rad

#endif
