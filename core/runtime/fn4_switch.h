/* fn4_switch.h -- runtime switches for in-process A/B blocks (test builds only).
 *
 * Ported to the DeepSeek-V4.1 line (ds41-gg, 10.11.2026) from fn4-c16 2d532ecc / 91f566cd: the core machinery only
 * (the registry, POST /admin/fn4, the apply at the step release, the epoch in the pass key, draft_adaptive,
 * epoch_bump), without fn4's qwen4exp and hc_* switches, so an A/B of an exact lever runs in one process and one
 * load instead of two starts (start-to-start spread is ~1-1.5 % on one config).
 *
 * A switch is set from the environment at start and may be changed while the server runs through POST /admin/fn4.
 * A request is checked whole before anything is queued: an unknown switch, a value the owner of the switch does not
 * take (the plugin that serves it, the deployment's drafter), or a switch nothing serves is refused in the response
 * (400, the reason by name), and nothing of that body is queued. An accepted change is queued by the server thread and
 * applied by the scheduler thread at the next step release (Engine::prepare_release), while every rank thread is
 * parked, so all ranks issue each step under one setting. Every applied change advances an epoch that Ctx::pass_key
 * mixes in, so no recorded pass from before the change is played after it. That holds for an architecture with a
 * pass-key callback and for one without (deepseek41 exports none; the core keys its decode, verify and draft passes
 * by the bucketed bound): the epoch is mixed into the core's own key after the callback's part, whichever applies.
 * At epoch 0 nothing is mixed in, so with the environment unset and no request every key is today's to the bit and
 * the same passes are recorded.
 *
 * GET reports what is APPLIED, read from its owner, never only what was asked:
 *   draft_adaptive  scheduler: 1 the adaptive draft window, 0 fixed -- the scheduler's own state (DraftController::
 *                   adaptive()), registered at its init and read back after every change; -1 when no scheduler has
 *                   registered. A request takes 1 / 0 / -1 (= back to --draft-window); 1 is refused when the
 *                   deployment's drafter cannot run the adaptive window ("draft_capable": 0).
 *   gg_gemv         libr4d (rad_fn4_*): gemm_nt_ggml's decode band (up to 48 rows), 0 the stock rows-a-wave / grid
 *                   rule, 1 the measured per-(type, N, K, M) form table (r4d_ds41_gg_forms.h), 2 lever D2's
 *                   multi-row form (r4d_ds41_ggml_mr.hip, gemm_nt_ggml and logits_gemm_ggml; what it declines runs
 *                   the stock rule); every form has the stock kernel's per-output arithmetic, so the bytes are the
 *                   same whichever runs. The value is the one libr4d applied (it reads RADIANCE_DS41_GG_GEMV at load
 *                   and takes only the forms it knows; another value there is the stock rule, named on stderr); -1
 *                   when no loaded object serves the switch. A request for a value the serving plugin does not know
 *                   is refused.
 *   drafter_mx      libr4d (rad_fn4_*): moe_gemm_ggml type=mxfp4 (the DSpark drafter's routed experts), 0 the stock
 *                   kernels, 1 the multi-row forms (r4d_ds41_mxfp4_mr.hip) in the stock path's summation order, the
 *                   same bits; RADIANCE_DS41_DRAFTER_MX=1 at load. -1 when no loaded object serves it.
 *   drafter_bf      libr4d (rad_fn4_*): gemm_nt bf16's decode band (the drafter's attention / main_proj / shared
 *                   expert), 0 the stock skinny WMMA kernel, 1 its prefetching form (r4d_gemm_bf16_nt_pf.hip), the
 *                   same chains and so the same bits; RADIANCE_DS41_DRAFTER_BF=1 at load.
 *   aql_acq         core (Ctx::haz_mark + the AQL replay): the address-based acquire rule in force, 0 off, 1 rule,
 *                   2 rule-w -- the switch once set, else RADIANCE_AQL_ACQ=rule | rule-w at start (ds41-ar-acq
 *                   4c9e6c2's start-time rule, now read through here). A change re-records every pass (the epoch),
 *                   since haz_mark marks the records at recording. RADIANCE_AQL_ACQ=none stays start-time only.
 *
 * "epoch_bump": 1 in a request starts a new epoch without changing a switch, so an A/A block pair re-records
 * exactly as an A/B pair does.
 *
 * A plugin takes part by exporting (each optional but the first)
 *   extern "C" int rad_fn4_switch(const char* name, int value)    apply: 1 applied, 0 not its switch, < 0 refused
 *   extern "C" int rad_fn4_accepts(const char* name, int value)   check only: 1 it would apply it, 0 not its switch,
 *                                                                 < 0 its switch but not that value
 *   extern "C" int rad_fn4_value(const char* name, int* applied)  1 and the value it applies, 0 not its switch
 * The kernel and architecture plugins are dlopen'ed RTLD_LOCAL, so the core finds the hooks in every loaded object
 * through its own handle (fn4_switch.cpp). */
#ifndef RAD_RUNTIME_FN4_SWITCH_H
#define RAD_RUNTIME_FN4_SWITCH_H

#include <cstdint>
#include <string>

namespace rad {
namespace fn4 {

enum Switch { kDraftAdaptive = 0, kGgGemv, kAqlAcq, kDrafterMx, kDrafterBf, kSwitches };

const char* switch_name(int i);
int  switch_index(const std::string& name);   /* -1 when unknown */
int  value(int i);                            /* the applied value, as GET reports it */
uint64_t epoch();                             /* applied changes so far; 0 until the first */
/* The AQL acquire rule: 0 off, 1 rule, 2 rule-w -- the aql_acq switch once set, else RADIANCE_AQL_ACQ (rule |
 * rule-w) at start. aql_acq_none: RADIANCE_AQL_ACQ=none (the unsafe experiment; start-time only). */
int  aql_acq_rule();
bool aql_acq_none();

/* The scheduler, at its init: whether its drafter can run the adaptive window, and whether it does. */
void draft_register(bool capable, bool adaptive);
/* The scheduler thread, after it applied a draft_adaptive change: the window the scheduler now runs. */
void note_draft_adaptive(bool adaptive);

/* Whether a request may set `name` to `value` (epoch_bump included); false with the reason. */
bool check(const std::string& name, int value, std::string* err);
/* Server thread: queue a value; false (with the reason) for anything check() refuses. */
bool request(const std::string& name, int value, std::string* err);
/* Scheduler thread, every rank thread parked: apply what is queued. Returns true when anything
 * changed; *draft_adaptive is the draft window to set: 1 adaptive, 0 fixed, kNoChange when it did
 * not change (a request for -1 arrives here as the configured window). */
constexpr int kNoChange = -2;
bool apply_pending(int step, int* draft_adaptive);
/* {"switches": {...}, "pending": {...}, "epoch": n, "applied_step": n, "plugins": n, "draft_capable": n} */
std::string state_json();

}  // namespace fn4
}  // namespace rad

#endif
