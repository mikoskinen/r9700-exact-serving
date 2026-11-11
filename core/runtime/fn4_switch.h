/* fn4_switch.h -- runtime switches for in-process A/B blocks (fn4, 9.11.2026; test builds only).
 *
 * A switch is set from the environment at start (RADIANCE_FN4_*), and may be changed while the server
 * runs through POST /admin/fn4. A change is queued by the server thread and applied by the scheduler
 * thread at the next step release (Engine::prepare_release), while every rank thread is parked, so all
 * ranks issue each step under one setting. Every applied change advances an epoch that Ctx::pass_key
 * mixes in, so no recorded pass from before the change is played after it.
 *
 *   no_route        core (Ctx::route_report): no routing reports                  RADIANCE_FN4_NO_ROUTE
 *   gather_inv      architecture plugin (rad_fn4_switch): the fused gather inverse RADIANCE_FN4_GATHER_INV
 *                   -- runtime changes need RADIANCE_FN4_SWITCHABLE=1 at start (both paths declared)
 *   hc_fusedown     libr4d (rad_fn4_switch): hc_read's folded down reduce          RADIANCE_FN4_HC_FUSEDOWN
 *                   -- 1 the first tail, 2 the low-register tail (v2)
 *   hc_normwave     libr4d (rad_fn4_switch): hc_read's wave norm scales            RADIANCE_FN4_HC_NORMWAVE
 *                   -- 1 at every row count, N >= 2 from N rows
 *   draft_adaptive  scheduler: --draft-window adaptive (1) or fixed (0); -1 = the configured window
 *   hc_writevec     libr4d (rad_fn4_switch): hc_write eight elements a thread      RADIANCE_FN4_HC_WRITEVEC
 *                   -- 1 at every row count, N >= 2 from N rows
 *   hc_readv        libr4d (rad_fn4_switch): hc_read's faster forms, a mask         RADIANCE_FN4_HC_READV
 *                   -- 1 the decode tiles (17-64 rows), 2 PF2's down product (prefill), 4 PF2's norm scales in the down product,
 *                   8 the norm scales from the preceding hc_write (W1b; the trunk hands them over only when started with
 *                   RADIANCE_FN4_HC_SS=1, which declares their buffer), 16 / 32 / 48 the codes stage at 2 / 4 / 5 groups a
 *                   wave (W2)
 *   ar_lanethreads  libr4d (rad_fn4_switch): the prefill all-reduce lane's block    RADIANCE_FN4_AR_LANE_THREADS
 *                   width -- 0 the resolved tunable, 256 / 512 / 1024
 *   hc_upmix        libr4d (rad_fn4_switch): hc_read's up-mix tile                  RADIANCE_FN4_HC_UPMIX
 *                   -- value & 7 PF2's form (0 stock, 1-4), + 8 the 64-row decode tile pipelined
 *   ar_laneblocks   libr4d (rad_fn4_switch): the prefill all-reduce lane's blocks   RADIANCE_FN4_AR_LANE_BLOCKS
 *                   -- 0 the resolved tunable (4), 4..24 in steps of 4
 *   aql_acq         core (Ctx::haz_mark + the AQL replay): the address-based acquire rule, 0 off, 1 rule,
 *                   2 rule-w (cohbench2 7.11.: a CU can serve a line it stored after another CU rewrote it, so
 *                   rule-w is the safe one); -1 until set = RADIANCE_AQL_ACQ=rule | rule-w at start
 *   exl3_pfprep     exl3r4d (rad_fn4_switch): gate_up's prep in the EXL3 prefill band -- 0 the build's (the map,
 *                   then the prep by map entry), 2 the map and the prep by row in one launch; no start value (the
 *                   plugin reads no environment), so it is set through POST /admin/fn4 only
 *
 *   exl3_dz         exl3r4d (rad_fn4_switch): fn4 DZ at run time -- the decode band's spare workgroups zero their own
 *                   dead rows directly (1) or by the scan (0); a plugin built at EXL3R4D_DZ 2 takes both  RADIANCE_FN4_X3DZ
 *   exl3_bi         exl3r4d (rad_fn4_switch): fn4 BI -- the decode band computes every row as the prefill band does (no
 *                   EPIC in down, no small-M split-K in gate_up); 0 the adopted forms                RADIANCE_FN4_X3BI
 *   exl3_dual       architecture plugin (rad_fn4_switch): fn4 BI-dual -- every trunk MoE layer's EXL3 down takes the
 *                   step's row bands (libr4d bi_band, exl3 interface v12): a mixed step's decode rows get the decode
 *                   band's bytes in the prefill band's pass; needs RADIANCE_FN4_X3DUAL=1 at start (declares the
 *                   bands)                                                                            RADIANCE_FN4_X3DUAL
 *   moe_pair        architecture plugin (rad_fn4_switch): fn4 P2x, a pipelined prefill step's two pieces through
 *                   ONE routed-expert pass (0 off, 1 the shared arm a piece after it, 2 the shared arms inside it);
 *                   needs RADIANCE_FN4_P2X=1 at start (the expert buffers declared two pieces tall)  RADIANCE_FN4_MOE_PAIR
 *
 * Start-time only (not switches): RADIANCE_FN4_HC_SS=1 (qwen4exp declares the hc_write -> hc_read scale buffer),
 * RADIANCE_FN4_CHAIN=1 (qwen4exp's probe says chained launches pay, so the core turns them on at bring-up).
 *
 * "epoch_bump": 1 in a request starts a new epoch without changing a switch, so an A/A block pair
 * re-records exactly as an A/B pair does.
 *
 * A plugin takes part by exporting  extern "C" int rad_fn4_switch(const char* name, int value):
 * 1 = applied, 0 = not its switch, < 0 = refused. */
#ifndef RAD_RUNTIME_FN4_SWITCH_H
#define RAD_RUNTIME_FN4_SWITCH_H

#include <cstdint>
#include <string>

namespace rad {
namespace fn4 {

enum Switch { kNoRoute = 0, kGatherInv, kHcFuseDown, kHcNormWave, kDraftAdaptive, kHcWriteVec,
              kHcReadV, kArLaneThreads, kHcUpMix, kArLaneBlocks, kAqlAcq, kExl3PfPrep, kExl3MapShare, kExl3Gd, kMoePair, kExl3Dz, kExl3Bi,
              kExl3Dual, kSwitches };

const char* switch_name(int i);
int  switch_index(const std::string& name);   /* -1 when unknown */
int  value(int i);                            /* the applied value */
bool no_route();
uint64_t epoch();                             /* applied changes so far */
/* The AQL acquire rule (core-acq 68cfbee3, ported 10.11.): 0 off, 1 rule, 2 rule-w -- the aql_acq switch once set,
 * else RADIANCE_AQL_ACQ (rule | rule-w) at start. A change re-records every pass (the epoch), since haz_mark marks
 * the records at recording. aql_acq_none: RADIANCE_AQL_ACQ=none (the unsafe experiment; start-time only). */
int  aql_acq_rule();
bool aql_acq_none();

/* Server thread: queue a value; false (with the reason) for an unknown switch or a bad value. */
bool request(const std::string& name, int value, std::string* err);
/* Scheduler thread, every rank thread parked: apply what is queued. Returns true when anything
 * changed; *draft_adaptive is the draft window to set (-1 when it did not change). */
bool apply_pending(int step, int* draft_adaptive);
/* {"switches": {...}, "pending": {...}, "epoch": n, "applied_step": n, "plugins": n} */
std::string state_json();

}  // namespace fn4
}  // namespace rad

#endif
