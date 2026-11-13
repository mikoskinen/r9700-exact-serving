#!/bin/bash
# dsm-yd2m.sh (12.11.2026, queue-ar/dsmain/yd2m, the A4 builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m.sh. Log tag "RESULT ydm ", out runs/xdsm/yd2m/x1 (YDM_OUT). A COPY of xpush/dsm-xpush.sh
# (itself a minimal copy of prof12/dsm-prof1.sh; tag xpu -> ydm, knobs XPU_ -> YDM_; the originals not changed; unused functions kept):
# the SAME start, pins, guards and image checks (int12's literals + lever A4's instead of int13's, gg_gemv_m16 0 in the base pin, the
# switch never in the env); P0, the arms, the readings and the summary are A4's. prof12/p12win.py is read in place; yd2m/ydsum.py sums.
# LEVER A4 (ds41-a4 = int12 bf8f557 + A4, libr4d fn4 gg_gemv_m16, 63): under gg_gemv 2 a decode-band call of M <= 16 rows takes the A4
#   table's form -- lever D2's multi-row GEMV with the slab's activation loads one slab ahead (XP), the grid fit's waves a block, H1R2,
#   MT up to 16 -- for gemm_nt_ggml, the 1U-GQ pair, folds a / e and the head; the per-output operation sequence is D2's (the stock
#   kernel's bytes at every row count). The table: RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt, written by this item's P0 bench.
# P0 (ONE CARD BY CHIP UUID, YDM_P0_UUID GPU-dbba5c851cfef7b3, before the start, no server):
#   (1) THE GATE BENCH r4d_selftest --perfm16 M 5..16 (YDM_PERFMS), 100 launches (YDM_PERFIT), two passes: D2's own pick (gg_gemv 2 = int12)
#       vs the A4 candidates at the c8 profile's dense shapes (attn_q_b / fold a, q_a + kv pair, attn_output_b, attention_output_a pair,
#       shared expert gate_up / down / fold e at rank 0's and ranks 1-3's widths, engram_kv at E2's 6400, the head), weights from DRAM
#       (rotated copies); bytes vs D2 and the stock band; M16GATE (M 8 / 16), M16EST (the c8 / c16 profile's calls a step x the saving).
#       The table = the keys whose fastest candidate beat D2 by >= 3 % in BOTH passes -> $O/stats/m16-forms.txt.
#   (2) THE BYTE CHECK r4d_selftest --mrcheck16 with that table loaded: every A4 form (16 pins) + the switch itself vs the stock band,
#       byte for byte, M 1..16, every DS dense shape (q5_K / q6_K / q8_0, the head f32), the pair (q_a + kv, out_a g0 + g1), folds a / e:
#       A4VERDICT PASS. A FAILURE STOPS THE ITEM (no start).
#   THE GATE: an empty table or a c8 bench saving under YDM_GATE_US (250) us a step stops A4 after P0 (not a VIRHE; YDM_FORCE=1 screens
#   anyway): the P0 bench is then A4's reading, logged "A4 gate NOT passed (...); A10 part runs alone", and THE A10 PART RUNS ALONE by
#   default (arms B / P: identity a1 / gs16 under B and P + a1-B vs ref12's a1, the staggered check under B, cold 136k B then P, c8
#   P B B P with the P12SPLIT newcomer-step ms, the A10 rule line); YDM_A10_ALONE=0 stops the item there instead.
# ONE START of the A4 image (YDM_IMAGE), ref12's pins exactly (read back) + gg_gemv_m16 0 in the base pin: S5T, draft_sample 0, SP 0 / 0,
#   M1 on, IX off, draft_depth0 3, sched_row_cap 48 (fn4 CAP48), ds41_arcoded 0, CBH 0 (k 2), exl3_x3d 1, ds41_e2 1, ds41_gq 3,
#   ds41_mix_wv 0 (MX), ds41_shexp_lane 0 (F4), RADIANCE_XPREFETCH=0, KVR=1 in the env, + RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt
#   (the P0 table; the start log must name its N forms: tabline); MAXSEQS YDM_MAXSEQS 16; the copy path (no ngshare, no ATTACH).
# LEVER A10 (the coordinator's extra arm; ds41-pfx 8b52bd9 merged: fn4 exl3_x3d_pf 63, served by A10's exl3r4d 90b1fef bb2d91f6 in this
#   image): exl3_x3d_pf 1 = the M > 16 band's K = 2 map-form expert GEMMs (newcomer / prefill steps) read a lane's eight trellis states
#   from one word pair; the same bytes. Its P0 = A10's staged pfxprobe (ypfx/bin, read in place; OLD int12 plugin vs this image's x3d_pf
#   0 / 1, K 2 + 2.5, M 1 / 17 / 64 / 512 / 2048) on the same card: a failure drops every P reading (VIRHE line), A4 goes on.
# ARMS (in-process, fn4, read back): B = gg_gemv_m16 0 + exl3_x3d_pf 0 (= int12), C = gg_gemv_m16 1 (the A4 table), P = exl3_x3d_pf 1
#   (A10, gg_gemv_m16 0).
# READINGS (in this order):
#   IDENTITY (strict, T 0): warm-up answers vs stock (identw); the greedy set c1 alone (a1) and at conc 16 (gs16, all in flight, cap 48)
#     under B, then C, then P: a1-C vs a1-B, gs16-C vs gs16-B, gs16-B vs a1-B, gs16-C vs a1-C (batch invariance under C), a1-B and a1-C
#     vs ref12's a1, a1-P vs a1-B, gs16-P vs a1-P; > 48-row steps during each gs16 = 0.
#   STAGGERED (strict, T 0; zepb's epbstag): YDM_LDCONC (4) decode streams + the 20 greedy probes one at a time, each with a FRESH
#     YDM_LDPF (8192)-token prefill joining 0.5 s in, so the probes' decode rows ride MIXED steps (stack4/loadid.py run); ld-B vs a1-B
#     right after the identity, and ld-C vs a1-C at the end when the budget allows (YDM_STAGC 1). A difference is a VIRHE line with the
#     first differing char / token; the item goes on. (EPB's staggered check failed 1 / 20; none ran on int12's path since 12.11. 02:09.)
#   A10 COLD: deepq 136000 (YDM_PDEPTHS) after a prefix-cache reset, 64 greedy tokens, B once then P once: TTFT, non-decode ms a step;
#     the answers byte-compared.
#   C8: the fixed harness (harness/ncmb3.py, the fixed prompt set reset per block, 1x8 seed 1, window time [10, 45] s), arms YDM_C8ORDER
#     (P B C C B P: A4's B C C B intact, A10's P at both ends; every arm's mean block position 3.5); per block p12win.py's wall split:
#     decode steps' ms a step and rows a step (A4's TARGET), newcomer / MIXED steps' ms (A10's), agg,
#     decode_share. int12 reference (prof1): decode 36.18 ms a step at 8.47 rows, agg 184.0. NO-REGRESSION GUARD: C's c8 decode-step ms
#     (step log) not above B's by more than YDM_DECTH (1.0) %.
#   1U ESSAY (the cheap confirm): per arm in YDM_U1ORDER (CB) ONE discarded warm-up read + YDM_NREADS (3) essay reads back to back
#     (stepprobe essay 512, T 0): decode tok/s, ms a decode step, stream MB a step (A4 does not apply at 1-2 rows: a no-change check).
#   C16: the same harness at --n 16 (1x16), arms YDM_C16ORDER (BC): decode ms a step, rows, agg, decode_share, > 48-row steps (0).
#   YDSUM (yd2m/ydsum.py): C vs B and P vs B for every reading -- the values, B as the reference, the % difference -- + the P0 gate
#     lines + "A10 joins the stack if 136k TTFT >= 1 % faster OR newcomer step >= 15 ms faster": PASS / NOT.
# GUARDS: ref12's (every switch read back, 0 request errors, the "i11" / "cbh" counters INFO, KVR read back) + the A4 table line. ~40 min
#   (P0 A4 ~6 + A10 ~3, start ~4.5, checks ~1, identity ~4.5, staggered B ~5, A10 cold ~2.5, c8 ~7.5, 1u ~1.5, c16 ~2.5, staggered C
#   ~5 if the budget allows, capture + summary ~1.5);
#   BUDGET YDM_BUDGET_MIN (50: a section past it skipped, logged). NO evict, no cc_sim, no power / host / fan settings; production compose files read only. At the end the raw
#   snapshots (/stats, /admin/fn4, /metrics, the 2 s samplers) go into one tar.gz; the engine logs are gzipped as captured; p0-check gzipped.
# MOCK: YDM_SW=0 = a server without the int9b .. A4 switches (queue-ar/tools/mockrad): arms B and C post nothing; P0 not emulated.
# DRYRUN=1 bash dsm-yd2m.sh : prints every command; checks the image literals (int9 .. int12 + A4, the plugin md5), every helper (+
#   selftests: prof12/p12win.py, yd2m/ydsum.py, the harness), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE}; RADIANCE_AQL_STAMP from AQLSTAMP)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${YDM_OUT:-$R/xdsm/yd2m/x1}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${YDM_IMAGE:?set YDM_IMAGE}
ITEM=${YDM_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (the ARC screen's base)
SW=${YDM_SW:-1}
PLAN=""; PLAN2=""; PLAN3=""   # no lever groups: the reference reads (refreads)
F4V=${YDM_F4:-0}; F4G=${YDM_F4G:-0}; MXC=${YDM_MXCONFIRM:-0}; NTOK=${YDM_NTOK:-64}; DEPTHS=${YDM_DEPTHS:-}; GSC=${YDM_GSCONC:-16}
I11DIR=${YDM_I11DIR:-$R/xdsm/i11u}; I11ID=${YDM_I11ID:-$R/xdsm/i11u/x1u-a-O.jsonl}; R12S=$HD/../prof12/p12win.py; YDS=$HD/ydsum.py; C48N=$HD/../cap48/ncmb.py; C48REF=${YDM_C48REF:-223.2}; C16DUR=${YDM_C16DUR:-30}; C16WARM=${YDM_C16WARM:-10}; NREADS=${YDM_NREADS:-3}
CBHK=${YDM_CBHK:-2}; EPR=${YDM_EPROMPTS:-essay,code}
MXV=${YDM_MX:-12}; EGEN=${YDM_EGEN:-512}; ARCV=0; DOID=0; STAMPN=0
MHC=${YDM_MHC:-1}; D0=${YDM_D0:-3}; CAP=${YDM_CAP:-48}; SPN=0; SPK=0
DOD=${YDM_D:-1}
P0=""   # no lever P0 (p0sel); A4's own P0 on one card (by chip UUID): p0a4
PERFMS=${YDM_PERFMS:-5,6,7,8,9,10,11,12,13,14,15,16}; PERFIT=${YDM_PERFIT:-100}; P0PERFS=${YDM_P0PERFS:-900}; P0CHECKS=${YDM_P0CHECKS:-900}
GATEUS=${YDM_GATE_US:-250}; FORCE=${YDM_FORCE:-0}; A4ON=1; GATEWHY=""
LDCONC=${YDM_LDCONC:-4}; LDPF=${YDM_LDPF:-8192}; STAGC=${YDM_STAGC:-1}   # the staggered-arrival identity (zepb's epbstag): streams, fresh prefill tokens; also under C
# lever A10 (ds41-pfx, fn4 exl3_x3d_pf; the coordinator's extra arm P): its P0 (pfxprobe, A10's staged ypfx/bin read in place), cold 136k
# B then P, the P blocks in the c8 order, a1 / gs16 under P. PFOK=0 (its P0 failed) drops every P reading (a VIRHE line), A4 goes on.
A10=${YDM_A10:-1}; PB=${YDM_PFXBIN:-$HD/../ypfx/bin}; PFTMO=${YDM_PFTMO:-600}; PFOLDMD5=9823cddefe8c9b06baedeb3e8070b99c; PFOK=1
PDEPTHS=${YDM_PDEPTHS-136000}; NTOK=${YDM_NTOK:-64}; PNCTH=${YDM_PNC_MS:-15}; PTTH=${YDM_PTTFT_PCT:-1}
case $A10 in 0|1) ;; *) echo "YDM_A10 0 / 1"; exit 1 ;; esac
for v in $PFTMO $PNCTH $PTTH $PDEPTHS; do case $v in ''|*[!0-9.]*) echo "an A10 knob is not a number ('$v')"; exit 1 ;; esac; done
case $PERFMS in *[!0-9,]*|'') echo "YDM_PERFMS rows 1..16, comma separated"; exit 1 ;; esac
for v in $PERFIT $P0PERFS $P0CHECKS $GATEUS $FORCE $LDCONC $LDPF $STAGC; do case $v in ''|*[!0-9]*) echo "a P0 knob is not a number ('$v')"; exit 1 ;; esac; done
CORD=${YDM_CORDER-}; C8ORD=${YDM_C8ORDER-PBCCBP}; U1ORD=${YDM_U1ORDER-CB}; C16ORD=${YDM_C16ORDER-BC}; DECTH=${YDM_DECTH:-1.0}; P0MIN=${YDM_P0MIN:-12}
for o in "$CORD" "$U1ORD" "$C16ORD"; do case $o in *[!BC]*) echo "YDM_U1ORDER / YDM_C16ORDER take B and C (got '$o')"; exit 1 ;; esac; done
case $C8ORD in *[!BCP]*) echo "YDM_C8ORDER takes B, C and P (A10) (got '$C8ORD')"; exit 1 ;; esac
case $DECTH in ''|*[!0-9.]*) echo "YDM_DECTH is a percentage (got '$DECTH')"; exit 1 ;; esac
case $P0MIN in ''|*[!0-9]*) echo "YDM_P0MIN a number"; exit 1 ;; esac
DISKMIN=${YDM_DISKMIN_MB:-11000}; BUDGET=${YDM_BUDGET_MIN:-50}; ALLOWOLD=${YDM_DRY_ALLOW_OLD:-0}
SIDE=${YDM_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${YDM_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "YDM_SW is 1 (the item) or 0 (a server without the int9b / int10 / int11 switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $MXV $EGEN $ARCV $DOID $STAMPN $MHC $D0 $CAP $DOD $DISKMIN $BUDGET $CBHK ${YDM_C8WIN:-35} ${YDM_C8WARM:-10} ${YDM_C4WIN:-20} ${YDM_C4WARM:-8} ${YDM_C1WIN:-20} ${YDM_C1WARM:-6} ${YDM_CSPACING:-6} ${YDM_C8X:-0} ${YDM_C4ID:-0} ${YDM_C8GENS:-0}; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case $MXV in 0|12|16|24) ;; *) echo "YDM_MX 0 / 12 / 16 / 24 (ds41_mix_wv's values)"; exit 1 ;; esac
case $F4V in ''|*[!0-9]*) echo "YDM_F4 0 / 1 / 2..64 (ds41_shexp_lane)"; exit 1 ;; esac; [ $F4V -le 64 ] || { echo "YDM_F4 0..64"; exit 1; }
[ $F4V -le 1 ] || [ $F4G = 1 ] || { echo "YDM_F4 $F4V (a row threshold) needs the F4G image: set YDM_F4G=1 (its literal is then checked)"; exit 1; }
for v in $F4G $MXC $NTOK $GSC; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
for v in $DEPTHS; do case $v in ''|*[!0-9]*) echo "YDM_DEPTHS takes token counts (got '$v')"; exit 1 ;; esac; done
[ -z "${YDM_MAXSEQS:-}" ] || case ${YDM_MAXSEQS} in *[!0-9]*) echo "YDM_MAXSEQS a number"; exit 1 ;; esac
[ $ARCV -le 1048576 ] || { echo "YDM_ARC 0..1048576 (ds41_arcoded's values)"; exit 1; }
[ $EGEN -ge 64 ] || { echo "YDM_EGEN >= 64"; exit 1; }
[ $D0 -le 3 ] || { echo "YDM_D0 0..3"; exit 1; }
[ $CAP -le 4096 ] || { echo "YDM_CAP 0..4096"; exit 1; }
[ $MHC -le 1048576 ] || { echo "YDM_MHC 0..1048576 (mhc_fuse's values)"; exit 1; }
case ${YDM_C8TH:-1.0} in ''|*[!0-9.]*) echo "YDM_C8TH is a percentage (got '${YDM_C8TH:-}')"; exit 1 ;; esac
[ $CBHK -ge 2 ] && [ $CBHK -le 64 ] || { echo "YDM_CBHK 2..64 (sched_cb_hold_k)"; exit 1; }
for e in ${YDM_C8PLAN-S:O}; do case $e in [SC]:O|[SC]:N|[SC]:H[0-9]*) ;; *) echo "YDM_C8PLAN entries are S|C:O|N|H<ms> (got '$e')"; exit 1 ;; esac; done
case $EPR in essay|essay,code) ;; *) echo "YDM_EPROMPTS essay or essay,code (got '$EPR')"; exit 1 ;; esac
for o in "${YDM_C1ORDER-OO}" "${YDM_C4ORDER-OO}"; do case $o in *[!ON]*) echo "the c1 / c4 orders take O and N (got '$o')"; exit 1 ;; esac; done
for c in $P0; do case $c in *:*[!0-9:]*|*::*|*:*:*:*) echo "YDM_P0 takes case:min_ok:seconds (got '$c')"; exit 1 ;; esac; done
for g in $PLAN $PLAN2 $PLAN3; do
  case ${g%%:*} in MX|MX12|GQ|E2|F4|C1T) ;; *) echo "YDM_PLAN groups are MX MX12 GQ E2 F4 C1T (got '${g%%:*}')"; exit 1 ;; esac
  for a in $(echo ${g#*:} | tr ',' ' '); do case $a in O|MX|MX12|GQ|E2|F4|N|NH|H[0-9]*|C[1-9][0-9]|C5) ;; *) echo "YDM_PLAN arm '$a' unknown (O MX MX12 GQ E2 F4 N C5..C95)"; exit 1 ;; esac; done
done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py                                                                                    # stack4's, read only
CTR=$HD/../i9b/i9bctr.py                                                                            # i9b's (INFO only), read only
NCMB=$HD/../nc/ncmb.py                                                                              # nc's, read only (its selftest only)
MBT2=$HD/../harness/mbt3.py; NCMB2=$HD/../harness/ncmb3.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only): c1 / c4 / c8 on mbt3's fixed schedule + FIXED PROMPT SET (TESTING.md 12.11. c4 METHOD FIX)
C4R=$HD/../c4lev/c4rows.py                                                                          # c4lev's: the step log by a step's rows, read in place
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
MBT=$S4/mbt.py; EP=$S4/editprobe.py; PRB=$S4/sdprobe.py                                            # stack4's, read only (d0blocks.py args)
D0B=$HD/../d0ab/d0blocks.py                                                                         # d0ab's (+ its s5ctr.py beside it), read only
U1S=$HD/../1u/u1sum.py                                                                              # the screen's (steps by verify rows), read in place
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT ydm $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= the ARC screen's + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1; KVR off)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS
  RADIANCE_DS41_MHC_FUSE=0 RADIANCE_DS41_GQ=3 RADIANCE_DS41_E2=1 RADIANCE_DS41_KV_REPLICATED=1"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD|now,0):,the,Engram,lookup,/,copy,/,kv,GEMM,(25600,x,6144)|rows,at,2,Engram,layer(s)"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up|stage_short,can,arm,the,prefill,stager,for,a,step,of,1025,..,2112,tokens|place_hb,can,reserve,on,rank,1"
# + lever M1's declare line (int9b; KVR off here: its lines are not looked for, readback checks they are absent; GQ / E2's lines: readback)
LITS="$LITS|lever,M1,(mhc_fuse),declared"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
# ---------------------------------------------------------------- the base pin (S5T, draft_sample 0, SP pinned, M1 on, IX off, CAP48, every new lever 0) and the arms
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # YDM_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
MJ="{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0, \"draft_depth0\": $D0, \"sched_row_cap\": $CAP, \"ds41_arcoded\": 0, \"ds41_mix_wv\": $MXV, \"ds41_gq\": 3, \"ds41_e2\": 1, \"ds41_shexp_lane\": $F4V, \"sched_cb_hold_ms\": 0, \"sched_cb_hold_k\": $CBHK, \"exl3_x3d\": 1, \"gg_gemv_m16\": 0, \"exl3_x3d_pf\": 0}"
[ $SW = 1 ] || MJ="{\"ds41_shexp_lane\": $([ $F4V -le 1 ] && echo $F4V || echo 0)}"   # the mock: the int8 switches only (F4's lane is int3's)
GQOK=1; E2OK=1; MXOK=1; GQP0=1; F4OK=1   # the levers available (readback / P0 may clear them: their arms are then SKIPPED, logged)
GQDECL=1; E2DECL=1                # declared on this image (readback: GET "i11" declared > 0); the start values swnote expects
armj() {   # armj ARM: the arm's own switch (JSON) on top of the base; under YDM_SW=0 only the int8 switches (F4, C1T)
  case $1 in
    O) echo '{}' ;;
    M0) [ $SW = 1 ] && echo '{"ds41_mix_wv": 0}' || echo '{}' ;;
    MX) [ $SW = 1 ] && echo "{\"ds41_mix_wv\": $MXV}" || echo '{}' ;;
    MX12) [ $SW = 1 ] && echo '{"ds41_mix_wv": 12}' || echo '{}' ;;
    GQ) [ $SW = 1 ] && echo '{"ds41_gq": 3}' || echo '{}' ;;
    E2) [ $SW = 1 ] && echo '{"ds41_e2": 1}' || echo '{}' ;;
    F4) echo '{"ds41_shexp_lane": 1}' ;;
    B) [ $SW = 1 ] && echo '{"gg_gemv_m16": 0}' || echo '{}' ;;   # A4's arms (before C*: dspark_conf's)
    C) [ $SW = 1 ] && echo '{"gg_gemv_m16": 1}' || echo '{}' ;;
    P) [ $SW = 1 ] && echo '{"exl3_x3d_pf": 1}' || echo '{}' ;;   # lever A10 (the extra arm): gg_gemv_m16 0
    C*) echo "{\"dspark_conf\": ${1#C}}" ;;
    H*) [ $SW = 1 ] && echo "{\"sched_cb_hold_ms\": ${1#H}, \"sched_cb_hold_k\": $CBHK}" || echo '{}' ;;
    N|NH) python3 -c 'import json,sys
sw, mxv, mx, gq, e2 = sys.argv[1] == "1", int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1", sys.argv[5] == "1"
d = {"ds41_shexp_lane": 1 if sys.argv[6] == "1" else 0}
if sw:
    if mx: d["ds41_mix_wv"] = mxv
    if gq: d["ds41_gq"] = 3
    if e2: d["ds41_e2"] = 1
    if sys.argv[7] == "NH": d.update(sched_cb_hold_ms=500, sched_cb_hold_k=int(sys.argv[8]))
print(json.dumps(d))' $SW $MXV $MXOK $GQOK $E2OK $F4OK $1 $CBHK ;;
  esac
}
NAMEOF() { case $1 in O) echo "O (THE REFERENCE PIN)" ;; M0) echo "M0 (the reference with ds41_mix_wv 0)" ;; MX) echo "MX (ds41_mix_wv $MXV)" ;; MX12) echo "MX12 (ds41_mix_wv 12)" ;;
  GQ) echo "GQ (ds41_gq 3)" ;; E2) echo "E2 (ds41_e2 1)" ;; F4) echo "F4 (ds41_shexp_lane 1)" ;;
  B) echo "B (gg_gemv_m16 0 + exl3_x3d_pf 0: int12)" ;; C) echo "C (gg_gemv_m16 1: lever A4's forms, the P0 table)" ;; P) echo "P (exl3_x3d_pf 1: lever A10, gg_gemv_m16 0)" ;; C*) echo "$1 (dspark_conf ${1#C})" ;;
  N) echo "N (every available int11 lever on, hold 0)" ;; NH) echo "NH (every available new lever on + sched_cb_hold_ms 500)" ;;
  H*) echo "$1 (sched_cb_hold_ms ${1#H}, k $CBHK)" ;; *) echo "$1" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$MJ" "$(armj $1)" "$x"
}
DESC="A4 SCREEN + A10 ARM (ds41-a4 = int12 + lever A4 + lever A10's merge, arms B = gg_gemv_m16 0 + exl3_x3d_pf 0 / C = gg_gemv_m16 1 / P = exl3_x3d_pf 1 in-process; the base pin both 0) NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 (1U-GQ / 1U-E2 declared) + RADIANCE_DS41_KV_REPLICATED=1 (KVR) $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN (card 0, every ${STAMPN}th pass)" || echo "(no stamping)") | SP / ARC / CBH off, copy path | reference pin = stack5's S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM
  RADIANCE_DS41_IDX_BALANCE RADIANCE_DS41_QNFOLD RADIANCE_DS41_SWFOLD RADIANCE_DRAFT_CTXSKIP RADIANCE_DS41_EMB_RAM
  RADIANCE_DS41_DSPARK_CONF RADIANCE_DS41_DSPARK_ALT RADIANCE_DS41_KVWIN_FP8 RADIANCE_DS41_DRAFTER_MX RADIANCE_DS41_DRAFTER_BF
  RADIANCE_DS41_GG_PF RADIANCE_DS41_GG_PF_TABLE RADIANCE_DS41_F32_PF RADIANCE_DS41_NVGATHER RADIANCE_DS41_DSPARK_WALK
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_DRAFT_SAMPLE RADIANCE_WEIGHT_SIDECAR
  RADIANCE_DS41_BI RADIANCE_DS41_BITRACE RADIANCE_DS41_BITRACE_PMAX RADIANCE_DRAFT_CTXSKIP_SLOTS RADIANCE_DRAFT_LEND_WEIGHTS
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_PLACE_HB_RANKS
  RADIANCE_PLACE_HB_DECAY RADIANCE_DS41_PSK_L20 RADIANCE_DS41_DSPARK_PFCTX RADIANCE_DS41_PSK_CR RADIANCE_SCHED_BI_CHUNK
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED RADIANCE_DEBUG_TIER_VERIFY
  RADIANCE_DS41_MHC_FUSE RADIANCE_DS41_IX_LDS RADIANCE_DS41_IX_NI RADIANCE_DS41_ARCODED RADIANCE_DECODE_ROW_CAP
  RADIANCE_DS41_MIX_WV RADIANCE_DS41_GQ RADIANCE_DS41_E2 RADIANCE_AQL_STAMP RADIANCE_AQL_STAMP_OUT RADIANCE_FN4_AR_XPUSH
  RADIANCE_DS41_GG_M16 RADIANCE_DS41_GG_M16_FORMS"
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xarc -> xpu, via arc/dsm-arc.sh): stackenv .. gate1 (modes: + nc/dsm-ncprof-p.sh's rule-w check under --profile-ops)
# (vramstart / gate1 show stack5's s5-n5 start beside this one)
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
capture() {   # capture FILE: the engine log with the daemon's receipt times (ncsteps.py reads them), gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before the start (sahb's)
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (the start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse the start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "$1 warm-up $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, step-log lines, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; step-log lines (RADIANCE_LOG_STEPS=1); literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  if [ -n "${PROFEVERY:-}" ]; then   # --profile-ops records no pass (ctx.cpp tape_on_ needs !profiling_): env rule-w, no plain rule line (nc/dsm-ncprof-p.sh)
    if echo "$CE" | grep -qx "RADIANCE_AQL_ACQ=rule-w" && [ "$NR" = 0 ]; then xlog "$1: rule-w replaced by the profile sync (not a speed): --profile-ops records no pass, RADIANCE_AQL_ACQ=rule-w in the container env, no plain rule line ($NW / $NR)"
    else xlog "$1: VIRHE RADIANCE_AQL_ACQ=rule-w not in the container env (or a plain rule line) under --profile-ops ($NW / $NR)"; ok=0; fi
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  fi
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever E1|lever SA / HB|lever BI declared|lever SD|lever KVR|lever M1|held once a slot|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|prefill staging:" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-3600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT ydm $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS5/s5-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack5 s5-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
ARMOK=1
arm() {   # arm ARM: the pin + the arm + epoch_bump, a nudge, the switches READ BACK (polled <= 30 s) -> 0, else VIRHE + 1
  local g r i
  fn4set "$(pinj $1 '{"epoch_bump": 1}')"; nudge
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; return 0; }
  for i in $(seq 1 30); do
    g=$(fn4get)
    r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
print("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"]))' "$g" "$(pinj $1)" 2>&1 | cut -c1-400)
    case "$r" in OK*) return 0 ;; esac
    sleep 1; nudge
  done
  xlog "arm $1: VIRHE the pin is not applied after 30 s ($r) -- no reading for this arm here"; ARMOK=0; return 1
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,re,sys
def rd(p):
    d = {}
    try:
        for l in open(p):
            m = re.match(r"^(radiance:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", l)
            if m: d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    except OSError: pass
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda k: b.get("radiance:" + k, 0.0) - a.get("radiance:" + k, 0.0)
nd = g("engine_steps_prefill_total") + g("engine_steps_mixed_total")
nds = g("engine_step_seconds_total") - g("engine_step_seconds_decode_total")
dt, da = g("draft_tokens_total"), g("draft_accepted_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da),
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
pskstat() {   # the live lever counters (cumulative, /stats): PSK, PSK-L20, PSK-CR, PSK-DS, E1
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  curl -s -m 20 $URL/stats 2>/dev/null | python3 -c 'import json,sys
s = json.load(sys.stdin); p = s.get("psk") or {}; e = s.get("engdd") or {}; f = s.get("dspark_pfctx") or {}
print("psk mode %s k %s latched %s planned %s cut %s rows %s layers %s mismatch %s | l20 %s / %s | cr %s hits %s not_needed %s fb %s tail %s | pfctx %s / %s | engdd %s / %s" % (
    p.get("mode"), p.get("k"), p.get("latched"), p.get("planned_steps"), p.get("cut_passes"), (p.get("rows_skipped_l21_29") or 0) + (p.get("rows_skipped_l30_39") or 0),
    p.get("layers_skipped"), p.get("plan_mismatch"), p.get("l20_cut_passes"), p.get("l20_rows_skipped"), p.get("cr"), p.get("cr_hits"), p.get("cr_not_needed"),
    p.get("cr_fallbacks"), p.get("tail_only_latched"), f.get("passes"), f.get("rows_skipped"), e.get("passes"), e.get("rows_deduped")))' 2>/dev/null | cut -c1-500
}
ssnap() {   # ssnap NAME: the full /stats and GET /admin/fn4 (the counter guard's snapshot)
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json; GET /admin/fn4 > $O/fn-$1.json"; return 0; }
  curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; curl -s -m 10 $URL/admin/fn4 > $O/fn-$1.json 2>/dev/null
}
sdelta() {   # sdelta A B -> JSON: checkpoints written, staged bytes (all ranks / rank 0), rank-0 staged layers, all-reduce bytes
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda d, *k: (lambda v: v if isinstance(v, (int, float)) else 0)(__import__("functools").reduce(lambda x, y: (x or {}).get(y) if isinstance(x, dict) else None, k, d))
print(json.dumps(dict(ckpts=int(g(b, "checkpoints_written") - g(a, "checkpoints_written")),
      staged_gb=round((g(b, "link", "staged_bytes") - g(a, "link", "staged_bytes")) / 1e9, 2),
      staged_gb_r0=round((g(b, "experts", "staged_bytes") - g(a, "experts", "staged_bytes")) / 1e9, 2),
      stage_layers_r0=int(g(b, "experts", "stage_layers") - g(a, "experts", "stage_layers")),
      ar_gb=round((g(b, "link", "ar_bytes") - g(a, "link", "ar_bytes")) / 1e9, 2))))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack5 s5-n5 $(cat $RS5/guard-s5-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414): alone, cmpx
alone() {   # alone TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0 (20 of 20 or a VIRHE)
  local tag=$1 t0=$(date +%s) n
  arm $2 || return 1
  pcreset $tag-$3
  ssnap $tag-$3-0
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  ssnap $tag-$3-1
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-$3.jsonl 2>/dev/null || echo 0)
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $(NAMEOF $2) $(armj $2)): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/$tag-$3.txt | cut -c1-160), $(( $(date +%s) - t0 )) s | INFO counters: $(ctrinfo $tag-$3-0 $tag-$3-1)"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1-$2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
budok() {   # budok NAME EST_MIN [LIMIT]: 0 = run the section; 1 = skip it (the time now + its estimate would pass the budget), logged (sp's)
  local lim=${3:-$BUDGET}
  [ "$DRY" = 1 ] && { echo "+ budget: $1 runs if elapsed + ~$2 min <= $lim min, else skipped (logged)"; return 0; }
  [ $(( $(elapsed) + $2 )) -le $lim ] && return 0
  xlog "$1 SKIPPED (budget): $(elapsed) min elapsed + ~$2 min > $lim min"; return 1
}
ctrinfo() {   # ctrinfo A B: every counter delta between the snapshots (i9b/i9bctr.py, no need / forbid: INFORMATION, never a VIRHE)
  [ "$DRY" = 1 ] && { echo "(dry: python3 -I -B $CTR guard --st0 st-$1 --st1 st-$2 --fn0 fn-$1 --fn1 fn-$2, printed as INFO)"; return 0; }
  timeout 60 python3 -I -B $CTR guard --st0 $O/st-$1.json --st1 $O/st-$2.json --fn0 $O/fn-$1.json --fn1 $O/fn-$2.json 2>&1 | tail -1 | sed 's/^OK | //' | cut -c1-900
}
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (e1/dsm-e1.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nodraft
  [ $ok = 1 ]
}
capd() {   # capd A B -> one line: the CAP48 "cap" deltas between GET /admin/fn4 snapshots A and B (in_force: B's)
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cap")
    except Exception: return None
a, b = rd(sys.argv[1]) or {}, rd(sys.argv[2])
if b is None: print("no \"cap\" in GET /admin/fn4"); sys.exit(0)
d = {k: (b.get(k, 0) - a.get(k, 0)) for k in ("steps_capped", "drafts_trimmed", "ahead_capped", "steps_over", "steps_gt48", "decode_steps") if isinstance(b.get(k), (int, float))}
print("cap in_force %s, max_rows %s | %s" % (b.get("in_force"), b.get("max_rows"), ", ".join("%s +%s" % kv for kv in d.items())))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null | cut -c1-400
}
# ---------------------------------------------------------------- this item's own functions
i11now() {   # i11now: GET /admin/fn4's new switches + the base's int9b / int10 ones and the "i11" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("ds41_mix_wv %s ds41_gq %s ds41_e2 %s ds41_shexp_lane %s dspark_conf %s dspark_conf_rows %s | mhc_fuse %s draft_depth0 %s sched_row_cap %s ds41_arcoded %s | i11 %s" % (
    w.get("ds41_mix_wv"), w.get("ds41_gq"), w.get("ds41_e2"), w.get("ds41_shexp_lane"), w.get("dspark_conf"), w.get("dspark_conf_rows"),
    w.get("mhc_fuse"), w.get("draft_depth0"), w.get("sched_row_cap"), w.get("ds41_arcoded"), json.dumps(d.get("i11"))))' 2>/dev/null | cut -c1-700
}
i11d() {   # i11d A B -> JSON: the "i11" counter deltas between GET /admin/fn4 snapshots A and B (present: whether B has "i11")
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("i11")
    except Exception: return None
a, b = rd(sys.argv[1]) or {}, rd(sys.argv[2])
o = {"present": b is not None}
for g, ks in (("gq", ("qkv_pairs", "out_pairs")), ("e2", ("splits", "rows")), ("mx", ("wide_launches", "wide_rows"))):
    for k in ks:
        x, y = (a.get(g) or {}).get(k, 0), ((b or {}).get(g) or {}).get(k, 0)
        o[g + "_" + k] = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
print(json.dumps(o))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo '{}'
}
i11g() {   # i11g ARM DELTAJSON: the counter guard of one read by the arm's full pin -> "" (ok) or a VIRHE text; YDM_SW=0: INFO only
  [ "$DRY" = 1 ] && return 0
  python3 -c 'import json,sys
d = json.loads(sys.argv[2] or "{}"); pin = json.loads(sys.argv[3] or "{}"); sw = sys.argv[4] == "1"
if not sw: print("INFO (YDM_SW=0) i11 %s" % json.dumps(d)); sys.exit(0)
if not d.get("present"): print("VIRHE GET /admin/fn4 has no \"i11\" counters"); sys.exit(0)
got = {"mx": d.get("mx_wide_launches") or 0, "gq": (d.get("gq_qkv_pairs") or 0) + (d.get("gq_out_pairs") or 0), "e2": d.get("e2_splits") or 0}
on = {"mx": (pin.get("ds41_mix_wv") or 0) > 0, "gq": (pin.get("ds41_gq") or 0) > 0, "e2": (pin.get("ds41_e2") or 0) > 0}
miss = [k for k in got if on[k] and not got[k] > 0]; extra = [k for k in got if not on[k] and got[k]]
if miss: print("VIRHE path did not run: no %s launch counted (on in the pin)" % "/".join(miss))
elif extra: print("VIRHE %s counted while off in the pin" % "/".join(extra))' "$1" "$2" "$(pinj $1)" $SW 2>&1 | head -1
}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"   # P0: CARD 0 only
monons() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock (host = container)
utcnow() { date +%s.%N; }                                                                      # the step lines' clock (docker logs -t, UTC)
p0sel() {   # p0sel: ONE card (CARD 0), no server -- the image's r4d_selftest --case mix_wv / gq_pair; a failed case SKIPs its lever's arms
  local c cs nmin tmo r rc f nok nfail nbyte lv
  for c in $P0; do
    cs=${c%%:*}; nmin=$(echo $c | cut -d: -f2); tmo=$(echo $c | cut -d: -f3); f=$O/p0-$cs.txt
    case $cs in mix_wv) lv=MX ;; gq_pair) lv=GQ ;; *) lv=- ;; esac
    if [ "$DRY" != 1 ] && [ $cs = gq_pair ] && [ $SW = 1 ] && ! timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF -- gq_pair /opt/radiance/bin/r4d_selftest' > /dev/null 2>&1; then
      xlog "P0 $cs: SKIP -- the image's r4d_selftest has no case gq_pair (the GQ commit's selftest is not in this image); GQ's arms rest on the identity guard"; continue
    fi
    run "gpurun ydm-p0-$cs-$$ $tmo $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $cs > $f 2>&1"; rc=$?
    if [ "$DRY" = 1 ]; then echo "+ RESULT ydm P0 $cs on CARD 0: rc 0, 0 FAIL, >= $nmin ok 'byte for byte' lines (bounded at $tmo s), else VIRHE and lever $lv's arms SKIPPED (the item goes on); gq_pair only where the image's selftest has the case; info lines logged"; continue; fi
    nok=$(grep -c '^  ok' $f); nfail=$(grep -c '^  FAIL' $f)
    case $cs in gq_pair) nbyte=$(grep '^  ok' $f | grep -c '== two launches') ;; *) nbyte=$(grep '^  ok' $f | grep -c 'byte for byte') ;; esac
    r="$nok ok, $nfail FAIL, $nbyte byte-identity lines (need $nmin), rc $rc"
    while IFS= read -r l; do xlog "P0 $cs info: ${l:0:600}"; done < <(grep -E '^  info' $f | head -12)
    if [ $rc = 0 ] && [ "$nfail" = 0 ] && [ "$nbyte" -ge $nmin ]; then
      xlog "P0 $cs on one card (r4d_selftest --case): OK $r | $(grep -E '^  ok' $f | sed 's/^ *ok *//' | cut -c1-110 | tr '\n' ';' | cut -c1-1500)"
    elif [ $SW = 1 ]; then
      xlog "P0 $cs on one card (r4d_selftest --case): VIRHE $r -> lever $lv's arms SKIPPED | $(grep -E '^  FAIL|FATAL|error|Error|fault|abort|stopped after' $f | head -6 | sed 's/^ *//' | cut -c1-200 | tr '\n' ';' | cut -c1-1200)"
      case $lv in MX) MXOK=0 ;; GQ) GQOK=0; GQP0=0 ;; esac
    else
      xlog "P0 $cs: failed under YDM_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on ($r)"
    fi
  done
  return 0
}
readback() {   # readback TAG: the start's words read back -- MHC_FUSE=0 / GQ=3 / E2=1 / AQL_STAMP in the env, M1's line, GQ / E2's lines and "i11".declared; KVR / ARCODED / ROW_CAP NOT in the env
  [ "$DRY" = 1 ] && { echo "+ readback $1: RADIANCE_DS41_MHC_FUSE=0, RADIANCE_DS41_GQ=3, RADIANCE_DS41_E2=1, RADIANCE_AQL_STAMP=$STAMPN in the container env; 'lever M1 (mhc_fuse) declared'; 'lever 1U-GQ (ds41_gq ..) declared|not declared' and 'lever 1U-E2 (ds41_e2 ..) declared' lines; GET /admin/fn4 \"i11\" gq.declared / e2.declared (0 -> that lever's arms SKIPPED, logged); RADIANCE_DS41_KV_REPLICATED=1 in the env + the 'lever KVR declared' line; ARCODED / DECODE_ROW_CAP / SCHED_CB_HOLD_* NOT in the env"; return 0; }
  local ce e kv ac cp st d m lg le gd ed ok=1
  ce=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  e=$(echo "$ce" | grep -cxE 'RADIANCE_DS41_MHC_FUSE=0|RADIANCE_DS41_GQ=3|RADIANCE_DS41_E2=1|RADIANCE_DS41_KV_REPLICATED=1'); kv=$(echo "$ce" | grep -c '^RADIANCE_DS41_KV_REPLICATED=1$')
  ac=$(echo "$ce" | grep -c '^RADIANCE_DS41_ARCODED=\|^RADIANCE_SCHED_CB_HOLD_\|^RADIANCE_FN4_AR_XPUSH=\|^RADIANCE_DS41_GG_M16='); cp=$(echo "$ce" | grep -c '^RADIANCE_DECODE_ROW_CAP=.'); st=$( [ $STAMPN -gt 0 ] && echo "$ce" | grep -cx "RADIANCE_AQL_STAMP=$STAMPN" || { echo "$ce" | grep -q "^RADIANCE_AQL_STAMP=." && echo 0 || echo 1; })
  d=$(docker logs $CN 2>&1 | grep -m1 'lever KVR declared' | sed 's/^radiance: //' | cut -c1-200)
  m=$(docker logs $CN 2>&1 | grep -m1 'lever M1 (mhc_fuse) declared' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-300)
  lg=$(docker logs $CN 2>&1 | grep -m1 'lever 1U-GQ (ds41_gq' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  le=$(docker logs $CN 2>&1 | grep -m1 'lever 1U-E2 (ds41_e2' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  read -r gd ed < <(fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin).get("i11") or {}
print((d.get("gq") or {}).get("declared", -1), (d.get("e2") or {}).get("declared", -1))' 2>/dev/null || echo "-1 -1")
  [ "$e" = 4 ] && [ "$kv" = 1 ] && [ "$ac" = 0 ] && [ "$cp" = 0 ] && [ "$st" = 1 ] && { [ -n "$d" ] || [ $SW = 0 ]; } && [ -n "$m" ] || ok=0
  if [ $SW = 1 ]; then
    [ "${gd:--1}" -gt 0 ] 2>/dev/null || { GQOK=0; GQDECL=0; xlog "$1: NOTE lever 1U-GQ not available here (i11.gq.declared ${gd:-?}; '${lg:-no GQ line}') -> GQ's arms SKIPPED, N without GQ"; }
    [ "${ed:--1}" -gt 0 ] 2>/dev/null || { E2OK=0; E2DECL=0; xlog "$1: NOTE lever 1U-E2 not available here (i11.e2.declared ${ed:-?}; '${le:-no E2 line}') -> E2's arms SKIPPED, N without E2"; }
    [ -n "$lg" ] && [ -n "$le" ] || ok=0
  fi
  xlog "$1 read back: env MHC_FUSE=0 / GQ=3 / E2=1 / KV_REPLICATED=1 $e of 4, AQL_STAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN || echo "unset/empty") $st of 1, KV_REPLICATED=1 $kv (want 1), ARCODED / SCHED_CB_HOLD_* / FN4_AR_XPUSH / GG_M16 $ac (want 0: CBH and A4 come by fn4), DECODE_ROW_CAP $cp (want 0) | KVR line: ${d:-NONE} | M1: ${m:-NONE} | GQ: ${lg:-NONE} | E2: ${le:-NONE} | i11 declared gq ${gd:-?} e2 ${ed:-?}$([ $ok = 1 ] || echo " -- VIRHE the start's words are not read back")"
  [ $ok = 1 ] || [ $SW = 0 ]
}
swnote() {   # swnote TAG: start values; refusals; the arms O and N applied and read back
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, draft_sample 0$([ $SW = 1 ] && echo "; mhc_fuse / ix_lds / ix_ni / ds41_arcoded / sched_row_cap / ds41_mix_wv / ds41_shexp_lane 0, ds41_gq 3 and ds41_e2 1 where declared (they start on), \"i11\" present" || echo "; int9b / int10 / int11 NOT CHECKED (YDM_SW=0)")); refusals stage_short 65537$([ $SW = 1 ] && echo " / ds41_mix_wv 13 / ds41_gq 4 / ds41_e2 2 / draft_depth0 4 / sched_row_cap 4097 / sched_cb_hold_ms 60001 / sched_cb_hold_k 1 / exl3_x3d 2 / gg_gemv_m16 2") -> 400; sched_cb_hold_ms 0 / k 2, exl3_x3d 0, gg_gemv_m16 0 and "cbh" at start$([ $F4V -gt 0 ] && echo "; ds41_shexp_lane $F4V accepted"); THE REFERENCE PIN O applied and read back each applied and read back ($(pinj O | cut -c1-200)...)"; return 0; }
  local s0 g c1=400 c2=400 c3=400 c5=400 c6=400 c7=400 c8=400 c9=400 c10=400 c4 a sets="" bad=""
  s0=$(fn4get)
  c4=$(fn4code '{"stage_short": 65537}')
  [ $SW = 1 ] && { c1=$(fn4code '{"ds41_mix_wv": 13}'); c2=$(fn4code '{"ds41_gq": 4}'); c3=$(fn4code '{"ds41_e2": 2}'); c5=$(fn4code '{"sched_row_cap": 4097}'); c6=$(fn4code '{"draft_depth0": 4}'); c7=$(fn4code '{"sched_cb_hold_ms": 60001}'); c8=$(fn4code '{"sched_cb_hold_k": 1}'); }
  if [ $SW = 1 ] && [ $F4V -gt 0 ]; then   # the reference's ds41_shexp_lane: accepted only where the arm is lane-safe (and N >= 2 only with F4G)
    local cf; cf=$(fn4code "{\"ds41_shexp_lane\": $F4V}")
    [ "$cf" = 200 ] || { bad="$bad ds41_shexp_lane=$F4V(HTTP $cf)"; xlog "$1: VIRHE ds41_shexp_lane $F4V refused (HTTP $cf) -- the reference pin cannot be applied"; }
  fi
  if [ $SW = 1 ]; then c9=$(fn4code '{"exl3_x3d": 2}'); c10=$(fn4code '{"gg_gemv_m16": 2}'); fi
  for a in O; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("x3d=%s,cbh=%s/%s,mx=%s,gq=%s,e2=%s,f4=%s,conf=%s/%s,arc=%s,cap=%s,mhc=%s,d0=%s,lend=%s,occ=%s,ds=%s,psk=%s/%s,ep=%s" % (w.get("exl3_x3d"), w.get("sched_cb_hold_ms"), w.get("sched_cb_hold_k"), w.get("ds41_mix_wv"), w.get("ds41_gq"), w.get("ds41_e2"), w.get("ds41_shexp_lane"), w.get("dspark_conf"), w.get("dspark_conf_rows"), w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("draft_depth0"), w.get("draft_lend"), w.get("draft_occ"), w.get("draft_sample"), w.get("ds41_psk"), w.get("ds41_psk_k"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | i11 %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("gg_gemv_m16", "exl3_x3d_pf", "exl3_x3d", "sched_cb_hold_ms", "sched_cb_hold_k", "ds41_mix_wv", "ds41_gq", "ds41_e2", "ds41_shexp_lane", "dspark_conf", "dspark_conf_rows", "ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "draft_sample", "ds41_psk", "gg_gemv", "aql_acq")}), json.dumps(d.get("i11")), d.get("epoch")))' 2>/dev/null | cut -c1-1200) | stage_short 65537 -> HTTP $c4$([ $SW = 1 ] && echo ", ds41_mix_wv 13 -> HTTP $c1, ds41_gq 4 -> HTTP $c2, ds41_e2 2 -> HTTP $c3, sched_row_cap 4097 -> HTTP $c5, draft_depth0 4 -> HTTP $c6, sched_cb_hold_ms 60001 -> HTTP $c7, sched_cb_hold_k 1 -> HTTP $c8, exl3_x3d 2 -> HTTP $c9, gg_gemv_m16 2 -> HTTP $c10" || echo " | int9b / int10 / int11 NOT CHECKED (YDM_SW=0)") | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; gq, e2 = sys.argv[3] == "1", sys.argv[4] == "1"; w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("ds41_psk") == 0 and w.get("draft_sample") == 0
if sw:
    ok = ok and all(w.get(k) == 0 for k in ("mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "sched_row_cap", "ds41_mix_wv", "ds41_shexp_lane"))
    ok = ok and w.get("ds41_gq") == (3 if gq else 0) and w.get("ds41_e2") == (1 if e2 else 0) and isinstance(s0.get("i11"), dict)
    ok = ok and w.get("sched_cb_hold_ms") == 0 and w.get("sched_cb_hold_k") == 2 and isinstance(s0.get("cbh"), dict) and w.get("exl3_x3d") == 0 and w.get("gg_gemv_m16") == 0 and w.get("exl3_x3d_pf") == 0
sys.exit(0 if ok else 1)' "$s0" $SW $GQDECL $E2DECL 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ "$c4" = 400 ] && [ "$c5" = 400 ] && [ "$c6" = 400 ] && [ "$c7" = 400 ] && [ "$c8" = 400 ] && [ "$c9" = 400 ] && [ "$c10" = 400 ] && [ -z "$bad" ]
}
# ---------------------------------------------------------------- the mb blocks (fixed-schedule harness) and lever CBH's c8 arms
C8WIN=${YDM_C8WIN:-35}; C8WARM=${YDM_C8WARM:-10}; C4WIN=${YDM_C4WIN:-20}; C4WARM=${YDM_C4WARM:-8}; C1WIN=${YDM_C1WIN:-20}; C1WARM=${YDM_C1WARM:-6}
C8PLAN=${YDM_C8PLAN-S:O}; CSPACE=${YDM_CSPACING:-6}; C8GENS=${YDM_C8GENS:-0}; MB8TH=${YDM_C8TH:-1.0}; MB8X=${YDM_C8X:-0}
MBC4=${YDM_C4ORDER-OO}; MBC1=${YDM_C1ORDER-OO}
mbx() {   # mbx N I ARM SCHED: ONE mb c<N> block at T 0 on the harness's FIXED, SEEDED schedule (harness/ncmb2.py --mode phase --sched fixed
          # --pattern 1xN --seed 1 --ignore-eos 1 --window time [warm, warm + WIN]; SCHED S = the default spacing gen // N, C = --spacing
          # YDM_CSPACING 8 tokens: the newcomers clustered) -> $O/mb.jsonl (agg tok/s, TTFT p50 / p90, decode share, the cbh deltas)
  local n=$1 i=$2 a=$3 sc=$4 lb=${5:-$3} nm=c$1-$2-$4-${5:-$3} base=$O/c8/c$1-$2-$4-${5:-$3} t0 j dl cb g win warm sp=""
  case $n in 1) win=$C1WIN; warm=$C1WARM ;; 4) win=$C4WIN; warm=$C4WARM ;; *) win=$C8WIN; warm=$C8WARM ;; esac
  [ $sc = C ] && sp="--spacing $CSPACE"
  local wo="--window time"; [ $n = 8 ] && [ $C8GENS -gt 0 ] && wo="--window gens --gens $C8GENS --edge 0.5"
  ssnap $nm-0
  arm $a || return 1
  t0=$(date +%s)
  run "(cd $D && timeout $((win + warm + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n $n --temperature 0 --tag ydm-$nm --dur $((warm + win)) --warm $warm --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x$n $sp --seed 1 $wo --ignore-eos 1 --prompt-set fixed --reset-per-block 1 > $base.txt 2>&1)"
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm mb c$n $nm ($(NAMEOF $a) $(armj $a), schedule $sc): harness/ncmb3.py fixed 1x$n seed 1${sp:+ $sp}, the fixed prompt set (reset per block), $([ "$wo" = "--window time" ] && echo "window time [$warm, $((warm + win))] s" || echo "$wo (~37-46 s)"), ignore_eos, T 0: agg tok/s, TTFT p50 / p90, decode_share, 0 errors (else a request-error line); cbh counters$( [ $n = 8 ] && echo " + guard (a hold arm moves holds, a hold-0 arm none)"); i11 INFO -> $O/mb.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $base.txt | cut -c4-)
  [ -n "$j" ] || { xlog "mb c$n $nm: VIRHE request error: no MB line ($(tail -3 $base.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  dl=$(i11d $nm-0 $nm-1); cb=$(cbhd $nm-0 $nm-1); g=$(cbhg $a "$cb")
  python3 -c 'import json,sys,statistics
d = json.loads(sys.argv[1])
rows, marks = [], []
try:
    for l in open(sys.argv[7]):
        r = json.loads(l)
        if r.get("ev") == "mark": marks.append(r.get("t"))
        elif r.get("ev") == "done" and not r.get("err") and r.get("ttft") is not None: rows.append(r)
except (OSError, ValueError): pass
if len(marks) >= 2: rows = [r for r in rows if marks[0] <= (r.get("t_first") or 0) <= marks[1]]
tt = sorted(r["ttft"] for r in rows)
p90 = tt[min(len(tt) - 1, int(0.9 * (len(tt) - 1) + 0.5))] if tt else None
marks_ = marks[:2]
print(json.dumps(dict(name=sys.argv[2], n=int(sys.argv[3]), arm=sys.argv[4], sched=sys.argv[5], agg_dec_tps=d.get("agg_dec_tps"), decode_share=d.get("decode_share"),
      ms_decode_step=d.get("ms_decode_step"), tok_per_decode_step=d.get("tok_per_decode_step"), acceptance=d.get("acceptance"), marks=marks_,
      prompt_set=d.get("prompt_set"), reset=d.get("reset"),
      per_stream_dec_p50=d.get("per_stream_dec_p50"), ttft_p50=d.get("ttft_p50"), ttft_p90_win=round(p90, 3) if p90 is not None else None, ttft_n=len(tt),
      requests=d.get("requests"), errors=d.get("errors"), window=d.get("window"), window_s=d.get("window_s"), spacing=d.get("spacing"), mix_client=d.get("mix_client"),
      i11=json.loads(sys.argv[6] or "{}"), cbh=json.loads(sys.argv[8] or "{}"), guard=sys.argv[9])))' "$j" "$nm" $n $lb $sc "$dl" $base-reqs.jsonl "$cb" "$g" >> $O/mb.jsonl
  xlog "mb c$n $nm ($(NAMEOF $a) $(armj $a)$([ $lb = $a ] || echo ", a DISCARDED warm-up block: never read"), schedule $sc, ${wo#--window }, T 0, $(( $(date +%s) - t0 )) s): $([ $n = 8 ] && echo "$cb" | python3 -c 'import json,sys
d = json.loads(sys.stdin.read() or "{}")
print("cbh holds %s: by_k %s vs by_timeout %s (by_idle %s, by_other %s), held_steps %s, hold_ms %s, cob_steps %s, cob_newcomers %s, single_steps %s, in force %s ms k %s || " % (d.get("holds"), d.get("by_k"), d.get("by_timeout"), d.get("by_idle"), d.get("by_other"), d.get("held_steps"), d.get("hold_ms"), d.get("cob_steps"), d.get("cob_newcomers"), d.get("single_steps"), d.get("in_force_ms"), d.get("k")), end="")' 2>/dev/null)$(tail -1 $O/mb.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, acceptance %s, %s tok / %s ms a decode step, TTFT p50 %s s / p90 %s s (%s in the window), decode_share %s, per-stream p50 %s, prompts %s (reset %s), %s requests, %s errors%s | spacing %s, client mix %s" % (d.get("agg_dec_tps"), d.get("acceptance"), d.get("tok_per_decode_step"), d.get("ms_decode_step"), d.get("ttft_p50"), d.get("ttft_p90_win"), d.get("ttft_n"),
      d.get("decode_share"), d.get("per_stream_dec_p50"), d.get("prompt_set"), d.get("reset"), d.get("requests"), d.get("errors"), "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors",
      d.get("spacing"), json.dumps(d.get("mix_client"))[:160]))' 2>&1 | cut -c1-600) $([ -n "$g" ] && echo " -- $g") | i11 (INFO) $dl"
}
cbhd() {   # cbhd A B -> JSON: lever CBH's "cbh" deltas between GET /admin/fn4 snapshots A and B (+ in_force_ms / k from B)
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cbh")
    except Exception: return None
a, b = rd(sys.argv[1]) or {}, rd(sys.argv[2])
o = {"present": b is not None, "in_force_ms": (b or {}).get("in_force_ms"), "k": (b or {}).get("k")}
for k in ("holds", "by_k", "by_timeout", "by_idle", "by_other", "hold_ms", "held_steps", "cob_steps", "cob_newcomers", "single_steps"):
    x, y = a.get(k, 0), (b or {}).get(k, 0)
    o[k] = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
print(json.dumps(o))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo '{}'
}
cbhg() {   # cbhg ARM DELTAJSON: lever CBH's guard of one block -> "" (ok) or a VIRHE text; YDM_SW=0: INFO only
  [ "$DRY" = 1 ] && return 0
  python3 -c 'import json,sys
d = json.loads(sys.argv[2] or "{}"); a = sys.argv[1]; sw = sys.argv[3] == "1"
if not sw: print("INFO (YDM_SW=0)"); sys.exit(0)
if not d.get("present"): print("VIRHE GET /admin/fn4 has no \"cbh\" counters (not the int11 image with CBH?)"); sys.exit(0)
hold = int(a[1:]) if a.startswith("H") and a[1:].isdigit() else (500 if a == "NH" else 0)
if d.get("in_force_ms") != hold: print("VIRHE sched_cb_hold_ms in force %s, the arm says %d" % (d.get("in_force_ms"), hold)); sys.exit(0)
if hold > 0 and not (d.get("holds") or 0) > 0: print("VIRHE path did not run: no hold in a %d ms block" % hold); sys.exit(0)
if hold == 0 and ((d.get("holds") or 0) != 0 or (d.get("held_steps") or 0) != 0): print("VIRHE %s hold(s) / %s held step(s) counted at sched_cb_hold_ms 0" % (d.get("holds"), d.get("held_steps")))' "$1" "$2" $SW 2>&1 | head -1
}
mbdelta() {   # mbdelta N SCHED X -> one line: arm X vs O over $O/mb.jsonl's c<N> blocks on SCHED (agg tok/s, TTFT p50 / p90, decode share) + "DELTA d"
  python3 -c 'import json,sys
n, sc, x = int(sys.argv[2]), sys.argv[3], sys.argv[4]; R = []
try: R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError: pass
R = [r for r in R if r.get("n") == n and r.get("sched") == sc and r.get("agg_dec_tps")]
o = [r for r in R if r["arm"] == "O"]; y = [r for r in R if r["arm"] == x]
m = lambda v, k: (sum(r.get(k) or 0 for r in v) / len(v)) if v else None
sp = lambda v: 100.0 * (max(r["agg_dec_tps"] for r in v) - min(r["agg_dec_tps"] for r in v)) / m(v, "agg_dec_tps") if len(v) > 1 else None
d = 100.0 * (m(y, "agg_dec_tps") / m(o, "agg_dec_tps") - 1) if o and y else None
f = lambda v: "%.2f" % v if v is not None else "-"
if x == "O":
    print("c%d %s O: %s tok/s (blocks %s), O-O spread %s, TTFT p50 %s / p90 %s s" % (n, sc, f(m(o, "agg_dec_tps")), "/".join("%.2f" % r["agg_dec_tps"] for r in o) or "-", ("%.2f %%" % sp(o)) if sp(o) is not None else "-", f(m(o, "ttft_p50")), f(m(o, "ttft_p90_win"))))
else:
    cb = (y[0].get("cbh") or {}) if y else {}
    print("c%d %s %s vs O: %s -> %s tok/s (%s; blocks %s vs %s), TTFT p50 %s -> %s s, p90 %s -> %s s, decode_share %s -> %s%s" % (n, sc, x, f(m(o, "agg_dec_tps")), f(m(y, "agg_dec_tps")),
          ("%+.2f %%" % d) if d is not None else "-", "/".join("%.2f" % r["agg_dec_tps"] for r in y) or "-", "/".join("%.2f" % r["agg_dec_tps"] for r in o) or "-",
          f(m(o, "ttft_p50")), f(m(y, "ttft_p50")), f(m(o, "ttft_p90_win")), f(m(y, "ttft_p90_win")), f(m(o, "decode_share")), f(m(y, "decode_share")),
          (" | cbh holds %s (by_k %s, by_timeout %s, by_idle %s, by_other %s), cob_steps %s, cob_newcomers %s, single_steps %s, held_steps %s, hold_ms %s" % (cb.get("holds"), cb.get("by_k"), cb.get("by_timeout"), cb.get("by_idle"), cb.get("by_other"), cb.get("cob_steps"), cb.get("cob_newcomers"), cb.get("single_steps"), cb.get("held_steps"), cb.get("hold_ms"))) if x.startswith("H") else ""))
print("DELTA %s" % ("%.4f" % d if d is not None else "nan"))' $O/mb.jsonl $1 $2 $3 2>/dev/null
}
c8mixall() {   # c8mix.py block over every c8 block run so far (one engine-log capture), then compare all -> $O/c8/all-compare.out
  local f b ins=""
  capture $O/c8-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per c8 block: python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $O/c8/<b>-reqs.jsonl --tag ydm-<b> --mbline $O/c8/<b>.txt --label <b> --json $O/c8/<b>-mix.json; compare over them"; return 0; }
  [ -s $C8MIX ] || { xlog "c8: c8mix.py absent -- no mix reading"; return 0; }
  for f in $(ls $O/c8/c8-*-reqs.jsonl 2>/dev/null); do
    b=$(basename $f -reqs.jsonl)
    [ -s $O/c8/$b-mix.json ] || timeout 300 python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $f --tag ydm-$b --mbline $O/c8/$b.txt --label $b --json $O/c8/$b-mix.json > $O/c8/$b-mix.out 2>&1
    [ -s $O/c8/$b-mix.json ] && ins="$ins --in $b=$O/c8/$b-mix.json"
  done
  [ -n "$ins" ] && timeout 120 python3 -I -B $C8MIX compare $ins > $O/c8/all-compare.out 2>&1
}
c4warm() {   # c4warm WHY: ONE DISCARDED mb c4 block under O, once a start, right before the c4 arms (the spread item, int10 base: the FIRST
           # c4 block after the start read 141.7 vs 164.3 / 167.0, blocks 2-3 1.6 % apart) -- recorded as arm "wO", never read
  budok "c4 warm-up ($1)" 1 || return 0
  WI=$((WI + 1)); mbx 4 w$WI O S wO
}
WI=0
C8I=0
c8run() {   # c8run "SCHED:ARM ..": the c8 blocks in order (a block an entry)
  local e
  for e in $1; do C8I=$((C8I + 1)); stopfile && return 0; mbx 8 $C8I ${e#*:} ${e%%:*}; done
}
cbhexp() {   # cbhexp: the spaced schedule's expectation per hold block (INFO): by_timeout = holds, by_k 0, cob_steps as hold 0's, TTFT p50 - hold 0's
  python3 -c 'import json,sys
R = []
try: R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError: pass
R = [r for r in R if r.get("n") == 8 and r.get("sched") == "S"]
o = [r for r in R if r["arm"] == "O"]
ocob = (o[0].get("cbh") or {}).get("cob_steps") if o else None; ot = o[0].get("ttft_p50") if o else None
out = []
for r in R:
    if not r["arm"].startswith("H"): continue
    c = r.get("cbh") or {}
    ok = c.get("holds") == c.get("by_timeout") and not c.get("by_k") and c.get("cob_steps") == ocob
    dt = (r.get("ttft_p50") - ot) if (r.get("ttft_p50") is not None and ot is not None) else None
    out.append("%s: holds %s by_timeout %s by_k %s cob_steps %s (hold 0: %s), TTFT p50 %s s (%s)%s" % (r["arm"], c.get("holds"), c.get("by_timeout"), c.get("by_k"), c.get("cob_steps"), ocob,
               "%+.3f" % dt if dt is not None else "-", "+%.3f expected" % (int(r["arm"][1:]) / 1000.0), "" if ok else " -- NOTE not the expected cost-side pattern"))
print((" | expectation: " + "; ".join(out)) if out else "", end="")' $O/mb.jsonl 2>/dev/null
}
mbsec() {   # the c8 plan (lever no-regression N vs O on S; lever CBH hold 0 / 250 / 500 ms on S and C), c8mix; then c4 O N N O and c1 O O
  local i a k l d8 cmp e sc x
  if [ ! -s $NCMB2 ] || [ ! -s $MBT2 ]; then xlog "mb blocks: VIRHE harness/ncmb3.py / mbt3.py absent at run time -- no c1 / c4 / c8 reading"; return 0; fi
  xlog "mb blocks: harness/mbt3.py via ncmb3.py (fixed seeded schedule + the FIXED PROMPT SET reset per block, pattern 1xN, seed 1, ignore_eos, T 0, --window time = mb's [warm, dur]; mbt3.py $(md5sum < $MBT2 | cut -c1-8), ncmb3.py $(md5sum < $NCMB2 | cut -c1-8), c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8)) | c8 [$C8PLAN] gens $C8GENS (S = spacing gen // 8, C = --spacing $CSPACE) | c4 [$MBC4] window ${C4WIN} s | c1 [$MBC1] window ${C1WIN} s | CBH k $CBHK"
  [ -n "$C8PLAN" ] && budok "mb c8 [$C8PLAN]" $(( $(echo $C8PLAN | wc -w) * (C8WIN + C8WARM + 30) / 60 + 1 )) && { c8run "$C8PLAN"; c8mixall; }
  if [ -n "$MBC4" ] && budok "mb c4 [$MBC4] (+ its warm-up)" $(( (${#MBC4} + 1) * (C4WIN + C4WARM + 20) / 60 + 1 )); then
    c4warm "once a start, before the c4 arms"
    i=0; for a in $(echo $MBC4 | grep -o .); do i=$((i + 1)); stopfile && return 0; mbx 4 $i $a S; done
  fi
  i=0; [ -n "$MBC1" ] && budok "mb c1 [$MBC1]" $(( ${#MBC1} * (C1WIN + C1WARM + 12) / 60 + 1 )) && for a in $(echo $MBC1 | grep -o .); do i=$((i + 1)); stopfile && return 0; mbx 1 $i $a S; done
  [ "$DRY" = 1 ] && { echo "+ mb c8 second pairs (X then O on the schedule) for S:N, S:H500, C:H500 when |X vs O| < $MB8TH % or c8mix compare DIFFERS, budget allowing (logged when skipped) | RESULT ydm MB lines: c8 per schedule each arm vs O (agg, TTFT p50 / p90, decode share, cbh counters), c4 N vs O + O-O, c1 O-O"; return 0; }
  if [ "$MB8X" = 1 ] && [ -n "$C8PLAN" ]; then
    for e in S:N S:H500 C:H500; do
      sc=${e%%:*}; x=${e#*:}; echo " $C8PLAN " | grep -q " $e " || continue
      d8=$(mbdelta 8 $sc $x | grep '^DELTA ' | cut -c7-); cmp=$(grep -E "C8MIXCMP c8-[0-9]+-$sc-$x .*DIFFER" $O/c8/all-compare.out 2>/dev/null | wc -l)
      case $x in H*) cmp=0 ;; esac   # a CBH arm's mix DIFFERS from hold 0 by design (the lever moves the co-batch mix): the delta rule only
      if awk -v d="$d8" -v t=$MB8TH 'BEGIN{ exit !(d == "nan" || (d < t && d > -t)) }' || [ "${cmp:-0}" -gt 0 ]; then
        budok "mb c8 second pair $sc:$x $sc:O ($x vs O ${d8} %, threshold +-$MB8TH %, c8mix DIFFERS ${cmp:-0})" 3 && c8run "$sc:$x $sc:O"
      else xlog "mb c8 $sc $x: no second pair ($x vs O ${d8} %, beyond +-$MB8TH %; the mixes not DIFFERS)"; fi
    done
    c8mixall
  fi
  for b in $(ls $O/c8/c8-*-mix.out 2>/dev/null); do xlog "c8 mix $(basename $b -mix.out): $(grep -m1 '^C8MIX ' $b | cut -c7- | sed 's/ -- VIRHE / -- c8mix guard (INFO): /' | cut -c1-600)"; done
  while IFS= read -r l; do xlog "c8 mix compare (INFO): ${l:9:500}"; done < <(grep '^C8MIXCMP ' $O/c8/all-compare.out 2>/dev/null)
  for sc in S C; do
    l=$(mbdelta 8 $sc O | grep -v '^DELTA '); [ -n "$l" ] && echo "$l" | grep -q 'O: [0-9]' && xlog "MB $l"
    for x in N H250 H500; do l=$(mbdelta 8 $sc $x | grep -v '^DELTA '); case "$l" in *" -> -"*|*"- -> "*) ;; "") ;; *) xlog "MB $l" ;; esac; done
  done
  l=$(mbdelta 4 S N | grep -v '^DELTA '); [ -n "$l" ] && xlog "MB $l (one measured block an arm; O = the two confirmation blocks)"
  l=$(python3 -c 'import json,sys
R = []
try: R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError: pass
o = [r["agg_dec_tps"] for r in R if r.get("n") == 4 and r.get("arm") == "O" and r.get("agg_dec_tps")]
if o:
    m = sum(o) / len(o); sp = 100.0 * (max(o) - min(o)) / m if len(o) > 1 else None
    print("O c4 confirm: %.2f tok/s (blocks %s after one discarded warm-up; B-vs-B spread %s) vs int10 base 165.7 (164.3 / 167.0): %+.2f %%; vs target 160: %+.2f %%" % (
        m, "/".join("%.2f" % v for v in o), ("%.2f %%" % sp) if sp is not None else "-", 100.0 * (m / 165.7 - 1), 100.0 * (m / 160.0 - 1)))' $O/mb.jsonl 2>/dev/null)
  [ -n "$l" ] && xlog "$l"
  l=$(mbdelta 1 S N | grep -v '^DELTA '); [ -n "$l" ] && xlog "MB $l | $(mbdelta 1 S O | grep -v '^DELTA ') (O N N O: the O-O spread is the c1 B-vs-B pair)"
  for sc in S C; do
    l=$(for x in H250 H500; do mbdelta 8 $sc $x | grep '^DELTA ' | cut -c7- | awk -v x=$x '{ if ($1 == "nan") printf "%s -, ", x; else printf "%s %+.2f %%, ", x, $1 }'; done)
    [ -n "$l" ] && xlog "CBH schedule $sc ($([ $sc = S ] && echo "the default 1x8 spacing, newcomers ~2.3 s apart: COST SIDE -- expected by_timeout = holds, by_k 0, cob_steps unchanged, TTFT about +H, agg flat or slightly down" || echo "--spacing $CSPACE, newcomers ~0.2 s apart: UPPER BOUND ON GAIN, NOT A REALISTIC TTFT PRICE -- pairs persist across generations, so the holds are mostly in generation 1"); k $CBHK; agg tok/s vs hold 0): ${l%, }$([ $sc = S ] && cbhexp)"
  done
  return 0
}
NB=0
blockx() {   # blockx LEVER ARM [PROMPTS]: ONE 1u read at T 0 under ARM (stepprobe PROMPTS EGEN, default YDM_EPROMPTS), the step-log (UTC)
          # window around it, the i11 counter window from before the arm's POST, the /stats deltas (stream / h2d bytes, promotions) -> $O/blocks.jsonl
  local lv=$1 a=$2 i=$NB nm=b$NB-$1-$2 j jcode u0 u1 m0 m1 dl g st EPR=${3:-$EPR}
  NB=$((NB + 1))
  ssnap $nm-0
  arm $a || return 1
  if [ "$DRY" = 1 ]; then
    run "(cd $D && timeout 900 python3 $SPB $EPR $EGEN > $O/$nm.txt 2>&1)"
    echo "+ RESULT ydm block $i $lv $a ($(NAMEOF $a) $(armj $a)): stepprobe $EPR $EGEN at T 0 (base1113's 1u read) -> decode tok/s, tok/step, acceptance, ms/step; windows utc + monotonic around it; i11 counters from before the POST (guard: on block moves its counter, O none) -> $O/blocks.jsonl"
    ssnap $nm-1; return 0
  fi
  u0=$(utcnow); m0=$(monons)
  run "(cd $D && timeout 900 python3 $SPB $EPR $EGEN > $O/$nm.txt 2>&1)"
  m1=$(monons); u1=$(utcnow)
  ssnap $nm-1
  j=$(grep -m1 '^J ' $O/$nm.txt | cut -c3-); jcode=""; [ $EPR = essay,code ] && jcode=$(grep '^J ' $O/$nm.txt | grep -m1 '"prompt": "code"' | cut -c3-)
  [ -n "$j" ] && { [ $EPR != essay,code ] || [ -n "$jcode" ]; } || { xlog "block $i $lv $a: VIRHE request error: no J line ($(tail -3 $O/$nm.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  dl=$(i11d $nm-0 $nm-1); g=$(i11g $a "$dl")
  st=$(python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b, j = rd(sys.argv[1]), rd(sys.argv[2]), json.loads(sys.argv[3])
g = lambda d, x, y: ((d.get(x) or {}).get(y) or 0)
n = j.get("decode_steps") or 0
sb, hb = g(b, "link", "stream_bytes") - g(a, "link", "stream_bytes"), g(b, "link", "h2d_bytes") - g(a, "link", "h2d_bytes")
print(json.dumps(dict(stream_mb=round(sb / 1e6, 1), h2d_mb=round(hb / 1e6, 1), stream_mb_step=round(sb / 1e6 / n, 2) if n else None, h2d_mb_step=round(hb / 1e6 / n, 2) if n else None,
      promotions=int(g(b, "experts", "promotions") - g(a, "experts", "promotions")), staged_mb=round((g(b, "experts", "staged_bytes") - g(a, "experts", "staged_bytes")) / 1e6, 1))))' $O/st-$nm-0.json $O/st-$nm-1.json "$j" 2>/dev/null || echo '{}')
  python3 -c 'import json,sys
d = dict(i=int(sys.argv[1]), lever=sys.argv[2], arm=sys.argv[3], j=json.loads(sys.argv[4]), i11=json.loads(sys.argv[5] or "{}"),
         utc0=float(sys.argv[6]), utc1=float(sys.argv[7]), mono0=int(sys.argv[8]), mono1=int(sys.argv[9]), pin=json.loads(sys.argv[10]), st=json.loads(sys.argv[12] or "{}"))
if sys.argv[11]: d["jc"] = json.loads(sys.argv[11])
print(json.dumps(d))' $i $lv $a "$j" "$dl" $u0 $u1 $m0 $m1 "$(pinj $a)" "$jcode" "$st" >> $O/blocks.jsonl
  xlog "read $i $lv $a ($(NAMEOF $a) $(armj $a)$([ $lv = U1W ] && echo ", a DISCARDED warm-up read"), 1u stepprobe $EPR $EGEN T 0): $(echo "$j" | cut -c1-260)$([ -n "$jcode" ] && echo " | code $(echo "$jcode" | cut -c1-200)") | stream / promotions $st | i11 $dl$([ -n "$g" ] && echo " | $g")"
}
coldx() {   # coldx DEPTH [ARM I]: deepq's fixed prompt at DEPTH after a prefix-cache reset, NTOK greedy tokens -> TTFT, cached, status
           # (arc/dsm-arc.sh's, the ARC guard left out) -> $O/cold.jsonl; ARM (default O) read back first, I its place in the order
  local dep=$1 a=${2:-O} ci=${3:-0} nm=cold-$1${2:+-${3:-0}-$2} t0=$(date +%s) j md
  arm $a || return 1
  pcreset $nm
  msnap $nm-0; ssnap $nm-0
  run "(cd $D && timeout 1500 python3 $DQ $O/$nm.txt $dep $NTOK > $O/$nm.out 2>&1)"
  msnap $nm-1; ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm cold $nm: $(NAMEOF $a) $(armj $a), deepq $dep cold, $NTOK tok: TTFT, cached, status (200 or a request-error line) -> $O/cold.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$nm.out | cut -c7-)
  [ -n "$j" ] || { xlog "cold $nm: VIRHE request error: no answer ($(tail -2 $O/$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  md=$(mdelta $nm-0 $nm-1)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); m = json.loads(sys.argv[2] or "{}")
d.update(nondecode_ms=m.get("nondecode_ms"), nondecode_steps=m.get("nondecode_steps"), depth=int(sys.argv[3]), name=sys.argv[4], text=sys.argv[5], arm=sys.argv[6], i=int(sys.argv[7]))
print(json.dumps(d))' "$j" "$md" "$dep" "$nm" "$O/$nm.txt" $a $ci >> $O/cold.jsonl
  xlog "cold $nm ($(NAMEOF $a) $(armj $a), deepq $dep cold after a reset, $NTOK greedy tokens): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok | non-decode steps %s, %s ms mean%s%s" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
      round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-", d.get("tokens"), m.get("nondecode_steps"), m.get("nondecode_ms"),
      "" if d.get("status") == 200 else " -- VIRHE request error (status %s)" % d.get("status"), "" if not d.get("cached") else " -- NOT COLD (cached %s): left out of the TTFT line" % d.get("cached")))' "$j" "$md" 2>&1 | cut -c1-500) | $(thermline $t0 $(date +%s))"
}
refreads() {   # the reference readings in their order (module header), each under the reference pin (read back)
  local p k l
  if budok "1u reads (2 warm-ups + 4 warm reads)" 2; then
    for p in essay code; do blockx U1W O $p || true; done
    for k in 1 2; do for p in essay code; do stopfile && return 0; blockx U1 O $p || true; done; done
  fi
  if [ $MXC = 1 ] && [ $SW = 1 ] && [ $MXV = 0 ] && budok "MX12 confirm (mix_wv 12 vs the base 0, one warm essay pair, INFO)" 1; then blockx MXC MX12 essay || true; blockx MXC O essay || true; fi
  if [ $MXC = 1 ] && [ $SW = 1 ] && [ $MXV != 0 ] && budok "MX12 confirm (mix_wv 0 vs $MXV, one warm essay pair, INFO)" 1; then blockx MXC M0 essay || true; blockx MXC O essay || true; fi
  [ $MXC = 1 ] && [ $SW = 0 ] && xlog "MX12 confirm: not under YDM_SW=0 (the mock has no ds41_mix_wv)"
  if budok "identity (a1 vs int11 O; gs$GSC vs a1)" 3; then
    alone ydm O a1
    if [ -s "$I11ID" ] || [ "$DRY" = 1 ]; then cmpx ydm a1-vs-int11-O $I11ID $O/ydm-a1.jsonl
    else xlog "identity a1 vs int11 O: VIRHE the int11 O file $I11ID is missing -- not compared"; fi
    gsx
    cmpx ydm gs$GSC-vs-a1 $O/ydm-a1.jsonl $O/ydm-gs$GSC.jsonl
    if [ "$DRY" != 1 ]; then
      for k in a1-vs-int11-O gs$GSC-vs-a1; do
        l=$(grep "^ydm-$k " $O/identity.txt 2>/dev/null | tail -1)
        case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity $k: VIRHE $l" ;; esac ;; *) xlog "identity $k: VIRHE ${l:-not compared} (strict, T 0 greedy)" ;; esac
      done
    fi
  else xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile || { [ $DOD = 1 ] && mbsec; }
  for k in $DEPTHS; do stopfile && return 0; budok "cold $k" $(( k / 50000 + 1 )) && coldx $k; done
}
gsx() {   # gsx [ARM]: greedyset.py's 20 prompts at conc GSC (the cap48 item's gs16: c16 batch invariance with CAP48), temperature 0;
         # ARM (default O) -> $O/ydm-gs$GSC[-ARM].jsonl
  local t0=$(date +%s) n a=${1:-O} g=gs$GSC${1:+-$1}
  arm $a || return 1; pcreset ydm-$g
  run "(cd $D && timeout 900 python3 $GS $URL $O/ydm-$g.jsonl --conc $GSC --max-tokens $MT > $O/ydm-$g.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $g: greedy set at conc $GSC under $(NAMEOF $a) $(armj $a) (MAXSEQS ${YDM_MAXSEQS:-the base value}): 20 of 20 answered"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/ydm-$g.jsonl 2>/dev/null || echo 0)
  xlog "$g (greedy set at conc $GSC, temperature 0, $MT tok, $(NAMEOF $a) $(armj $a), MAXSEQS ${MAXSEQS:-?}): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/ydm-$g.txt | cut -c1-160), $(( $(date +%s) - t0 )) s"
}
runplan() {   # runplan "GROUP:ARM,ARM .. ..": the lever groups in order, a block an arm; an unavailable lever's group SKIPPED, the budget per group
  local grp lv arms a nb est
  for grp in $1; do
    lv=${grp%%:*}; arms=$(echo ${grp#*:} | tr ',' ' '); nb=$(echo $arms | wc -w); est=$(( (nb * (EGEN / 40 + 8) + 59) / 60 ))
    case $lv in
      MX|MX12) [ $MXOK = 1 ] || { xlog "group $lv SKIPPED: lever 1U-MX not available (P0)"; continue; } ;;
      GQ) [ $GQOK = 1 ] || { xlog "group GQ SKIPPED: lever 1U-GQ not available ($([ $GQP0 = 1 ] && echo "not declared on this image" || echo "P0 failed"))"; continue; } ;;
      E2) [ $E2OK = 1 ] || { xlog "group E2 SKIPPED: lever 1U-E2 not declared on this image"; continue; } ;;
      F4) [ $F4OK = 1 ] || { xlog "group F4 SKIPPED: ds41_shexp_lane 1 refused on this start"; continue; } ;;
    esac
    stopfile && return 0
    budok "group $lv ($nb blocks)" $est || continue
    for a in $arms; do blockx $lv $a || true; done
  done
}
idcheck() {   # exactness: the greedy set alone (c1) under O then NH (every new lever on + sched_cb_hold_ms 500), strict; a hold never applies at c1
          # alone (it needs >= 2 decoders), so also the greedy set at conc 4 under O then H500 (hold 500 ms, every int11 lever off), strict
  local l
  if budok "identity c1 (O vs NH)" 3; then
    alone ydm O a-O
    alone ydm NH a-NH
    cmpx ydm NH-vs-O $O/ydm-a-O.jsonl $O/ydm-a-NH.jsonl
    if [ "$DRY" != 1 ]; then
      l=$(grep "^ydm-NH-vs-O " $O/identity.txt 2>/dev/null | tail -1)
      case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity greedy set c1 NH vs O: VIRHE $l" ;; esac ;;
        *) xlog "identity greedy set c1 NH ($(armj NH)) vs O: VIRHE ${l:-not compared} (every new lever is exact: the text must match)" ;; esac
    fi
  else xlog "identity c1 SKIPPED by the budget: VIRHE the all-on vs all-off greedy set did not run"; fi
  [ "${YDM_C4ID:-1}" = 1 ] || return 0
  if budok "identity c8 (H500 vs O)" 2; then
    c8rr ydm O c8-O
    c8rr ydm H500 c8-H500
    cmpx ydm c8-H500-vs-O $O/ydm-c8-O.jsonl $O/ydm-c8-H500.jsonl
    [ "$DRY" = 1 ] && return 0
    l=$(grep "^ydm-c8-H500-vs-O " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity greedy set c8 H500 vs O: VIRHE $l" ;; esac ;;
      *) xlog "identity greedy set at conc 8, sched_cb_hold_ms 500 vs 0: VIRHE ${l:-not compared} (a hold only delays admission: the text must match)" ;; esac
  else xlog "identity c8 (H500 vs O) SKIPPED by the budget: VIRHE the hold's identity did not run"; fi
}
c8rr() {   # c8rr TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at conc 8, temperature 0 (the answers' lengths vary,
          # so newcomers arrive while >= 2 decode: a hold applies); the cbh deltas of the run (its holds)
  local tag=$1 t0=$(date +%s) n cb
  arm $2 || return 1; pcreset $tag-$3
  ssnap $tag-$3-0
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 8 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  ssnap $tag-$3-1
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $tag $3: greedy set at conc 8 under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); the cbh holds of the run"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-$3.jsonl 2>/dev/null || echo 0)
  cb=$(cbhd $tag-$3-0 $tag-$3-1)
  xlog "$tag $3 (greedy set at conc 8, temperature 0, $MT tok, arm $(NAMEOF $2) $(armj $2)): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/$tag-$3.txt | cut -c1-120), $(( $(date +%s) - t0 )) s | cbh holds $(echo "$cb" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("%s (by_k %s, by_timeout %s, by_idle %s), held_steps %s, in force %s ms" % (d.get("holds"), d.get("by_k"), d.get("by_timeout"), d.get("by_idle"), d.get("held_steps"), d.get("in_force_ms")))' 2>/dev/null)"
}
c4rr() {   # c4rr TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at conc 4 (d0ab's), temperature 0
  local tag=$1 t0=$(date +%s) n
  arm $2 || return 1; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 4 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $tag $3: greedy set at conc 4 under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line)"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-$3.jsonl 2>/dev/null || echo 0)
  xlog "$tag $3 (greedy set at conc 4, temperature 0, $MT tok, arm $(NAMEOF $2) $(armj $2)): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/$tag-$3.txt | cut -c1-160), $(( $(date +%s) - t0 )) s"
}
c16x() {   # c16x: ONE c16 block, EXACTLY the CAP48 screen's phase mb16 (cap48/dsm-cap48.sh: arm, prefix-cache reset, snapshots, its runner
          # cap48/ncmb.py --mode phase --temperature 0 --n 16 --dur 30 --warm 10) -> agg, decode_share, the cap counters (GET "cap")
  local w=c16-cap t00=$(date +%s) j
  arm O || return 1
  pcreset $w; ssnap $w-0
  run "(cd $D && timeout $(( C16DUR + 1200 )) python3 -I -B $C48N --mb $MBP --mode phase --temperature 0 --n 16 --tag ydm-$w --dur $C16DUR --warm $C16WARM --reqs $O/c8/$w-reqs.jsonl --stats-prefix $O/c8/$w-st --samples $O/c8/$w-samp.jsonl > $O/c8/$w.txt 2>&1)"
  ssnap $w-1
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm c16 (CAP48 method: cap48/ncmb.py phase c16, T 0, dur $C16DUR warm $C16WARM, MAXSEQS ${MAXSEQS:-?}, sched_row_cap $CAP): agg tok/s, decode_share, cap steps_gt48 / steps_capped / drafts_trimmed (GET cap), vs CAP48 $C48REF in %"; return 0; }
  j=$(grep -m1 '^MB ' $O/c8/$w.txt | cut -c4-)
  [ -n "$j" ] || { xlog "c16 $w: VIRHE request error: no MB line ($(tail -3 $O/c8/$w.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  xlog "c16 $w (CAP48 method: cap48/ncmb.py phase c16, T 0, dur $C16DUR warm $C16WARM, MAXSEQS ${MAXSEQS:-?}, the reference pin with sched_row_cap $CAP, $(( $(date +%s) - t00 )) s): $(echo "$j" | python3 -c 'import json,sys
d = json.loads(sys.stdin.read()); ref = float(sys.argv[1])
a = d.get("agg_dec_tps")
print("agg %s tok/s (vs CAP48 %.1f: %s), decode_share %s, %s tok / %s ms a decode step, acceptance %s, steps %s, %s errors%s" % (a, ref, ("%+.1f %%" % (100.0 * (a - ref) / ref)) if a else "-",
      d.get("decode_share"), d.get("tok_per_decode_step"), d.get("ms_decode_step"), d.get("acceptance"), d.get("steps"), d.get("errors"), "" if not d.get("errors") else " -- VIRHE request errors"))' $C48REF 2>&1 | cut -c1-500) | cap $(capd $w-0 $w-1) | $(thermline $t00 $(date +%s))"
}
p1reads() {   # the readings in their order (module header), each under the reference pin (read back)
  local k
  if budok "1u confirm (1 warm-up + $NREADS essay reads)" 2; then
    blockx U1W O essay || true
    for k in $(seq 1 $NREADS); do stopfile && return 0; blockx U1 O essay || true; done
  fi
  stopfile && return 0
  budok "c8 block (fixed harness, wall split)" 2 && { mbx 8 1 O S || true; }
  stopfile && return 0
  budok "c16 block (CAP48 method)" 2 && { c16x || true; }
  stopfile && return 0
  if budok "identity gs$GSC vs a1 (all $GSC in flight, cap $CAP)" 2; then idx16 || true; fi
  return 0
}
idx16() {   # the strict greedy identity at concurrency GSC (16, MAXSEQS 16: all in flight) vs a1 (c1 alone, this start), T 0, cap 48 (CAP48's
            # gs16-vs-a1); a difference = VIRHE; the > 48-row steps during gs16 (GET "cap" steps_gt48 + the step log) must be 0; a1 vs ref12's a1 beside
  local k l u0 u1
  alone ydm O a1
  u0=$(utcnow); ssnap ydm-gs$GSC-0; gsx; ssnap ydm-gs$GSC-1; u1=$(utcnow)
  echo "$u0 $u1" > $O/gs$GSC.win
  cmpx ydm gs$GSC-vs-a1 $O/ydm-a1.jsonl $O/ydm-gs$GSC.jsonl
  if [ -s "$R/xdsm/ref12/xref12-a1.jsonl" ] || [ "$DRY" = 1 ]; then cmpx ydm a1-vs-ref12-a1 $R/xdsm/ref12/xref12-a1.jsonl $O/ydm-a1.jsonl
  else xlog "identity a1 vs ref12 a1: $R/xdsm/ref12/xref12-a1.jsonl missing -- not compared"; fi
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm gs$GSC: > 48-row steps during it = 0 under the cap (GET cap steps_gt48 delta + the step log in [$O/gs$GSC.win]), else VIRHE; any identity difference = VIRHE"; return 0; }
  for k in gs$GSC-vs-a1 a1-vs-ref12-a1; do
    l=$(grep "^ydm-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity $k: VIRHE $l" ;; esac ;; *) xlog "identity $k: VIRHE ${l:-not compared} (strict, T 0 greedy)" ;; esac
  done
  local g; g=$(python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cap") or {}
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(int(b.get("steps_gt48", 0) - a.get("steps_gt48", 0)) if b else "na")' $O/fn-ydm-gs$GSC-0.json $O/fn-ydm-gs$GSC-1.json 2>/dev/null)
  xlog "gs$GSC cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd ydm-gs$GSC-0 ydm-gs$GSC-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; })"
}
mwin() { python3 -c 'import json,sys
m = sorted(json.loads(l)["t"] for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("ev") == "mark")
print("%.6f:%.6f" % (m[0], m[1]) if len(m) >= 2 else "")' "$1" 2>/dev/null; }   # mwin REQS.jsonl -> "T0:T1" (the block's two marks)
summary() {   # p12win.py confirm (the 1u reads) + rows (their step log) + split / rows (the c8 window) + rows (the c16 window: > 48 rows)
  local EL=$O/ydm-n5-engine-full.log.gz w8 w16 W="" l u0 u1
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $R12S confirm $O/blocks.jsonl --lever U1 --expect-tps 90 --expect-mb 27 --ref-tps 82.24 --ref-mb 136.6 -> RESULT ydm P12CONFIRM"
    echo "+ python3 -I -B $R12S rows $EL --win r<i>:utc0:utc1 (each 1u read) --st r<i>:st0:st1:utc0:utc1 -> P12ROWS (ms by verify rows, stream MB a decode step)"
    echo "+ python3 -I -B $R12S split $EL --win c8:<marks> --mbline $O/c8/c8-1-S-O.txt --sw0/--sw1 $O/c8/c8-1-S-O-st-w0/1.json --target 200 --ref 'ref12 c8:8.28:36.1:0.803:184.5' -> P12SPLIT; rows c8 -> row mix + depth-0 share"
    echo "+ python3 -I -B $R12S rows / split $EL --win c16:<marks of $O/c8/c16-cap-reqs.jsonl> --mbline $O/c8/c16-cap.txt -> > 48-row steps, row mix, wall split"
    return 0
  fi
  [ -s $O/blocks.jsonl ] && timeout 120 python3 -I -B $R12S confirm $O/blocks.jsonl --lever U1 --expect-tps 90 --expect-mb 27 --ref-tps 82.24 --ref-mb 136.6 > $O/confirm.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1800} | NOTE MAXSEQS ${MAXSEQS:-?} vs ref12's 8"; done < <(grep -h '^P12CONFIRM' $O/confirm.txt 2>/dev/null)
  python3 -c 'import json,sys
o = sys.argv[1]; W = []
for l in open(sys.argv[2]):
    d = json.loads(l)
    if d.get("lever") not in ("U1", "U1W"): continue
    n = "r%d-%s" % (d["i"], d["lever"]); b = "b%d-%s-%s" % (d["i"], d["lever"], d["arm"])
    W += ["--win", "%s:%.6f:%.6f" % (n, d["utc0"], d["utc1"]), "--st", "%s:%s/st-%s-0.json:%s/st-%s-1.json:%.6f:%.6f" % (n, o, b, o, b, d["utc0"], d["utc1"])]
print("\n".join(W))' $O $O/blocks.jsonl > $O/u1win.args 2>/dev/null
  [ -s $O/u1win.args ] && { mapfile -t W < $O/u1win.args; timeout 300 python3 -I -B $R12S rows $EL "${W[@]}" --json $O/u1rows.json > $O/u1rows.txt 2>&1; }
  while IFS= read -r l; do xlog "1u ${l:0:1500}"; done < <(grep -h '^P12ROWS' $O/u1rows.txt 2>/dev/null)
  w8=$(mwin $O/c8/c8-1-S-O-reqs.jsonl)
  if [ -n "$w8" ]; then
    timeout 300 python3 -I -B $R12S split $EL --win c8:$w8 --mbline $O/c8/c8-1-S-O.txt --sw0 $O/c8/c8-1-S-O-st-w0.json --sw1 $O/c8/c8-1-S-O-st-w1.json --target 200 --ref "ref12 c8:8.28:36.1:0.803:184.5" > $O/c8split.txt 2>&1
    timeout 300 python3 -I -B $R12S rows $EL --win c8:$w8 --json $O/c8rows.json > $O/c8rows.txt 2>&1
    while IFS= read -r l; do xlog "c8 ${l:0:2400} | NOTE MAXSEQS ${MAXSEQS:-?} vs ref12's 8"; done < <(grep -hE '^P12(SPLIT|ROWS)' $O/c8split.txt $O/c8rows.txt 2>/dev/null)
  else xlog "summary: c8 -- no window (no marks in $O/c8/c8-1-S-O-reqs.jsonl)"; fi
  w16=$(mwin $O/c8/c16-cap-reqs.jsonl)
  if [ -n "$w16" ]; then
    timeout 300 python3 -I -B $R12S rows $EL --win c16:$w16 --json $O/c16rows.json > $O/c16rows.txt 2>&1
    timeout 300 python3 -I -B $R12S split $EL --win c16:$w16 --mbline $O/c8/c16-cap.txt --sw0 $O/c8/c16-cap-st-w0.json --sw1 $O/c8/c16-cap-st-w1.json --target 200 > $O/c16split.txt 2>&1
    while IFS= read -r l; do xlog "c16 ${l:0:2400}"; done < <(grep -hE '^P12(SPLIT|ROWS)' $O/c16rows.txt $O/c16split.txt 2>/dev/null)
  else xlog "summary: c16 -- no window (no marks in $O/c8/c16-cap-reqs.jsonl)"; fi
  if [ -s $O/gs$GSC.win ]; then
    read -r u0 u1 < $O/gs$GSC.win
    timeout 300 python3 -I -B $R12S rows $EL --win gs$GSC:$u0:$u1 --json $O/gsrows.json > $O/gsrows.txt 2>&1
    while IFS= read -r l; do xlog "gs$GSC step log (the > 48-row count must be 0 under the cap) ${l:0:1500}"; done < <(grep -h '^P12ROWS' $O/gsrows.txt 2>/dev/null)
    python3 -c 'import json,sys; d = json.load(open(sys.argv[1]))["windows"]; w = list(d.values())[0]; sys.exit(0 if w.get("gt48", 0) == 0 else 1)' $O/gsrows.json 2>/dev/null \
      || { [ $SW = 1 ] && xlog "gs$GSC step log: VIRHE > 48-row decode steps under the cap (or no step-log reading)" || xlog "gs$GSC step log: INFO > 48-row steps (SW=0 mock: no row cap emulated)"; }
  fi
  grep -lq '^Traceback' $O/confirm.txt $O/u1rows.txt $O/c8split.txt $O/c8rows.txt $O/c16rows.txt $O/c16split.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/confirm.txt $O/u1rows.txt $O/c8split.txt $O/c8rows.txt $O/c16rows.txt $O/c16split.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  return 0
}
# ---------------------------------------------------------------- lever A4's own functions (P0 on one card, the table line, identity, readings, summary)
P0UUID=${YDM_P0_UUID:-GPU-dbba5c851cfef7b3}   # P0's card BY CHIP UUID (ROCR_VISIBLE_DEVICES; gpu02 sysfs unique_id dbba5c851cfef7b3 = card4, 03:00.0)
DEV1="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e ROCR_VISIBLE_DEVICES=$P0UUID"
M16TAB=$STATSDIR/m16-forms.txt   # the P0 bench's table (the server reads it at /stats/m16-forms.txt: RADIANCE_DS41_GG_M16_FORMS)
p0a4() {   # p0a4: ONE card (P0UUID), no server -- (1) the gate bench r4d_selftest --perfm16 (D2's pick vs the A4 candidates, M PERFMS, two
           # passes) -> p0-perf.txt + the table (its M16FORM lines) -> M16TAB; (2) the byte check r4d_selftest --mrcheck16 with that table
           # loaded (RADIANCE_DS41_GG_M16_FORMS, a read-only mount): every A4 form + the switch vs the stock band, plain / pair / folds a e,
           # M 1..16 -> A4VERDICT PASS. -> 0 go; 1 STOP (a VIRHE: the check failed or did not run); 2 STOP (the GATE: no A4 gain worth a
           # screen -- the bench's c8 estimate under GATEUS us a step, or an empty table; not a VIRHE)
  local f=$O/p0-perf.txt c=$O/p0-check.txt rc n est sv
  run "gpurun ydm-p0-perf-$$ $P0PERFS $DEV1 --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --perfm16 $PERFMS $PERFIT > $f 2>&1"; rc=$?
  if [ "$DRY" = 1 ]; then
    echo "+ grep '^M16FORM ' $f | cut -c9- > $M16TAB (the table: one line a (type, N, K, M, pair) whose A4 form beat D2's by >= 3 % in both passes, bytes identical)"
    echo "+ RESULT ydm P0 bench (one card $P0UUID, bounded $P0PERFS s): rc 0, M16VERDICT bytes identical everywhere, M16EST c8 saving >= $GATEUS us a step bench (else the GATE stops the item: no screen), the M16GATE lines logged"
  else
    n=$(grep -c '^M16FORM ' $f); est=$(grep -m1 '^M16EST' $f); sv=$(echo "$est" | sed -n 's/.*(\([0-9.-]*\) us saved); c16.*/\1/p')
    grep '^M16FORM ' $f | cut -c9- > $M16TAB
    while IFS= read -r l; do xlog "P0 bench ${l:0:400}"; done < <(grep -E '^M16GATE|^M16EST|^M16VERDICT' $f | head -60)
    if [ $rc != 0 ] || ! grep -q '^M16VERDICT: .*bytes identical everywhere' $f; then
      if [ $SW = 0 ]; then xlog "P0 bench: failed under YDM_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"
      else xlog "P0 bench (r4d_selftest --perfm16): VIRHE rc $rc, $(grep -m1 '^M16VERDICT' $f || echo 'no M16VERDICT line') | $(grep -E 'OUTPUT DIFFERS|FAIL|error|fault|stopped after' $f | head -4 | cut -c1-300 | tr '\n' ';')"; return 1; fi
    else
      xlog "P0 bench: $n table entries -> $M16TAB (md5 $(md5sum < $M16TAB | cut -c1-8)) | c8 bench saving ${sv:-?} us a step (gate $GATEUS)"
    fi
  fi
  run "gpurun ydm-p0-check-$$ $P0CHECKS $DEV1 -v $M16TAB:/m16-forms.txt:ro -e RADIANCE_DS41_GG_M16_FORMS=/m16-forms.txt --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --mrcheck16 > $c 2>&1"; rc=$?
  if [ "$DRY" = 1 ]; then echo "+ RESULT ydm P0 byte check (one card, bounded $P0CHECKS s, the table mounted read only): rc 0, 0 FAIL, 'lever A4 (gg_gemv_m16): N forms from /m16-forms.txt' with N = the table's lines, every MRCHECK line 'identical in every form', A4VERDICT ... PASS (byte-identical) -- else VIRHE and the item STOPS"; return 0; fi
  local nchk nfail ntab v
  nchk=$(grep -c 'identical in every form' $c); nfail=$(grep -c '^  FAIL\|DIFFERS' $c); v=$(grep -m1 '^A4VERDICT' $c)
  ntab=$(grep -m1 'lever A4 (gg_gemv_m16):' $c | sed -n 's/.*: \([0-9]*\) forms from.*/\1/p')
  xlog "P0 byte check (r4d_selftest --mrcheck16, one card, M 1..16): rc $rc, $nchk MRCHECK lines identical in every form, $nfail FAIL / DIFFERS, table loaded ${ntab:-?} of $(grep -c . $M16TAB 2>/dev/null) | ${v:-no A4VERDICT line} | $(grep '^  ok' $c | grep -c 'A4 gemm_nt_ggml_pair') pair lines ok, fold lines $(grep -E '^  (ok|FAIL) +(rmsnorm|swiglu)' $c | cut -c1-6 | sort | uniq -c | tr '\n' ' ')"
  if [ $SW = 0 ]; then
    [ "${YDM_MOCK_GATE:-0}" = 1 ] && { GATEWHY="YDM_MOCK_GATE=1 under YDM_SW=0: the gate's path in a mock"; xlog "P0: $GATEWHY"; return 2; }
    xlog "P0: YDM_SW=0 (a mock) -- the mock goes on"; return 0
  fi
  case "$v" in *"PASS (byte-identical)"*) ;; *) xlog "P0 byte check: VIRHE A4VERDICT not PASS | $(grep -E '^  FAIL|DIFFERS|error|fault|stopped after' $c | head -5 | cut -c1-300 | tr '\n' ';')"; return 1 ;; esac
  [ "$rc" = 0 ] && [ "$nfail" = 0 ] && [ "${ntab:-x}" = "$(grep -c . $M16TAB)" ] || { xlog "P0 byte check: VIRHE rc $rc / FAIL $nfail / table ${ntab:-?} vs $(grep -c . $M16TAB)"; return 1; }
  if [ "$FORCE" != 1 ]; then
    [ "$(grep -c . $M16TAB)" -gt 0 ] || { GATEWHY="the bench found no A4 form >= 3 % faster than D2's at any key: empty table"; xlog "GATE: $GATEWHY -> no A4 screen (YDM_FORCE=1 runs it anyway)"; return 2; }
    awk -v s="${sv:-0}" -v g=$GATEUS 'BEGIN{exit !(s + 0 >= g + 0)}' || { GATEWHY="the bench's c8 saving ${sv:-?} us a step < $GATEUS, $(grep -c . $M16TAB) table entries"; xlog "GATE: $GATEWHY -> no A4 screen (YDM_FORCE=1 runs it anyway)"; return 2; }
  fi
  return 0
}
tabline() {   # tabline TAG: the start log's "lever A4 (gg_gemv_m16): N forms from /stats/m16-forms.txt" with N = the table's lines (else C = B)
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm $1: the engine log names 'lever A4 (gg_gemv_m16): N forms from /stats/m16-forms.txt' with N = the P0 table's lines (else VIRHE: arm C would run D2's forms)"; return 0; }
  local l n w
  l=$(docker logs $CN 2>&1 | grep -m1 'lever A4 (gg_gemv_m16):' | sed 's/^radiance: //' | cut -c1-300); n=$(echo "$l" | sed -n 's/.*: \([0-9]*\) forms from.*/\1/p'); w=$(grep -c . $M16TAB 2>/dev/null)
  if [ -n "$n" ] && [ "$n" = "$w" ]; then xlog "$1 A4 table: $l (the P0 table's $w lines)"
  elif [ $SW = 0 ]; then xlog "$1 A4 table: INFO (YDM_SW=0 mock) '${l:-no line}'"
  else xlog "$1 A4 table: VIRHE '${l:-no line}' vs the P0 table's ${w:-?} lines"; return 1; fi
}
idok() {   # idok KEY ...: each identity.txt line ydm-KEY must be IDENTICAL (strict, T 0 greedy), else VIRHE
  local k l
  [ "$DRY" = 1 ] && return 0
  for k in "$@"; do
    l=$(grep "^ydm-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity $k: VIRHE $l" ;; esac ;; *) xlog "identity $k: VIRHE ${l:-not compared} (strict, T 0 greedy)" ;; esac
  done
}
gt48() {   # gt48 NAME: the GET cap steps_gt48 delta between snapshots NAME-0 / NAME-1 -> a number or na
  [ "$DRY" = 1 ] && { echo 0; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cap") or {}
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(int(b.get("steps_gt48", 0) - a.get("steps_gt48", 0)) if b else "na")' $O/fn-$1-0.json $O/fn-$1-1.json 2>/dev/null || echo na
}
p0pf() {   # p0pf: lever A10's P0 (its staged ypfx/bin/pfxprobe, read in place) in THIS image on the P0 card -- OLD (the int12 plugin,
           # ypfx/bin/exl3r4d-int12.so 9823cdde) vs NEW (the image's A10 exl3r4d.so) x3d_pf 0 / 1, K 2 + K 2.5, M 1 / 17 / 64 / 512 / 2048, h'
           # and y byte for byte, the hooks, TIME (INFO) -> PFOK 1 ("PFX DONE ok", exit 0) or PFOK 0 + a VIRHE line (every P reading dropped;
           # A4's arms go on)
  local f=$O/p0-pfxprobe.txt rc m l
  [ $A10 = 1 ] || { PFOK=0; xlog "A10 (exl3_x3d_pf) not in this run (YDM_A10=0): no P0 / P readings"; return 0; }
  for m in pfxprobe exl3r4d-int12.so; do
    [ "$(md5sum < $PB/$m 2>/dev/null | cut -c1-32)" = "$(awk -v f=$m '$2 == f {print $1}' $PB/MD5SUMS 2>/dev/null)" ] \
      || { xlog "A10 P0: VIRHE $PB/$m missing or not the md5 in $PB/MD5SUMS -> the A10 arm (P) DROPPED"; [ "$DRY" = 1 ] || { PFOK=0; return 0; }; }
  done
  [ "$(md5sum < $PB/exl3r4d-int12.so 2>/dev/null | cut -c1-32)" = $PFOLDMD5 ] || { xlog "A10 P0: VIRHE $PB/exl3r4d-int12.so is not the int12 plugin ($PFOLDMD5) -> P DROPPED"; [ "$DRY" = 1 ] || { PFOK=0; return 0; }; }
  [ "$DRY" = 1 ] || { mkdir -p $O/p0 && chmod 777 $O/p0; }
  run "gpurun ydm-p0pf-$$ $PFTMO --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$P0UUID -e HIP_VISIBLE_DEVICES=0 -v $PB:/b:ro -v $O/p0:/o --entrypoint /b/pfxprobe $IMAGE /o --new /opt/radiance/share/radiance/kernels/exl3r4d.so --old /b/exl3r4d-int12.so > $f 2>&1"; rc=$?
  if [ "$DRY" = 1 ]; then echo "+ RESULT ydm A10 P0 on card $P0UUID (A10's pfxprobe $(md5sum < $PB/pfxprobe 2>/dev/null | cut -c1-8), bounded $PFTMO s): 'PFX DONE ok' + exit 0 (every BYTES row 0 differ at x3d_pf 0 and 1, K 2 and 2.5, S8 / E96, M 1 / 17 / 64 / 512 / 2048; HOOKS ok), else VIRHE and the A10 arm P DROPPED (A4 goes on); TIME lines INFO"; return 0; fi
  while IFS= read -r l; do xlog "A10 P0 ${l:0:500}"; done < <(grep -E '^PFX (DEV|HOOKS|TIME|DONE|VIRHE)' $f)
  if grep -q '^PFX DONE ok' $f && [ $rc = 0 ]; then
    xlog "A10 P0 on one card ($P0UUID): OK -- OLD vs NEW x3d_pf 0 / 1 byte-identical on every row ($(grep -c '^PFX BYTES .* 0 rows differ, new x3d1 vs old 0 rows differ' $f) of $(grep -c '^PFX BYTES .* M ' $f) shapes)"
    rm -rf $O/p0; gzip -f $f
    return 0
  fi
  [ $SW = 0 ] && { xlog "A10 P0 failed under YDM_SW=0 (a mock run: docker run --device is not emulated) -- the mock keeps P to exercise its readings (rc $rc)"; return 0; }
  PFOK=0
  xlog "A10 P0 on one card ($P0UUID): VIRHE rc $rc (124/137 = timeout), no 'PFX DONE ok' -> the A10 arm (P) DROPPED, A4 goes on | $(grep -E 'VIRHE|differ' $f | grep -v ' 0 rows differ, new x3d1 vs old 0 rows differ' | head -4 | tr '\n' ';' | cut -c1-600)"
  return 0
}
ydid() {   # the strict greedy identity: c1 alone (a1) and conc GSC (all in flight, cap 48) under B, then under C; T 0. Compared: a1-C vs a1-B,
           # gs16-C vs gs16-B, gs16-B vs a1-B, gs16-C vs a1-C (batch invariance under C), a1-B and a1-C vs ref12's a1
  local a g as="B"
  [ $A4ON = 1 ] && as="$as C"
  [ $PFOK = 1 ] && as="$as P"
  for a in $as; do
    alone ydm $a a1-$a
    u0=$(utcnow); ssnap ydm-gs$GSC-$a-0; gsx $a; ssnap ydm-gs$GSC-$a-1; u1=$(utcnow)
    [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-$a.win
    g=$(gt48 ydm-gs$GSC-$a)
    xlog "gs$GSC-$a cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd ydm-gs$GSC-$a-0 ydm-gs$GSC-$a-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; })"
  done
  cmpx ydm gs$GSC-B-vs-a1-B $O/ydm-a1-B.jsonl $O/ydm-gs$GSC-B.jsonl
  if [ $A4ON = 1 ]; then
    cmpx ydm a1-C-vs-a1-B $O/ydm-a1-B.jsonl $O/ydm-a1-C.jsonl
    cmpx ydm gs$GSC-C-vs-gs$GSC-B $O/ydm-gs$GSC-B.jsonl $O/ydm-gs$GSC-C.jsonl
    cmpx ydm gs$GSC-C-vs-a1-C $O/ydm-a1-C.jsonl $O/ydm-gs$GSC-C.jsonl
  fi
  if [ -s "$R/xdsm/ref12/xref12-a1.jsonl" ] || [ "$DRY" = 1 ]; then
    cmpx ydm a1-B-vs-ref12-a1 $R/xdsm/ref12/xref12-a1.jsonl $O/ydm-a1-B.jsonl
    [ $A4ON = 1 ] && cmpx ydm a1-C-vs-ref12-a1 $R/xdsm/ref12/xref12-a1.jsonl $O/ydm-a1-C.jsonl
  else xlog "identity a1 vs ref12 a1: $R/xdsm/ref12/xref12-a1.jsonl missing -- not compared"; fi
  if [ $PFOK = 1 ]; then
    cmpx ydm a1-P-vs-a1-B $O/ydm-a1-B.jsonl $O/ydm-a1-P.jsonl
    cmpx ydm gs$GSC-P-vs-a1-P $O/ydm-a1-P.jsonl $O/ydm-gs$GSC-P.jsonl
    idok a1-P-vs-a1-B gs$GSC-P-vs-a1-P
  fi
  idok gs$GSC-B-vs-a1-B
  [ $A4ON = 1 ] && idok a1-C-vs-a1-B gs$GSC-C-vs-gs$GSC-B gs$GSC-C-vs-a1-C
  [ -s "$R/xdsm/ref12/xref12-a1.jsonl" ] && idok a1-B-vs-ref12-a1
  [ -s "$R/xdsm/ref12/xref12-a1.jsonl" ] && [ $A4ON = 1 ] && idok a1-C-vs-ref12-a1
  return 0
}
stagx() {   # stagx ARM: STAGGERED arrivals under ARM (zepb/dsm-zepb.sh's epbstag, stack4/loadid.py run, read only): LDCONC decode streams + the
            # 20 greedy probes one at a time, each with a FRESH LDPF-token prefill joining 0.5 s in (its chunks ride the probe's decode: MIXED
            # steps) -> ld-ARM vs a1-ARM (this start's c1 alone), STRICT: a difference is a VIRHE line with the first differing char / token,
            # and the item goes on (does int12's path keep batch invariance in mixed steps? EPB's staggered check failed 1 / 20)
  local a=$1 t0=$(date +%s) l
  arm $a || return 1
  pcreset ydm-ld-$a; ssnap ydm-ld-$a-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/ydm-ld-$a.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/ydm-ld-$a.txt 2>&1)"
  ssnap ydm-ld-$a-1
  cmpx ydm ld-$a-vs-a1-$a $O/ydm-a1-$a.jsonl $O/ydm-ld-$a.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT ydm staggered $a: LOADID (probes, MIXED steps during the probes, prefills cold) + ld-$a vs a1-$a strict (a difference = VIRHE with the first differing char / token; the item goes on)"; return 0; }
  xlog "staggered $a ($(NAMEOF $a); loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s): $(grep -m1 '^LOADID ' $O/ydm-ld-$a.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/ydm-ld-$a.jsonl 2>&1 | cut -c1-200) | cap $(capd ydm-ld-$a-0 ydm-ld-$a-1)"
  l=$(grep "^ydm-ld-$a-vs-a1-$a " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "staggered $a: VIRHE $l" ;; esac ;;
    *) if [ $SW = 1 ]; then xlog "staggered $a: VIRHE ${l:-not compared} (strict: a decode row riding MIXED steps must keep its alone bytes; reported, the item goes on)"; else xlog "staggered $a: INFO (SW=0 mock) ${l:-not compared}"; fi ;; esac
  return 0
}
u1x() {   # u1x ARM: the cheap 1u essay confirm under ARM -- ONE discarded warm-up read (U1W) + NREADS reads back to back (U1), essay only
  local k
  blockx U1W $1 essay || true
  for k in $(seq 1 $NREADS); do stopfile && return 0; blockx U1 $1 essay || true; done
}
ydreads() {   # the readings in their order (module header): identity, c8 (C8ORD), 1u (U1ORD), c16 (C16ORD)
  local i a
  if budok "identity (a1 + gs$GSC under B, then C)" 4; then ydid || true; else xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if budok "staggered arrivals B (loadid.py run, conc $LDCONC + $LDPF-token prefills)" 5; then stagx B || true; else xlog "staggered B SKIPPED by the budget: VIRHE"; fi
  local k r
  if [ $PFOK = 1 ]; then for k in $PDEPTHS; do
    stopfile && return 0
    budok "A10 cold $k B + P" $(( 2 * (k / 100000 + 1) )) || continue
    coldx $k B 1 || continue
    coldx $k P 2 || continue
    [ "$DRY" = 1 ] && continue
    r=$(cmpfile $O/cold-$k-1-B.txt $O/cold-$k-2-P.txt); idrec "ydm-cold-$k-P-vs-B" "$r" "the $NTOK-token answers, byte for byte"
    case "$r" in IDENTICAL*) ;; *) xlog "identity cold-$k P vs B: VIRHE the answers differ (strict, T 0 greedy, the same prompt)" ;; esac
  done; fi
  i=0; for a in $(echo $C8ORD | grep -o .); do
    i=$((i + 1)); stopfile && return 0
    [ $a = P ] && [ $PFOK != 1 ] && { xlog "c8 block $i P SKIPPED (A10's P0 failed or YDM_A10=0)"; continue; }
    budok "c8 block $i $a" 2 && { mbx 8 $i $a S || true; }
  done
  for a in $(echo $U1ORD | grep -o .); do stopfile && return 0; budok "1u essay $a (1 warm-up + $NREADS reads)" 1 && u1x $a; done
  i=0; for a in $(echo $C16ORD | grep -o .); do i=$((i + 1)); stopfile && return 0; budok "c16 block $i $a" 2 && { mbx 16 $i $a S || true; }; done
  if [ $A4ON = 1 ] && [ $STAGC = 1 ]; then stopfile && return 0; budok "staggered arrivals C (budget allowing)" 5 && { stagx C || true; }; fi
  return 0
}
ydsummary() {   # per c8 / c16 block p12win.py split + rows (marks), the 1u reads' step-log rows, the gs16 step logs (> 48 rows), then ydsum.py
  local EL=$O/ydm-n5-engine-full.log.gz nm w a u0 u1 l W
  if [ "$DRY" = 1 ]; then
    echo "+ per mb block <c8|c16>-<i>-S-<arm>: python3 -I -B $R12S split $EL --win <name>:<marks> --mbline $O/c8/<name>.txt --sw0/--sw1 $O/c8/<name>-st-w0/1.json --target 200 > $O/split-<name>.txt; rows --json $O/rows-<name>.json"
    echo "+ python3 -I -B $R12S rows $EL --win r<i>-U1:utc0:utc1 (each 1u read) --st ... -> P12ROWS (ms by verify rows, stream MB a decode step)"
    echo "+ python3 -I -B $R12S rows $EL --win gs$GSC-<arm>:<u0>:<u1> (each gs$GSC): > 48-row decode steps = 0 under the cap, else VIRHE"
    echo "+ python3 -I -B $YDS $O --dec-th $DECTH --ref-nc 538.2 --ref-share 20.0 --ref-dec 36.18 --ref-agg 184.0 --a10-nc-ms $PNCTH --a10-ttft-pct $PTTH -> RESULT ydm YDSUM lines (C vs B and P vs B: values, B reference, % difference; the P0 gate lines; 'A10 joins the stack if 136k TTFT >= $PTTH % faster OR newcomer step >= $PNCTH ms faster': PASS / NOT)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip the P0 outputs"
    return 0
  fi
  for f in $(ls $O/c8/c8-*-reqs.jsonl $O/c8/c16-*-reqs.jsonl 2>/dev/null); do
    nm=$(basename $f -reqs.jsonl); w=$(mwin $f)
    [ -n "$w" ] || { xlog "summary: $nm -- no window (no marks in $f)"; continue; }
    timeout 300 python3 -I -B $R12S split $EL --win $nm:$w --mbline $O/c8/$nm.txt --sw0 $O/c8/$nm-st-w0.json --sw1 $O/c8/$nm-st-w1.json --target 200 > $O/split-$nm.txt 2>&1
    timeout 300 python3 -I -B $R12S rows $EL --win $nm:$w --json $O/rows-$nm.json > $O/rows-$nm.txt 2>&1
    while IFS= read -r l; do xlog "$nm ${l:0:1600}"; done < <(grep -hE '^P12SPLIT' $O/split-$nm.txt 2>/dev/null)
  done
  python3 -c 'import json,sys
o = sys.argv[1]; W = []
for l in open(sys.argv[2]):
    d = json.loads(l)
    if d.get("lever") not in ("U1", "U1W"): continue
    n = "r%d-%s-%s" % (d["i"], d["lever"], d["arm"]); b = "b%d-%s-%s" % (d["i"], d["lever"], d["arm"])
    W += ["--win", "%s:%.6f:%.6f" % (n, d["utc0"], d["utc1"]), "--st", "%s:%s/st-%s-0.json:%s/st-%s-1.json:%.6f:%.6f" % (n, o, b, o, b, d["utc0"], d["utc1"])]
print("\n".join(W))' $O $O/blocks.jsonl > $O/u1win.args 2>/dev/null
  [ -s $O/u1win.args ] && { mapfile -t W < $O/u1win.args; timeout 300 python3 -I -B $R12S rows $EL "${W[@]}" --json $O/u1rows.json > $O/u1rows.txt 2>&1; }
  while IFS= read -r l; do xlog "1u ${l:0:1500}"; done < <(grep -h '^P12ROWS' $O/u1rows.txt 2>/dev/null)
  for a in B C; do
    [ -s $O/gs$GSC-$a.win ] || continue
    read -r u0 u1 < $O/gs$GSC-$a.win
    timeout 300 python3 -I -B $R12S rows $EL --win gs$GSC-$a:$u0:$u1 --json $O/gsrows-$a.json > $O/gsrows-$a.txt 2>&1
    python3 -c 'import json,sys; d = json.load(open(sys.argv[1]))["windows"]; w = list(d.values())[0]; sys.exit(0 if w.get("gt48", 0) == 0 else 1)' $O/gsrows-$a.json 2>/dev/null \
      || { [ $SW = 1 ] && xlog "gs$GSC-$a step log: VIRHE > 48-row decode steps under the cap (or no step-log reading)" || xlog "gs$GSC-$a step log: INFO > 48-row steps (SW=0 mock: no row cap emulated)"; }
  done
  timeout 120 python3 -I -B $YDS $O --dec-th $DECTH --sw $SW --ref-nc 538.2 --ref-share 20.0 --ref-dec 36.18 --ref-agg 184.0 --a10-nc-ms $PNCTH --a10-ttft-pct $PTTH > $O/ydsum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2000}"; done < <(grep -h '^YDSUM' $O/ydsum.txt 2>/dev/null)
  grep -q '^YDSUM' $O/ydsum.txt 2>/dev/null || xlog "summary: VIRHE ydsum.py printed no YDSUM line ($(tail -3 $O/ydsum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -lq '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/u1rows.txt $O/ydsum.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/u1rows.txt $O/ydsum.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/p0-check.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
dstart() {   # P0 (the gate bench + the byte check, one card), then the screen start
  local tag=ydm-n5 p
  diskok $tag || return 1
  p0a4; p=$?
  [ $p = 1 ] && { xlog "$tag: VIRHE P0 (A4's bench / byte check on one card) failed -> NOT STARTED"; return 1; }
  p0pf
  if [ $p = 2 ]; then
    if [ $PFOK = 1 ] && [ "${YDM_A10_ALONE:-1}" = 1 ]; then
      A4ON=0; C8ORD=$(echo $C8ORD | tr -d C); U1ORD=""; C16ORD=""
      xlog "A4 gate NOT passed ($GATEWHY); A10 part runs alone (arms B / P: identity a1 / gs$GSC under B and P, the staggered check under B, cold [$PDEPTHS] B then P, c8 [$C8ORD], the A10 rule line; YDM_A10_ALONE=0 skips it)"
    else xlog "$tag: A4 gate NOT passed ($GATEWHY)$([ $PFOK = 1 ] || echo "; A10's P0 failed too") -> NOT STARTED (the P0 bench is the item's reading)"; return 1; fi
  fi
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt"
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${YDM_MAXSEQS:-}" ] && export MAXSEQS=$YDM_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (SCREEN; $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-900) | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  tabline $tag || { keeplog $tag-table; down; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || : > $O/blocks.jsonl
  xlog "THE BASE PIN B (ref12's + gg_gemv_m16 0 + exl3_x3d_pf 0, read back): $(pinj B | cut -c1-1200) | C = that + $(armj C) | P (A10, $([ $PFOK = 1 ] && echo on || echo DROPPED)) = that + $(armj P) | readings: identity a1 / gs$GSC under B, C$([ $PFOK = 1 ] && echo ", P"), cold [$([ $PFOK = 1 ] && echo $PDEPTHS)] B then P, c8 [$C8ORD] (fixed harness, wall split), 1u essay [$U1ORD] (1 warm-up + $NREADS), c16 [$C16ORD]"
  stopfile || ydreads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(i11now) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN YDM_SW=0 (NOT A READING): NO int9b .. int12 switch posted, the counters INFO (a mock run only)") | A4 SCREEN, ONE START, ref12's pins and start on the copy path (no ngshare, no ATTACH): $DESC | MAXSEQS ${YDM_MAXSEQS:-the base value} (ref12: 8) | P0 one card ($P0UUID): r4d_selftest --perfm16 (the gate bench: D2 vs the A4 candidates, M $PERFMS, two passes; the table) then --mrcheck16 with that table (every A4 form + the switch vs the stock band, plain / pair / folds, M 1..16; a failure stops the item; no gain >= $GATEUS us a c8 step = the GATE, no screen) | readings: identity a1 + gs$GSC under B then C (strict) | mb c8 [$C8ORD] (window ${C8WIN} s, wall split) | 1u essay [$U1ORD] (1 warm-up + $NREADS) | c16 [$C16ORD] (the same harness at n 16) | no-regression guard +$DECTH % | budget $BUDGET min | no evict"
IMGLITS="B|sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx,
B|stage_full_min
B|stage_full_pick
B|stage_full_min takes 0 (off) or 1 .. 1048576
B|full_pick_dropped
B|ds41_psk_cr
B|ds41_engdd
B|sched_noa
B|stage_short
B|stage_short takes 0 (off) or 1 .. 65536
B|cr_end_snapshots
B|tail_only_latched
B|lever SA / HB (ds41-sahb) set up, off: stage_short %s; place_hb %s
B|RADIANCE_PLACE_HB_RANKS
B|RADIANCE_LOG_STEPS
B|RADIANCE_PROFILE_EVERY
B|per-op timing, rank
B|n_seq=%lld n_tok=%lld
B|reset_prefix_cache
B|held once a slot
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
A|RADIANCE_DS41_KV_REPLICATED
A|lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now
K|gather_rows_w16_kernel
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|lever Z0-A set up"
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + YDM_DRY_ALLOW_OLD=1, a dry run against an image without them)
IMGLITS9B="B|stage_full_pick, mhc_fuse, ix_lds, ix_ni, ds41_arcoded, sched_row_cap, 
B|mhc_fuse
B|ix_lds
B|ix_ni
B|lds_launches
B|ni_cand_launches
B|dead_bytes
A|RADIANCE_DS41_MHC_FUSE
A|lever M1 (mhc_fuse) declared
A|mhc_post_collapse_rmsnorm_mix
K|mhc_post_collapse_rmsnorm_mix
K|ix_lds
K|ix_ni
R|mhc_pcn_mix
R|ix_lds
R|ix_ni
P|int9b-PROVENANCE.txt"
# int10's own (VIRHE when missing; SKIP only under DRYRUN=1 + YDM_DRY_ALLOW_OLD=1): lever ARC + draft_depth0 3
IMGLITS10="B|ds41_arcoded
B|coded_all_ranks
B|sched_row_cap
B|RADIANCE_DECODE_ROW_CAP
B|steps_gt48
B|3 (occupancy only)
B|draft depth 0 by OCCUPANCY ONLY
K|ds41_arcoded
K|RADIANCE_DS41_ARCODED
K|rad_arc_stats
K|ar_twoshot_4rank_exact (coded, ds41_arcoded)
R|ar_arcoded
R|rad_arc_stats
P|int10-PROVENANCE.txt"
# int11's own (VIRHE when missing; SKIP only under DRYRUN=1 + YDM_DRY_ALLOW_OLD=1): levers 1U-MX / 1U-GQ / 1U-E2, GET "i11"
IMGLITS11="B|ds41_arcoded, sched_row_cap, ds41_mix_wv, ds41_gq, ds41_e2, sched_cb_hold_ms, sched_cb_hold_k, 
B|sched_cb_hold_ms
B|sched_cb_hold_k
B|cob_newcomers
B|by_timeout
B|RADIANCE_SCHED_CB_HOLD_MS
K|gemm_nt_ggml_pair
R|gq_pair
B|ds41_mix_wv
B|ds41_gq
B|ds41_e2
B|wide_launches
B|qkv_pairs
B|out_pairs
B|rad_i11_stats
B|rad_mx_stats
A|RADIANCE_DS41_GQ
A|RADIANCE_DS41_E2
A|lever 1U-GQ (ds41_gq, RADIANCE_DS41_GQ, now %d)
A|lever 1U-E2 (ds41_e2, RADIANCE_DS41_E2, now %d)
A|engram_kv_part
A|rad_i11_stats
K|RADIANCE_DS41_MIX_WV
K|ds41_mix_wv
K|rad_mx_stats
R|mix_wv
R|rad_mx_stats
P|int11-PROVENANCE.txt"
# int12's own (VIRHE when missing; SKIP only under DRYRUN=1 + YDM_DRY_ALLOW_OLD=1): the X3D-F merge (fn4 exl3_x3d, the exl3r4d plugin);
# + lever F4G's literal with YDM_F4G=1 (the rows-gated ds41_shexp_lane)
IMGLITS12="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d,
B|exl3_x3d
X|exl3_x3d
P|int12-PROVENANCE.txt$([ $F4G = 1 ] && printf '\nA|lever F4G')"
# lever A4's own (VIRHE when missing; SKIP only under DRYRUN=1 + YDM_DRY_ALLOW_OLD=1): fn4 gg_gemv_m16 (63, libr4d's), the table loader,
# r4d_selftest's --mrcheck16 / --perfm16, a4-PROVENANCE
IMGLITS13="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, exl3_x3d_pf, gg_gemv_m16, epoch_bump)
B|exl3_x3d_pf
X|exl3_x3d_pf
X|exl3r4d_pf_gemm_x3d_kernel
X|exl3r4d_fn4_x3d_pf
B|gg_gemv_m16
K|gg_gemv_m16
K|RADIANCE_DS41_GG_M16_FORMS
K|lever A4 (gg_gemv_m16): %d forms from %s
K|r4d_gg_mr_pin_xp
R|--mrcheck16
R|--perfm16
R|A4VERDICT
R|M16FORM
P|a4-PROVENANCE.txt"
PLUGMD5=${YDM_PLUGMD5:-bb2d91f6}
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$( { echo "$IMGLITS"; echo "$IMGLITS9B" | sed 's/^/9/'; echo "$IMGLITS10" | sed 's/^/9/'; echo "$IMGLITS11" | sed 's/^/9/'; echo "$IMGLITS12" | sed 's/^/9/'; echo "$IMGLITS13" | sed 's/^/9/'; } | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'X=/opt/radiance/share/radiance/kernels/exl3r4d.so; B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; R=/opt/radiance/bin/r4d_selftest; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); g=""; case $l in 9*) g=9; l=${l#9} ;; esac; w=${l%%|*}; s=${l#*|}
  if [ $w = P ]; then test -s /opt/radiance/share/radiance/$s || echo "MISS$g $w|$s"; continue; fi
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; R) f=$R ;; X) f=$X ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS$g $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; for p in int7 int8 int9; do test -s /opt/radiance/share/radiance/$p-PROVENANCE.txt || echo "MISS $p-PROVENANCE.txt"; done; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in *"CHECKED "*) ;; *) xlog "VIRHE the image literal check did not run on $IMAGE: $(echo "$IMGMISS" | tail -3 | tr '\n' ';' | cut -c1-500)"; exit 1 ;; esac
MISS8=$(echo "$IMGMISS" | grep '^MISS ' | tr '\n' ';'); MISS9=$(echo "$IMGMISS" | grep '^MISS9 ' | sed 's/^MISS9 //' | tr '\n' ';')
[ -z "$MISS8" ] || { xlog "VIRHE $IMAGE is not the int9 image or later -- missing: $(echo "$MISS8" | cut -c1-900)"; exit 1; }
if [ -n "$MISS9" ]; then
  if [ "$DRY" = 1 ] && [ "$ALLOWOLD" = 1 ]; then
    echo "+ image check: SKIP (image lacks int9b / int10 / int11 literals; YDM_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the A4 (ds41-a4) image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
  fi
fi
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGLITS" | grep -c .) int9 / KVR / profiler literals present$([ -z "$MISS9" ] && echo " + all $(echo "$IMGLITS9B" | grep -c .) int9b + $(echo "$IMGLITS10" | grep -c .) int10 + $(echo "$IMGLITS11" | grep -c .) int11 + $(echo "$IMGLITS12" | grep -c .) int12 + $(echo "$IMGLITS13" | grep -c .) A4 + A10 literals (exl3_x3d_pf in the registry tail / radiance / exl3r4d.so, A10's prefill-band X3D kernel; gg_gemv_m16 in the registry tail / radiance / libr4d.so, RADIANCE_DS41_GG_M16_FORMS, the table line, r4d_gg_mr_pin_xp, r4d_selftest --mrcheck16 / --perfm16 / A4VERDICT / M16FORM, a4-PROVENANCE) (exl3_x3d in the registry tail / radiance / exl3r4d.so, int12-PROVENANCE$([ $F4G = 1 ] && echo ", lever F4G")) (the registry tail ending in ds41_mix_wv, ds41_gq, ds41_e2; GET i11; the GQ / E2 declare lines; libr4d's MIX_WV env and stats; r4d_selftest's mix_wv; int11-PROVENANCE)") + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
GQLIT=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF -- gemm_nt_ggml_pair /opt/radiance/share/radiance/kernels/libr4d.so && echo yes; grep -qzF -- gq_pair /opt/radiance/bin/r4d_selftest && echo case' 2>/dev/null | tr '\n' ' ')
[ "$DRY" = 1 ] && echo "+ lever 1U-GQ in this image (INFO): libr4d gemm_nt_ggml_pair $(case "$GQLIT" in *yes*) echo present ;; *) echo "ABSENT (GQ will read 'not declared': its arms SKIPPED)" ;; esac), r4d_selftest gq_pair $(case "$GQLIT" in *case*) echo present ;; *) echo "absent (P0 gq_pair SKIP)" ;; esac)"
PLUGGOT=$(timeout 60 docker run --rm --network none --entrypoint sh $IMAGE -c 'md5sum /opt/radiance/share/radiance/kernels/exl3r4d.so 2>/dev/null | cut -c1-8' 2>/dev/null)
if [ $SW = 0 ]; then [ "$DRY" = 1 ] && echo "+ exl3r4d.so md5: not checked (YDM_SW=0, a mock run)"
elif [ "${PLUGGOT:-}" != "$PLUGMD5" ]; then
  if [ "$DRY" = 1 ] && [ "$ALLOWOLD" = 1 ]; then echo "+ exl3r4d.so md5 ${PLUGGOT:-?} (want $PLUGMD5 = A10's ds41-x3d-pf 90b1fef): SKIP (YDM_DRY_ALLOW_OLD=1)"
  else xlog "VIRHE $IMAGE's kernels/exl3r4d.so md5 ${PLUGGOT:-?}, not the A10 plugin $PLUGMD5 (ds41-x3d-pf 90b1fef)"; exit 1; fi
else [ "$DRY" = 1 ] && echo "+ exl3r4d.so md5 $PLUGGOT = the A10 plugin (ds41-x3d-pf 90b1fef = int12's X3D-F + A10) -- OK"; fi
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $HD/../d0ab/s5ctr.py $MBT $EP $PRB $U1S $R12S $C48N $C4R $MBT2 $NCMB2 $C8MIX $YDS $DQ $(for m in pfxprobe exl3r4d-int12.so MD5SUMS; do [ $A10 = 1 ] && echo $PB/$m; done); do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $MBT $EP $PRB $U1S $R12S $MBT2 $NCMB2 $C8MIX $YDS 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
U1T=$(timeout 60 python3 -I -B $U1S selftest 2>&1 | tail -1); case "$U1T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$U1S(${U1T:0:300})" ;; esac
C48T=$(timeout 60 python3 -I -B $C48N --mb $MBP --selftest 2>&1 | tail -1); case "$C48T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C48N(${C48T:0:300})" ;; esac
R12T=$(timeout 60 python3 -I -B $R12S selftest 2>&1 | tail -1); case "$R12T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$R12S(${R12T:0:300})" ;; esac
C4T=$(timeout 60 python3 -I -B $C4R selftest 2>&1 | tail -1); case "$C4T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C4R(${C4T:0:300})" ;; esac
M2T=$(timeout 60 python3 -I -B $MBT2 --mb $MBP --selftest 2>&1 | tail -1); case "$M2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT2(${M2T:0:300})" ;; esac
N2T=$(timeout 60 python3 -I -B $NCMB2 --mb $MBP --selftest 2>&1 | tail -1); case "$N2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB2(${N2T:0:300})" ;; esac
CMT=$(timeout 60 python3 -I -B $C8MIX selftest 2>&1 | tail -1); case "$CMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C8MIX(${CMT:0:300})" ;; esac
XPT=$(timeout 60 python3 -I -B $YDS selftest 2>&1 | tail -1); case "$XPT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$YDS(${XPT:0:300})" ;; esac
python3 -I -B $D0B --help > /dev/null 2>&1 || HMISS="$HMISS run:d0blocks.py--help"
grep -q '"essay"' $TESTS || HMISS="$HMISS stepprobe-prompts:$TESTS"
FNS="stagx p0pf log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok c16x p1reads mwin idx16 ctrinfo i11now i11d i11g monons utcnow gpurun drafted readback swnote capd mbx cbhd cbhg mbdelta c4warm c8mixall c8run mbsec blockx coldx refreads gsx summary dstart p0a4 tabline idok gt48 ydid u1x ydreads ydsummary"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in O M0 B C P; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
f4 = int(sys.argv[9])
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0
    assert d["ds41_mix_wv"] == (0 if a == "M0" else int(sys.argv[7])) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == 1
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == f4
    assert d["gg_gemv_m16"] == (1 if a == "C" else 0) and d["exl3_x3d_pf"] == (1 if a == "P" else 0)
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -qF XS2_ENVFILE $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no XS2_ENVFILE)"
grep -qF 'RADIANCE_AQL_STAMP: ${AQLSTAMP:-}' $Y 2>/dev/null && grep -qF 'RADIANCE_AQL_STAMP_OUT: /stats/stamps.csv' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no RADIANCE_AQL_STAMP from AQLSTAMP / no stamp file under /stats)"
grep -qE 'RADIANCE_DS41_KV_REPLICATED|RADIANCE_DS41_MHC_FUSE|RADIANCE_DS41_IX_|RADIANCE_DS41_ARCODED|RADIANCE_DECODE_ROW_CAP|RADIANCE_DS41_MIX_WV|RADIANCE_DS41_GQ|RADIANCE_DS41_E2|RADIANCE_SCHED_CB_HOLD|RADIANCE_EXL3_X3D|RADIANCE_FN4_AR_XPUSH|RADIANCE_DS41_GG_M16|X3D_PF' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names a lever word: they come from the env file only)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*RADIANCE_DS41_GQ=3*RADIANCE_DS41_E2=1*RADIANCE_DS41_KV_REPLICATED=1*) ;; *) HMISS="$HMISS levenv:no-M1-GQ-E2-KVR" ;; esac
case "$LEVENV" in *RADIANCE_DS41_ARCODED*|*RADIANCE_DECODE_ROW_CAP*|*RADIANCE_SCHED_CB_HOLD*|*RADIANCE_FN4_AR_XPUSH*|*RADIANCE_DS41_GG_M16*) HMISS="$HMISS levenv:ARC-CAP-CBH-XPUSH-or-A4-at-start (A4's table is added by dstart after P0)" ;; esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="yd2m/ydsum.py $(md5sum < $YDS | cut -c1-8), prof12/p12win.py $(md5sum < $R12S | cut -c1-8), cap48/ncmb.py $(md5sum < $C48N | cut -c1-8), c4lev/c4rows.py $(md5sum < $C4R | cut -c1-8), 1u/u1sum.py $(md5sum < $U1S | cut -c1-8), i9b/i9bctr.py $(md5sum < $CTR | cut -c1-8) (INFO only), d0ab/d0blocks.py $(md5sum < $D0B | cut -c1-8) + s5ctr.py $(md5sum < $HD/../d0ab/s5ctr.py | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), harness mbt3.py $(md5sum < $MBT2 | cut -c1-8) / ncmb3.py $(md5sum < $NCMB2 | cut -c1-8) / c8mix.py $(md5sum < $C8MIX | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), dsm-yd2m.sh $(md5sum < $0 | cut -c1-8) (a copy of xpush/dsm-xpush.sh) (from 1u/dsm-1u.sh $(md5sum < $HD/../1u/dsm-1u.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | loadid.py $ST | ncmb.py $NMT | u1sum.py $U1T | p12win.py $R12T | ydsum.py $XPT | cap48/ncmb.py $C48T | c4rows.py $C4T | harness: mbt3.py $M2T / ncmb3.py $N2T / c8mix.py $CMT | d0blocks.py --help OK | all $(echo $FNS | wc -w) functions resolve, tools on PATH, every arm's pin (S5T, draft_sample 0, draft_lend 1, dspark_conf 1 / C*, dspark_conf_rows 1$([ $SW = 1 ] && echo ", SP 0, M1 on, IX off, draft_depth0 $D0, sched_row_cap $CAP, ds41_arcoded $ARCV, every new lever 0 but the arm's" || echo "; YDM_SW=0: no int9b / int10 / int11 switch")), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, AQLSTAMP -> RADIANCE_AQL_STAMP, no lever word), LEVENV with MHC_FUSE=0 + GQ=3 + E2=1, without KVR / ARC / CAP -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  for a in O B C P; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done
  echo "+ base pin + fixed pin (O): $(pinj O)"
  echo "+ readings: identity a1 / gs$GSC (MAXSEQS ${YDM_MAXSEQS:-base}) under B, C, P + staggered B (and C budget allowing; conc $LDCONC, $LDPF-token prefills) + a1-B / a1-C vs ref12 a1 ($([ -s $R/xdsm/ref12/xref12-a1.jsonl ] && echo present || echo MISSING)) | mb c8 [$C8ORD] ${C8WIN} s | 1u essay [$U1ORD] (1 warm-up + $NREADS) | c16 [$C16ORD] (fixed harness ncmb3, wall split) | no-regression guard +$DECTH % | P0 card $P0UUID ($(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | grep -c "^${P0UUID#GPU-}$") sysfs match)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/c8 $O/snap $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
ydsummary
