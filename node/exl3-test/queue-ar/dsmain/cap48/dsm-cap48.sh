#!/bin/bash
# dsm-cap48.sh (12.11.2026, queue-ar/dsmain/cap48, ds-main-port cap48 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48.sh. Log tag "RESULT xc48 ", out runs/xdsm/cap48 (XC48_OUT). Made from COPIES of
# kv4/dsm-kv4.sh (f89a6cc2: the n16 start (nvfp4, MAXSEQS 16, HOSTPOOL 19456, MML 524288, NVG 48), LITS / FIX / DET, armj / pinj
# / armset, modes, gate1, swstart, identw, galone / gsconc / lid16 (the gs16 / lid16 identity), cmpx / idrec, mbw, deepr / guard16,
# capture, the image / helper / sidecar / yaml checks; tag xkv4 -> xc48) and stack5/dsm-stack5.sh (63649414: the start words
# LEVENV incl. RADIANCE_LOG_STEPS=1 / RADIANCE_DS41_ENGDD=0 / RADIANCE_PLACE_HB_RANKS=1,2, the S5T pin, the c8 mb blocks' phase
# runner nc/ncmb.py -> copied here); + cap48probe.py (a COPY of stack4/loadid.py ee7d9663 + temperature / per-prompt seed + a set
# mode) + cap48sum.py (steps a window, the cap counters, identity / speed / guard summary); ncmb.py / ncone.py copied from kv4/
# (= nc/'s ded624e5 / a24570f8). stack4/loadid.py, greedyset.py, stack5/s5ctr.py, mb.py, deepq.py are read only. Originals unchanged.
# LEVER CAP48 (ds41-cap48 e5acfbc on ds41-int9 0cb6fd3; image radiance-local:ds41-cap48-<sha8>-5c13e555-l = the int9 image + its
#   bin/radiance): a scheduler cap on a step's DECODE rows (each decoding sequence's own row + its drafts) -- RADIANCE_DECODE_ROW_CAP
#   at start, fn4 sched_row_cap in process (0 off .. 4096). Over the cap, drafts are trimmed after the whole step is planned: the
#   largest common depth the room (cap - sequences) holds, the rest one each to the deeper sequences in plan order (the deepest
#   proposals first; equal depths: the later sequence). Prefill rows are not counted; every prefill piece is D3's (the trimmed
#   rows' budget goes unused). The step issued ahead is capped too. A step whose decoding sequences alone exceed the cap is left
#   (noted once). Exact: only how many drafts a step verifies moves; greedy output and a seeded T > 0 sample whose drafts are not
#   coupled are the target's alone (the verify keeps it). Under draft_sample >= 1 (lever SD's maximal coupling) a T > 0 row's
#   coupled draw depends on the draft at its position (abi/rad_spechub.h: "not seed-identical to the ordinary pick"), so a T 1
#   answer then depends on the depth trajectory -- distribution-exact, as every depth policy (adaptive, depth 0, draft_occ) is;
#   the strict T 1 identity therefore runs at draft_sample 0, and draft_sample 85 at T 1 is an OPTIONAL labelled reading.
# WHY (coordinator 12.11.): kv4 n16 answered 9 / 20 (gs16) and 8 / 20 (lid16) greedy prompts differently under load; under ds41_bi 2
#   the decode segment's q_a / kv_g / i_w / q_b, the indexer runs and attention run over the whole segment, and gemm_nt_ggml uses
#   its decode form only at M <= 48 (r4d_ds41_ggml.hip:3671). With the cap every decode piece stays in the <= 48-row forms.
# ONE START c16 = kv4's n16 (int9 cap48 image, nvfp4 latent, MAXSEQS 16, HOSTPOOL 19456, MML 524288, NST 5) with stack5's start
#   words + NVG 48 + RADIANCE_DECODE_ROW_CAP=XC48_CAP (48). Pin = the fixed pin (stack4's + draft_occ_k 32 + ds41_psk_k 8) + S5T
#   {ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 4, ds41_psk_k 8, ds41_psk_cr 1, ds41_psk_l20 1, dspark_pfctx 1,
#   ds41_engdd 1, sched_noa 2, sched_bi_chunk 3; stage_short / stage_pick / place_hb 0} + sched_row_cap. ARMS (POST + epoch_bump,
#   a nudge, every switch READ BACK before each phase):
#     S  S5T + cap XC48_CAP                       (speed: mb c16 / c8)
#     D  S5T + DET + cap                          (T 0 identity; DET = fixed depth 5: no adaptive / depth 0 / occupancy / hybrid /
#                                                  confidence cut -> 16 sequences x 6 = 96 decode rows a step without the cap)
#     Z  D + draft_sample 0                       (T 1 strict identity: the seeded pick path is independent of the drafts)
#     d  D with cap 0   /   z  Z with cap 0       (the in-process OFF control, T 0 / T 1)
# PHASES (XC48_PHASES, in this order; NAME:ARM:KIND; a phase that would end past XC48_BUDGET_MIN is skipped and logged):
#     a1:D:alone0  gs16:D:gs0  lid16:D:lid0  a1z:Z:set1  gs16z:Z:gs1  lid16z:Z:lid1  c16:S:mb16  c8:S:mb8  gs16o:d:gs0
#     lid16o:d:lid0  a2:D:alone0  lid16zo:z:lid1   [+ a1s:D:set1 lid16s:D:lid1 under XC48_INFO85=1]
#   alone0 = greedyset.py's 20 prompts ALONE (c1, T 0, 192 tok); gs0 = the same at conc 16; lid0 = stack4/loadid.py: each prompt
#   under XC48_LIDC (14) mb streams + one fresh ~XC48_LIDPF (1100) newcomer at +0.5 s = 16 sequences (kv4's lid16); set1 / gs1 /
#   lid1 = the same at T 1 with seed XC48_SEEDB + i (cap48probe.py set / run); mb16 / mb8 = nc's ncmb.py --mode phase (mbt.py's
#   phase_c at T 0) dur XC48_DUR 60 warm XC48_WARM 15. Prefix cache reset before each phase; GET /admin/fn4 + /stats at both edges.
# READINGS:
#   (a) IDENTITY (strict, B48): gs16 / lid16 vs a1 (T 0), gs16z / lid16z vs a1z (T 1, seeded, draft_sample 0) -- all byte-identical
#       required with the cap on; CONTROL (cap off, reported as counts): gs16o / lid16o vs a1, lid16zo vs a1z; S: a1 vs bz d0
#       (dsd-d0-gs-c1: the cap image alone = today's alone), a2 vs a1; I (cross-start): a1 vs kv4 n16-a1. Info85: lid16s vs a1s.
#   (b) STEPS: per window, the step log (RADIANCE_LOG_STEPS=1, docker logs -t; a decode step's n_tok IS its decode rows -- never
#       the line's first 8 entries; MIXED steps are counted, their decode rows read from the counter) + GET /admin/fn4 "cap"
#       deltas (steps_capped, drafts_trimmed, ahead_capped, steps_over, steps_gt48, decode_steps; max_rows a gauge) + deepseek41's
#       'lever BI stats' (> 48-row steps, granular 512 segmented steps). Cap on: steps_gt48 +0 and no decode step over 48 rows.
#   (c) SPEED: mb c16 / c8 at T 0 under S (cap on), labelled CROSS-START vs kv4 n16 c16 230.5 tok/s (~15.7 a stream; read from
#       runs/ds41-exl3.log "RESULT xkv4 speed n16:") and stack5 S5T c8 (read from "RESULT xs5 ARMS T 0": S5T 173.6 +- 2.6 at
#       MAXSEQS 8). No baseline re-runs.
# COUNTER GUARD (VIRHE): the cap counters move (the cap-on DET windows: steps_capped and drafts_trimmed > 0, steps_gt48 +0; the
#   alone windows and the cap-off windows capped nothing); the pins read back after every switch; PSK on the corrected /stats
#   psk counters over a cold ~16k prompt (stack5's s5ctr.py, T: psk + tail + l20 + pfctx + engdd; never layers_skipped); freshness
#   on cached == 0 (every lid newcomer cold; every mb newcomer cached 0 -- the prompt-size spread is information).
# ~29 min (estimate; budget XC48_BUDGET_MIN 35): start + checks + guard ~4.7, a1 / gs16 0.7, lid16 ~3.9, a1z / gs16z 0.7, lid16z
#   ~3.9, c16 1.9, c8 1.8, gs16o 0.3, lid16o ~3.9, a2 0.4, lid16zo ~3.9, compares 0.5, capture + parse 1.5 (+ ~4.3 with XC48_INFO85=1).
#   Disk guard XC48_DISKMIN_MB 11000 before the start.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact.
# Knobs: XC48_IMAGE, XC48_ITEM, XC48_OUT, XC48_CAP (48), XC48_PHASES, XC48_INFO85 (0), XC48_BIC (3), XC48_NVG (48), XC48_HP (19456),
#   XC48_MS (16), XC48_LIDC (14), XC48_LIDPF (1100), XC48_SEEDB (1000), XC48_MT (192), XC48_DUR (60), XC48_WARM (15), XC48_GDEP (16000),
#   XC48_DISKMIN_MB (11000), XC48_BUDGET_MIN (35), XC48_SIDECAR, XC48_SIDESHA, XC48_CAPSW (1; 0 = the arms carry no sched_row_cap
#   and every cap check is skipped -- ONLY for a server without the switch, e.g. the no-GPU mock's int8 templates; the item keeps 1).
# DRYRUN=1 bash dsm-cap48.sh : prints every command; checks the image literals (lever CAP48's switch / env / counters / start line +
#   kv4's and stack5's), every helper (files, parse, selftests), the sidecars, the yaml, the references, the disk; no server, no
#   GPU, no request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries the start words)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XC48_OUT:-$R/xdsm/cap48}; export STATSDIR=$O/stats
RK4=$R/xdsm/kv4; RD=$R/xdsm/draft; RM=$R/xdsm
export IMAGE=${XC48_IMAGE:?set XC48_IMAGE}
ITEM=${XC48_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cd-cap48}
CAP=${XC48_CAP:-48}; CAPSW=${XC48_CAPSW:-1}; INFO85=${XC48_INFO85:-0}; BIC=${XC48_BIC:-3}; NVG=${XC48_NVG:-48}; HP=${XC48_HP:-19456}; MS=${XC48_MS:-16}
LIDC=${XC48_LIDC:-14}; LIDPF=${XC48_LIDPF:-1100}; SEEDB=${XC48_SEEDB:-1000}; MT=${XC48_MT:-192}
DUR=${XC48_DUR:-60}; WARMS=${XC48_WARM:-15}; GDEP=${XC48_GDEP:-16000}; DISKMIN=${XC48_DISKMIN_MB:-11000}; BUDGET=${XC48_BUDGET_MIN:-35}
SIDE=${XC48_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XC48_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
DPH="a1:D:alone0 gs16:D:gs0 lid16:D:lid0 a1z:Z:set1 gs16z:Z:gs1 lid16z:Z:lid1 c16:S:mb16 c8:S:mb8 gs16o:d:gs0 lid16o:d:lid0 a2:D:alone0 lid16zo:z:lid1"
[ "$INFO85" = 1 ] && DPH="$DPH a1s:D:set1 lid16s:D:lid1"
PHASES=${XC48_PHASES:-$DPH}
for v in $CAP $BIC $NVG $HP $MS $LIDC $LIDPF $SEEDB $MT $DUR $WARMS $GDEP $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case $INFO85 in 0|1) ;; *) echo "XC48_INFO85 is 0 or 1 (got '$INFO85')"; exit 1 ;; esac
case $CAPSW in 0|1) ;; *) echo "XC48_CAPSW is 1 (the item) or 0 (a server without sched_row_cap: the mock) (got '$CAPSW')"; exit 1 ;; esac
[ $CAP -ge 1 ] && [ $CAP -le 4096 ] || { echo "XC48_CAP is 1 .. 4096 (got $CAP)"; exit 1; }
[ $MS -le $CAP ] || { echo "XC48_MS ($MS) above XC48_CAP ($CAP): the cap would leave every full step (steps_over)"; exit 1; }
case $BIC in 3|4) ;; *) echo "XC48_BIC is 3 or 4 (NOA needs option D)"; exit 1 ;; esac
[ $NVG -le 48 ] || { echo "XC48_NVG is 0 (off) .. 48"; exit 1; }
[ $WARMS -lt $DUR ] || { echo "the warm-up is not shorter than the phase"; exit 1; }
[ $((LIDC + 2)) -le $MS ] || { echo "XC48_LIDC ($LIDC) + the probe + the newcomer exceed MAXSEQS $MS"; exit 1; }
for p in $PHASES; do
  IFS=: read -r pn pa pk <<< "$p"
  case "$pn" in ''|*[!a-z0-9]*) echo "XC48_PHASES: a phase name is [a-z0-9]+ (got '$p')"; exit 1 ;; esac
  case "$pa" in S|D|Z|d|z) ;; *) echo "XC48_PHASES: arm S / D / Z / d / z (got '$p')"; exit 1 ;; esac
  case "$pk" in alone0|gs0|lid0|set1|gs1|lid1|mb16|mb8) ;; *) echo "XC48_PHASES: kind alone0 gs0 lid0 set1 gs1 lid1 mb16 mb8 (got '$p')"; exit 1 ;; esac
done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/../stack4/loadid.py; CTR=$HD/../stack5/s5ctr.py                                    # stack4's / stack5's, read only
PRB=$HD/cap48probe.py; NCMB=$HD/ncmb.py; NCONE=$HD/ncone.py; SUM=$HD/cap48sum.py           # this item's
MBP=$D/mb.py; TESTS=$D/ds41_tests.py; DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
TAG=c16; T0=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xc48 $*"; }
elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }
# ---------------------------------------------------------------- the start (stack5's start words; kv4 n16's MAXSEQS / latent / NVG)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32; HBRANKS=1,2
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS"
LEVENV=$(echo $LEVENV)
CAPENV="RADIANCE_DS41_NVGATHER=$NVG RADIANCE_DECODE_ROW_CAP=$CAP"     # + RADIANCE_DS41_LATENT=nvfp4 from the shell (the yaml's list)
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
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up"
LITS="$LITS|keys,mxfp4,(80,bytes,an,entry;,RADIANCE_DS41_IDXKEY)|kv:,the,pool,backs|kv:,the,cache,can,grow,to|VRAM,budget"
LITS="$LITS|lever,CAP48:,decode,row,cap"                      # + its value, checked apart in modes (XC48_CAP)
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": 32, "ds41_psk_k": 8}); print(json.dumps(d))' "$FIX5") || { echo "pin JSON"; exit 1; }
S5T="{\"ds41_bi\": 2, \"draft_sample\": 85, \"draft_occ\": 6, \"place_lf\": 0, \"ds41_psk\": 4, \"ds41_psk_k\": 8, \"ds41_psk_cr\": 1, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1, \"sched_noa\": 2, \"sched_bi_chunk\": $BIC, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0}"
# kv4's / dsm-pskl20.sh's deterministic drafting overlay: a fixed depth (5), no depth 0 / occupancy / lookup hybrid / confidence cut
DET='{"draft_adaptive": 0, "draft_depth0": 0, "draft_occ": 0, "draft_hybrid": 0, "dspark_conf": 0}'
armj() {   # armj S|D|Z|d|z: the arm's own switches (JSON; the cap last; none under XC48_CAPSW=0)
  python3 -c 'import json,sys
a, s5t, det, cap, sw = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
d = dict(s5t)
if a in "DZdz": d.update(det)
if a in "Zz": d["draft_sample"] = 0
if sw: d["sched_row_cap"] = cap if a in "SDZ" else 0
print(json.dumps(d))' "$1" "$S5T" "$DET" "$CAP" "$CAPSW"
}
NAMEOF() { case $1 in S) echo "S (S5T + cap $CAP)" ;; D) echo "D (S5T + DET + cap $CAP)" ;; Z) echo "Z (D + draft_sample 0)" ;; d) echo "d (D, cap 0)" ;; z) echo "z (Z, cap 0)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON ...]: the fixed pin + the arm + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$(armj $1)" "${@:2}"
}
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
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED RADIANCE_DECODE_ROW_CAP"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset (stack5's)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV $CAPENV; do export "$kv"; done
}
# ---------------------------------------------------------------- copied from dsm-kv4.sh (f89a6cc2; tag xkv4 -> xc48)
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden). NAME: S: strict,
  # B48: the > 48-row BI check under the cap (strict), C: the cap-off control (a count), I: information (labelled)
  [ "$DRY" = 1 ] && { echo "+ identity $1 ${3:+($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;;
    *) case $1 in C:*|I:*) xlog "identity $1${3:+ ($3)}: $2 -- reported (control / information)" ;;
         *) xlog "identity $1${3:+ ($3)}: VIRHE $2 -- reported, not hidden" ;; esac ;; esac
}
cmpx() {   # cmpx NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $2 $3 --json $O/cmp-${1#*:}.json -> identity $1"; return 0; }
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then
    case $1 in C:*|I:*) xlog "$1: not compared (missing $2 or $3)" ;; *) xlog "$1: VIRHE not compared (missing $2 or $3)" ;; esac
    echo "$1 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/cmp-${1#*:}.json --label "$1" "$2" "$3" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1" "${r:-VIRHE compare failed}" "$(basename $2) vs $(basename $3)"
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xc48 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "S:$1-warmup $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: MODEWORDS in the container, the latent read back (nvfp4), mover 0 kept free x4, rule-w, K1, ARGATHER 48,
  # step-log lines, the CAP48 start line, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env; 'latent nvfp4 (288 bytes an entry; RADIANCE_DS41_LATENT)' in the start log; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds =1; ARGATHER 48-row note; step-log lines; 'lever CAP48: decode row cap $CAP (RADIANCE_DECODE_ROW_CAP, fn4 sched_row_cap)' and no CAP48 note about --max-num-seqs (MAXSEQS $MS <= $CAP); literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL LL nz lit found="" miss="" want
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not as asked:$bad"; ok=0; }
  want="latent nvfp4 (288 bytes an entry; RADIANCE_DS41_LATENT)"
  LL=$(docker logs $CN 2>&1 | grep -m1 -o 'latent [a-z0-9]* ([0-9]* bytes an entry; RADIANCE_DS41_LATENT)')
  [ "$LL" = "$want" ] || { xlog "$1: VIRHE the latent mode read back is '$LL', asked '$want'"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'lever CAP48: the decode row cap .* covers up to' && { xlog "$1: VIRHE a CAP48 --max-num-seqs note at MAXSEQS $MS <= $CAP"; ok=0; }
  if [ $CAPSW = 1 ]; then docker logs $CN 2>&1 | grep -qF "lever CAP48: decode row cap $CAP (RADIANCE_DECODE_ROW_CAP, fn4 sched_row_cap)" \
    || { xlog "$1: VIRHE no 'lever CAP48: decode row cap $CAP (RADIANCE_DECODE_ROW_CAP, fn4 sched_row_cap)' start line"; ok=0; }; fi
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: latent read back '$LL', ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever CAP48|kv: the pool backs|kv: the cache can grow|lever PSK declared|lever BI declared|lever E1|chunk geometry" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-330 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  echo "$1 modes $([ $ok = 1 ] && echo OK || echo VIRHE) | latent '$LL'" >> $O/guard.txt
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
ARMOK=1
armset() {   # armset TAG ARM: the pin + the arm + epoch_bump, a nudge, every switch READ BACK (polled <= 30 s; else VIRHE, ARMOK 0)
  local g r i
  fn4set "$(pinj $2 '{"epoch_bump": 1}')"; nudge
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj $2) applied, nothing pending, cap in force $(armj $2 | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sched_row_cap", "not checked (XC48_CAPSW=0)"))') -> RESULT xc48 $1 arm $2 applied (read back)"; return 0; }
  for i in $(seq 1 30); do
    g=$(fn4get)
    r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
s = g["switches"]
bad = {k: s.get(k) for k, v in w.items() if s.get(k) != v}
cap = (g.get("cap") or {}).get("in_force")
if "sched_row_cap" in w and cap != w["sched_row_cap"]: bad["cap.in_force"] = cap
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | cap=%s bi=%s psk=%s/%s cr=%s bic=%s noa=%s occ=%s ds=%s adapt=%s d0=%s hyb=%s conf=%s epoch=%s" % (
    cap, s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("ds41_psk_cr"), s.get("sched_bi_chunk"), s.get("sched_noa"), s.get("draft_occ"),
    s.get("draft_sample"), s.get("draft_adaptive"), s.get("draft_depth0"), s.get("draft_hybrid"), s.get("dspark_conf"), g.get("epoch")))' "$g" "$(pinj $2)" 2>&1 | cut -c1-500)
    case "$r" in OK*) xlog "$1 arm $(NAMEOF $2) applied (read back): $r"; return 0 ;; esac
    sleep 1; nudge
  done
  xlog "$1 arm $2: VIRHE the pin is not applied after 30 s ($r) -- no reading under it"; echo "$1 pins VIRHE arm $2 $r" >> $O/guard.txt; ARMOK=0; return 1
}
gate1() {   # restart + ONE check; over the limit logged (a slow state shows in the 1-user ms of every start)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && st="SLOW (not retried)"
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim)"
}
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before the start
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (the start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse the start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
envline() { env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2600; }
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ]
}
pyx() {   # pyx NAME CMD...: one parser run, its last line logged (VIRHE when it fails)
  local nm=$1 rc; shift
  [ "$DRY" = 1 ] && { echo "+ $* > $O/$nm.out"; return 0; }
  timeout 900 "$@" > $O/$nm.out 2>&1; rc=$?
  xlog "$nm: $(tail -1 $O/$nm.out | cut -c1-3500)$([ $rc = 0 ] || echo " -- VIRHE rc $rc")"
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
snap() {   # snap NAME: GET /admin/fn4 and the full /stats -> NAME-fn4.json / NAME-st.json
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 > $O/$1-fn4.json; GET /stats > $O/$1-st.json"; return 0; }
  curl -s -m 10 $URL/admin/fn4 > $O/$1-fn4.json 2>/dev/null; curl -s -m 30 $URL/stats > $O/$1-st.json 2>/dev/null
}
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
now() { [ "$DRY" = 1 ] && echo T || date +%s.%N; }
wmark() { [ "$DRY" = 1 ] || echo "$2:$3:$4" >> $O/$1-wins.txt; }   # wmark TAG NAME T0 T1: a phase's window
# ---------------------------------------------------------------- the start and its checks
startenv() {   # stack5's start words + kv4 n16's MAXSEQS / latent nvfp4 / NVG + RADIANCE_DECODE_ROW_CAP
  stackenv
  export RADIANCE_DS41_LATENT=nvfp4      # the yaml's list passes it from the shell
  export MODEL=$M225; HP29=$HP; best_vars
  export MNBT=4160 NST=5 MAXSEQS=$MS MML=524288 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$TAG.env
  MODEWORDS=$(echo "$LEVENV $CAPENV RADIANCE_DS41_LATENT=nvfp4")
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV $CAPENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV $CAPENV > $XS2_ENVFILE; fi
}
swstart() {   # swstart TAG: the switches at start (stack4's start values; sched_row_cap = RADIANCE_DECODE_ROW_CAP; every switch the arms
  # name present; GET "cap" in force); ds41_psk 99 and sched_row_cap 4097 refused
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, ds41_bi 0, ds41_psk 0, sched_bi_chunk 0, sched_row_cap $CAP (RADIANCE_DECODE_ROW_CAP), cap.in_force $CAP, cap.steps_gt48 present, every switch of $(pinj z) present; POST ds41_psk 99 -> 400, sched_row_cap 4097 -> 400"; return 0; }
  local s0 c1 c2 r
  s0=$(fn4get); c1=$(fn4code '{"ds41_psk": 99}'); c2=400; [ $CAPSW = 1 ] && c2=$(fn4code '{"sched_row_cap": 4097}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; want = json.loads(sys.argv[4]); cap = int(sys.argv[5]); c = s0.get("cap") or {}; capsw = sys.argv[6] == "1"
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("sched_bi_chunk") == 0
ok = ok and (not capsw or (w.get("sched_row_cap") == cap and c.get("in_force") == cap and "steps_gt48" in c and "drafts_trimmed" in c))
miss = [k for k in want if k not in w]
ok = ok and not miss and sys.argv[2] == "400" and sys.argv[3] == "400"
print(("OK" if ok else "FAIL") + " sched_row_cap %s (cap in force %s, counts %s) ds41_bi %s ds41_psk %s/%s sched_bi_chunk %s draft_occ %s draft_sample %s | ds41_psk 99 -> HTTP %s, sched_row_cap 4097 -> HTTP %s%s" % (
    w.get("sched_row_cap"), c.get("in_force"), {k: v for k, v in c.items() if k != "in_force"}, w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k"),
    w.get("sched_bi_chunk"), w.get("draft_occ"), w.get("draft_sample"), sys.argv[2], sys.argv[3], (" | not in the registry: %s" % miss) if miss else "") + ("" if capsw else " | cap NOT CHECKED (XC48_CAPSW=0)"))' "$s0" "$c1" "$c2" "$(pinj z)" "$CAP" "$CAPSW" 2>&1 | cut -c1-700)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
SPEEDOK=1
guard16() {   # guard16 TAG: arm S read back, a cold ~GDEP deepq prompt (prefix cache reset), stack5's s5ctr.py guard over it:
  # PSK (latched / planned_steps / cut_passes / rows_skipped grow, plan_mismatch +0; never layers_skipped) + tail-only + L20 +
  # PSK-DS + E1 must move, SA / HB must not
  local tag=$1 t00=$(date +%s) j r
  armset $tag S || { SPEEDOK=0; return 1; }
  pcreset $tag-g16; snap $tag-g16-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-g16.txt $GDEP 512 > $O/$tag-g16.out 2>&1)"
  snap $tag-g16-1
  [ "$DRY" = 1 ] && { echo "+ PSK guard: python3 -I -B $CTR guard --need psk,tail,l20,pfctx,engdd --forbid sa,hb --st0/--st1 $O/$tag-g16-{0,1}-st.json --fn0/--fn1 $O/$tag-g16-{0,1}-fn4.json --hb 128 --hb-ranks $HBRANKS (cold deepq $GDEP: cached 0); a failed guard stops the speed readings"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-g16.out | cut -c7-)
  r=$(timeout 120 python3 -I -B $CTR guard --need psk,tail,l20,pfctx,engdd --forbid sa,hb --st0 $O/$tag-g16-0-st.json --st1 $O/$tag-g16-1-st.json \
      --fn0 $O/$tag-g16-0-fn4.json --fn1 $O/$tag-g16-1-fn4.json --hb 128 --hb-ranks $HBRANKS 2>&1 | tail -1 | cut -c1-900)
  local cached; cached=$(echo "$j" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("cached"))' 2>/dev/null)
  [ "$cached" = 0 ] || r="FAIL not cold (cached $cached) | $r"
  echo "$tag psk $r" >> $O/guard.txt
  case "$r" in OK*) xlog "$tag PSK guard (S5T, cold deepq $GDEP, cached $cached): $r | $(thermline $t00 $(date +%s))" ;;
    *) xlog "$tag PSK guard: VIRHE $r -- no speed reading in this start"; SPEEDOK=0 ;; esac
}
# ---------------------------------------------------------------- the phases (each: arm read back, prefix cache reset, snapshots, window)
phase() {   # phase NAME ARM KIND
  local nm=$1 a=$2 k=$3 w=$TAG-$1 t00=$(date +%s) ta tb j
  armset $w $a || return 1
  pcreset $w; snap $w-0; ta=$(now)
  case $k in
    alone0) run "(cd $D && timeout 900 python3 $GS $URL $O/$w.jsonl --conc 1 --max-tokens $MT > $O/$w.txt 2>&1)" ;;
    gs0)    run "(cd $D && timeout 900 python3 $GS $URL $O/$w.jsonl --conc $MS --max-tokens $MT > $O/$w.txt 2>&1)" ;;
    lid0)   run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$w.jsonl --conc $LIDC --warm 10 --offset 0.5 --pf-tok $LIDPF --max-tokens $MT > $O/$w.txt 2>&1)" ;;
    set1)   run "(cd $D && timeout 900 python3 -I $PRB set --url $URL --gs $GS --out $O/$w.jsonl --conc 1 --temperature 1 --seed-base $SEEDB --max-tokens $MT > $O/$w.txt 2>&1)" ;;
    gs1)    run "(cd $D && timeout 900 python3 -I $PRB set --url $URL --gs $GS --out $O/$w.jsonl --conc $MS --temperature 1 --seed-base $SEEDB --max-tokens $MT > $O/$w.txt 2>&1)" ;;
    lid1)   run "(cd $D && timeout 2700 python3 -I $PRB run --url $URL --gs $GS --mb $MBP --out $O/$w.jsonl --conc $LIDC --warm 10 --offset 0.5 --pf-tok $LIDPF --max-tokens $MT --temperature 1 --seed-base $SEEDB > $O/$w.txt 2>&1)" ;;
    mb16|mb8) local n=${k#mb}
            run "(cd $D && timeout $(( DUR + 1200 )) python3 -I -B $NCMB --mb $MBP --mode phase --temperature 0 --n $n --tag $w --dur $DUR --warm $WARMS --reqs $O/$w-reqs.jsonl --stats-prefix $O/$w-st --samples $O/$w-samp.jsonl > $O/$w.txt 2>&1)" ;;
  esac
  tb=$(now); snap $w-1; wmark $TAG $nm $ta $tb
  [ "$DRY" = 1 ] && return 0
  case $k in
    alone0|gs0) j=$(tail -1 $O/$w.txt | cut -c1-200) ;;
    set1|gs1)   j=$(grep -m1 '^SET ' $O/$w.txt | cut -c5- | cut -c1-400) ;;
    lid0|lid1)  j=$(grep -m1 '^LOADID ' $O/$w.txt | cut -c8- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("%s of %s answered, probe %s s mean | newcomers %s of %s cold, TTFT %s s mean | mixed steps a probe min %s / median %s | streams %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["mixed_min"], d["mixed_median"], d["c4_requests_done"], d["c4_errors"]))' 2>&1 | cut -c1-500) ;;
    mb*)        j=$(grep -m1 '^MB ' $O/$w.txt | cut -c4- | cut -c1-600) ;;
  esac
  xlog "$w ($k, arm $(NAMEOF $a)): ${j:-VIRHE no result line ($(tail -2 $O/$w.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))} | cap $(python3 -I -B $SUM capd --fn0 $O/$w-0-fn4.json --fn1 $O/$w-1-fn4.json 2>&1 | tail -1 | cut -c1-400) | $(( $(date +%s) - t00 )) s | $(thermline $t00 $(date +%s))"
}
pest() { case $1 in alone0|set1) echo 1 ;; gs0|gs1) echo 1 ;; lid0|lid1) echo 4 ;; mb16|mb8) echo 2 ;; *) echo 2 ;; esac; }
idhas() {   # a phase that ran (DRYRUN: every phase of XC48_PHASES)
  case " $DONEPH " in *" $1 "*) return 0 ;; esac
  [ "$DRY" = 1 ] && case " $PHASES " in *" $1:"*) return 0 ;; esac
  return 1
}
idf() { echo $O/$TAG-$1.jsonl; }
idall() {   # idall: the compares while the server is up (loadid.py cmp tokenizes the first differing token)
  local p
  idhas a1 && cmpx S:a1-vs-d0 $RD/dsd-d0-gs-c1.jsonl $(idf a1)
  idhas a1 && cmpx I:a1-vs-kv4-n16-a1 $RK4/n16-a1.jsonl $(idf a1)
  for p in gs16 lid16; do idhas $p && idhas a1 && cmpx B48:$p-vs-a1 $(idf a1) $(idf $p); done
  for p in gs16z lid16z; do idhas $p && idhas a1z && cmpx B48:$p-vs-a1z $(idf a1z) $(idf $p); done
  for p in gs16o lid16o; do idhas $p && idhas a1 && cmpx C:$p-vs-a1 $(idf a1) $(idf $p); done
  idhas lid16zo && idhas a1z && cmpx C:lid16zo-vs-a1z $(idf a1z) $(idf lid16zo)
  idhas a2 && idhas a1 && cmpx S:a2-vs-a1 $(idf a1) $(idf a2)
  idhas lid16s && idhas a1s && cmpx I:lid16s-vs-a1s $(idf a1s) $(idf lid16s)
  idhas a1s && idhas a1z && cmpx I:a1s-vs-a1z $(idf a1z) $(idf a1s)
  return 0
}
DONEPH=""
cstart() {   # the one start, its phases, compares, capture, parse
  diskok $TAG || return 1
  startenv
  xlog "$TAG: start (latent nvfp4, MAXSEQS $MS, HOSTPOOL $HP MiB, MML 524288; stack5's NST 5 start words + NVG $NVG + RADIANCE_DECODE_ROW_CAP=$CAP) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms $(for a in S D Z d z; do echo -n "$a $(armj $a) "; done)| env: $(envline)"
  gate1 $TAG 70 || { xlog "$TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $TAG || { xlog "$TAG: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $TAG || { keeplog $TAG-nodraft; down; return 1; }
  swstart $TAG || { keeplog $TAG-switch; down; return 1; }
  identw $TAG $WARM_REF_225
  snap $TAG-st0
  stopfile || guard16 $TAG
  local p pn pa pk e el
  for p in $PHASES; do
    stopfile && break
    IFS=: read -r pn pa pk <<< "$p"
    e=$(pest $pk); el=$(( ($(date +%s) - T0) / 60 ))
    if [ "$DRY" != 1 ] && [ $((el + e + 3)) -gt $BUDGET ]; then xlog "$TAG-$pn SKIPPED: $el min elapsed + ~$e min + ~3 min to finish > the budget $BUDGET min"; continue; fi
    case $pk in mb*) [ $SPEEDOK = 1 ] || { xlog "$TAG-$pn: VIRHE skipped -- the PSK guard failed (no speed reading in this start)"; continue; } ;; esac
    phase $pn $pa $pk && DONEPH="$DONEPH $pn"
  done
  idall
  [ "$DRY" = 1 ] || xlog "$TAG: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | PSK $(curl -s -m 10 $URL/stats 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("psk"))' 2>/dev/null | cut -c1-300) | cap $(fn4get | python3 -c 'import json,sys; print(json.load(sys.stdin).get("cap"))' 2>/dev/null) | BI $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200)"
  capture $O/$TAG-engine.log.gz
  down
  xlog "$TAG: done ($(elapsed) min into the item; phases:$DONEPH)"
}
report() {   # cap48sum.py report: steps a window (the step log, n_tok), the cap / BI counters, identity, speed vs the references, guards
  if [ "$DRY" = 1 ] || [ -s $O/$TAG-engine.log.gz ]; then
    pyx $TAG-steps python3 -I -B $SUM steps --log $O/$TAG-engine.log.gz $( [ -s $O/$TAG-wins.txt ] && sed 's/^/--win /' $O/$TAG-wins.txt | tr '\n' ' ') --json $O/$TAG-steps.json
  else xlog "$TAG-steps: no engine log (the start stopped before its capture, VIRHE above): no step windows"; fi
  run "python3 -I -B $SUM report --dir $O --tag $TAG --phases '$PHASES' --cap $CAP --capsw $CAPSW --dur $DUR --warm $WARMS --log $L > $O/report.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ every RESULT line of $O/report.txt -> the log (identity a pair, steps + cap counters a window, speed vs kv4 n16 c16 / stack5 S5T c8 (cross-start), guards, summary)"; return 0; }
  while IFS= read -r l; do case "$l" in "RESULT xc48 "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  local nv; nv=$(( $(grep -c VIRHE $O/report.txt 2>/dev/null) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  log "RESULT xc48 guards (item): $([ "$nv" = 0 ] && echo "OK (pins read back, the cap counters moved where it acts and nowhere else, PSK counters grew, fresh newcomers, no > 48-row step under the cap)" || echo "VIRHE in $nv place(s) -- see the lines above; a reading with VIRHE is not a reading") | phases done:$DONEPH"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $CAPSW = 1 ] || echo " | VIRHE XC48_CAPSW=0: NO cap switch posted, no cap check (a mock run only)") | lever CAP48 (RADIANCE_DECODE_ROW_CAP=$CAP at start, fn4 sched_row_cap in process) | one start: MAXSEQS $MS nvfp4, S5T pin | phases [$PHASES] | ~29 min (budget $BUDGET)"
IMGLITS="B|sched_row_cap
B|RADIANCE_DECODE_ROW_CAP
B|lever CAP48: decode row cap %d (RADIANCE_DECODE_ROW_CAP, fn4 sched_row_cap)
B|sched_row_cap takes 0 (off) or 1 .. 4096
B|steps_capped
B|drafts_trimmed
B|ahead_capped
B|steps_gt48
B|stage_full_pick, sched_row_cap, epoch_bump)
B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|kv: the pool backs %.0f%% of the worst case
B|draft_occ takes 0 (off) .. 64
B|RADIANCE_WEIGHT_SIDECAR
B|stage_full_min
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
A|lever BI stats: %lld segmented steps
A|steps with > 48 decode rows
A|latent %s (%lld bytes an entry; RADIANCE_DS41_LATENT)
A|RADIANCE_DS41_ENGDD
A|RADIANCE_DS41_PSK_CR
A|DSpark lever SD: the sampled full-vocabulary walk declared
K|RADIANCE_DS41_NVGATHER
K|dspark_walk_sample_step
S|mtp_couple_verify
X|sched_noa
X|ds41_psk_l20
X|dspark_pfctx
X|ds41_engdd
X|ds41_psk_cr"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"
test -s /opt/radiance/share/radiance/cap48-PROVENANCE.txt || echo "MISS cap48-PROVENANCE.txt"
test -s /opt/radiance/share/radiance/int9-PROVENANCE.txt || echo "MISS int9-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE does not carry what this item reads -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (lever CAP48's switch / env / start line / counters / range message / registry tail, kv4's latent / BI stats / step log / pool notes, stack5's S5 switches) + bin/rad-sidecar + cap48 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $CTR $PRB $NCMB $NCONE $SUM $DQ $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $CTR $PRB $NCMB $NCONE $SUM $DQ $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I $PRB selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PRB(${ST2:0:300})" ;; esac
ST3=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB(${ST3:0:300})" ;; esac
ST4=$(timeout 60 python3 -I -B $NCONE --mb $MBP --selftest 2>&1 | tail -1); case "$ST4" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCONE(${ST4:0:300})" ;; esac
ST5=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$ST5" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${ST5:0:300})" ;; esac
ST6=$(timeout 120 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$ST6" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${ST6:0:300})" ;; esac
REFS=$(timeout 60 python3 -I -B $SUM refs --log $L 2>&1 | tail -1)   # the cross-start references: information, never a stop (missing -> "-")
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv thermline idrec cmpx identw modes keeplog fn4get fn4set fn4code nudge armset gate1 dfree diskok envline drafted pyx pcreset snap capture now wmark startenv swstart guard16 phase pest idhas idf idall cstart report"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du seq; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in S D Z d z; do python3 -c 'import json,sys
d = json.loads(sys.argv[1]); a = sys.argv[2]; cap = int(sys.argv[3]); s5t = json.loads(sys.argv[4]); det = json.loads(sys.argv[5])
ok = d["gg_gemv"] == 2 and d["epoch_bump"] == 1 and d["ds41_psk_k"] == 8 and d["draft_occ_k"] == 32
ok = ok and d.get("sched_row_cap", "absent") == ((cap if a in "SDZ" else 0) if sys.argv[6] == "1" else "absent")
ok = ok and all(d[k] == v for k, v in s5t.items() if not (a in "DZdz" and k in det) and not (a in "Zz" and k == "draft_sample"))
ok = ok and (all(d[k] == v for k, v in det.items()) if a in "DZdz" else (d["draft_adaptive"] == 1 and d["draft_depth0"] == 1 and d["draft_occ"] == 6))
ok = ok and d["draft_sample"] == (0 if a in "Zz" else 85)
sys.exit(0 if ok else 1)' "$(pinj $a '{"epoch_bump": 1}')" $a $CAP "$S5T" "$DET" $CAPSW 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_DS41_LATENT:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$O/stats ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="cap48probe.py $(md5sum < $PRB | cut -c1-8) (copy of stack4/loadid.py $(md5sum < $LD | cut -c1-8) + T / seed / set), cap48sum.py $(md5sum < $SUM | cut -c1-8), ncmb.py $(md5sum < $NCMB | cut -c1-8) (kv4's $(md5sum < $HD/../kv4/ncmb.py 2>/dev/null | cut -c1-8)), ncone.py $(md5sum < $NCONE | cut -c1-8), stack5/s5ctr.py $(md5sum < $CTR | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; stack4/loadid.py $ST1 | cap48probe.py $ST2 | ncmb.py $ST3 | ncone.py $ST4 | s5ctr.py $ST5 | cap48sum.py $ST6 | $REFS | all $(echo $FNS | wc -w) functions resolve, tools on PATH, the arms' pins (S D Z d z), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_DS41_LATENT, /stats mount) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RK4/n16-a1.jsonl; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  case "$REFS" in "REFS OK"*) ;; *) xlog "note -- cross-start speed references not all in $L (${REFS:0:300}): the speed lines print '-' for them" ;; esac
  xlog "$ITEM: helpers md5: $HMD5 | references: $REFS"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RK4/n16-a1.jsonl; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in S D Z d z; do echo "+ arm $(NAMEOF $a): $(armj $a) | POST body $(pinj $a '{"epoch_bump": 1}')"; done
  for p in $PHASES; do IFS=: read -r pn pa pk <<< "$p"; echo "+ phase $pn: arm $(NAMEOF $pa), $pk (~$(pest $pk) min)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; : > $O/$TAG-wins.txt; }
stopfile || cstart
report
unset $SWITCHES
down
xlog "$ITEM: done in $(elapsed) min -- $O ($(dfree) MB free) | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
