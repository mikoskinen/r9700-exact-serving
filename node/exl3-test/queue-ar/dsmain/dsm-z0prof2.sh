#!/bin/bash
# dsm-z0prof2.sh (11.11.2026, queue-ar/dsmain, ds-main-port Z0 agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-hy-z0prof2.sh: THE PROFILED HALF OF hy-z0prof AGAIN. = dsm-z0prof.sh (md5 04b18610, copied; that file
# untouched) with three changes: (1) STARTS default "n5prof n0prof" (the stamped starts and their parse are hy-z0prof's,
# in the same out dir, read by the summary as they are); (2) modes(): under --profile-ops every pass runs eagerly, so no
# recorded pass prints "the acquire rule plays" -- hy-z0prof's n5prof stopped on "VIRHE rule-w is not what the recorded
# passes play" (03:18 UTC); here a profiled start checks RADIANCE_AQL_ACQ=rule-w in the container's env and no plain
# 'rule' line instead (a speed start keeps stack2's check); (3) a profiled start skips lib-29's warm-up check (a 16k
# prefill with synchronised ops took ~6 min at n5prof): one 4-token request, then the windows' own 15 s warm-up;
# (4) PREFILL / MIXED coverage (the lead's 03:2x correction: the stamped decode-step gap is only c4 +0.7 / c8 +0.9 ms while
# mb tok/s is -4.2 / -4.4 %, so about half the gap sits outside decode steps): each profiled start adds two PREFILL windows
# after its c4 / c8 windows (z0pf.py: 8 fresh prompts of mb's c4 / c8 size, ctx 1000 -> ~1170 tokens, max_tokens 1; pf1 one
# at a time = pure prefill steps, pf4 four at once = chunked / batched prefill), the NST 5 one under forced depth 0 with the
# lend engaged (as c4d0 / c8d0), with /stats by step class (prefill / mixed / decode) and the draft / lend / ctxskip counters
# at the window's ends -> per-op per rank at NST 5 vs NST 0 for a prefill chunk (the drafter's context pass over a new
# prompt shows as draft_* ops), TTFT per request; and z0steps.py reads hy-z0prof's stamp CSVs (card 0) BY STEP CLASS (decode
# / non-decode trunk passes = prefill / mixed / small passes = drafter refills and query passes, card idle before each class
# = host work around a step: admission, the lookup index build, the lend's state) per request and per pass, NST 5 vs NST 0,
# and classlines() logs the /stats split of every stamped window (decode vs prefill+mixed ms a step, per request, decode
# share, TTFT). The profile sync replaces rule-w in profiled starts and is logged as such (not a speed).
# ---- dsm-z0prof.sh's header:
# dsm-z0prof.sh (11.11.2026, queue-ar/dsmain, ds-main-port Z0 agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-hy-z0prof.sh. Log tag "RESULT xz0 ", out runs/xdsm/z0prof (XZ0_OUT).
# LEVER Z0 (depth-0 parity), STEP 1 = THE PROFILE: where a depth-0 step of the NST 5 stack2 server (D7 + back-off 3, hybrid 1,
# drafter_mx 1, walk 1, the small n5 set, VRAM levers, DL lend 1) spends the time a step of the NST 0 stack2 server does not.
# Gap (hxzzy-stack2, same image): NST 5 vs NST 0 c4 128.9 vs 138.5 tok/s (-6.9 %), c8 172.9 vs 180.3 (-4.1 %); c8 depth 0 ~1950
# cycles a phase. Stack2's own blocks already show the c4 gap is two things: block 0 held depth 0 (zero 1963, moves 8) at
# 137.1 tok/s, blocks 1-3 flipped (moves 24-28, zero 485-936) at 125.2-127.6 -- so this item measures both the step (forced
# depth 0) and the choice (natural windows, after a 1-user phase as in the blocks).
# FOUR STARTS of the ds41-int3 image (XZ0_IMAGE), EXL3 2.25, TP4, MML 524288, the stack2 env and serve line, copied from
# dsm-stack2.sh (md5 88808805, untouched) with stack2's item settings (HYB 1, drafter_mx 1, VRAM env turnover 0, gg_pf 1 +
# table, f32_pf 25, walk 1, small n5 set, VRAM levers all, lend 1 + back-off 3), each start's pin applied in process:
#   n5stamp  NST 5, RADIANCE_AQL_STAMP=1 (every played pass on card 0 timed per dispatch; ~8 % inflation, so stamped is
#            compared with stamped only): c4nat (stack2's pin, draft_probe 16 = the default, depth 0 re-armed: as a block's
#            first c4) -> a 1-user essay (stepprobe 512, as the blocks' c1 phases) -> c4nat2 -> c8nat -> c4d0 / c8d0 =
#            depth 0 FORCED (fn4 draft_depth0 2, the test hook: D = 0 at every cycle, draft_probe 4096 so no probe and the
#            lend engages after 32 cycles; warm 15 s so the lend has filled, vramstack5: 95 % in ~10 s).
#   n0stamp  NST 0 (stack2's N pin), RADIANCE_AQL_STAMP=1: c4, c8 (warm 15 s).
#   n5prof   NST 5, --profile-ops RADIANCE_PROFILE_EVERY=16 (synchronised ops: per rank, not a speed): c4d0, c8d0.
#   n0prof   NST 0, --profile-ops: c4, c8.
#   Every window: d0attrblocks.py window (read-only; mb.py phase, /stats + the draft counters at its ends with their
#   CLOCK_MONOTONIC ns, the followed log's line count under --profile-ops) -> ms a decode step, tok/s, tok a step, depth-0 /
#   probe share, D7 moves, rank-0 hit %, missed and host-stream a step, routed units in VRAM, passes a step; plus GET
#   /admin/fn4's lend / ctxskip / draft objects at its end (lent units, lend state, back-off).
#   Each start: restart (not gated: instrumented), warm-up + 1-user fastcheck (logged), stack2's modes / lever-line checks
#   and switch note, the VRAM lines of the start log (mover slots, static VRAM, KV size, arena / drafter lends).
# ANALYSIS (read-only helpers): d0attrprof.py stamps (per window: step period, trunk pass, other pass kinds a step, card idle,
# kernel families a pass; vs the NST 0 window of the same load), d0attrprof.py ops (per rank by class, rank 0's trunk / drafter
# ops), z0prof.py ar (per collective instance and rank: floor, above-floor wait, who arrives last), z0prof.py opscmp (depth 0
# vs NST 0 per class per rank), z0prof.py summary (the ranking: per load each candidate's ms a step from the stamps and the
# profile, /stats experts, VRAM units, probes + D7 choice from the natural windows) -> "RESULT xz0 rank ..." and
# "RESULT xz0 summary: ...". The stamp parse runs in the background during the profiled starts.
# ~34 min (evict ~1, n5stamp ~12, n0stamp ~7, n5prof ~8, n0prof ~7). NO cc_sim, no 2.9, no identity here (nothing changes
# output: switches the stack2 starts already proved IDENTICAL + the forced hook, exact by construction). No power / host / fan
# settings; production compose files read only (--project-directory).
# XZ0_IMAGE, XZ0_ITEM, XZ0_OUT, XZ0_STARTS (default "n5stamp n0stamp n5prof n0prof"), XZ0_WDUR / XZ0_WWARM (forced / NST 0
# windows, 55 / 15 s), XZ0_NDUR / XZ0_NWARM (natural windows, 60 / 10 s), XZ0_HYB, XZ0_DRAFTER_MX, XZ0_VRAMENV, XZ0_GGPF,
# XZ0_GGPF_TABLE, XZ0_F32PF, XZ0_WALK, XZ0_SMALL5, XZ0_VRAM, XZ0_LEND, XZ0_BORROW (stack2's item values by default).
# DRYRUN=1 bash dsm-z0prof.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of
#   its files: the int3 registry, depth 0 forced, the stamps, --profile-ops, every lever literal), EVERY HELPER (files present
#   and parsing, every shell function this body calls defined, tools on PATH, z0prof.py selftest), the sidecar, the yaml
#   (docker compose config --quiet); no server, no GPU, nothing written. A missing helper fails the dry run.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-stack2-tp4.yaml                      # stack2's compose file, used as it is (read only)
O=${XZ0_OUT:-$R/xdsm/z0prof}; export STATSDIR=$O/stats
export IMAGE=${XZ0_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l}
ITEM=${XZ0_ITEM:-57-p4zyzzzzz-ds-hy-z0prof2}; ARMS=stack; HYB=${XZ0_HYB:-1}; MATCH=8
DMX=${XZ0_DRAFTER_MX:-1}
case $DMX in 0|1) ;; *) echo "XZ0_DRAFTER_MX must be 0 or 1 (got '$DMX')"; exit 1 ;; esac
case $HYB in 0|1) ;; *) echo "XZ0_HYB must be 0 or 1 (got '$HYB')"; exit 1 ;; esac
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
DAB=$HD/d0attrblocks.py; DAP=$HD/d0attrprof.py; ZP=$HD/z0prof.py; DOP=$HD/ds_opprof_draft.py; ZPF=$HD/z0pf.py; ZS=$HD/z0steps.py
WDUR=${XZ0_WDUR:-55}; WWARM=${XZ0_WWARM:-15}; NDUR=${XZ0_NDUR:-60}; NWARM=${XZ0_NWARM:-10}
STARTS=${XZ0_STARTS:-n5prof n0prof}
for st in $STARTS; do case $st in n5stamp|n0stamp|n5prof|n0prof) ;; *) echo "XZ0_STARTS: unknown start '$st'"; exit 1 ;; esac; done
# ---------------------------------------------------------------- the stack2 env, pins and start checks (dsm-stack2.sh lines 128-235, XS2_ -> XZ0_, defaults = stack2's item)
# ---------------------------------------------------------------- the int3 levers (stack2)
GGPF=${XZ0_GGPF:-1}; F32PF=${XZ0_F32PF:-25}; WALK=${XZ0_WALK:-1}; SMALL5=${XZ0_SMALL5:-1}
GPTAB=${XZ0_GGPF_TABLE-q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422}
VRAMSEL=${XZ0_VRAM:-all}; LEND=${XZ0_LEND:-1}; BORROW=${XZ0_BORROW:-0}
SIDECAR=dspark-conf.f32                         # MODELDIR's confidence head (the release's), /models/dspark-conf.f32 in the container
case $GGPF in 0|1) ;; *) echo "XZ0_GGPF must be 0 or 1 (got '$GGPF')"; exit 1 ;; esac
case $F32PF in 0|1|2[0-5]) ;; *) echo "XZ0_F32PF must be 0, 1 or a form code 20..25 (got '$F32PF')"; exit 1 ;; esac
case $WALK in 0|1|2) ;; *) echo "XZ0_WALK must be 0, 1 or 2 (got '$WALK')"; exit 1 ;; esac
case $SMALL5 in 0|1) ;; *) echo "XZ0_SMALL5 must be 0 or 1 (got '$SMALL5')"; exit 1 ;; esac
case $LEND in 0|1) ;; *) echo "XZ0_LEND must be 0 or 1 (got '$LEND')"; exit 1 ;; esac
case $BORROW in 0|1) ;; *) echo "XZ0_BORROW must be 0 or 1 (got '$BORROW')"; exit 1 ;; esac
[ $BORROW = 1 ] && [ $LEND = 0 ] && { echo "XZ0_BORROW=1 needs XZ0_LEND=1 (the borrow prices lent steps)"; exit 1; }
[ $GGPF = 1 ] && [ -z "$GPTAB" ] && { echo "XZ0_GGPF=1 needs XZ0_GGPF_TABLE (empty)"; exit 1; }
case "$GPTAB" in *" "*) echo "XZ0_GGPF_TABLE takes no spaces"; exit 1 ;; esac
# THE VRAM TABLE (hxz-vramstack's start levers, one a line): name  NST 0 env  NST 5 env  NST 0 log literal  NST 5 log literal
#   (env NAME=VALUE[,..]; a log literal with ',' for a space the start log must carry when on, '-' none; a literal holds no comma)
VTAB='
arena   RADIANCE_DS41_ARENA_DECODE_ROWS=8  RADIANCE_DS41_ARENA_DECODE_ROWS=48  activation,arena:,8,tokens,use  activation,arena:,48,tokens,use
bf16    RADIANCE_DS41_ROUTER_BF16=1        RADIANCE_DS41_ROUTER_BF16=1         -                               -
emb     RADIANCE_DS41_EMB_RAM=1            RADIANCE_DS41_EMB_RAM=1             a,rank),ram                     a,rank),ram
shadow  RADIANCE_PLACE_SHADOW_SLOTS=16     RADIANCE_PLACE_SHADOW_SLOTS=16      -                               -
xpoff   RADIANCE_XPREFETCH=0               RADIANCE_XPREFETCH=0                expert,prefetch:,off            expert,prefetch:,off
'
VNAMES=$(echo "$VTAB" | awk 'NF && $1 !~ /^#/ {print $1}' | xargs)
[ "$VRAMSEL" = all ] && VRAMSEL=$VNAMES
for v in $VRAMSEL; do case " $VNAMES " in *" $v "*) ;; *) echo "XZ0_VRAM: no VRAM lever '$v' (have: $VNAMES, or all)"; exit 1 ;; esac; done
vf() { echo "$VTAB" | awk -v n="$1" -v k="$2" '$1 == n {print $k; exit}'; }   # vf LEVER FIELD
# levenv n0|n5: every start-time lever word of that start (exported by stackenv and written to the start's env file)
levenv() {
  local v w=""
  [ $GGPF = 1 ] && w="$w RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB"
  [ $F32PF != 0 ] && w="$w RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do w="$w $(vf $v $([ $1 = n0 ] && echo 2 || echo 3) | tr ',' ' ')"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && w="$w RADIANCE_DS41_DSPARK_WALK=$WALK"
    [ $SMALL5 = 1 ] && w="$w RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR"
    [ $LEND = 1 ] && w="$w RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  fi
  echo $w
}
# levlits n0|n5: the start-log literals the levers on at that start must leave ('|'-separated; ',' in a literal = a space)
levlits() {
  local v l x=""
  [ $GGPF = 1 ] && x="$x|gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1"
  [ $F32PF != 0 ] && x="$x|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do l=$(vf $v $([ $1 = n0 ] && echo 4 || echo 5)); [ "$l" = - ] || x="$x|$l"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && x="$x|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=$WALK,"
    [ $SMALL5 = 1 ] && x="$x|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
    [ $LEND = 1 ] && x="$x|draft,lend:,set,up"
  fi
  echo "${x#|}"
}
# the pins: hx-stack's + the int3 levers (NST 0: N = FIX0; NST 5: FIX + arm A / B)
P0=""; [ $GGPF = 1 ] && P0="$P0, \"gg_pf\": 1"; [ $F32PF != 0 ] && P0="$P0, \"f32_pf\": $F32PF"
P5=""; [ $WALK != 0 ] && P5="$P5, \"dspark_walk\": $WALK"
[ $SMALL5 = 1 ] && P5="$P5, \"ds41_argather\": 48, \"idx_split\": 1, \"ds41_qnfold\": 1, \"ds41_swfold\": 1, \"draft_ctxskip\": 1, \"dspark_conf\": 1, \"dspark_conf_rows\": 1"
[ $LEND = 1 ] && P5="$P5, \"draft_lend\": 1, \"draft_probe_backoff\": 3"
[ $BORROW = 1 ] && P5="$P5, \"draft_lend_borrow\": 1"
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"'}'
[ "$DMX" = 1 ] && FIX="${FIX%\}}, \"drafter_mx\": 1}"   # lever DK in the NST 5 pin (XZ0_DRAFTER_MX=1)
FIX="${FIX%\}}$P0$P5}"
ARM_A='{"draft_hybrid": '"$HYB"'}'
ARM_B='{"draft_hybrid": 1}'
FIX0="{\"gg_gemv\": 2, \"aql_acq\": 2, \"ds41_kvfold\": 1, \"ds41_ofold\": 1, \"ds41_onefill\": 1$P0}"   # the NST 0 start's pin (arm N = {})
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm's switches (+ extra, e.g. epoch_bump) as one POST body; arm N: FIX0 + extra
  local x=${2:-} base=$FIX; [ -n "$x" ] || x='{}'
  [ "$1" = N ] && { base=$FIX0; set -- '{}'; }
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$base" "$1" "$x"
}
DESC_N="NST 0 stack2 N: gg_gemv 2, rule-w, K1 folds c/d/f$([ $GGPF = 1 ] && echo ", gg_pf 1 (table)")$([ $F32PF != 0 ] && echo ", f32_pf $F32PF"), no drafter"
DESC_A="NST 5 stack2 A: gg_gemv 2, rule-w, K1 folds c/d/f$([ $GGPF = 1 ] && echo ", gg_pf 1 (table)")$([ $F32PF != 0 ] && echo ", f32_pf $F32PF"), DSpark NST 5 adaptive, depth 0, hybrid $HYB$([ $WALK != 0 ] && echo ", dspark_walk $WALK")$([ $SMALL5 = 1 ] && echo ", small n5 set (argather 48, idx_split, qnfold, swfold, ctxskip, conf + conf_rows 1)")$([ $LEND = 1 ] && echo ", lend + back-off 3")$([ $BORROW = 1 ] && echo " + borrow")"
DESC_B="NST 5 stack2 B: the same with hybrid 1 (match $MATCH)"
[ "$DMX" = 1 ] && DESC_A="$DESC_A, drafter_mx 1"
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
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH"
# the exact VRAM env at both starts (empty = none): each word NAME=VALUE, NAME one of $SWITCHES (unset first, passed by the yaml)
VRAMENV=${XZ0_VRAMENV-RADIANCE_PLACE_TURNOVER=0}
for kv in $VRAMENV; do
  case "$kv" in [A-Z]*=*) ;; *) echo "XZ0_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac
  case " $(echo $SWITCHES) " in *" ${kv%%=*} "*) ;; *) echo "XZ0_VRAMENV: ${kv%%=*} is not in the stack's switch list (unset at start, passed by the yaml)"; exit 1 ;; esac
done
[ -n "$VRAMENV" ] && { DESC_N="$DESC_N, VRAM env $VRAMENV"; DESC_A="$DESC_A, VRAM env $VRAMENV"; }
[ -n "$VRAMSEL" ] && { DESC_N="$DESC_N, VRAM levers [$VRAMSEL]"; DESC_A="$DESC_A, VRAM levers [$VRAMSEL]"; }
stackenv() {   # stackenv n0|n5: 105's t3 arm + gg_gemv 2 + rule-w + the three K1 folds declared (and on) at start + XZ0_VRAMENV
  # + the start's int3 lever words (levenv), every other switch unset; LEVENV = those words (the env file's content)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $VRAMENV; do export "$kv"; done
  LEVENV=$(levenv $1)
  for kv in $LEVENV; do export "$kv"; done
}
LEVENV=""
# ---------------------------------------------------------------- stack2's helpers (dsm-stack2.sh 236, 258-266, 288-348, 358-375, 381-410; tag xs2 -> xz0)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
VKNOBS=""
vrammodes() {   # vrammodes TAG: the VRAM env in the container's env, the mover's shadow / kept-free counts a rank, the expert
  # prefetch state -> VKNOBS (the modes line prints it); 1 when the env is missing, a rank has no mover line, or turnover 0
  # was asked and a rank still keeps slots free
  local MV XP CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [${VRAMENV:-none}] + lever env [$(echo $LEVENV | sed 's/RADIANCE_DS41_GG_PF_TABLE=[^ ]*/RADIANCE_DS41_GG_PF_TABLE=<table>/')]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { log "RESULT xz0 $1: VIRHE the VRAM / lever env did not reach the container:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { log "RESULT xz0 $1: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xz0 $1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; } ;;
  esac
  return 0
}
modes() {   # modes TAG n0|n5: the t3 line's notes + rule-w + the lever K1 note (all three declared and on) + the VRAM knobs + the int3
  # levers' declare lines (levlits) -> 0, else 1 (start stopped)
  local AR=4; [ $2 = n5 ] && [ $SMALL5 = 1 ] && AR=48
  [ "$DRY" = 1 ] && { echo "+ modes $1: int3 lever literals [$(levlits $2 | tr ',' ' ')]; ARGATHER note up to $AR rows; the env file's words in the container's env"; echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to $AR, no shared-expert-lane note, staging line, rule-w in every recorded pass (no plain rule), the lever K1 note (kvfold / ofold at > 0 layers, ones at > 0 MoE layers, ds41_kvfold=1 ds41_ofold=1 ds41_onefill=1); VRAM env [${VRAMENV:-none}] in the container's env, the mover's shadow / kept-free counts on 4 ranks$(case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*) echo " (each '0 more kept free')" ;; esac), expert prefetch state"; return 0; }
  local MLF SL AL XL KL IL NW NR ok=1 lit LL
  vrammodes $1 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xz0 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | interleave: ${IL:-no line} | VRAM: ${VKNOBS:-none}"
  log "RESULT xz0 $1 prefix resume fix (R1, ds41-resfix cee0ee3): $(docker logs $CN 2>&1 | grep -c "later publisher's bytes win") startup line(s) naming it (information); hits that took a later publisher's bytes so far: $(docker logs $CN 2>&1 | grep -c 'take a later') line(s)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xz0 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  if [ $AR = 48 ]; then
    case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) log "RESULT xz0 $1: VIRHE the ARGATHER note is not the 48-row one (RADIANCE_DS41_ARGATHER_MAX=48): '${AL}'"; ok=0 ;; esac
  else
    case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xz0 $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  fi
  [ -z "$XL" ] || { log "RESULT xz0 $1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xz0 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xz0 $1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xz0 $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  # rule-w from the start (RADIANCE_AQL_ACQ=rule-w): every recorded pass says so; plain `rule` must not appear
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  log "RESULT xz0 $1 acquire rule: $NW recorded-pass line(s) with RADIANCE_AQL_ACQ=rule-w, $NR with rule"
  if [ -n "${PROFEVERY:-}" ]; then   # z0prof2: --profile-ops records no pass; the env must carry rule-w, no plain rule line
    if docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep -qx 'RADIANCE_AQL_ACQ=rule-w' && [ "$NR" = 0 ]; then
      log "RESULT xz0 $1: rule-w replaced by the profile sync (not a speed): --profile-ops records no pass (core/runtime/ctx.cpp tape_on_ needs !profiling_), RADIANCE_AQL_ACQ=rule-w is in the container env, no plain rule line"
    else log "RESULT xz0 $1: VIRHE RADIANCE_AQL_ACQ=rule-w not in the container env (or a plain rule line) under --profile-ops"; ok=0; fi
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xz0 $1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  fi
  # the int3 levers' declare lines (levlits): each literal in the start log, else VIRHE
  LL=$(levlits $2); IFS='|' read -ra lit <<< "$LL"
  local found="" miss=""
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  log "RESULT xz0 $1 int3 lever lines: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever GPF|GPF-b|lever W1|context skip \(|confidence plane|ds41_qnfold=|ds41_swfold=|activation arena|expert prefetch: off|draft lend:" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2400)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xz0 $1 drafter lines"; return 0; }
  log "RESULT xz0 $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -10 | tr '\n' '|' | cut -c1-1600)"
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
swnote0() {   # swnote0 TAG: NST 0: GET at start (gg_gemv 2, aql_acq 2, the K1 three 1, no drafter); ds41_kvfold 2 and draft_depth0 1 refused; N's set applied
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_capable 0, epoch 0); POST ds41_kvfold 2 -> 400, draft_depth0 1 -> 400; POST N's set $(armset N '{"epoch_bump": 1}') + a 4-token request -> applied"; return 0; }
  local s0 s1 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_depth0": 1}')
  fn4set "$(armset N '{"epoch_bump": 1}')"; nudge; s1=$(fn4get)
  log "RESULT xz0 $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_depth0 1 -> HTTP $c3 | after N's set: $s1 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1 = json.loads(sys.argv[1]), json.loads(sys.argv[2]); n = json.loads(sys.argv[3]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and s0["draft_capable"] == 0 and s0["epoch"] == 0 and all(w.get(k, -1) != -1 for k in n)
ok = ok and all(s1["switches"].get(k) == v for k, v in n.items()) and not s1["pending"] and s1["epoch"] >= 1
sys.exit(0 if ok else 1)' "$s0" "$s1" "$(armset N)" 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
swnote() {   # swnote TAG: GET at start; ds41_kvfold 2 / draft_hybrid 2 refused; A's set (and B's in both mode) applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_adaptive 1, draft_depth0 0, draft_hybrid 0, draft_capable 1, epoch 0; every pinned switch served, not -1); POST ds41_kvfold 2 -> 400, draft_hybrid 2 -> 400; POST A's set $(armset "$ARM_A")$([ $ARMS = both ] && echo " then B's $(armset "$ARM_B")"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_hybrid": 2}')
  fn4set "$(armset "$ARM_A")"; nudge; s1=$(fn4get)
  if [ $ARMS = both ]; then fn4set "$(armset "$ARM_B")"; nudge; s2=$(fn4get); else s2=$s1; fi
  log "RESULT xz0 $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_hybrid 2 -> HTTP $c3 | after A's set: $s1$([ $ARMS = both ] && echo " | after B's set: $s2") | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, b = json.loads(sys.argv[4]), json.loads(sys.argv[5]); both = sys.argv[6] == "both"
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_hybrid"] == 0 and s0["draft_capable"] == 1 and s0["epoch"] == 0
ok = ok and all(w.get(k, -1) != -1 for k in a)   # every pinned switch served at start (stack2: the int3 levers too)
ok = ok and all(s1["switches"].get(k) == v for k, v in a.items()) and not s1["pending"] and s1["epoch"] >= 1
if both: ok = ok and all(s2["switches"].get(k) == v for k, v in b.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$ARM_A")" "$(armset "$ARM_B")" $ARMS 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
# ---------------------------------------------------------------- z0prof: windows, starts, analysis
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
NATX='{"draft_probe": 16, "epoch_bump": 1}'                          # stack2's natural pin: probes at the default 16
FORCEX='{"draft_depth0": 2, "draft_probe": 4096, "epoch_bump": 1}'   # depth 0 forced, no probes
winget() {   # winget FILE LABEL KEY1 KEY2: two fields of a window row (d0attrblocks.py window --out), "a:b"
  python3 -c 'import json,sys
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") == sys.argv[2]: print("%s:%s" % (d.get(sys.argv[3]), d.get(sys.argv[4])))' "$1" "$2" "$3" "$4" 2>/dev/null | tail -1
}
winlog() {   # winlog TAG FILE LABEL: one window's line (+ GET /admin/fn4's lend / ctxskip / draft objects at its end)
  [ "$DRY" = 1 ] && return 0
  local fj=$O/$3-fn4.json
  curl -s -m 10 $URL/admin/fn4 > $fj 2>/dev/null
  log "RESULT xz0 $1 window $3: $(python3 -c 'import json,sys
row = None
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") == sys.argv[2]: row = d
try: g = json.load(open(sys.argv[3]))
except Exception: g = {}
if row is None: print("no row"); sys.exit(0)
d = row; m, w = d.get("mb") or {}, d.get("win") or {}; z = w.get("draft") or {}; le = g.get("lend") or {}; cs = g.get("ctxskip") or {}
sw = d.get("applied") or {}
print("ok %s, switches %s | mb %s tok/s, %s ms a decode step, %s tok a step, p50 %s | in the window: %s decode steps (%s prefill/mixed), depth 0 %s probe %s moves %s (share %s / %s), %s ms a decode step (/stats), vram %s units, hit %s %%, missed %s / stream %s MB a step, passes a step %s | lend at the end: %s | ctxskip: %s | draft: %s" % (
    d.get("ok"), {k: sw.get(k) for k in ("draft_depth0", "draft_probe", "draft_probe_backoff", "draft_lend", "draft_hybrid", "draft_ctxskip", "gg_gemv") if k in sw},
    m.get("agg_dec_tps"), m.get("ms_decode_step"), m.get("tok_per_decode_step"), m.get("per_stream_dec_p50"), w.get("decode_steps"), w.get("prefill_mixed_steps"),
    z.get("zero_cycles"), z.get("probe_cycles"), z.get("zero_moves"), w.get("zero_share"), w.get("probe_share"), w.get("ms_decode_step"),
    w.get("vram_units_end"), w.get("hit_pct_rank0"), w.get("missed_step_rank0"), w.get("stream_MB_step"), w.get("passes_step"),
    json.dumps(le, separators=(",", ":"))[:600], json.dumps(cs, separators=(",", ":"))[:300], json.dumps(g.get("draft") or {}, separators=(",", ":"))[:300]))' $2 $3 $fj 2>&1 | tail -1 | cut -c1-2200)"
}
wrun() {   # wrun TAG LABEL PHASE DUR WARM SETJSON RESET(0|1) MARKCMD: one window (d0attrblocks.py window) into $O/TAG-win.jsonl
  local tag=$1 lab=$2 ph=$3 dur=$4 warm=$5 set=$6 rst=$7 mk=$8 W=$O/$1-win.jsonl
  stopfile && return 0
  run "timeout 900 python3 -I $DAB window --url $URL --mb-dir $D --label $lab --phase $ph --dur $dur --warm $warm ${set:+--set '$set'} $([ "$rst" = 1 ] && echo --reset) ${mk:+--mark-cmd '$mk'} --out $W > $O/$lab-win.txt 2>&1"
  winlog $tag $W $lab
}
essay1u() {   # essay1u TAG: one 1-user essay (stepprobe.py essay 512, greedy) between the natural c4 windows, as the blocks' c1 phases
  stopfile && return 0
  run "(cd $D && timeout 600 python3 stepprobe.py essay 512 > $O/$1-essay.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xz0 $1 1-user essay between the natural c4 windows: $(grep -m1 '^J ' $O/$1-essay.txt | cut -c3- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("%s tok/s, %s tok a step, acceptance %s" % (d.get("decode_tps"), d.get("tokens_per_step"), d.get("acceptance")))' 2>&1 | cut -c1-200)"
}
vramlines() {   # vramlines TAG: the start log's VRAM plan lines (mover slots / static, KV size, arena / drafter lends, allocations)
  [ "$DRY" = 1 ] && { echo "+ RESULT xz0 $1 VRAM plan lines (mover slab slots + static vram a rank, kv cache size, arena lends, drafter lend, device allocations)"; return 0; }
  log "RESULT xz0 $1 VRAM plan: $(docker logs $CN 2>&1 | grep -E '^I\[[0-3]\] mover: [0-9]+ slab slots|^I kv: the cache can grow|^I activation arena: lends|lever DL: |^I device allocations|^I\[0\] arena: |slots can be lent out of the drafter' | sed 's/^I *//' | cut -c1-230 | sort -u | tr '\n' '|' | cut -c1-2600)"
}
PARSE_PID=""
zstart() {   # zstart n5|n0 stamp|prof: one instrumented start and its windows
  local s=$1 kind=$2 tag=z0-$1-$2 ms P="" mk="" ok
  stackenv $s; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    n0) export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    n5) export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" ;;
  esac
  export AQLSTAMP= PROFEVERY= PROFSYNC= STATSDIR=$O/stats-$s-$kind
  case $kind in
    stamp) export AQLSTAMP=1 ;;
    prof)  export PROFEVERY=16 XARGS="$XARGS --profile-ops"; P=$O/prof-$s; mk="wc -l < $P/engine.log" ;;
  esac
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  mkd $STATSDIR
  run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete"
  log "RESULT xz0 $tag: start ($([ $s = n0 ] && echo "$DESC_N" || echo "$DESC_A"); $([ $kind = stamp ] && echo 'RADIANCE_AQL_STAMP=1, every played pass on card 0' || echo '--profile-ops RADIANCE_PROFILE_EVERY=16, synchronised ops'); NOT a speed) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  restart $tag || { keeplog $tag-fail; log "RESULT xz0 $tag: VIRHE not healthy -- no $kind reading"; down; return 1; }
  if [ $kind = prof ]; then nudge; [ "$DRY" = 1 ] || log "RESULT xz0 $tag: profiled start -- lib-29's warm-up check skipped (one 4-token request; each window warms 15 s)"
  else
    ms=$(check $tag 1)
    [ "$DRY" = 1 ] || log "RESULT xz0 $tag warm-up + 1-user fastcheck ($kind, instrumented, not gated): $ms ms/step (stack2 guards: NST 5 / NST 0 uninstrumented)"
  fi
  modes $tag $s || { log "RESULT xz0 $tag: VIRHE the modes are not stack2's -> start stopped"; down; return 1; }
  draftlines $tag
  vramlines $tag
  if [ "$DRY" != 1 ]; then
    ok=1
    if [ $s = n5 ]; then
      docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xz0 $tag: VIRHE no DSpark drafter line"; ok=0; }
      docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xz0 $tag: VIRHE no adaptive draft depth line"; ok=0; }
    else
      docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xz0 $tag: VIRHE a DSpark drafter line at NST 0"; ok=0; }
    fi
    curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null
    log "RESULT xz0 $tag experts at start: $(vramunits $O/$tag-stats0.json) | stack2 $([ $s = n0 ] && echo st-nst0 || echo st-stack): $(vramunits $R/xdsm/stack2/st-$([ $s = n0 ] && echo nst0 || echo stack)-stats0.json)"
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  if [ $s = n0 ]; then swnote0 $tag; else swnote $tag; fi \
    || { log "RESULT xz0 $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  if [ $kind = prof ]; then mkd $P; follow $P; fi
  if [ $s = n5 ]; then
    if [ $kind = stamp ]; then
      wrun $tag $tag-c4nat c4 $NDUR $NWARM "$(armset "$ARM_A" "$NATX")" 1 ""
      essay1u $tag
      wrun $tag $tag-c4nat2 c4 $NDUR $NWARM "$(armset "$ARM_A" '{"draft_probe": 16}')" 0 ""
      wrun $tag $tag-c8nat c8 $NDUR $NWARM "" 0 ""
    fi
    wrun $tag $tag-c4d0 c4 $WDUR $WWARM "$(armset "$ARM_A" "$FORCEX")" 0 "$mk"
    wrun $tag $tag-c8d0 c8 $WDUR $WWARM "$(armset "$ARM_A" '{"draft_depth0": 2, "draft_probe": 4096}')" 0 "$mk"
    if [ $kind = prof ]; then pfrun $tag $tag-pf1 1 "$(armset "$ARM_A" '{"draft_depth0": 2, "draft_probe": 4096}')" "$mk"
                              pfrun $tag $tag-pf4 4 "$(armset "$ARM_A" '{"draft_depth0": 2, "draft_probe": 4096}')" "$mk"; fi
  else
    wrun $tag $tag-c4 c4 $WDUR $WWARM "$(armset N '{"epoch_bump": 1}')" 0 "$mk"
    wrun $tag $tag-c8 c8 $WDUR $WWARM "$(armset N)" 0 "$mk"
    if [ $kind = prof ]; then pfrun $tag $tag-pf1 1 "$(armset N)" "$mk"; pfrun $tag $tag-pf4 4 "$(armset N)" "$mk"; fi
  fi
  run "sleep 2"
  if [ $kind = prof ]; then unfollow; else run "docker logs $CN > $O/$tag-engine-full.log 2>&1"; fi
  down   # the stamp writer appends as it goes
  log "RESULT xz0 $tag: done"
}
pfrun() {   # pfrun TAG LABEL CONC SETJSON MARKCMD: one prefill window (z0pf.py: 8 fresh ~1170-token prompts, max_tokens 1) -> $O/TAG-pf.jsonl
  local tag=$1 lab=$2 c=$3 set=$4 mk=$5 W=$O/$1-pf.jsonl
  stopfile && return 0
  run "timeout 900 python3 -I $ZPF --url $URL --label $lab --k 8 --ctx 1000 --conc $c ${set:+--set '$set'} ${mk:+--mark-cmd '$mk'} --out $W > $O/$lab-pf.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xz0 $tag prefill window $lab: $(python3 -c 'import json,sys
r = None
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") == sys.argv[2]: r = d
if r is None: print("no row"); sys.exit(0)
w = r.get("win") or {}
print("ok %s, %d prompts of ~%s tokens at %d at a time, cached %s | TTFT mean %s s (sorted %s) | steps %s (prefill %s, mixed %s, decode %s), %s ms a non-decode step, busy %s s of %s s | draft %s | lend %s | ctxskip %s | errors %s" % (
    r.get("ok"), r.get("k"), r.get("prompt_mean"), r.get("conc"), r.get("cached"), r.get("ttft_mean"), r.get("ttft_sorted"),
    w.get("steps"), w.get("prefill_steps"), w.get("mixed_steps"), w.get("decode_steps"), w.get("ms_nondecode_step"), w.get("busy_s"), w.get("secs"),
    json.dumps(w.get("draft") or {}, separators=(",", ":"))[:300], json.dumps(w.get("lend") or {}, separators=(",", ":"))[:500],
    json.dumps(w.get("ctxskip") or {}, separators=(",", ":"))[:300], r.get("errors")))' $W $lab 2>&1 | tail -1 | cut -c1-2000)"
}
classlines() {   # classlines: the /stats split of every stamped window of hy-z0prof by step class (read only)
  [ "$DRY" = 1 ] && { echo "+ RESULT xz0 step classes (/stats, stamped windows): per window decode steps x ms, prefill+mixed steps x ms, per request, decode share, TTFT"; return 0; }
  python3 -c 'import json,sys
for f in sys.argv[1:]:
    try: rows = [json.loads(l) for l in open(f) if l.strip()]
    except OSError: continue
    for d in rows:
        m, w = d.get("mb") or {}, d.get("win") or {}
        s, dd, ms, md = w.get("steps"), w.get("decode_steps"), w.get("ms_step"), w.get("ms_decode_step")
        if not (s and dd and ms and md): print("%s: no /stats split" % d.get("label")); continue
        nd = s - dd; tot = s * ms; dec = dd * md; non = tot - dec; req = m.get("requests") or 0
        print("%s: decode %d steps x %.2f ms = %.0f ms; prefill+mixed %d steps x %.0f ms = %.0f ms (%.1f %% of busy); per request (%d in the window) %.0f ms non-decode; decode tokens %s, mb %s tok/s, decode share %s, TTFT p50 %s s, prefill tokens %s" % (
            d.get("label"), dd, md, dec, nd, non / nd if nd else 0, non, 100 * non / tot if tot else 0, req, non / req if req else 0,
            m.get("dec_tok"), m.get("agg_dec_tps"), m.get("decode_share"), m.get("ttft_p50"), m.get("prefill_tok")))' $O/z0-n5-stamp-win.jsonl $O/z0-n0-stamp-win.jsonl 2>&1 \
    | while IFS= read -r l; do log "RESULT xz0 step classes (/stats, stamped) ${l:0:900}"; done
}
STEP_PID=""
stepparse() {   # z0steps.py on hy-z0prof's stamp CSVs (card 0) by step class, NST 5 windows and NST 0 windows, background (CPU)
  local C5=$O/stats-n5-stamp/stamps.csv.gz C0=$O/stats-n0-stamp/stamps.csv.gz W5=$O/z0-n5-stamp-win.jsonl W0=$O/z0-n0-stamp-win.jsonl x w a5="" a0="" r5="" r0="" n
  if [ "$DRY" = 1 ]; then echo "+ (background, nice) python3 -I $ZS $C5 --win c4nat:T0:T1 .. c8d0 --req ..; python3 -I $ZS $C0 --win c4 --win c8 --req .. -> $O/steps-n5.txt / steps-n0.txt"; return 0; fi
  [ -s $C5 ] && [ -s $C0 ] || { log "RESULT xz0: VIRHE stamp CSVs of hy-z0prof missing ($C5 / $C0) -- no step-class reading"; return 1; }
  for x in c4nat c4nat2 c8nat c4d0 c8d0; do w=$(winget $W5 z0-n5-stamp-$x t0 t1); n=$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d.get("label") == sys.argv[2]: print((d.get("mb") or {}).get("requests") or 0)' $W5 z0-n5-stamp-$x 2>/dev/null | tail -1)
    [ -n "$w" ] && [ "$w" != "None:None" ] && a5="$a5 --win n5$x:$w" && r5="$r5 --req n5$x:${n:-0}"; done
  for x in c4 c8; do w=$(winget $W0 z0-n0-stamp-$x t0 t1); n=$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d.get("label") == sys.argv[2]: print((d.get("mb") or {}).get("requests") or 0)' $W0 z0-n0-stamp-$x 2>/dev/null | tail -1)
    [ -n "$w" ] && [ "$w" != "None:None" ] && a0="$a0 --win n0$x:$w" && r0="$r0 --req n0$x:${n:-0}"; done
  ( nice -n 10 timeout 3000 python3 -I $ZS $C5 $a5 $r5 --json $O/steps-n5.json --label n5 > $O/steps-n5.txt 2>&1 &
    nice -n 10 timeout 3000 python3 -I $ZS $C0 $a0 $r0 --json $O/steps-n0.json --label n0 > $O/steps-n0.txt 2>&1 & wait ) &
  STEP_PID=$!
  log "RESULT xz0: step-class stamp reading in the background (pid $STEP_PID): n5 [$a5] n0 [$a0]"
}
stampparse() {   # after both stamped starts: d0attrprof.py stamps per load (depth-0 / natural windows vs the NST 0 window), background
  local C5=$O/stats-n5-stamp/stamps.csv C0=$O/stats-n0-stamp/stamps.csv W5=$O/z0-n5-stamp-win.jsonl W0=$O/z0-n0-stamp-win.jsonl ph a w x r
  if [ "$DRY" = 1 ]; then echo "+ (background, nice) for c4 / c8: python3 -I $DAP stamps $C5 --win <ph>d0:T0:T1 --win <ph>nat:T0:T1 [--win c4nat2] --ref n0<ph>:$C0:T0:T1 --json $O/stamps-<ph>.json; then gzip both CSVs"; return 0; fi
  [ -s $C5 ] && [ -s $C0 ] || { log "RESULT xz0: VIRHE a stamps CSV is missing ($(ls -la $C5 $C0 2>&1 | tr '\n' ' ' | cut -c1-300))"; return 1; }
  local cmds=""
  for ph in c4 c8; do
    a=""
    for x in ${ph}d0 ${ph}nat ${ph}nat2; do w=$(winget $W5 z0-n5-stamp-$x t0 t1); [ -n "$w" ] && [ "$w" != "None:None" ] && a="$a --win $x:$w"; done
    r=$(winget $W0 z0-n0-stamp-$ph t0 t1)
    [ -n "$a" ] && [ -n "$r" ] && [ "$r" != "None:None" ] || { log "RESULT xz0: VIRHE no $ph stamp windows (n5 '$a' n0 '$r')"; continue; }
    cmds="$cmds nice -n 10 timeout 3000 python3 -I $DAP stamps $C5 $a --ref n0$ph:$C0:$r --json $O/stamps-$ph.json --label $ph > $O/stamps-$ph.txt 2>&1 &"
  done
  [ -n "$cmds" ] || return 1
  ( eval "$cmds wait"; nice -n 10 gzip -f $C5 $C0 ) &
  PARSE_PID=$!
  log "RESULT xz0: stamps parsed in the background (pid $PARSE_PID)"
}
profparse() {   # after both profiled starts: d0attrprof.py ops per start, z0prof.py ar per start, z0prof.py opscmp per load
  local s lab wl x w a
  for s in n5 n0; do
    if [ "$DRY" = 1 ]; then echo "+ python3 -I $DAP ops $O/prof-$s/engine.log --win <win>:L0:L1 ... --json $O/ops-$s.json; python3 -I $ZP ar $O/prof-$s/engine.log --win ... --json $O/ar-$s.json; gzip the log"; continue; fi
    [ -s $O/prof-$s/engine.log ] || { log "RESULT xz0: VIRHE no profiled log $O/prof-$s/engine.log"; continue; }
    wl=""
    for x in $([ $s = n5 ] && echo c4d0 c8d0 || echo c4 c8); do
      w=$(winget $O/z0-$s-prof-win.jsonl z0-$s-prof-$x mark0 mark1)
      [ -n "$w" ] && [ "$w" != "None:None" ] && wl="$wl --win $x:$w"
    done
    for x in pf1 pf4; do   # z0prof2: the prefill windows
      w=$(winget $O/z0-$s-prof-pf.jsonl z0-$s-prof-$x mark0 mark1)
      [ -n "$w" ] && [ "$w" != "None:None" ] && wl="$wl --win $x:$w"
    done
    [ -n "$wl" ] || { log "RESULT xz0: VIRHE no profile windows for $s"; continue; }
    timeout 1800 python3 -I $DAP ops $O/prof-$s/engine.log $wl --json $O/ops-$s.json --label $s > $O/ops-$s.txt 2>&1
    timeout 1800 python3 -I $ZP ar $O/prof-$s/engine.log $wl --json $O/ar-$s.json --label $s > $O/ar-$s.txt 2>&1
    while IFS= read -r l; do log "RESULT xz0 prof $s ${l:0:2400}"; done < <(grep -hE '^(D0AP|Z0) ' $O/ops-$s.txt $O/ar-$s.txt | grep -v '^D0AP {' | sed -E 's/^(D0AP|Z0) //')
    nice -n 10 gzip -f $O/prof-$s/engine.log
  done
  for x in c4 c8 pf1 pf4; do
    case $x in c4|c8) a=${x}d0 ;; *) a=$x ;; esac
    run "timeout 600 python3 -I $ZP opscmp --a $O/ops-n5.json:$a --b $O/ops-n0.json:$x --json $O/opscmp-$x.json --label $x > $O/opscmp-$x.txt 2>&1"
    [ "$DRY" = 1 ] || while IFS= read -r l; do log "RESULT xz0 depth 0 vs NST 0 PROFILED ${l:0:2600}"; done < <(grep '^Z0 ' $O/opscmp-$x.txt | sed 's/^Z0 //')
  done
}
zsummary() {
  if [ -n "$PARSE_PID" ]; then wait $PARSE_PID 2>/dev/null; PARSE_PID=""; fi
  if [ -n "$STEP_PID" ]; then wait $STEP_PID 2>/dev/null; STEP_PID=""; fi
  classlines
  if [ "$DRY" != 1 ]; then
    for x in n5 n0; do [ -s $O/steps-$x.txt ] && while IFS= read -r l; do log "RESULT xz0 step classes (stamps, card 0) ${l:0:1800}"; done < <(grep '^Z0S ' $O/steps-$x.txt | sed 's/^Z0S //'); done
  fi
  local ph x
  if [ "$DRY" != 1 ]; then
    for ph in c4 c8; do
      [ -s $O/stamps-$ph.txt ] && while IFS= read -r l; do log "RESULT xz0 stamps ${l:0:2400}"; done < <(grep '^D0AP ' $O/stamps-$ph.txt | grep -v '^D0AP {' | sed 's/^D0AP //')
    done
  fi
  run "timeout 600 python3 -I $ZP summary --wins-n5 $O/z0-n5-stamp-win.jsonl --wins-n0 $O/z0-n0-stamp-win.jsonl --stamps-c4 $O/stamps-c4.json --stamps-c8 $O/stamps-c8.json --opscmp-c4 $O/opscmp-c4.json --opscmp-c8 $O/opscmp-c8.json --ar-n5 $O/ar-n5.json --ar-n0 $O/ar-n0.json --json $O/summary.json --label sum > $O/summary.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xz0 ${l:0:2600}"; done < <(grep '^Z0 sum summary ' $O/summary.txt | sed 's/^Z0 sum summary //')
  log "RESULT xz0 summary: $(grep '^Z0 sum summary ' $O/summary.txt | grep -E ' rank [1-5]: | \(stamped, /stats\)' | sed 's/^Z0 sum summary //' | cut -c1-400 | tr '\n' '#' | sed 's/#$//; s/#/ || /g' | cut -c1-5000)"
}
# ---------------------------------------------------------------- main
log "RESULT xz0 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] | windows: forced / NST 0 ${WDUR} s (warm $WWARM), natural ${NDUR} s (warm $NWARM) | NST 5 pin $(armset "$ARM_A") | NST 0 pin $(armset N) | levers: gg_pf $GGPF, f32_pf $F32PF, walk $WALK, small n5 $SMALL5, VRAM [${VRAMSEL:-none}], lend $LEND, VRAM env [${VRAMENV:-none}] | ref hxzzy-stack2 (NST 5 c4 128.9 / c8 172.9, NST 0 c4 138.5 / c8 180.3 tok/s)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, epoch_bump
B|ds41_onefill, draft_depth0, draft_probe, draft_hybrid,
B|draft depth 0: a step drafts nothing
B|takes 1 (on), 2 (forced, a test hook) or 0 (off)
B|draft hybrid: a sequence whose prompt lookup
B|RADIANCE_AQL_STAMP
B|RADIANCE_PROFILE_EVERY
B|--profile-ops
A|RADIANCE_DS41_KVFOLD
K|rmsnorm_rope_roundtrip_kv_store
K|r4d_gg_mr_pin"
[ "$DMX" = 1 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_DRAFTER_MX"
[ $GGPF = 1 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_GG_PF_TABLE"
[ $F32PF != 0 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_F32_PF"
[ $WALK != 0 ] && IMGLITS="$IMGLITS
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step"
[ $SMALL5 = 1 ] && IMGLITS="$IMGLITS
A|RADIANCE_DS41_ARGATHER_MAX
A|RADIANCE_DS41_QNFOLD
A|RADIANCE_DS41_SWFOLD
A|RADIANCE_DS41_DSPARK_CONF
B|RADIANCE_DRAFT_CTXSKIP
B|a confidence plane (the dspark_conf gate"
[ $LEND = 1 ] && IMGLITS="$IMGLITS
X|RADIANCE_DRAFT_LEND
X|RADIANCE_DRAFT_LEND_BORROW"
for v in $VRAMSEL; do for w in $(vf $v 2 | tr ',' ' ') $(vf $v 3 | tr ',' ' '); do IMGLITS="$IMGLITS
X|${w%%=*}"; done; done
IMGLITS=$(echo "$IMGLITS" | awk '!seen[$0]++')
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xz0: VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) log "RESULT xz0: VIRHE $IMAGE is not the int3 stack image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (int3 registry, depth 0 + forced hook, stamps, --profile-ops, the levers) -- OK"
# EVERY HELPER RESOLVES (DRYRUN and real): files present, python parses, every shell function called here defined, tools on
# PATH, z0prof.py's selftest, the sidecar's sha256, the yaml accepted by docker compose config. Any miss: VIRHE, exit 1.
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $DAB $DAP $ZP $DOP $ZPF $ZS $D/stepprobe.py $D/mb.py $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $DAB $DAP $ZP $DOP $ZPF $ZS $D/stepprobe.py $D/mb.py $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
for fn in log run mkd restart check down evict stopfile best_vars desc serve_args keeplog follow unfollow vrammodes modes draftlines \
          vramunits fn4get fn4set nudge fn4code swnote0 swnote armset stackenv levenv levlits vf winget winlog wrun essay1u vramlines \
          zstart stampparse profparse zsummary pfrun classlines stepparse; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee gzip nice find wc; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
ZST=$(cd $HD && timeout 120 python3 -I $ZP selftest 2>&1 | tail -1)
case "$ZST" in "Z0 selftest OK"*) ;; *) HMISS="$HMISS selftest:z0prof.py($ZST)" ;; esac
ZST2=$(cd $HD && timeout 120 python3 -I $ZPF selftest 2>&1 | tail -1); ZST3=$(cd $HD && timeout 120 python3 -I $ZS selftest 2>&1 | tail -1)
case "$ZST2" in "Z0PF selftest OK"*) ;; *) HMISS="$HMISS selftest:z0pf.py($ZST2)" ;; esac
case "$ZST3" in "Z0S selftest OK"*) ;; *) HMISS="$HMISS selftest:z0steps.py($ZST3)" ;; esac
for f in $O/stats-n5-stamp/stamps.csv.gz $O/stats-n0-stamp/stamps.csv.gz $O/z0-n5-stamp-win.jsonl $O/z0-n0-stamp-win.jsonl $O/stamps-c4.json $O/stamps-c8.json; do
  [ -s $f ] || HMISS="$HMISS input:$f(hy-z0prof's stamped half)"; done
ZST="$ZST; $ZST2; $ZST3"
if [ $SMALL5 = 1 ]; then
  if [ -s $MODELDIR/$SIDECAR ] && [ -s $MODELDIR/$SIDECAR.sha256 ]; then
    [ "$(sha256sum $MODELDIR/$SIDECAR | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(sha256 differs from .sha256)"
  else HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(+.sha256)"; fi
fi
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file \${XS2_ENVFILE})"
for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_AQL_STAMP RADIANCE_PROFILE_EVERY; do grep -q "^ *$v:" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(does not pass $v)"; done
for kv in $VRAMENV; do grep -q "^ *${kv%%=*}:" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(does not pass ${kv%%=*}, XZ0_VRAMENV)"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { log "RESULT xz0: VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, every function defined, tools on PATH, $ZST, sidecar $SIDECAR sha256 OK, yaml $(basename $Y) accepted by docker compose config -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xz0: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xz0: VIRHE $M225 missing or not verified"; exit 1; }
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $R/xdsm/stack2/st-stack-stats0.json $R/xdsm/stack2/st-nst0-stats0.json; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(a radiance container running now is information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
stepparse
evict
for st in $STARTS; do
  stopfile && break
  case $st in
    n5stamp) zstart n5 stamp ;;
    n0stamp) zstart n0 stamp; stampparse ;;
    n5prof)  zstart n5 prof ;;
    n0prof)  zstart n0 prof; profparse ;;
  esac
done
zsummary
unset $SWITCHES
export AQLSTAMP= PROFEVERY=
down
log "RESULT xz0 $ITEM: done -- $O"
