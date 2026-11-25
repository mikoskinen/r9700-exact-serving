#!/bin/bash
# dsm-dst.sh (11.11.2026, queue-ar/dsmain/dst, ds-main-port DST agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r04z-dst.sh. Log tag "RESULT xdst ", out runs/xdsm/dst (XDS_OUT). = dsm-psk.sh 841ce920 (r00z's body; the
# original not edited) with its start, pin, checks and loaded probe kept, and its arms replaced by the DST grid.
# LEVER DST (decode stall during prefill; coordinator 11.11.): c4 decode streams keep 26.6 tok/s (vs ~145-150) while one cold 64k
# prompt prefills beside them (r00z A). r00z's "max TPOT 6660 ms" was NOT a stall of a running stream: it was a c4 stream's
# request turnover (the gap from one 512-token answer's last token to the next request's first = that request's TTFT, its
# ~1000-token prompt prefilled beside the 64k chunks; engine ttft 6.65 s), counted by pskld.py's all-gap TPOT. The running
# streams' stall is the mixed step itself (TPOT p90 752-892 ms = every (interleave + 1)-th gap).
# SWITCHES (ds41-dst, core fn4, read at every step planned, applied at a step release): sched_interleave (--prefill-interleave's
# count: decode-only steps after a chunk beside decoding sequences; 8 today), sched_mixed_tok (the budget of a step that is
# not a lone prefill's span step; 2112 today -> 2048-row mixed chunks; 1088 / 576 -> 1024 / 512-row chunks on their grid
# under sched_bi_chunk 3 / 4), with sched_bi_chunk (lever BI class (ii), the KEPT option from r04, XDS_BIC) and ds41_psk.
# ONE NST 5 START = r00z's (the z0occ2 start + RADIANCE_DS41_PSK=0) + RADIANCE_LOG_STEPS=1 (a line a step: the probe's chunk
# boundaries per cell). Pin = r00z's (ds41_bi 2, draft_sample XDS_DS, Z0 switches 0, draft_occ_k 32) + sched_bi_chunk XDS_BIC.
# BYPASS SECTION (coordinator, the Claude Code case: a new turn = a prefix hit + a short uncached tail): sched_short_bypass 0 /
# XDS_BYPASS (1280 2112) at (8, 2112) and the best interleave at 2112 (block 1's most c4 tok/s with long-prompt TTFT <= 1.5 x
# base; XDS_BYPIL), PSK 0 / on, each beside c4 + a FIXED newcomer set (XDS_NEWC 4 deterministic ~1000-token prompts, 64 greedy
# tokens, sent together XDS_NEWC_AT 15 s after the long prompt); half A after block 1, half B (its mirror: ABBA) after block 2.
# Readings apart: the fixed newcomers' TTFT and the c4 turnovers' TTFT p50 / p90 / max, TPOT in a request p50 / p99 / max,
# the long prompt's TTFT, c4 tok/s; identity of each fixed newcomer's answer vs the same prompt alone (newc-alone, PSK 0 / on).
# The bypass cells also set sched_bypass_total XDS_BYPTOT (4096: piece by piece), and the last fixed newcomer is XDS_NEWC_LONG
# (3) x ~1000 tokens (a fresh ~3000-token prompt = alone's two interval pieces: two held steps at N 2112, an ordinary
# admission at N 1280).
# CELLS: interleave (XDS_ILS: 8 16 32) x mixed budget (XDS_MTS: 2112 1088 576) x PSK (0 / XDS_PSKV) in process, less XDS_SKIP
# (16:576 32:576, the budget axis's dearest corners, to stay under ~2.5 h); each = c4
# decode streams (mb closed loop, ctx ~1000, 512 tokens) + ONE cold ~64k deepq prompt (dstld.py --load, no extra prefill), a
# prefix-cache reset before each. Order: block 1 = base (8, 2112), the interleave axis at 2112, the budget axis at 8, the
# corners, base again; block 2 = block 1 reversed (each (IL, MT) cell twice, symmetric about the run's middle; within a
# cell PSK 0 / on in block 1, on / 0 in block 2 = ABBA on PSK). Block 2 stops at XDS_BUDGET_MIN minutes from the item's
# start (the cells it skipped are logged).
# GUARD (coordinator): every loaded run checks that its levers acted (dstguard.py over fn4 GET / /stats before and after it
# and the probe's chunk list): sched_short_bypass > 0 -> dst.bypassed grew; the kept sched_bi_chunk 3 / 4 -> every probe
# chunk end is alone's piece end or a grid point >= 128 rows a side, and a grid cut ran beside decode rows; sched_mixed_tok
# < 2112 -> no MIXED chunk over it; the interleave -> the steps between chunks; ds41_psk >= 1 -> /stats "psk".latched,
# planned_steps and layers_skipped grew and plan_mismatch stayed (XDS_PSK_STATS / XDS_PSK_ZERO). A failure is a VIRHE and the run gives no speed reading (summary: INVALID lines).
# READINGS per cell, over the probe's prefill window [sent, first token]: c4 aggregate tok/s, TPOT inside a request p50 / p90
# / p99 / max (dstld tpot_in_*) and pskld's all-gap tpot_pf_* (r00z-comparable), the c4 requests sent in the window (their
# TTFT p50 / max: the turnovers), the mixed-step ms and decode-step ms (/metrics deltas), the probe's TTFT, its chunk list
# (step log: count, sizes, ends off the cell's grid, MIXED count). IDENTITY: every loaded answer vs the alone answer of the
# same PSK value, and vs the base (alone, PSK 0, sched_bi_chunk 0); the alone answers vs each other and vs r00z's 64k A
# answer (cross-start).
# ~115-150 min (estimate from the sched_test sim fitted to r00z: start ~5, checks + smoke ~4, alone x3 + newcomer sets ~5, block 1
# ~30, bypass half A ~18, block 2 ~30 (stopped at XDS_BUDGET_MIN 150 - XDS_BYP_RESERVE_MIN 20 from the item start), today's
# cell x2 ~3, bypass half B ~18).
# NO cc_sim, no 2.9, no power / host / fan settings; production compose files read only. Exact by design: the interleave moves
# only which step carries a chunk; the budget moves chunk boundaries, kept on the grid by sched_bi_chunk 3 / 4 (class (ii)
# hypothesis: pieces >= 128 rows are byte-neutral) -- measured here per cell; a difference is reported, never enabled.
# Knobs: XDS_IMAGE (required), XDS_ITEM, XDS_OUT, XDS_BIC (4), XDS_PSKV (2: design 2 anchors; the guard reads design 2's counters), XDS_PSKK (4: ds41_psk_k), XDS_ILS ("8 16
#   32"), XDS_MTS ("2112 1088 576"), XDS_BLOCKS (2), XDS_BUDGET_MIN (150), XDS_LDEPTH (64000), XDS_DS (1), XDS_SIDECAR,
#   XDS_SIDESHA, XDS_R00Z (r00z's out dir, for the cross-start 64k reference), XDS_TODAY (1: the base cell under
#   sched_bi_chunk 0 = today's chunking first in block 1 and last in block 2, PSK 0), XDS_SKIP, XDS_BYPASS, XDS_BYPIL (auto),
#   XDS_NEWC (4), XDS_NEWC_AT (15), XDS_NEWC_TOK (64), XDS_BYP_RESERVE_MIN (20), XDS_NEWC_LONG (3), XDS_BYPTOT (4096),
#   XDS_PSK_STATS ("psk.latched psk.planned_steps psk.layers_skipped": dotted /stats paths that must grow in a PSK cell),
#   XDS_PSK_ZERO ("psk.plan_mismatch": paths that must not move) -- the live /stats "psk" counters of ds41-int7-pskglue
#   0391a6e (design 2; XDS_PSKV 2 / 3).
# DRYRUN=1 bash dsm-dst.sh : prints every command; checks the image literals (int5 + PSK + BI chunk + DST), every helper (files,
# parse, dstld.py selftest, deepq.py = dstld's request), the sidecars, the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XDS_OUT:-$R/xdsm/dst}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS3=$R/xdsm/stack3; RPSK=${XDS_R00Z:-$R/xdsm/psk}
export IMAGE=${XDS_IMAGE:?set XDS_IMAGE}
ITEM=${XDS_ITEM:-57-p4zyzzzzz-ds-r04z-dst}
BIC=${XDS_BIC:-4}; PSKV=${XDS_PSKV:-2}; ILS=${XDS_ILS:-8 16 32}; MTS=${XDS_MTS:-2112 1088 576}; BLOCKS=${XDS_BLOCKS:-2}
BUDGET=${XDS_BUDGET_MIN:-150}; LDEPTH=${XDS_LDEPTH:-64000}; TODAY=${XDS_TODAY:-1}
SKIP=${XDS_SKIP-16:576 32:576}            # grid cells left out (the budget axis's dearest corners, to stay under ~2.5 h)
BYPS=${XDS_BYPASS-1280 2112}; BYPIL=${XDS_BYPIL:-auto}   # sched_short_bypass values (0 always) and the second cell's interleave
NEWC=${XDS_NEWC:-4}; NEWC_AT=${XDS_NEWC_AT:-15}; NEWC_TOK=${XDS_NEWC_TOK:-64}; BYPRES=${XDS_BYP_RESERVE_MIN:-20}
NEWC_LONG=${XDS_NEWC_LONG:-3}; BYPTOT=${XDS_BYPTOT:-4096}   # the last fixed newcomer 3 x ~1000 tokens; sched_bypass_total
PSKPATHS=${XDS_PSK_STATS:-psk.latched psk.planned_steps psk.layers_skipped}   # /stats counters a PSK cell must grow (dstguard.py)
PSKZERO=${XDS_PSK_ZERO:-psk.plan_mismatch}                                      # ... and must not move
DS=${XDS_DS:-1}; OCCK=32; NTOK=512
SIDE=${XDS_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XDS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $BIC in 0|3|4) ;; *) echo "XDS_BIC takes 0, 3 or 4 (sched_bi_chunk; 1 / 2 ignore the mixed budget by design)"; exit 1 ;; esac
case $PSKV in ''|*[!0-9]*) echo "XDS_PSKV must be a ds41_psk value (got '$PSKV')"; exit 1 ;; esac
case $BLOCKS in 1|2) ;; *) echo "XDS_BLOCKS takes 1 or 2"; exit 1 ;; esac
for v in $BYPS $NEWC $NEWC_TOK $BYPRES $NEWC_LONG $BYPTOT; do case $v in ''|*[!0-9]*) echo "XDS_BYPASS / XDS_NEWC / XDS_NEWC_TOK / XDS_BYP_RESERVE_MIN must be numbers (got '$v')"; exit 1 ;; esac; done
for v in $BYPS; do [ $((v % 16)) = 0 ] && [ $v -le 4160 ] && [ $v -gt 0 ] || { echo "XDS_BYPASS: $v not in 16 .. 4160 / 16"; exit 1; }; done
case $BYPIL in auto) ;; ''|*[!0-9]*) echo "XDS_BYPIL takes auto or an interleave"; exit 1 ;; esac
for v in $ILS; do case $v in ''|*[!0-9]*) echo "XDS_ILS: numbers 0 .. 64 (got '$v')"; exit 1 ;; esac; [ $v -le 64 ] || { echo "XDS_ILS: $v > 64"; exit 1; }; done
for v in $MTS $LDEPTH $BUDGET; do case $v in ''|*[!0-9]*) echo "XDS_MTS / XDS_LDEPTH / XDS_BUDGET_MIN must be numbers (got '$v')"; exit 1 ;; esac; done
for v in $MTS; do [ $v -ge 128 ] && [ $v -le 2112 ] && [ $((v % 16)) = 0 ] || { echo "XDS_MTS: $v not in 128 .. 2112 / 16"; exit 1; }; done
case $DS in ''|*[!0-9]*) echo "XDS_DS must be a draft_sample value 0..200 (got '$DS')"; exit 1 ;; esac
IL0=$(echo $ILS | awk '{print $1}'); MT0=$(echo $MTS | awk '{print $1}')   # the base cell: the first of each list (8, 2112)
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DQ=$QA/deepq.py; PL=$HD/dstld.py; GD=$HD/dstguard.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T_ITEM=$(date +%s)
xlog() { log "RESULT xdst $*"; }
# ---------------------------------------------------------------- r00z's start (the z0occ2 item's + RADIANCE_DS41_PSK=0) + RADIANCE_LOG_STEPS=1
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
STEPENV="RADIANCE_LOG_STEPS=1"
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK|chunk,geometry:,a,prefill,alone,in,its,step,spans,up,to,2,intervals,from,its,second,interval,on;,every,other,step,is,planned,at,2112,tokens"
LITS="$LITS|scheduler:,--prefill-interleave,8:,after,a,prefill,chunk|scheduler:,--busy-prefill-chunk,auto:"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"ds41_bi": 2, "draft_sample": int(sys.argv[2]), "draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 0, "draft_occ_k": int(sys.argv[3]), "sched_bi_chunk": int(sys.argv[4]), "sched_short_bypass": 0, "sched_bypass_total": 0}); print(json.dumps(d))' "$FIX5" "$DS" "$OCCK" "$BIC") || { echo "pin JSON"; exit 1; }
# ds41_psk_k (the scheduler's anchor stride under ds41_psk 2 / 3) pinned in every cell: XDS_PSKK (coordinator 11.11.: k 4; added
# by the stack3 builder at the coordinator's request, the DST agent informed)
PSKK=${XDS_PSKK:-4}
case $PSKK in ''|*[!0-9]*) echo "XDS_PSKK must be ds41_psk_k, 1 .. 4096 (got '$PSKK')"; exit 1 ;; esac
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["ds41_psk_k"]=int(sys.argv[2]); print(json.dumps(d))' "$FIX" "$PSKK") || { echo "pin JSON (ds41_psk_k)"; exit 1; }
cellj() {   # cellj IL MT P [BIC [BYPASS [TOTAL]]]: the cell's switches (sched_mixed_tok 0 = the shared budget 2112; BIC '' = the kept one)
  python3 -c 'import json,sys; il, mt, p = (int(x) for x in sys.argv[1:4]); d = {"sched_interleave": il, "sched_mixed_tok": 0 if mt >= 2112 else mt, "ds41_psk": p}
if len(sys.argv) > 4 and sys.argv[4]: d["sched_bi_chunk"] = int(sys.argv[4])
if len(sys.argv) > 5 and sys.argv[5]: d["sched_short_bypass"] = int(sys.argv[5])
if len(sys.argv) > 6 and sys.argv[6]: d["sched_bypass_total"] = int(sys.argv[6])
print(json.dumps(d))' "$@"
}
pinj() {   # pinj CELLJSON [EXTRAJSON]: the pin + the cell + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$1" "$x"
}
DESC="r00z's NST 5 start (the z0occ2 start: stack3 S-arm, stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, ds41_bi 2 pinned, draft_sample $DS pinned, --draft-coupling on, DLW sidecar $SIDE; ring 129, Z0-W declared, rungs 8,48; Z0 switches 0, draft_occ_k $OCCK; RADIANCE_DS41_PSK=0) + $STEPENV; sched_bi_chunk $BIC pinned"
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
  RADIANCE_DS41_PSK RADIANCE_SCHED_BI_CHUNK RADIANCE_SCHED_BI_GRID RADIANCE_LOG_STEPS RADIANCE_DS41_BITRACE_PMIN
  RADIANCE_DS41_BITRACE_LMAX"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset (= dsm-psk.sh's)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
# ---------------------------------------------------------------- helpers = dsm-psk.sh 841ce920 lines 114-203 + 223-231 (modes also
# checks RADIANCE_LOG_STEPS=1; mdelta also reports the decode steps' mean ms)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk $1 warm-up identity vs $2"; return 0; }
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
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV $STEPENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2800)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xpsk $1 VRAM at start (experts per tier; vs stack3 s3-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS3/s3-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack3 s3-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
arm() {   # arm CELLJSON: the pin + the cell + epoch_bump, a nudge to apply it at a step (dsm-dst: a cell's JSON, quoted)
  fn4set "$(pinj "$1" '{"epoch_bump": 1}')"; nudge
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window (+ the decode steps' mean ms: dsm-dst)
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
ds_ = g("engine_step_seconds_decode_total"); dn_ = g("engine_steps_decode_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da), decode_ms=round(1000 * ds_ / dn_, 1) if dn_ > 0 else None,
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-300; }
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack3 s3-n5 $(cat $RS3/guard-s3-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- DST: switches, smoke, alone references, cells
cgrid() {   # cgrid MT: the grid a cell's mixed chunks end on (sched_bi_chunk 3 / 4 with Scheduler::dst_grid); 1024 under 0 (r04's coarse grid)
  local mt=$1 g
  case $BIC in 3) g=2048 ;; 4) g=1024 ;; *) echo 1024; return ;; esac
  if [ $mt -lt 2112 ]; then local d=2048; while [ $d -gt $((mt - 64)) ] && [ $((d / 2)) -ge 128 ]; do d=$((d / 2)); done; [ $d -lt $g ] && g=$d; fi
  echo $g
}
swnote() {   # swnote TAG: start values (the flag, the shared budget, sched_bi_chunk 0, ds41_psk 0 ...); refusals; two cells applied
  local s0 g c1 c2 c3 sets="" bad="" cj
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (sched_interleave 8, sched_mixed_tok 2112, sched_bi_chunk 0, ds41_psk 0, ds41_bi 0, gg_gemv 2, aql_acq 2, K1 three 1, draft_sample 0, draft_capable 1, epoch 0); sched_short_bypass 0, GET "dst"; refusals sched_mixed_tok 2176 / 1000, sched_interleave 65, sched_short_bypass 1000 -> 400; cells $(cellj 32 576 $PSKV) and $(cellj $IL0 $MT0 0) applied, GET read back"; return 0; }
  s0=$(fn4get)
  c1=$(fn4code '{"sched_mixed_tok": 2176}'); c2=$(fn4code '{"sched_mixed_tok": 1000}'); c3=$(fn4code '{"sched_interleave": 65}')
  c3=$c3$(fn4code '{"sched_short_bypass": 1000}')$(fn4code '{"sched_bypass_total": 1000}')
  for cj in "$(cellj 32 576 $PSKV)" "$(cellj $IL0 $MT0 0)"; do
    arm "$cj"; g=$(fn4get)
    python3 -c 'import json,sys
g, w, c = json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
w.update(c)
if w.get("sched_mixed_tok") == 0: w["sched_mixed_tok"] = 2112   # 0 = back to the shared budget; GET reports the budget
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$FIX" "$cj" 2>/dev/null || bad="$bad $cj"
    sets="$sets [$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("il=%s,mt=%s,bic=%s,psk=%s,bi=%s,ep=%s" % (w.get("sched_interleave"), w.get("sched_mixed_tok"), w.get("sched_bi_chunk"), w.get("ds41_psk"), w.get("ds41_bi"), d.get("epoch")))' 2>/dev/null)]"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1200) | sched_mixed_tok 2176 -> HTTP $c1, 1000 -> HTTP $c2, sched_interleave 65 + sched_short_bypass 1000 + sched_bypass_total 1000 -> HTTP $c3 | cells applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("sched_interleave") == 8 and w.get("sched_mixed_tok") == 2112 and w.get("sched_bi_chunk") == 0 and w.get("ds41_psk") == 0
ok = ok and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill"))
ok = ok and w.get("ds41_bi") == 0 and s0["epoch"] == 0 and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0
ok = ok and w.get("sched_short_bypass") == 0 and w.get("sched_bypass_total") == 0 and "dst" in s0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1$c2$c3" = 400400400400400 ] && [ -z "$bad" ]
}
smoke() {   # smoke TAG: the extreme cell (interleave 32, budget 576, PSK on) -- the greedy set at conc 4; healthy -> 0
  local tag=$1 t0=$(date +%s) n
  arm "$(cellj 32 576 $PSKV)"
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: cell $(cellj 32 576 $PSKV): greedy set conc 4 (32 tok) 20 of 20, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  if [ "$n" = 20 ] && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (interleave 32, budget 576, PSK $PSKV): greedy set conc 4 20 of 20, healthy, $(( $(date +%s) - t0 )) s | $(pskstat)"; return 0; fi
  xlog "$tag smoke: VIRHE $n of 20 answered or unhealthy | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
lslice() {   # lslice L0 L1 FILE: the step lines of the engine log between two line counts
  docker logs $CN 2>&1 | sed -n "$(($1 + 1)),${2}p" | grep -E ' step [0-9]+ (prefill|decode|MIXED) ' > $3
}
chunkline() {   # chunkline CHUNKSJSON GRID PROMPT -> "n chunks (m MIXED), sizes {..}, k end off the G grid, j under 128 rows"
  python3 -c 'import json,sys,collections
try: c = json.load(open(sys.argv[1]))
except Exception as e: print("no chunk list (%s)" % type(e).__name__); sys.exit(0)
if not c.get("ok"): print("no chunk list (%s)" % c.get("why")); sys.exit(0)
G, P = int(sys.argv[2]), int(sys.argv[3])
ends = [b + q for b, q in zip(c["boundaries"], c["q"])]
off = sum(1 for e in ends if e % G and e < P - 1)
small = sum(1 for q in c["q"][:-1] if q < 128)
h = collections.Counter(c["q"]).most_common(6)
print("%d chunks (%d MIXED), sizes {%s}, %d end(s) off the %d grid, %d non-final under 128 rows" % (c["n_chunks"], c["mixed_chunks"],
      ", ".join("%dx%d" % kv for kv in h), off, G, small))' "$@" 2>&1 | cut -c1-400
}
probe() {   # probe TAG NAME CELLJSON MODE(a|l|n) GRID: a prefix-cache reset, ONE cold LDEPTH deepq prompt alone (a), beside c4 (l), or
  # beside c4 + the fixed newcomer set sent NEWC_AT s after it (n: the bypass section; -> TAG-NAME-newc.json)
  # -> $O/TAG-NAME.{txt,json,log,stamps.json,steps.log,chunks.json}; a row in alone.jsonl / load.jsonl
  local tag=$1 nm=$2 cj=$3 mode=$4 G=$5 t0=$(date +%s) l0 l1 j md pt c
  arm "$cj"; pcreset $tag-$nm
  msnap $tag-$nm-0
  if [ $mode != a ] && [ "$DRY" != 1 ]; then fn4get > $O/$tag-$nm-fn4-0.json; curl -s -m 20 $URL/stats > $O/$tag-$nm-stats-0.json 2>/dev/null; fi
  if [ "$DRY" = 1 ]; then l0=L0; else l0=$(docker logs $CN 2>&1 | wc -l); fi
  run "(cd $D && timeout 3600 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $NTOK $([ $mode != a ] && echo "--load --conc 4 --warm 10 --pf-tok 0 --stamps $O/$tag-$nm-stamps.json ")$([ $mode = n ] && echo "--newc $NEWC --newc-at $NEWC_AT --newc-tok $NEWC_TOK --newc-long $NEWC_LONG --newc-out $O/$tag-$nm-newc.json ")--out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1
  if [ $mode != a ] && [ "$DRY" != 1 ]; then fn4get > $O/$tag-$nm-fn4-1.json; curl -s -m 20 $URL/stats > $O/$tag-$nm-stats-1.json 2>/dev/null; fi
  run "sleep 2"
  if [ "$DRY" = 1 ]; then
    [ $mode != a ] && echo "+ GET /admin/fn4 + /stats before and after; python3 -I $GD check (bypass: dst.bypassed grew; D: probe chunk ends on alone's ends or the $G grid >= 128 a side, a grid cut in a MIXED step; budget; interleave; PSK: [$PSKPATHS] grew, [$PSKZERO] stayed) -> VIRHE + no speed reading when a lever did not act"
    echo "+ docker logs $CN | lines L0+1..L1 step lines > $O/$tag-$nm-steps.log; python3 -I $PL chunks ... --json $O/$tag-$nm-chunks.json; RESULT xdst $tag $nm (cell $cj, $([ $mode = a ] && echo alone || echo 'beside c4')$([ $mode = n ] && echo " + $NEWC fixed newcomers at +$NEWC_AT s"), grid $G)"
    return 0
  fi
  l1=$(docker logs $CN 2>&1 | wc -l)
  lslice $l0 $l1 $O/$tag-$nm-steps.log
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  pt=$(echo "$j" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("prompt_tokens") or 0)' 2>/dev/null || echo 0)
  timeout 300 python3 -I $PL chunks $O/$tag-$nm-steps.log --prompt $pt --json $O/$tag-$nm-chunks.json > /dev/null 2>&1
  c=$(chunkline $O/$tag-$nm-chunks.json $G $pt)
  # the coordinator's guard: did the cell's levers act? (a VIRHE = no speed reading for this run)
  gr=""
  if [ $mode != a ]; then
    gr=$(timeout 120 python3 -I $GD check --cell "$(pinj "$cj" | python3 -c 'import json,sys; d=json.load(sys.stdin); c=json.loads(sys.argv[1]); print(json.dumps({k: d.get(k) for k in ("sched_interleave", "sched_mixed_tok", "ds41_psk", "sched_short_bypass")} | ({"sched_bi_chunk": c["sched_bi_chunk"]} if "sched_bi_chunk" in c else {})))' "$cj")" \
         --bic $BIC --grid $G --il $(echo "$cj" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sched_interleave", 8))') \
         --mt $(echo "$cj" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sched_mixed_tok") or 2112)') \
         --chunks $O/$tag-$nm-chunks.json --alone $O/$tag-a-p0-chunks.json --fn4a $O/$tag-$nm-fn4-0.json --fn4b $O/$tag-$nm-fn4-1.json \
         --stats-a $O/$tag-$nm-stats-0.json --stats-b $O/$tag-$nm-stats-1.json --psk-paths "$PSKPATHS" --psk-zero "$PSKZERO" 2>&1 | grep -m1 '^GUARD ' | cut -c7-)
    [ -n "$gr" ] || gr="VIRHE the guard gave no verdict"
    case "$gr" in VIRHE*) xlog "$tag $nm: $gr -- a lever did not act: no speed reading for this run" ;; esac
  fi
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], cell=json.loads(sys.argv[4]), grid=int(sys.argv[5]), chunks=sys.argv[6], text=sys.argv[7])
if sys.argv[9]: d.update(guard=sys.argv[9])
if sys.argv[9].startswith("VIRHE"): d.update(invalid=sys.argv[9])
try:
    ch = json.load(open(sys.argv[8]))
    d.update(n_chunks=ch.get("n_chunks"), chunk_q=ch.get("q"), chunk_b=ch.get("boundaries"), chunks_mixed=ch.get("mixed_chunks"))
except Exception: pass
print(json.dumps(d))' "$j" "$md" "$nm" "$cj" "$G" "$c" "$O/$tag-$nm.txt" "$O/$tag-$nm-chunks.json" "$gr" >> $O/$(case $mode in l) echo load ;; n) echo bypass ;; *) echo alone ;; esac).jsonl
  if [ $mode != a ]; then
    xlog "$tag $nm (cell $cj, c4 + one cold ~$LDEPTH-depth deepq prompt, T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | window %s s: c4 %s tok, agg %s tok/s | TPOT in a request p50 %s / p90 %s / p99 %s / max %s ms (all gaps p90 %s / max %s) | c4 requests sent in the window %s, their TTFT p50 %s / p90 %s / max %s ms | fixed newcomers %s (%s err, sent +%s s) TTFT p50 %s / p90 %s / max %s ms | mixed %s, non-decode %s at %s ms, decode steps %s at %s ms | c4 %s req, %s err" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_tokens"), d.get("c4_pf_agg_tps"),
    d.get("tpot_in_p50"), d.get("tpot_in_p90"), d.get("tpot_in_p99"), d.get("tpot_in_max"), d.get("tpot_pf_p90"), d.get("tpot_pf_max"),
    d.get("turn_req_n"), d.get("turn_ttft_p50"), d.get("turn_ttft_p90"), d.get("turn_ttft_max"), d.get("newc_n"), d.get("newc_err"), d.get("newc_sent_after_s"),
    d.get("newc_ttft_p50"), d.get("newc_ttft_p90"), d.get("newc_ttft_max"), m.get("mixed_steps"), m.get("nondecode_steps"), m.get("nondecode_ms"),
    m.get("decode_steps"), m.get("decode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-1100) | bypass $(fn4get | python3 -c 'import json,sys; print(json.load(sys.stdin).get("dst"))' 2>/dev/null) | guard: ${gr:0:400} | chunks: $c | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
  else
    xlog "$tag $nm (cell $cj, ALONE, cold ~$LDEPTH-depth deepq prompt, T 0): $(echo "$j" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok, %s chars" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("chars")))' 2>&1 | cut -c1-300) | non-decode $(echo "$md" | python3 -c 'import json,sys; m=json.load(sys.stdin); print("%s at %s ms" % (m.get("nondecode_steps"), m.get("nondecode_ms")))' 2>/dev/null) | chunks: $c | $(thermline $t0 $(date +%s))"
  fi
}
alones() {   # alones TAG: the alone references -- base (PSK 0, sched_bi_chunk 0), PSK 0 and PSK on at BIC; identity among them and vs r00z
  local tag=$1 r0
  probe $tag a-base "$(cellj $IL0 $MT0 0 0)" a 1024
  probe $tag a-p0 "$(cellj $IL0 $MT0 0)" a 1024
  probe $tag a-p$PSKV "$(cellj $IL0 $MT0 $PSKV)" a 1024
  newcalone $tag 0; newcalone $tag $PSKV
  [ "$DRY" = 1 ] && { echo "+ identity: a-p0 / a-p$PSKV vs a-base; a-base vs r00z's 64k A ($RPSK/ps-n5-d64000-1A.txt, cross-start, same prompt and start)"; return 0; }
  idrec "alone-p0-vs-base" "$(cmpfile $O/$tag-a-base.txt $O/$tag-a-p0.txt)" "alone: sched_bi_chunk $BIC vs 0 (alone keeps its pieces under 3 / 4)"
  idrec "alone-p$PSKV-vs-base" "$(cmpfile $O/$tag-a-base.txt $O/$tag-a-p$PSKV.txt)" "alone: PSK $PSKV vs 0"
  r0=$RPSK/ps-n5-d64000-1A.txt
  idrec "alone-base-vs-r00zA" "$(cmpfile $r0 $O/$tag-a-base.txt)" "alone base vs r00z's 64k A answer (cross-start; pskld text = deepq.py's)"
}
NS=0        # loaded runs so far (a run's name carries its index: the base cell runs four times)
cell() {   # cell TAG IL MT P B [BIC]: one loaded cell in block B (BIC: another sched_bi_chunk than the kept one, e.g. 0 = today);
  # identity vs the alone answer of its PSK value and vs the base
  local tag=$1 il=$2 mt=$3 p=$4 bc=${6:-} nm al
  NS=$((NS + 1)); nm=l$(printf %02d $NS)-i$2-m$3-p$4-b$5${bc:+-c$bc}
  stopfile && return 0
  probe $tag $nm "$(cellj $il $mt $p $bc)" l $([ -n "$bc" ] && [ "$bc" = 0 ] && echo 1024 || cgrid $mt)
  [ "$DRY" = 1 ] && return 0
  al=$O/$tag-a-p$p.txt; [ $p = 0 ] || [ -s $al ] || al=$O/$tag-a-p0.txt
  idrec "$nm-vs-alone" "$(cmpfile $al $O/$tag-$nm.txt)" "loaded vs alone, PSK $p"
  idrec "$nm-vs-base" "$(cmpfile $O/$tag-a-base.txt $O/$tag-$nm.txt)" "loaded vs the base (alone, PSK 0, sched_bi_chunk 0)"
}
ORDER1=""   # block 1: base, the interleave axis at the first budget, the budget axis at the first interleave, the corners, base
orders() {
  local il mt
  ORDER1="$IL0:$MT0"
  for il in $ILS; do [ $il = $IL0 ] || ORDER1="$ORDER1 $il:$MT0"; done
  for mt in $MTS; do [ $mt = $MT0 ] || ORDER1="$ORDER1 $IL0:$mt"; done
  for mt in $MTS; do [ $mt = $MT0 ] && continue; for il in $ILS; do [ $il = $IL0 ] || case " $SKIP " in *" $il:$mt "*) ;; *) ORDER1="$ORDER1 $il:$mt" ;; esac; done; done
  ORDER1="$ORDER1 $IL0:$MT0"
}
blocks() {   # blocks TAG: block 1 (PSK 0 then on in each cell), block 2 = block 1 reversed (on then 0), stopped at the budget
  local tag=$1 c il mt left skipped="" b2
  orders
  xlog "$tag block 1 order: $ORDER1 (each: PSK 0, then PSK $PSKV); block 2: reversed, PSK $PSKV then 0, until $BUDGET min from the item's start"
  [ "$TODAY" = 1 ] && [ $BIC != 0 ] && cell $tag $IL0 $MT0 0 1 0   # today's chunking (sched_bi_chunk 0) at the base cell, PSK 0
  for c in $ORDER1; do il=${c%:*}; mt=${c#*:}
    stopfile && return 0
    cell $tag $il $mt 0 1; cell $tag $il $mt $PSKV 1
  done
  bypick $tag
  bypass_half $tag A
  if [ $BLOCKS = 2 ]; then
    b2=$(echo $ORDER1 | tr ' ' '\n' | tac | tr '\n' ' ')
    for c in $b2; do il=${c%:*}; mt=${c#*:}
      stopfile && return 0
      # block 2 stops BYPRES min before the budget: the bypass section's second half (its ABBA pairs) runs after it
      if [ "$DRY" != 1 ] && [ $(( ($(date +%s) - T_ITEM) / 60 )) -ge $((BUDGET - BYPRES)) ]; then skipped="$skipped $c"; continue; fi
      cell $tag $il $mt $PSKV 2; cell $tag $il $mt 0 2
    done
    [ -z "$skipped" ] || xlog "$tag block 2: budget $BUDGET - $BYPRES min reached -- cells not sampled twice:$skipped"
    [ "$TODAY" = 1 ] && [ $BIC != 0 ] && ! stopfile && cell $tag $IL0 $MT0 0 2 0
  fi
  bypass_half $tag B
}
BIL=""      # the bypass section's second cell: the best interleave at the first budget (bypick)
bypick() {   # bypick TAG: XDS_BYPIL, or (auto) block 1's interleave != IL0 at MT0 with the most c4 tok/s whose long-prompt TTFT is
  # at most 1.5 x the base cell's (PSK 0 and on pooled); none: the smallest interleave above IL0
  local tag=$1
  if [ "$BYPIL" != auto ]; then BIL=$BYPIL; xlog "$tag bypass cells: interleave $IL0 and $BIL (XDS_BYPIL) at budget $MT0"; return 0; fi
  if [ "$DRY" = 1 ]; then BIL=$(echo $ILS | tr ' ' '\n' | grep -vx $IL0 | head -1); echo "+ bypick: block 1's best interleave at budget $MT0 (c4 tok/s, TTFT <= 1.5 x base) -> $BIL in the dry run"; return 0; fi
  BIL=$(python3 - "$O/load.jsonl" "$IL0" "$MT0" "$ILS" <<'PY2'
import json, statistics, sys
p, il0, mt0, ils = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), [int(x) for x in sys.argv[4].split()]
rows = []
try: rows = [json.loads(l) for l in open(p) if l.strip()]
except OSError: pass
def cell(il):
    return [r for r in rows if (r.get("cell") or {}).get("sched_interleave") == il and ((r.get("cell") or {}).get("sched_mixed_tok") or 2112) == mt0
            and (r.get("cell") or {}).get("sched_bi_chunk") is None]
def m(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
b = m([r.get("ttft") for r in cell(il0)])
best = None
for il in ils:
    if il == il0: continue
    c, t = m([r.get("c4_pf_agg_tps") for r in cell(il)]), m([r.get("ttft") for r in cell(il)])
    if c is None or t is None or b is None or t > 1.5 * b: continue
    if best is None or c > best[1]: best = (il, c)
print(best[0] if best else min([x for x in ils if x > il0] or [il0]))
PY2
)
  xlog "$tag bypass cells: interleave $IL0 and $BIL at budget $MT0 (auto: block 1's most c4 tok/s with long-prompt TTFT <= 1.5 x the base's)"
}
bypass_half() {   # bypass_half TAG A|B: the bypass section -- at (IL0, MT0) and (BIL, MT0), sched_short_bypass 0 and BYPS, PSK 0 / on,
  # each beside c4 + the fixed newcomer set; A: per cell bypass ascending, PSK 0 then on; B: the mirror (ABBA pairs)
  local tag=$1 h=$2 seq="" x il b p
  for il in $IL0 $BIL; do
    for b in 0 $BYPS; do seq="$seq $il:$b:0"; done
    for b in $(echo 0 $BYPS | tr ' ' '\n' | tac); do seq="$seq $il:$b:$PSKV"; done
  done
  [ $h = B ] && seq=$(echo $seq | tr ' ' '\n' | tac | tr '\n' ' ')
  xlog "$tag bypass half $h: (interleave:bypass:PSK)$seq; $NEWC fixed newcomers (~1000 tokens, $NEWC_TOK greedy tokens) at +$NEWC_AT s"
  for x in $seq; do il=${x%%:*}; b=${x#*:}; b=${b%%:*}; p=${x##*:}
    stopfile && return 0
    bcell $tag $il $b $p $h
  done
}
bcell() {   # bcell TAG IL BYPASS P HALF: one bypass run; identity of the long answer vs alone, of each fixed newcomer vs its alone answer
  local tag=$1 il=$2 b=$3 p=$4 nm al k r
  NS=$((NS + 1)); nm=l$(printf %02d $NS)-i$il-m$MT0-p$p-y$b-$5
  probe $tag $nm "$(cellj $il $MT0 $p '' $b $([ $b != 0 ] && echo $BYPTOT || echo 0))" n $(cgrid $MT0)
  [ "$DRY" = 1 ] && { echo "+ identity: $nm long answer vs alone p$p; each fixed newcomer vs $O/$tag-newc-alone-p$p.json"; return 0; }
  al=$O/$tag-a-p$p.txt; [ -s $al ] || al=$O/$tag-a-p0.txt
  idrec "$nm-vs-alone" "$(cmpfile $al $O/$tag-$nm.txt)" "bypass section: loaded vs alone, PSK $p"
  r=$(python3 -c 'import json,sys
try: a = {x["k"]: x for x in json.load(open(sys.argv[1]))}; l = {x["k"]: x for x in json.load(open(sys.argv[2]))}
except Exception as e: print("not compared (%s)" % type(e).__name__); sys.exit(0)
bad = [k for k in sorted(a) if not a[k].get("text") or (l.get(k) or {}).get("text") != a[k]["text"]]
print(("IDENTICAL (%d newcomers)" % len(a)) if a and not bad else ("DIFFERENT: newcomer(s) %s of %d" % (bad, len(a))))' $O/$tag-newc-alone-p$p.json $O/$tag-$nm-newc.json 2>&1 | cut -c1-200)
  idrec "$nm-newc-vs-alone" "$r" "bypass section: the fixed newcomers' answers vs each alone, PSK $p"
}
newcalone() {   # newcalone TAG P: the fixed newcomer set alone (one by one, nothing else running) under the base cell, PSK P
  local tag=$1 p=$2
  arm "$(cellj $IL0 $MT0 $p)"; pcreset $tag-newc-alone-p$p
  run "(cd $D && timeout 900 python3 -I $PL newc-alone --url $URL --tests $TESTS --mb $MBP --newc $NEWC --newc-tok $NEWC_TOK --newc-long $NEWC_LONG --newc-out $O/$tag-newc-alone-p$p.json > $O/$tag-newc-alone-p$p.log 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag fixed newcomers alone, PSK $p: $(grep -m1 '^NEWC ' $O/$tag-newc-alone-p$p.log | cut -c1-400)"
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xdst summary: per cell (interleave, budget, PSK) the means over its samples (c4 tok/s in the window, TPOT in a request p50 / p90 / p99 / max, the turnovers' TTFT max, mixed / decode step ms, probe TTFT, chunks) and vs the base cell (%); PSK on vs 0 per (interleave, budget); the identity table"; return 0; }
  python3 - "$O" "$IL0" "$MT0" "$PSKV" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, il0, mt0, pv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def m(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def f(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(b, a): return "%+.1f %%" % ((b / a - 1) * 100) if a and b else "-"
inv = [r for r in rows(os.path.join(o, "load.jsonl")) + rows(os.path.join(o, "bypass.jsonl")) if r.get("invalid")]
for r in inv:
    print("INVALID %s (no speed reading): %s" % (r.get("name"), r["invalid"][:400]))
ld = [r for r in rows(os.path.join(o, "load.jsonl")) if not r.get("invalid")]
cells = {}
for r in ld:
    c = r.get("cell") or {}
    mt = c.get("sched_mixed_tok") or 2112
    if c.get("sched_bi_chunk") is not None:   # a sched_bi_chunk other than the kept one (0 = today's chunking): its own line
        print("TODAY-CHUNKING sched_bi_chunk %s interleave %s budget %s PSK %s: c4 %s tok/s, TPOT in p90 %s / max %s ms, turnover TTFT max %s ms, mixed %s ms, TTFT %s s, chunks %s" % (
              c["sched_bi_chunk"], c.get("sched_interleave"), mt, c.get("ds41_psk"), r.get("c4_pf_agg_tps"), r.get("tpot_in_p90"), r.get("tpot_in_max"),
              r.get("turn_ttft_max"), r.get("nondecode_ms"), r.get("ttft"), r.get("chunks")))
        continue
    cells.setdefault((c.get("sched_interleave"), mt, c.get("ds41_psk")), []).append(r)
K = [("c4 tok/s", "c4_pf_agg_tps", 1), ("TPOT p50", "tpot_in_p50", 1), ("p90", "tpot_in_p90", 0), ("p99", "tpot_in_p99", 0),
     ("max", "tpot_in_max", 0), ("all-gap max", "tpot_pf_max", 0), ("turnover TTFT max", "turn_ttft_max", 0),
     ("mixed ms", "nondecode_ms", 0), ("decode ms", "decode_ms", 1), ("TTFT s", "ttft", 2), ("chunks", "n_chunks", 0)]
for key in sorted(cells, key=lambda k: (k[2], k[1] * -1, k[0])):
    rs = cells[key]; base = cells.get((il0, mt0, key[2]), [])
    parts = []
    for nm, k, n in K:
        a, b = m([r.get(k) for r in rs]), m([r.get(k) for r in base])
        parts.append("%s %s%s" % (nm, f(a, n), "" if key[:2] == (il0, mt0) or k == "n_chunks" else " (%s)" % pc(a, b)))
    print("CELL interleave %s budget %s PSK %s (n=%d; vs (%d, %d) same PSK): %s | samples: c4 %s, TTFT %s" % (key[0], key[1], key[2], len(rs), il0, mt0,
          ", ".join(parts), "/".join(f(r.get("c4_pf_agg_tps")) for r in rs), "/".join(f(r.get("ttft"), 2) for r in rs)))
for (il, mt, p) in sorted(cells):
    if p != 0 or (il, mt, pv) not in cells: continue
    a, b = cells[(il, mt, 0)], cells[(il, mt, pv)]
    print("PSK interleave %s budget %s: PSK %d vs 0: c4 tok/s %s, TTFT %s, mixed ms %s, TPOT in p90 %s" % (il, mt, pv,
          pc(m([r.get("c4_pf_agg_tps") for r in b]), m([r.get("c4_pf_agg_tps") for r in a])), pc(m([r.get("ttft") for r in b]), m([r.get("ttft") for r in a])),
          pc(m([r.get("nondecode_ms") for r in b]), m([r.get("nondecode_ms") for r in a])), pc(m([r.get("tpot_in_p90") for r in b]), m([r.get("tpot_in_p90") for r in a]))))
by = [r for r in rows(os.path.join(o, "bypass.jsonl")) if not r.get("invalid")]
bc = {}
for r in by:
    c = r.get("cell") or {}
    bc.setdefault((c.get("sched_interleave"), c.get("sched_short_bypass"), c.get("ds41_psk")), []).append(r)
KB = [("fixed newcomer TTFT p50", "newc_ttft_p50", 0), ("p90", "newc_ttft_p90", 0), ("max", "newc_ttft_max", 0),
      ("c4 turnover TTFT p50", "turn_ttft_p50", 0), ("p90", "turn_ttft_p90", 0), ("max", "turn_ttft_max", 0),
      ("TPOT in p50", "tpot_in_p50", 1), ("p99", "tpot_in_p99", 0), ("max", "tpot_in_max", 0),
      ("long TTFT s", "ttft", 2), ("c4 tok/s", "c4_pf_agg_tps", 1)]
for key in sorted(bc, key=lambda k: (k[0], k[2], k[1])):
    rs = bc[key]; base = bc.get((key[0], 0, key[2]), [])
    parts = []
    for nm, k, n in KB:
        a, b = m([r.get(k) for r in rs]), m([r.get(k) for r in base])
        parts.append("%s %s%s" % (nm, f(a, n), "" if key[1] == 0 else " (%s)" % pc(a, b)))
    print("BYPASS interleave %s sched_short_bypass %s PSK %s (n=%d; vs bypass 0): %s | samples: newcomer max %s, long TTFT %s" % (
          key[0], key[1], key[2], len(rs), ", ".join(parts), "/".join(f(r.get("newc_ttft_max"), 0) for r in rs), "/".join(f(r.get("ttft"), 2) for r in rs)))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if "IDENTICAL" not in l]
print("SUMMARY guard: %d run(s) invalid (a lever did not act) || identity: %s || cells %d, loaded runs %d, bypass runs %d" % (len(inv), ("IDENTICAL at every compare (%d)" % len(ident)) if ident and not bad else
      (("NOT IDENTICAL (%d of %d): " % (len(bad), len(ident)) + "; ".join(bad))[:1500] if ident else "none"), len(cells), len(ld), len(by)))
PY
  while IFS= read -r l; do xlog "${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=ds-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV $STEPENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV $STEPENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | cells interleave [$ILS] x budget [$MTS] x PSK [0 $PSKV], grids $(for mt in $MTS; do printf '%s->%s ' $mt $(cgrid $mt); done)| env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    docker logs $CN 2>&1 | grep -qE ' step [0-9]+ (prefill|decode|MIXED) n_seq=' || { xlog "$tag: VIRHE no step lines after the warm-up (RADIANCE_LOG_STEPS=1)"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag || { xlog "$tag: VIRHE the smoke failed -> stopped"; down; return 1; }
  stopfile || alones $tag
  stopfile || blocks $tag
  run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
  xlog "$tag: PSK stats at the end: $(pskstat) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
orders
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | cells (interleave:budget) $ORDER1 x PSK 0 / $PSKV, $BLOCKS block(s), budget $BUDGET min | alone base / p0 / p$PSKV | no evict | refs r00z 64k A (cross-start)"
IMGLITS="B|sched_interleave
B|sched_short_bypass
B|sched_short_bypass takes 0 (off)
B|sched_bypass_total
B|plan_mismatch
B|sched_mixed_tok
B|sched_bi_chunk
B|sched_interleave takes 0 .. 64
B|sched_mixed_tok takes 0 (back to the shared budget
B|RADIANCE_SCHED_BI_CHUNK
B|RADIANCE_LOG_STEPS
B|ds41_psk
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
K|dspark_walk_sample_step
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ_k takes 1 .. 4096
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not an image with lever DST + PSK + BI chunking -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) carries lever DST (sched_interleave / sched_mixed_tok), sched_bi_chunk, ds41_psk and the int5 levers: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present + bin/rad-sidecar -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $DQ $PL $GD $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $DQ $PL $GD $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST:0:300})" ;; esac
GST=$(timeout 60 python3 -I $GD selftest 2>&1 | tail -1); case "$GST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$GD(${GST:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-dstld:(${DQP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog cellj pinj stackenv keeplog thermline cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat gate1 cgrid swnote smoke lslice chunkline probe alones cell orders blocks bypick bypass_half bcell newcalone summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip tac; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for il in $ILS; do for mt in $MTS; do for p in 0 $PSKV; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj "$(cellj $il $mt $p)")" 2>/dev/null || HMISS="$HMISS json:$il:$mt:$p"; done; done; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, dstguard.py $GST, dstld.py $ST, deepq.py = dstld.py's deep request ($DQP), all $(echo $FNS | wc -w) functions resolve, tools on PATH, every cell's pin parses, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RPSK/ps-n5-d64000-1A.txt $WARM_REF_225; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: dstguard.py $(md5sum < $GD | cut -c1-8), dstld.py $(md5sum < $PL | cut -c1-8) (pskld.py $(md5sum < $HD/../psk/pskld.py 2>/dev/null | cut -c1-8)), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RPSK/ps-n5-d64000-1A.txt $WARM_REF_225 $RS3/s3-n5-stats0.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ pin base: $(pinj "$(cellj $IL0 $MT0 0)")"
  echo "+ grids: $(for mt in $MTS; do printf 'budget %s -> %s; ' $mt $(cgrid $mt); done)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
