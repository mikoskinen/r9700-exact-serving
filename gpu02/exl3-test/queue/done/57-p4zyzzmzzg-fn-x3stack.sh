#!/bin/bash
# 148-fn-x3stack.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~60 min, one arm)
# -- E7's validation (coordinator 10.11.): the Flash-Next EXL3 4.05 line (qwen3.8-flash-next-exl3.rad, EP4, all_vram) on the
# MAIN-LINE stack, + held x3pf's pipelined-prefill check: arm x3stk of 93r-fn-tp4ab.sh = radiance-local:c16-x3 (fn4-x3 =
# fn4-c16 91f566c + the seven fnx EXL3 commits -- the canonical exl3 encoding through moe_exl3_gate_up / moe_exl3_down,
# qwen4exp_x3 under the (qwen4exp, exl3) key, EP whole experts at >= 2 ranks, kept-bf16 experts, odd counts -- and exl3r4d
# from exl3-117-gu25c fa0e958 rebuilt at ABI 37), F1n + F2a + F2b + hcw + hc_readv 11 + HC_SS + hc_upmix 5, MNBT 4096 +
# --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest; TP4 MML 262144 MAXSEQS 16 MTP 3 wht6 210 W.
# Phases (93r, X93_SCREEN 0): gate, slow-state probe, produp pair + probe, mb c4 / c8 / c16 (180 s) with engine stats,
# THEN prefill 16k warm + fresh 64k / 128k (thermwin), the 33k copy against 122's e4r copy (X93_COPYREF e4r: the 1.17 DS
# core vs main -- information unless identical), mb pf, BetterBench, the c16 card-0 profile; no cc_sim. Compared (cmparm,
# X93_REF e4r, no re-run) against 122's e4r (mb c4 / c8 / c16 on the -fnx4 line, c16 913.9) -- hc_read was 17.5 % of the
# EXL3 c16 step there.
# DRYRUN=1 bash 148-fn-x3stack.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 148-fn-x3stack: the EXL3 4.05 line on the main-line stack (c16-x3), vs 122 e4r"
export ARMS=${X148_ARMS:-x3stk} X93_REF=${X93_REF:-e4r} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-e4r}
export X93_STK_RV=${X148_RV:-11} X93_STK_UM=${X148_UM:-5}
export X93_SCREEN=0 X93_SCREEN_PF=0 X93_CC_ARMS=none X93_BB_ARMS=x3stk X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-148-fn-x3stack}
B=; for f in "$H/93r-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93r-fn-tp4ab.sh"; do [ -e "$f" ] && { B=$f; break; }; done
[ -n "$B" ] || { log "RESULT fn4 148-fn-x3stack: VIRHE 93r-fn-tp4ab.sh not found beside $H"; exit 1; }
bash "$B"; rc=$?
exit $rc
