#!/bin/bash
# engine_check.sh (5.11.2026) -- the prep_row fix in the engine: the 4-card window's diag8 again with the fixed image, then
# production back. Copy to ~/gpu/ds41/exl3-test/ on gpu02 and run from there AFTER LiteLLM routes the Claude tiers away
# from gpu02 (LITELLM_SWITCHED=1 is window.sh's own precondition for stopping production). Needs the image loaded
# (docker load) on gpu02. Outputs: runs/sw-fix/ (d8: the 8 replies; diag8.txt), RESULT lines in runs/ds41-exl3.log.
#   1. window.sh pre,prodstop,evict   (image + model checks, production A and B down, page cache evicted)
#   2. exl3-restart.sh fix-diag8      (the window config: NST 0, ck 2048, span 2, interleave 8, MAXSEQS 8, host pool 19456)
#   3. diag_cc.py 8 at once, max 4000 -- the same seed-1 cc_sim sessions u0..u7 (33k-142k prompts) as diag8 at 09:11
#   4. [CCSIM=1] window.sh ccsim      (cc_sim u4 + u8, 2 x ~11 min more)
#   5. window.sh down,produp          (always, also after a failed step: EXL3 server down, production A then B, gated)
# Wall clock: ~1.5 min stop + evict, ~3.5 min restart, ~9 min diag8, ~4 min production = ~18 min (+ ~25 min with CCSIM=1).
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; O=$R/sw-fix; cd $T
export IMAGE=${IMAGE:-radiance-local:ds41-exl3-9e8ccb64-3684c0b1}
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
up() { PHASES=down,produp ./window.sh > $O/produp.out 2>&1; log "RESULT engine_check: production $(docker ps --format '{{.Names}} {{.Status}}' | grep prod | tr '\n' ';')"; }
[ "${LITELLM_SWITCHED:-0}" = 1 ] || { echo "LITELLM_SWITCHED=1 first (LiteLLM away from gpu02)"; exit 1; }
mkdir -p $O
log "RESULT engine_check: start IMAGE=$IMAGE"
LITELLM_SWITCHED=1 PHASES=pre,prodstop,evict ./window.sh > $O/window-pre.out 2>&1
docker ps --format '{{.Names}}' | grep -q '^radiance-prod' && { log "RESULT engine_check: VIRHE production still up after prodstop (see $O/window-pre.out); nothing run"; exit 1; }
trap up EXIT
$T/exl3-restart.sh fix-diag8 > $O/restart.txt 2>&1 || { log "RESULT engine_check: VIRHE restart failed: $(tail -1 $O/restart.txt | cut -c1-300)"; exit 1; }
timeout 1800 python3 -I $T/diag_cc.py http://127.0.0.1:8090 $O/d8 8 4000 > $O/diag8.txt 2>&1
grep '^DIAG ' $O/diag8.txt | while read -r l; do log "RESULT engine_check diag8: ${l:5}"; done
log "RESULT engine_check diag8: $(grep -c '^DIAG ' $O/diag8.txt) replies, finish tool_calls $(grep -c '"finish": "tool_calls"' $O/diag8.txt), length $(grep -c '"finish": "length"' $O/diag8.txt) (09:11 shipped image: 0 tool_calls, 8 length)"
[ "${CCSIM:-0}" = 1 ] && PHASES=ccsim ./window.sh > $O/window-ccsim.out 2>&1
exit 0   # the EXIT trap brings production back
