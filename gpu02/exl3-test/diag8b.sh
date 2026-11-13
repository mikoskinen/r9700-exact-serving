#!/bin/bash
# diag8b.sh (5.11.2026) -- diag8 again without prefill span/interleave (EXTRA2 empty; NST 0, ck 2048): does the garbage at 8 at once
# come from the interleaved prefill path (avoidable by config) or from the batched decode itself?
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
EXTRA2= $T/exl3-restart.sh sw-diag8b > $R/sw-diag/restart8b.txt 2>&1 || log "RESULT diag8b: VIRHE restart failed"
timeout 1800 python3 -I $T/diag_cc.py http://127.0.0.1:8090 $R/sw-diag/d8b 8 4000 > $R/sw-diag/diag8b.txt 2>&1
grep '^DIAG ' $R/sw-diag/diag8b.txt | while read -r l; do log "RESULT diag8b: ${l:5}"; done
log "RESULT diag8b: done ($(grep -c '^DIAG ' $R/sw-diag/diag8b.txt) replies)"
