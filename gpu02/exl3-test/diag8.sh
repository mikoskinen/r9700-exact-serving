#!/bin/bash
# diag8.sh (5.11.2026) -- batch check: the same cc_sim sessions u0..u7 sent 8 at once (cc_sim efforts, max 4000 tokens) to the EXL3
# server in the window config, after the 2-at-once diag ended normally (963 / 237 tokens, tool calls) while cc_sim u4/u8 ran to 13000.
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
env -u EXTRA2 $T/exl3-restart.sh sw-diag8 > $R/sw-diag/restart8.txt 2>&1 || log "RESULT diag8: VIRHE restart failed"
timeout 1800 python3 -I $T/diag_cc.py http://127.0.0.1:8090 $R/sw-diag/d8 8 4000 > $R/sw-diag/diag8.txt 2>&1
grep '^DIAG ' $R/sw-diag/diag8.txt | while read -r l; do log "RESULT diag8: ${l:5}"; done
log "RESULT diag8: done ($(grep -c '^DIAG ' $R/sw-diag/diag8.txt) replies)"
