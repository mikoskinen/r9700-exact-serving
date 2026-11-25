#!/bin/bash
# chain-sweep2.sh (5.11.2026; user: "ehkä diagnoosi ennen muita testejä?") -- window.sh was stopped after its ident restart (its ident
# and prefill phases skipped); the EXL3 server already runs the window config (NST 0, ck 2048, span 2, il 8) and tempmon + watchdog
# keep running: diag_cc.py at once (no restart), then sweep-exl3.sh on the cfg without its diag row, then window.sh PHASES=down,produp.
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
log "RESULT chain-sweep2: start 2 (11.05: 1st diag try lacked cc_sim load_corpus) -- restart in the window config, then diag"
O=$R/sw-diag; mkdir -p $O
env -u EXTRA2 $T/exl3-restart.sh sw-diag > $O/restart.txt 2>&1 || log "RESULT chain-sweep2: VIRHE diag restart failed: $(tail -1 $O/restart.txt | cut -c1-200)"
timeout 1800 python3 -I $T/diag_cc.py http://127.0.0.1:8090 $O/diag > $O/diag.txt 2>&1
grep '^DIAG ' $O/diag.txt | while read -r l; do log "RESULT sweep diag diag: ${l:5}"; done
grep -q '^DIAG ' $O/diag.txt || log "RESULT sweep diag diag: VIRHE $(tail -2 $O/diag.txt | tr '\n' ' ' | cut -c1-300)"
grep -v '^diag|' $T/sweep-exl3a.cfg > $T/sweep-exl3b.cfg
./sweep-exl3.sh $T/sweep-exl3b.cfg
PHASES=down,produp ./window.sh > $R/chain-sweep-up.out 2>&1
log "RESULT chain-sweep2: done; production $(docker ps --format '{{.Names}} {{.Image}}' | grep prod | tr '\n' ' ')"
