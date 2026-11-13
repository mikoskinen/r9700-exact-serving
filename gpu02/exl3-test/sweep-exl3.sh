#!/bin/bash
# sweep-exl3.sh CONFIGFILE (5.11.2026; user: "tee kattavasti erilaisia testejä tällä mallilla eri asetuksilla, kuten viimeksikin
# deepseekin kanssa. mtp:t yms.") -- ~/gpu/ds41/sweep.sh (3.11. DS41 sweep) for the EXL3 container: same line format
# TAG|ENV|PHASES|EXTRA2, same phases (probe, sane, c4, c8, pf, turns, after), restart via exl3-restart.sh (env.sh defaults:
# HOSTPOOL 19456, MAXSEQS 8, RAM KV tier). RESULT lines -> runs/ds41-exl3.log. Stop between configs: touch runs/sweep.stop
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $HOME/gpu/ds41
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
while IFS="|" read -r TAG ENVS PH X2; do
  { [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ]; } && continue
  [ -e $R/sweep.stop ] && { log "RESULT sweep: stopped by sweep.stop before $TAG"; break; }
  O=$R/sw-$TAG; mkdir -p $O
  if ! env EXTRA2="$X2" $ENVS $T/exl3-restart.sh sw-$TAG > $O/restart.txt 2>&1; then
    log "RESULT sweep $TAG: VIRHE restart failed ($ENVS $X2): $(tail -1 $O/restart.txt | cut -c1-300)"; continue; fi
  for p in ${PH//,/ }; do
    case $p in
      probe|after)
        timeout 600 python3 stepprobe.py essay,code 512 > $O/$p.txt 2>&1
        log "RESULT sweep $TAG $p (1 user, $ENVS $X2): $(python3 sweepfmt.py probe $O/$p.txt 2>&1)";;
      sane)
        timeout 900 python3 ds41_tests.py correct --only-q --effort none --outdir $O > $O/sane-q.txt 2>&1
        timeout 900 python3 ds41_tests.py copy --chars 15000 --runs 2 --outdir $O > $O/sane-copy.txt 2>&1
        log "RESULT sweep $TAG sane: $(python3 sweepfmt.py sane $O/sane-q.txt $O/sane-copy.txt 2>&1)";;
      diag)
        timeout 1800 python3 -I $T/diag_cc.py http://127.0.0.1:8090 $O/diag > $O/diag.txt 2>&1
        grep '^DIAG ' $O/diag.txt | while read -r l; do log "RESULT sweep $TAG diag: ${l:5}"; done
        grep -q '^DIAG ' $O/diag.txt || log "RESULT sweep $TAG diag: VIRHE $(tail -2 $O/diag.txt | tr '\n' ' ' | cut -c1-300)";;
      c4|c8|pf|turns)
        timeout 1500 python3 mb.py --tag $TAG --phases $p > $O/$p.txt 2>&1
        log "RESULT sweep $TAG $p ($ENVS $X2): $(grep '^MB ' $O/$p.txt | cut -c4- || tail -1 $O/$p.txt)";;
    esac
  done
  docker logs radiance-ds41-exl3 > $O/engine.log 2>&1
done < $1
log "RESULT sweep: done ($(basename $1))"
