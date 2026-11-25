#!/bin/bash
# 7.11. coordinator: end 57-nj after its 225:t3l arm (t3l16 arms would only fail the same gate), via runs/sweep.stop
# (lib-pr266 stopfile, checked before each arm); remove it again at 57-nj done line, before the next item starts.
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log
log() { echo "$(date "+%F %T") RESULT coordinator: $*" >> $L; }
timeout 5400 tail -n0 -F -s 0.2 $L | while read -r line; do
  case "$line" in
    *"RESULT x79 x79-225-t3l: done"*|*"RESULT x79 x79-225-t3l: VIRHE"*"arm "*) touch $R/sweep.stop; log "sweep.stop set after 225:t3l (skip t3l16 arms; NVFP4 latent decode +47 % under study)";;
    *"RESULT x79 79-shp-levels: done"*) rm -f $R/sweep.stop; log "sweep.stop removed (57-nj ended)"; exit 0;;
  esac
done
rm -f $R/sweep.stop
