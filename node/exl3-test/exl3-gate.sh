#!/bin/bash
# exl3-gate.sh TAG -- exl3-restart.sh TAG + the 3.11. guard (~/gpu/ds41/gated-restart.sh): warm-up (3 short requests + one 16k
# prefill), fastcheck (1-user greedy decode ms/step); if > GUARD_MS (default 21 here: UD-IQ1_S fast state 15-16.5 ms, EXL3
# expected 17-19.5) restart once and check again. Writes runs/guard-TAG.txt and a RESULT guard line. Exit 1 only if no healthy server.
set -u
TAG=$1; . "$(dirname "$0")/env.sh"; O=$R/guard; mkdir -p $O; G=$GUARD_MS
$T/exl3-restart.sh $TAG > $O/$TAG-restart1.txt 2>&1 || { tail -1 $O/$TAG-restart1.txt; exit 1; }
[ "$G" = 0 ] && { echo "unguarded" > $R/guard-$TAG.txt; exit 0; }
check() {
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O > $O/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --calibrate 0 > $O/$TAG-warm-pf.txt 2>&1)
  (cd $D && timeout 300 python3 fastcheck.py 384 > $O/$TAG-fc$1.txt 2>&1)
  grep '^FC ' $O/$TAG-fc$1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999
}
ms=$(check 1); st=fast
if awk -v m=$ms -v g=$G 'BEGIN{exit !(m > g)}'; then
  ms1=$ms; $T/exl3-restart.sh $TAG > $O/$TAG-restart2.txt 2>&1 || exit 1
  ms=$(check 2); st="fast-after-retry (first $ms1)"
  awk -v m=$ms -v g=$G 'BEGIN{exit !(m > g)}' && st="SLOW-after-retry (first $ms1)"
fi
echo "$st $ms" > $R/guard-$TAG.txt
echo "$(date '+%F %T') RESULT guard $TAG: $st, 1-user ${ms} ms/step (limit $G; UD-IQ1_S 3.11. fast 15.05-15.12)" >> $L
exit 0
