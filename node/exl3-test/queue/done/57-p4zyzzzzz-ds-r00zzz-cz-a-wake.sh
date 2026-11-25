#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-a-wake.sh (queue item, 12.11.2026 ~11:55 UTC, ds-main-port; USER GO via the coordinator after the 11:46 reboot)
# WAKE TEST of the reseated PCIE7 riser (chip 5943 was lost from the bus at 11:14:01, 1.5 s after a 4-card selftest woke the cards from
# full idle). NO engine, NO server. 5 cycles, each: (b) wait until all 4 cards read <= 15 W and stay there >= 60 s (full idle, like
# 11:13:47-58 before the loss; bounded 240 s, a NOTE if never reached) -> (a) the WAKE: libr4d's known-good four-rank case
# r4d_selftest --case ar_twoshot_4rank_exact on all 4 cards (passed 10.11. / 11.11.: 17 ok, ~18 s) under docker -t (line-buffered),
# 90 s limit -> checks: journalctl -k since the cycle start for "device lost|AER|GPU reset|MES" (0 lines wanted), each card's sysfs
# (the 4 chip unique_ids present, power readable, link speed / width). STOP AT THE FIRST LOSS: log VIRHE with the chip + cycle, then
# PARK every queued item (queue/[0-9][0-9]-*.sh -> queue/paused-gpulost/) so nothing else starts on a broken card. A selftest FAIL /
# non-zero rc without a kernel or sysfs sign is a VIRHE too (stop + park). Cards are selected as a set by chip UUID check (all 4 used).
# Out runs/xdsm/wake1112 (st-c<N>.txt, kern-c<N>.txt, sysfs-c<N>.txt, idle-c<N>.txt). ~8 min. DRYRUN=1: prints, runs nothing.
# MOCK=1: the selftest replaced by a 3 s sleep + fake ok lines (no device); everything else real (sysfs / journal reads, idle wait).
set -u
T=$HOME/gpu/ds41/exl3-test; L=${XWK_LOG:-$T/runs/ds41-exl3.log}; Q=${XWK_Q:-$T/queue}; O=${XWK_OUT:-$T/runs/xdsm/wake1112}
IMAGE=${XWK_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
CASE=ar_twoshot_4rank_exact; LIBP=/opt/radiance/share/radiance/kernels/libr4d.so; STB=/opt/radiance/bin/r4d_selftest
CYC=${XWK_CYCLES:-5}; IDLEW=${XWK_IDLE_W:-15}; IDLES=${XWK_IDLE_S:-60}; IDLEMAX=${XWK_IDLE_MAX:-240}; TST=${XWK_TST:-90}
CHIPS="4d61f52cafe37588 5943fea82aeead65 27d4920317825fce dbba5c851cfef7b3"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
DRY=${DRYRUN:-0}; MOCK=${MOCK:-0}; TAG="RESULT xwake"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pw() {  # pw DEVDIR -> watts (integer) or "NA"
  local h; h=$(ls -d $1/hwmon/hwmon* 2>/dev/null | head -1)
  local v; v=$(cat $h/power1_average 2>/dev/null || cat $h/power1_input 2>/dev/null) || { echo NA; return; }
  [ -n "$v" ] && echo $((v / 1000000)) || echo NA
}
cards() {  # one line a card: pci uid watts link
  for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue
    echo "$(basename $(readlink -f $d)) $(cat $d/unique_id 2>/dev/null || echo NOUID) $(pw $d) $(cat $d/current_link_speed 2>/dev/null | tr -d ' ')x$(cat $d/current_link_width 2>/dev/null)"
  done | sort -u
}
park() {  # park every queued item (stop at the first loss)
  mkdir -p $Q/paused-gpulost; local n=0
  for f in $Q/[0-9][0-9]-*.sh; do [ -e "$f" ] || continue; mv "$f" $Q/paused-gpulost/ && n=$((n + 1)); done
  log "$TAG: PARKED $n queued item(s) -> queue/paused-gpulost/ (wake test failed; nothing re-runs until the coordinator says)"
  echo "$(date '+%F %T') ds-main-port wake item: PARKED $n queued item(s) -> queue/paused-gpulost/ after a wake-test failure" >> $T/runs/queue-changes.log
}
chk() {  # chk CYCLE SINCE -> 0 ok, 1 loss (logs VIRHE)
  local c=$1 since=$2 k miss="" bad="" line
  k=$(journalctl -k --since "$since" --no-pager -o short-precise 2>&1 | grep -iE "device lost|AER|GPU reset|MES " | grep -viE "AER: enabled|_OSC" )
  echo "$k" > $O/kern-c$c.txt
  cards > $O/sysfs-c$c.txt
  for u in $CHIPS; do
    line=$(grep " $u " $O/sysfs-c$c.txt)
    if [ -z "$line" ]; then miss="$miss $u"; elif echo "$line" | grep -q " NA "; then bad="$bad $u(power unreadable)"; fi
  done
  if [ -n "$k" ] || [ -n "$miss" ] || [ -n "$bad" ]; then
    log "$TAG: VIRHE cycle $c: CARD LOSS SIGN -- kernel lines $( [ -n "$k" ] && echo "$k" | wc -l || echo 0) (first: $(echo "$k" | head -1 | cut -c1-200)) | chips missing:${miss:- none} | unreadable:${bad:- none} | sysfs: $(tr '\n' ';' < $O/sysfs-c$c.txt | cut -c1-400)"
    return 1
  fi
  log "$TAG cycle $c checks: kernel 0 lines (device lost / AER / GPU reset / MES), 4 of 4 chips present, power readable | $(tr '\n' ';' < $O/sysfs-c$c.txt | cut -c1-400)"
  return 0
}
idle() {  # idle CYCLE: all 4 cards <= IDLEW W for >= IDLES s (bounded IDLEMAX)
  local c=$1 t0 held=0 mx w
  t0=$(date +%s); : > $O/idle-c$c.txt
  while :; do
    mx=0; for w in $(cards | awk '{print $3}'); do [ "$w" = NA ] && w=999; [ $w -gt $mx ] && mx=$w; done
    echo "$(date +%T) max_w $mx" >> $O/idle-c$c.txt
    if [ $mx -le $IDLEW ]; then held=$((held + 2)); else held=0; fi
    [ $held -ge $IDLES ] && { log "$TAG cycle $c idle: all 4 cards <= $IDLEW W for $held s (after $(( $(date +%s) - t0 )) s)"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge $IDLEMAX ] && { log "$TAG cycle $c idle: NOTE full idle not held $IDLES s within $IDLEMAX s (max now $mx W) -- waking anyway"; return 0; }
    sleep 2
  done
}
# ---- checks
miss=0
docker image inspect $IMAGE > /dev/null 2>&1 || { echo "  (check) image $IMAGE MISSING"; miss=$((miss + 1)); }
P=$(timeout 60 docker run --rm --network none --entrypoint sh $IMAGE -c "grep -c $CASE $STB; test -s $LIBP && echo LIB" 2>&1 | tr '\n' ' ')
case "$P" in [1-9]*LIB*) echo "  (check) image has r4d_selftest with case $CASE + libr4d.so" ;; *) echo "  (check) case $CASE / libr4d MISSING in the image ($P)"; miss=$((miss + 1)) ;; esac
journalctl -k -n 1 --no-pager > /dev/null 2>&1 && echo "  (check) journalctl -k readable" || { echo "  (check) journalctl -k NOT readable"; miss=$((miss + 1)); }
echo "  (check) cards now: $(cards | tr '\n' ';')"
[ "$(cards | grep -cE "$(echo $CHIPS | tr ' ' '|')")" = 4 ] || { echo "  (check) not all 4 chips present"; miss=$((miss + 1)); }
if [ "$DRY" = 1 ]; then
  for c in $(seq 1 $CYC); do
    echo "+ cycle $c: idle until all 4 <= $IDLEW W for $IDLES s (max $IDLEMAX s); timeout -k 15 $TST docker run --rm --init -t $DEV --entrypoint $STB $IMAGE $LIBP --case $CASE > $O/st-c$c.txt; journalctl -k since cycle start | grep device lost|AER|GPU reset|MES; sysfs 4 chips"
  done
  echo "DRYRUN: $miss MISSING (nothing started on a GPU, nothing written)"; exit 0
fi
[ $miss = 0 ] || { log "$TAG: VIRHE $miss check(s) failed -- not run"; park; exit 1; }
docker ps --format '{{.Names}}' | grep -qi radiance && { log "$TAG: VIRHE a radiance container is running -- not now"; exit 1; }
mkdir -p $O
log "$TAG: start (wake test of the reseated PCIE7 riser, chip 5943): $CYC cycles x [full idle <= $IDLEW W >= $IDLES s -> $CASE on 4 cards (docker -t, $TST s) -> kernel + sysfs checks]$([ $MOCK = 1 ] && echo ' MOCK (no device)') -> $O"
for c in $(seq 1 $CYC); do
  idle $c
  since=$(date '+%Y-%m-%d %H:%M:%S'); t0=$(date +%s)
  if [ $MOCK = 1 ]; then sleep 3; printf '  ok    mock a\r\n  ok    mock b\r\n' > $O/st-c$c.txt; rc=0
  else
    timeout -k 15 $TST docker run --rm --init --stop-timeout 10 --name xwake-st-$c -t $DEV --entrypoint $STB $IMAGE $LIBP --case $CASE > $O/st-c$c.txt 2>&1; rc=$?
    docker rm -f xwake-st-$c > /dev/null 2>&1 || true
  fi
  dt=$(( $(date +%s) - t0 )); nok=$(grep -c '^  ok  ' $O/st-c$c.txt); nf=$(grep -c 'FAIL' $O/st-c$c.txt)
  log "$TAG cycle $c wake: $CASE rc $rc, $nok ok, $nf FAIL, $dt s$({ [ $rc = 124 ] || [ $rc = 137 ]; } && echo " -- STOPPED by the limit; last line: $(tr -d '\r' < $O/st-c$c.txt | tail -1 | cut -c1-200)")"
  sleep 3
  chk $c "$since" || { park; log "$TAG: done -- LOSS in cycle $c ($O)"; exit 1; }
  if [ $rc != 0 ] || [ $nf -gt 0 ] || [ $nok = 0 ]; then
    log "$TAG: VIRHE cycle $c: selftest not clean (rc $rc, $nok ok, $nf FAIL) without a kernel / sysfs loss sign"; park; log "$TAG: done -- stopped in cycle $c ($O)"; exit 1
  fi
done
log "$TAG: PASS -- $CYC of $CYC wakes from full idle clean on all 4 cards (kernel 0 lines, 4 of 4 chips every cycle) -- $O"
