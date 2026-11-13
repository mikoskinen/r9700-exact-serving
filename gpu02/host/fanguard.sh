#!/bin/bash
# fanguard.sh (3.11.2026 night) -- memory-temperature guard for the flat fan curves set by fanflat.sh (the curve itself follows the hotspot
# only; 2.11. at a flat 2500 rpm the 03:00 memory reached 96 C). Every 10 s: a card whose memory >= 92 C (or hotspot >= 95 C) gets +2 %
# (~150 rpm) on its flat part, at most 40 %, then a 120 s hold for that card. Log: ~/gpu/od-changes.log + ~/gpu/fanguard.log.
# 4.11. (user picked option 1 to quieten the fans): it also lowers. A card above its base % (~/gpu/fan-base.txt, the user's chosen flat
# values) whose memory has stayed <= 85 C for 30 min, with no change of its own in those 30 min, gets -2 %, never below the base.
ST=$HOME/gpu/fan-state.txt; BASE=$HOME/gpu/fan-base.txt; LG=$HOME/gpu/fanguard.log; MAXP=40; COOLC=85; COOLS=1800
declare -A HOLD COOL LAST BP MISS
[ -r $BASE ] && while read -r b p _; do [ -n "$b" ] && BP[$b]=$p; done < $BASE
echo "$(date '+%F %T') fanguard start (mem >= 92 C or hotspot >= 95 C -> +2 %, max $MAXP %; mem <= $COOLC C for $((COOLS / 60)) min -> -2 % down to base:$(for b in "${!BP[@]}"; do printf ' %s=%s' $b ${BP[$b]}; done))" >> $LG
while [ ! -e $HOME/gpu/fanguard.stop ]; do
  now=$(date +%s)
  while read -r b P TOP; do
    [ -z "$b" ] && continue
    d=/sys/bus/pci/devices/0000:$b.0; hw=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1)
    # 12.11.: a card missing from the bus (or renamed by a slot move) must not end the guard for the others
    jr=$(cat $hw/temp2_input 2>/dev/null); mr=$(cat $hw/temp3_input 2>/dev/null)
    if [ -z "$hw" ] || [ -z "$jr" ] || [ -z "$mr" ]; then
      [ -z "${MISS[$b]:-}" ] && echo "$(date +%F %T) fanguard $b: no readable hwmon (card missing?) -- skipped" >> $LG; MISS[$b]=1; continue
    fi
    unset "MISS[$b]"; j=$(( jr / 1000 )); m=$(( mr / 1000 ))
    if [ $m -le $COOLC ]; then COOL[$b]=${COOL[$b]:-$now}; else unset "COOL[$b]"; fi
    if { [ $m -ge 92 ] || [ $j -ge 95 ]; } && [ $now -ge ${HOLD[$b]:-0} ] && [ $P -lt $MAXP ]; then
      n=$(( P + 2 )); [ $n -gt $MAXP ] && n=$MAXP
      $HOME/gpu/fanflat.sh $b=$n TOP=$TOP > /dev/null 2>&1
      HOLD[$b]=$(( now + 120 )); LAST[$b]=$now
      echo "$(date '+%F %T') fanguard $b: mem $m C hotspot $j C -> flat $P -> $n % ($(cat $hw/fan1_input) rpm before)" | tee -a $LG >> $HOME/gpu/od-changes.log
    elif [ -n "${BP[$b]:-}" ] && [ $P -gt ${BP[$b]} ] && [ -n "${COOL[$b]:-}" ] \
         && [ $(( now - ${COOL[$b]} )) -ge $COOLS ] && [ $(( now - ${LAST[$b]:-0} )) -ge $COOLS ]; then
      n=$(( P - 2 )); [ $n -lt ${BP[$b]} ] && n=${BP[$b]}
      $HOME/gpu/fanflat.sh $b=$n TOP=$TOP > /dev/null 2>&1
      LAST[$b]=$now; COOL[$b]=$now
      echo "$(date '+%F %T') fanguard $b: mem $m C hotspot $j C, <= $COOLC C for $((COOLS / 60)) min -> flat $P -> $n % (base ${BP[$b]}, $(cat $hw/fan1_input) rpm before)" | tee -a $LG >> $HOME/gpu/od-changes.log
    fi
  done < $ST
  sleep 10
done
echo "$(date '+%F %T') fanguard stop (fanguard.stop)" >> $LG
