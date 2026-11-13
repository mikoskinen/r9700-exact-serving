#!/bin/bash
# fanflat.sh BUS=P [BUS=P ...] [TOP=50] (3.11.2026 night; user: "kun tuotanto käynnistyy, säädä fan profile tasaiseksi ... rauhallinen,
# pärjäisikö 2500 tai 2700") -- per-card flat fan curve on gpu02 R9700s: P % up to 85 C hotspot, linear to TOP % at 100 C (manual curve
# overrides the acoustic control; render group can write gpu_od, no sudo). Example: fanflat.sh 03:00=30 83:00=30 86:00=27 c3:00=29.
# Keeps the per-card % in ~/gpu/fan-state.txt (fanguard.sh reads it). Live only: a reboot restores acoustic 3500/80/25 (od-apply).
TOP=50; ST=$HOME/gpu/fan-state.txt; touch $ST
for a in "$@"; do case $a in TOP=*) TOP=${a#TOP=};; esac; done
exec 8> $HOME/gpu/.fan.lock; flock 8
for a in "$@"; do
  case $a in TOP=*) continue;; esac
  b=${a%%=*}; P=${a#*=}; d=/sys/bus/pci/devices/0000:$b.0; f=$d/gpu_od/fan_ctrl/fan_curve
  [ -w $f ] || { echo "$b: no writable fan_curve"; continue; }
  rn=$(basename $(ls -d $d/drm/renderD* | head -1)); exec 9</dev/dri/$rn
  for try in 1 2 3; do
    { echo "0 25 $P" > $f && echo "1 50 $P" > $f && echo "2 70 $P" > $f && echo "3 85 $P" > $f && echo "4 100 $TOP" > $f && echo c > $f; } && break; sleep 2; done
  exec 9<&-
  grep -v "^$b " $ST > $ST.tmp; echo "$b $P $TOP" >> $ST.tmp; mv $ST.tmp $ST
  echo "$b: $(sed -n 2,6p $f | tr '\n' ' ')"
done
echo "$(date '+%F %T') fanflat $* (flat <=85C hotspot, TOP @100C) live (user 3.11.: tasainen, rauhallinen ~2500-2700 rpm)" >> $HOME/gpu/od-changes.log
