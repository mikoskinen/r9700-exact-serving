#!/bin/bash
# Sets every Radeon AI PRO R9700 in the host to the power state of one results column.
# Usage: sudo run/power.sh 300   (card default 300 W cap, 0 mV, fans flat 90 % up to 85 C)
#        sudo run/power.sh 210   (210 W cap, -50 mV, fans flat 30 % up to 85 C)
# Both fan curves rise to the top value at 100 C (100 % at 300 W, 50 % at 210 W).
# The voltage offset needs amdgpu.ppfeaturemask=0xffffffff on the kernel command line.
# The settings last until the next reboot or driver reload.
set -u
case "${1:-}" in
  300) CAP=300000000 VO=0   FAN=90 TOP=100 ;;
  210) CAP=210000000 VO=-50 FAN=30 TOP=50 ;;
  *) echo "usage: $0 300|210" >&2; exit 1 ;;
esac
for d in /sys/class/drm/card*/device; do
  [ -e "$d/pp_od_clk_voltage" ] || continue
  h=$(ls -d "$d"/hwmon/hwmon* | head -1)
  rn=$(basename "$(ls -d "$d"/drm/renderD* | head -1)")
  exec 9</dev/dri/"$rn"    # keeps the card awake while its settings are written
  echo $CAP > "$h/power1_cap"
  for t in 1 2 3; do { echo "vo $VO" > "$d/pp_od_clk_voltage" && echo c > "$d/pp_od_clk_voltage"; } 2>/dev/null && break; sleep 2; done
  f=$d/gpu_od/fan_ctrl/fan_curve
  for t in 1 2 3; do
    { echo "0 25 $FAN" > "$f" && echo "1 50 $FAN" > "$f" && echo "2 70 $FAN" > "$f" && echo "3 85 $FAN" > "$f" &&
      echo "4 100 $TOP" > "$f" && echo c > "$f"; } 2>/dev/null && break; sleep 2
  done
  exec 9<&-
  echo "$(basename "$(readlink -f "$d")"): cap $(( $(cat "$h/power1_cap") / 1000000 )) W, offset $VO mV, fan $FAN % (top $TOP %)"
done
