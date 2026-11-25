#!/bin/bash
# od-apply.sh: apply R9700 undervolt + fan settings to every R9700 (idempotent). 31.10.: + TR/gpu01 chips df20.., cabf.. at -50 mV; dbba.., 4d61.. at -50 mV with stock-ish fans 3500/80/25 (user 31.10.). Logs to ~mikoskinen/gpu/od-apply.log. 2.11.: + gpu02 cards 3-4 27d4.. (03:00), 5943.. (86:00) at -50 mV, 3500/80/25 (user).
# Settings follow the physical GPU chip, identified by its amdgpu unique_id (a silicon property: PCI address, slot,
# NUMA node and HIP index all change when a card is moved, also to another machine).
#   chip unique_id    VDDGFX_OFFSET  acoustic_target_rpm  fan_target_temp(hotspot)  fan_minimum_pwm
#   729aa419825fd82f  -50 mV         2750                 85 C                      40 %   hot card (83:00, NUMA1, dp-a)
#   05e91b7419b5bee6  -50 mV         2750                 85 C                      40 %   console card (05:00 on root 00:02.0 x16, NUMA0, dp-b)
#   any other chip    0 mV           3500                 80 C                      25 %   stock voltage + conservative fan, logged as UNKNOWN
# 16.10. tuning (case closed, cards apart): same settings on both cards for an even hum; full sustained load settles at
# ~2750 rpm / junction ~86 C / hot-card memory ~93 C, idle ~1850 rpm. -50 mV ran 2 x 10 min full load without GPU errors.
# Stock R9700: 2100 rpm acoustic target, 85 C, min PWM 25 %, 0 mV.
# OD_DRYRUN=1 prints the chip -> settings mapping and current values without writing anything (no root needed).
# RDNA4/SMU14 quirk: a commit that changes only the voltage silently reverts PMFW fan control to defaults
# (sysfs keeps showing cached fan values). So all values are staged first and committed ONCE per card.
LOG=${OD_LOG:-/home/mikoskinen/gpu/od-apply.log}
[ -n "$OD_DRYRUN" ] && LOG=/dev/stdout
declare -A VO_C=([729aa419825fd82f]=-50 [05e91b7419b5bee6]=-50 [df207ee75b0e876f]=-50 [cabfff1e09a35eb4]=-50 [dbba5c851cfef7b3]=-50 [4d61f52cafe37588]=-50 [27d4920317825fce]=-50 [5943fea82aeead65]=-50)
declare -A RPM_C=([729aa419825fd82f]=2750 [05e91b7419b5bee6]=2750 [df207ee75b0e876f]=2750 [cabfff1e09a35eb4]=2750 [dbba5c851cfef7b3]=3500 [4d61f52cafe37588]=3500 [27d4920317825fce]=3500 [5943fea82aeead65]=3500)
declare -A TT_C=([729aa419825fd82f]=85 [05e91b7419b5bee6]=85 [df207ee75b0e876f]=85 [cabfff1e09a35eb4]=85 [dbba5c851cfef7b3]=80 [4d61f52cafe37588]=80 [27d4920317825fce]=80 [5943fea82aeead65]=80)
declare -A MP_C=([729aa419825fd82f]=40 [05e91b7419b5bee6]=40 [df207ee75b0e876f]=40 [cabfff1e09a35eb4]=40 [dbba5c851cfef7b3]=25 [4d61f52cafe37588]=25 [27d4920317825fce]=25 [5943fea82aeead65]=25)
DEF_VO=0 DEF_RPM=3500 DEF_TT=80 DEF_MP=25
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
# wait (up to 120 s) for both cards to appear before resolving addresses
for i in $(seq 1 60); do [ "$(grep -l 0x7551 /sys/bus/pci/devices/*/device 2>/dev/null | wc -l)" -ge ${OD_NCARDS:-1} ] && break; sleep 2; done
rc=0
for d in /sys/bus/pci/devices/*; do
  [ "$(cat $d/vendor 2>/dev/null)" = 0x1002 ] && [ "$(cat $d/device 2>/dev/null)" = 0x7551 ] || continue
  pci=$(basename $d); f=$d/gpu_od/fan_ctrl
  for i in $(seq 1 60); do [ -w $d/pp_od_clk_voltage ] && [ -e $f/fan_target_temperature ] && [ -s $d/unique_id ] && break; sleep 2; done
  if [ ! -w $d/pp_od_clk_voltage ]; then log "$pci: pp_od_clk_voltage not available/writable"; rc=1; continue; fi
  uid=$(cat $d/unique_id 2>/dev/null)
  if [ -n "$uid" ] && [ -n "${VO_C[$uid]}" ]; then vo=${VO_C[$uid]}; rpm=${RPM_C[$uid]}; tt=${TT_C[$uid]}; mp=${MP_C[$uid]}; tag=known
  else vo=$DEF_VO; rpm=$DEF_RPM; tt=$DEF_TT; mp=$DEF_MP; tag=UNKNOWN; fi
  log "$pci: chip ${uid:-?} ($tag, numa $(cat $d/numa_node)) -> vo ${vo}mV rpm $rpm ttemp $tt minpwm $mp"
  if [ -n "$OD_DRYRUN" ]; then
    log "$pci: DRYRUN, now: vo $(grep -A1 OD_VDDGFX_OFFSET: $d/pp_od_clk_voltage | tail -1) rpm $(sed -n 2p $f/acoustic_target_rpm_threshold) ttemp $(sed -n 2p $f/fan_target_temperature) minpwm $(sed -n 2p $f/fan_minimum_pwm)"
    continue
  fi
  # Runtime PM (BACO) loses the OD table on resume ("Failed to upload overdrive table!"), so keep the
  # card awake permanently. Writable only as root (the boot service); udev rule 99-r9700-runpm does the same.
  if [ -w $d/power/control ] && [ "$(cat $d/power/control)" != on ]; then echo on > $d/power/control && log "$pci: runtime PM disabled (power/control=on)"; fi
  rn=$(basename $(ls -d $d/drm/renderD* 2>/dev/null | head -1) 2>/dev/null)
  [ -n "$rn" ] && exec 9</dev/dri/$rn        # wake the card (runtime PM) and keep it awake while writing
  ok=0
  for try in 1 2 3 4 5; do
    { echo "vo $vo" > $d/pp_od_clk_voltage &&
      echo "$rpm" > $f/acoustic_target_rpm_threshold &&
      echo "$tt" > $f/fan_target_temperature &&
      echo "$mp" > $f/fan_minimum_pwm &&
      echo c > $d/pp_od_clk_voltage; } 2>>"$LOG" && { ok=1; break; }
    sleep 3
  done
  exec 9<&-
  for pc in $d/hwmon/hwmon*/power1_cap; do [ -w $pc ] && [ "$(cat $pc)" != 210000000 ] && echo 210000000 > $pc && log "$pci: power cap restored to 210 W"; done
  cur="vo $(grep -A1 OD_VDDGFX_OFFSET: $d/pp_od_clk_voltage | tail -1) rpm $(sed -n 2p $f/acoustic_target_rpm_threshold) ttemp $(sed -n 2p $f/fan_target_temperature) minpwm $(sed -n 2p $f/fan_minimum_pwm)"
  cur="$cur runpm $(cat $d/power/control)"
  if [ $ok = 1 ]; then log "$pci: applied -> $cur"; else log "$pci: FAILED (now: $cur)"; rc=1; fi
done
exit $rc
