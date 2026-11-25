# heatsoak.sh -- sourced by queue items (9.11.2026, queue-ar) for THERMAL LOGGING: thermnow / thermwin next to every prefill
# result (TESTING.md 9.11.: no soak time -- decode phases run before prefill after a (re)start, the mb window before the
# prefill window inside an A/B block, A B B A order; prefill is clock-bound at 210 W: 2786 MHz at 56 C vs 2630-2700 at
# 78-82 C, -2..-3 % over the first ~3 min after an idle start; decode does not move).
#   thermnow                     one line, every card now: pci:junction C/sclk MHz/power W (sysfs, read-only)
#   thermwin T0 T1               the window's full-power rows (power >= 195 W) of ~/gpu/temps/<day>.csv (templog.sh,
#                                10 s samples): per card median junction / sclk -> one line ("no full-power rows" if none)
#   heatsoak URL MBDIR [MIN_S] [MAX_S]
#                                DEPRECATED (user 9.11.: no added soak time) -- kept only so an old caller does not
#                                break; new items do not call it.
# Read-only on the cards (sysfs reads).
thermnow() {
  local d h out=""
  for d in /sys/class/drm/card*/device; do
    [ -e $d/unique_id ] || continue
    h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    out="$out $(basename $(readlink -f $d) | cut -c6-10):$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C/$(( $(cat $h/freq1_input 2>/dev/null || echo 0) / 1000000 ))MHz/$(( $(cat $h/power1_average 2>/dev/null || echo 0) / 1000000 ))W"
  done
  echo "${out# }"
}
thermmax() {   # the hottest card's junction now (C)
  local d h m=0 j
  for d in /sys/class/drm/card*/device; do
    [ -e $d/unique_id ] || continue
    h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); [ $j -gt $m ] && m=$j
  done
  echo $m
}
thermwin() {   # thermwin T0 T1 (epoch seconds)
  python3 - "$1" "$2" <<'PY' 2>/dev/null || echo "no temperature log"
import os, statistics, sys, time
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
days = {time.strftime("%Y%m%d", time.localtime(t)) for t in (t0, t1)}
rows = {}
for d in sorted(days):
    p = os.path.expanduser(f"~/gpu/temps/{d}.csv")
    if not os.path.exists(p): continue
    for l in open(p):
        f = l.strip().split(",")
        if len(f) < 7: continue
        try: t, pci, j, m, w, fan, s = float(f[0]), f[1], int(f[2]), int(f[3]), int(f[4]), f[5], int(f[6])
        except ValueError: continue
        if t0 <= t <= t1 and w >= 195: rows.setdefault(pci, []).append((j, s))
if not rows: print("no full-power rows"); sys.exit(0)
print(" ".join(f"{pci}:{statistics.median(x[0] for x in v):.0f}C/{statistics.median(x[1] for x in v):.0f}MHz(n{len(v)})" for pci, v in sorted(rows.items())))
PY
}
heatsoak() {   # heatsoak URL MBDIR [MIN_S] [MAX_S]
  local url=$1 mbdir=$2 mn=${3:-180} mx=${4:-480} t0 j0 j1 traj="" n=0
  if [ "${DRY:-0}" = 1 ]; then echo "+ heat-soak: mb c16 60 s windows on $url until >= ${mn} s and the hottest junction moves <= 2 C a window (max ${mx} s)"; return 0; fi
  t0=$(date +%s); j0=$(thermmax); traj="$j0"
  while :; do
    (cd $mbdir && DS41_URL=$url timeout 300 python3 mb.py --tag heatsoak --phases c16 --dur 60 --warm 10 > /dev/null 2>&1)
    n=$((n + 1)); j1=$(thermmax); traj="$traj>$j1"
    local el=$(( $(date +%s) - t0 ))
    if [ $el -ge $mn ] && [ $(( j1 - j0 )) -le 2 ] && [ $(( j0 - j1 )) -le 2 ]; then
      echo "heat-soak: $n windows, ${el} s, hottest junction $traj C (steady); now $(thermnow)"; return 0
    fi
    if [ $el -ge $mx ]; then
      echo "heat-soak: $n windows, ${el} s, hottest junction $traj C (NOT steady at the cap); now $(thermnow)"; return 1
    fi
    j0=$j1
  done
}
