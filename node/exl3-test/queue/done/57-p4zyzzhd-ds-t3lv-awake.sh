#!/bin/bash
# 105-ds-t3lv-awake.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, TP4, MML 524288; ~2 h for the four arms, ~28
# min an arm) -- 85b3's DeepSeek-V4.1 serve A/B (the packed compressed latent, t3lv, on the v3 tuned3l image) re-run with
# the cards kept out of the slow state by the gpu_metrics poller, BOTH sides in this window: t3 and t3lv at 2.9 and 2.25 bpw.
# WHY: 101 (9.11.) -- the R9700s fall into a sporadic slow state (memory-heavy kernels 2-3x slower from ~16 rows) unless
# the host polls each card's gpu_metrics; every DS serve run before the window-wide poller (104, 57-p4zyzzd) ran without
# it and without stamps, so its state is unknown -- 85b's t3l vs 57-p1's t3 included (t3l: 29p guard 22.95 vs 20.62 ms, c4
# 77.2 vs 80.1 tok/s, c8 95.6 vs 98.1; 225p guard 19.43 vs 15.93, c4 113.3 vs 122.6, c8 143.2 vs 154.0). Here both sides run
# under the poller, so the pair compares like states, and their absolute numbers against 57-p1 / 85b show whether DS moved.
# = 85b3-lat-serve.sh beside this file (item 79's body: gate, modes, warm-up answers vs stock, i1 copy, deep answer, probe,
# mb c4 / c8 / pf / turns, cc_sim u4 fixed-work trace-seeded, i2 copy, per-phase /stats; t3lv vs t3 cmparm / ccpair /
# copies / deep identity) with ARMS (X105_ARMS, default "29:t3 29:t3lv 225:t3 225:t3lv") and X79_SFX=pa (x79-29pa-*,
# x79-225pa-*: 57-p1's x79-29p-t3 and 85b / 85b3's rows stay), X79_REF=t3 (this run's t3). Both arms on X79_T3_IMAGE
# (radiance-local:ds41-exl3-117-641197d8-fa0e9580-tuned3l-hd4a11cab): t3 = RADIANCE_DS41_LATENT unset on it (the v3 libr4d
# leaves every bf16 kernel unchanged; only the packed latent forms differ). 85b3's own gate holds: runs only after a clean
# 85a3 ($LATB_OUT/SUMMARY.txt "85a CLEAN"; X85_FORCE=1 overrides).
# POLLER: the window-wide keepawake (104: runs/keepawake.pid) is checked alive at the start and the end and logged; if it
# is not running, this item starts its own (tools/keepawake.py gpu_metrics, X105_KA_MS or tools/pick103.sh on 103's table,
# else 5 ms; pid runs/keepawake-105.pid, log ~/gpu/watch/keepawake-105.log) and stops it at the end and in an EXIT trap.
# DRYRUN=1 bash 105-ds-t3lv-awake.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log
TL=; for f in "$H/tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/keepawake.py" ] && { TL=$f; break; }; done
B85=; for f in "$H/85b3-lat-serve.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/85b3-lat-serve.sh"; do [ -e "$f" ] && { B85=$f; break; }; done
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
kaalive() {   # kaalive PIDFILE: 0 when its pid runs keepawake.py
  local p; [ -s "$1" ] || return 1; p=$(cat "$1"); kill -0 "$p" 2>/dev/null && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q keepawake.py
}
kastate() { local p; for p in $R/keepawake.pid $R/keepawake-105.pid; do kaalive $p && { echo "pid $(cat $p) ($(basename $p)): $(tr '\0' ' ' < /proc/$(cat $p)/cmdline | grep -oE -- '--interval-ms [0-9.]+|--node [a-z_0-9]+' | tr '\n' ' ')"; return 0; }; done; echo "none alive"; return 1; }
OWN=0
stopown() { [ "$OWN" = 1 ] && kaalive $R/keepawake-105.pid && { kill $(cat $R/keepawake-105.pid) 2>/dev/null; log "RESULT x79 105 keepawake: own poller stopped (pid $(cat $R/keepawake-105.pid))"; }; OWN=0; }
[ -n "$B85" ] || { log "RESULT x79 105: VIRHE 85b3-lat-serve.sh not found beside $H"; exit 1; }
[ -n "$TL" ] || { log "RESULT x79 105: VIRHE tools/keepawake.py not found"; exit 1; }
if [ "$DRY" = 1 ]; then
  echo "+ check $R/keepawake.pid alive (keepawake.py); if not: setsid python3 -I $TL/keepawake.py --interval-ms \${X105_KA_MS:-pick103} --node gpu_metrics --cards all --log ~/gpu/watch/keepawake-105.log & -> $R/keepawake-105.pid; trap: stop it"
else
  if st=$(kastate); then
    log "RESULT x79 105 keepawake at start: window-wide poller alive -- $st"
  else
    ms=${X105_KA_MS:-$( [ -x $TL/pick103.sh ] && $TL/pick103.sh $R/fn4/103-table.txt 2>/dev/null || echo 5)}
    mkdir -p $HOME/gpu/watch
    setsid nohup python3 -I $TL/keepawake.py --interval-ms $ms --node gpu_metrics --cards all --log $HOME/gpu/watch/keepawake-105.log > /dev/null 2>&1 < /dev/null &
    echo $! > $R/keepawake-105.pid; OWN=1; sleep 2
    trap 'stopown' EXIT
    trap 'exit 143' TERM INT
    if kaalive $R/keepawake-105.pid; then log "RESULT x79 105 keepawake at start: no window-wide poller -> own poller started at $ms ms (pid $(cat $R/keepawake-105.pid), log ~/gpu/watch/keepawake-105.log)"
    else log "RESULT x79 105: VIRHE the own keepawake did not start -- not running the arms without it"; exit 1; fi
  fi
fi
log "RESULT x79 105-ds-t3lv-awake: start ARMS [${X105_ARMS:-29:t3 29:t3lv 225:t3 225:t3lv}] X79_SFX=pa X79_REF=t3 via $B85 | refs (no poller, state unknown): 57-p1 t3 29p guard 20.62 c4 80.1 c8 98.1 / 225p guard 15.93 c4 122.6 c8 154.0; 85b t3l 29p 22.95 / 77.2 / 95.6, 225p 19.43 / 113.3 / 143.2"
ARMS="${X105_ARMS:-29:t3 29:t3lv 225:t3 225:t3lv}" X79_SFX=${X105_SFX:-pa} X79_REF=t3 DRYRUN=$DRY bash $B85; rc=$?
[ "$DRY" = 1 ] || log "RESULT x79 105 keepawake at end: $(kastate) | 85b3 body exit $rc"
stopown
log "RESULT x79 105-ds-t3lv-awake: done (exit $rc)"
exit $rc
