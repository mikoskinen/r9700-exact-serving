#!/bin/bash
# ngram-keeper.sh (12.11.2026, ds-main-port ngshare builder) -- keeps the DeepSeek-V4.1 test starts' shared Engram n-gram
# tables (/dev/shm/ds41ng/ngram-<id>.tab, --ngram-placement ram=/ngram + ds41-ngshare-tp4.yaml) alive between sequential
# server starts, so a correctness-only start maps the ready file ("reused (probes match)") instead of copying ~103 GiB.
#
#   ngram-keeper.sh prep                make /dev/shm/ds41ng (owned by me; docker would make it root's) and show the room
#   ngram-keeper.sh start [--wait SEC] [--expect N]
#                                       hold LOCK_SH on every READY ngram-*.tab not held yet (idempotent); --wait: poll up to
#                                       SEC s until N files (default 1; the DS .rad has 2 tables) are ready and none is
#                                       half-filled
#   ngram-keeper.sh stop                release every hold (the files STAY; ngram-shm-cleanup.sh removes them)
#   ngram-keeper.sh status              files, ready state, every holder from /proc/locks, keeper pids
#
# WHY IT WORKS (Radiance core/format/table_share.cpp, radiance-int8 7e03384): every holder keeps LOCK_SH on the data file
# (:128); a leaving holder drops it (:232) and tries LOCK_EX|LOCK_NB 10 x 100 ms (:233-245); only the one that GETS LOCK_EX
# unlinks the lock file and the data file (:234-240). While this keeper holds LOCK_SH the leaving engine's LOCK_EX|LOCK_NB
# fails, so it closes without deleting (~0.9 s of tries per table at stop).
# WHY IT NEVER MAKES AN ENGINE WAIT: an engine's only blocking locks are LOCK_SH on the data file (:128, shared, compatible
# with ours) and LOCK_EX on <file>.lock (:168, the fill lock; the keeper never opens the lock file). The keeper locks a file
# only after its header says ready (classify, :44-75, via ngshare_hdr.py on the very file description it then locks), and
# with flock -n, so it never queues behind anyone either.
# THE HOLD: one long-lived `flock -s -n -o -E 75 /dev/fd/9 sleep infinity` per file, in its own session (setsid: pid = pgid,
# survives the caller and ssh hangups), pid file runs/ngshare/keeper-<id>.pid ("pid inode path"). The data file is opened
# on fd 9 WITHOUT O_CREAT (flock(1) itself would create a missing path); flock opens /dev/fd/9, the same inode even if it was
# unlinked meanwhile, so the keeper never makes a file. After the lock the inode's link count is checked, as the engine
# does (:119-147): a file deleted between open and lock is released at once.
#
# Env (tests only): NGSHARE_DIR (default /dev/shm/ds41ng), NGSHARE_RUN (default ~/gpu/ds41/exl3-test/runs/ngshare).
# Never /dev/shm itself: production's own table files live there (flashnext-prod-dp-*.yaml, /dev/shm:/ngram).
# No GPU, no docker, no sudo, no host settings.
set -u
SELF=$(readlink -f "$0"); HERE=$(dirname "$SELF")
DIR=${NGSHARE_DIR:-/dev/shm/ds41ng}
RUN=${NGSHARE_RUN:-$HOME/gpu/ds41/exl3-test/runs/ngshare}
KLOG=$RUN/keeper.log
HDR=$HERE/ngshare_hdr.py
HOLDCMD="flock -s -n -o -E 75 /dev/fd/9 sleep infinity"

die() { echo "ngram-keeper: $*" >&2; exit 2; }
case "$DIR" in /*) ;; *) die "NGSHARE_DIR must be absolute (got '$DIR')" ;; esac
DIR=${DIR%/}
[ "$(readlink -m "$DIR")" = /dev/shm ] && die "refusing /dev/shm itself: production's n-gram files live there"

klog() { mkdir -p "$RUN"; echo "$(date '+%F %T') $*" >> "$KLOG"; }

# /proc/locks names a file "MAJ:MIN:INO" with the device in hex (00:1b:56 for st_dev 0:27).
lockkey() { local a; a=($(stat -L -c '%Hd %Ld %i' "$1" 2>/dev/null)) || return 1; [ ${#a[@]} = 3 ] || return 1
            printf '%02x:%02x:%s\n' "${a[0]}" "${a[1]}" "${a[2]}"; }
# holders KEY -> "MODE PID" lines (granted locks only; a waiter's line carries "->" and is skipped)
holders() { awk -v k="$1" '$2=="FLOCK" && $6==k {print $4, $5}' /proc/locks; }
alive_keeper() { [ -r /proc/$1/cmdline ] && [ "$(tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null)" = "$HOLDCMD " ]; }
idof() { local b; b=$(basename "$1"); b=${b#ngram-}; echo "${b%.tab}"; }
pidfile() { echo "$RUN/keeper-$(idof "$1").pid"; }

# held F -> 0 if a live keeper holds the inode now at path F
held() { local pf p ino path cur; pf=$(pidfile "$1"); [ -r "$pf" ] || return 1
         read -r p ino path < "$pf" || return 1
         cur=$(stat -c %i "$1" 2>/dev/null) || return 1
         alive_keeper "$p" && [ "$path" = "$1" ] && [ "$ino" = "$cur" ]; }

hold() {   # hold F -> 0 held / 1 not
  local f=$1 id pid= sp rc=0 i got=0 spgone=0 nl ino cur pf hs
  id=$(idof "$f"); pf=$(pidfile "$f"); hs=$RUN/.spawn-$id
  mkdir -p "$RUN"; rm -f "$hs"
  setsid "$SELF" _hold "$f" "$id" "$hs" < /dev/null > /dev/null 2>> "$KLOG" &
  sp=$!   # = the holder's pid unless setsid had to fork; the holder writes its own pid to $hs just before exec flock
  for i in $(seq 1 50); do
    [ -z "$pid" ] && [ -s "$hs" ] && pid=$(cat "$hs")
    if [ -n "$pid" ] && awk -v p="$pid" '$2=="FLOCK" && $4=="READ" && $5==p {f=1} END {exit !f}' /proc/locks; then got=1; break; fi
    if [ $spgone = 0 ] && ! kill -0 "$sp" 2> /dev/null; then
      wait "$sp"; rc=$?; spgone=1
      [ $rc != 0 ] && break           # 66 / 65 / 75 from _hold or flock
    fi
    if [ $spgone = 1 ] && [ -n "$pid" ] && ! kill -0 "$pid" 2> /dev/null; then rc=unknown; break; fi
    sleep 0.1
  done
  rm -f "$hs"
  if [ $got != 1 ]; then
    [ -n "$pid" ] && kill -TERM -- -"$pid" 2> /dev/null
    kill -TERM -- -"$sp" 2> /dev/null
    case $rc in
      66) echo "  $f: gone before it could be opened -- nothing held" ;;
      65) echo "  $f: not ready on a second look ($(python3 -I "$HDR" "$f" 2>&1)) -- nothing held" ;;
      75) echo "  $f: busy (a leaving engine held LOCK_EX at that instant, table_share.cpp:234) -- nothing held; run start again" ;;
      *)  echo "  $f: no lock after 5 s (rc $rc) -- nothing held" ;;
    esac
    klog "keeper start: $f NOT held (rc $rc)"; return 1
  fi
  read -r nl ino < <(stat -L -c '%h %i' /proc/$pid/fd/9 2> /dev/null)
  cur=$(stat -c %i "$f" 2> /dev/null)
  if [ "${nl:-0}" = 0 ] || [ "${ino:-x}" != "${cur:-y}" ]; then   # the engine's own check, table_share.cpp:139-147
    kill -TERM -- -"$pid" 2> /dev/null
    echo "  $f: deleted or replaced while it was being locked -- released, nothing held"
    klog "keeper start: $f deleted/replaced during the lock, released"; return 1
  fi
  echo "$pid $ino $f" > "$pf"
  echo "  $f: HELD by keeper pid $pid (inode $ino, $(python3 -I "$HDR" "$f"))"
  klog "keeper start: held $f pid $pid inode $ino"
  return 0
}

scan() {   # -> globals READY PART OTHER (arrays of paths)
  READY=(); PART=(); OTHER=()
  local f s
  for f in "$DIR"/ngram-*.tab; do
    [ -f "$f" ] || continue
    s=$(python3 -I "$HDR" "$f" 2>&1)
    case $s in ready*) READY+=("$f") ;; fresh|partial) PART+=("$f") ;; *) OTHER+=("$f: $s") ;; esac
  done
}

cmd_prep() {
  [ -d "$DIR" ] || mkdir -p "$DIR" || die "cannot make $DIR"
  local u fs room
  u=$(stat -c %u "$DIR")
  [ "$u" = "$(id -u)" ] || die "$DIR belongs to uid $u, not $(id -u) (docker made it as root?): the engine (uid 1000) cannot create its file there, and it cannot be removed without its owner"
  [ -w "$DIR" ] || die "$DIR is not writable"
  fs=$(stat -f -c %T "$DIR"); [ "$fs" = tmpfs ] || echo "ngram-keeper: WARNING $DIR is on $fs, not tmpfs (the engine warns too: the table goes to disk)"
  room=$(df -B1 --output=avail "$DIR" | tail -1)
  echo "prep: $DIR ok (owner $(id -un), $fs), room $((room >> 30)) GiB, MemAvailable $(( $(awk '/^MemAvailable/{print $2}' /proc/meminfo) >> 20 )) GiB"
  [ $((room >> 30)) -ge 110 ] || echo "ngram-keeper: WARNING under 110 GiB free in $DIR (the DS tables take ~103 GiB)"
}

cmd_start() {
  local wait=0 expect=1 t_end f n_ok=0 n_fail=0
  while [ $# -gt 0 ]; do
    case $1 in
      --wait)   wait=${2:-}; case $wait in ''|*[!0-9]*) die "--wait takes seconds" ;; esac; shift 2 ;;
      --expect) expect=${2:-}; case $expect in ''|*[!0-9]*) die "--expect takes a count" ;; esac; shift 2 ;;
      *) die "start takes --wait SEC and --expect N (got '$1')" ;;
    esac
  done
  [ -d "$DIR" ] || { echo "start: no $DIR (run prep before the first start)"; return 1; }
  t_end=$(( $(date +%s) + wait ))
  while :; do
    scan
    if [ "$wait" -gt 0 ] && { [ ${#READY[@]} -lt "$expect" ] || [ ${#PART[@]} -gt 0 ]; } && [ "$(date +%s)" -lt "$t_end" ]; then sleep 1; continue; fi
    break
  done
  for f in "${PART[@]}"; do echo "  $f: not ready (being filled, or left half-filled by a killed engine) -- not touched"; done
  for f in "${OTHER[@]}"; do echo "  $f -- not touched"; done
  for f in "${READY[@]}"; do
    if held "$f"; then echo "  $f: already held (pid $(cut -d' ' -f1 "$(pidfile "$f")"))"; n_ok=$((n_ok + 1)); continue; fi
    if hold "$f"; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
  done
  echo "start: ${#READY[@]} ready (expected $expect), $n_ok held, $n_fail failed, ${#PART[@]} not ready, ${#OTHER[@]} other"
  [ $n_fail -gt 0 ] && return 3
  [ ${#READY[@]} -gt 0 ] || return 1
  return 0
}

cmd_stop() {
  local pf p ino path n=0 i
  for pf in "$RUN"/keeper-*.pid; do
    [ -e "$pf" ] || continue
    read -r p ino path < "$pf"
    if alive_keeper "$p"; then
      kill -TERM -- -"$p" 2> /dev/null
      for i in $(seq 1 20); do alive_keeper "$p" || break; sleep 0.1; done
      alive_keeper "$p" && kill -KILL -- -"$p" 2> /dev/null
      echo "  released $path (pid $p)"; klog "keeper stop: released $path pid $p"; n=$((n + 1))
    else
      echo "  stale pid file $pf (pid $p not a keeper) removed"; klog "keeper stop: stale $pf removed"
    fi
    rm -f "$pf"
  done
  echo "stop: $n hold(s) released; files stay in $DIR (ngram-shm-cleanup.sh removes them)"
}

cmd_status() {
  local f k s h pf p ino path any=0
  echo "dir $DIR: $( [ -d "$DIR" ] && df -h --output=used,avail "$DIR" | tail -1 | awk '{print "tmpfs used " $1 ", free " $2}' || echo missing ); MemAvailable $(( $(awk '/^MemAvailable/{print $2}' /proc/meminfo) >> 20 )) GiB"
  for f in "$DIR"/ngram-*.tab; do
    [ -f "$f" ] || continue; any=1
    s=$(python3 -I "$HDR" "$f" 2>&1); k=$(lockkey "$f")
    h=$(holders "$k" | while read -r m pid; do
          if alive_keeper "$pid"; then echo -n "$m:$pid(keeper) "; else echo -n "$m:$pid($(cat /proc/$pid/comm 2>/dev/null || echo ?)) "; fi; done)
    echo "  $(basename "$f")  $(( $(stat -c '%b * %B' "$f") >> 20 )) MiB in RAM  [$s]  holders: ${h:-none}$( [ -e "$f.lock" ] && echo '  (+ .lock)')"
  done
  [ $any = 1 ] || echo "  no ngram-*.tab"
  for pf in "$RUN"/keeper-*.pid; do
    [ -e "$pf" ] || continue
    read -r p ino path < "$pf"
    if alive_keeper "$p"; then echo "  keeper pid $p holds $path (inode $ino)"; else echo "  STALE pid file $pf (pid $p)"; fi
  done
}

case "${1:-}" in
  prep)   cmd_prep ;;
  start)  shift; cmd_start "$@" ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  _hold)  # internal: the setsid'd holder. fd 9 without O_CREAT; ready check on that description; then flock it (shared, -n).
          exec 9< "$2" || exit 66
          python3 -I "$HDR" /dev/fd/9 "$3" > /dev/null || exit 65
          echo $$ > "$4"
          exec $HOLDCMD ;;
  *) sed -n '2,13p' "$SELF" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
