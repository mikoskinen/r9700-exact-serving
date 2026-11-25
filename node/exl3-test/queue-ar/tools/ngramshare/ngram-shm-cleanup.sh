#!/bin/bash
# ngram-shm-cleanup.sh (12.11.2026, ds-main-port ngshare builder) -- frees /dev/shm/ds41ng (~103 GiB of RAM while the DS
# tables sit there): stops ngram-keeper.sh's holds and removes /dev/shm/ds41ng/ngram-*.tab and their .lock files.
# Run it before gpu02 produp (TODO.md) and before any SPEED start that must have the RAM.
#
#   ngram-shm-cleanup.sh          refuse unless no radiance container is running (docker ps: name or image matches
#                                 "radiance") AND no process but the keeper holds a lock on the files (/proc/locks)
#   DRYRUN=1 ngram-shm-cleanup.sh print what it would do, change nothing, log nothing
#
# Keeps the directory itself (empty; a later start without `ngram-keeper.sh prep` would otherwise get a root-owned one from
# docker). Never touches /dev/shm/ngram-*.tab (production's files). Logs to runs/queue-changes.log ("ngshare cleanup:").
# Env (tests only): NGSHARE_DIR, NGSHARE_RUN (as the keeper), NGSHARE_LOG (default runs/queue-changes.log).
set -u
SELF=$(readlink -f "$0"); HERE=$(dirname "$SELF")
DIR=${NGSHARE_DIR:-/dev/shm/ds41ng}; DIR=${DIR%/}
LOG=${NGSHARE_LOG:-$HOME/gpu/ds41/exl3-test/runs/queue-changes.log}
KEEP=$HERE/ngram-keeper.sh
DRY=${DRYRUN:-0}
case "$DIR" in /*) ;; *) echo "ngram-shm-cleanup: NGSHARE_DIR must be absolute"; exit 2 ;; esac
[ "$(readlink -m "$DIR")" = /dev/shm ] && { echo "ngram-shm-cleanup: refusing /dev/shm itself (production's n-gram files)"; exit 2; }

say() { echo "ngshare cleanup: $*"; [ "$DRY" = 1 ] || echo "$(date '+%F %T') ngshare cleanup: $*" >> "$LOG"; }
shmused() { df -B1 --output=used /dev/shm 2> /dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}'; }

# 1. no radiance container may run (a running engine may map the file; a starting one may be filling it)
if ! ps_out=$(docker ps --format '{{.Names}} {{.Image}}' 2>&1); then
  say "REFUSED: docker ps failed ($(echo "$ps_out" | head -1 | cut -c1-200)), so a running engine cannot be ruled out; nothing removed"; exit 1
fi
run=$(echo "$ps_out" | grep -i radiance)
if [ -n "$run" ]; then
  say "REFUSED: radiance container(s) running: $(echo "$run" | tr '\n' ';' | cut -c1-300) -- nothing removed ($DIR kept)"; exit 1
fi

# 2. no holder but the keeper (an engine outside docker, or one docker ps did not name)
files=(); for f in "$DIR"/ngram-*.tab; do [ -f "$f" ] && files+=("$f"); done
others=""
for f in "${files[@]}"; do
  a=($(stat -c '%Hd %Ld %i' "$f")); k=$(printf '%02x:%02x:%s' "${a[0]}" "${a[1]}" "${a[2]}")
  while read -r m pid; do
    [ -z "$pid" ] && continue
    [ "$(tr '\0' ' ' < /proc/$pid/cmdline 2> /dev/null)" = "flock -s -n -o -E 75 /dev/fd/9 sleep infinity " ] && continue
    others="$others $(basename "$f"):$m:$pid($(cat /proc/$pid/comm 2> /dev/null))"
  done < <(awk -v k="$k" '$2=="FLOCK" && $6==k {print $4, $5}' /proc/locks)
done
if [ -n "$others" ]; then say "REFUSED: other holder(s) of the files:$others -- nothing removed"; exit 1; fi

# 3. release the keeper, 4. remove the files
before=$(shmused)
if [ "$DRY" = 1 ]; then
  echo "+ $KEEP stop"
  for f in "${files[@]}"; do echo "+ rm -f $f $f.lock  ($(( $(stat -c '%b * %B' "$f") >> 20 )) MiB)"; done
  for l in "$DIR"/ngram-*.tab.lock; do [ -e "$l" ] && [ ! -e "${l%.lock}" ] && echo "+ rm -f $l  (orphan lock file)"; done
  exit 0
fi
kout=$(NGSHARE_DIR=$DIR "$KEEP" stop 2>&1 | tail -1 | sed 's/^stop: //; s/;.*//')
n=0; mib=0; names=""
for f in "${files[@]}"; do
  m=$(( $(stat -c '%b * %B' "$f" 2> /dev/null || echo 0) >> 20 ))
  rm -f "$f" "$f.lock" && { n=$((n + 1)); mib=$((mib + m)); names="$names $(basename "$f")"; }
done
for l in "$DIR"/ngram-*.tab.lock; do [ -e "$l" ] && rm -f "$l" && names="$names $(basename "$l")(orphan)"; done
say "removed $n table file(s), $((mib >> 10)) GiB ($mib MiB):${names:- none} from $DIR; keeper: $kout; /dev/shm used $before -> $(shmused)"
