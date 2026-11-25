# attach.sh (12.11.2026, queue-ar/dsmain/attach, ds-main-port ATTACH builder) -- OPT-IN reuse of a running DS test server between
# queue items with the same start signature (saves the ~232-247 s start). Functions only; SOURCED by queue/lib-pr266.sh and by
# queue-ar/dsmain/compose-guard.sh ONLY when ATTACH=1, KEEP=1 or SIGONLY=1 is set (compose-guard also when a record
# runs/server.sig / runs/server.keep exists). Nothing set and no record = today's code, unchanged. `bash attach.sh keeper ...` is the
# background keeper (below); `bash attach.sh selftest` checks the helpers.
#
# SIGNATURE (attachsig.py sig): sha256 of the image ID + serve_args + the yaml's / extends file's md5 + the yaml's ${VAR}s (MODEL,
#   MODELDIR, PROFEVERY ...; not STATSDIR / XS2_ENVFILE / IMAGE / secrets) + every exported RADIANCE_ / R4D_ (/ HSA_ ...) word + (real
#   start) env-file lines the shell does not carry. fn4 switches are not in it. STATSDIR is not either: a start with /stats writers
#   (BITRACE, MISS_LOG, AQL stamps ...) attaches only to a server with the same STATSDIR.
# RECORDS (ATTACH_STATE, default runs/): server.sig (sig cid= tag0= item0= t0= healthy_s= owner= item= statsdir= statsw= image=),
#   server.sig.txt (the hashed lines, for a diff), server.fn4base.json (GET /admin/fn4 right after the healthy start), server.keep
#   (next= sig= cid= owner= t= logpos= [claimed=]), attach-keeper.log. RESULT lines: "RESULT attach ..." in runs/ds41-exl3.log.
# FLOW
#   restart() ATTACH=1 (its FIRST call in a body only; a second call is a real restart, e.g. gate()'s slow-state retry): our container
#     up + record cid = it + not started by this item + /health 200 + idle (/stats running 0 / waiting 0, twice) + this start's
#     signature = the record's (+ the STATSDIR rule) -> the fn4 switches that differ from the start snapshot POSTed back (+ epoch_bump,
#     nudge, read back <= 30 s; the body's own pin / arm comes after) -> POST /reset_prefix_cache -> read back + idle ->
#     "RESULT attach TAG: attached to CID (sig XXXXXXXX), saved ~N s"; start-TAG.log / up-TAG.txt written as a start writes them;
#     exports ATTACH_ATTACHED=1, ATTACH_EPOCH0 (the fn4 epoch now; a body's "epoch 0 at start" check reads it), ATTACH_SINCE (UTC).
#     Anything else: our server stopped gracefully (docker stop -t 60, rm) and today's start runs. After a healthy start (ATTACH=1
#     or KEEP=1): the record is written.
#   down() KEEP=1: leaves the server up ONLY when: it is ours + this item started / attached it + /health 200 + idle + no
#     " RESULT ...VIRHE" line in the log since the item began (KEEP_VIRHE_OK=1 ignores them) + no runs/sweep.stop + the next queue
#     item (ls queue/[0-9][0-9]-*.sh, C and en_US order agreeing, head -1) has a "# ATTACH-SIG:" line + its SIGONLY pass (env -i,
#     SIGONLY=1 DRYRUN=1, docker / curl guard of sigonly-bin/ first on PATH) prints "ATTACH-SIG <sig> ATTACH=1" with the same sig.
#     Else `docker rm -f` as today. Repeat down() calls reuse the decision.
#   cguard() (the wrapper, before the body): the first call sets ATTACH_ITEM / ATTACH_RUNID / ATTACH_LOGPOS and the wrapper's EXIT
#     trap (att_epilogue). Our server kept for THIS item (ATTACH=1) -> claimed (ATTACH_CLAIMED_CN exported: the body's cards-free
#     check and cguard's wait skip it); kept for another item, or this item is not ATTACH=1 -> stopped gracefully; ours but not
#     kept -> claimed only when this item's own SIGONLY signature matches (ATTACH=1), else stopped. Not ours -> today's logic.
#   wrapper exit (att_epilogue): our server still up and not kept by this item for the CURRENT queue head -> stopped gracefully;
#     kept -> the keeper (setsid) watches: claimed -> done; another item starts / the named one exits unclaimed / sweep.stop /
#     KEEP_MAXWAIT (300 s) -> stopped. Non-attach items (Flash-Next, plain DS) therefore never find our server up.
# OPT IN (item wrapper): a line "# ATTACH-SIG: SIGONLY=1 DRYRUN=1 bash <item> prints the start signature (attach/attach.sh)" and
#   `export ATTACH=${ATTACH:-1} KEEP=${KEEP:-1}` before the cguard line; the body must be patched (attach/patch-body.sh: the
#   cards-free check skips ATTACH_CLAIMED_CN; swnote's "epoch 0 at start" reads ATTACH_EPOCH0). Knobs: KEEP_MAXWAIT (300),
#   KEEP_VIRHE_OK (0), ATTACH_SIGTIMEOUT (180), ATTACH_STATE / ATTACH_QDIR / ATTACH_LOG / ATTACH_T (mock runs).
ATT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATT_HOME=${ATTACH_T:-$HOME/gpu/ds41/exl3-test}
ATT_ST=${ATTACH_STATE:-$ATT_HOME/runs}
ATT_QD=${ATTACH_QDIR:-$ATT_HOME/queue}
ATT_LG=${ATTACH_LOG:-$ATT_HOME/runs/ds41-exl3.log}
ATT_CN=${ATTACH_CN:-radiance-ds41-exl3}
ATT_URL=${ATTACH_URL:-http://127.0.0.1:8090}
ATT_PY="python3 -I -B $ATT_DIR/attachsig.py"

att_now() { date '+%F %T'; }
att_log() { echo "$(att_now) RESULT attach $*" | tee -a "$ATT_LG"; }
att_put() { printf '%s\n' "$2" > "$1.tmp.$$" && mv -f "$1.tmp.$$" "$1"; }      # atomic; replaces (never writes through) a symlink
att_field() { [ -s "$1" ] || return 0; sed -n 1p "$1" | tr ' ' '\n' | sed -n "s/^$2=//p" | head -1; }
att_sig0() { [ -s "$ATT_ST/server.sig" ] && sed -n 1p "$ATT_ST/server.sig" | awk '{print $1}'; }
att_clear() { rm -f "$ATT_ST/server.sig" "$ATT_ST/server.sig.txt" "$ATT_ST/server.fn4base.json" "$ATT_ST/server.keep"; }
att_cst() { docker inspect -f '{{.State.Status}} {{.Id}}' "$ATT_CN" 2>/dev/null | head -1; }     # "running <id>" | "exited <id>" | ""
att_code() { curl -s -m "${2:-5}" -o /dev/null -w '%{http_code}' "$ATT_URL$1" 2>/dev/null; }
att_logsize() { stat -c %s "$ATT_LG" 2>/dev/null || echo 0; }
att_nudge() { curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{"prompt": "Hello", "n_predict": 4, "temperature": 0}' "$ATT_URL/completion" > /dev/null 2>&1; }
att_idle() {   # -> "IDLE" (0) | "BUSY ..." / "UNKNOWN ..." (1): /stats running 0 and waiting 0, read twice 1 s apart
  local i s
  for i in 1 2; do
    s=$(curl -s -m 10 "$ATT_URL/stats" 2>/dev/null | $ATT_PY idle - 2>/dev/null)
    [ "$s" = IDLE ] || { echo "${s:-UNKNOWN no /stats}"; return 1; }
    [ $i = 2 ] || sleep 1
  done
  echo IDLE
}
att_sig() {   # att_sig real|predict ARGS [TEXTOUT] -> "<sha256> <stats writers | ->" (the shell's IMAGE, Y, XS2_ENVFILE, MODEL ...)
  local id
  id=$(docker image inspect -f '{{.Id}}' "${IMAGE:-}" 2>/dev/null | head -1)
  [ -n "$id" ] || { echo "NOIMAGE -"; return 1; }
  if [ "$1" = real ] && [ -n "${XS2_ENVFILE:-}" ]; then
    $ATT_PY sig --yaml "${Y:-}" --image-id "$id" --args "$2" --envfile "$XS2_ENVFILE" ${3:+--text "$3"}
  else
    $ATT_PY sig --yaml "${Y:-}" --image-id "$id" --args "$2" ${3:+--text "$3"}
  fi
}
att_sigonly_print() { local s; s=$(att_sig predict "$1"); echo "ATTACH-SIG ${s%% *} ATTACH=${ATTACH:-0} KEEP=${KEEP:-0} statsw=${s#* }"; }
att_sigdiff() {   # the first differing signature lines, server -> this start (from the --text files; secret names are masked there)
  [ -s "$ATT_ST/server.sig.txt" ] && [ -s "$1" ] || { echo "no text to compare"; return 0; }
  $ATT_PY sigdiff "$ATT_ST/server.sig.txt" "$1" 2>&1 | cut -c1-400
}
att_short() { case "$1" in mock*) echo "${1: -8}" ;; *) echo "${1:0:12}" ;; esac; }   # a container id for a log line (the mock's ids differ at the end)
att_stop() {   # att_stop WHY: our server down gracefully (the engine releases the cards on SIGTERM; stop_grace_period 60 s), records gone
  local c id
  c=$(att_cst); id=${c#* }
  docker stop -t 60 "$ATT_CN" > /dev/null 2>&1; docker rm -f "$ATT_CN" > /dev/null 2>&1
  att_clear
  att_log "$1: our server $(att_short "$id") stopped (docker stop -t 60, rm)"
}
att_next() {   # the next queue item's name as chain-after would pick it; nothing (rc 1, why on stdout) when C / en_US order disagree
  local c e
  c=$(ls "$ATT_QD"/[0-9][0-9]-*.sh 2>/dev/null | LC_ALL=C sort | head -1)
  e=$(ls "$ATT_QD"/[0-9][0-9]-*.sh 2>/dev/null | LC_ALL=en_US.UTF-8 sort 2>/dev/null | head -1)
  [ -n "$c" ] || { echo "queue empty"; return 1; }
  [ "$c" = "$e" ] || { echo "C / en_US order disagree ($(basename "$c") / $(basename "$e"))"; return 1; }
  basename "$c"
}
att_sigonly() {   # att_sigonly ITEM -> "ATTACH-SIG <sig> ATTACH=x KEEP=y statsw=..." from a SIGONLY=1 DRYRUN=1 pass of ITEM (or nothing)
  local f=$1 v ev=()
  for v in HOME USER LOGNAME SHELL LANG $(compgen -e | grep -E '^(MOCKRAD_[A-Z0-9_]+|PYTHONDONTWRITEBYTECODE)$'); do
    [ -n "${!v+x}" ] && ev+=("$v=${!v}"); done
  ev+=("PATH=$ATT_DIR/sigonly-bin:$PATH")
  (cd "$ATT_HOME" 2>/dev/null; env -i "${ev[@]}" SIGONLY=1 DRYRUN=1 timeout "${ATTACH_SIGTIMEOUT:-180}" bash "$f" 2>/dev/null | grep -m1 '^ATTACH-SIG ')
}
att_virhe() {   # " RESULT ...VIRHE" lines since this item began (a body's "| no VIRHE line" summary is not one)
  [ -n "${ATTACH_LOGPOS:-}" ] || { echo "?"; return 0; }
  tail -c +$((ATTACH_LOGPOS + 1)) "$ATT_LG" 2>/dev/null | grep ' RESULT .*VIRHE' | grep -vc 'no VIRHE line'; }

att_record() {   # att_record TAG ARGS HEALTHY_S: after a healthy start (ATTACH=1 or KEEP=1)
  local tag=$1 a=$2 hs=$3 c id s
  c=$(att_cst); id=${c#* }
  [ "${c%% *}" = running ] || return 0
  s=$(att_sig real "$a" "$ATT_ST/server.sig.txt.tmp.$$") || { att_clear; att_log "record $tag: no signature ($s) -- this server is not attachable"; return 0; }
  curl -s -m 10 "$ATT_URL/admin/fn4" > "$ATT_ST/server.fn4base.json.tmp.$$" 2>/dev/null
  if ! $ATT_PY fn4ok "$ATT_ST/server.fn4base.json.tmp.$$"; then
    rm -f "$ATT_ST/server.fn4base.json.tmp.$$" "$ATT_ST/server.sig.txt.tmp.$$"; att_clear
    att_log "record $tag: GET /admin/fn4 unreadable -- this server is not attachable"; return 0; fi
  mv -f "$ATT_ST/server.fn4base.json.tmp.$$" "$ATT_ST/server.fn4base.json"; mv -f "$ATT_ST/server.sig.txt.tmp.$$" "$ATT_ST/server.sig.txt"
  rm -f "$ATT_ST/server.keep"
  att_put "$ATT_ST/server.sig" "${s%% *} cid=$id tag0=$tag item0=${ATTACH_ITEM:-manual} t0=$(date +%s) healthy_s=$hs owner=${ATTACH_RUNID:-manual-$$} item=${ATTACH_ITEM:-manual} statsdir=${STATSDIR:-} statsw=${s#* } image=${IMAGE:-}"
  att_log "record $tag: server $(att_short "$id") sig ${s:0:8} (healthy in $hs s; $(grep -c '' "$ATT_ST/server.sig.txt") signature lines; fn4 start snapshot epoch $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%s, %d switches" % (d.get("epoch"), len(d["switches"])))' "$ATT_ST/server.fn4base.json" 2>/dev/null))"
}

att_fn4reset() {   # -> 0 + a summary: the switches back at the start snapshot (only the ones that differ), epoch bumped, read back
  local cur d code i ck
  cur=$(curl -s -m 10 "$ATT_URL/admin/fn4" 2>/dev/null)
  d=$(echo "$cur" | $ATT_PY fn4diff "$ATT_ST/server.fn4base.json" - 2>/dev/null) || { echo "fn4diff failed (GET /admin/fn4 '${cur:0:80}')"; return 1; }
  if [ "$d" = "{}" ]; then
    ck=$(echo "$cur" | $ATT_PY fn4check "$ATT_ST/server.fn4base.json" - 2>/dev/null)
    case "$ck" in OK*) echo "fn4: no switch differed from the start snapshot"; return 0 ;; esac
  fi
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$($ATT_PY bump "$d")" "$ATT_URL/admin/fn4" 2>/dev/null)
  [ "$code" = 200 ] || { echo "POST /admin/fn4 $d -> HTTP $code"; return 1; }
  for i in $(seq 1 30); do
    att_nudge
    ck=$(curl -s -m 10 "$ATT_URL/admin/fn4" 2>/dev/null | $ATT_PY fn4check "$ATT_ST/server.fn4base.json" - 2>/dev/null)
    case "$ck" in OK*) echo "fn4: $($ATT_PY keys "$d") set back to the start snapshot + epoch_bump (read back after $i nudge(s))"; return 0 ;; esac
    sleep 1
  done
  echo "fn4 read back failed after 30 s: ${ck:0:300}"; return 1
}

att_try() {   # att_try TAG ARGS -> 0 = attached (restart() returns healthy), 1 = start as today
  local tag=$1 a=$2 c id rec=$ATT_ST/server.sig why="" st="" s="" fr pc ck ep hs t0 t1 tx
  [ -z "${ATT_TRIED:-}" ] || return 1
  ATT_TRIED=1; t0=$(date +%s)
  c=$(att_cst); id=${c#* }
  if [ "${c%% *}" != running ]; then att_clear; att_log "$tag: no server up -> start"; return 1; fi
  [ -s "$rec" ] || { att_log "$tag: $ATT_CN $(att_short "$id") up without a record (not started by an ATTACH / KEEP item) -> today's restart"; return 1; }
  [ "$(att_field "$rec" cid)" = "$id" ] || why="record cid $(att_short "$(att_field "$rec" cid)") != running $(att_short "$id")"
  [ -n "$why" ] || [ "$(att_field "$rec" owner)" != "${ATTACH_RUNID:-manual-$$}" ] || why="this item started it (a second restart is a real one)"
  [ -n "$why" ] || { st=$(att_code /health); [ "$st" = 200 ]; } || why="/health $st"
  [ -n "$why" ] || st=$(att_idle) || why="not idle ($st)"
  tx=$ATT_ST/.sig-$tag.txt.$$
  if [ -z "$why" ]; then
    s=$(att_sig real "$a" "$tx")
    [ "${s%% *}" = "$(att_sig0)" ] || why="signature ${s:0:8} != server $(att_sig0 | cut -c1-8) ($(att_sigdiff "$tx"))"
  fi
  rm -f "$tx"
  if [ -z "$why" ] && [ "${s#* }" != - ] && [ "$(att_field "$rec" statsdir)" != "${STATSDIR:-}" ]; then
    why="the start writes under /stats (${s#* }) and STATSDIR differs ($(att_field "$rec" statsdir) vs ${STATSDIR:-})"; fi
  [ -z "$why" ] || { att_stop "$tag: no attach: $why"; return 1; }
  fr=$(att_fn4reset) || { att_stop "$tag: no attach: $fr"; return 1; }
  pc=$(curl -s -m 60 -w ' HTTP %{http_code}' -X POST -d '' "$ATT_URL/reset_prefix_cache" 2>/dev/null)
  case "$pc" in *" HTTP 200") ;; *) att_stop "$tag: no attach: POST /reset_prefix_cache answered '${pc:0:160}'"; return 1 ;; esac
  ck=$(curl -s -m 10 "$ATT_URL/admin/fn4" 2>/dev/null | $ATT_PY fn4check "$ATT_ST/server.fn4base.json" - 2>/dev/null)
  case "$ck" in OK*) ;; *) att_stop "$tag: no attach: read back after the resets: ${ck:0:300}"; return 1 ;; esac
  st=$(att_idle) || { att_stop "$tag: no attach: not idle after the resets ($st)"; return 1; }
  ep=${ck##* }; hs=$(att_field "$rec" healthy_s); st=$(att_field "$rec" t0); t1=$(date +%s)
  mkdir -p "${O:-$ATT_ST}" "${STATSDIR:-$ATT_ST}"
  echo "$(att_now) attached to $ATT_CN $id (sig ${s:0:8}); started by $(att_field "$rec" item0) $(att_field "$rec" tag0)" > "${O:-$ATT_ST}/up-$tag.txt"
  docker logs "$ATT_CN" > "${O:-$ATT_ST}/start-$tag.log" 2>&1
  att_put "$rec" "$(sed -n 1p "$rec" | sed -E "s# owner=[^ ]*# owner=${ATTACH_RUNID:-manual-$$}#; s# item=[^ ]*# item=${ATTACH_ITEM:-manual}#; s# attached=[^ ]*##") attached=$tag@$t1"
  export ATTACH_ATTACHED=1 ATTACH_EPOCH0=$ep ATTACH_SINCE=$(date -u +%FT%TZ)
  att_log "$tag: attached to $(att_short "$id") (sig ${s:0:8}), saved ~$(( ${hs:-0} - (t1 - t0) )) s | CARRY-OVER (not reset): expert placement / heat, adaptive draft / lend / back-off state, epoch $ep not 0 -- a confirmation run uses a fresh start | $fr | prefix cache reset: ${pc% HTTP*} | read back OK, idle, epoch $ep | started by $(att_field "$rec" item0) $(att_field "$rec" tag0) (healthy in ${hs:-?} s) $(( t1 - ${st:-$t1} )) s ago; STATSDIR $(att_field "$rec" statsdir); attach took $((t1 - t0)) s"
  return 0
}

att_down() {   # down() with KEEP=1: keep the server for the next item, or `docker rm -f` as today
  local c id keep=$ATT_ST/server.keep rec=$ATT_ST/server.sig why="" st nx so s0 n t0
  t0=$(date +%s)
  c=$(att_cst); id=${c#* }
  if [ "${c%% *}" != running ]; then docker rm -f "$ATT_CN" > /dev/null 2>&1; att_clear; return 0; fi
  if [ -s "$keep" ] && [ "$(att_field "$keep" owner)" = "${ATTACH_RUNID:-}" ] && [ "$(att_field "$keep" cid)" = "$id" ] \
     && [ -z "$(att_field "$keep" claimed)" ] && [ "$(att_next)" = "$(att_field "$keep" next)" ]; then return 0; fi   # the decision of an earlier down()
  s0=$(att_sig0)
  { [ -s "$rec" ] && [ "$(att_field "$rec" cid)" = "$id" ]; } || why="no record for $(att_short "$id")"
  [ -n "$why" ] || [ "$(att_field "$rec" owner)" = "${ATTACH_RUNID:-}" ] || why="this item did not start or attach it"
  [ -n "$why" ] || [ ! -e "$ATT_HOME/runs/sweep.stop" ] || why="runs/sweep.stop"
  [ -n "$why" ] || { st=$(att_code /health); [ "$st" = 200 ]; } || why="/health $st"
  [ -n "$why" ] || st=$(att_idle) || why="not idle ($st)"
  [ -n "$why" ] || [ "${KEEP_VIRHE_OK:-0}" = 1 ] || { n=$(att_virhe); [ "$n" = 0 ]; } || why="$n VIRHE line(s) since this item began (KEEP_VIRHE_OK=1 ignores them)"
  [ -n "$why" ] || nx=$(att_next) || why="next item: $nx"
  [ -n "$why" ] || grep -q '^# ATTACH-SIG:' "$ATT_QD/$nx" 2>/dev/null || why="next item $nx declares no ATTACH (# ATTACH-SIG:)"
  if [ -z "$why" ]; then
    so=$(att_sigonly "$ATT_QD/$nx")
    case "$so" in "ATTACH-SIG "*" ATTACH=1 "*) [ "$(echo "$so" | awk '{print $2}')" = "$s0" ] || why="next item $nx has another signature ($(echo "$so" | awk '{print substr($2, 1, 8)}') vs ${s0:0:8})" ;;
      *) why="next item $nx: its SIGONLY pass gave '${so:-nothing}'" ;; esac
  fi
  if [ -n "$why" ]; then
    docker rm -f "$ATT_CN" > /dev/null 2>&1; att_clear
    att_log "down: server $(att_short "$id")${s0:+ (sig ${s0:0:8})} down: $why"; return 0; fi
  att_put "$keep" "next=$nx sig=$s0 cid=$id owner=$ATTACH_RUNID t=$(date +%s) logpos=$(att_logsize)"
  att_log "keep: server $(att_short "$id") (sig ${s0:0:8}) left up for $nx (healthy, idle, its SIGONLY signature matches; decided in $(( $(date +%s) - t0 )) s)"
}

att_cguard() {   # att_cguard CN: from cguard (the wrapper, then restart()'s own call)
  local cn=${1:-$ATT_CN} c id keep=$ATT_ST/server.keep rec=$ATT_ST/server.sig so age kt
  if [ -z "${ATTACH_RUNID:-}" ] && { [ "${ATTACH:-0}" = 1 ] || [ "${KEEP:-0}" = 1 ]; }; then
    export ATTACH_ITEM=$(basename "$0") ATTACH_RUNID="$(basename "$0" .sh)-$(date +%s)-$$" ATTACH_LOGPOS=$(att_logsize)
    [ -n "$(trap -p EXIT)" ] || trap 'att_epilogue' EXIT
  fi
  [ "$cn" = "$ATT_CN" ] || return 0
  c=$(att_cst); id=${c#* }
  [ "${c%% *}" = running ] || { att_clear; return 0; }
  if [ -s "$keep" ] && [ "$(att_field "$keep" cid)" = "$id" ]; then
    kt=$(att_field "$keep" t); age=$(( $(date +%s) - ${kt:-0} ))
    if [ "${ATTACH:-0}" = 1 ] && [ "$(att_field "$keep" next)" = "${ATTACH_ITEM:-}" ] && [ -z "$(att_field "$keep" claimed)" ] && [ $age -le "${KEEP_MAXWAIT:-300}" ]; then
      att_put "$keep" "$(sed -n 1p "$keep") claimed=$ATTACH_RUNID"
      export ATTACH_CLAIMED_CN=$cn
      att_log "cguard: claimed kept server $(att_short "$id") (sig $(att_field "$keep" sig | cut -c1-8)) for $ATTACH_ITEM (kept ${age} s ago)"
      return 0
    fi
    att_stop "cguard (${ATTACH_ITEM:-$(basename "$0")}): kept for $(att_field "$keep" next) ${age} s ago$([ "${ATTACH:-0}" = 1 ] || echo ", this item is not ATTACH=1")"
    return 0
  fi
  if [ -s "$rec" ] && [ "$(att_field "$rec" cid)" = "$id" ]; then
    if [ "${ATTACH:-0}" = 1 ] && [ -n "${ATTACH_ITEM:-}" ] && [ -f "$0" ] && [ "$(basename "$0")" = "$ATTACH_ITEM" ]; then
      so=$(att_sigonly "$0")
      if [ "$(echo "$so" | awk '{print $2}')" = "$(att_sig0)" ]; then
        att_put "$keep" "next=$ATTACH_ITEM sig=$(att_sig0) cid=$id owner=- t=$(date +%s) logpos=$(att_logsize) claimed=$ATTACH_RUNID"
        export ATTACH_CLAIMED_CN=$cn
        att_log "cguard: claimed our unkept server $(att_short "$id") for $ATTACH_ITEM (its SIGONLY signature matches)"; return 0
      fi
    fi
    att_stop "cguard (${ATTACH_ITEM:-$(basename "$0")}): our server, not kept for this item"
  fi
  return 0
}

att_epilogue() {   # the wrapper's EXIT trap: our server stays up only when this item's body kept it for the CURRENT queue head
  local rc=$? c id keep=$ATT_ST/server.keep why="" nx
  c=$(att_cst); id=${c#* }
  if [ "${c%% *}" = running ] && { [ "$(att_field "$ATT_ST/server.sig" cid)" = "$id" ] || [ "$(att_field "$keep" cid)" = "$id" ]; }; then
    { [ -s "$keep" ] && [ "$(att_field "$keep" cid)" = "$id" ] && [ "$(att_field "$keep" owner)" = "${ATTACH_RUNID:-}" ] && [ -z "$(att_field "$keep" claimed)" ]; } \
      || why="this item made no keep decision for it"
    [ -n "$why" ] || { nx=$(att_next) && [ "$nx" = "$(att_field "$keep" next)" ]; } || why="the queue head is now '$nx' (kept for $(att_field "$keep" next))"
    [ -n "$why" ] || [ ! -e "$ATT_HOME/runs/sweep.stop" ] || why="runs/sweep.stop"
    if [ -n "$why" ]; then att_stop "epilogue (${ATTACH_ITEM:-?}): $why"
    else
      ( setsid nohup bash "$ATT_DIR/attach.sh" keeper "$(att_field "$keep" next)" "$id" "$(att_field "$keep" logpos)" < /dev/null > /dev/null 2>&1 & )
      att_log "epilogue (${ATTACH_ITEM:-?}): server $(att_short "$id") stays up for $(att_field "$keep" next); keeper started (stops it on another item / sweep.stop / ${KEEP_MAXWAIT:-300} s)"
    fi
  fi
  return $rc
}

att_keeper() {   # att_keeper NEXT CID LOGPOS (background): until NEXT claims the kept server
  local nx=$1 cid=$2 pos=${3:-0} t0 c x keep=$ATT_ST/server.keep
  t0=$(date +%s)
  echo "$(att_now) keeper: watching $(att_short "$cid") for $nx" >> "$ATT_ST/attach-keeper.log"
  while :; do
    sleep 1
    c=$(att_cst)
    [ "$c" = "running $cid" ] || { echo "$(att_now) keeper: $(att_short "$cid") is gone ($c)" >> "$ATT_ST/attach-keeper.log"; return 0; }
    { [ -s "$keep" ] && [ "$(att_field "$keep" cid)" = "$cid" ]; } || { echo "$(att_now) keeper: keep record gone" >> "$ATT_ST/attach-keeper.log"; return 0; }
    [ -z "$(att_field "$keep" claimed)" ] || { echo "$(att_now) keeper: claimed by $(att_field "$keep" claimed)" >> "$ATT_ST/attach-keeper.log"; return 0; }
    [ ! -e "$ATT_HOME/runs/sweep.stop" ] || { att_stop "keeper: runs/sweep.stop while kept for $nx"; return 0; }
    # another item starting, or the named one ending unclaimed (the keeping item's own "exit" line is not a reason)
    x=$(tail -c +$((pos + 1)) "$ATT_LG" 2>/dev/null | grep -oE 'RESULT chain-after: queue item [^ ]+ (start|exit)' \
        | awk -v n="$nx" '($6 == "start" && $5 != n) || ($6 == "exit" && $5 == n) {print; exit}')
    [ -z "$x" ] || { att_stop "keeper: '${x#RESULT chain-after: }' while kept for $nx"; return 0; }
    [ $(( $(date +%s) - t0 )) -lt "${KEEP_MAXWAIT:-300}" ] || { att_stop "keeper: $nx did not claim it in ${KEEP_MAXWAIT:-300} s"; return 0; }
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    keeper) att_keeper "${2:?next}" "${3:?cid}" "${4:-0}"; exit 0 ;;
    selftest) $ATT_PY selftest ;;
    *) sed -n '2,40p' "$0" ;;
  esac
fi
