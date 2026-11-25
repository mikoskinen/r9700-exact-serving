# fn4lib.sh (12.11.2026, fn4; coordinator: attach-to-server for FN) -- SOURCE it in an FN queue item after its config (FNIMAGE,
# Y, CN, URL, O, the FN4* / start env, EXTRA) and its log / run / compose / down helpers. ATTACH TO A RUNNING SERVER:
#   fn4_start TAG       attach when a server named $CN runs with this item's START SIGNATURE (image id + compose file md5 +
#                       EXTRA + every start env var the compose reads), else down + start (the item's own restart TAG);
#                       on attach: /health, the prefix cache reset, every fn4 switch put back to its start value (the values
#                       the start read back, srvsw0), checked; returns 0 ready
#   fn4_end             at the item's end: keep the server only when the NEXT queued item is an FN item that declares
#                       '# FN4ATTACH' (it attaches or downs it); a watchdog downs the kept server if nothing attaches
#                       within FN4_KEEP_S (120) s. Otherwise down.
# Items opt in with the '# FN4ATTACH' header line; without it nothing changes. The DS items' "cards free" guards never meet a
# kept server for longer than the watchdog window.
FN4_KEEP_S=${FN4_KEEP_S:-120}
FN4_QDIR=${FN4_QDIR:-$HOME/gpu/ds41/exl3-test/queue}
fn4_sig() {
  { docker image inspect -f '{{.Id}}' "$FNIMAGE" 2>/dev/null; md5sum < "$Y" | cut -c1-32; echo "EXTRA=$EXTRA"
    env | grep -E '^(FN4[A-Z0-9]*|TP|CARDS|PLACE|MML|ROPE|MAXSEQS|MNBT|NST|WIREKB|PMIN|PMINB|FNMODEL|FNMODELDIR|AQLSTAMP|AQLCARDS|PROFEVERY)=' | sort; } \
    | md5sum | cut -c1-16
}
fn4_mark() {   # after a fresh start: the signature and the start switch values
  fn4_sig > $O/srvsig-$CN
  curl -s -m 10 $URL/admin/fn4 | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("switches", {})))' > $O/srvsw0-$CN.json 2>/dev/null
  rm -f $O/attached-$CN
}
fn4_attach() {   # 0 = attached and reset
  [ "$DRY" = 1 ] && { echo "+ attach check: $CN running with signature $(fn4_sig)?"; return 1; }
  [ -n "$(docker ps -q -f name=^${CN}$)" ] || return 1
  [ "$(cat $O/srvsig-$CN 2>/dev/null)" = "$(fn4_sig)" ] || { log "RESULT fn4 attach $CN: signature differs -> restart"; return 1; }
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] || return 1
  : > $O/attached-$CN
  local c0 sw0; c0=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache)
  sw0=$(cat $O/srvsw0-$CN.json 2>/dev/null); [ -n "$sw0" ] || return 1
  curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "$(echo "$sw0" | sed 's/}$/, "epoch_bump": 1}/')" $URL/admin/fn4 > /dev/null
  curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{"prompt": "Hello", "n_predict": 4, "temperature": 0}' $URL/completion > /dev/null
  local now; now=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1])["switches"]; w=json.loads(sys.argv[2]); sys.exit(0 if all(d.get(k) == v for k, v in w.items()) else 1)' "$now" "$sw0" 2>/dev/null \
    || { log "RESULT fn4 attach $CN: VIRHE the switches did not return to the start values -> restart"; return 1; }
  log "RESULT fn4 attach $CN: ATTACHED (signature $(cat $O/srvsig-$CN)), prefix cache reset HTTP $c0, switches back to the start values"
  return 0
}
fn4_start() {   # fn4_start TAG -> 0 ready
  fn4_attach && return 0
  restart "$1" || return 1
  [ "$DRY" = 1 ] || fn4_mark
}
fn4_end() {
  [ "$DRY" = 1 ] && { echo "+ fn4_end: keep $CN if the next queued item is an FN4ATTACH item, else down"; return 0; }
  local next; next=$(LC_ALL=en_US.UTF-8 ls $FN4_QDIR 2>/dev/null | grep -E '^57.*-fn-.*\.sh$' | LC_ALL=en_US.UTF-8 sort | head -1)
  local first; first=$(LC_ALL=en_US.UTF-8 ls $FN4_QDIR 2>/dev/null | grep -E '^57.*\.sh$' | LC_ALL=en_US.UTF-8 sort | head -1)
  if [ -n "$next" ] && [ "$next" = "$first" ] && grep -q '^# FN4ATTACH' "$FN4_QDIR/$next" 2>/dev/null; then
    log "RESULT fn4 keep $CN for $next (watchdog ${FN4_KEEP_S} s)"
    rm -f $O/attached-$CN
    ( sleep $FN4_KEEP_S; [ -e $O/attached-$CN ] || { docker compose --project-directory $C -f $Y down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1; } ) \
      > /dev/null 2>&1 < /dev/null & disown
    return 0
  fi
  down
}
