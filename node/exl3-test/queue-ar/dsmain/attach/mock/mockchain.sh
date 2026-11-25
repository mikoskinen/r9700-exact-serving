#!/bin/bash
# mockchain.sh (12.11.2026, queue-ar/dsmain/attach/mock, ds-main-port ATTACH builder) -- the ATTACH mock test: a mini chain-after.sh
# (same loop: ls NN-*.sh | sort | head -1, mv to done/, "RESULT chain-after: queue item N start / exit" lines, one at a time) over
# items/NN-*.sh in a sandbox queue dir (ATTACH_QDIR=$HOME/attq), run as the TARGET of queue-ar/tools/mockrad/mockrun.sh (HOME is the
# sandbox, docker / curl are the mock's shims; nothing reaches a real server or the real queue). Hooks after an item: 06-atE ->
# inject/06-zz-plain.sh into the queue (the coordinator adding an item after a keep decision); 09-atG -> the mock fault health=503
# on the kept server (MOCKCHAIN_FROM=NN skips the items before NN). Items 11-13: the ngshare path (mockrun --swap the keeper with mock/ngkeeper-stub.sh). Ends with the radiance containers left (must be none) and a summary of the "RESULT attach / xatm / xfn" lines.
#   cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad && ./mockrun.sh --name att-m1 --timeout 1500 --no-dryrun \
#     --fresh gpu/ds41/exl3-test/runs/xdsm/attmock --env MOCKRAD_STARTUP_S=8 \
#     --swap gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngram-keeper.sh=gpu/ds41/exl3-test/queue-ar/dsmain/attach/mock/ngkeeper-stub.sh ~/gpu/ds41/exl3-test/queue-ar/dsmain/attach/mock/mockchain.sh
set -u
[ "${DRYRUN:-0}" = 1 ] && { echo "+ mockchain.sh: DRYRUN, nothing run"; exit 0; }
case "${MOCKRAD_STATE:-}" in "") echo "mockchain.sh runs under queue-ar/tools/mockrad/mockrun.sh only (MOCKRAD_STATE unset)"; exit 2 ;; esac
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log
MD=$T/queue-ar/dsmain/attach/mock
export ATTACH_QDIR=$HOME/attq; Q=$ATTACH_QDIR
rm -rf "$Q"; mkdir -p "$Q/done" "$R"
cp "$MD"/items/[0-9][0-9]-*.sh "$Q/"
for f in "$Q"/[0-9][0-9]-*.sh; do [ "$(basename "$f" | cut -c1-2)" -ge "${MOCKCHAIN_FROM:-1}" ] 2>/dev/null || rm -f "$f"; done   # MOCKCHAIN_FROM=NN: start there
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
hook() {
  case $1 in
    06-atE.sh) cp "$MD/inject/06-zz-plain.sh" "$Q/"; log "MOCKCHAIN injected 06-zz-plain.sh after 06-atE's keep decision (the coordinator race)" ;;
    09-atG.sh) log "MOCKCHAIN mock fault health=503 on the kept server: $(curl -s -m 5 -X POST -H 'Content-Type: application/json' -d '{"set": {"health": 503}}' http://127.0.0.1:8090/mock/fault | cut -c1-120)" ;;
  esac
}
log "MOCKCHAIN start: $(ls $Q/[0-9][0-9]-*.sh | xargs -n1 basename | tr '\n' ' ')"
while true; do
  item=$(ls $Q/[0-9][0-9]-*.sh 2>/dev/null | sort | head -1)
  [ -n "$item" ] || break
  n=$(basename $item); mv $item $Q/done/$n
  log "RESULT chain-after: queue item $n start"; bash $Q/done/$n > $R/queue-$n.out 2>&1; log "RESULT chain-after: queue item $n exit $?"
  log "MOCKCHAIN after $n: radiance containers up: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')-; records: $(ls $R/server.sig $R/server.keep 2>/dev/null | xargs -rn1 basename | tr '\n' ' ')-"
  hook $n
done
sleep 4
log "MOCKCHAIN done: radiance containers up at the end: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')-"
echo "==== RESULT lines"
grep -E 'RESULT (attach|xatm|xfn|chain-after|cguard|restart-pr266|ngshare)|MOCKCHAIN' $L | cut -c1-400
echo "==== keeper log"; cat $R/attach-keeper.log 2>/dev/null
echo "==== n-gram keeper stub calls (must be prep / start only)"; cat $R/ngkeeper-stub.log 2>/dev/null
