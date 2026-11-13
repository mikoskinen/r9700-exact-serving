#!/bin/bash
# dsm-attmock.sh (12.11.2026, queue-ar/dsmain/attach/mock, ds-main-port ATTACH builder) -- NOT a queue item: the MOCK body of the
# ATTACH test items (attach/mock/items/NN-*.sh), run under queue-ar/tools/mockrad (mockchain.sh) only. It starts like the stack5
# family (lib-29.sh -> queue/lib-pr266.sh restart / down, ds41-stack2-tp4.yaml, stackenv + LEVENV written to XS2_ENVFILE, best_vars,
# NST 5 / MNBT ATM_MNBT, the patched cards-free check), then: the start state (fn4 switches + epoch vs ATTACH_EPOCH0, the swnote
# rule), a ~3k-token prompt twice (cached tokens: 0 on a fresh or reset server, then a hit), its own pin ATM_PIN (+ epoch_bump,
# read back), down. Tag "RESULT xatm ". Knobs: ATM_TAG, ATM_IMAGE, ATM_MNBT (4160), ATM_PIN (JSON), ATM_LEVX (extra start words),
# ATM_NGS (1 = source tools/ngramshare/ngshare.sh, as new DS bodies do).
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$QA/dsmain/ds41-stack2-tp4.yaml
# ATM_NGS=1: the n-gram reuse opt-in exactly as new DS bodies do it (ngshare.sh wraps restart(), Y -> ds41-ngshare-tp4.yaml, NGRAM=ram=/ngram)
[ "${ATM_NGS:-0}" = 1 ] && { . $HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngshare.sh || exit 1; }
TAG=${ATM_TAG:?set ATM_TAG}
O=${ATM_OUT:-$R/xdsm/attmock}/$TAG; export STATSDIR=$O/stats
export IMAGE=${ATM_IMAGE:?set ATM_IMAGE}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
PIN=${ATM_PIN:-'{"ds41_psk": 2, "sched_noa": 2}'}
xlog() { log "RESULT xatm $*"; }
SWITCHES="RADIANCE_DS41_IDXKEY RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_PLACE_TURNOVER
  RADIANCE_DS41_GG_PF RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_DS41_PSK RADIANCE_DS41_MHC_FUSE RADIANCE_DS41_KV_REPLICATED"
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_PSK=0 RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 ${ATM_LEVX:-}"
LEVENV=$(echo $LEVENV)
stackenv() { local kv; unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 RADIANCE_DS41_LEVELS=1 \
         RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done; }
fn4get() { curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
nudge() { curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{"prompt": "Hello", "n_predict": 4, "temperature": 0}' $URL/completion > /dev/null 2>&1; }
longq() {   # longq N -> cached tokens of a ~3k-token prompt (the same text every item: a hit shows a prefix cache that was not reset)
  python3 - "$URL" <<'PY'
import json, sys, urllib.request
txt = " ".join("line %d of the attach test document: the prefix cache must start empty after an attach" % i for i in range(260))
req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=json.dumps({"model": "deepseek-v4.1-flash", "messages": [
    {"role": "user", "content": txt}], "max_tokens": 8, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
u = d.get("usage") or {}
print("%s/%s" % ((u.get("prompt_tokens_details") or {}).get("cached_tokens"), u.get("prompt_tokens")))
PY
}
dstart() {
  local s0 ok e0 c1 c2 g i r
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=${ATM_MNBT:-4160} NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$TAG.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$TAG: start IMAGE=$IMAGE MNBT=$MNBT LEVENV '$LEVENV' ATTACH=${ATTACH:-0} KEEP=${KEEP:-0} claimed=${ATTACH_CLAIMED_CN:-none}"
  restart $TAG || { xlog "$TAG: VIRHE not healthy"; down; return 1; }
  [ "$DRY" = 1 ] && return 0
  s0=$(fn4get)
  ok=$(python3 -c 'import json,os,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; e0 = int(os.environ.get("ATTACH_EPOCH0") or 0)
bad = {k: w.get(k) for k in ("ds41_psk", "sched_noa", "draft_occ", "ds41_bi") if w.get(k) != 0}
print("OK" if not bad and s0["epoch"] == e0 and not s0["pending"] else "FAIL %s epoch %s (expected %s) pending %s" % (bad, s0["epoch"], e0, s0["pending"]))' "$s0" 2>&1)
  xlog "$TAG: start state (swnote rule: the pinned switches 0, epoch = ATTACH_EPOCH0 ${ATTACH_EPOCH0:-0}): $ok | attached=${ATTACH_ATTACHED:-0}$([ "${ok%% *}" = OK ] || echo " -- VIRHE")"
  c1=$(longq); c2=$(longq)
  xlog "$TAG: ~3k-token prompt twice: cached/prompt $c1 then $c2 (the first must be 0: a fresh or reset prefix cache)$([ "${c1%%/*}" = 0 ] || echo " -- VIRHE the prefix cache was not empty")"
  curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["epoch_bump"]=1; print(json.dumps(d))' "$PIN")" $URL/admin/fn4 > /dev/null
  for i in $(seq 1 20); do nudge; g=$(fn4get)
    r=$(python3 -c 'import json,sys; g=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); print("OK" if all(g["switches"].get(k)==v for k,v in w.items()) and not g["pending"] else "no")' "$g" "$PIN" 2>&1)
    [ "$r" = OK ] && break; sleep 1; done
  xlog "$TAG: pin $PIN applied: $r (epoch $(echo "$g" | python3 -c 'import json,sys; print(json.load(sys.stdin)["epoch"])'))"
  down
  xlog "$TAG: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
xlog "$TAG: item start"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -iE '^radiance' | grep -vxF "${ATTACH_CLAIMED_CN:-.}" | grep -q . && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
fi
mkd $O $STATSDIR
dstart
down
