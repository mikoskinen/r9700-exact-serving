#!/bin/bash
# 83-acqrule-ab.sh (queue item, 6.11.2026 night; AFTER 79A (its count decides) and 82-cohbench2 (decides rule / rule-w)) -- the address-based
# acquire rule in the engine, behind its switch: core-acq 68cfbee3 = 9c7ce498 + core-dbg 85088e7 + RADIANCE_AQL_ACQ=rule |
# rule-w (Ctx::haz_mark keeps a window of cache lines a queue and marks each dispatch whose reads cannot be stale; the
# replay plays those packets' agent-scope acquire as none; chained records, packets behind foreign ones and system
# acquires keep theirs). Unset = the engine as before (haz_mark does not compute the rule).
#   rule    the lines both read and written in the window are risky (tools/acqrule.py's count = 79A's number);
#           valid only if 82's ww_none arm saw 0 stale elements (a store leaves no line behind on its own CU)
#   rule-w  every written line is risky too (82's ww_none nonzero): far fewer packets qualify
# Image radiance-local:ds41-exl3-117-1fb143a5-f2f3c3f7-fuse-c68cfbee3 = the fuse image (77) with bin/radiance of core-acq.
# Arms (same image, fin-best line, MAXSEQS 8, MML 524288, 2.25 bpw .rad, gated 21 ms), ARMS default "off rule":
#   off     switch unset: identity copy, tools/greedyset.py (20 fixed greedy prompts) one at a time and at c4 (twice: the
#           c4 batch's own run-to-run floor), and stepprobe + mb c4 only if 79B's agent arm left none (runs/xfix/
#           xacq-agent-probe.txt): no baseline re-run
#   rule / rule-w : the recorded-pass notes ("the acquire rule plays N of the M dispatches"), identity copy, greedyset
#           at c1 and c4, stepprobe, mb c4. Byte for byte against off: the copy, the 20 c1 answers, the 20 c4 answers
#           (counted only where off's two c4 runs agree), the gate's warm-up greedy answers. ANY difference = the rule
#           lets a stale line be read: VIRHE, the arm's speed does not count.
# Expected: 79A's skippable count x ~0.9 us (gapbench 75) a step, minus nothing (no kernel changes).
# Hang protection: no new wait anywhere (only fence bits change); /health after the gate and every step, a dead
# server ends the item with logs saved and the compose DOWN.
# Estimate ~20 min an arm. DRYRUN=1 bash 83-acqrule-ab.sh : prints, writes nothing.
# Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh), ds41-fix2-tp4.yaml (passes RADIANCE_AQL_ACQ) and tools/greedyset.py.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3AQ=${X3AQ:-radiance-local:ds41-exl3-117-1fb143a5-f2f3c3f7-fuse-c68cfbee3}
export IMAGE=$X3AQ
ARMS=${ARMS:-"off rule"}
GS=$QD/tools/greedyset.py; [ -e $GS ] || GS=$(cd "$(dirname "$0")" && pwd)/tools/greedyset.py
REFP=$O/xacq-agent-probe.txt
log "RESULT xfix 83-acqrule-ab: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] compose $Y"
imgcheck $IMAGE
if [ "$DRY" != 1 ]; then
  grep -q "^ *RADIANCE_AQL_ACQ:" $Y || { log "RESULT xfix 83: VIRHE $Y does not pass RADIANCE_AQL_ACQ"; exit 1; }
  [ -e $GS ] || { log "RESULT xfix 83: VIRHE tools/greedyset.py missing"; exit 1; }
fi
mkd $O $STATSDIR
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE \
      RADIANCE_DEBUG_TAPE_DUMP RADIANCE_AQL_ACQ RADIANCE_DS41_AR_PCN
best_vars
quit() { run "docker logs $CN > $O/xaq-$1-engine.log 2>&1"; down; unset RADIANCE_AQL_ACQ; exit ${2:-1}; }
alive() {
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200 (else: logs, down, exit)"; return 0; }
  local st code
  st=$(docker inspect -f '{{.State.Status}} exit {{.State.ExitCode}}' $CN 2>/dev/null || echo gone)
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT xfix 83: VIRHE the $1 arm's server is not healthy after $2 (container: $st, /health $code; last E lines: $(docker logs $CN 2>&1 | grep -E '^E |trap|abort|HSA_STATUS' | tail -3 | tr '\n' ' ' | cut -c1-500)) -> logs saved, down, stopped"
  quit $1
}
# answers A B: the gate's warm-up greedy answers (ds41_tests.py correct --only-q) byte for byte
answers() {
  python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in a if (a[t].get("content"), a[t].get("finish")) != (b.get(t, {}).get("content"), b.get(t, {}).get("finish"))]
print(("IDENTICAL: %d answers" % len(a)) if a and not bad else ("DIFFERENT: " + " ".join(map(str, bad)) if a else "no answers"))' "$1" "$2" 2>&1 | cut -c1-500
}
# gset TAG CONC N: the 20 greedy prompts at concurrency CONC into $O/TAG-gs-N.jsonl
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-$3.jsonl --conc $2 --max-tokens 192 > $O/$1-gs-$3.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "+ greedyset --cmp $1 $2"; echo IDENTICAL; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
for arm in $ARMS; do
  case $arm in
    off)         unset RADIANCE_AQL_ACQ; tag=xaq-off ;;
    rule|rule-w) export RADIANCE_AQL_ACQ=$arm; tag=xaq-$arm ;;
    *)           log "RESULT xfix 83: unknown arm $arm"; continue ;;
  esac
  gate $tag 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 83: VIRHE $arm arm not healthy ($(desc))"; quit $arm; }
  alive $arm gate
  [ $rc = 2 ] && { log "RESULT xfix 83: VIRHE $arm arm slow twice -> stopped"; quit $arm; }
  if [ "$DRY" != 1 ]; then
    ev=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_AQL_ACQ=' | cut -d= -f2- || true)
    [ "$ev" = "$([ $arm = off ] || echo $arm)" ] || { log "RESULT xfix 83: VIRHE the $arm arm's container env says RADIANCE_AQL_ACQ='$ev' -> stopped"; quit $arm; }
  fi
  ident $tag;      alive $arm "the identity copy"
  gset $tag 1 c1;  alive $arm "greedyset c1"
  gset $tag 4 c4;  alive $arm "greedyset c4"
  if [ $arm = off ]; then
    gset $tag 4 c4b; alive $arm "greedyset c4 (second)"
    [ "$DRY" = 1 ] || log "RESULT xfix 83 off: greedyset $(head -1 $O/$tag-gs-c1.txt | cut -c12-90); c4 run against c4 run: $(gcmp $O/$tag-gs-c4.jsonl $O/$tag-gs-c4b.jsonl)"
    if [ -s $REFP ] && [ "$DRY" != 1 ]; then log "RESULT xfix 83 off: speed = 79B's agent arm ($REFP), not re-run"
    else probe $tag; alive $arm "the stepprobe"; mbp $tag c4; alive $arm "mb c4"; fi
  else
    probe $tag;    alive $arm "the stepprobe"
    mbp $tag c4;   alive $arm "mb c4"
  fi
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  if [ "$DRY" != 1 ] && [ $arm != off ]; then
    log "RESULT xfix 83 $arm notes: $(grep -o 'draft pass -\?[0-9]* at [0-9]* token(s) -- the acquire rule plays [0-9]* of the [0-9]* dispatches ([0-9]* of them ordered)' $O/$tag-engine.log | sed 's/ -- the acquire rule plays/:/; s/ dispatches//' | sort -u | head -12 | tr '\n' ';' | cut -c1-900)"
    if [ -s $O/copy-xaq-off-run1.txt ]; then
      identcmp xaq-off $tag || log "RESULT xfix 83: VIRHE the $arm rule changed the identity copy -> NOT safe"
      c1=$(gcmp $O/xaq-off-gs-c1.jsonl $O/$tag-gs-c1.jsonl); c4=$(gcmp $O/xaq-off-gs-c4.jsonl $O/$tag-gs-c4.jsonl)
      fl=$(gcmp $O/xaq-off-gs-c4.jsonl $O/xaq-off-gs-c4b.jsonl)
      wq=$(answers $O/guard/xaq-off-warm-q.txt $O/guard/$tag-warm-q.txt)
      log "RESULT xfix 83 $arm vs off: greedy c1 $c1 | c4 $c4 (off's own c4 floor: $fl) | warm-up answers $wq"
      case "$c1 $wq" in *DIFFERENT*) log "RESULT xfix 83: VIRHE the $arm rule changed greedy answers -> NOT safe" ;; esac
      case "$fl" in IDENTICAL*) case "$c4" in *DIFFERENT*) log "RESULT xfix 83: VIRHE the $arm rule changed the c4 answers (off's two c4 runs agree) -> NOT safe" ;; esac ;; esac
    else
      log "RESULT xfix 83: VIRHE no off arm in this run -> no byte comparison for $arm"
    fi
  fi
  stopfile && quit $arm 0
done
unset RADIANCE_AQL_ACQ
down
log "RESULT xfix 83-acqrule-ab: done | ref $REF_BEST524"
