#!/bin/bash
# 170-fn-fnres3.sh (queue item, 10.11.2026, queue-ar; gpu02, every radiance container DOWN; ~30 min) -- THE PREFIX-CACHE FIX,
# VERIFIED on Flash-Next (coordinator 10.11.: exact toward cold, cold output unchanged). 169 reproduced H3 on production's
# 1.15.1: after a short publisher A sharing the head's whole blocks, B's hit differed from B's cold at token 77 (token-0
# logprob 2.1e-3); the control (no A) was identical. The DS fix (ds41-resfix cee0ee3, core/mem prefix + kvtier: an
# existing entry under the publisher's deepest snapshot that no snapshot rests on, not live and not moving, takes the
# publisher's blocks) is ported unchanged: P-fix = radiance-local:1.15.1-resfix (stilldeadcode/radiance:1.15.1 with
# bin/radiance rebuilt from v1.15.1 + the five files; the unpatched rebuild equals stock after the RPATH rewrite), compose
# fn-prodres.yaml (one DP half, /ngram on tmpfs); M-fix = radiance-local:c16-rf (fn4-c16 + the fix), stack 3, nearest.
# 169's sequence per image (S1: A, B cold, B hit; S2 restart: B cold, B hit) and its four comparisons, PLUS: B cold and A
# vs 169's unpatched outputs byte for byte (the cold path unchanged), the fix's startup line and its takeover INFO lines,
# and on M the cold 33k copy vs 128's (fn4-abval2). PASS = S1 hit IDENTICAL to cold on both images, every cold equal to
# 169's. Power untouched. Production untouched (a patched image prepared and tested, not deployed).
# DRYRUN=1 bash 170-fn-fnres3.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4/fnres3; O9=$R/fn4/fnres2; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
QA=$HOME/gpu/ds41/exl3-test/queue-ar
RHO=$QA/dsmain/resume_ho.py; FHO=$QA/fnres_ho.py; HEADC=${X169_HEAD:-5400}
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
NGD=${X168_NGRAM:-/dev/shm/fnres-ngram}   # a tmpfs directory, as production's /dev/shm; removed at the end
DEPTH=${X168_DEPTH:-66000}; NONCE=${X168_NONCE:-fnres168}; TOK=${X168_TOKENS:-256}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
Y=; compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
setimg() {   # setimg P|M
  if [ $1 = P ]; then
    Y=$QA/fn-prodres.yaml; export FNIMAGE=${X170_PIMAGE:-radiance-local:1.15.1-resfix-cee0ee3} NGRAMDIR=$NGD
  else
    Y=$QA/fn-tp4z.yaml
    export FNIMAGE=${X170_MIMAGE:-radiance-local:c16-rf} FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
    export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
    export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats
    export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4X3C="" FN4QSA="" PROFEVERY=""
  fi
}
restart() {   # restart NAME -> 0 healthy
  local nm=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE compose -f $Y up -d; health <= 1500 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$nm.txt 2>&1
  for i in $(seq 1 300); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 170 $nm: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "RESULT fn4 170 $nm: healthy in $(( $(date +%s) - t0 )) s ($FNIMAGE $(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)); prefix-resume: $(grep -m1 -E -- '--prefix-resume' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-160)$(grep -q -- '--prefix-resume' $S || echo 'no startup line names it'); $(grep -m1 -E -- '--checkpoint-policy|--prefix-cache-ttl' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-120)"
}
rho() {   # rho OUTNAME long|short
  local ex=""; [ $2 = short ] && ex="--head-chars $HEADC --tokens 32" || ex="--tokens $TOK"
  run "(cd $D && timeout 1800 python3 -I $FHO run $O/$1 $2 --depth $DEPTH --nonce $NONCE $ex --top 2 > $O/$1.fho 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 170 $1: $(grep -h '^FHO ' $O/$1.fho | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read() or "{}")
ta, tb = d.get("tiers_after") or {}, d.get("tiers_before") or {}
print("prompt %s cached %s ttft %s tokens %s lp %s md5 %s ckpt_written %s -> %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("lp", "-"), d.get("md5"), tb.get("ckpt_written"), ta.get("ckpt_written")))' 2>&1 | cut -c1-240)"
}
cmp2() {   # cmp2 A B LABEL
  [ "$DRY" = 1 ] && { echo "+ resume_ho.py cmp $O/$1 $O/$2"; return 0; }
  log "RESULT fn4 170 $3 ($1 vs $2): $(cd $D && python3 -I $RHO cmp $O/$1 $O/$2 2>&1 | grep -h '^CMP ' | cut -c5- | cut -c1-500)"
}
cmpx() {   # cmpx ABS_A ABS_B LABEL: resume_ho cmp on absolute paths
  [ "$DRY" = 1 ] && { echo "+ resume_ho.py cmp $1 $2"; return 0; }
  log "RESULT fn4 170 $3: $(cd $D && python3 -I $RHO cmp $1 $2 2>&1 | grep -h '^CMP ' | cut -c5- | cut -c1-400)"
}
copyt() {   # copyt NAME: the 33k copy test (greedy), output $O/copy-fn4-fnres3-NAME-run1.txt
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-fn4-fnres3-$1 --outdir $O > $O/fnres3-$1-copy.txt 2>&1)"
}
log "RESULT fn4 170-fn-fnres3: start -- the prefix-cache fix: P-fix 1.15.1-resfix (one DP half, n-gram on tmpfs) and M-fix c16-rf, 169's sequence; nonce depth $DEPTH, head $HEADC chars, $TOK tokens"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 170: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -e $RHO ] && [ -e $FHO ] || { log "RESULT fn4 170: VIRHE $RHO or $FHO missing"; exit 1; }
  for im in ${X170_PIMAGE:-radiance-local:1.15.1-resfix-cee0ee3} ${X170_MIMAGE:-radiance-local:c16-rf}; do
    docker image inspect $im > /dev/null 2>&1 || { log "RESULT fn4 170: VIRHE image $im missing"; exit 1; }
  done
  [ "$(stat -f -c %T $(dirname $NGD) 2>/dev/null)" = tmpfs ] || { log "RESULT fn4 170: VIRHE $(dirname $NGD) is not a tmpfs"; exit 1; }
  [ "$(df --output=avail -B1G $(dirname $NGD) | tail -1 | tr -d ' ')" -ge 60 ] || { log "RESULT fn4 170: VIRHE under 60 GB free on $(dirname $NGD)"; exit 1; }
  mkdir -p $O $O/stats $O/guard $NGD
  trap 'down; rm -rf $NGD' EXIT
fi
for im in ${X170_ORDER:-P M}; do
  setimg $im
  # S1: A (the shared-head publisher), then B cold and its hit
  restart $im-s1 || continue
  rho $im-s1-A short
  rho $im-s1-cold long
  rho $im-s1-hit long
  # S2 (control): an empty cache (a fresh process), the long prompt cold first, then its hit
  restart $im-s2 || continue
  rho $im-s2-cold long
  rho $im-s2-hit long
  if [ $im = M ]; then
    copyt M
    if [ "$DRY" != 1 ]; then
      if cmp -s $R/fn4/copy-fn4-abval2-cold-run1.txt $O/copy-fn4-fnres3-M-run1.txt; then log "RESULT fn4 170 M-fix cold 33k copy vs 128's (fn4-abval2): IDENTICAL"
      else log "RESULT fn4 170 M-fix cold 33k copy vs 128's (fn4-abval2): VIRHE DIFFERENT ($(cmp $R/fn4/copy-fn4-abval2-cold-run1.txt $O/copy-fn4-fnres3-M-run1.txt 2>&1 | head -1 | cut -c1-120))"; fi
    fi
  fi
  run "docker logs $CN > $O/$im-engine.log 2>&1"
  down
  cmp2 $im-s1-cold $im-s1-hit "$im S1 hit vs S1 cold (after the shared-head publisher A)"
  cmp2 $im-s2-cold $im-s1-hit "$im S1 hit vs S2 cold (the H3 case: the hit beside a clean cold)"
  cmp2 $im-s2-cold $im-s2-hit "$im S2 hit vs S2 cold (control: no short publisher)"
  cmp2 $im-s1-cold $im-s2-cold "$im S1 cold vs S2 cold (cold across processes)"
  cmpx $O9/$im-s2-cold $O/$im-s2-cold "$im-fix cold vs 169's unpatched cold (the cold path unchanged)"
  cmpx $O9/$im-s1-A $O/$im-s1-A "$im-fix A vs 169's unpatched A"
  [ "$DRY" = 1 ] || log "RESULT fn4 170 $im-fix engine: startup '$(grep -m1 -o "later publisher's bytes win[^:]*" $O/start-$im-s1.log)'; takeovers $(grep -c "take a later publisher's bytes" $O/$im-engine.log) INFO lines: $(grep -m2 "take a later publisher's bytes" $O/$im-engine.log | sed 's/^[A-Z][^ ]* //' | cut -c1-160 | tr '\n' '|')"
done
log "RESULT fn4 170-fn-fnres3: done"
