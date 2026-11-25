#!/bin/bash
# 168P-fn-fnresP.sh (queue item, 10.11.2026, queue-ar; gpu02, every radiance container DOWN; ~15 min) -- 168's P ARM ALONE.
# 168's P server stopped at startup (out of memory): --ngram-placement ram pins the 47.68 GiB n-gram table from /ngram,
# and 168 gave it a directory on the root disk (36 GB free, not a tmpfs) instead of production's /dev/shm -- pinning
# failed at piece 36 of 48. Here /ngram is a tmpfs directory as in production (/dev/shm/fnres-ngram, 250 GB tmpfs, empty),
# removed at the end. Everything else as 168:
# CORRECTNESS: does a
# Flash-Next prefix-cache resume answer as its cold prefill does? (coordinator 10.11., from the DS line's finding R1 /
# hypothesis H3: PrefixCache::insert keeps a cached entry's KV blocks from their FIRST publisher, the linear-state snapshots
# are last-writer-wins from the requester's own bytes; a short request that first publishes the shared chat head (computed
# at small M) and a long prompt that recomputes it inside a 2048-row chunk then leave a mismatched pair, and a later hit
# restores one beside the other. Flash-Next's delta-net layers keep snapshotted state.) Two images, one after the other:
#   P  production's: stilldeadcode/radiance:1.15.1, ONE DP half as compose/flashnext-prod-dp-a.yaml serves it (TP2 83:00 +
#      86:00, prefix cache ttl 8 h, keep-all, no --prefix-resume flag = 1.15.1's default) -- compose fn-prodres.yaml (a test
#      container on :8090, own n-gram dir; production's files untouched);
#   M  our main stack 3 (iq4r): radiance-local:c16-q (fn4-c16 7302cad9), TP4, stack 3's switches, MNBT 4096 + span 2,
#      --prefix-resume nearest -- compose fn-tp4z.yaml.
# Per image: S1 = start, the warm-up's three short questions (ds41_tests correct --only-q: the short publisher), a long
# prompt (resume_ho.py nonce, ~70k tokens) COLD, then the same prompt as a HIT; S2 (control) = restart (an empty cache),
# the long prompt COLD first, then its HIT. Every request greedy, thinking off, 256 tokens, logprobs top 2 when served
# (resume_ho.py, the DS agent's driver, read-only). Compared byte for byte (+ logprobs): S1 hit vs S1 cold, S1 hit vs S2
# cold (the H3 case), S2 hit vs S2 cold, S1 cold vs S2 cold. The --prefix-resume mode each server runs, from its startup
# lines. Power untouched. Nothing goes into production from here.
# DRYRUN=1 bash 168P-fn-fnresP.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4/fnresP; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
QA=$HOME/gpu/ds41/exl3-test/queue-ar
RHO=$QA/dsmain/resume_ho.py
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
    Y=$QA/fn-prodres.yaml; export FNIMAGE=${X168_PIMAGE:-stilldeadcode/radiance:1.15.1} NGRAMDIR=$NGD
  else
    Y=$QA/fn-tp4z.yaml
    export FNIMAGE=${X168_MIMAGE:-radiance-local:c16-q} FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
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
  [ $ok = 1 ] || { log "RESULT fn4 168P $nm: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "RESULT fn4 168P $nm: healthy in $(( $(date +%s) - t0 )) s ($FNIMAGE $(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)); prefix-resume: $(grep -m1 -E -- '--prefix-resume' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-160)$(grep -q -- '--prefix-resume' $S || echo 'no startup line names it'); $(grep -m1 -E -- '--checkpoint-policy|--prefix-cache-ttl' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-120)"
}
rho() {   # rho OUTNAME: the long prompt, greedy, logprobs
  run "(cd $D && timeout 1800 python3 -I $RHO run $O/$1 nonce --depth $DEPTH --nonce $NONCE --tokens $TOK --top 2 > $O/$1.rho 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 168P $1: $(grep -h '^RHO ' $O/$1.rho | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read() or "{}")
print("prompt %s cached %s ttft %s tokens %s lp %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("lp", "-")))' 2>&1 | cut -c1-200)"
}
cmp2() {   # cmp2 A B LABEL
  [ "$DRY" = 1 ] && { echo "+ resume_ho.py cmp $O/$1 $O/$2"; return 0; }
  log "RESULT fn4 168P $3 ($1 vs $2): $(cd $D && python3 -I $RHO cmp $O/$1 $O/$2 2>&1 | grep -h '^CMP ' | cut -c5- | cut -c1-500)"
}
log "RESULT fn4 168P-fn-fnresP: start -- prefix-cache resume vs cold, production's 1.15.1 (one DP half) alone, n-gram table on tmpfs; nonce prompt depth $DEPTH, $TOK tokens"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 168P: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -e $RHO ] || { log "RESULT fn4 168P: VIRHE $RHO missing"; exit 1; }
  for im in ${X168_PIMAGE:-stilldeadcode/radiance:1.15.1} ${X168_MIMAGE:-radiance-local:c16-q}; do
    docker image inspect $im > /dev/null 2>&1 || { log "RESULT fn4 168P: VIRHE image $im missing"; exit 1; }
  done
  [ "$(stat -f -c %T $(dirname $NGD) 2>/dev/null)" = tmpfs ] || { log "RESULT fn4 168P: VIRHE $(dirname $NGD) is not a tmpfs"; exit 1; }
  [ "$(df --output=avail -B1G $(dirname $NGD) | tail -1 | tr -d ' ')" -ge 60 ] || { log "RESULT fn4 168P: VIRHE under 60 GB free on $(dirname $NGD)"; exit 1; }
  mkdir -p $O $O/stats $O/guard $NGD
  trap 'down; rm -rf $NGD' EXIT
fi
for im in ${X168_ORDER:-P}; do
  setimg $im
  # S1: the short publisher first, then the long prompt cold and its hit
  restart $im-s1 || continue
  run "(cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$im-s1-warm-q.txt 2>&1)"
  rho $im-s1-cold
  rho $im-s1-hit
  # S2 (control): an empty cache (a fresh process), the long prompt cold first, then its hit
  restart $im-s2 || continue
  rho $im-s2-cold
  rho $im-s2-hit
  run "docker logs $CN > $O/$im-engine.log 2>&1"
  down
  cmp2 $im-s1-cold $im-s1-hit "$im S1 hit vs S1 cold (after the short publisher)"
  cmp2 $im-s2-cold $im-s1-hit "$im S1 hit vs S2 cold (the H3 case: the hit beside a clean cold)"
  cmp2 $im-s2-cold $im-s2-hit "$im S2 hit vs S2 cold (control: no short publisher)"
  cmp2 $im-s1-cold $im-s2-cold "$im S1 cold vs S2 cold (cold across processes)"
done
log "RESULT fn4 168P-fn-fnresP: done"
