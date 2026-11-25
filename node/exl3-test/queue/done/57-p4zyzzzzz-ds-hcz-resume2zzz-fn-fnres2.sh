#!/bin/bash
# 169-fn-fnres2.sh (queue item, 10.11.2026, queue-ar; gpu02, every radiance container DOWN; ~25 min) -- H3's PRECONDITION
# ON FLASH-NEXT (coordinator 10.11.). 168 / 168P were vacuous for H3: the prefix cache inserts whole blocks only
# (PrefixCache::insert: n_full = n_tokens / block_size), and the warm-up's short questions share no whole block with the
# long prompt (only the chat template's few head tokens), so no entry of the long prompt came from another publisher.
# Here the short publisher A is the long prompt's own first ~5400 characters (~1400 tokens: several whole blocks, ONE
# prefill chunk under 2048 rows) followed by a different question (fnres_ho.py short), so A publishes the shared head's
# blocks first; the long prompt B (fnres_ho.py long = resume_ho's nonce prompt, ~68.6k tokens) cannot resume on them (A's
# only snapshot is its own tail, past the divergence), recomputes the head inside its own 2048-row chunk / 4096-row step,
# and insert keeps A's blocks (has_linear_) beside B's own snapshots; B's hit restores both. Images as 168: P production's
# 1.15.1 (one DP half, compose fn-prodres.yaml, /ngram on tmpfs, removed at the end) and M stack 3 (c16-q, nearest).
# Per image: S1 = start, A, B cold, B hit; S2 (control) = restart, B cold, B hit. Compared (text, token ids, logprobs):
# S1 hit vs S1 cold, S1 hit vs S2 cold (the H3 case), S2 hit vs S2 cold, S1 cold vs S2 cold. The --prefix-resume mode
# each server runs, from its startup lines (1.15.1 lists only options off their default). Power untouched.
# DRYRUN=1 bash 169-fn-fnres2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4/fnres2; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
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
  [ $ok = 1 ] || { log "RESULT fn4 169 $nm: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "RESULT fn4 169 $nm: healthy in $(( $(date +%s) - t0 )) s ($FNIMAGE $(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)); prefix-resume: $(grep -m1 -E -- '--prefix-resume' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-160)$(grep -q -- '--prefix-resume' $S || echo 'no startup line names it'); $(grep -m1 -E -- '--checkpoint-policy|--prefix-cache-ttl' $S | sed 's/^[A-Z] *//' | tr -s ' ' | cut -c1-120)"
}
rho() {   # rho OUTNAME long|short
  local ex=""; [ $2 = short ] && ex="--head-chars $HEADC --tokens 32" || ex="--tokens $TOK"
  run "(cd $D && timeout 1800 python3 -I $FHO run $O/$1 $2 --depth $DEPTH --nonce $NONCE $ex --top 2 > $O/$1.fho 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 169 $1: $(grep -h '^FHO ' $O/$1.fho | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read() or "{}")
ta, tb = d.get("tiers_after") or {}, d.get("tiers_before") or {}
print("prompt %s cached %s ttft %s tokens %s lp %s md5 %s ckpt_written %s -> %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("lp", "-"), d.get("md5"), tb.get("ckpt_written"), ta.get("ckpt_written")))' 2>&1 | cut -c1-240)"
}
cmp2() {   # cmp2 A B LABEL
  [ "$DRY" = 1 ] && { echo "+ resume_ho.py cmp $O/$1 $O/$2"; return 0; }
  log "RESULT fn4 169 $3 ($1 vs $2): $(cd $D && python3 -I $RHO cmp $O/$1 $O/$2 2>&1 | grep -h '^CMP ' | cut -c5- | cut -c1-500)"
}
log "RESULT fn4 169-fn-fnres2: start -- H3's precondition (a short publisher sharing whole head blocks), production's 1.15.1 (one DP half, n-gram on tmpfs) and stack 3 (c16-q); nonce depth $DEPTH, head $HEADC chars, $TOK tokens"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 169: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -e $RHO ] && [ -e $FHO ] || { log "RESULT fn4 169: VIRHE $RHO or $FHO missing"; exit 1; }
  for im in ${X168_PIMAGE:-stilldeadcode/radiance:1.15.1} ${X168_MIMAGE:-radiance-local:c16-q}; do
    docker image inspect $im > /dev/null 2>&1 || { log "RESULT fn4 169: VIRHE image $im missing"; exit 1; }
  done
  [ "$(stat -f -c %T $(dirname $NGD) 2>/dev/null)" = tmpfs ] || { log "RESULT fn4 169: VIRHE $(dirname $NGD) is not a tmpfs"; exit 1; }
  [ "$(df --output=avail -B1G $(dirname $NGD) | tail -1 | tr -d ' ')" -ge 60 ] || { log "RESULT fn4 169: VIRHE under 60 GB free on $(dirname $NGD)"; exit 1; }
  mkdir -p $O $O/stats $O/guard $NGD
  trap 'down; rm -rf $NGD' EXIT
fi
for im in ${X169_ORDER:-P M}; do
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
  run "docker logs $CN > $O/$im-engine.log 2>&1"
  down
  cmp2 $im-s1-cold $im-s1-hit "$im S1 hit vs S1 cold (after the shared-head publisher A)"
  cmp2 $im-s2-cold $im-s1-hit "$im S1 hit vs S2 cold (the H3 case: the hit beside a clean cold)"
  cmp2 $im-s2-cold $im-s2-hit "$im S2 hit vs S2 cold (control: no short publisher)"
  cmp2 $im-s1-cold $im-s2-cold "$im S1 cold vs S2 cold (cold across processes)"
done
log "RESULT fn4 169-fn-fnres2: done"
