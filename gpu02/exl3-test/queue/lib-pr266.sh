# lib-pr266.sh (6.11.2026) -- sourced by queue items 20-bakon.sh, 30-tune512.sh, 40-best.sh. NOT a queue item (chain-after.sh
# runs only queue/NN-*.sh). Lives in ~/gpu/ds41/exl3-test/queue/ beside ds41-pr266-tp4.yaml and 30-tune512.cfg; the items run
# from queue/done/ and find these files in queue/ (or beside themselves for a manual run).
#
# The stock PR #266 head image (radiance-local:ds41-pr266-9c7ce498, Radiance 1.17.0 base, no local changes) on the 3.11.
# UD-IQ1_S container ~/models/ds41/deepseek-v4.1-flash-1017-uncensored.rad. The kit's ds41-exl3-tp4.yaml mounts
# ~/models/ds41-exl3 at /models and always passes --max-num-seqs/--max-model-len, so these items bring their own compose file
# (ds41-pr266-tp4.yaml: MODELDIR mount, the whole serve line in ARGS) with the same project and container name
# (radiance-ds41-exl3), so chain-after.sh's watchdog and its final window.sh down stop it as they stop the kit's server.
# Functions mirror exl3-restart.sh / exl3-gate.sh (same RESULT line shapes) and write RESULT lines to runs/ds41-exl3.log.
#
# DRYRUN=1: every command is printed, nothing runs and nothing is written (no log line, no directory, no file).
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; C=$HOME/radiance-1.15.1/compose; D=$HOME/gpu/ds41
ME=$(cd "$(dirname "${BASH_SOURCE[1]:-$0}")" && pwd)
QD=$T/queue; [ -e $QD/ds41-pr266-tp4.yaml ] || QD=$ME; [ -e $QD/ds41-pr266-tp4.yaml ] || QD=$(dirname "$ME")
Y=$QD/ds41-pr266-tp4.yaml
O=$R/pr266                                   # outputs of 20/30/40
DRY=${DRYRUN:-0}
CN=radiance-ds41-exl3; URL=http://127.0.0.1:8090
export IMAGE=${IMAGE:-radiance-local:ds41-pr266-9c7ce498}
export MODEL=${MODEL:-deepseek-v4.1-flash-1017-uncensored.rad}
export MODELDIR=${MODELDIR:-$HOME/models/ds41}
export STATSDIR=${STATSDIR:-$O/stats}
RAMTIER="--prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all"
# References (3.11. = ~/gpu/ds41/runs/ds41-1103.log, same .rad on radiance-local:ds41-acc9ae58; bakon = docs/deepseek41-wip.md):
REF_BAKON="bakon TP4 (PCIe Gen5, 62 GiB RAM): essay 77.4 / code 77.2 tok/s, 16k prefill 3746 tok/s TTFT 4.24 s"
REF_311_NST5="3.11. NST5 ckint 8192 hp 12288 MAXSEQS 8 + RAM tier (sweep base): essay 34.4 tok/s 42.9 ms, code 77.2 tok/s 44.1 ms; c4 agg_gen 73.1 108.7 ms; c8 58.5; pf 2393.7; turns cold 22.49 hit 3.41 | 3.11. 0-start (hp 12288 MAXSEQS 4 NST5 ckint 8192): essay 33.6 / code 90.9 tok/s, 16k 2844 tok/s"
REF_311_NST0="3.11. NST0 fin-best (hp 12288 MAXSEQS 8 ckint 2048 span 2 IL8 RAM tier): 15.05-15.12 ms; essay 62.6 / code 68.5 tok/s; c4 125.7 26.4 ms; c8 164.0; pf 1474.5; turns cold 27.77 hit 2.16"

log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
mkd() { [ "$DRY" = 1 ] && { echo "+ mkdir -p $*"; return; }; mkdir -p "$@"; }

# The serve line from variables. Defaults = bakon's TP4 line (docs/deepseek41-wip.md) as written: no --num-speculative-tokens
# (auto: DSpark), no --max-num-seqs / --max-model-len (engine defaults 256 / the training context). --row-prefetch and
# --mapped-rewarm are documented there but this head's config.cpp does not parse them (row read-ahead on, rewarm off in code),
# so they are not passed (an unknown option stops startup).
#   HOSTPOOL NGRAM CKINT MNBT NST(empty = auto) MAXSEQS(empty) MML(empty) WIRE XARGS(extra flags)
serve_args() {
  echo "--tp 4 --tp-wire ${WIRE:-exact} --placement expert_tiered --host-pool-mib ${HOSTPOOL:-4096} --ngram-placement ${NGRAM:-disk}" \
       "--max-num-batched-tokens ${MNBT:-8192} --checkpoint-interval ${CKINT:-8192} --kv-cache-dtype bf16" \
       "${NST:+--num-speculative-tokens $NST} ${MAXSEQS:+--max-num-seqs $MAXSEQS} ${MML:+--max-model-len $MML} ${XARGS:-}" | tr -s ' '
}
desc() { echo "HOSTPOOL=${HOSTPOOL:-4096} NGRAM=${NGRAM:-disk} CKINT=${CKINT:-8192} MNBT=${MNBT:-8192} NST=${NST:-auto} MAXSEQS=${MAXSEQS:-default} MML=${MML:-default} XARGS='${XARGS:-}'"; }

# restart TAG -> 0 healthy / 1 not (RESULT line either way). Waits at most 15 min (3.11.: 151-172 s from compose up).
restart() {
  local tag=$1 a ok=0 t0 t1 i kv ex kvc pin mem vram
  a=$(serve_args)
  if [ "$DRY" = 1 ]; then
    [ "${SIGONLY:-0}" = 1 ] && type att_sigonly_print > /dev/null 2>&1 && { att_sigonly_print "$a"; exit 0; }   # 12.11. ATTACH: SIGONLY=1 prints the start signature
    echo "+ docker compose --project-directory $C -f $Y down; IMAGE=$IMAGE MODEL=$MODEL MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS='$a' docker compose --project-directory $C -f $Y up -d; wait /health <= 900 s; docker logs > $O/start-$tag.log"
    return 0; fi
  # 12.11. ATTACH (ds-main-port, opt-in, queue-ar/dsmain/attach/attach.sh): ATTACH=1 -> reuse our idle server with this start's signature
  [ "${ATTACH:-0}" = 1 ] && type att_try > /dev/null 2>&1 && att_try "$tag" "$a" && return 0
  mkdir -p $O $STATSDIR
  docker rm -f $CN > /dev/null 2>&1
  # 12.11. compose guard (ds-main-port): leftover Created / Dead / Exited test containers removed; a ghost entry -> fresh COMPOSE_PROJECT_NAME
  [ -s $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh ] && { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard > /dev/null; }
  t0=$(date +%s)
  ARGS="$a" docker compose --project-directory $C -f $Y up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 180); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  t1=$(date +%s)
  docker logs $CN > $O/start-$tag.log 2>&1
  if [ $ok != 1 ]; then
    log "RESULT restart-pr266-$tag: VIRHE not healthy after $((t1-t0)) s ($(desc)): $(grep -E '^E ' $O/start-$tag.log | head -4 | tr '\n' ' ' | cut -c1-700)"
    docker rm -f $CN > /dev/null 2>&1; return 1; fi
  kv=$(grep -m1 -E '^I kv: the pool' $O/start-$tag.log | cut -c3-200)
  ex=$(grep -m1 -E '^ +experts +[0-9.]+ GiB' $O/start-$tag.log | awk '{print $2, $3}')
  kvc=$(grep -m1 -E '^I +kv cache +[0-9.]+ GiB a card' $O/start-$tag.log | sed -E 's/^I +kv cache +//' | cut -c1-80)
  pin=$(grep -m1 -o 'pinned by this process [0-9.]* GiB' $O/start-$tag.log)
  mem=$(free -g | awk 'NR==2{print "used " $3 " / avail " $7 " GiB"}')
  vram=$(for d in /sys/class/drm/card*/device; do [ -e $d/mem_info_vram_used ] && printf "%s " $(( $(cat $d/mem_info_vram_used)/1048576 )); done)
  log "RESULT restart-pr266-$tag: healthy in $((t1-t0)) s (IMAGE=$IMAGE MODEL=$MODEL $(desc)); $kv; VRAM experts $ex, KV $kvc; $pin; RAM $mem; VRAM used MiB [$vram]; args: $a"
  local rc=$?
  [ "${ATTACH:-0}${KEEP:-0}" = 00 ] || { type att_record > /dev/null 2>&1 && att_record "$tag" "$a" $((t1-t0)); }   # 12.11. ATTACH: the start's record
  return $rc
}
# check TAG N -> prints the 1-user fastcheck ms/step after the 3.11. warm-up (= exl3-gate.sh check)
check() {
  local tag=$1 n=$2
  [ "$DRY" = 1 ] && { echo "+ warm-up (correct --only-q, prefill 16384) + fastcheck.py 384 -> $O/guard/$tag-fc$n.txt" >&2; echo 0; return; }
  mkdir -p $O/guard
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --calibrate 0 > $O/guard/$tag-warm-pf.txt 2>&1)
  (cd $D && timeout 300 python3 fastcheck.py 384 > $O/guard/$tag-fc$n.txt 2>&1)
  grep '^FC ' $O/guard/$tag-fc$n.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999
}
# gate TAG LIMIT_MS -> restart + check; over the limit: one restart and check again (= exl3-gate.sh). Writes $O/guard-TAG.txt.
# Returns 0 fast, 1 not healthy, 2 slow after the retry.
# Limits: NST 0 18 ms (3.11. fast 15.05-15.12, slow first tries 23.16 / 23.24); DSpark 60 ms (3.11. NST5 rounds 41.6-44.4 ms).
gate() {
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || return 1
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    ms1=$ms; restart $tag || return 1; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && st="SLOW-after-retry (first $ms1)"
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT guard pr266-$tag: $st, 1-user ${ms} ms/step (limit $lim; UD-IQ1_S 3.11. NST0 fast 15.05-15.12, NST5 41.6-44.4)"
  case $st in SLOW*) return 2 ;; esac   # slow state twice: the caller skips its measurements (as after-sweep.sh does)
}
guardtxt() { cat $O/guard-$1.txt 2>/dev/null; }
guard_lim() { [ "${NST:-auto}" = 0 ] && echo 18 || echo 60; }
probe() { # TAG
  run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT pr266 $1 probe (1 user, greedy, thinking off, 512 tok) [guard $(guardtxt $1)]: $(python3 $D/sweepfmt.py probe $O/$1-probe.txt 2>&1 | cut -c1-400)"
}
mbp() { # TAG PHASE
  run "(cd $D && timeout 1200 python3 mb.py --tag pr266-$1 --phases $2 > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT pr266 $1 mb $2 [guard $(guardtxt $1)]: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- || tail -1 $O/$1-$2.txt 2>/dev/null)"
}
pf16() { # TAG: two fresh ~16k prompts (first calibrates chars/token), as the 3.11. 16k rows
  run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --reps 2 > $O/$1-pf16.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT pr266 $1 cold prefill 16k: $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d.get("prompt_tokens"), d.get("prefill_tps"), d.get("ttft")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $O/$1-pf16.txt 2>&1 | cut -c1-300) | $REF_BAKON; 3.11. 16k 2844"
}
evict() {   # 6.11. 13.20: stop the test server first, so the page cache really drops (it used to run beside a live server)
  run "docker rm -f $CN > /dev/null 2>&1"
  run "python3 $T/evict-exl3.py >> $R/evict.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT pr266 evict: $(tail -1 $R/evict.txt 2>/dev/null)"
}
down() {   # 12.11. ATTACH: KEEP=1 -> att_down (keeps the server for the next item when it declares the same signature); else as before
  if [ "${KEEP:-0}" = 1 ] && [ "$DRY" != 1 ] && type att_down > /dev/null 2>&1; then att_down; return 0; fi
  run "docker rm -f $CN > /dev/null 2>&1"; }
stopfile() { [ -e $R/sweep.stop ]; }
# 12.11. ATTACH (ds-main-port): the attach / keep functions, sourced ONLY when ATTACH=1, KEEP=1 or SIGONLY=1 (unset = nothing more defined)
if [ "${ATTACH:-0}" = 1 ] || [ "${KEEP:-0}" = 1 ] || [ "${SIGONLY:-0}" = 1 ]; then
  [ -s $T/queue-ar/dsmain/attach/attach.sh ] && . $T/queue-ar/dsmain/attach/attach.sh; fi
