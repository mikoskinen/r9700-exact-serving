# lib-iqr.sh (6.11.2026) -- sourced by the IQR items 81-85 (DeepSeek-V4.1 routed experts as IQR-V4 computed trellis; the
# reference, its calibration, KL of every format, speed). NOT a queue item. Lives in ~/gpu/ds41/exl3-test/queue-iqr/ (the kit:
# this file, ds41-iqr-tp4.yaml, kl_line.py, gram_check.py, corpus/, iqr-bin/); an item finds it beside itself, in ../ (queue/
# when chain-after runs it from queue/done/) or in that kit directory.
# Sources queue/lib-pr266.sh (log / run / mkd / serve_args / desc / restart / check / probe / mbp / evict / down, DRYRUN=1 prints
# only) and replaces its compose file with ds41-iqr-tp4.yaml (CORPUS and WORK mounts) and its output dir with runs/iqr.
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -e "$KIT/ds41-iqr-tp4.yaml" ] || KIT=$HOME/gpu/ds41/exl3-test/queue-iqr
LIBPR=${LIBPR:-$HOME/gpu/ds41/exl3-test/queue/lib-pr266.sh}
. "$LIBPR" || { echo "lib-pr266.sh not found"; exit 1; }
Y=$KIT/ds41-iqr-tp4.yaml
O=$R/iqr
IQRIMAGE=${IQRIMAGE:-radiance-local:ds41-exl3-117-b77f7a85-f2f3c3f7-iqrmoe}   # ds41-iqrmoe b77f7a85 on the mx4l image
X3IMAGE=${X3IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}            # the 1.17 EXL3 image of the 6.11. rows
PRIMAGE=${PRIMAGE:-radiance-local:ds41-pr266-9c7ce498}                       # stock PR #266 head (UD-IQ1_S rows)
export STATSDIR=$O/stats CORPUS=$KIT/corpus WORK=${IQRWORK:-$O/work}
export PROFEVERY= PROFSYNC= AQLSTAMP=
IQRMODELS=${IQRMODELS:-$HOME/models/ds41-iqr}
# Per-layer calibration bytes (packed per-expert down Grams, RADGRAM2 layout 1): 384 x 10,621,472 + 4 x 104,857,632 (gate_up
# pooled, one a rank) + 4 x 21,233,696 (down pooled) = 4.583 GB; 40 layers 183.3 GB. Host RAM while recording: 1.02 GB a layer a
# rank (4.08 GB a layer over 4 ranks) + 2.04 GB pinned staging a rank; VRAM: 126 MB a recorded layer a rank + a 2.04 GB scratch.
CAL_LAYER_B=4583000000

# model NAME -> IMAGE MODELDIR MODEL HP (the host pool the KL and calibration runs start with; speed rows set their own)
model() {
  case $1 in
    ref)        IMAGE=$IQRIMAGE; MODELDIR=$HOME/models/ds41-release; MODEL=ds41-ref-q8dense-iq4engram.rad; HP=${HP_REF:-53760} ;;
    exl3-225)   IMAGE=$X3IMAGE; MODELDIR=$HOME/models/ds41-exl3; MODEL=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad; HP=19456 ;;
    exl3-29)    IMAGE=$X3IMAGE; MODELDIR=$HOME/models/ds41-exl3; MODEL=deepseek-v4.1-flash-1017-exl3-2.9bpw.rad; HP=31488 ;;
    udiq1s)     IMAGE=$PRIMAGE; MODELDIR=$HOME/models/ds41; MODEL=deepseek-v4.1-flash-1017-uncensored.rad; HP=12288 ;;
    iqr215)     IMAGE=$IQRIMAGE; MODELDIR=$IQRMODELS; MODEL=ds41-iqr215.rad; HP=19456 ;;
    iqr225)     IMAGE=$IQRIMAGE; MODELDIR=$IQRMODELS; MODEL=ds41-iqr225.rad; HP=21504 ;;
    iqr2l)      IMAGE=$IQRIMAGE; MODELDIR=$IQRMODELS; MODEL=ds41-iqr2l.rad; HP=30720 ;;
    iqr3s)      IMAGE=$IQRIMAGE; MODELDIR=$IQRMODELS; MODEL=ds41-iqr3s.rad; HP=36096 ;;
    iqr215-iq1strunk) IMAGE=$IQRIMAGE; MODELDIR=$IQRMODELS; MODEL=ds41-iqr215-iq1strunk.rad; HP=19456 ;;
    *) return 1 ;;
  esac
  export IMAGE MODELDIR MODEL
}
# have NAME -> 0 when its image and .rad (with a .sha256ok from a verified copy) are on this machine
have() {
  model $1 || return 1
  [ "$DRY" = 1 ] && return 0
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT iqr: $1 skipped: image $IMAGE missing"; return 1; }
  [ -f $MODELDIR/$MODEL ] || { log "RESULT iqr: $1 skipped: $MODELDIR/$MODEL missing"; return 1; }
  [ -s $MODELDIR/$MODEL.sha256ok ] || log "RESULT iqr: $1 note: $MODELDIR/$MODEL.sha256ok missing (copy not verified)"
  return 0
}
# guards every IQR item runs first: no radiance container (production or test), KFD idle (60 s wait)
guards() {   # NAME
  [ "$DRY" = 1 ] && return 0
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT $1: VIRHE a radiance container is running -- not now"; return 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT $1: VIRHE a KFD client is running (after 60 s wait)"; return 1; }
  return 0
}
# A one-shot mode run (KL record / score, calibration): the engine replaces the server, runs the corpus and exits.
# oneshot TAG DEADLINE_S -> 0 when it exited 0 in time. Sets ONESHOT_S (wall), ONESHOT_EC (exit code), ONESHOT_ST.
oneshot() {
  local tag=$1 dl=$2 a t0 st=exited
  a=$(serve_args)
  if [ "$DRY" = 1 ]; then
    echo "+ IMAGE=$IMAGE MODELDIR=$MODELDIR MODEL=$MODEL CORPUS=$CORPUS WORK=$WORK ARGS='$a' docker compose --project-directory $C -f $Y up -d; wait for exit <= ${dl}s; docker logs > $O/$tag.log"
    ONESHOT_S=0 ONESHOT_EC=0 ONESHOT_ST=dry; return 0; fi
  mkdir -p $O $STATSDIR $WORK
  docker rm -f $CN > /dev/null 2>&1
  t0=$(date +%s)
  ARGS="$a" docker compose --project-directory $C -f $Y up -d > $O/up-$tag.txt 2>&1 || { ONESHOT_ST="compose up failed"; ONESHOT_EC=-1; return 1; }
  while [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" = running ]; do
    [ $(( $(date +%s) - t0 )) -ge $dl ] && { docker stop -t 60 $CN > /dev/null; st=TIMEOUT; break; }
    sleep 15
  done
  docker logs $CN > $O/$tag.log 2>&1
  ONESHOT_EC=$(docker inspect -f '{{.State.ExitCode}}' $CN 2>/dev/null); ONESHOT_ST=$st; ONESHOT_S=$(( $(date +%s) - t0 ))
  docker rm -f $CN > /dev/null 2>&1
  [ "$ONESHOT_EC" = 0 ] && [ "$st" = exited ]
}
errs() { grep -E '^E |error|refus' $O/$1.log 2>/dev/null | head -4 | tr '\n' ' ' | cut -c1-600; }
# The flags every KL and calibration run shares (numerics: TP4 exact wire, bf16 KV, no speculation; placement does not change
# numbers). MML 65536 holds the corpora's longest document (24,576 tokens).
oneshot_vars() { export HOSTPOOL=$HP NGRAM=${NGRAM_OS:-ram} CKINT=8192 MNBT=8192 NST=0 MAXSEQS=8 MML=65536 WIRE=exact; }
# drun NAME DEADLINE_S OUTFILE DOCKER-RUN-ARGS... : a detached `docker run --name NAME ...`, waited for with a deadline (docker
# stop at it; a `timeout docker run` would leave the container running), its logs to OUTFILE, removed. 0 when it exited 0.
drun() {
  local n=$1 dl=$2 out=$3 t0 st=exited ec; shift 3
  if [ "$DRY" = 1 ]; then echo "+ docker run -d --name $n $* ; wait <= ${dl}s ; docker logs > $out"; return 0; fi
  docker rm -f $n > /dev/null 2>&1
  t0=$(date +%s); docker run -d --name $n "$@" > /dev/null || return 1
  while [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ]; do
    [ $(( $(date +%s) - t0 )) -ge $dl ] && { docker stop -t 10 $n > /dev/null; st=TIMEOUT; break; }
    sleep 5
  done
  docker logs $n > $out 2>&1; ec=$(docker inspect -f '{{.State.ExitCode}}' $n 2>/dev/null); docker rm $n > /dev/null 2>&1
  [ $st = TIMEOUT ] && echo "TIMEOUT after ${dl}s" >> $out
  [ "$ec" = 0 ] && [ $st = exited ]
}
