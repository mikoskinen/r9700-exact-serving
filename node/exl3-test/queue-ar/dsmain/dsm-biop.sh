#!/bin/bash
# dsm-biop.sh (11.11.2026, queue-ar/dsmain, ds41-bi, lever BI) -- NOT a queue item; the body of 57-p4zyzzzzzzzzzz-bi-a-opcheck.sh.
# Log tag "RESULT xbio ", out runs/xdsm/biop (XBIO_OUT). LEVER BI'S PER-OP ON-CARD BYTE COMPARE (USER RULING 11.11.: an answer
# that changes under load vs alone is a BUG). One card (HIP_VISIBLE_DEVICES=0), no server, every radiance container DOWN.
# The image's r4d_selftest --bicheck (ds41-bi, libr4d/r4d_selftest_bi.inc): for every row-wise op the DeepSeek trunk issues, at
# its TP4 shapes (dense gemm_nt_ggml q5_K / q6_K / q8_0 + the head + the K1 folds a / e, gemm_nt_f32 router / compressor, gemm_nt
# bf16 indexer / compressor, rmsnorm, the mhc ops and the fused collapse / post-collapse norms, rope / round trips, swiglu,
# cast, softmax_pool, router_scored_topk): a row ALONE (M = 1) vs the same row inside an M-row launch, byte for byte, M = 1..64,
# 65, 80, 96, 127..129, 192, 256, 512, 1024, 2112, 4160, probe rows 0, M/2, M-1; the kernel row the engine's band takes at M.
# Two arms (libr4d reads its form switches at load, so one process each):
#   stock  every libr4d form switch unset;
#   stack  the stack's: RADIANCE_DS41_GG_GEMV=2 (D2 multi-row decode band), RADIANCE_DS41_GG_PF=1 + the 7-entry GPF table,
#          RADIANCE_DS41_F32_PF=25 (GPF-b).
# One "RESULT xbio <arm> BICHECK ..." line a case (INVARIANT | FLIPS at M=<first> (<n> of <m> M differ: list) | rows by M band),
# then "RESULT xbio summary:" (per case stock / stack flip M). Measurement only: no lever changes, nothing into production.
# ~10-20 min. Attention / indexer / all-reduce are not in this mode (sequence / rank structure): the engine tracer item
# (57-p4zyzzzzzzzzzz-bi-b-trace, dsm-bilid.sh XBIL_TRACE=1) names them in the real engine.
# DRYRUN=1 bash dsm-biop.sh : prints every command; checks the image (r4d_selftest carries --bicheck and its case labels), the
#   helpers (functions via type -t, tools on PATH); no GPU, writes nothing.
# Knobs: XBIO_IMAGE (required), XBIO_ITEM, XBIO_OUT, XBIO_ARMS ("stock stack"), XBIO_CLASSES ("all"), XBIO_MS ("" = default).
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
O=${XBIO_OUT:-$R/xdsm/biop}
IMAGE=${XBIO_IMAGE:?set XBIO_IMAGE}
ITEM=${XBIO_ITEM:-dsm-biop}
ARMS=${XBIO_ARMS:-stock stack}; CLASSES=${XBIO_CLASSES:-all}; MS=${XBIO_MS:-}
for a in $ARMS; do case $a in stock|stack) ;; *) echo "XBIO_ARMS takes stock / stack (got '$a')"; exit 1 ;; esac; done
case "$CLASSES" in *[!a-z0-9,]*) echo "XBIO_CLASSES: a comma list of class names (got '$CLASSES')"; exit 1 ;; esac
case "$MS" in *[!0-9,]*) echo "XBIO_MS: a comma list of row counts (got '$MS')"; exit 1 ;; esac
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (dsm-attnfold2.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
armenv() {   # armenv stock|stack: the docker -e words of the arm
  case $1 in
    stock) echo "" ;;
    stack) echo "-e RADIANCE_DS41_GG_GEMV=2 -e RADIANCE_DS41_GG_PF=1 -e RADIANCE_DS41_GG_PF_TABLE=$GPTAB -e RADIANCE_DS41_F32_PF=25" ;;
  esac
}
onearm() {   # onearm ARM: the bicheck in one process, its lines logged
  local a=$1 rc t0=$(date +%s) l
  run "gpurun xbiop-$$-$a 2700 $DEV $(armenv $a) --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --bicheck $CLASSES $MS > $O/bicheck-$a.txt 2>&1"
  rc=$?
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xbio $a $(echo "$l" | cut -c1-900)"; done < <(grep -E '^(BICHECK|  BIFAIL|lever BI per-op)' $O/bicheck-$a.txt)
  log "RESULT xbio $a: rc $rc, $(( $(date +%s) - t0 )) s$([ $rc = 0 ] || echo " -- VIRHE (launch / setup failures or a crash: $(tail -3 $O/bicheck-$a.txt | tr '\n' ' ' | cut -c1-400))")"
}
summary() {   # summary: per case the first flip M under each arm, one line
  [ "$DRY" = 1 ] && { echo "+ RESULT xbio summary: per case stock / stack (INVARIANT | first flip M)"; return 0; }
  log "RESULT xbio summary: $(python3 - $O $ARMS <<'PY' 2>&1 | head -c 3500
import re, sys, os
o, arms = sys.argv[1], sys.argv[2:]
res = {}
for a in arms:
    try: lines = open(os.path.join(o, "bicheck-%s.txt" % a)).read().splitlines()
    except OSError: continue
    for l in lines:
        m = re.match(r"BICHECK (.+?): (INVARIANT|FLIPS at M=(\d+)|SKIP)", l)
        if m: res.setdefault(m.group(1), {})[a] = "inv" if m.group(2) == "INVARIANT" else ("skip" if m.group(2) == "SKIP" else "flip@" + m.group(3))
flips = {k: v for k, v in res.items() if any(x.startswith("flip") for x in v.values())}
print("%d cases; FLIP: %s; invariant in every arm: %d; skipped: %s" % (len(res),
      "; ".join("%s [%s]" % (k, " ".join("%s %s" % (a, v.get(a, "-")) for a in arms)) for k, v in flips.items()) or "none",
      sum(1 for v in res.values() if all(x == "inv" for x in v.values())),
      ", ".join(k for k, v in res.items() if any(x == "skip" for x in v.values())) or "none"))
PY
)"
}
log "RESULT xbio $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] classes $CLASSES M ${MS:-default} -> $O"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xbio: VIRHE image $IMAGE not on this host"; exit 1; }
LITS="--bicheck|BICHECK summary|lever BI per-op check|engram_kv|mhc_post_collapse_rmsnorm n=5120|router_scored_topk 384 top6"
MISS=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'S=/opt/radiance/bin/r4d_selftest; IFS="|"; for l in $0; do grep -qzF -- "$l" $S || echo "MISS $l"; done; echo CHECKED' "$LITS" 2>&1)
case "$MISS" in CHECKED*) ;; *) log "RESULT xbio: VIRHE $IMAGE's r4d_selftest lacks --bicheck: $(echo "$MISS" | grep MISS | tr '\n' ';' | cut -c1-400)"; exit 1 ;; esac
HMISS=""
for fn in log run mkd gpurun armenv onearm summary; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker python3 timeout grep awk; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
[ -z "$HMISS" ] || { log "RESULT xbio: VIRHE helpers do not resolve:$HMISS"; exit 1; }
if [ "$DRY" = 1 ]; then
  echo "+ image check: $IMAGE's r4d_selftest carries --bicheck and its case labels -- OK; helpers resolve -- OK"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses beside one)"
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xbio: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xbio: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
fi
mkd $O
for a in $ARMS; do onearm $a; done
summary
log "RESULT xbio $ITEM: done -- $O"
