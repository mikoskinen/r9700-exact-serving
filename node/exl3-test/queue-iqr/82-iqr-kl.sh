#!/bin/bash
# 82-iqr-kl.sh (6.11.2026; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN) -- DeepSeek-V4.1 quality of every
# expert format as the KL divergence to the official release (radiance's KL mode, docs/TOOLS.md "The KL mode"): the reference
# ds41-ref-q8dense-iq4engram.rad (MXFP4 experts as shipped, q8_0 dense, iq4_nl Engram) records its top-512 log-probabilities and
# tail mass at every position of the held-out corpus (corpus/kl-heldout.jsonl: 329,901 tokens, 103 documents, disjoint from the
# calibration corpus; --kld-top 512 = 4 KiB a position, ~1.4 GB), once; each candidate is scored against it: KL mean / median /
# p99 / p99.9 / max, top-1 agreement, PPL ratio, per source (agent-oh, agent-swe, chat, wiki-en, wiki-fi, code).
# MODELS (default "ref-self exl3-225 exl3-29 udiq1s"; later "iqr215 iqr225 iqr2l iqr3s iqr215-iq1strunk"): lib-iqr.sh's model()
# table (image, .rad, host pool). ref-self scores the reference against its own recording (the pipeline's floor: expected
# ~0; a nonzero mean is run-to-run nondeterminism). A model whose image or .rad is missing is skipped with a RESULT line.
# TRUNK CAVEAT: exl3-225, exl3-29 and udiq1s carry bakon3's uncensored (abliterated) q6_K-mix trunk, the reference and the IQR
# containers the release's q8_0 trunk; iqr215-iq1strunk (iqr215's experts on the UD-IQ1_S trunk) is the like-for-like row for
# exl3-225 (same trunk, same expert bytes).
# All runs: TP4 exact wire, bf16 KV, NST 0, MNBT 8192, MAXSEQS 8, MML 65536, Engram in RAM (placement changes no number).
# Steps: guards; record the reference unless $WORK/kl/ref/kld.json exists; each model: --kld-ref + --kld-out (deadline
# KL_DEADLINE, default 3600 s); RESULT line per model (kl_line.py). Outputs: $WORK/kl/<model>.json (+ .rows), runs/iqr/kl-*.log.
# DRYRUN=1 bash 82-iqr-kl.sh : prints only. Estimate: ~6 min start + 330K tokens of prefill (~4-9 min) a run: ~12-15 min a model,
# ~60-75 min for the reference + 4 candidates.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
for LIB in "$H/lib-iqr.sh" "$H/../lib-iqr.sh" "$HOME/gpu/ds41/exl3-test/queue-iqr/lib-iqr.sh"; do [ -e "$LIB" ] && break; done
. "$LIB" || { echo "lib-iqr.sh not found"; exit 1; }
MODELS=${MODELS:-ref-self exl3-225 exl3-29 udiq1s}
KD=$WORK/kl
guards iqr-kl || exit 1
mkd $KD
[ "$DRY" = 1 ] || [ -s $CORPUS/kl-heldout.jsonl ] || { log "RESULT iqr-kl: VIRHE $CORPUS/kl-heldout.jsonl missing"; exit 1; }
log "RESULT iqr-kl: start MODELS='$MODELS' corpus $(md5sum $CORPUS/kl-heldout.jsonl 2>/dev/null | cut -c1-8) -> $KD"
# 1. the reference's recording
if [ "$DRY" = 1 ] || [ ! -s $KD/ref/kld.json ]; then
  if have ref; then
    oneshot_vars; [ "$DRY" = 1 ] || rm -rf $KD/ref   # a recording that did not finish (no kld.json) is not resumable
    export XARGS="--kld-record /work/kl/ref --kld-corpus /corpus/kl-heldout.jsonl --kld-top 512"
    oneshot kl-ref-record ${KL_DEADLINE:-3600}; rc=$?
    if [ "$DRY" != 1 ]; then
      [ $rc = 0 ] && [ -s $KD/ref/kld.json ] || { log "RESULT iqr-kl ref record: VIRHE $ONESHOT_ST exit $ONESHOT_EC after $ONESHOT_S s: $(errs kl-ref-record)"; exit 1; }
      log "RESULT iqr-kl ref record: done in $ONESHOT_S s, $(du -sh $KD/ref | cut -f1); $(grep -m1 'kld: recording' $O/kl-ref-record.log | cut -c1-200)"
    fi
  else
    log "RESULT iqr-kl: VIRHE no reference recording and no reference .rad"; exit 1
  fi
fi
# 2. the candidates
for m in $MODELS; do
  stopfile && break
  mm=$m; [ $m = ref-self ] && mm=ref
  have $mm || continue
  oneshot_vars
  export XARGS="--kld-ref /work/kl/ref --kld-out /work/kl/$m.json"
  oneshot kl-$m ${KL_DEADLINE:-3600}; rc=$?
  [ "$DRY" = 1 ] && continue
  if [ $rc = 0 ] && [ -s $KD/$m.json ]; then
    log "RESULT iqr-kl $m: $(python3 -I $KIT/kl_line.py $KD/$m.json 2>&1 | cut -c1-900) [$MODEL, $IMAGE, ${ONESHOT_S} s]"
  else
    log "RESULT iqr-kl $m: VIRHE $ONESHOT_ST exit $ONESHOT_EC after $ONESHOT_S s: $(errs kl-$m)"
  fi
done
log "RESULT iqr-kl: done -> $KD"
