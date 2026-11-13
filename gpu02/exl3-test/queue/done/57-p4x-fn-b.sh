#!/bin/bash
# 84b-fn-b.sh (queue item, 7.11.2026, queue-ar) -- FN-B: 84-fn-tuned.sh with FNSET=B (see its header: model, image, arms, phases,
# references, time). Finds 84-fn-tuned.sh beside this file, else in ~/gpu/ds41/exl3-test/queue-ar/ or queue/.
# DRYRUN=1 bash 84b-fn-b.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
for S in "$H/84-fn-tuned.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/84-fn-tuned.sh" "$HOME/gpu/ds41/exl3-test/queue/84-fn-tuned.sh"; do
  [ -e "$S" ] && break
done
[ -e "$S" ] || { echo "84-fn-tuned.sh not found"; exit 1; }
FNSET=B exec bash "$S"
