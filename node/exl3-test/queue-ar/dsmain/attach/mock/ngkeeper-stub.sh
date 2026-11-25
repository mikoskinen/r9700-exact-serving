#!/bin/bash
# ngkeeper-stub.sh (12.11.2026, attach/mock) -- swapped in for tools/ngramshare/ngram-keeper.sh in the ATTACH mock (mockrun --swap):
# logs every call to runs/ngkeeper-stub.log (the test checks that no attach / keep decision ever calls stop / cleanup), touches nothing.
echo "$(date '+%F %T') ngram-keeper $*" >> "$HOME/gpu/ds41/exl3-test/runs/ngkeeper-stub.log"
case "${1:-}" in start) echo "mock keeper: 2 of 2 tables held (stub)" ;; prep) echo "mock prep" ;; *) echo "mock $*" ;; esac
exit 0
