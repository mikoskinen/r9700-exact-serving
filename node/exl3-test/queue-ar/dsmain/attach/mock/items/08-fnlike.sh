#!/bin/bash
# 08-fnlike.sh (ATTACH MOCK item, 12.11.2026; mockchain.sh only) -- a Flash-Next-like item: no cguard, no ATTACH line, refuses when any
# radiance container runs (= 195-fn-dp2.sh line 77's check). It must find the cards free.
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { echo "$(date '+%F %T') RESULT xfn: VIRHE a radiance container is running -- the cards are not free" | tee -a $L; exit 1; }
echo "$(date '+%F %T') RESULT xfn: cards free (no radiance container) -- a Flash-Next item would start here" | tee -a $L
