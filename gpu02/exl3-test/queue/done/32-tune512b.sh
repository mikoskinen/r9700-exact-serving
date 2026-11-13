#!/bin/bash
# 32-tune512b.sh (6.11. 12.25) -- re-run 30-tune512 from the working base (31-bakon-hp19456: bakon line + host pool 19456; the first
# 30 run had base 4096, which OOMs on this head, so every row failed). The hp rows are dropped (the base already has 19456).
Q=$HOME/gpu/ds41/exl3-test/queue; cd $Q || exit 1
[ -e 30-tune512.cfg.orig ] || cp 30-tune512.cfg 30-tune512.cfg.orig
grep -v -E "^hp[0-9]+\|" 30-tune512.cfg.orig > 30-tune512.cfg.new && mv 30-tune512.cfg.new 30-tune512.cfg
bash $Q/done/30-tune512.sh
