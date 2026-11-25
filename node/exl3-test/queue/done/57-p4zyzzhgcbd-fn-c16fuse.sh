#!/bin/bash
# 119-fn-tp4fuse.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h: selftests
# ~5 min + seven screen arms ~15 min each, X119_ARMS for fewer) -- the DECODE SCREEN of the fn4 exact fusions (TESTING.md
# 9.11. loop: the target metric first; the stack's full suite is a later item), each behind its own switch, on the
# candidate line (rmnIc0A's
# flags: --prefix-resume nearest --prefill-interleave 0 --draft-window adaptive; TP4 all_vram, MTP 3, 210 W).
# IMAGES: radiance-local:c16-297632f (base: main 451b136 + chain_pays 0, built whole in radiance-build) and the fix image
# (X93_IMAGE_C16F, default radiance-local:c16-fix: the same build tree plus fn4-c16's F2b / F2a / F1a / F1n commits, every
# switch off by default). 93k-fn-tp4ab.sh's arms:
#   c16b    the base                                  c16f0   the fix image, every switch off (the base's bytes)
#   c16R    F2b  RADIANCE_FN4_NO_ROUTE=1   -48 launches a trunk pass (aql_copy16 + its host step a MoE layer)
#   c16G    F2a  RADIANCE_FN4_GATHER_INV=1 -96 (gather_inv + gather_scatter a MoE layer)
#   c16D    F1a  RADIANCE_FN4_HC_FUSEDOWN=1 -97 (hc_down_reduce a hc_read)
#   c16N    F1n  RADIANCE_FN4_HC_NORMWAVE=1 (the norm scales at 16-byte loads: 115's hc_norm_scale 18.0 us at 33-64 rows
#                against 2.0 us at <= 32, 97 a pass = ~1.55 ms of a c16 step on card 0)
#   c16all  all four
# STEP 0 (card c3:00 alone, ~5 min, before any arm): r4d_selftest of the fix image's libr4d -- --case
# router_gemm_topk_scatter_inv_bf16 (F2a: every output byte for byte against the unfused launches, the inverse against the
# table built from sorted_tok, each shape launched twice), --case router_gemm_topk_scatter_bf16 (the unchanged row and its
# twin), --case hc_read_e4m3 (F1a and F1n against the pair byte for byte -- x, inject gains, both code planes and scales,
# d -- at M 16 17 24 32 33 48 64 65 and 300 under prefill pin 1, folded launched twice, plus the existing decode / prefill
# form checks). A FAIL (exit 1) or a crash stops the item before the arms; skips are logged.
# PHASES (93j X93_SCREEN=1 per arm): gate, slow-state probe, mb c4 / c8 / c16 with engine stats, the 33k copy test byte for
# byte against c16b's (X93_COPYREF), the c16 window's card-0 stampcat + gapcat + gapsize (X93_PROF16; card 0 stamps every
# 20th pass, which does not inflate the step) and a form check (which kernels each arm's stamped passes dispatched: a
# switch that did not take shows). Every arm against c16b (X93_REF): mb rows (cmparm), warm-up answers.
# DRYRUN=1 bash 119-fn-tp4fuse.sh : prints every command, runs and writes nothing.
set -u
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
C3=GPU-dbba5c851cfef7b3
FNC16F=${X93_IMAGE_C16F:-radiance-local:c16-fix}; export X93_IMAGE_C16F=$FNC16F
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
B93=; for f in "$H/93k-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93k-fn-tp4ab.sh"; do [ -e "$f" ] && { B93=$f; break; }; done
[ -n "$B93" ] || { log "RESULT fn4 119: VIRHE 93k-fn-tp4ab.sh not found beside $H"; exit 1; }
if [ "$DRY" = 1 ]; then
  for c in router_gemm_topk_scatter_inv_bf16 router_gemm_topk_scatter_bf16 hc_read_e4m3; do
    echo "+ timeout -k 15 1200 docker run --rm --init --name fn4-119-st-$c --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $FNC16F /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/119-selftest-$c.txt"
  done
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 119: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $FNC16F > /dev/null 2>&1 || { log "RESULT fn4 119: VIRHE image $FNC16F missing"; exit 1; }
  mkdir -p $O
  bad=0
  for c in router_gemm_topk_scatter_inv_bf16 router_gemm_topk_scatter_bf16 hc_read_e4m3; do
    f=$O/119-selftest-$c.txt
    timeout -k 15 1200 docker run --rm --init --name fn4-119-st-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $FNC16F \
      /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $f 2>&1
    rc=$?; docker rm -f fn4-119-st-$$ > /dev/null 2>&1
    np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f)
    log "RESULT fn4 119 selftest $c (card c3:00, $FNC16F): exit $rc, $np ok / $nf FAIL / $ns skip; fn4 lines: $(grep -E 'fn4|_inv |folded|wave norm' $f | grep -E ' ok | FAIL | skip ' | sed 's/^ *//' | cut -c1-140 | head -6 | tr '\n' '|' | cut -c1-900)$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -4 | sed 's/^ *//' | cut -c1-200 | tr '\n' '|')")"
    case $rc in 0|2) ;; *) bad=1 ;; esac
    [ $nf -gt 0 ] && bad=1
  done
  [ $bad = 0 ] || { log "RESULT fn4 119: VIRHE a selftest failed or crashed -- no arm runs ($O/119-selftest-*.txt)"; exit 1; }
fi
export ARMS=${X119_ARMS:-"c16b c16all c16N c16D c16G c16R c16f0"} X93_REF=${X93_REF:-c16b} X93_COPYREF=${X93_COPYREF:-c16b}
export X93_SCREEN=1 X93_PROF16=${X93_PROF16:-1}
export X93_HANG428=0 X93_ITEM=${X93_ITEM:-119-fn-tp4fuse} X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"rmnIc0A:c16b"}
log "RESULT fn4 119-fn-tp4fuse: decode screen, arms [$ARMS] via $(basename $B93), reference c16b, copy reference c16b"
DRYRUN=$DRY exec bash "$B93"
