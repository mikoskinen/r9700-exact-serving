#!/bin/bash
# 185-fn-bicheck2.sh (queue item, 12.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~5 min)
# -- fn4 BI hc_bi SCREEN (184 = the first per-op survey: hc_read flips from M 8 on both mixing forms, hc_write and the
# router invariant, the i8a8 linears skipped). Image c16-x3hb (fn4-x3 2bb616fe: hc_bi -- hc_read up to 64 rows through one
# form family; gemm_nt_q's shape hook answers i8a8; --bicheck R4D_BI_REF). Runs: (a) the line's hc switches + HC_BI 1, the
# row alone as reference (want hc_read INVARIANT to 64); (a2) the same at HC_BI 2 (today's per-row kernels to 64 rows: want
# the bf16 cases INVARIANT to 64 -- the reference-keeping form; E4M3 not covered); (b) the line's switches, reference = the 8-row tiled launch (do the
# stock forms above 8 agree among themselves); (c) the line's switches, stock, alone (the i8a8 linears now reached).
# --- 184's header follows ---
# 184-fn-bicheck.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- fn4 BI PER-OP SURVEY on one card: r4d_selftest --bicheck (fn4-x3 62e89b5f, ported from the ds41-bi line's): per op a row
# alone (M = 1) vs the same row inside an M-row launch, byte for byte, probe rows 0 / M/2 / M-1, M 1..64 and 65..4160, at
# Flash-Next's TP4 shapes -- hc_read (block input with codes, final mixer; mix bf16 = the EXL3 line, e4m3 = the iq4r line),
# hc_write (+ ss), the trunk's i8a8 linears (attn q|gate / k,v / o, gdn in / out, shared gate_up / down), the bf16 router.
# Twice: under the line's fn4 hc switches (RADIANCE_FN4_HC_READV 11, _HC_UPMIX 5, _HC_WRITEVEC 1, _HC_NORMWAVE 1, _HC_SS 1, as
# the stack starts) and with none (stock forms). Complements r10 bitrace (the engine's first differing op per probe): this
# names every flipping op and its first flipping M in one pass, so fixes can be screened in minutes. Image
# radiance-local:c16-x3bc (= c16-x3du + the bicheck). Measurement only.
# DRYRUN=1 bash 184-fn-bicheck.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/bicheck2
IMG=${X185_IMAGE:-radiance-local:c16-x3hb}; IMGID=${X184_IMGID:-}
CARD=${CARD:-0}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
bc() {   # bc NAME ENV-WORDS...: one --bicheck run -> $O/NAME.txt
  local nm=$1; shift
  local ev=""; for e in "$@"; do ev="$ev -e $e"; done
  if [ "$DRY" = 1 ]; then echo "+ timeout -k 15 1500 docker run ... $ev $IMG r4d_selftest libr4d.so --bicheck all > $O/$nm.txt"; return 0; fi
  timeout -k 15 1500 docker run --rm --init --name fn4-185-$$-$nm --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARD $ev --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --bicheck all > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f fn4-185-$$-$nm > /dev/null 2>&1
  log "RESULT fn4 185 $nm: exit $rc; $(grep -m1 '^BICHECK summary' $O/$nm.txt)"
  grep -E '^BICHECK [^s]' $O/$nm.txt | while IFS= read -r l; do log "RESULT fn4 185 $nm $(echo "$l" | cut -c1-600)"; done
}
log "RESULT fn4 185-fn-bicheck2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) card $CARD"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 185: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 185: VIRHE image $IMG missing"; exit 1; }
  mkdir -p $O
else
  echo "+ image $IMG: $(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19) (its r4d_selftest's --bicheck: checked on the staged tree before the image was made)"
fi
bc hcbi RADIANCE_FN4_HC_READV=11 RADIANCE_FN4_HC_UPMIX=5 RADIANCE_FN4_HC_WRITEVEC=1 RADIANCE_FN4_HC_NORMWAVE=1 RADIANCE_FN4_HC_SS=1 RADIANCE_FN4_HC_BI=1
bc hcbi2 RADIANCE_FN4_HC_READV=11 RADIANCE_FN4_HC_UPMIX=5 RADIANCE_FN4_HC_WRITEVEC=1 RADIANCE_FN4_HC_NORMWAVE=1 RADIANCE_FN4_HC_SS=1 RADIANCE_FN4_HC_BI=2
bc ref8 RADIANCE_FN4_HC_READV=11 RADIANCE_FN4_HC_UPMIX=5 RADIANCE_FN4_HC_WRITEVEC=1 RADIANCE_FN4_HC_NORMWAVE=1 RADIANCE_FN4_HC_SS=1 R4D_BI_REF=8
bc line RADIANCE_FN4_HC_READV=11 RADIANCE_FN4_HC_UPMIX=5 RADIANCE_FN4_HC_WRITEVEC=1 RADIANCE_FN4_HC_NORMWAVE=1 RADIANCE_FN4_HC_SS=1
log "RESULT fn4 185-fn-bicheck2: done"
