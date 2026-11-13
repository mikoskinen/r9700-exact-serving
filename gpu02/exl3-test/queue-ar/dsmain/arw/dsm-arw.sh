#!/bin/bash
# dsm-arw.sh (12.11.2026, ds-main-port arw builder; the body of items/57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-c-arw.sh) -- ALL FOUR cards,
# every radiance container DOWN, NO server, no evict; ~2-3 min (estimate), bounded at 12 min (selftest 300 s + arw 420 s).
# QUESTION (coordinator 12.11., one code reading, unproven): libr4d's exact four-rank two-shot all-reduce has a lossless CODED wire
# (~12.2 bits a bf16 value) that r4d_ar_entry.hip 1061-1068 selects only at `overlap` 1; DS declares its attention and MoE
# all-reduces without overlap (deepseek41.cpp 2391, arch/common/rad_block_moe_fp8.h 2021), so DS prefill always runs the PLAIN
# wire (all_reduce 280 ms of a 1224 ms 128k prefill step, 3784 us a call: runs/xdsm/dsm-225-prof-pf128-op.txt). Is the coded wire
# byte-identical to the plain one at DS's shapes, and faster? If both, it is a small build behind an fn4 switch.
# PHASES (image XARW_IMAGE, default the int9 image; its libr4d.so md5 pinned = XARW_LIBMD5, the library the harness was checked on):
#  P0 the image's r4d_selftest --case ar_twoshot_4rank_exact_lane (libr4d's own case, r4d_selftest.cpp 32264-32354 ar_lane_forms,
#     run at 37689-37694): four ranks on four cards, the lane row with `coded` 0 and 1 at its fixed sizes (4194304, 1024, 524352,
#     1055744, 20971520 = 4096 x 5120, 46137344; its rough pattern and the coded form's pattern), every rank against the fp32
#     ascending-rank fold bit for bit. It has no timing and no DS shape but 4096 rows, hence P1. A FAIL stops the reading.
#  P1 arw (arw/bin/arw, built by arw/build.sh from arw/bench/arw.hip in radiance-ds41-build:main-e736c019 = the int9 libr4d's
#     toolchain): the image's libr4d.so by dlopen, four ranks = four cards, one host thread a rank in one process (as r4d_selftest's
#     four-rank cases); P = row ar_twoshot_4rank_exact (DS's prefill row, no overlap; its tunable defaults: nblocks 0 = the size
#     rule, nthreads 1024, drain 3 -> 1 from 131072 elements) vs C = row ar_twoshot_4rank_exact_lane (the same entry at overlap 1
#     -> the coded launcher at 4 x 1024) and C8 / C16 / C24 (the coded launcher at a wider grid), at T x 5120 bf16 for T = 1100,
#     1500, 2048, 4096 (prefill) and 8, 16, 48 (decode; coded is selectable there by r4d_ar_twoshot_coded_ok, though DS's decode
#     all-reduce reaches the two-shot only through the one-shot row's ts_min hand-off, which has no coded path). The wire that ran
#     is SEEN on every launch (the harness exports the three launcher names and libr4d's PLT binds to them; DRYRUN proves it from
#     the library's GOT). Identity on patterns bin / act / actz, in place + out of place, every rank vs P (VIRHE, the reading stops)
#     and vs the fp32 fold; timing on act / bin: 100 interleaved isolated launches a variant (own = the last arriver's time:
#     median, p10, p90) + 20 back to back (the lockstep period).
#  P2 arw/arwsum.py -> RESULT xarw CHECKS / SHAPE (one a shape) / SUMMARY / ESTIMATE (the 128k step's 74 calls of 4096 rows).
# No power / host / fan settings, no compose file, no server, no :8090. Out runs/xdsm/arw (selftest-lane.txt, arw.txt, arwsum.txt).
# DRYRUN=1 bash dsm-arw.sh : prints every command; checks the binary / parser / image / library md5s, runs the parser's selftest and,
# in the image with NO GPU device, r4d_selftest --help and arw --selftest (dlopen, rows, ABI, the GOT binding, coded_ok at every
# shape); starts nothing on a GPU, writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
A=$H/arw
O=${XARW_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/arw}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XARW_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}
LIBMD5=${XARW_LIBMD5:-f8c6b9d3735ecd75a71900acff9490d7}   # the int9 image's libr4d.so (sha256 bf7b9d80.. = int9-PROVENANCE.txt)
SUMMD5=${XARW_SUMMD5:-9b90ccabca6804bbfc80fb583f8a624c}   # arwsum.py
TST=${XARW_TST:-150}; TARW=${XARW_TARW:-420}; TSP=${XARW_TSP:-90}   # seconds: P0 lane, P1, P0 plain (12.11. fix: was TST 300)
PCASE=ar_twoshot_4rank_exact                                # P0 plain: the known-good four-rank case (11.11. pre-move: 17 ok in 18 s)
ARGS=${XARW_ARGS:-}                                       # extra arw arguments (e.g. --reps 200)
DISKMIN=${XARW_DISKMIN_MB:-11000}
CASE=ar_twoshot_4rank_exact_lane
LIBP=/opt/radiance/share/radiance/kernels/libr4d.so
STB=/opt/radiance/bin/r4d_selftest
T="RESULT xarw"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
# ---- the files the item needs, each with its md5
miss=0
chk() {   # chk PATH [MD5]: present (and the md5 when given)
  if [ ! -e "$1" ]; then echo "  (check) $1 MISSING"; miss=$((miss + 1)); return 1; fi
  if [ -n "${2:-}" ]; then
    local m; m=$(md5sum < "$1" | cut -c1-32)
    if [ "$m" != "$2" ]; then echo "  (check) $1 md5 $m != $2 MISSING (stale copy)"; miss=$((miss + 1)); return 1; fi
  fi
  echo "  (check) $1 present${2:+, md5 ${2:0:8}}"
}
chk $A/bin/arw "$(awk '$2 == "arw" {print $1}' $A/bin/MD5SUMS 2>/dev/null)"
chk $A/arwsum.py $SUMMD5
ST=$(timeout 60 python3 -I $A/arwsum.py --selftest 2>&1 | tail -1)
case "$ST" in "SELFTEST OK"*) echo "  (check) arwsum.py --selftest: $ST" ;; *) echo "  (check) arwsum.py --selftest MISSING (${ST:0:300})"; miss=$((miss + 1)) ;; esac
IID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)
if [ -n "$IID" ]; then
  echo "  (check) image $IMAGE present ($IID)"
  # no GPU device in either container: the library md5, the case literal in r4d_selftest and libr4d.so, r4d_selftest --help
  PROBE=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c "md5sum $LIBP | cut -c1-32; grep -c $CASE $STB; grep -c $CASE $LIBP; $STB --help 2>&1 | grep -c -- '--case'" 2>&1 | tr '\n' ' ')
  set -- $PROBE
  if [ "${1:-}" = "$LIBMD5" ]; then echo "  (check) image libr4d.so md5 ${1:0:8} = the library arw was checked on"; else echo "  (check) image libr4d.so md5 '${1:-?}' != $LIBMD5 MISSING"; miss=$((miss + 1)); fi
  if [ "${2:-0}" -ge 1 ] 2>/dev/null && [ "${3:-0}" -ge 1 ] 2>/dev/null && [ "${4:-0}" -ge 1 ] 2>/dev/null; then
    echo "  (check) r4d_selftest --help (no device) has --case; '$CASE' in r4d_selftest ($2) and libr4d.so ($3)"
  else echo "  (check) r4d_selftest case $CASE / --case MISSING (probe: $PROBE)"; miss=$((miss + 1)); fi
  AST=$(timeout 120 docker run --rm --network none -v $A/bin:/b:ro --entrypoint /b/arw $IMAGE --lib $LIBP --selftest $ARGS 2>&1)
  if echo "$AST" | grep -q '^ARW SELFTEST OK' && [ "$(echo "$AST" | grep -c 'INTERPOSE .* -> OK$')" = 3 ] && ! echo "$AST" | grep -q 'coded_ok 0$'; then
    echo "  (check) arw --selftest in the image (no device): OK -- $(echo "$AST" | grep -m1 '^ARW LIB' | cut -c9-200)"
    echo "$AST" | grep -E '^ARW (INTERPOSE|TUNE|CODEDOK)' | sed 's/^/      /'
  else echo "  (check) arw --selftest in the image MISSING:"; echo "$AST" | tail -12 | sed 's/^/      /'; miss=$((miss + 1)); fi
else echo "  (check) image $IMAGE MISSING"; miss=$((miss + 1)); fi
FREE=$(df -Pm $HOME | awk 'NR == 2 {print $4}')
if [ "$DRY" = 1 ]; then
  echo "  (check) disk $FREE MB free ($([ "${FREE:-0}" -ge $DISKMIN ] && echo OK || echo "LOW: the run refuses below $DISKMIN"))"
  echo "+ mkdir -p $O"
  echo "  (dry) the item: $(docker ps --format '{{.Names}}' | grep -ci radiance) radiance container(s) up now (the run refuses while any is)"
else
  [ $miss = 0 ] || { log "$T: VIRHE $miss file(s) missing or stale (see the check lines) -- not run"; exit 1; }
  [ "${FREE:-0}" -ge $DISKMIN ] || { log "$T: VIRHE disk $FREE MB free < $DISKMIN -- not run"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "$T: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "$T: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  mkdir -p "$O"
fi
log "$T: start IMAGE=$IMAGE ($IID) libr4d ${LIBMD5:0:8} arw $(md5sum < $A/bin/arw 2>/dev/null | cut -c1-8) arwsum $(md5sum < $A/arwsum.py 2>/dev/null | cut -c1-8), 4 cards, disk $FREE MB -> $O (~3 min estimate, bounded 12 min; 12.11. fix: tty selftests, plain case first, a lane hang does not stop arw)"
# 12.11. FIX (ds-main-port, after the 11:14 run: lane case rc 137 at the 300 s limit with an EMPTY output file -- r4d_selftest's
# ok() is a plain printf, fully buffered into a file, so a SIGKILL drops everything it printed): both selftest runs now get a
# pseudo-TTY (docker run -t: stdio line-buffers on a tty; the image has no stdbuf), so a hang shows its last line; P0 plain first
# (the known-good case: does the four-rank selftest rendezvous work on the post-riser layout at all?); a lane HANG no longer stops
# the reading (arw runs: it binds the launchers itself and prints with fflush), a lane FAIL still does.
# P0 plain: libr4d's known-good four-rank two-shot case (post-move rendezvous check, informative, never stops the reading)
run "gpurun xarw-sp-\$\$ $TSP -t $DEV --entrypoint $STB $IMAGE $LIBP --case $PCASE > $O/selftest-plain.txt 2>&1"; rcp=$?
if [ "$DRY" != 1 ]; then
  echo "SELFTEST_RC $rcp" >> $O/selftest-plain.txt
  pok=$(grep -c '^  ok  ' $O/selftest-plain.txt); pf=$(grep -c 'FAIL' $O/selftest-plain.txt)
  log "$T selftest --case $PCASE (4 cards, post-move rendezvous check): rc $rcp, $pok ok, $pf FAIL$({ [ $rcp = 124 ] || [ $rcp = 137 ]; } && echo " -- STOPPED by the limit (hang); last line: $(grep -v '^SELFTEST_RC' $O/selftest-plain.txt | tr -d '\r' | tail -1 | cut -c1-200)")"
fi
# P0: libr4d's own lane case (plain and coded vs the fp32 fold at its sizes), line-buffered
run "gpurun xarw-st-\$\$ $TST -t $DEV --entrypoint $STB $IMAGE $LIBP --case $CASE > $O/selftest-lane.txt 2>&1"; rc0=$?
STCLEAN=1
if [ "$DRY" != 1 ]; then
  echo "SELFTEST_RC $rc0" >> $O/selftest-lane.txt
  nok=$(grep -c '^  ok  ' $O/selftest-lane.txt); ncod=$(grep '^  ok  ' $O/selftest-lane.txt | grep -c '(coded'); nf=$(grep -c 'FAIL' $O/selftest-lane.txt); ns=$(grep -c '^  skip' $O/selftest-lane.txt)
  hung=0; { [ $rc0 = 124 ] || [ $rc0 = 137 ]; } && hung=1
  log "$T selftest --case $CASE (4 cards): rc $rc0, $nok ok ($ncod on the coded form), $nf FAIL, $ns skip$([ $nf -gt 0 ] && echo " -- first: $(grep -m1 FAIL $O/selftest-lane.txt | tr -d '\r' | cut -c1-300)")$([ $hung = 1 ] && echo " -- STOPPED by the limit (hang); last line: $(grep -v '^SELFTEST_RC' $O/selftest-lane.txt | tr -d '\r' | tail -1 | cut -c1-200)")"
  if [ $nf -gt 0 ]; then STCLEAN=0
  elif [ $hung = 1 ]; then log "$T: lane case HUNG (not a byte difference) -- arw runs anyway (fix 12.11.: a hang does not stop the reading)"
  else { [ $rc0 = 0 ] && [ $ns = 0 ] && [ $nok -gt 0 ] && [ $ncod -gt 0 ]; } || STCLEAN=0; fi
fi
# P1: DS shapes, plain vs coded, identity + time (only after a clean P0: a difference stops the reading)
rc1=0
if [ $STCLEAN = 1 ]; then
  run "gpurun xarw-\$\$ $TARW $DEV -v $A/bin:/b:ro --entrypoint /b/arw $IMAGE --lib $LIBP $ARGS > $O/arw.txt 2>&1"; rc1=$?
  [ "$DRY" = 1 ] || log "$T arw (rc $rc1): $(grep -m1 '^ARW PEER' $O/arw.txt) | $(grep -m1 '^ARW DONE' $O/arw.txt || echo 'NO DONE: '"$(tail -1 $O/arw.txt | cut -c1-300)")"
else
  log "$T: VIRHE r4d_selftest's lane case is not clean -- the reading stops before arw (see $O/selftest-lane.txt)"
fi
run "python3 -I $A/arwsum.py $O > $O/arwsum.txt 2>&1"
if [ "$DRY" != 1 ]; then
  cat $O/arwsum.txt
  while IFS= read -r l; do log "$T ${l:0:2400}"; done < <(grep -E '^(CHECKS|SHAPE|SUMMARY|ESTIMATE|VIRHE)' $O/arwsum.txt)
  grep -q '^CHECKS' $O/arwsum.txt || log "$T: VIRHE arwsum.py printed no CHECKS line ($(tail -2 $O/arwsum.txt | tr '\n' ' ' | cut -c1-300))"
  [ $rc1 = 0 ] || log "$T: VIRHE arw exit $rc1 (1 = a VIRHE inside: a byte difference, a wrong wire, a failed launch; 2 = usage; 3 = HIP / setup; 124/137 = timeout (hang), else a crash)"
else
  echo "DRYRUN: $miss MISSING, 0 VIRHE (nothing started on a GPU, nothing written)"
  [ $miss = 0 ] || exit 1
fi
log "$T: done -- $O"
