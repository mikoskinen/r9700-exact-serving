#!/bin/bash
# dsm-rfst.sh (13.11.2026, ds41-fuse builder for ds-main-port; the body of items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16z-rfst.sh)
# -- lever FUSE-R's form 1 above form 2's band on ONE card BY CHIP UUID (ROCR_VISIBLE_DEVICES=GPU-<unique_id>, default
# GPU-dbba5c851cfef7b3), every radiance container DOWN, no server, no engine start, no compose; ~10-30 s expected (int15's P0 rfuse
# took 4 s), the selftest bounded at 300 s.
# WHY: ds41_rfuse 2 (int16) serves a MoE router segment of 49 .. 170 rows (a prefill segment; 170 = 1024 / top_k) with FORM 1
# (gemm_nt_f32 -> router_scored_topk_scatter), form 2 only up to 48 rows. The on-card selftest the int16 screen ran (zzzfuse P0,
# image 4419a17d60a7) covered M <= 48 only. This item runs the same selftest with M 49, 64, 96, 128, 160, 169, 170 added.
# WHAT: image radiance-local:ds41-fuse-e231d919-38b88ec6-st2 (be78eec84ba0) = 4419a17d60a7 + bin/r4d_selftest of ds41-fuse dcee256
# (bin/radiance, libr4d.so, deepseek41.so byte-identical to 4419a17d60a7, sha256 checked at build: build/fuse/sha-st2.txt).
# `r4d_selftest libr4d.so --case rfuse`:
#   - the stock chain gemm_nt_f32 -> router_scored_topk -> moe_scatter vs form 1, and vs form 2 where it applies (M <= 48; above
#     48 form 2 must refuse);
#   - byte for byte: logits, ids, weights, sorted_tok, expert_offset, expert_count + guard tails;
#   - bf16 and f32 weights x random / tied ranks / NaN rows x bases 0 / 96 / 288 x M 1-9, 11, 12, 13, 15, 16, 24, 32, 40, 48, 49, 64,
#     96, 128, 160, 169, 170.
# VERDICT "RFST M 49..170: PASS" = rc 0, 151 ok lines, 0 FAIL lines, the info line "rfuse: 450 of 450 ... byte-identical" and 42 ok
# lines of the new form-1-only rows. Anything else = FAIL, said with "int16 soak result VOID: fall back to int15" and every FAIL
# line logged. Not started (a guard below) = "NOT RUN", the int16 soak result UNCONFIRMED until this item runs.
# Out runs/xdsm/rfst (selftest.txt, ~40 kB). Measurement only. Tag "RESULT rfst ".
# GUARD (NOT RUN): the image id; the selftest literal of the new rows in the image; the card's unique_id in sysfs; disk >=
# RFST_DISKMIN_MB (11000) MB free on /, /tmp inodes < 80 %; a radiance container or KFD client up (60 s wait for KFD clients).
# DRYRUN=1 bash dsm-rfst.sh : prints every command, checks the image / literal / card / disk; starts nothing GPU-side, writes nothing.
# RFST_MOCK=1 (queue-ar/tools/mockrad only, never in the queue): the image-id check is logged, not counted (the mock's docker shim has
# fake ids), the KFD-client wait is skipped; RFST_MOCK_OUT=FILE stands in for the selftest's output (RFST_MOCK_RC its exit code,
# default 0), else the GPU run goes to the shim, which refuses it (exit 125 -> the FAIL path).
set -u
DRY=${DRYRUN:-0}
O=${RFST_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/rfst}
L=${RFST_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${RFST_IMAGE:-radiance-local:ds41-fuse-e231d919-38b88ec6-st2}; IMGID=${RFST_IMGID:-be78eec84ba0}
UUID=${RFST_UUID:-GPU-dbba5c851cfef7b3}
DISKMIN=${RFST_DISKMIN_MB:-11000}
TMO=${RFST_TMO:-300}
MOCK=${RFST_MOCK:-0}
LIT="form 2 refuses > 48 rows"
WANT_OK=151; WANT_NEW=42; WANT_CHK="450 of 450"
T="RESULT rfst"
VOID="int16 soak result VOID: fall back to int15"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e ROCR_VISIBLE_DEVICES=$UUID"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
mk() { if [ "$MOCK" = 1 ]; then echo "  (mock: not counted) $*"; else echo "  (check) $*"; miss=$((miss + 1)); fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
notrun() { log "$T: RFST M 49..170: NOT RUN ($*) -- int16 soak result UNCONFIRMED until this item runs"; exit 1; }
miss=0
gid=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)
if [ "$gid" != "$IMGID" ]; then mk "image $IMAGE id ${gid:-missing} != $IMGID"; else echo "  (check) image $IMAGE $gid"; fi
if timeout 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c "grep -qzF -- '$LIT' /opt/radiance/bin/r4d_selftest" > /dev/null 2>&1; then
  echo "  (check) r4d_selftest in the image has the M 49..170 rows ('$LIT')"
else echo "  (check) r4d_selftest in $IMAGE lacks '$LIT' (not the st2 selftest)"; miss=$((miss + 1)); fi
nuid=$(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | grep -cx "${UUID#GPU-}")
if [ "$nuid" != 1 ]; then echo "  (check) card $UUID: $nuid sysfs unique_id matches (want 1)"; miss=$((miss + 1)); else echo "  (check) card $UUID: in sysfs (unique_id)"; fi
free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9); ipct=$(df -i /tmp | awk 'NR==2 {print $5}' | tr -dc 0-9)
if [ "${free_mb:-0}" -lt $DISKMIN ] || [ "${ipct:-100}" -ge 80 ]; then echo "  (check) disk: / ${free_mb} MB free (min $DISKMIN), /tmp inodes ${ipct} % (max 79)"; miss=$((miss + 1)); else echo "  (check) disk: / ${free_mb} MB free, /tmp inodes ${ipct} %"; fi
if [ "$DRY" = 1 ]; then
  echo "+ mkdir -p $O; chmod 777 $O"
  echo "+ gpurun rfst-\$\$ $TMO $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case rfuse > $O/selftest.txt 2>&1"
  echo "+ RESULT rfst lines: 'RFST M 49..170: PASS|FAIL (rc, ok n/$WANT_OK, FAIL n, '$WANT_CHK', new rows n/$WANT_NEW)', on FAIL '$VOID' + every FAIL line; FUSEPERF / FUSEGATE / info"
  echo "  (dry) $(docker ps --format '{{.Names}}' | grep -ci radiance) radiance container(s) up now, $(ls -A /sys/class/kfd/kfd/proc 2>/dev/null | wc -l) KFD client(s) (the run refuses while any is)"
  echo "DRYRUN: $miss MISSING (nothing started, nothing written)"
  [ $miss = 0 ] || exit 1
  exit 0
fi
[ $miss = 0 ] || notrun "VIRHE $miss check(s) failed: image / selftest literal / card / disk"
docker ps --format '{{.Names}}' | grep -qi radiance && notrun "VIRHE a radiance container is running"
if [ "$MOCK" = 1 ]; then echo "  (mock: not checked) KFD clients: $(ls -A /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)"; else
for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
[ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || notrun "VIRHE a KFD client is running (after 60 s wait)"
fi
mkdir -p "$O"; chmod 777 "$O"
log "$T: start IMAGE=$IMAGE ($gid) card $UUID: r4d_selftest --case rfuse (form 1 at M 49..170 + the M <= 48 rows) -> $O (bounded $TMO s)"
if [ "$MOCK" = 1 ] && [ -n "${RFST_MOCK_OUT:-}" ]; then
  cat "$RFST_MOCK_OUT" > $O/selftest.txt; rc=${RFST_MOCK_RC:-0}; echo "  (mock) selftest output from $RFST_MOCK_OUT, rc $rc"
else
  gpurun rfst-$$ $TMO $DEV --entrypoint /opt/radiance/bin/r4d_selftest "$IMAGE" /opt/radiance/share/radiance/kernels/libr4d.so --case rfuse > $O/selftest.txt 2>&1; rc=$?
fi
F=$O/selftest.txt
nok=$(grep -c '^  ok  ' $F); nfail=$(grep -c '^  FAIL' $F)
nnew=$(grep '^  ok  ' $F | grep -cF "$LIT")
chk=$(grep -oE 'rfuse: [0-9]+ of [0-9]+ \(weights, mode, M, base\) checks byte-identical' $F | tail -1)
while IFS= read -r l; do log "$T ${l:0:300}"; done < <(grep -E '^  (FUSEPERF|FUSEGATE|info) ' $F)
why=""
[ "$rc" = 0 ] || why="$why rc $rc$( [ "$rc" = 124 ] || [ "$rc" = 137 ] && echo ' (timeout)');"
[ "$nok" = $WANT_OK ] || why="$why ok $nok != $WANT_OK;"
[ "$nfail" = 0 ] || why="$why $nfail FAIL line(s);"
case "$chk" in "rfuse: $WANT_CHK "*) ;; *) why="$why checks '$(echo "$chk" | awk 'NF {print $2 " of " $4}')' != '$WANT_CHK';" ;; esac
[ "$nnew" = $WANT_NEW ] || why="$why form-1-only rows $nnew != $WANT_NEW;"
chn=$(echo "$chk" | awk 'NF {print $2 " of " $4}'); chn=${chn:-missing}
cnt="rc $rc, ok $nok/$WANT_OK, FAIL $nfail, checks $chn byte-identical (want $WANT_CHK), form-1-only rows M 49..170 $nnew/$WANT_NEW"
if [ -z "$why" ]; then
  log "$T: RFST M 49..170: PASS ($cnt) -- form 1 byte for byte vs the stock chain at M 49..170 (and forms 1 / 2 at M <= 48): int16's soak result stands"
else
  log "$T: RFST M 49..170: FAIL ($cnt; why:${why%;}) -- $VOID"
  while IFS= read -r l; do l=${l#  FAIL  }; log "$T FAIL line: ${l:0:400}"; done < <(grep '^  FAIL' $F)
  [ "$nok$nfail" = 00 ] && log "$T: no ok / FAIL lines; tail: $(tail -3 $F | tr '\n' ' ' | cut -c1-300)"
fi
log "$T: done -- $F"
