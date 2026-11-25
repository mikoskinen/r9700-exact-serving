#!/bin/bash
# sigprobe.sh (12.11.2026, queue-ar/dsmain/attach/mock, ds-main-port ATTACH builder) -- a mockrun.sh TARGET (sandbox only):
#  (1) the SIGONLY signature of each recent real DS item wrapper (items/, as staged; env -i, SIGONLY=1 DRYRUN=1, guard bin first):
#      which items share a start, how long a SIGONLY pass takes;
#  (SIGPROBE_SKIP1=1 skips (1); the mock's "VIRHE XSPR_SW=0" / "XARC_SW=0" markers need --env KEEP_VIRHE_OK=1 for a keep)
#  (2) a REAL-BODY round trip in a sandbox queue: arc -> spread -> spread (patched bodies via mockrun --swap, ATTACH=1 KEEP=1 and the
#      "# ATTACH-SIG:" line added to wrapper copies, XARC_SW=0 / XSPR_SW=0 short knobs): keep / attach / record lines.
#   ./mockrun.sh --name att-real --timeout 1800 --no-dryrun --fresh gpu/ds41/exl3-test/runs/xdsm/arc --fresh gpu/ds41/exl3-test/runs/xdsm/spread \
#     --fresh gpu/ds41/exl3-test/runs/xdsm/spread2 --swap <lib-pr266 / compose-guard / the two bodies> ~/gpu/.../attach/mock/sigprobe.sh
set -u
[ "${DRYRUN:-0}" = 1 ] && { echo "+ sigprobe.sh: DRYRUN, nothing run"; exit 0; }
case "${MOCKRAD_STATE:-}" in "") echo "sigprobe.sh runs under queue-ar/tools/mockrad/mockrun.sh only"; exit 2 ;; esac
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; IT=$T/queue-ar/dsmain/items
. $T/queue-ar/dsmain/attach/attach.sh
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
[ "${SIGPROBE_SKIP1:-0}" = 1 ] || { log "SIGPROBE (1): SIGONLY signatures of the real item wrappers"
for w in 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc.sh 57-p4zyzzzzz-ds-r92z-spread.sh 57-p4zyzzzzz-ds-r92zz-ngreuse.sh \
         57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u.sh 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub.sh 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12.sh \
         57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-l-i9p.sh 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b.sh 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-k-cap48r.sh \
         57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-j-d0ab.sh 57-p4zyzzzzz-ds-r00zzz-cz-a-base.sh 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cf-arw2.sh \
         57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5.sh 57-p4zyzzzzz-ds-r92-ccfill.sh 57-p4zyzzzzz-ds-r00zzz-cz-mcl0probe.sh; do
  [ -f $IT/$w ] || { log "SIGPROBE $w: no such wrapper"; continue; }
  t0=$(date +%s%N); so=$(att_sigonly $IT/$w); t1=$(date +%s%N)
  log "SIGPROBE $w: $(echo "${so:-NONE (no ATTACH-SIG line: the DRY path did not reach restart)}" | awk '{ if ($1 == "ATTACH-SIG") print substr($2, 1, 12), $5; else print }') in $(( (t1 - t0) / 1000000 )) ms"
done; }
log "SIGPROBE (2): real-body round trip arc -> spread -> spread2 (ATTACH=1 KEEP=1, mock knobs)"
export ATTACH_QDIR=$HOME/attq2; Q=$ATTACH_QDIR; rm -rf $Q; mkdir -p $Q/done
mkq() {   # mkq SRC DST "KNOBS": a copy of a real wrapper with the ATTACH declaration, the opt-in exports and mock knobs
  { sed -n 1p $IT/$1; echo "# ATTACH-SIG: SIGONLY=1 DRYRUN=1 bash <this item> prints its start signature (queue-ar/dsmain/attach/attach.sh)"
    echo 'export ATTACH=${ATTACH:-1} KEEP=${KEEP:-1}'; echo "export $3"; sed -n '2,$p' $IT/$1; } > $Q/$2; }
mkq 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc.sh 01-arc.sh 'XARC_SW=0 XARC_COLD="12000B 12000C 6000C 6000B" XARC_NTOK=32 XARC_C8DUR=12 XARC_C8WARM=4 XARC_DDUR=12 XARC_DWARM=4 XARC_EGEN=128 XARC_BUDGET_MIN=120'
mkq 57-p4zyzzzzz-ds-r92z-spread.sh 02-spread.sh 'XSPR_SW=0 XSPR_DUR=12 XSPR_WARM=4 XSPR_ORDER="4 1" XSPR_BUDGET_MIN=120'
mkq 57-p4zyzzzzz-ds-r92z-spread.sh 03-spread2.sh "XSPR_SW=0 XSPR_DUR=12 XSPR_WARM=4 XSPR_ORDER=\"4 1\" XSPR_BUDGET_MIN=120 XSPR_OUT=$R/xdsm/spread2 XSPR_ITEM=spread2"
while true; do
  item=$(ls $Q/[0-9][0-9]-*.sh 2>/dev/null | sort | head -1); [ -n "$item" ] || break
  n=$(basename $item); mv $item $Q/done/$n
  log "RESULT chain-after: queue item $n start"; bash $Q/done/$n > $R/queue-$n.out 2>&1; log "RESULT chain-after: queue item $n exit $?"
  log "SIGPROBE after $n: radiance containers up: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')-"
done
sleep 3
log "SIGPROBE done: radiance containers up at the end: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')-"
