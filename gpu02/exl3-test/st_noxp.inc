# st_noxp.inc (6.11.2026) -- proposed after-sweep.sh step "noxp", for review; not sourced by anything yet.
# Engine build radiance-local:ds41-exl3-9e8ccb64-3684c0b1-noxp-493f9f89 = the window image with only /opt/radiance/bin/radiance
# replaced (branch ds41-noxp 493f9f89 on 9e8ccb6): the next-layer expert prefetch is not opened for a model wider than its
# predictor (n_embd 5120 > 4096; Ctx::xp_predict_run never predicts there), so no idle 4th stream / DMA queue / ring / thread
# a card. Expected start-log evidence: "I expert prefetch off: n_embd 5120 ..." and "hardware queues ... 3 a card" (now 4).
# Same config as the window's best instance (exl3-gate.sh defaults: NST0 CKINT 2048 span 2 IL8 MAXSEQS 8 MML 262144 RAM tier,
# HOSTPOOL 19456), same probe / mb args as window.sh, so every number compares 1:1 with today's window (guard 19.36 ms,
# probe 52.9 / 51.6 tok/s, c4 34.5 ms/step 99.0 tok/s, c8 54.5 ms/step 123.8 tok/s) on the same cards and image otherwise.
# Integration (coordinator): copy the image to gpu02 first (docker save | ssh gpu02 docker load); in after-sweep.sh add
#   . $T/st_noxp.inc            (after the other st_* functions)
#   EST[noxp]=13                (after the declare -A EST line)
#   and "noxp" to STEPS where it should run (e.g. right after live0, or last).
# Estimate: gate 4.5 + probe 0.6 + mb c4 3.6 + mb c8 3.6 = ~12.5 min. Uses after-sweep.sh's helpers (gate in_d log newdir
# probefmt mbline mbkey keeplog) and its AS_GUARD_MS guard (the slow-state rate with and without the 4th queue is the point:
# the guard line's first-try value is recorded either way; SLOW-after-retry -> VIRHE, measurements skipped, as elsewhere).
NOXP_IMAGE=${NOXP_IMAGE:-radiance-local:ds41-exl3-9e8ccb64-3684c0b1-noxp-493f9f89}
st_noxp() {
  local O=$AS/noxp q x
  if [ "$DRY" != 1 ] && ! docker image inspect "$NOXP_IMAGE" > /dev/null 2>&1; then
    log "RESULT after-sweep noxp: VIRHE image $NOXP_IMAGE not on gpu02 -> skipped"; return 0; fi
  gate noxp "IMAGE=$NOXP_IMAGE" NST=0 CKINT=2048 MAXSEQS=8 MML=262144 "EXTRA2=$BEST2X" || return 0
  newdir $O
  if [ "$DRY" != 1 ]; then
    q=$(grep -m1 -o 'hardware queues .*a card' $R/start-as-noxp.log 2>/dev/null | tr -s ' ')
    x=$(grep -m1 -o 'expert prefetch off[^;]*' $R/start-as-noxp.log 2>/dev/null | cut -c1-120)
    log "RESULT noxp start: ${x:-NO expert-prefetch-off line (old binary?)}; ${q:-hardware queues ?}; guard $(cat $R/guard-as-noxp.txt 2>/dev/null) | window (4 queues): fast 19.36"
  fi
  in_d 900 $O/probe.txt python3 stepprobe.py essay,code 512
  log "RESULT noxp probe (1 user, greedy, 512 tok; no idle prefetch queue): $(probefmt $O/probe.txt) | window, same config, 4 queues: $(probefmt $R/probe.txt)"
  in_d 1800 $O/mb-c4.txt python3 mb.py --tag noxp --phases c4
  log "RESULT noxp mb c4: $(mbline $O/mb-c4.txt) | window: $(mbkey $R/mb-c4.txt ms_decode_step) $(mbkey $R/mb-c4.txt agg_dec_tps)"
  in_d 1800 $O/mb-c8.txt python3 mb.py --tag noxp --phases c8
  log "RESULT noxp mb c8: $(mbline $O/mb-c8.txt) | window: $(mbkey $R/mb-c8.txt ms_decode_step) $(mbkey $R/mb-c8.txt agg_dec_tps)"
  keeplog $O
}
