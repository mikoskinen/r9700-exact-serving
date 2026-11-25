# ngshare.sh (12.11.2026, ds-main-port ngshare builder) -- the one-line opt-in for a NEW / copied DS body: source it AFTER
# the body sources its lib (lib-29.sh or lib-pr266.sh) and AFTER its `Y=$HD/ds41-stack2-tp4.yaml` line:
#
#     . $HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngshare.sh || exit 1
#
# It then (1) points Y at ds41-ngshare-tp4.yaml (= stack2's yaml + /dev/shm/ds41ng:/ngram), (2) wraps lib-pr266's restart()
# (so gate / gate1 too): NGRAM=ram=/ngram for the serve line (whatever best_vars set), `ngram-keeper.sh prep` before the
# start, and after a healthy start `ngram-keeper.sh start --wait 120 --expect 2` + one RESULT line naming filled / reused.
# CORRECTNESS-ONLY bodies (BI traces, identity / greedy sets, smoke): never in a speed body (4 KiB tmpfs pages).
# The tables stay in /dev/shm/ds41ng (~103 GiB RAM) after the body; ngram-shm-cleanup.sh frees them (before a speed block,
# before produp). DRYRUN=1 prints the added commands.
NGS_HERE=$HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare
if [ "$(basename "${Y:-}")" != ds41-stack2-tp4.yaml ]; then
  echo "ngshare.sh: Y is '${Y:-unset}', not ds41-stack2-tp4.yaml -- ds41-ngshare-tp4.yaml copies only that yaml"; return 1
fi
declare -F restart > /dev/null || { echo "ngshare.sh: no restart() -- source the body's lib first"; return 1; }
Y=$NGS_HERE/ds41-ngshare-tp4.yaml
eval "$(declare -f restart | sed '1s/^restart /_ngs_restart /')"
restart() {
  local tag=$1 rc k pl
  export NGRAM=ram=/ngram
  if [ "${DRY:-0}" = 1 ]; then
    echo "+ $NGS_HERE/ngram-keeper.sh prep"; _ngs_restart "$@"; rc=$?
    echo "+ $NGS_HERE/ngram-keeper.sh start --wait 120 --expect 2   (Y=$Y NGRAM=$NGRAM)"; return $rc
  fi
  "$NGS_HERE/ngram-keeper.sh" prep > /dev/null 2>&1 || {
    log "RESULT ngshare $tag: VIRHE $("$NGS_HERE/ngram-keeper.sh" prep 2>&1 | tail -1 | cut -c1-300) -- not started"; return 1; }
  _ngs_restart "$@"; rc=$?
  [ $rc = 0 ] || return $rc
  k=$("$NGS_HERE/ngram-keeper.sh" start --wait 120 --expect 2 2>&1 | tail -1)
  pl=$(grep -o 'n-gram table in pinned memory the cards read, .*' "$O/start-$tag.log" 2> /dev/null | cut -c1-200 | tr '\n' '|')
  log "RESULT ngshare $tag: ${pl:-no shared-table placement line (VIRHE?)} keeper: $k"
  return 0
}
