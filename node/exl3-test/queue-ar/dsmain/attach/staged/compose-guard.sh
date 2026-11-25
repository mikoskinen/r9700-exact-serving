# compose-guard.sh (12.11.2026, ds-main-port; coordinator: guard before compose up after the 11:59 failures) -- SOURCE it, then call
#   cguard [CONTAINER_NAME] [COMPOSE_PROJECT]   (defaults radiance-ds41-exl3 / radiance-ds41-exl3)
# Removes leftover containers of that project / name in Created, Dead or Exited state (docker rm -f, one at a time, each logged).
# Running containers are never touched. A remove that answers "No such container" is a GHOST entry (the daemon lists it but cannot
# act on it, or "removes" it with rc 0 while it stays listed: 622b5b7a9e64 after the 11:46 reboot): carry on, and -- because compose would try to restart / recreate the ghost and
# fail ("No such container", d0ab / cap48r 11:59) -- export COMPOSE_PROJECT_NAME=<project>-g<HHMMSS> so this item's compose up
# starts in a fresh project (container_name stays the same; compose files are NOT edited). Logs one RESULT line to runs/ds41-exl3.log.
cguard() {
  local cn=${1:-radiance-ds41-exl3} proj=${2:-radiance-ds41-exl3} lg=${CGUARD_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
  local rows id st out removed="" ghosts="" failed="" kept="" w=0
  if [ "${SIGONLY:-0}" = 1 ]; then return 0; fi   # 12.11. ATTACH: a SIGONLY pass never touches a container
  # 12.11. ATTACH (ds-main-port, opt-in, queue-ar/dsmain/attach/attach.sh): with ATTACH=1 / KEEP=1, or a record of a server an ATTACH /
  # KEEP item left up: claim our kept server for this item (exports ATTACH_CLAIMED_CN) or stop it gracefully (docker stop -t 60, rm).
  # No flag and no record = this block does nothing.
  local ast=${ATTACH_STATE:-$HOME/gpu/ds41/exl3-test/runs}
  if [ "${ATTACH:-0}" = 1 ] || [ "${KEEP:-0}" = 1 ] || [ -e $ast/server.sig ] || [ -e $ast/server.keep ]; then
    [ -s $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/attach/attach.sh ] && . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/attach/attach.sh && att_cguard "$cn"
  fi
  # 12.11. 13:25 (coordinator): a previous item's trap-down can lag a few seconds -- WAIT up to CGUARD_WAIT (60) s for running
  # radiance containers to stop before this item's own "cards free" check (never stops one; the item's check still decides)
  if [ -n "${ATTACH_CLAIMED_CN:-}" ]; then   # 12.11. ATTACH: the claimed server is not waited for
    while docker ps --format '{{.Names}}' 2>/dev/null | grep -iE '^radiance' | grep -vxF "$ATTACH_CLAIMED_CN" | grep -q . && [ $w -lt ${CGUARD_WAIT:-60} ]; do sleep 2; w=$((w + 2)); done
  else
  while docker ps --format '{{.Names}}' 2>/dev/null | grep -qiE '^radiance' && [ $w -lt ${CGUARD_WAIT:-60} ]; do sleep 2; w=$((w + 2)); done
  fi
  rows=$( { docker ps -a --filter "label=com.docker.compose.project=$proj" --format '{{.ID}}|{{.Status}}|{{.Names}}'
            docker ps -a --filter "name=$cn" --format '{{.ID}}|{{.Status}}|{{.Names}}'; } 2>/dev/null | sort -u )
  while IFS='|' read -r id st nm; do
    [ -n "$id" ] || continue
    case "$st" in
      Created*|Dead*|Exited*)
        out=$(docker rm -f "$id" 2>&1)
        if [ $? = 0 ] && ! docker ps -a --format '{{.ID}}' | grep -q "^${id:0:12}"; then removed="$removed $id(${st%% *}${nm:+ $nm})"
        elif docker ps -a --format '{{.ID}}' | grep -q "^${id:0:12}" && { [ -z "$out" ] || echo "$out" | grep -qE "No such container|^${id}"; }; then ghosts="$ghosts $id(${st%% *}, still listed after rm)"
        elif echo "$out" | grep -q "No such container"; then ghosts="$ghosts $id(${st%% *})"
        else failed="$failed $id(${st%% *}: $(echo "$out" | head -1 | cut -c1-80))"; fi ;;
      *) kept="$kept $id(${st%% *}${nm:+ $nm})" ;;
    esac
  done <<< "$rows"
  if [ -n "$ghosts" ]; then export COMPOSE_PROJECT_NAME=$proj-g$(date +%H%M%S); fi
  echo "$(date '+%F %T') RESULT cguard ($cn / $proj): removed:${removed:- none} | ghost (No such container, carried on):${ghosts:- none}$([ -n "$ghosts" ] && echo " -> COMPOSE_PROJECT_NAME=$COMPOSE_PROJECT_NAME for this item") | remove failed:${failed:- none} | not touched (running):${kept:- none} | waited ${w} s for running radiance containers" | tee -a $lg
  return 0
}
