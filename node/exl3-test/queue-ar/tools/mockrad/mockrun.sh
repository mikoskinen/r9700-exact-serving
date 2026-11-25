#!/bin/bash
# mockrun.sh (12.11.2026, queue-ar/tools/mockrad, ds-main-port mockrad builder) -- run a staged queue item (its wrapper or its body)
# END TO END against mockrad.py, NO GPU: every block / summary / parser stage of the body runs on canned but live server output,
# so a TypeError, a wrong anchor or a guard that VIRHEs every block shows up before the item costs GPU time.
#
#   mockrun.sh [options] ITEM_OR_BODY.sh
#     ITEM_OR_BODY   a real path (~/gpu/ds41/... or relative to queue-ar/dsmain); the SANDBOX COPY of it runs, never the real file
#     --name N       run name (default: the script's name + time); the sandbox is ${MOCKRAD_TMP:-/tmp/mockrad}/N
#     --env K=V      a knob for the run (repeatable): shorten it -- e.g. XS5_DUR=12 XS5_WARM=4 XS5_ORDER0=AS
#     --swap REL=SRC replace sandbox file REL (relative to ~) by a copy of SRC (real path, or relative to ~): test an old / new helper
#     --fresh REL    a directory (relative to ~) that starts empty (default: every `mkdir -p` the body's DRYRUN prints)
#     --fault F      a mockrad fault (repeatable; see mockrad.py --help): dead=psk, leak=engdd, noapply=ds41_psk, cached=40, ...
#     --timeout S    stop the body after S s (default 900)
#     --no-dryrun    skip the DRYRUN pass (then give --fresh for the output dir)
#     --no-replay    do not serve the recorded warm-up / greedy-set answers (identity vs the stock references then differs)
#     --decode-scale X / --prefill-scale X   mock speed (default 0.5 / 0.1 of the real step times)
#     --keep-sandbox keep the sandbox even when the run is clean (default: kept; the report names it)
# Output: a report (Python tracebacks with their files, VIRHE lines, bash errors, missing files, refused / not-emulated calls,
# unknown endpoints, refused fn4 POSTs, the leak check) in SANDBOX/report.txt and queue-ar/tools/mockrad/reports/N.txt.
# Exit 0 = no traceback, no bash error, no VIRHE; 1 = findings; 2 = the run itself could not be set up.
#
# SAFETY (what makes it a no-GPU, no-real-server run): HOME is a symlink farm (mkfarm.py: code copied, references linked, the
# output dir and the log fresh); PATH starts with shims: docker (never the real docker: compose up starts mockrad.py), curl
# (only the mock's port; :8090 is rewritten to it), sudo / ssh / systemctl / pkill / killall / rocm-smi / amd-smi / power tools
# refused; every python3 is a sandbox venv whose .pth loads mockguard.py (sockets: only the mock; writes outside the sandbox,
# GPU sysfs reads and posix_fadvise refused). The run is niced (19, idle I/O). Nothing under ~/gpu is written except
# tools/mockrad/reports/N.txt.
set -u
MR=$(cd "$(dirname "$0")" && pwd)
QA=$HOME/gpu/ds41/exl3-test/queue-ar
NAME=""; ENVS=(); SWAPS=(); FRESH=(); FAULTS=(); TIMEOUT=900; DODRY=1; REPLAY=1; DSC=0.5; PSC=0.1; TGT=""
while [ $# -gt 0 ]; do
  case $1 in
    --name) NAME=$2; shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    --swap) SWAPS+=("$2"); shift 2 ;;
    --fresh) FRESH+=("$2"); shift 2 ;;
    --fault) FAULTS+=("$2"); shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --no-dryrun) DODRY=0; shift ;;
    --no-replay) REPLAY=0; shift ;;
    --decode-scale) DSC=$2; shift 2 ;;
    --prefill-scale) PSC=$2; shift 2 ;;
    --keep-sandbox) shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) TGT=$1; shift ;;
  esac
done
[ -n "$TGT" ] || { sed -n '2,32p' "$0"; exit 2; }
case $TGT in /*) ;; ~*) TGT=${TGT/#\~/$HOME} ;; *) [ -e "$QA/dsmain/$TGT" ] && TGT=$QA/dsmain/$TGT || TGT=$(cd "$(dirname "$TGT")" && pwd)/$(basename "$TGT") ;; esac
TGT=$(realpath "$TGT")
[ -f "$TGT" ] || { echo "no such item / body: $TGT"; exit 2; }
case $TGT in "$HOME"/*) REL=${TGT#$HOME/} ;; *) echo "the item must live under $HOME (got $TGT)"; exit 2 ;; esac
[ -n "$NAME" ] || NAME=$(basename "$TGT" .sh)-$(date +%m%d-%H%M%S)
TMPR=${MOCKRAD_TMP:-/tmp/mockrad}
SB=$TMPR/$NAME; ST=$SB/state; H=$SB/home
[ -e "$SB" ] && { echo "sandbox $SB exists: choose another --name"; exit 2; }
mkdir -p "$ST" "$SB/bin" || exit 2
T0=$(date +%s)
say() { echo "$(date '+%F %T') mockrun $NAME: $*" | tee -a "$SB/mockrun.log"; }
NICE="nice -n 19 ionice -c3"
# ---------------------------------------------------------------- the sandbox
FARGS=(); for f in "${FRESH[@]}"; do FARGS+=(--fresh "$f"); done; for s in "${SWAPS[@]}"; do FARGS+=(--swap "$s"); done
$NICE /usr/bin/python3 -I -B "$MR/mkfarm.py" build --real "$HOME" --sb "$H" --manifest "$SB/manifest.json" "${FARGS[@]}" > "$SB/farm.txt" 2>&1 \
  || { cat "$SB/farm.txt"; echo "farm failed"; exit 2; }
say "farm: $(cat "$SB/farm.txt" | tail -1)"
PYV=$(/usr/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
mkdir -p "$SB/venv/bin" "$SB/venv/lib/python$PYV/site-packages"
ln -s /usr/bin/python3 "$SB/venv/bin/python3"; ln -s python3 "$SB/venv/bin/python"; ln -s python3 "$SB/venv/bin/python$PYV"
printf 'home = /usr/bin\ninclude-system-site-packages = true\nversion = %s\n' "$PYV" > "$SB/venv/pyvenv.cfg"
cp "$MR/shims/mockguard.py" "$SB/venv/lib/python$PYV/site-packages/"
echo "import mockguard" > "$SB/venv/lib/python$PYV/site-packages/00-mockguard.pth"
cp "$MR/shims/docker" "$MR/shims/curl" "$SB/bin/"; chmod +x "$SB/bin/docker" "$SB/bin/curl"
for c in sudo su ssh scp rsync systemctl service pkill killall reboot shutdown poweroff rocm-smi amd-smi nvidia-smi rocminfo \
         amdgpu-fan odset.sh fanset.sh pwrcap.sh crontab at wget nc ncat socat; do cp "$MR/shims/refuse" "$SB/bin/$c"; done
# the port: free, never 8090
PORT=""; mkdir -p "$TMPR/.ports"
for p in $(seq 18090 18189); do   # a reservation (atomic mkdir, released at the end) + a free socket
  mkdir "$TMPR/.ports/$p" 2>/dev/null || { [ -e "$TMPR/.ports/$p/owner" ] && ! kill -0 "$(cat "$TMPR/.ports/$p/owner")" 2>/dev/null && rm -rf "$TMPR/.ports/$p" && mkdir "$TMPR/.ports/$p" 2>/dev/null; } || continue
  echo $$ > "$TMPR/.ports/$p/owner"
  /usr/bin/python3 -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1])))' $p 2>/dev/null && { PORT=$p; break; }
  rm -rf "$TMPR/.ports/$p"
done
[ -n "$PORT" ] && [ "$PORT" != 8090 ] || { echo "no free mock port in 18090..18189"; exit 2; }
export_env() {
  export HOME="$H" PATH="$SB/bin:$SB/venv/bin:/usr/local/bin:/usr/bin:/bin" MOCKRAD_PORT=$PORT MOCKRAD_STATE="$ST" MOCKRAD_SANDBOX="$SB" \
         MOCKRAD_PY="$MR/mockrad.py" MOCKRAD_TEMPLATES="$MR/templates/ds41" MOCKRAD_DECODE_SCALE=$DSC MOCKRAD_PREFILL_SCALE=$PSC \
         MOCKRAD_FAULTS="${FAULTS[*]}" MOCKRAD_START_EXTRA="$ST/start-extra.txt" MOCKRAD_REPLAY="$([ $REPLAY = 1 ] && echo "$ST/replay.jsonl")" \
         MOCKRAD_START_LIB="$ST/startlib.json" \
         PYTHONDONTWRITEBYTECODE=1
  unset DRYRUN
}
# the guard must hold before anything runs
G=$(env MOCKRAD_PORT=$PORT MOCKRAD_STATE="$ST" MOCKRAD_SANDBOX="$SB" "$SB/venv/bin/python3" -I -c '
import socket, sys
ok = "mockguard" in sys.modules
try:
    socket.create_connection(("127.0.0.1", 22), timeout=2); ok = False
except ConnectionRefusedError: pass
try:
    open("'"$HOME"'/gpu/.mockguard-probe", "w"); ok = False
except PermissionError: pass
print("GUARD OK" if ok else "GUARD FAIL")' 2>&1 | tail -1)
[ "$G" = "GUARD OK" ] || { echo "the sandbox guard does not hold ($G) -- not running"; exit 2; }
: > "$ST/guard.log"
# the body's start-log literals (LITS="..." lines, comma = space, | between): echoed into the mock's start log
( set +u; eval "$(grep -hE '^[[:space:]]*LITS="?\$?LITS|^[[:space:]]*LITS=' "$TGT" "$(grep -oE 'B=\$HOME/[^ ]+\.sh' "$TGT" | head -1 | sed "s#B=\$HOME#$HOME#")" 2>/dev/null)" 2>/dev/null
  IFS='|' read -ra L <<< "${LITS:-}"; for l in "${L[@]}"; do [ -n "$l" ] && echo "${l//,/ }"; done ) > "$ST/start-extra.txt"
# recorded answers: the stock warm-up (ds41_tests correct --only-q) and the greedy set (d0) references, when present
[ $REPLAY = 1 ] && /usr/bin/python3 -I -B - "$HOME" "$ST/replay.jsonl" > "$ST/replay.txt" 2>&1 <<'PY'
import ast, importlib.util, json, os, sys
home, out = sys.argv[1:3]
n = 0
with open(out, "w") as f:
    try:
        src = open(home + "/gpu/ds41/ds41_tests.py").read()
        qs = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "qs" for t in node.targets):
                qs = dict(ast.literal_eval(node.value)); break
        ref = home + "/gpu/ds41/exl3-test/runs/x117/guard/x117-best524-warm-q.txt"
        for l in open(ref):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if d.get("test") in (qs or {}) and d.get("content") is not None:
                    f.write(json.dumps({"content": qs[d["test"]], "text": d["content"], "finish": d.get("finish"),
                                        "tokens": (d.get("usage") or {}).get("completion_tokens")}) + "\n"); n += 1
    except Exception as e:
        print("warm-up replay: %s" % e)
    try:
        spec = importlib.util.spec_from_file_location("gs", home + "/gpu/ds41/exl3-test/queue/tools/greedyset.py")
        gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
        for l in open(home + "/gpu/ds41/exl3-test/runs/xdsm/draft/dsd-d0-gs-c1.jsonl"):
            d = json.loads(l)
            if d.get("endpoint") == "completions":
                f.write(json.dumps({"prompt": gs.PROMPTS[d["i"]], "text": d.get("text") or "", "finish": d.get("finish"), "tokens": d.get("tokens")}) + "\n"); n += 1
    except Exception as e:
        print("greedy-set replay: %s" % e)
print("replay: %d recorded answers" % n)
PY
say "port $PORT, sandbox $SB, target ~/$REL (sandbox copy), knobs: ${ENVS[*]:-none}, faults: ${FAULTS[*]:-none}, swaps: ${SWAPS[*]:-none}; $(wc -l < "$ST/start-extra.txt") body literals echoed; $(tail -1 "$ST/replay.txt" 2>/dev/null)"
STGT="$H/$REL"
cleanup() {   # the mock and every process of the run, the port reservation
  ( export_env; "$SB/bin/docker" rm -f radiance-ds41-exl3 > /dev/null 2>&1 )
  [ -n "${PORT:-}" ] && [ "$(cat "$TMPR/.ports/$PORT/owner" 2>/dev/null)" = $$ ] && rm -rf "$TMPR/.ports/$PORT"
  [ -n "${BPID:-}" ] && kill -TERM -- -$BPID 2>/dev/null
  sleep 1; [ -n "${BPID:-}" ] && kill -KILL -- -$BPID 2>/dev/null
  return 0
}
trap cleanup EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- DRYRUN: the output dirs
if [ $DODRY = 1 ]; then
  ( export_env; for e in "${ENVS[@]}"; do export "$e"; done; cd "$H" && DRYRUN=1 $NICE timeout 900 bash "$STGT" ) > "$SB/dryrun.txt" 2>&1
  DRC=$?
  # every `+ mkdir -p A B ...` of the DRYRUN (up to the first ; && >), under runs/<x>/<y>: the top-most ones start empty
  mapfile -t DIRS < <(grep -oE '^\+ (run: )?mkdir -p [^;&>|]*' "$SB/dryrun.txt" | sed -E 's/^\+ (run: )?mkdir -p //' | tr ' ' '\n' \
    | grep "^$H/" | sed "s#^$H/##; s#/*\$##" | grep -E '^gpu/ds41/(exl3-test/)?runs/[^/]+/' | sort -u)
  for d in "${DIRS[@]}"; do
    skip=0; for e in "${DIRS[@]}"; do [ "$e" != "$d" ] && case $d in "$e"/*) skip=1 ;; esac; done
    [ $skip = 1 ] && continue
    rm -rf "${H:?}/$d"; mkdir -p "$H/$d"; FRESH+=("$d")
  done
  say "DRYRUN rc $DRC, $(wc -l < "$SB/dryrun.txt") lines; fresh output dirs: ${FRESH[*]:-none}"
  ( export_env; "$SB/bin/docker" rm -f x > /dev/null 2>&1 ); mv "$ST/docker-calls.log" "$ST/docker-calls-dryrun.log" 2>/dev/null
  mv "$ST/guard.log" "$ST/guard-dryrun.log" 2>/dev/null; : > "$ST/guard.log"
fi
# the start-log library (REAL files, read only; never the sandbox's own output): the 60 newest real DS starts (runs/xdsm/*/start-TAG.log); mockrad serves the one whose start-time env
# (the engine's own name dump) matches the container env best (an int9 KVR start gets the KVR declare lines, an nvfp4 start the latent line)
/usr/bin/python3 -I -B - "$HOME/gpu/ds41/exl3-test/runs/xdsm" "$ST/startlib.json" <<'PY'
import glob, json, os, sys
d, out = sys.argv[1:3]
L = []
for f in sorted(glob.glob(d + "/*/start-*.log"), key=lambda p: -os.path.getmtime(os.path.realpath(p))):
    if "crashed" not in f: L.append({"log": f})
json.dump(L[:60], open(out, "w"))
PY
# ---------------------------------------------------------------- the run
say "run (timeout $TIMEOUT s)"
( export_env; for e in "${ENVS[@]}"; do export "$e"; done; cd "$H" && exec setsid $NICE bash "$STGT" ) > "$SB/body.out" 2>&1 &
BPID=$!
RC=""; TO=0
for i in $(seq 1 $TIMEOUT); do
  if ! kill -0 $BPID 2>/dev/null; then wait $BPID; RC=$?; break; fi
  sleep 1
done
if [ -z "$RC" ]; then TO=1; say "TIMEOUT after $TIMEOUT s: stopping the body"; kill -TERM -- -$BPID 2>/dev/null; sleep 3; wait $BPID 2>/dev/null; RC=124; fi
cleanup; BPID=""
say "body rc $RC after $(( $(date +%s) - T0 )) s"
# ---------------------------------------------------------------- the report
/usr/bin/python3 -I -B "$MR/mkfarm.py" verify --manifest "$SB/manifest.json" > "$SB/verify.txt" 2>&1
/usr/bin/python3 -I -B "$MR/scan.py" --sb "$SB" --name "$NAME" --target "~/$REL" --rc "$RC" --timeout $TO --seconds $(( $(date +%s) - T0 )) \
  --knobs "${ENVS[*]:-}" --faults "${FAULTS[*]:-}" --swaps "${SWAPS[*]:-}" $(for f in "${FRESH[@]}"; do echo --fresh "$f"; done) > "$SB/report.txt" 2>&1
SRC=$?
mkdir -p "$MR/reports" && cp "$SB/report.txt" "$MR/reports/$NAME.txt"
cat "$SB/report.txt"
echo "report: $MR/reports/$NAME.txt | sandbox: $SB"
exit $SRC
