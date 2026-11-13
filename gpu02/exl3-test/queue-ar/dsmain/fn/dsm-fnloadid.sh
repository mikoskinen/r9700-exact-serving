#!/bin/bash
# dsm-fnloadid.sh (11.11.2026, queue-ar/dsmain/fn, ds-main-port) -- NOT a queue item; the body of 57-p4zyzzzzzzzzzz-dsm-fnloadid.sh.
# Log tag "RESULT xfnlid ", out runs/xdsm/fnloadid (XFNLID_OUT). IDENTITY UNDER LOAD ON FLASH-NEXT (Qwen3.8 Flash-Next).
# WHY: DS's xlid (57-p4zyzzzzz-ds-iej-z0ad-loadid) found 6 of 20 greedy answers differ when each prompt ran beside c4 streams
#   + one fresh ~32k prefill vs the same prompt alone (NST 0; the alone-again control 20 / 20 identical). The user rules that an
#   answer changing under load vs alone is a bug. This measures whether Flash-Next has it too: on the fn4 stream's two stacks and
#   on production's configuration. (For scale, not this method: fn4's greedyset c4 vs c4 floor already differs in 13-15 of 20 on
#   both lines, 146 / 156b; c1 vs c1 IDENTICAL.)
# THREE STARTS (XFNLID_STARTS "x3 s3 prod"), one a stack, the same procedure in each:
#   x3   the fn4 EXL3 line's stack = 173 x3arm's A side exactly: radiance-local:c16-x3arm (73d564388db8; fn4-x3 38aa3e7b = X3CODES +
#        the exl3_gd fix + the prefix-cache fix + the exl3_dz switch; exl3r4d 9ebbd24), qwen3.8-flash-next-exl3.rad, TP4 on all 4
#        cards, all_vram, MML 262144, MAXSEQS 16, MNBT 4096 + --prefill-span 2, --prefix-resume nearest, --prefill-interleave 0,
#        NST 3, wht6 min-kb 128; RADIANCE_FN4_* F2b / F2a / F1n / hcw / hc_readv 11 / hc_upmix 5 / HC_SS / X3CODES 1 / X3GD 1
#        (exl3_gd 1 at start), map 0 + dz 0 (173's env line, A = exl3_mapshare 0 + exl3_dz 0, checked by GET /admin/fn4);
#        greedy reference 156b fn4-x3gd2-gs-on-c1.jsonl (c16-x3g, X3CODES + exl3_gd 1); warm-up reference fn4-x3gd2-warm-q.txt;
#   s3   the fn4 iq4r stack 3 = 170 M-fix's: radiance-local:c16-rf (37f8739a4a60; fn4-c16 51a2ddc6 = stack 3 + the prefix-cache
#        fix), qwen3.8-next-flash-fp8-iq4r-moe.rad, the same serve line and stack-3 words without the EXL3 ones; greedy reference
#        146 fn4-acq-gs-off-c1.jsonl (c16-sw12 stack 3, aql_acq off); warm-up reference fn4-acq-warm-q.txt;
#   prod production's configuration: stilldeadcode/radiance:1.15.1 (cccbc0bd6bfe = compose/.env RADIANCE_IMAGE), ONE DP half as
#        compose/flashnext-prod-dp-a.yaml serves it (TP2 on 83:00 + 86:00, wht6 min-kb 128, 16 seqs, MML 524288 yarn:2,
#        expert_tiered + host pool 16 GiB, fp8 KV, prefix 60 GiB ttl 8 h, 512 keep-all checkpoints, MTP 3, MNBT 2048, reasoning
#        flags; container env = ROCR_VISIBLE_DEVICES only) in fn-lidprod.yaml (= fn4's fn-prodres.yaml, 168 / 170): test
#        container, loopback :8090, no --api-key-env, its own n-gram tmpfs directory (XFNLID_NGRAM /dev/shm/fnlid-ngram, made
#        here, removed at the end and on exit). The serve line is checked token for token against the production file (read as
#        text; only --host / --port / --api-key-env may differ) and the card list against it. No earlier greedy set exists for
#        this configuration: its alone run is the reference.
# In each start: restart + health, the stack checks (image id, the RADIANCE_FN4_* words and card list in the container's env,
#   GET /admin/fn4's switches for x3 / s3), the warm-up answers (ds41_tests.py correct --only-q --effort none, fn4's warm-up; vs
#   the fn4 run's own file where one exists, information), then
#   a1  the 20-prompt greedy set ALONE: greedyset.py c1, temperature 0, 192 tokens, after a prefix-cache reset;
#   ld  the same 20 prompts EACH under load (fnloadid.py run = loadid.py verbatim; prefix-cache reset first): mb.py's c4
#       closed-loop streams (ctx 1000, gen 512, effort medium, server-default sampling) the whole phase (10 s warm), and per probe
#       ONE fresh ~32k prefill (mb.py's pf prompt, time_ns nonce first: never cached) started 0.5 s after the probe; the next
#       probe when both are done. Per probe: the prefill's TTFT / cached tokens, its share of the probe's life, engine steps and
#       MIXED steps (radiance:engine_steps_mixed_total) during the probe;
#   a2  CONTROL (XFNLID_CTRL=1): the greedy set alone again after a reset -- a2 vs a1 tells a load effect from drift;
#   compares (fnloadid.py cmp: byte for byte + each differing prompt's first differing char and token via the server's
#       /tokenize): ld vs a1 (THE measurement), a2 vs a1, a1 vs the fn4 reference, ld vs the fn4 reference.
#   One summary line per start ("RESULT xfnlid fl-<stack> summary"), then "RESULT xfnlid summary:" over the starts.
# ~32 min (a start ~10: restart ~2 (106-126 s in 156b / 146 / 170), warm-up ~1, a1 ~1, ld ~4 (20 x a ~32k prefill beside
#   4 streams), a2 ~1, compares + logs ~1). NO cc_sim. No power / host / fan settings; production compose files read only
#   (--project-directory for .env / common.yaml as every fn4 item; flashnext-prod-dp-a.yaml and .env's RADIANCE_IMAGE line read
#   as text). Measurement only; nothing goes into production; no fn4 branch / image / item touched.
# DRYRUN=1 bash dsm-fnloadid.sh : prints every command; checks every start's image (id + a no-device, no-network grep of its
#   literals), every helper (files present and parsing, fnloadid.py selftest importing greedyset.py / mb.py read only, every
#   function this body calls resolves with `type -t`, tools on PATH), both yamls (docker compose config with each start's env;
#   only image / command / card list printed, never the environment), the prod serve line vs production's file, the models,
#   the references, the n-gram tmpfs; no server, no GPU, nothing written.
# Knobs: XFNLID_STARTS ("x3 s3 prod"), XFNLID_OUT, XFNLID_ITEM, XFNLID_OFFSET (0.5 s), XFNLID_PFTOK (32768), XFNLID_CONC (4),
#   XFNLID_WARM (10 s), XFNLID_MT (192, the greedy set's), XFNLID_CTRL (1 | 0), XFNLID_NGRAM (/dev/shm/fnlid-*).
set -u
DRY=${DRYRUN:-0}
HD="$(cd "$(dirname "$0")" && pwd)"
C=$HOME/radiance-1.15.1/compose; PRODY=$C/flashnext-prod-dp-a.yaml     # production: READ ONLY
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
O=${XFNLID_OUT:-$R/xdsm/fnloadid}
ITEM=${XFNLID_ITEM:-57-p4zyzzzzzzzzzz-dsm-fnloadid}
STARTS=${XFNLID_STARTS:-x3 s3 prod}; OFF=${XFNLID_OFFSET:-0.5}; PFT=${XFNLID_PFTOK:-32768}; CONC=${XFNLID_CONC:-4}
WARM=${XFNLID_WARM:-10}; MT=${XFNLID_MT:-192}; CTRL=${XFNLID_CTRL:-1}; NGD=${XFNLID_NGRAM:-/dev/shm/fnlid-ngram}
for s in $STARTS; do case $s in x3|s3|prod) ;; *) echo "XFNLID_STARTS takes x3 / s3 / prod (got '$s')"; exit 1 ;; esac; done
case $CTRL in 0|1) ;; *) echo "XFNLID_CTRL must be 0 or 1 (got '$CTRL')"; exit 1 ;; esac
for v in OFF PFT CONC WARM MT; do case ${!v} in ''|*[!0-9.]*) echo "$v must be a number (got '${!v}')"; exit 1 ;; esac; done
[[ $NGD =~ ^/dev/shm/fnlid-[A-Za-z0-9_-]+$ ]] || { echo "XFNLID_NGRAM must be /dev/shm/fnlid-<name> (got '$NGD')"; exit 1; }
GS=$T/queue/tools/greedyset.py; LD=$HD/fnloadid.py; MBP=$D/mb.py; DT=$D/ds41_tests.py
YT=$HD/fn-lidtp4.yaml; YP=$HD/fn-lidprod.yaml
CN=radiance-fnlid; URL=http://127.0.0.1:8090; export DS41_URL=$URL
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
X3DIR=$HOME/models/flashnext-exl3; X3MOD=qwen3.8-flash-next-exl3.rad
IQDIR=$HOME/models/flashnext-iq4r; IQMOD=qwen3.8-next-flash-fp8-iq4r-moe.rad    # = compose/.env RADIANCE_MODELS + production's --model
IMG_x3=radiance-local:c16-x3arm;       IID_x3=73d564388db8
IMG_s3=radiance-local:c16-rf;          IID_s3=37f8739a4a60
IMG_prod=stilldeadcode/radiance:1.15.1; IID_prod=cccbc0bd6bfe
REF_x3=$R/fn4/fn4-x3gd2-gs-on-c1.jsonl; WREF_x3=$R/fn4/guard/fn4-x3gd2-warm-q.txt
REF_s3=$R/fn4/fn4-acq-gs-off-c1.jsonl;  WREF_s3=$R/fn4/guard/fn4-acq-warm-q.txt
REF_prod="";                            WREF_prod=""
# the image literals each start needs (any of bin/radiance and the share/radiance .so files): the driver's three for all; the fn4
# switch endpoint and the stack's start words for x3 / s3; the prefix-cache fix's startup literal on the two fn4 images
LIT_all="radiance:engine_steps_mixed_total|reset_prefix_cache|/tokenize|/v1/completions"
LIT_x3="/admin/fn4|RADIANCE_FN4_HC_READV|RADIANCE_FN4_HC_SS|RADIANCE_FN4_X3CODES|RADIANCE_FN4_X3GD|exl3_mapshare|exl3_dz|later publisher"
LIT_s3="/admin/fn4|RADIANCE_FN4_HC_READV|RADIANCE_FN4_HC_SS|later publisher"
LIT_prod="prefix resume: exact"
# the RADIANCE_FN4_* words the tp4 yaml passes: NAME (after RADIANCE_FN4_) : the variable it reads (empty = off)
FN4MAP="NO_ROUTE:FN4NR GATHER_INV:FN4GI HC_FUSEDOWN:FN4FD HC_NORMWAVE:FN4NW SWITCHABLE:FN4SW HC_WRITEVEC:FN4WV HC_READV:FN4RV
  AR_LANE_THREADS:FN4AL HC_UPMIX:FN4UM HC_SS:FN4SS CHAIN:FN4CH X3MAP:FN4XM SPHC:FN4SPHC X3CODES:FN4X3C QSADUMP:FN4QSA P2X:FN4P2X
  MOE_PAIR:FN4MP X3GD:FN4X3GD X3DZ:FN4X3DZ"
Y=""; DESC=""
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
mkd() { if [ "$DRY" = 1 ]; then echo "+ mkdir -p $*"; else mkdir -p "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { [ -n "$Y" ] && run "compose down > /dev/null 2>&1"; run "docker rm -f $CN > /dev/null 2>&1"; }
stopfile() { [ -e $R/sweep.stop ]; }
keeplog() { run "docker logs $CN > $O/engine-$1.log 2>&1"; }
cleanup() { down; [ "$DRY" = 1 ] || rm -rf "$NGD"; }
stackenv() {   # stackenv x3|s3|prod: Y, the compose variables and DESC of that start (every other knob as the fn4 runs had it)
  unset RADIANCE_AQL_ACQ                       # the tp4 yaml passes it through from this shell when set: never here
  case $1 in
    x3|s3)
      Y=$YT
      export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
      export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats PROFEVERY=""
      export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4QSA="" FN4P2X="" FN4MP="" FN4X3DZ=""
      if [ $1 = x3 ]; then
        export FNIMAGE=$IMG_x3 FNMODELDIR=$X3DIR FNMODEL=$X3MOD FN4X3C=1 FN4X3GD=1
        DESC="fn4 EXL3 line = 173 x3arm A: X3CODES + exl3_gd 1, map 0 + dz 0, stack-3 words, TP4 all_vram, MNBT 4096 + span 2, nearest, interleave 0, NST 3"
      else
        export FNIMAGE=$IMG_s3 FNMODELDIR=$IQDIR FNMODEL=$IQMOD FN4X3C="" FN4X3GD=""
        DESC="fn4 iq4r stack 3 = 170 M-fix: F2b F2a F1n hcw hc_readv 11 hc_upmix 5 HC_SS, TP4 all_vram, MNBT 4096 + span 2, nearest, interleave 0, NST 3"
      fi ;;
    prod)
      Y=$YP; export FNIMAGE=$IMG_prod NGRAMDIR=$NGD
      DESC="production 1.15.1 config, one DP half (flashnext-prod-dp-a.yaml's serve line: TP2 83:00 + 86:00, expert_tiered, MML 524288 yarn:2, MNBT 2048, NST 3)" ;;
  esac
}
fn4words() { local m n v w=""; for m in $FN4MAP; do n=${m%%:*}; v=${m#*:}; [ -n "${!v}" ] && w="$w RADIANCE_FN4_$n=${!v}"; done; echo $w; }
cfgline() {   # cfgline: docker compose config of the current Y + env -> "image | container | cards | command" (the environment is
  # read inside python and never printed: only the card list)
  docker compose --project-directory $C -f $Y config --format json 2>/dev/null | python3 -I -c 'import json,sys
s = json.load(sys.stdin)["services"]["radiance"]; e = s.get("environment") or {}
c = s.get("command"); c = " ".join(c) if isinstance(c, list) else str(c)
print("%s | %s | cards %s | %s" % (s.get("image"), s.get("container_name"), e.get("ROCR_VISIBLE_DEVICES"), " ".join(c.split())))' 2>&1 | cut -c1-1500
}
prodline() {   # prodline: fn-lidprod.yaml's serve line vs production's flashnext-prod-dp-a.yaml (text), token for token without
  # --host / --port / --api-key-env, and the card list -> "MATCH ..." or "DIFFERS ..."
  python3 -I - "$PRODY" "$YP" <<'PY'
import re, sys
def rd(f):
    t = open(f).read().split("\n"); cmd, on, ind = [], False, 0
    for l in t:
        m = re.match(r"^(\s*)command:\s*>\s*$", l)
        if m: on, ind = True, len(m.group(1)); continue
        if on:
            if l.strip() and (len(l) - len(l.lstrip())) <= ind: on = False; continue
            cmd += l.split()
    cards = re.findall(r'ROCR_VISIBLE_DEVICES:\s*"([^"]*)"', "\n".join(t))
    out, skip = [], 0
    for i, w in enumerate(cmd):
        if skip: skip = 0; continue
        if w in ("--host", "--port", "--api-key-env"): skip = 1; continue
        out.append(w)
    return out, cards
p, q = rd(sys.argv[1]), rd(sys.argv[2])
if not p[0] or not q[0]: print("DIFFERS: no command block read (%d / %d tokens)" % (len(p[0]), len(q[0]))); sys.exit(1)
d = [x for x in set(p[0]) ^ set(q[0])]
if p == q: print("MATCH: %d tokens (without --host / --port / --api-key-env), cards %s" % (len(p[0]), ",".join(p[1]))); sys.exit(0)
print("DIFFERS: tokens only in one of them %s; production order %s vs test %s; cards %s vs %s" % (sorted(d), " ".join(p[0])[:300], " ".join(q[0])[:300], p[1], q[1])); sys.exit(1)
PY
}
restart() {   # restart TAG -> 0 healthy (<= 1500 s, 170's limit)
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE docker compose --project-directory $C -f $Y up -d; health <= 1500 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 300); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT xfnlid $tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT xfnlid $tag: healthy in $(( $(date +%s) - t0 )) s (container image $(docker inspect -f '{{.Image}}' $CN 2>/dev/null | cut -c8-19)); args $(docker inspect -f '{{json .Args}}' $CN 2>/dev/null | cut -c1-900) | $(grep -m1 -E '^I kv: ' $S | cut -c3-200)"
}
envchk() {   # envchk TAG x3|s3|prod: the container's RADIANCE_FN4_* words and card list are the start's (the key line is dropped
  # before anything is read or logged); prod: no RADIANCE_FN4_* word at all, production's card list
  local s=$2 CE bad="" m n v
  if [ "$DRY" = 1 ]; then
    if [ $s = prod ]; then echo "+ docker inspect $CN env: ROCR_VISIBLE_DEVICES=$C83,$C86, no RADIANCE_FN4_* word"
    else echo "+ docker inspect $CN env: ROCR_VISIBLE_DEVICES=$CARDS; RADIANCE_FN4_* = [$(fn4words)], every other RADIANCE_FN4_* word empty"; fi
    return 0
  fi
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep -v '^RADIANCE_API_KEY=' | grep -E '^(RADIANCE_|ROCR_VISIBLE_DEVICES=)')
  if [ $s = prod ]; then
    echo "$CE" | grep -qxF "ROCR_VISIBLE_DEVICES=$C83,$C86" || bad="$bad card-list"
    echo "$CE" | grep -q '^RADIANCE_FN4_' && bad="$bad RADIANCE_FN4_-words"
  else
    echo "$CE" | grep -qxF "ROCR_VISIBLE_DEVICES=$CARDS" || bad="$bad card-list"
    for m in $FN4MAP; do n=${m%%:*}; v=${m#*:}   # a set word must be there with its value; an empty one must not carry a value
      if [ -n "${!v}" ]; then echo "$CE" | grep -qxF "RADIANCE_FN4_$n=${!v}" || bad="$bad RADIANCE_FN4_$n"
      else echo "$CE" | grep -qE "^RADIANCE_FN4_$n=." && bad="$bad RADIANCE_FN4_$n"; fi
    done
  fi
  log "RESULT xfnlid $1 env in the container (non-empty RADIANCE_* / card words): $(echo "$CE" | grep -vE '=$' | tr '\n' ' ' | cut -c1-900)$([ -n "$bad" ] && echo " -- VIRHE not as asked:$bad")"
  [ -z "$bad" ]
}
fn4chk() {   # fn4chk TAG x3|s3: GET /admin/fn4 -> the stack's switches at start (both: no_route 1, gather_inv 1, hc_fusedown 0,
  # hc_normwave 1, hc_writevec 1, hc_readv 11, hc_upmix 5, moe_pair 0 / absent; x3: exl3_gd 1, exl3_mapshare 0, exl3_dz 0 / absent);
  # nothing pending. No switch is posted: the start values are the stack.
  [ "$DRY" = 1 ] && { echo "+ GET $URL/admin/fn4: no_route 1, gather_inv 1, hc_fusedown 0, hc_normwave 1, hc_writevec 1, hc_readv 11, hc_upmix 5, moe_pair 0$([ $2 = x3 ] && echo ", exl3_gd 1, exl3_mapshare 0, exl3_dz 0"), nothing pending"; return 0; }
  local s r; s=$(curl -s -m 10 $URL/admin/fn4 2>/dev/null)
  r=$(python3 -I -c 'import json,sys
try: d = json.loads(sys.argv[1]); w = d["switches"]
except Exception as e: print("unreadable (%s)" % type(e).__name__); sys.exit(1)
want = dict(no_route=1, gather_inv=1, hc_fusedown=0, hc_normwave=1, hc_writevec=1, hc_readv=11, hc_upmix=5)
if sys.argv[2] == "x3": want.update(exl3_gd=1, exl3_mapshare=0)
bad = ["%s=%s (want %s)" % (k, w.get(k), v) for k, v in want.items() if w.get(k) != v]
for k in (("moe_pair", "exl3_dz") if sys.argv[2] == "x3" else ("moe_pair",)):
    if w.get(k, 0) not in (0, -1): bad.append("%s=%s" % (k, w.get(k)))
if d.get("pending"): bad.append("pending %s" % d["pending"])
print(" ".join(bad)); sys.exit(1 if bad else 0)' "$s" $2 2>&1)
  local rc=$?
  log "RESULT xfnlid $1 fn4 switches at start: $(echo "$s" | cut -c1-700)$([ $rc = 0 ] || echo " -- VIRHE not the stack: $r")"
  return $rc
}
prodlines() {   # prodlines TAG: production-config start notes (prefix resume mode, the n-gram table) -- information
  [ "$DRY" = 1 ] && { echo "+ RESULT xfnlid $1 start notes: prefix resume / n-gram lines of the start log"; return 0; }
  log "RESULT xfnlid $1 start notes: $(grep -iE 'prefix resume|n-gram table|ngram' $O/start-$1.log | sed 's/^[A-Z] *//' | cut -c1-200 | head -4 | tr '\n' '|')"
}
warmup() {   # warmup TAG REF: fn4's warm-up (ds41_tests.py correct --only-q --effort none), answers vs the fn4 run's file (information)
  run "(cd $D && timeout 600 python3 $DT correct --only-q --effort none --outdir $O/guard > $O/guard/$1-warm-q.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  [ -n "$2" ] && [ -s "$2" ] || { log "RESULT xfnlid $1 warm-up: done (no earlier warm-up file for this configuration)"; return 0; }
  log "RESULT xfnlid $1 warm-up answers vs $(basename $2) (information): $(python3 -I -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if t in a and ((a[t].get("content") or "") != (b[t].get("content") or "") or a[t].get("finish") != b[t].get("finish"))]
n = len([t for t in b if t in a])
print(("IDENTICAL: %d answers" % n) if n and not bad else ("DIFFERENT: " + ", ".join(bad) if n else "no common answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)"
}
alive() {   # alive TAG WHAT: the server still answers, else its log is kept and the start ends (1)
  [ "$DRY" = 1 ] && return 0
  [ "$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && return 0
  docker logs $CN > $O/$1-dead-engine.log 2>&1
  log "RESULT xfnlid $1: VIRHE the server is not healthy after $2 (last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$1-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> start stopped"
  return 1
}
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback) so the next prompts prefill cold, not resumed from a1's blocks
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xfnlid $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
alone() {   # alone TAG NAME: a prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE -> $O/TAG-NAME.jsonl
  local tag=$1 t0=$(date +%s)
  pcreset $tag-$2
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$2.jsonl --conc 1 --max-tokens $MT > $O/$tag-$2.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xfnlid $tag $2 (greedy set c1 ALONE, temperature 0, $MT tok): $(tail -1 $O/$tag-$2.txt | cut -c1-300), $(( $(date +%s) - t0 )) s"
}
loaded() {   # loaded TAG: a prefix-cache reset, fnloadid.py run (mb c4 + per probe one fresh prefill) -> $O/TAG-ld.jsonl
  local tag=$1 t0=$(date +%s) j
  pcreset $tag-ld
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-ld.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-ld.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-ld.txt | cut -c8-)
  [ -n "$j" ] || { log "RESULT xfnlid $tag ld: VIRHE no LOADID line ($(tail -3 $O/$tag-ld.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  log "RESULT xfnlid $tag ld (the 20 prompts each under load: mb c$CONC + one fresh ~${PFT}-token prefill at +${OFF} s, temperature 0, $MT tok): $(python3 -I -c 'import json,sys
d = json.loads(sys.argv[1]); w = d.get("window") or {}
print("%s of %s answered, probe %s s mean | prefills %s tok mean, %s of %s cold, TTFT %s s mean (%s..%s) | in flight %s..%s %% (median) of a probe | mixed steps a probe min %s, median %s, total %s; probes without one %s | c4 %s requests done, %s errors | window %s s: %s ms/decode step, %s steps, %s prefill tok, %s decode tok%s" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_prompt_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["pf_ttft_min"], d["pf_ttft_max"],
    round(100 * (d["ov_frac_min"] or 0)), round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"], d["mixed_median"], d["mixed_total"],
    d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"], d["window_s"], w.get("ms_decode_step"), w.get("steps"),
    w.get("prefill_tok"), w.get("dec_tok"), (" | VIRHE errors: %s" % d["errors"]) if d.get("errors") else ""))' "$j" 2>&1 | cut -c1-1200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: fnloadid.py cmp (byte for byte + the first differing char / token via /tokenize) -> $O/TAG-cmp-NAME.json
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label '$1 $2' $3 $4"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then
    log "RESULT xfnlid $1 $2: not compared (missing:$([ -s "$3" ] || echo " $3")$([ -s "$4" ] || echo " $4"))"; echo "$1 $2 not compared" >> $O/identity.txt; return 0
  fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  echo "$1 $2 ${r:-compare failed}" >> $O/identity.txt
  log "RESULT xfnlid $1 $2: ${r:-VIRHE compare failed} ($(basename $3) vs $(basename $4))"
}
startsum() {   # startsum TAG DESC: the start's summary line
  [ "$DRY" = 1 ] && { echo "+ RESULT xfnlid $1 summary: loaded vs alone (n differ, first diff char / token each), control a2 vs a1, a1 / ld vs the fn4 reference, the load"; return 0; }
  python3 -I - "$O/$1" > $O/$1-summary.txt 2>&1 <<'PY'
import json, sys
p = sys.argv[1]
def js(f):
    try: return json.load(open(f))
    except Exception: return None
def one(n):
    d = js("%s-cmp-%s.json" % (p, n))
    if d is None: return "not compared"
    if d["identical"]: return "IDENTICAL (%d of %d)" % (d["n"], d["n"])
    if not d["n_a"] or not d["n_b"]: return "not compared (answers %d / %d)" % (d["n_a"], d["n_b"])
    return "%d of %d differ [%s]%s" % (d["n_diff"], d["n"], ", ".join("p%d char %s tok %s%s" % (e["i"], e["char"], "-" if e["tok"] is None else e["tok"],
           "" if e.get("mixed_b") is None else " (mixed %s)" % e["mixed_b"]) for e in d["diff"]), (" + missing %s" % d["missing"]) if d["missing"] else "")
ld = None
try: ld = json.loads(next(l for l in open(p + "-ld.txt") if l.startswith("LOADID "))[7:])
except Exception: pass
load = ("load: c4 %s requests (%s errors), %s of %s prefills cold ~%s tok TTFT %s s mean, in flight median %s %% of a probe, mixed steps a probe min %s / median %s"
        % (ld["c4_requests_done"], ld["c4_errors"], ld["pf_cold"], ld["of"], ld["pf_prompt_mean"], ld["pf_ttft_mean"], round(100 * (ld["ov_frac_median"] or 0)),
           ld["mixed_min"], ld["mixed_median"])) if ld else "load: no LOADID line"
print("LOADED vs ALONE: %s | control alone-after-load vs alone: %s | alone vs fn4 ref: %s | loaded vs fn4 ref: %s | %s"
      % (one("ld-vs-a1"), one("a2-vs-a1"), one("a1-vs-ref"), one("ld-vs-ref"), load))
PY
  log "RESULT xfnlid $1 summary ($2): $(head -c 2500 $O/$1-summary.txt)"
}
overall() {   # overall: one line over the starts that ran
  [ "$DRY" = 1 ] && { echo "+ RESULT xfnlid summary: per start loaded vs alone (n differ of 20, first diff positions), the control, vs the fn4 reference, the verdict"; return 0; }
  log "RESULT xfnlid summary: $(python3 -I - $O <<'PY' 2>&1 | head -c 2500
import json, os, sys
o = sys.argv[1]
def js(f):
    try: return json.load(open(f))
    except Exception: return None
parts, bad, ran = [], [], 0
for tag, lab in (("fl-x3", "EXL3 line (c16-x3arm, x3arm A)"), ("fl-s3", "iq4r stack 3 (c16-rf)"), ("fl-prod", "production 1.15.1 config (one DP half)")):
    c = {n: js(os.path.join(o, "%s-cmp-%s.json" % (tag, n))) for n in ("ld-vs-a1", "a2-vs-a1", "a1-vs-ref", "ld-vs-ref")}
    if c["ld-vs-a1"] is None:
        if os.path.exists(os.path.join(o, "start-%s.log" % tag)): parts.append("%s: not measured (start stopped, see its lines)" % lab)
        continue
    ran += 1
    f = lambda d: "-" if d is None else ("0/%d" % d["n"] if d["identical"] else "%d/%d%s" % (d["n_diff"], d["n"], " (+%d missing)" % len(d["missing"]) if d["missing"] else ""))
    first = ", ".join("p%d char %s tok %s" % (e["i"], e["char"], "-" if e["tok"] is None else e["tok"]) for e in c["ld-vs-a1"]["diff"][:8])
    parts.append("%s: loaded vs alone %s differ%s; control a2 vs a1 %s; alone vs fn4 ref %s; loaded vs fn4 ref %s"
                 % (lab, f(c["ld-vs-a1"]), " [" + first + "]" if first else "", f(c["a2-vs-a1"]), f(c["a1-vs-ref"]), f(c["ld-vs-ref"])))
    if not c["ld-vs-a1"]["identical"]:
        ctl = c["a2-vs-a1"]
        bad.append("%s (%s%s)" % (lab.split(" (")[0], f(c["ld-vs-a1"]), "" if ctl is None else ("; control identical: the load changes the answers" if ctl["identical"] else "; control differs too: drift, not only load")))
v = ("no start measured" if not ran else
     "IDENTITY HOLDS UNDER LOAD at every measured start (20/20 greedy answers byte-identical to the 1-user run)" if not bad else
     "ANSWERS CHANGE UNDER LOAD at " + ", ".join(bad))
print(" || ".join(parts + ["verdict: " + v]))
PY
)"
}
fstart() {   # fstart x3|s3|prod: one start: image, restart, checks, warm-up, a1, ld, a2, compares, summary
  local s=$1 tag=fl-$1 img iid ref wref pl
  local v1=IID_$s v2=REF_$s v3=WREF_$s
  stackenv $s; img=$FNIMAGE; iid=${!v1}; ref=${!v2}; wref=${!v3}
  log "RESULT xfnlid $tag: start ($DESC) IMAGE=$img ($iid) yaml $(basename $Y)$([ $s = prod ] || echo " MODEL=$FNMODELDIR/$FNMODEL FN4 [$(fn4words)] EXTRA [$EXTRA]")$([ $s = prod ] && echo " n-gram dir $NGD") | greedy reference ${ref:-none (the alone run is the reference)}"
  if [ "$DRY" != 1 ]; then
    [ "$(docker image inspect -f '{{.Id}}' $img 2>/dev/null | cut -c8-19)" = "$iid" ] || { log "RESULT xfnlid $tag: VIRHE image $img is not $iid -> start skipped"; return 1; }
    if [ $s = prod ]; then
      pl=$(prodline 2>&1); log "RESULT xfnlid $tag serve line vs production's $(basename $PRODY) (read as text): $pl"
      [ "$(grep -m1 '^RADIANCE_IMAGE=' $C/.env 2>/dev/null | cut -d= -f2-)" = "$IMG_prod" ] || log "RESULT xfnlid $tag: note -- compose/.env RADIANCE_IMAGE no longer names $IMG_prod (this start runs the pinned image)"
      [ "$(stat -f -c %T /dev/shm 2>/dev/null)" = tmpfs ] || { log "RESULT xfnlid $tag: VIRHE /dev/shm is not a tmpfs -> start skipped"; return 1; }
      rm -rf "$NGD"
      [ "$(df --output=avail -B1G /dev/shm | tail -1 | tr -d ' ')" -ge 60 ] || { log "RESULT xfnlid $tag: VIRHE under 60 GB free on /dev/shm -> start skipped"; return 1; }
      mkdir -p "$NGD"
    fi
  fi
  restart $tag || { keeplog $tag-fail; down; return 1; }
  envchk $tag $s || { keeplog $tag-env; down; return 1; }
  if [ $s = prod ]; then prodlines $tag; else fn4chk $tag $s || { keeplog $tag-switch; down; return 1; }; fi
  warmup $tag "$wref"
  alive $tag "the warm-up" || { down; return 1; }
  stopfile || alone $tag a1
  alive $tag "a1" || { down; return 1; }
  stopfile || loaded $tag
  alive $tag "the loaded phase" || { down; return 1; }
  [ $CTRL = 1 ] && ! stopfile && alone $tag a2
  alive $tag "a2" || { down; return 1; }
  cmpx $tag ld-vs-a1 $O/$tag-a1.jsonl $O/$tag-ld.jsonl
  [ $CTRL = 1 ] && cmpx $tag a2-vs-a1 $O/$tag-a1.jsonl $O/$tag-a2.jsonl
  if [ -n "$ref" ]; then cmpx $tag a1-vs-ref $ref $O/$tag-a1.jsonl; cmpx $tag ld-vs-ref $ref $O/$tag-ld.jsonl; fi
  startsum $tag "$DESC"
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  [ $s = prod ] && run "rm -rf $NGD"
  log "RESULT xfnlid $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xfnlid $ITEM: start, starts [$STARTS] | per start: greedy set c1 alone, then each prompt under mb c$CONC + one fresh ~${PFT}-token prefill at +${OFF} s (warm ${WARM} s), $MT tok, temperature 0$([ $CTRL = 1 ] && echo ", control alone again"); compares ld vs a1 (+ the fn4 greedy references) | images x3 $IMG_x3, s3 $IMG_s3, prod $IMG_prod"
# EVERY IMAGE, read only (DRYRUN and real): its id and a no-device, no-network grep of the literals each start needs
IMGBAD=""
for s in $STARTS; do
  v1=IMG_$s; v2=IID_$s; v3=LIT_$s; img=${!v1}; iid=${!v2}; lits="$LIT_all|${!v3}"
  got=$(docker image inspect -f '{{.Id}}' $img 2>/dev/null | cut -c8-19)
  [ "$got" = "$iid" ] || { IMGBAD="$IMGBAD $s:$img(id ${got:-missing}, want $iid)"; continue; }
  miss=$(echo "$lits" | tr '|' '\n' | timeout 180 docker run --rm -i --network none --entrypoint sh $img -c 'F="/opt/radiance/bin/radiance $(find /opt/radiance/share/radiance -name "*.so" 2>/dev/null | tr "\n" " ")"; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); o=0; for x in $F; do grep -qzF -- "$l" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $l"; done; echo "CHECKED $n"' 2>&1)
  case "$miss" in "CHECKED "*) [ "$DRY" = 1 ] && echo "+ image $s: $img ($iid): all $(echo "$miss" | sed 's/CHECKED //') literals present [$lits] -- OK" ;;
    *) IMGBAD="$IMGBAD $s:$img(missing: $(echo "$miss" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-300))" ;; esac
done
[ -z "$IMGBAD" ] || { log "RESULT xfnlid: VIRHE images:$IMGBAD"; exit 1; }
# EVERY HELPER RESOLVES (DRYRUN and real): files present, the python ones parse (ast, nothing written), fnloadid.py's selftest
# loads greedyset.py / mb.py read only, every shell function this body calls is defined (type -t), the tools on PATH, each
# start's yaml accepted by docker compose config with that start's env (only image / cards / command printed), the prod serve
# line = production's, the models and references present. Any miss: VIRHE (a dry run fails too).
HMISS=""
for f in $YT $YP $GS $LD $MBP $DT $PRODY $C/common.yaml $C/.env; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $MBP $DT 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1)
case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
FNS="log run mkd compose down stopfile keeplog cleanup stackenv fn4words cfgline prodline restart envchk fn4chk prodlines warmup alive
     pcreset alone loaded cmpx startsum overall fstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee stat df cut tr; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for s in $STARTS; do
  stackenv $s
  cl=$(cfgline); cards=$([ $s = prod ] && echo $C83,$C86 || echo $CARDS)
  case "$cl" in "$FNIMAGE | $CN | cards $cards | --model "*" --host 127.0.0.1 --port 8090 "*) [ "$DRY" = 1 ] && echo "+ compose $s ($(basename $Y)): $cl" ;;
    *) HMISS="$HMISS yaml-$s:$(basename $Y)(${cl:0:300})" ;; esac
  case $s in
    x3) for f in $X3DIR/$X3MOD; do [ -s $f ] || HMISS="$HMISS model:$f"; done ;;
    s3|prod) for f in $IQDIR/$IQMOD; do [ -s $f ] || HMISS="$HMISS model:$f"; done ;;
  esac
done
case " $STARTS " in *" prod "*)
  PL=$(prodline 2>&1) || HMISS="$HMISS prodline:(${PL:0:400})"
  [ "$DRY" = 1 ] && echo "+ prod serve line vs production's $(basename $PRODY): $PL"
  [ "$(grep -m1 '^RADIANCE_IMAGE=' $C/.env 2>/dev/null | cut -d= -f2-)" = "$IMG_prod" ] || HMISS="$HMISS prodimage:(compose/.env RADIANCE_IMAGE is not $IMG_prod)"
  [ "$(grep -m1 '^RADIANCE_MODELS=' $C/.env 2>/dev/null | cut -d= -f2-)" = "$IQDIR" ] || HMISS="$HMISS prodmodels:(compose/.env RADIANCE_MODELS is not $IQDIR)"
  [ "$(stat -f -c %T /dev/shm 2>/dev/null)" = tmpfs ] || HMISS="$HMISS ngram:(/dev/shm is not a tmpfs)" ;;
esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { log "RESULT xfnlid: VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, $ST, all $(echo $FNS | wc -w) functions resolve (type -t), tools on PATH, every start's yaml accepted by docker compose config, models present -- OK"
for s in $STARTS; do v1=REF_$s; v2=WREF_$s
  for f in ${!v1} ${!v2}; do if [ -s $f ]; then [ "$DRY" = 1 ] && echo "+ reference $s present: $f"; else log "RESULT xfnlid: note -- reference $f missing (not compared)"; fi; done; done
Y=""
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xfnlid: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  trap 'cleanup' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(a radiance container running now is information here; the real run refuses to start beside one)"
  echo "+ trap cleanup EXIT (down + rm -rf $NGD)"
fi
mkd $O $O/guard $O/stats
[ "$DRY" = 1 ] || : > $O/identity.txt
for s in $STARTS; do stopfile && break; fstart $s; done
overall
down
log "RESULT xfnlid $ITEM: done -- $O"
