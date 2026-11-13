#!/bin/bash
# dsm-vramstack.sh (10.11.2026, queue-ar/dsmain, ds-main-port VRAM agent; v2 for the integrated image) -- NOT a queue item; the
# body of 57-p4zyzzzzz-ds-hxz-vramstack.sh. Log tag xvl, out runs/xdsm/vramstack. Helpers from dsm-stack.sh / dsm-vramsweep.sh
# (copied); drivers vlblocks.py (own block loop on ggblocksdz2's phase helpers + vsblocks' /stats counters, read only: D7
# re-arm, settle block, the lend readiness wait and sampler) and vssum.py (the summary); yaml ds41-vramstack-tp4.yaml (the
# stack yaml + env_file ${VSL_ENVFILE}, so env names the stack yaml does not list reach the container).
#
# WHAT: every exact VRAM lever that exists, STACKED, on the integrated image radiance-local:ds41-int-c05de277-5c13e555-l
#   (VSL_IMAGE; ds41-int c05de27 = ds41-stack fe030c3 (R1 fix) + ds41-vram 92746a8 + ds41-dlend 3f0ac9c, every new lever off
#   unless set). Start-time levers (the table): arena decode level (8 rows at NST 0, 48 at NST 5), router / compressor bf16,
#   embedding to host, shadow slots 16, expert prefetch off; base (in both sides) turnover 0. In-process lever DL at NST 5:
#   arms A {draft_lend 0, back-off 0}, B {draft_lend 1, back-off 3}, D = B + draft_lend_borrow 1 (C of xdl dropped).
# DESIGN (~52 min measured from xdl's pace, the plan line's estimate ~56; the 2-start case): hx-stack's NST 0 and NST 5 starts, on the same image with turnover 0, are the OFF
#   sides (VSL_OFF auto: reused when hx-stack's <tag>-meta.txt has the same image id, VRAM env, NST 5 arms stack / hyb VSL_HYB /
#   match 8, mb 75 s, <= 24 h, >= 3 good blocks; else an own OFF start (auto) or no OFF side (hx)). Here two ON starts:
#   NST 0 ON: start levers (arena 8, bf16, emb, shadow, xpoff) + turnover 0 + RADIANCE_DEBUG_VRAM=1 (logs only);
#     blocks V x VSL_BLOCKS0 (4, as hx-stack's N); then the 136k cold prefill; identity under V.
#   NST 5 ON: start levers (arena 48, ...) + turnover 0 + lever DL declared off (RADIANCE_DRAFT_LEND=0, _PROBE_BACKOFF=0,
#     _LEND_BORROW=0) + debug; DSpark NST 5 adaptive, depth 0 and the hybrid (VSL_HYB 1, match 8) pinned on in process
#     as hx-stack (HYB=1) does, so A / B / D differ only in the lend switches. One SETTLE block of A
#     (c1, c4, c8 at 30 s; the cost classes learn there, left out), then VSL_FNORDER (ABDDBA; pairs inside ABD | DBA, each
#     arm's mean position equal). Every block RE-ARMS D7 (draft_depth0 0 at a step, then the block's set with draft_depth0
#     1 + epoch_bump) and, with draft_lend 0, waits until the slab is back (lend.state 0, draft_resident == draft_units).
#     Phase order c4 before c8 for every arm (= hx-stack and xdl): the lent-ratio samples D's borrow needs (>= 16 in a class,
#     draft.h kBorrowMin) persist for the whole start (reset only in DraftController::configure, draft.cpp:86-90), so the
#     B block before the first D block seeds them; c8-first in D only would confound D vs B. Then the 136k cold prefill under
#     A (the lend acts only at depth 0 under load, never in a lone prefill); identity under A, D and draft_lend -1 (lent at
#     every step; requests until lend.state 2 first, the state read again after).
#   A in the NST 5 ON start vs hx-stack's NST 5 = the start levers alone (hx-stack's first NST 5 block is dropped, as the
#   settle block is here: the first-c4-after-start artefact); D vs hx-stack = everything; B vs A, D vs A, D vs B in process.
# PER PHASE: /stats (VRAM units, rank-0 hit, missed a step, stream MB a step), draft counters (zero / probe cycles, zero
#   moves), lend (counter deltas incl. borrow_cycles, gauges) + a 2 s sampler (lent / transit share, lent units max, the time
#   to the first lent sample and to 95 % of the maximum = the fill rate, VRAM units max).
# DROPPED for the ~60 min budget: a second ABD|DBA round at NST 5 (+21 min; pairs stay n = 2 a comparison), identity under B
#   (D holds B's placement at c1; -1 checks the lent placement), the 136k under B / D (lend inactive in a lone prefill), arm C.
# SUMMARY (vssum.py): per configuration ON vs OFF per metric (value, reference, %, distance to target: 1u >= 90, c4 >= 160,
#   c8 >= 200, 136k <= 35 s), resident units vs the audit estimate and the slope's predicted ms; in-process pairs; the DL
#   diagnostics a arm. "RESULT xvl summary: ..." then "RESULT xvl <item>: done".
# THE LEVERS ARE DATA (the table below): a later lever = one line; a later image = the wrapper's VSL_IMAGE.
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# VSL_IMAGE, VSL_ITEM, VSL_NST, VSL_BASE, VSL_ON, VSL_OFF (auto | own | hx), VSL_OFF_MAXAGE_H, VSL_BLOCKS0 (4), VSL_BLOCKS5
# (4, NST 5 without fn4 levers), VSL_HYB (1: the NST 5 hybrid pinned on, as hx-stack), VSL_FNARMS, VSL_FNORDER, VSL_FNIDENT, VSL_FNDEEP, VSL_SETTLE (1 | 0), VSL_ONDEBUG, VSL_DUR,
# VSL_TABLE_EXTRA, VSL_HXDIR, VSL_DRAFTER_MX.
# VSL_DRAFTER_MX (0 | 1; default 0 = as before; 10.11.2026 evening, ds41-int2): 1 adds "drafter_mx": 1 (lever DK, libr4d's fn4
#   switch, nothing declared at start) to the NST 5 fixed pin FIX5, so every NST 5 start here (ON, and an own OFF) runs the
#   pin hx-stack's NST 5 start runs with XST_DRAFTER_MX=1; needs the image's drafter_mx (ds41-int2, checked with the image).
#   hxreuse compares hx-stack's recorded pin (<tag>-meta.txt pin=, sorted compact JSON) with this NST's own (FIX0 / FIX5):
#   a start without the same pin (drafter_mx included) is not reused.
# DRYRUN=1 bash dsm-vramstack.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep
#   of its files) and the yaml (docker compose config --quiet); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-vramstack-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/vramstack; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
RM=$R/xdsm                                      # item a's dsm-225: deep-dsm-225.txt(.json) (136k cold reference)
RX=${VSL_HXDIR:-$R/xdsm/stack}                  # hx-stack's outputs (the OFF side when reusable)
export IMAGE=${VSL_IMAGE:-radiance-local:ds41-int-c05de277-5c13e555-l}
ITEM=${VSL_ITEM:-57-p4zyzzzzz-ds-hxz-vramstack}
DUR=${VSL_DUR:-75}; NB0=${VSL_BLOCKS0:-4}; NB5=${VSL_BLOCKS5:-4}; NSTS=${VSL_NST:-n0 n5}; OFFMODE=${VSL_OFF:-auto}; MAXAGE=${VSL_OFF_MAXAGE_H:-24}
BASE=${VSL_BASE-t0}; ONSEL=${VSL_ON:-all}; MATCH=8; HYB=${VSL_HYB:-1}; SETTLE=${VSL_SETTLE:-1}; ONDEBUG=${VSL_ONDEBUG-RADIANCE_DEBUG_VRAM=1}
DMX=${VSL_DRAFTER_MX:-0}
# the in-process arms when an fn4 lever applies (the lever DL design); empty VSL_FNARMS = A (switches 0) / B (on) from the table
FNARMS=${VSL_FNARMS-'A={"draft_lend":0,"draft_probe_backoff":0,"draft_lend_borrow":0} B={"draft_lend":1,"draft_probe_backoff":3,"draft_lend_borrow":0} D={"draft_lend":1,"draft_probe_backoff":3,"draft_lend_borrow":1}'}
FNORDER=${VSL_FNORDER:-ABDDBA}; FNFULL=${VSL_FNFULL:-D}; FNDEEP=${VSL_FNDEEP:-A}
FNIDENT=${VSL_FNIDENT-'A D I={"draft_lend":-1,"draft_probe_backoff":0,"draft_lend_borrow":0}'}
case $OFFMODE in auto|own|hx) ;; *) echo "VSL_OFF must be auto, own or hx (got '$OFFMODE')"; exit 1 ;; esac
for n in "$NB0" "$NB5"; do case $n in [2-9]) ;; *) echo "VSL_BLOCKS0 / VSL_BLOCKS5 must be 2..9 (got '$n')"; exit 1 ;; esac; done
case $HYB in 0|1) ;; *) echo "VSL_HYB must be 0 or 1 (got '$HYB')"; exit 1 ;; esac
case $DMX in 0|1) ;; *) echo "VSL_DRAFTER_MX must be 0 or 1 (got '$DMX')"; exit 1 ;; esac
case $SETTLE in 0|1) ;; *) echo "VSL_SETTLE must be 0 or 1 (got '$SETTLE')"; exit 1 ;; esac
for kv in $ONDEBUG; do case "$kv" in RADIANCE_DEBUG_*=*) ;; *) echo "VSL_ONDEBUG takes RADIANCE_DEBUG_*=VALUE words (logs only; got '$kv')"; exit 1 ;; esac; done
if [ -n "$FNARMS" ]; then
  for a in $FNARMS; do case $a in [A-Z]=\{*\}) ;; *) echo "VSL_FNARMS takes X={json} words without spaces (got '$a')"; exit 1 ;; esac; done
  case $FNORDER in *[!A-Z]*|"") echo "VSL_FNORDER takes arm letters (got '$FNORDER')"; exit 1 ;; esac
  for c in $(echo $FNORDER | fold -w1) $FNFULL $FNDEEP; do case " $FNARMS " in *" $c={"*) ;; *) echo "arm $c is not in VSL_FNARMS"; exit 1 ;; esac; done
fi
for n in $NSTS; do case $n in n0|n5) ;; *) echo "VSL_NST takes n0 / n5 (got '$n')"; exit 1 ;; esac; done

# ---------------------------------------------------------------- THE LEVER TABLE (one lever a line; a line starting '#' is off)
# name  kind  setting  needs  nst  units  decl  logok  what (free text)
#   kind     env  = start-time env NAME=VALUE[,NAME=VALUE]: exported in the shell AND written to the start's env file, which the
#                   yaml's env_file passes (a name the stack yaml does not list reaches the container too) -> separate starts
#            flag = start-time serve flag(s), ',' for a space (--embedding-placement,ram) -> separate starts
#            fn4  = an in-process switch SWITCH=ON (off = 0; POST /admin/fn4) -> A (0) / B (ON) blocks in the ON start
#   needs    a literal the image must carry (grep -z in bin/radiance, architectures/deepseek41.so, kernels/*.so), '-' = none;
#            a lever under test that the image lacks is skipped with a note (a VSL_BASE lever stops the item)
#   nst      n0 | n5 | any -- the starts it applies to (the drafter only exists at NST 5)
#   units    the audit's expected resident routed units a rank, NST 0 / NST 5 (summary: measured delta vs this)
#   decl     fn4 only: NAME=VALUE[,..] declared at the ON start (A and B alike), '-' = none
#   logok    a literal the start log must carry when the lever is on (',' for a space), '-' = none
VSL_TABLE='
t0      env   RADIANCE_PLACE_TURNOVER=0           RADIANCE_PLACE_TURNOVER          any  32/48   -  0,more,kept,free                turnover 0: the mover keeps no lent slot free (exact, vramsweep T0 IDENTICAL; in hx-stack as XST_VRAMENV)
arena0  env   RADIANCE_DS41_ARENA_DECODE_ROWS=8   RADIANCE_DS41_ARENA_DECODE_ROWS  n0   68/0    -  activation,arena:,8,tokens,use   arena decode level of 8 rows + its index score plane (needs RADIANCE_DS41_LEVELS=1: in the stack env)
arena5  env   RADIANCE_DS41_ARENA_DECODE_ROWS=48  RADIANCE_DS41_ARENA_DECODE_ROWS  n5   0/57    -  activation,arena:,48,tokens,use  arena decode level of 48 rows (DSpark verify rows) + its index score plane
bf16    env   RADIANCE_DS41_ROUTER_BF16=1         RADIANCE_DS41_ROUTER_BF16        any  20/20   -  -                               router / compressor projections stored bf16 (source BF16: lossless)
emb     env   RADIANCE_DS41_EMB_RAM=1             RADIANCE_DS41_EMB_RAM            any  14/14   -  a,rank),ram                     token embedding pinned in host memory (no --embedding-placement flag passed)
shadow  env   RADIANCE_PLACE_SHADOW_SLOTS=16      RADIANCE_PLACE_SHADOW_SLOTS      any  32/65   -  -                               shadow slots 32 -> 16 a class (mover line shows the count)
xpoff   env   RADIANCE_XPREFETCH=0                RADIANCE_XPREFETCH               any  11/16   -  expert,prefetch:,off            expert prefetch off (ring + buffers back to the experts)
dlend   fn4   draft_lend=1                        draft_lend_borrow                n5   0/217   RADIANCE_DRAFT_LEND=0,RADIANCE_DRAFT_PROBE_BACKOFF=0,RADIANCE_DRAFT_LEND_BORROW=0  draft,lend:,set,up  lever DL: the drafter slab lent while depth 0 holds (arms: VSL_FNARMS A / B / D)
#kvfp8  env   ENVNAME=1                           ENVNAME                          any  11/11   -  -                               KV window fp8 storage (not built)
'
TABLE="$VSL_TABLE
${VSL_TABLE_EXTRA:-}"
LROWS=$(echo "$TABLE" | awk 'NF && $1 !~ /^#/')
lrow() { echo "$LROWS" | awk -v n="$1" '$1 == n {print; exit}'; }
lf() { lrow "$1" | awk -v k="$2" '{print $k}'; }
lwhat() { lrow "$1" | awk '{for (i = 1; i <= 8; i++) $i = ""; sub(/^ +/, ""); print}'; }
LNAMES=$(echo "$LROWS" | awk '{print $1}')
while read -r nm kd st nd ns un dc lo rest; do
  [ -n "${nm:-}" ] || continue
  [ -n "${lo:-}" ] || { echo "lever table: '$nm' has fewer than 8 fields"; exit 1; }
  case $kd in env|flag|fn4) ;; *) echo "lever table: '$nm' kind '$kd' is not env / flag / fn4"; exit 1 ;; esac
  case $ns in n0|n5|any) ;; *) echo "lever table: '$nm' nst '$ns' is not n0 / n5 / any"; exit 1 ;; esac
  case $un in [0-9]*/[0-9]*) ;; *) echo "lever table: '$nm' units '$un' is not N0/N5"; exit 1 ;; esac
  case $kd in
    env) for w in ${st//,/ }; do case $w in [A-Z]*=*) ;; *) echo "lever table: '$nm' env word '$w' is not NAME=VALUE"; exit 1 ;; esac; done ;;
    fn4) case $st in [a-z]*=*) ;; *) echo "lever table: '$nm' fn4 setting '$st' is not switch=value"; exit 1 ;; esac ;;
    flag) case $st in --*) ;; *) echo "lever table: '$nm' flag '$st' does not start with --"; exit 1 ;; esac ;;
  esac
  [ $kd = fn4 ] || [ "$dc" = - ] || { echo "lever table: '$nm' decl is for fn4 levers only"; exit 1; }
  [ "$(echo "$LNAMES" | grep -cx "$nm")" = 1 ] || { echo "lever table: '$nm' appears twice"; exit 1; }
done <<< "$LROWS"
for nm in $BASE; do lrow $nm | grep -q . || { echo "VSL_BASE: no table lever '$nm'"; exit 1; }; [ "$(lf $nm 2)" = fn4 ] && { echo "VSL_BASE: '$nm' is an fn4 lever (base levers are start-time)"; exit 1; }; done
inbase() { case " $BASE " in *" $1 "*) return 0 ;; esac; return 1; }
if [ "$ONSEL" = all ]; then ONL=""; for nm in $LNAMES; do inbase $nm || ONL="$ONL $nm"; done
else ONL=""; for nm in $ONSEL; do lrow $nm | grep -q . || { echo "VSL_ON: no table lever '$nm'"; exit 1; }; inbase $nm && { echo "VSL_ON: '$nm' is in VSL_BASE"; exit 1; }; ONL="$ONL $nm"; done; fi
appl() { local n=$(lf $1 5); [ $n = any ] || [ $n = $2 ]; }        # appl LEVER NST
envw() { local nm w; for nm in "$@"; do case $(lf $nm 2) in env) w=$(lf $nm 3) ;; fn4) w=$(lf $nm 7); [ "$w" = - ] && w="" ;; *) w="" ;; esac; echo "${w//,/ }"; done | xargs; }
flagw() { local nm; for nm in "$@"; do [ "$(lf $nm 2)" = flag ] && { w=$(lf $nm 3); echo "${w//,/ }"; }; done | xargs; }
units() { local nm s=0 k; [ $1 = n0 ] && k=1 || k=2; shift; for nm in "$@"; do s=$((s + $(lf $nm 6 | cut -d/ -f$k))); done; echo $s; }
fnjson() {   # fnjson on|off LEVERS...: the fn4 levers' switches as one JSON object (off = 0)
  local m=$1 nm a=""; shift
  for nm in "$@"; do [ "$(lf $nm 2)" = fn4 ] && a="$a $(lf $nm 3)"; done
  python3 -c 'import json,sys
d = {}
for w in sys.argv[2:]:
    k, v = w.split("=", 1)
    try: v = int(v)
    except ValueError: pass
    d[k] = v if sys.argv[1] == "on" else 0
print(json.dumps(d, separators=(",", ":")))' $m $a   # compact: an arm word A={...} must hold no space
}
# every env name the table can set: unset before each start (nothing leaks from the shell)
TENV=$(for nm in $LNAMES; do envw $nm; done | tr ' ' '\n' | sed 's/=.*//' | sort -u | xargs)

GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; VLB=$HD/vlblocks.py; VSB=$HD/vsblocks.py; GGB=$HD/ggblocksdz2.py; VSS=$HD/vssum.py; DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1}'
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"', "draft_hybrid": '"$HYB"'}'
[ "$DMX" = 1 ] && FIX5="${FIX5%\}}, \"drafter_mx\": 1}"   # lever DK in the NST 5 pin (VSL_DRAFTER_MX=1), as hx-stack's XST_DRAFTER_MX=1
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM"
stackenv() {   # stackenv ENVWORDS: hx-stack's stack env (105's t3 arm + gg_gemv 2 + rule-w + K1 declared and on) + the levers' env
  local kv
  unset $SWITCHES $TENV
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $1; do export "$kv"; done
}
armset() {   # armset NST ARMJSON [EXTRAJSON]: the NST's fixed pin + the arm's switches (+ extra) as one POST body
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' \
    "$([ $1 = n0 ] && echo "$FIX0" || echo "$FIX5")" "$2" "${3:-{\}}"
}
# ---------------------------------------------------------------- helpers (dsm-stack.sh / dsm-vramsweep.sh, copied; tag xvl)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || { keeplog $tag-fail; return 1; }; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT xvl guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3, NST 0 bz dsd-d0 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xvl $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xvl $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xvl $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xvl $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
VKNOBS=""
vramknobs() {   # vramknobs TAG "ENVWORDS" "FLAGS" LEVERS...: the levers reached the engine -> VKNOBS; 1 on a miss
  local tag=$1 ew=$2 fw=$3 MV XP CE CA kv bad="" nm lo nz; shift 3
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  CA=" $(docker inspect -f '{{join .Args " "}}' $CN 2>/dev/null) "
  for kv in $ew; do echo "$CE" | grep -qxF "$kv" || bad="$bad env:${kv%%=*}"; done
  [ -n "$fw" ] && case "$CA" in *" $fw "*) ;; *) bad="$bad flags:'$fw'" ;; esac
  for nm in "$@"; do lo=$(lf $nm 8); [ "$lo" = - ] && continue
    docker logs $CN 2>&1 | grep -qF -- "${lo//,/ }" || bad="$bad log:$nm('${lo//,/ }')"; done
  for kv in $ew; do   # a lever env the engine refused or could not apply (its warning names the variable; debug env apart)
    case $kv in RADIANCE_DEBUG_*) continue ;; esac
    docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -qE 'is not used|is not a count|refused|cannot' && bad="$bad refused:${kv%%=*}($(docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -m1 -E 'is not used|is not a count|refused|cannot' | cut -c1-160))"
  done
  VKNOBS="levers [$*] env [${ew:-none}] flags [${fw:-none}]$([ -n "$bad" ] && echo " MISSING:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { log "RESULT xvl $tag: VIRHE a lever did not reach the engine:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { log "RESULT xvl $tag: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  case " $ew " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xvl $tag: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; } ;;
  esac
  return 0
}
modes() {   # modes TAG "ENVWORDS" "FLAGS" LEVERS...: hx-stack's t3 + rule-w + K1 checks + the VRAM knobs -> 0, else 1
  local tag=$1 ew=$2 fw=$3; shift 3
  [ "$DRY" = 1 ] && { echo "+ modes $tag: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to 4, no shared-expert-lane note, staging line, rule-w in every recorded pass, the lever K1 note (all three declared, switches at 1); VRAM knobs: levers [$*] env [${ew:-none}] in the container env, flags [${fw:-none}] in its args, each lever's log literal, mover shadow / kept-free a rank$(case " $ew " in *" RADIANCE_PLACE_TURNOVER=0 "*) echo " (each '0 more kept free')" ;; esac), expert prefetch state"; return 0; }
  local MLF SL AL XL KL NW NR ok=1
  vramknobs $tag "$ew" "$fw" "$@" || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  log "RESULT xvl $tag modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | VRAM: $VKNOBS"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xvl $tag: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xvl $tag: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xvl $tag: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xvl $tag: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xvl $tag: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xvl $tag: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xvl $tag: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  [ $ok = 1 ] || keeplog $tag-nomode
  [ $ok = 1 ]
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xvl $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
lendwait() {   # lendwait STATE [full]: 4-token requests until lend.state == STATE (full: and draft_resident == draft_units), <= 90 s
  [ "$DRY" = 1 ] && { echo "+ 4-token requests until lend.state == $1$([ "${2:-}" = full ] && echo " and draft_resident == draft_units") (<= 90 s)"; return 0; }
  local t0=$(date +%s) r
  while :; do
    nudge
    r=$(fn4get | python3 -c 'import json,sys
L = json.loads(sys.stdin.read()).get("lend")
print("nolend" if L is None else "ok" if L.get("state") == int(sys.argv[1]) and (sys.argv[2] != "full" or L.get("draft_resident") == L.get("draft_units")) else "no")' $1 "${2:-}" 2>/dev/null)
    case $r in ok|nolend) return 0 ;; esac
    [ $(( $(date +%s) - t0 )) -gt 90 ] && return 1
    sleep 0.5
  done
}
swnote() {   # swnote TAG NST FNLEVERS ARM=JSON...: GET at start (the stack's start values; every fn4 lever's switch listed at 0;
  # lever DL declared: lend.capable 1); each arm's set applied at a step; then the slab back (lend.state 0, all resident)
  local tag=$1 nst=$2 fnl=$3 s0 s a j bad="" sw; shift 3
  sw=$(for a in $fnl; do lf $a 3 | cut -d= -f1; [ "$a" = dlend ] && echo "draft_probe_backoff draft_lend_borrow"; done | xargs)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, $([ $nst = n0 ] && echo "draft_capable 0" || echo "draft_adaptive 1, draft_depth0 0, draft_hybrid 0, draft_capable 1"), epoch 0, switches listed at 0: [$sw]$(case " $fnl " in *" dlend "*) echo ", lend.capable 1" ;; esac)); then per arm POST + a 4-token request -> applied: $(for a in "$@"; do echo -n "${a%%=*} $(armset $nst "${a#*=}") "; done); then the slab back"; return 0; }
  s0=$(fn4get)
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; n0 = sys.argv[2] == "n0"
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and (s0["draft_capable"] == 0 if n0 else (s0["draft_capable"] == 1 and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_hybrid"] == 0))
ok = ok and all(w.get(k) == 0 for k in sys.argv[3].split() if k)
ok = ok and (sys.argv[4] != "1" or (s0.get("lend") or {}).get("capable") == 1)
sys.exit(0 if ok else 1)' "$s0" $nst "$sw" "$(case " $fnl " in *" dlend "*) echo 1 ;; *) echo 0 ;; esac)" 2>/dev/null || bad="at start"
  for a in "$@"; do
    j=$(armset $nst "${a#*=}" '{"epoch_bump": 1}'); fn4set "$j"; nudge; s=$(fn4get)
    python3 -c 'import json,sys
s, want = json.loads(sys.argv[1]), json.loads(sys.argv[2]); want.pop("epoch_bump", None)
sys.exit(0 if all(s["switches"].get(k) == v for k, v in want.items()) and not s["pending"] and s["epoch"] >= 1 else 1)' "$s" "$j" 2>/dev/null || bad="$bad arm ${a%%=*}"
  done
  fn4set "$(armset $nst "${1#*=}" '{"epoch_bump": 1}')"; lendwait 0 full || bad="$bad (the slab not back after the arm checks)"
  log "RESULT xvl $tag switch note: at start $(echo "$s0" | cut -c1-1200) | after the arm checks: $(fn4get | cut -c1-900)$([ -n "$bad" ] && echo " -- VIRHE not as expected:$bad")"
  [ -z "$bad" ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xvl $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline KEY LABEL RESULT: identity.txt + one RESULT line (a difference is reported, not hidden)
  echo "$1 $3" >> $O/identity.txt
  case "$3" in IDENTICAL*|"not compared"*) log "RESULT xvl identity $2: $3" ;;
    *) log "RESULT xvl identity $2: VIRHE $3 -- every lever here is exact by construction, so this is a defect" ;; esac
}
deep() {   # deep TAG NST ARMJSON: the arm's set, a reset, the 136k cold prefill (deepq.py, 512 greedy tokens) -> TTFT + the answer vs dsm-225
  local tag=$1 t0=$(date +%s) j
  fn4set "$(armset $2 "$3" '{"epoch_bump": 1}')"; nudge
  lendwait 0 full || log "RESULT xvl $tag: note -- the drafter slab was not back (lend.state 0, all resident) within 90 s before the 136k prefill"
  pcreset $tag-deep
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$tag.txt 135000 512 > $O/deep-$tag.out 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xvl $tag 136k cold prefill: TTFT (cached must be 0) vs $RM/deep-dsm-225.txt.json; answer vs $RM/deep-dsm-225.txt"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/deep-$tag.out | cut -c7-)
  [ -n "$j" ] || { log "RESULT xvl $tag 136k cold prefill: VIRHE no answer ($(tail -2 $O/deep-$tag.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/deep-$tag.json
  log "RESULT xvl $tag 136k cold prefill (deepq.py fixed prompt, 512 greedy tokens): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
try: r = json.load(open(sys.argv[2]))
except Exception: r = {}
t, rt = d.get("ttft"), r.get("ttft")
print("status %s, prompt %s tok, cached %s%s, TTFT %s s (dsm-225 cold %s s%s), %s prefill tok/s, %s tok" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), "" if d.get("cached") == 0 else " -- VIRHE not cold",
    t, rt, ", %+.1f %%" % ((t / rt - 1) * 100) if t and rt else "", round(d["prompt_tokens"] / t, 1) if t and d.get("prompt_tokens") else "-", d.get("tokens")))' "$j" $RM/deep-dsm-225.txt.json 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
  idline deep-$tag "$tag 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold)" "$(cmpfile $O/deep-$tag.txt $RM/deep-dsm-225.txt)"
}
ident() {   # ident TAG NST ARM ARMJSON: the arm's set (draft_lend -1: requests until lend.state 2), a reset, the 28k copy and the
  # c1 greedy set vs bz d0 (draft_lend -1: the lend state read again after -- it must have stayed 2)
  local tag=$1-i$3 lent=0 st
  case "$4" in *'"draft_lend":-1'*|*'"draft_lend": -1'*) lent=1 ;; esac
  fn4set "$(armset $2 "$4" '{"epoch_bump": 1}')"; nudge
  if [ $lent = 1 ]; then lendwait 2 || log "RESULT xvl $tag: VIRHE lend.state never reached 2 (lent) within 90 s -- the identity runs anyway, read the state line"
  else lendwait 0 full || log "RESULT xvl $tag: note -- the drafter slab was not back within 90 s before the identity runs"; fi
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity $tag (arm $3): 28k copy vs bz d0, greedy set c1 vs bz d0$([ $lent = 1 ] && echo "; lend.state 2 throughout")"; return 0; }
  st=$(fn4get)
  log "RESULT xvl $tag identity ran under: $(echo "$st" | cut -c1-900)"
  [ $lent = 1 ] && [ "$(echo "$st" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("lend", {}).get("state"))' 2>/dev/null)" != 2 ] \
    && log "RESULT xvl $tag: note -- lend.state not 2 after the identity runs (the lent placement may not have held throughout)"
  idline copy-$tag "$tag 28k copy vs bz d0 (NST 0, stock)" "$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
  idline gs-$tag "$tag greedy set c1 vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)"
}
blocks() {   # blocks TAG NST ORDER SETTLEARM ARM=JSON...: vlblocks.py (NST 5: D7 re-armed every block; SETTLEARM '-' = none)
  local tag=$1 nst=$2 ord=$3 st=$4 args="" a fx; shift 4
  for a in "$@"; do args="$args --arm '$a'"; done
  [ $nst = n5 ] && args="$args --rearm"
  [ "$st" != - ] && args="$args --settle $st --settle-phases c1,c4,c8 --settle-dur 30"
  fx=$([ $nst = n0 ] && echo "$FIX0" || echo "$FIX5")
  run "timeout 7200 python3 -I $VLB --url $URL --mb-dir $D --edit $EP --label $tag $args --order $ord --dur $DUR --warm 10 --fixed '$fx' --ready-s 90 --sample-s 2 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xvl $tag ${l:0:1600}"; done < <(grep -E '^(BLOCK|GGB \{"error)' $O/$tag-blocks.txt)
  log "RESULT xvl $tag counters a phase (block phase: vram units end, rank-0 hit %, missed a step, stream MB a step | lent share, lent max, first lent s, 95 % s, borrow cycles): $(paste -d'|' <(grep '^VSX ' $O/$tag-blocks.txt | cut -c5-) <(grep '^VDL ' $O/$tag-blocks.txt | cut -c5-) | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    a, b = l.rstrip("\n").split("|", 1)
    try: x, d = json.loads(a), json.loads(b)
    except ValueError: continue
    if x.get("phase") not in ("c1", "c4", "c8"): continue
    out.append("b%s %s: %s u, hit %s, miss %s, stream %s%s" % (x.get("block"), x.get("phase"), x.get("vram_units_end"), x.get("hit_pct_rank0"), x.get("missed_step_rank0"),
               x.get("stream_MB_step"), "" if d.get("s_lent_share") is None else " | lent %s, max %s, first %s, 95 %% %s, borrow %s" % (d.get("s_lent_share"),
               d.get("s_lent_units_max"), d.get("s_t_first_lent"), d.get("s_t_to_95pct_max"), d.get("borrow_cycles"))))
print("; ".join(out))' 2>&1 | cut -c1-3000)"
}
# ---------------------------------------------------------------- one start
vstart() {   # vstart NST off|on: the start, its decode blocks (+ settle at NST 5), the 136k cold prefill, identity
  local nst=$1 side=$2 tag=vl-$1-$2 P lim lv ew fw arms ord deeparm idarms st fnl a rc
  P=$O/$tag
  if [ $side = off ]; then lv="$(blv $nst)"; fnl=""
  else lv="$(blv $nst) $(stl $nst) $(fnl $nst)"; fnl="$(fnl $nst)"; fi
  lv=$(echo $lv)
  ew=$(envw $lv); fw=$(flagw $lv)
  [ $side = on ] && ew=$(echo $ew $ONDEBUG)
  armplan $nst $side
  arms=$AP_ARMS; ord=$AP_ORDER; st=$AP_SETTLE; deeparm=$AP_DEEP; idarms=$AP_IDENT
  stackenv "$ew"; export MODEL=$M225; HP29=19456; best_vars
  case $nst in
    n0) lim=30; export MNBT=4160 NST=0 XARGS="$XARGS --live${fw:+ $fw}" ;;
    n5) lim=70; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive${fw:+ $fw}" ;;
  esac
  export VSL_ENVFILE=$O/$tag.env
  log "RESULT xvl $tag: start ($nst $side: the stack + levers [$lv]; arms $arms, settle $st, order $ord, 136k under $deeparm, identity under [$(for a in $idarms; do echo -n "${a%%=*} "; done)]) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')| env file $VSL_ENVFILE [${ew:-empty}]"
  mkd $P
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $ew > $VSL_ENVFILE"; else printf '%s\n' $ew > $VSL_ENVFILE; fi
  gate $tag $lim; rc=$?
  [ $rc = 1 ] && { log "RESULT xvl $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xvl $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag "$ew" "$fw" $lv || { log "RESULT xvl $tag: VIRHE the modes / VRAM knobs are not as expected -> start stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    if [ $nst = n5 ]; then docker logs $CN 2>&1 | grep -q "DSpark drafter" && docker logs $CN 2>&1 | grep -q "draft depth: adaptive" \
      || { log "RESULT xvl $tag: VIRHE no DSpark drafter / adaptive depth line"; keeplog $tag-nodraft; down; return 1; }
    else docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xvl $tag: VIRHE a DSpark drafter line at NST 0"; keeplog $tag-nodraft; down; return 1; }; fi
    case " $lv " in *" dlend "*)
      log "RESULT xvl $tag lever DL at start: $(docker logs $CN 2>&1 | grep -E '^I\[[0-3]\] mover: lever DL: ' | sed -E 's/^I\[([0-9])\] mover: lever DL: ([0-9]+) drafter units in ([0-9]+) class.*lends ([0-9]+) trunk slots.*/r\1 \2 units \3 class \4 slots/' | tr '\n' ',' | sed 's/,$//') (DL agent: 96 / 1 / ~235)"
      [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: lever DL: [0-9]+ drafter units')" = 4 ] && ! docker logs $CN 2>&1 | grep -qE 'mover: lever DL: .*nothing to lend' \
        || { log "RESULT xvl $tag: VIRHE lever DL has no lend on every rank"; keeplog $tag-nodl; down; return 1; } ;;
    esac
  fi
  swnote $tag $nst "$fnl" $arms || { log "RESULT xvl $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  run "curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null"
  [ "$DRY" = 1 ] || log "RESULT xvl $tag experts at start: $(vramunits $O/$tag-stats0.json) | hx-stack $([ $nst = n0 ] && echo st-nst0 || echo st-stack): $(vramunits $RX/$([ $nst = n0 ] && echo st-nst0 || echo st-stack)-stats0.json) | bz d0: $(vramunits $RD/dsd-d0/stats-start.json)"
  follow $P
  stopfile || blocks $tag $nst $ord $st $arms
  if ! stopfile; then a=$(echo $arms | tr ' ' '\n' | grep "^$deeparm=" | cut -d= -f2-); deep $tag $nst "$a"; fi
  for a in $idarms; do stopfile || ident $tag $nst ${a%%=*} "${a#*=}"; done
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  log "RESULT xvl $tag: done"
}
armplan() {   # armplan NST off|on -> AP_ARMS (X=json words), AP_ORDER, AP_SETTLE (arm or -), AP_DEEP, AP_IDENT (X=json words), AP_BASE, AP_FULL
  local nst=$1 side=$2 f a n
  f=$([ $side = on ] && fnl $nst)
  n=$([ $nst = n0 ] && echo $NB0 || echo $NB5)
  if [ $side = off ]; then AP_ARMS="O={}"; AP_ORDER=$(printf 'O%.0s' $(seq $n)); AP_BASE=O; AP_FULL=O
  elif [ -z "$f" ]; then AP_ARMS="V={}"; AP_ORDER=$(printf 'V%.0s' $(seq $n)); AP_BASE=V; AP_FULL=V
  elif [ -n "$FNARMS" ]; then AP_ARMS=$FNARMS; AP_ORDER=$FNORDER; AP_BASE=$(echo $FNORDER | cut -c1); AP_FULL=$FNFULL
  else AP_ARMS="A=$(fnjson off $f) B=$(fnjson on $f)"; AP_ORDER=ABBA; AP_BASE=A; AP_FULL=B; fi
  AP_SETTLE=-; [ $nst = n5 ] && [ $SETTLE = 1 ] && AP_SETTLE=$AP_BASE
  AP_DEEP=$AP_BASE; [ -n "$f" ] && [ -n "$FNARMS" ] && AP_DEEP=$FNDEEP
  if [ -n "$f" ] && [ -n "$FNARMS" ]; then
    AP_IDENT=""; for a in $FNIDENT; do case $a in *=*) AP_IDENT="$AP_IDENT $a" ;; *) AP_IDENT="$AP_IDENT $(echo $AP_ARMS | tr ' ' '\n' | grep "^$a=")" ;; esac; done
  else AP_IDENT=$AP_ARMS; fi
  AP_IDENT=$(echo $AP_IDENT)
}
# per NST: the base levers, the start-time levers and the fn4 levers under test that apply and that the image carries
blv() { local nm; for nm in $BASE; do appl $nm $1 && echo $nm; done | xargs; }
stl() { local nm; for nm in $ONOK; do appl $nm $1 && [ "$(lf $nm 2)" != fn4 ] && echo $nm; done | xargs; }
fnl() { local nm; for nm in $ONOK; do appl $nm $1 && [ "$(lf $nm 2)" = fn4 ] && echo $nm; done | xargs; }
hxreuse() {   # hxreuse NST -> 0 when hx-stack's start of that NST is this configuration's OFF side (same image id + env, ...)
  local t m now=$(date +%s); [ $1 = n0 ] && t=st-nst0 || t=st-stack
  m=$(cat $RX/$t-meta.txt 2>/dev/null) || { echo "no $RX/$t-meta.txt"; return 1; }
  [ -z "$(flagw $(blv $1))" ] || { echo "base flags are not in hx-stack"; return 1; }
  python3 - "$m" "$IID" "$(envw $(blv $1) | tr ' ' ',')" $1 $DUR $MAXAGE $now $RX/$t-blocks.jsonl $HYB "$([ $1 = n0 ] && echo "$FIX0" || echo "$FIX5")" <<'PY'
import json, sys
m = dict(kv.split("=", 1) for kv in sys.argv[1].split() if "=" in kv)
iid, env, nst, dur, maxage, now, bl, hyb = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7]), sys.argv[8], sys.argv[9]
pin = json.loads(sys.argv[10])   # this NST's fixed pin (FIX0 / FIX5: drafter_mx 1 with VSL_DRAFTER_MX=1)
why = []
try: hpin = json.loads(m["pin"])
except (KeyError, ValueError): hpin = None
if hpin != pin: why.append("its pin %s, here %s" % (m.get("pin", "(none recorded)"), json.dumps(pin, sort_keys=True, separators=(",", ":"))))
if m.get("image") != iid: why.append("image %s, here %s" % (m.get("image", "?")[7:19], iid[7:19]))
if m.get("vramenv", "") != env: why.append("its VRAM env [%s], base here [%s]" % (m.get("vramenv", ""), env))
if m.get("nst") != nst: why.append("nst %s" % m.get("nst"))
if nst == "n5" and (m.get("arms") != "stack" or m.get("hyb") != hyb or m.get("match") != "8"): why.append("NST 5 arms %s hyb %s match %s (here stack / %s / 8)" % (m.get("arms"), m.get("hyb"), m.get("match"), hyb))
if m.get("dur") != dur: why.append("mb %s s" % m.get("dur"))
try: age = (now - int(m.get("epoch", 0))) / 3600.0
except ValueError: age = 1e9
if age > maxage: why.append("%.1f h old" % age)
try: good = sum(1 for l in open(bl) if l.strip() and json.loads(l).get("ok"))
except (OSError, ValueError): good = 0
if good < 3: why.append("%d good blocks" % good)
print("reusable (%d good blocks, %.1f h old, same image, VRAM env and pin)" % (good, age) if not why else "not reusable: " + "; ".join(why))
sys.exit(1 if why else 0)
PY
}
# ---------------------------------------------------------------- main
IID=$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null)
[ -n "$IID" ] || { log "RESULT xvl: VIRHE image $IMAGE not on this host"; exit 1; }
# the image, read only (DRYRUN too): a stack image (registry with depth 0 / hybrid, reset_prefix_cache, K1, D2) + each lever's literal
NEEDS=$(for nm in $LNAMES; do n=$(lf $nm 4); [ "$n" = - ] || echo "$n"; done | sort -u | xargs)
HAVE=$(timeout 180 docker run --rm --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels
s=yes; for l in draft_depth0 draft_hybrid_match reset_prefix_cache; do grep -qzF "$l" $B || s=no; done
grep -qzF RADIANCE_DS41_KVFOLD $A && grep -qzF rmsnorm_rope_roundtrip_kv_store $K/libr4d.so && grep -qzF r4d_gg_mr_pin $K/libr4d.so || s=no
echo "STACK $s"
grep -qzF "draft_lend_borrow, drafter_mx, drafter_bf, epoch_bump" $B && grep -qzF RADIANCE_DS41_DRAFTER_MX $K/libr4d.so && echo "DMX yes" || echo "DMX no"
for l in "$@"; do o=0; for f in $B $A $K/*.so; do grep -qzF -- "$l" "$f" && { o=1; break; }; done; echo "NEED $o $l"; done' sh $NEEDS 2>/dev/null)
echo "$HAVE" | grep -qx "STACK yes" || { log "RESULT xvl: VIRHE $IMAGE is not a stack image (depth 0 / hybrid registry, reset_prefix_cache, K1 in deepseek41.so / libr4d, lever D2)"; exit 1; }
[ "$DMX" = 1 ] && ! echo "$HAVE" | grep -qx "DMX yes" && { log "RESULT xvl: VIRHE VSL_DRAFTER_MX=1 but $IMAGE has no drafter_mx (the seventeen-switch registry, libr4d's RADIANCE_DS41_DRAFTER_MX)"; exit 1; }
needok() { local n=$(lf $1 4); [ "$n" = - ] || echo "$HAVE" | grep -qxF "NEED 1 $n"; }
for nm in $BASE; do needok $nm || { log "RESULT xvl: VIRHE base lever $nm needs '$(lf $nm 4)', which $IMAGE lacks"; exit 1; }; done
ONOK=""; SKIP=""
for nm in $ONL; do if needok $nm; then ONOK="$ONOK $nm"; else SKIP="$SKIP $nm(lacks '$(lf $nm 4)')"; fi; done
ONOK=$(echo $ONOK)
log "RESULT xvl $ITEM: start IMAGE=$IMAGE (${IID:7:12}) | base (ON in both sides = hx-stack's XST_VRAMENV) [$BASE] | under test [${ONOK:-none}]$([ -n "$SKIP" ] && echo ", skipped:$SKIP") | NST [$NSTS], OFF $OFFMODE (max age $MAXAGE h), blocks NST 0 $NB0 / NST 5 $NB5 (fn4 arms [$(for x in $FNARMS; do echo -n "$x "; done)] order $FNORDER, settle $SETTLE, D7 re-armed every NST 5 block, full $FNFULL, 136k under $FNDEEP, identity [$(echo $FNIDENT | sed "s/={[^}]*}//g")]), NST 5 hybrid $HYB (match $MATCH, both sides), NST 5 drafter_mx $DMX (both sides), ON-start debug [${ONDEBUG:-none}], mb $DUR s, yaml $(basename $Y) | refs: OFF = hx-stack's start when reusable, else own; identity bz d0 (copy, greedy c1), dsm-225 (136k deep, cold TTFT 51.56 s); slope xvs c1 / c4 / c8 0.409 / 1.309 / 2.034 ms per 100 units | targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s, 136k <= 35 s"
for nm in $BASE $ONOK; do echo "  lever $nm: $(lf $nm 2) $(lf $nm 3) [nst $(lf $nm 5), audit units $(lf $nm 6), decl $(lf $nm 7)] $(lwhat $nm)"; done
# the plan: which starts run (in order), and each configuration's OFF side
STARTS=""; PLANS=""; k=0
for nst in $NSTS; do
  s=$(stl $nst); f=$(fnl $nst)
  if [ -z "$s$f" ]; then log "RESULT xvl $nst: no lever under test applies here -> no start"; continue; fi
  src=none
  if [ -n "$s" ]; then
    if [ $OFFMODE != own ] && why=$(hxreuse $nst); then src=hx
    elif [ $OFFMODE = hx ]; then src=none
    else src=own; [ $OFFMODE = own ] && why="VSL_OFF=own"; fi
    log "RESULT xvl $nst OFF side: $src -- hx-stack's $([ $nst = n0 ] && echo st-nst0 || echo st-stack) start: $why$([ $src = none ] && echo " (VSL_OFF=hx: no own OFF start; the start levers get no reference)")"
  fi
  k=$((k + 1))
  if [ $src = own ]; then [ $((k % 2)) = 1 ] && STARTS="$STARTS $nst:off $nst:on" || STARTS="$STARTS $nst:on $nst:off"; else STARTS="$STARTS $nst:on"; fi
  PLANS="$PLANS $nst:$src"
done
MIN=0; for st in $STARTS; do armplan ${st%%:*} ${st#*:}; MIN=$(( MIN + 5 + ${#AP_ORDER} * 7 / 2 + $([ $AP_SETTLE = - ] && echo 0 || echo 2) + 2 + $(echo $AP_IDENT | wc -w) )); done
[ -n "$STARTS" ] && MIN=$((MIN + 1))
log "RESULT xvl plan: starts [${STARTS:- none}] (~$MIN min) | OFF sides [${PLANS:- none}]"
if [ "$DRY" != 1 ]; then
  [ -n "$STARTS" ] && docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xvl: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL; do grep -q "^ *$v:" $Y || { log "RESULT xvl: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q 'VSL_ENVFILE' $Y || { log "RESULT xvl: VIRHE $Y has no env_file entry"; exit 1; }
  for f in $GS $EP $VLB $VSB $GGB $VSS $DQ $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xvl: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xvl: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json; do [ -s $f ] || log "RESULT xvl: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $VLB $VSB $GGB $VSS $DQ $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ image $IMAGE (${IID:7:12}): stack image OK; $(echo "$HAVE" | grep '^DMX') (VSL_DRAFTER_MX=$DMX; FIX5 $FIX5); lever literals: $(echo "$HAVE" | grep '^NEED' | tr '\n' ';')"
fi
# the yaml parses with its env_file (quiet: prints nothing, never the environment)
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x VSL_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || { log "RESULT xvl: VIRHE docker compose config refuses $Y"; exit 1; }
[ "$DRY" = 1 ] && echo "+ yaml $Y: docker compose config OK (env_file \${VSL_ENVFILE})"
if [ -z "$STARTS" ]; then
  log "RESULT xvl summary: nothing to measure on $IMAGE -- every lever this image carries is in the base [${BASE:-none}] (hx-stack's XST_VRAMENV), none under test$([ -n "$SKIP" ] && echo "; skipped:$SKIP"); no start, no GPU time"
  log "RESULT xvl $ITEM: done -- nothing run"
  exit 0
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/plan.txt; }
evict
for st in $STARTS; do stopfile && break; vstart ${st%%:*} ${st#*:}; done
for p in $PLANS; do
  nst=${p%%:*}; src=${p#*:}; s=$(stl $nst); f=$(fnl $nst); t=vl-$nst-on
  armplan $nst on
  case $src in
    hx)  [ $nst = n0 ] && h=st-nst0 || h=st-stack
         off="off_src=hx off_arm=$([ $nst = n0 ] && echo N || echo A) off_skip=$([ $nst = n5 ] && echo 1 || echo 0) off_blocks=$RX/$h-blocks.jsonl off_stats0=$RX/$h-stats0.json off_deep=$RX/deep-$h.json" ;;
    own) off="off_src=own off_arm=O off_skip=0 off_blocks=$O/vl-$nst-off-blocks.jsonl off_stats0=$O/vl-$nst-off-stats0.json off_deep=$O/deep-vl-$nst-off.json" ;;
    *)   off="off_src=none off_arm=- off_skip=0 off_blocks=- off_stats0=- off_deep=-" ;;
  esac
  l="PLAN nst=$nst levers=$(echo $s $f | tr ' ' ',') fn=$([ -n "$f" ] && echo $f | tr ' ' ',' || echo -) arms=$(for a in $AP_ARMS; do echo -n "${a%%=*},"; done) base=$AP_BASE full=$AP_FULL expected_st=$(units $nst $s) expected_fn=$(units $nst $f) on_tag=$t on_blocks=$O/$t-blocks.jsonl on_stats0=$O/$t-stats0.json on_deep=$O/deep-$t.json $off"
  if [ "$DRY" = 1 ]; then echo "+ $l >> $O/plan.txt"; else echo "$l" >> $O/plan.txt; fi
done
if [ "$DRY" = 1 ]; then
  echo "+ python3 $VSS $O/plan.txt $O/identity.txt $RM/deep-dsm-225.txt.json > $O/summary.txt -> RESULT xvl NST lines (ON vs OFF per metric + target distance, units vs the slope's prediction), PAIRS lines (fn4 B vs A), RESULT xvl summary"
else
  python3 $VSS $O/plan.txt $O/identity.txt $RM/deep-dsm-225.txt.json > $O/summary.txt 2>&1
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xvl ${l:0:3000}" ;; esac; done < $O/summary.txt
  log "RESULT xvl summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-4500)"
fi
unset $SWITCHES $TENV
down
log "RESULT xvl $ITEM: done -- $O"
