#!/usr/bin/env python3
"""mkbody.py SRC DST -- dsm-yswl.sh from res12/dsm-res12.sh (08873cb8): every edit an exact, asserted replacement."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
n_edits = 0


def rep(old, new, count=1):
    global s, n_edits
    c = s.count(old)
    assert c == count, "%d x (want %d): %r" % (c, count, old[:160])
    s = s.replace(old, new)
    n_edits += 1


# ---- the header (everything above `set -u`)
i = s.index("set -u\n")
HEADER = r'''#!/bin/bash
# dsm-yswl.sh (12.11.2026, queue-ar/dsmain/yswl, a builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl.sh. Log tag "RESULT xswl ", out runs/xdsm/yswl (XSW_OUT). A COPY of res12/dsm-res12.sh
# 08873cb8 (ref12's start, pins, guards and image checks, ngshare, resprobe, the read / c8 machinery; tag xres -> xswl, knobs XRS_ ->
# XSW_; the original not changed; its unused functions kept) + yswl/swlsum.py (the summary + the strict identity) + yswl/resprobe.py
# (= res12's, byte for byte) + stack4/loadid.py (the STAGGERED identity run, zepb's way, read only).
# THE SWL / L2 SCREEN on int12 + ds41-swl (radiance-local:ds41-swl-29c003e8-579d0bc2-l: int12 bf8f557 + fn4 place_swap_lent 63 /
#   heat_min_gain_x100 64 / heat_hyst_x100 65, bin/radiance only), ref12's pins exactly (read back): S5T, draft_sample 0, SP 0 / 0, M1 on,
#   IX off, draft_depth0 3, sched_row_cap 48 (fn4 CAP48), ds41_arcoded 0, CBH 0 (k 2), exl3_x3d 1, ds41_e2 1, ds41_gq 3, ds41_mix_wv 0,
#   ds41_shexp_lane 0, RADIANCE_XPREFETCH=0, RADIANCE_HEAT_MOVES=16, RADIANCE_PLACE_TURNOVER=0, shadow 16, KVR=1 in the env; MAXSEQS 16.
# WHY (LEVERS "RES12 residency screen", ANALYSIS 12.11.): heat promotions take only budgeted swap slots (mover.cpp:1438, upstream
#   7938081's filter); on DS (no file tier) that keeps every swap out of the LENT slots, so 63-74 % of promotion attempts starve
#   (starved_slab) and their units come in ~2 steps later through the lent fill, beside lukewarm ones. L1 SWL lets a heat promotion take
#   a free lent slot (the unit keeps its host copy, as a fill's does); L2 lowers the heat guard's min_gain (1.5 shipped). Placement only:
#   the same kernels read the same bytes, so every answer must be byte-identical across arms.
# ONE FRESH START (BASE = ref12's start on ngshare, copy path, no attach, compose guard in the item), the arms IN-PROCESS (fn4, read back
#   in GET /admin/fn4 AND on every rank in /stats experts.ranks[].place_swl / heat_min_gain_x100):
#     B       place_swap_lent 0, heat_min_gain_x100 150 (= int12 to the bit)        L1      place_swap_lent 1, heat_min_gain_x100 150
#     L1G075  place_swap_lent 1, heat_min_gain_x100 75                               L1G100  place_swap_lent 1, heat_min_gain_x100 100 (last, budget)
#   heat_hyst_x100 125 throughout (the shipped hysteresis).
# THE PLAN (XSW_STAG / XSW_ABBA / XSW_G100):
#   (1) STAGGERED identity under B (coordinator 12.11.: no staggered check on the int12 reference path since CAP48 / int10-12): a1B =
#       greedyset.py's 20 prompts c1 ALONE (T 0, 192 tok), then stack4/loadid.py run (c4 streams + the 20 probes one at a time, each with a
#       FRESH 8192-token prefill joining 0.5 s in: the probe's decode rows ride MIXED steps) -> ldB vs a1B STRICT (a difference = VIRHE,
#       the item goes on and reports it);
#   (2) B: its c8 block FIRST (harness/ncmb3.py --prompt-set fixed --reset-per-block 1, 35 s) + its P12SPLIT wall split, then code, essay,
#       code, essay (the two SWITCHED essays: each right after a code read) and essay (the WARM essay: after an essay);
#   (3) L1 the same; (4) the c8 ABBA halves: one more c8 block under L1, then one under B (c8 order B L1 L1 B);
#   (5) L1G075 as (2); (6) STAGGERED under L1 (budget): a1L1 alone + ldL1 vs a1L1 strict, a1L1 vs a1B strict; (7) L1G100 (budget, last).
#   Per read (rank 0's mover, /stats deltas): tok/s, ms / tokens a step, stream / h2d MB a step, starved_slab, promotions, direct
#   promotions (= promotions - rank 0's lent fills), lent fills, SWL lent promotions (lent.*.swaps), misses (rank 0 routed - resident).
# SUMMARY (swlsum.py, lines "SWL "): an ARM line each (c8 blocks + split, switched / warm essays, code reads), C8PAIR (the ABBA halves),
#   a VERDICT an arm vs the in-process B ("switch X %, warm W %, c8 Y %"), an INFO line vs res12 BASE-lf0 (another start, NOT re-run:
#   switched 19.30 ms / 81.66 tok/s, warm 17.41 / 90.58, c8 183.1), IDENTITY: every read's answer byte-compared with the first read of its
#   prompt in B -- any difference a VIRHE.
# GUARDS: ref12's (every switch read back, 0 request errors, the "i11" counters, KVR read back) + the arms' switches on every rank +
#   start values place_swap_lent 0 / heat_min_gain_x100 150 / heat_hyst_x100 125 and their range refusals + the strict identities +
#   the compose guard (the item). ~20 min (start ~5.5 with the n-gram fill, staggered B ~2.5, an arm ~1.7, a c8 half ~1, staggered L1
#   ~2.5, L1G100 ~1.7); BUDGET XSW_BUDGET_MIN 22 (a section that would pass it is skipped, logged: L1G100 first, then staggered L1).
#   NO evict, no cc_sim, no power / host / fan settings; production compose files read only.
# DISK (TESTING.md 12.11. DISK RULES): guard >= 11000 MB free before the start; the engine log gzipped at capture; at the end tidy()
#   gzips the start log and the c8 samples and bundles the /stats + fn4 snapshots (snapshots.tar.gz).
# MOCK: XSW_SW=0 = a server without the int9b .. swl switches (queue-ar/tools/mockrad; the arms post nothing new: NOT a reading).
# DRYRUN=1 bash dsm-yswl.sh : prints every command; checks the image literals (int9 .. int12 + swl, the plugin md5), every helper (+
#   selftests: resprobe.py, swlsum.py, loadid.py, p12win.py, the harness), the ngramshare files, both yamls, the sidecars, the disk.
'''
s = HEADER + s[i:]
n_edits += 1

# ---- names, tag, knobs (global)
s = s.replace("XRS_", "XSW_").replace("xres", "xswl")
s = s.replace("RESIDENCY SCREEN", "SWL SCREEN")
rep('O=${XSW_OUT:-$R/xdsm/res12}', 'O=${XSW_OUT:-$R/xdsm/yswl}')
rep('ITEM=${XSW_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzz-res}', 'ITEM=${XSW_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yswl}')
rep('RSUM=$HD/ressum.py;', 'RSUM=$HD/swlsum.py;')
rep('ARMS=${XSW_ARMS-BASE GAIN2 GAIN05 SHADOW32};', 'ARMS=${XSW_ARMS-BASE};')
rep('STEST=${XSW_START_MIN:-6}; SETEST=${XSW_SET_MIN:-3}   # budget estimates (min): a start (+ checks), a read set (8 reads + c8)',
    'STEST=${XSW_START_MIN:-6}; SETEST=${XSW_SET_MIN:-2}   # budget estimates (min): a start (+ checks), an arm (c8 + 5 reads)\n'
    'C8EST=${XSW_C8_MIN:-1}; STAGEST=${XSW_STAG_MIN:-3}   # a c8 ABBA half; a staggered identity (a1 alone + the loadid run)\n'
    'STAG=${XSW_STAG:-2}; ABBA=${XSW_ABBA:-1}; G100=${XSW_G100:-1}   # staggered: 0 none, 1 B, 2 B + L1 (budget); the c8 ABBA halves; L1G100 (budget)\n'
    'LDPF=${XSW_LDPF:-8192}; LDCONC=${XSW_LDCONC:-4}; C8ALL=""   # loadid.py run: the probes\' fresh prefill and the c4 streams (zepb\'s); every c8 block')
rep('C16=${XSW_C16:-1}; C16NM=""', 'C16=${XSW_C16:-0}; C16NM=""')
rep('DISKMIN=${XSW_DISKMIN_MB:-11000}; BUDGET=${XSW_BUDGET_MIN:-40};', 'DISKMIN=${XSW_DISKMIN_MB:-11000}; BUDGET=${XSW_BUDGET_MIN:-22};')
rep('''for v in $STEST $SETEST $C16; do case $v in''', '''for v in $STEST $SETEST $C16 $C8EST $STAGEST $LDPF $LDCONC; do case $v in''')
rep('''case $BASELF in 0|1|01|10) ;; *) echo "XSW_BASELF 0 / 1 / 01 / 10 (BASE's place_lf sets in order)"; exit 1 ;; esac''',
    '''case $BASELF in 0|1|01|10) ;; *) echo "XSW_BASELF 0 / 1 / 01 / 10 (BASE's place_lf sets in order)"; exit 1 ;; esac
case $STAG in 0|1|2) ;; *) echo "XSW_STAG 0 (none) / 1 (B) / 2 (B + L1)"; exit 1 ;; esac
case $ABBA$G100 in [01][01]) ;; *) echo "XSW_ABBA / XSW_G100 are 0 / 1"; exit 1 ;; esac''')

# ---- the arms: the base pin carries today's SWL / L2 values explicitly; B / L1 / L1G075 / L1G100
rep('''\\"sched_cb_hold_ms\\": 0, \\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1}"''',
    '''\\"sched_cb_hold_ms\\": 0, \\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1, \\"place_swap_lent\\": 0, \\"heat_min_gain_x100\\": 150, \\"heat_hyst_x100\\": 125}"''')
rep('''    LF1) echo '{"place_lf": 1}' ;;   # lever LF (fn4 place_lf 1: the lent slots' fill ranks by the long-memory decode heat; the mock has it too)\n''', '''    LF1) echo '{"place_lf": 1}' ;;   # lever LF (fn4 place_lf 1: the lent slots' fill ranks by the long-memory decode heat; the mock has it too)
    B|L1|L1G075|L1G100) [ $SW = 1 ] && { set -- $(swlwant $1); echo "{\\"place_swap_lent\\": $1, \\"heat_min_gain_x100\\": $2}"; } || echo '{}' ;;   # ds41-swl\n''')
rep('''NAMEOF() { case $1 in O) echo "O (THE REFERENCE PIN, place_lf 0)" ;;''',
    '''swlwant() { case $1 in B) echo "0 150" ;; L1) echo "1 150" ;; L1G075) echo "1 75" ;; L1G100) echo "1 100" ;; *) echo "0 150" ;; esac; }   # ARM -> place_swap_lent heat_min_gain_x100
NAMEOF() { case $1 in B) echo "B (int12: place_swap_lent 0, heat_min_gain_x100 150)" ;; L1) echo "L1 (SWL: place_swap_lent 1, heat_min_gain_x100 150)" ;;
  L1G075) echo "L1G075 (SWL + L2: place_swap_lent 1, heat_min_gain_x100 75)" ;; L1G100) echo "L1G100 (SWL + L2: place_swap_lent 1, heat_min_gain_x100 100)" ;; *) NAMEOF0 $1 ;; esac; }
NAMEOF0() { case $1 in O) echo "O (THE REFERENCE PIN, place_lf 0)" ;;''')

# ---- blockx: the SWL counters a read, and the arm's switches read back on every rank
rep('''      fills=lent(rb, "fills") - lent(ra, "fills"), fills_lf=lent(rb, "fills_lf") - lent(ra, "fills_lf"), place_lf=[r.get("place_lf") for r in rb])))' $O/st-$nm-0.json $O/st-$nm-1.json "$j" 2>/dev/null || echo '{}')''',
    '''      fills=lent(rb, "fills") - lent(ra, "fills"), fills_lf=lent(rb, "fills_lf") - lent(ra, "fills_lf"), place_lf=[r.get("place_lf") for r in rb],
      swaps=lent(rb, "swaps") - lent(ra, "swaps"), fills_r0=lent(rb[:1], "fills") - lent(ra[:1], "fills"), swaps_r0=lent(rb[:1], "swaps") - lent(ra[:1], "swaps"),
      direct=e("promotions") - (lent(rb[:1], "fills") - lent(ra[:1], "fills")),
      misses_r0=int(((rb[0].get("routed") or 0) - (rb[0].get("routed_res") or 0)) - ((ra[0].get("routed") or 0) - (ra[0].get("routed_res") or 0))) if ra and rb else None,
      place_swl=[r.get("place_swl") for r in rb], mg=[r.get("heat_min_gain_x100") for r in rb], hy=[r.get("heat_hyst_x100") for r in rb])))' $O/st-$nm-0.json $O/st-$nm-1.json "$j" 2>/dev/null || echo '{}')''')
rep('''  sg=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); want = 1 if sys.argv[2] == "LF1" else 0; sw = sys.argv[3] == "1"
pl = d.get("place_lf") or []
if pl and all(x == want for x in pl): sys.exit(0)
print(("VIRHE" if sw else "INFO (XSW_SW=0)") + " place_lf on the ranks (/stats) %s, the arm says %d" % (pl or "absent", want))' "$st" $a $SW 2>&1 | head -1)''',
    '''  sg=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); sw = sys.argv[3] == "1"; ws, wm = int(sys.argv[4]), int(sys.argv[5])
pl, ps, mg, hy = d.get("place_lf") or [], d.get("place_swl") or [], d.get("mg") or [], d.get("hy") or []
if pl and all(x == 0 for x in pl) and ps and all(x == ws for x in ps) and mg and all(x == wm for x in mg) and hy and all(x == 125 for x in hy): sys.exit(0)
print(("VIRHE" if sw else "INFO (XSW_SW=0)") + " the arm %s on the ranks (/stats): place_swl %s (want %d), heat_min_gain_x100 %s (want %d), heat_hyst_x100 %s (want 125), place_lf %s (want 0)" % (
    sys.argv[2], ps or "absent", ws, mg or "absent", wm, hy or "absent", pl or "absent"))' "$st" $a $SW $(swlwant $a) 2>&1 | head -1)''')
rep('''lent fills / fills_lf (all ranks), place_lf on every rank ($([ $a = LF1 ] && echo 1 || echo 0) x4, else VIRHE$([ $SW = 1 ] || echo "; INFO under XSW_SW=0")); i11 counters -> $O/blocks.jsonl"''',
    '''lent fills / fills_lf (all ranks), rank 0: lent fills, SWL lent promotions (swaps), direct promotions (promotions - fills), misses; on every rank place_swl / heat_min_gain_x100 / heat_hyst_x100 = the arm's ($(swlwant $a) 125) and place_lf 0 (else VIRHE$([ $SW = 1 ] || echo "; INFO under XSW_SW=0")); i11 counters -> $O/blocks.jsonl"''')

# ---- the plan: swlset / c8b / swlstag / swlplan / c8split replace resset
a = s.index("resset() {")
b = s.index("c16sum() {")
NEWFN = r'''swlset() {   # swlset K ARM: one in-process arm -- its c8 block FIRST (harness/ncmb3.py fixed prompts, reset per block, ${C8WIN} s), then
             # code, essay, code, essay (the two SWITCHED essays: each right after a code read) and essay (the WARM essay: after an essay)
  local k=$1 a=$2 r
  SETN=$a; SETK=$k
  budok "arm $a ($(NAMEOF $a): c8 + 5 reads)" $SETEST || return 0
  arm $a || { xlog "arm $a: VIRHE the pin is not applied -- no reading"; return 1; }
  [ "$DRY" = 1 ] || python3 -c 'import json,sys; print(json.dumps(dict(k=int(sys.argv[1]), set=sys.argv[2], arm=sys.argv[2], start=sys.argv[3], c8="c8-%s-S-%s" % (sys.argv[1], sys.argv[2]), kind="arm", desc=sys.argv[4], utc0=float(sys.argv[5]))))' $k $a $SARM "$(NAMEOF $a)" $(utcnow) >> $O/sets.jsonl
  xlog "arm $a (k $k, $(NAMEOF $a) $(armj $a)): read back GET /admin/fn4 $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print("place_swap_lent %s heat_min_gain_x100 %s heat_hyst_x100 %s place_lf %s" % tuple(w.get(k) for k in ("place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "place_lf")))' 2>/dev/null) | c8 block FIRST, then code, essay, code, essay (SWITCHED), essay (WARM)"
  mbx 8 $k $a S || true
  C8S="$C8S c8-$k-S-$a"; C8ALL="$C8ALL c8-$k-S-$a"
  stopfile && return 0
  for r in 1 2; do blockx XC $a code || true; stopfile && return 0; blockx XE $a essay || true; stopfile && return 0; done
  blockx XW $a essay || true
}
c8b() {   # c8b K ARM: one more c8 block of ARM, no reads (the ABBA halves: the c8 order B L1 L1 B)
  local k=$1 a=$2
  budok "c8 block $a (ABBA half)" $C8EST || return 0
  arm $a || { xlog "c8 $a: VIRHE the pin is not applied -- no block"; return 1; }
  [ "$DRY" = 1 ] || python3 -c 'import json,sys; print(json.dumps(dict(k=int(sys.argv[1]), set=sys.argv[2], arm=sys.argv[2], start=sys.argv[3], c8="c8-%s-S-%s" % (sys.argv[1], sys.argv[2]), kind="c8", desc=sys.argv[4], utc0=float(sys.argv[5]))))' $k $a $SARM "$(NAMEOF $a)" $(utcnow) >> $O/sets.jsonl
  mbx 8 $k $a S || true
  C8S="$C8S c8-$k-S-$a"; C8ALL="$C8ALL c8-$k-S-$a"
}
swlstag() {   # swlstag ARM: the STRICT staggered-arrival identity under ARM (zepb's epbstag): a1ARM = greedyset.py's 20 prompts c1 ALONE
              # (T 0, $MT tok), then stack4/loadid.py run (LDCONC c4 streams + the 20 probes one at a time, each with a FRESH LDPF-token prefill
              # joining 0.5 s in: the probe's decode rows ride MIXED steps) -> ldARM vs a1ARM strict; under L1 also a1L1 vs a1B strict (the
              # lever is placement only). A difference = VIRHE, reported, the item goes on.
  local A=$1 t0=$(date +%s) l k
  budok "staggered identity $A (a1 alone + loadid run)" $STAGEST || return 0
  alone xswl $A a1$A || { xlog "staggered $A: VIRHE a1$A not taken -- no staggered check under $A"; return 0; }
  arm $A || return 0
  pcreset xswl-ld$A; ssnap xswl-ld$A-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/xswl-ld$A.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/xswl-ld$A.txt 2>&1)"
  ssnap xswl-ld$A-1
  cmpx xswl ld$A-vs-a1$A $O/xswl-a1$A.jsonl $O/xswl-ld$A.jsonl
  [ $A = B ] || cmpx xswl a1$A-vs-a1B $O/xswl-a1B.jsonl $O/xswl-a1$A.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT xswl staggered $A: LOADID (probes, MIXED steps during the probes, prefills cold) + ld$A vs a1$A strict$([ $A = B ] || echo " + a1$A vs a1B strict") (a difference = VIRHE, the item goes on)"; return 0; }
  xlog "staggered $A ($(NAMEOF $A); loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s with a1): $(grep -m1 '^LOADID ' $O/xswl-ld$A.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/xswl-ld$A.jsonl 2>&1 | cut -c1-200) | cap $(capd xswl-ld$A-0 xswl-ld$A-1)"
  for k in ld$A-vs-a1$A $([ $A = B ] || echo a1$A-vs-a1B); do
    l=$(grep "^xswl-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) ;; *) xlog "staggered $A: $([ $SW = 1 ] && echo VIRHE || echo "INFO (XSW_SW=0)") ${l:-$k not compared} (strict: $([ $k = ld$A-vs-a1$A ] && echo "a decode row riding MIXED steps must keep its alone bytes" || echo "placement cannot change a T 0 answer")) -- reported, the item goes on" ;; esac
  done
}
swlplan() {   # THE PLAN in this one start (module header): staggered B, B, L1, the c8 ABBA halves (L1, B), L1G075, staggered L1, L1G100
  [ $STAG -ge 1 ] && { swlstag B || true; }; stopfile && return 0
  swlset 1 B || true; stopfile && return 0
  swlset 2 L1 || true; stopfile && return 0
  if [ $ABBA = 1 ]; then c8b 3 L1 || true; stopfile && return 0; c8b 4 B || true; stopfile && return 0; fi
  swlset 5 L1G075 || true; stopfile && return 0
  [ $STAG = 2 ] && { swlstag L1 || true; }; stopfile && return 0
  [ $G100 = 1 ] && { swlset 6 L1G100 || true; }
  return 0
}
c8split() {   # c8split TAG: p12win.py split over every c8 block of this start (its marks window, its st-w0 / st-w1 snapshots, its MB line)
              # -> $O/c8/<b>-split.txt, one P12SPLIT line a block (decode / newcomer-MIXED / host-gap shares, ms a decode step)
  local EL=$O/$1-engine-full.log.gz b w
  if [ "$DRY" = 1 ]; then for b in ${C8ALL:-c8-1-S-B}; do echo "+ python3 -I -B $R12S split $EL --win $b:<marks of $O/c8/$b-reqs.jsonl> --mbline $O/c8/$b.txt --sw0 $O/c8/$b-st-w0.json --sw1 $O/c8/$b-st-w1.json --target 200 --ref 'ref12 c8:8.28:36.1:0.803:184.5' > $O/c8/$b-split.txt -> RESULT xswl c8 split $b: P12SPLIT"; done; return 0; fi
  for b in $C8ALL; do
    w=$(mwin $O/c8/$b-reqs.jsonl)
    [ -n "$w" ] || { xlog "c8 split $b: VIRHE no window (no marks in $O/c8/$b-reqs.jsonl)"; continue; }
    timeout 300 python3 -I -B $R12S split $EL --win $b:$w --mbline $O/c8/$b.txt --sw0 $O/c8/$b-st-w0.json --sw1 $O/c8/$b-st-w1.json --target 200 --ref "ref12 c8:8.28:36.1:0.803:184.5" > $O/c8/$b-split.txt 2>&1
    xlog "c8 split $b: $(grep -m1 '^P12SPLIT' $O/c8/$b-split.txt | cut -c1-1200)$(grep -q '^Traceback' $O/c8/$b-split.txt && echo " -- VIRHE a Traceback in p12win.py: $(tail -1 $O/c8/$b-split.txt | cut -c1-200)")"
  done
}
'''
s = s[:a] + NEWFN + s[b:]
n_edits += 1

# ---- rstart: the plan instead of the place_lf read sets; c8split after the capture; c16 only when asked
rep('''  for lf in $([ $sa = BASE ] && echo $BASELF | sed 's/./& /g' || echo 0); do
    stopfile && break
    SETS=$((SETS + 1)); resset $SETS $sa $lf || true
  done
  capture $O/$tag-engine-full.log.gz
  c8mixset $tag
  [ $sa = BASE ] && c16sum $tag''',
    '''  swlplan || true
  capture $O/$tag-engine-full.log.gz
  c8mixset $tag
  c8split $tag
  [ $sa = BASE ] && [ $C16 = 1 ] && c16sum $tag''')
rep('''  xlog "$tag: at the end: $(i11now) | place_lf $(fn4get | python3 -c 'import json,sys; print(json.load(sys.stdin)["switches"].get("place_lf"))' 2>/dev/null) |''',
    '''  xlog "$tag: at the end: $(i11now) | $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print("place_lf %s place_swap_lent %s heat_min_gain_x100 %s heat_hyst_x100 %s" % tuple(w.get(k) for k in ("place_lf", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100")))' 2>/dev/null) |''')
rep('''    est=$((STEST + SETEST)); [ $sa = BASE ] && est=$((STEST + SETEST * ${#BASELF} + 2 * C16))''', '''    est=$((STEST + SETEST))''')

# ---- summary: swlsum.py, base B
rep('''summary() {   # res12/ressum.py over blocks.jsonl / mb.jsonl / sets.jsonl / the c8 mix files: a table line a set, a VERDICT line an arm vs BASE-lf0,''',
    '''summary() {   # yswl/swlsum.py over blocks.jsonl / mb.jsonl / sets.jsonl / the c8 mix + split files: an ARM line each, C8PAIR, a VERDICT an arm vs B,''')
rep('''  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $RSUM summary --dir $O --base BASE-lf0 > $O/summary.txt -> RESULT xswl RES12 lines: a set (switched essay ms a step / tok/s / stream MB / starved_slab, warm essay ms a step / stream MB, c8 agg / ms a step / decode_share / mix), a VERDICT an arm ('helps the switch by X % at c8 Y %'), IDENTITY (every read's answer vs the first of its prompt, across arms and sets; a difference = VIRHE)"; return 0; fi''',
    '''  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $RSUM summary --dir $O --base B > $O/summary.txt -> RESULT xswl SWL lines: ARM (c8 blocks + split, switched / warm essays, code: tok/s, ms a step, stream / h2d MB, starved_slab, direct promotions, lent fills, SWL lent promotions, misses), C8PAIR (ABBA halves), VERDICT an arm vs B ('switch X %, warm W %, c8 Y %'), INFO vs res12 BASE-lf0 (not re-run), IDENTITY (every read's answer vs the first of its prompt in B, byte for byte; a difference = VIRHE), TABLE"; return 0; fi''')
rep('''  timeout 120 python3 -I -B $RSUM summary --dir $O --base BASE-lf0 > $O/summary.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2400}"; done < <(grep -h '^RES12' $O/summary.txt 2>/dev/null)
  grep -q '^Traceback' $O/summary.txt && xlog "summary: VIRHE a Traceback in ressum.py''',
    '''  timeout 120 python3 -I -B $RSUM summary --dir $O --base B > $O/summary.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2400}"; done < <(grep -h '^SWL' $O/summary.txt 2>/dev/null)
  grep -q '^Traceback' $O/summary.txt && xlog "summary: VIRHE a Traceback in swlsum.py''')

# ---- swnote: the new switches' start values and range refusals
rep('''  local s0 g c1=400 c2=400 c3=400 c5=400 c6=400 c7=400 c8=400 c9=400 c4 a sets="" bad=""''',
    '''  local s0 g c1=400 c2=400 c3=400 c5=400 c6=400 c7=400 c8=400 c9=400 c10=400 c11=400 c4 a sets="" bad=""''')
rep('''  if [ $SW = 1 ]; then c9=$(fn4code '{"exl3_x3d": 2}'); fi''',
    '''  if [ $SW = 1 ]; then c9=$(fn4code '{"exl3_x3d": 2}'); c10=$(fn4code '{"place_swap_lent": 2}'); c11=$(fn4code '{"heat_min_gain_x100": 10001}'); fi''')
rep('''for k in ("exl3_x3d", "sched_cb_hold_ms", "sched_cb_hold_k",''', '''for k in ("place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "exl3_x3d", "sched_cb_hold_ms", "sched_cb_hold_k",''')
rep(''', sched_cb_hold_k 1 -> HTTP $c8, exl3_x3d 2 -> HTTP $c9" ||''', ''', sched_cb_hold_k 1 -> HTTP $c8, exl3_x3d 2 -> HTTP $c9, place_swap_lent 2 -> HTTP $c10, heat_min_gain_x100 10001 -> HTTP $c11" ||''')
rep('''    ok = ok and w.get("sched_cb_hold_ms") == 0 and w.get("sched_cb_hold_k") == 2 and isinstance(s0.get("cbh"), dict) and w.get("exl3_x3d") == 0''',
    '''    ok = ok and w.get("sched_cb_hold_ms") == 0 and w.get("sched_cb_hold_k") == 2 and isinstance(s0.get("cbh"), dict) and w.get("exl3_x3d") == 0
    ok = ok and w.get("place_swap_lent") == 0 and w.get("heat_min_gain_x100") == 150 and w.get("heat_hyst_x100") == 125   # ds41-swl: the int12 values''')
rep('''[ "$c8" = 400 ] && [ "$c9" = 400 ] && [ -z "$bad" ]''', '''[ "$c8" = 400 ] && [ "$c9" = 400 ] && [ "$c10" = 400 ] && [ "$c11" = 400 ] && [ -z "$bad" ]''')
rep('''/ sched_cb_hold_k 1 / exl3_x3d 2") -> 400; sched_cb_hold_ms 0 / k 2, exl3_x3d 0 and "cbh" at start''',
    '''/ sched_cb_hold_k 1 / exl3_x3d 2 / place_swap_lent 2 / heat_min_gain_x100 10001") -> 400; sched_cb_hold_ms 0 / k 2, exl3_x3d 0, place_swap_lent 0, heat_min_gain_x100 150, heat_hyst_x100 125 and "cbh" at start''')

# ---- the image: swl's own literals (strict), the default image
rep('''PLUGMD5=${XSW_PLUGMD5:-9823cdde}''', '''# ds41-swl's own (VIRHE when missing; SKIP only under DRYRUN=1 + XSW_DRY_ALLOW_OLD=1): levers SWL / L2 in bin/radiance
IMGLITSW="B|exl3_x3d, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, epoch_bump)
B|place_swap_lent
B|heat_min_gain_x100
B|heat_hyst_x100
B|RADIANCE_PLACE_SWAP_LENT
B|RADIANCE_HEAT_MIN_GAIN
B|heat promotions into free lent slots (lever SWL
B|place_swl
P|swl-PROVENANCE.txt"
PLUGMD5=${XSW_PLUGMD5:-9823cdde}''')
rep('''echo "$IMGLITS12" | sed 's/^/9/'; } | timeout 120''', '''echo "$IMGLITS12" | sed 's/^/9/'; echo "$IMGLITSW" | sed 's/^/9/'; } | timeout 120''')
rep('''xlog "VIRHE $IMAGE is not the int12 image -- missing:''', '''xlog "VIRHE $IMAGE is not the int12 + ds41-swl image -- missing:''')
rep(''' + $(echo "$IMGLITS12" | grep -c .) int12 literals''', ''' + $(echo "$IMGLITS12" | grep -c .) int12 + $(echo "$IMGLITSW" | grep -c .) ds41-swl literals''')

# ---- helpers: loadid's selftest is already run; swlsum's selftest under R12T; FNS; the arm pins checked; md5 line; the dry plan
rep('''FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj''', '''FNS="swlwant NAMEOF0 swlset c8b swlstag swlplan c8split log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj''')
rep(''' armenv ARMDESC armback resset c8mixset rstart ngclean tidy c16sum"''', ''' armenv ARMDESC armback c8mixset rstart ngclean tidy c16sum"''')
rep('''[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"''',
    '''python3 -c 'import json,sys
sw = sys.argv[1] == "1"; o = json.loads(sys.argv[2]); arms = dict(zip(("B", "L1", "L1G075", "L1G100"), (json.loads(x) for x in sys.argv[3:7])))
want = {"B": (0, 150), "L1": (1, 150), "L1G075": (1, 75), "L1G100": (1, 100)}
for a, p in arms.items():
    rest = {k: v for k, v in p.items() if k not in ("place_swap_lent", "heat_min_gain_x100")}
    oo = {k: v for k, v in o.items() if k not in ("place_swap_lent", "heat_min_gain_x100")}
    assert rest == oo, a
    if sw: assert (p["place_swap_lent"], p["heat_min_gain_x100"]) == want[a] and p["heat_hyst_x100"] == 125, a
    else: assert p == o, a
if sw: assert o["place_swap_lent"] == 0 and o["heat_min_gain_x100"] == 150 and o["heat_hyst_x100"] == 125
else: assert not any(k in o for k in ("place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))' $SW "$(pinj O)" "$(pinj B)" "$(pinj L1)" "$(pinj L1G075)" "$(pinj L1G100)" 2>/dev/null || HMISS="$HMISS json:B/L1/L1G075/L1G100(the SWL / L2 switches and nothing else vs O)"
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"''')
rep('''HMD5="res12/resprobe.py $(md5sum < $SPB | cut -c1-8) (stepprobe.py $(md5sum < $SPB0 | cut -c1-8) + the answer), res12/ressum.py $(md5sum < $RSUM | cut -c1-8),''',
    '''HMD5="yswl/resprobe.py $(md5sum < $SPB | cut -c1-8) (= res12's; stepprobe.py $(md5sum < $SPB0 | cut -c1-8) + the answer), yswl/swlsum.py $(md5sum < $RSUM | cut -c1-8),''')
rep('''dsm-res12.sh $(md5sum < $0 | cut -c1-8) (from prof12/dsm-prof1.sh f64daf3a)''', '''dsm-yswl.sh $(md5sum < $0 | cut -c1-8) (from res12/dsm-res12.sh 08873cb8)''')
rep('''| ressum.py $R12T |''', '''| swlsum.py $R12T |''')
rep('''  echo "+ arm LF1 ($(NAMEOF LF1)): $(armj LF1) -> pin $(pinj LF1 | python3 -c 'import json,sys; print("place_lf", json.load(sys.stdin)["place_lf"])')"''',
    '''  for a in B L1 L1G075 L1G100; do echo "+ in-process arm $(NAMEOF $a): $(armj $a) -> pin $(pinj $a | python3 -c 'import json,sys; d = json.load(sys.stdin); print("place_swap_lent %s heat_min_gain_x100 %s heat_hyst_x100 %s place_lf %s" % (d.get("place_swap_lent"), d.get("heat_min_gain_x100"), d.get("heat_hyst_x100"), d.get("place_lf")))')"; done''')
rep('''  echo "+ a read set: resprobe code / essay $EGEN T 0: code W, essay W, code, essay, code, essay, essay, essay | mb c8 one block ${C8WIN} s (harness/ncmb3.py fixed prompt set, reset per block) |''',
    '''  echo "+ the plan (one start): $([ $STAG -ge 1 ] && echo "staggered B (a1B alone + loadid conc $LDCONC, pf $LDPF, vs a1B strict), ")B, L1$([ $ABBA = 1 ] && echo ", c8 L1, c8 B (ABBA)"), L1G075$([ $STAG = 2 ] && echo ", staggered L1 (budget)")$([ $G100 = 1 ] && echo ", L1G100 (budget)") | an arm: mb c8 one block ${C8WIN} s FIRST (harness/ncmb3.py fixed prompt set, reset per block) + its P12SPLIT, then resprobe $EGEN T 0 code, essay, code, essay (switched), essay (warm) |''')

# ---- the main start line
rep('''| SWL SCREEN: arms [$ARMS] (one FRESH start each on ngshare, n-gram reuse; BASE runs place_lf [$(echo $BASELF | sed 's/./& /g' | xargs)] in-process), ref12's pins: $DESC | MAXSEQS ${XSW_MAXSEQS:-the base value} | a read set: code W, essay W, code, essay, code, essay (switched), essay, essay (warm), resprobe $EGEN T 0 + one c8 block (fixed harness, ${C8WIN} s) | identity: every read's answer across arms / sets | budget $BUDGET min | no evict | the n-gram tables cleaned at the end"''',
    '''| SWL SCREEN (int12 + ds41-swl): ONE FRESH start [$ARMS] on ngshare, the arms in-process by fn4: $([ $STAG -ge 1 ] && echo "staggered B, ")B, L1$([ $ABBA = 1 ] && echo ", c8 L1 + c8 B (ABBA)"), L1G075$([ $STAG = 2 ] && echo ", staggered L1")$([ $G100 = 1 ] && echo ", L1G100") (the last two on budget) | ref12's pins: $DESC | MAXSEQS ${XSW_MAXSEQS:-the base value} | an arm: one c8 block FIRST (fixed harness, ${C8WIN} s, + P12SPLIT), then code, essay, code, essay (switched), essay (warm), resprobe $EGEN T 0 | identity: every read's answer across arms, strict; staggered ld vs a1 strict | budget $BUDGET min | no evict | the n-gram tables cleaned at the end"''')

# ---- int12's registry-tail literal: the swl registry continues after exl3_x3d (its own tail is in IMGLITSW)
rep('''IMGLITS12="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, epoch_bump)''', '''IMGLITS12="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, ''')

open(dst, "w").write(s)
print("edits", n_edits, "->", dst, len(s.splitlines()), "lines")
