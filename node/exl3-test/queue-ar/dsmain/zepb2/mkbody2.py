#!/usr/bin/env python3
"""mkbody2.py: zepb2/dsm-zepb2.sh from a COPY of zepb/dsm-zepb.sh (the zepb item's body); every edit asserted."""
import re, sys

s = open(sys.argv[1]).read()            # zepb/dsm-zepb.sh
NEWF = open(sys.argv[2]).read()          # newf2.sh
out = sys.argv[3]


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "expected %d of %r, found %d" % (count, old[:100], n)
    s = s.replace(old, new)


def func(name):
    m = re.search(r"(?m)^%s\(\) \{.*?^\}\n" % re.escape(name), s, re.S)
    assert m, name
    return m.group(0)


# ---- header
h0 = s.index("# dsm-zepb.sh"); h1 = s.index("set -u\n")
HEADER = '''# dsm-zepb2.sh (12.11.2026, queue-ar/dsmain/zepb2, the EPB builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepbz2.sh. Log tag "RESULT xep2 ", out runs/xdsm/epb2/s1 and /s2 (XE2_OUT). A COPY of
# zepb/dsm-zepb.sh (7994c699; tag xepb -> xep2, knobs XEP_ -> XE2_; the original not changed): the SAME image, start, pins, guards,
# image checks and harness. WHY (coordinator 12.11.): zepb's O arm with the hot-8 extras declared (idle) read c8 128.7 tok/s at
# 56.1 ms a step vs prof1's 184.0 / 36.2 -- declaring the extras cost residency (zepb's start: slab 2117 vs 1849 slots, flex 7.8 vs
# 10.4 GB a rank), so zepb's S / E4 cannot be read against int12. LEVER EPB, MEASURE ONLY (user 12.11.); nothing into a base.
# START 1 (NO extras): RADIANCE_DS41_SLOTSUM=16, no placement file (table 0 only: the contiguous split, zero replicas, no extra
#   expert declared), SLOTSUM_ON=0. RESIDENT-UNIT CHECK vs prof1 (per rank GET /stats experts.ranks vram / flex_bytes + the start log's
#   "mover: N slab slots"; lower by > 1 % = VIRHE). Arms in process: O = ds41_slotsum 0 (today), S = ds41_slotsum 1 (the slot-order
#   sum alone) = the pure sum's cost / gain vs int12. P0 (4-card selftest) -> P0b S strict (a1 = a2, gs16 = a1, gs8 = a1, O a1 =
#   ref12's a1, S a1 != O a1) -> c8 O S S O (ncmb3 fixed, 35 s, P12SPLIT) -> c16 fixed-harness newcomer block O, S (ncmb3 1x16, cap
#   48, 35 s, P12SPLIT) -> 1u essay O, S (one discarded + 3 back to back each) -> staggered arrivals under S vs S's a1 (strict).
# INCIDENT GUARD (zepb c8-5-S-S, 12.11.): ncmb3 runs with --max-wait XE2_MAXWAIT 8 (a request that streams no text can no longer hold
#   the fixed schedule 120 s and collapse the time window); every c8 / c16 block is guarded (requests > 0, window >= 90 %, else VIRHE
#   and that arm's remaining blocks skipped); zepb2/streamdiag.py re-sends that incident's two requests under O, S (and E4), streamed
#   and non-streamed (INFO: what the server sent).
# START 2 (hot-4 extras only; skipped and logged when the budget does not allow): RADIANCE_DS41_EP_PLACE = zepb/place/hot4.place
#   (table 1). Resident units printed vs prof1. Arms O / E4 (= ds41_slotsum 1, ds41_place 1 here). P0b E4 (gs16 = a1, O a1 =
#   ref12's a1, E4 a1 != O a1, E4 a1 = START 1's S a1: placement invariance, strict) -> c8 O E4 E4 O -> 1u essay E4, O.
# No profiled start (the zepb item has one). BUDGET XE2_BUDGET_MIN 40 (start 1 ~22 min; start 2 runs if elapsed + ~15 <= budget).
# MOCK: XE2_SW=0 (queue-ar/tools/mockrad). DRYRUN=1 bash dsm-zepb2.sh : prints every command, writes nothing.
'''
s = s[:h0] + HEADER + s[h1:]

# ---- renames
s = s.replace("xepb", "xep2").replace("XEP_", "XE2_")
rep("O=${XE2_OUT:-$R/xdsm/epb}; export STATSDIR=$O/stats", "O=${XE2_OUT:-$R/xdsm/epb2}; export STATSDIR=$O/stats")
rep("ITEM=${XE2_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb}", "ITEM=${XE2_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepbz2}")
rep("BUDGET=${XE2_BUDGET_MIN:-55}", "BUDGET=${XE2_BUDGET_MIN:-40}")

# ---- knobs: one placement file (start 2), per-start orders
rep("SLROWS=${XE2_SLOTSUM:-16}; PLDIR=$HD/place; PL1=${XE2_PLACE1:-hot8.place}; PL2=${XE2_PLACE2:-hot4.place}",
    "SLROWS=${XE2_SLOTSUM:-16}; PLDIR=$HD/../zepb/place; PL1=${XE2_PLACE1:-hot4.place}   # zepb's hot-4 file, read in place (START 2's table 1)")
rep("C8ORDER=${XE2_C8ORDER-O E E O S E4}; U1ARMS=${XE2_U1ARMS-E O}; C16ARMS=${XE2_C16ARMS-E O}; C16NC=${XE2_C16NC:-1}; PPLAN=${XE2_PPLAN-E}",
    """C8ORDER1=${XE2_C8ORDER1-O S S O}; C16ARMS1=${XE2_C16ARMS1-O S}; U1ARMS1=${XE2_U1ARMS1-O S}
C8ORDER2=${XE2_C8ORDER2-O E4 E4 O}; U1ARMS2=${XE2_U1ARMS2-E4 O}; DOS2=${XE2_START2:-1}; S2MIN=${XE2_S2MIN:-15}
C8ORDER=$C8ORDER1; U1ARMS=$U1ARMS1; C16ARMS=""; C16NC=0; PPLAN=""; STARTN=1; NTAB=1   # zepb's names, set per start below""")
rep("DOPROF=${XE2_PROF:-1}", "DOPROF=0")

# ---- LEVENV: the placement words per start (start 1 none; start 2 table 1 = hot4)
rep('''  RADIANCE_DS41_SLOTSUM=$SLROWS RADIANCE_DS41_SLOTSUM_ON=0 RADIANCE_DS41_EP_PLACE=/stats/place/$PL1 RADIANCE_DS41_EP_PLACE2=/stats/place/$PL2
  RADIANCE_DS41_EP_PLACE_TAB=0"''', '''  RADIANCE_DS41_SLOTSUM=$SLROWS RADIANCE_DS41_SLOTSUM_ON=0 RADIANCE_DS41_EP_PLACE_TAB=0"''')
rep('''LEVENV=$(echo $LEVENV)\n''', '''LEVENV=$(echo $LEVENV)
LEVENV1=$LEVENV; LEVENV2="$LEVENV RADIANCE_DS41_EP_PLACE=/stats/place/$PL1"   # zepb2: START 1 no placement file, START 2 hot-4 as table 1
''')

# ---- arms
rep('''    E) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 1}' || echo '{}' ;;    # lever EPB: slot-order sum + table 1
    E4) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 2}' || echo '{}' ;;   # + table 2 (INFO)
''', '''    E4) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 1}' || echo '{}' ;;   # zepb2 START 2: the sum + table 1 = hot-4
''')
rep('''NAMEOF() { case $1 in O) echo "O (today's MoE path, ds41_slotsum 0; the extras declared)" ;; E) echo "E (EPB: ds41_slotsum 1, ds41_place 1 = $PL1)" ;;
  E4) echo "E4 (ds41_slotsum 1, ds41_place 2 = $PL2)" ;;''', '''NAMEOF() { case $1 in O) echo "O (today's MoE path, ds41_slotsum 0$([ $STARTN = 2 ] && echo "; the hot-4 extras declared, idle"))" ;;
  E4) echo "E4 (ds41_slotsum 1, ds41_place 1 = $PL1: the sum + hot-4 replication)" ;;''')

# ---- swnote: the past-the-end table is NTAB (1 at start 1, 2 at start 2)
rep('''c11=$(fn4code '{"ds41_place": 3}'); fi''', '''c11=$(fn4code "{\\"ds41_place\\": $NTAB}"); fi''')
rep('''ds41_slotsum 2 -> HTTP $c10, ds41_place 3 -> HTTP $c11 (want 400 / 400: 0 / 1 and tables 0..2 only)"''',
    '''ds41_slotsum 2 -> HTTP $c10, ds41_place $NTAB -> HTTP $c11 (want 400 / 400: 0 / 1 and tables 0..$((NTAB - 1)) only)"''')

# ---- epbnote: the start's tables
rep('''  case "$n" in *"MoE passes of up to $SLROWS rows"*"40 of 40 trunk layers"*"1 /stats/place/$PL1 (40 layers placed"*"2 /stats/place/$PL2 (40 layers placed"*"switches ds41_slotsum 0, ds41_place 0"*) ;; *) ok=0 ;; esac''',
    '''  local tpat="tables: 0 contiguous; extra experts a rank 0..0 over the layers"; [ $STARTN = 2 ] && tpat="1 /stats/place/$PL1 (40 layers placed"
  case "$n" in *"MoE passes of up to $SLROWS rows"*"40 of 40 trunk layers"*"$tpat"*"switches ds41_slotsum 0, ds41_place 0"*) ;; *) ok=0 ;; esac''')
rep('''tables: 0 contiguous, 1 /stats/place/$PL1 (40 layers placed ..), 2 /stats/place/$PL2 (..) ... switches ds41_slotsum 0, ds41_place 0'; no 'did not resolve'; VRAM at start vs prof1 ($P1STATS)"''',
    '''START 1 'tables: 0 contiguous; extra experts a rank 0..0', START 2 '1 /stats/place/$PL1 (40 layers placed ..)' ... switches ds41_slotsum 0, ds41_place 0'; no 'did not resolve'; VRAM at start vs prof1 ($P1STATS)"''')

# ---- dstart: per-start tag, P0 at start 1 only, the resident check, the readings by start
ds = func("dstart")
dn = ds.replace("  local tag=xep2-n5\n", "  local tag=xep2-s$STARTN\n")
dn = dn.replace("  p0epb || { STOPALL=1; return 1; }\n", "  if [ $STARTN = 1 ]; then p0epb || { STOPALL=1; return 1; }; fi\n")
dn = dn.replace('  run "cp $PLDIR/$PL1 $PLDIR/$PL2 $STATSDIR/place/"\n', '  run "cp $PLDIR/$PL1 $STATSDIR/place/"\n')
dn = dn.replace('''  epbnote $tag || { xlog "$tag: VIRHE lever EPB is not declared as asked -> stopped"; keeplog $tag-epb; down; STOPALL=1; return 1; }\n''',
                '''  epbnote $tag || { xlog "$tag: VIRHE lever EPB is not declared as asked -> stopped"; keeplog $tag-epb; down; STOPALL=1; return 1; }
  rescheck $tag $([ $STARTN = 1 ] && echo 1 || echo 0)
''')
dn = dn.replace("  stopfile || epbreads\n", "  if [ $STARTN = 1 ]; then stopfile || reads1; else stopfile || reads2; fi\n")
dn = dn.replace("start (START A, unprofiled; lever EPB declared: RADIANCE_DS41_SLOTSUM=$SLROWS, tables $PL1 / $PL2, ON=0; $DESC)",
                '''start (START $STARTN, unprofiled; lever EPB declared: RADIANCE_DS41_SLOTSUM=$SLROWS, $([ $STARTN = 1 ] && echo 'NO placement file (no extras)' || echo "table 1 = $PL1 (hot-4 extras)"), ON=0; $DESC)''')
dn = dn.replace("EPB readings: P0b determinism / BI, c8 [$C8ORDER], c16 [$C16ARMS] + newcomer E, 1u [$U1ARMS], staggered E, vs-int12 INFO",
                '''zepb2 START $STARTN readings: $([ $STARTN = 1 ] && echo "P0b S, c8 [$C8ORDER1], c16 newcomer [$C16ARMS1], 1u [$U1ARMS1], staggered S" || echo "P0b E4 (+ vs start 1's S), c8 [$C8ORDER2], 1u [$U1ARMS2]")''')
assert dn.count("reads1") == 1 and dn.count("rescheck") == 1 and "xep2-s$STARTN" in dn
s = s.replace(ds, dn)

# ---- summary: the start's engine log; the price line's words
rep("  local EL=$O/xep2-n5-engine-full.log.gz w f nm l a W=() ref", "  local EL=$O/xep2-s$STARTN-engine-full.log.gz w f nm l a W=() ref")
rep('''(the declaration'"'"'s price: extras + slot ops declared, idle under O)''', '''(the price of this start'"'"'s declaration, idle under O)''')

# ---- summary: a block whose window collapsed (< 1 s; the block guard VIRHEs it) is not split
rep('''    [ -n "$w" ] || { xlog "summary: $nm -- no window (no marks)"; continue; }''', '''    [ -n "$w" ] || { xlog "summary: $nm -- no window (no marks)"; continue; }
    awk -v w=$w 'BEGIN{split(w, x, ":"); exit !(x[2] - x[1] < 1)}' && { xlog "summary: $nm -- its window is under 1 s (the block guard's case): not split"; continue; }''')
# ---- the DRYRUN / helper checks
rep('''$OPNC $PLDIR/$PL1 $PLDIR/$PL2; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done''', '''$OPNC $PLDIR/$PL1 $P1DIR/xp1-n5-engine-full.log.gz $P1STATS $SDG; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
SDT=$(timeout 60 python3 -I -B $SDG --mb $MBP --mbt3 $MBT2 --selftest 2>&1 | tail -1); case "$SDT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SDG(${SDT:0:300})" ;; esac''')
rep('''    if not ok: print("place:%s" % f)' $PLDIR/$PL1 $PLDIR/$PL2 2>&1''', '''    if not ok: print("place:%s" % f)' $PLDIR/$PL1 2>&1''')
rep('''HMD5="zepb/place/$PL1 $(md5sum < $PLDIR/$PL1 | cut -c1-8), zepb/place/$PL2 $(md5sum < $PLDIR/$PL2 | cut -c1-8), ''', '''HMD5="zepb/place/$PL1 $(md5sum < $PLDIR/$PL1 | cut -c1-8), ''')
rep('''for a in O M0 E E4 S; do python3''', '''for a in O M0 E4 S; do python3''')
rep('''{"O": (0, 0), "M0": (0, 0), "E": (1, 1), "E4": (1, 2), "S": (1, 0)}[a]''', '''{"O": (0, 0), "M0": (0, 0), "E4": (1, 1), "S": (1, 0)}[a]''')
rep('''  for a in O E S E4; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done''', '''  for a in O S E4; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done''')
m = re.search(r'(?m)^  echo "\+ readings \(START A\):.*$', s)
s = s[:m.start()] + '''  echo "+ readings: START 1 (no extras; resident units vs prof1, VIRHE if lower by > 1 %): P0 4-card selftest [$P0CASES] -> P0b S (a1 / a2 / gs$GSC / gs8 + O a1, strict) -> c8 [$C8ORDER1] ${C8WIN} s (P12SPLIT each) -> c16 newcomer blocks [$C16ARMS1] -> 1u essay $EGEN [$U1ARMS1] (W + $NREADS) -> staggered S | START 2 (hot-4, $PL1; if elapsed + $S2MIN <= $BUDGET min): P0b E4 (gs$GSC, O a1, vs start 1's a1S) -> c8 [$C8ORDER2] -> 1u [$U1ARMS2] | no profiled start"''' + s[m.end():]
fns = re.search(r'FNS="log run mkd [^"]*"', s).group(0)
s = s.replace(fns, fns[:-1] + ' rescheck p0idx stagx reads1 reads2 mbg sdiag"')

# ---- mbx: a bounded first-token wait (the zepb incident: --max-wait 120 let one silent request collapse the time window)
rep(" --ignore-eos 1 --prompt-set fixed --reset-per-block 1 > $base.txt 2>&1)", " --ignore-eos 1 --prompt-set fixed --reset-per-block 1 --max-wait $MAXW > $base.txt 2>&1)")
rep("P0CASES=${XE2_P0CASES-", "MAXW=${XE2_MAXWAIT:-8}; SDG=$HD/streamdiag.py; ARMDEAD=\"\"\nP0CASES=${XE2_P0CASES-")
# ---- the new functions (before the trap)
rep("trap 'down' EXIT\n", NEWF + "trap 'down' EXIT\n")

# ---- main: START 1, its summary; START 2 (budget), its summary
old_main = s[s.index("ODIR=$O\nstopfile || dstart\n"):s.index("unset $SWITCHES\ndown\n")]
new_main = '''ODIR=$O
O=$ODIR/s1; export STATSDIR=$O/stats; mkd $O $O/guard $O/c8 $O/snap $STATSDIR; [ "$DRY" = 1 ] || : > $O/identity.txt
STARTN=1; NTAB=1; LEVENV=$LEVENV1; C8ORDER=$C8ORDER1; U1ARMS=$U1ARMS1
stopfile || dstart
summary
if [ $STOPALL = 1 ]; then xlog "$ITEM: STOPPED after START 1 (P0 / the EPB declaration / START 1's P0b failed): no START 2"
elif [ $DOS2 = 1 ] && ! stopfile; then
  if budok "START 2 (hot-4 extras: start + P0b E4 + c8 [$C8ORDER2] + 1u [$U1ARMS2])" $S2MIN; then
    O=$ODIR/s2; export STATSDIR=$O/stats; mkd $O $O/guard $O/c8 $O/snap $STATSDIR; [ "$DRY" = 1 ] || : > $O/identity.txt
    STARTN=2; NTAB=2; LEVENV=$LEVENV2; C8ORDER=$C8ORDER2; U1ARMS=$U1ARMS2; NB=0; ARMDEAD=""
    dstart
    summary
  fi
fi
'''
s = s.replace(old_main, new_main)
rep('''mkd $O $O/guard $O/c8 $O/snap $STATSDIR\n[ "$DRY" = 1 ] || : > $O/identity.txt\nODIR=$O''', '''mkd $O\nODIR=$O''')
assert "dstartp" not in new_main
open(out, "w").write(s)
print("wrote", out, len(s), "bytes,", s.count("\n"), "lines")
