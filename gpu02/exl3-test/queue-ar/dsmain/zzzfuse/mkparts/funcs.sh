FUCHKN=0; FUCHKLOG=""   # the fusion switches read back: checks done / the last section's arms
fusewant() {   # fusewant ARM -> "RF MF RC": the ds41_rfuse / ds41_mix_fin / route_copy_ov values the arm runs (B 0 0 0, F the ALL set, Y every
               # kept form on (identity), R1 1 0 0, R2 2 0 0, M 0 1 0, R 0 0 1; O = B)
  case $1 in F) echo "$FRF $FMF $FRC" ;; Y) echo "$YRF $YMF 1" ;; R1) echo "1 0 0" ;; R2) echo "2 0 0" ;; M) echo "0 1 0" ;; R) echo "0 0 1" ;; *) echo "0 0 0" ;; esac
}
fusechk() {   # fusechk ARM: the fusion switches READ BACK right after arm() applied ARM -- GET /admin/fn4 switches.ds41_rfuse / ds41_mix_fin / route_copy_ov = the arm's
              # (fusewant) -> 0 (logged per section); else VIRHE -> 1 (the block is not read); ZFU_SW=0 (mock: not posted): INFO, 0
  local want got
  want=$(fusewant $1)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: switches ds41_rfuse / ds41_mix_fin / route_copy_ov == $want under arm $1 + exl3_x3d $XB (asserted before the block; else VIRHE: the block is not read)"; return 0; }
  got=$(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin).get("switches") or {}; print(w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d"))' 2>/dev/null)
  FUCHKN=$((FUCHKN + 1)); FUCHKLOG="$FUCHKLOG $1:$(echo ${got:-none} | tr ' ' '/')"
  echo "$(date '+%F %T') arm $1 ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d ${got:-none} (want $want $XB)" >> $O/fuse-readback.txt
  [ "$got" = "$want $XB" ] && return 0
  [ $SW = 0 ] && return 0
  xlog "arm $1: VIRHE GET /admin/fn4 ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d ${got:-none} after the arm was applied (want $want $XB) -- this block is not read"; ARMOK=0; return 1
}
fusesay() {   # fusesay SECTION: one line -- the switch values read back before the section's blocks (then the list restarts)
  [ "$DRY" = 1 ] && return 0
  xlog "$1: ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d read back before each block (GET /admin/fn4, arm:values):$FUCHKLOG$([ $SW = 0 ] && echo " (INFO: ZFU_SW=0 mock, not posted)") | $FUCHKN check(s) so far, $O/fuse-readback.txt"
  FUCHKLOG=""
}
FRF=2; FMF=${ZFU_ALLMIX:-0}; FRC=${ZFU_ALLRCO:-1}; YRF=2; YMF=1; R1ON=1; R2ON=1; MON=1; RCON=1; R1WHY=""; R2WHY=""; MWHY=""; P0HALT=0; P0MARGIN=${ZFU_P0MARGIN_US:-1.0}; P0STOP=${ZFU_P0STOP:-1}; P0WHY=""   # the ALL arm F's values, the identity arm Y's, and the arms P0 keeps (p0fuse)
p0fuse() {   # P0 (ONE card by chip UUID P0UUID, no server, BEFORE the start): r4d_selftest --case rfuse and --case mixfin from the image -- every
             # fused form byte for byte vs the stock launches (FAIL lines) + the UNSTAMPED per-call times (FUSEPERF: the stock launches back to back
             # on one stream vs the fused launch, warm and cold, M 1 / 8 / 16 / 48). A form whose bytes differ: VIRHE, DROPPED (exact = byte-identical:
             # NEEDS USER); a form that does not beat the pair by >= ZFU_P0MARGIN_US 1.0 us at M 8 (warm and cold) or is slower cold at M 1 / 16:
             # DROPPED (no arm, out of F / Y). No form left (ZFU_P0STOP 1): the item STOPS after P0, no start. F = ds41_rfuse 2 (else 1, else 0) +
             # route_copy_ov ZFU_ALLRCO 1 + ds41_mix_fin ZFU_ALLMIX (default 0: FUSE-M is predicted to LOSE in the engine, module header).
  local c rc nf g k v drop
  for c in rfuse mixfin; do
    run "gpurun xzfuse-p0-$c-$$ $P0SEC $DEV1 --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/p0-$c.txt 2>&1"; rc=$?
    if [ "$DRY" = 1 ]; then
      echo "+ RESULT xzfuse P0 $c (one card $P0UUID, bounded $P0SEC s): rc 0, 0 FAIL, the ok lines, FUSEPERF (warm / cold us a call, chain vs fused) and FUSEGATE KEEP / DROP lines -> the arms kept"
      continue
    fi
    nf=$(grep -c '^  FAIL' $O/p0-$c.txt); k=$(grep -c '^  ok ' $O/p0-$c.txt)
    xlog "P0 $c (r4d_selftest --case $c, one card $P0UUID): rc $rc, $k ok, $nf FAIL$( [ $nf -gt 0 ] && echo " -- $(grep -m3 '^  FAIL' $O/p0-$c.txt | cut -c1-300 | tr '\n' ';')")"
    while IFS= read -r l; do xlog "P0 ${l#  }"; done < <(grep -E '^  (FUSEPERF|FUSEGATE|info )' $O/p0-$c.txt | head -24 | cut -c1-500)
    if [ $rc != 0 ] || [ $nf -gt 0 ] || [ $k = 0 ]; then
      if [ $SW = 0 ]; then xlog "P0 $c: failed under ZFU_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"; continue; fi
      case $c in rfuse) R1ON=0; R2ON=0; R1WHY="P0 bytes / rc: not byte-identical or the case did not run"; R2WHY=$R1WHY ;; mixfin) MON=0; MWHY="P0 bytes / rc: not byte-identical or the case did not run" ;; esac
      P0WHY="$P0WHY $c:bytes/rc"
      xlog "P0 $c: VIRHE the fused forms are not byte-identical to the stock launches (or the case did not run: rc $rc) -> $([ $c = rfuse ] && echo "arms R1 / R2 DROPPED, F without ds41_rfuse" || echo "arm M DROPPED, F without ds41_mix_fin") (exact = byte-identical: NEEDS USER)"
    fi
  done
  [ "$DRY" = 1 ] && return 0
  # THE GATE (ds-main-port 13.11. calibration from zzzlaunch: the stamped gap table overstates real launch time ~5x; a real dependent gap is
  # ~1 us, so no ~4 us is credited a removed launch): a form is KEPT only when its UNSTAMPED per-call time beats the stock launches run
  # back to back on one stream (r4d_selftest FUSEPERF, chain vs fused) by >= P0MARGIN us at M 8 (the c8 height) both warm and cold, and is
  # not slower cold at M 1 and M 16; else DROPPED -- no arm, out of F and Y (the selftest's own FUSEGATE line is INFO only)
  local gl gv g1 g2 gm
  gl=$(python3 -c 'import re,sys
mg = float(sys.argv[3])
def rows(path, kind):
    out = {}
    try:
        for l in open(path):
            m = re.search(r"FUSEPERF %s M=(\d+) (.*)" % kind, l)
            if m: out[int(m.group(1))] = [float(x) for x in re.findall(r"warm ([0-9.]+) cold ([0-9.]+)", m.group(2)) for x in x]
    except OSError: pass
    return out
def gate(t, fi):
    """t: {M: [w0, c0, w1, c1, ...]}, fused pair index fi -> (keep, text)"""
    txt, ok = [], True
    for M in (1, 8, 16, 48):
        v = t.get(M)
        if not v or len(v) < 2 * fi + 2:
            if M in (1, 8, 16): ok = False
            txt.append("M%d n/a" % M); continue
        w0, c0, wf, cf = v[0], v[1], v[2 * fi], v[2 * fi + 1]
        txt.append("M%d pair %.2f/%.2f fused %.2f/%.2f us (warm/cold, %+.2f/%+.2f)" % (M, w0, c0, wf, cf, wf - w0, cf - c0))
        if M == 8 and not (w0 - wf >= mg and c0 - cf >= mg): ok = False
        if M in (1, 16) and cf > c0: ok = False
    return ok, "; ".join(txt)
r, m = rows(sys.argv[1], "rfuse"), rows(sys.argv[2], "mixfin")
k1, t1 = gate(r, 1); k2, t2 = gate(r, 2); km, tm = gate(m, 1)
print("%d %d %d | form1 %s %s | form2 %s %s | mixfin %s %s" % (k1, k2, km, "KEEP" if k1 else "DROP", t1, "KEEP" if k2 else "DROP", t2, "KEEP" if km else "DROP", tm))' $O/p0-rfuse.txt $O/p0-mixfin.txt $P0MARGIN 2>&1)
  read -r g1 g2 gm _ <<< "$gl"
  xlog "P0 GATE (unstamped per-call time, the fused form must beat the stock launches back to back by >= $P0MARGIN us at M 8 warm AND cold, not slower cold at M 1 / 16): ${gl#* * * | }"
  if [ $SW = 0 ]; then xlog "P0 GATE: INFO only under ZFU_SW=0 (the mock runs no selftest) -- every arm kept"; g1=1; g2=1; gm=1; fi
  case "$g1$g2$gm" in [01][01][01]) ;; *) g1=0; g2=0; gm=0; xlog "P0 GATE: VIRHE the FUSEPERF lines did not parse ($gl) -> every form DROPPED" ;; esac
  [ "$g1" = 1 ] || { [ $R1ON = 1 ] && R1WHY="P0 per-call speed gate (bytes OK): not >= $P0MARGIN us faster than the pair at M 8 warm + cold, or slower cold at M 1 / 16"; R1ON=0; P0WHY="$P0WHY R1:gate"; }
  [ "$g2" = 1 ] || { [ $R2ON = 1 ] && R2WHY="P0 per-call speed gate (bytes OK): not >= $P0MARGIN us faster than the pair at M 8 warm + cold, or slower cold at M 1 / 16"; R2ON=0; P0WHY="$P0WHY R2:gate"; }
  [ "$gm" = 1 ] || { [ $MON = 1 ] && MWHY="P0 per-call speed gate (bytes OK): not >= $P0MARGIN us faster than the pair at M 8 warm + cold, or slower cold at M 1 / 16"; MON=0; P0WHY="$P0WHY M:gate"; }
  [ $R2ON = 1 ] || { [ $FRF = 2 ] && FRF=$([ $R1ON = 1 ] && echo 1 || echo 0); }
  [ $R1ON = 1 ] || { [ $FRF = 1 ] && FRF=0; }
  [ $MON = 1 ] || FMF=0
  YRF=$([ $R2ON = 1 ] && echo 2 || { [ $R1ON = 1 ] && echo 1 || echo 0; }); YMF=$MON   # identity: every byte-checked form on
  if [ $R1ON = 0 ] && [ $R2ON = 0 ] && [ $MON = 0 ] && [ "$P0STOP" = 1 ]; then
    P0HALT=1; xlog "P0 GATE: NO fusion form clears the bar -> the item STOPS here (no start; RCO alone was read by zzzlaunch) -- the FUSEPERF numbers above are the reading"
  fi
  [ "${ZFU_P0ONLY:-0}" = 1 ] && { P0HALT=1; xlog "P0: ZFU_P0ONLY=1 -> the item stops after P0 (a kbench-only run)"; }
  gzip -f $O/p0-rfuse.txt $O/p0-mixfin.txt 2>/dev/null || true
  xlog "P0: arms kept R1 $R1ON R2 $R2ON M $MON (R: RCO is launch mechanics, no kernel: no P0) | F (ALL) = ds41_rfuse $FRF + ds41_mix_fin $FMF + route_copy_ov $FRC | Y (identity, every exact form on) = ds41_rfuse $YRF + ds41_mix_fin $YMF + route_copy_ov 1${P0WHY:+ (dropped:$P0WHY)}"
  return 0
}
fuseid() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header) under B and the ALL-fusions-on arm Y (every form P0 kept: ds41_rfuse 2 +
             # ds41_mix_fin 1, whatever F's speed choice): every check runs (each failure named), then fusereads stops on IDFAIL
  local a g u0 u1
  idok n5   # the warm-up answers vs stock (identw at the start; no line = not compared = the stop)
  for a in $IDARMS; do
    stopfile && return 0
    alone xzfuse $a a1-$a
    if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx xzfuse a1-$a-vs-ref12-a1 $REF12A1 $O/xzfuse-a1-$a.jsonl; idok a1-$a-vs-ref12-a1
    else xlog "identity a1-$a: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-$a-reference-missing"; fi
  done
  if [ -s "$YZA1" ] || [ "$DRY" = 1 ]; then cmpx xzfuse a1-B-vs-zzk25s-a1-Z $YZA1 $O/xzfuse-a1-B.jsonl; idok a1-B-vs-zzk25s-a1-Z
  else xlog "identity a1-B vs zzk25s a1-Z (int15): NOTE $YZA1 missing -- not compared (ref12's a1 is the reference)"; fi
  stopfile && return 0
  u0=$(utcnow); ssnap xzfuse-gs$GSC-$GSA-0; gsx $GSA; ssnap xzfuse-gs$GSC-$GSA-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-$GSA.win
  g=$(gt48 xzfuse-gs$GSC-$GSA)
  xlog "gs$GSC-$GSA cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd xzfuse-gs$GSC-$GSA-0 xzfuse-gs$GSC-$GSA-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; }) | fuse counters over gs16: $(fusectr xzfuse-gs$GSC-$GSA-0 xzfuse-gs$GSC-$GSA-1)"
  cmpx xzfuse gs$GSC-$GSA-vs-a1-$GSA $O/xzfuse-a1-$GSA.jsonl $O/xzfuse-gs$GSC-$GSA.jsonl; idok gs$GSC-$GSA-vs-a1-$GSA
  stopfile && return 0
  stagx $LDA $LDA || true
  fusesay "identity"
  return 0
}
fusectr() {   # fusectr A B -> one line: GET /admin/fn4 "fuse" / "rco" counter deltas between the snapshots fn-A.json and fn-B.json (engagement: a recorded pass counts at recording)
  [ "$DRY" = 1 ] && { echo "(dry: fuse / rco deltas)"; return 0; }
  python3 -c 'import json,sys
def g(p, o):
    try: return (json.load(open(p)).get(o) or {})
    except (OSError, ValueError): return {}
a, b = g(sys.argv[1], "fuse"), g(sys.argv[2], "fuse")
ra, rb = g(sys.argv[1], "rco"), g(sys.argv[2], "rco")
k = ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches", "mixfin_rows")
print("declared %s, layers %s | " % (b.get("rf_declared"), b.get("rf_layers")) + ", ".join("%s +%s" % (x, (b.get(x) or 0) - (a.get(x) or 0)) for x in k)
      + " | rco in_force %s, " % rb.get("in_force") + ", ".join("%s +%s" % (x, (rb.get(x) or 0) - (ra.get(x) or 0)) for x in ("recordings", "copies", "freed")))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo "fuse counters n/a"
}
fusereads() {   # the readings in their order (module header): 1 identity (THE STOP on any failure), 2 c8 B F F B, 3 the single fusions (one c8 block each),
                # 4 c16 B / F, 5 1u essay B then F
  local i a k
  if budok "identity (warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA)" 5; then fuseid || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | arms B $(armj B) Y $(armj Y) on the int15 pin | the failing comparisons are in $O/identity.txt and $O/xzfuse-*-cmp-*.json"
    return 0
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run | F (ALL) = $(armj F)"
  # 2 c8 WORK-BASED (TESTING.md 12.11. WINDOW QUANTUM): --window gens, in-process GORD (B F F B)
  i=0; for a in $GORD; do i=$((i + 1)); stopfile && return 0
    budok "c8 work-based block $i $a (gens $C8GN)" 2 || continue
    mbx 8 $i $a G || true
  done
  fusesay "c8"
  # 3 the single fusions: one c8 gens block each (an arm P0 dropped for bytes is not run)
  for a in $SINGLES; do i=$((i + 1)); stopfile && return 0
    case $a in R1) [ $R1ON = 1 ] || { xlog "single R1: SKIPPED (${R1WHY:-dropped before the start (FUSE-R form 1 not declared)})"; continue; } ;;
               R2) [ $R2ON = 1 ] || { xlog "single R2: SKIPPED (${R2WHY:-dropped before the start (FUSE-R form 2 not declared)})"; continue; } ;;
               M) [ $MON = 1 ] || { xlog "single M: SKIPPED (${MWHY:-dropped})"; continue; } ;; R) [ $RCON = 1 ] || { xlog "single R: SKIPPED (route_copy_ov not available)"; continue; } ;; esac
    budok "c8 single block $i $a (gens $C8GN)" 2 || continue
    mbx 8 $i $a G || true
  done
  fusesay "c8 singles"
  # 4 c16: one gens block an arm (K16A, B F)
  if [ $C16N -gt 0 ]; then
    xlog "c16: [$K16A] one gens block each (--gens $C16GN = $((16 * C16GN)) newcomer steps)"
    i=0; for a in $K16A; do i=$((i + 1)); stopfile && return 0
      budok "c16 gens block $i $a" 2 || continue
      mbx 16 $i $a G || true
    done
    fusesay "c16"
  fi
  # 5 1u essay (ESSAY RESIDENCY WARM-UP + REF12 lesson: the same prompt back to back): B = one discarded + U1N, then F = one discarded + U1N
  [ $ONEU = 1 ] || return 0
  SETN=U1
  for a in $U1A; do stopfile && return 0
    budok "1u essay $a (1 discarded + $U1N back to back)" 1 || continue
    blockx U1W $a essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 $a essay || true; done
  done
  fusesay "1u"
  return 0
}
ncmset() {   # ncmset LOG LABEL MIN REQS...: ncmatch.py over the blocks (NAME = REQS minus -reqs.jsonl, ARM = its last field) -> ncm-LABEL.json / .txt
  local el=$1 lb=$2 mn=$3 f nm bl="" l; shift 3
  for f in "$@"; do [ -s "$f" ] || continue; nm=$(basename $f -reqs.jsonl); bl="$bl --block $nm:${nm##*-}:$f"; done
  [ -n "$bl" ] || return 0
  timeout 300 python3 -I -B $NCM $el --set $lb $bl --min $mn --json $O/ncm-$lb.json > $O/ncm-$lb.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1600}"; done < <(grep -h '^NCMATCH' $O/ncm-$lb.txt 2>/dev/null)
  grep -q '^Traceback' $O/ncm-$lb.txt 2>/dev/null && xlog "summary: VIRHE ncmatch.py $lb: $(grep -A4 '^Traceback' $O/ncm-$lb.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
  return 0
}
stepsx() {   # stepsx PLAIN REQS: the coordinator's steps.py (zzzfuse/steps.py = zzk25s's, md5 STPMD5) over one gens block: every non-decode step and every
             # step with a prefill segment in [w0 - 2 s, w1 + 2 s] with its wall and in-window flag -> c8/NAME-steps.txt; logs its in-window non-decode walls (INFO)
  local nm w t0 t1 w0 w1
  nm=$(basename $2 -reqs.jsonl); w=$(mwin $2)
  [ -n "$w" ] || return 0
  read -r t0 t1 w0 w1 < <(python3 -c 'import sys
from datetime import datetime, timezone
a, b = float(sys.argv[1]), float(sys.argv[2])
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
print(iso(a - 2.0), iso(b + 2.0), iso(a), iso(b))' ${w%%:*} ${w#*:})
  timeout 120 python3 -I -B $STP $1 $t0 $t1 $w0 $w1 > $O/c8/$nm-steps.txt 2>&1
  xlog "$nm steps.py (raw, INFO): $(awk '$NF == "True" && $3 != "decode" { n++; if ($(NF-1) != "None") s += $(NF-1) } END { printf "%d in-window non-decode step(s), walls %.1f ms", n, s }' $O/c8/$nm-steps.txt 2>/dev/null)$(grep -q '^Traceback' $O/c8/$nm-steps.txt 2>/dev/null && echo " -- VIRHE steps.py: $(tail -1 $O/c8/$nm-steps.txt | cut -c1-200)")"
}
fusesummary() {   # ncmatch.py's matched-prompt newcomer sums (c8 gens), steps.py's raw listings, then fusesum.py (every ZFUSE line)
  local EL=$O/xzfuse-n5-engine-full.log.gz f l
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $NCM $EL --set c8g --block c8-<i>-G-<arm>:<arm>:$O/c8/c8-<i>-G-<arm>-reqs.jsonl ... --min 4 --json $O/ncm-c8g.json (the matched-prompt newcomer sums, the first block's arm B the base)"
    echo "+ zcat $EL | grep step lines > $O/steplines.tmp; per gens block python3 -I -B $STP $O/steplines.tmp <w0-2s> <w1+2s> <w0> <w1> > $O/c8/<block>-steps.txt; rm the tmp"
    echo "+ python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --all F --want 'B=0/0/0,F=$FRF/$FMF/$FRC,Y=$YRF/$YMF/1,R1=1/0/0,R2=2/0/0,M=0/1/0,R=0/0/1' --xb $XB --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/fusesum.json -> RESULT xzfuse ZFUSE lines (BLOCK / ARM / SINGLE / MODEL / C16 / 1u / REFS / HEADLINE)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip start log, ld text"
    return 0
  fi
  if [ $STOPPED = 2 ]; then xlog "summary: STOPPED after P0 (no fusion form cleared the per-call gate, or ZFU_P0ONLY=1) -- no start, the P0 FUSEPERF / GATE lines are the reading"
  elif [ $STOPPED = 1 ]; then xlog "summary: STOPPED at the identity gate -- no speed block ran, no ZFUSE reading"
  elif [ -s $EL ]; then
    ncmset $EL c8g 4 $(ls $O/c8/c8-*-G-*-reqs.jsonl 2>/dev/null)   # ls order = the run order (c8-1 ..): the first arm (B) is the base
    zcat $EL 2>/dev/null | grep -E ' step [0-9]+ (prefill|decode|MIXED) n_seq=' > $O/steplines.tmp
    for f in $(ls $O/c8/c8-*-G-*-reqs.jsonl $O/c8/c16-*-G-*-reqs.jsonl 2>/dev/null); do stepsx $O/steplines.tmp $f; done
    rm -f $O/steplines.tmp
  else xlog "summary: NOTE no engine log $EL (the start did not come up): no step-log readings"; fi
  if [ $STOPPED = 0 ]; then
    timeout 300 python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --all F --want "B=0/0/0,F=$FRF/$FMF/$FRC,Y=$YRF/$YMF/1,R1=1/0/0,R2=2/0/0,M=0/1/0,R=0/0/1" --xb $XB --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/fusesum.json > $O/fusesum.txt 2>&1
    while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^ZFUSE' $O/fusesum.txt 2>/dev/null)
    grep -q '^ZFUSE' $O/fusesum.txt 2>/dev/null || xlog "summary: VIRHE fusesum.py printed no ZFUSE line ($(tail -3 $O/fusesum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  fi
  grep -lq '^Traceback' $O/fusesum.txt $O/ncm-*.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/fusesum.txt $O/ncm-*.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xzfuse-ld-*.txt $O/c8/*-steps.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
