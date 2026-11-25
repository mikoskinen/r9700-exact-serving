# ---------------------------------------------------------------- lever EPB (this item's own functions)
p0epb() {   # P0 (no server; the FOUR cards, chosen by chip unique_id -> ROCR_VISIBLE_DEVICES=GPU-..): r4d_selftest's lever-D3s cases
            # (P0CASES) -> 0 = every case clean (rc 0, 0 FAIL, > 0 ok); else VIRHE and the item STOPS (no server start)
  local c f rc nok nf ns bad=0 r devs u d4
  devs=$(for u in /sys/class/drm/card*/device/unique_id; do d4=$(dirname $u); [ "$(cat $d4/device 2>/dev/null)" = 0x7551 ] && echo "GPU-$(cat $u)"; done 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
  if [ "$DRY" != 1 ] && [ $SW = 1 ] && [ "$(echo "$devs" | tr ',' '\n' | grep -c .)" != 4 ]; then xlog "P0: VIRHE not four R9700 cards by unique_id ('$devs') -- the item stops"; return 1; fi
  local DEV4="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e ROCR_VISIBLE_DEVICES=$devs"
  mkd $O/p0
  for c in $P0CASES; do
    f=$O/p0/$c.txt
    run "gpurun xepb-p0-${c:0:24}-$$ $P0TMO $DEV4 --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $f 2>&1"; rc=$?
    if [ "$DRY" = 1 ]; then echo "+ RESULT xepb P0 $c on the four cards ($devs): rc 0, 0 FAIL, > 0 ok (bounded at $P0TMO s), else VIRHE and the item STOPS"; continue; fi
    nf=$(grep -c 'FAIL' $f); nok=$(grep -cE '^ *(ok|PASS)' $f); ns=$(grep -ciE '^ *skip' $f)
    r="$c: rc $rc, $nok ok, $nf FAIL, $ns skip$([ "$nf" -gt 0 ] && echo " -- first: $(grep -m1 FAIL $f | cut -c1-300)")$({ [ $rc = 124 ] || [ $rc = 137 ]; } && echo ' -- STOPPED by the limit (hang)')"
    if [ $rc = 0 ] && [ "$nf" = 0 ] && [ "$nok" -gt 0 ]; then xlog "P0 $r | $(grep -m1 'shapes took the two-shot' $f | sed 's/^ *//' | cut -c1-200)"
    elif [ $SW = 1 ]; then xlog "P0 VIRHE $r"; bad=1
    else xlog "P0 $c: failed under XEP_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on ($r)"; fi
  done
  [ "$DRY" = 1 ] && return 0
  [ $bad = 0 ] && { xlog "P0: lever D3s / EPB selftest CLEAN (slot-order sum = the oracle and the same bytes under every placement, both transports; placed sort = the host reference)"; return 0; }
  xlog "P0: VIRHE the selftest is not clean -> the item STOPS (no server start)"; return 1
}
epbnote() {   # epbnote TAG: lever EPB's start note (rank 0): passes of up to SLROWS rows, 40 of 40 trunk layers, tables 0 / 1 / 2 placed,
              # no "did not resolve" line, switches 0 / 0 at start; the extras a rank; VRAM at start vs prof1's int12 start (the reservation)
  [ "$DRY" = 1 ] && { echo "+ epbnote $1: 'the slot-order MoE sum (lever D3s / EPB) in the decode band, MoE passes of up to $SLROWS rows ... 40 of 40 trunk layers ... tables: 0 contiguous, 1 /stats/place/$PL1 (40 layers placed ..), 2 /stats/place/$PL2 (..) ... switches ds41_slotsum 0, ds41_place 0'; no 'did not resolve'; VRAM at start vs prof1 ($P1STATS)"; return 0; }
  local n u ok=1
  n=$(docker logs $CN 2>&1 | grep -m1 'the slot-order MoE sum (lever D3s / EPB)' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-900)
  u=$(docker logs $CN 2>&1 | grep -c 'the slot-order MoE path (lever D3s) did not resolve')
  case "$n" in *"MoE passes of up to $SLROWS rows"*"40 of 40 trunk layers"*"1 /stats/place/$PL1 (40 layers placed"*"2 /stats/place/$PL2 (40 layers placed"*"switches ds41_slotsum 0, ds41_place 0"*) ;; *) ok=0 ;; esac
  [ "$u" = 0 ] || ok=0
  xlog "$1 EPB note: ${n:-NONE} | 'did not resolve' lines $u (want 0) | VRAM at start (experts per tier, rank 0 / 1 ..; this start, then prof1's int12 start): $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" | ")
    except Exception as e: print("n/a (%s)" % type(e).__name__, end=" | ")' $O/$1-stats0.json $P1STATS 2>&1 | cut -c1-400)$([ $ok = 1 ] || { [ $SW = 1 ] && echo " -- VIRHE the EPB declaration is not as asked" || echo " -- INFO (XEP_SW=0 mock: the note is the mock's echo of the literal)"; })"
  [ $ok = 1 ] || [ $SW = 0 ]
}
p0id() {   # P0b at START A, before any speed reading: E's DETERMINISM (a1 vs a2 alone) and BATCH INVARIANCE (gs16 all in flight, gs8 vs a1),
           # O's a1 vs ref12's a1 (the declared-but-off path keeps today's bytes), E's a1 vs O's a1 (must DIFFER: the slot path ran) -> 0 / 1
  local k l bad=0 u0 u1
  alone xepb E a1E
  alone xepb E a2E
  cmpx xepb a2E-vs-a1E $O/xepb-a1E.jsonl $O/xepb-a2E.jsonl
  u0=$(utcnow); ssnap xepb-gs${GSC}E-0; gsx E gs${GSC}E; ssnap xepb-gs${GSC}E-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs${GSC}E.win
  cmpx xepb gs${GSC}E-vs-a1E $O/xepb-a1E.jsonl $O/xepb-gs${GSC}E.jsonl
  local gsc0=$GSC; GSC=8; gsx E gs8E; GSC=$gsc0   # the same rows inside 8-sequence steps too
  cmpx xepb gs8E-vs-a1E $O/xepb-a1E.jsonl $O/xepb-gs8E.jsonl
  alone xepb O a1O
  cmpx xepb a1O-vs-ref12-a1 $R12A1 $O/xepb-a1O.jsonl
  [ -s $P1A1 ] || [ "$DRY" = 1 ] && cmpx xepb a1O-vs-prof1-a1 $P1A1 $O/xepb-a1O.jsonl
  cmpx xepb a1E-vs-a1O-INFO $O/xepb-a1O.jsonl $O/xepb-a1E.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT xepb P0b: a2E vs a1E, gs${GSC}E vs a1E, gs8E vs a1E, a1O vs ref12 a1 IDENTICAL (strict) and a1E vs a1O DIFFERENT (the slot path ran), else VIRHE and the item STOPS"; return 0; }
  for k in a2E-vs-a1E gs${GSC}E-vs-a1E gs8E-vs-a1E a1O-vs-ref12-a1; do
    l=$(grep "^xepb-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) ;; *) xlog "P0b $k: VIRHE ${l:-not compared} (strict)"; bad=1 ;; esac
  done
  l=$(grep "^xepb-a1E-vs-a1O-INFO " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) [ $SW = 1 ] && { xlog "P0b a1E vs a1O: VIRHE IDENTICAL -- the slot path did not run under E (ds41_slotsum 1)"; bad=1; } ;; esac
  xlog "P0b gs${GSC}E cap (all $GSC in flight, sched_row_cap $CAP): $(capd xepb-gs${GSC}E-0 xepb-gs${GSC}E-1)"
  [ $bad = 0 ] && { xlog "P0b: E deterministic (a1 = a2) and batch-invariant at conc $GSC and 8 (gs$GSC = gs8 = a1); O = today's bytes (ref12's a1); E differs from O (the slot path ran)"; return 0; }
  [ $SW = 1 ] && return 1
  xlog "P0b: INFO under XEP_SW=0 (a mock run: the mock does not model the slot path) -- the mock goes on"; return 0
}
epbstag() {   # STAGGERED arrivals under E (stack4/loadid.py run, read only): LDCONC decode streams + the 20 greedy probes one at a time,
              # each with a FRESH LDPF-token prefill joining 0.5 s in (its chunks ride the probe's decode: MIXED steps) -> vs E's a1 alone, strict
  local t0=$(date +%s) l
  arm E || return 1
  pcreset xepb-ldE; ssnap xepb-ldE-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/xepb-ldE.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/xepb-ldE.txt 2>&1)"
  ssnap xepb-ldE-1
  cmpx xepb ldE-vs-a1E $O/xepb-a1E.jsonl $O/xepb-ldE.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT xepb staggered E: LOADID (probes, MIXED steps during the probes, prefills cold) + ldE vs a1E strict (a difference = VIRHE)"; return 0; }
  xlog "staggered E (loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s): $(grep -m1 '^LOADID ' $O/xepb-ldE.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/xepb-ldE.jsonl 2>&1 | cut -c1-200) | cap $(capd xepb-ldE-0 xepb-ldE-1)"
  l=$(grep "^xepb-ldE-vs-a1E " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) ;; *) [ $SW = 1 ] && xlog "staggered E: VIRHE ${l:-not compared} (strict: a decode row riding MIXED steps must keep its alone bytes)" ;; esac
}
epbreads() {   # START A's readings in their order (module header)
  local a i k
  if budok "P0b determinism / BI (E a1, a2, gs$GSC; O a1)" 5; then
    p0id || { xlog "P0b: VIRHE E is not deterministic / batch-invariant (or O is not today's bytes) -> the item STOPS: no speed reading on this arm"; STOPALL=1; return 1; }
  else STOPALL=1; return 1; fi
  i=1
  for a in $C8ORDER; do stopfile && return 0; budok "c8 block $i ($a)" 2 && { mbx 8 $i $a S || true; }; i=$((i + 1)); done
  for a in $C16ARMS; do stopfile && return 0; budok "c16 CAP48 method ($a)" 2 && { c16x $a || true; }; done
  stopfile && return 0
  [ $C16NC = 1 ] && budok "c16 fixed-harness newcomer block (E)" 2 && { mbx 16 1 E S || true; }
  for a in $U1ARMS; do
    stopfile && return 0
    budok "1u essay $a (one discarded + $NREADS back to back)" 2 || continue
    blockx ${a}U1W $a essay || true
    for k in $(seq 1 $NREADS); do blockx ${a}U1 $a essay || true; done
  done
  stopfile && return 0
  budok "staggered arrivals (E)" 5 && { epbstag || true; }
  stopfile && return 0
  if [ -s $R12A1 ] || [ "$DRY" = 1 ]; then cmpx xepb a1E-vs-ref12-a1-INFO $R12A1 $O/xepb-a1E.jsonl   # reading 5: INFO, expected to differ
  else xlog "vs-int12 texts: $R12A1 missing -- not compared"; fi
  return 0
}
dstartp() {   # START B: the PROFILED start (prof2's pattern: --profile-ops, RADIANCE_PROFILE_EVERY PEVERY; NOT a speed), c8 windows under PPLAN
  local tag=xepb-p5 a k=1 DISKMIN=$PDISK
  diskok $tag || return 1
  mkd $STATSDIR/place
  run "cp $PLDIR/$PL1 $PLDIR/$PL2 $STATSDIR/place/"
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${XEP_MAXSEQS:-}" ] && export MAXSEQS=$XEP_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on --profile-ops" PROFEVERY=$PEVERY
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (START B, PROFILED --profile-ops, RADIANCE_PROFILE_EVERY $PEVERY: not a speed; lever EPB declared as START A) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-600)"
  gate1 $tag 400 || { xlog "$tag: VIRHE not healthy ($(desc))"; down; export PROFEVERY=; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; export PROFEVERY=; return 1; }
  drafted $tag || { down; export PROFEVERY=; return 1; }
  readback $tag || { keeplog $tag-readback; down; export PROFEVERY=; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; down; export PROFEVERY=; return 1; }
  epbnote $tag || { down; export PROFEVERY=; return 1; }
  for a in $PPLAN; do stopfile && break; budok "profiled c8 window ($a)" 3 $((BUDGET + 10)) && { mbx 8 $k $a S || true; }; k=$((k + 1)); done
  run "docker stop -t 60 $CN > /dev/null 2>&1"        # the shutdown table closes the last interval (prof2's)
  capture $O/$tag-engine-full.log.gz
  down
  export PROFEVERY=
  xlog "$tag: done"
}
epbprof() {   # START B's reading: moe_gather_all_reduce(_slot) us a call and ms a decode step, the attention all_reduce us a call, the
              # per-rank MoE experts ms, from p12win.py's P12TOP / P12PROF lines -> RESULT EPBPROF vs prof2 (154.6 us / 6.11 ms; AR 50.6 us)
  [ "$DRY" = 1 ] && { echo "+ RESULT xepb EPBPROF c8<arm> per rank: gather_ar(_slot) ms a step / us a call, all_reduce us a call, the wait (gather_ar - all_reduce us), MoE experts ms; vs prof2 $P2GARUS us / $P2GARMS ms"; return 0; }
  [ -s $O/p-prof.out ] || { xlog "EPBPROF: no p-prof.out (no profiled window)"; return 0; }
  python3 -c 'import re,sys
g_us, g_ms, a_us = float(sys.argv[2]), float(sys.argv[3]), 50.6
L = open(sys.argv[1]).read().splitlines()
W = {}
for l in L:
    m = re.match(r"P12TOP (\S+) rank (\d+):", l)
    if m:
        w, r = m.group(1), int(m.group(2))
        d = W.setdefault(w, {}).setdefault(r, {})
        for op in ("moe_gather_all_reduce_slot", "moe_gather_all_reduce", "all_reduce", "moe_scatter_placed", "moe_exl3_gate_up", "moe_exl3_down"):
            x = re.search(r"(?:^|; |: )%s ([0-9.]+) ms ([0-9.]+) us" % re.escape(op), l.split("top ops", 1)[-1])
            if x: d[op] = (float(x.group(1)), float(x.group(2)))
    m = re.match(r"P12PROF (\S+) rank (\d+) .*?MoE experts ([0-9.]+) .*?\(coll ([0-9.]+) / gar ([0-9.]+)\)", l)
    if m: W.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {}).update(experts=float(m.group(3)), coll=float(m.group(4)), gar=float(m.group(5)))
for w, R in sorted(W.items()):
    parts, gu, gm, wt = [], [], [], []
    for r, d in sorted(R.items()):
        gk = "moe_gather_all_reduce_slot" if "moe_gather_all_reduce_slot" in d else "moe_gather_all_reduce"
        ga, ar = d.get(gk), d.get("all_reduce")
        if ga: gu.append(ga[1]); gm.append(ga[0])
        if ga and ar: wt.append(ga[1] - ar[1])
        parts.append("rank %d: %s %s ms a step / %s us a call, all_reduce %s us, wait %s us, MoE experts %s ms, gar %s ms, scatter_placed %s" % (r, gk,
            ga[0] if ga else "-", ga[1] if ga else "-", ar[1] if ar else "-", ("%.1f" % (ga[1] - ar[1])) if ga and ar else "-", d.get("experts", "-"), d.get("gar", "-"),
            ("%s ms" % d["moe_scatter_placed"][0]) if "moe_scatter_placed" in d else "-"))
    mean = lambda v: sum(v) / len(v) if v else None
    gum, gmm, wtm = mean(gu), mean(gm), mean(wt)
    print("EPBPROF %s (PROFILED, not a speed): gather_ar mean %s us a call (prof2 int12 %.1f: %s), %s ms a step (prof2 %.2f: %s), wait over the all_reduce mean %s us (prof2 %.1f) | %s" % (w,
        ("%.1f" % gum) if gum else "-", g_us, ("%+.1f %%" % (100 * (gum - g_us) / g_us)) if gum else "-", ("%.2f" % gmm) if gmm else "-", g_ms,
        ("%+.1f %%" % (100 * (gmm - g_ms) / g_ms)) if gmm else "-", ("%.1f" % wtm) if wtm is not None else "-", g_us - a_us, " | ".join(parts)))' $O/p-prof.out $P2GARUS $P2GARMS > $O/epbprof.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2400}"; done < <(grep -h '^EPBPROF' $O/epbprof.txt)
  grep -q '^EPBPROF' $O/epbprof.txt || xlog "EPBPROF: VIRHE no P12TOP / P12PROF line parsed ($(tail -2 $O/epbprof.txt | tr '\n' ' ' | cut -c1-300))"
}
epbtidy() {   # gzip the big text outputs (> 4 MB, never the identity / reqs jsonl the reports name), the disk after
  [ "$DRY" = 1 ] && { echo "+ gzip -1 every file > 4 MB under $O (not .gz, not *.jsonl); disk logged"; return 0; }
  find $O -type f -size +4M ! -name '*.gz' ! -name '*.jsonl' -exec gzip -1 {} + 2>/dev/null
  xlog "tidy: outputs $(du -sm $O 2>/dev/null | cut -f1) MB under $O (> 4 MB files gzipped); $(dfree) MB free on /"
}
