summary() {   # START A: p12win.py split + rows for every c8 block and the c16 blocks, confirm + rows for the 1u reads by arm, the gs16E
              # window's > 48-row steps, then the arm table (EPBAB: E / S / E4 vs O in process; O vs prof1 = the reservation's price)
  local EL=$O/xepb-n5-engine-full.log.gz w f nm l a W=() ref
  if [ "$DRY" = 1 ]; then
    echo "+ per c8 / c16 block (c8-<i>-S-<arm>, c16-1-S-E, c16-cap-<arm>): python3 -I -B $R12S split $EL --win <name>:<marks> --mbline --sw0 --sw1 --target 200 [--ref prof1_c8:8.32:36.18:0.801:184.0] -> P12SPLIT; rows -> P12ROWS"
    echo "+ 1u: python3 -I -B $R12S confirm $O/blocks.jsonl --lever <arm>U1 (each arm) -> P12CONFIRM; rows over every read -> P12ROWS"
    echo "+ gs${GSC}E: rows over its window (> 48-row steps must be 0); EPBAB: the c8 blocks by arm (agg, ms / tok a decode step, decode_share, TTFT p50) E / S / E4 vs O (%), O vs prof1 $P1C8; c16 E vs O and vs prof1 $P1C16"
    return 0
  fi
  [ -s $EL ] || { xlog "summary: no engine log ($EL) -- no step-log reading"; epbab; return 0; }
  for f in $O/c8/c8-*-S-*-reqs.jsonl $O/c8/c16-*-S-*-reqs.jsonl $O/c8/c16-cap-*-reqs.jsonl; do
    [ -s "$f" ] || continue
    nm=$(basename $f -reqs.jsonl); w=$(mwin $f)
    [ -n "$w" ] || { xlog "summary: $nm -- no window (no marks)"; continue; }
    ref=""; case $nm in c8-*) ref="--ref prof1_c8:8.32:36.18:0.801:184.0" ;; esac
    timeout 300 python3 -I -B $R12S split $EL --win $nm:$w --mbline $O/c8/$nm.txt --sw0 $O/c8/$nm-st-w0.json --sw1 $O/c8/$nm-st-w1.json --target 200 $ref > $O/split-$nm.txt 2>&1
    timeout 300 python3 -I -B $R12S rows $EL --win $nm:$w --json $O/rows-$nm.json > $O/rows-$nm.txt 2>&1
    while IFS= read -r l; do xlog "$nm ${l:0:2400}"; done < <(grep -hE '^P12(SPLIT|ROWS)' $O/split-$nm.txt $O/rows-$nm.txt 2>/dev/null)
  done
  for a in $U1ARMS; do
    [ -s $O/blocks.jsonl ] || break
    timeout 120 python3 -I -B $R12S confirm $O/blocks.jsonl --lever ${a}U1 --expect-tps 90 --expect-mb 27 --ref-tps $P1U1 --ref-mb 10.6 > $O/confirm-$a.txt 2>&1
    while IFS= read -r l; do xlog "1u $(NAMEOF $a) ${l:0:1800} | ref = prof1's int12 back-to-back essay $P1U1 tok/s"; done < <(grep -h '^P12CONFIRM' $O/confirm-$a.txt 2>/dev/null)
  done
  python3 -c 'import json,sys
o = sys.argv[1]; W = []
for l in open(sys.argv[2]):
    d = json.loads(l)
    if not d.get("lever", "").endswith(("U1", "U1W")): continue
    n = "r%d-%s" % (d["i"], d["lever"]); b = "b%d-%s-%s" % (d["i"], d["lever"], d["arm"])
    W += ["--win", "%s:%.6f:%.6f" % (n, d["utc0"], d["utc1"]), "--st", "%s:%s/st-%s-0.json:%s/st-%s-1.json:%.6f:%.6f" % (n, o, b, o, b, d["utc0"], d["utc1"])]
print("\n".join(W))' $O $O/blocks.jsonl > $O/u1win.args 2>/dev/null
  [ -s $O/u1win.args ] && { mapfile -t W < $O/u1win.args; timeout 300 python3 -I -B $R12S rows $EL "${W[@]}" --json $O/u1rows.json > $O/u1rows.txt 2>&1; }
  while IFS= read -r l; do xlog "1u ${l:0:1500}"; done < <(grep -h '^P12ROWS' $O/u1rows.txt 2>/dev/null)
  if [ -s $O/gs${GSC}E.win ]; then
    local u0 u1; read -r u0 u1 < $O/gs${GSC}E.win
    timeout 300 python3 -I -B $R12S rows $EL --win gs${GSC}E:$u0:$u1 --json $O/gsrows.json > $O/gsrows.txt 2>&1
    while IFS= read -r l; do xlog "gs${GSC}E step log (the > 48-row count must be 0 under the cap) ${l:0:1500}"; done < <(grep -h '^P12ROWS' $O/gsrows.txt 2>/dev/null)
    python3 -c 'import json,sys; d = json.load(open(sys.argv[1]))["windows"]; w = list(d.values())[0]; sys.exit(0 if w.get("gt48", 0) == 0 else 1)' $O/gsrows.json 2>/dev/null \
      || { [ $SW = 1 ] && xlog "gs${GSC}E step log: VIRHE > 48-row decode steps under the cap (or no step-log reading)" || xlog "gs${GSC}E step log: INFO > 48-row steps (SW=0 mock: no row cap emulated)"; }
  fi
  epbab
  grep -lq '^Traceback' $O/split-*.txt $O/rows-*.txt $O/confirm-*.txt $O/u1rows.txt $O/gsrows.txt $O/epbab.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/split-*.txt $O/rows-*.txt $O/confirm-*.txt $O/u1rows.txt $O/gsrows.txt $O/epbab.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  return 0
}
epbab() {   # EPBAB: START A's arm table from $O/mb.jsonl (c8 / c16 fixed-harness blocks), the P12SPLIT decode ms by block and the c16
            # CAP48-method MB lines: E / S / E4 vs O in process (mean over each arm's blocks, %), the B-vs-B spread, O vs prof1 (the price of
            # the declaration: extras + slot ops declared, idle under O)
  python3 -c 'import json,re,sys,glob,os
o, p1c8, p1c16 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
B = []
try: B = [json.loads(l) for l in open(os.path.join(o, "mb.jsonl")) if l.strip()]
except OSError: pass
sp = {}
for f in glob.glob(os.path.join(o, "split-c8-*.txt")):
    m = re.search(r"\(a\) decode (\d+) steps \d+ ms = [^,]+, ([0-9.]+) ms a step, ([0-9.]+) rows a step.*?\(b\) newcomer / MIXED (\d+) steps \d+ ms = [^,]+, (\S+) rows a step, (\S+) ms a step", open(f).read())
    if m: sp[os.path.basename(f)[6:-4]] = dict(dms=float(m.group(2)), rows=float(m.group(3)), nc=int(m.group(4)), ncms=m.group(6))
mean = lambda v: sum(v) / len(v) if v else None
pc = lambda a, b: ("%+.2f %%" % (100.0 * (a - b) / b)) if a is not None and b else "-"
for n in (8, 16):
    arms = {}
    for d in B:
        if d.get("n") != n or d.get("agg_dec_tps") is None: continue
        arms.setdefault(d["arm"], []).append(d)
    if not arms: continue
    agg = {a: mean([d["agg_dec_tps"] for d in v]) for a, v in arms.items()}
    ms = {a: mean([d["ms_decode_step"] for d in v if d.get("ms_decode_step") is not None]) for a, v in arms.items()}
    tk = {a: mean([d["tok_per_decode_step"] for d in v if d.get("tok_per_decode_step") is not None]) for a, v in arms.items()}
    ds = {a: mean([d["decode_share"] for d in v if d.get("decode_share") is not None]) for a, v in arms.items()}
    sm = {a: mean([sp[d["name"]]["dms"] for d in v if d["name"] in sp]) for a, v in arms.items()}
    rw = {a: mean([sp[d["name"]]["rows"] for d in v if d["name"] in sp]) for a, v in arms.items()}
    blk = "; ".join("%s %s %.1f" % (d["name"], d["arm"], d["agg_dec_tps"]) for v in arms.values() for d in v)
    lines = []
    for a in ("E", "S", "E4"):
        if a in arms and "O" in arms:
            lines.append("%s vs O: agg %s, ms a decode step (MB) %s, ms a decode step (split) %s, tok a step %s vs %s, rows a step %s vs %s, decode_share %s vs %s" % (a,
                pc(agg[a], agg["O"]), pc(ms[a], ms["O"]), pc(sm[a], sm["O"]), tk[a], tk["O"], rw[a], rw["O"], ds[a], ds["O"]))
    spread = "; ".join("%s B-vs-B %s" % (a, pc(max(d["agg_dec_tps"] for d in v), min(d["agg_dec_tps"] for d in v))) for a, v in arms.items() if len(v) > 1)
    ref = (" | O vs prof1 int12 c8 %.1f: %s (the declaration'"'"'s price: extras + slot ops declared, idle under O)" % (p1c8, pc(agg.get("O"), p1c8))) if n == 8 and "O" in agg else ""
    print("EPBAB c%d (fixed harness, in process; mean over blocks): %s | %s | spread: %s%s | blocks: %s" % (n,
        ", ".join("%s agg %.1f tok/s ms %.2f" % (a, agg[a], ms[a] or 0) for a in sorted(agg)), " || ".join(lines) or "-", spread or "-", ref, blk))
c16 = {}
for f in glob.glob(os.path.join(o, "c8", "c16-cap-*.txt")):
    for l in open(f):
        if l.startswith("MB "):
            try: c16[os.path.basename(f)[8:-4]] = json.loads(l[3:]); break
            except ValueError: pass
if c16:
    print("EPBAB c16 CAP48 method (cap48/ncmb.py phase c16, T 0): %s%s" % ("; ".join("%s agg %s tok/s (vs prof1 int12 %.1f: %s), %s ms / %s tok a decode step, decode_share %s" % (a, d.get("agg_dec_tps"), p1c16,
        pc(d.get("agg_dec_tps"), p1c16), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("decode_share")) for a, d in sorted(c16.items())),
        (" || E vs O in process: %s" % pc(c16["E"].get("agg_dec_tps"), c16["O"].get("agg_dec_tps"))) if "E" in c16 and "O" in c16 else ""))' $O $P1C8 $P1C16 > $O/epbab.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^EPBAB' $O/epbab.txt)
  grep -q '^Traceback' $O/epbab.txt && xlog "EPBAB: VIRHE the arm table failed ($(tail -2 $O/epbab.txt | tr '\n' ' ' | cut -c1-300))"
  return 0
}
