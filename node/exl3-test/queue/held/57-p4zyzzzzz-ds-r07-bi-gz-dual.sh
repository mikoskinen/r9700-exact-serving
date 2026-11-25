#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-dual.sh (queue item, 11.11.2026, ds-main-port BI agent, lever BI: BI-d + fn4's dual EXL3 + BI-e;
# sorts right after 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-ez-fn-dual (182), before 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid (C and
# en_US order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~190 min) -- tag "RESULT xbil " (A/B) and
# "RESULT xbid " (byte check), out runs/xdsm/bifdual and runs/xdsm/bifdual-op.
# Image IMG below: ds41-bi (int3 1116248 + lever BI .. BI-d 7cfb77b + fn4's arch header dc5830d + the commit IMG names) with fn4's
# radiance-exl3 fn4-dsdual 8d738e2 plugins (exl3 interface v13 = v8 + the EXL3 down's band operand). New ds41_bi values:
#   7 = 6 + BI-e: each dense GGUF linear of a mixed step as ONE gemm_nt_ggml_dual launch (decode rows <= 48 in lever D2's form,
#       every prefill row above the band in the WMMA form, one grid, each row its own launch's bytes; libr4d
#       r4d_ds41_ggml_dual.hip); a prefill-only step (every sequence > 48 rows) one gemm_nt_ggml over the step;
#   8 = 6 + fn4's dual EXL3 down: in a step with a prefill sequence ONE routed MoE pass (MoeFP8::pass_bi: router + top-k and
#       the shared arm by piece as 6's passes, one sort over [0, T), EXL3 gate_up + down once with the step's band -- decode
#       rows and prefill sequences of <= 16 rows marked, written on the card by two recorded iota ops -- one gather + AR /
#       ARGATHER);
#   9 = 8 + 7 (the full set; coordinator target: a mixed step within ~2 % of bi 0, c8 within 1 %).
# PHASES:
#   gate  fn4's 182 (bi-ez-fn-dual) must have run and passed: in its last run, for DeepSeek's geometries ds2 and ds25, every
#         MARKED (= decode band) and UNMARKED (= plain prefill band) comparison "0 differ", the control (the flag acts) with y
#         differing, no VIRHE, its done line; an explicit "RESULT fn4 182 ... PASS|FAIL" verdict line, if logged, decides.
#         Not PASS -> "RESULT xbil ...: VIRHE" and stop (nothing runs).
#   op    dsm-bidual.sh on ONE card (~20 min): r4d_selftest --dualcheck (gemm_nt_ggml_dual vs the two gemm_nt_ggml launches,
#         byte for byte, DS dense TP4 shapes, M0 1..48 x M1 49..4160, both block orders, + DUALTIME), --mrcheck and --gpcheck
#         (lever D2 / GPF forms vs the stock kernels: the kernels are now bodies behind their __global__ wrappers), arms stock
#         and stack. Not PASS -> 9 is dropped from the A/B below (logged), the rest runs.
#   A/B   dsm-bilid7.sh 92a3997e (unchanged), starts p0 p5 (profiled: one mb pf window a value -> device ms / launches added a
#         mixed step vs 0), n0 n5 (identity of 8 and 9: a1 vs bz d0, ld vs a1 with c4 + a fresh 32k prefill, c4 run-to-run,
#         a2 control; mb c4 / c8 / pf blocks 0 4 6 8 9 9 8 6 4 0, 60 s a phase: every value vs 0, and 6 vs 4 at both NST --
#         what held bi-f-pc gave). The in-process switch ds41_bi; RADIANCE_DS41_BI=0 declared at start.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-dual.sh : the gate's present verdict (information), dsm-bidual.sh's and
#   dsm-bilid7.sh's dry runs (image literals, helpers, sidecar, yaml); nothing runs, nothing written.
IMG=${XBIL_IMAGE:-radiance-local:ds41-bi-5bb7ce0c-8d738e2-l}
ITEM=57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-dual
T=$HOME/gpu/ds41/exl3-test; L=$T/runs/ds41-exl3.log; D=$T/queue-ar/dsmain
DRY=${DRYRUN:-0}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
for f in $D/dsm-bilid7.sh $D/dsm-bidual.sh; do [ -e $f ] || { log "RESULT xbil $ITEM: VIRHE $f missing"; exit 1; }; done
gate182() {   # PASS, or a VIRHE reason (one line)
  python3 - "$L" <<'PY'
import re, sys
try: lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError as e: print("VIRHE cannot read %s: %s" % (sys.argv[1], e)); sys.exit(0)
st = [i for i, l in enumerate(lines) if "RESULT fn4 182-fn-dual: start" in l]
if not st: print("VIRHE 182 has not run (no 'RESULT fn4 182-fn-dual: start' line)"); sys.exit(0)
run = lines[st[-1]:]
ver = [l for l in run if re.search(r"RESULT fn4 182\b.*\b(PASS|FAIL)\b", l) and "want 0 differ" not in l]
if ver:
    print("PASS" if re.search(r"\bPASS\b", ver[-1]) and not re.search(r"\bFAIL\b", ver[-1]) else "VIRHE 182 verdict: " + ver[-1][:300])
    sys.exit(0)
if not any("RESULT fn4 182-fn-dual: done" in l for l in run): print("VIRHE 182 has not finished (no done line)"); sys.exit(0)
vir = [l for l in run if "RESULT fn4 182" in l and "VIRHE" in l]
if vir: print("VIRHE 182 logged: " + vir[0][:300]); sys.exit(0)
seen, bad = set(), []
for l in run:
    m = re.search(r"RESULT fn4 182 (ds2|ds25) (MARKED|UNMARKED) ", l)
    if m:
        seen.add(m.group(1) + " " + m.group(2))
        if re.search(r"\b[1-9]\d* differ", l) or "missing" in l: bad.append(m.group(1) + " " + m.group(2) + " differs")
    m = re.search(r"RESULT fn4 182 (ds2|ds25) control", l)
    if m:
        seen.add(m.group(1) + " control")
        if not re.search(r"y \d+ slots, [1-9]\d* differ", l): bad.append(m.group(1) + " control: the band does not act on y")
need = {"ds2 MARKED", "ds2 UNMARKED", "ds25 MARKED", "ds25 UNMARKED", "ds2 control", "ds25 control"}
if need - seen: print("VIRHE 182 lines missing: " + ", ".join(sorted(need - seen))); sys.exit(0)
print("PASS" if not bad else "VIRHE 182 not PASS: " + "; ".join(bad))
PY
}
G=$(gate182)
if [ "$DRY" = 1 ]; then
  echo "+ gate 182 now: $G (the real run stops with a VIRHE unless PASS)"
elif [ "$G" != PASS ]; then
  log "RESULT xbil $ITEM: VIRHE gate -- fn4's 182 (bi-ez-fn-dual) is not PASS: $G -- nothing ran"; exit 1
else
  log "RESULT xbil $ITEM: gate -- fn4's 182 PASS (ds2 / ds25 marked = decode band, unmarked = prefill band, the band acts)"
fi
# op: the BI-e byte check on one card
VALS="0 4 6 8 9"; IDV="8 9"
XBID_IMAGE=$IMG XBID_ITEM=$ITEM-op XBID_OUT=$T/runs/xdsm/bifdual-op bash $D/dsm-bidual.sh
if [ "$DRY" != 1 ]; then
  if ! grep -q "RESULT xbid summary: PASS" <(sed -n "/RESULT xbid $ITEM-op: start/,\$p" $L); then
    VALS="0 4 6 8"; IDV="8"
    log "RESULT xbil $ITEM: the BI-e byte check is not PASS -- ds41_bi 9 (dual gemm_nt_ggml) dropped from the A/B: values [$VALS]"
  fi
fi
# A/B
export XBIL_IMAGE=$IMG XBIL_ITEM=$ITEM XBIL_OUT=$T/runs/xdsm/bifdual XBIL_STARTS=${XBIL_STARTS:-p0 p5 n0 n5}
export XBIL_EXTRAENV=RADIANCE_DS41_BI=0 XBIL_SW=ds41_bi XBIL_SWVALS="$VALS" XBIL_IDVALS="$IDV" XBIL_PROFVALS="$VALS"
export XBIL_C4RR=1 XBIL_TRACE=0 XBIL_CTRL=1
export XBIL_MB=c4,c8,pf XBIL_MBORDER="$VALS $(echo $VALS | tr ' ' '\n' | tac | tr '\n' ' ')" XBIL_MBDUR=60 XBIL_MBWARM=15
export XBIL_OFFSET=0.5 XBIL_PFTOK=32768
export XBIL_LITS="A|RADIANCE_DS41_BI
A|lever BI declared
A|with one banded MoE pass
A|a dual form (7 / 9)
K|gemm_nt_ggml_dual
B|ds41_bi"
bash $D/dsm-bilid7.sh
