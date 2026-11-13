#!/bin/bash
# 132-fn-stack.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~60 min, one
# arm) -- the STACK CONFIRMATION on comparable numbers (93-style, coordinator 10.11.): one start of the full main-line
# stack, arm stk of 93o-fn-tp4ab.sh beside this file:
#   radiance-local:c16-sw4 (fn4-c16 757e35a), F1n + F2a + F2b + hcw (hc_writevec 1) + hc_readv bit 0 (the decode tiles;
#   131 c16 1.0182, rounds 1.0129 / 1.0218 / 1.0198, c8 1.0125), MNBT 4096 + --prefill-span 2, --prefill-interleave 0,
#   --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6 210 W; no RADIANCE_FN4_SWITCHABLE (the
#   deployable declaration).
#   hc_readv bit 1 (PF2's 32-row down product) and ar_lanethreads 256 are added ONLY if 131's prefill blocks show them
#   winning EVERY round at every depth (16k / 64k / 128k B/A > 1 in each round, read from 131's abblocks2 output
#   $O/fn4-hcreadv-hcreadv-pf-c16.txt / fn4-hcreadv-arlane256-c16.txt); otherwise they stay off and the RESULT line says so.
# Phases (93o, X93_SCREEN 0): gate (1-user), slow-state probe, produp pair + probe (informational), mb c4 / c8 / c16
# (180 s each) with engine stats, THEN prefill 16k warm + fresh 64k / 128k (a thermal line each), the 33k copy test byte
# for byte against 128's cold copy (X93_COPYREF abval2-cold: c16-sw2 F1n; every switch since is exact by its identity
# check), mb pf, BetterBench --quick (1-user + the concurrency and prefill sweeps), the c16 card-0 profile; no cc_sim.
# Compared (93o cmparm, X93_REF mpB, no re-run) against 126's mpB (c16-sw + F1n + pipelined prefill: mb c16 982.8,
# prefill 16k / 54k / 120k 8154.2 / 8233.9 / 8166.3) and the user's targets (mb c16 >= 1000, prefill >= 8500 warm).
# DRYRUN=1 bash 132-fn-stack.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
# every round's prefill B/A above 1 at every depth in 131's blocks file -> "1", else "0 (reason)"
wins() {
  python3 - "$1" <<'PY' 2>/dev/null || echo "0 (not parsed)"
import json, sys
try:
    l = next(x for x in open(sys.argv[1]) if x.startswith("ABB "))
except (OSError, StopIteration):
    print("0 (no 131 result)"); sys.exit(0)
d = json.loads(l[4:])
pf = d.get("pf") or {}
deps = ("16384", "65536", "131072")
if not all(k in pf and pf[k]["rounds"] for k in deps):
    print("0 (131 incomplete)"); sys.exit(0)
rr = {k: pf[k]["rounds"] for k in deps}
ok = all(x > 1.0 for k in deps for x in rr[k])
print(("1" if ok else "0") + " (" + "; ".join("%dk %s" % (int(k) // 1024, "/".join("%.4f" % x for x in rr[k])) for k in deps) + ")")
PY
}
PFW=$(wins $O/fn4-hcreadv-hcreadv-pf-c16.txt); ALW=$(wins $O/fn4-hcreadv-arlane256-c16.txt)
RV=1; [ "${PFW%% *}" = 1 ] && RV=3
AL=""; [ "${ALW%% *}" = 1 ] && AL=256
log "RESULT fn4 132-fn-stack: the stack = F1n + F2a + F2b + hcw + hc_readv $RV + ar_lanethreads ${AL:-off} | hc_readv bit 1 (PF2 32-row down) $( [ $RV = 3 ] && echo ON || echo OFF): 131 prefill B/A rounds ${PFW#* } | ar_lanethreads 256 $( [ -n "$AL" ] && echo ON || echo OFF): 131 rounds ${ALW#* }"
export ARMS=${X132_ARMS:-stk} X93_REF=${X93_REF:-mpB} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-abval2-cold}
export X93_STK_RV=${X132_RV:-$RV} X93_STK_AL=${X132_AL:-$AL}
export X93_SCREEN=0 X93_SCREEN_PF=0 X93_CC_ARMS=none X93_BB_ARMS=stk X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-132-fn-stack}
B=; for f in "$H/93o-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93o-fn-tp4ab.sh"; do [ -e "$f" ] && { B=$f; break; }; done
[ -n "$B" ] || { log "RESULT fn4 132-fn-stack: VIRHE 93o-fn-tp4ab.sh not found beside $H"; exit 1; }
bash "$B"; rc=$?
# the targets, on the stack's own numbers (1000 tok/s at mb c16; 8500 tok/s at the 16k / 64k / 128k prefills)
[ "$DRY" = 1 ] && { echo "+ targets line from $O/fn4-stk-c16.txt, fn4-stk-pf16.txt, fn4-stk-pf64-128.txt"; exit $rc; }
log "RESULT fn4 132 targets: $(python3 - $O <<'PY' 2>&1 | cut -c1-600
import json, sys
O = sys.argv[1]
def mb(p):
    try:
        for l in open(f"{O}/fn4-stk-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
pf = {}
for nm in ("pf16", "pf64-128"):
    try:
        for l in open(f"{O}/fn4-stk-{nm}.txt"):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if d.get("status") == 200 and d.get("prefill_tps"): pf[d["target"]] = (d["prefill_tps"], d.get("prompt_tokens"))
    except (OSError, ValueError): pass
c16 = mb("c16").get("agg_dec_tps")
out = ["decode mb c16 %s tok/s vs 1000 (%s)" % (c16, "%+.1f %%" % ((c16 / 1000 - 1) * 100) if c16 else "n/a")]
for t in (16384, 65536, 131072):
    v = pf.get(t)
    out.append("prefill %dk %s tok/s (%s tok) vs 8500 (%s)" % (t // 1024, v[0] if v else None, v[1] if v else None, "%+.1f %%" % ((v[0] / 8500 - 1) * 100) if v else "n/a"))
print("; ".join(out))
PY
)"
exit $rc
