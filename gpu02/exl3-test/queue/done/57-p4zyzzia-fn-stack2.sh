#!/bin/bash
# 137-fn-stack2.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~60 min, one arm)
# -- the SECOND stack confirmation (coordinator 10.11., 132's format): arm stk2 of 93p-fn-tp4ab.sh beside this file =
#   radiance-local:c16-sw7 (fn4-c16 803f418), F1n + F2a + F2b + hcw (hc_writevec 1) + hc_readv 3 + hc_upmix 5 (136: prefill
#   B/A 1.0207 / 1.0202 / 1.0199, every round 1.019-1.022; decode flat; identity a + b IDENTICAL), MNBT 4096 + --prefill-span 2,
#   --prefill-interleave 0, --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6 210 W; no SWITCHABLE.
# Phases (93p, X93_SCREEN 0): gate (1-user), slow-state probe, produp pair + probe, mb c4 / c8 / c16 (180 s each) with engine
# stats, THEN prefill 16k warm + fresh 64k / 128k (a thermal line each), the 33k copy byte for byte against 128's cold copy
# (X93_COPYREF abval2-cold), mb pf, BetterBench --quick (1-user + the sweeps), the c16 card-0 profile; no cc_sim. Stamp
# analyses run after the arm (decode -> prefill back to back). Compared (cmparm, X93_REF stk, no re-run) against 132's stk
# (mb c16 1012.9, prefill 16k / 64k / 128k 8255.6 / 8352.5 / 8263.4) and the targets (mb c16 >= 1000, prefill >= 8500 warm).
# DRYRUN=1 bash 137-fn-stack2.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 137-fn-stack2: the stack = F1n + F2a + F2b + hcw + hc_readv 3 + hc_upmix 5 on c16-sw7 (136: prefill +2.0 % every round)"
export ARMS=${X137_ARMS:-stk2} X93_REF=${X93_REF:-stk} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-abval2-cold}
export X93_STK_RV=${X137_RV:-3} X93_STK_UM=${X137_UM:-5}
export X93_SCREEN=0 X93_SCREEN_PF=0 X93_CC_ARMS=none X93_BB_ARMS=stk2 X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-137-fn-stack2}
B=; for f in "$H/93p-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93p-fn-tp4ab.sh"; do [ -e "$f" ] && { B=$f; break; }; done
[ -n "$B" ] || { log "RESULT fn4 137-fn-stack2: VIRHE 93p-fn-tp4ab.sh not found beside $H"; exit 1; }
bash "$B"; rc=$?
# the targets, on the stack's own numbers (1000 tok/s at mb c16; 8500 tok/s at the 16k / 64k / 128k prefills)
[ "$DRY" = 1 ] && { echo "+ targets line from $O/fn4-stk2-c16.txt, fn4-stk2-pf16.txt, fn4-stk2-pf64-128.txt (and 132 stk)"; exit $rc; }
log "RESULT fn4 137 targets: $(python3 - $O <<'PY' 2>&1 | cut -c1-600
import json, sys
O = sys.argv[1]
arm = "stk2"
def mb(p):
    try:
        for l in open(f"{O}/fn4-{arm}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
pf = {}
for nm in ("pf16", "pf64-128"):
    try:
        for l in open(f"{O}/fn4-{arm}-{nm}.txt"):
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
