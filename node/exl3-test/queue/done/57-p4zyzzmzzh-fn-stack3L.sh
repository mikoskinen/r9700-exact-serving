#!/bin/bash
# 150-fn-stack3L.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~65 min, one arm)
# -- coordinator 10.11.: the iq4r TP4 stack 3 at PRODUCTION'S CONTEXT SETTINGS (every earlier stack run was at 262144):
# arm stk3L of 93s-fn-tp4ab.sh = 147's stk3 (radiance-local:c16-sw9, F1n + F2a + F2b + hcw + hc_readv 11 + HC_SS + hc_upmix
# 5, MNBT 4096 + --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest; TP4 all_vram MAXSEQS 16 MTP 3 wht6) with
# --max-model-len 524288 --rope-scaling yarn:2 (production dp-a / dp-b's). Phases (93s, X93_SCREEN 0): gate, slow-state
# probe, produp pair + probe, mb c4 / c8 / c16 (180 s), THEN prefill 16k warm + fresh 64k / 128k + a fresh ~256k prompt
# (thermwin each), the 33k copy (vs 128's cold copy: yarn changes the rotary scale, so a difference is information), mb
# pf, BetterBench, the c16 card-0 profile; the restart line's KV pool / full-length-session count. Compared (cmparm,
# X93_REF stk3, no re-run) against 147.
# DRYRUN=1 bash 150-fn-stack3L.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
log "RESULT fn4 150-fn-stack3L: stack 3 at --max-model-len 524288 --rope-scaling yarn:2 (production context), vs 147"
export ARMS=${X150_ARMS:-stk3L} X93_REF=${X93_REF:-stk3} X93_REF_RERUN=0 X93_COPYREF=${X93_COPYREF:-abval2-cold}
export X93_STK_RV=${X150_RV:-11} X93_STK_UM=${X150_UM:-5}
export X93_SCREEN=0 X93_SCREEN_PF=0 X93_CC_ARMS=none X93_BB_ARMS=stk3L X93_PROF16=${X93_PROF16:-1} X93_HANG428=0 X93_ITEM=${X93_ITEM:-150-fn-stack3L}
B=; for f in "$H/93s-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93s-fn-tp4ab.sh"; do [ -e "$f" ] && { B=$f; break; }; done
[ -n "$B" ] || { log "RESULT fn4 150-fn-stack3L: VIRHE 93s-fn-tp4ab.sh not found beside $H"; exit 1; }
bash "$B"; rc=$?
# the targets, on the stack's own numbers (1000 tok/s at mb c16; 8500 tok/s at the 16k / 64k / 128k prefills)
[ "$DRY" = 1 ] && { echo "+ targets line from $O/fn4-stk3L-c16.txt, fn4-stk3L-pf16.txt, fn4-stk3L-pf64-128.txt (and 147 stk3)"; exit $rc; }
log "RESULT fn4 150 targets: $(python3 - $O <<'PY' 2>&1 | cut -c1-600
import json, sys
O = sys.argv[1]
arm = "stk3L"
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
