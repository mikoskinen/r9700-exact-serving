#!/bin/bash
# 30-tune512.sh (queue item, 6.11.2026) -- runs 30-tune512.cfg (queue/, or beside this file) on the stock PR #266 head image and
# the UD-IQ1_S .rad: per row a gated restart (bakon's line + the row's one change; host pool from runs/pr266/base.env, which
# 20-bakon.sh writes), then the row's phases. The base row is NOT re-run: 20-bakon.sh's own outputs are the base. At the end
# every row is compared with the base per group (cfg SCORE) and runs/pr266/best.env is written for 40-best.sh.
#   BUDGET_MIN=190   wall clock for the rows; a row whose estimate (13 min, +1.5 per pf/turns) no longer fits is skipped
#   ROWS=a,b         only these tags (default: every row)          KV_MIN_TOK=524288  a variant may not cut the KV pool below
#   DRYRUN=1         print, write nothing                                             min(this, base) tokens to win its group
# Estimate: 13 rows x 13-15 min (evict + cold start ~4 min, warm-up 1.5, probe 1, c4 3.6, c8 3.7, +1.5 per pf/turns) = ~3 h 10 min;
# with the default budget (190 min) the last rows (pin5, hp19456) may be skipped.
# Stop between rows: touch runs/sweep.stop.
. "$(cd "$(dirname "$0")" && pwd)/../lib-pr266.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-pr266.sh" || { echo "lib-pr266.sh not found"; exit 1; }
CFG=$QD/30-tune512.cfg; [ -e $CFG ] || CFG=$ME/30-tune512.cfg
[ -e $CFG ] || { log "RESULT pr266 30-tune512: VIRHE $CFG missing"; exit 1; }
BUDGET_MIN=${BUDGET_MIN:-190}; KV_MIN_TOK=${KV_MIN_TOK:-524288}; T0=$(date +%s)
BASE_HP=$(sed -n 's/^HOSTPOOL=//p' $O/base.env 2>/dev/null); BASE_HP=${BASE_HP:-4096}
[ "$DRY" = 1 ] || [ -e $O/bakon-probe.txt ] || log "RESULT pr266 30-tune512: WARNING no 20-bakon outputs in $O -- the rows run, the choice has no base"
log "RESULT pr266 30-tune512: start (base host pool $BASE_HP, budget $BUDGET_MIN min, cfg $CFG)"
while IFS="|" read -r TAG ENVS PH X2 GRP SC; do
  { [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ]; } && continue
  [ -n "${ROWS:-}" ] && [[ ",$ROWS," != *",$TAG,"* ]] && continue
  stopfile && { log "RESULT pr266 30-tune512: sweep.stop before $TAG -> stop"; break; }
  est=14; [[ $PH == *pf* ]] && est=$((est + 2)); [[ $PH == *turns* ]] && est=$((est + 2))
  left=$(( BUDGET_MIN * 60 - ($(date +%s) - T0) ))
  [ $(( est * 60 )) -gt $left ] && { log "RESULT pr266 30-tune512 $TAG: skipped (estimate $est min > $((left / 60)) min left)"; continue; }
  if [[ $X2 == *"/stats/bakon.json"* ]] && [ "$DRY" != 1 ] && [ ! -s $STATSDIR/bakon.json ]; then
    log "RESULT pr266 30-tune512 $TAG: skipped (no $STATSDIR/bakon.json from 20-bakon)"; continue; fi
  ( # one row in a subshell: its variables do not leak into the next
    export HOSTPOOL=$BASE_HP NGRAM=disk CKINT=8192 MNBT=8192 NST= MAXSEQS= MML= XARGS=
    for kv in $ENVS; do export "$kv"; done
    [ "$HOSTPOOL" = "$BASE_HP" ] && [[ $ENVS == *HOSTPOOL* ]] && { log "RESULT pr266 30-tune512 $TAG: skipped (host pool $HOSTPOOL = base)"; exit 0; }
    export XARGS="$X2"
    evict   # every row starts from the same cold page cache as the base (20-bakon evicts first): the Engram disk tier reads through it
    gate tune-$TAG $(guard_lim); rc=$?
    [ $rc = 1 ] && { log "RESULT pr266 30-tune512 $TAG: VIRHE restart failed ($(desc))"; exit 0; }
    [ $rc = 2 ] && { log "RESULT pr266 30-tune512 $TAG: VIRHE slow state after one restart -> row skipped"; exit 0; }
    for p in ${PH//,/ }; do
      case $p in probe) probe tune-$TAG ;; c4|c8|pf|turns) mbp tune-$TAG $p ;; esac
    done
    run "docker logs $CN > $O/tune-$TAG-engine.log 2>&1"
    log "RESULT pr266 30-tune512 $TAG: done ($(desc); group $GRP, score $SC)"
  ) < /dev/null
done < $CFG
down
# ---- the choice: per group, the variant that beats the base on the group's score by >= 2 % (and keeps c4 >= 0.95 x base and
# the KV pool >= min(KV_MIN_TOK, base)); 'incl' groups are taken unless they regress. Writes best.env (sourced by 40-best.sh).
SEL=$(cat <<'PY'
import json, os, re, sys
O, cfg, kvmin, dry, base_hp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
def J(f, pre):
    try: return [json.loads(l[len(pre):]) for l in open(f) if l.startswith(pre)]
    except OSError: return []
def metrics(tag):
    m = {}
    for x in J(f"{O}/{tag}-probe.txt", "J "): m["code" if x.get("prompt", "").startswith("code") else "essay"] = x.get("decode_tps")
    for ph in ("c4", "c8", "pf", "turns"):
        for x in J(f"{O}/{tag}-{ph}.txt", "MB "):
            if ph in ("c4", "c8"): m[ph] = x.get("agg_dec_tps")
            elif ph == "pf": m["pf"], m["keep"] = x.get("prefill_tps"), x.get("decode_kept")
            else: m["hit"] = x.get("ttft_hit_mean")
    try:
        t = open(f"{O}/start-{tag}.log").read()
        k = re.search(r"kv cache +[0-9.]+ GiB a card: (\d+) tokens", t)
        m["kv"] = int(k.group(1)) if k else None
    except OSError: m["kv"] = None
    return m
base = metrics("bakon")
rows = []
for l in open(cfg):
    l = l.rstrip("\n")
    if not l or l.startswith("#"): continue
    tag, env, ph, x2, grp, sc = (l.split("|") + [""] * 6)[:6]
    rows.append(dict(tag=tag, env=env, x2=x2, grp=grp, sc=sc, m=metrics("tune-" + tag)))
def ok_kv(m):
    floor = min(kvmin, base.get("kv") or kvmin)
    return m.get("kv") is None or m["kv"] >= floor
chosen, notes = {}, []
for grp in dict.fromkeys(r["grp"] for r in rows):
    best, bscore = None, None
    for r in (r for r in rows if r["grp"] == grp):
        m, sc = r["m"], r["sc"]
        if not m.get("c4"): notes.append(f"{r['tag']}: no data"); continue
        c4ok = base.get("c4") is None or m["c4"] >= 0.95 * base["c4"]
        if sc == "incl":
            hitok = base.get("hit") is None or m.get("hit") is None or m["hit"] <= 1.10 * base["hit"]
            if c4ok and hitok and ok_kv(m): best, bscore = r, 0
            notes.append(f"{r['tag']}: c4 {m['c4']} (base {base.get('c4')}), hit {m.get('hit')} (base {base.get('hit')}) -> {'take' if best is r else 'drop'}")
            continue
        key = "keep" if sc == "keep" else sc
        v, b = m.get(key), base.get(key)
        if v is None or b is None: notes.append(f"{r['tag']}: {key} missing"); continue
        if sc == "keep" and (m.get("pf") or 0) < 0.6 * (base.get("pf") or 0): notes.append(f"{r['tag']}: pf {m.get('pf')} < 0.6 x base"); continue
        if (sc != "c4" and not c4ok) or not ok_kv(m): notes.append(f"{r['tag']}: c4 {m['c4']} / kv {m.get('kv')} fails the guard"); continue
        gain = v / b if b else 0
        notes.append(f"{r['tag']}: {key} {v} vs base {b} ({(gain - 1) * 100:+.1f} %)")
        if gain >= 1.02 and (bscore is None or v > bscore): best, bscore = r, v
    if best: chosen[grp] = best
env = {"HOSTPOOL": base_hp, "NGRAM": "disk", "CKINT": "8192", "MNBT": "8192", "NST": "", "MAXSEQS": "", "MML": ""}
xargs = []
for grp, r in chosen.items():
    for kv in r["env"].split():
        k, v = kv.split("=", 1); env[k] = v
    if r["x2"]: xargs.append(r["x2"])
single = max((r for r in rows if r["m"].get("c4")), key=lambda r: r["m"]["c4"], default=None)
out = ["# best.env written by 30-tune512.sh: " + ", ".join(f"{g}={r['tag']}" for g, r in chosen.items()) + (" (none beat the base)" if not chosen else "")]
out += [f"{k}={v}" for k, v in env.items()] + ["XARGS='" + " ".join(xargs) + "'"]
out += ["# fallback if this combination does not start: the single row with the highest c4"]
if single:
    fb = {"HOSTPOOL": base_hp, "NGRAM": "disk", "CKINT": "8192", "MNBT": "8192", "NST": "", "MAXSEQS": "", "MML": ""}
    for kv in single["env"].split():
        k, v = kv.split("=", 1); fb[k] = v
    out += [f"FB_TAG={single['tag']}"] + [f"FB_{k}={v}" for k, v in fb.items()] + ["FB_XARGS='" + single["x2"] + "'"]
print("CHOICE " + " | ".join(notes))
print("BEST " + ", ".join(f"{g}={r['tag']}" for g, r in chosen.items()) + f" -> {' '.join(f'{k}={v}' for k, v in env.items())} XARGS='{' '.join(xargs)}'")
if not dry:
    open(f"{O}/best.env", "w").write("\n".join(out) + "\n")
PY
)
if [ "$DRY" = 1 ]; then echo "+ python3 (choice) $O $CFG $KV_MIN_TOK -> $O/best.env"; else
  python3 -c "$SEL" $O $CFG $KV_MIN_TOK $DRY $BASE_HP > $O/tune-choice.txt 2>&1
  log "RESULT pr266 30-tune512 choice: $(grep '^BEST ' $O/tune-choice.txt | cut -c6- | cut -c1-600) || $(grep '^CHOICE ' $O/tune-choice.txt | cut -c8- | cut -c1-1500)$(grep -q '^BEST ' $O/tune-choice.txt || tail -3 $O/tune-choice.txt | tr '\n' ' ')"
fi
log "RESULT pr266 30-tune512: done in $(( ($(date +%s) - T0) / 60 )) min"
