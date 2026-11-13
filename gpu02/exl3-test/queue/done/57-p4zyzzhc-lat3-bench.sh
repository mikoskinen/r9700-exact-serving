#!/bin/bash
# 85a3-lat-bench.sh (queue item, 8.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN; ~15 min) -- the packed
# compressed latent's sparse attention (RADIANCE_DS41_LATENT=nvfp4) against bf16, on the v2 tuned3l libr4d (ds41-lat
# c2d9df0, 85a's "new") and the v3 one (ds41-lat 641197d), one card, at the serving shapes; then the v3 decode forms A/B.
# WHY. 85a v2 (57-p4zza, 8.11.): STEP +3.62 ms (16k) / +2.70 (135k) / +1.82 (524k) against the ~0.5 target; one row at 135k,
# ratio 2: topk 128 / 256 / 512 / 1024 / 2048 = 16.6 / 23.8 / 110.5 / 214.9 / 199.7 us (bf16 9.5 / 11.2 / 15.6 / 28.4 /
# 62.0). The 256 -> 512 jump is not a form switch: one row x 16 heads = 16 row-heads is the 1-head direct-PV form at every
# topk (LDS 6.6 -> 10 KiB, 180 VGPRs, no spill either way); and the same shape ran 65.4 us at 135k ratio 1 and at 524k
# (16k and 135k ratio 2: 110-111) -- bimodal, so part of it is not the selections. What the form does per key: its PV walked
# the packed blocks in 16-key slots (two registers a key: code and scale byte), each slot's loads behind the previous
# slot's sums -- twice the plain walk's steps -- at 1629 instructions a block (50.9 a key a PV thread; the plain walk 12),
# on the 8 of the WG's 32 waves that own PV (one head a WG at one row). A latency-bound walk of nk/16 steps (~1-1.5 us a
# step fast, ~3 slow) fits all five points.
# v3 (ds41-lat 641197d, exact: valchk3 131072 values bit for bit, r4d_emu ds41attn 1024 / 2048 / 480 selections over head
# groups 1/2/4/8): ds_e4m3_half by shift + one exact multiply (15 -> ~7 instructions; score loop 1530 -> 1360 an iteration);
# the packed PV a 32-key block a step with a key's two bytes in one register (d16 lo/hi), two blocks in flight as the plain
# walk; one head a PV thread decodes by LDS tables (code byte -> two (float) 2|e2m1| with sign, scale byte -> s/2): PV 50.9
# -> 31.8 instructions a key; head group 8 keeps the register decode (no spill: 179 VGPRs). The bf16 kernels' ISA unchanged.
# EXPECTED (estimate, not measured): one row 135k ratio 2 topk 512 nvfp4 ~30-45 us (v2 65-110), the PV steps halved and its
# VALU -38 %; STEP +1.0 .. +1.6 ms at 135k (v2 +2.70). If the bimodal ~+45 us state stays, it is outside these kernels.
# WHAT. ds41lat_bench (queue-ar src/ds41lat_bench.cpp, md5 below): 85a's cases on both images, plus
#   sweep    one row at 135k ratio 2: topk 128 256 320 384 448 512 640 768 1024 1536 2048 (where a cliff sits, if any)
#   repeats  135k:2:1:1:512 seeds 1 (x3), 2, 3; 16k:2:1:1:512 x2 more; 524k:1:1:1:512 x2 more (fresh process each:
#            run-to-run and data-to-data spread of the 65 / 110 us modes)
#   forms    the v3 image only, RADIANCE_DS41_NVFORM=0..9 (0-6 the rule's forms, 7 one head tiled at 512 threads, 8 four
#            heads direct at 512 threads (two heads a PV thread), 9 two heads direct at 512) at 135k ratio 2: 1 row topk
#            512 and 2048, 4 rows of one sequence, 16 rows x 16 sequences (topk 512) -- a form the LDS refuses falls back
#            to the rule (same numbers as the rule's form)
# CHECKS. Every line EXACT (nvfp4 == bf16 bit for bit; packed entries decode to the source values), nvfp4 hash == bf16 hash
# per line, bf16 hash old == new per case (the bf16 kernels unchanged), every forced form's hash == the case's bf16 hash.
# Any MISMATCH / hash difference = VIRHE, exit 1; 85b3-lat-serve.sh runs only after "85a CLEAN" in $O/SUMMARY.txt.
# END. The 85a table (median over repeats, min-max), STEP and SPLIT lines old/new, the sweep, the repeats, the forms table.
# Every run bounded (timeout -k 15 900, --init, named, docker rm -f). The images have busybox sh, no bash: the in-container
# loop is POSIX sh. Guards: no radiance container, KFD idle (60 s), the bench binary (md5 below) and both images present.
# DRYRUN=1 bash 85a3-lat-bench.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${LATB_BIN:-}
for c in "$Q/bin" "$HOME/gpu/ds41/exl3-test/queue-ar/bin" "$HOME/gpu/ds41/exl3-test/queue/done/bin" "$HOME/gpu/ds41/exl3-test/queue/bin"; do
  [ -z "$B" ] && [ -x "$c/ds41lat_bench" ] && B=$(cd "$c" && pwd)
done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin}
BMD5=${LATB_MD5:-28523010}
O=${LATB_OUT:-$HOME/gpu/ds41/exl3-test/runs/latb3}
L=${LATB_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
OLD=${LATB_OLD:-radiance-local:ds41-exl3-117-c2d9df00-fa0e9580-tuned3l-hd4a11cab}
NEW=${LATB_NEW:-radiance-local:ds41-exl3-117-641197d8-fa0e9580-tuned3l-hd4a11cab}
CARD=${CARD:-0}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
LIB=/opt/radiance/share/radiance/kernels/libr4d.so
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
# a case: ctx:ratio:rows:seqs[:topk[:seed]] (topk 512, seed 1 by default)
CASES=""
for c in 16384 135000 524288; do for r in 2 1; do
  for rs in 1:1 4:4 8:8 16:16 4:1; do CASES="$CASES $c:$r:$rs"; done
  [ $c -lt 524288 ] && CASES="$CASES $c:$r:24:24"
done; done
for k in 128 256 320 384 448 640 768 1024 1536 2048; do CASES="$CASES 135000:2:1:1:$k"; done
CASES="$CASES 135000:2:1:1:512:1 135000:2:1:1:512:1 135000:2:1:1:512:2 135000:2:1:1:512:3 16384:2:1:1:512:1 16384:2:1:1:512:1 524288:1:1:1:512:1 524288:1:1:1:512:1"
FCASES="135000:2:1:1:512 135000:2:1:1:2048 135000:2:4:1:512 135000:2:16:16:512"
FORMS=${LATB_FORMS:-"0 1 2 3 4 5 6 7 8 9"}
benchset() {   # benchset TAG IMAGE "CASES" [NVFORM]: every case in one bounded container
  local w=$1 img=$2 args="$3" f=${4:-} cn=latb3-$1-$$ e=""
  [ -n "$f" ] && e="-e RADIANCE_DS41_NVFORM=$f"
  run "timeout -k 15 900 docker run --rm --init --stop-timeout 10 --name $cn $DEV $e -v $B:/b:ro --entrypoint sh $img -c 'for x in $args; do IFS=:; set -- \$x; IFS=\" \"; c=\$1; r=\$2; m=\$3; s=\$4; k=\${5:-512}; sd=\${6:-1}; /b/ds41lat_bench --lib $LIB --tag $w --ctx \$c --ratio \$r --rows \$m --seqs \$s --topk \$k --seed \$sd || echo \"RESULT ds41lat $w ctx \$c ratio \$r rows \$m seqs \$s topk \$k: VIRHE exit \$?\"; echo \"SEED \$sd\"; done' > $O/bench-$w.txt 2>&1; echo \"exit \$?\" >> $O/bench-$w.txt; docker rm -f $cn > /dev/null 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT latb3: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT latb3: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/ds41lat_bench" ] || { log "RESULT latb3: VIRHE $B/ds41lat_bench missing"; exit 1; }
  [ "$(md5sum "$B/ds41lat_bench" | cut -c1-8)" = $BMD5 ] || { log "RESULT latb3: VIRHE $B/ds41lat_bench is not md5 $BMD5"; exit 1; }
  for i in $OLD $NEW; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT latb3: VIRHE image $i missing"; exit 1; }; done
  mkdir -p "$O"; rm -f "$O/SUMMARY.txt" "$O"/bench-*.txt
else
  echo "+ mkdir -p $O"
fi
log "RESULT latb3: start card $CARD, $(echo $CASES | wc -w) cases on old (v2) $OLD ($(docker image inspect -f '{{.Id}}' $OLD 2>/dev/null | cut -c8-19)) and new (v3) $NEW ($(docker image inspect -f '{{.Id}}' $NEW 2>/dev/null | cut -c8-19)), then forms [$FORMS] x $(echo $FCASES | wc -w) cases on new; bench $B/ds41lat_bench -> $O"
benchset old $OLD "$CASES"
benchset new $NEW "$CASES"
for f in $FORMS; do benchset f$f $NEW "$FCASES" $f; done
[ "$DRY" = 1 ] && { echo "+ python3 (summary of $O/bench-*.txt) > $O/SUMMARY.txt"; log "RESULT latb3: dry run done"; exit 0; }
python3 - $O "$FORMS" > $O/SUMMARY.txt 2>&1 <<'PY'
import re, sys, os, statistics
O, forms = sys.argv[1], sys.argv[2].split()
pat = re.compile(r"^RESULT ds41lat (\w+) ctx (\d+) ratio (\d+) rows (\d+) seqs (\d+) heads \d+ keys \d+\+(\d+): (\w+) .*?attn bf16 ([\d.]+) us, nvfp4 ([\d.]+) us .*?entry_store bf16 ([\d.]+) us, nvfp4 ([\d.]+) us; hash bf16 (\w+) nvfp4 (\w+)")
res, bad = {}, []
def read(w):
    pend = None
    p = os.path.join(O, f"bench-{w}.txt")
    if not os.path.exists(p): bad.append(f"{p} missing"); return
    for l in open(p):
        if l.startswith("RESULT ds41lat") and "VIRHE" in l: bad.append(l.strip()[:200]); pend = None; continue
        m = pat.match(l)
        if m:
            pend = (tuple(int(x) for x in m.group(2, 3, 4, 5, 6)),
                    dict(ok=m.group(7), bf=float(m.group(8)), nv=float(m.group(9)), sbf=float(m.group(10)),
                         snv=float(m.group(11)), hb=m.group(12), hn=m.group(13)))
            continue
        if l.startswith("SEED ") and pend:
            k, d = pend; res.setdefault((w,) + k + (int(l.split()[1]),), []).append(d); pend = None
for w in ["old", "new"] + [f"f{f}" for f in forms]: read(w)
med = lambda v: statistics.median(v)
def agg(w, k, s=1):
    L = res.get((w,) + k + (s,)); return L
for (w, *k), L in res.items():
    for d in L:
        if d["ok"] != "EXACT": bad.append(f"{w} {tuple(k)}: {d['ok']}")
        if d["hn"] != d["hb"]: bad.append(f"{w} {tuple(k)}: nvfp4 hash != bf16 hash")
    if len({d["hb"] for d in L}) > 1: bad.append(f"{w} {tuple(k)}: bf16 hash differs between repeats")
keys = sorted({k[1:] for k in res if k[0] in ("old", "new")})
print("ctx     ratio rows seqs topk seed n | v2 bf16   nvfp4 (x)  [min-max]        | v3 bf16   nvfp4 (x)  [min-max]        | store v3 | checks")
for k in keys:
    o, n = res.get(("old",) + k), res.get(("new",) + k)
    if not o or not n: bad.append(f"case {k}: missing on {'old' if not o else 'new'}"); continue
    chk = []
    if o[0]["hb"] != n[0]["hb"]: chk.append("bf16 hash v2 != v3"); bad.append(f"case {k}: bf16 hash v2 != v3")
    f = lambda L, x: med([d[x] for d in L])
    rng = lambda L: f"[{min(d['nv'] for d in L):6.1f}-{max(d['nv'] for d in L):6.1f}]"
    print(f"{k[0]:7d} {k[1]:5d} {k[2]:4d} {k[3]:4d} {k[4]:4d} {k[5]:4d} {len(n)} | {f(o,'bf'):7.1f} {f(o,'nv'):7.1f} ({f(o,'nv')/f(o,'bf'):5.2f}) {rng(o)} | "
          f"{f(n,'bf'):7.1f} {f(n,'nv'):7.1f} ({f(n,'nv')/f(n,'bf'):5.2f}) {rng(n)} | {f(n,'snv'):5.1f}    | {'ok' if not chk else '; '.join(chk)}")
def nvd(w, c, r, rows=1, seqs=1, topk=512, seed=1):
    L = res.get((w, c, r, rows, seqs, topk, seed)); return med([d["nv"] - d["bf"] for d in L]) if L else float("nan")
for c in sorted({k[0] for k in keys}):
    eo, en = 18 * nvd("old", c, 2) + 20 * nvd("old", c, 1), 18 * nvd("new", c, 2) + 20 * nvd("new", c, 1)
    print(f"STEP ctx {c}: 1 row, 38 layers (18 at ratio 2, 20 at ratio 1): nvfp4 - bf16 = {eo/1000:+.2f} ms v2, {en/1000:+.2f} ms v3 "
          f"(85a v2 run: +3.62 / +2.70 / +1.82 ms at 16k / 135k / 524k; the gate reference 2.9 +9.65, 2.25 +10.5 at a short prompt)")
for w in ("old", "new"):
    pts = sorted((k[4], med([d["nv"] - d["bf"] for d in L])) for k, L in ((kk[1:], v) for kk, v in res.items() if kk[0] == w)
                 if k[0:4] == (135000, 2, 1, 1) and k[5] == 1)
    if len(pts) >= 3:
        n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
        sl = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        print(f"SPLIT {'v2' if w == 'old' else 'v3'}: nvfp4 - bf16 = {my - sl * mx:+.1f} us + {sl * 1000:.1f} ns a selected key "
              f"(topk {[x for x, _ in pts]}; at 512: {my - sl * mx + sl * 512:+.1f} us)")
        print(f"SWEEP {'v2' if w == 'old' else 'v3'} (135k ratio 2, 1 row; topk: nvfp4 - bf16 us): " +
              " ".join(f"{x}:{y:+.1f}" for x, y in pts))
print("REPEATS (nvfp4 us, every fresh process; v2 | v3):")
for k in keys:
    if k[0:4] in ((135000, 2, 1, 1), (16384, 2, 1, 1), (524288, 1, 1, 1)) and k[4] == 512:
        o, n = res.get(("old",) + k, []), res.get(("new",) + k, [])
        print(f"  ctx {k[0]} ratio {k[1]} seed {k[5]}: " + " ".join(f"{d['nv']:.1f}" for d in o) + " | " + " ".join(f"{d['nv']:.1f}" for d in n))
fk = sorted({k[1:6] for k in res if k[0].startswith("f")})
if fk:
    print("FORMS (v3, RADIANCE_DS41_NVFORM; nvfp4 us; 'rule' = the unforced v3 form; a refused form falls back to the rule):")
    print("  case (ctx ratio rows seqs topk)  rule   " + " ".join(f"{('f' + f):>7s}" for f in forms))
    for k in fk:
        rule = res.get(("new",) + k + (1,))
        row = f"  {str(k):30s} {med([d['nv'] for d in rule]) if rule else float('nan'):6.1f} "
        for f in forms:
            L = res.get((f"f{f}",) + k + (1,))
            row += f" {med([d['nv'] for d in L]):7.1f}" if L else "       -"
            if L and rule and L[0]["hn"] != rule[0]["hb"]: bad.append(f"form {f} {k}: hash != the case's bf16")
        print(row)
for b in bad: print("BAD", b)
print("85a CLEAN" if not bad and keys else "85a NOT CLEAN")
PY
grep -q "^85a CLEAN" $O/SUMMARY.txt || { log "RESULT latb3: VIRHE $(grep -m3 -E '^BAD|Traceback|Error' $O/SUMMARY.txt | tr '\n' ' ' | cut -c1-400) (bench-*.txt, SUMMARY.txt in $O)"; exit 1; }
log "RESULT latb3: done -- every case EXACT on both libs and every form, bf16 hashes v2 == v3 | $(grep -E '^(STEP|SPLIT|SWEEP)' $O/SUMMARY.txt | tr '\n' ' ' | cut -c1-1200)"
log "RESULT latb3: forms -- $(sed -n '/^FORMS/,/^BAD\|^85a/p' $O/SUMMARY.txt | grep -E '^  \(' | tr -s ' ' | tr '\n' '|' | cut -c1-900)"
log "RESULT latb3: table $O/SUMMARY.txt -- $(grep -E '^ *(16384|135000|524288) +[12] +1 +1 +512 +1 ' $O/SUMMARY.txt | tr -s ' ' | tr '\n' '|' | cut -c1-700)"
