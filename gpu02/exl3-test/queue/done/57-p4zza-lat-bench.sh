#!/bin/bash
# 85a-lat-bench.sh (queue item, 8.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN; ~6-8 min) -- the packed
# compressed latent's sparse attention (RADIANCE_DS41_LATENT=nvfp4) against bf16, on the tuned3 libr4d (spilling) and the
# tuned3l libr4d (ds41-lat d046078, the fix), one card, at the serving shapes.
# WHY. 57-nj: t3l (t3 + LATENT=nvfp4) gated 30.26 ms at 2.9 vs t3 20.62 (+9.65 ms), 2.25 26.40 vs 15.83 (+10.5): the same
# absolute cost, so in the latent path. Cause (code objects of the tuned3 libr4d, gfx1201): every 1024-thread decode form of
# attn_sparse_sink with the packed latent spills -- head group 1/2/4/8: 192 VGPRs + 383/283/282/403 spilled, 700-916 bytes of
# scratch a lane; the bf16 forms 150-173 VGPRs, none. 38 of the 40 layers run that attention (compress ratios 2 x18 and
# 1 x20, KV shared from layers 2/8/14/20): 9.65 ms / 38 = ~250 us more a call than bf16's ~17 us. The fix (PV ring in
# 16-key slots behind an asm fence, scores unpacked a word at a time behind ds_dec_step) leaves 180/174/173/160 VGPRs and
# no spill; the same fmas in the same order (r4d_emu ds41attn: nvfp4 == bf16 bit for bit in every form, libref too), and
# the bf16 kernels' ISA is identical to tuned3's.
# WHAT. ds41lat_bench (queue-ar src/ds41lat_bench.cpp) dlopens an image's libr4d and runs attn_sparse_sink over a bf16 and a
# packed cache holding the same roundtrip latent (16 heads = a TP4 rank, window 128, 512 selections), plus entry_store bf16
# vs nvfp4 (the per-step latent write). Cases: ctx 16384 / 135000 / 524288 x ratio 2 / 1 x rows 1, 4, 8, 16 (one row a
# sequence: head groups 1, 2, 4, 8 of the direct-PV forms) and 4 rows of one sequence (a verify pass); rows 24 (the tiled
# form) at 16k / 135k. Each case on both images: old = tuned3, new = tuned3l.
# CHECKS. Every case EXACT (nvfp4 output == bf16 output bit for bit; packed entries decode to the source values) on both
# libs, and the bf16 output hash old == new (the bf16 kernels unchanged). Any MISMATCH or hash difference = VIRHE, exit 1;
# 85b-lat-serve.sh runs only after a clean 85a ($O/SUMMARY.txt "85a CLEAN").
# END. A table per case: bf16 / nvfp4 us old and new, and the 1-row step estimate 18 x d(ratio 2) + 20 x d(ratio 1).
# Every run bounded (timeout -k 15 900, --init, named, docker rm -f). The images have busybox sh, no bash (8.11.: the first
# try died "exec bash failed"): the in-container loop is POSIX sh. Guards: no radiance container, KFD idle (60 s), the
# bench binary (md5 below) and both images present.
# DRYRUN=1 bash 85a-lat-bench.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${LATB_BIN:-}
for c in "$Q/bin" "$HOME/gpu/ds41/exl3-test/queue-ar/bin" "$HOME/gpu/ds41/exl3-test/queue/done/bin" "$HOME/gpu/ds41/exl3-test/queue/bin"; do
  [ -z "$B" ] && [ -x "$c/ds41lat_bench" ] && B=$(cd "$c" && pwd)
done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin}
BMD5=${LATB_MD5:-28523010}
O=${LATB_OUT:-$HOME/gpu/ds41/exl3-test/runs/latb}
L=${LATB_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
OLD=${LATB_OLD:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab}
NEW=${LATB_NEW:-radiance-local:ds41-exl3-117-c2d9df00-fa0e9580-tuned3l-hd4a11cab}
CARD=${CARD:-0}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
LIB=/opt/radiance/share/radiance/kernels/libr4d.so
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
CASES=""
for c in 16384 135000 524288; do for r in 2 1; do
  for rs in 1:1 4:4 8:8 16:16 4:1; do CASES="$CASES $c:$r:$rs"; done
  [ $c -lt 524288 ] && CASES="$CASES $c:$r:24:24"
done; done
# the split by key count (8.11. v2): one row at 135k, ratio 2, 128 .. 2048 selections (512 is in the set above): the slope
# of nvfp4 - bf16 over the selections is the cost a packed key adds (unpack in the score and the PV), the intercept the rest
for k in 128 256 1024 2048; do CASES="$CASES 135000:2:1:1:$k"; done
benchset() {   # benchset WHICH IMAGE: every case in one bounded container
  local w=$1 img=$2 cn=latb-$1-$$ args="" x
  args="$CASES"
  run "timeout -k 15 900 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint sh $img -c 'for x in $args; do IFS=:; set -- \$x; IFS=\" \"; c=\$1; r=\$2; m=\$3; s=\$4; k=\${5:-512}; /b/ds41lat_bench --lib $LIB --tag $w --ctx \$c --ratio \$r --rows \$m --seqs \$s --topk \$k || echo \"RESULT ds41lat $w ctx \$c ratio \$r rows \$m seqs \$s topk \$k: VIRHE exit \$?\"; done' > $O/bench-$w.txt 2>&1; echo \"exit \$?\" >> $O/bench-$w.txt; docker rm -f $cn > /dev/null 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT latb: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT latb: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/ds41lat_bench" ] || { log "RESULT latb: VIRHE $B/ds41lat_bench missing"; exit 1; }
  [ "$(md5sum "$B/ds41lat_bench" | cut -c1-8)" = $BMD5 ] || { log "RESULT latb: VIRHE $B/ds41lat_bench is not md5 $BMD5"; exit 1; }
  for i in $OLD $NEW; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT latb: VIRHE image $i missing"; exit 1; }; done
  mkdir -p "$O"; rm -f "$O/SUMMARY.txt"
else
  echo "+ mkdir -p $O"
fi
log "RESULT latb: start card $CARD, $(echo $CASES | wc -w) cases on old $OLD ($(docker image inspect -f '{{.Id}}' $OLD 2>/dev/null | cut -c8-19)) and new $NEW ($(docker image inspect -f '{{.Id}}' $NEW 2>/dev/null | cut -c8-19)), bench $B/ds41lat_bench -> $O"
benchset old $OLD
benchset new $NEW
[ "$DRY" = 1 ] && { echo "+ python3 (summary of $O/bench-old.txt + bench-new.txt) > $O/SUMMARY.txt"; log "RESULT latb: dry run done"; exit 0; }
python3 - $O > $O/SUMMARY.txt 2>&1 <<'PY'
import re, sys, os
O = sys.argv[1]
pat = re.compile(r"^RESULT ds41lat (\w+) ctx (\d+) ratio (\d+) rows (\d+) seqs (\d+) heads \d+ keys \d+\+(\d+): (\w+) .*?attn bf16 ([\d.]+) us, nvfp4 ([\d.]+) us .*?entry_store bf16 ([\d.]+) us, nvfp4 ([\d.]+) us; hash bf16 (\w+) nvfp4 (\w+)")
res, bad = {}, []
for w in ("old", "new"):
    for l in open(os.path.join(O, f"bench-{w}.txt")):
        if l.startswith("RESULT ds41lat") and "VIRHE" in l: bad.append(l.strip()[:200]); continue
        m = pat.match(l)
        if not m: continue
        k = tuple(int(x) for x in m.group(2, 3, 4, 5, 6))
        res[(w,) + k] = dict(ok=m.group(7), bf=float(m.group(8)), nv=float(m.group(9)), sbf=float(m.group(10)),
                             snv=float(m.group(11)), hb=m.group(12), hn=m.group(13))
keys = sorted({k[1:] for k in res})
print("ctx     ratio rows seqs topk | old bf16    nvfp4 (x)      | new bf16    nvfp4 (x)      | store bf16/nvfp4 new | checks")
for k in keys:
    o, n = res.get(("old",) + k), res.get(("new",) + k)
    if not o or not n: bad.append(f"case {k}: missing on {'old' if not o else 'new'}"); continue
    chk = []
    if o["ok"] != "EXACT": chk.append("old " + o["ok"])
    if n["ok"] != "EXACT": chk.append("new " + n["ok"])
    if o["hb"] != n["hb"]: chk.append("bf16 hash old != new")
    if n["hn"] != n["hb"] or o["hn"] != o["hb"]: chk.append("nvfp4 hash != bf16")
    if chk: bad.append(f"case {k}: " + ", ".join(chk))
    print(f"{k[0]:7d} {k[1]:5d} {k[2]:4d} {k[3]:4d} {k[4]:4d} | {o['bf']:8.1f} {o['nv']:8.1f} ({o['nv']/o['bf']:4.2f}) | "
          f"{n['bf']:8.1f} {n['nv']:8.1f} ({n['nv']/n['bf']:4.2f}) | {n['sbf']:6.1f} / {n['snv']:6.1f}     | {'ok' if not chk else '; '.join(chk)}")
for c in sorted({k[0] for k in keys}):
    def d(w, r):
        x = res.get((w, c, r, 1, 1, 512)); return (x["nv"] - x["bf"]) if x else float("nan")
    eo, en = 18 * d("old", 2) + 20 * d("old", 1), 18 * d("new", 2) + 20 * d("new", 1)
    print(f"STEP ctx {c}: 1 row, 38 layers (18 at ratio 2, 20 at ratio 1): nvfp4 - bf16 = {eo/1000:+.2f} ms old, {en/1000:+.2f} ms new "
          f"(gate reference 2.9: +9.65 ms, 2.25: +10.5 ms at a short prompt)")
for w in ("old", "new"):   # the split by selections: least squares of nvfp4 - bf16 over topk at 1 row, 135k, ratio 2
    pts = [(k[5], res[k]["nv"] - res[k]["bf"]) for k in res if k[0] == w and k[1:5] == (135000, 2, 1, 1)]
    if len(pts) >= 3:
        n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
        sl = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        print(f"SPLIT {w}: nvfp4 - bf16 = {my - sl * mx:+.1f} us + {sl * 1000:.1f} ns a selected key (topk {sorted(x for x, _ in pts)}; "
              f"at 512: {my - sl * mx + sl * 512:+.1f} us)")
for b in bad: print("BAD", b)
print("85a CLEAN" if not bad and keys else "85a NOT CLEAN")
PY
grep -q "^85a CLEAN" $O/SUMMARY.txt || { log "RESULT latb: VIRHE $(grep -m3 -E '^BAD|Traceback|Error' $O/SUMMARY.txt | tr '\n' ' ' | cut -c1-400) (bench-old/new.txt, SUMMARY.txt in $O)"; exit 1; }
log "RESULT latb: done -- every case EXACT on both libs, bf16 hashes old == new | $(grep -E '^(STEP|SPLIT)' $O/SUMMARY.txt | tr '\n' ' ' | cut -c1-900)"
log "RESULT latb: table $O/SUMMARY.txt -- $(grep -E '^ *(16384|135000|524288) +[12] +1 +1 +512 ' $O/SUMMARY.txt | tr -s ' ' | tr '\n' '|' | cut -c1-700)"
