#!/bin/bash
# dsm-emb.sh (11.11.2026, queue-ar/dsmain/da, ds-main-port DA agent; coordinator's request 05:20 UTC) -- NOT a queue item; the body
# of 57-p4zyzzzzz-ds-iej-sz-embscreen.sh. Tag "RESULT xdae ", out runs/xdsm/da (the pilot's). THE DRAFTER-EMBEDDING SCREEN, offline,
# on the pilot's own KL dump (runs/xdsm/da/stats/kl, kept by iei for this item): the drafter reads the trunk's token_embd (q6_K in
# the .rad, 129.46 MiB a rank, embed_lookup_ggml; no table of its own). Measured on the workstation: q6_K vs the release's BF16
# embed.weight per-row relative L2 error 1.77 % (cosine 0.99984); a GGUF card measured Q3_K vs BF16 at 41.1 vs 45.5 % acceptance.
# Before any engine work (a bf16 table = +186.6 MiB a rank sharded, or 1.23 GiB replicated), the bf16 port (trainer/, validated vs
# the engine: first-token agreement 0.942) answers whether the table moves acceptance at all: datrain.py eval on the held-out
# documents (simulated fixed-5 acceptance, the same split as the pilot's training), FOUR runs in parallel, one card each:
#   GPU 0 the container's drafter + q6_K table   GPU 1 the same + the release's BF16 table (~/gpu/ds41/da/emb-bf16.bin, sha256 aa1ac979)
#   GPU 2 the pilot's best.ovl + q6_K table      GPU 3 best.ovl + the BF16 table
# then the BF16 table (1.3 GB) is deleted; the KL dump stays for 57-p4zyzzzzzzzzz-da-f-fidelity, which deletes it. ~8 min. Nothing is served; no cc_sim, no 2.9.
# DRYRUN=1 bash dsm-emb.sh : prints every command; checks the trainer image, the torchenv, the trainer (parses, CPU selftest),
#   every helper; the inputs (present, or produced by ieg / iei); no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
O=${DAE_OUT:-$R/xdsm/da}; STATSDIR=$O/stats
ITEM=${DAE_ITEM:-57-p4zyzzzzz-ds-iej-sz-embscreen}
TR=$HD/trainer; TENV=$HOME/gpu/ds41/da/torchenv; TIMG=rocm/dev-ubuntu-24.04:latest
EMB=$HOME/gpu/ds41/da/emb-bf16.bin
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
xlog() { log "RESULT xdae $*"; }
edocker() {   # edocker NAME GPU ARGS...: the trainer image on one card, the model ro, /w = the pilot's stats, /e = the BF16 table
  local n=$1 g=$2; shift 2
  local dev=""; [ "$g" != - ] && dev="--device /dev/kfd --device /dev/dri --group-add $(stat -c %g /dev/kfd) --group-add video -e HIP_VISIBLE_DEVICES=$g"
  local w=""; [ "$g" != - ] && w="-v $STATSDIR:/w -v $EMB:/e/emb-bf16.bin:ro"
  echo "docker run --rm --name dae-$n --network none -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/env:/t -e OMP_NUM_THREADS=8 $dev --ipc host --shm-size 8g -v $TENV:/env:ro -v $TR:/t:ro -v $MODELDIR:/m:ro $w --entrypoint python3 $TIMG /t/datrain.py $*"
}
trap 'docker rm -f dae-q0 dae-b0 dae-q1 dae-b1 > /dev/null 2>&1' EXIT
trap 'exit 143' TERM INT
HMISS=""
for fn in log run mkd stopfile xlog edocker; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for f in $TR/datrain.py $TR/dspark.py $TR/radread.py $MODELDIR/$M225 $TENV/torch/__init__.py; do [ -e $f ] || HMISS="$HMISS file:$f"; done
for f in $TR/datrain.py $TR/dspark.py $TR/radread.py; do python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || HMISS="$HMISS parse:$f"; done
grep -q -- "--embed-file" $TR/datrain.py || HMISS="$HMISS datrain.py(no --embed-file)"
docker image inspect $TIMG > /dev/null 2>&1 || HMISS="$HMISS image:$TIMG"
if [ -z "$HMISS" ]; then
  STT=$(timeout 600 $(edocker selftest - selftest) 2>&1)
  echo "$STT" | grep -q '"ok": true' || HMISS="$HMISS selftest:datrain.py($(echo "$STT" | tail -2 | tr '\n' ' ' | cut -c1-200))"
fi
INPUTS=""
for f in $STATSDIR/kl/kld.json $STATSDIR/kl/dump-dspark_main_normed.json $EMB; do [ -s $f ] || INPUTS="$INPUTS $f"; done
OVL=""; [ -s $STATSDIR/ovl/best.ovl ] && OVL=/w/ovl/best.ovl
if [ "$DRY" = 1 ]; then
  echo "+ item $ITEM, out $O, trainer $TIMG ($(docker image inspect -f '{{.Id}}' $TIMG 2>/dev/null | cut -c8-19)) + $TENV, table $EMB, overlay ${OVL:-none (GPU 2/3 skipped)}"
  echo "+ inputs: ${INPUTS:-all present}"
  [ -z "$HMISS" ] && echo "+ helpers: every file present and parsing, every function defined, datrain.py selftest (CPU) ok -- OK" || { echo "+ MISSING:$HMISS"; exit 1; }
else
  [ -z "$HMISS" ] || { xlog "$ITEM: VIRHE helpers missing:$HMISS"; exit 1; }
  [ -z "$INPUTS" ] || { xlog "$ITEM: VIRHE inputs missing:$INPUTS -> nothing run"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "$ITEM: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
fi
xlog "$ITEM: start | the drafter-embedding screen (offline, the bf16 port on the pilot's KL dump, held-out documents): q6_K (the container's) vs the release's BF16 table, the container's drafter and the pilot's best.ovl"
run "$(edocker q0 0 eval --device cuda:0 --head32 --data /w/kl --max-anchors 384) > $O/emb-q6-base.txt 2>&1 &"
run "$(edocker b0 1 eval --device cuda:0 --head32 --data /w/kl --max-anchors 384 --embed-file /e/emb-bf16.bin) > $O/emb-bf16-base.txt 2>&1 &"
if [ -n "$OVL" ]; then
  run "$(edocker q1 2 eval --device cuda:0 --head32 --data /w/kl --max-anchors 384 --overlay $OVL) > $O/emb-q6-ovl.txt 2>&1 &"
  run "$(edocker b1 3 eval --device cuda:0 --head32 --data /w/kl --max-anchors 384 --overlay $OVL --embed-file /e/emb-bf16.bin) > $O/emb-bf16-ovl.txt 2>&1 &"
fi
run "wait"
if [ "$DRY" != 1 ]; then
  for t in q6-base bf16-base q6-ovl bf16-ovl; do
    [ -e $O/emb-$t.txt ] || continue
    xlog "$t: $(grep -E '"event": "(embed_replaced|eval)"' $O/emb-$t.txt | cut -c5- | tr '\n' ' ' | cut -c1-1200) $(grep -m1 -E 'Traceback|Error' $O/emb-$t.txt | cut -c1-200)"
  done
  xlog "summary (held-out simulated tokens a step at fixed window 5, and p1): $(python3 -c 'import json,sys,os
O=sys.argv[1]
def ev(t):
    try:
        for l in open(os.path.join(O, "emb-%s.txt" % t)):
            if l.startswith("DAT ") and "\"event\": \"eval\"" in l: return json.loads(l[4:])
    except OSError: pass
    return None
out=[]
for base in ("base", "ovl"):
    q, b = ev("q6-" + base), ev("bf16-" + base)
    if not q or not b: continue
    for k in sorted(x for x in q if isinstance(q[x], dict) and x in b):
        out.append("%s %s: q6_K %.3f (p1 %.3f) -> BF16 %.3f (p1 %.3f), %+.2f %%" % (base, k, q[k]["tokens_a_step"], q[k]["cond"][0], b[k]["tokens_a_step"], b[k]["cond"][0], 100 * (b[k]["tokens_a_step"] / q[k]["tokens_a_step"] - 1)))
print(" | ".join(out) or "none")' $O 2>&1 | cut -c1-2400)"
fi
run "rm -f $EMB"
[ "$DRY" = 1 ] || xlog "cleanup: the BF16 table deleted; the KL dump kept for 57-p4zyzzzzzzzzz-da-f-fidelity (it deletes it) | disk $(df -h / | awk 'NR==2{print $4}') free"
xlog "$ITEM: done -- $O"
