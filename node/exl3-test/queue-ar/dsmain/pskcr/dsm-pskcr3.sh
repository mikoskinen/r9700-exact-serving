#!/bin/bash
# dsm-pskcr3.sh (12.11.2026, queue-ar/dsmain/pskcr, ds-main-port pskcr2 builder) -- NOT a queue item; the body of
#   57-p4zyzzzzz-ds-r00zzz-cz-v-scr3.sh. = dsm-pskcr2.sh f2b81d07 (not edited) with its helper pcr3.py (= pcr2.py a7c03dc3 + the
#   ds41-pskcr2 rules and the turn chains; pcr2.py not edited); pskld.py the same pskl20 copy (dca443b6). Log tag "RESULT xcr ",
#   out runs/xdsm/pskcr3 (XCR_OUT).
# WHY (r00zzz-cz-tz-scr2, 12.11. 03:29-03:50: identity 33 / 33, guards OK, tail-only cold -12.8 % at 150k): DEFECT 1 -- a request
#   served by a cone replay kept NO snapshot (Scheduler::psk_ckpt dropped every replay chunk's, the replay's end H included, and a
#   replayed prompt's tail snapshot IS H whenever it ends within a grid step of its match), so the next turn of the conversation
#   replayed again: HIT turn:40960 B 1.21 s, C / T 2.64 / 2.67 s (8192 replayed again at 34816). DEFECT 2 -- hit 24576 C +0.76 s
#   (round 1: 20480 replayed to 18432 and kept no snapshot there, so 24576 replayed [10240, 18432) where B resumed at its 18432
#   snapshot) and hit 4096 C +0.32 s (a replay of the first chunk [0, 2048) over all 40 layers' cones of 2048 is slower than its
#   cold chunk; the same holds for any grid point one step over a snapshot: two replay passes ~1.36-1.41 s vs one chunk
#   ~1.0-1.26 s).
# IMAGE ds41-pskcr2 (= ds41-pskcr e9e529d + d4540a8: psk_ckpt keeps the snapshot at a replay's end H -- the state the request
#   itself goes on from; psk_cr_resume replays only where the grid point lies MORE than one grid step (4096) over the snapshot the
#   hit resumes at (0: none), else the hit resumes there / cold (the cheaper exact path); /stats psk += cr_cheaper,
#   cr_end_snapshots). deepseek41.so is e9e529d's source unchanged.
# ONE NST 5 START = stack4's start exactly (as dsm-pskcr2.sh) + RADIANCE_LOG_STEPS=1. Arms in process (POST /admin/fn4, read back):
#   B = ds41_psk 2 k 4, ds41_psk_cr 0; C = ds41_psk 2 k 4 + ds41_psk_cr 1; T = ds41_psk 4 (tail-only) + ds41_psk_cr 1;
#   XCR_L20 / XCR_PF default 0 (PSK-CR measured alone, as scr2).
# EVERY ARM ON ITS OWN TEXT WITH A NONCE: each section's prompts open with the line "# nonce ARM-EPOCH" (pcr3.py SPEC@NONCE).
# ORDER: gate1 -> modes -> VRAM -> switch note -> warm-up answers -> smoke under T -> COUNTER GUARD (per arm, the baseline AFTER
#   the arm switch as dsm-pskcr2.sh: reset, cold 16k, a 12288 hit -- C / T must replay (grid point 10240, no snapshot under it)
#   and take the end snapshot (cr_end_snapshots), B none of it) -> HIT CASES per arm on its own text (reset, cold 64k, then with
#   no reset hits at 40960 / 20480 / 24576 / 4096 -- scr2's round-1 order, 24576 right after 20480; each hit's expected resume
#   from its own prompt tokens and the snapshots the text holds, pcr3.py expect: C / T replay only more than a grid step over a
#   snapshot) -> B's COLD references + IDENTITY -> TURN CHAINS per arm, two kinds: chR = reset, cold 64k, turn 1 = a hit at 43500
#   (its grid point 38912 / 43008 lies two or three steps over K4's anchor 30720: C and T replay, B falls back), turns 2-4 =
#   previous prompt + its answer + a new user message; chA = reset, turn 1 COLD at 48000, turns 2-4 the same -- per turn cached
#   (asserted: pcr3.py expect / expect_turn), cr_hits / rows_replayed deltas (C / T: 0 after turn 1 -- else VIRHE), snapshots,
#   TTFT -> B's COLD references of every turn's full text + IDENTITY -> COLD TTFT T B B T at 135000 (~150k tok) -> SUMMARY.
# ~27 min (estimate): start + checks + smoke ~6, guard ~1, hit cases ~2.5 + refs ~2.5, chains ~3 + refs ~5, cold TTFT ~2.5,
#   arm switches / resets / log / summary ~4. Budget <= ~35 min.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact by design:
#   bytes measured and reported, a difference is reported, never hidden.
# Disk: >= XCR_DISKMIN_MB (11000) free on / before the start; engine logs written gzipped only.
# Knobs: XCR_IMAGE, XCR_ITEM, XCR_OUT, XCR_L20 (0), XCR_PF (0), XCR_PSKK (4), XCR_OCC (6), XCR_OCCK (32), XCR_DSA (85),
#   XCR_HDEP (64000), XCR_HITS ("40960 20480 24576 4096"), XCR_CRD1 (43500), XCR_CAD (48000), XCR_NTURN (4), XCR_DEPTHS (135000),
#   XCR_TORDER (TBBT), XCR_DISKMIN_MB (11000), XCR_SIDECAR, XCR_SIDESHA.
# DRYRUN=1 bash dsm-pskcr3.sh : prints every command; checks the image literals (the 45-switch registry, PSK-CR's literals, the
#   ds41-pskcr2 /stats keys, pskcr2-PROVENANCE), every helper (files, parse, selftests, the text offsets), the sidecars, the yaml;
#   no server, no GPU, writes nothing.
# ---- dsm-pskcr2.sh's header (its body is this one's base) ----
# 12.11.2026 head fix (backup dsm-pskcr2.sh.bak-20261112-head = 595eca02): the SHARED HEAD section only, helpers pcr2.py (= pcr.py
#   b2f9646b + the head's 6th spec field, hcal / hjoin / hexpect; pcr.py not edited). cz-scr's head VIRHE 'cached 38912, expected
#   14336' was the test's artefact: the expected resumes came from HEADT 24576 planned tokens while the head texts tokenise far
#   denser (P1 42444 tok for 24576 + 3000 planned; real shared head ~39632) -- every head prompt was IDENTICAL to its cold ref.
#   Now: each arm's head prompts open with their own line "# head ARM-EPOCH-0" (texts no other section / arm shares); before P1
#   three 1-token requests under the arm (a reset first; the calibration nonce "-1": no snapshot P1-P4 can use, P1 resets again)
#   MEASURE the head: HEADTOK = P(header + head + task) - P(header + task), D = P(header + head + task) - P(task); the shared
#   length's window [D - 3, D + 19 + NSH] (template opening 0-16 tok, join +-3, NSH = the chars the tails also share, <= 1 tok
#   each); P2 / P3 expected by expect()'s rule on that window (B the deepest snapshot under it, C / T the grid point under it by
#   a cone replay unless a snapshot sits there), P4 = P3's branch snapshot inside P3's window (never below P2's resume). A window
#   holding a grid point or snapshot, or a failed calibration: information lines, not VIRHE. Identity stays strict. Nothing else
#   changed except qrun's --ntok (QNTOK, default HNTOK) for the 1-token requests.
# dsm-pskcr2.sh (12.11.2026, ds-main-port) = dsm-pskcr.sh 588994f8 + ONE change: the counter guard's baseline is each arm's first
#   request's pre-snapshot (taken after the arm switch), not the snapshot before the switch -- r00zzz-cz-scr's B VIRHE
#   'tail_only_latched +1' was one admission latched under the smoke's arm T between the old baseline and the switch
#   (st-guard-B-0 mode 4 -> st-guard-B-c-0 mode 2, tail_only 29 -> 30; B's own cold / hit: tail_only 30 -> 30).
# dsm-pskcr.sh (12.11.2026, queue-ar/dsmain/pskcr, ds-main-port PSK-CR builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-scr.sh. Log tag "RESULT xcr ", out runs/xdsm/pskcr (XCR_OUT). Copied from dsm-pskl20.sh (a7bacfdc: the
# start, LEVENV, the fixed pin, modes / swnote / smoke / qrun / msnap / mdelta / ssnap / cmpfile / idrec, the image check) and
# dsm-psk2b.sh (the hit design: each arm on its OWN text, cachedof); neither file is edited.
# WHY (ARCH-AUDIT-20261112 lever 3, the coordinator 12.11.): lever PSK-CR, the EXACT form of the official bounded replay (tech report
#   3.2.2). Under design 2 a prefix hit with no snapshot at the grid point under it falls back to the anchor below and recomputes
#   up to K chunks (measured: K4 worst +3.04 s, K8 mean +4.11 s). With ds41_psk_cr 1 the hit resumes AT the grid point `at` by a
#   cone replay: the cached global KV (latent + index keys, every row's) adopted up to `at`, the sequence started at the grid point
#   under at - 5152 with no state, its grid chunks up to `at` run every layer only over at's cone (layer l: the last 160 + (39 - l)
#   x 128 rows) with no global-KV store; the window state at `at` is then a cold prefill's. Estimate ~0.7-1.0 s a hit. It also
#   allows TAIL-ONLY anchors (ds41_psk 4: the prompt's end, its tail snapshot, checkpoint_at only; cold = K8's 31.4 s or better)
#   and serves the shared head's 2nd / 3rd prompt (branch_at) without the anchor fallback.
# IMAGE ds41-pskcr (= ds41-pskl20 25aab56 + e9e529d: fn4 ds41_psk_cr appended (45 switches), ds41_psk accepts 4; /stats psk cr_*
#   and tail_only_*; default 0 = the pskl20 image's behaviour).
# ONE NST 5 START = stack4's start exactly (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start": stack3 S env +
#   z0occ2's Z0 words + RADIANCE_DS41_PSK=0, MNBT 4160, --live --draft-window adaptive --draft-coupling on) + RADIANCE_LOG_STEPS=1;
#   the engine log captured by docker logs -t (gzipped). Arms in process (POST /admin/fn4: stack4's fixed pin + S's pins
#   {ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 2, ds41_psk_k 4} + {ds41_psk_l20 XCR_L20, dspark_pfctx XCR_PF} +
#   the arm + epoch_bump, a nudge, the applied switches READ BACK every time):
#     B = ds41_psk 2 k 4, ds41_psk_cr 0 (today: the anchor fallback);  C = ds41_psk 2 k 4 + ds41_psk_cr 1;  T = ds41_psk 4 (tail-only)
#     + ds41_psk_cr 1.
#   XCR_L20 / XCR_PF (default 1 = PSK-L20 / PSK-DS on in EVERY arm): THE COORDINATOR SETS THEM FROM r00zzz-cz-pskl20's RESULT BEFORE
#   THIS RUN (that item is queued ahead of this one): 0 where its lever did not pass.
# ORDER: gate1 -> modes (stack4's literals + the PSK-L20 and PSK-CR declare lines) -> VRAM -> switch note (start values ds41_psk_cr
#   0 / ds41_psk 0; ds41_psk_cr 2 and ds41_psk 5 refused; B / C / T applied and read back) -> warm-up answers -> smoke under T ->
#   COUNTER GUARD (each arm on its own text: reset, cold ~16k, a hit at ~12288 -- the grid point 10240 has no K4 anchor: C / T must
#   move cr_hits / cr_rows_replayed / cr_passes / cr_layer_rows_run / cr_layer_rows_skipped / cr_chains_skipped with no cr fallback
#   and no mixed replay pass, T also tail_only_latched; B must move no cr / tail-only counter and fall back (fallback_resumes);
#   design 2's latched / planned_steps grow and plan_mismatch does not in every arm; PSK-L20 / PSK-DS's counters grow where their
#   knob is 1. A failure = VIRHE and NO speed reading) -> HITS round 1 (B C T, each on its OWN text: reset, cold 64k, then with no
#   reset hits at 40960 / 20480 / 24576 / 4096 (mid-interval: grid points 38912 / 18432 / 22528 / 2048 +- 2048 for the text's
#   tokens a char, none a K4 anchor: mid-prompt hits with no anchor, 4096 the first-chunk replay; the expected resume of every
#   hit derived from its own prompt tokens and the snapshots its text holds, pcr.py expect), the 64k prompt again (its tail
#   snapshot), and a FOLLOW-UP TURN on the 40960 hit (its
#   answer + a new user message: a hit followed by more turns); cached asserted (C / T: the grid point; B: K4's anchor below; the
#   tail; the turn 38912 everywhere); then B's COLD references on that text, a reset each: IDENTITY hit vs cold of the same full
#   text) -> SHARED HEAD (each arm its own nonce'd head of 24576 planned tokens + four 3000-token tails; the head calibrated first
#   by three 1-token requests: P1 cold, P2 / P3 hits at the head (expected from the MEASURED shared length, 12.11. head fix
#   above), P3 takes the head's snapshot (branch_at), P4 resumes there; P2-P4 vs B cold) -> HITS UNDER LOAD
#   (c4 decode streams + one 45056 hit right after the arm's round-1 hits: grid point 43008, B the snapshot below; vs B's cold
#   45056 alone: strict
#   when IDENTICAL, else class (ii) information unless B's own loaded hit is IDENTICAL) -> HITS round 2 (T C B on new texts, the
#   same depths: a second TTFT sample, cached asserted) -> COLD TTFT ABBA T B B T at 136k and 64k (fresh distinct texts, a reset
#   each) -> SUMMARY (guard, counters, hit resumes, hit TTFT per depth C / T vs B, cold TTFT T vs B, identity).
# ~40 min (estimate): start + checks + smoke ~9, guard ~1.5, hits round 1 ~3.5 + refs ~4, shared head ~1.5 + refs ~1.5, loaded hits
#   ~2.5, hits round 2 ~3.5, cold TTFT 8 runs ~4.5, log + summary ~2, arm switches / resets ~5. Budget <= ~60 min.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact by design:
#   bytes measured and reported, a difference is reported, never hidden.
# Disk: >= XCR_DISKMIN_MB (11000) free on / before the start (the 10 GB guard; docker's json log of a RADIANCE_LOG_STEPS start lives
#   until the container is removed); engine logs written gzipped only.
# Knobs: XCR_IMAGE, XCR_ITEM, XCR_OUT, XCR_L20 (1), XCR_PF (1), XCR_PSKK (4), XCR_OCC (6), XCR_OCCK (32), XCR_DSA (85),
#   XCR_HDEP (64000), XCR_HITS ("40960 20480 24576 4096"), XCR_LHIT (45056), XCR_HEAD (24576), XCR_HTAIL (3000),
#   XCR_DEPTHS ("135000 64000"), XCR_TORDER (TBBT), XCR_ROUND2 (1), XCR_DISKMIN_MB (11000), XCR_SIDECAR, XCR_SIDESHA.
# DRYRUN=1 bash dsm-pskcr.sh : prints every command; checks the image literals (the 45-switch registry, PSK-CR's literals, exports
#   and /stats keys), every helper (files, parse, selftests, the text offsets), the sidecars, the yaml; no server, no GPU, writes
#   nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XCR_OUT:-$R/xdsm/pskcr3}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS4=$R/xdsm/stack4
export IMAGE=${XCR_IMAGE:?set XCR_IMAGE}
ITEM=${XCR_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-v-scr3}
L20ON=${XCR_L20:-0}; PFON=${XCR_PF:-0}
PSKK=${XCR_PSKK:-4}; OCC=${XCR_OCC:-6}; OCCK=${XCR_OCCK:-32}; DSA=${XCR_DSA:-85}
HDEP=${XCR_HDEP:-64000}; HITS=${XCR_HITS:-40960 20480 24576 4096}; CRD1=${XCR_CRD1:-43500}; CAD=${XCR_CAD:-48000}; NTURN=${XCR_NTURN:-4}
DEPTHS=${XCR_DEPTHS:-135000}; TORDER=${XCR_TORDER:-TBBT}; DISKMIN=${XCR_DISKMIN_MB:-11000}
GDEP=16000; GHIT=12288; HNTOK=64; TNTOK=64
SIDE=${XCR_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XCR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
# the text offsets (ds41_tests.SRC file index, < its 624 files): guard 10.. (one an arm), hit cases 300.., chains chR 360.. and chA
# 370.. (one an arm each), TTFT 30.. (4 a depth); every section's prompts also open with the arm's nonce line
GOFF=10; HOFF=300; CROFF=360; CAOFF=370; TOFF=30
case $TORDER in *[!BCT]*|"") echo "XCR_TORDER takes B / C / T (got '$TORDER')"; exit 1 ;; esac
for v in $L20ON $PFON $PSKK $OCC $OCCK $DSA $HDEP $HITS $CRD1 $CAD $NTURN $DEPTHS $DISKMIN; do
  case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case "$L20ON$PFON" in 00|01|10|11) ;; *) echo "XCR_L20 / XCR_PF take 0 or 1"; exit 1 ;; esac
[ ${#TORDER} -le 4 ] || { echo "XCR_TORDER: at most 4 runs a depth (4 text offsets each)"; exit 1; }
for h in $HITS $CRD1; do [ $h -lt $HDEP ] || { echo "XCR_HITS / XCR_CRD1 must be below XCR_HDEP"; exit 1; }; done
[ $NTURN -ge 2 ] && [ $NTURN -le 5 ] || { echo "XCR_NTURN takes 2 .. 5 (pcr3.py has 4 follow-up messages)"; exit 1; }
[ ${#DEPTHS} -gt 0 ] || { echo "XCR_DEPTHS empty"; exit 1; }
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
PL=$HD/pskld.py; PX=$HD/pcr3.py   # pcr3.py: pcr2.py + the ds41-pskcr2 rules (cheaper path, the replay's end snapshot), chains, nonces
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xcr $*"; }
# ---------------------------------------------------------------- the start (= dsm-stack4.sh's + RADIANCE_LOG_STEPS=1)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
STARTENV="$LEVENV RADIANCE_LOG_STEPS=1"
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"   # the PSK-L20 declare line (lever PSK-L20 (fn4 ds41_psk_l20, ...)
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"   # the PSK-CR declare line (lever PSK-CR (fn4 ds41_psk_cr, ...)
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" "$OCCK") || { echo "pin JSON"; exit 1; }
SPIN="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"ds41_psk_k\": $PSKK, \"ds41_psk_l20\": $L20ON, \"dspark_pfctx\": $PFON"
armj() {   # armj B|C|T: S's pins (+ XCR_L20 / XCR_PF) + the arm's mode and replay switch (JSON)
  case $1 in
    B) echo "{$SPIN, \"ds41_psk\": 2, \"ds41_psk_cr\": 0}" ;;
    C) echo "{$SPIN, \"ds41_psk\": 2, \"ds41_psk_cr\": 1}" ;;
    T) echo "{$SPIN, \"ds41_psk\": 4, \"ds41_psk_cr\": 1}" ;;
  esac
}
armn() { echo $1; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 stack4 start (stack3 S: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, --draft-coupling on, DLW sidecar $SIDE; z0occ2's ring 129, Z0-W declared, rungs 8,48; PSK declared off) + RADIANCE_LOG_STEPS=1 | fixed pin: stack4's (stack3's + Z0 switches 0, draft_occ_k $OCCK) | S pins: ds41_bi 2, draft_sample $DSA, draft_occ $OCC, place_lf 0, ds41_psk_k $PSKK, ds41_psk_l20 $L20ON, dspark_pfctx $PFON | B: ds41_psk 2 cr 0, C: ds41_psk 2 cr 1, T: ds41_psk 4 (tail-only) cr 1"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM
  RADIANCE_DS41_IDX_BALANCE RADIANCE_DS41_QNFOLD RADIANCE_DS41_SWFOLD RADIANCE_DRAFT_CTXSKIP RADIANCE_DS41_EMB_RAM
  RADIANCE_DS41_DSPARK_CONF RADIANCE_DS41_DSPARK_ALT RADIANCE_DS41_KVWIN_FP8 RADIANCE_DS41_DRAFTER_MX RADIANCE_DS41_DRAFTER_BF
  RADIANCE_DS41_GG_PF RADIANCE_DS41_GG_PF_TABLE RADIANCE_DS41_F32_PF RADIANCE_DS41_NVGATHER RADIANCE_DS41_DSPARK_WALK
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_DRAFT_SAMPLE RADIANCE_WEIGHT_SIDECAR
  RADIANCE_DS41_BI RADIANCE_DS41_BITRACE RADIANCE_DS41_BITRACE_PMAX RADIANCE_DRAFT_CTXSKIP_SLOTS RADIANCE_DRAFT_LEND_WEIGHTS
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_PSK_L20 RADIANCE_DS41_DSPARK_PFCTX RADIANCE_DS41_PSK_CR"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset (stack4's)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $STARTENV; do export "$kv"; done
}
dfree() { df -Pm / | awk 'NR==2{print $4}'; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
nowt() { date +%s.%N; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  [ "$DRY" = 1 ] && { echo "+ identity $1${3:+ ($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference (stack4's)
  [ "$DRY" = 1 ] && { echo "+ RESULT xcr $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "info-warm $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS (stack4's)
  [ "$DRY" = 1 ] && { echo "+ modes $1: STARTENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $STARTENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|draft floor|draft refill" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-300 | sort -u | tr '\n' '|' | cut -c1-3200)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xcr $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS4/s4-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack4 s4-n5) | /stats psk + dspark_pfctx at start (cr 0): $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps(s.get("psk")), json.dumps(s.get("dspark_pfctx")))' $O/$1-stats0.json 2>&1 | cut -c1-900)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || xlog "$1 prefix cache reset: VIRHE HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
ARMBAD=0
arm() {   # arm B|L|D|X [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge to apply it at a step, the switches READ BACK
  local want g
  want=$(pinj $1 "${2:-}")
  fn4set "$(pinj $1 "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1] or "{}"); d["epoch_bump"]=1; print(json.dumps(d))' "${2:-}")")"; nudge
  [ "$DRY" = 1 ] && return 0
  g=$(fn4get)
  python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$want" 2>/dev/null && return 0
  ARMBAD=$((ARMBAD + 1))
  xlog "VIRHE arm $(armn $1)${2:+ + $2}: the applied switches are not the pin: $(echo "$g" | cut -c1-900)"
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window (stack4's)
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,re,sys
def rd(p):
    d = {}
    try:
        for l in open(p):
            m = re.match(r"^(radiance:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", l)
            if m: d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    except OSError: pass
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda k: b.get("radiance:" + k, 0.0) - a.get("radiance:" + k, 0.0)
nd = g("engine_steps_prefill_total") + g("engine_steps_mixed_total")
nds = g("engine_step_seconds_total") - g("engine_step_seconds_decode_total")
dt, da = g("draft_tokens_total"), g("draft_accepted_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da),
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
ssnap() { [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json"; return 0; }; curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; }
sdelta() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; python3 -I $PX sdelta $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'; }
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; echo "$(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-260) | $(docker logs $CN 2>&1 | grep 'lever PSK-DS:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200) | $(docker logs $CN 2>&1 | grep -c 'cone replay from') replay admissions logged | VIRHE lines $(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')"; }
jmerge() { python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]:
    try: d.update(json.loads(a or "{}"))
    except ValueError: pass
print(json.dumps(d))' "$@"; }
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm) (stack4's)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- one request, recorded
qrun() {   # qrun TAG ARM KIND NAME SPEC RESET [WANT [WANTREPLAY [LOAD]]]: arm, (reset), one request (pcr3.py run --spec SPEC:
  # deep:DEPTH:OFF[@NONCE] | chain:DEPTH:OFF:ANS1,..[@NONCE]), /metrics + /stats deltas -> deep.jsonl + a RESULT line; WANT
  # the cached count it must report ("" none), WANTREPLAY 1 when a C / T arm must have served it by a cone replay; LOAD "load":
  # c4 decode streams beside it (pskld.py --load)
  local tag=$1 a=$2 kind=$3 nm=$4 spec=$5 rs=$6 w=${7:-} wr=${8:-0} ld=${9:-} t0=$(date +%s) j md sd lab
  arm $a
  [ "$rs" = reset ] && pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 2400 python3 -I $PX run --pskld $PL --url $URL --tests $TESTS --spec $spec --ntok ${QNTOK:-$HNTOK} --out $O/$tag-$nm.json --text $O/$tag-$nm.txt$([ "$ld" = load ] && echo " --load --mb $MBP") > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xcr $tag $nm: arm $a $kind $spec$([ "$rs" = reset ] || echo ", NO reset")${ld:+, under c4 load}${w:+, cached must be $w}$([ "$wr" = 1 ] && echo ", a cone replay"): TTFT, cached, /stats deltas (psk cr_* incl. cr_cheaper / cr_end_snapshots + design 2 + L20 + pfctx, staged GB, snapshots) -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.out | cut -c8-)
  md=$(mdelta $tag-$nm-0 $tag-$nm-1); sd=$(sdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  lab=$(echo ${spec%@*} | cut -d: -f1-2)
  python3 -c 'import json,sys
d = {}
for x in sys.argv[1:4]: d.update(json.loads(x or "{}"))
w = sys.argv[9]
d.update(name=sys.argv[4], arm=sys.argv[5], kind=sys.argv[6], spec=sys.argv[7].split("@")[0], label=sys.argv[8], want=int(w) if w else None,
         want_replay=sys.argv[10] == "1", reset=sys.argv[11] == "reset", load=sys.argv[12] == "load", text=sys.argv[13])
f = sys.argv[7].split(":")
d["depth"] = int(f[1]) + (int(f[3]) if f[0] == "head" else 0)
print(json.dumps(d))' "$j" "$md" "$sd" "$nm" "$a" "$kind" "$spec" "$lab" "$w" "$wr" "$rs" "${ld:-alone}" "$O/$tag-$nm.txt" >> $O/deep.jsonl
  xlog "$tag $nm (arm $a, $kind, $spec, ${QNTOK:-$HNTOK} greedy tokens$([ "$rs" = reset ] || echo ", NO reset")${ld:+, under c4 load}): $(python3 -c 'import json,sys
d, m, s = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}"), json.loads(sys.argv[3] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok, %s chars | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB | psk planned %s cut %s fallback %s mismatch %s | CR hits %s rows %s passes %s layer-rows run %s / skipped %s chains %s mixed %s fallbacks %s not-needed %s cheaper %s end-snapshots %s | tail-only %s refused %s | L20 cut %s | pfctx passes %s%s" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("chars"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), s.get("ckpts"), s.get("staged_gb"), s.get("psk_planned_steps"), s.get("psk_cut_passes"), s.get("psk_fallback_resumes"),
    s.get("psk_plan_mismatch"), s.get("psk_cr_hits"), s.get("psk_cr_rows_replayed"), s.get("psk_cr_passes"), s.get("psk_cr_layer_rows_run"),
    s.get("psk_cr_layer_rows_skipped"), s.get("psk_cr_chains_skipped"), s.get("psk_cr_mixed_passes"), s.get("psk_cr_fallbacks"),
    s.get("psk_cr_not_needed"), s.get("psk_cr_cheaper"), s.get("psk_cr_end_snapshots"), s.get("psk_tail_only_latched"), s.get("psk_tail_only_refused"), s.get("psk_l20_cut_passes"), s.get("pfctx_passes"),
    (" | c4 under the probe: agg %s tok/s, TPOT p90 %s ms, %s errors" % (d.get("c4_pf_agg_tps"), d.get("tpot_pf_p90"), d.get("c4_errors"))) if d.get("load") else ""))' "$j" "$md" "$sd" 2>&1 | cut -c1-1100) | $(thermline $t0 $(date +%s))"
  if [ -n "$w" ]; then
    local c; c=$(cachedof $nm)
    if [ "$c" = "$w" ]; then xlog "$tag $nm: cached $c = expected ($([ "$wr" = 1 ] && echo "the grid point under the match, by a cone replay" || echo "arm $a's resume"))"
    else xlog "$tag $nm: VIRHE cached $c, expected $w"; echo "$nm VIRHE cached $c want $w" >> $O/guard.txt; fi
  fi
}
cachedof() { [ "$DRY" = 1 ] && { echo 0; return 0; }; python3 -c 'import json,sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
x = [d for d in r if d.get("name") == sys.argv[2]]
print(x[-1].get(sys.argv[3]) if x else -1)' $O/deep.jsonl "$1" "${2:-cached}" 2>/dev/null || echo -1; }
want() { python3 -I $PX want $1 $2 $PSKK 2>/dev/null; }   # want ARM DEPTH: B K's anchor below; C / T the grid point under it by a replay where that lies > a grid step over the snapshot below, else that snapshot
# ---------------------------------------------------------------- the start's checks
swnote() {   # swnote TAG: start values; refusals; every arm applied at a step and read back, then B
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (ds41_psk_cr 0, ds41_psk 0, ds41_psk_k 8, ds41_psk_l20 0, dspark_pfctx 0, ds41_bi 0, draft_occ_k 32, epoch 0); refusals ds41_psk_cr 2 / ds41_psk 5 -> 400; arms B $(armj B) | C $(armj C) | T $(armj T) applied + read back, then B"; return 0; }
  local s0 c1 c2 a b0=$ARMBAD
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_psk_cr": 2}'); c2=$(fn4code '{"ds41_psk": 5}')
  for a in B C T B; do arm $a; done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("ds41_psk_cr %s ds41_psk %s/%s ds41_psk_l20 %s dspark_pfctx %s ds41_bi %s draft_occ_k %s epoch %s" % (w.get("ds41_psk_cr"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_l20"), w.get("dspark_pfctx"), w.get("ds41_bi"), w.get("draft_occ_k"), d.get("epoch")))' 2>&1 | cut -c1-300) | ds41_psk_cr 2 -> HTTP $c1, ds41_psk 5 -> HTTP $c2 | arms B C T B applied, $((ARMBAD - b0)) not as pinned"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("ds41_psk_cr") == 0 and w.get("ds41_psk_l20") == 0 and w.get("dspark_pfctx") == 0 and w.get("ds41_psk") == 0 and w.get("ds41_psk_k") == 8 and w.get("ds41_bi") == 0
ok = ok and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and s0["epoch"] == 0 and w.get("draft_occ_k") == 32 and s0["draft_capable"] == 1
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ $ARMBAD = $b0 ]
}
smoke() {   # smoke TAG: under T (tail-only + the replay switch): the greedy set at conc 4 (mixed steps) + mb c4 20 s at T 1 (stack4's)
  local tag=$1 t0=$(date +%s) n e
  arm T
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  run "(cd $D && timeout 600 python3 mb.py --tag xcr-smoke --phases c4 --dur 20 --warm 5 > $O/$tag-smoke-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm T $(armj T): greedy set conc 4 (32 tok) 20 of 20 + mb c4 20 s at T 1 with 0 errors, /health 200, no design-2 plan-mismatch line"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  e=$(grep -m1 '^MB ' $O/$tag-smoke-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && case "$e" in *" 0 errors"*) true ;; *) false ;; esac && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200 && \
     [ "$(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')" = 0 ]; then
    xlog "$tag smoke (arm T): greedy set conc 4 20 of 20, mb c4 T 1: $e, healthy, $(( $(date +%s) - t0 )) s | psk: $(pskstat)"; return 0; fi
  xlog "$tag smoke (arm T): VIRHE $n of 20 answered, mb c4 T 1: $e, unhealthy or a design-2 VIRHE line | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus|VIRHE' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
SPEEDOK=1
guard() {   # guard TAG: the counter guard -- each arm on its own text: reset, cold GDEP, a hit at GHIT (no reset); the pair's /stats
  # delta checked (pcr.py guard): C / T replay, B falls back
  local tag=$1 a i=0 r nn
  for a in B C T; do
    stopfile && return 0
    nn=$a-$(date +%s)
    ssnap $tag-guard-$a-0
    qrun $tag $a guard guard-$a-c deep:$GDEP:$((GOFF + i))@$nn reset
    qrun $tag $a guardhit guard-$a-h deep:$GHIT:$((GOFF + i))@$nn noreset "$(want $a $GHIT)" $([ $a = B ] && echo 0 || echo 1)
    ssnap $tag-guard-$a-1
    i=$((i + 1))
    if [ "$DRY" = 1 ]; then echo "+ guard arm $a: pcr3.py guard $a <the cold + hit pair's /stats deltas> $L20ON $PFON (C / T: the cr counters + cr_end_snapshots grow, no cr fallback; B: none)"; continue; fi
    r=$(python3 -I $PX guard $a "$(sdelta $tag-guard-$a-c-0 $tag-guard-$a-1)" $L20ON $PFON 2>&1 | cut -c1-900)   # baseline AFTER the arm switch (12.11.: the old baseline caught a request latched under the previous arm)
    echo "$a $r" >> $O/guard.txt
    case "$r" in OK*) xlog "$tag counter guard arm $a: $r" ;;
      *) xlog "$tag counter guard arm $a: VIRHE $r -- a lever that silently does nothing (or one that moved where it must not): no speed reading"; SPEEDOK=0 ;; esac
  done
  grep -q VIRHE $O/guard.txt 2>/dev/null && { xlog "$tag: VIRHE a guard hit resumed where it must not -- no speed reading"; SPEEDOK=0; }
  [ $ARMBAD = 0 ] || { xlog "$tag: VIRHE $ARMBAD arm switch(es) not applied as pinned -- no speed reading"; SPEEDOK=0; }
}
# ---------------------------------------------------------------- hits
# SNAPS: the snapshots the current text holds (from every run on it: pcr3.py snaps -- K's anchors, the tail, a replay's end); a
# hit's expected resume from its own prompt tokens (pcr3.py expect / expect_turn). DRYRUN: placeholders.
SNAPS=""
declare -A SNT   # SNAPS by text offset (the cold runs, hits and B's references on it: reset_prefix_cache keeps the records)
declare -A NON   # the nonce of each text offset's section (its refs repeat the same full text)
ptof() { [ "$DRY" = 1 ] && { echo $2; return 0; }; cachedof $1 prompt_tokens; }
addsnaps() {   # addsnaps NAME ARM [CKPT_AT]: the snapshots run NAME wrote join SNAPS (a replayed hit's end among them)
  [ "$DRY" = 1 ] && return 0
  local s; s=$(python3 -I $PX snaps $2 "$(cachedof $1)" "$(cachedof $1 prompt_tokens)" $PSKK ${3:-0} "$(cachedof $1 psk_cr_hits)" 2>/dev/null)
  SNAPS=$(echo "$SNAPS,$s" | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n -u | paste -sd,)
}
wantrec() {   # wantrec NAME WANT REPLAY: the expected resume into NAME's deep.jsonl row
  python3 -c 'import json,sys
p, nm, w, wr = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
r = [json.loads(l) for l in open(p) if l.strip()]
for d in r:
    if d.get("name") == nm: d["want"], d["want_replay"] = w, wr
open(p, "w").write("".join(json.dumps(d) + "\n" for d in r))' $O/deep.jsonl $1 $2 $3
}
rescheck() {   # rescheck TAG NAME ARM E: E = "WANT REPLAY" (pcr3.py expect / expect_turn) or "" (not asserted) -- cached, the replay
  # (C / T: counted where expected, none where the snapshot or a cold prefill was the cheaper path), the replay's end snapshot
  local tag=$1 nm=$2 a=$3 e=$4 w wr c h
  c=$(cachedof $nm); h=$(cachedof $nm psk_cr_hits)
  if [ -z "$e" ]; then xlog "$tag $nm: expected resume not asserted (a grid point or snapshot inside the match's range; cached $c, prompt $(cachedof $nm prompt_tokens), replays $h)"; return 0; fi
  w=${e% *}; wr=${e#* }
  wantrec $nm $w $wr
  if [ "$c" = "$w" ]; then xlog "$tag $nm: cached $w = expected ($([ "$wr" = 1 ] && echo "the grid point under the match, by a cone replay" || echo "the snapshot below / at it, or cold (0)"))"
  else xlog "$tag $nm: VIRHE cached $c, expected $w (snapshots $SNAPS)"; echo "$nm VIRHE cached $c want $w" >> $O/guard.txt; fi
  if [ $a != B ] && [ "$wr" = 1 ] && [ "$h" != 1 ]; then
    xlog "$tag $nm: VIRHE no cone replay counted (cr_hits $h)"; echo "$nm VIRHE cr_hits $h" >> $O/guard.txt; fi
  if [ $a != B ] && [ "$wr" = 0 ] && [ "$h" != 0 ]; then
    xlog "$tag $nm: VIRHE a cone replay (cr_hits $h) where the snapshot / a cold prefill is the cheaper exact path"; echo "$nm VIRHE cr_hits $h (cheaper path)" >> $O/guard.txt; fi
  if [ $a != B ] && [ "$h" = 1 ] && [ "$(cachedof $nm psk_cr_end_snapshots)" -lt 1 ] 2>/dev/null; then
    xlog "$tag $nm: VIRHE the replay took no snapshot at its end (cr_end_snapshots $(cachedof $nm psk_cr_end_snapshots))"; echo "$nm VIRHE no end snapshot" >> $O/guard.txt; fi
}
hitq() {   # hitq TAG ARM KIND NAME SPEC: one hit (no reset), its expected resume checked after it (pcr3.py expect on its own prompt
  # tokens and the text's snapshots), then its snapshots recorded
  local tag=$1 a=$2 kind=$3 nm=$4 spec=$5
  qrun $tag $a $kind $nm $spec noreset "" 0
  [ "$DRY" = 1 ] && { echo "+ $nm: expected resume = pcr3.py expect $a <its prompt tokens> <the text's snapshots> (B: the snapshot below; C / T: the grid point under the match by a cone replay where it lies more than one grid step over the snapshot below, else that snapshot / cold); a replay must have taken its end snapshot"; return 0; }
  rescheck $tag $nm $a "$(python3 -I $PX expect $a "$(cachedof $nm prompt_tokens)" "$SNAPS" $PSKK 2>/dev/null)"
  addsnaps $nm $a
}
hitblock() {   # hitblock TAG ARM DOFF: the arm on its own nonce'd text -- reset, cold HDEP, then hits at HITS (no reset; scr2's round-1
  # order: 24576 right after 20480, 4096 the first-chunk hit)
  local tag=$1 a=$2 off=$3 h p=h$2 nn=$2-$(date +%s)
  stopfile && return 0
  NON[$off]=$nn
  SNAPS=""
  qrun $tag $a hitcold $p-c$HDEP deep:$HDEP:$off@$nn reset 0
  addsnaps $p-c$HDEP $a
  for h in $HITS; do stopfile && break; hitq $tag $a hit $p-$h deep:$h:$off@$nn; done
  SNT[$off]=$SNAPS
}
refblock() {   # refblock TAG ARM DOFF: B's COLD references on arm ARM's text (the same nonce'd full texts, a reset each), then
  # IDENTITY hit vs cold
  local tag=$1 a=$2 off=$3 h p=h$2 nn=${NON[$3]:-$2-0}
  for h in $HDEP $HITS; do stopfile && return 0; qrun $tag B ref $p-ref$h deep:$h:$off@$nn reset 0; done
  [ "$DRY" = 1 ] && { echo "+ identity: $p-c$HDEP vs $p-ref$HDEP (cold vs B cold), $p-H vs $p-refH for H in $HITS (hit vs B cold)"; return 0; }
  idrec "hit-$a-cold$HDEP-vs-Bcold" "$(cmpfile $O/$tag-$p-ref$HDEP.txt $O/$tag-$p-c$HDEP.txt)" "$a cold $HDEP vs B cold, text $off"
  for h in $HITS; do
    idrec "hit-$a-$h-vs-Bcold" "$(cmpfile $O/$tag-$p-ref$h.txt $O/$tag-$p-$h.txt)" "$a: depth $h resumed at $(cachedof $p-$h) (replays $(cachedof $p-$h psk_cr_hits)) vs B's cold answer (text $off)"
  done
}
# ---------------------------------------------------------------- turn chains
declare -A CHS   # the specs of each chain's turns (CHS[<chain><arm><k>]) for the references
chainblock() {   # chainblock TAG ARM R|A DOFF: a conversation of NTURN turns on the arm's own nonce'd text. R: reset, cold HDEP, turn 1 =
  # a hit at CRD1 (C / T: a replay -- two or more grid steps over K4's anchor or no snapshot at all; B: the anchor below); A: reset,
  # turn 1 COLD at CAD. Turns 2..NTURN = the previous prompt + its answer + a new user message (pcr3.py chain:..). Per turn: the
  # expected resume (turn 1: pcr3.py expect, turns >= 2: expect_turn on [previous prompt - 16, prompt - 8]), cached, replays
  # (C / T: none after turn 1 -- a replay there is the defect: VIRHE), snapshots, TTFT
  local tag=$1 a=$2 kd=$3 off=$4 p=ch$3$2 nn=$2-$(date +%s) k ans="" spec d1 pp
  stopfile && return 0
  SNAPS=""
  if [ $kd = R ]; then
    d1=$CRD1
    qrun $tag $a chcold $p-c deep:$HDEP:$off@$nn reset 0
    addsnaps $p-c $a
    stopfile && return 0
    spec=deep:$d1:$off@$nn
    qrun $tag $a chain $p-t1 $spec noreset "" 0
    [ "$DRY" = 1 ] && echo "+ $p-t1: expected resume = pcr3.py expect $a <its prompt tokens> <the cold text's snapshots> (C / T: a cone replay to the grid point 38912 / 43008; B: K4's anchor 30720)"
    [ "$DRY" = 1 ] || rescheck $tag $p-t1 $a "$(python3 -I $PX expect $a "$(cachedof $p-t1 prompt_tokens)" "$SNAPS" $PSKK 2>/dev/null)"
  else
    d1=$CAD
    spec=deep:$d1:$off@$nn
    qrun $tag $a chain $p-t1 $spec reset 0
    [ "$DRY" = 1 ] || rescheck $tag $p-t1 $a "0 0"
  fi
  CHS[${kd}${a}1]=$spec
  addsnaps $p-t1 $a
  for k in $(seq 2 $NTURN); do
    stopfile && return 0
    ans="$ans${ans:+,}$O/$tag-$p-t$((k - 1)).txt"
    spec=chain:$d1:$off:$ans@$nn
    CHS[$kd$a$k]=$spec
    qrun $tag $a chain $p-t$k $spec noreset "" 0
    if [ "$DRY" = 1 ]; then echo "+ $p-t$k: turn $k = turn $((k - 1))'s prompt + its answer + follow-up $((k - 1)); expected resume = pcr3.py expect_turn $a <turn $((k - 1))'s prompt tokens> <its prompt tokens> <snapshots> (every arm: turn $((k - 1))'s snapshot -- the replay's end / its tail -- or one grid step below; C / T: NO replay)"; continue; fi
    pp=$(cachedof $p-t$((k - 1)) prompt_tokens)
    rescheck $tag $p-t$k $a "$(python3 -I $PX expect_turn $a "$pp" "$(cachedof $p-t$k prompt_tokens)" "$SNAPS" $PSKK 2>/dev/null)"
    if [ "$(cachedof $p-t$k psk_cr_hits)" != 0 ]; then
      xlog "$tag $p-t$k: VIRHE turn $k replayed (cr_hits $(cachedof $p-t$k psk_cr_hits), rows $(cachedof $p-t$k psk_cr_rows_replayed)) -- a later turn must resume from a snapshot"
      echo "$p-t$k VIRHE replay at turn $k" >> $O/guard.txt; fi
    addsnaps $p-t$k $a
  done
  [ "$DRY" = 1 ] || xlog "$tag $p: chain done -- cached $(for k in $(seq 1 $NTURN); do printf '%s/' "$(cachedof $p-t$k)"; done | sed 's#/$##'), replays $(for k in $(seq 1 $NTURN); do printf '%s/' "$(cachedof $p-t$k psk_cr_hits)"; done | sed 's#/$##'), end snapshots $(for k in $(seq 1 $NTURN); do printf '%s/' "$(cachedof $p-t$k psk_cr_end_snapshots)"; done | sed 's#/$##'), TTFT $(for k in $(seq 1 $NTURN); do printf '%s/' "$(cachedof $p-t$k ttft)"; done | sed 's#/$##') s | snapshots on the path $SNAPS"
}
chainref() {   # chainref TAG ARM R|A: B's COLD references of every turn's full text (a reset each; chA's cold turn 1 only for C / T),
  # then IDENTITY turn vs cold
  local tag=$1 a=$2 kd=$3 p=ch$3$2 k
  for k in $(seq 1 $NTURN); do
    stopfile && return 0
    [ $kd = A ] && [ $k = 1 ] && [ $a = B ] && continue
    qrun $tag B ref $p-ref$k "${CHS[$kd$a$k]:-deep:0:0}" reset 0
  done
  [ "$DRY" = 1 ] && { echo "+ identity: $p-t1..t$NTURN vs B cold $p-ref1..ref$NTURN (chA: B's turn 1 is itself cold)"; return 0; }
  for k in $(seq 1 $NTURN); do
    [ $kd = A ] && [ $k = 1 ] && [ $a = B ] && continue
    idrec "chain-$kd-$a-t$k-vs-Bcold" "$(cmpfile $O/$tag-$p-ref$k.txt $O/$tag-$p-t$k.txt)" "$a: chain $kd turn $k (resumed at $(cachedof $p-t$k), replays $(cachedof $p-t$k psk_cr_hits)) vs B's cold answer of the same full text"
  done
}
ttftabba() {   # ttftabba TAG: per depth, TORDER (T vs B), each run a fresh distinct text (TOFF + 4 x depth index + run), a reset each
  local tag=$1 dep di=0 a k
  for dep in $DEPTHS; do
    k=0
    for a in $(echo $TORDER | grep -o .); do
      stopfile && return 0
      qrun $tag $a ttft t$dep-$((k + 1))$a deep:$dep:$((TOFF + 4 * di + k)) reset 0
      k=$((k + 1))
    done
    di=$((di + 1))
  done
}
ident() {   # ident TAG: per arm its hit cases, B's cold references + identity; then per arm its two turn chains (chR, chA), B's cold
  # references of every turn + identity
  local tag=$1 a i=0
  for a in B C T; do hitblock $tag $a $((HOFF + i)); i=$((i + 1)); done
  i=0; for a in B C T; do refblock $tag $a $((HOFF + i)); i=$((i + 1)); done
  i=0; for a in B C T; do chainblock $tag $a R $((CROFF + i)); chainblock $tag $a A $((CAOFF + i)); i=$((i + 1)); done
  for a in B C T; do chainref $tag $a R; chainref $tag $a A; done
}
steplog() { capture $O/$1-engine-full.log.gz; }   # the engine log once (docker logs -t, gzipped): the replay steps' walls for later
summary() {
  [ "$DRY" = 1 ] && { echo "+ python3 -I $PX summary --out $O --k $PSKK: GUARD, COUNTERS (incl. no replay where the cheaper path was expected, every replay's end snapshot), HIT RESUMES, CHAIN per kind (per turn cached / replayed / snapshots / TTFT, C / T vs B) + CHAIN REPLAYS (at most one a chain, none after turn 1), HIT TTFT per depth (C / T vs B), TTFT cold (T vs B), IDENTITY"; return 0; }
  python3 -I $PX summary --out $O --k $PSKK > $O/summary.txt 2>&1
  while IFS= read -r l; do xlog "summary: ${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=cr-n5 ok i a
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $STARTENV > $XS2_ENVFILE"; else printf '%s\n' $STARTENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms B $(armj B) C $(armj C) T $(armj T) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2300)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag || { xlog "$tag: VIRHE the smoke failed -> stopped"; down; return 1; }
  stopfile || guard $tag
  stopfile || ident $tag
  if [ $SPEEDOK = 1 ]; then
    stopfile || ttftabba $tag
  else
    xlog "$tag: VIRHE a guard failed (above): no cold TTFT reading (the hit / chain TTFTs are information only) -- identity stands"
  fi
  steplog $tag
  if [ "$DRY" = 1 ]; then echo "+ RESULT xcr $tag: PSK stats at the end (the log's last PSK lines, replay admissions logged, /stats psk, E lines, arms not as pinned)"
  else xlog "$tag: PSK stats at the end: $(pskstat) | /stats psk $(curl -s -m 30 $URL/stats 2>/dev/null | python3 -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps(s.get("psk")), json.dumps(s.get("dspark_pfctx")))' 2>&1 | cut -c1-1400) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | arms not as pinned: $ARMBAD"; fi
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | guard B C T (cold $GDEP + a $GHIT hit; the baseline after each arm switch) | hit cases B C T at $HDEP -> [$HITS] vs B's cold refs | turn chains B C T: chR (cold $HDEP, turn 1 a $CRD1 hit) + chA (turn 1 cold $CAD), $NTURN turns each, vs B's cold refs | cold TTFT $TORDER at [$DEPTHS] fresh texts | every arm's texts nonce'd | XCR_L20 $L20ON XCR_PF $PFON | no evict | ~27 min (estimate, budget <= ~35)"
IMGLITS="B|sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx, ds41_psk_cr, epoch_bump
B|ds41_psk_cr
B|cr_hits
B|cr_rows_replayed
B|cr_fallback_geometry
B|cr_layer_rows_skipped
B|cr_chains_skipped
B|cr_mixed_passes
B|tail_only_latched
B|tail_only_refused
B|cr_cheaper
B|cr_end_snapshots
B|rad_psk_plan2
B|rad_psk_cr_reach
B|by a cone replay from %lld (lever PSK-CR)
B|ds41_psk_l20
B|dspark_pfctx
B|l20_rows_skipped
B|ds41_psk_k takes 1 .. 4096
B|RADIANCE_LOG_STEPS
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
A|rad_psk_plan2
A|rad_psk_cr_reach
A|rad_psk_plan
A|rad_psk_stats
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|RADIANCE_DS41_PSK_CR
A|ds41_psk_cr
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-DS: %lld context passes cut
A|lever PSK design 2: VIRHE
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK: %lld cut steps
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|draft_occ takes 0 (off) .. 64
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/pskcr2-PROVENANCE.txt || echo "MISS pskcr2-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-pskcr2 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-pskcr2 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 45-switch registry ending dspark_pfctx, ds41_psk_cr; PSK-CR's declare line, env name, exports rad_psk_plan2 / rad_psk_cr_reach, the replay admission line, /stats cr_* / tail_only_* + ds41-pskcr2's cr_cheaper / cr_end_snapshots; PSK-L20 / PSK-DS and stack4's int7 levers) + bin/rad-sidecar + pskcr2-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $PL $PX $DQ $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -B -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $PL $PX $DQ $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
PST=$(timeout 60 python3 -I -B $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$PST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${PST:0:300})" ;; esac
PXT=$(cd $D && timeout 120 python3 -I -B $PX selftest --pskld $PL --tests $TESTS 2>&1 | tail -1); case "$PXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PX(${PXT:0:300})" ;; esac
OFFS=$(for i in 0 1 2; do printf "%s," $((GOFF + i)) $((HOFF + i)) $((CROFF + i)) $((CAOFF + i)); done; for i in $(seq 0 7); do printf "%s," $((TOFF + i)); done; echo 1200)
TXT=$(cd $D && timeout 120 python3 -I -B $PX texts --pskld $PL --tests $TESTS --doffs $OFFS 2>&1 | tail -1); case "$TXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS texts:(${TXT:0:300})" ;; esac
WA=$(want C 40960):$(want B 40960):$(want T 20480):$(want C 20480):$(want B 20480):$(want C $GHIT):$(want T $GHIT):$(want B $GHIT):$(want C $CRD1):$(want C 4096)
[ "$WA" = "38912:30720:18432:14336:14336:10240:10240:0:43008:0" ] || HMISS="$HMISS want(C/B 40960, T/C/B 20480, C/T/B $GHIT, C $CRD1, C 4096 at K$PSKK = $WA, K4 + ds41-pskcr2 want 38912:30720:18432:14336:14336:10240:10240:0:43008:0)"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj armn pinj stackenv dfree keeplog capture thermline nowt cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap sdelta pskstat jmerge gate1 qrun cachedof want swnote smoke guard ptof addsnaps wantrec rescheck hitq hitblock refblock chainblock chainref ttftabba ident steplog summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du paste; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B C T; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, pskld.py $PST, pcr3.py $PXT, text offsets $TXT, want() at K$PSKK $WA, all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(dfree)" -ge "$DISKMIN" ] || { xlog "VIRHE $(dfree) MB free on / < $DISKMIN MB: not starting (the 10 GB guard + docker's step log)"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RS4/s4-n5-stats0.json; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: pcr3.py $(md5sum < $PX | cut -c1-8), pskld.py $(md5sum < $PL | cut -c1-8) (pskl20's copy), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8) | $(dfree) MB free on /"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RS4/s4-n5-stats0.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in B C T; do echo "+ arm $a: $(pinj $a)"; done
  echo "+ disk: $(dfree) MB free on / (the real run refuses to start below $DISKMIN)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; }
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
