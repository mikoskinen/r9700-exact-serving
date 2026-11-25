#!/bin/bash
# dsm-d0ab.sh (13.11.2026, queue-ar/dsmain/d0ab, ds-main-port base builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-ce-d0ab.sh. A COPY of base/dsm-baseline.sh (6be30f5e; unchanged), itself a copy of
# stack5/dsm-stack5.sh (63649414). Log tag "RESULT xd0 ", out XS5_OUT (the item: runs/xdsm/d0ab). Beside it: d0blocks.py = a COPY of
# stack5/s5blocks.py (ab789985) + --settle S (an idle sleep after each arm's read-back, before the counter baseline and the phases);
# s5ctr.py = an unchanged copy of stack5/s5ctr.py (256dfd9a; d0blocks imports it from beside itself); c4split.py = the coordinator's
# /tmp/c4split.py (a4568c4b, unchanged). stack5's other helpers read in place, read only.
# WHY: the post-riser baseline (base1113) c4 fell -7.7 % vs stack5 S5T: the drafter's depth-0 controller (draft_depth0 1 + draft_lend 1
# in stack5's fixed pin) held c4 at depth 0 for 97 % / 77 % of steps (pre-move 0 %); plain 4-row steps 29.3 -> 23.4 ms p50, 16-row
# drafting steps stayed 61-64 ms; realized depth-0 yield 0.166 tok/ms vs drafting 0.19-0.20.
# ONE START = the baseline's (int9 image, stack5's NST 5 start, KVR / SP off, card map). ARMS at runtime (fn4 POST + READ BACK), each =
# the fixed pin + stack5's S5T switches + the depth-0 controller pair (overrides the fixed pin's draft_depth0 1 / draft_lend 1):
#   A = draft_depth0 1, draft_lend 1 (the baseline)   B = draft_depth0 0 (drafting always on)   C = draft_depth0 1, draft_lend 0
# PHASES: start + modes + VRAM + card map + switch note (A / B / C applied and read back) + warm-up vs stock + smoke under A -> cold
# ~16k guard under A / B / C (S5T's lever counters) -> c4 BLOCKS XS5_ORDER0 (A B C C B A): mb c4 only at T 0 (stack4/mbt.py, the
# baseline's c4 phase), dur XS5_DUR 30 warm XS5_WARM 10, SETTLE XD0_SETTLE 5 s idle after each read-back (+ warm 10 excluded);
# per block agg tok/s, ms and tokens a decode step, acceptance, decode share, draftph c4 (zero_cycles / zero_moves / probe_cycles),
# statsph c4 (d0_share, lent_cycles); block guard: SA / HB must not move (c4-only blocks need no lever move) -> the engine log
# captured right after the c4 blocks + c4split.py per arm (plain vs drafting 4-row decode steps: share, p50 wall, ms/token) -> ONE B
# EXTRA block: mb c1 + mb c8 (c8 through nc/ncmb.py, old loop; the baseline's harness and args) + the c8 drafting check (occ rule:
# occ_cycles > 0, acceptance 0, ~8 tok/step; drafting ON is a NOTE, not an error) -> IDENTITY: the greedy set at conc 4 (T 0) under A,
# B, C; B vs A and C vs A strict (loadid.py cmp; depth changes only the verify count -- any difference is VIRHE) -> SUMMARY: per arm
# mean +- half-range, B vs A %, C vs A %, the baseline base1113 S5T and the pre-move stack5 S5T c4 150.6 references.
# NO T 1, no prefill depths, no loaded runs, no turn chains, no c8 newcomer parse. BUDGET XS5_BUDGET_MIN (the item: 17): the B extra
# and the identity are skipped (logged) past it.
# ------------------------------------------------------------------------------------------- the baseline's header (unchanged)
# dsm-baseline.sh (13.11.2026, queue-ar/dsmain/base, ds-main-port base builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-a-base.sh. A COPY of stack5/dsm-stack5.sh (63649414; unchanged) for ONE post-riser BASELINE start
# (cards re-seated 13.11.: 83:00 on its own root complex, 03 and c3 behind risers). Log tag "RESULT xb1 " (stack5's "xs5" in code
# lines only), out XS5_OUT (the item: runs/xdsm/base1113). stack5's helpers are read in place, read only (s5blocks.py / s5ctr.py /
# s5chain.py from ../stack5, stack4's / nc's / pskcr's as stack5 reads them).
# CHANGES vs stack5 (nothing else changed; the knobs XS5_* are stack5's):
#   - turn chains and their cold refs under A: OFF (no call; logged "turn chains: SKIPPED");
#   - identity phase = ONLY greedyset's 20 prompts c1 ALONE under S5T (a1-T), loadid.py cmp vs stack5's runs/xdsm/stack5/
#     s5-n5-a1-T.jsonl (no lid / c4 run-to-run / a2 / copy); the warm-up vs stock and the cold 135k vs dsm-225 compares stay;
#   - T 1 with an empty XS5_ORDER1 logs "T 1: none" (stack5 would log a TRIMMED line with nothing to trim);
#   - a PHASE BUDGET (bud): the extra c8 block, each cold prefill run and the identity are skipped AND LOGGED when the elapsed
#     minutes + their estimate + a reserve (identity + end) would pass XS5_BUDGET_MIN;
#   - basechk after the VRAM line: the RANK -> CARD MAP from the start log's "card <PCI> rank N" lines (+ the link chain, the
#     link-below-max warnings) vs stack5's pre-move map; KVR OFF (RADIANCE_DS41_KV_REPLICATED unset in stackenv, no 'lever KVR
#     declared' line), SP OFF (fn4 stage_full_min / stage_full_pick 0 at start when the image has them);
#   - c8mix: offline later (harness/c8mix.py on t0-engine.log.gz after the run; no extra c8 block on the new harness here);
#   - image check: the fn4 registry literal is int9's (+ stage_full_min, stage_full_pick before epoch_bump) + int9-PROVENANCE;
#   - summary: + the pre-move stack5 S5T cross-start cell per T 0 metric (post vs pre %), stack5's S5T cold TTFT per depth, and
#     stack5's S5T c8 newcomer walls.
# PER-RANK EXPERT TIME is NOT measured here (the step log gives step walls only): it needs a profiled start
#   (RADIANCE_PROFILE_EVERY), not in this item.
# -------------------------------------------------------------------------------------------------- stack5's header (unchanged)
# dsm-stack5.sh (12.11.2026, queue-ar/dsmain/stack5, ds-main-port stack5 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5.sh. Log tag "RESULT xs5 ", out runs/xdsm/stack5 (XS5_OUT). Made from a COPY of
# stack4/dsm-stack4.sh (21ce8c8e: the start, readings, guards, ABBA structure, identity, summary) and of stack4/s4blocks.py
# (0a2cca92 -> s5blocks.py: + read-back, the counter guard per block, the c8 taps); the originals are not changed. + s5ctr.py (the
# counter guard) and s5chain.py (the 3-turn chain, pskcr/pcr2.py's turn spec loaded read only). Read only beside it: stack4/
# (mbt.py, loadid.py, pskld.py, sdprobe.py, editprobe.py), nc/ncmb.py + nc/ncsteps.py, pskcr/pcr2.py + pskcr/pskld.py.
# THE DS STACK CONFIRMATION 5 on the ds41-int8 image (ds41-int7-pskglue 5093ce9 + ds41-pskl20 25aab56 + ds41-pskcr e9e529d + ds41-e1
# 920438e + ds41-noa 0f0294b + ds41-sahb 647830e + ds41-pskcr2 d4540a8 (the PSK-CR turn-chain fix: a replay keeps its snapshot at
# its end; no replay where the snapshot below is the cheaper exact path); every new switch unset = the int7 image: graph and start
# log identical, build/int8 hostcheck). ONE NST 5 START = stack4's start (runs/ds41-exl3.log
# 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start") + RADIANCE_LOG_STEPS=1 + the declares the levers' items used: RADIANCE_DS41_ENGDD=0
# (lever E1, e1/dsm-e1.sh), RADIANCE_PLACE_HB_RANKS=1,2 (lever HB, sahb/dsm-sahb.sh; read only under place_hb > 0). PSK-L20 / PSK-DS
# / PSK-CR need no start word (lever PSK is declared by RADIANCE_DS41_PSK=0, stack4's; their switches start at 0), nor does NOA
# (--prefill-span 2 gives the shared step) or SA (the span's shared level).
# ARMS (each = the fixed pin (stack4's + ds41_psk_k XS5_PSKK) + its own switches, POST + epoch_bump, a nudge, READ BACK; every counter
# baseline is taken AFTER the read-back):
#   A   today's base = stack4's A pin (ds41_bi 0, draft_sample 0, draft_occ 0, place_lf 0, ds41_psk 0) with every new switch 0
#       (sched_bi_chunk 0, sched_noa 0, ds41_psk_l20 0, dspark_pfctx 0, ds41_engdd 0, ds41_psk_cr 0, stage_short / pick / place_hb 0)
#   S   S5: sched_bi_chunk XS5_BIC (3 = D3) + sched_noa XS5_NOA (2) + ds41_psk 2 k XS5_PSKK (8) + ds41_psk_l20 1 + dspark_pfctx 1 +
#       ds41_engdd 1 + draft_sample XS5_DSA (85) + draft_occ XS5_OCC (6) + ds41_bi 2 + place_lf 0 (stack4 S's other pins)
#   T   S5T: S5 with ds41_psk 4 (tail-only) + ds41_psk_cr 1 -- under XS5_T=1 (the int8 image carries pskcr2's turn-chain fix d4540a8)
#   X   S5X: S5 + SA / HB at the usahb item's values: stage_short XS5_SSN (1025), stage_pick XS5_PICK (1 = usahb's default K; set
#       it from usahb's screen when its verdict is in), place_hb XS5_HBR (128) on RADIANCE_PLACE_HB_RANKS XS5_HBRANKS (1,2);
#       in process only under XS5_X=1 (the item sets XS5_X=0: usahb's verdict 12.11. -- SA / HB give no c8 gain, HB costs 1u speed)
# Per start, in this order (decode before prefill):
#   disk >= XS5_DISKMIN_MB (11000) -> gate1 -> modes (stack4's literals + E1 / PSK-L20 / PSK-CR / SA-HB declare lines + step-log lines)
#   -> VRAM at start -> switch note (every new switch 0 at start, SA / HB capable; ds41_psk 99 / draft_occ 65 / sched_noa 3 /
#   stage_short 65537 refused; every arm applied and READ BACK) -> warm-up answers vs stock -> smoke under X (and T): greedy set
#   conc 4 + mb c4 20 s at T 1 -> COLD GUARD: a cold ~XS5_GDEP (16000) deepq prompt under each arm, the counter guard (s5ctr.py):
#   A moves no lever counter; S / T / X move PSK (latched, planned_steps, cut_passes, rows_skipped; plan_mismatch +0; NEVER
#   layers_skipped), PSK-L20 (l20_cut_passes, l20_rows_skipped), PSK-DS (dspark_pfctx passes / rows_skipped), E1 (engdd passes /
#   rows_deduped); T also tail_only_latched; X also hb_reserve = R on the HB ranks. A failed arm: VIRHE, it leaves every order (A or S
#   failed: no speed reading at all, identity only) ->
#   T 0 BLOCKS XS5_ORDER0 (A S X X S A A S X X S A; with T: A S T X X T S A): s5blocks.py, stack4's phases c1 (stepprobe essay /
#     code) / edit / mbc1 / c4 / c8 / pf at temperature 0 (mbt.py), dur XS5_DUR 60 warm XS5_WARM 15; c8 through nc/ncmb.py
#     --mode phase (mb.phase_c itself + per-request taps + /stats at the window's edges); per block the COUNTER GUARD (S / T / X:
#     PSK, L20, PSK-DS, E1, NOA (pieces, rows) move, T tail-only latches, X: SA short passes on every rank + HB reserve = R and
#     held > 0 on the HB ranks; levers off in the arm move nothing; A moves nothing) -- a failed block is flagged, no reading
#   C8 NEWCOMER SHARE (after the blocks, from the step log, nc/ncsteps.py c8 read only, a block each): decode_share, the newcomer's
#     own step wall (p50), MIXED / prefill newcomers, the wait; freshness judged on cached == 0 only (the prompt-size spread is
#     information: ncsteps' 900..1300 window flags mb's 1335-1631-token prompts)
#   T 1 BLOCKS XS5_ORDER1 (A S X X S A; with T: A S T X X T S A): sdprobe essay / code / edit at the server default (T 1, seed 1234,
#     thinking on); guard: S / T / X latch PSK (T tail-only, X the HB reserve), A moves nothing. TRIMMED FIRST when over budget.
#   PREFILL in process, XS5_PORDER (A S X X S A / A S T X X T S A): cold TTFT at 135k and 64k (deepq.py's fixed prompt, a prefix-
#     cache reset before each, 512 greedy tokens), the cold guard per run -> the c4 + cold 64k decode-under-prefill reading,
#     XS5_LORDER (pskld.py --load; guard per run)
#   TURN CHAINS: per arm a 3-turn chain (s5chain.py: pcr2.py's turn spec + a third turn), turn 1 cold ~XS5_CDEPTH (48000) tokens
#     after a prefix-cache reset, each later turn = previous + answer (XS5_CNTOK 256 greedy tokens) + follow-up, the arm's own text
#     offset (ds41_tests.py source-file index 400 / 430 / 460 / 490: deepq's prompt starts at 0, pskcr2 used 10..341) and nonce; per turn prompt / cached / TTFT and the counters (CR: cr_hits / cr_not_needed / cr_passes / fallbacks);
#     (+ cr_cheaper / cr_end_snapshots); chain guard (T: PSK-CR looked at the hits -- replayed, not needed or cheaper -- no fallback);
#     then every chain's turns 2 and 3 re-run COLD under A (refs)
#   IDENTITY at T 0, strict: alone A / S / X (/ T) (greedy set c1) vs bz d0 and vs A; under load: S and X each greedy prompt under mb
#     c4 + a fresh ~32k prefill beside its decode vs alone; c4 run-to-run S; alone again S; the 28k copy under S vs bz d0; every
#     136k cold answer vs dsm-225 and vs A at its depth; the 64k answer under c4 load vs alone (S / T / X strict -- D3 keeps
#     alone's chunks; A at sched_bi_chunk 0: the KNOWN class (ii) difference, labelled); ON HITS: every chain's turn-2 / turn-3
#     answer (a prefix hit) vs the same prompt cold under A
# SUMMARY: per T 0 / T 1 phase and arm mean +- t95, S vs A, T vs S, X vs S (pooled + adjacent pairs); targets (1u >= 90, c4 >= 160,
#   c8 >= 200, 136k <= 35 s); cross-start stack4 (int7 image) A / S, labelled; c8 newcomer share per arm; prefill / loaded / chain
#   tables; guards; identity verdict.
# ~116 min estimate with A / S / T (the item's default, XS5_X=0; start + checks + the T smoke ~7, cold guard ~2, T 0 12 blocks
#   A S T T S A A S T T S A x ~5.1 = ~61, c8 parse ~2, T 1 6 (A S T T S A) x ~1.2 = ~7, prefill 6 runs a depth ~10, loaded 6 ~8,
#   chains + refs ~5, identity ~13, parsing ~2); ~115 with A / S / T / X (8 T 0 blocks A S T X X T S A). BUDGET XS5_BUDGET_MIN 150: T 1 is TRIMMED (skipped, logged) when the time at its start + its estimate + the
#   rest's (15 min an arm) would pass the budget. NO evict.
# NO cc_sim, no 2.9; no power / host / fan settings; production compose files read only (--project-directory). Exact by design
#   (every lever byte-identical); identity reported, never hidden.
# Knobs: XS5_IMAGE, XS5_ITEM, XS5_OUT, XS5_T (0 | 1), XS5_X (1 | 0), XS5_PSKK (8), XS5_BIC (3), XS5_NOA (2), XS5_OCC (6), XS5_OCCK (32),
#   XS5_DSA (85), XS5_SSN (1025), XS5_PICK (1), XS5_HBR (128), XS5_HBRANKS (1,2), XS5_ORDER0, XS5_ORDER1, XS5_PORDER, XS5_LORDER, XS5_DUR
#   (60), XS5_WARM (15), XS5_EDIT (1 | 0), XS5_GDEP (16000), XS5_DEPTHS ("135000 64000"), XS5_LDEPTH (64000), XS5_CDEPTH (48000),
#   XS5_CNTOK (256), XS5_DISKMIN_MB (11000), XS5_BUDGET_MIN (150), XS5_SIDECAR, XS5_SIDESHA.
# DRYRUN=1 bash dsm-stack5.sh : prints every command; checks the image literals (every lever's registry name, range message, counters,
#   declare line), every helper (files, parse, selftests: loadid.py / mbt.py / pskld.py / ncmb.py / ncsteps.py / s5ctr.py / s5chain.py /
#   pcr2.py texts, s5blocks.py / sdprobe.py --help), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XS5_OUT:-$R/xdsm/stack5}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS4=$R/xdsm/stack4
RS5=$R/xdsm/stack5                               # base: stack5's out dir, the pre-move reference (read only)
RB=$R/xdsm/base1113                              # d0ab: the post-riser baseline's out dir (read only)
export IMAGE=${XS5_IMAGE:?set XS5_IMAGE}
ITEM=${XS5_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5}
DOT=${XS5_T:-0}; DOX=${XS5_X:-1}; PSKK=${XS5_PSKK:-8}; BIC=${XS5_BIC:-3}; NOA=${XS5_NOA:-2}; OCC=${XS5_OCC:-6}; OCCK=${XS5_OCCK:-32}; DSA=${XS5_DSA:-85}
SSN=${XS5_SSN:-1025}; PICK=${XS5_PICK:-1}; HBR=${XS5_HBR:-128}; HBRANKS=${XS5_HBRANKS:-1,2}
case "$DOT$DOX" in 00) DO0=ASSAASSA; DO1=ASSA; DOP=ASSA ;; 01) DO0=ASXXSAASXXSA; DO1=ASXXSA; DOP=ASXXSA ;; 10) DO0=ASTTSAASTTSA; DO1=ASTTSA; DOP=ASTTSA ;;
  11) DO0=ASTXXTSA; DO1=ASTXXTSA; DOP=ASTXXTSA ;; *) DO0=; DO1=; DOP= ;; esac
ORDER0=${XS5_ORDER0-$DO0}; ORDER1=${XS5_ORDER1-$DO1}; PORDER=${XS5_PORDER:-$DOP}; LORDER=${XS5_LORDER-$DOP}
DUR=${XS5_DUR:-60}; WARMS=${XS5_WARM:-15}; EDIT=${XS5_EDIT:-1}
GDEP=${XS5_GDEP:-16000}; DEPTHS=${XS5_DEPTHS:-135000 64000}; LDEPTH=${XS5_LDEPTH:-64000}; CDEPTH=${XS5_CDEPTH:-48000}; CNTOK=${XS5_CNTOK:-256}; NTOK=512
DISKMIN=${XS5_DISKMIN_MB:-11000}; BUDGET=${XS5_BUDGET_MIN:-150}
SIDE=${XS5_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XS5_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
OFF=0.5; PFT=32768; CONC=4; WARM=10; MT=192; SEED=1234
case $DOT in 0|1) ;; *) echo "XS5_T must be 0 or 1 (got '$DOT')"; exit 1 ;; esac
case $DOX in 0|1) ;; *) echo "XS5_X must be 0 or 1 (got '$DOX')"; exit 1 ;; esac
case $EDIT in 0|1) ;; *) echo "XS5_EDIT must be 0 or 1 (got '$EDIT')"; exit 1 ;; esac
ARMS=AS$([ $DOT = 1 ] && echo T)$([ $DOX = 1 ] && echo X)
ARMS=ABC   # d0ab: A / B / C = S5T + the depth-0 controller pair (armj)
for o in "$ORDER0" "$ORDER1" "$PORDER" "$LORDER"; do case $o in *[!$ARMS]*) echo "an order names an arm outside $ARMS ('$o'; T needs XS5_T=1, X XS5_X=1)"; exit 1 ;; esac; done
[ -n "$PORDER" ] || { echo "XS5_PORDER is empty"; exit 1; }
for v in $PSKK $BIC $NOA $OCC $OCCK $DSA $SSN $PICK $HBR $DUR $WARMS $LDEPTH $GDEP $CDEPTH $CNTOK $DISKMIN $BUDGET $DEPTHS; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case $HBRANKS in *[!0-9,]*|"") echo "XS5_HBRANKS takes a comma list of ranks (got '$HBRANKS')"; exit 1 ;; esac
[ $BIC = 3 ] || [ $BIC = 4 ] || { echo "XS5_BIC is 3 or 4 (NOA needs option D)"; exit 1; }
[ $NOA -ge 1 ] && [ $NOA -le 2 ] && [ $PSKK -ge 1 ] && [ $PSKK -le 4096 ] && [ $SSN -ge 1 ] && [ $SSN -le 65536 ] && [ $PICK -le 64 ] && [ $HBR -ge 1 ] && [ $HBR -le 4096 ] \
  || { echo "XS5_NOA 1..2, XS5_PSKK 1..4096, XS5_SSN 1..65536, XS5_PICK 0..64, XS5_HBR 1..4096"; exit 1; }
[ $WARMS -lt $DUR ] || { echo "the warm-up is not shorter than the phase"; exit 1; }
case " $DEPTHS " in *" $LDEPTH "*) ;; *) [ -z "$LORDER" ] || { echo "XS5_LDEPTH ($LDEPTH) must be one of XS5_DEPTHS ($DEPTHS): the loaded answer is compared with the alone one"; exit 1; } ;; esac
PH0=c4; SETTLE=${XD0_SETTLE:-5}; T1P=essay,code$([ $EDIT = 1 ] && echo ,edit)
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py; MBT=$S4/mbt.py; PRB=$S4/sdprobe.py; EP=$S4/editprobe.py; PL=$S4/pskld.py      # stack4's, read only
S5D=$HD/../stack5                                                                                   # base: stack5's helpers, read only
GGB=$HD/d0blocks.py; C4S=$HD/c4split.py; CTR=$S5D/s5ctr.py; CHN=$S5D/s5chain.py                                     # stack5's
NCMB=$HD/../nc/ncmb.py; NCS=$HD/../nc/ncsteps.py                                                  # nc's, read only
PCR2=$HD/../pskcr/pcr2.py; PLC=$HD/../pskcr/pskld.py                                              # pskcr's, read only
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xd0 $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= stack4's + RADIANCE_LOG_STEPS=1 + the lever declares)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
# + this item's levers' declare lines (the lever items' own literals: e1/dsm-e1.sh, pskl20/dsm-pskl20.sh, pskcr/dsm-pskcr2.sh, sahb/dsm-sahb.sh)
LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD|now,0):,the,Engram,lookup,/,copy,/,kv,GEMM,(25600,x,6144)|rows,at,2,Engram,layer(s)"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up|stage_short,can,arm,the,prefill,stager,for,a,step,of,1025,..,2112,tokens|place_hb,can,reserve,on,rank,1"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
NEW0='"sched_bi_chunk": 0, "sched_noa": 0, "ds41_psk_l20": 0, "dspark_pfctx": 0, "ds41_engdd": 0, "ds41_psk_cr": 0, "stage_short": 0, "stage_pick": 0, "place_hb": 0'
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
T5J="$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0"
armj() {   # d0ab: armj A|B|C = stack5's S5T switches + the depth-0 controller pair (overrides the fixed pin's draft_depth0 1 / draft_lend 1)
  case $1 in
    A) echo "{$T5J, \"draft_depth0\": 1, \"draft_lend\": 1}" ;;
    B) echo "{$T5J, \"draft_depth0\": 0, \"draft_lend\": 1}" ;;
    C) echo "{$T5J, \"draft_depth0\": 1, \"draft_lend\": 0}" ;;
  esac
}
NAMEOF() { case $1 in A) echo "A (S5T, depth0 1 lend 1)" ;; B) echo "B (S5T, depth0 0)" ;; C) echo "C (S5T, depth0 1 lend 0)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
ALLOFF=psk,l20,pfctx,engdd,noa,sa,hb,cr
needs() {   # d0ab: every arm carries S5T's levers; c4 / c1 / c8 blocks need no lever move (SA / HB forbidden); cold / load / chain: S5T's
  case $1 in blk|t1) echo "/sa,hb" ;; *) echo "psk,tail,l20,pfctx,engdd/sa,hb" ;; esac
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS; int9, KVR / SP off) | fixed pin: stack4's + ds41_psk_k $PSKK | d0ab arms A $(armj A), B: draft_depth0 0, C: draft_lend 0"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_PLACE_HB_RANKS
  RADIANCE_PLACE_HB_DECAY RADIANCE_DS41_PSK_L20 RADIANCE_DS41_DSPARK_PFCTX RADIANCE_DS41_PSK_CR RADIANCE_SCHED_BI_CHUNK
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED"
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xs5)
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
capture() {   # capture FILE: the engine log with the daemon's receipt times (ncsteps.py reads them), gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before the start (sahb's)
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (the start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse the start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "$1 warm-up $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, step-log lines, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; step-log lines (RADIANCE_LOG_STEPS=1); literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever E1|lever SA / HB|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|prefill staging:" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-3600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xd0 $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS4/s4-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack4 s4-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
ARMOK=1
arm() {   # arm ARM: the pin + the arm + epoch_bump, a nudge, the switches READ BACK (polled <= 30 s) -> 0, else VIRHE + 1
  local g r i
  fn4set "$(pinj $1 '{"epoch_bump": 1}')"; nudge
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; return 0; }
  for i in $(seq 1 30); do
    g=$(fn4get)
    r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
print("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"]))' "$g" "$(pinj $1)" 2>&1 | cut -c1-400)
    case "$r" in OK*) return 0 ;; esac
    sleep 1; nudge
  done
  xlog "arm $1: VIRHE the pin is not applied after 30 s ($r) -- no reading for this arm here"; ARMOK=0; return 1
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window
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
pskstat() {   # the live lever counters (cumulative, /stats): PSK, PSK-L20, PSK-CR, PSK-DS, E1
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  curl -s -m 20 $URL/stats 2>/dev/null | python3 -c 'import json,sys
s = json.load(sys.stdin); p = s.get("psk") or {}; e = s.get("engdd") or {}; f = s.get("dspark_pfctx") or {}
print("psk mode %s k %s latched %s planned %s cut %s rows %s layers %s mismatch %s | l20 %s / %s | cr %s hits %s not_needed %s fb %s tail %s | pfctx %s / %s | engdd %s / %s" % (
    p.get("mode"), p.get("k"), p.get("latched"), p.get("planned_steps"), p.get("cut_passes"), (p.get("rows_skipped_l21_29") or 0) + (p.get("rows_skipped_l30_39") or 0),
    p.get("layers_skipped"), p.get("plan_mismatch"), p.get("l20_cut_passes"), p.get("l20_rows_skipped"), p.get("cr"), p.get("cr_hits"), p.get("cr_not_needed"),
    p.get("cr_fallbacks"), p.get("tail_only_latched"), f.get("passes"), f.get("rows_skipped"), e.get("passes"), e.get("rows_deduped")))' 2>/dev/null | cut -c1-500
}
ssnap() {   # ssnap NAME: the full /stats and GET /admin/fn4 (the counter guard's snapshot)
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json; GET /admin/fn4 > $O/fn-$1.json"; return 0; }
  curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; curl -s -m 10 $URL/admin/fn4 > $O/fn-$1.json 2>/dev/null
}
sdelta() {   # sdelta A B -> JSON: checkpoints written, staged bytes (all ranks / rank 0), rank-0 staged layers, all-reduce bytes
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda d, *k: (lambda v: v if isinstance(v, (int, float)) else 0)(__import__("functools").reduce(lambda x, y: (x or {}).get(y) if isinstance(x, dict) else None, k, d))
print(json.dumps(dict(ckpts=int(g(b, "checkpoints_written") - g(a, "checkpoints_written")),
      staged_gb=round((g(b, "link", "staged_bytes") - g(a, "link", "staged_bytes")) / 1e9, 2),
      staged_gb_r0=round((g(b, "experts", "staged_bytes") - g(a, "experts", "staged_bytes")) / 1e9, 2),
      stage_layers_r0=int(g(b, "experts", "stage_layers") - g(a, "experts", "stage_layers")),
      ar_gb=round((g(b, "link", "ar_bytes") - g(a, "link", "ar_bytes")) / 1e9, 2))))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'
}
ctrg() {   # ctrg KIND ARM A B: the counter guard between snapshots A and B -> "OK | ..." / "FAIL ... | ..."
  local s=$(needs $1 $2)
  [ "$DRY" = 1 ] && { echo "OK (dry: $CTR guard --need ${s%%/*} --forbid ${s#*/} --hb $HBR --hb-ranks $HBRANKS)"; return 0; }
  timeout 60 python3 -I -B $CTR guard --need "${s%%/*}" --forbid "${s#*/}" --st0 $O/st-$3.json --st1 $O/st-$4.json --fn0 $O/fn-$3.json --fn1 $O/fn-$4.json \
    --hb $HBR --hb-ranks $HBRANKS 2>&1 | tail -1 | cut -c1-1200
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, 512 greedy tokens -> TTFT, cached, acceptance, ms, the cold guard
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md gr
  arm $a || return 1
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOK > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  gr=$(ctrg cold $a $tag-$nm-0 $tag-$nm-1)
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 $tag $nm: arm $a depth $dep: TTFT, cached, acceptance, non-decode step ms, /stats deltas, counter guard ($gr) -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-)
  md=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); d.update(json.loads(sys.argv[2] or "{}")); print(json.dumps(d))' "$(mdelta $tag-$nm-0 $tag-$nm-1)" "$(sdelta $tag-$nm-0 $tag-$nm-1)" 2>/dev/null || mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], arm=sys.argv[4], depth=int(sys.argv[5]), reset=sys.argv[6] != "noreset", text=sys.argv[7], guard=sys.argv[8].split(" | ")[0])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$dep" "${5:-reset}" "$O/$tag-$nm.txt" "$gr" >> $O/deep.jsonl
  case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
  xlog "$tag $nm (arm $(NAMEOF $a), deepq depth $dep, $NTOK greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | decode acceptance %s (%s of %s drafted) | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB (rank 0 %s GB, %s layers), all-reduce %s GB" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
    round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars"), m.get("acc"), m.get("draft_acc"), m.get("draft_tok"), m.get("nondecode_steps"), m.get("nondecode_ms"),
    m.get("ckpts"), m.get("staged_gb"), m.get("staged_gb_r0"), m.get("stage_layers_r0"), m.get("ar_gb")))' "$j" "$md" 2>&1 | cut -c1-600) | counters: $gr | $(thermline $t0 $(date +%s))"
  case "$gr" in OK*) return 0 ;; *) return 2 ;; esac
}
coldabba() {   # coldabba TAG: per depth, the arms in PORDER, each cold; identity: 135k vs dsm-225, every arm vs the first A of its depth
  local tag=$1 dep a k fa
  for dep in $DEPTHS; do
    k=0; fa=""
    for a in $(echo $PORDER | grep -o .); do
      k=$((k + 1)); stopfile && return 0
      bud "prefill d$dep-$k$a (cold, arm $(NAMEOF $a))" $((1 + dep / 100000)) 2 || continue
      deepx $tag $a $dep d$dep-$k$a
      [ "$DRY" = 1 ] && continue
      [ -s $O/$tag-d$dep-$k$a.txt ] || continue
      [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-d$dep-$k$a.txt; }
      [ $dep = 135000 ] && idrec "deep135-$k$a-vs-dsm225" "$(cmpfile $O/$tag-d$dep-$k$a.txt $RM/deep-dsm-225.txt)" "135k answer vs dsm-225 NST 0 stock cold"
      [ -n "$fa" ] && [ "$O/$tag-d$dep-$k$a.txt" != "$fa" ] && idrec "d$dep-$k$a-vs-A" "$(cmpfile $fa $O/$tag-d$dep-$k$a.txt)" "depth $dep, vs the first A"
    done
  done
}
copyx() {   # copyx TAG ARM: the 28k copy (dsm-stack2.sh copyt's request) after a reset -> vs bz d0
  local tag=$1 t0=$(date +%s)
  arm $2 || return 1; pcreset $tag-copy$2
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag-$2 --outdir $O > $O/$tag-copy$2.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag copy arm $(NAMEOF $2) (after a cold ~31k prompt): $(grep '^J ' $O/$tag-copy$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
  idrec "copy-$2-vs-d0" "$(cmpfile $O/copy-$tag-$2-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" "28k copy vs bz d0 (NST 0 stock)"
}
loadx() {   # loadx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (pskld.py --load, no extra prefill); the load guard
  local tag=$1 a=$2 nm=$3 t0=$(date +%s) j md gr
  arm $a || return 1; pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $NTOK --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  gr=$(ctrg load $a $tag-$nm-0 $tag-$nm-1)
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 $tag $nm: arm $a: mixed-step ms, c4 tok/s in the prefill window (agg / per-stream p50), TPOT, TTFT, counter guard ($gr) -> $O/load.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {}); d.update(name=sys.argv[3], arm=sys.argv[4], text=sys.argv[5], guard=sys.argv[6].split(" | ")[0])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$O/$tag-$nm.txt" "$gr" >> $O/load.jsonl
  case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
  xlog "$tag $nm (arm $(NAMEOF $a), c4 decode + one cold ~$LDEPTH-depth deepq prompt, T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 %s tok, agg %s tok/s, per-stream p50 %s tok/s, TPOT p50 %s / p90 %s / max %s ms | mixed steps %s (non-decode %s at %s ms mean) | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_tokens"), d.get("c4_pf_agg_tps"),
    d.get("c4_pf_stream_tps_p50"), d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), d.get("tpot_pf_max"), m.get("mixed_steps"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-700) | counters: $gr | $(thermline $t0 $(date +%s))"
}
loadabba() {   # loadabba TAG: the arms in LORDER; identity: each loaded 64k answer vs the same arm's alone 64k answer and vs A
  local tag=$1 a k al fa=""
  [ -n "$LORDER" ] || return 0
  k=0
  for a in $(echo $LORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    loadx $tag $a ld-$k$a
    [ "$DRY" = 1 ] && continue
    [ -s $O/$tag-ld-$k$a.txt ] || continue
    al=$(ls $O/$tag-d$LDEPTH-*$a.txt 2>/dev/null | head -1)
    [ -n "$al" ] && idrec "ld-$k$a-vs-alone$a" "$(cmpfile $al $O/$tag-ld-$k$a.txt)" "the ~$LDEPTH prompt under c4 load vs alone, same arm"
    [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-ld-$k$a.txt; }
    [ -n "$fa" ] && [ "$O/$tag-ld-$k$a.txt" != "$fa" ] && idrec "ld-$k$a-vs-ldA" "$(cmpfile $fa $O/$tag-ld-$k$a.txt)" "loaded, vs the first loaded A"
  done
}
# ---------------------------------------------------------------- stack4's own functions, for the stack5 arms
swnote() {   # swnote TAG: start values; refusals; every arm applied and read back, then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, ds41_bi 0, ds41_psk 0, place_lf 0, draft_occ 0, draft_occ_k 32, draft_sample 0 + sample.capable 1, draft_capable 1, epoch 0; sched_bi_chunk 0, sched_noa 0, ds41_psk_l20 0, dspark_pfctx 0, ds41_engdd 0, ds41_psk_cr 0, stage_short / stage_pick / place_hb 0, sahb stage_capable 1 hb_capable 1 on 4 ranks); refusals ds41_psk 99 / draft_occ 65 / sched_noa 3 / stage_short 65537 -> 400; arms $(for a in $(echo $ARMS | grep -o .); do echo -n "$a $(pinj $a) | "; done)each applied and read back, then A"; return 0; }
  local s0 g c1 c2 c3 c4 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_psk": 99}'); c2=$(fn4code '{"draft_occ": 65}'); c3=$(fn4code '{"sched_noa": 3}'); c4=$(fn4code '{"stage_short": 65537}')
  for a in $(echo $ARMS | grep -o .) A; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("bi=%s,ds=%s,occ=%s,psk=%s/%s,l20=%s,pf=%s,ed=%s,cr=%s,bic=%s,noa=%s,ss=%s,pick=%s,hb=%s,ep=%s" % (w.get("ds41_bi"), w.get("draft_sample"), w.get("draft_occ"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_l20"), w.get("dspark_pfctx"), w.get("ds41_engdd"), w.get("ds41_psk_cr"), w.get("sched_bi_chunk"), w.get("sched_noa"), w.get("stage_short"), w.get("stage_pick"), w.get("place_hb"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1600) | ds41_psk 99 -> HTTP $c1, draft_occ 65 -> HTTP $c2, sched_noa 3 -> HTTP $c3, stage_short 65537 -> HTTP $c4 | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; sa = s0.get("sahb") or {}
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("place_lf") == 0 and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and (s0.get("sample") or {}).get("capable") == 1
ok = ok and all(w.get(k) == 0 for k in ("sched_bi_chunk", "sched_noa", "ds41_psk_l20", "dspark_pfctx", "ds41_engdd", "ds41_psk_cr", "stage_short", "stage_pick", "place_hb"))
ok = ok and sa.get("stage_capable") == 1 and sa.get("hb_capable") == 1 and len(sa.get("ranks") or []) == 4
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ "$c4" = 400 ] && [ -z "$bad" ]
}
smoke() {   # smoke TAG ARM: the greedy set at conc 4 (mixed steps) + mb c4 20 s at T 1 (the coupled path) under ARM
  local tag=$1 sa=$2 t0=$(date +%s) n e
  arm $sa || return 1
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke$sa.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke$sa.txt 2>&1)"
  run "(cd $D && timeout 600 python3 mb.py --tag xs5-smoke --phases c4 --dur 20 --warm 5 > $O/$tag-smoke$sa-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm $sa $(armj $sa): greedy set conc 4 (32 tok) 20 of 20 + mb c4 20 s at T 1 with 0 errors, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke$sa.jsonl 2>/dev/null || echo 0)
  e=$(grep -m1 '^MB ' $O/$tag-smoke$sa-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && case "$e" in *" 0 errors"*) true ;; *) false ;; esac && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (arm $(NAMEOF $sa)): greedy set conc 4 20 of 20, mb c4 T 1: $e, healthy, $(( $(date +%s) - t0 )) s | $(pskstat)"; return 0; fi
  xlog "$tag smoke (arm $(NAMEOF $sa)): VIRHE $n of 20 answered, mb c4 T 1: $e, or unhealthy | $(docker logs --tail 3000 $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke$sa; return 1
}
SPEEDOK=1
cguard() {   # cguard TAG: before any speed reading, a cold ~GDEP prompt under every arm and its counter guard; a failed arm leaves the orders
  local tag=$1 a r bad=""
  for a in $(echo $ARMS | grep -o .); do
    stopfile && return 0
    deepx $tag $a $GDEP guard-$a; r=$?
    if [ $r != 0 ] && [ "$DRY" != 1 ]; then bad="$bad $a"; fi
  done
  [ "$DRY" = 1 ] && { echo "+ cold guard: arms $ARMS each a cold ~${GDEP}-depth deepq prompt after a reset; s5ctr.py on the /stats + /admin/fn4 around it: A moves nothing ($ALLOFF), S needs [$(needs cold S)], T [$(needs cold T)], X [$(needs cold X)] (need/forbid); a failed arm leaves every order (A or S: no speed reading)"; return 0; }
  echo "cold guard failed arms:${bad:- none}" >> $O/guard.txt
  [ -z "$bad" ] && { xlog "$tag cold guard: OK in every arm ($ARMS)"; return 0; }
  case "$bad" in *A*|*S*) SPEEDOK=0; xlog "$tag cold guard: VIRHE in arm(s)$bad -- A / S failed: no speed reading (identity only)" ;;
    *) xlog "$tag cold guard: VIRHE in arm(s)$bad -- they leave every order and the chains"
       for a in $bad; do ORDER0=${ORDER0//$a/}; ORDER1=${ORDER1//$a/}; PORDER=${PORDER//$a/}; LORDER=${LORDER//$a/}; ARMS=${ARMS//$a/}; done ;; esac
}
blocks() {   # blocks TAG KIND: t0 = XS5_ORDER0 (stack4's phases at temperature 0, c8 through ncmb.py), t1 = XS5_ORDER1 (sdprobe at the server default)
  local tag=$1 k=$2 ord ph mbx="" args a gk
  if [ $k = t0 ]; then ord=$ORDER0; ph=$PH0; gk=blk; mbx="--mbtemp 0 --mbt $MBT --c8-ncmb $NCMB --c8-dir $O/c8"; else ord=$ORDER1; ph=t1; gk=t1; fi
  [ -n "$ord" ] || return 0
  args=""; for a in $(echo $ord | grep -o . | awk '!s[$0]++'); do args="$args --arm '$a=$(armj $a)' --guard '$a=$(needs $gk $a)'"; done
  run "mkdir -p $O/c8 $O/snap; timeout 14400 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --seed $SEED --t1-prompts $T1P --label $tag-$k $args --hb $HBR --hb-ranks $HBRANKS --snap-dir $O/snap --settle $SETTLE --order $ord --phases $ph --dur $DUR --warm $WARMS $mbx --fixed '$FIX' --out $O/$tag-$k-blocks.jsonl > $O/$tag-$k-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do case "$l" in "GUARD "*"FAIL"*) xlog "$tag $k VIRHE ${l:0:1500}" ;; *) xlog "$tag $k ${l:0:1700}" ;; esac; done < <(grep -E '^(BLOCK|T1|GUARD|OCC|STATS|GGB \{"error)|Traceback' $O/$tag-$k-blocks.txt)
  xlog "$tag $k engine: $(pskstat) | draft $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("draft")), json.dumps(d.get("sample")), json.dumps(d.get("noa")))' 2>/dev/null | cut -c1-600) | $(docker logs --since 4h $CN 2>&1 | grep -cE '^E ') E lines"
}
ncparse() {   # ncparse: the c8 newcomer share per T 0 block from the step log (nc/ncsteps.py c8, read only; freshness = cached 0)
  local b
  capture $O/t0-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per T 0 block: python3 -I -B $NCS c8 --log $O/t0-engine.log.gz --reqs $O/c8/<label>-b<i><arm>-c8-reqs.jsonl --tag <label>-b<i><arm>-c8 --sa/--sb the window's /stats --samples --mbline -> $O/nc8-<block>.json; RESULT xd0 c8 newcomer share lines"; return 0; }
  for b in $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    if l.strip():
        r = json.loads(l); print("%s-b%d%s" % (r["label"], r["block"], r["arm"]))' $O/s5-n5-t0-blocks.jsonl 2>/dev/null); do
    [ -s $O/c8/$b-c8-reqs.jsonl ] || { xlog "c8 $b: no ncmb taps (block not run or flagged)"; continue; }
    timeout 900 python3 -I -B $NCS c8 --log $O/t0-engine.log.gz --reqs $O/c8/$b-c8-reqs.jsonl --tag $b-c8 --sa $O/c8/$b-c8-st-w0.json --sb $O/c8/$b-c8-st-w1.json \
      --samples $O/c8/$b-c8-samp.jsonl --mbline $O/c8/$b-c8.txt --label "c8 $b" --json $O/nc8-$b.json > $O/nc8-$b.out 2>&1
    xlog "c8 newcomers $b: $(python3 -c 'import json,sys
d = json.load(open(sys.argv[1])); n = d["newcomers"]; rows = n.get("rows") or []
g = [x for x in d.get("guard", []) if not x.startswith("not fresh")]   # freshness: cached 0 only (the size window is information)
warm = [x for x in rows if (x.get("cached") or 0) != 0]
if warm: g.append("%d newcomer(s) with cached > 0" % len(warm))
ps = sorted(x["prompt"] for x in rows if x.get("prompt"))
print("decode_share %s (MB %s), agg_dec_tps %s | newcomers %d (%d found): %d MIXED / %d prefill, own step wall p50 %s / mean %s ms, wait p50 %s ms, TTFT p50 %s s | prompt sizes %s..%s (information) | MIXED steps %s at p50 %s ms, decode step p50 %s ms%s" % (
    (d.get("stats") or {}).get("decode_share"), d["mb"].get("decode_share"), d["mb"].get("agg_dec_tps"), n["n"], n["found"], n["mixed"], n["prefill"],
    n["wall_ms"].get("p50"), n["wall_ms"].get("mean"), n["wait_ms"].get("p50"), n["ttft_s"].get("p50"), ps[0] if ps else "-", ps[-1] if ps else "-",
    d["classes"]["MIXED"]["steps"], d["classes"]["MIXED"]["wall_ms"].get("p50"), d["classes"]["decode"]["wall_ms"].get("p50"),
    (" -- VIRHE " + "; ".join(g)) if g else ""))' $O/nc8-$b.json 2>&1 | tail -1 | cut -c1-900)"
  done
}
t1budget() {   # t1budget: 0 = run T 1; 1 = trim it (the time now + its estimate + the rest's would pass XS5_BUDGET_MIN)
  local e n est rest
  e=$(elapsed); n=${#ORDER1}; est=$(( (n * 13 + 9) / 10 )); rest=$(( 15 * ${#ARMS} ))
  [ "$DRY" = 1 ] && { echo "+ T 1 budget: at its start elapsed + ~$est min (T 1, $n blocks) + ~$rest min (prefill, loaded, chains, identity) <= $BUDGET min, else T 1 is trimmed (logged)"; return 0; }
  [ $((e + est + rest)) -le $BUDGET ] && return 0
  xlog "T 1 TRIMMED (budget): $e min elapsed + ~$est min T 1 + ~$rest min for the rest > $BUDGET min -- no T 1 reading"; return 1
}
CHAINS=""
chainx() {   # chainx TAG ARM OFFSET: the 3-turn chain under ARM (reset first): per turn prompt / cached / TTFT + the counters; chain guard
  local tag=$1 a=$2 off=$3 nc t j gr k n0 p=$tag-ch$2
  nc="$a-$T00"
  arm $a || return 1; pcreset $p
  ssnap $p-0; n0=$p-0
  for t in 1 2 3; do
    stopfile && return 0
    run "(cd $D && timeout 1800 python3 -I -B $CHN run --pcr2 $PCR2 --pskld $PLC --tests $TESTS --url $URL --turn $t --depth $CDEPTH --off $off --nonce $nc --ans1 $O/$p-t1.txt --ans2 $O/$p-t2.txt --ntok $CNTOK --out $O/$p-t$t.json --text $O/$p-t$t.txt > $O/$p-t$t.log 2>&1)"
    ssnap $p-$t
    [ "$DRY" = 1 ] && continue
    j=$(grep -m1 '^LONGID ' $O/$p-t$t.log | cut -c8-)
    [ -n "$j" ] || { xlog "$p turn $t: VIRHE no LONGID line ($(tail -3 $O/$p-t$t.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
    k=$(timeout 60 python3 -I -B $CTR guard --st0 $O/st-$p-$((t - 1)).json --st1 $O/st-$p-$t.json --fn0 $O/fn-$p-$((t - 1)).json --fn1 $O/fn-$p-$t.json 2>&1 | tail -1 | sed 's/^OK | //' | cut -c1-500)
    python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(arm=sys.argv[2], turn=int(sys.argv[3]), text=sys.argv[4], ctr=sys.argv[5]); print(json.dumps(d))' "$j" "$a" "$t" "$O/$p-t$t.txt" "$k" >> $O/chain.jsonl
    xlog "$p turn $t (arm $(NAMEOF $a), ~$CDEPTH-token first turn at offset $off, nonce $nc, $CNTOK greedy tokens): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1]); print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens")))' "$j" 2>&1 | cut -c1-300) | counters: $k"
  done
  [ "$DRY" = 1 ] && { echo "+ chain guard arm $a: [$(needs chain $a)] (need/forbid) over turns 1-3"; CHAINS="$CHAINS $a:$off"; return 0; }
  gr=$(ctrg chain $a $p-0 $p-3)
  echo "chain $a $gr" | cut -c1-300 >> $O/guard.txt
  case "$gr" in OK*) xlog "$p chain guard: $gr" ;; *) xlog "$p chain guard: VIRHE $gr" ;; esac
  CHAINS="$CHAINS $a:$off"
}
chains() {   # chains TAG: one chain per arm (its own text offset and nonce), then every chain's turns 2 / 3 COLD under A -> identity on hits
  local tag=$1 a i=0 c off t p
  for a in $(echo $ARMS | grep -o .); do i=$((i + 1)); stopfile && return 0; chainx $tag $a $((370 + i * 30)); done   # text file offsets 400, 430, ... (ds41_tests SRC index; deepq is 0, pskcr2 10..341)
  for c in $CHAINS; do
    a=${c%%:*}; off=${c#*:}; p=$tag-ch$a
    for t in 2 3; do
      stopfile && return 0
      [ "$DRY" = 1 ] || [ -s $O/$p-t$t.txt ] || continue
      arm A || return 1; pcreset $p-ref$t
      run "(cd $D && timeout 1800 python3 -I -B $CHN run --pcr2 $PCR2 --pskld $PLC --tests $TESTS --url $URL --turn $t --depth $CDEPTH --off $off --nonce $a-$T00 --ans1 $O/$p-t1.txt --ans2 $O/$p-t2.txt --ntok $CNTOK --out $O/$p-ref$t.json --text $O/$p-ref$t.txt > $O/$p-ref$t.log 2>&1)"
      [ "$DRY" = 1 ] && continue
      xlog "$p ref $t (turn $t's prompt COLD under A): $(grep -m1 '^LONGID ' $O/$p-ref$t.log | cut -c8- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("prompt %s tok, cached %s, TTFT %s s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft")))' 2>&1 | cut -c1-200)"
      idrec "hit-$a-t$t-vs-coldA" "$(cmpfile $O/$p-ref$t.txt $O/$p-t$t.txt)" "turn $t of the $(NAMEOF $a) chain (a prefix hit) vs the same prompt cold under A"
    done
  done
}
alone() {   # alone TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t0=$(date +%s)
  arm $2 || return 1; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $(NAMEOF $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
lidrun() {   # lidrun TAG ARM NAME: loadid.py run -- each greedy prompt under mb c4 + one fresh ~32k prefill beside its decode
  local tag=$1 t0=$(date +%s) j
  arm $2 || return 1; pcreset $tag-$3
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$3.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-$3.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $3: VIRHE no LOADID line ($(tail -3 $O/$tag-$3.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $3 (each prompt under mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s, temperature 0, arm $(NAMEOF $2)): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | prefills %s of %s cold, TTFT %s s mean | in flight %s %% (median) of a probe | mixed steps a probe min %s / median %s; probes without one %s | c4 %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"],
    d["mixed_median"], d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-700), $(( $(date +%s) - t0 )) s | $(pskstat)"
}
c4rr() {   # c4rr TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at conc 4
  local tag=$1 t0=$(date +%s)
  arm $2 || return 1; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 4 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set at conc 4, temperature 0, $MT tok, arm $(NAMEOF $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1 $2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
ident() {   # ident TAG: identity at T 0 -- alone A / S / T / X, under load S and X, run-to-run S, alone again S, the copy under S
  local tag=$1 a
  for a in $(echo $ARMS | grep -o .); do stopfile || alone $tag $a a1-$a; done
  stopfile || lidrun $tag S lid-S
  case $ARMS in *X*) stopfile || lidrun $tag X lid-X ;; esac
  stopfile || c4rr $tag S c4a-S
  stopfile || c4rr $tag S c4b-S
  stopfile || alone $tag S a2-S
  stopfile || copyx $tag S
  cmpx $tag a1-A-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-A.jsonl
  cmpx $tag a1-S-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-S.jsonl
  for a in $(echo $ARMS | grep -o . | grep -v A); do cmpx $tag a1-$a-vs-a1-A $O/$tag-a1-A.jsonl $O/$tag-a1-$a.jsonl; done
  cmpx $tag lid-S-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-lid-S.jsonl
  case $ARMS in *X*) cmpx $tag lid-X-vs-a1-X $O/$tag-a1-X.jsonl $O/$tag-lid-X.jsonl ;; esac
  cmpx $tag c4b-vs-c4a-S $O/$tag-c4a-S.jsonl $O/$tag-c4b-S.jsonl
  cmpx $tag c4a-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-c4a-S.jsonl
  cmpx $tag a2-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-a2-S.jsonl
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 summary: per T / phase / arm mean +- t95, S5 vs A, S5T vs S5, S5X vs S5 (pooled + adjacent pairs), targets, cross-start stack4 (int7) A / S, c8 newcomer share per arm, prefill / loaded / chain tables, guards, identity"; return 0; }
  python3 - "$O" "$RS4" "$PORDER" "$LORDER" "$RS5" > $O/summary.txt 2>&1 <<'PY'
import glob, json, math, os, re, statistics, sys
o, rs4, porder, lorder, rs5 = sys.argv[1:6]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def g(*path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f
def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(a, b): return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"
def mean_arm(rs, arm, get): return ci([get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm])[0]
def pairs(rs, x, y, get):
    pr = []
    for p, q in zip(rs, rs[1:]):
        if {p.get("arm"), q.get("arm")} == {x, y} and p.get("ok") and q.get("ok"):
            px, py = (p, q) if p["arm"] == x else (q, p)
            gx, gy = get(px["res"]), get(py["res"])
            if gx and gy is not None: pr.append("%+.1f" % (100 * (gy / gx - 1)))
    return "/".join(pr) or "-"
NAME = {"A": "A", "S": "S5", "T": "S5T", "X": "S5X"}
s4 = {k: rows(os.path.join(rs4, "s4-n5-%s-blocks.jsonl" % k)) for k in ("t0", "t1")}
s5pre = rows(os.path.join(rs5, "s5-n5-t0-blocks.jsonl"))   # base: stack5 12.11. (pre-move), its S5T blocks
KINDS = {
 "t0": ("T 0", "ASTX", [("S", "A"), ("T", "S"), ("X", "S")],
        [("1u essay", g("c1", "essay", "decode_tps"), 90.0, 1), ("1u code", g("c1", "code", "decode_tps"), 90.0, 1), ("1u edit", g("edit", "decode_tps"), 90.0, 1),
         ("mb c1", g("mbc1", "agg_dec_tps"), 90.0, 1), ("c4", g("c4", "agg_dec_tps"), 160.0, 1), ("c4 p50", g("c4", "per_stream_dec_p50"), None, 1),
         ("c4 tok/step", g("c4", "tok_per_decode_step"), None, 2), ("c4 d0 share", g("statsph", "c4", "d0_share"), None, 3),
         ("c8", g("c8", "agg_dec_tps"), 200.0, 1), ("c8 p50", g("c8", "per_stream_dec_p50"), None, 1), ("c8 tok/step", g("c8", "tok_per_decode_step"), None, 2),
         ("c8 decode_share", g("c8", "decode_share"), None, 3), ("c8 d0 share", g("statsph", "c8", "d0_share"), None, 3), ("pf prefill tok/s", g("pf", "prefill_tps"), None, 1),
         ("pf decode kept", g("pf", "decode_kept"), None, 3), ("pf ms/step during", g("pf", "ms_step_during"), None, 1)]),
 "t1": ("T 1 (sdprobe, seed 1234, thinking on)", "ASTX", [("S", "A"), ("T", "S"), ("X", "S")],
        [("1u essay", g("t1", "essay", "decode_tps"), 90.0, 1), ("1u code", g("t1", "code", "decode_tps"), 90.0, 1), ("1u edit", g("t1", "edit", "decode_tps"), 90.0, 1),
         ("1u essay tok/step", g("t1", "essay", "tokens_per_step"), None, 2), ("1u code tok/step", g("t1", "code", "tokens_per_step"), None, 2),
         ("1u edit tok/step", g("t1", "edit", "tokens_per_step"), None, 2)])}
XREF = {"t0": [("stack4 A", s4["t0"], "A"), ("stack4 S", s4["t0"], "S")], "t1": [("stack4 A", s4["t1"], "A"), ("stack4 S85", s4["t1"], "8")]}
def tgt(v, t):
    if t is None or v is None: return ""
    return "; >= %g: %s" % (t, "MET" if v >= t else "%.1f short" % (t - v))
out, best, gfail = [], {}, []
for k, (lab, arms, prs, M) in KINDS.items():
    rs = rows(os.path.join(o, "s5-n5-%s-blocks.jsonl" % k))
    if not rs: continue
    gfail += ["%s b%d %s: %s" % (k, r["block"], NAME.get(r["arm"]), ((r.get("res") or {}).get("ctr") or {}).get("guard", "?")[:160]) for r in rs if not (((r.get("res") or {}).get("ctr") or {}).get("guard", "") == "OK")]
    parts = []
    for nm, get, t, nd in M:
        mA = mean_arm(rs, "A", get)
        cells = []
        for a in arms:
            m, h, n = ci([get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == a])
            if m is None: continue
            c = "%s %s%s" % (NAME[a], f1(m, nd), "" if h is None else " +- " + f1(h, nd))
            if a != "A" and mA: c += " (%s vs A)" % pc(m, mA)
            c += tgt(m, t)
            cells.append(c)
            if t and a != "A": best.setdefault("%s %s" % (nm, k.upper()), []).append((NAME[a], m, t))
        for x, y in prs:
            mx, my = mean_arm(rs, y, get), mean_arm(rs, x, get)
            if mx is not None and my is not None:
                cells.append("%s vs %s %s (pairs %s)" % (NAME[x], NAME[y], pc(my, mx), pairs(rs, y, x, get)))
        if t:
            xr = []
            for xl, xrs, xa in XREF[k]:
                v = [get(r["res"]) for r in xrs if r.get("ok") and r.get("arm") == xa and get(r["res"]) is not None]
                if v: xr.append("%s %s" % (xl, f1(statistics.mean(v), nd)))
            if xr: cells.append("[cross-start, int7 image: %s]" % ", ".join(xr))
        if k == "t0":
            v5 = [get(r["res"]) for r in s5pre if r.get("ok") and r.get("arm") == "T" and get(r["res"]) is not None]
            mT = mean_arm(rs, "T", get)
            if v5: cells.append("[PRE-MOVE stack5 S5T %s (%d blocks); post vs pre %s]" % (f1(statistics.mean(v5), nd), len(v5), pc(mT, statistics.mean(v5))))
        if cells: parts.append("%s: %s" % (nm, " | ".join(cells)))
    flagged = sum(1 for r in rs if not r.get("ok"))
    l = "%s (%d blocks, %d flagged): %s" % (lab, len(rs), flagged, " || ".join(parts))
    out.append(l); print("ARMS " + l)
# c8 newcomer share per arm (ncsteps.py c8 per T 0 block; flagged blocks left out)
okb = {"%s-b%d%s" % (r["label"], r["block"], r["arm"]): r["arm"] for r in rows(os.path.join(o, "s5-n5-t0-blocks.jsonl")) if r.get("ok")}
nc = {}
for p in glob.glob(os.path.join(o, "nc8-*.json")):
    b = os.path.basename(p)[4:-5]
    if b not in okb: continue
    try: d = json.load(open(p))
    except Exception: continue
    if any((x.get("cached") or 0) != 0 for x in d["newcomers"].get("rows") or []): continue
    nc.setdefault(okb[b], []).append(d)
if nc:
    cells = []
    for a in "ASTX":
        if a not in nc: continue
        ds = nc[a]
        sh = ci([(d.get("stats") or {}).get("decode_share") for d in ds])[0]
        w = ci([d["newcomers"]["wall_ms"].get("p50") for d in ds])[0]
        wt = ci([d["newcomers"]["wait_ms"].get("p50") for d in ds])[0]
        mx = sum(d["newcomers"]["mixed"] for d in ds); pf = sum(d["newcomers"]["prefill"] for d in ds); n = sum(d["newcomers"]["n"] for d in ds)
        cells.append("%s decode_share %s, newcomer step wall p50 %s ms, wait p50 %s ms, %d newcomers (%d MIXED / %d prefill) over %d block(s)" % (NAME[a], f1(sh, 3), f1(w, 0), f1(wt, 0), n, mx, pf, len(ds)))
    pre = []
    for p in sorted(glob.glob(os.path.join(rs5, "nc8-*T.json"))):
        try: d = json.load(open(p)); pre.append("%s %s" % (os.path.basename(p)[4:-5].split("-")[-1], d["newcomers"]["wall_ms"].get("p50")))
        except Exception: pass
    print("C8NEW " + " | ".join(cells) + " [cross-start, int7: ncprof S newcomer step p50 581.5 ms, decode_share 0.793]" + (" [PRE-MOVE stack5 S5T newcomer step wall p50 ms: %s]" % ", ".join(pre) if pre else ""))
# prefill per arm (guard FAIL runs left out)
dp = [r for r in rows(os.path.join(o, "deep.jsonl")) if r.get("reset") and not r.get("name", "").startswith("guard")]
pp = []
for dep in sorted({r["depth"] for r in dp}, reverse=True):
    s = [r for r in dp if r["depth"] == dep and str(r.get("guard", "OK")).startswith("OK")]
    if not s: continue
    tt = {a: ci([r.get("ttft") for r in s if r["arm"] == a])[0] for a in "ASTX"}
    cells = ["%s %s s%s" % (NAME[a], f1(tt[a], 2), "" if a == "A" else " (%s vs A)" % pc(tt[a], tt["A"])) for a in "ASTX" if tt[a] is not None]
    sg = lambda arm, k: ci([r.get(k) for r in s if r["arm"] == arm])[0]
    l = "cold %d (%s tok) TTFT: %s | staged GB %s | snapshots %s%s" % (dep, s[0].get("prompt_tokens"), ", ".join(cells),
        ", ".join("%s %s" % (NAME[a], f1(sg(a, "staged_gb"), 1)) for a in "ASTX" if tt[a] is not None),
        ", ".join("%s %s" % (NAME[a], f1(sg(a, "ckpts"), 1)) for a in "ASTX" if tt[a] is not None),
        "" if all(r.get("cached") == 0 for r in s) else " -- VIRHE a run was not cold")
    if dep == 135000:
        bs = min([(tt[a], NAME[a]) for a in "STX" if tt[a] is not None] or [(None, "-")], key=lambda x: x[0] or 1e9)
        pass
    d5 = [r.get("ttft") for r in rows(os.path.join(rs5, "deep.jsonl")) if r.get("reset") and r.get("arm") == "T" and r.get("depth") == dep and not r.get("name", "").startswith("guard") and r.get("ttft")]
    if d5 and tt.get("T"): l += " | PRE-MOVE stack5 S5T %s s (post vs pre %s)" % (f1(statistics.mean(d5), 2), pc(tt["T"], statistics.mean(d5)))
    if dep == 135000:
        if bs[0]: l += "; best %s vs the 35 s target: %s; cross-start (no evict) stack3 n0 50.03 s, n5 51.33 s" % (bs[1], "MET" if bs[0] <= 35 else "%.1f s over" % (bs[0] - 35)); best.setdefault("135k TTFT s", []).append((bs[1], bs[0], 35.0))
    pp.append(l)
ld = [r for r in rows(os.path.join(o, "load.jsonl")) if str(r.get("guard", "OK")).startswith("OK")]
if ld:
    cells = []
    for nm, kk, nd in (("mixed-step ms", "nondecode_ms", 0), ("c4 agg tok/s in the prefill window", "c4_pf_agg_tps", 1),
                       ("c4 per-stream p50", "c4_pf_stream_tps_p50", 1), ("TPOT max ms", "tpot_pf_max", 0), ("probe TTFT s", "ttft", 2)):
        v = {a: ci([r.get(kk) for r in ld if r["arm"] == a])[0] for a in "ASTX"}
        cells.append("%s %s" % (nm, ", ".join("%s %s%s" % (NAME[a], f1(v[a], nd), "" if a == "A" else " (%s)" % pc(v[a], v["A"])) for a in "ASTX" if v[a] is not None)))
    pp.append("c4 + one cold ~64k prompt (%s): %s" % (lorder, " | ".join(cells)))
ch = rows(os.path.join(o, "chain.jsonl"))
if ch:
    cells = []
    for a in "ASTX":
        t = sorted([r for r in ch if r["arm"] == a], key=lambda r: r["turn"])
        if t: cells.append("%s %s" % (NAME[a], " / ".join("t%d %s tok cached %s TTFT %s s" % (r["turn"], r.get("prompt_tokens"), r.get("cached"), r.get("ttft")) for r in t)))
    pp.append("turn chains: " + " | ".join(cells))
for l in pp: print("PREFILL " + l)
bl = []
for nm, vs in best.items():
    low = nm.startswith("135k")
    a, v, t = (min if low else max)(vs, key=lambda x: x[1])
    bl.append("%s best %s %s (%s)" % (nm, a, f1(v, 2 if low else 1), ("MET" if v <= t else "%.1f s over" % (v - t)) if low else ("MET" if v >= t else "%.1f short" % (t - v))))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
# A runs sched_bi_chunk 0: its loaded 64k answer vs alone is the KNOWN class (ii) difference (2048-row mixed chunks vs alone's 4096),
# labelled; a loaded S / T / X answer vs a loaded A is information (A not batch-invariant). Every other compare is strict.
known = [l for l in ident if re.match(r"ld-\d+A-vs-aloneA ", l)]
info = [l for l in ident if re.match(r"ld-\d+[STX]-vs-ldA ", l)]
strict = [l for l in ident if l not in known and l not in info]
bad = [l for l in strict if not ("IDENTICAL" in l and "differ" not in l)]
lab = lambda ls: "; ".join(l.split(" ")[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "differ" not in l else "differs") for l in ls)
try: guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
except OSError: guards = []
print("SUMMARY targets: %s || guards: %s; flagged blocks: %s || identity (strict): %s || known class (ii) at sched_bi_chunk 0 (A): %s || information (vs a loaded A): %s || prefill: %s || %s" % (
      " | ".join(bl) or "-", " | ".join(guards) or "none", "; ".join(gfail)[:1500] or "none",
      ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:900] if strict else "none"),
      lab(known) or "-", lab(info) or "-", " | ".join(pp) or "-", " || ".join(out)))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
# ---------------------------------------------------------------- base: the baseline item's own functions
bud() {   # bud NAME EST_MIN RESERVE_MIN: 0 = run; 1 = skip (logged) when elapsed + EST + RESERVE would pass XS5_BUDGET_MIN
  local e; e=$(elapsed)
  [ "$DRY" = 1 ] && { echo "+ budget $1: run only if elapsed + ~$2 min + $3 min reserve <= $BUDGET min, else SKIPPED (logged)"; return 0; }
  [ $((e + $2 + $3)) -le $BUDGET ] && return 0
  xlog "$1 SKIPPED (budget): $e min elapsed + ~$2 min + $3 min reserve > $BUDGET min"; return 1
}
basechk() {   # basechk TAG: the rank -> card map from the start log (vs stack5's pre-move map); KVR off; SP off
  local f=$O/start-$1.log m m5 w kv sp
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 $1 rank -> card map (start log 'card <PCI>  rank N' + link chain; vs stack5 $RS5/start-s5-n5.log) | KVR off (no 'lever KVR declared' line) | SP off (fn4 stage_full_min / stage_full_pick 0 if present)"; return 0; }
  [ -s $f ] || docker logs $CN > $f 2>&1
  cmap() { python3 -c 'import re,sys
try: L = open(sys.argv[1], errors="replace").read().split("\n")
except OSError: print("n/a"); sys.exit(0)
out = []
for i, l in enumerate(L):
    m = re.match(r"^I\s+card (\S+)\s+rank (\d+) \(device (\d+)\)", l)
    if not m: continue
    lk = L[i + 1] if i + 1 < len(L) else ""
    lm = re.match(r"^I\s+link\s+(\S+) (\S+ GT/s x\d+)", lk)
    out.append((int(m.group(2)), "rank %s = %s (device %s; root %s at %s)" % (m.group(2), m.group(1).replace("0000:", ""), m.group(3), lm.group(1) if lm else "?", lm.group(2) if lm else "?")))
print(", ".join(x for _, x in sorted(out)) or "no card lines")' "$1" 2>&1 | cut -c1-900; }
  m=$(cmap $f); m5=$(cmap $RS5/start-s5-n5.log)
  w=$(grep -E '^I +WARN link-below-max' $f 2>/dev/null | sed -E 's/^I +WARN link-below-max: card 0000:([0-9a-f:.]+) \((rank [0-9])\).* runs ([^ ]+ GT\/s x[0-9]+) of ([^ ]+ GT\/s x[0-9]+).*/\1 \2 \3 of \4/' | tr '\n' ';' | cut -c1-500)
  xlog "$1 RANK -> CARD MAP (post-riser): $m | link-below-max warnings: ${w:-none} | pre-move (stack5 start 12.11.): $m5"
  if grep -q 'lever KVR declared' $f 2>/dev/null; then kv="VIRHE a 'lever KVR declared' line (KVR must be OFF)"; else kv="OFF (no 'lever KVR declared' line, RADIANCE_DS41_KV_REPLICATED unset)"; fi
  sp=$(fn4get | python3 -c 'import json,sys
w = json.load(sys.stdin).get("switches", {}); k = [x for x in ("stage_full_min", "stage_full_pick") if x in w]
print("not in this image" if not k else ("OFF (" + ", ".join("%s %s" % (x, w[x]) for x in k) + ")" if all(w[x] == 0 for x in k) else "VIRHE not 0: " + ", ".join("%s %s" % (x, w[x]) for x in k)))' 2>&1 | cut -c1-200)
  xlog "$1 int9 levers at start: KVR $kv | SP $sp"
}
identb() {   # identb TAG: ONLY the greedy set c1 ALONE under S5T, vs stack5's S5T answers (loadid.py cmp)
  local tag=$1
  alone $tag T a1-T
  cmpx $tag a1-T-vs-stack5-a1-T $RS5/s5-n5-a1-T.jsonl $O/$tag-a1-T.jsonl
}
# ---------------------------------------------------------------- d0ab: this item's own functions
c4split() {   # c4split: the engine log right after the c4 blocks; c4split.py per arm (plain vs drafting 4-row decode steps)
  local a
  capture $O/c4-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per arm $ARMS: python3 -I -B $C4S $O/c4-engine.log.gz $O/s5-n5-t0-blocks.jsonl ARM -> RESULT xd0 c4 step split lines"; return 0; }
  for a in $(echo $ARMS | grep -o .); do
    timeout 300 python3 -I -B $C4S $O/c4-engine.log.gz $O/s5-n5-t0-blocks.jsonl $a > $O/c4split-$a.txt 2>&1
    while IFS= read -r l; do xlog "c4 step split arm $(NAMEOF $a): ${l:0:900}"; done < <(grep -E '^block|Traceback|Error' $O/c4split-$a.txt)
    grep -qE '^block' $O/c4split-$a.txt || xlog "c4 step split arm $(NAMEOF $a): no block line ($(tail -2 $O/c4split-$a.txt | tr '\n' ' ' | cut -c1-300))"
  done
}
bxblock() {   # bxblock TAG: ONE extra block under B: mb c1 + mb c8 (c8 through nc/ncmb.py, old loop), the baseline's harness and args
  local tag=$1
  run "mkdir -p $O/c8 $O/snap; timeout 3600 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --seed $SEED --t1-prompts $T1P --label $tag-bx --arm 'B=$(armj B)' --guard 'B=$(needs blk B)' --hb $HBR --hb-ranks $HBRANKS --snap-dir $O/snap --settle $SETTLE --order B --phases mbc1,c8 --dur $DUR --warm $WARMS --mbtemp 0 --mbt $MBT --c8-ncmb $NCMB --c8-dir $O/c8 --fixed '$FIX' --out $O/$tag-bx-blocks.jsonl > $O/$tag-bx-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ B extra c8 drafting check (occ rule: occ_cycles > 0, acceptance 0, ~8 tok/step; drafting ON = a NOTE)"; return 0; }
  while IFS= read -r l; do case "$l" in "GUARD "*"FAIL"*) xlog "$tag bx VIRHE ${l:0:1500}" ;; *) xlog "$tag bx ${l:0:1700}" ;; esac; done < <(grep -E '^(BLOCK|GUARD|OCC|STATS|GGB \{"error)|Traceback' $O/$tag-bx-blocks.txt)
  xlog "B extra c8 drafting check: $(python3 -c 'import json,sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]; res = r["res"]; c = res.get("c8") or {}; d = (res.get("draftph") or {}).get("c8") or {}
occ, acc, tps = d.get("occ_cycles"), c.get("acceptance"), c.get("tok_per_decode_step")
off = (occ or 0) > 0 and not acc and tps is not None and abs(tps - 8) <= 0.5
print("%s (occ_cycles %s, zero_cycles %s, probe_cycles %s, acceptance %s, tok/step %s, %s tok/s; block ok=%s)" % (
    "drafting stays OFF at c8 under the occ rule" if off else "NOTE: drafting looks ON at c8 under B (not an error; read the c8 numbers with it)",
    occ, d.get("zero_cycles"), d.get("probe_cycles"), acc, tps, c.get("agg_dec_tps"), r.get("ok")))' $O/$tag-bx-blocks.jsonl 2>&1 | tail -1 | cut -c1-600)"
}
identd() {   # identd TAG: the greedy set at conc 4 (T 0) under A, B, C; B vs A and C vs A strict (loadid.py cmp): any difference is VIRHE
  local tag=$1 a l
  for a in A B C; do stopfile || c4rr $tag $a c4-$a; done
  for a in B C; do
    cmpx $tag c4-$a-vs-c4-A $O/$tag-c4-A.jsonl $O/$tag-c4-$a.jsonl
    [ "$DRY" = 1 ] && continue
    l=$(grep "^$tag-c4-$a-vs-c4-A " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity c4 $a vs A: VIRHE $l" ;; esac ;;
      *) xlog "identity c4 $a vs A: VIRHE ${l:-not compared} (depth changes only the verify count: the text must match)" ;; esac
  done
}
summaryd() {   # summaryd: per arm mean +- half-range, B vs A %, C vs A %, base1113 S5T + pre-move stack5 S5T c4 150.6; B extra; identity
  [ "$DRY" = 1 ] && { echo "+ RESULT xd0 summary: per arm A / B / C c4 agg tok/s, per-stream p50, ms / tok a decode step, acceptance, decode_share, zero_cycles / zero_moves / probe_cycles, d0_share, lent_cycles (mean +- half-range, per block), B vs A %, C vs A %, base1113 S5T, pre-move 150.6; B extra mb c1 / c8 vs base1113; identity"; return 0; }
  python3 - "$O" "$RB" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, rb = sys.argv[1:3]
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def g(r, *k):
    x = r
    for kk in k:
        if not isinstance(x, dict): return None
        x = x.get(kk)
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
def ms(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    return statistics.mean(v), ((max(v) - min(v)) / 2 if len(v) > 1 else None), len(v)
def f(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(a, b): return "%+.1f %%" % ((a / b - 1) * 100) if a is not None and b else "-"
NAME = {"A": "A (d0 1, lend 1)", "B": "B (d0 0)", "C": "C (d0 1, lend 0)"}
R = rows(os.path.join(o, "s5-n5-t0-blocks.jsonl"))
base = [r for r in rows(os.path.join(rb, "s5-n5-t0-blocks.jsonl")) if r.get("ok") and r.get("arm") == "T"]
M = [("c4 agg tok/s", ("c4", "agg_dec_tps"), 1), ("c4 per-stream p50", ("c4", "per_stream_dec_p50"), 1), ("ms/decode step", ("c4", "ms_decode_step"), 1),
     ("tok/decode step", ("c4", "tok_per_decode_step"), 2), ("acceptance", ("c4", "acceptance"), 3), ("decode_share", ("c4", "decode_share"), 3),
     ("zero_cycles", ("draftph", "c4", "zero_cycles"), 0), ("zero_moves", ("draftph", "c4", "zero_moves"), 0), ("probe_cycles", ("draftph", "c4", "probe_cycles"), 0),
     ("d0_share", ("statsph", "c4", "d0_share"), 3), ("lent_cycles", ("statsph", "c4", "lent_cycles"), 0)]
flag = ["%d%s" % (r["block"], r["arm"]) for r in R if not r.get("ok")]
for nm, k, nd in M:
    cells, mean = [], {}
    for a in "ABC":
        v = [g(r["res"], *k) for r in R if r.get("ok") and r.get("arm") == a]
        m, h, n = ms(v); mean[a] = m
        if m is None: continue
        cells.append("%s %s%s [blocks %s]" % (NAME[a], f(m, nd), "" if h is None else " +- " + f(h, nd), "/".join(f(x, nd) for x in v if x is not None)))
    cells += ["B vs A %s" % pc(mean.get("B"), mean.get("A")), "C vs A %s" % pc(mean.get("C"), mean.get("A"))]
    bm = ms([g(r["res"], *k) for r in base])[0]
    if bm is not None: cells.append("baseline base1113 S5T %s (dur 60 warm 15)" % f(bm, nd))
    if k == ("c4", "agg_dec_tps"):
        cells.append("vs pre-move stack5 S5T 150.6: A %s, B %s, C %s" % (pc(mean.get("A"), 150.6), pc(mean.get("B"), 150.6), pc(mean.get("C"), 150.6)))
    print("D0AB %s: %s" % (nm, " | ".join(cells)))
for r in rows(os.path.join(o, "s5-n5-bx-blocks.jsonl")):
    cells = []
    for ph in ("mbc1", "c8"):
        bm = ms([g(x["res"], ph, "agg_dec_tps") for x in base])[0]
        cells.append("%s %s tok/s, %s tok/step, acc %s, %s ms/step, decode_share %s, occ_cycles %s, d0_share %s (baseline base1113 S5T %s tok/s at dur 60 warm 15: %s)" % (
            ph, g(r["res"], ph, "agg_dec_tps"), g(r["res"], ph, "tok_per_decode_step"), g(r["res"], ph, "acceptance"), g(r["res"], ph, "ms_decode_step"),
            g(r["res"], ph, "decode_share"), g(r["res"], "draftph", ph, "occ_cycles"), g(r["res"], "statsph", ph, "d0_share"), f(bm), pc(g(r["res"], ph, "agg_dec_tps"), bm)))
    print("D0AB B extra (ok=%s): %s" % (r.get("ok"), " | ".join(cells)))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
print("SUMMARY %d c4 blocks (%d flagged%s) | identity: %s" % (len(R), len(flag), (": " + ", ".join(flag)) if flag else "", " ; ".join(ident) or "none"))
PY
  while IFS= read -r l; do xlog "${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=s5-n5 a
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms $(for a in $(echo $ARMS | grep -o .); do echo -n "$(NAMEOF $a) $(armj $a) "; done)| env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    a=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; a=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; a=0; }
    [ $a = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  basechk $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  for a in $(echo $ARMS | grep -oE '[TX]'); do smoke $tag $a || { xlog "$tag: VIRHE the smoke under $(NAMEOF $a) failed -> stopped"; down; return 1; }; done
  case $ARMS in *[TX]*) ;; *) smoke $tag A || { xlog "$tag: VIRHE the smoke under A failed -> stopped"; down; return 1; } ;; esac
  stopfile || cguard $tag
  if [ $SPEEDOK = 1 ]; then
    stopfile || blocks $tag t0
    stopfile || c4split
    stopfile || { bud "B extra (mb c1 + mb c8 under B)" 2 2 && bxblock $tag; }
    xlog "T 1, prefill depths, loaded runs: none (d0ab item)"
  else
    xlog "$tag: VIRHE a guard failed (above): no T 0 / T 1 blocks, no prefill or loaded readings -- chains and identity only"
  fi
  xlog "turn chains: SKIPPED (d0ab item)"
  stopfile || { bud "identity (greedy set c4 under A / B / C)" 2 0 && identd $tag; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: lever counters at the end: $(pskstat) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | arms $ARMS | T 0 order ${ORDER0:-none} phases $PH0 (mb T 0, dur $DUR warm $WARMS; c8 via ncmb.py) | T 1 order ${ORDER1:-none} (sdprobe $T1P seed $SEED thinking on; trimmed first over budget) | prefill $PORDER at [$DEPTHS], loaded c4 + $LDEPTH ${LORDER:-none} | turn chains ~$CDEPTH + 2 hits | identity T 0 strict | counter guard everywhere | budget $BUDGET min | no evict | targets 1u >= 90, c4 >= 160, c8 >= 200, 135k <= 35 s"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx,
B|ds41_psk_cr, ds41_engdd, sched_noa, stage_short, stage_pick, place_hb, stage_full_min, stage_full_pick, epoch_bump)
B|ds41_psk_k
B|ds41_psk_l20
B|dspark_pfctx
B|ds41_psk_cr
B|ds41_engdd
B|sched_noa
B|stage_short
B|stage_pick
B|place_hb
B|sched_noa takes 0 (off)
B|stage_short takes 0 (off) or 1 .. 65536
B|place_hb takes 0 (off) or 1 .. 4096
B|ds41_psk_k takes 1 .. 4096
B|pf_steps_4plus
B|rows_deduped
B|guard48
B|rad_engdd_stats
B|rad_psk_plan2
B|rad_dspark_pfctx_stats
B|l20_cut_passes
B|rows_skipped_l30_39
B|cr_hits
B|cr_not_needed
B|cr_cheaper
B|cr_end_snapshots
B|tail_only_latched
B|hb_promotions
B|short_steps
B|lever SA / HB (ds41-sahb) set up, off: stage_short %s; place_hb %s
B|RADIANCE_PLACE_HB_RANKS
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|RADIANCE_SCHED_BI_CHUNK
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|RADIANCE_DS41_PSK_L20
A|RADIANCE_DS41_DSPARK_PFCTX
A|RADIANCE_DS41_PSK_CR
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|rad_psk_plan2
A|rad_dspark_pfctx_stats
A|RADIANCE_DS41_ENGDD
A|lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now
A|rad_engdd_stats
A|engram_kv_u
K|gather_rows_w16_kernel
B|place_lf
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
K|dspark_walk_sample_step
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ takes 0 (off) .. 64
B|draft_occ_k takes 1 .. 4096
B|occ_cycles
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"; test -s /opt/radiance/share/radiance/int8-PROVENANCE.txt || echo "MISS int8-PROVENANCE.txt"; test -s /opt/radiance/share/radiance/int9-PROVENANCE.txt || echo "MISS int9-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the int9 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int9 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the fn4 registry with ds41_psk_l20 / dspark_pfctx / ds41_psk_cr / ds41_engdd / sched_noa / stage_short / stage_pick / place_hb and the unknown-switch tail, their range messages, the PSK / L20 / CR / PSK-DS / E1 / NOA / SA-HB counters and declare lines, the step log, BI / SD / DLW / Z0 / LF / PSK, the /metrics counters) + bin/rad-sidecar + int7-PROVENANCE + int8-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $GGB $C4S $HD/s5ctr.py $CTR $CHN $MBT $PRB $EP $PL $PLC $PCR2 $NCMB $NCS $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $GGB $C4S $HD/s5ctr.py $CTR $CHN $MBT $PRB $EP $PL $PLC $PCR2 $NCMB $NCS $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
PST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$PST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${PST:0:300})" ;; esac
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
NST_=$(timeout 120 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$NST_" in *OK*) ;; *) HMISS="$HMISS selftest:$NCS(${NST_:0:300})" ;; esac
CTT=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT:0:300})" ;; esac
CHT=$(cd $D && timeout 120 python3 -I -B $CHN selftest --pcr2 $PCR2 --pskld $PLC --tests $TESTS 2>&1 | tail -1); case "$CHT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CHN(${CHT:0:300})" ;; esac
TXT=$(cd $D && timeout 120 python3 -I -B $PCR2 texts --pskld $PLC --tests $TESTS --doffs $(i=0; for a in $(echo $ARMS | grep -o .); do i=$((i + 1)); echo -n "$((370 + i * 30)),"; done | sed 's/,$//') 2>&1 | tail -1); case "$TXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS texts:$PCR2(${TXT:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-pskld:(${DQP:0:200})" ;; esac
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:s5blocks.py--help"
(cd $D && python3 -I -B $PRB --help > /dev/null 2>&1) || HMISS="$HMISS run:sdprobe.py--help"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj needs stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta ctrg gate1 deepx coldabba copyx loadx loadabba swnote smoke cguard blocks ncparse t1budget chainx chains alone lidrun c4rr cmpx ident summary dstart bud basechk identb c4split bxblock identd summaryd"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in $(echo $ARMS | grep -o .); do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; for k in blk t1 cold load chain; do [ -n "$(needs $k $a)" ] || HMISS="$HMISS needs:$k:$a"; done; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, pskld.py $PST, ncmb.py $NMT, ncsteps.py $NST_, s5ctr.py $CTT, s5chain.py $CHT, pcr2.py texts $TXT, deepq.py = pskld.py's deep request ($DQP), s5blocks.py / sdprobe.py --help, all $(echo $FNS | wc -w) functions resolve, tools on PATH, every arm's pin and guard spec, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-t0-blocks.jsonl $RS4/s4-n5-t1-blocks.jsonl $RS5/s5-n5-a1-T.jsonl $RS5/t0-engine.log.gz $RS5/s5-n5-t0-blocks.jsonl $RS5/deep.jsonl $RS5/start-s5-n5.log $RB/s5-n5-t0-blocks.jsonl; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: d0blocks.py $(md5sum < $GGB | cut -c1-8), c4split.py $(md5sum < $C4S | cut -c1-8), s5ctr.py $(md5sum < $CTR | cut -c1-8), s5chain.py $(md5sum < $CHN | cut -c1-8), mbt.py $(md5sum < $MBT | cut -c1-8), loadid.py $(md5sum < $LD | cut -c1-8), pskld.py $(md5sum < $PL | cut -c1-8) / pskcr $(md5sum < $PLC | cut -c1-8), pcr2.py $(md5sum < $PCR2 | cut -c1-8), ncmb.py $(md5sum < $NCMB | cut -c1-8), ncsteps.py $(md5sum < $NCS | cut -c1-8), sdprobe.py $(md5sum < $PRB | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-stats0.json $RS4/s4-n5-t0-blocks.jsonl $RS4/s4-n5-t1-blocks.jsonl $RS5/s5-n5-a1-T.jsonl $RS5/t0-engine.log.gz $RS5/s5-n5-t0-blocks.jsonl $RS5/deep.jsonl $RS5/start-s5-n5.log $RB/s5-n5-t0-blocks.jsonl $C4S $HD/s5ctr.py; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in $(echo $ARMS | grep -o .); do echo "+ arm $a ($(NAMEOF $a)): $(armj $a) | guard blk [$(needs blk $a)] t1 [$(needs t1 $a)] cold [$(needs cold $a)] chain [$(needs chain $a)]"; done
  echo "+ fixed pin: $FIX"
  echo "+ orders: T 0 $ORDER0 | T 1 $ORDER1 | prefill $PORDER | loaded $LORDER"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summaryd
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
