#!/bin/bash
# dsm-tree6.sh (11.11.2026, queue-ar/dsmain, ds-main-port T1, the coordinator's request 06:4x UTC) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-*-tree6.sh. THE T1 SPEED MEASUREMENT AT THE SHAPE THAT PASSED (tree5or: or15 2 x 5 and or14 3 x 3 forced, 2 x 2
# IDENTICAL, every correct-path row right): B = P2 (2 branches x depth 5, priced by D7, at most 15 nodes = 16 verify rows at one
# user) vs A = dspark_tree 0, in-process ABBA, on the stack2 NST 5 levers this image has: hx-stack's pins + the hybrid (1, match
# 8) + lever DL (declared off at start, draft_lend 1 + draft_probe_backoff 3 in the pin) + the VRAM levers at start (router bf16,
# Engram rows in RAM, 16 shadow slots, expert prefetch off, arena 48 decode rows) + turnover 0. NOT on this image (ds41-int base,
# so not in either arm): int3's W1 walk, drafter_mx, the small n5 set (argather 48, idx_split, K1 a / e, ctx skip, conf), GPF
# -- T1 on int3 needs a merge (19 conflicting files: the drafter's walk vs the tree, fn4, attention kernels).
# ORDER: the oracle (or15, the P2 shape forced: 15 nodes) FIRST as the gate; then the main start: AB = ABBA A / B, mb 75 s:
# 1u essay / code / edit at temperature 0 (stepprobe 512 / editprobe 1500) and at the server default (da/daprobe.py --mode sampled
# --seed 1234: no temperature sent = the engine's temperature 1), mb c4 / c8 (treeblocks2.py); ID = identity under B at
# temperature 0 (28k copy + greedy set c1 vs bz d0). No screen. Out runs/xdsm/tree6, tag xtr6.
# --- dsm-tree5.sh's header (the shapes, the cause of the 21-row failure):
# dsm-tree5.sh (11.11.2026, queue-ar/dsmain, ds-main-port T1) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-*-tree5*.sh.
# = dsm-tree3.sh (md5 e4029314) with every tree at most 15 nodes (16 verify rows at one user). WHY: the T1 oracle's only wrong
# token came from a 21-row verify, and the row-count sweep (tree4, 11.11. 05:19-05:34, NO tree, tree unset, --spec-ngram K
# --debug-draft-oracle) reproduced it without T1: chain verifies of 21 rows DIFFERENT at token 39 (31 of 4290 accepted, the
# tree's exact signature), 13 / 9 / 6 rows IDENTICAL. The trunk's EXL3 routed-expert GEMMs run the decode band at M <= 16
# tokens and the prefill form above it (radiance-exl3 kernels/exl3r4d/exl3r4d.hip, prefill.c), which its own test holds to a
# tolerance, not to the decode band's bits (tests/gpu/prefill/test_prefill.py: 4e-3 relative L2): a step of more than 16
# rows is not the 1-row decode's arithmetic. tree3's trace: every row on a correct path of 11 tree steps (main chain and
# branches, after branch commits) had the chain's target. So: oracle arms or15 (2 branches x 5 forced: 15 nodes, 16 rows)
# and or14 (3 x 3 forced: 14 nodes, 15 rows); screen arms within 15 nodes (T0, F1, F2 = 2 x 5, F3d3 = 3 x 3, F1d2, P2 = 2 x 5
# priced, P3d3 = 3 x 3 priced); identity under B and F2. Out runs/xdsm/tree5, tag xtr5.
# --- dsm-tree3.sh's header:
# = dsm-tree2.sh (md5 f94299ee) for the 11.11. 02:46 result: the tree oracle ran past the crash but was DIFFERENT (first diff at
# token 39 of 256; 31 drafts accepted of 4480, 0.14 a step, where the chain harness accepts 2.08 a step). Changes: the yaml
# ds41-tree3-tp4.yaml (+ RADIANCE_DEBUG_TREE_STEPS), the image's verify-path trace (ds41-tree 2fc434e) on in every oracle start
# (RADIANCE_DEBUG_TREE_STEPS=400), and after each oracle start treetrace.py's five readings as "RESULT xtr6 <tag> trace" lines
# (harness: the main chain's ids vs the plain run's; staging: ids / positions / spec_parent vs the host's tree; root: row 0's
# target; rows: every node on a correct path, target vs the chain's next id, by depth, main / branch; accept: the rule) and
# FIRSTBAD (the first step whose emission leaves the plain run). A THIRD ORACLE START FIRST, or20t0: the tree declared at
# dspark_tree 0 (the chain of 5 through the tree verify path, no branch), so a failure there is not the branches'. A check
# whose answer is DIFFERENT is not repeated (deterministic). Oracle-first gate as dsm-tree2. Out runs/xdsm/tree3, tag xtr3.
# --- dsm-tree2.sh's header:
# = dsm-tree.sh (10.11., the T1 run of 10.11. 23:31 that produced no data: the engine stopped at the first verify step, E "op
# 'qsa_tail_store' ... names buffer 'tree_slot_pos', which this op never declared", in the main start AND both oracle starts; the
# guard read that as "SLOW 999" and the oracle as "DIFFERENT ()") for an image with that fixed (ds41-tree, the tree ops' buffer
# declarations), plus HARNESS FIXES:
#   - gate: an engine that stopped on an error at the gate's first requests (any "E " line in its log) is VIRHE ENGINE ERROR
#     with the E lines, kept log, no retry (rc 3), not "SLOW-after-retry (first 999)" (999 = fastcheck printed no number);
#   - t0cost: tree 0 at NST 20 (every decode step runs the tree path on the chain of 5 -- the scheduler stages spec_parent for a
#     tree drafter whatever dspark_tree is) vs the NST 5 references, as RESULT lines: the gate's 1-user ms/step vs hx-stack's
#     st-stack NST 5 guard (both before any fn4 pin), and in the summary arm A (= T0) vs hx-stack's arm A per metric (hx-stack
#     A pins drafter_mx 1, which this image does not have: noted). This bounds the lever (the tree must beat its T0 cost);
#   - oracle: a check whose request failed (no ORACLE line) is VIRHE ERROR with oracle.py's error and the engine's E lines, not
#     DIFFERENT; check 2 is skipped when the server is down; the draft counters are /metrics' (draft_tokens_total,
#     draft_accepted_total, engine_steps_decode_total deltas) beside the /stats tree counters below;
#   - DRYRUN=1 also checks that every helper function and helper file the body calls resolves.
#   - ORDER: the tree oracle (OR) first; the main start (S AB ID) only when every oracle check is IDENTICAL with trees that
#     carry branches verified (/stats tree_wide > 0) -- XTR_OR_FIRST=0 keeps the 10.11. order. The oracle checks also read
#     the engine's /stats tree counters (ds41-tree 6b143a9: tree_verifies / _nodes / _wide / _branch_nodes / _branch_paths /
#     _branch_accepted, present only under a tree layout); treeblocks.py picks them up per block on its own.
# Out runs/xdsm/tree2, tag "RESULT xtr6". The 10.11. run's files stay in runs/xdsm/tree.
# --- dsm-tree.sh's header (10.11.):
# LEVER T1, THE DSPARK TREE VERIFY AT POSITION 1 (deepseek41 + core, branch ds41-tree on ds41-int c05de27): a drafting step
# verifies the walk's 5-node main chain plus up to 3 alternative branches at position 1 (the walk's next-best first drafts,
# each walked on, depth up to 5) as one tree; the trunk's compressor, indexer, Engram and window attention are tree-aware and
# every emitted token is the target's own argmax along the accepted path (exact). One image (XTR_IMAGE), EXL3 2.25, TP4,
# MML 524288, the hx-stack start (dsm-stack.sh's NST 5 start: the t3 env + lib-29 best_vars + MNBT 4160 + --live
# --draft-window adaptive; RADIANCE_DS41_GG_GEMV=2, RADIANCE_AQL_ACQ=rule-w, RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL=1 at start;
# the VRAM env RADIANCE_PLACE_TURNOVER=0; in process the stack pin gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill
# 1, draft_adaptive 1, draft_depth0 1, draft_hybrid 1, draft_hybrid_match 8) EXCEPT --num-speculative-tokens 20 (the tree's
# 5 + 3 x 5 node slots) and RADIANCE_DS41_DSPARK_TREE=1 (+ _B / _L / _FORCE: the switches' start values, 0 / 5 / 0).
# In process (POST /admin/fn4): dspark_tree (0..3, alternative branches at position 1; 0 = today's chain of 5),
# dspark_tree_depth (1..5, their depth), dspark_tree_force (1 = every drafting step verifies the whole tree, whatever the
# adaptive window; 0 = D7's cost model -- step_cost by node count -- prices the extra nodes, so they auto-reduce under load).
# PHASES (XTR_PHASES, default "S AB ID OR"), one main start for S / AB / ID, then the oracle starts:
#   S   SCREEN, c1 essay + code only (stepprobe, XTR_SGEN tokens, each alone between /stats + /metrics snapshots), one block an
#       arm, order XTR_SORDER (default T0 F1 F2 F3 F1d2 P3 T0): T0 = dspark_tree 0; F1 / F2 / F3 = 1 / 2 / 3 branches of depth
#       5 forced; F1d2 = 1 branch of depth 2 forced; P3 = 3 branches of depth 5 priced by D7. Per arm: tok/s, tokens a step,
#       ms a decode step, verify rows a step (1 + verified nodes a step), acceptance, the D7 counters. THE VERIFY-ROW COST at
#       c1: the least-squares line ms a step = a + b x rows a step over every screen block (essay and code apart) ->
#       "RESULT xtr6 rowcost". THE PICK: the non-T0 arm with the highest geometric mean of its essay and code tok/s ratios to
#       T0's mean (XTR_PICK=<arm> overrides; no T0 or no other good arm -> P3).
#   AB  in-process ABBA (treeblocks.py, epoch_bump every block), A = T0, B = the pick, mb XTR_DUR (75) s: c1 essay / code
#       (stepprobe 512) / edit (editprobe 1500 tok), mb c4 / c8 (warm 10), each with its verified nodes a step and D7 counters;
#       per metric each arm's mean +- t95, B/A pooled and per pair.
#   ID  identity under B and under F3 (if B is not F3): the arm + epoch_bump, a prefix-cache reset, the 28k copy and the 20-prompt
#       greedy set at c1 vs bz's d0 (runs/xdsm/draft, NST 0, stock) -- byte for byte.
#   OR  THE TREE ORACLE (after the main start is down; dsm-oracle.sh's driver oracle.py): two starts, the stack env +
#       RADIANCE_DS41_DSPARK_TREE=1 _B=3 _L=5 _FORCE=1, NST 20, --no-prefix-cache --debug-draft-oracle (bz d0's ids,
#       runs/xdsm/draft/oracle-ids.txt, 30 % corrupted by the harness, which on a tree puts the true id into an alternative
#       branch where the main chain's first draft is corrupted): or20 fixed window, or20a --draft-window adaptive; each two
#       oracle.py checks (one prompt, 256 greedy ids, ignore_eos): IDENTICAL to d0's ids with draft_n > accepted > 0 is the
#       pass; the /stats tree counters and the serve log's oracle / tree lines around them (accepted paths through a branch
#       where the engine reports them). Under the harness no step is issued ahead (not a speed).
# SUMMARY: per metric B and A (value, B/A %, pairs t95), each vs the references -- hx-stack's NST 5 stack (runs/xdsm/stack
#   st-stack-blocks.jsonl arm A, when it ran), hd-hybrid's DSpark + D7 + hybrid arm (runs/xdsm/hybrid3 h2-dspark-blocks.jsonl
#   arm B; else essay 57.0, code 96.6, edit 146.7, c4 114.6, c8 147.2), NST 0 + gg_gemv 2 (gz gg2-gemv2 B blocks: essay 62.5,
#   code 64.7, c4 131.7, c8 170.0; 1u edit from hybrid3's N blocks) -- the % difference and the distance to the targets
#   (1u >= 90 tok/s, c4 >= 160, c8 >= 200); tokens a step B vs A; the row cost; identity and oracle verdicts. One final
#   "RESULT xtr6 summary:" line, then "RESULT xtr6 <item>: done".
# ~55 min: evict ~1; main start ~6 (gate, modes, switch note, warm-up); S ~10 (7 blocks, F3 slow); AB ~18 (4 blocks); ID ~7;
#   OR ~11 (two starts, two checks each); summary <1. NO cc_sim, no 2.9, no BetterBench. No power / host / fan settings;
#   production compose files read only (--project-directory). Exact levers only; nothing goes into production here.
# XTR_IMAGE, XTR_ITEM, XTR_PHASES, XTR_SORDER, XTR_SGEN (512), XTR_DUR (75), XTR_PICK, XTR_HYB (1), XTR_DRAFTER_MX (0; 1 needs an
#   image with drafter_mx), XTR_TREE_B / XTR_TREE_L / XTR_TREE_FORCE (the main start's start values, 0 / 5 / 0), XTR_VRAMENV
#   (default RADIANCE_PLACE_TURNOVER=0).
# DRYRUN=1 bash dsm-tree.sh : prints every command; checks the image is on this host (docker image inspect + a no-device,
#   no-network grep of its files; XTR_DRY_NOIMAGE=1 skips that, for a syntax run away from gpu02); no server, no GPU, nothing
#   written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-tree6-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/tree6; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity: copy + greedy set; oracle-ids.txt)
RG=$R/xdsm/gg2                                  # gz-gemvacq's NST 0 + gg_gemv 2 blocks
RS=$R/xdsm/stack                                # hx-stack's NST 5 stack blocks (when it ran)
RH=$R/xdsm/hybrid3                              # hd-hybrid's DSpark + D7 + hybrid blocks (arm B) and its NST 0 blocks (arm N)
export IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-SHA8-5c13e555-l}
ITEM=${XTR_ITEM:-57-p4zyzzzzz-ds-tree6}
PHASES=${XTR_PHASES-"AB ID OR"}
SORDER=${XTR_SORDER:-"T0 F1 F2 F3d3 F1d2 P2 P3d3 T0"}
SGEN=${XTR_SGEN:-512}; DUR=${XTR_DUR:-75}; HYB=${XTR_HYB:-1}; MATCH=8; DMX=${XTR_DRAFTER_MX:-0}
TB0=${XTR_TREE_B:-0}; TL0=${XTR_TREE_L:-5}; TF0=${XTR_TREE_FORCE:-0}
for p in $PHASES; do case $p in S|AB|ID|OR) ;; *) echo "XTR_PHASES takes S AB ID OR (got '$p')"; exit 1 ;; esac; done
case $HYB in 0|1) ;; *) echo "XTR_HYB must be 0 or 1"; exit 1 ;; esac
case $DMX in 0|1) ;; *) echo "XTR_DRAFTER_MX must be 0 or 1"; exit 1 ;; esac
case "$TB0$TL0$TF0" in [0-3][1-5][01]) ;; *) echo "XTR_TREE_B 0..3, XTR_TREE_L 1..5, XTR_TREE_FORCE 0|1 (got $TB0 $TL0 $TF0)"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; TB=$HD/treeblocks2.py; ORC=$HD/oracle.py; TT=$HD/treetrace.py; DAP=$HD/da/daprobe.py; SEED=${XTR_SEED:-1234}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
NSTT=20                                          # 5 main + 3 x 5 branch node slots (RADIANCE_DS41_DSPARK_TREE=1 needs it)
# the hx-stack's NST 5 pin (dsm-stack.sh FIX + arm A with HYB), held by every block
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid": '"$HYB"', "draft_hybrid_match": '"$MATCH"', "draft_lend": 1, "draft_probe_backoff": 3}'
[ "$DMX" = 1 ] && FIX="${FIX%\}}, \"drafter_mx\": 1}"
# the arms: dspark_tree (branches), dspark_tree_depth, dspark_tree_force
armjson() {
  case $1 in
    T0)   echo '{"dspark_tree": 0, "dspark_tree_depth": 5, "dspark_tree_force": 0}' ;;
    F1)   echo '{"dspark_tree": 1, "dspark_tree_depth": 5, "dspark_tree_force": 1}' ;;
    F2)   echo '{"dspark_tree": 2, "dspark_tree_depth": 5, "dspark_tree_force": 1}' ;;
    F3)   echo '{"dspark_tree": 3, "dspark_tree_depth": 5, "dspark_tree_force": 1}' ;;
    F3d3) echo '{"dspark_tree": 3, "dspark_tree_depth": 3, "dspark_tree_force": 1}' ;;
    P2)   echo '{"dspark_tree": 2, "dspark_tree_depth": 5, "dspark_tree_force": 0}' ;;
    P3d3) echo '{"dspark_tree": 3, "dspark_tree_depth": 3, "dspark_tree_force": 0}' ;;
    F1d2) echo '{"dspark_tree": 1, "dspark_tree_depth": 2, "dspark_tree_force": 1}' ;;
    P3)   echo '{"dspark_tree": 3, "dspark_tree_depth": 5, "dspark_tree_force": 0}' ;;
    *) return 1 ;;
  esac
}
armdesc() {
  case $1 in
    T0) echo "dspark_tree 0 (the chain of 5, today's proposal)" ;; F1) echo "1 branch x depth 5, forced" ;;
    F2) echo "2 branches x depth 5, forced" ;; F3) echo "3 branches x depth 5, forced (21 verify rows)" ;;
    F1d2) echo "1 branch x depth 2, forced" ;; P3) echo "3 branches x depth 5, priced by D7 (step cost by node count)" ;;
    F3d3) echo "3 branches x depth 3, forced (14 nodes, 15 verify rows)" ;; P2) echo "2 branches x depth 5, priced by D7 (at most 15 nodes)" ;;
    P3d3) echo "3 branches x depth 3, priced by D7 (at most 14 nodes)" ;;
    *) echo "$1" ;;
  esac
}
for a in $SORDER; do armjson $a > /dev/null || { echo "XTR_SORDER names an unknown arm '$a' (T0 F1 F2 F3d3 F1d2 P2 P3d3; F3 / P3 are 21 rows)"; exit 1; }; done
case " $SORDER " in *" T0 "*) ;; *) echo "XTR_SORDER must hold T0 (the reference)"; exit 1 ;; esac
[ -n "${XTR_PICK:-}" ] && { armjson $XTR_PICK > /dev/null && [ $XTR_PICK != T0 ] || { echo "XTR_PICK must be one of F1 F2 F3d3 F1d2 P2 P3d3"; exit 1; }; }
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm's switches (+ extra) as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$FIX" "$1" "$x"
}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DS41_DSPARK_TREE RADIANCE_DS41_DSPARK_TREE_B RADIANCE_DS41_DSPARK_TREE_L RADIANCE_DS41_DSPARK_TREE_FORCE
  RADIANCE_DEBUG_ORACLE_TREE RADIANCE_DEBUG_TREE_STEPS
  RADIANCE_DS41_ROUTER_BF16 RADIANCE_DS41_EMB_RAM RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH RADIANCE_DS41_ARENA_DECODE_ROWS
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW"
VRAMENV=${XTR_VRAMENV-"RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_ARENA_DECODE_ROWS=48 RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"}
VNAMES=""
for kv in $VRAMENV; do VNAMES="$VNAMES ${kv%%=*}"; done
for kv in $VRAMENV; do
  case "$kv" in [A-Z]*=*) ;; *) echo "XTR_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac
  case " $(echo $SWITCHES) " in *" ${kv%%=*} "*) ;; *) echo "XTR_VRAMENV: ${kv%%=*} is not in the switch list"; exit 1 ;; esac
done
treeenv() {   # treeenv B L FORCE: the stack's start env (dsm-stack.sh stackenv) + lever T1 declared, its switches' start values
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $VRAMENV; do export "$kv"; done
  export RADIANCE_DS41_DSPARK_TREE=1 RADIANCE_DS41_DSPARK_TREE_B=$1 RADIANCE_DS41_DSPARK_TREE_L=$2 RADIANCE_DS41_DSPARK_TREE_FORCE=$3
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
engerr() {   # engerr: the running engine's E lines (rank prefix folded, unique, first 3) on one line; empty when it logged none
  [ "$DRY" = 1 ] && return 0
  docker logs $CN 2>&1 | grep -E '^E ' | sed -E 's/^E rank [0-9]+ step [0-9]+: /E /' | awk '!s[$0]++' | head -3 | cut -c1-400 | tr '\n' ' '
}
engstop() {   # engstop TAG PHASE: after a phase, an engine that logged E lines stopped -> VIRHE, log kept, start down, rc 1
  [ "$DRY" = 1 ] && { echo "+ engine E lines after $2 -> VIRHE, start stopped"; return 0; }
  local ee; ee=$(engerr); [ -z "$ee" ] && return 0
  unfollow; keeplog $1-engerr-$2; down
  log "RESULT xtr6 $1: VIRHE the engine stopped on an error during $2 -> start stopped (log $O/engine-$1-engerr-$2.log) -- $ee"
  return 1
}
STGUARD() { awk '{print $NF}' $RS/guard-st-stack.txt 2>/dev/null; }   # hx-stack's NST 5 guard ms/step (when it ran)
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check; rc 3 = the engine stopped on an error
  local tag=$1 lim=$2 ms ms1 st=fast ee
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && { echo "+ engine E lines after the gate's requests -> VIRHE ENGINE ERROR (rc 3, no retry); else the limit $lim with one retry"; return 0; }
  ee=$(engerr)
  if [ -n "$ee" ]; then
    keeplog $tag-engerr; echo "ENGINE-ERROR $ms" > $O/guard-$tag.txt
    log "RESULT xtr6 guard $tag: VIRHE ENGINE ERROR at the gate's first requests (fastcheck ${ms}: no number) -- $ee"
    return 3
  fi
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || { keeplog $tag-fail; return 1; }; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    ee=$(engerr)
    [ -n "$ee" ] && { keeplog $tag-engerr; echo "ENGINE-ERROR $ms" > $O/guard-$tag.txt
                      log "RESULT xtr6 guard $tag: VIRHE ENGINE ERROR at the retry's requests (first $ms1) -- $ee"; return 3; }
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT xtr6 guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts sm-n5 34.48, wk-n5 34.91, depth0b / hybrid 35.2-35.3; hx-stack st-stack NST 5 $(STGUARD || true))"
  case $st in SLOW*) return 2 ;; esac
}
t0cost() {   # t0cost: the gate's 1-user ms/step = tree 0 at NST 20 (the tree path on the chain of 5), vs the NST 5 starts
  [ "$DRY" = 1 ] && { echo "+ RESULT xtr6 t0cost guard: tree 0 at NST 20 ms/step vs hx-stack st-stack NST 5 ($RS/guard-st-stack.txt) and NST 5 starts 34.48-35.3"; return 0; }
  local g s; g=$(awk '{print $NF}' $O/guard-tr-main.txt 2>/dev/null); s=$(STGUARD)
  log "RESULT xtr6 t0cost guard: tree 0 at NST 20 (every decode step a tree verify of the chain of 5: tree_rows, the tree attention / index scores / qsa_work_tree, no attn fold d, the context pass's commits) ${g:-?} ms/step at 1 user vs hx-stack st-stack NST 5 ${s:-not run} ms/step ($(python3 -c 'import sys
try: print("%+.1f %%" % ((float(sys.argv[1]) / float(sys.argv[2]) - 1) * 100))
except Exception: print("-")' "${g:-x}" "${s:-x}")) | NST 5 starts sm-n5 34.48, wk-n5 34.91, depth0b / hybrid 35.2-35.3 ($(python3 -c 'import sys
try: print("%+.1f %% vs 34.91" % ((float(sys.argv[1]) / 34.91 - 1) * 100))
except Exception: print("-")' "${g:-x}")) | both gates before any fn4 pin (no drafter_mx in either); start sizing NST 20 vs NST 5 (10.11. logs): arena +8.4 MiB at 256 tokens, KV 6.04 vs 6.09-6.10 GiB a card, experts 17.92 GiB both, promoted units 2467 vs 2473"
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xtr6 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xtr6 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xtr6 $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xtr6 $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the t3 notes + rule-w + the lever K1 note + the VRAM env + the tree lines -> 0, else 1 (start stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate), ARGATHER 4-row note, no shared-expert-lane note, staging line, rule-w in every recorded pass, the lever K1 note (all three declared, =1), VRAM env [${VRAMENV:-none}] in the container, 4 mover lines$(case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*) echo " each '0 more kept free'" ;; esac); the DSpark drafter line, 'draft trees: asked', the drafter's node count (20), the adaptive draft depth line"; return 0; }
  local MLF SL AL XL KL NW NR MV CE kv bad="" ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  log "RESULT xtr6 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | mover: ${MV:-no line} | tree env: $(echo "$CE" | grep -E '^RADIANCE_DS41_DSPARK_TREE' | tr '\n' ' ')"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xtr6 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xtr6 $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xtr6 $1: VIRHE a shared-expert-lane note: '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xtr6 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xtr6 $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xtr6 $1: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  for kv in $VRAMENV RADIANCE_DS41_DSPARK_TREE=1; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { log "RESULT xtr6 $1: VIRHE the start env did not reach the container:$bad"; ok=0; }
  case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')" = 4 ] \
      || { log "RESULT xtr6 $1: VIRHE turnover 0 asked but not every rank keeps 0 slots free: $MV"; ok=0; } ;;
  esac
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xtr6 $1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft trees: asked" || { log "RESULT xtr6 $1: VIRHE no 'draft trees: asked' line (the tree is not declared: RADIANCE_DS41_DSPARK_TREE / NST 20?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xtr6 $1: VIRHE no adaptive draft depth line"; ok=0; }
  docker logs $CN 2>&1 | grep -qE "'dspark' proposes $NSTT token" || log "RESULT xtr6 $1: note -- no \"'dspark' proposes $NSTT token(s)\" line (information)"
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xtr6 $1 drafter / speculation / tree lines"; return 0; }
  log "RESULT xtr6 $1 drafter / speculation / tree lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|draft trees|tree|oracle' | sed 's/^[A-Z][^ ]* //' | sort -u | head -14 | tr '\n' '|' | cut -c1-2400)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
pcreset() {
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xtr6 $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote() {   # swnote TAG: GET at start; the tree switches' refusals; T0's set, F3's set, T0's set again, each applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_adaptive 1, draft_capable 1, epoch 0, dspark_tree $TB0, dspark_tree_depth $TL0, dspark_tree_force $TF0); POST dspark_tree 4 -> 400, dspark_tree_depth 0 -> 400, dspark_tree_force 2 -> 400, ds41_kvfold 2 -> 400; POST $(armset "$(armjson F3)"), then $(armset "$(armjson T0)"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c1 c2 c3 c4
  s0=$(fn4get); c1=$(fn4code '{"dspark_tree": 4}'); c2=$(fn4code '{"dspark_tree_depth": 0}'); c3=$(fn4code '{"dspark_tree_force": 2}'); c4=$(fn4code '{"ds41_kvfold": 2}')
  fn4set "$(armset "$(armjson F2)")"; nudge; s1=$(fn4get)
  fn4set "$(armset "$(armjson T0)")"; nudge; s2=$(fn4get)
  log "RESULT xtr6 $1 switch note: at start $s0 | dspark_tree 4 -> HTTP $c1, dspark_tree_depth 0 -> HTTP $c2, dspark_tree_force 2 -> HTTP $c3, ds41_kvfold 2 -> HTTP $c4 | after F3's set: $s1 | after T0's set: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); f3, t0 = json.loads(sys.argv[4]), json.loads(sys.argv[5])
b0, l0, x0 = (int(v) for v in sys.argv[6:9]); w = s0["switches"]
ok = w.get("gg_gemv") == 2 and w.get("aql_acq") == 2 and all(w.get(k) == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and w.get("draft_adaptive") == 1 and s0.get("draft_capable") == 1 and s0.get("epoch") == 0 \
     and w.get("dspark_tree") == b0 and w.get("dspark_tree_depth") == l0 and w.get("dspark_tree_force") == x0
ok = ok and all(s1["switches"].get(k) == v for k, v in f3.items()) and not s1["pending"] and s1["epoch"] >= 1
ok = ok and all(s2["switches"].get(k) == v for k, v in t0.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$(armjson F2)")" "$(armset "$(armjson T0)")" $TB0 $TL0 $TF0 2>/dev/null \
    && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ "$c4" = 400 ]
}
treelines() {   # treelines FILE TAG: every BLOCK line and the TREE summary's metrics as RESULT lines
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xtr6 $2 ${l:0:1800}"; done < <(grep -E '^BLOCK ' $1)
  grep -q '^TREE ' $1 || log "RESULT xtr6 $2: VIRHE no TREE summary line ($(tail -2 $1 | tr '\n' ' ' | cut -c1-300))"
}
screen() {   # screen TAG: one block an arm (c1 essay + code), the row cost, the pick
  local args="" a
  for a in $(echo $SORDER | tr ' ' '\n' | awk '!s[$0]++'); do args="$args --arm '$a=$(armjson $a)'"; done
  run "timeout 3600 python3 -I $TB --url $URL --mb-dir $D --edit $EP --label $1-s $args --order '$SORDER' --pair T0,F2 --phases essay,code --gen $SGEN --fixed '$FIX' --rowcost --out $O/$1-screen-blocks.jsonl > $O/$1-screen-blocks.txt 2>&1"
  if [ "$DRY" = 1 ]; then echo "+ RESULT xtr6 screen lines, RESULT xtr6 rowcost (essay / code: ms a row, intercept, r2, rows range), RESULT xtr6 pick (geometric mean of essay and code tok/s vs T0)"; PICK=${XTR_PICK:-P2}; return 0; fi
  treelines $O/$1-screen-blocks.txt screen
  python3 - $O/$1-screen-blocks.txt "${XTR_PICK:-}" > $O/$1-screen.txt 2>&1 <<'PY'
import json, math, sys
try: d = json.loads(next(l for l in open(sys.argv[1]) if l.startswith("TREE "))[5:])
except Exception as e:
    print("PICK P2 (no screen summary: %s)" % e); sys.exit(0)
M = d.get("metrics", {})
def mean(metric, arm): return ((M.get(metric) or {}).get("arms") or {}).get(arm, {}).get("mean")
arms = [a for a in d.get("arms", {})]
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
e0, c0 = mean("c1 essay tok/s", "T0"), mean("c1 code tok/s", "T0")
score = {}
for a in arms:
    e, c = mean("c1 essay tok/s", a), mean("c1 code tok/s", a)
    rel = lambda v, r: "%+.1f %%" % ((v / r - 1) * 100) if v and r else "-"
    print("SCREEN %s: essay %s tok/s (%s vs T0) %s tok/step %s ms/step %s rows/step acc %s | code %s tok/s (%s vs T0) %s tok/step %s ms/step %s rows/step acc %s" % (
        a, f(e), rel(e, e0), f(mean("c1 essay tok/step", a)), f(mean("c1 essay ms/step", a)), f(mean("c1 essay rows/step", a)),
        f(mean("c1 essay acc", a), 3), f(c), rel(c, c0), f(mean("c1 code tok/step", a)), f(mean("c1 code ms/step", a)),
        f(mean("c1 code rows/step", a)), f(mean("c1 code acc", a), 3)))
    if a != "T0" and e and c and e0 and c0: score[a] = math.sqrt((e / e0) * (c / c0))
for nm, rc in (d.get("rowcost") or {}).items():
    fit = rc.get("fit")
    pts = " ".join("%s(%s, %s)" % (p[0], f(p[1]), f(p[2])) for p in rc.get("points", []))
    print("ROWCOST %s: %s | points (arm, rows a step, ms a step): %s" % (nm, (
        "ms a step = %.2f + %.3f x rows (r2 %s, n %d, rows %.2f..%.2f): %.2f ms a verify row at one user" % (
            fit["intercept_ms"], fit["ms_per_row"], fit["r2"], fit["n"], fit["rows_min"], fit["rows_max"], fit["ms_per_row"])) if fit else "no fit (rows do not vary)", pts))
forced = sys.argv[2]
if forced: pick, why = forced, "XTR_PICK"
elif score: pick = max(score, key=score.get); why = "highest geometric mean of essay and code tok/s vs T0"
else: pick, why = "P2", "no T0 or no other good arm: the priced arm"
print("PICK %s (%s; scores %s; %d blocks, %d flagged)" % (pick, why, ", ".join("%s %.4f" % kv for kv in sorted(score.items(), key=lambda kv: -kv[1])) or "-",
                                                       d.get("blocks", 0), d.get("blocks_bad", 0)))
PY
  while IFS= read -r l; do
    case "$l" in SCREEN*) log "RESULT xtr6 screen ${l#SCREEN }" ;; ROWCOST*) log "RESULT xtr6 rowcost ${l#ROWCOST }" ;; PICK*) log "RESULT xtr6 pick: ${l#PICK }" ;; *) log "RESULT xtr6 screen: $l" ;; esac
  done < $O/$1-screen.txt
  PICK=$(grep -m1 '^PICK ' $O/$1-screen.txt | awk '{print $2}'); armjson ${PICK:-x} > /dev/null || PICK=P2
}
abblocks() {   # abblocks TAG: ABBA, A = T0, B = $PICK
  run "timeout 7200 python3 -I $TB --url $URL --mb-dir $D --edit $EP --label $1-ab --arm 'A=$(armjson T0)' --arm 'B=$(armjson $PICK)' --order 'A B B A' --pair A,B --phases essay,code,edit,essay_s,code_s,edit_s,c4,c8 --dur $DUR --warm 10 --seed $SEED --daprobe $DAP --fixed '$FIX' --out $O/$1-ab-blocks.jsonl > $O/$1-ab-blocks.txt 2>&1"
  treelines $O/$1-ab-blocks.txt ab
}
copyt() {
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xtr6 $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
ident() {   # ident TAG ARM: the arm's set + epoch_bump, a prefix-cache reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 arm=$2 r
  fn4set "$(armset "$(armjson $arm)" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity ($arm: $(armdesc $arm)): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  log "RESULT xtr6 $tag identity ran under: $(fn4get | cut -c1-700)"
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  echo "copy-$arm $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xtr6 $tag identity ($arm: $(armdesc $arm)): 28k copy vs bz d0 (NST 0, stock): $r" ;;
    *) log "RESULT xtr6 $tag identity ($arm: $(armdesc $arm)): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden" ;; esac
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
  echo "gs-$arm $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xtr6 $tag identity ($arm: $(armdesc $arm)): greedy set c1 vs bz d0: $r" ;;
    *) log "RESULT xtr6 $tag identity ($arm: $(armdesc $arm)): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
}
mainstart() {   # the main start: S, AB, ID (each when in XTR_PHASES)
  local tag=tr-main P=$O/tr-main rc
  treeenv $TB0 $TL0 $TF0; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=$NSTT XARGS="$XARGS --live --draft-window adaptive"
  log "RESULT xtr6 $tag: start (the hx-stack NST 5 start at NST $NSTT + RADIANCE_DS41_DSPARK_TREE=1, start values dspark_tree $TB0 / depth $TL0 / force $TF0; pin $FIX) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  # limit 70 ms/step: the stream's NST 5 DSpark gate (fast 34.48-35.3, 2x); the 10.11. run measured no NST 20 value (engine error)
  gate $tag 70; rc=$?
  [ $rc = 1 ] && { log "RESULT xtr6 $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-600)"; down; return 1; }
  [ $rc = 3 ] && { log "RESULT xtr6 $tag: VIRHE the engine stopped on an error at the gate -> start skipped (log $O/engine-$tag-engerr.log)"; down; return 1; }
  t0cost
  [ $rc = 2 ] && { log "RESULT xtr6 $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xtr6 $tag: VIRHE the modes are not the stack + tree line's -> start stopped"; down; return 1; }
  draftlines $tag
  swnote $tag || { log "RESULT xtr6 $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  PICK=${XTR_PICK:-P2}
  case " $PHASES " in *" S "*) stopfile || screen $tag ;; esac
  engstop $tag screen || return 1
  log "RESULT xtr6 $tag: B arm for AB / ID = $PICK ($(armdesc $PICK): $(armjson $PICK))"
  case " $PHASES " in *" AB "*) stopfile || abblocks $tag ;; esac
  engstop $tag AB || return 1
  case " $PHASES " in *" ID "*)
    stopfile || ident $tag-iB $PICK
    ;;
  esac
  engstop $tag ID || return 1
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  log "RESULT xtr6 $tag: done"
}
ocheck() {   # ocheck TAG N: one oracle.py check (+ /metrics' draft counters around it) -> the result line and $O/oracle.txt;
             # rc 1 when the request failed (no ORACLE line: ERROR, with oracle.py's error and the engine's E lines)
  run "curl -s -m 15 $URL/metrics > $O/$1-check$2-metrics0.txt 2>/dev/null"
  run "curl -s -m 15 $URL/stats > $O/$1-check$2-stats0.json 2>/dev/null"
  run "timeout 900 python3 -I $ORC check $URL $STATSDIR/oracle-ids.txt $O/$1-check$2.json > $O/$1-check$2.txt 2>&1"
  run "curl -s -m 15 $URL/metrics > $O/$1-check$2-metrics1.txt 2>/dev/null"
  run "curl -s -m 15 $URL/stats > $O/$1-check$2-stats1.json 2>/dev/null"
  [ "$DRY" = 1 ] && return 0
  local r dm v ee tr tw; r=$(grep -m1 '^ORACLE' $O/$1-check$2.txt | cut -c8- | cut -c1-300)
  dm=$(python3 -c 'import re,sys
K = ["radiance:draft_tokens_total", "radiance:draft_accepted_total", "radiance:engine_steps_decode_total", "radiance:decode_tokens_total"]
def ld(p):
    try: t = open(p).read()
    except OSError: return {}
    o = {}
    for k in K:
        m = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", t, re.M)
        if m: o[k] = float(m.group(2))
    return o
a, b = ld(sys.argv[1]), ld(sys.argv[2])
d = {k.split(":")[1]: b[k] - a.get(k, 0) for k in K if k in b}
s = d.get("engine_steps_decode_total") or 0
print(("/metrics deltas " + ", ".join("%s %g" % kv for kv in d.items()) + ("; drafted nodes a decode step %.2f, accepted a step %.2f" % (d.get("draft_tokens_total", 0) / s, d.get("draft_accepted_total", 0) / s) if s else "")) if d else "no /metrics draft counters")' $O/$1-check$2-metrics0.txt $O/$1-check$2-metrics1.txt 2>&1 | cut -c1-500)
  # the engine's /stats tree counters (ds41-tree 6b143a9: tree_verifies, tree_nodes, tree_wide, tree_branch_nodes,
  # tree_branch_paths, tree_branch_accepted; only under a tree layout): their deltas over the check; tw = tree_wide's delta
  tr=$(python3 -c 'import json,sys
def ld(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = ld(sys.argv[1]), ld(sys.argv[2])
t = {k: v - (a.get(k) or 0) for k, v in b.items() if k.startswith("tree_") and isinstance(v, (int, float))}
print(("/stats " + ", ".join("%s +%g" % kv for kv in t.items())) if t else "no tree counters in /stats")' $O/$1-check$2-stats0.json $O/$1-check$2-stats1.json 2>&1 | cut -c1-400)
  # the tree requirement: branch trees verified (tree_wide), or at dspark_tree 0 (or20t0) tree verifies at all
  local tkey=tree_wide; case $1 in *t0) tkey=tree_verifies ;; esac
  tw=$(echo "$tr" | grep -oE "$tkey \+[0-9]+" | grep -oE '[0-9]+$'); tw=${tw:--1}
  if [ -z "$r" ]; then
    ee=$(engerr)
    echo "$1-check$2 ERROR" >> $O/oracle.txt
    log "RESULT xtr6 $1 check $2: VIRHE ERROR -- the request failed, no ids to compare (oracle.py: $(grep -E 'Error|error' $O/$1-check$2.txt | tail -1 | cut -c1-200)) | engine: ${ee:-no E line} | /health $(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health 2>/dev/null)"
    return 1
  fi
  case "$r" in
    *'"identical": true'*)
      v=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); n, a = d.get("draft_n") or 0, d.get("draft_n_accepted") or 0; print("ok" if n > a > 0 else "vacuous")' "$r" 2>/dev/null)
      if [ "$v" = ok ] && [ "$tw" -gt 0 ]; then echo "$1-check$2 IDENTICAL" >> $O/oracle.txt; log "RESULT xtr6 $1 check $2: IDENTICAL to d0's ids, trees verified ($tkey +$tw) and drafts rejected ($r) | $tr | $dm"
      elif [ "$v" = ok ]; then echo "$1-check$2 NOTREE" >> $O/oracle.txt; log "RESULT xtr6 $1 check $2: IDENTICAL to d0's ids but VIRHE no tree with branches verified ($tr) -- the check tested chains only ($r) | $dm"
      else echo "$1-check$2 VACUOUS" >> $O/oracle.txt; log "RESULT xtr6 $1 check $2: IDENTICAL to d0's ids but VIRHE vacuous -- no rejected draft ($r) | $tr | $dm"; fi ;;
    *) echo "$1-check$2 DIFFERENT" >> $O/oracle.txt; log "RESULT xtr6 $1 check $2: VIRHE DIFFERENT from d0's ids ($r) -- the tree verify path is not exact | $tr | $dm | engine: $(engerr)" ;;
  esac
  return 0
}
oarm() {   # oarm or20|or20a: a tree-oracle start (3 branches x depth 5 forced at start), two checks
  local a=$1 tag=tr-$1
  case $a in or20t0) treeenv 0 5 0 ;; or15) treeenv 2 5 1 ;; or14) treeenv 3 3 1 ;; *) treeenv 3 5 1 ;; esac
  export RADIANCE_DEBUG_ORACLE_TREE=1 RADIANCE_DEBUG_TREE_STEPS=400 MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=$NSTT XARGS="$XARGS --live --no-prefix-cache --debug-draft-oracle /stats/oracle-ids.txt"
  case $a in or20|or20t0|or15|or14) ;; or20a) export XARGS="$XARGS --draft-window adaptive" ;; *) log "RESULT xtr6: VIRHE unknown oracle arm $a"; return 1 ;; esac
  log "RESULT xtr6 $tag: start IMAGE=$IMAGE MODEL=$MODEL NST $NSTT, $(case $a in or20t0) echo "dspark_tree 0 at start (the chain of 5 through the tree verify path)" ;; or15) echo "2 branches x depth 5 forced at start (15 nodes, 16 verify rows)" ;; or14) echo "3 branches x depth 3 forced at start (14 nodes, 15 verify rows)" ;; *) echo "the tree forced at start (3 branches x depth 5, 21 rows)" ;; esac) + --debug-draft-oracle $(desc) env: $(env | grep -E '^RADIANCE_(DS41_DSPARK_TREE|DEBUG_ORACLE_TREE|DEBUG_TREE_STEPS)' | sort | tr '\n' ' ')"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xtr6 $tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-600)"; echo "$tag-start FAILED" >> $O/oracle.txt; return 1; }
  if [ "$DRY" = 1 ]; then echo "+ RESULT xtr6 $tag harness lines (scheduler: every proposal replaced; engine: no step ahead beside 'dspark'; draft trees: asked; the oracle's tree line)"
  else
    log "RESULT xtr6 $tag harness lines: $(docker logs $CN 2>&1 | grep -iE -- '--debug-draft-oracle|DSpark drafter|speculation: |draft trees|oracle' | sed 's/^[A-Z] *//' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
    docker logs $CN 2>&1 | grep -q "every proposal of the draft head" && docker logs $CN 2>&1 | grep -q "beside the draft head 'dspark': no step is issued ahead" \
      || log "RESULT xtr6 $tag: VIRHE the serve log lacks the harness-beside-a-head lines"
    docker logs $CN 2>&1 | grep -q "draft trees: asked" \
      || log "RESULT xtr6 $tag: VIRHE no 'draft trees: asked' under the harness -- the oracle verified chains, not trees (the checks below then test the chain path only)"
  fi
  if ! ocheck $tag 1 && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health 2>/dev/null)" != 200 ]; then
    echo "$tag-check2 SKIPPED" >> $O/oracle.txt; log "RESULT xtr6 $tag check 2: skipped -- the server is down after check 1's error"
  elif [ "$DRY" != 1 ] && tail -1 $O/oracle.txt | grep -q ' DIFFERENT$'; then
    log "RESULT xtr6 $tag check 2: skipped -- check 1 was DIFFERENT (deterministic)"
  else
    ocheck $tag 2
  fi
  keeplog $tag; down
  # treetrace.py's readings over the start's trace lines (check 1's verifies first)
  if [ "$DRY" = 1 ]; then echo "+ python3 -I $TT $O/engine-$tag.log $STATSDIR/oracle-ids.txt -> RESULT xtr6 $tag trace lines"
  else
    python3 -I $TT $O/engine-$tag.log $STATSDIR/oracle-ids.txt > $O/$tag-trace.txt 2>&1
    while IFS= read -r l; do log "RESULT xtr6 $tag trace ${l:0:900}"; done < $O/$tag-trace.txt
  fi
  log "RESULT xtr6 $tag: done"
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xtr6 ab lines (per metric A / B mean +- t95, B/A pooled + pairs), RESULT xtr6 summary: per metric B, A, B/A %, vs hx-stack NST 5 / hd-hybrid DSpark+D7+hybrid / NST 0 + gg_gemv 2, target distance; tok/step B vs A; the row cost; identity; oracle"; return 0; }
  python3 - $O/tr-main-ab-blocks.txt $O/tr-main-screen.txt $RS/st-stack-blocks.jsonl $RH/h2-dspark-blocks.jsonl $RG/gg2-gemv2-blocks.jsonl $RH/h2-nst0-blocks.jsonl $O/identity.txt $O/oracle.txt "$PICK" "$(armdesc $PICK)" > $O/summary.txt 2>&1 <<'PY'
import json, statistics, sys
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def lines(p):
    try: return [l.strip() for l in open(p) if l.strip()]
    except OSError: return []
try: d = json.loads(next(l for l in open(sys.argv[1]) if l.startswith("TREE "))[5:])
except Exception: d = {}
scr = lines(sys.argv[2]); stack, hyb, gz, h2n = (rows(p) for p in sys.argv[3:7])
ident, orc = lines(sys.argv[7]), lines(sys.argv[8]); pick, pdesc = sys.argv[9], sys.argv[10]
M = d.get("metrics", {})
def mean(metric, arm): return ((M.get(metric) or {}).get("arms") or {}).get(arm, {}).get("mean")
def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
def ref_get(res, ph, key):   # ggblocksdz2 / ggblocks shapes: res.c1.<prompt>, res.c1.rows[], res.<ph>
    if ph in ("essay", "code"):
        c = res.get("c1") or {}
        if isinstance(c.get(ph), dict): return num(c[ph].get(key))
        r = next((x for x in (c.get("rows") or []) if x.get("prompt") == ph), None)
        return num((r or {}).get(key)) if r else num((res.get(ph) or {}).get(key))
    return num((res.get(ph) or {}).get(key))
def ref(rs, arm, ph, key, hard=None):
    v = [x for x in (ref_get(r.get("res") or {}, ph, key) for r in rs if r.get("ok") and r.get("arm") == arm) if x is not None]
    return (statistics.mean(v), "") if v else (hard, " (logged)" if hard is not None else "")
f1 = lambda v, n=1: "-" if v is None else format(v, ".%df" % n)
pct = lambda v, r: ("%+.1f %%" % ((v / r - 1) * 100)) if v and r else "-"
def tgt(v, t):
    if t is None or v is None: return ""
    return "; target >= %g: %s" % (t, "MET (+%.1f)" % (v - t) if v >= t else "%.1f short, +%.1f %% needed" % (t - v, (t / v - 1) * 100))
# (label, treeblocks metric, ref phase, ref key, hd-hybrid hard, NST0 hard, target)
ROWS = [("1u essay tok/s", "c1 essay tok/s", "essay", "decode_tps", 57.0, 62.5, 90.0),
        ("1u code tok/s", "c1 code tok/s", "code", "decode_tps", 96.6, 64.7, 90.0),
        ("1u edit tok/s", "c1 edit tok/s", "edit", "decode_tps", 146.7, None, 90.0),
        ("c4 aggregate tok/s", "c4 aggregate tok/s", "c4", "agg_dec_tps", 114.6, 131.7, 160.0),
        ("c8 aggregate tok/s", "c8 aggregate tok/s", "c8", "agg_dec_tps", 147.2, 170.0, 200.0)]
STEP = [("1u essay", "c1 essay"), ("1u code", "c1 code"), ("1u edit", "c1 edit"), ("c4", "c4"), ("c8", "c8")]
out = []
for nm, ent in M.items():
    arms = ent.get("arms") or {}
    a, b = arms.get("A", {}), arms.get("B", {})
    print("AB %s: A %s%s | B %s%s | B/A pooled %s, pairs %s (mean %s, t95 +- %s %%)" % (nm, f1(a.get("mean"), 2), "" if a.get("t95") is None else " +- " + f1(a["t95"], 2),
          f1(b.get("mean"), 2), "" if b.get("t95") is None else " +- " + f1(b["t95"], 2),
          "-" if ent.get("pooled") is None else "%+.2f %%" % ((ent["pooled"] - 1) * 100),
          " ".join("%+.2f" % ((x - 1) * 100) for x in ent.get("pairs", [])) or "-",
          "-" if ent.get("pair_mean") is None else "%+.2f %%" % ((ent["pair_mean"] - 1) * 100), "-" if ent.get("ci95") is None else "%.2f" % (ent["ci95"] * 100)))
for lab, mk, ph, key, hh, h0, t in ROWS:
    va, vb = mean(mk, "A"), mean(mk, "B")
    ent = M.get(mk) or {}
    vs, _ = ref(stack, "A", ph, key)
    vh, nh = ref(hyb, "B", ph, key, hh)
    v0, n0 = ref(gz, "B", ph, key, h0) if ph != "edit" else ref(h2n, "N", ph, key)
    out.append("%s: B %s vs A %s (%s; pairs t95 +- %s %%) | vs hx-stack NST 5 %s (%s) | vs hd-hybrid DSpark+D7+hybrid %s%s (%s) | vs NST 0 + gg_gemv 2 %s%s (%s)%s" % (
        lab, f1(vb), f1(va), pct(vb, va), "-" if ent.get("ci95") is None else "%.1f" % (ent["ci95"] * 100),
        f1(vs), pct(vb, vs) if vs else "not run", f1(vh), nh, pct(vb, vh), f1(v0), n0, pct(vb, v0), tgt(vb, t) + (" (A%s)" % tgt(va, t)[1:] if tgt(va, t) else "")))
    # the T0 cost (tree 0 at NST 20, the tree path on the chain of 5) vs hx-stack's NST 5 stack: what the tree must win back
    print("T0COST %s: A = tree 0 at NST 20 %s vs hx-stack NST 5 arm A %s (%s) [hx-stack A pins drafter_mx 1, absent here]" % (
        lab, f1(va), f1(vs) if vs else "not run", pct(va, vs) if vs else "-"))
import re as _re
_t0 = next((l for l in scr if l.startswith("SCREEN T0:")), None)
if _t0:
    for ph in ("essay", "code"):
        _m = _re.search(ph + r" ([0-9.]+) tok/s", _t0); _v = float(_m.group(1)) if _m else None
        _s, _ = ref(stack, "A", ph, "decode_tps")
        print("T0COST screen 1u %s: T0 = tree 0 at NST 20 %s tok/s vs hx-stack NST 5 arm A %s (%s) [drafter_mx 1 there]" % (ph, f1(_v), f1(_s) if _s else "not run", pct(_v, _s) if _s else "-"))
ts = []
for lab, p in STEP:
    k = "%s tok/step" % p
    va, vb = mean(k, "A"), mean(k, "B")
    rk = ("%s rows/step" % p) if p.startswith("c1") else ("%s drafts/step" % p)
    ts.append("%s %s -> %s (%s; %s %s -> %s)" % (lab, f1(va, 2), f1(vb, 2), pct(vb, va), "verify rows a step" if p.startswith("c1") else "verified nodes a step",
                                                 f1(mean(rk, "A"), 2), f1(mean(rk, "B"), 2)))
rc = [l[8:] for l in scr if l.startswith("ROWCOST ")]
pk = next((l[5:] for l in scr if l.startswith("PICK ")), "no screen")
s = "B = %s (%s; pick: %s) vs A = dspark_tree 0, in process ABBA, mb 75 s || %s || tokens a step A -> B: %s || row cost at c1: %s || identity: %s || oracle: %s || %d AB blocks, %d flagged" % (
    pick, pdesc, pk, " || ".join(out), "; ".join(ts), " ; ".join(x.split(" | points")[0] for x in rc) or "no screen",
    ", ".join(ident) or "none recorded", ", ".join(orc) or "not run", d.get("blocks", 0), d.get("blocks_bad", 0))
print("SUMMARY " + s)
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xtr6 ${l:0:2500}" ;; esac; done < $O/summary.txt
  log "RESULT xtr6 summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-5000)"
}
# ---------------------------------------------------------------- main
PICK=${XTR_PICK:-P2}
log "RESULT xtr6 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) phases [$PHASES] | main start NST $NSTT + RADIANCE_DS41_DSPARK_TREE=1 (start values B $TB0 L $TL0 force $TF0), pin $FIX, VRAM env [${VRAMENV:-none}] | screen order [$SORDER] (c1 essay + code, $SGEN tok) | AB A = T0 vs B = the pick${XTR_PICK:+ (XTR_PICK $XTR_PICK)}, ABBA, mb $DUR s | identity under B and F2 vs bz d0 | oracle FIRST ${XTR_OARMS:-or15} (2 x 5 / 3 x 3 forced, at most 16 verify rows; d0 ids) | refs hx-stack NST 5 stack, hd-hybrid, gz gg2-gemv2 (NST 0 + D2) | targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s"
# the image, on this host, read only (in DRYRUN too): the stack's features (depth 0 / hybrid, reset_prefix_cache, the R1 fix,
# K1, D2) and lever T1 (the tree switches in the registry, the plugin's env, the tree ops in libr4d)
if [ "$DRY" = 1 ] && [ "${XTR_DRY_NOIMAGE:-0}" = 1 ]; then
  echo "+ image check SKIPPED (XTR_DRY_NOIMAGE=1, a syntax run): $IMAGE"
else
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xtr6: VIRHE image $IMAGE not on this host"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF "ds41_onefill, draft_depth0, draft_probe, draft_hybrid," /opt/radiance/bin/radiance && grep -qzF "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -qzF "draft hybrid: a sequence whose prompt lookup" /opt/radiance/bin/radiance && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF "bytes win where no snapshot rests on the first copy" /opt/radiance/bin/radiance && grep -qzF dspark_tree_depth /opt/radiance/bin/radiance && grep -qzF dspark_tree_force /opt/radiance/bin/radiance && grep -qzF tree_branch_paths /opt/radiance/bin/radiance && grep -qzF RADIANCE_DEBUG_TREE_STEPS /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_DSPARK_TREE /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF attn_sparse_sink_tree /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF qsa_work_tree /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xtr6: VIRHE $IMAGE is not a T1 tree image on the stack (depth 0 / hybrid, reset_prefix_cache, R1 fix, K1, D2, the dspark_tree switches, the /stats tree counters (6b143a9), RADIANCE_DS41_DSPARK_TREE in deepseek41.so, attn_sparse_sink_tree / qsa_work_tree in libr4d)"; exit 1; }
  [ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is a T1 tree image on the stack -- OK"
  if [ "$DMX" = 1 ]; then
    [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF drafter_mx /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_DRAFTER_MX /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
      || { log "RESULT xtr6: VIRHE XTR_DRAFTER_MX=1 but $IMAGE has no drafter_mx"; exit 1; }
  fi
fi
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xtr6: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_DSPARK_TREE RADIANCE_DS41_DSPARK_TREE_B RADIANCE_DS41_DSPARK_TREE_L RADIANCE_DS41_DSPARK_TREE_FORCE RADIANCE_DEBUG_TREE_STEPS $VNAMES; do
    [ -e $Y ] && grep -q "^ *$v:" $Y || { log "RESULT xtr6: VIRHE $Y missing or does not pass $v"; exit 1; }; done
  for f in $GS $EP $TB $ORC $TT $DAP $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xtr6: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xtr6: VIRHE $M225 missing or not verified"; exit 1; }
  case " $PHASES " in *" OR "*) [ -s $RD/oracle-ids.txt ] || { log "RESULT xtr6: VIRHE no oracle reference ids $RD/oracle-ids.txt"; exit 1; } ;; esac
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RG/gg2-gemv2-blocks.jsonl $RH/h2-dspark-blocks.jsonl $RS/st-stack-blocks.jsonl; do [ -s $f ] || log "RESULT xtr6: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $TB $ORC $TT $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/oracle-ids.txt $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RG/gg2-gemv2-blocks.jsonl $RH/h2-dspark-blocks.jsonl $RH/h2-nst0-blocks.jsonl $RS/st-stack-blocks.jsonl; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for v in RADIANCE_DS41_DSPARK_TREE RADIANCE_DS41_DSPARK_TREE_B RADIANCE_DS41_DSPARK_TREE_L RADIANCE_DS41_DSPARK_TREE_FORCE $VNAMES; do echo "+ the yaml passes $v: $(grep -c "^ *$v:" $Y 2>/dev/null)"; done
  # every helper the body calls resolves: the libraries' functions, the helper files, the python drivers parse
  hmiss=""
  for fn in log run mkd desc best_vars serve_args restart check evict down stopfile; do
    [ "$(type -t $fn)" = function ] || hmiss="$hmiss $fn()"; done
  for fn in thermwin thermnow; do [ "$(type -t $fn)" = function ] || echo "+ note: $fn() not defined (thermal lines read 'n/a')"; done
  for f in $Y $GS $EP $TB $ORC $TT $DAP $D/stepprobe.py $D/mb.py $D/ds41_tests.py $D/fastcheck.py $C; do [ -e "$f" ] || hmiss="$hmiss $f"; done
  for f in $GS $EP $TB $ORC $TT $DAP; do python3 -I -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || hmiss="$hmiss $f(syntax)"; done
  for v in D R L CN URL C; do [ -n "${!v:-}" ] || hmiss="$hmiss \$$v"; done
  [ -z "$hmiss" ] && echo "+ HELPERS OK: functions log run mkd desc best_vars serve_args restart check evict down stopfile, files yaml greedyset editprobe treeblocks oracle stepprobe mb ds41_tests fastcheck, compose dir $C" \
                  || { echo "+ HELPERS MISSING:$hmiss"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/oracle.txt; }
evict
has() { case " $PHASES " in *" $1 "*) return 0 ;; esac; return 1; }
orphase() { has OR || return 0; run "cp $RD/oracle-ids.txt $STATSDIR/oracle-ids.txt"; for a in ${XTR_OARMS:-or15}; do stopfile && break; oarm $a; done; }
orpass() { [ "$DRY" = 1 ] && return 0; [ -s $O/oracle.txt ] && ! grep -qv ' IDENTICAL$' $O/oracle.txt; }
if [ "${XTR_OR_FIRST:-1}" = 1 ]; then
  # THE ORACLE FIRST (11.11.): the speed phases run only when every oracle check is IDENTICAL with trees verified --
  # an inexact tree path has no speed worth measuring, and its cards go to the next item ~40 min earlier
  orphase
  if has S || has AB || has ID; then
    if ! has OR || orpass; then stopfile || mainstart
    else log "RESULT xtr6: VIRHE the tree oracle is not IDENTICAL in every check ($(tr '\n' ' ' < $O/oracle.txt)) -> the speed phases [S AB ID] skipped"; fi
  fi
else
  if has S || has AB || has ID; then stopfile || mainstart; fi
  orphase
fi
summary
unset $SWITCHES
down
log "RESULT xtr6 $ITEM: done -- $O"
