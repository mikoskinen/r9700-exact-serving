# DPA: data-parallel attention for DeepSeek-V4.1 EXL3 2.25 on Radiance TP4 (design study, 12.11.2026)

This is a design study only: nothing was built and no GPU was used.

**Sources**
- Code: `~/gpu/ds41/exl3/radiance-int7` (ds41-int7 0391a6e). "cpp" means `arch/deepseek41/deepseek41.cpp`.
- Start log: gpu02 `~/gpu/ds41/exl3-test/runs/xdsm/stack4/start-s4-n5.log`, cited as "log:N".
- Profiles: `xdsm/dsm-225-prof-{c4,c8,pf128}-op.txt`. These are PROFILED device ms, not speeds.
- Speeds: stack4 `summary.txt`. c8 = 173.3 tok/s at 8.07 tok/step = 46.6 ms a step.
- LEVERS rows:
  - VA: shard sizes and the expert-unit slope.
  - E2: the 1-user step of 14.883 ms; a large GEMV runs at 568 GB/s.
  - NVFP4 latent / NVG.
- Served config: TP4, `--max-num-seqs 8`, `--max-model-len 524288`, `--num-speculative-tokens 5`, MNBT 4160 (log:15528-15552).
- Model: D = 5120, 64 heads × 512 with one latent KV head, q_lora 1280, o_groups 8 × o_rank 1024, window 128. The indexer has 32 heads × 128, top-k 512, keys mxfp4. KV sources are 2/8/14 (ratio 2) and 20 (ratio 1).

## 0. Verdict

**Capacity.** DPA gives each rank the paged KV of its own sequences only. That is 4× the tokens per rank (×4 at an even spread). Today every rank holds every sequence's latent and index keys (2760 B/token) plus its windows, tails and DSpark ring (6.9 MiB a sequence).

**Price (SGLang-style DPA, "DPA-W").** The owner of a row needs all 64 heads' q_b / o_a / o_b. That costs:
- +2.55 GiB a rank, about 275 expert units. By the VA slope this is ~+1.1 ms at c1 and ~+5.5 ms at c8.
- +2.74 GB of weight reads in every decode step, on every rank that holds a row: about +3..+8 ms a step.

**The exchange form ("DPA-X").** It keeps the weights sharded and moves q and o between ranks instead. It has no VRAM cost, but it adds 2 collectives a layer, i.e. +80 sync points a step.

**What DPA saves at decode is small.**
- The attention reads top-k 512 + window 128 entries a row whatever the context, so its replicated KV read does not grow with context.
- The replicated decode `index_score` grows with rows × context:
  - At c8 with short context it is 0.48 ms a step (profiled).
  - At 96 rows and 128k it is about 7 ms.

**Estimated net effect (step time, no build):**

| | 1 user | c8 (depth 0, 8 rows) | c16 steps of ≥ 96 rows at long context |
|---|---|---|---|
| DPA-W | +4..+9 ms on 14.9 ms (+27..+61 %) | +8..+13 ms on 46.6 ms (+17..+28 %) | ~break-even, because option B halves the collective bytes (§4) |
| DPA-X | +1.6..+2.4 ms (+11..+16 %) | +2.4..+8 ms | +7..+10 ms |

**Prefill.** One long prompt's attention would land on its owner: +55..+80 % per 4096-row step. Prefill therefore has to stay TP, with the sequence's KV replicated until its prefill ends (§2.5).

**Capacity against TP + NVFP4 latent.** The NVFP4 latent is exact and decodes at about bf16 speed with NVG. On today's 7.26 GiB pool, TP-nvfp4 already backs 15.7 full-length sessions a rank: 100 / 97 / 48 % of MAXSEQS 8 / 16 / 32. DPA saves VRAM over TP-nvfp4 only when the live KV exceeds 3.4 GiB a rank, i.e. 3.97M tokens (for example 32 × 124k or 16 × 248k).

**Recommendation.**
- Do not build DPA for the 2-16-user goal.
- Do the spin-off P0 now (§6a). It is exact and about 1 h of agent time, and gives the host tier 4× capacity.
- Keep DPA as the MAXSEQS-32 / 1M-context lever behind the gate P1 (§5).

## 1. Where the replication is today (TP4, per rank)

### 1.1 Caches

Every cache is declared identically on every rank (cpp:1659-1662). One block table serves every rank: rank 0 picks the blocks and the followers copy them (core/mem/kv.h:252-254, 390-393, 505-507).

| cache | group (log) | layers | bytes per rank | replicated |
|---|---|---|---|---|
| compressed latent | kv_compressed_r2 / _r1 (log:15448, 15450) | 2/8/14 at r2, 20 at r1 | 3 × 1024 / 2 + 1024 = **2560 B/token** bf16. nvfp4 = 288 B an entry, so 3 × 144 + 288 = **720 B/token** (cpp:1534) | ×4 |
| index keys | kv_index_r2 / _r1 (log:15447, 15449) | 2/8/14/20 | 3 × 80 / 2 + 80 = **200 B/token** mxfp4 (bf16 keys: 640) (cpp:1539, 2010-2025) | ×4 |
| paged total | | | **2760 B/token** (920 with nvfp4). The log's tier entry of 5.39 KiB a 2-token entry (log:15436) agrees | ×4 |
| window | kv_window (log:15444) | all 40 | 1024 B a key; 128 + 5 keys use 10 blocks of 16, so **6.25 MiB a sequence**. The pool is 348 blocks = 217.5 MiB: 8 sequences plus about 260 blocks for the 4160-row chunk (cpp:1665-1673, window + max_spec) | ×4 |
| compressor tails | kv_comp_tail_r2 (log:15446) | 2/8/14 | (2 + 5) rows × 1024 × f32 = 28 KiB a layer. A state is 84 KiB, ×2 states under speculation, so **168 KiB a sequence** (cpp:1687-1695; kept on every rank: cpp:1899) | ×4 |
| DSpark ring | kv_dspark (log:15451) | 3 drafter stages | 48 KiB a 16-token block; 10 blocks = **480 KiB a sequence** (cpp:2606-2639; attention ranks: cpp:1852) | ×4 |
| Engram | kv_engram_tokens (log:15445) | 1, 14 | **64 B a sequence** (conv state). The n-gram table is one host copy, 103 GiB pinned (log:15391-15393) | ×4, negligible |
| host tier | (log:15437) | | **21.56 KiB an entry = 4 × 5.39 KiB**. DS41 does not declare `rad_kv_group_replicated` | ×4 |
| checkpoints | (log:15435, 15454) | | 6.33 MiB a rank a slot (window + tails), 25.33 MiB a slot | ×4 |

### 1.2 Compute

All of the following refers to today's TP4 step.

**Replicated on every rank, all rows:**
- The stream ops: mhc_mix, collapse, norm and post (cpp:1866-1882, 2175-2190).
- The Engram GEMM (cpp:1838).
- q_a and its norm (cpp:1921, 1926).
- wkv, the window key's norm / rope / round trip, and its store (cpp:1941-1966).
- At the kv sources: the compressor (gemm_nt_f32 2hd × D), qsa_work / softmax_pool, norm, index-K, and the latent / index / tail stores (cpp:1970-2040).
- The indexer q / w (cpp:2042-2049).
- `index_score` at decode and verify: the row split exists only for steps of ≥ 2 runs of 256 rows at ≥ 4096 live positions ("decode and verify steps never see it", cpp:1415; log:14508).
- At prefill the split deals 256-row runs over the ranks and all-gathers the selections (cpp:2077-2112).

**Sharded:**
- q_b: 16 heads a rank (cpp:1923).
- Attention: 16 heads a rank, on the same replicated latent (MQA), so every rank reads the same selected entries (cpp:2117-2131).
- o rope / round trip.
- o_a: 2 of 8 groups a rank (cpp:2152-2162).
- o_b: a 2048-column K-slice a rank, followed by the exact all-reduce (cpp:2164-2174; g.wire_exact = 1, cpp:1371).
- MoE: EP with 96 experts a rank; the shared expert is split by width; the router is replicated (cpp:1339-1355, 2203-2207).
- Head: split by vocabulary (cpp:1358).

**Measured (profiled ms a step a rank):**
- c8: dense_gemm 18.2, collective 11.5, attention 4.2, mhc 2.75, indexer 0.58.
- P128 (4096-row step): dense 289, collective 305 (all_reduce 280 + all_gather 25), mhc 118, indexer 110, attention 75.

## 2. Exact design (DPA-W, the reference)

### 2.1 Ownership and rows

**Owner.** At admission each sequence gets an owner rank o(s): the rank with the fewest live paged tokens, ties broken by fewest rows. The owner holds the sequence's latent, index keys, window, tails, DSpark ring and Engram state. The other ranks hold nothing for it.

**Core change.** A new group mark, "owned", next to `rad_kv_group_zero` and `rad_kv_group_replicated` (abi/rad_builder.h:602, 613). It needs:
- per-rank allocators;
- per-rank block tables and slot maps. RadBatch already gets a per-rank view of each group (core/sched/batch.cpp:853-870), but today all of them point at one staged table.

**Row order.** The scheduler orders decode rows by owner, and the prefill sequences by owner. Each rank's rows are then contiguous runs. The plugin issues over those runs with step_bi's row-slice helpers R / P / segcu (cpp:3902-3940), the same machinery lever BI uses.

### 2.2 Per layer: option A, streams replicated (minimal)

| part | rows | what runs |
|---|---|---|
| before attention | all rows, every rank (as today) | Engram, mhc_mix_a, collapse + norm → xn |
| q path | owner rows only | q_a → norm. q_b as **4 launches of today's shard shape** [8192 × 1280]: identical per element by construction. One 64-head launch only if a bicheck shows equal bytes. Then rope over 64 heads |
| KV side | owner rows only | wkv → norm / rope / round trip → window store. At the sources: compressor → tail → pool → norm → index-K → latent / index stores, into the owner's caches only |
| indexer | owner rows only | iq / iw → `index_score` over the owner's keys → top-k. The selections stay local, so no `sel_gather` at decode |
| attention | owner rows only | 64 heads (sinks 64) → o rope / round trip |
| o_a | owner rows only | 8 group GEMVs: today's shapes, 4× the count |
| o_b | owner rows only | **4 K-slice launches of today's [5120 × 2048]** → four bf16 partials P0..P3 → a local `sum4`: `acc = 0.0f; acc += P0; ... acc += P3; one rounding`. This is exactly the exact all-reduce's arithmetic: ascending rank, f32, one rounding at the store (libr4d/r4d_ar_oneshot_Nrank_exact.hip:6-8, 24-40) |
| rows the rank does not own | non-owners | write zeros into attn_out. This is the attention-zero rank's existing path (cpp:1884-1918) |
| collective | all ranks | **today's all-reduce, unchanged** (cpp:2168-2174). A DP row sums to 0 + 0 + v + 0 + 0 = v bit for bit in f32. v is never −0, because the owner's sum starts at +0.0f |
| after attention | all rows, every rank (as today) | mhc_post, mix_f, collapse-norm, MoE (EP4) and its all-reduce, post_f. The DSpark drafter stays TP, taking its taps from the replicated streams. The vocab head is unchanged |

A later step replaces that all-reduce with an all-gather of the owner rows. Each rank then receives 0.75·M × 10 KiB instead of 3·M (one-shot) or 1.5·M (two-shot) × 10 KiB.

### 2.3 Option B: DP for everything except the MoE

- The streams live on the owner only, so the attention needs no collective.
- Before the MoE: an all-gather of act.x (D a row).
- After the MoE: a reduce-scatter. It is the first half of the two-shot exact all-reduce, with the same f32 ascending order. Each rank keeps its own rows' sums.
- Per layer this moves **one all-reduce's bytes instead of two**, with the same collective count.
- It needs:
  - a reduce-scatter form of `moe_gather_all_reduce` (ARGATHER, ≤ 48 rows);
  - an all-gather of the DSpark taps;
  - an all-gather of the logit rows' final hidden for the vocab-split head.
- It also removes the replicated mhc and Engram work per row. That work is prefill-relevant; at decode it is launch-bound.

### 2.4 DPA-X (weights stay sharded)

1. q_b runs on all rows, per rank, as today.
2. An all-to-all moves q: each owned row receives 3 × 16 KiB.
3. The owner runs the 64-head attention.
4. o, after its fp8 round trip, goes back exactly as e4m3 codes + ue8m0 scales: 33 KiB a row instead of 64 KiB bf16. This is lossless because e4m3's 3 mantissa bits fit bf16.
5. o_a / o_b / the all-reduce run exactly as today.

It has no VRAM cost and no extra weight reads, but 2 more collectives a layer. It cannot serve prefill: a 4096-row chunk's q is 192 MiB a layer into the owner.

### 2.5 Prefill and the indexer

**Prefill concentration.** With the keys owned, the prefill row split is impossible, because only the owner has the keys. One prompt's attention block then runs on one rank. Using P128 numbers:
- q_b / o_a / o_b, about 150 of the 289 ms dense: ×4 on the owner;
- attention, 75 ms: ×4;
- together **+675 ms on a 1224 ms step (+55 %)**;
- **+330 ms more** if the indexer is not split.

**Fix (phase P4): prefill in TP mode.**
- The prefilling sequence's KV is written on all ranks, with the head split exactly as today.
- When its prefill ends, the 3 non-owners free their copies.
- TP rows and DP rows share the one existing all-reduce in a mixed step: TP rows contribute partials, DP rows contribute v on the owner and zeros elsewhere.
- PSK's cone cut is per sequence and unchanged.

**Variant DPA-L.** Keep the index keys replicated (200 B/token) and own only the latent and window. The indexer, including the prefill split, then stays exactly as today. Capacity per token is 2560 / 4 + 200 = 840 B (bf16) or 380 B (nvfp4), against 2760 / 920 today (×3.3 / ×2.4).

### 2.6 Exactness, row by row

**Holds by construction:**
- **Shard shapes.** q_b, o_a and o_b run as today's launches, with today's weights, M and K (DPA-W). DPA-X does not change them at all.
- **The o_b sum.** `sum4` copies the exact all-reduce's order. Adding zeros in the unchanged all-reduce is exact, and so is an all-gather.
- **Row-invariant ops.** Every op the BI list calls M-invariant is byte-equal on the owner: gemm_nt_f32 (compressor), rmsnorm_bf16, mhc_mix / collapse / post, rope / quant round trip, index_topk / candidates, the compressor chain, the stores, all_gather, and the exact all-reduce.

**Needs a check: attention with 64 heads in one launch.** The kernel runs one workgroup per (d half, head group, row), and each head's scores, max, exp and sums run in its own order (libr4d/r4d_ds41_sparse.hip:15-19). Bytes should therefore match 4 × 16-head launches. This is a one-card bicheck. The fallback is 4 launches, but that reads the KV 4× again.

**Where bytes can differ: the row-count bands.**

| op | decode form | other form |
|---|---|---|
| gemm_nt_ggml, attn_sparse_sink, index_score | ≤ 48 rows | WMMA above (kBiBand = 48, cpp:1137; kWmRows, r4d_ds41_sparse.hip:2301) |
| the KV-side fused norm, n = 512 | below 73 rows | above |
| option B only: mhc_(post_)collapse_rmsnorm, n = 5120 | below 33 rows | 33-120, and 121 up (kBiNorm = 32, cpp:1138) |

The owner launches M_k ≤ M rows. Consequences:
1. **Alone (one sequence):** M_k = M, so bytes equal today's.
2. **Under ds41_bi:** today already issues decode rows at ≤ 48 a launch (≤ 32 a fused norm), and each prefill sequence at its own rows (cpp:3806-3841). The owner's segments are the same classes, so bytes equal today's BI output.
3. **Under ds41_bi 0 with a step over 48 rows:** today's row ran WMMA, while the owner may now run the decode form. The bytes then differ from today's loaded output but equal the row's alone bytes. That is a BI improvement, not a regression.

**New dependencies.** The owner is fixed per sequence at admission, so a row's bytes never depend on which rank owns it. Every rank runs the same kernels.

## 3. Capacity (per rank; 524288-token sessions; per-sequence state included)

Pool = today's 7.26 GiB a rank (log:15474). The tokens column is distinct context tokens in the whole engine.

| layout | tokens | sessions | backing at MAXSEQS 8 / 16 / 32 | VRAM a rank for 100 % backing at 8 / 16 / 32 |
|---|---|---|---|---|
| TP4 bf16 (today) | 2.74M | 5.2 | 65 / 32 / 16 % | 11.0 / 21.8 / 43.5 GiB |
| TP4 nvfp4 | 8.22M | 15.7 | 100 / 97 / 48 % | 3.8 / 7.5 / 14.8 |
| DPA-W bf16, extra weights taken from the experts | 11.0M | 21.0 | 100 / 100 / 65 % | 5.4 / 8.1 / 13.6 (incl. +2.55) |
| DPA-W bf16, extra weights taken from the pool (4.71 GiB left) | 7.0M | 13.4 | 100 / 84 / 42 % | — |
| DPA-W nvfp4, extra weights from the experts | 33.1M | 63 | 100 / 100 / 100 % | 3.6 / 4.5 / 6.4 |
| DPA-X nvfp4 (no extra weights) | 33.1M | 63 | 100 / 100 / 100 % | 0.9 / 1.8 / 3.6 + 0.16 window chunk |

**Break-even.** DPA-W saves 3/4 of the live KV and costs 2.55 GiB. It pays above 3.4 GiB of live KV a rank: 1.32M tokens with the bf16 latent, 3.97M with nvfp4.

**Idle sessions.** The host tier carries idle sessions already, so VRAM needs to hold only the active ones. Claude Code at 10-16 users with 50-200k contexts is 0.5-3.2M live tokens, which TP-nvfp4 holds with room to spare.

## 4. Cost per step (per rank; floor 10.7 GB/s each way; rank 1 = 83:00 is the slowest link)

| item | today | DPA-W option A | DPA-W option B | DPA-X |
|---|---|---|---|---|
| attention weights read a step | q_b + o_a + o_b shards 0.87 GiB (+ q_a, wkv, replicated) | **+2.74 GB → +3..+8 ms**. 568 GB/s for big GEMVs, ~150 GB/s for o_a's 3.4 MB groups (22.5 us each) unless grouped | same | 0 |
| VRAM | | −2.55 GiB = −275 units (VA slope: c1 +1.1, c4 +3.6, c8 +5.5 ms) | same | 0 |
| collectives, c8 (8 rows) | 2 ARs a layer. Latency- and skew-bound: 146 us a call profiled; bytes floor 46 us a layer | same | AG + RS: same count, 1/4 of the one-shot bytes, so ~0 at this size | +2 all-to-alls a layer: +80 sync points ≈ +2.4..+8 ms |
| collectives, 96-row steps | two-shot 2 × 1.44 MiB a layer = 11.0 ms a step at the floor | AG: −2.75 ms | **−5.5 ms** | +1.71 MiB a layer ≈ +6.7 ms |
| `index_score` at decode | every rank scores every row. Plane writes 13 MiB a row a step whatever the context (−inf up to width = max_ctx / r + 1; cpp:2053, r4d_ds41_sparse.hip:406-408), plus keys L × 200 B a row | /4: c8 short −0.36 ms profiled; 96 rows at 128k ≈ −5.5 ms | same | same |
| attention core | head groups × rows read the KV | about the same (64 heads × M/4 rows = 16 × M) | same | same |

**Load imbalance.**
- Decode under DPA-W is weight-bound, so the skew comes only from rows (D7 depth 1..6 a sequence) and context. With 2 sequences a rank at c8 and contexts of 400k vs 20k, the index-key reads give about 1 ms of skew a step. Each skew is paid at the next all-reduce or all-gather.
- At c1-c3, ranks without rows idle while the owner reads the full weights on the critical path.
- Prefill concentrates on one rank unless P4 is built.

**BI.** BI is unaffected (§2.6). The new rule: when MAXSEQS exceeds 8, steps with more than 48 decode rows are reached more often, and per-rank segments must stay ≤ 48. This is the standing BI requirement for MAXSEQS > 8.

**PSK / CR / prefix cache.**
- PSK is unchanged: the cut is per sequence, and under P4 the prefill runs in TP mode.
- A prefix-cache entry lives on its owner. Admission prefers the rank holding the longest hit.
- Otherwise the prefix is copied peer to peer, which is exact:
  - a 30k shared Claude Code prefix is 83 MB, about 8 ms;
  - its checkpoint (window + tails) is 6.33 MiB.
- Under P4, a hit during prefill needs the prefix on all ranks: about 23 ms over the peers, or the host tier restored to all 4 cards in parallel.
- PSK-CR replays on the owner only.
- The host tier becomes per owner, and a restore can land on any rank, which makes it the natural way to rebalance.

## 5. Build phases (agent wall-clock)

| phase | content | effort |
|---|---|---|
| P0 (§6a, independent) | `rad_kv_group_replicated` on kv_compressed_r* / kv_index_r* | ~30 min build + ~30-45 min identity / RADIANCE_DEBUG_TIER_VERIFY item |
| **P1 gate: the minimal measurable step, no build** | One card, ~15 min. **rad-kbench** `gemm_nt_ggml` at the full vs shard shapes: q_b 32768 vs 8192 × 1280, o_b 5120 × 8192 vs 2048, o_a 8 vs 2 groups of 1024 × 4096. M = 1, 2, 4, 8, 16, 48. Plus a bicheck of attn_sparse_sink with 64 heads in one launch vs 4 × 16 on the same rows. Rule: if (full − shard) × 40 layers ≥ 3 ms a step, DPA-W cannot pay below about 64 rows a step × long context. | ~15 min GPU |
| P2 | Plugin only, on the replicated cache with the core unchanged. RADIANCE_DS41_DPA: declare all 4 shards per rank; deal rows by sequence % 4; issue the owner-row attention block (§2.2); a libr4d `sum4` with an emulator test; zeros into the unchanged all-reduce. Measures identity (alone / c4 / BI) and speed at c1 / c4 / c8 / prefill. | ~5-7 h + ~2 h queue |
| P3 | Core ownership: the owned group mark, per-rank allocators and tables, admission by rank, prefix-cache affinity + peer copy, tier and checkpoints per owner, preemption per rank, vram_budget / stats. The largest and riskiest part. | ~2-3 days |
| P4 | TP-mode prefill with temporary replication, then demotion to owned | ~1 day |
| P5 | Option B collectives: AG / RS around the MoE, a reduce-scatter form of moe_gather_all_reduce, gathers for taps and the head | ~1-1.5 days |

## 6. Spin-offs

**a. P0, tier replication flag. Exact; capacity of the host tier ×4.**
- DS41 does not declare its latent or index groups replicated. GLM-5 Next does (arch/glm5next/glm5next.cpp:560-561). spec.md:1225 says that is the intent for "an MLA latent cache computed from a replicated projection".
- Today the host tier stores 4 copies: 21.56 KiB an entry (log:15437).
- With the flag, an entry is 5.39 KiB. The 84 GiB tier then holds **32.7M tokens instead of 8.17M**, and copies 1/4 of the bytes on spill and restore.
- Declare it only when no rank is attention-zero (aw == world). Check the claim with RADIANCE_DEBUG_TIER_VERIFY.

**b. DPA-I, decode index row split.**
- Deal one decode `index_score` launch's rows over the 4 ranks. Rows stay ≤ 48 a rank, so they keep the decode band, which is per row: grid y = row (r4d_ds41_sparse.hip:393-414).
- All-gather the selections through the existing `sel_gather` (cpp:2077-2081): 2 KiB a row a layer, 8 all-gathers a step.
- It takes DPA's real decode gain with no weight cost: about −5 ms at 96 rows / 128k. At 8 rows the 8 sync points cost more than they save.
- Gate it by rows × live positions, like split_min_live.
- The −inf plane writes (13 MiB a row a step) are IX's known finding and are fixed there independently.

**c. Not exact or not designed here.** Another capacity path: the latent is read sparsely (512 entries a row a layer), so a host-resident latent with a VRAM entry cache is possible in principle. That is a separate study.
