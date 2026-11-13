# Qwen3.8 Flash-Next: official architecture vs our engine (audit 12.11.2026 ~17:30 FI)

The audit agent read the official sources against the fn4-x3 line: radiance-x3 at a8bb3add, plus the SP-hc WIP branch fn4-x3-sphc
ef791455. It only read code: no build, no GPU. Code paths:
- "mtp" = arch/common/rad_block_mtp_hc.h
- "qsa" = arch/common/rad_qsa.h
- "arch" = arch/qwen4exp_fp8/qwen4exp_fp8.cpp
- "hc" = arch/common/rad_block_hc.h

Every gain below is an ESTIMATE from LEVERS calibration (launch gap +2.9 / 3.3 / 5.1 us at c4 / c8 / c16; per-op profiles 120 / 147 /
151). None has been measured.

## Sources
- [TR] Technical report "On the Design of Qwen3.8-Next Architecture", 26.08.2026:
  https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf (repo HEAD 69885871). Line refs = pdftotext -layout.
- [MC] Model card, config and generation config: https://huggingface.co/Qwen/Qwen3.8-Flash-Next (sha 812d96e1). Blog:
  https://qwen.ai/blog?id=qwen3.8-flash-next (not fetched; the card and the report cover it).
- [HF] transformers main 9b6e5e20: src/transformers/models/qwen4_exp/modeling_qwen4_exp.py. This is the reference forward. It has no
  MTP: `_keys_to_ignore_on_load_unexpected = ["^mtp.*"]`.
- [VL] vLLM main f0787a9a: vllm/models/qwen4_exp/{common,amd,nvidia}/*, csrc/qwen4_exp/hc_up_gate_mix.cu and
  tests/models/qwen4_exp/*. Recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- [SG] SGLang main d193c34a: python/sglang/srt/models/qwen4_exp{,_mtp}.py and srt/layers/attention/qsa/*. Cookbook:
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- [LC] llama.cpp master 3ba1ce0e: src/models/qwen4exp.cpp and conversion/qwen4exp.py.
- Local copies of all of these, untrusted downloads, are under the session scratchpad in dl-gh/, dl-hf/, dl-ref/, dl-vllm/, dl-sgl/ and
  dl-llama/.

## Corrections to what we assumed
- **No DeepSeek-style cone in the trunk (negative result).** All 48 layers are causal decoders. Every prompt row's output feeds later
  layers' GDN state or QSA K/V. The MTP head also reads the trunk's pre-final-mixer wide stream at EVERY prompt row ([VL]
  amd/mtp.py:12 "scheme A"; ours mtp:8-12) to build its own K/V. So no layer can drop prompt rows.
  - The only cone is inside the one-layer MTP head. There only K/V and index keys need all rows; the attention, MoE and mixer need the
    drafting row only (lever 2b).
- **"hc" is officially "Gated Residual" (GR) and has no branch-mixing operator** ([TR] §2.2, Eq. 30-34; [HF] 997-1032).
  - The read is: per-branch RMSNorm (1 + w) → the low-rank gate σ(W_u · silu(W_d · N / 4)) → mean over the 4 branches.
  - The write gains are 2σ(W_inject · N / 4). They are computed in the READ, from the same normed stream. hc_read already does this.
  - The official inference design is "read and write each fused into a single kernel ... the widened stream is traversed once per
    block in each direction" ([TR] l.984-986).
  - vLLM's AMD path fuses the write into the next read's norm (amd/hyperconnection.py:152-169 `hc_combine_norm`). That is our W1b
    (hc_readv 11 + HC_SS).
- **QSA has no dense mode officially.** Each query attends to the top min(512, complete blocks) blocks of 4 tokens plus the tokens of
  its incomplete tail block ([TR] Eq. 16 / 19; [HF] 670-773).
  - Our dense path below 2051 visible tokens is an exact equivalence (every block is selected), not an official rule. Our engine's rule
    matches the reference, so this is not a bug.
  - A query whose own block just completed (pos % 4 == 3) has no guaranteed self-attention: its block competes in the top-k. docs/QSA.md
    already encodes this ("partial page only when (pos+1) % 4 != 0").
  - The budget is 2048 tokens = 512 blocks, plus up to 3 tail tokens.
- **MTP, official form** ([MC], [TR] Fig. 1 and §2.1.2, [VL] amd/mtp.py, [SG] qwen4_exp_mtp.py):
  - One QSA + MoE layer with two GR connections and its own final mixer, run recursively and "trained with multi-steps".
  - Official serving depth is 3: vLLM recipe num_speculative_tokens 3; SGLang AMD speculative-num-steps 3 / draft tokens 4. This matches
    ours.
  - Two official details differ from ours:
    - **pre_fc_norm_hidden is a FLAT RMSNorm over the whole 10240 row** (one rsqrt) in vLLM ([VL] amd/mtp.py:250 + :326; nvidia/mtp.py:259
      + :336) and in SGLang (qwen4_exp_mtp.py:84 + :112). Ours is GROUPED, one rsqrt per 2560 sub-stream: mtp:189 `ngroup = hc.hc`.
      docs/OPS.md `mtp_enter` calls ngroup "the one thing the checkpoint does not say" and records no A/B. Only llama.cpp is grouped
      (qwen4exp.cpp:629).
    - **Top-k reuse across draft steps.** The QSA selection is made once per draft round (on the "draft extend" = our history pass) and
      reused by the later draft steps:
      - [TR] l.502-503 and Table 4: accept length 4.07 vs 4.06, "no significant change".
      - [VL] amd/mtp.py:281-293 `set_skip_topk` / `compact_topk_indices`.
      - [SG] qwen4_exp.py:1979-1983: "the indexer never runs inside the decode graph".
      - Ours runs the full indexer in every round (mtp:504).
- **PLE (n-gram embedding).** Config `ple_layer_ids [2]` is 1-based ([HF] 1264: `layer_idx + 1 in ple_layer_ids`), i.e. our layer
  index 1. Ours matches (arch:1045, `- 1`).
  - Bigram + trigram, 8 hash heads each.
  - The token history is cut at EOS 248044 ([HF] 1109-1123 `_shift_right_ignore_eos`). Our PleLayer carries `eos`; this audit did not
    re-verify the segment reset in the kernel.
- **MoE.** Softmax over all 512 → top-10 → renormalised (`norm_topk_prob` True, [HF] config default) + 1 shared expert of 640 with a
  sigmoid gate. There are no routing groups, no capacity / token dropping and no bias correction ([HF] 954-994).
  - Official serving is TP4, or "TEP" (TP + expert parallel) per the vLLM recipe ("consistently faster than plain TP"). Our EXL3 EP4 is
    that direction.
- **GDN.** The state is f32 (`mamba_ssm_dtype float32`) and the chunk is 64 ([HF] torch_chunk_gated_delta_rule); ours matches.
  - Spec-decode state: SGLang's `--enable-linear-replayssm-spec` is our `anchor` form (one committed state plus replay of the accepted
    prefix; rad_block_gdn.h:229-242).
- **YaRN.** Native context is 262144. The card recommends static YaRN factor 2 for ~524k and 4 for 1M, and only when needed, because
  static YaRN "potentially" hurts short texts. This is a quality note, not a speed lever.
- **What the official stacks quantise that we do not** (none of it is exact; see the last section): an fp8 indexer key cache (vLLM recipe
  `indexer_kv_dtype fp8`), a bf16 GDN state (SGLang AMD), Octave k3v4 KV (vLLM ROCm), and the FP8 residual stream ([TR] l.981-984: a
  report claim only, implemented in neither vLLM nor SGLang).

## Exact levers not yet exploited, ranked

### 1. MTP-FLAT: the MTP entry norm as the official flat RMSNorm (ngroup 1)
- **Official basis:** [VL] amd and nvidia mtp.py and [SG] qwen4_exp_mtp.py all normalise the 10240 row with one rsqrt.
- **Exactness:** drafts only. The target bytes are unchanged by construction; only acceptance moves.
- **Estimated gain:** unknown until measured, and it may be 0 or negative (then keep grouped).
  - At depth 3, +1 pp of per-position acceptance ≈ +1.5 % tok/step (α 0.67-0.81).
  - SGLang reports accept length 3.5-3.7 of 4 (non-thinking) and ~2.5 (thinking). Our mb acceptance of ~0.67 sits inside that band, so
    it does not decide the question.
  - Affects c1, c4 and c16 alike; prefill 0.
- **Build:** ~0.5-1 h. Add a start-time env for ngroup; libr4d `mtp_enter` already takes `ngroup` (OPS.md: "one declaration away").
  Then an A B B A start A/B: mb c1 / c4 / c16 tok/s + acceptance, and the target copy checked IDENTICAL.
- **Where:** mtp:189 (`ngroup = hc.hc`) and the `mtp_enter` declare.
- **Status vs LEVERS:** not in LEVERS.

### 2. MTP-QSA: the official index reuse (a) plus the cone inside the MTP layer (b)
- **(a) Reuse.** Draft rounds 2..d skip the indexer entirely and reuse round 1's per-row `sel` table + `seqused`. Round 1 is the history
  pass's drafting rows, compacted like `op_dsel` does for the stream. This skips the indexer projection GEMM, qprep, qsa_work,
  qsa_block_key, qsa_tail_store, qsa_score and qsa_select, about 7 launches a round.
  - The official semantics attend to exactly round 1's token set: vLLM freezes the rows and the valid-count column together.
  - Optional variant: extend `seqused` by the round index so a round also sees its own new tail tokens. This departs from official and is
    an acceptance A/B only.
  - The skipped tail and block-key stores are not needed for drafts: the next history pass rewrites the committed positions, and the
    ring is keyed by position.
- **(b) Cone.** In a history pass with drafting rows, store K/V + block keys + tail for all T rows (`qsa.store` + `attn.store_kv`, as the
  non-final-chunk branch at mtp:493-500 already does). Then run qsa_score/select, the attention, o_proj and the attention write only on
  the n_draft_out rows.
  - Today qsa.step + attn.step + hc_attn.write run on all T rows (mtp:504-507), and only the MoE onward is cut to the drafting rows
    (mtp:510-521).
  - At decode T is each sequence's committed rows (a + 1 ≈ 3-3.8 a sequence at depth 3) against 1 drafting row.
- **Exactness:** drafts only. The target is unchanged by construction. Draft bits change only via (a), which is the official behaviour,
  and via row-count-dependent kernel forms on the drafting rows in (b).
- **Estimated gain:**
  - c16 +0.6..1.5 %. Per step: 2 rounds x (7 launches x 5.1 us + qsa_score over 2-6 MB of block keys a sequence + select) ≈ 0.2-0.4 ms,
    plus (b) ~40 fewer MTP attention rows ≈ 0.1-0.15 ms, against 37 ms (exact wire). It grows with context.
  - c4 +0.4..1 %; c1 +0.4..0.8 % (the rounds are latency-bound).
  - Prefill ≈ 0. On a short prompt's last chunk (b) saves one QSA layer's attention over up to 4096 rows: ≤ ~1 % below ~4k tokens.
- **Build:** 3-5 h.
- **Where:** mtp `step()` (history branch mtp:489-512; the rounds path at mtp:504); qsa (a select-rows entry and a carry of `sel` /
  `seqused` / `nc` for the drafting rows).
- **Status vs LEVERS:** not in LEVERS.

### 3. QSA-PREP: the official fused QSA prepare, one launch per QSA layer
- **What it fuses:** indexer q-norm + RoPE, the raw-key ring update, block-key pooling + norm + RoPE + store, the main attention's
  q/k-norm + RoPE + gate split, and the K/V cache write. Ours issues `op_qprep`, `op_work`, `op_bkey`, `op_tail` (qsa:292-343) plus the
  attention block's fused qk / kv op: 5 launches.
- **Official basis:** [VL] amd/ops/qsa_prepare.py:550-586, enabled by amd/qsa.py:512 and nvidia/qsa.py:430-441. The comment reads "One
  launch does the indexer prepare, the main QK-norm/RoPE/gate and the main K/V cache write".
- **Ordering without a grid wait:** vLLM reads a completing block's keys by "two masked loads": this step's keys from the input, older
  ones from the ring. It never reads what the launch writes, so our "qsa_work before qsa_tail_store" ordering needs no grid wait.
- **Exactness:** same per-element arithmetic (vLLM: "Round to BF16 like the unfused path"); a byte selftest against today's five ops.
  Every op is per row / per block, so the fusion is batch-invariant.
- **Estimated gain:** ~4 launches x 13-15 QSA calls a step.
  - c16 -0.2..-0.25 ms (+0.5..0.7 %).
  - c4 +0.6..0.8 %, c1 +0.7..1 %.
  - Prefill ~+0.1 %.
  - It overlaps with lever 2(a) on the MTP rounds.
- **Build:** 6-10 h: libr4d kernel + libref + selftest + arch wiring.
- **Where:** qsa `store_keys` / `step`; rad_block_attn_gated_fp8.h (fused qk / kv store); a new libr4d r4d_qsa_prepare.hip.
- **Status vs LEVERS:** supersedes D-1b ("qsa_work + qsa_block_key + qsa_tail_store into qsa_select", ~-0.2 ms c16, IDEA).

### 4. MIX-OUT: the final mixer only on the output rows
- **Today:** `bi_read_pieces(c, m, m.mixer, T)` runs over all T rows (arch:2246). The MTP head reads `h`, not the mixer output, so only
  `out_ids` rows are needed. vLLM also mixes all rows (amd/model.py:580); this is our own lever.
- **Exactness:** only if the output rows are read with the piece's own form. A 1-row read takes another K split
  (hc_down_kchunk by row tiles), so the read must be pinned to the piece's kc and checked with --bicheck. Without the pin the first
  token's logits bits change.
- **Estimated gain:** prefill +0.2..0.3 % at 64k-128k (1 of 97 hc_read calls a piece). Decode 0: every verify row is an output row.
- **Build:** 2-3 h.
- **Where:** arch:2246.
- **Status vs LEVERS:** not in LEVERS.

### 5. Already in progress, with official confirmation (no new build)
- **SP-hc on the EXACT wire (largest).**
  - Official: vLLM's `enable_hc_sp` ([VL] common/hyperconnection.py:41; amd/model.py:312-343): reduce-scatter → GR write + read on T/4
    rows → all-gather of the bf16 block input. The all-gather is a byte copy on the exact wire.
  - Our WIP fn4-x3-sphc ef791455 already has the exact raw reduce-scatter (ascending-rank fp32 fold = the exact all-reduce's order).
  - The remaining exactness hinge, as that commit notes: the quarter's hc_read at 512 rows takes `hc_down_kchunk` at 4 row tiles,
    today's at 16. PF2 itself runs from kPf2MinRows 512 (libr4d r4d_hc_bf16.hip:5109), so the fix is to pin kc to the full piece's row
    tiles.
  - Estimated prefill TTFT -10..-25 % at 64k-128k. The ceiling is the exact-wire all-reduce lane: the same bytes as today's two-shot,
    ~4.9 s at 64k / ~9.7 s at 128k at ~9 GB/s effective. Decode 0.
  - LEVERS line 110 marks it "not exact" only because of the ti8 wire. On the exact wire the line now runs on, that reason no longer
    applies.
- **P2x (one routed pass over both pieces).**
  - Byte-identical (162).
  - Official chunked-prefill sizes (SGLang 16384) point the same way.
  - The EXL3 port is blocked by the pair-gather operand-sizing bug (189).
  - Est. -2..-3.8 % prefill.

### Not a lever after reading
- **Indexer projection fused into the q/k/v GEMM** ([VL] nvidia/qsa.py:448 `fuse_indexer_projection`). vLLM fuses only when both use the
  same quant method. Our indexer projection is bf16 and the trunk is int8, so a joint GEMM means an int8 indexer, which changes the
  selection: not exact.
- **The P-2 q-tiled union walk** has no official analog: vLLM and SGLang prefill sparse attention is per query row. Keep it as our own
  lever.

## Cross-check: already done (official = ours)
- **GR read** as one op: norm + down + inject (PF2 `fold`, ncol 324 = vLLM's merged down+inject GEMM, padded to 336) + up + mix +
  codes.
- **Write fused with the next read's norm:** W1b.
- **Decode MoE gather + shared gate + all-reduce + write fused:** `ar_gather_hc_write`. This matches SGLang's TP4 qwen4_decode finalize.
- **PLE:** hashed and host-gathered ahead of layer 0 so the gather overlaps the first layer ([TR] l.1031: "allowing host-memory
  prefetching to overlap with the computation of the first layer").
- **QSA:**
  - Block keys are computed once and cached (the reference recomputes them).
  - Decode scoring uses 4-row tiles (official next_n 4).
  - Dense below the bound is an exact equivalence.
  - Our rule = [HF] (q-norm then RoPE; mean in f32 → bf16 → k-norm → RoPE at the block start; relu-sum / √128; top min(512, nc); tail
    appended).
- **MTP:**
  - Depth 3.
  - It reads the pre-mixer wide stream.
  - Non-final prefill chunks only store K/V, block keys and the tail (mtp:493-500).
  - Its own grouped GR norms (vLLM `hc_per_branch_norm=True` for the MTP mixer).
- **GDN:** anchor + replay state.
- **KV:** fp8; YaRN 2 at 524k.
- **Experts:** EP4 on the EXL3 line (official "TEP").

## Not exact / do not copy
- **FP8 residual stream** ([TR] l.981-984, "almost no loss"; implemented in neither vLLM nor SGLang).
- **GR sparse read (top-2 branches)** (rejected by Qwen after post-training, [TR] l.973-980).
- **IndexShare** (top-k shared across QSA layers; rejected in [TR] Fig. 5a).
- **fp8 indexer key cache** (vLLM recipe `--attention-config.indexer_kv_dtype fp8`). It changes the block-key bytes and therefore the
  selection.
- **Octave KV k3v4 / k3v3** (vLLM ROCm `kv_cache_dtype octave_*`).
- **bf16 GDN state** (SGLang AMD `--mamba-ssm-dtype bfloat16`; the official dtype is f32).
- **MXFP4 / NVFP4 checkpoints, fp8 attention, fp8 PLE "auto-switch".** These are format changes. Our fixed-scale E4M3 PLE table is
  already our reference.
- **EPLB / redundant or heat-placed experts** (vLLM `enable_eplb`, our E4). They change the per-rank partial-sum grouping.
- **int8 indexer projection** (needed for vLLM-style qkv + indexer GEMM fusion).
- **SP-hc on the lossy ti8 wire** (the all-gather would carry block inputs lossily). Only the exact-wire form qualifies.
- **"Short softmax" routing** (softmax over the top-10 logits only). Mathematically equal, but the bits differ from
  softmax-512-then-renormalise. The gain is negligible.
- **Skipping the MTP history pass over early prompt rows.** Output-exact, but the drafter's K/V gets holes, so acceptance drops; not
  recommended.
- **Dense attention above 2051 visible tokens.** It is not the trained model (docs/QSA.md already refuses it).

## Correctness notes from the reading
- **Matches:**
  - The QSA indexer and its tail rule.
  - GR read / write (two separate `/ hc`, mean, 2σ inject).
  - PLE position (layer index 1) and conv (kernel 4, dilation 3, state 9).
  - MTP indexing (head i reads h(i) and emb(t_{i+1})).
  - MoE routing, assuming the router renormalises (not re-read here).
- **Mismatch:** the MTP pre-fc norm (lever 1). Drafts only.
- **Not verified here:** the PLE EOS segment reset (`_shift_right_ignore_eos`) in our ids kernel. A prompt with token 248044 mid-sequence
  would show it.
