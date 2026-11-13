# DeepSeek V4.1 Flash: official architecture vs our engine (audit 12.11.2026 ~00:50 FI)

Audit agent read the official sources against ds41-int7-pskglue 5093ce9. The coordinator did a second reading of the
layer-20 claim: report lines 135-137 and 293 / 1094; the cpp:1140-1146 + psk_declare `c.cand < first` check.
"cpp" = arch/deepseek41/deepseek41.cpp on that branch.

## Sources
- [R] Tech report: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf (arXiv 2609.19969)
- [MC] Model cards: deepseek-ai/DeepSeek-V4.1-Flash and -1017
- [MP] / [K] / [E]: inference/model.py, kernel.py, engram.py in the HF repo -1017. The local copy
  /home/mikoskinen/models/ds41-release/hf-src/DeepSeek-V4.1-Flash-1017 is blob-identical to HF HEAD 08c8657f;
  the only change upstream is chat_template.jinja, added 10.11.
- [DI] DeepInfer: https://github.com/deepseek-ai/DeepInfer (hugging_face/config.py, v4_modules/{yoco,attn}.py)
- [VB] vLLM blog, 2026-10-07: deepseek-v41-flash
- [SG] SGLang cookbook: DeepSeek-V4_1

## Corrections to what we assumed
- **Causal Encoder-Decoder.** The encoder is layers 0-19 and the decoder is layers 20-39. Decoder global KV
  (latent + index K) is projected from the final encoder hidden state, i.e. layer 20's attention input
  ([R] §2.2, lines 135-137, 293, 1094). So layer 20 also belongs to the decoder: only its global-KV projection
  needs every row. PSK's "every row through the last source" is one layer too conservative.
- Modes: Full at 2/8/14/20; Reindex at 24/28/32/36 (they score layer 20's keys with their own Q); Reuse everywhere
  else with ratio > 0. compress_ratios has 43 entries: layers 0-1 and the DSpark stages are 0 (SWA only).
- "PLE layers" = the Engram layers [1, 14]. GGUF ple.layers maps to engram_layer_ids (cpp:216).
- DSpark taps = the hc-mean of the INPUT of layers 37-39 ([MP]:1267-1269).
- Size: 8B prefill / 16B decode active; 552B backbone + 196B Engram.

## Exact levers not yet exploited (sizes are estimates), in build order
1. **PSK-L20: split layer 20.** On all rows: collapse, norm, compressor, index-K/latent stores. On the cone
   C_20 = 2592 rows only: q path, indexer scoring, candidates, attention, MoE, window store.
   - Size: -1.5..-2.5 s at 136k cold (K4 33.25 -> ~31 s). Also lighter mixed steps, and layer 20's experts no
     longer staged per cut chunk.
   - Why exact: the same cone rule. Rows outside C_20 reach later layers only through latent / index K, which
     still come from layer 20's input on every row.
   - Needs: keep cf7bbc7's deal fix, and relax psk_declare's `c.cand < first` (cpp:3413).
2. **PSK-DS: DSpark prefill context pass over each prefill sequence's last window + 1 rows only.** Today it runs
   over every trunk row (cpp:3594-3618, scheduler.cpp:4157-4199).
   - Size: -0.3..-0.45 s at 136k. It also applies to SHORT prefills, i.e. the c8 newcomer step (NC lever).
   - Why exact: target output does not depend on drafts, and the drafter ring keeps only the last ~128 rows
     ([MP]:1046-1054).
3. **PSK-CR: exact cone replay on a prefix hit**, the exact form of the official bounded replay ([R] §3.2.2).
   - How: resume at the deepest chunk-grid point without a snapshot, then run layer l over rows
     [H - (160 + (39-l)·128), H) with no global-KV stores.
   - Size: ~0.7-1.0 s per hit, vs K4 worst +3.04 s and K8 mean +4.11 s. It also allows tail-only anchors
     (cold = K8's 31.4 s or better) and removes the shared-head third-prompt problem.
   - The largest build of the list.
4. **One-pass mHC in prefill:** fold mhc_mix's projection and sum of squares into mhc_post_collapse_rmsnorm.
   - Size: up to -0.5 s at 136k.
   - Exact only if the FIN partial order is reproduced: needs a byte check (libr4d/r4d_ds41_elem.hip:849-1070).
5. **Engram kv GEMM dedup in prefill:** unique n-gram rows, then scatter.
   - Size: (repeat share) × ~1.0 s at 136k. Count the repeat share host-side first.
6. **Engram kv projection split by column across ranks, decode band only:** ~+2 % at 1 user, ~0 at c8.
7. **Indexer score -> top-k fusion / exact fp8 block dot.** Measure index_score vs index_topk ms first.

## Not exact (user decision only)
- Official SWA Bounded Replay is approximate (vLLM / SGLang default; "not numerically equivalent").

## Correctness check
- No risk found: quantisation floors, candidate pin/ties, visibility, RoPE per layer kind, DSpark tap point,
  clamps and softplus all match.
- Deliberate deviations (our own reference): Engram table IQ4_NL, engram_kv q6_K, EXL3 2.25 experts, rank
  partials rounded to bf16 before the cross-rank f32 sum, route-weight ulp ordering.

## Memory / capacity (10+ users)
- FP4 main KV is 890 B/token officially. Our default slot is 1024 B bf16 latent, ~2760 B/token including the index.
- RADIANCE_DS41_LATENT=nvfp4 packs it; its slowdown is open. This is a capacity lever for MAXSEQS > 8.
