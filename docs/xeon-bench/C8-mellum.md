# C8 — JetBrains Mellum2.1-12B-A2.5B-Thinking gate (xeon, 2026-10-11)

Quick screen, single R9700 (HIP_VISIBLE_DEVICES=1), ctx 131072, bench_fresh.py (fresh prompts, no prefix-cache inflation).

| Engine / quant | decode short | decode @17.5k | prefill @17.5k | tool call | notes |
|---|---|---|---|---|---|
| llama.cpp HIP Q8_0 (12.9 GB) | 134.0 | 120.0 | 5753 | OK (get_weather, valid JSON) | |
| llama.cpp HIP MXFP4_MOE (7.0 GB) | **156.1** | **146.5** | **5964** | OK | best |
| vLLM radiance 1.0.0 bf16 (safetensors) | 113.1 | 35.8 | 3468 | OK (hermes parser) | KV 328k tokens @0.92 GMU; decode collapses at depth |

Sanity completion (Q8_0): reasoning_content + correct concise is_prime(); thinking output separated correctly by --jinja.

llama.cpp flags: -ngl 999 -fa on --fit off -c 131072 --jinja -np 1 --no-cache-prompt (image llamacpp-main-hip:gfx1201).

Reference: Radiance TP2 Qwen3.8-27B 166 / 136-160 decode, 1865 prefill; Ornith Q8 2-GPU HIP 95/88, 3469 prefill.

## Verdict
Fast on one card (MXFP4: ~156/147 decode, ~6k prefill — 3x Radiance prefill) and tool calls work, but it is a 2.5B-active
12B model; decode is no better than the Radiance 27B TP2 setup, so it is only interesting as a cheap single-GPU
side model (e.g. second card while the other runs something else), not as the main coder. vLLM path not worth pursuing
(decode drops to 36 tok/s at 17k). Quality not evaluated.
