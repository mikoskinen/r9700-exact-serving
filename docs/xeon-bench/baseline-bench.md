# xeon Radiance baseline throughput bench (2026-10-12 UTC, run 16:50-17:22 UTC)

## Conditions (baseline)
- Server: systemd `radiance`, container radiance-tp2-vision, vLLM 0.31 / Radiance 1.2.1, amd/Qwen3.8-27B-Quark-AWQ-MXFP4 + DFlash2 W4A16 (num_speculative_tokens 7, RADIANCE_DYNAMIC_DRAFT=1 schedule 1:8,2:7,4:6,8:5,16:4), TP2 (--disable-custom-all-reduce), max-num-seqs 8, max-num-batched-tokens 1024, prefix caching ON, max-model-len 262144, kv fp8, cudagraph FULL_DECODE_ONLY.
- GPU0 06:00.0 behind root port 00:03.0 = PCIe Gen3 (8 GT/s) x8, NUMA node 0; GPU1 83:00.0 behind 80:03.0 = Gen3 x16, NUMA node 1 (TP2 spans both sockets).
- kernel.numa_balancing=1; power_dpm_force_performance_level=auto; power cap 210 W per card; kernel 7.0.0-38-generic.
- Power/temps from ~/gpu/temps-dual.csv (2 s samples) over each run's active window; KV%% / acceptance from server log (10 s stats lines).
- Client: `vllm bench serve` inside container, openai-chat backend, --ignore-eos on ALL runs (incl. C), num-warmups 0, fresh random seed per run.
- C dataset: 32 prompts, each 13.0-15.9k tokens of distinct python3.12 stdlib .py files + 'Refactor the most complex function above and explain the changes.' (chat template applied). Note: runs sample from the same 32, so C_c4/C_c8 had some prefix-cache reuse (server cumulative hit rate up to 9.4%% by C_c8) -> C_c8 input tok/s slightly inflated.

## Results

| run | conc/prompts | dur s | out tok/s total | per-req tok/s (1000/TPOT) | TPOT ms | TTFT p50 / p95 s | input tok/s (avg over run) | req/s | KV% peak | DFlash accept len (mean of log lines) | W avg g0+g1 | W peak g0/g1 | Tj peak g0/g1 | Tmem peak g0/g1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_c1 | 1/4 | 62 | 32.5 | 101.9 | 9.8 | 9.13 / 13.78 | 1040 | 0.065 | 3.0 | 3.07 | 352 | 239/268 | 70/69 | 56/60 |
| A_c4 | 4/8 | 98 | 41.0 | 17.8 | 56.3 | 15.21 / 39.52 | 1312 | 0.082 | 12.2 | 2.75 | 368 | 276/324 | 81/78 | 70/76 |
| A_c8 | 8/16 | 169 | 47.5 | 9.3 | 107.4 | 22.20 / 68.70 | 1520 | 0.095 | 24.0 | 2.74 | 408 | 284/323 | 82/84 | 76/82 |
| A_c16 | 16/32 | 330 | 48.5 | 8.2 | 121.7 | 94.05 / 143.05 | 1553 | 0.097 | 24.0 | 2.91 | 415 | 321/268 | 86/87 | 82/86 |
| B_c1 | 1/3 | 139 | 10.8 | 67.7 | 14.8 | 39.05 / 39.20 | 1377 | 0.022 | 6.6 | 2.27 | 410 | 322/307 | 86/88 | 78/86 |
| C_c1 | 1/4 | 47 | 42.6 | 131.4 | 7.6 | 8.01 / 8.26 | 1198 | 0.085 | 3.0 | 4.07 | 411 | 255/235 | 81/83 | 74/80 |
| C_c4 | 4/8 | 76 | 52.8 | 22.4 | 44.7 | 9.25 / 31.13 | 1571 | 0.106 | 12.0 | 4.05 | 397 | 285/303 | 81/84 | 74/82 |
| C_c8 | 8/16 | 112 | 71.4 | 13.3 | 75.2 | 11.55 / 39.39 | 2086 | 0.143 | 22.7 | 3.43 | 403 | 322/248 | 82/85 | 76/84 |

Columns: per-req tok/s = 1000/mean TPOT (includes prefill interference from chunked prefill of other requests); input tok/s = total input tokens / duration; W avg = mean(g0)+mean(g1).

## Status log
```
START A_c1 16:50:46
END A_c1 16:52:36
START A_c4 16:52:41
END A_c4 16:55:08
START A_c8 16:55:13
END A_c8 16:58:52
START C_c1 16:58:57
END C_c1 17:00:32
START C_c4 17:00:37
END C_c4 17:02:39
START C_c8 17:02:44
END C_c8 17:05:25
START A_c16 17:05:30
END A_c16 17:11:54
START B_c1 17:11:59
END B_c1 17:15:08
START B_c4 17:15:13
STOP time limit, B_c4 aborted 17:21:32
END B_c4 17:21:32
DONE
```
## Exact commands
```
A_c1: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 1 --num-prompts 4 --seed 2705415611 --result-filename A_c1.json --dataset-name random --random-input-len 16000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
A_c4: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 4 --num-prompts 8 --seed 308423873 --result-filename A_c4.json --dataset-name random --random-input-len 16000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
A_c8: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 8 --num-prompts 16 --seed 2471530372 --result-filename A_c8.json --dataset-name random --random-input-len 16000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
C_c1: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 1 --num-prompts 4 --seed 2638717315 --result-filename C_c1.json --dataset-name custom --dataset-path /root/.cache/huggingface/bench/code32.jsonl --custom-output-len 500 --ignore-eos
C_c4: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 4 --num-prompts 8 --seed 1288430679 --result-filename C_c4.json --dataset-name custom --dataset-path /root/.cache/huggingface/bench/code32.jsonl --custom-output-len 500 --ignore-eos
C_c8: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 8 --num-prompts 16 --seed 460031806 --result-filename C_c8.json --dataset-name custom --dataset-path /root/.cache/huggingface/bench/code32.jsonl --custom-output-len 500 --ignore-eos
A_c16: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 16 --num-prompts 32 --seed 3259927211 --result-filename A_c16.json --dataset-name random --random-input-len 16000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
B_c1: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 1 --num-prompts 3 --seed 1518117246 --result-filename B_c1.json --dataset-name random --random-input-len 64000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
B_c4: docker exec radiance-tp2-vision vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8000 --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --served-model-name qwen3.8-27b --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 --num-warmups 0 --save-result --result-dir /root/.cache/huggingface/bench/res --max-concurrency 4 --num-prompts 8 --seed 3192128330 --result-filename B_c4.json --dataset-name random --random-input-len 64000 --random-output-len 500 --random-range-ratio 0 --ignore-eos
```
Dataset builder: ~/.cache/huggingface/mkds.py (run with docker exec -e HF_HUB_OFFLINE=1 radiance-tp2-vision python3 /root/.cache/huggingface/mkds.py). Runner: ~/gpu/xeon-bench/baseline/run.sh; raw results ~/.cache/huggingface/bench/res/*.json, outputs ~/gpu/xeon-bench/baseline/*.out, server log slices *.srvlog.
