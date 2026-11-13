## q36-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.6-35b-a3b`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **30.0%** (95% bootstrap CI 0.0-60.0) |
| strict pass (reward = 1) | 3/10 = 30.0% (Wilson 95% CI 10.8-60.3) |
| wall time | 0.13 h (2026-10-21 16:46Z .. 16:54Z) |
| throughput | 75.5 trials/hour |
| agent time per trial | median 1.4 min, p90 3.3 min, sum 0.2 h |
| agent tokens (harbor) | in 4.49 M (cached 4.02 M), out 0.04 M; per trial in 449 k / out 3.7 k |
| exceptions | none |
| proxy requests | 106 (0 failed); tokens in 4.49 M (cached 4.02 M), out 0.04 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 25% |
| hard | 3 | 0% |
| easy | 3 | 67% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> qwen3.6-35b-a3b | ok | 65 |
| cc-sonnet-b -> qwen3.6-35b-a3b | ok | 41 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 3.3 | 10 | 1242 | 8.8 |  |
| broken-python | easy | 1.00 | 1.4 | 15 | 326 | 1.5 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 3.9 | 15 | 700 | 10.4 |  |
| csv-json-jsonl-merger | medium | 0.00 | 0.3 | 15 | 26 | 0.2 |  |
| db-migration-local-storage | hard | 0.00 | 3.3 | 10 | 1559 | 13.4 |  |
| fix_async_worker_queue | hard | 0.00 | 0.4 | 10 | 52 | 0.2 |  |
| jq-data-processing | easy | 1.00 | 1.6 | 15 | 502 | 2.3 |  |
| log-summary | easy | 0.00 | 0.1 | 15 | 26 | 0.1 |  |
| malicious-package-forensics | medium | 0.00 | 0.1 | 15 | 27 | 0.1 |  |
| python-api-rate-limit | medium | 0.00 | 0.1 | 15 | 27 | 0.2 |  |
