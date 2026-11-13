## flash-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **100.0%** (95% bootstrap CI 100.0-100.0) |
| strict pass (reward = 1) | 10/10 = 100.0% (Wilson 95% CI 72.2-100.0) |
| wall time | 0.28 h (2026-10-21 14:44Z .. 15:01Z) |
| throughput | 36.2 trials/hour |
| agent time per trial | median 3.3 min, p90 5.1 min, sum 0.7 h |
| agent tokens (harbor) | in 6.45 M (cached 6.09 M), out 0.08 M; per trial in 645 k / out 7.7 k |
| exceptions | none |
| proxy requests | 162 (0 failed); tokens in 6.45 M (cached 6.09 M), out 0.08 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 100% |
| hard | 3 | 100% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 162 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 1.00 | 2.4 | 10 | 229 | 2.5 |  |
| broken-python | easy | 1.00 | 3.3 | 15 | 315 | 3.8 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 12.5 | 15 | 2482 | 23.0 |  |
| csv-json-jsonl-merger | medium | 1.00 | 2.9 | 15 | 253 | 3.3 |  |
| db-migration-local-storage | hard | 1.00 | 5.1 | 10 | 726 | 9.7 |  |
| fix_async_worker_queue | hard | 1.00 | 2.4 | 10 | 224 | 2.3 |  |
| jq-data-processing | easy | 1.00 | 3.4 | 15 | 364 | 4.5 |  |
| log-summary | easy | 1.00 | 1.6 | 15 | 132 | 0.6 |  |
| malicious-package-forensics | medium | 1.00 | 3.7 | 15 | 1306 | 17.0 |  |
| python-api-rate-limit | medium | 1.00 | 2.4 | 15 | 418 | 9.9 |  |
