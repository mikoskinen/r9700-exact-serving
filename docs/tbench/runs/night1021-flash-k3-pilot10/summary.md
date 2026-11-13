## night1021-flash-k3-pilot10
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **100.0%** (95% bootstrap CI 100.0-100.0) |
| strict pass (reward = 1) | 10/10 = 100.0% (Wilson 95% CI 72.2-100.0) |
| wall time | 0.18 h (2026-10-21 19:54Z .. 20:05Z) |
| throughput | 54.3 trials/hour |
| agent time per trial | median 2.8 min, p90 3.5 min, sum 0.5 h |
| agent tokens (harbor) | in 4.19 M (cached 3.89 M), out 0.05 M; per trial in 419 k / out 5.4 k |
| exceptions | none |
| proxy requests | 113 (0 failed); tokens in 4.19 M (cached 3.89 M), out 0.05 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 100% |
| hard | 3 | 100% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 113 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 1.00 | 2.8 | 10 | 234 | 3.0 |  |
| broken-python | easy | 1.00 | 3.5 | 15 | 260 | 4.6 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 7.0 | 15 | 566 | 11.3 |  |
| csv-json-jsonl-merger | medium | 1.00 | 2.0 | 15 | 167 | 1.4 |  |
| db-migration-local-storage | hard | 1.00 | 3.2 | 10 | 336 | 4.1 |  |
| fix_async_worker_queue | hard | 1.00 | 2.8 | 10 | 343 | 2.7 |  |
| jq-data-processing | easy | 1.00 | 2.7 | 15 | 234 | 2.5 |  |
| log-summary | easy | 1.00 | 1.7 | 15 | 160 | 0.7 |  |
| malicious-package-forensics | medium | 1.00 | 3.1 | 15 | 1415 | 14.7 |  |
| python-api-rate-limit | medium | 1.00 | 2.1 | 15 | 476 | 8.7 |  |
