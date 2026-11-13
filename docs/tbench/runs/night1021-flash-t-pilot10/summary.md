## night1021-flash-t-pilot10
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **100.0%** (95% bootstrap CI 100.0-100.0) |
| strict pass (reward = 1) | 10/10 = 100.0% (Wilson 95% CI 72.2-100.0) |
| wall time | 0.20 h (2026-10-21 19:24Z .. 19:36Z) |
| throughput | 50.8 trials/hour |
| agent time per trial | median 3.2 min, p90 3.7 min, sum 0.5 h |
| agent tokens (harbor) | in 4.92 M (cached 4.57 M), out 0.07 M; per trial in 492 k / out 7.4 k |
| exceptions | none |
| proxy requests | 131 (0 failed); tokens in 4.92 M (cached 4.57 M), out 0.07 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 100% |
| hard | 3 | 100% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 131 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 1.00 | 2.2 | 10 | 204 | 2.7 |  |
| broken-python | easy | 1.00 | 3.7 | 15 | 428 | 6.1 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 7.7 | 15 | 1817 | 23.0 |  |
| csv-json-jsonl-merger | medium | 1.00 | 3.6 | 15 | 368 | 6.5 |  |
| db-migration-local-storage | hard | 1.00 | 3.2 | 10 | 312 | 5.4 |  |
| fix_async_worker_queue | hard | 1.00 | 2.3 | 10 | 256 | 2.5 |  |
| jq-data-processing | easy | 1.00 | 2.5 | 15 | 207 | 3.5 |  |
| log-summary | easy | 1.00 | 1.4 | 15 | 132 | 0.5 |  |
| malicious-package-forensics | medium | 1.00 | 3.2 | 15 | 678 | 13.5 |  |
| python-api-rate-limit | medium | 1.00 | 2.2 | 15 | 520 | 10.0 |  |
