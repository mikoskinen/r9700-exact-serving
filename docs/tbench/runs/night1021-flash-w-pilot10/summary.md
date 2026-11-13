## night1021-flash-w-pilot10
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **90.0%** (95% bootstrap CI 70.0-100.0) |
| strict pass (reward = 1) | 9/10 = 90.0% (Wilson 95% CI 59.6-98.2) |
| wall time | 0.20 h (2026-10-21 18:58Z .. 19:10Z) |
| throughput | 48.9 trials/hour |
| agent time per trial | median 2.9 min, p90 3.8 min, sum 0.5 h |
| agent tokens (harbor) | in 5.52 M (cached 5.21 M), out 0.06 M; per trial in 552 k / out 6.4 k |
| exceptions | none |
| proxy requests | 138 (0 failed); tokens in 5.52 M (cached 5.21 M), out 0.06 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 100% |
| hard | 3 | 67% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 138 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 2.0 | 10 | 174 | 2.1 |  |
| broken-python | easy | 1.00 | 3.0 | 15 | 255 | 3.3 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 8.1 | 15 | 1753 | 19.8 |  |
| csv-json-jsonl-merger | medium | 1.00 | 2.8 | 15 | 235 | 3.0 |  |
| db-migration-local-storage | hard | 1.00 | 3.8 | 10 | 483 | 6.1 |  |
| fix_async_worker_queue | hard | 1.00 | 2.9 | 10 | 342 | 3.0 |  |
| jq-data-processing | easy | 1.00 | 2.0 | 15 | 200 | 1.8 |  |
| log-summary | easy | 1.00 | 1.5 | 15 | 159 | 0.7 |  |
| malicious-package-forensics | medium | 1.00 | 3.8 | 15 | 1591 | 16.7 |  |
| python-api-rate-limit | medium | 1.00 | 2.0 | 15 | 331 | 7.5 |  |
