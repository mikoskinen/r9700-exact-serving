## night1021-flash-s-pilot10
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **90.0%** (95% bootstrap CI 70.0-100.0) |
| strict pass (reward = 1) | 9/10 = 90.0% (Wilson 95% CI 59.6-98.2) |
| wall time | 0.30 h (2026-10-21 18:23Z .. 18:41Z) |
| throughput | 33.0 trials/hour |
| agent time per trial | median 3.8 min, p90 4.3 min, sum 0.7 h |
| agent tokens (harbor) | in 6.68 M (cached 6.29 M), out 0.11 M; per trial in 668 k / out 11.2 k |
| exceptions | none |
| proxy requests | 145 (0 failed); tokens in 6.68 M (cached 6.29 M), out 0.11 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 100% |
| hard | 3 | 67% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 145 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 3.8 | 10 | 287 | 6.1 |  |
| broken-python | easy | 1.00 | 2.2 | 15 | 219 | 1.8 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 14.0 | 15 | 3720 | 45.0 |  |
| csv-json-jsonl-merger | medium | 1.00 | 3.6 | 15 | 398 | 5.4 |  |
| db-migration-local-storage | hard | 1.00 | 3.9 | 10 | 317 | 6.0 |  |
| fix_async_worker_queue | hard | 1.00 | 2.6 | 10 | 230 | 2.8 |  |
| jq-data-processing | easy | 1.00 | 3.8 | 15 | 255 | 6.0 |  |
| log-summary | easy | 1.00 | 1.6 | 15 | 106 | 0.4 |  |
| malicious-package-forensics | medium | 1.00 | 4.3 | 15 | 410 | 18.9 |  |
| python-api-rate-limit | medium | 1.00 | 4.1 | 15 | 734 | 19.1 |  |
