## sabi-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/sabi-noema`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **80.0%** (95% bootstrap CI 50.0-100.0) |
| strict pass (reward = 1) | 8/10 = 80.0% (Wilson 95% CI 49.0-94.3) |
| wall time | 0.29 h (2026-10-21 09:59Z .. 10:17Z) |
| throughput | 34.8 trials/hour |
| agent time per trial | median 3.3 min, p90 5.3 min, sum 0.7 h |
| agent tokens (harbor) | in 5.32 M (cached 4.71 M), out 0.10 M; per trial in 532 k / out 10.1 k |
| exceptions | none |
| proxy requests | 130 (0 failed); tokens in 5.32 M (cached 4.71 M), out 0.10 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 75% |
| hard | 3 | 67% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> sabi-noema | ok | 65 |
| cc-sonnet-a -> sabi-noema | ok | 65 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 3.3 | 10 | 485 | 8.2 |  |
| broken-python | easy | 1.00 | 2.7 | 15 | 303 | 3.4 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 13.1 | 15 | 1226 | 27.3 |  |
| csv-json-jsonl-merger | medium | 1.00 | 1.4 | 15 | 137 | 1.7 |  |
| db-migration-local-storage | hard | 1.00 | 5.2 | 10 | 648 | 13.4 |  |
| fix_async_worker_queue | hard | 1.00 | 2.7 | 10 | 292 | 3.6 |  |
| jq-data-processing | easy | 1.00 | 2.7 | 15 | 518 | 4.5 |  |
| log-summary | easy | 1.00 | 1.0 | 15 | 162 | 0.9 |  |
| malicious-package-forensics | medium | 1.00 | 5.3 | 15 | 466 | 19.5 |  |
| python-api-rate-limit | medium | 0.00 | 4.9 | 15 | 1081 | 18.5 |  |
