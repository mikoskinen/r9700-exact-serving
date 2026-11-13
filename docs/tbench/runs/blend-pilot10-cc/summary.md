## blend-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen-blend-27b`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **60.0%** (95% bootstrap CI 30.0-90.0) |
| strict pass (reward = 1) | 6/10 = 60.0% (Wilson 95% CI 31.3-83.2) |
| wall time | 0.26 h (2026-10-21 13:48Z .. 14:04Z) |
| throughput | 38.3 trials/hour |
| agent time per trial | median 4.4 min, p90 8.8 min, sum 0.9 h |
| agent tokens (harbor) | in 3.22 M (cached 2.87 M), out 0.05 M; per trial in 322 k / out 4.9 k |
| exceptions | none |
| proxy requests | 87 (0 failed); tokens in 3.22 M (cached 2.87 M), out 0.05 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 50% |
| hard | 3 | 33% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> qwen-blend-27b | ok | 46 |
| cc-sonnet-b -> qwen-blend-27b | ok | 41 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 4.4 | 10 | 430 | 3.1 |  |
| broken-python | easy | 1.00 | 3.0 | 15 | 245 | 1.7 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 11.6 | 15 | 490 | 11.2 |  |
| csv-json-jsonl-merger | medium | 1.00 | 2.9 | 15 | 111 | 1.6 |  |
| db-migration-local-storage | hard | 0.00 | 5.2 | 10 | 276 | 4.6 |  |
| fix_async_worker_queue | hard | 1.00 | 3.8 | 10 | 287 | 2.4 |  |
| jq-data-processing | easy | 1.00 | 2.9 | 15 | 499 | 1.6 |  |
| log-summary | easy | 1.00 | 1.4 | 15 | 106 | 0.7 |  |
| malicious-package-forensics | medium | 0.00 | 8.5 | 15 | 144 | 9.1 |  |
| python-api-rate-limit | medium | 0.00 | 8.8 | 15 | 632 | 13.0 |  |
