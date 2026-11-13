## qwen38-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-27b`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **80.0%** (95% bootstrap CI 50.0-100.0) |
| strict pass (reward = 1) | 8/10 = 80.0% (Wilson 95% CI 49.0-94.3) |
| wall time | 0.32 h (2026-10-21 15:28Z .. 15:47Z) |
| throughput | 31.4 trials/hour |
| agent time per trial | median 3.7 min, p90 11.9 min, sum 1.0 h |
| agent tokens (harbor) | in 6.88 M (cached 6.21 M), out 0.15 M; per trial in 688 k / out 15.1 k |
| exceptions | AgentTimeoutError: 1 |
| proxy requests | 163 (0 failed); tokens in 7.99 M (cached 7.29 M), out 0.16 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 75% |
| hard | 3 | 67% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> qwen3.8-27b | ok | 115 |
| cc-sonnet-b -> qwen3.8-27b | ok | 48 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 3.7 | 10 | 328 | 7.1 |  |
| broken-python | easy | 1.00 | 2.6 | 15 | 220 | 2.6 |  |
| cpp-daemon-sighup-segfault | medium | 1.00 | 15.0 | 15 | 1926 | 35.3 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 3.0 | 15 | 274 | 4.5 |  |
| db-migration-local-storage | hard | 1.00 | 3.7 | 10 | 383 | 9.8 |  |
| fix_async_worker_queue | hard | 1.00 | 2.3 | 10 | 368 | 6.1 |  |
| jq-data-processing | easy | 1.00 | 4.4 | 15 | 890 | 9.7 |  |
| log-summary | easy | 1.00 | 2.2 | 15 | 193 | 2.0 |  |
| malicious-package-forensics | medium | 1.00 | 9.7 | 15 | 483 | 32.9 |  |
| python-api-rate-limit | medium | 0.00 | 11.9 | 15 | 1812 | 41.3 |  |
