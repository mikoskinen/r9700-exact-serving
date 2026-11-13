## sage-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/sage-a1.5`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **50.0%** (95% bootstrap CI 20.0-80.0) |
| strict pass (reward = 1) | 5/10 = 50.0% (Wilson 95% CI 23.7-76.3) |
| wall time | 0.32 h (2026-10-21 12:30Z .. 12:49Z) |
| throughput | 31.4 trials/hour |
| agent time per trial | median 4.6 min, p90 15.0 min, sum 1.1 h |
| agent tokens (harbor) | in 5.56 M (cached 4.94 M), out 0.19 M; per trial in 556 k / out 19.1 k |
| exceptions | AgentTimeoutError: 4 |
| proxy requests | 126 (0 failed); tokens in 6.67 M (cached 5.81 M), out 0.22 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 25% |
| hard | 3 | 33% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> sage-a1.5 | ok | 80 |
| cc-sonnet-b -> sage-a1.5 | ok | 46 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 4.6 | 10 | 384 | 13.3 |  |
| broken-python | easy | 1.00 | 1.5 | 15 | 310 | 3.1 |  |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 1210 | 41.1 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 2.4 | 15 | 160 | 7.0 |  |
| db-migration-local-storage | hard | 1.00 | 10.0 | 10 | 1472 | 40.0 | AgentTimeoutError |
| fix_async_worker_queue | hard | 0.00 | 1.3 | 10 | 53 | 3.4 |  |
| jq-data-processing | easy | 1.00 | 2.3 | 15 | 662 | 6.3 |  |
| log-summary | easy | 1.00 | 0.6 | 15 | 133 | 0.8 |  |
| malicious-package-forensics | medium | 0.00 | 15.0 | 15 | 223 | 33.3 | AgentTimeoutError |
| python-api-rate-limit | medium | 0.00 | 15.0 | 15 | 950 | 42.7 | AgentTimeoutError |
