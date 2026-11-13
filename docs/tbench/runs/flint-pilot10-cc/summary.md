## flint-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/flint`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **60.0%** (95% bootstrap CI 30.0-90.0) |
| strict pass (reward = 1) | 6/10 = 60.0% (Wilson 95% CI 31.3-83.2) |
| wall time | 0.28 h (2026-10-23 02:55Z .. 03:12Z) |
| throughput | 36.3 trials/hour |
| agent time per trial | median 9.3 min, p90 14.0 min, sum 1.4 h |
| agent tokens (harbor) | in 5.87 M (cached 5.44 M), out 0.11 M; per trial in 587 k / out 11.5 k |
| exceptions | AgentTimeoutError: 3 |
| proxy requests | 150 (2 failed); tokens in 6.10 M (cached 5.60 M), out 0.12 M |
| proxy non-local deployments | **2** {(None, 'claude-sonnet-4-5', None): 2} |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 50% |
| hard | 3 | 33% |
| easy | 3 | 100% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> flint | ok | 74 |
| cc-sonnet-a -> flint | ok | 74 |
| None -> claude-sonnet-4-5 | fail | 2 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 10.0 | 10 | 1573 | 16.9 | AgentTimeoutError |
| broken-python | easy | 1.00 | 4.1 | 15 | 473 | 4.6 |  |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 675 | 18.7 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 5.6 | 15 | 203 | 5.1 |  |
| db-migration-local-storage | hard | 0.00 | 10.0 | 10 | 367 | 11.4 | AgentTimeoutError |
| fix_async_worker_queue | hard | 1.00 | 7.6 | 10 | 490 | 4.1 |  |
| jq-data-processing | easy | 1.00 | 7.7 | 15 | 536 | 7.9 |  |
| log-summary | easy | 1.00 | 1.6 | 15 | 205 | 1.4 |  |
| malicious-package-forensics | medium | 1.00 | 14.0 | 15 | 425 | 21.8 |  |
| python-api-rate-limit | medium | 0.00 | 9.3 | 15 | 925 | 23.1 |  |
