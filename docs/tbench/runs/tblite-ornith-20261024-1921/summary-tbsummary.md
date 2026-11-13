## tblite-ornith-20261024-1921
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/ornith-1.5-35b-w4a8`, proxy mode `sabi`, n-concurrent 24, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| score (mean reward) | **52.0%** (95% bootstrap CI 39.7-64.2) |
| strict pass (reward = 1) | 28/60 = 46.7% (Wilson 95% CI 34.6-59.1) |
| wall time | 1.14 h (2026-10-24 16:21Z .. 17:30Z) |
| throughput | 52.7 trials/hour |
| agent time per trial | median 11.0 min, p90 15.5 min, sum 11.7 h |
| agent tokens (harbor) | in 82.82 M (cached 79.55 M), out 1.62 M; per trial in 1380 k / out 27.1 k |
| exceptions | AgentTimeoutError: 33, UnknownApiError: 1 |
| proxy requests | 1360 (2 failed); tokens in 85.18 M (cached 80.91 M), out 1.77 M |
| proxy non-local deployments | **2** {(None, 'claude-sonnet-4-5', None): 2} |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 27 | 42% |
| easy | 18 | 56% |
| hard | 15 | 65% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> ornith-1.5-35b-w4a8 | ok | 753 |
| cc-sonnet-b -> ornith-1.5-35b-w4a8 | ok | 605 |
| None -> claude-sonnet-4-5 | fail | 2 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| acl-permissions-inheritance | medium | 1.00 | 2.4 | 15 | 171 | 4.3 |  |
| anomaly-detection-ranking | hard | 1.00 | 5.0 | 60 | 1231 | 10.5 |  |
| api-endpoint-permission-canonicalizer | medium | 0.00 | 15.0 | 15 | 238 | 21.7 | AgentTimeoutError |
| application-debug | hard | 1.00 | 7.5 | 10 | 490 | 16.7 |  |
| auth_token_race_condition | hard | 1.00 | 1.3 | 10 | 140 | 1.4 |  |
| bandit-delayed-feedback | medium | 0.00 | 15.0 | 15 | 893 | 28.1 | AgentTimeoutError |
| bash-log-processor-fix | hard | 0.75 | 17.5 | 60 | 868 | 32.3 |  |
| broken-python | easy | 1.00 | 2.1 | 15 | 254 | 3.5 |  |
| build-merkle-tree-cli-sha512 | easy | 1.00 | 15.0 | 15 | 1619 | 33.2 | AgentTimeoutError |
| build-system-task-ordering | medium | 0.00 | 15.0 | 15 | 160 | 32.6 | AgentTimeoutError |
| california-housing-api | easy | 1.00 | 9.3 | 10 | 1100 | 18.5 |  |
| chained-forensic-extraction_20260101_011957 | hard | 1.00 | 7.4 | 30 | 845 | 17.1 |  |
| container-registry-optimization | hard | 0.00 | 5.0 | 5 | 84 | 0.3 | AgentTimeoutError |
| convolutional-layers | easy | 0.00 | 12.0 | 12 | 629 | 22.4 | AgentTimeoutError |
| cosign-keyless-signing | easy | 1.00 | 15.0 | 15 | 1556 | 28.6 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 901 | 24.8 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | 0.00 | 15.0 | 15 | 552 | 31.5 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 3.1 | 15 | 337 | 5.8 |  |
| db-migration-local-storage | hard | 1.00 | 8.9 | 10 | 979 | 17.5 |  |
| distributed-test-execution-scheduler | medium | 1.00 | 20.0 | 20 | 1618 | 36.8 | AgentTimeoutError |
| ekf-localization | easy | 1.00 | 9.7 | 10 | 833 | 20.2 |  |
| fix-js-network-controller | medium | 1.00 | 7.1 | 10 | 292 | 12.4 |  |
| fix_async_worker_queue | hard | 1.00 | 4.6 | 10 | 621 | 10.0 |  |
| hydra-debug-slurm-mode | easy | 0.00 | 6.0 | 6 | 637 | 10.0 | AgentTimeoutError |
| industrial-kiln-controller | medium | 0.00 | 10.0 | 10 | 121 | 10.2 | AgentTimeoutError |
| iot-device-registration-server | medium | 1.00 | 12.5 | 60 | 1000 | 22.9 |  |
| jq-data-processing | easy | 1.00 | 4.0 | 15 | 395 | 9.1 |  |
| jsonl-aggregator | easy | 1.00 | 1.8 | 15 | 224 | 3.6 |  |
| log-summary | easy | 1.00 | 0.7 | 15 | 162 | 1.4 |  |
| malicious-package-forensics | medium | 0.00 | 15.0 | 15 | 724 | 18.0 | AgentTimeoutError |
| maven-slf4j-conflict | medium | 1.00 | 10.0 | 10 | 1158 | 18.1 | AgentTimeoutError |
| mech-system | hard | 0.00 | 15.0 | 15 | 513 | 33.5 | AgentTimeoutError |
| mlflow-register | easy | 0.00 | 4.0 | 15 | 512 | 5.4 |  |
| monorepo-changelog-cli | medium | 0.00 | 10.0 | 10 | 276 | 20.6 | AgentTimeoutError |
| mtls-cert-rotation | medium | 0.00 | 10.0 | 10 | 667 | 20.7 | AgentTimeoutError |
| multi-labeller | hard | 0.00 | 15.0 | 15 | 308 | 33.7 | AgentTimeoutError |
| neural-architecture-search-final | medium | 0.00 | 13.3 | 13 | 350 | 17.6 | AgentTimeoutError |
| neutron-submission | medium | 0.00 | 15.0 | 15 | 245 | 17.2 | AgentTimeoutError |
| pandas-etl | easy | 1.00 | 2.8 | 15 | 273 | 5.8 |  |
| pgn-chess-repair-puzzles | medium | 0.00 | 57.9 | 60 | 17258 | 191.4 | UnknownApiError |
| python-api-rate-limit | medium | 0.00 | 15.0 | 15 | 635 | 29.4 | AgentTimeoutError |
| raft-log-repair-concurrent-access | easy | 0.00 | 15.0 | 15 | 117 | 18.3 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | 0.00 | 6.0 | 6 | 768 | 11.5 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | 0.00 | 13.3 | 13 | 479 | 23.5 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | 1.00 | 15.0 | 15 | 2229 | 34.4 | AgentTimeoutError |
| sales-data-csv-analysis | medium | 1.00 | 7.7 | 60 | 292 | 15.1 |  |
| scan-linux-persistence-artifacts | medium | 0.00 | 15.0 | 15 | 369 | 31.3 | AgentTimeoutError |
| schedule-vacation | easy | 1.00 | 8.3 | 15 | 1352 | 17.1 |  |
| security-breach-incident-response | medium | 0.72 | 11.0 | 60 | 1669 | 25.2 |  |
| security-incident-log-analysis | medium | 0.88 | 6.4 | 60 | 942 | 11.2 |  |
| service-deployment-wave-planner | hard | 1.00 | 30.0 | 30 | 10357 | 139.1 | AgentTimeoutError |
| sign-vector-game | medium | 0.00 | 15.0 | 15 | 514 | 36.0 | AgentTimeoutError |
| submission_a63937a5_20251224_152124 | hard | 1.00 | 18.2 | 30 | 3102 | 51.0 |  |
| supply-chain-fulfillment | easy | 0.00 | 10.0 | 10 | 466 | 17.5 | AgentTimeoutError |
| systemd-log-monitoring | easy | 0.00 | 15.0 | 15 | 1887 | 39.1 | AgentTimeoutError |
| task-xxe-exploit | medium | 1.00 | 5.0 | 5 | 697 | 11.2 | AgentTimeoutError |
| todos-api | medium | 0.00 | 8.3 | 8 | 212 | 17.2 | AgentTimeoutError |
| token-auth-websocket | hard | 1.00 | 25.9 | 30 | 10954 | 112.3 |  |
| tsl-test-case-generation | medium | 0.83 | 15.5 | 60 | 1515 | 64.4 |  |
| vimscript-vim-quine | easy | 0.00 | 15.0 | 15 | 3036 | 50.0 | AgentTimeoutError |
