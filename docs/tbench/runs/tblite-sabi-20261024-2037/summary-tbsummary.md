## tblite-sabi-20261024-2037
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/sabi-noema`, proxy mode `sabi`, n-concurrent 24, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| score (mean reward) | **42.2%** (95% bootstrap CI 30.7-54.1) |
| strict pass (reward = 1) | 22/60 = 36.7% (Wilson 95% CI 25.6-49.3) |
| wall time | 1.22 h (2026-10-24 17:37Z .. 18:51Z) |
| throughput | 49.2 trials/hour |
| agent time per trial | median 13.3 min, p90 20.0 min, sum 13.1 h |
| agent tokens (harbor) | in 32.78 M (cached 29.28 M), out 0.93 M; per trial in 546 k / out 15.5 k |
| exceptions | AgentTimeoutError: 34 |
| proxy requests | 670 (0 failed); tokens in 33.90 M (cached 29.86 M), out 0.98 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 27 | 35% |
| easy | 18 | 44% |
| hard | 15 | 53% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> sabi-noema | ok | 351 |
| cc-sonnet-a -> sabi-noema | ok | 319 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| acl-permissions-inheritance | medium | 1.00 | 7.8 | 15 | 199 | 3.6 |  |
| anomaly-detection-ranking | hard | 1.00 | 8.5 | 60 | 438 | 5.0 |  |
| api-endpoint-permission-canonicalizer | medium | 1.00 | 15.0 | 15 | 259 | 18.5 | AgentTimeoutError |
| application-debug | hard | 0.00 | 7.0 | 10 | 272 | 5.7 |  |
| auth_token_race_condition | hard | 1.00 | 4.8 | 10 | 139 | 1.2 |  |
| bandit-delayed-feedback | medium | 0.00 | 15.0 | 15 | 214 | 14.1 | AgentTimeoutError |
| bash-log-processor-fix | hard | 0.89 | 27.1 | 60 | 610 | 20.9 |  |
| broken-python | easy | 1.00 | 6.2 | 15 | 219 | 2.8 |  |
| build-merkle-tree-cli-sha512 | easy | 1.00 | 15.0 | 15 | 554 | 8.6 | AgentTimeoutError |
| build-system-task-ordering | medium | 0.00 | 15.0 | 15 | 167 | 21.3 | AgentTimeoutError |
| california-housing-api | easy | 0.00 | 10.0 | 10 | 356 | 5.9 | AgentTimeoutError |
| chained-forensic-extraction_20260101_011957 | hard | 1.00 | 14.2 | 30 | 441 | 12.4 |  |
| container-registry-optimization | hard | 0.00 | 5.0 | 5 | 55 | 0.2 | AgentTimeoutError |
| convolutional-layers | easy | 0.00 | 12.0 | 12 | 188 | 14.5 | AgentTimeoutError |
| cosign-keyless-signing | easy | 0.00 | 15.0 | 15 | 458 | 15.4 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 310 | 6.3 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | 0.00 | 15.0 | 15 | 247 | 13.4 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 7.2 | 15 | 201 | 3.0 |  |
| db-migration-local-storage | hard | 1.00 | 10.0 | 10 | 279 | 9.8 | AgentTimeoutError |
| distributed-test-execution-scheduler | medium | 0.00 | 20.0 | 20 | 588 | 21.6 | AgentTimeoutError |
| ekf-localization | easy | 1.00 | 6.8 | 10 | 177 | 4.5 |  |
| fix-js-network-controller | medium | 1.00 | 9.2 | 10 | 163 | 9.0 |  |
| fix_async_worker_queue | hard | 1.00 | 9.2 | 10 | 361 | 3.8 |  |
| hydra-debug-slurm-mode | easy | 0.00 | 6.0 | 6 | 110 | 4.0 | AgentTimeoutError |
| industrial-kiln-controller | medium | 0.00 | 10.0 | 10 | 126 | 14.7 | AgentTimeoutError |
| iot-device-registration-server | medium | 1.00 | 21.7 | 60 | 545 | 12.6 |  |
| jq-data-processing | easy | 1.00 | 9.9 | 15 | 666 | 7.1 |  |
| jsonl-aggregator | easy | 1.00 | 4.5 | 15 | 169 | 2.4 |  |
| log-summary | easy | 1.00 | 5.0 | 15 | 134 | 0.9 |  |
| malicious-package-forensics | medium | 0.00 | 15.0 | 15 | 158 | 18.2 | AgentTimeoutError |
| maven-slf4j-conflict | medium | 0.00 | 10.0 | 10 | 235 | 2.5 | AgentTimeoutError |
| mech-system | hard | 1.00 | 15.0 | 15 | 277 | 16.3 | AgentTimeoutError |
| mlflow-register | easy | 0.00 | 15.0 | 15 | 335 | 6.0 | AgentTimeoutError |
| monorepo-changelog-cli | medium | 1.00 | 10.0 | 10 | 164 | 12.1 | AgentTimeoutError |
| mtls-cert-rotation | medium | 0.00 | 10.0 | 10 | 293 | 7.9 | AgentTimeoutError |
| multi-labeller | hard | 1.00 | 15.0 | 15 | 316 | 17.9 | AgentTimeoutError |
| neural-architecture-search-final | medium | 0.00 | 13.3 | 13 | 226 | 19.4 | AgentTimeoutError |
| neutron-submission | medium | 0.00 | 15.0 | 15 | 226 | 13.9 | AgentTimeoutError |
| pandas-etl | easy | 1.00 | 7.4 | 15 | 202 | 2.7 |  |
| pgn-chess-repair-puzzles | medium | 0.00 | 60.0 | 60 | 7221 | 140.5 | AgentTimeoutError |
| python-api-rate-limit | medium | 0.00 | 15.0 | 15 | 330 | 12.4 | AgentTimeoutError |
| raft-log-repair-concurrent-access | easy | 0.00 | 15.0 | 15 | 264 | 23.9 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | 0.00 | 6.0 | 6 | 181 | 4.3 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | 0.00 | 13.3 | 13 | 381 | 4.3 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | 0.00 | 14.3 | 15 | 329 | 18.7 |  |
| sales-data-csv-analysis | medium | 1.00 | 20.2 | 60 | 517 | 54.4 |  |
| scan-linux-persistence-artifacts | medium | 0.00 | 14.5 | 15 | 396 | 15.0 |  |
| schedule-vacation | easy | 1.00 | 10.9 | 15 | 305 | 9.4 |  |
| security-breach-incident-response | medium | 0.72 | 22.6 | 60 | 1456 | 58.4 |  |
| security-incident-log-analysis | medium | 0.88 | 9.5 | 60 | 361 | 5.5 |  |
| service-deployment-wave-planner | hard | 0.00 | 30.0 | 30 | 5924 | 44.9 | AgentTimeoutError |
| sign-vector-game | medium | 0.00 | 15.0 | 15 | 442 | 14.9 | AgentTimeoutError |
| submission_a63937a5_20251224_152124 | hard | 0.00 | 11.7 | 30 | 388 | 9.4 |  |
| supply-chain-fulfillment | easy | 0.00 | 9.9 | 10 | 194 | 11.3 |  |
| systemd-log-monitoring | easy | 0.00 | 15.0 | 15 | 1172 | 19.0 | AgentTimeoutError |
| task-xxe-exploit | medium | 0.00 | 5.0 | 5 | 82 | 2.8 | AgentTimeoutError |
| todos-api | medium | 0.00 | 8.3 | 8 | 62 | 0.2 | AgentTimeoutError |
| token-auth-websocket | hard | 0.00 | 6.3 | 30 | 112 | 9.0 |  |
| tsl-test-case-generation | medium | 0.84 | 15.0 | 60 | 347 | 55.5 |  |
| vimscript-vim-quine | easy | 0.00 | 15.0 | 15 | 1242 | 33.2 | AgentTimeoutError |
