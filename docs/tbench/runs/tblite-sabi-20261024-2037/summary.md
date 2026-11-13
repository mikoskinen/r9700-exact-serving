## tblite-sabi-20261024-2037
OpenThoughts-TBLite @ f075e463 (Harbor `openthoughts-tblite@2.0`), subset `easy,medium` (60 tasks, statuses oracle,oracle-partial,public-solved), attempts 1
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine `sabi-noema`, proxy mode `sabi`, n-concurrent 24, agent timeout x1.0, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| **score** (mean reward) | **42.2%** (95% bootstrap CI 30.7%-54.1%) |
| strict pass (reward = 1) | 22/60 = 36.7% (Wilson 95% CI 25.6%-49.3%) |
| agent timeouts | 34/60 (of which 6 still reward 1: the verifier grades what is on disk) |
| other exceptions | none |
| wall time | 1.22 h (2026-10-24 17:37Z .. 18:51Z), 49.2 trials/h |
| agent minutes per trial | median 12.7, mean 13.1, p90 20.0, max 60.0; sum 13.1 h |
| non-agent minutes per trial (build+setup+verify) | median 0.51, max 2.8 |
| tokens per trial (harbor) | in mean 546 k (median 279 k, cached 89%), out mean 15.5 k (median 11.3 k); total in 32.78 M, out 0.93 M |
| proxy requests | 670 (0 failed) |
| proxy non-local deployments | none |

| difficulty (author label) | trials | score | strict pass (Wilson 95% CI) | timeouts | agent min median |
|---|---|---|---|---|---|
| easy | 18 | 44.4% | 8/18 (24.6%-66.3%) | 10 | 10.5 |
| medium | 27 | 35.0% | 7/27 (13.2%-44.7%) | 17 | 15.0 |
| hard | 15 | 52.6% | 7/15 (24.8%-69.9%) | 7 | 10.0 |

| validation status | trials | score | strict |
|---|---|---|---|
| oracle | 56 | 39.3% | 22/56 |
| oracle-partial | 4 | 83.2% | 0/4 |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> sabi-noema | ok | 351 |
| cc-sonnet-a -> sabi-noema | ok | 319 |

### per task

| task | diff | status | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|---|
| broken-python | easy | oracle | 1.00 | 6.2 | 15 | 219 | 2.8 |  |
| build-merkle-tree-cli-sha512 | easy | oracle | 1.00 | 15.0 | 15 | 554 | 8.6 | AgentTimeoutError |
| california-housing-api | easy | oracle | 0.00 | 10.0 | 10 | 356 | 5.9 | AgentTimeoutError |
| convolutional-layers | easy | oracle | 0.00 | 12.0 | 12 | 188 | 14.5 | AgentTimeoutError |
| cosign-keyless-signing | easy | oracle | 0.00 | 15.0 | 15 | 458 | 15.4 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | oracle | 0.00 | 15.0 | 15 | 247 | 13.4 | AgentTimeoutError |
| ekf-localization | easy | oracle | 1.00 | 6.8 | 10 | 177 | 4.5 |  |
| hydra-debug-slurm-mode | easy | oracle | 0.00 | 6.0 | 6 | 110 | 4.0 | AgentTimeoutError |
| jq-data-processing | easy | oracle | 1.00 | 9.9 | 15 | 666 | 7.1 |  |
| jsonl-aggregator | easy | oracle | 1.00 | 4.5 | 15 | 169 | 2.4 |  |
| log-summary | easy | oracle | 1.00 | 5.0 | 15 | 134 | 0.9 |  |
| mlflow-register | easy | oracle | 0.00 | 15.0 | 15 | 335 | 6.0 | AgentTimeoutError |
| pandas-etl | easy | oracle | 1.00 | 7.4 | 15 | 202 | 2.7 |  |
| raft-log-repair-concurrent-access | easy | oracle | 0.00 | 15.0 | 15 | 264 | 23.9 | AgentTimeoutError |
| schedule-vacation | easy | oracle | 1.00 | 10.9 | 15 | 305 | 9.4 |  |
| supply-chain-fulfillment | easy | oracle | 0.00 | 9.9 | 10 | 194 | 11.3 |  |
| systemd-log-monitoring | easy | oracle | 0.00 | 15.0 | 15 | 1172 | 19.0 | AgentTimeoutError |
| vimscript-vim-quine | easy | oracle | 0.00 | 15.0 | 15 | 1242 | 33.2 | AgentTimeoutError |
| acl-permissions-inheritance | medium | oracle | 1.00 | 7.8 | 15 | 199 | 3.6 |  |
| api-endpoint-permission-canonicalizer | medium | oracle | 1.00 | 15.0 | 15 | 259 | 18.5 | AgentTimeoutError |
| bandit-delayed-feedback | medium | oracle | 0.00 | 15.0 | 15 | 214 | 14.1 | AgentTimeoutError |
| build-system-task-ordering | medium | oracle | 0.00 | 15.0 | 15 | 167 | 21.3 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | oracle | 0.00 | 15.0 | 15 | 310 | 6.3 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | oracle | 1.00 | 7.2 | 15 | 201 | 3.0 |  |
| distributed-test-execution-scheduler | medium | oracle | 0.00 | 20.0 | 20 | 588 | 21.6 | AgentTimeoutError |
| fix-js-network-controller | medium | oracle | 1.00 | 9.2 | 10 | 163 | 9.0 |  |
| industrial-kiln-controller | medium | oracle | 0.00 | 10.0 | 10 | 126 | 14.7 | AgentTimeoutError |
| iot-device-registration-server | medium | oracle | 1.00 | 21.7 | 60 | 545 | 12.6 |  |
| malicious-package-forensics | medium | oracle | 0.00 | 15.0 | 15 | 158 | 18.2 | AgentTimeoutError |
| maven-slf4j-conflict | medium | oracle | 0.00 | 10.0 | 10 | 235 | 2.5 | AgentTimeoutError |
| monorepo-changelog-cli | medium | oracle | 1.00 | 10.0 | 10 | 164 | 12.1 | AgentTimeoutError |
| mtls-cert-rotation | medium | oracle | 0.00 | 10.0 | 10 | 293 | 7.9 | AgentTimeoutError |
| neural-architecture-search-final | medium | oracle | 0.00 | 13.3 | 13 | 226 | 19.4 | AgentTimeoutError |
| neutron-submission | medium | oracle | 0.00 | 15.0 | 15 | 226 | 13.9 | AgentTimeoutError |
| pgn-chess-repair-puzzles | medium | oracle | 0.00 | 60.0 | 60 | 7221 | 140.5 | AgentTimeoutError |
| python-api-rate-limit | medium | oracle | 0.00 | 15.0 | 15 | 330 | 12.4 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | oracle | 0.00 | 14.3 | 15 | 329 | 18.7 |  |
| sales-data-csv-analysis | medium | oracle | 1.00 | 20.2 | 60 | 517 | 54.4 |  |
| scan-linux-persistence-artifacts | medium | oracle | 0.00 | 14.5 | 15 | 396 | 15.0 |  |
| security-breach-incident-response | medium | oracle-partial | 0.72 | 22.6 | 60 | 1456 | 58.4 |  |
| security-incident-log-analysis | medium | oracle-partial | 0.88 | 9.5 | 60 | 361 | 5.5 |  |
| sign-vector-game | medium | oracle | 0.00 | 15.0 | 15 | 442 | 14.9 | AgentTimeoutError |
| task-xxe-exploit | medium | oracle | 0.00 | 5.0 | 5 | 82 | 2.8 | AgentTimeoutError |
| todos-api | medium | oracle | 0.00 | 8.3 | 8 | 62 | 0.2 | AgentTimeoutError |
| tsl-test-case-generation | medium | oracle-partial | 0.84 | 15.0 | 60 | 347 | 55.5 |  |
| anomaly-detection-ranking | hard | oracle | 1.00 | 8.5 | 60 | 438 | 5.0 |  |
| application-debug | hard | oracle | 0.00 | 7.0 | 10 | 272 | 5.7 |  |
| auth_token_race_condition | hard | oracle | 1.00 | 4.8 | 10 | 139 | 1.2 |  |
| bash-log-processor-fix | hard | oracle-partial | 0.89 | 27.1 | 60 | 610 | 20.9 |  |
| chained-forensic-extraction_20260101_011957 | hard | oracle | 1.00 | 14.2 | 30 | 441 | 12.4 |  |
| container-registry-optimization | hard | oracle | 0.00 | 5.0 | 5 | 55 | 0.2 | AgentTimeoutError |
| db-migration-local-storage | hard | oracle | 1.00 | 10.0 | 10 | 279 | 9.8 | AgentTimeoutError |
| fix_async_worker_queue | hard | oracle | 1.00 | 9.2 | 10 | 361 | 3.8 |  |
| mech-system | hard | oracle | 1.00 | 15.0 | 15 | 277 | 16.3 | AgentTimeoutError |
| multi-labeller | hard | oracle | 1.00 | 15.0 | 15 | 316 | 17.9 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | oracle | 0.00 | 6.0 | 6 | 181 | 4.3 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | oracle | 0.00 | 13.3 | 13 | 381 | 4.3 | AgentTimeoutError |
| service-deployment-wave-planner | hard | oracle | 0.00 | 30.0 | 30 | 5924 | 44.9 | AgentTimeoutError |
| submission_a63937a5_20251224_152124 | hard | oracle | 0.00 | 11.7 | 30 | 388 | 9.4 |  |
| token-auth-websocket | hard | oracle | 0.00 | 6.3 | 30 | 112 | 9.0 |  |
