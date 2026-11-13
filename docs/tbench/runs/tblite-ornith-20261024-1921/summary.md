## tblite-ornith-20261024-1921
OpenThoughts-TBLite @ f075e463 (Harbor `openthoughts-tblite@2.0`), subset `easy,medium` (60 tasks, statuses oracle,oracle-partial,public-solved), attempts 1
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine `ornith-1.5-35b-w4a8`, proxy mode `sabi`, n-concurrent 24, agent timeout x1.0, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| **score** (mean reward) | **52.0%** (95% bootstrap CI 39.7%-64.2%) |
| strict pass (reward = 1) | 28/60 = 46.7% (Wilson 95% CI 34.6%-59.1%) |
| agent timeouts | 33/60 (of which 7 still reward 1: the verifier grades what is on disk) |
| other exceptions | UnknownApiError: 1 |
| wall time | 1.14 h (2026-10-24 16:21Z .. 17:30Z), 52.7 trials/h |
| agent minutes per trial | median 10.5, mean 11.7, p90 15.5, max 57.9; sum 11.7 h |
| non-agent minutes per trial (build+setup+verify) | median 0.54, max 1.1 |
| tokens per trial (harbor) | in mean 1380 k (median 635 k, cached 96%), out mean 27.1 k (median 18.5 k); total in 82.82 M, out 1.62 M |
| proxy requests | 1360 (2 failed) |
| proxy non-local deployments | **2** {(None, 'claude-sonnet-4-5', None): 2} |

| difficulty (author label) | trials | score | strict pass (Wilson 95% CI) | timeouts | agent min median |
|---|---|---|---|---|---|
| easy | 18 | 55.6% | 10/18 (33.7%-75.4%) | 9 | 9.5 |
| medium | 27 | 42.3% | 9/27 (18.6%-52.2%) | 18 | 13.3 |
| hard | 15 | 65.0% | 9/15 (35.7%-80.2%) | 6 | 8.9 |

| validation status | trials | score | strict |
|---|---|---|---|
| oracle | 56 | 50.0% | 28/56 |
| oracle-partial | 4 | 79.4% | 0/4 |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-a -> ornith-1.5-35b-w4a8 | ok | 753 |
| cc-sonnet-b -> ornith-1.5-35b-w4a8 | ok | 605 |
| None -> claude-sonnet-4-5 | fail | 2 |

### per task

| task | diff | status | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|---|
| broken-python | easy | oracle | 1.00 | 2.1 | 15 | 254 | 3.5 |  |
| build-merkle-tree-cli-sha512 | easy | oracle | 1.00 | 15.0 | 15 | 1619 | 33.2 | AgentTimeoutError |
| california-housing-api | easy | oracle | 1.00 | 9.3 | 10 | 1100 | 18.5 |  |
| convolutional-layers | easy | oracle | 0.00 | 12.0 | 12 | 629 | 22.4 | AgentTimeoutError |
| cosign-keyless-signing | easy | oracle | 1.00 | 15.0 | 15 | 1556 | 28.6 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | oracle | 0.00 | 15.0 | 15 | 552 | 31.5 | AgentTimeoutError |
| ekf-localization | easy | oracle | 1.00 | 9.7 | 10 | 833 | 20.2 |  |
| hydra-debug-slurm-mode | easy | oracle | 0.00 | 6.0 | 6 | 637 | 10.0 | AgentTimeoutError |
| jq-data-processing | easy | oracle | 1.00 | 4.0 | 15 | 395 | 9.1 |  |
| jsonl-aggregator | easy | oracle | 1.00 | 1.8 | 15 | 224 | 3.6 |  |
| log-summary | easy | oracle | 1.00 | 0.7 | 15 | 162 | 1.4 |  |
| mlflow-register | easy | oracle | 0.00 | 4.0 | 15 | 512 | 5.4 |  |
| pandas-etl | easy | oracle | 1.00 | 2.8 | 15 | 273 | 5.8 |  |
| raft-log-repair-concurrent-access | easy | oracle | 0.00 | 15.0 | 15 | 117 | 18.3 | AgentTimeoutError |
| schedule-vacation | easy | oracle | 1.00 | 8.3 | 15 | 1352 | 17.1 |  |
| supply-chain-fulfillment | easy | oracle | 0.00 | 10.0 | 10 | 466 | 17.5 | AgentTimeoutError |
| systemd-log-monitoring | easy | oracle | 0.00 | 15.0 | 15 | 1887 | 39.1 | AgentTimeoutError |
| vimscript-vim-quine | easy | oracle | 0.00 | 15.0 | 15 | 3036 | 50.0 | AgentTimeoutError |
| acl-permissions-inheritance | medium | oracle | 1.00 | 2.4 | 15 | 171 | 4.3 |  |
| api-endpoint-permission-canonicalizer | medium | oracle | 0.00 | 15.0 | 15 | 238 | 21.7 | AgentTimeoutError |
| bandit-delayed-feedback | medium | oracle | 0.00 | 15.0 | 15 | 893 | 28.1 | AgentTimeoutError |
| build-system-task-ordering | medium | oracle | 0.00 | 15.0 | 15 | 160 | 32.6 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | oracle | 0.00 | 15.0 | 15 | 901 | 24.8 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | oracle | 1.00 | 3.1 | 15 | 337 | 5.8 |  |
| distributed-test-execution-scheduler | medium | oracle | 1.00 | 20.0 | 20 | 1618 | 36.8 | AgentTimeoutError |
| fix-js-network-controller | medium | oracle | 1.00 | 7.1 | 10 | 292 | 12.4 |  |
| industrial-kiln-controller | medium | oracle | 0.00 | 10.0 | 10 | 121 | 10.2 | AgentTimeoutError |
| iot-device-registration-server | medium | oracle | 1.00 | 12.5 | 60 | 1000 | 22.9 |  |
| malicious-package-forensics | medium | oracle | 0.00 | 15.0 | 15 | 724 | 18.0 | AgentTimeoutError |
| maven-slf4j-conflict | medium | oracle | 1.00 | 10.0 | 10 | 1158 | 18.1 | AgentTimeoutError |
| monorepo-changelog-cli | medium | oracle | 0.00 | 10.0 | 10 | 276 | 20.6 | AgentTimeoutError |
| mtls-cert-rotation | medium | oracle | 0.00 | 10.0 | 10 | 667 | 20.7 | AgentTimeoutError |
| neural-architecture-search-final | medium | oracle | 0.00 | 13.3 | 13 | 350 | 17.6 | AgentTimeoutError |
| neutron-submission | medium | oracle | 0.00 | 15.0 | 15 | 245 | 17.2 | AgentTimeoutError |
| pgn-chess-repair-puzzles | medium | oracle | 0.00 | 57.9 | 60 | 17258 | 191.4 | UnknownApiError |
| python-api-rate-limit | medium | oracle | 0.00 | 15.0 | 15 | 635 | 29.4 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | oracle | 1.00 | 15.0 | 15 | 2229 | 34.4 | AgentTimeoutError |
| sales-data-csv-analysis | medium | oracle | 1.00 | 7.7 | 60 | 292 | 15.1 |  |
| scan-linux-persistence-artifacts | medium | oracle | 0.00 | 15.0 | 15 | 369 | 31.3 | AgentTimeoutError |
| security-breach-incident-response | medium | oracle-partial | 0.72 | 11.0 | 60 | 1669 | 25.2 |  |
| security-incident-log-analysis | medium | oracle-partial | 0.88 | 6.4 | 60 | 942 | 11.2 |  |
| sign-vector-game | medium | oracle | 0.00 | 15.0 | 15 | 514 | 36.0 | AgentTimeoutError |
| task-xxe-exploit | medium | oracle | 1.00 | 5.0 | 5 | 697 | 11.2 | AgentTimeoutError |
| todos-api | medium | oracle | 0.00 | 8.3 | 8 | 212 | 17.2 | AgentTimeoutError |
| tsl-test-case-generation | medium | oracle-partial | 0.83 | 15.5 | 60 | 1515 | 64.4 |  |
| anomaly-detection-ranking | hard | oracle | 1.00 | 5.0 | 60 | 1231 | 10.5 |  |
| application-debug | hard | oracle | 1.00 | 7.5 | 10 | 490 | 16.7 |  |
| auth_token_race_condition | hard | oracle | 1.00 | 1.3 | 10 | 140 | 1.4 |  |
| bash-log-processor-fix | hard | oracle-partial | 0.75 | 17.5 | 60 | 868 | 32.3 |  |
| chained-forensic-extraction_20260101_011957 | hard | oracle | 1.00 | 7.4 | 30 | 845 | 17.1 |  |
| container-registry-optimization | hard | oracle | 0.00 | 5.0 | 5 | 84 | 0.3 | AgentTimeoutError |
| db-migration-local-storage | hard | oracle | 1.00 | 8.9 | 10 | 979 | 17.5 |  |
| fix_async_worker_queue | hard | oracle | 1.00 | 4.6 | 10 | 621 | 10.0 |  |
| mech-system | hard | oracle | 0.00 | 15.0 | 15 | 513 | 33.5 | AgentTimeoutError |
| multi-labeller | hard | oracle | 0.00 | 15.0 | 15 | 308 | 33.7 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | oracle | 0.00 | 6.0 | 6 | 768 | 11.5 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | oracle | 0.00 | 13.3 | 13 | 479 | 23.5 | AgentTimeoutError |
| service-deployment-wave-planner | hard | oracle | 1.00 | 30.0 | 30 | 10357 | 139.1 | AgentTimeoutError |
| submission_a63937a5_20251224_152124 | hard | oracle | 1.00 | 18.2 | 30 | 3102 | 51.0 |  |
| token-auth-websocket | hard | oracle | 1.00 | 25.9 | 30 | 10954 | 112.3 |  |
