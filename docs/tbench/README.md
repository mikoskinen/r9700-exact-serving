# Terminal-Bench (TBLite) for the local models, through the LiteLLM test proxy

Agentic benchmark for comparing local engines (production Qwen3.8 Flash-Next vs Sabi-Noema 27B and Muse Glimmer 30B)
end to end: **agent -> LiteLLM test proxy (:4001) -> local engine**. Task containers run here on fedora-homeamd
(32 cores, 62 GB RAM, 1.7 TB free); xeon only serves the model. Prepared and validated 2026-10-21. No model has been
evaluated yet.

## What was chosen and why

| Variant (Oct 2026) | Tasks | Notes |
|---|---|---|
| Terminal-Bench 1.0 | 80 | old `tb` CLI, archived (saturated) |
| Terminal-Bench 2.0 | 89 | Nov 2025, Harbor registry `terminal-bench@2.0`; frontier models score below ~65% |
| Terminal-Bench 2.1 | 89 | same tasks as 2.0, 26 fixed (bugs, timeouts, reward hacking) |
| Terminal-Bench 3.0 / 4.0 | 74 / 66 | 4.0: 8 h agent timeout per task, 11 multi-container tasks, 3 need an H100. Not feasible here |
| **OpenThoughts-TBLite** ("Terminal-Bench Lite") | **100** | Feb 2026, OpenThoughts Agent team + Snorkel AI + Bespoke Labs, Apache-2.0. Difficulty-calibrated with Claude Haiku 4.5 (40 easy / 26 medium / 26 hard / 8 extreme), correlation r = 0.911 with TB2, 2.6-8x faster than TB2. Built for non-frontier models |

"Terminal-Bench Lite" in vendor tables is, as far as I could find, this OpenThoughts-TBLite set. No primary source used
that exact name, so treat the mapping as likely, not certain. It is in the Harbor registry as `openthoughts-tblite@2.0`
(source `github.com/open-thoughts/OpenThoughts-TBLite` @ `f075e463`). It is the right size for 27-35B local models:
TB2's hard tasks give such models only ~10-25% and run 3-8x longer.

**Harness:** Harbor 0.24.0 (`harbor run`, Apache-2.0), the official successor of the `tb` CLI. It is installed in
`.venv` (Python 3.13, LiteLLM 1.106.1 inside), and the tasks run in local Docker.

**Task set used by default: 60 validated tasks** (`tblite-tasks-validated60.txt`). These are the 100 tasks minus
`tblite-exclude.txt`:
* 14 tasks whose reference solution fails here (deterministic, re-checked). Reasons are in `tblite-exclude-broken.txt`:
  upstream dependency drift, wrong reference outputs, a verifier `npm install` failure, and so on.
* 26 tasks that ship no reference solution (`solve.sh` only prints "no solution written"). Their images build and their
  graders run and score an empty answer as 0, but the oracle cannot show they are solvable.
* The 60 kept are 56 tasks that the oracle passes with reward 1.0, plus 4 partial-credit tasks where the reference
  scores 0.61-0.92.
* For the 86-task set (only the broken ones excluded): `EXCLUDE_FILE=tblite-exclude-broken.txt`.

Scores on either subset are **not comparable to published TBLite numbers**. Those use other harnesses and agents, and
all 100 tasks. Compare our models with each other.

## Layout

```
run-tb.sh                       run one benchmark (label, agent, model alias, concurrency)
tools/tbsummary.py              summary / comparison of runs (also called by run-tb.sh)
tools/proxy_check.py            preflight: alias -> tier, parallel limits, no :8090, no fallback (reads proxy config)
tools/netaudit.sh, netaudit_report.py, netaudit-image/   optional tcpdump audit of every task container
tools/task_meta.py, tblite-meta.tsv                      per-task timeouts / resources / difficulty
datasets/openthoughts-tblite/   the 100 tasks (harbor dataset download openthoughts-tblite@2.0)
agent-bin/claude-2.1.303        pinned Claude Code binary (copy of the host's 2.1.303), bind-mounted read-only into tasks
tblite-exclude.txt, tblite-exclude-broken.txt, tblite-tasks-validated60.txt, tblite-pilot10.txt
runs/<label>/                   job/ (Harbor job dir), run-meta.json, harbor.log, proxy.log, summary.md, [netaudit/]
runs/oracle-tblite-n16, runs/oracle-recheck-fails-n6     oracle validation
runs/dryrun-*                   chain checks without a model backend (all trials fail by design)
```

## GPU-window procedure

```bash
# 0. engine(s) up on xeon (test ports :8091 flash / :8094+:8095 sabi). Then the TEST proxy with parallel limits >= n.
#    Claude Code may run subagents in parallel, so leave headroom: 16 for n=8, 32 for n=16.
cd ~/litellm-test && ./run.sh flash --all-parallel 32        # or: ./run.sh sabi --all-parallel 32 [--sabi-model ...]

# 1. pilot / gate (10 tasks, ~15-30 min). Check summary.md: proxy failures, non-local deployments, auxiliary calls
cd ~/gpu/tbench
NETAUDIT=1 TASKS_FILE=tblite-pilot10.txt ./run-tb.sh pilot-flash-cc claude-code claude-sonnet-4-5 8

# 2. full runs (60 validated tasks); put the engine model in the label, because the alias is the same for every model
./run-tb.sh flash-cc-n8   claude-code claude-sonnet-4-5 8
./run-tb.sh flash-cc-n16  claude-code claude-sonnet-4-5 16
./run-tb.sh flash-t2-n8   terminus    claude-sonnet-4-5 8

# 3. compare (paired per task, bootstrap CI of the difference, McNemar on strict passes)
.venv/bin/python -I tools/tbsummary.py runs/flash-cc-n8 runs/sabi-cc-n8 runs/muse-cc-n8

cd ~/litellm-test && ./stop.sh
```

`run-tb.sh <label> <claude-code|terminus> [model-alias=claude-sonnet-4-5] [n-concurrent=8]`. Knobs (environment):
`TASKS`, `TASKS_FILE`, `N_TASKS`, `ATTEMPTS` (harbor -k), `MAX_RETRIES` (re-run a trial that ended in an exception),
`TIMEOUT_MULT` (agent timeouts only), `NETAUDIT=1`, `PREFLIGHT_CALL=0`, `FORCE=1`, `EXCLUDE_FILE`, `CC_VERSION`,
`CC_DISALLOWED_TOOLS`, `PROXY_CONTAINER`. Re-running the same command with the same label **resumes** the job:
finished trials are kept, and unfinished trial dirs are deleted and rerun. Harbor refuses a resume if any option
changed. Use a new label for a new configuration.

What `run-tb.sh` does:
1. **Preflight.** Checks `/health/liveliness` on 127.0.0.1:4001 and reads `/app/config.yaml` from the running test
   container. The alias must resolve to a local `hosted_vllm` tier with no fallback and no `:8090`, and every row of
   that tier must have `max_parallel_requests` >= n. Otherwise it aborts; LiteLLM answers the excess with an
   immediate 429. For claude-code it also checks that a task container can reach `http://<tailscale-ip>:4001`. Last, it
   sends **one** tiny chat request (skip with `PREFLIGHT_CALL=0`).
2. **Run.** Harbor runs in a subshell from which every exported variable is removed except HOME/USER/LANG/TERM/DOCKER_*.
   A calling Claude Code session's `ANTHROPIC_BASE_URL` (production :4000) or tokens therefore cannot leak in.
   * **claude-code:** `ANTHROPIC_BASE_URL=http://<tailscale-ip>:4001` (127.0.0.1 inside a container is the container
     itself) and `ANTHROPIC_API_KEY=<master key>`. Harbor sets `ANTHROPIC_MODEL`, the opus/sonnet/haiku defaults and
     `CLAUDE_CODE_SUBAGENT_MODEL`, all to the alias. `CLAUDE_CONFIG_DIR` is a fresh dir per trial inside the container,
     and the host `~/.claude` is never mounted. Settings: `--permission-mode=bypassPermissions`, WebSearch/WebFetch
     disallowed, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `DISABLE_AUTOUPDATER/TELEMETRY/ERROR_REPORTING/BUG_COMMAND=1`,
     `DISABLE_NON_ESSENTIAL_MODEL_CALLS=1`. The Claude Code binary 2.1.303 is mounted read-only at
     `/usr/local/bin/claude`, so Harbor's version check matches and nothing is downloaded (agent setup 0.2 s).
   * **terminus (Terminus-2):** runs in the Harbor process on the host. It uses LiteLLM `openai/<alias>` with
     `api_base=http://127.0.0.1:4001/v1` and `OPENAI_API_KEY`, `model_info` (229k in / 32k out) taken from the proxy
     config, and `x-claude-code-agent-id: <trial>`, so the proxy affinity pins each trial to one server. The key never
     enters a container.
3. **After the run.** Appends the proxy's CCHINT/QWENAFF/PFXGATE/LOCALFIX lines for the run window to `proxy.log`.
   Redacts the key wherever it appears in run files (it can only get there if the agent prints its environment) and
   lists the redacted files in `key-redactions.txt`. Writes `summary.md`.

`summary.md` contents: score (mean reward, Harbor's metric) with a bootstrap 95% CI; strict pass rate (reward = 1)
with a Wilson CI; wall time and trials/hour at the run's concurrency; agent time per trial; Harbor-reported tokens;
exceptions; proxy requests/tokens/failures per deployment, with any **non-local deployment** and **auxiliary-class**
requests flagged; per-difficulty and per-task tables. With 60 tasks the CI half-width is about ±12 pp, so differences
below ~15 pp need repeated attempts (`ATTEMPTS=3`) or the paired comparison.

## Validation done (2026-10-21, no model involved)

* **Oracle**, 100 tasks at n=16: all 100 images built and all verifiers ran. Results: 56 reward 1.0, 4 partial,
  14 reference solutions fail, 26 have no reference solution. The 18 non-passing tasks with references were re-run at
  n=6 with identical rewards. Wall time 13 min including cold image builds; a rebuild from cache takes 6-23 s per task.
  Verifier time on the 60 tasks: median 9 s, p90 29 s, max 226 s.
* **Chain without a model**, with the test proxy in flash mode and the engines down:
  * claude-code: proxy auth works (`apiKeySource: ANTHROPIC_API_KEY`). Every request went to `cc-sonnet` /
    `cc-opus` -> `qwen3.8-flash-next` at `:8091` and failed with a connect error, with no fallback.
  * terminus: same path through `/v1/chat/completions`.
  * tcpdump audit of the containers: in the agent phase, Claude Code connected **only to <tailscale-ip>:4001** and made
    **no DNS lookups** (no anthropic.com, statsig, sentry).
  * Claude Code reports `analytics_disabled: true` and only builtin plugins. No key was found in any run file.
  * The test proxy was stopped again afterwards.

## Runtime estimates (60 validated tasks; calibrate with the pilot, whose summary prints trials/hour)

Sum of agent timeouts: 20.2 h (8 tasks have 60 min, the median is 15 min). Overhead per trial is ~0.5-1 min (cached
build, start, verifier). The wall time can never be shorter than the longest single trial (~60 min).

| | n=8 | n=16 |
|---|---|---|
| upper bound (every trial hits its timeout) | 2.5 h | 1.3 h + tail ≈ 1.5 h |
| Flash-Next (fast MoE, both GPUs) | ~1.5-2 h | ~1-1.5 h |
| Sabi-Noema 27B / Muse Glimmer 30B (dense; slower per stream) | ~2-2.5 h | ~1.3-1.5 h |
| pilot10 (Σ timeouts 2.25 h) | 15-30 min | ~15 min |
| 86-task set (Σ timeouts 46.2 h) | 3-5 h (max 5.8) | 2-3 h (max 2.9 + tail) |

Assumptions: published TBLite runs average ~20-25 sandbox-minutes per task with fast cloud APIs (Haiku 4.5: 76 min at
32 concurrent). Local models will spend 40-80% of their timeout budget. Terminus is usually a bit faster than Claude
Code (smaller prompts, no subagents). A full matrix of 3 models x 2 agents x 2 concurrency levels is ~20-25 GPU-hours,
so gate first: pilot, then n=8 claude-code for all three models, and the rest only where needed.

## Risks and caveats

* **Speed confounds quality.** Timeouts are wall-clock. A slower engine, or n=16 instead of 8, completes fewer turns
  per task and scores lower. That is fair for "what works in our setup", but for model quality alone consider
  `TIMEOUT_MULT=2-3` (the upper bound scales with it). Report n with every score.
* **Auxiliary calls go to the cloud.** The proxy's `local_fixes` reroutes Claude Code requests with
  `x-claude-code-request-class: auxiliary` to `cc-aux` (DeepSeek on Fireworks), as in production. That costs money and
  means a few non-local completions. None were seen in the dry runs, but those never got a successful first turn. The
  pilot's `summary.md` shows the count ("auxiliary-class requests", "proxy non-local deployments"). If any appear, the
  fix is a test-proxy change, which I did not make (read-only).
* **No Anthropic fallback is possible.** There is no real Anthropic key in any container. Claude Code's only endpoint is
  the proxy, `--fallback-model` is unset, and the test config has no fallbacks for local tiers (checked by
  `proxy_check.py`). Claude Code retries 5xx/429 itself (~14 requests/min per trial while the engine was down).
* **Internet is required** during the run: verifiers install pytest/uv (631 outbound connections in the oracle run).
  Agents also legitimately use pip/apt/maven, and Terminus-2 apt-installs tmux at setup. An outage turns into false
  failures. Base layers are cached (~24 GB of Docker build cache, kept 60 days unless `docker builder prune`).
* **The key reaches containers (claude-code only).** It is the proxy master key (LAN/tailnet only). It is visible in the
  host `ps` output of `docker exec -e ...` during a trial, and an agent that dumps its environment writes it to trial
  logs. The run script redacts such occurrences afterwards.
* **Version pinning.** Claude Code is pinned at 2.1.303, the host's version, which the proxy hooks are tuned for.
  Upgrading means copying the new binary to `agent-bin/claude-<v>` and setting `CC_VERSION`. Harbor 0.24.0 and the
  dataset commit are pinned by the venv and `datasets/`. The task Dockerfiles themselves pull unpinned
  `apt`/`pip` packages at build time, but the build cache freezes them.
* **Claude Code features.** It has Task/Workflow subagents, so concurrent requests can exceed n (hence the parallel
  headroom). Disable them with `CC_DISALLOWED_TOOLS=Workflow,Task` if needed. The proxy's prefix gate serializes the
  first turns of agents that share the same ~20k-token system prompt, as in production, so the first minutes of a
  claude-code run are slower.
* **Partial-credit graders.** Some tasks give partial credit. "Score" is the mean reward and "strict pass" counts only
  reward 1. Both are reported.
* **Proxy log.** `proxy.log` contains every request in the window, including traffic from anything else using the test
  proxy at the same time. Keep the proxy dedicated during a run.

## Sources

tbench.ai/docs (TB 4.0 run page); harborframework.com/docs/datasets/running-tbench; Harbor registry
(`registry.json`: terminal-bench 2.0 = 89 tasks, openthoughts-tblite 2.0 = 100 tasks);
github.com/open-thoughts/OpenThoughts-TBLite and openthoughts.ai/blog/openthoughts-tblite; EvalScope TB 4.0 page
(66 tasks, 8 h timeout, 3 H100 tasks); arXiv 2601.11868 (Terminal-Bench paper, ICLR 2026).
