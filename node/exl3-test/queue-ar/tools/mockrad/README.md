# mockrad: a no-GPU mock of the Radiance server (DS-V4.1 queue)

`queue-ar/tools/mockrad/` on gpu02. It runs a staged item's body end to end before the item is queued. A DRYRUN only prints
commands. Under the mock, every block, parser and summary stage runs against live, plausible server output. A crash in a
helper (`TypeError`), a wrong anchor or a guard that VIRHEs every block shows up in minutes, on CPU.

## Run it on a new item (the exact command)

```bash
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad
./mockrun.sh --name <short-name> --timeout 900 \
  --env <PFX>_DUR=12 --env <PFX>_WARM=4 --env <PFX>_ORDER0=AS ... \
  items/57-<item>.sh        # or the body itself: <dir>/dsm-<x>.sh (paths relative to queue-ar/dsmain, or absolute)
```

- **Knobs.** Use `--env` with the body's own knobs to keep a run at 5 to 10 minutes: shorter windows (`*_DUR`, `*_WARM`),
  fewer blocks (`*_ORDER*`), smaller depths (`*_DEPTHS`, `*_GDEP`, `*_CDEPTH`) and fewer tokens. Keep every stage the body
  has, so that each parser runs at least once.
- **Output.** Example: stack5 with `XS5_DUR=12 XS5_WARM=4 XS5_ORDER0=AS XS5_ORDER1=T XS5_PORDER=AS XS5_LORDER=A
  "XS5_DEPTHS=12000 6000" XS5_LDEPTH=6000 XS5_GDEP=4000 XS5_CDEPTH=6000 XS5_CNTOK=48` and `--decode-scale 0.3` runs in
  about 6 minutes.
- **Report.** The report is printed and saved as `reports/<name>.txt`. The sandbox stays at `/tmp/mockrad/<name>`, holding
  the body's output dir, its RESULT log (`home/gpu/ds41/exl3-test/runs/ds41-exl3.log`), `body.out` and the mock's logs in
  `state/`.
- **Exit code.** 0 means clean: no traceback, no bash error and no VIRHE. 1 means there are findings.
- **Swapping a helper.** To test an old or alternative helper without touching the real file, use
  `--swap gpu/ds41/exl3-test/queue-ar/dsmain/stack5/s5blocks.py=gpu/ds41/exl3-test/queue-ar/dsmain/stack5/s5blocks.py.bak-20261112-spshadow`.
- **Firing a guard on purpose.** Use `--fault dead=psk`, `leak=engdd`, `noapply=ds41_psk`, `wrong=draft_occ:5`,
  `refuse=sched_noa`, `cached=40`, `nocache`, `differ=ds41_psk`, `err500=0.05`, `stall=2`, `mismatch`, `drop=psk`,
  `health=503` or `slow=3`. For example, `cached=40` must make a newcomer-freshness guard VIRHE, and `dead=psk` must make
  every S block's counter guard FAIL.

## What the report lists

| Section | Contents |
|---|---|
| TRACEBACKS | Every Python traceback, from the output dir's files, `body.out` and the log, with its exception and innermost frame |
| BASH ERRORS | Unbound variable, syntax error, command not found, integer expression and similar errors |
| VIRHE LINES | Grouped VIRHE lines. One that compares with a real-run reference is labelled `mock: real ref` and is expected |
| MISSING | Lines with "No such file", Errno 2, "missing" or "not compared" |
| MOCK SIDE | Refused connects, writes and commands; docker calls the mock does not emulate; unknown endpoints; fn4 POSTs refused (the body's own range probes are counted apart); the leak check |

## What the mock covers

`mockrad.py` uses the standard library only and listens on 18090 and up, never 8090. The schemas come from real int8
snapshots in `templates/ds41/`.

- **Read endpoints.** `/stats` (psk, dspark_pfctx, engdd, experts, link, steps*, step_ns*, draft_positions), `/admin/fn4`
  GET and POST, `/metrics`, `/health`, `/v1/models`, `/tokenize` and `/reset_prefix_cache`.
- **Switches.** POSTs are range-checked: 400 on an unknown or out-of-range value. A switch is applied at the next engine
  step; `epoch_bump` adds 1 to the epoch and `applied_step` is set.
- **Completions.** `/v1/chat/completions` and `/v1/completions` work streamed (SSE) and plain, and `/completion` is
  supported too.
- **Engine.** It plans steps like the scheduler: decode rows of 1 plus drafted tokens, prefill chunks up to MNBT,
  MIXED chunks capped by `sched_mixed_tok`, and MAXSEQS admission.
- **Counters.** Every counter a body or parser reads advances according to the switch state. That covers PSK, L20,
  PSK-DS, E1, NOA, SA, HB, PSK-CR and tail, plus draft, sample and lend.
- **Answers.** Text is deterministic, taken from a hash of the prompt, so identity compares work. Copy and edit prompts
  are echoed. The stock warm-up and greedy-set (d0) answers are replayed, so those compares are IDENTICAL.
- **Token counts.** Counts are calibrated on real numbers: deepq 16k gives 15777 and mb's prompts span 860..1520.
  Prefix hits are floored to the 2048 checkpoint grid.
- **Engine log.** It contains a real start log, the body's own LITS literals, the runtime lines (recorded pass,
  rule-w; none under the BI tracer), fn4, request and reset lines, and step lines with `docker logs -t` timestamps. With
  `RADIANCE_PROFILE_EVERY` set, it also holds the real per-op table.
- **Which start log.** `mockrun.sh` lists the 60 newest real `runs/xdsm/*/start-*.log`. `mockrad.py` serves the one
  whose start-time env-name dump (`I   RADIANCE_...`) matches the container's env best, with rare names weighted most. A
  combination no real start had (cc: nvfp4 + KVR) gets the missing name's lines from the best log that has it.
- **Ports.** Several runs can go at once. Each run reserves its own port (`/tmp/mockrad/.ports`), never 8090.
- **Deploying a change.** Deploy a changed tool with write-to-temp + `mv`. A running `bash mockrun.sh` reads its file as
  it goes, so an in-place copy breaks a run in progress.

**`mockrun.sh`** creates no edits anywhere. It sets up the run in five steps:

1. **Sandbox HOME.** It builds a symlink farm with `mkfarm.py`: code is copied and references are linked. The output dir
   and the log start fresh.
2. **Shims.** It puts shims first on PATH:
   - `docker` never calls the real docker. Its `compose up` starts the mock with the yaml's environment and env_file. It
     emulates `logs`, `inspect` and `ps`, and runs image-literal checks against a permissive image.
   - `curl` reaches only the mock.
   - sudo, ssh, pkill, systemctl and the power tools are refused.
3. **Python guard.** Every `python3` is a sandbox venv whose `.pth` loads `mockguard.py`. `:8090` goes to the mock and
   other sockets are refused. Writes outside the sandbox, GPU sysfs reads and `posix_fadvise` are refused too.
4. **DRYRUN pass.** A DRYRUN finds the output dirs.
5. **Real run and report.** It runs the body niced, scans the results and checks the farm's real files for leaks.

## Limits (what it cannot catch)

- **Engine behaviour.** Any real engine behaviour: speed, exactness, numerics, lever gains, whether a counter really moves
  on the image. The mock moves the counters the way the guards expect, so a guard that is wrong about the engine passes.
  With `--fault` you can at least check that the guard can fire.
- **Image-specific output.** New start-log lines, new `/stats` or `/admin/fn4` fields and new switches of a not-yet-built
  image. An unknown switch gets 400, which is reported. Add the field to `templates/ds41/*.json` if the item relies on it.
- **GPU-side artefacts.** The bitrace `.bin` dumps (`RADIANCE_DS41_BITRACE`), miss logs, KLD dumps, stamps and DA taps.
  A guard that needs them VIRHEs under the mock, so read those VIRHE lines as expected. Also GPU selftests and benches
  (`docker run --device ...`), which are reported as not emulated.
- **Identity against real references.** Compares with real-run references other than the warm-up and greedy set differ
  (`mock: real ref`). Timing-dependent logic only meets mock timings: budgets, trims and thermal lines.
- **Other models.** Flash-Next (fn) items would need FN templates (`MOCKRAD_TEMPLATES`).
