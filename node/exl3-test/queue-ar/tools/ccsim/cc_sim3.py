#!/usr/bin/env python3
"""cc_sim.py -- Claude-Code-like load: N agents, multi-turn sessions, long growing context, prefix-cache hits (stdlib only).

cc_sim3 (10.11.2026, fn4): a REPLAY method for A/B runs with MTP on, where outputs are not batch-invariant (kernel forms
follow the row count, so 9 of 16 users diverged on their first turn in 123 and every later prompt was built from the
run's own outputs). --record: the run also writes <out>.outputs.jsonl, each request's reply (reasoning, content, tool
calls, finish) and its session end flag. --replay REF.jsonl (REF recorded, --trace-seeded, --cpt-fixed, --turns): no
live session state -- every request's messages are built offline from REF's replies, so all replays send the same
prompts request for request (only the system prompt's run-id line differs, so replays do not hit each other's prefix
cache), each request is issued OPEN-LOOP at REF's start time t (not after the previous turn), max_tokens = REF's
generated length with the natural stop (no ignore_eos). --pair-replay A B: per-request paired rates over the kept
turns -- TTFT, decode tok/s, end-to-end time (geometric mean ratio B/A with a t 95 % interval of the mean log ratio),
draft acceptance (mean difference, t 95 %), the prompt-length check, and work ratios -- as one PAIR_REPLAY json line.
--replay-closed (with --replay; 10.11.2026, after 139): the same prompts, CLOSED-LOOP per user -- a user's first request at
REF's start time, every later one when that user's previous reply has ended IN THIS RUN plus REF's gap after it (REF's
think / tool time). An open-loop replay keeps REF's arrival times whatever this server does: an arm slower than REF's
falls behind, a session's next turn arrives before its previous reply has ended (so its prefix is not cached yet and it
is prefilled again), the queue and the extra prefill feed each other, and TTFT becomes queue wait (139: 170 of 219
requests overlapped, 2.6x the uncached tokens). Closed-loop keeps every user at most one request in flight as REF did.

  cc_sim.py --urls http://127.0.0.1:8000 --users 10 --duration 1800 --warmup 300 --key-env RADIANCE_API_KEY
  cc_sim.py --urls http://127.0.0.1:8094,http://127.0.0.1:8095 --route least-loaded --users 12 --key-env RADIANCE_API_KEY
  cc_sim.py --summarize runs/ccsim-1102-1830.jsonl [--warmup 300]     # re-summarize a finished run

Model of one user: Claude Code sessions back to back. A session starts with [system: unique session line + agent
instructions + environment + project files] + `--tools` tool schemas + [user: task], U(--start-min, --start-max) tokens.
Each turn = one streaming /v1/chat/completions (model = first /v1/models id, reasoning on like conc_multi.py: top-level
reasoning_effort=medium). --gen-mode natural (default): no ignore_eos, max_tokens = --gen-max; the model reasons and
stops at its own tool call like in production. --gen-mode forced: ignore_eos + max_tokens ~ LN(--gen-median,
--gen-sigma) = exact lengths, but on Radiance the tokens after the model's natural stop are junk that collapses MTP
draft acceptance (smoke 2.11.: 0.02-0.10 vs 0.4-0.75) and halves per-stream decode, so use it only for controlled
length sweeps. --gen-mode reflen --ref REF.jsonl (cc_sim2, 9.11.2026): each request's max_tokens is the length the
reference run (a natural, --turns, --trace-seeded run of the same args) generated for the same (user, user turn), so
every arm does the reference's work and never runs far past a natural stop; --reflen-eos stop (default): no ignore_eos,
a reply that stops earlier stops (logged "short"); ignore: ignore_eos up to that length (exact lengths). Per request
the class is logged (rl: exact = the reference length with a natural finish, cut = stopped by the length where the
reference stopped naturally -- it would have run past, short, noref = no reference request), and a REFLEN line sums them. The reply (reasoning + content) goes back into the history as an assistant message with the model's
own tool call(s) (or one fabricated Read / Bash / Grep / Edit / Glob / TodoWrite call if it made none), followed by
role=tool result(s) of lognormal total size built from the Python stdlib sources (cat -n file reads, grep hits, diffs,
pytest output, prose for other tools); a reply that ends with finish=stop gets a short user follow-up instead. Then
the agent "runs the tool" (think time) and sends the next turn = previous prompt + reply + tool result, so the
server's prefix cache hits like production. When the next prompt would pass --ctx-cap the session ends and a new one
(new prefix = cold prefill) starts. All synthetic content is deterministic from --seed; the system prompt's first line
also carries a per-run id (default: start time, --run-id to fix it) so a re-run does not hit the previous run's cache
(Radiance --prefix-cache-ttl is hours). Note: the chat template renders the tool schemas BEFORE the system text, so
the --tools block (identical in every session, like production's fixed 109 Claude Code tools) is a shared prefix:
a new session's first turn gets ~12k tokens cached with the default 109 tools; --tools 0 removes that sharing.
--preage (default on): each user's first session is pre-filled with synthetic history to U(start, 0.95 x cap) tokens,
so the run starts with the steady-state context mix instead of every agent at 20-40k (that first request is a cold
prefill; keep it inside --warmup). --shared-prefix K: additionally the first K tokens of the system text are
identical across the run's sessions (same Claude Code system prompt); default 0 = unique after the tool block.

Defaults calibrated 2.11.2026 on the gpu02 production engine log (167 Claude Code requests via LiteLLM, 19 min,
2 sessions, prompt p50 109k / max 168k, cache 94 %, finish tool_calls 164/167): generated p50 300 / p90 1063 /
max 7.8k (natural mode leaves this to the model; forced mode: lognormal median 350, sigma 1.0; both capped at
--gen-max 13000); tool result per turn (prompt growth - previous gen) p50 288 / p90 1243 / max 4.9k -> lognormal
300, sigma 1.2, 20..8000; think (reply done -> next request) p50 0.8 / p90 2.0 / max 80 s -> lognormal 0.8, sigma 1.0,
0.2..90 s; reasoning stays in the history (growth >= previous gen in 137/137 turns); 109 tools per request.
The heavier "0.5-8k tool result, 2-15 s think" variant: --tool-median 2000 --tool-sigma 0.8 --tool-min 500
--think-median 0 --think-min 2 --think-max 15 (--think-median 0 = uniform between min and max).

Routing (several --urls): each session is pinned to one URL (sticky, like the LiteLLM affinity): --route hash
(default: crc32(session id) % n) or --route least-loaded (new session -> URL with fewest in-flight requests, then
fewest open sessions). Auth: --key-env NAME (preferred) or --api-key KEY; the key is never printed or logged.
Output: JSON lines (--out, default <script dir>/runs/ccsim-MMDD-HHMM[-tag].jsonl): one config line, one line per
request (t = start s since run start, url, user, session, turn, prompt_est, prompt / cached / gen / reasoning from the
server usage, ttft, total, decode tok/s, server timings), one summary line. Summary over requests started in
[--warmup, --duration): requests, req/min, TTFT p50/p95/max, per-stream decode p50, aggregate generated tok/s and
mean in-flight (both time-clipped to the window), prompt / cached tokens and hit %, prompt growth per turn.
After --duration no new request starts; running ones may finish until --duration + --drain, then are aborted.
"""
import argparse, glob, json, math, os, random, re, statistics, sys, sysconfig, threading, time, urllib.error, \
    urllib.request, zlib

STDLIB = sysconfig.get_paths()["stdlib"]
CORPUS = []  # (relative path, text) of the Python stdlib sources: real code, deterministic for a given Python
IDENTS = []  # def / class names from CORPUS, for prose and tool arguments
CUT_MARKS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<tool_call>", "</tool_call>")


def load_corpus():
    for f in sorted(glob.glob(STDLIB + "/**/*.py", recursive=True)):
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:  # noqa: BLE001
            continue
        if len(t) > 400:
            CORPUS.append((os.path.relpath(f, STDLIB), t))
    IDENTS.extend(sorted({m for _, t in CORPUS[::5] for m in re.findall(r"^\s*(?:def|class) ([A-Za-z_]\w{3,})", t, re.M)}))


# ----------------------------------------------------------------------------------------------- synthetic content
INTRO = """You are an interactive command-line coding agent. You help the user with software engineering tasks in the
repository described below: fixing bugs, adding features, refactoring, explaining code and running tests.

# Working style
- Read the relevant files before you change them; never guess at the contents of a file you have not opened.
- Prefer small, focused edits that follow the existing conventions of the codebase (naming, typing, error handling).
- After an edit, run the narrowest test command that covers the change, then the wider suite if that passes.
- Keep the user informed with short status lines; do not paste whole files back to the user.
- Use the dedicated tools (Read, Edit, Grep, Glob) instead of shell equivalents such as cat, sed or find.
- When several independent lookups are needed, issue the tool calls in parallel in a single response.
- Never commit, push or delete files unless the user explicitly asks for it.

# Tool results
Tool results arrive as tool messages. File reads are shown with line numbers (cat -n format). Grep results are shown
as path:line:text. Long outputs may be truncated; read the file directly if you need more context.

# Task tracking
Use the TodoWrite tool to plan multi-step work and mark items done as soon as they are finished.
"""
CLAUDE_MD = """# Project notes (CLAUDE.md)
- Python 3.14, stdlib only; run tests with `python -m pytest -q tests/` (fast) or `make check` (slow, includes lint).
- Public functions need docstrings and type hints; keep modules under 1500 lines.
- Error messages are lower-case and never end with a period.
- The CI runs on Linux and macOS; avoid platform-specific paths.
"""
TASKS = ["The test suite fails in {p}; find the cause and fix it without changing the public API.",
         "Add type hints and docstrings to the functions in {p}, then run the tests.",
         "Refactor {f} in {p} into smaller helpers and keep the behaviour identical.",
         "There is a bug report: {f} returns the wrong value for empty input. Reproduce it with a test and fix it.",
         "Explain how {c} and {f} interact in {p}, then propose and implement a simplification.",
         "Port the error handling in {p} to the new exception hierarchy and update the callers.",
         "Write unit tests that cover the edge cases of {f} in {p}.",
         "Profile {f} in {p} and make it faster without changing its output."]
SENTS = ["Let me check how {f} is used in {p}.", "The failing assertion comes from {f}; I need to look at {c} first.",
         "{c}.{f} returns early when the input is empty, so the caller in {p} never sees the error.",
         "I'll grep for {f} to find the other call sites.", "That matches the traceback: {f} is called with a None {v}.",
         "Next I will update {p} and rerun the tests.", "The test expects {v} to be normalized before {f} runs.",
         "There are two options: patch {f} directly or wrap it in {c}.", "The diff looks right, but {v} is still unused.",
         "Now the remaining failure is in {f}; the fixture builds {c} without {v}.", "Good, the edit applied cleanly."]
FOLLOWUPS = ["Looks good. Now add tests for the edge cases of {f}.", "Run the full test suite and fix anything that fails.",
             "Thanks. Do the same for {f} in {p}.", "Please also update the docstrings in {p}.",
             "That broke {c}; have a look at {p} again."]
TOOL_W = [("Read", 35), ("Bash", 25), ("Grep", 15), ("Edit", 15), ("Glob", 5), ("TodoWrite", 5)]
GREP_TOK = ["def ", "self.", "return ", "import ", "raise ", "if not ", "except "]


def fmt(rng, s):
    return s.format(f=rng.choice(IDENTS), c=rng.choice(IDENTS), p="src/" + rng.choice(CORPUS)[0],
                    v=rng.choice(IDENTS).lower())


def prose(rng, nchars):
    out, n = [], 0
    while n < nchars:
        s = fmt(rng, rng.choice(SENTS)); out.append(s); n += len(s) + 1
    return " ".join(out)[:nchars]


def file_lines(rng, nchars, numbered=True):
    """consecutive lines of a random stdlib file (continuing into the next file), cat -n style"""
    i = rng.randrange(len(CORPUS)); path, text = CORPUS[i]; lines = text.split("\n")
    ln = rng.randrange(max(1, len(lines) - 20)); out, n = [], 0
    while n < nchars:
        if ln >= len(lines):
            i = (i + 1) % len(CORPUS); lines = CORPUS[i][1].split("\n"); ln = 0
        s = f"{ln + 1:6}\t{lines[ln]}" if numbered else lines[ln]
        out.append(s); n += len(s) + 1; ln += 1
    return path, "\n".join(out)[:nchars]


def grep_lines(rng, nchars, pat):
    i = rng.randrange(len(CORPUS)); out, n = [], 0
    for step in range(2 * len(CORPUS)):  # bounded: a rare pattern falls back to a common one after 300 files
        if n >= nchars:
            break
        if step == 300:
            pat = "return "
        path, text = CORPUS[(i + step) % len(CORPUS)]
        for k, line in enumerate(text.split("\n")):
            if pat in line:
                s = f"src/{path}:{k + 1}:{line.strip()}"; out.append(s); n += len(s) + 1
                if n >= nchars:
                    break
    return "\n".join(out)[:nchars]


def tool_turn(rng, ntok, cpt, proj, name=None):
    """tool call (fabricated, or `name` = the model's own call) + its result of ~ntok tokens -> (name, args, result)"""
    name = name or rng.choices([t for t, _ in TOOL_W], [w for _, w in TOOL_W])[0]
    nchars = max(20, int(ntok * cpt)); root = f"/home/dev/{proj}"
    if name == "Read":
        path, body = file_lines(rng, nchars)
        return name, {"file_path": f"{root}/src/{path}"}, body
    if name == "Grep":
        pat = rng.choice(GREP_TOK) + (rng.choice(IDENTS) if rng.random() < 0.3 else "")
        body = grep_lines(rng, nchars, pat) or "No matches found"
        return name, {"pattern": pat, "path": "src", "output_mode": "content", "-n": True}, body
    if name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path, old = file_lines(rng, rng.randint(80, 400), numbered=False)
        new = old.replace("    ", "        ", 1) + "\n"
        head = f"The file {root}/src/{path} has been updated. Here's the result of running `cat -n` on a snippet:\n"
        _, body = file_lines(rng, max(0, nchars - len(head)))
        return name, {"file_path": f"{root}/src/{path}", "old_string": old, "new_string": new}, head + body
    if name == "Glob":
        i = rng.randrange(len(CORPUS)); out, n = [], 0
        while n < nchars:
            s = f"{root}/src/{CORPUS[i][0]}"; out.append(s); n += len(s) + 1; i = (i + 1) % len(CORPUS)
        return name, {"pattern": "src/**/*.py"}, "\n".join(out)[:nchars]
    if name == "TodoWrite":
        todos = [{"content": fmt(rng, rng.choice(SENTS)), "status": rng.choice(["pending", "in_progress", "completed"]),
                  "activeForm": "Working"} for _ in range(rng.randint(2, 6))]
        body = "Todos have been modified successfully. Ensure that you continue to use the todo list to track progress."
        return name, {"todos": todos}, (body + "\n" + prose(rng, nchars))[:max(nchars, len(body))]
    if name != "Bash":  # Task / WebFetch / WebSearch / mcp__* / anything else the model calls: a prose report
        return name, {"query": fmt(rng, rng.choice(SENTS))}, prose(rng, nchars)
    # Bash: pytest output, a diff or a log, ~nchars
    kind = rng.random(); mod = rng.choice(CORPUS)[0].replace("/", "_").removesuffix(".py")
    if kind < 0.5:
        cmd = f"python -m pytest -q tests/test_{mod}.py"
        out, n = [f"$ {cmd}", "============================= test session starts =============================="], 0
        while n < nchars:
            if rng.random() < 0.08:
                _, tb = file_lines(rng, rng.randint(300, 1200), numbered=False)
                s = f"FAILED tests/test_{mod}.py::test_{rng.choice(IDENTS).lower()} - AssertionError\n{tb}"
            else:
                s = f"tests/test_{mod}.py::test_{rng.choice(IDENTS).lower()} PASSED{' ' * 20}[{rng.randint(1, 99):3d}%]"
            out.append(s); n += len(s) + 1
    elif kind < 0.8:
        cmd = "git diff"
        path, body = file_lines(rng, nchars, numbered=False)
        ln = rng.randint(1, 900)
        out = [f"diff --git a/src/{path} b/src/{path}", f"--- a/src/{path}", f"+++ b/src/{path}", f"@@ -{ln},40 +{ln},41 @@"]
        out += [rng.choice("      +-") + s for s in body.split("\n")]
    else:
        cmd = f"python -m {mod} --verbose 2>&1 | tail -200"
        path, body = file_lines(rng, nchars, numbered=False)
        out = [f"DEBUG {rng.randint(10, 59)}:{rng.randint(10, 59)}.{rng.randint(100, 999)} {s.strip()}"
               for s in body.split("\n") if s.strip()]
    return name, {"command": cmd, "description": "Run " + cmd.split()[0]}, "\n".join(out)[:nchars]


def schema(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": req, "additionalProperties": False}}}


def build_tools(n):
    core = [
        schema("Bash", "Executes a given bash command in a persistent shell session with optional timeout. Use it for "
               "terminal operations like git, package managers, test runners and builds; do not use it for file reads, "
               "searches or edits, which have dedicated tools. Quote paths that contain spaces. Output longer than "
               "30000 characters is truncated. Commands run in the project root unless they change directory. "
               "Prefer running several independent commands in parallel tool calls over chaining them with &&.",
               {"command": ("string", "The command to execute"), "description": ("string", "Clear, concise "
                "description of what this command does in 5-10 words"), "timeout": ("number", "Optional timeout "
                "in milliseconds (max 600000)"), "run_in_background": ("boolean", "Run the command in the background")},
               ["command"]),
        schema("Read", "Reads a file from the local filesystem. The file_path parameter must be an absolute path. By "
               "default it reads up to 2000 lines from the beginning of the file; use offset and limit for long files. "
               "Results are returned using cat -n format, with line numbers starting at 1. Can read images, PDFs "
               "and notebooks. Reading a directory is an error; use Bash ls for that.",
               {"file_path": ("string", "The absolute path to the file to read"), "offset": ("number", "The line "
                "number to start reading from"), "limit": ("number", "The number of lines to read")}, ["file_path"]),
        schema("Edit", "Performs exact string replacements in files. You must read the file before editing it. The "
               "edit fails if old_string is not unique in the file; provide more surrounding context or use "
               "replace_all. Preserve the exact indentation as it appears after the line number prefix.",
               {"file_path": ("string", "The absolute path to the file to modify"), "old_string": ("string", "The "
                "text to replace"), "new_string": ("string", "The text to replace it with"), "replace_all":
                ("boolean", "Replace all occurrences of old_string")}, ["file_path", "old_string", "new_string"]),
        schema("Write", "Writes a file to the local filesystem, overwriting the existing file if there is one. Read "
               "an existing file before overwriting it. Prefer editing existing files over creating new ones.",
               {"file_path": ("string", "The absolute path to the file to write"), "content": ("string", "The "
                "content to write to the file")}, ["file_path", "content"]),
        schema("Grep", "A powerful search tool built on ripgrep. Supports full regex syntax, glob and type filters "
               "and three output modes: content shows matching lines, files_with_matches shows only file paths, "
               "count shows match counts. Use multiline for patterns that span lines.",
               {"pattern": ("string", "The regular expression pattern to search for"), "path": ("string", "File "
                "or directory to search in"), "glob": ("string", "Glob pattern to filter files"), "output_mode":
                ("string", "content | files_with_matches | count"), "-n": ("boolean", "Show line numbers"),
                "-i": ("boolean", "Case insensitive search"), "head_limit": ("number", "Limit output lines")},
               ["pattern"]),
        schema("Glob", "Fast file pattern matching tool that works with any codebase size. Returns matching file "
               "paths sorted by modification time.", {"pattern": ("string", "The glob pattern to match files "
               "against"), "path": ("string", "The directory to search in")}, ["pattern"]),
        schema("TodoWrite", "Create and manage a structured task list for the current coding session. Use it for "
               "multi-step tasks; mark exactly one task in_progress at a time and complete tasks immediately.",
               {"todos": ("array", "The updated todo list")}, ["todos"]),
        schema("Task", "Launch a new agent to handle complex, multi-step tasks autonomously. The agent has its own "
               "context and returns a single final report.", {"description": ("string", "A short description of "
               "the task"), "prompt": ("string", "The task for the agent to perform"), "subagent_type": ("string",
               "The type of agent to use")}, ["description", "prompt"]),
        schema("WebFetch", "Fetches content from a URL and processes it with a prompt.", {"url": ("string", "The URL "
               "to fetch"), "prompt": ("string", "What to extract from the page")}, ["url", "prompt"]),
        schema("WebSearch", "Searches the web and returns result snippets.", {"query": ("string", "The search query")},
               ["query"]),
    ]
    r = random.Random(4242); servers = ["github", "jira", "slack", "postgres", "sentry", "docs", "grafana", "k8s"]
    verbs = ["get", "list", "create", "update", "delete", "search", "describe", "run"]
    tools, seen = core[:n], set()
    while len(tools) < n:
        name = f"mcp__{r.choice(servers)}__{r.choice(verbs)}_{r.choice(IDENTS).lower()[:24]}"
        if name in seen:
            continue
        seen.add(name)
        props = {r.choice(IDENTS).lower()[:20] + str(k): (r.choice(["string", "number", "boolean"]), prose(r, r.randint(20, 70)))
                 for k in range(r.randint(1, 3))}
        tools.append(schema(name, prose(r, r.randint(60, 200)), props, list(props)[:1]))
    return tools


# ----------------------------------------------------------------------------------------------- HTTP (conc_multi style)
def headers(key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def model_id(base, key):
    req = urllib.request.Request(base + "/v1/models", headers=headers(key))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"][0]["id"]


def stream_chat(base, key, payload, hard_stop, read_timeout):
    res = dict(err="", aborted=False, usage={}, timings={}, finish=None, tf=None)
    rs, cs, tcs = [], [], {}
    res["t0"] = time.time()
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(), headers(key))
        with urllib.request.urlopen(req, timeout=read_timeout) as r:
            for line in r:
                if time.time() > hard_stop:
                    res["aborted"] = True
                    break
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    res["usage"] = ev["usage"]
                if ev.get("timings"):
                    res["timings"] = ev["timings"]
                for c in ev.get("choices", []):
                    d = c.get("delta") or {}
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if rc:
                        rs.append(rc)
                    if d.get("content"):
                        cs.append(d["content"])
                    for tc in d.get("tool_calls") or []:  # the model's own tool calls (streamed in pieces)
                        e = tcs.setdefault(tc.get("index", len(tcs)), {"id": "", "name": "", "arguments": ""})
                        f = tc.get("function") or {}
                        e["id"] = e["id"] or tc.get("id") or ""
                        e["name"] = e["name"] or f.get("name") or ""
                        e["arguments"] += f.get("arguments") or ""
                    if (rc or d.get("content") or d.get("tool_calls")) and res["tf"] is None:
                        res["tf"] = time.time()
                    if c.get("finish_reason"):
                        res["finish"] = c["finish_reason"]
    except urllib.error.HTTPError as e:
        try:
            body = e.read()[:200].decode(errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        res["err"] = f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        res["err"] = f"{type(e).__name__}: {str(e)[:200]}"
    res["t1"] = time.time(); res["reasoning"] = "".join(rs); res["content"] = "".join(cs)
    res["tool_calls"] = [tcs[k] for k in sorted(tcs)]
    return res


def clean(s):
    for m in CUT_MARKS:
        k = s.find(m)
        if k >= 0:
            s = s[:k]
    return s.strip()


# ----------------------------------------------------------------------------------------------- simulation
def lognormal(rng, median, sigma, lo, hi):
    return min(hi, max(lo, median * math.exp(sigma * rng.gauss(0.0, 1.0))))


class Router:
    def __init__(self, urls, mode):
        self.urls, self.mode, self.lock = urls, mode, threading.Lock()
        self.inflight = {u: 0 for u in urls}; self.open = {u: 0 for u in urls}

    def pick(self, sid):
        with self.lock:
            if len(self.urls) == 1:
                u = self.urls[0]
            elif self.mode == "least-loaded":
                u = min(self.urls, key=lambda x: (self.inflight[x], self.open[x], self.urls.index(x)))
            else:
                u = self.urls[zlib.crc32(sid.encode()) % len(self.urls)]
            self.open[u] += 1
            return u

    def close(self, u):
        with self.lock:
            self.open[u] -= 1

    def add(self, u, d):
        with self.lock:
            self.inflight[u] += d


class Sim:
    def __init__(self, a, urls, model, key, tools, out_path):
        self.a, self.urls, self.model, self.key, self.tools = a, urls, model, key, tools
        self.tools_chars = len(json.dumps(tools)) if tools else 0
        self.router = Router(urls, a.route); self.stop = threading.Event(); self.lock = threading.Lock()
        self.recs = []; self.cpt = a.cpt; self.out = open(out_path, "a"); self.out_path = out_path
        self.run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
        self.ref = load_ref(a.ref) if a.gen_mode == "reflen" else {}
        self.rec_out = open(out_path.replace(".jsonl", "") + ".outputs.jsonl", "a") if a.record else None
        if a.replay:
            rcfg = replay_ref_cfg(a.replay)
            # the replay's prompts are the reference's: same seed / cpt / tools / effort / context settings
            for k in ("seed", "cpt", "tools", "ctx_cap", "start_min", "start_max", "preage", "shared_prefix", "effort",
                      "drop_reasoning", "tool_median", "tool_sigma", "tool_min", "tool_max", "gen_median", "gen_sigma",
                      "gen_max", "turns", "warmup_turns"):
                if k in rcfg:
                    setattr(a, k, rcfg[k])
            if not (rcfg.get("trace_seeded") and rcfg.get("cpt_fixed") and rcfg.get("turns")):
                sys.exit("--replay: the reference must be a --turns --trace-seeded --cpt-fixed run")
            self.cpt = a.cpt
            a.gen_mode = "natural"   # the reference's lengths as max_tokens, the natural stop (no ignore_eos)
            # a run id of its own (so replays never hit each other's or the reference's prefix cache) of the reference's
            # length: the session builder sizes its fill from the head's length, and a longer id would draw other files
            if len(self.run_id) != len(str(rcfg.get("run_id", ""))):
                sys.exit(f"--replay: run id '{self.run_id}' must have the reference's length ({rcfg.get('run_id')})")
            if self.run_id == str(rcfg.get("run_id")):   # started in the reference's second: change one digit
                self.run_id = self.run_id[:-1] + str((int(self.run_id[-1]) + 1) % 10 if self.run_id[-1].isdigit() else 0)
        self.shared = ""
        if a.shared_prefix > 0:
            r = random.Random(f"{a.seed}/shared")
            self.shared = CLAUDE_MD + "\n# Reference material\n" + file_lines(r, int(a.shared_prefix * a.cpt))[1] + "\n"

    def write(self, obj):
        with self.lock:
            self.out.write(json.dumps(obj) + "\n"); self.out.flush()

    def est(self, nchars):
        return int(nchars / self.cpt)

    @staticmethod
    def msg_chars(m):
        n = len(m.get("content") or "") + len(m.get("reasoning_content") or "") + 12
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["arguments"]) + len(tc["function"]["name"]) + 30
        return n

    def new_session(self, u, k):
        a = self.a; sid = f"u{u}s{k}"; rng = random.Random(f"{a.seed}/{sid}")
        target = rng.uniform(a.start_min, a.start_max); proj = f"proj{rng.randrange(1000)}"
        user_txt = fmt(rng, rng.choice(TASKS)) + "\nStart by reading the relevant files."
        sess_line = f"[cc_sim run={self.run_id} session={sid} seed={a.seed}]\n"
        env = (f"<env>\nWorking directory: /home/dev/{proj}\nIs directory a git repo: Yes\nPlatform: linux\n"
               f"Today's date: 2026-11-02\n</env>\n")
        if self.shared:  # identical first part across the run's sessions, then the unique session line
            head = f"[cc_sim run={self.run_id}]\n" + INTRO + self.shared + sess_line + env + CLAUDE_MD
        else:
            head = sess_line + INTRO + env + CLAUDE_MD
        fill = int((target - self.est(self.tools_chars + len(head) + len(user_txt))) * self.cpt)
        parts, n = [head, "\n# Files already in context\n"], 0
        while n < fill:
            path, body = file_lines(rng, min(fill - n, rng.randint(4000, 30000)))
            s = f'<file path="src/{path}">\n{body}\n</file>\n'; parts.append(s); n += len(s)
        msgs = [{"role": "system", "content": "".join(parts)}, {"role": "user", "content": user_txt}]
        sess = dict(sid=sid, user=u, rng=rng, proj=proj, msgs=msgs, turn=0, effort=rng.choice(a.effort.split(",")),
                    chars=sum(self.msg_chars(m) for m in msgs), preaged=0)
        if a.preage and k == 0:  # synthetic history up to a steady-state context size
            goal = rng.uniform(target, 0.95 * a.ctx_cap); i = 0
            while self.est(sess["chars"] + self.tools_chars) < goal:
                g = int(lognormal(rng, a.gen_median, a.gen_sigma, 16, a.gen_max))
                self.append_turn(sess, prose(rng, int(g * 0.6 * self.cpt)), prose(rng, int(g * 0.4 * self.cpt)), f"h{i}")
                i += 1
            sess["preaged"] = i
        return sess

    def append_turn(self, sess, reasoning, content, tag, calls=None, finish=None, rng=None):
        """assistant reply + its tool call(s) (the model's own when valid, else one fabricated) + tool result(s) of
        lognormal total size; a reply that ended with finish=stop and no call gets a short user follow-up instead.
        Returns the estimated tokens added after the reply's own tokens."""
        a = self.a; rng = rng or sess["rng"]
        am = {"role": "assistant", "content": content}
        if reasoning and not a.drop_reasoning:
            am["reasoning_content"] = reasoning
        if finish == "stop" and not calls:
            um = {"role": "user", "content": fmt(rng, rng.choice(FOLLOWUPS))}
            sess["msgs"] += [am, um]; c = self.msg_chars(am) + self.msg_chars(um); sess["chars"] += c
            return self.est(self.msg_chars(um))
        ntok = lognormal(rng, a.tool_median, a.tool_sigma, a.tool_min, a.tool_max)
        valid = []
        for c in calls or []:
            try:
                ok = c["name"] and isinstance(json.loads(c["arguments"]), dict)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                valid.append(c)
        tcs, tms = [], []
        for i, c in enumerate(valid or [None]):
            name, args, body = tool_turn(rng, ntok / max(1, len(valid)), self.cpt, sess["proj"], c and c["name"])
            cid = (c and c["id"]) or f"call_{sess['sid']}_{tag}_{i}"
            tcs.append({"id": cid, "type": "function",
                        "function": {"name": name, "arguments": c["arguments"] if c else json.dumps(args)}})
            tms.append({"role": "tool", "tool_call_id": cid, "content": body})
        am["tool_calls"] = tcs
        sess["msgs"] += [am] + tms; c = self.msg_chars(am) + sum(self.msg_chars(m) for m in tms); sess["chars"] += c
        return self.est(sum(self.msg_chars(m) for m in tms) + (0 if valid else 40 + len(tcs[0]["function"]["arguments"])))

    def payload(self, sess, max_tokens):
        a = self.a
        p = {"model": self.model, "messages": sess["msgs"], "max_tokens": max_tokens, "stream": True,
             "stream_options": {"include_usage": True}}
        if self.tools:
            p["tools"] = self.tools
        if a.gen_mode == "forced" or (a.gen_mode == "reflen" and a.reflen_eos == "ignore"):
            p["ignore_eos"] = True
        if a.temperature is not None:
            p["temperature"] = a.temperature
        if sess["effort"] == "off":
            p["chat_template_kwargs"] = {"enable_thinking": False}
        elif sess["effort"] != "default":
            p["reasoning_effort"] = sess["effort"]
        return p

    def turn(self, sess, url, t_zero, hard_stop):
        a, rng = self.a, sess["rng"]
        sess["turn"] += 1
        # --trace-seeded: this turn's draws (generation length, tool result size and content) from their own stream, so a
        # reply that differs between two runs (batch-dependent numerics at temperature 0, another tool call parsed) cannot
        # shift the next turns' lengths: every run's sessions keep the same token counts and the arms see the same load
        trng = random.Random(f"{a.seed}/{sess['sid']}/t{sess['turn']}") if a.trace_seeded else None
        gen_t = int(lognormal(trng or rng, a.gen_median, a.gen_sigma, 16, a.gen_max))
        max_tokens = gen_t if a.gen_mode == "forced" else a.gen_max
        ref = self.ref.get((sess["user"], sess.get("uturn", 0))) if a.gen_mode == "reflen" else None
        if ref:
            max_tokens = max(1, ref[0])
        prompt_est = self.est(sess["chars"] + self.tools_chars)
        self.router.add(url, 1)
        try:
            r = stream_chat(url, self.key, self.payload(sess, max_tokens), hard_stop, a.read_timeout)
        finally:
            self.router.add(url, -1)
        u, tm = r["usage"], r["timings"]
        gen = u.get("completion_tokens") or self.est(len(r["reasoning"]) + len(r["content"]))
        prompt = u.get("prompt_tokens") or 0
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = tm.get("cache_n")
        tf = r["tf"]; total = r["t1"] - r["t0"]
        rec = dict(t=round(r["t0"] - t_zero, 3), url=url, user=sess["user"], session=sess["sid"], turn=sess["turn"],
                   uturn=sess.get("uturn", 0),
                   preaged=sess["preaged"], effort=sess["effort"], max_tokens=max_tokens, prompt_est=prompt_est,
                   prompt=prompt, cached=cached, gen=gen,
                   reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   ttft=round(tf - r["t0"], 3) if tf else None, total=round(total, 3),
                   decode=round((gen - 1) / (r["t1"] - tf), 1) if tf and r["t1"] > tf and gen > 1 else None,
                   finish=r["finish"], srv_ttft=round(tm["ttft_ms"] / 1000, 3) if tm.get("ttft_ms") else None,
                   srv_prompt_n=tm.get("prompt_n"), srv_prefill_tps=round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
                   draft_acc=round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
                   model_calls=len(r["tool_calls"]))
        if a.gen_mode == "reflen":
            rec["ref_gen"], rec["ref_finish"] = (ref[0], ref[1]) if ref else (None, None)
            rec["rl"] = reflen_class(rec, a.reflen_eos == "ignore")
        if r["err"]:
            rec["err"] = r["err"]
        if r["aborted"]:
            rec["aborted"] = True
        if prompt > 2000 and not r["err"] and not a.cpt_fixed:  # chars/token calibration for the content builders
            self.cpt = 0.8 * self.cpt + 0.2 * ((sess["chars"] + self.tools_chars) / prompt)
        ok = not r["err"] and not r["aborted"]
        if ok and a.record:
            rec["_out"] = {"reasoning": clean(r["reasoning"]), "content": clean(r["content"]),
                           "calls": r["tool_calls"], "finish": r["finish"]}
        if ok:
            new = self.append_turn(sess, clean(r["reasoning"]), clean(r["content"]), str(sess["turn"]),
                                   r["tool_calls"], r["finish"], rng=trng)
            # next prompt = this prompt + reply + tool result (server count when available, else the estimate)
            ctx_next = prompt + gen + new if prompt else self.est(sess["chars"] + self.tools_chars)
            rec["ctx_next"] = ctx_next
            rec["last"] = ctx_next > a.ctx_cap
        else:
            rec["last"] = True
        return rec, ok

    def user_loop(self, u, t_zero, deadline, hard_stop):
        a = self.a; rng = random.Random(f"{a.seed}/user{u}")
        if self.stop.wait(rng.uniform(0, a.ramp)):
            return
        k = n = 0
        more = lambda: not self.stop.is_set() and time.time() < deadline and (a.turns <= 0 or n < a.turns)  # noqa: E731
        while more():
            sess = self.new_session(u, k); k += 1
            url = self.router.pick(sess["sid"])
            try:
                while more():
                    n += 1; sess["uturn"] = n
                    rec, ok = self.turn(sess, url, t_zero, hard_stop)
                    if a.think_median > 0:
                        think = lognormal(rng, a.think_median, a.think_sigma, a.think_min, a.think_max)
                    else:
                        think = rng.uniform(a.think_min, a.think_max)
                    rec["think"] = round(think if ok else max(think, 5.0), 2)
                    out = rec.pop("_out", None)
                    if out is not None and self.rec_out:
                        with self.lock:
                            self.rec_out.write(json.dumps({"user": rec["user"], "uturn": rec["uturn"], "session": rec["session"],
                                                           "turn": rec["turn"], "last": rec["last"], **out}) + "\n")
                            self.rec_out.flush()
                    with self.lock:
                        self.recs.append(rec)
                    self.write(rec)
                    if self.stop.wait(rec["think"]) or rec["last"]:
                        break
            finally:
                self.router.close(url)

    def run(self):
        a = self.a
        t_zero = time.time(); deadline = t_zero + a.duration; hard_stop = deadline + a.drain
        self.write({"config": {**{k: v for k, v in vars(a).items() if k not in ("api_key",)}, "model": self.model,
                               "urls": self.urls, "run_id": self.run_id, "start": time.strftime("%F %T"),
                               "tools_n": len(self.tools), "tools_tokens_est": self.est(self.tools_chars)}})
        if a.replay and a.replay_closed:
            sched = self.build_schedule(a.replay)
            last_t = max(x["t_ref"] for x in sched)
            # a slower arm takes longer than REF: requests run until --duration (at least REF's span), then --drain
            deadline = t_zero + max(last_t + 1.0, a.duration); hard_stop = deadline + a.drain
            by_user = {}
            for x in sched:
                by_user.setdefault(x["user"], []).append(x)
            th = [threading.Thread(target=self.replay_user, args=(sorted(v, key=lambda x: x["uturn"]), t_zero, hard_stop),
                                   daemon=True) for v in by_user.values()]
            print(f"# replay of {a.replay}: {len(sched)} requests of {len(by_user)} users, CLOSED-LOOP per user (each next "
                  f"request at this run's previous reply end + the reference's gap), run id {self.run_id}", flush=True)
        elif a.replay:
            sched = self.build_schedule(a.replay)
            last_t = max(x["t_ref"] for x in sched)
            deadline = t_zero + last_t + 1.0; hard_stop = deadline + a.drain
            th = [threading.Thread(target=self.replay_one, args=(x, t_zero, hard_stop), daemon=True) for x in sched]
            print(f"# replay of {a.replay}: {len(sched)} requests over {last_t:.0f} s, open-loop at the reference's start "
                  f"times, run id {self.run_id}", flush=True)
        else:
            th = [threading.Thread(target=self.user_loop, args=(u, t_zero, deadline, hard_stop), daemon=True)
                  for u in range(a.users)]
        [t.start() for t in th]
        nxt = t_zero + a.progress if a.progress > 0 else None
        try:
            while (time.time() < deadline or a.replay) and any(t.is_alive() for t in th):
                time.sleep(max(0.2, min(1.0, deadline - time.time())))
                if nxt and time.time() >= nxt:
                    nxt += a.progress; self.progress(t_zero)
        except KeyboardInterrupt:
            print("# interrupted: stopping new requests", flush=True)
        self.stop.set()
        for t in th:
            t.join(timeout=max(0.0, hard_stop - time.time()) + 30)
        with self.lock:
            recs = list(self.recs)
        if a.turns > 0:
            s = summarize_turns(recs, a.warmup_turns, a)
        else:
            s = summarize(recs, a.warmup, min(a.duration, time.time() - t_zero), a)
        s["cpt_final"] = round(self.cpt, 3)
        self.write({"summary": s}); self.out.close()
        if self.rec_out:
            self.rec_out.close()
        print_summary(s)
        if a.gen_mode == "reflen":
            print(reflen_line(recs, a.warmup_turns), flush=True)
        print(f"# per-request lines + summary: {self.out_path}", flush=True)

    def build_schedule(self, ref_path):
        """Every request of the reference, its messages rebuilt from the reference's replies (append_turn with the
        reference's reasoning / content / tool calls / finish and the turn's own trace-seeded stream, as turn() does),
        its start time and generated length the reference's."""
        a = self.a
        recs = {}; outs = {}
        for line in open(ref_path):
            if line.strip():
                o = json.loads(line)
                if "turn" in o and "t" in o:
                    recs[(o["user"], o.get("uturn", 0))] = o
        op = ref_path.replace(".jsonl", "") + ".outputs.jsonl"
        if not os.path.exists(op):
            sys.exit(f"--replay: {op} missing (the reference must be a --record run)")
        for line in open(op):
            if line.strip():
                o = json.loads(line)
                outs[(o["user"], o["uturn"])] = o
        sched = []
        for u in range(a.users):
            k = n = 0
            while a.turns <= 0 or n < a.turns:
                sess = self.new_session(u, k); k += 1
                ended = False
                while a.turns <= 0 or n < a.turns:
                    n += 1; sess["uturn"] = n; sess["turn"] += 1
                    key = (u, n); r = recs.get(key); o = outs.get(key)
                    if r is None or o is None or r.get("err") or r.get("aborted"):
                        ended = True; break
                    trng = random.Random(f"{a.seed}/{sess['sid']}/t{sess['turn']}")
                    lognormal(trng, a.gen_median, a.gen_sigma, 16, a.gen_max)   # turn()'s gen_t draw, so trng matches
                    sched.append(dict(t_ref=r["t"], user=u, uturn=n, session=sess["sid"], turn=sess["turn"],
                                      effort=sess["effort"], msgs=list(sess["msgs"]), max_tokens=max(1, int(r["gen"])),
                                      ref=r, preaged=sess["preaged"], prompt_est=self.est(sess["chars"] + self.tools_chars)))
                    self.append_turn(sess, o["reasoning"], o["content"], str(sess["turn"]), o["calls"], o["finish"], rng=trng)
                    if o.get("last", r.get("last")):
                        break
                if ended:
                    break
        sched.sort(key=lambda x: x["t_ref"])
        return sched

    def replay_user(self, items, t_zero, hard_stop):
        """--replay-closed: one user's requests in order, each when this run's previous one has ended plus REF's gap."""
        prev_end = prev_ref = None
        for x in items:
            if prev_end is None:
                start = t_zero + x["t_ref"]
                gap = None
            else:
                gap = max(0.0, x["t_ref"] - (prev_ref["t"] + (prev_ref.get("total") or 0.0)))
                start = prev_end + gap
            if time.time() >= hard_stop:
                return
            end = self.replay_one(x, t_zero, hard_stop, start_at=start, gap=gap)
            if end is None:
                return
            prev_end, prev_ref = end, x["ref"]

    def replay_one(self, x, t_zero, hard_stop, start_at=None, gap=None):
        a = self.a
        if self.stop.wait(max(0.0, (start_at if start_at is not None else t_zero + x["t_ref"]) - time.time())):
            return None
        url = self.router.pick(x["session"])
        sess = {"msgs": x["msgs"], "effort": x["effort"]}
        self.router.add(url, 1)
        infl = sum(self.router.inflight.values())
        try:
            r = stream_chat(url, self.key, self.payload(sess, x["max_tokens"]), hard_stop, a.read_timeout)
        finally:
            self.router.add(url, -1)
            self.router.close(url)
        u, tm = r["usage"], r["timings"]
        gen = u.get("completion_tokens") or self.est(len(r["reasoning"]) + len(r["content"]))
        prompt = u.get("prompt_tokens") or 0
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = tm.get("cache_n")
        tf = r["tf"]; total = r["t1"] - r["t0"]; rf = x["ref"]
        rec = dict(t=round(r["t0"] - t_zero, 3), url=url, user=x["user"], session=x["session"], turn=x["turn"],
                   uturn=x["uturn"], preaged=x["preaged"], effort=x["effort"], max_tokens=x["max_tokens"],
                   prompt_est=x["prompt_est"], prompt=prompt, cached=cached, gen=gen,
                   reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   ttft=round(tf - r["t0"], 3) if tf else None, total=round(total, 3),
                   decode=round((gen - 1) / (r["t1"] - tf), 1) if tf and r["t1"] > tf and gen > 1 else None,
                   finish=r["finish"], srv_ttft=round(tm["ttft_ms"] / 1000, 3) if tm.get("ttft_ms") else None,
                   srv_prompt_n=tm.get("prompt_n"),
                   srv_prefill_tps=round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
                   draft_acc=round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
                   model_calls=len(r["tool_calls"]), replay=1, inflight_start=infl, ref_t=rf["t"], ref_gen=rf["gen"],
                   ref_prompt=rf.get("prompt"), ref_ttft=rf.get("ttft"), last=bool(rf.get("last")),
                   think=round(gap, 3) if gap is not None else 0.0)
        if start_at is not None:
            rec["closed"] = 1
        if r["err"]:
            rec["err"] = r["err"]
        if r["aborted"]:
            rec["aborted"] = True
        with self.lock:
            self.recs.append(rec)
        self.write(rec)
        return time.time()

    def progress(self, t_zero):
        now = time.time() - t_zero
        with self.lock:
            last = [r for r in self.recs if r["t"] + r["total"] >= now - self.a.progress]
            n_all, n_err = len(self.recs), sum(1 for r in self.recs if r.get("err"))
        ok = [r for r in last if not r.get("err") and r.get("ttft") is not None]
        tt = sorted(r["ttft"] for r in ok); pp = sorted(r["prompt"] or r["prompt_est"] for r in ok)
        infl = sum(self.router.inflight.values())
        print(f"[{now:6.0f}s] done={n_all} err={n_err} in_flight={infl} open_sessions={sum(self.router.open.values())}"
              f" | last {self.a.progress:.0f}s: req={len(ok)} ttft_p50={pct(tt, 50):.2f}s ttft_max={tt[-1] if tt else 0:.2f}s"
              f" gen={sum(r['gen'] for r in ok)} prompt_p50={pct(pp, 50) / 1000:.0f}k cpt={self.cpt:.2f}", flush=True)


# ----------------------------------------------------------------------------------------------- summary
def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))] if v else 0.0


def overlap(a, b, w0, w1):
    return max(0.0, min(b, w1) - max(a, w0))


def summarize(recs, w0, w1, a=None):
    span = max(1e-9, w1 - w0)
    win = [r for r in recs if w0 <= r["t"] < w1]
    ok = [r for r in win if not r.get("err") and not r.get("aborted") and r.get("ttft") is not None]
    gen_win = busy = 0.0
    for r in recs:
        busy += overlap(r["t"], r["t"] + r["total"], w0, w1)
        if r.get("ttft") is not None and r.get("gen") and r["total"] > r["ttft"]:
            x, y = r["t"] + r["ttft"], r["t"] + r["total"]
            gen_win += r["gen"] * overlap(x, y, w0, w1) / (y - x)
    tt = [r["ttft"] for r in ok]; dec = [r["decode"] for r in ok if r.get("decode")]
    P = [r["prompt"] or r["prompt_est"] for r in ok]; C = [r["cached"] or 0 for r in ok]
    G = [r["gen"] for r in ok]; RS = [(r["reasoning"], r["gen"]) for r in ok if r.get("reasoning") is not None]
    unc = [p - c for p, c in zip(P, C)]
    by = {}
    for r in sorted((r for r in recs if not r.get("err") and r.get("prompt")), key=lambda r: (r["session"], r["turn"])):
        by.setdefault(r["session"], []).append(r)
    growth, toolt = [], []
    for v in by.values():
        for x, y in zip(v, v[1:]):
            if y["turn"] == x["turn"] + 1 and w0 <= y["t"] < w1:
                growth.append(y["prompt"] - x["prompt"]); toolt.append(y["prompt"] - x["prompt"] - x["gen"])
    pre = [r["srv_prefill_tps"] for r in ok if r.get("srv_prefill_tps") and (r.get("srv_prompt_n") or 0) >= 2000]
    acc = [r["draft_acc"] for r in ok if r.get("draft_acc") is not None]
    per_url = {}
    for r in ok:
        per_url[r["url"]] = per_url.get(r["url"], 0) + 1
    think = [r["think"] for r in win if r.get("think") is not None]
    fin = {}
    for r in ok:
        fin[r.get("finish") or "none"] = fin.get(r.get("finish") or "none", 0) + 1
    rnd = lambda x, d=1: round(x, d)  # noqa: E731
    return dict(window_s=[rnd(w0), rnd(w1)], users=getattr(a, "users", None), requests=len(ok),
                errors=sum(1 for r in win if r.get("err")), aborted=sum(1 for r in win if r.get("aborted")),
                first_err=next((r["err"] for r in win if r.get("err")), ""),
                req_per_min=rnd(len(ok) / span * 60, 2), ttft_p50=rnd(pct(tt, 50), 2), ttft_p95=rnd(pct(tt, 95), 2),
                ttft_max=rnd(max(tt or [0]), 2), decode_p50=rnd(pct(dec, 50)), decode_p10=rnd(pct(dec, 10)),
                agg_gen_tps=rnd(gen_win / span), mean_inflight=rnd(busy / span, 2),
                prompt_p50=int(pct(P, 50)), prompt_max=max(P or [0]), prompt_tokens=sum(P), cached_tokens=sum(C),
                cache_hit_pct=rnd(100.0 * sum(C) / sum(P), 1) if sum(P) else 0.0, uncached_p50=int(pct(unc, 50)),
                gen_p50=int(pct(G, 50)), gen_mean=rnd(statistics.mean(G)) if G else 0, gen_max=max(G or [0]),
                reasoning_pct=rnd(100.0 * sum(x for x, _ in RS) / max(1, sum(g for _, g in RS)), 1) if RS else None,
                growth_p50=int(pct(growth, 50)), growth_mean=int(statistics.mean(growth)) if growth else 0,
                tool_tok_p50=int(pct(toolt, 50)), sessions=len({r["session"] for r in ok}),
                cold_starts=sum(1 for r in ok if r["turn"] == 1), think_mean=rnd(statistics.mean(think), 2) if think else 0,
                srv_prefill_tps_p50=rnd(pct(pre, 50)) if pre else None,
                draft_acc_mean=rnd(statistics.mean(acc), 3) if acc else None, finish=fin, per_url=per_url)


def summarize_turns(recs, warmup_turns, a=None):
    """--turns: the same work in every run -- each user's turns after its first `warmup_turns`, whatever the
    server's speed -- and the run's makespan (the last reply's end), the figure a faster server shortens."""
    kept = [r for r in recs if r.get("uturn", 0) > warmup_turns]
    if not kept:
        return summarize(recs, 0.0, 1e-9, a)
    w0 = min(r["t"] for r in kept); w1 = max(r["t"] + r["total"] for r in kept)
    s = summarize(kept, w0, w1 + 1e-6, a)
    s["mode"] = f"turns {getattr(a, 'turns', None)} a user, first {warmup_turns} excluded"
    s["makespan_s"] = round(max(r["t"] + r["total"] for r in recs), 1)
    s["server_s"] = round(sum(r["total"] for r in kept), 1)   # the kept requests' summed latency
    return s


def load_ref(path):
    """(user, user turn) -> (generated tokens, finish) of a reference run's kept requests."""
    out = {}
    if not path:
        sys.exit("--gen-mode reflen needs --ref REF.jsonl")
    for line in open(path):
        if not line.strip():
            continue
        o = json.loads(line)
        if "turn" in o and not o.get("err") and not o.get("aborted") and o.get("gen"):
            out[(o["user"], o.get("uturn", 0))] = (int(o["gen"]), o.get("finish"))
    if not out:
        sys.exit(f"--ref {path}: no requests")
    return out


def reflen_class(r, ignore_eos=False):
    """exact / cut / short / long / noref / err (see --gen-mode reflen). With ignore_eos every reply runs to the
    reference length, so a natural stop is not seen: exact unless the server returned another length."""
    if r.get("err") or r.get("aborted"):
        return "err"
    if r.get("ref_gen") is None:
        return "noref"
    g, rg = r.get("gen") or 0, r["ref_gen"]
    if g < rg:
        return "short"
    if g > rg:
        return "long"
    if not ignore_eos and r.get("finish") == "length" and r.get("ref_finish") not in (None, "length"):
        return "cut"
    return "exact"


def reflen_line(recs, warmup_turns):
    kept = [r for r in recs if r.get("uturn", 0) > (warmup_turns or 0) and "rl" in r]
    n = len(kept)
    if not n:
        return "REFLEN no reflen requests"
    c = {}
    for r in kept:
        c[r["rl"]] = c.get(r["rl"], 0) + 1
    short_tok = sum(r["ref_gen"] - (r.get("gen") or 0) for r in kept if r["rl"] == "short")
    ref_tok = sum(r["ref_gen"] for r in kept if r.get("ref_gen"))
    return ("REFLEN %d kept requests: " % n + ", ".join(f"{k} {v} ({100.0 * v / n:.1f} %)" for k, v in sorted(c.items()))
            + f" | shortfall {short_tok} of {ref_tok} reference tokens ({100.0 * short_tok / ref_tok if ref_tok else 0:.2f} %)")


def pair(path_a, path_b):
    """--pair A B: the same requests of two --turns runs, matched by (user, user turn): per-request ratios B/A of
    TTFT, end-to-end time and per-stream decode rate (geometric mean, median, share where B is faster, a bootstrap
    95 % interval of the geometric mean), after checking that the matched requests carried the same prompt and
    generation lengths."""
    def load(p):
        cfg, out = {}, {}
        for line in open(p):
            if line.strip():
                o = json.loads(line)
                if "config" in o: cfg = o["config"]
                elif "turn" in o and not o.get("err") and not o.get("aborted") and o.get("ttft") is not None:
                    out[(o["user"], o.get("uturn", 0))] = o
        return cfg, out
    ca, A = load(path_a); cb, B = load(path_b)
    keys = sorted(set(A) & set(B))
    wt = max(ca.get("warmup_turns", 0) or 0, cb.get("warmup_turns", 0) or 0)
    keys = [k for k in keys if k[1] > wt]
    same = [k for k in keys if A[k]["gen"] == B[k]["gen"] and abs((A[k]["prompt"] or 0) - (B[k]["prompt"] or 0)) <= 64]
    print(f"PAIR {len(keys)} matched requests after turn {wt}, {len(same)} with the same gen and prompt (+-64 tokens)")
    rng = random.Random(7)
    def stat(name, f):
        v = [f(A[k], B[k]) for k in same]; v = [x for x in v if x and x > 0]
        if not v:
            print(f"PAIR {name}: no data"); return
        lg = [math.log(x) for x in v]; gm = math.exp(sum(lg) / len(lg))
        bs = sorted(math.exp(sum(rng.choice(lg) for _ in lg) / len(lg)) for _ in range(1000))
        print(f"PAIR {name} B/A: geomean {gm:.3f} [{bs[25]:.3f}, {bs[974]:.3f}], median {pct(v, 50):.3f}, "
              f"B faster in {100.0 * sum(1 for x in v if x < 1) / len(v):.0f} % of {len(v)}")
    stat("ttft", lambda x, y: y["ttft"] / x["ttft"] if x["ttft"] else None)
    stat("total", lambda x, y: y["total"] / x["total"] if x["total"] else None)
    stat("decode time", lambda x, y: (x["decode"] / y["decode"]) if x.get("decode") and y.get("decode") else None)
    ma = max(r["t"] + r["total"] for r in A.values()); mb = max(r["t"] + r["total"] for r in B.values())
    print(f"PAIR makespan A {ma:.0f} s, B {mb:.0f} s ({(mb / ma - 1) * 100:+.1f} %)")


def replay_ref_cfg(path):
    for line in open(path):
        if line.strip():
            o = json.loads(line)
            if "config" in o:
                return o["config"]
    sys.exit(f"--replay {path}: no config line")


def pair_replay(path_a, path_b):
    """--pair-replay A B: two --replay runs of the same reference, matched by (user, user turn), kept turns only."""
    def load(p):
        cfg, out = {}, {}
        for line in open(p):
            if line.strip():
                o = json.loads(line)
                if "config" in o: cfg = o["config"]
                elif "turn" in o and not o.get("err") and not o.get("aborted") and o.get("ttft") is not None:
                    out[(o["user"], o.get("uturn", 0))] = o
        return cfg, out
    ca, A = load(path_a); cb, B = load(path_b)
    wt = max(ca.get("warmup_turns", 0) or 0, cb.get("warmup_turns", 0) or 0)
    keys = sorted(k for k in set(A) & set(B) if k[1] > wt)
    T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}
    def tcrit(n):
        for d in sorted(T95, reverse=True):
            if n - 1 >= d: return T95[d] if n - 1 < 40 else 1.96
        return 12.706
    def logci(v):
        lg = [math.log(x) for x in v if x and x > 0]
        if len(lg) < 2: return None
        m = statistics.mean(lg); h = tcrit(len(lg)) * statistics.stdev(lg) / math.sqrt(len(lg))
        return {"n": len(lg), "ratio": round(math.exp(m), 4), "lo": round(math.exp(m - h), 4), "hi": round(math.exp(m + h), 4),
                "median": round(statistics.median([math.exp(x) for x in lg]), 4)}
    def diffci(v):
        if len(v) < 2: return None
        m = statistics.mean(v); h = tcrit(len(v)) * statistics.stdev(v) / math.sqrt(len(v))
        return {"n": len(v), "diff": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4)}
    pm = sum(1 for k in keys if abs((A[k]["prompt"] or 0) - (B[k]["prompt"] or 0)) > 8)
    out = {"a": path_a, "b": path_b, "matched": len(keys), "prompt_mismatch_gt8": pm,
           "ttft": logci([B[k]["ttft"] / A[k]["ttft"] for k in keys if A[k]["ttft"]]),
           "decode_tps": logci([B[k]["decode"] / A[k]["decode"] for k in keys
                                if A[k].get("decode") and B[k].get("decode") and A[k]["gen"] >= 32 and B[k]["gen"] >= 32]),
           "total": logci([B[k]["total"] / A[k]["total"] for k in keys if A[k]["total"]]),
           "draft_acc": diffci([B[k]["draft_acc"] - A[k]["draft_acc"] for k in keys
                                if A[k].get("draft_acc") is not None and B[k].get("draft_acc") is not None]),
           "gen_ratio": round(sum(B[k]["gen"] for k in keys) / max(1, sum(A[k]["gen"] for k in keys)), 4),
           "same_gen": sum(1 for k in keys if A[k]["gen"] == B[k]["gen"])}
    # TTFT by uncached prompt (A's counts)
    by = {}
    for lo, hi, nm in ((0, 2000, "<2k"), (2000, 8000, "2-8k"), (8000, 32000, "8-32k"), (32000, 10 ** 9, ">32k")):
        v = [B[k]["ttft"] / A[k]["ttft"] for k in keys if A[k]["ttft"] and lo <= (A[k]["prompt"] or 0) - (A[k]["cached"] or 0) < hi]
        by[nm] = logci(v)
    out["ttft_by_uncached"] = by
    ma = max(r["t"] + r["total"] for r in A.values()); mb = max(r["t"] + r["total"] for r in B.values())
    out["makespan"] = [round(ma, 1), round(mb, 1)]
    def agg(D):
        g = sum(D[k]["gen"] for k in keys); t = sum(D[k]["total"] - D[k]["ttft"] for k in keys if D[k]["total"] > D[k]["ttft"])
        return g / t if t else None
    ga, gb = agg(A), agg(B)
    out["gen_per_decode_s"] = [round(ga, 2) if ga else None, round(gb, 2) if gb else None]
    # Validity (139): a replay pair compares arms only while both ran REF's load -- no request started before its
    # user's previous reply had ended in that run (an open-loop arm that falls behind overlaps, re-prefills and queues),
    # and both prefilled about the same uncached tokens. VALID needs overlap <= 5 % of the kept turns in each arm and
    # the uncached ratio within 0.9-1.1.
    def overlap(D):
        by_u = {}
        for (u, n), r in D.items():
            by_u.setdefault(u, []).append((n, r))
        o = 0
        for v in by_u.values():
            v.sort(key=lambda x: x[0])
            for (n0, r0), (n1, r1) in zip(v, v[1:]):
                if n1 > wt and r1["t"] < r0["t"] + r0["total"] - 0.05:
                    o += 1
        return o
    unc = lambda D: sum((D[k]["prompt"] or 0) - (D[k]["cached"] or 0) for k in keys)
    oa, ob, ua, ub = overlap(A), overlap(B), unc(A), unc(B)
    ur = ub / ua if ua else None
    ok = oa <= 0.05 * len(keys) and ob <= 0.05 * len(keys) and ur is not None and 0.9 <= ur <= 1.1
    out["validity"] = {"overlap": [oa, ob], "uncached": [ua, ub], "uncached_ratio": round(ur, 4) if ur else None,
                       "closed": [bool(ca.get("replay_closed")), bool(cb.get("replay_closed"))],
                       "verdict": "VALID" if ok else "INVALID (an arm did not run the reference's load: compare closed-loop)"}
    print("PAIR_REPLAY " + json.dumps(out), flush=True)
    return out


def print_summary(s):
    k = lambda x: f"{x / 1000:.1f}k"  # noqa: E731
    print(f"SUMMARY window {s['window_s'][0]:.0f}-{s['window_s'][1]:.0f} s, users={s['users']}\n"
          f"  requests {s['requests']} ok ({s['errors']} err, {s['aborted']} aborted), {s['req_per_min']} req/min, "
          f"mean in-flight {s['mean_inflight']}, sessions {s['sessions']} (cold starts {s['cold_starts']}), "
          f"mean think {s['think_mean']} s\n"
          f"  TTFT p50 {s['ttft_p50']} s, p95 {s['ttft_p95']} s, max {s['ttft_max']} s | decode per stream p50 "
          f"{s['decode_p50']} tok/s (p10 {s['decode_p10']}) | aggregate generated {s['agg_gen_tps']} tok/s\n"
          f"  prompt p50 {k(s['prompt_p50'])} max {k(s['prompt_max'])}, total {s['prompt_tokens'] / 1e6:.2f}M; cached "
          f"{s['cached_tokens'] / 1e6:.2f}M = {s['cache_hit_pct']} %; uncached per request p50 {k(s['uncached_p50'])}; "
          f"server prefill p50 {s['srv_prefill_tps_p50']} tok/s\n"
          f"  gen p50 {s['gen_p50']} mean {s['gen_mean']} max {s['gen_max']}, reasoning {s['reasoning_pct']} %; "
          f"growth/turn p50 {s['growth_p50']} mean {s['growth_mean']} (tool part p50 {s['tool_tok_p50']}); "
          f"draft acceptance {s['draft_acc_mean']}\n  finish {s.get('finish')} | per url {s['per_url']}"
          + (f"\n  first error: {s['first_err']}" if s["first_err"] else ""), flush=True)
    print("SUMMARY_JSON " + json.dumps(s), flush=True)


def resummarize(path, warmup, duration):
    cfg, recs = {}, []
    for line in open(path):
        if not line.strip():
            continue
        o = json.loads(line)
        if "config" in o:
            cfg = o["config"]
        elif "turn" in o:
            recs.append(o)
    w0 = cfg.get("warmup", 0) if warmup is None else warmup
    w1 = cfg.get("duration", max((r["t"] for r in recs), default=0) + 1) if duration is None else duration
    ns = argparse.Namespace(users=cfg.get("users"), turns=cfg.get("turns", 0))
    s = summarize_turns(recs, cfg.get("warmup_turns", 0) or 0, ns) if cfg.get("turns") else summarize(recs, w0, w1, ns)
    print_summary(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", help="URL or comma list; several -> each session pinned to one (see --route)")
    ap.add_argument("--route", choices=["hash", "least-loaded"], default="hash")
    ap.add_argument("--users", type=int, default=1, help="concurrent simulated agents")
    ap.add_argument("--duration", type=float, default=600, help="s; no new requests after this")
    ap.add_argument("--warmup", type=float, default=120, help="s at the start excluded from the summary")
    ap.add_argument("--drain", type=float, default=180, help="s after --duration before running requests are aborted")
    ap.add_argument("--ramp", type=float, default=20, help="user u starts at U(0, ramp) s")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-id", default="", help="first system-prompt line id (default: start time, defeats cross-run cache)")
    ap.add_argument("--ctx-cap", type=int, default=170000, help="session ends when the next prompt would pass this")
    ap.add_argument("--start-min", type=int, default=20000)
    ap.add_argument("--start-max", type=int, default=40000)
    ap.add_argument("--no-preage", dest="preage", action="store_false", help="first sessions start at start-min..max")
    ap.add_argument("--shared-prefix", type=int, default=0, help="tokens of system prompt shared by all sessions of the run")
    ap.add_argument("--tools", type=int, default=109, help="tool schemas per request (10 Claude-Code-like + synthetic MCP)")
    ap.add_argument("--gen-median", type=float, default=350)
    ap.add_argument("--gen-sigma", type=float, default=1.0)
    ap.add_argument("--gen-max", type=int, default=13000)
    ap.add_argument("--gen-mode", choices=["natural", "forced", "reflen"], default="natural",
                    help="natural (default): the model stops at its own tool call, max_tokens=gen-max; forced: "
                         "ignore_eos + max_tokens ~ LN(gen-median, gen-sigma) (exact lengths, but junk after the "
                         "natural stop collapses MTP draft acceptance -> decode too slow)")
    ap.add_argument("--ref", help="--gen-mode reflen: the reference run's jsonl (natural, --turns, --trace-seeded)")
    ap.add_argument("--reflen-eos", choices=["stop", "ignore"], default="stop",
                    help="reflen: stop (default) lets a reply end at its own stop before the reference length; "
                         "ignore sets ignore_eos so every reply is exactly the reference length")
    ap.add_argument("--tool-median", type=float, default=300, help="tool result tokens per turn (lognormal)")
    ap.add_argument("--tool-sigma", type=float, default=1.2)
    ap.add_argument("--tool-min", type=int, default=20)
    ap.add_argument("--tool-max", type=int, default=8000)
    ap.add_argument("--think-median", type=float, default=0.8, help="s, lognormal; 0 = uniform think-min..think-max")
    ap.add_argument("--think-sigma", type=float, default=1.0)
    ap.add_argument("--think-min", type=float, default=0.2)
    ap.add_argument("--think-max", type=float, default=90)
    ap.add_argument("--effort", default="medium", help="medium (default) | low | high | xhigh | off | default; comma list = per session")
    ap.add_argument("--drop-reasoning", action="store_true", help="do not send earlier turns' reasoning back")
    ap.add_argument("--temperature", type=float, default=None, help="default: not sent (server default)")
    ap.add_argument("--cpt", type=float, default=3.5, help="initial chars/token estimate (then calibrated from usage)")
    ap.add_argument("--model", default="")
    ap.add_argument("--key-env", default="", help="env variable holding the bearer key (e.g. RADIANCE_API_KEY)")
    ap.add_argument("--api-key", default="", help="bearer key (prefer --key-env)")
    ap.add_argument("--read-timeout", type=float, default=900, help="s without any byte before a request fails")
    ap.add_argument("--progress", type=float, default=60, help="progress line every s (0 = off)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="", help="JSON lines file (default <script dir>/runs/ccsim-MMDD-HHMM[-tag].jsonl)")
    ap.add_argument("--summarize", default="", help="print the summary of an earlier --out file and exit")
    ap.add_argument("--turns", type=int, default=0,
                    help="each user runs exactly this many turns (across its sessions) and stops: every run does the "
                         "same requests whatever the server's speed; --duration is then only a safety cap (0 = the "
                         "time window, as before)")
    ap.add_argument("--warmup-turns", type=int, default=0, help="--turns: each user's first N turns excluded from the summary")
    ap.add_argument("--cpt-fixed", action="store_true",
                    help="keep --cpt instead of calibrating it from usage, so new sessions' content (and token counts) "
                         "is the same in every run")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="compare two --turns runs request by request and exit")
    ap.add_argument("--trace-seeded", action="store_true",
                    help="each turn's generation length and tool result from a stream of its own (seed, session, turn), so "
                         "replies that differ between runs do not change the later turns' token counts (pair every request)")
    ap.add_argument("--record", action="store_true", help="also write <out>.outputs.jsonl: every reply, for --replay")
    ap.add_argument("--replay", default="", help="REF.jsonl of a --record --turns --trace-seeded --cpt-fixed run: send its "
                                                   "requests open-loop at its start times, prompts built from its replies")
    ap.add_argument("--replay-closed", action="store_true",
                    help="with --replay: closed-loop per user -- each request when this run's previous reply of the same "
                         "user has ended plus the reference's gap after it (prompts unchanged); for arms of unequal capacity")
    ap.add_argument("--pair-replay", nargs=2, metavar=("A", "B"), help="paired per-request rates of two --replay runs, then exit")
    a = ap.parse_args()
    if a.pair_replay:
        pair_replay(*a.pair_replay)
        return
    if a.pair:
        pair(*a.pair)
        return
    if a.summarize:
        resummarize(a.summarize, a.warmup if "--warmup" in sys.argv else None,
                    a.duration if "--duration" in sys.argv else None)
        return
    if not a.urls:
        ap.error("--urls is required")
    key = a.api_key or (os.environ.get(a.key_env, "") if a.key_env else "")
    if a.key_env and not key:
        sys.exit(f"env {a.key_env} is empty")
    urls = [u.rstrip("/") for u in a.urls.split(",") if u]
    model = a.model or model_id(urls[0], key)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                f"ccsim-{time.strftime('%m%d-%H%M')}{'-' + a.tag if a.tag else ''}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    load_corpus()
    tools = build_tools(a.tools) if a.tools > 0 else []
    sim = Sim(a, urls, model, key, tools, out)
    print(f"# cc_sim model={model} urls={urls} route={a.route} users={a.users} duration={a.duration:.0f}s "
          f"warmup={a.warmup:.0f}s cap={a.ctx_cap} start={a.start_min}-{a.start_max} preage={a.preage} "
          f"tools={len(tools)} (~{sim.est(sim.tools_chars)} tok) gen={a.gen_mode}"
          f"{'~LN(%g,%g)' % (a.gen_median, a.gen_sigma) if a.gen_mode == 'forced' else ''}<={a.gen_max} "
          f"tool~LN({a.tool_median:.0f},{a.tool_sigma}) think~{'LN(%g,%g)' % (a.think_median, a.think_sigma) if a.think_median > 0 else 'U'}"
          f"[{a.think_min},{a.think_max}]s effort={a.effort} auth={'yes' if key else 'no'} run={sim.run_id} "
          f"{time.strftime('%F %T')}", flush=True)
    sim.run()


if __name__ == "__main__":
    main()
