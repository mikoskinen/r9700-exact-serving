#!/usr/bin/env python3
"""mockrad.py (12.11.2026, queue-ar/tools/mockrad, ds-main-port mockrad builder) -- a NO-GPU mock of the Radiance server for the
DeepSeek-V4.1 queue: standard library only, CPU only, nothing of the real server, no card, no docker. It exists so a new item's
block / summary / parser scripts run END TO END before the item is queued (mockrun.sh drives a body against it; a DRYRUN cannot
see a TypeError three hours into a body, a wrong anchor or a guard that VIRHEs every block).

  mockrad.py [--port 18090] [--state DIR] [--templates templates/ds41] [--serve-args "..."] [--env-json ENV.json]
             [--decode-scale 0.5] [--prefill-scale 0.1] [--tokenizer regex|chars] [--cpt 3.9] [--replay R.jsonl]
             [--start-extra LINES.txt] [--startup-s 1] [--fault NAME[=V] ...] [--seed 1]

Serves (schemas = the real int8 image's, from real snapshots in --templates; every counter a body / parser reads ADVANCES with the
simulated engine and the switch state):
  GET  /health /v1/models /props
  GET  /stats            the real /stats (psk, dspark_pfctx, engdd, experts, link, steps*, step_ns*, draft_positions, ...)
  GET  /admin/fn4        switches / pending / epoch / applied_step / draft / accept / sahb / lend / sample / noa / ctxskip
  POST /admin/fn4        range-checked like the image (unknown switch or out of range -> 400, nothing queued); applied at the
                         next engine step (a nudge request makes one), epoch + 1 with epoch_bump, applied_step = that step
  GET  /metrics          the real Prometheus text, the counters mb.py / mbt.py / fastcheck / stepprobe / the bodies read live
  POST /v1/chat/completions, /v1/completions (SSE stream or not), /completion (llama.cpp shape, the bodies' nudge)
  POST /reset_prefix_cache  {"entries_dropped": n}      POST /tokenize {"content": s} -> {"tokens": [...]}
  GET  /mock/state       the mock's own state;  POST /mock/fault {"set": {...}, "clear": [...]} changes the faults at run time
Text: deterministic from a hash of the rendered prompt (+ seed when sampled), so identity compares work (same prompt -> same
answer, alone or under load, any arm) unless a fault makes an arm differ. A copy prompt (<<<BEGIN>>> .. <<<END>>>) and an
edit prompt (editprobe's) are echoed. --replay serves recorded answers for exact prompts (the warm-up / greedy-set references).
Tokens: --tokenizer regex (default) = a GPT-style pre-tokenizer count x 0.9365, calibrated on the real server's numbers (deepq
16000 -> 15777 tokens, a 48k chain turn -> 45766, mb's ~1k prompts 860..1520 vs the real 919..1386); chars = len / --cpt.
Prefix cache: block hashes of the rendered prompt; a hit is floored to the checkpoint grid (--checkpoint-interval of the serve
line, default 2048), like the real hits (43008 / 47104).
Engine: one thread plans steps like the real scheduler (decode rows 1 + drafted, prefill chunks up to --max-num-batched-tokens,
MIXED chunks capped at sched_mixed_tok when decode rows ride along, at most --max-num-seqs admitted); RADIANCE_LOG_STEPS=1 in the
container env writes "step K prefill|decode|MIXED n_seq= n_tok=  [i] slot= q= ctx=" lines with docker-logs -t timestamps into
STATE/engine.log (the docker shim serves it), after the real start log (templates/start.log + --start-extra lines);
RADIANCE_PROFILE_EVERY=N (or --profile-ops in the serve line) dumps the real per-op timing table every N steps.
Faults (--fault, repeatable; also POST /mock/fault), so the guards can be seen firing:
  dead=LEVER      the lever's counters never move (psk, pskp, l20, pfctx, engdd, noa, sa, hb, cr, tail, draft, steplog)
  leak=LEVER      the lever's counters move with its switch 0
  noapply=SW      a POST of SW is accepted but stays pending (read-back fails)
  wrong=SW:V      SW reads back as V after it is applied
  refuse=SW       a POST naming SW gets 400
  cached=N        every request reports N cached tokens (a newcomer-freshness guard must fire)
  nocache         the prefix cache never hits
  differ=SW[:V]   answers diverge after 16 tokens while SW is V (any nonzero without :V) -- identity must say DIFFERENT
  err500=P        a fraction P of completion requests get HTTP 500
  stall=S         every completion waits S s before its first token
  mismatch        psk plan_mismatch grows with every planned prefill
  drop=KEY        /stats leaves out KEY (e.g. psk, engdd, experts)
  health=CODE     /health answers CODE
  slow=X          every step takes X times longer
"""
import argparse, collections, copy, hashlib, http.server, json, math, os, random, re, socketserver, sys, threading, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
TOKRE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d{1,3}| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+""")
KTOK = 0.9365
WORDS = ("the of and to in a is that for it as with was on be by this are from or at an which have not but code function "
         "module value returns data file each when list we can will should more one all also new used use any into if "
         "then so these other than its time two may first way each line call class method object string type error test "
         "case python state step cache prefix token layer expert rank card memory read write path name key result loop "
         "thread lock queue server request client handler buffer index offset size count total mean change review").split()
LEVERS = ("psk", "pskp", "l20", "pfctx", "engdd", "noa", "sa", "hb", "cr", "tail", "draft", "steplog")
# the image's range messages ("draft_occ takes 0 (off) .. 64", "stage_short takes 0 (off) or 1 .. 65536", ...)
RANGES = {"ds41_psk": (0, 4), "draft_occ": (0, 64), "draft_occ_k": (1, 4096), "sched_noa": (0, 2), "stage_short": (0, 65536),
          "stage_pick": (0, 64), "place_hb": (0, 4096), "ds41_psk_k": (1, 4096), "draft_sample": (0, 200), "ds41_bi": (0, 4),
          "sched_bi_chunk": (0, 8), "ds41_psk_l20": (0, 1), "dspark_pfctx": (0, 1), "ds41_psk_cr": (0, 1), "ds41_engdd": (0, 1),
          "gg_gemv": (0, 2), "aql_acq": (0, 3), "draft_probe": (0, 4096), "sched_mixed_tok": (1, 65536), "sched_interleave": (0, 64)}
ENVSW = {"RADIANCE_AQL_ACQ": ("aql_acq", {"none": 0, "rule": 1, "rule-w": 2}), "RADIANCE_DS41_ARGATHER_MAX": ("ds41_argather", None)}


def ts(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(t))) + ".%09dZ" % int((t - int(t)) * 1e9)


class Tok:
    def __init__(self, kind, cpt):
        self.kind, self.cpt = kind, cpt

    def count(self, s):
        if not s: return 0
        if self.kind == "chars": return max(1, int(round(len(s) / self.cpt)))
        n = 0
        for m in TOKRE.finditer(s):
            L = len(m.group(0).strip())
            n += (L + 5) // 6 if L > 6 else 1
        return max(1, int(round(n * KTOK)))

    @staticmethod
    def pieces(s):
        return [m.group(0) for m in TOKRE.finditer(s)]


class Req:
    __slots__ = ("id", "kind", "prompt_tok", "cached", "pf_done", "gen", "gen_i", "reason_n", "max_tokens", "finish", "state", "slot",
                 "q", "t_arr", "t_first", "t_end", "cancel", "blocks", "rid", "temp", "rng", "first_chunk", "hit", "tool")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))


class Mock:
    def __init__(self, a):
        self.a = a
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.t_start = time.time()
        td = a.templates if os.path.isabs(a.templates) else os.path.join(HERE, a.templates)
        self.st_tpl = json.load(open(os.path.join(td, "stats.json")))
        self.fn_tpl = json.load(open(os.path.join(td, "fn4.json")))
        self.m_tpl = open(os.path.join(td, "metrics.txt")).read().splitlines()
        self.start_lines = open(os.path.join(td, "start.log"), errors="replace").read().splitlines()
        po = os.path.join(td, "perop.txt")
        self.perop = open(po, errors="replace").read().splitlines() if os.path.exists(po) else []
        rt = os.path.join(td, "runtime.log")
        self.runtime = open(rt, errors="replace").read().splitlines() if os.path.exists(rt) else []
        self.runtime_done = False
        self.env = {}
        if a.env_json and os.path.exists(a.env_json):
            self.env = json.load(open(a.env_json))
        self.model = self.st_tpl.get("model") or "deepseek-v4.1-flash"
        self.start_src = os.path.join(td, "start.log")
        if a.start_lib and os.path.exists(a.start_lib):
            # the real start log whose start-time env (the engine's own name dump, "I   RADIANCE_..." lines) best matches this
            # container's env: names weighted by rarity across the library (an nvfp4 start gets an nvfp4 log, a KVR start a KVR log)
            NRE = re.compile(r"^I   ((?:RADIANCE_|R4D_)[A-Z0-9_]+)$")
            logs = []
            for e in json.load(open(a.start_lib)):
                try:
                    with open(e["log"], errors="replace") as f:
                        logs.append((e["log"], {m.group(1) for m in map(NRE.match, (l.rstrip("\n") for l in f)) if m}))
                except OSError:
                    continue
            logs = [x for x in logs if x[1]]
            if logs:
                df = collections.Counter(n for _, ns in logs for n in ns)
                mine = {k for k, v in self.env.items() if k.startswith(("RADIANCE_", "R4D_")) and str(v) != ""}
                w = lambda n: math.log(1 + len(logs) / (1 + df.get(n, 0)))  # noqa: E731
                sc = [(sum(w(n) if (n in mine) == (n in ns) else -w(n) for n in (mine | ns)), -k, path) for k, (path, ns) in enumerate(logs)]
                sc.sort(reverse=True)
                best = sc[0]
                rdl = lambda p: [l.rstrip("\n") for l in open(p, errors="replace")  # noqa: E731
                                 if not l.startswith("radiance: dbg") and not re.search(r" step \d+ (prefill|decode|MIXED) n_seq=", l)]
                mask = lambda l: re.sub(r"\d+(\.\d+)?", "#", l)  # noqa: E731
                try:
                    L = rdl(best[2])
                    names = dict(logs)
                    base = names[best[2]]
                    # a start-word combination no real start had (e.g. nvfp4 + KVR): for an env name the base lacks (or has and the
                    # container lacks), the base's lines naming it (its default / set state) are REPLACED by the lines naming it in
                    # the best log that matches the container on that name; the donor's other lines the base lacks are appended,
                    # except those naming an env the base declares
                    seen, add, donors = {mask(l) for l in L}, [], []
                    for n in sorted(((mine ^ base) & set(df)), key=lambda n: df[n]):
                        want = n in mine
                        d = next((p for _, _, p in sc if (n in names[p]) == want), None)
                        if not d: continue
                        donors.append("%s%s <- %s" % ("" if want else "no ", n, d.rsplit("/", 2)[-2] + "/" + os.path.basename(d)))
                        DL = rdl(d)
                        mine_l = [l for l in DL if n in l]
                        pos = next((k for k, l in enumerate(L) if n in l), None)
                        L = [l for l in L if n not in l]
                        if pos is None: pos = len(L)
                        L[pos:pos] = mine_l
                        seen.update(mask(l) for l in mine_l)
                        if want:
                            for l in DL:
                                m = mask(l)
                                if m in seen or any(b_ in l for b_ in base): continue
                                seen.add(m); add.append(l)
                    if L:
                        self.start_lines = L + add
                        self.start_src = "%s (env-name score %.1f of %d logs%s)" % (best[2], best[0], len(logs),
                                                                                  "; + %d line(s) for %s" % (len(add), ", ".join(donors)) if donors else "")
                except OSError:
                    pass
        sa = a.serve_args or ""
        def opt(name, d):
            m = re.search(r"--%s[ =](\S+)" % re.escape(name), sa)
            return int(m.group(1)) if m and m.group(1).lstrip("-").isdigit() else d
        self.mnbt, self.maxseqs = opt("max-num-batched-tokens", 8192), opt("max-num-seqs", 8)
        self.nst, self.ckint = opt("num-speculative-tokens", 5), opt("checkpoint-interval", 2048)
        self.mml = opt("max-model-len", 524288)
        pe = self.env.get("RADIANCE_PROFILE_EVERY") or ""
        self.profile_every = int(pe) if str(pe).isdigit() and int(pe) > 0 else (8 if "--profile-ops" in sa else 0)
        self.logsteps = str(self.env.get("RADIANCE_LOG_STEPS", "")) == "1" or a.log_steps
        self.hb_ranks = tuple(int(x) for x in str(self.env.get("RADIANCE_PLACE_HB_RANKS", "1,2")).split(",") if x.strip().isdigit())
        self.tok = Tok(a.tokenizer, a.cpt)
        acq = str(self.env.get("RADIANCE_AQL_ACQ", "rule-w"))
        self.runtime = [l.replace("RADIANCE_AQL_ACQ=rule-w)", "RADIANCE_AQL_ACQ=%s)" % acq) for l in self.runtime]
        if self.env.get("RADIANCE_DS41_BITRACE"):   # under lever BI's tracer the real engine records no pass (eager issue)
            self.runtime = [l for l in self.runtime if not l.startswith("I recorded pass")]
        self.faults = {}
        for f in a.fault or []: self.set_fault(f)
        self.replay = {}
        if a.replay and os.path.exists(a.replay):
            for l in open(a.replay):
                try: d = json.loads(l)
                except ValueError: continue
                k = ("p", d["prompt"]) if "prompt" in d else ("c", d.get("content"))
                self.replay[k] = d
        # switches: the template's start state, then the container env's start values
        self.sw = dict(self.fn_tpl.get("switches") or {})
        for k, v in self.env.items():
            nm, conv = None, None
            if k in ENVSW: nm, conv = ENVSW[k]
            elif k.startswith("RADIANCE_"):
                r = k[len("RADIANCE_"):].lower()
                for c in (r, r[5:] if r.startswith("ds41_") else None, "ds41_" + r):
                    if c and c in self.sw: nm = c; break
            if not nm or nm not in self.sw: continue
            sv = str(v).strip()
            if conv and sv in conv: self.sw[nm] = conv[sv]
            elif re.fullmatch(r"-?\d+", sv) and nm not in ("ds41_argather",): self.sw[nm] = int(sv)
        self.pending, self.bump, self.epoch, self.applied_step = {}, False, 0, -1
        self.K = 0
        self.c = collections.defaultdict(float)
        self.psk = {k: 0 for k in (self.st_tpl.get("psk") or {})}
        self.pfctx = {k: 0 for k in (self.st_tpl.get("dspark_pfctx") or {"mode": 0, "passes": 0, "rows_skipped": 0, "rows_run": 0})}
        self.engdd = {k: 0 for k in (self.st_tpl.get("engdd") or {"passes": 0, "rows_deduped": 0, "rows_pooled": 0})}
        ft = self.fn_tpl
        self.draft = {k: 0 for k in (ft.get("draft") or {})}
        self.accept = {k: 0 for k in (ft.get("accept") or {})}
        self.sample = {k: 0 for k in (ft.get("sample") or {})}; self.sample["capable"] = 1
        self.lend = {k: (v if k in ("capable", "probe_interval", "draft_resident", "draft_units", "slab_mib", "w_mib") else 0) for k, v in (ft.get("lend") or {}).items()}
        self.noa = {k: 0 for k in (ft.get("noa") or {})}
        self.ctxskip = {k: (v if k in ("capable", "ring_slots", "ring_mib") else 0) for k, v in (ft.get("ctxskip") or {}).items()}
        self.dst = {k: 0 for k in (ft.get("dst") or {})}
        rk = ((ft.get("sahb") or {}).get("ranks") or [{"rank": r} for r in range(4)])
        self.sahb = [{k: (v if k == "rank" else 0) for k, v in r.items()} for r in rk]
        self.dpos = [[0, 0] for _ in range(max(1, len(self.st_tpl.get("draft_positions") or [0] * 5)))]
        nr = len((self.st_tpl.get("experts") or {}).get("ranks") or [0] * 4)
        self.rk = [collections.defaultdict(float) for _ in range(nr)]
        self.reqs, self.slots, self.nreq = [], {}, 0
        self.cache, self.cache_n = set(), 0
        self.unknown = collections.Counter()
        self.stop = False
        self.rng = random.Random(a.seed)
        os.makedirs(a.state, exist_ok=True)
        self.logf = open(os.path.join(a.state, "engine.log"), "a", buffering=1)
        self.alog = open(os.path.join(a.state, "mockrad-events.log"), "a", buffering=1)
        self.write_start()
        self.ready_at = time.time() + a.startup_s
        threading.Thread(target=self.engine, daemon=True).start()

    # ------------------------------------------------------------------ faults / logs
    def set_fault(self, f):
        k, _, v = f.partition("=")
        if k not in ("dead", "leak", "noapply", "wrong", "refuse", "cached", "nocache", "differ", "err500", "stall", "mismatch", "drop", "health", "slow"):
            raise SystemExit("unknown fault %r" % f)
        if k in ("dead", "leak", "noapply", "refuse", "drop"):
            self.faults.setdefault(k, set()).update(x for x in v.split(",") if x)
        elif k == "wrong":
            sw, _, val = v.partition(":"); self.faults.setdefault("wrong", {})[sw] = int(val)
        elif k == "differ":
            sw, _, val = v.partition(":"); self.faults.setdefault("differ", {})[sw] = int(val) if val else None
        else:
            self.faults[k] = v if v else True

    def dead(self, lv):
        return lv in self.faults.get("dead", ())

    def leak(self, lv):
        return lv in self.faults.get("leak", ())

    def event(self, s):
        self.alog.write("%s %s\n" % (time.strftime("%F %T"), s))

    def logline(self, s, t=None):
        self.logf.write("%s %s\n" % (ts(t or time.time()), s))

    def write_start(self):
        t = time.time() - 0.5
        extra = []
        if self.a.start_extra and os.path.exists(self.a.start_extra):
            extra = [l.rstrip("\n") for l in open(self.a.start_extra, errors="replace") if l.strip()]
        lines = self.start_lines + ["I mock: %s" % l for l in extra]
        lines.append("I mock: mockrad.py (no GPU) serving the real start log %s; serve line: %s" % (self.start_src, self.a.serve_args or "-"))
        dt = 0.4 / max(1, len(lines))
        for i, l in enumerate(lines):
            self.logline(l, t + i * dt)

    # ------------------------------------------------------------------ prompt handling
    def render(self, p, chat):
        if not chat:
            return p.get("prompt") if isinstance(p.get("prompt"), str) else json.dumps(p.get("prompt"))
        out = []
        for m in p.get("messages") or []:
            c = m.get("content")
            if isinstance(c, list): c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            out.append("<|%s|>%s" % (m.get("role"), c or ""))
            if m.get("tool_calls"): out.append(json.dumps(m["tool_calls"]))
        if p.get("tools"): out.insert(0, "<|tools|>" + json.dumps(p["tools"]))
        return "".join(out) + "<|Assistant|>"

    def blocks(self, s, B=512):
        h, out = b"", []
        for i in range(0, len(s) - len(s) % B, B):
            h = hashlib.blake2b(h + s[i:i + B].encode("utf-8", "surrogatepass"), digest_size=12).digest()
            out.append(h)
        return out

    def lookup(self, s, blocks, ptok):
        if "nocache" in self.faults: return 0
        n = 0
        for h in blocks:
            if h in self.cache: n += 1
            else: break
        if not n: return 0
        mt = self.tok.count(s[:n * 512])
        c = (mt // self.ckint) * self.ckint
        return max(0, min(c, ptok - 1))

    def gentext(self, rendered, p, chat, max_tokens):
        """-> (reasoning pieces, content pieces, finish, tool)"""
        last = ""
        if chat:
            ms = p.get("messages") or []
            us = [m for m in ms if m.get("role") == "user"]
            last = (us[-1].get("content") if us else "") or ""
            if isinstance(last, list): last = "".join(x.get("text", "") for x in last if isinstance(x, dict))
        else:
            last = rendered
        temp = p.get("temperature", 1.0)
        seed = p.get("seed")
        greedy = temp is not None and float(temp) == 0.0
        salt = "" if greedy else ("seed%s" % seed if seed is not None else "rand%d" % self.rng.randrange(1 << 30))
        eff = p.get("reasoning_effort")
        think = chat and ((eff not in (None, "none")) or bool((p.get("chat_template_kwargs") or {}).get("enable_thinking")))
        ignore_eos = bool(p.get("ignore_eos"))
        h = int(hashlib.sha256((rendered + "|" + salt).encode("utf-8", "surrogatepass")).hexdigest()[:16], 16)
        rg = random.Random(h)
        tool = None
        if chat and p.get("tools") and (p.get("messages") or [{}])[-1].get("role") == "user":
            fn = (p["tools"][0].get("function") or {})
            props = ((fn.get("parameters") or {}).get("properties") or {})
            args = {}
            for k in ((fn.get("parameters") or {}).get("required") or list(props)[:1]):
                caps = re.findall(r"\b[A-Z][a-z]{2,}\b", last)
                args[k] = caps[-1] if caps else "x"
            if "unit" in props and "celsius" in last.lower(): args["unit"] = "celsius"
            tool = {"name": fn.get("name") or "tool", "arguments": json.dumps(args)}
        rep = None
        if greedy:
            rep = self.replay.get(("p", rendered)) if not chat else (self.replay.get(("c", last)) if len(p.get("messages") or []) == 1 else None)
        if rep is not None:
            pcs = self.tok.pieces(rep.get("text") or "")
            n = int(rep.get("tokens") or len(pcs))
            # spread the recorded text over its recorded token count
            if pcs and n > 0:
                per = max(1, int(math.ceil(len(pcs) / n)))
                pcs = ["".join(pcs[i:i + per]) for i in range(0, len(pcs), per)]
            pcs = pcs[:max_tokens]
            fin = rep.get("finish") or ("length" if len(pcs) >= max_tokens else "stop")
            return [], pcs, fin, None
        m = re.search(r"<<<BEGIN>>>\n(.*)\n<<<END>>>", last, re.S)
        if m:
            src = m.group(1).strip("\n")   # the model copies the text, not the newline after the marker
            pcs = self.tok.pieces(src)
        elif "Output the whole file" in last or "Output only the file" in last:
            src = last.split("\n\n", 1)[1] if "\n\n" in last else last
            pcs = self.tok.pieces(src)
        else:
            pcs = []
            nat = 16 + h % 600
            want = max_tokens if ignore_eos else min(max_tokens, nat)
            cap = True
            for i in range(want + 1):
                w = rg.choice(WORDS)
                if cap: w = w.capitalize(); cap = False
                tail = ""
                r = rg.random()
                if r < 0.07: tail, cap = ".", True
                elif r < 0.10: tail = ","
                elif r < 0.115: tail, cap = ".\n\n", True
                pcs.append((" " if i and not pcs[-1].endswith("\n") else "") + w + tail)
        df = self.faults.get("differ") or {}
        for sw, val in df.items():
            cur = self.sw.get(sw)
            if (val is None and cur) or (val is not None and cur == val):
                rg2 = random.Random(h ^ 0x5A5A5A)
                pcs = pcs[:16] + [" " + rg2.choice(WORDS) for _ in pcs[16:]]
        n_nat = len(pcs)
        if ignore_eos and len(pcs) < max_tokens:
            pcs = pcs + [" " + rg.choice(WORDS) for _ in range(max_tokens - len(pcs))]
        rpcs = []
        if think and not tool:
            nr = max(4, min(len(pcs) // 3, 160))
            rr = random.Random(h ^ 0xABCDEF)
            rpcs = ["Let"] + [" " + rr.choice(WORDS) for _ in range(nr - 1)]
        total = rpcs + pcs
        if len(total) > max_tokens:
            k = max_tokens
            rpcs, pcs = total[:min(len(rpcs), k)], total[len(rpcs):k] if k > len(rpcs) else []
            fin = "length"
        else:
            fin = "stop" if not ignore_eos or n_nat >= max_tokens else "length"
            if ignore_eos: fin = "length"
        if tool: return [], [], "tool_calls", tool
        return rpcs, pcs, fin, None

    def new_request(self, p, chat, max_tokens):
        rendered = self.render(p, chat)
        ptok = self.tok.count(rendered) + (6 if chat else 1) + 4 * (len(p.get("messages") or []) if chat else 0)
        blocks = self.blocks(rendered)
        with self.lock:
            cached = self.lookup(rendered, blocks, ptok)
            if "cached" in self.faults: cached = min(ptok - 1, int(self.faults["cached"]))
            rp, cp, fin, tool = self.gentext(rendered, p, chat, max_tokens)
            self.nreq += 1
            gen = [("r", x) for x in rp] + [("c", x) for x in cp]
            if tool: gen = [("t", tool)]
            r = Req(id=self.nreq, kind="chat" if chat else "text", prompt_tok=ptok, cached=cached, pf_done=0, gen=gen, gen_i=0,
                    reason_n=len(rp), max_tokens=max_tokens, finish=fin, state="prefill", slot=None, q=None, t_arr=time.time(),
                    t_first=None, t_end=None, cancel=False, blocks=blocks, temp=p.get("temperature"),
                    rng=random.Random(self.nreq * 7919 + 13), first_chunk=True, hit=cached > 0, tool=tool)
            import queue as _q
            r.q = _q.Queue()
            self.reqs.append(r)
            self.c["total_requests"] += 1
            if cached:
                self.c["attn_cached_tokens"] += cached
            self.cond.notify_all()
        self.event("req %d %s prompt %d cached %d gen %d max %d finish %s" % (r.id, r.kind, ptok, cached, len(gen), max_tokens, fin))
        r.rid = ("chatcmpl-" if chat else "cmpl-") + hashlib.md5(b"%d" % r.id).hexdigest()[:24]
        if chat:
            self.logline("I req %s chat %s from 127.0.0.1:%d model=%s msgs=%d tools=%d prompt=%d tok n=1 max_tokens=%d" % (
                r.rid, "stream" if p.get("stream") else "once", 40000 + r.id % 20000, self.model, len(p.get("messages") or []),
                len(p.get("tools") or []), ptok, max_tokens))
        else:
            self.logline("I req %s completion from 127.0.0.1:%d model=%s prompts=1 prompt=%d tok n=1 max_tokens=%d" % (
                r.rid, 40000 + r.id % 20000, self.model, ptok, max_tokens))
        return r

    # ------------------------------------------------------------------ engine
    def engine(self):
        while True:
            with self.cond:
                while not self.stop and not any(not r.cancel and r.state != "done" for r in self.reqs):
                    self.cond.wait(0.5)
                if self.stop: return
                plan = self.plan()
            if plan is None:
                time.sleep(0.005); continue
            wall = self.step_time(plan)
            time.sleep(wall)
            with self.cond:
                self.commit(plan, wall)
                self.cond.notify_all()

    def plan(self):
        K = self.K
        if self.pending:
            self.apply_pending(K)
        live = [r for r in self.reqs if not r.cancel and r.state != "done"]
        for r in [r for r in self.reqs if r.cancel and r.state != "done"]:
            self.finish(r, cancelled=True)
        adm = [r for r in live if r.slot is not None]
        for r in live:
            if r.slot is None and len(adm) < self.maxseqs:
                free = next(s for s in range(64) if s not in self.slots)
                r.slot = free; self.slots[free] = r.id; adm.append(r)
        sw = self.sw
        ents, dec, pfr = [], [], []
        for r in adm:
            if r.state == "decode":
                d = 0
                if self.nst > 0 and not self.dead("draft"):
                    d = self.nst
                    if sw.get("draft_depth0") and r.rng.random() < 0.18: d = 0
                a = 0
                while a < d and r.rng.random() < 0.74: a += 1
                n = min(1 + a, len(r.gen) - r.gen_i)
                ctx = r.prompt_tok + r.gen_i
                dec.append((r, d, a, n)); ents.append((r.slot, 1 + d, ctx))
        budget = self.mnbt - sum(1 + d for _, d, _, _ in dec)
        if dec: budget = min(budget, int(sw.get("sched_mixed_tok") or 2112) if sw.get("sched_bi_chunk") else budget)
        for r in adm:
            if r.state == "prefill" and budget > 0:
                rem = r.prompt_tok - r.cached - r.pf_done
                ch = min(rem, budget)
                if ch <= 0: continue
                pfr.append((r, ch, r.cached + r.pf_done)); ents.append((r.slot, ch, r.cached + r.pf_done))
                budget -= ch
        if not ents: return None
        ph = "MIXED" if dec and pfr else ("decode" if dec else "prefill")
        if dec and not self.runtime_done:
            self.runtime_done = True
            for l in self.runtime: self.logline(l)
        self.K += 1
        if self.logsteps and not self.dead("steplog"):
            self.logline("I step %d %s n_seq=%d n_tok=%d" % (K, ph, len(ents), sum(q for _, q, _ in ents)) +
                         "".join("  [%d] slot=%d q=%d ctx=%d" % (i, s, q, c) for i, (s, q, c) in enumerate(ents)))
        return {"K": K, "ph": ph, "dec": dec, "pf": pfr, "n_tok": sum(q for _, q, _ in ents), "n_seq": len(ents)}

    def step_time(self, p):
        nd = len(p["dec"]); dr = sum(d for _, d, _, _ in p["dec"]); pt = sum(ch for _, ch, _ in p["pf"])
        dms = (20.0 + 1.9 * nd + 0.9 * dr) if nd else 0.0
        pms = (0.27 * pt + 25.0) if pt else 0.0
        ms = dms * self.a.decode_scale + pms * self.a.prefill_scale
        if "slow" in self.faults:
            try: ms *= float(self.faults["slow"])
            except (TypeError, ValueError): ms *= 3
        return max(0.002, ms / 1000.0)

    def apply_pending(self, K):
        keep = {}
        for k, v in self.pending.items():
            if k in self.faults.get("noapply", ()): keep[k] = v; continue
            self.sw[k] = self.faults.get("wrong", {}).get(k, v)
        ch = ", ".join("%s=%s" % (k, v) for k, v in self.pending.items() if k not in keep)
        self.pending = keep
        bump = self.bump
        if self.bump: self.epoch += 1; self.bump = False
        self.applied_step = K
        self.logline("I fn4: switches %s applied at step %d (epoch %d)" % (", ".join(x for x in (ch, "epoch_bump" if bump else "") if x), K, self.epoch))
        self.event("applied at step %d: epoch %d, still pending %s" % (K, self.epoch, keep))

    def on(self, lv, sw_on):
        if self.dead(lv): return False
        return bool(sw_on) or self.leak(lv)

    def commit(self, p, wall):
        c, sw, ps = self.c, self.sw, self.psk
        ns = int(wall * 1e9)
        c["steps"] += 1; c["steps_" + p["ph"].lower()] += 1; c["step_ns"] += ns
        if p["ph"] == "decode": c["step_ns_decode"] += ns
        if p["ph"] != "decode": c["steps_ahead"] += 0
        else: c["steps_ahead"] += 1
        nd = len(p["dec"])
        pt = sum(ch for _, ch, _ in p["pf"])
        c["prefill_tokens"] += pt
        # experts / link: routed rows, a hit share, streamed bytes
        rows = p["n_tok"]
        c["routed"] += rows * 6; c["routed_res"] += int(rows * 6 * 0.94); c["dispatches"] += 40
        c["ar_calls"] += 86; c["ar_bytes"] += rows * 5120 * 2 * 86
        c["stream_bytes"] += 2.8e8 + rows * 1.2e6; c["h2d_bytes"] += 1e6 * rows
        for i, rk in enumerate(self.rk):
            rk["stream_bytes"] += 7e7 + rows * 3e5; rk["routed"] += rows * 6; rk["routed_res"] += int(rows * 6 * 0.94)
            rk["miss_cold"] += rows // 50; rk["hit_trunk"] += rows * 4
        if pt > 1024:
            c["stage_layers"] += 40; c["staged_units"] += 900; c["staged_bytes"] += 8.5e9; c["stage_copies"] += 40
            c["link_staged_bytes"] += 3.4e10
        # drafting
        for r, d, a, n in p["dec"]:
            c["draft_tokens"] += d; c["draft_accepted"] += a
            if d == 0:
                self.draft["zero_cycles"] = self.draft.get("zero_cycles", 0) + (0 if self.dead("draft") else 1)
                c["rows_unspec"] += 1
            for k in range(min(d, len(self.dpos))):
                self.dpos[k][0] += 1
                if k < a: self.dpos[k][1] += 1
            if d:
                self.accept["head_drafted"] = self.accept.get("head_drafted", 0) + d
                self.accept["head_accepted"] = self.accept.get("head_accepted", 0) + a
            if sw.get("draft_occ") and "occ_cycles" in self.draft and not self.dead("draft"):
                self.draft["occ_cycles"] += 1
                if r.rng.random() < 0.05: self.draft["occ_moves"] += 1
            if sw.get("draft_sample") and r.temp not in (0, 0.0) and "rows_sampled" in self.sample: self.sample["rows_sampled"] += 1
            elif "rows_greedy" in self.sample: self.sample["rows_greedy"] += 1
            if "rounds_bound" in self.sample: self.sample["rounds_bound"] += 1
            if sw.get("draft_ctxskip") and d == 0 and "skipped_steps" in self.ctxskip: self.ctxskip["skipped_steps"] += 1
            if sw.get("draft_lend") and "lent_cycles" in self.lend and d == 0: self.lend["lent_cycles"] += 1; self.lend["lent_plain_steps"] += 1
            self.emit(r, n)
        if nd and "probe_cycles" in self.draft and self.K % 16 == 0: self.draft["probe_cycles"] += 1
        # prefill chunks: the levers
        npf = len(p["pf"])
        if npf:
            k = "pf_steps_%s" % (npf if npf < 4 else "4plus")
            if k in self.noa: self.noa[k] += 1
        psk_on = self.on("psk", sw.get("ds41_psk"))
        for r, ch, ctx in p["pf"]:
            first = r.first_chunk; r.first_chunk = False
            # checkpoints on the grid
            c["ckpts"] += (ctx + ch) // self.ckint - ctx // self.ckint
            if psk_on or self.leak("pskp"):
                if first:
                    ps["latched"] = ps.get("latched", 0) + 1
                    if sw.get("ds41_psk") == 4 and self.on("tail", True): ps["tail_only_latched"] = ps.get("tail_only_latched", 0) + 1
                    if r.hit and "fallback_resumes" in ps: ps["fallback_resumes"] += 1
                for kk in ("planned_steps", "engine_planned_passes", "plugin_planned_passes"):
                    if kk in ps: ps[kk] += 1
                if "mismatch" in self.faults: ps["plan_mismatch"] = ps.get("plan_mismatch", 0) + 1
                if ch >= 128:
                    ps["cut_passes"] = ps.get("cut_passes", 0) + 1
                    ps["rows_skipped_l21_29"] = ps.get("rows_skipped_l21_29", 0) + ch * 7
                    ps["rows_skipped_l30_39"] = ps.get("rows_skipped_l30_39", 0) + ch * 11
                    ps["layers_skipped"] = ps.get("layers_skipped", 0) + 15
                    if self.on("l20", sw.get("ds41_psk_l20")):
                        ps["l20_cut_passes"] = ps.get("l20_cut_passes", 0) + 1; ps["l20_rows_skipped"] = ps.get("l20_rows_skipped", 0) + max(1, ch // 4)
                if first and r.hit and self.on("cr", sw.get("ds41_psk_cr")):
                    if (r.cached % (2 * self.ckint)) == 0 and "cr_not_needed" in ps: ps["cr_not_needed"] += 1
                    else:
                        ps["cr_hits"] = ps.get("cr_hits", 0) + 1; ps["cr_passes"] = ps.get("cr_passes", 0) + 1
                        ps["cr_rows_replayed"] = ps.get("cr_rows_replayed", 0) + min(r.cached, 2048)
                    if "cr_end_snapshots" in ps: ps["cr_end_snapshots"] += 1
            elif self.leak("psk") and ch >= 128:
                ps["cut_passes"] = ps.get("cut_passes", 0) + 1
            if self.on("pfctx", sw.get("dspark_pfctx")):
                self.pfctx["passes"] = self.pfctx.get("passes", 0) + 1
                self.pfctx["rows_skipped"] = self.pfctx.get("rows_skipped", 0) + max(1, int(ch * 0.87))
                self.pfctx["rows_run"] = self.pfctx.get("rows_run", 0) + max(1, int(ch * 0.13))
            if ch >= 256 and self.on("engdd", sw.get("ds41_engdd")):
                self.engdd["passes"] = self.engdd.get("passes", 0) + 2
                self.engdd["rows_deduped"] = self.engdd.get("rows_deduped", 0) + max(1, ch // 5)
                self.engdd["rows_pooled"] = self.engdd.get("rows_pooled", 0) + ch
                if "steps" in self.engdd: self.engdd["steps"] += 1
            if nd and sw.get("sched_bi_chunk") and self.on("noa", sw.get("sched_noa")):
                if first: self.noa["admitted"] = self.noa.get("admitted", 0) + 1
                self.noa["pieces"] = self.noa.get("pieces", 0) + 1; self.noa["rows"] = self.noa.get("rows", 0) + ch
                if first and "steps" in self.noa: self.noa["steps"] += 1
            elif nd and self.leak("noa"):
                self.noa["pieces"] = self.noa.get("pieces", 0) + 1
            r.pf_done += ch
            if r.cached + r.pf_done >= r.prompt_tok:
                for h in r.blocks:
                    if h not in self.cache: self.cache.add(h); self.cache_n += 1
                r.state = "decode"
                self.emit(r, 1)
        if p["pf"] and (self.on("sa", sw.get("stage_short"))):
            lo = int(sw.get("stage_short") or 1025)
            if lo <= p["n_tok"] <= 2112 or self.leak("sa"):
                for rk in self.sahb:
                    for kk, inc in (("short_steps", 1), ("short_layers", 40), ("short_units", 30), ("short_bytes", 2.9e8)):
                        if kk in rk: rk[kk] += inc
                    if sw.get("stage_pick") and "pick_kept" in rk: rk["pick_kept"] += 25; rk["pick_dropped"] += 5
        for rk in self.sahb:
            if p["n_tok"] > 1024 and "big_steps" in rk: rk["big_steps"] += 1
            hb = self.on("hb", sw.get("place_hb"))
            if hb and rk.get("rank") in self.hb_ranks:
                R = int(sw.get("place_hb") or 0) or 64
                rk["hb_reserve"] = R
                if nd: rk["hb_reserved"] = min(R, rk.get("hb_reserved", 0) + 2); rk["hb_resident"] = rk["hb_reserved"]
                if nd and self.K % 10 == 0: rk["hb_promotions"] = rk.get("hb_promotions", 0) + 1
            elif not sw.get("place_hb"):
                rk["hb_reserve"] = 0; rk["hb_reserved"] = 0; rk["hb_resident"] = 0
        if self.profile_every and p["K"] % self.profile_every == 0 and self.perop:
            t = time.time()
            for rank in range(4):
                for l in self.perop:
                    self.logline(l.replace("rank 0.", "rank %d." % rank), t)
                t += 0.005

    def emit(self, r, n):
        if r.state == "done" or r.cancel: return
        now = time.time()
        out = r.gen[r.gen_i:r.gen_i + n]
        r.gen_i += len(out)
        if out and r.t_first is None: r.t_first = now
        self.c["decode_tokens"] += len(out)
        if out: r.q.put(("tok", out))
        if r.gen_i >= len(r.gen):
            self.finish(r)

    def finish(self, r, cancelled=False):
        if r.state == "done": return
        r.state = "done"; r.t_end = time.time()
        if r.slot is not None: self.slots.pop(r.slot, None)
        c = self.c
        if cancelled:
            c["client_disconnect"] += 1; c["succ_abort"] += 1
        else:
            n = len(r.gen)
            c["gen_tokens"] += n; c["prompt_tokens_v"] += r.prompt_tok; c["succ_" + r.finish] += 1
            el = r.t_end - r.t_arr; tt = (r.t_first or r.t_end) - r.t_arr
            self.logline("I req %s done from 127.0.0.1:%d finish=%s prompt=%d (cache %d) gen=%d reason=%d ttft=%.2fs %.1f tok/s %.2fs" % (
                r.rid or ("req%d" % r.id), 40000 + r.id % 20000, r.finish, r.prompt_tok, r.cached, n, r.reason_n, tt,
                (n - 1) / max(1e-3, el - tt), el))
            c["ttft_sum"] += (r.t_first or r.t_end) - r.t_arr; c["ttft_n"] += 1
            c["e2e_sum"] += r.t_end - r.t_arr; c["e2e_n"] += 1
            c["req_gen_sum"] += n; c["req_prompt_sum"] += r.prompt_tok
        r.q.put(("end", None))
        self.reqs = [x for x in self.reqs if x.state != "done"]

    # ------------------------------------------------------------------ views
    def running(self):
        return sum(1 for r in self.reqs if r.slot is not None and r.state != "done"), sum(1 for r in self.reqs if r.slot is None and r.state != "done")

    def stats(self):
        with self.lock:
            s = copy.deepcopy(self.st_tpl)
            c = self.c
            run, wait = self.running()
            now = time.time()
            s.update(uptime_s=int(now - self.t_start), started_at=int(self.t_start), now_ms=int(now * 1000), in_flight=run + wait,
                     running=run, waiting=wait, total_requests=int(c["total_requests"]), steps=int(c["steps"]),
                     steps_prefill=int(c["steps_prefill"]), steps_mixed=int(c["steps_mixed"]), steps_decode=int(c["steps_decode"]),
                     steps_held=int(c["steps_held"]), step_ns=int(c["step_ns"]), step_ns_decode=int(c["step_ns_decode"]),
                     checkpoints_written=float(c["ckpts"]), preemptions=0, rows_unspec=int(c["rows_unspec"]),
                     draft_accept=(c["draft_accepted"] / c["draft_tokens"]) if c["draft_tokens"] else 0.0,
                     prefix_hit_rate=(c["attn_cached_tokens"] / max(1.0, c["prefill_tokens"] + c["attn_cached_tokens"])),
                     requests=[{"id": r.id, "prompt": r.prompt_tok, "cached": r.cached, "generated": r.gen_i, "state": r.state} for r in self.reqs][:64])
            if "psk" in s:
                s["psk"] = dict(self.psk); s["psk"].update(mode=self.sw.get("ds41_psk", 0), k=self.sw.get("ds41_psk_k", 8), cr=self.sw.get("ds41_psk_cr", 0))
            if "dspark_pfctx" in s:
                s["dspark_pfctx"] = dict(self.pfctx); s["dspark_pfctx"]["mode"] = self.sw.get("dspark_pfctx", 0)
            if "engdd" in s: s["engdd"] = dict(self.engdd)
            if isinstance(s.get("draft_positions"), list):
                s["draft_positions"] = [{"reached": a, "accepted": b, "rate": (b / a) if a else 0.0} for a, b in self.dpos]
            e = s.get("experts")
            if isinstance(e, dict):
                for k, ck in (("routed", "routed"), ("routed_res", "routed_res"), ("dispatches", "dispatches"), ("stage_layers", "stage_layers"),
                              ("staged_units", "staged_units"), ("staged_bytes", "staged_bytes"), ("stage_copies", "stage_copies")):
                    if k in e: e[k] = int(c[ck]) if k != "staged_bytes" else float(c[ck])
                for i, r in enumerate(e.get("ranks") or []):
                    if i >= len(self.rk): break
                    rk = self.rk[i]
                    r["routed"] = int(rk["routed"]); r["routed_res"] = int(rk["routed_res"]); r["stream_bytes"] = float(rk["stream_bytes"])
                    sl = r.get("slots")
                    if isinstance(sl, dict):
                        sl["miss"] = {k: (int(rk["miss_cold"]) if k == "cold" else 0) for k in (sl.get("miss") or {"cold": 0})}
                        sl["hit"] = {k: (int(rk["hit_trunk"]) if k == "trunk" else 0) for k in (sl.get("hit") or {"trunk": 0})}
            ln = s.get("link")
            if isinstance(ln, dict):
                for k, ck in (("ar_calls", "ar_calls"), ("ar_bytes", "ar_bytes"), ("h2d_bytes", "h2d_bytes"), ("staged_bytes", "link_staged_bytes"),
                              ("stream_bytes", "stream_bytes")):
                    if k in ln: ln[k] = c[ck] if isinstance(ln[k], float) else int(c[ck])
            for k in self.faults.get("drop", ()):
                s.pop(k, None)
            return s

    def fn4(self):
        with self.lock:
            f = copy.deepcopy(self.fn_tpl)
            f.update(switches=dict(self.sw), pending=dict(self.pending), epoch=self.epoch, applied_step=self.applied_step)
            for k, v in (("draft", self.draft), ("accept", self.accept), ("sample", self.sample), ("lend", self.lend), ("noa", self.noa),
                         ("ctxskip", self.ctxskip), ("dst", self.dst)):
                if k in f: f[k] = dict(v)
            if "sahb" in f:
                f["sahb"]["ranks"] = [dict(r) for r in self.sahb]
            return f

    def fn4_post(self, body):
        if not isinstance(body, dict): return 400, {"error": "a JSON object of switches"}
        bad = []
        for k, v in body.items():
            if k == "epoch_bump": continue
            if k not in self.sw: bad.append("unknown switch %s (known: %s, epoch_bump)" % (k, ", ".join(self.sw)))
            elif not isinstance(v, int) or isinstance(v, bool): bad.append("%s takes an integer" % k)
            elif k in RANGES and not (RANGES[k][0] <= v <= RANGES[k][1]): bad.append("%s takes %d .. %d (got %d)" % (k, RANGES[k][0], RANGES[k][1], v))
            elif k in self.faults.get("refuse", ()): bad.append("%s refused (mock fault)" % k)
        if bad:
            self.event("fn4 POST refused: %s" % "; ".join(bad)[:600])
            self.logline("I req refused from 127.0.0.1:%d -> 400 invalid_request_error: %s" % (50000 + len(bad), "; ".join(bad)[:600]))
            return 400, {"error": "; ".join(bad)}
        self.logline("I fn4: switches %s" % json.dumps(body, separators=(",", ":")))
        with self.lock:
            for k, v in body.items():
                if k == "epoch_bump":
                    if v: self.bump = True
                    continue
                self.pending[k] = v
            self.event("fn4 POST queued %s" % json.dumps(body)[:600])
            return 200, {"queued": {k: v for k, v in body.items()}, "pending": dict(self.pending), "epoch": self.epoch}

    def metrics(self):
        with self.lock:
            c = self.c
            run, wait = self.running()
            M = {"radiance:engine_steps_total": c["steps"], "radiance:engine_steps_prefill_total": c["steps_prefill"],
                 "radiance:engine_steps_mixed_total": c["steps_mixed"], "radiance:engine_steps_decode_total": c["steps_decode"],
                 "radiance:engine_steps_held_total": c["steps_held"], "radiance:engine_steps_ahead_total": c["steps_ahead"],
                 "radiance:engine_step_seconds_total": c["step_ns"] / 1e9, "radiance:engine_step_seconds_decode_total": c["step_ns_decode"] / 1e9,
                 "radiance:prefill_tokens_total": c["prefill_tokens"], "radiance:decode_tokens_total": c["decode_tokens"],
                 "radiance:draft_tokens_total": c["draft_tokens"], "radiance:draft_accepted_total": c["draft_accepted"],
                 "radiance:decode_rows_unspeculated_total": c["rows_unspec"], "vllm:generation_tokens_total": c["gen_tokens"],
                 "vllm:prompt_tokens_total": c["prompt_tokens_v"], "vllm:num_preemptions_total": 0, "vllm:num_requests_running": run,
                 "vllm:num_requests_waiting": wait, "radiance:client_disconnect_total": c["client_disconnect"],
                 "radiance:linear_checkpoints_written_total": c["ckpts"], "radiance:attn_cached_tokens_total": c["attn_cached_tokens"],
                 "radiance:prefix_hit_cut_tokens_total": c["hit_cut_tokens"],
                 "vllm:time_to_first_token_seconds_sum": c["ttft_sum"], "vllm:time_to_first_token_seconds_count": c["ttft_n"],
                 "vllm:e2e_request_latency_seconds_sum": c["e2e_sum"], "vllm:e2e_request_latency_seconds_count": c["e2e_n"],
                 "vllm:request_prompt_tokens_sum": c["req_prompt_sum"], "vllm:request_prompt_tokens_count": c["e2e_n"],
                 "vllm:request_generation_tokens_sum": c["req_gen_sum"], "vllm:request_generation_tokens_count": c["e2e_n"],
                 "radiance:draft_acceptance_ratio": (c["draft_accepted"] / c["draft_tokens"]) if c["draft_tokens"] else 0.0}
            out = []
            for l in self.m_tpl:
                m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})? (\S+)$", l)
                if not m: out.append(l); continue
                nm, lab = m.group(1), m.group(2) or ""
                v = None
                if nm in M: v = M[nm]
                elif nm == "vllm:request_success_total":
                    fr = re.search(r'finished_reason="(\w+)"', lab); v = c["succ_" + fr.group(1)] if fr else 0
                elif nm == "radiance:http_responses_total":
                    cl = re.search(r'class="(\w+)"', lab); v = c["http_" + cl.group(1)] if cl else 0
                elif nm in ("radiance:draft_position_reached_total", "radiance:draft_position_accepted_total", "radiance:draft_position_acceptance_ratio"):
                    pp = re.search(r'position="(\d+)"', lab); i = int(pp.group(1)) if pp else 0
                    if i < len(self.dpos):
                        a, b = self.dpos[i]
                        v = a if nm.endswith("reached_total") else (b if nm.endswith("accepted_total") else ((b / a) if a else 0.0))
                if v is None: out.append(l); continue
                out.append("%s%s %s" % (nm, lab, repr(float(v)) if isinstance(v, float) and not float(v).is_integer() else int(v)))
            return "\n".join(out) + "\n"


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    M = None

    def log_message(self, fmt, *a):
        pass

    def path_only(self):
        p = self.path
        if p.startswith("http://") or p.startswith("https://"):
            p = "/" + p.split("/", 3)[3] if p.count("/") >= 3 else "/"
        return p.split("?", 1)[0]

    def send(self, code, obj, ctype="application/json"):
        b = obj if isinstance(obj, (bytes, bytearray)) else (obj.encode() if isinstance(obj, str) else json.dumps(obj).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
        with self.M.lock: self.M.c["http_%dxx" % (code // 100)] += 1

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip(): return {}
        return json.loads(raw.decode("utf-8", "surrogatepass"))

    def do_GET(self):
        M, p = self.M, self.path_only()
        if p == "/health":
            if "health" in M.faults: return self.send(int(M.faults["health"]), {"status": "mock fault"})
            if time.time() < M.ready_at: return self.send(503, {"status": "loading"})
            return self.send(200, {"status": "ok"})
        if p == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": M.model, "object": "model", "owned_by": "radiance", "max_model_len": M.mml}]})
        if p == "/stats": return self.send(200, M.stats())
        if p == "/admin/fn4": return self.send(200, M.fn4())
        if p == "/metrics": return self.send(200, M.metrics(), "text/plain; version=0.0.4")
        if p == "/props": return self.send(200, {"model": M.model, "max_model_len": M.mml, "mock": True})
        if p == "/mock/state":
            with M.lock:
                return self.send(200, {"faults": {k: (sorted(v) if isinstance(v, set) else v) for k, v in M.faults.items()}, "switches": M.sw,
                                       "pending": M.pending, "epoch": M.epoch, "step": M.K, "requests": len(M.reqs), "cache_blocks": len(M.cache),
                                       "unknown": dict(M.unknown), "counters": dict(M.c), "config": {"mnbt": M.mnbt, "maxseqs": M.maxseqs, "nst": M.nst,
                                       "ckint": M.ckint, "profile_every": M.profile_every, "log_steps": M.logsteps, "hb_ranks": M.hb_ranks}})
        M.unknown["GET " + p] += 1; M.event("unknown endpoint GET %s" % p)
        return self.send(404, {"error": "not found (mock): GET %s" % p})

    def do_POST(self):
        M, p = self.M, self.path_only()
        try:
            b = self.body()
        except Exception as e:
            return self.send(400, {"error": "bad JSON: %s" % e})
        if p == "/admin/fn4":
            code, out = M.fn4_post(b); return self.send(code, out)
        if p == "/reset_prefix_cache":
            with M.lock:
                n = M.cache_n; M.cache.clear(); M.cache_n = 0
            M.event("prefix cache reset: %d blocks" % n)
            M.logline("I prefix cache: reset, %d entries dropped" % n)
            return self.send(200, {"entries_dropped": n})
        if p in ("/tokenize", "/v1/tokenize"):
            s = b.get("content") if isinstance(b.get("content"), str) else (b.get("prompt") or "")
            return self.send(200, {"tokens": [zlib.crc32(x.encode("utf-8", "surrogatepass")) % 129280 for x in Tok.pieces(s)]})
        if p == "/detokenize":
            return self.send(200, {"content": ""})
        if p == "/mock/fault":
            with M.lock:
                for k in b.get("clear") or []: M.faults.pop(k, None)
                for k, v in (b.get("set") or {}).items(): M.set_fault("%s=%s" % (k, v) if v not in (None, True, "") else k)
                return self.send(200, {"faults": {k: (sorted(v) if isinstance(v, set) else v) for k, v in M.faults.items()}})
        if p == "/mock/log":
            M.logline(str(b.get("line", ""))); return self.send(200, {"ok": True})
        if p in ("/v1/chat/completions", "/v1/completions", "/completion", "/completions"):
            return self.complete(p, b)
        M.unknown["POST " + p] += 1; M.event("unknown endpoint POST %s" % p)
        return self.send(404, {"error": "not found (mock): POST %s" % p})

    def complete(self, p, b):
        M = self.M
        chat = p == "/v1/chat/completions"
        llama = p == "/completion"
        if "err500" in M.faults and M.rng.random() < float(M.faults["err500"]):
            return self.send(500, {"error": {"message": "mock fault err500"}})
        if chat and not isinstance(b.get("messages"), list): return self.send(400, {"error": {"message": "messages missing"}})
        mt = b.get("n_predict") if llama else b.get("max_tokens", b.get("max_completion_tokens"))
        mt = int(mt) if isinstance(mt, (int, float)) and mt > 0 else 2048
        stream = bool(b.get("stream")) and not llama
        if "stall" in M.faults: time.sleep(float(M.faults["stall"]))
        r = M.new_request(b, chat, mt)
        rid = "mock-%d" % r.id
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        usage = lambda: {"prompt_tokens": r.prompt_tok, "completion_tokens": r.gen_i, "total_tokens": r.prompt_tok + r.gen_i,  # noqa: E731
                         "prompt_tokens_details": {"cached_tokens": r.cached},
                         "completion_tokens_details": {"reasoning_tokens": min(r.reason_n, r.gen_i)}}
        if not stream:
            rs, cs, tool = [], [], None
            while True:
                ev = r.q.get()
                if ev[0] == "end": break
                for kind, x in ev[1]:
                    if kind == "r": rs.append(x)
                    elif kind == "c": cs.append(x)
                    else: tool = x
            fin = r.finish if r.gen_i >= len(r.gen) else "length"
            if llama:
                return self.send(200, {"content": "".join(cs), "tokens_predicted": r.gen_i, "tokens_evaluated": r.prompt_tok, "stop": True,
                                       "stop_type": "limit" if fin == "length" else "eos", "tokens_cached": r.cached})
            if chat:
                msg = {"role": "assistant", "content": "".join(cs) if not tool else None}
                if rs: msg["reasoning_content"] = "".join(rs)
                if tool: msg["tool_calls"] = [{"id": "call_mock_%d" % r.id, "type": "function", "function": tool}]
                ch = {"index": 0, "message": msg, "finish_reason": fin}
            else:
                ch = {"index": 0, "text": "".join(cs), "finish_reason": fin}
            return self.send(200, {"id": rid, "object": obj, "created": created, "model": M.model, "choices": [ch], "usage": usage()})
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            def w(d):
                self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode()); self.wfile.flush()
            base = {"id": rid, "object": obj + ".chunk", "created": created, "model": M.model}
            if chat: w(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
            while True:
                ev = r.q.get()
                if ev[0] == "end": break
                for kind, x in ev[1]:
                    if chat:
                        if kind == "r": d = {"reasoning_content": x}
                        elif kind == "c": d = {"content": x}
                        else: d = {"tool_calls": [{"index": 0, "id": "call_mock_%d" % r.id, "type": "function", "function": x}]}
                        w(dict(base, choices=[{"index": 0, "delta": d, "finish_reason": None}]))
                    else:
                        w(dict(base, choices=[{"index": 0, "text": x if kind != "t" else json.dumps(x), "finish_reason": None}]))
            fin = r.finish if r.gen_i >= len(r.gen) else "length"
            if chat: w(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": fin}]))
            else: w(dict(base, choices=[{"index": 0, "text": "", "finish_reason": fin}]))
            if (b.get("stream_options") or {}).get("include_usage"): w(dict(base, choices=[], usage=usage()))
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            with M.lock: M.c["http_2xx"] += 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            with M.lock:
                r.cancel = True; M.cond.notify_all()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("MOCKRAD_PORT") or 18090))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--state", default=os.environ.get("MOCKRAD_STATE") or os.path.join(HERE, "state"))
    ap.add_argument("--templates", default=os.environ.get("MOCKRAD_TEMPLATES") or "templates/ds41")
    ap.add_argument("--serve-args", default="")
    ap.add_argument("--env-json", default="")
    ap.add_argument("--decode-scale", type=float, default=float(os.environ.get("MOCKRAD_DECODE_SCALE") or 0.5))
    ap.add_argument("--prefill-scale", type=float, default=float(os.environ.get("MOCKRAD_PREFILL_SCALE") or 0.1))
    ap.add_argument("--tokenizer", choices=("regex", "chars"), default=os.environ.get("MOCKRAD_TOKENIZER") or "regex")
    ap.add_argument("--cpt", type=float, default=3.9)
    ap.add_argument("--replay", default=os.environ.get("MOCKRAD_REPLAY") or "")
    ap.add_argument("--start-extra", default=os.environ.get("MOCKRAD_START_EXTRA") or "")
    ap.add_argument("--start-lib", default=os.environ.get("MOCKRAD_START_LIB") or "",
                    help="JSON list of {log, env}: real start logs and their start words; the closest to the container env is served")
    ap.add_argument("--startup-s", type=float, default=float(os.environ.get("MOCKRAD_STARTUP_S") or 1.0))
    ap.add_argument("--log-steps", action="store_true", help="write step lines even without RADIANCE_LOG_STEPS=1")
    ap.add_argument("--fault", action="append", default=[x for x in (os.environ.get("MOCKRAD_FAULTS") or "").split() if x])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.port == 8090:
        raise SystemExit("mockrad.py never listens on 8090 (the real server's port)")
    M = Mock(a)
    H.M = M
    for i in range(40):   # a restart: the previous instance may still be closing its socket
        try:
            srv = Server((a.host, a.port), H); break
        except OSError as e:
            if i == 39: M.event("cannot listen on %d: %s" % (a.port, e)); raise
            time.sleep(0.25)
    with open(os.path.join(a.state, "mockrad.port"), "w") as f: f.write("%d\n" % a.port)
    M.event("listening on %s:%d (pid %d) mnbt %d maxseqs %d nst %d ckint %d profile_every %d log_steps %s faults %s" % (
        a.host, a.port, os.getpid(), M.mnbt, M.maxseqs, M.nst, M.ckint, M.profile_every, M.logsteps, M.faults))
    import signal
    def bye(*_):
        M.stop = True
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, bye); signal.signal(signal.SIGINT, bye)
    srv.serve_forever(poll_interval=0.2)
    M.event("stopped")


if __name__ == "__main__":
    main()
