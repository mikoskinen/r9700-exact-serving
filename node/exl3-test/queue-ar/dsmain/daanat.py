#!/usr/bin/env python3
"""daanat.py (11.11.2026, queue-ar dsmain, ds-main-port, lever DA screen, tag xda2) -- THE REJECTION ANATOMY OF THE DSPARK
DRAFTER ON THE EXL3 2.25 TARGET: how much of today's prose miss a drafter adapted to this target (self-distillation, lever DA)
could win back at most. Measures, changes nothing: the proposals are the running drafter's, the output is the target's.

  daanat.py collect --url URL --log ENGINE_LOG --out PREFIX [--prose 20] [--code 6] [--max-tokens 384] [--top 20]
  daanat.py analyse PREFIX-rows.jsonl [--now-essay 64.54] [--now-code 104.0]
  daanat.py selftest

collect: one request at a time (c1) to /v1/chat/completions (temperature 0, reasoning_effort none, logprobs + top_logprobs N,
  never a user transcript: the synthetic prompts below, the 1-user probe's essay and code prompts first). The server returns per
  generated token its id, its raw logprob and the N likeliest alternatives (core/server/oai.cpp lp_entry; under speculation each
  emitted token's logprobs come from its own verify row, core/engine.cpp:1902). The engine runs with -v, so every DSpark block
  pass logs "drafter 'dspark': sequence S proposes d1 .. d5" (core/engine_draft.cpp:675); the lines written while the request
  ran (ENGINE_LOG = the followed docker log, byte offsets before / after) are that request's proposals. Then the first prose
  prompt is sent again WITHOUT logprobs: its text must equal the logprobs run's (logprobs change no output).
analyse: per request the replay of perpos.py (a block proposed after anchor p predicts ids[p+1 .. p+5]; its accepted run is the
  leading matches; the anchor moves by run + 1; first anchor 0 or 1, the better). At every scored position the target's own
  distribution is known (greedy: the emitted token is its top 1, p_top1 = exp(logprob)). Each block's FIRST REJECTION is
  classed:
    A confident : p_top1 >= 0.8 -- the target was sure and the drafter proposed something else (a drafter matched to this
                  target should mostly get these: the part DA addresses most directly)
    B near tie  : not A, and the drafter's token is in the target's top N within ln 2 of its top 1 (p_d >= p_top1 / 2) --
                  where the 2.25 quantisation can flip the argmax away from the full-precision target's (the label shift)
    C other     : the target itself was unsure and the drafter's token was not near its top (prose entropy; no drafter
                  that predicts 5 tokens ahead from 3 layers wins most of these)
  Bounds on tokens a step at the fixed window 5 (chain model 1 + sum_k prod_{j<=k} c_j over conditional acceptance c_j):
    now      c_j = accepted / reached (and the replay's own mean run + 1, which the model must match)
    +A       every A rejection accepted
    +A+B     every A and B rejection accepted (the ADDRESSABLE bound: DA's upper bound in this screen)
    dist     every rejected position credited p_top1 (a drafter that matches the target's own next-token distribution there)
    perfect  6.0
  and the estimated tok/s of each at today's round time (walk2 B, adaptive: essay 64.54 tok/s at 1.90-1.93 tokens a step, code
  104.0) -- an ESTIMATE (round time held; deeper accepted chains make adaptive rounds longer), with the tokens a step the 1-user
  prose target (>= 90 tok/s) needs at today's round: x 90 / 64.54 = x 1.395.
Prints "DALINE ..." lines (one per finding) and one "DASUM {json}". Standard library only."""
import argparse, json, math, os, re, sys, time, urllib.request

PAT = re.compile(r"drafter '[^']*': sequence (\d+) proposes((?: -?\d+)+)")

# the 1-user probe's prompts first (ds41_tests.py PROMPTS, 3.11.), then synthetic ones written for this screen
PROSE = [
    "Write a long, detailed essay (about 900 words) on the history of the printing press and its effects on European science, "
    "religion and politics. Use several paragraphs.",
    "Explain, for a general audience, how vaccines train the immune system. Use several paragraphs and concrete examples.",
    "Write a short story of about 600 words about a lighthouse keeper who finds a message in a bottle.",
    "Write a well-structured essay of about 700 words on the causes and consequences of the 2008 financial crisis.",
    "Write a travel guide section that describes a three-day visit to Kyoto, with practical tips for each day.",
    "Write a persuasive letter to a city council that argues for more protected bicycle lanes, with concrete examples.",
    "Describe the water cycle and its role in the climate, as a section of a textbook chapter for high-school students.",
    "Write a detailed review of a mid-range mechanical keyboard: build quality, typing feel, software and value for money.",
    "Summarise the plot and the main themes of Shakespeare's Hamlet in an essay of about 600 words.",
    "Write a blog post with practical advice for an adult who is starting to learn the piano.",
    "Explain in plain language how a modern CPU executes instructions: pipelining, caches and branch prediction.",
    "Write a historical narrative about the construction of the Panama Canal, from the first French attempt to its opening.",
    "Write a reflective essay on what makes a good teacher, with examples from different subjects.",
    "Write a friendly guide to the basics of personal finance: budgeting, saving, debt and investing.",
    "Write the introduction and the full method of a recipe for a traditional Neapolitan pizza, with some history of the dish.",
    "Write a news article about a fictional small town that opens its first public library.",
    "Compare solar, wind, hydro and geothermal power in an essay that discusses their costs and their limits.",
    "Write a graduation speech about curiosity and persistence.",
    "Explain the theory of evolution by natural selection with three well-known examples.",
    "Write a long, descriptive passage about a walk through a forest in autumn.",
]
CODE = [
    "Write a complete, well-commented Python module that implements a thread-safe LRU cache with TTL expiry, a decorator that "
    "uses it, and a set of unittest test cases. Aim for about 150 lines of code.",
    "Write a Python script that reads a CSV file of transactions and prints monthly totals per category; use argparse and "
    "handle malformed rows.",
    "Implement a binary search tree in C++ with insert, delete, find and in-order traversal, and a small main() that tests it.",
    "Write a JavaScript module with debounce and throttle functions, JSDoc comments and usage examples.",
    "Write a Rust program that tokenizes simple arithmetic expressions and evaluates them with operator precedence, with unit "
    "tests.",
    "Write a Bash script that backs up a directory to a dated tar.gz archive, keeps the last 7 archives and logs what it did.",
]
NOW = {"prose": 64.54, "code": 104.0}      # walk2 B (dspark_walk 1, adaptive) tok/s: essay, code (LEVERS W1, 11.11.)
TARGET = 90.0
LN2 = math.log(2.0)
BINS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0000001]


def http(url, body=None, timeout=900):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def model_id(url):
    try:
        return (http(url + "/v1/models", timeout=60).get("data") or [{}])[0].get("id") or "default"
    except Exception:
        return "default"


def chat(url, model, prompt, n, top):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0,
            "top_p": 1.0, "reasoning_effort": "none", "stream": False}
    if top:
        body.update({"logprobs": True, "top_logprobs": top})
    return http(url + "/v1/chat/completions", body)


def logsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def collect(a):
    url = a.url.rstrip("/")
    model = model_id(url)
    jobs = [("prose", i, p) for i, p in enumerate(PROSE[:a.prose])] + [("code", i, p) for i, p in enumerate(CODE[:a.code])]
    first_text = None
    with open(a.out + "-rows.jsonl", "w") as fo:
        for kind, i, p in jobs:
            o0 = logsize(a.log)
            t0 = time.time()
            d = chat(url, model, p, a.max_tokens, a.top)
            dt = time.time() - t0
            time.sleep(a.settle)                      # the followed docker log catches up
            o1 = logsize(a.log)
            with open(a.log, "rb") as f:
                f.seek(o0)
                chunk = f.read(max(0, o1 - o0)).decode(errors="replace")
            props = {}
            for m in PAT.finditer(chunk):
                props.setdefault(m.group(1), []).append([int(x) for x in m.group(2).split()])
            seq, blocks = max(props.items(), key=lambda kv: len(kv[1])) if props else (None, [])
            c = d["choices"][0]
            lp = (c.get("logprobs") or {}).get("content") or []
            text = (c.get("message") or {}).get("content") or ""
            if kind == "prose" and i == 0:
                first_text = text
            row = {"kind": kind, "i": i, "seq": seq, "seqs_in_window": len(props), "finish": c.get("finish_reason"),
                   "s": round(dt, 3), "ids": [e.get("id") for e in lp], "lp": [e.get("logprob") for e in lp],
                   "top": [[[t.get("id"), t.get("logprob")] for t in (e.get("top_logprobs") or [])] for e in lp],
                   "blocks": blocks, "timings": d.get("timings"), "usage": d.get("usage"), "chars": len(text)}
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            print("DACOLLECT " + json.dumps({"kind": kind, "i": i, "tokens": len(lp), "blocks": len(blocks), "seqs": len(props),
                                             "s": round(dt, 2), "finish": c.get("finish_reason"),
                                             "draft_n": (d.get("timings") or {}).get("draft_n"),
                                             "draft_n_accepted": (d.get("timings") or {}).get("draft_n_accepted")}), flush=True)
    # logprobs change no output: the first prose prompt again without them
    lpid = "not run"
    if a.prose and first_text is not None:
        d = chat(url, model, PROSE[0], a.max_tokens, 0)
        t2 = ((d["choices"][0].get("message") or {}).get("content") or "")
        f = next((k for k, (x, y) in enumerate(zip(first_text, t2)) if x != y), None if len(first_text) == len(t2) else min(len(first_text), len(t2)))
        lpid = ("IDENTICAL (%d chars)" % len(t2)) if t2 == first_text else ("DIFFERENT at char %s (%d vs %d chars)" % (f, len(first_text), len(t2)))
    with open(a.out + "-lpid.txt", "w") as f:
        f.write(lpid + "\n")
    print("DALPID " + lpid)
    return 0


def replay(ids, blocks, p0):
    """-> list of (depth 0..4, pos, accepted, proposed id), runs"""
    p, recs, runs = p0, [], []
    for d in blocks:
        if p + 1 >= len(ids):
            break
        run = 0
        for j, t in enumerate(d[:5]):
            pos = p + 1 + j
            if pos >= len(ids):
                break
            if t == ids[pos]:
                recs.append((j, pos, True, t)); run += 1
            else:
                recs.append((j, pos, False, t)); break
        runs.append(run)
        p += run + 1
    return recs, runs


def chain(c):
    t, prod = 1.0, 1.0
    for x in c:
        prod *= x
        t += prod
    return t


def new_acc():
    return {"reached": [0] * 5, "acc": [0] * 5, "A": [0] * 5, "B": [0] * 5, "C": [0] * 5, "ptrej": [0.0] * 5,
            "ptsum": [0.0] * 5, "rank": {"2": 0, "3-5": 0, "6-20": 0, ">20": 0}, "bins": [[0, 0] for _ in BINS[:-1]],
            "conf9": [0, 0], "blocks": 0, "tokens": 0, "runsum": 0, "eng_acc": 0, "rep_acc": 0, "eng_n": 0, "rows": 0,
            "rows_bad": 0}


def score_row(r, S):
    ids, lp, top, blocks = r["ids"], r["lp"], r["top"], r["blocks"]
    if not ids or not blocks or len(lp) != len(ids) or len(top) != len(ids):
        S["rows_bad"] += 1
        return
    best = max((replay(ids, blocks, p0) + (p0,) for p0 in (0, 1)), key=lambda x: (sum(1 for q in x[0] if q[2]), -x[2]))
    recs, runs, _ = best
    S["rows"] += 1
    S["blocks"] += len(runs); S["runsum"] += sum(x + 1 for x in runs); S["tokens"] += len(ids)
    S["rep_acc"] += sum(1 for q in recs if q[2])
    tm = r.get("timings") or {}
    if tm.get("draft_n_accepted") is not None:
        S["eng_acc"] += int(tm["draft_n_accepted"]); S["eng_n"] += 1
    for j, pos, ok, d in recs:
        lpt = lp[pos]
        pt = math.exp(lpt) if lpt is not None else 0.0
        S["reached"][j] += 1; S["ptsum"][j] += pt
        b = next(k for k in range(len(BINS) - 1) if BINS[k] <= pt < BINS[k + 1])
        S["bins"][b][0] += 1
        if pt >= 0.9:
            S["conf9"][0] += 1
        if ok:
            S["acc"][j] += 1; S["bins"][b][1] += 1
            if pt >= 0.9:
                S["conf9"][1] += 1
            continue
        alts = sorted(((t[1], t[0]) for t in top[pos] if t[1] is not None), reverse=True)
        rank = next((k + 1 for k, (_, tid) in enumerate(alts) if tid == d), None)
        lpd = next((v for v, tid in alts if tid == d), None)
        S["rank"]["2" if rank == 2 else "3-5" if rank and rank <= 5 else "6-20" if rank else ">20"] += 1
        S["ptrej"][j] += pt
        if pt >= 0.8:
            S["A"][j] += 1
        elif lpd is not None and lpt is not None and lpt - lpd <= LN2:
            S["B"][j] += 1
        else:
            S["C"][j] += 1


def summarise(kind, S, now_tps):
    out = {"kind": kind, "rows": S["rows"], "rows_bad": S["rows_bad"], "blocks": S["blocks"], "tokens": S["tokens"]}
    if not S["blocks"]:
        return out, ["%s: no scored block (rows %d, bad %d)" % (kind, S["rows"], S["rows_bad"])]
    R = S["reached"]
    c = lambda num: [num[j] / R[j] if R[j] else 0.0 for j in range(5)]
    c_now = c(S["acc"])
    c_a = c([S["acc"][j] + S["A"][j] for j in range(5)])
    c_ab = c([S["acc"][j] + S["A"][j] + S["B"][j] for j in range(5)])
    c_d = c([S["acc"][j] + S["ptrej"][j] for j in range(5)])
    rep = S["runsum"] / S["blocks"]
    b = {"replay": rep, "now": chain(c_now), "+A": chain(c_a), "+A+B": chain(c_ab), "dist": chain(c_d), "perfect": 6.0}
    rej = sum(S["A"]) + sum(S["B"]) + sum(S["C"])
    need = rep * TARGET / now_tps if now_tps else None
    out.update({"cond_now": c_now, "cond_+A": c_a, "cond_+A+B": c_ab, "cond_dist": c_d, "bounds": b, "reached": R,
                "accepted": S["acc"], "A": S["A"], "B": S["B"], "C": S["C"], "rank": S["rank"],
                "mean_ptop1": [S["ptsum"][j] / R[j] if R[j] else None for j in range(5)],
                "bins": {"%.2f-%.2f" % (BINS[k], min(1.0, BINS[k + 1])): S["bins"][k] for k in range(len(BINS) - 1)},
                "acc_at_ptop1_ge_0.9": [S["conf9"][1], S["conf9"][0]], "engine_accepted": S["eng_acc"] if S["eng_n"] else None,
                "replay_accepted": S["rep_acc"], "now_tps_ref": now_tps, "needed_tokens_a_step_for_90": need})
    f3 = lambda xs: " ".join("%.3f" % x for x in xs)
    pct = lambda x, y: "%+.1f %%" % ((x / y - 1) * 100) if y else "-"
    est = lambda t: now_tps * t / rep if now_tps else None
    L = []
    L.append("%s: %d requests (%d unusable), %d tokens, %d blocks; tokens a step (fixed window 5) replay %.3f, chain model %.3f; "
             "drafted accepted: engine %s vs replay %d; conditional acceptance p1..p5 %s (accepted / reached %s)" % (
                 kind, S["rows"], S["rows_bad"], S["tokens"], S["blocks"], rep, b["now"],
                 S["eng_acc"] if S["eng_n"] else "-", S["rep_acc"], f3(c_now),
                 " ".join("%d/%d" % (S["acc"][j], R[j]) for j in range(5))))
    L.append("%s: first rejections %d -- A confident (target p_top1 >= 0.8) %d (%.1f %%), B near tie (drafter's token within ln 2 "
             "of the target's top 1) %d (%.1f %%), C other %d (%.1f %%); by position A %s / B %s / C %s; the drafter's token's rank "
             "in the target's top %s: 2 %d, 3-5 %d, 6-20 %d, beyond %d" % (
                 kind, rej, sum(S["A"]), 100.0 * sum(S["A"]) / max(1, rej), sum(S["B"]), 100.0 * sum(S["B"]) / max(1, rej),
                 sum(S["C"]), 100.0 * sum(S["C"]) / max(1, rej), S["A"], S["B"], S["C"], "N", S["rank"]["2"],
                 S["rank"]["3-5"], S["rank"]["6-20"], S["rank"][">20"]))
    L.append("%s: target confidence -- mean p_top1 at p1..p5 %s; acceptance by p_top1 bin %s; acceptance where p_top1 >= 0.9: "
             "%d/%d (%.3f)" % (kind, " ".join("-" if x is None else "%.3f" % x for x in out["mean_ptop1"]),
                                " ".join("[%s] %d/%d" % (k, v[1], v[0]) for k, v in out["bins"].items()),
                                S["conf9"][1], S["conf9"][0], S["conf9"][1] / max(1, S["conf9"][0])))
    L.append("%s: BOUNDS tokens a step (chain, fixed 5) now %.3f | +A %.3f (%s) | +A+B %.3f (%s, DA's addressable bound) | "
             "distribution-matched %.3f (%s) | perfect 6.0 -- est. tok/s at today's round (walk2 %s %.2f tok/s at the replay's "
             "%.3f): +A %.1f, +A+B %.1f, dist %.1f; the 1-user target 90 tok/s needs ~%.3f tokens a step (x %.3f) -> +A+B %s" % (
                 kind, b["now"], b["+A"], pct(b["+A"], b["now"]), b["+A+B"], pct(b["+A+B"], b["now"]), b["dist"],
                 pct(b["dist"], b["now"]), "essay" if kind == "prose" else "code", now_tps or 0, rep,
                 est(b["+A"] * rep / b["now"]) or 0, est(b["+A+B"] * rep / b["now"]) or 0, est(b["dist"] * rep / b["now"]) or 0,
                 need or 0, (TARGET / now_tps) if now_tps else 0,
                 ("REACHES it" if need and b["+A+B"] * rep / b["now"] >= need else "does NOT reach it")))
    return out, L


def analyse(a):
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    res, lines = {}, []
    for kind in ("prose", "code"):
        S = new_acc()
        for r in rows:
            if r.get("kind") == kind:
                score_row(r, S)
        o, L = summarise(kind, S, a.now_essay if kind == "prose" else a.now_code)
        res[kind] = o; lines += L
    for l in lines:
        print("DALINE " + l)
    print("DASUM " + json.dumps(res))
    return 0 if all(res[k].get("blocks") for k in res if res[k].get("rows")) else 1


def selftest():
    """a fabricated request: the replay, the classes and the bounds on known numbers"""
    ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lp = [math.log(0.95)] * len(ids)
    lp[2] = math.log(0.5); lp[7] = math.log(0.4)
    top = [[[t, lp[k]], [999, math.log(0.01)]] for k, t in enumerate(ids)]
    top[2] = [[12, math.log(0.5)], [77, math.log(0.3)]]          # pos 2: the drafter's 77 is the target's #2, within ln 2 -> B
    top[7] = [[17, math.log(0.4)], [55, math.log(0.05)]]         # pos 7: 66 not in the top -> C
    blocks = [[11, 77, 1, 1, 1],                                  # anchor 0: p1 ok, p2 rejects at pos 2 (B) -> anchor 2
              [13, 14, 15, 16, 66],                               # anchor 2: 4 accepted, p5 66 vs 17 at pos 7 (C) -> anchor 7
              [18, 5, 0, 0, 0],                                   # anchor 7: p1 ok, p2 5 vs 19 at pos 9, p_top1 .95 (A) -> 9
              [20, 21, 0, 0, 0]]                                  # anchor 9: 2 accepted, then the end
    r = {"kind": "prose", "ids": ids, "lp": lp, "top": top, "blocks": blocks, "timings": {"draft_n_accepted": 8}}
    S = new_acc(); score_row(r, S)
    ok = (S["acc"] == [4, 2, 1, 1, 0] and S["reached"] == [4, 4, 1, 1, 1] and S["A"] == [0, 1, 0, 0, 0]
          and S["B"] == [0, 1, 0, 0, 0] and S["C"] == [0, 0, 0, 0, 1] and S["rank"] == {"2": 1, "3-5": 0, "6-20": 0, ">20": 2}
          and S["runsum"] == 2 + 5 + 2 + 3 and S["blocks"] == 4 and S["rep_acc"] == 8 and S["eng_acc"] == 8)
    o, L = summarise("prose", S, 64.54)
    ok = ok and abs(o["bounds"]["perfect"] - 6.0) < 1e-9 and o["bounds"]["+A+B"] >= o["bounds"]["+A"] >= o["bounds"]["now"]
    ok = ok and o["bounds"]["dist"] >= o["bounds"]["now"] and abs(chain([1, 1, 1, 1, 1]) - 6.0) < 1e-9
    print("DASELFTEST " + ("ok" if ok else "FAIL") + " " + json.dumps({"acc": S["acc"], "reached": S["reached"], "A": S["A"], "B": S["B"],
                                                                     "C": S["C"], "rank": S["rank"], "bounds": o.get("bounds")}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd")
    c = sp.add_parser("collect")
    c.add_argument("--url", required=True); c.add_argument("--log", required=True); c.add_argument("--out", required=True)
    c.add_argument("--prose", type=int, default=20); c.add_argument("--code", type=int, default=6)
    c.add_argument("--max-tokens", type=int, default=384); c.add_argument("--top", type=int, default=20)
    c.add_argument("--settle", type=float, default=1.5)
    n = sp.add_parser("analyse")
    n.add_argument("rows"); n.add_argument("--now-essay", type=float, default=NOW["prose"])
    n.add_argument("--now-code", type=float, default=NOW["code"])
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "collect":
        return collect(a)
    if a.cmd == "analyse":
        return analyse(a)
    if a.cmd == "selftest":
        return selftest()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
