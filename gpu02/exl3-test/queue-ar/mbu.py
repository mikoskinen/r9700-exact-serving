#!/usr/bin/env python3
"""mb.py (3.11.2026) -- fixed-load micro-benchmark for the DS41 settings sweeps (server :8090, no key).

  mb.py --tag T [--phases c4,c8,pf,turns] [--dur 200] [--warm 20] [--ctx 1000] [--gen 512]

c<N>  : N closed-loop streams for --dur s (each finishes its request and sends the next; distinct ~--ctx-token code
        prompt + task, fresh nonce, reasoning_effort medium, server-default sampling like cc_sim, max_tokens --gen).
        Window [warm, dur]: /metrics deltas -> ms per decode step, tokens per decode step, acceptance, decode / prefill
        share of engine busy time; aggregate generated tok/s (generation_tokens_total delta / window); per-request decode
        tok/s p50 (requests started in the window); TTFT p50.
pf    : 4 decode streams + one fresh ~32k-token prefill at warm+15 s: the prefill's TTFT and the streams' chunk rate
        while it runs vs before it (decode stall from prefill chunks).
turns : 1 session: ~60k-token context, then 5 turns each adding ~1.5k tokens (previous reply + tool-like text): TTFT,
        cached tokens and recompute (prompt - cached) per turn.
One JSON line per phase, prefixed "MB ".
"""
import argparse, glob, json, os, re, statistics, sysconfig, threading, time, urllib.request

URL = __import__("os").environ.get("MB_URL", "http://127.0.0.1:8090")   # mbu.py (fn4 12.11.): mb.py with MB_URL
SRC = sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/**/*.py", recursive=True))
TASKS = ["Write a Python implementation of a red-black tree with tests.", "Explain this code and rewrite the core in Rust.",
         "Write a C LRU cache with a hash map and a doubly linked list.", "Write a TypeScript typed event emitter with tests.",
         "Write a Go HTTP server with graceful shutdown and middleware.", "Refactor the code above into smaller functions.",
         "Write a SQL schema for a library system and ten queries.", "Write a bash log-rotation script with tests."]


def text(n_chars, off):
    parts, k = [], 0
    for f in SRC[off % len(SRC):] + SRC:
        try:
            s = open(f, encoding="utf-8").read()
        except Exception:
            continue
        parts.append(f"\n# === {os.path.basename(f)} ===\n" + s); k += len(s)
        if k >= n_chars:
            break
    return "".join(parts)[:n_chars]


def metrics():
    txt = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    d = {}
    for l in txt.splitlines():
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})? ([0-9.eE+-]+)$", l)
        if m and "bucket" not in m.group(1):
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
    return d


def mdelta(b, a):
    g = lambda k: a.get(k, 0) - b.get(k, 0)  # noqa: E731
    sd, ss = g("radiance:engine_steps_decode_total"), g("radiance:engine_step_seconds_decode_total")
    st, sst = g("radiance:engine_steps_total"), g("radiance:engine_step_seconds_total")
    return dict(ms_decode_step=round(1000 * ss / max(1, sd), 1), tok_per_decode_step=round(g("radiance:decode_tokens_total") / max(1, sd), 2),
                acceptance=round(g("radiance:draft_accepted_total") / max(1, g("radiance:draft_tokens_total")), 3),
                decode_share=round(ss / max(1e-9, sst), 3), steps=int(st), busy_s=round(sst, 1),
                prefill_tok=int(g("radiance:prefill_tokens_total")), cut_tok=int(g("radiance:prefix_hit_cut_tokens_total")),
                gen_tok=int(g("vllm:generation_tokens_total")), dec_tok=int(g("radiance:decode_tokens_total")),
                preempt=int(g("vllm:num_preemptions_total")))


def stream(msgs, max_tokens, effort="medium", stamps=None, extra=None):
    p = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if effort:
        p["reasoning_effort"] = effort
    if extra:
        p.update(extra)
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                s = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if s:
                    now = time.time(); tf = tf or now; tl = now; out.append(s)
                    if stamps is not None:
                        stamps.append(now)
    u = usage or {}
    n = u.get("completion_tokens") or 0
    return dict(t0=t0, ttft=(tf - t0) if tf else None, n=n, dec=((n - 1) / (tl - tf)) if (tf and tl and tl > tf and n > 1) else None,
                prompt=u.get("prompt_tokens"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), text="".join(out))


def closed_loop(n, dur, ctx, gen, stop_evt, recs, stamps_by=None):
    def worker(i):
        k = 0
        while not stop_evt.is_set():
            q = f"# req {time.time_ns()} {i}\n" + text(int(ctx * 3.9), i * 11 + k) + "\n\n" + TASKS[(i + k) % len(TASKS)]
            try:
                r = stream([{"role": "user", "content": q}], gen, stamps=None if stamps_by is None else stamps_by[i])
                r["w"] = i; recs.append(r)
            except Exception as e:
                recs.append(dict(err=str(e)[:200], t0=time.time()))
            k += 1
    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in th:
        t.start()
    return th


def phase_c(n, a):
    stop = threading.Event(); recs = []
    t_start = time.time()
    th = closed_loop(n, a.dur, a.ctx, a.gen, stop, recs)
    time.sleep(a.warm); b = metrics(); tw0 = time.time()
    time.sleep(a.dur - a.warm); m = metrics(); tw1 = time.time()
    stop.set()
    for t in th:
        t.join(timeout=600)
    win = [r for r in recs if not r.get("err") and tw0 <= r["t0"] < tw1]
    dec = [r["dec"] for r in win if r.get("dec")]; tt = [r["ttft"] for r in win if r.get("ttft")]
    d = mdelta(b, m)
    # agg_dec_tps: engine decode-token counter over the window (per step, includes partial requests). agg_gen_tps_q: the
    # vllm generation counter, which moves only when a request completes (quantized by requests x max_tokens; 3.11. fix).
    return dict(phase=f"c{n}", window_s=round(tw1 - tw0, 1), agg_dec_tps=round(d["dec_tok"] / (tw1 - tw0), 1),
                agg_gen_tps_q=round(d["gen_tok"] / (tw1 - tw0), 1), per_stream_dec_p50=round(statistics.median(dec), 1) if dec else None,
                ttft_p50=round(statistics.median(tt), 2) if tt else None, requests=len(win), errors=sum(1 for r in recs if r.get("err")), **d)


def phase_pf(a):
    stop = threading.Event(); recs = []; stamps = {i: [] for i in range(4)}
    th = closed_loop(4, 0, a.ctx, a.gen, stop, recs, stamps)
    time.sleep(a.warm + 15)
    b = metrics()
    q = f"# pf {time.time_ns()}\n" + text(int(32768 * 3.9), 777) + "\n\nSummarise the code above in one sentence."
    r = stream([{"role": "user", "content": q}], 1, effort="none")
    m = metrics()
    p0, p1 = r["t0"], r["t0"] + (r["ttft"] or 0)
    time.sleep(10); stop.set()
    for t in th:
        t.join(timeout=600)
    allst = sorted(s for v in stamps.values() for s in v)
    before = [s for s in allst if p0 - 20 <= s < p0]; during = [s for s in allst if p0 <= s < p1]
    rb = len(before) / 20.0; rd = len(during) / max(1e-9, p1 - p0)
    gaps = []
    for v in stamps.values():
        w = [s for s in v if p0 - 1 <= s <= p1 + 1]
        gaps += [y - x for x, y in zip(w, w[1:])]
    d = mdelta(b, m)
    return dict(phase="pf", prefill_tokens=r["prompt"], prefill_ttft=round(r["ttft"], 2), prefill_tps=round(r["prompt"] / r["ttft"], 1),
                chunk_rate_before=round(rb, 1), chunk_rate_during=round(rd, 1), decode_kept=round(rd / max(1e-9, rb), 3),
                max_gap_s=round(max(gaps), 2) if gaps else None, steps_during=d["steps"], ms_step_during=round(1000 * d["busy_s"] / max(1, d["steps"]), 1))


def phase_turns(a):
    base = text(int(60000 * 3.9), 1234)
    msgs = [{"role": "system", "content": f"session {time.time_ns()}"}, {"role": "user", "content": base + "\n\nReview the code above. Start with the first file."}]
    out = []
    for t in range(6):
        b = metrics()
        r = stream(msgs, 300, effort="none")
        m = metrics(); d = mdelta(b, m)
        out.append(dict(turn=t, prompt=r["prompt"], cached=r["cached"], recompute=(r["prompt"] or 0) - (r["cached"] or 0),
                        ttft=round(r["ttft"], 2), cut=d["cut_tok"]))
        msgs = msgs + [{"role": "assistant", "content": r["text"]},
                       {"role": "user", "content": "Tool output:\n" + text(int(1500 * 3.9), 300 + t * 7) + "\n\nContinue the review."}]
    hits = out[1:]
    return dict(phase="turns", ttft_hit_mean=round(statistics.mean(o["ttft"] for o in hits), 2),
                recompute_mean=int(statistics.mean(o["recompute"] for o in hits)), cold_ttft=out[0]["ttft"], cold_prompt=out[0]["prompt"],
                turns=[(o["prompt"], o["cached"], o["ttft"]) for o in out])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--phases", default="c4,c8,pf,turns")
    ap.add_argument("--dur", type=float, default=200)
    ap.add_argument("--warm", type=float, default=20)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    a = ap.parse_args()
    with urllib.request.urlopen(URL + "/v1/models", timeout=60) as r:
        MODEL = json.load(r)["data"][0]["id"]
    for ph in a.phases.split(","):
        if ph.startswith("c"):
            res = phase_c(int(ph[1:]), a)
        elif ph == "pf":
            res = phase_pf(a)
        else:
            res = phase_turns(a)
        print("MB " + json.dumps(dict(tag=a.tag, **res)), flush=True)
