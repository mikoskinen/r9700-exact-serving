#!/usr/bin/env python3
"""ncone.py (12.11.2026, queue-ar/dsmain/nc, ds-main-port NC-profile builder) -- ALONE newcomers for dsm-ncprof.sh.

N fresh ~CTX-token prompts sent ONE AT A TIME to the DS41 server, each answered with ONE token (max_tokens 1), so every
step the engine runs for them is that prompt's own prefill step (pf1.py's one-token pattern). The prompt is mb.py's
closed-loop prompt, built the same way (loaded READ ONLY from --mb, importlib, no bytecode): a nonce first
("# req <ns> <i>"), mb.text(ctx * 3.9 chars, i * 11 + k), "\\n\\n", one of mb.TASKS -- the same text, template and
reasoning effort ("medium", mb.stream's default) as a c8 newcomer, so the prompt is ~1099 tokens like the c8 ones.
Temperature --temperature (0). Between prompts: a --pause, then GET /stats (a compact copy, NCSTATS fields), so
snapshot j and j + 1 bracket prompt j exactly (one step each).

  ncone.py --mb ~/gpu/ds41/mb.py --url URL --n 12 --tag T [--ctx 1000] [--k0 500] --out REQS.jsonl --stats STATS.jsonl
  ncone.py --mb ~/gpu/ds41/mb.py --selftest      offline: mb.py loads, the prompt builder and the /stats compactor work.

REQS.jsonl: one line a prompt {j, t0, ttft, t_resp, t_end, prompt, cached, n, err}; t_resp = t_end - t0 (a one-token
reply may carry no content delta, so ttft can be None). STATS.jsonl: {k, t, s: compact /stats}. Prints "NCONE {summary}".
Standard library only.
"""
import argparse, importlib.util, json, statistics, sys, time, urllib.request

sys.dont_write_bytecode = True


def load(path):
    spec = importlib.util.spec_from_file_location("mb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compact(st):
    """The /stats fields the NC item reads (a full /stats is ~21 KB with the 4128-unit tier list)."""
    if not isinstance(st, dict): return {}
    e, ln = st.get("experts") or {}, st.get("link") or {}
    keep = ("steps", "steps_prefill", "steps_mixed", "steps_decode", "steps_held", "step_ns", "step_ns_decode",
            "checkpoints_written", "running", "waiting", "total_requests", "preemptions", "prefix_evictions")
    out = {k: st.get(k) for k in keep if k in st}
    out["experts"] = {k: e.get(k) for k in ("staged_units", "staged_bytes", "stage_copies", "stage_layers", "stage_overflow",
                                             "routed", "routed_res", "dispatches", "promotions", "demotions", "n_units") if k in e}
    out["experts"]["tiers"] = {x.get("name"): x.get("units") for x in e.get("tiers") or [] if isinstance(x, dict)}
    rk = []
    for r in e.get("ranks") or []:
        sl = r.get("slots") or {}
        rk.append({"rank": r.get("rank"), "stream_bytes": r.get("stream_bytes"), "routed": r.get("routed"),
                   "routed_res": r.get("routed_res"), "vram": r.get("vram"),
                   "miss": dict(sl.get("miss") or {}), "hit": dict(sl.get("hit") or {})})
    out["experts"]["ranks"] = rk
    out["link"] = {k: ln.get(k) for k in ("h2d_bytes", "d2h_bytes", "staged_bytes", "stream_bytes", "ar_bytes", "ar_calls",
                                          "prefetch_bytes") if k in ln}
    if isinstance(st.get("psk"), dict): out["psk"] = st["psk"]
    return out


def http(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def prompt(mb, j, ctx, k0):
    i, k = j % 8, k0 + j // 8
    return f"# req {time.time_ns()} {i}\n" + mb.text(int(ctx * 3.9), i * 11 + k) + "\n\n" + mb.TASKS[(i + k) % len(mb.TASKS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--k0", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--out")
    ap.add_argument("--stats")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb)
    need = ("stream", "text", "TASKS", "URL")
    miss = [n for n in need if not hasattr(mb, n)]
    if a.selftest:
        ok = not miss
        q = prompt(mb, 9, 1000, 500) if ok else ""
        c = compact({"steps": 3, "experts": {"staged_units": 5, "tiers": [{"name": "vram", "units": 1}],
                                             "ranks": [{"rank": 0, "stream_bytes": 7, "slots": {"miss": {"cold": 1}}}]},
                     "link": {"h2d_bytes": 9}})
        ok = ok and q.startswith("# req ") and " 1\n" in q[:40] and len(q) > 3000 and c["experts"]["ranks"][0]["miss"]["cold"] == 1 \
            and c["experts"]["tiers"]["vram"] == 1 and c["link"]["h2d_bytes"] == 9
        print("SELFTEST " + ("OK: mb.py %s loads (%s), prompt %d chars, compactor" % (a.mb, ", ".join(need), len(q)) if ok
                             else "FAILED: missing %s or prompt/compactor wrong" % miss))
        return 0 if ok else 1
    if miss or not a.out or not a.stats:
        print("NCONE " + json.dumps({"error": "mb.py lacks %s or no --out / --stats" % miss}), flush=True)
        return 1
    mb.URL = a.url
    mb.MODEL = http(a.url + "/v1/models", 60)["data"][0]["id"]
    recs = []
    with open(a.stats, "a") as fs, open(a.out, "a") as fo:
        def snap(k):
            try: s = compact(http(a.url + "/stats", 30))
            except Exception as e: s = {"error": str(e)[:200]}
            fs.write(json.dumps({"tag": a.tag, "k": k, "t": time.time(), "s": s}) + "\n"); fs.flush()
        snap(0)
        for j in range(a.n):
            q = prompt(mb, j, a.ctx, a.k0)
            try:
                r = mb.stream([{"role": "user", "content": q}], 1, effort=a.effort, extra={"temperature": a.temperature})
                te = time.time()
                d = dict(tag=a.tag, j=j, t0=r["t0"], ttft=r["ttft"], t_resp=te - r["t0"], t_end=te, prompt=r["prompt"],
                         cached=r["cached"], n=r["n"], err=None)
            except Exception as e:
                te = time.time()
                d = dict(tag=a.tag, j=j, t0=None, ttft=None, t_resp=None, t_end=te, prompt=None, cached=None, n=0, err=str(e)[:200])
            recs.append(d)
            fo.write(json.dumps(d) + "\n"); fo.flush()
            time.sleep(a.pause)
            snap(j + 1)
    ok = [r for r in recs if not r["err"]]
    tr = [r["t_resp"] for r in ok if r["t_resp"]]
    print("NCONE " + json.dumps(dict(tag=a.tag, n=len(recs), errors=len(recs) - len(ok),
                                     prompt_min=min((r["prompt"] or 0) for r in ok) if ok else None,
                                     prompt_max=max((r["prompt"] or 0) for r in ok) if ok else None,
                                     cached_max=max((r["cached"] or 0) for r in ok) if ok else None,
                                     t_resp_p50=round(statistics.median(tr), 3) if tr else None)), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
