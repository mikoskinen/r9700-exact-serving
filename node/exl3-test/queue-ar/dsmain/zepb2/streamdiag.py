#!/usr/bin/env python3
"""streamdiag.py (12.11.2026, queue-ar/dsmain/zepb2, the EPB builder) -- the zepb incident (c8-5-S-S, 18:47:27-18:49:46): two
first-generation requests of the fixed c8 harness (harness/mbt3.py prompt_of(i, 0), seed 1, ctx 1000: worker 0 at max_tokens 192
and worker 4 at 128) finished on the server (gen 192 / 128, all reasoning) but the client saw NO text delta (t_first null), so the
fixed schedule waited --max-wait 120 s for group 5's first token and the time window collapsed to 0 s. This sends EXACTLY those
requests (mb.stream's body: chat, reasoning_effort medium, ignore_eos, the temperature, stream + include_usage) under the arm set
now, streamed and again non-streamed, and reports what came back: SSE events, events whose delta carries non-empty content /
reasoning_content, the first such time, the streamed text's head; the non-streamed message's content / reasoning_content heads,
finish_reason and usage. mb.py / mbt3.py are loaded READ ONLY (importlib); nothing of theirs changes.

  streamdiag.py --mb ~/gpu/ds41/mb.py --mbt3 ../harness/mbt3.py --temperature 0 [--pairs 0:192,4:128] [--label S] [--out F.jsonl]
  streamdiag.py --mb ~/gpu/ds41/mb.py --mbt3 ../harness/mbt3.py --selftest
Last line "STREAMDIAG {json}"; exit 0 = every request answered (the verdict is in the line: EMPTY_STREAM when a request streamed no
text). Standard library only."""
import argparse, importlib.util, json, sys, time, types, urllib.request

sys.dont_write_bytecode = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def body(mb, q, mt, temp, stream):
    p = {"model": mb.MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": mt, "stream": stream,
         "reasoning_effort": "medium", "ignore_eos": True, "temperature": temp}
    if stream: p["stream_options"] = {"include_usage": True}
    return p


def post(url, p, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(p).encode(),
                                                         {"Content-Type": "application/json"}), timeout=timeout)


def streamed(url, p):
    t0 = time.time(); ev = nonempty = 0; first = None; text = []; fin = None; usage = None; keys = set()
    with post(url, p) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev += 1
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {}) or {}
                keys.update(k for k, v in dl.items() if v not in (None, ""))
                s = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if s:
                    nonempty += 1; first = first or time.time(); text.append(s)
                if c.get("finish_reason"): fin = c["finish_reason"]
    t = "".join(text)
    return dict(events=ev, nonempty=nonempty, first_s=round(first - t0, 3) if first else None, keys=sorted(keys), finish=fin,
                n=(usage or {}).get("completion_tokens"), text_head=t[:240], text_chars=len(t), wall_s=round(time.time() - t0, 2))


def plain(url, p):
    t0 = time.time()
    with post(url, p) as r:
        d = json.load(r)
    c = (d.get("choices") or [{}])[0]; m = c.get("message") or {}
    rc = m.get("reasoning_content") or m.get("reasoning") or ""; ct = m.get("content") or ""
    return dict(finish=c.get("finish_reason"), n=(d.get("usage") or {}).get("completion_tokens"), reasoning_chars=len(rc), content_chars=len(ct),
                reasoning_head=rc[:240], content_head=ct[:240], tool_calls=len(m.get("tool_calls") or []), wall_s=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True); ap.add_argument("--mbt3", required=True)
    ap.add_argument("--url", default=None); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--pairs", default="0:192,4:128"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--label", default=""); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb, "mb_ro"); mbt = load(a.mbt3, "mbt3_ro")
    if a.url: mb.URL = a.url
    ns = types.SimpleNamespace(seed=a.seed, ctx=a.ctx, prompt_set="fixed")
    pairs = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.pairs.split(",") if x]
    if a.selftest:
        ok = all(hasattr(mb, x) for x in ("URL", "text", "TASKS")) and hasattr(mbt, "prompt_of")
        mb.MODEL = "selftest"
        q = mbt.prompt_of(mb, ns, 0, 0) if ok else ""
        ok = ok and q.startswith("# req fixed-s1-k0 0\n") and len(q) > 3000 and body(mb, q, 192, 0.0, True)["ignore_eos"] is True
        print("SELFTEST %s: prompt_of(0, 0) %d chars, head %r" % ("OK" if ok else "FAIL", len(q), q[:24])); return 0 if ok else 1
    with urllib.request.urlopen(mb.URL + "/v1/models", timeout=60) as r:   # as ncmb3.py sets it
        mb.MODEL = json.load(r)["data"][0]["id"]
    rows, bad = [], 0
    for i, mt in pairs:
        q = mbt.prompt_of(mb, ns, i, 0)
        try:
            s = streamed(mb.URL, body(mb, q, mt, a.temperature, True))
            n = plain(mb.URL, body(mb, q, mt, a.temperature, False))
        except Exception as e:
            bad += 1; rows.append(dict(i=i, max_tokens=mt, err=str(e)[:200])); continue
        rows.append(dict(i=i, max_tokens=mt, prompt_chars=len(q), stream=s, plain=n,
                         verdict="EMPTY_STREAM" if s["nonempty"] == 0 else "ok"))
    if a.out:
        with open(a.out, "a") as f:
            for r in rows: f.write(json.dumps(dict(label=a.label, **r), ensure_ascii=False) + "\n")
    for r in rows:
        if "err" in r: print("DIAG %s w%d max %d: error %s" % (a.label, r["i"], r["max_tokens"], r["err"])); continue
        s, n = r["stream"], r["plain"]
        print("DIAG %s w%d max %d: %s | stream %d events, %d with text (first %s s, keys %s), finish %s, n %s, %d chars %r | non-stream finish %s, n %s, reasoning %d chars %r, content %d chars %r, tool_calls %d" % (
            a.label, r["i"], r["max_tokens"], r["verdict"], s["events"], s["nonempty"], s["first_s"], ",".join(s["keys"]), s["finish"], s["n"], s["text_chars"],
            s["text_head"][:100], n["finish"], n["n"], n["reasoning_chars"], n["reasoning_head"][:100], n["content_chars"], n["content_head"][:80], n["tool_calls"]))
    print("STREAMDIAG " + json.dumps(dict(label=a.label, requests=len(rows), errors=bad,
                                          empty_stream=sum(1 for r in rows if r.get("verdict") == "EMPTY_STREAM"))))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
