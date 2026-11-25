#!/usr/bin/env python3
"""resprobe.py (12.11.2026, queue-ar/dsmain/res12, ds-main-port residency screen) -- ~/gpu/ds41/stepprobe.py (3.11.) + the answer.

  resprobe.py PROMPTS GEN [TEXTOUT]   per prompt (comma list of ds41_tests.PROMPTS names): ONE chat request at temperature 0,
                                      reasoning_effort none, max_tokens GEN, streamed; prints stepprobe's J line (the same keys,
                                      the same arithmetic) + status, finish, text_chars, text_md5; with TEXTOUT the answer(s) go
                                      to that file as JSON {"reads": [{"prompt", "status", "finish", "tokens", "content"}]} (the
                                      greedy-identity guard compares them across arms; placement cannot change a T 0 answer).
  resprobe.py selftest                no server: the J line's arithmetic on a fake request / metric pair (= stepprobe's).
A request error prints a J line with status != 200 and no speed keys (stepprobe raised a KeyError there).
"""
import hashlib, json, os, re, sys, urllib.request, importlib.util

KEYS = ["radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total", "radiance:draft_tokens_total",
        "radiance:draft_accepted_total", "radiance:decode_tokens_total", "radiance:engine_steps_total", "radiance:engine_step_seconds_total"]


def load_tests():
    spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    return t


def metrics(url):
    txt = urllib.request.urlopen(url + "/metrics").read().decode()
    out = {}
    for k in KEYS:
        mm = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)
        out[k] = float(mm.group(2)) if mm else None
    return out


def jline(name, r, b, a):
    """stepprobe's J dict (same keys / rounding) + status / finish / text_chars / text_md5."""
    content = r.get("content") or ""
    extra = dict(status=r.get("status"), finish=r.get("finish"), text_chars=len(content),
                 text_md5=hashlib.md5(content.encode()).hexdigest()[:12])
    if r.get("status") != 200 or not r.get("t_first") or not r.get("t_last"):
        return dict(prompt=name[:20], tokens=0, error=(r.get("raw_err") or "no tokens")[:200], **extra)
    d = {k.split(":")[1]: (a[k] - b[k]) if a.get(k) is not None and b.get(k) is not None else None for k in KEYS}
    n = (r.get("usage") or {}).get("completion_tokens") or 0
    dt = r["t_last"] - r["t_first"]
    steps = d["engine_steps_decode_total"] or 1
    return dict(prompt=name[:20], tokens=n, decode_tps=round((n - 1) / dt, 2) if dt > 0 else None, decode_steps=steps,
                ms_per_decode_step=round(1000 * (d["engine_step_seconds_decode_total"] or 0) / steps, 2),
                tokens_per_step=round(n / steps, 2), acceptance=round((d["draft_accepted_total"] or 0) / max(1, d["draft_tokens_total"] or 0), 3),
                all_steps=d["engine_steps_total"], all_step_s=round(d["engine_step_seconds_total"] or 0, 2), **extra)


def selftest():
    b = {k: 0.0 for k in KEYS}
    a = {"radiance:engine_steps_decode_total": 324.0, "radiance:engine_step_seconds_decode_total": 6.2046, "radiance:draft_tokens_total": 400.0,
         "radiance:draft_accepted_total": 279.2, "radiance:decode_tokens_total": 512.0, "radiance:engine_steps_total": 325.0,
         "radiance:engine_step_seconds_total": 6.34}
    r = dict(status=200, t_first=100.0, t_last=106.206, content="Abc", finish="length", usage={"completion_tokens": 512})
    j = jline("essay", r, b, a)
    assert j["tokens"] == 512 and j["decode_steps"] == 324.0 and j["ms_per_decode_step"] == 19.15 and j["tokens_per_step"] == 1.58, j
    assert j["decode_tps"] == round(511 / 6.206, 2) and j["acceptance"] == 0.698 and j["all_step_s"] == 6.34 and j["status"] == 200, j
    assert j["text_chars"] == 3 and j["text_md5"] == hashlib.md5(b"Abc").hexdigest()[:12], j
    e = jline("code", dict(status=500, raw_err="boom"), b, a)
    assert e["status"] == 500 and e["tokens"] == 0 and "decode_tps" not in e and e["error"] == "boom", e
    s = json.dumps(j)
    assert s.startswith('{"prompt": "essay", "tokens": 512, "decode_tps": '), s
    print("SELFTEST OK resprobe.py (stepprobe J arithmetic, error J, text md5)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    t = load_tests()
    t.MODEL = t.model_id()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["essay", "code"]
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    tout = sys.argv[3] if len(sys.argv) > 3 else None
    reads = []
    for name in names:
        q = t.PROMPTS.get(name, name)
        b = metrics(t.URL)
        r = t.chat({"messages": [{"role": "user", "content": q}], "max_tokens": gen, "temperature": 0.0, "reasoning_effort": "none"})
        a = metrics(t.URL)
        j = jline(name, r, b, a)
        reads.append(dict(prompt=name[:20], status=r.get("status"), finish=r.get("finish"), tokens=j.get("tokens"), content=r.get("content") or ""))
        print("J " + json.dumps(j), flush=True)
    if tout:
        with open(tout, "w") as f:
            json.dump({"reads": reads}, f)


if __name__ == "__main__":
    main()
