#!/usr/bin/env python3
"""k1cold.py (10.11.2026, queue-ar dsmain, ds41-k1 agent; dsm-attnfold2.sh's identity phase) -- the identity probes run COLD:
every request carries "cache_prompt": false (the server's prefix cache is neither read nor relied on; ds41_tests.py prefill's
fresh-prefill switch), so two arms of one process each prefill the whole prompt from position 0 and their answers compare
fold against fold, never a prefix-cache resume against a cold run. The prompts are the same bytes as the reference runs'.

  k1cold.py copy  [--prefix-chars --chars --max-tokens --runs --offset --tag --outdir]
                                                    ds41_tests.py's cmd_copy with cache_prompt false on every chat request
  k1cold.py deep  OUT [DEPTH] [TOKENS]              deepq.py's fixed ~DEPTH-token prompt + TOKENS greedy tokens (same bytes,
                                                    same OUT / OUT.json / "DEEPQ {json}" line), cache_prompt false
  k1cold.py gs    URL OUT [--conc N] [--max-tokens M] --gs PATH/greedyset.py
                                                    greedyset.py's 20 prompts, cache_prompt false; then one line
                                                    "K1COLD gs {answered, cached_total, cached_max}" from each answer's usage
Every mode reports the server's cached-token count (cached > 0 = the bypass did not hold: read that probe as a resume).
Standard library only; ds41_tests.py is ~/gpu/ds41/ds41_tests.py, as deepq.py and stepprobe.py load it."""
import importlib.util, json, os, sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tests():
    t = load("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
    orig = t.chat
    def chat(body, *a, **k):
        b = dict(body); b["cache_prompt"] = False
        return orig(b, *a, **k)
    t.chat = chat
    return t


def deep(a):
    t = tests()
    out = a[0]
    depth = int(a[1]) if len(a) > 1 else 135000
    ntok = int(a[2]) if len(a) > 2 else 512
    CPT, OFF = 3.95, 1200   # deepq.py's: the same bytes
    t.MODEL = t.model_id()
    body = t.text_chars(int(depth * CPT), offset=OFF)
    task = ("\n\nYou have read the source files above. Write a long, detailed technical review of this code: explain what "
            "the main modules do, how they depend on each other, which functions are the most complex and why, and "
            "propose concrete refactorings with example code.")
    msgs = [{"role": "user", "content": "# deepq fixed prompt v1\n" + body + task}]
    r = t.chat({"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True},
               timeout=7200)
    u = r.get("usage") or {}
    text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
    open(out, "w").write(text)
    info = dict(status=r.get("status"), prompt_tokens=u.get("prompt_tokens"), cached=t.cached(u), ttft=round(r.get("ttft") or 0, 2),
                tokens=u.get("completion_tokens"), finish=r.get("finish"), chars=len(text), err=r.get("raw_err"), cold=True)
    json.dump(info, open(out + ".json", "w"))
    print("DEEPQ " + json.dumps(info), flush=True)
    return 0 if r.get("status") == 200 else 1


def gs(a):
    path = a[a.index("--gs") + 1]
    g = load("greedyset", path)
    seen = []
    orig = g.post
    def post(url, p, body, timeout):
        b = dict(body); b["cache_prompt"] = False
        d = orig(url, p, b, timeout)
        u = d.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        seen.append(det.get("cached_tokens", u.get("cached_tokens")))
        return d
    g.post = post
    sys.argv = [path] + [x for i, x in enumerate(a) if x != "--gs" and (i == 0 or a[i - 1] != "--gs")]
    av = sys.argv[1:]
    def opt(k, d):
        return type(d)(av[av.index(k) + 1]) if k in av else d
    pos = [x for j, x in enumerate(av) if not x.startswith("--") and (j == 0 or not av[j - 1].startswith("--"))]
    rc = g.run(pos[0].rstrip("/"), pos[1], opt("--conc", 1), opt("--max-tokens", 192), opt("--timeout", 300))
    ints = [x for x in seen if isinstance(x, int)]
    print("K1COLD gs " + json.dumps({"requests": len(seen), "cached_total": sum(ints) if ints else None,
                                     "cached_max": max(ints) if ints else None, "cached_unknown": len(seen) - len(ints)}))
    return rc


def copy(a):
    """ds41_tests.py's `copy` (its argparse defaults; it has no main()), every chat request cache_prompt false."""
    import argparse
    t = tests()
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=15000)
    ap.add_argument("--offset", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=7000)
    ap.add_argument("--outdir", default=os.path.expanduser("~/gpu/ds41/runs"))
    ap.add_argument("--prefix-chars", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default="")
    ns = ap.parse_args(a)
    t.MODEL = t.model_id()
    r = t.cmd_copy(ns)
    return r if isinstance(r, int) else 0


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("copy", "deep", "gs"):
        print(__doc__); return 2
    return {"copy": copy, "deep": deep, "gs": gs}[a[0]](a[1:])


if __name__ == "__main__":
    sys.exit(main())
