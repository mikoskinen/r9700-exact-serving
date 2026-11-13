#!/usr/bin/env python3
"""resume_ho.py (10.11.2026, queue-ar/dsmain, ds41 resume agent) -- the driver of 57-p4zyzzzzz-ds-ho-resume (dsm-resume.sh):
does a prefix-cache resume answer as the cold prefill of the same prompt does, byte for byte, on the DS41 server (:8090,
DS41_URL)? Prompts are deepq.py's (the same bytes for KIND deep); every request is greedy (temperature 0, thinking off,
ignore_eos, TOKENS tokens), streamed as deepq.py streams it.

  resume_ho.py run OUT KIND [--depth N] [--swap-chars C] [--nonce S] [--tokens 512] [--top 2]
      KIND deep   deepq.py's fixed prompt v1 (stdlib filler at offset 1200, 3.95 chars a token, depth 135000 = 136216
                  tokens on this .rad); its cold answer is runs/xdsm/deep-dsm-225.txt
      KIND swap   deep with the filler's last C chars replaced by other filler (offset 9000, same length): the prompt departs
                  from deep about C / 3.9 tokens before its end, so a hit on deep's chain (or deep's on this one's) resumes at
                  the interval checkpoint below the departure (129024 for C 15500), not at the tail snapshot
      KIND nonce  "# nonce S" + deepq's prompt at --depth N tokens: a chain no earlier request shares
    + logprobs (top_logprobs --top; --top -1 = none; a server without sample_logprobs answers 400 -> the request is sent
    again without them, noted as lp "refused"). Writes OUT (the text, deepq's form), OUT.json (status, prompt_tokens, cached,
    ttft, tokens, chars, the /sessions tier counters before and after), OUT.lp.json (per token: id, logprob, top [id, lp]).
    Prints "RHO {json}".
  resume_ho.py reset          POST /reset_prefix_cache (a loopback operator) -> "RHO {json}"
  resume_ho.py cmp A B        A vs B: the text (IDENTICAL / DIFFERENT at char k), the token ids (first differing token), and
                              with both .lp.json: |logprob A - B| at token 0, the first token whose logprob differs at all,
                              the largest difference before the first differing token, and each run's top-2 margin there
                              -> "CMP {json}"
  resume_ho.py groups F...    the files grouped by identical bytes (md5 8) -> "GROUPS {json}"
"""
import hashlib, importlib.util, json, os, sys, time, urllib.error, urllib.request

spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
URL = t.URL
CPT, OFF = 3.95, 1200            # deepq.py's
SWAP_OFF = 9000
TASK = ("\n\nYou have read the source files above. Write a long, detailed technical review of this code: explain what "
        "the main modules do, how they depend on each other, which functions are the most complex and why, and "
        "propose concrete refactorings with example code.")


def arg(name, default, conv=str):
    a = sys.argv
    return conv(a[a.index(name) + 1]) if name in a and a.index(name) + 1 < len(a) else default


def get_json(path, method="GET", timeout=60):
    req = urllib.request.Request(URL + path, data=b"" if method == "POST" else None, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"err": e.read().decode(errors="replace")[:300]}
    except Exception as e:
        return None, {"err": str(e)[:300]}


def tiers():
    """The /sessions counters a prefix hit can move: tier promotions (a restore from host memory), restores and their
    stall, demotions, linear snapshots written and tokens they let requests skip."""
    st, d = get_json("/sessions")
    if st != 200:
        return {"err": st}
    mv, sl, li = d.get("moved") or {}, d.get("stall") or {}, d.get("linear") or {}
    return {"promoted": mv.get("promoted"), "to_host": mv.get("to_host"), "dropped": mv.get("dropped"),
            "restores": sl.get("restores"), "restore_ms": round(sl.get("restore_ms") or 0, 1),
            "ckpt_written": li.get("checkpoints_written"), "linear_cached": li.get("cached_tokens"),
            "sessions": len(d.get("sessions") or [])}


def prompt(kind):
    depth = arg("--depth", 135000, int)
    body = t.text_chars(int(depth * CPT), offset=OFF)
    head = "# deepq fixed prompt v1\n"
    if kind == "deep":
        if depth != 135000:
            sys.exit("deep is deepq.py's prompt at depth 135000; use nonce for other depths")
    elif kind == "swap":
        c = arg("--swap-chars", 15500, int)
        other = t.text_chars(c, offset=SWAP_OFF)
        if len(other) != c or other == body[-c:]:
            sys.exit("swap: no distinct filler of %d chars" % c)
        body = body[:-c] + other
    elif kind == "nonce":
        head = "# nonce %s\n" % arg("--nonce", "x", str) + head
    else:
        sys.exit("unknown KIND " + kind)
    return [{"role": "user", "content": head + body + TASK}]


def chat_lp(payload, timeout=7200):
    """deepq's request (ds41_tests.chat, streamed) keeping each token's logprob entry."""
    p = dict(payload); p["model"] = t.MODEL; p["stream"] = True; p["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; content = []; reasoning = []; usage = None; finish = None; lps = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for c in ev.get("choices", []):
                    d = c.get("delta", {})
                    if d.get("content"):
                        content.append(d["content"]); tf = tf or time.time()
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if rc:
                        reasoning.append(rc); tf = tf or time.time()
                    for e in ((c.get("logprobs") or {}).get("content") or []):
                        lps.append({"id": e.get("id"), "lp": e.get("logprob"),
                                    "top": [[a.get("id"), a.get("logprob")] for a in (e.get("top_logprobs") or [])]})
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
    except urllib.error.HTTPError as e:
        return dict(status=e.code, raw_err=e.read().decode(errors="replace")[:600])
    return dict(status=200, ttft=(tf - t0) if tf else None, content="".join(content), reasoning="".join(reasoning),
                usage=usage, finish=finish, lps=lps)


def cmd_run():
    out, kind = sys.argv[2], sys.argv[3]
    ntok, top = arg("--tokens", 512, int), arg("--top", 2, int)
    t.MODEL = t.model_id()
    pay = {"messages": prompt(kind), "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True}
    lp_state = "off"
    if top >= 0:
        pay.update(logprobs=True, top_logprobs=top); lp_state = "on"
    before = tiers()
    r = chat_lp(pay)
    if r.get("status") == 400 and top >= 0 and "logprob" in (r.get("raw_err") or ""):
        pay.pop("logprobs", None); pay.pop("top_logprobs", None); lp_state = "refused"
        r = chat_lp(pay)
    after = tiers()
    u = r.get("usage") or {}
    text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
    open(out, "w").write(text)
    lps = r.get("lps") or []
    json.dump(lps, open(out + ".lp.json", "w"))
    info = dict(kind=kind, status=r.get("status"), prompt_tokens=u.get("prompt_tokens"), cached=t.cached(u),
                ttft=round(r.get("ttft") or 0, 2), tokens=u.get("completion_tokens"), finish=r.get("finish"),
                chars=len(text), lp=lp_state, lp_tokens=len(lps), md5=hashlib.md5(text.encode()).hexdigest()[:8],
                tiers_before=before, tiers_after=after, err=r.get("raw_err"))
    json.dump(info, open(out + ".json", "w"))
    print("RHO " + json.dumps(info), flush=True)
    return 0 if r.get("status") == 200 else 1


def cmd_reset():
    st, d = get_json("/reset_prefix_cache", method="POST")
    print("RHO " + json.dumps({"reset_status": st, **(d if isinstance(d, dict) else {"body": d}), "tiers": tiers()}),
          flush=True)
    return 0 if st == 200 else 1


def rd(f):
    try:
        return open(f).read()
    except OSError:
        return None


def rdlp(f):
    try:
        v = json.load(open(f + ".lp.json"))
        return v if v else None
    except (OSError, ValueError):
        return None


def margin(e):
    tp = e.get("top") or []
    return round(tp[0][1] - tp[1][1], 6) if len(tp) >= 2 and tp[0][1] is not None and tp[1][1] is not None else None


def cmd_cmp():
    a, b = sys.argv[2], sys.argv[3]
    ta, tb = rd(a), rd(b)
    o = {"a": os.path.basename(a), "b": os.path.basename(b)}
    if ta is None or tb is None:
        o["text"] = "not compared (missing %s)" % (a if ta is None else b)
        print("CMP " + json.dumps(o)); return 0
    f = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None if len(ta) == len(tb) else min(len(ta), len(tb)))
    o["text"] = "IDENTICAL (%d chars)" % len(ta) if ta == tb else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(ta), len(tb))
    o["identical"] = ta == tb
    la, lb = rdlp(a), rdlp(b)
    if la and lb:
        n = min(len(la), len(lb))
        k = next((i for i in range(n) if la[i]["id"] != lb[i]["id"]), None)
        o["first_token_diff"] = k
        d0 = [abs(la[i]["lp"] - lb[i]["lp"]) for i in range(k if k is not None else n)
              if la[i]["lp"] is not None and lb[i]["lp"] is not None]
        o["lp_diff_tok0"] = d0[0] if d0 else None
        o["first_lp_diff_at"] = next((i for i, d in enumerate(d0) if d != 0), None)
        o["lp_diff_max_before"] = max(d0) if d0 else None
        o["lp_equal_tokens_before"] = sum(1 for d in d0 if d == 0)
        if k is not None:
            o["margin_at_diff"] = {"a": margin(la[k]), "b": margin(lb[k])}
    else:
        o["lp"] = "not compared (no logprobs in %s)" % ("both" if not la and not lb else (a if not la else b))
    print("CMP " + json.dumps(o), flush=True)
    return 0


def cmd_groups():
    g = {}
    for f in sys.argv[2:]:
        s = rd(f)
        key = hashlib.md5(s.encode()).hexdigest()[:8] if s is not None else "missing"
        g.setdefault(key, []).append(os.path.basename(f))
    print("GROUPS " + json.dumps(g), flush=True)
    return 0


if __name__ == "__main__":
    cmds = {"run": cmd_run, "reset": cmd_reset, "cmp": cmd_cmp, "groups": cmd_groups}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
