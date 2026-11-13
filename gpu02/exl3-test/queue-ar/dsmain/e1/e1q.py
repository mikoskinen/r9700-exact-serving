#!/usr/bin/env python3
"""e1q.py (12.11.2026, item 57-p4zyzzzzz-ds-r00zzz-cz-reng, ds-main-port E1 builder) -- one long greedy request for lever E1's
cold TTFT and identity readings, sent through ds41_tests.py's chat() (loaded READ ONLY via importlib, as pskld.py does):

  e1q.py run --tests T --url URL --src code --file code-body.txt --depth D --ntok N --out J --text TXT
      THE REALISTIC CODE PROMPT: "# e1 code prompt v1\\n" + the first int(D x CODE_CPT) characters of code-body.txt + TASK.
      code-body.txt = our own repo sources (radiance-int7 0391a6e: core/*.cpp, arch/deepseek41/deepseek41.cpp, libr4d/*.hip,
      round-robin over the three sorted lists, a file's first 48000 characters under a '// ==== path ====' header; built
      on the workstation by mkcode.py, no transcripts): 446500 characters = 135388 tokens by the model's tokenizer
      (rad-tokcheck), so CODE_CPT = 3.2979 characters a token. Depths share their prefix.
  e1q.py run --tests T --url URL --src deep --pskld P --doff O --depth D --ntok N --out J --text TXT
      deepq.py's request on another TEXT: pskld.py's deep_payload with its DEEP_OFF set to O, the ds41_tests.SRC file the body
      starts at (p2bq.py's construction; deepq.py's 1200 wraps to file 0). Two offsets share no prefix, so a prefix hit on one
      text can only resume at snapshots its own runs wrote (/reset_prefix_cache keeps the checkpoint records).
  Both: max_tokens N, temperature 0, reasoning off, ignore_eos (deepq's settings). Prints "E1Q {json}" (status,
  prompt_tokens, cached, ttft, tokens, finish, chars, seconds, err), writes it to J and the answer (content, then
  "\\n[reasoning]\\n" + reasoning if any) to TXT.
  e1q.py selftest --tests T --pskld P --file F [--md5 M] [--doff O1 --doff2 O2]
      offline: the code body loads (its md5 = M when given), depth 16000's body is a prefix of 64000's of 135000's; deep
      offsets O1 / O2 (100 / 300) are inside ds41_tests.SRC (a larger one wraps to file 0, deepq's text) and give different
      first blocks, different from deepq's (1200); the payloads carry the settings above.
Standard library only."""
import argparse, hashlib, importlib.util, json, sys, time

sys.dont_write_bytecode = True
CODE_CPT = 3.2979
CODE_HEAD = "# e1 code prompt v1\n"
TASK = ("\n\nYou have read the source files above. Write a long, detailed technical review of this code: explain what "
        "the main modules do, how they depend on each other, which functions are the most complex and why, and "
        "propose concrete refactorings with example code.")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def code_body(path, depth):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    return s[: int(depth * CODE_CPT)]


def code_payload(path, depth, ntok):
    msgs = [{"role": "user", "content": CODE_HEAD + code_body(path, depth) + TASK}]
    return {"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True}


def deep_payload(pl, t, depth, ntok, doff):
    pl.DEEP_OFF = doff
    return pl.deep_payload(t, depth, ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "selftest"))
    ap.add_argument("--tests", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--src", choices=("code", "deep"), default="code")
    ap.add_argument("--file")
    ap.add_argument("--md5")
    ap.add_argument("--pskld")
    ap.add_argument("--doff", type=int, default=100)
    ap.add_argument("--doff2", type=int, default=300)
    ap.add_argument("--depth", type=int, default=135000)
    ap.add_argument("--ntok", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--text")
    a = ap.parse_args()
    t = load(a.tests, "ds41_tests_ro")
    if a.mode == "selftest":
        ok, why = True, []
        try:
            raw = open(a.file, "rb").read()
            md5 = hashlib.md5(raw).hexdigest()
            if a.md5 and not md5.startswith(a.md5):
                ok = False; why.append("md5 %s != %s" % (md5[:8], a.md5))
            b16, b64, b135 = (code_body(a.file, d) for d in (16000, 64000, 135000))
            if not (b135.startswith(b64) and b64.startswith(b16) and len(b16) < len(b64) < len(b135)):
                ok = False; why.append("code depths are not prefixes")
            if len(raw) < int(135000 * CODE_CPT):
                ok = False; why.append("code body shorter than 135000 tokens' characters (%d)" % len(raw))
            p = code_payload(a.file, 16000, 256)
            if not (p["messages"][0]["content"].startswith(CODE_HEAD) and p["messages"][0]["content"].endswith(TASK) and
                    p["max_tokens"] == 256 and p["temperature"] == 0.0 and p.get("ignore_eos")):
                ok = False; why.append("code payload")
        except OSError as e:
            ok = False; why.append("code body: %s" % e); md5 = "-"
        if a.pskld:
            pl = load(a.pskld, "pskld_ro")
            d6, d8, dq = (deep_payload(pl, t, 2000, 256, o)["messages"][0]["content"] for o in (a.doff, a.doff2, 1200))
            h = "# deepq fixed prompt v1\n"
            if not (0 < a.doff < len(t.SRC) and 0 < a.doff2 < len(t.SRC)):
                ok = False; why.append("deep offsets %d / %d not inside SRC's %d files" % (a.doff, a.doff2, len(t.SRC)))
            if not (d6.startswith(h) and d8.startswith(h) and d6[:300] != d8[:300] and d6[:300] != dq[:300] and d8[:300] != dq[:300]):
                ok = False; why.append("deep offsets %d / %d / 1200 do not give three texts" % (a.doff, a.doff2))
            x4, x6 = (deep_payload(pl, t, d, 256, a.doff)["messages"][0]["content"] for d in (40000, 64000))
            if not x6.startswith(x4[: -len(pl.DEEP_TASK)]):
                ok = False; why.append("deep depths of one offset do not share their body")
        print(("SELFTEST OK (code body md5 %s, %d bytes; depths share their prefix; deep offsets %d / %d / 1200 differ, SRC %d files)" % (md5[:8], len(raw), a.doff, a.doff2, len(t.SRC)))
              if ok else "SELFTEST FAIL " + "; ".join(why))
        return 0 if ok else 1
    url = a.url.rstrip("/")
    t.URL = url
    t.MODEL = t.model_id()
    if a.src == "code":
        payload = code_payload(a.file, a.depth, a.ntok)
    else:
        payload = deep_payload(load(a.pskld, "pskld_ro"), t, a.depth, a.ntok, a.doff)
    t0 = time.time()
    err = None
    try:
        r = t.chat(dict(payload), timeout=7200)
    except Exception as e:
        r, err = {}, str(e)[:300]
    t1 = time.time()
    u = r.get("usage") or {}
    text = (r.get("content") or "") + ("\n[reasoning]\n" + r["reasoning"] if r.get("reasoning") else "")
    with open(a.text, "w") as f:
        f.write(text)
    info = dict(src=a.src, depth=a.depth, doff=a.doff if a.src == "deep" else None, status=r.get("status"),
                prompt_tokens=u.get("prompt_tokens"), cached=t.cached(u) if u else None, ttft=round(r.get("ttft") or 0, 3),
                tokens=u.get("completion_tokens"), finish=r.get("finish"), chars=len(text), seconds=round(t1 - t0, 2),
                err=err or r.get("raw_err"))
    with open(a.out, "w") as f:
        json.dump(info, f)
    print("E1Q " + json.dumps(info), flush=True)
    return 0 if r.get("status") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
