#!/usr/bin/env python3
"""editprobe.py [CHARS] [MAX_TOKENS] [OFFSET] (10.11.2026, queue-ar dsmain, ds41-gg agent) -- a repetition-heavy 1-user
probe beside stepprobe.py: an EDIT task whose reply re-emits a given file region, the case prompt lookup drafts for.

The region is ~CHARS characters (default 9000, ~2.5k tokens) of ds41_tests.py's text_chars filler (the Python standard
library's own sources, files back to back; never transcripts) from file OFFSET (default 7). The request (greedy,
temperature 0, thinking off, max_tokens MAX_TOKENS, default 1500) asks for the same file back with one small change: a
one-line docstring added to one named function or method (the first defined in the region's middle half), everything else
character for character, no fences, no commentary. The reply is longer than max_tokens allows, so it is cut; that is
the point (decode while copying).

Metrics from /metrics deltas around the request, as stepprobe.py: decode tok/s, ms a decode step, tokens a step,
draft acceptance. Echo check: difflib ratio of the reply (fences stripped) against the source prefix of the same
length; under 0.8 (a refusal, a summary, a rewrite) the line says ECHO NOT and the numbers are not a copy's.
Prints one "J {json}" line. Standard library only; reads ds41_tests.py from ~/gpu/ds41."""
import difflib, importlib.util, json, os, re, sys, urllib.request

spec = importlib.util.spec_from_file_location("t", os.path.expanduser("~/gpu/ds41/ds41_tests.py"))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
t.MODEL = t.model_id()
KEYS = ["radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total", "radiance:draft_tokens_total",
        "radiance:draft_accepted_total", "radiance:decode_tokens_total", "radiance:engine_steps_total",
        "radiance:engine_step_seconds_total"]


def m():
    txt = urllib.request.urlopen(t.URL + "/metrics").read().decode()
    out = {}
    for k in KEYS:
        mm = re.search("^" + re.escape(k) + r"(\{[^}]*\})? ([0-9.eE+-]+)$", txt, re.M)
        out[k] = float(mm.group(2)) if mm else None
    return out


def main():
    chars = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    gen = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    src = t.text_chars(chars, offset=off)
    # cut at a line end, so the region is whole lines
    src = src[:src.rfind("\n") + 1] if "\n" in src else src
    defs = [(mm.start(), mm.group(1)) for mm in re.finditer(r"^[ \t]*def ([A-Za-z_]\w*)\(", src, re.M)]
    mid = [d for d in defs if len(src) // 4 <= d[0] <= 3 * len(src) // 4]
    pick = (mid or defs or [(0, None)])[0][1]
    if pick:
        task = ("Add a one-line docstring to the function `%s` (its first definition below). Output the whole "
                "file with that one change and everything else exactly as it is, character for character, including "
                "comments and blank lines. Output only the file: no code fences, no explanation." % pick)
    else:
        task = ("Add a one-line comment `# reviewed` as the first line. Output the whole file with that one change and "
                "everything else exactly as it is, character for character. Output only the file: no code fences, no "
                "explanation.")
    q = task + "\n\n" + src
    b = m()
    r = t.chat({"messages": [{"role": "user", "content": q}], "max_tokens": gen, "temperature": 0.0,
                "reasoning_effort": "none"})
    a = m()
    d = {k.split(":")[1]: (a[k] - b[k]) if (a[k] is not None and b[k] is not None) else None for k in KEYS}
    n = (r.get("usage") or {}).get("completion_tokens") or 0
    dt = (r.get("t_last") or 0) - (r.get("t_first") or 0)
    steps = d["engine_steps_decode_total"] or 1
    out = r.get("content") or ""
    body = re.sub(r"^\s*```[a-zA-Z]*\n", "", out)
    body = re.sub(r"\n```\s*$", "\n", body)
    ref = src[:max(1, len(body))]
    ratio = difflib.SequenceMatcher(None, ref, body, autojunk=False).ratio() if body else 0.0
    res = dict(prompt="edit", function=pick, status=r.get("status"), finish=r.get("finish"),
               prompt_tokens=(r.get("usage") or {}).get("prompt_tokens"), tokens=n,
               decode_tps=round((n - 1) / dt, 2) if dt > 0 and n > 1 else None, decode_steps=steps,
               ms_per_decode_step=round(1000 * (d["engine_step_seconds_decode_total"] or 0) / steps, 2),
               tokens_per_step=round(n / steps, 2),
               acceptance=round((d["draft_accepted_total"] or 0) / max(1, d["draft_tokens_total"] or 0), 3),
               drafted=d["draft_tokens_total"], accepted=d["draft_accepted_total"],
               echo_ratio=round(ratio, 4), echo_ok=ratio >= 0.8, out_chars=len(body), src_chars=len(src))
    print("J " + json.dumps(res), flush=True)
    return 0 if r.get("status") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
