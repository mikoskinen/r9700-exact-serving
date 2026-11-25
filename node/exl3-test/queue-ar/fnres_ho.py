#!/usr/bin/env python3
"""fnres_ho.py (10.11.2026, queue-ar, fn4 169) -- H3's precondition on Flash-Next: a SHORT request that shares the long
prompt's head over whole prefix-cache blocks, so it publishes those blocks first (computed in its own, smaller prefill
chunk), then the long prompt cold (it cannot resume there: the short request's only snapshot is its own tail, past the
divergence) and its hit (KV blocks of the head from the short publisher, the linear snapshot from the long prompt).
Uses the DS agent's resume_ho.py (read-only) for the prompt, the streamed greedy request with logprobs and the output
files, so `resume_ho.py cmp` compares them.

  fnres_ho.py run OUT long  --depth N --nonce S [--tokens 256] [--top 2]
      resume_ho's nonce prompt (the chat template's user turn; thinking off)
  fnres_ho.py run OUT short --depth N --nonce S --head-chars H [--tokens 32] [--top 2]
      the same prompt's first H characters of its user content, then a different short question -- the shared head
      (H ~ 5400 chars ~ 1400 tokens: several whole blocks, one prefill chunk under 2048 rows)
Prints "FHO {json}" (as resume_ho's RHO: prompt_tokens, cached, ttft, tokens, lp, md5, tiers before / after).
"""
import hashlib, importlib.util, json, os, sys

RHO = os.path.expanduser("~/gpu/ds41/exl3-test/queue-ar/dsmain/resume_ho.py")
spec = importlib.util.spec_from_file_location("rho", RHO)
rho = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rho)
t = rho.t
QUESTION = "\n\nIgnore any task above. In one short sentence: what kind of text is this?"


def main():
    if len(sys.argv) < 4 or sys.argv[1] != "run" or sys.argv[3] not in ("long", "short"):
        sys.exit(__doc__)
    out, kind = sys.argv[2], sys.argv[3]
    msgs = rho.prompt("nonce")              # reads --depth / --nonce from argv
    if kind == "short":
        h = rho.arg("--head-chars", 5400, int)
        c = msgs[0]["content"]
        if h >= len(c):
            sys.exit("--head-chars past the prompt")
        msgs = [{"role": "user", "content": c[:h] + QUESTION}]
    ntok = rho.arg("--tokens", 32 if kind == "short" else 256, int)
    top = rho.arg("--top", 2, int)
    t.MODEL = t.model_id()
    pay = {"messages": msgs, "max_tokens": ntok, "temperature": 0.0, "reasoning_effort": "none", "ignore_eos": True}
    lp_state = "off"
    if top >= 0:
        pay.update(logprobs=True, top_logprobs=top)
        lp_state = "on"
    before = rho.tiers()
    r = rho.chat_lp(pay)
    if r.get("status") == 400 and top >= 0 and "logprob" in (r.get("raw_err") or ""):
        pay.pop("logprobs", None); pay.pop("top_logprobs", None); lp_state = "refused"
        r = rho.chat_lp(pay)
    after = rho.tiers()
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
    print("FHO " + json.dumps(info))


if __name__ == "__main__":
    main()
