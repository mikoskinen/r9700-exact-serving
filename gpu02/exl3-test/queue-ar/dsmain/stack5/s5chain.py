#!/usr/bin/env python3
"""s5chain.py (12.11.2026, queue-ar/dsmain/stack5, ds-main-port stack5 builder) -- the TURN CHAIN of 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5:
a 3-turn conversation per arm, each later turn = the previous turn's messages + its answer + a follow-up, so turns 2 and 3 are
prefix hits on the turn before. The turn spec is pskcr/pcr2.py's (loaded READ ONLY via importlib, md5 in the item log; it is not
edited): turn 1 = pcr2.payload("deep:DEPTH:OFF") (deepq.py's request on text offset OFF, pskld.deep_payload), turn 2 =
pcr2.payload("turn:DEPTH:OFF:ANS1") (turn 1's message, its answer from ANS1 with the reasoning part dropped (pcr2.answer_of), then
pcr2.FOLLOW); turn 3 = turn 2's messages + its answer (ANS2, pcr2.answer_of) + FOLLOW2 below. This file adds only the third turn
and a NONCE line ("# s5 chain NONCE\\n") at the head of the first user message, so every arm's chain (its own OFF and nonce) shares
no snapshot-sized prefix with another arm's. The request, the TTFT / usage readings and the answer file are pskld.py's run() (READ
ONLY, as pcr2.py's run does it). Standard library only.

  s5chain.py run --pcr2 P --pskld PL --tests T --url URL --turn 1|2|3 --depth D --off OFF --nonce N [--ans1 F] [--ans2 F]
                 --ntok N --out J --text TXT          -> pskld's "LONGID {json}" line
  s5chain.py selftest --pcr2 P --pskld PL --tests T  offline: turn 1 = pcr2's deep payload + the nonce line; turn 2 starts with turn
                 1's message and is pcr2's turn payload; turn 3 starts with turn 2's messages; two nonces / offsets share no
                 1000-char prefix of the first message
"""
import argparse, importlib.util, json, os, sys, tempfile

sys.dont_write_bytecode = True
FOLLOW2 = ("\n\nGood. Now rewrite the one function you consider the weakest, in full, with your improvements applied, and say in "
           "two sentences why the new version is better.")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def nline(nonce):
    return "# s5 chain %s\n" % nonce


def payload(pc, pl, t, turn, depth, off, nonce, ans1, ans2, ntok):
    if turn == 1:
        d = pc.payload(pl, t, "deep:%d:%d" % (depth, off), ntok)
    elif turn in (2, 3):
        d = pc.payload(pl, t, "turn:%d:%d:%s" % (depth, off, ans1), ntok)
        if turn == 3:
            a2 = pc.answer_of(ans2)
            if a2 is None: raise SystemExit("turn 3: no answer file %s" % ans2)
            d["messages"] = list(d["messages"]) + [{"role": "assistant", "content": a2}, {"role": "user", "content": FOLLOW2.strip()}]
    else:
        raise SystemExit("--turn takes 1, 2 or 3")
    m0 = dict(d["messages"][0])
    m0["content"] = nline(nonce) + m0["content"]
    d["messages"] = [m0] + list(d["messages"][1:])
    return d


def selftest(a):
    pc, pl, t = load(a.pcr2, "pcr2_ro"), load(a.pskld, "pskld_ro"), load(a.tests, "ds41_tests_ro")
    with tempfile.TemporaryDirectory() as td:
        f1, f2 = os.path.join(td, "a1.txt"), os.path.join(td, "a2.txt")
        with open(f1, "w") as f: f.write("First answer.\n[reasoning]\nhidden")
        with open(f2, "w") as f: f.write("Second answer.")
        p1 = payload(pc, pl, t, 1, 2000, 3000, "X-1", f1, f2, 64)
        ref = pc.payload(pl, t, "deep:2000:3000", 64)
        p2 = payload(pc, pl, t, 2, 2000, 3000, "X-1", f1, f2, 64)
        ref2 = pc.payload(pl, t, "turn:2000:3000:%s" % f1, 64)
        p3 = payload(pc, pl, t, 3, 2000, 3000, "X-1", f1, f2, 64)
        q1 = payload(pc, pl, t, 1, 2000, 9000, "Y-1", f1, f2, 64)
    ok = p1["messages"][0]["content"] == nline("X-1") + ref["messages"][0]["content"] and len(p1["messages"]) == 1
    ok = ok and p2["messages"][0] == p1["messages"][0] and [m["content"] for m in p2["messages"][1:]] == [m["content"] for m in ref2["messages"][1:]]
    ok = ok and p2["messages"][1]["content"] == "First answer." and p2["messages"][1]["role"] == "assistant"
    ok = ok and p3["messages"][:3] == p2["messages"][:3] and p3["messages"][3] == {"role": "assistant", "content": "Second answer."}
    ok = ok and p3["messages"][4]["content"] == FOLLOW2.strip() and len(p3["messages"]) == 5
    ok = ok and os.path.commonprefix([p1["messages"][0]["content"], q1["messages"][0]["content"]]) == "# s5 chain "
    ok = ok and all(k in p3 for k in ("max_tokens",)) and p3.get("temperature") == ref.get("temperature")
    print("SELFTEST OK (3 turns, nonce + offset distinct)" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "selftest"))
    ap.add_argument("--pcr2", required=True)
    ap.add_argument("--pskld", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--turn", type=int, default=1)
    ap.add_argument("--depth", type=int, default=48000)
    ap.add_argument("--off", type=int, default=0)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--ans1", default="")
    ap.add_argument("--ans2", default="")
    ap.add_argument("--ntok", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--text")
    a = ap.parse_args()
    if a.mode == "selftest": return selftest(a)
    if not a.nonce: raise SystemExit("--nonce is required")
    pc, pl, t = load(a.pcr2, "pcr2_ro"), load(a.pskld, "pskld_ro"), load(a.tests, "ds41_tests_ro")
    p = payload(pc, pl, t, a.turn, a.depth, a.off, a.nonce, a.ans1, a.ans2, a.ntok)
    pl.deep_payload = lambda _t, _d, _n: p   # pskld.run asks for its deep payload: this one (as pcr2.py run does)
    ns = argparse.Namespace(url=a.url, tests=a.tests, mb=None, which="deep", out=a.out, text=a.text, load=False, conc=4,
                            warm=10, offset=0.5, pf_tok=0, ctx=1000, gen=512, depth=a.depth, ntok=a.ntok)
    return pl.run(ns)


if __name__ == "__main__":
    sys.exit(main())
