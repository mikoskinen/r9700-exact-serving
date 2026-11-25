#!/usr/bin/env python3
"""dagen.py (11.11.2026, queue-ar/dsmain/da, lever DA pilot) -- the pilot's corpus for the KL-record dump. Standard library only.
Public or synthetic text only: synthetic prompts written for this pilot, Wikipedia extracts (da/public), the Python standard
library's sources. Never a user transcript, a LiteLLM log or any user data.

  val URL LOG OUT                         against a serving NST 5 start (-v, fixed window 5, walk 1):
      1. VALIDATION at c1: VAL prompts (2 Finnish, 2 English, 2 code), each rendered by /tokenize (messages, reasoning none),
         then /completion with those ids (greedy, 256 tokens, return_tokens); the engine's -v proposal lines that appeared
         meanwhile (the followed log LOG) are replayed against the reply (daanat.py's replay, first anchor 0 or 1) into
         absolute anchors -> OUT/val.json [{source, prompt_len, blocks: [[anchor, [5 ids]], ...]}]
         + OUT/corpus-val.jsonl
  gen URL OUT --minutes M --conc C        2. SELF-GENERATED (target greedy) at cC until M minutes: GEN prompts (Finnish /
         English prose, code), 448 tokens each -> OUT/corpus-gen.jsonl {"ids": prompt ids + reply ids, "source": "<kind>:<i>"}
         (the KL corpus form)
  public PUBDIR OUT --fi N --en N --code N   teacher-forced public documents of ~1600-token slices (characters / ratio),
      BOS-led, as KL corpus lines {"prompt", "source": "pub-fi:<title>"} -> OUT/corpus-pub.jsonl
  selftest                                the replay on a fabricated reply; the prompt sets' sizes and kinds
Prints "DAGEN {json}" lines."""
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

PAT = re.compile(r"drafter '[^']*': sequence (\d+) proposes((?: -?\d+)+)")
BOS = "<｜begin▁of▁sentence｜>"

VAL = [
    ("val-fi", "Kirjoita noin 300 sanan kuvaus siitä, miten suomalainen kesämökkiviikonloppu yleensä kulkee perjantaista sunnuntaihin."),
    ("val-fi", "Selitä selkeällä yleiskielellä, miten sähköverkko pysyy tasapainossa, kun tuuli- ja aurinkovoiman osuus kasvaa."),
    ("val-en", "Write a short essay of about 300 words on why public parks matter for city life, with two concrete examples."),
    ("val-en", "Explain how a refrigerator moves heat from the inside to the outside, step by step, for a curious teenager."),
    ("val-code", "Write a Python function that merges overlapping intervals in a list, with type hints, a docstring and three unit tests."),
    ("val-code", "Write a C function that reverses a singly linked list in place, a small struct definition and a main() that tests it."),
]

FI = [
    "Kirjoita pitkä ja huolellinen essee Suomen metsien merkityksestä taloudelle, luonnolle ja ihmisten hyvinvoinnille.",
    "Kerro tarina vanhasta kalastajasta, joka löytää rantakiveltä outoja merkkejä. Kirjoita noin 500 sanaa.",
    "Selitä lukiolaiselle, miten rokotteet harjoittavat immuunijärjestelmää. Käytä esimerkkejä ja useita kappaleita.",
    "Kirjoita blogikirjoitus siitä, miten aloittaa juoksuharrastus aikuisena turvallisesti ja motivoituneena.",
    "Kuvaile syksyistä kävelyä suomalaisessa sekametsässä aistien kautta: värit, äänet, tuoksut ja tunnelma.",
    "Kirjoita mielipidekirjoitus kunnanvaltuustolle pyöräteiden lisäämisen puolesta konkreettisin perustein.",
    "Selitä, miten pankkien korot vaikuttavat asuntolainaan ja mitä lainanottajan kannattaa huomioida.",
    "Kirjoita opastava teksti hapanjuurileivän leipomisesta alusta loppuun, vinkkeineen ja yleisine virheineen.",
    "Kerro Suomen talvisodan pääpiirteet tiiviisti mutta huolellisesti, syistä seurauksiin.",
    "Kirjoita uutisartikkeli kuvitteellisesta pikkukaupungista, joka avaa ensimmäisen julkisen saunansa satamaan.",
    "Pohdi esseessä, mikä tekee hyvästä opettajasta. Käytä esimerkkejä eri oppiaineista.",
    "Selitä tavallisella kielellä, miten tietokoneen prosessori suorittaa käskyjä: liukuhihna, välimuistit ja haarautumisen ennustus.",
    "Kirjoita matkaopas kolmen päivän vierailusta Lapissa talvella, käytännön vinkein jokaiselle päivälle.",
    "Kirjoita novelli, jossa pieni tyttö ja hänen isoisänsä rakentavat linnunpöntön keväällä.",
    "Vertaile esseessä aurinko-, tuuli- ja vesivoimaa Suomen olosuhteissa: kustannukset, rajoitteet ja tulevaisuus.",
    "Laadi ohje, miten kotitalous voi pienentää sähkönkulutustaan talvella ilman, että asumismukavuus kärsii.",
    "Kirjoita puhe ylioppilasjuhlaan uteliaisuudesta ja sinnikkyydestä.",
    "Kerro, miten Kalevala syntyi ja miksi sillä on ollut niin suuri merkitys suomalaiselle kulttuurille.",
    "Selitä, mitä tekoäly on ja mitä se ei ole, tavalliselle lukijalle ilman teknistä jargonia.",
    "Kirjoita arvostelu kuvitteellisesta kotimaisesta elokuvasta: juoni, näyttelijät, kuvaus ja kokonaisuus.",
    "Kuvaile päivä hiljaisessa saaristokylässä kesäkuun alussa aamusta iltaan.",
    "Kirjoita ohjeistus uudelle työntekijälle ensimmäisestä työviikosta pienessä ohjelmistoyrityksessä.",
    "Selitä vesikierron vaiheet ja niiden merkitys ilmastolle oppikirjan luvun tyyliin.",
    "Kirjoita kirje ystävälle, joka on muuttamassa ulkomaille, ja anna hänelle neuvoja ja rohkaisua.",
    "Pohdi, miten etätyö on muuttanut suomalaista työelämää ja millaisia hyviä ja huonoja puolia sillä on.",
    "Kirjoita historiallinen kertomus Saimaan kanavan rakentamisesta ja sen vaikutuksesta Itä-Suomeen.",
    "Selitä, miten kasvihuoneilmiö toimii ja miksi hiilidioksidin pitoisuus ilmakehässä on tärkeä.",
    "Kirjoita käytännön opas sienestykseen aloittelijalle: lajit, turvallisuus ja säilöntä.",
    "Kerro, miten suomalainen koulujärjestelmä on rakentunut ja mitkä ovat sen vahvuudet.",
    "Kirjoita tarina kahdesta naapurista, jotka riitelevät aidasta mutta päätyvät ystäviksi.",
    "Selitä, miten nuotit ja rytmi toimivat, kun aikuinen aloittaa pianonsoiton opiskelun.",
    "Kirjoita essee kirjastojen merkityksestä demokratialle ja sivistykselle.",
    "Kuvaile jääkiekko-ottelun tunnelmaa katsomossa kuin urheilutoimittaja.",
    "Laadi tiivis mutta kattava selitys inflaatiosta ja sen vaikutuksesta tavallisen ihmisen arkeen.",
    "Kirjoita kuvaus revontulista ja siitä, miten ne syntyvät, sekä runollisesti että tieteellisesti.",
]
EN = [
    "Write a long, careful essay on how railways changed nineteenth-century Europe: trade, cities, time and war.",
    "Tell a story of about 500 words about a night-shift nurse who notices something odd on the hospital's top floor.",
    "Explain for a general audience how the immune system tells friend from foe, with several concrete examples.",
    "Write a blog post with practical advice for someone who wants to start cycling to work in a rainy city.",
    "Describe a walk along a rocky coastline at dawn, focusing on colours, sounds, smells and mood.",
    "Write an opinion piece for a local paper arguing that every school should teach basic cooking.",
    "Explain how interest rates affect mortgages and what a first-time buyer should keep in mind.",
    "Write a detailed guide to growing tomatoes on a balcony, from seed to harvest, with common mistakes.",
    "Summarise the causes and the course of the French Revolution in a careful essay of about 700 words.",
    "Write a news article about a fictional village that builds its own small hydroelectric plant.",
    "Reflect in an essay on what makes a good manager, with examples from different kinds of work.",
    "Explain in plain words how a search engine finds and ranks web pages.",
    "Write a travel guide section for a three-day visit to Lisbon, with practical tips for each day.",
    "Write a short story in which a retired clockmaker repairs a stranger's broken pocket watch.",
    "Compare nuclear, wind and solar power in an essay that weighs their costs, risks and limits.",
    "Write a guide for a household that wants to cut its energy use in winter without being cold.",
    "Write a commencement speech about patience and craftsmanship.",
    "Explain how plate tectonics shapes mountains, earthquakes and volcanoes, with three examples.",
    "Explain what artificial intelligence is and what it is not, for a reader with no technical background.",
    "Write a review of a fictional independent film: plot, acting, cinematography and overall impression.",
    "Describe a quiet day in a small fishing harbour in early summer, from morning to evening.",
    "Write an onboarding note for a new hire's first week at a small software company.",
    "Explain the carbon cycle and its role in the climate as a section of a textbook chapter.",
    "Write a letter to a friend who is moving abroad, with advice and encouragement.",
    "Discuss how remote work has changed office life, weighing its benefits and drawbacks.",
    "Write a historical narrative about the building of the Brooklyn Bridge.",
    "Explain how vaccines are tested before approval, phase by phase.",
    "Write a beginner's guide to bird watching: equipment, places, habits and etiquette.",
    "Describe how a modern public library works behind the scenes.",
    "Write a story about two rival bakers on the same street who end up working together.",
    "Explain the basics of music theory for an adult beginner: notes, scales, chords and rhythm.",
    "Write an essay on why maps matter, from ancient charts to satellite navigation.",
    "Describe the atmosphere at a football match as a sports reporter would.",
    "Explain inflation and how it affects everyday life, with simple numbers.",
    "Explain how the northern lights form, both poetically and scientifically.",
]
CODE = [
    "Write a complete Python module implementing a priority task scheduler with retries and exponential backoff, with docstrings and unittest tests.",
    "Write a Python script that parses an nginx access log and prints the top 10 IPs, paths and status codes; use argparse.",
    "Implement a min-heap in C++ with push, pop, top and a small main() that tests it.",
    "Write a JavaScript module with a tiny event emitter (on, off, once, emit), JSDoc comments and examples.",
    "Write a Rust program that reads a CSV of temperatures and prints per-city min, max and mean, with unit tests.",
    "Write a Bash script that rotates log files in a directory, compresses old ones and keeps the last 14.",
    "Write a Go HTTP server with two JSON endpoints (health and echo), graceful shutdown and a test.",
    "Write SQL to create tables for a small library (books, members, loans) and five useful queries with comments.",
    "Write a Python dataclass-based inventory system with add, remove, search and a JSON save/load, plus tests.",
    "Implement a trie in Java with insert, search, startsWith and delete, and a main method that tests it.",
    "Write a TypeScript function that debounces async calls and cancels stale ones, with types and examples.",
    "Write a C program that counts words, lines and characters in a file like wc, with error handling.",
    "Write a Python implementation of Dijkstra's algorithm on an adjacency list, with a docstring and tests.",
    "Write a Rust struct for a fixed-size ring buffer with push, pop and iteration, and unit tests.",
    "Write a Python CLI tool that renames photos by their EXIF date (assume a get_exif_date helper exists), with dry-run mode.",
    "Write a C++ class for a 2D matrix with addition, multiplication, transpose and stream output, plus tests in main().",
    "Write a Kotlin data class and repository for todo items with an in-memory store and unit tests.",
    "Write a Python async web crawler limited to one domain and N concurrent requests, using asyncio and aiohttp-style pseudocode.",
    "Write a shell script that checks disk usage on every mount and emails a warning above 90 percent.",
    "Write a Python module that validates and normalises Finnish personal identity codes (henkilötunnus), with tests.",
]


def http(url, body=None, timeout=900):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def render(url, prompt):
    d = http(url + "/tokenize", {"messages": [{"role": "user", "content": prompt}], "reasoning_effort": "none"})
    return [int(t) for t in d["tokens"]]


def complete(url, ids, n):
    d = http(url + "/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0, "return_tokens": True})
    return [int(t) for t in (d.get("tokens") or [])], d


def replay(ids, blocks, p0):
    p, runs, anchors = p0, [], []
    for d in blocks:
        if p + 1 >= len(ids):
            break
        anchors.append(p)
        run = 0
        for j, t in enumerate(d[:5]):
            if p + 1 + j >= len(ids) or t != ids[p + 1 + j]:
                break
            run += 1
        runs.append(run)
        p += run + 1
    return anchors, runs


def logsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def val_cmd(a):
    url = a.url.rstrip("/")
    os.makedirs(a.out, exist_ok=True)
    corpus = open(os.path.join(a.out, "corpus-val.jsonl"), "w")
    val = []
    # 1. validation, c1, the -v proposals
    for i, (kind, p) in enumerate(VAL):
        ids = render(url, p)
        o0 = logsize(a.log)
        t0 = time.time()
        rep, d = complete(url, ids, a.val_tokens)
        time.sleep(a.settle)
        with open(a.log, "rb") as f:
            f.seek(o0)
            chunk = f.read(max(0, logsize(a.log) - o0)).decode(errors="replace")
        props = {}
        for m in PAT.finditer(chunk):
            props.setdefault(m.group(1), []).append([int(x) for x in m.group(2).split()])
        seq, blocks = max(props.items(), key=lambda kv: len(kv[1])) if props else (None, [])
        best = max(((replay(rep, blocks, p0), p0) for p0 in (0, 1)), key=lambda x: (sum(x[0][1]), -x[1]))
        (anchors, runs), p0 = best
        src = "%s:%d" % (kind, i)
        val.append({"source": src, "prompt_len": len(ids), "first_anchor": p0,
                    "blocks": [[len(ids) + an, b[:5]] for an, b in zip(anchors, blocks)]})
        corpus.write(json.dumps({"ids": ids + rep, "source": src}) + "\n")
        print("DAGEN " + json.dumps({"phase": "val", "source": src, "prompt": len(ids), "reply": len(rep), "blocks": len(blocks),
                                     "replayed": len(anchors), "tokens_a_step": round(sum(r + 1 for r in runs) / max(1, len(runs)), 3),
                                     "seqs_in_window": len(props), "s": round(time.time() - t0, 1)}), flush=True)
    json.dump(val, open(os.path.join(a.out, "val.json"), "w"))
    corpus.close()
    return 0 if all(v["blocks"] for v in val) else 1


def continuation_jobs(pubdir, per_lang=120, n_code=60):
    """more target-greedy text once the written prompts are spent: continue a public paragraph (Wikipedia fi / en) or the
    head of a Python standard-library module, in the same style"""
    rnd = random.Random(13)
    jobs = []
    for lang, head in (("fi", "Jatka tätä tekstiä luontevasti samalla tyylillä, noin 400 sanaa:\n\n"),
                       ("en", "Continue this text naturally in the same style, about 400 words:\n\n")):
        path = os.path.join(pubdir or "", "wiki-%s.jsonl" % lang)
        paras = []
        if os.path.exists(path):
            for l in open(path):
                paras += [p.strip() for p in json.loads(l)["text"].split("\n") if len(p.strip()) >= 400]
        rnd.shuffle(paras)
        jobs += [("gen-%s" % lang, 1000 + k, head + p[:1200]) for k, p in enumerate(paras[:per_lang])]
    files = []
    for d in sorted(glob.glob("/usr/lib/python3*/*/")):
        if os.path.basename(d.rstrip("/")) in ("collections", "ctypes", "dataclasses", "functools", "pathlib", "tomllib",
                                               "zipfile", "statistics", "re", "string", "textwrap", "venv"):
            files += sorted(glob.glob(d + "*.py"))
    rnd.shuffle(files)
    for k, f in enumerate(files[:n_code]):
        try:
            src = "".join(open(f, encoding="utf-8").readlines()[:50])
        except Exception:
            continue
        jobs.append(("gen-code", 1000 + k, "Continue this Python module with well-commented code:\n\n```python\n" + src + "```"))
    rnd.shuffle(jobs)
    return jobs


def gen(a):
    url = a.url.rstrip("/")
    os.makedirs(a.out, exist_ok=True)
    corpus = open(os.path.join(a.out, "corpus-gen.jsonl"), "w")
    lock = threading.Lock()
    # 2. self-generated, cC, until the deadline
    jobs = [("gen-fi", i, p) for i, p in enumerate(FI)] + [("gen-en", i, p) for i, p in enumerate(EN)] + \
           [("gen-code", i, p) for i, p in enumerate(CODE)]
    random.Random(7).shuffle(jobs)
    jobs = continuation_jobs(a.pubdir) + jobs          # popped from the end: the written prompts first, then continuations
    deadline = time.time() + 60 * a.minutes
    stats = {"docs": 0, "tokens": 0, "errors": 0}

    def worker():
        while time.time() < deadline:
            with lock:
                if not jobs:
                    return
                kind, i, p = jobs.pop()
            try:
                ids = render(url, p)
                rep, _ = complete(url, ids, a.gen_tokens)
            except Exception as e:
                with lock:
                    stats["errors"] += 1
                print("DAGEN " + json.dumps({"phase": "gen", "error": str(e)[:200]}), flush=True)
                continue
            with lock:
                corpus.write(json.dumps({"ids": ids + rep, "source": "%s:%d" % (kind, i)}) + "\n")
                corpus.flush()
                stats["docs"] += 1
                stats["tokens"] += len(ids) + len(rep)
    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    corpus.close()
    print("DAGEN " + json.dumps({"phase": "gen", "s": round(time.time() - t0, 1), "left": len(jobs), **stats}), flush=True)
    return 0


def public(a):
    rnd = random.Random(11)
    out = open(a.out, "w")
    n = {"fi": 0, "en": 0, "code": 0}

    def emit(kind, title, text):
        out.write(json.dumps({"prompt": BOS + text, "source": "pub-%s:%s" % (kind, title[:60])}, ensure_ascii=False) + "\n")

    for lang, budget, cpt in (("fi", a.fi, 3.4), ("en", a.en, 4.3)):
        path = os.path.join(a.pubdir, "wiki-%s.jsonl" % lang)
        arts = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
        rnd.shuffle(arts)
        for art in arts:
            text = art["text"]
            for k in range(0, len(text), int(1600 * cpt)):
                if n[lang] >= budget:
                    break
                piece = text[k:k + int(1600 * cpt)]
                if len(piece) < 1500:
                    continue
                emit(lang, art["title"], piece)
                n[lang] += int(len(piece) / cpt)
    # code: the Python standard library's own sources, packages ds41_tests.py's copy text does not start from
    stdlib = sorted(glob.glob("/usr/lib/python3*/*/"))
    files = []
    for d in stdlib:
        if os.path.basename(d.rstrip("/")) in ("email", "json", "http", "asyncio", "concurrent", "logging", "xml", "urllib",
                                               "importlib", "sqlite3", "unittest", "multiprocessing"):
            files += sorted(glob.glob(d + "**/*.py", recursive=True))
    rnd.shuffle(files)
    for f in files:
        if n["code"] >= a.code:
            break
        try:
            text = open(f, encoding="utf-8").read()
        except Exception:
            continue
        for k in range(0, len(text), 5600):
            piece = text[k:k + 5600]
            if len(piece) < 1500 or n["code"] >= a.code:
                break
            emit("code", os.path.relpath(f, "/usr/lib"), piece)
            n["code"] += len(piece) // 3
    out.close()
    print("DAGEN " + json.dumps({"phase": "public", "est_tokens": n, "out": a.out}), flush=True)
    return 0


def selftest():
    ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    blocks = [[11, 77, 1, 1, 1], [13, 14, 15, 16, 66], [18, 5, 0, 0, 0], [20, 21, 0, 0, 0]]
    anchors, runs = replay(ids, blocks, 0)
    ok = anchors == [0, 2, 7, 9] and runs == [1, 4, 1, 2]
    ok = ok and len(VAL) == 6 and len(FI) >= 30 and len(EN) >= 30 and len(CODE) >= 20
    print("DAGEN " + json.dumps({"selftest": "ok" if ok else "FAIL", "anchors": anchors, "runs": runs,
                                 "prompts": {"val": len(VAL), "fi": len(FI), "en": len(EN), "code": len(CODE)}}))
    return 0 if ok else 1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    v = sp.add_parser("val")
    v.add_argument("url"); v.add_argument("log"); v.add_argument("out")
    v.add_argument("--val-tokens", type=int, default=256); v.add_argument("--settle", type=float, default=1.5)
    g = sp.add_parser("gen")
    g.add_argument("url"); g.add_argument("out")
    g.add_argument("--minutes", type=float, default=10); g.add_argument("--conc", type=int, default=8)
    g.add_argument("--gen-tokens", type=int, default=448); g.add_argument("--pubdir", default="")
    p = sp.add_parser("public")
    p.add_argument("pubdir"); p.add_argument("out")
    p.add_argument("--fi", type=int, default=90000); p.add_argument("--en", type=int, default=50000)
    p.add_argument("--code", type=int, default=50000)
    sp.add_parser("selftest")
    a = ap.parse_args()
    return {"val": val_cmd, "gen": gen, "public": public, "selftest": lambda a: selftest()}.get(a.cmd, lambda a: 2)(a)


if __name__ == "__main__":
    sys.exit(main())
