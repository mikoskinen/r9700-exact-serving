#!/usr/bin/env python3
"""mkfarm.py (12.11.2026, queue-ar/tools/mockrad) -- the sandbox HOME of a mockrun.sh run, so a body runs UNCHANGED (its
$HOME/gpu/ds41/... paths, lib-29.sh / lib-pr266.sh's T / R / L / D / C) while nothing it writes reaches the real tree.

  mkfarm.py build --real ~ --sb SB/home --manifest SB/manifest.json [--fresh REL ...] [--swap REL=SRC ...] [--copy-max BYTES]
  mkfarm.py verify --manifest SB/manifest.json      -> every real file a farm symlink points at whose size / mtime changed

Layout: gpu/ds41 is rebuilt directory by directory (real directories in the sandbox): code (.sh .py .yaml .yml .cfg .inc .env,
<= --copy-max) is COPIED, every other file is a symlink to the real one (references, models, binaries: read only); __pycache__
is left out; runs/ds41-exl3.log, runs/sweep.stop and runs/queue-changes.log are NOT linked (the body's log lines go to a fresh
sandbox log; a real stop file does not stop the mock). models/ is linked file by file the same way (a new file lands in the
sandbox). Every other entry of ~/gpu and radiance-1.15.1 is one directory symlink (read only use). --fresh REL: that directory
starts empty (the item's output dir: an earlier real run's files must not be appended to or truncated through a symlink).
--swap REL=SRC: the sandbox file REL is replaced by a copy of SRC (e.g. an old helper from its backup) -- the real files stay."""
import argparse, json, os, shutil, sys, time

CODE = (".sh", ".py", ".yaml", ".yml", ".cfg", ".inc", ".env")
NOLINK = ("gpu/ds41/exl3-test/runs/ds41-exl3.log", "gpu/ds41/exl3-test/runs/sweep.stop", "gpu/ds41/exl3-test/runs/queue-changes.log",
          "gpu/ds41/runs/sweep.stop")


def farm(real, sb, rel, man, copy_max, stats):
    src = os.path.join(real, rel)
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__"]
        rd = os.path.relpath(dp, real)
        if rd == "gpu/ds41/exl3-test/queue-ar/tools":   # the mock's own tool dir: one symlink (its reports are not the item's)
            if "mockrad" in dns:
                os.makedirs(os.path.join(sb, rd), exist_ok=True)
                os.symlink(os.path.join(dp, "mockrad"), os.path.join(sb, rd, "mockrad")); dns.remove("mockrad"); stats["dirlinks"] += 1
        os.makedirs(os.path.join(sb, rd), exist_ok=True)
        for d in list(dns):
            p = os.path.join(dp, d)
            if os.path.islink(p):   # a symlinked directory stays a symlink (to its real target)
                os.symlink(os.path.realpath(p), os.path.join(sb, rd, d)); dns.remove(d); stats["dirlinks"] += 1
        for f in fns:
            r = os.path.join(rd, f)
            if r in NOLINK: stats["skipped"] += 1; continue
            p = os.path.join(dp, f)
            try: st = os.stat(p)
            except OSError: continue
            if not (os.path.isfile(p)): continue
            dst = os.path.join(sb, r)
            if f.endswith(CODE) and st.st_size <= copy_max and not rel.startswith("models"):
                shutil.copy2(p, dst); stats["copied"] += 1
            else:
                tgt = os.path.realpath(p)
                os.symlink(tgt, dst); stats["linked"] += 1
                man[tgt] = [st.st_size, st.st_mtime_ns]


def build(a):
    t0 = time.time()
    real, sb = os.path.realpath(os.path.expanduser(a.real)), os.path.abspath(a.sb)
    os.makedirs(sb, exist_ok=True)
    man, stats = {}, {"copied": 0, "linked": 0, "dirlinks": 0, "skipped": 0}
    farm(real, sb, "gpu/ds41", man, a.copy_max, stats)
    for e in sorted(os.listdir(os.path.join(real, "gpu"))):
        if e != "ds41" and not os.path.exists(os.path.join(sb, "gpu", e)):
            os.symlink(os.path.realpath(os.path.join(real, "gpu", e)), os.path.join(sb, "gpu", e)); stats["dirlinks"] += 1
    if os.path.isdir(os.path.join(real, "models")):
        farm(real, sb, "models", man, 0, stats)
    for e in ["radiance-1.15.1"] + list(a.link or []):
        p = os.path.join(real, e)
        if os.path.exists(p) and not os.path.exists(os.path.join(sb, e)):
            os.symlink(os.path.realpath(p), os.path.join(sb, e)); stats["dirlinks"] += 1
    for f in a.fresh or []:
        d = os.path.join(sb, f)
        if os.path.realpath(d).startswith(sb + os.sep) and os.path.isdir(d) and not os.path.islink(d):
            shutil.rmtree(d)
        elif os.path.islink(d):
            os.remove(d)
        os.makedirs(d, exist_ok=True)
        stats.setdefault("fresh", []).append(f)
    for s in a.swap or []:
        r, src = s.split("=", 1)
        if not os.path.isabs(src): src = os.path.join(real, src)
        dst = os.path.join(sb, r)
        if os.path.lexists(dst): os.remove(dst)
        shutil.copy2(src, dst)
        stats.setdefault("swapped", []).append("%s <- %s" % (r, src))
    json.dump(man, open(a.manifest, "w"))
    stats["seconds"] = round(time.time() - t0, 1)
    print("FARM " + json.dumps(stats))


def verify(a):
    man = json.load(open(a.manifest))
    ch = []
    for p, (sz, mt) in man.items():
        try: st = os.stat(p)
        except OSError: ch.append((p, "gone")); continue
        if st.st_size != sz or st.st_mtime_ns != mt: ch.append((p, "size %d -> %d" % (sz, st.st_size)))
    print("VERIFY %d linked real files, %d changed during the run%s" % (len(man), len(ch), "".join("\n  CHANGED %s (%s)" % c for c in ch[:50])))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("build", "verify"))
    ap.add_argument("--real", default="~")
    ap.add_argument("--sb")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fresh", action="append")
    ap.add_argument("--swap", action="append")
    ap.add_argument("--link", action="append")
    ap.add_argument("--copy-max", type=int, default=2_000_000)
    a = ap.parse_args()
    return build(a) if a.mode == "build" else verify(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
