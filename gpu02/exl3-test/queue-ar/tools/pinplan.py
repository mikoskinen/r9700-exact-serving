#!/usr/bin/env python3
"""pinplan.py CPULAT.txt RANK_BDFS MODE (9.11.2026, queue-ar, item 112) -- RADIANCE_RANK_CPUS for the ranks from cpulat's
CCDMAP line. RANK_BDFS: the ranks' cards in rank order, comma-separated ("c3:00.0,83:00.0,86:00.0,03:00.0"). MODE near:
every rank on the CCD whose CPUs reach its card fastest (cpulat's kern round trip); far: on the slowest one. Two ranks that
want the same CCD share it one CCX each (Zen 2 EPYC 7302P: CCD k = CPUs 4k..4k+3 and their SMT siblings +16; CCX = two
cores of it); a third and fourth take their next choice. Prints two lines:
  PLAN <mode> rank0=<bdf>:ccd<k>[x<half>]:<cpus> ...  (with the kern medians the choice rests on)
  CPUS <list0>;<list1>;...                             (the RADIANCE_RANK_CPUS value)"""
import sys


def ccd_cpus(k, half=None):
    a = 4 * k
    if half is None:
        return f"{a}-{a + 3},{a + 16}-{a + 19}"
    b = a + 2 * half
    return f"{b}-{b + 1},{b + 16}-{b + 17}"


def main():
    txt = open(sys.argv[1]).read().splitlines()
    ranks = sys.argv[2].split(",")
    mode = sys.argv[3]
    line = next((l for l in txt if l.startswith("CCDMAP")), None)
    if not line:
        print("PLAN none: no CCDMAP line in " + sys.argv[1]); print("CPUS "); return 1
    order, med = {}, {}
    for f in line.split()[1:]:
        _, rest = f.split("=", 1)
        # card=<bdf>:<order>:<medians>, bdf itself holds two colons ("83:00.0")
        parts = rest.split(":")
        bdf = ":".join(parts[:2]); o = [int(x) for x in parts[2].split(",")]; m = [float(x) for x in parts[3].split(",")]
        order[bdf] = o if mode == "near" else o[::-1]
        med[bdf] = dict(zip(o, m))
    take = {}   # ccd -> ranks on it
    choice = {}
    for r, bdf in enumerate(ranks):
        if bdf not in order:
            print(f"PLAN none: card {bdf} not in the CCDMAP line"); print("CPUS "); return 1
    # the ranks with the clearest preference first
    def margin(bdf):
        o = order[bdf]
        return abs(med[bdf][o[1]] - med[bdf][o[0]]) if len(o) > 1 else 0.0
    for r in sorted(range(len(ranks)), key=lambda r: -margin(ranks[r])):
        bdf = ranks[r]
        for k in order[bdf]:
            if len(take.get(k, [])) < 2:
                take.setdefault(k, []).append(r); choice[r] = k; break
    cpus, plan = [], []
    for r, bdf in enumerate(ranks):
        k = choice[r]
        on = take[k]
        half = None if len(on) == 1 else on.index(r)
        cl = ccd_cpus(k, half)
        cpus.append(cl)
        plan.append(f"rank{r}={bdf}:ccd{k}{'' if half is None else 'x%d' % half}:{cl}(kern {med[bdf][k]:.2f} us; "
                    f"best {min(med[bdf].values()):.2f} worst {max(med[bdf].values()):.2f})")
    print(f"PLAN {mode} " + " ".join(plan))
    print("CPUS " + ";".join(cpus))
    return 0


if __name__ == "__main__":
    sys.exit(main())
