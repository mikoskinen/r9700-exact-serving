#!/usr/bin/env python3
"""mkcode.py ROOT OUT.txt CHARS [CAP] -- lever E1's realistic code prompt body: our own repo sources (ROOT = a radiance worktree),
round-robin over three lists (core/*.cpp, arch/deepseek41/deepseek41.cpp, libr4d/*.hip, each sorted), each file's first CAP
(48000) characters under a '// ==== path ====' header, cut at CHARS characters. Prints the files taken."""
import glob, os, sys
root, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 48000
lists = [sorted(glob.glob(os.path.join(root, "core", "*.cpp"))), [os.path.join(root, "arch", "deepseek41", "deepseek41.cpp")],
         sorted(glob.glob(os.path.join(root, "libr4d", "*.hip")))]
parts, k, taken, i = [], 0, [], 0
while k < n and any(lists):
    l = lists[i % 3]; i += 1
    if not l: continue
    f = l.pop(0)
    s = open(f, encoding="utf-8").read()[:cap]
    h = "\n// ==== %s ====\n" % os.path.relpath(f, root)
    parts.append(h + s); k += len(h) + len(s); taken.append(os.path.relpath(f, root))
text = "".join(parts)[:n]
open(out, "w", encoding="utf-8").write(text)
print("%d chars from %d files: %s" % (len(text), len(taken), " ".join(taken)))
