#!/usr/bin/env python3
"""ngshare_hdr.py (12.11.2026, ds-main-port ngshare builder) -- read an n-gram share file's header the way Radiance's
classify() does (core/format/table_share.cpp:44-75, radiance-int8 7e03384), WITHOUT taking part in the protocol: no lock,
read only, never creates a file. Used by ngram-keeper.sh so that the keeper locks a file only once it is marked ready.

  python3 -I ngshare_hdr.py FILE [ID]   -> one line: "ready <table bytes> <file bytes> filler_pid <pid>" | "fresh" |
                                           "partial" | "foreign: <why>"; exit 0 only for ready (1 otherwise, 2 usage/IO).
FILE may be /dev/fd/N (the keeper checks the very file description it is about to lock). ID defaults to the <id> of the
name ngram-<id>.tab (table_share.cpp:77-80); with /dev/fd/N give it explicitly.

Header (table_share.cpp:28-35, host-endian, x86 = little): char magic[8] "RADNGRM1", u32 version (1), u32 ready,
u64 bytes, u64 filler_pid, char id[24] -> 56 bytes. File length when ready: 2 MiB + bytes rounded up to 2 MiB
(table_share_file_bytes, table_share.cpp:90-92).
"""
import os
import re
import struct
import sys

MAGIC = b"RADNGRM1"
VERSION = 1
HEADER = struct.Struct("<8sIIQQ24s")     # sizeof(Header) == 56
HUGE = 2 << 20                           # kHuge (table_share.cpp:22) == kTableShareHeader (table_share.h:26)


def file_bytes(n):
    return HUGE + ((n + HUGE - 1) & ~(HUGE - 1))


def classify(fd, ident):
    """(state, detail): state in fresh / partial / ready / foreign, in table_share.cpp:44-75's order."""
    st = os.fstat(fd)
    if st.st_size == 0:
        return "fresh", ""                                                       # :50
    raw = os.pread(fd, HEADER.size, 0)
    if len(raw) != HEADER.size:
        return "foreign", "its header is %d bytes short" % (HEADER.size - len(raw))  # :52-56
    if raw == bytes(HEADER.size):
        return "partial", ""                                                     # :57-58
    magic, version, ready, nbytes, pid, hid = HEADER.unpack(raw)
    if magic != MAGIC or version != VERSION:
        return "foreign", "it is not a table file of this version"              # :59-62
    hid = hid.split(b"\0", 1)[0].decode("ascii", "replace")
    if ident is not None and hid[:16] != ident:
        return "foreign", "it holds table '%s', not '%s'" % (hid[:16], ident)    # :63-67 (the id half)
    if not ready:
        return "partial", ""                                                     # :68
    if st.st_size != file_bytes(nbytes):
        return "foreign", "marked ready but %d bytes long, not %d" % (st.st_size, file_bytes(nbytes))  # :69-73
    return "ready", "%d %d filler_pid %d" % (nbytes, st.st_size, pid)            # :74


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    path = argv[1]
    ident = argv[2] if len(argv) == 3 else None
    if ident is None:
        m = re.fullmatch(r"ngram-([0-9a-f]{16})\.tab", os.path.basename(path))
        ident = m.group(1) if m else None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)       # no O_CREAT: a missing file stays missing
    except OSError as e:
        print("missing: %s" % e.strerror)
        return 2
    try:
        state, detail = classify(fd, ident)
    finally:
        os.close(fd)
    print(state + (": " if state == "foreign" else " ") + detail if detail else state)
    return 0 if state == "ready" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
