#!/bin/sh
# Lukee AMD Data Fabricin DRAM-osoitekartat (D18F0x110/0x114) joka CS-instanssilta (UMC 0-7)
# epasuoran FICAA/FICAD-polun kautta ja tulostaa, mitka kanavat palvelevat mitakin osoitealuetta.
# Ajo: sudo ./df-map.sh
[ "$(id -u)" = 0 ] || { echo "aja sudolla"; exit 1; }
rd(){ f=$(( 1 | ($2 & 0x3FC) | ($1<<16) )); setpci -s 00:18.4 0x5c.l=$(printf %x $f); setpci -s 00:18.4 0x98.l; }
for i in 0 1 2 3 4 5 6 7; do
  for m in 0 1 2 3; do
    echo "cs$i map$m b=0x$(rd $i $((0x110+8*m))) l=0x$(rd $i $((0x114+8*m)))"
  done
done | python3 -c '
import sys,re
modes={0:"ei lomitusta",1:"2 kanavaa",3:"4 kanavaa (koodi 3)",5:"8 kanavaa",7:"16 kanavaa",0xC:"COD4 2ch hash",0xD:"COD2 4ch hash",0xE:"COD1 8ch hash"}
ch="01234567"
for line in sys.stdin:
    m=re.match(r"cs(\d) map(\d) b=0x([0-9a-f]+) l=0x([0-9a-f]+)",line.strip())
    if not m: continue
    i,mi,b,l=int(m[1]),int(m[2]),int(m[3],16),int(m[4],16)
    if not b&1: continue
    base=(b>>12)<<28; lim=((l>>12)<<28)+(1<<28)
    print(f"UMC{i} (instanssi {ch[i]}, ei suoraan emon kirjain): alue {base/2**30:6.2f} - {lim/2**30:6.2f} GiB  lomitus={modes.get((b>>4)&0xF,hex((b>>4)&0xF))}  osoitebitti={8+((b>>8)&7)}  dstFabricId=0x{l&0x3ff:03x}  mmio-hole={(b>>1)&1}")
'
