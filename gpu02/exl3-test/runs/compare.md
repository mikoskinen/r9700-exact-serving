| mittari | yksikkö | UD-IQ1_S 3.11. | EXL3 | ero | huom. |
|---|---|---:|---:|---:|---|
| 1 käyttäjä, essee | tok/s | 62,60 | 52,90 | -15,5 % | 3.11. 15,9 ms/askel |
| 1 käyttäjä, koodi | tok/s | 68,50 | 51,59 | -24,7 % | 3.11. 14,5 ms/askel |
| 1 käyttäjä, askel (koodi) | ms | 14,50 | 19,37 | +33,6 % |  |
| 4 virtaa, dekoodi yhteensä | tok/s | 125,7 | 99,00 | -21,2 % | 3.11. sama kokoonpano |
| 8 virtaa, dekoodi yhteensä | tok/s | 164,0 | 123,8 | -24,5 % | 3.11. ckint 2048 ilman span/lomitusta |
| 31k esitäyttö 4 dekooderin rinnalla | tok/s | 1 475 | 1 284 | -12,9 % | 3.11. TTFT 21,3 s |
| 56k istunto, kylmä TTFT | s | 27,80 | 29,91 | +7,6 % |  |
| jatkovuoro, TTFT ka | s | 2,16 | 2,27 | +5,1 % | 3.11. uudelleenlaskenta 3 318 |
| kylmä esitäyttö 16k | tok/s | 2 844 | 1 975 | -30,5 % | aamun liput: DSpark 5, ckint 8192 |
| kylmä esitäyttö ~59k | tok/s | 2 641 | 2 105 | -20,3 % |  |
| kylmä esitäyttö ~140k | tok/s | 2 316 | 1 963 | -15,2 % |  |
| kylmä esitäyttö 266k (1M-tila) | tok/s | 1 805 | - | - |  |
| kylmä esitäyttö 501k | tok/s | 1 287 | - | - |  |
| kylmä esitäyttö ~644k | tok/s | 1 081 | - | - |  |
| jatkovuoro syvimmässä istunnossa | s | 8,01 | - | - | 3.11. 644k: uudelleenlaskenta 4 421 |
| Claude Code -simulaatio 4 käyttäjää, TTFT p50 | s | 3,92 | 3,88 | -1,0 % |  |
| Claude Code -simulaatio 4 käyttäjää, per agentti | tok/s | 29,10 | 21,50 | -26,1 % |  |
| Claude Code -simulaatio 4 käyttäjää, yhteensä | tok/s | 92,50 | 71,90 | -22,3 % |  |
| Claude Code -simulaatio 8 käyttäjää, TTFT p50 | s | 4,31 | 4,75 | +10,2 % |  |
| Claude Code -simulaatio 8 käyttäjää, per agentti | tok/s | 14,40 | 10,30 | -28,5 % |  |
| Claude Code -simulaatio 8 käyttäjää, yhteensä | tok/s | 94,50 | 63,80 | -32,5 % |  |
