#!/bin/bash
# pick103.sh TABLE (9.11.2026, queue-ar): the longest gpu_metrics poll interval (ms) of 103's B<n> conditions whose every
# start came out fast; 5 (101's proven interval) when the table is missing or no B<n> passed. Prints one number.
awk '$2 ~ /^B[0-9]+$/ { n = substr($2, 2) + 0; seen[n] = 1; if ($3 != "fast") bad[n] = 1 }
     END { best = 5; for (n in seen) if (!(n in bad) && n + 0 > best) best = n + 0; print best }' "${1:-/dev/null}" 2>/dev/null || echo 5
