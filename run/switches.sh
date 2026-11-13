#!/bin/bash
# Usage: run/switches.sh run/qwen/switches.json [URL]
#        run/switches.sh run/deepseek/switches.json [URL]
# Waits until the server is healthy, sets the runtime switches that the results ran with, and checks
# that every switch took effect. The engine applies new switch values at its next step, so the script
# sends a 4-token request after the switches.
set -u
F=$1
URL=${2:-http://127.0.0.1:8090}
until curl -sf "$URL/health" > /dev/null; do sleep 5; done
curl -sf -X POST -H 'Content-Type: application/json' -d @"$F" "$URL/admin/fn4" > /dev/null \
  || { echo "POST $URL/admin/fn4 failed" >&2; exit 1; }
for try in 1 2 3 4 5 6 7 8 9 10; do
  curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"prompt": "Hello", "n_predict": 4, "temperature": 0}' "$URL/completion" > /dev/null
  if curl -s "$URL/admin/fn4" | python3 -c '
import json, sys
want = json.load(open(sys.argv[1]))
want.pop("epoch_bump", None)
got = json.load(sys.stdin)
now = got.get("switches", {})
off = {k: [v, now.get(k)] for k, v in want.items() if now.get(k) != v}
if off or got.get("pending"):
    print("waiting:", json.dumps(off or got.get("pending")), file=sys.stderr)
    sys.exit(1)' "$F"; then
    echo "all $(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))) - 1)' "$F") switches set from $F"
    exit 0
  fi
  sleep 3
done
echo "the switches did not take effect, see: curl -s $URL/admin/fn4" >&2
exit 1
