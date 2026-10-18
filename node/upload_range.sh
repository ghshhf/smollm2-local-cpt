#!/bin/bash
# Upload base shards [start..end] (2-digit) to ECS /file/ as b200_<ii>, verify via POST 'saved'.
cd "D:/work/_local_ai/train_15b/node" || exit 1
ECS="http://47.99.131.12:8081"
S=$1; E=$2
for i in $(seq -w "$S" "$E"); do
  f="parts/r4base_bin.$i"
  n="b200_$i"
  local_sz=$(wc -c < "$f")
  resp=$(curl -s -m300 -X POST "$ECS/upload?name=$n&key=k0u4mail" --data-binary @"$f")
  saved=$(printf '%s' "$resp" | python3 -c "import sys,json;
try: print(json.load(sys.stdin).get('saved',-1))
except: print(-1)")
  if [ "$saved" = "$local_sz" ]; then
    echo "OK $n saved=$saved"
  else
    echo "FAIL $n saved=$saved want=$local_sz resp=$resp"
  fi
done
echo "RANGE_DONE $S-$E"
