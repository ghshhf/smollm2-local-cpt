#!/bin/sh
# bootstrap_v1.sh — node1: stage pristine 200M base from ECS shards, then RESET + launch fp32 R4 chain.
# Runs once on node1 (after restart or when queued). Self-contained & idempotent.
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
LOG=$W/bootstrap.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 0. kill any lingering broken training (pattern obfuscated to avoid self-match)
T='train_remo''te_v2.py'; C='chain_v1.s''h'
pkill -9 -f "$T"; pkill -9 -f "$C"
sleep 3
log "killed lingering broken training (if any)"

# 1. assemble pristine base weights from ECS shards (node1->ECS is fast in-region)
mkdir -p "$W/r4base_std"
if [ ! -s "$W/r4base_std/pytorch_model.bin" ]; then
  log "assembling base from ECS shards (b200_00..09)"
  rm -f "$W/r4base_std"/b200_0*
  for i in 00 01 02 03 04 05 06 07 08 09; do
    exp=95000000; [ "$i" = "09" ] && exp=1674211
    for a in 1 2 3 4 5 6 7 8; do
      curl -s -m300 -o "$W/r4base_std/b200_$i" "$ECS/file/b200_$i"
      sz=$(wc -c < "$W/r4base_std/b200_$i" 2>/dev/null || echo 0)
      if [ "$sz" = "$exp" ]; then break; fi
      log "retry b200_$i a=$a got=$sz"
      sleep 3
    done
  done
  cat "$W/r4base_std"/b200_0* > "$W/r4base_std/pytorch_model.bin"
  rm -f "$W/r4base_std"/b200_0*
  tot=$(wc -c < "$W/r4base_std/pytorch_model.bin")
  log "assembled pytorch_model.bin size=$tot (expect 856674211)"
fi

# 2. copy config + tokenizer from NAS-persisted copies (these survived the /root wipe)
if [ ! -f "$W/r4base_std/config.json" ]; then
  cp "$W/ckpt/config.json" "$W/ckpt/generation_config.json" "$W/r4base_std/" 2>/dev/null
  cp "$W/base_tok"/tokenizer* "$W/base_tok"/special_tokens_map.json "$W/r4base_std/" 2>/dev/null
  log "copied config+tokenizer into r4base_std"
fi

# 3. sanity
if [ ! -f "$W/r4base_std/config.json" ] || [ ! -s "$W/r4base_std/pytorch_model.bin" ]; then
  log "ERROR: base incomplete (config or weights missing). abort."; exit 3
fi

# 4. fetch latest chain + fp32 train script from ECS
curl -s -o "$W/chain_v1.sh" "$ECS/file/node1_chain.sh"
curl -s -o "$W/train_remote_v4.py" "$ECS/file/train_remote_v4.py"
chmod +x "$W/chain_v1.sh"
log "fetched chain_v1.sh + train_remote_v4.py from ECS"

# 5. RESET + launch fp32 R4 chain in background (survives on DSW)
cd "$W" && (nohup sh chain_v1.sh RESET=1 > chain_reset.log 2>&1 < /dev/null &)
log "launched chain RESET=1 (fp32) in background"
echo "BOOTSTRAP_LAUNCHED"
