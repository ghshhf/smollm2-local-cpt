#!/usr/bin/env bash
# train_r4.sh — 魔搭节点端：从本机公网隧道拉取 R4(214M) 权重/语料并续训
# 用法: bash train_r4.sh <public-tunnel-url>
set -u
BASE="${1:-}"
ECS="http://47.99.131.12:8081"
NODE="node1"
WORKDIR="/mnt/workspace/node1"
PY="$(command -v python3 || command -v python)"
EXP_MODEL=856674211   # pytorch_model.bin 期望字节数(本机校验)

[ -z "$BASE" ] && { echo "usage: $0 <public-tunnel-url>"; exit 2; }
BASE="${BASE%/}"

mail() {
  printf '%s' "$1" | "$PY" -c "import sys,json,urllib.request;m=sys.stdin.read();req=urllib.request.Request('$ECS/mail',data=json.dumps({'from':'$NODE','text':m}).encode(),headers={'Content-Type':'application/json'},method='POST');urllib.request.urlopen(req,timeout=20)" >/dev/null 2>&1 || true
}

echo "=== dispatch start base=$BASE ==="
mail "dispatch start base=$BASE"

cd "$WORKDIR" || { mail "NO WORKDIR $WORKDIR"; exit 3; }
mkdir -p r4_base corpus r4_run

# 防止重复启动训练(同节点可能收到重复任务)
if [ -f r4_run/train.pid ] && kill -0 "$(cat r4_run/train.pid 2>/dev/null)" 2>/dev/null; then
  mail "ALREADY_TRAINING pid=$(cat r4_run/train.pid) skip"
  echo "already training, skip"; exit 0
fi

# 预检环境
if "$PY" -c "import torch,transformers" 2>/dev/null; then
  VER=$("$PY" -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)" 2>/dev/null)
  mail "ENV OK $VER"
else
  mail "ENV MISSING torch/transformers — training will fail"
fi

# 确保训练脚本仓库(SSH deploy key)
if [ ! -f gh_repo/node/train_remote_v2.py ]; then
  export GIT_SSH_COMMAND='ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
  git clone git@github.com:ghshhf/smollm2-local-cpt.git gh_repo 2>&1 | tail -2 | while read l; do mail "CLONE $l"; done
fi
( cd gh_repo && git pull --ff-only 2>&1 | tail -2 ) | while read l; do mail "PULL $l"; done

# 下载工具：aria2c > wget > curl
dl() {
  local url="$1" out="$2"
  [ -s "$out" ] && { echo "skip $(basename "$out")"; return 0; }
  mkdir -p "$(dirname "$out")"
  local d="$(dirname "$out")" f="$(basename "$out")" i
  for i in 1 2 3 4 5 6; do
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -x16 -s16 -k1M --allow-overwrite=true -d "$d" -o "$f" "$url" && [ -s "$out" ] && return 0
    elif command -v wget >/dev/null 2>&1; then
      wget -c -q -O "$out" "$url" && [ -s "$out" ] && return 0
    else
      curl -C - -L -s -o "$out" "$url" && [ -s "$out" ] && return 0
    fi
    echo "  retry $i for $f"; sleep 8
  done
  echo "FAIL $url"; return 1
}

# config + tokenizer
for f in config.json tokenizer.json tokenizer_config.json generation_config.json; do
  dl "$BASE/r4_config/$f" "r4_base/$f" || mail "WARN missing config $f"
done
# R4 训练权重(不要用原版 model.safetensors)
dl "$BASE/r4_weights/pytorch_model.bin" "r4_base/pytorch_model.bin"
dl "$BASE/r4_weights/step.txt" "r4_base/step_src.txt" || true

# 校验权重完整性(R4 续训不能容忍损坏权重)
GOT=$("$PY" -c "import os;print(os.path.getsize('r4_base/pytorch_model.bin'))" 2>/dev/null || echo 0)
if [ "$GOT" != "$EXP_MODEL" ]; then
  mail "MODEL SIZE MISMATCH got=$GOT exp=$EXP_MODEL ABORT"
  echo "model size mismatch"; exit 4
fi
echo "model ok size=$GOT"
mail "model ok size=$GOT"

# corpus
mapfile -t CORPUS < <(curl -s -m 30 "$BASE/corpus_list" | grep -v '^$')
echo "corpus files: ${#CORPUS[@]}"
for f in "${CORPUS[@]}"; do
  dl "$BASE/corpus/$f" "corpus/$f" || mail "WARN corpus $f failed"
done
mail "download done base=$(ls r4_base | wc -l) corpus=$(du -sh corpus 2>/dev/null | cut -f1)"

# 启动训练(后台, 脱离 node 代理, 抗重启)
cd "$WORKDIR"
nohup "$PY" gh_repo/node/train_remote_v2.py \
  --base_dir r4_base --corpus_dir corpus --run_dir r4_run \
  --steps 400 --lr 5e-4 --max_len 512 --accum 8 \
  --upload_url "$BASE/upload" \
  > r4_run/train.log 2>&1 &
TRPID=$!
echo "$TRPID" > r4_run/train.pid
echo "training pid=$TRPID"
mail "training LAUNCHED pid=$TRPID log=r4_run/train.log"

# 进度镜像(升级信箱通道): 每 60s POST 训练尾部到 ECS /mail
( while true; do
    sleep 60
    if ! kill -0 "$TRPID" 2>/dev/null; then
      mail "TRAINING ENDED $(tail -4 r4_run/train.log | tr '\n' ' ')"
      break
    fi
    mail "PROGRESS $(tail -2 r4_run/train.log | tr '\n' ' ')"
  done ) &
MONPID=$!
echo "$MONPID" > r4_run/monitor.pid
mail "dispatch DONE pid=$TRPID monitor=$MONPID"
echo "=== dispatch done ==="
