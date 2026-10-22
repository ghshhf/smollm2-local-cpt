#!/bin/sh
# start_align.sh —— 在 node1 上后台起 align 训练，立即返回（不堵 polaris_node 的命令循环）
#
# 为什么单独成脚本：上一版把 setsid nohup 直接内联在 board 命令里，
# 那条 bash 迟迟不退出，把 polaris_node 的单线程命令循环堵了 2 分钟，
# 后面排队的命令直接「超时未领取」。脚本自己 exit，就不会连累别人。
#
# 用法： sh start_align.sh <llm路径> <steps> <threads> <日志名> [额外参数...]
set -u
cd /mnt/workspace/node1 || exit 1
LLM="${1:-chain/r4base_std}"
STEPS="${2:-800}"
THREADS="${3:-3}"
LOG="${4:-logs/audio_align.log}"
if [ $# -ge 4 ]; then shift 4; else shift $#; fi

mkdir -p logs
PIDF="logs/$(basename "$LOG" .log).pid"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDF") log=$LOG"
  exit 0
fi

setsid nohup /usr/local/bin/python3 -u tools/audio_attach.py \
  --stage align --use-cache --steps "$STEPS" \
  --val-speakers S0018,S0019 --eval-n 10 --threads "$THREADS" \
  --llm "$LLM" --whisper models/whisper-small \
  --save "/tmp/audio_$(basename "$LOG" .log)" "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") llm=$LLM steps=$STEPS threads=$THREADS log=$LOG"
sleep 4
tail -5 "$LOG" 2>/dev/null
exit 0
