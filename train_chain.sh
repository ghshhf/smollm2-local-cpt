#!/usr/bin/env bash
# 排队蒸馏链：V3 -> V4 -> V5 ... 自动接训，数据驱动，幂等可重跑。
# 用法：nohup bash train_chain.sh > _train_chain.log 2>&1 < /dev/null & disown
set -u
cd "D:/work/_local_ai/train_15b" || exit 1
VE="D:/work/_local_ai/train_15b/venv/Scripts/python.exe"
BASE="model/Qwen2.5-1.5B-Instruct"
CHAINLOG="D:/work/_local_ai/train_15b/_train_chain.log"

log(){ echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >> "$CHAINLOG"; }

# 作业清单： 名称|数据|输出目录|步数
JOBS=(
  "v3|data/role_data_v3.jsonl|out_lora_v3|320"
  "v4|data/role_data_v4.jsonl|out_lora_v4|560"
  "v5|data/role_data_v5.jsonl|out_lora_v5|560"
  "v6|data/role_data_v6.jsonl|out_lora_v6|760"
)

is_running(){ tasklist.exe 2>/dev/null | grep -qi "[p]ython.exe"; }

# 等当前在跑的训练结束（用于接管外部已启动的同名作业），或判定崩溃
wait_or_crash(){
  local ckpt="$1"
  while true; do
    [ -d "$ckpt" ] && return 0          # 已完成
    if is_running; then sleep 60; else return 1; fi  # 仍在跑则等；进程没了且没 checkpoint = 崩溃
  done
}

log "===== 蒸馏链启动 ====="

for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME DATA OUT STEPS <<< "$job"
  CKPT="$OUT/checkpoint-$STEPS"
  log "--- 处理 $NAME (data=$DATA out=$OUT steps=$STEPS) ---"

  # 1) 已完成 -> 跳过（幂等）
  if [ -d "$CKPT" ]; then
    log "$NAME 已完成($CKPT 存在)，跳过"
    continue
  fi

  # 2) 数据缺失 -> 知识汇待命，停止链（放入对应数据即自动续跑）
  if [ ! -f "$DATA" ]; then
    log "$NAME 数据缺失($DATA) -> 链在此停车待命。放入该文件后重跑本脚本即可自动续训。"
    break
  fi

  # 3) 若同名训练正在跑（接管外部已启动的作业）-> 等它结束
  if is_running; then
    log "$NAME 检测到训练进程在跑，接管并等待完成..."
    if wait_or_crash "$CKPT"; then
      log "$NAME 接管完成"
      continue
    fi
    # 进程没了但没 checkpoint：当作崩溃，下面重新拉起
    log "$NAME 接管失败(进程消失且无 checkpoint)，重新拉起"
  fi

  # 4) 正式拉起本作业 —— nohup+disown 脱离链进程树，避免链被平台回收时连累训练
  log "=== 启动 $NAME 训练(nohup 脱离链) ==="
  nohup "$VE" -u train_lora.py \
    --model_path "$BASE" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps "$STEPS" \
    --lora_r 16 --lora_alpha 32 \
    --max_seq_length 1024 \
    --per_device_batch_size 1 \
    --grad_accumulation 4 \
    --learning_rate 2e-4 > "$OUT/_train.log" 2>&1 < /dev/null & disown
  vpid=$!
  log "$NAME 已启动 pid=$vpid，链轮询等待完成（链本身被回收不影响本作业）..."

  while is_running && [ ! -d "$CKPT" ]; do sleep 60; done

  if [ -d "$CKPT" ]; then
    log "$NAME 训练完成 OK (checkpoint-$STEPS 已落盘)"
  else
    log "$NAME 轮询结束但 checkpoint-$STEPS 缺失 -> 链停止，待人工排查"
    break
  fi
done

log "===== 蒸馏链本次运行结束 ====="
