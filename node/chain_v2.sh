#!/bin/sh
# chain_v2.sh v2 — 0.3B 阶段: Net2Net 扩容(0.2B->0.3B) + 四分类储备课程
# 由 chain_v1.sh 在 R6 封顶(done>=6800)后 exec 接管(pid 保留, 不并发)。
# 课程: R7 知识 / R8 古籍 / R9 小说 / R10 公众号, 每个学满 budget 再进下一个。
# 加固点与 chain_v1.sh v3 完全一致(单实例/不等删 run_tmp/残留接管/权重校验/热升级)。
W=/mnt/workspace/node1/chain
ECS=http://47.99.131.12:8081
LOG=$W/chain_v2.log
LOCK=$W/chain.pid
mkdir -p "$W"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

done_n() { cat "$W/done_steps.txt"; }
heartbeat() { echo "$(date '+%m-%d %H:%M:%S') pid=$$ done=$(done_n) stage=$1" > "$W/heartbeat"; }

# ---------- 1) 单实例守卫 ----------
# 注: chain_v1 用 exec 交棒过来时 PID 相同, 守卫直接放行; 只有"另有活链"才排队等待。
# 同 chain_v1.sh: 一次 ps 快照取 pid/ppid, 在同一份映射里做祖先回溯剔除自身后代(详见 v1 注释里的三个坑)。
other_chain() {
  ps -eo pid,ppid,cmd 2>/dev/null | awk -v self="$$" '
    ($3=="sh"||$3=="dash"||$3=="bash"||$3=="/bin/sh"||$3=="/bin/bash") && $4 ~ /chain_v[0-9]+\.sh$/ {
      if ($1+0 != self+0) key[$1+0] = $2+0
    }
    END {
      sep = ""
      for (p in key) {
        q = p; mine = 0; n = 0
        while (n < 25) {
          if (q+0 == self+0) { mine = 1; break }
          if (q in key) q = key[q]; else break
          n++
        }
        if (!mine) { printf "%s%s", sep, p; sep = " " }
      }
    }'
}

guard() {
  i=0
  while :; do
    OTH=""
    if [ -f "$LOCK" ]; then
      OLD=$(cat "$LOCK" 2>/dev/null)
      if [ -n "$OLD" ] && [ "$OLD" != "$$" ] && kill -0 "$OLD" 2>/dev/null \
         && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q 'chain_v'; then
        OTH="$OLD"
      fi
    fi
    [ -z "$OTH" ] && OTH=$(other_chain)
    [ -z "$OTH" ] && break
    i=$((i+1))
    log "GUARD: 检测到另一条链 pid=$OTH 存活, 排队等待接管(${i}min) —— 不并发, 不抢 run_tmp"
    sleep 60
    if [ "$i" -ge 240 ]; then log "GUARD: 等了 4h 仍未接管, 本实例退出(原链继续跑, 无损)"; exit 0; fi
  done
  echo $$ > "$LOCK"
  if [ -f "$W/stop" ]; then rm -f "$W/stop"; log "GUARD: 接管完成, 已清除遗留 stop 标记"; fi
}

ckpt_ok() {
  D=$1
  [ -f "$D/step.txt" ] || return 1
  for f in "$D/model.safetensors" "$D/pytorch_model.bin"; do
    if [ -f "$f" ]; then
      sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
      [ "$sz" -gt 100000000 ] && return 0
    fi
  done
  return 1
}

salvage() {
  RD="$W/run_tmp/checkpoint-latest"
  if ckpt_ok "$RD"; then
    GOT=$(cat "$RD/step.txt" 2>/dev/null || echo 0)
    S=$(done_n)
    rm -rf "$W/ckpt_old"; mv "$W/ckpt" "$W/ckpt_old" 2>/dev/null
    mv "$RD" "$W/ckpt"
    echo $((S + GOT)) > "$W/done_steps.txt"
    log "SALVAGE: 接管上一轮未 promote 的 ckpt (+$GOT) -> done=$(done_n)"
  fi
}

trainer_alive() {
  ps -eo args 2>/dev/null | grep -v grep | grep -q "train_remote_v4.py"
}

maybe_upgrade() {
  case "$0" in *chain_v2.sh) ;; *) return 0 ;; esac
  curl -s -m 60 -o "$W/.chain2_new" "$ECS/file/chain_v2.sh" 2>/dev/null
  if [ ! -s "$W/.chain2_new" ] || [ "$(stat -c %s "$W/.chain2_new" 2>/dev/null || echo 0)" -lt 1000 ]; then
    rm -f "$W/.chain2_new"; return 0
  fi
  if cmp -s "$W/.chain2_new" "$0"; then rm -f "$W/.chain2_new"; return 0; fi
  if ! sh -n "$W/.chain2_new" 2>/dev/null; then
    log "UPGRADE: 新脚本语法校验失败, 保持当前版本"; rm -f "$W/.chain2_new"; return 0
  fi
  log "UPGRADE: 热升级 chain_v2.sh (pid $$ 保留, 当前 chunk 边界生效)"
  mv "$W/.chain2_new" "$0" || { rm -f "$W/.chain2_new"; return 0; }
  exec sh "$0"
}

guard

# ---- 0) 一次性 Net2Net 扩容(若仍是 0.2B config) ----
OLD_INTER=$(python3 -c "import json;print(json.load(open('$W/ckpt/config.json'))['intermediate_size'])" 2>/dev/null || echo 0)
if [ "$OLD_INTER" -lt 4608 ]; then
  log "=== Net2Net 0.3B 扩容触发 (inter=$OLD_INTER) ==="
  if [ ! -f "$W/expand_300m_node1.py" ]; then
    curl -s -o "$W/expand_300m_node1.py" "$ECS/file/expand_300m_node1.py"
    log "fetched expand_300m_node1.py"
  fi
  python3 "$W/expand_300m_node1.py" "$W/ckpt" >> "$LOG" 2>&1
  RC=$?
  if [ "$RC" -ne 0 ]; then
    log "FATAL: expand 失败 (rc=$RC), 停链"; exit 1
  fi
  rm -f "$W/ckpt/pytorch_model.bin"
  log "=== 扩容完成, 0.3B config 生效 (inter=4608 layers=36) ==="
else
  log "已是 0.3B config (inter=$OLD_INTER), 跳过扩容"
fi

# ---- 储备语料分期软链(corpus_$STAGE/zh_text_$STAGE.jsonl) ----
stage_reserve() {
  STAGE=$1; SRC=$2
  CDIR="$W/corpus_$STAGE"
  if [ ! -f "$CDIR/zh_text_$STAGE.jsonl" ]; then
    [ -e "$CDIR" ] && rm -rf "$CDIR"
    mkdir -p "$CDIR"
    ln -sfn "$W/corpus_reserve/$SRC" "$CDIR/zh_text_$STAGE.jsonl" 2>/dev/null \
      || cp "$W/corpus_reserve/$SRC" "$CDIR/zh_text_$STAGE.jsonl"
    log "stage_reserve $STAGE -> corpus_reserve/$SRC"
  fi
}

# ---- 续训 chunk(与 chain_v1 run_chunk 同语义, 仅读 0.3B 的 safetensors) ----
run_chunk() {
  STAGE=$1; N=$2; LR=$3; CORPUS="$W/corpus_$STAGE"
  maybe_upgrade
  heartbeat "$STAGE"
  S=$(done_n)
  log "chunk $STAGE start: done=$S chunk=$N lr=$LR"

  i=0
  while trainer_alive; do
    i=$((i+1))
    log "WAIT: 检测到训练进程仍在运行(${i}min), 不删 run_tmp, 继续等"
    sleep 60
  done
  salvage
  rm -rf "$W/run_tmp"

  if [ ! -f "/root/c_cache_$STAGE.npy" ] && [ -f "$W/cache_$STAGE.npy" ]; then
    cp "$W/cache_$STAGE.npy" "/root/c_cache_$STAGE.npy"
  fi
  BN=/root/r4work/base_next
  rm -rf "$BN"; mkdir -p "$BN"
  cp "$W/base_tok"/tokenizer* "$W/base_tok"/special_tokens_map.json "$BN/"
  cp "$W/ckpt/config.json" "$W/ckpt/generation_config.json" "$BN/" 2>/dev/null
  cp "$W/ckpt/model.safetensors" "$BN/" 2>/dev/null
  python3 "$W/train_remote_v4.py" --base_dir "$BN" --corpus_dir "$CORPUS" \
    --run_dir "$W/run_tmp" --cache "/root/c_cache_$STAGE.npy" \
    --steps "$N" --lr "$LR" --warmup 50 --upload_url "${ECS}/upload" >> "$LOG" 2>&1

  RD="$W/run_tmp/checkpoint-latest"
  GOT=$(cat "$RD/step.txt" 2>/dev/null || echo 0)
  if [ "$GOT" -gt 0 ] && ckpt_ok "$RD"; then
    rm -rf "$W/ckpt_old"; mv "$W/ckpt" "$W/ckpt_old" 2>/dev/null
    mv "$RD" "$W/ckpt"
    echo $((S + GOT)) > "$W/done_steps.txt"
    log "chunk $STAGE end: promoted (+$GOT) -> done=$(done_n)"
  else
    log "chunk $STAGE end: NO ckpt (got=$GOT), keep old; 退避 600s 后重试"
    sleep 600
  fi
  [ -f "$W/cache_$STAGE.npy" ] || cp "/root/c_cache_$STAGE.npy" "$W/cache_$STAGE.npy" 2>/dev/null
  [ -f "$W/stop" ] && { log "stop flag -> halt"; exit 0; }
}

# ---- 0.3B 课程(四分类储备, 与原 R 段同构: 每阶段学满 budget 再进化) ----
# R7 知识 backbone: 储备知识再压实
stage_reserve R7 zh_r_knowledge.jsonl
while [ "$(done_n)" -lt 8300 ]; do
  run_chunk R7 400 3e-4
done
log "=== R7 complete @ $(done_n) — 0.3B 知识 backbone ==="

# R8 古籍
stage_reserve R8 zh_r_ancient.jsonl
while [ "$(done_n)" -lt 9800 ]; do
  run_chunk R8 400 3e-4
done
log "=== R8 complete @ $(done_n) — 0.3B 古籍 ==="

# R9 小说
stage_reserve R9 zh_r_novels.jsonl
while [ "$(done_n)" -lt 11300 ]; do
  run_chunk R9 400 3e-4
done
log "=== R9 complete @ $(done_n) — 0.3B 小说 ==="

# R10 公众号(当代语料, 收尾)
stage_reserve R10 zh_r_weixin.jsonl
while [ "$(done_n)" -lt 12800 ]; do
  run_chunk R10 400 3e-4
done
log "=== R10 complete @ $(done_n) — 0.3B 四分类储备课程封顶; 等本地 holdout CE 验收 + 下一档扩容决策 ==="
