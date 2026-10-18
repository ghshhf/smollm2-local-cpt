#!/bin/sh
# chain_v1.sh v3 — R4/R5/R6 分块续训链 (加固版)
# 用法: RESET=1 sh chain_v1.sh   (清废权重, 回标准 200M 重训 —— 已有链存活时拒绝执行)
#
# 加固点 (针对"双链互删 run_tmp 白跑 2h"事故):
#   1) guard()        : 单实例锁。已有链存活 -> 排队等待接管, 绝不并发跑两条链
#   2) trainer_alive(): chunk 开跑前若训练进程仍在, 死等; 绝不删它正在写的 run_tmp
#   3) salvage()      : 残留 run_tmp/ckpt 先接管 promote 再清 —— 断点重续最后保险
#   4) ckpt_ok()      : 权重 <100MB 视为截断, 拒绝 promote(防 mv 覆盖掉好权重)
#   5) maybe_upgrade(): 每个 chunk 边界从 ECS 拉新版, sh -n 校验后 exec 热升级(PID 不变)
#   6) 失败退避        : 拿不到 ckpt 睡 600s 再重试, 不空转刷日志
W=/mnt/workspace/node1/chain
R4BASE=$W/r4base_std
ECS=http://47.99.131.12:8081
LOG=$W/chain.log
LOCK=$W/chain.pid
mkdir -p "$W"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# RESET 既可作环境变量(RESET=1 sh x)也可作位置参数(sh x RESET=1)
case "$1" in RESET=*) RESET="${1#RESET=}" ;; esac

# ---------- 1) 单实例守卫 ----------
# 找 "argv[1] 是 chain_v*.sh 的 sh 进程"(排除 /bin/sh -c "...chain_v1.sh..." 这类派活命令)。
#
# ⚠️ 踩过的两个坑(dash, 都实机验证过):
#   ① $(...)/管道 fork 的子 shell 会继承父进程 cmdline "sh chain_v1.sh" -> 把"自己"当外链, 死等 4h 后退出
#   ② 事后读 /proc/<pid>/status 判祖先 -> 子进程可能已退出被 init 收养(ppid 变 1), 又误判成外链
#   ③ POSIX sh 无 local, 辅助函数里的循环变量会覆盖调用方同名变量
# => 解法: 一次 ps 快照同时取 pid/ppid, 在同一份 pid->ppid 映射里做祖先回溯, 全部由 awk 一次算完。
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
    # A) pidfile(主判据: 简单确定)
    if [ -f "$LOCK" ]; then
      OLD=$(cat "$LOCK" 2>/dev/null)
      if [ -n "$OLD" ] && [ "$OLD" != "$$" ] && kill -0 "$OLD" 2>/dev/null \
         && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q 'chain_v'; then
        OTH="$OLD"
      fi
    fi
    # B) 进程表扫描(兜底: pidfile 丢失/被删时仍能防双链)
    [ -z "$OTH" ] && OTH=$(other_chain)
    [ -z "$OTH" ] && break
    i=$((i+1))
    log "GUARD: 检测到另一条链 pid=$OTH 存活, 排队等待接管(${i}min) —— 不并发, 不抢 run_tmp"
    sleep 60
    if [ "$i" -ge 240 ]; then log "GUARD: 等了 4h 仍未接管, 本实例退出(原链继续跑, 无损)"; exit 0; fi
  done
  echo $$ > "$LOCK"
  # 接管: 清掉上一次人工/迁移用的 stop 标记, 否则新链跑完一个 chunk 就自杀
  if [ -f "$W/stop" ]; then rm -f "$W/stop"; log "GUARD: 接管完成, 已清除遗留 stop 标记"; fi
}

# ---------- 4) ckpt 完整性 ----------
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

# ---------- 3) 残留 ckpt 接管 ----------
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

# ---------- 2) 训练进程探测 ----------
trainer_alive() {
  ps -eo args 2>/dev/null | grep -v grep | grep -q "train_remote_v4.py"
}

# ---------- 5) 热升级 ----------
maybe_upgrade() {
  case "$0" in *chain_v1.sh) ;; *) return 0 ;; esac
  curl -s -m 60 -o "$W/.chain_new" "$ECS/file/node1_chain.sh" 2>/dev/null
  if [ ! -s "$W/.chain_new" ] || [ "$(stat -c %s "$W/.chain_new" 2>/dev/null || echo 0)" -lt 1000 ]; then
    rm -f "$W/.chain_new"; return 0
  fi
  if cmp -s "$W/.chain_new" "$0"; then rm -f "$W/.chain_new"; return 0; fi
  if ! sh -n "$W/.chain_new" 2>/dev/null; then
    log "UPGRADE: 新脚本语法校验失败, 保持当前版本"; rm -f "$W/.chain_new"; return 0
  fi
  log "UPGRADE: 热升级 chain_v1.sh (pid $$ 保留, 当前 chunk 边界生效)"
  mv "$W/.chain_new" "$0" || { rm -f "$W/.chain_new"; return 0; }
  exec sh "$0"
}

guard

# 训练脚本(从 ECS /file/ 拉 v4 fp32 修复版)
if [ ! -f "$W/train_remote_v4.py" ]; then
  curl -s -o "$W/train_remote_v4.py" "$ECS/file/train_remote_v4.py"
  log "fetched train_remote_v4.py from ECS"
fi

# ---- RESET: 清废权重, 回到标准 SmolLM2-200M 基座 ----
if [ "${RESET:-0}" = "1" ]; then
  rm -rf "$W/ckpt" "$W/ckpt_old" /root/c_cache_*.npy "$W"/cache_*.npy
  rm -f "$W/done_steps.txt"
  if [ ! -f "$R4BASE/pytorch_model.bin" ]; then
    log "RESET ERROR: r4base missing, cannot init W/ckpt"; exit 2
  fi
  mkdir -p "$W/ckpt"
  cp "$R4BASE"/pytorch_model.bin "$R4BASE"/model.safetensors "$R4BASE"/config.json "$R4BASE"/generation_config.json \
     "$R4BASE"/tokenizer* "$R4BASE"/special_tokens_map.json "$W/ckpt/" 2>/dev/null
  echo 0 > "$W/ckpt/step.txt"; echo 0 > "$W/done_steps.txt"
  log "RESET done: W/ckpt re-init from standard 200M base, done=0"
fi

# ---- 初始化(非 RESET 且 W/ckpt 缺失): 从标准基座建 W/ckpt ----
if [ ! -f "$W/done_steps.txt" ]; then
  echo 0 > "$W/done_steps.txt"
  if [ ! -f "$W/ckpt/pytorch_model.bin" ]; then
    mkdir -p "$W/ckpt"
    cp "$R4BASE"/pytorch_model.bin "$R4BASE"/model.safetensors "$R4BASE"/config.json "$R4BASE"/generation_config.json \
       "$R4BASE"/tokenizer* "$R4BASE"/special_tokens_map.json "$W/ckpt/" 2>/dev/null
    echo 0 > "$W/ckpt/step.txt"; log "init ckpt from r4base"
  fi
fi
# base_tok 常驻 NAS(持久)
if [ ! -f "$W/base_tok/tokenizer.json" ] && [ -f "$W/ckpt/tokenizer.json" ]; then
  mkdir -p "$W/base_tok"; cp "$W/ckpt"/tokenizer* "$W/ckpt"/special_tokens_map.json "$W/base_tok/" 2>/dev/null
  log "base_tok saved to NAS"
fi

done_n() { cat "$W/done_steps.txt"; }
heartbeat() { echo "$(date '+%m-%d %H:%M:%S') pid=$$ done=$(done_n) stage=$1" > "$W/heartbeat"; }

stage_corpus() {
  STAGE=$1
  CDIR="$W/corpus_$STAGE"
  low=$(echo "$STAGE" | tr 'A-Z' 'a-z')
  if [ ! -f "$CDIR/zh_text_$STAGE.jsonl" ]; then
    # 若 CDIR 是非语料目录(空/残缺), 清掉以便建软链
    if [ -e "$CDIR" ]; then rm -rf "$CDIR"; fi
    # 大小写不敏感回退: NAS 实际为 corpus_r4/zh_text_r4.jsonl
    if [ -f "$W/corpus_$low/zh_text_$low.jsonl" ] && [ "$low" != "$STAGE" ]; then
      ln -sfn "$W/corpus_$low" "$CDIR" 2>/dev/null
      if [ ! -f "$CDIR/zh_text_$STAGE.jsonl" ]; then
        ln -sfn "$W/corpus_$low/zh_text_$low.jsonl" "$CDIR/zh_text_$STAGE.jsonl" 2>/dev/null \
          || cp "$W/corpus_$low/zh_text_$low.jsonl" "$CDIR/zh_text_$STAGE.jsonl" 2>/dev/null
      fi
      log "corpus $STAGE resolved via case-insensitive fallback -> corpus_$low"
    elif [ -f "/root/r4work/corpus/zh_text_$STAGE.jsonl" ]; then
      mkdir -p "$CDIR"; cp "/root/r4work/corpus/zh_text_$STAGE.jsonl" "$CDIR/"
      log "corpus $STAGE staged from /root"
    else
      log "corpus $STAGE ERROR: missing"; exit 3
    fi
  fi
}

run_chunk() {
  STAGE=$1; N=$2; LR=$3; CORPUS="$W/corpus_$STAGE"
  maybe_upgrade          # 热升级(安全点: chunk 边界, 训练器未启动)
  heartbeat "$STAGE"
  S=$(done_n)
  log "chunk $STAGE start: done=$S chunk=$N lr=$LR"

  # 防线 A: 上一条链的训练器若还在跑, 死等它写完 —— 绝不带走它的 run_tmp
  i=0
  while trainer_alive; do
    i=$((i+1))
    log "WAIT: 检测到训练进程仍在运行(${i}min), 不删 run_tmp, 继续等"
    sleep 60
  done
  # 防线 B: 残留 ckpt 先接管 promote(断点重续), 剩下的才是垃圾
  salvage
  rm -rf "$W/run_tmp"

  # cache: /root 本地盘跑(随机读快), NAS 留底
  if [ ! -f "/root/c_cache_$STAGE.npy" ] && [ -f "$W/cache_$STAGE.npy" ]; then
    cp "$W/cache_$STAGE.npy" "/root/c_cache_$STAGE.npy"
  fi
  BN=/root/r4work/base_next
  rm -rf "$BN"; mkdir -p "$BN"
  cp "$W/base_tok"/tokenizer* "$W/base_tok"/special_tokens_map.json "$BN/"
  cp "$W/ckpt/config.json" "$W/ckpt/generation_config.json" "$BN/"
  cp "$W/ckpt/pytorch_model.bin" "$W/ckpt/model.safetensors" "$BN/" 2>/dev/null
  # fp32 训练(v4, 去 bf16 autocast — 这是修复的核心)
  # -u 必须加: stdout 重定向到文件时 Python 默认块缓冲(4KB), 崩溃 traceback
  # 会压在缓冲区里随进程消失 -> 日志看起来"戛然而止"却查不出原因(07:09 踩坑)。
  # --threads 8: 关键! 214M 小模型 + max_len=512 时 GEMM 太小, 64 线程纯同步开销互抢
  # 内存带宽, 反而比 8 线程慢 ~4 倍。train_remote_v4.py 自带实测: 8线程=18.7s/步(219 tok/s),
  # 64线程=>75s/步。用户要的"过去极限速度"正是 8 线程跑出来的, 必须显式锁 8。
  # (之前不传、吃 64 线程亲和力, 是"回到过去"没真正生效的根因)
  python3 -u "$W/train_remote_v4.py" --base_dir "$BN" --corpus_dir "$CORPUS" \
    --run_dir "$W/run_tmp" --cache "/root/c_cache_$STAGE.npy" \
    --steps "$N" --lr "$LR" --warmup 50 --threads 8 --upload_url "${ECS}/upload" >> "$LOG" 2>&1

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

# ---- R4 段 (语言基础) 预算 1200 (fp32, lr 5e-4) ----
stage_corpus R4
while [ "$(done_n)" -lt 1200 ]; do
  run_chunk R4 400 5e-4
done
log "=== R4 complete @ $(done_n) ==="

# ---- R5 段 (知识教育) 预算 2800 (lr 3e-4) ----
stage_corpus R5
while [ "$(done_n)" -lt 2800 ]; do
  run_chunk R5 400 3e-4
done
log "=== R5 complete @ $(done_n) ==="

# ---- R6 段 (全量知识封顶, 0.2B 级知识天花板) 预算 4000 (lr 2e-4) ----
# R6 = 合并 R4/R5 已学广度 + 全量百科/古籍/科普/文言, 把 0.2B 能装的知识压实到平台期
# 跑满预算后停, 等本地 holdout CE 验收; 若 plateau -> 触发 Net2Net 0.3B 扩容
# corpus: NAS corpus_r6/zh_text_r6.jsonl (本地 build_r6_reserve.py 产出, 经 ECS 中继上 NAS)
stage_corpus R6
while [ "$(done_n)" -lt 6800 ]; do
  run_chunk R6 400 2e-4
done
log "=== R6 complete @ $(done_n) — 0.2B 知识封顶 done (等本地 holdout CE 验收 + Net2Net 0.3B 扩容决策) ==="

# ---- 0.3B 阶段接管: R6 封顶(done>=6800)后, 从 ECS 拉最新 chain_v2.sh 续跑 ----
# (Net2Net 扩容 0.2B->0.3B + 四分类储备课程 R7/R8/R9/R10)
# 未就绪(ECS 无 chain_v2.sh)则等待下次轮询, 绝不提前打断当前训练。
if [ "$(done_n)" -ge 6800 ]; then
  curl -s -o "$W/chain_v2.sh.new" "$ECS/file/chain_v2.sh"
  if [ -s "$W/chain_v2.sh.new" ] && sh -n "$W/chain_v2.sh.new" 2>/dev/null; then
    mv "$W/chain_v2.sh.new" "$W/chain_v2.sh"
    log "=== R6 done @ $(done_n), exec 转交 chain_v2.sh (0.3B 阶段, pid 保留) ==="
    exec sh "$W/chain_v2.sh"
  else
    rm -f "$W/chain_v2.sh.new"
    log "R6 done 但 chain_v2.sh 未就绪/语法错, 等待 staging(下次轮询接管)"
  fi
fi
