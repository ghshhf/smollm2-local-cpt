#!/bin/bash
# dream_daemon.sh — 魔搭 node1 上的「做梦自进化集群」常驻
#
# 2026-10-18 升级为「拉满」模式：
#   旧版是单进程串行(--n 4) + 60s sleep，单推理只占 1 核，CPU 长期 ~12% 没吃满。
#   新版起 DREAM_WORKERS 个 worker 并发(self-healing 各自 while true)，每个 worker
#   限 1 个 BLAS 线程(OMP/MKL=1)，合计正好压满 cgroup 硬限的 8 核。node1 现专跑自进化，
#   cgroup 把总 CPU 死死压在 8 核，ffmpeg/numpy 即便开多线程也只会被节流、不会超。
#
# 并发安全：triple_bank.py / dream_elite.py 的 conn() 已上 WAL + busy_timeout=30000，
#   8 个 worker 经子进程(add/query/trial)并发打库不会 "database is locked"。
#
# 为什么不是 cron：容器重启后 /root 全清，cron 留不住；脚本/状态放 NAS(/mnt/workspace)。
#
# 用法: DREAM_WORKERS=8 nohup bash /mnt/workspace/node1/dream/dream_daemon.sh \
#         > /mnt/workspace/node1/dream/logs/daemon.log 2>&1 &

set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
mkdir -p logs

BASE="${DREAM_BASE:-/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct}"
LORA="${DREAM_LORA:-}"          # 可选：编程 LoRA 适配器目录(含 adapter_config.json)。空=裸模型
N="${DREAM_N:-4}"
SLEEP="${DREAM_SLEEP:-5}"
WORKERS="${DREAM_WORKERS:-8}"
LORA_ARG=""
[ -n "$LORA" ] && LORA_ARG="--lora $LORA"

# 每 worker 限 1 个 BLAS 线程：WORKERS 个 worker 合计 = WORKERS 核，正好压满 8 核。
# 这些 env 会被 python 子进程继承；cgroup 再兜底节流，绝不超 8 核。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[swarm] $(date '+%F %T') start base=$BASE lora=${LORA:-none} n=$N workers=$WORKERS threads=1"

# worker 退出(含被 kill)时把它的直接子进程一起带走，避免孤儿 python 继续跑。
cleanup() {
    echo "[swarm] $(date '+%F %T') 收到退出信号，清理 worker 进程组"
    pkill -P $$ 2>/dev/null          # 杀本脚本的直接子进程(worker 函数)
    pkill -f '[d]ream_loop.py' 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup TERM INT

worker() {
    local id=$1
    # 错峰启动：避免 8 个 worker 同时从 NAS 读 954M 模型造成惊群(Thundering Herd)。
    sleep $((id * 3))
    # 错峰任务段：每个 worker 按 id 平移 offset，8×4=32 > 池长 29，保证全池(含物理世界函数)都被覆盖。
    # 否则所有 worker 都从 DREAM_BANK[0] 顺序取，永远只做前 N 条，队尾任务饿死。
    local off=$(( (id - 1) * N ))
    while true; do
        echo "[worker $id] $(date '+%F %T') begin offset=$off"
        python3 -u tools/dream_loop.py \
            --base "$BASE" \
            ${LORA_ARG:+$LORA_ARG} \
            --no-memory \
            --n "$N" \
            --offset "$off" \
            >> "logs/worker_${id}.log" 2>&1
        rc=$?
        echo "[worker $id] $(date '+%F %T') end rc=$rc sleep=${SLEEP}s"
        sleep "$SLEEP"
    done
}

# 启动 WORKERS 个 worker(各自 self-healing)
for w in $(seq 1 "$WORKERS"); do
    worker "$w" &
done

# 轻量银行快照(单独进程，不阻塞 worker；模式串带括号，pkill 不会误杀自身)
( while true; do
    sleep 120
    python3 - <<'PY' >> logs/bank_growth.log 2>&1
import sqlite3, time
try:
    c = sqlite3.connect("data/triple_bank.db", timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    n = c.execute("select count(*) from triples").fetchone()[0]
    r = c.execute("select count(*) from rejects").fetchone()[0]
    print("%s triples=%d rejects=%d" % (time.strftime("%F %T"), n, r))
except Exception as e:
    print("%s snapshot_err=%s" % (time.strftime("%F %T"), e))
PY
  done ) &
SNAP_PID=$!

wait
# wait 返回(理论上不会，因 worker 永久循环)后兜底清理快照进程
kill "$SNAP_PID" 2>/dev/null
