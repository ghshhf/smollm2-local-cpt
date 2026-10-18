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
# 产粮温度/任务池均为「热文件」，worker 每轮循环重读：
#   data/dream_temp.txt  一行一个温度(0=贪心确定性；0.5=产粮多样性)
#   data/dream_pool.json 任务池(改种子=改这个文件)
# 改文件 => 下轮 worker 自动生效，零停集群零重启(2026-10-18 热更新化)。
# A/B 评测走 dream_loop 默认 temp=0 贪心，不读热文件，两臂确定性不受影响。
# 内置池大小仅作热文件缺失时的兜底。
POOL_DEFAULT=$(python3 -c "import sys;sys.path.insert(0,'tools');import dream_loop;print(len(dream_loop.DREAM_BANK))" 2>/dev/null || echo 69)
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
    while true; do
        # ── 热更新三件套：每轮重读，改文件即生效，永不停集群 ──────────
        # ① 温度热文件(缺省/非法值回退 0.5)
        local T
        T=$(cat data/dream_temp.txt 2>/dev/null)
        case "$T" in ''|*[!0-9.]*) T=0.5 ;; esac
        # ② 任务池热文件(JSON 在则热挂，条数=其中 "task" 键数；md5 前缀进日志便于远程核对)
        local poolf=data/dream_pool.json pl=0 poolargs="" pooln="$POOL_DEFAULT" pf="-"
        if [ -f "$poolf" ]; then
            pl=$(grep -c '"task"' "$poolf" 2>/dev/null) || pl=0
            [ -z "$pl" ] && pl=0
            if [ "$pl" -gt 0 ]; then
                poolargs="--pool $poolf"
                pooln="$pl"
                pf=$(md5sum "$poolf" | cut -c1-8)
            fi
        fi
        # ③ offset 每轮随机轮换：固定 offset 会把每个 worker 钉死在 32 条切片上
        #   (实测：池扩到 69 后队尾 37 条含物理种子永远饿死；且贪心生成下同一
        #   任务反复产逐字节相同的代码，银行 441 行只有 17 个唯一定码)。
        #   RANDOM%pooln + dream_loop 内部取模 => 全池覆盖。
        local off=$(( RANDOM % pooln ))
        echo "[worker $id] $(date '+%F %T') begin offset=$off temp=$T pool=$pooln/$pf"
        python3 -u tools/dream_loop.py \
            --base "$BASE" \
            ${LORA_ARG:+$LORA_ARG} \
            --no-memory \
            --n "$N" \
            --offset "$off" \
            --temp "$T" \
            ${poolargs:+$poolargs} \
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
