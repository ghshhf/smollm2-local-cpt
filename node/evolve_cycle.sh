#!/bin/bash
# evolve_cycle.sh — node1 自进化闭环 · 单周期编排（无人在场，自己拍板）
#
# 流程: 闸门(新增样本够?) → 停集群让核 → evolve_train.py 自训 LoRA
#       → A/B(同 offset 轮转评测集, 贪心解码, 两臂公平) → 裁决:
#         lora 通过数 > 裸模 → ADOPT: DREAM_LORA 热切换重启
#         否则              → REJECT: 适配器归档, 裸模重启
#       → 更新 .evolve_state.json + logs/evolve_history.log + 报告落 ECS
#
# 🔴 一切判断看数据不看感觉；失败路径必须把集群复活(宁可不进化不能停机)。
set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
B="http://47.99.131.12:8081/file"
K="?key=k0u4mail"
BASE=/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct
LORA=$D/models/evolve_lora
THRESH=20            # 银行新增多少条才值得训一轮
STATE=$D/.evolve_state.json
HIST=$D/logs/evolve_history.log
R=/tmp/evolve_report.txt
: > "$R"

# 互斥锁：上周期没跑完不许叠
if ! mkdir /tmp/evolve.lock 2>/dev/null; then
  echo "[cycle] 上周期仍在跑，跳过"; exit 0
fi
trap 'rmdir /tmp/evolve.lock 2>/dev/null' EXIT

# ── 工具函数 ────────────────────────────────────────────────────────
stop_daemon() {
  pkill -f '[d]ream_daemon.sh' 2>/dev/null; sleep 1
  pkill -f '[d]ream_loop.py' 2>/dev/null; sleep 3
  echo "remaining=$(ps -eo args | grep -cE '[d]ream_loop|[d]ream_daemon')"
}
start_daemon() {  # $1=none|lora_path
  if [ -n "${1:-}" ] && [ "$1" != "none" ]; then
    export DREAM_LORA="$1"
  else
    unset DREAM_LORA 2>/dev/null || true
  fi
  export DREAM_WORKERS=8
  nohup bash dream_daemon.sh > logs/daemon.log 2>&1 &
  DPID=$!
  sleep 5
  echo "[daemon] relaunched pid=$DPID mode=${1:-none}"
}
bank_count() {
  python3 -c "import sqlite3;c=sqlite3.connect('data/triple_bank.db',timeout=30);c.execute('PRAGMA busy_timeout=30000');print(c.execute('select count(*) from triples').fetchone()[0])"
}

# ── 1) 状态 + 闸门 ─────────────────────────────────────────────────
if [ -f "$STATE" ]; then
  CYCLE=$(python3 -c "import json;print(json.load(open('$STATE')).get('cycle',0))")
  LAST=$(python3 -c "import json;print(json.load(open('$STATE')).get('last_bank',0))")
  ADOPTED=$(python3 -c "import json;print('1' if json.load(open('$STATE')).get('adopted') else '0')")
else
  CYCLE=0; LAST=0; ADOPTED=0
  # 首跑基线=当前银行量，避免把存量当增量
  LAST=$(bank_count)
  python3 -c "import json;json.dump({'cycle':0,'last_bank':$LAST,'adopted':False,'lora_path':None,'history':[]},open('$STATE','w'))"
fi
NOW=$(bank_count)
NEW=$((NOW - LAST))
UNIQ=$(python3 -c "import sqlite3;c=sqlite3.connect('data/triple_bank.db',timeout=30);c.execute('PRAGMA busy_timeout=30000');print(c.execute('select count(distinct code_text) from triples where code_text is not null').fetchone()[0])")
echo "[gate] cycle=$CYCLE bank=$NOW last=$LAST new=$NEW uniq=$UNIQ thresh=$THRESH adopted=$ADOPTED" | tee -a "$R"

# 🔴 双闸门：银行新增≥THRESH 且 唯一样本≥UNIQ_MIN(20)。
# 教训：8 worker 同批题产出高度趋同，253 条入库唯一定码仅 15——只看 bank 增量
# 会每周期"停集群→唯一样本不够→空转失败"，白折腾集群。
if [ "$NEW" -lt "$THRESH" ] || [ "$UNIQ" -lt 20 ]; then
  echo "[gate] 条件不足(new<$THRESH 或 uniq<20)，不训。"
  # 保底：闸门不过也要确保集群活着（宁可不进化，不能停机）
  if ! pgrep -f '[d]ream_daemon.sh' > /dev/null; then
    if [ "$ADOPTED" = "1" ] && [ -d "$LORA" ]; then
      start_daemon "$LORA" | tee -a "$R"
    else
      start_daemon none | tee -a "$R"
    fi
  fi
  echo "[gate] DONE(new<$THRESH) $(date '+%F %T')" >> "$R"
  cat "$R"
  exit 0
fi

# ── 2) 停集群让核 ──────────────────────────────────────────────────
echo "[stop] 停做梦集群 $(date '+%F %T')" | tee -a "$R"
stop_daemon | tee -a "$R"

# ── 3) 自训 ────────────────────────────────────────────────────────
echo "[train] 开始自训 $(date '+%F %T')" | tee -a "$R"
python3 -u evolve_train.py --out models/evolve_lora > logs/evolve_train.log 2>&1
TRC=$?
echo "train_rc=$TRC" | tee -a "$R"
tail -3 logs/evolve_train.log | sed 's/^/  [train] /' | tee -a "$R"
if [ $TRC -ne 0 ]; then
  echo "[train] 训练未成(rc=$TRC)，裸模复活集群" | tee -a "$R"
  start_daemon none | tee -a "$R"
  echo "$(date '+%F %T') cycle=$CYCLE VERDICT=train_fail rc=$TRC bank=$NOW" >> "$HIST"
  python3 -c "import json;s=json.load(open('$STATE'));s['cycle']=$CYCLE+1;s['last_bank']=$(bank_count);json.dump(s,open('$STATE','w'))"
  echo "[cycle] DONE(train_fail) $(date '+%F %T')" >> "$R"
  cat "$R"
  exit 0
fi

# ── 4) A/B（同 offset，贪心解码；offset 随周期轮转防过拟合固定题）──
OFF=$(( (CYCLE * 8) % 18 ))
echo "[ab] offset=$OFF $(date '+%F %T')" | tee -a "$R"
python3 -u tools/dream_loop.py --base "$BASE" --no-memory --n 8 --offset $OFF > logs/ev_ab_bare.log 2>&1
BRC=$?
python3 -u tools/dream_loop.py --base "$BASE" --lora "$LORA" --no-memory --n 8 --offset $OFF > logs/ev_ab_lora.log 2>&1
LRC=$?
BNUM=$(grep -ac 'ACCEPT 入库' logs/ev_ab_bare.log)
LNUM=$(grep -ac 'ACCEPT 入库' logs/ev_ab_lora.log)
echo "bare_rc=$BRC bare_accept=$BNUM / lora_rc=$LRC lora_accept=$LNUM" | tee -a "$R"

# ── 5) 裁决 + 热切换/归档 ──────────────────────────────────────────
VERDICT=reject
if [ "$LRC" -eq 0 ] && [ "$LNUM" -gt "$BNUM" ]; then
  VERDICT=adopt
  start_daemon "$LORA" | tee -a "$R"
  echo "[verdict] ADOPT：lora $LNUM > bare $BNUM，热切换" | tee -a "$R"
else
  TS=$(date +%Y%m%d_%H%M%S)
  mv "$LORA" "${LORA}_rej_${TS}" 2>/dev/null
  start_daemon none | tee -a "$R"
  echo "[verdict] REJECT：lora $LNUM <= bare $BNUM，适配器归档" | tee -a "$R"
fi

# ── 6.5) 燃料外送：银行导出 → ECS（本地 nightly_rsi 拉取自训 polaris 代码模块）──
if [ -f tools/export_triples_to_sft.py ]; then
  if python3 tools/export_triples_to_sft.py >> "$R" 2>&1; then
    curl -s -m 60 -X POST --data-binary @data/dream_code_sft.jsonl \
      "$B/upload?name=dream_fuel.jsonl&key=k0u4mail" > /dev/null 2>&1
    echo "[fuel] exported+uploaded $(wc -l < data/dream_code_sft.jsonl) triples" | tee -a "$R"
  else
    echo "[fuel] export FAILED(不阻塞周期)" | tee -a "$R"
  fi
fi

# ── 6) 状态 + 历史 + 报告 ──────────────────────────────────────────
NB=$(bank_count)
python3 - "$STATE" "$CYCLE" "$NB" "$VERDICT" "$BNUM" "$LNUM" "$OFF" <<'PY'
import json, sys
st, cy, nb, vd, bn, ln, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7])
s = json.load(open(st)) if st and open(st) else {}
s["cycle"] = cy + 1
s["last_bank"] = nb
s["adopted"] = (vd == "adopt")
s["lora_path"] = "/mnt/workspace/node1/dream/models/evolve_lora" if vd == "adopt" else None
s.setdefault("history", []).append(
    {"cycle": cy, "verdict": vd, "bare": bn, "lora": ln, "offset": off,
     "bank": nb, "ts": __import__("time").strftime("%F %T")})
s["history"] = s["history"][-30:]
json.dump(s, open(st, "w"), ensure_ascii=False, indent=1)
PY
echo "$(date '+%F %T') cycle=$CYCLE VERDICT=$VERDICT bare=$BNUM lora=$LNUM offset=$OFF bank=$NB" >> "$HIST"

# 报告上 ECS（文件名带周期号防信箱缓存）
cp "$R" "/tmp/evolve_report_c$CYCLE.txt" 2>/dev/null
curl -s -X POST --data-binary @"$R" "$B/upload?name=evolve_report_c${CYCLE}.txt&key=k0u4mail" > /dev/null 2>&1
echo "[cycle] DONE($VERDICT) $(date '+%F %T')" | tee -a "$R"
cat "$R"
