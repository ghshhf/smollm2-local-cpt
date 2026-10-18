#!/bin/bash
# n1_ab_test.sh — node1 A/B 对照：裸 Qwen0.5B vs 挂代码 LoRA
#
# 背景：本地训完的代码 LoRA（out_lora_code_qwen）在通用代码盲测上是「格式满分/
# 内容负迁移」。但它对做梦引擎(同底座 Qwen2.5-0.5B)有没有用，只能用数据说话——
# 梦幻任务是确定性顺序(DREAM_BANK[i%18]，--n 8 固定跑同一批 8 题) +
# 贪心解码(temp=0)，所以两边同题同 prompt，输出差异只可能来自 LoRA 权重。
#
# 指标：run() 结尾的「通过(入库) X / 失败(落 rejects) Y」= 过完 vid_verify +
# task_fit 全闸门的真通过（ELITE skip 不计入，指标无污染）。
#
# 注意：A/B 期间停 daemon 让出全核，避免 exec 90s 超时造成误判；跑完不自动重启，
# 由外层按结果决定 daemon 配置（裸跑 or DREAM_LORA=...）。
set -u
D=/mnt/workspace/node1/dream
cd "$D" || exit 1
B="http://47.99.131.12:8081/file"
K="?key=k0u4mail"
R=/tmp/ab_report.txt
: > "$R"
BASE=/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct
LORA_DIR=/mnt/workspace/node1/models/code_lora_qwen

echo "[ab] === $(date '+%F %T') 停 daemon（让出全核）===" >> "$R"
pkill -f '[d]ream_daemon.sh' 2>/dev/null
sleep 1
pkill -f '[d]ream_loop.py' 2>/dev/null
sleep 3
echo "remaining=$(ps -eo args | grep -cE '[d]ream_loop|[d]ream_daemon')" >> "$R"

echo "[ab] === 从 ECS 拉 LoRA 适配器 ===" >> "$R"
mkdir -p "$LORA_DIR"
curl -s -o "$LORA_DIR/adapter_config.json" "$B/lora_adapter_config.json$K"
curl -s -o "$LORA_DIR/adapter_model.safetensors" "$B/lora_adapter_model.safetensors$K"
echo "cfg=$(wc -c < "$LORA_DIR/adapter_config.json") safetensors=$(wc -c < "$LORA_DIR/adapter_model.safetensors")" >> "$R"
python3 -c "import json;c=json.load(open('$LORA_DIR/adapter_config.json'));print('base_in_cfg=',c['base_model_name_or_path'],'r=',c['r'],'targets=',c['target_modules'])" >> "$R" 2>&1

echo "[ab] === A: 裸模型 --n 8 ===" >> "$R"
python3 -u tools/dream_loop.py --base "$BASE" --no-memory --n 8 > logs/ab_bare.log 2>&1
echo "bare_rc=$?" >> "$R"

echo "[ab] === B: 挂 LoRA --n 8 ===" >> "$R"
python3 -u tools/dream_loop.py --base "$BASE" --lora "$LORA_DIR" --no-memory --n 8 > logs/ab_lora.log 2>&1
echo "lora_rc=$?" >> "$R"

echo "[ab] === 汇总 ===" >> "$R"
for tag in bare lora; do
  L="logs/ab_${tag}.log"
  SUM=$(grep -a '通过(入库)' "$L" | tail -1 | sed 's/^ *//')
  echo "[$tag] $SUM" >> "$R"
  echo "[$tag] ACCEPT=$(grep -ac 'ACCEPT 入库' "$L") vid_verify_fail=$(grep -ac 'REJECT vid_verify_fail' "$L") task_mismatch=$(grep -ac 'REJECT task_mismatch' "$L") 无视频=$(grep -ac '无视频产出' "$L") exec失败=$(grep -ac '\[exec\] 失败' "$L") 复读拒收=$(grep -ac '复读指纹' "$L")" >> "$R"
done

# 逐题对照（任务顺序确定性，按出现位置配对）
python3 - <<'PY' >> "$R" 2>&1
import re
def parse(path):
    tasks, verdicts = [], []
    cur = None
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.rstrip()
        m = re.match(r'\s*任务:\s*(.+)$', line)
        if m:
            cur = m.group(1).strip(); tasks.append(cur); continue
        if 'ACCEPT 入库' in line: verdicts.append(('PASS', line.split('ACCEPT')[1].strip()[:40]))
        elif 'REJECT vid_verify_fail' in line: verdicts.append(('vid_fail', ''))
        elif 'REJECT task_mismatch' in line: verdicts.append(('mismatch', ''))
        elif '无视频产出' in line: verdicts.append(('no_video', ''))
        elif '复读指纹' in line: verdicts.append(('echo', ''))
    return tasks, verdicts
tb, vb = parse('logs/ab_bare.log')
tl, vl = parse('logs/ab_lora.log')
print('%-4s %-42s %-10s %-10s' % ('#', 'task', 'bare', 'lora'))
for i in range(max(len(tb), len(tl))):
    t = (tb[i] if i < len(tb) else (tl[i] if i < len(tl) else '?'))[:42]
    x = vb[i][0] if i < len(vb) else '-'
    y = vl[i][0] if i < len(vl) else '-'
    print('%-4d %-42s %-10s %-10s' % (i, t, x, y))
PY

echo "[ab] DONE $(date '+%F %T')" >> "$R"
cat "$R"
