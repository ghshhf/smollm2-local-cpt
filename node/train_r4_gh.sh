#!/usr/bin/env bash
# =============================================================================
# train_r4_gh.sh v3 — GitHub-only 路径：数据全部来自 git 仓库分片，不依赖隧道
# 流程: mihomo(Clash) 代理 → git pull → 拼回权重/语料(reassemble) → 后台续训
#       → 训完把新 checkpoint 分片 push 回仓库 output/ 目录
# 用法(经 /task 派发): bash /mnt/workspace/node1/gh_repo/node/train_r4_gh.sh
# 幂等: reassemble 拼回后 sha256 校验跳过已完成的; 训练进程在跑则不重复启动。
# =============================================================================
set -u

GH_REPO="/mnt/workspace/node1/gh_repo"
WORKDIR="/mnt/workspace/node1/r4train"
ECS="http://47.99.131.12:8081"
PY="$(command -v python3 || command -v python)"
LOG="$WORKDIR/train_r4_gh.log"
mkdir -p "$WORKDIR/bin"

mail() {
  printf '%s' "$1" | "$PY" -c "import sys,json,urllib.request;m=sys.stdin.read();req=urllib.request.Request('$ECS/mail',data=json.dumps({'from':'node1','text':m}).encode(),headers={'Content-Type':'application/json'},method='POST');urllib.request.urlopen(req,timeout=20)" >/dev/null 2>&1 || true
}

echo "=== [$(date '+%F %T')] train_r4_gh v3 start ===" | tee -a "$LOG"
mail "gh-path v3 start"

# ---------- 0) mihomo(Clash.Meta) 代理, 资产来自仓库(不经隧道) ----------
MH="$WORKDIR/bin/mihomo"; MHD="$WORKDIR/bin/mihomo_cfg"
PROXY=""
if [ ! -x "$MH" ] && [ -f "$GH_REPO/node/assets/mihomo.gz" ]; then
  gunzip -c "$GH_REPO/node/assets/mihomo.gz" > "$MH" && chmod +x "$MH"
fi
if [ -x "$MH" ]; then
  if [ ! -f "$MHD/config.yaml" ]; then
    mkdir -p "$MHD"
    cp "$GH_REPO/node/assets/sub_clash.yaml" "$MHD/config.yaml"
  fi
  if ! pgrep -f "bin/mihomo -d" >/dev/null 2>&1; then
    ( nohup "$MH" -d "$MHD" > "$MHD/mihomo.log" 2>&1 < /dev/null & )
  fi
  for i in $(seq 1 20); do
    if curl -s -m 3 -x http://127.0.0.1:7890 -o /dev/null -w "%{http_code}" http://www.gstatic.com/generate_204 2>/dev/null | grep -q "204"; then
      PROXY="http://127.0.0.1:7890"; break
    fi
    sleep 1
  done
  [ -n "$PROXY" ] && mail "mihomo proxy up :7890" || mail "mihomo up but no net via proxy, direct mode"
fi

# ---------- 1) git pull(SSH 走 clash 的 CONNECT 代理, 无 nc 则直连) ----------
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
if [ -n "$PROXY" ] && command -v nc >/dev/null 2>&1; then
  export GIT_SSH_COMMAND="$GIT_SSH_COMMAND -o ProxyCommand='nc -X connect -x 127.0.0.1:7890 %h %p'"
fi
if [ -d "$GH_REPO/.git" ]; then
  ( cd "$GH_REPO" && git pull -q origin main ) \
    && mail "git pull ok" || mail "git pull fail(non-fatal, using local copy)"
else
  mail "FATAL: $GH_REPO missing (.git)"
  exit 2
fi

# ---------- 2) 拼回权重/语料(幂等+sha256) ----------
"$PY" "$GH_REPO/node/reassemble_r4.py" --out "$WORKDIR" > "$WORKDIR/reassemble.log" 2>&1
RTAIL=$(tail -3 "$WORKDIR/reassemble.log" | tr '\n' ' ')
mail "reassemble: ${RTAIL:0:180}"
grep -q "RESULT FAIL=0" "$WORKDIR/reassemble.log" \
  || { mail "FATAL: reassemble incomplete — see reassemble.log"; exit 3; }

# ---------- 3) 环境自检 + 后台启动训练 ----------
"$PY" -c "import torch,transformers" 2>/dev/null \
  || { mail "FATAL: torch/transformers missing"; exit 6; }

# 上传地址: 信箱里若有活的 TUNNEL_URL 就用(可选加速), 否则训练完走 git 回传
TUNNEL=""
T=$(curl -s -m 15 "${ECS}/mail?from=local" 2>/dev/null | "$PY" -c "
import sys, json
try:
    ms = [m for m in json.load(sys.stdin).get('messages', []) if 'TUNNEL_URL=' in str(m.get('text', ''))]
    print(str(ms[-1]['text']).split('TUNNEL_URL=')[1].split()[0] if ms else '')
except Exception:
    print('')
" 2>/dev/null)
[ -n "$T" ] && TUNNEL="${T%%/}"

if pgrep -f "train_remote_v2.py" >/dev/null 2>&1; then
  mail "train already running, skip launch"
else
  ( nohup "$PY" "$GH_REPO/node/train_remote_v2.py" \
      --base_dir "$WORKDIR/r4_base" \
      --corpus_dir "$WORKDIR/corpus" \
      --run_dir "$WORKDIR/r4_run" \
      --upload_url "${TUNNEL:-http://47.99.131.12:8081}/upload" \
      --steps 400 --lr 5e-4 \
      > "$WORKDIR/r4_run_train.log" 2>&1 < /dev/null & )
  sleep 2
  mail "TRAIN_STARTED pid=$(pgrep -f train_remote_v2.py | head -1)"
fi

# ---------- 4) 进度镜像 + 训完自动回传(git push 分片) ----------
(
  end=$((SECONDS+21600))
  while [ "$SECONDS" -lt "$end" ]; do
    sleep 120
    if ! pgrep -f "train_remote_v2.py" >/dev/null 2>&1; then
      t=$(tail -c 300 "$WORKDIR/r4_run_train.log" 2>/dev/null | tr '\n' ' ' | tr '\t' ' ')
      mail "TRAIN_EXIT tail: ${t:0:200}"
      break
    fi
    l=$(grep -E "loss|it\]|step" "$WORKDIR/r4_run_train.log" 2>/dev/null | tail -1 | tr '\n' ' ' | tr '\t' ' ')
    mail "progress ${l:0:180}"
  done
  # ---- 回传: checkpoint 分片 push 回仓库(隧道不可用时的主回传通道) ----
  BIN="$WORKDIR/r4_run/checkpoint-final/pytorch_model.bin"
  [ -f "$BIN" ] || BIN="$WORKDIR/r4_run/checkpoint-latest/pytorch_model.bin"
  if [ -f "$BIN" ]; then
    mail "pushing checkpoint back via git ..."
    ( cd "$GH_REPO" && git pull -q origin main ) 2>>"$LOG"
    ODIR="$GH_REPO/output/R4"
    mkdir -p "$ODIR"
    rm -f "$ODIR"/*.part* 2>/dev/null
    split -b 95m -d -a 2 "$BIN" "$ODIR/pytorch_model.bin.part"
    ( cd "$GH_REPO" \
      && git add -f output/ \
      && git commit -q -m "R4 checkpoint return $(date '+%F %T')" \
      && git push -q origin main ) \
      && mail "CHECKPOINT_PUSHED $(ls "$ODIR" | wc -l) parts" \
      || mail "checkpoint push FAIL (deploy key readonly?)"
  else
    mail "no checkpoint file found for return"
  fi
) < /dev/null > /dev/null 2>&1 &

echo "=== dispatch complete ===" | tee -a "$LOG"
mail "train_r4_gh v3 dispatch complete"
