#!/usr/bin/env bash
# =============================================================================
# train_r4.sh v2 — 魔搭节点端 R4(214M) 续训编排（断点续传版）
# 广播协议三件套: ①清单广播(manifest: 路由/文件名 字节数 sha256)
#                ②断点续传(aria2c -c + Range 多连接)
#                ③校验重传(逐文件字节核对, 不符即续传重试; sha256 复核)
# 幂等: 已完整文件直接跳过 —— 中断后重跑同一指令即自动续传补缺。
# 用法(由信箱派发):
#   TUNNEL=https://xxx.trycloudflare.com [VMESS='<vmess json>'] \
#     curl -fsSL $TUNNEL/scripts/train_r4.sh -o /tmp/train_r4.sh && bash /tmp/train_r4.sh
# =============================================================================
set -u

TUNNEL="${TUNNEL:-}"
ECS="http://47.99.131.12:8081"
WORKDIR="/mnt/workspace/node1/r4train"
GH_REPO="/mnt/workspace/node1/gh_repo"
LOG="$WORKDIR/train_r4.log"
PY="$(command -v python3 || command -v python)"
mkdir -p "$WORKDIR/r4_base" "$WORKDIR/corpus" "$WORKDIR/bin"

# 隧道地址自愈: 从 ECS 信箱取最新的 TUNNEL_URL= 消息(本机看门狗在隧道重建时发布)
resolve_tunnel() {
  local u
  u=$(curl -s -m 15 "${ECS}/mail?from=local" 2>/dev/null | "$PY" -c "
import sys, json
try:
    ms = [m for m in json.load(sys.stdin).get('messages', []) if 'TUNNEL_URL=' in str(m.get('text', ''))]
    print(str(ms[-1]['text']).split('TUNNEL_URL=')[1].split()[0] if ms else '')
except Exception:
    print('')
" 2>/dev/null)
  [ -n "$u" ] && TUNNEL="${u%%/}"
  return 0
}

mail() {
  printf '%s' "$1" | "$PY" -c "import sys,json,urllib.request;m=sys.stdin.read();req=urllib.request.Request('$ECS/mail',data=json.dumps({'from':'node1','text':m}).encode(),headers={'Content-Type':'application/json'},method='POST');urllib.request.urlopen(req,timeout=20)" >/dev/null 2>&1 || true
}

resolve_tunnel
[ -z "$TUNNEL" ] && { echo "ERR: no TUNNEL (env empty and mailbox has none)"; exit 2; }

echo "=== [$(date '+%F %T')] train_r4 v2 start TUNNEL=$TUNNEL ===" | tee -a "$LOG"
mail "train_r4 v2 start tunnel=$TUNNEL"

# ---------- 0. Clash(mihomo) 代理：内核+订阅YAML均经隧道获取, 任何失败静默回退直连 ----------
# mihomo = Clash.Meta 官方 Linux 内核; sub_clash.yaml = 用户机场订阅(flag=clash 原生导出)
PROXY=""
MH="$WORKDIR/bin/mihomo"
MHD="$WORKDIR/bin/mihomo_cfg"
if [ ! -x "$MH" ]; then
  curl -fsSL -m 300 --retry 3 -C - -o "$WORKDIR/bin/mihomo.gz" "$TUNNEL/bin/mihomo.gz" \
    && gunzip -c "$WORKDIR/bin/mihomo.gz" > "$MH" && chmod +x "$MH" \
    || mail "mihomo binary fetch fail, direct mode"
fi
if [ -x "$MH" ] && [ ! -f "$MHD/config.yaml" ]; then
  mkdir -p "$MHD"
  curl -fsSL -m 120 --retry 3 -o "$MHD/config.yaml" "$TUNNEL/bin/sub_clash.yaml" \
    || mail "clash yaml fetch fail, direct mode"
fi
if [ -x "$MH" ] && [ -f "$MHD/config.yaml" ]; then
  pkill -f "bin/mihomo -d" 2>/dev/null; sleep 1
  ( nohup "$MH" -d "$MHD" > "$MHD/mihomo.log" 2>&1 < /dev/null & )
  PROXY_OK=0
  for i in $(seq 1 20); do
    if curl -s -m 3 -x http://127.0.0.1:7890 -o /dev/null -w "%{http_code}" http://www.gstatic.com/generate_204 2>/dev/null | grep -q "204"; then
      PROXY_OK=1; break
    fi
    sleep 1
  done
  if [ "$PROXY_OK" = "1" ]; then
    PROXY="http://127.0.0.1:7890"
    mail "mihomo proxy up :7890"
  else
    mail "mihomo started but proxy check fail, direct mode; tail: $(tail -c 150 "$MHD/mihomo.log" 2>/dev/null)"
  fi
fi

# ---------- 1. 同步脚本仓库(失败不影响主流程, 主脚本走隧道) ----------
# SSH 经 HTTP 代理需 ProxyCommand(nc -X connect); 无 nc 则直连兜底
if [ -n "$PROXY" ] && command -v nc >/dev/null 2>&1; then
  export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand='nc -X connect -x 127.0.0.1:7890 %h %p'"
fi
if [ -d "$GH_REPO/.git" ]; then
  if [ -n "$PROXY" ] && command -v nc >/dev/null 2>&1; then
    ( cd "$GH_REPO" && GIT_SSH_COMMAND="$GIT_SSH_COMMAND" git pull -q origin main ) \
      && mail "git pull ok(ssh via clash)" \
      || { ( cd "$GH_REPO" && git pull -q origin main ) && mail "git pull ok(direct fallback)" || mail "git pull fail(non-fatal)"; }
  else
    ( cd "$GH_REPO" && git pull -q origin main ) \
      && mail "git pull ok" || mail "git pull fail(non-fatal)"
  fi
fi

# ---------- 2. 拉清单 ----------
MANIFEST="$WORKDIR/manifest.txt"
if ! curl -fsSL -m 120 --retry 5 -C - -o "$MANIFEST" "$TUNNEL/scripts/manifest.txt"; then
  mail "FATAL: manifest fetch fail (tunnel dead?)"
  echo "manifest fetch fail" | tee -a "$LOG"
  exit 3
fi
echo "manifest lines: $(wc -l < "$MANIFEST")" | tee -a "$LOG"

# ---------- 3. 断点续传 + 校验 ----------
dir_for() {
  case "$1" in
    r4_config/*|r4_weights/*) echo "$WORKDIR/r4_base" ;;
    corpus/*)                 echo "$WORKDIR/corpus" ;;
    *)                        echo "" ;;
  esac
}

dl_verify() { # <route/file> <bytes> <sha256|->
  local rel="$1" want="$2" sha="$3"
  local dir; dir=$(dir_for "$rel")
  [ -z "$dir" ] && return 1
  local out="$dir/$(basename "$rel")" cur tries=0 url
  while :; do
    cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    [ "$cur" = "$want" ] && break
    tries=$((tries+1))
    if [ "$tries" -gt 30 ]; then
      mail "DL_FAIL $rel cur=$cur want=$want"
      echo "DL_FAIL $rel cur=$cur want=$want" | tee -a "$LOG"
      return 1
    fi
    resolve_tunnel          # 隧道被回收换址时, 重试自动跟到新地址
    url="$TUNNEL/$rel"
    aria2c -c -x8 -s8 -k1M --file-allocation=none --auto-file-renaming=false \
      --allow-overwrite=true --max-tries=0 --retry-wait=3 --timeout=40 \
      --connect-timeout=15 --summary-interval=60 --console-log-level=warn \
      -d "$dir" -o "$(basename "$rel")" "$url" >> "$LOG" 2>&1
    sleep 3
  done
  if [ "$sha" != "-" ]; then
    local actual; actual=$("$PY" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');[h.update(b) for b in iter(lambda:f.read(1<<22),b'')];print(h.hexdigest())" "$out" 2>/dev/null)
    if [ "$actual" != "$sha" ]; then
      mail "SHA_FAIL $rel (corrupt, removed)"
      echo "SHA_FAIL $rel" | tee -a "$LOG"
      rm -f "$out"
      return 1
    fi
  fi
  echo "OK $rel ($want)" | tee -a "$LOG"
  return 0
}

FAIL=0
while read -r rel want sha; do
  [ -z "${rel:-}" ] && continue
  dl_verify "$rel" "$want" "$sha" || FAIL=$((FAIL+1))
done < "$MANIFEST"

if [ "$FAIL" -gt 0 ]; then
  mail "DOWNLOAD_INCOMPLETE fail=$FAIL — rerun same command to resume"
  echo "incomplete: $FAIL" | tee -a "$LOG"
  exit 4
fi
mail "all downloads verified"

# ---------- 4. 环境自检 + 后台启动训练 ----------
"$PY" -c "import torch,transformers" 2>/dev/null \
  || { mail "FATAL: torch/transformers missing"; exit 6; }

if pgrep -f "train_remote_v2.py" >/dev/null 2>&1; then
  mail "train already running, skip launch"
else
  resolve_tunnel   # 上传地址取最新隧道
  ( nohup "$PY" "$GH_REPO/node/train_remote_v2.py" \
      --base_dir "$WORKDIR/r4_base" \
      --corpus_dir "$WORKDIR/corpus" \
      --run_dir "$WORKDIR/r4_run" \
      --upload_url "$TUNNEL/upload" \
      --steps 400 --lr 5e-4 \
      > "$WORKDIR/r4_run_train.log" 2>&1 < /dev/null & )
  sleep 2
  mail "TRAIN_STARTED pid=$(pgrep -f train_remote_v2.py | head -1)"
fi

# ---------- 5. 进度镜像(每 120s 报 step/loss, 上限 6 小时) ----------
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
) < /dev/null > /dev/null 2>&1 &

echo "=== dispatch complete ===" | tee -a "$LOG"
mail "train_r4 v2 dispatch complete"
