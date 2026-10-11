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
[ -z "$TUNNEL" ] && { echo "ERR: TUNNEL env required"; exit 2; }
TUNNEL="${TUNNEL%/}"

WORKDIR="/mnt/workspace/node1/r4train"
GH_REPO="/mnt/workspace/node1/gh_repo"
ECS="http://47.99.131.12:8081"
LOG="$WORKDIR/train_r4.log"
PY="$(command -v python3 || command -v python)"
mkdir -p "$WORKDIR/r4_base" "$WORKDIR/corpus" "$WORKDIR/bin"

mail() {
  printf '%s' "$1" | "$PY" -c "import sys,json,urllib.request;m=sys.stdin.read();req=urllib.request.Request('$ECS/mail',data=json.dumps({'from':'node1','text':m}).encode(),headers={'Content-Type':'application/json'},method='POST');urllib.request.urlopen(req,timeout=20)" >/dev/null 2>&1 || true
}

echo "=== [$(date '+%F %T')] train_r4 v2 start TUNNEL=$TUNNEL ===" | tee -a "$LOG"
mail "train_r4 v2 start tunnel=$TUNNEL"

# ---------- 0. 可选代理(VMESS 提供时给 git 用; 任何失败静默回退直连) ----------
PROXY=""
if [ -n "${VMESS:-}" ]; then
  if curl -fsSL -m 180 --retry 3 -C - -o "$WORKDIR/bin/sing-box" "$TUNNEL/bin/sing-box" \
     && chmod +x "$WORKDIR/bin/sing-box"; then
    cat > "$WORKDIR/bin/mksb.py" <<'PYEOF'
import base64, json, os
j = json.loads(os.environ["VMESS"])
out = {
  "log": {"level": "warn"},
  "inbounds": [{"type": "socks", "tag": "in", "listen": "127.0.0.1", "listen_port": 1080}],
  "outbounds": [{
    "type": "vmess", "server": j["add"], "server_port": int(j["port"]),
    "uuid": j["id"], "alter_id": int(j.get("aid") or 0),
    "security": j.get("scy") or "auto",
    "transport": {"type": "ws", "path": j.get("path", "/"),
                  "headers": {"Host": j.get("host") or j["add"]}}
  }]
}
if j.get("tls"):
    out["outbounds"][0]["tls"] = {"enabled": True, "server_name": j.get("sni") or ""}
print(json.dumps(out))
PYEOF
    if VMESS="$VMESS" "$PY" "$WORKDIR/bin/mksb.py" > "$WORKDIR/bin/sbconfig.json" 2>/dev/null; then
      ( nohup "$WORKDIR/bin/sing-box" run -c "$WORKDIR/bin/sbconfig.json" \
          > "$WORKDIR/bin/sb.log" 2>&1 < /dev/null & )
      sleep 2
      if "$WORKDIR/bin/sing-box" version >/dev/null 2>&1; then
        PROXY="socks5://127.0.0.1:1080"
        mail "proxy up (git use only)"
      fi
    fi
  fi
  [ -z "$PROXY" ] && mail "proxy unavailable, direct mode"
fi

# ---------- 1. 同步脚本仓库(失败不影响主流程, 主脚本走隧道) ----------
if [ -d "$GH_REPO/.git" ]; then
  if [ -n "$PROXY" ]; then
    ( cd "$GH_REPO" && git -c http.proxy="$PROXY" pull -q origin main ) \
      && mail "git pull ok(proxy)" || mail "git pull fail(non-fatal)"
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
  local out="$dir/$(basename "$rel")" url="$TUNNEL/$rel" cur tries=0
  while :; do
    cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    [ "$cur" = "$want" ] && break
    tries=$((tries+1))
    if [ "$tries" -gt 30 ]; then
      mail "DL_FAIL $rel cur=$cur want=$want"
      echo "DL_FAIL $rel cur=$cur want=$want" | tee -a "$LOG"
      return 1
    fi
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
