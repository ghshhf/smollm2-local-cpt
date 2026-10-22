#!/bin/sh
# join.sh — 一条命令把任意 Linux 机器接入广播协议（此后不用再贴命令）
#
# 用法（在目标机器上贴这一条就行）：
#   curl -sS "http://47.99.131.12:8081/file/join.sh?key=k0u4mail" -o /tmp/join.sh && bash /tmp/join.sh node2
#
# 它做四件事：
#   1. 拉节点端 polaris_node.py（纯标准库，不需要 pip / torch）
#   2. 写配置 ~/.polaris/node.json
#   3. 按优先级装开机钩子：systemd > 魔搭 prestart > crontab @reboot > /etc/rc.local > ~/.bashrc
#   4. 起永久看护循环（死了 5 秒后自己重来）
#
# 卸载：bash /tmp/join.sh <node> stop
set -u

ECS="http://47.99.131.12:8081"
KEY="k0u4mail"
DIR="$HOME/.polaris"
NODE="${1:-}"
ACT="${2:-start}"

if [ -z "$NODE" ]; then
  NODE=$(hostname 2>/dev/null | cut -c1-16)
  echo "[join] 没给节点名，用主机名：$NODE"
fi

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
if [ -z "$PY" ]; then
  echo "[join][fatal] 找不到 python3，这台机器接不了。"
  exit 1
fi
echo "[join] python = $PY ($($PY -V 2>&1))"

# ---------------------------------------------------------------- 停止
if [ "$ACT" = "stop" ]; then
  pkill -f 'polaris_node[.]py run' 2>/dev/null && echo "[join] 节点进程已停" || echo "[join] 本来就没在跑"
  pkill -f 'polaris_node[.]py keep' 2>/dev/null
  for f in /etc/dsw/bin/prestart/99_polaris_node.sh \
           /etc/systemd/system/polaris-node.service /etc/rc.local; do
    [ -f "$f" ] && grep -q polaris_node "$f" 2>/dev/null && rm -f "$f" && echo "[join] 已移除钩子 $f"
  done
  if command -v crontab >/dev/null 2>&1; then
    crontab -l 2>/dev/null | grep -v 'polaris_node' | crontab - 2>/dev/null
  fi
  sed -i '/polaris_node/d' "$HOME/.bashrc" 2>/dev/null
  echo "[join] 卸载完毕"
  exit 0
fi

# ---------------------------------------------------------------- 1. 拉节点端
mkdir -p "$DIR"
echo "[join] 拉节点端…"
curl -sS -m 120 "$ECS/file/polaris_node.py?key=$KEY" -o "$DIR/polaris_node.py" || {
  echo "[join][fatal] 下载失败：网络到不了信箱？"; exit 1; }
# 基础体检：必须是个能跑的 python 文件
if ! head -2 "$DIR/polaris_node.py" | grep -q python; then
  echo "[join][fatal] 拉下来的不是脚本，检查 ECS 信箱是否在线"; exit 1; fi
SIZE=$(wc -c < "$DIR/polaris_node.py")
echo "[join] polaris_node.py ${SIZE} 字节"

# ---------------------------------------------------------------- 2. 写配置
"$PY" "$DIR/polaris_node.py" init --node "$NODE" --poll "${POLARIS_POLL:-1}" >/dev/null || {
  echo "[join][fatal] 配置写失败"; exit 1; }
echo "[join] 配置：$DIR/node.json（node=$NODE）"

# ---------------------------------------------------------------- 3. 开机钩子
GUARD='[ -f "$HOME/.polaris/polaris_node.py" ] && ! pgrep -f "polaris_node[.]py run" >/dev/null 2>&1 && (setsid nohup sh -c '"'"'while true; do POLARIS_ECHO=0 '"$PY"' "$HOME/.polaris/polaris_node.py" run; sleep 5; done'"'"' </dev/null >>"$HOME/.polaris/crash.log" 2>&1 &)'
HOOK="none"

if [ -d /etc/dsw/bin/prestart ] && [ -w /etc/dsw/bin/prestart ]; then
  cat > /etc/dsw/bin/prestart/99_polaris_node.sh <<EOF
#!/bin/sh
# 由 join.sh 安装：魔搭 DSW 官方开机钩子（entrypoint-main 会 source prestart/*_*）
for i in \$(seq 1 60); do [ -f "\$HOME/.polaris/polaris_node.py" ] && break; sleep 3; done
$GUARD
EOF
  chmod 644 /etc/dsw/bin/prestart/99_polaris_node.sh
  HOOK="prestart（魔搭）"

elif command -v systemctl >/dev/null 2>&1 && [ -w /etc/systemd/system ] 2>/dev/null; then
  cat > /etc/systemd/system/polaris-node.service <<EOF
[Unit]
Description=polaris node (broadcast protocol)
After=network-online.target

[Service]
Type=simple
Environment=HOME=$HOME
ExecStart=/bin/sh -c 'while true; do $PY $DIR/polaris_node.py run; sleep 5; done'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload >/dev/null 2>&1
  systemctl enable polaris-node >/dev/null 2>&1 && HOOK="systemd"

elif command -v crontab >/dev/null 2>&1; then
  ( crontab -l 2>/dev/null | grep -v 'polaris_node'; echo "@reboot $GUARD" ) | crontab - 2>/dev/null \
    && HOOK="crontab @reboot"
fi

[ "$HOOK" = "none" ] && [ -w /etc/rc.local ] 2>/dev/null && {
  grep -q polaris_node /etc/rc.local 2>/dev/null || echo "$GUARD" >> /etc/rc.local
  HOOK="/etc/rc.local"
}

# ~/.bashrc 永远兜一层（开终端就顺手确保在跑，幂等）
grep -q 'polaris_node' "$HOME/.bashrc" 2>/dev/null || {
  printf '%s\n' "# polaris: 打开终端就确保节点在跑(幂等)" "$GUARD" >> "$HOME/.bashrc"
}
echo "[join] 开机钩子：$HOOK ＋ ~/.bashrc 兜底"

# ---------------------------------------------------------------- 4. 起看护循环
pkill -f 'polaris_node[.]py run' 2>/dev/null
# 🔴 必须等旧进程真的退出：否则两个实例会同时处理同一个 seq，重复执行 + 互相覆盖结果
i=0
while [ $i -lt 20 ]; do
  pgrep -f 'polaris_node[.]py run' >/dev/null 2>&1 || break
  sleep 0.5
  i=$((i + 1))
done
[ $i -ge 20 ] && echo "[join][warn] 旧进程 10 秒还没退，仍继续启动（可能出现两个实例）"
setsid nohup sh -c "while true; do POLARIS_ECHO=0 $PY $DIR/polaris_node.py run; sleep 5; done" \
  </dev/null >>"$DIR/crash.log" 2>&1 &
sleep 4

# ---------------------------------------------------------------- 5. 自检
if pgrep -f 'polaris_node[.]py run' >/dev/null 2>&1; then
  echo "[join] 节点已在跑：pid=$(pgrep -f 'polaris_node[.]py run' | head -1)"
else
  echo "[join][warn] 进程没起来，看这两个日志：$DIR/node.log（做了什么）、$DIR/crash.log（崩在哪）"
  tail -5 "$DIR/crash.log" 2>/dev/null
  tail -5 "$DIR/node.log" 2>/dev/null
fi
echo "[join] 打一次心跳…"
"$PY" "$DIR/polaris_node.py" hb 2>&1 | tail -1
echo
echo "[join] 完成。本机接下来会自动领白板上的命令，不用再贴命令了。"
echo "[join] 本机日志：$DIR/node.log（叙事）／$DIR/crash.log（崩溃堆栈）"
