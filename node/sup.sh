#!/usr/bin/env bash
# node/sup.sh — 总管的启停外壳（node1 上没有 systemd/cron，这是唯一的进程管理层）
#
#   bash node/sup.sh            # 幂等启动（已在跑就什么都不做）
#   bash node/sup.sh guard      # 外层环：每 60s 确保总管活着，看到 STOP 就退出（开机钩子调这个）
#   bash node/sup.sh stop       # 手动全停（= touch STOP + 兜底 kill）
#   bash node/sup.sh start      # 恢复（= 删 STOP + 启动）
#   bash node/sup.sh status     # 一眼看状态
DIR=/mnt/workspace/node1
STOP="$DIR/STOP"
LOG="$DIR/logs/supervisor.log"
cd "$DIR" || exit 1
mkdir -p "$DIR/logs"
PAT='node/supervisor[.]py'

alive_pid() { pgrep -f "$PAT" 2>/dev/null | head -1; }

start() {
  rm -f "$STOP"
  P=$(alive_pid)
  if [ -n "$P" ]; then echo "[sup] 已在跑 pid=$P"; return 0; fi
  # 日志由 supervisor.py 自己写 LOG，这里只接住异常栈，避免同一行记两遍
  setsid nohup python3 -u node/supervisor.py >>/dev/null 2>>"$DIR/logs/supervisor.err" </dev/null &
  sleep 1
  echo "[sup] 启动 pid=$(alive_pid)"
}

stop() {
  touch "$STOP"
  P=$(alive_pid)
  [ -n "$P" ] && kill "$P" 2>/dev/null
  echo "[sup] 已停（STOP=$STOP）。要恢复: bash node/sup.sh start"
}

status() {
  P=$(alive_pid)
  echo "总管 : ${P:-未运行}"
  [ -n "$P" ] && ps -o pid,etime,pcpu,cmd -p "$P" | tail -n +2
  echo "STOP : $([ -f "$STOP" ] && echo '存在(已手动停)' || echo '无')"
  echo "心跳 : $(tail -n 1 "$DIR/logs/supervisor.hb" 2>/dev/null || echo 无)"
  echo "任务 :"
  python3 - "$DIR/node/tasks.json" <<'PY' 2>/dev/null || echo "  (无 tasks.json)"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
for t in d.get("tasks",[]):
    print("  - %-14s %s" % (t.get("name"), "开" if t.get("enabled",True) else "关"))
PY
}

case "${1:-start}" in
  start|"") start ;;
  stop)     stop ;;
  status)   status ;;
  guard)
    while true; do
      [ -f "$STOP" ] && exit 0
      [ -n "$(alive_pid)" ] || start >>"$LOG" 2>&1
      sleep 60
    done ;;
  *) echo "用法: bash node/sup.sh [start|stop|status|guard]"; exit 2 ;;
esac
