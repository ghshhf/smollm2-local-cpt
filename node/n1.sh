#!/usr/bin/env bash
# n1.sh — 本机直连 node1(魔搭) 的隧道入口
#
# 背景(2026-10-18):
#   信箱 mb.py 是轮询制(0.35~1s 一个来回), 只能单行命令, 大文件几乎搬不动。
#   现在 node1 反向把 22 端口挂到 ECS 回环(127.0.0.1:9022), 本机经 ECS 跳板直连:
#     - 支持多行命令/管道/heredoc, 有真实 stdout + exit code
#     - 毫秒级到秒级, 不再"发电报"
#     - scp 可传小文件(<2MB 合适); 大文件走 ms_relay.py(见下)
#
# 用法:
#   ./n1.sh "command"                 跑命令(可多行, 用 $'...' 或管道)
#   ./n1.sh -f script.sh              把本地脚本推过去执行
#   ./n1.sh --put <本地> <远端>       传文件(scp, 适合小文件)
#   ./n1.sh --get <远端> <本地>
#   ./n1.sh --alive                   探活
#
# 大文件(>5MB) 一律走魔搭仓库中转, 别用 scp(ECS 只有 ~94KB/s):
#   python tools/ms_relay.py push <文件> --as inbox/名字
#   ./n1.sh "cd /mnt/workspace/node1/dream && python3 tools/ms_relay.py pull inbox/名字 --out /mnt/workspace/node1/inbox"
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${N1_SSH_CONFIG:-/d/work/_local_ai/_tunnel/ssh_config}"
[ -f "$CFG" ] || { echo "[err] 找不到 ssh 配置: $CFG"; exit 2; }
export MSYS_NO_PATHCONV=1
SSH=(ssh.exe -F "$CFG" -o ConnectTimeout=15 -o LogLevel=ERROR)
SCP=(scp.exe -F "$CFG" -o LogLevel=ERROR)

case "${1:-}" in
  --alive)
    "${SSH[@]}" node1 "hostname; date '+%F %H:%M:%S'; uptime | cut -d, -f1-2; echo OK"
    ;;
  --put)
    [ $# -eq 3 ] || { echo "用法: $0 --put <本地> <远端>"; exit 2; }
    "${SCP[@]}" "$2" "node1:$3" && echo "put ok: $3"
    ;;
  --get)
    [ $# -eq 3 ] || { echo "用法: $0 --get <远端> <本地>"; exit 2; }
    "${SCP[@]}" "node1:$2" "$3" && echo "get ok: $3"
    ;;
  -f)
    [ $# -ge 2 ] || { echo "用法: $0 -f <本地脚本>"; exit 2; }
    TMP="/tmp/n1run_$$.sh"
    "${SCP[@]}" "$2" "node1:$TMP" || exit 1
    "${SSH[@]}" node1 "bash $TMP; rc=\$?; rm -f $TMP; exit \$rc"
    ;;
  ""|-h|--help)
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *)
    "${SSH[@]}" node1 "$*"
    ;;
esac
