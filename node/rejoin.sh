#!/bin/sh
# rejoin.sh —— 一键诊断 + 重启节点接入（贴一条就能定位掉线原因）
#
# 用法：
#   curl -sS "http://47.99.131.12:8081/file/rejoin.sh?key=k0u4mail" -o /tmp/rejoin.sh && bash /tmp/rejoin.sh node2
#
# 它做四件事：
#   1. 打印机器身份（判断是不是被换成了新容器）
#   2. 探测到 ECS 信箱的连通性（判断是网络还是进程问题）
#   3. 清掉旧进程，重新接入
#   4. 等 8 秒，自己验证心跳有没有上去
set -u
ECS="http://47.99.131.12:8081"
KEY="k0u4mail"
NODE="${1:-node2}"

echo "========== 1. 这台机器是谁 =========="
echo "hostname : $(hostname 2>/dev/null)"
echo "时间     : $(date -u +%FT%TZ) (UTC)"
echo "内核     : $(uname -sr 2>/dev/null)"
echo "运行时长 : $(uptime -p 2>/dev/null || uptime 2>/dev/null)"
echo "CPU      : $(nproc 2>/dev/null) 核"
echo "内存     : $(free -m 2>/dev/null | awk '/Mem:/{print $2" MB 总 / "$3" MB 已用"}')"
echo "磁盘     : $(df -h / 2>/dev/null | awk 'NR==2{print $2" 总 / "$5" 已用"}')"
echo "python3  : $(command -v python3 || echo 没有)"
# 容器 id 变过就说明是被回收重开了新机器
if [ -f /proc/1/cgroup ]; then
  echo "容器标识 : $(head -1 /proc/1/cgroup 2>/dev/null | cut -c1-60)"
fi

echo ""
echo "========== 2. 到信箱通不通 =========="
T0=$(date +%s)
CODE=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$ECS/files?key=$KEY" 2>/dev/null || echo FAIL)
T1=$(date +%s)
if [ "$CODE" = "200" ]; then
  echo "信箱    : 通 (HTTP 200, 耗时 $((T1 - T0))s)"
else
  echo "信箱    : 不通 (code=$CODE)  ← 网络问题，重启也没用"
fi

echo ""
echo "========== 3. 旧进程状态 =========="
OLD=$(ps -ef 2>/dev/null | grep '[p]olaris_node' | wc -l)
echo "polaris_node 进程数: $OLD"
ps -ef 2>/dev/null | grep '[p]olaris_node' | head -3
LOGD="$HOME/.polaris"
if [ -f "$LOGD/node.log" ]; then
  echo "--- 最后 8 行旧日志 ---"
  tail -8 "$LOGD/node.log" 2>/dev/null
fi

echo ""
echo "========== 4. 清掉旧的，重新接入 =========="
curl -sS -m 60 "$ECS/file/join.sh?key=$KEY" -o /tmp/join.sh || {
  echo "[fatal] join.sh 下载失败"; exit 1; }
sh /tmp/join.sh "$NODE" stop 2>/dev/null
sleep 1
pkill -f 'polaris_node[.]py' 2>/dev/null
sleep 1
sh /tmp/join.sh "$NODE"
RC=$?
echo "[rejoin] join.sh 返回 $RC"

echo ""
echo "========== 5. 等 8 秒，验证心跳 =========="
sleep 8
HB=$(curl -sS -m 20 "$ECS/file/hb_${NODE}.json?key=$KEY" 2>/dev/null)
if [ -z "$HB" ]; then
  echo "心跳    : 没有回传 ← 机器上起了但传不上去"
else
  echo "心跳    : $HB"
fi
echo ""
echo "========== 完 =========="
echo "把上面「1. 这台机器是谁」和「5. 心跳」两段发给我，我就能判断原因。"
