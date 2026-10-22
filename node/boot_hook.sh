# 99_polaris_start.sh — Polaris 开机自启钩子（node1）
#
# 为什么放这里：node1 是 K8s Pod，没有 systemd、没有 cron。
# 但 /etc/dsw/bin/entrypoints/entrypoint-main 会在启动时把
# /etc/dsw/bin/prestart/*_* 里的脚本按文件名顺序 `source` 一遍 —— 这就是官方留的钩子。
# 安装位置: /etc/dsw/bin/prestart/99_polaris_start.sh
#
# 坑（都踩过）：
#   1) prestart 阶段 NAS 可能还没挂上，所以先等 /mnt/workspace 出现，最多等 180s；
#   2) 这个文件是被 source 的，绝不能 exit，也不能阻塞前台 —— 全部丢进子 shell + setsid；
#   3) 有 STOP 文件时 guard 会立刻退出，等于"用户手动关了就真的不启动"。
#
# 用户 2026-10-19 要求：开机后不该再需要手贴任何命令；只有手动关才会停。

POLARIS_ROOT=/mnt/workspace/node1

(
  i=0
  while [ $i -lt 180 ]; do
    [ -f "$POLARIS_ROOT/node/sup.sh" ] && break
    sleep 1
    i=$((i + 1))
  done
  if [ -f "$POLARIS_ROOT/node/sup.sh" ]; then
    cd "$POLARIS_ROOT" || exit 0
    mkdir -p logs
    echo "[boot-hook $(date '+%F %T')] prestart 钩子触发，拉起 sup.sh guard" >> logs/supervisor.log
    setsid nohup bash node/sup.sh guard </dev/null >>logs/supervisor.log 2>&1 &
  else
    echo "[boot-hook $(date '+%F %T')] 等 180s 仍没有 $POLARIS_ROOT/node/sup.sh，放弃" \
      >> /tmp/polaris_boot_hook.log 2>&1
  fi
) &
