#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n1_keepalive.py — 云端训练链保活(每 5 分钟由 Windows 计划任务调用)。

背景: node1 是魔搭免费 DSW 实例, 会被回收/重启; 重启后 node agent 会自己回来,
但训练链(chain_v1.sh)不会自动拉起 —— 历史上因此白丢过 13 分钟(06:15~06:28)。

设计要点:
  1. 判断逻辑全部放在 node1 侧执行(幂等), 本地只负责定时派发。
     这样即使任务在节点离线期间排队, 等实例回来也会一次性正确执行。
  2. chain_v1.sh 自带单实例守卫, 重复派发无害; pgrep 前置判断只是为了避免
     往队列里堆无意义任务。
  3. 派发失败(节点离线)静默返回, 不重试轰炸 —— 下次定时自然会再发。

用法: python n1_keepalive.py        # 单次派发
      python n1_keepalive.py --dry  # 只打印不发送
"""
import json
import sys
import urllib.request

ECS = "http://47.99.131.12:8081"

# node1 侧执行的保活命令。注意: 用 sh -c 包一层, 避免 dash 对 & 的解析差异。
KEEPALIVE_CMD = r"""
W=$(ls -d /mnt/workspace/node1/chain /workspace/node1/chain 2>/dev/null | head -1)
if [ -z "$W" ]; then
  echo "[keepalive $(date '+%m-%d %H:%M:%S')] chain dir not found, give up"
  exit 0
fi
P=$(cat "$W/chain.pid" 2>/dev/null)
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null && tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q "chain_v"; then
  echo "[keepalive $(date '+%m-%d %H:%M:%S')] chain alive pid=$P, nothing to do"
else
  echo "[keepalive $(date '+%m-%d %H:%M:%S')] DEAD (pid=$P) -> relaunch chain_v1.sh"
  ( setsid sh "$W/chain_v1.sh" >> "$W/chain.log" 2>&1 < /dev/null & )
  sleep 3
  echo "[keepalive $(date '+%m-%d %H:%M:%S')] new chain pid=$(cat "$W/chain.pid" 2>/dev/null)"
fi
""".strip()


def main():
    dry = "--dry" in sys.argv
    payload = {"cmd": KEEPALIVE_CMD}
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    try:
        req = urllib.request.Request(
            f"{ECS}/task",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            print(r.read().decode("utf-8", "ignore"))
        return 0
    except Exception as e:
        # 节点离线时静默失败: 计划任务下次会再试, 不刷错误噪音。
        print(f"[keepalive] dispatch failed (node offline?): {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 不调 sys.exit(): IPython 的 ! 命令会把 SystemExit 显示成红色报错,
    # 其实一切正常(return 0), 但看着吓人。
    main()
