#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""node/board.py — 广播协议**本机端**（在 Windows 本机跑，往白板上写命令、收结果）

和 node/q.py 的分工：
  q.py      = node1 **专属**，管它上面的常驻服务与作业队列（点对点、只有 node1 有）
  board.py  = **所有机器通用**，一条命令广播出去，各节点各写各的格子

黑板协议（服务端零改动，全部走 ECS 信箱现成接口）：
  board.json            {seq, cmd, target, timeout, ts}   <- 本工具覆写
  hb_<node>.json        节点心跳                           <- 节点每 30s 覆写
  out_<node>_<seq>.txt  节点输出                           <- 节点执行完上传，不覆盖

  python node/board.py send "uname -a"                 # 广播给全部已登记节点
  python node/board.py send "df -h" -t node2            # 只给 node2
  python node/board.py send "sleep 300" -w 10 -T 600    # 只等 10 秒，命令侧超时 600 秒
  python node/board.py nodes                            # 谁活着
  python node/board.py reg node2                        # 登记新节点（装完 join.sh 后做一次）
  python node/board.py board                            # 看当前白板上是什么
  python node/board.py out node2 1792391036000          # 翻旧结果
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from polaris_node import DEFAULT, base, http_get, http_put  # noqa: E402

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

REG = os.path.join(ROOT, "data", "nodes.json")
C = dict(DEFAULT)


def load_reg():
    try:
        with open(REG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"nodes": []}


def save_reg(r):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    with open(REG, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)


def get_text(name):
    raw = http_get(C, name)
    return None if raw is None else raw.decode("utf-8", "replace")


def cmd_send(a):
    # 🔴 seq 必须单调递增：只用毫秒时间戳的话，本机时钟被 NTP 往回拨一次，
    #    新 seq 就会小于节点已记的 last_seq，那条命令会被**永久跳过**。
    old = 0
    prev = get_text("board.json")
    if prev:
        try:
            old = int(json.loads(prev).get("seq") or 0)
        except ValueError:
            old = 0
    seq = max(int(time.time() * 1000), old + 1)
    body = {
        "seq": seq,
        "cmd": a.cmd,
        "target": a.target,
        "timeout": a.timeout,
        "ts": int(time.time()),
    }
    http_put(C, "board.json", json.dumps(body, ensure_ascii=False).encode("utf-8"))
    print("[board] seq=%d target=%s 已上板" % (seq, a.target))

    if a.target == "*":
        want = load_reg().get("nodes") or []
    else:
        want = [a.target]
    if not want:
        print("[board] 没有登记过任何节点，先 python node/board.py reg <名字>")
        return 1

    print("[board] 等 %d 个节点的回传（最多 %.0fs）：%s" % (len(want), a.wait, ", ".join(want)))
    done = {}
    deadline = time.time() + a.wait
    while time.time() < deadline and len(done) < len(want):
        for n in want:
            if n in done:
                continue
            txt = get_text("out_%s_%d.txt" % (n, seq))
            if txt is not None:
                done[n] = txt
        if len(done) < len(want):
            time.sleep(0.15)

    rc = 0
    for n in want:
        print("\n" + "=" * 60)
        if n in done:
            print("### %s" % n)
            print(done[n].rstrip())
        else:
            claimed = get_text("run_%s_%d.json" % (n, seq))
            if claimed:
                print("### %s  —— 已领取，但 %.0fs 内没跑完（不是没收到，是还在跑）" % (n, a.wait))
            else:
                print("### %s  —— 超时未领取（节点可能掉线或是轮询还没轮到）" % n)
            rc = 2
    print("=" * 60)
    print("[board] %d/%d 有回传" % (len(done), len(want)))
    return rc


def discover_nodes():
    """自动发现：信件箱 /files 里所有心跳文件就是「来过的机器」，不必手工 reg。

    约定两种后缀：
      hb_<node>.json  当前版 polaris_node.py 写的（JSON，带 ts/load/host）
      hb_<node>.txt   早期 node_v5.py 写的（纯文本，无时间戳标准，只提示不参与派活）
    """
    url = "%s/files?key=%s" % (base(C), C["key"])
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return [], []
    cur, old = [], []
    for it in (d.get("files") or []):
        n = (it.get("n") or "") if isinstance(it, dict) else ""
        if n.startswith("hb_") and n.endswith(".json"):
            cur.append(n[3:-5])
        elif n.startswith("hb_") and n.endswith(".txt"):
            old.append(n[3:-4])
    return sorted(set(cur)), sorted(set(old))


def cmd_nodes(a):
    reg = load_reg().get("nodes") or []
    auto, old = discover_nodes()
    fresh = [n for n in auto if n not in reg]
    if fresh:
        r = load_reg()
        r.setdefault("nodes", [])
        for n in fresh:
            if n not in r["nodes"]:
                r["nodes"].append(n)
        save_reg(r)
        print("[nodes] 自动发现并登记：%s" % ", ".join(fresh))
    names = sorted(set(reg) | set(auto))
    if not names:
        print("[nodes] 没发现任何节点。机器上装 polaris_node.py 后会自己出现在这里。")
        return 1
    now = time.time()
    print("%-10s %-10s %-14s %s" % ("节点", "状态", "心跳", "负载/主机"))
    for n in names:
        txt = get_text("hb_%s.json" % n)
        if not txt:
            print("%-10s %-10s %-14s %s" % (n, "无心跳", "-", "-"))
            continue
        try:
            h = json.loads(txt)
        except ValueError:
            print("%-10s %-10s %-14s %s" % (n, "心跳损坏", "-", "-"))
            continue
        age = now - float(h.get("ts") or 0)
        state = "在线" if age < 90 else ("陈旧" if age < 600 else "离线")
        print("%-10s %-10s %-14s %.2f / %s" % (
            n, state, "%.0fs 前" % age, float(h.get("load") or 0), h.get("host") or "-"))
    if old:
        print("\n[老版残留] %s —— node_v5.py 时代的心跳，不参与派活；要接管就重装一次："
              % ", ".join(old))
        print("           curl -sS '%s/file/join.sh?key=%s' -o /tmp/join.sh && bash /tmp/join.sh <名字>"
              % (base(C), C["key"]))
    return 0


def cmd_reg(a):
    r = load_reg()
    if a.name in r.get("nodes", []):
        print("[reg] %s 已在册" % a.name)
        return 0
    r.setdefault("nodes", []).append(a.name)
    save_reg(r)
    print("[reg] 已登记 %s（现共 %d 个）" % (a.name, len(r["nodes"])))
    return 0


def cmd_unreg(a):
    r = load_reg()
    if a.name in r.get("nodes", []):
        r["nodes"].remove(a.name)
        save_reg(r)
        print("[unreg] 已移除 %s" % a.name)
    else:
        print("[unreg] 没有 %s" % a.name)
    return 0


def cmd_board(a):
    txt = get_text("board.json")
    print("[board] " + (txt.strip() if txt else "（白板为空）"))
    return 0


def cmd_out(a):
    txt = get_text("out_%s_%s.txt" % (a.node, a.seq))
    print(txt.rstrip() if txt else "[out] 没有 out_%s_%s.txt" % (a.node, a.seq))
    return 0 if txt else 2


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="fn")

    p = sub.add_parser("send", help="广播一条命令")
    p.add_argument("cmd")
    p.add_argument("-t", "--target", default="*")
    p.add_argument("-w", "--wait", type=float, default=90)
    p.add_argument("-T", "--timeout", type=int, default=300)
    p.set_defaults(fn=cmd_send)

    p = sub.add_parser("nodes", help="各节点心跳")
    p.set_defaults(fn=cmd_nodes)

    p = sub.add_parser("reg", help="登记节点")
    p.add_argument("name")
    p.set_defaults(fn=cmd_reg)

    p = sub.add_parser("unreg", help="移除节点")
    p.add_argument("name")
    p.set_defaults(fn=cmd_unreg)

    p = sub.add_parser("board", help="看当前白板")
    p.set_defaults(fn=cmd_board)

    p = sub.add_parser("out", help="翻历史结果")
    p.add_argument("node")
    p.add_argument("seq")
    p.set_defaults(fn=cmd_out)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help()
        return 1
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
