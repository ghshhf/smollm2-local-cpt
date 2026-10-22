#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""node/q.py — 远程作业队列的**本地**命令行（在 Windows 本机跑，操作的是 node1 上的队列）

设计：队列文件 `/mnt/workspace/node1/node/queue.json` 是唯一真相，由 node1 上的
supervisor.py 消费；本工具只做"拉下来 → 改 → 推回去"，不常驻、不留状态。

  python node/q.py add "python3 tools/xxx.py --a 1" -n 训练A -p 1
  python node/q.py ls                    # 看队列（含 状态/优先级/耗时）
  python node/q.py go 3                  # 🔴 插队：把 3 号提到 P0，正在跑的低优先级会被冻住
  python node/q.py prio 3 2              # 改优先级
  python node/q.py stop 3                # 暂停（SIGSTOP 整组，进度和内存都留着）
  python node/q.py cont 3                # 继续
  python node/q.py rm 3                  # 删掉（跑着的一并停）
  python node/q.py retry 3               # 失败的重排
  python node/q.py clear                 # 清掉已完成/失败的
  python node/q.py slots 2               # 并发数（4 核建议 1，最多 2）
  python node/q.py log 3 -n 20           # 看该作业日志尾部

优先级：数字越小越优先（0=最高，9=最低）。默认 1。
插队做法：把高优先级作业塞进来后，总管会把正在跑的低优先级作业 SIGSTOP 冻住，
插队作业跑完再 SIGCONT 接着跑 —— 不是杀掉重来。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RC = os.path.join(HERE, "rc.py")
REMOTE_Q = "node/queue.json"
PY = sys.executable


def sh(args, timeout=120):
    p = subprocess.run([PY, RC] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def pull():
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    rc, out, err = sh(["get", REMOTE_Q, tmp])
    data = None
    if os.path.exists(tmp):
        try:
            with open(tmp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        os.unlink(tmp)
    if data is None:
        data = {"slots": 1, "items": []}
    data.setdefault("slots", 1)
    data.setdefault("items", [])
    return data


def push(q):
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=1)
    rc, out, err = sh(["put", tmp, REMOTE_Q])
    os.unlink(tmp)
    return rc == 0 and "[put]" in out


def find(q, key):
    key = str(key)
    for it in q["items"]:
        if str(it.get("id")) == key:
            return it
    for it in q["items"]:
        if it.get("name") == key:
            return it
    return None


def fmt_dur(sec):
    sec = int(sec)
    if sec < 120:
        return "%ds" % sec
    if sec < 7200:
        return "%dm" % (sec // 60)
    return "%.1fh" % (sec / 3600.0)


def cmd_add(a):
    q = pull()
    nid = max([int(i["id"]) for i in q["items"] if str(i.get("id", "")).isdigit()] or [0]) + 1
    it = {
        "id": str(nid),
        "name": a.name or ("job%d" % nid),
        "cmd": a.cmd,
        "prio": int(a.prio),
        "state": "pending",
        "attempts": 0,
        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": a.log or ("logs/q_%d.log" % nid),
    }
    if a.pattern:
        it["pattern"] = a.pattern
    q["items"].append(it)
    push(q)
    print("[q] 已排队 #%d %s  P%d  %s" % (nid, it["name"], it["prio"], a.cmd[:60]))
    if int(a.prio) == 0:
        print("    这是 P0：正在跑的低优先级作业会被冻住让位")


def cmd_ls(a):
    q = pull()
    items = sorted(q["items"], key=lambda i: (int(i.get("prio", 1)), str(i.get("id"))))
    if not items:
        print("[q] 队列空（slots=%s）" % q.get("slots", 1))
        return
    mark = {"pending": "等到", "running": "在跑", "paused": "冻住", "done": "完成", "failed": "失败"}
    print("[q] slots=%s  共 %d 条" % (q.get("slots", 1), len(items)))
    print("  %-3s %-16s %-4s %-6s %-5s %-8s %s" % ("id", "名称", "优先级", "状态", "尝试", "耗时", "命令"))
    for it in items:
        dur = ""
        if it.get("started") and it.get("state") in ("running", "paused"):
            try:
                t0 = time.mktime(time.strptime(it["started"], "%Y-%m-%d %H:%M:%S"))
                dur = fmt_dur(time.time() - t0)
            except Exception:
                pass
        print("  %-3s %-16s P%-3s %-6s %-5s %-8s %s" % (
            it.get("id"), (it.get("name") or "")[:16], it.get("prio", 1),
            mark.get(it.get("state"), it.get("state")), it.get("attempts", 0), dur,
            (it.get("cmd") or "")[:52]))


def cmd_go(a):
    q = pull()
    it = find(q, a.key)
    if not it:
        sys.exit("找不到作业 %s" % a.key)
    it["prio"] = 0
    it["state"] = "pending" if it.get("state") in ("done", "failed") else it.get("state")
    push(q)
    print("[q] #%s 提到 P0（插队）。总管下一轮(≤30s)会把低优先级冻住让它先跑。" % it["id"])


def cmd_prio(a):
    q = pull()
    it = find(q, a.key)
    if not it:
        sys.exit("找不到作业 %s" % a.key)
    it["prio"] = int(a.n)
    push(q)
    print("[q] #%s 优先级 → P%d" % (it["id"], it["prio"]))


def cmd_rm(a):
    q = pull()
    it = find(q, a.key)
    if not it:
        sys.exit("找不到作业 %s" % a.key)
    if it.get("pid"):
        pid = it["pid"]
        sh(["sh", "kill -TERM -%d 2>/dev/null; kill %d 2>/dev/null; echo done" % (pid, pid)])
    q["items"] = [x for x in q["items"] if x is not it]
    push(q)
    print("[q] 已删 #%s %s" % (it["id"], it.get("name")))


def cmd_stop(a):
    q = pull()
    it = find(q, a.key)
    if not it or not it.get("pid"):
        sys.exit("作业不在跑")
    sh(["sh", "kill -STOP -%d" % it["pid"]])
    it["state"] = "paused"
    push(q)
    print("[q] #%s 已冻结（内存和进度都留着）" % it["id"])


def cmd_cont(a):
    q = pull()
    it = find(q, a.key)
    if not it or not it.get("pid"):
        sys.exit("作业没有 pid")
    sh(["sh", "kill -CONT -%d" % it["pid"]])
    it["state"] = "running"
    it.pop("preempted", None)
    push(q)
    print("[q] #%s 已继续" % it["id"])


def cmd_retry(a):
    q = pull()
    it = find(q, a.key)
    if not it:
        sys.exit("找不到作业 %s" % a.key)
    it["state"], it["attempts"], it["pid"] = "pending", 0, None
    push(q)
    print("[q] #%s 已重排" % it["id"])


def cmd_clear(a):
    q = pull()
    keep = [x for x in q["items"] if x.get("state") not in ("done", "failed")]
    n = len(q["items"]) - len(keep)
    q["items"] = keep
    push(q)
    print("[q] 清掉 %d 条已完成/失败" % n)


def cmd_slots(a):
    q = pull()
    q["slots"] = int(a.n)
    push(q)
    print("[q] 并发 slots = %d" % q["slots"])


def cmd_log(a):
    q = pull()
    it = find(q, a.key)
    if not it:
        sys.exit("找不到作业 %s" % a.key)
    log = it.get("log") or ("logs/q_%s.log" % it["id"])
    rc, out, err = sh(["sh", "tail -n %d %s" % (a.n, log)])
    print(out.strip() or "(日志还空)")


def main():
    ap = argparse.ArgumentParser(prog="q", description="远程作业队列（可插队）")
    sub = ap.add_subparsers(dest="op")

    p = sub.add_parser("add", help="加作业"); p.add_argument("cmd")
    p.add_argument("-n", "--name", default=""); p.add_argument("-p", "--prio", default=1)
    p.add_argument("-l", "--log", default=""); p.add_argument("-P", "--pattern", default="")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("ls", help="看队列"); p.set_defaults(fn=cmd_ls)

    p = sub.add_parser("go", help="插队到 P0"); p.add_argument("key"); p.set_defaults(fn=cmd_go)

    p = sub.add_parser("prio", help="改优先级"); p.add_argument("key"); p.add_argument("n")
    p.set_defaults(fn=cmd_prio)

    p = sub.add_parser("rm", help="删作业"); p.add_argument("key"); p.set_defaults(fn=cmd_rm)
    p = sub.add_parser("stop", help="冻结"); p.add_argument("key"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("cont", help="继续"); p.add_argument("key"); p.set_defaults(fn=cmd_cont)
    p = sub.add_parser("retry", help="重排"); p.add_argument("key"); p.set_defaults(fn=cmd_retry)
    p = sub.add_parser("clear", help="清已完成/失败"); p.set_defaults(fn=cmd_clear)
    p = sub.add_parser("slots", help="并发数"); p.add_argument("n"); p.set_defaults(fn=cmd_slots)

    p = sub.add_parser("log", help="看作业日志"); p.add_argument("key")
    p.add_argument("-n", type=int, default=20); p.set_defaults(fn=cmd_log)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
