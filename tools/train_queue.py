#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练排队机：强制「同一时刻只有一个训练在跑」。

设计（最小化，无过度工程）：
  - 单一插槽锁 .train_lock：dispatcher 跑训练前原子创建；进程结束即删。
  - 队列 .train_queue.json：submit 追加任务，run 顺序取出，跑完即删。
  - 约定：所有训练都必须经由本机 dispatcher（schtasks SYSTEM）发起，
    直接裸跑 train_lora.py 不受约束，故「训练一律走排队机」是铁律。

命令：
  python tools/train_queue.py submit <name> <cmd> [cwd]
  python tools/train_queue.py status
  python tools/train_queue.py run        # dispatcher 主循环（由 schtasks SYSTEM 调起）
  python tools/train_queue.py clear      # 清空队列（不动锁）
"""
import json
import os
import sys
import time
import datetime
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, ".train_lock")
QUEUE = os.path.join(ROOT, ".train_queue.json")


def _load_queue():
    if not os.path.exists(QUEUE):
        return []
    try:
        with open(QUEUE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_queue(q):
    with open(QUEUE, "w", encoding="utf-8") as f:
        json.dump(q, f, ensure_ascii=False, indent=2)


def _lock_held_by_dead():
    """锁存在但持有者已死 → 返回 True（应清理，避免死锁永久阻塞）。"""
    if not os.path.exists(LOCK):
        return False
    try:
        with open(LOCK, "r", encoding="utf-8") as f:
            info = json.load(f)
        pid = info.get("pid")
        if pid is None:
            return False
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                           capture_output=True, text=True)
        return str(pid) not in r.stdout
    except Exception:
        return False


def _acquire(job_name):
    if os.path.exists(LOCK):
        if _lock_held_by_dead():
            try:
                os.remove(LOCK)
            except OSError:
                pass
        else:
            return False
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({
            "job": job_name,
            "pid": os.getpid(),
            "start": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, ensure_ascii=False)
    return True


def _release():
    try:
        if os.path.exists(LOCK):
            os.remove(LOCK)
    except OSError:
        pass


def cmd_submit(name, cmd, cwd=None):
    q = _load_queue()
    q.append({
        "name": name,
        "cmd": cmd,
        "cwd": cwd,
        "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    _save_queue(q)
    print(f"[queue] submitted: {name}  (pending={len(q)})")


def cmd_status():
    q = _load_queue()
    lock = None
    if os.path.exists(LOCK):
        try:
            with open(LOCK, "r", encoding="utf-8") as f:
                lock = f.read().strip()
        except Exception:
            lock = "(unreadable)"
    print(f"[queue] lock: {lock or 'none'}")
    print(f"[queue] pending ({len(q)}):")
    for i, j in enumerate(q, 1):
        print(f"  {i}. {j['name']}  ::  {j['cmd']}")


def cmd_run():
    print(f"[queue] dispatcher start @ "
          f"{datetime.datetime.now().isoformat(timespec='seconds')}")
    while True:
        q = _load_queue()
        if not q:
            print("[queue] empty, dispatcher exit")
            break
        job = q[0]
        if not _acquire(job["name"]):
            print("[queue] lock held elsewhere, wait 30s ...")
            time.sleep(30)
            continue
        try:
            print(f"[queue] >>> run: {job['name']}  cmd={job['cmd']}")
            t0 = time.time()
            r = subprocess.run(job["cmd"], shell=True, cwd=job.get("cwd"),
                               text=True)
            dt = time.time() - t0
            print(f"[queue] <<< done: {job['name']}  rc={r.returncode}  "
                  f"{dt / 60:.1f}min")
        finally:
            _release()
        q = _load_queue()
        if q and q[0]["name"] == job["name"]:
            q.pop(0)
            _save_queue(q)


def cmd_clear():
    _save_queue([])
    print("[queue] queue cleared")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    op = sys.argv[1]
    if op == "submit":
        if len(sys.argv) < 4:
            print("usage: submit <name> <cmd> [cwd]")
            return
        cmd_submit(sys.argv[2], sys.argv[3],
                   sys.argv[4] if len(sys.argv) > 4 else None)
    elif op == "status":
        cmd_status()
    elif op == "run":
        cmd_run()
    elif op == "clear":
        cmd_clear()
    else:
        print(f"unknown op: {op}")
        print(__doc__)


if __name__ == "__main__":
    main()
