#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""node/supervisor.py — 节点总管：自愈 + 作业队列(可插队) + 防空闲回收 + 手动停机

四件事：
  1. **常驻服务自愈**（tasks.json）：隧道这种要一直活着的东西，死了 30s 内拉起。
  2. **作业队列**（queue.json）：一次性作业按优先级排队跑，slots 控制并发
     （CPU 只有 4 核，默认 1 个作业，避免互相抢）。
  3. **插队 / 抢占**：prio 数字越小越优先。高优先级作业进来时，
     正在跑的低优先级作业会被 SIGSTOP 冻住（不是杀掉，进度和内存都留着），
     插队作业跑完再 SIGCONT 接着跑。
  4. **防空闲回收 + 手动停机**：心跳 + 打 Jupyter 端口；`touch STOP` 是全停开关，
     protect=true 的控制通道（隧道）不剪。

为什么要抢占而不是"等它跑完"：用户 2026-10-19 的要求是"各种各样的任务都挂起来，
还能插队"。CPU 作业动辄一两个小时，如果只能 FIFO，插队等于没有。

用法:
  python3 node/supervisor.py            # 前台跑
  python3 node/supervisor.py --once     # 只扫一轮
  bash node/sup.sh                      # 幂等启动
  bash node/sup.sh stop|start|status    # 停机/恢复/看状态
  python3 node/q.py add "cmd" --prio 0  # 本地加作业（见 node/q.py）
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = os.path.join(ROOT, "STOP")
TASKS = os.path.join(ROOT, "node", "tasks.json")
QUEUE = os.path.join(ROOT, "node", "queue.json")
LOGDIR = os.path.join(ROOT, "logs")
LOG = os.path.join(LOGDIR, "supervisor.log")
HB = os.path.join(LOGDIR, "supervisor.hb")
JUP = "http://127.0.0.1:8088/"
INTERVAL = int(os.environ.get("SUP_INTERVAL", "30") or 30)


def say(msg):
    line = "[sup %s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def all_cmdlines():
    """扫 /proc 拿所有进程命令行。比 pgrep 稳：不受 shell 引号/pkill 自杀影响。"""
    out = {}
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        if pid == me:
            continue
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if cmd:
            out[pid] = cmd
    return out


def alive(pattern, cmds=None):
    cmds = cmds if cmds is not None else all_cmdlines()
    return [p for p, c in cmds.items() if pattern and pattern in c]


def pid_alive(pid):
    """活着吗？🔴 僵尸进程必须算死：Popen 没被 reap 时 os.kill(pid,0) 依然成功，
    但进程其实早退出了（实测把队列卡死：jump 跑完了却一直显示"在跑"，音频一直被冻着）。"""
    try:
        with open("/proc/%d/stat" % int(pid), "rb") as f:
            st = f.read().rsplit(b")", 1)[-1].split()[:1]
        if not st or st[0] == b"Z":
            return False
    except (OSError, ValueError, TypeError):
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def reap_zombies():
    """收尸：不 reap 就会攒僵尸，且让 pid_alive 误判。"""
    while True:
        try:
            p, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except OSError:
            return
        if p == 0:
            return


def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        say("%s 读不了(%r)，用默认值" % (os.path.basename(path), e))
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def start_job(name, cmd, log, cwd=None):
    """起一个作业：独立 session（方便整组 SIGSTOP/SIGCONT），日志追加。"""
    path = log if os.path.isabs(log) else os.path.join(ROOT, log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "ab", buffering=0)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd or ROOT, stdin=subprocess.DEVNULL,
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                         close_fds=True)
    return p.pid


# ---------------------------------------------------------------- 队列

def q_truthy_done(it):
    d = it.get("done_file")
    return bool(d) and os.path.exists(d if os.path.isabs(d) else os.path.join(ROOT, d))


def q_reconcile(q, cmds, slots=1):
    """把队列状态对齐现实，**分三轮**（顺序很重要）：

      1) running → done/failed/pending：先把"在跑"的真相摸清；
      2) paused → running：有空位就解冻（必须在第 1 轮之后数 running，
         否则会把刚跑完的作业还算成占着坑，导致插队结束后原作业一直冻着）；
      3) pending → running：接管已经在跑的（总管重启过）。

    🔴 曾经把三轮揉成一轮按 id 顺序走，结果 jump 跑完后 filler(P9) 抢走了唯一的坑，
       音频作业(P1)一直冻着 —— 被插队的作业必须优先于新排队作业恢复。
    """
    changed = False

    # ---- 第 1 轮：在跑的到底是死是活
    for it in q["items"]:
        if it.get("state") != "running":
            continue
        pid = it.get("pid")
        if pid and pid_alive(pid):
            continue
        if q_truthy_done(it):
            it["state"], it["pid"] = "done", None
            say("作业 %s 完成" % it.get("name"))
        else:
            it["attempts"] = int(it.get("attempts", 0)) + 1
            it["pid"] = None
            if it["attempts"] >= int(it.get("max_attempts", 3)):
                it["state"] = "failed"
                say("作业 %s 失败（第 %d 次，放弃）" % (it.get("name"), it["attempts"]))
            else:
                it["state"] = "pending"
                say("作业 %s 中断 → 重排（第 %d 次）" % (it.get("name"), it["attempts"]))
        changed = True

    # ---- 第 2 轮：被插队冻住的，有空位就接着跑
    running = [x for x in q["items"] if x.get("state") == "running"]
    for it in q["items"]:
        if it.get("state") != "paused":
            continue
        if len(running) >= slots:
            break
        pid = it.get("pid")
        if pid and pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGCONT)
            except OSError:
                pass
            it["state"] = "running"
            it.pop("preempted", None)
            running.append(it)
            say("作业 %s 解冻，继续" % it.get("name"))
        else:
            it["state"], it["pid"] = "pending", None
        changed = True

    # ---- 第 3 轮：接管已经在跑的 pending（比如总管重启过）
    for it in q["items"]:
        if it.get("state") != "pending":
            continue
        if q_truthy_done(it):
            it["state"], it["pid"] = "done", None
            say("作业 %s 已有完成标记 → 标 done" % it.get("name"))
            changed = True
            continue
        pat = it.get("pattern")
        if pat:
            found = alive(pat, cmds)
            if found:
                it["state"], it["pid"] = "running", found[0]
                say("接管已在跑的作业 %s pid=%s" % (it.get("name"), found[0]))
                changed = True
    return changed


def q_pick(q):
    pend = [i for i in q["items"] if i.get("state") == "pending"]
    if not pend:
        return None
    pend.sort(key=lambda i: (int(i.get("prio", 1)), str(i.get("added", ""))))
    return pend[0]


def q_step(q, cmds):
    slots = int(q.get("slots", 1) or 1)
    changed = q_reconcile(q, cmds, slots)
    running = [i for i in q["items"] if i.get("state") == "running"]
    nxt = q_pick(q)
    if nxt is None:
        return changed
    urgent = int(nxt.get("prio", 1))
    if len(running) >= slots:
        victim = max(running, key=lambda i: int(i.get("prio", 1)))
        if urgent < int(victim.get("prio", 1)) and victim.get("pid"):
            try:
                os.killpg(os.getpgid(victim["pid"]), signal.SIGSTOP)
                victim["state"] = "paused"
                victim["preempted"] = True
                say("插队：%s(P%s) 让位给 %s(P%s)，已冻结 pid=%s"
                    % (victim.get("name"), victim.get("prio", 1),
                       nxt.get("name"), urgent, victim["pid"]))
                changed = True
            except OSError as e:
                say("冻结 %s 失败(%r)，这轮不插队" % (victim.get("name"), e))
                return changed
        else:
            return changed
    # 每个作业自动带 done 标记：成功才 touch。这样"完成"是客观事实，
    # 不依赖 pid 判断，也避免队列文件并发写导致已完成的活被重跑。
    done = nxt.get("done_file") or ("logs/q_%s.done" % nxt.get("id", "x"))
    nxt["done_file"] = done
    full = "rm -f %s ; { %s ; } ; _rc=$? ; [ $_rc -eq 0 ] && touch %s ; exit $_rc" % (
        done, nxt["cmd"], done)
    nxt["pid"] = start_job(nxt.get("name", "job"), full,
                            nxt.get("log") or "logs/q_%s.log" % nxt.get("id", "x"))
    nxt["state"] = "running"
    nxt["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    nxt.pop("preempted", None)
    say("起作业 %s(P%s) pid=%s" % (nxt.get("name"), nxt.get("prio", 1), nxt["pid"]))
    return True


# ---------------------------------------------------------------- 心跳

def heartbeat():
    try:
        with open(HB, "a", encoding="utf-8") as f:
            f.write("%d\n" % int(time.time()))
        if os.path.getsize(HB) > 40000:
            with open(HB, encoding="utf-8") as f:
                lines = f.readlines()[-1000:]
            with open(HB, "w", encoding="utf-8") as f:
                f.writelines(lines)
    except OSError:
        pass
    try:
        urllib.request.urlopen(JUP, timeout=3).read(64)
    except Exception:
        pass


# ---------------------------------------------------------------- 主循环

def main():
    once = "--once" in sys.argv
    os.makedirs(LOGDIR, exist_ok=True)
    say("启动 pid=%d root=%s interval=%ds" % (os.getpid(), ROOT, INTERVAL))
    fails, last_start, tick = {}, {}, 0
    while True:
        tick += 1
        if os.path.exists(STOP):
            cmds = all_cmdlines()
            killed = 0
            for t in read_json(TASKS, {"tasks": []}).get("tasks", []):
                if t.get("protect"):
                    continue
                for pid in alive(t.get("pattern", ""), cmds):
                    try:
                        os.kill(pid, 15)
                        killed += 1
                    except OSError:
                        pass
            q = read_json(QUEUE, {"slots": 1, "items": []})
            for it in q["items"]:
                if it.get("pid"):
                    try:
                        os.killpg(os.getpgid(it["pid"]), signal.SIGTERM)
                        killed += 1
                    except OSError:
                        pass
                if it.get("state") in ("running", "paused"):
                    it["state"], it["pid"] = "pending", None
            write_json(QUEUE, q)
            say("看到 STOP → 停掉 %d 个作业进程（控制通道保留），总管退出" % killed)
            return 0

        cmds = all_cmdlines()
        for t in read_json(TASKS, {"tasks": []}).get("tasks", []):
            name = t.get("name", "?")
            if not t.get("enabled", True):
                continue
            if alive(t.get("pattern") or t["cmd"], cmds):
                fails[name] = 0
                continue
            d = t.get("done_file")
            if d and os.path.exists(d if os.path.isabs(d) else os.path.join(ROOT, d)):
                continue
            n = fails.get(name, 0)
            if time.time() - last_start.get(name, 0) < min(300, 15 * (2 ** min(n, 4))):
                continue
            pid = start_job(name, t["cmd"], t.get("log") or "logs/%s.log" % name, t.get("cwd"))
            last_start[name], fails[name] = time.time(), n + 1
            say("%s 不在 → 拉起（第 %d 次，pid=%s）" % (name, n + 1, pid))

        q = read_json(QUEUE, {"slots": 1, "items": []})
        if q_step(q, cmds):
            write_json(QUEUE, q)
        reap_zombies()

        if tick % 5 == 1:
            heartbeat()
        if once:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
