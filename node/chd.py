#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chd.py — 直连通道守护进程（Channel Daemon）

为什么需要:
  Windows 版 ssh 客户端**不支持 ControlMaster**（实测 mux_client_request_session
  直接报 Connection reset）。而走 ProxyJump 每敲一条命令都要握手两次
  （本机→ECS→node1），一次 1.1~1.3 秒，比信箱快不了多少。

  这里把 SSH 长连接常驻在一个后台进程里，rc.py 通过本地 TCP 提交命令，
  单命令只剩"一次执行往返"，实测 ~80~150ms。

链路:
  本机 --ssh(ecs_key)--> ECS(Windows) --direct-tcpip--> 127.0.0.1:9022
      --反向隧道(tunnel.sh)--> node1:22

用法:
  python node/chd.py --daemon     # 后台常驻（rc.py 会自动拉起）
  python node/chd.py --status     # 看状态
  python node/chd.py --stop       # 停

协议（本地 127.0.0.1:19122，一行 JSON 请求，一行 JSON 响应）:
  请求: {"cmd": "ls -l", "timeout": 30}
  响应: {"rc": 0, "out": "...", "err": "..."}
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "logs", "chd.log")

ECS_HOST = "47.99.131.12"
ECS_USER = "Administrator"
ECS_KEY = os.path.join(os.path.dirname(ROOT), "deploy_aliyun", "ecs_key")
NODE1_USER = "root"
NODE1_KEY = os.path.join(os.path.dirname(ROOT), "_tunnel", "node1_key")
TUNNEL_PORT = 9022
LISTEN = ("127.0.0.1", 19122)

_lock = threading.Lock()
_state = {"ecs": None, "t": None, "born": 0, "cmds": 0}


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect():
    """建立 本机→ECS→(隧道)→node1 的两跳 SSH，返回 node1 的 Transport"""
    import paramiko

    ecs = paramiko.SSHClient()
    ecs.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ecs.connect(ECS_HOST, 22, username=ECS_USER, key_filename=ECS_KEY,
                timeout=12, banner_timeout=12, auth_timeout=12,
                look_for_keys=False, allow_agent=False)

    ch = ecs.get_transport().open_channel(
        "direct-tcpip", ("127.0.0.1", TUNNEL_PORT), ("127.0.0.1", 0), timeout=12)

    t = paramiko.Transport(ch)
    t.banner_timeout = 15
    t.start_client(timeout=15)
    key = paramiko.Ed25519Key.from_private_key_file(NODE1_KEY)
    t.auth_publickey(NODE1_USER, key)

    _state["ecs"] = ecs
    _state["t"] = t
    _state["born"] = time.time()
    log("connected (2-hop ok)")
    return t


def transport():
    t = _state.get("t")
    if t is not None and t.is_active():
        return t
    with _lock:
        t = _state.get("t")
        if t is not None and t.is_active():
            return t
        log("transport dead, reconnecting")
        return connect()


def shell():
    """常驻 shell 通道: 一次打开, 之后每条命令只剩 1 个往返。

    为什么不用 exec_command: 它每条命令都要 通道open + exec + exit-status 三次往返
    （实测通道 190~220ms）。用一条常驻 `bash --norc` 写命令、读标记，只剩 1 个 RTT（~90ms）。
    """
    ch = _state.get("sh")
    if ch is not None and not ch.closed:
        return ch
    t = transport()
    ch = t.open_session(timeout=15)
    ch.exec_command("cd /mnt/workspace/node1 && exec bash --norc")
    ch.settimeout(1.0)
    _state["sh"] = ch
    log("shell channel open")
    return ch


def run_shell(cmd, timeout=60):
    ch = shell()
    nonce = "%d" % int(time.time() * 1000 % 10 ** 9)
    mark = "__RCE%s__" % nonce
    ch.sendall(('{ %s ; } 2>&1; echo "%s$?"\n' % (cmd, mark)).encode("utf-8"))
    buf = b""
    mb_ = mark.encode()
    deadline = time.time() + timeout
    while True:
        try:
            if ch.recv_ready():
                buf += ch.recv(65536)
        except socket.timeout:
            pass
        if mb_ in buf:
            break
        if time.time() > deadline:
            try:
                ch.close()
            except Exception:
                pass
            _state["sh"] = None
            return 124, buf.decode("utf-8", "replace"), "(超时 %ds)" % timeout
        if not ch.recv_ready():
            time.sleep(0.001)
    text = buf.decode("utf-8", "replace")
    i = text.rfind(mark)
    out = text[:i].rstrip("\n")
    tail = text[i + len(mark):].strip().split("\n")[0]
    try:
        rc = int(tail or 0)
    except Exception:
        rc = 0
    _state["cmds"] += 1
    return rc, out, ""


def run(cmd, timeout=60):
    if os.environ.get("CHD_EXECMODE") == "1":
        return run_exec(cmd, timeout)
    try:
        return run_shell(cmd, timeout)
    except Exception as e:
        log("shell mode failed (%r), fallback to exec mode" % (e,))
        _state["sh"] = None
        return run_exec(cmd, timeout)


def run_exec(cmd, timeout=60):
    t = transport()
    ch = t.open_session(timeout=15)
    ch.exec_command(cmd)
    out, err = [], []
    deadline = time.time() + timeout
    ch.settimeout(1.0)
    while True:
        moved = False
        try:
            if ch.recv_ready():
                out.append(ch.recv(65536))
                moved = True
            if ch.recv_stderr_ready():
                err.append(ch.recv_stderr(65536))
                moved = True
        except socket.timeout:
            pass
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        if time.time() > deadline:
            ch.close()
            return 124, b"".join(out).decode("utf-8", "replace"), "(超时 %ds)" % timeout
        if not moved:
            time.sleep(0.002)
    rc = ch.recv_exit_status()
    ch.close()
    _state["cmds"] += 1
    return rc, b"".join(out).decode("utf-8", "replace"), b"".join(err).decode("utf-8", "replace")


def handle(conn):
    try:
        conn.settimeout(3600)
        buf = b""
        while not buf.endswith(b"\n"):
            part = conn.recv(4096)
            if not part:
                return
            buf += part
        req = json.loads(buf.decode("utf-8"))
        rc, out, err = run(req.get("cmd", ""), float(req.get("timeout", 60)))
        conn.sendall((json.dumps({"rc": rc, "out": out, "err": err}) + "\n").encode("utf-8"))
    except Exception as e:
        try:
            conn.sendall((json.dumps({"rc": 255, "out": "", "err": "daemon: %r" % (e,)}) + "\n").encode("utf-8"))
        except Exception:
            pass
        log("handle error: %r" % (e,))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def serve():
    # 🔴 首次连接失败不能退出：否则进程一退，19122 没人监听，
    #    rc.py 只能回落信箱（表现为"明明修好了却还是 2.5 秒"）。
    try:
        transport()
    except Exception as e:
        log("initial connect failed: %r (will retry on demand)" % (e,))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(LISTEN)
    s.listen(16)
    log("listening on %s:%d" % LISTEN)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def client(cmd, timeout=60):
    s = socket.create_connection(LISTEN, timeout=10)
    # 建连用 10s，但读结果必须按命令预算放宽 —— 否则长命令（如全盘 sync 17s）
    # 会在第 10 秒被读超时打断，白白回落信箱。
    s.settimeout(max(30.0, timeout + 30))
    s.sendall((json.dumps({"cmd": cmd, "timeout": timeout}) + "\n").encode("utf-8"))
    data = b""
    while not data.endswith(b"\n"):
        part = s.recv(65536)
        if not part:
            break
        data += part
    s.close()
    return json.loads(data.decode("utf-8", "replace"))


def alive():
    try:
        client("echo CHD_ALIVE", 8)
        return True
    except Exception:
        return False


def spawn():
    """后台拉起守护进程（Windows 用 DETACHED_PROCESS，脱离当前控制台）"""
    exe = sys.executable
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([exe, os.path.join(HERE, "chd.py"), "--daemon"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, close_fds=True, **kw)
    for _ in range(60):
        time.sleep(0.25)
        if alive():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("cmd", nargs="?", default="")
    a = ap.parse_args()

    if a.daemon:
        serve()
    elif a.status:
        ok = alive()
        print("daemon alive:", ok)
        if ok:
            t0 = time.time()
            r = client("hostname; uptime", 15)
            print("probe %.0fms rc=%s\n%s" % ((time.time() - t0) * 1000, r["rc"], r["out"].strip()))
    elif a.stop:
        try:
            client("pkill -f chd_self_kill_marker_unused", 3)
        except Exception:
            pass
        print("请在任务管理器结束 python chd.py，或直接结束占用 19122 的进程")
    elif a.cmd:
        t0 = time.time()
        r = client(a.cmd, 120)
        print(r["out"], end="")
        print("[chd] %.0fms rc=%s" % ((time.time() - t0) * 1000, r["rc"]), file=sys.stderr)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
