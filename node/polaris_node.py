#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""polaris_node.py — 广播协议**节点端**（单文件 · 纯标准库 · 无 pip 依赖）

为什么是"广播"而不是"点对点队列"：
  老路 `D:/work/_local_ai/relay/heartbeat_relay.py` 已经验证过——所有节点读**同一块白板**，
  各自把结果写到自己的格子，彼此可见。好处是**加机器不用改任何代码**：
  新机器装完就自动看见黑板上的命令，不需要在服务端登记地址、分配队列。

黑板协议（三样东西，全部走 ECS 信箱现成的 /upload 与 /file，不改服务端）：
  board.json              <- 本地 `node/board.py send` 覆写；{seq, cmd, target, timeout, ts}
  hb_<node>.json          <- 节点每 30s 覆写；证明活着
  out_<node>_<seq>.txt    <- 每个节点执行完自己上传；同一 seq 不覆盖，历史可回溯

节点只认 seq：黑板 seq 比自己记的大，就去执行；比 target 不匹配就跳过（但 seq 照样推进，
避免重启后补跑一堆过期命令）。**取走不销毁**，所以多节点并发读同一块板也不会互相抢。

用法：
  python3 polaris_node.py run                  # 前台跑（watchdog 的 while true 里就这么用）
  python3 polaris_node.py once                 # 只处理一轮，便于排错
  python3 polaris_node.py hb                   # 只打一次心跳
  python3 polaris_node.py init --node node2    # 写配置
  python3 polaris_node.py status               # 看自己配置与上次 seq

配置：~/.polaris/node.json  {host, port, key, node, poll, timeout}
状态：~/.polaris/state.json {last_seq}
日志：~/.polaris/node.log
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HOME = os.path.expanduser("~/.polaris")
CONF = os.path.join(HOME, "node.json")
STATE = os.path.join(HOME, "state.json")
LOG = os.path.join(HOME, "node.log")

DEFAULT = {
    "host": "47.99.131.12",
    "port": 8081,
    "key": "k0u4mail",
    "node": "",
    "poll": 0.3,
    "timeout": 300,
}
BOARD = "board.json"
MAX_OUT = 256 * 1024
# 看护循环把 stdout 重定向到 crash.log；如果 say() 同时 print 又写 node.log，
# 每条都会出现两遍。看护进程设 POLARIS_ECHO=0，交互式（once/hb）默认 1。
ECHO = os.environ.get("POLARIS_ECHO", "1") == "1"


def say(msg):
    line = "%s %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    if ECHO:
        print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if os.path.getsize(LOG) > 4 * 1024 * 1024:
            os.replace(LOG, LOG + ".1")
    except OSError:
        pass


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def cfg():
    c = dict(DEFAULT)
    c.update(load_json(CONF, {}))
    return c


def base(c):
    return "http://%s:%d" % (c["host"], c["port"])


def http_get(c, name, timeout=20):
    """读信箱里的文件；不存在返回 None。

    🔴 不要改成长连接(keep-alive)：实测正相反——空闲后复用要 1060ms，
       新建连接只要 70~90ms。ECS 上那个服务是单线程 HttpListener，
       一条空闲连接会把后面的连接挡住。所以每次都用新连接、发完就断。
    """
    url = "%s/file/%s?key=%s" % (base(c), urllib.parse.quote(name), c["key"])
    for i in (0, 1, 2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, OSError):
            if i == 2:
                return None
            time.sleep(0.15 * (i + 1))
    return None


def http_put(c, name, data, timeout=60):
    """覆写信箱里的文件（/upload 是覆盖语义）。失败重试两次再抛。"""
    url = "%s/upload?name=%s&key=%s" % (base(c), urllib.parse.quote(name), c["key"])
    last = None
    for i in (0, 1, 2):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:      # noqa: BLE001  网络抖动一律重试
            last = e
            if i < 2:
                time.sleep(0.15 * (i + 1))
    raise last


def heartbeat(c):
    hb = {
        "node": c["node"],
        "ts": int(time.time()),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "pid": os.getpid(),
        "load": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0,
    }
    http_put(c, "hb_%s.json" % c["node"], json.dumps(hb).encode())
    return hb


def run_cmd(cmd, timeout):
    """整组执行；超时按进程组杀，避免留下孙子进程。

    🔴 输出必须落临时文件，不能用 PIPE 收：3GB 内存的机器上一条
       `cat 大日志` 就能把 PIPE 全读进内存直接 OOM。落盘后只读前 256KB。
    """
    f = tempfile.TemporaryFile()
    try:
        try:
            p = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                start_new_session=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        except OSError as e:
            return "spawn failed: %s" % e
        try:
            p.wait(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), 15)
                p.wait(timeout=10)
            except Exception:      # noqa: BLE001
                try:
                    os.killpg(os.getpgid(p.pid), 9)
                except OSError:
                    pass
            rc = "timeout(%ss)" % timeout
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        raw = f.read(MAX_OUT)
    finally:
        f.close()
    txt = raw.decode("utf-8", "replace")
    if size > MAX_OUT:
        txt += "\n...[截断，原始 %d 字节]" % size
    return "rc=%s\n%s" % (rc, txt)


def once(c, verbose=True):
    """处理一轮：看板 -> 该我就跑 -> 回传。返回处理了几条。"""
    st = load_json(STATE, {"last_seq": 0})
    last = int(st.get("last_seq") or 0)

    raw = http_get(c, BOARD)
    if not raw:
        return 0
    try:
        b = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        # 极端情况：正好读到半截覆写。下一轮再来，不当错误。
        return 0

    seq = int(b.get("seq") or 0)
    if seq <= last:
        return 0

    st["last_seq"] = seq
    save_json(STATE, st)

    target = str(b.get("target") or "*")
    if target not in ("*", c["node"]):
        if verbose:
            say("[skip] seq=%d target=%s（不是给我的）" % (seq, target))
        return 0

    cmd = str(b.get("cmd") or "")
    if not cmd.strip():
        return 0
    tmo = int(b.get("timeout") or c["timeout"])
    say("[run ] seq=%d target=%s 超时=%ds" % (seq, target, tmo))
    say("       $ %s" % cmd.replace("\n", " ; ")[:200])
    # 领取标记：本机 board.py 超时没等到结果时，靠它区分「已领取还在跑」和「根本没领到」
    try:
        http_put(c, "run_%s_%d.json" % (c["node"], seq),
                 json.dumps({"node": c["node"], "seq": seq, "ts": int(time.time())}).encode())
    except Exception as e:      # noqa: BLE001
        say("[warn] 领取标记没传上去（不影响执行）: %s" % e)
    t0 = time.time()
    out = run_cmd(cmd, tmo)
    el = time.time() - t0
    say("[done] seq=%d %.1fs 输出 %d 字符" % (seq, el, len(out)))
    head = "node=%s seq=%d elapsed=%.1fs\n---\n" % (c["node"], seq, el)
    try:
        http_put(c, "out_%s_%d.txt" % (c["node"], seq), (head + out).encode("utf-8"))
    except Exception as e:
        say("[warn] 回传失败: %s" % e)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "once", "hb", "init", "status"])
    ap.add_argument("--node", default="")
    ap.add_argument("--host", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--key", default="")
    ap.add_argument("--poll", type=float, default=0)
    a = ap.parse_args()

    if a.action == "init":
        c = cfg()
        for k in ("node", "host", "key"):
            if getattr(a, k):
                c[k] = getattr(a, k)
        if a.port:
            c["port"] = a.port
        if a.poll:
            c["poll"] = a.poll
        if not c["node"]:
            c["node"] = (os.uname().nodename if hasattr(os, "uname") else "node")[:16]
        save_json(CONF, c)
        print(json.dumps(c, ensure_ascii=False, indent=2))
        return 0

    c = cfg()
    if a.node:
        c["node"] = a.node
    if not c["node"]:
        c["node"] = (os.uname().nodename if hasattr(os, "uname") else "node")[:16]

    if a.action == "status":
        print(json.dumps({"conf": c, "state": load_json(STATE, {}),
                          "log": LOG}, ensure_ascii=False, indent=2))
        return 0

    if a.action == "hb":
        print(json.dumps(heartbeat(c), ensure_ascii=False))
        return 0

    if a.action == "once":
        n = once(c)
        print("[once] 处理 %d 条" % n)
        return 0

    say("===== 节点 %s 上线  poll=%ss =====" % (c["node"], c["poll"]))
    last_hb = 0.0
    fails = 0
    while True:
        try:
            once(c)
            fails = 0
        except Exception as e:
            fails += 1
            say("[err ] %s（连续 %d 次）" % (e, fails))
        now = time.time()
        if now - last_hb >= 30:
            try:
                heartbeat(c)
                last_hb = now
            except Exception as e:      # noqa: BLE001  心跳失败不能静默
                say("[warn] 心跳失败: %s" % e)
        time.sleep(min(30, c["poll"] + min(fails, 5) * 2))


if __name__ == "__main__":
    sys.exit(main())
