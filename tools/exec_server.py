#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exec_server.py — 魔搭实例上的执行服务（隧道后端）

为什么存在（2026-10-18 用户定调）:
  信箱是轮询制(0.35~1s 一个来回), 本质是发电报。魔搭是完整 Linux,
  用 frp 把本机 HTTP 请求直接打到实例内, 一次往返就是真实延迟,
  可以把远程当本机用: 执行命令、读写文件、看日志。

协议（全部 JSON，token 放 header X-Token 或 query t）:
  GET  /health                       -> {"ok":true,"host":..,"load":..}
  POST /exec   {"cmd":..,"cwd":..,"timeout":..,"bg":false}
  POST /put    ?path=..  (raw body)  -> 写文件
  GET  /get    ?path=..              -> 原始字节
  GET  /tail   ?path=..&n=50         -> 最后 n 行
  GET  /ls     ?path=..              -> 目录列表

用法: python3 exec_server.py [--port 8010] [--token k0u4mail2026]
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.parse import unquote

TOKEN = "k0u4mail2026"
MAX_BODY = 256 * 1024 * 1024


def jbody(h, obj, code=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    h.send_response(code)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "polaris-exec/1.0"

    def log_message(self, fmt, *args):
        # 静音默认日志，避免高 QPS 时刷屏
        pass

    # ── 鉴权 ──────────────────────────────────────────────────
    def _auth(self, q):
        tok = self.headers.get("X-Token") or (q.get("t") or [""])[0]
        if tok != TOKEN:
            jbody(self, {"ok": False, "err": "bad token"}, 403)
            return False
        return True

    # ── GET ───────────────────────────────────────────────────
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/health":
            try:
                load = os.getloadavg()
            except Exception:
                load = (0, 0, 0)
            return jbody(self, {"ok": True, "host": socket.gethostname(),
                                "load": [round(x, 2) for x in load],
                                "ts": int(time.time())})
        if not self._auth(q):
            return

        if u.path in ("/get", "/tail", "/ls"):
            path = (q.get("path") or [""])[0]
            if not path:
                return jbody(self, {"ok": False, "err": "missing path"}, 400)

        if u.path == "/get":
            if not os.path.isfile(path):
                return jbody(self, {"ok": False, "err": "not found"}, 404)
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    b = f.read(262144)
                    if not b:
                        break
                    self.wfile.write(b)
            return

        if u.path == "/tail":
            n = int((q.get("n") or ["50"])[0])
            if not os.path.isfile(path):
                return jbody(self, {"ok": False, "err": "not found"}, 404)
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                back = min(size, 256 * 1024)
                f.seek(size - back)
                chunk = f.read().decode("utf-8", "replace")
            lines = chunk.splitlines()[-n:]
            return jbody(self, {"ok": True, "lines": lines, "size": size,
                                "mtime": int(os.path.getmtime(path))})

        if u.path == "/ls":
            if not os.path.isdir(path):
                return jbody(self, {"ok": False, "err": "not a dir"}, 400)
            out = []
            with os.scandir(path) as it:
                for e in it:
                    try:
                        st = e.stat()
                        out.append({"name": e.name, "dir": e.is_dir(),
                                    "size": st.st_size, "mtime": int(st.st_mtime)})
                    except Exception:
                        pass
            out.sort(key=lambda x: (-x["mtime"]))
            return jbody(self, {"ok": True, "items": out[:500]})

        return jbody(self, {"ok": False, "err": "unknown endpoint"}, 404)

    # ── POST ──────────────────────────────────────────────────
    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._auth(q):
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return jbody(self, {"ok": False, "err": "body too large"}, 413)
        raw = self.rfile.read(n) if n else b""

        if u.path == "/put":
            path = (q.get("path") or [""])[0]
            if not path:
                return jbody(self, {"ok": False, "err": "missing path"}, 400)
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
            return jbody(self, {"ok": True, "path": path, "size": len(raw)})

        if u.path != "/exec":
            return jbody(self, {"ok": False, "err": "unknown endpoint"}, 404)

        try:
            spec = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            return jbody(self, {"ok": False, "err": "bad json: %s" % e}, 400)

        cmd = spec.get("cmd") or ""
        if not cmd:
            return jbody(self, {"ok": False, "err": "missing cmd"}, 400)
        cwd = spec.get("cwd") or "/mnt/workspace/node1"
        timeout = float(spec.get("timeout") or 120)
        bg = bool(spec.get("bg"))

        if bg:
            # 长任务：立刻返回 pid，日志自己用 /tail 看
            logf = spec.get("log") or "/mnt/workspace/node1/bg_exec.log"
            os.makedirs(os.path.dirname(logf), exist_ok=True)
            with open(logf, "ab") as lf:
                p = subprocess.Popen(["/bin/sh", "-c", cmd], cwd=cwd,
                                     stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            return jbody(self, {"ok": True, "pid": p.pid, "log": logf})

        t0 = time.time()
        try:
            p = subprocess.run(["/bin/sh", "-c", cmd], cwd=cwd,
                               capture_output=True, timeout=timeout)
            return jbody(self, {"ok": p.returncode == 0, "rc": p.returncode,
                                "out": p.stdout.decode("utf-8", "replace"),
                                "err": p.stderr.decode("utf-8", "replace"),
                                "sec": round(time.time() - t0, 3)})
        except subprocess.TimeoutExpired:
            return jbody(self, {"ok": False, "err": "timeout %.0fs" % timeout,
                                "sec": round(time.time() - t0, 3)}, 504)
        except Exception as e:
            return jbody(self, {"ok": False, "err": "%s: %s" % (type(e).__name__, e)}, 500)


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--token", default=TOKEN)
    a = ap.parse_args()
    TOKEN = a.token
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    srv.daemon_threads = True
    print("[exec] listening on 0.0.0.0:%d token=%s..." % (a.port, a.token[:4]), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
