#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb.py — 快速信箱客户端(keep-alive + 0.35s 轮询), 替代 cmd_run.py 的 4s 粒度

为什么快:
  - cmd_run.py 用 urllib 每次新建 TCP 连接 + 4s 轮询粒度 => 白等最多 4s。
  - mb.py 复用同一条 HTTP keep-alive 连接, 0.35s 轮询, 一有内容立刻返回。
  - 远程侧 node_v6 已把领任务间隔从 10s 降到 1s(见 node_v6.py), 两端一起提速。

用法:
  python mb.py "shell 命令"                    # 发出去并等回传
  python mb.py "shell 命令" -t 120             # 自定义超时秒数
  python mb.py "shell 命令" -n myfile.txt      # 指定回传文件名(默认按时间生成)
  python mb.py "shell 命令" --bg               # 只发不等
"""
import argparse, http.client, json, sys, time

ECS_HOST = "47.99.131.12"
ECS_PORT = 8081
KEY = "k0u4mail"


class Conn:
    def __init__(self):
        self.c = None

    def _new(self):
        self.c = http.client.HTTPConnection(ECS_HOST, ECS_PORT, timeout=30)
        return self.c

    def req(self, method, path, body=None, ctype="application/json", timeout=30):
        for attempt in (0, 1):
            try:
                c = self.c or self._new()
                c.timeout = timeout
                c.request(method, path, body=body, headers={"Content-Type": ctype})
                r = c.getresponse()
                data = r.read()
                if r.status >= 300:
                    raise IOError(f"HTTP {r.status}")
                return data
            except Exception:
                try:
                    self.c.close()
                except Exception:
                    pass
                self.c = None
                if attempt:
                    raise
        return b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("-t", "--timeout", type=float, default=180)
    ap.add_argument("-n", "--name", default="")
    ap.add_argument("--bg", action="store_true")
    a = ap.parse_args()

    tid = "mb-%d" % int(time.time() * 1000)
    name = a.name or ("mb_%s.txt" % tid)
    full = a.cmd + (" 2>&1 | curl -s -m 900 -X POST 'http://%s:%d/upload?name=%s&key=%s' "
                    "--data-binary @-" % (ECS_HOST, ECS_PORT, name, KEY))

    c = Conn()
    t0 = time.time()
    body = json.dumps({"cmd": full, "id": tid, "ts": int(time.time())}).encode()
    c.req("POST", "/task", body)
    sent = time.time() - t0
    print(f"[mb] 已投递 {sent*1000:.0f}ms  name={name}", file=sys.stderr)
    if a.bg:
        print(name)
        return

    deadline = time.time() + a.timeout
    text = ""
    polls = 0
    while time.time() < deadline:
        try:
            raw = c.req("GET", f"/file/{name}?key={KEY}", timeout=20)
            text = raw.decode("utf-8", "replace")
        except Exception:
            text = ""
        polls += 1
        if text.strip():
            break
        time.sleep(0.35)
    el = time.time() - t0
    print(f"[mb] 回传 {el:.2f}s ({polls} 次轮询)", file=sys.stderr)
    print(text if text.strip() else "(无回传)")


if __name__ == "__main__":
    main()
