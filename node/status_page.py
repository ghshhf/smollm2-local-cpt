#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""status_page.py — node1 训练监控常驻服务（本机 8082）。
- HTTP 线程：GET / 渲染监控 HTML（读共享 train_status.json）
- 主循环：每 60s 经 ECS /task 派轻量查询拿 chain.log tail/done_steps/进程数，写共享 json
- 单实例：启动时 8082 已被监听则退出（供 schtasks 每分钟重拉时幂等）
- 数据源铁律：ECS 同名文件锁 -> 输出文件名按计数持续轮换
"""
import json
import os
import re
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_JSON = os.path.join(BASE, "train_status.json")
CNT_FILE = os.path.join(BASE, ".status_poll_cnt")
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
HEALTH_JSON = os.path.join(BASE, "health_state.json")  # health_probe 写的
POLL_EVERY = 60
R6_TARGET = 6800  # 封顶步数；超过后页面自动切到 0.3B 显示

DEFAULT_STATE = {
    "ok": False, "poll_ts": 0, "done_steps": 0, "chunk_step": 0, "chunk_total": 400,
    "loss": None, "lr": None, "tok_s": None, "sec_per_step": None, "eta_s": None,
    "train_procs": 0, "health": "UNKNOWN", "health_ts": "", "fail_streak": -1,
    "raw_tail": "", "note": "booting",
}


def next_name():
    n = 0
    try:
        n = int(open(CNT_FILE).read().strip())
    except Exception:
        pass
    n += 1
    with open(CNT_FILE, "w") as f:
        f.write(str(n))
    return "train_poll_%03d.txt" % n


def poll_once():
    """派一条轻量查询到 node1，解析回传，更新共享 json。"""
    st = dict(DEFAULT_STATE)
    try:
        st.update(json.load(open(STATE_JSON, encoding="utf-8")))
    except Exception:
        pass
    st["poll_ts"] = int(time.time())
    name = next_name()
    cmd = ("{ date +%%s; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; "
           "tail -2 /mnt/workspace/node1/chain/chain.log 2>/dev/null; "
           "pgrep -c -f train_remote_v4 2>/dev/null || echo 0; } 2>&1 | "
           "curl -s -m60 -X POST 'http://47.99.131.12:8081/upload?name=%s&key=%s' --data-binary @-" % (name, KEY))
    try:
        body = json.dumps({"cmd": cmd, "id": "status-poll-" + name, "ts": int(time.time())}).encode()
        req = urllib.request.Request(ECS + "/task", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        st["note"] = "dispatch failed: %s" % e
        _save(st)
        return
    # 等 node_v5 轮询执行 + 回传落地（轮询 10s 级，等 25s）
    txt = ""
    for _ in range(6):
        time.sleep(5)
        try:
            txt = urllib.request.urlopen(
                ECS + "/file/%s?key=%s" % (name, KEY), timeout=30).read().decode("utf-8", "replace")
        except Exception:
            txt = ""
        if txt.strip():
            break
    st["raw_tail"] = txt[-600:]
    _parse(txt, st)
    _save(st)


def _parse(txt, st):
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    if not lines:
        st["note"] = "no reply"
        return
    # 第一行是 date +%s 的 unix ts；done_steps；chain.log tail 2 行；进程数
    for l in lines:
        if re.match(r"^\d{9,10}$", l.strip()) and "now_ts" not in st:
            st["now_ts"] = int(l.strip())
    m = re.search(r"^(\d{3,6})\s*$", txt, re.M)
    if m:
        st["done_steps"] = int(m.group(1))
    for l in lines:
        mm = re.search(r"step (\d+)/(\d+) loss=([\d.]+) lr=([\d.e-]+) t=(\d+)s tok/s=(\d+)", l)
        if mm:
            st["chunk_step"] = int(mm.group(1))
            st["chunk_total"] = int(mm.group(2))
            st["loss"] = float(mm.group(3))
            st["lr"] = mm.group(4)
            st["tok_s"] = int(mm.group(6))
            t_now = int(mm.group(5))
            st["sec_per_step"] = round(t_now / max(1, st["chunk_step"]), 2)
    if re.search(r"^0\s*$", txt, re.M) or "NO_TRAIN" in txt:
        pass
    # 进程数：最后一段独立小数字（chain.log 行之后）
    nums = [int(x) for x in re.findall(r"^(\d+)\s*$", txt, re.M)]
    big = [n for n in nums if n < 100]
    if big:
        st["train_procs"] = big[-1] if len(big) >= 1 else 0
    # 健康态来自 health_probe 的 health_state.json
    try:
        h = json.load(open(HEALTH_JSON, encoding="utf-8"))
        st["health"] = h.get("status", "UNKNOWN")
        st["health_ts"] = h.get("last_ok", "")
        st["fail_streak"] = h.get("fail_streak", -1)
    except Exception:
        pass
    # ETA：剩余 chunk 步数 * 每步秒数
    if st.get("sec_per_step") and st.get("chunk_total"):
        remain = max(0, st["chunk_total"] - st.get("chunk_step", 0))
        st["eta_s"] = int(remain * st["sec_per_step"])
    st["note"] = "ok"


def _save(st):
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATE_JSON)


def _fmt_dur(s):
    if s is None:
        return "-"
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return ("%dh%02dm" % (h, m)) if h else ("%dm%02ds" % (m, s))


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>node1 训练监控</title>
<style>
body{background:#0f1419;color:#e6e1cf;font-family:Consolas,'Courier New',monospace;margin:0;padding:24px}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:18px;font-weight:600;color:#f0c674;margin:0 0 4px}
.sub{color:#7a8599;font-size:12px;margin-bottom:20px}
.lamp{display:inline-block;width:10px;height:10px;border-radius:50%%;margin-right:6px;vertical-align:middle}
.GREEN{background:#3fd68f;box-shadow:0 0 6px #3fd68f}.STALE{background:#f0c674;box-shadow:0 0 6px #f0c674}
.DEAD{background:#e5566d;box-shadow:0 0 6px #e5566d}.UNKNOWN{background:#7a8599}
.bar{height:22px;background:#1c2330;border-radius:4px;overflow:hidden;margin:14px 0 6px;position:relative}
.fill{height:100%%;background:linear-gradient(90deg,#2d7dd2,#3fd68f);transition:width .5s}
.pct{position:absolute;inset:0;text-align:center;font-size:12px;line-height:22px;color:#fff;text-shadow:0 1px 2px #000}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}
.card{background:#1c2330;border:1px solid #2a3446;border-radius:6px;padding:10px 12px}
.k{color:#7a8599;font-size:11px;margin-bottom:4px}
.v{color:#e6e1cf;font-size:16px;font-weight:600}
.v.hl{color:#f0c674}
pre{background:#1c2330;border:1px solid #2a3446;border-radius:6px;padding:10px 12px;font-size:11px;
color:#9fb1c9;overflow-x:auto;margin-top:16px;white-space:pre-wrap}
.foot{color:#565f73;font-size:11px;margin-top:14px;line-height:1.7}
a{color:#5eaae0}
</style></head><body><div class="wrap">
<h1>node1 训练监控 <span class="lamp %(health)s"></span><span style="font-size:13px">%(health)s</span></h1>
<div class="sub">魔搭 node1 · 64核 · NAS /mnt/workspace · 数据经 ECS 47.99.131.12:8081 中转 · 页面 30s 自动刷新</div>
<div class="bar"><div class="fill" style="width:%(pct).2f%%"></div><div class="pct">%(done)s / %(target)s 步（%(pct).1f%%）</div></div>
<div class="grid">
<div class="card"><div class="k">当前 chunk</div><div class="v">%(chunk)s</div></div>
<div class="card"><div class="k">loss（最新）</div><div class="v hl">%(loss)s</div></div>
<div class="card"><div class="k">tok/s</div><div class="v">%(toks)s</div></div>
<div class="card"><div class="k">每步耗时</div><div class="v">%(sps)s</div></div>
<div class="card"><div class="k">lr</div><div class="v">%(lr)s</div></div>
<div class="card"><div class="k">预计本 chunk 跑完</div><div class="v hl">%(eta)s</div></div>
<div class="card"><div class="k">训练进程数</div><div class="v">%(procs)s</div></div>
<div class="card"><div class="k">探活失败累计</div><div class="v">%(fs)s</div></div>
</div>
<pre>%(tail)s</pre>
<div class="foot">
数据时间：%(upd)s · 回传：%(hts)s · 下次轮询：60s<br>
活文件直读（curl）：<a href="http://47.99.131.12:8081/file/node_health_latest.txt?key=%(key)s">node_health_latest.txt</a>
· 封顶 %(target)s 后自动 Net2Net 扩 0.3B(350M)，本页随之切换
</div>
</div></body></html>"""


def render():
    try:
        st = json.load(open(STATE_JSON, encoding="utf-8"))
    except Exception:
        st = dict(DEFAULT_STATE)
    done = st.get("done_steps") or 0
    target = R6_TARGET
    pct = min(100.0, 100.0 * done / target) if target else 0
    upd = time.strftime("%H:%M:%S", time.localtime(st.get("poll_ts") or time.time()))
    return HTML_PAGE % {
        "health": st.get("health", "UNKNOWN"),
        "done": done, "target": target, "pct": pct,
        "chunk": "step %s/%s" % (st.get("chunk_step", "?"), st.get("chunk_total", "?")),
        "loss": ("%.4f" % st["loss"]) if st.get("loss") is not None else "-",
        "toks": st.get("tok_s") or "-",
        "sps": (str(st.get("sec_per_step")) + "s") if st.get("sec_per_step") else "-",
        "lr": st.get("lr") or "-",
        "eta": _fmt_dur(st.get("eta_s")),
        "procs": st.get("train_procs", "-"),
        "fs": st.get("fail_streak", "-"),
        "tail": st.get("raw_tail", "").strip() or "(等待首次回传…)",
        "upd": upd, "hts": st.get("health_ts", "-"), "key": KEY,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html", "/status"):
            self.send_error(404)
            return
        body = render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def port_in_use(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def main():
    if port_in_use(8082):
        print("[status] 8082 already serving, exit (idempotent)")
        return
    if not os.path.exists(STATE_JSON):
        _save(dict(DEFAULT_STATE))
    t = threading.Thread(target=lambda: HTTPServer(("127.0.0.1", 8082), Handler).serve_forever(),
                         daemon=True)
    t.start()
    print("[status] serving http://localhost:8082  (poll every %ds)" % POLL_EVERY, flush=True)
    while True:
        try:
            poll_once()
        except Exception as e:
            print("[status] poll error: %s" % e, flush=True)
        time.sleep(POLL_EVERY)


if __name__ == "__main__":
    main()
