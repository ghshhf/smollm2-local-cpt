#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""health_probe.py — 本地 SYSTEM 级定时探活 node1 训练实例。
机制: 经 ECS /task 派活 -> node1 跑 probe_health.sh 回显 -> 本地读回传判定存活。
判活: 回传含 "TOK=<本次ts>" 且 "alive"。
容错: 单次无回可能是 node1 队列积压(重任务期), 故连续 2 次无回才判 DEAD, 避免误报。
输出:
  - 最新状态写回 ECS node_health_latest.txt (对话侧随时可读)
  - 本地 D:\work\_local_ai\train_15b\node\health_state.json (容错)
  - DEAD 时额外写 alert (本地 + ECS node_dead_<ts>.txt)
"""
import json, time, urllib.request, os, re, sys

ECS      = "http://47.99.131.12:8081"
NODE     = "/mnt/workspace/node1"
STATE    = r"D:\work\_local_ai\train_15b\node\health_state.json"
ALERTDIR = r"D:\work\_local_ai\train_15b\node\alerts"


def post_task(body):
    try:
        req = urllib.request.Request(ECS + "/task", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=15)
        return True
    except Exception as e:
        sys.stderr.write("post err %r\n" % e)
        return False


def read_back(name):
    try:
        return urllib.request.urlopen(ECS + "/file/" + name, timeout=15).read().decode("utf-8", "replace")
    except Exception:
        return None


def upload(name, text):
    try:
        req = urllib.request.Request(ECS + "/upload?name=%s&key=k0u4mail" % name,
                                     data=text.encode("utf-8", "replace"),
                                     headers={"Content-Type": "text/plain; charset=utf-8"}, method="POST")
        urllib.request.urlopen(req, timeout=20)
    except Exception as e:
        sys.stderr.write("upload err %r\n" % e)


def main():
    ts  = int(time.time())
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    cmd = "bash %s/probe_health.sh %d" % (NODE, ts)

    post_task({"cmd": cmd, "id": "probe_%d" % ts, "ts": ts})
    time.sleep(75)
    resp = read_back("probe_%d.txt" % ts)
    alive = (resp is not None) and ("alive" in resp) and ("TOK=%d" % ts in resp)

    state = {"last_ok": now, "fail_streak": 0, "status": "GREEN", "done_steps": "?", "detail": ""}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass

    done = "?"
    if resp:
        m = re.search(r"done_steps=(\d+)", resp)
        if m:
            done = m.group(1)

    if alive:
        state["last_ok"] = now
        state["fail_streak"] = 0
        state["status"] = "GREEN"
        state["done_steps"] = done
    else:
        state["fail_streak"] = int(state.get("fail_streak", 0)) + 1
        if state["fail_streak"] >= 2:
            state["status"] = "DEAD"
            msg = ("NODE1_DEAD at %s (fail_streak=%d)\n"
                   "请到 ModelScope 控制台粘贴 bootstrap 三行重启:\n"
                   "import urllib.request\n"
                   "urllib.request.urlretrieve('http://47.99.131.12:8081/file/bootstrap_all.py?key=k0u4mail','/mnt/workspace/node1/bootstrap_all.py')\n"
                   "exec(open('/mnt/workspace/node1/bootstrap_all.py').read())" % (now, state["fail_streak"]))
            try:
                os.makedirs(ALERTDIR, exist_ok=True)
                open(os.path.join(ALERTDIR, "NODE_DEAD_%d.txt" % ts), "w", encoding="utf-8").write(msg)
            except Exception as e:
                sys.stderr.write("alert local err %r\n" % e)
            upload("node_dead_%d.txt" % ts, msg)
        else:
            state["status"] = "UNSURE"
    state["detail"] = "alive" if alive else "no-response"

    try:
        json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:
        sys.stderr.write("state write err %r\n" % e)

    latest = "STATUS=%s\nTS=%s\nDONE_STEPS=%s\nFAIL_STREAK=%s\nLAST_OK=%s" % (
        state["status"], now, done, state["fail_streak"], state["last_ok"])
    upload("node_health_latest.txt", latest)
    print("probe done: status=%s done=%s" % (state["status"], done))


if __name__ == "__main__":
    main()
