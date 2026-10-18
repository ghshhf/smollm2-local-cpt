# -*- coding: utf-8 -*-
"""node_v6.py — ModelScope notebook node agent v6 (fast beacon protocol over ECS)

相对 v5 的升级:
  1. POLL_S 10s -> 1.0s, 且领到任务后立即 continue 连续抽干队列(0 间隔)。
     这是回传慢的主因: v5 平均白等 5s。
  2. 单实例文件锁 /tmp/node_v6.lock — 防止两个轮询器同时抢任务、重复执行。
     (v5 起了两份, 直接导致同一任务被执行两次)
  3. 幂等: 同一个 task_id 只执行一次(内存 + 最近 200 条落盘), 服务端重复投递也不会重复跑。
  4. keepalive 让路: 检测到 GPU 自驾锁(/tmp/autopilot5.lock 或 autopilot4)时不烧 CPU,
     免得 8 核 45 秒的保活把 dataloader 的 CPU 抢走。保活只在本机真空闲时做。
  5. 支持 long-poll: 先试 GET /task?wait=20, 若服务端不支持(秒回空)自动降级为 1s 轮询。

Protocol:
  GET  /task      -> fetch & consume next task  {id, cmd}
  POST /result    -> log result server-side {node, task_id, code, out}
  POST /mail      -> mirror results/heartbeat/online to mailbox
  GET  /file/<n>  -> download whitelisted file

Launch: setsid nohup python3 node_v6.py > node_v6.log 2>&1 < /dev/null &
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

ECS = "http://47.99.131.12:8081"
NODE = "node1"
WORKDIR = "/mnt/workspace/node1"
POLL_S = 1.0
POLL_S_FAST = 0.25
HB_S = 300
MAX_OUT = 30000
TASK_TIMEOUT = 900
KEEPALIVE_S = 600
BURN_S = 45
LONGPOLL = True
LONGPOLL_WAIT = 20
LOCK = "/tmp/node_v6.lock"
SEEN = "/tmp/node_v6_seen.json"
GPU_LOCKS = ("/tmp/autopilot4.lock", "/tmp/autopilot5.lock", "/tmp/autopilot3.lock")

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


def load_seen():
    try:
        return set(json.load(open(SEEN)))
    except Exception:
        return set()


def save_seen(s):
    try:
        json.dump(list(s)[-200:], open(SEEN, "w"))
    except Exception:
        pass


def gpu_busy():
    return any(os.path.exists(p) for p in GPU_LOCKS)


def run_task(tid, cmd):
    log(f"task {tid}: {cmd[:120]}")
    t0 = time.time()
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
    log(f"task {tid} done code={code} in {time.time()-t0:.1f}s")
    try:
        http_json(f"{ECS}/result", {"node": NODE, "task_id": tid, "code": code, "out": out},
                  method="POST", timeout=60)
    except Exception as e:
        log(f"/result failed: {e}")
    try:
        mail(json.dumps({"type": "result", "task_id": tid, "code": code, "out": out[:8000]},
                        ensure_ascii=False))
    except Exception as e:
        log(f"mirror failed: {e}")


def keepalive():
    if gpu_busy():
        log("keepalive skipped (GPU 自驾在跑, 不抢 CPU)")
        return
    log("idle -> keepalive burn")
    try:
        p = subprocess.run([sys.executable, "-c", KEEPALIVE_CODE],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=BURN_S * 4)
        log(p.stdout.decode(errors="replace").strip()[:120])
    except Exception as e:
        log(f"keepalive failed: {e}")


def acquire_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            os.kill(pid, 0)
            log(f"已有 node_v6 pid={pid} 在跑, 退出(防双轮询)")
            sys.exit(0)
        except Exception:
            pass
    open(LOCK, "w").write(str(os.getpid()))


def main():
    acquire_lock()
    log(f"node_v6 up, ecs={ECS}, poll={POLL_S}s longpoll={LONGPOLL}")
    try:
        mail(json.dumps({"type": "online", "ver": 6, "ts": int(time.time())}))
    except Exception as e:
        log(f"online post failed: {e}")

    seen = load_seen()
    last_hb = 0.0
    last_work = time.time()
    lp_ok = LONGPOLL
    while True:
        t_poll = time.time()
        got = False
        try:
            url = f"{ECS}/task" + (f"?wait={LONGPOLL_WAIT}" if lp_ok else "")
            r = http_json(url, timeout=LONGPOLL_WAIT + 20 if lp_ok else 40)
            if lp_ok and time.time() - t_poll < 0.3 and not r.get("id"):
                lp_ok = False
                log("服务端不支持 long-poll(秒回空) -> 降级 1s 轮询")
            if r.get("id") and "cmd" in r:
                tid = str(r["id"])
                if tid in seen:
                    log(f"task {tid} 重复投递, 跳过")
                else:
                    seen.add(tid); save_seen(seen)
                    run_task(tid, r["cmd"])
                    got = True
                last_work = time.time()
                continue                      # 连续抽干队列, 不 sleep
        except urllib.error.HTTPError as e:
            if e.code not in (404, 502, 503):
                log(f"task poll http {e.code}")
            time.sleep(POLL_S)
        except Exception as e:
            log(f"task poll error: {e}")
            time.sleep(POLL_S)
        if time.time() - last_work > KEEPALIVE_S:
            keepalive()
            last_work = time.time()
        if time.time() - last_hb > HB_S:
            last_hb = time.time()
            try:
                mail(json.dumps({"type": "alive", "ts": int(time.time())}))
            except Exception:
                pass
        if not got:
            time.sleep(POLL_S_FAST if time.time() - last_work < 60 else POLL_S)


if __name__ == "__main__":
    main()
