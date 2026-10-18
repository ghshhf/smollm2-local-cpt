# -*- coding: utf-8 -*-
"""watchdog_v2: 耐心盯 L3 精炼直到出结果, 完成即自动回传报告.
- 完成判定: 输出文件 >10MB 且大小连续 3 次不变 (或进程退出但文件已生成)
- 每 5 分钟回传心跳; 150 分钟超时兜底; 进程死且无产出则立即报 FAILED
"""
import os, subprocess, time

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
OUT = "/mnt/workspace/node1/corpus_04b/creative_refined/creative_articles_l3.jsonl"
LOG = "/mnt/workspace/node1/creative_fill.log"
PAT = "l3_creative_fill.py"
TIMEOUT = 150 * 60          # 150 分钟兜底
STABLE_NEED = 3             # 连续 3 次大小不变视为写完成
SLEEP = 60


def sh(c, t=30):
    try:
        return subprocess.run(["bash", "-c", c], capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception as e:
        return "ERR:%s" % e


def upload(name, text):
    try:
        subprocess.run(["curl", "-s", "-X", "POST", "%s/upload?name=%s&key=%s" % (ECS, name, KEY),
                        "--data-binary", text.encode("utf-8")], timeout=25)
    except Exception:
        pass


def snapshot():
    ps = sh("ps -eo pid,etime,time,pcpu,rss,cmd | grep '%s' | grep -v grep || echo NONE" % PAT)
    sz = os.path.getsize(OUT) if os.path.exists(OUT) else -1
    log = sh("tail -n 6 " + LOG) if os.path.exists(LOG) else "NO_LOG"
    return ps, sz, log


t0 = time.time()
last_hb = 0.0
last_sz = -1
stable = 0

while True:
    ps, sz, log = snapshot()
    now = time.time() - t0
    alive = ("NONE" not in ps)

    # ---- 完成判定 ----
    if sz > 10_000_000:
        stable = stable + 1 if sz == last_sz else 0
        last_sz = sz
        if stable >= STABLE_NEED or (not alive):
            upload("creative_fill_DONE2.txt",
                   "CREATIVE_FILL_DONE\n"
                   "输出: %s\n大小: %.1f MB\n总耗时: %.1f min\n"
                   "--- 日志尾部 ---\n%s" % (OUT, sz / 1e6, now / 60.0, sh("tail -n 25 " + LOG)))
            break

    # ---- 异常退出 ----
    if not alive and sz < 0:
        upload("creative_fill_DONE2.txt",
               "CREATIVE_FILL_FAILED\n进程已退出且无输出 (%.1f min)\n"
               "--- 日志 ---\n%s" % (now / 60.0, sh("tail -n 30 " + LOG)))
        break

    # ---- 心跳 ----
    if time.time() - last_hb > 300:
        last_hb = time.time()
        upload("creative_wd2_HB.txt",
               "WD2 @%.0f min | alive=%s | out_size=%s\n"
               "--- proc ---\n%s\n--- log ---\n%s" % (now / 60.0, alive, sz, ps, log))

    # ---- 超时兜底 ----
    if now > TIMEOUT:
        upload("creative_fill_DONE2.txt",
               "CREATIVE_FILL_TIMEOUT (150min)\nout_size=%s\n--- proc ---\n%s\n"
               "--- log ---\n%s" % (sz, ps, sh("tail -n 25 " + LOG)))
        break

    time.sleep(SLEEP)
