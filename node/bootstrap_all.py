#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_all.py — node1(魔搭) 一键恢复

背景：魔搭免费实例会被回收/重启，容器内所有进程(训练链 + 三个守护)全部消失，
      但 /mnt/workspace 是 NAS 持久盘，权重与语料都在，只需重新拉起进程。

用法 —— Jupyter(Python 格) 贴这一行即可，永久可复用：
    !curl -s -o /mnt/workspace/node1/bootstrap_all.py "http://47.99.131.12:8081/file/bootstrap_all.py?key=k0u4mail" && exec(open("/mnt/workspace/node1/bootstrap_all.py").read())

幂等：重复执行安全（已存在的进程不重复拉起，缺失的资产自动从 ECS 补）。
自回报：执行完自动把诊断结果传回 ECS，跑完我这边能看到。
"""
import os, subprocess, time

W   = "/mnt/workspace/node1"
CH  = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
REP = []


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


def say(m):
    print(m, flush=True)
    REP.append(str(m))


def alive(pat):
    r = sh("pgrep -f '%s'" % pat).stdout.strip()
    return r


def ensure(path, name, min_bytes=300):
    """本地缺失或过小 -> 从 ECS 拉（NAS 是持久盘通常都在，这是兜底）"""
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return "OK(%.1fKB)" % (os.path.getsize(path) / 1024.0)
    url = "%s/file/%s?key=%s" % (ECS, name, KEY)
    sh('curl -s -m 180 -o "%s" "%s"' % (path, url))
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return "已从ECS补(%dB)" % os.path.getsize(path)
    return "!! 补齐失败"


say("=== node1 恢复 @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===")
say("uptime: " + " ".join(sh("uptime").stdout.split()[-3:]))
_ds = CH + "/done_steps.txt"
say("done_steps = " + (open(_ds).read().strip() if os.path.exists(_ds) else "?"))
if os.path.exists(CH + "/stop"):
    say("!! 发现 stop 标记，先清除")
    os.remove(CH + "/stop")
if os.path.exists(CH + "/chain.pid"):
    say("旧 chain.pid = " + open(CH + "/chain.pid").read().strip() + "（已死进程，guard 会自动放行）")

# ---------- 0) 关键资产兜底 ----------
say("--- 资产 ---")
for p in ["chain_v1.sh", "chain_v2.sh", "train_remote_v4.py",
          "expand_300m_node1.py", "chain_watch.sh"]:
    say("  %-24s %s" % (p, ensure(os.path.join(CH, p), p)))
for p in ["watchd.sh", "maild.sh"]:
    say("  %-24s %s" % (p, ensure(os.path.join(W, p), p)))

# ---------- 1) 派活通道 node_v5.py（主） ----------
v5 = None
for c in [W + "/node_v5.py", W + "/gh_repo/node_v5.py", CH + "/node_v5.py"]:
    if os.path.exists(c):
        v5 = c
        break
if v5 is None:
    v5 = W + "/node_v5.py"
    say("  node_v5.py          " + ensure(v5, "node_v5.py", 1000))
if alive("node_[v]5.py"):
    say("node_v5.py  : 已在运行")
else:
    sh("cd %s && setsid nohup python3 -u %s > %s/node_v5.log 2>&1 < /dev/null &" % (W, v5, W))
    time.sleep(3)
    say("node_v5.py  : 已拉起 -> " + (alive("node_[v]5.py") or "失败"))

# ---------- 2) watchd 派活旁路 ----------
if alive("watch[d].sh"):
    say("watchd.sh   : 已在运行")
elif os.path.exists(W + "/watchd.sh"):
    sh("setsid nohup bash %s/watchd.sh > %s/watchd.log 2>&1 < /dev/null &" % (W, W))
    time.sleep(2)
    say("watchd.sh   : 已拉起 -> " + (alive("watch[d].sh") or "失败"))
else:
    say("watchd.sh   : 无（node_v5.py 主通道已足够）")

# ---------- 3) maild 信标 ----------
if alive("mail[d].sh"):
    say("maild.sh    : 已在运行")
elif os.path.exists(W + "/maild.sh"):
    sh("setsid nohup bash %s/maild.sh > %s/maild.log 2>&1 < /dev/null &" % (W, W))
    time.sleep(2)
    say("maild.sh    : 已拉起 -> " + (alive("mail[d].sh") or "失败"))
else:
    say("maild.sh    : 无")

# ---------- 4) 训练链（自动从 done_steps 续训；R6 满 6800 后 exec chain_v2.sh 进 0.3B） ----------
if alive("chain_[v]1.sh") or alive("chain_[v]2.sh"):
    say("训练链      : 已在运行 -> " + (alive("chain_[v]1.sh") or alive("chain_[v]2.sh")))
else:
    lg = CH + "/chain_resume_" + time.strftime("%H%M%S") + ".log"
    sh("cd %s && setsid nohup sh %s/chain_v1.sh > %s 2>&1 < /dev/null &" % (CH, CH + "/chain_v1.sh", lg))
    time.sleep(8)
    say("训练链      : 已拉起 log=%s" % lg)
    say("             -> " + (alive("chain_[v]1.sh") or "启动中/失败, 看日志"))

# ---------- 5) cache 提示 ----------
if not os.path.exists("/root/c_cache_R6.npy"):
    say("提示: /root 是容器本地盘(重启已清空)。若 NAS 上无 cache_R6.npy，"
        "trainer 会重新 tokenize R6 语料(272MB，需几分钟)，属正常。")

time.sleep(2)
say("--- 当前进程 ---")
say(sh("ps -eo pid,etime,args | grep -E 'train_[r]emote|chain_[v]|node_[v]5|watch[d]|mail[d]' "
       "| head -8").stdout.strip() or "(空)")
say("=== bootstrap 结束 ===")

# 自回报
try:
    open("/tmp/boot_report.txt", "w").write("\n".join(REP))
    sh('curl -s -m 40 --data-binary @/tmp/boot_report.txt "%s/upload?name=boot_r1.txt&key=%s"'
       % (ECS, KEY))
    print("[已回报 boot_r1.txt]")
except Exception as e:
    print("[回报失败] %s" % e)