#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_all.py — node1(魔搭) 一键恢复 v2
（node_v6 拉取器 + 训练链看门狗 + 做梦集群 + 自进化闭环，全部幂等，缺失资产自动从 ECS 补，自回报）

背景：魔搭免费实例会被回收/重启，容器内所有进程(训练链 + 派活器 + 做梦集群)全部消失，
      但 /mnt/workspace 是 NAS 持久盘，权重与语料都在，只需重新拉起进程。

用法 —— 魔搭终端贴这一行即可（永久可复用，主控侧可随时改 ECS 上的本文件）：
    curl -s -o /mnt/workspace/node1/bootstrap_all.py "http://47.99.131.12:8081/file/bootstrap_all.py?key=k0u4mail" && cd /mnt/workspace/node1 && python3 bootstrap_all.py

幂等：重复执行安全（已存在的进程不重复拉起，缺失的资产自动从 ECS 补）。
自回报：执行完自动把诊断结果传回 ECS（boot_rX.txt），主控侧可读。
"""
import os, subprocess, time

W   = "/mnt/workspace/node1"
D   = W + "/dream"
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
    return sh("pgrep -f '%s'" % pat).stdout.strip()


def ensure(path, name, min_bytes=300):
    """本地缺失或过小 -> 从 ECS 拉（NAS 是持久盘通常都在，这是兜底）"""
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return "OK(%.1fKB)" % (os.path.getsize(path) / 1024.0)
    url = "%s/file/%s?key=%s" % (ECS, name, KEY)
    sh('curl -s -m 180 -o "%s" "%s"' % (path, url))
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return "已从ECS补(%dB)" % os.path.getsize(path)
    return "!! 补齐失败"


def find_lora():
    for cand in [D + "/models/evolve_lora"]:
        if os.path.isdir(cand):
            return cand
    return None


say("=== node1 恢复 v2 @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===")
say("uptime: " + " ".join(sh("uptime").stdout.split()[-3:]))
_ds = CH + "/done_steps.txt"
say("done_steps = " + (open(_ds).read().strip() if os.path.exists(_ds) else "?"))
if os.path.exists(CH + "/stop"):
    say("!! 发现 stop 标记，先清除")
    os.remove(CH + "/stop")

# ---------- 0) 关键资产兜底 ----------
say("--- 资产 ---")
for p in ["node_v6.py", "dream_daemon.sh", "evolve_loop.sh", "evolve_cycle.sh"]:
    say("  %-22s %s" % (p, ensure(os.path.join(W, p), p)))
# tunnel_up.sh 位于 dream/ 下，但 ECS 上的文件名是裸名（url 不能带子目录）
say("  %-22s %s" % ("dream/tunnel_up.sh",
                    ensure(os.path.join(D, "tunnel_up.sh"), "tunnel_up.sh")))

# ---------- 1) 派活通道 node_v6.py（主，主控侧远程指挥命脉） ----------
if alive("node_v[56].py"):
    say("node_v6.py  : 已在运行")
else:
    sh("cd %s && setsid nohup python3 -u %s > %s/node_v6.log 2>&1 < /dev/null &" % (W, W + "/node_v6.py", W))
    time.sleep(4)
    say("node_v6.py  : 已拉起 -> " + (alive("node_v[56].py") or "失败"))

# ---------- 1b) SSH 反向隧道（本机 n1.sh 直连命脉；2026-10-19 补，此前缺失导致恢复后仍直连不上） ----------
_tu = os.path.join(D, "tunnel_up.sh")
if not os.path.exists(_tu):
    say("tunnel      : !! 缺 %s" % _tu)
else:
    sh("bash %s" % _tu)
    time.sleep(5)
    _st = sh("bash %s --status" % _tu).stdout.strip()
    say("tunnel      : " + (_st.splitlines()[0] if _st else "无输出"))

# ---------- 2) 训练链看门狗已删除(2026-10-19 用户裁定: R7 属 0.2B 期过时链) ----------

# ---------- 3) 做梦集群（RSI 自进化闭环） ----------
ENABLE_DREAM = 0   # 2026-10-19 起本阶段关闭，专供 0.4B 数据准备
lora = find_lora()
say("evolve_lora : " + (lora or "无(裸模型模式复产)"))

# 3a) dream daemon（8 worker，携带已 ADOPT 的适配器）
if not ENABLE_DREAM:
    say("dream_daemon: 已停用(ENABLE_DREAM=0)")
elif alive("dream_[d]aemon.sh"):
    say("dream_daemon: 已在运行")
else:
    lora_arg = ("DREAM_LORA=%s " % lora) if lora else ""
    sh("cd %s && DREAM_WORKERS=4 %ssetsid nohup bash dream_daemon.sh >> logs/daemon.log 2>&1 < /dev/null &"
       % (D, lora_arg))
    time.sleep(3)
    say("dream_daemon: 已拉起 -> " + (alive("dream_[d]aemon.sh") or "失败"))

# 3b) evolve_loop（自进化常驻触发器，闸门自保）
if not ENABLE_DREAM:
    say("evolve_loop : 已停用(ENABLE_DREAM=0)")
elif alive("evolve_[l]oop.sh"):
    say("evolve_loop : 已在运行")
else:
    sh("cd %s && setsid nohup bash evolve_loop.sh >> logs/evolve_loop.log 2>&1 < /dev/null &" % D)
    time.sleep(2)
    say("evolve_loop : 已拉起 -> " + (alive("evolve_[l]oop.sh") or "失败"))

time.sleep(15)
say("--- 当前进程 ---")
say(sh("ps -eo pid,etime,args | grep -E 'node_v[0-9]|chain_[vw]|dream_[dl]|evolve_[lc]|dream_loop' | grep -v grep | head -12").stdout.strip() or "(空)")
say("DONE=%s BANK=%s" % (
    (open(_ds).read().strip() if os.path.exists(_ds) else "?"),
    sh("python3 -c \"import sqlite3;c=sqlite3.connect('%s/data/triple_bank.db',timeout=30);c.execute('PRAGMA busy_timeout=30000');print(c.execute('select count(*) from triples').fetchone()[0],c.execute('select count(distinct code_text) from triples').fetchone()[0])\"" % D).stdout.strip()))
say("=== bootstrap 结束 ===")

# 自回报（文件名带时间戳轮换——ECS 同名文件会读到旧快照，绝不复用固定名）
try:
    tag = time.strftime("boot_%H%M%S")
    open("/tmp/%s.txt" % tag, "w").write("\n".join(REP))
    sh('curl -s -m 40 --data-binary @/tmp/%s.txt "%s/upload?name=%s.txt&key=%s"' % (tag, ECS, tag, KEY))
    print("[已回报 %s.txt]" % tag)
except Exception as e:
    print("[回报失败] %s" % e)
