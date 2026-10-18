#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_autorun.py — A 路做梦 RSI 周期自转调度器（由 schtasks /ru SYSTEM 调用）

与 dream_loop.py 解耦的职责：
  - 原子单实例锁：上轮没跑完就跳过本周期，绝不叠加吃 CPU（与 CodeLoraP2 训练错峰）。
  - 自管 utf-8 日志到 logs/dream_autorun_YYYY-MM-DD.log（SYSTEM 账户无控制台，只读日志）。
  - 随机任务微扰：从 matplotlib 变体模板随机抽 + 随机参数，使每轮产物略不同，
    持续丰富三元组银行 → few-shot 检索持续增强（RSI 自我改良的核心燃料）。
  - 同进程 import dream_loop.Dreamer，模型只加载一次，跑 --rounds 轮后干净退出。

宪法红线：只跑 coder 底座(0.5B Qwen + 代码LoRA)产已验证代码 → 回灌银行；北极星做代码
任务时从同一银行检索 few-shot（A 路设计）。绝不碰 0.3B 底座写代码（实测 0% pass）。
"""
import argparse
import datetime
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
DREAM_DIR = os.path.join(ROOT, "data", "dreams")
LOCK = os.path.join(DREAM_DIR, ".autorun_lock")

# 任务模板（带随机参数微扰，使产物持续有新变体注入银行）
TEMPLATES = [
    ("UI可视化", "用 matplotlib 画一个余弦波动画(0~{A}π, {F} 帧)，保存为 out.mp4",
     "把 sin 换成 cos；from matplotlib.animation import FuncAnimation; anim.save('out.mp4', writer='ffmpeg')"),
    ("UI可视化", "用 matplotlib 画正弦+余弦叠加动画(0~{A}π, {F} 帧)，保存为 out.mp4",
     "两条曲线 y1=sin(x), y2=cos(x) 同一张图；FuncAnimation + save ffmpeg"),
    ("UI可视化", "用 matplotlib 画衰减正弦波动画(0~{A}π, {F} 帧，振幅随时间衰减)，保存为 out.mp4",
     "y = sin(x) * exp(-0.05*x)；FuncAnimation + save ffmpeg"),
    ("UI可视化", "用 matplotlib 画多曲线对比动画(正弦/方波/三角波各一条, 0~{A}π, {F} 帧)，保存为 out.mp4",
     "numpy 造 square/triangle；ax.plot 三条；FuncAnimation + save ffmpeg"),
    ("UI可视化", "用 matplotlib 画李萨如(Lissajous)曲线动画(参数 m,n 随机), {F} 帧，保存为 out.mp4",
     "x=sin(m*t), y=sin(n*t)；FuncAnimation + save ffmpeg"),
    ("游戏模拟", "用 matplotlib 画旋转螺旋线动画({F} 帧)，保存为 out.mp4",
     "极坐标 r=t, theta=t；转直角坐标 plot；FuncAnimation + save ffmpeg"),
]


def make_spec():
    domain, tpl, hint = random.choice(TEMPLATES)
    a = random.choice([2, 3, 4, 5, 6]) if "{A}" in tpl else 0
    f = random.choice([80, 100, 120, 150])
    task = tpl.format(A=a, F=f)
    return {"domain": domain, "verify": "video", "task": task, "hint": hint}


def log_path():
    return os.path.join(LOGS, "dream_autorun_%s.log" % datetime.date.today().isoformat())


def acquire_lock():
    # NTFS 跨进程原子创建（O_EXCL）：若已存在说明上轮还在跑
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(DREAM_DIR, exist_ok=True)
    # SYSTEM 账户无控制台：stdout/stderr 全部重定向到 utf-8 日志
    lf = open(log_path(), "a", encoding="utf-8", buffering=1)
    sys.stdout = lf
    sys.stderr = lf
    print("\n===== DreamLoop autorun @ %s =====" % datetime.datetime.now().isoformat())

    if not acquire_lock():
        print("[lock] 上轮仍在运行，跳过本周期（防 CPU 叠加）")
        lf.close()
        return
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--base", required=True)
        ap.add_argument("--lora", default=None)
        ap.add_argument("--rounds", type=int, default=2)
        a = ap.parse_args()

        sys.path.insert(0, HERE)
        import dream_loop as dl

        d = dl.Dreamer(a.base, a.lora, use_latent=False, steps=8, lr=0.05)
        stats = {"pass": 0, "fail": 0}
        for i in range(a.rounds):
            spec = make_spec()
            ok = d.one_dream(spec, use_memory=True)
            stats["pass" if ok else "fail"] += 1
        print("[autorun] 本轮结束 pass=%d fail=%d" % (stats["pass"], stats["fail"]))
    except Exception as e:
        import traceback
        print("[autorun] 异常: %s" % e)
        traceback.print_exc()
    finally:
        release_lock()
        lf.close()


if __name__ == "__main__":
    main()
