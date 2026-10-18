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
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
DREAM_DIR = os.path.join(ROOT, "data", "dreams")
LOCK = os.path.join(DREAM_DIR, ".autorun_lock")
TRAIN_LOCK = os.path.join(ROOT, ".train_lock")


def pid_alive(pid):
    """跨进程探活：Windows 用 OpenProcess，其它平台用 kill(pid,0)。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def training_running():
    """训练锁存在且持有者 pid 仍存活才算在跑（stale 锁不挡路）。

    宪法第一定律：一次只跑一个训练。做梦是 CPU 密集推理，必须让路。
    """
    try:
        with open(TRAIN_LOCK, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None
    pid = j.get("pid")
    if pid and pid_alive(pid):
        return j
    return None

# 任务模板（带随机参数微扰，使产物持续有新变体注入银行）
# 结构: (skill, task 模板, hint)
#   skill = 技能轴，显式标注（生成端知道在练什么，别让精英档案的正则去猜）。
#   第一版全是 matplotlib 数学曲线，精英档案实测塌成 1 个桶 —— 根因是任务域太单一。
#   这里按技能轴铺开：三角波/多序列/衰减/参数/极坐标/曲面三维/场热力/级数/粒子/排名/分形
#   + 跨域任务（排序可视化、生命游戏、刚体动画），让 MAP-Elites 网格真正有格子可填。
TEMPLATES = [
    ("三角波", "用 matplotlib 画一个余弦波动画(0~{A}π, {F} 帧)，保存为 out.mp4",
     "把 sin 换成 cos；from matplotlib.animation import FuncAnimation; anim.save('out.mp4', writer='ffmpeg')"),
    ("多序列对比", "用 matplotlib 画正弦+余弦叠加动画(0~{A}π, {F} 帧)，保存为 out.mp4",
     "两条曲线 y1=sin(x), y2=cos(x) 同一张图；FuncAnimation + save ffmpeg"),
    ("衰减振荡", "用 matplotlib 画衰减正弦波动画(0~{A}π, {F} 帧，振幅随时间衰减)，保存为 out.mp4",
     "y = sin(x) * exp(-0.05*x)；FuncAnimation + save ffmpeg"),
    ("多序列对比", "用 matplotlib 画多曲线对比动画(正弦/方波/三角波各一条, 0~{A}π, {F} 帧)，保存为 out.mp4",
     "numpy 造 square/triangle；ax.plot 三条；FuncAnimation + save ffmpeg"),
    ("参数曲线", "用 matplotlib 画李萨如(Lissajous)曲线动画(参数 m,n 随机), {F} 帧，保存为 out.mp4",
     "x=sin(m*t), y=sin(n*t)；FuncAnimation + save ffmpeg"),
    ("极坐标曲线", "用 matplotlib 画旋转螺旋线动画({F} 帧)，保存为 out.mp4",
     "极坐标 r=t, theta=t；转直角坐标 plot；FuncAnimation + save ffmpeg"),
    ("曲面三维", "用 matplotlib 画 3D 曲面旋转动画(0~{A}π, {F} 帧)，保存为 out.mp4",
     "from mpl_toolkits.mplot3d import Axes3D; ax = fig.add_subplot(111, projection='3d'); "
     "Z = sin(sqrt(X**2+Y**2))；FuncAnimation 每帧 ax.view_init 转视角；save ffmpeg"),
    ("场热力图", "用 matplotlib 画二维波动方程热力图动画({F} 帧)，保存为 out.mp4",
     "ax.imshow(Z, cmap='viridis', animated=True)；每帧更新 Z；FuncAnimation + save ffmpeg；"
     "不要用 ax.clear()，用 set_data 更稳"),
    ("级数逼近", "用 matplotlib 画傅里叶级数逼近方波动画(前 {A} 项, {F} 帧)，保存为 out.mp4",
     "y = sum(sin((2k-1)x)/(2k-1) for k in 1..N)；每帧增加项数看逼近过程；save ffmpeg"),
    ("粒子统计", "用 matplotlib 画粒子布朗运动散点动画({A}00 个粒子, {F} 帧)，保存为 out.mp4",
     "np.cumsum(np.random.randn(N,2)*0.1, axis=0)；ax.scatter；FuncAnimation；save ffmpeg"),
    ("排名条形图", "用 matplotlib 画条形图竞赛动画(10 根柱子随机排名变化, {F} 帧)，保存为 out.mp4",
     "每帧 np.random 扰动数值后排序重绘 ax.barh；barh 要 set_width 而不是 clear；save ffmpeg"),
    ("分形", "用 matplotlib 画 Julia 集缩放动画({F} 帧)，保存为 out.mp4",
     "复平面 z=z^2+c 迭代 80 次，ax.imshow 显示逃逸时间；每帧缩小视野；save ffmpeg"),
    ("排序搜索", "用 matplotlib 把冒泡排序的每一步画成柱状图动画({A}0 个随机数, {F} 帧)，保存为 out.mp4",
     "每帧做一次相邻比较交换并 ax.bar 重绘；FuncAnimation；save ffmpeg"),
    ("生命游戏", "实现康威生命游戏(Game of Life)并演化 {F} 步，把过程录成 out.mp4",
     "numpy 20x20 随机活细胞；用卷积或邻域求和更新；ax.imshow 每帧 set_data；save ffmpeg"),
    ("刚体动画", "用 matplotlib 画小球抛体运动+反弹动画({F} 帧)，保存为 out.mp4",
     "y = v*t - 0.5*g*t**2，触地反弹速度反向衰减；ax.plot 或 scatter 更新；save ffmpeg"),
]


def make_spec():
    skill, tpl, hint = random.choice(TEMPLATES)
    a = random.choice([2, 3, 4, 5, 6]) if "{A}" in tpl else 0
    f = random.choice([80, 100, 120, 150])
    task = tpl.format(A=a, F=f)
    return {"skill": skill, "domain": skill, "verify": "video", "task": task, "hint": hint}


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

    tr = training_running()
    if tr:
        print("[skip] 检测到训练在跑 (pid=%s job=%s)，本周期让路"
              "（第一定律：一次只跑一个训练）" % (tr.get("pid"), tr.get("job")))
        lf.close()
        return

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
