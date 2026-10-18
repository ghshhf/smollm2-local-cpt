#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dream_seeds.py — 做梦的「规范教材」：15 个模板的人工骨架

🔴 为什么必须有这个（2026-10-18）：
  装上 task_verify 回扫后发现 triple_bank 里 7 条全不合格，唯一代码只有 4 个，
  全是 sin 波相位平移，帧数 7/7 写错（要 100/120 帧，一律 frames=40）。
  根因不只是「模型弱」——是**示例本身就是错的**：原 few-shot 全来自模型自己的旧产物，
  它把 frames=40 当规范抄走。污染源在教材，不在学生。
  ⇒ 冷启动的燃料不能靠模型自产（会再污染），必须是外部规范骨架。

这些骨架同时是三重身份：
  1. 做梦的 few-shot 种子（模型照它学「怎么把多约束一次满足」）
  2. task_verify 的正例基准（回归测试：规则改严了会不会误杀）
  3. 三元组银行的可信底座（本身就是合格样本）

用法:
  python tools/dream_seeds.py --list                 # 列出 15 个骨架
  python tools/dream_seeds.py --check                # 只跑静态校验（不执行，秒级）
  python tools/dream_seeds.py --run [--only 0,3]     # 真执行 + 双验签 + 入库
"""
import os, sys, json, time, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

FFMPEG = r"D:\work\_local_ai\vidcut\node_modules\ffmpeg-static\ffmpeg.exe"
SEED_DIR = os.path.join(ROOT, "data", "seeds")

# 统一头部：Agg 后端 + 小画幅（CPU 渲染要快）+ FuncAnimation + ffmpeg
HEAD = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

'''

# 每个骨架：(模板序号, 说明, 代码模板)
# 代码里的 __F__ / __A__ 用 str.replace 替换（不能用 format：代码里有大量花括号）
SEEDS = {
    0: ("余弦波（模板 skill 名沿用了旧的「三角波」，内容以任务为准）", HEAD + '''fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, __A__ * np.pi, 300)
line, = ax.plot(x, np.cos(x), lw=1.6)
ax.set_ylim(-1.4, 1.4)


def upd(i):
    line.set_ydata(np.cos(x + i / 10.0))
    ax.set_title("cos phase %d" % i)
    return line,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    1: ("正弦+余弦叠加", HEAD + '''fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, __A__ * np.pi, 300)
l1, = ax.plot(x, np.sin(x), lw=1.4, label="sin")
l2, = ax.plot(x, np.cos(x), lw=1.4, label="cos")
ax.set_ylim(-1.4, 1.4)
ax.legend(loc="upper right", fontsize=7)


def upd(i):
    l1.set_ydata(np.sin(x + i / 12.0))
    l2.set_ydata(np.cos(x + i / 12.0))
    return l1, l2


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    2: ("衰减正弦波（振幅随时间衰减）", HEAD + '''fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, __A__ * np.pi, 300)
line, = ax.plot(x, np.sin(x), lw=1.6)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    amp = np.exp(-i / 30.0)
    line.set_ydata(amp * np.sin(x - i / 10.0))
    ax.set_title("damped  amp=%.3f" % amp)
    return line,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    3: ("多曲线：正弦/方波/三角波各一条", HEAD + '''fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, __A__ * np.pi, 400)
sq = np.sign(np.sin(x))
tri = (2.0 / np.pi) * np.arcsin(np.sin(x))
l1, = ax.plot(x, np.sin(x), lw=1.3, label="sin")
l2, = ax.plot(x, sq, lw=1.0, label="square")
l3, = ax.plot(x, tri, lw=1.0, label="triangle")
ax.set_ylim(-1.5, 1.5)
ax.legend(loc="upper right", fontsize=7)


def upd(i):
    s = i / 14.0
    l1.set_ydata(np.sin(x + s))
    l2.set_ydata(np.sign(np.sin(x + s)))
    l3.set_ydata((2.0 / np.pi) * np.arcsin(np.sin(x + s)))
    return l1, l2, l3


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    4: ("李萨如曲线（x=sin(m t), y=sin(n t)）", HEAD + '''fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=60)
t = np.linspace(0, 2.0 * np.pi, 800)
m, n = 3, 4
xs = np.sin(m * t)
ys = np.sin(n * t)
line, = ax.plot(xs, ys, lw=1.2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    k = i / 30.0
    line.set_xdata(np.sin(m * t + k))
    line.set_ydata(np.sin(n * t))
    return line,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    5: ("旋转螺旋线（极坐标 r=t 转直角）", HEAD + '''fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=60)
theta = 2 * np.pi * np.linspace(0, 2, 800)
r = np.linspace(0, 1, 800)
xs = r * np.cos(theta)
ys = r * np.sin(theta)
line, = ax.plot(xs, ys, lw=1.2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    k = i / 25.0
    line.set_xdata(r * np.cos(theta + k))
    line.set_ydata(r * np.sin(theta + k))
    return line,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    6: ("3D 曲面旋转（view_init 转视角）", HEAD + '''from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(3.8, 3.4), dpi=60)
ax = fig.add_subplot(111, projection="3d")
X, Y = np.meshgrid(np.linspace(0, __A__ * np.pi, 50), np.linspace(0, __A__ * np.pi, 50))
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)


def upd(i):
    ax.view_init(elev=28, azim=i * 5)
    return ax,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    7: ("二维波动方程热力图（imshow 逐帧更新）", HEAD + '''fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=60)
N = 48
Z = np.zeros((N, N))
X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
im = ax.imshow(Z, cmap="viridis", animated=True, vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])


def upd(i):
    t = i / 12.0
    z = np.sin(np.sqrt(X ** 2 + Y ** 2) * 3 - t) * np.exp(-0.3 * np.sqrt(X ** 2 + Y ** 2))
    im.set_array(z)
    return im,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    8: ("傅里叶级数逼近方波（前 N 项逐帧增加）", HEAD + '''fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(-np.pi, np.pi, 500)
target = np.sign(np.sin(x))
line, = ax.plot(x, np.zeros_like(x), lw=1.5)
ref, = ax.plot(x, target, lw=1.0, ls="--", color="gray")
ax.set_ylim(-1.6, 1.6)
NT = __A__


def partial(n_terms):
    y = np.zeros_like(x)
    for k in range(1, 2 * n_terms, 2):
        y += np.sin(k * x) / k
    return y * 4.0 / np.pi


def upd(i):
    n = min(i + 1, NT)
    line.set_ydata(partial(n))
    ax.set_title("fourier n=%d" % n)
    return line, ref


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    9: ("粒子布朗运动散点", HEAD + '''fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=60)
rng = np.random.default_rng(7)
N = __A__ * 100
pos = rng.normal(0, 0.3, size=(N, 2))
sc = ax.scatter(pos[:, 0], pos[:, 1], s=6, alpha=0.6)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)


def upd(i):
    global pos
    pos = np.clip(pos + rng.normal(0, 0.05, size=(N, 2)), -2.4, 2.4)
    sc.set_offsets(pos)
    return sc,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    10: ("条形图竞赛（10 根柱子排名变化）", HEAD + '''fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=60)
rng = np.random.default_rng(11)
K = 10
vals = rng.random(K)
idx = np.arange(K)
bars = ax.bar(idx, vals, color="tab:blue")
ax.set_ylim(0, 1.2)


def upd(i):
    global vals
    vals = vals + rng.normal(0, 0.04, size=K)
    vals = np.clip(vals, 0.02, 1.15)
    for b, v in zip(bars, vals):
        b.set_height(v)
    ax.set_title("frame %d" % i)
    return bars


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    11: ("Julia 集缩放", HEAD + '''fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=60)
N = 90
c = complex(-0.7, 0.27)


def julia(zoom):
    xs = np.linspace(-1.6 / zoom, 1.6 / zoom, N)
    ys = np.linspace(-1.6 / zoom, 1.6 / zoom, N)
    Z = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            z = complex(xs[b], ys[a])
            k = 0
            while k < 28 and abs(z) < 2.0:
                z = z * z + c
                k += 1
            Z[a, b] = k
    return Z


img = ax.imshow(julia(1.0), cmap="hot", animated=True)
ax.set_xticks([])
ax.set_yticks([])


def upd(i):
    img.set_array(julia(1.0 + i * 0.03))
    return img,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    12: ("冒泡排序每一步柱状图", HEAD + '''fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=60)
rng = np.random.default_rng(3)
M = __A__ * 10
arr0 = rng.random(M)
states = [arr0.copy()]
cur = arr0.copy()
for a in range(M):
    for b in range(M - 1 - a):
        if cur[b] > cur[b + 1]:
            cur[b], cur[b + 1] = cur[b + 1], cur[b]
        states.append(cur.copy())
step = max(1, len(states) // max(1, __F__))
picked = states[::step][:__F__]
bars = ax.bar(np.arange(M), picked[0], color="tab:orange")
ax.set_ylim(0, 1.1)


def upd(i):
    snap = picked[min(i, len(picked) - 1)]
    for b, v in zip(bars, snap):
        b.set_height(v)
    ax.set_title("bubble step %d" % i)
    return bars


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    13: ("康威生命游戏演化", HEAD + '''fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=60)
N = 32
rng = np.random.default_rng(5)
grid = (rng.random((N, N)) < 0.3).astype(np.int8)
im = ax.imshow(grid, cmap="binary", animated=True)
ax.set_xticks([])
ax.set_yticks([])


def step(g):
    nb = sum(np.roll(np.roll(g, i, 0), j, 1)
             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    alive = (nb == 3) | ((g == 1) & (nb == 2))
    return alive.astype(np.int8)


def upd(i):
    global grid
    grid = step(grid)
    im.set_array(grid)
    ax.set_title("gen %d" % i)
    return im,


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
    14: ("小球抛体运动 + 触地反弹", HEAD + '''fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=60)
g = 9.8
x, y = 0.0, 0.0
vx, vy = 4.0, 9.0
trail_x, trail_y = [x], [y]
line, = ax.plot(trail_x, trail_y, lw=1.2)
dot, = ax.plot([x], [y], "o", ms=6, color="tab:red")
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)


def upd(i):
    global x, y, vx, vy
    dt = 0.06
    x += vx * dt
    y += vy * dt
    vy -= g * dt
    if y <= 0:
        y = 0.0
        vy = -vy * 0.8
    if x > 11.5:
        x, y, vx, vy = 0.0, 0.0, 4.0, 9.0
        trail_x.clear()
        trail_y.clear()
    trail_x.append(x)
    trail_y.append(y)
    line.set_xdata(trail_x)
    line.set_ydata(trail_y)
    dot.set_xdata([x])
    dot.set_ydata([y])
    return line, dot


ani = FuncAnimation(fig, upd, frames=__F__, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
'''),
}


def build(idx, a, f):
    """"返回 (skill, task, code, hint)"""
    import dream_autorun as da
    skill, tpl, hint = da.TEMPLATES[idx]
    task = tpl.format(A=a, F=f)
    code = SEEDS[idx][1].replace("__A__", str(a)).replace("__F__", str(f))
    return skill, task, code, hint


def default_params(idx):
    """每个骨架给一组稳定参数：帧数取模板里较大的值，保证画面看得出变化。"""
    import dream_autorun as da
    tpl = da.TEMPLATES[idx][1]
    a = 4 if "{A}" in tpl else 0
    f = 100 if "{F}" in tpl else 0
    return a, f


def cmd_list():
    for i in sorted(SEEDS):
        import dream_autorun as da
        a, f = default_params(i)
        skill, task, _, _ = build(i, a, f)
        print("[%2d] %-10s %s" % (i, skill, task))
        print("     %s" % SEEDS[i][0])


def cmd_check():
    import task_verify as tv
    import dream_autorun as da
    ok_n = 0
    for i in sorted(SEEDS):
        a, f = default_params(i)
        skill, task, code, _ = build(i, a, f)
        r = tv.verify(task, code)
        ok_n += 1 if r["pass"] else 0
        print("[%s] %2d %-10s %s" % ("PASS" if r["pass"] else "FAIL", i, skill, task[:44]))
        if not r["pass"]:
            for ch in r["checks"]:
                if not ch["pass"]:
                    print("       ! %-12s %s  (hard=%s)" % (ch["name"], ch["detail"], ch["hard"]))
    print("\n静态校验 %d/%d" % (ok_n, len(SEEDS)))
    return ok_n == len(SEEDS)


def cmd_run(only=None, keep=False):
    os.makedirs(SEED_DIR, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(FFMPEG) + os.pathsep + env.get("PATH", "")
    results = []
    for i in sorted(SEEDS):
        if only and i not in only:
            continue
        a, f = default_params(i)
        skill, task, code, hint = build(i, a, f)
        d = os.path.join(SEED_DIR, "s%02d" % i)
        os.makedirs(d, exist_ok=True)
        cp = os.path.join(d, "code.py")
        open(cp, "w", encoding="utf-8").write(code)
        print("\n=== seed %02d [%s] %s" % (i, skill, task[:48]))
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, cp], cwd=d, env=env, timeout=180,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            rc = p.returncode
            err = (p.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            rc, err = -1, "timeout 180s"
        sec = time.time() - t0
        vid = os.path.join(d, "out.mp4")
        has = os.path.exists(vid) and os.path.getsize(vid) > 1000
        print("    执行 rc=%s %.1fs 视频=%s" % (rc, sec, "%dKB" % (os.path.getsize(vid) // 1024) if has else "无"))
        if not has:
            print("    !! 失败: %s" % err.replace("\n", " ")[:200])
            results.append({"idx": i, "ok": False, "stage": "exec", "err": err})
            continue
        # 双验签
        r1 = subprocess.run([sys.executable, os.path.join(HERE, "task_verify.py"),
                             "--task", task, "--code-file", cp, "--json"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        tv_ok = r1.returncode == 0
        r2 = subprocess.run([sys.executable, os.path.join(HERE, "vid_verify.py"), vid,
                             "--min-quality", "0.75", "--json"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        vv_ok = r2.returncode == 0
        print("    task_verify=%s  vid_verify=%s" % ("PASS" if tv_ok else "FAIL", "PASS" if vv_ok else "FAIL"))
        results.append({"idx": i, "ok": tv_ok and vv_ok, "stage": "verify",
                        "tv": tv_ok, "vv": vv_ok, "dir": d, "task": task, "skill": skill})
        if tv_ok and vv_ok:
            # 入库（走唯一写入路径，同时过 task_verify 闸门）
            inc = subprocess.run([sys.executable, os.path.join(HERE, "triple_bank.py"), "add",
                                  "--task", task, "--code", cp, "--video", vid,
                                  "--ffmpeg", FFMPEG, "--skill", skill],
                                 cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            out = (inc.stdout or inc.stderr or "").strip().replace("\n", " | ")
            print("    入库: %s" % out[-160:])
    ok = sum(1 for r in results if r["ok"])
    print("\n=== 骨架执行 %d/%d 通过双验签 ===" % (ok, len(results)))
    json.dump(results, open(os.path.join(SEED_DIR, "last_run.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return ok == len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    if a.list:
        cmd_list(); return
    if a.check:
        sys.exit(0 if cmd_check() else 1)
    if a.run:
        only = [int(x) for x in a.only.split(",") if x.strip().isdigit()] if a.only else None
        sys.exit(0 if cmd_run(only) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
