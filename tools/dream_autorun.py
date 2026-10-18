#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_autorun.py — A 路做梦 RSI 周期自转调度器（由 schtasks /ru SYSTEM 调用）

与 dream_loop.py 解耦的职责：
  - 原子单实例锁：上轮没跑完就跳过本周期，绝不叠加吃 CPU（与 CodeLoraP2 训练错峰）。
  - 自管 utf-8 日志到 logs/dream_autorun_YYYY-MM-DD.log（SYSTEM 账户无控制台，只读日志）。
  - 定向出题（原为纯随机）：make_spec() 三路混合 —— 填 MAP-Elites 空格子 / 攻
    rejects 短板 / 保留随机探索。实测空洞技能命中率 64.3%→87.7%，已占桶的 5 类
    从 35.7% 压到 12.3%。持续丰富三元组银行 → few-shot 检索持续增强（RSI 的燃料）。
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


def _elite_skills():
    """精英档案里已经占了桶的技能集合（= MAP-Elites 网格已填的格子）。

    读不到就返回空集：出题绝不能因为档案坏了而停摆。
    """
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import dream_elite as de
        _, _, rows = de.EliteArchive().stats()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


def _reject_skills():
    """rejects 里失败过的技能名 —— 失败样本是判「能力边界」的唯一信源。

    实测 2026-10-18：rejects 仅 1 条，所以这一路现在基本等同于随机；
    等跑够 50+ 轮、reason 分布散开，它才真正接管「攻短板」的职责。
    """
    try:
        import sqlite3
        p = os.path.join(ROOT, "data", "triple_bank.db")
        if not os.path.exists(p):
            return set()
        c = sqlite3.connect(p)
        tasks = [t or "" for (t,) in c.execute("SELECT task FROM rejects").fetchall()]
        c.close()
        if not tasks:
            return set()
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import dream_elite as de
        return {de.infer_skill(t) for t in tasks}
    except Exception:
        return set()


def _reject_focus():
    """从 rejects.detail 解析「上次具体栽在哪个维度」→ {维度: 次数}。

    🔴 为什么「攻短板」不能只做到技能粒度（2026-10-18 实测）：
      装上 task_verify 后回扫发现，7 条入库样本里 7 条都栽在同一个坑 ——
      任务要 100/120 帧，代码一律写 frames=40。技能维度上这是「三角波/衰减振荡/
      参数曲线」等多种技能，看上去失败得很分散；但真实病因只有一个：
      模型把 few-shot 示例里的 frames=40 当成规范抄走（示例全是 40）。
      ⇒ 只按技能重抽同一个题，模型还是会写 40。必须在出题时把这个坑点亮。

    detail 来自 task_verify 的 checks（name/pass），兼容旧 vid_verify blob（解析失败即跳过）。
    """
    out = {}
    try:
        import sqlite3
        p = os.path.join(ROOT, "data", "triple_bank.db")
        if not os.path.exists(p):
            return out
        c = sqlite3.connect(p)
        rows = c.execute("SELECT reason, detail FROM rejects").fetchall()
        c.close()
    except Exception:
        return out
    for reason, detail in rows:
        if not detail:
            continue
        try:
            j = json.loads(detail)
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        # 格式一：task_verify 原始输出（checks 数组）
        for ch in (j.get("checks") or []):
            if isinstance(ch, dict) and not ch.get("pass") and ch.get("name"):
                out[ch["name"]] = out.get(ch["name"], 0) + 1
        # 格式二：bank_purge 迁移时写的摘要（fails = ["维度: 说明", ...]）
        for s in (j.get("fails") or []):
            if isinstance(s, str) and ":" in s:
                name = s.split(":", 1)[0].strip()
                if name:
                    out[name] = out.get(name, 0) + 1
        if reason:
            out["_reason:" + str(reason)] = out.get("_reason:" + str(reason), 0) + 1
    return out


# 失败维度 → prompt 里要点亮的硬约束（把上一次的具体错因写进 hint）
FOCUS_TIPS = {
    "帧数符合": "帧数必须与任务完全一致：FuncAnimation(fig, upd, frames=N) 的 N 就写任务给的那个数字，"
               "不许用 40 或任何默认值",
    "定义域符合": "linspace 的上界必须等于任务给的 Kπ，例如 0~6π 就写 np.linspace(0, 6*np.pi, 200)",
    "任务要素齐备": "任务里点名的每一条曲线/每一种波形都必须真的出现在代码里：要余弦就写 np.cos，"
                 "要方波就写 np.sign(np.sin(x))，要三角波就写 2*np.pi*np.arcsin(np.sin(x))，"
                 "要衰减就乘 np.exp(-k*x)",
    "输出文件名符合": "anim.save 的文件名必须与任务给定的完全一致",
    "产出动画": "必须用 FuncAnimation 生成动画并 save 成 mp4，不许只画静态图",
}


def _focus_hint(focus, task=""):
    """把失败维度翻译成 prompt 里的一段硬提示；没有可点亮的项就返回空串。

    按任务相关性过滤：定义域这条只在任务真的给了 π 定义域时才注入，
    否则给「小球抛体」这种题加「linspace 上界要等于 Kπ」只会误导。
    """
    tips = []
    task = task or ""
    for name, desc in FOCUS_TIPS.items():
        if not focus.get(name):
            continue
        if name == "定义域符合" and ("π" not in task and "pi" not in task.lower()):
            continue
        tips.append("- %s" % desc)
    if not tips:
        return ""
    return ("\n【上次同类任务的实测错误，这次必须避免】\n" + "\n".join(tips))


def make_spec():
    """出题三路混合：把「纯随机」换成「有方向的进化压力」。

    病灶（2026-10-18 实测）：原实现从 15 个模板等概率抽，结果精英档案 5 个桶
    全挤在曲线族（三角波/衰减振荡/参数曲线/极坐标/多序列），而曲面三维、场热力图、
    级数逼近、粒子统计、排名条形图、分形、排序搜索、生命游戏、刚体动画
    **9 类技能一个桶都没占** —— 已经会的反复做，不会的永远抽不到。
    MAP-Elites 的全部价值就在「填空位」，不做这步就只是个好看的排行榜。

    三路：
      1. 填格子：优先抽精英档案里还没占桶的技能 → 逼多样性
      2. 攻短板：优先抽 rejects 里失败过的技能 → 攻能力边界
      3. 纯随机：保留探索，防局部最优

    「填格子」的权重自适应（这是关键，固定 40% 太保守）：
      w_hole = 0.4 + 0.4 * (空洞数 / 技能总数)
      空洞 9/14 时 w_hole≈0.66；填满后自动回落到 0.4，各技能趋于均等 —— 自平衡。
    实测（400 次抽样）：固定 40% 时空洞命中 79.5%，自适应 0.66 时约 88%。

    降级规则：某一源缺数据就把权重让给随机路，绝不因缺数据不出题。
    """
    by_skill = {}
    for rec in TEMPLATES:
        by_skill.setdefault(rec[0], []).append(rec)

    have = _elite_skills()
    holes = [s for s in by_skill if s not in have]
    bad = _reject_skills() & set(by_skill)

    w_hole = 0.4 + 0.4 * (len(holes) / max(1, len(by_skill))) if holes else 0.0
    w_bad = 0.30

    r = random.random()
    pick = None
    if r < w_hole:
        pick = random.choice(by_skill[random.choice(holes)])
    elif r < w_hole + w_bad and bad:
        pick = random.choice(by_skill[random.choice(sorted(bad))])
    if pick is None:
        pick = random.choice(TEMPLATES)

    skill, tpl, hint = pick
    a = random.choice([2, 3, 4, 5, 6]) if "{A}" in tpl else 0
    f = random.choice([80, 100, 120, 150])
    task = tpl.format(A=a, F=f)
    # 失败回灌：把 rejects 里的具体错因点亮进 hint（不只是按技能重抽同一个题）
    hint = hint + _focus_hint(_reject_focus(), task)
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
