#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_loop.py v2 — 睡眠·做梦自进化编排器（可跑态）

闭环：
  记忆检索(few-shot 种子示例) → (0.3B 底座 + 编程 LoRA) 生成代码
       → 沙箱执行(subprocess+硬超时) → 视频验收(vid_verify 闸门)
       → 入库(精英档案 + 三元组银行) + 失败料回灌 rejects

相对 v1 的关键修正（v1 生成出"代码块"死字，根因是 prompt 与训练协议不同构）：
  - 用代码模块专用 ChatML 协议(SYSTEM='你是北极星，主人的私有编程助手…')，
    与训练数据逐字同构（对齐 code_eval.py 的 build_prompt）。
  - 加载 0.3B 底座 + 编程 LoRA(PyFT 挂载，target=q/k/v/o，守保函数+E-7)。
  - 生成用 model.generate + eos=<|im_end|>，再按围栏块取码（同 code_eval）。
  - DREAM_BANK 全部是「贴近种子 sin 波示例」的 matplotlib 变体 —— 弱模型靠
    few-shot 检索照着改 sin→cos 等，才大概率产出可执行视频，真过 vid_verify。
  - latent 适配(T2-Z 测试时隐变量)作为可选嵌入残差 hook，默认关，避免干扰代码正确性。

宪法红线(保函数 + E-7)：主干冻绝，只动 LoRA(已训好) / 测试时 latent / 可删记忆层。
代码执行一律 subprocess + 硬超时，绝不拖垮主进程。

依赖：venv_ttt(torch 2.7.1+cpu / transformers 4.57.3 / peft 0.21.2) + ffmpeg
用法：
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/dream_loop.py --smoke
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/dream_loop.py \
      --base D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018 \
      --lora D:/work/_local_ai/train_15b/out_lora_code_03b/checkpoint-150 \
      --n 4 --ttt-off
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
def _find_ffmpeg():
    """跨平台定位 ffmpeg：先认 Windows 静态包，再回落系统 PATH（魔搭 Linux 走这条）。"""
    cand = "D:/work/_local_ai/vidcut/node_modules/ffmpeg-static/ffmpeg.exe"
    if os.path.exists(cand):
        return cand
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = _find_ffmpeg()
DREAM_DIR = os.path.join(ROOT, "data", "dreams")
IM_END = "<|im_end|>"

def inject_imports(code):
    """执行前补缺失 import（模型常漏 FuncAnimation/numpy/pyplot/matplotlib Agg）。
    治本容错：模型不可靠，不靠提示保证 import 完整；注入后的完整代码也作为入库样本。"""
    head = []
    if "matplotlib.use" not in code:
        head.append('import matplotlib; matplotlib.use("Agg")')
    # 无条件预置 plt import：Qwen 若已自带 import 也不冲突（重复 import 无害）。
    # 关键：fig,ax 行在 head 末尾、位于 Qwen 原码之前，必须保证 plt 已先 import，否则 NameError。
    if ("plt." in code or "pyplot" in code):
        head.append("import matplotlib.pyplot as plt")
    if ("np." in code or "numpy" in code) and "import numpy" not in code:
        head.append("import numpy as np")
    if "FuncAnimation" in code and "from matplotlib.animation" not in code and "import FuncAnimation" not in code:
        head.append("from matplotlib.animation import FuncAnimation")
    # 强制标准 fig 结构：只要 Qwen 没显式写 `fig, ax = plt.subplots()`，就在 head 末尾注入一行。
    # 🔴 2026-10-18 修复要点：
    #  1) fig,ax 行必须进 head、且 head 排在 import 之后（已保证 plt 先 import），不再 prepend 到代码最顶。
    #  2) 不做字符串替换（旧版 `plt.figure()`→`fig` 会把 `fig = plt.figure()` 改成 `fig = fig` 致 NameError）。
    #     若 Qwen 自己另外定义 fig/ax，注入的这行会被覆盖，无害；若 Qwen 用 plt.gcf() 取当前 fig，取到的就是注入的 fig。
    if "fig, ax" not in code:
        head.append('fig, ax = plt.subplots()')
    if head:
        code = "\n".join(head) + "\n" + code
    # 🔴 2026-10-18 容错：强制 blit=False。blit=True 要求 update 精确 return 可变 artist 元组，
    # 弱模型极易写错导致 FuncAnimation 崩或产出静止 mp4；blit=False 永远安全（仅略慢），契合「保产出」优先。
    import re
    code = re.sub(r'blit\s*=\s*True', 'blit=False', code)
    # 弱模型常写成 `plt.savefig('out.mp4')` —— savefig 不支持 mp4 会崩。若已建好 FuncAnimation(ani)，
    # 直接把 savefig 改为 ani.save 并补 writer=fmpeg，保住动画产出。
    if 'FuncAnimation' in code and 'plt.savefig' in code:
        code = re.sub(r'plt\.savefig\s*\(', 'ani.save(', code)
        # savefig 常带 format='mp4'，但 ani.save 指定 writer 后不能再给 format（冲突会崩），一律剥掉。
        code = re.sub(r",?\s*format\s*=\s*['\"]mp4['\"]", '', code)
        if 'writer=' not in code:
            code = re.sub(r"ani\.save\(([^)]*)\)", r"ani.save(\1, writer='ffmpeg', fps=20)", code)
    return code

SYSTEM = ("你是北极星，主人的私有编程助手。只输出一个 python 代码块，配一两句说明。\n"
          "硬性要求：\n"
          "1. 开头固定: import matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation; import numpy as np\n"
          "2. 动画必须用下面【正确结构】，update 必须定义且 return 一个元组，FuncAnimation 必须用 blit=False：\n"
          "   fig, ax = plt.subplots()\n"
          "   x = np.linspace(0, 4*np.pi, 200)\n"
          "   line, = ax.plot(x, np.sin(x))\n"
          "   def update(frame):\n"
          "       line.set_ydata(np.sin(x + frame / 10.0))\n"
          "       return line,\n"
          "   ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)\n"
          "   ani.save(\"out.mp4\", writer=\"ffmpeg\", fps=20)\n"
          "3. 结尾必须 ani.save(\"out.mp4\", writer=\"ffmpeg\")，禁止 plt.show()。\n"
          "4. 必须严格用「fig, ax = plt.subplots()」拿到 fig，禁止用 plt.gcf() 替代 fig；可改 update 内部与初始曲线，但 FuncAnimation 调用方式与 update 签名不要变。\n"
          "5. 任务点名的波形必须真的画出来：要求方波/三角波/余弦/衰减/多曲线，就严格实现对应公式(如 np.cos / np.sign(np.sin(x)) / np.exp(-k*x) / 多条 plot)，不要偷懒只画一条 sin。要求几条曲线就画几条、各用不同颜色。代码可直接 python 运行。")

# 做梦任务库：全部是「贴近种子 sin 波示例」的 matplotlib 变体。
# 统一要求生成 out.mp4 —— 使 vid_verify 成为天然闸门；few-shot 检索把种子
# 示例喂进去，弱模型照着改，才大概率过闸（这正是 RSI 自举：从已知好例出发造变体）。
DREAM_BANK = [
    # —— 无硬要素：纯 sin / 双 sin 变体（Qwen 几乎必过，快速积累正样本）——
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个正弦波动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "标准结构：fig,ax=plt.subplots(); x=np.linspace(0,4*np.pi,200); line,=ax.plot(x,np.sin(x)); update 里 line.set_ydata(np.sin(x+frame/10.0)); FuncAnimation(fig,update,frames=100,blit=False); ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个正弦波动画(频率=2, 0~4π, 100 帧)，保存为 out.mp4",
     "hint": "y=np.sin(2*x)；其余同标准正弦动画结构；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画两条不同颜色的正弦曲线(相位差 π/2, 0~4π, 100 帧)，保存为 out.mp4",
     "hint": "l1,=ax.plot(x,np.sin(x),'b'); l2,=ax.plot(x,np.cos(x),'r'); update 同时 set_ydata 两条；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画双正弦叠加拍频动画 y=sin(1.1x)+sin(0.9x) (0~20π, 200 帧)，保存为 out.mp4",
     "hint": "y=np.sin(1.1*x)+np.sin(0.9*x)；标准动画结构；frames=200；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个正弦波动画并填充曲线下面积(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "ax.fill_between(x, 0, np.sin(x)) 在 update 里重画；标准动画结构；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 单 HARD 要素：明确给公式，降低要素缺失 ——
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个余弦波动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "把上面 sin 换成 np.cos 即可；FuncAnimation + ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画正弦+余弦叠加的动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "必须画两条：l1=sin(x), l2=np.cos(x)，各不同颜色；update 里两条都 set_ydata；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个衰减正弦波动画(0~6π, 120 帧，振幅随时间衰减)，保存为 out.mp4",
     "hint": "y = np.sin(x) * np.exp(-0.05*x)；衰减必须真实出现在公式里；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个余弦波动画(带标题'余弦波'和网格, 0~2π, 80 帧)，保存为 out.mp4",
     "hint": "np.cos(x)；ax.set_title('余弦波'); ax.grid(True)；frames=80；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个高频余弦波动画(频率=5, 0~4π, 120 帧)，保存为 out.mp4",
     "hint": "y=np.cos(5*x)；标准动画结构；frames=120；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画正弦+余弦叠加动画并加图例(图例标注 y1/y2, 0~4π, 100 帧)，保存为 out.mp4",
     "hint": "两条曲线各 ax.plot(..., label='y1'/'y2')；ax.legend()；必须含 np.cos；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个余弦相位移动动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "y=np.cos(x - frame/10.0)；标准动画结构；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个方波动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "用 np.sign(np.sin(x)) 造方波(无需额外库)；必须真实出现 sign/where；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个三角波动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "用 2*np.arcsin(np.sin(x)) 造三角波(无需额外库)；必须真实出现 arcsin；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个阻尼振荡动画 y=sin(x)*exp(-0.1x) (0~8π, 150 帧)，保存为 out.mp4",
     "hint": "y=np.sin(x)*np.exp(-0.1*x)；衰减 exp 必须真实出现；frames=150；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 双 HARD / SOFT：保留多样性，权重最低 ——
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个多曲线对比动画(正弦/方波/三角波各一条, 0~4π, 100 帧)，保存为 out.mp4",
     "hint": "三条曲线都要：sin(x) + np.sign(np.sin(x))(方波) + 2*np.arcsin(np.sin(x))(三角波)，各不同颜色；ax.plot 三条；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个李萨如(Lissajous)曲线动画, 100 帧，保存为 out.mp4",
     "hint": "x=np.sin(3*t), y=np.sin(2*t)；标准动画结构(画参数曲线)；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "游戏模拟", "verify": "video",
     "task": "用 matplotlib 画一个旋转螺旋线动画(100 帧)，保存为 out.mp4",
     "hint": "极坐标 r=t, theta=t；x=r*np.cos(theta), y=r*np.sin(theta) 转直角坐标 plot；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 物理世界函数（2026-10-18 加入：让小模型内化现实规律，动画=可验证的世界模型）——
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画牛顿冷却定律动画：一杯热水温度按 T(t)=T_env+(T0-T_env)*exp(-k*t) 指数衰减(100 帧)，保存为 out.mp4",
     "hint": "T0=90, T_env=25, k=0.03；横轴时间 0~100，曲线逐帧延伸到 t=frame；同时用颜色从红渐变到蓝表示降温；公式 exp 必须真实出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画理想气体微观模型：N 个粒子在方盒内匀速运动并弹性碰撞壁，右侧柱状图实时显示粒子数分布(120 帧)，保存为 out.mp4",
     "hint": "np.random 生成 60 个粒子的位置/速度，每帧位置+=速度*dt，越界则速度反向；散点 ax.scatter 画粒子；右侧子图 ax.bar 画瞬时统计；粒子+柱状都要出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画一维热传导动画：金属杆初始中间热两端冷，温度分布随热方程扩散趋于均匀(150 帧)，保存为 out.mp4",
     "hint": "离散热方程 T_new[i]=T[i]+alpha*dt/dx^2*(T[i+1]-2*T[i]+T[i-1])；初始高斯温度峰；用 imshow 热力图或按位置 plot 曲线逐帧更新，颜色随时间变化；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画卡诺循环动画：p-V 图上依次经历等温膨胀、绝热膨胀、等温压缩、绝热压缩四过程闭合循环(100 帧)，保存为 out.mp4",
     "hint": "两段等温 p=nRT/V、两段绝热 p*V^gamma=常数；曲线逐段画出形成闭合圈，工作点标记同步移动；多曲线+闭合路径必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画光的折射与反射动画：光线以可调入射角从空气射入水中，画出反射光线和折射光线并标注角度(80 帧)，保存为 out.mp4",
     "hint": "斯涅尔定律 n1*sin(theta1)=n2*sin(theta2)，空气 n=1.0、水 n=1.33；界面画水平线，法线画虚线；入射角逐帧从 10° 增到 70°，反射角=入射角、折射角按公式算；ax.legend 图例标注三条光线；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画杨氏双缝干涉动画：两列相干余弦波叠加，屏幕上形成明暗相间干涉条纹且随波长变化(120 帧)，保存为 out.mp4",
     "hint": "两列波 y1=np.cos(k*x1), y2=np.cos(k*x2)，x1/x2 为到两缝距离；屏上强度 I=|y1+y2|^2；画波面曲线+屏上条纹，波长逐帧变化；余弦+多曲线必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画多普勒效应动画：波源向右移动，前方波纹压缩、后方拉伸(100 帧)，保存为 out.mp4",
     "hint": "以固定时间间隔从波源当前位置发出同心圆环波纹；波源每帧 x+=v*dt，v<波速；用 ax.scatter 或 ax.plot 画圆弧，前面圆环间距变小后面变大；余弦波纹+散点；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画弹簧振子简谐运动动画：物块在弹簧上振动，右侧同步画位移-时间余弦曲线(100 帧)，保存为 out.mp4",
     "hint": "位移 x=A*np.cos(omega*t)；左侧画弹簧折线+物块方块随位移移动，右侧画已走过的位移曲线逐帧延伸；A=1.0, omega=2.0；余弦必须真实出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画抛体运动对比动画：不同发射角度的多条抛物线轨迹同时展开并落点标记(100 帧)，保存为 out.mp4",
     "hint": "v0=20, g=9.8；对 30°/45°/60° 分别算 x=v0*cos(a)*t, y=v0*sin(a)*t-0.5*g*t^2；三条曲线不同颜色逐帧画出，落点画散点标记；多曲线+散点；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画绳上驻波动画：行波与反射波叠加形成驻波，节点与波腹位置随波长变化(120 帧)，保存为 out.mp4",
     "hint": "y1=np.sin(k*x-omega*t), y2=np.sin(k*x+omega*t) 反射波；y=y1+y2=2*np.sin(k*x)*np.cos(omega*t)；节点画虚线标注；波长逐帧变；余弦/正弦+多曲线；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画熵增加动画：N 个粒子初始聚集在盒子左半，屏障消失后自发扩散布满全盒，实时显示左右粒子数差衰减到零(150 帧)，保存为 out.mp4",
     "hint": "np.random 生成 50 个粒子初始 x 全在左半[0,0.5]，随机速度；每帧位置+=速度*dt，全区间 [0,1] 弹性反射；左侧子图画粒子散点，右侧画左右计数差随帧数衰减曲线；粒子+衰减必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 通用函数库·数学函数域（2026-10-18 扩充：一次加够，集群直接起跑不愁粮）——
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画指数函数增长动画 y=exp(0.15x) (0~40, 100 帧，曲线逐帧向右延伸)，保存为 out.mp4",
     "hint": "x=np.linspace(0,40,200)；y=np.exp(0.15*x)；update 里 set_xdata(x[:frame*2]) 逐步延伸；指数增长必须真实出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画对数函数增长动画 y=ln(x) (0.1~50, 100 帧，与 y=x 对比)，保存为 out.mp4",
     "hint": "x=np.linspace(0.1,50,200)；画两条：l1=np.log(x), l2=x；对数增速放缓与直线对比，多曲线+图例；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画幂函数族对比动画 y=x^2, y=x^3, y=sqrt(x) (0~5, 100 帧)，保存为 out.mp4",
     "hint": "三条不同颜色：x**2, x**3, np.sqrt(x)；多曲线同时逐帧延伸；ax.legend 标注；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画有理函数动画 y=1/x 展示双曲线与渐近线 (0.1~10, 100 帧)，保存为 out.mp4",
     "hint": "x 正区间 y=1/x；画竖直渐近线 ax.axvline(0,ls='--') 和水平渐近线 ax.axhline(0,ls='--')；曲线逐帧延伸；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画 sigmoid 函数动画 y=1/(1+exp(-x)) 并同步画其导数钟形曲线 (100 帧)，保存为 out.mp4",
     "hint": "y=1/(1+np.exp(-x))；导数 y*(1-y) 也是高斯形；两条曲线不同颜色同步逐帧延伸；指数 exp 必须真实出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画 tanh 与 sigmoid 对比动画 (-6~6, 100 帧)，保存为 out.mp4",
     "hint": "l1=np.tanh(x), l2=1/(1+np.exp(-x))；两条曲线各颜色；多曲线+图例；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画高斯钟形曲线动画 y=exp(-x^2/(2*sigma^2)) 且 sigma 逐帧变化 (-5~5, 100 帧)，保存为 out.mp4",
     "hint": "sigma 从 0.5 渐变到 3；y=np.exp(-x**2/(2*sigma**2))；衰减曲线逐帧重画；指数 exp 必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画 sinc 函数振荡衰减动画 y=sin(x)/x (-20π~20π, 100 帧)，保存为 out.mp4",
     "hint": "x=np.linspace(-20*np.pi,20*np.pi,800) 去掉 0 点；y=np.sin(x)/x；振荡+衰减；曲线逐帧延伸；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画阶跃函数逼近动画：sigmoid 温度参数逐帧降低逼近方波 (100 帧)，保存为 out.mp4",
     "hint": "y=1/(1+np.exp(-T*x))，T 从 1 增到 100，曲线越来越陡最终像方波；方波要素由极限形态呈现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画多项式函数动画 y=x^3-3x+1 展示三个零点与切线滚动 (-3~3, 100 帧)，保存为 out.mp4",
     "hint": "y=x**3-3*x+1；导数 3*x**2-3 作切线斜率；零点位置画竖虚线标注；曲线+切线；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画分段函数动画：x<0 时 y=x^2，x>=0 时 y=sqrt(x)，接点处连续 (100 帧)，保存为 out.mp4",
     "hint": "np.where(x<0, x**2, np.sqrt(np.abs(x)))；分段用两种颜色画；接点 x=0 画标记点；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "数学函数", "verify": "video",
     "task": "用 matplotlib 画调幅 AM 信号动画 y=(1+0.5*sin(2π*0.5t))*sin(2π*5t) (200 帧)，保存为 out.mp4",
     "hint": "载波 sin(2π*5t) 高频，包络 1+0.5*sin(2π*0.5t) 低频；画包络虚线+信号实线；余弦/正弦+多曲线；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 通用函数库·几何曲线域 ——
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画摆线动画：半径 r 的圆沿直线滚动，圆上一点画出摆线 x=r(t-sin t), y=r(1-cos t) (150 帧)，保存为 out.mp4",
     "hint": "t 从 0 增到 6π；同时画滚动圆(ax.plot 圆参数方程)和已生成的摆线轨迹；圆滚动+轨迹逐帧延伸；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画心形线极坐标动画 r=1-cos(theta) (0~2π 逐帧绘制)，保存为 out.mp4",
     "hint": "theta 数组逐帧增长；x=r*np.cos(theta), y=r*np.sin(theta)；心形逐段画出；极坐标转直角必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画玫瑰线动画 r=cos(3*theta) 三叶玫瑰逐帧旋转绽放 (150 帧)，保存为 out.mp4",
     "hint": "theta 0~2π；x=r*cos(theta), y=r*sin(theta)；每帧整体旋转一个小角度；花瓣逐帧显现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画椭圆动画：参数方程 x=a*cos t, y=b*sin t，两个焦点标记且到焦点距离之和恒定 (100 帧)，保存为 out.mp4",
     "hint": "a=3,b=2；焦点 (±c,0), c=sqrt(a²-b²)；画动点到两焦点的连线证明距离和=2a；余弦+文本标注；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画星形线动画 x=cos³t, y=sin³t 与内切圆滚动 (150 帧)，保存为 out.mp4",
     "hint": "x=np.cos(t)**3, y=np.sin(t)**3；四尖星逐段画出；余弦/正弦立方必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画正多边形逼近圆动画：边数从 3 逐帧增到 60 (120 帧)，保存为 out.mp4",
     "hint": "n 从 3 到 60；顶点 angle=np.linspace(0,2π,n+1)；x=np.cos(angle),y=np.sin(angle)；多边形逐帧变圆；网格背景；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画毕达哥拉斯树分形生长动画(迭代 8 层, 150 帧逐层展开)，保存为 out.mp4",
     "hint": "递归/迭代画正方形+上方直角三角形；每帧多画一层；自相似结构；文本标注层数；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "几何曲线", "verify": "video",
     "task": "用 matplotlib 画李萨如曲线族动画：不同频率比 (a:b) 逐帧切换 (3:2→5:4→2:3, 150 帧)，保存为 out.mp4",
     "hint": "x=np.sin(a*t), y=np.sin(b*t+phase)；频率比逐帧变；李萨如逐段重画；图例标注当前 a:b；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 通用函数库·信号处理域 ——
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画方波傅里叶级数合成动画：谐波项数 N 从 1 增到 20，合成波越来越像方波 (150 帧)，保存为 out.mp4",
     "hint": "y=Σ(4/π)·sin((2k-1)x)/(2k-1)；每帧 N+1；方波形态由级数极限逼近；正弦级数必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画锯齿波合成动画：谐波项数从 1 增到 15 (120 帧)，保存为 out.mp4",
     "hint": "锯齿波 y=Σ(2(-1)^(k+1)/k)·sin(kx)；逐帧加项；曲线逐步变陡；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画信号频谱动画：三个正弦分量叠加，右侧柱状图实时显示各分量幅度 (120 帧)，保存为 out.mp4",
     "hint": "y=sin(2π*1x)+0.5*sin(2π*3x)+0.25*sin(2π*5x)；左图波形右图 ax.bar 频谱；柱状图必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画采样混叠动画：高频正弦用低频率采样，逐帧降低采样率展示混叠假象 (120 帧)，保存为 out.mp4",
     "hint": "原信号 sin(2π*10t)；采样点每帧减少；散点标记采样点+原曲线；散点+正弦；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画调频 FM 信号动画：调制幅度逐帧增大，瞬时频率变化可见 (150 帧)，保存为 out.mp4",
     "hint": "y=sin(2π*5*t + m*sin(2π*0.5*t))；m 从 0 增到 10；相位调制使波形疏密变化；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "信号处理", "verify": "video",
     "task": "用 matplotlib 画滤波动画：含噪信号逐帧被移动平均平滑，噪声方差衰减曲线同步显示 (150 帧)，保存为 out.mp4",
     "hint": "signal=sin(2π*x)+np.random.normal(0,0.3)；卷积/滑动平均窗口逐帧加宽；衰减曲线显示残余误差；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 通用函数库·算法可视化域 ——
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画梯度下降动画：在碗形函数 f(x,y)=x²+y² 上从高处逐帧下滑到谷底 (120 帧)，保存为 out.mp4",
     "hint": "x,y 初始 (4,4)；每帧 x-=lr*2x, y-=lr*2y；画等高线背景+下降轨迹红线+当前点标记；轨迹逐帧延伸；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画蒙特卡洛估 π 动画：随机散点落入单位正方形，内切圆计数逐帧更新，π 估计值收敛到 3.14 (200 帧)，保存为 out.mp4",
     "hint": "np.random.uniform(-1,1,(n,2))；判定 x²+y²<=1；散点分两色+右侧 π 估计曲线；散点+柱状/曲线；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画一维随机游走动画：粒子逐帧左右随机移动，位移方差随时间扩散 (200 帧)，保存为 out.mp4",
     "hint": "每帧位置 += choice([-1,1])；画轨迹折线+当前点；右侧画位移绝对值包络 ±sqrt(t) 理论曲线；粒子+网格；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画冒泡排序动画：随机柱高逐帧比较交换变成有序，右侧显示比较次数 (150 帧)，保存为 out.mp4",
     "hint": "arr=np.random.permutation(20)；每帧做一步比较交换；ax.bar 柱状图颜色随比较高亮；柱状图必须出现；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画 BFS 洪水填充动画：从网格中心出发逐帧向四邻扩散，访问次序着色 (150 帧)，保存为 out.mp4",
     "hint": "30×30 网格；队列扩散；已访问格子 imshow 上色；网格+散点；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画康威生命游戏动画：随机初始网格按规则逐帧演化 (150 帧)，保存为 out.mp4",
     "hint": "40×40 随机 0/1；每帧统计八邻域计数应用规则；imshow 二值图；网格演化；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画 K-means 聚类动画：散点数据逐帧重新分配簇并更新中心 (150 帧)，保存为 out.mp4",
     "hint": "np.random 生成 3 簇高斯点；每帧算到 3 中心距离重分配+更新中心；散点按簇着色+中心大标记；散点+图例；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "算法可视化", "verify": "video",
     "task": "用 matplotlib 画 DLA 分形聚集生长动画：粒子随机游走粘附到簇上形成分形 (200 帧)，保存为 out.mp4",
     "hint": "中心种子点；粒子从外围随机释放游走，邻接簇即粘附；散点画簇；粒子+网格；ani.save('out.mp4',writer='ffmpeg')"},
    # —— 通用函数库·物理补充域 ——
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画单摆动画：摆锤摆动同时右侧画相图(角度-角速度)轨迹 (150 帧)，保存为 out.mp4",
     "hint": "θ''=-(g/L)sinθ 欧拉积分；左图摆线+摆锤，右图 ax.plot 相图逐帧延伸；相图是闭合椭圆(小角近似简谐)；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画双耦合振子动画：两个振子通过弹簧连接，能量此消彼长形成拍频 (200 帧)，保存为 out.mp4",
     "hint": "两自由度耦合方程欧拉积分；画两振子位移曲线+中间弹簧；能量交换出现拍频包络；余弦+多曲线；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画二维波动方程动画：薄膜中心受击后波纹扩散衰减 (150 帧)，保存为 out.mp4",
     "hint": "40×40 网格离散热方程 u_new=2u-u_old+c²∇²u；imshow 热力图逐帧更新；波纹扩散+衰减；网格；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画洛伦兹吸引子动画：三维混沌轨迹逐帧绘制 (dx,dy,dz 方程积分, 300 帧)，保存为 out.mp4",
     "hint": "σ=10,ρ=28,β=8/3；dt=0.005 欧拉积分；ax.plot 三维轨迹线逐帧延伸；蝴蝶翅膀双叶结构；轨迹+文本标注参数；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画放射性衰变动画：剩余原子核数指数衰减，半衰期位置逐条标注 (150 帧)，保存为 out.mp4",
     "hint": "N=N0*exp(-λt)；画衰减曲线+半衰期竖虚线(T½=ln2/λ)+逐帧减少的粒子散点；衰减+散点；ani.save('out.mp4',writer='ffmpeg')"},
    {"domain": "物理仿真", "verify": "video",
     "task": "用 matplotlib 画三棱镜色散动画：白光入射棱镜分解成七色光谱，各色折射角不同 (120 帧)，保存为 out.mp4",
     "hint": "棱镜画三角形；七种颜色光线入射角相同但折射角按波长略不同(红小紫大)；光线用不同颜色 ax.plot；文本标注红/橙/黄/绿/蓝/靛/紫；ani.save('out.mp4',writer='ffmpeg')"},
]


def ensure_dirs():
    os.makedirs(DREAM_DIR, exist_ok=True)


# ── 记忆检索(喂 few-shot) ───────────────────────────────────────────────
def retrieve_examples(task, k=2, elite_k=1):
    """从三元组银行抽 top-k 相似历史成功代码，作为做梦的 few-shot 上下文。

    🔴 只认库里的 code_text 快照，不按 code_path 读磁盘：
       triple_bank 的设计原则是「存快照不存路径」（文件会被改/被清），
       旧实现 `open(code=路径)` 在 data/dreams 被清理后会让 few-shot 静默失效，
       且可能读到训练后被改过的版本 —— RSI 的燃料链会断在这里。

    🔴 elite_k：额外从 MAP-Elites 精英档案抽 1 条「不同技能」的示例。
       银行 query 是按任务相似度取 top-k，天然偏向同一类任务 → 越喂越窄。
       精英档案的职责正好相反：保多样。让这两路各出一条，才是真闭环
       （此前精英档案只写不读，等于白建）。
    """
    ex = []
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "triple_bank.py"), "query",
             "--task", task, "-k", str(k), "--show-code"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=ROOT)
        out = p.stdout or ""
        blocks, cur = [], None
        for line in out.splitlines():
            if line.startswith("["):          # 新条目头 → 上一块结束
                cur = None
                continue
            if line.strip() == "---":         # 代码块开始
                cur = []
                blocks.append(cur)
                continue
            if cur is not None:
                cur.append(line[8:] if line.startswith(" " * 8) else line)
        ex = [("\n".join(b)).strip() for b in blocks]
        ex = [t[:1800] for t in ex if t][:k]
    except Exception as e:
        print("[retrieve] 银行检索跳过(err=%s)" % e)

    if elite_k > 0:
        try:
            sys.path.insert(0, HERE)
            import dream_elite as de
            # 跨示例去重阈值必须比「同桶判重」宽：同桶 0.82 是为了防"同一解重画一遍"，
            # 但不同技能的 matplotlib 动画骨架本来就高度相似（import/FuncAnimation/save
            # 三段样板就超 0.82），用 0.82 会把所有候选误杀 → 实测一条都注不进去。
            # 0.95 只拦逐字节级重复（实测 #6/#8 J=1.000，sin/cos J=0.802 放过）。
            EX_DUP = 0.95
            skill = de.infer_skill(task)
            cand = de.EliteArchive().diverse(k=elite_k + 2, avoid_skills={skill})
            for it in cand:
                code = (it.get("code") or "").strip()
                if not code:
                    continue
                # 与银行已取的示例近重复就不重复喂（省 token，也免得强化同一解法）
                g = de.ngrams(code)
                if any(de.jaccard(g, de.ngrams(e)) >= EX_DUP for e in ex):
                    continue
                ex.append(code[:1600])
                print("[retrieve] 精英注入 %s (q=%.2f)" % (it["skill"], it["quality"]))
                break
        except Exception as e:
            print("[retrieve] 精英档案跳过(err=%s)" % e)

    return ex


# ── 代码生成：代码模块专用 ChatML 协议（与训练数据逐字同构） ──────────────
def build_prompt(task, hint, examples):
    few = ""
    if examples:
        few = "\n# 参考过往成功写法(仅供参考风格，不要照抄):\n" + "\n\n".join(
            "# --- example %d ---\n%s" % (i + 1, e) for i, e in enumerate(examples))
    usr = "%s\n提示: %s%s" % (task, hint, few)
    # 与 code_eval.build_prompt 完全一致的手工 ChatML
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (SYSTEM, usr))


def extract_code(text):
    """取第一个带语言标注的围栏块；无围栏时若全文像 python 代码则整段取。

    🔴 2026-10-18 修复：Qwen 在 "只输出一个 python 代码块" 指令下会吐裸代码
    （无 ```python 围栏）。旧正则只认围栏→代码长度=0→整批 REJECT。加 fallback
    让裸代码也能抽出，交给 run_code 实际验证，不冤杀有效草稿。
    """
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", text, re.S)
    if blocks:
        for lang, body in blocks:
            if lang.strip():
                return lang.strip().lower(), body
        return "", blocks[0][1]
    # 无围栏 fallback（模型直接吐裸代码）：全文含 import/def/print 即视为 python 代码
    s = (text or "").strip()
    if s and ("import " in s or "def " in s or "print(" in s):
        return "", s
    return None, None


# few-shot 复读的指纹：命中即判污染，不许入库。
POLLUTION_MARKS = (
    "参考过往成功写法",
    "# --- example",
    "example %d ---",          # 占位，check_pollution 里单独处理
    "<|im_start|>",
    "<|im_end|>",
)


def check_pollution(code):
    """检测模型把 prompt 内容复读进了代码。返回命中的指纹串，干净则 None。

    🔴 真实事故（2026-10-18 排查）：精英档案 #6 的 code_text 开头就是
       "# 参考过往成功写法(仅供参考风格，不要照抄):\\n# --- example 1 ---\\nimport matplotlib..."
       —— 模型把 few-shot 标题和分隔符一起复读了出来，vid_verify 只看
       「有没有产出视频」→ 放行入库 → 下一轮又当 few-shot 喂回去。
       不拦的话污染自我放大：复读 → 入库 → 更强复读，RSI 直接退化成复读机。
    这条闸门的成本是 0（纯字符串匹配），收益是守住 RSI 的燃料纯度。
    """
    if not code:
        return None
    for m in POLLUTION_MARKS:
        if "%d" in m:
            continue
        if m in code:
            return m
    if re.search(r"example\s*\d+\s*---", code):
        return "example N ---"
    return None


# ── 模型：0.3B 底座 + 编程 LoRA（PEFT, q/k/v/o） ──────────────────────
def build_model(base, lora=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32,
                                                trust_remote_code=True)
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora, is_trainable=False)
        print("[model] 已挂编程 LoRA: %s" % lora)
    model.eval()
    return model, tok


def gen_code(model, tok, prompt, max_new=512, temp=0.0, top_p=0.92):
    """生成并截断到 <|im_end|>，返回模型续写文本。"""
    eos = tok.convert_tokens_to_ids(IM_END)
    ids = tok(prompt, return_tensors="pt").input_ids
    import torch
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=(temp if temp > 0 else 1.0),
            top_p=(top_p if temp > 0 else 1.0),
            eos_token_id=eos, pad_token_id=(tok.pad_token_id or eos))
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    if IM_END in gen:
        gen = gen.split(IM_END)[0]
    return gen


# ── 可选：测试时 latent 残差 hook（T2-Z 思路，默认关） ─────────────────
class LatentHook:
    def __init__(self, latent):
        self.latent = latent.detach()
        self.h = None

    def attach(self, model):
        emb = model.get_input_embeddings()
        lat = self.latent

        def fn(module, inp, out):
            # out: [B, T, H]
            return out + lat.view(1, 1, -1).to(out.dtype).to(out.device)

        self.h = emb.register_forward_hook(fn)

    def detach(self):
        if self.h:
            self.h.remove()
            self.h = None


def adapt_latent(model, tok, support, steps=8, lr=0.05):
    """训练一个零初始化 latent 向量（next-token CE），只优化 latent。"""
    import torch
    H = model.config.hidden_size
    latent = torch.zeros(H, requires_grad=True)
    opt = torch.optim.SGD([latent], lr=lr)
    ids = tok(support, return_tensors="pt").input_ids
    emb = model.get_input_embeddings()
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        e = emb(ids) + latent.view(1, 1, -1)
        out = model(inputs_embeds=e)
        loss = torch.nn.functional.cross_entropy(out.logits[0, :-1], ids[0, 1:])
        loss.backward()
        opt.step()
        last = loss.item()
    return latent, last


# ── 沙箱执行 ───────────────────────────────────────────────────────────
def run_code(code, run_id, timeout=90):
    d = os.path.join(DREAM_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    fpath = os.path.join(d, "code.py")
    open(fpath, "w", encoding="utf-8").write(code)
    env = os.environ.copy()
    _ffdir = os.path.dirname(FFMPEG)
    if _ffdir:  # 系统 PATH 里已有 ffmpeg 时不存在目录概念，别注入空项
        env["PATH"] = _ffdir + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, fpath],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=d, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "rc": None, "out": "", "err": "timeout",
                "video": None, "sec": time.time() - t0, "dir": d}
    sec = time.time() - t0
    out, err, rc = p.stdout, p.stderr, p.returncode
    vids = [os.path.join(d, fn) for fn in os.listdir(d)
            if fn.lower().endswith((".mp4", ".avi", ".mov", ".webm"))]
    return {"ok": rc == 0, "reason": "exec", "rc": rc, "out": out, "err": err,
            "video": (vids[0] if vids else None), "sec": sec, "dir": d}


# ── 验收 + 入库 ────────────────────────────────────────────────────────
def accept(task, code, video, quality, run_id, skill=None):
    """入库 = 三元组银行 + 精英档案，两者都在 triple_bank add 里一次完成（单一写入路径）。

    skill 透传给银行 → 银行再传给精英档案分桶：生成端知道在练什么能力，
    就别让精英档案的正则去猜（正则只作外部任务的兜底）。
    """
    hit = check_pollution(code)
    if hit:
        print("   [accept] 拒收：代码含 few-shot 复读指纹(%s)" % hit)
        return False
    codefile = os.path.join(DREAM_DIR, run_id, "code.py")
    cmd = [sys.executable, os.path.join(HERE, "triple_bank.py"), "add",
           "--task", task, "--code", codefile, "--video", video or "__none__",
           "--ffmpeg", FFMPEG]
    if skill:
        cmd += ["--skill", skill]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=ROOT)
        for line in (p.stdout or p.stderr).strip().splitlines():
            print("   [accept]", line)
        return p.returncode == 0
    except Exception as e:
        print("   [accept] 失败 err=%s" % e)
        return False


def record_trial(task, used_memory, passed):
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "triple_bank.py"), "trial",
             "--task", task, "--memory", "on" if used_memory else "off",
             "--passed", "yes" if passed else "no", "--note", "dream_loop"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    except Exception:
        pass


# ── 主循环 ─────────────────────────────────────────────────────────────
class Dreamer:
    def __init__(self, base, lora, use_latent, steps, lr, ffmpeg=None):
        global FFMPEG
        if ffmpeg:
            FFMPEG = ffmpeg
        self.model, self.tok = build_model(base, lora)
        self.use_latent = use_latent
        self.steps = steps
        self.lr = lr
        ensure_dirs()

    def one_dream(self, spec, use_memory=True):
        task = spec["task"]
        hint = spec.get("hint", "")
        run_id = "run_%s_%d" % (time.strftime("%Y%m%d_%H%M%S"),
                                int(time.time() * 1000) % 1000)
        print("\n=== 做梦 %s | 域=%s | 记忆=%s ===" % (run_id, spec["domain"], "on" if use_memory else "off"))
        print("   任务: %s" % task)

        examples = retrieve_examples(task, k=2) if use_memory else []
        if examples:
            print("   [memory] 检索到 %d 个 few-shot 示例" % len(examples))

        hook = None
        if self.use_latent:
            support = (task + " " + hint + " " + " ".join(examples[:1])) if examples else (task + " " + hint)
            latent, lloss = adapt_latent(self.model, self.tok, support, self.steps, self.lr)
            print("   [latent] 末步 loss=%.4f" % lloss)
            hook = LatentHook(latent)
            hook.attach(self.model)

        try:
            prompt = build_prompt(task, hint, examples)
            gen = gen_code(self.model, self.tok, prompt, max_new=512, temp=0.0)
            lang, code = extract_code(gen)
            print("   [gen] lang=%s 代码长度=%d" % (lang or "-", len(code or "")))
            if not code:
                print("   [gen] 未解析出代码 → REJECT")
                record_trial(task, use_memory, False)
                return False

            # few-shot 复读闸门：模型把 prompt 抄进代码时，跑出来的视频可能"正常"，
            # 但这份代码是污染的燃料，入银行会自我放大（见 check_pollution 注释）。
            hit = check_pollution(code)
            if hit:
                print("   [gate] 检测到 few-shot 复读(%s) → REJECT" % hit)
                record_trial(task, use_memory, False)
                return False

            code = inject_imports(code)   # 执行前补缺失 import（治本容错，注入后代码作为入库样本）
            res = run_code(code, run_id, timeout=90)
            if not res["ok"]:
                print("   [exec] 失败 rc=%s err=%s" % (res["rc"], (res["err"] or "")[:200]))
                record_trial(task, use_memory, False)
                return False
            print("   [exec] 成功 用时%.1fs 产出视频=%s" % (res["sec"], bool(res["video"])))

            if res["video"]:
                ok = accept(task, code, res["video"], 1.0, run_id, skill=spec.get("skill"))
                record_trial(task, use_memory, ok)
                return ok
            else:
                print("   [verify] 无视频产出 → REJECT(三元组银行只收过 vid_verify 的视频)")
                record_trial(task, use_memory, False)
                return False
        finally:
            if hook:
                hook.detach()

    def run(self, n, use_memory=True, offset=0):
        specs = [DREAM_BANK[(i + offset) % len(DREAM_BANK)] for i in range(n)]
        stats = {"pass": 0, "fail": 0}
        for spec in specs:
            try:
                ok = self.one_dream(spec, use_memory=use_memory)
                stats["pass" if ok else "fail"] += 1
            except Exception as e:
                print("   [dream] 异常=%s" % e)
                stats["fail"] += 1
        print("\n=== 做梦轮次结束 ===")
        print("   通过(入库) %d / 失败(落 rejects) %d" % (stats["pass"], stats["fail"]))
        return stats


def smoke():
    """不加载模型的管线冒烟：验证 exec → 视频验收通路。

    🔴 不落库：smoke 产物是「能跑」的假教材（print('DONE') + testsrc 测试图），
    一旦入库就会像历史上那条「冒烟测试-视频闸门」一样污染精英档案。
    """
    ensure_dirs()
    rid = "smoke_%d" % int(time.time())
    code = "print('DONE')\n"
    res = run_code(code, rid, timeout=30)
    print("[smoke] exec ok=%s video=%s" % (res["ok"], res["video"]))
    vid_path = os.path.join(DREAM_DIR, "smoke_test.mp4")
    fcmd = [FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
            "-pix_fmt", "yuv420p", vid_path]
    pr = subprocess.run(fcmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if pr.returncode != 0:
        print("[smoke] ffmpeg 产片失败: %s" % (pr.stderr or "")[-300:])
    else:
        print("[smoke] ffmpeg 产片 OK")
        vp = subprocess.run(
            [sys.executable, os.path.join(HERE, "vid_verify.py"), vid_path, "--ffmpeg", FFMPEG],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("[smoke] 视频验收 rc=%d（只验签，不落库）" % vp.returncode)
        for line in (vp.stdout or "").strip().splitlines()[-4:]:
            print("   [smoke]", line)
    print("[smoke] 完成（exec→验收 通路 OK；smoke 不入库）")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018")
    ap.add_argument("--lora", default=None, help="编程模块 LoRA 目录(含 adapter_config.json)")
    ap.add_argument("--n", type=int, default=4, help="做梦轮数")
    ap.add_argument("--offset", type=int, default=0,
                    help="任务起始偏移(轮转评测集用，A/B 两臂须同值才公平)")
    ap.add_argument("--steps", type=int, default=8, help="latent 适配步数(仅 --use-latent)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--use-latent", action="store_true", help="开启测试时 latent 适配(默认关)")
    ap.add_argument("--no-memory", action="store_true", help="不做记忆检索(基线对照)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--smoke", action="store_true", help="不加载模型，只测管线")
    a = ap.parse_args()

    if a.smoke:
        smoke()
        return
    d = Dreamer(a.base, a.lora, use_latent=a.use_latent, steps=a.steps,
                lr=a.lr, ffmpeg=a.ffmpeg)
    d.run(a.n, use_memory=not a.no_memory, offset=a.offset)


if __name__ == "__main__":
    main()
