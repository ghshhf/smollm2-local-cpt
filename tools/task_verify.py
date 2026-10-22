#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task_verify.py — 任务约束符合度校验（做梦闭环的第 7 道闸门）

🔴 为什么必须有这一层（2026-10-18 实测发现）:
    triple_bank 里 7 条「验签通过」的三元组，只有 4 个唯一代码，两两 3-gram
    相似度 0.84~1.00，而且没有一条实现自己任务的要求：
      #3 要 100 帧，代码 frames=40
      #4 要 0~6π / 120 帧 / 振幅衰减，代码 4π / frames=40 / 振幅恒定
      #6 要「正弦+余弦叠加」，代码只有 np.sin(x)
      #9 要李萨如曲线、#10 要螺旋线，代码还是 sin 波相位平移
    根因：vid_verify 的 6 项全是「视频能不能播 / 画面动不动」，属于**产物层**验收；
    它永远不知道代码做的是不是题目要求的事。质检缺失 ⇒ 做梦产出的「精英」是错误教材。

设计原则（刻意克制）:
  - 纯静态，不执行任何生成代码（执行级校验属 code_verify 的另一层）
  - 只判**能高置信机验**的约束；抽不到或判不了的项标 skipped，不计入分母，
    避免用脆规则误杀（"精 = 先读懂数据再定规则"）
  - 与 vid_verify 同构：checks 列表 + score + hard + quality + pass，
    便于 dream_loop 直接串在 vid_verify 之后

可判约束（从任务文本抽取）:
  1. frames   「N 帧」                 vs 代码 frames=N
  2. domain   「0~Kπ」                vs 代码 linspace 上界 == K*pi
  3. outfile  「保存为 X.mp4」         vs 代码 save('X')
  4. symbols  关键词→必需符号(余弦/方波/三角波/衰减/散点/柱状/饼/热力/向量场...)
  5. anim     任务含「动画」           vs 代码有 FuncAnimation/ArtistAnimation 且 save

用法:
  python tools/task_verify.py --task "..." --code-file path.py [--json]
  python tools/task_verify.py --bank [--db data/triple_bank.db] [--json]
退出码: 0=PASS  1=FAIL
"""
import os, sys, re, json, argparse, sqlite3, ast

# 关键词 → 代码里必须出现的 token（任一命中即算）
SYMBOL_RULES = [
    ("余弦",      ["np.cos", "cos(", "numpy.cos"]),
    ("正切",      ["np.tan", "tan("]),
    ("方波",      ["square(", "sign(", "sgn(", "heaviside", "piecewise"]),
    ("三角波",    ["sawtooth(", "triangle", "arcsin", "abs(", "tri("]),
    ("衰减",      ["exp(", "np.exp", "** -", "**(-", "(1/2)", "0.5**", "(0.5)",
                  "np.power", "halflife", "half_life", "half-life"]),
    ("散点",      ["scatter("]),
    # 柱状(竖 bar) / 条形(横 barh) 静态 token 分不开（中文任务池常混用，模型统一用 bar()），
    # 二者共用 bar(/barh( 都接受，区分竖/横交给识图线（semantic_match）
    ("柱状",      ["bar(", "barh("]),
    ("条形",      ["bar(", "barh("]),
    ("饼图",      ["pie("]),
    ("热力图",    ["imshow(", "heatmap", "pcolormesh"]),
    ("等高线",    ["contour(", "contourf("]),
    ("向量场",    ["quiver("]),
    ("直方图",    ["hist("]),
    ("螺旋",      ["cos(2", "cos(3", "2 * np.pi * t", "2*np.pi*t", "helix", "spiral"]),
    ("李萨如",    ["lissajous", "sin(3", "sin(2 * t)", "sin(5"]),
    ("三维",      ["Axes3D", "projection='3d'", 'projection="3d"', "mplot3d"]),
    ("柱体",      ["bar3d("]),
    ("曲面",      ["plot_surface("]),
    ("粒子",      ["scatter(", "Line2D", "particle", "particles"]),
    ("文本",      ["text(", "annotate("]),
    ("网格",      ["grid("]),
    ("图例",      ["legend("]),
    # --- 元素级（2026-10-19 补）：唯一标准 API，缺=真没做，跨任务通用 ---
    ("填充",      ["fill_between(", "fill("]),
    ("标题",      ["set_title(", "suptitle("]),
    ("频谱",      ["fft(", "rfft(", "fftfreq", "np.fft"]),
    ("极坐标",    ["polar"]),
    ("排序",      ["sort(", "sorted(", "swap(", "bubble"]),
    ("布朗运动",  ["cumsum(", "brownian"]),
    # --- 物理/算法（2026-10-19 补）：写法不唯一 → SOFT 软告警，不硬杀 ---
    ("梯度下降",  ["gradient", "descent", "learning_rate", "derivative"]),
    ("碰撞",      ["collision", "bounce", "velocity", "momentum"]),
    ("生命游戏",  ["conway", "game_of_life", "neighbor", "np.roll"]),
    ("扩散",      ["diffusion", "heat", "laplacian", "np.roll"]),
    ("单摆",      ["pendulum", "theta"]),
    ("自由落体",  ["gravity", "freefall", "9.8"]),
    ("椭圆",      ["Ellipse(", "ellipse"]),
    ("心形",      ["cardioid", "heart", "polar"]),
    ("混叠",      ["alias", "downsample"]),
    ("调频",      ["instantaneous", "frequency_modulation"]),
    ("卡诺",      ["carnot", "isothermal", "adiabatic"]),
    ("毕达哥拉斯树", ["pythagoras", "recursive", "tree"]),
    ("摆线",      ["cycloid"]),
    ("星形线",    ["astroid", "hypocycloid"]),
    ("抛物线",    ["parabola", "projectile"]),
    ("分形",      ["fractal", "julia", "mandelbrot"]),
    # --- 函数族（2026-10-19 二轮补）：唯一标准 API，缺=真没做 → HARD ---
    ("对数",      ["np.log", "log(", "ln(", "np.log10", "math.log", "logspace"]),
    ("指数",      ["exp(", "np.exp", "2**", "e**", "np.power", "math.exp",
                  ")**", "(1/2)", "0.5**"]),
    ("sigmoid",   ["sigmoid", "expit", "1/(1+exp", "1 / (1 + exp", "1/(1+exp"]),
    ("sinc",      ["sinc", "sin(x)/x", "sin(x) / x", "np.sinc"]),
    ("幂函数",    ["sqrt(", "x**2", "x**3", "x**0.5", "np.power", "cbrt"]),
    ("分段函数",  ["piecewise", "np.where", "x<0", "x>=0", "np.select"]),
    ("阶跃",      ["heaviside", "step(", "np.sign", "sign("]),
    ("渐近线",    ["asymptote", "1/x", "np.inf", "axvline", "axhline"]),
    ("谐波合成",  ["harmonic", "fourier", "series", "n=1", "harmonics"]),
    ("拍频",      ["beat", "拍频", "sin(1.1", "sin(0.9", "1.1x", "0.9x"]),
    ("驻波",      ["standing", "antinode", "node"]),
    # --- 物理仿真（2026-10-19 二轮补）---
    ("干涉",      ["interference", "fringe", "slit", "double_slit", "明暗"]),
    ("多普勒",    ["doppler", "mach", "compressed", "波源"]),
    ("折射",      ["refraction", "snell", "arcsin", "折射角", "入射角"]),
    ("色散",      ["dispersion", "prism", "spectrum", "rainbow", "七色"]),
    ("折射角",    ["arcsin", "snell", "refraction"]),
    ("弹跳",      ["bounce", "rebound", "elasticity", "触地"]),
    ("涟漪",      ["ripple", "expanding", "同心圆", "涟漪"]),
    ("台球",      ["billiard", "pool", "friction", "滑行"]),
    ("冷却",      ["cooling", "newton", "T_env", "t_env", "热水"]),
    ("理想气体",  ["gas", "molecule", "elastic", "方盒"]),
    ("波动方程",  ["wave_equation", "laplacian", "c*dt", "c**2", "薄膜"]),
    ("调幅",      ["am_", "amplitude", "(1+0.5", "modulation"]),
    ("阻尼",      ["damping", "damped", "阻尼"]),
    ("行进波",    ["traveling", "phase", "k*x", "omega*t", "向右平移"]),
    ("旋转",      ["rotate", "rotation", "transform", "绕中心", "affine"]),
    ("轨迹",      ["trail", "轨迹", "留下"]),
    # --- 算法/随机（2026-10-19 二轮补）---
    ("聚类",      ["kmeans", "cluster", "centroid", "k_means", "簇"]),
    ("滤波",      ["filter", "convolve", "moving_average", "移动平均", "smooth"]),
    ("蒙特卡洛",  ["monte", "montecarlo", "mc_", "随机散点", "pi_est", "estimate"]),
    ("随机游走",  ["random_walk", "randn", "randint", "walk", "左右随机"]),
    ("DLA聚集",   ["dla", "cluster", "aggregate", "粘附", "游走粘附"]),
    ("熵增",      ["entropy", "自发", "屏障", "左右粒子"]),
    ("查找",      ["search", "指针", "命中", "linear_search", "sequential", "binary_search"]),
    ("零点",      ["zeros", "factor", "切线", "fzero", "newton", "bisect"]),
    ("导数",      ["derivative", "gradient", "tangent", "diff(", "切线"]),
    ("逼近",      ["polygon", "apothem", "逼近", "n_sides", "inscribed"]),
    ("进度条",    ["progress", "percentage", "barh(", "0%", "100%"]),
    ("扩张",      ["expanding", "半径", "radius", "r +=", "从 0"]),
    ("闪烁",      ["twinkle", "flicker", "alpha", "明暗", "star"]),
    ("圆周运动",  ["omega", "角速度", "圆轨道", "angular_velocity"]),
    # --- 三轮补（2026-10-19）：sin 套壳高频 target ---
    # 远程银行实测：这些要素 task 说得明明白白，模型只会吐 np.sin(x) 相位平移。
    # 特征给全，交给 verify() 里的「语义套壳拦截」做三重连坐判定（不在这里硬判）。
    # 螺线只用 theta/θ/spiral/helix 判——「r=」过宽（会命中小写 writer= 里的 r=）
    ("螺线",      ["spiral", "helix", "theta", "θ"]),
    ("随机信号",  ["np.random", "random.", "randn(", "rand(", ".normal("]),
    ("热传导",    ["diffusion", "heat", "laplacian", "np.roll", "conduction"]),
    ("弹簧振子",  ["spring", "hooke", "k*x", "omega", "oscillat"]),
]

# ============================================================
# H5 / HTML5 小游戏规则（2026-10-19 补）
# 产物是单个 .html（纯本地、双击即玩），不是 Python/matplotlib。
# ⚠️ 这些规则挂在 verify() 的 HTML 分支上，不能走 ast（HTML 会 parse 失败）。
# 分层口径同 Python 侧：HARD=唯一标准 API（缺=真没做）；SOFT=写法不唯一。
# ============================================================
H5_RULES = [
    # --- HARD：画布与游戏循环，没这两样连"游戏"都不是 ---
    ("画布",    ["<canvas", "getcontext('2d')", 'getcontext("2d")', "getcontext(`2d`)"]),
    ("帧循环",  ["requestanimationframe", "setinterval"]),
    # --- HARD：输入方式（"点击就能玩"的最低门槛）---
    ("键盘控制", ["keydown", "keyup", "keycode", "e.key", "arrowup", "arrowdown",
                "arrowleft", "arrowright", "'w'", '"w"', "'a'", '"a"',
                "'s'", '"s"', "'d'", '"d"']),
    ("鼠标点击", ["addeventlistener('click", 'addeventlistener("click',
                "onclick", "mousedown", "mouseup", "pointerdown"]),
    ("触屏支持", ["touchstart", "touchend", "touchmove", "ontouchstart"]),
    # --- HARD：游戏性骨架 ---
    ("分数",    ["score"]),
    ("碰撞检测", ["collision", "collide", "intersect", "hitbox", "aabb",
                "checkcollision", "碰撞"]),
    ("游戏结束", ["gameover", "game_over", "game over", "isgameover"]),
    ("重新开始", ["restart", "reset(", "重新开始", "重开"]),
    ("倒计时",  ["countdown", "timer", "timeleft", "settimeout"]),
    # --- SOFT：常见玩法要素，写法不唯一，软告警不硬杀 ---
    ("重力",    ["gravity", "jump", "jump()", "falling"]),
    ("音效",    ["audio(", "newaudio", "oscillator", "playsound", "beep"]),
    ("玩家角色", ["player", "sprite", "hero", "self.bird", "bird"]),
    ("障碍物",  ["obstacle", "pipe", "barrier", "enemy"]),
    ("子弹射击", ["bullet", "shoot", "fire()"]),
    ("食物",    ["food", "apple", "pellet", "eat"]),
    ("移动",    ["velocity", "vx", "vy", "dx", "dy", "speed"]),
    ("边界反弹", ["bounce", "wall", "boundary", "bounds", "outofbounds"]),
    ("暂停",    ["pause", "paused", "ispaused"]),
    ("最高分",  ["best", "highscore", "localstorage", "local-storage"]),
    ("鼠标跟随", ["mousemove", "mousex", "mousey", "followmouse"]),
    ("自动对手", ["auto", "ai", "computer", "bot"]),
    ("难度递增", ["level", "difficulty", "hard", "speedup", "加速"]),
    ("生命值",  ["hp", "lives", "health", "命"]),
    ("网格格子", ["grid", "tile", "cell", "cellsize", "cols", "rows"]),
    ("随机生成", ["random", "math.random"]),
    ("颜色填充", ["fillstyle", "strokestyle", "color"]),
    ("文字显示", ["filltext", "text", "font"]),
    ("绘制图元", ["fillrect", "drawimage", "beginpath", "arc(", "moveto", "lineto"]),
    # --- 二轮补（2026-10-19）：具体游戏类型的核心机制 ---
    ("俄罗斯方块", ["tetris", "rotatetile", "clearrow", "line_clear", "spawnpiece"]),
    ("扫雷",      ["mine", "flag", "reveal", "leftclick", "rightclick", "数字提示"]),
    ("记忆翻牌",  ["memory", "flip", "card", "pair", "配对"]),
    ("打地鼠",    ["whack", "mole", "hole", "随机出现"]),
    ("弹球台",    ["pinball", "flipper", "paddle", "弹球"]),
    ("塔防",      ["tower", "defend", "path", "waypoint", "spawn"]),
    ("赛车",      ["race", "track", "lane", "lap", "赛道"]),
    ("平台跳跃",  ["platform", "ground", "double_jump", "coyote"]),
    ("迷宫",      ["maze", "wall", "path", "exit", "迷宫"]),
    ("双人对战",  ["p1", "p2", "two_player", "local", "双人"]),
    ("计时挑战",  ["time_attack", "timeattack", "限时", "倒计时"]),
    ("连击",      ["combo", "连击", "streak"]),
    ("护盾无敌",  ["shield", "invincible", "invul", "无敌", "star"]),
    ("道具拾取",  ["powerup", "power_up", "pickup", "item", "道具"]),
    ("波次",      ["wave", "spawn", "round", "波次", "enemywave"]),
    ("Boss",      ["boss", "首领"]),
    ("掉落奖励",  ["drop", "loot", "掉落", "reward"]),
    ("升级强化",  ["upgrade", "evolve", "升级", "强化", "levelup"]),
    ("弹药装填",  ["ammo", "reload", "弹药", "装弹"]),
    ("瞄准射击",  ["aim", "crosshair", "瞄准", "准心"]),
    ("冲刺",      ["dash", "sprint", "冲刺", "shift"]),
    ("粒子特效",  ["particle", "effect", "trail", "特效", "拖尾"]),
    ("屏幕震动",  ["shake", "screenshake", "震动"]),
    ("慢动作",    ["slowmo", "slow_mo", "timescale", "慢动作"]),
    ("摄像机跟随", ["camera", "viewport", "follow", "摄像机"]),
    ("开始界面",  ["menu", "startscreen", "start", "开始", "title"]),
    ("暂停界面",  ["paused", "pause", "resume", "暂停"]),
    ("排行榜",    ["leaderboard", "rank", "highscore", "排行"]),
    ("虚拟按键",  ["joystick", "dpad", "button", "虚拟按键", "方向键"]),
    ("滑动手势",  ["swipe", "gesture", "滑动", "dragstart"]),
    ("拖拽",      ["drag", "draggable", "拖拽", "mousedown"]),
    ("全屏",      ["fullscreen", "requestfullscreen", "全屏"]),
    ("分辨率自适应", ["resize", "devicepixelratio", "innerwidth", "自适应"]),
    ("背景音乐",  ["bgm", "music", "背景音乐", "loop", "音频"]),
    ("截图分享",  ["todataurl", "screenshot", "截图", "download"]),
]

# H5 侧分层：HARD 缺了就不是这个游戏；SOFT 缺了只提示
H5_HARD = {"画布", "帧循环", "键盘控制", "鼠标点击", "触屏支持",
           "分数", "碰撞检测", "游戏结束", "重新开始", "倒计时",
           # 二轮补：核心玩法机制，缺了就不是这个游戏
           "俄罗斯方块", "扫雷", "记忆翻牌", "打地鼠", "弹球台", "塔防",
           "赛车", "平台跳跃", "迷宫", "双人对战", "计时挑战", "波次",
           "弹药装填", "瞄准射击", "道具拾取", "升级强化"}
H5_SOFT = {"重力", "音效", "玩家角色", "障碍物", "子弹射击", "食物",
           "移动", "边界反弹", "暂停", "最高分", "鼠标跟随", "自动对手",
           "难度递增", "生命值", "网格格子", "随机生成", "颜色填充",
           "文字显示", "绘制图元",
           # 二轮补
           "连击", "护盾无敌", "Boss", "掉落奖励", "冲刺", "粒子特效",
           "屏幕震动", "慢动作", "摄像机跟随", "开始界面", "暂停界面",
           "排行榜", "虚拟按键", "滑动手势", "拖拽", "全屏",
           "分辨率自适应", "背景音乐", "截图分享"}


# 「要素缺失」的软硬分层（2026-10-18 自检发现：一律软会让「缺余弦」蒙混过关）
#   HARD: 有唯一标准写法的函数，缺了就是真没做（np.cos / np.sign(sin) / arcsin / exp）
#   SOFT: 写法很不唯一（螺旋可以极坐标也可以参数方程），硬判会误杀
HARD_SYMBOLS = {"余弦", "正切", "方波", "三角波", "衰减", "散点", "柱状", "条形", "饼图",
                "热力图", "等高线", "向量场", "直方图", "三维", "柱体", "曲面",
                "填充", "标题", "频谱", "极坐标", "排序", "布朗运动",
                # 二轮补：唯一标准 API，缺=真没做
                "对数", "指数", "sigmoid", "sinc", "幂函数", "分段函数", "阶跃",
                "渐近线", "谐波合成", "拍频", "干涉", "多普勒", "折射", "色散",
                "折射角", "弹跳", "涟漪", "台球", "冷却", "理想气体", "波动方程",
                "聚类", "滤波", "蒙特卡洛", "随机游走", "DLA聚集", "熵增",
                "查找", "逼近", "进度条", "圆周运动"}
SOFT_SYMBOLS = {"螺旋", "李萨如", "粒子", "文本", "网格", "图例",
                "梯度下降", "碰撞", "生命游戏", "扩散", "单摆", "自由落体",
                "椭圆", "心形", "驻波", "混叠", "调频", "卡诺", "毕达哥拉斯树",
                "摆线", "星形线", "抛物线", "分形",
                # 二轮补：写法不唯一，软告警不硬杀
                "调幅", "阻尼", "行进波", "旋转", "轨迹", "零点", "导数",
                "扩张", "闪烁"}


def extract_constraints(task):
    """从任务文本抽约束。返回 dict；抽不到的键不存在。"""
    c = {}
    t = task or ""

    m = re.search(r"(\d+)\s*帧", t)
    if m:
        c["frames"] = int(m.group(1))

    m = re.search(r"0\s*[~～至到\-]\s*(\d+(?:\.\d+)?)\s*[πp]i?", t)
    if m:
        c["domain_pi"] = float(m.group(1))

    m = re.search(r"保存为\s*([\w./\\-]+\.(?:mp4|gif|avi|webm|html))", t)
    if m:
        c["outfile"] = m.group(1)

    # H5 / HTML5 小游戏：产物是 .html，规则挂 H5_RULES（不走 Python 的 SYMBOL_RULES）
    if re.search(r"html|HTML|H5|h5|小游戏|网页游戏|canvas|浏览器|Canvas", t):
        c["h5"] = True
        h5syms = [kw for kw, _ in H5_RULES if kw in t]
        if h5syms:
            c["h5_symbols"] = h5syms
    # 明确要「小游戏」的信号（用于拦 Python 冒充 HTML），比 h5 更严：
    # 排除「保存为 out.html」这类纯 matplotlib 导出场景
    if re.search(r"小游戏|网页游戏|H5|h5|canvas\s*游戏|html\s*游戏", t):
        c["h5_game"] = True

    m = re.search(r"(\d+(?:\.\d+)?)\s*秒", t)
    if m:
        c["seconds"] = float(m.group(1))

    m = re.search(r"(\d+)\s*fps", t, re.I)
    if m:
        c["fps"] = int(m.group(1))

    if "动画" in t or "动图" in t:
        c["animated"] = True

    if not c.get("h5"):
        syms = [kw for kw, _ in SYMBOL_RULES if kw in t]
        if syms:
            c["symbols"] = syms
    return c


def _num(node):
    """从 AST 节点静态求值成 float（失败返回 None）。支持常数/加减乘除/np.pi。"""
    try:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in ("pi", "PI"):
            return None  # 裸 pi 无法确定数值，交给上层判
        if isinstance(node, ast.Attribute) and node.attr in ("pi", "PI"):
            return 3.141592653589793
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = _num(node.operand)
            return -v if v is not None else None
        if isinstance(node, ast.BinOp):
            a, b = _num(node.left), _num(node.right)
            if a is None or b is None:
                return None
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Div) and b != 0:
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
    except Exception:
        return None
    return None


def _is_pi_expr(node):
    """判断表达式是否含 π（用于 linspace 上界识别）。"""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in ("pi", "PI"):
            return True
        if isinstance(n, ast.Name) and n.id in ("pi", "PI"):
            return True
    return False


def analyze_code(code):
    """静态抽取代码事实。不执行。"""
    facts = {"frames": None, "linspace_uppers": [], "linspace_has_pi": False,
             "save_name": None, "has_anim": False, "fps": None,
             "tokens": set(), "ok_parse": False, "error": None}
    try:
        tree = ast.parse(code or "")
        facts["ok_parse"] = True
    except SyntaxError as e:
        facts["error"] = "SyntaxError: %s" % e
        tree = None

    # token 集合（文本级，供 symbol 规则用）
    low = (code or "").lower().replace(" ", "")
    facts["tokens"] = low

    if tree is None:
        return facts

    def call_name(node):
        f = node.func
        if isinstance(f, ast.Attribute):
            return (getattr(f.value, "id", "") + "." + f.attr).lower(), f.attr.lower()
        if isinstance(f, ast.Name):
            return f.id.lower(), f.id.lower()
        return "", ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted, base = call_name(node)

        # frames / fps：关键字参数或位置参数
        for kw in node.keywords:
            if kw.arg == "frames":
                v = _num(kw.value)
                if v is not None:
                    facts["frames"] = int(v)
                else:
                    # 可能是 len(x) / 变量 -> 试图找赋值
                    facts["frames_expr"] = ast.unparse(kw.value) if hasattr(ast, "unparse") else "?"
            if kw.arg == "fps":
                v = _num(kw.value)
                if v is not None:
                    facts["fps"] = int(v)

        if base in ("funcanimation", "artistanimation"):
            facts["has_anim"] = True
            # FuncAnimation(fig, func, frames=N, ...) 的第 3 个位置参数也是 frames
            # （生成代码常用位置参数而非关键字，漏识别会误杀合格产物）
            if base == "funcanimation" and facts["frames"] is None and len(node.args) >= 3:
                v = _num(node.args[2])
                if v is not None:
                    facts["frames"] = int(v)

        if base == "linspace" and len(node.args) >= 2:
            upper = node.args[1]
            facts["linspace_has_pi"] = facts["linspace_has_pi"] or _is_pi_expr(upper)
            v = _num(upper)
            if v is not None:
                # 收全部上界：3D 曲面/热力图会建 X/Y 两个网格，只留最后一个会漏判
                facts["linspace_uppers"].append(v)
            elif hasattr(ast, "unparse"):
                facts.setdefault("linspace_exprs", []).append(ast.unparse(upper))

        if base == "save" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                facts["save_name"] = a0.value

    # 补充：frames= 也可能是普通赋值后传入，扫一遍 Assign
    if facts["frames"] is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
               and isinstance(node.value.value, int):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and "frame" in tgt.id.lower():
                        facts["frames"] = facts["frames"] or None  # 不采信，交由调用点
    return facts


def _looks_html(code):
    """是不是 HTML/JS 产物。H5 小游戏的交付物是单个 .html。"""
    low = (code or "").lower()
    return ("<html" in low or "<!doctype" in low or "<script" in low
            or "<canvas" in low)


def verify_html(task, code, cons):
    """H5 / HTML5 小游戏的静态验收。

    文本级 token 匹配：不执行、不用 ast（HTML 过不了 ast.parse）。
    管"有没有做"（画布/循环/输入/分数/碰撞），不管"画得对不对"（归识图线）。
    """
    checks = []

    def chk(name, ok, detail, hard=True):
        checks.append({"name": name, "pass": bool(ok), "detail": detail,
                       "score": 1.0 if ok else 0.0, "hard": bool(hard)})

    low = (code or "").lower().replace(" ", "")

    # 1. 是完整 HTML 文档
    has_html = "<html" in low and "</html>" in low
    chk("完整 HTML 文档", has_html,
        "有 <html>/</html>" if has_html else "缺 <html> 或 </html>", hard=True)

    # 2. 单文件自包含：纯本地、双击即玩的硬条件（外链脚本/样式=依赖网络）
    ext = (re.findall(r"<script[^>]+src=", low)
           + re.findall(r'<link[^>]+href=["\']https?://', low))
    chk("单文件自包含", not ext,
        "无外部依赖" if not ext
        else "发现 %d 处外链资源（纯本地游戏不该有）" % len(ext), hard=False)

    # 3. outfile
    if "outfile" in cons:
        want = os.path.basename(cons["outfile"])
        chk("输出文件名符合", want.endswith(".html"),
            "任务要求 %s" % want, hard=True)

    # 4. 游戏要素（分层同 Python 侧）
    if cons.get("h5_symbols"):
        missing = []
        for kw in cons["h5_symbols"]:
            pats = dict(H5_RULES)[kw]
            if not any(p.lower().replace(" ", "") in low for p in pats):
                missing.append(kw)
        hard_missing = [m for m in missing if m in H5_HARD]
        chk("游戏要素齐备", not missing,
            "缺: %s" % "、".join(missing) if missing
            else "全中: %s" % "、".join(cons["h5_symbols"]),
            hard=bool(hard_missing))

    if not checks:
        return {"checks": checks, "quality": 1.0, "pass": True,
                "constraints": cons, "facts_summary": {"h5": True,
                                                       "note": "H5 无可机验约束"}}

    quality = sum(c["score"] for c in checks) / len(checks)
    hard_fail = any(c["hard"] and not c["pass"] for c in checks)
    return {"checks": checks, "quality": round(quality, 4),
            "pass": not hard_fail, "constraints": cons,
            "facts_summary": {"h5": True}}


def verify(task, code):
    """返回 {checks, quality, pass, constraints}。"""
    cons = extract_constraints(task)
    facts = analyze_code(code)
    checks = []

    def chk(name, ok, detail, hard=True, score=None):
        checks.append({"name": name, "pass": bool(ok), "detail": detail,
                       "score": (1.0 if ok else 0.0) if score is None else score,
                       "hard": bool(hard)})

    # H5 路由：HTML/JS 不走 ast（必失败），转文本级 H5 规则
    if _looks_html(code):
        return verify_html(task, code, cons)

    # 任务明确要 H5 小游戏，代码却不是 HTML → 硬拦（Python 冒充 HTML 游戏）
    if cons.get("h5_game"):
        chk("HTML 产物", False,
            "任务要求 H5/网页小游戏，代码却非 HTML 文档", hard=True)
        return {"checks": checks, "quality": 0.0, "pass": False,
                "constraints": cons, "facts_summary": {"h5_expected": True}}

    if not facts["ok_parse"]:
        chk("代码可解析", False, facts["error"] or "parse failed", hard=True)
        return {"checks": checks, "quality": 0.0, "pass": False,
                "constraints": cons, "facts_summary": {"ok_parse": False}}

    # 1. frames
    if "frames" in cons:
        want = cons["frames"]
        got = facts["frames"]
        if got is None:
            chk("帧数符合", False,
                "任务要求 %d 帧，代码里未找到常量 frames=N（可能为变量/表达式）" % want)
        else:
            chk("帧数符合", got == want, "任务 %d 帧 / 代码 %d 帧" % (want, got))

    # 1b. fps（时间帧：任务要求 N fps，代码 save(fps=N) 是否一致，2026-10-19 补）
    if "fps" in cons:
        want = cons["fps"]
        got = facts["fps"]
        if got is None:
            chk("帧率符合", False,
                "任务要求 %d fps，代码里未找到常量 fps=N" % want)
        else:
            chk("帧率符合", got == want, "任务 %d fps / 代码 %d fps" % (want, got))

    # 2. domain（任一 linspace 上界命中即可：3D 曲面会建 X/Y 两个同域网格）
    if "domain_pi" in cons:
        want = cons["domain_pi"] * 3.141592653589793
        uppers = facts["linspace_uppers"]
        if not uppers:
            expr = ", ".join(facts.get("linspace_exprs", [])) or "未找到 linspace"
            chk("定义域符合", False, "任务 0~%gπ，代码上界表达式=%s" % (cons["domain_pi"], expr))
        else:
            ok = any(abs(u - want) < 1e-6 for u in uppers)
            chk("定义域符合", ok, "任务 0~%gπ (%.4f) / 代码上界 %s"
                % (cons["domain_pi"], want, ["%.4f" % u for u in uppers]))

    # 3. outfile
    if "outfile" in cons:
        want = os.path.basename(cons["outfile"])
        got = facts["save_name"]
        ok = bool(got) and os.path.basename(got) == want
        chk("输出文件名符合", ok, "任务 %s / 代码 %s" % (want, got))

    # 4. symbols（分层：缺标准函数→硬拦；缺写法不唯一的→软告警）
    #    取舍：做梦产物重跑一次很便宜（几秒），错误教材入库的污染代价极高
    #    （会被 few-shot 检索反复放大，见 check_pollution）。故宁可严。
    if cons.get("symbols"):
        toks = facts["tokens"]
        missing = []
        for kw in cons["symbols"]:
            pats = dict(SYMBOL_RULES)[kw]
            if not any(p.lower().replace(" ", "") in toks for p in pats):
                missing.append(kw)
        hard_missing = [m for m in missing if m in HARD_SYMBOLS]
        chk("任务要素齐备", not missing,
            "缺: %s" % "、".join(missing) if missing else "全中: %s" % "、".join(cons["symbols"]),
            hard=bool(hard_missing))

    # 4b. 语义套壳拦截（2026-10-19 三轮补）
    #    实测病：远程银行 673 条里 13.4% 是这一种——
    #      #724 task「随机信号折线从左向右延伸」→ 代码 line.set_ydata(np.sin(x+frame/10))
    #      #722 task「阿基米德螺线 r 与 θ 成正比」→ 代码还是 np.sin(x)
    #    为什么 symbols 层拦不住：sin 模板结构完全合法（FuncAnimation+save 齐全、可执行、
    #    出视频），而螺线/随机信号这些要素归 SOFT（写法不唯一，硬判会误杀），只软告警，
    #    于是 quality=0.857 / pass=True 直接入库，再被 few-shot 检索反复放大。
    #    三重连坐判据（只杀套壳，不碰真实现）：
    #      ① task 抽出「非波类」要素 —— 波/光谱用 sin 本就是对的，必须排除，否则误杀
    #      ② 这些非波要素的特征一个都没命中
    #      ③ 代码里确实出现 sin 模板特征（set_ydata(np.sin( / np.sin(x+ / np.sin(x)）
    #    三条同真 ⇒ 模型在拿 sin 模板硬套一个它不是的题，判 task_mismatch 硬拦。
    if cons.get("symbols"):
        toks = facts["tokens"]
        rule_map = dict(SYMBOL_RULES)
        # 波/光谱家族：用 sin/cos 实现就是真做对了
        WAVE_FAMILY = {"余弦", "正切", "方波", "三角波", "衰减", "拍频", "谐波合成",
                       "驻波", "调频", "调幅", "阻尼", "行进波", "干涉", "多普勒",
                       "折射", "色散", "折射角", "混叠", "闪烁", "sinc"}
        # 装饰类零件：缺了是「少个零件」，不是「画错东西」。交给上面的
        # 「任务要素齐备」判，套壳层不兼职（否则措辞会变成「代码只有正弦波」误导）。
        DECOR = {"图例", "文本", "网格", "标题", "填充"}
        non_wave = [kw for kw in cons["symbols"]
                    if kw not in WAVE_FAMILY and kw not in DECOR]
        if non_wave:
            def _hit(kw):
                return any(p.lower().replace(" ", "") in toks for p in rule_map.get(kw, []))

            missing_all = [kw for kw in non_wave if not _hit(kw)]
            SIN_TPL = ("set_ydata(np.sin(", "np.sin(x+", "np.sin(x)", "np.sin(x,")
            is_sin_shell = any(p in toks for p in SIN_TPL)
            if len(missing_all) == len(non_wave) and is_sin_shell:
                chk("语义套壳拦截", False,
                    "task 要「%s」，代码只有正弦波相位平移，无对应实现" % "、".join(non_wave),
                    hard=True)

    # 5. animated
    if cons.get("animated"):
        ok = facts["has_anim"] and bool(facts["save_name"])
        chk("产出动画", ok,
            "FuncAnimation=%s save=%s" % (facts["has_anim"], facts["save_name"]))

    if not checks:
        return {"checks": checks, "quality": 1.0, "pass": True,
                "constraints": cons, "facts_summary": {"note": "无可机验约束，跳过"}}

    quality = sum(c["score"] for c in checks) / len(checks)
    hard_fail = any(c["hard"] and not c["pass"] for c in checks)
    return {"checks": checks, "quality": round(quality, 4), "pass": not hard_fail,
            "constraints": cons,
            "facts_summary": {k: v for k, v in facts.items() if k != "tokens"}}


def selftest():
    """正例 + 反例自检。只有反例测得出「漏检」，测不出「误杀」——
    误杀比漏检更贵（会把合格产物挡在银行外，做梦白跑），所以正例必须一起测。"""
    GOOD = [
        ("用 matplotlib 画一个衰减正弦波动画(0~6π, 120 帧，振幅随时间衰减)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 6 * np.pi, 200)\n"
         "line, = ax.plot(x, np.exp(-0.05 * x) * np.sin(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.exp(-0.05 * x) * np.sin(x + i / 10.0))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=120, blit=False)\n"
         "ani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        ("用 matplotlib 画正弦+余弦叠加动画(0~4π, 100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4 * np.pi, 300)\n"
         "l1, = ax.plot(x, np.sin(x))\nl2, = ax.plot(x, np.cos(x))\n\ndef upd(i):\n"
         "    l2.set_ydata(np.cos(x + i / 9.0))\n    return l1, l2\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        ("随便画一张有意思的图",   # 无可机验约束 → 应放行（不能因为抽不出约束就卡死）
         "import matplotlib\nmatplotlib.use('Agg')\nprint('hi')\n"),
        # 半衰期写法 (1/2)**(t/T)：验证「衰减」规则已补半衰期（2026-10-19）
        ("用 matplotlib 画放射性衰变动画：原子核数按半衰期指数衰减(150 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nt = np.linspace(0, 10, 200)\nN0 = 100.0\nT = 2.0\n"
         "N = N0 * (1/2)**(t/T)\nline, = ax.plot(t, N)\n\ndef upd(i):\n"
         "    line.set_ydata(N0 * (1/2)**(t[:i+1]/T))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=150)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # 填充 + 标题：验证元素级规则（2026-10-19）
        ("用 matplotlib 画正弦波动画并填充曲线下面积(带标题'面积', 0~4π, 100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4 * np.pi, 200)\n"
         "ax.fill_between(x, np.sin(x), alpha=0.3)\nax.set_title('面积')\n"
         "line, = ax.plot(x, np.sin(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.sin(x + i/10.0))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # --- H5 小游戏正例（2026-10-19）：完整可玩，应放行 ---
        ("写一个 HTML 小游戏：canvas 画布上有一个小球，鼠标点击画布让小球跳起，"
         "碰到障碍物扣分，分数归零游戏结束，按 R 重新开始。保存为 game.html",
         "<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>跳球</title></head>\n"
         "<body><canvas id='c' width='400' height='600'></canvas>\n"
         "<script>\n"
         "var c=document.getElementById('c'),g=c.getContext('2d');\n"
         "var score=100,vy=0,gravity=0.5;\n"
         "var obs=[{x:300,y:400,w:20,h:80}];\n"
         "function loop(){g.clearRect(0,0,400,600);\n"
         "  g.fillRect(50,300,20,20);\n"
         "  requestAnimationFrame(loop);}\n"
         "c.addEventListener('click',function(){vy=-8;});\n"
         "document.addEventListener('keydown',function(e){if(e.key==='r')restart();});\n"
         "function restart(){score=100;}\n"
         "function checkCollision(a,b){return a.x<b.x+b.w&&a.y<b.y+b.h;}\n"
         "if(score<=0){gameOver=true;}\n"
         "loop();\n"
         "</script></body></html>\n"),
        # --- H5 正例：贪吃蛇（键盘 + 网格 + 食物 + 碰撞）---
        ("写一个 HTML 贪吃蛇小游戏：canvas 网格，方向键控制蛇移动，吃到食物加分，"
         "撞墙游戏结束。保存为 snake.html",
         "<!DOCTYPE html>\n<html><body><canvas id='g' width='400' height='400'></canvas>\n"
         "<script>\n"
         "var cv=document.getElementById('g'),ctx=cv.getContext('2d');\n"
         "var cell=20,cols=20,rows=20,score=0,snake=[],food={x:5,y:5};\n"
         "var grid=[];\n"
         "document.addEventListener('keydown',function(e){"
         "if(e.key==='ArrowUp'||e.key==='w')dir={x:0,y:-1};});\n"
         "function tick(){ctx.clearRect(0,0,400,400);\n"
         "  snake.forEach(function(s){ctx.fillRect(s.x*cell,s.y*cell,cell,cell);});\n"
         "  ctx.fillRect(food.x*cell,food.y*cell,cell,cell);\n"
         "  setTimeout(tick,120);}\n"
         "function checkCollision(h){return h.x<0||h.y<0;}\n"
         "if(score>10){gameOver=true;}\n"
         "tick();\n</script></body></html>\n"),
        # --- 二轮补规则回归用例（2026-10-19）---
        # 对数函数：np.log 是唯一标准 API
        ("用 matplotlib 画对数函数增长动画 y=ln(x) (0.1~50, 100 帧，与 y=x 对比)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0.1, 50, 200)\n"
         "line, = ax.plot(x, np.log(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.log(x[:i+1]))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # 双缝干涉：interference/fringe
        ("用 matplotlib 画杨氏双缝干涉动画：两列相干余弦波叠加，屏幕上形成明暗相间干涉条纹且随波长变化(120 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nxs = np.linspace(-1, 1, 400)\n\n"
         "def upd(i):\n    lam = 0.5 + i * 0.001\n"
         "    y = np.cos(np.pi * xs / lam) ** 2   # interference fringe pattern\n"
         "    ax.clear()\n    ax.plot(xs, y)\n    return []\n\n"
         "ani = FuncAnimation(fig, upd, frames=120)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # frames 用位置参数传 FuncAnimation(fig, upd, 100)：验证位置参数识别（2026-10-19 修）
        ("用 matplotlib 画一个正弦波动画(0~4π, 100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4 * np.pi, 200)\n"
         "line, = ax.plot(x, np.sin(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.sin(x + i / 10.0))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, 100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # 条形图用 bar()（中文任务池混用，模型统一用 bar）：验证「条形」归 HARD 后 bar() 仍放行
        ("用 matplotlib 画一个条形图：5 个分类的数值对比",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "fig, ax = plt.subplots()\n"
         "ax.bar([0,1,2,3,4], [5,3,8,2,7])\n"),
        # fps 约束（任务 10 fps，代码 save(fps=10)）：验证 fps 静态检查（2026-10-19 补）
        ("用 matplotlib 画一个正弦波动画(0~4π, 100 帧, 10 fps)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4 * np.pi, 200)\n"
         "line, = ax.plot(x, np.sin(x))\n\ndef upd(i):\n"
         "    line.set_ydata(np.sin(x + i / 10.0))\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # --- 三轮正例（2026-10-19）：套壳拦截不能误杀真实现 ---
        # 阿基米德螺线真用极坐标 r=aθ → 必须放行（不能因为出现 np.sin 就拦）
        ("用 matplotlib 画阿基米德螺线 r 与 θ 成正比 (0~4π, 100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\ntheta = np.linspace(0, 4 * np.pi, 300)\n"
         "r = 0.5 * theta\nx, y = r * np.cos(theta), r * np.sin(theta)\n"
         "line, = ax.plot(x[:2], y[:2])\n\ndef upd(i):\n"
         "    line.set_data(x[:i+2], y[:i+2])\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
        # 随机信号真用 np.random → 必须放行
        ("用 matplotlib 画一个随机信号生成动画：折线信号从左向右逐帧延伸 (100 帧)，保存为 out.mp4",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\nimport numpy as np\n\n"
         "fig, ax = plt.subplots()\nsig = np.random.randn(100)\n"
         "line, = ax.plot([], [])\n\ndef upd(i):\n"
         "    line.set_data(np.arange(i+1), sig[:i+1])\n    return line,\n\n"
         "ani = FuncAnimation(fig, upd, frames=100)\nani.save('out.mp4', writer='ffmpeg', fps=10)\n"),
    ]
    BAD = [
        ("用 matplotlib 画一个正弦波动画(0~4π, 100 帧)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=40)\n"
         "ani.save('out.mp4')\n", "帧数"),
        ("用 matplotlib 画正弦+余弦叠加的动画(0~4π, 100 帧)，保存为 out.mp4",
         "import numpy as np\nfrom matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\n"
         "ani=FuncAnimation(None, lambda i: None, frames=100)\nani.save('out.mp4')\n", "要素"),
        # sin 冒充衰减（无 exp/半衰期）：验证「衰减」规则仍能拦（2026-10-19）
        ("用 matplotlib 画一个衰减正弦波动画(0~6π, 120 帧，振幅随时间衰减)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,6*np.pi,200)\nline,=plt.plot(x, np.sin(x))\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=120)\n"
         "ani.save('out.mp4')\n", "衰减"),
        # --- H5 反例（2026-10-19）：有 canvas 但没循环/没输入/没游戏结束 → 应拦住 ---
        ("写一个 HTML 小游戏：canvas 画布上有一个小球，鼠标点击画布让小球跳起，"
         "碰到障碍物扣分，分数归零游戏结束，按 R 重新开始。保存为 game.html",
         "<!DOCTYPE html>\n<html><body><canvas id='c' width='400' height='600'></canvas>\n"
         "<script>\nvar g=document.getElementById('c').getContext('2d');\n"
         "g.fillRect(50,300,20,20);\n</script></body></html>\n", "H5要素"),
        # --- H5 反例：不是 HTML 文档，却宣称 H5 任务 → 应拦住 ---
        ("写一个 HTML 小游戏：canvas 画布上画个圆，点击开始，保存为 game.html",
         "print('hello')\n", "HTML文档"),
        # --- 二轮反例：sin 冒充对数（无 np.log）→ 应拦住 ---
        ("用 matplotlib 画对数函数增长动画 y=ln(x) (0.1~50, 100 帧，与 y=x 对比)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0.1,50,200)\nline,=plt.plot(x, np.sin(x))\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=100)\n"
         "ani.save('out.mp4')\n", "对数"),
        # --- 二轮反例：sin 冒充干涉（无 interference/fringe）→ 应拦住 ---
        ("用 matplotlib 画杨氏双缝干涉动画：两列相干余弦波叠加形成干涉条纹(120 帧)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(-1,1,400)\nline,=plt.plot(x, np.sin(x))\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=120)\n"
         "ani.save('out.mp4')\n", "干涉"),
        # 任务要 H5 小游戏，代码却是 Python（能过 Python 分支）→ h5_game 硬拦（2026-10-19 修）
        ("写一个 HTML 贪吃蛇小游戏：canvas 网格，方向键控制蛇移动，吃到食物加分，撞墙游戏结束。保存为 snake.html",
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
         "fig, ax = plt.subplots()\nx = np.linspace(0, 4*np.pi, 200)\n"
         "line, = ax.plot(x, np.sin(x))\n"
         "fig.savefig('snake.html')\n", "HTML产物"),
        # sin 冒充圆周运动（无 omega/角速度/圆轨道）→ 圆周运动 HARD 拦（2026-10-19 收紧「匀速」）
        ("用 matplotlib 画一个匀速圆周运动动画：小球绕圆心做圆周运动(100 帧)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\nline,=plt.plot(x, np.sin(x))\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=100)\n"
         "ani.save('out.mp4')\n", "圆周运动"),
        # 任务 10 fps 但代码 save(fps=5) → fps 静态检查拦（2026-10-19 补）
        ("用 matplotlib 画一个正弦波动画(0~4π, 100 帧, 10 fps)，保存为 out.mp4",
         "import numpy as np\nimport matplotlib.pyplot as plt\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x=np.linspace(0,4*np.pi,200)\nline,=plt.plot(x, np.sin(x))\n"
         "ani=FuncAnimation(plt.gcf(), lambda i: None, frames=100)\n"
         "ani.save('out.mp4', fps=5)\n", "帧率"),
        # --- 三轮反例（2026-10-19）：语义套壳，sin 模板硬套非波任务 → 必须拦 ---
        # 真实病历 #722：task 要螺线，代码是 np.sin(x) 相位平移
        ("用 matplotlib 画一个阿基米德螺线绘制动画：r 与 θ 成正比的螺线逐帧绘制 (100 帧)，保存为 out.mp4",
         "import matplotlib.pyplot as plt\nimport numpy as np\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x = np.linspace(0, 4*np.pi, 200)\nline, = plt.plot(x, np.sin(x))\n"
         "def upd(i):\n    line.set_ydata(np.sin(x + i/10.0))\n    return line,\n"
         "ani = FuncAnimation(plt.gcf(), upd, frames=100)\n"
         "ani.save('out.mp4', writer='ffmpeg')\n", "语义套壳"),
        # 真实病历 #724：task 要随机信号，代码是 np.sin(x)
        ("用 matplotlib 画一个随机信号生成动画：折线信号从左向右逐帧延伸 (100 帧)，保存为 out.mp4",
         "import matplotlib.pyplot as plt\nimport numpy as np\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x = np.linspace(0, 4*np.pi, 200)\nline, = plt.plot(x, np.sin(x))\n"
         "def upd(i):\n    line.set_ydata(np.sin(x + i/10.0))\n    return line,\n"
         "ani = FuncAnimation(plt.gcf(), upd, frames=100)\n"
         "ani.save('out.mp4', writer='ffmpeg')\n", "语义套壳"),
        # task 要一维热传导，代码是 sin 波 → 套壳
        ("用 matplotlib 画一维热传导动画：金属杆初始中间热两端冷，温度分布随热传导逐渐均匀 (150 帧)，保存为 out.mp4",
         "import matplotlib.pyplot as plt\nimport numpy as np\n"
         "from matplotlib.animation import FuncAnimation\n"
         "x = np.linspace(0, 4*np.pi, 200)\nline, = plt.plot(x, np.sin(x))\n"
         "def upd(i):\n    line.set_ydata(np.sin(x + i/10.0))\n    return line,\n"
         "ani = FuncAnimation(plt.gcf(), upd, frames=150)\n"
         "ani.save('out.mp4', writer='ffmpeg')\n", "语义套壳"),
    ]
    n_ok = 0
    for task, code in GOOD:
        r = verify(task, code)
        ok = r["pass"]
        n_ok += 1 if ok else 0
        print("  [%s] 正例  %s" % ("PASS" if ok else "FAIL", task[:44]))
        if not ok:
            for ch in r["checks"]:
                if not ch["pass"]:
                    print("         ! %s: %s" % (ch["name"], ch["detail"]))
    for task, code, expect_catch in BAD:
        r = verify(task, code)
        ok = not r["pass"]
        n_ok += 1 if ok else 0
        print("  [%s] 反例应拦住(%s)  %s" % ("PASS" if ok else "FAIL", expect_catch, task[:34]))
    total = len(GOOD) + len(BAD)
    print("\n自检 %d/%d" % (n_ok, total))
    return n_ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--code-file")
    ap.add_argument("--bank", action="store_true", help="回扫 triple_bank 全表")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "triple_bank.db"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.bank:
        c = sqlite3.connect(a.db)
        rows = list(c.execute("select id, task, code_text from triples order by id"))
        results, npass = [], 0
        for i, task, code in rows:
            r = verify(task, code)
            r["id"] = i
            r["task"] = task
            results.append(r)
            npass += 1 if r["pass"] else 0
        if a.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print("=" * 78)
            print("triple_bank 任务符合度回扫  共 %d 条  通过 %d  失败 %d"
                  % (len(rows), npass, len(rows) - npass))
            print("=" * 78)
            for r in results:
                print("\n#%s %s" % (r["id"], "PASS ✅" if r["pass"] else "FAIL ❌"))
                print("  task: %s" % r["task"])
                for ch in r["checks"]:
                    mark = "PASS" if ch["pass"] else ("FAIL" if ch["hard"] else "WARN")
                    print("    [%-4s] %-12s %s" % (mark, ch["name"], ch["detail"]))
        sys.exit(0 if npass == len(rows) else 1)

    if not a.code_file:
        ap.error("需要 --code-file 或 --bank")
    code = open(a.code_file, encoding="utf-8", errors="replace").read()
    task = a.task or ""
    r = verify(task, code)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print("任务:", task)
        print("=" * 70)
        print("约束:", json.dumps(r["constraints"], ensure_ascii=False))
        for ch in r["checks"]:
            mark = "PASS" if ch["pass"] else ("FAIL" if ch["hard"] else "WARN")
            print("  [%-4s] %-12s %s" % (mark, ch["name"], ch["detail"]))
        print("  结论: %s   quality=%.2f" % ("PASS ✅" if r["pass"] else "FAIL ❌", r["quality"]))
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
