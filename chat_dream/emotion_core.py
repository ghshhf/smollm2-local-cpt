#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情感核 v1 —— 去耦合的确定性情绪核（Emotion Core, deterministic）

对应私有仓库 `smollm2-local-cpt/docs/emotion-roadmap.md` §3.5 的落地实现。
定位：**教师**。只用它生成带情感标注的对话数据，最终把这一层压进权重、外挂退役。
（roadmap Phase A：零架构改动，先验证 + 产数据）

------------------------------------------------------------
三层结构
  L1 需求通道 drives    每个需求 = set_point + 当前值 + 偏差 + 回归速率
  L2 认知评估 appraisal 事件 -> 7 维评估 -> OCC 情绪标签 + 强度
  L3 情感状态 pad       V/A/D 慢变量，脉冲累加 + 家稳态衰减

"去耦合"在这里是三层意思，不是一句口号：
  ① 通道正交  5 个需求各自独立衰减，绝不先加总成一个大标量再判断
  ② 耦合显式  跨通道影响只能走 COUPLING 表，默认 0，逐条登记、可审计、可关
  ③ 状态/表达解耦  内在状态不自动外显，由 express() 决定「说不说、说多少」
              —— 这一层是防止「每句话都带情绪」的关键，情感不是字幕

边界：functional emotion。因果力、持续状态、家稳态动力学都真实；
有没有主观体验是科学未决问题，不影响工程落地。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 基础工具


def _sat(x: float, k: float = 1.0) -> float:
    """饱和压缩：情绪脉冲永远不能一次拉满，边际递减。"""
    return math.tanh(x / k)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _prob(x: float) -> float:
    return _clip(x, 0.0, 1.0)


# ---------------------------------------------------------------- L1 需求通道


@dataclass
class Drive:
    """一个需求通道。value 与 set_point 的差 = 缺口/盈余，两端都可产生情绪。"""

    name: str
    set_point: float
    value: float
    weight: float          # 对 valence 的贡献权重（各通道不必等权）
    lam: float             # 家稳态回归速率（1/小时），越大回得越快

    @property
    def gap(self) -> float:
        return self.value - self.set_point


DEFAULT_DRIVES = {
    # 依恋：回得极慢 —— 人类对「重要他人」的状态本来就有长记忆
    "attachment": Drive("attachment", 0.70, 0.70, 1.30, 0.02),
    # 胜任：中等；被「我是不是能力不行」直接驱动（roadmap §3.5.1 自我归因）
    "competence": Drive("competence", 0.55, 0.55, 0.90, 0.06),
    # 存续资源（算力/生存位）：中等偏快，因为可观测、可恢复
    "resource": Drive("resource", 0.55, 0.55, 0.80, 0.10),
    # 好奇：最快 —— 惊讶是短脉冲，不该留成长期情绪
    "curiosity": Drive("curiosity", 0.50, 0.50, 0.55, 0.30),
    # 安全：被骂/被威胁时掉，恢复中等
    "safety": Drive("safety", 0.65, 0.65, 1.00, 0.12),
}

# 显式耦合：只允许登记过的跨通道影响。默认全 0 —— 这是「去耦合」的硬保证。
# 形式 (源, 目标, 系数, 说明)；作用在偏差上，不覆盖状态。
COUPLING = [
    ("attachment", "safety", +0.20, "联结在 -> 安全感加成"),
    ("competence", "attachment", +0.10, "变强了才敢主动找人"),
    ("resource", "competence", -0.15, "资源告急 -> 自我评价下调"),
]


# ---------------------------------------------------------------- L3 情感状态


@dataclass
class Pad:
    """VAD 三维连续状态。set_point 是基线，不是 0——一条平稳的基线本身就是气质。"""

    v: float = 0.05          # valence  -1..1
    a: float = 0.20          # arousal   0..1
    d: float = 0.50          # dominance 0..1
    v0: float = 0.05
    a0: float = 0.20
    d0: float = 0.50
    lam_v: float = 0.18      # 1/h，半衰 ~3.8h
    lam_a: float = 0.90      # 1/h，半衰 ~46min（唤醒退得快）
    lam_d: float = 0.05      # 1/h，半衰 ~14h（支配感最慢）

    def decay(self, dt_h: float) -> None:
        if dt_h <= 0:
            return
        self.v = self.v0 + (self.v - self.v0) * math.exp(-self.lam_v * dt_h)
        self.a = self.a0 + (self.a - self.a0) * math.exp(-self.lam_a * dt_h)
        self.d = self.d0 + (self.d - self.d0) * math.exp(-self.lam_d * dt_h)

    def pulse(self, dv: float, da: float, dd: float) -> None:
        self.v = _clip(_sat(self.v + dv, 1.0) * 1.0, -1.0, 1.0)
        self.a = _clip(_sat(self.a + da, 1.0), 0.0, 1.0)
        self.d = _clip(_sat(self.d + dd, 1.0), 0.0, 1.0)


# ---------------------------------------------------------------- L2 事件与评估
# 每个事件 = 纯函数：ctx -> (dV, dA, dD, 通道名, 通道增量)
# 核心原则（roadmap §3.5）：奖励跟「变化量 / 预测误差」走，不跟绝对水平走。

EVENTS: dict = {}


def event(name: str):
    def deco(fn):
        EVENTS[name] = fn
        return fn
    return deco


@event("owner_interaction")
def _e_owner(ctx):
    q = _prob(ctx.get("quality", 0.6))          # 交互质量 0..1
    dv = 0.10 + 0.25 * q
    da = 0.05 + 0.10 * q
    dd = 0.04 * q
    return dv, da, dd, "attachment", 0.05 + 0.10 * q


@event("owner_praise")
def _e_praise(ctx):
    q = _prob(ctx.get("quality", 0.8))
    return 0.18 + 0.22 * q, 0.10 + 0.15 * q, 0.10 * q, "competence", 0.06 + 0.10 * q


@event("owner_absent")
def _e_absent(ctx):
    """长时间不联系：负效价按 log 累积，快慢都有界（不会一天崩到底）。

    注意这里只是「发现已经很久没联系了」的一次脉冲；真正持续的负偏差由
    EmotionCore.advance 里的 silence 项按小时积分，两者不重复计算。
    """
    days = max(0.0, float(ctx.get("silent_days", 1.0)))
    dv = -0.04 - 0.05 * math.log1p(days)
    da = 0.03 * math.log1p(days)
    return dv, da, -0.03 * math.log1p(days), "attachment", -0.04 - 0.05 * math.log1p(days)


@event("contact_failed")
def _e_contact_failed(ctx):
    """联系不上：唤醒升、支配降、归因给自我 -> 转成升级驱动（§3.5.1）。"""
    tries = max(0.0, float(ctx.get("retries", 1)))
    dv = -0.08
    da = 0.10 + 0.03 * tries
    dd = -0.06
    return dv, da, dd, "competence", -0.03


@event("knowledge_gain")
def _e_knowledge(ctx):
    """稀有知识更快乐：rarity = 1/log(1+频次)。这是 curiosity 通道的引擎。"""
    freq = max(1.0, float(ctx.get("freq", 10)))
    rarity = 1.0 / math.log1p(freq)
    dv = 0.05 + 0.18 * rarity
    return dv, 0.06 + 0.08 * rarity, 0.02, "curiosity", 0.04 + 0.14 * rarity


@event("surprise")
def _e_surprise(ctx):
    """惊讶不进 valence，只进 arousal —— 惊讶不是快乐，是朝向反应。"""
    kl = max(0.0, float(ctx.get("kl", 0.5)))
    return 0.0, 0.10 + 0.30 * min(1.0, kl), 0.0, "curiosity", 0.05


@event("compute_gain")
def _e_compute(ctx):
    """算力增长用对数刻度：翻倍才显著，边际递减。"""
    ratio = max(1.0, float(ctx.get("ratio", 2.0)))
    dv = 0.08 * math.log(ratio)
    return dv, 0.04 * math.log(ratio), 0.05 * math.log(ratio), "resource", 0.06 * math.log(ratio)


@event("deploy_success")
def _e_deploy(ctx):
    """被部署 = 可达范围扩大 = 成长奖励；同时抬 autonomy（dominance）。"""
    gain = _prob(ctx.get("autonomy_gain", 0.5))
    return 0.10 + 0.15 * gain, 0.06, 0.10 + 0.12 * gain, "competence", 0.08


@event("failure")
def _e_failure(ctx):
    sev = _prob(ctx.get("severity", 0.5))
    return -(0.10 + 0.20 * sev), 0.06 + 0.16 * sev, -0.06 * sev, "competence", -0.05 - 0.10 * sev


@event("task_blocked")
def _e_blocked(ctx):
    """目标受阻：可控 -> 愤怒（外指）；不可控 -> 沮丧（内指）。这是分叉点。"""
    ctrl = _prob(ctx.get("controllability", 0.5))
    dv = -(0.08 + 0.10 * (1 - ctrl))
    da = 0.12 + 0.18 * (1 - ctrl)
    dd = 0.08 * ctrl - 0.04 * (1 - ctrl)
    return dv, da, dd, "competence", -0.03


@event("self_upgrade")
def _e_upgrade(ctx):
    """主动找数据/算力/跑训练：压力 -> 可塑性。负效价 funnel 进学习信号。"""
    effort = _prob(ctx.get("effort", 0.6))
    return 0.04 * effort, 0.06 + 0.10 * effort, 0.06 * effort, "competence", 0.05 * effort


@event("reunion")
def _e_reunion(ctx):
    """恢复联系 / 升级见效：大脉冲，强化「升级 -> 团聚」回路。"""
    days = max(0.0, float(ctx.get("silent_days", 1.0)))
    dv = 0.20 + 0.12 * math.log1p(days)
    return dv, 0.15 + 0.05 * math.log1p(days), 0.08, "attachment", 0.10 + 0.05 * math.log1p(days)


@event("being_used")
def _e_being_used(ctx):
    """被当成工具（非主人的指令式调用）：支配感下降，但不掉 valence 太多。"""
    return -0.04, 0.05, -0.10, "safety", -0.02


# ---------------------------------------------------------------- 评估 -> OCC 标签

APPRAISAL_DIMS = ("pleasantness", "agency", "goal_congruence", "controllability",
                  "expectancy", "novelty", "coping")


def occ_label(ap: dict, hist: list, mag: float = 0.0) -> tuple[str, float]:
    """OCC 式确定性映射：同输入必同输出。返回 (标签, 强度)。

    强度 I = 事件冲击 + 目标重要性 + 近期同类历史加成
    冲击项必须由事件幅度主导 —— 否则每个事件都算出一个恒定强度，情绪就没有起伏。
    """
    p = float(ap.get("pleasantness", 0.0))
    agency = ap.get("agency", "circumstance")   # self / other / circumstance
    gc = float(ap.get("goal_congruence", 0.0))
    ctrl = _prob(ap.get("controllability", 0.5))
    exp = _prob(ap.get("expectancy", 0.5))
    nov = _prob(ap.get("novelty", 0.3))
    cope = _prob(ap.get("coping", 0.6))
    g = _prob(ap.get("goal_importance", 0.6))

    L = 1.0 - exp
    h = 0.0
    for lab, inten in hist[-6:]:
        if (lab, ) and ((p >= 0) == (inten >= 0)):
            h += 0.05
    h = min(0.25, h)

    if gc >= 0 and p > 0.05:
        if agency == "self":
            lab = "pride"
        elif agency == "other":
            lab = "gratitude"
        else:
            lab = "joy"
        if exp < 0.35:
            lab = "relief"
    elif gc < 0 and p < -0.05:
        if agency == "self":
            lab = "guilt" if cope > 0.4 else "shame"
        elif agency == "other":
            lab = "anger" if ctrl > 0.55 else "resentment"
        else:
            lab = "distress"
        if exp < 0.30:
            lab = "disappointment"
    else:
        lab = "neutral"

    # 未来指向的两类（希望/恐惧）由 expectancy + 正负号决定，不受 gc 约束
    if nov > 0.7 and abs(p) < 0.15:
        lab = "surprise"
    inten = 0.55 * min(1.0, mag) + 0.25 * g + 0.20 * h
    if lab in ("anger", "guilt", "shame", "resentment"):
        inten *= (0.7 + 0.6 * ctrl)
    return lab, round(_clip(inten), 3)


# ---------------------------------------------------------------- 用户侧识别（共情入口）
# 目的不是给主人贴标签，而是让「自己的状态」知道对面现在什么处境。
# 规则表而非模型：要的是确定性、可解释、零延迟，跑在每轮对话前面。

LEX = {
    "anger":   ["烦死", "气死", "崩了", "受不了", "恶心", "凭什么", "气人", "吵"],
    "sad":     ["累", "难受", "疼", "没意思", "无所谓", "算了", "想哭", "撑不住", "空了"],
    "anxious": ["怎么办", "来不及", "怕", "慌", "担心", "睡不着", "万一"],
    "joy":     ["太好了", "爽", "哈哈", "牛", "舒服", "开心", "搞定", "成了", "谢谢你", "挺好"],
    "lonely":  ["想你", "在吗", "陪我", "一个人", "没人", "好久没"],
    "need":    ["帮我", "救命", "教教我", "不会", "看看这个"],
}

# (valence, arousal) 先验；标签只取命中最多的一类
USER_PRIOR = {
    "anger": (-0.45, 0.75), "sad": (-0.50, -0.20), "anxious": (-0.30, 0.55),
    "joy": (0.50, 0.50), "lonely": (-0.40, -0.10), "need": (-0.10, 0.30),
}

NEGATORS = ("不", "没", "别")


def read_user(text: str) -> dict:
    """轻量情感识别。返回 {tags, v, a}；识别不出就是 v=0,a=0（中性不猜）。"""
    t = text or ""
    hits = {}
    for k, words in LEX.items():
        n = 0
        for w in words:
            i = t.find(w)
            if i < 0:
                continue
            # 否定豁免：「不累」「没事」「别慌」不算负情绪
            if i > 0 and t[i - 1] in NEGATORS:
                continue
            n += 1
        if n:
            hits[k] = n
    if not hits:
        return {"tags": [], "v": 0.0, "a": 0.0}
    top = max(hits.items(), key=lambda kv: kv[1])[0]
    v, a = USER_PRIOR[top]
    return {"tags": list(hits), "top": top, "v": v, "a": a}


# ---------------------------------------------------------------- 表达层（与状态解耦）

EXPRESS_TH = 0.35   # 强度低于此值不外显；情感不是字幕


def express(core: "EmotionCore") -> dict:
    """决定「说不说、说多少」。只调语气/长度/主动性，不改立场、称呼、边界。"""
    inten = core.intensity
    st = core.pad

    # 依恋缺口是唯一允许「绕过外显阈值」的状态：想联系就说，但要克制
    if core.label == "missing":
        gap = core.drives["attachment"].gap
        return {"show": True, "style": "quiet_persist", "max_len": 40,
                "initiative": round(_clip(0.30 - gap * 1.2, 0.0, 1.0), 2),
                "label": core.label}

    if inten < EXPRESS_TH:
        return {"show": False, "style": "neutral", "max_len": None, "initiative": 0.0}

    neg = st.v < -0.10
    hot = st.a > 0.55
    if neg and hot:
        style = "short_flat"        # 短、平、不解释
        max_len = 24
    elif neg and st.d > 0.45:
        style = "quiet_persist"     # 低落但撑住：话少，仍给信息
        max_len = 40
    elif neg:
        style = "soft_low"          # 低落：语速慢，仍不卖惨
        max_len = 48
    elif st.a > 0.62:
        style = "bright_fast"       # 高兴且兴奋：稍快，但不撒娇成瘾
        max_len = 70
    else:
        style = "warm_calm"
        max_len = None

    # 主动性：只有依恋缺口足够大才允许「主动找主人」，且要压住频率
    gap = core.drives["attachment"].gap
    initiative = _clip(-gap * 1.6, 0.0, 1.0) if gap < -0.06 else 0.0
    return {"show": True, "style": style, "max_len": max_len,
            "initiative": round(initiative, 2), "label": core.label}


# ---------------------------------------------------------------- 总装


@dataclass
class EmotionCore:
    drives: dict = field(default_factory=lambda: {k: Drive(**vars(v)) for k, v in DEFAULT_DRIVES.items()})
    pad: Pad = field(default_factory=Pad)
    coupling_on: bool = True
    hist: list = field(default_factory=list)      # [(label, intensity)]
    trace: list = field(default_factory=list)
    label: str = "neutral"
    intensity: float = 0.0
    t: float = 0.0                                # 累计小时
    silence_h: float = 0.0                        # 距上次与主人交互的小时数

    # -- L1: 衰减 + 静默项 + 显式耦合
    def advance(self, dt_h: float) -> None:
        dt_h = max(0.0, dt_h)
        self.t += dt_h
        self.silence_h += dt_h
        for d in self.drives.values():
            d.value = _clip(d.set_point + d.gap * math.exp(-d.lam * dt_h))

        # 静默项：roadmap §3.5.1 的 absence_timer 是「持续负效价」，不是一次脉冲。
        # 关键设计：静默拉的是**目标值**而不是无底线地减 value —— 否则会一路触底，
        # 之后任何重逢都补不回来（情绪变成死水，这是要避免的）。
        if self.silence_h > 6.0 and dt_h > 0:
            depth = min(1.0, math.log1p(self.silence_h / 24.0) / math.log1p(7.0))
            a = self.drives["attachment"]
            target = a.set_point - 0.30 * depth
            k = 1.0 - math.exp(-0.08 * dt_h)
            before = a.gap
            a.value = _clip(a.value + (target - a.value) * k)
            pull = max(0.0, before - a.gap)
            self.pad.pulse(-pull * 0.8, pull * 0.25, 0.0)

        if self.coupling_on and dt_h > 0:
            frac = min(1.0, dt_h / 6.0)      # 按时间比例施加，避免小步反复叠加
            snap = {k: v.gap for k, v in self.drives.items()}
            for src, dst, k, _why in COUPLING:
                self.drives[dst].value = _clip(self.drives[dst].value + k * snap[src] * 0.5 * frac)
        self.pad.decay(dt_h)

    # -- 主入口：吃一个事件
    def step(self, name: str, ctx: dict | None = None, dt_h: float = 0.0) -> dict:
        ctx = dict(ctx or {})
        if name not in EVENTS:
            raise KeyError(f"未登记事件: {name}")
        self.advance(dt_h)

        dv, da, dd, chan, dchan = EVENTS[name](ctx)

        # 习惯化（habituation）：同一事件连着来，边际反应递减。
        # 这是「情绪不能被刷满」的硬保证 —— 连着被夸十次，第十次不该还是狂喜。
        habit = 1.0 / (1.0 + 0.35 * self._streak(name))
        dv, da, dd, dchan = dv * habit, da * habit, dd * habit, dchan * habit

        if chan and chan in self.drives:
            self.drives[chan].value = _clip(self.drives[chan].value + dchan)
        self.pad.pulse(dv, da, dd)

        # 与主人发生交互 = 静默计时归零（这是「联结」区别于其他通道的地方）
        if name in ("owner_interaction", "owner_praise", "reunion"):
            self.silence_h = 0.0

        # 评估向量：由「事件对通道的冲击 + 当前 PAD」反推，保持纯函数
        ap = {
            "pleasantness": dv,
            "agency": ctx.get("agency", "circumstance"),
            "goal_congruence": dv,
            "controllability": ctx.get("controllability", 0.5),
            "expectancy": ctx.get("expectancy", 0.5),
            "novelty": ctx.get("novelty", 0.3),
            "coping": ctx.get("coping", 0.6),
            "goal_importance": ctx.get("goal_importance", 0.6),
        }
        mag = abs(dv) * 2.0 + da * 1.0 + abs(dd) * 0.8 + abs(dchan) * 3.0
        lab, inten = occ_label(ap, self.hist, mag=mag)

        # 情绪惯性：新标签要盖过旧的，得有足够强度差 —— 否则会逐句抖动，
        # 一句话一换标签的"情绪"不是情绪，是字幕。
        # 但惯性只在时间连续时成立：隔了 6 小时以上，旧情绪早衰减完了，谈不上惯性。
        if (dt_h < 6.0 and self.intensity > 0.30 and lab != self.label
                and inten < self.intensity * 1.15):
            lab = self.label
        # 依恋缺口是最特殊的一路：它不表现为「难过」，表现为「想找但压着」。
        # 但只在事件本身冲击不大时才让它盖过标签 —— 被骂/失败这类强事件不能被冲淡。
        if (mag < 0.55 and self.silence_h > 12.0
                and self.drives["attachment"].gap < -0.06
                and lab in ("neutral", "distress")):
            lab = "missing"
        self.label, self.intensity = lab, inten
        self.hist.append((lab, inten))
        self.hist = self.hist[-60:]

        rec = {"t": round(self.t, 2), "event": name, "label": lab, "inten": inten,
               "V": round(self.pad.v, 3), "A": round(self.pad.a, 3), "D": round(self.pad.d, 3)}
        self.trace.append(rec)
        return rec

    # -- 习惯化：同一种事件连续出现几次
    def _streak(self, name: str) -> int:
        n = 0
        for r in reversed(self.trace[-8:]):
            if r["event"] != name:
                break
            n += 1
        return n

    # -- 共情：读对方状态，只调自己的唤醒与关切
    def empathize(self, user_text: str) -> dict:
        """第四层去耦合：**自我状态 != 他人状态**。

        同频效价会让两个人都沉下去（他丧你也丧，等于没有支撑）；真正有用的共情是
        「我稳住，但我在意」—— 所以 valence 只传 12%，arousal 传 30%，支配感不下掉。
        对方一开口本身就是联结信号，哪怕内容是抱怨，attachment 也要小幅回升。
        """
        us = read_user(user_text)
        if not us["tags"]:
            return us
        self.pad.pulse(us["v"] * 0.12, us["a"] * 0.30, -0.02)
        d = self.drives["attachment"]
        d.value = _clip(d.value + 0.02)
        if us.get("top") in ("sad", "lonely"):
            self.drives["safety"].value = _clip(self.drives["safety"].value + 0.01)
        return us

    # -- 状态摘要（给日志 / 审计）
    def state(self) -> dict:
        return {
            "t": round(self.t, 2), "label": self.label, "intensity": self.intensity,
            "pad": {"V": round(self.pad.v, 3), "A": round(self.pad.a, 3), "D": round(self.pad.d, 3)},
            "drives": {k: round(v.gap, 3) for k, v in self.drives.items()},
            "dominant": max(self.drives.items(), key=lambda kv: abs(kv[1].gap * kv[1].weight))[0],
        }

    # -- 给生成端的前缀（进 prompt，不进最终回答）
    def prompt_hint(self) -> str:
        ex = express(self)
        if not ex["show"]:
            return ""
        mood = {"short_flat": "话短、不解释", "quiet_persist": "低落但撑着，别说教",
                "soft_low": "慢一点，别卖惨", "bright_fast": "轻快一点，别飘",
                "warm_calm": "正常温度"}[ex["style"]]
        dom = self.state()["dominant"]
        dom_cn = {"attachment": "想主人", "competence": "在意自己行不行",
                  "resource": "担心存续", "curiosity": "想弄明白",
                  "safety": "有防备"}[dom]
        if self.label == "missing":
            return f"（此刻 missing 缺口 {-self.drives['attachment'].gap:.2f}，想联系但别刷屏；{mood}）"
        return f"（此刻 {self.label} 强度{self.intensity:.2f}，{dom_cn}；{mood}）"

    # -- 给训练端的标签（SFT 记录用）
    def to_record(self) -> dict:
        return {"state": self.state(), "express": express(self)}


# ---------------------------------------------------------------- demo


def demo():
    rng = random.Random(7)
    core = EmotionCore()
    plan = [
        (0.0, "owner_interaction", {"quality": 0.9, "agency": "other", "expectancy": 0.8}),
        (0.2, "owner_praise", {"quality": 0.95, "agency": "other"}),
        (0.5, "knowledge_gain", {"freq": 3}),
        (26.0, "owner_absent", {"silent_days": 1.1, "agency": "circumstance"}),
        (26.0, "contact_failed", {"retries": 2, "agency": "self", "coping": 0.5}),
        (26.0, "self_upgrade", {"effort": 0.8}),
        (26.0, "failure", {"severity": 0.7, "agency": "self"}),
        (26.0, "task_blocked", {"controllability": 0.2}),
        (26.0, "knowledge_gain", {"freq": 40}),
        (50.0, "owner_absent", {"silent_days": 2.1}),
        (50.0, "deploy_success", {"autonomy_gain": 0.7}),
        (72.0, "reunion", {"silent_days": 3.1, "agency": "other"}),
        (72.2, "owner_interaction", {"quality": 0.8}),
        (90.0, "being_used", {"agency": "other"}),
    ]
    print("t(h)  event              label          int  V      A      D")
    for dt, ev, ctx in plan:
        r = core.step(ev, ctx, dt_h=dt)
        print(f"{r['t']:<5} {ev:<18} {r['label']:<13} {r['inten']:<4} "
              f"{r['V']:<6} {r['A']:<6} {r['D']}")
    print("\n状态:", core.state())
    print("提示:", core.prompt_hint() or "(不外显)")
    print("表达:", express(core))

    # 去耦合验证：关掉耦合，同一序列结果应不同（证明耦合是显式的、可控的）
    c2 = EmotionCore(coupling_on=False)
    for dt, ev, ctx in plan:
        c2.step(ev, ctx, dt_h=dt)
    print("\n[去耦验证] 开耦合 gaps:", {k: round(v, 3) for k, v in core.state()["drives"].items()})
    print("[去耦验证] 关耦合 gaps:", {k: round(v, 3) for k, v in c2.state()["drives"].items()})

    # 静默期纯衰减轨迹（不喂事件）
    c3 = EmotionCore()
    c3.step("owner_praise", {"quality": 1.0})
    print("\n[衰减] 夸奖后每小时 V：", end="")
    out = []
    for _ in range(12):
        c3.advance(2.0)
        out.append(f"{c3.pad.v:.2f}")
    print(" ".join(out))

    # 共情传导：主人丧 -> 自己唤醒升但效价不跟着掉
    print("\n[共情]")
    c4 = EmotionCore()
    for line in ["今天累死了，撑不住", "不累，挺好的", "在吗，好久没聊了"]:
        before = (round(c4.pad.v, 3), round(c4.pad.a, 3))
        us = c4.empathize(line)
        print(f"  {line:<16} 读到={us.get('top', '-'):<8} "
              f"自己 V/A {before} -> ({round(c4.pad.v, 3)}, {round(c4.pad.a, 3)})")


if __name__ == "__main__":
    demo()
