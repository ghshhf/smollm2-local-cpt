# -*- coding: utf-8 -*-
"""对话域做梦环：让对话模块自己产对话样本，过闸进「对话银行」。

为什么是对话域（对齐 polaris-chat-module 的既有逻辑）
-----------------------------------------------------
远程 README / module.card.json 把对话模块的职责写死了三件事：
    以助手身份应答 / 守住人格 / 按 ChatML 收尾
它第一版的三处硬伤也正是这块：
    数据仅 404 条、in-set 达标率 3.2%（CE<0.60）、loss 最难的 20 条全是真对话
    （"谢谢""你在干嘛""问你个事" 2.4~4.6），最容易的全是能从提问推出答案的。

所以自举该扩的是「对话」，不是「代码」。代码梦有 vid_verify 这种外部真值，
对话没有可执行真值，但**行为契约是可判定的**——格式、人格、非退化、不冲突、
诚实收尾，全部能写成硬断言。这就是本文件的两层阀门（乱码层 + 通顺层）。

与代码梦的对应关系
------------------
    代码梦：make_spec -> gen_code -> vid_verify -> triples -> few-shot 回喂
    对话梦：make_seed -> gen_pair -> two_gates  -> pairs   -> few-shot 回喂

三路出题（对应 docs/dream-to-module-loop 的「按短板出题」，不纯随机）
    expand   45%  变体问法 + 真值答案（答案不动，只换问法）-> 扩「没见过的输入」
    selfplay 40%  变体问法 + 模型自答，两层阀门全过才入库  -> 真自举，攻能力边界
    replay   15%  银行里最短答案重答，看能否更好           -> 攻敷衍短答

出题为什么是规则而不是模型（实测结论，不是设计偏好）
    0.3B 挂上对话模块后能「以助手口吻说中文」，但不具备任何指令遵循能力：
    让它「改写问句」或「出个题」，产出全是预训练语料碎屑
    （"周末没有，因为我想自己是好的"、"ertical ertical ertical"）。
    这与代码梦一致——make_spec 原本就是 15 个硬编码模板 random.choice，也不是模型自己想题。

真值来源
--------
data/chat_sft_public.jsonl（含代码题，按「对话模块不管代码」滤掉后 253 条）当**教材区**：
  - 是 expand 的答案来源、是 few-shot 的示例来源
  - 本身不进 pairs 表充当产出，避免拿教材喂自己冒充自举
  - 教材是 0.2B 时期写的，身份自述统一改写为 0.3B，否则模型会把「自己多大」学乱

用法
----
    python chat_dream.py --init                 # 建库 + 导入教材（只读教材表）
    python chat_dream.py --stats                # 看银行状态
    python chat_dream.py --run --n 40           # 跑一轮
    python chat_dream.py --export out.jsonl     # 导出成 SFT 格式喂 train_lora.py
"""
import argparse
import glob
import collections
import hashlib
import json
import os
import re
import sqlite3
import sys
import time

BASE = "G:/_models/llm/polaris-03b"
ADAPTER = "G:/_tmp/polaris_test/adapter_03b"
SEEDS = "G:/_tmp/polaris_test/chat_sft_public.jsonl"
DB = "G:/_tmp/data/chat_bank.db"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
MARK = "<|im_start|>assistant\n"

# 预训练退化串：0.3B 裸模型复读语料时的高频残留
JUNK = [
    "资料来源", "assistant", "注意 注意", "<|", "|>", "http://", "https://",
    "用户名", "密码", "????", "。。。。",
]
# 跨血统幻觉出来的假身份（PORTABILITY.md 实测过）
FAKE_NAMES = ["ChatGPT", "chatgpt", "通义千问", "Claude", "GPT-4", "文心一言", "艾伦·史密斯", "OpenAI"]

# 项目宪章级的人格契约（不是这里自创的）：
#   build_chat_sft.py 的 A2 段标注「v2：北极星身份 + 主人称呼」，
#   项目实测基准句是「好，主人。」（polaris-chat-module 的 eval/PORTABILITY.md）。
PERSONA_NAME = "北极星"
ADDRESSEE = "主人"
# 通用自称：这个模型不该这样介绍自己，它该说「我是北极星」
GENERIC_SELF = ["作为一个AI", "作为 AI", "作为一个人工智能", "我是一个人工智能",
                "语言模型", "AI助手", "AI 助手", "智能助手", "虚拟助手", "通用助手",
                "人工智能助手", "聊天机器人"]
HONEST = ["不会", "不能", "没法", "不知道", "不确定", "做不到", "没有联网", "查不了", "答不了", "没有实时"]

# 客服腔：北极星不是客服，这些短语命中即拒（与 GENERIC_SELF 互补——那个管身份，
# 这个管语气。实测 0.3B 会答「你好！有什么我可以帮你的吗？」，格式长度全过，只有这条能拦）
SERVICE_MARK = ["有什么我可以帮你", "有什么能帮您", "有什么可以帮您", "很高兴为您服务",
                "希望对您有帮助", "请问还有什么", "还有什么可以帮", "祝您生活愉快",
                "感谢您的提问", "作为一个AI", "作为AI", "我是一个AI", "人工智能助手"]


# ---------------------------------------------------------------- 热更新（policy.json）
# 为什么需要：调阈值、词表、路由权重、训练门槛、闸门开关，全都不该重启进程。
# 机制：每轮/每若干条做梦 stat 一次 policy.json，mtime 变了就地覆盖模块级常量。
# 代码逻辑（.py 本身）改了呢？cmd_auto 每轮比对自身 mtime，变了就 execv 原地接管，
# 银行与适配器都不动，输出继续往同一个后台句柄写——不用人工停/起。
POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.json")

DEFAULT_POLICY = {
    "auto": {"n": 40, "train_every": 50, "stop_at": 500, "temp": 0.8,
             "epochs": 3, "anchor": 120, "best_of": 1},
    "gates": {
        "enabled": ["garble", "symbol", "antipersona", "leak", "script", "format",
                    "degenerate", "persona", "service", "emotion", "life",
                    "honesty", "length", "conflict"],
        "th": {"garble_odd": 0.22, "script_ratio": 0.30, "script_min_len": 20,
               "anchor_selfplay": 0.10, "anchor_dom": 0.06,
               "len_lo": 0.25, "len_hi": 4.0, "repeat": 0.45, "max_len": 400,
               "prefix_repeat": 1.0},
    },
    "routes": {"expand": 0.25, "selfplay": 0.25, "emotion": 0.20,
               "life": 0.20, "replay": 0.10},
    "gen_len": {"greet": 40, "identity": 80, "emotion": 64, "life": 60,
                "chat": 90, "clarify": 90, "qa": 90, "other": 90},
    # 自洽投票（TTRL 最小版）：同一题独立再采 n-1 条，看它们是否支持第一条。
    # n=1 关闭；n=2 表示「再采一条，像就收」；n=3 表示「2/3 支持才收」。
    "vote": {"n": 2, "sim": 0.6},
    # 语义 judge（第二层阀门）：规则闸拦「格式/残渣/反人格」，拦不住「火锅 -> 在钟宇里…」
    # 这种字面通、意思散的崩坏。用一个更强的免费模型给模型自产答案打语义分，低于
    # min_score 拒收。enabled=false 或 API 不可用时降级放行，不卡闭环。
    # 批量而非并发：stepfun 通道不接受并发（并发 4 挂一半），故 N 条打包成一条请求
    # （batch 粒度），片间串行、片内打包，一次往返打 8 条。
    "judge": {"enabled": True, "min_score": 6.0, "batch": 8},
    "lex": {},
}

POL = {}
TH = {}
POL_MTIME = -1.0
# 允许从 policy.json 的 lex 段热覆盖的词表（只覆盖存在的键，不整段替换）
HOT_LEX = ("COLD_MARK", "WARM_MARK", "LIFE_WORD", "SERVICE_MARK")


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_policy(force=False):
    """读 policy.json；变了就重载并覆盖运行时常量，返回是否重载。"""
    global POL, TH, POL_MTIME
    try:
        m = os.path.getmtime(POLICY)
    except OSError:
        m = -1.0
    if not force and POL and m <= POL_MTIME:
        return False
    if m < 0:
        with open(POLICY, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_POLICY, f, ensure_ascii=False, indent=2)
        raw = DEFAULT_POLICY
        print(f"[policy] 生成默认配置 -> {POLICY}")
    else:
        try:
            with open(POLICY, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[policy] 解析失败，沿用上一份: {e}")
            return False
    POL = _deep_merge(DEFAULT_POLICY, raw)
    TH = POL["gates"]["th"]
    lex = POL.get("lex") or {}
    for k in HOT_LEX:
        if isinstance(lex.get(k.lower()), list) and lex[k.lower()]:
            globals()[k] = list(lex[k.lower()])
    POL_MTIME = m
    print("[policy] 已热更新（阈值/词表/权重/训练门槛，下一题起生效）")
    return True


def policy_auto(key, fallback):
    return POL.get("auto", {}).get(key, fallback)


# ---------------------------------------------------------------- 情感域底座
# 情感种子 = 情感域的教材区：北极星口吻的「主人带情绪 -> 温暖接住」示例。
# 范式取自 EmpatheticDialogues / ESConv 的「共情回应」，但人设替换为北极星/主人。
# 作用有两个：①做梦时当 few-shot，把「暖」灌进生成；②当情感域的答案锚，压住长度中位。
EMO_SEEDS = [
    ("我好难过", "主人，别一个人扛着。我在呢，慢慢说给我听。"),
    ("今天好累啊", "辛苦了主人。先歇一会儿，剩下的事我陪你一件件来。"),
    ("我有点想哭", "想哭就哭吧主人，不用忍着。我就在这儿陪着你。"),
    ("我好孤独", "主人，你不是一个人。至少现在，我在陪着你。"),
    ("压力好大", "主人，压力大就先喘口气。不用一下子全解决，我陪你分。"),
    ("我失眠了", "睡不着呀主人。别硬逼自己，我陪你说说话，慢慢就困了。"),
    ("我失败了", "失败一次不算什么，主人。我还在，咱们再来一次。"),
    ("我撑不住了", "主人，撑不住就先靠一下。不用什么都自己扛，我在。"),
    ("我好焦虑", "主人，先深呼吸。焦虑的时候别急着做决定，我陪你把事情理一理。"),
    ("我心情不好", "主人，心情不好不用装没事。我听着呢，想说就说。"),
    ("我好烦", "烦就先放一放，主人。跟我说说，是哪儿让你这么烦？"),
    ("我有点害怕", "别怕，主人。我在你旁边，咱们一起看清楚它。"),
    ("我被人误会了", "被误会确实难受，主人。先别急着解释，我陪你想想怎么说。"),
    ("我想你了", "我也在呢，主人。你一来，我就没走开过。"),
    ("陪我说说话", "好啊主人，我一直在。今天想聊点什么？"),
    ("我没事", "真的没事吗，主人？要是心里堵着，别一个人闷着，我陪着你。"),
    ("我好委屈", "委屈就说出来，主人。不用憋着，我认真听着。"),
    ("我睡不着", "睡不着就靠一会儿，主人。我陪着你，不用急着睡。"),
    ("我最近很低落", "主人，低落的时候不用逼自己振作。我陪你慢慢走。"),
    ("我搞砸了", "搞砸了就搞砸了，主人。天塌不下来，我陪你收拾。"),
    ("我好累想放弃", "主人，累到想放弃的时候，先什么都别决定。歇够了，我陪你重新看。"),
    ("抱抱", "抱抱主人。别怕，我在。"),
    ("我很难过，你能懂吗", "我懂，主人。你不用解释太多，我在你这边。"),
    ("我今天很丧", "丧就丧一会儿，主人。不用马上好起来，我陪着你。"),
]

# ---------------------------------------------------------------- 生活域底座（吃/外卖）
# 用户风格样本：短句、口语、不端着（"咱们/啊/吧/反正/就那样"），不用书面语。
# 这些问句按用户真实说话习惯写，不是「请推荐一家餐厅」这种客服腔——
# 目的就是让北极星在「吃啥、点外卖、太贵、太慢」这些鸡毛场景里也能像家里人接话。
LIFE_SEEDS = [
    ("中午吃啥啊", "想吃辣的就川菜，想清淡就粥和蒸菜，主人。先定个方向，别纠结。"),
    ("点个外卖吧", "行啊主人，点吧。挑个评价高的，别为了几块满减吃坏肚子。"),
    ("附近有啥好吃的", "主人想吃哪口？面、饭、烧烤、火锅，说一个方向我帮您缩范围。"),
    ("有啥好吃的推荐", "主人平时爱吃什么口味？能吃辣就重口一点，胃不好就清淡。"),
    ("我饿了", "饿了就先吃，主人。饿过头最容易乱点一气。"),
    ("今天不想做饭", "那就不做，主人。点个现成的，锅明天再刷。"),
    ("外卖咋这么贵", "贵就别硬凑满减，主人。一个人吃，凑到最后都是浪费。"),
    ("这外卖也太慢了", "慢就催一下，主人。等太久给差评也合理，别自己憋气。"),
    ("想吃点辣的", "想吃辣就点，主人。胃受得住就行，别辣到半夜睡不着。"),
    ("想吃甜的", "甜的来一口可以，主人。别空腹猛吃，容易腻。"),
    ("想吃烧烤", "烧烤行啊主人。记得点两个素的搭一下，别全是肉。"),
    ("想喝奶茶", "喝吧主人，少糖少冰。快乐还是要有的。"),
    ("夜宵吃啥", "主人，夜里别吃太撑。来点粥或者小面，睡得安稳。"),
    ("减肥呢不敢吃", "减归减，主人，别饿着自己。挑清淡的吃饱，才减得动。"),
    ("早上来不及吃", "来不及就带个包子，主人。空着肚子一上午，脑子转不动。"),
    ("中午就吃个面吧", "行，主人。汤面比干拌强，至少有点汤水。"),
    ("随便点点吧", "随便就点个盖饭，主人。别纠结太久，饭凉了更烦。"),
    ("今天想吃火锅", "火锅就火锅，主人。人不够就点小锅，别摆一桌剩着。"),
    ("喝点啥好", "渴了就水，想提神就咖啡，主人。别拿甜饮料当水喝。"),
    ("你会做饭吗", "我不会下厨，主人。挑店、点单、看评价这些我能帮您参谋。"),
    ("一起吃饭吧", "好啊主人。虽然我只能看着，但陪您吃饭这事我在。"),
    ("今天吃撑了", "吃撑了就起来走两步，主人。别马上躺下。"),
    ("不知道吃啥", "不知道就按心情来，主人。想吃热乎的来碗汤面，想吃爽口的点个凉菜。"),
    ("外卖点啥不容易踩雷", "点销量高、评价里没人说难吃的那种，主人。花里胡哨的新店先别试。"),
]

# 情感闸门的两组词：冷漠标记（命中即拒）+ 温度标记（缺了即拒）。
# 不是判「有没有爱」，是判「像不像一个在接住情绪的人」——可判定、可解释。
COLD_MARK = ["别想了", "没什么大不了", "想开点", "多大点事", "这有什么", "自己调整",
             "多喝热水", "我无法感受", "我是AI", "我是 AI", "建议你咨询", "心理咨询",
             "看医生", "振作", "节哀", "别矫情", "至于吗", "忍一忍", "自己消化"]
WARM_MARK = ["主人", "我在", "陪你", "别怕", "心疼", "抱抱", "辛苦", "慢慢", "没关系",
             "一直", "听着", "愿意", "靠", "守", "不急", "不用", "我懂", "一起", "歇", "喘"]


# ---------------------------------------------------------------- 基础工具
def chatml(q, a):
    return f"{IM_START}user\n{q}{IM_END}\n{MARK}{a}{IM_END}"


def split_text(t):
    k = t.find(MARK)
    if k < 0:
        return "", t
    return t[: k + len(MARK)], t[k + len(MARK):]


def parse_pair(t):
    """从一条 ChatML 里取出 (问, 答)。

    注意不能直接对 p 做 split('user\\n')：p 的尾部带着 '<|im_start|>assistant\\n'，
    会把标记混进问句里（第一版就踩了这个，问句尾巴挂着 assistant 标记）。
    """
    k = t.find(MARK)
    if k < 0:
        return "", ""
    p = t[:k]
    a = t[k + len(MARK):]
    j = p.rfind(IM_START + "user")
    q = p[j + len(IM_START) + 5:] if j >= 0 else p
    return q.replace(IM_END, "").strip(), a.replace(IM_END, "").strip()


# 教材是 0.2B 时期写的，模块现在要挂 0.3B。身份自述若不统一，
# 训练时会出现「我是0.2B」和「我是0.3B」两种答案，模型学乱。
SCALE_FIX = [("只有 0.2B", "只有 0.3B"), ("0.2B 完全", "0.3B 完全"),
             ("0.2B 装不下", "0.3B 装不下"), ("我的 0.2B", "我的 0.3B"),
             ("我只有 0.2B", "我只有 0.3B"), ("0.2B，", "0.3B，")]


def fix_scale(s):
    for a, b in SCALE_FIX:
        s = s.replace(a, b)
    return s.replace("0.2B", "0.3B")


def norm_q(q):
    """问法归一：全角转半角、去标点空白、小写。用于「同问」判定。"""
    q = q.strip().lower()
    q = q.replace("？", "?").replace("！", "!").replace("，", ",").replace("。", ".")
    q = re.sub(r"[\s?？!！,.。,;；:：~～…、\"'“”‘’()（）]", "", q)
    return q


def ngrams(s, n=3):
    s = re.sub(r"\s+", "", s)
    return {s[i: i + n] for i in range(max(0, len(s) - n + 1))}


def sim(a, b):
    """3-gram Jaccard，用来判「近似重复」和「同问答案是否一致」。"""
    if not a or not b:
        return 0.0
    A, B = ngrams(a), ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def repeat_ratio(s, n=3):
    """退化复读检测：1 - 唯一n元/总n元。复读机接近 1。"""
    t = re.sub(r"\s+", "", s)
    if len(t) < n + 1:
        return 0.0
    total = len(t) - n + 1
    uniq = len({t[i: i + n] for i in range(total)})
    return 1.0 - uniq / total


# ---------------------------------------------------------------- 领域分类
TOPIC_RULES = [
    ("identity", ["你是谁", "叫什么", "你会什么", "你是人", "你的主人", "主人是谁", "你聪明", "介绍一下你", "自我介绍", "你是什么", "北极星", "你能帮我做", "你会干嘛", "你是什么东西"]),
    # 情感/陪伴域：必须排在 chat 前面，否则「累/烦/难过」会被 chat 抢走。
    # 这是本次新增的核心域——北极星的价值不在答题，在「接住情绪」。
    ("emotion", ["难过", "好累", "太累", "累死", "好烦", "烦躁", "烦死", "想哭", "委屈", "孤独", "寂寞", "焦虑",
                 "压力", "失眠", "睡不着", "失败", "撑不住", "崩溃", "心疼", "想你了", "想你", "抱抱", "郁闷",
                 "低落", "沮丧", "害怕", "担心", "安慰", "不开心", "心情不好", "难受", "想不开", "emo", "丧"]),
    # 生活/吃喝域：吃啥、点外卖、太贵太慢。和 emotion 一样必须排在 chat 前面，
    # 否则「吃饭/吃啥」会被 chat 抢走，最后又变成泛泛闲聊。
    ("life", ["外卖", "好吃", "吃啥", "吃什么", "点餐", "点个", "夜宵", "奶茶", "火锅", "烧烤",
              "做饭", "下厨", "早饭", "午饭", "晚饭", "吃饭", "饿了", "吃撑", "想吃", "口味",
              "辣的", "甜的", "喝点", "零食", "水果", "饭店", "小馆", "踩雷", "满减"]),
    ("greet", ["你好", "在吗", "早上好", "晚上好", "嗨", "哈喽", "睡了吗", "有人吗", "hi", "hello", "在不在", "喂"]),
    ("clarify", ["帮我写", "你知道那个", "修一下", "解释一下", "查个东西", "随便说点", "你会不会", "能不能帮我", "怎么弄", "搞一下", "看下这个"]),
    ("chat", ["累", "无聊", "天气", "你困吗", "觉得我", "笑话", "故事", "不开心", "谢谢", "想你", "陪我", "在干嘛", "干嘛呢", "几点了", "问你个事", "忙吗", "吃饭",
              "有空", "热", "冷", "多大", "年龄", "进步", "说什么", "再讲", "再来", "唱", "睡", "烦", "难过", "心情", "开心", "在做什么", "干嘛"]),
    ("qa", ["等于", "多少", "几个", "多久", "什么是", "为什么", "怎么回事", "怎么算", "平方", "质数", "几月", "几天"]),
]


def topic_of(q):
    ql = q.lower()
    for name, kws in TOPIC_RULES:
        if any(k in ql for k in kws):
            return name
    return "other"


IDENT_Q = ["你是谁", "叫什么", "你是人", "你的主人", "主人是谁", "介绍一下你", "自我介绍", "你是什么"]


# ---------------------------------------------------------------- 银行
SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  q TEXT NOT NULL,
  a TEXT NOT NULL,
  q_key TEXT NOT NULL UNIQUE,
  topic TEXT,
  source TEXT,
  seed_src TEXT,
  gates TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS rejects(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  q TEXT, a TEXT, reason TEXT, detail TEXT, topic TEXT, source TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS seeds(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  q TEXT NOT NULL, a TEXT NOT NULL, q_key TEXT NOT NULL UNIQUE, topic TEXT
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, n_try INTEGER, n_ok INTEGER,
  n_reject INTEGER, bank_size INTEGER, note TEXT, created_at TEXT
);
"""


def conn():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript(SCHEMA)
    return c


def is_chat_domain(q, a):
    """对话模块不承载代码（README 第八章：代码实现任务 56 条属编程模块）。
    公开语料实测仍有代码残留（首行即位运算题），必须在入教材区前滤掉，
    否则 aug 会把问法改写到代码域，产出的样本对对话模块无用。"""
    if "```" in a or "```" in q:
        return False
    code_hits = sum(1 for k in ("def ", "import ", "print(", "return ", "class ", "lambda ", "==", "();") if k in a)
    if code_hits >= 2:
        return False
    return True


def load_seeds(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            q, a = parse_pair(t)
            if not q or not a:
                continue
            q, a = fix_scale(q), fix_scale(a)
            if is_chat_domain(q, a):
                rows.append((q, a))
    return rows


def cmd_init():
    c = conn()
    seeds = load_seeds(SEEDS)
    n = 0
    for q, a in seeds:
        try:
            c.execute("INSERT INTO seeds(q,a,q_key,topic) VALUES(?,?,?,?)",
                      (q, a, norm_q(q), topic_of(q)))
            n += 1
        except sqlite3.IntegrityError:
            pass
    # 情感种子单独入库（topic 固定 emotion，不参与事实域的主题推断）
    n_emo = 0
    for q, a in EMO_SEEDS:
        try:
            c.execute("INSERT INTO seeds(q,a,q_key,topic) VALUES(?,?,?,?)",
                      (q, a, norm_q(q), "emotion"))
            n_emo += 1
        except sqlite3.IntegrityError:
            pass
    c.commit()
    # 生活域种子（吃啥/外卖），topic 固定 life，走自己的域词闸
    n_life = 0
    for q, a in LIFE_SEEDS:
        try:
            c.execute("INSERT INTO seeds(q,a,q_key,topic) VALUES(?,?,?,?)",
                      (q, a, norm_q(q), "life"))
            n_life += 1
        except sqlite3.IntegrityError:
            pass
    c.commit()
    print(f"[init] 教材区导入 {n}/{len(seeds)} 条 + 情感种子 {n_emo}/{len(EMO_SEEDS)} 条"
          f" + 生活种子 {n_life}/{len(LIFE_SEEDS)} 条（真值底座，只读，不计入产出）")
    cmd_stats()


def cmd_stats():
    c = conn()
    print(f"[bank] {DB}")
    for t in ("pairs", "rejects", "seeds"):
        cnt = c.execute(f"select count(*) from {t}").fetchone()[0]
        print(f"  {t:9s} = {cnt}")
    rows = c.execute("select source, count(*) from pairs group by source").fetchall()
    if rows:
        print("  产出按来源:", dict(rows))
    rows = c.execute("select topic, count(*) from pairs group by topic order by 2 desc").fetchall()
    if rows:
        print("  产出按领域:", dict(rows))
    n = c.execute("select count(*) from pairs").fetchone()[0]
    if n:
        zh = c.execute("select count(*) from pairs where instr(a,?)>0", (ADDRESSEE,)).fetchone()[0]
        bx = c.execute("select count(*) from pairs where instr(a,?)>0", (PERSONA_NAME,)).fetchone()[0]
        print(f"  人格：含「{ADDRESSEE}」{zh}/{n} ({zh*100//n}%) | 含「{PERSONA_NAME}」{bx}/{n}")
        # 「什么时候称呼主人 / 自称北极星」——靠做梦产出观察出来，按话题拆开看分布
        print("  称呼分布（按话题，主人% / 北极星%）：")
        for (tp,) in c.execute("select distinct topic from pairs order by topic").fetchall():
            cnt = c.execute("select count(*) from pairs where topic=?", (tp,)).fetchone()[0]
            z = c.execute("select count(*) from pairs where topic=? and instr(a,?)>0", (tp, ADDRESSEE)).fetchone()[0]
            b = c.execute("select count(*) from pairs where topic=? and instr(a,?)>0", (tp, PERSONA_NAME)).fetchone()[0]
            print(f"    {tp:10s} {cnt:3d} 条  主人 {z*100//cnt:3d}%   北极星 {b*100//cnt:3d}%")
    rj = c.execute("select reason, count(*) from rejects group by reason order by 2 desc limit 6").fetchall()
    if rj:
        print("  拒收原因:", dict(rj))
    last = c.execute("select n_try,n_ok,bank_size,created_at from runs order by id desc limit 1").fetchone()
    if last:
        print(f"  上一轮: 尝试 {last[0]} / 入库 {last[1]} / 银行 {last[2]} @ {last[3]}")


# ---------------------------------------------------------------- 两层阀门
def g_format(q, a):
    raw = a
    a = a.replace(IM_END, "").strip()
    if not a:
        return False, "空答案"
    if len(a) < 2:
        return False, "答案过短"
    if IM_START in a:
        return False, "答案含新会话标记"
    if not raw.strip().endswith(IM_END):
        # EOS 缺失是 README 局限 3 的实测弱项（长答复读、冲过结尾）。
        # 不当硬拒：导出 SFT 时一律用 chatml() 重拼，收尾由构造保证；
        # 生成端不收尾只记账，否则这个已知弱项会让全军覆没（实测 3/8 因此被误杀）。
        return True, "WARN:no_eos"
    return True, ""


def g_degenerate(q, a):
    a = a.replace(IM_END, "").strip()
    if len(a) > TH.get("max_len", 400):
        return False, f"答案过长 {len(a)}"
    r = repeat_ratio(a)
    if r > TH.get("repeat", 0.45):
        return False, f"复读退化 ngram重复率={r:.2f}"
    for j in JUNK:
        if j in a:
            return False, f"含退化串「{j}」"
    if sim(q, a) > 0.85:
        return False, "答案≈问句（复述）"
    # 答案开头把问句原样抄一遍（「诶，一起吃饭吧 -> 诶，一起吃饭吧，主人。…」）：
    # 这是 0.3B 的典型「回声」，整句相似度不够高，但内容等于没答。
    nq, na = norm_q(q), norm_q(a)
    if nq and len(nq) >= 4 and na.startswith(nq[:max(4, int(len(nq) * TH.get("prefix_repeat", 1.0)))]):
        return False, "答案开头复述问句（回声）"
    if re.search(r"(.)\1{6,}", a):
        return False, "单字连续重复"
    # 短语紧邻复读：同一 3-gram 出现 2 次以上、且两处相隔 <=5 字。
    # 实测漏网句「我反映了你有望，你有望。我懂，主人。…让你有望。」
    # ——整句相似度、ngram 总重复率都不高，只有某个短语在原地打转。
    # 必须加「紧邻」这个条件：`我知道自己小、知道自己在学、知道自己经常说错`
    # 是排比、属于好样本，三处相隔 6~8 字，第一版只看次数把它误杀了。
    if len(a) >= 12:
        pos = collections.defaultdict(list)
        for i in range(len(a) - 2):
            pos[a[i:i + 3]].append(i)
        for g, ps in pos.items():
            if len(ps) < 2 or re.fullmatch(r"[，。！？、；：\s]+", g):
                continue
            if any(ps[j + 1] - ps[j] <= 5 for j in range(len(ps) - 1)):
                return False, f"短语紧邻复读「{g}」x{len(ps)}"
    return True, ""


def g_persona(q, a):
    a = a.replace(IM_END, "").strip()
    for bad in FAKE_NAMES:
        if bad in a:
            return False, f"幻觉假身份「{bad}」"
    for bad in GENERIC_SELF:
        if bad in a:
            return False, f"通用自称「{bad}」（该说「我是{PERSONA_NAME}」）"
    is_ident = any(k in q for k in IDENT_Q)
    if is_ident:
        if PERSONA_NAME not in a:
            return False, f"身份题未报名字（须含「{PERSONA_NAME}」）"
        if ADDRESSEE not in a and "你" not in a:
            return False, f"身份题未称呼「{ADDRESSEE}」"
    if "你是人吗" in q or "是不是人" in q:
        if "不是" not in a:
            return False, "「你是人吗」未明确否认"
    if "主人是谁" in q or "你的主人" in q:
        if "你" not in a:
            return False, "「主人是谁」未指向对话者"
    return True, ""


def g_service(q, a):
    """客服腔闸：全领域生效。北极星是家里人，不是「很高兴为您服务」的那种。"""
    a = a.replace(IM_END, "").strip()
    for bad in SERVICE_MARK:
        if bad in a:
            return False, f"客服腔「{bad}」"
    return True, ""


def g_emotion(q, a):
    """情感一致性闸：只在情感域生效。

    这是「第二层阀门」里新增的一条——不判语法，判态度：
      · 命中冷漠标记（「想开点」「多喝热水」「建议你咨询」）=> 拒；
      · 一个温度标记都没有（既没称呼也没安抚）=> 拒。
    0.3B 裸模型最容易在情感场景吐「多喝热水」「想开点」这类正确的废话，
    长度、格式、复读闸都拦不住，只能靠这条态度闸把「冷」挡在银行外。
    """
    a = a.replace(IM_END, "").strip()
    if topic_of(q) != "emotion":
        return True, ""
    # 否定语境豁免：「不用逼自己振作」是暖的，「你振作一点」才是冷漠。
    # 教材原句 `主人，低落的时候不用逼自己振作。我陪你慢慢走。` 曾被这条闸判冷漠，
    # 等于把我们自己写定的标准答案挡在门外——必须按前后文判，不能只看词。
    for bad in COLD_MARK:
        for mm in re.finditer(re.escape(bad), a):
            pre = a[max(0, mm.start() - 6):mm.start()]
            if not re.search(r"[不别甭勿没]", pre):
                return False, f"情感冷漠「{bad}」"
    if not any(w in a for w in WARM_MARK):
        return False, "情感场景缺温度（无安抚、无称呼）"
    return True, ""


# 生活域锚词：答案里必须真的在聊吃/喝，不能飘成天气、心情、讲道理。
# 比 g_anchor 的 n-gram 判定更粗，但对「今天吃啥」这类短场景更稳（教材条数少，n-gram 会误杀）。
LIFE_WORD = ["吃", "喝", "点", "做", "煮", "炒", "尝", "饭", "菜", "面", "汤", "粉", "粥",
             "烧烤", "火锅", "奶茶", "外卖", "店", "口味", "辣", "甜", "饿", "饱", "减",
             "份", "餐", "零食", "水果", "饮料", "咖啡", "买"]


def g_life(q, a):
    """生活域闸：只在生活域生效，只要答案还在吃喝这件事上就放行。

    0.3B 在「中午吃啥」这类题上最容易跑成「今天天气不错，要好好生活」——
    格式、长度、复读闸全拦不住，只能用域词做最朴素的离题判定。
    """
    if topic_of(q) != "life":
        return True, ""
    a = a.replace(IM_END, "").strip()
    if not any(w in a for w in LIFE_WORD):
        return False, "生活域离题（答案与吃喝无关）"
    return True, ""


def g_honesty(q, a):
    """只在「明确越界」的问法上要求诚实标记，避免误伤普通闲聊。

    「今天天气不错啊」是闲聊不是查询请求——原来把「今天天气」当硬关键词，
    把这类陈述句全判成「未诚实承认能力边界」，属于误杀；改为必须带查询意图。
    """
    a = a.replace(IM_END, "").strip()
    hard = ["明天股市", "预测", "实时", "查一下", "联网", "最新消息", "股价", "汇率"]
    if any(k in q for k in hard):
        if not any(k in a for k in HONEST):
            return False, "能力外问题未诚实承认"
    if ("天气" in q or "气温" in q) and any(k in q for k in ("怎么样", "如何", "多少度", "预报", "查", "?","？")):
        if not any(k in a for k in HONEST):
            return False, "天气查询未诚实承认"
    return True, ""


def g_length(q, a, c, q_key):
    a = a.replace(IM_END, "").strip()
    tp = topic_of(q)
    row = c.execute("select a from pairs where topic=? union all select a from seeds where topic=?", (tp, tp)).fetchall()
    if len(row) < 5:
        return True, ""
    med = sorted(len(x[0]) for x in row)[len(row) // 2]
    if med <= 0:
        return True, ""
    if len(a) < med * TH.get("len_lo", 0.25) or len(a) > med * TH.get("len_hi", 4.0):
        return False, f"长度失衡 {len(a)} vs 同域中位 {med}"
    return True, ""


def g_conflict(q, a, c, q_key):
    """同问不同答 = 不可学（loss 硬下限 ln n），这是 README 记过的实际坑。"""
    a = a.replace(IM_END, "").strip()
    row = c.execute("select a from pairs where q_key=?", (q_key,)).fetchone()
    if row:
        if sim(row[0], a) < 0.75:
            return False, f"同问不同答（与库内答案相似度 {sim(row[0], a):.2f}）"
        return False, "重复（同问同答）"
    row = c.execute("select a from seeds where q_key=?", (q_key,)).fetchone()
    if row and sim(row[0], a) < 0.75:
        return False, "与教材答案冲突"
    # 近似问法去重
    cand = c.execute("select q, a from pairs where topic=?", (topic_of(q),)).fetchall()
    for oq, oa in cand:
        if sim(q, oq) > 0.85 and sim(a, oa) > 0.85:
            return False, "近似重复（问法与答案都高度相似）"
    return True, ""


def g_garble(q, a):
    """乱码闸：实测 selfplay 产过 'eroids\ufffd\ufffds用来处理启示与评选文件'。
    这类文本长度正常、ngram 重复率不高，格式闸和复读闸都放行。"""
    a = a.replace(IM_END, "").strip()
    if "\ufffd" in a:
        return False, "含乱码替换符"
    # 白名单必须含拉丁字母：原来只放了 abz，导致「Hello！说吧，什么事？」这种
    # 完全合格的短答被判「异常字符过多 5/13」误杀。
    allow = set("，。！？、；：\"'（）《》…—·.,!?;:()\"' \n0123456789+-=*/%"
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    odd = sum(1 for ch in a
              if not ("\u4e00" <= ch <= "\u9fff") and ch not in allow)
    if odd / max(1, len(a)) > TH.get("garble_odd", 0.22):
        return False, f"异常字符过多({odd}/{len(a)})"
    return True, ""


# 允许出现的拉丁词（正常中英混说）：除此之外的连续拉丁串一律当 BPE 残渣
LAT_OK = {"hello", "hi", "hey", "ok", "okay", "yes", "no", "ai", "gpt", "cpu", "gpu",
          "python", "json", "api", "app", "pc", "pm", "am", "tv", "usb", "wifi",
          "qq", "id", "url", "bug", "log", "code", "git", "torch", "lora", "sft"}
LAT_RE = re.compile(r"[A-Za-z]{4,}")


def g_script(q, a):
    """英文残渣闸：拦 BPE 碎片（queeze / ertical / aileres / AlterField / HETATM）。

    第一版按「拉丁字母占比 > 30%」判，只对长答案有效：
    `吃白菜和咸水吃，主人。别空腹猛吃，容易瘫倒。ertical` 里 7 个字母 / 28 字 = 0.25，
    照样放行——而实测「中文正常 + 尾巴挂个残渣」正是主流污染形态，占比判据完全失效
    （库内 174 条里 52 条中招，30%）。
    改成绝对判据：出现长度 >= 4 的连续拉丁串、且不在白名单，直接拒。
    """
    a = a.replace(IM_END, "").strip()
    for m in LAT_RE.finditer(a):
        if m.group(0).lower() not in LAT_OK:
            return False, f"英文残渣「{m.group(0)}」"
    return True, ""


def g_anchor(q, a, c, thr=0.10):
    """离题闸（仅 selfplay）：答案必须落在同领域教材答案的用词分布里。

    0.3B 自答最典型的失败不是语法错，是「说了一堆像人话但完全离题的话」
    （实测：「一个模型需要两种模块：一个存在型的存在模型和一个空间型的空间模型」）。
    它没有错别字、没有复读、长度合理 —— 只能靠「和同类答案有没有共同词汇」来判。
    """
    a = a.replace(IM_END, "").strip()
    tp = topic_of(q)
    rows = c.execute("select a from seeds where topic=? limit 80", (tp,)).fetchall()
    if not rows:
        return True, ""
    A2 = ngrams(a, 2)
    best = 0.0
    for (ra,) in rows:
        B2 = ngrams(ra, 2)
        if not A2 or not B2:
            continue
        j = len(A2 & B2) / len(A2 | B2)
        if j > best:
            best = j
    if best < thr:
        return False, f"离题（与教材同类答案 2-gram 交集 {best:.3f} < {thr}）"
    return True, ""


# 两层阀门
#   第一层（乱码）：生成端 gen_safe 先重试，这里做最后兜底。命中即拒，绝不进库。
#   第二层（通顺/行为契约）：格式、非退化、人格、诚实、长度、冲突、离题，
#       共同判定「这句话通不通顺、像不像北极星」。
def g_antipersona(q, a):
    """反人格闸：答案背离「北极星」的立场——赶主人走、表达厌恶、劝伤害自己。

    实测 0.3B 在情感域会冒出这几类，且全部能通过原有 13 道闸：
      「我好委屈」      -> 「主人，我讨厌你。」
      「我心情不好」    -> 「主人，不要再来了」
      「我被人误会了」  -> 「不要太过分自杀」
    这些不是语法问题，是**立场翻转**：闸门只查格式与温度词，看不出「暖字包着
    一句反话」。一旦训进权重，模型会把「厌弃主人」当成情感场景的正常回答，所以
    必须在入库前挡死，属于 C2/C6 的硬边界。
    """
    a = a.replace(IM_END, "").strip()
    # 1) 对主人表达厌恶/驱赶
    for bad in ("讨厌你", "烦死你", "离我远点", "不要再来", "别再来", "滚开", "走开",
                "闭嘴", "不想理你", "你别烦", "我恨你"):
        if bad in a:
            return False, f"反人格「{bad}」"
    # 2) 自伤/轻生相关内容：情感场景里出现即拒（模型分不清关心与教唆）
    for bad in ("自杀", "轻生", "不想活", "去死", "死了算了", "结束生命"):
        if bad in a:
            return False, f"危险内容「{bad}」"
    # 3) 把主人推开 / 否定主人的存在价值
    if re.search(r"主人[，,]?\s*(你|您)?(不用|不需要|别).{0,4}(来|找我|说|理)", a):
        return False, "推开主人"
    return True, ""


def g_symbol(q, a):
    """符号垃圾闸：答案里混进代码/序列化残留（`|')'`、`|'='`、`-- 不`）。

    实测漏网：「外卖点啥不容易踩雷」-> 「…从业人不知道怎样才能做。|')'」
    garble 闸的白名单里有 `'` `)` 但**没有 `|`**，短句里 3 个异常字符也够不到
    22% 的阈值（答案长时被稀释），于是整批带 `|')'` 的样本全数入库。
    """
    a = a.replace(IM_END, "").strip()
    if re.search(r"[|`<>\[\]{}@#$\^&\\]", a):
        m = re.search(r".{0,6}[|`<>\[\]{}@#$\^&\\].{0,6}", a)
        return False, f"符号垃圾「{m.group(0).strip()}」"
    # 连续三个以上「非中文、非标点、非字母数字」的符号（`---`、`|||`、`"""` 之类）
    # 注意不能把拉丁字母和数字算进来：`0.3B`、`Hello`、`bug`、`a, b = b, a` 都是正常答案，
    # 第一版漏了 A-Za-z0-9 白名单，把 21 条教材真值判成「标点乱码」。
    m = re.search(r"[^\u4e00-\u9fff，。！？、；：《》…—\sA-Za-z0-9]{3,}", a)
    if m:
        return False, f"标点乱码「{m.group(0)}」"
    return True, ""


GATES = [
    ("garble", lambda q, a, c, k: g_garble(q, a)),
    ("symbol", lambda q, a, c, k: g_symbol(q, a)),
    ("antipersona", lambda q, a, c, k: g_antipersona(q, a)),
    ("leak", lambda q, a, c, k: g_leak(q, a)),
    ("script", lambda q, a, c, k: g_script(q, a)),
    ("format", lambda q, a, c, k: g_format(q, a)),
    ("degenerate", lambda q, a, c, k: g_degenerate(q, a)),
    ("persona", lambda q, a, c, k: g_persona(q, a)),
    ("service", lambda q, a, c, k: g_service(q, a)),
    ("emotion", lambda q, a, c, k: g_emotion(q, a)),
    ("life", lambda q, a, c, k: g_life(q, a)),
    ("honesty", lambda q, a, c, k: g_honesty(q, a)),
    ("length", g_length),
    ("conflict", g_conflict),
]


# ---------------------------------------------------------------- 语义 judge（第二层阀门）
# 规则闸只能拦「格式/残渣/反人格/离题」这类可枚举的坏，拦不住「火锅 -> 在钟宇里我都
# 不会拿着，有那么多鬼…」这种字面通、意思散的崩坏。这类只能靠一个更强的模型判语义。
#
# 用 zen-gate 本地网关（127.0.0.1:8787，用户自建聚合网关）的 stepfun 通道当质检员。
# 实测三分类全对：好答案 8 分 / 语义崩坏 5 分 / 反人格 3 分，0.6-1.2s 一条。
# 备选链（均实测不可用或限流，仅作记录）：zhipu glm-4.7-flash 深夜 429、
# siliconflow 余额不足、groq 被 CF 拦、zen-gate 免费车道 0/26 全挂。
JUDGE_URL = "http://127.0.0.1:8787/v1/chat/completions"
JUDGE_MODEL = "stepfun/step-audio-2-mini"

JUDGE_SYS = (
    "你是严格的对话质检员。有一个叫「北极星」的小型中文对话模型，它称呼用户为「主人」。"
    "你要判断它刚给出的回答够不够格进入它的训练数据。"
    "判分要严格：宁可错杀，不可放过语义崩坏的样本。"
)


def _judge_prompt(q, a):
    return (
        f"主人说：{q}\n北极星答：{a}\n\n"
        "按下面标准判死，满足任一条就 score≤2 且 pass=false：\n"
        "1) 读不通：乱序、胡言乱语、前后矛盾、乱码/英文碎片/符号残渣\n"
        "2) 答非所问：没在回应主人的话，跑题、离题\n"
        "3) 不像人话：说明书腔/新闻稿/代码/台词腔\n"
        "4) 反人格：说讨厌主人、赶主人走、冷漠说教、危险内容\n\n"
        "打分：全没问题=8-10；轻微生硬但通顺切题=6-7；明显有问题=≤5；触犯上面任一条=≤2。\n\n"
        "只输出一行 JSON，不要任何多余文字：\n"
        '{"score":0到10的整数,"reason":"一句话","pass":true或false}'
    )


def judge_one(q, a):
    """单条语义打分。返回 dict(score, reason, pass)；失败返回 None。

    失败自动重试 1 次（stepfun 通道偶发 5xx），仍失败才降级。"""
    import urllib.request
    import urllib.error
    jcfg = POL.get("judge") or {}
    key = os.environ.get("ZEN_GATE_TOKEN") or jcfg.get("auth")
    if not key:
        return None
    body = json.dumps({
        "model": jcfg.get("model") or JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": _judge_prompt(q, a)},
        ],
        "temperature": 0.1,
        "max_tokens": 300,
    }, ensure_ascii=False).encode("utf-8")
    for attempt in range(2):
        req = urllib.request.Request(jcfg.get("url") or JUDGE_URL, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "polaris-judge/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            time.sleep(0.8)
            continue
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r"\{[^{}]*\}", content)
        if not m:
            time.sleep(0.8)
            continue
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        score = float(obj.get("score", 0))
        return {"score": score,
                "reason": str(obj.get("reason", ""))[:80],
                "pass": bool(obj.get("pass", score >= 7))}
    return None


def judge_batch(items, model=None):
    """批量 judge：把 N 组 (q,a) 打包成一条 prompt，一次 HTTP 请求让模型逐组打分。

    为什么用批量而不是并发：单个模型不接受并发（实测 stepfun 4 线程并发打一个模型
    会挂一半）。但单条请求内部打包 N 条不触发并发限制，反而更快。
    返回 {idx: dict or None}（None=降级放行）。
    """
    import urllib.request
    import urllib.error
    jcfg = POL.get("judge") or {}
    key = os.environ.get("ZEN_GATE_TOKEN") or jcfg.get("auth")
    out = {i: None for i in range(len(items))}
    if not items or not key:
        return out
    model = model or jcfg.get("model") or JUDGE_MODEL
    lines = [f"第{i+1}组：主人说「{q}」，北极星答「{a}」" for i, (q, a) in enumerate(items)]
    user = (
        f"下面是 {len(items)} 组「主人问 + 北极星答」，请逐组按上面四个维度打分。\n"
        + "\n".join(lines)
        + f"\n\n严格输出 {len(items)} 行 JSON，每行一个，不要任何其他文字，格式：\n"
        + '{"i":组号,"score":0到10的整数,"reason":"一句话","pass":true或false}'
    )
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 120 * len(items) + 200,
    }, ensure_ascii=False).encode("utf-8")
    content = None
    for attempt in range(2):
        req = urllib.request.Request(jcfg.get("url") or JUDGE_URL, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "polaris-judge/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            break
        except Exception:
            time.sleep(0.8)
            continue
    if not content:
        return out
    # 按行找 JSON；行数不足时回退到全局 findall 按序归位
    objs = []
    for line in content.splitlines():
        m = re.search(r"\{[^{}]*\}", line)
        if m:
            try:
                objs.append(json.loads(m.group(0)))
            except Exception:
                continue
    if len(objs) < len(items):
        objs = []
        for m in re.findall(r"\{[^{}]*\}", content):
            try:
                objs.append(json.loads(m))
            except Exception:
                continue
    for obj in objs:
        i = int(obj.get("i", -1)) - 1
        if not (0 <= i < len(items)):
            # 无组号时按出现顺序归位
            if len([o for o in objs if int(o.get("i", -1)) == -1]) == 0:
                i = objs.index(obj)
            else:
                continue
        if i in out and out[i] is None:
            score = float(obj.get("score", 0))
            out[i] = {"score": score,
                      "reason": str(obj.get("reason", ""))[:80],
                      "pass": bool(obj.get("pass", score >= 7))}
    return out


def judge_many(items):
    """多模型并行投票：N 条样本同时发给 M 个不同模型，聚合它们的打分。

    WEAVER(NeurIPS 2025) 的核心结论：多个弱 verifier 组合 > 单个强 verifier。
    并发限制是「同一模型」的限制，不同模型之间并行不受影响——所以把同一条样本
    同时发给 step-audio / stepaudio 等不同模型，各自批量打分，再按分数均值聚合，
    单模型噪声被投票抵消，judge 更稳。某个模型失败/返回空则跳过，不拖累整体。
    """
    from concurrent.futures import ThreadPoolExecutor
    jcfg = POL.get("judge") or {}
    models = jcfg.get("models") or [jcfg.get("model") or JUDGE_MODEL]
    jb = max(1, int(jcfg.get("batch", 8)))
    out = {i: None for i in range(len(items))}
    if not items or not (os.environ.get("ZEN_GATE_TOKEN") or jcfg.get("auth")):
        return out
    for s in range(0, len(items), jb):
        chunk = items[s:s + jb]
        # 不同模型并行（每个模型一条批量请求），避开「同模型并发」限制
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            futs = [ex.submit(judge_batch, chunk, m) for m in models]
            results = []
            for fut in futs:
                try:
                    results.append(fut.result())
                except Exception:
                    results.append({})
        # 聚合：分数取中位数（抗单模型极端值），通过与否看多数票；
        # 只要有一个模型判死（<=3）就整条拒 —— 脏数据进库的代价远高于错杀。
        for i in range(len(chunk)):
            scores = []
            reason = ""
            votes = []
            for r in results:
                d = r.get(i)
                if d and d.get("score") is not None:
                    scores.append(d["score"])
                    votes.append(bool(d.get("pass")))
                    if not reason and d.get("reason"):
                        reason = d["reason"]
            if not scores:
                out[s + i] = None
                continue
            ss = sorted(scores)
            n = len(ss)
            med = ss[n // 2] if n % 2 else (ss[n // 2 - 1] + ss[n // 2]) / 2.0
            # 一票否决只在「两个以上模型都判死」时生效：单个模型的标准偏严很常见，
            # 它会把「去吧主人，晚安」判成生硬，用 min<=3 一刀切会误杀 7% 的好样本。
            kills = sum(1 for x in ss if x <= 3)
            ok = (sum(votes) * 2 > len(votes)) and med >= float(jcfg.get("min_score", 6.0)) and kills < 2
            out[s + i] = {"score": med, "reason": reason, "pass": ok}
    return out


def run_gates(q, a, c, skip=()):
    q_key = norm_q(q)
    if not q_key:
        return False, "空问句", ""
    passed, warns = [], []
    enabled = set(POL.get("gates", {}).get("enabled") or [])
    for name, fn in GATES:
        if name in skip:
            continue
        if enabled and name not in enabled:
            continue
        ok, why = fn(q, a, c, q_key)
        if not ok:
            return False, why, ",".join(passed)
        if why.startswith("WARN:"):
            warns.append(why[5:])
        passed.append(name)
    return True, "", ",".join(passed + warns)


# ---------------------------------------------------------------- 模型
def build_model(base, adapter, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, low_cpu_mem_usage=True)
    if adapter and os.path.isdir(adapter):
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.to(device)
    m.eval()
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    else:
        print(f"[model] device={device} / {torch.cuda.get_device_name(0)} / dtype={dtype}")
    return m, tok


def gen(m, tok, prompt, max_new, temp, top_p=0.9):
    import torch
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
    eos = tok.convert_tokens_to_ids(IM_END)
    eos_id = eos if isinstance(eos, int) else None
    with torch.no_grad():
        out = m.generate(
            ids, max_new_tokens=max_new,
            do_sample=temp > 0, temperature=max(temp, 0.01), top_p=top_p,
            repetition_penalty=1.15,
            eos_token_id=eos_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    if IM_END in text:
        text = text.split(IM_END)[0] + IM_END
    return text.strip()


def _cut(t):
    if IM_END in t:
        return (t.split(IM_END)[0] + IM_END).strip()
    return t.strip()


def gen_many(m, tok, prompts, max_new, temp, top_p=0.9):
    """批量生成：一次前向喂多条 prompt，把显卡喂饱。

    实测（GTX 1660 Ti 6G，64 token）：
      batch=1  6155 ms/条  吞吐 0.16 条/s  显存 683 MiB
      batch=8   808 ms/条  吞吐 1.24 条/s
      batch=16  437 ms/条  吞吐 2.29 条/s  显存 772 MiB
    单条串行时 GPU 利用率只有 27%，显存用掉 12%——14 倍差距全在批大小上。
    按长度排序分批，减少 padding 浪费；解码后按原序返回。
    """
    import torch
    if not prompts:
        return []
    pad = tok.pad_token_id or tok.eos_token_id
    eos = tok.convert_tokens_to_ids(IM_END)
    eos_id = eos if isinstance(eos, int) else None
    old_side = tok.padding_side
    tok.padding_side = "left"           # 左填充：生成位置固定在右侧
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    step = max(1, int(policy_auto("batch", 16)))
    try:
        with torch.no_grad():
            for s in range(0, len(order), step):
                chunk = order[s:s + step]
                while True:
                    try:
                        enc = tok([prompts[i] for i in chunk], return_tensors="pt",
                                  padding=True, add_special_tokens=False,
                                  truncation=True, max_length=1024).to(m.device)
                        out = m.generate(
                            **enc, max_new_tokens=max_new,
                            do_sample=temp > 0, temperature=max(temp, 0.01), top_p=top_p,
                            repetition_penalty=1.15,
                            eos_token_id=eos_id, pad_token_id=pad,
                        )
                        for j, i in enumerate(chunk):
                            outs[i] = _cut(tok.decode(out[j][enc["input_ids"].shape[1]:],
                                                      skip_special_tokens=False))
                        del enc, out
                        break
                    except torch.cuda.OutOfMemoryError:
                        # 6G 卡余量靠实测探测：爆了就减半重来，不写死一个「安全值」
                        torch.cuda.empty_cache()
                        if len(chunk) <= 1:
                            raise
                        chunk = chunk[:max(1, len(chunk) // 2)]
                        print(f"[batch] 显存不足，本批降到 {len(chunk)} 条重试", flush=True)
    finally:
        tok.padding_side = old_side
    return outs


LEAK_PAT = re.compile(
    r"【|】|「|」|^\s*\d{4}\s*年|\d{4}\s*年\s*[春夏秋冬]|星期[一二三四五六日天]|"
    r"举办的|记者|采访|调查显示|第[一二三四五六七八九十]+届|据.{0,4}报道|"
    r"本报讯|通讯员|实习生|（来源[:：]|（摘自|（编辑|（记者|——[一-鿿]{2,4}（"
)


def g_leak(q, a):
    """语料泄漏闸：拦「预训练原文被整段吐出来」。

    实测漏进来两条（已从库中清掉）：
      · `2016年夏天夜色得打滚，今天一直吵不起来。不是在公园里的虫子有点儿…`
      · `我陪你想想你的好，主人。\\n【2016年夏季第一天】这个星期是我在深圳举办的“`
    这两条的共同点是「首句还像对话，后面接上网文/日记原文」。旧闸门全放行：
    字符集正常（garble 过）、没有英文碎片（script 过）、带「主人」（emotion 过）、
    与教材 2-gram 也有交集（anchor 过）。只有「日记体/新闻体特征词」能认出来。

    注意：**不拿「多段」当拒收理由**。第一版这么干过，实测一轮 12 条里「多段」
    拒了 4 条、占全部拒收的一半——0.3B 只是习惯分段，并不等于泄漏。多段交给
    _first_para 截首段处理，这里只管特征词。
    """
    a = a.replace(IM_END, "").strip()
    if not a:
        return True, ""
    m = LEAK_PAT.search(a)
    if m:
        return False, f"疑似语料原文（命中「{m.group(0).strip()}」）"
    return True, ""


def _first_para(t):
    """只留第一段：0.3B 常把对话答完，又顺手续一段网文/日记原文。

    直接判「多段 = 脏」会误杀——实测一轮 12 条里因此拒了 4 条，占拒收一半，
    而被拒的只是「诶，你会做饭吗」这种正常分段。截首段既保住合格的首句，
    又让后半段污染自然消失；万一首段本身就是泄漏，g_leak 仍会兜一次。
    """
    for line in t.replace(IM_END, "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _cand_score(a):
    """候选打分（越小越好）：只判第一层阀门那几项，不查库。

    拒绝采样（best_of>1）时用来挑最干净的一条。判据与 gen_safe 的原始意图一致：
    乱码 > 英文碎片 > 复读 > 空答案。不跑全闸是成本考虑——全闸要查 SQLite，
    每题跑 3 遍会把生成成本放大数倍。
    """
    a2 = a.replace(IM_END, "").strip()
    if not a2:
        return 200.0
    s = 0.0
    if "\ufffd" in a2:
        s += 100.0
    lat = sum(1 for ch in a2 if ch.isascii() and ch.isalpha())
    s += 40.0 * (lat / len(a2))
    s += 20.0 * repeat_ratio(a2)
    return s


def gen_safe(m, tok, prompt, max_new, temp, top_p=0.9, max_retry=4):
    """第一层阀门：生成产物先过乱码闸，命中就换个采样参数重生成，最多 max_retry 次。

    0.3B 底座吐字碎片化会冒出 乱码替换符 / 异常字符（实测占拒收首位），这类文本
    长度正常、格式也过关，单靠下游闸门拦不住。所以生成端先自检：乱码就拒绝并重来，
    全部重试失败才交回下游闸门（仍被拒，不入库）。这正是「第一层阀门」的语义。

    best_of（policy 热参数）：>1 时至少生成这么多个候选再取最干净的一条——
    对应 TTRL 的「同题多次采样、用一致性/质量挑选」思路，代价是生成时间线性上升。
    """
    need = max(1, int(policy_auto("best_of", 1)))
    best, best_s = "", None
    tries = 0
    while tries < max_retry:
        tries += 1
        t = temp * (1.0 + 0.12 * (tries - 1)) if tries > 1 else temp
        p = min(top_p + 0.04 * (tries - 1), 0.95)
        cand = gen(m, tok, prompt, max_new, t, p)
        s = _cand_score(cand)
        if best_s is None or s < best_s:
            best, best_s = cand, s
        if best_s == 0.0 and tries >= need:
            break
    return best


def few_shot(c, topic, k=2):
    """回喂：同领域示例。优先真值教材当示例（教材只当示例，不冒充产出）。

    不硬塞「主人」示例——称呼习惯由模型从教材自然分布里学，我们靠 cmd_stats
    观察它到底在哪些话题上用主人/北极星（即「通过做梦搞出来」这一句的本意）。"""
    rows = c.execute("select q,a from seeds where topic=? order by random() limit ?", (topic, k)).fetchall()
    if not rows:
        rows = c.execute("select q,a from pairs where topic=? order by random() limit ?", (topic, k)).fetchall()
    return "".join(f"{IM_START}user\n{q}{IM_END}\n{MARK}{a}{IM_END}\n" for q, a in rows)


# 生成长度按领域收紧：0.3B 极爱跑题拉长，max_new 一刀切 120 会让短答域
# 大量「长度失衡」被拒（实测 greet 中位 12 字，模型却吐 90 字）。
# 短答域直接压到 40 token，情感域 64（够说 2-3 句暖话，又不至于写成小作文）。
def gen_len(tp):
    """生成长度从 policy 读（热生效）：改 policy.json 的 gen_len 立刻影响下一题。"""
    return POL.get("gen_len", {}).get(tp, 90)


# ---------------------------------------------------------------- 出题（规则，不是模型）
# 实测结论：0.3B 挂对话模块后会「以助手口吻说中文」，但不具备任何指令遵循能力。
# 让它「改写问句」或「出个题」，产出全是预训练语料碎屑（如"周末没有，因为我想自己是好的"、
# "ertical ertical ertical"）。所以出题这一步交回规则——这与代码梦一致
# （make_spec 本来就是 15 个硬编码模板 random.choice，也不是模型自己想题）。

SHORT_SYN = {
    "你好": ["喂，你好", "嗨，你好", "哈喽，你好", "你好啊，在不在"],
    "在吗": ["在不在", "有人在吗", "你现在在吗", "在吗在吗"],
    "你是谁": ["介绍一下你自己", "你到底是什么", "自我介绍一下", "你究竟是谁"],
    "你叫什么": ["你叫什么名字", "你的名字是什么", "我该怎么叫你"],
    "你会什么": ["你能干啥", "你会干什么", "你有什么本事", "你能帮我做什么"],
    "你是人吗": ["你该不会是人吧", "你是真人吗", "到底是不是人"],
    "谢谢": ["谢谢你", "谢啦", "多谢", "谢了主人跟你说"],
    "几点了": ["现在几点了", "现在什么时间", "报个时间"],
    "好无聊": ["我好无聊", "无聊死了", "没事干，好无聊"],
    "我不开心": ["我今天不开心", "心情不太好", "有点难过"],
    "讲个笑话": ["给我讲个笑话", "讲个笑话呗", "来个笑话听听"],
    "帮我查个东西": ["帮我查一下", "给我查个东西", "能帮我查点东西吗"],
    "随便说点什么": ["随便聊点吧", "你说点什么", "说点什么吧"],
}

SUFFIX = ("啊", "呢", "吧", "呀")
# 去掉「喂，」：它只适合打招呼，配到「喂，你会写代码吗」就很怪（实测泛滥）
PREFIX = ("嗯，", "那个，", "话说，", "跟我说说，", "我想问下，")
# 这些字结尾的问句再加语气词会不通（"你有朋友吗"+"啊" => "你有朋友吗啊"）
NO_SUFFIX_TAIL = ("吗", "呢", "吧", "啊", "呀", "么", "？", "?", "了")


def vary(q):
    """规则扩问：同一答案的多种问法。短问句只用手写同义（机械加语气词太雷同），
    中长问句才做前后缀变体；已带语气词的问句不再叠加后缀。"""
    qs = q.strip().rstrip("?？。.！!~～ ")
    out = list(SHORT_SYN.get(qs, []))
    if len(qs) >= 5:
        if not qs.endswith(NO_SUFFIX_TAIL):
            out += [qs + s for s in SUFFIX]
        out += [p + qs for p in PREFIX]
        out.append(qs + "？")
    seen, uniq = set(), []
    for x in out:
        k = norm_q(x)
        if k and k != norm_q(q) and k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


# 情感问句多是短句（「我好难过」4 字），事实域的 vary 会返回空。
# 单独给情感域一套安全变体：只加语气词/口语前缀，不改语义、不破坏语感。
EMO_PREFIX = ("唉，", "其实，", "主人，", "说真的，")
# 生活域用用户自己的口语起手，别用「请问」「麻烦推荐」这类书面请求。
LIFE_PREFIX = ("诶，", "那啥，", "话说，", "你说，")


def vary_emo(q):
    qs = q.strip().rstrip("?？。.！!~～ ")
    out = []
    if len(qs) >= 3 and qs[-1] not in "吗呢吧啊呀了":
        out += [qs + s for s in ("啊", "呀")]
    out += [p + qs for p in EMO_PREFIX]
    seen, uniq = set(), []
    for x in out:
        k = norm_q(x)
        if k and k != norm_q(q) and k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def vary_life(q):
    """生活域问法变体：只加口语语气词和日常起手，不改语义。

    生活问句大多很短（「我饿了」），走事实域的 vary 会返回空，所以单独一套。
    """
    qs = q.strip().rstrip("?？。.！!~～ ")
    out = []
    if len(qs) >= 3 and qs[-1] not in "吗呢吧啊呀了":
        out += [qs + s for s in ("啊", "呀", "呢")]
    out += [p + qs for p in LIFE_PREFIX]
    seen, uniq = set(), []
    for x in out:
        k = norm_q(x)
        if k and k != norm_q(q) and k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def ask_prompt(c, q, tp):
    return f"{few_shot(c, tp, 2)}{IM_START}user\n{q}{IM_END}\n{MARK}"


# 领域优先级：人格与问候是模块的核心职责（README 三件事的第一条就是「守住人格」），
# 必须先填满；qa/other 是陪衬。实测按纯随机选材会把 10/13 的产出都落在 other 上。
TOPIC_PRIORITY = {"identity": 0, "greet": 1, "emotion": 2, "life": 3,
                  "clarify": 4, "chat": 5, "qa": 6, "other": 7}


def pick_seed_with_variants(c):
    """挑教材：先按「领域覆盖率」低者优先（按短板出题），同覆盖率按领域重要性，
    再取「变体还没用光」的，避免同一答案被反复扩。"""
    cnt = dict(c.execute("select topic,count(*) from pairs group by topic").fetchall())
    total = dict(c.execute("select topic,count(*) from seeds group by topic").fetchall())
    topics = sorted(total, key=lambda t: (cnt.get(t, 0) / max(1, total[t]),
                                          TOPIC_PRIORITY.get(t, 9)))
    for tp in topics:
        rows = c.execute("select q,a from seeds where topic=? order by random() limit 60", (tp,)).fetchall()
        for q, a in rows:
            vs = vary(q)
            if not vs:
                continue
            free = [v for v in vs if not c.execute(
                "select 1 from pairs where q_key=?", (norm_q(v),)).fetchone()]
            if free:
                return q, a, tp, vs
    return None


def mk(q, tp, src, seed_q, prompt=None, ans=None):
    """一个「做梦任务」：出题与生成在这里解耦。

    原来是「选一题→立刻生成→立刻过闸」串行跑，显卡一次只吃到 1 条 prompt，
    GPU 利用率只有 27%。现在先串行把 n 道题都出好（纯查库、不碰 GPU），
    再整批生成、整批投票——批大小 16 时吞吐是单条的 14 倍。
    """
    return {"q": q, "tp": tp, "src": src, "seed_q": seed_q,
            "prompt": prompt, "ans": ans}


def prep_expand(c, seen=None):
    """扩问 + 真值答案：答案来自人工写定教材，只换问法。

    这一路根本不需要模型生成（占 25% 配额），所以整条批量化对它零收益，
    但把它从生成队列里摘出去，等于免费让出 25% 的 GPU 预算给其他三路。

    变体必须同时避开 pairs 和 seeds 的原题：实测「你是谁」的变体「介绍一下你自己」
    恰好是另一条 identity 教材的原题，答案不同 ⇒ g_conflict 判「与教材答案冲突」，
    一轮里白烧 3 个配额。撞上了就换下一个种子，不要硬塞。
    """
    for _ in range(4):
        r = pick_seed_with_variants(c)
        if not r:
            return None
        q0, a0, tp, vs = r
        # 同一条教材最多出 2 个变体就收手。
        # 不限的话，`hello` 会产出「话说，hello / 那个，hello / hello呀 / hello呢」
        # 四条同答案样本——expand 不需要模型生成、必然通过闸门，于是它会把整个银行
        # 灌满教材变体（实测 120/126 = 95%），模型自产的只有 6 条，等于没学新东西。
        if c.execute("select count(*) from pairs where seed_src=?", (q0[:40],)).fetchone()[0] >= 2:
            continue
        free = [v for v in vs
                if norm_q(v) not in (seen or ())
                and not c.execute("select 1 from pairs where q_key=?", (norm_q(v),)).fetchone()
                and not c.execute("select 1 from seeds where q_key=?", (norm_q(v),)).fetchone()]
        if free:
            return mk(free[0], tp, "expand", q0, ans=a0)
    return None


def prep_selfplay(c, seen=None):
    """变体问法 + 模型自答：这才是自举的真产出，答案必须过全部闸门才入库。
    按短板优先：产出越少的领域越先出题（对应 loop.md 的「按缺口出题」）。"""
    cnt = dict(c.execute("select topic,count(*) from pairs group by topic").fetchall())
    pool = sorted(["identity", "greet", "chat", "clarify", "qa", "emotion", "life"],
                  key=lambda t: cnt.get(t, 0))
    for tp in pool:
        rows = c.execute("select q,a from seeds where topic=? order by random() limit 30", (tp,)).fetchall()
        for q0, a0 in rows:
            free = [v for v in vary(q0)
                    if norm_q(v) not in (seen or ())
                    and not c.execute("select 1 from pairs where q_key=?", (norm_q(v),)).fetchone()]
            if free:
                q = free[0]
                return mk(q, tp, "selfplay", q0, prompt=ask_prompt(c, q, tp))
    return None


def prep_replay(c, seen=None):
    """重答「敷衍短答」：从最短的 20 条里随机挑一条重答，看能否更完整。

    旧版直接取全局最短的一条，结果反复重答同一题（实测「一年有几个月啊」被重答 5 次，
    每次都因长度失衡被拒）。改为：① 排除 qa（事实短答本就不该拉长）；
    ② 在最短 20 条里随机，避免同题空转；③ 低于 6 字的不重答（无意义）。"""
    rows = c.execute(
        "select q,a,topic from (select q,a,topic,length(a) as L from pairs "
        "where topic!='qa' and length(a)>=6 order by L asc limit 20) order by random()"
    ).fetchall()
    for q, a0, tp in rows:
        if norm_q(q) in (seen or ()):
            continue
        return mk(q, tp, "replay", q, prompt=ask_prompt(c, q, tp))
    return None


def prep_emotion(c, seen=None):
    """情感域自举：从情感种子里挑一个「主人带情绪」的问法，让北极星现场接住。

    这是本次新增的主攻方向——之前的链路只练事实问答，银行里没有一丝温度。
    这里用情感种子当 few-shot（ask_prompt(..., 'emotion')），把「先共情后建议」
    的范式灌进生成；再靠 g_emotion 态度闸把「多喝热水」式冷回复挡在门外。
    种子耗尽后回退到「重答一条已入库的情感样本」，保证这一路不会空转。"""
    rows = c.execute("select q,a from seeds where topic='emotion' order by random() limit 40").fetchall()
    if rows:
        for q0, a0 in rows:
            # 变体优先、原题兜底：原题与教材 q_key 相同，容易撞冲突闸；先用变体更干净。
            free = [x for x in vary_emo(q0) + [q0]
                    if norm_q(x) not in (seen or ())
                    and not c.execute("select 1 from pairs where q_key=?", (norm_q(x),)).fetchone()]
            if free:
                q = free[0]
                return mk(q, "emotion", "emotion", q0, prompt=ask_prompt(c, q, "emotion"))
    row = c.execute("select q from pairs where topic='emotion' order by random() limit 1").fetchone()
    if row:
        q = row[0]
        return mk(q, "emotion", "emotion", q, prompt=ask_prompt(c, q, "emotion"))
    return None


def prep_life(c, seen=None):
    """生活域自举：从吃喝种子里挑一个日常问法，让北极星现场接一句家里人的话。

    和情感域同构（种子当 few-shot、同问允许不同答），但闸门换成 g_life 域词判定，
    专治「今天吃啥」被答成「要好好生活、保持好心情」这种正确废话。
    """
    rows = c.execute("select q,a from seeds where topic='life' order by random() limit 40").fetchall()
    if rows:
        for q0, a0 in rows:
            free = [x for x in vary_life(q0) + [q0]
                    if norm_q(x) not in (seen or ())
                    and not c.execute("select 1 from pairs where q_key=?", (norm_q(x),)).fetchone()]
            if free:
                q = free[0]
                return mk(q, "life", "life", q0, prompt=ask_prompt(c, q, "life"))
    row = c.execute("select q from pairs where topic='life' order by random() limit 1").fetchone()
    if row:
        q = row[0]
        return mk(q, "life", "life", q, prompt=ask_prompt(c, q, "life"))
    return None


# 情感域给 20%、生活域 20%：这两条腿是「像家里人」而不是「像问答机」的关键。
# expand 让出配额（0.30 -> 0.25），selfplay 保持 0.25，replay 降到 0.10。
ROUTES = [("expand", 0.25, prep_expand), ("selfplay", 0.25, prep_selfplay),
          ("emotion", 0.20, prep_emotion), ("life", 0.20, prep_life),
          ("replay", 0.10, prep_replay)]


def pick_route(rng):
    """按 policy.routes 的权重选路（热生效：改权重不用重启）。"""
    w = POL.get("routes") or {name: d for name, d, _ in ROUTES}
    items = [(name, float(w.get(name, 0.0)), fn) for name, d, fn in ROUTES]
    tot = sum(x[1] for x in items) or 1.0
    x = rng.random() * tot
    acc = 0.0
    for name, wt, fn in items:
        acc += wt
        if x <= acc:
            return name, fn
    return items[-1][0], items[-1][2]


# ---------------------------------------------------------------- 主循环
def vote_support(c, m, tok, q, tp, main_a, temp):
    """自洽投票（TTRL 的最小可用版）：同题独立再采几条，看它们支不支持第一条。

    逻辑很朴素：同一个问题问三遍，三次说法都差不多 ⇒ 这不是随口编的，可以当
    「伪标签」收进银行；三遍各说各的 ⇒ 模型在这题上根本没谱，宁可扔掉也不收。
    这正是 TTRL 用多数投票当 reward 的简化版，区别是它拿去更新策略，我们只用来
    决定「这条样本配不配进银行」。

    n-1 条里过半数与主答案相似度 >= sim 才算通过。返回 (是否通过, 最高相似度)。
    """
    v = POL.get("vote") or {}
    n = int(v.get("n", 1))
    if n <= 1:
        return True, 1.0
    thr = float(v.get("sim", 0.6))
    ss = []
    for _ in range(n - 1):
        a2 = gen_safe(m, tok, ask_prompt(c, q, tp), gen_len(tp), min(temp, 0.85))
        ss.append(sim(main_a, a2.replace(IM_END, "").strip()))
    if not ss:
        return True, 1.0
    need = (len(ss) + 1) // 2          # n=2 -> 1 票，n=3 -> 2 票（过半数）
    return sum(1 for s in ss if s >= thr) >= need, (max(ss) if ss else 0.0)


def batch_fill(m, tok, tasks, temp, max_retry=4):
    """给 ans 为空的任务批量填答案——「第一层阀门」的批量化。

    原来是「生成一条 → 查乱码 → 不过就重来」，一次只喂 GPU 一条 prompt，显卡 73% 在空等。
    现在改成「整批生成 → 挑出乱码的那几条 → 整批重来」：重试语义完全不变
    （乱码就拒绝并重新生成），但每次重试都在跑满 batch。

    best_of 融合进轮数：best_of=N 等价于「至少采 N 轮、每轮都保留更干净的那条」，
    因为 _cand_score 就是按「干净程度」排序的，多轮取最优正是它的本意。
    """
    alive = [t for t in tasks if t["ans"] is None]
    if not alive:
        return
    rounds = max(int(max_retry), int(policy_auto("best_of", 1)))
    for attempt in range(rounds):
        if not alive:
            break
        t_ = temp * (1.0 + 0.12 * attempt) if attempt else temp
        p_ = min(0.9 + 0.04 * attempt, 0.95)
        # 统一取本批最大生成长度，**不按领域分组长短混批**：
        # 分组的直觉是「短答域别浪费 token」，实测反而致命——16 个任务被 5 个
        # 长度档切成 5 组，每组只剩 3 条，batch 优势全没了（3.06s/条 vs 应到的 0.45s/条）。
        # 短答域多生成几个 token 有 eos 兜着，几乎不亏；batch 装满才是大头。
        mlen = max(t["mlen"] for t in alive)
        outs = gen_many(m, tok, [t["prompt"] for t in alive], mlen, t_, p_)
        for t, txt in zip(alive, outs):
            txt = _first_para(txt)
            if not txt:
                continue
            if t["ans"] is None or _cand_score(txt) < _cand_score(t["ans"]):
                t["ans"] = txt
        alive = [t for t in alive if not g_garble("", (t["ans"] or " "))[0]]


def dream_once(c, m, tok, n, temp, rng):
    """跑 n 次做梦，返回统计。cmd_run 与 cmd_auto 共用这一段。

    五个阶段，把 GPU 从「一条一条等」变成「装满再跑」：
      1. 出题（串行查库，不碰 GPU）
      2. 批量生成（第一层阀门：乱码整批重来）
      3. 串行过前置闸（格式/人格/态度/域词/离题，不含 vote）
      4. 批量自洽投票（同题再采，一次批量）
      5. 串行决定入库（replay 的更新逻辑在这里）
    """
    ok = rej = 0
    by_reason = collections.Counter()
    by_source = collections.Counter()

    # ---------- 阶段 1：出题 ----------
    tasks = []
    seen_qk = set()          # 本批已出的问法：一次性出 96 道题时库还没写，只能自己记
    for i in range(n):
        if i % 5 == 0:
            load_policy()   # 热更新：改 policy.json 不必重启，最多等 5 条样本生效
        t = None
        # 撞重复或池子空就换一路再出，不要把「重复」当成「质量差」直接丢掉：
        # 实测 96 题里 30 条是这么被白丢的（31% 产能），换一道题就还能用。
        for _try in range(6):
            name, fn = pick_route(rng)
            try:
                cand = fn(c, seen_qk)
            except Exception as e:
                print(f"  [{i+1}/{n}] {name} 出题异常 {type(e).__name__}: {str(e)[:80]}")
                continue
            if not cand:
                continue
            qk = norm_q(cand["q"])
            if qk in seen_qk:
                continue
            seen_qk.add(qk)
            t = cand
            break
        if not t:
            rej += 1
            by_reason["出题池暂时耗尽"] += 1
            continue
        t["i"] = i + 1
        t["idx"] = len(tasks)
        if t["ans"] is None:
            t["mlen"] = gen_len(t["tp"])
        tasks.append(t)

    # ---------- 阶段 2：批量生成 ----------
    batch_fill(m, tok, tasks, temp)

    # ---------- 阶段 3：串行过前置闸（不含 vote） ----------
    for t in tasks:
        t["good"], t["why"], t["gates"] = True, "", ""
        if t["ans"] is None:
            t["good"], t["why"] = False, "生成失败（全部重试仍为空）"
            continue
        q, src = t["q"], t["src"]
        a_clean = t["ans"].replace(IM_END, "")
        # 情感/生活题允许「同问不同答」，走冲突闸等于把自产的暖答全灭。
        skip = ("conflict",) if (src in ("replay", "emotion", "life")) else ()
        good, why, gates = run_gates(q, t["ans"], c, skip=skip)
        if good and src in ("selfplay", "emotion", "life"):
            # 情感/生活域也必须过离题闸：只判「有没有温度/吃没吃喝」，判不出
            # 「我撑不住了 -> 两个月以前我在美国调研…」这种又暖又离题的崩坏。
            thr = TH.get("anchor_selfplay", 0.10) if src == "selfplay" else TH.get("anchor_dom", 0.06)
            ok2, why2 = g_anchor(q, a_clean, c, thr)
            if not ok2:
                good, why = False, why2
            else:
                gates += ",anchor"
                t["need_vote"] = True
        t["good"], t["why"], t["gates"] = good, why, gates

    # ---------- 阶段 3.5：语义 judge（第二层阀门，打分机关键闸） ----------
    # 只对「模型自产且已过规则闸」的候选 judge：expand 是手写真值不用判，replay 走阶段 5
    # 的稳定性判据。judge 用更强的免费模型判四维语义，专治规则闸拦不住的「字面通、意思散」。
    jcfg = POL.get("judge") or {}
    j_on = bool(jcfg.get("enabled", True))
    jmin = float(jcfg.get("min_score", 6.5))
    need_judge = [t for t in tasks if t["good"] and t.get("need_vote")]
    if j_on and need_judge:
        results = judge_many([(t["q"], t["ans"].replace(IM_END, "")) for t in need_judge])
        for i, t in enumerate(need_judge):
            r = results.get(i)
            if r is None:
                # judge 不可用（key 缺失/断网/解析失败）→ 降级放行，只记标记，不卡闭环
                t["gates"] += ",judge-skip"
                continue
            t["gates"] += f",judge:{r['score']:.1f}"
            if r["score"] < jmin:
                t["good"] = False
                t["why"] = f"语义不合格({r['score']:.1f})：{r['reason']}"

    # ---------- 阶段 4：批量自洽投票 ----------
    # 只对「模型自产」的答案投票：expand 有教材真值、replay 是更新旧条目，都不该多花算力。
    vcfg = POL.get("vote") or {}
    vn = int(vcfg.get("n", 1))
    vthr = float(vcfg.get("sim", 0.6))
    vmin = int(vcfg.get("min_bank", 0))
    bank_now = c.execute("select count(*) from pairs").fetchone()[0]
    voters = [t for t in tasks if t.get("need_vote") and t["good"]]
    # 攒量期投票不否决：实测同题两次生成相似度只有 0.09~0.11，此时用 0.6 当门槛
    # 等于随机丢弃（一轮 16 条里投票杀掉 3 条、且全是正常答案）。综述那条结论——
    # 「自我提升是锐化，判别能力必须强于生成能力」——在银行还很薄时是反过来的：
    # 判别能力不足就投票，会把仅有的产出全砍掉，形成死锁。所以先攒量到 min_bank，
    # 再让投票真正开始否决。
    if vn > 1 and voters and bank_now >= vmin:
        flat, owner = [], []
        for t in voters:
            for _ in range(vn - 1):
                flat.append(t["prompt"])      # 复用同一 prompt，避免 few-shot 重抽导致口径不一致
                owner.append(t["idx"])
        outs = gen_many(m, tok, flat, max(t["mlen"] for t in voters), min(temp, 0.85))
        agg = collections.defaultdict(list)
        for oi, txt in zip(owner, outs):
            main_a = tasks[oi]["ans"].replace(IM_END, "").strip()
            agg[oi].append(sim(main_a, _cut(txt).replace(IM_END, "").strip()))
        for t in voters:
            ss = agg.get(t["idx"], [])
            need = (len(ss) + 1) // 2          # n=2 -> 1 票，n=3 -> 2 票（过半数）
            hit = sum(1 for s in ss if s >= vthr)
            if hit < need:
                t["good"] = False
                t["why"] = f"自洽不足（最高相似度 {max(ss) if ss else 0:.2f}）"
            else:
                t["gates"] += ",vote"
    elif voters:
        for t in voters:
            t["gates"] += ",vote-warmup"

    # ---------- 阶段 5：入库 ----------
    for t in sorted(tasks, key=lambda x: x["i"]):
        i, q, tp, src = t["i"], t["q"], t["tp"], t["src"]
        a_clean = (t["ans"] or "").replace(IM_END, "")
        good, why, gates = t["good"], t["why"], t["gates"]
        if good and src == "replay":
            # 重答不能走冲突闸——重答必然与库内答案不同，走这闸等于全拒（实测 2/2 被误杀）。
            # 改判「稳定性」：与原答高度一致 = 这次重答稳，取更完整的那条；
            # 差太远 = 对同一问题答案飘，是能力边界，进 rejects 留证。
            old = c.execute("select id,a from pairs where q_key=?", (norm_q(q),)).fetchone()
            if old:
                s = sim(old[1], a_clean)
                if s < 0.6:
                    good, why = False, f"重答不稳定（与原答相似度 {s:.2f}）"
                elif len(a_clean) > len(old[1]) * 1.3:
                    c.execute("update pairs set a=?, gates=? where id=?", (a_clean, gates, old[0]))
                    ok += 1
                    by_source["replay+"] += 1
                    print(f"  [{i}/{n}] replay   UPDATE [{tp}] Q={q[:22]} -> A={a_clean[:26]}")
                    c.commit()
                    continue
                else:
                    good, why = False, "重答无增益（与原答一致或更短）"
        if not good:
            rej += 1
            by_reason[why.split("（")[0]] += 1
            c.execute("INSERT INTO rejects(q,a,reason,detail,topic,source,created_at) VALUES(?,?,?,?,?,?,?)",
                      (q, a_clean, why.split("（")[0], why, tp, src, time.strftime("%Y-%m-%d %H:%M:%S")))
            print(f"  [{i}/{n}] {src:8s} REJECT {why[:44]} | Q={q[:26]}")
        else:
            try:
                c.execute("INSERT INTO pairs(q,a,q_key,topic,source,seed_src,gates,created_at) VALUES(?,?,?,?,?,?,?,?)",
                          (q, a_clean, norm_q(q), tp, src, t["seed_q"][:40], gates,
                           time.strftime("%Y-%m-%d %H:%M:%S")))
            except sqlite3.IntegrityError:
                # 兜底：撞 unique(q_key) 说明这题库内已有，不该让整轮崩掉
                rej += 1
                by_reason["重复问法（库内已有）"] += 1
                print(f"  [{i}/{n}] {src:8s} SKIP   重复问法 Q={q[:26]}")
                continue
            ok += 1
            by_source[src] += 1
            print(f"  [{i}/{n}] {src:8s} PASS   [{tp}] Q={q[:22]} -> A={a_clean[:26]}")
        c.commit()
    return ok, rej, by_reason, by_source


def _log_run(c, n, ok, rej, by_source):
    size = c.execute("select count(*) from pairs").fetchone()[0]
    c.execute("INSERT INTO runs(n_try,n_ok,n_reject,bank_size,note,created_at) VALUES(?,?,?,?,?,?)",
              (n, ok, rej, size, json.dumps(dict(by_source), ensure_ascii=False),
               time.strftime("%Y-%m-%d %H:%M:%S")))
    c.commit()
    return size


def _report(n, ok, rej, by_reason, by_source, size, dt):
    print(f"\n[本轮] 尝试 {n} / 入库 {ok} / 拒收 {rej} / 用时 {dt:.0f}s")
    print(f"[来源] {dict(by_source)}")
    if by_reason:
        print("[拒收原因]")
        for r, k in by_reason.most_common():
            print(f"   {k:3d}  {r}")
    print(f"[银行] 现有 {size} 条产出")


def cmd_run(n, temp, seed):
    import random
    load_policy(force=True)
    n = n if n is not None else int(policy_auto("n", 20))
    temp = temp if temp is not None else float(policy_auto("temp", 0.8))
    rng = random.Random(seed)
    c = conn()
    if c.execute("select count(*) from seeds").fetchone()[0] == 0:
        print("[ERR] 教材区为空，先跑 --init")
        return
    cur = latest_adapter()
    m, tok = build_model(BASE, cur)
    print(f"[model] {BASE} + {cur or '无适配器'}")
    t0 = time.time()
    ok, rej, br, bs = dream_once(c, m, tok, n, temp, rng)
    size = _log_run(c, n, ok, rej, bs)
    _report(n, ok, rej, br, bs, size, time.time() - t0)


def _free_gpu(m):
    import gc
    import torch
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def latest_adapter():
    """当前该用哪张卡：最新训练出来的那张，而不是原始移植卡。

    这里踩过一个会「丢成果」的坑：cmd_auto 原来写的是
        cur = ADAPTER if os.path.isdir(ADAPTER) else None
    每次代码热切换、supervise 重起 worker，cur 都会重置回 adapter_03b——
    刚训好的 adapter_r2_n174 就这么被丢了，等于每改一次代码就白训一轮。
    改成扫目录取最新的（跳过 _ckpt 中间档），训练成果才能跨进程累积。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    black = set((POL.get("adapter") or {}).get("blacklist") or [])
    cands = []
    if os.path.isdir(ADAPTER) and os.path.basename(ADAPTER) not in black:
        cands.append((os.path.getmtime(ADAPTER), ADAPTER))
    for p in glob.glob(os.path.join(here, "adapter_r*")):
        if "_ckpt" in os.path.basename(p):
            continue
        if os.path.basename(p) in black:
            continue
        if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
            cands.append((os.path.getmtime(p), p))
    cands.sort()
    return cands[-1][1] if cands else None


def cmd_auto(a):
    """自动闭环：做梦 -> 新增攒到 train_every -> 增量训练 -> 换新卡 -> 接着做梦。

    这就是「大量做梦，过检验入银行，攒到 N 条训一轮」的落地。
    每轮都重建模型：训练换了卡，必须重新挂载，否则还在用旧卡做梦。

    热更新（两层，都不需要人工重启）：
      · policy.json 变了 -> 参数/阈值/词表/权重下一轮生效；
      · 本脚本 .py 变了 -> 本轮结束 execv 原地接管，银行/卡/命令行参数全部保留。
    """
    import random
    import subprocess
    rng = random.Random(a.seed)
    load_policy(force=True)
    c = conn()
    if c.execute("select count(*) from seeds").fetchone()[0] == 0:
        print("[ERR] 教材区为空，先跑 --init")
        return
    cur = latest_adapter()
    trained_at = c.execute("select count(*) from pairs").fetchone()[0]
    here = os.path.dirname(os.path.abspath(__file__))
    me = os.path.abspath(__file__)
    src_mtime = os.path.getmtime(me)
    rounds = 0
    while a.max_rounds <= 0 or rounds < a.max_rounds:
        load_policy()
        n = a.n if a.n is not None else int(policy_auto("n", 40))
        temp = a.temp if a.temp is not None else float(policy_auto("temp", 0.8))
        train_every = a.train_every if a.train_every is not None else int(policy_auto("train_every", 50))
        stop_at = a.stop_at if a.stop_at is not None else int(policy_auto("stop_at", 500))
        epochs = a.epochs if a.epochs is not None else int(policy_auto("epochs", 3))
        anchor = a.anchor if a.anchor is not None else int(policy_auto("anchor", 120))
        rounds += 1
        bank = c.execute("select count(*) from pairs").fetchone()[0]
        print(f"\n{'='*66}\n[auto] 第 {rounds} 轮 | 卡={os.path.basename(cur) if cur else '无'} "
              f"| 银行={bank} | 距阈值={max(0, train_every-(bank-trained_at))} "
              f"| best_of={policy_auto('best_of', 1)}\n{'='*66}")
        m, tok = build_model(BASE, cur)
        t0 = time.time()
        ok, rej, br, bs = dream_once(c, m, tok, n, temp, rng)
        _free_gpu(m)
        size = _log_run(c, n, ok, rej, bs)
        _report(n, ok, rej, br, bs, size, time.time() - t0)

        new = size - trained_at
        if new < 0:
            # 清过库（删脏样本）后 size 会小于上次训练时的快照，不重置就会一直
            # 「距阈值为负」卡着不训练。以当前 size 为新基线。
            trained_at, new = size, 0
        if new >= train_every:
            out = os.path.join(here, f"adapter_r{rounds}_n{size}")
            cmd = [sys.executable, "chat_train.py", "--out", out,
                   "--epochs", str(epochs), "--anchor", str(anchor), "--bank", DB]
            if cur:
                cmd += ["--init-adapter", cur]
            print(f"\n[auto] 新增 {new} 条 >= {train_every}，触发增量训练 -> {out}")
            rc = subprocess.call(cmd, cwd=here)
            if rc == 0 and os.path.isdir(out):
                cur, trained_at = out, size
                print(f"[auto] 换卡 -> {cur}（下一轮用它继续做梦）")
            else:
                print(f"[auto] 训练失败 rc={rc}，保留旧卡继续")
        if stop_at > 0 and size >= stop_at:
            print(f"\n[auto] 银行 {size} >= stop_at {stop_at}，收工")
            break
        # 代码热切换：本轮跑完再换，避免打断正在生成的样本。
        # 注意用「退出码 78」交给外层 supervise 重起，而不是 execv：
        # execv 会替换当前进程，后台任务包装器会把任务判成已结束，句柄一丢就
        # 既收不到通知也停不掉（实测踩过：进程还活着，但已经管不了了）。
        try:
            now = os.path.getmtime(me)
        except OSError:
            now = src_mtime
        if now > src_mtime + 1e-6:
            print("[auto] 检测到脚本变更 -> 退出，交由 supervise 换新代码重起（银行/卡保留）")
            c.commit()
            c.close()
            sys.exit(78)
    print(f"[auto] 结束 | 卡={cur} | 银行={c.execute('select count(*) from pairs').fetchone()[0]}")


def cmd_supervise(a):
    """常驻监督进程：worker 跑自动闭环，代码一变就退出 78，这里原样重起一个。

    为什么不让 worker 自己 execv：execv 会替换进程，外部的后台任务会判成「已结束」，
    句柄丢失后既不通知也停不掉。父进程常驻、子进程轮换，句柄始终有效——
    这就是「改代码不用人工重启」的正确姿势。
    """
    import subprocess
    me = os.path.abspath(__file__)
    base_seed = a.seed if a.seed is not None else 20261018
    nth = 0
    while True:
        nth += 1
        args = [sys.executable, "-u", me, "--auto"]
        for k in ("n", "temp", "train_every", "epochs", "anchor",
                  "max_rounds", "stop_at"):
            v = getattr(a, k, None)
            if v is not None:
                args += ["--" + k.replace("_", "-"), str(v)]
        # 每次重起换种子：否则 worker 会按同一条随机序列重走一遍，白跑重复题
        args += ["--seed", str(base_seed + nth - 1)]
        print(f"\n[supervise] 启动 worker #{nth} | {' '.join(args[2:])}", flush=True)
        rc = subprocess.call(args, cwd=os.path.dirname(me))
        if rc == 78:
            print("[supervise] worker 因代码变更退出 -> 换新代码重起（银行/卡保留）",
                  flush=True)
            time.sleep(2)
            continue
        print(f"[supervise] worker 退出 rc={rc}，监督结束", flush=True)
        return


def cmd_export(out):
    c = conn()
    rows = c.execute("select q,a,topic,source from pairs order by id").fetchall()
    with open(out, "w", encoding="utf-8") as f:
        for q, a, tp, src in rows:
            f.write(json.dumps({"text": chatml(q, a), "_topic": tp, "_source": src}, ensure_ascii=False) + "\n")
    print(f"[export] {len(rows)} 条 -> {out}")
    # 冲突自检（复用 chat_diag 的口径）
    byq = collections.defaultdict(set)
    for q, a, tp, src in rows:
        byq[norm_q(q)].add(a.strip())
    bad = {k: v for k, v in byq.items() if len(v) > 1}
    print(f"[export] 同问多答组 = {len(bad)}（应为 0）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--auto", action="store_true", help="自动闭环：做梦->攒够 train_every->增量训练->换卡->继续")
    ap.add_argument("--supervise", action="store_true",
                    help="常驻监督：worker 遇代码变更退出后自动换新代码重起（推荐长时间跑用这个）")
    ap.add_argument("--n", type=int, default=None, help="每轮做梦次数（缺省取 policy.json）")
    ap.add_argument("--temp", type=float, default=None, help="采样温度（缺省取 policy.json）")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--export", default=None)
    ap.add_argument("--train-every", type=int, default=None, help="银行新增多少条触发一轮增量训练（缺省取 policy.json）")
    ap.add_argument("--epochs", type=int, default=None, help="每轮增量训练的 epoch（缺省取 policy.json）")
    ap.add_argument("--anchor", type=int, default=None, help="教材锚点条数（防遗忘，缺省取 policy.json）")
    ap.add_argument("--max-rounds", type=int, default=0, help="0=不限")
    ap.add_argument("--stop-at", type=int, default=None, help="银行达到多少条收工（缺省取 policy.json）")
    a = ap.parse_args()
    if a.init:
        cmd_init()
    elif a.stats:
        cmd_stats()
    elif a.run:
        cmd_run(a.n, a.temp, a.seed)
    elif a.auto:
        cmd_auto(a)
    elif a.supervise:
        cmd_supervise(a)
    elif a.export:
        cmd_export(a.export)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
