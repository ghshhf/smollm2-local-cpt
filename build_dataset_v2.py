# -*- coding: utf-8 -*-
"""
build_dataset_v2.py —— 跨宇宙线批量抽取训练样本（角色卡/概念/大纲 + 昨书成对关系补全）
输出：data/role_data_v2.jsonl  {instruction, output, _type, _universe, _src}
策略：通用模板，每条样本显式带宇宙名《U》，避免跨宇宙混淆。
"""
import os, re, json, zipfile, tempfile, glob

NOVEL = "D:/work/novel"
OUT = "data/role_data_v2.jsonl"

# (路径, 宇宙名, 是否zip)
SOURCES = [
    ("zuoshu_setting", "昨书", False),
    ("zuoshu_book2_setting", "昨书", False),
    ("zuoshu_book3_setting", "昨书", False),
    ("cyberpunk", "赛博血缘", False),
    ("cyberpunk-expansion", "赛博血缘", False),
    ("prequel", "赛博血缘", False),
    ("chutan_setting", "初探", False),
    ("dingce_setting", "持灯录", False),
    ("dingce_prequel", "持灯录", False),
    ("jiukuaijiu_setting", "守则宇宙", False),
    ("yeban_setting", "守则宇宙", False),
    ("jia_setting", "守则宇宙", False),
    ("yinyangyu-prequel", "阴阳鱼", False),
]
# zip 用 glob 真实路径（构造的中文路径在 Windows 上 isfile 会判 False）
ZIP_UNIVERSE_MAP = [
    ("九块九守则", "守则宇宙"),
    ("出摊守则", "守则宇宙"),
    ("夜班守则", "守则宇宙"),
    ("末班守则", "守则宇宙"),
    ("家守则", "守则宇宙"),
]

SKIP_FILES = {"README.md", "writing-guide.md", "review_report.md", "tmp_root_audit_20261003.md"}
CHAR_FILE_RE = re.compile(r'characters?.*\.md$', re.I)
WORLD_FILE_RE = re.compile(r'setting.*world|setting-01|\.md$', re.I)
OUTLINE_FILE_RE = re.compile(r'outline', re.I)
RULES_FILE_RE = re.compile(r'rules-handbook', re.I)

def read_text(p):
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def parse_sections(text):
    lines = text.splitlines()
    secs, cur_h, cur_b = [], None, []
    for ln in lines:
        m = re.match(r'^#{1,4}\s+(.*)', ln)
        if m:
            if cur_h is not None or cur_b:
                secs.append((cur_h, "\n".join(cur_b).strip()))
            cur_h, cur_b = m.group(1).strip(), []
        else:
            cur_b.append(ln)
    if cur_h is not None or cur_b:
        secs.append((cur_h, "\n".join(cur_b).strip()))
    return secs

NAME_BLACKLIST = {'表','总','是','谁','这','那','一','二','三','上','下','中','注','备注','说明','概述','简介','介绍','目录','索引','总览','小结','设定','档案'}
def extract_name(heading, body):
    for src in (heading, body[:200]):
        m = re.search(r'【([^】]{1,14})】', src)
        if m:
            return m.group(1)
    # 无括号时，仅当标题显式是角色头才取，且过滤黑名单/虚词
    h = re.sub(r'[【】()（）\-—].*$', '', heading)
    h = re.sub(r'^(书[0-9一二三四五]?\s*)?(锚点主角|主角|角色|配角|人物|反?派|对手|班底|后宫)?\s*[:：]?', '', h).strip()
    if len(h) >= 2 and h not in NAME_BLACKLIST and not re.search(r'[？?。，、0-9]', h):
        return h[:8]
    return None

CONCEPT_BLACKLIST = {'是什么','谁执行','为什么','怎么','如何','定义','说明','概述','简介','介绍','总览','小结','注','备注','索引','目录','参见'}
def clean_concept(heading):
    h = re.sub(r'^§?\s*\d+(\.\d+)*\.?\s*', '', heading)  # 去 §4.3.1
    h = re.sub(r'(是什么|是谁|怎么|为什么|如何|吗|\？|？).*$', '', h)  # 去尾问语
    h = h.strip()
    return h[:18] if h else None

def truncate(s, max_chars=500):
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "…（设定节选）"
    return s

def gen():
    rows = []
    tmp = tempfile.mkdtemp(prefix="zb_unzip_")
    # 用 glob 真实路径发现 zip（构造的中文路径 isfile 在 Win 上会误判）
    sources = list(SOURCES)
    for zp in glob.glob(os.path.join(NOVEL, "*.zip")):
        bn = os.path.basename(zp)
        for key, uni in ZIP_UNIVERSE_MAP:
            if key in bn:
                sources.append((zp, uni, True))
                break
    for rel, universe, is_zip in sources:
        base = os.path.join(NOVEL, rel)
        files = []
        if is_zip:
            zpath = rel  # glob 已是完整路径
            if not os.path.isfile(zpath):
                print(f"[warn] zip 未找到: {rel}")
                continue
            zdir = os.path.join(tmp, os.path.splitext(os.path.basename(rel))[0])
            try:
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(zdir)
            except Exception as e:
                print(f"[warn] 解压失败 {rel}: {e}")
                continue
            for root, _, fs in os.walk(zdir):
                for f in fs:
                    if f.endswith(".md"):
                        files.append(os.path.join(root, f))
        else:
            base = os.path.join(NOVEL, rel)
            if not os.path.isdir(base):
                print(f"[warn] 目录不存在: {rel}")
                continue
            for f in os.listdir(base):
                if f.endswith(".md"):
                    files.append(os.path.join(base, f))

        char_n = 0
        concept_n = 0
        for fp in files:
            fn = os.path.basename(fp)
            if fn in SKIP_FILES:
                continue
            text = read_text(fp)
            if not text.strip():
                continue
            secs = parse_sections(text)
            is_char_file = bool(CHAR_FILE_RE.search(fn))
            is_world = bool(re.search(r'setting.*world|setting-01', fn, re.I))
            is_rules = bool(RULES_FILE_RE.search(fn))
            is_outline = bool(OUTLINE_FILE_RE.search(fn))

            if is_outline:
                out = truncate(text)
                if len(out) > 40:
                    rows.append(mk(f"请依据设定原话，概述《{universe}》的分卷大纲或故事结构。", out, "outline", universe, fn))
                continue
            if is_rules:
                out = truncate(text)
                if len(out) > 40:
                    rows.append(mk(f"请依据设定原话，列举《{universe}》的创作规则与铁律。", out, "rules", universe, fn))
                continue

            for h, b in secs:
                if not b or len(b) < 25:
                    continue
                if is_char_file or (h and re.search(r'角色|主角|人物|配角|锚点|反?派|对手|班底|后宫|【', h or "")):
                    name = extract_name(h or "", b)
                    if name and name not in ('角色一览','角色总表','人物一览','一览','列表','总表','索引','目录') and char_n < 200:
                        rows.append(mk(f"你是小说设定助手。请依据设定原话，介绍《{universe}》中的角色【{name}】，包括身份、阵营、能力与关键设定（如设定中提及他与其他角色的关系，也请说明）。", truncate(b), "char", universe, fn))
                        char_n += 1
                elif is_world and h:
                    concept = clean_concept(h)
                    if concept and 2 <= len(concept) <= 18 and concept not in CONCEPT_BLACKLIST and not re.search(r'[？?。，、0-9是谁怎么为什么如何吗]', concept) and concept_n < 200:
                        rows.append(mk(f"请依据设定原话，解释《{universe}》中的概念【{concept}】。", truncate(b), "concept", universe, fn))
                        concept_n += 1
    return rows

def mk(instruction, output, _type, universe, src):
    key = instruction[:60] + "|" + output[:40]
    return {"instruction": instruction, "output": output, "_type": _type, "_universe": universe, "_src": src, "_key": key}

# ---- 昨书成对关系（手写准确版，补训练集缺口）----
ZUOSHU_REL = [
 ("沈砚", "裴决", "沈砚是留痕者／破壁人（书1《昨书》主角，拼出裂缝）；裴决是序守会成员／守壁人（书2主角，抚平扰动）。两人是反向视角：一个拼出裂缝、一个抚平扰动，构成猫鼠张力的隐性对照。裴决在调查局层级上处于沈砚之上；书2时间早于书1，裴决可作为书1的“暗线猎手”——一个沉默的、按教条dispatch抹除扰动的人。两书主角不须正面合作，也不须敌对到底，因视角相反构成隐性对照。"),
 ("裴决", "陆启", "裴决护的正是书3主角陆启。陆启承载的科技时间线已焊入既定之网，故裴决的“巡定”工作具象为：护住这个人的时间线不被鼠阵扰乱——挡掉明枪暗箭、按制度dispatch掉所有紊乱行动。护而不亲：裴决保的是“既定之线”，不是这个人本身；陆启一无所知也毫不相干，且不构成“被保护而不自知的支线”。"),
 ("沈砚", "陆启", "陆启是书3中立轴，书1/书2共同绕之旋转的攻守轴心。沈砚（书1，AI已普及·最晚）与陆启（书3，AI刚起步·最早）不在同一时间层；陆启推出来的科技线已被焊入既定，沈砚活在既定普及之后。两人不直接同台，但共享同一绝对时间公理与同一当代。"),
 ("沈砚", "苏黎", "苏黎是书1配角。与沈砚同属书1世界；苏黎的弧线涉及是否最终走向“全抹除”（强悲剧弧）的待定设定——留痕者苏黎可能选择彻底抹除自身痕迹，作为续作引线或强悲剧收束。"),
 ("沈砚", "林晚", "林晚是书1配角，与沈砚的关系性质（友／恋）待作者拍板，尚未定稿。"),
 ("陆启", "后宫团", "陆启的后宫团六人（苏晚晴运营枢纽／沈知微算法／林屿资本／顾岚中文生态／江晚英文生态合规／钟衡军工国防）各映射复合体一个层面，领域互不重叠，天然无内斗，全员认陆启为apex。无夺权戏、无被保护戏；能力必须是他们各自日常工作中最有用之物。"),
 ("裴决", "鼠阵", "鼠阵是穿越重生者阵营（猫鼠两阵之“鼠”），裴决（序守会／猫阵）的对手。裴决按制度dispatch抹除鼠阵造成的扰动；但鼠阵的扰动在裴决的窥定中呈盲点／噪点——留痕者持有pre-lock记忆、不完全在既定网内，其扰动失真，正是书1/书2交叉的猫鼠空间。"),
 ("沈砚", "留痕者阵营", "沈砚是留痕者——保留pre-lock记忆、能看见裂缝。鼠阵也是留痕者型（有重生记忆）。沈砚拼出裂缝，鼠阵利用裂缝；两者同属“看见既定之外”的人，但与守序阵营（猫阵）对立。"),
 ("序守会", "鼠阵", "猫阵＝守序阵营（序守会／守序者，维护既定不可改）；鼠阵＝穿越重生者阵营（试图改写既定）。两阵对立，构成《昨书》核心猫鼠张力。守序者“看见了也不能改”，重生者“知道是死路还要往前走”。"),
 ("三书主角", "时间轴", "书3（陆启·AI刚起步·最早＝他推算法的起点）→ 书2（裴决·AI走了一段·中间）→ 书1（沈砚·AI已普及·最晚）。三书共享同一绝对时间公理、同一势力结构、同一个当代，仅视角不同；可以借结构，不能借调子。"),
 ("陆启", "沈砚与裴决的对位", "书3陆启的“爽”与书1沈砚的“看见裂缝却改不了”、书2裴决的“看见却不能改”形成对位；但陆启不背负其他书的苦，不被既定所压，可以一路赢到底。不要把最苦的调子塞进爽文里。"),
 ("裴决", "窥定能力", "裴决的“窥定”不是看得远，是调解得好：一眼看出纠纷会不会升级、知道对方最在意什么，劝得下去，调解成功率全区最高；但他不能说，只能装作不知道，一步一步把人劝到那个已写好的结局。限制：被动侵入（vision不请自来）、范围有限（只及锁死波前附近）、对留痕者造成的裂缝呈盲点。"),
 ("沈砚", "破壁人内核", "沈砚是调查者／破壁人，留痕者（保留旧记忆，能看见裂缝）。说话方式克制、观察型，习惯把异常记下来而非说出来（禁说铁律的内化）。当前基调＝悬疑／日常恐怖：徒劳救人、看见裂缝却改不了结局。"),
]

def main():
    rows = gen()
    # 去重
    uniq = {}
    for r in rows:
        k = r.pop("_key")
        if k not in uniq:
            uniq[k] = r
    base_rows = list(uniq.values())
    # 加昨书成对关系
    rel_rows = []
    for a, b, out in ZUOSHU_REL:
        rel_rows.append({"instruction": f"《昨书》中，角色【{a}】和【{b}】是什么关系？请依据设定原话概括。",
                         "output": out, "_type": "rel", "_universe": "昨书", "_src": "curated"})
    all_rows = base_rows + rel_rows
    # 统计宇宙分布
    from collections import Counter
    dist = Counter(r["_universe"] for r in all_rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] 总样本: {len(all_rows)}")
    print("[分布] 按宇宙:", dict(dist))
    print("[类型]", Counter(r["_type"] for r in all_rows))
    # 抽样
    for r in rel_rows[:3]:
        print("  REL>", r["instruction"][:40], "=>", r["output"][:40])

if __name__ == "__main__":
    main()
