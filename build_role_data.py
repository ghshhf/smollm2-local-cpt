# -*- coding: utf-8 -*-
"""
把 novel/ 下各宇宙线 characters.md 转成 LoRA 训练集（instruction/output）。
目标：让 1.5B 模型“理解角色卡 / 角色关系”。
输出：data/role_data.jsonl
"""
import re, json, os, glob

ROOT = "D:/work/novel"
OUT = "D:/work/_local_ai/train_15b/data/role_data_clean.jsonl"

UNI = {
    "zuoshu_setting": "昨书",
    "cyberpunk": "赛博血缘",
    "chutan_setting": "出摊守则",
    "dingce_prequel": "持灯录·前传",
    "dingce_setting": "持灯录",
    "jia_setting": "家",
    "jiukuaijiu_setting": "九块九",
    "moban_setting": "模板",
    "yeban_setting": "夜班",
}
SKIP = {"主角", "配角", "盟友", "对抗面", "书局", "序守会", "关系", "书3",
        "二、", "三、", "一、", "异常事务调查局", "调查局", "组织", "非角色"}

def name_of(head):
    h = re.sub(r"^#{2,3}\s*", "", head).strip()
    h = re.sub(r"^\d+[\.\、]\s*", "", h)
    m = re.search(r"\[([^\]]{1,14})\]", h)
    if m:
        return m.group(1).strip()
    base = re.split(r"[（(]", h)[0].strip()
    m = re.search(r"\*\*(.+?)\*\*", base)
    if m:
        return m.group(1).strip()
    base = re.split(r"[:：]", base)[0].strip().strip(" -*")
    return base[:10] if base else None

def split_blocks(text):
    blocks, cur = [], None
    for ln in text.splitlines():
        if re.match(r"^#{2,3}\s", ln):
            if cur:
                blocks.append(cur)
            cur = {"head": ln, "body": []}
        elif cur is not None:
            cur["body"].append(ln)
    if cur:
        blocks.append(cur)
    return blocks

def clean(body_lines):
    # 去 markdown 表格/标题噪音，合成纯文本
    out = []
    for ln in body_lines:
        ln = ln.rstrip()
        if not ln.strip():
            continue
        if re.match(r"^\s*\|.*\|\s*$", ln):        # markdown 表格行
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cells and all(set(c) <= set("-: ") for c in cells):
                continue                            # 分隔行
            ln = ("：" if len(cells) == 2 else " ").join(cells)
        else:
            ln = re.sub(r"^\s*[-*]\s*", "• ", ln)   # 列表
            ln = re.sub(r"\*\*(.+?)\*\*", r"\1", ln)  # 去加粗
            ln = re.sub(r"`([^`]+)`", r"\1", ln)    # 去代码
        out.append(ln)
    return "\n".join(out).strip()

def main():
    rows = []
    files = glob.glob(os.path.join(ROOT, "*", "characters.md"))
    for f in files:
        uni_key = os.path.basename(os.path.dirname(f))
        uni = UNI.get(uni_key, uni_key)
        text = open(f, encoding="utf-8").read()
        blocks = split_blocks(text)
        for b in blocks:
            name = name_of(b["head"])
            body = clean(b["body"])
            if not name or name in SKIP:
                continue
            if name[0].isdigit():       # 子小节编号（如 “1.1 三样本事”）
                continue
            if len(name) > 6:           # 角色名通常 2~4 字，过长必是子标题
                continue
            if re.search(r"本事|事情|设定|锚点|字典|清单|真相|内核|冲突|能力|钩子|说明|原则|档案|总览|角色|小节|关系", name):
                continue
            if len(body) < 40:          # 分组标题/空块跳过
                continue
            if "非角色" in body or "非单个角色" in body:
                continue
            rows.append({
                "instruction": f"你是小说设定助手。请依据设定原话，介绍《{uni}》中的角色【{name}】，"
                               f"包括身份、阵营、能力与关键设定（如设定中提及他与其他角色的关系，也请说明）。",
                "output": body,
                "_type": "char",
            })
            # —— 关系问答（从含“关系”的句子抽）——
            rel_pat = re.search(r"与(.{1,8}?)关系", body)
            for sent in re.split(r"[。！？\n]", body):
                if "关系" in sent and len(sent) > 6:
                    m = rel_pat or re.search(r"与(.{1,8}?)关系", sent)
                    if m:
                        y = m.group(1).strip("的 、，的")
                        if 1 <= len(y) <= 8:
                            rows.append({
                                "instruction": f"《{uni}》中，角色【{name}】和【{y}】是什么关系？",
                                "output": sent.strip() + "。",
                                "_type": "rel",
                            })
                            break
        # 该宇宙线角色名单
        names = []
        for b in blocks:
            n = name_of(b["head"])
            if n and n not in SKIP and len(clean(b["body"])) >= 40 and "非角色" not in clean(b["body"]):
                names.append(n)
        if names:
            rows.append({
                "instruction": f"《{uni}》这条故事线里有哪些主要角色？请列出角色名并各用一句话点出其定位。",
                "output": "、".join(f"【{n}】" for n in names) + "。" +
                          "（各角色详细设定见对应角色卡："
                          + "；".join(f"{n}——见角色卡" for n in names[:6]) + "）",
                "_type": "list",
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["_type"] for r in rows)
    print(f"[ok] 总样本 {len(rows)}  -> {dict(c)}")
    print(f"[ok] 已写 {OUT}")
    for r in rows[:2]:
        print("---- 样例 ----")
        print("Q:", r["instruction"])
        print("A:", r["output"][:120], "...")

if __name__ == "__main__":
    main()
