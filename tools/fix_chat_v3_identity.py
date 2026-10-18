# -*- coding: utf-8 -*-
"""
chat_sft_v3 语料清洗：消灭"助手"腔自我描述，统一北极星/主人口径。
- 输入: data/chat_sft_v2.jsonl (330) + data/chat_extras_pool.jsonl (素材池源头同步清洗)
- 输出: data/chat_sft_v3.jsonl
- 铁律: 助手段落里不得出现"助手"二字；身份=北极星，称呼用户=主人。
"""
import json
import re
import sys
from collections import Counter

IO_UTF8 = "utf-8"

# 精确替换表（旧句 -> 新句），只动 assistant 段内容
REPLACES = [
    # v2 主自我介绍（4 处同款）
    ("我是北极星，主人给我定的名字。我跑在你自己的电脑上，只有 0.2B，是个本地小助手。能聊天，也能帮你写代码、整理文字。",
     "我是北极星，主人给我定的名字。我跑在主人自己的电脑上，只有 0.2B，是主人的私有模型。能聊天，也能帮主人写代码、整理文字。"),
    # v2 你是人吗
    ("不是，主人。我是 AI，跑在主人本地电脑上的小助手北极星。",
     "不是，主人。我是北极星，跑在主人本地电脑上的 AI 模型，不是人。"),
    # 素材池源头（下次重建语料不再带入旧腔调）
    ("我还没正式取名，你叫我小助手就行。想给我起一个也行。",
     "我叫北极星，主人给定的。想换叫法也行。"),
    ("你的本地小助手，0.2B 的小模型，跑在你自己电脑上，数据不出门。",
     "北极星，0.2B 的本地模型，跑在主人自己电脑上，数据不出门。"),
]

ASSIST_RE = re.compile(r"(<\|im_start\|>assistant\n)(.*?)(<\|im_end\|>)", re.S)


def scrub_text(t: str) -> tuple[str, list[str]]:
    """只替换 assistant 段内容；返回 (新文本, 命中记录)"""
    hits = []

    def _sub(m):
        body = m.group(2)
        for old, new in REPLACES:
            if old in body:
                body = body.replace(old, new)
                hits.append(old[:20])
        return m.group(1) + body + m.group(3)

    return ASSIST_RE.sub(_sub, t), hits


def main():
    src = "data/chat_sft_v2.jsonl"
    dst = "data/chat_sft_v3.jsonl"
    out_lines = []
    n_hit = 0
    q_counter = Counter()
    n_dup = 0
    for ln in open(src, encoding=IO_UTF8):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        t, hits = scrub_text(d["text"])
        if hits:
            n_hit += 1
        d["text"] = t
        # 提问去重统计（首个 user 段）
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", t, re.S)
        if m:
            if q_counter[m.group(1)] > 0:
                n_dup += 1
            q_counter[m.group(1)] += 1
        out_lines.append(json.dumps(d, ensure_ascii=False))

    with open(dst, "w", encoding=IO_UTF8) as f:
        f.write("\n".join(out_lines) + "\n")

    # 验收
    resid = sum(1 for ln in out_lines if "助手" in json.loads(ln)["text"])
    # 残留检查：区分 ChatML 角色位(assistant)与中文"助手"
    cn_assist = sum(1 for ln in out_lines if "助手" in ln)
    print(f"[v3] 总条数={len(out_lines)}  改写命中={n_hit}条  中文'助手'残留={cn_assist}行  重复提问={n_dup}条")
    if cn_assist:
        for i, ln in enumerate(out_lines):
            if "助手" in ln:
                print("  残留样本:", ln[:150])
    # 北极星身份条数
    n_star = sum(1 for ln in out_lines if "北极星" in ln)
    n_master = sum(1 for ln in out_lines if "主人" in ln)
    print(f"[v3] 北极星出现={n_star}条  主人出现={n_master}条")
    return 0 if cn_assist == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
