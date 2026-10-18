# -*- coding: utf-8 -*-
"""clean_corpus.py — 语料清洗(只剔 prompt 模板残留)。

原则(用户 2026-10-14 明确):
  **不动成人/色情内容**。模型往那个方向倒, 本质是"知识量太少、只会往唯一熟悉的分布退化",
  解法是灌更多知识(R5/R6 及后续), 不是删内容——删了只会让它更没东西可说。
  参照: 跟构建本地知识库一个道理, 靠扩充覆盖而非靠裁剪。

  真正需要清的是 **数据集指令文本被当正文灌进语料**(实测: 模型反复输出
  「你是一位小说写手。你是一位擅长「末日生存」小说的写手。第一人称」)。
  这是数据污染, 不是内容问题。

清洗粒度 = 行级(外科手术式), 不是整篇丢弃:
  模板行通常独立成行, 剔掉它即可保留正文, 避免把整篇文章扔掉损失知识。
  另做: 完全相同文本 sha1 去重(不损失知识, 只去重复 boilerplate)。
  剔除后剩余 <200 字的整篇丢弃(内容已被掏空)。

用法:
  python3 clean_corpus.py --scan  in.jsonl           # 只统计
  python3 clean_corpus.py in.jsonl out.jsonl         # 产出
"""
import argparse, hashlib, json, re, sys

# ---- 模板行特征(命中即剔该行) ----
LINE_PATTERNS = [
    r"你是一位(?:网络)?(?:小说)?写手",
    r"你是一个(?:网络)?(?:小说)?写手",
    r"你是一位擅长",
    r"你是一个擅长",
    r"作为(?:一位|一个)?(?:网络)?(?:小说)?写手",
    r"你现在(?:是|扮演|充当)",
    r"请扮演",
    r"^(?:写作要求|创作要求|文章要求|设定如下|背景设定|人物设定|故事梗概)\s*[:：]",
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>",
    r"^(?:Human|Assistant|User|AI|System)\s*:",
    r"^###\s*(?:指令|回答|提示|系统|正文)",
]
LINE_RE = [re.compile(p) for p in LINE_PATTERNS]

# 极短行且像指令片段(如单独一行"第一人称"/"第三人称")
SHORT_HINT = re.compile(r"^.{0,12}(?:人称|视角).{0,4}$")

# 只剔"被掏空"的篇: 门槛与训练器一致(<20 字训练器本来也会跳过)。
# 绝不用 200 字这类高门槛——R5/R6 是知识语料, 大量条目(卡片/问答/词条)本来就短,
# 用高门槛等于在删知识, 与"只剔模板残留、其余保留"的原则冲突。
MIN_KEEP = 20


def bad_line(l):
    s = l.strip()
    if not s:
        return False
    for r in LINE_RE:
        if r.search(s):
            return True
    if len(s) <= 16 and SHORT_HINT.match(s):
        return True
    return False


def norm_key(s):
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w一-鿿]", "", s)
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def clean_text(t):
    lines = t.split("\n")
    keep = [l for l in lines if not bad_line(l)]
    return "\n".join(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()

    if not a.scan and not a.out:
        print("need out file or --scan"); sys.exit(2)

    n_in = n_out = n_drop_short = n_dup = n_tmpl = 0
    seen = set()
    fout = open(a.out, "w", encoding="utf-8") if (a.out and not a.scan) else None

    with open(a.inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
                txt = obj.get("text", "")
            except Exception:
                txt = line
                obj = {"text": txt}
            orig_len = len(txt)
            txt2 = clean_text(txt)
            if len(txt2) != orig_len:
                n_tmpl += 1
            if len(txt2) < MIN_KEEP:
                n_drop_short += 1
                continue
            k = norm_key(txt2)
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            n_out += 1
            if fout:
                obj["text"] = txt2
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if fout:
        fout.close()

    print("FILE=%s" % a.inp)
    print("  原文数      = %d" % n_in)
    print("  含模板残留  = %d" % n_tmpl)
    print("  过短(<20字)弃 = %d" % n_drop_short)
    print("  重复弃      = %d" % n_dup)
    print("  保留        = %d" % n_out)


if __name__ == "__main__":
    main()
