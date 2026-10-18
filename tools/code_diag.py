"""编程语料体检：判断这份数据能不能直接喂 SFT，以及配比是否合理。

用法:
  python tools/code_diag.py --data data/corpus_code_v1.jsonl
  python tools/code_diag.py --data data/corpus_code_v1.jsonl --model backups/0.2B/r6_final_20261016   # 加 token 口径
"""
import argparse
import collections
import json
import re
import statistics


def lang_of(text):
    m = re.search(r"```(\w+)", text)
    return (m.group(1).lower() if m else "无代码块")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="传主干路径则统计 token/汉字")
    a = ap.parse_args()

    rows = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    keys = collections.Counter()
    for r in rows:
        keys.update(r.keys())
    print(f"[结构] 条数 = {len(rows)}  字段 = {dict(keys)}")

    ins = [r.get("instruction", "").strip() for r in rows]
    outs = [r.get("output", "").strip() for r in rows]
    has_inp = sum(1 for r in rows if r.get("input"))

    print(f"[结构] instruction 唯一 = {len(set(ins))}  output 唯一 = {len(set(outs))}")
    print(f"[结构] 带 input 字段 = {has_inp} 条")
    byi = collections.defaultdict(set)
    for i, o in zip(ins, outs):
        byi[i].add(o)
    conf = {k: v for k, v in byi.items() if len(v) > 1}
    print(f"[结构] 同问多答组 = {len(conf)}  冗余条数 = {sum(len(v)-1 for v in byi.values())}")
    for k, v in list(conf.items())[:3]:
        print(f"   冲突: {k[:36]} -> {len(v)} 种答案")

    empty_o = sum(1 for o in outs if len(o) < 4)
    no_code = sum(1 for o in outs if "```" not in o)
    print(f"[质量] 空/极短 output = {empty_o}  不含代码块 = {no_code} ({no_code/len(rows)*100:.1f}%)")

    ol = sorted(len(o) for o in outs)
    il = sorted(len(i) for i in ins)
    print(f"[长度] instruction 中位={il[len(il)//2]} 最大={il[-1]}")
    print(f"[长度] output 中位={ol[len(ol)//2]} 均值={sum(ol)//len(ol)} 最大={ol[-1]}")

    langs = collections.Counter(lang_of(o) for o in outs)
    print("[语言] " + "  ".join(f"{k}:{v}" for k, v in langs.most_common(8)))

    # 题型粗分
    cat = collections.Counter()
    for i in ins:
        if any(k in i for k in ("如何", "怎么", "怎样")):
            cat["怎么做"] += 1
        elif any(k in i for k in ("是什么", "什么是", "区别", "为什么", "原理")):
            cat["概念/原理"] += 1
        elif any(k in i for k in ("实现", "写一个", "写个", "函数", "代码")):
            cat["写代码"] += 1
        elif any(k in i for k in ("报错", "错误", "失败", "异常", "bug")):
            cat["排错"] += 1
        else:
            cat["其他"] += 1
    print("[题型] " + "  ".join(f"{k}:{v}({v/len(rows)*100:.1f}%)" for k, v in cat.most_common()))

    print("[抽样 3 条]")
    for r in rows[:3]:
        print("  Q:", r.get("instruction", "")[:60])
        print("  A:", r.get("output", "").replace("\n", " ")[:80])

    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        han = tok_total = 0
        lens = []
        for i, o in zip(ins, outs):
            n = len(tok(o, add_special_tokens=False).input_ids)
            tok_total += n
            lens.append(n)
            han += sum(1 for ch in o if "\u4e00" <= ch <= "\u9fff")
        print(f"[token] output 总 token = {tok_total:,}  汉字 = {han:,}  token/汉字 = {tok_total/max(1,han):.2f}")
        print(f"[token] output 逐条中位 = {statistics.median(lens):.0f}  >512 的条数 = {sum(1 for x in lens if x > 512)}  >1024 = {sum(1 for x in lens if x > 1024)}")


if __name__ == "__main__":
    main()
