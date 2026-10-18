"""编程语料清洗：删掉模板垃圾答案，输出干净的 SFT 语料。

实测发现：2934 条里有 1390 条的 output 是同一个模板——

    思考步骤（启发式分析，非 LLM 推理）：
    1. 任务：{question}
    2. 语言/结构：python，识别到的模式——基础控制流+函数封装。
    3. 关键边界：空输入、重复/极值、资源清理（关闭/释放）...

这个模板：①不含任何真代码；②语言识别固定写死 python（问 JS/GitHub Actions 也是 python）；
③和同题的正确答案配对，构成"同问两答"，模型只能学成平均。必须删。

用法:
  python tools/clean_code.py --in data/corpus_code_v1.jsonl --out data/corpus_code_v2.jsonl
"""
import argparse
import collections
import json
import re

JUNK_MARKS = ["非 LLM 推理", "非LLM推理", "启发式分析", "思考步骤"]
JUNK_HEAD = re.compile(r"^\s*(思考步骤|推理步骤|分析步骤)")


def is_junk(o):
    head = o[:60]
    return bool(JUNK_HEAD.match(head)) or any(k in head for k in JUNK_MARKS)


def quality(o):
    """同一问题若有多个非垃圾答案，取信息量最高的一条。"""
    has_code = 1 if "```" in o else 0
    return (has_code, -abs(len(o) - 500))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    kept, junk, merged = [], 0, 0
    byi = collections.defaultdict(list)
    for r in rows:
        q = r["instruction"].strip()
        o = r["output"].strip()
        if is_junk(o):
            junk += 1
            continue
        byi[q].append(o)

    for q, outs in byi.items():
        if len(outs) > 1:
            merged += len(outs) - 1
        best = sorted(outs, key=quality)[-1]
        kept.append({"instruction": q, "output": best})

    print(f"输入 {len(rows)} 条")
    print(f"  删除·模板垃圾 = {junk} 条 ({junk/len(rows)*100:.1f}%)")
    print(f"  合并·同题多答 = {merged} 条")
    print(f"输出 {len(kept)} 条 → {a.out}")

    n_code = sum(1 for r in kept if "```" in r["output"])
    print(f"  其中含代码块 = {n_code} ({n_code/len(kept)*100:.1f}%)  纯解释 = {len(kept)-n_code}")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
