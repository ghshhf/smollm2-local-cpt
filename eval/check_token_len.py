"""
数据集 token 长度体检（防静默截断）
====================================
为什么必须做：train_lora.py 的 max_seq_length=1024。超长样本会被**无声截断**，
如果被切掉的正好是【答案】区，模型学不到输出，而训练日志不会有任何报错 —— 典型静默质量事故。

本脚本只用 tokenizer（不加载 3GB 模型权重），内存开销极小，可与训练并行跑。

用法：
  <venv>/Scripts/python.exe eval/check_token_len.py --data data/role_data_v8.jsonl
  <venv>/Scripts/python.exe eval/check_token_len.py --data data/role_data_v8.jsonl --limit 1024
"""
import argparse
import json
import os

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")


def build_text(ex: dict) -> str:
    """与 train_lora.py build_text 保持完全一致。"""
    instr = ex.get("instruction", "").strip()
    return f"下面是一项任务。\n{instr}\n输出：{ex.get('output','').strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "role_data_v8.jsonl"))
    ap.add_argument("--limit", type=int, default=1024, help="训练器的 max_seq_length")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]

    lens, answer_cut = [], []
    for i, r in enumerate(rows):
        t = build_text(r)
        n = len(tok(t)["input_ids"])
        lens.append(n)
        if n > args.limit:
            # 答案区是否落在被截断之外
            full = t
            ans_pos = full.find("【答案】")
            answer_cut.append((i, n, "答案区被切" if ans_pos > 0 else "无答案区"))

    lens.sort()
    n = len(lens)

    def pct(p):
        return lens[min(int(n * p), n - 1)]

    over = [x for x in lens if x > args.limit]
    print(f"数据集: {args.data}")
    print(f"  样本数 {n}｜max_seq_length={args.limit}")
    print(f"  min={lens[0]}  median={pct(.5)}  p90={pct(.9)}  p99={pct(.99)}  max={lens[-1]}")
    print(f"  超过上限的样本: {len(over)} ({len(over)/n:.2%})")
    if answer_cut:
        print(f"  ⚠️ 其中答案区会被切掉的: {len(answer_cut)} 条，示例(idx, len):")
        for x in answer_cut[:10]:
            print(f"     {x}")
    else:
        print("  ✓ 无样本会因截断丢掉答案区")


if __name__ == "__main__":
    main()
