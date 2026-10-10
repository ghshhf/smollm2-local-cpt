"""
按 token 长度分流：保证训练集不触发静默截断
=============================================
背景：train_lora.py 的 max_seq_length=1024。超长样本会被无声截断，
若切掉的恰好是【答案】区，模型会学到「输出一半就停」且无任何报错。

做法：把超长样本从主训练集分流到 *_long.jsonl 单独存放，
主训练集 100% 保证「答案区完整」。长样本留待将来用更大的 max_seq_length 单独训。

用法（必须用 venv python，需要 transformers）：
  <venv>/Scripts/python.exe filter_len.py --data data/role_data_v8.jsonl
  <venv>/Scripts/python.exe filter_len.py --data data/role_data_v8.jsonl --long data/role_data_v8_long.jsonl
"""
import argparse
import json
import os

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")


def build_text(ex: dict) -> str:
    return f"下面是一项任务。\n{ex.get('instruction','').strip()}\n输出：{ex.get('output','').strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--long", default=None, help="超长样本存放处；默认 <data> 同名加 _long")
    ap.add_argument("--limit", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]  # 先全读入，再安全覆写
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    keep, long_rows = [], []
    for r in rows:
        if len(tok(build_text(r))["input_ids"]) <= args.limit:
            keep.append(r)
        else:
            long_rows.append(r)

    long_path = args.long or args.data.replace(".jsonl", "_long.jsonl")
    with open(args.data, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(long_path, "w", encoding="utf-8") as f:
        for r in long_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"主训练集(安全) -> {args.data}: {len(keep)} 条")
    print(f"超长分流       -> {long_path}: {len(long_rows)} 条")
    print(f"  (全部 ≤ {args.limit} token，答案区完整)")


if __name__ == "__main__":
    main()
