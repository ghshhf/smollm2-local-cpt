# -*- coding: utf-8 -*-
"""node1_gen.py — 在 node1 上用当前权重做生成抽样(判水平用)。

注意: chain 的 checkpoint 目录只存模型(config+权重), tokenizer 常驻 NAS 的 base_tok/,
      所以 tokenizer 要单独指定。

用法: python3 node1_gen.py --base <ckpt_dir> --tok <tokenizer_dir> [--max_new 70]
输出: 纯文本, 由调用方管道上传到 ECS。
"""
import argparse, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("续写·文学", "夜色像潮水一样漫过"),
    ("常识·事实", "中国的首都是"),
    ("续写·故事", "从前有座山，山里有座庙，"),
    ("知识·解释", "人工智能是指"),
    ("日常·口语", "今天天气很好，我们"),
]


def _safe_load_tokenizer(base_dir):
    """规避 transformers 新版把 extra_special_tokens(list) 当 dict 的崩溃"""
    import json as _json, os
    tc = os.path.join(base_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        try:
            d = _json.load(open(tc, "r", encoding="utf-8"))
            if isinstance(d.get("extra_special_tokens"), list):
                d["extra_special_tokens"] = {}
                _json.dump(d, open(tc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass
    try:
        return AutoTokenizer.from_pretrained(base_dir)
    except Exception:
        return AutoTokenizer.from_pretrained(base_dir, use_fast=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/workspace/node1/chain/ckpt")
    ap.add_argument("--tok", default="/mnt/workspace/node1/chain/base_tok")
    ap.add_argument("--max_new", type=int, default=70)
    a = ap.parse_args()

    try:
        n_th = len(__import__("os").sched_getaffinity(0))
    except Exception:
        n_th = 8
    torch.set_num_threads(min(n_th, 8))

    print("BASE=%s" % a.base)
    print("TOK=%s" % a.tok)
    tok = _safe_load_tokenizer(a.tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print("PARAMS=%.1fM" % (n / 1e6))
    print("=" * 60)

    for tag, p in PROMPTS:
        ids = tok(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                                 repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id or 0)
        txt = tok.decode(out[0], skip_special_tokens=True)
        print("[%s] 提示: %s" % (tag, p))
        print("  输出: %s" % txt)
        print("-" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("GEN_ERR %r" % e)
        sys.exit(1)
