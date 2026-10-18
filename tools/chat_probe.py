#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chat_probe.py — 直接问本地权重, 看对话/知识基线(不改任何权重)

用法:
  python tools/chat_probe.py --model backups/0.3B/polaris-03b_20261018
  python tools/chat_probe.py --model backups/0.2B/r6_final_20261016 --n 40

判读口径: 这些权重只做过续预训练、没有对话 SFT, 所以看三件事——
  1) 会不会顺着"答:"接、有没有对话意识
  2) 知识题是否落在地表(首都/常识)
  3) 中文输出是否成句、有没有碎词/英文混出
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("寒暄", "你好，你是谁？\n答："),
    ("自我介绍", "介绍一下你自己。\n答："),
    ("常识", "中国的首都是哪里？\n答："),
    ("知识", "什么是比特币？\n答："),
    ("续写", "人工智能的发展"),
    ("身份", "你的名字是"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    nparam = sum(p.numel() for p in llm.parameters())
    print(f"== {a.model}")
    print(f"   params={nparam/1e6:.1f}M  layers={llm.config.num_hidden_layers}  "
          f"hidden={llm.config.hidden_size}  inter={llm.config.intermediate_size}\n")

    for tag, p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        t0 = time.time()
        with torch.no_grad():
            out = llm.generate(ids, max_new_tokens=a.n, do_sample=False,
                               repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        new = out[0][ids.shape[1]:]
        txt = tok.decode(new, skip_special_tokens=True)
        tps = new.shape[0] / max(1e-6, dt)
        print(f"[{tag}] {p!r}")
        print(f"   -> {txt.strip()!r}")
        print(f"   ({new.shape[0]} tok / {dt:.1f}s = {tps:.1f} tok/s)\n")


if __name__ == "__main__":
    main()
