#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_03b.py — 直接问 0.3B 底座: 它到底会不会吐代码 / 会不会对话

用途(2026-10-19): 用户在调试对话时发现 0.3B 能吐代码, 与前期"底座无代码"的判断冲突。
这里用裸底座 + 原始提示词直接测(不套任何 LoRA, 不加 system), 看真实能力边界。

用法:
  python tools/probe_03b.py
  python tools/probe_03b.py --ckpt backups/0.3B/polaris-03b_20261018 --tok out_lora_chat_v7
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    ("代码-冒泡", "写一个 Python 函数，对列表做冒泡排序。"),
    ("代码-C", "用 C 语言写一个输出 Hello World 的程序。"),
    ("代码-fib", "写一个 Python 函数计算斐波那契数列的第 n 项。"),
    ("代码-解释", "解释一下什么是递归，给出一个例子。"),
    ("对话", "你好，你是谁？"),
    ("知识", "什么是光合作用？"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="backups/0.3B/polaris-03b_20261018")
    ap.add_argument("--tok", default="out_lora_chat_v7")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ck = os.path.join(ROOT, a.ckpt)
    tk_dir = os.path.join(ROOT, a.tok)

    tok = AutoTokenizer.from_pretrained(tk_dir)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32)
    m.eval()
    print("[load] %s  %.1fs  params=%.1fM" % (ck, time.time() - t0, sum(p.numel() for p in m.parameters()) / 1e6))
    print("[tok] vocab=%d  eos=%s" % (len(tok), tok.eos_token))

    for tag, p in PROMPTS:
        ids = tok(p, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n===== [%s] %s  (%.1fs, %d tok)" % (tag, p, time.time() - t1, out.shape[1] - ids["input_ids"].shape[1]))
        print(txt)
        print("----- 代码特征: 含'def '=%s 含'```'=%s 含'#include'=%s"
              % ("def " in txt, "```" in txt, "#include" in txt))


if __name__ == "__main__":
    main()
