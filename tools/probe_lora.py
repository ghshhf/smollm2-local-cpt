#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_lora.py — 测 0.3B 挂 LoRA 之后到底会不会吐代码

用途(2026-10-19): 用户在调试对话时看到"能吐代码", 需定位能力来自底座还是适配器。
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
    ("对话", "你好，你是谁？"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="backups/0.3B/polaris-03b_20261018")
    ap.add_argument("--adapter", default="models/chat_lora_v4_port03b")
    ap.add_argument("--tok", default="out_lora_chat_v7")
    ap.add_argument("--max-new", type=int, default=120)
    a = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, a.tok))
    m = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, a.base), torch_dtype=torch.float32)
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, os.path.join(ROOT, a.adapter))
    m.eval()
    print("[load] base=%s adapter=%s" % (a.base, a.adapter))

    for tag, p in PROMPTS:
        ids = tok(p, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n===== [%s] %s (%.0fs)" % (tag, p, time.time() - t1))
        print(txt)
        print("----- 含'def '=%s 含'```'=%s" % ("def " in txt, "```" in txt))


if __name__ == "__main__":
    main()
