# -*- coding: utf-8 -*-
"""对照2: 基座 fp32 生成 (不转 bf16), 验证是否 bf16 导致 logits 坍缩"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model = model.to(torch.float32).eval()  # 强制 fp32
print("[info] BASE fp32, params=%.1fM" % (sum(p.numel() for p in model.parameters())/1e6))

prompts = [
    "The capital of France is",
    "夜色像潮水一样漫过城市的边缘，她把那封没有署名的信",
]
for prompt in prompts:
    ids = tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, do_sample=False, repetition_penalty=1.15)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("=" * 60)
    print("PROMPT:", prompt)
    print("-" * 60)
    print(text)
    print()
