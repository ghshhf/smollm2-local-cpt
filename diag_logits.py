# -*- coding: utf-8 -*-
"""诊断: 直接看前向 logits 的 argmax, 与 model.generate 对照"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model = model.to(torch.float32).eval()

prompt = "The capital of France is"
ids = tok.encode(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(ids).logits[0, -1]
topk = torch.topk(logits, 5)
print("[top-5 next-token predictions]")
for sc, idx in zip(topk.values, topk.indices):
    print(f"  {sc.item():.3f}  id={idx.item()}  -> {tok.decode([idx.item()])!r}")

# 手动贪心 20 步
print("\n[manual greedy 20 steps]")
gen_ids = ids[0].tolist()
with torch.no_grad():
    for _ in range(20):
        t = torch.tensor([gen_ids])
        lg = model(t).logits[0, -1]
        nxt = int(torch.argmax(lg))
        gen_ids.append(nxt)
print(tok.decode(gen_ids, skip_special_tokens=True))
