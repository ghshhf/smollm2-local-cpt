# -*- coding: utf-8 -*-
"""诊断: 当前 0.2B checkpoint 对中文/英文 prompt 的下一步 logits 分布"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT = r"D:\work\_local_ai\train_15b\out_smollm2_200m_pt\checkpoint-latest\pytorch_model.bin"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd, strict=False)
model = model.to(torch.float32).eval()

tests = [
    ("中文小说", "夜色像潮水一样漫过城市的边缘，她把那封没有署名的信"),
    ("中文语感", "春天来了，"),
    ("英文知识", "The capital of France is"),
]
for tag, prompt in tests:
    ids = tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    topk = torch.topk(logits, 8)
    print(f"\n[{tag}] prompt={prompt!r}")
    for sc, idx in zip(topk.values, topk.indices):
        print(f"  {sc.item():7.3f}  id={idx.item():6d}  -> {tok.decode([idx.item()])!r}")
