# -*- coding: utf-8 -*-
"""对照测试: chat template 格式 vs 裸文本续写。"""
import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT = r"D:\work\_local_ai\train_15b\out_smollm2_200m_pt\checkpoint-latest\pytorch_model.bin"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
model = model.to(torch.bfloat16).eval()

QS = ["水加热到100度会发生什么？", "17 + 26 等于多少？直接给答案。", "你是谁？"]
for q in QS:
    text = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=90, do_sample=False, repetition_penalty=1.15)
    print("="*66)
    print("[chat]", q)
    print(tok.decode(out[0], skip_special_tokens=True)[len(text):])
