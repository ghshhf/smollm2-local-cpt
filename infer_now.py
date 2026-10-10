# -*- coding: utf-8 -*-
"""
直接对续训底座权重做推理测试（非 LoRA）。
加载: 基座 SmolLM2-200M 的 config+tokenizer, 权重覆盖为 checkpoint-latest/pytorch_model.bin
用法:
  venv\Scripts\python.exe infer_now.py
"""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT = r"D:\work\_local_ai\train_15b\out_smollm2_200m_pt\checkpoint-latest\pytorch_model.bin"

print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
sd = torch.load(CKPT, map_location="cpu")
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"[info] load_state_dict -> missing={len(miss)} unexpected={len(unexp)}")
if miss: print("   missing:", miss[:5])
if unexp: print("   unexpected:", unexp[:5])
model = model.to(torch.bfloat16).eval()
print(f"[info] model ready, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

# 测试集: 覆盖当前已训能力(小说续写/中文语感) 与 尚未训能力(知识/代码/推理) 做对照
PROMPTS = [
    ("小说续写", "夜色像潮水一样漫过城市的边缘，她把那封没有署名的信"),
    ("中文语感", "自从有了人工智能，普通人写文章这件事"),
    ("知识(未训)", "请简要说明光合作用的基本过程："),
    ("代码(未训)", "def quick_sort(arr):"),
    ("推理(未训)", "小明有3个苹果，又买了5个，分给朋友4个，还剩几个？"),
]

GEN = 120
for tag, prompt in PROMPTS:
    ids = tok.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=GEN, do_sample=False, repetition_penalty=1.15)
    dt = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    print("=" * 64)
    print(f"[{tag}]  耗时 {dt:.1f}s  prompt: {prompt}")
    print("-" * 64)
    print(text)
    print()
