# -*- coding: utf-8 -*-
"""用当前 checkpoint 出样张：检验这一轮预训练实际学到了什么。"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "out_smollm2_135m_pt/checkpoint-latest"
BASE = "D:/work/_local_ai/models/SmolLM2-135M"
PROMPTS = [
    "夜色降临，城市的灯光一盏盏亮起来，他站在桥头，",
    "春天的田野上，",
    "老师走进教室，对我们说：",
]

import os
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
sd = torch.load(os.path.join(CKPT, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(sd)
model.eval()

for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=120, do_sample=True, temperature=0.85,
            top_p=0.92, repetition_penalty=1.15, pad_token_id=tok.eos_token_id,
        )
    print("=" * 60)
    print("PROMPT:", p)
    print("OUT   :", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
