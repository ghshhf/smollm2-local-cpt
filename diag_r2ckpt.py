# -*- coding: utf-8 -*-
"""诊断: 直接加载 R2 训练后的 135M checkpoint 本身, 看是否连贯"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-135M"
CKPT = r"D:\work\_local_ai\train_15b\out_smollm2_135m_pt\checkpoint-latest\pytorch_model.bin"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
sd = torch.load(CKPT, map_location="cpu")
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"[load R2 ckpt] missing={len(miss)} unexpected={len(unexp)}")
model = model.to(torch.float32).eval()

prompts = ["夜色像潮水一样漫过城市的边缘，她把那封没有署名的信", "春天来了，"]
for prompt in prompts:
    gen_ids = tok.encode(prompt, return_tensors="pt")[0].tolist()
    with torch.no_grad():
        for _ in range(60):
            nxt = int(torch.argmax(model(torch.tensor([gen_ids])).logits[0, -1]))
            if nxt == tok.eos_token_id:
                break
            gen_ids.append(nxt)
    print("=" * 60)
    print("PROMPT:", prompt)
    print("-" * 60)
    print(tok.decode(gen_ids, skip_special_tokens=True))
    print()
