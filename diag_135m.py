# -*- coding: utf-8 -*-
"""对照: 扩张前 135M 底座 (R2小说训过) 对中文 prompt 的手动贪心续写"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
model = model.to(torch.float32).eval()
print("[info] 135M base, params=%.1fM" % (sum(p.numel() for p in model.parameters())/1e6))

prompts = [
    "夜色像潮水一样漫过城市的边缘，她把那封没有署名的信",
    "春天来了，",
]
for prompt in prompts:
    gen_ids = tok.encode(prompt, return_tensors="pt")[0].tolist()
    with torch.no_grad():
        for _ in range(60):
            t = torch.tensor([gen_ids])
            nxt = int(torch.argmax(model(t).logits[0, -1]))
            if nxt == tok.eos_token_id:
                break
            gen_ids.append(nxt)
    print("=" * 60)
    print("PROMPT:", prompt)
    print("-" * 60)
    print(tok.decode(gen_ids, skip_special_tokens=True))
    print()
