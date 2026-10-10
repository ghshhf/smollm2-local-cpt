# -*- coding: utf-8 -*-
"""决定性对照: 200M(from_pretrained) vs R2-ckpt(载135M) 的下一步 top-5 预测"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE135 = r"D:\work\_local_ai\models\SmolLM2-135M"
BASE200 = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT135 = r"D:\work\_local_ai\train_15b\out_smollm2_135m_pt\checkpoint-latest\pytorch_model.bin"

tok = AutoTokenizer.from_pretrained(BASE135, trust_remote_code=True)

# 200M
m200 = AutoModelForCausalLM.from_pretrained(BASE200, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True).to(torch.float32).eval()
# R2 ckpt -> 135M
m135 = AutoModelForCausalLM.from_pretrained(BASE135, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
m135.load_state_dict(torch.load(CKPT135, map_location="cpu"), strict=False)
m135 = m135.to(torch.float32).eval()

prompt = "夜色像潮水一样漫过城市的边缘，她把那封没有署名的信"
ids = tok.encode(prompt, return_tensors="pt")

def top5(model, tag):
    with torch.no_grad():
        lg = model(ids).logits[0, -1]
    tk = torch.topk(lg, 5)
    print(f"[{tag}] top-5:", [tok.decode([i.item()]) for i in tk.indices])

top5(m135, "R2-135M(连贯基准)")
top5(m200, "扩展200M(待测)")
