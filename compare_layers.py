# -*- coding: utf-8 -*-
"""逐层数值比对: 200M vs R2-135M 隐藏态是否一致 (函数是否保住)"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE135 = r"D:\work\_local_ai\models\SmolLM2-135M"
BASE200 = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT135 = r"D:\work\_local_ai\train_15b\out_smollm2_135m_pt\checkpoint-latest\pytorch_model.bin"
tok = AutoTokenizer.from_pretrained(BASE135, trust_remote_code=True)

m200 = AutoModelForCausalLM.from_pretrained(BASE200, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True).to(torch.float32).eval()
m135 = AutoModelForCausalLM.from_pretrained(BASE135, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
m135.load_state_dict(torch.load(CKPT135, map_location="cpu"), strict=False)
m135 = m135.to(torch.float32).eval()

ids = tok.encode("夜色像潮水一样漫过城市的边缘", return_tensors="pt")

def hidden_at(model, layer_idx):
    outs = model(ids, output_hidden_states=True)
    h = outs.hidden_states[layer_idx][0, -1]   # 最后 token 的隐藏态
    return h

for L in [0, 1, 5, 29]:
    h135 = hidden_at(m135, L)
    h200 = hidden_at(m200, L)
    # 200M 隐藏态前576维 应≈ 135M 隐藏态(576维)
    a = h135[:576]
    b = h200[:576]
    d = (a - b).abs().max().item()
    print(f"layer {L}: max|diff 前576维| = {d:.4f}   135M前6={a[:6].tolist()}   200M前6={b[:6].tolist()}")
