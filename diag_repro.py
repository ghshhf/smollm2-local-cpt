# -*- coding: utf-8 -*-
"""本机最小复现: 用本地 200M 基座 + 本地 R4 语料, 对比 fp32 与 bf16-autocast 的初始 loss。
目的: 定位 node1 训练脚本 loss 飙到 31 的根因。"""
import json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
CORPUS = r"D:\work\_local_ai\train_15b\pretrain_corpus\zh_text_r4.jsonl"
MAXLEN = 256
SEED = 12345

tok = AutoTokenizer.from_pretrained(BASE)
docs = []
with open(CORPUS, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3000:
            break
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line); t = o.get("text", "")
        except Exception:
            t = line
        if len(t) >= 20:
            docs.append(t)
arr = np.concatenate([np.array(tok.encode(d, add_special_tokens=False), dtype=np.int64) for d in docs])
print(f"[cache] {len(arr)} tokens from {len(docs)} docs")

rng = np.random.default_rng(SEED)
s = int(rng.integers(0, len(arr) - MAXLEN - 1))
b = torch.from_numpy(arr[s:s + MAXLEN + 1].astype(np.int64))
print(f"[slice] start={s} len={len(b)}")

for tag, use_bf16 in [("fp32", False), ("bf16-autocast", True)]:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model.eval()
    if use_bf16:
        model = model.to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
        with torch.no_grad():
            out = model(input_ids=b[:-1].unsqueeze(0), labels=b[1:].unsqueeze(0))
    print(f"[{tag}] initial loss = {out.loss.item():.4f}   ({time.time()-t0:.1f}s)")
