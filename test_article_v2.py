# -*- coding: utf-8 -*-
"""
文章/成稿生成实测：基座 vs v2 LoRA 同题写作对比。
目标：验证"这轮喂的文章/设定"是否让模型在「同人宇宙内 prose 写作」上有提升——
不仅仅能答设定题，而是能写出贴合设定、用对角色名/规则、基调正确的连贯中文段落。

指标：
  - 生成两段同题 prose（base / v2），打印 + 落盘 data/article_test_v2.json
  - 轻量客观分：检测每段是否命中该题预期的"设定锚点词"（角色名/规则名/宇宙术语）
  - 读者（用户）主观判断文风与一致性
"""
import json, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

BASE = "model/Qwen2.5-1.5B-Instruct"
LORA = "out_lora_v2/checkpoint-320"

# (提示, 宇宙, 预期锚点词列表) —— 锚点词用于客观判断"有没有用上设定"
PROMPTS = [
    ("请以《昨书》书1主角沈砚的视角，写一段他第一次在墙面上看见“裂缝”时的内心独白。要求贴合“留痕者能看见既定之网、却改不了结局”的基调（悬疑／日常恐怖）。",
     "昨书", ["沈砚", "裂缝", "留痕", "既定", "破壁"]),
    ("请以《赛博血缘》为背景，写一段评分系统对一名“降带者”下达降带裁决的现场场景，体现“纸→账→信用→算法→协议”衰变链中算法时代的规则。",
     "赛博血缘", ["降带", "评分", "算法", "协议", "信用", "账", "网格"]),
    ("请以《持灯录》为背景，写一个里长在核对“黄册”与“实征册”时发现二者不符（飞洒／诡寄）的片段，体现丁册四册机制。",
     "持灯录", ["黄册", "实征册", "里长", "册", "丁册", "飞洒", "诡寄"]),
    ("请以《阴阳鱼》为背景，写一段“霸道派”修士与“温养派”修士在论道场上对峙的对话。",
     "阴阳鱼", ["霸道", "温养", "修士", "论道", "修行"]),
]

GEN = dict(max_new_tokens=260, do_sample=True, temperature=0.75, top_p=0.9, repetition_penalty=1.15)

def hit_score(text, anchors):
    return sum(1 for a in anchors if a in text), len(anchors)

print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print("[load] 独立基座 A (干净) ...", flush=True)
mA = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
base_pipe = pipeline("text-generation", model=mA, tokenizer=tok, torch_dtype=torch.float32)

print("[load] 独立基座 B + 合并 v2 LoRA ...", flush=True)
mB = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
pm = PeftModel.from_pretrained(mB, LORA)
mB = pm.merge_and_unload()
lora_pipe = pipeline("text-generation", model=mB, tokenizer=tok, torch_dtype=torch.float32)

results = []
for i, (prompt, uni, anchors) in enumerate(PROMPTS, 1):
    b = base_pipe(prompt, **GEN)[0]["generated_text"]
    b = b[len(prompt):]
    l = lora_pipe(prompt, **GEN)[0]["generated_text"]
    l = l[len(prompt):]
    bh, bt = hit_score(b, anchors)
    lh, lt = hit_score(l, anchors)
    print(f"\n================= 题{i} [{uni}] =================", flush=True)
    print(f"提示: {prompt}", flush=True)
    print(f"--- 干净基座 (锚点命中 {bh}/{bt}) ---", flush=True)
    print(b.strip(), flush=True)
    print(f"--- v2 LoRA (锚点命中 {lh}/{lt}) ---", flush=True)
    print(l.strip(), flush=True)
    print("================================================", flush=True)
    results.append({
        "idx": i, "universe": uni, "prompt": prompt, "anchors": anchors,
        "base_text": b.strip(), "base_hits": bh, "base_total": bt,
        "lora_text": l.strip(), "lora_hits": lh, "lora_total": lt,
    })

with open("data/article_test_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

tb = sum(r["base_hits"] for r in results); T = sum(r["base_total"] for r in results)
tl = sum(r["lora_hits"] for r in results)
print(f"\n[score] 锚点命中 基座 {tb}/{T}  vs  v2 LoRA {tl}/{T}", flush=True)
print("[done] article_test_v2 finished", flush=True)
