# -*- coding: utf-8 -*-
"""推理时能力实测：r6_final(0.2B) 裸底座的自然对话水平。
不训练、不挂卡、不采样技巧，greedy 生成，看它现在到底能回什么。
"""
import io, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CKPT = r"D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if not tok.chat_template:
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
model.eval()
print("[info] loaded %.1fM params in %.0fs" % (sum(p.numel() for p in model.parameters()) / 1e6, time.time() - t0))

QS = [
    "你好",
    "你是谁？",
    "今天天气怎么样？",
    "给我讲个笑话",
    "1+1等于几？",
    "帮我写一个Python函数，计算斐波那契数列第n项",
]
for q in QS:
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, return_tensors="pt")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, do_sample=False, repetition_penalty=1.15)
    ans = tok.decode(out[0], skip_special_tokens=True)[len(text):]
    print("=" * 66)
    print("[Q] %s  (%.0fs)" % (q, time.time() - t1))
    print("[A] %s" % ans.strip()[:300])
