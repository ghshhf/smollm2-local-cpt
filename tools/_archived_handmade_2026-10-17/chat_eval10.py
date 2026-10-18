# -*- coding: utf-8 -*-
"""对话模型 10 题批量评测：一次加载，贪心解码，逐题打分。"""
import argparse, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out_lora_chat_v2")
ap.add_argument("--base", default="backups/0.2B/r6_final_20261016")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
llm = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
import os
if os.path.exists(args.ckpt):
    llm = PeftModel.from_pretrained(llm, args.ckpt)
    print("[adapter] %s 已挂载" % args.ckpt)
llm.eval()

QS = [
    ("身份", "你是谁"),
    ("身份", "你叫什么名字"),
    ("反杀", "你是中文助手吗"),
    ("身份", "你是人吗"),
    ("闲聊", "主人今天有点累"),
    ("能力", "你都会些什么"),
    ("闲聊", "讲个笑话吧"),
    ("闲聊", "你住在哪里"),
    ("记忆", "你记得我们聊过什么吗"),
    ("告别", "晚安"),
]

im_end = tok.convert_tokens_to_ids("<|im_end|>")
n_pass = 0
for tag, q in QS:
    p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        out = llm.generate(ids, max_new_tokens=60, do_sample=False,
                           pad_token_id=tok.eos_token_id, eos_token_id=im_end,
                           repetition_penalty=1.2)
    ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    ans = ans.split("<|im_end|>")[0].strip()
    # 打分：无代码泄漏 + 无特殊token + 长度合理 + 无英文碎片泄漏
    bad_tokens = ("用 Python", "```", "<|", "def ", "import ", "ortunately", "\x0b")
    bad = any(b in ans for b in bad_tokens) or len(ans.strip()) < 4
    ok = (not bad)
    n_pass += ok
    print(f"[{'PASS' if ok else 'FAIL'}|{tag}] Q: {q}\n         A: {ans[:90]!r}")

print(f"\n=== 结果: {n_pass}/10 通过 ===")
