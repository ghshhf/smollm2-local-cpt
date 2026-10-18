# -*- coding: utf-8 -*-
"""
r6_final + LoRA(chat) 聊天推理
用法:
  python tools/chat_infer.py --ckpt out_lora_chat_v1/checkpoint-63 --msg "你好"
  python tools/chat_infer.py --ckpt out_lora_chat_v1/checkpoint-63 --interactive
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = r"D:\work\_local_ai\train_15b\backups\0.2B\r6_final_20261016"


def build_prompt(history, msg):
    """ChatML 多轮拼接。history=[(user, bot), ...]"""
    parts = []
    for u, b in history:
        parts.append("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>\n" % (u, b))
    parts.append("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % msg)
    return "".join(parts)


def chat(model, tok, history, msg, max_new=160, temp=0.7):
    prompt = build_prompt(history, msg)
    ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            top_p=0.9 if temp > 0 else None,
            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    # 截到第一个 im_end
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text[: text.index(stop)]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\_local_ai\train_15b\out_lora_chat_v1\checkpoint-63")
    ap.add_argument("--msg", default=None, help="单次提问")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    print("[load] base:", BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    print("[load] LoRA:", args.ckpt)
    model = PeftModel.from_pretrained(model, args.ckpt)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    print("[ready] 参数量:", sum(p.numel() for p in model.parameters()) // 1000, "K")

    history = []
    if args.msg:
        ans = chat(model, tok, history, args.msg, args.max_new, args.temp)
        print("\n[用户] %s\n[模型] %s" % (args.msg, ans))
    elif args.interactive:
        print("输入问题，空行退出：")
        while True:
            try:
                msg = input("\n[用户] ").strip()
            except EOFError:
                break
            if not msg:
                break
            ans = chat(model, tok, history, msg, args.max_new, args.temp)
            print("[模型] %s" % ans)
            history.append((msg, ans))
            if len(history) > 6:
                history = history[-6:]
    else:
        # 默认自测三连
        tests = [
            "你好，请你介绍一下你自己。",
            "用 Python 写一个判断质数的函数。",
            "1 加 1 等于几？",
        ]
        for t in tests:
            ans = chat(model, tok, history, t, args.max_new, args.temp)
            print("\n[用户] %s\n[模型] %s" % (t, ans))
            history.append((t, ans))


if __name__ == "__main__":
    main()
