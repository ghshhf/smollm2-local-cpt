"""
训练后推理对比：加载基座 + LoRA 适配器，给定提示词输出续写。
用法：
  venv\Scripts\python.exe infer.py --base model/Qwen2.5-1.5B-Instruct --lora out_lora --prompt "用一句大白话解释什么是复利。"
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--lora", default=None, help="LoRA 适配器目录；不传则只用基座")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if args.lora:
        print(f"[info] 合并 LoRA: {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float32)
    out = pipe(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.1)
    generated = out[0]["generated_text"]
    print("\n========== 输出 ==========")
    print(generated)
    print("==========================")


if __name__ == "__main__":
    main()
