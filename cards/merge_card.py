"""
知识卡 · 多卡合并器（用时才合）
================================
新模式（2026-10-08 用户裁定）：每张卡只训一个领域的小 LoRA，
**要几个能力就在用时把几张卡按比例合成**，产出一个独立模型目录，加载时无需 peft。

数学：W = W_base + Σ_i w_i * (alpha_i/r_i) * (B_i @ A_i)
权重和接近 1.0 最自然；叠加多个领域时常略小于 1（避免互相打架）。

用法：
  # 单卡
  python merge_card.py --adapters out_lora_reason --out model_reason
  # 多卡合成：写作全能模型（设定 + 文风）
  python merge_card.py --adapters out_lora_v3,out_lora_prose --weights 1,1 --out model_write
  # 全能：设定 + 文风 + 代码 + 推理
  python merge_card.py --adapters out_lora_v3,out_lora_prose,out_lora_code,out_lora_reason `
                       --weights 1,0.8,0.8,0.8 --out model_all
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_combine import apply_lora_deltas  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")


def parse_list(s: str, cast=float):
    return [cast(x) for x in s.replace("，", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="要合成的卡目录，逗号分隔")
    ap.add_argument("--weights", default=None, help="每张卡的权重，逗号分隔；默认全部 1.0")
    ap.add_argument("--adapter", default=None, help="(兼容旧用法) 单卡目录")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", required=True, help="输出独立模型目录")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()

    adapters = args.adapter.split(",") if args.adapter else parse_list(args.adapters, str)
    weights = parse_list(args.weights, float) if args.weights else [1.0] * len(adapters)
    if len(weights) != len(adapters):
        ap.error(f"卡数 {len(adapters)} 与权重数 {len(weights)} 不一致")

    for a in adapters:
        if not os.path.isdir(a):
            ap.error(f"卡目录不存在: {a}")

    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    print(f"[merge] 基座: {args.base}")
    print(f"[merge] 合成 {len(adapters)} 张卡: {', '.join(adapters)}")
    print(f"[merge] 权重: {weights}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt, trust_remote_code=True)

    rep = apply_lora_deltas(model, adapters, weights)

    if rep["modules_applied"] != rep["modules_total"]:
        print(f"[warn] 有 {rep['modules_total'] - rep['modules_applied']} 层未命中基座，"
              f"示例: {rep['skipped'][:3]}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"[merge] 已合成 -> {args.out}")
    print("        (可直接 AutoModelForCausalLM.from_pretrained 加载，无需 peft)")


if __name__ == "__main__":
    main()
