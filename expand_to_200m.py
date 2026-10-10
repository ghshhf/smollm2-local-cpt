# -*- coding: utf-8 -*-
"""
expand_to_200m.py  v2 — 保函数扩容 (function-preserving)

v1 教训 (2026-10-10):
  v1 把隐藏维 576->640 并改 GQA 头数 (9Q/3KV -> 10Q/5KV)。
  GQA 的 Q头->KV头 映射是按位置整除分配的, 头数一变, 全部 30 层的
  注意力配对被打乱, 已压缩的 R2 记忆函数被打碎 -> 生成乱码。

v2 原则: 扩容绝不改 注意力/隐藏维/头数/RoPE/层数。
  只加宽 MLP intermediate: 1536 -> 3072。
  新增 gate/up 行 = 0, 新增 down 列 = 0 (死神经元, 函数严格保持,
  梯度会逐步激活它们)。
  参数量: 134.5M + 30层*3*576*1536 ≈ 213M (0.2B 档)。

源: out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin (R2 训完, 中文连贯)
目标: D:/work/_local_ai/models/SmolLM2-200M
校验: 扩容前后同 prompt logits max|diff| 必须≈0, 否则报错退出。
"""
import json, shutil, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

SRC_CKPT = "D:/work/_local_ai/train_15b/out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin"
SRC_DIR = "D:/work/_local_ai/models/SmolLM2-135M"
DST_DIR = "D:/work/_local_ai/models/SmolLM2-200M"
NEW_INTER = 3072

def main():
    # 1) 源权重 (R2 训完的 135M, fp32)
    sd = torch.load(SRC_CKPT, map_location="cpu")
    sd = {k: v.float() for k, v in sd.items()}

    # 2) 新 config: 与 135M 完全一致, 仅 intermediate_size 变
    old_cfg = json.load(open(f"{SRC_DIR}/config.json"))
    old_inter = old_cfg["intermediate_size"]
    for k in ("torch_dtype", "dtype"):
        old_cfg.pop(k, None)
    old_cfg["intermediate_size"] = NEW_INTER
    old_cfg["torch_dtype"] = "float32"
    cfg = LlamaConfig(**{k: v for k, v in old_cfg.items()
                         if k not in ("architectures", "transformers_version",
                                      "is_llama_config", "rope_interleaved",
                                      "rope_scaling")})
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.float()

    # 3) 变换权重: MLP 加宽, 新增行/列 = 0; 其余原样复制
    new_sd = {}
    for k, v in sd.items():
        if "gate_proj" in k or "up_proj" in k:
            # [old_inter, hidden] -> [new_inter, hidden], 新行 0
            w = torch.zeros(NEW_INTER, v.shape[1])
            w[:v.shape[0]] = v
            new_sd[k] = w
        elif "down_proj" in k:
            # [hidden, old_inter] -> [hidden, new_inter], 新列 0
            w = torch.zeros(v.shape[0], NEW_INTER)
            w[:, :v.shape[1]] = v
            new_sd[k] = w
        else:
            new_sd[k] = v.clone()

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # tied 模型: lm_head 不在 sd 里属正常
    bad = [k for k in missing if "lm_head" not in k]
    if bad or unexpected:
        print("FATAL keys  missing:", bad[:5], " unexpected:", list(unexpected)[:5])
        sys.exit(1)

    # 4) 保函数校验: 同 prompt, 新旧模型 logits 必须一致
    tok = AutoTokenizer.from_pretrained(SRC_DIR)
    src_model = AutoModelForCausalLM.from_pretrained(SRC_DIR).float()
    try:
        src_sd = torch.load(SRC_CKPT, map_location="cpu")
        src_model.load_state_dict({k: v.float() for k, v in src_sd.items()}, strict=False)
    except Exception:
        pass  # 源目录已是 R2 权重时可忽略
    src_model.eval(); model.eval()
    prompt = "夜色像潮水一样漫过"
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lo = src_model(**ids).logits
        ln = model(**ids).logits
    diff = (lo - ln).abs().max().item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M  (目标 0.2B 档)")
    print(f"保函数校验: logits max|diff| = {diff:.6f}")
    if diff > 1e-3:
        print("FATAL: 函数未保持, 不落盘")
        sys.exit(1)

    # 5) 落盘
    shutil.rmtree(DST_DIR, ignore_errors=True)
    model.save_pretrained(DST_DIR, safe_serialization=True)
    tok.save_pretrained(DST_DIR)
    print(f"OK 已写入 {DST_DIR}")

if __name__ == "__main__":
    main()
