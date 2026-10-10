# -*- coding: utf-8 -*-
"""
expand_300m.py  v2 — 保函数扩容 (function-preserving)

v1 教训 (2026-10-10): 改隐藏维/GQA头数会打碎已压缩记忆 (见 expand_to_200m.py v2 注释)。

v2 原则: 不改 注意力/隐藏维/头数/RoPE。
  ① MLP intermediate 3072 -> 4608, 新增行/列 = 0 (死神经元, 函数严格保持)。
  ② 层数 30 -> 36: 新增 6 层 o_proj/down_proj 输出全 0 = 恒等层 (函数严格保持,
     梯度会逐步激活)。
  参数量: 214M + MLP加宽80M + 6层恒等(~57M) ≈ 350M (0.3B 档)。

源: out_smollm2_200m_pt/checkpoint-latest/pytorch_model.bin (R5 训完)
目标: D:/work/_local_ai/models/SmolLM2-300M
校验: 扩容前后同 prompt logits max|diff| 必须≈0, 否则不落盘。
"""
import json, shutil, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

SRC_CKPT = "D:/work/_local_ai/train_15b/out_smollm2_200m_pt/checkpoint-latest/pytorch_model.bin"
SRC_DIR = "D:/work/_local_ai/models/SmolLM2-200M"
DST_DIR = "D:/work/_local_ai/models/SmolLM2-300M"
NEW_INTER = 4608
OLD_LAYERS = 30
NEW_LAYERS = 36

def main():
    sd = torch.load(SRC_CKPT, map_location="cpu")
    sd = {k: v.float() for k, v in sd.items()}

    old_cfg = json.load(open(f"{SRC_DIR}/config.json"))
    for k in ("torch_dtype", "dtype"):
        old_cfg.pop(k, None)
    old_cfg["intermediate_size"] = NEW_INTER
    old_cfg["num_hidden_layers"] = NEW_LAYERS
    old_cfg["torch_dtype"] = "float32"
    cfg = LlamaConfig(**{k: v for k, v in old_cfg.items()
                         if k not in ("architectures", "transformers_version",
                                      "is_llama_config", "rope_interleaved",
                                      "rope_scaling")})
    model = AutoModelForCausalLM.from_config(cfg)
    model = model.float()

    new_sd = {}
    for k, v in sd.items():
        if "gate_proj" in k or "up_proj" in k:
            w = torch.zeros(NEW_INTER, v.shape[1]); w[:v.shape[0]] = v
            new_sd[k] = w
        elif "down_proj" in k:
            w = torch.zeros(v.shape[0], NEW_INTER); w[:, :v.shape[1]] = v
            new_sd[k] = w
        else:
            new_sd[k] = v.clone()

    # 新增 6 层: 全零注意力输出 + 全零 MLP 输出 = 恒等
    for i in range(OLD_LAYERS, NEW_LAYERS):
        p = f"model.layers.{i}."
        h = old_cfg["hidden_size"]; it = NEW_INTER
        new_sd[p+"self_attn.q_proj.weight"] = torch.zeros(h, h)
        new_sd[p+"self_attn.k_proj.weight"] = torch.zeros(*new_sd["model.layers.0.self_attn.k_proj.weight"].shape)
        new_sd[p+"self_attn.v_proj.weight"] = torch.zeros(*new_sd["model.layers.0.self_attn.v_proj.weight"].shape)
        new_sd[p+"self_attn.o_proj.weight"] = torch.zeros(h, h)
        new_sd[p+"mlp.gate_proj.weight"] = torch.zeros(it, h)
        new_sd[p+"mlp.up_proj.weight"] = torch.zeros(it, h)
        new_sd[p+"mlp.down_proj.weight"] = torch.zeros(h, it)
        new_sd[p+"input_layernorm.weight"] = torch.ones(h)
        new_sd[p+"post_attention_layernorm.weight"] = torch.ones(h)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    bad = [k for k in missing if "lm_head" not in k and "rotary" not in k]
    if bad or unexpected:
        print("FATAL keys  missing:", bad[:8], " unexpected:", list(unexpected)[:5])
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(SRC_DIR)
    src_model = AutoModelForCausalLM.from_pretrained(SRC_DIR).float()
    try:
        src_sd = torch.load(SRC_CKPT, map_location="cpu")
        src_model.load_state_dict({k: v.float() for k, v in src_sd.items()}, strict=False)
    except Exception:
        pass
    src_model.eval(); model.eval()
    prompt = "夜色像潮水一样漫过"
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lo = src_model(**ids).logits
        ln = model(**ids).logits
    diff = (lo - ln).abs().max().item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M  (目标 0.3B 档)")
    print(f"保函数校验: logits max|diff| = {diff:.6f}")
    if diff > 1e-3:
        print("FATAL: 函数未保持, 不落盘")
        sys.exit(1)

    shutil.rmtree(DST_DIR, ignore_errors=True)
    model.save_pretrained(DST_DIR, safe_serialization=True)
    tok.save_pretrained(DST_DIR)
    print(f"OK 已写入 {DST_DIR}")

if __name__ == "__main__":
    main()
