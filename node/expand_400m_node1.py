# -*- coding: utf-8 -*-
"""
expand_400m_node1.py — node1 端保函数扩容 (function-preserving) 0.3B -> 0.4B

与 expand_300m_node1.py 完全同构, 只改一处策略:
  0.3B 现况: inter=4608, layers=36, hidden=576 => 346.9M
  0.4B 目标: ≈ 400M
  方案 = **只加宽 MLP**(宪法允许的两种操作之一), 不动层数:
      inter 4608 -> 5504 (=43x128, 对齐 128 便于 GPU GEMM)
      增量 = 36 层 x 3 矩阵 x 576 x (5504-4608) = 36 x 3 x 576 x 896 ≈ 55.8M
      => 346.9 + 55.8 ≈ 402.7M (0.4B 档)
  为什么不加层: 深度已 36 层, 0.4B 档优先补宽度;
      且加层需要新层恒等初始化, 收益不如加宽稳(见 0.2B->0.3B 的实测)。

保函数保证:
  - MLP 新增行/列全 0 => 死神经元, 前向输出严格不变
  - 不改注意力维/头数/RoPE/hidden/层数/词表
  - 落盘前做 logits max|diff| < 1e-3 校验, 不过则不落盘

用法: python3 expand_400m_node1.py <ckpt_dir>
"""
import json, sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

NEW_INTER = 5504  # 43*128


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/node1/chain/ckpt_gpu/checkpoint-latest"
    print(f"[expand400] src={src}")

    cfg_path = os.path.join(src, "config.json")
    old_cfg = json.load(open(cfg_path))
    OLD_INTER = int(old_cfg["intermediate_size"])
    OLD_LAYERS = int(old_cfg["num_hidden_layers"])
    print(f"[expand400] OLD inter={OLD_INTER} layers={OLD_LAYERS} -> NEW inter={NEW_INTER} layers={OLD_LAYERS}(不变)")

    print("[expand400] 加载原模型(读 safetensors)...")
    src_model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).float()
    src_model.eval()
    sd = {k: v.detach().cpu().float().clone() for k, v in src_model.state_dict().items()}

    for k in ("torch_dtype", "dtype"):
        old_cfg.pop(k, None)
    old_cfg["intermediate_size"] = NEW_INTER
    old_cfg["torch_dtype"] = "float32"
    cfg = LlamaConfig(**{k: v for k, v in old_cfg.items()
                         if k not in ("architectures", "transformers_version",
                                      "is_llama_config", "rope_interleaved",
                                      "rope_scaling")})
    model = AutoModelForCausalLM.from_config(cfg).float()
    model.eval()

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

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    bad = [k for k in missing if "lm_head" not in k and "rotary" not in k]
    if bad or unexpected:
        print("FATAL keys  missing:", bad[:8], " unexpected:", list(unexpected)[:5])
        sys.exit(1)

    TOK_SRC = src
    if not os.path.exists(os.path.join(src, "tokenizer.json")):
        for cand in ("/root/r4work/base_next", "/mnt/workspace/node1/chain/base_tok"):
            if os.path.exists(os.path.join(cand, "tokenizer.json")):
                TOK_SRC = cand
                break
    print(f"[expand400] tokenizer src={TOK_SRC}")
    tok = AutoTokenizer.from_pretrained(TOK_SRC)
    ids = tok("夜色像潮水一样漫过", return_tensors="pt")
    with torch.no_grad():
        lo = src_model(**ids).logits
        ln = model(**ids).logits
    diff = (lo - ln).abs().max().item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M  (目标 0.4B 档)")
    print(f"保函数校验: logits max|diff| = {diff:.6f}")
    if diff > 1e-3:
        print("FATAL: 函数未保持, 不落盘")
        sys.exit(1)

    model.save_pretrained(src, safe_serialization=True)
    binp = os.path.join(src, "pytorch_model.bin")
    if os.path.exists(binp):
        os.remove(binp)
        print("[expand400] 已删旧 pytorch_model.bin")
    print(f"OK 0.4B 已写入 {src} (inter={NEW_INTER} layers={OLD_LAYERS})")


if __name__ == "__main__":
    main()
