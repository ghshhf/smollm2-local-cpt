# -*- coding: utf-8 -*-
"""
expand_300m_node1.py — node1 端保函数扩容 (function-preserving) 0.2B -> 0.3B

原则(血泪, 见 expand_to_200m/300m v2):
  绝不改 注意力/隐藏维/头数/RoPE/层数以外的结构。
  ① MLP intermediate 3072 -> 4608: 新增行/列 = 0 (死神经元, 函数严格保持)。
  ② 层数 30 -> 36: 新增 6 层 o_proj/down_proj 输出全 0 = 恒等层 (函数严格保持)。
  参数量 ≈ 214M + 80M(MLP加宽) + 56M(6层恒等) ≈ 350M (0.3B 档)。

与本地 expand_300m.py 的差异(适配 node1):
  - 源/目标 = 同一目录 (W/ckpt), 原地扩容, 不需跨机搬权重。
  - 不 torch.load(.bin) —— node1 可能 torch<2.6(CVE-2025-32434 拒 pickle),
    改用 AutoModelForCausalLM.from_pretrained(.safetensors) 取 state_dict。
  - 落盘用 safetensors, 并删掉旧 pytorch_model.bin(防 trainer 读旧权重)。
  - 保函数校验: 同 prompt logits max|diff| 必须 < 1e-3, 否则不落盘、非零退出。

用法: python3 expand_300m_node1.py <ckpt_dir>
"""
import json, shutil, sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/node1/chain/ckpt"
    print(f"[expand] src={src}")

    # 1) 原 config(动态读, 不硬编码)
    cfg_path = os.path.join(src, "config.json")
    old_cfg = json.load(open(cfg_path))
    OLD_INTER = int(old_cfg["intermediate_size"])
    OLD_LAYERS = int(old_cfg["num_hidden_layers"])
    NEW_INTER = OLD_INTER * 3 // 2          # 3072 -> 4608
    NEW_LAYERS = OLD_LAYERS + 6             # 30 -> 36
    print(f"[expand] OLD inter={OLD_INTER} layers={OLD_LAYERS} -> NEW inter={NEW_INTER} layers={NEW_LAYERS}")

    # 2) 原权重(经 safetensors 加载, 绕开 torch<2.6 的 bin pickle CVE)
    print("[expand] 加载原模型(读 safetensors)...")
    src_model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).float()
    src_model.eval()
    sd = {k: v.detach().cpu().float().clone() for k, v in src_model.state_dict().items()}

    # 3) 新 config: 仅改 intermediate_size / num_hidden_layers, 其余完全一致
    for k in ("torch_dtype", "dtype"):
        old_cfg.pop(k, None)
    old_cfg["intermediate_size"] = NEW_INTER
    old_cfg["num_hidden_layers"] = NEW_LAYERS
    old_cfg["torch_dtype"] = "float32"
    cfg = LlamaConfig(**{k: v for k, v in old_cfg.items()
                         if k not in ("architectures", "transformers_version",
                                      "is_llama_config", "rope_interleaved",
                                      "rope_scaling")})
    model = AutoModelForCausalLM.from_config(cfg).float()
    model.eval()

    # 4) 变换权重: MLP 加宽(新增 0), 其余原样; 末 6 层恒等(全 0 输出)
    new_sd = {}
    for k, v in sd.items():
        if "lm_head" in k:
            # 与 embed 绑定(或独立): 保持原样即可, 隐藏维未变
            new_sd[k] = v.clone()
            continue
        if "gate_proj" in k or "up_proj" in k:
            w = torch.zeros(NEW_INTER, v.shape[1]); w[:v.shape[0]] = v
            new_sd[k] = w
        elif "down_proj" in k:
            w = torch.zeros(v.shape[0], NEW_INTER); w[:, :v.shape[1]] = v
            new_sd[k] = w
        else:
            new_sd[k] = v.clone()

    # 新增 6 层: 全零注意力输出 + 全零 MLP 输出 = 恒等
    h = old_cfg["hidden_size"]; it = NEW_INTER
    k_shape = new_sd["model.layers.0.self_attn.k_proj.weight"].shape
    v_shape = new_sd["model.layers.0.self_attn.v_proj.weight"].shape
    for i in range(OLD_LAYERS, NEW_LAYERS):
        p = f"model.layers.{i}."
        new_sd[p+"self_attn.q_proj.weight"] = torch.zeros(h, h)
        new_sd[p+"self_attn.k_proj.weight"] = torch.zeros(*k_shape)
        new_sd[p+"self_attn.v_proj.weight"] = torch.zeros(*v_shape)
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

    # 5) 保函数校验: 同 prompt, 新旧模型 logits 必须一致
    # ckpt 目录历史上不含 tokenizer(训练产物只有 config/safetensors/step.txt),
    # tokenizer 常驻 base_tok/ —— 2026-10-16 踩坑: 直接 from_pretrained(ckpt) 会
    # fallback 到 sentencepiece 慢速路径, transformers 4.55 下 TypeError: not a string 崩掉。
    TOK_SRC = src
    if not os.path.exists(os.path.join(src, "tokenizer.json")):
        for cand in (os.path.join(os.path.dirname(os.path.abspath(src)), "base_tok"),
                     "/mnt/workspace/node1/chain/base_tok"):
            if os.path.exists(os.path.join(cand, "tokenizer.json")):
                TOK_SRC = cand
                break
    print(f"[expand] tokenizer src={TOK_SRC}")
    tok = AutoTokenizer.from_pretrained(TOK_SRC)
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

    # 6) 落盘(原地覆盖 ckpt 目录, 仅 safetensors, 删旧 bin 防 trainer 读旧权重)
    model.save_pretrained(src, safe_serialization=True)
    tok.save_pretrained(src)
    binp = os.path.join(src, "pytorch_model.bin")
    if os.path.exists(binp):
        os.remove(binp)
        print("[expand] 已删旧 pytorch_model.bin")
    print(f"OK 0.3B 已写入 {src} (inter={NEW_INTER} layers={NEW_LAYERS})")

if __name__ == "__main__":
    main()
