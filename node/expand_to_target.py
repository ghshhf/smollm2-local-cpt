# -*- coding: utf-8 -*-
"""
expand_to_target.py — 通用保函数扩容: 给定目标参数量(M), 自动求解 MLP 宽度与层数。

宪法约束(绝不违反):
  - 不改 hidden_size / 注意力头数 / GQA / RoPE / 词表 / 位置编码
  - 只允许两种操作: ① MLP 加宽(新增行列全 0 = 死神经元) ② 新增层(输出全 0 = 恒等层)
  - 落盘前 logits max|diff| < 1e-3 校验, 不过不落盘

求解策略(先宽后深, 控制宽深比):
  1) 优先加宽 MLP: inter 上限 = WIDE_CAP * hidden (默认 12x), 且对齐 128(GEMM 友好)
  2) 加宽到上限还不够 => 剩余靠新增恒等层补齐
  参考档位(hidden=576, layers=36):
      0.3B 346.9M inter=4608
      0.4B ~403M  inter=5504
      0.5B ~503M  inter=6912 layers=37
      0.6B ~605M  inter=6912 layers=45

用法: python3 expand_to_target.py <ckpt_dir> <target_M>
"""
import json, sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

WIDE_CAP = 12.0   # inter / hidden 上限(仅 --allow-wide 时启用)
DEPTH_FIRST = "--allow-wide" not in sys.argv   # 默认深度优先(宪法 §8.2 硬裁决)


def main():
    src = sys.argv[1]
    target_M = float(sys.argv[2])
    print(f"[expand] src={src} target={target_M}M")

    old_cfg = json.load(open(os.path.join(src, "config.json")))
    H = int(old_cfg["hidden_size"])
    OLD_INTER = int(old_cfg["intermediate_size"])
    OLD_LAYERS = int(old_cfg["num_hidden_layers"])

    src_model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).float()
    src_model.eval()
    sd = {k: v.detach().cpu().float().clone() for k, v in src_model.state_dict().items()}
    cur = sum(p.numel() for p in src_model.parameters()) / 1e6

    # 每层注意力参数(精确, 从权重统计)
    attn_per = sum(v.numel() for k, v in sd.items() if ".self_attn." in k and ".layers.0." in k)
    mlp_per_inter = 3 * H  # inter 每 +1, 每层 MLP 增 3*H

    def params_at(inter, layers):
        base = cur - OLD_LAYERS * (attn_per + mlp_per_inter * OLD_INTER)
        return base + layers * (attn_per + mlp_per_inter * inter)

    # 🔴 宪法 §8.2 硬裁决: 深度优先是唯一严格保函数扩容方向。
    #    实测(三值口径): 加零输出层/Block 跳变 0; MLP 加宽跳变 +1.70% (γ=mean|W| 被补零稀释)。
    #    0.5B 走 BitNet b1.58 => 必须加层, 不加宽。
    per_layer_new = (attn_per + mlp_per_inter * OLD_INTER) / 1e6
    need = target_M * 1e6 - cur * 1e6
    if need <= 0:
        print(f"[expand] 当前 {cur:.1f}M 已达目标, 无需扩容"); return 0
    if need < per_layer_new * 1e6 * 0.5:
        # 重启续跑时 src 已是目标量级, max(1,...) 会硬加一层把 400M 顶到 408.8M(踩过)
        print(f"[expand] 当前 {cur:.1f}M 已接近目标 {target_M}M "
              f"(缺口 {need/1e6:.2f}M 不足半层), 跳过扩容"); return 0
    if DEPTH_FIRST:
        add_layers = max(1, int(need / (per_layer_new * 1e6) + 0.5))
        new_inter = OLD_INTER              # 宽度不动
        new_layers = OLD_LAYERS + add_layers
    print(f"[expand] 求解: inter {OLD_INTER}->{new_inter}, layers {OLD_LAYERS}->{new_layers} "
          f"=> {params_at(new_inter, new_layers)/1e6:.1f}M (目标 {target_M}M)")

    for k in ("torch_dtype", "dtype"):
        old_cfg.pop(k, None)
    old_cfg["intermediate_size"] = new_inter
    old_cfg["num_hidden_layers"] = new_layers
    old_cfg["torch_dtype"] = "float32"
    cfg = LlamaConfig(**{k: v for k, v in old_cfg.items()
                         if k not in ("architectures", "transformers_version",
                                      "is_llama_config", "rope_interleaved", "rope_scaling")})
    model = AutoModelForCausalLM.from_config(cfg).float()
    model.eval()

    new_sd = {}
    for k, v in sd.items():
        if "gate_proj" in k or "up_proj" in k:
            w = torch.zeros(new_inter, v.shape[1]); w[:v.shape[0]] = v; new_sd[k] = w
        elif "down_proj" in k:
            w = torch.zeros(v.shape[0], new_inter); w[:, :v.shape[1]] = v; new_sd[k] = w
        else:
            new_sd[k] = v.clone()

    k_shape = new_sd["model.layers.0.self_attn.k_proj.weight"].shape
    v_shape = new_sd["model.layers.0.self_attn.v_proj.weight"].shape
    for i in range(OLD_LAYERS, new_layers):
        p = f"model.layers.{i}."
        new_sd[p + "self_attn.q_proj.weight"] = torch.zeros(H, H)
        new_sd[p + "self_attn.k_proj.weight"] = torch.zeros(*k_shape)
        new_sd[p + "self_attn.v_proj.weight"] = torch.zeros(*v_shape)
        new_sd[p + "self_attn.o_proj.weight"] = torch.zeros(H, H)
        new_sd[p + "mlp.gate_proj.weight"] = torch.zeros(new_inter, H)
        new_sd[p + "mlp.up_proj.weight"] = torch.zeros(new_inter, H)
        new_sd[p + "mlp.down_proj.weight"] = torch.zeros(H, new_inter)
        new_sd[p + "input_layernorm.weight"] = torch.ones(H)
        new_sd[p + "post_attention_layernorm.weight"] = torch.ones(H)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    bad = [k for k in missing if "lm_head" not in k and "rotary" not in k]
    if bad or unexpected:
        print("FATAL missing:", bad[:8], "unexpected:", list(unexpected)[:5]); sys.exit(1)

    TOK_SRC = src
    if not os.path.exists(os.path.join(src, "tokenizer.json")):
        for cand in ("/root/r4work/base_next", "/mnt/workspace/node1/chain/base_tok"):
            if os.path.exists(os.path.join(cand, "tokenizer.json")):
                TOK_SRC = cand; break
    tok = AutoTokenizer.from_pretrained(TOK_SRC)
    with torch.no_grad():
        lo = src_model(**tok("夜色像潮水一样漫过", return_tensors="pt")).logits
        ln = model(**tok("夜色像潮水一样漫过", return_tensors="pt")).logits
    diff = (lo - ln).abs().max().item()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {n:.2f}M   保函数校验 logits max|diff| = {diff:.6f}")
    if diff > 1e-3:
        print("FATAL: 函数未保持, 不落盘"); sys.exit(1)

    model.save_pretrained(src, safe_serialization=True)
    b = os.path.join(src, "pytorch_model.bin")
    if os.path.exists(b):
        os.remove(b)
    print(f"OK {n:.1f}M 已写入 {src} (inter={new_inter} layers={new_layers})")


if __name__ == "__main__":
    main()
