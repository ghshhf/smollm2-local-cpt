"""
LoRA 多卡加权合成核心（自己实现，不依赖 peft 版本）
====================================================
为什么不用 peft 的 add_weighted_adapter：
  本机 peft 0.21.2 **已经移除**了 PeftModel.add_weighted_adapter（实测 hasattr=False），
  且新版 API 位置多次变动。而多卡合并的数学非常简单，直接手算最稳：

      W_new = W_base + Σ_i  w_i * (alpha_i / r_i) * (B_i @ A_i)

这样做的好处：
  1. 任意张数、任意权重都能合（"使用的时候才合起来"）
  2. 不依赖 peft 行为差异，将来升级不会挂
  3. 每张卡的贡献权重可精确调控（即用即合，不用重新训练）
"""
import json
import os

import torch
from safetensors import safe_open

ADAPTER_PREFIX = "base_model.model."


def read_adapter_weights(adapter_dir: str):
    """读一张卡的 LoRA 权重，返回 {模块名: {'A': tensor, 'B': tensor}} + scaling。"""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    r = float(cfg.get("r", 16))
    alpha = float(cfg.get("lora_alpha", 32))
    scaling = alpha / r

    if "rank_pattern" in cfg and cfg["rank_pattern"]:
        raise RuntimeError(f"{adapter_dir}: 不支持 rank_pattern（各 rank 不同），暂无法手算合并")

    w_path = None
    for cand in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(adapter_dir, cand)
        if os.path.isfile(p):
            w_path = p
            break
    if w_path is None:
        raise FileNotFoundError(f"{adapter_dir}: 找不到 adapter_model.safetensors / .bin")

    pairs = {}
    if w_path.endswith(".safetensors"):
        with safe_open(w_path, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k).to(torch.float32)
                for suffix, slot in (("lora_A.weight", "A"), ("lora_B.weight", "B")):
                    if k.endswith(suffix):
                        mod = k[len(ADAPTER_PREFIX):] if k.startswith(ADAPTER_PREFIX) else k
                        mod = mod[: -len("." + suffix)]
                        pairs.setdefault(mod, {})[slot] = t
                        break
    else:
        sd = torch.load(w_path, map_location="cpu")
        for k, t in sd.items():
            for suffix, slot in (("lora_A.weight", "A"), ("lora_B.weight", "B")):
                if k.endswith(suffix):
                    mod = k[len(ADAPTER_PREFIX):] if k.startswith(ADAPTER_PREFIX) else k
                    mod = mod[: -len("." + suffix)]
                    pairs.setdefault(mod, {})[slot] = t.to(torch.float32)
                    break
    return pairs, scaling


def apply_lora_deltas(model, adapters: list[str], weights: list[float] | None = None,
                      verbose: bool = True):
    """
    把多张 LoRA 卡按权重叠加基座权重上（原地修改 model）。
    返回统计 dict，便于打印报告。
    """
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters) if adapters else []
    assert len(adapters) == len(weights), "卡数与权重数必须一致"

    sd = model.state_dict()
    report = {"cards": [], "modules_total": 0, "modules_applied": 0, "skipped": []}

    for path, w in zip(adapters, weights):
        pairs, scaling = read_adapter_weights(path)
        applied, missed = 0, []
        name = os.path.basename(path.rstrip("\\/")) or path
        for mod, d in pairs.items():
            if "A" not in d or "B" not in d:
                missed.append(mod + "(缺A/B)")
                continue
            base_key = f"{mod}.weight"
            if base_key not in sd:
                missed.append(mod + "(基座无此层)")
                continue
            base_w = sd[base_key]
            delta = (d["B"] @ d["A"]) * (scaling * w)
            if tuple(delta.shape) != tuple(base_w.shape):
                missed.append(f"{mod}(形状 {tuple(delta.shape)} vs {tuple(base_w.shape)})")
                continue
            base_w.add_(delta)
            applied += 1
        report["cards"].append({
            "card": name,
            "weight": w,
            "scaling": scaling,
            "applied": applied,
            "missed": missed[:5],
            "n_pairs": len(pairs),
        })
        report["modules_total"] += len(pairs)
        report["modules_applied"] += applied
        if missed:
            report["skipped"].extend(missed[:5])
        if verbose:
            flag = "OK" if applied == len(pairs) else f"部分命中 {applied}/{len(pairs)}"
            print(f"  [{name}] w={w:g} scaling={scaling:g} -> 合并 {applied} 层 {flag}")

    model.load_state_dict(sd)
    return report
