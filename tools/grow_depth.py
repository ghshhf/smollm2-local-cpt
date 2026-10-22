# -*- coding: utf-8 -*-
"""0.3B(36 层) → 0.4B(42 层) 深度扩容，保函数。

宪法：扩容必须保函数——不动注意力维、隐藏维、RoPE、词表，只增加深度。

为什么是 42 层：
    每层 (q 576² + k/v 576×192×2 + o 576² + gate/up 576×4608×2 + down 4608×576)
        = 8,847,360
    36 × 8.847M = 318.5M，加 tie 的 embed 49152×576 = 28.3M ⇒ 346.9M  ✓ 对上现有 0.3B
    42 × 8.847M + 28.3M = 400.0M                                          ✓ 正好 0.4B

新层的插法（这是关键，做错就白扩容）：
    复制相邻层 → 把 **o_proj 和 down_proj 清零**，其余保留。
    前向：注意力输出被 o_proj 吃掉、MLP 输出被 down_proj 吃掉 ⇒ 该层输出恒为 0 ⇒ 残差流不变
          ⇒ 42 层等价于 36 层，max|logits diff| 必须 = 0。
    反向：o_proj 的梯度 = 上游梯度 × 注意力输出（非零），down_proj 的梯度 = 上游梯度 × MLP 中间激活（非零）
          ⇒ 梯度通，不是「整块零」（整块零会让 ∂L/∂W = 0，扩容白做）。

    ❌ 不这么做的后果：若把 gate/up 也清零，MLP 中间激活为 0，down_proj 梯度变 0，新层永远醒不过来。

用法：
    python tools/grow_depth.py --src backups/0.3B/polaris-03b_20261018 \
                               --dst models/polaris-04b-depth42 --add 6
    python tools/grow_depth.py --src ... --dst ... --add 6 --verify
"""
import argparse
import json
import os
import shutil
import sys

import torch

ZERO_OUT = ("o_proj", "down_proj")


def build_mapping(old_n, add):
    """均匀插入：每 old_n//add 层后面插一个新层。返回 [(new_idx, src_old_idx|None)]"""
    if add <= 0:
        return [(i, i) for i in range(old_n)]
    stride = max(1, old_n // add)
    mapping, ni = [], 0
    for i in range(old_n):
        mapping.append((ni, i))
        ni += 1
        if (i + 1) % stride == 0 and sum(1 for _, s in mapping if s is None) < add:
            mapping.append((ni, None))
            ni += 1
    # 若还没插够，补在末尾
    while sum(1 for _, s in mapping if s is None) < add:
        mapping.append((ni, None))
        ni += 1
    return mapping


def grow(src, dst, add=6, dtype=torch.float32):
    from safetensors.torch import load_file, save_file

    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    old_n = int(cfg["num_hidden_layers"])
    new_n = old_n + add

    print("[grow] %d 层 → %d 层（+%d）" % (old_n, new_n, add))
    print("[grow] 读权重 ...")
    sd = load_file(os.path.join(src, "model.safetensors"))
    print("[grow] 张量数 %d，元素 %.1fM" % (
        len(sd), sum(v.numel() for v in sd.values()) / 1e6))

    mapping = build_mapping(old_n, add)
    new_sd = {}
    zerod = []

    for key, val in sd.items():
        if ".layers." not in key:
            new_sd[key] = val                      # embed / norm / lm_head 原样
            continue
        parts = key.split(".")
        li = int(parts[parts.index("layers") + 1])
        rest = parts[parts.index("layers") + 2:]
        for ni, si in mapping:
            if si == li:
                new_sd[".".join(parts[:2] + [str(ni)] + rest)] = val
    # 第二遍：新层从它前面那一层复制，再按 ZERO_OUT 清零
    for ni, si in mapping:
        if si is not None:
            continue
        prev = [s for n2, s in mapping if n2 == ni - 1][0]
        if prev is None:
            prev = 0
        for key, val in list(sd.items()):
            if ".layers.%d." % prev not in key:
                continue
            nk = key.replace(".layers.%d." % prev, ".layers.%d." % ni)
            if any(z in key for z in ZERO_OUT):
                new_sd[nk] = torch.zeros_like(val)
                zerod.append(nk)
            else:
                new_sd[nk] = val.clone()
        print("[grow] 新层 %2d ← 复制第 %d 层，o_proj/down_proj 清零" % (ni, prev))

    cfg["num_hidden_layers"] = new_n
    os.makedirs(dst, exist_ok=True)
    save_file(new_sd, os.path.join(dst, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, fn))

    n_par = sum(v.numel() for v in new_sd.values())
    print("[grow] 完成 → %s" % dst)
    print("[grow] 参数 %.2fM（%.3fB）  清零张量 %d 个" % (
        n_par / 1e6, n_par / 1e9, len(zerod)))
    return new_n


def verify(src, dst, n_prompts=3, seq=64):
    """保函数校验：同一输入，max|logits diff| 必须为 0。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    print("[verify] 加载两个模型 ...")
    m1 = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    m2 = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float32).eval()

    texts = ["今天天气不错，我们去公园散步。", "def foo(x):\n    return x + 1",
             "北极星是主人的私有模型。"][:n_prompts]
    worst = 0.0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids
            a = m1(ids).logits
            b = m2(ids).logits
            d = (a - b).abs().max().item()
            worst = max(worst, d)
            print("[verify] len=%d  max|diff| = %.9f" % (ids.shape[1], d))

    ok = worst < 1e-5
    print("[verify] 最差 max|diff| = %.9f  ⇒ %s" % (worst, "保函数成立" if ok else "*** 失败 ***"))
    print("VERIFY_MAX_DIFF=%.9f" % worst)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--add", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    grow(a.src, a.dst, a.add)
    if a.verify:
        return 0 if verify(a.src, a.dst) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
