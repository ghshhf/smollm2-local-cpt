#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ttt_latent.py — 测试时隐变量适配（T2-Z 启发，P1 · 推理时进化）

机制（守保函数 + E-7）:
  - 主干(北极星 0.3B)冻结，只训练一个紧凑 latent 向量（形状=hidden，恒定维）。
  - latent 以「加性残差」注入残差流（embedding 级），绝不碰 gate/up/down 可扩维。
  - 测试时从一个/几个 support 交互里用几步梯度优化 latent（next-token CE），
    让模型适配未见环境/认主个性化，T2-Z 同款思路：冻结世界模型 + 优化 conditioning latent。
  - 与 In-Place-TTT fast weights 正交：fast weights 装知识，latent 装「场景/身份」偏移。

用法（需 venv_ttt + 0.3B）:
  venv_ttf/Scripts/python.exe tools/ttt_latent.py --base <0.3B> --adapt-text "支持语料" --query "任务"
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAdapter(nn.Module):
    """紧凑 latent：单个 (hidden,) 向量，加性残差注入，零初始化（初始=无偏移）。"""

    def __init__(self, hidden):
        super().__init__()
        self.latent = nn.Parameter(torch.zeros(hidden))

    def bias(self):
        return self.latent  # (hidden,)


def adapt(model, tok, support_text, steps=12, lr=0.05):
    """测试时优化 latent：冻结主干，仅 latent 可训。返回适配好的 adapter。"""
    hidden = model.config.hidden_size
    adapter = LatentAdapter(hidden).to(next(model.parameters()).device)
    adapter.train()
    opt = torch.optim.Adam([adapter.latent], lr=lr)
    ids = tok(support_text, return_tensors="pt").input_ids
    emb0 = model.get_input_embeddings()
    losses = []
    for step in range(steps):
        opt.zero_grad()
        emb = emb0(ids) + adapter.latent.unsqueeze(0)   # 残差流加性注入
        out = model(inputs_embeds=emb)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(),
            ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return adapter, losses


@torch.no_grad()
def generate(model, tok, adapter, prompt, n_new=60):
    """全量重前向生成（与 ttt_infer 一致：cache 不延续 fast weights，宁可慢也要对）。"""
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    b = adapter.latent.unsqueeze(0)
    for _ in range(n_new):
        emb = model.get_input_embeddings()(ids) + b
        out = model(inputs_embeds=emb)
        nxt = int(out.logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="北极星底座路径(如 0.3B)")
    ap.add_argument("--adapt-text", default="", help="测试时支持语料(优化 latent 用)")
    ap.add_argument("--query", default="请用一句话介绍北京：")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-new", type=int, default=60)
    ap.add_argument("--ttt-on", action="store_true", help="同时开启 In-Place TTT fast weights")
    a = ap.parse_args()

    # 复用 ttt_infer 的 build（含恒等初始化 ttt_conv、权重对拍纪律）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ttt_infer import build, get_tok
    model = build(base=a.base, ttt_on=a.ttt_on,
                  ttt_layers=list(range(22, 30)) if a.ttt_on else None)
    tok = get_tok(a.base)

    if a.adapt_text:
        print("[latent] 测试时优化 latent (%d 步, lr=%.3f) ..." % (a.steps, a.lr))
        adapter, losses = adapt(model, tok, a.adapt_text, a.steps, a.lr)
        print("[latent] 末步 loss=%.4f" % losses[-1])
        out = generate(model, tok, adapter, a.query, a.n_new)
        print("[生成·含 latent] %s" % out[len(a.query):][:300])
    else:
        # 仅演示：零 latent（= 无偏移 baseline）
        adapter = LatentAdapter(model.config.hidden_size)
        out = generate(model, tok, adapter, a.query, a.n_new)
        print("[生成·无latent baseline] %s" % out[len(a.query):][:300])


if __name__ == "__main__":
    main()
