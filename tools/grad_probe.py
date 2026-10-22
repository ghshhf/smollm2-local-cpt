# -*- coding: utf-8 -*-
"""align 阶段梯度诊断：直接量 projector 的 grad norm，别猜。

用法: python tools/grad_probe.py --llm models/polaris-03b --whisper models/whisper-small \
        --cache data/audio/aud_feats.pt --steps 4 --aud-tokens 150
输出每步 loss 与 grad_norm（按层、按 parametri 分组），并对比不同 aud-tokens。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


class MLPProjector(nn.Module):
    def __init__(self, a_dim, l_dim):
        super().__init__()
        self.fc1 = nn.Linear(a_dim, l_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(l_dim, l_dim, bias=True)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--cache", default="data/audio/aud_feats.pt")
    ap.add_argument("--aud-tokens", type=int, default=150)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    torch.set_num_threads(3)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[load] %s" % a.llm, flush=True)
    tok = AutoTokenizer.from_pretrained(a.llm)
    llm = AutoModelForCausalLM.from_pretrained(a.llm, torch_dtype=torch.float32)
    try:
        llm = llm.float()
    except Exception:
        pass
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    hid = llm.config.hidden_size
    print("[llm] hidden=%d layers=%d vocab=%d"
          % (hid, llm.config.num_hidden_layers, llm.config.vocab_size), flush=True)

    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    keys = [k for k in cache.keys()]
    print("[cache] %d 条, 单条 shape=%s" % (len(keys), tuple(cache[keys[0]].shape)), flush=True)

    f0 = cache[keys[0]].float()
    n_tok = a.aud_tokens
    if n_tok > 0 and n_tok < f0.size(0):
        idx = torch.linspace(0, f0.size(0) - 1, n_tok).round().long()
        pooled = f0[idx]
    else:
        pooled = f0
    print("[pool] %d 帧 -> %d token, 池化后 std=%.4f norm=%.3f"
          % (f0.size(0), pooled.size(0), float(pooled.std()), float(pooled.norm())), flush=True)

    # 区分度：随机取 8 条，两两余弦
    smp = [cache[k].float() for k in keys[:8]]
    smp = torch.stack([s[torch.linspace(0, s.size(0) - 1, n_tok).round().long()] for s in smp])
    smp = smp.mean(dim=1)                      # (8, 768) 全局平均，看特征本身有没有区分度
    nz = smp / smp.norm(dim=1, keepdim=True)
    cos = nz @ nz.t()
    off = cos[~torch.eye(len(smp), dtype=torch.bool)]
    print("[feat] 原始特征两两余弦 mean=%.3f min=%.3f max=%.3f"
          % (float(off.mean()), float(off.min()), float(off.max())), flush=True)

    proj = MLPProjector(f0.size(1), hid)
    opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=0.0)

    # 用 prompt 里真实存在的 token 造目标，避免依赖 pairs 文本
    PROMPT = "<|im_start|>user\n请把这段语音转成文字。<|im_end|>\n<|im_start|>assistant\n"
    tgt_text = "家中接连有了四位重病患者"
    pids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    tgt = tok(tgt_text, return_tensors="pt", add_special_tokens=False).input_ids
    eos = tok("<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
    tgt = torch.cat([tgt, eos], dim=1)
    temb = llm.get_input_embeddings()(torch.cat([pids, tgt], dim=1))
    print("[shape] prompt=%d tgt=%d" % (pids.size(1), tgt.size(1)), flush=True)

    for s in range(1, a.steps + 1):
        use = keys[(s - 1) % len(keys)]
        fr = cache[use].float()
        if n_tok > 0 and n_tok < fr.size(0):
            fr = fr[torch.linspace(0, fr.size(0) - 1, n_tok).round().long()]
        atok = proj(fr.unsqueeze(0))
        inputs = torch.cat([atok, temb], dim=1)
        t0 = time.time()
        out = llm(inputs_embeds=inputs)
        np_ = atok.size(1) + pids.size(1)
        logits = out.logits[:, np_ - 1:-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = {}
        for n, p in proj.named_parameters():
            g = p.grad
            gn[n] = float(g.norm()) if g is not None else -1.0
        gn_all = float(torch.sqrt(sum((p.grad.norm() ** 2) for p in proj.parameters() if p.grad is not None)))
        out_norm = float(atok.norm())
        print("[step %d] loss=%.4f  audtok_norm=%.4f  total_grad=%.3e  %.1fs"
              % (s, float(loss), out_norm, gn_all, time.time() - t0), flush=True)
        for n, v in gn.items():
            print("    %-12s grad=%.3e" % (n, v), flush=True)
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()

    print("[done] 若 total_grad 在 1e-8 量级或逐层递减剧烈 => 梯度消失（36 层冻结主干导致）", flush=True)


if __name__ == "__main__":
    main()
