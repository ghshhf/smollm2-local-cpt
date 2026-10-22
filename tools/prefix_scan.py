# -*- coding: utf-8 -*-
"""扫 prefix 长度：判定「随机 soft token 数量」是不是 loss 暴涨的毒源。

同一批样本、同一个随机 projector，只改 aud_tokens。
若 loss 随长度单调上升 => 长 prefix 是毒源，应大幅缩短。
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class MLPProjector(nn.Module):
    def __init__(self, a_dim, l_dim):
        super().__init__()
        self.fc1 = nn.Linear(a_dim, l_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(l_dim, l_dim)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


PROMPT = "<|im_start|>user\n请把这段语音转成文字。<|im_end|>\n<|im_start|>assistant\n"


def pool(x, n):
    if n <= 0 or n >= x.size(0):
        return x
    idx = torch.linspace(0, x.size(0) - 1, n).round().long()
    return x[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--cache", default="data/audio/aud_feats.pt")
    ap.add_argument("--pairs", default="data/audio/aishell_pairs.jsonl")
    ap.add_argument("--n-samples", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(3)
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("utt") in cache][:a.n_samples]
    print("[data] %d 条样本" % len(rows), flush=True)

    pids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    eos = tok("<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
    print("[len] prompt=%d" % pids.size(1), flush=True)

    torch.manual_seed(7)
    proj = MLPProjector(cache[rows[0]["utt"]].size(1), hid)

    print("\n%-9s %-9s %-9s %-9s %s" % ("aud_tok", "avg_loss", "aud_norm", "总长度", "说明"))
    print("-" * 62)
    for n_tok in [1, 4, 8, 16, 32, 64, 150]:
        losses = []
        norm = 0.0
        L = 0
        for r in rows:
            fr = pool(cache[r["utt"]].float(), n_tok)
            atok = proj(fr.unsqueeze(0))
            norm = float(atok.norm())
            g = tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids
            g = torch.cat([g, eos], dim=1)
            temb = llm.get_input_embeddings()(torch.cat([pids, g], dim=1))
            inputs = torch.cat([atok, temb], dim=1)
            with torch.no_grad():
                out = llm(inputs_embeds=inputs)
            np_ = atok.size(1) + pids.size(1)
            logits = out.logits[:, np_ - 1:-1, :]
            ls = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       g.reshape(-1)))
            losses.append(ls)
            L = inputs.size(1)
        avg = sum(losses) / len(losses)
        tag = "← 疑似毒源" if avg > 12 else ("正常" if avg < 7 else "偏高")
        print("%-9d %-9.3f %-9.2f %-9d %s" % (n_tok, avg, norm, L, tag), flush=True)

    print("\n[ref] 纯文本基线 NLL≈2.5-2.7，随机猜 ln(49152)=10.80", flush=True)


if __name__ == "__main__":
    main()
