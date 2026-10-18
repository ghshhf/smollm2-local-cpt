# -*- coding: utf-8 -*-
"""
compression_bench.py — 「压缩即模型」本地实测探针 (Delétang et al. 2309.10668 框架)
================================================================
量一个模型在给定语料上的 **adjusted rate**（把模型体积算进去的真实压缩率）：

    adjusted_rate = (NLL_bits + model_bits) / corpus_bits

  - NLL_bits      : 模型在语料上的总负对数似然（/ln2 转 bit）——预测能力
  - model_bits    : 模型自身的存储成本——fp32 / fp16 / 三值(1.58)+fp16嵌入 三种口径
  - corpus_bits   : 语料 UTF-8 原始 bit 数
  - 基线           : gzip / xz 的 bits（无损压缩对照）

用法:
  venv/Scripts/python.exe eval/compression_bench.py \
      --ckpt out_smollm2_135m_pt/checkpoint-latest/pytorch_model.bin \
      --text <某中文文本.txt> [--max-bytes 300000] [--seq-len 1024]

方法论注意（诚实边界）:
  * 三值口径的 model_bits 是**存储账**（非嵌入参数×1.58 + 嵌入×16），
    NLL 仍来自 fp32 权重 ⇒ 三值 adjusted rate 是【乐观下界】，
    前提是 BitNet 原生 QAT 能把 NLL 拉回 fp32 水平（这正是 0.5B 要验证的）。
  * 语料须为**未参与训练**的文本；来源与比例记录进输出 JSON。
  * seq_len 默认 1024（与评测纪律一致，超长截断会低估 NLL，故分块累计）。
"""
import argparse
import gzip
import json
import lzma
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
LN2 = math.log(2.0)

# 三值口径的有效 bits/param（BitNet 原配方：权重三值 + 嵌入 fp16）
B_TERNARY_WEIGHT = 1.58
B_FP16 = 16.0


def count_params(model):
    """拆出 非嵌入 / 嵌入+lm_head 参数量（tied 时只算一次）。"""
    emb_ids = set()
    n_emb = 0
    for name, p in model.named_parameters():
        if "embed" in name.lower() or "lm_head" in name.lower():
            if id(p) not in emb_ids:
                emb_ids.add(id(p))
                n_emb += p.numel()
    n_total = sum(p.numel() for p in model.parameters())
    return n_total - n_emb, n_emb


def model_bits(n_nonemb, n_emb):
    """三种存储口径（bit）。"""
    return {
        "fp32": (n_nonemb + n_emb) * 32.0,
        "fp16": (n_nonemb + n_emb) * 16.0,
        "ternary_w+fp16_emb": n_nonemb * B_TERNARY_WEIGHT + n_emb * B_FP16,
    }


@torch.no_grad()
def total_nll_bits(model, tok, text, seq_len, batch=4, device="cpu"):
    """分块累计 -log2 p(token)，返回 (nll_bits, n_tokens)。"""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    chunks = [ids[i : i + seq_len] for i in range(0, n - 1, seq_len)]
    total_bits = 0.0
    seen = 0
    t0 = time.time()
    for ci in range(0, len(chunks), batch):
        part = chunks[ci : ci + batch]
        maxlen = max(len(c) for c in part)
        input_ids = torch.full((len(part), maxlen), tok.eos_token_id, dtype=torch.long)
        attn = torch.zeros((len(part), maxlen), dtype=torch.long)
        for j, c in enumerate(part):
            input_ids[j, : len(c)] = torch.tensor(c)
            attn[j, : len(c)] = 1
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()  # [B, T, V]
        logp = torch.log_softmax(logits, dim=-1)
        for j, c in enumerate(part):
            L = len(c) - 1
            tgt = torch.tensor(c[1:])
            lp = logp[j, :L, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            total_bits += (-lp.sum()).item() / LN2
            seen += L
        if ci % (batch * 8) == 0:
            r = seen / max(time.time() - t0, 1e-9)
            print(f"  [probe] {seen}/{n-1} tok, {r:.0f} tok/s", file=sys.stderr)
    return total_bits, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pytorch_model.bin（覆盖到 SmolLM2-200M 架构）")
    ap.add_argument("--text", required=True, help="未参与训练的中文文本文件")
    ap.add_argument("--max-bytes", type=int, default=300_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    raw = open(args.text, "rb").read()[: args.max_bytes]
    text = raw.decode("utf-8", errors="ignore")
    corpus_bits = len(raw) * 8.0

    # 无损压缩基线
    gz_bits = len(gzip.compress(raw, 9)) * 8.0
    xz_bits = len(lzma.compress(raw, preset=9)) * 8.0

    print(f"[load] {BASE} + {args.ckpt}", file=sys.stderr)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # 自动探测真实架构（防 base 目录 config 与 ckpt 不一致）
    key0 = next((k for k in sd if k.endswith("mlp.gate_proj.weight")), None)
    if key0 is not None:
        true_inter = sd[key0].shape[0]
        if getattr(cfg, "intermediate_size", None) != true_inter:
            print(f"[cfg] intermediate_size {cfg.intermediate_size} -> {true_inter} (auto)", file=sys.stderr)
            cfg.intermediate_size = true_inter
    ek = next((k for k in sd if "embed_tokens" in k), None)
    if ek is not None and cfg.vocab_size != sd[ek].shape[0]:
        print(f"[cfg] vocab_size {cfg.vocab_size} -> {sd[ek].shape[0]} (auto)", file=sys.stderr)
        cfg.vocab_size = sd[ek].shape[0]
    model = AutoModelForCausalLM.from_config(cfg)
    miss, unexp = model.load_state_dict(sd, strict=False)
    bad = [k for k in sd if k in model.state_dict() and sd[k].shape != model.state_dict()[k].shape]
    if bad:
        print(f"[load][WARN] {len(bad)} 个键形状不匹配（前5: {bad[:5]}）——NLL 不可信!", file=sys.stderr)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} mismatch={len(bad)}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model.eval()

    n_nonemb, n_emb = count_params(model)
    mb = model_bits(n_nonemb, n_emb)
    print(f"[params] non-embed={n_nonemb/1e6:.1f}M embed(+head)={n_emb/1e6:.1f}M", file=sys.stderr)

    nll_bits, n_tok = total_nll_bits(model, tok, text, args.seq_len)
    bpt = nll_bits / max(n_tok, 1)  # bits per token
    # token/汉字（中文场景的输入压缩口径）
    n_han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")

    result = {
        "tag": args.tag or os.path.basename(args.ckpt),
        "corpus": {"path": args.text, "bytes": len(raw), "han_chars": n_han,
                   "tokens": n_tok, "bits": corpus_bits,
                   "bits_per_token": corpus_bits / max(n_tok, 1)},
        "baselines_bits": {"gzip": gz_bits, "xz": xz_bits},
        "model": {"non_embed_params": n_nonemb, "embed_params": n_emb,
                  "storage_bits": mb},
        "nll_bits": nll_bits,
        "bits_per_token_nll": bpt,
        "adjusted_rate": {
            k: (nll_bits + v) / corpus_bits for k, v in mb.items()
        },
        "online_rate_nll_only": nll_bits / corpus_bits,
        "token_per_han": n_tok / max(n_han, 1),
        "notes": "三值口径 adjusted_rate 为乐观下界（NLL 来自 fp32 权重，前提=BitNet QAT 拉回 NLL）",
    }

    out = os.path.join(ROOT, "eval", f"compression_{result['tag'].replace(os.sep,'_')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 压缩实测（越低越好） =====")
    print(f"语料: {len(raw)/1024:.0f} KB / {n_tok} tok / {n_han} 汉字  ({result['token_per_han']:.3f} tok/汉字)")
    print(f"NLL: {bpt:.3f} bit/token   online_rate={result['online_rate_nll_only']:.4f}")
    print(f"gzip={gz_bits/corpus_bits:.4f}  xz={xz_bits/corpus_bits:.4f}")
    for k, v in result["adjusted_rate"].items():
        print(f"adjusted_rate[{k}] = {v:.4f}   (model {mb[k]/8/1e6:.1f} MB)")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
