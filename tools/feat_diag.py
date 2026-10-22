# -*- coding: utf-8 -*-
"""音频特征区分度诊断（中心化后再判，别被 Whisper 的公共分量骗）。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "audio", "aud_feats.pt")
PAIRS = os.path.join(ROOT, "data", "audio", "aishell_pairs.jsonl")


def cos_stats(x):
    """x: (N,D) -> (mean,min,max) 余弦"""
    n = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    c = n @ n.t()
    off = c[~torch.eye(len(x), dtype=torch.bool)]
    return float(off.mean()), float(off.min()), float(off.max())


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    keys = list(cache.keys())
    print("[cache] %d 条" % len(keys))
    print("[cache] key 样例: %s" % [str(k)[:40] for k in keys[:3]])

    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8") if l.strip()]
    print("[pairs] %d 行, 样例 utt=%s" % (len(rows), rows[0].get("utt")))
    hit = sum(1 for r in rows if r.get("utt") in cache)
    print("[HIT] pairs 的 utt 命中缓存: %d/%d = %.1f%%"
          % (hit, len(rows), 100.0 * hit / max(1, len(rows))))

    N = 24
    smp_keys = keys[:N]
    xs = torch.stack([cache[k].float() for k in smp_keys])       # (N,150,768)
    print("[raw] 全局 std=%.4f  mean_norm=%.2f"
          % (float(xs.std()), float(xs.mean(dim=(0, 1)).norm())))

    # 不同音频：整条展平
    flat = xs.reshape(N, -1)
    m, lo, hi = cos_stats(flat)
    print("[cos 原样] 不同音频两两余弦  mean=%.4f min=%.4f max=%.4f" % (m, lo, hi))

    # 中心化（减全体均值）
    mu = xs.mean(dim=(0, 1), keepdim=True)
    ce = (flat - mu.reshape(1, -1))
    m, lo, hi = cos_stats(ce)
    print("[cos 中心化] 不同音频两两余弦  mean=%.4f min=%.4f max=%.4f" % (m, lo, hi))

    # 相对能量：去掉均值后还剩多少
    tot = float((flat ** 2).sum())
    resid = float((ce ** 2).sum())
    print("[能量] 去均值后剩余 %.2f%%（<10%% 说明几乎只有公共分量）" % (100.0 * resid / max(tot, 1e-8)))

    # 同一条音频内部 150 帧之间的余弦
    intra = []
    for i in range(min(4, N)):
        v = xs[i]                                      # (150,768)
        n = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        c = n @ n.t()
        intra.append(float(c[~torch.eye(len(v), dtype=torch.bool)].mean()))
    print("[cos 帧间] 同一条音频内部平均余弦 %s" % ["%.4f" % z for z in intra])

    # 帧位置上：相邻音频同一帧位置的余弦
    n0 = xs[0] / xs[0].norm(dim=1, keepdim=True).clamp_min(1e-8)
    n1 = xs[1] / xs[1].norm(dim=1, keepdim=True).clamp_min(1e-8)
    pos = (n0 * n1).sum(dim=1)
    print("[cos 同位] 两条音频逐帧余弦 mean=%.4f min=%.4f max=%.4f"
          % (float(pos.mean()), float(pos.min()), float(pos.max())))


if __name__ == "__main__":
    main()
