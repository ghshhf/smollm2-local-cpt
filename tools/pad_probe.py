# -*- coding: utf-8 -*-
"""验证「1500 帧里有多少是 padding」——判定池化是否把静音平均进了 soft token。

Whisper encoder 恒定输出 1500 帧（50 Hz × 30 s）。
AISHELL 句子 3~5 s => 有效帧 150~250，占 10~17%，其余全是 padding。
若 cache 里 150 帧的相邻差异在后段骤降为 ~0，即证明大部分 token 是静音。
"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    keys = list(cache.keys())

    # 音频时长（若 pairs 里有）
    durs = {}
    if a.pairs:
        try:
            for l in open(a.pairs, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                if r.get("dur"):
                    durs[r["utt"]] = float(r["dur"])
        except Exception:
            pass

    print("%-20s %-8s %-9s %-9s %-9s %s"
          % ("utt", "时长s", "前段Δ", "后段Δ", "骤降点", "结论"))
    print("-" * 78)
    for k in keys[:a.n]:
        x = cache[k].float()                  # (150, 768)
        d = (x[1:] - x[:-1]).norm(dim=1)      # 149 个相邻差
        n = len(d)
        head = float(d[:max(1, n // 5)].mean())
        tail = float(d[-max(1, n // 5):].mean())
        # 骤降点：第一个连续 8 帧差异都低于 head*3% 的位置
        th = head * 0.03
        cut = -1
        run = 0
        for i, v in enumerate(d):
            if float(v) < th:
                run += 1
                if run >= 8:
                    cut = i - 7
                    break
            else:
                run = 0
        ratio = tail / max(head, 1e-8)
        dur = durs.get(k)
        verdict = ("严重 padding 污染" if ratio < 0.02 else
                   "部分 padding" if ratio < 0.15 else "基本满信息")
        print("%-20s %-8s %-9.1f %-9.1f %-9s %s"
              % (k, ("%.2f" % dur) if dur else "?",
                 head, tail, ("%d/150" % cut) if cut >= 0 else "-", verdict))

    # 汇总：估算平均有效帧
    eff = []
    for k in keys[:max(a.n, 40)]:
        x = cache[k].float()
        d = (x[1:] - x[:-1]).norm(dim=1)
        head = float(d[:max(1, len(d) // 5)].mean())
        th = head * 0.03
        run, cut = 0, len(d)
        for i, v in enumerate(d):
            if float(v) < th:
                run += 1
                if run >= 8:
                    cut = i - 7
                    break
            else:
                run = 0
        eff.append(cut)
    print("\n[汇总] 前 %d 条中，平均有效帧 = %.0f / 150（%.0f%%），中位数 %d"
          % (len(eff), sum(eff) / len(eff), 100.0 * sum(eff) / len(eff) / 150,
             sorted(eff)[len(eff) // 2]))
    print("[结论] 有效帧占比 <25%% 时，池化到 150 token 会把绝大部分算力花在静音上，")
    print("       且所有音频的静音段特征高度相似 => 不同音频余弦被抬到 0.86 就是这个原因。")


if __name__ == "__main__":
    main()
