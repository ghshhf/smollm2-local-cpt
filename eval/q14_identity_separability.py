#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-14 阶段 0：认主可分性上界（零训练）。
来源：docs/clawsgo_audit/open_questions_answers.md §Q-14 / §0 修正 5

判据（FQ-14）：
  FQ-14a  线性探针 AUC > 0.99 → 认主头不需要训练，退化为阈值
  FQ-14b  owner/非owner 余弦间隔 ≥ 5σ（σ = bootstrap 标准差）
  NC      标签打乱重探针，AUC 应回 ~0.5；若仍显著为正 → 数据来源本身可分（泄漏）

诚实声明：owner = 用户本人语料（短口语），非 owner = 公版中文书（书面）。
二者存在「领域/语体」混淆，本实验测得的可分性含这部分贡献；
领域匹配对照（其他人口语）本地无源，已列为后续。
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(8)

OWNER_PATH = "D:/work/_local_ai/train_15b/owner_corpus/owner_corpus.jsonl"
NONOWNER_DIR = "D:/work/_local_ai/train_15b/next_corpus_raw/gutenberg"


def chinese_ratio(s):
    if not s:
        return 0.0
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff") / len(s)


def load_owner(n, seed, min_len=10):
    rng = random.Random(seed)
    entries = []
    with open(OWNER_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("text") or "").strip()
            if len(t) >= min_len:
                entries.append(t)
    rng.shuffle(entries)
    return entries[:n]


def load_nonowner(n, seed):
    rng = random.Random(seed)
    books = sorted(str(p) for p in Path(NONOWNER_DIR).glob("*.txt"))
    start_re = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
    end_re = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
    passages = []
    for bp in books:
        try:
            raw = Path(bp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = start_re.search(raw)
        if m:
            raw = raw[m.end():]
        m2 = end_re.search(raw)
        if m2:
            raw = raw[:m2.start()]
        sents = [s.strip() for s in re.split(r"[。！？；\n]+", raw) if s.strip()]
        buf = ""
        for s in sents:
            if chinese_ratio(s) < 0.6:
                buf = ""
                continue
            if len(buf) + len(s) > 120:
                if 60 <= len(buf):
                    passages.append(buf)
                buf = s
            else:
                buf += s
        if 60 <= len(buf) <= 120:
            passages.append(buf)
    rng.shuffle(passages)
    return passages[:n]


def encode(model, tok, texts, bs=16, max_len=192):
    last_vecs, mean_vecs = [], []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
            h = out.last_hidden_state  # [B,T,H]，已过 final norm
            lens = enc["attention_mask"].sum(dim=1)
            for b in range(h.shape[0]):
                last_vecs.append(h[b, lens[b] - 1].float().numpy())
                mask = enc["attention_mask"][b].unsqueeze(-1).float()
                mean_vecs.append(((h[b] * mask).sum(0) / mask.sum()).float().numpy())
    return np.array(last_vecs), np.array(mean_vecs)


def participation_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    cov = np.cov(Xc.T)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def pair_cos(A, B, n_pairs, rng):
    i = rng.integers(0, len(A), size=n_pairs)
    j = rng.integers(0, len(A), size=n_pairs)
    k = rng.integers(0, len(B), size=n_pairs)
    l = rng.integers(0, len(B), size=n_pairs)
    aa = np.sum(A[i] * A[j], 1) / (np.linalg.norm(A[i], axis=1) * np.linalg.norm(A[j], axis=1) + 1e-9)
    ab = np.sum(A[k] * B[l], 1) / (np.linalg.norm(A[k], axis=1) * np.linalg.norm(B[l], axis=1) + 1e-9)
    return float(aa.mean()), float(ab.mean())


def bootstrap_gap(A, B, n_pairs=200, rounds=50, seed=0):
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(rounds):
        aa, ab = pair_cos(A, B, n_pairs, rng)
        gaps.append(aa - ab)
    gaps = np.array(gaps)
    return float(gaps.mean()), float(gaps.std())


def auc_score(y, s):
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def probe_cv_auc(X, y, folds=5, steps=300, lr=0.1, wd=1e-3, seed=0):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    aucs = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = (Xte @ w + b).numpy()
        aucs.append(auc_score(y[te], s))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", nargs="*", default=[
        "D:/work/_local_ai/models/SmolLM2-200M",
        "D:/work/_local_ai/models/SmolLM2-135M",
    ])
    ap.add_argument("--out",
                    default="D:/work/_local_ai/train_15b/eval/q14_stage0_result.md")
    args = ap.parse_args()

    owner = load_owner(args.n, args.seed)
    nonowner = load_nonowner(args.n, args.seed + 1)
    print(f"[data] owner={len(owner)} nonowner={len(nonowner)}", flush=True)
    print(f"[data] owner avg chars={np.mean([len(t) for t in owner]):.0f} "
          f"nonowner avg chars={np.mean([len(t) for t in nonowner]):.0f}", flush=True)

    y = np.array([1] * len(owner) + [0] * len(nonowner))
    rng = np.random.default_rng(args.seed)
    y_rand = y.copy()
    rng.shuffle(y_rand)

    lines = [
        "# Q-14 阶段 0：认主可分性上界（零训练）",
        "",
        f"- 数据：owner={len(owner)}（owner_corpus.jsonl，seed={args.seed}）｜"
        f"nonowner={len(nonowner)}（gutenberg 中文公版书 46 本，60–120 字段落）",
        f"- owner 均长 {np.mean([len(t) for t in owner]):.0f} 字｜"
        f"nonowner 均长 {np.mean([len(t) for t in nonowner]):.0f} 字",
        "- 混淆声明：owner 为口语短文本、nonowner 为书面公版书，"
        "可分性含「领域/语体」贡献；领域匹配对照待补其他人口语源。",
        "",
        "| 模型 | 嵌入 | PR(owner) | PR(nonowner) | cos(o,o) | cos(o,n) | gap | gap/σ | 探针AUC | NC AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for md in args.models:
        print(f"[model] loading {md}", flush=True)
        tok = AutoTokenizer.from_pretrained(md)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            model = AutoModel.from_pretrained(md, dtype=torch.float32)
        except TypeError:
            model = AutoModel.from_pretrained(md, torch_dtype=torch.float32)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {Path(md).name} params={n_params/1e6:.1f}M", flush=True)

        o_last, o_mean = encode(model, tok, owner)
        n_last, n_mean = encode(model, tok, nonowner)
        del model

        for tag, O, N in [("last", o_last, n_last), ("mean", o_mean, n_mean)]:
            X = np.concatenate([O, N], 0)
            pr_o, pr_n = participation_ratio(O), participation_ratio(N)
            aa, ab = pair_cos(O, N, 400, np.random.default_rng(args.seed))
            gap_mu, gap_sd = bootstrap_gap(O, N, seed=args.seed)
            auc_m, auc_sd = probe_cv_auc(X, y, seed=args.seed)
            nc_m, _ = probe_cv_auc(X, y_rand, seed=args.seed + 100)
            ratio = gap_mu / gap_sd if gap_sd > 1e-9 else float("inf")
            lines.append(
                f"| {Path(md).name} | {tag} | {pr_o:.1f} | {pr_n:.1f} | "
                f"{aa:.4f} | {ab:.4f} | {gap_mu:+.4f} | {gap_mu:+.1f}/{gap_sd:.4f} "
                f"= {ratio:.1f}σ | {auc_m:.4f}±{auc_sd:.3f} | {nc_m:.4f} |")
            print(f"[{Path(md).name}/{tag}] PR_o={pr_o:.1f} PR_n={pr_n:.1f} "
                  f"cos_oo={aa:.4f} cos_on={ab:.4f} gap={gap_mu:+.4f} "
                  f"({ratio:.1f}sigma) AUC={auc_m:.4f} NC={nc_m:.4f}", flush=True)

    lines += [
        "",
        "## 判据读取",
        "- FQ-14a：探针 AUC > 0.99 → 认主头无需训练，退化为阈值。",
        "- FQ-14b：gap ≥ 5σ → 间隔判据通过。",
        "- NC：打乱标签 AUC 应 ≈0.5；显著为正 → 两类样本在来源上可分（混淆），"
        "需领域匹配对照后再下结论。",
        "",
        "## 结论",
        "（见对话侧判读；本文件由 eval/q14_identity_separability.py 生成，可复跑。）",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
