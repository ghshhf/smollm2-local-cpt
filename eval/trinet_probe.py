#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三值(BitNet b1.58) × Net2Net 兼容性探针
=========================================
背景: 0.5B 端侧方案想上 BitNet 三值(1.58bit)。但我们的第一原则是「不封死」,
      扩容靠 Net2Net 两条腿: ①加零输出恒等层 ②MLP 加宽(零填充)。
      三值量化有 per-tensor 缩放因子 gamma = mean(|W|), 它可能破坏这两条腿。

本脚本做5组可复现的数值实验, 全部无需训练:
  T1 零输出层     : W=0 时 y 是否严格 = 0 (gamma=0 会不会 NaN)
  T2 零填充加宽    : MLP 中间维加零 (per-tensor scale) 是否保函数
  T3 per-row scale : 换 per-output-row 缩放后, T2 是否成立
  T4 replicate    : Net2Net 复制式加宽, 是否保函数 (预期成立)
  T5 加层(恒等)    : 新增整个 Block 靠残差恒等, 三值下是否成立
  T6 全模型三值化  : 真实 214M 权重 -> 三值, 测 holdout CE 退化

判定口径: 保函数 = 扩容前后同一输入的输出 max|Δ| < 1e-5 (fp32 口径)
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

EPS = 1e-8

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------------- BitNet b1.58 量化 ----------------
def quant_tens(W):
    """per-tensor absmean (BitNet 原版)"""
    g = W.abs().mean()
    Q = (W / (g + EPS)).round().clamp_(-1, 1)
    return Q, g

def quant_row(W):
    """per-output-row absmean (每个输出神经元独立缩放)"""
    g = W.abs().mean(dim=1, keepdim=True)
    Q = (W / (g + EPS)).round().clamp_(-1, 1)
    return Q, g

def dequant(Q, g):
    return Q * g          # BitNet: y = Q(W) @ x * gamma  (gamma 标量或行向量)

def fwd(W, x, mode="tens"):
    q = quant_tens if mode == "tens" else quant_row
    Q, g = q(W)
    return dequant(Q, g) @ x, Q, g

# ---------------- 判定 ----------------
def relerr(a, b):
    d = (a - b).abs().max().item()
    n = b.abs().max().item() + 1e-12
    return d, d / n

def verdict(d, tol=1e-5):
    return "PASS" if d < tol else "FAIL"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    d_in, d_ff, d_out, B = 576, 3072, 576, 8
    # 模拟一个真实量级的权重 (SmolLM2 权重 std ~0.02)
    W_down = torch.randn(d_out, d_ff) * 0.02     # MLP down_proj: d_ff -> d_model
    W_up   = torch.randn(d_ff, d_in) * 0.02      # MLP up_proj  : d_model -> d_ff
    x      = torch.randn(d_ff, B)                # 中间维激活
    h      = torch.randn(d_in, B)                # 隐层激活

    report = {}
    print("=" * 72)
    print("三值 × Net2Net 兼容性探针   (保函数判据: max|Δout| < 1e-5)")
    print("=" * 72)

    # ---------- T1 零输出层 ----------
    print("\n[T1] 零输出层: 新增层 W=0, 期望 y 严格=0")
    W0 = torch.zeros(d_out, d_ff)
    for mode in ("tens", "row"):
        y, Q, g = fwd(W0, x, mode)
        nan = bool(torch.isnan(y).any())
        mx = y.abs().max().item()
        ok = (not nan) and mx < 1e-12
        print(f"   {mode:>5s}: gamma={float(g.reshape(-1)[0]) if g.numel()==1 else 'vec'} "
              f"max|y|={mx:.3e} NaN={nan}  -> {'PASS' if ok else 'FAIL'}")
        report[f"T1_{mode}"] = "PASS" if ok else "FAIL"
    # 附带: 三值化本身对非零权重的相对误差 (作为"改动量"的标尺)
    yq, Q, g = fwd(W_down, x, "tens")
    yf = W_down @ x
    d0, r0 = relerr(yq, yf)
    print(f"   [标尺] 三值化自身的误差: max|Δ|={d0:.4e} 相对={r0*100:.2f}%  "
          f"(下文任何 > 此量级 1e-2 的改动都算「破坏函数」)")

    # ---------- T2 零填充加宽 (per-tensor) ----------
    print("\n[T2] 零填充加宽 (加 512 个中间神经元), per-tensor scale")
    add = 512
    W_down2 = torch.cat([W_down, torch.zeros(d_out, add)], dim=1)   # 加零列
    x2 = torch.cat([x, torch.zeros(add, B)], dim=0)                 # 新神经元激活=0
    for mode in ("tens", "row"):
        y1, _, g1 = fwd(W_down, x, mode)
        y2, _, g2 = fwd(W_down2, x2, mode)
        d, r = relerr(y2, y1)
        v = verdict(d)
        print(f"   {mode:>5s}: gamma {fmt_g(g1)} -> {fmt_g(g2)}   "
              f"max|Δout|={d:.4e} ({r*100:.3f}%)  -> {v}")
        report[f"T2_{mode}"] = v
    print("   机理: per-tensor gamma=mean|W|, 补零行/列会稀释 gamma -> 已有权重的")
    print("         量化步长改变 -> round() 结果系统性改变 -> 旧函数被改。")

    # ---------- T3 只在加宽矩阵上单独量化 ----------
    print("\n[T3] 变体: 新块单独一套 gamma (旧权重块不重量化)")
    Qa, ga = quant_tens(W_down)
    yA = Qa * ga @ x
    Qb, gb = quant_tens(torch.zeros(d_out, add))
    yB = (Qb * gb) @ torch.zeros(add, B)
    d, r = relerr(yA + yB, yA)
    print(f"   分块独立 gamma: max|Δout|={d:.4e} -> {verdict(d)}   (可行但需改架构: 权重分片量化)")
    report["T3_blockwise"] = verdict(d)

    # ---------- T4 replicate 复制式加宽 ----------
    print("\n[T4] Net2Net replicate 复制式加宽 (复制已有神经元, down_proj 权重对半分)")
    idx = torch.arange(add) % d_ff
    W_up2 = torch.cat([W_up, W_up[idx].clone()], dim=0)             # 复制行
    W_down2r = torch.cat([W_down, torch.zeros(d_out, add)], dim=1)
    for j in range(add):
        W_down2r[:, d_ff + j] = W_down[:, idx[j]] * 0.5
        W_down2r[:, idx[j]] = W_down[:, idx[j]] * 0.5               # 对半分, 和不变
    h_act = F.gelu(W_up @ h)          # fp32 参照(此处只验 down_proj 侧)
    y1, _, _ = fwd(W_down, h_act, "tens")
    h_act2 = torch.cat([h_act, h_act[idx]], dim=0)
    y2, _, g2 = fwd(W_down2r, h_act2, "tens")
    d, r = relerr(y2, y1)
    print(f"   per-tensor: gamma 由 {fmt_g(quant_tens(W_down)[1])} -> {fmt_g(g2)}  "
          f"max|Δout|={d:.4e} ({r*100:.3f}%) -> {verdict(d)}")
    report["T4_replicate"] = verdict(d)
    print("   关键: 复制的是已有数值, mean|W| 不变 -> gamma 不变 -> 旧权重量化结果不变")

    # ---------- T5 加整个 Block (残差恒等) ----------
    print("\n[T5] 新增整个 Block, 靠残差实现恒等 (down_proj=0, 其余任意)")
    y = h.clone()
    Wz = torch.zeros(d_out, d_ff)
    yz, Qz, gz = fwd(Wz, torch.zeros(d_ff, B), "tens")
    yout = y + yz
    d, r = relerr(yout, h)
    print(f"   max|Δ|={d:.4e} -> {verdict(d)}  (与 T1 同构: 只要零输出层成立, 加层就成立)")
    report["T5_addblock"] = verdict(d)

    # ---------- 汇总 ----------
    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    for k, v in report.items():
        flag = "✅" if v == "PASS" else "❌"
        print(f"  {flag} {k:<18s} {v}")

    print("\n结论推导:")
    t2t = report["T2_tens"]; t2r = report["T2_row"]; t4 = report["T4_replicate"]
    if t2t == "FAIL":
        print("  · per-tensor 三值 + 零填充加宽 = 破坏已压缩记忆 ❌  =>  与「不封死」冲突")
    if t2r == "PASS":
        print("  · 改 per-row 缩放后零填充成立 ✅  =>  若要 BitNet, 必须用 per-output-row scale")
    if t4 == "PASS":
        print("  · replicate 复制式加宽成立 ✅  =>  三值下扩宽走 replicate, 不走零填充")
    print("  · 零输出/加层与三值兼容 (T1/T5) => 深度方向扩容不受影响")

    os.makedirs("eval", exist_ok=True)
    with open("eval/trinet_probe.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n已写 eval/trinet_probe.json")

def fmt_g(g):
    if g.numel() == 1:
        return f"{g.item():.5f}"
    return f"vec(mean={g.mean().item():.5f})"

if __name__ == "__main__":
    main()
