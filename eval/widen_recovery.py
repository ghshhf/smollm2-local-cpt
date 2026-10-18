#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三值模型 Net2Net 加宽后「能否训回」验证
=======================================
上一轮实测: fp32 加宽 CE 跳变 0, 三值加宽 CE 跳变 +1.70%。
关键分歧: +1.70% 是「瞬时扰动」还是「永久损伤」?
  - 若是瞬时扰动(续训能回到加宽前水平甚至更低) -> 不封死, BitNet 可用
  - 若是永久损伤(续训回不去)                    -> 封死, 0.5B 不用 BitNet

流程(三值与 fp32 各一臂, 同种子同预算):
  warm(热身训练) -> CE_A
  Net2Net 加宽    -> CE_B   (跳变 = CE_B - CE_A)
  heal(恢复训练)  -> CE_C
判定: recovery = (CE_B - CE_C) / (CE_B - CE_A),  >=1.0 表示完全恢复并继续变好
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trial_1h import Net, widen_mlp, eval_ce, batches, build_tokens, log

def run(net, data, seq, bs, accum, lr, seconds, mask, tag):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    n = data.shape[0]
    torch.manual_seed(0)
    perm = torch.randperm(n)
    i = 0
    t0 = time.time()
    step = 0
    while time.time() - t0 < seconds:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if i + bs > n:
                perm = torch.randperm(n); i = 0
            idx = data[perm[i:i + bs]]
            i += bs
            out = net(idx, mask)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)),
                                   idx[:, 1:].reshape(-1))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
    return step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/work/_local_ai/train_15b/pretrain_corpus/zh_text_r5.jsonl")
    ap.add_argument("--tok", default="D:/work/_local_ai/models/SmolLM2-135M")
    ap.add_argument("--ntok", type=int, default=3_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warm", type=int, default=240)
    ap.add_argument("--heal", type=int, default=240)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok)
    V = len(tok)
    toks = build_tokens(args.corpus, tok, args.ntok, args.seq)
    data = batches(toks, args.seq, args.bs)
    nh = max(1, data.shape[0] // 20)
    train, hold = data[:-nh], data[-nh:]
    mask = torch.triu(torch.full((args.seq, args.seq), -1e9), 1)[None, None]

    out = {}
    for tag, ternary in (("fp32", False), ("ternary", True)):
        torch.manual_seed(1234)
        net = Net(V, 256, 1024, 8, 4, 2, ternary)
        log(f"=== {tag} ===")
        log(f"[{tag}] warm 开始"); s1 = run(net, train, args.seq, args.bs, args.accum, args.lr, args.warm, mask, tag)
        ceA = eval_ce(net, hold, args.seq, args.bs, mask)
        widen_mlp(net, 256)
        ceB = eval_ce(net, hold, args.seq, args.bs, mask)
        # 加宽后优化器需重建(参数形状变了)
        log(f"[{tag}] heal 开始"); s2 = run(net, train, args.seq, args.bs, args.accum, args.lr, args.heal, mask, tag)
        ceC = eval_ce(net, hold, args.seq, args.bs, mask)
        jump = ceB - ceA
        rec = (ceB - ceC) / jump if abs(jump) > 1e-9 else float("inf")
        msg = (f"[{tag}] 热身{s1}步 CE={ceA:.4f} | 加宽后 CE={ceB:.4f} "
               f"(跳变{jump:+.4f}) | 恢复{s2}步 CE={ceC:.4f} | "
               f"恢复率={rec:.2f}× | 最终 vs 加宽前: {ceC-ceA:+.4f}")
        log(msg)
        out[tag] = {"ce_before": ceA, "ce_after_widen": ceB, "jump": jump,
                    "ce_after_heal": ceC, "recovery": rec, "delta_vs_before": ceC - ceA,
                    "warm_steps": s1, "heal_steps": s2}

    print("\n" + "=" * 68)
    print("判定")
    print("=" * 68)
    for tag in ("fp32", "ternary"):
        r = out[tag]
        ok = r["ce_after_heal"] <= r["ce_before"] + 1e-6
        print(f"  {'✅' if ok else '❌'} {tag:<8s} 加宽前 {r['ce_before']:.4f} -> "
              f"跳变 {r['jump']:+.4f} -> 恢复后 {r['ce_after_heal']:.4f} "
              f"({'已恢复并继续改善' if ok else '未恢复, 属永久损伤'})")
    t = out["ternary"]
    print()
    if t["ce_after_heal"] <= t["ce_before"]:
        print("  ⇒ 三值加宽的 +1.7% 只是瞬时扰动, 续训即可恢复并继续下降。")
        print("  ⇒ 三值与「不封死」不冲突, 0.5B 可采用 BitNet b1.58。")
    else:
        print("  ⇒ 三值加宽造成永久损伤, 需走分块 gamma 方案, 否则 0.5B 不用 BitNet。")

    os.makedirs("eval", exist_ok=True)
    with open("eval/widen_recovery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写 eval/widen_recovery.json")

if __name__ == "__main__":
    main()
