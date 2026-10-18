#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一小时本地试验: 「密度夯实」 + 三值(BitNet)训练可行性 + Net2Net 扩容扰动
=========================================================================
背景(用户原话):
  "相当于 0.1 0.1 的不断走上最高...把每一个 0.1 级的密度夯实到极致, 才会进击下一轮突破"
  "压缩算法不只有内部压缩, 还有输出压缩、输入压缩"

本机 12 核 CPU, 1 小时算力预算 ≈ 1.1e15 FLOPs, 物理上不可能把 214M 训出能力。
所以本试验不追求"能力提升", 而是回答三个能指导 0.5B 架构决策的问题:

  Q1 三值(BitNet b1.58)量化感知训练(QAT)在本机是否稳定收敛? 相比 fp32 差多少?
  Q2 三值模型做 Net2Net 扩容时, 扰动有多大? (fp32 对照组应 ≈ 0)
      -> 直接检验「不封死」第一原则在三值下是否还成立
  Q3 夯实曲线的形状: loss 随 token 数下降的密度, 给出外推判据

设计: 同架构 ~20M 代理模型两臂对照 (fp32 / 三值QAT), 同 token 预算, 同数据, 同种子。
      最后对两臂各做一次 Net2Net 加宽(中间维 1024->1280), 测扩容瞬间 loss 跳变。
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ============================ 模型 ============================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class BitLinear(nn.Module):
    """BitNet b1.58: 三值 {-1,0,1} + per-tensor absmean 缩放, STE 直通估计"""
    def __init__(self, in_f, out_f, eps=1e-8):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.W, 0, 0.02)
        self.eps = eps
    def quant(self):
        W = self.W
        g = W.abs().mean()
        Q = (W / (g + self.eps)).round().clamp_(-1, 1)
        Q = W + (Q - W).detach()          # STE: 前向用 Q, 反传梯度直穿
        return Q * g
    def forward(self, x):
        return F.linear(x, self.quant())

def make_linear(in_f, out_f, ternary):
    return BitLinear(in_f, out_f) if ternary else nn.Linear(in_f, out_f, bias=False)

class Block(nn.Module):
    def __init__(self, d, dff, nh, nkv, ternary):
        super().__init__()
        self.nh, self.nkv, self.d = nh, nkv, d
        self.hd = d // nh
        L = lambda i, o: make_linear(i, o, ternary)
        self.q, self.k, self.v, self.o = L(d, d), L(d, d // (nh // nkv)), L(d, d // (nh // nkv)), L(d, d)
        self.up, self.gate, self.down = L(d, dff), L(d, dff), L(dff, d)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd) + mask
        y = torch.matmul(att.softmax(-1), v).transpose(1, 2).reshape(B, T, self.d)
        x = x + self.o(y)
        h2 = self.n2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x

def rope(t, cos, sin):
    # t: (B, nh, T, hd)
    hd = t.shape[-1]
    c, s = cos[:, :, :t.shape[2], :hd], sin[:, :, :t.shape[2], :hd]
    t2 = torch.cat([-t[..., hd // 2:], t[..., :hd // 2]], dim=-1)
    return t * c + t2 * s

class Net(nn.Module):
    def __init__(self, vocab, d, dff, nl, nh, nkv, ternary):
        super().__init__()
        self.d, self.dff, self.nl, self.ternary = d, dff, nl, ternary
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, 0, 0.02)
        self.blocks = nn.ModuleList([Block(d, dff, nh, nkv, ternary) for _ in range(nl)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight        # tied
        self.register_buffer("cos", None, persistent=False)
    def forward(self, idx, mask=None):
        B, T = idx.shape
        x = self.emb(idx)
        pos = torch.arange(T, device=idx.device).float()
        hd = self.d // (self.blocks[0].nh)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=idx.device).float() / hd))
        ang = torch.outer(pos, inv)                       # (T, hd/2)
        c, s = ang.cos(), ang.sin()
        cos = torch.cat([c, c], -1)[None, None]           # (1,1,T,hd) 标准 RoPE
        sin = torch.cat([s, s], -1)[None, None]
        if mask is None:
            mask = torch.triu(torch.full((T, T), -1e9, device=idx.device), 1)[None, None]
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.head(self.nf(x))

def widen_mlp(net, add):
    """Net2Net 加宽: MLP 中间维 dff -> dff+add (零填充)。
       fp32 下严格保函数(新神经元激活=0)。三值下 gamma 被稀释 -> 有扰动。"""
    d, old = net.d, net.dff
    new = old + add
    for b in net.blocks:
        for nm in ("up", "gate"):
            W = getattr(b, nm).W if net.ternary else getattr(b, nm).weight
            W2 = torch.zeros(new, d)
            W2[:old].copy_(W.data)
            W.data = W2
        Wd = b.down.weight if not net.ternary else b.down.W
        W2 = torch.zeros(d, new)
        W2[:, :old].copy_(Wd.data)
        Wd.data = W2
    net.dff = new
    return net

# ============================ 数据 ============================
def build_tokens(corpus, tok, ntok, seq):
    import json as _j
    buf = []
    got = 0
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = _j.loads(line).get("text", line)
            except Exception:
                t = line
            if len(t) < 8:
                continue
            ids = tok.encode(t, add_special_tokens=False)
            buf.extend(ids)
            buf.append(tok.eos_token_id)
            got += len(ids) + 1
            if got >= ntok:
                break
    log(f"语料 token 数: {len(buf):,}")
    return torch.tensor(buf[: (len(buf) // seq) * seq], dtype=torch.long)

def batches(toks, seq, bs):
    n = toks.numel() // seq
    x = toks[: n * seq].view(n, seq)
    return x

# ============================ 训练一臂 ============================
def train_arm(net, data, seq, bs, accum, lr, seconds, tag, holdout, logf):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    n = data.shape[0]
    torch.manual_seed(0)
    t0 = time.time(); tlog = time.time()
    step = 0; toks = 0
    curve = []
    mask = torch.triu(torch.full((seq, seq), -1e9), 1)[None, None]
    perm = torch.randperm(n)
    i = 0
    while True:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if i + bs > n:
                perm = torch.randperm(n); i = 0
            idx = data[perm[i:i + bs]]
            i += bs
            out = net(idx, mask)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), idx[:, 1:].reshape(-1))
            (loss / accum).backward()
            toks += idx.numel()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 10 == 0 or time.time() - t0 >= seconds:
            el = time.time() - t0
            ce = eval_ce(net, holdout, seq, bs, mask)
            msg = (f"[{tag}] step {step:>5d}  tok={toks/1e6:.2f}M  "
                   f"train_loss={loss.item():.4f}  holdout_CE={ce:.4f}  {el:.0f}s  "
                   f"({toks/max(el,1):.0f} tok/s)")
            log(msg); logf.write(msg + "\n"); logf.flush()
            curve.append({"step": step, "tokens": toks, "train_loss": loss.item(),
                          "holdout_ce": ce, "sec": el})
            tlog = time.time()
        if time.time() - t0 >= seconds:
            break
    return {"tag": tag, "steps": step, "tokens": toks, "sec": time.time() - t0,
            "curve": curve, "final_ce": curve[-1]["holdout_ce"] if curve else None}

@torch.no_grad()
def eval_ce(net, holdout, seq, bs, mask, max_b=8):
    net.eval()
    tot, cnt = 0.0, 0
    for i in range(0, min(holdout.shape[0], bs * max_b), bs):
        idx = holdout[i:i + bs]
        out = net(idx, mask)
        l = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), idx[:, 1:].reshape(-1))
        tot += l.item(); cnt += 1
    net.train()
    return tot / max(cnt, 1)

# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="D:/work/_local_ai/train_15b/pretrain_corpus/zh_text_r5.jsonl")
    ap.add_argument("--tok", default="D:/work/_local_ai/models/SmolLM2-135M")
    ap.add_argument("--out", default="D:/work/_local_ai/train_15b/trial_1h")
    ap.add_argument("--arm_seconds", type=int, default=1080, help="每臂训练秒数(默认 18 分钟)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--nl", type=int, default=8)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--nkv", type=int, default=2)
    ap.add_argument("--ntok", type=int, default=8_000_000)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    logf = open(os.path.join(args.out, "trial_log.txt"), "a", encoding="utf-8")
    log("线程数:", args.threads)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok)
    V = len(tok)
    log("词表:", V)

    all_toks = build_tokens(args.corpus, tok, args.ntok, args.seq)
    data = batches(all_toks, args.seq, args.bs)
    # holdout: 最后 5% 不参与训练
    nh = max(1, data.shape[0] // 20)
    train, hold = data[:-nh], data[-nh:]
    log(f"训练块 {train.shape[0]}  holdout 块 {hold.shape[0]}  "
        f"({train.numel()/1e6:.2f}M token)")

    res = {}
    for tag, ternary in (("fp32", False), ("ternary", True)):
        torch.manual_seed(1234)
        net = Net(V, args.d, args.dff, args.nl, args.nh, args.nkv, ternary)
        p = sum(x.numel() for x in net.parameters())
        log(f"--- {tag} 参数量 {p/1e6:.1f}M (含 tied embedding {V*args.d/1e6:.1f}M) ---")
        # 扩容前基线 CE
        mask = torch.triu(torch.full((args.seq, args.seq), -1e9), 1)[None, None]
        ce0 = eval_ce(net, hold, args.seq, args.bs, mask)
        log(f"[{tag}] 初始 holdout_CE = {ce0:.4f}")
        r = train_arm(net, train, args.seq, args.bs, args.accum, args.lr,
                      args.arm_seconds, tag, hold, logf)
        r["params"] = p
        # ---- Net2Net 加宽扰动测试 ----
        ce_before = eval_ce(net, hold, args.seq, args.bs, mask)
        widen_mlp(net, 256)
        ce_after = eval_ce(net, hold, args.seq, args.bs, mask)
        jump = ce_after - ce_before
        msg = (f"[{tag}] Net2Net 加宽 dff {args.dff}->{args.dff+256}: "
               f"CE {ce_before:.4f} -> {ce_after:.4f}  跳变 = {jump:+.4f} "
               f"({jump/max(ce_before,1e-9)*100:+.2f}%)")
        log(msg); logf.write(msg + "\n"); logf.flush()
        r["ce_before_widen"] = ce_before
        r["ce_after_widen"] = ce_after
        r["widen_jump"] = jump
        r["widen_jump_pct"] = jump / max(ce_before, 1e-9) * 100
        res[tag] = r

    # ================= 报告 =================
    lines = []
    A = lines.append
    A("# 一小时本地试验报告: 密度夯实 / 三值可行性 / Net2Net 扩容扰动\n")
    A(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- 硬件: 本机 CPU {args.threads} 线程 (CUDA 不可用)")
    A(f"- 代理模型: hidden={args.d} layers={args.nl} heads={args.nh}/{args.nkv} "
      f"dff={args.dff} seq={args.seq} tied-emb vocab={V}")
    A(f"- 数据: {os.path.basename(args.corpus)} 前 {args.ntok/1e6:.0f}M token, "
      f"训练 {train.numel()/1e6:.2f}M / holdout {hold.numel()/1e6:.2f}M\n")
    A("## 一、两臂训练对照\n")
    A("| 指标 | fp32 | 三值 BitNet b1.58 |")
    A("|---|---|---|")
    for k, lab in (("params", "参数量(M)"), ("steps", "优化器步数"), ("tokens", "消耗 token"),
                   ("sec", "耗时(s)"), ("final_ce", "最终 holdout CE")):
        f = res["fp32"].get(k); t = res["ternary"].get(k)
        fmt = (lambda v: f"{v/1e6:.1f}") if k == "params" else (
            lambda v: f"{v/1e6:.2f}M" if k == "tokens" else (
                lambda v: f"{v:.4f}" if k == "final_ce" else (lambda v: f"{v:.0f}" if k == "sec" else str(v))))
        A(f"| {lab} | {fmt(f)} | {fmt(t)} |")
    A("")
    A("## 二、Net2Net 加宽扰动(「不封死」核心检验)\n")
    A("| 臂 | 加宽前 CE | 加宽后 CE | 跳变 | 跳变% |")
    A("|---|---|---|---|---|")
    for tag in ("fp32", "ternary"):
        r = res[tag]
        A(f"| {tag} | {r['ce_before_widen']:.4f} | {r['ce_after_widen']:.4f} | "
          f"{r['widen_jump']:+.4f} | {r['widen_jump_pct']:+.2f}% |")
    A("")
    fp_j = abs(res["fp32"]["widen_jump"]); tn_j = abs(res["ternary"]["widen_jump"])
    A("**判定:**")
    A(f"- fp32 臂跳变 {fp_j:.4f} (理论应严格 0, 非零即为实现误差)")
    A(f"- 三值臂跳变 {tn_j:.4f}")
    if tn_j > max(fp_j, 1e-3) * 3:
        A(f"- ❌ 三值下 Net2Net 加宽产生 {tn_j/max(fp_j,1e-6):.0f}× 于 fp32 的扰动 "
          "→ 与「不封死」冲突, 需分块 gamma 或改 per-row scale")
    else:
        A(f"- ✅ 三值下加宽扰动与 fp32 同量级 → 三值与 Net2Net 兼容")
    A("")
    A("## 三、夯实曲线(loss vs token)\n")
    A("```")
    A(f"{'step':>6s} {'fp32_CE':>9s} {'tern_CE':>9s} {'fp32_tok':>10s} {'tern_tok':>10s}")
    n = min(len(res["fp32"]["curve"]), len(res["ternary"]["curve"]))
    for i in range(n):
        a = res["fp32"]["curve"][i]; b = res["ternary"]["curve"][i]
        A(f"{a['step']:>6d} {a['holdout_ce']:>9.4f} {b['holdout_ce']:>9.4f} "
          f"{a['tokens']/1e6:>9.2f}M {b['tokens']/1e6:>9.2f}M")
    A("```\n")
    A("> 注: 两臂均为**极度欠训练**状态(每参数不足 0.2 token, Chinchilla 要求 ~20)。")
    A("> 本曲线的价值在于**两臂同预算下的相对斜率**, 不在于绝对 loss。\n")

    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    log("报告已写:", os.path.join(args.out, "report.md"))
    logf.close()

if __name__ == "__main__":
    main()
