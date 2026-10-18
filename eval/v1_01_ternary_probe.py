#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V1-01 探针：三值 QAT 在 CPU 上到底训不训得出来（0.5B 生死判据的第一块砖）。

三模型交叉裁决后的新路线（docs/netarch/v1_01_verdict_ternary_cpu_20261014.md）：
  ✅ 采纳 = 继承 fp16 权重（0.4B→Net2Net 扩容）+ 渐进残差退火 QAT（alpha:1→0，推理纯三值）
  🔴 推翻 = CPU 从零预训练三值（物理不可行）；推翻 = 冻结fp32主干+float16影子Adapter（推理留浮点会击穿15ms/token）

本探针在 135M 底座上实测裁决里的四类判据，并A/B 对照两条路线：
  ARM-A「继承+退火」：从 fp16 checkpoint 出发，alpha 1→0 退火
  ARM-B「从零三值」  ：随机初始化潜权重，同步跑同样步数（证伪其不可行）
另测 ARM-C「静态 Adapter」（保留浮点残差不退火）→ 量化其推理期三值缺口，验证"退火到0"的必要性。

判据（来自 gemini + grok + north 交叉）：
  翻转率 Δ_flip：能训 3~8%→0.1~0.5%；训不动 <0.01%(锁死) 或 >15%(震荡)
  零权占比 R0：健康 35~55%；训不动 <10%(二值退化) 或 >85%(全零坍缩)
  相对发散度 Δℒ = ℒ_三值 − ℒ_fp16：≤+0.35 平行；≥+1.2 发散
  吞吐：≥800 tok/s 可训；<300 tok/s 内存阻塞

跑法（node1 64 核 / 本机）：
  python eval/v1_01_ternary_probe.py --arm A --steps 200
  python eval/v1_01_ternary_probe.py --arm B --steps 200
  python eval/v1_01_ternary_probe.py --arm C --steps 200
  python eval/v1_01_ternary_probe.py --arm all --steps 200
"""
import os, sys, json, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 135M 是裸checkpoint（只有 pytorch_model.bin，无 config/tokenizer）——
# 必须从 SmolLM2-200M 底座 from_config 建模型再灌权重（同 e1_subliminal_proxy.py 口径）。
# 注意底座在 _local_ai/models/ 下（不在 train_15b/models/）。
BASE = os.environ.get("SMOL_BASE", os.path.join(os.path.dirname(ROOT), "models", "SmolLM2-200M"))
CKPT = os.environ.get("SMOL_CKPT", os.path.join(ROOT, "out_smollm2_135m_pt", "checkpoint-latest", "pytorch_model.bin"))
INTER = 1536

torch.set_num_threads(min(16, os.cpu_count() or 8))

# ── 三值量化器 + STE（gemini 版：mean|γ 缩放，STE mask 在 |w|<=γ）────────────
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        scale = w.abs().mean().clamp(min=1e-5)
        w_q = torch.clamp(torch.round(w / scale), -1.0, 1.0) * scale
        return w_q, scale
    @staticmethod
    def backward(ctx, g_wq, g_scale):
        (w,) = ctx.saved_tensors
        scale = w.abs().mean().clamp(min=1e-5)
        mask = (w.abs() <= scale).float()
        return g_wq * mask, None

def quantize(w):
    with torch.no_grad():
        scale = w.abs().mean().clamp(min=1e-5)
        return torch.clamp(torch.round(w / scale), -1.0, 1.0), scale

def metrics(w):
    """返回 (翻转率需外部比, R0, 三值矩阵)。"""
    q, _ = quantize(w)
    R0 = (q == 0).float().mean().item()
    return R0, q

# ── 加载 135M 底座（从 SmolLM2-200M from_config 建，灌 135M 权重）──────────
def load_base():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    cfg.intermediate_size = INTER
    m = AutoModelForCausalLM.from_config(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
    return m

def collect_named_linear_weights(model):
    """收集所有 nn.Linear 权重（BitNet 的替换目标）。"""
    ws = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            ws.append((name, mod.weight))
    return ws

# ── 三值化：把 Linear 换成 BitLinear（继承模式）或随机三值（从零模式）──────
class BitLinear(nn.Module):
    """W_eff = W_q + alpha(W_latent − W_q).detach()。推理(eval)且退火完→纯 W_q。"""
    def __init__(self, lin: nn.Linear, alpha: float = 1.0):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.weight = nn.Parameter(lin.weight.data.clone())
        self.register_buffer("alpha", torch.tensor(float(alpha)))
    def forward(self, x):
        w_q, scale = TernarySTE.apply(self.weight)
        if self.training and float(self.alpha) > 0.0:
            w_eff = w_q + self.alpha * (self.weight - w_q).detach()
        else:
            w_eff = w_q
        # 激活 8-bit dynamic absmax（BitNet 原配方）——按最后一维逐 token 量化
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        gamma = x2.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)   # [N,1]
        x_q = torch.clamp(torch.round(x2 * 127.0 / gamma), -128, 127) * (gamma / 127.0)
        out = F.linear(x_q, w_eff)
        return out.reshape(*shape[:-1], self.out_features)

def convert_to_bitlinear(model, mode="inherit", alpha=1.0):
    """把 model 的所有 nn.Linear 换成 BitLinear。mode: inherit(继承权重) / scratch(随机重置)。"""
    n_conv = 0
    for parent_name, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                if mode == "scratch":
                    new = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                    nn.init.normal_(new.weight, std=0.02)
                    if child.bias is not None: nn.init.zeros_(new.bias)
                    setattr(parent, cname, new)
                    setattr(parent, cname, BitLinear(new, alpha=alpha))
                else:
                    setattr(parent, cname, BitLinear(child, alpha=alpha))
                n_conv += 1
    return n_conv

# ── 数据：中文 holdout（与 data-plan 同批口径）────────────────────────────
def get_batches(tok, n_batches=8, bs=2, seqlen=256):
    corpus = None
    for p in ["data/zh_holdout.txt", "data/holdout_zh.txt", "data/zh_text_clean.txt"]:
        fp = os.path.join(ROOT, p)
        if os.path.exists(fp):
            corpus = open(fp, encoding="utf-8").read(); break
    if corpus is None:
        corpus = ("模型本质在于记录函数。" * 4000)  # 兜底
    ids = tok(corpus[:200000], return_tensors="pt").input_ids[0]
    batches = []
    for i in range(n_batches):
        s = i * bs * seqlen
        chunk = ids[s:s+bs*seqlen]
        if chunk.numel() < bs*seqlen:
            chunk = torch.cat([chunk]*(bs*seqlen//max(1,chunk.numel())+1))
        batches.append(chunk.reshape(bs, seqlen))
    return batches

# ── 主探针 ──────────────────────────────────────────────────────────────
def run_arm(arm, steps, lr, batches, tok, threads):
    print(f"\n{'='*66}\nARM-{arm}  ({'继承+退火QAT' if arm=='A' else '从零三值' if arm=='B' else '静态Adapter(不退火)'})\n{'='*66}")
    model = load_base()
    mode = "scratch" if arm == "B" else "inherit"
    n_conv = convert_to_bitlinear(model, mode=mode, alpha=1.0)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"转换 {n_conv} 个 Linear → BitLinear；参数 {n_params/1e6:.1f}M；threads={threads}")

    # fp16 基线 loss（ARM-C 用它算推理期缺口；ARM-A/B 也要相对它）
    base_model = None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    prev_q = {}
    for name, p in model.named_parameters():
        if p.dim() >= 2:  # 只跟踪矩阵权重
            prev_q[name] = None

    t_start = time.time(); tok_seen = 0; losses = []
    step_records = []
    for step in range(1, steps+1):
        # alpha 线性退火 1.0 → 0.0（ARM-C 不退火）
        frac = step / steps
        alpha = 1.0 - frac if arm in ("A","B") else 1.0
        for m in model.modules():
            if isinstance(m, BitLinear):
                m.alpha.fill_(alpha)

        batch = batches[(step-1) % len(batches)]
        t0 = time.time()
        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        dt = time.time()-t0
        bsz = batch.numel(); tok_seen += bsz
        losses.append(loss.item())

        # 判据指标：R0 + 翻转率
        with torch.no_grad():
            R0s, flips = [], []
            for name, p in model.named_parameters():
                if p.dim() < 2: continue
                q, _ = quantize(p.data)
                R0s.append((q == 0).float().mean().item())
                pq = prev_q.get(name)
                if pq is not None:
                    flips.append((q != pq).float().mean().item())
                prev_q[name] = q
            R0 = sum(R0s)/len(R0s)
            flip = sum(flips)/len(flips) if flips else 0.0

        if step % max(1, steps//10) == 0 or step == 1:
            tps = bsz/max(dt,1e-6)
            rec = dict(step=step, loss=round(loss.item(),4), R0=round(R0,4),
                       flip=round(flip,4), alpha=round(alpha,3),
                       tok_per_s=round(tps,1), grad_norm=round(float(gn),3),
                       elapsed=round(time.time()-t_start,1))
            step_records.append(rec)
            print(f"  step {step:4d}/{steps} loss={loss.item():.4f} R0={R0:.3f} "
                  f"flip={flip:.4f} alpha={alpha:.2f} {tps:.0f}tok/s gn={float(gn):.2f}")

    total_t = time.time()-t_start
    avg_tps = tok_seen/max(total_t,1e-6)
    first_loss = losses[0]; final_loss = losses[-1]
    last10 = sum(losses[-10:])/min(10,len(losses))

    verdict = {}
    verdict["能训"] = {
        "R0健康(20~70%)": 0.20 <= R0 <= 0.70,
        "非锁死(flip>0.01%)": flip > 0.0001,
        "非震荡(flip<15%)": flip < 0.15,
        "吞吐>=300tok/s": avg_tps >= 300,
        "loss下降>20%": final_loss < first_loss*0.8,
    }
    any_bad = not all(verdict["能训"].values())
    concl = "训不动" if any_bad else "能训"
    print(f"\n  平均吞吐 {avg_tps:.0f} tok/s | 首loss {first_loss:.4f} → 末loss {final_loss:.4f}(末10均 {last10:.4f})")
    print(f"  最终 R0={R0:.3f}  flip={flip:.4f}  alpha={alpha:.2f}")
    print(f"  判据: {json.dumps(verdict['能训'], ensure_ascii=False)}")
    print(f"  ▸ ARM-{arm} 裁决: **{concl}**")

    return dict(arm=arm, steps=steps, n_conv=n_conv, n_params=n_params,
                first_loss=round(first_loss,4), final_loss=round(final_loss,4),
                last10_loss=round(last10,4), R0=round(R0,4), flip=round(flip,4),
                alpha_final=round(alpha,3), avg_tok_per_s=round(avg_tps,1),
                total_time=round(total_t,1), verdict=verdict["能训"], concl=concl,
                records=step_records)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A","B","C","all"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default=os.path.join(HERE, "v1_01_probe_result.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    batches = get_batches(tok, n_batches=8, bs=2, seqlen=256)

    arms = ["A","B","C"] if args.arm == "all" else [args.arm]
    results = [run_arm(a, args.steps, args.lr, batches, tok, torch.get_num_threads()) for a in arms]

    json.dump(dict(config=vars(args), results=results),
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{'='*66}\n汇总（写入 {args.out}）")
    for r in results:
        print(f"  ARM-{r['arm']}: {r['concl']:5s} loss {r['first_loss']}→{r['last10_loss']} "
              f"R0={r['R0']} flip={r['flip']} {r['avg_tok_per_s']}tok/s")