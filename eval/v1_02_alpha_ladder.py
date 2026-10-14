"""
V1-02·α 阶梯对照实验（零训练，4 次前向）
=====================================
用途：一次性切开 Q1 的两种可能解释——"三值化静态冲击太大"vs "训练不够"。

背景（本轮实测得到的关键事实）：
- V1-01 探针 200 步 @ lr=1e-5，flip 恒为 0.0000
- 实测量化：边界带内权重占 34~46%（翻转物理可行），但 200 步位移仅量化步长的
  0.004%~0.031% ⇒ **200 步预算内物理上不可能观测到符号翻转**
- 同时 α从 1.0 线性退火到 0.0，而权重几乎没动
  ⇒ loss 从 0.29 涨到 2.29 是 **"把能用的 fp32 表示原地换成未经训练的纯三值表示"** 的必然代价

所以真正的未知只剩一个：**静态三值化冲击本身有多大？**
- 若 α=0 的 holdout CE 暴涨→ 继承式失败主因是"退火太快，来不及补偿"（可修：慢退火 / 只三值FFN / 提预算）
- 若 α=0 的 CE 与 fp32 接近 → 模型其实能承受三值化，V1-01 的 loss 暴涨另有原因

设计：固定权重不动，扫 α 阶梯，各测一次 holdout CE。
零优化步、零梯度，纯前向 ⇒ 本机 12 线程几分钟内出结果。
"""

import os, sys, json, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DEFAULT = os.environ.get("SMOL_BASE", os.path.join(ROOT, "models", "SmolLM2-200M"))
BASE = os.environ.get("SMOL_BASE", BASE_DEFAULT)
CKPT = os.environ.get("SMOL_CKPT", os.path.join(ROOT, "out_smollm2_135m_pt", "checkpoint-latest", "pytorch_model.bin"))
INTER = 1536
EVAL_TOKENS = 200_000

# α 阶梯：重点是 1.0(应完全等价 fp32) 和 0.0(纯三值)
ALPHAS = [1.0, 0.9, 0.7, 0.4, 0.2, 0.0]


def load_fp32_model():
    """从 checkpoint 自身反推 config（135M 是裸 ckpt，与 SmolLM2-200M 底座的
    intermediate_size 不同：底座 3072，135M 为 1536。硬编码底座会全层 size mismatch）"""
    from transformers import AutoConfig, AutoModelForCausalLM
    sd = torch.load(CKPT, map_location="cpu")

    def infer(keys, default):
        for k in keys:
            if k in sd:
                return sd[k].shape[-1]
        return default

    cfg = AutoConfig.from_pretrained(BASE)
    cfg.max_position_embeddings = INTER

    # ---- 全部结构参数从 ckpt 的真实 shape 反推（135M 与 200M 底座结构不同：
    #      hidden=576 / inter=1536 / q=576(9头x64) / kv=192(3头x64)）
    q = sd["model.layers.0.self_attn.q_proj.weight"]
    k = sd["model.layers.0.self_attn.k_proj.weight"]
    g = sd["model.layers.0.mlp.gate_proj.weight"]
    cfg.hidden_size = q.shape[1]                      # 576
    cfg.intermediate_size = g.shape[0]                # 1536（gate_proj 是 [inter, hidden]）
    cfg.num_hidden_layers = 1 + max(int(kk.split(".")[2]) for kk in sd
                                     if kk.startswith("model.layers.") and kk.endswith(".self_attn.q_proj.weight"))
    # head_dim=64（SmolLM2 血统）；由 q/k 的行数整除定头数
    for hd in (64, 128, 32):
        if q.shape[0] % hd == 0 and k.shape[0] % hd == 0:
            cfg.num_attention_heads = q.shape[0] // hd      # 9
            cfg.num_key_value_heads = k.shape[0] // hd      # 3
            break
    try:
        cfg.head_dim = q.shape[0] // cfg.num_attention_heads
    except Exception:
        pass
    try:
        cfg.sliding_window = None
    except Exception:
        pass

    print(f"[cfg] hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"inter={cfg.intermediate_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"head_dim={getattr(cfg,'head_dim','?')}")

    m = AutoModelForCausalLM.from_config(cfg)
    try:
        missing, unexpected = m.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        print(f"[FATAL] load_state_dict 失败（结构仍不匹配）：{str(e)[:400]}")
        raise
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    # 硬校验：主干线性层必须全部加载成功（否则 CE 无意义）
    bad = [k for k in missing if any(s in k for s in ["q_proj", "k_proj", "v_proj", "o_proj",
                                                    "gate_proj", "up_proj", "down_proj"])]
    if bad:
        print(f"[FATAL] {len(bad)} 个主干层未加载，放弃评测：{bad[:4]}")
        raise SystemExit(3)
    for k in list(missing)[:4]:
        print(f"        missing(非主干): {k}")
    m.eval()
    return m


def quantize(w):
    with torch.no_grad():
        s = w.abs().mean().clamp(min=1e-5)
        return torch.clamp(torch.round(w / s), -1.0, 1.0), s


class TernaryView(nn.Module):
    """不改动原模型：按 alpha 把 Linear 权重替换为 W_q + alpha*(W - W_q) 的等效矩阵做前向"""

    def __init__(self, model, alpha):
        super().__init__()
        self.model = model
        self.alpha = float(alpha)

    def forward(self, *a, **kw):
        # 通过临时替换 weight.data 实现，finally 恢复
        saved = []
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                q, _ = quantize(w)
                saved.append((m, w))
                m.weight.data = q + self.alpha * (w - q)
        try:
            return self.model(*a, **kw)
        finally:
            for m, w in saved:
                m.weight.data = w


@torch.no_grad()
def holdout_ce(model, tok, bs=8, max_batches=40):
    """在中文 holdout 上算交叉熵（每 token 平均）"""
    nll, ntok = 0.0, 0
    seq = INTER
    nb = 0
    for i in range(0, len(tok) - seq - 1, seq):
        if nb >= max_batches:
            break
        ids = torch.tensor(tok[i:i + seq], dtype=torch.long).unsqueeze(0)
        out = model(ids).logits
        lp = torch.nn.functional.cross_entropy(
            out[:, :-1].reshape(-1, out.shape[-1]).float(),
            ids[:, 1:].reshape(-1))
        ntok += ids[:, 1:].numel()
        nll += lp.item() * ids[:, 1:].numel()
        nb += 1
    return nll / max(ntok, 1), nb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "v1_02_alpha_ladder.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    print("=" * 74)
    print("V1-02 alpha 阶梯对照：固定权重，扫 alpha，各测 holdout CE（零训练）")
    print("=" * 74)

    tok = load_tokenized_holdout()
    print(f"[data] holdout tokens={len(tok):,}")

    t0 = time.time()
    model = load_fp32_model()

    # 先测 fp32 基线（原样）
    base_ce, nb = holdout_ce(model, tok)
    print(f"\n[alpha=--]  fp32 原模型          CE={base_ce:.4f}  ({nb} batches, {time.time()-t0:.0f}s)")

    rows = [dict(alpha=None, ce=round(base_ce, 4), label="fp32 baseline")]
    ce_map = {"fp32": base_ce}

    for al in ALPHAS:
        t1 = time.time()
        view = TernaryView(model, al)
        ce, _ = holdout_ce(view, tok)
        ce_map[f"{al}"] = ce
        d = ce - base_ce
        rows.append(dict(alpha=al, ce=round(ce, 4), delta=round(d, 4)))
        print(f"[alpha={al:.1f}]  三值化插值          CE={ce:.4f}  delta={d:+.4f}  ({time.time()-t1:.0f}s)")

    # ---- 裁决 ----
    pure = ce_map.get("0.0", base_ce)
    delta_pure = pure - base_ce
    verdicts = {
        "static_shock_small(d<0.3)": delta_pure < 0.3,
        "static_shock_moderate(0.3<=d<1.2)": 0.3 <= delta_pure < 1.2,
        "static_shock_large(d>=1.2)": delta_pure >= 1.2,
        "monotonic_in_alpha": all(ce_map[f"{ALPHAS[i]}"] >= ce_map[f"{ALPHAS[i+1]}"] - 1e-6
                                  for i in range(len(ALPHAS) - 1)),
    }
    print("\n" + "=" * 74)
    print("裁决")
    print("=" * 74)
    for k, v in verdicts.items():
        print(f"  {'PASS' if v else 'no  '}  {k}")
    print(f"\n  纯三值(alpha=0)相对 fp32: delta={delta_pure:+.4f}")
    if delta_pure >= 1.2:
        print("  => 静态冲击很大：三值化本身就是主要障碍，继承式失败根因是")
        print("     '退火太快+权重没动，来不及补偿'。可修路径：慢退火 / 只三值FFN / 大幅提预算。")
    elif delta_pure < 0.3:
        print("  => 静态冲击很小：模型权重本就能承受三值化，V1-01 的 loss 暴涨另有原因")
        print("     （需查 lr 尺度 / STE 实现 / 数据分布，不在'三值化固有代价'上）。")
    else:
        print("  => 静态冲击中等：三值化代价真实但可补偿，需找该走多长的退火路径。")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, verdicts=verdicts,
                       delta_pure=round(delta_pure, 4),
                       model="SmolLM2-135M (inherit, NO training)",
                       tokens=len(tok), threads=a.threads), f, indent=2, ensure_ascii=False)
    print(f"\n[out] {a.out}")


def load_tokenized_holdout():
    """复用已有中文 holdout 语料；若不存在则现读一份训练语料当代理"""
    cands = [
        os.path.join(ROOT, "eval", "holdout_zh.json"),
        os.path.join(ROOT, "data", "holdout_zh.json"),
    ]
    for c in cands:
        if os.path.exists(c):
            d = json.load(open(c, encoding="utf-8"))
            if isinstance(d, dict) and "input_ids" in d:
                return d["input_ids"][:EVAL_TOKENS]
            if isinstance(d, list):
                return d[:EVAL_TOKENS]
    # 兜底：从语料文件现读
    from transformers import AutoTokenizer
    print("[data] 未找到预存 holdout，改从语料现取一段（评测独立性会被记为 caveat）")
    tk = AutoTokenizer.from_pretrained(BASE)
    import glob
    for pat in [os.path.join(ROOT, "data", "**", "*.jsonl"), os.path.join(ROOT, "**", "zh*.jsonl")]:
        for fp in sorted(glob.glob(pat, recursive=True))[:20]:
            try:
                ids = []
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            o = json.loads(line)
                        except Exception:
                            continue
                        t = o.get("text") or o.get("content") or o.get("text_raw")
                        if not t:
                            continue
                        ids += tk(t, add_special_tokens=False)["input_ids"]
                        if len(ids) >= EVAL_TOKENS:
                            break
                if len(ids) > 4096:
                    return ids[:EVAL_TOKENS]
            except Exception:
                continue
    raise SystemExit("找不到可用中文语料做 holdout")


if __name__ == "__main__":
    main()