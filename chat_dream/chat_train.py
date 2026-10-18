# -*- coding: utf-8 -*-
"""对话模块增量训练（GPU 版）。

为什么不用远程那份 train_lora.py
--------------------------------
那份是 0.2B / 无 CUDA 时期写的：`use_cpu=True`、`bf16=False`、`fp16=False`、fp32 CPU，
且 target_modules 还带着 MLP 三件套（gate/up/down）——而模块新配置明确只挂 q/k/v/o，
MLP 输出维随 intermediate_size 变化，0.3B 上（inter 4608）那批张量直接作废。
所以这里重写，配方按 module.card.json 的 training 段来：
    r=16 / alpha=32 / dropout=0.05 / targets=q,k,v,o / completion_only_loss

增量怎么增
----------
    --init-adapter 现有卡   -> 以它为起点继续训（不是从零），这就是「增量」
    数据 = 银行全部产出 + 教材锚点采样   -> 只喂新数据会灾难性遗忘，锚点用来压住

验收（照 module.card 的 acceptance）
-----------------------------------
训练前后各量一次 completion CE：in_set_ce < 0.60 才算「这条学会了」。
记训练前后的 CE 差，是判断这轮增量到底有没有用的唯一硬指标。

用法
----
    python chat_train.py --out out_chat_v2
    python chat_train.py --out out_chat_v3 --init-adapter out_chat_v2
"""
import argparse
import json
import os
import random
import sqlite3
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

IM_END = "<|im_end|>"
MARK = "<|im_start|>assistant\n"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def chatml(q, a):
    return f"<|im_start|>user\n{q}{IM_END}\n{MARK}{a}{IM_END}"


def encode(tok, text, maxlen):
    """切成 (ids, labels)：prompt 段 label 全 -100，只在 assistant 段算 loss。
    不开这个掩码，模型会学成「什么都往下续写」，推理时模仿用户输入。"""
    k = text.find(MARK)
    if k < 0:
        ids = tok(text, add_special_tokens=False).input_ids[:maxlen]
        return ids, list(ids)
    pids = tok(text[: k + len(MARK)], add_special_tokens=False).input_ids
    cids = tok(text[k + len(MARK):], add_special_tokens=False).input_ids
    ids = (pids + cids)[:maxlen]
    lab = ([-100] * len(pids) + cids)[:maxlen]
    return ids, lab


def collate(batch, pad_id, device):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long, device=device)
    lab = torch.full((len(batch), L), -100, dtype=torch.long, device=device)
    am = torch.zeros((len(batch), L), dtype=torch.long, device=device)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a, device=device)
        lab[i, : len(b)] = torch.tensor(b, device=device)
        am[i, : len(a)] = 1
    return ids, lab, am


def load_bank(db):
    """读银行，并在训练前用闸门再筛一遍。

    这一步是血的教训：银行里有 48 条带 BPE 残渣的样本（`queeze` / `<jupyter_output>` /
    `kinesisfirehose`）漏进来后被训了 3 epoch，产出的 adapter_r2_n174 从此疯狂吐代码
    token，一轮 96 题里 58 条拒收全是残渣，通过率从 23% 掉到 12%。
    脏数据只要训一轮就能把模型带偏，所以训练前必须自己再验一次，不能只信入库时的判定
    （入库判定会随闸门升级而变化，历史数据不会自动重判）。
    """
    c = sqlite3.connect(db)
    rows = c.execute("select q,a from pairs order by id").fetchall()
    out, dropped = [], 0
    try:
        import chat_dream as d
        for q, a in rows:
            if not q.strip() or not a.strip():
                continue
            if not (d.g_script(q, a)[0] and d.g_degenerate(q, a)[0] and d.g_garble(q, a)[0]):
                dropped += 1
                continue
            out.append((q, a))
        if dropped:
            print(f"[data] 闸门复核丢弃 {dropped} 条脏样本（不参与训练）")
        return out
    except Exception as e:
        print(f"[data] 闸门复核不可用（{type(e).__name__}: {e}），按原样使用 {len(rows)} 条")
        return [(q, a) for q, a in rows if q.strip() and a.strip()]


def load_seeds(path, k, rng):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            kk = t.find(MARK)
            if kk < 0:
                continue
            q = t[:kk]
            j = q.rfind("<|im_start|>user")
            q = (q[j + len("<|im_start|>user"):] if j >= 0 else q).replace(IM_END, "").strip()
            a = t[kk + len(MARK):].replace(IM_END, "").strip()
            if "```" in a or "```" in q or len(a) < 2:
                continue
            rows.append((q, a))
    rng.shuffle(rows)
    return rows[:k] if k else rows


@torch.no_grad()
def measure_ce(model, tok, pairs, maxlen, device, tag=""):
    model.eval()
    tot, ntok, per = 0.0, 0, []
    for q, a in pairs:
        ids, lab = encode(tok, chatml(q, a), maxlen)
        n = sum(1 for x in lab if x != -100)
        if n == 0:
            continue
        out = model(input_ids=torch.tensor([ids], device=device),
                    labels=torch.tensor([lab], device=device))
        tot += float(out.loss) * n
        ntok += n
        per.append(float(out.loss))
    if not ntok:
        return 0.0, 0.0
    ce = tot / ntok
    under = sum(1 for x in per if x < 0.6)
    print(f"[CE{tag}] 均值={ce:.4f}  ppl={2.718281828 ** ce:.2f}  达标(<0.60)={under}/{len(per)}")
    return ce, under / max(1, len(per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="G:/_models/llm/polaris-03b")
    ap.add_argument("--bank", default="G:/_tmp/data/chat_bank.db")
    ap.add_argument("--seeds", default="G:/_tmp/polaris_test/chat_sft_public.jsonl")
    ap.add_argument("--init-adapter", default=None, help="增量起点（已有卡），不传=从零训")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=320)
    ap.add_argument("--anchor-ratio", type=float, default=0.25,
                    help="旧数据占训练集的比例。银行是累积的，主要防「变体问法把原始问法盖掉」。"
                         "社区经验 replay 占 10-20%%（再多会稀释新信号）")
    ap.add_argument("--val-n", type=int, default=30,
                    help="固定验证集条数，从教材里抽、训练时永不见。只降训练 loss 说明不了任何事")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    vals_rng = random.Random(999)   # 验证集用固定种子，保证每轮同一批，CE 才可比
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] 无 CUDA，回退 CPU（会很慢）")
    print(f"[info] device={device} | torch={torch.__version__}")

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bank = load_bank(a.bank)
    all_seeds = load_seeds(a.seeds, 0, rng)
    # 固定验证集：先从教材抽出来，往后训练永远不碰它。
    # 只降训练 loss 什么都说明不了（梯度下降的本职），过拟合的信号是训练降、验证升。
    vals_rng.shuffle(all_seeds)
    val = all_seeds[: a.val_n]
    pool = all_seeds[a.val_n:]
    # 锚点条数按占比反推：anchor/(bank+anchor) = anchor_ratio
    n_anchor = int(len(bank) * a.anchor_ratio / max(1e-6, 1 - a.anchor_ratio))
    n_anchor = max(30, min(200, n_anchor))
    anchors = pool[:n_anchor]
    # 关系型种子强制进训练集：情感（被主人带情绪时怎么接）与指令澄清（被纠正/放权时怎么应）
    # 是本项目的主攻方向，靠随机锚点抽太容易被漏（38+33 条散在 345 条里）。
    emo = []
    try:
        _c = sqlite3.connect(a.bank)
        emo = [(q, x) for q, x in _c.execute(
            "select q,a from seeds where topic in ('emotion','clarify')").fetchall()
               if q.strip() and x.strip()]
        _c.close()
    except Exception:
        pass
    data = bank + anchors + emo
    if emo:
        print(f"[data] 关系型种子强制入训 {len(emo)} 条（emotion + clarify）")
    if not data:
        print("[ERR] 没数据可训")
        return
    rng.shuffle(data)
    print(f"[data] 银行 {len(bank)} + 锚点 {len(anchors)}"
          f"（占训练集 {len(anchors)/max(1,len(data))*100:.0f}%）= {len(data)} 条"
          f" | 固定验证集 {len(val)} 条")

    print(f"[info] 加载主干 {a.base}（fp32, {device}）")
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False

    if a.init_adapter and os.path.isdir(a.init_adapter):
        from peft import PeftModel
        print(f"[info] 增量起点（已有卡）: {a.init_adapter}")
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        model.to(device)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        print(f"[info] 从零起 LoRA r=16 alpha=32 targets={TARGETS}")
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                         lora_dropout=0.05, target_modules=TARGETS, bias="none")
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    print("\n[baseline] 训练前")
    ce0, rate0 = measure_ce(model, tok, bank, a.maxlen, device, "-前 银行in-set")
    vce0, vrate0 = measure_ce(model, tok, val, a.maxlen, device, "-前 验证集")

    enc = [encode(tok, chatml(q, x), a.maxlen) for q, x in data]
    enc = [e for e in enc if any(t != -100 for t in e[1])]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps_per_ep = max(1, (len(enc) + a.batch - 1) // a.batch)
    total = steps_per_ep * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    pad_id = tok.pad_token_id

    model.train()
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        rng.shuffle(enc)
        run = []
        for i in range(0, len(enc), a.batch):
            ids, lab, am = collate(enc[i: i + a.batch], pad_id, device)
            out = model(input_ids=ids, attention_mask=am, labels=lab)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run.append(float(loss))
            if step % 10 == 0:
                print(f"  ep{ep+1} step {step}/{total} loss={sum(run[-10:])/len(run[-10:]):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
            if a.save_steps and step % a.save_steps == 0:
                model.save_pretrained(f"{a.out}_ckpt{step}")
        v, _ = measure_ce(model, tok, val, a.maxlen, device, f"-ep{ep+1} 验证集")
        model.train()
        print(f"  [ep{ep+1}] 训练 loss={sum(run)/len(run):.4f} | 验证 CE={v:.4f}"
              f"{'  <-- 验证在涨，开始过拟合' if ep > 0 and v > 0 else ''}")

    print("\n[after] 训练后")
    ce1, rate1 = measure_ce(model, tok, bank, a.maxlen, device, "-后 银行in-set")
    vce1, vrate1 = measure_ce(model, tok, val, a.maxlen, device, "-后 验证集")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"\n[done] 已保存 {a.out}  耗时 {time.time()-t0:.0f}s")
    print(f"[验收] in-set  CE {ce0:.4f} -> {ce1:.4f}  达标率 {rate0*100:.0f}% -> {rate1*100:.0f}%"
          f"   判据<0.60 : {'PASS' if ce1 < 0.60 else 'NOT YET'}")
    overfit = "是" if vce1 > vce0 else "否"
    print(f"[验收] 验证集 CE {vce0:.4f} -> {vce1:.4f}  达标率 {vrate0*100:.0f}% -> {vrate1*100:.0f}%"
          f"   过拟合={overfit}")


if __name__ == "__main__":
    main()
