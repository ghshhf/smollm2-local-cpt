#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sft_full.py — 北极星 0.3B 恒维 SFT（只更新注意力四投影，主干其余冻结）

层级定位（2026-10-19 校正）:
  训练栈是 预训练 -> (CPT) -> **SFT** -> 偏好对齐(DPO/RL) -> 领域适配器。
  本脚本站在**第 3 层 SFT**：用成对的 (指令, 标准答案) 教能力。
  过去把第 5 层的工具（LoRA 领域适配器）拿去教能力，是层级错位 —— 只学会了壳。

更新方式（不是全参，也不是外挂旁路）:
  只把主干里**恒定维**的那部分参数原地改掉: q_proj / k_proj / v_proj / o_proj。
  embedding、词表、MLP 宽维全部冻结 —— 那些是知识所在，本轮不动。
  依据 docs/dream-to-module-loop-2026-10.md 的接口契约: 模块/更新只在恒维上表达，
  不碰可扩维。自进化环怎么转，这里就怎么接着转，只是把"外挂适配器"换成"原地写恒维"。

为什么不是全参:
  1) 意义: 全参是拿一小撮种子拽着整个主干走，前科是把指令能力冲成复读；
  2) 算力: 346.9M 全参的梯度+优化器状态本机扛不住；
  3) 可逆: 恒维可以按块留档、逐块 A/B 回滚，全参改完没法拆。

配方依据(不是拍脑袋):
  SmolLM2 官方(arXiv 2502.02737): SFT lr=3.0e-4, global batch=128, 2 epochs;
    且 135M/360M 用的是**过滤掉复杂难样本的 SmolTalk** —— 小模型配小数据。
  SmolTulu(arXiv 2412.08347): 小模型"学习率/批大小"比值调高更利于推理能力。
  => 本项目取 lr=2e-4(介于官方 3e-4 与线性缩放的 1e-4 之间), 有效 batch=32, 2 epochs。

内存(本机 16GB, 空闲约 6GB):
  参数 fp32 1.39G + 梯度 1.39G + Adafactor 行/列统计 ~0.02G
  + 梯度检查点后激活 ~0.2G  =>  峰值约 3.2~4.5G, 可跑。
  (用 AdamW 光优化器状态就 2.78G, 会顶到 7G+, 所以这里必须 Adafactor)

用法:
  python tools/sft_full.py --data data/card_code*.jsonl --epochs 2
  python tools/sft_full.py --data data/card_code.jsonl --steps 20      # 测速
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 项目自用模板, 与 build_chat_sft.py 完全一致(SmolLM2 ChatML, 1/2 号 token)
def wrap(q, a):
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n%s<|im_end|>" % (SYSTEM, q.strip(), a.strip()))

# 🔴 必须与 tools/code_eval.py 的 build_prompt 逐字同构（含 system 段）。
#    2026-10-19 踩坑：训练侧漏了 system 段，评测侧有 —— 同一份权重在评测时看到的
#    开头是训练时没见过的，20 题里 18 题连代码块都吐不出来，全是题干复读。
SYSTEM = "你是北极星，主人的私有编程助手。回答给可运行的代码，配一两句关键说明。"
PROMPT_PREFIX = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n")


def load_rows(patterns, limit=0):
    rows = []
    seen = set()
    dup = 0
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            n = 0
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                q = o.get("instruction") or o.get("prompt") or o.get("question") or ""
                a = o.get("output") or o.get("response") or o.get("answer") or ""
                if q and a:
                    h = hashlib.md5((q.strip() + a.strip()).encode("utf-8")).hexdigest()
                    if h in seen:      # 子卡与母集大量重复, 必须去重
                        dup += 1
                        continue
                    seen.add(h)
                    rows.append((q, a))
                    n += 1
            print("[data] %-30s %6d" % (os.path.basename(f), n))
    if limit:
        rows = rows[:limit]
    print("[data] total %d (%d 条重复已丢弃)" % (len(rows), dup))
    return rows


def encode(tok, q, a, max_len):
    """返回 (input_ids, labels)，label 只对 assistant 段计 loss"""
    p_ids = tok(PROMPT_PREFIX % q, add_special_tokens=False)["input_ids"]
    a_ids = tok(a + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    if len(ids) > max_len:
        ids = ids[:max_len]
    if len(ids) <= len(p_ids):          # 回答被截没了, 丢掉
        return None
    lab = [-100] * min(len(p_ids), len(ids)) + ids[min(len(p_ids), len(ids)):]
    return ids, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="backups/0.3B/polaris-03b_20261018")
    ap.add_argument("--tok", default="out_lora_chat_v7")
    ap.add_argument("--data", nargs="+", default=["data/card_code*.jsonl"])
    ap.add_argument("--out", default="models/sft_code_03b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条(0=全部)")
    ap.add_argument("--max-steps", type=int, default=0, help="硬上限(测速用)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--accum", type=int, default=32, help="梯度累积, 有效 batch = accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--threads", default="auto",
                    help="CPU 线程: 数字, 或 auto=现场标定有效核数(默认)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    a = ap.parse_args()

    # 自适应线程：按现场标定的有效核数开，而不是按 nproc / cgroup 配额开。
    # 2026-10-19 实测 node1：配额 8 核但有效算力 3.1~3.2 核（宿主机超卖），
    # 开 7 线程比开 3 线程慢 16%；而闲时可能真给到 8 核，所以必须每次现场标定。
    try:
        if str(a.threads).strip().lower() in ("auto", "0", ""):
            if os.path.join(ROOT, "tools") not in sys.path:
                sys.path.insert(0, os.path.join(ROOT, "tools"))
            from cpu_probe import effective_cores
            a.threads = effective_cores()
            print("[cpu] 自适应线程 = %d (现场标定)" % a.threads)
        else:
            a.threads = int(a.threads)
    except Exception as e:
        a.threads = 4
        print("[cpu] 标定失败(%r)，退用 %d 线程" % (e, a.threads))
    torch.set_num_threads(a.threads)
    torch.manual_seed(7)

    # 自适应设备唯一入口（2026-10-19）：cuda > mps > cpu，换机器不改参数
    try:
        from dev_pick import device as _dev_pick
        device = _dev_pick()
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device] %s%s" % (device, (" · " + torch.cuda.get_device_name(0)) if device == "cuda" else ""))

    rows = load_rows(a.data, a.limit)
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, a.tok))
    assert len(tok) == 49152, "tokenizer 词表不对: %d" % len(tok)

    # 预编码, 丢弃超长/空样本
    enc = []
    for q, rr in rows:
        e = encode(tok, q, rr, a.max_len)
        if e:
            enc.append(e)
    print("[data] 可用 %d 条, 平均 %d token" % (len(enc), sum(len(x[0]) for x in enc) // max(1, len(enc))))

    # dtype 参数名兼容: transformers >=4.56 用 dtype=, 老版只认 torch_dtype=
    # (node1 魔搭实例内置老版 transformers, 直接传 dtype= 会 TypeError 崩在加载权重那一步)
    try:
        m = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, a.llm), dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, a.llm), torch_dtype=torch.float32)
    m = m.to(device)
    m.train()
    if (not a.no_grad_ckpt) and device == "cpu":
        # 🔴 2026-10-19 远程踩坑（node1 崩在 backward）:
        #    gradient_checkpointing 默认 use_reentrant=True，它要求"至少一个输入 requires_grad"
        #    才会为检查点段建图。本脚本只解冻 q/k/v/o，第一层的输入是冻结的 embedding，
        #    input_ids 也不带梯度 => 检查点段不建图 => loss 没有 grad_fn =>
        #    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        #    本机旧 torch 侥幸不崩，远程新版必崩。统一改 use_reentrant=False（参数梯度照常回传）。
        try:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:      # 老版 transformers 不认这个 kwarg
            m.gradient_checkpointing_enable()
            if hasattr(m, "enable_input_require_grads"):
                m.enable_input_require_grads()
        m.config.use_cache = False
    use_amp = device == "cuda"

    # --- 恒维锁定：只解冻注意力四投影，其余全部冻结 ---
    TRAIN_PAT = re.compile(r"(^|\.)(q_proj|k_proj|v_proj|o_proj)\.")
    for n_, p_ in m.named_parameters():
        p_.requires_grad = bool(TRAIN_PAT.search(n_))
    trainable = [p_ for p_ in m.parameters() if p_.requires_grad]
    if not trainable:
        raise SystemExit("[fatal] 没匹配到任何 q/k/v/o 参数，检查模型结构")

    n_param = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in trainable)
    print("[model] %.1fM 参数, 可训练 %.1fM (%.1f%%) — 恒维 q/k/v/o"
          % (n_param / 1e6, n_train / 1e6, 100.0 * n_train / n_param))
    print("[model] 冻结 %d 个张量, 解冻 %d 个, 例: %s"
          % (len(list(m.parameters())) - len(trainable), len(trainable),
             ", ".join(n_ for n_, p_ in m.named_parameters() if p_.requires_grad)[:110]))

    opt = Adafactor(trainable, lr=a.lr, scale_parameter=False, relative_step=False,
                    warmup_init=False, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(enc) / a.accum)
    total = int(round(steps_per_epoch * a.epochs))
    if a.max_steps:
        total = min(total, a.max_steps)
    print("[plan] %d 步 (每 epoch %d 步, 有效 batch %d)" % (total, steps_per_epoch, a.accum))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total - a.warmup)
        return a.lr * 0.1 + (a.lr * 0.9) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    order = list(range(len(enc)))
    torch.manual_seed(7)
    import random
    random.shuffle(order)
    ptr = 0
    t0 = time.time()
    tok_seen = 0
    run_loss = 0.0
    run_n = 0

    for step in range(total):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(a.accum):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            ids, lab = enc[order[ptr]]
            ptr += 1
            t = torch.tensor([ids], dtype=torch.long, device=device)
            l = torch.tensor([lab], dtype=torch.long, device=device)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(input_ids=t, labels=l)
            else:
                out = m(input_ids=t, labels=l)
            loss = out.loss / a.accum
            if step == 0 and not loss.requires_grad:
                raise SystemExit(
                    "[fatal] loss 不带 grad_fn —— 可训练参数没进计算图。"
                    "最常见原因: gradient_checkpointing 的 use_reentrant=True 且首层输入无梯度。")
            loss.backward()
            loss_sum += loss.item()
            tok_seen += len(ids)
            run_loss += loss.item()
            run_n += 1
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if (step + 1) % a.log_every == 0 or step == 0:
            el = time.time() - t0
            print("[step %4d/%4d] loss=%.4f  lr=%.2e  %.1fs (%.2fs/步, %d tok/s)"
                  % (step + 1, total, run_loss / max(1, run_n), lr, el, el / (step + 1),
                     tok_seen / max(1e-6, el)), flush=True)
            run_loss = 0.0
            run_n = 0

        if (step + 1) % a.save_every == 0 or (step + 1) == total:
            p = os.path.join(ROOT, a.out)
            m.save_pretrained(p, safe_serialization=True)
            tok.save_pretrained(p)
            with open(os.path.join(p, "sft_meta.json"), "w", encoding="utf-8") as g:
                json.dump({"step": step + 1, "total": total, "lr": a.lr, "accum": a.accum,
                           "max_len": a.max_len, "data": a.data, "epochs": a.epochs,
                           "rows": len(enc), "optimizer": "Adafactor", "full_param": False,
                           "targets": "q_proj,k_proj,v_proj,o_proj",
                           "trainable_M": round(n_train / 1e6, 2),
                           "total_M": round(n_param / 1e6, 2)},
                          g, ensure_ascii=False, indent=1)
            print("[save] %s @ step %d" % (p, step + 1), flush=True)

    print("[done] %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
