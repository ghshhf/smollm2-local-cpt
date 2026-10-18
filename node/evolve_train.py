#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evolve_train.py — node1 自进化闭环的训练腿（在 node1 上跑，CPU 自训）

把 triple_bank.db 里过了双闸门的 (task, code_text) 训成动画域 LoRA。
🔴 协议同构铁律：训练样本的 prompt 必须 evolve 到与 dream_loop.build_prompt
逐字一致 —— 直接 import dream_loop 复用 SYSTEM / DREAM_BANK / build_prompt，
银行里没存 hint 就用 DREAM_BANK 查表，保证「训的就是推理时见的」。

历史教训（2026-10-18 代码 LoRA 负迁移）：通用问答语料 + lr 2e-4 会把
"写脚本落盘" 冲成 "写定义不调用"。本训练器对策：
  - 语料全部来自银行已过闸样本（天然是"可执行+产视频"形态）
  - lr 降到 1e-4、只 1 epoch、completion-only loss（不冲 system/user 协议）

用法(node1, D=/mnt/workspace/node1/dream):
  python3 -u evolve_train.py --out models/evolve_lora
退出码: 0=训完并落盘  2=样本不足(未达 --min-samples)  1=其他错误
"""
import argparse
import json
import os
import random
import sqlite3
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dream_loop import SYSTEM, DREAM_BANK, build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--db", default=os.path.join(HERE, "data", "triple_bank.db"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "evolve_lora"))
    ap.add_argument("--min-samples", type=int, default=24)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1408)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.set_num_threads(8)
    random.seed(a.seed)

    # ── 1) 读银行 + 按任务查 hint ─────────────────────────────────────
    hint_of = {e["task"]: e.get("hint", "") for e in DREAM_BANK}
    c = sqlite3.connect(a.db, timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    rows = c.execute("SELECT id, task, code_text FROM triples ORDER BY id").fetchall()
    c.close()

    seen, samples = set(), []
    for tid, task, code in rows:
        code = (code or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        samples.append((tid, task, code))
    random.shuffle(samples)
    samples = samples[: a.max_samples]

    print("[data] bank_triples=%d dedup=%d cap=%d -> n=%d" % (
        len(rows), len(seen), a.max_samples, len(samples)))
    if len(samples) < a.min_samples:
        print("[data] 样本不足(<%d)，本周期不训" % a.min_samples)
        return 2

    # ── 2) 构造同构样本: prompt=build_prompt(task,hint,[]) completion=围栏块+im_end ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    recs, skipped = [], 0
    for tid, task, code in samples:
        prompt = build_prompt(task, hint_of.get(task, ""), [])
        completion = "```python\n%s\n```\n<|im_end|>" % code
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt + completion, add_special_tokens=False)["input_ids"]
        if len(full_ids) > a.max_len or len(full_ids) <= len(p_ids):
            skipped += 1
            continue
        recs.append((tid, task, p_ids, full_ids))
    print("[data] usable=%d skipped_too_long=%d" % (len(recs), skipped))
    if len(recs) < a.min_samples:
        print("[data] 可用样本不足(<%d)，本周期不训" % a.min_samples)
        return 2

    # 溯源落盘
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "evolve_dataset_last.jsonl"), "w", encoding="utf-8") as f:
        for tid, task, p_ids, full_ids in recs:
            f.write(json.dumps({"triple_id": tid, "task": task,
                                "n_tokens": len(full_ids)}, ensure_ascii=False) + "\n")

    # ── 3) 载模型 + LoRA(q/k/v/o，遵守接口契约第0条：只挂恒定维) ──────────
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
    model.config.use_cache = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    # ── 4) 训练循环（batch=1，completion-only loss） ───────────────────
    order = list(range(len(recs)))
    n_epoch_steps = len(order)
    total_steps = int(n_epoch_steps * a.epochs)
    step, t0, loss_sum = 0, time.time(), 0.0
    done = False
    for ep in range(int(a.epochs) + 1):
        if done:
            break
        random.shuffle(order)
        for idx in order:
            if step >= total_steps:
                done = True
                break
            _, _, p_ids, full_ids = recs[idx]
            ids = torch.tensor([full_ids], dtype=torch.long)
            labels = torch.tensor([[-100] * len(p_ids) + full_ids[len(p_ids):]],
                                  dtype=torch.long)
            out = model(input_ids=ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            step += 1
            if step % 20 == 0 or step == total_steps:
                spd = (time.time() - t0) / step
                print("[train] step %d/%d loss=%.4f avg=%.4f %.1fs/step"
                      % (step, total_steps, float(loss.detach()), loss_sum / step, spd),
                      flush=True)

    print("[train] done steps=%d avg_loss=%.4f elapsed=%.0fs"
          % (step, loss_sum / max(step, 1), time.time() - t0))

    # ── 5) 落盘适配器 ──────────────────────────────────────────────────
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    print("[save] adapter -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
