#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4 System 1 —— 直答蒸馏 (答案-only loss)  [v0.1 脚手架]
========================================================
阶段 A 数据管线: 读 {question, answer, difficulty} JSONL,
把 prompt+answer 拼成一条, 仅对 answer token 算 LM loss (prompt 部分 label=-100)。
这是标准 SFT 式蒸馏, 先训稳定的 System 1, 不碰递归/路由。

复用本项目 CPU fp32 训练范式 (参考 train_smollm2_pt.py)。
⚠️ 脚手架: 单卡 CPU、固定 lr、含梯度累积与分块续训钩子占位; 未含 router/latent 分支。
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class QADistillDataset(Dataset):
    """
    JSONL: {"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}
    prompt 模板: "问题：{q}\n答案："
    标签: prompt 段全 -100, 仅 answer+一张 EOS 参与 loss。
    """

    PROMPT_TMPL = "问题：{q}\n答案："

    def __init__(self, path, tokenizer, max_len=512):
        self.tok = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q, a = d.get("question"), d.get("answer")
                if not q or not a:
                    continue
                self.examples.append((q, a))
        log(f"蒸馏样本数: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        q, a = self.examples[idx]
        prompt = self.PROMPT_TMPL.format(q=q)
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        a_ids = self.tok.encode(a, add_special_tokens=False) + [self.tok.eos_token_id]
        input_ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        # 左对齐补 pad
        pad = self.max_len - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [self.tok.pad_token_id] * pad
            labels = labels + [-100] * pad
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * (self.max_len - pad) + [0] * pad, dtype=torch.long),
        }


def collate(batch):
    return {
        k: torch.stack([b[k] for b in batch])
        for k in ("input_ids", "labels", "attention_mask")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True, help="蒸馏 JSONL 路径")
    ap.add_argument("--output_dir", default="D:/work/_local_ai/train_15b/b4/out_distill")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true", default=False)
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    open(os.path.join(a.output_dir, "train.pid"), "w").write(str(os.getpid()))

    try:
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass
    device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.train()
    if a.compile:
        try:
            model = torch.compile(model, backend="inductor", dynamic=True)
            log("torch.compile OK")
        except Exception as e:
            log("torch.compile 失败:", str(e)[:120])

    ds = QADistillDataset(a.data, tok, a.max_len)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, drop_last=True)
    opt = AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    start_step = 0
    ckpt = os.path.join(a.output_dir, "checkpoint-latest")
    if a.resume and os.path.isdir(ckpt):
        sd = torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(sd)
        try:
            start_step = int(open(os.path.join(ckpt, "step.txt")).read().strip())
        except Exception:
            pass
        log("续训 from step", start_step)

    total_steps = a.max_steps or (len(loader) // a.grad_accum * a.epochs)
    warm = a.warmup_steps

    def lr_lambda(s):
        if s < warm:
            return s / max(1, warm)
        p = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, p))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.to(device)

    log(f"开始直答蒸馏: max_steps={total_steps} lr={a.lr} accum={a.grad_accum}")
    micro = 0
    opt_step = start_step
    t0 = time.time()
    model.zero_grad()
    while opt_step < total_steps:
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                opt_step += 1
                if opt_step % 5 == 0 or opt_step == total_steps:
                    el = time.time() - t0
                    log(f"step {opt_step:>5d}/{total_steps} loss={out.loss.item():.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e} {el:.0f}s")
                    t0 = time.time()
                if opt_step % a.save_steps == 0:
                    os.makedirs(ckpt, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(ckpt, "pytorch_model.bin"))
                    open(os.path.join(ckpt, "step.txt"), "w").write(str(opt_step))
                if opt_step >= total_steps:
                    break
        if opt_step >= total_steps:
            break

    os.makedirs(ckpt, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt, "pytorch_model.bin"))
    open(os.path.join(ckpt, "step.txt"), "w").write(str(opt_step))
    model.save_pretrained(a.output_dir)
    tok.save_pretrained(a.output_dir)
    log("直答蒸馏完成 ->", a.output_dir)


if __name__ == "__main__":
    main()
