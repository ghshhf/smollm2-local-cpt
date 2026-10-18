#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4 Stage B —— 固定步数潜在递归训练 (System 2 训练骨架)  [v0.1]
================================================================
对齐强模型 2026-10-15 给的 Stage B 方案 + 本机已跑通的推理接口
(latent_reasoner.py 的 inputs_embeds 回灌 + DynamicCache 累积)。

关键设计 (与强模型骨架一致, 并修两类正确性坑):
  - 只训练外挂 LatentAdapter (恒等初始化), 冻结 backbone + lm_head。
  - 潜在步骤之间不做词表投影; 只在末态 + teacher-force 答案时投影。
  - 训练信号沿递归计算图回传: loss -> lm_head -> backbone -> virtual embedding
    -> adapter。backbone 冻结但仍可微 (梯度穿过权重回到 adapter)。
  - 梯度路径: prefill 在 no_grad 下 (省内存); 递归步 + 答案 teacher-forcing
    必须保留 grad, 且 **不可 detach cache** (否则切断前面潜在步梯度)。

本文件修掉的骨架坑:
  ① 位置连续性: 显式传 cache_position = 当前缓存长度起的 arange,
     attention_mask 留 None (5.x 增量解码最稳写法), 保证 RoPE 连续。
  ② 缓存长度自检: 冒烟断言 cache.get_seq_length() == prompt_len + steps
     (+ 答案 teacher-force 长度), 对应强模型"缓存长度与递归步数一致"要求。

数据/前缀: 复用 train_distill.py 的 PROMPT_TMPL = "问题：{q}\n答案：",
            保证 System1(直答) 与 System2(递归) 用同一 prompt 前缀, 对照公平。

⚠️ 第一版刻意 batch_size=1 / 无 padding / FP32 / CPU, 先验证实现
   (loss 可反传、adapter 真更新、缓存长度正确), 再上量。
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# 必须与 b4/train_distill.py 完全一致, 否则 System1/System2 对照不公平。
PROMPT_TMPL = "问题：{q}\n答案："


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class LatentAdapter(nn.Module):
    """可训练外挂模块, 恒等初始化 (up 权重/偏置全 0 => delta 初始为 0)。

    放在每一潜在步喂回 backbone 之前:  virtual_embedding = state + delta(state)
    state 是 backbone 末层 post-norm 的连续表示 (embedding 空间由 adapter 桥接)。
    """

    def __init__(self, hidden_size, bottleneck=128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        # 恒等起点: 初态 virtual_embedding == state, 训练从"无改动"开始。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, state):
        # state: (B, H)
        delta = self.up(F.gelu(self.down(self.norm(state))))
        return state + delta


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def latent_answer_loss(model, adapter, prompt_ids, answer_ids, num_steps=2,
                       return_cache=False):
    """一条样本 (batch=1)。prompt_ids:(1,P)  answer_ids:(1,A) 含 EOS。

    返回 CE loss。可选返回 cache 供缓存长度自检。
    """
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    A = answer_ids.shape[1]
    backbone = model.model

    # 1) prefill prompt, 建初始 KV cache (no_grad: 冻结 backbone, 不须梯度)。
    cache = DynamicCache()
    cp = torch.arange(P, device=device)
    with torch.no_grad():
        out = backbone(input_ids=prompt_ids, past_key_values=cache,
                       use_cache=True, cache_position=cp)
        state = out.last_hidden_state[:, -1, :]          # (1,H) post-norm

    # 2) 固定步数潜在递归 (保留 grad, 不 detach cache)。
    cur_len = P
    for step in range(num_steps):
        ve = adapter(state).unsqueeze(1)                  # (1,1,H)
        cp = torch.arange(cur_len, cur_len + 1, device=device)
        out = backbone(inputs_embeds=ve, past_key_values=cache,
                       use_cache=True, cache_position=cp)
        cache = out.past_key_values
        state = out.last_hidden_state[:, -1, :]           # (1,H)
        cur_len += 1

    # 3) 末态预测首个答案 token。
    first_logits = model.lm_head(state).unsqueeze(1)      # (1,1,V)

    # 4) teacher-force 剩余答案 token (真实 token 经 embed_tokens)。
    if A > 1:
        ans_in = answer_ids[:, :-1]                       # (1, A-1)
        cp = torch.arange(cur_len, cur_len + (A - 1), device=device)
        out = backbone(input_ids=ans_in, past_key_values=cache,
                       use_cache=True, cache_position=cp)
        cache = out.past_key_values
        rest_logits = model.lm_head(out.last_hidden_state)  # (1, A-1, V)
        cur_len += (A - 1)
        logits = torch.cat([first_logits, rest_logits], dim=1)
    else:
        logits = first_logits

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        answer_ids.reshape(-1),
    )
    if return_cache:
        return loss, cache, cur_len
    return loss


def train_step(model, adapter, optimizer, prompt_ids, answer_ids, num_steps=2):
    model.eval()
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    loss = latent_answer_loss(model, adapter, prompt_ids, answer_ids, num_steps)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


# --------------------------------------------------------------------------
# 数据: 复用与 train_distill 同模板, 产出 (prompt_ids, answer_ids) 单样本 (无 pad)
# --------------------------------------------------------------------------
def build_sample(tok, q, a, max_prompt=256):
    prompt = PROMPT_TMPL.format(q=q)
    p_ids = tok.encode(prompt, add_special_tokens=False)[:max_prompt]
    a_ids = tok.encode(a, add_special_tokens=False) + [tok.eos_token_id]
    return (torch.tensor([p_ids], dtype=torch.long),
            torch.tensor([a_ids], dtype=torch.long))


# --------------------------------------------------------------------------
# Gate 2 冒烟: 在 135M 上确认 实现正确 (loss 有限 / grad>0 / 参数更新 / 缓存长度)
# --------------------------------------------------------------------------
def gate2_smoke(model_dir, device="cpu"):
    log("载入模型 (FP32, CPU) ...")
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device)
    freeze_backbone(model)
    H = model.config.hidden_size
    adapter = LatentAdapter(H).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.01)

    # 内嵌样本, 不依赖外部数据文件即可跑。
    qa = [("17 × 24 等于多少？", "408"),
          ("中国的首都是哪里？", "北京"),
          ("3 加 5 等于几？", "8")]

    for num_steps in (1, 2):
        log(f"==== num_steps={num_steps} ====")
        for q, a in qa:
            pids, aids = build_sample(tok, q, a)
            pids = pids.to(device); aids = aids.to(device)
            P, A = pids.shape[1], aids.shape[1]

            # 缓存长度自检 (no_grad 下单独跑, 不污染训练图)。
            with torch.no_grad():
                loss, cache, cur_len = latent_answer_loss(
                    model, adapter, pids, aids, num_steps, return_cache=True)
                expect = P + num_steps + (A - 1)
                ok_len = (cache.get_seq_length() == expect)
                log(f"  cache_len={cache.get_seq_length()} expect={expect} "
                    f"match={ok_len} loss={float(loss):.4f}")

            # 参数更新自检。
            before = [p.detach().clone() for p in adapter.parameters()]
            l, g = train_step(model, adapter, opt, pids, aids, num_steps)
            after = list(adapter.parameters())
            changed = any(not torch.equal(b, a.detach())
                         for b, a in zip(before, after))
            finite = torch.isfinite(torch.tensor(l))
            log(f"  [{q}] loss={l:.4f} grad_norm={g:.4f} "
                f"finite={finite} adapter_updated={changed}")
            assert finite, "loss 非有限"
            assert g > 0, "grad_norm 应为正"
            assert changed, "adapter 参数未更新"
            assert ok_len, "缓存长度 != prompt+steps+answer"
    log("GATE2_SMOKE_PASS ✅  loss可反传 / 梯度>0 / 参数更新 / 缓存长度正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", default=None,
                    help="QA JSONL ({question,answer}); 不传则只跑冒烟")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 Gate2 实现冒烟 (不依赖 --data)")
    ap.add_argument("--num_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--max_prompt", type=int, default=256)
    ap.add_argument("--out_dir", default="D:/work/_local_ai/train_15b/b4/out_latent")
    a = ap.parse_args()

    device = "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device)
    freeze_backbone(model)
    H = model.config.hidden_size
    adapter = LatentAdapter(H).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr, weight_decay=0.01)

    if a.smoke or a.data is None:
        gate2_smoke(a.model_dir, device)
        return

    os.makedirs(a.out_dir, exist_ok=True)

    # 最小训练循环 (batch=1, 无 padding) —— 真实 B1 训练占位。
    import json
    samples = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("question") and d.get("answer"):
                samples.append((d["question"], d["answer"]))
    log(f"B1 训练样本: {len(samples)}  num_steps={a.num_steps} steps={a.steps}")

    t0 = time.time()
    for i in range(a.steps):
        q, ans = samples[i % len(samples)]
        pids, aids = build_sample(tok, q, ans, a.max_prompt)
        pids = pids.to(device); aids = aids.to(device)
        l, g = train_step(model, adapter, opt, pids, aids, a.num_steps)
        if i % 10 == 0 or i == a.steps - 1:
            el = time.time() - t0
            log(f"step {i:>5d}/{a.steps} loss={l:.4f} grad={g:.4f} {el:.0f}s")
            t0 = time.time()
    torch.save(adapter.state_dict(),
               f"{a.out_dir}/adapter_steps{a.num_steps}.pt")
    log("B1 adapter 已保存 ->", a.out_dir)


if __name__ == "__main__":
    main()
