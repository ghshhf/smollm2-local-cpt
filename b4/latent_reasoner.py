#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4 System 1/2 —— 潜在状态递归模块 (System 2 核心)  [v0.2 脚手架]
=============================================================
设计: COCONUT 风格。把"思维"表示成一个连续向量 (latent), 作为虚拟 token 的
      embedding, 反复喂回整个 backbone (decoder-only Llama), 每步把它的 KV 追
      加到 cache, 使后续思维/答案都能 attend 到前面的思维。最后用末态直接预测
      答案首个 token, 再正常自回归续写完整答案。

为什么这样写 (对比 v0.1 的坐标 bug):
  v0.1 用"全模型+norm 后的末位 hidden"去初始化 state, 却只让 reason 跑
  shared_layers、decode 跑剩余层 —— state 初始空间与解码空间对不上。
  v0.2 改成: latent 始终 = backbone 全模型 + final norm 的连续表示, 直接当
  下一位置的 embedding 喂回, 与 COCONUT 原论文一致, 且 KV 自然累积。

KV 管理: 直接用 transformers 的 DynamicCache, 让 model.model 自己维护/追加
         KV, 避免手工 tuple/list 在 4.40+ 上的兼容坑。

基础模型: 本项目 HF SmolLM2 (LlamaForCausalLM, GQA), CPU fp32/bf16。
复用方式: 零改动 backbone 结构, 仅在外层把 latent 当 inputs_embeds 反复调用
          model.model(...)。即 §10.3 "复用现有权重 + 少量外挂模块"。

⚠️ v0.2 仍然只是脚手架, 不是训好的系统:
   - 默认"共享全部层"做思维步; 部分层递归 (§10 消融项) 见 LatentReasoner 注释。
   - 未含 router / stop_head (确认递归有用后再加 router.py / stop_controller.py)。
   - 本文件不训练, 只做推理/评测链路。训练见 train_distill.py (System1 基线)。
"""
import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


class LatentReasoner(nn.Module):
    """
    把 backbone 当"思维引擎": 给定当前 latent (B,1,H) 与一个 DynamicCache,
    跑一遍整个 model.model, 返回更新后的 latent (B,1,H)。cache 会被模型就地
    追加 (思维彼此可见, 这是 COCONUT 的关键)。

    部分层变体 (消融用, 当前未启用): 若只让 [start,end) 层参与思维步, 则需把
    latent 逐层跑 shared_layers 并自行处理 norm/残差/KV, 复杂度更高; v0.2 先
    用"共享全部层"把机制跑通, 再开该消融。
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.n_layers = base_model.config.num_hidden_layers
        self.adapter = None

    @torch.no_grad()
    def thought_step(self, latent, cache):
        """一步思维: latent 作为虚拟 token 的 embedding 喂回 backbone。

        若 HybridModel 绑定了训练过的 LatentAdapter, 在喂回前先用 adapter
        对末位状态做恒等起点的残差更新 (与 b4/train_latent.py 训练时一致)。
        """
        if self.adapter is not None:
            x = latent[:, -1, :]                 # (B,H)
            x = self.adapter(x).unsqueeze(1)      # (B,1,H)
            latent = x
        out = self.base.model(
            inputs_embeds=latent,
            past_key_values=cache,
            use_cache=True,
        )
        return out.last_hidden_state          # (B,1,H) post final norm

    @torch.no_grad()
    def forward(self, latent, cache, steps):
        """递推 steps 步思维, 返回末态 latent (B,1,H)。cache 已被就地更新。"""
        for _ in range(steps):
            latent = self.thought_step(latent, cache)
        return latent


class HybridModel(nn.Module):
    """
    包装 base + LatentReasoner, 提供 Gate 2 实验所需接口:
      - encode_prefix(input_ids)      -> (cache, init_latent)
      - reason(cache, init_latent, k) -> latent              固定 k 步思维
      - decode_logits(latent)         -> (B,1,V)            首答案 token 分布
      - generate_answer(input_ids,k)  -> str                 完整贪心答案
      - solve_first_token(input_ids,k)-> int                 仅首 token (快速对照)
    """

    def __init__(self, base_model, tok=None, adapter=None):
        super().__init__()
        self.base = base_model
        self.reasoner = LatentReasoner(base_model)
        self.reasoner.adapter = adapter
        self.eos_id = base_model.config.eos_token_id
        self.tok = tok

    @torch.no_grad()
    def encode_prefix(self, input_ids):
        cache = DynamicCache()
        out = self.base.model(input_ids=input_ids,
                              past_key_values=cache, use_cache=True)
        # prompt 末位 (post final norm) 作为第一个连续思维向量 —— COCONUT 初始化
        init_latent = out.last_hidden_state[:, -1:, :]   # (B,1,H)
        return cache, init_latent

    @torch.no_grad()
    def reason(self, cache, init_latent, steps):
        return self.reasoner(init_latent, cache, steps)  # 末态 latent (B,1,H)

    @torch.no_grad()
    def decode_logits(self, latent):
        return self.base.lm_head(latent)                # (B,1,V)

    @torch.no_grad()
    def solve_first_token(self, input_ids, steps):
        cache, lat = self.encode_prefix(input_ids)   # cache, init_latent
        lat = self.reason(cache, lat, steps)          # 末态 latent
        logits = self.decode_logits(lat)
        return int(logits.argmax(-1)[0, 0])

    @torch.no_grad()
    def generate_answer(self, input_ids, steps, max_new=64):
        """
        端到端: 编码 -> k 步思维 -> 贪心解码完整答案。
        首答案 token 由 latent 直接预测; 其后每个真实 token 仍走完整 backbone
        并累积 KV (与 model.generate 等价, 仅首位置由 latent 播种)。
        """
        device = input_ids.device
        cache, lat = self.encode_prefix(input_ids)   # cache, init_latent
        lat = self.reason(cache, lat, steps)          # 末态 latent (B,1,H)

        # 首 token
        logits = self.decode_logits(lat)                 # (B,1,V)
        tok = int(logits.argmax(-1)[0, 0])
        seq = [tok]
        if tok == self.eos_id:
            return self.tok.decode(seq, skip_special_tokens=True)

        # 自回归续写
        for _ in range(max_new - 1):
            emb = self.base.model.embed_tokens(
                torch.tensor([[tok]], device=device))    # (B,1,H)
            out = self.base.model(inputs_embeds=emb,
                                  past_key_values=cache, use_cache=True)
            lat = out.last_hidden_state                  # (B,1,H)
            logits = self.base.lm_head(lat)
            tok = int(logits.argmax(-1)[0, 0])
            seq.append(tok)
            if tok == self.eos_id:
                break
        return self.tok.decode(seq, skip_special_tokens=True)


def load_hybrid(model_dir, dtype=torch.float32, adapter_path=None):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    adapter = load_adapter(adapter_path, model.config.hidden_size) if adapter_path else None
    hybrid = HybridModel(model, tok, adapter=adapter)
    return hybrid, tok


def load_adapter(adapter_path, hidden_size, device="cpu"):
    """延迟导入 LatentAdapter (避免与训练脚本循环依赖 / 路径问题)。"""
    from b4.train_latent import LatentAdapter
    a = LatentAdapter(hidden_size)
    sd = torch.load(adapter_path, map_location=device)
    a.load_state_dict(sd)
    a.eval()
    return a.to(device=device, dtype=torch.float32)


def _smoke_test(model_dir, bf16=False):
    dtype = torch.bfloat16 if bf16 else torch.float32
    hybrid, tok = load_hybrid(model_dir, dtype)
    tests = [
        ("17 × 24 等于多少？", "408"),
        ("一个数的两倍加 7 等于 31，这个数是多少？", "12"),
        ("中国的首都是哪里？", "北京"),
    ]
    for q, gold in tests:
        ids = tok(q, return_tensors="pt")["input_ids"]
        print(f"\n[Q] {q}  (gold={gold})")
        for steps in (0, 1, 2, 4, 8):
            t0 = time.perf_counter()
            ans = hybrid.generate_answer(ids, steps, max_new=32)
            dt = (time.perf_counter() - t0) * 1000
            mark = "OK" if (gold in ans or ans in gold) else "  "
            print(f"  steps={steps:>2d} [{mark}] {dt:7.1f}ms  -> {ans!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="HF SmolLM2 权重目录 (本项目 0.3B/0.4B 续训输出, "
                         "或本机 models/SmolLM2-135M 等做机制验证)")
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    _smoke_test(a.model_dir, a.bf16)
