"""ttt_infer.py — In-Place TTT 推理时学习 harness（本地三线·推理时进化）

把 r6_final_20261016（进化完成的 0.2B）权重装进 ByteDance-Seed/In-Place-TTT 官方
Llama 实现，开启 TTT 模式，做「装知识前后」对比。fast weights = 各 MLP 末端
down_proj，纯前向 chunk 累积外积更新，无反传/无优化器/无 loss 曲线。

环境：venv_ttt（torch 2.7.1+cpu / transformers 4.57.3 / einops / opt_einsum）
⚠️ 官方代码要 transformers 4.57.3，与训练 venv（5.19）冲突 ⇒ 独立环境，绝不混用。

用法：
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/ttt_infer.py --sanity
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/ttt_infer.py --demo --doc <文件> --query "任务"

设计纪律：
  - ttt_conv 恒等初始化（depthwise conv 中心 tap=1）⇒ 目标信号初始=原始 embedding，
    不引入随机噪声（随机初始化会让 fast weights 更新变成噪声驱动）
  - ttt_proj=False ⇒ 零额外参数（纯 Hebbian 外积更新， mdlARC 同款思路）
  - ttt_target="input_embed"（官方 hf 实现里唯一走通的 target 路径）
  - 对拍验收：TTT 关时与原版 transformers Llama 的 logits 最大差 < 1e-3，
    否则说明权重没挂对，后面实验全作废
  - 已知限制：TTT 的 fast-weight 状态在 KV-cache 增量解码中不延续（delta 由
    当前 forward 的序列现算）⇒ --demo 用全量重前向生成，慢但正确
"""
import argparse
import json
import os
import sys

import torch

REPO = "D:/work/_local_ai/_research/In-Place-TTT"
BASE = "D:/work/_local_ai/train_15b/backups/0.2B/r6_final_20261016"
TTT_LAYERS_DEFAULT = list(range(22, 30))  # 后 8 层开 TTT（浅层管语法不动）


def build(ttt_on: bool = True, base: str = None, ttt_layers=None, ttt_chunk: int = 512,
          ttt_lr: float = 0.3, ttt_proj: bool = False):
    """底座可参数化：base=None 时用模块常量 BASE(0.2B r6_final)；传 0.3B 路径则用北极星 0.3B。"""
    sys.path.insert(0, REPO)
    from hf_models.hf_llama.configuration_llama import LlamaConfig
    from hf_models.hf_llama.modeling_llama import LlamaForCausalLM

    base = base or BASE
    cfg = LlamaConfig.from_pretrained(base)
    # 🔴 基座 r6_final 是 tie_word_embeddings=True；官方 TTT 配置默认 False ⇒ lm_head 不会被加载
    # （state_dict 里没有 lm_head.weight），随机初始化会让 logits 全是垃圾、实验全作废。
    cfg.tie_word_embeddings = True
    cfg.ttt_mode = bool(ttt_on)
    cfg.ttt_layers = list(ttt_layers) if ttt_on else []
    cfg.ttt_chunk = ttt_chunk
    cfg.ttt_lr = ttt_lr
    cfg.ttt_proj = ttt_proj
    cfg.ttt_target = "input_embed"
    model = LlamaForCausalLM(cfg)
    model = model.float().eval()

    from safetensors.torch import load_file
    sd = load_file(os.path.join(base, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 只允许缺 ttt_* 新模块 与 lm_head(当 tie_word_embeddings=True 时 lm_head 与
    # embed_tokens 共享权重，state_dict 不含 lm_head.weight 属预期，非架构不匹配)
    bad_missing = [k for k in missing if not k.endswith("ttt_conv.weight")
                   and not k.endswith("ttt_proj.weight")
                   and not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    print("[load] params=%.1fM missing=%d(全为ttt_*则正常: %s) unexpected=%d"
          % (sum(p.numel() for p in model.parameters()) / 1e6,
             len(missing), all(k.endswith(("ttt_conv.weight", "ttt_proj.weight")) for k in missing),
             len(unexpected)))
    if unexpected:
        print("[load] UNEXPECTED KEYS: %s" % unexpected[:5])
    if bad_missing:
        print("[load] !! 非ttt缺失键(架构不匹配?): %s" % bad_missing[:5])

    # 恒等初始化 ttt_conv：depthwise conv 权重形状 (hidden,1,5)，中心 tap=1
    n_conv = 0
    for name, m in model.named_modules():
        if hasattr(m, "ttt_conv") and getattr(m, "ttt_conv") is not None:
            w = torch.zeros_like(m.ttt_conv.weight)
            w[:, 0, w.shape[-1] // 2] = 1.0
            with torch.no_grad():
                m.ttt_conv.weight.copy_(w)
            n_conv += 1
    print("[init] ttt_conv identity-init on %d layers" % n_conv)
    return model


def get_tok(base: str = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base or BASE)


@torch.no_grad()
def logits_of(model, tok, text):
    ids = tok(text, return_tensors="pt")
    out = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
    return out.logits[0, -1]


@torch.no_grad()
def gen_full(model, tok, prompt, n_new=60):
    """全量重前向生成（TTT 状态下 cache 不延续 fast weights，宁可慢也要对）"""
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    for _ in range(n_new):
        out = model(input_ids=ids)
        nxt = int(out.logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if nxt == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


def sanity():
    tok = get_tok()
    prompt = "人工智能的三个层次是：感知、"
    print("== 1) 权重对拍：官方TTT实现(TTT关) vs 原版 transformers Llama ==")
    m_off = build(ttt_on=False)
    lg_off = logits_of(m_off, tok, prompt)
    from transformers import LlamaForCausalLM
    ref = LlamaForCausalLM.from_pretrained(BASE).float().eval()
    lg_ref = logits_of(ref, tok, prompt)
    d = (lg_off - lg_ref).abs().max().item()
    print("[cmp] max|logit diff| = %.2e  %s" % (d, "PASS" if d < 1e-3 else "FAIL(权重没挂对!)"))
    print("[cmp] argmax off=%d ref=%d" % (int(lg_off.argmax()), int(lg_ref.argmax())))

    print("== 2) TTT 开：同一 prompt 前向（fast weights 应被激活）==")
    m_on = build(ttt_on=True, ttt_layers=TTT_LAYERS_DEFAULT)
    lg_on = logits_of(m_on, tok, prompt)
    d2 = (lg_on - lg_off).abs().max().item()
    print("[cmp] max|logit diff TTT_on vs off| = %.2e  %s" % (d2, "TTT生效(有差异)" if d2 > 1e-4 else "⚠️ 无差异=TTT没起作用"))

    print("== 3) 生成对比（各60 token）==")
    print("[TTT off]", gen_full(m_off, tok, "请用一句话介绍北京：", 60).split("：", 1)[-1][:120])
    print("[TTT  on]", gen_full(m_on, tok, "请用一句话介绍北京：", 60).split("：", 1)[-1][:120])


def demo(doc_text, query):
    tok = get_tok()
    m_off = build(ttt_on=False)
    m_on = build(ttt_on=True, ttt_layers=TTT_LAYERS_DEFAULT)
    prompt = doc_text + "\n\n根据以上资料回答问题：" + query + "\n答："
    print("== 无文档 baseline ==")
    print(gen_full(m_off, tok, query + "\n答：", 80)[len(query) + 2:][:300])
    print("== 有文档 + TTT off（纯上下文学习）==")
    print(gen_full(m_off, tok, prompt, 80)[len(prompt):][:300])
    print("== 有文档 + TTT on（fast weights 装知识）==")
    print(gen_full(m_on, tok, prompt, 80)[len(prompt):][:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--doc", default=None, help="文档文件路径（demo 用）")
    ap.add_argument("--query", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    if a.layers:
        TTT_LAYERS_DEFAULT = [int(x) for x in a.layers.split(",")]
    if a.sanity:
        sanity()
    elif a.demo:
        doc = open(a.doc, encoding="utf-8").read() if a.doc else ""
        demo(doc, a.query)
    else:
        ap.print_help()
