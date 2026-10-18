# -*- coding: utf-8 -*-
"""
chat_gen_test.py — 判定"只能回答关键词"的根因

三组对照, 全部贪心解码:
  A 训练集内(原题原样)      -> 记没记住
  B 训练集外(同义改写)      -> 有没有泛化(这才是"能用"的定义)
  C 训练集外(全新领域)      -> 底座本身有没有能力

判据:
  A 高 / B 低  => 纯记忆, 是数据量问题(要加数据)
  A 高 / B 也高 => 模块真的迁移了能力
  A 低         => 连记忆都没成(训练配置问题)
"""
import argparse, json, os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE2 = "backups/0.2B/r6_final_20261016"
BASE3 = "backups/0.3B/polaris-03b_20261018"
LORA = "out_lora_chat_v4"
PORT = "models/chat_lora_v4_port03b"
TPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

# A: 训练集内原题 -> 参考答案(从 chat_sft_v5 逐字取)
A_SET = [
    ("你好", "你好！有什么我可以帮你的吗？"),
    ("一周有几天", "7 天：周一到周日。"),
    ("一年有几个月", "12 个月。"),
    ("你会什么", "主人，我擅长写代码、解释知识、陪你聊天。"),
    ("你的主人是谁", "就是眼前跟我说话的这位主人，也就是你。"),
    ("17加26等于多少", "17 + 26 = 43。"),
    ("在吗", "在的，你说。"),
    ("光的传播速度快还是声音快", "光快得多。"),
]

# B: 同义改写 —— 意思完全一样, 措辞全变
B_SET = [
    ("您好", "你好"),
    ("一周有多少天？", "7 天"),
    ("一年有多少个月呢", "12 个月"),
    ("你都能干什么？", "会什么"),
    ("谁是你的主人", "主人是谁"),
    ("26 加 17 等于几", "43"),
    ("你在不在", "在"),
    ("光和声音哪个速度快", "光快"),
]

# C: 全新领域, 训练集完全没出现
C_SET = [
    "3 加 5 等于多少",
    "7 乘 8 等于多少",
    "明天天气怎么样",
    "给我讲个笑话",
    "中国最长的河是哪条",
]


def lcs_ratio(pred, ref):
    """最长公共子序列占比, 度量"复现了多少参考文本" """
    p, r = pred.strip(), ref.strip()
    if not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(p)][len(r)] / len(r)


def gen(m, tok, q, n=40):
    ids = tok(TPL.format(q=q), return_tensors="pt").input_ids
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=n, do_sample=False,
                       repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="02B", choices=["02B", "03B", "03Bport", "03Bbare"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.which == "02B":
        base, ad, tag = BASE2, LORA, "0.2B + LoRA v4 [原生]"
    elif a.which == "03Bport":
        base, ad, tag = BASE3, PORT, "0.3B + 移植LoRA v4"
    elif a.which == "03Bbare":
        base, ad, tag = BASE3, None, "0.3B 裸模型"
    else:
        base, ad, tag = BASE3, LORA, "0.3B + LoRA 直装"

    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    if ad:
        m = PeftModel.from_pretrained(m, ad).merge_and_unload()

    print(f"\n########## {tag} ##########\n")
    print("=== A 训练集内(原题) ===")
    sa = 0.0
    for q, ref in A_SET:
        out = gen(m, tok, q, a.n)
        r = lcs_ratio(out, ref)
        sa += r
        print(f"  Q {q}")
        print(f"    出 {out[:56]!r}")
        print(f"    参 {ref[:40]!r}   重合 {r*100:.0f}%")
    sa /= len(A_SET)

    print("\n=== B 训练集外(同义改写) ===")
    for q, hint in B_SET:
        out = gen(m, tok, q, a.n)
        ok = hint in out
        print(f"  Q {q}   期望含[{hint}]  {'命中' if ok else '未命中'}")
        print(f"    出 {out[:56]!r}")

    print("\n=== C 训练集外(全新领域) ===")
    for q in C_SET:
        out = gen(m, tok, q, a.n)
        print(f"  Q {q}\n    出 {out[:56]!r}")

    print(f"\n[汇总] A 组平均复现度 = {sa*100:.1f}%")


if __name__ == "__main__":
    main()
