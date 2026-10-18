#!/usr/bin/env python3
# base_check.py — node1: 基座权重验收。结构对 != 可用, 必须跑
#   ① 中文 holdout 交叉熵  ② 2 条贪心生成
# 判定: CE 应 ~2-4 (本地 135M/200M 同批语料参考 ~2.x); 若 >10 则权重/词表有问题。
import os, sys, json, torch

BASE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/node1/chain/r4base_std"
TOKDIR = sys.argv[2] if len(sys.argv) > 2 else BASE
CORP = sys.argv[3] if len(sys.argv) > 3 else "/mnt/workspace/node1/chain/corpus_r4/zh_text_r4.jsonl"
NLINE = int(sys.argv[4]) if len(sys.argv) > 4 else 300

from transformers import AutoModelForCausalLM, AutoTokenizer
import json as _json

# --- tokenizer: 归一化 extra_special_tokens(list->dict) 规避新版 transformers 崩溃 ---
# 注意: tokenizer 目录与模型目录分开传(ckpt 目录里的 tokenizer 文件可能不完整)
try:
    cfgp = os.path.join(TOKDIR, "tokenizer_config.json")
    if os.path.exists(cfgp):
        c = _json.load(open(cfgp, encoding="utf-8"))
        if isinstance(c.get("extra_special_tokens"), list):
            c["extra_special_tokens"] = {}
            _json.dump(c, open(cfgp, "w", encoding="utf-8"), ensure_ascii=False)
            print("[tok] normalized extra_special_tokens list->{}")
except Exception as e:
    print("[tok] warn", repr(e)[:120])

try:
    tok = AutoTokenizer.from_pretrained(TOKDIR)
except Exception as e:
    print("[tok] fast load failed:", repr(e)[:120])
    tok = AutoTokenizer.from_pretrained(TOKDIR, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
model.eval()
print("[load] params=%.1fM vocab=%s" % (sum(p.numel() for p in model.parameters()) / 1e6,
                                        getattr(tok, "vocab_size", "?")))

# --- 取 holdout 文本 ---
texts = []
with open(CORP, encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= NLINE:
            break
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            t = o.get("text") or o.get("content") or ""
        except Exception:
            t = line
        if t:
            texts.append(t)
print("[data] lines=%d chars=%d" % (len(texts), sum(len(t) for t in texts)))

blob = "\n".join(texts)
ids = tok(blob, return_tensors="pt", add_special_tokens=False).input_ids[0]
print("[data] tokens=%d" % ids.numel())

# --- CE over sliding windows ---
MAXLEN = 512
STRIDE = 512
nll, ntok = 0.0, 0
with torch.no_grad():
    for s in range(0, int(ids.numel()) - MAXLEN - 1, STRIDE):
        # 正确口径: input_ids == labels, 由 HF 内部 shift
        w = ids[s:s + MAXLEN].unsqueeze(0)
        out = model(input_ids=w, labels=w)
        n = MAXLEN - 1
        nll += out.loss.item() * n
        ntok += n
        if ntok >= 50000:
            break
ce = nll / max(ntok, 1)
print("=== HOLDOUT CE = %.4f (tokens=%d) ===" % (ce, ntok))

# --- 贪心生成 2 条 ---
for prompt in ["中国的历史", "人工智能"]:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=40, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("GEN[%s]: %s" % (prompt, tok.decode(g[0], skip_special_tokens=True)[:160]))
print("BASE_CHECK_DONE")
