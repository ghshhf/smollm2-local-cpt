#!/usr/bin/env python3
# quick_ce.py — 本地快速基座验收: 小样本中文 holdout CE + 贪心生成。
# 用法: python quick_ce.py <model_dir> [tokenizer_dir] [corpus] [windows]
import sys, os, json, math, torch

md = sys.argv[1]
tok_dir = sys.argv[2] if len(sys.argv) > 2 else md
corp = sys.argv[3] if len(sys.argv) > 3 else "D:/work/_local_ai/train_15b/pretrain_corpus/zh_text_r4.jsonl"
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 20

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained(tok_dir)
if not os.path.exists(os.path.join(md, "config.json")):
    cfg = AutoConfig.from_pretrained(tok_dir)
    print("config borrowed from", tok_dir)
    m = AutoModelForCausalLM.from_pretrained(md, config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True)
else:
    m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float32, low_cpu_mem_usage=True)
m.eval()
np_ = sum(p.numel() for p in m.parameters())
print("MODEL=%s params=%.1fM vocab=%s tie=%s" % (md, np_ / 1e6, m.config.vocab_size, m.config.tie_word_embeddings))
print("ln(vocab)=%.3f" % math.log(m.config.vocab_size))

texts = []
with open(corp, encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 120:
            break
        line = line.strip()
        if not line:
            continue
        try:
            texts.append(json.loads(line).get("text", ""))
        except Exception:
            texts.append(line)
blob = "\n".join([t for t in texts if t])
ids = tok(blob, return_tensors="pt", add_special_tokens=False).input_ids[0]
print("tokens=%d chars=%d" % (ids.numel(), len(blob)))

MAXLEN, nll, ntok = 512, 0.0, 0
with torch.no_grad():
    for s in range(0, int(ids.numel()) - MAXLEN - 1, MAXLEN):
        # 正确口径: input_ids == labels(完整 chunk), 由 HF 内部做 shift。
        # 若写成 labels=input_ids[:,1:] 会被再移一次 -> 目标错位一格, CE 虚高。
        w = ids[s:s + MAXLEN].unsqueeze(0)
        out = m(input_ids=w, labels=w)
        n = MAXLEN - 1
        nll += out.loss.item() * n
        ntok += n
        if ntok >= (NW - 1) * 511:
            break
print("=== CE = %.4f (tokens=%d, windows=%d) ===" % (nll / max(ntok, 1), ntok, ntok // 511 + 1))

for prompt in ["中国的历史", "人工智能"]:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=32, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("GEN[%s]: %s" % (prompt, tok.decode(g[0], skip_special_tokens=True)[:120]))
print("QUICK_CE_DONE")
