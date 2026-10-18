#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""chat_blind_eval.py — 对话线盲测（训练集之外的输入 + 关键点命中）。

判据说明（对应 eval/RESULT.md 的验收铁律）：
  in-set CE 只能证明"记住了训练集"，不能证明"会用"。
  这里用没进过训练集的新问题，看模型答得是否在点上（key 命中）。
  key_hit = 命中关键词数 / 该题关键词数（无关键词的题记为 None，人工看）。

用法：
  python tools/chat_blind_eval.py --model backups/0.3B/polaris-03b_20261018 \
      --adapter out_lora_chat_v7 --data data/chat_holdout_v7.jsonl \
      --out logs/chat_blind_v7.json
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "你是北极星，主人叫黄果强。说话简短、直接、不装懂。"
A_MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def build_prompt(q):
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n%s" % (SYS, q, A_MARK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "chat_holdout_v7.jsonl"))
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--rept-penalty", type=float, default=1.25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "chat_blind.json"))
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False).eval()
        print("[adapter] %s" % a.adapter)
    eos_id = tok.convert_tokens_to_ids(IM_END)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print("[data] 盲测 %d 题" % len(rows))

    results, hits, total_kw, n_kw_q = [], 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        pid = tok(build_prompt(r["q"]), return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(pid, max_new_tokens=a.max_new, do_sample=False,
                                 repetition_penalty=a.rept_penalty,
                                 eos_token_id=eos_id, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][pid.size(1):], skip_special_tokens=True)
        txt = txt.split(IM_END)[0].strip()
        kws = r.get("keywords") or []
        if kws:
            k = sum(1 for w in kws if w.lower() in txt.lower())
            total_kw += k
            n_kw_q += 1
            hit = k / len(kws)
        else:
            hit = None
        hits += (hit or 0)
        results.append({"q": r["q"], "a": txt, "keywords": kws,
                        "hit": hit, "note": r.get("note", "")})
        if i < a.show:
            print("-" * 56)
            print("Q: %s   [%s]" % (r["q"], r.get("note", "")))
            print("A: %s" % txt[:160])
            if hit is not None:
                print("   命中 %d/%d = %.0f%%" % (
                    int(hit * len(kws)), len(kws), hit * 100))

    n = len(results)
    with_kw = [x for x in results if x["hit"] is not None]
    covered = sum(1 for x in with_kw if x["hit"] >= 0.5)
    full = sum(1 for x in with_kw if x["hit"] >= 0.99)
    empty = sum(1 for x in results if len(x["a"]) < 2)
    summary = {
        "total": n,
        "avg_hit": round(hits / max(1, n_kw_q), 4),
        "mid_hit_rate(>=50%)": round(covered / max(1, len(with_kw)), 4),
        "full_hit_rate(100%)": round(full / max(1, len(with_kw)), 4),
        "empty_output": empty,
        "avg_chars": round(sum(len(x["a"]) for x in results) / max(1, n), 1),
        "seconds": round(time.time() - t0, 1),
        "model": a.model,
        "adapter": a.adapter,
        "results": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n===== 盲测汇总 =====")
    print("  平均命中率      %.1f%%" % (summary["avg_hit"] * 100))
    print("  过半数命中       %.1f%% (%d/%d)" % (
        summary["mid_hit_rate(>=50%)"] * 100, covered, len(with_kw)))
    print("  全命中           %.1f%%" % (summary["full_hit_rate(100%)"] * 100))
    print("  空输出/退化      %d 条" % empty)
    print("  平均回答字数     %.1f" % summary["avg_chars"])
    print("  用时             %.1fs" % summary["seconds"])
    print("  明细 -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
