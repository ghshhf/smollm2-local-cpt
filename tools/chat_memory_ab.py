#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""chat_memory_ab.py — 记忆层 A/B 对照：同一批题，memory off vs on。

背景(PROJECT.md 第 6 线挂账): TinyContext 记忆层 7 条 profile 常驻,
但「记忆到底帮不帮对话」从未量化。本工具做同参 30 题对照:
  off 臂: 原样 system(身份一句)
  on  臂: system + TinyContext profile(常驻) + recall 检索注入
E-7 合规: 只读记忆层注入上下文, 不碰模型参数。

用法(venv_ttt, tinycontext 只装在这):
  venv_ttt/Scripts/python.exe tools/chat_memory_ab.py \
      --model backups/0.3B/polaris-03b_20261018 --adapter out_lora_chat_v7 \
      --out logs/chat_memory_ab.json

环境变量(可选): TINYCONTEXT_DB 指记忆库路径, 默认 %LOCALAPPDATA%\tinycontext\memories.db
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS_BASE = "你是北极星，主人叫黄果强。说话简短、直接、不装懂。"
A_MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def get_tc_config():
    """显式构造 TinyContextConfig(坑1: API 不读环境变量, 必须传 config)。"""
    import tinycontext as tc
    db = os.environ.get("TINYCONTEXT_DB") or os.path.expandvars(
        r"%LOCALAPPDATA%\tinycontext\memories.db")
    models_dir = os.environ.get("TINYCONTEXT_MODELS_DIR") or os.path.expandvars(
        r"%LOCALAPPDATA%\tinycontext\models")
    cfg = tc.TinyContextConfig(
        memory_db_path=db,
        models_dir=models_dir,
        embedding_model="BAAI/bge-small-zh-v1.5",
    )
    return tc, cfg


def recall_for(tc, cfg, q, top_k=3):
    """拼出注入到 system 的记忆文本: 常驻 profile + 检索命中 episodic。"""
    try:
        res = tc.recall_memories(q, top_k=top_k, config=cfg)
    except Exception as e:
        return "", "[recall 异常: %s]" % e
    parts = []
    prof = res.get("profile") or []
    if prof:
        parts.append("[记忆·身份]\n" + "\n".join("- %s" % m["content"] for m in prof))
    mem = res.get("memories") or []
    if mem:
        parts.append("[记忆·相关]\n" + "\n".join("- %s" % m["content"] for m in mem))
    return "\n\n".join(parts), ""


def build_prompt(q, memory_text=""):
    sys_txt = SYS_BASE
    if memory_text:
        sys_txt = "%s\n\n%s" % (SYS_BASE, memory_text)
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n%s" % (sys_txt, q, A_MARK))


def gen_one(model, tok, eos_id, q, memory_text, max_new, rep):
    pid = tok(build_prompt(q, memory_text), return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(pid, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=rep,
                             eos_token_id=eos_id, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][pid.size(1):], skip_special_tokens=True)
    txt = txt.split(IM_END)[0].strip()
    return txt


def score(txt, kws):
    if not kws:
        return None
    k = sum(1 for w in kws if w.lower() in txt.lower())
    return k / len(kws)


def arm_summary(items):
    hits = [x for x in items if x["hit"] is not None]
    avg = sum(x["hit"] for x in hits) / max(1, len(hits)) if hits else 0.0
    covered = sum(1 for x in hits if x["hit"] >= 0.5)
    full = sum(1 for x in hits if x["hit"] >= 0.99)
    empty = sum(1 for x in items if len(x["a"]) < 2)
    return {
        "avg_hit": round(avg, 4),
        "mid_hit_rate(>=50%)": round(covered / max(1, len(hits)), 4),
        "full_hit_rate(100%)": round(full / max(1, len(hits)), 4),
        "empty_output": empty,
        "avg_chars": round(sum(len(x["a"]) for x in items) / max(1, len(items)), 1),
        "n": len(items),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "chat_holdout_v7.jsonl"))
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--rept-penalty", type=float, default=1.25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3, help="on 臂每次召回 episodic 条数")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "chat_memory_ab.json"))
    ap.add_argument("--disable-on", action="store_true", help="只跑 off 臂(调试)")
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

    tc, cfg = None, None
    if not a.disable_on:
        tc, cfg = get_tc_config()
        print("[memory] db=%s" % cfg.memory_db_path)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print("[data] 盲测 %d 题" % len(rows))
    t0 = time.time()

    per_item = []
    off_items, on_items = [], []
    for i, r in enumerate(rows):
        q = r["q"]
        kws = r.get("keywords") or []
        # off 臂
        off_a = gen_one(model, tok, eos_id, q, "", a.max_new, a.rept_penalty)
        off_hit = score(off_a, kws)
        off_items.append({"a": off_a, "hit": off_hit})
        # on 臂
        mem_text, mem_err = "", ""
        if tc is not None:
            mem_text, mem_err = recall_for(tc, cfg, q, top_k=a.top_k)
        on_a = gen_one(model, tok, eos_id, q, mem_text, a.max_new, a.rept_penalty)
        on_hit = score(on_a, kws)
        on_items.append({"a": on_a, "hit": on_hit})

        per_item.append({
            "q": q, "note": r.get("note", ""), "keywords": kws,
            "off_a": off_a, "on_a": on_a,
            "off_hit": off_hit, "on_hit": on_hit,
            "delta_hit": (on_hit - off_hit) if (off_hit is not None and on_hit is not None) else None,
            "mem_err": mem_err,
        })
        if i < 30:
            print("-" * 60)
            print("Q: %s   [%s]" % (q, r.get("note", "")))
            print("  off: %s" % off_a[:140])
            if tc is not None:
                print("  on : %s" % on_a[:140])
                if off_hit is not None and on_hit is not None:
                    print("  Δ命中 %.0f%% -> %.0f%%" % (off_hit * 100, on_hit * 100))

    off_sum = arm_summary(off_items)
    on_sum = arm_summary(on_items) if tc is not None else None
    summary = {
        "model": a.model, "adapter": a.adapter,
        "off": off_sum, "on": on_sum,
        "delta_avg_hit": (on_sum["avg_hit"] - off_sum["avg_hit"]) if on_sum else None,
        "seconds": round(time.time() - t0, 1),
        "per_item": per_item,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n===== 记忆层 A/B 对照 =====")
    print("  off 平均命中率   %.1f%%" % (off_sum["avg_hit"] * 100))
    if on_sum:
        print("  on  平均命中率   %.1f%%" % (on_sum["avg_hit"] * 100))
        d = summary["delta_avg_hit"]
        print("  Δ (on-off)       %+.1f%%" % (d * 100))
        print("  on  过半数命中   %.1f%%" % (on_sum["mid_hit_rate(>=50%)"] * 100))
    print("  off 空输出/退化   %d 条" % off_sum["empty_output"])
    print("  用时              %.1fs" % summary["seconds"])
    print("  明细 -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
