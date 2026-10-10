"""
V7 held-out 评测台（纯新增，不碰训练）
=====================================
用途：量化「LoRA 比基座聪明多少」——前沿 RLVR 的核心就是**可验证奖励**，
没有度量就没法说"优化了"。本脚本给每个 held-out 任务生成回答，
按 must_contain 关键词做 pass@1 判分，输出分类型通过率。

为何现在不跑：fp32 1.5B 推理要 ~6GB，与训练抢内存；
请在训练完成/错峰时段跑（如 V7 训完、或单独窗口）。

用法：
  python eval/eval_v7.py --data eval/holdout_v7.jsonl                  # 推理题基线(基座)
  python eval/eval_v7.py --data eval/holdout_v7.jsonl --adapter out_lora_v7
  python eval/eval_v7.py --data eval/holdout_v3.jsonl --adapter out_lora_v3  # 小说设定卡

双指标：
  kw     = 关键词命中率（术语能否复现，偏严格）
  overlap= 与设定原文参考答案的字符 3-gram F1（语义贴近度，抗改写噪声）
两个一起看，避免单一指标骗人。

输出：eval/result_<tag>.json  + 控制台分类型通过率表。
"""
import argparse
import json
import os
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")
HOLDOUT = os.path.join(ROOT, "eval", "holdout_v7.jsonl")


def build_prompt(instruction: str) -> str:
    return f"下面是一项任务。\n{instruction}\n输出："


def load_model(adapter: str | None, dtype: str = "bf16"):
    torch.set_num_threads(8)  # i5-12450H 12 线程里给推理固定 8，避免调度抖动
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 320) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def score(text: str, must: list[str]) -> float:
    t = text.lower()
    if not must:
        return 1.0
    hit = sum(1 for k in must if k.lower() in t)
    return hit / len(must)


def ngrams(s: str, n: int = 3) -> set:
    s = re.sub(r"\s+", "", s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def overlap_score(text: str, ref: str, n: int = 3) -> float:
    """字符 n-gram F1：模型不必逐字复述，但贴近原文才算学会。"""
    a, b = ngrams(text, n), ngrams(ref, n)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA 目录；不传=只评基座")
    ap.add_argument("--data", default=None, help="评测集；默认 holdout_v7.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--tag", default=None, help="结果文件后缀，默认取 adapter 目录名")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="推理精度；CPU 上 bf16 比 fp32 快 ~54%（实测 3.36 vs 2.18 tok/s）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（评测集已打乱，即随机子集）")
    args = ap.parse_args()
    tag = args.tag or (os.path.basename(args.adapter) if args.adapter else "base")
    holdout = args.data or HOLDOUT

    print(f"[eval] 评测集={os.path.basename(holdout)} adapter={args.adapter or 'BASE'} "
          f"dtype={args.dtype} limit={args.limit or 'all'} ...")
    model, tok = load_model(args.adapter, args.dtype)
    rows = [json.loads(l) for l in open(holdout, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    per_type = {}
    records = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        prompt = build_prompt(r["instruction"])
        ans = generate(model, tok, prompt, args.max_new_tokens)
        s = score(ans, r.get("must_contain", []))
        ov = overlap_score(ans, r.get("ref_out", "")) if r.get("ref_out") else None
        per_type.setdefault(r["type"], []).append((s, ov))
        records.append({"type": r["type"], "score": s, "overlap": ov, "answer": ans})
        ovs = f" overlap={ov:.3f}" if ov is not None else ""
        print(f"  [{i:02d}/{len(rows)}] {r['type']:6s} kw={s:.2f}{ovs}")

    def _avg(idx, vals):
        v = [x[idx] for x in vals if x[idx] is not None]
        return sum(v) / len(v) if v else None

    overall = sum(r["score"] for r in records) / len(records)
    ov_all = _avg(1, [(r["score"], r["overlap"]) for r in records])
    by_type = {k: {"kw": _avg(0, v), "overlap": _avg(1, v)} for k, v in per_type.items()}
    result = {"tag": tag, "data": os.path.basename(holdout),
              "overall_kw": overall, "overall_overlap": ov_all,
              "by_type": by_type, "records": records,
              "elapsed_s": round(time.time() - t0, 1), "n": len(records)}

    out_path = os.path.join(ROOT, "eval", f"result_{tag}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n=== 评测结果 [{tag}] 关键词={overall:.3f}", end="")
    print(f"  原文贴近度={ov_all:.3f} ===" if ov_all is not None else " ===")
    for k, v in sorted(by_type.items()):
        ov = f"  overlap={v['overlap']:.3f}" if v["overlap"] is not None else ""
        print(f"  {k:8s}: kw={v['kw']:.3f}{ov}")
    print(f"  明细 -> {out_path}")


if __name__ == "__main__":
    main()
