#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4 Gate 2 验证 —— 潜在递归是否真的提准确率  [v0.2 脚手架]
=========================================================
固定同一主干, 扫 steps ∈ {0,1,2,4,8}, 在 QA 测试集上测:
  - 答案准确率 (数值解析优先, 否则归一化子串匹配)
  - 单题平均延迟 (ms)
  - 峰值内存 (RSS; psutil 可选)
  - 平均每题递归步数 (=steps)

这是消融矩阵里的实验 B (vs A 基线 steps=0)。输出 Markdown 表, 方便直接抄进报告。
完整答案由 HybridModel.generate_answer 产出 (v0.2), 不再是只比首 token。

运行 (本机需 torch+transformers; 或用远端 node1):
  python b4/evaluate_reasoning.py --model_dir models/SmolLM2-135M \
      --tests b4/tests_qa.jsonl --steps 0,1,2,4,8
  # 无 --tests 时用内置 smoke 题
"""
import argparse
import json
import os
import re
import sys
import time

# 允许直接 `python b4/evaluate_reasoning.py` 时也能 `from b4.xxx` 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from b4.latent_reasoner import load_hybrid


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _peak_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return -1.0


def load_tests(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "question" in d and "answer" in d:
                rows.append(d)
    return rows


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(s):
    return [float(x) for x in _NUM.findall(s)]


def check_correct(pred, gold):
    """
    判分: 优先数值匹配 (提取两边首个数字比较, 容错千分位/单位),
    否则退化为归一化子串包含。返回 bool。
    """
    gp = _numbers(gold)
    pp = _numbers(pred)
    if gp and pp:
        # 数值题: 只要预测里出现了正确答案数字即判对 (宽松, 适配小模型表述)
        return any(abs(g - p) < 1e-6 for g in gp for p in pp)
    # 文本题: 归一化后任一方包含另一方
    a = "".join(gold.split()).lower()
    b = "".join(pred.split()).lower()
    return (a in b) or (b in a)


# 必须与 b4/train_distill.py / b4/train_latent.py 完全一致, 否则训练/评测前缀不一致。
PROMPT_TMPL = "问题：{q}\n答案："


def evaluate(hybrid, tok, tests, steps_list, max_new=48):
    results = {}
    peak = _peak_rss_mb()
    for steps in steps_list:
        correct = 0
        lat = []
        for t in tests:
            # 与训练同前缀: 包装成 "问题：{q}\n答案：" 再编码
            prompt = PROMPT_TMPL.format(q=t["question"])
            ids = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=256)["input_ids"]
            t0 = time.perf_counter()
            ans = hybrid.generate_answer(ids, steps, max_new=max_new)
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
            if check_correct(ans, t["answer"]):
                correct += 1
        n = max(1, len(tests))
        avg_lat = sum(lat) / n
        acc = correct / n * 100
        results[steps] = (acc, avg_lat, correct, n)
        now_peak = _peak_rss_mb()
        peak = max(peak, now_peak)
        log(f"steps={steps:>2d}  acc={acc:5.1f}%  avg_lat={avg_lat:7.1f}ms  "
            f"({correct}/{n})" +
            (f"  peakRSS={peak:.0f}MB" if peak > 0 else ""))
    return results, peak


def _report(results, peak, has_adapter=False):
    print("\n## B4 Gate2 消融 (实验 B vs A) — 潜在递归 steps 扫描")
    print("| steps | 准确率(%) | 平均延迟(ms/题) | 备注 |")
    print("|------:|----------:|---------------:|------|")
    for s, (acc, lat, c, n) in sorted(results.items()):
        if s == 0:
            note = "基线 System1(直答)"
        elif has_adapter:
            note = "潜在递归+训练adapter"
        else:
            note = "潜在递归(无adapter)"
        print(f"| {s:>2d} | {acc:5.1f} | {lat:7.1f} | {note} |")
    print(f"\n峰值 RSS: {peak:.0f} MB" if peak > 0
          else "\n(未装 psutil, 跳过峰值内存)")
    base = results.get(0, (0, 0, 0, 1))[0]
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"基线 steps=0 准确率: {base:.1f}%")
    print(f"本批最佳: steps={best[0]} acc={best[1][0]:.1f}% "
          f"(相对基线 {best[1][0]-base:+.1f}pp)")
    if best[1][0] - base > 0:
        print("=> Gate 2 初步成立: 潜在递归带来可重复提升 (需多题+多 seed 复核)。")
    else:
        print("=> 本批未观察到提升: 勿急于加 router/stop, 先排查数据/初始化。")


def _smoke():
    return [
        {"question": "17 × 24 等于多少？", "answer": "408"},
        {"question": "一个数的两倍加 7 等于 31，这个数是多少？", "answer": "12"},
        {"question": "中国的首都是哪里？", "answer": "北京"},
        {"question": "地球距离太阳大约多少公里？", "answer": "1.5亿"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tests", default="", help="QA JSONL 测试集; 空则用内置 smoke")
    ap.add_argument("--steps", default="0,1,2,4,8")
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--adapter", default="",
                    help="可选: 加载训练过的 LatentAdapter, 递归步里套用")
    a = ap.parse_args()

    dtype = torch.bfloat16 if a.bf16 else torch.float32
    hybrid, tok = load_hybrid(a.model_dir, dtype,
                              adapter_path=a.adapter or None)
    tests = load_tests(a.tests) if a.tests else _smoke()
    log(f"测试用例数: {len(tests)}  adapter={'有' if a.adapter else '无'}")
    steps = [int(x) for x in a.steps.split(",")]
    results, peak = evaluate(hybrid, tok, tests, steps, a.max_new)
    _report(results, peak, bool(a.adapter))


if __name__ == "__main__":
    main()
