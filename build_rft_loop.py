"""
RFT 闭环（Rejection-sampling Fine-Tuning / STaR）
=================================================
2026 主线：SFT 之后必须接「可验证信号纠偏」(RLVR / 过程奖励)。
本机 CPU 跑不了真 RL(需要 rollout × 反传 × 价值网络)，但**我们的任务是代码——代码天然可验证**。

于是用「verifier」替代 reward model：
    模型自采样 K 条 → 抽【答案】区代码 → 沙箱执行 → 只保留跑得通的 → 回流做成 SFT 数据

这就是 Best-of-N 蒸馏 / RFT(STaR)：不需要 RL infra，直接喂给现有 train_lora.py 即可。
同时对齐 PROSE(2026.09) 的教训——**只看最终答案会被 exploit**，因此除了"能跑通"，
还要求代码里有实质内容(非空/非纯注释)，并把原问题的【思考】结构一并保留作为过程信号。

⚠️ 安全：本脚本会**执行模型生成的代码**。只在能接受这一步的环境里跑，
产物也建议过一眼再用。仅对 Python 代码块做执行验证(JS/SQL/Bash 无法干净沙箱，诚实地跳过)。

用法（务必在训练空档跑，别和训练抢 ~6GB 内存）：
  python build_rft_loop.py --adapter out_lora_v7 --n_tasks 60 --k 4
  python build_rft_loop.py --adapter out_lora_v7 --n_tasks 200 --k 4 --out data/role_data_rft_v2.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")
SRC = os.path.join(ROOT, "data", "role_data_v8.jsonl")

CODE_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
ANS_RE = re.compile(r"【答案】\n(.*?)【/答案】", re.DOTALL)
THINK_RE = re.compile(r"【思考】\n(.*?)\n【/思考】", re.DOTALL)


def exec_code(code: str, timeout: int = 6):
    """本地沙箱跑一段 Python；返回 (ok, err)。"""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return (r.returncode == 0, (r.stderr or r.stdout).strip()[:200])
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:  # noqa: BLE001
        return (False, str(e)[:200])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def has_substance(code: str) -> bool:
    """PROSE 教训：防 exploit——跑通但没内容(纯注释/空实现)不该拿奖励。"""
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    return len(lines) >= 3 and any(("=" in l or "def " in l or "return" in l or "import " in l) for l in lines)


def load_model(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def sample(model, tok, prompt: str, k: int, max_new_tokens: int):
    outs = []
    ids0 = tok(prompt, return_tensors="pt").input_ids
    for _ in range(k):
        out = model.generate(ids0, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.85, top_p=0.95, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(out[0][ids0.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="LoRA 卡目录；留空=仅基座")
    ap.add_argument("--n_tasks", type=int, default=60, help="抽样多少条任务做 RFT")
    ap.add_argument("--k", type=int, default=4, help="每条采样几条候选")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "role_data_rft_v1.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    # 只要含 Python 代码块的条目：verifier 才有用
    pool = [r for r in rows if "```python" in r["output"] or "```py" in r["output"]]
    random.seed(args.seed)
    random.shuffle(pool)
    tasks = pool[:args.n_tasks]
    print(f"[rft] 候选任务池={len(pool)}，本次取 {len(tasks)} 条 × k={args.k}")

    print(f"[rft] 加载模型 adapter={args.adapter or 'BASE'} ...")
    model, tok = load_model(args.adapter or None)

    kept, stats = [], {"total": 0, "pass_exec": 0, "has_code": 0}
    for i, r in enumerate(tasks, 1):
        prompt = f"下面是一项任务。\n{r['instruction']}\n输出："
        for gen in sample(model, tok, prompt, args.k, args.max_new_tokens):
            stats["total"] += 1
            ans_m = ANS_RE.search(gen)
            body = ans_m.group(1) if ans_m else gen
            codes = CODE_RE.findall(body)
            if not codes:
                continue
            stats["has_code"] += 1
            code = max(codes, key=len).strip()
            if not has_substance(code):
                continue
            ok, err = exec_code(code)
            if not ok:
                continue
            stats["pass_exec"] += 1
            think_m = THINK_RE.search(gen)
            reasoning = think_m.group(1).strip() if think_m else ""
            kept.append({
                "instruction": r["instruction"],
                "output": f"【思考】\n{reasoning}\n【/思考】\n【答案】\n```python\n{code}\n```\n【/答案】"
                if reasoning else f"【答案】\n```python\n{code}\n```\n【/答案】",
                "_type": "rft_verified",
                "_universe": r.get("_universe", ""),
                "_src": f"rft(verified:exec, adapter={os.path.basename(args.adapter) if args.adapter else 'base'})",
            })
        if i % 10 == 0 or i == len(tasks):
            print(f"  进度 {i}/{len(tasks)}  累计通过={stats['pass_exec']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rate = stats["pass_exec"] / stats["total"] if stats["total"] else 0
    print(f"\n[rft] 完成 -> {args.out}")
    print(f"  采样总数 {stats['total']}｜含代码块 {stats['has_code']}｜执行通过 {stats['pass_exec']} ({rate:.1%})")
    print(f"  产出验证过的训练样本 {len(kept)} 条")


if __name__ == "__main__":
    main()
