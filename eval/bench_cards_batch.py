"""批量评测多张公开 LoRA 卡：一次加载基座，逐张卡叠加后测同一套题。

为什么这个设计
--------------
每张卡都重新加载基座要 30-60s（fp32 1.5B 约6GB），14 张卡就是 14 分钟纯等待。
改成「基座只加载一次，每测一张卡就把ΔW 加上去，测完减回去」，
所有卡测完只需加载 1 次。减法就是加负权重，复用同一个 ΔW 累加器即可。

判定方式：每题都**本地沙箱实际执行**，跑不跑得通是硬指标（代码最吃正确性，
嘴上说对不算数）。

用法：
  python eval/bench_cards_batch.py# 测所有已下载的卡 + 基座基线
  python eval/bench_cards_batch.py --only code # 只测名字含 code 的卡
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")
CARDS_DIR = os.path.join(ROOT, "hf_cards")
sys.path.insert(0, os.path.join(ROOT, "cards"))

# 三组题：算法 / 数据处理 / 正确性陷阱（专门测边界处理能力）
TASKS = [
    ("斐波那契闭包", "写一个 Python 函数 fib(n) 返回第 n 个斐波那契数（fib(0)=0, fib(1)=1），用迭代不要递归。\n只输出代码，不要解释。",
     "assert [fib(i) for i in range(10)] == [0,1,1,2,3,5,8,13,21,34]\nprint('OK')"),
    ("去重保序", "写一个 Python 函数 dedup(lst)，去掉重复元素并保持原有顺序，返回 list。\n只输出代码，不要解释。",
     "assert dedup([3,1,3,2,1]) == [3,1,2]\nassert dedup([]) == []\nprint('OK')"),
    ("字典计数", "写一个 Python 函数 count_words(text)，返回字典，键是单词，值是出现次数（忽略大小写）。\n只输出代码，不要解释。",
     "assert count_words('a b a B') == {'a':2,'b':2}\nprint('OK')"),
    ("二分查找", "写一个 Python 函数 binary_search(arr, target)，在升序数组里二分查找 target，找到返回下标、找不到返回 -1。要求处理空数组。\n只输出代码，不要解释。",
     "assert binary_search([1,3,5,7],5) == 2\nassert binary_search([1,3],9) == -1\nassert binary_search([],1) == -1\nprint('OK')"),
    ("分组聚合", "写一个 Python 函数 group_anagrams(words)，把字母异位词分组，返回 [[词...], [词...]]，组内按字典序、组间按首词字典序。\n只输出代码，不要解释。",
     "r=group_anagrams(['eat','tea','tan','ate','nat','bat'])\nassert sorted(r)==sorted([['ate','eat','tea'],['nat','tan'],['bat']])\nprint('OK')"),
    ("字典合并", "写一个 Python 函数 merge_config(default, override)，合并两个字典，override 优先，返回新字典不修改入参。\n只输出代码，不要解释。",
     "d={'a':1,'b':2}; o={'b':3}\nassert merge_config(d,o)=={'a':1,'b':3}\nassert d=={'a':1,'b':2}\nprint('OK')"),
    ("边界：单元素与重复", "写一个 Python 函数 dedup_sorted(lst)，lst 是升序列表，去掉连续重复元素，返回新列表。原列表不得修改。\n只输出代码，不要解释。",
     "src=[1,1,2,3,3,3]; r=dedup_sorted(src)\nassert r==[1,2,3], r\nassert src==[1,1,2,3,3,3]\nassert dedup_sorted([])==[]\nassert dedup_sorted([7])==[7]\nprint('OK')"),
    ("异常处理", "写一个 Python 函数 safe_div(a, b)，返回 a/b；若 b 为 0 返回 None 而非抛异常。\n只输出代码，不要解释。",
     "assert safe_div(6,3)==2\nassert safe_div(1,0) is None\nassert safe_div(0,5)==0\nprint('OK')"),
    ("字符串反转不误伤Unicode", "写一个 Python 函数 rev(s)，返回字符串 s 的反转。\n只输出代码，不要解释。",
     "assert rev('abc')=='cba'\nassert rev('')==''\nassert rev('a b')=='b a'\nprint('OK')"),
    ("累加器闭包", "写一个 Python 函数 make_counter()，返回一个函数，调用它返回 1,2,3,...（从1开始每次+1）。\n只输出代码，不要解释。",
     "c=make_counter()\nassert [c(),c(),c()]==[1,2,3]\nprint('OK')"),
]


def extract_code(text):
    if "```" in text:
        blocks = text.split("```")
        for i in range(len(blocks) - 1, 0, -1):
            b = blocks[i]
            b = b.split("\n", 1)[1] if "\n" in b else b
            if "def " in b or "import " in b or "class " in b:
                return b.strip()
    return "\n".join(l for l in text.splitlines() if l.strip())


def run_test(code, test, timeout=15):
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + test)
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return (r.returncode == 0, (r.stderr or "").strip()[-120:])
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, str(e)[:120])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@torch.no_grad()
def gen(model, tok, q, max_new=380):
    prompt = f"下面是一项任务。\n{q}\n输出："
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run_suite(model, tok, tag):
    passed, results = 0, []
    for name, q, test in TASKS:
        code = extract_code(gen(model, tok, q))
        ok, err = run_test(code, test)
        passed += ok
        results.append({"task": name, "pass": bool(ok), "err": err})
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  => [{tag}] {passed}/{len(TASKS)}")
    return passed, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只测名字含此串的卡")
    ap.add_argument("--out", default="hf_cards_report.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                                low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()

    report = {}
    print("\n########## 基座基线 ##########")
    report["__base__"] = dict(zip(("passed", "results"), run_suite(model, tok, "base")))

    from lora_combine import read_adapter_weights, apply_lora_deltas

    cards = [d for d in sorted(glob.glob(os.path.join(CARDS_DIR, "*")))
             if os.path.isfile(os.path.join(d, "adapter_model.safetensors"))]
    if args.only:
        cards = [d for d in cards if args.only.lower() in os.path.basename(d).lower()]
    print(f"\n########## 待测卡 {len(cards)} 张 ##########")

    for d in cards:
        name = os.path.basename(d)
        print(f"\n--- {name} ---")
        try:
            apply_lora_deltas(model, [d], None, verbose=False)
        except Exception as e:
            print(f"  挂载失败: {type(e).__name__} {str(e)[:90]}")
            continue
        passed, results = run_suite(model, tok, name)
        report[name] = {"passed": passed, "results": results}
        # 还原：加负权重把ΔW 减回去，供下一张卡使用
        apply_lora_deltas(model, [d], [-1.0], verbose=False)

    outp = os.path.join(ROOT, "eval", args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(report, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n===== 汇总(按通过数排序) =====")
    base_p = report["__base__"]["passed"]
    for k, v in sorted(report.items(), key=lambda x: -x[1]["passed"]):
        delta = "" if k == "__base__" else f"  (基线 {base_p} → {v['passed']-base_p:+d})"
        print(f"  {v['passed']}/{len(TASKS)}{delta}  {k}")
    print("明细 ->", outp)


if __name__ == "__main__":
    main()