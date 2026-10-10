"""实测公开的代码 LoRA 卡：挂上去跑真实代码题，看它水平如何。

背景
----
bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16 与我们的配置**四项全同**：
  基座 Qwen/Qwen2.5-1.5B-Instruct | r16/alpha32 | dropout 0.05 | 全 7 个线性层
所以可以**零适配**挂到我们基座上（挂卡本质就是 W_base + (alpha/r)*B@A 一次矩阵加法）。

为什么专门测代码卡：代码**最吃正确性**——写错一个符号就运行不了，
不像文案写错了读起来还通顺。所以代码是检验"这张卡到底有没有真本事"的最好探针。

测什么
------
每题都不给答案，只给题目，然后用**本地沙箱实际执行**看跑不跑得通。
这是硬指标：模型嘴上说对，跑不通就是错。

对照组
------
  base  = 纯基座（无卡）
  hf    = 公开代码卡
  ours  = 我们自己的 out_lora_code_lang（若已训完）
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "Qwen2.5-1.5B-Instruct")
sys.path.insert(0, os.path.join(ROOT, "cards"))

# 真实代码题：题目里明确给规格，考察"会不会写"而不是"会不会说"
TASKS = [
    ("斐波那契闭包",
     "写一个 Python 函数 fib(n) 返回第 n 个斐波那契数（fib(0)=0, fib(1)=1），用迭代不要递归。\n"
     "只输出代码，不要解释。"),
    ("去重保序",
     "写一个 Python 函数 dedup(lst)，去掉重复元素并保持原有顺序，返回 list。\n"
     "只输出代码，不要解释。"),
    ("字典计数",
     "写一个 Python 函数 count_words(text)，返回字典，键是单词，值是出现次数（忽略大小写）。\n"
     "只输出代码，不要解释。"),
    ("二分查找",
     "写一个 Python 函数 binary_search(arr, target)，在升序数组里二分查找 target，"
     "找到返回下标、找不到返回 -1。要求处理空数组。\n只输出代码，不要解释。"),
    ("分组聚合",
     "写一个 Python 函数 group_anagrams(words)，把字母异位词分组，"
     "返回 [[词...], [词...]]，组内按字典序、组间按首词字典序。\n只输出代码，不要解释。"),
    ("带默认值的配置合并",
     "写一个 Python 函数 merge_config(default, override)，合并两个字典，"
     "override 优先，返回新字典不修改入参。\n只输出代码，不要解释。"),
]

TESTS = {
    "斐波那契闭包": "assert [fib(i) for i in range(10)] == [0,1,1,2,3,5,8,13,21,34]\nprint('OK')",
    "去重保序": "assert dedup([3,1,3,2,1]) == [3,1,2]\nassert dedup([]) == []\nprint('OK')",
    "字典计数": "assert count_words('a b a B') == {'a':2,'b':2}\nprint('OK')",
    "二分查找": "assert binary_search([1,3,5,7],5) == 2\nassert binary_search([1,3],9) == -1\nassert binary_search([],1) == -1\nprint('OK')",
    "分组聚合": "r=group_anagrams(['eat','tea','tan','ate','nat','bat'])\nassert sorted(r)==sorted([['ate','eat','tea'],['nat','tan'],['bat']])\nprint('OK')",
    "带默认值的配置合并": "d={'a':1,'b':2}; o={'b':3}\nassert merge_config(d,o)=={'a':1,'b':3}\nassert d=={'a':1,'b':2}\nprint('OK')",
}


def extract_code(text):
    """取最后一段代码：优先```块，否则找含 def 的行。"""
    if "```" in text:
        blocks = text.split("```")
        for i in range(len(blocks) - 1, 0, -1):
            b = blocks[i]
            b = b.split("\n", 1)[1] if "\n" in b else b
            if "def " in b or "import " in b or "class " in b:
                return b.strip()
    lines = [l for l in text.splitlines()]
    code = [l for l in lines if l.strip()]
    return "\n".join(code)


def run_test(code, test, timeout=15):
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code + "\n\n" + test)
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return (r.returncode == 0, (r.stderr or "").strip()[-160:])
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, str(e)[:160])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def build_prompt(q):
    return f"下面是一项任务。\n{q}\n输出："


@torch.no_grad()
def gen(model, tok, prompt, max_new=400):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="LoRA 卡目录；留空=只测基座")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or (os.path.basename(os.path.normpath(args.adapter)) if args.adapter else "base")

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                                low_cpu_mem_usage=True, trust_remote_code=True)
    if args.adapter:
        from lora_combine import apply_lora_deltas
        apply_lora_deltas(model, [os.path.normpath(args.adapter)], None, verbose=True)
    model.eval()

    print(f"\n===== 测试 [{tag}] =====")
    passed = 0
    results = []
    for name, q in TASKS:
        ans = gen(model, tok, build_prompt(q))
        code = extract_code(ans)
        ok, err = run_test(code, TESTS[name])
        passed += ok
        results.append({"task": name, "pass": bool(ok), "err": err, "code_len": len(code)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + ("" if ok else f"  <- {err[:90]}"))
    print(f"\n结果: {passed}/{len(TASKS)} 通过")
    outp = os.path.join(ROOT, "eval", f"hfcard_{tag}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump({"tag": tag, "passed": passed, "total": len(TASKS), "results": results},
              open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("明细 ->", outp)


if __name__ == "__main__":
    main()