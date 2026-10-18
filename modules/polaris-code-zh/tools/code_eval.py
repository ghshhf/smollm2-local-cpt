"""编程模块两层验收：格式层（模块职责）+ 内容层（主干职责）。

格式层判据 —— 模块必须保证：
  1. 生成以 <|im_end|> 收尾或不超长截断
  2. 含 ``` 代码块，且带语言标注
  3. 代码块外有说明文字（不是光秃秃一块代码）

内容层判据 —— 主干提供知识，模块负责把它搬出来：
  strict : 生成的代码单独执行，stdout 与 expect 逐字符相等（归一化空白）
  loose  : expect 的关键片段出现在 stdout 里
  fmt    : 无 expect 的题，只看代码里有没有出现规定的 API 关键词

用法：
  # 裸模型基线（必须先跑，否则不知道模块带来了什么）
  python tools/code_eval.py --model backups/0.3B/polaris-03b_20261018 --tag bare03b

  # 挂模块
  python tools/code_eval.py --model backups/0.3B/polaris-03b_20261018 \
      --adapter out_lora_code_03b --tag code03b
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

SYSTEM = "你是北极星，主人的私有编程助手。回答给可运行的代码，配一两句关键说明。"
IM_END = "<|im_end|>"

LANG_ALIAS = {
    "python": "python", "py": "python",
    "javascript": "js", "js": "js", "node": "js",
    "bash": "bash", "sh": "bash", "shell": "bash",
    "sql": "sql",
}


def build_prompt(q):
    """强制手工 ChatML，与训练数据的拼接方式逐字同构。

    不能用 tokenizer.apply_chat_template：训练集是预先拼好的 text 字段，
    一旦模板与数据不一致，评测就不是在量同一个协议。
    """
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def extract_code(text):
    """取第一个带语言标注的围栏块；没有标注也接受。"""
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", text, re.S)
    if not blocks:
        return None, None
    for lang, body in blocks:
        if lang.strip():
            return LANG_ALIAS.get(lang.strip().lower(), lang.strip().lower()), body
    lang, body = blocks[0]
    return "", body


def judge_format(text, finished):
    code_lang, code = extract_code(text)
    outside = re.sub(r"```[A-Za-z0-9_+-]*\s*\n.*?```", "", text, flags=re.S).strip()
    outside = outside.replace(IM_END, "").strip()
    return {
        "has_code_block": code is not None,
        "lang_tagged": bool(code_lang),
        "lang": code_lang or "",
        "has_prose": len(outside) >= 4,
        "has_eos": finished,
        "code": code or "",
    }


def norm(s):
    return re.sub(r"\s+", "", s or "")


_BASH = None


def find_bash():
    """找一个真 POSIX shell。Windows 上的 system32\\bash.exe 是 WSL 存根，
    没装发行版时会直接失败，必须排除。"""
    global _BASH
    if _BASH:
        return _BASH
    cands = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\WorkBuddy\resources\vendor\PortableGit\usr\bin\bash.exe",
        r"C:\Program Files\WorkBuddy\resources\vendor\PortableGit\bin\bash.exe",
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                p = subprocess.run([c, "-c", "echo ok"], capture_output=True,
                                   text=True, timeout=5)
                if p.returncode == 0 and "ok" in (p.stdout or ""):
                    _BASH = c
                    return c
            except Exception:
                continue
    for c in ("bash",):
        w = shutil.which(c)
        if w and "system32" not in w.lower():
            _BASH = w
            return w
    _BASH = ""
    return ""


def run_code(lang, code, timeout=15):
    """返回 (ok, stdout, stderr)。ok 表示进程正常结束（returncode 0）。"""
    if not code:
        return False, "", "no code"
    lang = LANG_ALIAS.get((lang or "").lower(), (lang or "").lower())
    suffix = {"python": ".py", "js": ".js", "bash": ".sh"}.get(lang)
    if suffix is None:
        return False, "", f"skip-lang:{lang}"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=os.environ.get("TEMP", "."))
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        if lang == "python":
            cmd = [sys.executable, path]
        elif lang == "js":
            cmd = ["node", path]
        else:
            sh = find_bash()
            if not sh:
                return False, "", "no-posix-shell"
            cmd = [sh, path]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, env=env, errors="replace")
        return p.returncode == 0, p.stdout or "", (p.stderr or "")[:400]
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except FileNotFoundError as e:
        return False, "", f"runner-missing:{e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def judge_content(task, fmt):
    """返回 (verdict, detail)。verdict ∈ pass/strict_fail/loose_pass/fmt_pass/fail/skip"""
    code = fmt["code"]
    # 模型漏写语言标注时，用题目声明的语言兜底（不该因为格式瑕疵丢掉内容判定）
    lang = fmt["lang"] or task.get("lang", "")
    if not code:
        return "fail", "无代码块"
    expect = task.get("expect")
    if expect is None:
        # 无标准输出的题：只看 API 关键词是否出现在代码里
        low = code.lower()
        hits = [k for k in task.get("loose", []) if k.lower() in low]
        need = max(1, (len(task.get("loose", [])) + 1) // 2)
        return ("fmt_pass", f"命中 {len(hits)}/{len(task.get('loose', []))}") if len(hits) >= need \
            else ("fail", f"仅命中 {hits}")
    ok, out, err = run_code(lang, code)
    if not ok:
        return "strict_fail", f"执行失败: {err[:120] or 'returncode!=0'}"
    if norm(out) == norm(expect):
        return "pass", "stdout 完全匹配"
    loose = task.get("loose", [])
    hit = [k for k in loose if norm(k) in norm(out)]
    if loose and len(hit) == len(loose):
        return "loose_pass", f"stdout={out.strip()[:60]!r} 未精确匹配但关键片段全中"
    return "strict_fail", f"stdout={out.strip()[:80]!r} 期望={expect[:60]!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="data/code_holdout_v1.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt,
                                                trust_remote_code=True)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"[info] 已挂模块: {a.adapter}")
    model.eval()
    cfg = model.config
    print(f"[info] hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"inter={cfg.intermediate_size} vocab={cfg.vocab_size}")
    eos = tok.convert_tokens_to_ids(IM_END)

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    if a.limit:
        tasks = tasks[: a.limit]

    results = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        prompt = build_prompt(t["query"])
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id or eos)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        finished = IM_END in gen
        gen = gen.split(IM_END)[0]
        fmt = judge_format(gen, finished)
        verdict, detail = judge_content(t, fmt)
        results.append({"id": t["id"], "verdict": verdict, "detail": detail,
                        "format": {k: v for k, v in fmt.items() if k != "code"},
                        "text": gen[:1200]})
        print(f"[{i:2d}/{len(tasks)}] {t['id']:<14} {verdict:<12} "
              f"code={int(fmt['has_code_block'])} lang={fmt['lang'] or '-':<7} "
              f"prose={int(fmt['has_prose'])} eos={int(finished)}  {detail[:70]}")

    dt = time.time() - t0
    n = len(results)
    agg = {k: sum(1 for r in results if r["verdict"] == k)
           for k in ("pass", "loose_pass", "fmt_pass", "strict_fail", "fail")}
    fmt_rate = {
        "code_block": sum(1 for r in results if r["format"]["has_code_block"]) / n,
        "lang_tagged": sum(1 for r in results if r["format"]["lang_tagged"]) / n,
        "has_prose": sum(1 for r in results if r["format"]["has_prose"]) / n,
        "has_eos": sum(1 for r in results if r["format"]["has_eos"]) / n,
    }
    print("\n" + "=" * 70)
    print(f"[{a.tag}] {n} 题  用时 {dt / 60:.1f} 分  ({dt / n:.0f}s/题)")
    print(f"  格式层: 代码块 {fmt_rate['code_block']:.0%}  语言标注 {fmt_rate['lang_tagged']:.0%}  "
          f"配说明 {fmt_rate['has_prose']:.0%}  正常收尾 {fmt_rate['has_eos']:.0%}")
    print(f"  内容层: strict {agg['pass']}  loose {agg['loose_pass']}  "
          f"fmt-only {agg['fmt_pass']}  执行失败 {agg['strict_fail']}  无产出 {agg['fail']}")
    print(f"  严格通过率 {agg['pass']}/{n} = {agg['pass'] / n:.0%}   "
          f"宽松通过率 {(agg['pass'] + agg['loose_pass'] + agg['fmt_pass'])}/{n} = "
          f"{(agg['pass'] + agg['loose_pass'] + agg['fmt_pass']) / n:.0%}")
    print("=" * 70)

    payload = {"tag": a.tag, "model": a.model, "adapter": a.adapter,
               "n": n, "seconds": round(dt, 1), "agg": agg,
               "format_rate": {k: round(v, 4) for k, v in fmt_rate.items()},
               "results": results}
    out = a.out or f"logs/code_eval_{a.tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[info] 明细写入 {out}")


if __name__ == "__main__":
    main()
