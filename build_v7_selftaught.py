"""
V7 推理卡 · 可扩展自教管道（持续强化用）
========================================
数据来源：role_data_v6.jsonl 的全部 code_* 任务（1958 条）
填充策略（按优先级）：
  1) 范式任务(20 条) -> 用 agent 亲手写的真推理种子 build_v7_seed_selfteacher.py 产出
  2) 其余任务      -> 若给 --teacher 端点+--model，调强模型写真实 CoT（真强化）
                    否则用 启发式分析器（基于代码实际结构的任务级推理，诚实标注 heuristic）
产物：data/role_data_v7_selftaught.jsonl

用法：
  python build_v7_selftaught.py                         # 种子(真) + 其余(启发式)
  python build_v7_selftaught.py --teacher URL --model M  # 种子(真) + 其余(真老师 CoT)
"""
import argparse, json, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v6.jsonl")
SEED = os.path.join(ROOT, "data", "role_data_v7_selftaught_seed.jsonl")
EXTRA = os.path.join(ROOT, "data", "v7_self_cot_extra.jsonl")
BANK2 = os.path.join(ROOT, "data", "v7_self_cot_extra2.jsonl")
BANK3 = os.path.join(ROOT, "data", "v7_self_cot_extra3.jsonl")  # 跨领域多样性库(系统/数学/测试/安全/存储/工程方法)
OUT = os.path.join(ROOT, "data", "role_data_v7_selftaught.jsonl")
CODE_TYPES = {"code_fn", "code_distill", "code_paradigm"}


def detect_lang(out):
    m = re.search(r"```(\w+)", out)
    if m: return m.group(1)
    if "def " in out or "import " in out: return "python"
    if "function" in out or "const " in out or "=>" in out: return "javascript"
    if "SELECT" in out.upper(): return "sql"
    if out.strip().startswith("$") or "bash" in out.lower(): return "bash"
    return "python"


def extract_code(out):
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", out, re.S)
    return blocks[0].strip() if blocks else out.strip()


def heuristic_reason(instr, code):
    """诚实的启发式：基于代码真实结构产出任务级推理，标注 heuristic。"""
    lang = detect_lang(code)
    pts = []
    # 算法模式识别
    if re.search(r"for\s+\w+\s+in\s+range\(len\(", code) and ("while" in code or "mid" in code):
        pts.append("二分/双指针收缩区间")
    if "stack" in code or "st =" in code or "st.append" in code:
        pts.append("用栈维护单调/嵌套关系")
    if "heapq" in code or "MinHeap" in code or "_up" in code or "_down" in code:
        pts.append("堆维护极值/优先队列")
    if "cache" in code.lower() and ("order" in code or "lru" in code.lower()):
        pts.append("LRU 用 dict+顺序列表淘汰最久未用")
    if "WeakMap" in code or "seen" in code:
        pts.append("记忆化+WeakMap 打破循环引用")
    if "AbortController" in code or "timeout" in code.lower() or "ClientTimeout" in code:
        pts.append("超时控制防止请求永久挂起")
    if "retry" in code.lower() or "退避" in instr or "backoff" in code.lower():
        pts.append("指数退避重试隔离瞬时故障")
    if "Semaphore" in code or "limit" in code.lower() or "mapLimit" in code:
        pts.append("信号量/worker 池限制并发度")
    if "async" in code or "await" in code:
        pts.append("异步非阻塞，避免线程切换开销")
    if not pts:
        pts.append("基础控制流+函数封装")
    feat = "、".join(pts)
    return (
        "思考步骤（启发式分析，非 LLM 推理）：\n"
        f"1. 任务：{instr.strip()}\n"
        f"2. 语言/结构：{lang}，识别到的模式——{feat}。\n"
        "3. 关键边界：空输入、重复/极值、资源清理（关闭/释放）、异常路径都要覆盖。\n"
        "4. 实现如下：\n"
        f"{code}"
    )


def teacher_reason(instr, code, url, model):
    import urllib.request
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是推理教练。对任务先输出'思考：'开头的逐步推理（点出取舍/边界/为什么这么写），再输出'实现：'开头的代码。只输出这两部分。"},
            {"role": "user", "content": f"任务：{instr}\n参考实现：\n{code}"},
        ], "temperature": 0.3,
    }
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    code_rows = [r for r in rows if r.get("_type") in CODE_TYPES]

    # 1) 真推理字典：范式种子(20) + 扩展真推理(22) + 第二扩展库(58) 合并，instruction -> 整条数据
    genuine = {}
    for path in (SEED, EXTRA, BANK2, BANK3):
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                s = json.loads(l)
                genuine.setdefault(s["instruction"], s)  # 前者优先
    print(f"  真推理字典载入: {len(genuine)} 条 (范式种子 + 扩展手写 + 第二扩展库)")

    out = list(genuine.values())
    filled = 0
    for r in code_rows:
        instr = r.get("instruction", "").strip()
        if instr in genuine:
            continue  # 已是真老师推理
        code = extract_code(r.get("output", ""))
        if not instr or not code:
            continue
        if args.teacher and args.model:
            try:
                cot = teacher_reason(instr, code, args.teacher, args.model)
                src = f"v7_selftaught(teacher:{args.model})"
            except Exception as e:
                print(f"[warn] 老师失败回退启发式: {e}")
                cot = heuristic_reason(instr, code)
                src = "v7_selftaught(heuristic-fallback)"
        else:
            cot = heuristic_reason(instr, code)
            src = "v7_selftaught(heuristic)"
        out.append({"instruction": instr, "output": cot,
                    "_type": "reason_cot", "_universe": "代码", "_src": src})
        filled += 1

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    real = sum(1 for o in out if "teacher:self" in o["_src"] or "teacher:" in o["_src"])
    print(f"V7 自教数据集 -> {OUT}  总计 {len(out)} 条")
    print(f"  真老师推理(agent/LLM): {real} 条 | 启发式: {len(out)-real} 条 | 本次新增填充: {filled}")


if __name__ == "__main__":
    main()
