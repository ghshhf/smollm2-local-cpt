"""
V7 推理卡 · 生产线
==================
目标：产出自带「思考链(CoT)」的训练数据，让 1.5B 小模型学会"先想后做"，
从而提升推理表现。这是目前对小模型最有效的推理增益杠杆（蒸馏强模型的推理过程）。

两种模式：
  1) 离线模板版（默认，无需任何外部模型）：
     把现有代码数据重排成「思考步骤 → 实现」格式，教模型结构化输出。
     质量有限（推理是模板化的），但零依赖、立即可训。
  2) 老师蒸馏版（--teacher 指向 OpenAI 兼容端点）：
     让强推理老师对每条任务先产出真实思考链，小模型学这个"想"的节奏。
     这才是真正的推理增益来源，需配置可用的老师端点。

用法：
  python build_dataset_v7_reason.py                        # 离线模板版 -> data/role_data_v7.jsonl
  python build_dataset_v7_reason.py --teacher http://127.0.0.1:8787/v1 --model <teacher_id>
  python build_dataset_v7_reason.py --src data/role_data_v6.jsonl --out data/role_data_v7.jsonl

训练（等 V3~V6 跑完后，作为下一槽位）：
  python train_lora.py --model_path model/Qwen2.5-1.5B-Instruct \
      --data_path data/role_data_v7.jsonl --output_dir out_lora_v7 \
      --max_steps 760 --lora_r 16 --lora_alpha 32 --max_seq_length 1024 \
      --per_device_batch_size 1 --grad_accumulation 4 --learning_rate 2e-4
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DEFAULT = os.path.join(ROOT, "data", "role_data_v6.jsonl")
OUT_DEFAULT = os.path.join(ROOT, "data", "role_data_v7.jsonl")
CODE_TYPES = {"code_fn", "code_distill", "code_paradigm"}


def detect_lang(out: str) -> str:
    m = re.search(r"```(\w+)", out)
    if m:
        return m.group(1)
    if "def " in out or "import " in out or "print(" in out:
        return "python"
    if "function" in out or "const " in out or "=>" in out or "document." in out:
        return "javascript"
    if "SELECT" in out.upper() or "INSERT" in out.upper():
        return "sql"
    if out.strip().startswith("$") or "bash" in out.lower():
        return "bash"
    return "python"


def extract_code(out: str) -> str:
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", out, re.S)
    return blocks[0].strip() if blocks else out.strip()


def template_reason(instruction: str, output: str) -> str:
    """离线模板：结构化思考步骤 + 原实现。"""
    lang = detect_lang(output)
    code = extract_code(output)
    lines = [l for l in code.splitlines() if l.strip()]
    feats = []
    if re.search(r"\bfor\b|\bwhile\b", code):
        feats.append("迭代/循环")
    if re.search(r"def \w+\(.*self", code) or "class " in code:
        feats.append("面向对象封装")
    if "{" in code and "}" in code and (":" in code):
        feats.append("字典/映射结构")
    if re.search(r"return .*\[.*for .*in", code):
        feats.append("列表推导/流式处理")
    feat_txt = "、".join(feats) if feats else "基础控制流"
    return (
        "思考步骤：\n"
        f"1. 明确目标：{instruction.strip()}\n"
        f"2. 选型：用 {lang} 实现，核心结构是{feat_txt}，约 {len(lines)} 行。\n"
        "3. 边界处理：先排除空/异常输入，再走主路径；兼顾时间复杂度与可读性。\n"
        "4. 实现如下：\n"
        f"{output.strip()}"
    )


def teacher_reason(instruction: str, output: str, teacher_url: str, teacher_model: str) -> str:
    """老师蒸馏：调用 OpenAI 兼容端点让强模型产出真实思考链。"""
    import urllib.request
    payload = {
        "model": teacher_model,
        "messages": [
            {"role": "system", "content": "你是推理教练。对用户给的任务，先输出'思考：'开头的逐步推理，"
                                          "再输出'实现：'开头的最终代码。只输出这两部分。"},
            {"role": "user", "content": f"任务：{instruction}\n参考实现：\n{output}"},
        ],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        teacher_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--teacher", default="", help="OpenAI 兼容端点，如 http://127.0.0.1:8787/v1")
    ap.add_argument("--model", default="", help="老师模型 id（--teacher 时必填）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    code_rows = [r for r in rows if r.get("_type") in CODE_TYPES]
    out = []
    for r in code_rows:
        instr = r.get("instruction", "").strip()
        o = r.get("output", "").strip()
        if not instr or not o:
            continue
        if args.teacher and args.model:
            try:
                cot = teacher_reason(instr, o, args.teacher, args.model)
                src_tag = f"v7_reason(teacher:{args.model})"
            except Exception as e:
                print(f"[warn] 老师调用失败，回退模板: {e}")
                cot = template_reason(instr, o)
                src_tag = "v7_reason(offline-template)"
        else:
            cot = template_reason(instr, o)
            src_tag = "v7_reason(offline-template)"
        out.append({
            "instruction": instr,
            "output": cot,
            "_type": "reason_cot",
            "_universe": "代码",
            "_src": src_tag,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"V7 推理卡生成 {len(out)} 条 -> {args.out}")
    print(f"  模式: {'老师蒸馏' if (args.teacher and args.model) else '离线模板'}（离线模板质量有限，建议配 --teacher 重跑以获得真实推理增益）")


if __name__ == "__main__":
    main()
