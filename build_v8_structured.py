"""
V8 数据集构建（2026 前沿对齐版）
================================
对 V7(2011 条)做**纯数据层升级**，不动 trainer、不碰正在跑的训练：

1) 过程/步骤标签（对齐 PROSE 2026「奖励每一步推理，而非只看最终答案」）
   ——把每条输出里的编号步骤解析出来，按关键词打 step-type 标签：
      [界定约束] [方案选择] [结构选型] [边界处理] [验证] [分析]
   让 loss 作用在「有结构的步骤」上，而不是一坨无差别文本。

2) SWITCH 边界令牌（对齐 SWITCH 2026「<latent>…</latent> 把隐推理区包起来」）
   ——用不会被分词器吞掉的全角标记把「思考区 / 答案区」显式分开：
      【思考】…【/思考】  【答案】…【/答案】
   好处：① 模型学会先想完再答；② 评测/使用时可精确抽取答案区（不再靠猜最后一个代码块）；
        ③ 未来压推理 token 时，思考区是天然的压缩对象。
   ⚠️ 没用 <latent> 这类特殊 token：加特殊 token 要 resize 词嵌入，
      会让基座加载/V3-V6 已有卡的权重对不上。全角标记是同效且零风险的替代。

输出：data/role_data_v8.jsonl（指令不变，output 换成带结构的新格式）
"""
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v7_selftaught.jsonl")
OUT = os.path.join(ROOT, "data", "role_data_v8.jsonl")

STEP_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)

# 步骤类型判定（有序：先命中先用）
STEP_RULES = [
    ("边界处理", ["边界", "空", "冲突", "重复", "极值", "异常", "失败", "释放", "清理",
                  "已存在", "覆盖", "环", "超时", "精度", "退化"]),
    ("结构选型", ["复杂度", "索引", "哈希", "栈", "队列", "递归", "字典", "数组", "链表",
                  "树", "堆", "矩阵", "向量", "O(n", "O(1", "O(log", "结构"]),
    ("界定约束", ["必须", "要求", "约束", "不能", "幂等", "至少", "保证", "只能", "前提"]),
    ("方案选择", ["用", "选", "拆", "改", "替代", "对比", "权衡", "取舍", "优先", "为什么",
                  "避免", "相比"]),
    ("验证", ["测试", "验证", "断言", "检查", "确认", "跑一遍", "单测"]),
]


def tag_step(text: str) -> str:
    for name, kws in STEP_RULES:
        if any(k in text for k in kws):
            return name
    return "分析"


def parse_output(output: str):
    """把旧 output 拆成 (过程步骤列表, 答案正文)。"""
    codes = CODE_RE.findall(output)

    # 去掉代码块后的文本，用于抽步骤
    rest = CODE_RE.sub("", output)

    steps, body_lines = [], []
    header_kw = ("思考步骤", "实现：", "实现如下", "实现:")
    for line in rest.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(header_kw):
            continue
        m = STEP_RE.match(s)
        if m:
            steps.append(m.group(2).strip())
        else:
            body_lines.append(s)

    answer = ""
    if codes:
        answer = "\n\n".join(c.strip() for c in codes if c.strip())
    elif body_lines:
        answer = "\n".join(body_lines).strip()

    return steps, answer


def is_boilerplate(step: str, instruction: str) -> bool:
    """剔掉启发式条目里的样板噪声：它们不是推理，只是占位/复述。"""
    s = step.strip().rstrip("：:").strip()
    # 纯占位头
    if s in {"实现如下", "实现", "答案", "输出", "代码", "代码如下"}:
        return True
    # 启发式条目的固定开场：「任务：<把指令复述一遍>」——模型已见过 instruction，纯复读
    if s.startswith("任务") and len(s) < 200:
        return True
    # 兜底：整句被指令包含的短行，同样是复述
    if _norm(s) and _norm(s) in _norm(instruction) and len(s) < 80:
        return True
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def build_row(row: dict) -> dict:
    instr = row["instruction"]
    steps, answer = parse_output(row["output"])
    steps = [s for s in steps if not is_boilerplate(s, instr)]
    tagged = []
    for st in steps:
        tagged.append(f"[{tag_step(st)}] {st}")

    if tagged:
        reasoning = "\n".join(tagged)
    else:
        # 解析不出编号步骤：把原文当整体思考(低质量但不丢数据)
        reasoning = row["output"].strip()[:400]

    new_output = f"【思考】\n{reasoning}\n【/思考】\n【答案】\n{answer}\n【/答案】"

    return {
        "instruction": row["instruction"],
        "output": new_output,
        "_type": row.get("_type", ""),
        "_universe": row.get("_universe", ""),
        "_src": row.get("_src", ""),
        "_format": "v8_step_tagged_switch_boundary",
    }


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    out, label_counter = [], Counter()
    think_lines = 0
    think_re = re.compile(r"【思考】\n(.*?)\n【/思考】", re.DOTALL)
    label_re = re.compile(r"^\[(.+?)\]\s")

    for r in rows:
        nr = build_row(r)
        out.append(nr)
        m = think_re.search(nr["output"])
        if m:
            for line in m.group(1).splitlines():
                lm = label_re.match(line)
                if lm:
                    label_counter[lm.group(1)] += 1
                    think_lines += 1

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"V8 已生成 -> {OUT}")
    print(f"  总条数: {len(out)}  (源自 V7: {len(rows)})")
    print(f"  思考区有效步骤行: {think_lines}")
    print("  步骤标签分布(仅思考区):")
    for k, v in label_counter.most_common():
        pct = v / think_lines * 100 if think_lines else 0
        print(f"    {k:8s} {v:5d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
