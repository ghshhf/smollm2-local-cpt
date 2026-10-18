"""把 instruction/output 语料转成 ChatML（对话模块同构格式），供 LoRA SFT 使用。

宪法要求 SFT 注入 system 位，所以每条前面带一条固定 system。
格式与中文对话模块完全一致，推理代码可复用。

用法:
  python tools/build_code_chatml.py --in data/corpus_code_v2.jsonl --out data/code_sft_v1.jsonl
"""
import argparse
import json

SYS = "你是北极星，主人的私有编程助手。回答给可运行的代码，配一两句关键说明。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--no_system", action="store_true", help="不加 system 位（与旧对话模块保持完全同构）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    out = []
    for r in rows:
        q = r["instruction"].strip()
        ans = r["output"].strip()
        if not q or not ans:
            continue
        sys_part = "" if a.no_system else f"<|im_start|>system\n{SYS}<|im_end|>\n"
        text = f"{sys_part}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{ans}<|im_end|>"
        out.append({"text": text})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"输入 {len(rows)} 条 → 输出 {len(out)} 条 ChatML  → {a.out}")
    print(f"system 位: {'未注入' if a.no_system else '已注入'}")
    print("样例:", out[0]["text"][:140].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
