"""
知识卡 · 读卡器（按卡切换模式）
================================
加载 1.5B 基座 + 指定 LoRA 卡，进入对话循环。
训练数据格式（与 train_lora.py build_text 一致）：
  "下面是一项任务。\n{instruction}\n输出：{output}"

用法（新模式：每张卡一个领域，用时才合）：
  python mount.py --adapter out_lora_v3                      # 单卡：小说设定
  python mount.py --adapters out_lora_v3,out_lora_prose      # 合两张：设定+文风
  python mount.py --adapters out_lora_v3,out_lora_code,out_lora_reason --weights 1,0.8,0.8
  python mount.py                                            # 仅基座（无卡）
  python mount.py --list                                     # 列出当前存在的卡
  python mount.py --adapter out_lora_code --think 5 --exec   # 思考模式 + 代码执行验证

卡与领域（见 data/card_*.jsonl 与 train_chain.py JOBS）：
  out_lora_v3     小说设定百科   role_data_v3 (624)
  out_lora_prose  行文风格       card_prose  (470)
  out_lora_code   编程           card_code   (1463)
  out_lora_reason 推理           card_reason (1471)

说明：
- fp32 1.5B 在 CPU 上加载约 30-60s，生成偏慢，属正常。
- --think K (K>1)：2026 前沿结论——**把推理算力从预训练挪到推理期(测试时计算)**，
  小模型多采样+自验证可逼近甚至超过大得多的基础模型(Snell/DeepMind: 1B+TTC≈405B)。
  本机无端点、不训 RL，故用「多采样 + 自一致性投票 + (可选)代码执行验证」实现。
- 默认(不加 --think)行为完全不变。
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "Qwen2.5-1.5B-Instruct")


CARD_DESC = {
    "out_lora_v3": "小说设定百科(概念/角色/关系/大纲)",
    "out_lora_prose": "行文风格与叙事延续",
    "out_lora_code": "编程(函数/知识蒸馏/范式)",
    "out_lora_reason": "推理(过程标签+思考/答案边界)",
}


def discover_cards():
    """扫描 out_lora_*，只要有 adapter_model.safetensors（含 checkpoint-*）就算可用卡。"""
    import glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cards = {}
    for name in sorted(glob.glob(os.path.join(root, "out_lora_*"))):
        if not os.path.isdir(name):
            continue
        done = os.path.isfile(os.path.join(name, "adapter_model.safetensors"))
        ckpts = sorted(glob.glob(os.path.join(name, "checkpoint-*")), reverse=True)
        if done or ckpts:
            path = name if done else ckpts[0]
            cards[os.path.basename(name)] = path + ("（半成品 checkpoint）" if not done else "")
    return cards


def build_prompt(instruction, inp=""):
    if inp:
        return f"下面是一项任务。\n{instruction}\n输入：{inp}\n输出："
    return f"下面是一项任务。\n{instruction}\n输出："


def load(base, adapters, weights=None):
    """加载基座并把 1~N 张卡按权重并入（用时才合）。合成后是普通 HF 模型，与 peft 解耦。"""
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, dtype=torch.float32)
    if adapters:
        apply_lora_deltas(model, adapters, weights, verbose=True)
        names = "+".join(os.path.basename(a.rstrip("\\/")) for a in adapters)
        tag = f"基座+{names}"
    else:
        tag = "基座"
    model.eval()
    return model, tok, tag


def extract_answer(text: str) -> str:
    """从一条完整生成里抽取「最终答案签名」：优先最后一个代码块，否则末尾一段。"""
    blocks = re.findall(r"```(?:[a-zA-Z]+)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return lines[-1].strip() if lines else text.strip()


def exec_code(code: str, timeout: int = 5):
    """本地沙箱跑一段代码，返回 (ok, 错误信息)。仅用于思考模式的代码卡验证。"""
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


@torch.no_grad()
def think(model, tok, prompt: str, k: int, max_new_tokens: int, exec_mode: bool):
    """思考模式(测试时计算)：采样 K 条推理路径 → 验证/投票 → 返回胜出答案与统计。"""
    cands = []
    for _ in range(k):
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.92, pad_token_id=tok.pad_token_id,
        )
        cands.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())

    scored = []
    for c in cands:
        ans = extract_answer(c)
        ok, note = True, ""
        if exec_mode and ("def " in ans or "import " in ans):
            ok, note = exec_code(ans)
        scored.append((c, ans, ok, note))

    runnable = [s for s in scored if s[2]]
    pool = runnable if runnable else scored
    sig = Counter(s[1] for s in pool)
    best_sig = sig.most_common(1)[0][0]
    best = next(s for s in pool if s[1] == best_sig)
    return best[0], len(cands), len(runnable), best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default="", help="单张 LoRA 卡目录，如 out_lora_code；留空=仅基座")
    ap.add_argument("--adapters", default="", help="多张卡目录，逗号分隔；与 --adapter 二选一")
    ap.add_argument("--weights", default="", help="多卡时的权重，逗号分隔；默认全 1.0")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--list", action="store_true", help="仅列出可用卡后退出")
    ap.add_argument("--think", type=int, default=1,
                   help="思考模式：每条问题采样 K 条推理路径再投票(测试时计算)；1=关")
    ap.add_argument("--exec", action="store_true",
                   help="思考模式下对提取出的代码块做本地执行验证(沙箱 timeout=5s)，优先选能跑通的")
    args = ap.parse_args()

    if args.list:
        cards = discover_cards()
        print("可用知识卡：")
        for k, v in cards.items():
            desc = CARD_DESC.get(k, "")
            print(f"  {k:18s} -> {v}   {desc}")
        if not cards:
            print("  (暂无；训练完成并产出 adapter_model.safetensors 后才会出现)")
        return

    adapters, weights = [], None
    if args.adapters:
        adapters = [a.strip() for a in args.adapters.replace("，", ",").split(",") if a.strip()]
        weights = ([float(x) for x in args.weights.replace("，", ",").split(",") if x.strip()]
                   if args.weights else None)
    elif args.adapter:
        adapters = [args.adapter]
    for a in adapters:
        if not os.path.isdir(a):
            ap.error(f"卡目录不存在: {a} （用 --list 查看可用卡）")

    print("[mount] 加载模型中（CPU fp32，约 30-60s）...")
    model, tok, tag = load(args.base, adapters, weights)
    print(f"[mount] 已加载：{tag}。输入指令，空行或 Ctrl-C 退出。")

    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[mount] 退出。")
            break
        if not q:
            break
        prompt = build_prompt(q)
        if args.think > 1:
            gen, n, runnable, err = think(model, tok, prompt, args.think,
                                          args.max_new_tokens, args.exec)
            tag = f"[think k={n}, 可运行={runnable}]"
            print(f"模型> {tag}")
            print(gen.strip())
            if args.exec and err:
                print(f"  (执行验证备注: {err})")
        else:
            ids = tok(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print("模型>", gen.strip())


if __name__ == "__main__":
    main()
