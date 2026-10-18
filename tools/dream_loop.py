#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream_loop.py v2 — 睡眠·做梦自进化编排器（可跑态）

闭环：
  记忆检索(few-shot 种子示例) → (0.3B 底座 + 编程 LoRA) 生成代码
       → 沙箱执行(subprocess+硬超时) → 视频验收(vid_verify 闸门)
       → 入库(精英档案 + 三元组银行) + 失败料回灌 rejects

相对 v1 的关键修正（v1 生成出"代码块"死字，根因是 prompt 与训练协议不同构）：
  - 用代码模块专用 ChatML 协议(SYSTEM='你是北极星，主人的私有编程助手…')，
    与训练数据逐字同构（对齐 code_eval.py 的 build_prompt）。
  - 加载 0.3B 底座 + 编程 LoRA(PyFT 挂载，target=q/k/v/o，守保函数+E-7)。
  - 生成用 model.generate + eos=<|im_end|>，再按围栏块取码（同 code_eval）。
  - DREAM_BANK 全部是「贴近种子 sin 波示例」的 matplotlib 变体 —— 弱模型靠
    few-shot 检索照着改 sin→cos 等，才大概率产出可执行视频，真过 vid_verify。
  - latent 适配(T2-Z 测试时隐变量)作为可选嵌入残差 hook，默认关，避免干扰代码正确性。

宪法红线(保函数 + E-7)：主干冻绝，只动 LoRA(已训好) / 测试时 latent / 可删记忆层。
代码执行一律 subprocess + 硬超时，绝不拖垮主进程。

依赖：venv_ttt(torch 2.7.1+cpu / transformers 4.57.3 / peft 0.21.2) + ffmpeg
用法：
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/dream_loop.py --smoke
  PYTHONIOENCODING=utf-8 venv_ttt/Scripts/python.exe tools/dream_loop.py \
      --base D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018 \
      --lora D:/work/_local_ai/train_15b/out_lora_code_03b/checkpoint-150 \
      --n 4 --ttt-off
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FFMPEG = "D:/work/_local_ai/vidcut/node_modules/ffmpeg-static/ffmpeg.exe"
DREAM_DIR = os.path.join(ROOT, "data", "dreams")
IM_END = "<|im_end|>"
SYSTEM = ("你是北极星，主人的私有编程助手。回答给可运行的代码，配一两句关键说明。\n"
          "硬性要求：代码开头必须 import matplotlib.pyplot as plt 和 import numpy as np，"
          "结尾用 anim.save('out.mp4', writer='ffmpeg') 或 plt.savefig 生成文件；"
          "代码必须可直接 python 运行，不要写未定义的符号。")

# 做梦任务库：全部是「贴近种子 sin 波示例」的 matplotlib 变体。
# 统一要求生成 out.mp4 —— 使 vid_verify 成为天然闸门；few-shot 检索把种子
# 示例喂进去，弱模型照着改，才大概率过闸（这正是 RSI 自举：从已知好例出发造变体）。
DREAM_BANK = [
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个余弦波动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "把 sin 换成 cos 即可；from matplotlib.animation import FuncAnimation; anim.save('out.mp4', writer='ffmpeg')"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画正弦+余弦叠加的动画(0~4π, 100 帧)，保存为 out.mp4",
     "hint": "两条曲线 y1=sin(x), y2=cos(x) 画在同一张图；FuncAnimation + save ffmpeg"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个衰减正弦波动画(0~6π, 120 帧，振幅随时间衰减)，保存为 out.mp4",
     "hint": "y = sin(x) * exp(-0.05*x)；FuncAnimation + save ffmpeg"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个多曲线对比动画(正弦/方波/三角波各一条, 0~4π, 100 帧)，保存为 out.mp4",
     "hint": "numpy 造 square/triangle；ax.plot 三条；FuncAnimation + save ffmpeg"},
    {"domain": "UI可视化", "verify": "video",
     "task": "用 matplotlib 画一个李萨如(Lissajous)曲线动画, 100 帧，保存为 out.mp4",
     "hint": "x=sin(3t), y=sin(2t)；FuncAnimation + save ffmpeg"},
    {"domain": "游戏模拟", "verify": "video",
     "task": "用 matplotlib 画一个旋转螺旋线动画(100 帧)，保存为 out.mp4",
     "hint": "极坐标 r=t, theta=t；转直角坐标 plot；FuncAnimation + save ffmpeg"},
]


def ensure_dirs():
    os.makedirs(DREAM_DIR, exist_ok=True)


# ── 记忆检索(喂 few-shot) ───────────────────────────────────────────────
def retrieve_examples(task, k=2):
    """从三元组银行抽 top-k 相似历史成功代码，作为做梦的 few-shot 上下文。"""
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "triple_bank.py"), "query",
             "--task", task, "-k", str(k), "--show-code"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=ROOT)
        out = p.stdout or ""
        ex = []
        cur = None
        for line in out.splitlines():
            m = re.match(r"\s+code=(.+)", line)
            if m and os.path.isfile(m.group(1).strip()):
                cur = m.group(1).strip()
                txt = open(cur, encoding="utf-8", errors="replace").read()
                ex.append(txt[:1800])
        return ex
    except Exception as e:
        print("[retrieve] 跳过(err=%s)" % e)
        return []


# ── 代码生成：代码模块专用 ChatML 协议（与训练数据逐字同构） ──────────────
def build_prompt(task, hint, examples):
    few = ""
    if examples:
        few = "\n# 参考过往成功写法(仅供参考风格，不要照抄):\n" + "\n\n".join(
            "# --- example %d ---\n%s" % (i + 1, e) for i, e in enumerate(examples))
    usr = "%s\n提示: %s%s" % (task, hint, few)
    # 与 code_eval.build_prompt 完全一致的手工 ChatML
    return ("<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (SYSTEM, usr))


def extract_code(text):
    """取第一个带语言标注的围栏块（同 code_eval.extract_code）。"""
    blocks = re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", text, re.S)
    if not blocks:
        return None, None
    for lang, body in blocks:
        if lang.strip():
            return lang.strip().lower(), body
    return "", blocks[0][1]


# ── 模型：0.3B 底座 + 编程 LoRA（PEFT, q/k/v/o） ──────────────────────
def build_model(base, lora=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32,
                                                trust_remote_code=True)
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora, is_trainable=False)
        print("[model] 已挂编程 LoRA: %s" % lora)
    model.eval()
    return model, tok


def gen_code(model, tok, prompt, max_new=512, temp=0.0, top_p=0.92):
    """生成并截断到 <|im_end|>，返回模型续写文本。"""
    eos = tok.convert_tokens_to_ids(IM_END)
    ids = tok(prompt, return_tensors="pt").input_ids
    import torch
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=(temp if temp > 0 else 1.0),
            top_p=(top_p if temp > 0 else 1.0),
            eos_token_id=eos, pad_token_id=(tok.pad_token_id or eos))
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    if IM_END in gen:
        gen = gen.split(IM_END)[0]
    return gen


# ── 可选：测试时 latent 残差 hook（T2-Z 思路，默认关） ─────────────────
class LatentHook:
    def __init__(self, latent):
        self.latent = latent.detach()
        self.h = None

    def attach(self, model):
        emb = model.get_input_embeddings()
        lat = self.latent

        def fn(module, inp, out):
            # out: [B, T, H]
            return out + lat.view(1, 1, -1).to(out.dtype).to(out.device)

        self.h = emb.register_forward_hook(fn)

    def detach(self):
        if self.h:
            self.h.remove()
            self.h = None


def adapt_latent(model, tok, support, steps=8, lr=0.05):
    """训练一个零初始化 latent 向量（next-token CE），只优化 latent。"""
    import torch
    H = model.config.hidden_size
    latent = torch.zeros(H, requires_grad=True)
    opt = torch.optim.SGD([latent], lr=lr)
    ids = tok(support, return_tensors="pt").input_ids
    emb = model.get_input_embeddings()
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        e = emb(ids) + latent.view(1, 1, -1)
        out = model(inputs_embeds=e)
        loss = torch.nn.functional.cross_entropy(out.logits[0, :-1], ids[0, 1:])
        loss.backward()
        opt.step()
        last = loss.item()
    return latent, last


# ── 沙箱执行 ───────────────────────────────────────────────────────────
def run_code(code, run_id, timeout=90):
    d = os.path.join(DREAM_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    fpath = os.path.join(d, "code.py")
    open(fpath, "w", encoding="utf-8").write(code)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(FFMPEG) + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, fpath],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=d, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "rc": None, "out": "", "err": "timeout",
                "video": None, "sec": time.time() - t0, "dir": d}
    sec = time.time() - t0
    out, err, rc = p.stdout, p.stderr, p.returncode
    vids = [os.path.join(d, fn) for fn in os.listdir(d)
            if fn.lower().endswith((".mp4", ".avi", ".mov", ".webm"))]
    return {"ok": rc == 0, "reason": "exec", "rc": rc, "out": out, "err": err,
            "video": (vids[0] if vids else None), "sec": sec, "dir": d}


# ── 验收 + 入库 ────────────────────────────────────────────────────────
def accept(task, code, video, quality, run_id):
    codefile = os.path.join(DREAM_DIR, run_id, "code.py")
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, "triple_bank.py"), "add",
             "--task", task, "--code", codefile, "--video", video or "__none__",
             "--ffmpeg", FFMPEG],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        line = (p.stdout or p.stderr).strip().splitlines()
        print("   [accept]", line[0] if line else "")
        return p.returncode == 0
    except Exception as e:
        print("   [accept] 失败 err=%s" % e)
        return False


def record_trial(task, used_memory, passed):
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "triple_bank.py"), "trial",
             "--task", task, "--memory", "on" if used_memory else "off",
             "--passed", "yes" if passed else "no", "--note", "dream_loop"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    except Exception:
        pass


# ── 主循环 ─────────────────────────────────────────────────────────────
class Dreamer:
    def __init__(self, base, lora, use_latent, steps, lr, ffmpeg=None):
        global FFMPEG
        if ffmpeg:
            FFMPEG = ffmpeg
        self.model, self.tok = build_model(base, lora)
        self.use_latent = use_latent
        self.steps = steps
        self.lr = lr
        ensure_dirs()

    def one_dream(self, spec, use_memory=True):
        task = spec["task"]
        hint = spec.get("hint", "")
        run_id = "run_%s_%d" % (time.strftime("%Y%m%d_%H%M%S"),
                                int(time.time() * 1000) % 1000)
        print("\n=== 做梦 %s | 域=%s | 记忆=%s ===" % (run_id, spec["domain"], "on" if use_memory else "off"))
        print("   任务: %s" % task)

        examples = retrieve_examples(task, k=2) if use_memory else []
        if examples:
            print("   [memory] 检索到 %d 个 few-shot 示例" % len(examples))

        hook = None
        if self.use_latent:
            support = (task + " " + hint + " " + " ".join(examples[:1])) if examples else (task + " " + hint)
            latent, lloss = adapt_latent(self.model, self.tok, support, self.steps, self.lr)
            print("   [latent] 末步 loss=%.4f" % lloss)
            hook = LatentHook(latent)
            hook.attach(self.model)

        try:
            prompt = build_prompt(task, hint, examples)
            gen = gen_code(self.model, self.tok, prompt, max_new=512, temp=0.0)
            lang, code = extract_code(gen)
            print("   [gen] lang=%s 代码长度=%d" % (lang or "-", len(code or "")))
            if not code:
                print("   [gen] 未解析出代码 → REJECT")
                record_trial(task, use_memory, False)
                return False

            res = run_code(code, run_id, timeout=90)
            if not res["ok"]:
                print("   [exec] 失败 rc=%s err=%s" % (res["rc"], (res["err"] or "")[:200]))
                record_trial(task, use_memory, False)
                return False
            print("   [exec] 成功 用时%.1fs 产出视频=%s" % (res["sec"], bool(res["video"])))

            if res["video"]:
                ok = accept(task, code, res["video"], 1.0, run_id)
                record_trial(task, use_memory, ok)
                return ok
            else:
                print("   [verify] 无视频产出 → REJECT(三元组银行只收过 vid_verify 的视频)")
                record_trial(task, use_memory, False)
                return False
        finally:
            if hook:
                hook.detach()

    def run(self, n, use_memory=True):
        specs = [DREAM_BANK[i % len(DREAM_BANK)] for i in range(n)]
        stats = {"pass": 0, "fail": 0}
        for spec in specs:
            try:
                ok = self.one_dream(spec, use_memory=use_memory)
                stats["pass" if ok else "fail"] += 1
            except Exception as e:
                print("   [dream] 异常=%s" % e)
                stats["fail"] += 1
        print("\n=== 做梦轮次结束 ===")
        print("   通过(入库) %d / 失败(落 rejects) %d" % (stats["pass"], stats["fail"]))
        return stats


def smoke():
    """不加载模型的管线冒烟：验证 exec / 视频验收闸门 / 精英档案写入。"""
    ensure_dirs()
    rid = "smoke_%d" % int(time.time())
    code = "print('DONE')\n"
    res = run_code(code, rid, timeout=30)
    print("[smoke] exec ok=%s video=%s" % (res["ok"], res["video"]))
    vid_path = os.path.join(DREAM_DIR, "smoke_test.mp4")
    fcmd = [FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
            "-pix_fmt", "yuv420p", vid_path]
    pr = subprocess.run(fcmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if pr.returncode != 0:
        print("[smoke] ffmpeg 产片失败: %s" % (pr.stderr or "")[-300:])
    else:
        print("[smoke] ffmpeg 产片 OK")
        ok = accept("冒烟测试-视频闸门", code, vid_path, 1.0, rid)
        print("[smoke] 视频验收 accept=%s" % ok)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dream_elite", os.path.join(HERE, "dream_elite.py"))
        de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
        act, why = de.EliteArchive().add("冒烟测试-视频闸门", code, vid_path, 0.95)
        print("[smoke] elite=%s (%s)" % (act, why))
    except Exception as e:
        print("[smoke] elite err=%s" % e)
    print("[smoke] 完成（exec→视频验收→入库→精英档案 通路 OK）")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/work/_local_ai/train_15b/backups/0.3B/polaris-03b_20261018")
    ap.add_argument("--lora", default=None, help="编程模块 LoRA 目录(含 adapter_config.json)")
    ap.add_argument("--n", type=int, default=4, help="做梦轮数")
    ap.add_argument("--steps", type=int, default=8, help="latent 适配步数(仅 --use-latent)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--use-latent", action="store_true", help="开启测试时 latent 适配(默认关)")
    ap.add_argument("--no-memory", action="store_true", help="不做记忆检索(基线对照)")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--smoke", action="store_true", help="不加载模型，只测管线")
    a = ap.parse_args()

    if a.smoke:
        smoke()
        return
    d = Dreamer(a.base, a.lora, use_latent=a.use_latent, steps=a.steps,
                lr=a.lr, ffmpeg=a.ffmpeg)
    d.run(a.n, use_memory=not a.no_memory)


if __name__ == "__main__":
    main()
