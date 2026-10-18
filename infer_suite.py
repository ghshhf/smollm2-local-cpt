# -*- coding: utf-8 -*-
"""
全面能力测试套件: 对 checkpoint-latest 做 12 维能力对照。
覆盖: 科学知识(3) / 创造写作(2) / 数学(2) / 翻译(1) / 常识(1) / 指令跟随(1) / 自我认知(1) / 反事实创造(1)
用法: venv/Scripts/python.exe infer_suite.py
"""
import time, sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = r"D:\work\_local_ai\models\SmolLM2-200M"
CKPT = r"D:\work\_local_ai\train_15b\out_smollm2_200m_pt\checkpoint-latest\pytorch_model.bin"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd, strict=False)
model = model.to(torch.bfloat16).eval()
print(f"[info] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  (R4 step10 checkpoint)\n")

# (类别, 提示词, 生成token数)
TESTS = [
    ("科学·物理",   "水在标准大气压下加热到100摄氏度会发生什么？请解释其中的原理：", 100),
    ("科学·生物",   "植物为什么会向光生长？简单说明：", 100),
    ("科学·天文",   "月亮为什么会围绕地球转动？", 100),
    ("创造·诗歌",   "写一首关于夜空的四行短诗：", 80),
    ("创造·科幻",   "公元2150年，人类在火星的第一座城市迎来了第一位机器人市长。故事的开头：", 120),
    ("数学·算术",   "计算：17 + 26 = ?  请直接给出答案。", 40),
    ("数学·应用",   "一根绳子长12米，剪成每段3米，可以剪几段？", 80),
    ("翻译·中译英",  "把下面这句话翻译成英文：今天天气很好，我们去公园散步。", 60),
    ("常识",        "为什么冬天往玻璃杯里倒开水杯子容易裂？", 100),
    ("指令跟随",    "请列出三种水果的名字，每行一个。", 60),
    ("自我认知",    "你是谁？", 80),
    ("反事实创造",  "如果人类可以在水下呼吸，世界会发生什么变化？", 120),
]

PASS_HINT = {
    "科学·物理": "应提到沸腾/汽化/水蒸气",
    "科学·生物": "应提到向光性/生长素/光",
    "科学·天文": "应提到引力/质量",
    "数学·算术": "正确答案=43",
    "数学·应用": "正确答案=4段",
}

results = []
for tag, prompt, gen_n in TESTS:
    ids = tok.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=gen_n, do_sample=False, repetition_penalty=1.15)
    dt = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    results.append((tag, dt, prompt, gen))
    print("=" * 70)
    print(f"[{tag}]  {dt:.1f}s")
    print("-" * 70)
    print(text)
    print()

print("=" * 70)
print("SUMMARY")
for tag, dt, _, gen in results:
    hint = PASS_HINT.get(tag, "")
    ok = ""
    if hint:
        import re
        m = re.search(r"(\d+)", hint.split("=")[-1]) if "=" in hint else None
        if "43" in hint: ok = "✅含43" if "43" in gen else "❌未含43"
        elif "4段" in hint: ok = "✅含4" if re.search(r"4\s*段|剪.*4|答案.*4", gen) else "❌未含4"
        elif "沸腾" in hint: ok = "✅物理词" if any(w in gen for w in ["沸腾","汽化","蒸汽","水蒸气"]) else "❌无关"
        elif "向光" in hint or "生长素" in hint: ok = "✅生物词" if any(w in gen for w in ["向光","生长素","光"]) else "❌无关"
        elif "引力" in hint: ok = "✅引力词" if any(w in gen for w in ["引力","吸引","质量"]) else "❌无关"
    print(f"  {tag:8s} {dt:5.1f}s  {ok}")
