import sys, re, time
sys.path.insert(0, "/mnt/workspace/node1/dream/tools")
import dream_loop as dl

base = "/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct"
print("[diag] loading", base)
model, tok = dl.build_model(base, None)
task = "用 matplotlib 画一个余弦波动画(0~4π, 100 帧)，保存为 out.mp4"
hint = "把 sin 换成 cos 即可"

# A) 干净（无 few-shot）—— 已知能出码
pa = dl.build_prompt(task, hint, [])
ga = dl.gen_code(model, tok, pa, max_new=512, temp=0.0)
print("[A] 无few-shot 围栏数=%d" % len(re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", ga, re.S)))

# C) 带 few-shot 检索——复现 probe 失败
print("\n[C] 检索 few-shot ...")
try:
    examples = dl.retrieve_examples(task, k=2)
except Exception as e:
    print("  retrieve_examples err:", e); examples = []
print("  examples count =", len(examples))
for i, e in enumerate(examples):
    print("  ex%d: len=%d head=%r" % (i, len(e), e[:80]))
pc = dl.build_prompt(task, hint, examples)
t = time.time()
gc = dl.gen_code(model, tok, pc, max_new=512, temp=0.0)
print("\n[C] RAW GEN 前 700:")
print(gc[:700])
print("\n[C] 耗时 %.1fs 围栏块数=%d" % (time.time()-t,
      len(re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", gc, re.S))))
