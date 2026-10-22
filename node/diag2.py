import sys, re
sys.path.insert(0, "/mnt/workspace/node1/dream/tools")
import dream_loop as dl
base = "/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct"
model, tok = dl.build_model(base, None)
print("SYSTEM repr head:", repr(dl.SYSTEM[:350]))
task = "用 matplotlib 画一个余弦波动画(0~4π, 100 帧)，保存为 out.mp4"
prompt = dl.build_prompt(task, "把 sin 换成 cos 即可", [])
print("PROMPT repr:", repr(prompt[:400]))
gen = dl.gen_code(model, tok, prompt, max_new=512, temp=0.0)
print("\nRAW GEN 前 700:")
print(gen[:700])
print("\n围栏数:", len(re.findall(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", gen, re.S)))
print("含```python?:", "```python" in gen, " 含```?:", "```" in gen)
