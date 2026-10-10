"""临时基准：CPU 上 fp32 vs bf16 的生成速度，决定评测用哪个精度。（一次性工具）"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "model/Qwen2.5-1.5B-Instruct"
PROMPT = "下面是一项任务。\n写一段关于远方城市的描写。\n输出："

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
ids = tok(PROMPT, return_tensors="pt")

for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    torch.set_num_threads(8)
    t_load = time.time()
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=dt, low_cpu_mem_usage=True,
                                             trust_remote_code=True)
    m.eval()
    load_s = time.time() - t_load
    with torch.no_grad():
        m.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        t0 = time.time()
        out = m.generate(**ids, max_new_tokens=48, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        el = time.time() - t0
    n = out.shape[1] - ids["input_ids"].shape[1]
    print(f"[{name}] 加载 {load_s:.1f}s | {n} tokens / {el:.1f}s = {n/el:.2f} tok/s")
    del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
