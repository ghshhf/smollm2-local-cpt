"""
v2 LoRA 验证（修正版）：用**两个独立基座实例**，避免 merge_and_unload 的 in-place
污染导致「基座组」实际显示合并后结果。
同时打印诊断：adapter 是否加载、合并是否 in-place、合并后权重增量。
推理走 CPU fp32。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

BASE = "model/Qwen2.5-1.5B-Instruct"
LORA = "out_lora_v2/checkpoint-320"

QUESTIONS = [
    "请说明《昨书》中沈砚和裴决的关系，用一两句话概括。",
    "请说明《昨书》中沈砚的身份与职责。",
    "请说明《昨书》中裴决的身份与职责。",
    "请说明《守则宇宙》中设定「九块九」的核心规则。",
    "请说明《赛博血缘》中衰变链（纸→账→信用→算法→协议）的层级含义。",
]

print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# ---- 独立基座 A：干净基准 ----
print("[load] 独立基座 A (干净) ...", flush=True)
mA = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
base_pipe = pipeline("text-generation", model=mA, tokenizer=tok, torch_dtype=torch.float32)

# ---- 独立基座 B：加载并合并 LoRA ----
print("[load] 独立基座 B + 合并 v2 LoRA ...", flush=True)
mB = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
pm = PeftModel.from_pretrained(mB, LORA)
n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
w0 = mB.model.layers[0].self_attn.q_proj.weight.clone()
mB = pm.merge_and_unload()
w1 = mB.model.layers[0].self_attn.q_proj.weight
print(f"[diag] LoRA 可训练参数量 = {n_train}", flush=True)
print(f"[diag] 合并是否 in-place 改了基座权重 = {not torch.equal(w0, w1)}", flush=True)
print(f"[diag] 合并后相对基座的权重增量 max = {(w1 - w0).abs().max().item():.6f}", flush=True)
lora_pipe = pipeline("text-generation", model=mB, tokenizer=tok, torch_dtype=torch.float32)

for qi, q in enumerate(QUESTIONS, 1):
    b = base_pipe(q, max_new_tokens=160, do_sample=False, repetition_penalty=1.1)[0]["generated_text"]
    b = b[len(q):]  # 去掉 prompt 前缀
    l = lora_pipe(q, max_new_tokens=160, do_sample=False, repetition_penalty=1.1)[0]["generated_text"]
    l = l[len(q):]
    print(f"\n########## Q{qi}: {q} ##########", flush=True)
    print("--- 干净基座 ---", flush=True)
    print(b.strip(), flush=True)
    print("--- v2 LoRA ---", flush=True)
    print(l.strip(), flush=True)
    print("####################################", flush=True)

print("\n[done] verify_v2 finished", flush=True)
