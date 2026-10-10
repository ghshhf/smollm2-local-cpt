"""
对比推理：同一份基座上，分别测「原模型」与「LoRA 微调后」，验证角色卡是否学会。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

BASE = "model/Qwen2.5-1.5B-Instruct"
LORA = "out_lora_role"
PROMPT = "下面是一项任务。\n请说明《昨书》中沈砚和裴决的关系，用一到两句话概括。\n输出："

print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

def gen(model):
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, torch_dtype=torch.float32)
    out = pipe(PROMPT, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
    return out[0]["generated_text"]

print("[load] 基座 ...")
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
print("\n========== 基座（无 LoRA） ==========")
print(gen(m))
print("=====================================")

print("[load] 合并 LoRA ...")
m2 = PeftModel.from_pretrained(m, LORA)
m2 = m2.merge_and_unload()
print("\n========== LoRA 微调后 ==========")
print(gen(m2))
print("==================================")
