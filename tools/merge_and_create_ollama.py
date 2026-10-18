# -*- coding: utf-8 -*-
"""Merge chat v3 LoRA into base with peft's own merge_and_unload, then convert to GGUF (official converter)."""
import os, shutil, subprocess, time, sys

ROOT = r"D:\work\_local_ai\train_15b"
# 可传参: merge_and_create_ollama.py <adapter_dir_name> <merged_dir_name> <gguf_name> [ollama_model_name]
ADAPTER_NAME = sys.argv[1] if len(sys.argv) > 1 else "out_lora_chat_v3"
MERGED_NAME = sys.argv[2] if len(sys.argv) > 2 else "northstar_chat_v3_merged"
GGUF_NAME = sys.argv[3] if len(sys.argv) > 3 else "northstar_chat_v3_f16.gguf"
OLLAMA_MODEL = sys.argv[4] if len(sys.argv) > 4 else "northstar"

PY_TRAIN = os.path.join(ROOT, "venv", "Scripts", "python.exe")
PY_TTT = os.path.join(ROOT, "venv_ttt", "Scripts", "python.exe")
BASE = os.path.join(ROOT, "backups", "0.2B", "r6_final_20261016")
ADAPTER = os.path.join(ROOT, ADAPTER_NAME)
MERGED = os.path.join(ROOT, "models", MERGED_NAME)
GGUF_DIR = os.path.join(ROOT, "models", "gguf")

def log(m):
    print(time.strftime("[%H:%M:%S] "), m, flush=True)

# ---- 1. merge with peft ----
log("merging LoRA into base via peft.merge_and_unload ...")
code = (
    "import torch, os, shutil\n"
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
    "from peft import PeftModel\n"
    f"m = AutoModelForCausalLM.from_pretrained(r'{BASE}', dtype=torch.float32)\n"
    f"p = PeftModel.from_pretrained(m, r'{ADAPTER}')\n"
    "merged = p.merge_and_unload()\n"
    f"merged.save_pretrained(r'{MERGED}')\n"
    f"AutoTokenizer.from_pretrained(r'{BASE}').save_pretrained(r'{MERGED}')\n"
    "print('MERGE_OK')\n"
)
r = subprocess.run([PY_TRAIN, "-c", code], capture_output=True, text=True)
log("merge rc=%d %s" % (r.returncode, (r.stdout or "").strip()[-200:]))
if "MERGE_OK" not in (r.stdout or ""):
    log("STDERR: " + (r.stderr or "")[-1500:])
    raise SystemExit(1)

# ---- 2. convert merged -> GGUF ----
os.makedirs(GGUF_DIR, exist_ok=True)
gguf = os.path.join(GGUF_DIR, GGUF_NAME)
r2 = subprocess.run([PY_TTT, r"D:\work\_local_ai\llama.cpp\convert_hf_to_gguf.py", MERGED,
                     "--outfile", gguf], capture_output=True, text=True)
log("gguf rc=%d" % r2.returncode)
log((r2.stdout or r2.stderr or "")[-500:])
if not os.path.exists(gguf):
    raise SystemExit(2)
log("gguf size: %.1f MB" % (os.path.getsize(gguf) / 1048576))

# ---- 3. Modelfile without ADAPTER ----
mf = os.path.join(GGUF_DIR, "Modelfile." + OLLAMA_MODEL)
with open(mf, "w", encoding="utf-8") as f:
    f.write(f"""FROM {gguf}
TEMPLATE \"\"\"{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_predict 128
PARAMETER temperature 0
""")
log("Modelfile: " + mf)

# ---- 4. ollama create ----
OLLAMA = r"C:\Users\admin\AppData\Local\Programs\Ollama\ollama.exe"
r3 = subprocess.run([OLLAMA, "create", OLLAMA_MODEL, "-f", mf], capture_output=True, text=True)
log("create rc=%d" % r3.returncode)
log((r3.stdout or "")[-300:]); log((r3.stderr or "")[-300:])
log("MERGE_PIPELINE_DONE" if r3.returncode == 0 else "CREATE_FAILED")