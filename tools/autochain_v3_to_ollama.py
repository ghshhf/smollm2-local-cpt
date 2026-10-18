# -*- coding: utf-8 -*-
"""Auto-chain: wait for chat LoRA v3 training to finish -> convert adapter to GGUF -> prepare Ollama Modelfile -> create model if ollama ready."""
import os, time, subprocess, glob

ROOT = r"D:\work\_local_ai\train_15b"
PY = os.path.join(ROOT, "venv_ttt", "Scripts", "python.exe")
LLAMA = r"D:\work\_local_ai\llama.cpp"
BASE_GGUF = os.path.join(ROOT, "models", "gguf", "r6_final_base_f16.gguf")
ADAPTER_SRC = os.path.join(ROOT, "out_lora_chat_v3")
ADAPTER_GGUF = os.path.join(ROOT, "models", "gguf", "chat_v3_adapter_f16.gguf")
MODELFILE = os.path.join(ROOT, "models", "gguf", "Modelfile.northstar")

def log(m):
    print(time.strftime("[%H:%M:%S] "), m, flush=True)

def train_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                         capture_output=True, text=True).stdout
    # count python processes whose presence maps to training: fallback -> check log mtime growth
    return False

def log_active():
    """training considered active if log mtime changed in last 90s"""
    p = os.path.join(ROOT, "logs", "chat_lora_v3.log")
    try:
        return time.time() - os.path.getmtime(p) < 90
    except OSError:
        return False

# ---- 1. wait for training to end ----
for i in range(240):  # up to 4h
    if not log_active():
        # double-confirm with one more check 60s later
        log("log idle, confirming in 60s ...")
        time.sleep(60)
        if not log_active():
            log("training finished")
            break
    time.sleep(60)
else:
    log("TIMEOUT waiting training")
    raise SystemExit(1)

# verify final checkpoint exists
fin = sorted(glob.glob(os.path.join(ADAPTER_SRC, "checkpoint-*")), key=os.path.getmtime)
if fin:
    log("checkpoints: " + ", ".join(os.path.basename(c) for c in fin))

# ---- 2. convert LoRA -> GGUF ----
t0 = time.time()
r = subprocess.run([PY, os.path.join(LLAMA, "convert_lora_to_gguf.py"), ADAPTER_SRC,
                    "--base", os.path.join(ROOT, "backups", "0.2B", "r6_final_20261016"),
                    "--outfile", ADAPTER_GGUF, "--outtype", "f16"],
                   capture_output=True, text=True)
log("convert rc=%d %.0fs" % (r.returncode, time.time() - t0))
log((r.stdout or "")[-800:])
if r.returncode != 0:
    log("STDERR: " + (r.stderr or "")[-1500:])
    raise SystemExit(2)

# ---- 3. Modelfile ----
with open(MODELFILE, "w", encoding="utf-8") as f:
    f.write(f"""FROM {BASE_GGUF}
ADAPTER {ADAPTER_GGUF}
TEMPLATE \"\"\"{{{{- if .Messages }}{{{{- range $i, $_ := .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_predict 128
PARAMETER temperature 0
""")
log("Modelfile written: " + MODELFILE)

# ---- 4. ollama create if available ----
ollama_exe = r"C:\Users\admin\AppData\Local\Programs\Ollama\ollama.exe"
for i in range(120):  # wait up to 2h for ollama install
    if os.path.exists(ollama_exe):
        log("ollama found, creating model ...")
        r2 = subprocess.run([ollama_exe, "create", "northstar", "-f", MODELFILE],
                            capture_output=True, text=True)
        log("create rc=%d" % r2.returncode)
        log((r2.stdout or "")[-500:]); log((r2.stderr or "")[-500:])
        break
    time.sleep(60)
else:
    log("ollama not installed yet; Modelfile ready for manual step")
log("AUTOCHAIN_DONE")
