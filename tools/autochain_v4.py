# -*- coding: utf-8 -*-
"""Wait for chat LoRA v4 training to finish, then merge + convert + ollama create, then smoke-test."""
import os, time, subprocess

ROOT = r"D:\work\_local_ai\train_15b"
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "logs", "chat_lora_v4.log")
OLLAMA = r"C:\Users\admin\AppData\Local\Programs\Ollama\ollama.exe"

def log(m):
    print(time.strftime("[%H:%M:%S] "), m, flush=True)

def idle():
    try:
        return time.time() - os.path.getmtime(LOG) > 90
    except OSError:
        return False

for _ in range(300):
    if idle():
        time.sleep(60)
        if idle():
            break
    time.sleep(60)
else:
    log("TIMEOUT"); raise SystemExit(1)

tail = open(LOG, encoding="utf-8", errors="ignore").read()[-4000:]
log("training tail: " + tail.replace("\n", " | ")[-200:])
if "EXITCODE=0" not in tail:
    log("training did not finish cleanly, abort merge")
    raise SystemExit(2)

log("merging v4 adapter ...")
r = subprocess.run([PY, os.path.join(ROOT, "tools", "merge_and_create_ollama.py"),
                    "out_lora_chat_v4", "northstar_chat_v4_merged",
                    "northstar_chat_v4_f16.gguf", "northstar"],
                   capture_output=True, text=True)
log("merge/create rc=%d" % r.returncode)
for line in (r.stdout or "").splitlines():
    if any(k in line for k in ["create rc", "gguf size", "MERGED", "DONE", "FAIL", "merge rc"]):
        log(line.strip())
log((r.stderr or "")[-400:])

log("smoke test via ollama api:")
for q in ["\u4f60\u662f\u8c01", "\u4f60\u53eb\u4ec0\u4e48\u540d\u5b57", "\u4f60\u662f\u4e2d\u6587\u52a9\u624b\u5417"]:
    rq = subprocess.run(["curl", "-s", "--max-time", "120", "http://127.0.0.1:11434/api/chat",
                         "-d", '{"model":"northstar","messages":[{"role":"user","content":"%s"}],"stream":false,"options":{"temperature":0,"num_predict":80}}' % q],
                        capture_output=True, text=True)
    try:
        import json
        d = json.loads(rq.stdout)
        log("Q %s => %s" % (q, d.get("message", {}).get("content", "").replace("\n", " ")[:120]))
    except Exception as e:
        log("Q %s => parse fail %s" % (q, e))
log("V4_CHAIN_DONE")