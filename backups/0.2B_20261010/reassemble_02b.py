#!/usr/bin/env python3
# reassemble_02b.py — 把 backups/0.2B_20261010 的 9 个分片拼回 model.safetensors
# 用法：python reassemble_02b.py [输出目录]
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "restored")
PARTS = 9
FINAL_SIZE = 856595368          # 0.2B (SmolLM2-200M config) fp32... 实际 bf16 权重字节数
FINAL_SHA256_PREFIX = "0ee38bf6a52383ff"  # 2026-10-10 本地副本实测值

os.makedirs(OUT_DIR, exist_ok=True)
dst = os.path.join(OUT_DIR, "model.safetensors")
h = hashlib.sha256()
with open(dst, "wb") as out:
    for i in range(PARTS):
        p = os.path.join(HERE, "model.safetensors.part%02d" % i)
        with open(p, "rb") as f:
            while True:
                buf = f.read(8 << 20)
                if not buf:
                    break
                out.write(buf)
                h.update(buf)
size = os.path.getsize(dst)
print("restored -> %s (%d bytes)" % (dst, size))
assert size == FINAL_SIZE, "size mismatch: %d != %d" % (size, FINAL_SIZE)
if FINAL_SHA256_PREFIX:
    got = h.hexdigest()[:len(FINAL_SHA256_PREFIX)]
    assert got == FINAL_SHA256_PREFIX, "sha256 mismatch: %s != %s" % (got, FINAL_SHA256_PREFIX)
    print("sha256 OK:", h.hexdigest())
else:
    print("sha256:", h.hexdigest())
# config 四件套一并带过去
for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
    src = os.path.join(HERE, name)
    if os.path.exists(src):
        with open(src, "rb") as f, open(os.path.join(OUT_DIR, name), "wb") as g:
            g.write(f.read())
print("done. load with: AutoModelForCausalLM.from_pretrained('%s')" % OUT_DIR)
